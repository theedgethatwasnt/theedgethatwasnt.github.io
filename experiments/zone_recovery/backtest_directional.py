"""
Two directional strategies from H4 TopsBots S/R — no zone recovery grid.

STRATEGY A — Bounce:
  Small S/R penetration → expect bounce back to opposite boundary.
  LONG at support, SHORT at resistance.
  TP = act_h + tgt_frac*ZW (LONG) / act_l - tgt_frac*ZW (SHORT).
  Exit: TP hit OR max_hold bars (no hard SL).
  Filter: entry depth <= max_depth_frac * ZW (only shallow crossings).

STRATEGY B — Breakout:
  SIGNIFICANT S/R penetration → expect continuation past the broken level.
  SHORT at support break, LONG at resistance break.
  TP = act_l - tp_frac*ZW below broken support (or act_h + tp_frac*ZW above broken resistance).
  Entry filter: entry depth >= min_depth_frac * ZW (only deep crossings qualify).
  Exit: TP hit OR max_hold bars.

Pipeline:
  1. IS parameter sweep (all combos, 12 pairs)
  2. Walk-forward (3 IS chunks, winner = best min-chunk SQN)
  3. Monte Carlo permutation (2000 shuffles)
  4. OOS final validation (last 30%)
"""

import os, sys, math, time
import numpy as np
import pandas as pd
from numba import njit
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))
import backtest_fixed_zw_h4 as bt

DATA_DIR    = os.path.expanduser('~/projects/fx-core/data/m5_ohlc')
PAIRS       = ['AUD_JPY','AUD_USD','CAD_JPY','CHF_JPY','EUR_GBP','EUR_JPY',
               'EUR_USD','GBP_JPY','GBP_USD','NZD_JPY','NZD_USD','USD_JPY']
PIP_USD_MAP = {'AUD_JPY':0.000067,'AUD_USD':0.000100,'CAD_JPY':0.000069,
               'CHF_JPY':0.000107,'EUR_GBP':0.000126,'EUR_JPY':0.000064,
               'EUR_USD':0.000100,'GBP_JPY':0.000091,'GBP_USD':0.000100,
               'NZD_JPY':0.000061,'NZD_USD':0.000100,'USD_JPY':0.000064}
PIP_MAP     = {p: 0.01 if 'JPY' in p else 0.0001 for p in PAIRS}
SPREAD      = 1.4
UNITS       = 1_000
IS_FRAC     = 0.70
N_WF_CHUNKS = 3
N_MC        = 2000
MAX_ZW_PIPS = 150
MIN_ZW_PIPS_GLOBAL = 20   # skip degenerate zones

# ── Parameter grids ───────────────────────────────────────────────────────────

# Strategy A (Bounce): TP + time exit
A_TGT_FRACS    = [0.0, 0.10, 0.25]         # TP margin beyond opposite boundary
A_MAX_DEPTHS   = [0.10, 0.25, 0.50, 9.99]  # max entry depth as frac of ZW (9.99=unlimited)
A_MAX_HOLDS    = [288, 576, 1440]           # 1d, 2d, 5d in M5 bars

# Strategy B (Breakout): continuation past broken level
B_MIN_DEPTHS   = [0.10, 0.25, 0.50]        # min entry depth frac of ZW to qualify
B_TP_FRACS     = [0.50, 1.00, 1.50, 2.00]  # TP = act_l - B_TP_FRAC*ZW below broken support
B_MAX_HOLDS    = [288, 576, 1440]


# ══════════════════════════════════════════════════════════════════════════════
# JIT: Strategy A — Bounce (TP + time exit, no SL)
# ══════════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _sim_bounce(close_a, open_a, high_a, low_a, act_h, act_l,
                tgt_frac, max_depth_frac, max_hold, min_zw, max_zw, pip, spread):
    """
    LONG at support / SHORT at resistance.
    Entry filter: depth of penetration <= max_depth_frac * ZW.
    TP: opposite boundary + tgt_frac*ZW. Exit on TP or time limit.
    """
    n = len(close_a)
    pnl_a = np.empty(n, dtype=np.float64)
    ext_a = np.empty(n, dtype=np.int64)   # 1=tp, 0=time
    nc = 0; i = 0
    while i < n:
        uh = act_h[i]; ul = act_l[i]
        if uh != uh or ul != ul or uh <= ul: i += 1; continue
        zw = (uh - ul) / pip
        if zw < min_zw or zw > max_zw: i += 1; continue
        entry = close_a[i]
        if entry <= ul:
            dr = 1.0
            depth = (ul - entry) / pip          # pips below support
        elif entry >= uh:
            dr = -1.0
            depth = (entry - uh) / pip          # pips above resistance
        else:
            i += 1; continue
        # Depth filter: skip if too deep (deep entries are breakout candidates)
        if depth > max_depth_frac * zw:
            i += 1; continue
        tb = tgt_frac * zw * pip
        tp_p = uh + tb if dr == 1.0 else ul - tb

        entry_bar = i; i += 1; closed = False; ep = entry; et = 0
        while i < n and not closed:
            if (i - entry_bar) >= max_hold:
                ep = close_a[i - 1]; closed = True; break
            hi = high_a[i]; lo = low_a[i]; cl = close_a[i]
            bull = cl >= open_a[i]
            for p2 in range(2):
                if closed: break
                chk_h = (p2 == 0 and bull) or (p2 == 1 and not bull)
                if chk_h:
                    if dr == 1.0 and hi >= tp_p:
                        ep = tp_p; et = 1; closed = True
                else:
                    if dr == -1.0 and lo <= tp_p:
                        ep = tp_p; et = 1; closed = True
            if not closed: i += 1

        pnl = dr * (ep - entry) / pip - spread
        if nc < n: pnl_a[nc] = pnl; ext_a[nc] = et
        nc += 1
        if not closed: break
    return pnl_a[:nc], ext_a[:nc]


# ══════════════════════════════════════════════════════════════════════════════
# JIT: Strategy B — Breakout (continuation)
# ══════════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _sim_breakout(close_a, open_a, high_a, low_a, act_h, act_l,
                  min_depth_frac, tp_frac, max_hold, min_zw, max_zw, pip, spread):
    """
    SHORT when price significantly penetrates support (continuation down).
    LONG when price significantly penetrates resistance (continuation up).
    Entry filter: depth >= min_depth_frac * ZW.
    TP: broken_level - tp_frac*ZW (SHORT) or broken_level + tp_frac*ZW (LONG).
    The TP projects tp_frac zone-widths beyond the broken boundary.
    """
    n = len(close_a)
    pnl_a = np.empty(n, dtype=np.float64)
    ext_a = np.empty(n, dtype=np.int64)
    nc = 0; i = 0
    while i < n:
        uh = act_h[i]; ul = act_l[i]
        if uh != uh or ul != ul or uh <= ul: i += 1; continue
        zw = (uh - ul) / pip
        if zw < min_zw or zw > max_zw: i += 1; continue
        entry = close_a[i]
        if entry < ul:
            depth = (ul - entry) / pip
            if depth < min_depth_frac * zw: i += 1; continue
            # Significant support break → SHORT (continuation down)
            dr = -1.0
            tp_p = ul - tp_frac * zw * pip      # project below broken support
        elif entry > uh:
            depth = (entry - uh) / pip
            if depth < min_depth_frac * zw: i += 1; continue
            # Significant resistance break → LONG (continuation up)
            dr = 1.0
            tp_p = uh + tp_frac * zw * pip      # project above broken resistance
        else:
            i += 1; continue

        entry_bar = i; i += 1; closed = False; ep = entry; et = 0
        while i < n and not closed:
            if (i - entry_bar) >= max_hold:
                ep = close_a[i - 1]; closed = True; break
            hi = high_a[i]; lo = low_a[i]; cl = close_a[i]
            bull = cl >= open_a[i]
            for p2 in range(2):
                if closed: break
                chk_h = (p2 == 0 and bull) or (p2 == 1 and not bull)
                if chk_h:
                    if dr == 1.0 and hi >= tp_p:
                        ep = tp_p; et = 1; closed = True
                else:
                    if dr == -1.0 and lo <= tp_p:
                        ep = tp_p; et = 1; closed = True
            if not closed: i += 1

        pnl = dr * (ep - entry) / pip - spread
        if nc < n: pnl_a[nc] = pnl; ext_a[nc] = et
        nc += 1
        if not closed: break
    return pnl_a[:nc], ext_a[:nc]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def sqn(a):
    if len(a) < 20: return 0.0
    s = a.std()
    return 0.0 if s == 0 else float(a.mean() / s * math.sqrt(len(a)))

def sharpe(pnl_arr, bars_per_day=288):
    """Daily Sharpe — annualized."""
    if len(pnl_arr) < 20: return 0.0
    days = max(1, len(pnl_arr) // 10)
    chunk = max(1, len(pnl_arr) // days)
    daily = np.array([pnl_arr[k*chunk:(k+1)*chunk].sum() for k in range(days)])
    s = daily.std()
    return 0.0 if s == 0 else float(daily.mean() / s * math.sqrt(252))

def score(pair_pnls):
    all_p = np.concatenate([p for p in pair_pnls if len(p) > 0]) if pair_pnls else np.zeros(1)
    n_pos = sum(1 for p in pair_pnls if p.sum() > 0)
    return sqn(all_p), n_pos, float(all_p.sum()), len(all_p)


# ── JIT warmup ────────────────────────────────────────────────────────────────
_d = np.ones(400, dtype=np.float64); _h = _d*1.01; _l = _d*0.99
_ah = _d*1.005; _al = _d*0.995
_sim_bounce(_d,_d,_h,_l,_ah,_al, 0.25,0.5,288,20.0,150.0,0.0001,1.4)
_sim_breakout(_d,_d,_h,_l,_ah,_al, 0.25,1.0,288,20.0,150.0,0.0001,1.4)
print("JIT compiled\n", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Load all data once
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...", flush=True)
data = {}
for pair in PAIRS:
    pip = PIP_MAP[pair]
    df = pd.read_parquet(f'{DATA_DIR}/{pair}_M5.parquet').sort_index()
    df.columns = [c.lower() for c in df.columns]
    oa=df['open'].values.astype(np.float64); ha=df['high'].values.astype(np.float64)
    la=df['low'].values.astype(np.float64);  ca=df['close'].values.astype(np.float64)
    h4h, h4l = bt.build_h4_sr(ha, la)
    n = len(ca); n_is = int(n * IS_FRAC)
    data[pair] = dict(oa=oa,ha=ha,la=la,ca=ca,h4h=h4h,h4l=h4l,
                      n=n,n_is=n_is,pip=pip,pu=PIP_USD_MAP[pair])
n_is_avg = int(np.mean([data[p]['n_is'] for p in PAIRS]))
print(f"  {len(PAIRS)} pairs loaded  IS={IS_FRAC*100:.0f}%  OOS={100-IS_FRAC*100:.0f}%\n", flush=True)


def run_sim(strategy, params, pair, sl):
    """Run either strategy on a slice. Returns pnl array."""
    d = data[pair]
    s = sl.start or 0; e = min(sl.stop, d['n_is'] if sl.stop <= d['n_is'] else d['n'])
    if e <= s: return np.zeros(0)
    args = (d['ca'][s:e], d['oa'][s:e], d['ha'][s:e], d['la'][s:e],
            d['h4h'][s:e], d['h4l'][s:e])
    if strategy == 'A':
        tf, md, mh = params
        pa, _ = _sim_bounce(*args, tf, md, mh, MIN_ZW_PIPS_GLOBAL, MAX_ZW_PIPS, d['pip'], SPREAD)
    else:
        md, tp, mh = params
        pa, _ = _sim_breakout(*args, md, tp, mh, MIN_ZW_PIPS_GLOBAL, MAX_ZW_PIPS, d['pip'], SPREAD)
    return pa


def sweep_strategy(strategy, grid, label):
    """IS sweep for one strategy. Returns sorted result list."""
    print(f"\n{'═'*72}")
    print(f"  STRATEGY {strategy} — {label}  ({len(grid)} combos × {len(PAIRS)} pairs)")
    print(f"{'═'*72}")
    t0 = time.time(); results = []
    for params in grid:
        pair_pnls = [run_sim(strategy, params, p, slice(0, data[p]['n_is'])) for p in PAIRS]
        agg_sqn, n_pos, tot_pips, tot_tr = score(pair_pnls)
        results.append(dict(params=params, sqn=agg_sqn, n_pos=n_pos,
                            pips=tot_pips, trades=tot_tr, pair_pnls=pair_pnls))
    results.sort(key=lambda x: x['sqn'], reverse=True)
    print(f"  Sweep done {time.time()-t0:.1f}s\n")
    if strategy == 'A':
        hdr = f"  {'tgt':>5} {'maxD':>5} {'hold':>5}"
    else:
        hdr = f"  {'minD':>5} {'tp':>5} {'hold':>5}"
    print(hdr + f" | {'SQN':>6} {'pos':>5} {'pips':>10} {'trades':>7}")
    print("  " + "─"*55)
    for r in results[:15]:
        p = r['params']
        if strategy == 'A':
            md_s = f"{p[1]:.2f}" if p[1] < 9 else "all"
            ph = f"{p[2]//288:.0f}d" if p[2] >= 288 else f"{p[2]//12}h"
            ps = f"  {p[0]:>5.2f} {md_s:>5} {ph:>5}"
        else:
            ph = f"{p[2]//288:.0f}d" if p[2] >= 288 else f"{p[2]//12}h"
            ps = f"  {p[0]:>5.2f} {p[1]:>5.2f} {ph:>5}"
        sym = '🟢' if r['sqn'] > 1 else ('🟡' if r['sqn'] > 0 else '🔴')
        print(f"{ps} | {r['sqn']:>6.2f}{sym} {r['n_pos']:>4}/12 {r['pips']:>+10,.0f} {r['trades']:>7,}")
    return results


def wf_validate(strategy, results, label):
    """Walk-forward on top 10 IS candidates."""
    print(f"\n{'═'*72}")
    print(f"  STRATEGY {strategy} WF — {label}")
    print(f"{'═'*72}")
    chunk_sz = n_is_avg // N_WF_CHUNKS
    wf_res = []
    for r in results[:10]:
        params = r['params']; cscores = []; cposs = []
        for ch in range(N_WF_CHUNKS):
            s = ch * chunk_sz
            e = (ch+1)*chunk_sz if ch < N_WF_CHUNKS-1 else n_is_avg
            pp = [run_sim(strategy, params, p, slice(s, e)) for p in PAIRS]
            agg_sqn, n_pos, _, _ = score(pp)
            cscores.append(agg_sqn); cposs.append(n_pos)
        wf_res.append(dict(params=params, is_sqn=r['sqn'], wf_min=min(cscores),
                           cs=cscores, cp=cposs))
    wf_res.sort(key=lambda x: x['wf_min'], reverse=True)
    if strategy == 'A':
        hdr = f"  {'tgt':>5} {'maxD':>5} {'hold':>5}"
    else:
        hdr = f"  {'minD':>5} {'tp':>5} {'hold':>5}"
    print(f"\n{hdr} | {'IS':>6} | {'C1':>6} {'C2':>6} {'C3':>6} | {'WFmin':>6}")
    print("  " + "─"*62)
    for r in wf_res:
        p = r['params']
        if strategy == 'A':
            md_s = f"{p[1]:.2f}" if p[1] < 9 else "all"
            ph = f"{p[2]//288:.0f}d" if p[2] >= 288 else f"{p[2]//12}h"
            ps = f"  {p[0]:>5.2f} {md_s:>5} {ph:>5}"
        else:
            ph = f"{p[2]//288:.0f}d" if p[2] >= 288 else f"{p[2]//12}h"
            ps = f"  {p[0]:>5.2f} {p[1]:>5.2f} {ph:>5}"
        cs_str = "  ".join(f"{s:>5.2f}" for s in r['cs'])
        pos_str = "/".join(str(p2) for p2 in r['cp'])
        sym = '🟢' if r['wf_min'] > 1 else ('🟡' if r['wf_min'] > 0 else '🔴')
        print(f"{ps} | {r['is_sqn']:>5.2f} | {cs_str} | {r['wf_min']:>5.2f}{sym}  [{pos_str}]")
    winner = wf_res[0]
    p = winner['params']
    if strategy == 'A':
        md_s = f"{p[1]:.2f}" if p[1] < 9 else "all"
        ph = f"{p[2]//288:.0f}d" if p[2] >= 288 else f"{p[2]//12}h"
        desc = f"tgt={p[0]} maxDepth={md_s} hold={ph}"
    else:
        ph = f"{p[2]//288:.0f}d" if p[2] >= 288 else f"{p[2]//12}h"
        desc = f"minDepth={p[0]} tp={p[1]}×ZW hold={ph}"
    print(f"\n  ✅ {strategy} WINNER: {desc}  wf_min={winner['wf_min']:.2f}")
    return winner


def mc_permutation(strategy, winner, label):
    """2000-shuffle permutation test on IS data."""
    print(f"\n{'═'*72}")
    print(f"  STRATEGY {strategy} MC — {label}  ({N_MC} shuffles)")
    print(f"{'═'*72}", flush=True)
    params = winner['params']
    # Real IS result
    real_usd = sum(
        run_sim(strategy, params, p, slice(0, data[p]['n_is'])).sum() * data[p]['pu'] * UNITS
        for p in PAIRS
    )

    @njit(cache=True)
    def _run_shuffled_bounce(ca,oa,ha,la,ah,al,sf_dirs,tf2,md2,mh2,mz2,mxz2,pip2,spr2):
        """Bounce sim with shuffled directions (swap LONG↔SHORT on each entry)."""
        n=len(ca); tp=0.0; di=0; i=0
        while i<n and di<len(sf_dirs):
            uh=ah[i]; ul=al[i]
            if uh!=uh or ul!=ul or uh<=ul: i+=1; continue
            zw=(uh-ul)/pip2
            if zw<mz2 or zw>mxz2: i+=1; continue
            e=ca[i]
            if e<=ul: depth=(ul-e)/pip2
            elif e>=uh: depth=(e-uh)/pip2
            else: i+=1; continue
            if depth>md2*zw: i+=1; continue
            dr=sf_dirs[di]; di+=1
            tb2=tf2*zw*pip2
            tp_p=uh+tb2 if dr==1.0 else ul-tb2
            eb=i; i+=1; closed=False; ep=e; et=0
            while i<n and not closed:
                if (i-eb)>=mh2: ep=ca[i-1]; closed=True; break
                hi=ha[i]; lo=la[i]; cl=ca[i]; bull=cl>=oa[i]
                for p2 in range(2):
                    if closed: break
                    chk_h=(p2==0 and bull) or (p2==1 and not bull)
                    if chk_h:
                        if dr==1.0 and hi>=tp_p: ep=tp_p; et=1; closed=True
                    else:
                        if dr==-1.0 and lo<=tp_p: ep=tp_p; et=1; closed=True
                if not closed: i+=1
            tp+=dr*(ep-e)/pip2-spr2
        return tp

    @njit(cache=True)
    def _run_shuffled_breakout(ca,oa,ha,la,ah,al,sf_dirs,md2,tp2,mh2,mz2,mxz2,pip2,spr2):
        """Breakout sim with shuffled directions."""
        n=len(ca); tot=0.0; di=0; i=0
        while i<n and di<len(sf_dirs):
            uh=ah[i]; ul=al[i]
            if uh!=uh or ul!=ul or uh<=ul: i+=1; continue
            zw=(uh-ul)/pip2
            if zw<mz2 or zw>mxz2: i+=1; continue
            e=ca[i]
            if e<ul: depth=(ul-e)/pip2
            elif e>uh: depth=(e-uh)/pip2
            else: i+=1; continue
            if depth<md2*zw: i+=1; continue
            orig_ul=ul; orig_uh=uh
            dr=sf_dirs[di]; di+=1
            if dr==-1.0: tp_p=orig_ul-tp2*zw*pip2
            else:        tp_p=orig_uh+tp2*zw*pip2
            eb=i; i+=1; closed=False; ep=e
            while i<n and not closed:
                if (i-eb)>=mh2: ep=ca[i-1]; closed=True; break
                hi=ha[i]; lo=la[i]; cl=ca[i]; bull=cl>=oa[i]
                for p2 in range(2):
                    if closed: break
                    chk_h=(p2==0 and bull) or (p2==1 and not bull)
                    if chk_h:
                        if dr==1.0 and hi>=tp_p: ep=tp_p; closed=True
                    else:
                        if dr==-1.0 and lo<=tp_p: ep=tp_p; closed=True
                if not closed: i+=1
            tot+=dr*(ep-e)/pip2-spr2
        return tot

    # Warmup shuffled sims
    _dv = np.array([1.0,-1.0]*200, dtype=np.float64)
    _run_shuffled_bounce(_d,_d,_h,_l,_ah,_al,_dv,0.25,0.5,288,20.0,150.0,0.0001,1.4)
    _run_shuffled_breakout(_d,_d,_h,_l,_ah,_al,_dv,0.25,1.0,288,20.0,150.0,0.0001,1.4)

    rng = np.random.RandomState(42); exceed = 0; t0 = time.time()
    # Pre-count entries per pair for direction shuffling
    entry_counts = {}
    for p in PAIRS:
        d = data[p]; n_is = d['n_is']
        cnt = run_sim(strategy, params, p, slice(0, n_is))
        entry_counts[p] = len(cnt)

    for mc_i in range(N_MC):
        shuf_usd = 0.0
        for p in PAIRS:
            d = data[p]; n_is = d['n_is']
            nc = entry_counts[p]
            sf = rng.choice(np.array([-1.0,1.0]), nc).astype(np.float64)
            if strategy == 'A':
                tf, md, mh = params
                pips = _run_shuffled_bounce(d['ca'][:n_is],d['oa'][:n_is],d['ha'][:n_is],
                                             d['la'][:n_is],d['h4h'][:n_is],d['h4l'][:n_is],
                                             sf,tf,md,mh,MIN_ZW_PIPS_GLOBAL,MAX_ZW_PIPS,d['pip'],SPREAD)
            else:
                md, tp, mh = params
                pips = _run_shuffled_breakout(d['ca'][:n_is],d['oa'][:n_is],d['ha'][:n_is],
                                               d['la'][:n_is],d['h4h'][:n_is],d['h4l'][:n_is],
                                               sf,md,tp,mh,MIN_ZW_PIPS_GLOBAL,MAX_ZW_PIPS,d['pip'],SPREAD)
            shuf_usd += pips * d['pu'] * UNITS
        if shuf_usd >= real_usd:
            exceed += 1
        if (mc_i+1) % 500 == 0:
            print(f"  MC {mc_i+1}/{N_MC}  p={exceed/(mc_i+1):.4f}  {time.time()-t0:.0f}s", flush=True)

    p_val = exceed / N_MC
    print(f"\n  Real IS USD:   ${real_usd:+,.0f}")
    print(f"  perm_p={p_val:.4f}  ({'✅ CAUSAL' if p_val < 0.05 else '❌ NOT SIGNIFICANT'})")
    return p_val


def oos_report(strategy, winner, perm_p, label):
    """Final OOS validation."""
    print(f"\n{'═'*72}")
    print(f"  STRATEGY {strategy} OOS — {label}")
    params = winner['params']
    if strategy == 'A':
        tf, md, mh = params
        md_s = f"{md:.2f}" if md < 9 else "all"
        ph = f"{mh//288:.0f}d" if mh >= 288 else f"{mh//12}h"
        print(f"  tgt={tf}  maxDepth={md_s}  hold={ph}")
    else:
        md, tp, mh = params
        ph = f"{mh//288:.0f}d" if mh >= 288 else f"{mh//12}h"
        print(f"  minDepth={md}×ZW  tp={tp}×ZW  hold={ph}")
    print(f"{'═'*72}")
    print(f"  {'Pair':<10} {'Win%':>6} {'TP%':>6} {'Time%':>6} "
          f"{'AvgW':>7} {'AvgL':>7} {'SQN':>6} {'n':>5} {'Total$':>9}")
    print("  " + "─"*70)
    oos_pnls = []; tot_usd = 0.0; tot_pips = 0.0; tot_nc = 0
    for p in PAIRS:
        d = data[p]; sl = slice(d['n_is'], d['n'])
        if strategy == 'A':
            tf, md, mh = params
            pa, ea = _sim_bounce(d['ca'][d['n_is']:],d['oa'][d['n_is']:],
                                  d['ha'][d['n_is']:],d['la'][d['n_is']:],
                                  d['h4h'][d['n_is']:],d['h4l'][d['n_is']:],
                                  tf,md,mh,MIN_ZW_PIPS_GLOBAL,MAX_ZW_PIPS,d['pip'],SPREAD)
        else:
            md, tp, mh = params
            pa, ea = _sim_breakout(d['ca'][d['n_is']:],d['oa'][d['n_is']:],
                                    d['ha'][d['n_is']:],d['la'][d['n_is']:],
                                    d['h4h'][d['n_is']:],d['h4l'][d['n_is']:],
                                    md,tp,mh,MIN_ZW_PIPS_GLOBAL,MAX_ZW_PIPS,d['pip'],SPREAD)
        if len(pa) == 0:
            print(f"  {p:<10} (no trades)"); continue
        nc = len(pa); usd = pa.sum() * d['pu'] * UNITS
        ntp = int((ea==1).sum()); nw = int((pa>0).sum())
        aw = pa[pa>0].mean() if nw > 0 else 0.0
        al = pa[pa<=0].mean() if (nc-nw) > 0 else 0.0
        sym = '🟢' if usd > 0 else '🔴'
        print(f"  {p:<10} {nw/nc*100:>5.1f}% {ntp/nc*100:>5.1f}% {(nc-ntp)/nc*100:>5.1f}% "
              f"{aw:>+6.1f}p {al:>+6.1f}p {sqn(pa):>5.2f} {nc:>5,} ${usd:>+8,.0f} {sym}")
        oos_pnls.append(pa); tot_usd += usd; tot_pips += pa.sum(); tot_nc += nc

    all_oos = np.concatenate(oos_pnls) if oos_pnls else np.zeros(1)
    n_pos   = sum(1 for p in oos_pnls if p.sum() > 0)
    avg_oos_bars = int(np.mean([d['n'] - d['n_is'] for d in data.values()]))
    ppd = tot_pips / (avg_oos_bars / 288)
    oos_sqn = sqn(all_oos)
    print("  " + "─"*70)
    print(f"  {'TOTAL':<10}{'':>6}{'':>6}{'':>6}{'':>7}{'':>7} "
          f"{oos_sqn:>5.2f} {tot_nc:>5,} ${tot_usd:>+8,.0f}")
    print(f"\n  Pairs+: {n_pos}/12  pips/day: {ppd:+.1f}  SQN: {oos_sqn:.2f}  perm_p: {perm_p:.4f}")
    gates = {
        'SQN > 1.0':         oos_sqn > 1.0,
        'Pairs >= 10/12':    n_pos >= 10,
        'pips/day > 0':      ppd > 0,
        'perm_p < 0.05':     perm_p < 0.05,
        'WF min-chunk > 0':  winner['wf_min'] > 0,
    }
    for g, v in gates.items():
        print(f"  {'✅' if v else '❌'} {g}")
    ng = sum(gates.values())
    verdict = '✅ PASS' if ng==5 else f'🟡 PARTIAL ({ng}/5)' if ng>=3 else f'❌ FAIL ({ng}/5)'
    print(f"\n  {verdict}")
    return ng


# ══════════════════════════════════════════════════════════════════════════════
# RUN BOTH STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════
grid_A = list(product(A_TGT_FRACS, A_MAX_DEPTHS, A_MAX_HOLDS))
grid_B = list(product(B_MIN_DEPTHS, B_TP_FRACS,  B_MAX_HOLDS))

# Strategy A
res_A    = sweep_strategy('A', grid_A, 'BOUNCE — TP + time exit, no SL')
win_A    = wf_validate('A', res_A, 'BOUNCE')
pval_A   = mc_permutation('A', win_A, 'BOUNCE')
gates_A  = oos_report('A', win_A, pval_A, 'BOUNCE')

# Strategy B
res_B    = sweep_strategy('B', grid_B, 'BREAKOUT — significant S/R penetration')
win_B    = wf_validate('B', res_B, 'BREAKOUT')
pval_B   = mc_permutation('B', win_B, 'BREAKOUT')
gates_B  = oos_report('B', win_B, pval_B, 'BREAKOUT')

# Summary
print(f"\n{'═'*72}")
print("  FINAL SUMMARY")
print(f"{'═'*72}")
print(f"  Strategy A (Bounce):    {gates_A}/5 gates  perm_p={pval_A:.4f}")
print(f"  Strategy B (Breakout):  {gates_B}/5 gates  perm_p={pval_B:.4f}")
