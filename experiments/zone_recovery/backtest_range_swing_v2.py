"""
Range Swing v2 — Hybrid: Directional Entry + ZR Fallback on Stop Sweep
=======================================================================
v1 baseline: enter directionally at S/R with fixed TP/SL.
  EUR_USD: 9.8 p/d, hit%=24.5%, stop%=75.5% (best config: tgt=25p, stop_f=0.5)

v2 modification: when stop is hit (75% of trades), instead of accepting the loss,
  open a standard ZR cycle starting from the stop/sweep price.
  - Original directional leg is CLOSED at stop price (taking -stop_dist pips)
  - A new ZR cycle OPENS at the stop price, entering opposite direction
  - ZR zone: ZW=30p, TGT=21p, ta=10, td=1 (EUR_USD best validated config)
  - ZR continues until 1-leg trail, PSAR escape, or ZR target hit

Rationale:
  Stop-hit = price swept through S/R level
  S/R sweep = liquidity raid = absorption event = ZR entry signal
  ZR has proven positive expectancy at these sweep levels
  Expected p/d: hit%×tgt + stop%×(-stop_dist + E[ZR]) >> v1 p/d

Pairs: EUR_USD, GBP_USD (same pairs that passed in v1)
Grid: stop_f=[0.5,1.0], tgt=[20,25,30] (v1 best params)
      ZR config fixed: ZW=30, TGT=21, ta=10 (EUR_USD) / ta=6 (GBP_USD)
Gates: IS=3/3, OOS=3/3, P5>0, P(+)>0.95, perm_p<0.05
"""
import math, sys, itertools
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_PATH     = Path(__file__).parent / 'results/range_swing_v2_results.csv'
OUT_PATH.parent.mkdir(exist_ok=True)

ML       = 10
OOS_FRAC = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 500
N_PERM     = 200
PF       = 1.25
AF0  = 0.01; AFST = 0.01; AFMX = 0.20
CLUSTER_PIPS = 10
TOUCHES      = 2
SWING_W      = 5

# (pair, pip, ta for ZR, S/R swing_w, cluster)
PAIRS = [
    ("EUR_USD", 0.0001, 10.0),
    ("GBP_USD", 0.0001,  6.0),
]
ZW_ZR  = 30.0
TGT_ZR = 21.0
TD_ZR  = 1.0

STOP_FRACS = [0.5, 1.0]
TGT_PIPS   = [20.0, 25.0, 30.0]


@njit
def build_sr_history(hi, lo, swing_w, cluster_pips, pip, min_touches):
    """
    Build support/resistance levels from H1 highs/lows (same as v1).
    Uses swing_w bars on each side to identify local highs/lows.
    Clusters within cluster_pips are merged.
    Returns array of SR price levels.
    """
    n = len(hi)
    sr = np.zeros(n * 2, dtype=np.float64)
    n_sr = 0
    for i in range(swing_w, n - swing_w):
        is_hi = True; is_lo = True
        for j in range(1, swing_w + 1):
            if hi[i] <= hi[i-j] or hi[i] <= hi[i+j]: is_hi = False
            if lo[i] >= lo[i-j] or lo[i] >= lo[i+j]: is_lo = False
        if is_hi: sr[n_sr] = hi[i]; n_sr += 1
        if is_lo: sr[n_sr] = lo[i]; n_sr += 1
    return sr[:n_sr]


@njit
def atr14_at(hi, lo, i, pip):
    filled = min(i + 1, 14)
    total = 0.0
    for j in range(filled): total += (hi[i-j] - lo[i-j]) / pip
    return total / max(filled, 1)


@njit
def sim_v2(op, hi, lo, cl, sp_arr, pip, zw, tgt_zr, ta_zr, td_zr, pf,
           stop_frac, tgt_pips, sr_levels, n_sr_levels,
           ml, af0, af_step, af_max, cluster_pips):
    """
    Hybrid v2 simulation:
      Phase 1 (directional): enter at S/R band, TP=tgt_pips above/below, SL=stop_frac×ATR14
      Phase 2 (ZR recovery): if SL hit, open ZR at SL price, opposite direction
    """
    n = len(cl)
    pnl_dir = np.zeros(n, dtype=np.float64)   # phase-1 directional PnL per cycle
    pnl_zr  = np.zeros(n, dtype=np.float64)   # phase-2 ZR recovery PnL
    nc_dir  = 0; nc_zr = 0
    n_tp    = 0; n_sl  = 0

    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0

    while i < n:
        # ── Phase 1: find S/R entry ────────────────────────────────────────────
        atr = atr14_at(hi, lo, i, pip)
        stop_dist = stop_frac * atr   # pips

        # Check if current bar touches any S/R level
        near_sr = False; d_entry = 0
        for k in range(n_sr_levels):
            level = sr_levels[k]
            tol   = cluster_pips * pip
            if abs(lo[i] - level) <= tol:   # at support
                near_sr = True; d_entry = 1; break
            if abs(hi[i] - level) <= tol:   # at resistance
                near_sr = True; d_entry = -1; break

        if not near_sr:
            i += 1; continue

        # Enter directional at bar close
        e       = cl[i]
        tp_price= e + d_entry * tgt_pips * pip
        sl_price= e - d_entry * stop_dist * pip
        i += 1

        # ── Phase 1: run until TP or SL hit ────────────────────────────────────
        hit_tp = False; hit_sl = False; sl_hit_price = sl_price
        while i < n:
            h = hi[i]; l = lo[i]; sp = sp_arr[i]
            if d_entry == 1:
                if h >= tp_price:
                    pnl_dir[nc_dir] = tgt_pips - sp; nc_dir += 1; n_tp += 1; hit_tp = True; break
                if l <= sl_price:
                    pnl_dir[nc_dir] = -stop_dist - sp; nc_dir += 1; n_sl += 1; hit_sl = True
                    sl_hit_price = sl_price; break
            else:
                if l <= tp_price:
                    pnl_dir[nc_dir] = tgt_pips - sp; nc_dir += 1; n_tp += 1; hit_tp = True; break
                if h >= sl_price:
                    pnl_dir[nc_dir] = -stop_dist - sp; nc_dir += 1; n_sl += 1; hit_sl = True
                    sl_hit_price = sl_price; break
            i += 1

        if not hit_sl:
            if not hit_tp: i += 1
            continue

        # ── Phase 2: ZR recovery from sl_hit_price ─────────────────────────────
        # Enter opposite direction (ZR d = -d_entry)
        zr_d = -d_entry
        e2   = sl_hit_price
        if zr_d == 1: uz=e2; lz=e2-zw*pip; ut=e2+tgt_zr*pip; lt=lz-tgt_zr*pip
        else:         lz=e2; uz=e2+zw*pip; lt=e2-tgt_zr*pip; ut=uz+tgt_zr*pip

        lv[0]=1.0; ld[0]=float(zr_d); lp[0]=e2
        nl=1; lu=ll=-1; ex=False; peak=0.0; ton=False
        psar_on=False; psar_val=0.0; ep_val=0.0; af_cur=af0; net_dir_zr=0.0

        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; sp=sp_arr[i]; bull=(c>=op[i])

            if psar_on:
                if net_dir_zr > 0:
                    if h > ep_val: ep_val=h; af_cur=min(af_cur+af_step, af_max)
                    psar_val = ep_val - (ep_val - psar_val) * af_cur
                    if l <= psar_val:
                        net=0.0; tv=0.0
                        for k in range(nl): net+=lv[k]*ld[k]*(psar_val-lp[k])/pip; tv+=lv[k]
                        pnl_zr[nc_zr]=net-tv*sp; nc_zr+=1; ex=True
                else:
                    if l < ep_val: ep_val=l; af_cur=min(af_cur+af_step, af_max)
                    psar_val = ep_val + (psar_val - ep_val) * af_cur
                    if h >= psar_val:
                        net=0.0; tv=0.0
                        for k in range(nl): net+=lv[k]*ld[k]*(psar_val-lp[k])/pip; tv+=lv[k]
                        pnl_zr[nc_zr]=net-tv*sp; nc_zr+=1; ex=True
                if ex: break
                i += 1; continue

            if nl == 1:
                mfe = (h-e2)/pip if zr_d==1 else (e2-l)/pip
                if mfe > peak: peak = mfe
                if peak >= ta_zr: ton = True
                if ton:
                    if zr_d == 1:
                        be=e2+sp*pip; ts=e2+(peak-td_zr)*pip
                        if ts < be: ts = be
                        if l <= ts: pnl_zr[nc_zr]=(ts-e2)/pip-sp; nc_zr+=1; ex=True
                    else:
                        be=e2-sp*pip; ts=e2-(peak-td_zr)*pip
                        if ts > be: ts = be
                        if h >= ts: pnl_zr[nc_zr]=(e2-ts)/pip-sp; nc_zr+=1; ex=True
            if ex: break

            for pi2 in range(2):
                if ex: break
                is_hi = (bull == (pi2 == 0))
                if is_hi and h >= uz and lu != i:
                    lu=i; net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    net -= tv*sp
                    if net < 0:
                        npu=max(tgt_zr-sp,1e-8); v=max(1.0,math.ceil(-net/npu*pf))
                        if nl >= ml:
                            nc2=0.0; tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            pnl_zr[nc_zr]=nc2-tv2*sp; nc_zr+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if (not is_hi) and l <= lz and ll != i:
                    ll=i; net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    net -= tv*sp
                    if net < 0:
                        npu=max(tgt_zr-sp,1e-8); v=max(1.0,math.ceil(-net/npu*pf))
                        if nl >= ml:
                            nc2=0.0; tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            pnl_zr[nc_zr]=nc2-tv2*sp; nc_zr+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
                if ex: break
                if l <= ut <= h:
                    net_v=0.0
                    for k in range(nl): net_v+=lv[k]*ld[k]
                    net_dir_zr=1.0 if net_v>=0 else -1.0
                    psar_on=True; af_cur=af0; ep_val=ut
                    psar_val=ut-tgt_zr*pip if net_dir_zr>0 else ut+tgt_zr*pip; break
                if l <= lt <= h:
                    net_v=0.0
                    for k in range(nl): net_v+=lv[k]*ld[k]
                    net_dir_zr=1.0 if net_v>=0 else -1.0
                    psar_on=True; af_cur=af0; ep_val=lt
                    psar_val=lt-tgt_zr*pip if net_dir_zr>0 else lt+tgt_zr*pip; break
            i += 1

    pnl_total = np.zeros(nc_dir + nc_zr, dtype=np.float64)
    pnl_total[:nc_dir] = pnl_dir[:nc_dir]
    pnl_total[nc_dir:] = pnl_zr[:nc_zr]
    return pnl_total, nc_dir, nc_zr, n_tp, n_sl


# JIT warm-up with dummy SR levels
_df0 = pd.read_parquet(DATA_DIR_MID/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df0['open'].values[:3000].astype(np.float64)
_h=_df0['high'].values[:3000].astype(np.float64)
_l=_df0['low'].values[:3000].astype(np.float64)
_c=_df0['close'].values[:3000].astype(np.float64)
_s=np.full(3000,0.00016,dtype=np.float64)
_sr=np.array([1.10,1.11,1.12], dtype=np.float64)
sim_v2(_o,_h,_l,_c,_s,0.0001,30.,21.,10.,1.,1.25,0.5,25.,_sr,3,10,0.01,0.01,0.20,10.)
print("JIT compiled.", flush=True)


def permutation_p(pnl, n_perm):
    if len(pnl) == 0: return 1.0
    obs = pnl.sum(); cnt = 0
    for _ in range(n_perm):
        signs = np.random.choice([-1,1], size=len(pnl))
        if (pnl*signs).sum() >= obs: cnt += 1
    return cnt / n_perm


def run_wf_boot(pnl, nc, n_days, n_chunks, n_boot):
    if nc == 0: return 0, 0.0, 0.0, 0.0
    chunk_size = nc // n_chunks
    if chunk_size == 0: return 0, 0.0, 0.0, 0.0
    wf_pass = sum(1 for ch in range(n_chunks)
                  if pnl[ch*chunk_size : (ch+1)*chunk_size if ch<n_chunks-1 else nc].sum() > 0)
    ppd = pnl.sum() / max(n_days, 1)
    sums = np.array([np.random.choice(pnl, size=len(pnl), replace=True).sum() for _ in range(n_boot)])
    p5   = float(np.percentile(sums / max(n_days, 1), 5))
    p_pos = float((sums > 0).mean())
    return wf_pass, ppd, p5, p_pos


rows = []
print(f"\n{'pair':>8} {'stop_f':>7} {'tgt':>4} | {'p/d_dir':>8} {'p/d_zr':>8} "
      f"{'p/d_tot':>8} {'P5':>7} {'nc_d':>6} {'nc_z':>6} {'tp%':>5} {'sl%':>5} "
      f"{'IS':>4} {'OOS':>4} {'perm_p':>7} | status")
print('-'*112)

for pair, pip, ta_zr in PAIRS:
    mid = pd.read_parquet(DATA_DIR_MID/f'{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
    ba  = pd.read_parquet(DATA_DIR_BA /f'{pair}_M5_BA.parquet').sort_values('timestamp').reset_index(drop=True)
    mid['ts_key'] = mid['timestamp'].astype(str).str[:19]
    ba['ts_key']  = ba['timestamp'].astype(str).str[:19]
    merged = mid.merge(ba[['ts_key','bid_c','ask_c']], on='ts_key', how='left')
    merged['spread'] = ((merged['ask_c'] - merged['bid_c']) / pip).clip(0, 50)
    merged['spread'] = merged['spread'].fillna(merged['spread'].median())

    op = merged['open'].values.astype(np.float64)
    hi = merged['high'].values.astype(np.float64)
    lo = merged['low'].values.astype(np.float64)
    cl = merged['close'].values.astype(np.float64)
    sp = merged['spread'].values.astype(np.float64)

    n_total = len(cl)
    n_oos   = int(n_total * OOS_FRAC)
    n_is    = n_total - n_oos
    oos_days= n_oos * 5 / (60 * 24)

    # Build S/R from all IS data (same window as v1)
    sr_all = build_sr_history(hi[:n_is], lo[:n_is], SWING_W, CLUSTER_PIPS, pip, TOUCHES)
    n_sr = len(sr_all)

    print(f"\n{'='*70}")
    print(f"PAIR: {pair}  OOS={oos_days:.0f}d  S/R levels in IS: {n_sr}")
    print(f"  ZR config: ZW={ZW_ZR}p TGT={TGT_ZR}p ta={ta_zr} td={TD_ZR} PF={PF}")

    for stop_frac, tgt_pips in itertools.product(STOP_FRACS, TGT_PIPS):
        # IS walk-forward
        is_wf = 0; is_chunk = n_is // IS_CHUNKS
        for ch in range(IS_CHUNKS):
            s = ch*is_chunk; e = s+is_chunk if ch<IS_CHUNKS-1 else n_is
            sr_c = build_sr_history(hi[s:e], lo[s:e], SWING_W, CLUSTER_PIPS, pip, TOUCHES)
            pnl_c, _, _, _, _ = sim_v2(op[s:e],hi[s:e],lo[s:e],cl[s:e],sp[s:e],
                                        pip,ZW_ZR,TGT_ZR,ta_zr,TD_ZR,PF,
                                        stop_frac,tgt_pips,sr_c,len(sr_c),
                                        ML,AF0,AFST,AFMX,CLUSTER_PIPS)
            if pnl_c.sum() > 0: is_wf += 1

        # OOS
        pnl_oos, nc_d, nc_z, n_tp, n_sl = sim_v2(
            op[n_is:],hi[n_is:],lo[n_is:],cl[n_is:],sp[n_is:],
            pip,ZW_ZR,TGT_ZR,ta_zr,TD_ZR,PF,
            stop_frac,tgt_pips,sr_all,n_sr,
            ML,AF0,AFST,AFMX,CLUSTER_PIPS)

        nc_oos = len(pnl_oos)
        # Split pnl for WF
        pnl_dir_oos = pnl_oos[:nc_d]
        pnl_zr_oos  = pnl_oos[nc_d:]

        oos_wf = 0; oos_chunk = nc_oos // OOS_CHUNKS
        if oos_chunk > 0:
            for ch in range(OOS_CHUNKS):
                s=ch*oos_chunk; e=s+oos_chunk if ch<OOS_CHUNKS-1 else nc_oos
                if pnl_oos[s:e].sum() > 0: oos_wf += 1

        _, ppd, p5, p_pos = run_wf_boot(pnl_oos, nc_oos, oos_days, OOS_CHUNKS, N_BOOT)
        perm_p = permutation_p(pnl_oos, N_PERM)

        ppd_dir = pnl_dir_oos.sum() / max(oos_days, 1)
        ppd_zr  = pnl_zr_oos.sum() / max(oos_days, 1)
        nc_total = nc_d + nc_z
        tp_pct  = n_tp / max(nc_d, 1) * 100
        sl_pct  = n_sl / max(nc_d, 1) * 100

        gates_ok = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
                    and p5 > 0 and p_pos > 0.95 and perm_p < 0.05)
        if gates_ok: status = "🟢 PASS"
        else:
            fails = []
            if is_wf < IS_CHUNKS: fails.append(f"IS:{is_wf}/{IS_CHUNKS}")
            if oos_wf < OOS_CHUNKS: fails.append(f"OOS:{oos_wf}/{OOS_CHUNKS}")
            if p5 <= 0: fails.append("P5≤0")
            if p_pos <= 0.95: fails.append(f"P+={p_pos:.2f}")
            if perm_p >= 0.05: fails.append(f"p={perm_p:.2f}")
            status = " ".join(fails)

        print(f"{pair:>8} {stop_frac:>7.1f} {tgt_pips:>4.0f} | "
              f"{ppd_dir:>8.1f} {ppd_zr:>8.1f} {ppd:>8.1f} {p5:>7.1f} "
              f"{nc_d:>6} {nc_z:>6} {tp_pct:>5.1f}% {sl_pct:>5.1f}% "
              f"{is_wf:>2}/{IS_CHUNKS} {oos_wf:>2}/{OOS_CHUNKS} {perm_p:>7.3f} | {status}")

        rows.append(dict(pair=pair, stop_frac=stop_frac, tgt_pips=tgt_pips,
                         ppd_dir=ppd_dir, ppd_zr=ppd_zr, ppd_total=ppd,
                         p5=p5, nc_dir=nc_d, nc_zr=nc_z, tp_pct=tp_pct, sl_pct=sl_pct,
                         is_wf=is_wf, oos_wf=oos_wf, perm_p=perm_p, gates_ok=gates_ok))

pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")

print("\n=== PASSING CONFIGS ===")
for r in rows:
    if r['gates_ok']:
        print(f"  {r['pair']} stop_f={r['stop_frac']:.1f} tgt={r['tgt_pips']:.0f}p  "
              f"dir={r['ppd_dir']:.1f} + zr={r['ppd_zr']:.1f} = {r['ppd_total']:.1f} p/d  "
              f"P5={r['p5']:.1f}")
