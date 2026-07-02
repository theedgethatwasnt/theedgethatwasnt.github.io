"""
Exp 1 — Absorption Filter on ZR Entry
======================================
Hypothesis: ZR's 5+ leg blowup cycles are caused by entering during "liquidity runs"
(large momentum bars with no wick rejection). Blocking those entries reduces deep-leg
cycles without destroying cycle frequency.

Three candle-quality signals tested as entry gates:
  A. body_thresh  — block if adverse-direction body/range > threshold
  B. wick_thresh  — require rejection wick toward entry direction > threshold
  C. mom_bars     — block if last K consecutive bars all run against entry direction

Direction d is NOT flipped on a skipped bar — it stays ready until a qualifying bar.

Baseline: EUR_USD ZW=30 TGT=21 ta=10 td=1 gate=0 PSAR af=0.01
           → 1,442 p/d OOS, P5=688, IS=3/3 OOS=3/3, nc=484

Cross-pairs: EUR_JPY (ZW=50 ta=6), GBP_USD (ZW=30 ta=6)

Grid: 5 × 4 × 4 = 80 combinations per pair
Output: results/zr_absorption_results.csv
"""
import math, sys, itertools
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_PATH     = Path(__file__).parent / 'results/zr_absorption_results.csv'
OUT_PATH.parent.mkdir(exist_ok=True)

PF       = 1.25
ML       = 10
OOS_FRAC = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 500
N_PERM     = 200

# PSAR escape (deployed best)
AF0  = 0.01
AFST = 0.01
AFMX = 0.20

# Pairs: (pair, pip, ZW, TGT, ta, td)
PAIRS = [
    ("EUR_USD", 0.0001, 30.0, 21.0, 10.0, 1.0),
    ("EUR_JPY", 0.01,   50.0, 25.0,  6.0, 1.0),
    ("GBP_USD", 0.0001, 30.0, 21.0,  6.0, 1.0),
]

# ── Parameter grid ────────────────────────────────────────────────────────────
# body_thresh: block entry if adverse-direction body/range > this
#   0.0 = disabled (no body filter)
#   0.4 = block if body > 40% of range in wrong direction (very permissive)
#   0.6 = moderate
#   0.7 = strict (only enters on near-doji or wick bars)
BODY_THRESHS  = [0.0, 0.4, 0.5, 0.6, 0.7]

# wick_thresh: require rejection wick toward entry direction > this fraction of range
#   0.0 = disabled
#   0.10 = 10% of range as wick (very small wick OK)
#   0.20 = 20%
#   0.30 = prominent wick required
WICK_THRESHS  = [0.0, 0.10, 0.20, 0.30]

# mom_bars: block if last K bars ALL have body going against entry direction
#   0 = disabled
#   2 = block if 2 consecutive adverse bars
#   3 = block if 3 consecutive adverse bars
#   5 = block if 5 consecutive adverse bars
MOM_BARS_LIST = [0, 2, 3, 5]

# Total: 5 × 4 × 4 = 80 combinations × 3 pairs = 240 runs


# ── Numba simulation ──────────────────────────────────────────────────────────
@njit
def check_absorption(op, hi, lo, cl, i, d, body_thresh, wick_thresh, mom_bars):
    """
    Returns True if bar i is an acceptable ZR entry given direction d.
    d = +1 → LONG entry wanted; d = -1 → SHORT entry wanted.

    Signal A (body_thresh > 0):
        For LONG: block if bearish body fraction > body_thresh
        For SHORT: block if bullish body fraction > body_thresh
        Adverse body = large move against the entry direction = aggression/run signal.

    Signal B (wick_thresh > 0):
        For LONG: require lower wick fraction ≥ wick_thresh (price rejected downward)
        For SHORT: require upper wick fraction ≥ wick_thresh (price rejected upward)
        Rejection wick = limit order absorption present at that level.

    Signal C (mom_bars > 0):
        Block if the last mom_bars bars all had a close that moved against d.
        E.g., for LONG: last 3 bars all bearish = momentum run downward, don't fade.
    """
    rng = hi[i] - lo[i]
    eps = 1e-10

    # Signal A — adverse body filter
    if body_thresh > 0.0 and rng > eps:
        if d == 1:   # LONG: adverse = bearish body (open > close)
            adverse_body = max(0.0, op[i] - cl[i]) / rng
        else:        # SHORT: adverse = bullish body (close > open)
            adverse_body = max(0.0, cl[i] - op[i]) / rng
        if adverse_body > body_thresh:
            return False

    # Signal B — directional wick requirement
    if wick_thresh > 0.0 and rng > eps:
        lo_body = cl[i] if cl[i] < op[i] else op[i]   # bottom of candle body
        hi_body = cl[i] if cl[i] > op[i] else op[i]   # top of candle body
        if d == 1:   # LONG: need lower wick (price tested low then rejected)
            lower_wick = (lo_body - lo[i]) / rng
            if lower_wick < wick_thresh:
                return False
        else:        # SHORT: need upper wick
            upper_wick = (hi[i] - hi_body) / rng
            if upper_wick < wick_thresh:
                return False

    # Signal C — consecutive momentum bars
    if mom_bars > 0 and i >= mom_bars:
        all_adverse = True
        for j in range(1, mom_bars + 1):
            prev = i - j
            if d == 1:   # LONG: adverse prev bar = close < open (bearish)
                if cl[prev] >= op[prev]:   # bullish prev bar breaks run
                    all_adverse = False
                    break
            else:        # SHORT: adverse prev bar = close > open (bullish)
                if cl[prev] <= op[prev]:
                    all_adverse = False
                    break
        if all_adverse:
            return False

    return True


@njit
def sim_zr_absorption(op, hi, lo, cl, sp_arr, pip, pf, ml, zw, tgt, ta, td,
                      af0, af_step, af_max,
                      body_thresh, wick_thresh, mom_bars):
    """
    ZR with PSAR escape + absorption entry filter.
    Direction d is preserved across skipped bars (not flipped on skip).
    Returns: (pnl array, nlegs array, etype array, nc, n_skipped)
    """
    n     = len(cl)
    pnl   = np.zeros(n, dtype=np.float64)
    nlegs = np.zeros(n, dtype=np.int32)
    etype = np.zeros(n, dtype=np.int32)
    nc       = 0
    n_skip   = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1

    while i < n:
        # ── Absorption gate ────────────────────────────────────────────────
        if not check_absorption(op, hi, lo, cl, i, d, body_thresh, wick_thresh, mom_bars):
            n_skip += 1
            i += 1
            continue   # d is NOT flipped — keep same direction for next bar

        # ── Entry ─────────────────────────────────────────────────────────
        e = cl[i]
        if d == 1: uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:      lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lc=0; ex=False; peak=0.0; ton=False
        psar_on=False; psar_val=0.0; ep_val=0.0; af_cur=af0; net_dir=0.0
        i += 1

        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; sp=sp_arr[i]; bull=(c>=op[i])

            # ── PSAR escape ────────────────────────────────────────────────
            if psar_on:
                if net_dir > 0:
                    if h > ep_val: ep_val=h; af_cur=min(af_cur+af_step, af_max)
                    psar_val = ep_val - (ep_val - psar_val) * af_cur
                    if l <= psar_val:
                        net=0.0; tv=0.0
                        for k in range(nl): net+=lv[k]*ld[k]*(psar_val-lp[k])/pip; tv+=lv[k]
                        pnl[nc]=net-tv*sp; nlegs[nc]=nl; etype[nc]=5; nc+=1; ex=True
                else:
                    if l < ep_val: ep_val=l; af_cur=min(af_cur+af_step, af_max)
                    psar_val = ep_val + (psar_val - ep_val) * af_cur
                    if h >= psar_val:
                        net=0.0; tv=0.0
                        for k in range(nl): net+=lv[k]*ld[k]*(psar_val-lp[k])/pip; tv+=lv[k]
                        pnl[nc]=net-tv*sp; nlegs[nc]=nl; etype[nc]=5; nc+=1; ex=True
                if ex: break
                i += 1; continue

            # ── 1-leg trailing stop ────────────────────────────────────────
            if nl == 1:
                mfe = (h-e)/pip if d==1 else (e-l)/pip
                if mfe > peak: peak = mfe
                if peak >= ta: ton = True
                if ton:
                    if d == 1:
                        be=e+sp*pip; ts=e+(peak-td)*pip
                        if ts < be: ts = be
                        if l <= ts: pnl[nc]=(ts-e)/pip-sp; nlegs[nc]=1; etype[nc]=1; nc+=1; ex=True
                    else:
                        be=e-sp*pip; ts=e-(peak-td)*pip
                        if ts > be: ts = be
                        if h >= ts: pnl[nc]=(e-ts)/pip-sp; nlegs[nc]=1; etype[nc]=1; nc+=1; ex=True
            if ex: break

            # ── Zone crossings then escape targets ─────────────────────────
            for pi2 in range(2):
                if ex: break
                is_hi = (bull == (pi2 == 0))
                if is_hi and h >= uz and lc != 1:
                    lc = 1; net=0.0; tv=0.0; nv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]; nv+=lv[k]*ld[k]
                    net -= tv*sp
                    if net < 0 and nv < 0:  # net short → long recovery valid
                        npu=max(tgt-sp,1e-8); v=max(1.0,math.ceil(-net/npu*pf))
                        if nl >= ml:
                            nc2=0.0; tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp; nlegs[nc]=nl; etype[nc]=3; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if (not is_hi) and l <= lz and lc != -1:
                    lc = -1; net=0.0; tv=0.0; nv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]; nv+=lv[k]*ld[k]
                    net -= tv*sp
                    if net < 0 and nv > 0:  # net long → short recovery valid
                        npu=max(tgt-sp,1e-8); v=max(1.0,math.ceil(-net/npu*pf))
                        if nl >= ml:
                            nc2=0.0; tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp; nlegs[nc]=nl; etype[nc]=3; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
                if ex: break
                if l <= ut <= h:
                    net_v=0.0
                    for k in range(nl): net_v+=lv[k]*ld[k]
                    net_dir=1.0 if net_v>=0 else -1.0
                    psar_on=True; af_cur=af0; ep_val=ut
                    psar_val=ut-tgt*pip if net_dir>0 else ut+tgt*pip
                    break
                if l <= lt <= h:
                    net_v=0.0
                    for k in range(nl): net_v+=lv[k]*ld[k]
                    net_dir=1.0 if net_v>=0 else -1.0
                    psar_on=True; af_cur=af0; ep_val=lt
                    psar_val=lt-tgt*pip if net_dir>0 else lt+tgt*pip
                    break
            i += 1

        d = -d

    return pnl[:nc], nlegs[:nc], etype[:nc], nc, n_skip


# ── JIT warm-up ──────────────────────────────────────────────────────────────
_df0 = pd.read_parquet(DATA_DIR_MID/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df0['open'].values[:2000].astype(np.float64)
_h=_df0['high'].values[:2000].astype(np.float64)
_l=_df0['low'].values[:2000].astype(np.float64)
_c=_df0['close'].values[:2000].astype(np.float64)
_s=np.full(2000, 0.00016, dtype=np.float64)
sim_zr_absorption(_o,_h,_l,_c,_s, 0.0001,1.25,10,30.,21.,10.,1., 0.01,0.01,0.20, 0.0,0.0,0)
print("JIT compiled.", flush=True)


# ── WF + bootstrap helper ─────────────────────────────────────────────────────
def permutation_p(pnl, n_perm):
    """One-sided permutation test: P(shuffled_sum >= observed_sum)."""
    if len(pnl) == 0: return 1.0
    obs = pnl.sum()
    cnt = 0
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=len(pnl))
        if (pnl * signs).sum() >= obs: cnt += 1
    return cnt / n_perm


def run_wf_boot(pnl, nc, n_days, n_chunks, n_boot, is_mode):
    """Walk-forward gate + bootstrap P5/P(+). Returns (wf_pass, ppd, p5, p_pos)."""
    chunk_size = nc // n_chunks
    if chunk_size == 0:
        return 0, 0.0, 0.0, 0.0
    wf_pass = 0
    for ch in range(n_chunks):
        s = ch * chunk_size
        e = s + chunk_size if ch < n_chunks - 1 else nc
        if pnl[s:e].sum() > 0:
            wf_pass += 1
    ppd = pnl.sum() / max(n_days, 1)
    # Bootstrap P5 and P(+)
    sums = np.array([np.random.choice(pnl, size=len(pnl), replace=True).sum()
                     for _ in range(n_boot)])
    p5    = float(np.percentile(sums / max(n_days, 1), 5))
    p_pos = float((sums > 0).mean())
    return wf_pass, ppd, p5, p_pos


# ── Main sweep ────────────────────────────────────────────────────────────────
rows = []
total = len(PAIRS) * len(BODY_THRESHS) * len(WICK_THRESHS) * len(MOM_BARS_LIST)
done  = 0

for pair, pip, zw, tgt, ta, td in PAIRS:
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

    n_total   = len(cl)
    n_oos     = int(n_total * OOS_FRAC)
    n_is      = n_total - n_oos
    oos_days  = n_oos * 5 / (60 * 24)   # M5 bars → calendar days

    print(f"\n{'='*70}")
    print(f"PAIR: {pair}  ZW={zw}p TGT={tgt}p ta={ta} td={td}  OOS={oos_days:.0f}d")
    print(f"{'body_thr':>8} {'wick_thr':>8} {'mom':>4} | {'p/d':>8} {'P5':>8} "
          f"{'nc':>6} {'skip%':>6} {'1L%':>5} {'5+%':>5} {'IS':>4} {'OOS':>4} {'perm_p':>7} | status")
    print('-'*80)

    for body_thresh, wick_thresh, mom_bars in itertools.product(
            BODY_THRESHS, WICK_THRESHS, MOM_BARS_LIST):

        done += 1
        tag = f"b{body_thresh:.1f}_w{wick_thresh:.2f}_m{mom_bars}"

        # IS walk-forward
        is_wf = 0
        is_chunk = n_is // IS_CHUNKS
        for ch in range(IS_CHUNKS):
            s = ch * is_chunk
            e = s + is_chunk if ch < IS_CHUNKS - 1 else n_is
            p, _, _, nc_ch, _ = sim_zr_absorption(
                op[s:e], hi[s:e], lo[s:e], cl[s:e], sp[s:e],
                pip, PF, ML, zw, tgt, ta, td, AF0, AFST, AFMX,
                body_thresh, wick_thresh, mom_bars)
            if nc_ch > 0 and p.sum() > 0: is_wf += 1

        # OOS walk-forward
        oos_wf = 0
        oos_chunk = n_oos // OOS_CHUNKS
        oos_pnl_all = []
        for ch in range(OOS_CHUNKS):
            s = n_is + ch * oos_chunk
            e = n_is + (s - n_is + oos_chunk) if ch < OOS_CHUNKS - 1 else n_total
            p, _, _, nc_ch, _ = sim_zr_absorption(
                op[s:e], hi[s:e], lo[s:e], cl[s:e], sp[s:e],
                pip, PF, ML, zw, tgt, ta, td, AF0, AFST, AFMX,
                body_thresh, wick_thresh, mom_bars)
            if nc_ch > 0 and p.sum() > 0: oos_wf += 1
            oos_pnl_all.append(p)

        # Full OOS metrics
        pnl_full, nlegs_full, _, nc_full, n_skip = sim_zr_absorption(
            op[n_is:], hi[n_is:], lo[n_is:], cl[n_is:], sp[n_is:],
            pip, PF, ML, zw, tgt, ta, td, AF0, AFST, AFMX,
            body_thresh, wick_thresh, mom_bars)

        n_entries_tried = (n_oos - n_is) + nc_full + n_skip   # rough approx
        skip_pct  = 100.0 * n_skip / max(n_skip + nc_full, 1)
        l1_pct    = 100.0 * (nlegs_full == 1).sum() / max(nc_full, 1)
        l5p_pct   = 100.0 * (nlegs_full >= 5).sum() / max(nc_full, 1)
        ppd       = pnl_full.sum() / max(oos_days, 1) if nc_full > 0 else 0.0

        wf_boot, ppd2, p5, p_pos = run_wf_boot(
            pnl_full, nc_full, oos_days, OOS_CHUNKS, N_BOOT, is_mode=False)
        perm_p = permutation_p(pnl_full, N_PERM)

        gate_pass  = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
                      and p5 > 0 and p_pos > 0.95 and perm_p < 0.05)
        status = "🟢 PASS" if gate_pass else f"{is_wf}/{oos_wf} p={perm_p:.2f}"

        print(f"{body_thresh:>8.1f} {wick_thresh:>8.2f} {mom_bars:>4d} | "
              f"{ppd:>8.1f} {p5:>8.1f} {nc_full:>6d} {skip_pct:>5.1f}% "
              f"{l1_pct:>4.1f}% {l5p_pct:>4.1f}% {is_wf:>4}/{IS_CHUNKS} "
              f"{oos_wf:>4}/{OOS_CHUNKS} {perm_p:>7.3f} | {status}")
        sys.stdout.flush()

        rows.append(dict(
            pair=pair, zw=zw, tgt=tgt, ta=ta, td=td,
            body_thresh=body_thresh, wick_thresh=wick_thresh, mom_bars=mom_bars,
            ppd=round(ppd, 1), p5=round(p5, 1), p_pos=round(p_pos, 3),
            nc=nc_full, skip_pct=round(skip_pct, 1),
            l1_pct=round(l1_pct, 1), l5p_pct=round(l5p_pct, 1),
            is_wf=is_wf, oos_wf=oos_wf, perm_p=round(perm_p, 3),
            gate_pass=gate_pass,
        ))

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
print(f"\n\nResults saved → {OUT_PATH}")

# ── Summary: top configs per pair by p/d (gate pass only) ────────────────────
print("\n=== TOP PASSING CONFIGS PER PAIR (gate=PASS, sorted by p/d) ===")
for pair, _, _, _, _, _ in PAIRS:
    sub = df[(df.pair == pair) & df.gate_pass].sort_values('ppd', ascending=False)
    if sub.empty:
        print(f"\n{pair}: NO passing configs")
        continue
    print(f"\n{pair} — {len(sub)} passing configs:")
    print(sub[['body_thresh','wick_thresh','mom_bars','ppd','p5','nc',
               'skip_pct','l1_pct','l5p_pct','perm_p']].head(10).to_string(index=False))

# Baseline reference
print("\nBaseline (no filter): EUR_USD p/d=1442, P5=688, nc=484, 5+%=3.5")
