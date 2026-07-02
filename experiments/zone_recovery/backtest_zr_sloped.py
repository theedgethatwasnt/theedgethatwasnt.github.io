"""
Exp 3 — Sloped Channel ZR
===========================
Hypothesis: replacing fixed horizontal zones with a linear regression channel
allows ZR to be trend-aware at entry time. Price touching the lower channel band
in an uptrend is a mean-reversion entry; touching the upper band in a downtrend.
Once entered, zone geometry is FIXED (standard ZR math applies unchanged).

Key design choices:
  1. Entry direction determined by band position (NOT alternating)
     → LONG when price ≤ lower_band; SHORT when price ≥ upper_band
     → No entry when price is inside the channel (wait for band touch)
  2. Optional absorption filter: require wick rejection at band (same as Exp 1)
  3. Zone geometry at entry: standard ZR anchored to entry price
     → uz = entry, lz = entry - ZW (for LONG); ut = entry + TGT; lt = lz - TGT
     → ZW = channel width (regression N × slope ≈ ATR × multiplier)
     OR ZW = fixed pips (simpler, tested first)
  4. PSAR escape trail on multi-leg exits (same as deployed)
  5. Zone fixed for cycle duration — midline only moves between cycles

Two zone width strategies tested:
  ZW_MODE = "fixed":  ZW is a fixed pip value (same as current ZR)
  ZW_MODE = "atr":    ZW = atr20 × zw_atr_mult (adaptive to current volatility)

Parameters:
  reg_N         : regression lookback bars {20, 30, 50, 80}
  zw_mode       : {"fixed", "atr"}
  zw_fixed      : fixed ZW pips — {20, 30, 50} (for zw_mode="fixed")
  zw_atr_mult   : ZW = ATR20 × mult — {2.0, 3.0, 4.0, 5.0} (for "atr")
  tgt_frac      : TGT = ZW × tgt_frac — {0.5, 0.7, 1.0}
  ta            : trail activation pips — {6, 8, 10}
  wick_thresh   : absorption wick at channel touch — {0.0, 0.15, 0.25}
  min_channel_width : skip entry if channel width < X pips (avoids degen zones)

Pairs: EUR_USD (primary), EUR_JPY, GBP_USD
Gate: IS=3/3, OOS=3/3, P5>0, P(+)>95%
Compare to: horizontal ZW baseline on same pair/period

Output: results/zr_sloped_results.csv
"""
import math, sys, itertools
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_PATH     = Path(__file__).parent / 'results/zr_sloped_results.csv'
OUT_PATH.parent.mkdir(exist_ok=True)

PF       = 1.25
ML       = 10
TD       = 1.0
OOS_FRAC = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 500
N_PERM     = 200
AF0  = 0.01; AFST = 0.01; AFMX = 0.20   # PSAR escape

PAIRS = [
    ("EUR_USD", 0.0001),
    ("EUR_JPY", 0.01),
    ("GBP_USD", 0.0001),
]

# ── Parameter grid ────────────────────────────────────────────────────────────
REG_NS       = [20, 30]               # trimmed: 50/80 add runtime, marginal info
ZW_MODES     = ["fixed", "atr"]
ZW_FIXEDS    = [30.0, 50.0]
ZW_ATR_MULTS = [2.0, 3.0, 4.0, 5.0]
TGT_FRACS    = [0.5, 0.7, 1.0]
TA_VALUES    = [6.0, 10.0]            # trimmed: keep endpoints
WICK_THRESHS = [0.0, 0.15]           # trimmed: 0.25 consistently worse
MIN_CHAN_WIDTH= 5.0    # pips — skip if channel width < this (degenerate)


# ── Numba helpers ─────────────────────────────────────────────────────────────
@njit
def linreg_at(cl, i, N):
    """
    Causal linear regression of last N closes ending at bar i.
    Returns (slope_pips_per_bar, midline_at_i).
    Uses least-squares: y = slope * x + intercept, x=0..N-1, y=close values.
    """
    if i < N - 1:
        return 0.0, cl[i]
    x_mean = (N - 1) * 0.5
    y_mean = 0.0
    for j in range(N):
        y_mean += cl[i - N + 1 + j]
    y_mean /= N
    num = 0.0; den = 0.0
    for j in range(N):
        x = float(j) - x_mean
        y = cl[i - N + 1 + j] - y_mean
        num += x * y
        den += x * x
    slope = num / den if den > 1e-12 else 0.0
    intercept = y_mean - slope * x_mean
    mid = intercept + slope * (N - 1)   # midline value AT bar i (last point)
    return slope, mid


@njit
def atr20_at(hi, lo, i, pip, window=20):
    """Causal ATR20 in pips at bar i."""
    filled = min(i + 1, window)
    total  = 0.0
    for j in range(filled):
        total += (hi[i - j] - lo[i - j]) / pip
    return total / max(filled, 1)


@njit
def sim_zr_sloped(op, hi, lo, cl, sp_arr, pip, pf, ml, td,
                  af0, af_step, af_max,
                  reg_N, zw_mode_fixed,  # 1=fixed, 0=atr
                  zw_fixed_pips, zw_atr_mult,
                  tgt_frac, ta, wick_thresh, min_chan_width_pips):
    """
    ZR with sloped channel entry.

    Entry logic (per flat bar):
      1. Compute regression midline + channel bands
      2. If price at lower band AND (wick_thresh=0 OR lower wick OK) → LONG
         If price at upper band AND (wick_thresh=0 OR upper wick OK) → SHORT
         Else skip bar
      3. Fix zone at entry: uz=entry, lz=entry-ZW, ut=entry+TGT, lt=lz-TGT (LONG)
      4. Run standard ZR break-even sizing + PSAR escape

    ZW sources:
      zw_mode_fixed=1 → ZW = zw_fixed_pips
      zw_mode_fixed=0 → ZW = ATR20 × zw_atr_mult

    Returns: (pnl, nlegs, etype, nc, n_skip)
    """
    n     = len(cl)
    pnl   = np.zeros(n, dtype=np.float64)
    nlegs = np.zeros(n, dtype=np.int32)
    etype = np.zeros(n, dtype=np.int32)
    nc     = 0
    n_skip = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i  = 0

    while i < n:
        # ── Channel computation ────────────────────────────────────────────
        if i < reg_N:
            i += 1; n_skip += 1; continue

        _, mid = linreg_at(cl, i, reg_N)
        atr20  = atr20_at(hi, lo, i, pip)

        if zw_mode_fixed:
            zw = zw_fixed_pips
        else:
            zw = atr20 * zw_atr_mult
            zw = min(zw, 80.0)   # cap ATR-ZW at 80p — prevents convex sizing explosion
        zw = max(zw, min_chan_width_pips)   # floor

        half = (zw / 2.0) * pip
        upper_band = mid + half
        lower_band = mid - half
        tgt = zw * tgt_frac

        # ── Entry direction from band position ─────────────────────────────
        at_lower = (lo[i] <= lower_band)
        at_upper = (hi[i] >= upper_band)

        if not at_lower and not at_upper:
            i += 1; n_skip += 1; continue   # price inside channel, skip

        # When BOTH bands touched (rare — wide bar spans full channel), prefer
        # the direction consistent with close position
        if at_lower and at_upper:
            at_lower = cl[i] < mid
            at_upper = not at_lower

        d = 1 if at_lower else -1

        # ── Absorption wick check ─────────────────────────────────────────
        if wick_thresh > 0.0:
            rng = hi[i] - lo[i] + 1e-10
            lo_body = cl[i] if cl[i] < op[i] else op[i]
            hi_body = cl[i] if cl[i] > op[i] else op[i]
            if d == 1:
                lower_wick = (lo_body - lo[i]) / rng
                if lower_wick < wick_thresh:
                    i += 1; n_skip += 1; continue
            else:
                upper_wick = (hi[i] - hi_body) / rng
                if upper_wick < wick_thresh:
                    i += 1; n_skip += 1; continue

        # ── Zone geometry (fixed at entry) ─────────────────────────────────
        e  = cl[i]
        if d == 1: uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:      lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip

        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak=0.0; ton=False
        psar_on=False; psar_val=0.0; ep_val=0.0; af_cur=af0; net_dir=0.0
        i += 1

        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; sp=sp_arr[i]; bull=(c>=op[i])

            # PSAR escape
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

            # 1-leg trailing stop
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

            # Zone crossings → hedge legs; escape targets → PSAR
            for pi2 in range(2):
                if ex: break
                is_hi = (bull == (pi2 == 0))
                if is_hi and h >= uz and lu != i:
                    lu = i; net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    net -= tv*sp
                    if net < 0:
                        npu=max(tgt-sp,1e-8); v=max(1.0,math.ceil(-net/npu*pf))
                        if nl >= ml:
                            nc2=0.0; tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp; nlegs[nc]=nl; etype[nc]=3; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if (not is_hi) and l <= lz and ll != i:
                    ll = i; net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    net -= tv*sp
                    if net < 0:
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

    return pnl[:nc], nlegs[:nc], etype[:nc], nc, n_skip


# ── JIT warm-up ──────────────────────────────────────────────────────────────
_df0 = pd.read_parquet(DATA_DIR_MID/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df0['open'].values[:3000].astype(np.float64)
_h=_df0['high'].values[:3000].astype(np.float64)
_l=_df0['low'].values[:3000].astype(np.float64)
_c=_df0['close'].values[:3000].astype(np.float64)
_s=np.full(3000, 0.00016, dtype=np.float64)
sim_zr_sloped(_o,_h,_l,_c,_s, 0.0001,1.25,10,1., 0.01,0.01,0.20,
              30, 1, 30., 3.0, 0.7, 10., 0.0, 5.0)
print("JIT compiled.", flush=True)


# ── WF + bootstrap ────────────────────────────────────────────────────────────
def run_wf(op,hi,lo,cl,sp,pip,n_start,n_end,n_chunks,
           reg_N,zw_mode_int,zw_fixed,zw_atr,tgt_frac,ta,wick_thresh):
    chunk = (n_end - n_start) // n_chunks
    wf_pass = 0
    for ch in range(n_chunks):
        s = n_start + ch * chunk
        e = n_start + (ch + 1) * chunk if ch < n_chunks - 1 else n_end
        p,_,_,nc_ch,_ = sim_zr_sloped(
            op[s:e],hi[s:e],lo[s:e],cl[s:e],sp[s:e],pip,PF,ML,TD,
            AF0,AFST,AFMX, reg_N,zw_mode_int,zw_fixed,zw_atr,tgt_frac,ta,wick_thresh,MIN_CHAN_WIDTH)
        if nc_ch > 0 and p.sum() > 0: wf_pass += 1
    return wf_pass


def bootstrap_p5(pnl, n_days, n_boot):
    if len(pnl) == 0: return 0.0, 0.0, 0.0
    sums = np.array([np.random.choice(pnl, size=len(pnl), replace=True).sum()
                     for _ in range(n_boot)])
    ppd  = pnl.sum() / max(n_days, 1)
    p5   = float(np.percentile(sums / max(n_days, 1), 5))
    ppos = float((sums > 0).mean())
    return ppd, p5, ppos


def permutation_p(pnl, n_perm):
    """One-sided permutation test: P(shuffled_sum >= observed_sum)."""
    if len(pnl) == 0: return 1.0
    obs = pnl.sum()
    cnt = 0
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=len(pnl))
        if (pnl * signs).sum() >= obs: cnt += 1
    return cnt / n_perm


# ── Main sweep ────────────────────────────────────────────────────────────────
rows = []

for pair, pip in PAIRS:
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
    n_total  = len(cl)
    n_oos    = int(n_total * OOS_FRAC)
    n_is     = n_total - n_oos
    oos_days = n_oos * 5 / (60 * 24)

    print(f"\n{'='*76}")
    print(f"PAIR: {pair}  OOS={oos_days:.0f}d")
    print(f"{'reg_N':>6} {'mode':>6} {'ZW/atr':>7} {'tgt_f':>5} {'ta':>4} {'wick':>5} | "
          f"{'p/d':>8} {'P5':>8} {'nc':>6} {'skip%':>6} {'1L%':>5} {'5+%':>5} "
          f"{'IS':>4} {'OOS':>4} {'perm_p':>7} | status")
    print('-'*84)

    for reg_N, zw_mode, tgt_frac, ta, wick_thresh in itertools.product(
            REG_NS, ZW_MODES, TGT_FRACS, TA_VALUES, WICK_THRESHS):

        if zw_mode == "fixed":
            zw_vals = ZW_FIXEDS
            atr_vals = [3.0]   # unused but needed for signature
        else:
            zw_vals = [30.0]   # unused
            atr_vals = ZW_ATR_MULTS
        zw_mode_int = 1 if zw_mode == "fixed" else 0

        for zw_v in zw_vals:
            for atr_v in atr_vals:
                zw_label = f"{zw_v}p" if zw_mode == "fixed" else f"{atr_v}×ATR"

                is_wf = run_wf(op,hi,lo,cl,sp,pip, 0,n_is, IS_CHUNKS,
                               reg_N,zw_mode_int,zw_v,atr_v,tgt_frac,ta,wick_thresh)
                oos_wf= run_wf(op,hi,lo,cl,sp,pip, n_is,n_total, OOS_CHUNKS,
                               reg_N,zw_mode_int,zw_v,atr_v,tgt_frac,ta,wick_thresh)

                pnl_full,nlegs_full,_,nc_full,n_skip = sim_zr_sloped(
                    op[n_is:],hi[n_is:],lo[n_is:],cl[n_is:],sp[n_is:],pip,
                    PF,ML,TD, AF0,AFST,AFMX,
                    reg_N,zw_mode_int,zw_v,atr_v,tgt_frac,ta,wick_thresh,MIN_CHAN_WIDTH)

                if nc_full == 0:
                    continue

                skip_pct = 100.0 * n_skip / max(n_skip + nc_full, 1)
                l1_pct   = 100.0 * (nlegs_full == 1).sum() / max(nc_full, 1)
                l5p_pct  = 100.0 * (nlegs_full >= 5).sum() / max(nc_full, 1)
                ppd, p5, p_pos = bootstrap_p5(pnl_full, oos_days, N_BOOT)
                perm_p = permutation_p(pnl_full, N_PERM)

                gate_pass = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
                             and p5 > 0 and p_pos > 0.95 and perm_p < 0.05)
                status = "🟢 PASS" if gate_pass else f"{is_wf}/{oos_wf} p={perm_p:.2f}"

                print(f"{reg_N:>6d} {zw_mode:>6} {zw_label:>7} {tgt_frac:>5.1f} "
                      f"{ta:>4.0f} {wick_thresh:>5.2f} | "
                      f"{ppd:>8.1f} {p5:>8.1f} {nc_full:>6d} {skip_pct:>5.1f}% "
                      f"{l1_pct:>4.1f}% {l5p_pct:>4.1f}% "
                      f"{is_wf:>4}/{IS_CHUNKS} {oos_wf:>4}/{OOS_CHUNKS} {perm_p:>7.3f} | {status}")
                sys.stdout.flush()

                rows.append(dict(
                    pair=pair, reg_N=reg_N, zw_mode=zw_mode,
                    zw_val=zw_v if zw_mode=="fixed" else 0,
                    atr_mult=atr_v if zw_mode=="atr" else 0,
                    tgt_frac=tgt_frac, ta=ta, wick_thresh=wick_thresh,
                    ppd=round(ppd,1), p5=round(p5,1), p_pos=round(p_pos,3),
                    nc=nc_full, skip_pct=round(skip_pct,1),
                    l1_pct=round(l1_pct,1), l5p_pct=round(l5p_pct,1),
                    is_wf=is_wf, oos_wf=oos_wf, perm_p=round(perm_p,3),
                    gate_pass=gate_pass,
                ))

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
print(f"\n\nResults saved → {OUT_PATH}")

print("\n=== PASSING CONFIGS (all gates) ===")
passes = df[df.gate_pass].sort_values('ppd', ascending=False)
if passes.empty:
    print("None passed all gates.")
    print("\nBest OOS=3/3 configs:")
    print(df[df.oos_wf==3].sort_values('ppd', ascending=False).head(15)[
        ['pair','reg_N','zw_mode','zw_val','atr_mult','tgt_frac','ta','wick_thresh',
         'ppd','p5','nc','l1_pct','l5p_pct','is_wf','oos_wf']].to_string(index=False))
else:
    print(passes[['pair','reg_N','zw_mode','zw_val','atr_mult','tgt_frac','ta',
                  'wick_thresh','ppd','p5','nc','l1_pct','l5p_pct','perm_p']].to_string(index=False))

print("\n--- COMPARISON vs HORIZONTAL BASELINE ---")
print("EUR_USD horizontal: 1,442 p/d P5=688 IS=3/3 OOS=3/3")
print("EUR_JPY horizontal: 3,106 p/d P5=352 IS=3/3 OOS=3/3")
print("GBP_USD horizontal: 5,046 p/d P5=1306 IS=3/3 OOS=3/3")
