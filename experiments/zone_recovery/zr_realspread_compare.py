"""
3-way real-spread comparison for validated ZR configs.

Variant A  — const 1.4p everywhere (existing baseline)
Variant B  — real bid/ask spread, NO entry gate (enter even during dead zone,
              but hedge sizing uses actual spread so nt2 is always correct)
Variant C  — real bid/ask spread + entry gate (skip if spread > MAX_GATE)

Requires: data/m5_ba/{pair}_M5_BA.parquet (fetch with fetch_m5_ba.py)

Output: zr_realspread_compare_results.csv
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_PATH     = Path(__file__).parent / 'zr_realspread_compare_results.csv'

SPREAD_CONST = 1.4
MAX_GATE     = 2.5   # pip threshold for Variant C entry gate
MAX_LEGS     = 10
PF           = 1.25
OOS_FRAC     = 0.30

# Validated configs from Sessions 022/025
CONFIGS = [
    ("CHF_JPY", 40.0, 20.0, 0.01,  5, 1),
    ("CHF_JPY", 40.0, 20.0, 0.01,  3, 1),
    ("AUD_JPY", 50.0, 25.0, 0.01,  4, 1),
    ("EUR_JPY", 50.0, 25.0, 0.01,  3, 1),
    ("EUR_JPY", 50.0, 25.0, 0.01, 30, 3),
    ("NZD_JPY", 40.0, 20.0, 0.01,  2, 1),
    ("USD_JPY", 40.0, 20.0, 0.01,  7, 1),
    ("GBP_USD", 30.0, 15.0, 0.0001,5, 1),
    ("CAD_JPY", 50.0, 12.5, 0.01,  2, 1),
]


@njit
def sim_zr_varspread(op, hi, lo, cl, spread_arr, pip, pf, ml, zw, tgt, ta, td,
                     max_entry_spread):
    """
    ZR trail sim with per-bar spread array.
    spread_arr[i] is used both for entry gate (if max_entry_spread > 0)
    and for nt2 hedge sizing + exit P&L on each bar.
    Returns (cycle_pnl, n_trail, n_zr, n_skipped)
    """
    n = len(cl)
    cycle_pnl = np.zeros(n, dtype=np.float64)
    nc = 0; n_trail = 0; n_zr = 0; n_skipped = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1
    while i < n:
        sp_entry = spread_arr[i]
        if max_entry_spread > 0 and sp_entry > max_entry_spread:
            n_skipped += 1; i += 1; continue

        e = cl[i]
        if d == 1:
            uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:
            lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak_mfe=0.0; trail_on=False
        i += 1
        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; bull=c>=op[i]
            sp = spread_arr[i]
            if nl == 1:
                cur_mfe = (h-e)/pip if d==1 else (e-l)/pip
                if cur_mfe > peak_mfe: peak_mfe = cur_mfe
                if peak_mfe >= ta: trail_on = True
                if trail_on:
                    if d == 1:
                        ts = e + (peak_mfe-td)*pip
                        if l <= ts:
                            cycle_pnl[nc]=(ts-e)/pip-sp; nc+=1; n_trail+=1; ex=True
                    else:
                        ts = e - (peak_mfe-td)*pip
                        if h >= ts:
                            cycle_pnl[nc]=(e-ts)/pip-sp; nc+=1; n_trail+=1; ex=True
            if ex: break
            for pn in range(2):
                if ex: break
                dh=(bull and pn==0) or (not bull and pn==1)
                if l<=ut<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    cycle_pnl[nc]=net-tv*sp; nc+=1; n_zr+=1; ex=True; break
                if l<=lt<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    cycle_pnl[nc]=net-tv*sp; nc+=1; n_zr+=1; ex=True; break
                if dh and h>=uz and lu!=i:
                    lu=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*sp          # <-- uses actual spread for hedge sizing
                    if nt2>=0:
                        if c>=ut: cycle_pnl[nc]=nt2; nc+=1; n_zr+=1; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            cycle_pnl[nc]=net-tv2*sp; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if not dh and l<=lz and ll!=i:
                    ll=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*sp          # <-- uses actual spread for hedge sizing
                    if nt2>=0:
                        if c<=lt: cycle_pnl[nc]=nt2; nc+=1; n_zr+=1; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            cycle_pnl[nc]=net-tv2*sp; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            i += 1
        d = -d
    return cycle_pnl[:nc], n_trail, n_zr, n_skipped


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR_MID / 'CHF_JPY_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_sp0 = np.full(2000, SPREAD_CONST)
_o=_df0.open.values[:2000].astype(np.float64); _h=_df0.high.values[:2000].astype(np.float64)
_l=_df0.low.values[:2000].astype(np.float64);  _c=_df0.close.values[:2000].astype(np.float64)
sim_zr_varspread(_o,_h,_l,_c,_sp0,0.01,PF,MAX_LEGS,40.,20.,5.,1.,0.)
print("done.\n")

rows = []
print(f"{'Pair':<10} {'ta':>4} {'td':>4} | "
      f"{'A const':>9} | {'B real-ngt':>10} {'Δ B-A':>8} | "
      f"{'C real+gt':>10} {'Δ C-A':>8} {'skip%':>7} | "
      f"{'B neg%':>7} {'C neg%':>7} | best")
print('─' * 115)

for pair, zw, tgt, pip, ta, td in CONFIGS:
    ba_path = DATA_DIR_BA / f'{pair}_M5_BA.parquet'
    mid_path = DATA_DIR_MID / f'{pair}_M5.parquet'

    if not ba_path.exists():
        print(f"[{pair}] BA parquet missing — run fetch_m5_ba.py"); continue
    if not mid_path.exists():
        print(f"[{pair}] mid parquet missing"); continue

    # Load mid-price for OHLC (backtest uses mid for ZR zone decisions)
    df_mid = pd.read_parquet(mid_path).sort_values('timestamp').reset_index(drop=True)
    # Load BA for actual spread
    df_ba  = pd.read_parquet(ba_path).sort_values('timestamp').reset_index(drop=True)

    # Align on timestamp — take intersection
    df_mid['ts_key'] = df_mid['timestamp'].astype(str).str[:19]
    df_ba['ts_key']  = df_ba['timestamp'].astype(str).str[:19]
    merged = df_mid.merge(df_ba[['ts_key','bid_c','ask_c']], on='ts_key', how='inner')
    merged = merged.sort_values('ts_key').reset_index(drop=True)

    nb = len(merged)
    is_end   = int(nb * (1 - OOS_FRAC))
    oos_days = (nb - is_end) / (24 * 12)

    # Build arrays
    op = merged.open.values.astype(np.float64)
    hi = merged.high.values.astype(np.float64)
    lo = merged.low.values.astype(np.float64)
    cl = merged.close.values.astype(np.float64)
    sp_real = ((merged.ask_c - merged.bid_c) / pip).clip(lower=0.5).values.astype(np.float64)
    sp_const = np.full(nb, SPREAD_CONST)

    # OOS only
    op_o = op[is_end:]; hi_o = hi[is_end:]; lo_o = lo[is_end:]; cl_o = cl[is_end:]
    sp_real_o  = sp_real[is_end:]
    sp_const_o = sp_const[is_end:]

    ta_f = float(ta); td_f = float(td)

    # Variant A: constant 1.4p, no gate
    cyc_a, nt_a, nz_a, _ = sim_zr_varspread(
        op_o, hi_o, lo_o, cl_o, sp_const_o, pip, PF, MAX_LEGS, zw, tgt, ta_f, td_f, 0.)

    # Variant B: real spread, no gate
    cyc_b, nt_b, nz_b, _ = sim_zr_varspread(
        op_o, hi_o, lo_o, cl_o, sp_real_o,  pip, PF, MAX_LEGS, zw, tgt, ta_f, td_f, 0.)

    # Variant C: real spread, gate at MAX_GATE
    cyc_c, nt_c, nz_c, n_skip = sim_zr_varspread(
        op_o, hi_o, lo_o, cl_o, sp_real_o,  pip, PF, MAX_LEGS, zw, tgt, ta_f, td_f, MAX_GATE)

    ppd_a = cyc_a.sum() / oos_days if len(cyc_a) else 0
    ppd_b = cyc_b.sum() / oos_days if len(cyc_b) else 0
    ppd_c = cyc_c.sum() / oos_days if len(cyc_c) else 0

    neg_b = 100*(cyc_b < 0).mean() if len(cyc_b) else 0
    neg_c = 100*(cyc_c < 0).mean() if len(cyc_c) else 0
    skip_pct = 100 * n_skip / max(len(cyc_c) + n_skip, 1)

    da = ppd_b - ppd_a
    dc = ppd_c - ppd_a
    best = "B" if ppd_b >= ppd_c and ppd_b >= ppd_a else ("C" if ppd_c >= ppd_b else "A")

    # Spread stats for info
    sp_med = np.median(sp_real_o)
    sp_p90 = np.percentile(sp_real_o, 90)
    sp_gt25 = 100*(sp_real_o > 2.5).mean()

    print(f"{pair:<10} {ta:>4} {td:>4} | "
          f"{ppd_a:>9.1f} | {ppd_b:>10.1f} {da:>+8.1f} | "
          f"{ppd_c:>10.1f} {dc:>+8.1f} {skip_pct:>6.1f}% | "
          f"{neg_b:>6.1f}% {neg_c:>6.1f}% | {best}  "
          f"[sp: med={sp_med:.2f}p p90={sp_p90:.2f}p >{MAX_GATE:.0f}p={sp_gt25:.1f}%]")
    sys.stdout.flush()

    rows.append(dict(
        pair=pair, ta=ta, td=td,
        ppd_const=round(ppd_a,1), ppd_real_nogate=round(ppd_b,1), ppd_real_gate=round(ppd_c,1),
        delta_b=round(da,1), delta_c=round(dc,1),
        neg_pct_b=round(neg_b,2), neg_pct_c=round(neg_c,2),
        skip_pct=round(skip_pct,1),
        sp_median=round(float(sp_med),3), sp_p90=round(float(sp_p90),3),
        pct_gt25=round(float(sp_gt25),1),
        best=best,
    ))

pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(rows)} rows → {OUT_PATH}")
print()
print("A = constant 1.4p spread everywhere (existing baseline)")
print("B = real spread, no entry gate (correct hedge sizing throughout)")
print("C = real spread + skip entry when spread > 2.5p")
print("Δ = vs A  |  best = highest p/d variant")
