"""
ZR spread sensitivity — two analyses in one script:

1. Variable spread backtest: replaces constant 1.4p with per-bar time-of-day (TOD)
   model. Spreads widen during Asian dead zone (21-22 UTC) and tighten during
   London/NY overlap (13-16 UTC). Compares validated configs against the constant-
   spread baseline to show real-world degradation.

2. Spread gate analysis: shows how a MAX_ENTRY_SPREAD filter (matching the live
   strategy's new gate) affects cycle count and ppd — confirms it eliminates
   expensive-spread entries without destroying frequency.

TOD multiplier table (UTC hour → spread multiplier on base 1.4p):
  Dead zone  21-22h → 1.90-1.95× (2.66-2.73p)
  Sydney     00-06h → 1.40-1.60× (1.96-2.24p)
  Tokyo      06-08h → 1.20-1.30× (1.68-1.82p)
  London     08-12h → 1.00-1.05× (1.40-1.47p)
  Lon/NY     12-17h → 0.90-0.95× (1.26-1.33p)  ← tightest
  NY solo    17-21h → 1.05-1.30× (1.47-1.82p)

Output: zr_spread_sensitivity_results.csv
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR = Path('/path/to/projects/fx-core/data/m5_ohlc')
OUT_PATH = Path(__file__).parent / 'zr_spread_sensitivity_results.csv'

SPREAD_CONST = 1.4
MAX_LEGS     = 10
PF           = 1.25
OOS_FRAC     = 0.30
WF_CHUNKS    = 3
N_BOOT       = 2000

# Time-of-day spread multiplier table (index = UTC hour 0..23)
# Base spread × multiplier → pip spread at that hour
_TOD_MULT = np.array([
    1.60,   # 00 UTC — Sydney dead
    1.55,   # 01
    1.50,   # 02
    1.45,   # 03
    1.40,   # 04 — Sydney/Asia active
    1.40,   # 05
    1.30,   # 06 — Tokyo/Sydney overlap
    1.20,   # 07 — London pre-open
    1.05,   # 08 — London open
    1.00,   # 09
    1.00,   # 10
    1.00,   # 11
    0.95,   # 12 — Lon/NY overlap (tightest)
    0.90,   # 13
    0.90,   # 14
    0.90,   # 15
    0.95,   # 16
    1.05,   # 17 — NY solo
    1.10,   # 18
    1.20,   # 19
    1.30,   # 20
    1.90,   # 21 — dead zone (NY close, Asia not open)
    1.95,   # 22
    1.75,   # 23
], dtype=np.float64)


# configs to test: (pair, zw, tgt, pip, ta, td)  — validated from sessions 022/025
CONFIGS = [
    ("CHF_JPY", 40.0, 20.0, 0.01,  5, 1),
    ("CHF_JPY", 40.0, 20.0, 0.01,  3, 1),
    ("CHF_JPY", 40.0, 20.0, 0.01,  6, 5),
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
    max_entry_spread: skip new cycle if spread_arr[i] > this value (0 = no gate).
    Returns (cycle_pnl, n_trail, n_zr, n_spread_skipped)
    """
    n = len(cl)
    cycle_pnl = np.zeros(n, dtype=np.float64)
    nc = 0; n_trail = 0; n_zr = 0; n_skipped = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1
    while i < n:
        sp_entry = spread_arr[i]
        # Spread gate on entry
        if max_entry_spread > 0 and sp_entry > max_entry_spread:
            n_skipped += 1
            i += 1
            continue

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
                    nt2-=tv*sp
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
                    nt2-=tv*sp
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


def build_tod_spread_arr(timestamps: pd.Series,
                         base_spread: float = SPREAD_CONST) -> np.ndarray:
    """Time-of-day spread model. Dead zone (21-22 UTC) → ~2.7p; Lon/NY → ~1.26p."""
    hours = timestamps.dt.hour.values.astype(np.int32)
    return (base_spread * _TOD_MULT[hours]).astype(np.float64)


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR/'CHF_JPY_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_sp0 = np.full(2000, SPREAD_CONST)
_o=_df0.open.values[:2000].astype(np.float64); _h=_df0.high.values[:2000].astype(np.float64)
_l=_df0.low.values[:2000].astype(np.float64);  _c=_df0.close.values[:2000].astype(np.float64)
sim_zr_varspread(_o,_h,_l,_c,_sp0,0.01,PF,MAX_LEGS,40.,20.,5.,1.,0.)
print("done.\n")

# Print TOD spread table for inspection
print("TOD spread profile (UTC hour → spread pips):")
for h, m in enumerate(_TOD_MULT):
    bar = '█' * int(m * 10)
    print(f"  {h:>2}h  {SPREAD_CONST*m:.2f}p  {bar}")
print()

rng  = np.random.default_rng(42)
rows = []

print(f"{'Pair':<10} {'ta':>4} {'td':>4} | {'const p/d':>10} {'tod p/d':>10} {'delta':>8} | "
      f"{'gate p/d':>10} {'gate skip%':>10} {'neg%':>7} | verdict")
print('─' * 100)

for pair, zw, tgt, pip, ta, td in CONFIGS:
    parquet = DATA_DIR / f'{pair}_M5.parquet'
    if not parquet.exists():
        print(f"[{pair}] missing"); continue

    df = pd.read_parquet(parquet).sort_values('timestamp').reset_index(drop=True)
    nb = len(df)
    is_end   = int(nb * (1 - OOS_FRAC))
    oos_days = (nb - is_end) / (24 * 12)

    op=df.open.values.astype(np.float64); hi=df.high.values.astype(np.float64)
    lo=df.low.values.astype(np.float64);  cl=df.close.values.astype(np.float64)

    sp_tod   = build_tod_spread_arr(df['timestamp'])
    sp_const = np.full(nb, SPREAD_CONST)

    # Run all three scenarios on OOS slice
    cyc_const, nt_c, nz_c, _ = sim_zr_varspread(
        op[is_end:],hi[is_end:],lo[is_end:],cl[is_end:],sp_const[is_end:],
        pip,PF,MAX_LEGS,zw,tgt,float(ta),float(td),0.)
    cyc_tod, nt_v, nz_v, _ = sim_zr_varspread(
        op[is_end:],hi[is_end:],lo[is_end:],cl[is_end:],sp_tod[is_end:],
        pip,PF,MAX_LEGS,zw,tgt,float(ta),float(td),0.)
    cyc_gate, nt_g, nz_g, n_skip = sim_zr_varspread(
        op[is_end:],hi[is_end:],lo[is_end:],cl[is_end:],sp_tod[is_end:],
        pip,PF,MAX_LEGS,zw,tgt,float(ta),float(td),2.5)

    ppd_c = cyc_const.sum() / oos_days if len(cyc_const) else 0
    ppd_v = cyc_tod.sum()   / oos_days if len(cyc_tod)   else 0
    ppd_g = cyc_gate.sum()  / oos_days if len(cyc_gate)  else 0

    neg_pct_v    = 100*(cyc_tod  < 0).mean() if len(cyc_tod)  else 0
    neg_pct_gate = 100*(cyc_gate < 0).mean() if len(cyc_gate) else 0
    skip_pct     = 100 * n_skip / max(len(cyc_gate) + n_skip, 1)
    delta        = ppd_v - ppd_c
    delta_gate   = ppd_g - ppd_c

    verdict = ("✅ robust" if abs(delta) < 0.05*abs(ppd_c) and neg_pct_v < 3
               else ("⚠️  mild" if abs(delta) < 0.15*abs(ppd_c)
               else "❌ sensitive"))

    print(f"{pair:<10} {ta:>4} {td:>4} | {ppd_c:>10.1f} {ppd_v:>10.1f} {delta:>+8.1f} | "
          f"{ppd_g:>10.1f} {skip_pct:>9.1f}% {neg_pct_gate:>6.1f}% | {verdict}")
    sys.stdout.flush()

    rows.append(dict(pair=pair, ta=ta, td=td, zw=zw, tgt=tgt,
                     ppd_const=round(ppd_c,1), ppd_tod=round(ppd_v,1),
                     ppd_gate=round(ppd_g,1),
                     delta_tod=round(delta,1), delta_gate=round(delta_gate,1),
                     neg_pct_tod=round(neg_pct_v,2), neg_pct_gate=round(neg_pct_gate,2),
                     skip_pct=round(skip_pct,1)))

pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(rows)} rows → {OUT_PATH}")

print()
print("=== INTERPRETATION ===")
print("const p/d  : backtest with fixed 1.4p spread (existing model)")
print("tod p/d    : per-bar time-of-day spread (London/NY ~1.26p, dead zone ~2.7p)")
print("delta      : tod minus const — positive means Lon/NY concentration wins")
print("gate p/d   : tod spread + skip entries where spread > 2.5p")
print("gate skip% : fraction of would-be entries blocked by spread gate")
print("neg%       : fraction of individual cycles with negative P&L (with gate)")
print()
print("Key: if gate p/d ≥ tod p/d, the spread gate is free alpha (blocks dead-zone entries)")
print("     if tod p/d ≥ const p/d, the strategy naturally concentrates in tight-spread hours")
