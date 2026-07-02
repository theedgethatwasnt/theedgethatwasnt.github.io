"""
Bootstrap MC on random-trail CHF_JPY (and other validated random pairs).
Params: N=1, ZW=40, tgt=20, ta=5, td=3 (the live config).
Reports per-cycle P&L distribution + P5, P50, P95, Sharpe, P(+).
"""
import math
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR = Path('/path/to/projects/fx-core/data/m5_ohlc')

SPREAD   = 1.4
MAX_LEGS = 10
PF       = 1.25
N_BOOT   = 2000
OOS_FRAC = 0.30

CONFIGS = [
    # pair,   zw,   tgt,  ta,  td,  pip
    ("CHF_JPY", 40.0, 20.0, 5.0, 3.0, 0.01),
    ("NZD_JPY", 40.0, 20.0, 5.0, 3.0, 0.01),
    ("USD_JPY", 40.0, 20.0, 10.0, 5.0, 0.01),
]


@njit
def sim_zr_trail_cycles(op, hi, lo, cl, pip, spread, pf, ml, zw, tgt, ta, td):
    """Random-entry alternating ZR with trailing stop. Returns per-cycle P&L array."""
    n = len(cl)
    cycle_pnl = np.zeros(n, dtype=np.float64)
    nc = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1
    while i < n:
        e = cl[i]
        if d == 1:
            uz = e; lz = e - zw*pip; ut = e + tgt*pip; lt = lz - tgt*pip
        else:
            lz = e; uz = e + zw*pip; lt = e - tgt*pip; ut = uz + tgt*pip
        lv[0] = 1.0; ld[0] = float(d); lp[0] = e
        nl = 1; lu = ll = -1; ex = False
        peak_mfe = 0.0; trail_on = False
        i += 1
        while i < n and not ex:
            h = hi[i]; l = lo[i]; c = cl[i]; bull = c >= op[i]
            if nl == 1:
                cur_mfe = (h - e) / pip if d == 1 else (e - l) / pip
                if cur_mfe > peak_mfe:
                    peak_mfe = cur_mfe
                if peak_mfe >= ta:
                    trail_on = True
                if trail_on:
                    if d == 1:
                        ts = e + (peak_mfe - td) * pip
                        if l <= ts:
                            cycle_pnl[nc] = (ts - e) / pip - spread
                            nc += 1; ex = True
                    else:
                        ts = e - (peak_mfe - td) * pip
                        if h >= ts:
                            cycle_pnl[nc] = (e - ts) / pip - spread
                            nc += 1; ex = True
            if ex:
                break
            for pn in range(2):
                if ex:
                    break
                dh = (bull and pn == 0) or (not bull and pn == 1)
                if l <= ut <= h:
                    net = 0.0; tv = 0.0
                    for k in range(nl):
                        net += lv[k] * ld[k] * (ut - lp[k]) / pip; tv += lv[k]
                    cycle_pnl[nc] = net - tv*spread; nc += 1; ex = True; break
                if l <= lt <= h:
                    net = 0.0; tv = 0.0
                    for k in range(nl):
                        net += lv[k] * ld[k] * (lt - lp[k]) / pip; tv += lv[k]
                    cycle_pnl[nc] = net - tv*spread; nc += 1; ex = True; break
                if dh and h >= uz and lu != i:
                    lu = i; nt2 = 0.0; tv = 0.0
                    for k in range(nl):
                        nt2 += lv[k] * ld[k] * (ut - lp[k]) / pip; tv += lv[k]
                    nt2 -= tv * spread
                    if nt2 >= 0:
                        if c >= ut:
                            cycle_pnl[nc] = nt2; nc += 1; ex = True; break
                    else:
                        v = max(1.0, math.ceil(-nt2 / tgt * pf))
                        if nl >= ml:
                            net = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net += lv[k] * ld[k] * (c - lp[k]) / pip; tv2 += lv[k]
                            cycle_pnl[nc] = net - tv2*spread; nc += 1; ex = True; break
                        lv[nl] = v; ld[nl] = 1.0; lp[nl] = uz; nl += 1
                if not dh and l <= lz and ll != i:
                    ll = i; nt2 = 0.0; tv = 0.0
                    for k in range(nl):
                        nt2 += lv[k] * ld[k] * (lt - lp[k]) / pip; tv += lv[k]
                    nt2 -= tv * spread
                    if nt2 >= 0:
                        if c <= lt:
                            cycle_pnl[nc] = nt2; nc += 1; ex = True; break
                    else:
                        v = max(1.0, math.ceil(-nt2 / tgt * pf))
                        if nl >= ml:
                            net = 0.0; tv2 = 0.0
                            for k in range(nl):
                                net += lv[k] * ld[k] * (c - lp[k]) / pip; tv2 += lv[k]
                            cycle_pnl[nc] = net - tv2*spread; nc += 1; ex = True; break
                        lv[nl] = v; ld[nl] = -1.0; lp[nl] = lz; nl += 1
            i += 1
        d = -d
    return cycle_pnl[:nc]


print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR / 'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o = _df0.open.values[:2000].astype(np.float64)
_h = _df0.high.values[:2000].astype(np.float64)
_l = _df0.low.values[:2000].astype(np.float64)
_c = _df0.close.values[:2000].astype(np.float64)
sim_zr_trail_cycles(_o, _h, _l, _c, 0.0001, SPREAD, PF, MAX_LEGS, 30.0, 15.0, 5.0, 3.0)
print("done.\n")

rng = np.random.default_rng(42)

print(f"{'Pair':<10} {'ppd':>8} {'c/day':>7} {'ppc':>7} | {'P5':>7} {'P50':>7} {'P95':>7} {'Sharpe':>7} {'P(+)':>6} | gates")
print("─" * 90)

for pair, zw, tgt, ta, td, pip in CONFIGS:
    df = pd.read_parquet(DATA_DIR / f'{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
    op = df.open.values.astype(np.float64)
    hi = df.high.values.astype(np.float64)
    lo = df.low.values.astype(np.float64)
    cl = df.close.values.astype(np.float64)
    nb = len(cl)
    is_end = int(nb * (1 - OOS_FRAC))
    oos_op = op[is_end:]; oos_hi = hi[is_end:]
    oos_lo = lo[is_end:]; oos_cl = cl[is_end:]
    oos_days = len(oos_cl) / (24 * 12)

    cyc = sim_zr_trail_cycles(oos_op, oos_hi, oos_lo, oos_cl, pip, SPREAD, PF, MAX_LEGS, zw, tgt, ta, td)
    nc = len(cyc)
    obs_ppd = cyc.sum() / oos_days
    ppc     = cyc.mean()
    cday    = nc / oos_days

    boot = np.array([rng.choice(cyc, size=nc, replace=True).sum() / oos_days for _ in range(N_BOOT)])
    p5, p50, p95 = np.percentile(boot, [5, 50, 95])
    sharpe   = boot.mean() / (boot.std() + 1e-9)
    prob_pos = np.mean(boot > 0)

    gate_p5   = p5 > 0
    gate_prob = prob_pos > 0.95
    gates     = int(gate_p5) + int(gate_prob)

    print(f"{pair:<10} {obs_ppd:>8.1f} {cday:>7.1f} {ppc:>7.2f} | "
          f"{p5:>7.1f} {p50:>7.1f} {p95:>7.1f} {sharpe:>7.2f} {prob_pos:>6.3f} | "
          f"{'✅' if gate_p5 else '❌'}p5  {'✅' if gate_prob else '❌'}P(+)  {gates}/2")

    print(f"  Cycle P&L dist: min={cyc.min():.1f}  max={cyc.max():.1f}  "
          f"trail%={100*np.mean(cyc < tgt - td):.0f}% (cycles exiting below tgt)")
    print(f"  Per-cycle histogram: <0={100*np.mean(cyc<0):.1f}%  "
          f"0-10={100*np.mean((cyc>=0)&(cyc<10)):.1f}%  "
          f"10-50={100*np.mean((cyc>=10)&(cyc<50)):.1f}%  "
          f"50+={100*np.mean(cyc>=50):.1f}%")
    print()
