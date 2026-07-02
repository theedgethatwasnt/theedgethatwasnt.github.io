"""Random-entry Zone Recovery sweep on EUR_USD."""
import numpy as np
import pandas as pd
import math
from numba import njit
from pathlib import Path

df = pd.read_parquet('/path/to/projects/fx-core/data/m5_ohlc/EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
op = df.open.to_numpy(float); hi = df.high.to_numpy(float)
lo = df.low.to_numpy(float);  cl = df.close.to_numpy(float)
PIP = 0.0001; SPREAD = 1.4; MAX_LEGS = 10

n_total = len(df)
span_days = n_total * 5 / (60*24 * 5/7)
print(f"EUR_USD  {len(df):,} M5 bars  ≈ {span_days:.0f} trading days  ({span_days/252:.1f} yrs)\n")

@njit
def sim_random_zr(op, hi, lo, cl, pip, spread, pf, ml,
                  entry_every_n, zw_pips, tgt_pips):
    n = len(cl)
    total = 0.0; nc = 0; nt = 0; nm = 0; sl = 0.0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)

    i = 0; direction = 1
    while i < n:
        entry = cl[i]
        if direction == 1:
            uz = entry; lz = entry - zw_pips*pip
            ut = entry + tgt_pips*pip; lt = lz - tgt_pips*pip
        else:
            lz = entry; uz = entry + zw_pips*pip
            lt = entry - tgt_pips*pip; ut = uz + tgt_pips*pip
        lv[0]=1.0; ld[0]=float(direction); lp[0]=entry
        nl=1; lu=ll=-1; exited=False
        i += 1

        while i < n and not exited:
            h = hi[i]; l = lo[i]; c = cl[i]; bull = c >= op[i]
            for pass_n in range(2):
                if exited: break
                do_hi = (bull and pass_n==0) or (not bull and pass_n==1)
                if l <= ut <= h:
                    net = 0.0; tv = 0.0
                    for k in range(nl): net += lv[k]*ld[k]*(ut-lp[k])/pip; tv += lv[k]
                    total += net - tv*spread; nc += 1; nt += 1; sl += nl; exited = True; break
                if l <= lt <= h:
                    net = 0.0; tv = 0.0
                    for k in range(nl): net += lv[k]*ld[k]*(lt-lp[k])/pip; tv += lv[k]
                    total += net - tv*spread; nc += 1; nt += 1; sl += nl; exited = True; break
                if do_hi and h >= uz and lu != i:
                    lu = i
                    net_t = 0.0; tv = 0.0
                    for k in range(nl): net_t += lv[k]*ld[k]*(ut-lp[k])/pip; tv += lv[k]
                    net_t -= tv*spread
                    if net_t >= 0:
                        if c >= ut: total += net_t; nc += 1; nt += 1; sl += nl; exited = True; break
                    else:
                        vol = max(1.0, math.ceil(-net_t / tgt_pips * pf))
                        if nl >= ml:
                            net = 0.0; tv = 0.0
                            for k in range(nl): net += lv[k]*ld[k]*(c-lp[k])/pip; tv += lv[k]
                            total += net - tv*spread; nc += 1; nm += 1; sl += nl; exited = True; break
                        lv[nl]=vol; ld[nl]=1.0; lp[nl]=uz; nl += 1
                if not do_hi and l <= lz and ll != i:
                    ll = i
                    net_t = 0.0; tv = 0.0
                    for k in range(nl): net_t += lv[k]*ld[k]*(lt-lp[k])/pip; tv += lv[k]
                    net_t -= tv*spread
                    if net_t >= 0:
                        if c <= lt: total += net_t; nc += 1; nt += 1; sl += nl; exited = True; break
                    else:
                        vol = max(1.0, math.ceil(-net_t / tgt_pips * pf))
                        if nl >= ml:
                            net = 0.0; tv = 0.0
                            for k in range(nl): net += lv[k]*ld[k]*(c-lp[k])/pip; tv += lv[k]
                            total += net - tv*spread; nc += 1; nm += 1; sl += nl; exited = True; break
                        lv[nl]=vol; ld[nl]=-1.0; lp[nl]=lz; nl += 1
            i += 1

        direction = -direction
        # wait entry_every_n bars before next entry
        i += entry_every_n - 1

    return total, nc, nt, nm, sl/max(nt+nm,1)

# warmup
_ = sim_random_zr(op[:2000], hi[:2000], lo[:2000], cl[:2000], PIP, SPREAD, 1.25, MAX_LEGS, 1, 30.0, 7.5)
print("JIT compiled")

print(f"\n{'entry_N':>8} {'ZW':>4} {'tgt_f':>6} | {'pips':>9} {'p/day':>7} {'c/day':>7} {'ppc':>7} {'ml%':>6} {'avgl':>5}")
print("─"*72)

for entry_n in [1, 3, 6, 12, 24, 48]:
    for zw in [20.0, 30.0, 40.0, 56.0]:
        for tgt_f in [0.25, 0.50, 1.00]:
            tgt = zw * tgt_f
            tp, nc, nt, nm, avgl = sim_random_zr(op, hi, lo, cl, PIP, SPREAD, 1.25, MAX_LEGS, entry_n, zw, tgt)
            ppd = tp / span_days; cpd = nc / span_days
            ppc = tp / max(nc, 1); ml_pct = nm / max(nc, 1) * 100
            flag = " 🟢" if tp > 0 else " 🔴"
            print(f"{entry_n:>8} {zw:>4.0f} {tgt_f:>6.2f} | {tp:>9.0f} {ppd:>7.1f} {cpd:>7.2f} {ppc:>7.1f} {ml_pct:>5.1f}% {avgl:>5.1f}{flag}")
    print()
