"""
Stops/TP sweep on the structural fade (H1 S/R, kD=0.7, hi-vol). How much of the ~+1.1p gross
reversion can the exit capture? Sweep trail kT and TP kTP (ATR units; kTP=0 -> trail only).
Reuses the validated scaled build/run (real per-bar spread, IS/OOS 60-40, 12 pairs).
"""
import sys, gc
import numpy as np
sys.path.insert(0,"/path/to/projects/fx-core/research/experiments/spread_band_random")
from validate_structural_fade_scaled import build, run, PAIRS, M
RNG=np.random.default_rng(17)
KD=0.7
GRID=[(kt,ktp) for kt in (1.0,1.5,2.0,2.5,3.0) for ktp in (0.0,0.5,1.0,1.5)]

store={g:[] for g in GRID}
for pair in PAIRS:
    try: d=build(pair)
    except Exception as e: print(f"  skip {pair}: {e}",flush=True); continue
    for (kt,ktp) in GRID:
        pnl,grs,tss,isis=run(d["opn"],d["high"],d["low"],d["close"],d["sp"],d["regime"],d["vrel"],
                             d["actH"],d["actL"],d["atr"],d["ts"],KD,kt,ktp,1,d["calm_thr"],M,d["is_cut"],d["pipv"])
        store[(kt,ktp)].append((pair,pnl,grs,tss,isis))
    print(f"  {pair}: done",flush=True); del d; gc.collect()

print("\nStops/TP sweep — H1 structural fade, kD=0.7, hi-vol. 12 pairs, real spread, OOS.")
print(f"{'trail':>5} {'TP':>5} | {'n':>7} {'OOS net':>8} {'OOS grs':>8} {'net WR':>6} {'grsMC':>6} | {'pairs+ net/grs':>14}")
print("-"*80)
best=None
for g in GRID:
    P=np.concatenate([x[1] for x in store[g]]); G=np.concatenate([x[2] for x in store[g]])
    I=np.concatenate([x[4] for x in store[g]]).astype(bool)
    oos=P[~I]; oosg=G[~I]
    bG=np.array([RNG.choice(oosg,len(oosg),replace=True).mean() for _ in range(1200)])
    npos=ngpos=npairs=0
    for (pr,pn,gr,ts,ii) in store[g]:
        ii=ii.astype(bool); o=pn[~ii]; og=gr[~ii]
        if len(o)>=20: npairs+=1; npos+=o.mean()>0; ngpos+=og.mean()>0
    tplbl = "none" if g[1]==0 else f"{g[1]:.1f}"
    print(f"{g[0]:>5.1f} {tplbl:>5} | {len(P):>7,} {oos.mean():>+8.3f} {oosg.mean():>+8.3f} "
          f"{100*(oos>0).mean():>5.1f}% {(bG<=0).mean():>6.3f} | net{npos}/grs{ngpos}")
    if best is None or oosg.mean()>best[1]: best=(g,oosg.mean(),oos.mean())
print(f"\nMax-gross exit: trail={best[0][0]} TP={best[0][1]} -> gross {best[1]:+.3f}  net {best[2]:+.3f}")
print("Ceiling check: if max gross still < per-pair spread, no exit reaches net+ on retail.")
