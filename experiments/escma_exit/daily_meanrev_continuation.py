"""Daily-scale: after a LARGE daily bar (|z|>THR), does the next H days continue or mean-revert?
Net of real spread (trivial at daily scale). 12 pairs, IS/OOS 70/30. Tests the long-horizon regime
where the project's only real signals have lived."""
import pandas as pd, numpy as np, gc
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
M5=Path("/path/to/projects/fx-core/data/m5_ba")
PAIRS=["USD_JPY","EUR_USD","GBP_USD","AUD_USD","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","CHF_JPY","NZD_USD","EUR_GBP"]
THR=1.5; HORIZONS=[1,2,3,5]; IS_FRAC=0.70; ZWIN=20
def daily(pair):
    pip=0.01 if "JPY" in pair else 0.0001
    t=pd.read_parquet(M5/f"{pair}_M5_BA.parquet",columns=["timestamp","close","bid_c","ask_c"])
    t["timestamp"]=pd.to_datetime(t["timestamp"]); t=t.set_index("timestamp")
    d=t.resample("1D").agg(close=("close","last"),bid=("bid_c","last"),ask=("ask_c","last")).dropna()
    d["sp"]=(d["ask"]-d["bid"])/pip
    d["ret"]=(d["close"]-d["close"].shift(1))/pip
    med=d["ret"].rolling(ZWIN).median(); mad=(d["ret"]-med).abs().rolling(ZWIN).median()
    d["z"]=(d["ret"]-med)/(1.4826*mad.replace(0,np.nan))
    del t; gc.collect(); return d, pip

# collect per-pair shock records
recs={H:[] for H in HORIZONS}
for p in PAIRS:
    d,pip=daily(p); n=len(d); cut=int(n*IS_FRAC)
    c=d["close"].values; z=d["z"].values; sp=d["sp"].values; ret=d["ret"].values
    for i in range(ZWIN+2, n):
        if not (abs(z[i])>THR): continue
        dirn=1 if ret[i]>0 else -1
        for H in HORIZONS:
            if i+H>=n: continue
            fwd=(c[i+H]-c[i])/pip
            recs[H].append((p, abs(z[i]), dirn*fwd, sp[i], i<cut, dirn))
print(f"{'='*92}\nDAILY large-bar -> forward outcome  (12 pairs, net spread, +1/2/3/5 days, IS/OOS)\n"
      f"continuation expectancy = E[dir*fwd] (>0 momentum, <0 mean-reversion)\n{'='*92}")
print(f"{'horizon':>7s} {'n_oos':>6s} {'cont E[dir*fwd]':>16s} {'cont net/td':>11s} {'mean-rev net/td':>15s} {'MR WR':>6s}  verdict")
for H in HORIZONS:
    D=pd.DataFrame(recs[H],columns=["pair","z","dfwd","sp","is_","dirn"]); oos=D[~D["is_"]]
    e=oos["dfwd"].mean()
    cont_net=(oos["dfwd"]-oos["sp"]).mean()
    mr_net=(-oos["dfwd"]-oos["sp"]).mean()
    mr_wr=100*((-oos["dfwd"])>0).mean()
    win="mean-reversion" if mr_net>cont_net else "continuation"
    best=max(cont_net,mr_net)
    flag="🟢 net+" if best>0 else "🔴"
    print(f"{H:>6d}d {len(oos):>6d} {e:>+16.2f} {cont_net:>+11.2f} {mr_net:>+15.2f} {mr_wr:>5.0f}%  {win} {flag}")
# per-pair mean-rev at H=3 (universal?)
print("\nper-pair MEAN-REVERSION net pnl/trade at H=3d (trade against the daily move):")
D=pd.DataFrame(recs[3],columns=["pair","z","dfwd","sp","is_","dirn"]); oos=D[~D["is_"]]
pos=0
for p in PAIRS:
    s=oos[oos.pair==p]
    if len(s)<10: continue
    mr=(-s["dfwd"]-s["sp"]).mean(); pos+= mr>0
    print(f"  {p:8s} n={len(s):>4d}  MR_net={mr:+.2f}p  WR={100*((-s['dfwd'])>0).mean():.0f}%")
print(f"  -> mean-reversion net-positive on {pos}/12 pairs at H=3d")
print("="*92)
