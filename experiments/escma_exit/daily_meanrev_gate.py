"""Full gate on FADE-a-large-daily-bar: walk-forward folds + Monte-Carlo sign-shuffle.
Cooldown = hold length -> non-overlapping trades -> valid MC. 12 pairs."""
import pandas as pd, numpy as np, gc
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
M5=Path("/path/to/projects/fx-core/data/m5_ba")
PAIRS=["USD_JPY","EUR_USD","GBP_USD","AUD_USD","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","CHF_JPY","NZD_USD","EUR_GBP"]
THR=1.5; ZWIN=20; NFOLD=5; NMC=2000
def daily(pair):
    pip=0.01 if "JPY" in pair else 0.0001
    t=pd.read_parquet(M5/f"{pair}_M5_BA.parquet",columns=["timestamp","close","bid_c","ask_c"])
    t["timestamp"]=pd.to_datetime(t["timestamp"]); t=t.set_index("timestamp")
    d=t.resample("1D").agg(close=("close","last"),bid=("bid_c","last"),ask=("ask_c","last")).dropna()
    d["sp"]=(d["ask"]-d["bid"])/pip; d["ret"]=(d["close"]-d["close"].shift(1))/pip
    med=d["ret"].rolling(ZWIN).median(); mad=(d["ret"]-med).abs().rolling(ZWIN).median()
    d["z"]=(d["ret"]-med)/(1.4826*mad.replace(0,np.nan)); gc.collect(); return d,pip
def collect(H):
    pp={}
    for p in PAIRS:
        d,pip=daily(p); n=len(d)
        c=d["close"].values; z=d["z"].values; sp=d["sp"].values; ret=d["ret"].values
        cd=0; tr=[]
        for i in range(ZWIN+2,n-H):
            if cd>0: cd-=1; continue
            if abs(z[i])<=THR: continue
            cd=H; dirn=1 if ret[i]>0 else -1
            tr.append((i,-dirn*(c[i+H]-c[i])/pip - sp[i]))   # fade pnl, net spread
        pp[p]=(tr,n,n/1.0)
    return pp
for H in [3,5]:
    pp=collect(H)
    # full-sample aggregate p/d (per-pair pnl_sum / trading-days)
    agg=sum(sum(x[1] for x in tr)/(n) for tr,n,_ in pp.values())  # pnl per trading-day, summed across pairs
    ntot=sum(len(tr) for tr,_,_ in pp.values())
    allpnl=np.array([x[1] for tr,_,_ in pp.values() for x in tr])
    # WALK-FORWARD: NFOLD contiguous folds by bar index, per pair
    pairfold_pos=0; fold_agg=[0.0]*NFOLD; cells=0
    for p,(tr,n,_) in pp.items():
        edges=np.linspace(0,n,NFOLD+1).astype(int)
        for f in range(NFOLD):
            fpnl=[x[1] for x in tr if edges[f]<=x[0]<edges[f+1]]
            if len(fpnl)>=3:
                cells+=1; s=sum(fpnl)/((edges[f+1]-edges[f]))
                fold_agg[f]+=s; pairfold_pos+= (sum(fpnl)>0)
    # MC sign-shuffle (preserve per-pair days)
    rng=np.random.default_rng(7)
    real=agg; beat=0
    pair_arr=[(np.array([x[1] for x in tr]),n) for tr,n,_ in pp.values()]
    for _ in range(NMC):
        sh=sum((a*np.where(rng.random(len(a))>0.5,1.0,-1.0)).sum()/n for a,n in pair_arr)
        if sh>=real: beat+=1
    mc_p=beat/NMC
    print(f"\n{'='*78}\nFADE large daily bar, hold {H}d  | trades={ntot}  agg p/day(sum pairs)={agg:+.1f}")
    print(f"  WALK-FORWARD ({NFOLD} folds): pair×fold positive = {pairfold_pos}/{cells}  "
          f"({100*pairfold_pos/cells:.0f}%)")
    print(f"    per-fold aggregate p/day: " + "  ".join(f"F{i+1}:{v:+.1f}" for i,v in enumerate(fold_agg)))
    print(f"  MONTE CARLO sign-shuffle (n={NMC}): mc_p = {mc_p:.4f}  "
          f"{'🟢 SIGNIFICANT' if mc_p<0.05 else '🔴 not significant'}")
    print(f"  per-trade: mean={allpnl.mean():+.2f}p  WR={100*(allpnl>0).mean():.0f}%  median={np.median(allpnl):+.2f}p")
print("="*78)
