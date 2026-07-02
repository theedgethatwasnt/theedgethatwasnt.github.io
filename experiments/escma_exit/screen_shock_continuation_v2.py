"""v2 — EARLY flag-breakout: shock -> first pullback -> resume (break of running extreme) -> ride.
Enters DURING the cascade, not after the full leg. Net spread, multi-pair, IS/OOS."""
import gc, numpy as np, pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
ROOT=Path("/path/to/projects/fx-core"); S5=ROOT/"data/s5_ohlc"
PAIRS=["USD_JPY","GBP_JPY","EUR_JPY","AUD_JPY","EUR_USD","GBP_USD"]
THR,ZW,MADW=2.5,6,2048; E=48; PB_MIN=2.0; HORIZONS=[60,120,180]; IS_FRAC=0.70
def zc(close,pip):
    n=len(close);vel=np.empty(n);vel[:ZW]=0.0;vel[ZW:]=(close[ZW:]-close[:n-ZW])/pip
    vs=pd.Series(vel);rm=vs.rolling(MADW,min_periods=50).median()
    rmad=(vs-rm).abs().rolling(MADW,min_periods=50).median()
    return ((vs-rm)/(1.4826*rmad.clip(lower=1e-6))).fillna(0).values.astype(np.float64),vel
def proc(pair,H):
    pip=0.01 if "JPY" in pair else 0.0001
    t=pd.read_parquet(S5/f"{pair}_S5_BA.parquet",columns=["close","bid_c","ask_c"])
    close=t["close"].to_numpy().astype(np.float64);bid=t["bid_c"].to_numpy().astype(np.float64);ask=t["ask_c"].to_numpy().astype(np.float64)
    n=len(close);z,vel=zc(close,pip);is_end=int(n*IS_FRAC);cd=0;recs=[]
    for ti in range(MADW+100,n-E-H-2):
        if cd>0: cd-=1;continue
        if abs(z[ti])<=THR: continue
        cd=E
        d=1 if vel[ti]>0 else -1
        run=close[ti];pulled=False;pbx=close[ti];ent=None
        for j in range(ti+1,ti+E):
            if d==-1:
                if close[j]<run:
                    run=close[j]
                    if pulled: ent=j;break
                else:
                    if close[j]-run>=PB_MIN*pip: pulled=True;pbx=max(pbx,close[j])
            else:
                if close[j]>run:
                    run=close[j]
                    if pulled: ent=j;break
                else:
                    if run-close[j]>=PB_MIN*pip: pulled=True;pbx=min(pbx,close[j])
        if ent is None or ent+H>=n: continue
        leg=abs(close[ti]-run)/pip
        depth=abs(pbx-run)/(leg*pip) if leg>0 else 1.0
        pnl=(bid[ent]-ask[ent+H])/pip if d==-1 else (bid[ent+H]-ask[ent])/pip
        recs.append((depth,pnl,ent<is_end))
    del t,close,bid,ask,z,vel;gc.collect();return recs
for H in HORIZONS:
    print(f"\n=== EARLY flag-breakout  H=+{H*5//60}min  E={E*5//60}min  ({len(PAIRS)} pairs, net spread) ===")
    rec=[]
    for p in PAIRS: rec+=[(p,)+r for r in proc(p,H)]
    D=pd.DataFrame(rec,columns=["pair","depth","pnl","is_"]);oos=D[~D["is_"]]
    for nm,m in [("shallow<0.33",oos["depth"]<0.33),("mid",((oos["depth"]>=0.33)&(oos["depth"]<0.66))),("deep>0.66",oos["depth"]>=0.66),("ALL",oos["depth"]>=-9)]:
        s=oos[m]
        if len(s)<20: print(f"  {nm:12s} n={len(s)} few");continue
        print(f"  {nm:12s} n={len(s):>5d} mean={s['pnl'].mean():>+7.3f} WR={100*(s['pnl']>0).mean():>3.0f}% {'🟢' if s['pnl'].mean()>0 else '🔴'}")
    print("  shallow per-pair:", "  ".join(f"{p}:{oos[(oos.pair==p)&(oos.depth<0.33)]['pnl'].mean():+.2f}(n{len(oos[(oos.pair==p)&(oos.depth<0.33)])})" for p in PAIRS))
