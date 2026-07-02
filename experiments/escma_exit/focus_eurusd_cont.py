"""EUR_USD-focused shock->shelf->breakout continuation, MIN_LEG swept (EUR_USD pip scale)."""
import numpy as np, pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
S5=Path("/path/to/projects/fx-core/data/s5_ohlc")
THR,PB,ZW,MADW=2.5,44,6,2048; W,BO=120,30; HORIZONS=[120,180]; IS_FRAC=0.70
pip=0.0001
t=pd.read_parquet(S5/"EUR_USD_S5_BA.parquet",columns=["close","bid_c","ask_c"])
close=t["close"].to_numpy().astype(np.float64);bid=t["bid_c"].to_numpy().astype(np.float64);ask=t["ask_c"].to_numpy().astype(np.float64)
n=len(close)
sp_med=np.median((ask-bid)/pip); print(f"EUR_USD bars={n}  median spread={sp_med:.2f}p")
vs=pd.Series(np.concatenate([[0.0]*ZW,(close[ZW:]-close[:n-ZW])/pip]))
rm=vs.rolling(MADW,min_periods=50).median();rmad=(vs-rm).abs().rolling(MADW,min_periods=50).median()
z=((vs-rm)/(1.4826*rmad.clip(lower=1e-6))).fillna(0).values; vel=vs.values
is_end=int(n*IS_FRAC)
def run(MIN_LEG,H):
    cd=0;recs=[]
    for ti in range(MADW+100,n-PB-W-BO-H-2):
        if cd>0: cd-=1;continue
        if abs(z[ti])<=THR: continue
        cd=(PB+W)//2; d=1 if vel[ti]>0 else -1
        seg=close[ti:ti+PB+1]
        ext=seg.min() if d==-1 else seg.max(); t_ext=ti+(int(seg.argmin()) if d==-1 else int(seg.argmax()))
        leg=abs(close[ti]-ext)/pip
        if leg<MIN_LEG: continue
        cw=close[t_ext:t_ext+W+1]
        shelf=cw.min() if d==-1 else cw.max(); bounce=cw.max() if d==-1 else cw.min()
        depth=(bounce-ext)/(leg*pip) if d==-1 else (ext-bounce)/(leg*pip)
        dec=t_ext+W; ent=None
        for j in range(dec,dec+BO+1):
            if (d==-1 and close[j]<shelf) or (d==1 and close[j]>shelf): ent=j;break
        if ent is None or ent+H>=n: continue
        pnl=(bid[ent]-ask[ent+H])/pip if d==-1 else (bid[ent+H]-ask[ent])/pip
        recs.append((depth,pnl,ent<is_end))
    return pd.DataFrame(recs,columns=["depth","pnl","is_"])
for H in HORIZONS:
    print(f"\n=== EUR_USD continuation H=+{H*5//60}min (net spread {sp_med:.1f}p) ===")
    for ML in [3,5,8]:
        D=run(ML,H); oos=D[~D["is_"]]
        if len(oos)<20: print(f"  MIN_LEG={ML}p: n={len(oos)} few"); continue
        sh=oos[oos.depth<0.33]; dp=oos[oos.depth>=0.66]
        print(f"  MIN_LEG={ML}p  n_oos={len(oos):>4d}  ALL={oos.pnl.mean():+.2f}  "
              f"shallow(n{len(sh)})={sh.pnl.mean() if len(sh)>=15 else float('nan'):+.2f}  "
              f"deep(n{len(dp)})={dp.pnl.mean() if len(dp)>=15 else float('nan'):+.2f}  "
              f"shallowWR={100*(sh.pnl>0).mean() if len(sh)>=15 else float('nan'):.0f}%")
