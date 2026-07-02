#!/usr/bin/env python3
"""
TOUCH-LADDER — do breakouts through heavily-REHEARSED levels, with volume, continue?
User hypothesis: resting orders stack at price levels (a support/resistance ladder); the
NUMBER OF TOUCHES ("rehearsals") at a level proxies the resting liquidity / how tested it is;
breaking a heavily-rehearsed level WITH volume means those orders were consumed (real intent)
=> continuation, while a thin-volume break is a fake-out that reverts.

Test (causal, multi-pair, multi-TF, net spread, IS/OOS):
  - Resistance R = rolling max(high, L) [support S = rolling min(low, L)], excluding current bar.
  - touches@level = # of prior bars in the window whose high came within EPS pips of R
    (resp. low within EPS of S) — the "rehearsals".
  - break event: close crosses above R (long) / below S (short), having been on the other side.
  - break_vol = bar tick-volume / trailing-mean(L).
  - forward = break-direction return over hold H, in pips, minus spread.
  - Bucket forward by touches-tercile × break_vol-tercile (IS thresholds, R5); IS vs OOS.
Hypothesis predicts: high-touch × high-vol breaks continue (positive, stationary).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import duckdb

PROJECT=Path("/path/to/projects/fx-core"); DATA=PROJECT/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
TF_RULE={"M15":"15min","H1":"1h","H4":"4h"}
IS_FRAC=0.6


def load_ohlcv(con,pair,rule):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,open,high,low,close,volume FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    r=df.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    return r


def collect(TF,L,H,EPS_PIPS):
    con=duckdb.connect(); rule=TF_RULE[TF]
    T={'touch':[], 'vol':[], 'fwd':[], 'is':[]}
    for pair,(pip,sp) in PAIRS.items():
        r=load_ohlcv(con,pair,rule)
        if len(r)<L+H+50: continue
        o=r.open.values; h=r.high.values; l=r.low.values; c=r.close.values; v=r.volume.values
        n=len(c); eps=EPS_PIPS*pip
        # rolling resistance/support over prior L bars (shifted, excludes current)
        Rs=pd.Series(h).rolling(L).max().shift(1).values
        Ss=pd.Series(l).rolling(L).min().shift(1).values
        vmean=pd.Series(v).rolling(L).mean().shift(1).values
        is_end=int(n*IS_FRAC)
        last_break=-10
        for t in range(L+1,n-H):
            if np.isnan(Rs[t]) or np.isnan(vmean[t]) or vmean[t]<=0: continue
            if t-last_break < H: continue                       # non-overlapping-ish
            # long break: prior close below R, current close above
            up = c[t-1]<=Rs[t] and c[t]>Rs[t]
            dn = c[t-1]>=Ss[t] and c[t]<Ss[t]
            if not (up or dn): continue
            if up:
                lvl=Rs[t]; touches=int(np.sum(h[t-L:t] >= lvl-eps)); d=1
            else:
                lvl=Ss[t]; touches=int(np.sum(l[t-L:t] <= lvl+eps)); d=-1
            bvol=v[t]/vmean[t]
            fwd=d*(c[t+H]-c[t])/pip - sp
            T['touch'].append(touches); T['vol'].append(bvol); T['fwd'].append(fwd); T['is'].append(t<is_end)
            last_break=t
    con.close()
    return (np.array(T['touch']),np.array(T['vol']),np.array(T['fwd']),np.array(T['is']))


def run(TF,L,H,EPS):
    touch,vol,fwd,ism=collect(TF,L,H,EPS)
    if len(fwd)<200:
        print(f"  {TF} L{L} H{H}: too few breaks ({len(fwd)})"); return
    print(f"\n===== {TF} L={L} H={H} eps={EPS}p — {len(fwd)} breaks (IS {ism.sum()}/OOS {(~ism).sum()}) =====")
    # marginal: correlation of touches and vol with forward continuation (IS)
    def ic(a,b):
        a=a[ism]; b=b[ism]
        if len(a)<50: return float('nan')
        ra=pd.Series(a).rank().values; rb=pd.Series(b).rank().values; return np.corrcoef(ra,rb)[0,1]
    print(f"  IS rank-corr: touches→fwd={ic(touch,fwd):+.3f}  vol→fwd={ic(vol,fwd):+.3f}  (positive => hypothesis)")
    tq=np.quantile(touch[ism],[.33,.66]); vq=np.quantile(vol[ism],[.33,.66])
    tb=np.digitize(touch,tq); vb=np.digitize(vol,vq)
    print(f"  forward continuation (pips, net spread) [IS // OOS], touch row × vol col:")
    print(f"  {'':<12}{'volLo':>16}{'volMid':>16}{'volHi':>16}")
    for tr in range(3):
        cells=[]
        for vc in range(3):
            mi=(tb==tr)&(vb==vc)&ism; mo=(tb==tr)&(vb==vc)&~ism
            ie=fwd[mi].mean() if mi.sum()>15 else float('nan'); oe=fwd[mo].mean() if mo.sum()>15 else float('nan')
            cells.append(f"{ie:+6.1f}//{oe:+6.1f}")
        lab=['touchLo','touchMid','touchHi'][tr]
        print(f"  {lab:<12}"+"".join(f"{c:>16}" for c in cells))
    # headline: high-touch × high-vol
    mi=(tb==2)&(vb==2)&ism; mo=(tb==2)&(vb==2)&~ism
    if mi.sum()>15 and mo.sum()>15:
        print(f"  >>> HIGH-touch × HIGH-vol break: IS {fwd[mi].mean():+.2f}p (n{mi.sum()}, WR{(fwd[mi]>0).mean()*100:.0f}%) | "
              f"OOS {fwd[mo].mean():+.2f}p (n{mo.sum()}, WR{(fwd[mo]>0).mean()*100:.0f}%)")


if __name__=="__main__":
    print("TOUCH-LADDER — breakout continuation vs rehearsals(touches) × break-volume. Net spread, IS/OOS, 12 pairs.")
    for TF,L,H,EPS in [("M15",48,24,5),("H1",24,12,5),("H1",48,24,8),("H4",20,12,10),("H4",30,20,15)]:
        run(TF,L,H,EPS)
    print("\n  LEVER if high-touch×high-vol breaks continue positively IS AND OOS (and corr signs positive).")
