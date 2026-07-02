#!/usr/bin/env python3
"""
Consecutive-FADE needle search, part 2 — the CHARACTER of the run (user's features):
  range_atr  = avg H1 bar range over the run / ATR14   (wide/climactic vs narrow/grind run)
  run_eff    = |net move over run| / sum(bar ranges)   (clean directional push vs choppy)
  closepos   = avg directional close-position in bar    (each bar closing at its extreme = strong)
Bucket the FADE (N>=4, 6-bar hold) by each, IS-thresholded, OOS-confirmed, 12 pairs.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import duckdb

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6; N=4; HOLD=6


def load(con,pair):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,open,high,low,close,volume FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    return df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()


def atr(h,l,c,n=14):
    pc=np.empty_like(c);pc[0]=c[0];pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values


def main():
    con=duckdb.connect()
    P={'pnl':[], 'is':[], 'rng':[], 'eff':[], 'cpos':[]}
    for pair,(pip,sp) in PAIRS.items():
        r=load(con,pair); o=r.open.values;h=r.high.values;l=r.low.values;c=r.close.values;n=len(c)
        a=atr(h,l,c,14); sign=np.where(c>=o,1,-1); runlen=np.ones(n,int)
        for i in range(1,n): runlen[i]=runlen[i-1]+1 if sign[i]==sign[i-1] else 1
        rng=h-l
        is_end=int(n*IS_FRAC)
        for t in range(max(N,20), n-HOLD):
            if runlen[t]>=N and (runlen[t]>runlen[t-1] or runlen[t]==N) and a[t]>0:
                w=slice(t-N+1, t+1)
                tot_rng=rng[w].sum()
                if tot_rng<=0: continue
                net=abs(c[t]-o[t-N+1])
                # directional close position per bar (up: (c-l)/rng ; down: (h-c)/rng), avg over run
                cp=np.mean([ (c[i]-l[i])/rng[i] if sign[i]>0 else (h[i]-c[i])/rng[i] for i in range(t-N+1,t+1) if rng[i]>0])
                d=-sign[t]
                P['pnl'].append(d*(c[t+HOLD]-c[t])/pip - sp); P['is'].append(t<is_end)
                P['rng'].append(rng[w].mean()/a[t]); P['eff'].append(net/tot_rng); P['cpos'].append(cp)
    con.close()
    pnl=np.array(P['pnl']);isf=np.array(P['is'])
    print(f"Consecutive-FADE character search (N>={N}, hold={HOLD}). {len(pnl)} events.")
    print(f"  baseline FADE: IS {pnl[isf].mean():+.2f}p  OOS {pnl[~isf].mean():+.2f}p\n")
    def cut(name,feat):
        feat=np.array(feat); q=np.quantile(feat[isf],[.33,.66]); b=np.digitize(feat,q)
        print(f"  === by {name} (IS-thresholded terciles) ===")
        for k,lab in [(0,'low'),(1,'mid'),(2,'high')]:
            mi=(b==k)&isf; mo=(b==k)&~isf
            print(f"    {lab:>4}: IS {pnl[mi].mean():+6.2f}  OOS {pnl[mo].mean():+6.2f} (n {mo.sum()}, WR {(pnl[mo]>0).mean()*100:.0f}%)")
        print()
    cut("range/ATR  (wide vs narrow run)", P['rng'])
    cut("run efficiency  (clean push vs choppy)", P['eff'])
    cut("close-position  (bars closing at extreme)", P['cpos'])
    print("  Needle if a tercile is positive IS AND OOS. Else the run's character doesn't lift it over spread.")


if __name__=="__main__":
    main()
