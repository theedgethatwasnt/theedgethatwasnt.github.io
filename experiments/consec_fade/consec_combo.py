#!/usr/bin/env python3
"""
Top combination: COUNTER-trend run (best cell, -0.68 OOS) + EXTENSION filter, fade.
At each N-bar H1 run that is COUNTER to the higher-TF trend (SMA50), add features:
  travel_atr = |net run move| / ATR        (climactic big pop vs small)
  reached    = (close - SMA50) / ATR        (how far the pop is from the mean; ~0 = into resistance)
Fade (= trade with the higher-TF trend). IS-select the buckets, OOS-confirm, per-pair breadth.
12 pairs, 6-bar hold, net per-pair spread. If a slice is +IS +OOS broad -> needle (layer M1 next).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import duckdb

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6; N=4; HOLD=6; SMA=50


def load(con,pair):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,open,high,low,close FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    return df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()


def atr(h,l,c,n=14):
    pc=np.empty_like(c);pc[0]=c[0];pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values


def main():
    con=duckdb.connect()
    P={'pnl':[], 'is':[], 'travel':[], 'reached':[], 'pair':[]}
    for pair,(pip,sp) in PAIRS.items():
        r=load(con,pair); o=r.open.values;h=r.high.values;l=r.low.values;c=r.close.values;n=len(c)
        sma=pd.Series(c).rolling(SMA).mean().values; a=atr(h,l,c,14)
        sign=np.where(c>=o,1,-1); runlen=np.ones(n,int)
        for i in range(1,n): runlen[i]=runlen[i-1]+1 if sign[i]==sign[i-1] else 1
        is_end=int(n*IS_FRAC)
        for t in range(max(N,SMA), n-HOLD):
            if runlen[t]>=N and (runlen[t]>runlen[t-1] or runlen[t]==N) and not np.isnan(sma[t]) and a[t]>0:
                trend=1 if c[t]>sma[t] else -1
                if sign[t]==trend: continue                 # counter-trend runs only
                d=trend                                      # fade = trade with higher-TF trend
                P['pnl'].append(d*(c[t+HOLD]-c[t])/pip - sp); P['is'].append(t<is_end)
                P['travel'].append(abs(c[t]-o[t-N+1])/a[t]); P['reached'].append((c[t]-sma[t])/a[t]); P['pair'].append(pair)
    con.close()
    pnl=np.array(P['pnl']);isf=np.array(P['is']);pr=np.array(P['pair'])
    print(f"COUNTER-trend run + extension, FADE (N>={N},hold={HOLD}). {len(pnl)} events (IS {isf.sum()}/OOS {(~isf).sum()}).")
    print(f"  baseline counter-trend fade: IS {pnl[isf].mean():+.2f}p  OOS {pnl[~isf].mean():+.2f}p\n")
    def cut(name,feat):
        feat=np.array(feat); q=np.quantile(feat[isf],[.33,.66]); b=np.digitize(feat,q)
        print(f"  === by {name} (IS-thresholded) ===")
        for k,lab in [(0,'low'),(1,'mid'),(2,'high')]:
            mi=(b==k)&isf; mo=(b==k)&~isf
            npos=sum(1 for p in PAIRS if ((pr==p)&(b==k)&~isf).sum()>20 and pnl[(pr==p)&(b==k)&~isf].mean()>0)
            print(f"    {lab:>4}: IS {pnl[mi].mean():+6.2f}  OOS {pnl[mo].mean():+6.2f} (n {mo.sum()}, WR {(pnl[mo]>0).mean()*100:.0f}%, pairs+ {npos}/12)")
        print()
    cut("travel/ATR (climactic pop)", P['travel'])
    cut("reached SMA (close-SMA)/ATR — abs", np.abs(P['reached']))
    # combo: high-travel AND near-SMA
    tv=np.array(P['travel']); re=np.abs(np.array(P['reached']))
    thi=np.quantile(tv[isf],.66); rlo=np.quantile(re[isf],.5)
    cm=(tv>=thi)&(re<=rlo)
    print(f"  === combo: big travel (top tercile) AND pop reached the mean (close near SMA) ===")
    print(f"    IS {pnl[cm&isf].mean():+.2f}p  OOS {pnl[cm&~isf].mean():+.2f}p (n {int((cm&~isf).sum())}, WR {(pnl[cm&~isf]>0).mean()*100:.0f}%)")
    print("\n  Needle if a slice is +IS +OOS with breadth. Else close consec-fade as same wall.")


if __name__=="__main__":
    main()
