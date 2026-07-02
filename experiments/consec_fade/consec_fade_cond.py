#!/usr/bin/env python3
"""
Consecutive-bar FADE — the 'needle' search. Core (N>=4 H1 up/down run, fade, 6-bar hold) is
just below the spread floor. Condition on hour-of-day (UTC) and tick-volume to see if a slice
clears it. Discipline: thresholds/buckets read on IS, reported OOS as the sealed test, multi-pair.
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


def main():
    con=duckdb.connect()
    pnl=[];isf=[];hod=[];vrel=[]
    for pair,(pip,sp) in PAIRS.items():
        r=load(con,pair)
        o=r.open.values;c=r.close.values;v=r.volume.values;ts=r.index;n=len(c)
        sign=np.where(c>=o,1,-1); runlen=np.ones(n,int)
        for i in range(1,n): runlen[i]=runlen[i-1]+1 if sign[i]==sign[i-1] else 1
        vmean=pd.Series(v).rolling(24).mean().shift(1).values
        is_end=int(n*IS_FRAC)
        for t in range(max(N,25), n-HOLD):
            if runlen[t]>=N and (runlen[t]>runlen[t-1] or runlen[t]==N):
                if vmean[t] and vmean[t]>0:
                    d=-sign[t]
                    pnl.append(d*(c[t+HOLD]-c[t])/pip - sp); isf.append(t<is_end)
                    hod.append(ts[t].hour); vrel.append(v[t]/vmean[t])
    con.close()
    pnl=np.array(pnl);isf=np.array(isf);hod=np.array(hod);vrel=np.array(vrel)
    print(f"Consecutive-FADE needle search (N>={N}, hold={HOLD}). {len(pnl)} events (IS {isf.sum()}/OOS {(~isf).sum()}).")
    print(f"  baseline FADE: IS {pnl[isf].mean():+.2f}p  OOS {pnl[~isf].mean():+.2f}p\n")

    # 1) hour-of-day: IS exp per hour, then take IS-positive hours, confirm OOS
    print("  === by hour-of-day (UTC) — IS exp then OOS for IS-positive hours ===")
    good=[]
    rows=[]
    for h in range(24):
        mi=(hod==h)&isf; mo=(hod==h)&~isf
        if mi.sum()<50: continue
        ie=pnl[mi].mean(); oe=pnl[mo].mean() if mo.sum() else float('nan')
        rows.append((h,ie,oe,mi.sum()))
        if ie>0: good.append(h)
    for h,ie,oe,n_ in rows:
        mark=' *IS+' if ie>0 else ''
        print(f"    h{h:02d}  IS {ie:+6.2f}  OOS {oe:+6.2f}  (n_is {n_}){mark}")
    if good:
        gm=np.isin(hod,good)
        print(f"  >>> IS-positive hours {good}: OOS exp {pnl[gm&~isf].mean():+.2f}p (n {int((gm&~isf).sum())}, WR {(pnl[gm&~isf]>0).mean()*100:.0f}%)")

    # 2) volume tercile (IS thresholds)
    vt=np.quantile(vrel[isf],[.33,.66]); vb=np.digitize(vrel,vt)
    print("\n  === by trigger-bar volume tercile (IS-thresholded) ===")
    for k,lab in [(0,'loVol'),(1,'midVol'),(2,'hiVol')]:
        mi=(vb==k)&isf; mo=(vb==k)&~isf
        print(f"    {lab:>6}: IS {pnl[mi].mean():+.2f}p  OOS {pnl[mo].mean():+.2f}p (n {mo.sum()}, WR {(pnl[mo]>0).mean()*100:.0f}%)")
    print("\n  Needle if a session/volume slice is positive IS AND OOS with decent n. Else same spread wall.")


if __name__=="__main__":
    main()
