#!/usr/bin/env python3
"""
Structural-location lens on the first-touch edge: fade the FIRST touch of the PRIOR-DAY
high/low (PDH/PDL) — the most-watched structural levels — with the same recipe that worked for
swing levels: low-volume filter + ATR target/SL, contrarian. H1 bars (intraday touch of a daily
level), held up to HCAP H1 bars, net per-pair spread, 12 pairs, IS/OOS. Reports all-touches and
the low-volume subset (the filter that made first-touch significant).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import duckdb

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6; EPS=8; HCAP=12; TGT=2.0; SL=2.0; VW=20


def loads(con,pair):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,open,high,low,close,volume FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    h1=df.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    d1=df.resample("1D").agg({"high":"max","low":"min"}).dropna()
    return h1,d1


def atr(h,l,c,n=14):
    pc=np.empty_like(c);pc[0]=c[0];pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values


def main():
    con=duckdb.connect()
    A={'pnl':[], 'is':[], 'vrel':[]}
    for pair,(pip,sp) in PAIRS.items():
        h1,d1=loads(con,pair)
        h=h1.high.values;l=h1.low.values;c=h1.close.values;v=h1.volume.values;ts=h1.index;n=len(c)
        a=atr(h,l,c,14); eps=EPS*pip
        vmean=pd.Series(v).rolling(VW).mean().shift(1).values
        # prior-day high/low mapped to each H1 bar (most recent COMPLETED day)
        dh=d1.high; dl=d1.low; ddates=d1.index.normalize()
        pdh=np.full(n,np.nan); pdl=np.full(n,np.nan)
        dmap={d:(dh.iloc[i-1],dl.iloc[i-1]) for i,d in enumerate(ddates) if i>=1}
        cur_day=None; touched_h=False; touched_l=False
        is_end=int(n*IS_FRAC)
        for t in range(VW+1, n-HCAP-1):
            day=ts[t].normalize()
            if day!=cur_day:
                cur_day=day; touched_h=False; touched_l=False
            if day not in dmap or a[t]<=0 or np.isnan(vmean[t]) or vmean[t]<=0: continue
            PH,PL=dmap[day]; vrel=v[t]/vmean[t]
            d=0; entry=c[t]
            if (not touched_h) and h[t]>=PH-eps:
                touched_h=True; d=-1                      # fade short off PDH
            elif (not touched_l) and l[t]<=PL+eps:
                touched_l=True; d=1                       # fade long off PDL
            if d==0: continue
            tp=entry+d*TGT*a[t]; slp=entry-d*SL*a[t]
            res=None
            for j in range(t+1,t+1+HCAP):
                if d==-1:
                    if h[j]>=slp: res=-(slp-entry)/pip-sp if False else (entry-slp)/pip-sp; res=-abs(slp-entry)/pip-sp; break
                    if l[j]<=tp: res=abs(entry-tp)/pip-sp; break
                else:
                    if l[j]<=slp: res=-abs(entry-slp)/pip-sp; break
                    if h[j]>=tp: res=abs(tp-entry)/pip-sp; break
            if res is None: res=d*(c[t+HCAP]-entry)/pip-sp
            A['pnl'].append(res); A['is'].append(t<is_end); A['vrel'].append(vrel)
    con.close()
    pnl=np.array(A['pnl']);isf=np.array(A['is']);vr=np.array(A['vrel'])
    print(f"PDH/PDL first-touch FADE + ATR SL/TP. {len(pnl)} events (IS {isf.sum()}/OOS {(~isf).sum()}).")
    print(f"  ALL touches:   IS {pnl[isf].mean():+.2f}p  OOS {pnl[~isf].mean():+.2f}p  WR {(pnl[~isf]>0).mean()*100:.0f}%")
    vmed=np.median(vr[isf])
    for lab,m in [("loVol(<=med)",vr<=vmed),("hiVol(>med)",vr>vmed)]:
        oo=pnl[m&~isf]; ii=pnl[m&isf]
        print(f"  {lab:<13}: IS {ii.mean():+.2f}p  OOS {oo.mean():+.2f}p (n {len(oo)}, WR {(oo>0).mean()*100:.0f}%)")
    print("\n  +IS +OOS (esp. loVol) => structural-level first-touch edge generalizes to PDH/PDL.")


if __name__=="__main__":
    main()
