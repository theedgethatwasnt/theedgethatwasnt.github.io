#!/usr/bin/env python3
"""
Counter-trend run fade with STRUCTURAL SL/TP at varying risk:reward.
Entry: counter-trend N-bar run, fade (trade with higher-TF SMA50 trend). SL = the run's extreme
(last bar's high for a short / low for a long) + buffer; TP = RR x SL-distance in the fade
direction. Walk H1 bars to first touch (R2 within-bar sequencing), time-capped. Net per-pair
spread. Sweep RR. 12 pairs, IS/OOS. (On a near-directionless price, RR is a martingale — WR
moves, expectancy ~ -spread; this checks whether structure + RR escapes that.)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import duckdb

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6; N=4; SMA=50; MAXHOLD=24; BUF=0.15  # SL buffer as fraction of bar range


def load(con,pair):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,open,high,low,close FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    return df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()


def main():
    con=duckdb.connect()
    RRs=[0.5,1.0,1.5,2.0,3.0]
    # collect per-event: entry, sl_price, dir, is, then resolve per-RR
    ev=[]
    for pair,(pip,sp) in PAIRS.items():
        r=load(con,pair); o=r.open.values;h=r.high.values;l=r.low.values;c=r.close.values;n=len(c)
        sma=pd.Series(c).rolling(SMA).mean().values
        sign=np.where(c>=o,1,-1); runlen=np.ones(n,int)
        for i in range(1,n): runlen[i]=runlen[i-1]+1 if sign[i]==sign[i-1] else 1
        is_end=int(n*IS_FRAC)
        for t in range(max(N,SMA), n-MAXHOLD-1):
            if runlen[t]>=N and (runlen[t]>runlen[t-1] or runlen[t]==N) and not np.isnan(sma[t]):
                trend=1 if c[t]>sma[t] else -1
                if sign[t]==trend: continue
                d=trend; entry=c[t]; rng=h[t]-l[t]
                if d==-1: sl=h[t]+BUF*rng          # short: stop above run high
                else:     sl=l[t]-BUF*rng          # long: stop below run low
                sld=abs(sl-entry)/pip
                if sld<1: continue
                ev.append((pair,pip,sp,t,entry,sl,d,sld,t<is_end,h,l))
    con.close()
    print(f"Counter-trend fade + structural SL/TP. {len(ev)} events. Sweep RR (TP = RR x SL).\n")
    print(f"  {'RR':>4}{'IS_exp':>9}{'OOS_exp':>9}{'IS_WR':>7}{'OOS_WR':>8}{'pairs+':>8}")
    # pre-extract arrays per pair for the walk
    for RR in RRs:
        ispnl=[];oospnl=[];per={p:[] for p in PAIRS}
        for (pair,pip,sp,t,entry,sl,d,sld,isf,h,l) in ev:
            tp = entry - d*RR*sld*pip*(-1) if False else entry + d*RR*sld*pip  # TP in trade dir
            # d=-1 short: tp=entry - RR*sld*pip ; d=+1 long: tp=entry + RR*sld*pip
            res=None
            for j in range(t+1, t+1+MAXHOLD):
                hi=h[j]; lo=l[j]
                # within-bar order: for short, adverse=high(SL) then favorable=low(TP) on up bars
                if d==-1:
                    if hi>=sl: res=-sld-sp; break
                    if lo<=tp: res=RR*sld-sp; break
                else:
                    if lo<=sl: res=-sld-sp; break
                    if hi>=tp: res=RR*sld-sp; break
            if res is None:
                res = d*(0)  # time cap: approximate flat exit at last close ~ 0 move (rare); use -sp
                res = -sp
            (ispnl if isf else oospnl).append(res); per[pair].append((isf,res))
        ispnl=np.array(ispnl);oospnl=np.array(oospnl)
        npos=sum(1 for p in PAIRS if [x[1] for x in per[p] if not x[0]] and np.mean([x[1] for x in per[p] if not x[0]])>0)
        print(f"  {RR:>4}{ispnl.mean():>+9.2f}{oospnl.mean():>+9.2f}{(ispnl>0).mean()*100:>6.0f}%{(oospnl>0).mean()*100:>7.0f}%{npos:>6}/12")
    print("\n  +IS +OOS across RR => structural SL/TP escapes the wall. Else RR is a martingale on ~directionless price.")


if __name__=="__main__":
    main()
