#!/usr/bin/env python3
"""
Consecutive run + higher-TF trend: is the FADE-vs-CONTINUE decision set by the bigger trend?
At each N-bar H1 run, higher-TF trend = sign(close - SMA50_H1) (~2-day). Measure forward
return (6-bar hold, net spread) for trading WITH the run (continue), AGAINST it (fade), and
WITH the higher-TF trend — split by whether the run is aligned with that trend or counter.
Hypothesis: with-trend run -> continue pays; counter-trend run -> fade pays; i.e. trade-with-
the-higher-TF-trend is the unifying rule. 12 pairs, IS/OOS.
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


def main():
    con=duckdb.connect()
    E={'rundir':[], 'trend':[], 'fwd':[], 'is':[]}
    for pair,(pip,sp) in PAIRS.items():
        r=load(con,pair); o=r.open.values;c=r.close.values;n=len(c)
        sma=pd.Series(c).rolling(SMA).mean().values
        sign=np.where(c>=o,1,-1); runlen=np.ones(n,int)
        for i in range(1,n): runlen[i]=runlen[i-1]+1 if sign[i]==sign[i-1] else 1
        is_end=int(n*IS_FRAC)
        for t in range(max(N,SMA), n-HOLD):
            if runlen[t]>=N and (runlen[t]>runlen[t-1] or runlen[t]==N) and not np.isnan(sma[t]):
                trend=1 if c[t]>sma[t] else -1
                E['rundir'].append(sign[t]); E['trend'].append(trend)
                E['fwd'].append((c[t+HOLD]-c[t])/pip)   # raw forward move (apply dir + spread later)
                E['is'].append(t<is_end)
    con.close()
    rd=np.array(E['rundir']);tr=np.array(E['trend']);fwd=np.array(E['fwd']);isf=np.array(E['is'])
    SPREAD=2.5  # approx avg, net applied uniformly for the comparison
    def stat(mask, d_arr):
        # d_arr: per-event trade direction; pnl = d*fwd - spread
        m=mask
        pnl=d_arr[m]*fwd[m]-SPREAD
        im=isf[m]
        return pnl[im].mean(), pnl[~im].mean(), len(pnl)
    print(f"Consecutive run (N>={N},hold={HOLD}) × higher-TF trend (SMA{SMA}). {len(fwd)} events. ~spread {SPREAD}p.\n")
    allm=np.ones(len(fwd),bool)
    for lab,d in [("continue (with run)", rd), ("fade (against run)", -rd), ("WITH higher-TF trend", tr)]:
        i,o,nn=stat(allm,d); print(f"  {lab:<22} IS {i:+6.2f}  OOS {o:+6.2f}  (n {nn})")
    print()
    aligned = rd==tr
    print("  === split: run aligned-with vs counter-to the higher-TF trend ===")
    for amask,alab in [(aligned,'run WITH trend'),(~aligned,'run COUNTER trend')]:
        for lab,d in [("continue", rd), ("fade", -rd)]:
            i,o,nn=stat(amask,d); print(f"    {alab:<18} {lab:<9} IS {i:+6.2f}  OOS {o:+6.2f}  (n {nn})")
    print("\n  Edge if 'WITH higher-TF trend' (or a with/counter cell) is positive IS AND OOS net spread.")


if __name__=="__main__":
    main()
