#!/usr/bin/env python3
"""
CONSECUTIVE-BAR FADE — counter-trend exhaustion. After N consecutive same-direction H1 bars,
fade (short after N up, long after N down). Tests whether a short run signals reversal (the
user's idea) or continuation (the big-bar trap). H1, 12 pairs, mid + fixed spread, IS/OOS.
Then condition on hour-of-day and tick-volume to look for a 'needle' (IS-select, OOS-confirm).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import duckdb

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6


def load(con,pair):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,open,high,low,close,volume FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    return df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()


def main():
    con=duckdb.connect()
    # gather all fade events across pairs: (forward_pips_net, is_is, N_runlen, hod, vrel, pair)
    print("CONSECUTIVE-BAR FADE — after N same-dir H1 bars, fade. Net spread, 12 pairs, IS/OOS.\n")
    # also test CONTINUATION (go WITH the run) as the control, to see which way it leans
    for HOLD in (1, 3, 6):
        print(f"===== hold = {HOLD} H1 bar(s) =====")
        print(f"  {'N':>2}{'dir':>6}{'IS_exp':>9}{'OOS_exp':>9}{'IS_WR':>7}{'OOS_WR':>8}{'n':>7}{'pairs+':>8}")
        for N in (2,3,4,5):
            for fade,lab in [(1,'FADE'),(-1,'WITH')]:
                ie=[];oe=[];per={}
                for pair,(pip,sp) in PAIRS.items():
                    r=load(con,pair)
                    o=r.open.values;c=r.close.values;n=len(c)
                    bull=(c>=o).astype(int); sign=np.where(c>=o,1,-1)
                    # run length of consecutive same-direction bars ending at t
                    runlen=np.ones(n,int);
                    for i in range(1,n):
                        runlen[i]=runlen[i-1]+1 if sign[i]==sign[i-1] else 1
                    is_end=int(n*IS_FRAC); pe=[]
                    for t in range(N, n-HOLD):
                        if runlen[t]==N:                       # exactly N (fresh exhaustion point)
                            d = -sign[t]*fade if fade==1 else sign[t]   # fade: against run; with: along
                            # for 'WITH' (fade=-1) d=sign[t]; for FADE d=-sign[t]
                            d = -sign[t] if fade==1 else sign[t]
                            pnl=d*(c[t+HOLD]-c[t])/pip - sp
                            (ie if t<is_end else oe).append(pnl); pe.append((t<is_end,pnl))
                    per[pair]=pe
                isn=np.array(ie); oon=np.array(oe)
                pos=sum(1 for p in per.values() if [x[1] for x in p if not x[0]] and np.mean([x[1] for x in p if not x[0]])>0)
                print(f"  {N:>2}{lab:>6}{isn.mean():>+9.2f}{oon.mean():>+9.2f}{(isn>0).mean()*100:>6.0f}%{(oon>0).mean()*100:>7.0f}%{len(isn)+len(oon):>7}{pos:>6}/12")
        print()
    con.close()
    print("  FADE>0 net spread IS+OOS multi-pair => exhaustion edge. If FADE<0 and WITH>0 => runs continue (trap).")


if __name__=="__main__":
    main()
