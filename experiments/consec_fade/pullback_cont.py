#!/usr/bin/env python3
"""
Pullback-CONTINUATION (user idea): after N up H1 bars + 1 correction (down) bar, enter LONG back
into the trend. SL = the correction bar's low; TP = the prior trend high (run high). Symmetric
for downtrends. Structural R:R set by the pullback depth. Walk H1 to first touch (R2 sequencing),
time-capped, net per-pair spread. 12 pairs, IS/OOS. Sweep N (trend length).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import duckdb

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6; MAXHOLD=24


def load(con,pair):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,open,high,low,close FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    return df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()


def main():
    con=duckdb.connect()
    print("Pullback-continuation: N up bars + 1 correction -> enter trend. SL=pullback extreme, TP=prior high.\n")
    print(f"  {'N':>2}{'IS_exp':>9}{'OOS_exp':>9}{'IS_WR':>7}{'OOS_WR':>8}{'avgRR':>7}{'n':>7}{'pairs+':>8}")
    for N in (2,3,4):
        per={p:[] for p in PAIRS}; rrs=[]
        for pair,(pip,sp) in PAIRS.items():
            r=load(con,pair); o=r.open.values;h=r.high.values;l=r.low.values;c=r.close.values;n=len(c)
            sign=np.where(c>=o,1,-1); runlen=np.ones(n,int)
            for i in range(1,n): runlen[i]=runlen[i-1]+1 if sign[i]==sign[i-1] else 1
            is_end=int(n*IS_FRAC)
            for t in range(N+1, n-MAXHOLD-1):
                # up-run of >=N ending at t-1, correction (down) at t
                if runlen[t-1]>=N and sign[t-1]==1 and sign[t]==-1:
                    d=1; entry=c[t]; sl=l[t]; tp=h[t-N:t].max()   # prior trend high
                elif runlen[t-1]>=N and sign[t-1]==-1 and sign[t]==1:
                    d=-1; entry=c[t]; sl=h[t]; tp=l[t-N:t].min()  # prior trend low
                else:
                    continue
                risk=abs(entry-sl)/pip; reward=abs(tp-entry)/pip
                if risk<1 or reward<1 or (d==1 and tp<=entry) or (d==-1 and tp>=entry): continue
                rrs.append(reward/risk)
                res=None
                for j in range(t+1, t+1+MAXHOLD):
                    if d==1:
                        if l[j]<=sl: res=-risk-sp; break
                        if h[j]>=tp: res=reward-sp; break
                    else:
                        if h[j]>=sl: res=-risk-sp; break
                        if l[j]<=tp: res=reward-sp; break
                if res is None: res=d*(c[t+MAXHOLD]-entry)/pip - sp
                per[pair].append((t<is_end,res))
        ip=[x[1] for p in per for x in per[p] if x[0]]; op=[x[1] for p in per for x in per[p] if not x[0]]
        ip=np.array(ip);op=np.array(op)
        npos=sum(1 for p in PAIRS if [x[1] for x in per[p] if not x[0]] and np.mean([x[1] for x in per[p] if not x[0]])>0)
        print(f"  {N:>2}{ip.mean():>+9.2f}{op.mean():>+9.2f}{(ip>0).mean()*100:>6.0f}%{(op>0).mean()*100:>7.0f}%{np.mean(rrs):>7.2f}{len(ip)+len(op):>7}{npos:>6}/12")
    con.close()
    print("\n  +IS +OOS broad => pullback-continuation edge. (Prior: SMA-pullback was 3/12 WF — mostly USD_JPY drift.)")


if __name__=="__main__":
    main()
