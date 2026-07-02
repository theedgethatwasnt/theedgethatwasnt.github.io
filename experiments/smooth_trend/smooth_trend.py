#!/usr/bin/env python3
"""
SMOOTH TREND / declining-ATR grind — from the user's GBP_JPY charts: a smooth uptrend (SMA3
hugging SMA14, price laddering up) runs while ATR CONTRACTS; the move turns choppy when ATR
EXPANDS. Hypothesis: enter a trend that is grinding with contracting ATR, hold ONE position
(pay spread once), exit when ATR expands (chop) or the SMA stack flips.

Test (H1, 12 pairs, net spread, IS/OOS): compare the ATR-gated grind vs an UNGATED trend
baseline (same SMA entry, exit only on flip/cap) to isolate the value of the declining-ATR gate
+ expansion exit. Per-pair + portfolio + temporal WF.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb
import duckdb

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6
RISE=6; ASLOPE=6; CAP=48; EXPAND=1.2


def load(con,pair):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,high,low,close FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    return df.resample("1h").agg({"high":"max","low":"min","close":"last"}).dropna()


def atr(h,l,c,n=14):
    pc=np.empty_like(c); pc[0]=c[0]; pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values


@nb.njit(cache=True)
def sim(c, s3,s14, a, ama, pip, sp, gated, ts_idx, is_end):
    n=len(c); pos=0; entry=0.0; ebar=-1
    P=np.empty(n); E=np.empty(n,np.int64); k=0
    for i in range(max(RISE,ASLOPE,20),n):
        if pos==0:
            trend_up = s3[i]>s14[i] and s14[i]>s14[i-RISE]
            trend_dn = s3[i]<s14[i] and s14[i]<s14[i-RISE]
            gate = (a[i]<ama[i] and a[i]<a[i-ASLOPE]) if gated else True
            if trend_up and gate: pos=1; entry=c[i]; ebar=i
            elif trend_dn and gate: pos=-1; entry=c[i]; ebar=i
        else:
            flip = (pos==1 and s3[i]<s14[i]) or (pos==-1 and s3[i]>s14[i])
            expand = (a[i] > ama[i]*EXPAND) if gated else False
            cap = (i-ebar)>=CAP
            if flip or expand or cap:
                P[k]=(c[i]-entry)/pip*pos - sp; E[k]=ebar; k+=1; pos=0
    return P[:k], E[:k]


def run(gated):
    con=duckdb.connect()
    allnet=[]; allis=[]; per={}
    for pair,(pip,sp) in PAIRS.items():
        r=load(con,pair)
        if len(r)<500: continue
        c=r.close.values; h=r.high.values; l=r.low.values
        s3=pd.Series(c).rolling(3).mean().values; s14=pd.Series(c).rolling(14).mean().values
        a=atr(h,l,c,14); ama=pd.Series(a).rolling(20).mean().values
        s3=np.nan_to_num(s3); s14=np.nan_to_num(s14); ama=np.nan_to_num(ama,nan=1e18)
        is_end=int(len(c)*IS_FRAC)
        P,E=sim(c,s3,s14,a,ama,pip,sp,gated,0,is_end)
        ism=E<is_end
        per[pair]=(P[ism],P[~ism])
        allnet.append(P); allis.append(ism)
    con.close()
    return per


def report(name,per):
    isn=np.concatenate([v[0] for v in per.values()]); oosn=np.concatenate([v[1] for v in per.values()])
    def dd(x):
        if not len(x): return 0
        cum=x.cumsum(); return float((cum-np.maximum.accumulate(cum)).min())
    pos=sum(1 for v in per.values() if len(v[1]) and v[1].mean()>0)
    print(f"  {name:<16} IS exp {isn.mean():+.2f}p (n{len(isn)}) | OOS exp {oosn.mean():+.2f}p (n{len(oosn)}) "
          f"WR{(oosn>0).mean()*100:.0f}% OOS_DD{dd(oosn):.0f}  OOS pairs+ {pos}/{len(per)}")
    return oosn


def main():
    _c=np.zeros(60); sim(_c,_c,_c,_c+1,_c+1,.01,2.,True,0,30)
    print("SMOOTH TREND (declining-ATR grind) vs ungated trend. H1, 12 pairs, net spread, IS/OOS.")
    g=run(True); u=run(False)
    report("ATR-gated grind", g)
    report("ungated trend", u)
    # per-pair for the gated version
    print("\n  ATR-gated per-pair OOS: "+"  ".join(f"{p}:{(v[1].mean() if len(v[1]) else 0):+.1f}" for p,v in g.items()))
    print("\n  Gate adds value if ATR-gated OOS exp > ungated AND broad+stationary. Trend-follow prior: usually fails net spread.")


if __name__=="__main__":
    main()
