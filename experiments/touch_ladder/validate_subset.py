#!/usr/bin/env python3
"""
First-touch H4 reversion — the deploy gate, done honestly.
  1. Select the pair subset on IS ONLY (pairs with IS expectancy>0). R8: no OOS peeking.
  2. Confirm that IS-selected subset OOS (sealed).
  3. TRUE temporal walk-forward: split the full timeline into 3 calendar thirds; the subset
     must be positive in all three (not one regime).
  4. Bootstrap MC on the subset's pooled trades.
Two H4 configs for robustness. Net spread.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import duckdb

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6; RNG=np.random.default_rng(17)


def load(con,pair):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,high,low,close FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    return df.resample("4h").agg({"high":"max","low":"min","close":"last"}).dropna()


def trades(con,pair,L,H,EPS):
    pip,sp=PAIRS[pair]; r=load(con,pair)
    h=r.high.values; l=r.low.values; c=r.close.values; ts=r.index.values; n=len(c); eps=EPS*pip
    Rs=pd.Series(h).rolling(L).max().shift(1).values; Ss=pd.Series(l).rolling(L).min().shift(1).values
    out=[]; last=-10; is_cut=int(n*IS_FRAC)
    for t in range(L+1,n-H):
        if np.isnan(Rs[t]) or t-last<H: continue
        R=Rs[t]; S=Ss[t]
        up=h[t]>=R-eps and h[t-1]<R-eps and c[t]<=R
        dn=l[t]<=S+eps and l[t-1]>S+eps and c[t]>=S
        if not(up or dn): continue
        if up: tch=int(np.sum(h[t-L:t]>=R-eps)); d=-1
        else:  tch=int(np.sum(l[t-L:t]<=S+eps)); d=+1
        if tch>1: continue
        out.append((ts[t], d*(c[t+H]-c[t])/pip - sp, t<is_cut)); last=t
    return out


def main():
    con=duckdb.connect()
    for L,H,EPS in [(30,12,15),(25,10,12)]:
        print(f"\n===== H4 L={L} H={H} eps={EPS}p =====")
        per={}
        for p in PAIRS:
            tr=trades(con,p,L,H,EPS)
            if len(tr)>=40: per[p]=tr
        # 1. select on IS
        sub=[]
        for p,tr in per.items():
            ispnl=[x[1] for x in tr if x[2]]
            if len(ispnl)>=15 and np.mean(ispnl)>0: sub.append(p)
        print(f"  IS-selected pairs (IS exp>0): {sub}")
        # 2. OOS confirmation on the subset
        oos=np.array([x[1] for p in sub for x in per[p] if not x[2]])
        iss=np.array([x[1] for p in sub for x in per[p] if x[2]])
        print(f"  subset IS exp {iss.mean():+.2f}p (n{len(iss)}) | OOS exp {oos.mean():+.2f}p (n{len(oos)}) WR{(oos>0).mean()*100:.0f}%")
        # per-pair OOS of the selected subset
        print("  selected-pair OOS: "+"  ".join(
            f"{p}:{np.mean([x[1] for x in per[p] if not x[2]]):+.1f}" for p in sub))
        # 3. true temporal WF over ALL selected-subset trades, 3 calendar thirds
        allt=sorted([x for p in sub for x in per[p]], key=lambda z:z[0])
        tsarr=np.array([z[0] for z in allt]); pn=np.array([z[1] for z in allt])
        edges=[tsarr[0], tsarr[len(tsarr)//3], tsarr[2*len(tsarr)//3], tsarr[-1]]
        print("  temporal WF (3 calendar thirds, selected subset):")
        for j in range(3):
            m=(tsarr>=edges[j])&(tsarr<=edges[j+1])
            print(f"    third{j+1} {str(edges[j])[:10]}..{str(edges[j+1])[:10]}: exp {pn[m].mean():+.2f}p (n{m.sum()}, WR{(pn[m]>0).mean()*100:.0f}%)")
        # 4. MC on subset OOS
        boot=np.array([RNG.choice(oos,len(oos),replace=True).mean() for _ in range(3000)])
        print(f"  MC subset OOS: 95%CI[{np.percentile(boot,2.5):+.2f},{np.percentile(boot,97.5):+.2f}]  P(<=0)={(boot<=0).mean():.4f}")
    con.close()
    print("\n  DEPLOY-WORTHY if IS-selected subset is OOS+ , positive in all 3 calendar thirds, MC P(<=0)<0.05.")


if __name__=="__main__":
    main()
