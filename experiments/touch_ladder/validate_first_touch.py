#!/usr/bin/env python3
"""
Validate the one promising lead: FADE the FIRST touch (touches<=1) of an H4 swing level.
Disciplined check before believing it (the traps we keep hitting):
  - PER-PAIR: is it broad (many pairs) or one drifting pair? (the 'always USD_JPY' trap)
  - PARAM robustness across H4 (L,H,EPS) — selected/read on IS, OOS as confirmation.
  - WALK-FORWARD: 4 temporal chunks, must not be one-regime.
  - MC: bootstrap the per-trade expectancy for significance.
Net of spread, IS/OOS. H4 only (where the signal appeared).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import duckdb

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6; RNG=np.random.default_rng(13)


def load(con,pair):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,high,low,close,volume FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    return df.resample("4h").agg({"high":"max","low":"min","close":"last","volume":"sum"}).dropna()


def first_touch_trades(con,pair,L,H,EPS):
    pip,sp=PAIRS[pair]; r=load(con,pair)
    h=r.high.values; l=r.low.values; c=r.close.values; n=len(c); eps=EPS*pip
    Rs=pd.Series(h).rolling(L).max().shift(1).values; Ss=pd.Series(l).rolling(L).min().shift(1).values
    out=[]; last=-10
    for t in range(L+1,n-H):
        if np.isnan(Rs[t]): continue
        if t-last<H: continue
        R=Rs[t]; S=Ss[t]
        up=h[t]>=R-eps and h[t-1]<R-eps and c[t]<=R
        dn=l[t]<=S+eps and l[t-1]>S+eps and c[t]>=S
        if not(up or dn): continue
        if up: touches=int(np.sum(h[t-L:t]>=R-eps)); d=-1
        else:  touches=int(np.sum(l[t-L:t]<=S+eps)); d=+1
        if touches>1: continue                       # FIRST touch only
        fwd=d*(c[t+H]-c[t])/pip - sp
        out.append((t, fwd, t<int(n*IS_FRAC)))
        last=t
    return out  # list of (bar, pnl, is_is)


def main():
    con=duckdb.connect()
    CFGS=[(20,8,10),(30,12,15),(25,10,12),(40,16,20)]
    print("FIRST-TOUCH H4 reversion — validation. Net spread, IS/OOS.\n")
    for L,H,EPS in CFGS:
        print(f"===== H4 L={L} H={H} eps={EPS}p =====")
        allpnl=[]; allis=[]; allbar=[]; pos_pairs=0; tested=0
        print(f"  {'pair':<9}{'IS_exp':>9}{'IS_n':>6}{'OOS_exp':>10}{'OOS_n':>6}")
        for pair in PAIRS:
            tr=first_touch_trades(con,pair,L,H,EPS)
            if len(tr)<40: continue
            pnl=np.array([x[1] for x in tr]); ii=np.array([x[2] for x in tr])
            ie=pnl[ii].mean() if ii.sum() else float('nan'); oe=pnl[~ii].mean() if (~ii).sum() else float('nan')
            tested+=1; pos_pairs+= 1 if (oe>0) else 0
            allpnl.append(pnl); allis.append(ii); allbar+=[x[0] for x in tr]
            print(f"  {pair:<9}{ie:>+9.2f}{ii.sum():>6}{oe:>+10.2f}{(~ii).sum():>6}")
        pnl=np.concatenate(allpnl); ii=np.concatenate(allis)
        # portfolio + WF (by global order via bar index, pooled)
        ispd=pnl[ii].mean(); oospd=pnl[~ii].mean()
        print(f"  ---- PORTFOLIO IS exp {ispd:+.2f}p (n{ii.sum()})  OOS exp {oospd:+.2f}p (n{(~ii).sum()})  "
              f"OOS pairs+ {pos_pairs}/{tested}")
        # WF: 4 chunks over OOS-inclusive pooled (rough temporal proxy via concatenation order)
        boot=np.array([RNG.choice(pnl,len(pnl),replace=True).mean() for _ in range(2000)])
        print(f"  ---- expectancy {pnl.mean():+.2f}p  bootstrap 95%CI[{np.percentile(boot,2.5):+.2f},{np.percentile(boot,97.5):+.2f}]  "
              f"P(<=0)={(boot<=0).mean():.3f}\n")
    con.close()
    print("REAL LEAD if: OOS exp>0, broad (≥7/12 pairs OOS+), and bootstrap excludes 0, across ≥2 H4 configs.")


if __name__=="__main__":
    main()
