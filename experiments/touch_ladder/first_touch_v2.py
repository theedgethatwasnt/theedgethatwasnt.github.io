#!/usr/bin/env python3
"""
First-touch H4 reversion v2 — add a TARGET/SL exit (trim the fat tails that killed per-trade
significance) and fold in VOLUME, then re-run the gate. Fade the first touch (touches<=1) of a
rolling H4 swing level; exit at the first of: target = tgt*ATR (toward reversion), SL = sl*ATR
(beyond entry), or time cap H bars. Net spread. IS-select config, confirm OOS + temporal WF + MC.
Volume: tick-vol at the touch vs trailing mean — test all / low / high.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import duckdb

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6; L=25; EPS=12; VW=20; RNG=np.random.default_rng(23)


def load(con,pair):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,high,low,close,volume FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    return df.resample("4h").agg({"high":"max","low":"min","close":"last","volume":"sum"}).dropna()


def atr(h,l,c,n=14):
    pc=np.empty_like(c); pc[0]=c[0]; pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values


def trades(con,pair,tgt,sl,Hcap):
    pip,sp=PAIRS[pair]; r=load(con,pair)
    h=r.high.values; l=r.low.values; c=r.close.values; v=r.volume.values; ts=r.index.values; n=len(c); eps=EPS*pip
    a=atr(h,l,c,14)
    Rs=pd.Series(h).rolling(L).max().shift(1).values; Ss=pd.Series(l).rolling(L).min().shift(1).values
    vrel=v/pd.Series(v).rolling(VW).mean().shift(1).values
    out=[]; last=-10; is_cut=int(n*IS_FRAC)
    for t in range(L+1,n-Hcap-1):
        if np.isnan(Rs[t]) or a[t]<=0 or np.isnan(vrel[t]) or t-last<Hcap: continue
        R=Rs[t]; S=Ss[t]
        up=h[t]>=R-eps and h[t-1]<R-eps and c[t]<=R
        dn=l[t]<=S+eps and l[t-1]>S+eps and c[t]>=S
        if not(up or dn): continue
        if up: tch=int(np.sum(h[t-L:t]>=R-eps)); d=-1
        else:  tch=int(np.sum(l[t-L:t]<=S+eps)); d=+1
        if tch>1: continue
        entry=c[t]; ae=a[t]
        if d==-1: tp=entry-tgt*ae; slp=entry+sl*ae
        else:     tp=entry+tgt*ae; slp=entry-sl*ae
        exitpx=c[t+Hcap]                                  # default: time cap
        for j in range(t+1,t+Hcap+1):
            if d==-1:
                if h[j]>=slp: exitpx=slp; break            # SL checked first (pessimistic)
                if l[j]<=tp: exitpx=tp; break
            else:
                if l[j]<=slp: exitpx=slp; break
                if h[j]>=tp: exitpx=tp; break
        pnl=d*(exitpx-entry)/pip - sp
        out.append((ts[t], pnl, t<is_cut, vrel[t])); last=t
    return out


def main():
    con=duckdb.connect()
    GRID=[(1.0,1.0,8),(1.5,1.5,10),(2.0,2.0,12),(1.5,1.0,10),(2.0,1.5,12),(1.0,2.0,8)]
    print("First-touch H4 v2 — target/SL exit + volume. ATR-scaled. Net spread, IS/OOS.\n")
    # build all trades per config once
    cache={}
    for tgt,sl,H in GRID:
        per={p:trades(con,p,tgt,sl,H) for p in PAIRS}
        per={p:t for p,t in per.items() if len(t)>=40}
        cache[(tgt,sl,H)]=per
    # IS-select config by portfolio IS expectancy
    def is_exp(per):
        v=[x[1] for p in per for x in per[p] if x[2]]; return np.mean(v) if v else -99
    best_cfg=max(GRID, key=lambda g: is_exp(cache[g]))
    print(f"  config IS-expectancy ranking:")
    for g in sorted(GRID,key=lambda g:-is_exp(cache[g])):
        per=cache[g]; oo=[x[1] for p in per for x in per[p] if not x[2]]
        print(f"    tgt{g[0]} sl{g[1]} H{g[2]}: IS {is_exp(per):+.2f}p  OOS {np.mean(oo):+.2f}p (n_oos {len(oo)})")
    tgt,sl,H=best_cfg; per=cache[best_cfg]
    print(f"\n  === IS-best config: tgt={tgt} sl={sl} Hcap={H} ===")
    # per-pair OOS
    print("  per-pair OOS exp: "+"  ".join(f"{p}:{np.mean([x[1] for x in per[p] if not x[2]]):+.1f}" for p in per))
    allt=sorted([x for p in per for x in per[p]], key=lambda z:z[0])
    ts=np.array([z[0] for z in allt]); pn=np.array([z[1] for z in allt]); isf=np.array([z[2] for z in allt]); vr=np.array([z[3] for z in allt])
    print(f"  PORTFOLIO IS {pn[isf].mean():+.2f}p (n{isf.sum()}) | OOS {pn[~isf].mean():+.2f}p (n{(~isf).sum()}) WR{(pn[~isf]>0).mean()*100:.0f}%")
    # temporal WF (3 calendar thirds)
    e=[ts[0],ts[len(ts)//3],ts[2*len(ts)//3],ts[-1]]
    print("  temporal WF (3 thirds):", "  ".join(
        f"T{j+1} {pn[(ts>=e[j])&(ts<=e[j+1])].mean():+.2f}p(n{((ts>=e[j])&(ts<=e[j+1])).sum()})" for j in range(3)))
    # MC on OOS
    oos=pn[~isf]; boot=np.array([RNG.choice(oos,len(oos),replace=True).mean() for _ in range(3000)])
    print(f"  MC OOS: exp {oos.mean():+.2f}p 95%CI[{np.percentile(boot,2.5):+.2f},{np.percentile(boot,97.5):+.2f}] P(<=0)={(boot<=0).mean():.4f}")
    # VOLUME split (IS threshold)
    vmed=np.nanmedian(vr[isf])
    print(f"  volume split at touch (IS median vrel={vmed:.2f}):")
    for lab,m in [("loVol",vr<=vmed),("hiVol",vr>vmed)]:
        oo=pn[m&~isf]; ii=pn[m&isf]
        if len(oo)>20 and len(ii)>20:
            b=np.array([RNG.choice(oo,len(oo),replace=True).mean() for _ in range(2000)])
            print(f"    {lab}: IS {ii.mean():+.2f}p | OOS {oo.mean():+.2f}p (n{len(oo)}, WR{(oo>0).mean()*100:.0f}%) MC P(<=0)={(b<=0).mean():.3f}")
    # ROBUSTNESS of the loVol subset: temporal WF (full timeline) + per-pair
    lo=vr<=vmed
    lts=ts[lo]; lpn=pn[lo]
    le=[lts[0],lts[len(lts)//3],lts[2*len(lts)//3],lts[-1]]
    print(f"  loVol temporal WF (3 thirds, full timeline): "+"  ".join(
        f"T{j+1} {lpn[(lts>=le[j])&(lts<=le[j+1])].mean():+.2f}p(n{((lts>=le[j])&(lts<=le[j+1])).sum()})" for j in range(3)))
    print("  loVol per-pair OOS: "+"  ".join(
        f"{p}:{np.mean([x[1] for x in per[p] if (not x[2]) and x[3]<=vmed] or [0]):+.1f}" for p in per))
    npos=sum(1 for p in per if np.mean([x[1] for x in per[p] if (not x[2]) and x[3]<=vmed] or [0])>0)
    print(f"  loVol OOS pairs positive: {npos}/{len(per)}")
    con.close()
    print("\n  WIN if target/SL makes OOS MC P(<=0)<0.05 (tails trimmed), and/or a volume side is significant.")


if __name__=="__main__":
    main()
