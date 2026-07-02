"""
m5_regime_busyfade.py — push the lead: fade the SMA9/200 exhaustion thrust AT an S/R level on
BUSY (climactic) volume. The selectivity-stack found this cell at -0.56 p/tr (best broad fade).
Here: tighten the volume cut, check IS/OOS robustness (not a mined cell), per-pair breadth, and
two reversion exits. Net of real spread. Build the trade table once, then slice in memory.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq, os
from numba import njit
from m5_regime_exits import blkres, sma, atr, run_exit, entries, PAIRS, DATA, FAST, SLOW

W=5; RNG=96; KN=1000

@njit
def amddp_series(o5,h5,l5,c5,d,pip,n_m5,beta=0.05):
    out=np.full(n_m5,np.nan); n5=len(c5)
    for b in range(W-1,n_m5):
        start=(b-W+1)*60; end=b*60+60
        if end>n5: break
        entry=o5[start]; peak=0.0; ddsum=0.0
        for k in range(start,end):
            best=d*((h5[k] if d==1 else l5[k])-entry)/pip; worst=d*((l5[k] if d==1 else h5[k])-entry)/pip
            if best>peak: peak=best
            gb=peak-worst
            if gb>0: ddsum+=gb
        finalp=d*(c5[end-1]-entry)/pip; a=finalp-beta*ddsum
        if finalp>0 and a<0: a=0.001
        out[b]=a
    return out

def pctK(s,N):
    s=pd.Series(s); mn=s.rolling(N,min_periods=N//2).min(); mx=s.rolling(N,min_periods=N//2).max()
    return ((s-mn)/(mx-mn+1e-12)).to_numpy()

def build():
    rows=[]
    for p in PAIRS:
        f=DATA.format(p)
        if not os.path.exists(f): continue
        t=pq.read_table(f,columns=["open","high","low","close","bid_c","ask_c","volume","timestamp"])
        o,h,l,c,bid,ask,vol=(t.column(k).to_numpy().astype(np.float64) for k in
            ["open","high","low","close","bid_c","ask_c","volume"])
        ts=t.column("timestamp").to_numpy()
        O,H,L,C,BID,ASK=blkres(o,h,l,c,bid,ask); nb=len(C)
        TS=ts[:nb*60].reshape(nb,60)[:,0]; Vv=vol[:nb*60].reshape(nb,60).sum(1)
        pip=0.01 if "JPY" in p else 0.0001; sp=float(np.median(ASK-BID)/pip)
        s9,s200=sma(C,FAST),sma(C,SLOW); A=atr(H,L,C)
        kL=pctK(amddp_series(o,h,l,c,1,pip,nb),KN); kS=pctK(amddp_series(o,h,l,c,-1,pip,nb),KN)
        hiN=pd.Series(H).rolling(RNG).max().to_numpy(); loN=pd.Series(L).rolling(RNG).min().to_numpy()
        rp=(C-loN)/np.maximum(hiN-loN,1e-9)
        v3=pd.Series(Vv).rolling(3).sum().to_numpy(); medv=np.nanmedian(Vv)*3
        ei,di=entries(O,C,s9,s200); sig=ei-1
        f_tp =run_exit(O,H,L,C,s9,s200,A,s9,ei,-di,pip,sp,13,2.0,2.0)    # TP2+trail
        f_br =run_exit(O,H,L,C,s9,s200,A,s9,ei,-di,pip,sp, 4,1.5,1.5)    # ATR bracket
        for j in range(len(ei)):
            i=sig[j]; d0=di[j]
            rows.append((p, TS[i], f_tp[j], f_br[j],
                         (kS[i] if d0==-1 else kL[i]),
                         ((1.0-rp[i]) if d0==-1 else rp[i]),
                         v3[i]/medv if medv>0 else np.nan))
    return pd.DataFrame(rows,columns=["pair","ts","tp","br","exhaust","at_sr","vrel"]).dropna()

def stat(s,col="tp",rng=None):
    if len(s)<30: return f"n={len(s):4d}  (too few)"
    v=s[col].values; o=v.mean()
    if rng is None: rng=np.random.default_rng(0)
    null=np.array([(v*rng.choice(np.array([-1.,1.]),len(v))).mean() for _ in range(800)])
    return f"n={len(s):4d}  net {o:+6.2f}  WR {100*(v>0).mean():4.1f}%  MCp={float((np.abs(null)>=abs(o)).mean()):.3f}"

def main():
    df=build(); rng=np.random.default_rng(0)
    base=(df.at_sr>=0.7)&(df.exhaust>=0.5)
    print(f"Busy-volume exhaustion-S/R FADE (TP2+trail). {len(df)} fades. Net of spread.\n"+"="*78)
    print("  -- tighten the volume cut (AT_SR>=0.7, fade exhaustion) --")
    for vq in (0.70,0.80,0.90,0.95):
        m=base & (df.vrel>=df.vrel.quantile(vq))
        print(f"   BUSY vol >= {vq:.2f}q:  {stat(df[m],'tp',rng)}")
    print("  -- the lead cell, two exits --")
    lead=base & (df.vrel>=df.vrel.quantile(0.80))
    print(f"   TP2+trail :  {stat(df[lead],'tp',rng)}")
    print(f"   ATR bracket: {stat(df[lead],'br',rng)}")
    print("  -- IS/OOS robustness (70/30 by time) on the lead cell --")
    d2=df[lead].sort_values("ts"); cut=int(len(d2)*0.7)
    print(f"   IS : {stat(d2.iloc[:cut],'tp',rng)}")
    print(f"   OOS: {stat(d2.iloc[cut:],'tp',rng)}")
    print("  -- per-pair net (lead cell, TP2+trail): breadth check --")
    g=df[lead].groupby("pair").tp.agg(["mean","count"]).sort_values("mean")
    print("   " + "  ".join(f"{p}:{r['mean']:+.1f}(n{int(r['count'])})" for p,r in g.iterrows()))
    print(f"   pairs net-positive: {(g['mean']>0).sum()}/{len(g)}")

if __name__=="__main__": main()
