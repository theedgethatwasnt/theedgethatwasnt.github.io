"""
m5_regime_selectivity.py — the synthesis test: can STACKING the three validated exhaustion-fade
filters concentrate the M5 fade enough to clear the spread (the way first-touch H4 does)?

Fade the SMA9/200 trigger (trade = -signal), exit TP2+trail-step 2p (best fade exit). Three
causal filters at the signal bar, each pointed the same way in earlier tests:
  EXHAUST  = Stochastic-%K of the AMDDP5-into-entry series (signal dir) near its 1000-bar PEAK
  QUIET    = low 3-bar tick volume (relative to pair median)
  AT_SR    = near the swing extreme we're fading from (range-position at the faded end)
Then walk the selectivity ladder: all -> each filter -> full stack at tightening cutoffs.
Net of real spread. If net crosses 0 as selectivity rises, concentration beat the toll.
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

def main():
    rng=np.random.default_rng(0); rows=[]
    for p in PAIRS:
        f=DATA.format(p)
        if not os.path.exists(f): continue
        t=pq.read_table(f,columns=["open","high","low","close","bid_c","ask_c","volume"])
        o,h,l,c,bid,ask,vol=(t.column(k).to_numpy().astype(np.float64) for k in
            ["open","high","low","close","bid_c","ask_c","volume"])
        O,H,L,C,BID,ASK=blkres(o,h,l,c,bid,ask); nb=len(C)
        Vv=vol[:nb*60].reshape(nb,60).sum(1); pip=0.01 if "JPY" in p else 0.0001
        sp=float(np.median(ASK-BID)/pip); s9,s200=sma(C,FAST),sma(C,SLOW); A=atr(H,L,C)
        kL=pctK(amddp_series(o,h,l,c, 1,pip,nb),KN); kS=pctK(amddp_series(o,h,l,c,-1,pip,nb),KN)
        hiN=pd.Series(H).rolling(RNG).max().to_numpy(); loN=pd.Series(L).rolling(RNG).min().to_numpy()
        rp=(C-loN)/(np.maximum(hiN-loN,1e-9))
        v3=pd.Series(Vv).rolling(3).sum().to_numpy(); medv=np.nanmedian(Vv)*3
        ei,di=entries(O,C,s9,s200); sig=ei-1
        fpnl=run_exit(O,H,L,C,s9,s200,A,s9,ei,-di,pip,sp,13,2.0,2.0)     # fade, TP2+trail-step
        for j in range(len(ei)):
            i=sig[j]; d0=di[j]
            ex=kS[i] if d0==-1 else kL[i]                                # exhaustion %K (signal dir)
            sr=(1.0-rp[i]) if d0==-1 else rp[i]                          # at faded extreme
            vr=v3[i]/medv if medv>0 else np.nan
            rows.append((fpnl[j], ex, vr, sr))
    df=pd.DataFrame(rows,columns=["pnl","exhaust","vrel","at_sr"]).dropna()
    N=len(df)
    def rep(name,m):
        s=df[m]
        if len(s)<30: print(f"  {name:42s} n={len(s):5d}  (too few)"); return
        o=s.pnl.mean(); null=np.array([(s.pnl.values*rng.choice(np.array([-1.,1.]),len(s))).mean() for _ in range(800)])
        mcp=float((np.abs(null)>=abs(o)).mean())
        flag=" 🟢" if (o>0 and mcp<0.05) else ""
        print(f"  {name:42s} n={len(s):5d} ({100*len(s)/N:4.1f}%)  net {o:+6.2f}  WR {100*(s.pnl>0).mean():4.1f}%  MCp={mcp:.3f}{flag}")
    print(f"Selectivity stack — fade SMA9/200 (TP2+trail), {N} base fade trades. Net of spread.")
    print("="*92)
    rep("ALL fades", df.index==df.index)
    print("  -- single filters (top tercile of each) --")
    rep("EXHAUST: AMDDP %K >= 0.67 (near peak)", df.exhaust>=0.67)
    rep("QUIET: volume <= 33rd pct", df.vrel<=df.vrel.quantile(0.33))
    rep("AT_SR: faded-extremity >= 0.67", df.at_sr>=0.67)
    print("  -- stacked, tightening --")
    for q in (0.50,0.67,0.80,0.90):
        m=(df.exhaust>=q)&(df.at_sr>=q)&(df.vrel<=df.vrel.quantile(1-q))
        rep(f"EXHAUST & AT_SR >= {q:.2f}  &  QUIET <= {1-q:.2f}", m)
    print("  -- contrast: exhaust+SR but BUSY volume --")
    rep("EXHAUST & AT_SR >=0.8  &  BUSY vol>=0.8q", (df.exhaust>=0.8)&(df.at_sr>=0.8)&(df.vrel>=df.vrel.quantile(0.8)))

if __name__=="__main__": main()
