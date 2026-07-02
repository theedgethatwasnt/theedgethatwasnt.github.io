"""
m5_dualslope.py — entry on AGREEMENT of the SMA-fast and SMA-200 momentum (slopes), each above
its own magnitude threshold; exit on the fast/slow SMA CROSSOVER. M5 from S5, causal next-bar-open
fills, real per-bar spread. Sweeps the fast-SMA period and the slope-strength threshold, 12 pairs.

slope = SMA[i]-SMA[i-3] (3-bar), normalized by ATR. Enter when sign(slope_fast)==sign(slope_slow)!=0
AND |slope_fast|>=KF*ATR AND |slope_slow|>=KS*ATR. Trade that direction. Exit when sign(fast-slow)
flips from its value at entry (the crossover). One position at a time.
"""
import numpy as np, pyarrow.parquet as pq, os
from numba import njit
from m5_regime_exits import blkres, sma, atr, PAIRS, DATA, SLOW

@njit
def sim(O,C,sf,ss,A,bid,ask,KF,KS,pip):
    n=len(C); pos=0; entry=0.0; c0=0; spd=0.0
    pn=np.empty(n); dr=np.empty(n); m=0
    for i in range(SLOW+3,n-1):
        if np.isnan(sf[i]) or np.isnan(ss[i]) or np.isnan(A[i]): continue
        if pos!=0:
            cs = 1 if sf[i]>ss[i] else (-1 if sf[i]<ss[i] else c0)
            if cs!=c0:                                            # fast/slow crossover from entry
                ex=O[i+1]; pn[m]=(pos*(ex-entry)/pip)-spd; dr[m]=pos; m+=1; pos=0
            continue
        sfs=sf[i]-sf[i-3]; sss=ss[i]-ss[i-3]
        df = 1 if sfs>0 else (-1 if sfs<0 else 0)
        ds = 1 if sss>0 else (-1 if sss<0 else 0)
        if df!=0 and df==ds and abs(sfs)>=KF*A[i] and abs(sss)>=KS*A[i]:
            entry=O[i+1]; pos=df; spd=(ask[i+1]-bid[i+1])/pip      # full round-trip spread at fill
            c0 = 1 if sf[i]>ss[i] else (-1 if sf[i]<ss[i] else 0)
    return pn[:m], dr[:m]

def main():
    rng=np.random.default_rng(0)
    print("Dual-slope agreement entry (SMA-fast & SMA200 momentum agree + strong), crossover exit. Net of spread.")
    print("="*96)
    for fast in (5,9,13,21):
        for KF in (0.10,0.20,0.40):
            KS=0.02
            allp=[]; npos=0; tot_days=0.0
            for p in PAIRS:
                f=DATA.format(p)
                if not os.path.exists(f): continue
                t=pq.read_table(f,columns=["open","high","low","close","bid_c","ask_c"])
                o,h,l,c,bid,ask=(t.column(k).to_numpy().astype(np.float64) for k in
                    ["open","high","low","close","bid_c","ask_c"])
                O,H,L,C,BID,ASK=blkres(o,h,l,c,bid,ask); pip=0.01 if "JPY" in p else 0.0001
                sf,ss=sma(C,fast),sma(C,SLOW); A=atr(H,L,C)
                pn,dr=sim(O,C,sf,ss,A,BID,ASK,KF,KS,pip)
                if len(pn): allp.append(pn); npos+=pn.mean()>0
                tot_days=max(tot_days,len(C)/288.0)
            P=np.concatenate(allp) if allp else np.array([0.])
            o=P.mean(); null=np.array([(P*rng.choice(np.array([-1.,1.]),len(P))).mean() for _ in range(800)])
            mcp=float((np.abs(null)>=abs(o)).mean())
            flag=" 🟢" if (o>0 and mcp<0.05 and npos>=7) else ""
            print(f"  SMA{fast:2d}/200  KF={KF:.2f}  —  {len(P):6d} tr ({len(P)/tot_days:4.1f}/day)  "
                  f"net {o:+6.2f}  WR {100*(P>0).mean():4.1f}%  {P.sum()/tot_days:+7.1f} p/day  {npos:2d}/12+  MCp={mcp:.3f}{flag}")

if __name__=="__main__": main()
