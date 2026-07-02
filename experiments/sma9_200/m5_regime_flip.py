"""
m5_regime_flip.py — CONTRARIAN flip of the SMA9/200 entry, on the ACCELERATING thrusts.

Entry signal = same regime first-occurrence trigger, but we FADE it (go long when the rule
says short) and only on accelerating thrusts: the 3 bars into the signal all moved in the
ORIGINAL signal direction, accelerating (r1>=r2>=r3>0), with total > thr spreads. Those were
the worst trades for the trend system (exhaustion) -> best fade candidates. Reversion exits
(quick ATR bracket / time cap) rather than holding to the far regime flip. Net of real spread.
"""
import numpy as np, pyarrow.parquet as pq, os
from numba import njit
from m5_regime_exits import blkres, sma, atr, run_exit, PAIRS, DATA, FAST, SLOW

@njit
def entries_flip(O,C,s9,s200,SP,thr):
    n=len(C); pos=0; ei=np.empty(n,np.int64); di=np.empty(n,np.int64); k=0
    for i in range(SLOW,n-1):
        if np.isnan(s9[i]) or np.isnan(s200[i]): continue
        if pos!=0:
            if (pos==-1 and s9[i]>=s200[i]) or (pos==1 and s9[i]<=s200[i]): pos=0
            continue
        down=s9[i]<s200[i]; up=s9[i]>s200[i]
        sc=down and (C[i]<s9[i]) and not((s9[i-1]<s200[i-1]) and (C[i-1]<s9[i-1]))
        lc=up   and (C[i]>s9[i]) and not((s9[i-1]>s200[i-1]) and (C[i-1]>s9[i-1]))
        d0=0
        if sc: d0=-1
        elif lc: d0=1
        if d0!=0:
            pos=d0                                   # one trigger per regime (matches first-occurrence)
            sp=SP[i]
            if sp>0:
                r1=d0*(C[i]-C[i-1])/sp; r2=d0*(C[i-1]-C[i-2])/sp; r3=d0*(C[i-2]-C[i-3])/sp
                if r1>0 and r2>0 and r3>0 and r1>=r2 and r2>=r3 and (r1+r2+r3)>thr:
                    ei[k]=i+1; di[k]=-d0; k+=1        # FLIP: fade the thrust
    return ei[:k], di[:k]

EXITS=[("MA-cross (hold to flip)",0,0.,0.), ("ATR bracket TP1.5/SL1.5",4,1.5,1.5),
       ("ATR bracket TP2/SL1",4,2.0,1.0), ("ATR bracket TP1/SL1.5",4,1.0,1.5),
       ("time-cap 12 bars",10,12.,0.), ("time-cap 36 bars",10,36.,0.),
       ("peak-giveback a1/g0.5",9,1.0,0.5),
       ("hard TP 3p",3,3.0,0.), ("hard TP 4p",3,4.0,0.), ("hard TP 6p",3,6.0,0.),
       ("TP2 + trail-step 2p",13,2.0,2.0),
       ("TP3 + trail-step 2p",13,3.0,2.0), ("TP4 + trail-step 2p",13,4.0,2.0)]

def main():
    rng=np.random.default_rng(0)
    print("CONTRARIAN flip on accelerating thrusts — fade the SMA9/200 trigger. Net of real spread.")
    for thr in (1.0,2.0,3.0):
        # preload + collect per-exit pooled pnl at this threshold
        pooled={lab:[] for lab,_,_,_ in EXITS}; nent=0; tot_days=0.0
        for p in PAIRS:
            f=DATA.format(p)
            if not os.path.exists(f): continue
            t=pq.read_table(f,columns=["open","high","low","close","bid_c","ask_c"])
            o,h,l,c,bid,ask=(t.column(k).to_numpy().astype(np.float64) for k in
                ["open","high","low","close","bid_c","ask_c"])
            O,H,L,C,BID,ASK=blkres(o,h,l,c,bid,ask); pip=0.01 if "JPY" in p else 0.0001
            SP=ASK-BID; sp_pips=float(np.median(SP)/pip)
            s9,s200=sma(C,FAST),sma(C,SLOW); A=atr(H,L,C)
            ei,di=entries_flip(O,C,s9,s200,SP,thr); nent+=len(ei); tot_days=max(tot_days,len(C)/288.)
            if len(ei)==0: continue
            for lab,code,p1,p2 in EXITS:
                pnl=run_exit(O,H,L,C,s9,s200,A,s9,ei,di,pip,sp_pips,code,float(p1),float(p2))
                pooled[lab].append(pnl)
        print(f"\n  thr>{thr:.0f} spreads, accelerating thrusts  —  {nent} fade trades  ({nent/tot_days:.1f}/day)")
        print(f"  {'exit':26s} {'net p/tr':>9} {'win%':>6} {'p/day':>8}   MC-p")
        for lab,_,_,_ in EXITS:
            if not pooled[lab]: continue
            v=np.concatenate(pooled[lab]); obs=v.mean()
            null=np.array([(v*rng.choice(np.array([-1.,1.]),len(v))).mean() for _ in range(1500)])
            mcp=float((np.abs(null)>=abs(obs)).mean())
            star = " 🟢" if (obs>0 and mcp<0.05) else ""
            print(f"  {lab:26s} {obs:+9.2f} {100*(v>0).mean():5.1f}% {v.sum()/tot_days:+8.1f}   {mcp:.3f}{star}")

if __name__=="__main__": main()
