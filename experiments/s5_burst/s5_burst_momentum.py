"""
s5_burst_momentum.py — micro-momentum at native S5 cadence.
Rule: an S5 bar whose range (H-L) exceeds k * the rolling-average M1 bar range (an explosive
5-second burst), followed by a same-direction S5 bar (confirmation) -> enter at the OPEN of the
3rd S5 bar in that direction, hold HOLD S5 bars (60s = 12), exit. Causal (enter after both bars
close), real per-bar spread deducted, 12 pairs. avg-M1-range is rolling + uses only minutes
completed before the bar (no leak).
"""
import numpy as np, pyarrow.parquet as pq, os, gc
from numba import njit

PAIRS=["USD_JPY","EUR_USD","GBP_USD","AUD_USD","EUR_JPY","GBP_JPY",
       "AUD_JPY","CAD_JPY","NZD_JPY","CHF_JPY","NZD_USD","EUR_GBP"]
DATA="data/s5_ohlc/{}_S5_BA.parquet"
RA=60   # rolling window (in M1 bars) for the average M1 range

def avg_m1_range_per_s5(H,L):
    n=len(H); nb=n//12
    h=H[:nb*12].reshape(nb,12).max(1); l=L[:nb*12].reshape(nb,12).min(1)
    m1r=h-l
    cs=np.cumsum(m1r); avg=np.full(nb,np.nan)
    avg[RA:]=(cs[RA:]-cs[:-RA])/RA                 # rolling mean ending at M1 bar m (causal)
    out=np.full(n,np.nan)
    for i in range(n):
        mi=i//12-1                                  # last COMPLETED M1 bar before bar i
        if mi>=RA: out[i]=avg[mi]
    return out

@njit
def sim(O,H,L,C,bid,ask,avgM1,hold,kthr,pip):
    n=len(C); m=0
    pn=np.empty(n//hold+2); dr=np.empty(n//hold+2)
    i=RA*12
    while i < n-hold-3:
        if np.isnan(avgM1[i]): i+=1; continue
        rng=H[i]-L[i]
        d = 1 if C[i]>O[i] else (-1 if C[i]<O[i] else 0)
        if d!=0 and rng > kthr*avgM1[i]:
            d2 = 1 if C[i+1]>O[i+1] else (-1 if C[i+1]<O[i+1] else 0)
            if d2==d:
                e=i+2; ent=O[e]; ex=C[e+hold-1]
                sp=(ask[e]-bid[e])/pip
                pn[m]=(d*(ex-ent)/pip)-sp; dr[m]=d; m+=1
                i=e+hold; continue                  # non-overlapping
        i+=1
    return pn[:m], dr[:m]

def main():
    rng=np.random.default_rng(0)
    for kthr in (1.0,1.5):
        for hold in (6,12,24):
            allp=[]; npos=0; tot_days=0.0
            for p in PAIRS:
                f=DATA.format(p)
                if not os.path.exists(f): continue
                t=pq.read_table(f,columns=["open","high","low","close","bid_c","ask_c"])
                O,H,L,C,bid,ask=(t.column(k).to_numpy().astype(np.float64) for k in
                    ["open","high","low","close","bid_c","ask_c"])
                pip=0.01 if "JPY" in p else 0.0001
                avgM1=avg_m1_range_per_s5(H,L)
                pn,dr=sim(O,H,L,C,bid,ask,avgM1,hold,kthr,pip)
                if len(pn): allp.append(pn); npos+=pn.mean()>0
                tot_days=max(tot_days,len(C)/17280.0)
                del O,H,L,C,bid,ask,avgM1; gc.collect()
            P=np.concatenate(allp) if allp else np.array([0.])
            o=P.mean(); null=np.array([(P*rng.choice(np.array([-1.,1.]),len(P))).mean() for _ in range(800)])
            mcp=float((np.abs(null)>=abs(o)).mean())
            print(f"k>{kthr} M1-range, hold {hold*5:3d}s ({hold} S5)  —  {len(P):6d} trades "
                  f"({len(P)/tot_days:5.1f}/day)  net {o:+6.2f} p/tr  WR {100*(P>0).mean():4.1f}%  "
                  f"{P.sum()/tot_days:+7.1f} p/day  {npos}/12+  MC p={mcp:.3f}")

if __name__=="__main__": main()
