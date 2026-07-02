"""
HOLD-TIME sweep for the structural fade. The trajectory shows favorable velocity is still
POSITIVE at 60min (+0.011 p/min) -- the 60-min cap cuts the reversion mid-drift. Test whether
a longer hold lets the drift clear spread (bridging toward the first-touch-H4 +9.49p edge,
which is the same contrarian-at-structure reversion held ~48h).

Exit = hold H bars then close, with a protective SL at SL_ATR*ATR (bound the losers).
Sweep H in {1,2,4,8,16,24}h. 12 pairs, IS/OOS, real per-bar spread, NET+GROSS+breadth.
"""
import sys, gc
import numpy as np
from numba import njit
sys.path.insert(0,"/path/to/projects/fx-core/research/experiments/spread_band_random")
from validate_structural_fade_scaled import build, PAIRS, M, GAP_SECS, VW, VHI
KD=0.7; SL_ATR=4.0
RNG=np.random.default_rng(17)
HOLDS=[720,1440,2880,5760,11520,17280]   # bars: 1,2,4,8,16,24 h (S5)


@njit(cache=True)
def run(opn,high,low,close,spread,regime,vrel,actH,actL,atr,ts,HOLD,calm_thr,is_cut,pipv):
    n=high.shape[0]; mt=n//4+8
    o_pnl=np.empty(mt,np.float64); o_grs=np.empty(mt,np.float64); o_ts=np.empty(mt,np.int64); o_is=np.empty(mt,np.int64)
    t=0; i=max(M+1,VW+1)
    while i<n-1:
        if regime[i]>calm_thr: i+=1; continue
        aH=actH[i]; aL=actL[i]; ae=atr[i]
        if aH<=0.0 or aL<=0.0 or ae<=0.0: i+=1; continue
        Dp=KD*ae; c=close[i]; dR=aH-c; dS=c-aL
        near_R= dR>=0.0 and dR<=Dp; near_S= dS>=0.0 and dS<=Dp
        if not(near_R or near_S): i+=1; continue
        at_R= near_R and (not near_S or dR<=dS); d= -1.0 if at_R else 1.0
        s=0.0
        for q in range(i-M+1,i+1): s+=vrel[q]
        if s/M<VHI: i+=1; continue
        entry=c; sp=spread[i]/pipv; sl=SL_ATR*ae
        last=i+HOLD
        if last>n-1: last=n-1
        exit_i=-1; pnl=0.0; j=i+1
        while j<=last:
            if ts[j]-ts[j-1]>GAP_SECS:
                jj=j-1; pnl=d*(close[jj]-entry)/pipv-sp; exit_i=jj; break
            if d>0.0:
                if entry-low[j]>=sl: pnl=-sl/pipv-sp; exit_i=j; break
            else:
                if high[j]-entry>=sl: pnl=-sl/pipv-sp; exit_i=j; break
            j+=1
        if exit_i<0:
            exit_i=last; pnl=d*(close[last]-entry)/pipv-sp
        o_pnl[t]=pnl; o_grs[t]=pnl+sp; o_ts[t]=ts[i]; o_is[t]=1 if i<is_cut else 0
        t+=1
        if exit_i<=i: exit_i=i+1
        i=exit_i
    return o_pnl[:t],o_grs[:t],o_ts[:t],o_is[:t]


def main():
    store={h:[] for h in HOLDS}; span=None
    for pair in PAIRS:
        try: d=build(pair)
        except Exception as e: print(f"  skip {pair}: {e}",flush=True); continue
        if span is None: span=(d["ts"][-1]-d["ts"][0])/86400
        for h in HOLDS:
            pn,gr,tss,ii=run(d["opn"],d["high"],d["low"],d["close"],d["sp"],d["regime"],d["vrel"],
                             d["actH"],d["actL"],d["atr"],d["ts"],h,d["calm_thr"],d["is_cut"],d["pipv"])
            store[h].append((pair,pn,gr,ii))
        print(f"  {pair}: done",flush=True); del d; gc.collect()
    print("\nHOLD-time sweep — structural fade, SL=4*ATR, hold-to-cap. 12 pairs, OOS, real spread.")
    print("Refs: 60min best -0.72 net | first-touch-H4 (~48h) +9.49 net.\n")
    print(f"{'hold':>5} | {'n':>7} {'/day':>5} | {'IS net':>7} {'OOS net':>8} {'OOS grs':>8} {'net WR':>6} {'netMC':>6} | {'pairs+ net':>10}")
    print("-"*92)
    best=None
    for h in HOLDS:
        P=np.concatenate([x[1] for x in store[h]]); G=np.concatenate([x[2] for x in store[h]])
        I=np.concatenate([x[3] for x in store[h]]).astype(bool); oos=P[~I]; oosg=G[~I]
        bN=np.array([RNG.choice(oos,len(oos),replace=True).mean() for _ in range(1200)])
        npos=npairs=0
        for (pr,pn,gr,ii) in store[h]:
            ii=ii.astype(bool); o=pn[~ii]
            if len(o)>=20: npairs+=1; npos+=o.mean()>0
        hh=f"{h*5//3600}h"
        print(f"{hh:>5} | {len(P):>7,} {len(P)/span/12:>5.2f} | {P[I].mean():>+7.3f} {oos.mean():>+8.3f} "
              f"{oosg.mean():>+8.3f} {100*(oos>0).mean():>5.1f}% {(bN<=0).mean():>6.3f} | {npos:>3}/12")
        if best is None or oos.mean()>best[1]: best=(hh,oos.mean(),oosg.mean(),npos)
    print(f"\nBest NET hold: {best[0]} -> net {best[1]:+.3f} gross {best[2]:+.3f} pairs+ {best[3]}/12")
    print("If OOS net rises (toward 0+/positive) as hold extends: the 60-min cap was the problem; drift needs time.")


if __name__=="__main__":
    main()
