"""
"Market-breathing" exit: the path to MFE is a zigzag of pushes + pullbacks. During a healthy
trend pullbacks correct ~30-50% of the run; as it exhausts they deepen toward ~100%. So exit
when the give-back from the peak exceeds FRAC of the PEAK GAIN (adaptive to the run size), not
a fixed pip/ATR amount.

Exit per trade (structural-fade entry, H1 S/R + hi-vol + calm):
  - peak = max favorable excursion from entry (price).
  - once peak >= ACT*ATR: exit when (peak - current_fav) >= FRAC * peak   [gave back FRAC of the run]
    => locks in (1-FRAC) of the peak gain.
  - pre-activation protective stop at S0*ATR; time cap 60min; gap.
Sweep FRAC (0.25..0.7) x ACT. 12 pairs, IS/OOS, real per-bar spread, NET+GROSS+capture.
"""
import sys, gc
import numpy as np
from numba import njit
sys.path.insert(0,"/path/to/projects/fx-core/research/experiments/spread_band_random")
from validate_structural_fade_scaled import build, PAIRS, M, HOLD_BARS, GAP_SECS, VW, VHI
KD=0.7; S0=1.0
RNG=np.random.default_rng(17)
GRID=[(fr,act) for fr in (0.25,0.4,0.55,0.7) for act in (0.3,0.6,1.0)]


@njit(cache=True)
def run(opn,high,low,close,spread,regime,vrel,actH,actL,atr,ts,FRAC,ACT,calm_thr,is_cut,pipv):
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
        entry=c; sp=spread[i]/pipv; act=ACT*ae; s0=S0*ae
        last=i+HOLD_BARS
        if last>n-1: last=n-1
        peak=0.0; exit_i=-1; pnl=0.0; j=i+1
        while j<=last:
            if ts[j]-ts[j-1]>GAP_SECS:
                jj=j-1; pnl=d*(close[jj]-entry)/pipv-sp; exit_i=jj; break
            h=high[j]; l=low[j]; bull=close[j]>=opn[j]
            # favorable extreme & adverse extreme this bar (in trade dir)
            if d>0.0:
                favhi=h-entry; cur_adv=l-entry          # current favorable using low
            else:
                favhi=entry-l; cur_adv=entry-h
            # R2: bull bar -> update peak(high) then check giveback(low); bear -> check then update
            def_chk=0
            if bull:
                if favhi>peak: peak=favhi
                if peak>=act and (peak-cur_adv)>=FRAC*peak:
                    lock=peak-FRAC*peak
                    pnl=lock/pipv-sp; exit_i=j; break
                if peak<act and (-cur_adv)>=s0:
                    pnl=-s0/pipv-sp; exit_i=j; break
            else:
                if peak>=act and (peak-cur_adv)>=FRAC*peak:
                    lock=peak-FRAC*peak; pnl=lock/pipv-sp; exit_i=j; break
                if peak<act and (-cur_adv)>=s0:
                    pnl=-s0/pipv-sp; exit_i=j; break
                if favhi>peak: peak=favhi
            j+=1
        if exit_i<0:
            exit_i=last; pnl=d*(close[last]-entry)/pipv-sp
        o_pnl[t]=pnl; o_grs[t]=pnl+sp; o_ts[t]=ts[i]; o_is[t]=1 if i<is_cut else 0
        t+=1
        if exit_i<=i: exit_i=i+1
        i=exit_i
    return o_pnl[:t],o_grs[:t],o_ts[:t],o_is[:t]


def main():
    store={g:[] for g in GRID}
    for pair in PAIRS:
        try: d=build(pair)
        except Exception as e: print(f"  skip {pair}: {e}",flush=True); continue
        for g in GRID:
            pn,gr,tss,ii=run(d["opn"],d["high"],d["low"],d["close"],d["sp"],d["regime"],d["vrel"],
                             d["actH"],d["actL"],d["atr"],d["ts"],g[0],g[1],d["calm_thr"],d["is_cut"],d["pipv"])
            store[g].append((pair,pn,gr,ii))
        print(f"  {pair}: done",flush=True); del d; gc.collect()
    print("\n'Breathing' exit: give back FRAC of the peak run (adaptive). H1 structural fade, hi-vol, 12 pairs, OOS.")
    print("Refs: static-trail best net -0.72 | MFE ceiling +6.4 | gross edge +1.1.\n")
    print(f"{'FRAC':>5} {'ACT':>4} | {'n':>7} {'OOS net':>8} {'OOS grs':>8} {'net WR':>6} {'netMC':>6} | {'pairs+ net':>10}")
    print("-"*72)
    best=None
    for g in GRID:
        P=np.concatenate([x[1] for x in store[g]]); G=np.concatenate([x[2] for x in store[g]])
        I=np.concatenate([x[3] for x in store[g]]).astype(bool); oos=P[~I]; oosg=G[~I]
        bN=np.array([RNG.choice(oos,len(oos),replace=True).mean() for _ in range(1200)])
        npos=npairs=0
        for (pr,pn,gr,ii) in store[g]:
            ii=ii.astype(bool); o=pn[~ii]
            if len(o)>=20: npairs+=1; npos+=o.mean()>0
        print(f"{g[0]:>5.2f} {g[1]:>4.1f} | {len(P):>7,} {oos.mean():>+8.3f} {oosg.mean():>+8.3f} "
              f"{100*(oos>0).mean():>5.1f}% {(bN<=0).mean():>6.3f} | {npos:>3}/12")
        if best is None or oos.mean()>best[1]: best=(g,oos.mean(),oosg.mean(),npos)
    print(f"\nBest NET: FRAC={best[0][0]} ACT={best[0][1]} -> net {best[1]:+.3f} gross {best[2]:+.3f} pairs+ {best[3]}/12")


if __name__=="__main__":
    main()
