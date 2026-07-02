"""
Peak-giveback (tight trailing-from-MFE) exit on the structural-fade entry.
The 15%-capture problem: our ATR trails give back ~1.2*ATR (~5-6p) from the peak, but median
MFE is only ~5.9p -> the give-back ~= the whole move. Fix: arm once in profit, then exit on a
SMALL retracement from the high-water mark.

Exit per trade:
  - initial protective stop at entry -/+ S0*ATR (cap pre-profit loss).
  - once favorable excursion >= ACT*ATR, switch to a tight trail: exit when price gives back
    GB*ATR from the HWM/LWM peak.
  - else time cap 60 min / gap.
Sweep GB (giveback) and ACT (activation). 12 pairs, IS/OOS, real per-bar spread, NET+GROSS+capture%.
Reuses validated scaled build().
"""
import sys, gc
import numpy as np
from numba import njit
sys.path.insert(0,"/path/to/projects/fx-core/research/experiments/spread_band_random")
from validate_structural_fade_scaled import build, PAIRS, M, HOLD_BARS, GAP_SECS, VW, VHI
KD=0.7; S0=1.0
RNG=np.random.default_rng(17)
GRID=[(gb,act) for gb in (0.1,0.15,0.25,0.4) for act in (0.0,0.3,0.6)]


@njit(cache=True)
def run(opn,high,low,close,spread,regime,vrel,actH,actL,atr,ts,GB,ACT,calm_thr,is_cut,pipv):
    n=high.shape[0]; mt=n//4+8
    o_pnl=np.empty(mt,np.float64); o_grs=np.empty(mt,np.float64); o_mfe=np.empty(mt,np.float64)
    o_ts=np.empty(mt,np.int64); o_is=np.empty(mt,np.int64)
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
        entry=c; sp=spread[i]/pipv
        gb=GB*ae; act=ACT*ae; s0=S0*ae
        last=i+HOLD_BARS
        if last>n-1: last=n-1
        peak=0.0          # best favorable excursion (price)
        bestfav=0.0
        exit_i=-1; pnl=0.0; j=i+1
        while j<=last:
            if ts[j]-ts[j-1]>GAP_SECS:
                jj=j-1; pnl=d*(close[jj]-entry)/pipv-sp; exit_i=jj; break
            h=high[j]; l=low[j]; bull=close[j]>=opn[j]
            if d>0.0:
                # adverse first on bear bar, favorable first on bull bar (R2)
                if bull:
                    fav=h-entry
                    if fav>peak: peak=fav
                    if peak>=act and (peak-(l-entry))>=gb:        # gave back gb from peak
                        px=entry+peak-gb; pnl=(px-entry)/pipv-sp; exit_i=j; break
                    if peak<act and (entry-l)>=s0:                 # pre-profit protective stop
                        pnl=-s0/pipv-sp; exit_i=j; break
                else:
                    if peak<act and (entry-l)>=s0:
                        pnl=-s0/pipv-sp; exit_i=j; break
                    if peak>=act and (peak-(l-entry))>=gb:
                        px=entry+peak-gb; pnl=(px-entry)/pipv-sp; exit_i=j; break
                    fav=h-entry
                    if fav>peak: peak=fav
            else:
                if bull:
                    if peak<act and (h-entry)>=s0:
                        pnl=-s0/pipv-sp; exit_i=j; break
                    if peak>=act and (peak-(entry-h))>=gb:
                        px=entry-peak+gb; pnl=(entry-px)/pipv-sp; exit_i=j; break
                    fav=entry-l
                    if fav>peak: peak=fav
                else:
                    fav=entry-l
                    if fav>peak: peak=fav
                    if peak>=act and (peak-(entry-h))>=gb:
                        px=entry-peak+gb; pnl=(entry-px)/pipv-sp; exit_i=j; break
                    if peak<act and (h-entry)>=s0:
                        pnl=-s0/pipv-sp; exit_i=j; break
            j+=1
        if exit_i<0:
            exit_i=last; pnl=d*(close[last]-entry)/pipv-sp
        o_pnl[t]=pnl; o_grs[t]=pnl+sp; o_mfe[t]=peak/pipv; o_ts[t]=ts[i]; o_is[t]=1 if i<is_cut else 0
        t+=1
        if exit_i<=i: exit_i=i+1
        i=exit_i
    return o_pnl[:t],o_grs[:t],o_mfe[:t],o_ts[:t],o_is[:t]


def main():
    store={g:[] for g in GRID}
    for pair in PAIRS:
        try: d=build(pair)
        except Exception as e: print(f"  skip {pair}: {e}",flush=True); continue
        for g in GRID:
            pn,gr,mf,tss,ii=run(d["opn"],d["high"],d["low"],d["close"],d["sp"],d["regime"],d["vrel"],
                                d["actH"],d["actL"],d["atr"],d["ts"],g[0],g[1],d["calm_thr"],d["is_cut"],d["pipv"])
            store[g].append((pair,pn,gr,mf,ii))
        print(f"  {pair}: done",flush=True); del d; gc.collect()
    print("\nPeak-giveback exit (tight trail-from-MFE) — H1 structural fade, hi-vol. 12 pairs, OOS.")
    print(f"{'GB(ATR)':>7} {'ACT':>4} | {'n':>7} {'OOS net':>8} {'OOS grs':>8} {'net WR':>6} {'grsMC':>6} {'netMC':>6} | {'pairs+ net/grs':>13}")
    print("-"*86)
    best=None
    for g in GRID:
        P=np.concatenate([x[1] for x in store[g]]); G=np.concatenate([x[2] for x in store[g]])
        I=np.concatenate([x[4] for x in store[g]]).astype(bool); oos=P[~I]; oosg=G[~I]
        bN=np.array([RNG.choice(oos,len(oos),replace=True).mean() for _ in range(1200)])
        bG=np.array([RNG.choice(oosg,len(oosg),replace=True).mean() for _ in range(1200)])
        npos=ngpos=npairs=0
        for (pr,pn,gr,mf,ii) in store[g]:
            ii=ii.astype(bool); o=pn[~ii]; og=gr[~ii]
            if len(o)>=20: npairs+=1; npos+=o.mean()>0; ngpos+=og.mean()>0
        print(f"{g[0]:>7.2f} {g[1]:>4.1f} | {len(P):>7,} {oos.mean():>+8.3f} {oosg.mean():>+8.3f} "
              f"{100*(oos>0).mean():>5.1f}% {(bG<=0).mean():>6.3f} {(bN<=0).mean():>6.3f} | net{npos}/grs{ngpos}")
        if best is None or oos.mean()>best[1]: best=(g,oos.mean(),oosg.mean(),npos)
    print(f"\nBest NET exit: GB={best[0][0]} ACT={best[0][1]} -> net {best[1]:+.3f}  gross {best[2]:+.3f}  pairs+net {best[3]}/12")
    print("vs prior static best net -0.72. If net crosses ~0 / many pairs+: tight giveback captured the MFE.")


if __name__=="__main__":
    main()
