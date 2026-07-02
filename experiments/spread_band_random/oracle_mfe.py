"""
ORACLE / MFE ceiling for the structural-fade entries. The question: could a BETTER exit
clear spread? Our trail/TP sweeps only bound the exit STRUCTURES we tried. The true ceiling
is the MFE (max favorable excursion) per trade = what a perfect-foresight exit could bank.

Same entries (structural fade H1 S/R, kD=0.7, hi-vol, per-pair calm gate). For each trade,
scan the full 60-min window (gap-stopped) and record:
  MFE_gross = max favorable excursion (pips)   MAE_gross = max adverse excursion (pips)
  MFE_net   = MFE_gross - spread               (best ANY causal exit could net, upper bound)
  reach%    = fraction of trades whose MFE_net > 0 (reach a profitable exit at some point)
If even the ORACLE (MFE_net) is small vs spread -> no exit clears it (proven).
If MFE_net is large -> exit NOT exhausted; capture is a peak-timing/forecast problem.
12 pairs, IS/OOS 60-40, real per-bar spread.
"""
import sys, gc
import numpy as np
from numba import njit
sys.path.insert(0,"/path/to/projects/fx-core/research/experiments/spread_band_random")
from validate_structural_fade_scaled import build, PAIRS, M, HOLD_BARS, GAP_SECS, VW, VHI
KD=0.7
RNG=np.random.default_rng(17)


@njit(cache=True)
def mfe(opn,high,low,close,spread,regime,vrel,actH,actL,atr,ts,calm_thr,is_cut,pipv):
    n=high.shape[0]; mt=n//4+8
    o_mfe=np.empty(mt,np.float64); o_mae=np.empty(mt,np.float64); o_sp=np.empty(mt,np.float64)
    o_tpk=np.empty(mt,np.float64); o_is=np.empty(mt,np.int64)
    t=0; i=max(M+1,VW+1)
    while i<n-1:
        if regime[i]>calm_thr: i+=1; continue
        aH=actH[i]; aL=actL[i]; ae=atr[i]
        if aH<=0.0 or aL<=0.0 or ae<=0.0: i+=1; continue
        Dp=KD*ae; c=close[i]; dR=aH-c; dS=c-aL
        near_R= dR>=0.0 and dR<=Dp; near_S= dS>=0.0 and dS<=Dp
        if not(near_R or near_S): i+=1; continue
        at_R= near_R and (not near_S or dR<=dS)
        d= -1.0 if at_R else 1.0
        s=0.0
        for q in range(i-M+1,i+1): s+=vrel[q]
        if s/M<VHI: i+=1; continue
        entry=c; sp=spread[i]/pipv
        last=i+HOLD_BARS
        if last>n-1: last=n-1
        best_fav=-1e9; worst_adv=-1e9; t_peak=0; j=i+1; ended=last
        while j<=last:
            if ts[j]-ts[j-1]>GAP_SECS: ended=j-1; break
            if d>0.0:
                fav=(high[j]-entry)/pipv; adv=(entry-low[j])/pipv
            else:
                fav=(entry-low[j])/pipv; adv=(high[j]-entry)/pipv
            if fav>best_fav: best_fav=fav; t_peak=j-i
            if adv>worst_adv: worst_adv=adv
            j+=1
        # advance one trade per ~window so trades are roughly non-overlapping (like the strategy)
        o_mfe[t]=best_fav; o_mae[t]=worst_adv; o_sp[t]=sp; o_tpk[t]=t_peak*5.0/60.0; o_is[t]=1 if i<is_cut else 0
        t+=1
        i=ended if ended>i else i+1
    return o_mfe[:t],o_mae[:t],o_sp[:t],o_tpk[:t],o_is[:t]


def main():
    A=[[],[],[],[],[]]  # mfe,mae,sp,tpk,is
    perpair={}
    for pair in PAIRS:
        try: d=build(pair)
        except Exception as e: print(f"  skip {pair}: {e}",flush=True); continue
        mf,ma,sp,tp,ii=mfe(d["opn"],d["high"],d["low"],d["close"],d["sp"],d["regime"],d["vrel"],
                           d["actH"],d["actL"],d["atr"],d["ts"],d["calm_thr"],d["is_cut"],d["pipv"])
        for k,arr in enumerate((mf,ma,sp,tp,ii)): A[k].append(arr)
        cl=(mf<500)&(mf>-1)&(ma<500)&(ma>-1)
        oo=(~ii.astype(bool))&cl
        perpair[pair]=(np.median(sp[oo]), np.median(mf[oo]-sp[oo]), np.median(mf[oo]), np.median(ma[oo]),
                       100*((mf[oo]-sp[oo])>0).mean())
        print(f"  {pair}: {len(mf):,} trades",flush=True); del d; gc.collect()
    mf=np.concatenate(A[0]); ma=np.concatenate(A[1]); sp=np.concatenate(A[2]); tp=np.concatenate(A[3])
    I=np.concatenate(A[4]).astype(bool)
    clean=(mf<500)&(mf>-1)&(ma<500)&(ma>-1)            # drop bad-tick outliers
    print(f"  (dropped {(~clean).sum()} bad-tick outlier trades of {len(mf):,})")
    mf,ma,sp,tp,I = mf[clean],ma[clean],sp[clean],tp[clean],I[clean]
    oo=~I
    print(f"\nORACLE / MFE ceiling — structural-fade entries, OOS (n={oo.sum():,}). pips.")
    print(f"  mean spread            : {sp[oo].mean():.2f}")
    print(f"  mean MFE (gross peak)  : {mf[oo].mean():.2f}   (median {np.median(mf[oo]):.2f}, p90 {np.percentile(mf[oo],90):.2f})")
    print(f"  mean MAE (adverse peak): {ma[oo].mean():.2f}")
    print(f"  mean MFE_net (peak-sp) : {(mf[oo]-sp[oo]).mean():+.2f}   <- ORACLE ceiling (perfect-foresight exit)")
    print(f"  reach% (MFE_net>0)     : {100*((mf[oo]-sp[oo])>0).mean():.1f}%   (trades that touch profit at some point)")
    print(f"  median time-to-peak    : {np.median(tp[oo]):.1f} min   (mean {tp[oo].mean():.1f})")
    print(f"\n  realized best exit (TP1.5/trail) OOS net was ~ -0.72p, gross +1.26p.")
    print(f"  capture ratio (gross/MFE): {1.26/mf[oo].mean()*100:.0f}%  -> MFE LEFT ON TABLE: {mf[oo].mean()-1.26:.2f}p gross\n")
    print(f"{'pair':>9} {'spread':>7} {'MFE':>6} {'MAE':>6} {'MFE_net':>8} {'reach%':>7}")
    print("-"*52)
    for p,(s,mn,mfm,mae,rc) in perpair.items():
        print(f"{p:>9} {s:>7.2f} {mfm:>6.2f} {mae:>6.2f} {mn:>+8.2f} {rc:>6.1f}%")
    npos=sum(1 for v in perpair.values() if v[1]>0)
    print(f"\n  pairs with ORACLE MFE_net > 0: {npos}/12")
    print("  If oracle MFE_net >> 0 broadly: exit NOT exhausted, but capturing it = peak-timing (a forecast problem).")


if __name__=="__main__":
    main()
