"""
Average trade TRAJECTORY for the structural-fade entries: does the in-trade-direction path
zigzag up, wane, flatten, and roll over? And does the favorable VELOCITY wane before the peak
(so a momentum-exhaustion exit could bank near the top instead of giving it back)?

For each trade, sample the signed gross pnl d*(close[i+k]-entry)/pip and the running MFE at
fixed checkpoints k (bars since entry). Pool 12 pairs (OOS). Report:
  mean signed path, mean running-MFE, and mean per-segment velocity (slope) vs time.
"""
import sys, gc
import numpy as np
from numba import njit
sys.path.insert(0,"/path/to/projects/fx-core/research/experiments/spread_band_random")
from validate_structural_fade_scaled import build, PAIRS, M, HOLD_BARS, GAP_SECS, VW, VHI
KD=0.7
CHK=np.array([6,12,24,36,48,72,120,180,240,360,480,600,720],dtype=np.int64)  # bars: 30s..60min


@njit(cache=True)
def traj(opn,high,low,close,spread,regime,vrel,actH,actL,atr,ts,calm_thr,is_cut,pipv,chk):
    n=high.shape[0]; nc=chk.shape[0]
    sP=np.zeros(nc); sM=np.zeros(nc); cnt=np.zeros(nc)
    n_tr=0; i=max(M+1,VW+1)
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
        if i>=is_cut:                                  # OOS only
            entry=c; peak=0.0; ci=0; gap_end=10**18
            # find gap end
            for k in range(1,chk[nc-1]+1):
                if i+k>n-1: gap_end=k-1; break
                if ts[i+k]-ts[i+k-1]>GAP_SECS: gap_end=k-1; break
            for k in range(1,chk[nc-1]+1):
                if k>gap_end: break
                fav=d*(high[i+k]-entry)/pipv if d>0 else d*(low[i+k]-entry)/pipv
                # signed favorable extreme this bar
                fe= (high[i+k]-entry)/pipv if d>0 else (entry-low[i+k])/pipv
                if fe>peak: peak=fe
                if ci<nc and k==chk[ci]:
                    sP[ci]+=d*(close[i+k]-entry)/pipv
                    sM[ci]+=peak
                    cnt[ci]+=1.0
                    ci+=1
            n_tr+=1
        # advance ~one window
        nb=i+HOLD_BARS
        if nb>n-1: nb=n-1
        j=i+1
        while j<=nb and ts[j]-ts[j-1]<=GAP_SECS: j+=1
        i=j if j>i else i+1
    return sP,sM,cnt,n_tr


def main():
    nc=len(CHK); SP=np.zeros(nc); SM=np.zeros(nc); CN=np.zeros(nc); NT=0
    for pair in PAIRS:
        try: d=build(pair)
        except Exception as e: print(f"  skip {pair}: {e}",flush=True); continue
        sP,sM,cnt,nt=traj(d["opn"],d["high"],d["low"],d["close"],d["sp"],d["regime"],d["vrel"],
                          d["actH"],d["actL"],d["atr"],d["ts"],d["calm_thr"],d["is_cut"],d["pipv"],CHK)
        SP+=sP; SM+=sM; CN+=cnt; NT+=nt
        print(f"  {pair}: {nt:,} OOS trades",flush=True); del d; gc.collect()
    mp=SP/CN; mm=SM/CN
    print(f"\nAverage structural-fade trajectory (OOS, {NT:,} trades). pips, in trade direction.\n")
    print(f"{'t(min)':>7} {'mean close':>11} {'mean MFE':>9} {'seg vel(p/min)':>15}")
    print("-"*46)
    prev_c=0.0; prev_t=0.0
    for k in range(nc):
        tmin=CHK[k]*5/60.0
        vel=(mp[k]-prev_c)/(tmin-prev_t) if tmin>prev_t else 0.0
        print(f"{tmin:>7.1f} {mp[k]:>+11.3f} {mm[k]:>+9.3f} {vel:>+15.3f}")
        prev_c=mp[k]; prev_t=tmin
    pk=int(np.argmax(mm))
    print(f"\nMean MFE peaks by ~{CHK[pk]*5/60:.0f} min (running MFE still rising = peak not yet reached at that checkpoint).")
    print("If mean close rises then flattens/declines while MFE keeps rising slowly => trades give back after peak.")
    print("If segment velocity decays toward 0 before the close peak => momentum-exhaustion exit is detectable.")


if __name__=="__main__":
    main()
