"""
Path A v2: volume-optimize the best HF entry (contrarian fade of an extension, calm-band gated).
Volume study says: high-vol move => continues (genuine break); low-vol move => reverts.
So FADE should prefer LOW-volume extensions; FOLLOW should prefer HIGH-volume.

Entry at bar i: move = (close[i]-close[i-M])/pip over M bars. |move|>=N triggers.
 mode fade  -> trade against the move; follow -> with it.
 vol side: classify the extension by mean vrel over the last M bars:
   lo = mean_vrel <= VLO ;  hi = mean_vrel >= VHI ;  all = no vol filter.
Gate: calm band (trailing-12-bar avg spread <= CALM_THR). Exit trail=5 + TP=2, 60-min, gap.
One trade at a time; spread up front; MID PnL; SOP R2.

Usage: python3 pathA_v2_volume.py [PAIR]
"""
import numpy as np, duckdb
from numba import njit

PAIR="EUR_USD"
K_SPREAD=12; HOLD_BARS=720; GAP_SECS=60; TRAIL=5.0; TP=2.0; CALM_THR=1.50
VW=240                     # ~20 min trailing window for the volume mean (S5)
VLO=0.85; VHI=1.20
PIP=0.01 if PAIR.endswith("JPY") else 0.0001
PATH=f"data/s5_ohlc/{PAIR}_S5_BA.parquet"
# two frequency tiers: VERY-HIGH-freq (small N, ~tens/day) + HIGH-freq (N>=5, ~few/day)
CFGS=[(6,2),(12,2),(6,3),(12,3),(12,5),(24,7)]   # (M bars, N pips); 6bars=30s,12=1min,24=2min


@njit(cache=True)
def run(opn,high,low,close,spread,regime,vrel,ts,M,N,mode,volside,calm_thr,trail_pips,tp_pips):
    n=high.shape[0]; trail=trail_pips*PIP; mt=n//4+8
    out=np.empty(mt,np.float64); t=0; i=max(M+1,VW+1)
    while i<n-1:
        if regime[i]/PIP>calm_thr: i+=1; continue
        # extension volume conviction = mean vrel over last M bars
        s=0.0
        for q in range(i-M+1,i+1): s+=vrel[q]
        mv=s/M
        if volside<0 and mv>VLO: i+=1; continue      # want low-vol only
        if volside>0 and mv<VHI: i+=1; continue       # want high-vol only
        move=(close[i]-close[i-M])/PIP
        d=0.0
        if mode<0:
            if move>=N: d=-1.0
            elif move<=-N: d=1.0
        else:
            if move>=N: d=1.0
            elif move<=-N: d=-1.0
        if d==0.0: i+=1; continue
        entry=close[i]; sp=spread[i]/PIP; tp_off=(tp_pips+sp)*PIP
        last=i+HOLD_BARS
        if last>n-1: last=n-1
        hwm=entry; lwm=entry; exit_i=-1; pnl=0.0; j=i+1
        while j<=last:
            if ts[j]-ts[j-1]>GAP_SECS:
                jj=j-1; pnl=d*(close[jj]-entry)/PIP-sp; exit_i=jj; break
            bull=close[j]>=opn[j]; h=high[j]; l=low[j]
            if d>0.0:
                tl=entry+tp_off
                if bull:
                    if h>=tl: pnl=tp_pips; exit_i=j; break
                    if h>hwm: hwm=h
                    if l<=hwm-trail: pnl=(hwm-trail-entry)/PIP-sp; exit_i=j; break
                else:
                    if l<=hwm-trail: pnl=(hwm-trail-entry)/PIP-sp; exit_i=j; break
                    if h>=tl: pnl=tp_pips; exit_i=j; break
                    if h>hwm: hwm=h
            else:
                tl=entry-tp_off
                if bull:
                    if h>=lwm+trail: pnl=(entry-(lwm+trail))/PIP-sp; exit_i=j; break
                    if l<=tl: pnl=tp_pips; exit_i=j; break
                    if l<lwm: lwm=l
                else:
                    if l<=tl: pnl=tp_pips; exit_i=j; break
                    if l<lwm: lwm=l
                    if h>=lwm+trail: pnl=(entry-(lwm+trail))/PIP-sp; exit_i=j; break
            j+=1
        if exit_i<0:
            exit_i=last; pnl=d*(close[last]-entry)/PIP-sp
        out[t]=pnl; t+=1
        if exit_i<=i: exit_i=i+1
        i=exit_i
    return out[:t]


def main():
    print(f"Loading {PATH} ...")
    df=duckdb.sql(f"SELECT epoch(timestamp)::BIGINT ts, open, high, low, close, (ask_c-bid_c) sp, volume "
                  f"FROM '{PATH}' WHERE ask_c>bid_c ORDER BY timestamp").df()
    a=lambda c,t: df[c].to_numpy(t)
    ts=a("ts",np.int64); opn=a("open",np.float64); high=a("high",np.float64)
    low=a("low",np.float64); close=a("close",np.float64); sp=a("sp",np.float64); vol=a("volume",np.float64)
    cs=np.cumsum(sp); regime=np.empty_like(sp)
    regime[:K_SPREAD]=cs[:K_SPREAD]/(np.arange(K_SPREAD)+1); regime[K_SPREAD:]=(cs[K_SPREAD:]-cs[:-K_SPREAD])/K_SPREAD
    # vrel = vol / trailing-VW mean (causal)
    import pandas as pd
    vmean=pd.Series(vol).rolling(VW).mean().shift(1).values
    vrel=np.where(vmean>0, vol/vmean, 1.0); vrel=np.nan_to_num(vrel,nan=1.0)
    print(f"{len(sp):,} bars. Calm gate<= {CALM_THR}p, exit trail{TRAIL:.0f}/TP{TP:.0f}. "
          f"VLO={VLO} VHI={VHI} (mean vrel over M bars).\n")
    print(f"{'mode':>7} {'vol':>4} {'M(min)':>6} {'N':>3} | {'trades':>7} {'/day':>5} | {'mean':>7} {'win%':>5} {'net':>9}")
    print("-"*72)
    span_days=(ts[-1]-ts[0])/86400
    for mode,mlbl in [(-1,"fade"),(1,"follow")]:
        for volside,vlbl in [(0,"all"),(-1,"lo"),(1,"hi")]:
            for M,N in CFGS:
                pnl=run(opn,high,low,close,sp,regime,vrel,ts,M,float(N),float(mode),float(volside),
                        CALM_THR,TRAIL,TP)
                if len(pnl)<200: continue
                print(f"{mlbl:>7} {vlbl:>4} {M*5//60:>6} {N:>3} | {len(pnl):>7,} {len(pnl)/span_days:>5.1f} | "
                      f"{pnl.mean():>+7.3f} {100*(pnl>0).mean():>4.1f}% {pnl.sum():>+9,.0f}")
        print()
    print("Baseline fade-all best ~ -1.19 p/trade. Target: volume filter -> toward/above 0.")


if __name__=="__main__":
    main()
