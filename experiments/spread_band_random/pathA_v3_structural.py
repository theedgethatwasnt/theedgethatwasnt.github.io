"""
Path A v3 — give the HF entry a DIRECTION from STRUCTURE: proximity to H1 support/resistance
(TopsBots active swing high/low). Then layer volume + calm-band filters + trail/TP exit.

Structure: H1 TopsBots active HSP (resistance) / LSP (support), causal (use previous completed H1).
Per S5 bar: d_R=(act_h-close)/pip, d_S=(close-act_l)/pip.
Entry when within D pips of a level:
  - sfade  : near resistance -> SHORT; near support -> LONG   (reversion off the level)
  - sbreak : near resistance -> LONG ; near support -> SHORT   (break of the level)
Filters: calm band (trailing spread<=CALM_THR) + volume side (all/lo/hi on M-bar mean vrel).
Exit: trail=5 + TP=2, 60-min, gap. One trade at a time; spread up front; MID; SOP R2.

Usage: python3 pathA_v3_structural.py [PAIR]
"""
import sys, numpy as np, pandas as pd, duckdb
from numba import njit
sys.path.insert(0, "/path/to/projects/fx-core")
from lib.swing_indicators import compute_swing_features

PAIR="EUR_USD"
K_SPREAD=12; HOLD_BARS=720; GAP_SECS=60; TRAIL=5.0; TP=2.0; CALM_THR=1.50
VW=240; VLO=0.85; VHI=1.20
PIP=0.01 if PAIR.endswith("JPY") else 0.0001
PATH=f"data/s5_ohlc/{PAIR}_S5_BA.parquet"
Ds=[3.0,5.0,8.0,12.0]


@njit(cache=True)
def run(opn,high,low,close,spread,regime,vrel,actH,actL,ts,D,mode,volside,calm_thr,trail_pips,tp_pips,M):
    n=high.shape[0]; trail=trail_pips*PIP; mt=n//4+8
    out=np.empty(mt,np.float64); t=0; i=max(M+1,VW+1)
    while i<n-1:
        if regime[i]/PIP>calm_thr: i+=1; continue
        aH=actH[i]; aL=actL[i]
        if aH<=0.0 or aL<=0.0: i+=1; continue
        c=close[i]
        dR=(aH-c)/PIP; dS=(c-aL)/PIP            # >0 if below R / above S
        near_R = dR>=0.0 and dR<=D
        near_S = dS>=0.0 and dS<=D
        if not(near_R or near_S): i+=1; continue
        # choose the nearer level if both
        at_R = near_R and (not near_S or dR<=dS)
        if mode<0:   # sfade
            d = -1.0 if at_R else 1.0
        else:        # sbreak
            d = 1.0 if at_R else -1.0
        # volume filter on the approach (mean vrel over last M bars)
        if volside!=0:
            s=0.0
            for q in range(i-M+1,i+1): s+=vrel[q]
            mv=s/M
            if volside<0 and mv>VLO: i+=1; continue
            if volside>0 and mv<VHI: i+=1; continue
        entry=c; sp=spread[i]/PIP; tp_off=(tp_pips+sp)*PIP
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
    g=lambda c,t: df[c].to_numpy(t)
    ts=g("ts",np.int64); opn=g("open",np.float64); high=g("high",np.float64)
    low=g("low",np.float64); close=g("close",np.float64); sp=g("sp",np.float64); vol=g("volume",np.float64)
    cs=np.cumsum(sp); regime=np.empty_like(sp)
    regime[:K_SPREAD]=cs[:K_SPREAD]/(np.arange(K_SPREAD)+1); regime[K_SPREAD:]=(cs[K_SPREAD:]-cs[:-K_SPREAD])/K_SPREAD
    vmean=pd.Series(vol).rolling(VW).mean().shift(1).values
    vrel=np.nan_to_num(np.where(vmean>0,vol/vmean,1.0),nan=1.0)

    # ---- H1 TopsBots active S/R, causal, mapped to S5 ----
    print("Building H1 TopsBots S/R levels ...")
    tindex=pd.to_datetime(ts,unit="s",utc=True)
    h1=(pd.DataFrame({"high":high,"low":low,"close":close},index=tindex)
        .resample("1h").agg({"high":"max","low":"min","close":"last"}).dropna())
    _,_,act_h,act_l,_=compute_swing_features(h1.high.values,h1.low.values,h1.close.values)
    h1_start=(h1.index.view("int64")//1_000_000_000).astype(np.int64)
    # each S5 bar -> previous completed H1 bar's active levels (causal)
    pos=np.searchsorted(h1_start,ts,side="right")-1-1     # -1 contain, -1 previous completed
    pos=np.clip(pos,0,len(act_h)-1)
    actH=np.nan_to_num(act_h[pos],nan=0.0); actL=np.nan_to_num(act_l[pos],nan=0.0)
    span_days=(ts[-1]-ts[0])/86400
    print(f"{len(sp):,} S5 bars, {len(h1):,} H1 bars. Calm<= {CALM_THR}p, exit trail{TRAIL:.0f}/TP{TP:.0f}.\n")

    print(f"{'mode':>7} {'vol':>4} {'D(p)':>4} | {'trades':>7} {'/day':>5} | {'mean':>7} {'win%':>5} {'net':>9}")
    print("-"*66)
    for mode,mlbl in [(-1,"sfade"),(1,"sbreak")]:
        for volside,vlbl in [(0,"all"),(-1,"lo"),(1,"hi")]:
            for D in Ds:
                pnl=run(opn,high,low,close,sp,regime,vrel,actH,actL,ts,D,float(mode),float(volside),
                        CALM_THR,TRAIL,TP,24)
                if len(pnl)<200: continue
                print(f"{mlbl:>7} {vlbl:>4} {D:>4.0f} | {len(pnl):>7,} {len(pnl)/span_days:>5.1f} | "
                      f"{pnl.mean():>+7.3f} {100*(pnl>0).mean():>4.1f}% {pnl.sum():>+9,.0f}")
        print()
    print("Baselines: random calm -1.46, best HF fade -1.19. Target: structure -> direction -> toward 0+.")


if __name__=="__main__":
    main()
