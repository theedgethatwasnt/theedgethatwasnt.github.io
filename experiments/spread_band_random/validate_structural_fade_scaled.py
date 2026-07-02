"""
VALIDATE structural-fade with PER-PAIR-SCALED settings (fix the EUR_USD-bias of the fixed-setting run).
 - calm gate  = per-pair IS-median trailing spread (SOP R5: IS-only threshold), not a fixed 1.5p
 - proximity D = k_D * ATR(14,H1)  (ATR units, not fixed pips)
 - trail/TP    = k_trail/k_tp * ATR(14,H1) frozen at entry  (ATR units)
 - hi-vol gate = mean vrel(last 24 S5) >= 1.2
IS/OOS 60/40 per pair, IS-select (k_D, k_trail/k_tp, vol), OOS+MC+per-pair+WF, NET and GROSS.
Memory-safe: one pair at a time.
"""
import sys, gc
import numpy as np, pandas as pd, duckdb
from numba import njit
sys.path.insert(0, "/path/to/projects/fx-core")
from lib.swing_indicators import compute_swing_features

PAIRS=["EUR_USD","USD_JPY","GBP_USD","AUD_USD","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY",
       "CHF_JPY","NZD_JPY","NZD_USD","EUR_GBP"]
K_SPREAD=12; HOLD_BARS=720; GAP_SECS=60; VW=240; VHI=1.20; M=24; IS_FRAC=0.6
# grid: (k_D, k_trail, k_tp, vol)  vol 0=all 1=hi
GRID=[(kd,kt,ktp,v) for kd in (0.4,0.7) for (kt,ktp) in ((0.7,0.3),(1.2,0.5)) for v in (0,1)]
RNG=np.random.default_rng(17)


def hatr(h,l,c,n=14):
    pc=np.empty_like(c); pc[0]=c[0]; pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values


@njit(cache=True)
def run(opn,high,low,close,spread,regime,vrel,actH,actL,atr,ts,kD,kT,kTP,volside,calm_thr,M_,is_cut,pipv):
    n=high.shape[0]; mt=n//4+8
    o_pnl=np.empty(mt,np.float64); o_grs=np.empty(mt,np.float64); o_ts=np.empty(mt,np.int64); o_is=np.empty(mt,np.int64)
    t=0; i=max(M_+1,VW+1)
    while i<n-1:
        if regime[i]>calm_thr: i+=1; continue          # per-pair IS-median spread gate (price units)
        aH=actH[i]; aL=actL[i]; ae=atr[i]
        if aH<=0.0 or aL<=0.0 or ae<=0.0: i+=1; continue
        Dp=kD*ae; trail=kT*ae                            # ATR-scaled, price units
        c=close[i]; dR=aH-c; dS=c-aL
        near_R= dR>=0.0 and dR<=Dp; near_S= dS>=0.0 and dS<=Dp
        if not(near_R or near_S): i+=1; continue
        at_R= near_R and (not near_S or dR<=dS)
        d= -1.0 if at_R else 1.0
        if volside>0:
            s=0.0
            for q in range(i-M_+1,i+1): s+=vrel[q]
            if s/M_<VHI: i+=1; continue
        entry=c; sp=spread[i]/pipv                       # spread in pips
        tp_pips=kTP*ae/pipv                              # ATR-scaled TP in pips
        tp_off=kTP*ae + spread[i]                        # price move for +tp net
        last=i+HOLD_BARS
        if last>n-1: last=n-1
        hwm=entry; lwm=entry; exit_i=-1; pnl=0.0; j=i+1
        while j<=last:
            if ts[j]-ts[j-1]>GAP_SECS:
                jj=j-1; pnl=d*(close[jj]-entry)/pipv-sp; exit_i=jj; break
            bull=close[j]>=opn[j]; h=high[j]; l=low[j]
            if d>0.0:
                tl=entry+tp_off
                if bull:
                    if h>=tl: pnl=tp_pips; exit_i=j; break
                    if h>hwm: hwm=h
                    if l<=hwm-trail: pnl=(hwm-trail-entry)/pipv-sp; exit_i=j; break
                else:
                    if l<=hwm-trail: pnl=(hwm-trail-entry)/pipv-sp; exit_i=j; break
                    if h>=tl: pnl=tp_pips; exit_i=j; break
                    if h>hwm: hwm=h
            else:
                tl=entry-tp_off
                if bull:
                    if h>=lwm+trail: pnl=(entry-(lwm+trail))/pipv-sp; exit_i=j; break
                    if l<=tl: pnl=tp_pips; exit_i=j; break
                    if l<lwm: lwm=l
                else:
                    if l<=tl: pnl=tp_pips; exit_i=j; break
                    if l<lwm: lwm=l
                    if h>=lwm+trail: pnl=(entry-(lwm+trail))/pipv-sp; exit_i=j; break
            j+=1
        if exit_i<0:
            exit_i=last; pnl=d*(close[last]-entry)/pipv-sp
        o_pnl[t]=pnl; o_grs[t]=pnl+sp; o_ts[t]=ts[i]; o_is[t]= 1 if i<is_cut else 0
        t+=1
        if exit_i<=i: exit_i=i+1
        i=exit_i
    return o_pnl[:t],o_grs[:t],o_ts[:t],o_is[:t]


def build(pair):
    pipv=0.01 if pair.endswith("JPY") else 0.0001
    df=duckdb.sql(f"SELECT epoch(timestamp)::BIGINT ts, open, high, low, close, (ask_c-bid_c) sp, volume "
                  f"FROM 'data/s5_ohlc/{pair}_S5_BA.parquet' WHERE ask_c>bid_c ORDER BY timestamp").df()
    g=lambda c,t: df[c].to_numpy(t)
    ts=g("ts",np.int64); opn=g("open",np.float64); high=g("high",np.float64)
    low=g("low",np.float64); close=g("close",np.float64); sp=g("sp",np.float64); vol=g("volume",np.float64)
    cs=np.cumsum(sp); regime=np.empty_like(sp)
    regime[:K_SPREAD]=cs[:K_SPREAD]/(np.arange(K_SPREAD)+1); regime[K_SPREAD:]=(cs[K_SPREAD:]-cs[:-K_SPREAD])/K_SPREAD
    vmean=pd.Series(vol).rolling(VW).mean().shift(1).values
    vrel=np.nan_to_num(np.where(vmean>0,vol/vmean,1.0),nan=1.0)
    tindex=pd.to_datetime(ts,unit="s",utc=True)
    h1=(pd.DataFrame({"high":high,"low":low,"close":close},index=tindex)
        .resample("1h").agg({"high":"max","low":"min","close":"last"}).dropna())
    _,_,act_h,act_l,_=compute_swing_features(h1.high.values,h1.low.values,h1.close.values)
    a_h1=hatr(h1.high.values,h1.low.values,h1.close.values,14)
    h1_start=(h1.index.view("int64")//1_000_000_000).astype(np.int64)
    pos=np.clip(np.searchsorted(h1_start,ts,side="right")-2,0,len(act_h)-1)
    actH=np.nan_to_num(act_h[pos],nan=0.0); actL=np.nan_to_num(act_l[pos],nan=0.0)
    atr=np.nan_to_num(a_h1[pos],nan=0.0)
    is_cut=int(len(close)*IS_FRAC)
    calm_thr=float(np.nanmedian(regime[:is_cut]))       # per-pair IS-median trailing spread (price)
    return dict(opn=opn,high=high,low=low,close=close,sp=sp,regime=regime,vrel=vrel,actH=actH,
                actL=actL,atr=atr,ts=ts,is_cut=is_cut,calm_thr=calm_thr,pipv=pipv)


def main():
    store={g:[] for g in GRID}
    for pair in PAIRS:
        try: d=build(pair)
        except Exception as e: print(f"  skip {pair}: {e}",flush=True); continue
        for g in GRID:
            kd,kt,ktp,v=g
            pnl,grs,tss,isis=run(d["opn"],d["high"],d["low"],d["close"],d["sp"],d["regime"],d["vrel"],
                                 d["actH"],d["actL"],d["atr"],d["ts"],kd,kt,ktp,v,d["calm_thr"],M,d["is_cut"],d["pipv"])
            store[g].append((pair,pnl,grs,tss,isis))
        n0=len(store[GRID[0]][-1][1])
        print(f"  {pair}: done ({n0:,} trades @ {GRID[0]})  calm_thr={d['calm_thr']/d['pipv']:.2f}p",flush=True)
        del d; gc.collect()

    def pool(g):
        P=np.concatenate([x[1] for x in store[g]]); G=np.concatenate([x[2] for x in store[g]])
        T=np.concatenate([x[3] for x in store[g]]); I=np.concatenate([x[4] for x in store[g]]).astype(bool)
        return P,G,T,I
    print("\n=== grid (IS-select on NET IS mean) ===")
    print(f"{'kD':>4}{'kT':>5}{'kTP':>5}{'vol':>4} | {'n':>7} {'IS net':>8} {'OOS net':>8} {'IS grs':>7} {'OOS grs':>8} {'pairs+net':>9}")
    isscore={}
    for g in GRID:
        P,G,T,I=pool(g); isscore[g]=P[I].mean() if I.sum() else -9
        # per-pair net+ count (oos)
        npos=0
        for (pr,pn,gr,ts2,ii) in store[g]:
            ii=ii.astype(bool); o=pn[~ii]
            if len(o)>=20 and o.mean()>0: npos+=1
        print(f"{g[0]:>4.1f}{g[1]:>5.1f}{g[2]:>5.1f}{('hi' if g[3] else 'all'):>4} | {len(P):>7,} "
              f"{P[I].mean():>+8.3f} {P[~I].mean():>+8.3f} {G[I].mean():>+7.3f} {G[~I].mean():>+8.3f} {npos:>7}/12")
    best=max(GRID,key=lambda g:isscore[g]); P,G,T,I=pool(best); oos=P[~I]; oosg=G[~I]
    print(f"\n=== IS-SELECTED: kD={best[0]} kT={best[1]} kTP={best[2]} vol={'hi' if best[3] else 'all'} ===")
    print(f"  NET  OOS {oos.mean():+.3f} WR{100*(oos>0).mean():.1f}%   GROSS OOS {oosg.mean():+.3f}")
    b=np.array([RNG.choice(oos,len(oos),replace=True).mean() for _ in range(3000)])
    bg=np.array([RNG.choice(oosg,len(oosg),replace=True).mean() for _ in range(3000)])
    print(f"  MC NET P(<=0)={(b<=0).mean():.4f}   MC GROSS P(<=0)={(bg<=0).mean():.4f}")
    order=np.argsort(T[~I]); po=oos[order]; go=oosg[order]; e=[0,len(po)//3,2*len(po)//3,len(po)]
    print("  OOS WF NET  : "+"  ".join(f"T{k+1} {po[e[k]:e[k+1]].mean():+.3f}" for k in range(3)))
    print("  OOS WF GROSS: "+"  ".join(f"T{k+1} {go[e[k]:e[k+1]].mean():+.3f}" for k in range(3)))
    print("  per-pair OOS:")
    npos=ngpos=npairs=0
    for (pr,pn,gr,ts2,ii) in store[best]:
        ii=ii.astype(bool); o=pn[~ii]; og=gr[~ii]
        if len(o)<20: print(f"    {pr:<9} (thin n_oos={len(o)})"); continue
        npairs+=1; npos+=o.mean()>0; ngpos+=og.mean()>0
        print(f"    {pr:<9} net {o.mean():+6.3f}  gross {og.mean():+6.3f}  n_oos {len(o):,}")
    print(f"  pairs OOS+: NET {npos}/{npairs}  GROSS {ngpos}/{npairs}")


if __name__=="__main__":
    main()
