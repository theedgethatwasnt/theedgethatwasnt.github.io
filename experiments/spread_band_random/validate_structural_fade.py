"""
VALIDATE the structural-fade HF strategy (fade H1 S/R + hi-vol + calm band, trail5/TP2).
Disciplined gates (the traps we keep hitting):
  - IS/OOS 60/40 PER PAIR (temporal). Select (D, vol-side) on IS only; OOS = confirmation.
  - PER-PAIR breadth across 12 pairs (avoid single-pair drift).
  - MC bootstrap of OOS per-trade expectancy, P(<=0).
  - Temporal WF (3 thirds of pooled OOS).
  - NET (after spread) AND GROSS (pre-spread) — is the DIRECTIONAL signal real even if spread kills net?
Memory-safe: one pair's S5 in RAM at a time; keep only small per-trade arrays.
"""
import sys, gc
import numpy as np, pandas as pd, duckdb
from numba import njit
sys.path.insert(0, "/path/to/projects/fx-core")
from lib.swing_indicators import compute_swing_features

PAIRS=["EUR_USD","USD_JPY","GBP_USD","AUD_USD","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY",
       "CHF_JPY","NZD_JPY","NZD_USD","EUR_GBP"]
K_SPREAD=12; HOLD_BARS=720; GAP_SECS=60; TRAIL=5.0; TP=2.0; CALM_THR=1.50
VW=240; VHI=1.20; M=24; IS_FRAC=0.6
GRID=[(D,vol) for D in (3.0,5.0,8.0) for vol in (0,1)]   # vol 0=all, 1=hi
RNG=np.random.default_rng(17)


@njit(cache=True)
def run(opn,high,low,close,spread,regime,vrel,actH,actL,ts,D,volside,calm_thr,trail_pips,tp_pips,M_,is_cut,pipv):
    n=high.shape[0]; trail=trail_pips*pipv; mt=n//4+8
    o_pnl=np.empty(mt,np.float64); o_grs=np.empty(mt,np.float64); o_ts=np.empty(mt,np.int64); o_is=np.empty(mt,np.int64)
    t=0; i=max(M_+1,VW+1)
    while i<n-1:
        if regime[i]/pipv>calm_thr: i+=1; continue
        aH=actH[i]; aL=actL[i]
        if aH<=0.0 or aL<=0.0: i+=1; continue
        c=close[i]; dR=(aH-c)/pipv; dS=(c-aL)/pipv
        near_R= dR>=0.0 and dR<=D; near_S= dS>=0.0 and dS<=D
        if not(near_R or near_S): i+=1; continue
        at_R= near_R and (not near_S or dR<=dS)
        d= -1.0 if at_R else 1.0                 # sfade
        if volside>0:
            s=0.0
            for q in range(i-M_+1,i+1): s+=vrel[q]
            if s/M_<VHI: i+=1; continue
        entry=c; sp=spread[i]/pipv; tp_off=(tp_pips+sp)*pipv
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
    h1_start=(h1.index.view("int64")//1_000_000_000).astype(np.int64)
    pos=np.clip(np.searchsorted(h1_start,ts,side="right")-2,0,len(act_h)-1)
    actH=np.nan_to_num(act_h[pos],nan=0.0); actL=np.nan_to_num(act_l[pos],nan=0.0)
    is_cut=int(len(close)*IS_FRAC)
    return dict(opn=opn,high=high,low=low,close=close,sp=sp,regime=regime,vrel=vrel,
                actH=actH,actL=actL,ts=ts,is_cut=is_cut,pipv=pipv)


def main():
    # accumulate per-pair per-config trades (small arrays only)
    store={g:[] for g in GRID}     # g -> list of (pair, pnl, grs, ts, isis)
    for pair in PAIRS:
        try:
            d=build(pair)
        except Exception as e:
            print(f"  skip {pair}: {e}"); continue
        for (D,vol) in GRID:
            pnl,grs,tss,isis=run(d["opn"],d["high"],d["low"],d["close"],d["sp"],d["regime"],d["vrel"],
                                 d["actH"],d["actL"],d["ts"],D,vol,CALM_THR,TRAIL,TP,M,d["is_cut"],d["pipv"])
            store[(D,vol)].append((pair,pnl,grs,tss,isis))
        print(f"  {pair}: done ({len(store[GRID[0]][-1][1]):,} trades @ D3/all)", flush=True)
        del d; gc.collect()

    # pool + IS-select on net IS expectancy
    def pool(g):
        P=np.concatenate([x[1] for x in store[g]]); G=np.concatenate([x[2] for x in store[g]])
        T=np.concatenate([x[3] for x in store[g]]); I=np.concatenate([x[4] for x in store[g]]).astype(bool)
        return P,G,T,I
    print("\n=== config grid (IS-select on NET IS mean) ===")
    print(f"{'D':>4} {'vol':>4} | {'n':>7} {'IS net':>8} {'OOS net':>8} {'IS gross':>9} {'OOS gross':>10}")
    isscore={}
    for g in GRID:
        P,G,T,I=pool(g)
        isscore[g]=P[I].mean()
        print(f"{g[0]:>4.0f} {('hi' if g[1] else 'all'):>4} | {len(P):>7,} {P[I].mean():>+8.3f} {P[~I].mean():>+8.3f} "
              f"{G[I].mean():>+9.3f} {G[~I].mean():>+10.3f}")
    best=max(GRID,key=lambda g:isscore[g])
    P,G,T,I=pool(best)
    oos=P[~I]; oosg=G[~I]
    print(f"\n=== IS-SELECTED config: D={best[0]:.0f} vol={'hi' if best[1] else 'all'} ===")
    print(f"  trades {len(P):,}  (IS {I.sum():,} / OOS {(~I).sum():,})")
    print(f"  NET   : IS {P[I].mean():+.3f}  OOS {oos.mean():+.3f}  OOS WR {100*(oos>0).mean():.1f}%")
    print(f"  GROSS : IS {G[I].mean():+.3f}  OOS {oosg.mean():+.3f}  (directional edge pre-spread)")
    b=np.array([RNG.choice(oos,len(oos),replace=True).mean() for _ in range(3000)])
    bg=np.array([RNG.choice(oosg,len(oosg),replace=True).mean() for _ in range(3000)])
    print(f"  MC OOS NET  : 95%CI[{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}] P(<=0)={(b<=0).mean():.4f}")
    print(f"  MC OOS GROSS: 95%CI[{np.percentile(bg,2.5):+.3f},{np.percentile(bg,97.5):+.3f}] P(<=0)={(bg<=0).mean():.4f}")
    # temporal WF (3 thirds of OOS by ts)
    order=np.argsort(T[~I]); to=T[~I][order]; po=oos[order]; go=oosg[order]
    e=[0,len(po)//3,2*len(po)//3,len(po)]
    print("  OOS WF thirds NET  : "+"  ".join(f"T{k+1} {po[e[k]:e[k+1]].mean():+.3f}" for k in range(3)))
    print("  OOS WF thirds GROSS: "+"  ".join(f"T{k+1} {go[e[k]:e[k+1]].mean():+.3f}" for k in range(3)))
    # per-pair OOS
    print("  per-pair OOS net:")
    npos=ngpos=0
    for (pair,pnl,grs,tss,isis) in store[best]:
        ii=isis.astype(bool); o=pnl[~ii]; og=grs[~ii]
        if len(o)<20: continue
        npos+= 1 if o.mean()>0 else 0; ngpos+= 1 if og.mean()>0 else 0
        print(f"    {pair:<9} net {o.mean():+6.3f}  gross {og.mean():+6.3f}  (n_oos {len(o):,})")
    print(f"  pairs OOS-positive: NET {npos}/12   GROSS {ngpos}/12")
    print("\nGATE: deployable only if OOS net>0, broad, MC<0.05. Gross>0 broad => real signal, spread-blocked.")


if __name__=="__main__":
    main()
