"""
Level-STRENGTH structural fade (RACS-style): instead of fading the single last swing pivot,
weight each H1 swing level by CLUSTERING + RECENCY DECAY, and bucket fades by level strength.

Strength of a swing at (idx, price): 1 + sum over PRIOR same-type swings j within
|price_j - price| <= TOL*ATR of exp(-(idx-idx_j)/TAU_bars).  (recency decay = your 1/time-ago)
=> a level touched many times recently scores high; an old lone pivot scores ~1.

Fade entry = same validated scaled config (kD=0.7 D, trail=1.2*ATR, TP=0.5*ATR, hi-vol,
per-pair IS-median calm gate). Record the faded level's strength; bucket OOS by strength.
12 pairs, IS/OOS 60/40, NET (real per-bar spread) + GROSS. Memory-safe.
"""
import sys, gc
import numpy as np, pandas as pd, duckdb
from numba import njit
sys.path.insert(0, "/path/to/projects/fx-core")
from lib.swing_indicators import topsbots_swings

PAIRS=["EUR_USD","USD_JPY","GBP_USD","AUD_USD","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY",
       "CHF_JPY","NZD_JPY","NZD_USD","EUR_GBP"]
K_SPREAD=12; HOLD_BARS=720; GAP_SECS=60; VW=240; VHI=1.20; M=24; IS_FRAC=0.6
KD=0.7; KT=1.2; KTP=0.5                     # best scaled exit
TAU=400.0                                   # recency decay (H1 bars; ~3-4 weeks)
TOL=0.6                                     # cluster tolerance in ATR units
RNG=np.random.default_rng(17)


def hatr(h,l,c,n=14):
    pc=np.empty_like(c); pc[0]=c[0]; pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values


def levels_with_strength(h1h,h1l,h1c):
    """Per-H1-bar active resistance/support value + recency-decayed cluster strength."""
    nH=len(h1c); a=hatr(h1h,h1l,h1c,14)
    sig=topsbots_swings(h1h,h1l)                       # (idx,'H'|'L',value), confirmed (1-bar lookahead)
    # strength per swing
    strg={}
    for typ in ('H','L'):
        S=[(idx,val) for (idx,tt,val) in sig if tt==typ]
        idxs=np.array([s[0] for s in S]); vals=np.array([s[1] for s in S])
        for k in range(len(S)):
            if k==0: strg[(typ,idxs[k])]=1.0; continue
            jp=idxs[:k]; jv=vals[:k]
            tol=TOL*a[idxs[k]] if a[idxs[k]]>0 else TOL*1e-4
            near=np.abs(jv-vals[k])<=tol
            w=np.exp(-(idxs[k]-jp[near])/TAU).sum() if near.any() else 0.0
            strg[(typ,idxs[k])]=1.0+float(w)
    swby={s[0]:(s[1],s[2]) for s in sig}
    act_h=np.full(nH,np.nan); act_l=np.full(nH,np.nan)
    str_h=np.zeros(nH); str_l=np.zeros(nH)
    ch=cl=np.nan; csh=csl=0.0
    for i in range(nH):
        if i in swby:
            tt,vv=swby[i]
            if tt=='H': ch=vv; csh=strg[('H',i)]
            else:       cl=vv; csl=strg[('L',i)]
        act_h[i]=ch; str_h[i]=csh; act_l[i]=cl; str_l[i]=csl
    return act_h,act_l,str_h,str_l,a


@njit(cache=True)
def run(opn,high,low,close,spread,regime,vrel,actH,actL,strH,strL,atr,ts,calm_thr,is_cut,pipv):
    n=high.shape[0]; mt=n//4+8
    o_pnl=np.empty(mt,np.float64); o_grs=np.empty(mt,np.float64); o_ts=np.empty(mt,np.int64)
    o_is=np.empty(mt,np.int64); o_str=np.empty(mt,np.float64)
    t=0; i=max(M+1,VW+1)
    while i<n-1:
        if regime[i]>calm_thr: i+=1; continue
        aH=actH[i]; aL=actL[i]; ae=atr[i]
        if aH<=0.0 or aL<=0.0 or ae<=0.0: i+=1; continue
        Dp=KD*ae; trail=KT*ae
        c=close[i]; dR=aH-c; dS=c-aL
        near_R= dR>=0.0 and dR<=Dp; near_S= dS>=0.0 and dS<=Dp
        if not(near_R or near_S): i+=1; continue
        at_R= near_R and (not near_S or dR<=dS)
        d= -1.0 if at_R else 1.0
        lvl_str= strH[i] if at_R else strL[i]
        s=0.0
        for q in range(i-M+1,i+1): s+=vrel[q]
        if s/M<VHI: i+=1; continue                       # hi-vol gate
        entry=c; sp=spread[i]/pipv; tp_pips=KTP*ae/pipv; tp_off=KTP*ae+spread[i]
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
        o_pnl[t]=pnl; o_grs[t]=pnl+sp; o_ts[t]=ts[i]; o_is[t]=1 if i<is_cut else 0; o_str[t]=lvl_str
        t+=1
        if exit_i<=i: exit_i=i+1
        i=exit_i
    return o_pnl[:t],o_grs[:t],o_ts[:t],o_is[:t],o_str[:t]


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
    act_h,act_l,str_h,str_l,a_h1=levels_with_strength(h1.high.values,h1.low.values,h1.close.values)
    h1_start=(h1.index.view("int64")//1_000_000_000).astype(np.int64)
    pos=np.clip(np.searchsorted(h1_start,ts,side="right")-2,0,len(act_h)-1)
    actH=np.nan_to_num(act_h[pos],nan=0.0); actL=np.nan_to_num(act_l[pos],nan=0.0)
    strH=np.nan_to_num(str_h[pos],nan=0.0); strL=np.nan_to_num(str_l[pos],nan=0.0)
    atr=np.nan_to_num(a_h1[pos],nan=0.0); is_cut=int(len(close)*IS_FRAC)
    calm_thr=float(np.nanmedian(regime[:is_cut]))
    return dict(opn=opn,high=high,low=low,close=close,sp=sp,regime=regime,vrel=vrel,actH=actH,actL=actL,
                strH=strH,strL=strL,atr=atr,ts=ts,is_cut=is_cut,calm_thr=calm_thr,pipv=pipv)


def main():
    allp=[]; allg=[]; allt=[]; alli=[]; alls=[]
    for pair in PAIRS:
        try: d=build(pair)
        except Exception as e: print(f"  skip {pair}: {e}",flush=True); continue
        pn,gr,tss,isis,strn=run(d["opn"],d["high"],d["low"],d["close"],d["sp"],d["regime"],d["vrel"],
                                d["actH"],d["actL"],d["strH"],d["strL"],d["atr"],d["ts"],d["calm_thr"],d["is_cut"],d["pipv"])
        allp.append(pn); allg.append(gr); allt.append(tss); alli.append(isis); alls.append(strn)
        print(f"  {pair}: {len(pn):,} trades  median_str={np.median(strn):.2f}  max_str={strn.max():.1f}",flush=True)
        del d; gc.collect()
    P=np.concatenate(allp); G=np.concatenate(allg); I=np.concatenate(alli).astype(bool); ST=np.concatenate(alls)
    print(f"\nTotal {len(P):,} trades. Exit kD={KD} trail={KT}*ATR TP={KTP}*ATR hi-vol. TAU={TAU} TOL={TOL}*ATR.")
    print(f"Overall: NET OOS {P[~I].mean():+.3f}  GROSS OOS {G[~I].mean():+.3f}\n")
    # bucket OOS by level strength (quintiles of IS strength)
    qe=np.unique(np.round(np.quantile(ST[I],[0,.2,.4,.6,.8,1.0]),2))
    print(f"{'strength band':>16} | {'n_oos':>7} {'NET oos':>8} {'GROSS oos':>9} {'net WR':>7}")
    print("-"*60)
    for a_,b_ in zip(qe[:-1],qe[1:]):
        m=((ST>=a_)&(ST<b_)) if b_!=qe[-1] else ((ST>=a_)&(ST<=b_))
        o=P[m&~I]; og=G[m&~I]
        if len(o)<50: continue
        bN=np.array([RNG.choice(o,len(o),replace=True).mean() for _ in range(1500)])
        print(f"{a_:6.2f}-{b_:<8.2f} | {len(o):>7,} {o.mean():>+8.3f} {og.mean():>+9.3f} {100*(o>0).mean():>6.1f}%  "
              f"netP(<=0)={(bN<=0).mean():.3f}")
    # top-strength subset (>= IS 80th pct): net + breadth + MC
    thr=np.quantile(ST[I],0.8)
    hi=ST>=thr; oN=P[hi&~I]; oG=G[hi&~I]
    bN=np.array([RNG.choice(oN,len(oN),replace=True).mean() for _ in range(3000)])
    print(f"\nTOP-strength subset (>= IS p80 = {thr:.2f}): n_oos {len(oN):,}  "
          f"NET {oN.mean():+.3f} (MC P<=0 {(bN<=0).mean():.4f})  GROSS {oG.mean():+.3f}")
    # per-pair net on top subset
    npos=npairs=0; off=0
    for k,pair in enumerate([p for p in PAIRS]):
        pn=allp[k]; ii=alli[k].astype(bool); st=alls[k]
        o=pn[(st>=thr)&~ii]
        if len(o)>=20: npairs+=1; npos+=o.mean()>0
    print(f"  top-strength pairs OOS net-positive: {npos}/{npairs}")
    print("\nRead: if NET rises with strength and top bucket -> ~0+, clustering/recency sharpens the fade.")


if __name__=="__main__":
    main()
