"""
Structural-fade TIMEFRAME sweep: build swing S/R on TF in {60,90,120,150,180,210,240}min.
Hypothesis: higher-TF S/R => bigger reversion moves vs (fixed) spread => gross edge may
outgrow spread => net toward 0+. Fixed exit (validated): kD=0.7 D, trail=1.2*ATR, TP=0.5*ATR,
hi-vol, per-pair IS-median calm gate. 12 pairs, IS/OOS 60-40, real per-bar spread, gross+net.
"""
import sys, gc
import numpy as np, pandas as pd, duckdb
from numba import njit
sys.path.insert(0, "/path/to/projects/fx-core")
from lib.swing_indicators import compute_swing_features

PAIRS=["EUR_USD","USD_JPY","GBP_USD","AUD_USD","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY",
       "CHF_JPY","NZD_JPY","NZD_USD","EUR_GBP"]
K_SPREAD=12; HOLD_BARS=720; GAP_SECS=60; VW=240; VHI=1.20; M=24; IS_FRAC=0.6
KD=0.7; KT=1.2; KTP=0.5
TFS=[60,90,120,150,180,210,240]
RNG=np.random.default_rng(17)


def hatr(h,l,c,n=14):
    pc=np.empty_like(c); pc[0]=c[0]; pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values


@njit(cache=True)
def run(opn,high,low,close,spread,regime,vrel,actH,actL,atr,ts,calm_thr,is_cut,pipv):
    n=high.shape[0]; mt=n//4+8
    o_pnl=np.empty(mt,np.float64); o_grs=np.empty(mt,np.float64); o_ts=np.empty(mt,np.int64); o_is=np.empty(mt,np.int64)
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
        s=0.0
        for q in range(i-M+1,i+1): s+=vrel[q]
        if s/M<VHI: i+=1; continue
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
        o_pnl[t]=pnl; o_grs[t]=pnl+sp; o_ts[t]=ts[i]; o_is[t]=1 if i<is_cut else 0
        t+=1
        if exit_i<=i: exit_i=i+1
        i=exit_i
    return o_pnl[:t],o_grs[:t],o_ts[:t],o_is[:t]


def load_s5(pair):
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
    is_cut=int(len(close)*IS_FRAC); calm_thr=float(np.nanmedian(regime[:is_cut]))
    tindex=pd.to_datetime(ts,unit="s",utc=True)
    return dict(ts=ts,opn=opn,high=high,low=low,close=close,sp=sp,regime=regime,vrel=vrel,
                is_cut=is_cut,calm_thr=calm_thr,pipv=pipv,tindex=tindex)


def structure(d, tf_min):
    base=pd.DataFrame({"high":d["high"],"low":d["low"],"close":d["close"]},index=d["tindex"])
    tf=base.resample(f"{tf_min}min").agg({"high":"max","low":"min","close":"last"}).dropna()
    _,_,act_h,act_l,_=compute_swing_features(tf.high.values,tf.low.values,tf.close.values)
    a_tf=hatr(tf.high.values,tf.low.values,tf.close.values,14)
    tf_start=(tf.index.view("int64")//1_000_000_000).astype(np.int64)
    pos=np.clip(np.searchsorted(tf_start,d["ts"],side="right")-2,0,len(act_h)-1)
    return (np.nan_to_num(act_h[pos],nan=0.0),np.nan_to_num(act_l[pos],nan=0.0),np.nan_to_num(a_tf[pos],nan=0.0))


def main():
    span=None
    store={tf:[] for tf in TFS}     # tf -> list of (pnl,grs,ts,is)
    for pair in PAIRS:
        try: d=load_s5(pair)
        except Exception as e: print(f"  skip {pair}: {e}",flush=True); continue
        if span is None: span=(d["ts"][-1]-d["ts"][0])/86400
        for tf in TFS:
            actH,actL,atr=structure(d,tf)
            pn,gr,tss,isis=run(d["opn"],d["high"],d["low"],d["close"],d["sp"],d["regime"],d["vrel"],
                               actH,actL,atr,d["ts"],d["calm_thr"],d["is_cut"],d["pipv"])
            store[tf].append((pn,gr,tss,isis))
            del actH,actL,atr
        print(f"  {pair}: done all TFs",flush=True); del d; gc.collect()

    print(f"\nStructural fade by S/R timeframe (12 pairs, IS/OOS 60-40, real per-bar spread).")
    print(f"Exit: D={KD}*ATR trail={KT}*ATR TP={KTP}*ATR hi-vol.\n")
    print(f"{'TF(min)':>7} | {'n':>7} {'/day':>5} | {'IS net':>7} {'OOS net':>8} {'OOS grs':>8} {'grsMC':>6} {'netMC':>6} | {'WF gross':>22} | {'pairs+':>7}")
    print("-"*108)
    for tf in TFS:
        P=np.concatenate([x[0] for x in store[tf]]); G=np.concatenate([x[1] for x in store[tf]])
        T=np.concatenate([x[2] for x in store[tf]]); I=np.concatenate([x[3] for x in store[tf]]).astype(bool)
        oos=P[~I]; oosg=G[~I]
        bN=np.array([RNG.choice(oos,len(oos),replace=True).mean() for _ in range(1500)])
        bG=np.array([RNG.choice(oosg,len(oosg),replace=True).mean() for _ in range(1500)])
        order=np.argsort(T[~I]); go=oosg[order]; e=[0,len(go)//3,2*len(go)//3,len(go)]
        wf="  ".join(f"{go[e[k]:e[k+1]].mean():+.2f}" for k in range(3))
        npos=ngpos=npairs=0
        for (pn,gr,tss,isis) in store[tf]:
            ii=isis.astype(bool); o=pn[~ii]; og=gr[~ii]
            if len(o)>=20: npairs+=1; npos+=o.mean()>0; ngpos+=og.mean()>0
        print(f"{tf:>7} | {len(P):>7,} {len(P)/span/12:>5.1f} | {P[I].mean():>+7.3f} {oos.mean():>+8.3f} "
              f"{oosg.mean():>+8.3f} {(bG<=0).mean():>6.3f} {(bN<=0).mean():>6.3f} | {wf:>22} | net{npos}/grs{ngpos}")
    print("\n/day = per-pair avg. Read: does OOS net rise (toward 0+) as TF increases? Gross should grow with TF.")


if __name__=="__main__":
    main()
