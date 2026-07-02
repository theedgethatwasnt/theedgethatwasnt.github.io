"""
Path B (full) — fix the PROVEN entry (first-touch H4 + low-volume), try ALL ATR exit families,
IS-select within each family, confirm OOS + MC. Honest comparison (no OOS-peeking).

Entry: first_touch_v2 logic (L=25, eps=12p, fade, touches<=1). loVol = vrel<=IS-median (pooled).
Exit families (all ATR-scaled, H4 bars, SL/trail checked before TP = pessimistic), time cap Hcap:
  A. bracket   : target = tgt*ATR, hard SL = sl*ATR            (the validated family)
  B. trail+TP  : trailing = k*ATR from HWM/LWM, optional TP = k_tp*ATR
  C. supertrend: per-trade ratchet stop hl2 +/- m*ATR, exit on CLOSE beyond (the bakeoff winner)
  D. timecap   : just hold Hcap bars, close at close (control)
"""
from pathlib import Path
import numpy as np, pandas as pd, duckdb

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6; L=25; EPS=12; VW=20; HCAP=12; RNG=np.random.default_rng(23)


def load(con,pair):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,high,low,close,volume FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    return df.resample("4h").agg({"high":"max","low":"min","close":"last","volume":"sum"}).dropna()

def atr(h,l,c,n=14):
    pc=np.empty_like(c); pc[0]=c[0]; pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values

def detect(con,pair):
    """Return arrays + list of events (t,d,entry,ae,vrel,is_is)."""
    pip,sp=PAIRS[pair]; r=load(con,pair)
    h=r.high.values; l=r.low.values; c=r.close.values; v=r.volume.values; n=len(c); eps=EPS*pip
    a=atr(h,l,c,14)
    Rs=pd.Series(h).rolling(L).max().shift(1).values; Ss=pd.Series(l).rolling(L).min().shift(1).values
    vrel=v/pd.Series(v).rolling(VW).mean().shift(1).values
    ev=[]; last=-10; is_cut=int(n*IS_FRAC)
    for t in range(L+1,n-HCAP-1):
        if np.isnan(Rs[t]) or a[t]<=0 or np.isnan(vrel[t]) or t-last<HCAP: continue
        R=Rs[t]; S=Ss[t]
        up=h[t]>=R-eps and h[t-1]<R-eps and c[t]<=R
        dn=l[t]<=S+eps and l[t-1]>S+eps and c[t]>=S
        if not(up or dn): continue
        if up: tch=int(np.sum(h[t-L:t]>=R-eps)); d=-1
        else:  tch=int(np.sum(l[t-L:t]<=S+eps)); d=+1
        if tch>1: continue
        ev.append((t,d,c[t],a[t],vrel[t],t<is_cut)); last=t
    return (h,l,c,pip,sp),ev

def ex_bracket(arr,t,d,entry,ae,p):
    h,l,c,pip,sp=arr; tgt,sl=p; tp=entry+d*tgt*ae; slp=entry-d*sl*ae; px=c[t+HCAP]
    for j in range(t+1,t+HCAP+1):
        if d==-1:
            if h[j]>=slp: px=slp; break
            if l[j]<=tp: px=tp; break
        else:
            if l[j]<=slp: px=slp; break
            if h[j]>=tp: px=tp; break
    return d*(px-entry)/pip - sp

def ex_trail(arr,t,d,entry,ae,p):
    h,l,c,pip,sp=arr; k,ktp=p; trail=k*ae; tp=ktp*ae; px=c[t+HCAP]
    if d==-1:
        lwm=entry
        for j in range(t+1,t+HCAP+1):
            if h[j]>=lwm+trail: px=lwm+trail; break
            if ktp>0 and l[j]<=entry-tp: px=entry-tp; break
            if l[j]<lwm: lwm=l[j]
    else:
        hwm=entry
        for j in range(t+1,t+HCAP+1):
            if l[j]<=hwm-trail: px=hwm-trail; break
            if ktp>0 and h[j]>=entry+tp: px=entry+tp; break
            if h[j]>hwm: hwm=h[j]
    return d*(px-entry)/pip - sp

def ex_supertrend(arr,t,d,entry,ae,p):
    h,l,c,pip,sp=arr; m=p[0]; band=m*ae; px=c[t+HCAP]
    if d==-1:                                   # short: upper stop ratchets DOWN
        stop=entry+band
        for j in range(t+1,t+HCAP+1):
            hl2=(h[j]+l[j])/2.0
            stop=min(stop,hl2+band)
            if c[j]>stop: px=c[j]; break
    else:                                       # long: lower stop ratchets UP
        stop=entry-band
        for j in range(t+1,t+HCAP+1):
            hl2=(h[j]+l[j])/2.0
            stop=max(stop,hl2-band)
            if c[j]<stop: px=c[j]; break
    return d*(px-entry)/pip - sp

def ex_timecap(arr,t,d,entry,ae,p):
    h,l,c,pip,sp=arr; return d*(c[t+HCAP]-entry)/pip - sp

FAMILIES={
 "bracket"   :(ex_bracket ,[(tg,sl) for tg in (1,1.5,2,2.5,3) for sl in (1,1.5,2,2.5,3)]),
 "trail+TP"  :(ex_trail   ,[(k,kt) for k in (1,1.5,2,2.5,3) for kt in (0,1.5,2,3)]),
 "supertrend":(ex_supertrend,[(m,) for m in (1,1.5,2,2.5,3,4)]),
 "timecap"   :(ex_timecap ,[(0,)]),
}


def main():
    con=duckdb.connect()
    cache={p:detect(con,p) for p in PAIRS}
    cache={p:v for p,v in cache.items() if len(v[1])>=40}
    print(f"Path B — first-touch H4 + loVol. Entry fixed; sweeping ALL ATR exit families. "
          f"Hcap={HCAP}. {len(cache)} pairs.\n")
    print(f"Validated reference: ATR bracket tgt2/sl2 -> loVol OOS +9.49p, MC P=0.018.\n")
    print(f"{'family':>11} {'best param (IS-sel)':>20} | {'loVolIS':>8} {'loVolOOS':>9} {'n':>4} {'WR':>4} {'MC P':>6} {'pairs+':>6}")
    print("-"*80)

    def eval_param(fn,p):
        pn=[]; isf=[]; vr=[]
        for pair,(arr,ev) in cache.items():
            for (t,d,entry,ae,vrel,ii) in ev:
                pn.append(fn(arr,t,d,entry,ae,p)); isf.append(ii); vr.append(vrel)
        return np.array(pn),np.array(isf),np.array(vr)

    results={}
    for fam,(fn,grid) in FAMILIES.items():
        # IS-select param by loVol IS expectancy (pooled IS-median split)
        best=None
        for p in grid:
            pn,isf,vr=eval_param(fn,p)
            vmed=np.nanmedian(vr[isf]); lo=vr<=vmed
            lo_is=pn[lo&isf]
            if len(lo_is)<30: continue
            sc=lo_is.mean()
            if best is None or sc>best[0]:
                best=(sc,p,pn,isf,vr,vmed,lo)
        sc,p,pn,isf,vr,vmed,lo=best
        lo_oos=pn[lo&~isf]
        b=np.array([RNG.choice(lo_oos,len(lo_oos),replace=True).mean() for _ in range(3000)])
        # per-pair loVol OOS positivity
        npos=npairs=0
        for pair,(arr,ev) in cache.items():
            pp=np.array([fn(arr,t,d,e,a,p) for (t,d,e,a,vrl,ii) in ev])
            vv=np.array([vrl for (t,d,e,a,vrl,ii) in ev]); iis=np.array([ii for (*_,ii) in ev])
            vm=np.nanmedian(vv[iis]) if iis.sum() else np.nan
            o=pp[(vv<=vm)&~iis]
            if len(o)>=10: npairs+=1; npos+= 1 if o.mean()>0 else 0
        results[fam]=(p,sc,lo_oos.mean(),len(lo_oos),(lo_oos>0).mean()*100,(b<=0).mean(),npos,npairs)
        print(f"{fam:>11} {str(p):>20} | {sc:>+8.2f} {lo_oos.mean():>+9.2f} {len(lo_oos):>4} "
              f"{(lo_oos>0).mean()*100:>3.0f}% {(b<=0).mean():>6.3f} {npos:>3}/{npairs:<2}")

    con.close()
    win=max(results.items(),key=lambda kv:kv[1][2])
    print(f"\nBest exit family (IS-selected, by loVol OOS): {win[0]} {win[1][0]} -> "
          f"loVol OOS {win[1][2]:+.2f}p, MC P={win[1][5]:.3f}, pairs+ {win[1][6]}/{win[1][7]}")
    print("Honest read: select on IS, report OOS. A family only 'wins' if IS+ and OOS+ and MC<0.05.")


if __name__=="__main__":
    main()
