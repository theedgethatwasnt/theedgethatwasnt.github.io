"""
Path B: take the PROVEN entry (first-touch H4 reversion + low-volume filter) and replace
its ATR target/SL exit with OUR exit toolkit — an ATR-scaled TRAILING STOP (+ optional
ATR-scaled TP) + time cap — re-optimized at the swing (H4) horizon. Does the toolkit
transfer / improve on the validated +9.49 p/trade loVol OOS baseline?

Entry identical to first_touch_v2.py (L=25, eps=12p, fade first touch, touches<=1).
Exit per trade (H4 bars, SL/trail checked before TP = pessimistic):
  - trailing stop = k_trail * ATR(14,H4) from the high/low-water mark
  - optional fixed TP = k_tp * ATR toward reversion (k_tp=0 -> trail only)
  - time cap Hcap H4 bars
Net spread. loVol = vrel <= IS-median. IS-select on portfolio IS exp, confirm OOS+MC.
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


def trades(con,pair,k_trail,k_tp):
    pip,sp=PAIRS[pair]; r=load(con,pair)
    h=r.high.values; l=r.low.values; c=r.close.values; v=r.volume.values; ts=r.index.values
    n=len(c); eps=EPS*pip; a=atr(h,l,c,14)
    Rs=pd.Series(h).rolling(L).max().shift(1).values; Ss=pd.Series(l).rolling(L).min().shift(1).values
    vrel=v/pd.Series(v).rolling(VW).mean().shift(1).values
    out=[]; last=-10; is_cut=int(n*IS_FRAC)
    for t in range(L+1,n-HCAP-1):
        if np.isnan(Rs[t]) or a[t]<=0 or np.isnan(vrel[t]) or t-last<HCAP: continue
        R=Rs[t]; S=Ss[t]
        up=h[t]>=R-eps and h[t-1]<R-eps and c[t]<=R
        dn=l[t]<=S+eps and l[t-1]>S+eps and c[t]>=S
        if not(up or dn): continue
        if up: tch=int(np.sum(h[t-L:t]>=R-eps)); d=-1
        else:  tch=int(np.sum(l[t-L:t]<=S+eps)); d=+1
        if tch>1: continue
        entry=c[t]; ae=a[t]; trail=k_trail*ae; tp=k_tp*ae
        exitpx=c[t+HCAP]                                   # default time cap
        if d==-1:                                          # SHORT: trail above, TP below
            lwm=entry
            for j in range(t+1,t+HCAP+1):
                if h[j] >= lwm+trail: exitpx=lwm+trail; break       # trail (loss-cap) first
                if k_tp>0 and l[j] <= entry-tp: exitpx=entry-tp; break
                if l[j] < lwm: lwm=l[j]
        else:                                              # LONG: trail below, TP above
            hwm=entry
            for j in range(t+1,t+HCAP+1):
                if l[j] <= hwm-trail: exitpx=hwm-trail; break
                if k_tp>0 and h[j] >= entry+tp: exitpx=entry+tp; break
                if h[j] > hwm: hwm=h[j]
        pnl=d*(exitpx-entry)/pip - sp
        out.append((ts[t], pnl, t<is_cut, vrel[t])); last=t
    return out


def main():
    con=duckdb.connect()
    GRID=[(k_tr,k_tp) for k_tr in (1.0,1.5,2.0,2.5,3.0) for k_tp in (0.0,1.5,2.0,3.0)]
    print(f"Path B — first-touch H4 + loVol, exit = {{k_trail*ATR trailing + k_tp*ATR TP}} (k_tp=0 -> trail only), "
          f"Hcap={HCAP} H4-bars. Net spread.\n")
    print(f"Baseline (validated ATR target/SL exit, tgt2/sl2): loVol OOS +9.49p, MC P=0.018, 7/12 pairs.\n")
    cache={g:{p:trades(con,p,g[0],g[1]) for p in PAIRS} for g in GRID}
    cache={g:{p:t for p,t in per.items() if len(t)>=40} for g,per in cache.items()}

    print(f"{'k_trail':>7} {'k_tp':>5} | {'allOOS':>7} | {'loVolIS':>8} {'loVolOOS':>9} {'n':>5} {'WR':>4} "
          f"{'MC P':>6} {'pairs+':>6}")
    print("-"*72)
    rows=[]
    for g in GRID:
        per=cache[g]
        if not per: continue
        allt=[x for p in per for x in per[p]]
        pn=np.array([z[1] for z in allt]); isf=np.array([z[2] for z in allt]); vr=np.array([z[3] for z in allt])
        # loVol split at IS-median vrel (per the validated method)
        vmed=np.nanmedian(vr[isf])
        lo=vr<=vmed
        oos_all=pn[~isf].mean()
        lo_is=pn[lo&isf]; lo_oos=pn[lo&~isf]
        if len(lo_oos)<30: continue
        b=np.array([RNG.choice(lo_oos,len(lo_oos),replace=True).mean() for _ in range(2000)])
        # per-pair loVol OOS positivity
        npos=0; npairs=0
        for p in per:
            vv=np.array([x[3] for x in per[p]]); pp=np.array([x[1] for x in per[p]]); ii=np.array([x[2] for x in per[p]])
            vm=np.nanmedian(vv[ii]) if ii.sum() else np.nan
            o=pp[(vv<=vm)&~ii]
            if len(o)>=10:
                npairs+=1; npos+= 1 if o.mean()>0 else 0
        rows.append((g,oos_all,lo_is.mean(),lo_oos.mean(),len(lo_oos),(lo_oos>0).mean()*100,(b<=0).mean(),npos,npairs))
        print(f"{g[0]:>7.1f} {g[1]:>5.1f} | {oos_all:>+7.2f} | {lo_is.mean():>+8.2f} {lo_oos.mean():>+9.2f} "
              f"{len(lo_oos):>5} {(lo_oos>0).mean()*100:>3.0f}% {(b<=0).mean():>6.3f} {npos:>3}/{npairs:<2}")
    best=max(rows,key=lambda r:r[3])
    print(f"\nBEST loVol-OOS exit: k_trail={best[0][0]} k_tp={best[0][1]} -> "
          f"loVol OOS {best[3]:+.2f}p (n{best[4]}, WR{best[5]:.0f}%, MC P={best[6]:.3f}, pairs+ {best[7]}/{best[8]})")
    con.close()


if __name__=="__main__":
    main()
