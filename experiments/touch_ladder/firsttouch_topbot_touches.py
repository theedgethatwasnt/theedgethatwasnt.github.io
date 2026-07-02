"""
firsttouch_topbot_touches.py — does TopsBots-style swing-pivot S/R (recency = most-recent
confirmed fractal pivot) change the first-touch edge, and does the EDGE survive to 2nd/3rd touch?
Causal fractal swing detector (width w, confirmed at p+w), active level = latest pivot, per-level
touch counter. Run the +32p selectivity stack (low-vol + with-daily-trend + big-overshoot) on the
1st / 2nd / 3rd touch. Compare to the Donchian first-touch baseline. Net of (fixed) spread, OOS(60/40).
"""
import numpy as np, pandas as pd, duckdb
from scipy.stats import spearmanr
from firsttouch_traitminer import load, atr, PAIRS, IS_FRAC, EPS, VW, TGT, SLm, HCAP, trades as donchian_trades

def swing_trades(con,pair,touch_n,w=3):
    pip,sp=PAIRS[pair]; r,d1=load(con,pair)
    o=r.open.values; h=r.high.values; l=r.low.values; c=r.close.values; v=r.volume.values
    ts=r.index; n=len(c); eps=EPS*pip; a=atr(h,l,c,14)
    vrel=v/pd.Series(v).rolling(VW).mean().shift(1).values
    d1s=d1.reindex(ts,method="ffill"); d1ret=np.sign(d1s.diff().shift(1).values)
    R=np.nan; S=np.nan; rt=0; st=0; out=[]; last=-10; is_cut=int(n*IS_FRAC)
    for t in range(2*w+1,n-HCAP-1):
        p=t-w                                                  # causally confirm a pivot at p (needs bars up to t)
        if h[p]>=h[p-w:p+w+1].max() and h[p]>h[p-1] and h[p]>h[p+1]: R=h[p]; rt=0
        if l[p]<=l[p-w:p+w+1].min() and l[p]<l[p-1] and l[p]<l[p+1]: S=l[p]; st=0
        if np.isnan(R) or np.isnan(S) or a[t]<=0 or np.isnan(vrel[t]) or t-last<HCAP: continue
        up=h[t]>=R-eps and h[t-1]<R-eps and c[t]<=R
        dn=l[t]<=S+eps and l[t-1]>S+eps and c[t]>=S
        d=0; lvl=0.0
        if up: rt+=1; d=(-1 if rt==touch_n else 0); lvl=R
        elif dn: st+=1; d=(+1 if st==touch_n else 0); lvl=S
        if d==0: continue
        entry=c[t]; ae=a[t]
        overshoot=((h[t]-R) if d==-1 else (S-l[t]))/ae
        with_d1=float(d*(-d1ret[t]) if not np.isnan(d1ret[t]) else 0.0)
        tp=entry-TGT*ae if d==-1 else entry+TGT*ae; slp=entry+SLm*ae if d==-1 else entry-SLm*ae
        exitpx=c[t+HCAP]
        for j in range(t+1,t+HCAP+1):
            if d==-1:
                if h[j]>=slp: exitpx=slp; break
                if l[j]<=tp: exitpx=tp; break
            else:
                if l[j]<=slp: exitpx=slp; break
                if h[j]>=tp: exitpx=tp; break
        out.append(dict(pair=pair,is_is=t<is_cut,pnl=d*(exitpx-entry)/pip-sp,
                        vrel=vrel[t],overshoot=overshoot,with_d1=with_d1)); last=t
    return out

def stack_oos(df,rng):
    IS=df[df.is_is]; OOS=df[~df.is_is]
    if len(IS)<30 or len(OOS)<20: return None
    vm=IS.vrel.median(); om=IS.overshoot.median()
    def ev(m):
        s=OOS[m]
        if len(s)<15: return (len(s),np.nan,np.nan,np.nan)
        boot=np.array([rng.choice(s.pnl.values,len(s)).mean() for _ in range(2000)])
        return (len(s),s.pnl.mean(),100*(s.pnl>0).mean(),(boot<=0).mean())
    return dict(all=ev(OOS.index==OOS.index),
               lowvol=ev(OOS.vrel<vm),
               stack=ev((OOS.vrel<vm)&(OOS.with_d1<=0)&(OOS.overshoot>om)))

def main():
    con=duckdb.connect(); rng=np.random.default_rng(0)
    print("Level + touch-number test — +32p selectivity stack OOS, net spread.  (n | OOS pnl | WR | P<=0)")
    print("="*94)
    # Donchian first-touch reference
    dch=pd.DataFrame([x for p in PAIRS for x in donchian_trades(con,p)])
    dch=dch.rename(columns={});
    s=stack_oos(dch,rng)
    def row(name,s):
        if s is None: print(f"  {name:34s}  (too few)"); return
        for key in ("all","lowvol","stack"):
            n,e,w,P=s[key]
            print(f"  {name+' / '+key:40s} n={n:4d}  OOS {e:+6.2f}p  WR {w:3.0f}%  P<=0 {P:.3f}")
        print()
    row("Donchian 1st-touch (baseline)", s)
    for tn in (1,2,3):
        df=pd.DataFrame([x for p in PAIRS for x in swing_trades(con,p,tn)])
        print(f"  --- TopsBots swing level, touch #{tn}  ({len(df)} trades) ---")
        row(f"swing touch#{tn}", stack_oos(df,rng))

if __name__=="__main__": main()
