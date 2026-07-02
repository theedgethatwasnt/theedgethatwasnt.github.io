"""
firsttouch_traitminer.py — hone in on the traits that make first-touch H4 reversions clear the
spread AND THEN SOME. Reproduces the v2 first-touch trades (config tgt2/sl2/H12), tags each with
rich CAUSAL features at the touch + a full-window net-MFE ("headroom") label, then mines what
precedes the big reversions (IC + regularized LightGBM) and validates any trait-selected subset
OUT-OF-SAMPLE (IS 60% / OOS 40%, the v2 split) against the +1.59p portfolio baseline.
Label that matters = MOVE SIZE (net-MFE), not win/loss.
"""
import numpy as np, pandas as pd, duckdb, lightgbm as lgb
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6; L=25; EPS=12; VW=20; TGT,SLm,HCAP=2.0,2.0,12
FEATS=["vrel","atr_rel","swing_atr","overshoot","rejection","approach","range_pos","with_d1","hour","dow"]

def load(con,pair):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,open,high,low,close,volume FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    h4=df.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    d1=df["close"].resample("1D").last().dropna()
    return h4,d1

def atr(h,l,c,n=14):
    pc=np.empty_like(c); pc[0]=c[0]; pc[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    return pd.Series(tr).ewm(alpha=1/n,adjust=False).mean().values

def trades(con,pair):
    pip,sp=PAIRS[pair]; r,d1=load(con,pair)
    o=r.open.values; h=r.high.values; l=r.low.values; c=r.close.values; v=r.volume.values
    ts=r.index; n=len(c); eps=EPS*pip; a=atr(h,l,c,14)
    Rs=pd.Series(h).rolling(L).max().shift(1).values; Ss=pd.Series(l).rolling(L).min().shift(1).values
    vrel=v/pd.Series(v).rolling(VW).mean().shift(1).values
    arel=a/pd.Series(a).rolling(50).mean().shift(1).values
    d1s=d1.reindex(ts,method="ffill"); d1ret=np.sign(d1s.diff().shift(1).values)  # prior daily move sign (causal)
    out=[]; last=-10; is_cut=int(n*IS_FRAC)
    for t in range(L+1,n-HCAP-1):
        if np.isnan(Rs[t]) or a[t]<=0 or np.isnan(vrel[t]) or np.isnan(arel[t]) or t-last<HCAP: continue
        R=Rs[t]; S=Ss[t]; rng=h[t]-l[t]+1e-9
        up=h[t]>=R-eps and h[t-1]<R-eps and c[t]<=R
        dn=l[t]<=S+eps and l[t-1]>S+eps and c[t]>=S
        if not(up or dn): continue
        if up: tch=int(np.sum(h[t-L:t]>=R-eps)); d=-1
        else:  tch=int(np.sum(l[t-L:t]<=S+eps)); d=+1
        if tch>1: continue
        entry=c[t]; ae=a[t]
        # features (all causal, fade-direction-consistent)
        overshoot=((h[t]-R) if up else (S-l[t]))/ae
        rejection=((R-c[t]) if up else (c[t]-S))/rng
        approach=(abs(c[t]-c[t-3]))/ae
        range_pos=((c[t]-S) if up else (R-c[t]))/(R-S+1e-9)     # near the faded extreme -> ~1
        with_d1 = float(d*(-d1ret[t]) if not np.isnan(d1ret[t]) else 0.0)  # fade with the daily trend?
        # exit (v2) + full-window net-MFE
        if d==-1: tp=entry-TGT*ae; slp=entry+SLm*ae
        else:     tp=entry+TGT*ae; slp=entry-SLm*ae
        exitpx=c[t+HCAP]; best=entry
        for j in range(t+1,t+HCAP+1):
            if d==-1:
                if l[j]<best: best=l[j]
                if h[j]>=slp: exitpx=slp; break
                if l[j]<=tp: exitpx=tp; break
            else:
                if h[j]>best: best=h[j]
                if l[j]<=slp: exitpx=slp; break
                if h[j]>=tp: exitpx=tp; break
        pnl=d*(exitpx-entry)/pip-sp
        mfe=(d*(best-entry)/pip)-sp                              # headroom: best reversion available, net spread
        out.append(dict(pair=pair,ts=ts[t],is_is=t<is_cut,pnl=pnl,mfe=mfe,hold=j-t,
            vrel=vrel[t],atr_rel=arel[t],swing_atr=(R-S)/ae,overshoot=overshoot,
            rejection=rejection,approach=approach,range_pos=range_pos,with_d1=with_d1,
            hour=ts[t].hour,dow=ts[t].dayofweek)); last=t
    return out

def main():
    con=duckdb.connect(); rng=np.random.default_rng(0)
    df=pd.DataFrame([x for p in PAIRS for x in trades(con,p)])
    IS=df[df.is_is]; OOS=df[~df.is_is]
    print(f"first-touch H4 pool: {len(df)} trades  (IS {len(IS)} / OOS {len(OOS)})")
    print(f"baseline portfolio:  IS pnl {IS.pnl.mean():+.2f}p  OOS pnl {OOS.pnl.mean():+.2f}p  OOS WR {100*(OOS.pnl>0).mean():.0f}%")
    print(f"available headroom:  OOS net-MFE p50 {OOS.mfe.median():.1f}p  mean {OOS.mfe.mean():.1f}p  (vs ~2-4p spread)\n"+"="*78)
    print("  (1) IS IC of each trait vs net-MFE (headroom) and vs pnl:")
    for c in FEATS:
        print(f"     {c:10s}  IC_mfe={spearmanr(IS[c],IS.mfe).statistic:+.3f}   IC_pnl={spearmanr(IS[c],IS.pnl).statistic:+.3f}")
    # (2) regularized LightGBM on net-MFE (IS->OOS), small-sample-safe
    reg=lgb.LGBMRegressor(n_estimators=200,learning_rate=0.03,num_leaves=15,min_child_samples=40,
        subsample=0.8,colsample_bytree=0.8,random_state=0,verbose=-1).fit(IS[FEATS],IS.mfe)
    pred=reg.predict(OOS[FEATS]); ic=spearmanr(pred,OOS.mfe).statistic
    print(f"\n  (2) LightGBM->net-MFE: OOS IC(pred,mfe)={ic:+.3f}   importance: "
          + ", ".join(f"{k}:{int(val)}" for k,val in pd.Series(reg.feature_importances_,index=FEATS).sort_values(ascending=False).items()))
    # (3) trait-selected subset OOS expectancy vs baseline — does honing lift it?
    print("\n  (3) OOS expectancy of trait-selected subsets (threshold set on IS), vs baseline:")
    def oos_cut(name,mask_oos):
        s=OOS[mask_oos]
        if len(s)<20: print(f"     {name:34s} n={len(s):4d}  (too few)"); return
        boot=np.array([rng.choice(s.pnl.values,len(s)).mean() for _ in range(3000)])
        print(f"     {name:34s} n={len(s):4d}  OOS pnl {s.pnl.mean():+5.2f}p  WR {100*(s.pnl>0).mean():3.0f}%  P(<=0)={(boot<=0).mean():.3f}")
    oos_cut("ALL (baseline)", OOS.index==OOS.index)
    vmed=IS.vrel.median()
    oos_cut(f"low volume (vrel<IS median {vmed:.2f})", OOS.vrel<vmed)
    oos_cut("high rejection (>IS p67)", OOS.rejection>IS.rejection.quantile(0.67))
    oos_cut("counter-daily-trend (against)", OOS.with_d1>0)
    oos_cut("WITH daily trend (buy dip)", OOS.with_d1<=0)
    oos_cut("model net-MFE top tercile", pred>=np.quantile(pred,0.67))
    oos_cut("lowvol & WITH-daily-trend", (OOS.vrel<vmed)&(OOS.with_d1<=0))
    oos_cut("lowvol & WITH-trend & big overshoot", (OOS.vrel<vmed)&(OOS.with_d1<=0)&(OOS.overshoot>IS.overshoot.median()))

if __name__=="__main__": main()
