"""
m5_regime_lgbm.py — Can LightGBM predict which SMA9/200-regime M5 trades win,
from causal pre-entry features? OOS-tested (time-split), per the project's anti-overfit rule.

Per-trade features, all known at the signal bar i (causal):
  r1,r2,r3      per-bar return over the 3 bars into entry, signed IN TRADE DIR, spread units
                (momentum "as it develops"; r1 = most recent)
  mom3          sum of the three
  v1,v2,v3      per-bar tick volume over those 3 bars, x pair-median (volume "as it develops")
  vol3          mean of the three
  regime_dist   (SMA9-SMA200) signed in trade dir / spread  (how deep the regime)
  dist_sma9     (close-SMA9)  signed in trade dir / spread  (how far past the fast MA)
  atr_sp        ATR14 / spread (volatility regime)
  spread_pips   the toll itself
  range_pos     position in the rolling-N high/low channel [0=support,1=resistance]  (S/R proximity)
  room          distance to the extreme in the TRADE direction / spread (target room)
  hour          hour-of-day (UTC)
  dir           +1 long / -1 short
Label: win = pnl>0. Also keep pnl for the decisive "does the model's pick clear spread?" test.
"""
import numpy as np, pyarrow.parquet as pq, pandas as pd, os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

PAIRS = ["USD_JPY","EUR_USD","GBP_USD","AUD_USD","EUR_JPY","GBP_JPY",
         "AUD_JPY","CAD_JPY","NZD_JPY","CHF_JPY","NZD_USD","EUR_GBP"]
DATA="data/s5_ohlc/{}_S5_BA.parquet"; BARS_PER=60; FAST,SLOW=9,200; RNG=96  # S/R lookback ~8h

def blkres(o,h,l,c,bid,ask,vol,ts,bars=BARS_PER):
    n=len(c); nb=n//bars
    def B(a): return a[:nb*bars].reshape(nb,bars)
    return (B(o)[:,0],B(h).max(1),B(l).min(1),B(c)[:,-1],B(bid)[:,-1],B(ask)[:,-1],
            B(vol).sum(1), ts[:nb*bars].reshape(nb,bars)[:,0])

def sma(c,p):
    out=np.full(len(c),np.nan); cs=np.cumsum(c); out[p-1:]=(cs[p-1:]-np.concatenate([[0.0],cs[:-p]]))/p; return out
def atr(h,l,c,p=14):
    n=len(c); tr=np.empty(n); tr[0]=h[0]-l[0]
    for i in range(1,n): tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    return pd.Series(tr).rolling(p).mean().to_numpy()

FEATS=["r1","r2","r3","mom3","v1","v2","v3","vol3","regime_dist","dist_sma9",
       "atr_sp","spread_pips","range_pos","room","bb_width_rel","bb_widening","hour","dir"]

def build_pair(p):
    f=DATA.format(p)
    t=pq.read_table(f,columns=["open","high","low","close","bid_c","ask_c","volume","timestamp"])
    o,h,l,c,bid,ask,vol=(t.column(k).to_numpy().astype(np.float64) for k in
        ["open","high","low","close","bid_c","ask_c","volume"])
    ts=t.column("timestamp").to_numpy()
    O,H,L,C,BID,ASK,Vv,TS=blkres(o,h,l,c,bid,ask,vol,ts)
    pip=0.01 if "JPY" in p else 0.0001
    SP=(ASK-BID); medV=float(np.median(Vv)); a=atr(H,L,C)
    s9,s200=sma(C,FAST),sma(C,SLOW)
    hiN=pd.Series(H).rolling(RNG).max().to_numpy(); loN=pd.Series(L).rolling(RNG).min().to_numpy()
    bbw=4.0*pd.Series(C).rolling(20).std().to_numpy()                 # Bollinger width (price, k=2 -> ±2σ)
    bbw_med=pd.Series(bbw).rolling(RNG).median().to_numpy()           # its recent normal
    hour=pd.to_datetime(TS).hour
    rows=[]; pos=0; entry=0.0; ei=0; sig=0
    for i in range(SLOW,len(C)-1):
        if np.isnan(s9[i]) or np.isnan(s200[i]): continue
        hsp=SP[i+1]*0.5; fo=O[i+1]
        if pos!=0:
            if (pos==-1 and s9[i]>=s200[i]) or (pos==1 and s9[i]<=s200[i]):
                px=(entry-(fo+hsp)) if pos==-1 else ((fo-hsp)-entry)
                rows[-1]["pnl"]=px/pip; pos=0
            continue
        down=s9[i]<s200[i]; up=s9[i]>s200[i]
        sc=down and (C[i]<s9[i]) and not((s9[i-1]<s200[i-1]) and (C[i-1]<s9[i-1]))
        lc=up   and (C[i]>s9[i]) and not((s9[i-1]>s200[i-1]) and (C[i-1]>s9[i-1]))
        d=0
        if sc: d=-1
        elif lc: d=1
        if d!=0 and not np.isnan(a[i]) and not np.isnan(hiN[i]) and not np.isnan(bbw_med[i]) and bbw_med[i]>0 and SP[i]>0:
            sp=SP[i]
            rng_pos=(C[i]-loN[i])/max(hiN[i]-loN[i],1e-9)
            room=((C[i]-loN[i]) if d==-1 else (hiN[i]-C[i]))/sp     # room in trade dir
            rows.append(dict(
                r1=d*(C[i]-C[i-1])/sp, r2=d*(C[i-1]-C[i-2])/sp, r3=d*(C[i-2]-C[i-3])/sp,
                mom3=d*(C[i]-C[i-3])/sp,
                v1=Vv[i]/medV, v2=Vv[i-1]/medV, v3=Vv[i-2]/medV, vol3=(Vv[i]+Vv[i-1]+Vv[i-2])/(3*medV),
                regime_dist=d*(s9[i]-s200[i])/sp, dist_sma9=d*(C[i]-s9[i])/sp,
                atr_sp=a[i]/sp, spread_pips=sp/pip, range_pos=rng_pos, room=room,
                bb_width_rel=bbw[i]/bbw_med[i],                       # <1 = compressed vs recent
                bb_widening=(bbw[i]-bbw[i-3])/bbw_med[i],             # >0 = widening into entry
                hour=int(hour[i]), dir=d, ts=TS[i], sig=i, pnl=np.nan))
            entry=(fo-hsp) if d==-1 else (fo+hsp); pos=d; ei=i+1
    df=pd.DataFrame(rows); return df[~df.pnl.isna()].copy()

def main():
    dfs=[build_pair(p) for p in PAIRS if os.path.exists(DATA.format(p))]
    df=pd.concat(dfs,ignore_index=True).sort_values("ts").reset_index(drop=True)
    df["win"]=(df.pnl>0).astype(int)
    n=len(df); cut=int(n*0.70)
    tr,te=df.iloc[:cut], df.iloc[cut:]
    print(f"trades: {n}  (IS {len(tr)} / OOS {len(te)}, time-split)  base win-rate IS={tr.win.mean():.3f} OOS={te.win.mean():.3f}")
    print(f"overall mean pnl: IS {tr.pnl.mean():+.2f}  OOS {te.pnl.mean():+.2f} p/tr")
    clf=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=31,
        min_child_samples=200,subsample=0.8,colsample_bytree=0.8,random_state=0,verbose=-1)
    clf.fit(tr[FEATS],tr.win)
    proba=clf.predict_proba(te[FEATS])[:,1]
    auc=roc_auc_score(te.win,proba)
    print(f"\nLightGBM OOS AUC (predict win) = {auc:.4f}   (0.50 = no skill)")
    imp=pd.Series(clf.feature_importances_,index=FEATS).sort_values(ascending=False)
    print("feature importance (gain-split):"); print(imp.to_string())
    # decisive test: does selecting the model's top-predicted trades clear the spread OOS?
    te=te.copy(); te["proba"]=proba
    te["q"]=pd.qcut(te.proba,5,labels=[1,2,3,4,5])
    print("\nOOS trades binned by model P(win)  —  n | mean pnl p/tr | win%")
    for q in [1,2,3,4,5]:
        s=te[te.q==q]
        print(f"  P(win) Q{q}  n={len(s):5d}  {s.pnl.mean():+6.2f}  {100*s.win.mean():4.1f}%")
    best=te[te.q==5]
    print(f"\nmodel's top-quintile OOS pick (by P(win)): {best.pnl.mean():+.2f} p/tr over {len(best)} trades  "
          f"({'CLEARS spread (net-positive)' if best.pnl.mean()>0 else 'still net-NEGATIVE — does not clear spread'})")
    # fairer test: regress on pnl directly (win-rate != profit here) and select top predicted-pnl
    reg=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.03,num_leaves=31,min_child_samples=200,
        subsample=0.8,colsample_bytree=0.8,random_state=0,verbose=-1)
    reg.fit(tr[FEATS],tr.pnl)
    pred=reg.predict(te[FEATS]); ic=float(np.corrcoef(pred,te.pnl)[0,1])
    te["pp"]=pred; te["qp"]=pd.qcut(te.pp.rank(method="first"),5,labels=[1,2,3,4,5])
    print(f"\nLightGBM REGRESSION on pnl — OOS IC(pred, pnl) = {ic:+.4f}")
    print("OOS trades binned by model PREDICTED pnl  —  n | mean pnl p/tr | win%")
    for q in [1,2,3,4,5]:
        s=te[te.qp==q]; print(f"  predpnl Q{q}  n={len(s):5d}  {s.pnl.mean():+6.2f}  {100*s.win.mean():4.1f}%")
    bb=te[te.qp==5]
    print(f"top predicted-pnl quintile OOS: {bb.pnl.mean():+.2f} p/tr  "
          f"({'CLEARS spread' if bb.pnl.mean()>0 else 'still net-NEGATIVE — does not clear spread'})")

if __name__=="__main__": main()
