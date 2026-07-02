"""
m5_regime_amddp_ts.py — AMDDP5 of the run INTO the entry as a multi-window indicator, with the
"pain" (accumulated drawdown) integrated at S5 cadence (finer than the M5 bars the signal uses).

For each SMA9/200 signal (M5 bar i; entry would be i+1 open), and each lookback W in {3,5,7,10,13}
M5 bars, score a hypothetical signal-direction trade entered at the open of the W-th M5 bar back:
   AMDDP5_W = final_pnl − 0.05 * accumulated_drawdown   (+ profit floor)
accumulated_drawdown is summed over the **S5 ticks** composing those W M5 bars (M5 bar j = S5
[j*60 : j*60+60]); the window ends at the LAST S5 of bar i (the signal close) — strictly before
the i+1 entry, so NO lookahead. Features (the other 18) stay on M5 via build_pair.

Then: (1) the AMDDP5_W profile, (2) does its slope/acceleration predict the trade outcome,
(3) does it predict inside LightGBM alongside the other indicators?
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq, os
from numba import njit
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from m5_regime_lgbm import build_pair, FEATS, PAIRS, DATA

WINDOWS=(3,5,7,10,13)

@njit
def amddp_s5_batch(o5,h5,l5,c5,sigs,dirs,W,pip,beta=0.05):
    m=len(sigs); out=np.full(m,np.nan); n5=len(c5)
    for t in range(m):
        i=sigs[t]; d=dirs[t]; start=(i-W+1)*60; end=i*60+60   # S5 [start,end): bars i-W+1..i
        if start<0 or end>n5: continue
        entry=o5[start]; peak=0.0; ddsum=0.0
        for k in range(start,end):
            best  = d*((h5[k] if d==1 else l5[k])-entry)/pip   # best unrealized (signal dir)
            worst = d*((l5[k] if d==1 else h5[k])-entry)/pip   # worst unrealized
            if best>peak: peak=best
            gb=peak-worst
            if gb>0: ddsum+=gb                                  # S5-cadence accumulated pain
        finalp=d*(c5[end-1]-entry)/pip
        a=finalp-beta*ddsum
        if finalp>0 and a<0: a=0.001
        out[t]=a
    return out

def main():
    parts=[]
    for p in PAIRS:
        f=DATA.format(p)
        if not os.path.exists(f): continue
        df=build_pair(p)                                       # 18 M5 feats + sig + dir + pnl + ts
        t=pq.read_table(f,columns=["open","high","low","close"])
        o5,h5,l5,c5=(t.column(k).to_numpy().astype(np.float64) for k in ["open","high","low","close"])
        pip=0.01 if "JPY" in p else 0.0001
        # per-pair spread for normalization (median S5 close spread already in df.spread_pips)
        sp=float(np.median(df.spread_pips.values)) if len(df) else 1.0
        sig=df.sig.values.astype(np.int64); dr=df["dir"].values.astype(np.int64)
        for W in WINDOWS:
            df[f"a5_{W}"]=amddp_s5_batch(o5,h5,l5,c5,sig,dr,W,pip)/sp   # spread units
        # MA of the AMDDP5(W=5) TIME SERIES over the last 5 signal bars (smooth the indicator)
        stk=np.vstack([amddp_s5_batch(o5,h5,l5,c5,sig-j,dr,5,pip)/sp for j in range(5)])
        df["a5_ma5"]=np.nanmean(stk,axis=0)                            # smoothed AMDDP level
        df["a5_ma_dev"]=df["a5_5"]-df["a5_ma5"]                        # current vs its MA (>0 = above trend)
        parts.append(df)
    df=pd.concat(parts,ignore_index=True).dropna(subset=[f"a5_{W}" for W in WINDOWS])
    df["a5_slope"]=df.a5_3-df.a5_13                            # recent vs longer context (>0 = accelerating clean run)
    df["a5_accel"]=df.a5_3-2*df.a5_7+df.a5_13                  # curvature of the profile
    AF=[f"a5_{W}" for W in WINDOWS]+["a5_slope","a5_accel","a5_ma5","a5_ma_dev"]
    print(f"AMDDP5-into-entry, S5-cadence pain, {len(df)} trades. outcome=MA-cross net p/tr (base {df.pnl.mean():+.2f})")
    print("="*80)
    print("  (1) IC(feature, trade pnl)  —  Spearman rank correlation, full sample")
    for c in AF:
        ic=spearmanr(df[c],df.pnl).statistic
        print(f"     {c:10s}  IC={ic:+.4f}")
    print("\n  (2) outcome binned by AMDDP profile slope (increasing/accelerating run into entry)")
    for c in ["a5_5","a5_slope","a5_accel"]:
        qs=df[c].quantile(np.linspace(0,1,6)).values; print(f"   {c}:")
        for k in range(5):
            lo,hi=qs[k],qs[k+1]; m=(df[c]>=lo)&(df[c]<=hi) if k==4 else (df[c]>=lo)&(df[c]<hi)
            s=df[m]; print(f"     Q{k+1} [{lo:+6.2f},{hi:+6.2f}] n={len(s):5d}  net {s.pnl.mean():+6.2f}  WR {100*(s.pnl>0).mean():4.1f}%")
    # (3) LightGBM: AMDDP alone vs AMDDP + the other 18
    df=df.sort_values("ts").reset_index(drop=True); df["win"]=(df.pnl>0).astype(int)
    cut=int(len(df)*0.70); tr,te=df.iloc[:cut],df.iloc[cut:]
    def fit_auc(cols):
        c=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=31,min_child_samples=200,
            subsample=0.8,colsample_bytree=0.8,random_state=0,verbose=-1).fit(tr[cols],tr.win)
        return c, roc_auc_score(te.win,c.predict_proba(te[cols])[:,1])
    _,auc_a=fit_auc(AF)
    clf,auc_b=fit_auc(AF+FEATS)
    print(f"\n  (3) LightGBM OOS AUC (predict win):")
    print(f"     AMDDP features alone        : {auc_a:.4f}")
    print(f"     AMDDP + the other 18 inds   : {auc_b:.4f}   (0.50 = no skill)")
    imp=pd.Series(clf.feature_importances_,index=AF+FEATS).sort_values(ascending=False)
    print("     top-12 feature importance in the combined model:")
    print(imp.head(12).to_string().replace("\n","\n       "))
    amrank=[list(imp.index).index(c)+1 for c in AF]
    print(f"     AMDDP features' ranks among {len(AF+FEATS)}: {dict(zip(AF,amrank))}")

if __name__=="__main__": main()
