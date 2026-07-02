#!/usr/bin/env python3
"""
Stage 3 — Momentum/Efficiency EXIT overlay on the regime-gated daily MR.
=======================================================================
Entries are FIXED (so we measure only the exit's contribution):
  regime-gated daily MR — z-score(close vs SMA10) ±2σ entry, gated by
  vol_pct<0.5 & ac60<0.  Direction = mean-reversion (long when z<-2, short z>+2).
Baseline exit = fixed 3 trading-day hold.

During the hold we monitor INTRADAY M5 with the user's features:
  momentum_ppm = (close - close[t-W])/pip/W_min   (signed pips/min)
  efficiency   = |net|/path_length                (clean vs wiggle)
  mom_norm     = momentum_ppm / ATR_per_min       (vol units → one threshold all pairs)

Exit rules (same entries each time):
  BASE      : hold to 3-day boundary
  STOP60    : fixed 60p stop (known to bound the MAE tail), else 3-day hold
  ADV_CLEAN : exit if adverse mom_norm > thr AND efficiency > thr_eff
              (the adverse move has become a clean/accelerating trend → cut)
  ADV+STOP  : ADV_CLEAN and a 60p hard stop

Compare avg pnl/trade, WR, MAE tail, mean hold, and 4-fold WF. The win we want:
higher pnl AND/OR a smaller MAE tail AND/OR weak folds rescued. Read-only.
"""
import numpy as np, pandas as pd
from pathlib import Path
from numba import njit
import warnings; warnings.filterwarnings("ignore")

PROJECT=Path(__file__).resolve().parents[3]; DATA=PROJECT/"data"/"m5_ba"
RES=Path(__file__).parent/"results"; RES.mkdir(exist_ok=True)
PAIRS=["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
       "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY={"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
def pip_sz(p): return 0.01 if p in JPY else 0.0001
SMA_N,ZT,HOLD_D=10,2.0,3
W_MON=30          # intraday monitoring window (minutes)

@njit(cache=True)
def monitor(close,bid,ask,mom_norm,eff,entry_pos,dir_,em,bound,rule,thr,thr_eff,stop):
    worst=0.0
    for j in range(entry_pos+1, bound+1):
        cur=(close[j]-em)*dir_
        if cur<worst: worst=cur
        if stop>0 and cur<=-stop:
            return em-dir_*stop, j, worst
        if rule==2:  # adverse-clean
            adv=-mom_norm[j]*dir_
            if (not np.isnan(adv)) and adv>thr and (not np.isnan(eff[j])) and eff[j]>thr_eff:
                return close[j], j, worst
    return close[bound], bound, worst

def atr(d,n=14):
    h,l,c=d["high"],d["low"],d["close"]
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def build(pair):
    df=pd.read_parquet(DATA/f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df=df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    pip=pip_sz(pair); bw=W_MON//5; c=df["close"]
    net=(c-c.shift(bw))/pip; mom=net/W_MON
    path=(c.diff().abs()/pip).rolling(bw).sum(); eff=net.abs()/path.replace(0,np.nan)
    atr_pm=(atr(df,14)/pip)/5.0                      # ATR per minute (pips)
    mom_norm=mom/atr_pm.replace(0,np.nan)
    # daily bars + signal + regime
    d=df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last",
                              "bid_c":"last","ask_c":"last"}).dropna()
    dc=d["close"]; ret=dc.pct_change()
    z=(dc-dc.rolling(SMA_N).mean())/dc.rolling(SMA_N).std()
    d["sig"]=np.where(z<-ZT,1.0,np.where(z>ZT,-1.0,0.0))
    d["vol_pct"]=ret.rolling(20).std().rolling(252).rank(pct=True)
    d["ac60"]=ret.rolling(60).corr(ret.shift(1))
    d["spread"]=(d["ask_c"]-d["bid_c"])/pip
    # map each daily bar to last M5 integer position of that day
    pos=pd.Series(np.arange(len(df)),index=df.index)
    last_m5=pos.groupby(df.index.normalize()).last()
    d["m5pos"]=last_m5.reindex(d.index).values
    arr=dict(close=(c/pip).values,bid=(df["bid_c"]/pip).values,ask=(df["ask_c"]/pip).values,
             mom_norm=mom_norm.values,eff=eff.values,pip=pip)
    return d.reset_index(), arr

def run_rule(D, rule, thr, thr_eff, stop, lo, hi):
    """Pool trades for a given exit rule over fractional index window [lo,hi) of daily bars."""
    allp=[]; allmae=[]; allhold=[]; pp=[]
    for pair in PAIRS:
        d,a=D[pair]; n=len(d); A=int(n*lo); B=int(n*hi)
        close=a["close"]; bid=a["bid"]; ask=a["ask"]; mn=a["mom_norm"]; ef=a["eff"]
        sig=d["sig"].values; vp=d["vol_pct"].values; ac=d["ac60"].values
        m5pos=d["m5pos"].values; spread=d["spread"].values
        pnls=[]
        for i in range(max(A,1), min(B, n-HOLD_D)):
            if sig[i]==0: continue
            if np.isnan(vp[i]) or np.isnan(ac[i]): continue
            if not (vp[i]<0.5 and ac[i]<0): continue          # regime gate
            ep=int(m5pos[i]); bnd=int(m5pos[i+HOLD_D])
            if ep<=0 or bnd<=ep or np.isnan(close[ep]): continue
            dir_=sig[i]; em=close[ep]
            ex_mid,ex_pos,worst=monitor(close,bid,ask,mn,ef,ep,dir_,em,bnd,rule,thr,thr_eff,stop)
            pnl=(ex_mid-em)*dir_-spread[i]
            pnls.append(pnl); allmae.append(worst); allhold.append((ex_pos-ep)*5/60/24)  # days
        if pnls:
            allp.append(np.array(pnls))
            pp.append(1 if np.mean(pnls)>0 else 0)
    if not allp: return None
    arr=np.concatenate(allp); mae=np.array(allmae)
    t=arr.mean()/(arr.std(ddof=1)/np.sqrt(len(arr))) if arr.std()>0 else np.nan
    return dict(n=len(arr),avg=arr.mean(),t=t,wr=(arr>0).mean()*100,pairs_pos=sum(pp),
                mae_med=np.percentile(mae,50),mae_p90=np.percentile(mae,10),
                mae_worst=mae.min(),hold=np.mean(allhold))

print("Loading + building 12 pairs (M5 + daily) …")
D={p:build(p) for p in PAIRS}

RULES=[
  ("BASE (3d hold)",      dict(rule=0,thr=0,thr_eff=0,stop=0)),
  ("STOP60",              dict(rule=0,thr=0,thr_eff=0,stop=60)),
  ("ADV_CLEAN t1.0 e0.5", dict(rule=2,thr=1.0,thr_eff=0.5,stop=0)),
  ("ADV_CLEAN t1.5 e0.5", dict(rule=2,thr=1.5,thr_eff=0.5,stop=0)),
  ("ADV_CLEAN t1.0 e0.3", dict(rule=2,thr=1.0,thr_eff=0.3,stop=0)),
  ("ADV+STOP60 t1.5 e0.5",dict(rule=2,thr=1.5,thr_eff=0.5,stop=60)),
]
print("\n[Full-sample comparison — same fixed entries]")
print(f"  {'rule':<22}{'avg':>7}{'t':>6}{'wr':>5}{'n':>6}{'pairs+':>7}{'MAE_med':>8}{'MAE_p90':>8}{'MAE_wst':>8}{'hold_d':>7}")
base=None
for name,p in RULES:
    r=run_rule(D,p["rule"],p["thr"],p["thr_eff"],p["stop"],0.0,1.0)
    if name.startswith("BASE"): base=r
    dpa = "" if base is None else f"  (Δ{r['avg']-base['avg']:+.1f})"
    print(f"  {name:<22}{r['avg']:>7.2f}{r['t']:>6.2f}{r['wr']:>4.0f}%{r['n']:>6d}{r['pairs_pos']:>6d}/12"
          f"{r['mae_med']:>8.0f}{r['mae_p90']:>8.0f}{r['mae_worst']:>8.0f}{r['hold']:>7.2f}{dpa}")

print("\n[4-fold WF — BASE vs best ADV_CLEAN vs ADV+STOP]")
for name,p in [("BASE (3d hold)",dict(rule=0,thr=0,thr_eff=0,stop=0)),
               ("ADV_CLEAN t1.5 e0.5",dict(rule=2,thr=1.5,thr_eff=0.5,stop=0)),
               ("ADV+STOP60 t1.5 e0.5",dict(rule=2,thr=1.5,thr_eff=0.5,stop=60))]:
    folds=[run_rule(D,p["rule"],p["thr"],p["thr_eff"],p["stop"],k/4,(k+1)/4) for k in range(4)]
    line=" ".join(f"f{i+1}={f['avg']:+.1f}(n{f['n']})" if f else f"f{i+1}=NA" for i,f in enumerate(folds))
    allpos=all(f and f['avg']>0 for f in folds)
    print(f"  {name:<22} {'ALL+' if allpos else '    '}  {line}")
print("\nDone.")
