#!/usr/bin/env python3
"""
Regime-time daily mean-reversion.
=================================
The daily z-score MR edge is real but regime-dependent (4-fold WF: +6.8/-1.8/
+8.6/+23.0). Goal: find a CAUSAL regime variable, measured at entry, that
separates the winning MR trades from the losing ones — then gate on it and check
whether the gated strategy's 4-fold walk-forward becomes consistently positive.

Base (fixed): z-score(close vs SMA10) ±2σ entry, 3-day hold, 12 pairs daily.
Regime vars (all causal, value known at entry bar):
  - vol_pct  : 20d realized-vol percentile over trailing 252d  (low vol → range?)
  - ac60     : 60d lag-1 autocorrelation of daily returns      (<0 → MR regime)
  - trend50  : |close - SMA50| / ATR20                         (high → strong trend)
  - zdepth   : |entry z|                                       (deeper → snapback?)

Discovery uses no-stop pnl (clean effect). Deployable check uses a 60p stop.
Success bar: gated 4-fold WF all-folds-positive. Read-only on data/m5_ba.
"""
import numpy as np, pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data" / "m5_ba"
RESULTS = Path(__file__).parent / "results"; RESULTS.mkdir(exist_ok=True)
PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
def pip_sz(p): return 0.01 if p in JPY else 0.0001
SMA_N, ZT, HOLD = 10, 2.0, 3

def atr(d, n=20):
    h,l,c = d["high"], d["low"], d["close"]
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def build(pair):
    df = pd.read_parquet(DATA/f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    d = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last",
                                "bid_c":"last","ask_c":"last"}).dropna()
    c = d["close"]; ret = c.pct_change()
    z = (c - c.rolling(SMA_N).mean())/c.rolling(SMA_N).std()
    vol20 = ret.rolling(20).std()
    d["sig"]    = np.where(z < -ZT, 1.0, np.where(z > ZT, -1.0, 0.0))
    d["zdepth"] = z.abs()
    d["vol_pct"]= vol20.rolling(252).rank(pct=True)
    d["ac60"]   = ret.rolling(60).corr(ret.shift(1))
    d["trend50"]= (c - c.rolling(50).mean()).abs() / atr(d,20)
    return d, pip_sz(pair)

def trades(d, pip, lo_frac, hi_frac, stop_pips=0.0):
    c=d["close"].values; hi=d["high"].values; lo=d["low"].values
    bid=d["bid_c"].values; ask=d["ask_c"].values; s=d["sig"].values
    reg = {k:d[k].values for k in ("zdepth","vol_pct","ac60","trend50")}
    n=len(c); a=int(n*lo_frac); b=int(n*hi_frac)
    out_pnl=[]; out_reg={k:[] for k in reg}
    for i in range(max(a,1), min(b, n-HOLD)):
        if s[i]==0 or np.isnan(reg["vol_pct"][i]) or np.isnan(reg["ac60"][i]): continue
        dir_=s[i]; em=c[i]; sp=(ask[i]-bid[i])/pip
        stop_lvl=(em-dir_*stop_pips*pip) if stop_pips>0 else None
        exit_mid=None
        for j in range(i+1,i+HOLD+1):
            if stop_pips>0 and ((lo[j]<=stop_lvl) if dir_==1 else (hi[j]>=stop_lvl)):
                exit_mid=stop_lvl; break
        if exit_mid is None: exit_mid=c[i+HOLD]
        out_pnl.append((exit_mid-em)*dir_/pip - sp)
        for k in reg: out_reg[k].append(reg[k][i])
    return np.array(out_pnl), {k:np.array(v) for k,v in out_reg.items()}

print("Loading + building 12 pairs (daily) …")
cache={p:build(p) for p in PAIRS}

# Pool ALL trades (full history) for regime discovery
P=[]; R={k:[] for k in ("zdepth","vol_pct","ac60","trend50")}
for p,(d,pip) in cache.items():
    pn,rg=trades(d,pip,0.0,1.0,0.0)
    P.append(pn)
    for k in R: R[k].append(rg[k])
P=np.concatenate(P); R={k:np.concatenate(v) for k,v in R.items()}
print(f"  pooled trades: {len(P)}   overall avg={P.mean():+.2f}p  wr={(P>0).mean()*100:.0f}%\n")

print("="*78)
print("REGIME CONDITIONING — avg pnl by quintile of each regime var (full pool)")
print("="*78)
for k in ("vol_pct","ac60","trend50","zdepth"):
    v=R[k]; ok=~np.isnan(v); vv=v[ok]; pp=P[ok]
    qs=np.quantile(vv,[0,.2,.4,.6,.8,1.0])
    print(f"\n  {k}:")
    print(f"    {'quintile':<10}{'range':<22}{'avg_pnl':>9}{'wr':>6}{'n':>7}")
    for qi in range(5):
        lo,hi=qs[qi],qs[qi+1]
        m=(vv>=lo)&(vv<=hi) if qi==4 else (vv>=lo)&(vv<hi)
        if m.sum()==0: continue
        t = pp[m].mean()/(pp[m].std(ddof=1)/np.sqrt(m.sum())) if pp[m].std()>0 else np.nan
        print(f"    Q{qi+1:<9}[{lo:>7.3f},{hi:>7.3f}]{pp[m].mean():>9.2f}{(pp[m]>0).mean()*100:>5.0f}%{m.sum():>7d}  t={t:+.2f}")

def wf(gate_fn, stop=0.0):
    """4-fold WF of the gated strategy (gate_fn(reg_dict_at_pool)->bool mask)."""
    res=[]
    for kf in range(4):
        P2=[];
        for p,(d,pip) in cache.items():
            pn,rg=trades(d,pip,kf/4,(kf+1)/4,stop)
            if len(pn)==0: continue
            mask=gate_fn(rg)
            P2.append(pn[mask])
        arr=np.concatenate(P2) if P2 else np.array([0.0])
        t=arr.mean()/(arr.std(ddof=1)/np.sqrt(len(arr))) if len(arr)>1 and arr.std()>0 else np.nan
        res.append((arr.mean(),t,(arr>0).mean()*100,len(arr)))
    return res

print("\n"+"="*78); print("GATED 4-FOLD WALK-FORWARD  (success = all 4 folds positive)"); print("="*78)
gates = {
 "ungated":            (lambda r: np.ones(len(r['vol_pct']),bool), 0.0),
 "vol_pct<0.5":        (lambda r: r['vol_pct']<0.5, 0.0),
 "vol_pct<0.3":        (lambda r: r['vol_pct']<0.3, 0.0),
 "ac60<0":             (lambda r: r['ac60']<0, 0.0),
 "ac60<-0.1":          (lambda r: r['ac60']<-0.1, 0.0),
 "trend50<1.0":        (lambda r: r['trend50']<1.0, 0.0),
 "vol_pct<0.5 & ac60<0":(lambda r:(r['vol_pct']<0.5)&(r['ac60']<0), 0.0),
}
for name,(gf,stp) in gates.items():
    folds=wf(gf,stp)
    allpos = all(f[0]>0 for f in folds)
    line=" ".join(f"f{i+1}={f[0]:+.1f}(t{f[1]:+.1f},n{f[3]})" for i,f in enumerate(folds))
    print(f"  {name:<22} {'ALL+' if allpos else '    '}  {line}")

print("\n[Best gate + 60p stop — deployable check]")
best=(lambda r:(r['vol_pct']<0.5)&(r['ac60']<0))
for stp in (0.0,60.0):
    folds=wf(best,stp); allpos=all(f[0]>0 for f in folds)
    line=" ".join(f"f{i+1}={f[0]:+.1f}" for i,f in enumerate(folds))
    print(f"  stop={'none' if stp==0 else f'{stp:.0f}p':<5} {'ALL+' if allpos else '    '}  {line}  "
          f"(n_total={sum(f[3] for f in folds)})")
print("\nSaved analysis.")
