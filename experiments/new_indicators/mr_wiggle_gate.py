#!/usr/bin/env python3
"""
MR regime gate: lagging vol  vs  sharper wiggle/chop detectors.
===============================================================
Insight: we predict the WIGGLE (volatility/chop) well, DIRECTION poorly, and
spread kills intraday direction bets. So use a sharp wiggle detector to GATE the
one spread-tolerant directional edge — the daily mean-reversion.

Same MR entries every time (z-score(SMA10) ±2σ, 3-day hold, mid-based, spread
deducted). Vary ONLY the regime gate. All gate features are causal rolling-rank
percentiles. Compare each gate on a FAIR post-warmup 4-fold walk-forward (drop the
first 252 days so every fold is equally warmed — fixes fold-1 starvation).

Gates (trade MR only when true):
  G0 ungated
  G1 vol_pct<0.5 & ac60<0           (current: 20d realized-vol pct + 60d autocorr)
  G2 eff_K_pct<0.5                  (intraday EFFICIENCY regime: recently choppy)
  G3 path_pct<0.5                   (intraday TRAVEL: calm)
  G4 fastvol_pct<0.5                (5d realized-vol pct: faster than 20d)
  G5 eff_K_pct<0.5 & ac60<0
  G6 eff_K_pct<0.5 & vol_pct<0.5
Win = all-4-folds-positive AND flatter across folds than G1 (less recency lean).
Read-only on data/m5_ba.
"""
import numpy as np, pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

PROJECT=Path(__file__).resolve().parents[3]; DATA=PROJECT/"data"/"m5_ba"
RES=Path(__file__).parent/"results"; RES.mkdir(exist_ok=True)
PAIRS=["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
       "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY={"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
def pip_sz(p): return 0.01 if p in JPY else 0.0001
SMA_N,ZT,HOLD=10,2.0,3; WARMUP=252

def build(pair):
    df=pd.read_parquet(DATA/f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df=df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    pip=pip_sz(pair); c=df["close"]
    # intraday per-day efficiency & path from M5
    day=df.index.normalize()
    dayv=day.values
    new_day=np.empty(len(dayv),bool); new_day[0]=True; new_day[1:]=dayv[1:]!=dayv[:-1]
    absd_v=c.diff().abs().values.copy(); absd_v[new_day]=0.0  # zero out overnight gap
    gc=c.groupby(day)
    dnet=(gc.last()-gc.first()).abs()/pip
    dpath=pd.Series(absd_v,index=df.index).groupby(day).sum()/pip
    day_eff=(dnet/dpath.replace(0,np.nan))                   # daily intraday efficiency
    # daily bars (entries/exits)
    d=df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last",
                              "bid_c":"last","ask_c":"last"}).dropna()
    dc=d["close"]; ret=dc.pct_change()
    z=(dc-dc.rolling(SMA_N).mean())/dc.rolling(SMA_N).std()
    d["sig"]=np.where(z<-ZT,1.0,np.where(z>ZT,-1.0,0.0))
    d["spread"]=(d["ask_c"]-d["bid_c"])/pip
    d["pip"]=pip
    # regime features (causal rolling-rank percentiles)
    d["vol_pct"]=ret.rolling(20).std().rolling(252).rank(pct=True)
    d["ac60"]=ret.rolling(60).corr(ret.shift(1))
    de=day_eff.reindex(d.index)
    dp=dpath.reindex(d.index)
    d["eff_K_pct"]=de.rolling(10).mean().rolling(252).rank(pct=True)
    d["path_pct"]=dp.rolling(10).mean().rolling(252).rank(pct=True)
    d["fastvol_pct"]=ret.rolling(5).std().rolling(252).rank(pct=True)
    return d.reset_index(drop=True)

def trade_pnls(d, gate, lo, hi):
    """MR trades where gate(row)==True, over POST-WARMUP fractional window [lo,hi)."""
    c=d["close"].values; hi_=d["high"].values; lo_=d["low"].values
    bid=d["bid_c"].values; ask=d["ask_c"].values; sig=d["sig"].values
    sp=d["spread"].values; pip=d["pip"].values[0]
    G=gate(d).values
    n=len(d); start=WARMUP
    A=start+int((n-start)*lo); B=start+int((n-start)*hi)
    out=[]
    for i in range(max(A,1), min(B, n-HOLD)):
        if sig[i]==0 or not G[i]: continue
        dir_=sig[i]; em=c[i]
        out.append((c[i+HOLD]-em)*dir_/pip - sp[i])
    return np.array(out)

GATES={
 "G0 ungated":        lambda d: pd.Series(True,index=d.index),
 "G1 vol&ac (curr)":  lambda d: (d["vol_pct"]<0.5)&(d["ac60"]<0),
 "G2 eff_chop":       lambda d: d["eff_K_pct"]<0.5,
 "G3 path_calm":      lambda d: d["path_pct"]<0.5,
 "G4 fastvol":        lambda d: d["fastvol_pct"]<0.5,
 "G5 eff&ac":         lambda d: (d["eff_K_pct"]<0.5)&(d["ac60"]<0),
 "G6 eff&vol":        lambda d: (d["eff_K_pct"]<0.5)&(d["vol_pct"]<0.5),
}

print("Loading + building 12 pairs …")
D={p:build(p) for p in PAIRS}

def pool(gate, lo, hi):
    allp=[]; pp=0
    for p in PAIRS:
        pn=trade_pnls(D[p], gate, lo, hi)
        if len(pn): allp.append(pn); pp += 1 if pn.mean()>0 else 0
    if not allp: return None
    a=np.concatenate(allp)
    t=a.mean()/(a.std(ddof=1)/np.sqrt(len(a))) if a.std()>0 else np.nan
    return dict(n=len(a),avg=a.mean(),t=t,wr=(a>0).mean()*100,pairs=pp)

print("\n[Full post-warmup sample — same MR entries, vary gate]")
print(f"  {'gate':<18}{'avg':>7}{'t':>6}{'wr':>5}{'n':>6}{'pairs+':>7}")
for name,g in GATES.items():
    r=pool(g,0.0,1.0)
    print(f"  {name:<18}{r['avg']:>7.2f}{r['t']:>6.2f}{r['wr']:>4.0f}%{r['n']:>6d}{r['pairs']:>6d}/12")

print("\n[Fair 4-fold WF — post-warmup, equal folds]  (win = all+ AND flat across folds)")
print(f"  {'gate':<18}{'f1':>9}{'f2':>9}{'f3':>9}{'f4':>9}   verdict")
rows=[]
for name,g in GATES.items():
    fs=[pool(g,k/4,(k+1)/4) for k in range(4)]
    avgs=[f['avg'] if f else np.nan for f in fs]
    ns=[f['n'] if f else 0 for f in fs]
    allpos=all((not np.isnan(a)) and a>0 for a in avgs)
    spread_fold=np.nanmax(avgs)-np.nanmin(avgs)
    v="ALL+" if allpos else "    "
    print(f"  {name:<18}"+"".join(f"{a:>9.1f}" for a in avgs)+f"   {v} range={spread_fold:.0f}")
    rows.append(dict(gate=name,f1=avgs[0],f2=avgs[1],f3=avgs[2],f4=avgs[3],
                     allpos=allpos,fold_range=round(spread_fold,1),
                     n=sum(ns)))
pd.DataFrame(rows).to_csv(RES/"mr_wiggle_gate.csv",index=False)
print("\nfold n's (G1 vs best):")
for name in ("G1 vol&ac (curr)","G2 eff_chop","G5 eff&ac"):
    g=GATES[name]; ns=[ (pool(g,k/4,(k+1)/4) or {'n':0})['n'] for k in range(4)]
    print(f"  {name:<18} {ns}")
print("\nSaved → results/mr_wiggle_gate.csv")
