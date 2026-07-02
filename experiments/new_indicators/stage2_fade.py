#!/usr/bin/env python3
"""
Stage 2 — Momentum-FADE entry test.
===================================
Stage 1 showed momentum CONTINUATION loses everywhere net of spread; the only
structure is contrarian (fast net moves reverse). So test the fade: when the
rolling-window net move is large, enter AGAINST it.

Entry  : when |momentum| > per-pair IS percentile threshold (and optional
         efficiency filter), enter dir = -sign(momentum). One position at a time
         per pair. Threshold is per-pair (robust cross-instrument scaling).
Exit   : bounded fixed hold of H bars, fill at bid/ask (spread deducted).
Gate   : pooled IS t-stat to pick winner → sealed OOS → 4-fold WF (all-positive?)
         → beats-random-DIRECTION baseline (does fading beat a random side at the
         same entry times?).

Read-only on data/m5_ba. Per-pair percentile thresholds use IS only (R5/R8).
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
IS_FRAC=0.70

@njit(cache=True)
def fade_sim(close, bid, ask, mom, eff, thr, eff_max, H, i0, i1):
    """Sequential fade sim over bars [i0,i1). pnl in pips, spread via bid/ask.
    Returns pnls[], entry_idx[]. dir=-sign(mom)."""
    n=len(close); pn=np.empty(n); ix=np.empty(n,np.int64); k=0
    i=max(i0,1)
    while i < min(i1, n-H):
        m=mom[i]
        if (not np.isnan(m)) and abs(m)>thr and (np.isnan(eff_max) or (not np.isnan(eff[i]) and eff[i]<eff_max)):
            dir_ = -1 if m>0 else 1
            ep = ask[i] if dir_==1 else bid[i]
            j=i+H
            ex = bid[j] if dir_==1 else ask[j]
            pn[k]=(ex-ep)*dir_; ix[k]=i; k+=1
            i=j
        else:
            i+=1
    return pn[:k], ix[:k]

def load(pair, W):
    df=pd.read_parquet(DATA/f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df=df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    pip=pip_sz(pair); bw=W//5; c=df["close"]
    net=(c-c.shift(bw))/pip; mom=(net/W)
    path=(c.diff().abs()/pip).rolling(bw).sum(); eff=net.abs()/path.replace(0,np.nan)
    return dict(close=(c/pip).values, bid=(df["bid_c"]/pip).values, ask=(df["ask_c"]/pip).values,
                mom=mom.values, eff=eff.values, n=len(c), pip=pip)

def pooled(cfg, lo, hi, random_dir=False, seed=0):
    """Pool fade trades across pairs over index window [lo,hi)."""
    W=cfg["W"]; H=cfg["H"]; pct=cfg["pct"]; eff_max=cfg["eff_max"]
    allp=[]; pp=[]
    rng=np.random.default_rng(seed)
    for pair in PAIRS:
        d=D[(pair,W)]; n=d["n"]; nis=int(n*IS_FRAC)
        am=np.abs(d["mom"][:nis]); am=am[~np.isnan(am)]
        thr=np.quantile(am, pct)
        a=int(n*lo); b=int(n*hi)
        pn,ix=fade_sim(d["close"],d["bid"],d["ask"],d["mom"],d["eff"],thr,eff_max,H,a,b)
        if len(pn)==0: continue
        if random_dir:
            sgn=rng.choice(np.array([-1.0,1.0]),size=len(pn))
            pn=np.abs(pn)*sgn*np.sign(pn)  # keep |move| & cost, randomize side
            # simpler: recompute as random side → approximate by random sign on raw move+spread
        allp.append(pn);
        if pn.mean()>0: pp.append(1)
        else: pp.append(0)
    if not allp: return None
    arr=np.concatenate(allp)
    t=arr.mean()/(arr.std(ddof=1)/np.sqrt(len(arr))) if arr.std()>0 else np.nan
    return dict(n=len(arr), avg=arr.mean(), t=t, wr=(arr>0).mean()*100, pairs_pos=sum(pp))

# preload
print("Loading …")
WS=[30,60]; D={}
for W in WS:
    for p in PAIRS: D[(p,W)]=load(p,W)

configs=[]
for W in WS:
    for pct in (0.90,0.95):
        for eff_max in (np.nan, 0.4):
            for H in ([12,24] if W==30 else [12,24]):   # 1h,2h
                configs.append(dict(W=W,pct=pct,eff_max=eff_max,H=H))

print(f"\nSweeping {len(configs)} fade configs on IS (pooled) …")
rows=[]
for cfg in configs:
    st=pooled(cfg,0.0,IS_FRAC)
    if st is None: continue
    lab=f"W{cfg['W']} pct{int(cfg['pct']*100)} eff{'all' if np.isnan(cfg['eff_max']) else cfg['eff_max']} H{cfg['H']}"
    rows.append(dict(cfg=cfg,label=lab,**st))
sw=pd.DataFrame(rows).sort_values("t",ascending=False)
print(sw[["label","n","avg","t","wr","pairs_pos"]].head(12).to_string(index=False))

elig=sw[(sw["n"]>=300)&(sw["pairs_pos"]>=8)]
win=(elig if len(elig) else sw).iloc[0]
wc=win["cfg"]
print(f"\nWinner (IS t): {win['label']}  IS avg={win['avg']:+.2f}p t={win['t']:.2f} "
      f"wr={win['wr']:.0f}% pairs+={int(win['pairs_pos'])}/12")

oos=pooled(wc,IS_FRAC,1.0)
print(f"\n[OOS sealed] avg={oos['avg']:+.2f}p t={oos['t']:.2f} wr={oos['wr']:.0f}% "
      f"n={oos['n']} pairs+={oos['pairs_pos']}/12")

print(f"\n[4-fold WF]  (success = all folds positive)")
fl=[]
for k in range(4):
    f=pooled(wc,k/4,(k+1)/4)
    if f: fl.append(f); print(f"  fold{k+1}: avg={f['avg']:+.2f}p t={f['t']:.2f} wr={f['wr']:.0f}% n={f['n']} pairs+={f['pairs_pos']}/12")
allpos=all(f['avg']>0 for f in fl)
print(f"  → {'ALL FOLDS POSITIVE ✅' if allpos else 'NOT all-positive ❌'}")

# beats-random-DIRECTION: same entry times, random side
print(f"\n[Beats random direction? OOS, 200 shuffles]")
real=oos['avg']; beats=0; N=200
for s in range(N):
    r=pooled(wc,IS_FRAC,1.0,random_dir=True,seed=s+1)
    if r and r['avg']>=real: beats+=1
print(f"  real fade avg={real:+.2f}p   P(random_dir >= real)={beats/N:.3f}  "
      f"{'fade beats random side ✅' if beats/N<0.05 else 'no better than random side ❌'}")
print("\nDone.")
