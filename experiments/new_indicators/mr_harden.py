#!/usr/bin/env python3
"""
Harden the wiggle-gated daily MR → paper-test spec.
===================================================
Fixed: MR entries z-score(SMA10) ±2σ, gate eff_K_pct<0.5 & vol_pct<0.6 (tuned),
3-day hold, 12 pairs, post-warmup.

Three hardening checks:
 (1) OVERLAP-ROBUST SIGNIFICANCE. 3-day holds overlap and same-day trades across
     pairs are correlated → raw pooled t is inflated. Report:
       - raw pooled t (reference, inflated)
       - non-overlapping per-pair subset t (greedy, ≥HOLD apart → ~independent)
       - daily-clustered t (one obs per entry-date = mean pnl that day → robust to
         same-day cross-pair correlation)
       - stationary block-bootstrap p(mean<=0) on the daily-clustered series.
 (2) CAPACITY/FREQUENCY. trades/yr overall + per pair, max/mean concurrent open
     positions across the 12 pairs, avg hold, trades-per-year-bucket distribution.
 (3) SIZING STOP sweep {none,100,80,60,40p}: avg/t/WR + MAE tail + 4-fold WF — pick
     the widest stop that bounds the −650p tail while keeping edge & all-folds+.
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
SMA_N,ZT,HOLD,WARMUP=10,2.0,3,252
EFF_CUT,VOL_CUT=0.5,0.6

def build(pair):
    df=pd.read_parquet(DATA/f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df=df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    pip=pip_sz(pair); c=df["close"]
    day=df.index.normalize(); dayv=day.values
    nd=np.empty(len(dayv),bool); nd[0]=True; nd[1:]=dayv[1:]!=dayv[:-1]
    absd=c.diff().abs().values.copy(); absd[nd]=0.0
    gc=c.groupby(day); dnet=(gc.last()-gc.first()).abs()/pip
    dpath=pd.Series(absd,index=df.index).groupby(day).sum()/pip
    day_eff=dnet/dpath.replace(0,np.nan)
    d=df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last",
                              "bid_c":"last","ask_c":"last"}).dropna()
    dc=d["close"]; ret=dc.pct_change()
    z=(dc-dc.rolling(SMA_N).mean())/dc.rolling(SMA_N).std()
    d["sig"]=np.where(z<-ZT,1.0,np.where(z>ZT,-1.0,0.0))
    d["spread"]=(d["ask_c"]-d["bid_c"])/pip
    d["vol_pct"]=ret.rolling(20).std().rolling(252).rank(pct=True)
    d["eff_pct"]=day_eff.reindex(d.index).rolling(10).mean().rolling(252).rank(pct=True)
    d["date"]=d.index
    return d.reset_index(drop=True), pip

print("Loading + building 12 pairs …")
D={};
for p in PAIRS: D[p]=build(p)
N=len(D[PAIRS[0]][0]); YRS=(N-WARMUP)/252.0

def gated_trades(d, pip, stop=0.0, lo=0.0, hi=1.0, want_meta=False):
    c=d["close"].values; hi_=d["high"].values; lo_=d["low"].values
    sig=d["sig"].values; sp=d["spread"].values
    ek=d["eff_pct"].values; vp=d["vol_pct"].values; dates=d["date"].values
    n=len(d); s=WARMUP; A=s+int((n-s)*lo); B=s+int((n-s)*hi)
    pnl=[]; ent=[]; mae=[]; edate=[]
    for i in range(max(A,1),min(B,n-HOLD)):
        if sig[i]==0 or np.isnan(ek[i]) or np.isnan(vp[i]): continue
        if not (ek[i]<EFF_CUT and vp[i]<VOL_CUT): continue
        dr=sig[i]; em=c[i]; stop_lvl=(em-dr*stop*pip) if stop>0 else None
        worst=0.0; ex=None
        for j in range(i+1,i+HOLD+1):
            cur=((lo_[j]-em) if dr==1 else (em-hi_[j]))/pip
            if cur<worst: worst=cur
            if stop>0 and ((lo_[j]<=stop_lvl) if dr==1 else (hi_[j]>=stop_lvl)):
                ex=stop_lvl; break
        if ex is None: ex=c[i+HOLD]
        pnl.append((ex-em)*dr/pip-sp[i]); ent.append(i); mae.append(worst); edate.append(dates[i])
    if want_meta: return np.array(pnl),np.array(ent),np.array(mae),np.array(edate)
    return np.array(pnl)

def tstat(a): return a.mean()/(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 and a.std()>0 else np.nan

# ── (1) SIGNIFICANCE ──────────────────────────────────────────────────────────
raw=[]; nonov=[]; daily_rows=[]
for p in PAIRS:
    d,pip=D[p]
    pn,ent,mae,ed=gated_trades(d,pip,0.0,want_meta=True)
    if len(pn)==0: continue
    raw.append(pn)
    # non-overlapping greedy (entries >= HOLD apart)
    keep=[]; last=-10**9
    for k in range(len(ent)):
        if ent[k]>=last+HOLD: keep.append(k); last=ent[k]
    nonov.append(pn[keep])
    for k in range(len(pn)): daily_rows.append((ed[k],pn[k]))
raw=np.concatenate(raw); nonov=np.concatenate(nonov)
dd=pd.DataFrame(daily_rows,columns=["date","pnl"]).groupby("date")["pnl"].mean()
daily=dd.values

def block_boot(x, p_block=1/3, n=5000):
    # stationary bootstrap p(mean<=0)
    rng=np.random.default_rng(0); m=len(x); cnt=0
    for _ in range(n):
        idx=[]; i=rng.integers(m)
        while len(idx)<m:
            idx.append(i)
            if rng.random()<p_block: i=rng.integers(m)
            else: i=(i+1)%m
        if np.mean(x[idx[:m]])<=0: cnt+=1
    return cnt/n

print("\n[1] OVERLAP-ROBUST SIGNIFICANCE (gate eff<%.1f vol<%.1f)"%(EFF_CUT,VOL_CUT))
print(f"  raw pooled        : avg={raw.mean():+.2f}p  t={tstat(raw):.2f}  n={len(raw)}  (INFLATED ref)")
print(f"  non-overlapping   : avg={nonov.mean():+.2f}p  t={tstat(nonov):.2f}  n={len(nonov)}")
print(f"  daily-clustered   : avg={daily.mean():+.2f}p  t={tstat(daily):.2f}  n_days={len(daily)}")
print(f"  block-bootstrap p(mean<=0) on daily series: {block_boot(daily):.4f}")

# ── (2) CAPACITY / CONCURRENCY ────────────────────────────────────────────────
occ={}; per_pair_n={}; holds=[]; yr_counts={}
for p in PAIRS:
    d,pip=D[p]
    pn,ent,mae,ed=gated_trades(d,pip,0.0,want_meta=True)
    per_pair_n[p]=len(pn)
    for k in range(len(ent)):
        i=ent[k]
        for j in range(i,i+HOLD):
            dt=d["date"].values[min(j,len(d)-1)]
            occ[dt]=occ.get(dt,0)+1
        holds.append(HOLD)
        y=pd.Timestamp(ed[k]).year; yr_counts[y]=yr_counts.get(y,0)+1
occ_s=pd.Series(occ)
tot=sum(per_pair_n.values())
print("\n[2] CAPACITY / FREQUENCY")
print(f"  total gated trades: {tot}  →  {tot/YRS:.0f}/yr across 12 pairs ({tot/YRS/12:.1f}/pair/yr)")
print(f"  per-pair trades   : "+", ".join(f"{p.split('_')[0]}{per_pair_n[p]}" for p in PAIRS))
print(f"  concurrent open positions across 12 pairs: max={int(occ_s.max())}  mean={occ_s.mean():.2f}  p95={int(np.percentile(occ_s,95))}")
print(f"  trades by year    : "+", ".join(f"{y}:{yr_counts[y]}" for y in sorted(yr_counts)))

# ── (3) SIZING STOP SWEEP ─────────────────────────────────────────────────────
print("\n[3] SIZING-STOP SWEEP (full sample + 4-fold WF)")
print(f"  {'stop':>6}{'avg':>8}{'t':>6}{'wr':>5}{'n':>6}{'MAE_p99':>9}{'MAE_wst':>9}   4-fold WF")
for stop in (0,100,80,60,40):
    allp=[]; allmae=[]
    folds=[[] for _ in range(4)]
    for p in PAIRS:
        d,pip=D[p]
        pn,ent,mae,ed=gated_trades(d,pip,float(stop),want_meta=True)
        if len(pn): allp.append(pn); allmae.append(mae)
        for k in range(4):
            fp=gated_trades(d,pip,float(stop),k/4,(k+1)/4)
            if len(fp): folds[k].append(fp)
    a=np.concatenate(allp); m=np.concatenate(allmae)
    fav=[np.concatenate(f).mean() if f else np.nan for f in folds]
    allpos=all((not np.isnan(x)) and x>0 for x in fav)
    tag="none" if stop==0 else f"{stop}p"
    print(f"  {tag:>6}{a.mean():>8.2f}{tstat(a):>6.2f}{(a>0).mean()*100:>4.0f}%{len(a):>6d}"
          f"{np.percentile(m,1):>9.0f}{m.min():>9.0f}   "
          +" ".join(f"{x:+.0f}" for x in fav)+("  ALL+" if allpos else ""))
print("\nDone.")
