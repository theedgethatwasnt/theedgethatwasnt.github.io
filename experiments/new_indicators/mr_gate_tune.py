#!/usr/bin/env python3
"""
Tune the G6 wiggle-gate cutoffs — IS-select (stability), OOS-seal.
=================================================================
G6 = trade daily MR only when  eff_K_pct < eff_cut  AND  vol_pct < vol_cut.
Currently both cutoffs sit at the 0.5 median; sweep them properly.

Discipline (don't overfit thresholds):
  - Sweep the 5×5 cutoff grid on IS ONLY (first 70% of post-warmup range).
  - Select by STABILITY: maximize the WORSE of the two IS sub-fold avgs
    (min(IS_fold1, IS_fold2)), subject to a capacity floor n_IS >= 120.
    (We want a reliably-positive edge, not a peak average.)
  - Then evaluate the chosen cutoffs ONCE on sealed OOS (last 30%): avg/t/wr/
    pairs/2-fold + capacity. Show full 4-fold WF for context (folds 3-4 = OOS,
    NOT used in selection).
Same MR entries throughout; vary only the gate cutoffs. Read-only on data/m5_ba.
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
IS_FRAC=0.70
CUTS=[0.3,0.4,0.5,0.6,0.7]
N_FLOOR=120

def build(pair):
    df=pd.read_parquet(DATA/f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df=df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    pip=pip_sz(pair); c=df["close"]
    day=df.index.normalize(); dayv=day.values
    new_day=np.empty(len(dayv),bool); new_day[0]=True; new_day[1:]=dayv[1:]!=dayv[:-1]
    absd_v=c.diff().abs().values.copy(); absd_v[new_day]=0.0
    gc=c.groupby(day)
    dnet=(gc.last()-gc.first()).abs()/pip
    dpath=pd.Series(absd_v,index=df.index).groupby(day).sum()/pip
    day_eff=dnet/dpath.replace(0,np.nan)
    d=df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last",
                              "bid_c":"last","ask_c":"last"}).dropna()
    dc=d["close"]; ret=dc.pct_change()
    z=(dc-dc.rolling(SMA_N).mean())/dc.rolling(SMA_N).std()
    d["sig"]=np.where(z<-ZT,1.0,np.where(z>ZT,-1.0,0.0))
    d["spread"]=(d["ask_c"]-d["bid_c"])/pip; d["pip"]=pip
    d["vol_pct"]=ret.rolling(20).std().rolling(252).rank(pct=True)
    d["eff_K_pct"]=day_eff.reindex(d.index).rolling(10).mean().rolling(252).rank(pct=True)
    # precompute per-trade pnl + gate features as arrays
    return d.reset_index(drop=True)

print("Loading + building 12 pairs …"); D={p:build(p) for p in PAIRS}

def pnls_window(d, eff_cut, vol_cut, lo, hi):
    c=d["close"].values; sig=d["sig"].values; sp=d["spread"].values; pip=d["pip"].values[0]
    ek=d["eff_K_pct"].values; vp=d["vol_pct"].values
    n=len(d); s=WARMUP; A=s+int((n-s)*lo); B=s+int((n-s)*hi)
    out=[]
    for i in range(max(A,1),min(B,n-HOLD)):
        if sig[i]==0: continue
        if np.isnan(ek[i]) or np.isnan(vp[i]): continue
        if not (ek[i]<eff_cut and vp[i]<vol_cut): continue
        out.append((c[i+HOLD]-c[i])*sig[i]/pip - sp[i])
    return np.array(out)

def pool(eff_cut,vol_cut,lo,hi):
    allp=[]; pp=0
    for p in PAIRS:
        v=pnls_window(D[p],eff_cut,vol_cut,lo,hi)
        if len(v): allp.append(v); pp+=1 if v.mean()>0 else 0
    if not allp: return None
    a=np.concatenate(allp)
    t=a.mean()/(a.std(ddof=1)/np.sqrt(len(a))) if a.std()>0 else np.nan
    return dict(n=len(a),avg=a.mean(),t=t,wr=(a>0).mean()*100,pairs=pp)

# years in post-warmup range (approx, from one pair)
nd=len(D[PAIRS[0]]); yrs_full=(nd-WARMUP)/252.0

# ── IS sweep (first 70% post-warmup), select by stability ─────────────────────
print(f"\n[IS cutoff sweep — select by min(IS sub-fold avg), n_IS>={N_FLOOR}]")
print(f"  {'eff<':>5}{'vol<':>5}{'IS_avg':>8}{'IS_t':>6}{'n_IS':>6}{'IS_f1':>7}{'IS_f2':>7}{'minf':>7}")
rows=[]
for ec in CUTS:
    for vc in CUTS:
        full=pool(ec,vc,0.0,IS_FRAC)
        if full is None: continue
        f1=pool(ec,vc,0.0,IS_FRAC/2); f2=pool(ec,vc,IS_FRAC/2,IS_FRAC)
        a1=f1['avg'] if f1 else np.nan; a2=f2['avg'] if f2 else np.nan
        minf=np.nanmin([a1,a2])
        rows.append(dict(ec=ec,vc=vc,IS_avg=full['avg'],IS_t=full['t'],n_IS=full['n'],
                         IS_f1=a1,IS_f2=a2,minf=minf))
sw=pd.DataFrame(rows)
for _,r in sw.sort_values("minf",ascending=False).iterrows():
    print(f"  {r['ec']:>5.1f}{r['vc']:>5.1f}{r['IS_avg']:>8.1f}{r['IS_t']:>6.2f}{int(r['n_IS']):>6d}"
          f"{r['IS_f1']:>7.1f}{r['IS_f2']:>7.1f}{r['minf']:>7.1f}")

elig=sw[sw["n_IS"]>=N_FLOOR].copy()
win=elig.sort_values("minf",ascending=False).iloc[0]
ec,vc=float(win["ec"]),float(win["vc"])
print(f"\nSelected cutoffs (IS, stability): eff<{ec}  vol<{vc}  "
      f"[IS avg={win['IS_avg']:.1f} t={win['IS_t']:.2f} n={int(win['n_IS'])} minfold={win['minf']:.1f}]")

# ── Sealed OOS (last 30%) ─────────────────────────────────────────────────────
oos=pool(ec,vc,IS_FRAC,1.0)
of1=pool(ec,vc,IS_FRAC,IS_FRAC+(1-IS_FRAC)/2); of2=pool(ec,vc,IS_FRAC+(1-IS_FRAC)/2,1.0)
print(f"\n[Sealed OOS — evaluated once]  eff<{ec} vol<{vc}")
print(f"  avg={oos['avg']:+.2f}p  t={oos['t']:.2f}  wr={oos['wr']:.0f}%  n={oos['n']}  pairs+={oos['pairs']}/12")
print(f"  OOS 2-fold: {of1['avg']:+.1f} / {of2['avg']:+.1f}  ({'both+' if of1['avg']>0 and of2['avg']>0 else 'NOT both+'})")
cap_yr=oos['n']/ (yrs_full*(1-IS_FRAC))
print(f"  capacity ≈ {cap_yr:.0f} trades/yr across 12 pairs ({cap_yr/12:.1f}/pair/yr)")

# ── Full 4-fold WF for context (folds 3-4 = OOS, not used in selection) ───────
print(f"\n[Full 4-fold WF — context]  eff<{ec} vol<{vc}")
fs=[pool(ec,vc,k/4,(k+1)/4) for k in range(4)]
print("  "+" ".join(f"f{i+1}={f['avg']:+.1f}(n{f['n']})" if f else f"f{i+1}=NA" for i,f in enumerate(fs))
      +f"   {'ALL+' if all(f and f['avg']>0 for f in fs) else 'not all+'}")

# ── vs median baseline (0.5,0.5) ──────────────────────────────────────────────
b=pool(0.5,0.5,IS_FRAC,1.0)
print(f"\n[Compare] median(0.5,0.5) OOS: avg={b['avg']:+.1f} t={b['t']:.2f} n={b['n']} pairs+={b['pairs']}/12")
print(f"          tuned({ec},{vc}) OOS: avg={oos['avg']:+.1f} t={oos['t']:.2f} n={oos['n']} pairs+={oos['pairs']}/12")
sw.to_csv(RES/"mr_gate_tune.csv",index=False)
print("\nSaved → results/mr_gate_tune.csv")
