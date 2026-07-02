#!/usr/bin/env python3
"""
Deepen the one real lead: daily-horizon mean-reversion.
========================================================
The multi-TF screen found daily RSI2 mean-reversion was the only signal with a
spread-net edge (+4.7 pips/trade, IC +0.14, 10/12 pairs) but weak per-pair power
(1/12 t>2). Here we deepen it properly:

  1. SWEEP mean-reversion configs (RSI + z-score variants × thresholds × hold) on
     IS ONLY, POOLING all 12 pairs' trades for statistical power.
  2. Pick the winner by pooled IS t-stat (with n>=200 and >=8 pairs positive).
  3. Confirm that ONE config on OOS exactly once (SOP R8 — OOS sealed).
  4. Stress the winner:
       a. STOP-LOSS sweep — does the edge survive a stop? (the momentum book's
          "edge" vanished with any stop; a real MR edge should tolerate one.)
       b. MAE distribution — is there a hidden falling-knife tail?
       c. Kaufman Efficiency-Ratio regime filter — fade only in choppy regimes.

Honest gate throughout: enter at signal-bar close (pay ask/bid), exit after `hold`
days at close (bid/ask) or earlier if stop hit. Spread deducted by construction.
Read-only on data/m5_ba.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
RESULTS = Path(__file__).parent / "results"; RESULTS.mkdir(exist_ok=True)

PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY   = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC = 0.70
def pip_sz(p): return 0.01 if p in JPY else 0.0001


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn.replace(0, np.nan))

def zscore(c, k):
    m = c.rolling(k).mean(); s = c.rolling(k).std()
    return (c - m) / s.replace(0, np.nan)

def kaufman_er(c, n):
    chg = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n).sum()
    return chg / vol.replace(0, np.nan)


def trades_for_signal(d, sig, pip, hold, stop_pips=0.0):
    """Per-trade pnl (pips) + MAE (pips), all in pip units. Mid-based path; one
    round-trip spread deducted at entry. If stop_pips>0 and the intrabar daily
    low/high (for long/short) breaches the stop level, exit at the stop."""
    c = d["close"].values; hi = d["high"].values; lo = d["low"].values
    bid = d["bid_c"].values; ask = d["ask_c"].values
    s = sig.values.astype(float); n = len(c)
    pnls = []; maes = []
    for i in range(n - hold):
        if s[i] == 0: continue
        dir_ = s[i]
        em = c[i]                                   # entry mid
        sp = (ask[i] - bid[i]) / pip                # round-trip spread cost (pips)
        stop_lvl = (em - dir_ * stop_pips * pip) if stop_pips > 0 else None
        worst = 0.0; exit_mid = None
        for j in range(i+1, i+hold+1):
            cur = ((lo[j]-em) if dir_ == 1 else (em-hi[j])) / pip   # adverse excursion, pips (<0)
            if cur < worst: worst = cur
            if stop_pips > 0:
                if (lo[j] <= stop_lvl) if dir_ == 1 else (hi[j] >= stop_lvl):
                    exit_mid = stop_lvl; break
        if exit_mid is None:
            exit_mid = c[i+hold]
        pnls.append((exit_mid - em) * dir_ / pip - sp)
        maes.append(worst)
    return np.array(pnls), np.array(maes)


def build_daily():
    cache = {}
    for pair in PAIRS:
        df = pd.read_parquet(DATA / f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
        df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
        d = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last",
                                    "bid_c":"last","ask_c":"last"}).dropna()
        cache[pair] = (d, pip_sz(pair), int(len(d)*IS_FRAC))
    return cache

def make_signal(d, kind, p):
    if kind == "rsi":
        r = rsi(d["close"], p["n"]); s = pd.Series(0.0, index=d.index)
        s[r < p["lo"]] = 1.0; s[r > p["hi"]] = -1.0
        return s
    else:  # zscore
        z = zscore(d["close"], p["k"]); s = pd.Series(0.0, index=d.index)
        s[z < -p["zt"]] = 1.0; s[z > p["zt"]] = -1.0
        return s

def pooled_stats(cache, kind, p, hold, lo_frac, hi_frac, stop=0.0):
    """Pool trades across pairs over the index window [lo_frac,hi_frac) of each pair."""
    allp = []; pairs_pos = 0; per_pair = []
    for pair,(d,pip,n_is) in cache.items():
        n = len(d); lo = int(n*lo_frac); hi = int(n*hi_frac)
        sig = make_signal(d, kind, p).copy()
        mask = np.zeros(n, dtype=bool); mask[lo:hi] = True
        sig[~pd.Series(mask, index=d.index)] = 0.0
        pn, _ = trades_for_signal(d, sig, pip, hold, stop)
        if len(pn) > 0:
            allp.append(pn); per_pair.append(pn.mean())
            if pn.mean() > 0: pairs_pos += 1
    if not allp: return None
    arr = np.concatenate(allp)
    t = arr.mean()/(arr.std(ddof=1)/np.sqrt(len(arr))) if arr.std()>0 else np.nan
    return dict(ntr=len(arr), avg=arr.mean(), t=t, pairs_pos=pairs_pos,
                wr=float((arr>0).mean()*100))

print("Loading daily bars …"); cache = build_daily()
days_oos = np.mean([ (len(d)-n_is) for d,_,n_is in cache.values() ])
print(f"  12 pairs, ~{int(np.mean([len(d) for d,_,_ in cache.values()]))} daily bars each.\n")

# ── 1. IS sweep (pool, IS window = [0, IS_FRAC)) ─────────────────────────────────
configs = []
for n in (2,3):
    for lo,hi in ((10,90),(15,85),(20,80)):
        for hold in (1,2,3):
            configs.append(("rsi", dict(n=n,lo=lo,hi=hi), hold))
for k in (10,20):
    for zt in (1.5,2.0):
        for hold in (1,2,3):
            configs.append(("zscore", dict(k=k,zt=zt), hold))

print(f"Sweeping {len(configs)} mean-reversion configs on IS (pooled 12 pairs) …")
rows = []
for kind,p,hold in configs:
    st = pooled_stats(cache, kind, p, hold, 0.0, IS_FRAC)
    if st is None: continue
    label = f"{kind}({p})_h{hold}"
    rows.append(dict(label=label, kind=kind, **p, hold=hold, **st))
sw = pd.DataFrame(rows).sort_values("t", ascending=False)
sw.to_csv(RESULTS/"mr_is_sweep.csv", index=False)
print(sw[["label","ntr","avg","t","wr","pairs_pos"]].head(12).to_string(index=False))

# ── 2. pick winner (IS t-stat, n>=200, pairs_pos>=8) ─────────────────────────────
elig = sw[(sw["ntr"]>=200) & (sw["pairs_pos"]>=8)]
if len(elig)==0: elig = sw
win = elig.iloc[0]
print(f"\nWinner by IS pooled t-stat: {win['label']}  "
      f"IS avg={win['avg']:+.2f}p t={win['t']:.2f} wr={win['wr']:.0f}% pairs+={int(win['pairs_pos'])}/12")
w_kind = win["kind"]; w_hold = int(win["hold"])
if w_kind=="rsi": w_params={"n":int(win["n"]),"lo":int(win["lo"]),"hi":int(win["hi"])}
else: w_params={"k":int(win["k"]),"zt":float(win["zt"])}

# ── 3. OOS confirm (once) ────────────────────────────────────────────────────────
oos = pooled_stats(cache, w_kind, w_params, w_hold, IS_FRAC, 1.0)
port_pd = oos["avg"] * oos["ntr"] / days_oos
print(f"\n[OOS — sealed, evaluated once]  {win['label']}")
print(f"  avg={oos['avg']:+.2f}p  t={oos['t']:.2f}  wr={oos['wr']:.0f}%  "
      f"n={oos['ntr']}  pairs+={oos['pairs_pos']}/12  portfolio p/d≈{port_pd:+.1f}")

# ── 3b. Walk-forward consistency (4 sequential folds) on winner ───────────────────
print(f"\n[Walk-forward — 4 sequential folds]  is the edge consistent or regime-luck?")
print(f"  {'fold':>6} {'avg_pnl':>8} {'t':>6} {'wr':>5} {'n':>6} {'pairs+':>7}")
for k in range(4):
    f = pooled_stats(cache, w_kind, w_params, w_hold, k/4, (k+1)/4)
    if f is None: continue
    print(f"  {k+1:>6} {f['avg']:>+8.2f} {f['t']:>6.2f} {f['wr']:>4.0f}% {f['ntr']:>6d} {f['pairs_pos']:>5d}/12")

# ── 4a. stop-loss sweep on winner (OOS) ──────────────────────────────────────────
print(f"\n[Stop-loss sensitivity — OOS]  does the edge survive a stop? "
      f"(momentum book did NOT)")
print(f"  {'stop':>6} {'avg_pnl':>8} {'t':>6} {'wr':>5} {'n':>6}")
for stop in (0, 100, 60, 40, 25, 15):
    s = pooled_stats(cache, w_kind, w_params, w_hold, IS_FRAC, 1.0, stop=float(stop))
    tag = "none" if stop==0 else f"{stop}p"
    print(f"  {tag:>6} {s['avg']:>+8.2f} {s['t']:>6.2f} {s['wr']:>4.0f}% {s['ntr']:>6d}")

# ── 4b. MAE distribution on winner (OOS, no stop) ────────────────────────────────
maes_all=[]
for pair,(d,pip,n_is) in cache.items():
    sig = make_signal(d, w_kind, w_params).copy()
    m=np.zeros(len(d),bool); m[n_is:]=True; sig[~pd.Series(m,index=d.index)]=0.0
    _, mae = trades_for_signal(d, sig, pip, w_hold, 0.0)
    if len(mae): maes_all.append(mae)
mae=np.concatenate(maes_all)
print(f"\n[MAE distribution — OOS, no stop]  falling-knife check:")
print(f"  median={np.percentile(mae,50):.0f}p  p90(worst)={np.percentile(mae,10):.0f}p  "
      f"p99={np.percentile(mae,1):.0f}p  worst={mae.min():.0f}p")

# ── 4c. Kaufman ER regime filter (OOS) ───────────────────────────────────────────
print(f"\n[Kaufman ER regime filter — OOS]  fade only when ER < thr (choppy):")
print(f"  {'er_thr':>6} {'avg_pnl':>8} {'t':>6} {'wr':>5} {'n':>6}")
for er_thr in (1.01, 0.5, 0.4, 0.3, 0.25):
    allp=[]
    for pair,(d,pip,n_is) in cache.items():
        sig = make_signal(d, w_kind, w_params).copy()
        er = kaufman_er(d["close"], 10)
        gate = (er < er_thr)
        m=np.zeros(len(d),bool); m[n_is:]=True
        sig[~(pd.Series(m,index=d.index) & gate.fillna(False))] = 0.0
        pn,_=trades_for_signal(d,sig,pip,w_hold,0.0)
        if len(pn): allp.append(pn)
    arr=np.concatenate(allp) if allp else np.array([0.0])
    t=arr.mean()/(arr.std(ddof=1)/np.sqrt(len(arr))) if arr.std()>0 else np.nan
    tag="off" if er_thr>1 else f"{er_thr}"
    print(f"  {tag:>6} {arr.mean():>+8.2f} {t:>6.2f} {(arr>0).mean()*100:>4.0f}% {len(arr):>6d}")

print(f"\nSaved → results/mr_is_sweep.csv")
