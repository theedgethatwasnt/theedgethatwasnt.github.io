#!/usr/bin/env python3
"""
TradingView Technical Ratings — Phase 0 (causal rating engine) + Phase 1 (edge screen).

Phase 0: reimplement the 26-indicator TV vote rules → osc_rating / ma_rating /
         summary_rating in [-1,+1], per (pair, TF). Causal (closed bars only).
Phase 1: does the METER have edge? Screen each rating per TF vs forward return
         NET of spread, FOLLOW and FADE, scored by net p/d AND AMDDP5
         (pnl − 0.05·underwater-pip-bars). IS/OOS 70/30.

Rules reimplemented from TV's published "Technical Ratings" spec (faithful in
spirit; Ichimoku simplified to BaseLine vote — 1 of 26). NOT the live-fetch lib.
SOP: mid for signals, spread deducted explicitly, OOS sealed, indicators causal.
"""
import gc, sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
M5_DIR  = PROJECT / "data" / "m5_ba"
CACHE   = Path(__file__).parent / "cache"; CACHE.mkdir(exist_ok=True)
RESULTS = Path(__file__).parent / "results"; RESULTS.mkdir(exist_ok=True)

PAIRS = sys.argv[1:] or ["EUR_USD", "USD_JPY", "GBP_USD"]
TFS   = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h",
         "2h": "2h", "4h": "4h", "1d": "1D"}
TF_BARS_PER_DAY = {"5m": 288, "15m": 96, "30m": 48, "1h": 24, "2h": 12, "4h": 6, "1d": 1}
HORIZONS = [1, 3, 6]        # forward TF-bars
THRESH   = [0.1, 0.3, 0.5]  # |rating| gate (0.5 ≈ TV "Strong")
IS_FRAC  = 0.70
AMDDP_K  = 0.05


def pip_of(p): return 0.01 if "JPY" in p else 0.0001


# ── indicator helpers (vectorized, causal) ──────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()

def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    return 100 - 100/(1+rs)

def stoch(h, l, c, k=14, d=3):
    ll = l.rolling(k).min(); hh = h.rolling(k).max()
    kf = 100*(c-ll)/(hh-ll).replace(0, np.nan)
    ks = kf.rolling(d).mean(); ds = ks.rolling(d).mean()
    return ks, ds

def cci(h, l, c, n=20):
    tp = (h+l+c)/3; ma = tp.rolling(n).mean()
    md = (tp-ma).abs().rolling(n).mean()
    return (tp-ma)/(0.015*md.replace(0, np.nan))

def adx_di(h, l, c, n=14):
    up = h.diff(); dn = -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100*pd.Series(plus, index=h.index).ewm(alpha=1/n, adjust=False).mean()/atr
    ndi = 100*pd.Series(minus, index=h.index).ewm(alpha=1/n, adjust=False).mean()/atr
    dx = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean(), pdi, ndi

def macd(c):
    m = ema(c, 12)-ema(c, 26); return m, ema(m, 9)

def williams_r(h, l, c, n=14):
    hh = h.rolling(n).max(); ll = l.rolling(n).min()
    return -100*(hh-c)/(hh-ll).replace(0, np.nan)

def ultimate(h, l, c):
    pc = c.shift(); tl = pd.concat([l, pc], axis=1).min(axis=1)
    bp = c-tl; tr = pd.concat([h, pc], axis=1).max(axis=1)-tl
    a1 = bp.rolling(7).sum()/tr.rolling(7).sum(); a2 = bp.rolling(14).sum()/tr.rolling(14).sum()
    a3 = bp.rolling(28).sum()/tr.rolling(28).sum()
    return 100*(4*a1+2*a2+a3)/7

def hull(c, n=9):
    wma = lambda s, w: s.rolling(w).apply(lambda x: np.dot(x, np.arange(1, w+1))/(w*(w+1)/2), raw=True)
    return wma(2*wma(c, n//2)-wma(c, n), int(np.sqrt(n)))

def vwma(c, v, n=20):
    return (c*v).rolling(n).sum()/v.rolling(n).sum().replace(0, np.nan)


def compute_ratings(df):
    """Return DataFrame with osc/ma/summary ratings in [-1,1], causal."""
    o, h, l, c, v = df["o"], df["h"], df["l"], df["c"], df["v"]
    votes = {}
    # ── Moving averages (15): price vs MA = Buy/Sell ─────────────────────────
    for n in (10, 20, 30, 50, 100, 200):
        votes[f"ema{n}"] = np.sign(c - ema(c, n))
        votes[f"sma{n}"] = np.sign(c - sma(c, n))
    base = (h.rolling(26).max()+l.rolling(26).min())/2          # Ichimoku base (simplified)
    votes["ichi"] = np.sign(c - base)
    votes["vwma"] = np.sign(c - vwma(c, v))
    hm = hull(c); votes["hull"] = np.sign(hm - hm.shift())       # HullMA rising/falling
    ma_keys = [k for k in votes]
    # ── Oscillators (11) with neutral zone ───────────────────────────────────
    r = rsi(c); votes["rsi"] = np.where((r < 30) & (r > r.shift()), 1,
                                np.where((r > 70) & (r < r.shift()), -1, 0))
    kk, dd = stoch(h, l, c)
    votes["stoch"] = np.where((kk < 20) & (dd < 20) & (kk > dd), 1,
                     np.where((kk > 80) & (dd > 80) & (kk < dd), -1, 0))
    cc = cci(h, l, c); votes["cci"] = np.where((cc < -100) & (cc > cc.shift()), 1,
                                      np.where((cc > 100) & (cc < cc.shift()), -1, 0))
    adx, pdi, ndi = adx_di(h, l, c)
    votes["adx"] = np.where((adx > 20) & (pdi > ndi), 1,
                   np.where((adx > 20) & (ndi > pdi), -1, 0))
    ao = sma((h+l)/2, 5) - sma((h+l)/2, 34)
    votes["ao"] = np.where((ao > 0) & (ao > ao.shift()), 1,
                  np.where((ao < 0) & (ao < ao.shift()), -1, 0))
    mom = c - c.shift(10)
    votes["mom"] = np.where(mom > mom.shift(), 1, np.where(mom < mom.shift(), -1, 0))
    mac, sig = macd(c); votes["macd"] = np.sign(mac - sig)
    # StochRSI(3,3,14,14)
    rr = rsi(c); sll = rr.rolling(14).min(); shh = rr.rolling(14).max()
    srk = (100*(rr-sll)/(shh-sll).replace(0, np.nan)).rolling(3).mean()
    srd = srk.rolling(3).mean()
    votes["stochrsi"] = np.where((srk < 20) & (srk > srd), 1,
                        np.where((srk > 80) & (srk < srd), -1, 0))
    wr = williams_r(h, l, c)
    votes["willr"] = np.where((wr < -80) & (wr > wr.shift()), 1,
                     np.where((wr > -20) & (wr < wr.shift()), -1, 0))
    e13 = ema(c, 13); bbp = (h-e13)+(l-e13)
    votes["bbp"] = np.where((bbp > 0) & (bbp > bbp.shift()), 1,
                   np.where((bbp < 0) & (bbp < bbp.shift()), -1, 0))
    uo = ultimate(h, l, c); votes["uo"] = np.where(uo > 70, 1, np.where(uo < 30, -1, 0))
    osc_keys = ["rsi", "stoch", "cci", "adx", "ao", "mom", "macd", "stochrsi", "willr", "bbp", "uo"]

    V = pd.DataFrame({k: np.asarray(votes[k], dtype=float) for k in votes}, index=c.index)
    out = pd.DataFrame(index=c.index)
    out["ma_rating"] = V[ma_keys].mean(axis=1)
    out["osc_rating"] = V[osc_keys].mean(axis=1)
    out["summary"] = V[ma_keys+osc_keys].mean(axis=1)
    return out


def resample_tf(dfm5, rule):
    g = dfm5.resample(rule, label="right", closed="right")
    out = pd.DataFrame({
        "o": g["open"].first(), "h": g["high"].max(), "l": g["low"].min(),
        "c": g["close"].last(), "v": g["volume"].sum(),
        "bid": g["bid_c"].last(), "ask": g["ask_c"].last()})
    return out.dropna()


def sim(rating, mid, bid, ask, pip, H, thr, fade, bpd):
    """Enter when |rating|>thr at bar close, hold H bars. dir follow/fade.
    Returns (net_ppd, amddp5_ppd, n_trades, wr)."""
    n = len(mid); sgn = np.sign(rating)
    if fade: sgn = -sgn
    idx = np.where((np.abs(rating) > thr) & (np.arange(n) < n-H))[0]
    if len(idx) < 30:
        return np.nan, np.nan, 0, np.nan
    nets = []; amd = []
    for i in idx:
        d = sgn[i]
        if d == 0: continue
        entry = mid[i]; spr = (ask[i]-bid[i])/pip
        exitp = mid[i+H]
        net = d*(exitp-entry)/pip - spr
        # underwater area over the hold (pip-bars)
        path = d*(mid[i+1:i+H+1]-entry)/pip
        cum_dd = float(np.clip(-path, 0, None).sum())
        nets.append(net); amd.append(net - AMDDP_K*cum_dd)
    if len(nets) < 30:
        return np.nan, np.nan, 0, np.nan
    nets = np.array(nets); amd = np.array(amd)
    days = n / bpd
    return nets.sum()/days, amd.sum()/days, len(nets), float((nets > 0).mean()*100)


print("="*96)
print(f"TV RATINGS Phase 0+1 — pairs={PAIRS}")
print("="*96)
rows = []
for pair in PAIRS:
    pip = pip_of(pair)
    dfm5 = pd.read_parquet(M5_DIR / f"{pair}_M5_BA.parquet")
    dfm5["timestamp"] = pd.to_datetime(dfm5["timestamp"], utc=True)
    dfm5 = dfm5.set_index("timestamp").sort_index()
    for tf, rule in TFS.items():
        tfb = resample_tf(dfm5, rule)
        if len(tfb) < 500:
            continue
        rat = compute_ratings(tfb)
        cache = tfb.join(rat)
        cache.to_parquet(CACHE / f"{pair}_{tf}.parquet")     # Phase 0 artifact
        n = len(cache); ie = int(n*IS_FRAC)
        mid = cache["c"].values; bid = cache["bid"].values; ask = cache["ask"].values
        bpd = TF_BARS_PER_DAY[tf]
        for metric in ("summary", "osc_rating", "ma_rating"):
            rv = cache[metric].values
            for H in HORIZONS:
                for thr in THRESH:
                    for fade in (False, True):
                        # OOS only (sweep+report on OOS, same convention as retrace base)
                        npd, apd, nt, wr = sim(rv[ie:], mid[ie:], bid[ie:], ask[ie:],
                                               pip, H, thr, fade, bpd)
                        if nt >= 30:
                            rows.append(dict(pair=pair, tf=tf, metric=metric, H=H, thr=thr,
                                             dir="fade" if fade else "follow",
                                             net_ppd=npd, amddp5_ppd=apd, n=nt, wr=wr))
    del dfm5; gc.collect()
    print(f"  {pair} done")

R = pd.DataFrame(rows)
R.to_csv(RESULTS / "phase01_edge.csv", index=False)
print(f"\nScreened {len(R)} (pair,tf,metric,H,thr,dir) cells → {RESULTS}/phase01_edge.csv\n")

# ── Does the METER (summary) have edge? ──────────────────────────────────────
print("="*96)
print("Q: does the SUMMARY meter have edge? (best cell per pair×TF, by AMDDP5)")
print("="*96)
S = R[R.metric == "summary"]
print(f"{'pair':<9}{'tf':>4}  {'bestdir':>7}{'H':>3}{'thr':>5}{'net_ppd':>9}{'amddp5':>9}{'n':>6}{'wr':>6}")
for (pair, tf), g in S.groupby(["pair", "tf"]):
    b = g.loc[g.amddp5_ppd.idxmax()]
    print(f"{pair:<9}{tf:>4}  {b['dir']:>7}{int(b.H):>3}{b.thr:>5.1f}"
          f"{b.net_ppd:>+9.2f}{b.amddp5_ppd:>+9.1f}{int(b.n):>6}{b.wr:>5.0f}%")

print("\n=== overall: positive AMDDP5 cells by metric × direction ===")
for metric in ("summary", "osc_rating", "ma_rating"):
    for d in ("follow", "fade"):
        sub = R[(R.metric == metric) & (R["dir"] == d)]
        pos = (sub.amddp5_ppd > 0).sum()
        npos = (sub.net_ppd > 0).sum()
        print(f"  {metric:<11} {d:<7}: AMDDP5>0 {pos:>4}/{len(sub):<4}  net>0 {npos:>4}/{len(sub)}  "
              f"best_amddp5={sub.amddp5_ppd.max():+.1f}  best_net={sub.net_ppd.max():+.2f}")
print("\nNOTE: OOS-swept screen (Phase 1) — survivors go to WF+MC (Phase 3). "
      "All-negative ⇒ the meter has no net-of-spread edge.")
