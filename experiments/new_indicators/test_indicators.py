#!/usr/bin/env python3
"""
Honest indicator screen — IC + spread-deducted bounded-hold edge, multi-timeframe.
==================================================================================
Tests TRIX, Vortex, Fisher, RSI2, Kaufman-ER across M15/H1/H4/D1 on 12 pairs.

The gate that matters (lessons from the momentum-book blowup):
  - NOT a TP-only/no-SL p/d (that just defers losses).
  - Bounded fixed-hold exit: enter at signal bar close (pay ask/bid), exit H bars
    later at close (bid/ask). Spread is therefore deducted by construction.
  - Headline metric = mean P&L per trade in pips (net of spread) + its t-stat,
    on OOS only. A signal "has edge" iff avg_pnl_pips > 0 with |t|>2 across pairs.
  - IC = corr(sign(signal), forward mid return) reported alongside for context.
  - Both TREND (sign) and CONTRARIAN (-sign) framings tested — the IC study said
    every high-IC indicator on this book is contrarian, so we check both.

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
TFS  = {"M15":"15min","H1":"1h","H4":"4h","D1":"1D"}
HOLD = 1   # hold H native TF bars then exit (bounded)
BARS_PER_DAY = {"M15":96,"H1":24,"H4":6,"D1":1}

def pip_sz(p): return 0.01 if p in JPY else 0.0001


# ── Indicators (causal) ─────────────────────────────────────────────────────────
def ind_trix(o,h,l,c, n=14):
    e = c.ewm(span=n, adjust=False).mean()
    e = e.ewm(span=n, adjust=False).mean()
    e = e.ewm(span=n, adjust=False).mean()
    trix = e.pct_change() * 1e4
    return trix  # >0 up-momentum

def ind_vortex(o,h,l,c, n=14):
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    vmp = (h - l.shift()).abs(); vmm = (l - h.shift()).abs()
    vip = vmp.rolling(n).sum() / tr.rolling(n).sum()
    vim = vmm.rolling(n).sum() / tr.rolling(n).sum()
    return vip - vim  # >0 up-trend

def ind_fisher(o,h,l,c, n=10):
    med = (h + l) / 2
    mn = med.rolling(n).min(); mx = med.rolling(n).max()
    rng = (mx - mn).replace(0, np.nan)
    x = (2 * (med - mn) / rng - 1).clip(-0.999, 0.999)
    x = x.ewm(alpha=0.33, adjust=False).mean().clip(-0.999, 0.999)
    fish = 0.5 * np.log((1 + x) / (1 - x))
    return fish.ewm(alpha=0.5, adjust=False).mean()

def ind_rsi2(o,h,l,c, n=2):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)   # 0..100; <10 oversold, >90 overbought

def ind_ker(o,h,l,c, n=10):
    chg = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n).sum()
    return chg / vol.replace(0, np.nan)  # 0..1 trend efficiency


# Each indicator → a ±1/0 signal. (name, fn, signal_rule, default_framing_note)
def sig_trix(v):   return np.sign(v).fillna(0)
def sig_vortex(v): return np.sign(v).fillna(0)
def sig_fisher(v): return np.sign(v).fillna(0)
def sig_rsi2(v):
    s = pd.Series(0.0, index=v.index); s[v < 10] = 1.0; s[v > 90] = -1.0
    return s  # raw rule is contrarian-long-on-oversold; we test as-is + flip

INDICATORS = [
    ("TRIX14",  ind_trix,   sig_trix),
    ("Vortex14",ind_vortex, sig_vortex),
    ("Fisher10",ind_fisher, sig_fisher),
    ("RSI2",    ind_rsi2,   sig_rsi2),
]


def eval_signal(rs, sig, pip, n_is, hold):
    """Returns dict of IS/OOS IC + spread-deducted avg pnl/t-stat for a ±1 signal."""
    c   = rs["close"].values
    bid = rs["bid_c"].values; ask = rs["ask_c"].values
    s   = sig.values.astype(float)
    n   = len(c)
    # forward mid return (pips) for IC
    fwd = np.full(n, np.nan); fwd[:-hold] = (c[hold:] - c[:-hold]) / pip
    # bounded-hold pnl in pips, spread-deducted via bid/ask fills
    pnl = np.full(n, np.nan)
    bid_f = np.full(n, np.nan); ask_f = np.full(n, np.nan)
    bid_f[:-hold] = bid[hold:]; ask_f[:-hold] = ask[hold:]
    long  = s == 1; short = s == -1
    pnl[long]  = (bid_f[long]  - ask[long])  / pip
    pnl[short] = (bid[short]   - ask_f[short]) / pip
    def stats(lo, hi):
        m = (s != 0) & ~np.isnan(fwd); m[:lo] = False; m[hi:] = False
        if m.sum() < 30: return (np.nan, np.nan, np.nan, 0)
        ic = np.corrcoef(s[m], fwd[m])[0,1] if s[m].std()>0 else np.nan
        pm = (s != 0) & ~np.isnan(pnl); pm[:lo] = False; pm[hi:] = False
        pv = pnl[pm]
        avg = pv.mean(); t = avg / (pv.std(ddof=1)/np.sqrt(len(pv))) if pv.std()>0 else np.nan
        return (ic, avg, t, len(pv))
    ic_is, avg_is, t_is, n_tr_is = stats(0, n_is)
    ic_oo, avg_oo, t_oo, n_tr_oo = stats(n_is, n)
    return dict(ic_is=ic_is, ic_oos=ic_oo, avg_is=avg_is, avg_oos=avg_oo,
                t_oos=t_oo, n_oos=n_tr_oo)


rows = []
for pair in PAIRS:
    pip = pip_sz(pair)
    df = pd.read_parquet(DATA / f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    for tf_name, tf in TFS.items():
        rs = df.resample(tf).agg({"open":"first","high":"max","low":"min","close":"last",
                                   "bid_c":"last","ask_c":"last","volume":"sum"}).dropna()
        if len(rs) < 500: continue
        n_is = int(len(rs) * IS_FRAC)
        for iname, ifn, sfn in INDICATORS:
            v = ifn(rs["open"], rs["high"], rs["low"], rs["close"])
            base = sfn(v)
            for framing, sgn in (("trend", 1), ("contra", -1)):
                sig = (base * sgn).fillna(0)
                r = eval_signal(rs, sig, pip, n_is, HOLD)
                rows.append(dict(pair=pair, tf=tf_name, ind=iname, framing=framing, **r))

res = pd.DataFrame(rows)
res.to_csv(RESULTS / "indicator_screen.csv", index=False)

# ── Aggregate across pairs per (tf, ind, framing) ────────────────────────────────
print("="*96)
print("INDICATOR SCREEN — aggregate across 12 pairs.  Gate: OOS avg pnl/trade > 0 (spread-deducted)")
print("="*96)
print(f"{'TF':<4} {'indicator':<9} {'framing':<7} {'meanIC_oos':>10} {'mean_avgPnl_oos':>15} "
      f"{'pairs_pnl>0':>11} {'pairs_t>2':>9} {'verdict':>9}")
agg = []
for (tf, ind, fr), g in res.groupby(["tf","ind","framing"]):
    ic = g["ic_oos"].mean()
    avg = g["avg_oos"].mean()
    p_pos = int((g["avg_oos"] > 0).sum())
    p_sig = int(((g["avg_oos"] > 0) & (g["t_oos"] > 2)).sum())
    verdict = "EDGE" if (p_pos >= 9 and p_sig >= 6 and avg > 0) else ("look" if p_pos>=8 else "")
    agg.append(dict(tf=tf, ind=ind, framing=fr, meanIC=ic, mean_avgPnl=avg,
                    pairs_pos=p_pos, pairs_sig=p_sig, verdict=verdict))
agg = pd.DataFrame(agg).sort_values("mean_avgPnl", ascending=False)
for _, r in agg.iterrows():
    print(f"{r['tf']:<4} {r['ind']:<9} {r['framing']:<7} {r['meanIC']:>10.4f} "
          f"{r['mean_avgPnl']:>15.3f} {r['pairs_pos']:>11d} {r['pairs_sig']:>9d} {r['verdict']:>9}")
agg.to_csv(RESULTS / "indicator_screen_agg.csv", index=False)
print("\nTop by mean OOS avg-pnl/trade (pips, spread-deducted):")
print(agg.head(12).to_string(index=False))
print(f"\nSaved → results/indicator_screen.csv + indicator_screen_agg.csv")
