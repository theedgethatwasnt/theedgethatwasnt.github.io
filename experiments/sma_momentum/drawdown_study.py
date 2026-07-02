#!/usr/bin/env python3
"""
Drawdown study — fx-price-mom-live (Config B: M15+M5 lags=(1,3,8) TP=10p)
Computes OOS equity curve by merging all 12 pairs' trades in time order,
then calculates drawdown statistics in pips and dollars at 50u.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"

PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY   = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC = 0.70
LAGS    = (1, 3, 8)
TP      = 10.0
UNITS   = 50
USDJPY  = 155.0
GBPUSD  = 1.27

SP_GATES = {  # IS P90, hardcoded from MC validation
    "GBP_JPY": 4.00, "CAD_JPY": 2.60, "EUR_JPY": 2.50, "AUD_JPY": 2.30,
    "USD_JPY": 2.10, "NZD_JPY": 3.10, "CHF_JPY": 3.70, "NZD_USD": 2.00,
    "EUR_USD": 1.70, "AUD_USD": 1.60, "GBP_USD": 2.40, "EUR_GBP": 2.00,
}

def pip_sz(pair): return 0.01 if pair in JPY else 0.0001

def pip_val_usd(pair):
    if pair in JPY:         return 0.01 * UNITS / USDJPY
    if pair == "EUR_GBP":   return 0.0001 * UNITS * GBPUSD
    return 0.0001 * UNITS


def build_signal(df, lags, tf1, tf2):
    moms = []
    for tf in [tf1, tf2]:
        rs  = df["close"].resample(tf).last().dropna()
        rs_s = rs.shift(1).reindex(df.index, method="ffill")
        for k in lags:
            moms.append(rs_s - rs_s.shift(k))
    score = (pd.concat(moms, axis=1) > 0).sum(axis=1)
    n_ind = len(moms)
    sig   = pd.Series(np.int8(0), index=df.index)
    sig[score == n_ind] = np.int8(1)
    sig[score == 0]     = np.int8(-1)
    return sig


def simulate_with_timestamps(df, sig, pip, sp_gate, tp_pips):
    """Returns list of (exit_timestamp, pnl_pips) for OOS trades."""
    bid = df["bid_c"].values.astype(np.float64)
    ask = df["ask_c"].values.astype(np.float64)
    mid = df["close"].values.astype(np.float64)
    sp  = (ask - bid) / pip
    s   = sig.values
    ts  = df.index
    n   = len(df)
    trades = []
    in_trade = False; dir_ = 0; ep = 0.0
    for i in range(1, n):
        if in_trade:
            if (mid[i] - ep) / pip * dir_ >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl = (exit_px - ep) / pip * dir_ - sp[i]
                trades.append((ts[i], pnl))
                in_trade = False
        else:
            nd = s[i - 1]
            if nd != 0 and sp[i] <= sp_gate:
                ep = ask[i] if nd == 1 else bid[i]
                dir_ = nd; in_trade = True
    return trades


print("Loading data and running OOS simulations …")
all_trades = []  # (timestamp, pnl_pips, pnl_usd, pair)

for pair in PAIRS:
    df = pd.read_parquet(DATA / f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    pip  = pip_sz(pair)
    pv   = pip_val_usd(pair)
    n_is = int(len(df) * IS_FRAC)
    sg   = SP_GATES[pair]

    sig = build_signal(df, LAGS, "15min", "5min")
    oos_df  = df.iloc[n_is:]
    oos_sig = sig.iloc[n_is:]

    trades = simulate_with_timestamps(oos_df, oos_sig, pip, sg, TP)
    for ts, pnl_p in trades:
        all_trades.append((ts, pnl_p, pnl_p * pv, pair))
    print(f"  {pair}: {len(trades)} trades")

# Sort by exit timestamp → portfolio equity curve
all_trades.sort(key=lambda x: x[0])
print(f"\nTotal OOS trades: {len(all_trades)}")

pnl_pips = np.array([t[1] for t in all_trades])
pnl_usd  = np.array([t[2] for t in all_trades])
pairs_seq = [t[3] for t in all_trades]

# ── Equity curves ─────────────────────────────────────────────────────────────
eq_pips = np.cumsum(pnl_pips)
eq_usd  = np.cumsum(pnl_usd)

def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd   = equity - peak
    return dd.min(), dd

def drawdown_series(equity):
    """Returns array of drawdown depths at each trough."""
    peak = np.maximum.accumulate(equity)
    dd   = equity - peak
    # find individual drawdown episodes
    dds = []
    in_dd = False
    cur_max = 0.0
    for v in dd:
        if v < 0:
            in_dd = True
            if v < cur_max: cur_max = v
        elif in_dd:
            dds.append(abs(cur_max))
            cur_max = 0.0; in_dd = False
    if in_dd: dds.append(abs(cur_max))
    return np.array(dds)

mdd_pips, dd_series_pips = max_drawdown(eq_pips)
mdd_usd,  dd_series_usd  = max_drawdown(eq_usd)

dd_episodes_pips = drawdown_series(eq_pips)
dd_episodes_usd  = drawdown_series(eq_usd)

# ── Consecutive losses ────────────────────────────────────────────────────────
losses = (pnl_pips < 0).astype(int)
max_consec = cur_consec = 0
for l in losses:
    cur_consec = cur_consec + 1 if l else 0
    max_consec = max(max_consec, cur_consec)

# ── OOS period ────────────────────────────────────────────────────────────────
oos_days = len(all_trades) / (sum(1 for _ in all_trades if _[3] == "USD_JPY") /
    (np.array([t[0] for t in all_trades if t[3]=="USD_JPY"])[-1] -
     np.array([t[0] for t in all_trades if t[3]=="USD_JPY"])[0]).days) \
    if False else None  # skip this calc

# Simple: use total pips / ppd
ppd = pnl_pips.sum() / (len(pnl_pips) / (len(pnl_pips) / 30.4))  # ≈30.4 p/d
n_oos_days = pnl_pips.sum() / 30.4  # rough

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"DRAWDOWN ANALYSIS — fx-price-mom-live (M15+M5, lags=(1,3,8), TP=10p, 50u)")
print(f"{'='*62}")
print(f"\n── Trade P&L distribution ──────────────────────────────────")
print(f"  Total OOS trades   : {len(pnl_pips)}")
print(f"  Winners            : {(pnl_pips>0).sum()}  ({(pnl_pips>0).mean()*100:.1f}%)")
print(f"  Losers             : {(pnl_pips<0).sum()}  ({(pnl_pips<0).mean()*100:.1f}%)")
print(f"  Avg win pips       : {pnl_pips[pnl_pips>0].mean():.1f}p")
print(f"  Avg loss pips      : {pnl_pips[pnl_pips<0].mean():.1f}p")
print(f"  Win/loss ratio     : {pnl_pips[pnl_pips>0].mean() / abs(pnl_pips[pnl_pips<0].mean()):.1f}×")
print(f"  Max consec losses  : {max_consec}")

print(f"\n── Drawdown — PIPS ─────────────────────────────────────────")
print(f"  Max drawdown (OOS) : {abs(mdd_pips):.1f}p")
if len(dd_episodes_pips):
    for p in [50,75,90,95,99]:
        print(f"  P{p:2d} episode DD    : {np.percentile(dd_episodes_pips,p):.1f}p")
    print(f"  Avg episode DD     : {dd_episodes_pips.mean():.1f}p")
    print(f"  DD episodes total  : {len(dd_episodes_pips)}")

print(f"\n── Drawdown — USD (50u, approx rates) ──────────────────────")
print(f"  Max drawdown (OOS) : ${abs(mdd_usd):.3f}")
if len(dd_episodes_usd):
    for p in [50,75,90,95,99]:
        print(f"  P{p:2d} episode DD    : ${np.percentile(dd_episodes_usd,p):.3f}")
    print(f"  Avg episode DD     : ${dd_episodes_usd.mean():.3f}")

print(f"\n── Drawdown as % of NAV ($15.97) ───────────────────────────")
nav = 15.97
print(f"  Max DD % of NAV    : {abs(mdd_usd)/nav*100:.2f}%  (${abs(mdd_usd):.3f})")
if len(dd_episodes_usd):
    print(f"  P90 episode DD %   : {np.percentile(dd_episodes_usd,90)/nav*100:.2f}%  (${np.percentile(dd_episodes_usd,90):.3f})")
    print(f"  P99 episode DD %   : {np.percentile(dd_episodes_usd,99)/nav*100:.2f}%  (${np.percentile(dd_episodes_usd,99):.3f})")

print(f"\n── Recovery ────────────────────────────────────────────────")
print(f"  Total OOS pips     : {pnl_pips.sum():+.0f}p")
print(f"  Total OOS USD      : ${pnl_usd.sum():+.3f}")
print(f"  Rough OOS days     : {pnl_pips.sum()/30.4:.0f}d")
print(f"  DD/daily-rate      : {abs(mdd_pips)/30.4:.1f} days to recover max DD")
print()
