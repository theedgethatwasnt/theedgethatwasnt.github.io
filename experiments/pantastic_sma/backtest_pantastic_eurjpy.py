#!/usr/bin/env python3
"""
PantasticSMA backtest on EUR_JPY 30-tick bars built from S5 BA data.

Logic (port of PantasticSMABot.cs):
  tick bar  = aggregate S5 bars until cumsum(volume) >= TICK_SIZE
  SMA(close, sma_p) on tick bars
  rate      = (sma[i] - sma[i - lookback]) / lookback   (price units / bar)
  rate >  threshold → Long  (close short if open, open long)
  rate < -threshold → Short (close long  if open, open short)
  else              → Flat  (close all)

Spread cost: half-spread paid on every open and close (mid ± spread/2 = ask/bid).
P&L in pips  (JPY pairs: 1 pip = 0.01).

IS / OOS split: first 70% of tick bars = IS, last 30% = OOS.
Also sweeps tick_size and threshold to find the best config.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

S5_PATH = ROOT / "data" / "s5_ohlc" / "EUR_JPY_S5_BA.parquet"
PIP     = 0.01   # 1 pip for JPY pairs

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading EUR_JPY S5 BA data…")
df = pd.read_parquet(S5_PATH).sort_values("timestamp").reset_index(drop=True)
print(f"  {len(df):,} S5 bars  "
      f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")

mid_o  = ((df["bid_o"] + df["ask_o"]) / 2).values.astype(np.float64)
mid_h  = ((df["bid_h"] + df["ask_h"]) / 2).values.astype(np.float64)
mid_l  = ((df["bid_l"] + df["ask_l"]) / 2).values.astype(np.float64)
mid_c  = ((df["bid_c"] + df["ask_c"]) / 2).values.astype(np.float64)
spread = (df["ask_c"] - df["bid_c"]).values.astype(np.float64)
volume = df["volume"].values.astype(np.int32)
ts_ns  = df["timestamp"].values.astype(np.int64)

# ── Tick bar builder ──────────────────────────────────────────────────────────
@njit
def build_tick_bars(mid_o, mid_h, mid_l, mid_c, spread, volume, ts_ns, tick_size):
    n = len(mid_c)
    tb_o  = np.empty(n, np.float64)
    tb_h  = np.empty(n, np.float64)
    tb_l  = np.empty(n, np.float64)
    tb_c  = np.empty(n, np.float64)
    tb_sp = np.empty(n, np.float64)
    tb_ts = np.empty(n, np.int64)
    nb = 0

    cum_vol = 0
    b_o = b_h = b_l = b_c = b_sp = 0.0
    b_ts = np.int64(0)
    in_bar = False

    for i in range(n):
        if not in_bar:
            b_o  = mid_o[i]; b_h = mid_h[i]; b_l = mid_l[i]
            b_ts = ts_ns[i]; in_bar = True
        else:
            if mid_h[i] > b_h: b_h = mid_h[i]
            if mid_l[i] < b_l: b_l = mid_l[i]
        b_c   = mid_c[i]
        b_sp  = spread[i]
        cum_vol += volume[i]

        if cum_vol >= tick_size:
            tb_o[nb] = b_o; tb_h[nb] = b_h; tb_l[nb] = b_l
            tb_c[nb] = b_c; tb_sp[nb] = b_sp; tb_ts[nb] = b_ts
            nb += 1
            cum_vol = 0; in_bar = False

    return tb_o[:nb], tb_h[:nb], tb_l[:nb], tb_c[:nb], tb_sp[:nb], tb_ts[:nb]

# ── Strategy ──────────────────────────────────────────────────────────────────
@njit
def run_strategy(tb_c, tb_sp, sma_p, lookback, threshold, pip, start_i, end_i):
    """Run on tb_c[start_i:end_i]. Returns (trades_pips, equity_curve, n_trades)."""
    n       = end_i - start_i
    warmup  = sma_p + lookback + 1

    sma_buf    = np.zeros(sma_p, np.float64)
    sma_sum    = 0.0
    sma_filled = 0
    sma_vals   = np.full(n, np.nan)

    trades_pips = np.empty(n, np.float64)
    n_trades    = 0
    equity      = np.zeros(n, np.float64)
    cum_pnl     = 0.0
    pos         = 0
    entry_price = 0.0

    for ii in range(n):
        i = start_i + ii
        # rolling SMA
        oldest   = sma_buf[ii % sma_p]
        sma_sum  = sma_sum - oldest + tb_c[i]
        sma_buf[ii % sma_p] = tb_c[i]
        if sma_filled < sma_p:
            sma_filled += 1
        if sma_filled >= sma_p:
            sma_vals[ii] = sma_sum / sma_p

        equity[ii] = cum_pnl

        if ii < warmup:
            continue
        prev_ii = ii - lookback
        if np.isnan(sma_vals[ii]) or np.isnan(sma_vals[prev_ii]):
            continue

        rate = (sma_vals[ii] - sma_vals[prev_ii]) / lookback
        # counter-trend: fade SMA momentum (mean-reversion)
        new_sig = -1 if rate > threshold else (1 if rate < -threshold else 0)

        if new_sig == pos:
            continue

        sp      = tb_sp[i]
        half_sp = sp / 2.0

        # close existing
        if pos != 0:
            ep = tb_c[i] - half_sp if pos == 1 else tb_c[i] + half_sp
            pnl = (ep - entry_price) / pip if pos == 1 else (entry_price - ep) / pip
            cum_pnl += pnl
            trades_pips[n_trades] = pnl
            n_trades += 1

        # open new
        if new_sig == 1:
            entry_price = tb_c[i] + half_sp
        elif new_sig == -1:
            entry_price = tb_c[i] - half_sp
        pos = new_sig
        equity[ii] = cum_pnl

    # close final
    if pos != 0:
        i  = end_i - 1
        ii = n - 1
        sp = tb_sp[i]; half_sp = sp / 2.0
        ep = tb_c[i] - half_sp if pos == 1 else tb_c[i] + half_sp
        pnl = (ep - entry_price) / pip if pos == 1 else (entry_price - ep) / pip
        cum_pnl += pnl
        trades_pips[n_trades] = pnl
        n_trades += 1
        equity[ii] = cum_pnl

    return trades_pips[:n_trades], equity, cum_pnl

# ── Stats helper ──────────────────────────────────────────────────────────────
def stats(trades, equity, tb_ts, start_i, end_i, label):
    ts0   = pd.Timestamp(tb_ts[start_i], unit='ns', tz='UTC')
    ts1   = pd.Timestamp(tb_ts[end_i - 1], unit='ns', tz='UTC')
    cal_d = (ts1 - ts0).total_seconds() / 86400
    t_d   = cal_d * 5 / 7

    n  = len(trades)
    if n == 0:
        print(f"  [{label}] No trades.")
        return {}

    wins   = trades[trades > 0]
    losses = trades[trades <= 0]
    wr     = len(wins) / n * 100
    aw     = wins.mean()  if len(wins)   else 0.0
    al     = losses.mean() if len(losses) else 0.0
    pf     = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf
    total  = trades.sum()
    ppd    = total / t_d if t_d else 0.0
    peak   = np.maximum.accumulate(equity)
    max_dd = (equity - peak).min()

    return dict(label=label, dates=f"{ts0.date()}→{ts1.date()}",
                n=n, trades_per_day=n/t_d, wr=wr, avg_win=aw, avg_loss=al,
                pf=pf, total=total, ppd=ppd, max_dd=max_dd)

def print_stats(s):
    if not s: return
    print(f"\n  [{s['label']}] {s['dates']}")
    print(f"    Trades  : {s['n']:,}  ({s['trades_per_day']:.0f}/day)")
    print(f"    Win rate: {s['wr']:.1f}%   Avg W: +{s['avg_win']:.2f}p  Avg L: {s['avg_loss']:.2f}p")
    print(f"    P-Factor: {s['pf']:.2f}   Total: {s['total']:+.1f}p   p/day: {s['ppd']:+.2f}p")
    print(f"    Max DD  : {s['max_dd']:.1f}p")

# ── Sweep ─────────────────────────────────────────────────────────────────────
TICK_SIZES  = [15, 30, 50, 100]
THRESHOLDS  = [0.001, 0.003, 0.005, 0.010]
SMA_P       = 7
LOOKBACK    = 5

print("\nJIT warm-up…")

results = []

for tick_size in TICK_SIZES:
    print(f"\n{'═'*60}")
    print(f"  tick_size={tick_size}  building bars…")
    tb_o, tb_h, tb_l, tb_c, tb_sp, tb_ts = build_tick_bars(
        mid_o, mid_h, mid_l, mid_c, spread, volume, ts_ns, tick_size)
    N = len(tb_c)
    split = int(N * 0.70)
    print(f"  {N:,} tick bars  IS={split:,}  OOS={N-split:,}")

    for thr in THRESHOLDS:
        tag = f"t{tick_size}_thr{thr}"

        tr_is,  eq_is,  _ = run_strategy(tb_c, tb_sp, SMA_P, LOOKBACK, thr, PIP, 0, split)
        tr_oos, eq_oos, _ = run_strategy(tb_c, tb_sp, SMA_P, LOOKBACK, thr, PIP, split, N)

        s_is  = stats(tr_is,  eq_is,  tb_ts, 0,     split, f"IS  {tag}")
        s_oos = stats(tr_oos, eq_oos, tb_ts, split, N,     f"OOS {tag}")

        print_stats(s_is)
        print_stats(s_oos)

        if s_oos:
            results.append(dict(tick_size=tick_size, threshold=thr,
                                is_ppd=s_is['ppd'],  oos_ppd=s_oos['ppd'],
                                is_pf=s_is['pf'],    oos_pf=s_oos['pf'],
                                is_wr=s_is['wr'],    oos_wr=s_oos['wr'],
                                oos_dd=s_oos['max_dd']))

# ── Summary table ─────────────────────────────────────────────────────────────
if results:
    res = pd.DataFrame(results).sort_values("oos_ppd", ascending=False)
    print(f"\n{'═'*60}")
    print("  SWEEP SUMMARY — ranked by OOS p/day")
    print(f"{'═'*60}")
    print(res.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
