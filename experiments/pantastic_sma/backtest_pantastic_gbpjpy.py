#!/usr/bin/env python3
"""
PantasticSMA backtest on GBP_JPY 30-tick bars from S5 BA data.
Counter-trend variant: fade SMA momentum (mean-reversion).

GBP_JPY spread is wide (~3.6p median) so thresholds are scaled up vs EUR_JPY.
Only 30 days of data — no IS/OOS split; runs full period + reports results.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

S5_PATH = ROOT / "data" / "s5_ohlc" / "GBP_JPY_S5_BA.parquet"
PIP     = 0.01   # JPY pair

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading GBP_JPY S5 BA data…")
df = pd.read_parquet(S5_PATH).sort_values("timestamp").reset_index(drop=True)
print(f"  {len(df):,} S5 bars  "
      f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")

sp_pips = (df["ask_c"] - df["bid_c"]) / PIP
print(f"  Spread: min={sp_pips.min():.2f}p  med={sp_pips.median():.2f}p  "
      f"p90={sp_pips.quantile(0.9):.2f}p  max={sp_pips.max():.2f}p\n")

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
            b_o = mid_o[i]; b_h = mid_h[i]; b_l = mid_l[i]
            b_ts = ts_ns[i]; in_bar = True
        else:
            if mid_h[i] > b_h: b_h = mid_h[i]
            if mid_l[i] < b_l: b_l = mid_l[i]
        b_c  = mid_c[i]
        b_sp = spread[i]
        cum_vol += volume[i]

        if cum_vol >= tick_size:
            tb_o[nb] = b_o; tb_h[nb] = b_h; tb_l[nb] = b_l
            tb_c[nb] = b_c; tb_sp[nb] = b_sp; tb_ts[nb] = b_ts
            nb += 1
            cum_vol = 0; in_bar = False

    return tb_o[:nb], tb_h[:nb], tb_l[:nb], tb_c[:nb], tb_sp[:nb], tb_ts[:nb]

# ── Strategy (counter-trend: fade momentum) ───────────────────────────────────
@njit
def run_strategy(tb_c, tb_sp, sma_p, lookback, threshold, pip):
    n      = len(tb_c)
    warmup = sma_p + lookback + 1

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

    for i in range(n):
        oldest  = sma_buf[i % sma_p]
        sma_sum = sma_sum - oldest + tb_c[i]
        sma_buf[i % sma_p] = tb_c[i]
        if sma_filled < sma_p:
            sma_filled += 1
        if sma_filled >= sma_p:
            sma_vals[i] = sma_sum / sma_p

        equity[i] = cum_pnl

        if i < warmup:
            continue
        prev_i = i - lookback
        if np.isnan(sma_vals[i]) or np.isnan(sma_vals[prev_i]):
            continue

        rate = (sma_vals[i] - sma_vals[prev_i]) / lookback
        # counter-trend: fade the momentum
        new_sig = -1 if rate > threshold else (1 if rate < -threshold else 0)

        if new_sig == pos:
            continue

        sp      = tb_sp[i]
        half_sp = sp / 2.0

        if pos != 0:
            ep  = tb_c[i] - half_sp if pos == 1 else tb_c[i] + half_sp
            pnl = (ep - entry_price) / pip if pos == 1 else (entry_price - ep) / pip
            cum_pnl += pnl
            trades_pips[n_trades] = pnl
            n_trades += 1

        if new_sig == 1:
            entry_price = tb_c[i] + half_sp
        elif new_sig == -1:
            entry_price = tb_c[i] - half_sp
        pos = new_sig
        equity[i] = cum_pnl

    if pos != 0:
        sp = tb_sp[n - 1]; half_sp = sp / 2.0
        ep = tb_c[n-1] - half_sp if pos == 1 else tb_c[n-1] + half_sp
        pnl = (ep - entry_price) / pip if pos == 1 else (entry_price - ep) / pip
        cum_pnl += pnl
        trades_pips[n_trades] = pnl
        n_trades += 1
        equity[n-1] = cum_pnl

    return trades_pips[:n_trades], equity, cum_pnl

# ── Stats ─────────────────────────────────────────────────────────────────────
def stats(trades, equity, tb_ts, label):
    ts0   = pd.Timestamp(tb_ts[0],  unit='ns', tz='UTC')
    ts1   = pd.Timestamp(tb_ts[-1], unit='ns', tz='UTC')
    cal_d = (ts1 - ts0).total_seconds() / 86400
    t_d   = cal_d * 5 / 7

    n = len(trades)
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

    return dict(label=label, n=n, trades_per_day=n/t_d,
                wr=wr, avg_win=aw, avg_loss=al, pf=pf,
                total=total, ppd=ppd, max_dd=max_dd)

def print_stats(s):
    if not s: return
    print(f"  [{s['label']}]")
    print(f"    Trades  : {s['n']:,}  ({s['trades_per_day']:.0f}/day)")
    print(f"    Win rate: {s['wr']:.1f}%   Avg W: +{s['avg_win']:.2f}p  Avg L: {s['avg_loss']:.2f}p")
    print(f"    P-Factor: {s['pf']:.2f}   Total: {s['total']:+.1f}p   p/day: {s['ppd']:+.2f}p")
    print(f"    Max DD  : {s['max_dd']:.1f}p")

# ── Sweep ─────────────────────────────────────────────────────────────────────
# GBP_JPY spread ~3.6p median — need meaningful thresholds (price units)
# 1 pip = 0.01, so 0.05 = 5 pips/bar, 0.10 = 10 pips/bar
TICK_SIZES  = [15, 30, 50, 100]
THRESHOLDS  = [0.010, 0.020, 0.050, 0.100]  # 1p, 2p, 5p, 10p per bar
SMA_P       = 7
LOOKBACK    = 5

print("JIT warm-up…\n")

results = []

for tick_size in TICK_SIZES:
    print(f"{'═'*60}")
    print(f"  tick_size={tick_size}  building bars…")
    tb_o, tb_h, tb_l, tb_c, tb_sp, tb_ts = build_tick_bars(
        mid_o, mid_h, mid_l, mid_c, spread, volume, ts_ns, tick_size)
    N = len(tb_c)
    print(f"  {N:,} tick bars")

    for thr in THRESHOLDS:
        tag = f"t{tick_size}_thr{thr:.3f}"
        trades, equity, _ = run_strategy(tb_c, tb_sp, SMA_P, LOOKBACK, thr, PIP)
        s = stats(trades, equity, tb_ts, tag)
        print_stats(s)
        if s:
            results.append(dict(tick_size=tick_size, threshold=thr,
                                ppd=s['ppd'], pf=s['pf'], wr=s['wr'],
                                trades_per_day=s['trades_per_day'],
                                avg_win=s['avg_win'], avg_loss=s['avg_loss'],
                                max_dd=s['max_dd'], total=s['total']))

# ── Summary ───────────────────────────────────────────────────────────────────
if results:
    res = pd.DataFrame(results).sort_values("ppd", ascending=False)
    print(f"\n{'═'*60}")
    print("  SWEEP SUMMARY — ranked by p/day")
    print(f"{'═'*60}")
    print(res.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
