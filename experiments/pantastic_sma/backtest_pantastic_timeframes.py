#!/usr/bin/env python3
"""
PantasticSMA — multi-timeframe backtest on EUR_JPY.

WhiteMethod (price/min, timeframe-neutral):
  rate = (sma[i] - sma[i-lookback]) / (lookback * bar_minutes)
  threshold = 0.003 price/min  (original cTrader default)

Timeframes tested: 30-tick bars (from S5), M5, H1
Both directions: trend-follow (original) and counter-trend (fade)
IS/OOS split on M5/H1 (5.5yr data).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

PIP      = 0.01   # EUR_JPY JPY pair
SMA_P    = 7
LOOKBACK = 5

# ── Helpers ───────────────────────────────────────────────────────────────────
@njit
def build_tick_bars(mid_o, mid_h, mid_l, mid_c, spread, volume, ts_ns, tick_size):
    n = len(mid_c)
    tb_o  = np.empty(n, np.float64); tb_h = np.empty(n, np.float64)
    tb_l  = np.empty(n, np.float64); tb_c = np.empty(n, np.float64)
    tb_sp = np.empty(n, np.float64); tb_ts = np.empty(n, np.int64)
    nb = 0
    cum_vol = 0
    b_o = b_h = b_l = b_c = b_sp = 0.0; b_ts = np.int64(0); in_bar = False
    for i in range(n):
        if not in_bar:
            b_o = mid_o[i]; b_h = mid_h[i]; b_l = mid_l[i]; b_ts = ts_ns[i]; in_bar = True
        else:
            if mid_h[i] > b_h: b_h = mid_h[i]
            if mid_l[i] < b_l: b_l = mid_l[i]
        b_c = mid_c[i]; b_sp = spread[i]; cum_vol += volume[i]
        if cum_vol >= tick_size:
            tb_o[nb] = b_o; tb_h[nb] = b_h; tb_l[nb] = b_l
            tb_c[nb] = b_c; tb_sp[nb] = b_sp; tb_ts[nb] = b_ts
            nb += 1; cum_vol = 0; in_bar = False
    return tb_o[:nb], tb_h[:nb], tb_l[:nb], tb_c[:nb], tb_sp[:nb], tb_ts[:nb]

@njit
def run_strategy(tb_c, tb_sp, bar_min_arr, sma_p, lookback, threshold, pip, counter_trend, start_i, end_i):
    """
    bar_min_arr: per-bar duration in minutes (scalar encoded as length-1 array for fixed TF,
                 or per-bar array for variable-duration tick bars).
    """
    n      = end_i - start_i
    warmup = sma_p + lookback + 1
    sma_buf = np.zeros(sma_p, np.float64); sma_sum = 0.0; sma_filled = 0
    sma_vals = np.full(n, np.nan)
    trades_p = np.empty(n, np.float64); n_tr = 0
    equity = np.zeros(n, np.float64); cum_pnl = 0.0
    pos = 0; entry_price = 0.0
    variable_dur = len(bar_min_arr) > 1

    for ii in range(n):
        i = start_i + ii
        oldest  = sma_buf[ii % sma_p]
        sma_sum = sma_sum - oldest + tb_c[i]
        sma_buf[ii % sma_p] = tb_c[i]
        if sma_filled < sma_p: sma_filled += 1
        if sma_filled >= sma_p: sma_vals[ii] = sma_sum / sma_p

        equity[ii] = cum_pnl
        if ii < warmup: continue
        prev_ii = ii - lookback
        if np.isnan(sma_vals[ii]) or np.isnan(sma_vals[prev_ii]): continue

        bm = bar_min_arr[i] if variable_dur else bar_min_arr[0]
        rate = (sma_vals[ii] - sma_vals[prev_ii]) / (lookback * bm)

        if counter_trend:
            new_sig = -1 if rate > threshold else (1 if rate < -threshold else 0)
        else:
            new_sig = 1 if rate > threshold else (-1 if rate < -threshold else 0)

        if new_sig == pos: continue
        sp = tb_sp[i]; half_sp = sp / 2.0

        if pos != 0:
            ep  = tb_c[i] - half_sp if pos == 1 else tb_c[i] + half_sp
            pnl = (ep - entry_price) / pip if pos == 1 else (entry_price - ep) / pip
            cum_pnl += pnl; trades_p[n_tr] = pnl; n_tr += 1

        if new_sig == 1:  entry_price = tb_c[i] + half_sp
        elif new_sig == -1: entry_price = tb_c[i] - half_sp
        pos = new_sig; equity[ii] = cum_pnl

    if pos != 0:
        sp = tb_sp[end_i - 1]; half_sp = sp / 2.0
        ep = tb_c[end_i-1] - half_sp if pos == 1 else tb_c[end_i-1] + half_sp
        pnl = (ep - entry_price) / pip if pos == 1 else (entry_price - ep) / pip
        cum_pnl += pnl; trades_p[n_tr] = pnl; n_tr += 1; equity[n-1] = cum_pnl

    return trades_p[:n_tr], equity, cum_pnl

def calc_stats(trades, equity, tb_ts, start_i, end_i):
    ts0   = pd.Timestamp(tb_ts[start_i],  unit='ns', tz='UTC')
    ts1   = pd.Timestamp(tb_ts[end_i - 1], unit='ns', tz='UTC')
    cal_d = (ts1 - ts0).total_seconds() / 86400
    t_d   = cal_d * 5 / 7
    n = len(trades)
    if n == 0: return None
    wins   = trades[trades > 0]; losses = trades[trades <= 0]
    wr     = len(wins) / n * 100
    pf     = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf
    ppd    = trades.sum() / t_d if t_d else 0.0
    peak   = np.maximum.accumulate(equity)
    max_dd = (equity - peak).min()
    return dict(n=n, tpd=n/t_d, wr=wr, aw=wins.mean() if len(wins) else 0,
                al=losses.mean() if len(losses) else 0, pf=pf,
                total=trades.sum(), ppd=ppd, max_dd=max_dd,
                dates=f"{ts0.date()}→{ts1.date()}")

def show(label, s):
    if not s: return
    print(f"  {label:30s}  n={s['n']:5d}({s['tpd']:4.0f}/d)  "
          f"WR={s['wr']:4.1f}%  W={s['aw']:+5.2f}p  L={s['al']:+5.2f}p  "
          f"PF={s['pf']:.2f}  p/d={s['ppd']:+7.2f}  DD={s['max_dd']:7.1f}p  {s['dates']}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. TICK BARS from S5 (EUR_JPY)
# ══════════════════════════════════════════════════════════════════════════════
print("━"*80)
print("TICK BARS (EUR_JPY S5 → 30-tick)  WhiteMethod threshold=0.003 price/min")
print("━"*80)

s5 = pd.read_parquet(ROOT / "data" / "s5_ohlc" / "EUR_JPY_S5_BA.parquet").sort_values("timestamp").reset_index(drop=True)
mid_o  = ((s5.bid_o + s5.ask_o) / 2).values.astype(np.float64)
mid_h  = ((s5.bid_h + s5.ask_h) / 2).values.astype(np.float64)
mid_l  = ((s5.bid_l + s5.ask_l) / 2).values.astype(np.float64)
mid_c  = ((s5.bid_c + s5.ask_c) / 2).values.astype(np.float64)
spread = (s5.ask_c - s5.bid_c).values.astype(np.float64)
vol    = s5.volume.values.astype(np.int32)
ts_ns  = s5.timestamp.values.astype(np.int64)

tb_o, tb_h, tb_l, tb_c, tb_sp, tb_ts = build_tick_bars(mid_o, mid_h, mid_l, mid_c, spread, vol, ts_ns, 30)
N = len(tb_c)
print(f"  {N:,} tick bars  {pd.Timestamp(tb_ts[0], unit='ns', tz='UTC').date()} → {pd.Timestamp(tb_ts[-1], unit='ns', tz='UTC').date()}")

# per-bar duration in minutes
dt_ns = np.diff(tb_ts, prepend=tb_ts[0]).astype(np.float64)
dt_ns[0] = dt_ns[1]  # first bar: use second bar's gap
bar_min_arr = np.clip(dt_ns / 1e9 / 60, 0.01, 60.0)  # cap outliers (weekend gaps)

for direction, ct in [("trend", False), ("counter", True)]:
    for thr in [0.003, 0.010, 0.030]:
        tr, eq, _ = run_strategy(tb_c, tb_sp, bar_min_arr, SMA_P, LOOKBACK, thr, PIP, ct, 0, N)
        s = calc_stats(tr, eq, tb_ts, 0, N)
        show(f"{direction:7s} thr={thr:.3f}", s)

# ══════════════════════════════════════════════════════════════════════════════
# 2. M5 (EUR_JPY, 5.5yr)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "━"*80)
print("M5 (EUR_JPY, 5.5yr)  WhiteMethod threshold=0.003 price/min  bar_min=5")
print("━"*80)

m5 = pd.read_parquet(ROOT / "data" / "m5_ba" / "EUR_JPY_M5_BA.parquet").sort_values("timestamp").reset_index(drop=True)
mc  = ((m5.bid_c + m5.ask_c) / 2).values.astype(np.float64)
msp = (m5.ask_c - m5.bid_c).values.astype(np.float64)
mts = m5.timestamp.values.astype(np.int64)
bar_min_m5 = np.array([5.0])  # fixed
NM = len(mc)
split_m5 = int(NM * 0.70)

for direction, ct in [("trend", False), ("counter", True)]:
    for thr in [0.003, 0.010, 0.030]:
        tr_is,  eq_is,  _ = run_strategy(mc, msp, bar_min_m5, SMA_P, LOOKBACK, thr, PIP, ct, 0, split_m5)
        tr_oos, eq_oos, _ = run_strategy(mc, msp, bar_min_m5, SMA_P, LOOKBACK, thr, PIP, ct, split_m5, NM)
        si = calc_stats(tr_is,  eq_is,  mts, 0, split_m5)
        so = calc_stats(tr_oos, eq_oos, mts, split_m5, NM)
        show(f"IS  {direction:7s} thr={thr:.3f}", si)
        show(f"OOS {direction:7s} thr={thr:.3f}", so)
        print()

# ══════════════════════════════════════════════════════════════════════════════
# 3. H1 (aggregate M5 → H1)
# ══════════════════════════════════════════════════════════════════════════════
print("━"*80)
print("H1 (EUR_JPY M5→H1 aggregated)  WhiteMethod threshold=0.003 price/min  bar_min=60")
print("━"*80)

m5['mid_c'] = (m5.bid_c + m5.ask_c) / 2
m5['mid_o'] = (m5.bid_o + m5.ask_o) / 2
m5['mid_h'] = (m5.bid_h + m5.ask_h) / 2
m5['mid_l'] = (m5.bid_l + m5.ask_l) / 2
m5['ts_floor'] = m5.timestamp.dt.floor('h')

h1 = m5.groupby('ts_floor').agg(
    timestamp  = ('timestamp', 'first'),
    mid_o      = ('mid_o',   'first'),
    mid_h      = ('mid_h',   'max'),
    mid_l      = ('mid_l',   'min'),
    mid_c      = ('mid_c',   'last'),
    ask_c      = ('ask_c',   'last'),
    bid_c      = ('bid_c',   'last'),
).reset_index(drop=True)

hc  = h1.mid_c.values.astype(np.float64)
hsp = (h1.ask_c - h1.bid_c).values.astype(np.float64)
hts = h1.timestamp.values.astype(np.int64)
bar_min_h1 = np.array([60.0])
NH = len(hc)
split_h1 = int(NH * 0.70)

print(f"  {NH:,} H1 bars  {h1.timestamp.iloc[0].date()} → {h1.timestamp.iloc[-1].date()}")

for direction, ct in [("trend", False), ("counter", True)]:
    for thr in [0.003, 0.010, 0.030]:
        tr_is,  eq_is,  _ = run_strategy(hc, hsp, bar_min_h1, SMA_P, LOOKBACK, thr, PIP, ct, 0, split_h1)
        tr_oos, eq_oos, _ = run_strategy(hc, hsp, bar_min_h1, SMA_P, LOOKBACK, thr, PIP, ct, split_h1, NH)
        si = calc_stats(tr_is,  eq_is,  hts, 0, split_h1)
        so = calc_stats(tr_oos, eq_oos, hts, split_h1, NH)
        show(f"IS  {direction:7s} thr={thr:.3f}", si)
        show(f"OOS {direction:7s} thr={thr:.3f}", so)
        print()
