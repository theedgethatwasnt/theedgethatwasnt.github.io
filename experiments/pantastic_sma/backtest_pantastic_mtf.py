#!/usr/bin/env python3
"""
PantasticSMA — M5 signal × H1 agreement filter (EUR_JPY, 5.5yr).

Enter M5 trend-follow signal only when H1 PantasticSMA agrees (same direction).
H1 rate is causal: each M5 bar uses the H1 bar that closed at or before that bar.

Sweep: m5_thr × h1_thr × both directions.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

PIP      = 0.01   # EUR_JPY
SMA_P    = 7
LOOKBACK = 5

# ── Numba kernel ───────────────────────────────────────────────────────────────
@njit
def run_mtf(mc, msp, h1_rate, m5_thr, h1_thr, pip, counter_trend, start_i, end_i):
    """M5 entry with H1 agreement filter. h1_thr=0 means sign-only filter."""
    n      = end_i - start_i
    warmup = SMA_P + LOOKBACK + 1
    sma_buf = np.zeros(SMA_P, np.float64); sma_sum = 0.0; sma_filled = 0
    sma_vals = np.full(n, np.nan)
    trades_p = np.empty(n, np.float64); n_tr = 0
    equity = np.zeros(n, np.float64); cum_pnl = 0.0
    pos = 0; entry_price = 0.0
    BAR_MIN = 5.0

    for ii in range(n):
        i = start_i + ii
        oldest  = sma_buf[ii % SMA_P]
        sma_sum = sma_sum - oldest + mc[i]
        sma_buf[ii % SMA_P] = mc[i]
        if sma_filled < SMA_P: sma_filled += 1
        if sma_filled >= SMA_P: sma_vals[ii] = sma_sum / SMA_P

        equity[ii] = cum_pnl
        if ii < warmup: continue
        prev_ii = ii - LOOKBACK
        if np.isnan(sma_vals[ii]) or np.isnan(sma_vals[prev_ii]): continue

        rate_m5 = (sma_vals[ii] - sma_vals[prev_ii]) / (LOOKBACK * BAR_MIN)
        r_h1    = h1_rate[i]
        if np.isnan(r_h1): continue

        # M5 signal
        if counter_trend:
            m5_sig = -1 if rate_m5 > m5_thr else (1 if rate_m5 < -m5_thr else 0)
        else:
            m5_sig =  1 if rate_m5 > m5_thr else (-1 if rate_m5 < -m5_thr else 0)

        # H1 agreement: trade direction must match H1 momentum direction
        if m5_sig == 1:
            filtered = 1 if r_h1 >= h1_thr else 0
        elif m5_sig == -1:
            filtered = -1 if r_h1 <= -h1_thr else 0
        else:
            filtered = 0

        if filtered == pos: continue
        sp = msp[i]; half_sp = sp / 2.0

        if pos != 0:
            ep  = mc[i] - half_sp if pos == 1 else mc[i] + half_sp
            pnl = (ep - entry_price) / pip if pos == 1 else (entry_price - ep) / pip
            cum_pnl += pnl; trades_p[n_tr] = pnl; n_tr += 1

        if filtered == 1:  entry_price = mc[i] + half_sp
        elif filtered == -1: entry_price = mc[i] - half_sp
        pos = filtered; equity[ii] = cum_pnl

    if pos != 0:
        sp = msp[end_i - 1]; half_sp = sp / 2.0
        ep = mc[end_i-1] - half_sp if pos == 1 else mc[end_i-1] + half_sp
        pnl = (ep - entry_price) / pip if pos == 1 else (entry_price - ep) / pip
        cum_pnl += pnl; trades_p[n_tr] = pnl; n_tr += 1; equity[n-1] = cum_pnl

    return trades_p[:n_tr], equity, cum_pnl


@njit
def run_m5_baseline(mc, msp, m5_thr, pip, counter_trend, start_i, end_i):
    """M5-only baseline (no H1 filter) for comparison."""
    n      = end_i - start_i
    warmup = SMA_P + LOOKBACK + 1
    sma_buf = np.zeros(SMA_P, np.float64); sma_sum = 0.0; sma_filled = 0
    sma_vals = np.full(n, np.nan)
    trades_p = np.empty(n, np.float64); n_tr = 0
    equity = np.zeros(n, np.float64); cum_pnl = 0.0
    pos = 0; entry_price = 0.0
    BAR_MIN = 5.0

    for ii in range(n):
        i = start_i + ii
        oldest  = sma_buf[ii % SMA_P]
        sma_sum = sma_sum - oldest + mc[i]
        sma_buf[ii % SMA_P] = mc[i]
        if sma_filled < SMA_P: sma_filled += 1
        if sma_filled >= SMA_P: sma_vals[ii] = sma_sum / SMA_P

        equity[ii] = cum_pnl
        if ii < warmup: continue
        prev_ii = ii - LOOKBACK
        if np.isnan(sma_vals[ii]) or np.isnan(sma_vals[prev_ii]): continue

        rate_m5 = (sma_vals[ii] - sma_vals[prev_ii]) / (LOOKBACK * BAR_MIN)
        if counter_trend:
            sig = -1 if rate_m5 > m5_thr else (1 if rate_m5 < -m5_thr else 0)
        else:
            sig =  1 if rate_m5 > m5_thr else (-1 if rate_m5 < -m5_thr else 0)

        if sig == pos: continue
        sp = msp[i]; half_sp = sp / 2.0

        if pos != 0:
            ep  = mc[i] - half_sp if pos == 1 else mc[i] + half_sp
            pnl = (ep - entry_price) / pip if pos == 1 else (entry_price - ep) / pip
            cum_pnl += pnl; trades_p[n_tr] = pnl; n_tr += 1

        if sig == 1:  entry_price = mc[i] + half_sp
        elif sig == -1: entry_price = mc[i] - half_sp
        pos = sig; equity[ii] = cum_pnl

    if pos != 0:
        sp = msp[end_i - 1]; half_sp = sp / 2.0
        ep = mc[end_i-1] - half_sp if pos == 1 else mc[end_i-1] + half_sp
        pnl = (ep - entry_price) / pip if pos == 1 else (entry_price - ep) / pip
        cum_pnl += pnl; trades_p[n_tr] = pnl; n_tr += 1; equity[n-1] = cum_pnl

    return trades_p[:n_tr], equity, cum_pnl


# ── Stats ──────────────────────────────────────────────────────────────────────
def calc_stats(trades, equity, ts_ns, start_i, end_i):
    ts0   = pd.Timestamp(ts_ns[start_i],  unit='ns', tz='UTC')
    ts1   = pd.Timestamp(ts_ns[end_i - 1], unit='ns', tz='UTC')
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
    if not s:
        print(f"  {label:45s}  no trades")
        return
    print(f"  {label:45s}  n={s['n']:5d}({s['tpd']:5.1f}/d)  "
          f"WR={s['wr']:4.1f}%  W={s['aw']:+5.2f}p  L={s['al']:+5.2f}p  "
          f"PF={s['pf']:.2f}  p/d={s['ppd']:+7.2f}  DD={s['max_dd']:7.1f}p")


# ══════════════════════════════════════════════════════════════════════════════
# Load M5 data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading EUR_JPY M5 BA data...")
m5 = pd.read_parquet(ROOT / "data" / "m5_ba" / "EUR_JPY_M5_BA.parquet").sort_values("timestamp").reset_index(drop=True)
m5['mid_c'] = (m5.bid_c + m5.ask_c) / 2
mc  = m5.mid_c.values.astype(np.float64)
msp = (m5.ask_c - m5.bid_c).values.astype(np.float64)
mts = m5.timestamp.values.astype(np.int64)
NM  = len(mc)
split_m5 = int(NM * 0.70)
print(f"  {NM:,} M5 bars  {m5.timestamp.iloc[0].date()} → {m5.timestamp.iloc[-1].date()}")
print(f"  IS: {m5.timestamp.iloc[0].date()} → {m5.timestamp.iloc[split_m5-1].date()}")
print(f"  OOS:{m5.timestamp.iloc[split_m5].date()} → {m5.timestamp.iloc[-1].date()}")

# ── Build H1 from M5, compute causal rate ──────────────────────────────────────
print("\nBuilding H1 rates (causal, shift-1)...")
m5['ts_floor'] = m5.timestamp.dt.floor('h')
h1 = m5.groupby('ts_floor').agg(
    timestamp = ('timestamp', 'first'),
    mid_c     = ('mid_c',   'last'),
    ask_c     = ('ask_c',   'last'),
    bid_c     = ('bid_c',   'last'),
).reset_index(drop=True)

h1['sma']  = h1.mid_c.rolling(SMA_P, min_periods=SMA_P).mean()
h1['rate'] = (h1['sma'] - h1['sma'].shift(LOOKBACK)) / (LOOKBACK * 60.0)
# Causal: shift by 1 so each M5 bar sees the CLOSED H1 bar, not the in-progress one
h1['rate_causal'] = h1['rate'].shift(1)
# "available_at" = next H1 open = this H1 timestamp + 1h (the bar is complete at open of next bar)
h1['avail_ts'] = h1.timestamp + pd.Timedelta(hours=1)

print(f"  {len(h1):,} H1 bars, {h1.rate_causal.notna().sum()} with valid rate")

# Align H1 rate onto M5 timestamps (forward-fill)
h1_lookup = h1[['avail_ts', 'rate_causal']].dropna().rename(columns={'avail_ts': 'timestamp'})
m5_merged = pd.merge_asof(
    m5[['timestamp']].copy(),
    h1_lookup.sort_values('timestamp'),
    on='timestamp',
    direction='backward'
)
h1_rate_aligned = m5_merged['rate_causal'].values.astype(np.float64)
valid_h1 = (~np.isnan(h1_rate_aligned)).sum()
print(f"  Aligned: {valid_h1:,}/{NM:,} M5 bars have valid H1 rate ({100*valid_h1/NM:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# Baseline: M5 standalone
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "━"*90)
print("BASELINE — M5 standalone (no H1 filter)")
print("━"*90)
for direction, ct in [("trend", False), ("counter", True)]:
    for thr in [0.003, 0.010, 0.030]:
        tr_is,  eq_is,  _ = run_m5_baseline(mc, msp, thr, PIP, ct, 0, split_m5)
        tr_oos, eq_oos, _ = run_m5_baseline(mc, msp, thr, PIP, ct, split_m5, NM)
        si = calc_stats(tr_is,  eq_is,  mts, 0, split_m5)
        so = calc_stats(tr_oos, eq_oos, mts, split_m5, NM)
        show(f"IS  {direction:7s} m5_thr={thr:.3f}", si)
        show(f"OOS {direction:7s} m5_thr={thr:.3f}", so)
        print()

# ══════════════════════════════════════════════════════════════════════════════
# MTF: M5 + H1 agreement filter
# ══════════════════════════════════════════════════════════════════════════════
for direction, ct in [("trend", False), ("counter", True)]:
    print("\n" + "━"*90)
    print(f"M5 + H1 FILTER — {direction.upper()}-FOLLOW")
    print("━"*90)
    for m5_thr in [0.003, 0.010, 0.030]:
        for h1_thr in [0.0, 0.003, 0.010, 0.030]:
            h1_label = "sign" if h1_thr == 0.0 else f"{h1_thr:.3f}"
            label = f"m5={m5_thr:.3f} h1>={h1_label}"
            tr_is,  eq_is,  _ = run_mtf(mc, msp, h1_rate_aligned, m5_thr, h1_thr, PIP, ct, 0, split_m5)
            tr_oos, eq_oos, _ = run_mtf(mc, msp, h1_rate_aligned, m5_thr, h1_thr, PIP, ct, split_m5, NM)
            si = calc_stats(tr_is,  eq_is,  mts, 0, split_m5)
            so = calc_stats(tr_oos, eq_oos, mts, split_m5, NM)
            show(f"IS  {label}", si)
            show(f"OOS {label}", so)
            print()
