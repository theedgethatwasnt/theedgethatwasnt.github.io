#!/usr/bin/env python3
"""
Export unified training parquets with ALL candidate features from M5 OHLC.
Used for feature set experiments A-F.

Features:
  tec_5        — Signed Kaufman ER (5-bar), arctan
  bb_width     — Bollinger Band width / close
  h1_slope     — H1 linreg slope (3 H1 bars from M5), arctan
  stoch_d      — Stochastic %D (14,3)
  range_pos_30 — 30-bar range position
  gap_norm     — Gap between consecutive bars / prev range
  macd_hist    — MACD histogram / ATR
  aroon_osc    — Aroon oscillator (25-bar)
  mc_d_a       — ASI momentum direction
  hl_price     — Price higher-lows (binary, from TopsBots)

Output: data/unified_indicators/{PAIR}_unified.parquet
"""

import sys, os, gc
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.asi_indicator import compute_asi, sma_jit, compute_mc_on_series, TF_BARS_S5, TF_WEIGHTS, N_TFS
from lib.swing_indicators import compute_all_swing_features

OHLC_DIR = PROJECT_ROOT / "data" / "m5_ohlc"
OUT_DIR = PROJECT_ROOT / "data" / "unified_indicators"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_PAIRS = [
    "EUR_JPY", "USD_JPY", "GBP_JPY", "AUD_JPY",
    "CAD_JPY", "CHF_JPY", "NZD_JPY",
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "EUR_GBP",
]


@njit(cache=True)
def compute_tec(closes, window=5):
    n = len(closes)
    out = np.zeros(n)
    hp = np.pi / 2.0
    for i in range(window, n):
        net = closes[i] - closes[i - window]
        path = 0.0
        for j in range(i - window + 1, i + 1):
            path += abs(closes[j] - closes[j - 1])
        if path > 0:
            out[i] = np.arctan((net / path) / 0.3) / hp
    return out


@njit(cache=True)
def compute_h1_slope(closes, bar_period=12, slope_bars=3):
    n = len(closes)
    out = np.zeros(n)
    hp = np.pi / 2.0
    lookback = slope_bars * bar_period
    for i in range(lookback, n):
        vals = np.zeros(slope_bars)
        for k in range(slope_bars):
            vals[slope_bars - 1 - k] = closes[i - k * bar_period]
        x_mean = (slope_bars - 1) / 2.0
        y_mean = 0.0
        for k in range(slope_bars): y_mean += vals[k]
        y_mean /= slope_bars
        num = 0.0; den = 0.0
        for k in range(slope_bars):
            xd = k - x_mean
            num += xd * (vals[k] - y_mean)
            den += xd * xd
        if den > 0:
            slope = num / den
            rng = vals.max() - vals.min()
            out[i] = np.arctan((slope / rng * 3.0) if rng > 0 else 0.0) / hp
    return out


def compute_bb_width(close, period=20, std_mult=2.0):
    sma = pd.Series(close).rolling(period, min_periods=1).mean().values
    std = pd.Series(close).rolling(period, min_periods=1).std().values
    std = np.where(std > 0, std, 1e-10)
    return (sma + std_mult * std - (sma - std_mult * std)) / np.where(sma > 0, sma, 1e-10)


def compute_stoch_d(high, low, close, k_period=14, d_period=3):
    lowest = pd.Series(low).rolling(k_period, min_periods=1).min().values
    highest = pd.Series(high).rolling(k_period, min_periods=1).max().values
    rng = highest - lowest
    rng = np.where(rng > 0, rng, 1e-10)
    k = (close - lowest) / rng
    return pd.Series(k).rolling(d_period, min_periods=1).mean().values


@njit(cache=True)
def compute_range_pos(high, low, period=30):
    n = len(high)
    out = np.zeros(n)
    for i in range(period, n):
        hh = high[i]; ll = low[i]
        for j in range(i - period, i):
            if high[j] > hh: hh = high[j]
            if low[j] < ll: ll = low[j]
        rng = hh - ll
        if rng > 0: out[i] = ((high[i] + low[i]) / 2.0 - ll) / rng
    return out


def compute_gap_norm(open_arr, close_arr, high, low):
    prev_close = np.roll(close_arr, 1); prev_close[0] = close_arr[0]
    prev_high = np.roll(high, 1); prev_high[0] = high[0]
    prev_low = np.roll(low, 1); prev_low[0] = low[0]
    prev_range = np.maximum(prev_high - prev_low, 1e-10)
    return (open_arr - prev_close) / prev_range


def compute_macd_hist(close):
    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    hist = ema_fast - ema_slow - pd.Series(ema_fast - ema_slow).ewm(span=9, adjust=False).mean().values
    tr = np.maximum(np.maximum(close, np.roll(close, 1)) - np.minimum(close, np.roll(close, 1)),
                    np.abs(close - np.roll(close, 1)))
    tr[0] = 0
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    return hist / np.where(atr > 0, atr, 1e-10)


@njit(cache=True)
def compute_aroon_osc(high, low, period=25):
    n = len(high)
    out = np.zeros(n)
    for i in range(period, n):
        hh_idx = 0; ll_idx = 0
        hh_val = high[i - period]; ll_val = low[i - period]
        for j in range(1, period + 1):
            if high[i - period + j] >= hh_val: hh_val = high[i - period + j]; hh_idx = j
            if low[i - period + j] <= ll_val: ll_val = low[i - period + j]; ll_idx = j
        out[i] = ((period - (period - hh_idx)) - (period - (period - ll_idx))) / period
    return out


def compute_mc_d_a(o, h, l, c, n):
    asi = compute_asi(o, h, l, c, n)
    smooth = sma_jit(asi, 5, n)
    mc_d, _ = compute_mc_on_series(smooth, n, TF_BARS_S5, TF_WEIGHTS, N_TFS)
    return mc_d


def main():
    for pair in ALL_PAIRS:
        out_path = OUT_DIR / f"{pair}_unified.parquet"
        if out_path.exists():
            print(f"{pair}: exists, skip"); continue

        print(f"\n=== {pair} ===", flush=True)
        df = pd.read_parquet(OHLC_DIR / f"{pair}_M5.parquet")
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        c = df["close"].values.astype(np.float64)
        n = len(c)

        print(f"  {n:,} bars. Computing...", flush=True)
        tec = compute_tec(c, 5)
        bb_w = compute_bb_width(c)
        h1_sl = compute_h1_slope(c, 12, 3)
        stoch = compute_stoch_d(h, l, c)
        rng_pos = compute_range_pos(h, l, 30)
        gap = compute_gap_norm(o, c, h, l)
        macd = compute_macd_hist(c)
        aroon = compute_aroon_osc(h, l, 25)
        mc_d = compute_mc_d_a(o, h, l, c, n)

        # TopsBots price swing features
        asi = compute_asi(o, h, l, c, n)
        swing = compute_all_swing_features(o, h, l, c, asi)
        hl_price = swing["hl_price"].astype(np.float64)

        out_df = pd.DataFrame({
            "timestamp": df["timestamp"], "mid_close": c,
            "tec_5": tec, "bb_width": bb_w, "h1_slope": h1_sl,
            "stoch_d": stoch, "range_pos_30": rng_pos, "gap_norm": gap,
            "macd_hist": macd, "aroon_osc": aroon, "mc_d_a": mc_d,
            "hl_price": hl_price,
        })
        out_df.to_parquet(out_path, engine="pyarrow", index=False)
        print(f"  SAVED: {out_path.name}")
        del df, out_df; gc.collect()

    print("\nDone!")


if __name__ == "__main__":
    main()
