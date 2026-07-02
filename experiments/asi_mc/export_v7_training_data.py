#!/usr/bin/env python3
"""
Export V7 training parquets — 6 SHAP-selected indicators from M5 OHLC.

Inputs (SHAP rank): bb_width (#1), stoch_d (#2), macd_hist (#3),
                    range_pos_30 (#4), aroon_osc (#6), mc_d_a (#13)
+ UPnL (trade state, added at eval time) = 7 inputs total.

Source: data/m5_ohlc/{PAIR}_M5.parquet (5yr OHLC, 393K bars)
Output: data/v7_indicators/{PAIR}_v7.parquet
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

OHLC_DIR = PROJECT_ROOT / "data" / "m5_ohlc"
OUT_DIR = PROJECT_ROOT / "data" / "v7_indicators"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_PAIRS = [
    "EUR_JPY", "USD_JPY", "GBP_JPY", "AUD_JPY",
    "CAD_JPY", "CHF_JPY", "NZD_JPY",
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "EUR_GBP",
]


# ── Indicator functions (matching rank_indicators_comprehensive.py exactly) ──

def compute_bb_width(close, period=20, std_mult=2.0):
    sma = pd.Series(close).rolling(period, min_periods=1).mean().values
    std = pd.Series(close).rolling(period, min_periods=1).std().values
    std = np.where(std > 0, std, 1e-10)
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return (upper - lower) / np.where(sma > 0, sma, 1e-10)


def compute_stoch_d(high, low, close, k_period=14, d_period=3):
    lowest = pd.Series(low).rolling(k_period, min_periods=1).min().values
    highest = pd.Series(high).rolling(k_period, min_periods=1).max().values
    rng = highest - lowest
    rng = np.where(rng > 0, rng, 1e-10)
    k = (close - lowest) / rng
    d = pd.Series(k).rolling(d_period, min_periods=1).mean().values
    return d


def compute_macd_hist(close):
    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    histogram = macd_line - signal_line
    # Normalize by ATR
    tr = np.maximum(np.maximum(close, np.roll(close, 1)) - np.minimum(close, np.roll(close, 1)),
                    np.abs(close - np.roll(close, 1)))
    tr[0] = 0
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    atr = np.where(atr > 0, atr, 1e-10)
    return histogram / atr


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
        if rng > 0:
            out[i] = ((high[i] + low[i]) / 2.0 - ll) / rng
    return out


@njit(cache=True)
def compute_aroon_osc(high, low, period=25):
    n = len(high)
    out = np.zeros(n)
    for i in range(period, n):
        hh_idx = 0; ll_idx = 0
        hh_val = high[i - period]; ll_val = low[i - period]
        for j in range(1, period + 1):
            if high[i - period + j] >= hh_val:
                hh_val = high[i - period + j]; hh_idx = j
            if low[i - period + j] <= ll_val:
                ll_val = low[i - period + j]; ll_idx = j
        bars_hh = period - hh_idx; bars_ll = period - ll_idx
        out[i] = ((period - bars_hh) - (period - bars_ll)) / period
    return out


def compute_mc_d_a(o, h, l, c, n):
    """ASI → SMA5 → MC(D) — curator-identical."""
    asi = compute_asi(o, h, l, c, n)
    smooth = sma_jit(asi, 5, n)
    mc_d, mc_dd = compute_mc_on_series(smooth, n, TF_BARS_S5, TF_WEIGHTS, N_TFS)
    return mc_d


def main():
    for pair in ALL_PAIRS:
        out_path = OUT_DIR / f"{pair}_v7.parquet"
        if out_path.exists():
            print(f"{pair}: exists, skip")
            continue

        print(f"\n=== {pair} ===", flush=True)
        ohlc_path = OHLC_DIR / f"{pair}_M5.parquet"
        if not ohlc_path.exists():
            print(f"  ERROR: {ohlc_path} not found")
            continue

        df = pd.read_parquet(ohlc_path)
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        c = df["close"].values.astype(np.float64)
        n = len(c)
        print(f"  {n:,} M5 bars", flush=True)

        print("  Computing indicators...", flush=True)
        bb_w = compute_bb_width(c, 20, 2.0)
        stoch = compute_stoch_d(h, l, c, 14, 3)
        macd = compute_macd_hist(c)
        rng_pos = compute_range_pos(h, l, 30)
        aroon = compute_aroon_osc(h, l, 25)
        mc_d = compute_mc_d_a(o, h, l, c, n)

        out_df = pd.DataFrame({
            "timestamp": df["timestamp"],
            "mid_close": c,
            "bb_width": bb_w,
            "stoch_d": stoch,
            "macd_hist": macd,
            "range_pos_30": rng_pos,
            "aroon_osc": aroon,
            "mc_d_a": mc_d,
        })

        out_df.to_parquet(out_path, engine="pyarrow", index=False)
        print(f"  SAVED: {out_path.name}")
        print(f"  Ranges: bb_w=[{bb_w[500:].min():.4f},{bb_w[500:].max():.4f}] "
              f"stoch=[{stoch[500:].min():.4f},{stoch[500:].max():.4f}] "
              f"macd=[{macd[500:].min():.4f},{macd[500:].max():.4f}]")
        del df, o, h, l, c, out_df; gc.collect()

    print("\nDone!")


if __name__ == "__main__":
    main()
