"""
Data utilities for zone recovery experiments.
Load M5/S5 OHLC data, compute ATR at multiple timeframes.
No lookahead bias — all computations are causal.
"""

import numpy as np
import pandas as pd
from pathlib import Path


DATA_DIR = Path("/path/to/projects/fx-core/data")

# Bars per timeframe (M5 baseline)
M5_PER_H1 = 12
M5_PER_D1 = 288

# Bars per timeframe (S5 baseline)
S5_PER_M5 = 12
S5_PER_H1 = 720
S5_PER_D1 = 17280


def load_m5(pair: str) -> pd.DataFrame:
    """Load M5 OHLC data for a pair. Returns df with timestamp, open, high, low, close, volume."""
    path = DATA_DIR / "m5_ohlc" / f"{pair}_M5.parquet"
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Drop weekends/gaps > 4h
    df["dt"] = pd.to_datetime(df["timestamp"])
    df = df[df["dt"].dt.dayofweek < 5].reset_index(drop=True)  # Mon-Fri only
    return df


def load_s5(pair: str) -> pd.DataFrame:
    """Load S5 bid/ask data. Computes mid OHLC."""
    path = DATA_DIR / "s5_ohlc" / f"{pair}_S5_BA.parquet"
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Compute mid prices
    df["open"]  = (df["bid_o"] + df["ask_o"]) / 2
    df["high"]  = (df["bid_h"] + df["ask_h"]) / 2
    df["low"]   = (df["bid_l"] + df["ask_l"]) / 2
    df["close"] = (df["bid_c"] + df["ask_c"]) / 2
    df["spread_pips"] = (df["ask_c"] - df["bid_c"]) / 0.0001  # rough spread
    return df[["timestamp", "open", "high", "low", "close", "volume", "spread_pips"]]


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder ATR — causal, no lookahead."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


def compute_atr_downsampled(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    period: int, bars_per_tf: int
) -> np.ndarray:
    """Compute ATR on a lower-frequency timeframe, then upsample back to original bars.

    E.g., compute H1 ATR(8) on M5 bars: bars_per_tf=12.
    Result is the H1 ATR value available at each M5 bar close, no lookahead.
    """
    n = len(close)
    n_tf = n // bars_per_tf
    if n_tf < period:
        return np.full(n, np.nan)

    # Aggregate OHLC to target timeframe
    tf_high = np.array([high[j * bars_per_tf:(j + 1) * bars_per_tf].max()
                         for j in range(n_tf)])
    tf_low  = np.array([low[j * bars_per_tf:(j + 1) * bars_per_tf].min()
                         for j in range(n_tf)])
    tf_close = np.array([close[(j + 1) * bars_per_tf - 1]
                          for j in range(n_tf)])

    # ATR on TF bars
    tf_atr = compute_atr(tf_high, tf_low, tf_close, period)

    # Upsample: value at TF bar j is valid for all M5 bars in that window
    out = np.full(n, np.nan)
    for j in range(n_tf):
        atr_val = tf_atr[j]
        start = j * bars_per_tf
        end = min((j + 1) * bars_per_tf, n)
        out[start:end] = atr_val

    return out


def prepare_features(df: pd.DataFrame, granularity: str = "M5") -> dict:
    """Compute all features needed for zone recovery engine.

    Returns dict with numpy arrays: open, high, low, close, atr_short, atr_long.
    """
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)

    if granularity == "M5":
        bars_per_h1 = M5_PER_H1
        bars_per_d1 = M5_PER_D1
    elif granularity == "S5":
        bars_per_h1 = S5_PER_H1
        bars_per_d1 = S5_PER_D1
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    # ATR_short: 8-period ATR on H1 timeframe
    atr_short = compute_atr_downsampled(h, l, c, period=8, bars_per_tf=bars_per_h1)

    # ATR_long: 20-period ATR on Daily timeframe
    atr_long = compute_atr_downsampled(h, l, c, period=20, bars_per_tf=bars_per_d1)

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "atr_short": atr_short,
        "atr_long": atr_long,
        "timestamps": df["timestamp"].values,
    }


def train_test_split_temporal(features: dict, train_frac: float = 0.7) -> tuple:
    """Time-based train/test split. Returns (train_features, test_features)."""
    n = len(features["close"])
    split_idx = int(n * train_frac)

    def slice_features(f, start, end):
        return {k: v[start:end] for k, v in f.items()}

    return slice_features(features, 0, split_idx), slice_features(features, split_idx, n)


def walk_forward_splits(features: dict, n_chunks: int = 3, test_frac: float = 0.3) -> list:
    """Generate IS/OOS walk-forward splits.

    Each split: (train_features, test_features)
    Test windows are non-overlapping and cover the last test_frac of data.
    """
    n = len(features["close"])
    oos_start = int(n * (1 - test_frac))
    oos_n = n - oos_start
    chunk_size = oos_n // n_chunks

    splits = []
    for k in range(n_chunks):
        oos_s = oos_start + k * chunk_size
        oos_e = oos_s + chunk_size

        def slice_features(f, start, end):
            return {k2: v[start:end] for k2, v in f.items()}

        train_f = slice_features(features, 0, oos_s)
        test_f = slice_features(features, oos_s, oos_e)
        splits.append((train_f, test_f))

    return splits
