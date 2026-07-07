"""d1_data.py — COT Contrarian Positioning: D1 price panel loader + realized-vol + spread.

Loads data/d1_deep_ba/<PAIR>_D1.parquet (fetched by fetch_d1.py; see that module's
docstring for the "why not data/d1_deep/" collision note). Provides:
  - load_pair(pair): DataFrame indexed by calendar date (normalized from the tz-aware
    OANDA bar-start timestamp), columns open/high/low/close/bid_c/ask_c/volume.
  - trading_calendar(pairs): union of dates across pairs (release_lag.py's action-date
    resolution needs a shared FX-market-open calendar, not any one pair's quirks).
  - realized_vol_pips(df, pair, window=63): trailing (STRICTLY PAST, shift(1)-safe) stdev
    of daily close-to-close pip returns — the "63-day realized-vol scaling" the
    pre-registration's Signal section specifies for equal-risk sizing.
  - median_spread_pips(df, pair): per-pair median (ask_c - bid_c) in pips.

R1/R6 note: realized_vol_pips is deliberately shifted by 1 day relative to its own
`date` index (vol "as of" date D uses only returns strictly BEFORE D) so sizing a
position entered on date D never uses date D's own (not-yet-realized) return.
"""
import os

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Override for deployment layouts that don't mirror the full fx-core repo nesting (e.g.
# Hetzner /root/work/code_cot/ is a flat directory, not 3 levels under a repo root) —
# set COT_D1_DATA_DIR to point at wherever data/d1_deep_ba/*.parquet was rsynced to.
DATA_DIR = os.environ.get("COT_D1_DATA_DIR") or os.path.join(REPO, "data", "d1_deep_ba")


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def load_pair(pair: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{pair}_D1.parquet")
    df = pd.read_parquet(path)
    df = df.rename(columns={"timestamp": "time"}) if "timestamp" in df.columns else df
    df["raw_ts"] = pd.to_datetime(df["time"])  # original tz-aware bar-start instant, kept
    # for carry_splice (rollover-day counting is timestamp-sensitive, NY 17:00 anchored) —
    # the normalized `date` index below is for LOOKUP/matching only.
    df["date"] = df["raw_ts"].dt.normalize()
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    pip = pip_size(pair)
    df["daily_ret_pips"] = df["close"].diff() / pip
    df["spread_pips"] = (df["ask_c"] - df["bid_c"]) / pip
    return df


def trading_calendar(pairs, data_dir: str = DATA_DIR) -> pd.DatetimeIndex:
    idx = None
    for pair in pairs:
        d = load_pair(pair, data_dir).index
        idx = d if idx is None else idx.union(d)
    return idx.sort_values()


def realized_vol_pips(df: pd.DataFrame, window: int = 63, min_periods: int = 20) -> pd.Series:
    """Trailing stdev of daily pip returns, SHIFTED by 1 so the value indexed at date D
    uses only returns from dates < D (no same-day lookahead)."""
    vol = df["daily_ret_pips"].rolling(window=window, min_periods=min_periods).std(ddof=1)
    return vol.shift(1)


def median_spread_pips(df: pd.DataFrame) -> float:
    return float(df["spread_pips"].median())
