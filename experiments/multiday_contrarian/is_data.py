#!/usr/bin/env python3
"""
is_data.py — Task A5 shared IS-only data loader. HARD OOS GUARD.

IS_END is the pre-registered IS/OOS boundary (PREREGISTRATION.md "Data (fixed)":
IS ~= 2020-11-11 -> 2024-09-25, OOS = 2024-09-25 -> 2026-05-21, SEALED). Every script
in the A5 battery must load data ONLY through `load_pair_is()` below:

  1. The IS_END filter is pushed down to pyarrow at parquet-read time (`filters=`), so
     OOS rows are never even deserialized off disk, let alone read into a Python object.
  2. A second, independent assertion re-checks the loaded max timestamp < IS_END after
     the read, in case a future caller ever changes the filter and forgets this file.

No other module in this directory may call `pd.read_parquet` directly on `m5_ba/*` —
route every load through here.
"""
import os

import pandas as pd

IS_END = pd.Timestamp("2024-09-25T00:00:00", tz="UTC")

PAIRS = [
    "AUD_JPY", "AUD_USD", "CAD_JPY", "CHF_JPY", "EUR_GBP", "EUR_JPY",
    "EUR_USD", "GBP_JPY", "GBP_USD", "NZD_JPY", "NZD_USD", "USD_JPY",
]


def to_utc(ts):
    """Normalize any timestamp-like (numpy datetime64, naive/aware pd.Timestamp) to a
    tz-aware UTC pd.Timestamp. harness.py's M5 arrays come from `.values` on a tz-aware
    UTC series, which numpy silently strips to naive datetime64 (still a UTC instant) — this
    re-attaches the tz so every downstream comparison against IS_END (tz-aware) is safe,
    and so CSV round-trips carry an explicit UTC offset."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def load_pair_is(pair, data_dir):
    """Load one pair's M5 bid/ask parquet, hard-filtered to the sealed IS window."""
    path = os.path.join(data_dir, f"{pair}_M5_BA.parquet")
    df = pd.read_parquet(path, filters=[("timestamp", "<", IS_END)])
    if len(df) == 0:
        raise RuntimeError(f"{pair}: 0 IS rows loaded from {path} — check path/filter")
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].max() < IS_END, (
        f"{pair}: OOS LEAK — loaded max ts {df['timestamp'].max()} >= IS_END {IS_END}"
    )
    return df
