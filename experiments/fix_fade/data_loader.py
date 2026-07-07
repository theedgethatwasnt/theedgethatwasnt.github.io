#!/usr/bin/env python3
"""
data_loader.py — London-Fix Fade IS-only data loader. HARD OOS GUARD.

IS_END derivation (task brief: "IS ceiling = first 70% of 2020-11-11 -> 2026-05-21, compute
the exact date, hard-assert in loaders"). PREREGISTRATION.md's window is stated as calendar
dates (no time-of-day), so the exact boundary is computed as the calendar-time fraction of the
full stated window, floored to the nearest 5-min bar boundary (M5 bars only exist on 5-min
marks, so any finer-grained cutoff would be unreachable anyway):

    start   = 2020-11-11T00:00:00 UTC
    end     = 2026-05-21T00:00:00 UTC
    span    = end - start                = 2017 days exactly
    is_end  = (start + span * 0.70).floor("5min")
            = 2024-09-22T21:35:59 UTC -> floored -> 2024-09-22T21:35:00 UTC
    (actual IS fraction of the full span: 0.699999656 — 5-min floor rounds down by <1s)

OOS (2024-09-22T21:35:00 UTC -> 2026-05-21T00:00:00 UTC) stays SEALED: no script in this
directory may read past IS_END. Every loader call is routed through `load_pair_is()` below,
which (1) pushes the IS_END filter down to pyarrow at parquet-read time so OOS rows are never
deserialized, and (2) re-asserts the loaded max timestamp < IS_END after the read as a second,
independent guard.
"""
import os

import pandas as pd

IS_END = pd.Timestamp("2024-09-22T21:35:00", tz="UTC")

PAIRS = [
    "AUD_JPY", "AUD_USD", "CAD_JPY", "CHF_JPY", "EUR_GBP", "EUR_JPY",
    "EUR_USD", "GBP_JPY", "GBP_USD", "NZD_JPY", "NZD_USD", "USD_JPY",
]


def pip_of(pair):
    return 0.01 if pair.endswith("_JPY") else 0.0001


def to_utc(ts):
    """Normalize any timestamp-like (numpy datetime64, naive/aware pd.Timestamp) to a
    tz-aware UTC pd.Timestamp — numpy `.values` arrays silently strip tz info to naive
    datetime64 (still a UTC instant); this re-attaches it before any comparison against
    IS_END or CSV round-trip."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def load_pair_is(pair, data_dir):
    """Load one pair's M5 bid/ask parquet, hard-filtered to the sealed IS window
    (timestamp < IS_END). Columns: timestamp,open,high,low,close,bid_o,bid_h,bid_l,bid_c,
    ask_o,ask_h,ask_l,ask_c,volume. `timestamp` is the bar's OPEN time (see fixtime.py
    docstring)."""
    path = os.path.join(data_dir, f"{pair}_M5_BA.parquet")
    df = pd.read_parquet(path, filters=[("timestamp", "<", IS_END)])
    if len(df) == 0:
        raise RuntimeError(f"{pair}: 0 IS rows loaded from {path} — check path/filter")
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].max() < IS_END, (
        f"{pair}: OOS LEAK — loaded max ts {df['timestamp'].max()} >= IS_END {IS_END}"
    )
    return df
