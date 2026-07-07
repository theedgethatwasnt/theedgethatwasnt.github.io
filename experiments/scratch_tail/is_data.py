#!/usr/bin/env python3
"""
is_data.py — scratch_tail IS-only data loader. HARD OOS GUARD.

IS_END is the pre-registered IS/OOS boundary (PREREGISTRATION.md "Data (fixed)": IS = first
70% (-> ~= 2024-09-25), OOS = final 30%, SEALED). Every script in this battery must load data
ONLY through `load_pair_is()` below — same two-layer guard convention as
research/experiments/multiday_contrarian/is_data.py (pushdown filter at parquet-read time +
independent post-read assertion).

The 6 pairs are the deployed sma_scratch set (signal.PAIRS), a strict subset of the 12-pair
multiday_contrarian universe.
"""
import os

import pandas as pd

from signal import PAIRS

IS_END = pd.Timestamp("2024-09-25T00:00:00", tz="UTC")


def to_utc(ts):
    """Normalize any timestamp-like to a tz-aware UTC pd.Timestamp (searchsorted-derived
    numpy datetime64 values are naive-but-UTC; this re-attaches the tz)."""
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
