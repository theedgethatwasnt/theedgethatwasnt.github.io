#!/usr/bin/env python3
"""
is_data.py — fx_factors shared IS-only D1 data loader. HARD OOS GUARD.

IS_END is the pre-registered IS/OOS boundary (PREREGISTRATION.md "Data (fixed)": IS = first
70% (-> ~= 2024-08-30), OOS = final 30%, SEALED). Mirrors multiday_contrarian/is_data.py's
pattern exactly (R6): the IS_END filter is pushed down to pyarrow at parquet-read time
(`filters=`) so OOS M5 rows are never even deserialized, PLUS a second, independent
post-read assertion. D1 bars are then built from the IS-only M5 frame via bars.m5_to_d1
(reused verbatim, not reimplemented) — R1 (closed-bars-only) in bars.py additionally drops
any trailing partial D1 bar, so a stray partial day sitting right at the IS_END cut is
dropped automatically rather than silently included half-formed.

No other module in this directory may call `pd.read_parquet` directly on `m5_ba/*` — route
every load through `load_pair_is_d1()` below.
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multiday_contrarian"))
from bars import m5_to_d1  # noqa: E402

IS_END = pd.Timestamp("2024-08-30T00:00:00", tz="UTC")

PAIRS = [
    "AUD_JPY", "AUD_USD", "CAD_JPY", "CHF_JPY", "EUR_GBP", "EUR_JPY",
    "EUR_USD", "GBP_JPY", "GBP_USD", "NZD_JPY", "NZD_USD", "USD_JPY",
]

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]


def to_utc(ts):
    """Normalize any timestamp-like to a tz-aware UTC pd.Timestamp (see multiday_contrarian/
    is_data.py's identical helper — numpy .values on a tz-aware series silently strips tz)."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def load_pair_is_m5(pair, data_dir):
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


def load_pair_is_d1(pair, data_dir):
    """M5 IS-only -> D1 (bars.m5_to_d1, reused verbatim). Belt-and-suspenders OOS guard
    re-checked on the aggregated D1 output too."""
    m5 = load_pair_is_m5(pair, data_dir)
    d1 = m5_to_d1(m5)
    if len(d1) == 0:
        raise RuntimeError(f"{pair}: 0 IS D1 bars aggregated — check M5 coverage")
    assert to_utc(d1["timestamp"].max()) < IS_END, (
        f"{pair}: OOS LEAK in D1 aggregation — max ts {d1['timestamp'].max()} >= IS_END {IS_END}"
    )
    return d1


def load_spx_is(cross_asset_dir):
    """SPX500_USD D1 (mid OHLC), IS-filtered. This is GATE data only (pre-reg: 'extended-
    history context from data/cross_asset ... is GATE data only, never signal') — used solely
    to compute the SPX SMA(200) risk-off gate, never as a factor input."""
    path = os.path.join(cross_asset_dir, "SPX500_USD_D1.parquet")
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df = df[df["time"] < IS_END].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(f"0 IS SPX rows loaded from {path}")
    assert df["time"].max() < IS_END, "SPX OOS LEAK"
    return df
