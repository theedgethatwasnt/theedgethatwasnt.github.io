#!/usr/bin/env python3
"""
bars.py — SMA-Scratch Tail-Bounding Test (scratch_tail): H1/M30 aggregation on the OANDA
*broker grid*, shared by the harness (R6 — "one code path" so any later live port can never
diverge on bar edges).

Unlike H4/D1 (see research/experiments/multiday_contrarian/bars.py), which OANDA anchors to
NY 17:00, OANDA serves H1 and M30 candles on plain **top-of-hour / top-of-half-hour UTC**
boundaries (PREREGISTRATION.md "Data (fixed)": "H1 = top-of-hour UTC as OANDA serves it; the
live service polls granularity H1/M30 directly"). This is simpler than the H4/D1 case: no NY
wall-clock anchor, no DST-fold ambiguity — floor() on a tz-aware UTC timestamp is unambiguous
at every boundary, every day of the year.

Bars are built from M5 MID OHLC; volume is the tick-volume sum across the M5 bars in the bin;
bid_c/ask_c are the LAST M5 bar's bid/ask close in the bin (bar-level diagnostics only — the
harness's per-trade spread cost still comes from the M5 entry/exit bar directly, per R3a).

R1 (closed bars only): the trailing bar is dropped if the underlying M5 data does not reach
its nominal close (bin_start + bar_minutes) — mid-series bins that are merely thin (e.g.
weekend closures) are kept, since those bars ARE closed, just illiquid.
"""
from datetime import timedelta

import pandas as pd

_COLS = ["timestamp", "open", "high", "low", "close", "volume", "bid_c", "ask_c"]


def _resample_utc_floor(df, minutes):
    """Shared core for m5_to_h1 / m5_to_m30. `minutes` in {30, 60}."""
    if len(df) == 0:
        return df.iloc[0:0][_COLS].copy()

    d = df.sort_values("timestamp").reset_index(drop=True)
    bin_start = d["timestamp"].dt.floor(f"{minutes}min")

    g = d.assign(_bin=bin_start).groupby("_bin", sort=True)
    agg = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        bid_c=("bid_c", "last"),
        ask_c=("ask_c", "last"),
    ).reset_index().rename(columns={"_bin": "timestamp"})

    # R1: drop the trailing partial bar — one whose nominal close extends past the last
    # M5 bar we actually have.
    last_m5_ts = d["timestamp"].max()
    bin_end = agg["timestamp"] + timedelta(minutes=minutes)
    complete = bin_end <= last_m5_ts + timedelta(minutes=5)
    agg = agg[complete].reset_index(drop=True)

    return agg[_COLS]


def m5_to_h1(df):
    """Aggregate an M5-BA dataframe (columns: timestamp,open,high,low,close,bid_c,ask_c,volume,
    tz-aware UTC timestamps) into top-of-hour-UTC H1 bars. Drops the trailing partial bar (R1)."""
    return _resample_utc_floor(df, minutes=60)


def m5_to_m30(df):
    """Same as m5_to_h1 but 30-minute bars, top-of-half-hour UTC."""
    return _resample_utc_floor(df, minutes=30)
