#!/usr/bin/env python3
"""
bars.py — Multi-day contrarian program (Workstream A2): H4/D1 aggregation, shared by any
later live port (R6 — "one code path" so backtest and live can never diverge on bar edges).

OANDA anchors its H4 candles to New York 17:00 (the FX trading-day rollover), giving six
4-hour bins per day: 17:00, 21:00, 01:00, 05:00, 09:00, 13:00 (all NY local time). D1 candles
use the same 17:00 anchor with a single 24h bin per day. Because the anchor is a LOCAL
wall-clock hour, the bin grid shifts by exactly one hour in UTC terms across a US DST
transition (spring-forward/fall-back) — this module is anchored in NY wall-clock time so it
tracks OANDA's actual grid through DST, not a fixed UTC offset.

Implementation note (DST safety): bin-start dates are computed in NAIVE wall-clock
arithmetic (no tz attached) and the NY tz is attached only at the very end via
`tz_localize`. The 01:00 anchor DOES fall inside the US fall-back fold (clocks fall back
2:00am -> 1:00am, so local "01:00-02:00" occurs twice on that one night/year) — caught by
test_bars.py's DST test failing against an earlier version of this module that assumed no
anchor was ever ambiguous. We resolve this the same way as an unresolvable/rare edge (R9,
documented, not silently guessed): `tz_localize(..., ambiguous="NaT", nonexistent="NaT")`
turns that handful of bin-starts (≈2 per year, out of ~2,190 H4 bins/year — a ~0.1% loss)
into NaT, and pandas' groupby drops NaT keys by default, so the underlying M5 bars in that
narrow window are silently excluded from any output bar rather than mis-assigned to the
wrong side of the fold. Direction of bias: negligible (a couple of dropped bars/year).

R1 (closed bars only): the trailing bar is dropped if the underlying M5 data does not reach
its nominal close (bin_start + bar_hours) — mid-series bins that are merely thin (weekend
closures) are kept, since those bars ARE closed, just illiquid.

Bars are built from M5 MID OHLC (open/high/low/close columns); volume is the tick-volume sum
across the M5 bars in the bin; bid_c/ask_c are the LAST M5 bar's bid/ask close in the bin —
i.e. the H4/D1 bar's own closing spread (per-trade spread cost still comes from the M5 entry
bar per R3a, this is provided for bar-level diagnostics/gate columns only).
"""
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
ANCHOR_HOUR = 17  # NY local


def _resample_ny_anchored(df, bar_hours):
    """Shared core for m5_to_h4 / m5_to_d1. `bar_hours` in {4, 24}."""
    if len(df) == 0:
        return df.iloc[0:0][["timestamp", "open", "high", "low", "close", "volume", "bid_c", "ask_c"]].copy()

    d = df.sort_values("timestamp").reset_index(drop=True)
    ts_ny = d["timestamp"].dt.tz_convert(NY)
    hour = ts_ny.dt.hour
    minute = ts_ny.dt.minute

    minutes_since_anchor = ((hour - ANCHOR_HOUR) % 24) * 60 + minute
    bar_minutes = bar_hours * 60
    bin_idx = (minutes_since_anchor // bar_minutes).astype(int)
    # NOT taken mod 24: this is a total-hours offset from ref_day_naive's midnight (can exceed
    # 24, e.g. 17+4*2=25 for an 01:00 NY bin whose ref day is "yesterday" — see below). Wrapping
    # it mod 24 was an earlier bug: it silently shifted bins back a full calendar day whenever
    # the raw offset exceeded 24 (caught by test_h4_keeps_thin_weekend_bins_mid_series_not_just_trailing).
    bin_start_hour_offset = ANCHOR_HOUR + bar_hours * bin_idx

    # All date arithmetic done NAIVE (no tz) to avoid any DST-crossing addition ambiguity;
    # the NY tz is attached fresh at the end via tz_localize (see module docstring).
    day_naive = ts_ny.dt.tz_localize(None).dt.normalize()
    ref_day_naive = day_naive - pd.to_timedelta(np.where(hour < ANCHOR_HOUR, 1, 0), unit="D")
    bin_start_naive = ref_day_naive + pd.to_timedelta(bin_start_hour_offset, unit="h")
    # ambiguous/nonexistent -> NaT (see module docstring): the ~2/year fall-back-ambiguous
    # 01:00 bin-starts are dropped rather than guessed; groupby excludes NaT keys by default.
    bin_start_ny = bin_start_naive.dt.tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
    bin_start_utc = bin_start_ny.dt.tz_convert("UTC")

    d = d.assign(_bin=bin_start_utc)
    g = d.groupby("_bin", sort=True)
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
    bin_end = agg["timestamp"] + timedelta(hours=bar_hours)
    complete = bin_end <= last_m5_ts + timedelta(minutes=5)
    agg = agg[complete].reset_index(drop=True)

    return agg[["timestamp", "open", "high", "low", "close", "volume", "bid_c", "ask_c"]]


def m5_to_h4(df):
    """Aggregate an M5-BA dataframe (columns: timestamp,open,high,low,close,bid_c,ask_c,volume,
    tz-aware UTC timestamps) into NY-17:00-anchored H4 bars (mid OHLC + summed tick volume +
    last bid_c/ask_c). Drops the trailing partial bar (R1)."""
    return _resample_ny_anchored(df, bar_hours=4)


def m5_to_d1(df):
    """Same as m5_to_h4 but daily (NY 17:00 -> next-day NY 17:00)."""
    return _resample_ny_anchored(df, bar_hours=24)
