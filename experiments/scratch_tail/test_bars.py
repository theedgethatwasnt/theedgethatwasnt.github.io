"""Tests for bars.py (broker-grid H1/M30 aggregation). Run on the Hetzner box per CLAUDE.md:
    rsync -az research/experiments/scratch_tail/ root@HETZNER:/root/work/code/scratch_tail/
    ssh root@HETZNER '/root/venv/bin/python -m pytest /root/work/code/scratch_tail/test_bars.py -x -q'
"""
import numpy as np
import pandas as pd
import pytest

from bars import m5_to_h1, m5_to_m30


def _make_m5(start, end, freq="5min", volume=1):
    """Continuous synthetic M5-BA dataframe (mid OHLC + bid_c/ask_c + volume) over [start, end)."""
    ts = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")
    n = len(ts)
    price = 1.10000 + np.arange(n) * 1e-5
    return pd.DataFrame({
        "timestamp": ts,
        "open": price,
        "high": price + 2e-5,
        "low": price - 2e-5,
        "close": price + 1e-5,
        "bid_c": price + 0.99e-5,
        "ask_c": price + 1.01e-5,
        "volume": volume,
    })


# ── anchor correctness: pure UTC, no DST sensitivity ──────────────────────────
def test_h1_anchors_are_top_of_hour_utc():
    df = _make_m5("2024-03-08T00:00:00Z", "2024-03-10T00:00:00Z")
    h1 = m5_to_h1(df)
    assert (h1["timestamp"].dt.minute == 0).all() and (h1["timestamp"].dt.second == 0).all()


def test_m30_anchors_are_top_of_half_hour_utc():
    df = _make_m5("2024-03-08T00:00:00Z", "2024-03-10T00:00:00Z")
    m30 = m5_to_m30(df)
    assert (m30["timestamp"].dt.minute.isin([0, 30])).all()
    assert (m30["timestamp"].dt.second == 0).all()


def test_h1_boundary_identical_across_us_dst_transition():
    """Unlike H4/D1 (NY-anchored), the H1/M30 broker grid must NOT shift across a US DST
    transition — it's plain UTC top-of-hour, insensitive to any local-time convention."""
    df = _make_m5("2024-03-08T00:00:00Z", "2024-03-13T00:00:00Z")
    h1 = m5_to_h1(df)
    # every bin is exactly 1h apart in UTC, no gap/duplicate around the Mar-10 US spring-forward
    diffs = h1["timestamp"].diff().dropna().unique()
    assert list(diffs) == [pd.Timedelta(hours=1)]


# ── R1: closed bars only — trailing partial bar dropped ─────────────────────
def test_h1_drops_trailing_partial_bar():
    full = _make_m5("2024-06-03T00:00:00Z", "2024-06-04T00:00:00Z")     # 288 bars, 24 bins
    partial = _make_m5("2024-06-04T00:00:00Z", "2024-06-04T00:25:00Z")  # 5 more bars (partial 25th)
    df = pd.concat([full, partial], ignore_index=True)
    h1 = m5_to_h1(df)
    assert len(h1) == 24
    assert df["volume"].sum() == 293
    assert h1["volume"].sum() == 288


def test_m30_drops_trailing_partial_bar():
    full = _make_m5("2024-06-03T00:00:00Z", "2024-06-03T06:00:00Z")     # 72 bars, 12 bins
    partial = _make_m5("2024-06-03T06:00:00Z", "2024-06-03T06:10:00Z")  # 2 more bars (partial 13th)
    df = pd.concat([full, partial], ignore_index=True)
    m30 = m5_to_m30(df)
    assert len(m30) == 12
    assert m30["volume"].sum() == 72


def test_h1_keeps_thin_weekend_bins_mid_series_not_just_trailing():
    """A bin that is merely THIN (fewer M5 bars, e.g. weekend market closure) but which the
    data has already moved past (not the trailing edge) must be KEPT — R1 cares about whether
    the bar has closed, not about bar completeness/liquidity."""
    full_start = _make_m5("2024-06-03T00:00:00Z", "2024-06-03T01:00:00Z")  # 1 full bin (12 bars)
    gap_bin = _make_m5("2024-06-03T01:00:00Z", "2024-06-03T01:15:00Z")     # 3 bars only (thin)
    resume = _make_m5("2024-06-03T02:00:00Z", "2024-06-03T05:00:00Z")     # rest, full bins
    df = pd.concat([full_start, gap_bin, resume], ignore_index=True)
    h1 = m5_to_h1(df)
    assert len(h1) == 5                     # bins for 00,01,02,03,04 all present, incl. thin 01
    assert (h1["volume"] > 0).all()
    thin_row = h1.iloc[1]
    assert thin_row["volume"] == 3           # thin bin kept with its true (low) bar count


# ── volume conservation ───────────────────────────────────────────────────────
def test_h1_volume_conservation_exact_multiple_of_bins():
    df = _make_m5("2024-06-02T00:00:00Z", "2024-06-04T00:00:00Z", volume=3)  # 2 days, vol=3/bar
    h1 = m5_to_h1(df)
    assert len(h1) == 48  # 2 days * 24 bins
    assert h1["volume"].sum() == df["volume"].sum()
    assert (h1["volume"] == 12 * 3).all()  # 12 M5 bars/bin * vol 3 each


def test_m30_volume_conservation_and_bin_count():
    df = _make_m5("2024-06-02T00:00:00Z", "2024-06-03T00:00:00Z", volume=2)  # 1 full day
    m30 = m5_to_m30(df)
    assert len(m30) == 48  # 1 day * 48 half-hour bins
    assert m30["volume"].sum() == df["volume"].sum()
    assert (m30["volume"] == 6 * 2).all()  # 6 M5 bars/bin * vol 2 each


# ── OHLC correctness (sanity, catches swapped agg cols) ──────────────────────
def test_h1_ohlc_first_last_max_min():
    df = _make_m5("2024-06-03T05:00:00Z", "2024-06-03T06:00:00Z")
    h1 = m5_to_h1(df)
    assert len(h1) == 1
    row = h1.iloc[0]
    assert row["open"] == pytest.approx(df["open"].iloc[0])
    assert row["close"] == pytest.approx(df["close"].iloc[-1])
    assert row["high"] == pytest.approx(df["high"].max())
    assert row["low"] == pytest.approx(df["low"].min())
    assert row["bid_c"] == pytest.approx(df["bid_c"].iloc[-1])
    assert row["ask_c"] == pytest.approx(df["ask_c"].iloc[-1])


def test_m30_ohlc_first_last_max_min():
    df = _make_m5("2024-06-03T05:00:00Z", "2024-06-03T05:30:00Z")
    m30 = m5_to_m30(df)
    assert len(m30) == 1
    row = m30.iloc[0]
    assert row["open"] == pytest.approx(df["open"].iloc[0])
    assert row["close"] == pytest.approx(df["close"].iloc[-1])
    assert row["high"] == pytest.approx(df["high"].max())
    assert row["low"] == pytest.approx(df["low"].min())


def test_h1_bins_are_monotonic_and_unique():
    df = _make_m5("2024-06-02T00:00:00Z", "2024-06-09T00:00:00Z")  # 1 week
    h1 = m5_to_h1(df)
    assert h1["timestamp"].is_monotonic_increasing
    assert h1["timestamp"].is_unique
    assert len(h1) == 24 * 7
