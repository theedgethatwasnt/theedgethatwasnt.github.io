"""Tests for bars.py (Workstream A2). Run on the Hetzner box per CLAUDE.md directive:
    rsync -az research/experiments/multiday_contrarian/ root@HETZNER:/root/multiday/code/
    ssh root@HETZNER 'cd /root/multiday/code && /root/venv/bin/python -m pytest test_bars.py -x -q'
"""
import numpy as np
import pandas as pd
import pytest

from bars import m5_to_h4, m5_to_d1

NY_HOURS = {17, 21, 1, 5, 9, 13}


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


# ── anchor correctness across a DST boundary (US spring-forward: 2024-03-10 02:00->03:00) ──
def test_h4_anchor_hours_are_always_ny_1721_1_5_9_13():
    df = _make_m5("2024-03-08T00:00:00Z", "2024-03-12T00:00:00Z")
    h4 = m5_to_h4(df)
    ny = h4["timestamp"].dt.tz_convert("America/New_York")
    assert set(ny.dt.hour.unique()) <= NY_HOURS
    assert (ny.dt.minute == 0).all() and (ny.dt.second == 0).all()


def test_h4_anchor_shifts_utc_hour_by_one_across_dst_spring_forward():
    """17:00 NY = 22:00 UTC in EST (before Mar 10 2024), but 21:00 UTC in EDT (after) —
    confirms the grid tracks NY wall-clock, not a fixed UTC offset."""
    df = _make_m5("2024-03-08T00:00:00Z", "2024-03-13T00:00:00Z")
    h4 = m5_to_h4(df)
    ny = h4["timestamp"].dt.tz_convert("America/New_York")
    before = h4[(ny.dt.hour == 17) & (h4["timestamp"].dt.date == pd.Timestamp("2024-03-08").date())]
    after = h4[(ny.dt.hour == 17) & (h4["timestamp"].dt.date == pd.Timestamp("2024-03-11").date())]
    assert len(before) == 1 and len(after) == 1
    assert before["timestamp"].iloc[0].hour == 22  # EST: UTC-5
    assert after["timestamp"].iloc[0].hour == 21    # EDT: UTC-4


def test_h4_survives_fall_back_dst_ambiguous_hour_without_crashing():
    """US fall-back (2024-11-03: clocks fall back 02:00->01:00 local) makes local '01:00' —
    one of our six H4 anchors — occur TWICE that night. This must not raise (must resolve via
    the documented ambiguous='NaT' drop, not crash) and must not corrupt neighboring bins."""
    df = _make_m5("2024-11-01T00:00:00Z", "2024-11-05T00:00:00Z")
    h4 = m5_to_h4(df)  # must not raise
    ny = h4["timestamp"].dt.tz_convert("America/New_York")
    assert set(ny.dt.hour.unique()) <= NY_HOURS
    # bins are still monotonically increasing (no bin assigned out of order / duplicated)
    assert h4["timestamp"].is_monotonic_increasing
    assert h4["timestamp"].is_unique


def test_h4_bins_are_exactly_4_hours_apart_in_ny_wall_clock():
    """Even though DST changes the UTC gap, consecutive NY-local bin starts must always be
    exactly the next of the six {17,21,1,5,9,13} anchors — no bin dropped/duplicated at the
    transition."""
    df = _make_m5("2024-03-08T00:00:00Z", "2024-03-12T00:00:00Z")
    h4 = m5_to_h4(df)
    ny_hours = h4["timestamp"].dt.tz_convert("America/New_York").dt.hour.tolist()
    order = [17, 21, 1, 5, 9, 13]
    start_idx = order.index(ny_hours[0])
    expected = [order[(start_idx + i) % 6] for i in range(len(ny_hours))]
    assert ny_hours == expected


# NOTE: in June (EDT, UTC-4) NY 17:00 == 21:00 UTC — bin-aligned test windows below start /
# end on 21:00 UTC so a "24h = 6 bins" window lands exactly on bin edges. (Using UTC midnight
# would NOT be bin-aligned and was an earlier test-construction bug caught by these tests
# failing against a correct implementation — see git history / the report.)

# ── R1: closed bars only — trailing partial bar dropped ─────────────────────
def test_h4_drops_trailing_partial_bar():
    # Exactly 6 full H4 bins (24h = 288 M5 bars), plus 5 extra M5 bars spilling into a 7th
    # (incomplete) bin that must NOT appear in the output.
    full = _make_m5("2024-06-02T21:00:00Z", "2024-06-03T21:00:00Z")  # 288 bars, 6 bins
    partial = _make_m5("2024-06-03T21:00:00Z", "2024-06-03T21:25:00Z")  # 5 more bars
    df = pd.concat([full, partial], ignore_index=True)
    h4 = m5_to_h4(df)
    assert len(h4) == 6
    assert df["volume"].sum() == 293          # 288 + 5 fed in
    assert h4["volume"].sum() == 288           # the partial 7th bin's 5 bars excluded


def test_h4_keeps_thin_weekend_bins_mid_series_not_just_trailing():
    """A bin that is merely THIN (fewer M5 bars, e.g. weekend market closure) but which the
    data has already moved past (i.e. it is not the trailing edge) must be KEPT — R1 cares
    about whether the bar has closed, not about bar completeness/liquidity."""
    thin_start = _make_m5("2024-06-02T21:00:00Z", "2024-06-03T01:00:00Z")  # 1 full bin (48 bars)
    # simulate a weekend gap: only 6 bars (30 min) inside the *next* bin, then normal data resumes
    gap_bin = _make_m5("2024-06-03T01:00:00Z", "2024-06-03T01:30:00Z")     # 6 bars only (thin)
    resume = _make_m5("2024-06-03T05:00:00Z", "2024-06-03T21:00:00Z")     # rest of the day, full bins
    df = pd.concat([thin_start, gap_bin, resume], ignore_index=True)
    h4 = m5_to_h4(df)
    assert len(h4) == 6                     # all 6 bins for the day present, incl. the thin one
    assert (h4["volume"] > 0).all()
    thin_row = h4.iloc[1]
    assert thin_row["volume"] == 6           # thin bin kept with its true (low) bar count


# ── volume conservation ───────────────────────────────────────────────────────
def test_h4_volume_conservation_exact_multiple_of_bins():
    df = _make_m5("2024-06-02T21:00:00Z", "2024-06-04T21:00:00Z", volume=3)  # 2 days, vol=3/bar
    h4 = m5_to_h4(df)
    assert len(h4) == 12  # 2 days * 6 bins
    assert h4["volume"].sum() == df["volume"].sum()
    assert (h4["volume"] == 48 * 3).all()  # 48 M5 bars/bin * vol 3 each


def test_d1_volume_conservation_and_anchor():
    df = _make_m5("2024-06-02T21:00:00Z", "2024-06-05T21:00:00Z", volume=2)  # 3 full days
    d1 = m5_to_d1(df)
    assert len(d1) == 3
    ny = d1["timestamp"].dt.tz_convert("America/New_York")
    assert (ny.dt.hour == 17).all() and (ny.dt.minute == 0).all()
    assert d1["volume"].sum() == df["volume"].sum()
    assert (d1["volume"] == 288 * 2).all()  # 288 M5 bars/day * vol 2 each


def test_d1_drops_trailing_partial_bar():
    full = _make_m5("2024-06-02T21:00:00Z", "2024-06-03T21:00:00Z")   # 1 full day
    partial = _make_m5("2024-06-03T21:00:00Z", "2024-06-04T03:00:00Z")  # 6h into day 2 only
    df = pd.concat([full, partial], ignore_index=True)
    d1 = m5_to_d1(df)
    assert len(d1) == 1
    assert d1["volume"].iloc[0] == 288


# ── OHLC correctness (sanity, not explicitly required but cheap + catches swapped agg cols) ──
def test_h4_ohlc_first_last_max_min():
    # 2024-06-03 21:00 UTC == 2024-06-03 17:00 NY (EDT, UTC-4) — exactly one H4 bin.
    df = _make_m5("2024-06-03T21:00:00Z", "2024-06-04T01:00:00Z")
    h4 = m5_to_h4(df)
    assert len(h4) == 1
    row = h4.iloc[0]
    assert row["open"] == pytest.approx(df["open"].iloc[0])
    assert row["close"] == pytest.approx(df["close"].iloc[-1])
    assert row["high"] == pytest.approx(df["high"].max())
    assert row["low"] == pytest.approx(df["low"].min())
    assert row["bid_c"] == pytest.approx(df["bid_c"].iloc[-1])
    assert row["ask_c"] == pytest.approx(df["ask_c"].iloc[-1])
