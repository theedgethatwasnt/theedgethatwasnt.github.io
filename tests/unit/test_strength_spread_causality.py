"""Tripwire for the 4-17 StrengthSpread lookahead bug.

Bug history
-----------
OANDA H1 bar timestamp T represents the interval [T, T+1h]; the bar's close
price is only knowable at time T+1h. The original
`precompute_strength_spread.py` used ``merge_asof(m5, ss_h1, direction='backward')``
without shifting the H1 timestamps, so every M5 bar in the window [T, T+1h]
received ss_h1[T] — a value that encodes up to 55 minutes of future data.

A PINN-CMA experiment on 2026-04-17 hit +203 pips/day OOS with the ``inputs``
mode, which is the only mode that consumes this feature. The bug was caught
pre-deploy because the statistical gates (MCMC Phase 1) flagged it as
implausible, triggering an RCA. Fix: shift ss_h1 timestamps by +1h before
merging. See ``research/experiments/cma_5in/precompute_strength_spread.py``.

This test is the tripwire. It runs against a synthetic H1 panel + M5 panel
so it does not depend on on-disk data, and asserts:
  1. The fixed code path never returns an H1 spread value whose source bar
     closes AFTER the requesting M5 timestamp.
  2. The old (unshifted) code path DOES violate that invariant, to prove this
     test would have caught the bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _make_synthetic_panel():
    """Make an H1 series where each bar's value = the bar's close time, in
    units of hours since epoch. Makes "which H1 bar did this M5 come from"
    trivially traceable."""
    # 100 hours of H1 data starting at a known timestamp
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    h1_ts = pd.date_range(start=start, periods=100, freq="1h", tz="UTC")
    # Make the ss_h1 value = an integer that encodes the bar's CLOSE time
    # (bar @ T closes at T+1h). That way a leak is detectable: if M5 @ t gets
    # a value > t (in hour units), it's looking ahead.
    close_times_hours = np.arange(1, 101)  # bar 0 closes at hr 1, bar 99 at hr 100
    ss_h1 = pd.DataFrame({
        "timestamp": h1_ts,
        "ss_h1": close_times_hours.astype(float),
    })
    # M5 timestamps covering the same range
    m5_ts = pd.date_range(start=start, periods=100 * 12, freq="5min", tz="UTC")
    m5 = pd.DataFrame({"timestamp": m5_ts})
    return ss_h1, m5, start


def _apply_fixed_merge(ss_h1: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    """The FIXED code path: shift H1 timestamps by +1h before merge_asof."""
    ss_shifted = ss_h1.copy()
    ss_shifted["timestamp"] = ss_shifted["timestamp"] + pd.Timedelta(hours=1)
    return pd.merge_asof(
        m5.sort_values("timestamp"),
        ss_shifted.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )


def _apply_buggy_merge(ss_h1: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    """The ORIGINAL (buggy) code path: no shift."""
    return pd.merge_asof(
        m5.sort_values("timestamp"),
        ss_h1.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )


def _hours_since(ts: pd.Timestamp, start: pd.Timestamp) -> float:
    return (ts - start).total_seconds() / 3600.0


def test_fixed_merge_has_no_lookahead():
    ss_h1, m5, start = _make_synthetic_panel()
    merged = _apply_fixed_merge(ss_h1, m5)
    # For every non-NaN row, the served value (= source H1 bar's close time in
    # hours) must be <= the M5 timestamp's hours-since-start.
    merged = merged.dropna().reset_index(drop=True)
    m5_hours = np.array([_hours_since(t, start) for t in merged["timestamp"]])
    violations = (merged["ss_h1"].values > m5_hours + 1e-9).sum()
    assert violations == 0, (
        f"fixed merge leaks: {violations}/{len(merged)} rows contain an ss_h1 "
        f"value whose source H1 bar closes AFTER the requesting M5 ts"
    )


def test_buggy_merge_is_caught_by_the_same_invariant():
    """Sanity: if someone removes the shift, this test fails."""
    ss_h1, m5, start = _make_synthetic_panel()
    merged = _apply_buggy_merge(ss_h1, m5)
    merged = merged.dropna().reset_index(drop=True)
    m5_hours = np.array([_hours_since(t, start) for t in merged["timestamp"]])
    violations = (merged["ss_h1"].values > m5_hours + 1e-9).sum()
    # We *expect* most rows to violate (every M5 except the exact H1 boundary)
    assert violations > 0.5 * len(merged), (
        f"expected the unshifted merge to leak >50% of rows, got "
        f"{violations}/{len(merged)}. Has the H1-timestamp convention "
        f"changed?"
    )


def test_edge_case_m5_right_at_h1_close_gets_correct_value():
    """M5 bar at ts = H1 bar T's close time (T+1h) SHOULD see ss_h1[T]."""
    ss_h1, m5, start = _make_synthetic_panel()
    merged = _apply_fixed_merge(ss_h1, m5)
    # M5 @ start + 1h should get the very first H1 bar's ss value (= 1.0)
    match = merged[merged["timestamp"] == start + pd.Timedelta(hours=1)]
    assert not match.empty
    assert abs(match.iloc[0]["ss_h1"] - 1.0) < 1e-9, (
        f"expected ss=1.0 (first H1 bar's close time) at start+1h, got "
        f"{match.iloc[0]['ss_h1']}"
    )


def test_edge_case_m5_one_bar_before_close_sees_prior_h1():
    """M5 @ 00:55 (5 min before first H1 closes) should see NaN (no H1 closed
    yet). This ensures we're not still serving ss_h1[T=0] to bars inside
    [0, 1h]."""
    ss_h1, m5, start = _make_synthetic_panel()
    merged = _apply_fixed_merge(ss_h1, m5)
    match = merged[merged["timestamp"] == start + pd.Timedelta(minutes=55)]
    assert not match.empty
    assert pd.isna(match.iloc[0]["ss_h1"]), (
        f"expected NaN (no H1 bar has closed yet at start+55min), got "
        f"{match.iloc[0]['ss_h1']}"
    )
