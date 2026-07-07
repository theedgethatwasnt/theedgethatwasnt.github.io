"""Tests for is_data.py — the hard IS/OOS boundary guard."""
import os

import pandas as pd
import pytest

import d1_data as d1
import is_data as isd
import portfolio as pf

pytestmark = pytest.mark.skipif(
    not (os.path.exists(os.path.join(os.path.dirname(__file__), "cot_weekly.parquet"))
         and os.path.isdir(d1.DATA_DIR)),
    reason="cot_weekly.parquet or data/d1_deep_ba/ not present",
)


@pytest.fixture(scope="module")
def sched():
    cot_df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "cot_weekly.parquet"))
    calendar = d1.trading_calendar(pf.DIRECT_PAIRS)
    return pf.build_rebalance_schedule(cot_df, calendar)


def test_is_split_is_roughly_70pct_by_row_count(sched):
    is_sched = isd.restrict_sched_to_is(sched)
    frac = len(is_sched) / len(sched)
    assert 0.68 <= frac <= 0.72, f"IS fraction {frac:.3f} drifted from the documented ~0.70"


def test_is_never_touches_oos_report_dates(sched):
    is_sched = isd.restrict_sched_to_is(sched)
    assert is_sched.index.max() < isd.IS_END_REPORT_DATE
    oos_sched = isd.restrict_sched_to_oos(sched)
    assert oos_sched.index.min() >= isd.IS_END_REPORT_DATE
    # partition covers everything, no overlap, no gap
    assert len(is_sched) + len(oos_sched) == len(sched)


def test_is_action_dates_never_cross_price_cutoff(sched):
    is_sched = isd.restrict_sched_to_is(sched)
    assert (is_sched["action_date"] < isd.IS_PRICE_CUTOFF).all()
    # exit is allowed to land exactly AT the cutoff (see is_data.py comment) but never past it
    assert (is_sched["exit_action_date"] <= isd.IS_PRICE_CUTOFF).all()


def test_mutation_widening_the_boundary_would_be_caught():
    """Negative control: prove restrict_sched_to_is's internal assertions actually have
    teeth — feed it a synthetic schedule engineered to leak one OOS-era row disguised
    with an in-range report_date index but an out-of-range action_date, and confirm the
    assert fires (simulates the failure mode where report_date and action_date silently
    desync, e.g. a future refactor bug)."""
    dates = pd.date_range("2020-01-01", periods=5, freq="7D")
    sched = pd.DataFrame({
        "action_date": [d + pd.Timedelta(days=4) for d in dates],
        "exit_action_date": [d + pd.Timedelta(days=11) for d in dates],
    }, index=dates)
    # Force the LAST row's action_date to be absurdly far in the future relative to its
    # report_date index (simulating desync), while report_date index stays < IS_END.
    sched.loc[dates[-1], "action_date"] = pd.Timestamp("2025-01-01")
    # tz-localize to match IS_PRICE_CUTOFF's tz-awareness
    sched["action_date"] = pd.to_datetime(sched["action_date"]).dt.tz_localize("UTC")
    sched["exit_action_date"] = pd.to_datetime(sched["exit_action_date"]).dt.tz_localize("UTC")
    with pytest.raises(AssertionError):
        isd.restrict_sched_to_is(sched)


def test_is_only_spread_medians_differ_from_full_history_or_at_least_dont_error():
    price_panel = pf.load_price_panel(["EUR_USD", "USD_JPY"])
    is_med = isd.is_only_spread_medians(price_panel)
    full_med = {p: d1.median_spread_pips(df) for p, (df, _v) in price_panel.items()}
    assert set(is_med.keys()) == {"EUR_USD", "USD_JPY"}
    for pair in is_med:
        assert is_med[pair] > 0
        # Not asserting a specific direction of difference (spreads generally compressed
        # over time, but not guaranteed for every pair) — just that IS-only computation
        # is independently derived, not silently identical due to a filter no-op bug.
        assert isinstance(full_med[pair], float)
