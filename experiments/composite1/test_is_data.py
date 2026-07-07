"""test_is_data.py — HARD IS/OOS boundary guard tests for Composite 1."""
import os

import pandas as pd
import pytest

import _paths  # noqa: F401
import d1_data as d1
import is_data as isd

pytestmark = pytest.mark.skipif(
    not os.path.isdir(d1.DATA_DIR), reason="data/d1_deep_ba/ not present"
)


def test_boundary_constants_match_preregistration():
    assert isd.COT_IS_END_REPORT_DATE == pd.Timestamp("2021-02-02")
    assert isd.AXIS1_IS_ENTRY_CUTOFF == pd.Timestamp("2021-02-01", tz="UTC")
    # composite1's own entry cutoff must sit strictly inside (never past)
    # cot_positioning's own, looser IS_PRICE_CUTOFF (2021-02-07)
    assert isd.AXIS1_IS_ENTRY_CUTOFF < pd.Timestamp("2021-02-07", tz="UTC")


def test_load_pair_is_never_returns_a_row_at_or_past_ceiling():
    df = isd.load_pair_is("EUR_USD")
    assert len(df) > 1000
    assert df.index.max() < isd.DATA_LOAD_CEILING
    # and it must actually contain data near the boundary (buffer isn't accidentally huge
    # or the load accidentally truncated way earlier than intended)
    assert df.index.max() > isd.AXIS1_IS_ENTRY_CUTOFF - pd.Timedelta(days=10)


def test_restrict_cot_to_is_never_returns_a_row_at_or_past_boundary():
    cot_path = os.path.join(_paths.COT_CODE_DIR, "cot_weekly.parquet")
    if not os.path.exists(cot_path):
        pytest.skip("cot_weekly.parquet not present")
    cot_df = pd.read_parquet(cot_path)
    out = isd.restrict_cot_to_is(cot_df)
    assert len(out) > 0
    assert out["report_date"].max() < isd.COT_IS_END_REPORT_DATE
    assert len(out) < len(cot_df)


def test_assert_trade_is_is_passes_valid_and_rejects_oos():
    isd.assert_trade_is_is(pd.Timestamp("2020-06-01", tz="UTC"), pd.Timestamp("2020-06-10", tz="UTC"))
    with pytest.raises(AssertionError):
        isd.assert_trade_is_is(pd.Timestamp("2021-02-01", tz="UTC"), pd.Timestamp("2021-02-10", tz="UTC"))
    with pytest.raises(AssertionError):
        isd.assert_trade_is_is(pd.Timestamp("2021-01-15", tz="UTC"), pd.Timestamp("2021-04-01", tz="UTC"))


def test_mutation_widening_the_entry_cutoff_would_be_caught():
    """Negative control (mirrors cot_positioning/test_is_data.py's mutation test): proves
    assert_trade_is_is actually has teeth against a desynced/leaked trade."""
    leaked_entry = isd.AXIS1_IS_ENTRY_CUTOFF + pd.Timedelta(days=1)
    with pytest.raises(AssertionError):
        isd.assert_trade_is_is(leaked_entry, leaked_entry + pd.Timedelta(days=5))
