"""Tests for d1_data.py."""
import os

import numpy as np
import pandas as pd
import pytest

import d1_data as d1

DIRECT_PAIRS = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"]


def _have_data():
    return os.path.isdir(d1.DATA_DIR) and len(os.listdir(d1.DATA_DIR)) > 0


pytestmark = pytest.mark.skipif(not _have_data(), reason="data/d1_deep_ba/ not fetched")


def test_load_pair_eur_usd_basic_shape():
    df = d1.load_pair("EUR_USD")
    assert len(df) > 6000
    assert df.index.is_monotonic_increasing
    assert df.index.duplicated().sum() == 0
    assert df.index.max() <= pd.Timestamp("2026-05-21", tz=df.index.tz)  # ceiling respected


def test_all_seven_direct_legs_loadable():
    for pair in DIRECT_PAIRS:
        df = d1.load_pair(pair)
        assert len(df) > 5000, pair
        assert not df["bid_c"].isna().any()
        assert not df["ask_c"].isna().any()
        assert (df["ask_c"] >= df["bid_c"]).all(), f"{pair}: crossed bid/ask found"


def test_median_spread_positive_and_reasonable():
    for pair in DIRECT_PAIRS:
        df = d1.load_pair(pair)
        med = d1.median_spread_pips(df)
        assert 0 < med < 20, f"{pair}: median spread {med} out of sane bounds"


def test_trading_calendar_union_covers_all_pairs():
    cal = d1.trading_calendar(DIRECT_PAIRS)
    for pair in DIRECT_PAIRS:
        df = d1.load_pair(pair)
        assert df.index.isin(cal).all()


def test_realized_vol_uses_only_past_returns_synthetic():
    """Synthetic price series: flat for 100 days, then one huge jump on day 101. The
    vol computed AT day 101 (shift(1) applied) must NOT reflect that day's own jump —
    only day 102's vol should show the jump."""
    dates = pd.date_range("2020-01-01", periods=110, freq="D")
    price = np.concatenate([np.full(100, 1.1000), np.full(10, 1.1000)])
    price[100] = 1.2000  # a 1000-pip jump on day index 100 (the 101st day)
    df = pd.DataFrame({"close": price}, index=dates)
    pip = 0.0001
    df["daily_ret_pips"] = df["close"].diff() / pip

    vol = d1.realized_vol_pips(df, window=20, min_periods=5)
    # At the jump date itself, vol must still reflect ONLY prior (zero-return) history.
    assert vol.iloc[100] == pytest.approx(0.0, abs=1e-9)
    # The day AFTER the jump, vol must be strictly positive (jump has entered the window).
    assert vol.iloc[101] > 0


def test_realized_vol_nan_before_min_periods():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    rng = np.random.default_rng(0)
    price = 1.10 + np.cumsum(rng.normal(0, 0.0005, 30))
    df = pd.DataFrame({"close": price}, index=dates)
    df["daily_ret_pips"] = df["close"].diff() / 0.0001
    vol = d1.realized_vol_pips(df, window=20, min_periods=10)
    assert vol.iloc[:10].isna().all()
