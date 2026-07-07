"""Tests for cot_signal.py — z-score, ranking, currency->pair expression."""
import numpy as np
import pandas as pd
import pytest

import cot_signal as sig


def _synthetic_cot(currency, dates_or_n, frac_series):
    """`dates_or_n`: either an explicit DatetimeIndex/array of report_dates, or an int
    week-count (in which case a fresh Tuesday-weekly range from 2005-01-04 is generated —
    only safe when the caller does NOT need a specific date offset, e.g. a single-currency
    test with no cross-currency alignment requirement)."""
    if isinstance(dates_or_n, (int, np.integer)):
        dates = pd.date_range("2005-01-04", periods=dates_or_n, freq="7D")
    else:
        dates = pd.DatetimeIndex(dates_or_n)
    return pd.DataFrame({
        "currency": currency,
        "report_date": dates,
        "net_noncomm_frac_oi": frac_series,
    })


def test_zscore_nan_before_full_window():
    n = 300
    rng = np.random.default_rng(0)
    frac = rng.normal(0, 0.1, n)
    df = _synthetic_cot("EUR", n, frac)
    z = sig.compute_zscore_panel(df)
    assert z["z"].iloc[: sig.Z_MIN_PERIODS - 1].isna().all()
    assert z["z"].iloc[sig.Z_MIN_PERIODS - 1 :].notna().all()


def test_zscore_matches_manual_rolling_calc():
    n = 250
    rng = np.random.default_rng(1)
    frac = rng.normal(0, 0.1, n)
    df = _synthetic_cot("EUR", n, frac)
    z = sig.compute_zscore_panel(df).sort_values("report_date").reset_index(drop=True)

    # Manual check at the last row: z = (x - mean(last 156)) / std(last 156, ddof=1)
    window = frac[-sig.Z_WINDOW:]
    expected = (frac[-1] - window.mean()) / window.std(ddof=1)
    assert z["z"].iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_zscore_extreme_value_gets_large_z():
    n = 200
    frac = np.zeros(n)
    frac[-1] = 5.0  # wildly crowded relative to a flat history
    df = _synthetic_cot("EUR", n, frac)
    z = sig.compute_zscore_panel(df)
    assert z["z"].iloc[-1] > 5.0  # z-score should be large-positive (many std devs out)


def test_build_rebalance_panel_requires_all_currencies():
    dates = pd.date_range("2005-01-04", periods=200, freq="7D")
    frames = []
    for i, ccy in enumerate(sig.CURRENCIES):
        # NZD (last currency) starts 10 weeks late -> its first 10 weeks are simply absent
        n_start = 10 if ccy == "NZD" else 0
        d = dates[n_start:]
        frac = np.random.default_rng(i).normal(0, 0.1, len(d))
        frames.append(_synthetic_cot(ccy, d, frac))
    cot_df = pd.concat(frames, ignore_index=True)
    z = sig.compute_zscore_panel(cot_df)
    panel = sig.build_rebalance_panel(z)
    assert set(panel.columns) == set(sig.CURRENCIES)
    assert panel.isna().sum().sum() == 0
    # First valid week must be >= the 156th week of the LATEST-starting currency (NZD)
    assert panel.index.min() >= dates[10 + sig.Z_MIN_PERIODS - 1]


def test_select_legs_top_bottom_no_overlap():
    z_row = pd.Series({"EUR": 2.5, "JPY": -1.8, "GBP": 0.3, "CHF": -2.9, "AUD": 1.1, "CAD": -0.2, "NZD": 3.2})
    top, bottom = sig.select_legs(z_row)
    # sorted desc: NZD(3.2) EUR(2.5) AUD(1.1) GBP(0.3) CAD(-0.2) JPY(-1.8) CHF(-2.9)
    assert top == ["NZD", "EUR"]      # highest z first
    assert bottom == ["JPY", "CHF"]   # tail of the desc-sorted list = 2 lowest z
    assert set(top) & set(bottom) == set()


def test_select_legs_bottom_is_most_crowded_short_first_by_value():
    z_row = pd.Series({"A": 0.0, "B": -3.0, "C": -1.0, "D": 1.0, "E": 2.0, "F": -0.5, "G": 0.2})
    top, bottom = sig.select_legs(z_row)
    assert top == ["E", "D"]
    # bottom = 2 lowest z: B(-3.0), C(-1.0) — order is ranked-ascending tail, i.e. [C, B] from
    # the descending-sorted index tail: sorted desc = [E,D,A,G,F,C,B], last 2 = [C,B]
    assert bottom == ["C", "B"]


def test_legs_for_arm_contrarian_and_momentum_are_opposite_signs():
    top, bottom = ["EUR", "NZD"], ["CHF", "CAD"]
    contrarian = dict(sig.legs_for_arm(top, bottom, "contrarian"))
    momentum = dict(sig.legs_for_arm(top, bottom, "momentum"))
    assert set(contrarian.keys()) == set(momentum.keys()) == {"EUR", "NZD", "CHF", "CAD"}
    for ccy in contrarian:
        assert contrarian[ccy] == -momentum[ccy]
    # Top (crowded long) -> contrarian SHORTS it (view_direction = -1)
    assert contrarian["EUR"] == -1 and contrarian["NZD"] == -1
    # Bottom (crowded short) -> contrarian LONGS it (view_direction = +1)
    assert contrarian["CHF"] == +1 and contrarian["CAD"] == +1


def test_legs_for_arm_unknown_raises():
    with pytest.raises(ValueError):
        sig.legs_for_arm(["EUR"], ["JPY"], "bogus")


@pytest.mark.parametrize("ccy,expected_pair,expected_sign", [
    ("EUR", "EUR_USD", +1),
    ("GBP", "GBP_USD", +1),
    ("AUD", "AUD_USD", +1),
    ("NZD", "NZD_USD", +1),
    ("JPY", "USD_JPY", -1),
    ("CHF", "USD_CHF", -1),
    ("CAD", "USD_CAD", -1),
])
def test_pair_direction_long_currency(ccy, expected_pair, expected_sign):
    """Going LONG each currency (view_direction=+1) must map to the documented pair/sign:
    base currencies (EUR/GBP/AUD/NZD) -> +1 on their USD pair; quote currencies
    (JPY/CHF/CAD, since USD is the base in USD_JPY/USD_CHF/USD_CAD) -> -1 (long JPY means
    SHORT USD_JPY, since a falling USD_JPY = JPY strengthening)."""
    pair, direction = sig.pair_direction(ccy, +1)
    assert pair == expected_pair
    assert direction == expected_sign


def test_pair_direction_short_currency_flips_sign():
    for ccy in sig.CURRENCIES:
        pair_long, dir_long = sig.pair_direction(ccy, +1)
        pair_short, dir_short = sig.pair_direction(ccy, -1)
        assert pair_long == pair_short
        assert dir_long == -dir_short
