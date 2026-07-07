"""Tests for currency_index.py. Run on the Hetzner box:
    rsync -az research/experiments/fx_factors/ root@HETZNER:/root/work/code_factors/fx_factors/
    ssh root@HETZNER 'cd /root/work/code_factors/fx_factors && /root/venv/bin/python -m pytest test_currency_index.py -x -q'
"""
import numpy as np
import pandas as pd
import pytest

import currency_index
from currency_index import REQUIRED_PAIRS_FOR_INDEX, build_panels, spx_gate_signal
from is_data import CURRENCIES


def _make_d1(dates, closes, pip=0.0001):
    n = len(dates)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "timestamp": pd.DatetimeIndex(dates),
        "open": closes,
        "high": closes + pip,
        "low": closes - pip,
        "close": closes,
        "bid_c": closes - pip / 2,
        "ask_c": closes + pip / 2,
        "volume": 100,
    })


def _panel_fixture():
    dates = pd.date_range("2023-01-02", periods=5, freq="D", tz="UTC")
    pair_d1 = {
        "EUR_USD": _make_d1(dates, [1.10, 1.10, 1.10, 1.10, 1.10]),
        "GBP_USD": _make_d1(dates, [1.25, 1.25, 1.25, 1.25, 1.25]),
        "AUD_USD": _make_d1(dates, [0.65, 0.65, 0.65, 0.65, 0.65]),
        "NZD_USD": _make_d1(dates, [0.60, 0.60, 0.60, 0.60, 0.60]),
        "USD_JPY": _make_d1(dates, [150.0, 150.0, 150.0, 150.0, 150.0]),
        "CAD_JPY": _make_d1(dates, [110.0, 110.0, 110.0, 110.0, 110.0]),
        "CHF_JPY": _make_d1(dates, [170.0, 170.0, 170.0, 170.0, 170.0]),
    }
    return dates, pair_d1


def test_required_pairs_matches_expression_pairs():
    expr_pairs = sorted({p for p, _ in currency_index.EXPRESSION.values()})
    assert expr_pairs == sorted(REQUIRED_PAIRS_FOR_INDEX)


def test_usd_column_is_always_one():
    _, pair_d1 = _panel_fixture()
    usd_per_x, xusd = build_panels(pair_d1)
    assert (usd_per_x["USD"] == 1.0).all()
    assert (xusd["USD"] == 1.0).all()


def test_direct_usd_pairs_pass_through_unchanged():
    _, pair_d1 = _panel_fixture()
    usd_per_x, _ = build_panels(pair_d1)
    assert np.allclose(usd_per_x["EUR"], 1.10)
    assert np.allclose(usd_per_x["GBP"], 1.25)
    assert np.allclose(usd_per_x["AUD"], 0.65)
    assert np.allclose(usd_per_x["NZD"], 0.60)


def test_jpy_is_inverse_of_usd_jpy():
    _, pair_d1 = _panel_fixture()
    usd_per_x, _ = build_panels(pair_d1)
    assert np.allclose(usd_per_x["JPY"], 1.0 / 150.0)


def test_cad_chf_triangulation_through_jpy():
    """usd_per_x[CAD] = CAD_JPY / USD_JPY (JPY/CAD divided by JPY/USD = USD/CAD)."""
    _, pair_d1 = _panel_fixture()
    usd_per_x, _ = build_panels(pair_d1)
    assert np.allclose(usd_per_x["CAD"], 110.0 / 150.0)
    assert np.allclose(usd_per_x["CHF"], 170.0 / 150.0)


def test_xusd_is_reciprocal_of_usd_per_x():
    _, pair_d1 = _panel_fixture()
    usd_per_x, xusd = build_panels(pair_d1)
    for ccy in CURRENCIES:
        assert np.allclose(xusd[ccy].values, 1.0 / usd_per_x[ccy].values)


def test_build_panels_uses_intersection_of_dates_not_union():
    dates, pair_d1 = _panel_fixture()
    # Drop the last date from one pair only -> intersection should shrink by 1.
    pair_d1["EUR_USD"] = pair_d1["EUR_USD"].iloc[:-1].reset_index(drop=True)
    usd_per_x, _ = build_panels(pair_d1)
    assert len(usd_per_x) == len(dates) - 1


def test_spx_gate_true_above_sma_false_below():
    n = 260
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    # Flat at 100 for warmup, then a sharp drop below the trailing SMA(200) for the tail.
    closes = np.full(n, 100.0)
    closes[-5:] = 50.0
    spx_df = pd.DataFrame({"time": dates, "close": closes})

    risk_on_date = dates[220]   # still flat at 100 == SMA(200) -> gate open (>=)
    risk_off_date = dates[-1]   # dropped to 50, well below SMA(200) -> gate closed

    out = spx_gate_signal(spx_df, [risk_on_date, risk_off_date])
    assert out[risk_on_date] == True  # noqa: E712
    assert out[risk_off_date] == False  # noqa: E712


def test_spx_gate_is_causal_no_lookahead():
    """A future crash must not affect an earlier gate reading."""
    n = 260
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    closes = np.full(n, 100.0)
    closes[-1] = 1.0  # crash on the very last bar only
    spx_df = pd.DataFrame({"time": dates, "close": closes})

    early_date = dates[210]
    out = spx_gate_signal(spx_df, [early_date])
    assert out[early_date] == True  # noqa: E712
