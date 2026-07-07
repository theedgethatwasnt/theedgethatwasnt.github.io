"""Tests for factors.py. Run on the Hetzner box (see test_currency_index.py header)."""
import numpy as np
import pandas as pd
import pytest

from currency_index import NON_USD_CURRENCIES
from factors import composite_score, momentum_score, rank_select, value_score
from is_data import CURRENCIES


# ── rank_select ───────────────────────────────────────────────────────────────
def test_rank_select_basic_top3_bottom3():
    scores = pd.Series(
        {"USD": 0.0, "EUR": 5.0, "GBP": 4.0, "JPY": -5.0, "AUD": 3.0, "NZD": -4.0, "CAD": 0.5, "CHF": -0.5}
    )
    d = rank_select(scores)
    longs = {c for c, v in d.items() if v == 1}
    shorts = {c for c, v in d.items() if v == -1}
    assert longs == {"EUR", "GBP", "AUD"}
    assert shorts == {"JPY", "NZD", "CHF"}
    assert d["CAD"] == 0
    assert set(d.keys()) == set(NON_USD_CURRENCIES)


def test_rank_select_usd_score_never_changes_selection():
    """Removing USD from an already-ranked list never changes the relative order of what
    remains (module docstring claim) -- selection must be identical regardless of USD's own
    score value."""
    base = {"EUR": 5.0, "GBP": 4.0, "JPY": -5.0, "AUD": 3.0, "NZD": -4.0, "CAD": 0.5, "CHF": -0.5}
    for usd_score in (-100.0, -1.0, 0.0, 1.0, 100.0):
        scores = pd.Series({**base, "USD": usd_score})
        d = rank_select(scores)
        assert {c for c, v in d.items() if v == 1} == {"EUR", "GBP", "AUD"}
        assert {c for c, v in d.items() if v == -1} == {"JPY", "NZD", "CHF"}


def test_rank_select_shrinks_book_when_nans_present():
    scores = pd.Series(
        {"USD": 0.0, "EUR": 5.0, "GBP": 4.0, "JPY": np.nan, "AUD": np.nan, "NZD": -4.0, "CAD": np.nan, "CHF": -0.5}
    )
    # non-USD non-NaN candidates: EUR, GBP, NZD, CHF (n=4) -> k = 4//2 = 2
    d = rank_select(scores)
    longs = {c for c, v in d.items() if v == 1}
    shorts = {c for c, v in d.items() if v == -1}
    assert longs == {"EUR", "GBP"}
    assert shorts == {"NZD", "CHF"}
    assert d["JPY"] == 0 and d["AUD"] == 0 and d["CAD"] == 0


def test_rank_select_all_nan_is_fully_flat():
    scores = pd.Series({c: np.nan for c in CURRENCIES})
    d = rank_select(scores)
    assert all(v == 0 for v in d.values())


# ── momentum_score ───────────────────────────────────────────────────────────
def test_momentum_score_matches_hand_calc():
    dates = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")
    usd_per_x = pd.DataFrame(index=dates)
    usd_per_x["USD"] = 1.0
    for ccy in ("EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF"):
        usd_per_x[ccy] = 1.0  # flat baseline
    rebal = dates[380]
    # Bump EUR up at t-6mo so its 12-1 momentum should be clearly positive.
    usd_per_x.loc[usd_per_x.index <= (rebal - pd.DateOffset(months=6)), "EUR"] = 1.0
    usd_per_x.loc[usd_per_x.index > (rebal - pd.DateOffset(months=6)), "EUR"] = 1.10

    s = momentum_score(usd_per_x, rebal)
    assert s["EUR"] > 0
    assert np.isclose(s["USD"], 0.0)


def test_momentum_score_excludes_most_recent_month():
    """12-1 momentum: a price jump strictly inside the most recent month must NOT move the
    score (it's excluded by the skip-month convention)."""
    dates = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")
    usd_per_x = pd.DataFrame(index=dates)
    usd_per_x["USD"] = 1.0
    for ccy in ("EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF"):
        usd_per_x[ccy] = 1.0
    rebal = dates[380]
    s_before = momentum_score(usd_per_x, rebal)

    usd_per_x2 = usd_per_x.copy()
    usd_per_x2.loc[usd_per_x2.index > (rebal - pd.DateOffset(days=10)), "EUR"] = 5.0
    s_after = momentum_score(usd_per_x2, rebal)
    assert np.isclose(s_before["EUR"], s_after["EUR"])


def test_momentum_score_nan_when_insufficient_history():
    dates = pd.date_range("2023-06-01", periods=30, freq="D", tz="UTC")  # < 12mo of history
    usd_per_x = pd.DataFrame(index=dates)
    usd_per_x["USD"] = 1.0
    usd_per_x["EUR"] = 1.10
    for ccy in ("GBP", "AUD", "NZD", "JPY", "CAD", "CHF"):
        usd_per_x[ccy] = 1.0
    rebal = dates[-1]
    s = momentum_score(usd_per_x, rebal)
    assert np.isnan(s["EUR"])


# ── value_score ──────────────────────────────────────────────────────────────
def test_value_score_positive_when_market_cheaper_than_ppp(monkeypatch):
    """market xusd > ppp -> currency buys MORE units per USD than PPP predicts -> undervalued
    -> positive score (pre-reg: 'cheap = long')."""
    import factors as factors_mod

    def fake_ppp(ccy, d):
        return {"USD": 1.0, "EUR": 0.80}.get(ccy, np.nan)

    monkeypatch.setattr(factors_mod, "ppp_asof", fake_ppp)

    dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    xusd = pd.DataFrame(index=dates)
    xusd["USD"] = 1.0
    xusd["EUR"] = 0.90  # market: 0.90 EUR per USD, PPP: 0.80 EUR per USD -> EUR cheap
    for ccy in ("GBP", "AUD", "NZD", "JPY", "CAD", "CHF"):
        xusd[ccy] = np.nan

    s = value_score(xusd, dates[-1])
    assert s["EUR"] > 0
    assert np.isclose(s["USD"], 0.0)


def test_value_score_negative_when_market_richer_than_ppp(monkeypatch):
    import factors as factors_mod

    def fake_ppp(ccy, d):
        return {"USD": 1.0, "EUR": 0.90}.get(ccy, np.nan)

    monkeypatch.setattr(factors_mod, "ppp_asof", fake_ppp)

    dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    xusd = pd.DataFrame(index=dates)
    xusd["USD"] = 1.0
    xusd["EUR"] = 0.70  # market gives FEWER EUR per USD than PPP -> EUR expensive
    for ccy in ("GBP", "AUD", "NZD", "JPY", "CAD", "CHF"):
        xusd[ccy] = np.nan

    s = value_score(xusd, dates[-1])
    assert s["EUR"] < 0


# ── composite_score ──────────────────────────────────────────────────────────
def test_composite_score_averages_ranks_and_handles_partial_nan():
    carry_s = pd.Series({"USD": 0.0, "EUR": 1.0, "GBP": 2.0, "AUD": 3.0, "NZD": -1.0, "JPY": -2.0, "CAD": -3.0, "CHF": 0.5})
    mom_s = pd.Series({"USD": 0.0, "EUR": 3.0, "GBP": 1.0, "AUD": 2.0, "NZD": -2.0, "JPY": -3.0, "CAD": -1.0, "CHF": np.nan})
    val_s = pd.Series({c: np.nan for c in CURRENCIES})  # value entirely unavailable this month

    comp = composite_score(carry_s, mom_s, val_s)
    assert comp["USD"] == 0.0
    # CHF's composite should still be finite (averaged over carry+mom only, val NaN excluded).
    assert np.isfinite(comp["CHF"])
    assert set(comp.index) == set(CURRENCIES)
