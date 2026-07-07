"""Tests for rebalance_engine.py (incl. Gate 1 self-test and Gate 2 carry-accrual parity —
PREREGISTRATION.md "Gates before OOS"). Run on the Hetzner box (see test_currency_index.py
header) — carry_pips() needs the real fred_rates.parquet + financing_snapshot json that live
next to multiday_contrarian/carry_model.py, which this program imports directly (R6, reuse).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multiday_contrarian"))
from carry_model import carry_pips, pip_of  # noqa: E402

from currency_index import NON_USD_CURRENCIES
from null_r10 import run_null
from rebalance_engine import (
    build_master_calendar,
    build_rebalance_schedule,
    month_end_signal_dates,
    run_portfolio,
)


def _make_d1(dates, closes, pip=0.0001, spread_pips=1.0):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "timestamp": pd.DatetimeIndex(dates),
        "open": closes, "high": closes + pip, "low": closes - pip, "close": closes,
        "bid_c": closes - spread_pips * pip / 2,
        "ask_c": closes + spread_pips * pip / 2,
        "volume": 100,
    })


# ── calendar / schedule ──────────────────────────────────────────────────────
def test_month_end_signal_dates_picks_last_bar_of_each_month():
    dates = pd.bdate_range("2023-01-02", "2023-03-31", tz="UTC")
    signals = month_end_signal_dates(pd.DatetimeIndex(dates))
    assert [d.month for d in signals] == [1, 2, 3]
    for sd in signals:
        month_dates = [d for d in dates if d.month == sd.month]
        assert sd == max(month_dates)


def test_build_rebalance_schedule_execution_is_next_bar_after_signal():
    dates = pd.bdate_range("2023-01-02", "2023-03-31", tz="UTC")
    calendar = pd.DatetimeIndex(dates)
    schedule = build_rebalance_schedule(calendar)
    cal_list = list(calendar)
    assert len(schedule) == 2  # Jan, Feb month-ends have a following bar; Mar (last) does not
    for sd, ed in schedule:
        i = cal_list.index(sd)
        assert cal_list[i + 1] == ed
        assert ed > sd


def test_build_rebalance_schedule_drops_trailing_signal_with_no_next_bar():
    dates = pd.bdate_range("2023-01-02", "2023-01-31", tz="UTC")  # ends exactly on month-end
    schedule = build_rebalance_schedule(pd.DatetimeIndex(dates))
    assert schedule == []


def test_build_master_calendar_is_intersection_across_expression_pairs():
    dates = pd.bdate_range("2023-01-02", periods=10, tz="UTC")
    pair_d1 = {p: _make_d1(dates, np.full(10, 1.0)) for p in
               ("EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "CAD_JPY", "CHF_JPY")}
    pair_d1["EUR_USD"] = pair_d1["EUR_USD"].iloc[:-2].reset_index(drop=True)  # missing last 2
    cal = build_master_calendar(pair_d1)
    assert len(cal) == 8


# ── run_portfolio: single-leg hand-calc consistency ──────────────────────────
def _seven_pair_fixture(n=85, seed=0):
    dates = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
    rng = np.random.default_rng(seed)
    eur_prices = 1.10 + np.cumsum(rng.normal(0, 0.0003, size=n))
    pair_d1 = {
        "EUR_USD": _make_d1(dates, eur_prices),
        "GBP_USD": _make_d1(dates, np.full(n, 1.25)),
        "AUD_USD": _make_d1(dates, np.full(n, 0.65)),
        "NZD_USD": _make_d1(dates, np.full(n, 0.60)),
        "USD_JPY": _make_d1(dates, np.full(n, 150.0), pip=0.01),
        "CAD_JPY": _make_d1(dates, np.full(n, 110.0), pip=0.01),
        "CHF_JPY": _make_d1(dates, np.full(n, 170.0), pip=0.01),
    }
    return pair_d1


def test_run_portfolio_single_leg_matches_independent_recalc():
    pair_d1 = _seven_pair_fixture()
    calendar = build_master_calendar(pair_d1)
    schedule = build_rebalance_schedule(calendar)
    assert len(schedule) >= 2

    def direction_fn(sd):
        return {c: (1 if c == "EUR" else 0) for c in NON_USD_CURRENCIES}

    monthly_df, legs_df = run_portfolio(pair_d1, schedule, direction_fn)
    assert len(legs_df) == len(monthly_df)  # exactly one active leg (EUR) every rebalance
    row = legs_df.iloc[0]
    assert row["currency"] == "EUR"
    assert row["pair"] == "EUR_USD"
    assert row["trade_direction"] == 1
    assert row["weight"] == pytest.approx(1.0)  # sole active leg -> full weight regardless of vol

    pip = pip_of("EUR_USD")
    expected_gross = (row["exit_px"] - row["entry_px"]) / pip
    assert row["gross_pips"] == pytest.approx(expected_gross)

    expected_carry = carry_pips("EUR_USD", 1, row["entry_ts"], row["exit_ts"], markup_mult=1.0)
    assert row["carry_pips"] == pytest.approx(expected_carry)

    expected_net = row["gross_pips"] - row["spread_rt_pips"] + row["carry_pips"]
    assert row["net_pips"] == pytest.approx(expected_net)
    assert monthly_df.iloc[0]["net_pips"] == pytest.approx(row["net_pips"])


def test_run_portfolio_jpy_leg_sign_is_inverted():
    """Long JPY must SHORT the USD_JPY pair (currency_index.EXPRESSION['JPY'] = (USD_JPY,-1))
    -- a falling USD_JPY should produce a POSITIVE gross_pips for a long-JPY leg."""
    n = 85  # >= 3 months so build_rebalance_schedule yields >=2 usable rebalances (run_portfolio
    # needs a NEXT schedule entry to mark each position's exit to, per its own docstring)
    dates = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
    usdjpy_prices = np.linspace(150.0, 148.0, n)  # USD_JPY falling -> JPY strengthening
    pair_d1 = {
        "EUR_USD": _make_d1(dates, np.full(n, 1.10)),
        "GBP_USD": _make_d1(dates, np.full(n, 1.25)),
        "AUD_USD": _make_d1(dates, np.full(n, 0.65)),
        "NZD_USD": _make_d1(dates, np.full(n, 0.60)),
        "USD_JPY": _make_d1(dates, usdjpy_prices, pip=0.01),
        "CAD_JPY": _make_d1(dates, np.full(n, 110.0), pip=0.01),
        "CHF_JPY": _make_d1(dates, np.full(n, 170.0), pip=0.01),
    }
    calendar = build_master_calendar(pair_d1)
    schedule = build_rebalance_schedule(calendar)
    assert len(schedule) >= 2

    def direction_fn(sd):
        return {c: (1 if c == "JPY" else 0) for c in NON_USD_CURRENCIES}

    _, legs_df = run_portfolio(pair_d1, schedule, direction_fn)
    row = legs_df.iloc[0]
    assert row["trade_direction"] == -1  # long JPY (+1) * sign(-1) = -1 on the pair
    assert row["exit_px"] < row["entry_px"]  # USD_JPY fell
    assert row["gross_pips"] > 0  # short USD_JPY profits when it falls


# ── Gate 2: carry-accrual parity (+-5%) ───────────────────────────────────────
def test_carry_accrual_parity_bulk_vs_independent_day_loop():
    """Independent day-loop re-derivation of carry (summing carry_pips() one calendar day at a
    time) vs the engine's own bulk single-call sum, over a ~1-month holding period spanning
    several rollovers incl. a Wednesday triple-swap. Must agree within +-5% (Gate 2)."""
    entry_ts = pd.Timestamp("2023-01-05T00:00:00", tz="UTC")
    exit_ts = pd.Timestamp("2023-02-05T00:00:00", tz="UTC")

    bulk = carry_pips("EUR_USD", 1, entry_ts, exit_ts, markup_mult=1.0)

    day_loop_total = 0.0
    d = entry_ts
    while d < exit_ts:
        nxt = min(d + pd.Timedelta(days=1), exit_ts)
        day_loop_total += carry_pips("EUR_USD", 1, d, nxt, markup_mult=1.0)
        d = nxt

    assert bulk != 0.0
    rel_diff = abs(day_loop_total - bulk) / abs(bulk)
    assert rel_diff < 0.05, f"Gate 2 FAIL: bulk={bulk} day_loop={day_loop_total} rel_diff={rel_diff:.4f}"


def test_carry_accrual_parity_short_direction_and_jpy_cross():
    entry_ts = pd.Timestamp("2023-03-01T00:00:00", tz="UTC")
    exit_ts = pd.Timestamp("2023-04-01T00:00:00", tz="UTC")
    for pair, direction in (("EUR_USD", -1), ("USD_JPY", 1), ("CAD_JPY", -1)):
        bulk = carry_pips(pair, direction, entry_ts, exit_ts, markup_mult=1.0)
        day_loop_total = 0.0
        d = entry_ts
        while d < exit_ts:
            nxt = min(d + pd.Timedelta(days=1), exit_ts)
            day_loop_total += carry_pips(pair, direction, d, nxt, markup_mult=1.0)
            d = nxt
        if bulk == 0.0:
            assert abs(day_loop_total) < 1e-9
        else:
            rel_diff = abs(day_loop_total - bulk) / abs(bulk)
            assert rel_diff < 0.05, f"{pair} dir={direction}: bulk={bulk} day_loop={day_loop_total}"


# ── Gate 1: harness self-test (random weights ~= -costs) ─────────────────────
def test_spread_cost_bookkeeping_is_exactly_one_pip_per_rebalance():
    """Precise (non-statistical) version of the Gate-1 mechanism: with a KNOWN constant
    round-trip spread of 1.0 pip on every pair and weights normalized to sum to 1.0 across
    active legs, the spread-drag component of every single rebalance's net must equal EXACTLY
    -1.0 pip (sum_i w_i * spread_rt_pips_i = sum_i w_i * 1.0 = 1.0, since sum_i w_i = 1.0
    identically) -- a deterministic bookkeeping identity, checked directly (no seeds/stats
    needed). This isolates and proves the cost-accounting mechanism that the statistical
    Gate-1 self-test below (and the real one in results/gate_table.csv) relies on."""
    n = 260
    dates = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
    spread_pips = 1.0

    def _rw_pair(seed, price0, pip):
        rng = np.random.default_rng(seed)
        closes = price0 + np.cumsum(rng.normal(0, pip * 3, size=n))
        return _make_d1(dates, closes, pip=pip, spread_pips=spread_pips)

    pair_d1 = {
        "EUR_USD": _rw_pair(1, 1.10, 0.0001), "GBP_USD": _rw_pair(2, 1.25, 0.0001),
        "AUD_USD": _rw_pair(3, 0.65, 0.0001), "NZD_USD": _rw_pair(4, 0.60, 0.0001),
        "USD_JPY": _rw_pair(5, 150.0, 0.01), "CAD_JPY": _rw_pair(6, 110.0, 0.01),
        "CHF_JPY": _rw_pair(7, 170.0, 0.01),
    }
    calendar = build_master_calendar(pair_d1)
    schedule = build_rebalance_schedule(calendar)
    assert len(schedule) >= 6

    def direction_fn(sd):
        pattern = [1, 1, 1, -1, -1, -1, 0]
        return dict(zip(NON_USD_CURRENCIES, pattern))

    monthly_df, legs_df = run_portfolio(pair_d1, schedule, direction_fn)
    assert len(monthly_df) >= 6
    for sig_d, legs in legs_df.groupby("signal_date"):
        assert legs["weight"].sum() == pytest.approx(1.0, abs=1e-9)
        spread_drag = float((legs["weight"] * legs["spread_rt_pips"]).sum())
        assert spread_drag == pytest.approx(1.0, abs=1e-9)
        month_row = monthly_df[monthly_df["signal_date"] == sig_d].iloc[0]
        expected_net = float((legs["weight"] * legs["net_pips"]).sum())
        assert month_row["net_pips"] == pytest.approx(expected_net, abs=1e-9)


def test_self_test_random_weights_is_negative_and_bounded():
    """Statistical Gate-1 self-test on synthetic data: a zero-drift random-walk market with a
    KNOWN constant 1.0-pip round-trip spread. Spread drag alone is exactly -1.0 pip/rebalance
    (proven above); on top of that, REAL broker-truth carry (carry_model, used unmodified here)
    is not guaranteed zero-mean under a random direction draw -- OANDA's longRate/shortRate can
    BOTH be unfavorable for a given pair (the measured 'pinch' markup), so a random long/short
    selection can show a genuine additional negative carry drag. The self-test therefore checks
    the REALISTIC hypothesis (negative, no phantom edge) with the same generous magnitude bound
    used by compute_gates.py's real Gate 1 (|.| < 50p), not a precise zero-carry theoretical
    value. (The real Gate 1 check on actual IS data lives in results/gate_table.csv, produced
    by run_is_battery.py -> null_r10.py on real 12-pair history.)"""
    n = 260
    dates = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
    spread_pips = 1.0

    def _rw_pair(seed, price0, pip):
        rng = np.random.default_rng(seed)
        closes = price0 + np.cumsum(rng.normal(0, pip * 3, size=n))
        return _make_d1(dates, closes, pip=pip, spread_pips=spread_pips)

    pair_d1 = {
        "EUR_USD": _rw_pair(1, 1.10, 0.0001), "GBP_USD": _rw_pair(2, 1.25, 0.0001),
        "AUD_USD": _rw_pair(3, 0.65, 0.0001), "NZD_USD": _rw_pair(4, 0.60, 0.0001),
        "USD_JPY": _rw_pair(5, 150.0, 0.01), "CAD_JPY": _rw_pair(6, 110.0, 0.01),
        "CHF_JPY": _rw_pair(7, 170.0, 0.01),
    }
    calendar = build_master_calendar(pair_d1)
    schedule = build_rebalance_schedule(calendar)
    assert len(schedule) >= 6

    null_df = run_null(pair_d1, schedule, n_seeds=20)
    mean_net = null_df["mean_net_pips"].mean()
    assert mean_net < 0
    assert abs(mean_net) < 50.0, f"null mean implausibly large: {mean_net:+.4f}p"
