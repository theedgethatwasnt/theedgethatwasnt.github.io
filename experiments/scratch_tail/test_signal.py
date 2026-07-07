"""Tests for signal.py (verbatim SMA16/six_of_six/ATR port + vectorized precompute). Run on
Hetzner: /root/venv/bin/python -m pytest /root/work/code/scratch_tail/test_signal.py -x -q
"""
from collections import deque

import numpy as np
import pandas as pd
import pytest

from bars import m5_to_h1, m5_to_m30
from signal import (CONFIGS, LAGS, PAIRS, SMA_N, TP_PIPS, build_pair_signal, six_of_six, sma_n,
                     wilder_atr)


# ── CONFIGS sanity: verbatim frozen params match PREREGISTRATION.md's table ──
def test_configs_match_preregistration_table():
    expected = {
        "USD_JPY": (576, 0.5), "NZD_USD": (288, 1.5), "GBP_USD": (144, 1.5),
        "CAD_JPY": (96, 0.5), "AUD_USD": (144, 2.0), "GBP_JPY": (72, 1.5),
    }
    assert set(PAIRS) == set(expected)
    for c in CONFIGS:
        exp_ts, exp_k = expected[c.pair]
        assert c.T_s_bars == exp_ts
        assert c.k_atr == pytest.approx(exp_k)
        # per the module docstring's documented discrepancy: quality filter inactive for all 6
        assert c.T_q_bars is None and c.X_pips is None


def test_pip_values_match_jpy_convention():
    for c in CONFIGS:
        expected_pip = 0.01 if c.pair.endswith("_JPY") else 0.0001
        assert c.pip == pytest.approx(expected_pip)


# ── sma_n / six_of_six: basic unit behavior ───────────────────────────────────
def test_sma_n_returns_none_before_enough_history():
    d = deque(maxlen=64)
    for v in range(10):
        d.append(v)
    assert sma_n(d, SMA_N) is None


def test_sma_n_matches_plain_mean_of_last_n():
    d = deque(maxlen=64)
    for v in np.random.default_rng(1).normal(size=40):
        d.append(v)
    got = sma_n(d, SMA_N)
    assert got == pytest.approx(np.mean(list(d)[-SMA_N:]))


def test_six_of_six_returns_zero_without_enough_lag_history():
    d = deque(maxlen=64)
    for v in range(20):  # enough for SMA_N=16 current, not for lag=15 (needs 31)
        d.append(1.0 + v * 0.001)
    assert six_of_six(d, LAGS) == 0


def test_six_of_six_fires_up_on_strictly_rising_series():
    d = deque(maxlen=64)
    for v in range(40):
        d.append(1.0 + v * 0.01)   # monotonically rising -> current SMA16 > every lagged SMA16
    assert six_of_six(d, LAGS) == 1


def test_six_of_six_fires_down_on_strictly_falling_series():
    d = deque(maxlen=64)
    for v in range(40):
        d.append(10.0 - v * 0.01)
    assert six_of_six(d, LAGS) == -1


def test_six_of_six_zero_on_flat_series():
    d = deque(maxlen=64)
    for _ in range(40):
        d.append(1.0)
    assert six_of_six(d, LAGS) == 0


# ── wilder_atr: sanity ─────────────────────────────────────────────────────────
def test_wilder_atr_none_before_period_plus_one_bars():
    h, l, c = deque(maxlen=64), deque(maxlen=64), deque(maxlen=64)
    for i in range(10):
        h.append(1.001); l.append(0.999); c.append(1.000)
    assert wilder_atr(h, l, c, 14) is None


def test_wilder_atr_positive_on_noisy_series():
    rng = np.random.default_rng(3)
    h, l, c = deque(maxlen=64), deque(maxlen=64), deque(maxlen=64)
    price = 1.10
    for _ in range(30):
        price += rng.normal(0, 0.001)
        h.append(price + 0.0005); l.append(price - 0.0005); c.append(price)
    v = wilder_atr(h, l, c, 14)
    assert v is not None and v > 0


# ── build_pair_signal: hand-built small example, checked against a direct
#    (independent, non-vectorized) replay of the verbatim functions ──────────
def _make_m5(start, periods, freq="5min", step=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    steps = rng.normal(0, step, size=periods)
    mid = 1.10000 + np.cumsum(steps)
    wig = np.abs(rng.normal(0, step * 0.3, size=periods))
    close = mid
    open_ = mid + rng.normal(0, step * 0.1, size=periods)
    high = np.maximum(open_, close) + wig
    low = np.minimum(open_, close) - wig
    spread = 1.5e-4
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low, "close": close,
        "bid_c": close - spread / 2, "ask_c": close + spread / 2,
        "volume": np.maximum(1, rng.normal(500, 100, size=periods)).astype(int),
    })


def test_build_pair_signal_output_shape_and_range():
    df = _make_m5("2024-01-01T00:00:00Z", periods=12 * 24 * 60, seed=7)  # 60 days of M5
    dir_signal, atr_h1 = build_pair_signal("EUR_USD", df)
    assert len(dir_signal) == len(df) == len(atr_h1)
    assert set(np.unique(dir_signal)).issubset({-1, 0, 1})
    # ATR must be NaN early (warmup), then strictly positive once available
    valid = ~np.isnan(atr_h1)
    assert valid.any()
    assert (atr_h1[valid] > 0).all()


def test_build_pair_signal_dir_signal_is_causal_no_lookahead():
    """Truncating the input to bar k must not change dir_signal for any bar < k (a lookahead
    check: nothing in the future can influence a past decision)."""
    df = _make_m5("2024-01-01T00:00:00Z", periods=12 * 24 * 40, seed=11)
    full_dir, full_atr = build_pair_signal("EUR_USD", df)
    cut = len(df) - 500
    trunc_dir, trunc_atr = build_pair_signal("EUR_USD", df.iloc[:cut].reset_index(drop=True))
    # allow the last few bars before the cut to differ only if they depend on an H1/M30 bar
    # that straddles the cut (aggregation edge) — check a safely-interior prefix instead.
    safe = cut - 200
    assert np.array_equal(full_dir[:safe], trunc_dir[:safe])
    both_valid = ~np.isnan(full_atr[:safe]) & ~np.isnan(trunc_atr[:safe])
    assert np.allclose(full_atr[:safe][both_valid], trunc_atr[:safe][both_valid])


def test_build_pair_signal_matches_direct_deque_replay_at_spot_checks():
    """Cross-check the vectorized precompute against an independent, straightforward
    (non-searchsorted) forward-fill replay at several spot-check M5 indices."""
    df = _make_m5("2024-01-01T00:00:00Z", periods=12 * 24 * 20, seed=5)  # 20 days
    dir_signal, atr_h1 = build_pair_signal("EUR_USD", df)

    h1 = m5_to_h1(df)
    m30 = m5_to_m30(df)
    h1_close = (h1["timestamp"] + pd.Timedelta(hours=1)).to_numpy()
    m30_close = (m30["timestamp"] + pd.Timedelta(minutes=30)).to_numpy()

    # independent reference: recompute h_sig/m_sig at each H1/M30 bar via fresh deques (not
    # reusing signal.py's own _replay_tf_signal, to actually cross-check the module)
    def ref_series(closes):
        d = deque(maxlen=64)
        out = []
        for c in closes:
            d.append(c)
            out.append(six_of_six(d, LAGS))
        return np.array(out, dtype=np.int8)

    h1_sig_ref = ref_series(h1["close"].to_numpy())
    m30_sig_ref = ref_series(m30["close"].to_numpy())

    m5_ts = df["timestamp"].to_numpy()
    for idx in [100, 2000, 5000, len(df) - 1]:
        t = m5_ts[idx]
        h_pos = np.searchsorted(h1_close, t, side="right") - 1
        m_pos = np.searchsorted(m30_close, t, side="right") - 1
        h_sig = h1_sig_ref[h_pos] if h_pos >= 0 else 0
        m_sig = m30_sig_ref[m_pos] if m_pos >= 0 else 0
        expected = 1 if (h_sig == 1 and m_sig == 1) else (-1 if (h_sig == -1 and m_sig == -1) else 0)
        assert dir_signal[idx] == expected, f"idx={idx}: expected {expected}, got {dir_signal[idx]}"
