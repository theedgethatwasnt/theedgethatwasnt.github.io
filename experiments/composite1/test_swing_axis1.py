"""test_swing_axis1.py — TDD for the Axis-1 D1 swing-extreme detector (swing_axis1.py).

Synthetic-bar construction strategy: a "background" bar geometry with constant true range
(gapless closes) gives an exactly-stable Wilder ATR once warmed up, and a single
sharply-higher/lower "peak"/"trough" bar sets R/S while staying far enough from the
background level that eps (0.25xATR, empirically always well under half the R-to-background
gap in this construction) can never cross into background territory — so these tests use a
generous margin instead of hand-deriving the exact ATR/eps value, and remain robust to the
precise Wilder-decay arithmetic.
"""
import numpy as np
import pytest

from swing_axis1 import ATR_N, D1State, EPS_ATR_FRAC, HCAP, L, SL_ATR, TGT_ATR, TOUCH_MAX, VW, _atr, process_d1_bar

BG_HIGH, BG_LOW, BG_CLOSE = 100.00, 99.80, 99.90
PEAK_HIGH = 100.50   # background high (100.00) -> peak gap = 0.50, comfortably > any eps here
TROUGH_LOW = 99.30   # background low (99.80) -> trough gap = 0.50


def _bg_bar(vol=100.0):
    return {"open": BG_CLOSE, "high": BG_HIGH, "low": BG_LOW, "close": BG_CLOSE, "volume": vol}


def _feed(state, bars):
    last = None
    for b in bars:
        last = process_d1_bar(state, b)
    return last


def _warm_background(n=60, vol=100.0):
    """n background bars, enough to fully warm L, VW, and ATR_N."""
    state = D1State(pair="EUR_USD")
    for _ in range(n):
        process_d1_bar(state, _bg_bar(vol=vol))
    return state


def test_no_signal_before_warmup():
    state = D1State(pair="EUR_USD")
    sig = None
    for _ in range(max(L, VW)):
        sig = process_d1_bar(state, _bg_bar())
    assert sig is None


def test_flat_zero_atr_series_never_signals():
    """A perfectly flat series (zero true range) must never fire: atr<=0 guard rejects
    every bar outright."""
    state = D1State(pair="EUR_USD")
    flat = {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 100.0}
    fired = False
    for _ in range(200):
        if process_d1_bar(state, flat) is not None:
            fired = True
    assert not fired


def test_touch_up_first_touch_low_volume_fires_and_fades_short():
    state = _warm_background(n=60)
    # Insert ONE peak bar (sets R), well inside the L=25-bar lookback window, followed by a
    # handful more background bars before the touch bar.
    process_d1_bar(state, {"open": BG_CLOSE, "high": PEAK_HIGH, "low": BG_LOW, "close": BG_CLOSE, "volume": 100.0})
    for _ in range(10):
        process_d1_bar(state, _bg_bar())

    atr_before = _atr(state.bars)
    assert atr_before > 0
    eps = EPS_ATR_FRAC * atr_before
    assert eps < (PEAK_HIGH - BG_HIGH), "test construction assumption violated: eps too large"

    touch_bar = {"open": BG_CLOSE, "high": PEAK_HIGH - 0.02, "low": BG_LOW, "close": BG_CLOSE, "volume": 10.0}
    sig = process_d1_bar(state, touch_bar)

    assert sig is not None, f"expected a touch-up signal (eps={eps:.4f})"
    assert sig["direction"] == -1  # fade the HIGH: short
    assert sig["touches"] == 1
    assert sig["r"] == pytest.approx(PEAK_HIGH)


def test_touch_down_first_touch_low_volume_fires_and_fades_long():
    state = _warm_background(n=60)
    process_d1_bar(state, {"open": BG_CLOSE, "high": BG_HIGH, "low": TROUGH_LOW, "close": BG_CLOSE, "volume": 100.0})
    for _ in range(10):
        process_d1_bar(state, _bg_bar())

    atr_before = _atr(state.bars)
    eps = EPS_ATR_FRAC * atr_before
    assert eps < (BG_LOW - TROUGH_LOW), "test construction assumption violated: eps too large"

    touch_bar = {"open": BG_CLOSE, "high": BG_HIGH, "low": TROUGH_LOW + 0.02, "close": BG_CLOSE, "volume": 10.0}
    sig = process_d1_bar(state, touch_bar)

    assert sig is not None
    assert sig["direction"] == +1  # fade the LOW: long
    assert sig["touches"] == 1
    assert sig["s"] == pytest.approx(TROUGH_LOW)


def test_high_volume_touch_bar_is_rejected():
    """Same clean touch-up geometry as the passing test, but touch-bar volume is ABOVE the
    prior VW=20-bar mean (100) -- must be rejected (low-volume-only gate)."""
    state = _warm_background(n=60)
    process_d1_bar(state, {"open": BG_CLOSE, "high": PEAK_HIGH, "low": BG_LOW, "close": BG_CLOSE, "volume": 100.0})
    for _ in range(10):
        process_d1_bar(state, _bg_bar())

    touch_bar = {"open": BG_CLOSE, "high": PEAK_HIGH - 0.02, "low": BG_LOW, "close": BG_CLOSE, "volume": 500.0}
    sig = process_d1_bar(state, touch_bar)
    assert sig is None


def test_second_touch_in_window_rejects_the_would_be_first_touch():
    """Two near-peak approaches inside the L-bar window (the level-setting peak itself, plus
    a second bar that also gets within eps of R) means the CURRENT touch is no longer the
    'first' touch of that level within the lookback -- must be rejected (touches>TOUCH_MAX)."""
    state = _warm_background(n=50)
    process_d1_bar(state, {"open": BG_CLOSE, "high": PEAK_HIGH, "low": BG_LOW, "close": BG_CLOSE, "volume": 100.0})
    for _ in range(5):
        process_d1_bar(state, _bg_bar())
    # second near-touch of the same level, still comfortably inside eps
    process_d1_bar(state, {"open": BG_CLOSE, "high": PEAK_HIGH - 0.01, "low": BG_LOW, "close": BG_CLOSE, "volume": 100.0})
    for _ in range(5):
        process_d1_bar(state, _bg_bar())

    touch_bar = {"open": BG_CLOSE, "high": PEAK_HIGH - 0.02, "low": BG_LOW, "close": BG_CLOSE, "volume": 10.0}
    sig = process_d1_bar(state, touch_bar)
    assert sig is None, "a level tested twice already should not fire a fresh 'first touch'"


def test_close_above_r_is_not_a_fade_touch():
    """If the touch bar CLOSES beyond the level (not back inside it), it's a breakout, not a
    fade-able touch -- must be rejected even though the high/prev-high geometry matches."""
    state = _warm_background(n=60)
    process_d1_bar(state, {"open": BG_CLOSE, "high": PEAK_HIGH, "low": BG_LOW, "close": BG_CLOSE, "volume": 100.0})
    for _ in range(10):
        process_d1_bar(state, _bg_bar())

    touch_bar = {"open": BG_CLOSE, "high": PEAK_HIGH + 0.10, "low": BG_LOW, "close": PEAK_HIGH + 0.10, "volume": 10.0}
    sig = process_d1_bar(state, touch_bar)
    assert sig is None


def test_frozen_params_match_preregistration():
    assert L == 25
    assert VW == 20
    assert ATR_N == 14
    assert EPS_ATR_FRAC == 0.25
    assert TOUCH_MAX == 1
    assert TGT_ATR == 2.0
    assert SL_ATR == 4.0
    assert HCAP == 10
