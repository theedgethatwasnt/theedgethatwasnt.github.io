"""Tests for harness.py (Workstream A3). Run on the Hetzner box per CLAUDE.md directive:
    rsync -az research/experiments/multiday_contrarian/ root@HETZNER:/root/multiday/code/
    ssh root@HETZNER 'cd /root/multiday/code && /root/venv/bin/python -m pytest test_harness.py -x -q'
"""
import numpy as np
import pandas as pd
import pytest

from carry_model import carry_pips, pip_of
from harness import H4State, _scan_barriers, process_h4_bar, simulate_pair


# ── _scan_barriers: barrier + gap-through-stop fill semantics ────────────────
def _mkarr(times, opens, highs, lows):
    ts = np.array(times, dtype="datetime64[ns]")
    return np.array(opens), np.array(highs), np.array(lows), ts


def test_scan_barriers_clean_tp_touch_long():
    o, h, l, ts = _mkarr(
        ["2024-01-01T00:00", "2024-01-01T00:05", "2024-01-01T00:10"],
        [1.1000, 1.1010, 1.1055],
        [1.1005, 1.1015, 1.1060],
        [1.0995, 1.1005, 1.1050],
    )
    cap_ts = np.datetime64("2024-01-01T01:00")
    hit = _scan_barriers(o, h, l, ts, entry_pos=0, direction=1, tp=1.1050, sl=1.0950, cap_ts=cap_ts)
    assert hit == (1.1050, ts[2], "tp")


def test_scan_barriers_clean_sl_touch_long():
    o, h, l, ts = _mkarr(
        ["2024-01-01T00:00", "2024-01-01T00:05", "2024-01-01T00:10"],
        [1.1000, 1.0970, 1.0940],
        [1.1005, 1.0975, 1.0945],
        [1.0995, 1.0945, 1.0930],
    )
    cap_ts = np.datetime64("2024-01-01T01:00")
    hit = _scan_barriers(o, h, l, ts, entry_pos=0, direction=1, tp=1.1050, sl=1.0950, cap_ts=cap_ts)
    # bar index 1 (00:05): low=1.0945 <= sl=1.0950, open=1.0970 not yet past sl -> clean sl fill
    assert hit == (1.0950, ts[1], "sl")


def test_scan_barriers_sl_checked_before_tp_same_bar():
    """A bar whose range technically spans BOTH tp and sl (impossible to know true intrabar
    path from OHLC alone) must resolve to SL (R2-conservative)."""
    o, h, l, ts = _mkarr(
        ["2024-01-01T00:00", "2024-01-01T00:05"],
        [1.1000, 1.1000],
        [1.1000, 1.1060],   # bar 1 high clears tp=1.1050
        [1.1000, 1.0940],   # bar 1 low also clears sl=1.0950
    )
    cap_ts = np.datetime64("2024-01-01T01:00")
    hit = _scan_barriers(o, h, l, ts, entry_pos=0, direction=1, tp=1.1050, sl=1.0950, cap_ts=cap_ts)
    assert hit == (1.0950, ts[1], "sl")


def test_scan_barriers_gap_through_stop_long_fills_at_open_not_stop_price():
    o, h, l, ts = _mkarr(
        ["2024-01-01T00:00", "2024-01-01T00:05"],
        [1.1000, 1.0900],   # bar 1 opens BELOW sl=1.0950 already (gap down through the stop)
        [1.1000, 1.0905],
        [1.0995, 1.0890],
    )
    cap_ts = np.datetime64("2024-01-01T01:00")
    hit = _scan_barriers(o, h, l, ts, entry_pos=0, direction=1, tp=1.1050, sl=1.0950, cap_ts=cap_ts)
    assert hit == (1.0900, ts[1], "sl_gap")  # filled at the gapped OPEN, not the nominal 1.0950


def test_scan_barriers_gap_through_stop_short_fills_at_open_not_stop_price():
    o, h, l, ts = _mkarr(
        ["2024-01-01T00:00", "2024-01-01T00:05"],
        [1.1000, 1.1100],   # short: stop is ABOVE entry; bar 1 gaps up through it
        [1.1000, 1.1105],
        [1.0995, 1.1095],
    )
    cap_ts = np.datetime64("2024-01-01T01:00")
    hit = _scan_barriers(o, h, l, ts, entry_pos=0, direction=-1, tp=1.0950, sl=1.1050, cap_ts=cap_ts)
    assert hit == (1.1100, ts[1], "sl_gap")


def test_scan_barriers_entry_bar_own_range_checked_not_skipped():
    """The entry bar's OWN high/low (after its open, which is the fill price) must still be
    checked — price can hit a barrier within the same 5-min bar it was entered on."""
    o, h, l, ts = _mkarr(
        ["2024-01-01T00:00", "2024-01-01T00:05"],
        [1.1000, 1.1000],
        [1.1060, 1.1000],   # TP cleared WITHIN the entry bar itself (index 0)
        [1.0995, 1.0995],
    )
    cap_ts = np.datetime64("2024-01-01T01:00")
    hit = _scan_barriers(o, h, l, ts, entry_pos=0, direction=1, tp=1.1050, sl=1.0950, cap_ts=cap_ts)
    assert hit == (1.1050, ts[0], "tp")


def test_scan_barriers_no_hit_before_cap_returns_none():
    o, h, l, ts = _mkarr(
        ["2024-01-01T00:00", "2024-01-01T00:05", "2024-01-01T00:10"],
        [1.1000, 1.1001, 1.1002],
        [1.1002, 1.1003, 1.1004],
        [1.0998, 1.0999, 1.1000],
    )
    cap_ts = np.datetime64("2024-01-01T00:10")  # excludes bar index 2 (>= cap_ts breaks the loop)
    hit = _scan_barriers(o, h, l, ts, entry_pos=0, direction=1, tp=1.1050, sl=1.0950, cap_ts=cap_ts)
    assert hit is None


# ── process_h4_bar: verbatim first-touch signal port ─────────────────────────
def _make_signal_bars(direction="short", n=30, peak_idx=10):
    """30 H4 bars: flat everywhere except one bar (peak_idx) that sets a swing extreme, and
    the final bar that re-tests it on low volume — hand-built to deterministically fire (or
    not) the exact `_on_new_bar` entry condition ported into process_h4_bar."""
    base_ts = pd.Timestamp("2021-01-04T22:00:00Z")
    bars = []
    for i in range(n):
        ts = base_ts + pd.Timedelta(hours=4 * i)
        if i == peak_idx:
            if direction == "short":
                bars.append(dict(timestamp=ts, open=1.1000, high=1.1020, low=1.0980, close=1.0990, volume=500))
            else:
                bars.append(dict(timestamp=ts, open=1.1000, high=1.1000, low=1.0960, close=1.0990, volume=500))
        elif i == n - 1:
            if direction == "short":
                bars.append(dict(timestamp=ts, open=1.1000, high=1.1015, low=1.0995, close=1.1005, volume=400))
            else:
                bars.append(dict(timestamp=ts, open=1.1000, high=1.1005, low=1.0965, close=1.0995, volume=400))
        else:
            bars.append(dict(timestamp=ts, open=1.1000, high=1.1000, low=1.0980, close=1.0990, volume=500))
    return bars


def test_process_h4_bar_first_touch_fade_short():
    bars = _make_signal_bars("short")
    state = H4State(pair="EUR_USD")
    pip = pip_of("EUR_USD")
    sigs = [process_h4_bar(state, b, pip) for b in bars]
    assert all(s is None for s in sigs[:-1])
    sig = sigs[-1]
    assert sig is not None
    assert sig["direction"] == -1        # fade the resistance touch: go short
    assert sig["touches"] == 1
    assert sig["vrel"] == pytest.approx(0.8)
    assert sig["atr"] > 0


def test_process_h4_bar_first_touch_fade_long():
    bars = _make_signal_bars("long")
    state = H4State(pair="EUR_USD")
    pip = pip_of("EUR_USD")
    sigs = [process_h4_bar(state, b, pip) for b in bars]
    assert all(s is None for s in sigs[:-1])
    sig = sigs[-1]
    assert sig is not None
    assert sig["direction"] == +1         # fade the support touch: go long
    assert sig["touches"] == 1


def test_process_h4_bar_rejects_when_volume_not_low():
    bars = _make_signal_bars("short")
    bars[-1]["volume"] = 700  # vrel = 700/500 = 1.4 > VREL_MAX(1.16)
    state = H4State(pair="EUR_USD")
    pip = pip_of("EUR_USD")
    sigs = [process_h4_bar(state, b, pip) for b in bars]
    assert all(s is None for s in sigs)


def test_process_h4_bar_rejects_when_touches_exceed_max():
    bars = _make_signal_bars("short")
    # add a second spike inside the lookback window so the level has been tested twice already
    bars[15]["high"] = 1.1020
    state = H4State(pair="EUR_USD")
    pip = pip_of("EUR_USD")
    sigs = [process_h4_bar(state, b, pip) for b in bars]
    assert all(s is None for s in sigs)


# ── carry wired per held-days (integration, not a re-test of A1's own math) ──
def _make_rw_m5(n_days, seed, pair="EUR_USD", start="2021-01-04T22:00:00Z", spread_pips=1.7,
                 vol_mean=500.0, vol_std=140.0, step_pip_mult=3.0):
    """step_pip_mult=3.0 was calibrated (not arbitrary) so H4 ATR(14) lands ~30 pips, comparable
    to EUR_USD's real scale — with the default 0.5 tried first, ATR(H4)~5-10p made the frozen
    EPS=12p touch tolerance SWALLOW almost the whole 25-bar window (everything looks like a
    "touch"), so `touches > TOUCH_MAX` rejected every single candidate and 0 signals ever fired.
    Confirmed by sweeping step_pip_mult on the Hetzner box: 0.5/1.0 -> 0 signals, 2.0 -> 20,
    3.0 -> 45 signals per 2 synthetic years — this is a self-test calibration, not a claim
    about real EUR_USD volatility."""
    rng = np.random.default_rng(seed)
    pip = pip_of(pair)
    n = n_days * 24 * 12  # 12 M5 bars/hour
    ts = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    steps = rng.normal(0.0, pip * step_pip_mult, size=n)
    mid = 1.10000 + np.cumsum(steps)
    wiggle = np.abs(rng.normal(0.0, pip * 0.3, size=n))
    open_ = mid + rng.normal(0.0, pip * 0.1, size=n)
    close = mid
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    spread = spread_pips * pip
    bid_c = close - spread / 2.0
    ask_c = close + spread / 2.0
    volume = np.maximum(1, rng.normal(vol_mean, vol_std, size=n)).astype(int)
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low, "close": close,
        "bid_c": bid_c, "ask_c": ask_c, "volume": volume,
    })


def test_carry_is_wired_per_trade_matches_a1_directly():
    """Integration check: every trade harness produces must carry a `carry_pips` value that
    EXACTLY matches an independent carry_model.carry_pips() call with that trade's own
    pair/direction/entry_ts/exit_ts — confirms the wiring (right args, not silently zeroed),
    not a re-test of A1's own rate math (already covered by test_carry_model.py)."""
    df = _make_rw_m5(n_days=365 * 3, seed=7)
    trades = simulate_pair("EUR_USD", df, arm="signal")
    assert len(trades) >= 5, "need at least a few trades for this to be a meaningful check"
    for t in trades:
        expected = carry_pips("EUR_USD", t["direction"], t["entry_ts"], t["exit_ts"])
        assert t["carry_pips"] == pytest.approx(expected, abs=1e-12)


def test_carry_scales_with_held_days_not_flat():
    """Trades with more held-hours (more rollovers crossed) must show carry magnitude that
    tracks holding duration, not a constant/flat per-trade value (which would indicate carry
    isn't actually keyed off entry_ts/exit_ts)."""
    df = _make_rw_m5(n_days=365 * 4, seed=11)
    trades = simulate_pair("EUR_USD", df, arm="signal")
    assert len(trades) >= 10
    hours = np.array([t["hours_held"] for t in trades])
    carry = np.array([abs(t["carry_pips"]) for t in trades])
    # not ALL trades can have identical (e.g. zero) carry if hold durations vary meaningfully
    assert hours.std() > 1.0  # sanity: durations actually vary in this sample
    assert not np.allclose(carry, carry[0]), "carry must vary across trades, not be a constant"


# ── synthetic-RW self-test (gate 1 of the pre-registration) ──────────────────
def test_synthetic_rw_no_phantom_edge():
    """On a true random walk, NEITHER the coin-flip control NOR the fade signal can have a
    real gross edge (optional-stopping: any non-anticipating direction rule has zero expected
    gross P&L on a martingale). This is PREREGISTRATION.md gate 1: coin's net expectancy ≈
    -(spread+carry) and signal ≈ coin — no phantom edge from the harness itself."""
    df = _make_rw_m5(n_days=365 * 8, seed=42)
    sig_trades = simulate_pair("EUR_USD", df, arm="signal", seed=20260706)
    coin_trades = simulate_pair("EUR_USD", df, arm="coin", seed=20260706)

    assert len(sig_trades) >= 20, f"too few signal trades ({len(sig_trades)}) for a meaningful self-test"
    assert len(coin_trades) >= 20, f"too few coin trades ({len(coin_trades)}) for a meaningful self-test"

    def gross_stats(trades):
        v = np.array([t["gross_pips"] for t in trades])
        n = len(v)
        se = v.std(ddof=1) / np.sqrt(n)
        return v.mean(), se, n

    mean_g_sig, se_g_sig, n_sig = gross_stats(sig_trades)
    mean_g_coin, se_g_coin, n_coin = gross_stats(coin_trades)

    # (a) neither arm's GROSS pnl is distinguishable from zero (no phantom edge, no matter
    # which non-anticipating direction rule chose the trade).
    assert abs(mean_g_sig) < 3 * se_g_sig, (
        f"signal arm shows a gross edge on a random walk: {mean_g_sig:+.3f}p (se={se_g_sig:.3f}, n={n_sig})"
    )
    assert abs(mean_g_coin) < 3 * se_g_coin, (
        f"coin arm shows a gross edge on a random walk: {mean_g_coin:+.3f}p (se={se_g_coin:.3f}, n={n_coin})"
    )

    # (b) signal ≈ coin (same absence of structure, within combined sampling noise).
    se_diff = np.sqrt(se_g_sig**2 + se_g_coin**2)
    assert abs(mean_g_sig - mean_g_coin) < 4 * se_diff, (
        f"signal ({mean_g_sig:+.3f}p) diverges from coin ({mean_g_coin:+.3f}p) beyond sampling "
        f"noise (se_diff={se_diff:.3f}) — phantom edge in the fade-signal logic itself"
    )

    # (c) net ≈ -(spread + carry): with gross ≈ 0, net is cost-dominated and must be negative
    # and of the right order of magnitude (a couple of pips — NOT ~zero, NOT tens of pips).
    mean_net_coin = np.mean([t["net_pips"] for t in coin_trades])
    mean_spread_coin = np.mean([t["spread_rt_pips"] for t in coin_trades])
    mean_carry_coin = np.mean([t["carry_pips"] for t in coin_trades])
    assert mean_net_coin < 0
    assert mean_net_coin == pytest.approx(mean_g_coin - mean_spread_coin + mean_carry_coin, abs=1e-9)
    assert 0 < mean_spread_coin < 10  # sane pip-scale for a 1.7p-spread synthetic EUR_USD


def test_continuation_arm_uses_identical_signal_timestamps():
    """The continuation control arm must fire on the EXACT same signal bars as the fade
    signal (only the chosen trade direction differs) — 'identical timestamps' per the
    pre-registration's control-arm definition."""
    df = _make_rw_m5(n_days=365 * 2, seed=99)
    sig_trades = simulate_pair("EUR_USD", df, arm="signal", seed=20260706)
    cont_trades = simulate_pair("EUR_USD", df, arm="continuation", seed=20260706)
    assert len(sig_trades) == len(cont_trades)
    assert len(sig_trades) >= 5
    for a, b in zip(sig_trades, cont_trades):
        assert a["signal_ts"] == b["signal_ts"]
        assert a["direction"] == -b["direction"]
