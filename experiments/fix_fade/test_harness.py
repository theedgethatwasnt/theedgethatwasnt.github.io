"""Tests for harness.py — drift computation, bar-index arithmetic, 3-arm identical-timestamp
construction, real spread cost, excursion tracking, and gate 1 (RW self-test)."""
import numpy as np
import pandas as pd
import pytest

from data_loader import pip_of
from fixtime import london_fix_close_utc
from harness import DEFAULT_SEED, THRESH_PIPS, build_trades, find_signal_events
from rw_selftest import run_rw_selftest


def _mk_day_bars(date, pip, drift_pips, spread_pips=1.5, post_move_pips=0.0,
                  extra_low_pips=0.0, extra_high_pips=0.0):
    """Build one full day (288 M5 bars, 00:00-23:55 UTC open times) around a chosen London-fix
    date, with a KNOWN pre-fix drift (`drift_pips` = mid(16:00 Ldn) - mid(15:00 Ldn)) and a
    KNOWN post-entry move (`post_move_pips`, applied linearly over the 60-min hold so the exit
    close lands exactly `post_move_pips` pips above the entry open). `extra_low/high_pips`
    inject one deliberate excursion beyond the entry/exit prices, to test MFE/MAE tracking."""
    fix_close = london_fix_close_utc(date)
    # Narrow window (+/- 3h around the fix) — comfortably covers prior(-65m)/fix(-5m)/entry(0)/
    # exit(+55m) with margin, while staying well under 24h so consecutive calendar days' windows
    # (used by tests that stitch several days together) never overlap.
    ts = pd.date_range(fix_close - pd.Timedelta(hours=3), periods=6 * 12, freq="5min", tz="UTC")

    base = 1.10000
    mid = np.full(len(ts), base)

    fix_bar_open = fix_close - pd.Timedelta(minutes=5)      # closes AT the fix
    prior_bar_open = fix_close - pd.Timedelta(minutes=65)   # closes 60min before the fix
    entry_bar_open = fix_close                              # opens the instant the fix bar closes

    idx = {t: i for i, t in enumerate(ts)}
    prior_idx = idx[prior_bar_open]
    fix_idx = idx[fix_bar_open]
    entry_idx = idx[entry_bar_open]

    # Flat before prior_bar's close; jump by drift_pips exactly between prior close and fix close.
    mid[:prior_idx + 1] = base
    mid[prior_idx + 1:fix_idx + 1] = base + drift_pips * pip
    # Flat from fix close through entry open (entry_px = mid at entry_idx open = same as fix close).
    entry_px_value = base + drift_pips * pip
    mid[fix_idx + 1:entry_idx + 1] = entry_px_value
    # Post-entry: ramp linearly (bars entry_idx+1 .. entry_idx+11) to entry_px+post_move by the
    # exit bar's close (entry_idx+11 = the 12th/last bar of the hold window). entry_idx itself
    # (k=0, the entry bar's own open==close in this flat-bar fixture) is LEFT UNTOUCHED at
    # entry_px_value so `entry_px` is exactly known for excursion-math assertions below.
    hold_bars = 12
    for k in range(1, hold_bars):
        j = entry_idx + k
        frac = k / (hold_bars - 1)
        mid[j] = entry_px_value + post_move_pips * pip * frac
    tail_val = mid[entry_idx + hold_bars - 1]
    mid[entry_idx + hold_bars:] = tail_val

    open_ = mid.copy()
    close = mid.copy()
    high = mid.copy()
    low = mid.copy()
    # extra_low/high are deliberate intrabar wicks on bar entry_idx+1, expressed relative to the
    # KNOWN entry_px_value (not that bar's own ramped mid) so excursion-tracking tests can assert
    # an exact pip figure regardless of post_move_pips.
    if extra_high_pips:
        high[entry_idx + 1] = max(high[entry_idx + 1], entry_px_value + extra_high_pips * pip)
    if extra_low_pips:
        low[entry_idx + 1] = min(low[entry_idx + 1], entry_px_value - extra_low_pips * pip)

    spread = spread_pips * pip
    bid_c = close - spread / 2.0
    ask_c = close + spread / 2.0

    df = pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low, "close": close,
        "bid_c": bid_c, "ask_c": ask_c,
    })
    return df, dict(fix_idx=fix_idx, prior_idx=prior_idx, entry_idx=entry_idx)


# ── find_signal_events: drift computation + threshold + bar-index arithmetic ─────────────────
def test_drift_above_threshold_fires_signal():
    pair = "EUR_USD"
    pip = pip_of(pair)
    df, ix = _mk_day_bars("2023-05-03", pip, drift_pips=8.0)
    events, stats = find_signal_events(pair, df)
    assert len(events) == 1
    assert events[0].D_pips == pytest.approx(8.0, abs=1e-6)
    assert stats["n_signal"] == 1
    assert stats["n_below_threshold"] == 0


def test_drift_below_threshold_no_signal():
    pair = "EUR_USD"
    pip = pip_of(pair)
    df, ix = _mk_day_bars("2023-05-03", pip, drift_pips=3.0)  # < THRESH_PIPS=5
    events, stats = find_signal_events(pair, df)
    assert len(events) == 0
    assert stats["n_below_threshold"] == 1


def test_drift_exactly_at_threshold_fires():
    pair = "EUR_USD"
    pip = pip_of(pair)
    df, ix = _mk_day_bars("2023-05-03", pip, drift_pips=THRESH_PIPS)
    events, stats = find_signal_events(pair, df)
    assert len(events) == 1


def test_negative_drift_fires_and_fade_goes_long():
    pair = "EUR_USD"
    pip = pip_of(pair)
    df, ix = _mk_day_bars("2023-05-03", pip, drift_pips=-9.0)
    events, _ = find_signal_events(pair, df)
    assert len(events) == 1
    assert events[0].D_pips == pytest.approx(-9.0, abs=1e-6)
    trades = build_trades(events, "fade", pair)
    assert trades[0]["direction"] == 1  # D<0 -> fade goes AGAINST the drift -> long


def test_entry_is_next_bar_open_after_fix_bar_closes():
    pair = "EUR_USD"
    pip = pip_of(pair)
    df, ix = _mk_day_bars("2023-05-03", pip, drift_pips=6.0)
    events, _ = find_signal_events(pair, df)
    ev = events[0]
    assert ev.entry_ts == pd.Timestamp(df["timestamp"].iloc[ix["entry_idx"]])
    assert ev.entry_px == pytest.approx(df["open"].iloc[ix["entry_idx"]])


def test_exit_is_60_minutes_after_entry_at_close():
    pair = "EUR_USD"
    pip = pip_of(pair)
    df, ix = _mk_day_bars("2023-05-03", pip, drift_pips=6.0, post_move_pips=4.0)
    events, _ = find_signal_events(pair, df)
    ev = events[0]
    assert ev.exit_ts == ev.entry_ts + pd.Timedelta(minutes=60)
    exit_idx = ix["entry_idx"] + 11
    assert ev.exit_px == pytest.approx(df["close"].iloc[exit_idx])


@pytest.mark.parametrize("spring", ["2023-03-26", "2023-10-29"])
def test_signal_detection_across_dst_transition_day(spring):
    """The harness's exact-bar-match logic must not break on the DST transition day itself
    (16:00 London's UTC hour flips, but the M5 grid is continuous UTC — no gap introduced by
    this synthetic fixture)."""
    pair = "EUR_USD"
    pip = pip_of(pair)
    df, ix = _mk_day_bars(spring, pip, drift_pips=7.0)
    events, stats = find_signal_events(pair, df)
    assert len(events) == 1
    assert stats["n_missing_grid"] == 0


# ── excursion (MFE/MAE) tracking ──────────────────────────────────────────────
def test_excursion_tracks_worst_adverse_move_for_long_fade():
    pair = "EUR_USD"
    pip = pip_of(pair)
    # D=-9 -> fade goes long; inject a deliberate dip 6p below entry within the hold window.
    df, ix = _mk_day_bars("2023-05-03", pip, drift_pips=-9.0, post_move_pips=2.0, extra_low_pips=6.0)
    events, _ = find_signal_events(pair, df)
    trades = build_trades(events, "fade", pair)
    t = trades[0]
    assert t["direction"] == 1
    assert t["mae_pips"] == pytest.approx(6.0, abs=1e-6)


def test_excursion_swaps_for_opposite_direction_same_event():
    """continuation trades the OPPOSITE direction of fade on the SAME event — its mfe/mae must
    be the fade trade's mae/mfe swapped, not independently recomputed."""
    pair = "EUR_USD"
    pip = pip_of(pair)
    df, ix = _mk_day_bars("2023-05-03", pip, drift_pips=-9.0, post_move_pips=2.0, extra_low_pips=6.0,
                           extra_high_pips=1.5)
    events, _ = find_signal_events(pair, df)
    fade = build_trades(events, "fade", pair)[0]
    cont = build_trades(events, "continuation", pair)[0]
    assert fade["direction"] == -cont["direction"]
    assert fade["mfe_pips"] == pytest.approx(cont["mae_pips"], abs=1e-9)
    assert fade["mae_pips"] == pytest.approx(cont["mfe_pips"], abs=1e-9)


# ── spread cost ────────────────────────────────────────────────────────────
def test_spread_cost_matches_manual_calc():
    pair = "EUR_USD"
    pip = pip_of(pair)
    df, ix = _mk_day_bars("2023-05-03", pip, drift_pips=6.0, post_move_pips=3.0, spread_pips=2.0)
    events, _ = find_signal_events(pair, df)
    trades = build_trades(events, "fade", pair)
    t = trades[0]
    # spread_pips fixed at 2.0 on every bar in this fixture -> round-trip = (2.0+2.0)/2 = 2.0
    assert t["spread_rt_pips"] == pytest.approx(2.0, abs=1e-6)
    assert t["net_pips"] == pytest.approx(t["gross_pips"] - 2.0, abs=1e-6)


# ── 3-arm identical timestamps (R10) ──────────────────────────────────────
def test_three_arms_share_identical_timestamps():
    pair = "EUR_USD"
    pip = pip_of(pair)
    df1, _ = _mk_day_bars("2023-05-03", pip, drift_pips=7.0)
    events, _ = find_signal_events(pair, df1)
    fade = build_trades(events, "fade", pair)
    coin = build_trades(events, "coin", pair)
    cont = build_trades(events, "continuation", pair)
    assert len(fade) == len(coin) == len(cont) == 1
    for a, b in zip(fade, coin):
        assert a["entry_ts"] == b["entry_ts"] and a["exit_ts"] == b["exit_ts"]
    for a, b in zip(fade, cont):
        assert a["entry_ts"] == b["entry_ts"] and a["exit_ts"] == b["exit_ts"]
        assert a["direction"] == -b["direction"]


def test_coin_arm_reproducible_with_same_seed():
    pair = "EUR_USD"
    pip = pip_of(pair)
    # Build several days with alternating drift signs so multiple events exist.
    dfs = []
    for i, d in enumerate(["2023-05-03", "2023-05-04", "2023-05-05", "2023-05-08", "2023-05-09"]):
        sub, _ = _mk_day_bars(d, pip, drift_pips=6.0 + i)
        dfs.append(sub)
    df = pd.concat(dfs).drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    events, stats = find_signal_events(pair, df)
    assert stats["n_signal"] >= 3
    coin1 = build_trades(events, "coin", pair, seed=DEFAULT_SEED)
    coin2 = build_trades(events, "coin", pair, seed=DEFAULT_SEED)
    assert [t["direction"] for t in coin1] == [t["direction"] for t in coin2]


# ── month-end tagging wired into events ───────────────────────────────────
def test_month_end_flag_true_on_last_trading_day():
    pair = "EUR_USD"
    pip = pip_of(pair)
    # 2023-05-31 is a Wednesday (last trading day of May 2023).
    df, _ = _mk_day_bars("2023-05-31", pip, drift_pips=6.0)
    events, _ = find_signal_events(pair, df)
    assert events[0].month_end is True


def test_month_end_flag_false_on_a_mid_month_day():
    pair = "EUR_USD"
    pip = pip_of(pair)
    df, _ = _mk_day_bars("2023-05-15", pip, drift_pips=6.0)
    events, _ = find_signal_events(pair, df)
    assert events[0].month_end is False


# ── gate 1: RW self-test (no phantom edge on a random walk) ──────────────
def test_synthetic_rw_no_phantom_edge():
    result = run_rw_selftest(n_days=365 * 3, seed=42)
    assert result["checks"]["enough_trades"], result["event_stats"]
    assert result["checks"]["fade_gross_not_distinguishable_from_zero"], (
        f"fade arm shows a gross edge on a random walk: {result['mean_gross_fade']:+.3f}p "
        f"(se={result['se_gross_fade']:.3f}, n={result['n_fade']})"
    )
    assert result["checks"]["coin_gross_not_distinguishable_from_zero"], (
        f"coin arm shows a gross edge on a random walk: {result['mean_gross_coin']:+.3f}p "
        f"(se={result['se_gross_coin']:.3f}, n={result['n_coin']})"
    )
    assert result["checks"]["fade_approx_coin"], (
        f"fade ({result['mean_gross_fade']:+.3f}p) diverges from coin "
        f"({result['mean_gross_coin']:+.3f}p) beyond sampling noise — phantom edge in the "
        "fade-signal logic itself"
    )
    assert result["checks"]["net_coin_negative"]
    assert result["checks"]["net_coin_cost_dominated_sane_scale"]
    assert result["pass"]
