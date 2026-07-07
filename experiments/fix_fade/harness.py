#!/usr/bin/env python3
"""
harness.py — London-Fix Fade (PREREGISTRATION.md, LOCKED 2026-07-07): signal detection +
3-arm backtest engine. One code path (R6): `find_signal_events()` is the entry-condition
logic; `build_trades()` turns an event list + arm into trade rows on IDENTICAL timestamps
(R10) — only the assigned direction differs between arms.

Rule (verbatim from PREREGISTRATION.md):
  D = mid(16:00 Ldn) - mid(15:00 Ldn)   [close-to-close over the 60 minutes before the fix]
  Enter AGAINST D at the first M5 open after the fix bar closes, |D| >= 5 pips required.
  Exit 60 minutes later at close. No TP/SL — bounded by the 1-hour cap.

Bar-index arithmetic (fixtime.py docstring: `timestamp` = bar OPEN time, bar closes at
`timestamp + 5min`; M5 bars are 5 min apart so 60 min = 12 bars):
  fix_close_utc  = fixtime.london_fix_close_utc(date)             # the fix instant, 16:00 Ldn
  fix_bar        : timestamp == fix_close_utc - 5min               (closes AT the fix)
  prior_bar      : timestamp == fix_close_utc - 65min               (closes 60min before the fix,
                                                                       i.e. "mid(15:00 Ldn)")
  entry_bar      : timestamp == fix_close_utc                      (opens the instant the fix
                                                                       bar closes — "first M5
                                                                       open after the fix bar
                                                                       closes")
  exit_bar       : entry_idx + 11                                  (closes at fix_close_utc +
                                                                       60min — "60 minutes later
                                                                       at close"; entry_idx+0 is
                                                                       the 1st of the 12 bars
                                                                       spanning the hour, so the
                                                                       12th/last is entry_idx+11)
A day is dropped (no trade, no signal) if any of fix_bar/prior_bar/entry_bar is missing from
the grid (weekend/holiday/data gap) or exit_bar would run past the end of the loaded array —
these are diagnostic counts, not silently absorbed (see `find_signal_events`'s return stats).

Cost model (locked; no sensitivity sweep — PREREGISTRATION.md says "NO sweeps"):
  spread_rt_pips = (entry-bar spread + exit-bar spread) / 2, in pips — one full round-trip
    charged as half at each leg (R3; same convention as multiday_contrarian/harness.py).
  net_pips = gross_pips - spread_rt_pips   (no carry: max hold is 60 minutes, never crosses a
    daily rollover — 16:00 London is 15:00-16:00 UTC, hours away from OANDA's ~21-22:00 UTC
    end-of-day rollover).

Excursion (no SL — bounded by the 60-min cap; reported per-trade, not used for exit): scanned
on mid high/low across the 12 entry-to-exit-inclusive bars, in the trade's own direction.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_loader import pip_of
from fixtime import LONDON, is_last_business_day_of_month, london_fix_close_utc

ARMS = ("fade", "coin", "continuation")
DEFAULT_SEED = 20260708          # PREREGISTRATION.md: "coin (seed 20260708)"
THRESH_PIPS = 5.0                # PREREGISTRATION.md: "single |D|>=5 pips threshold"
DRIFT_BARS = 12                  # 60 min / 5 min: prior_bar is 12 bars before fix_bar
HOLD_BARS = 12                   # 60 min / 5 min: exit_bar is entry_idx + (HOLD_BARS - 1)


@dataclass
class SignalEvent:
    pair: str
    date: object              # python date (London-local calendar date of the fix)
    fix_close_utc: pd.Timestamp
    D_pips: float
    entry_ts: pd.Timestamp
    entry_px: float
    exit_ts: pd.Timestamp
    exit_px: float
    entry_spread_pips: float
    exit_spread_pips: float
    mfe_if_long_pips: float   # both non-negative; per-arm mfe/mae derived from these + direction
    mae_if_long_pips: float
    month_end: bool


def find_signal_events(pair, df):
    """Scan `df` (one pair's IS-only M5 BA dataframe, sorted, tz-aware UTC `timestamp`) for
    every London-fix day whose |D| >= THRESH_PIPS, with a complete bar grid (fix/prior/entry
    bars present, exit bar in range). Returns (events: list[SignalEvent], stats: dict) — stats
    counts candidate days and every drop reason, for transparency (not silently absorbed)."""
    pip = pip_of(pair)
    ts = df["timestamp"].values  # datetime64[ns] (UTC instant, tz stripped by .values)
    n = len(ts)
    open_ = df["open"].values
    close_ = df["close"].values
    high_ = df["high"].values
    low_ = df["low"].values
    bid_c = df["bid_c"].values
    ask_c = df["ask_c"].values

    ts_start = pd.Timestamp(ts[0]).tz_localize("UTC")
    ts_end = pd.Timestamp(ts[-1]).tz_localize("UTC")
    london_start = ts_start.tz_convert(LONDON).date()
    london_end = ts_end.tz_convert(LONDON).date()
    dates = pd.date_range(london_start, london_end, freq="D")

    def exact_idx(target_ts_utc):
        """Return the row index whose timestamp EXACTLY equals target_ts_utc (as np.datetime64
        UTC instant), or None if absent (weekend/holiday/gap) — never a nearest-match guess."""
        t64 = target_ts_utc.tz_convert("UTC").to_datetime64()
        i = np.searchsorted(ts, t64, side="left")
        if i < n and ts[i] == t64:
            return int(i)
        return None

    events = []
    stats = {"n_candidate_days": len(dates), "n_missing_grid": 0, "n_exit_out_of_range": 0,
              "n_below_threshold": 0, "n_signal": 0}

    for d in dates:
        fix_close = london_fix_close_utc(d)
        fix_idx = exact_idx(fix_close - pd.Timedelta(minutes=5))
        prior_idx = exact_idx(fix_close - pd.Timedelta(minutes=5 + DRIFT_BARS * 5))
        entry_idx = exact_idx(fix_close)
        if fix_idx is None or prior_idx is None or entry_idx is None:
            stats["n_missing_grid"] += 1
            continue
        exit_idx = entry_idx + (HOLD_BARS - 1)
        if exit_idx >= n:
            stats["n_exit_out_of_range"] += 1
            continue

        D_pips = (close_[fix_idx] - close_[prior_idx]) / pip
        if abs(D_pips) < THRESH_PIPS - 1e-9:  # epsilon guards the ">=" boundary against fp noise
            stats["n_below_threshold"] += 1
            continue

        entry_ts = pd.Timestamp(ts[entry_idx]).tz_localize("UTC")
        exit_ts = entry_ts + pd.Timedelta(minutes=60)
        entry_px = float(open_[entry_idx])
        exit_px = float(close_[exit_idx])
        entry_spread_pips = float(ask_c[entry_idx] - bid_c[entry_idx]) / pip
        exit_spread_pips = float(ask_c[exit_idx] - bid_c[exit_idx]) / pip

        window_high = float(np.max(high_[entry_idx:exit_idx + 1]))
        window_low = float(np.min(low_[entry_idx:exit_idx + 1]))
        mfe_if_long_pips = (window_high - entry_px) / pip
        mae_if_long_pips = (entry_px - window_low) / pip

        events.append(SignalEvent(
            pair=pair, date=d.date(), fix_close_utc=fix_close, D_pips=float(D_pips),
            entry_ts=entry_ts, entry_px=entry_px, exit_ts=exit_ts, exit_px=exit_px,
            entry_spread_pips=entry_spread_pips, exit_spread_pips=exit_spread_pips,
            mfe_if_long_pips=mfe_if_long_pips, mae_if_long_pips=mae_if_long_pips,
            month_end=is_last_business_day_of_month(d),
        ))
        stats["n_signal"] += 1

    return events, stats


def _direction_for_arm(event, arm, rng):
    raw_dir = -1 if event.D_pips > 0 else 1   # fade = AGAINST D
    if arm == "fade":
        return raw_dir
    if arm == "continuation":
        return -raw_dir
    # coin
    return 1 if rng.random() < 0.5 else -1


def build_trades(events, arm, pair, seed=DEFAULT_SEED):
    """Turn a signal-event list into trade rows for one arm. Events are processed in
    chronological order (as returned by find_signal_events) so a fresh `np.random.default_rng
    (seed)` stream draws the i-th coin-flip for the i-th event, deterministically (same
    convention as multiday_contrarian/harness.py's simulate_pair)."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    pip = pip_of(pair)
    rng = np.random.default_rng(seed)
    trades = []
    for ev in events:
        direction = _direction_for_arm(ev, arm, rng)
        gross_pips = direction * (ev.exit_px - ev.entry_px) / pip
        spread_rt_pips = (ev.entry_spread_pips + ev.exit_spread_pips) / 2.0
        net_pips = gross_pips - spread_rt_pips
        if direction > 0:
            mfe_pips, mae_pips = ev.mfe_if_long_pips, ev.mae_if_long_pips
        else:
            mfe_pips, mae_pips = ev.mae_if_long_pips, ev.mfe_if_long_pips
        trades.append({
            "pair": pair, "arm": arm, "date": ev.date, "direction": direction,
            "D_pips": ev.D_pips, "fix_close_utc": ev.fix_close_utc,
            "entry_ts": ev.entry_ts, "entry_px": ev.entry_px,
            "exit_ts": ev.exit_ts, "exit_px": ev.exit_px,
            "gross_pips": gross_pips, "spread_rt_pips": spread_rt_pips, "net_pips": net_pips,
            "mfe_pips": mfe_pips, "mae_pips": mae_pips, "month_end": ev.month_end,
        })
    return trades


def simulate_pair(pair, df, arm="fade", seed=DEFAULT_SEED):
    """Convenience wrapper: find events then build trades for one (pair, arm). Prefer calling
    `find_signal_events` once and `build_trades` per-arm directly when you need all 3 arms on
    the SAME event list (guarantees identical timestamps by construction, R10) — this wrapper
    re-scans events every call, which is fine for tests/one-off use but wasteful in the battery
    runner."""
    events, _ = find_signal_events(pair, df)
    return build_trades(events, arm, pair, seed=seed)
