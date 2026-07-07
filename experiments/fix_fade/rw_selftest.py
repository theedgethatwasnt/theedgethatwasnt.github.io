#!/usr/bin/env python3
"""
rw_selftest.py — Gate 1 of PREREGISTRATION.md's "Gates" list: "RW self-test" (random-walk
harness self-test). On a true random walk, NO non-anticipating direction rule (fade, coin, or
continuation) can have a real GROSS edge (optional-stopping: expected gross P&L of any
non-anticipating rule is zero on a martingale). This module builds the synthetic RW, runs all
3 arms through the real harness, and returns the numbers + pass/fail — shared by
test_harness.py (asserts on it) and make_summary.py (reports it), so there is exactly one
place this self-test's logic lives (R6).

Calibration (documented, not arbitrary — same spirit as multiday_contrarian/test_harness.py's
step_pip_mult note): D_pips (PREREGISTRATION.md's mid(16:00)-mid(15:00), a 12-bar cumulative
return) has std = pip * step_pip_mult * sqrt(12) for i.i.d. Gaussian per-bar steps. With
step_pip_mult=1.5, std(D) ~= 5.2 pips, so P(|D|>=5) ~= 34% of fix-days — enough signal events
(~250+ over 3 synthetic years) for the self-test's sampling-error bars to be tight without
making every single day a "signal" (which would just be testing a degenerate always-on rule).
"""
import numpy as np
import pandas as pd

from data_loader import pip_of
from harness import DEFAULT_SEED, build_trades, find_signal_events


def make_rw_m5(n_days, seed, pair="EUR_USD", start="2021-01-04T00:00:00Z", spread_pips=1.5,
               step_pip_mult=1.5):
    """Continuous (no weekend gaps — irrelevant for this self-test's purpose) synthetic M5
    random walk. `timestamp` = bar OPEN time, matching the real data convention."""
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
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low, "close": close,
        "bid_c": bid_c, "ask_c": ask_c,
    })


def _stats(trades, col):
    v = np.array([t[col] for t in trades], dtype=float)
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return float(v.mean()), float(se), n


def run_rw_selftest(n_days=365 * 3, seed=42, pair="EUR_USD"):
    df = make_rw_m5(n_days=n_days, seed=seed, pair=pair)
    events, event_stats = find_signal_events(pair, df)
    fade_trades = build_trades(events, "fade", pair, seed=DEFAULT_SEED)
    coin_trades = build_trades(events, "coin", pair, seed=DEFAULT_SEED)

    mean_g_fade, se_g_fade, n_fade = _stats(fade_trades, "gross_pips")
    mean_g_coin, se_g_coin, n_coin = _stats(coin_trades, "gross_pips")
    mean_net_coin, _, _ = _stats(coin_trades, "net_pips")
    mean_spread_coin, _, _ = _stats(coin_trades, "spread_rt_pips")

    se_diff = float(np.sqrt(se_g_fade ** 2 + se_g_coin ** 2))

    checks = {
        "enough_trades": n_fade >= 20 and n_coin >= 20,
        "fade_gross_not_distinguishable_from_zero": abs(mean_g_fade) < 3 * se_g_fade,
        "coin_gross_not_distinguishable_from_zero": abs(mean_g_coin) < 3 * se_g_coin,
        "fade_approx_coin": abs(mean_g_fade - mean_g_coin) < 4 * se_diff,
        "net_coin_negative": mean_net_coin < 0,
        "net_coin_cost_dominated_sane_scale": 0 < mean_spread_coin < 10,
    }
    passed = all(checks.values())

    return {
        "pass": bool(passed), "checks": checks, "event_stats": event_stats,
        "n_fade": n_fade, "n_coin": n_coin,
        "mean_gross_fade": mean_g_fade, "se_gross_fade": se_g_fade,
        "mean_gross_coin": mean_g_coin, "se_gross_coin": se_g_coin,
        "mean_net_coin": mean_net_coin, "mean_spread_coin": mean_spread_coin,
    }


if __name__ == "__main__":
    import argparse
    import json
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    result = run_rw_selftest()
    print(json.dumps(result, indent=2, default=str))
    with open(os.path.join(args.out_dir, "rw_selftest_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    if not result["pass"]:
        raise SystemExit("RW self-test FAILED — see checks above")
