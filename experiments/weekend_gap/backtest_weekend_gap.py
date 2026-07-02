#!/usr/bin/env python3
"""
Experiment B: Weekend Gap Strategy
===================================
Book Review V2 Campaign — 2026-04-24

Implements the weekend gap strategy from §20.20 of the book, with the
three engineering fixes identified as needed before deployment:
  1. Hard stop loss (50 pips) — protects against BOJ-style tail events
  2. Central bank calendar filter — skip weekends after major CB meetings
  3. Correlation cap — treat all USD-related gaps as 1 risk unit

Book finding (§20.20):
    92.2% fill rate, +9,271 pips across 12 pairs over 5 years (2021-2026)
    Sweet spot: 10-20 pip gaps → 93% win rate, +4.88 pips/trade

Causality:
    Gap = Sunday_open - Friday_close (both known at Sunday open) → causal
    Strategy direction: fade gap → causal
    Stop loss and hold period → outcome, not a feature → correct

Data: data/m5_ohlc/{PAIR}_M5.parquet
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "m5_ohlc"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Pairs & spreads ───────────────────────────────────────────────────
PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY",
    "CHF_JPY", "NZD_JPY", "NZD_USD", "EUR_GBP",
]
PIP_SIZE = {p: (0.01 if "JPY" in p else 0.0001) for p in PAIRS}
SPREAD_PIPS = {
    "EUR_USD": 1.6, "GBP_USD": 1.9, "USD_JPY": 1.7, "AUD_USD": 1.3,
    "EUR_JPY": 2.3, "GBP_JPY": 3.3, "AUD_JPY": 2.1, "CAD_JPY": 2.3,
    "CHF_JPY": 3.5, "NZD_JPY": 2.7, "NZD_USD": 1.5, "EUR_GBP": 1.4,
}

# ── Central bank calendar (known high-risk weekends to skip) ──────────
# Format: (year, month, day) = Friday date of the risky weekend
# Sources: BOJ policy meetings, emergency rate decisions
RISKY_WEEKENDS: set = {
    # BOJ December 2022 yield curve control shock
    (2022, 12, 16),
    # BOJ January 2023 surprise hawkish
    (2023, 1, 20),
    # BOJ March 2023 yield control tweak
    (2023, 3, 10),
    # SVB collapse weekend 2023
    (2023, 3, 10),
    # SNB emergency 2023
    (2023, 3, 17),
    # BOJ July 2023 YCC tweak
    (2023, 7, 28),
    # BOJ October 2023 YCC modification
    (2023, 10, 27),
    # BOJ January 2024 meeting (expectation volatility)
    (2024, 1, 19),
    # BOJ March 2024 negative rate exit
    (2024, 3, 15),
    # BOJ July 2024 hike + carry unwind
    (2024, 7, 26),
    # US Election weekend Nov 2024
    (2024, 11, 1),
    # BOJ January 2025
    (2025, 1, 24),
}

# USD-correlated pairs (same direction on dollar risk-off/on)
USD_PAIRS = {"EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD"}


# ── Data loading ──────────────────────────────────────────────────────

def load_m5_pair(pair: str) -> pd.DataFrame:
    """Load M5 OHLC. File uses integer index with 'timestamp' column."""
    fpath = DATA_DIR / f"{pair}_M5.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"No M5 data for {pair}: {fpath}")
    df = pd.read_parquet(fpath)
    # timestamp is Unix ms or similar; coerce to datetime
    if "timestamp" in df.columns:
        df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("dt").sort_index()
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
    return df[["open", "high", "low", "close"]].astype(float)


# ── Gap detection ─────────────────────────────────────────────────────

def detect_gaps(df: pd.DataFrame, pair: str) -> list:
    """
    Detect all weekend gaps (Friday close → Sunday open).

    Returns list of dicts with gap metadata.
    Uses OANDA M5 bar timestamps. Friday = weekday 4, Sunday = weekday 6.
    """
    pip = PIP_SIZE[pair]
    spread = SPREAD_PIPS[pair]

    gaps = []

    # Group by date to find Friday closes and Sunday opens
    # Friday: weekday()==4, Sunday: weekday()==6 (0=Mon, 6=Sun in pandas)
    idx = df.index

    # Find last bar of each Friday
    fri_mask = idx.dayofweek == 4
    fri_bars = df[fri_mask]

    # Find first bar of each Sunday (FX market reopens Sunday evening UTC)
    sun_mask = idx.dayofweek == 6
    sun_bars = df[sun_mask]

    # For each Friday, find the next Sunday open
    fri_by_week = {}
    for ts in fri_bars.index:
        yr, wk = ts.isocalendar()[:2]
        key = (yr, wk)
        if key not in fri_by_week:
            fri_by_week[key] = ts
        else:
            fri_by_week[key] = max(fri_by_week[key], ts)

    sun_by_week = {}
    for ts in sun_bars.index:
        yr, wk = ts.isocalendar()[:2]
        key = (yr, wk)
        if key not in sun_by_week:
            sun_by_week[key] = ts
        else:
            sun_by_week[key] = min(sun_by_week[key], ts)

    # Pair them up
    for (yr, wk), fri_ts in sorted(fri_by_week.items()):
        # Sunday of the same calendar week or the following week
        for d_wk in [wk, wk + 1] if wk < 52 else [wk]:
            if (yr, d_wk) in sun_by_week:
                sun_ts = sun_by_week[(yr, d_wk)]
                if sun_ts > fri_ts:
                    break
        else:
            continue

        fri_close = df.loc[fri_ts, "close"]
        sun_open = df.loc[sun_ts, "open"]

        gap_pips = (sun_open - fri_close) / pip
        gap_abs = abs(gap_pips)
        gap_dir = 1 if gap_pips > 0 else -1  # +1 = gap up, -1 = gap down

        # Friday date (for calendar filter)
        fri_date = (fri_ts.year, fri_ts.month, fri_ts.day)

        gaps.append({
            "pair": pair,
            "fri_ts": fri_ts,
            "sun_ts": sun_ts,
            "fri_date": fri_date,
            "fri_close": fri_close,
            "sun_open": sun_open,
            "gap_pips": gap_pips,
            "gap_abs": gap_abs,
            "gap_dir": gap_dir,
            "is_risky_weekend": fri_date in RISKY_WEEKENDS,
        })

    return gaps


# ── Trade simulation ──────────────────────────────────────────────────

def simulate_gap_trades(df: pd.DataFrame, gaps: list, pair: str,
                        stop_pips: float = 50.0,
                        max_hold_bars: int = 576,  # 48h of M5 = 576 bars
                        target_buffer: float = 0.10,
                        gap_min: float = 5.0,
                        gap_max: float = 1000.0,
                        apply_calendar_filter: bool = True,
                        apply_stop: bool = True) -> list:
    """
    Simulate fade-the-gap trades.

    Direction: FADE the gap (if gap up → short, target = fri_close * 0.9 buffer)
    Exit: target hit OR stop hit OR max_hold reached
    Spread charged at entry.
    """
    pip = PIP_SIZE[pair]
    spread = SPREAD_PIPS[pair]
    trades = []

    for g in gaps:
        gap_abs = g["gap_abs"]
        if gap_abs < gap_min or gap_abs > gap_max:
            continue
        if apply_calendar_filter and g["is_risky_weekend"]:
            continue

        sun_ts = g["sun_ts"]
        fri_close = g["fri_close"]
        gap_dir = g["gap_dir"]

        # Entry: at Sunday open, fade the gap
        entry_price = g["sun_open"]
        # Spread: long pays ask (sun_open + half_spread), short pays bid (sun_open - half_spread)
        half_spread = (spread * pip) / 2
        entry_adj = entry_price + (-gap_dir) * half_spread  # fade direction pays spread

        # Target: Friday close with 10% buffer (closer than exact close)
        target = fri_close + gap_dir * gap_abs * pip * target_buffer
        # For gap-up (gap_dir=+1), we're SHORT → target is BELOW entry
        # fri_close is below sun_open (gap up), buffer moves target slightly above fri_close
        # target = fri_close + gap_abs * pip * target_buffer (slightly above fri_close for gap-up short)

        # Actual target for the trade
        if gap_dir > 0:  # gap up → short → target is fri_close (below)
            trade_target = fri_close + gap_abs * pip * target_buffer
        else:  # gap down → long → target is fri_close (above)
            trade_target = fri_close - gap_abs * pip * target_buffer

        # Stop loss
        if apply_stop:
            if gap_dir > 0:  # short → stop is ABOVE entry
                stop_price = entry_price + stop_pips * pip
            else:  # long → stop is BELOW entry
                stop_price = entry_price - stop_pips * pip
        else:
            stop_price = None

        # Simulate bar-by-bar
        sun_iloc = df.index.get_loc(sun_ts) if sun_ts in df.index else None
        if sun_iloc is None:
            continue

        result_pips = None
        exit_reason = None
        exit_ts = None

        for bar_offset in range(max_hold_bars + 1):
            bar_iloc = sun_iloc + bar_offset
            if bar_iloc >= len(df):
                break

            bar = df.iloc[bar_iloc]
            bar_high = bar["high"]
            bar_low = bar["low"]

            if bar_offset == 0:
                # Entry bar — already entered at sun_open
                continue

            if gap_dir > 0:  # short trade
                # Check target (price falls to target)
                if bar_low <= trade_target:
                    pips_raw = (entry_adj - trade_target) / pip
                    result_pips = pips_raw - spread  # spread at entry already in entry_adj
                    exit_reason = "target"
                    exit_ts = df.index[bar_iloc]
                    break
                # Check stop
                if stop_price and bar_high >= stop_price:
                    pips_raw = (entry_adj - stop_price) / pip
                    result_pips = pips_raw - spread
                    exit_reason = "stop"
                    exit_ts = df.index[bar_iloc]
                    break
            else:  # long trade
                # Check target (price rises to target)
                if bar_high >= trade_target:
                    pips_raw = (trade_target - entry_adj) / pip
                    result_pips = pips_raw - spread
                    exit_reason = "target"
                    exit_ts = df.index[bar_iloc]
                    break
                # Check stop
                if stop_price and bar_low <= stop_price:
                    pips_raw = (stop_price - entry_adj) / pip
                    result_pips = pips_raw - spread
                    exit_reason = "stop"
                    exit_ts = df.index[bar_iloc]
                    break

        if result_pips is None:
            # Max hold expired — exit at market
            bar_iloc = min(sun_iloc + max_hold_bars, len(df) - 1)
            exit_price = df.iloc[bar_iloc]["close"]
            if gap_dir > 0:
                result_pips = (entry_adj - exit_price) / pip - spread
            else:
                result_pips = (exit_price - entry_adj) / pip - spread
            exit_reason = "timeout"
            exit_ts = df.index[bar_iloc]

        trades.append({
            "pair": pair,
            "fri_ts": g["fri_ts"],
            "sun_ts": sun_ts,
            "exit_ts": exit_ts,
            "gap_pips": round(g["gap_pips"], 2),
            "gap_abs": round(gap_abs, 2),
            "gap_dir": gap_dir,
            "result_pips": round(result_pips, 2),
            "exit_reason": exit_reason,
            "is_risky_weekend": g["is_risky_weekend"],
        })

    return trades


# ── Correlation cap ───────────────────────────────────────────────────

def apply_correlation_cap(all_trades: list, max_usd_risk_units: int = 3) -> list:
    """
    Treat all USD-related pairs gapping in the same direction as correlated.
    Cap total USD-correlated trades per weekend to max_usd_risk_units.

    Non-USD pairs are always included (they're genuinely independent).
    """
    # Group trades by weekend (fri_ts week)
    from collections import defaultdict
    by_weekend = defaultdict(list)
    for t in all_trades:
        key = (t["fri_ts"].year, t["fri_ts"].isocalendar()[1])
        by_weekend[key].append(t)

    filtered = []
    for key, weekend_trades in sorted(by_weekend.items()):
        usd_trades = [t for t in weekend_trades if t["pair"] in USD_PAIRS]
        non_usd_trades = [t for t in weekend_trades if t["pair"] not in USD_PAIRS]

        # Count USD trade directions
        usd_up = [t for t in usd_trades if t["gap_dir"] > 0]
        usd_down = [t for t in usd_trades if t["gap_dir"] < 0]

        # Keep max_usd_risk_units from each correlated group (keep first N by gap size)
        usd_up_sorted = sorted(usd_up, key=lambda t: -t["gap_abs"])[:max_usd_risk_units]
        usd_down_sorted = sorted(usd_down, key=lambda t: -t["gap_abs"])[:max_usd_risk_units]

        # All non-USD trades pass through
        filtered.extend(non_usd_trades)
        filtered.extend(usd_up_sorted)
        filtered.extend(usd_down_sorted)

    return filtered


# ── Main analysis ─────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-pips", type=float, default=50.0)
    parser.add_argument("--gap-min", type=float, default=5.0)
    parser.add_argument("--gap-max", type=float, default=60.0)
    parser.add_argument("--no-stop", action="store_true")
    parser.add_argument("--no-calendar", action="store_true")
    parser.add_argument("--no-correlation-cap", action="store_true")
    parser.add_argument("--max-hold-h", type=float, default=48.0)
    parser.add_argument("--perms", type=int, default=1000)
    parser.add_argument("--oos-frac", type=float, default=0.4)
    args = parser.parse_args()

    max_hold_bars = int(args.max_hold_h * 12)  # 12 M5 bars per hour

    print(f"\n{'='*60}")
    print("Experiment B: Weekend Gap Strategy")
    print(f"Stop: {args.stop_pips}p  Gap range: {args.gap_min}–{args.gap_max}p")
    print(f"Calendar filter: {not args.no_calendar}  Correlation cap: {not args.no_correlation_cap}")
    print(f"{'='*60}\n")

    all_trades = []
    pair_counts = {}

    for pair in PAIRS:
        try:
            df = load_m5_pair(pair)
        except FileNotFoundError as e:
            print(f"  SKIP {pair}: {e}")
            continue

        print(f"  Loading {pair}: {len(df):,} M5 bars "
              f"({df.index[0].date()} → {df.index[-1].date()})")

        gaps = detect_gaps(df, pair)
        trades = simulate_gap_trades(
            df, gaps, pair,
            stop_pips=args.stop_pips,
            max_hold_bars=max_hold_bars,
            gap_min=args.gap_min,
            gap_max=args.gap_max,
            apply_calendar_filter=not args.no_calendar,
            apply_stop=not args.no_stop,
        )
        pair_counts[pair] = len(trades)
        all_trades.extend(trades)
        print(f"    {len(gaps)} gaps detected → {len(trades)} trades after filters")

    if not all_trades:
        print("No trades generated")
        return

    # Apply correlation cap
    if not args.no_correlation_cap:
        n_before = len(all_trades)
        all_trades = apply_correlation_cap(all_trades)
        n_after = len(all_trades)
        print(f"\nCorrelation cap: {n_before} → {n_after} trades ({n_before - n_after} removed)")

    trades_df = pd.DataFrame(all_trades)
    trades_df["sun_ts"] = pd.to_datetime(trades_df["sun_ts"])
    trades_df = trades_df.sort_values("sun_ts")

    print(f"\n{'='*60}")
    print(f"TOTAL TRADES: {len(trades_df)}")
    print(f"Win rate: {(trades_df['result_pips'] > 0).mean()*100:.1f}%")
    print(f"Total pips: {trades_df['result_pips'].sum():.1f}")
    print(f"Mean pips/trade: {trades_df['result_pips'].mean():.2f}")
    print(f"\nBy exit reason:")
    print(trades_df.groupby("exit_reason")["result_pips"].agg(["count", "mean", "sum"]).round(2))

    # By gap size bucket
    bins = [0, 5, 10, 20, 50, 9999]
    labels = ["<5", "5-10", "10-20", "20-50", "50+"]
    trades_df["gap_bucket"] = pd.cut(trades_df["gap_abs"], bins=bins, labels=labels)
    print(f"\nBy gap size:")
    print(trades_df.groupby("gap_bucket", observed=True)["result_pips"].agg(
        ["count", "mean", "sum",
         lambda x: (x > 0).mean()]).rename(
         columns={"<lambda_0>": "win_rate"}).round(3))

    # IS/OOS split
    trades_df["year"] = trades_df["sun_ts"].dt.year
    all_years = sorted(trades_df["year"].unique())
    n_oos_years = max(2, int(len(all_years) * args.oos_frac))
    is_years = all_years[:-n_oos_years]
    oos_years = all_years[-n_oos_years:]

    is_trades = trades_df[trades_df["year"].isin(is_years)]
    oos_trades = trades_df[trades_df["year"].isin(oos_years)]

    print(f"\nIS years: {is_years} ({len(is_trades)} trades)")
    print(f"OOS years: {oos_years} ({len(oos_trades)} trades)")

    def weekly_sharpe(df_t):
        """Weekly returns (one trade per weekend per pair → aggregate by weekend)."""
        df_t = df_t.copy()
        df_t["week"] = df_t["sun_ts"].dt.isocalendar().week.astype(int)
        df_t["iso_yr"] = df_t["sun_ts"].dt.isocalendar().year.astype(int)
        weekly = df_t.groupby(["iso_yr", "week"])["result_pips"].mean()
        if len(weekly) < 4 or weekly.std() < 1e-6:
            return float("nan")
        return float(weekly.mean() / weekly.std(ddof=1) * np.sqrt(52))

    oos_sharpe = weekly_sharpe(oos_trades)
    is_sharpe = weekly_sharpe(is_trades)
    print(f"\nIS Sharpe (weekly): {is_sharpe:.3f}")
    print(f"OOS Sharpe (weekly): {oos_sharpe:.3f}")

    # Walk-forward (year by year on OOS)
    fold_sharpes = []
    for yr in oos_years:
        yr_trades = oos_trades[oos_trades["year"] == yr]
        s = weekly_sharpe(yr_trades)
        fold_sharpes.append(float(s) if not np.isnan(s) else 0.0)
        print(f"  WF fold {yr}: Sharpe={s:.3f}  ({len(yr_trades)} trades)")

    # Permutation test (shuffle trade outcomes across time)
    rng = np.random.default_rng(42)
    oos_results = oos_trades["result_pips"].values
    perm_sharpes = []
    oos_copy = oos_trades.copy()
    for _ in range(args.perms):
        shuffled = rng.permutation(oos_results)
        oos_copy["result_pips"] = shuffled
        perm_sharpes.append(weekly_sharpe(oos_copy))
    perm_sharpes = [s for s in perm_sharpes if not np.isnan(s)]
    p_value = float(np.mean(np.array(perm_sharpes) >= oos_sharpe))
    print(f"\nPermutation p={p_value:.4f}  ({args.perms} shuffles)")

    # Bootstrap CI
    oos_week_pips = oos_trades.groupby(
        [oos_trades["sun_ts"].dt.isocalendar().year.astype(int),
         oos_trades["sun_ts"].dt.isocalendar().week.astype(int)]
    )["result_pips"].mean().values
    boot_sharpes = []
    for _ in range(1000):
        idx = rng.integers(0, len(oos_week_pips), size=len(oos_week_pips))
        s = oos_week_pips[idx].mean() / (oos_week_pips[idx].std(ddof=1) + 1e-12) * np.sqrt(52)
        boot_sharpes.append(s)
    ci_lo = float(np.percentile(boot_sharpes, 2.5))
    ci_hi = float(np.percentile(boot_sharpes, 97.5))
    print(f"Bootstrap CI 95%: [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Drop-one-year
    drop_one = {}
    for yr in oos_years:
        kept = oos_trades[oos_trades["year"] != yr]
        s = weekly_sharpe(kept)
        drop_one[str(int(yr))] = round(float(s), 4)  # str key for JSON
    print(f"Drop-one: {drop_one}")
    all_drop_one_positive = all(s > 0 for s in drop_one.values())

    # 2026 breakdown
    yr2026 = oos_trades[oos_trades["year"] == 2026]
    if len(yr2026) > 0:
        print(f"\n2026 breakdown ({len(yr2026)} trades):")
        print(yr2026.groupby("exit_reason")["result_pips"].agg(["count", "mean", "sum"]).to_string())
        print(f"  Stop rate: {(yr2026['exit_reason'] == 'stop').mean()*100:.1f}%")

    # Gate summary
    g1 = oos_sharpe > 0.5
    g2 = p_value < 0.05
    g3 = ci_lo > 0.0
    g4 = all(s > 0 for s in fold_sharpes)
    g5 = all_drop_one_positive
    n_gates = sum([g1, g2, g3, g4, g5])
    gate_str = "".join(["✅" if g else "❌" for g in [g1, g2, g3, g4, g5]])

    print(f"\n{'='*60}")
    print(f"5-GATE SUMMARY: {gate_str}  ({n_gates}/5 gates passed)")
    print(f"  Gate 1 (point est):  OOS Sharpe={oos_sharpe:.3f} > 0.5  {'✅' if g1 else '❌'}")
    print(f"  Gate 2 (permutation): p={p_value:.4f} < 0.05            {'✅' if g2 else '❌'}")
    print(f"  Gate 3 (bootstrap):  CI lower={ci_lo:.3f} > 0           {'✅' if g3 else '❌'}")
    print(f"  Gate 4 (walk-fwd):   {sum(1 for s in fold_sharpes if s > 0)}/{len(fold_sharpes)} folds positive  {'✅' if g4 else '❌'}")
    print(f"  Gate 5 (drop-one):   all years positive: {all_drop_one_positive}  {'✅' if g5 else '❌'}")
    print(f"\n  VERDICT: {'🟢 DEPLOYABLE (5/5)' if n_gates == 5 else f'🔴 NOT DEPLOYABLE ({n_gates}/5)'}")

    # Save results
    summary = {
        "config": {
            "stop_pips": args.stop_pips,
            "gap_min": args.gap_min,
            "gap_max": args.gap_max,
            "apply_stop": not args.no_stop,
            "apply_calendar": not args.no_calendar,
            "apply_correlation_cap": not args.no_correlation_cap,
        },
        "total_trades": len(trades_df),
        "oos_sharpe": round(oos_sharpe, 4),
        "is_sharpe": round(is_sharpe, 4),
        "oos_total_pips": round(oos_trades["result_pips"].sum(), 2),
        "oos_mean_pips": round(oos_trades["result_pips"].mean(), 2),
        "oos_win_rate": round((oos_trades["result_pips"] > 0).mean(), 4),
        "perm_p": round(p_value, 4),
        "boot_ci_lo": round(ci_lo, 4),
        "boot_ci_hi": round(ci_hi, 4),
        "wf_fold_sharpes": fold_sharpes,
        "drop_one": drop_one,
        "gates_passed": n_gates,
        "gate_str": gate_str,
        "gate_detail": {"g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5},
    }

    out_path = RESULTS_DIR / "weekend_gap_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    trades_df.to_csv(RESULTS_DIR / "weekend_gap_trades.csv", index=False)
    print(f"\nSaved: {out_path}")
    print(f"Saved trades: {RESULTS_DIR / 'weekend_gap_trades.csv'}")

    return summary


if __name__ == "__main__":
    main()
