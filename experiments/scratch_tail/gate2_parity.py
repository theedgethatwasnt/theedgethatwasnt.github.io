#!/usr/bin/env python3
"""
gate2_parity.py — Gate 2 (R7 parity, BLOCKING): replay the live paper window
(2026-06-15 -> present) through the harness (arm A: identical to the deployed
`services/strategy_sma_scratch_paper/main.py`, no stop, no overlay) and compare against the
`sma_scratch_%` paper trades recorded in the VPS trades.duckdb trail.

Per PREREGISTRATION.md gate 2: trade-count tolerance ±10%, per-trade expectancy tolerance
±1.0p. This is the ONE place in the whole battery allowed to touch 2026-06-15+ data — never
for arm evaluation (that is is_data.load_pair_is()'s hard-sealed IS window only).

Run on Hetzner:
  /root/venv/bin/python3 gate2_parity.py \
      --parity-data-dir /root/work/data/parity_m5_ba \
      --trades-db /root/work/trades_2026-07-06.duckdb \
      --out-dir results
"""
import argparse
import json
import os

import duckdb
import numpy as np
import pandas as pd

from harness import ARMS, simulate_portfolio
from signal import PAIRS

TOL_COUNT_FRAC = 0.10
TOL_EXPECTANCY_PIPS = 1.0


def load_live_trades(db_path):
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        "SELECT pair, direction, entry_price, exit_price, entry_time, exit_time, pnl_pips, "
        "exit_reason, hours_held, mfe_pips, mae_pips, label "
        "FROM trades WHERE is_paper=true AND label LIKE 'sma_scratch_%' AND exit_price IS NOT NULL "
        "ORDER BY entry_time"
    ).fetchdf()
    con.close()
    return df


def load_parity_m5(data_dir, pair):
    path = os.path.join(data_dir, f"{pair}_M5_BA.parquet")
    df = pd.read_parquet(path)
    return df.sort_values("timestamp").reset_index(drop=True)


def run_harness_over_parity_window(data_dir):
    pairs_m5 = {p: load_parity_m5(data_dir, p) for p in PAIRS}
    result = simulate_portfolio(pairs_m5, ARMS["A"])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-data-dir", required=True)
    ap.add_argument("--trades-db", required=True)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    live = load_live_trades(args.trades_db)
    print(f"live sma_scratch_% closed trades: {len(live)}", flush=True)
    print(live.groupby("pair")["pnl_pips"].agg(["count", "mean", "sum"]).to_string(), flush=True)

    result = run_harness_over_parity_window(args.parity_data_dir)
    harness_trades = pd.DataFrame(result["trades"])
    if len(harness_trades):
        # harness entry_ts/exit_ts descend from _build_master_grid's DatetimeIndex.to_numpy(),
        # which strips tz (numpy datetime64 has no tz concept) leaving naive-but-UTC values;
        # pd.to_datetime(..., utc=True) re-attaches UTC so comparisons against the (tz-aware)
        # live-trail timestamps below don't raise/silently miscompare.
        harness_trades["entry_ts"] = pd.to_datetime(harness_trades["entry_ts"], utc=True)
        harness_trades["exit_ts"] = pd.to_datetime(harness_trades["exit_ts"], utc=True)
    # Restrict the harness replay to trades whose ENTRY falls within the live trail's own
    # observed window (the harness warms up from 2026-06-10, several days before the live
    # trail's earliest recorded entry, but the live trail may also have started slightly
    # after the harness's own first-eligible signal — align on the live trail's actual span
    # so both sides are compared over the identical clock window).
    live_start = pd.Timestamp(live["entry_time"].min()).tz_localize("UTC")
    live_end = pd.Timestamp(live["entry_time"].max()).tz_localize("UTC")
    if len(harness_trades):
        mask = (harness_trades["entry_ts"] >= live_start) & (harness_trades["entry_ts"] <= live_end)
        harness_window = harness_trades[mask].copy()
    else:
        harness_window = harness_trades

    print(f"\nharness (arm A) replayed trades in the live window [{live_start}, {live_end}]: "
          f"{len(harness_window)}", flush=True)
    if len(harness_window):
        print(harness_window.groupby("pair")["net_pips"].agg(["count", "mean", "sum"]).to_string(), flush=True)

    n_live = len(live)
    n_harness = len(harness_window)
    count_ratio = (n_harness / n_live) if n_live else float("nan")
    count_pass = n_live > 0 and abs(n_harness - n_live) <= TOL_COUNT_FRAC * n_live

    live_expectancy = float(live["pnl_pips"].mean()) if n_live else float("nan")
    harness_expectancy = float(harness_window["net_pips"].mean()) if n_harness else float("nan")
    expectancy_diff = (harness_expectancy - live_expectancy) if (n_live and n_harness) else float("nan")
    expectancy_pass = n_live > 0 and n_harness > 0 and abs(expectancy_diff) <= TOL_EXPECTANCY_PIPS

    overall_pass = count_pass and expectancy_pass

    summary = {
        "n_live": n_live, "n_harness": n_harness, "count_ratio": count_ratio,
        "count_pass": count_pass, "count_tolerance_frac": TOL_COUNT_FRAC,
        "live_expectancy_pips": live_expectancy, "harness_expectancy_pips": harness_expectancy,
        "expectancy_diff_pips": expectancy_diff, "expectancy_pass": expectancy_pass,
        "expectancy_tolerance_pips": TOL_EXPECTANCY_PIPS,
        "gate2_pass": overall_pass,
        "live_window_start": str(live_start), "live_window_end": str(live_end),
    }

    per_pair_rows = []
    for pair in PAIRS:
        lp = live[live["pair"] == pair]
        hp = harness_window[harness_window["pair"] == pair] if len(harness_window) else harness_window
        per_pair_rows.append({
            "pair": pair,
            "n_live": len(lp), "live_mean": float(lp["pnl_pips"].mean()) if len(lp) else float("nan"),
            "n_harness": len(hp), "harness_mean": float(hp["net_pips"].mean()) if len(hp) else float("nan"),
        })
    per_pair = pd.DataFrame(per_pair_rows)

    print("\n=== GATE 2 SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    print("\n=== per-pair ===")
    print(per_pair.to_string(index=False))

    with open(os.path.join(args.out_dir, "gate2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    per_pair.to_csv(os.path.join(args.out_dir, "gate2_per_pair.csv"), index=False)
    live.to_csv(os.path.join(args.out_dir, "gate2_live_trades.csv"), index=False)
    if len(harness_window):
        harness_window.to_csv(os.path.join(args.out_dir, "gate2_harness_trades.csv"), index=False)

    print(f"\nGATE 2: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
