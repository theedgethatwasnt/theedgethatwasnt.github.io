#!/usr/bin/env python3
"""
run_is_battery.py — London-Fix Fade primary IS battery.

12 pairs x 3 arms (fade / coin seed=20260708 / continuation), IS window only
(`load_pair_is()` enforces the hard OOS guard — see data_loader.py). Events are found ONCE
per pair (`find_signal_events`) and shared across all 3 arms (`build_trades`) so the arms
trade on IDENTICAL timestamps by construction (R10) — not a post-hoc join. One pair fully
processed before moving to the next; `del` + `gc.collect()` between pairs (CLAUDE.md
memory-safety default; also: never run two heavy backtests in the same process concurrently).

Usage (on Hetzner):
  /root/venv/bin/python3 run_is_battery.py \
      --data-dir /root/work/data/m5_ba --out-dir results
"""
import argparse
import gc
import json
import os

import numpy as np
import pandas as pd

from data_loader import IS_END, PAIRS, load_pair_is, to_utc
from harness import ARMS, DEFAULT_SEED, build_trades, find_signal_events


def run_battery(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    event_stats = []

    for pair in PAIRS:
        print(f"[{pair}] loading IS data...", flush=True)
        df = load_pair_is(pair, data_dir)
        print(f"[{pair}] {len(df)} M5 IS rows, {df['timestamp'].min()} -> {df['timestamp'].max()}",
              flush=True)

        events, stats = find_signal_events(pair, df)
        stats["pair"] = pair
        event_stats.append(stats)
        print(f"[{pair}] events: {stats}", flush=True)

        for arm in ARMS:
            trades = build_trades(events, arm, pair, seed=DEFAULT_SEED)
            for t in trades:
                t["entry_ts"] = to_utc(t["entry_ts"])
                t["exit_ts"] = to_utc(t["exit_ts"])
                t["fix_close_utc"] = to_utc(t["fix_close_utc"])
            all_rows.extend(trades)
            n = len(trades)
            mean_net = np.mean([t["net_pips"] for t in trades]) if n else float("nan")
            mean_gross = np.mean([t["gross_pips"] for t in trades]) if n else float("nan")
            print(f"[{pair}] arm={arm}: n={n} mean_gross={mean_gross:+.3f}p mean_net={mean_net:+.3f}p",
                  flush=True)

        del df, events
        gc.collect()

    trades_df = pd.DataFrame(all_rows)
    out_path = os.path.join(out_dir, "is_battery_trades.csv")
    trades_df.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(trades_df)} rows", flush=True)

    stats_path = os.path.join(out_dir, "event_stats.json")
    with open(stats_path, "w") as f:
        json.dump(event_stats, f, indent=2, default=str)
    print(f"wrote {stats_path}", flush=True)

    # Belt-and-suspenders OOS guard on the assembled output.
    assert pd.Timestamp(trades_df["entry_ts"].max()) < IS_END, "OOS LEAK in entry_ts"
    assert pd.Timestamp(trades_df["exit_ts"].max()) < IS_END + pd.Timedelta(minutes=65), (
        "exit_ts unexpectedly far past IS_END — investigate before trusting downstream gates"
    )
    return trades_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    run_battery(args.data_dir, args.out_dir)
