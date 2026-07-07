#!/usr/bin/env python3
"""
run_is_battery.py — scratch_tail primary IS battery. Loads all 6 pairs' IS-window M5-BA data
(is_data.load_pair_is() enforces the hard OOS guard), runs the full pre-declared arm set via
harness.run_battery() (PHASE1 ungated references A/E/coin_A/coin_E, then PHASE2 gated arms
B/C/D/coin_D/coin_overlay using their declared reference's blocked-state timeline), and writes
per-trade CSVs + open-at-end + floating-eq-trace artifacts for compute_gates.py / make_summary.py.

Unlike multiday_contrarian's run_is_battery.py (which processes pairs independently, one at a
time, `del`+`gc.collect()` between them — CLAUDE.md memory-safety default), scratch_tail's
overlay arms (B/C/D/coin_D/coin_overlay) are inherently PORTFOLIO-JOINT (the whole point of the
experiment: entries in one pair can be blocked by a reference run's cross-pair equity trend),
so all 6 pairs must be loaded together for one run_battery() call. Total IS data for 6 pairs
(~410k M5 bars/pair * 6) is a few hundred MB — well within the box's 15GB RAM — so this is a
deliberate, documented exception to the "one pair at a time" default, not an oversight.

Usage (on Hetzner):
  /root/venv/bin/python3 run_is_battery.py --data-dir /root/work/data/m5_ba --out-dir results
"""
import argparse
import gc
import os

import pandas as pd

from harness import ARMS, PHASE1, PHASE2, run_battery
from is_data import IS_END, load_pair_is
from signal import PAIRS


def run(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"loading IS data for {len(PAIRS)} pairs from {data_dir} ...", flush=True)
    pairs_m5 = {}
    for pair in PAIRS:
        df = load_pair_is(pair, data_dir)
        print(f"  [{pair}] {len(df)} IS rows, {df['timestamp'].min()} -> {df['timestamp'].max()}", flush=True)
        pairs_m5[pair] = df
    gc.collect()

    print("\nrunning full arm battery (this can take a while — 6 pairs x 8 arm-runs) ...", flush=True)
    results = run_battery(pairs_m5, spread_mult=1.0, markup_mult=1.0)

    all_trades = []
    all_open = []
    all_floating = []
    all_blocked = []
    for name in PHASE1 + PHASE2:
        res = results[name]
        for t in res["trades"]:
            t = dict(t)
            t["arm"] = name
            all_trades.append(t)
        for o in res["open_at_end"]:
            o = dict(o)
            o["arm"] = name
            all_open.append(o)
        for ts, val in res["floating_eq_trace"]:
            all_floating.append({"arm": name, "ts": ts, "floating_eq_pips": val})
        for ts, blocked in res["blocked_trace"]:
            all_blocked.append({"arm": name, "ts": ts, "blocked": bool(blocked)})
        n = len(res["trades"])
        mean_net = (sum(t["net_pips"] for t in res["trades"]) / n) if n else float("nan")
        n_open = len(res["open_at_end"])
        print(f"[{name}] n={n} mean_net={mean_net:+.3f}p open_at_end={n_open}", flush=True)

    trades_df = pd.DataFrame(all_trades)
    open_df = pd.DataFrame(all_open)
    floating_df = pd.DataFrame(all_floating)
    blocked_df = pd.DataFrame(all_blocked)

    trades_path = os.path.join(out_dir, "is_battery_trades.csv")
    open_path = os.path.join(out_dir, "is_battery_open_at_end.csv")
    floating_path = os.path.join(out_dir, "is_battery_floating_eq.csv")
    blocked_path = os.path.join(out_dir, "is_battery_blocked_trace.csv")
    trades_df.to_csv(trades_path, index=False)
    open_df.to_csv(open_path, index=False)
    floating_df.to_csv(floating_path, index=False)
    blocked_df.to_csv(blocked_path, index=False)
    print(f"\nwrote {trades_path}: {len(trades_df)} rows", flush=True)
    print(f"wrote {open_path}: {len(open_df)} rows", flush=True)
    print(f"wrote {floating_path}: {len(floating_df)} rows", flush=True)
    print(f"wrote {blocked_path}: {len(blocked_df)} rows", flush=True)

    # OOS guard on the assembled output (belt-and-suspenders; every timestamp used descends
    # from load_pair_is()'s already-filtered arrays).
    if len(trades_df):
        assert pd.Timestamp(trades_df["entry_ts"].max()) < IS_END, "OOS LEAK in entry_ts"
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    run(args.data_dir, args.out_dir)
