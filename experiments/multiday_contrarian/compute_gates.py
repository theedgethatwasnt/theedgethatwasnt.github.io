#!/usr/bin/env python3
"""
compute_gates.py — Task A5 gate table (PREREGISTRATION.md gates 3-6; gate 1 done in
test_harness.py, gate 2 done+recorded in Amendment 1). Reads results/is_battery_trades.csv
(written by run_is_battery.py) and produces:

  results/is_per_pair_summary.csv   — per pair x arm: n, WR, gross, net_base/spread1.5/carry2.0, timeout_share
  results/is_portfolio_summary.csv  — pooled-across-pairs, same columns, per arm
  results/gate_table.csv            — gate 3/4/5/6 PASS/FAIL + the numbers behind each

Gate definitions (verbatim from PREREGISTRATION.md "Gates before ... OOS"):
  3. IS net expectancy > 0 at 1.0x costs, and > coin arm.       (portfolio-pooled, signal arm)
  4. Walk-forward: 3 IS thirds, net-positive in >=2 of 3, none < -2p/trade.
  5. MC: day-block bootstrap P(net <= 0) < 0.05 on IS.          (2000 resamples by UTC day)
  6. Breadth: >= 6/12 pairs gross-positive IS.                  (signal arm, per-pair gross mean)

All "portfolio" numbers below pool trades across the 12 pairs at the trade level (equal
weight per trade, not per-pair-averaged) — the same convention the primary battery uses
throughout this program's memory entries (e.g. "portfolio p/d" sums).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from is_data import IS_END, PAIRS

N_THIRDS = 3
N_BOOT = 2000
BOOT_SEED = 20260706


def per_group_stats(df):
    n = len(df)
    if n == 0:
        return dict(n=0, wr=float("nan"), gross=float("nan"), net_base=float("nan"),
                    net_spread1p5=float("nan"), net_carry2p0=float("nan"), timeout_share=float("nan"))
    wr = float((df["net_base"] > 0).mean())
    timeout_share = float(df["exit_reason"].isin(["timecap", "data_end"]).mean())
    return dict(
        n=n, wr=wr,
        gross=float(df["gross_pips"].mean()),
        net_base=float(df["net_base"].mean()),
        net_spread1p5=float(df["net_spread1p5"].mean()),
        net_carry2p0=float(df["net_carry2p0"].mean()),
        timeout_share=timeout_share,
    )


def build_per_pair_and_portfolio(trades_df):
    rows = []
    for pair in PAIRS:
        for arm in ("signal", "coin", "continuation"):
            sub = trades_df[(trades_df["pair"] == pair) & (trades_df["arm"] == arm)]
            rows.append({"pair": pair, "arm": arm, **per_group_stats(sub)})
    per_pair = pd.DataFrame(rows)

    port_rows = []
    for arm in ("signal", "coin", "continuation"):
        sub = trades_df[trades_df["arm"] == arm]
        port_rows.append({"pair": "PORTFOLIO", "arm": arm, **per_group_stats(sub)})
    portfolio = pd.DataFrame(port_rows)
    return per_pair, portfolio


def walk_forward_thirds(signal_df):
    """Split the IS window [min entry_ts, IS_END) into N_THIRDS equal-DURATION chunks
    (time-based, not count-based — pairs' trade counts differ) and compute pooled
    signal-arm net_base mean in each."""
    entry_ts = pd.to_datetime(signal_df["entry_ts"])
    is_start = entry_ts.min()
    is_end = IS_END
    edges = pd.date_range(is_start, is_end, periods=N_THIRDS + 1)
    thirds = []
    for i in range(N_THIRDS):
        lo, hi = edges[i], edges[i + 1]
        mask = (entry_ts >= lo) & (entry_ts < hi) if i < N_THIRDS - 1 else (entry_ts >= lo) & (entry_ts <= hi)
        sub = signal_df[mask]
        n = len(sub)
        mean_net = float(sub["net_base"].mean()) if n else float("nan")
        thirds.append({"third": i + 1, "start": str(lo), "end": str(hi), "n": n, "mean_net_base": mean_net})
    return thirds


def day_block_bootstrap(signal_df, n_boot=N_BOOT, seed=BOOT_SEED):
    """Block bootstrap by UTC calendar day of entry_ts: resample DAYS with replacement
    (not individual trades), pool all trades that fell on the resampled days, compute the
    pooled mean net_base per resample. Returns (p_le_zero, boot_means)."""
    df = signal_df.copy()
    df["entry_day"] = pd.to_datetime(df["entry_ts"]).dt.date
    days = df["entry_day"].unique()
    day_to_vals = {d: df.loc[df["entry_day"] == d, "net_base"].values for d in days}
    n_days = len(days)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample_days = rng.choice(days, size=n_days, replace=True)
        vals = np.concatenate([day_to_vals[d] for d in sample_days])
        boot_means[b] = vals.mean() if len(vals) else np.nan
    p_le_zero = float(np.mean(boot_means <= 0))
    return p_le_zero, boot_means


def compute_all_gates(trades_df):
    per_pair, portfolio = build_per_pair_and_portfolio(trades_df)

    port_signal = trades_df[trades_df["arm"] == "signal"]
    port_coin = trades_df[trades_df["arm"] == "coin"]

    # Gate 3
    net_signal = float(port_signal["net_base"].mean())
    net_coin = float(port_coin["net_base"].mean())
    gate3_pass = bool((net_signal > 0) and (net_signal > net_coin))

    # Gate 4
    thirds = walk_forward_thirds(port_signal)
    n_pos = sum(1 for t in thirds if not np.isnan(t["mean_net_base"]) and t["mean_net_base"] > 0)
    none_catastrophic = all(np.isnan(t["mean_net_base"]) or t["mean_net_base"] >= -2.0 for t in thirds)
    gate4_pass = bool(n_pos >= 2 and none_catastrophic)

    # Gate 5
    p_le_zero, boot_means = day_block_bootstrap(port_signal)
    gate5_pass = bool(p_le_zero < 0.05)

    # Gate 6
    per_pair_signal = per_pair[per_pair["arm"] == "signal"]
    n_gross_pos = int((per_pair_signal["gross"] > 0).sum())
    gate6_pass = bool(n_gross_pos >= 6)

    gate_table = pd.DataFrame([
        {"gate": 3, "name": "IS net>0 @1.0x AND > coin", "pass": gate3_pass,
         "detail": f"signal_net={net_signal:+.3f}p coin_net={net_coin:+.3f}p"},
        {"gate": 4, "name": "WF 3 thirds >=2/3 net-positive, none<-2p", "pass": gate4_pass,
         "detail": f"n_pos={n_pos}/3 thirds={[round(t['mean_net_base'],3) for t in thirds]}"},
        {"gate": 5, "name": "day-block bootstrap P(net<=0)<0.05", "pass": gate5_pass,
         "detail": f"P(net<=0)={p_le_zero:.4f} boot_mean={boot_means.mean():+.3f}p n_boot={N_BOOT}"},
        {"gate": 6, "name": "breadth >=6/12 pairs gross-positive", "pass": gate6_pass,
         "detail": f"n_gross_pos={n_gross_pos}/12"},
    ])

    extra = {
        "portfolio_signal_net_base": net_signal,
        "portfolio_coin_net_base": net_coin,
        "portfolio_continuation_net_base": float(trades_df.loc[trades_df["arm"] == "continuation", "net_base"].mean()),
        "wf_thirds": thirds,
        "mc_p_le_zero": p_le_zero,
        "mc_boot_mean": float(boot_means.mean()),
        "mc_boot_ci_2p5": float(np.percentile(boot_means, 2.5)),
        "mc_boot_ci_97p5": float(np.percentile(boot_means, 97.5)),
        "breadth_n_gross_pos": n_gross_pos,
    }
    return per_pair, portfolio, gate_table, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-csv", default="results/is_battery_trades.csv")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    trades_df = pd.read_csv(args.trades_csv)
    for col in ("signal_ts", "entry_ts", "exit_ts"):
        trades_df[col] = pd.to_datetime(trades_df[col], utc=True)
    per_pair, portfolio, gate_table, extra = compute_all_gates(trades_df)

    per_pair.to_csv(os.path.join(args.out_dir, "is_per_pair_summary.csv"), index=False)
    portfolio.to_csv(os.path.join(args.out_dir, "is_portfolio_summary.csv"), index=False)
    gate_table.to_csv(os.path.join(args.out_dir, "gate_table.csv"), index=False)
    with open(os.path.join(args.out_dir, "gate_extra.json"), "w") as f:
        json.dump(extra, f, indent=2, default=str)

    print(gate_table.to_string(index=False))
    print(json.dumps(extra, indent=2, default=str))


if __name__ == "__main__":
    main()
