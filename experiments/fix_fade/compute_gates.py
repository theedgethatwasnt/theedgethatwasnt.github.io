#!/usr/bin/env python3
"""
compute_gates.py — London-Fix Fade gate table (PREREGISTRATION.md "Gates":
"RW self-test; IS net>0 & >coin; WF thirds >=2/3; breadth >=6/12 gross; user gate; OOS once.")

Gate 1 (RW self-test) is a pytest check (test_harness.py::test_synthetic_rw_no_phantom_edge),
not computed here — its pass/fail is recorded into gate_table.csv by the caller (run_all.sh)
after the test suite runs. Gates 2-4 below are computed from results/is_battery_trades.csv
(written by run_is_battery.py):

  2. IS net expectancy > 0 (fade arm, pooled across pairs) AND > coin arm's net expectancy.
  3. Walk-forward: 3 IS thirds (equal duration), fade arm net-positive in >=2 of 3.
  4. Breadth: >=6/12 pairs gross-positive IS (fade arm, per-pair gross mean).

Also written (informative, NOT a locked gate — supplementary rigor per the task brief's
"report whatever falls out"): day-block bootstrap 95% CI on the fade arm's pooled net_pips,
and the pre-declared month-end (last trading day) vs rest split (fade arm gross & net) —
reported, never searched, never gated.

Outputs: results/is_per_pair_summary.csv, results/is_portfolio_summary.csv,
results/gate_table.csv, results/gate_extra.json, results/month_end_split.csv.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from data_loader import IS_END, PAIRS

N_THIRDS = 3
N_BOOT = 2000
BOOT_SEED = 20260708


def per_group_stats(df):
    n = len(df)
    if n == 0:
        return dict(n=0, wr=float("nan"), gross=float("nan"), net=float("nan"))
    wr = float((df["net_pips"] > 0).mean())
    return dict(n=n, wr=wr, gross=float(df["gross_pips"].mean()), net=float(df["net_pips"].mean()))


def build_per_pair_and_portfolio(trades_df):
    rows = []
    for pair in PAIRS:
        for arm in ("fade", "coin", "continuation"):
            sub = trades_df[(trades_df["pair"] == pair) & (trades_df["arm"] == arm)]
            rows.append({"pair": pair, "arm": arm, **per_group_stats(sub)})
    per_pair = pd.DataFrame(rows)

    port_rows = []
    for arm in ("fade", "coin", "continuation"):
        sub = trades_df[trades_df["arm"] == arm]
        port_rows.append({"pair": "PORTFOLIO", "arm": arm, **per_group_stats(sub)})
    portfolio = pd.DataFrame(port_rows)
    return per_pair, portfolio


def walk_forward_thirds(fade_df):
    """Split the IS window [min entry_ts, IS_END) into N_THIRDS equal-DURATION chunks
    (time-based, not count-based) and compute pooled fade-arm net_pips mean in each."""
    entry_ts = pd.to_datetime(fade_df["entry_ts"])
    is_start = entry_ts.min()
    is_end = IS_END
    edges = pd.date_range(is_start, is_end, periods=N_THIRDS + 1)
    thirds = []
    for i in range(N_THIRDS):
        lo, hi = edges[i], edges[i + 1]
        mask = (entry_ts >= lo) & (entry_ts < hi) if i < N_THIRDS - 1 else (entry_ts >= lo) & (entry_ts <= hi)
        sub = fade_df[mask]
        n = len(sub)
        mean_net = float(sub["net_pips"].mean()) if n else float("nan")
        thirds.append({"third": i + 1, "start": str(lo), "end": str(hi), "n": n, "mean_net": mean_net})
    return thirds


def day_block_bootstrap(fade_df, n_boot=N_BOOT, seed=BOOT_SEED):
    """Block bootstrap by UTC calendar day of entry_ts: resample DAYS with replacement, pool
    trades on the resampled days, compute pooled mean net_pips per resample."""
    df = fade_df.copy()
    df["entry_day"] = pd.to_datetime(df["entry_ts"]).dt.date
    days = df["entry_day"].unique()
    day_to_vals = {d: df.loc[df["entry_day"] == d, "net_pips"].values for d in days}
    n_days = len(days)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample_days = rng.choice(days, size=n_days, replace=True)
        vals = np.concatenate([day_to_vals[d] for d in sample_days])
        boot_means[b] = vals.mean() if len(vals) else np.nan
    p_le_zero = float(np.mean(boot_means <= 0))
    return p_le_zero, boot_means


def month_end_split(fade_df):
    rows = []
    for label, mask in (("last_trading_day", fade_df["month_end"] == True),   # noqa: E712
                         ("rest", fade_df["month_end"] == False)):            # noqa: E712
        sub = fade_df[mask]
        rows.append({"group": label, **per_group_stats(sub)})
    return pd.DataFrame(rows)


def compute_all_gates(trades_df):
    per_pair, portfolio = build_per_pair_and_portfolio(trades_df)

    port_fade = trades_df[trades_df["arm"] == "fade"]
    port_coin = trades_df[trades_df["arm"] == "coin"]

    # Gate 2
    net_fade = float(port_fade["net_pips"].mean())
    net_coin = float(port_coin["net_pips"].mean())
    gate2_pass = bool((net_fade > 0) and (net_fade > net_coin))

    # Gate 3
    thirds = walk_forward_thirds(port_fade)
    n_pos = sum(1 for t in thirds if not np.isnan(t["mean_net"]) and t["mean_net"] > 0)
    gate3_pass = bool(n_pos >= 2)

    # Gate 4
    per_pair_fade = per_pair[per_pair["arm"] == "fade"]
    n_gross_pos = int((per_pair_fade["gross"] > 0).sum())
    gate4_pass = bool(n_gross_pos >= 6)

    # Supplementary (not a locked gate)
    p_le_zero, boot_means = day_block_bootstrap(port_fade)
    me_split = month_end_split(port_fade)

    gate_table = pd.DataFrame([
        {"gate": 2, "name": "IS net>0 AND > coin", "pass": gate2_pass,
         "detail": f"fade_net={net_fade:+.3f}p coin_net={net_coin:+.3f}p"},
        {"gate": 3, "name": "WF 3 thirds >=2/3 net-positive", "pass": gate3_pass,
         "detail": f"n_pos={n_pos}/3 thirds={[round(t['mean_net'], 3) for t in thirds]}"},
        {"gate": 4, "name": "breadth >=6/12 pairs gross-positive", "pass": gate4_pass,
         "detail": f"n_gross_pos={n_gross_pos}/12"},
    ])

    extra = {
        "portfolio_fade_net": net_fade,
        "portfolio_fade_gross": float(port_fade["gross_pips"].mean()),
        "portfolio_coin_net": net_coin,
        "portfolio_coin_gross": float(port_coin["gross_pips"].mean()),
        "portfolio_continuation_net": float(trades_df.loc[trades_df["arm"] == "continuation", "net_pips"].mean()),
        "portfolio_continuation_gross": float(trades_df.loc[trades_df["arm"] == "continuation", "gross_pips"].mean()),
        "wf_thirds": thirds,
        "mc_p_le_zero_supplementary": p_le_zero,
        "mc_boot_mean_supplementary": float(boot_means.mean()),
        "mc_boot_ci_2p5_supplementary": float(np.percentile(boot_means, 2.5)),
        "mc_boot_ci_97p5_supplementary": float(np.percentile(boot_means, 97.5)),
        "breadth_n_gross_pos": n_gross_pos,
        "worst_mae_pips_fade": float(port_fade["mae_pips"].max()) if len(port_fade) else float("nan"),
        "mean_mae_pips_fade": float(port_fade["mae_pips"].mean()) if len(port_fade) else float("nan"),
        "mean_mfe_pips_fade": float(port_fade["mfe_pips"].mean()) if len(port_fade) else float("nan"),
    }
    if len(port_fade):
        worst_row = port_fade.loc[port_fade["mae_pips"].idxmax()]
        extra["worst_mae_trade_pair"] = str(worst_row["pair"])
        extra["worst_mae_trade_date"] = str(worst_row["date"])
        extra["worst_mae_trade_D_pips"] = float(worst_row["D_pips"])
    return per_pair, portfolio, gate_table, extra, me_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-csv", default="results/is_battery_trades.csv")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    trades_df = pd.read_csv(args.trades_csv)
    for col in ("fix_close_utc", "entry_ts", "exit_ts"):
        trades_df[col] = pd.to_datetime(trades_df[col], utc=True)
    trades_df["month_end"] = trades_df["month_end"].astype(bool)

    per_pair, portfolio, gate_table, extra, me_split = compute_all_gates(trades_df)

    per_pair.to_csv(os.path.join(args.out_dir, "is_per_pair_summary.csv"), index=False)
    portfolio.to_csv(os.path.join(args.out_dir, "is_portfolio_summary.csv"), index=False)
    gate_table.to_csv(os.path.join(args.out_dir, "gate_table.csv"), index=False)
    me_split.to_csv(os.path.join(args.out_dir, "month_end_split.csv"), index=False)
    with open(os.path.join(args.out_dir, "gate_extra.json"), "w") as f:
        json.dump(extra, f, indent=2, default=str)

    print(gate_table.to_string(index=False))
    print(me_split.to_string(index=False))
    print(json.dumps(extra, indent=2, default=str))


if __name__ == "__main__":
    main()
