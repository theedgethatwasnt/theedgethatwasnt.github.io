#!/usr/bin/env python3
"""
compute_stage1_gates.py — momentum_confirm Stage-1 gates 1-4 (PREREGISTRATION.md
"Stage-1 gates"), computed on the primary spread_mult=1.0 momentum run.

Gate 1 — apparatus self-test: R10 null (200 random-weight portfolios, same schedule/costs)
  mean net approx -costs (negative, small in magnitude — pure cost drag, no phantom edge).
Gate 2 — deep-segment net > 0 AND > null 95th percentile (~180 rebalances 2005-2020).
Gate 3 — WF: deep segment THIRDS (chronological), net-positive >= 2/3, none < -40 p/rebalance.
Gate 4 — regime robustness: pre-declared split at 2013 (calendar year of signal_date), positive
  in at least one of the two halves AND the other half not catastrophic (< -40 p/rebalance).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

NULL_SELFTEST_ABS_BOUND_PIPS = 50.0
CATASTROPHIC_PIPS = -40.0
SPLIT_YEAR = 2013


def compute_gates(monthly_df, null_df):
    mom = (monthly_df[monthly_df["variant"] == "momentum"]
           .sort_values("signal_date").reset_index(drop=True))
    mom["signal_date"] = pd.to_datetime(mom["signal_date"], utc=True)
    net = mom["net_pips"]

    deep_net = float(net.mean())
    null_mean = float(null_df["mean_net_pips"].mean())
    null_p95 = float(np.percentile(null_df["mean_net_pips"], 95))

    # Gate 1
    gate1_pass = bool(null_mean < 0 and abs(null_mean) < NULL_SELFTEST_ABS_BOUND_PIPS)

    # Gate 2
    gate2_pass = bool(deep_net > 0 and deep_net > null_p95)

    # Gate 3 — chronological thirds
    n = len(mom)
    edges = [0, n // 3, 2 * n // 3, n]
    thirds = [mom.iloc[edges[i]:edges[i + 1]] for i in range(3)]
    third_means = [float(t["net_pips"].mean()) if len(t) else float("nan") for t in thirds]
    n_positive_thirds = sum(1 for m in third_means if not np.isnan(m) and m > 0)
    none_catastrophic_thirds = all((np.isnan(m) or m >= CATASTROPHIC_PIPS) for m in third_means)
    gate3_pass = bool(n_positive_thirds >= 2 and none_catastrophic_thirds)

    # Gate 4 — pre-declared 2013 split (calendar year of signal_date)
    is_second_half = mom["signal_date"].dt.year >= SPLIT_YEAR
    first_half = mom[~is_second_half]
    second_half = mom[is_second_half]
    fh_mean = float(first_half["net_pips"].mean()) if len(first_half) else float("nan")
    sh_mean = float(second_half["net_pips"].mean()) if len(second_half) else float("nan")
    at_least_one_positive = bool((not np.isnan(fh_mean) and fh_mean > 0) or
                                  (not np.isnan(sh_mean) and sh_mean > 0))
    other_not_catastrophic = True
    if not np.isnan(fh_mean) and not np.isnan(sh_mean):
        if fh_mean <= 0:
            other_not_catastrophic = fh_mean >= CATASTROPHIC_PIPS
        if sh_mean <= 0:
            other_not_catastrophic = other_not_catastrophic and (sh_mean >= CATASTROPHIC_PIPS)
    gate4_pass = bool(at_least_one_positive and other_not_catastrophic)

    gate_table = pd.DataFrame([
        {"gate": 1, "name": "apparatus self-test: null mean approx -costs (<0, |.|<50p)",
         "pass": gate1_pass,
         "detail": f"null_mean={null_mean:+.4f}p/rebal (n_seeds={len(null_df)})"},
        {"gate": 2, "name": "deep-segment momentum net>0 AND > null p95", "pass": gate2_pass,
         "detail": f"deep_net={deep_net:+.4f}p null_p95={null_p95:+.4f}p null_mean={null_mean:+.4f}p n={n}"},
        {"gate": 3, "name": "WF thirds: >=2/3 positive, none < -40p/rebal", "pass": gate3_pass,
         "detail": f"thirds_mean=[{third_means[0]:+.3f}, {third_means[1]:+.3f}, {third_means[2]:+.3f}] "
                    f"n_positive={n_positive_thirds}/3"},
        {"gate": 4, "name": f"regime split @ {SPLIT_YEAR}: >=1 half positive, other not < -40p",
         "pass": gate4_pass,
         "detail": f"pre_{SPLIT_YEAR}(n={len(first_half)})={fh_mean:+.4f}p "
                    f"post_{SPLIT_YEAR}(n={len(second_half)})={sh_mean:+.4f}p"},
    ])

    extra = {
        "deep_net": deep_net,
        "null_mean": null_mean,
        "null_p95": null_p95,
        "null_p5": float(np.percentile(null_df["mean_net_pips"], 5)),
        "null_std": float(null_df["mean_net_pips"].std(ddof=1)),
        "n_rebalances": n,
        "third_means": third_means,
        "n_positive_thirds": n_positive_thirds,
        "split_year": SPLIT_YEAR,
        "pre_split_mean": fh_mean,
        "pre_split_n": int(len(first_half)),
        "post_split_mean": sh_mean,
        "post_split_n": int(len(second_half)),
        "cum_net_pips": float(net.sum()),
        "max_dd_pips": float((net.cumsum() - net.cumsum().cummax()).min()) if n else float("nan"),
    }
    return gate_table, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    monthly_df = pd.read_csv(os.path.join(args.out_dir, "stage1_monthly.csv"))
    null_df = pd.read_csv(os.path.join(args.out_dir, "stage1_null_r10.csv"))
    gate_table, extra = compute_gates(monthly_df, null_df)
    gate_table.to_csv(os.path.join(args.out_dir, "stage1_gate_table.csv"), index=False)
    with open(os.path.join(args.out_dir, "stage1_gate_extra.json"), "w") as f:
        json.dump(extra, f, indent=2, default=str)
    print(gate_table.to_string(index=False))
    print(json.dumps(extra, indent=2, default=str))


if __name__ == "__main__":
    main()
