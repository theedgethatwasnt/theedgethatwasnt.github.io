#!/usr/bin/env python3
"""
compute_gates.py — fx_factors Gates 1-4 (PREREGISTRATION.md "Gates before OOS"). Gates 1 and
2 are PROVEN by test_rebalance_engine.py (pytest, run on the box before this script); this
script re-derives their headline numbers from the already-computed battery outputs for the
results table (a numeric confirmation on the real IS run, not a substitute for the pytest
proof) and computes gates 3 and 4 in full from the battery outputs.

Gate 1 — harness self-test: R10 null (200 random-weight portfolios, run on real IS data by
  run_is_battery.py) mean net is negative and small in magnitude (pure cost drag, no phantom
  edge) — the rigorous version (checked against an independently-estimated average round-trip
  cost) is test_rebalance_engine.py::test_self_test_random_weights_approx_negative_costs.
Gate 2 — carry-accrual parity vs carry_model (+-5%): test_rebalance_engine.py::
  test_carry_accrual_parity (independent day-loop re-derivation vs the engine's own bulk
  carry_pips() call).
Gate 3 — gated carry IS: net > 0 AND > null's 95th percentile.
Gate 4 — WF: IS halves (chronological) both net-positive.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

NULL_SELFTEST_ABS_BOUND_PIPS = 50.0  # generous sanity bound for "small in magnitude"


def compute_gates(monthly_df, null_df):
    carry_gated = (monthly_df[monthly_df["variant"] == "carry_gated"]
                   .sort_values("signal_date").reset_index(drop=True))
    is_net = float(carry_gated["net_pips"].mean())
    null_mean = float(null_df["mean_net_pips"].mean())
    null_p95 = float(np.percentile(null_df["mean_net_pips"], 95))

    gate1_pass = bool(null_mean < 0 and abs(null_mean) < NULL_SELFTEST_ABS_BOUND_PIPS)
    gate3_pass = bool(is_net > 0 and is_net > null_p95)

    n = len(carry_gated)
    half = n // 2
    first_half, second_half = carry_gated.iloc[:half], carry_gated.iloc[half:]
    fh_mean = float(first_half["net_pips"].mean()) if len(first_half) else float("nan")
    sh_mean = float(second_half["net_pips"].mean()) if len(second_half) else float("nan")
    gate4_pass = bool(not np.isnan(fh_mean) and not np.isnan(sh_mean) and fh_mean > 0 and sh_mean > 0)

    gate_table = pd.DataFrame([
        {"gate": 1, "name": "harness self-test: null mean approx -costs (<0, |.|<50p)", "pass": gate1_pass,
         "detail": f"null_mean={null_mean:+.4f}p/rebal (n_seeds={len(null_df)}); rigorous check in test_rebalance_engine.py"},
        {"gate": 2, "name": "carry-accrual parity vs carry_model (+-5%)", "pass": None,
         "detail": "proven in test_rebalance_engine.py::test_carry_accrual_parity (see pytest output, not re-derived here)"},
        {"gate": 3, "name": "gated carry IS net>0 AND > null p95", "pass": gate3_pass,
         "detail": f"carry_gated_net={is_net:+.4f}p null_p95={null_p95:+.4f}p null_mean={null_mean:+.4f}p"},
        {"gate": 4, "name": "WF halves both net-positive", "pass": gate4_pass,
         "detail": f"first_half(n={len(first_half)})={fh_mean:+.4f}p second_half(n={len(second_half)})={sh_mean:+.4f}p"},
    ])

    extra = {
        "carry_gated_is_net": is_net,
        "null_mean": null_mean,
        "null_p95": null_p95,
        "n_rebalances_carry_gated": n,
        "wf_first_half_mean": fh_mean,
        "wf_second_half_mean": sh_mean,
    }
    return gate_table, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    monthly_df = pd.read_csv(os.path.join(args.out_dir, "is_monthly.csv"))
    null_df = pd.read_csv(os.path.join(args.out_dir, "null_r10.csv"))
    gate_table, extra = compute_gates(monthly_df, null_df)
    gate_table.to_csv(os.path.join(args.out_dir, "gate_table.csv"), index=False)
    with open(os.path.join(args.out_dir, "gate_extra.json"), "w") as f:
        json.dump(extra, f, indent=2, default=str)
    print(gate_table.to_string(index=False))
    print(json.dumps(extra, indent=2, default=str))


if __name__ == "__main__":
    main()
