#!/usr/bin/env python3
"""run_is_battery.py — COT Contrarian Positioning: primary IS battery (IS ONLY — OOS never
touched, per PREREGISTRATION.md gate 5 / R8). Builds the joint COT x D1-price rebalance
schedule, restricts to IS (is_data.restrict_sched_to_is — hard guard), computes IS-only
spread medians, then the 3 arms' weekly returns (contrarian / momentum / 200-replicate
null), writes results/is_battery.csv + results/data_coverage.json.

Usage (on Hetzner):
  /root/venv/bin/python3 run_is_battery.py --out-dir results
"""
import argparse
import json
import os

import pandas as pd

import d1_data as d1
import is_data as isd
import portfolio as pf

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-parquet", default=os.path.join(HERE, "cot_weekly.parquet"))
    ap.add_argument("--data-dir", default=d1.DATA_DIR)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--n-null", type=int, default=pf.N_NULL)
    ap.add_argument("--spread-mult", type=float, default=1.0)
    ap.add_argument("--markup-mult", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/6] loading COT weekly panel...", flush=True)
    cot_df = pd.read_parquet(args.cot_parquet)
    print(f"      {len(cot_df)} rows, {cot_df['currency'].nunique()} currencies, "
          f"{cot_df['report_date'].min().date()} -> {cot_df['report_date'].max().date()}", flush=True)

    print("[2/6] building trading calendar + full rebalance schedule...", flush=True)
    calendar = d1.trading_calendar(pf.DIRECT_PAIRS, data_dir=args.data_dir)
    full_sched = pf.build_rebalance_schedule(cot_df, calendar)
    print(f"      full joint schedule: {len(full_sched)} weeks, "
          f"{full_sched['action_date'].min()} -> {full_sched['action_date'].max()}", flush=True)

    print("[3/6] restricting to IS (hard guard)...", flush=True)
    is_sched = isd.restrict_sched_to_is(full_sched)
    print(f"      IS: {len(is_sched)} weeks ({len(is_sched)/len(full_sched)*100:.1f}% of joint window), "
          f"{is_sched['action_date'].min()} -> {is_sched['action_date'].max()}", flush=True)

    print("[4/6] loading D1 price panel + computing IS-only spread medians...", flush=True)
    price_panel = pf.load_price_panel(pf.DIRECT_PAIRS, data_dir=args.data_dir)
    spread_medians = isd.is_only_spread_medians(price_panel)
    print(f"      spread medians (IS-only, pips): {spread_medians}", flush=True)

    print(f"[5/6] building weekly returns (3 arms + {args.n_null} null replicates)...", flush=True)
    is_out, n_dropped = pf.build_weekly_returns_from_schedule(
        is_sched, price_panel, spread_medians,
        spread_mult=args.spread_mult, markup_mult=args.markup_mult, n_null=args.n_null,
    )
    print(f"      resolved {len(is_out)}/{len(is_sched)} weeks ({n_dropped} dropped)", flush=True)

    out_path = os.path.join(args.out_dir, "is_battery.csv")
    is_out.to_csv(out_path, index=False)
    print(f"[6/6] wrote {out_path}", flush=True)

    # Belt-and-suspenders re-check on the assembled output.
    assert pd.to_datetime(is_out["action_date"]).max() < isd.IS_PRICE_CUTOFF, "OOS LEAK in assembled output"

    coverage = {
        "cot_rows": int(len(cot_df)),
        "cot_currencies": sorted(cot_df["currency"].unique().tolist()),
        "cot_span": [str(cot_df["report_date"].min().date()), str(cot_df["report_date"].max().date())],
        "price_pairs": pf.DIRECT_PAIRS,
        "price_ceiling": "2026-05-21",
        "full_joint_schedule_weeks": int(len(full_sched)),
        "full_joint_schedule_span": [str(full_sched["action_date"].min()), str(full_sched["action_date"].max())],
        "is_weeks": int(len(is_sched)),
        "is_span": [str(is_sched["action_date"].min()), str(is_sched["action_date"].max())],
        "is_resolved_weeks": int(len(is_out)),
        "is_dropped_weeks": int(n_dropped),
        "is_fraction_of_joint_window": len(is_sched) / len(full_sched),
        "spread_medians_is_only_pips": spread_medians,
        "n_null_replicates": args.n_null,
        "spread_mult": args.spread_mult,
        "markup_mult": args.markup_mult,
    }
    with open(os.path.join(args.out_dir, "data_coverage.json"), "w") as f:
        json.dump(coverage, f, indent=2, default=str)
    print(json.dumps(coverage, indent=2, default=str))


if __name__ == "__main__":
    main()
