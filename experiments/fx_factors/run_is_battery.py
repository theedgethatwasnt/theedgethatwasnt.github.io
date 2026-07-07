#!/usr/bin/env python3
"""
run_is_battery.py — fx_factors: orchestrates the IS-only battery. Loads 12-pair D1 IS data
once (via is_data.load_pair_is_d1 — hard OOS-sealed), builds the currency-index panels + SPX
gate, builds the monthly rebalance schedule, then runs:
  - carry_gated   (PRIMARY, confirmatory)
  - carry_ungated (secondary)
  - momentum      (secondary)
  - value         (secondary)
  - composite     (secondary)
  - R10 null (200 seeds)
and writes results/is_monthly.csv, results/is_legs.csv, results/null_r10.csv.

Usage (on Hetzner):
  /root/venv/bin/python3 run_is_battery.py \
      --data-dir /root/work/data/m5_ba --cross-asset-dir /root/work/data/cross_asset \
      --out-dir results
"""
import argparse
import gc
import os

import numpy as np
import pandas as pd

import currency_index
import factors
from is_data import IS_END, PAIRS, load_pair_is_d1, load_spx_is
from null_r10 import run_null
from rebalance_engine import build_master_calendar, build_rebalance_schedule, run_portfolio


def load_all_pairs(data_dir):
    pair_d1 = {}
    for pair in PAIRS:
        print(f"[{pair}] loading IS D1...", flush=True)
        df = load_pair_is_d1(pair, data_dir)
        pair_d1[pair] = df
        print(f"[{pair}] {len(df)} IS D1 bars, {df['timestamp'].min()} -> {df['timestamp'].max()}", flush=True)
    return pair_d1


def build_direction_fn(score_dict):
    def fn(sd):
        return factors.rank_select(score_dict[sd])
    return fn


def run_battery(data_dir, cross_asset_dir, out_dir, n_null_seeds=200):
    os.makedirs(out_dir, exist_ok=True)
    pair_d1 = load_all_pairs(data_dir)
    spx = load_spx_is(cross_asset_dir)
    gc.collect()

    usd_per_x, xusd = currency_index.build_panels(pair_d1)
    calendar = build_master_calendar(pair_d1)
    schedule = build_rebalance_schedule(calendar)
    signal_dates = [sd for sd, _ in schedule]
    print(f"{len(schedule)} monthly rebalances scheduled, {signal_dates[0]} -> {signal_dates[-1]}", flush=True)

    gate_series = currency_index.spx_gate_signal(spx, signal_dates)

    def gate_fn(d):
        return bool(gate_series.loc[d])

    carry_scores, mom_scores, val_scores, comp_scores = {}, {}, {}, {}
    for sd in signal_dates:
        carry_scores[sd] = factors.carry_score(sd)
        mom_scores[sd] = factors.momentum_score(usd_per_x, sd)
        val_scores[sd] = factors.value_score(xusd, sd)
        comp_scores[sd] = factors.composite_score(carry_scores[sd], mom_scores[sd], val_scores[sd])

    variants = {
        "carry_gated": (build_direction_fn(carry_scores), gate_fn),
        "carry_ungated": (build_direction_fn(carry_scores), None),
        "momentum": (build_direction_fn(mom_scores), None),
        "value": (build_direction_fn(val_scores), None),
        "composite": (build_direction_fn(comp_scores), None),
    }

    all_monthly, all_legs = [], []
    for name, (dfn, gfn) in variants.items():
        monthly_df, legs_df = run_portfolio(pair_d1, schedule, dfn, gate_fn=gfn)
        monthly_df["variant"] = name
        legs_df["variant"] = name
        all_monthly.append(monthly_df)
        all_legs.append(legs_df)
        mean_net = monthly_df["net_pips"].mean() if len(monthly_df) else float("nan")
        print(f"[{name}] n_rebal={len(monthly_df)} mean_net={mean_net:+.4f} p/rebalance", flush=True)

    monthly_all = pd.concat(all_monthly, ignore_index=True)
    legs_all = pd.concat(all_legs, ignore_index=True)
    monthly_all.to_csv(os.path.join(out_dir, "is_monthly.csv"), index=False)
    legs_all.to_csv(os.path.join(out_dir, "is_legs.csv"), index=False)
    print(f"wrote is_monthly.csv ({len(monthly_all)} rows), is_legs.csv ({len(legs_all)} rows)", flush=True)

    print(f"running R10 null ({n_null_seeds} seeds)...", flush=True)
    null_df = run_null(pair_d1, schedule, n_seeds=n_null_seeds)
    null_df.to_csv(os.path.join(out_dir, "null_r10.csv"), index=False)
    print(f"null mean={null_df['mean_net_pips'].mean():+.4f} "
          f"p95={np.percentile(null_df['mean_net_pips'], 95):+.4f}", flush=True)

    # Belt-and-suspenders OOS guard: every date used descends from load_pair_is_d1()'s
    # IS-filtered arrays, so this should be trivially true by construction.
    assert pd.Timestamp(monthly_all["execution_date"].max()) < IS_END, (
        "OOS LEAK in execution_date"
    )
    assert pd.Timestamp(monthly_all["next_execution_date"].max()) < IS_END, (
        "OOS LEAK in next_execution_date (holding-period exit)"
    )

    del pair_d1
    gc.collect()
    return monthly_all, legs_all, null_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cross-asset-dir", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-null-seeds", type=int, default=200)
    args = ap.parse_args()
    run_battery(args.data_dir, args.cross_asset_dir, args.out_dir, n_null_seeds=args.n_null_seeds)
