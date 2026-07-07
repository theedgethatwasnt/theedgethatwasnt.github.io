#!/usr/bin/env python3
"""
run_stage1_battery.py — momentum_confirm Stage 1: runs the FROZEN momentum rule (imported
verbatim from research/experiments/fx_factors — factors.py / currency_index.py /
rebalance_engine.py / null_r10.py, not reimplemented) on the deep 2005->2020-10 D1 segment
(stage1_data.py, hard-sealed).

No parameter differs from the fx_factors sweep's momentum variant: same rank_select (top-3/
bottom-3 of 7 non-USD currencies), same 12-1 momentum score, same 63-day inverse-vol weighting,
same run_portfolio() cost engine. The ONLY things that differ are (1) the input data (deep D1
mid + synthetic constant-median-spread bid/ask, vs the M5-BA-aggregated 2020-11+ data) and (2)
there is no SPX risk-off gate applied (that gate was pre-registered for the CARRY factor only,
never for momentum — fx_factors/run_is_battery.py itself runs "momentum" with gate_fn=None).

Runs at spread_mult=1.0 (primary, gated) and spread_mult=1.5 (sensitivity, reported only).
Also derives a carry-free gross column per rebalance from the legs table (gross_pips -
spread_rt_pips, i.e. price return net of spread but with the carry term stripped out — PREREGISTRATION.md
"Costs": "mitigated by reporting carry-free gross alongside" — carry's pre-2020 splice has
"direction of bias unknown").

Usage (on Hetzner, code_mom/momentum_confirm/):
  /root/venv/bin/python3 run_stage1_battery.py \
      --data-dir /root/work/code_mom/data/d1_deep --out-dir results
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FX_FACTORS_DIR = HERE.parent / "fx_factors"
sys.path.insert(0, str(FX_FACTORS_DIR))

# Import fx_factors's OWN is_data FIRST (while FX_FACTORS_DIR is still sys.path[0]) so it lands
# in sys.modules["is_data"] before currency_index.py's internal sys.path.insert(0,
# multiday_contrarian) shadows the name with multiday_contrarian's *different* is_data.py (same
# filename, unrelated module, no CURRENCIES) — exactly the import order fx_factors/
# run_is_battery.py itself relies on (its own top-level `from is_data import ...`).
import is_data as _fx_factors_is_data  # noqa: E402,F401

import currency_index  # noqa: E402
import factors  # noqa: E402
from null_r10 import run_null  # noqa: E402
from rebalance_engine import build_master_calendar, build_rebalance_schedule, run_portfolio  # noqa: E402

from stage1_data import STAGE1_END, load_all_pairs_stage1  # noqa: E402

N_NULL_SEEDS = 200


def build_direction_fn(score_dict):
    def fn(sd):
        return factors.rank_select(score_dict[sd])
    return fn


def carry_free_gross_by_rebalance(legs_df):
    """Portfolio carry-free gross P&L per rebalance: sum_i w_i * (gross_pips_i -
    spread_rt_pips_i) — price return net of spread, WITHOUT the carry term."""
    g = legs_df.copy()
    g["contrib"] = g["weight"] * (g["gross_pips"] - g["spread_rt_pips"])
    out = g.groupby("signal_date")["contrib"].sum()
    out.name = "carry_free_gross_pips"
    return out


def run_battery(data_dir, out_dir, n_null_seeds=N_NULL_SEEDS):
    os.makedirs(out_dir, exist_ok=True)
    pair_d1 = load_all_pairs_stage1(data_dir, currency_index.REQUIRED_PAIRS_FOR_INDEX)

    usd_per_x, xusd = currency_index.build_panels(pair_d1)
    calendar = build_master_calendar(pair_d1)
    schedule = build_rebalance_schedule(calendar)
    signal_dates = [sd for sd, _ in schedule]
    print(f"{len(schedule)} monthly rebalances scheduled, {signal_dates[0]} -> {signal_dates[-1]}", flush=True)

    mom_scores = {sd: factors.momentum_score(usd_per_x, sd) for sd in signal_dates}
    direction_fn = build_direction_fn(mom_scores)

    all_monthly, all_legs = [], []
    for spread_mult, tag in [(1.0, "momentum"), (1.5, "momentum_spread1.5x")]:
        monthly_df, legs_df = run_portfolio(pair_d1, schedule, direction_fn, spread_mult=spread_mult)
        cfg = carry_free_gross_by_rebalance(legs_df)
        monthly_df = monthly_df.merge(cfg, left_on="signal_date", right_index=True, how="left")
        monthly_df["variant"] = tag
        legs_df["variant"] = tag
        all_monthly.append(monthly_df)
        all_legs.append(legs_df)
        mean_net = monthly_df["net_pips"].mean() if len(monthly_df) else float("nan")
        mean_gross = monthly_df["carry_free_gross_pips"].mean() if len(monthly_df) else float("nan")
        print(f"[{tag}] n_rebal={len(monthly_df)} mean_net={mean_net:+.4f} "
              f"mean_carry_free_gross={mean_gross:+.4f} p/rebalance", flush=True)

    monthly_all = pd.concat(all_monthly, ignore_index=True)
    legs_all = pd.concat(all_legs, ignore_index=True)
    monthly_all.to_csv(os.path.join(out_dir, "stage1_monthly.csv"), index=False)
    legs_all.to_csv(os.path.join(out_dir, "stage1_legs.csv"), index=False)
    print(f"wrote stage1_monthly.csv ({len(monthly_all)} rows), stage1_legs.csv ({len(legs_all)} rows)", flush=True)

    print(f"running R10 null ({n_null_seeds} seeds, spread_mult=1.0)...", flush=True)
    null_df = run_null(pair_d1, schedule, n_seeds=n_null_seeds, spread_mult=1.0)
    null_df.to_csv(os.path.join(out_dir, "stage1_null_r10.csv"), index=False)
    print(f"null mean={null_df['mean_net_pips'].mean():+.4f} "
          f"p95={np.percentile(null_df['mean_net_pips'], 95):+.4f}", flush=True)

    # Belt-and-suspenders segment guard.
    assert pd.Timestamp(monthly_all["execution_date"].max()) < STAGE1_END, "SEGMENT LEAK in execution_date"
    assert pd.Timestamp(monthly_all["next_execution_date"].max()) < STAGE1_END, (
        "SEGMENT LEAK in next_execution_date (holding-period exit)"
    )

    return monthly_all, legs_all, null_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-null-seeds", type=int, default=N_NULL_SEEDS)
    args = ap.parse_args()
    run_battery(args.data_dir, args.out_dir, n_null_seeds=args.n_null_seeds)
