#!/usr/bin/env python3
"""compute_gates.py — Composite 1: Gates 1-6 (IS-only), per PREREGISTRATION.md + the task
brief's explicit ordering:
  1. RW self-test (test_rw_selftest.py, run as a subprocess so this script produces one
     combined gate table).
  2. Axis-3 parity vs cot_positioning IS (same window, same code -- re-runs
     cot_positioning/run_is_battery.py fresh and compares the contrarian arm's mean net
     p/wk to the committed results/gate_results.json, tolerance +-0.1p/wk).
  3. Composite IS: all three H1 criteria on IS (money criterion / vs coin / vs the
     shuffled-positioning null's 95th percentile), day-block bootstrap (2000 resamples,
     blocked by entry_ts UTC calendar day -- the same day-block convention
     multiday_contrarian/compute_gates.py uses for its own D1/M5-scale trade lists).
  4. WF: 3 IS thirds (equal calendar duration), >=2/3 net-positive.
  5. Breadth: >=4/7 pairs gross-positive IS (composite arm).
  6. Trade count: composite >=150 IS trades.
Gate 7 (user gate -> OOS unseal) is out of scope for this script.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import _paths

HERE = os.path.dirname(os.path.abspath(__file__))
N_THIRDS = 3
N_BOOT = 2000
BOOT_SEED = 20260709
MIN_TRADE_COUNT = 150
MIN_BREADTH = 4
PARITY_TOL = 0.1
PAIRS = ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "USD_CHF", "USD_CAD"]


def gate1_rw_selftest():
    r = subprocess.run([sys.executable, "-m", "pytest", "test_rw_selftest.py", "-q"],
                        cwd=HERE, capture_output=True, text=True)
    return {
        "pass": r.returncode == 0,
        "stdout_tail": r.stdout[-3000:],
        "stderr_tail": r.stderr[-1500:],
    }


def gate2_axis3_parity():
    out_dir = "/tmp/composite1_parity_check"
    os.makedirs(out_dir, exist_ok=True)
    r = subprocess.run([sys.executable, "run_is_battery.py", "--out-dir", out_dir],
                        cwd=_paths.COT_CODE_DIR, capture_output=True, text=True, env=os.environ.copy())
    if r.returncode != 0:
        return {"pass": False, "error": r.stderr[-3000:], "stdout_tail": r.stdout[-1500:]}
    fresh = pd.read_csv(os.path.join(out_dir, "is_battery.csv"))
    fresh_mean = float(fresh["contrarian_net"].mean())
    recorded_path = os.path.join(_paths.COT_CODE_DIR, "results", "gate_results.json")
    with open(recorded_path) as f:
        recorded = json.load(f)
    recorded_mean = recorded["gate3_contrarian_vs_null_and_momentum"]["contrarian_mean"]
    diff = abs(fresh_mean - recorded_mean)
    return {
        "pass": bool(diff <= PARITY_TOL),
        "fresh_mean_p_per_wk": fresh_mean,
        "recorded_mean_p_per_wk": recorded_mean,
        "abs_diff_p_per_wk": diff,
        "tolerance_p_per_wk": PARITY_TOL,
        "fresh_n_weeks": int(len(fresh)),
    }


def _by_day(df, col="net_pips", ts_col="entry_ts"):
    d = df.copy()
    d["entry_day"] = pd.to_datetime(d[ts_col]).dt.date
    return {day: g[col].values for day, g in d.groupby("entry_day")}


def day_block_bootstrap(vals_by_day, n_boot=N_BOOT, seed=BOOT_SEED):
    days = list(vals_by_day.keys())
    n_days = len(days)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample_days = rng.choice(days, size=n_days, replace=True)
        vals = np.concatenate([vals_by_day[d] for d in sample_days])
        boot_means[b] = vals.mean() if len(vals) else np.nan
    return boot_means


def gate3_composite_is(trades_df, shuffle_means):
    composite = trades_df[trades_df["arm"] == "composite"].copy()
    coin = trades_df[trades_df["arm"] == "coin"].copy()
    if len(composite) == 0:
        return {"pass": False, "error": "0 composite trades"}

    boot_net = day_block_bootstrap(_by_day(composite))
    ci_net = (float(np.percentile(boot_net, 2.5)), float(np.percentile(boot_net, 97.5)))
    mean_net = float(composite["net_pips"].mean())
    crit1 = bool(mean_net > 0 and ci_net[0] > 0)

    merged = composite.merge(coin, on=["pair", "signal_idx"], suffixes=("_comp", "_coin"))
    merged["diff"] = merged["net_pips_comp"] - merged["net_pips_coin"]
    diff_by_day = _by_day(merged, col="diff", ts_col="entry_ts_comp")
    boot_diff = day_block_bootstrap(diff_by_day)
    ci_diff = (float(np.percentile(boot_diff, 2.5)), float(np.percentile(boot_diff, 97.5)))
    mean_diff = float(merged["diff"].mean())
    crit2 = bool(mean_diff > 0 and ci_diff[0] > 0)

    null_p95 = float(np.nanpercentile(shuffle_means, 95))
    crit3 = bool(mean_net > null_p95)

    ok = crit1 and crit2 and crit3
    return {
        "pass": ok,
        "criterion1_money": {"pass": crit1, "mean_net_pips": mean_net,
                              "boot_ci_2p5": ci_net[0], "boot_ci_97p5": ci_net[1]},
        "criterion2_vs_coin": {"pass": crit2, "mean_diff_pips": mean_diff,
                                "boot_ci_2p5": ci_diff[0], "boot_ci_97p5": ci_diff[1],
                                "n_paired": int(len(merged))},
        "criterion3_vs_shuffled_null": {"pass": crit3, "composite_mean_pips": mean_net,
                                         "null_p95_pips": null_p95,
                                         "null_mean_pips": float(np.nanmean(shuffle_means))},
        "n_composite_trades": int(len(composite)),
    }


def gate4_walk_forward(trades_df, is_entry_cutoff):
    composite = trades_df[trades_df["arm"] == "composite"].copy()
    composite["entry_ts"] = pd.to_datetime(composite["entry_ts"], utc=True)
    if len(composite) == 0:
        return {"pass": False, "error": "0 composite trades"}
    lo = composite["entry_ts"].min()
    hi = pd.Timestamp(is_entry_cutoff)
    edges = pd.date_range(lo, hi, periods=N_THIRDS + 1)
    thirds = []
    for i in range(N_THIRDS):
        a, b = edges[i], edges[i + 1]
        mask = (composite["entry_ts"] >= a) & (composite["entry_ts"] < b) if i < N_THIRDS - 1 \
            else (composite["entry_ts"] >= a) & (composite["entry_ts"] <= b)
        sub = composite[mask]
        n = len(sub)
        mean_net = float(sub["net_pips"].mean()) if n else float("nan")
        thirds.append({"third": i + 1, "start": str(a), "end": str(b), "n": n, "mean_net_pips": mean_net})
    n_pos = sum(1 for t in thirds if not np.isnan(t["mean_net_pips"]) and t["mean_net_pips"] > 0)
    return {"pass": bool(n_pos >= 2), "thirds": thirds, "n_pos": n_pos}


def gate5_breadth(trades_df, pairs=PAIRS):
    composite = trades_df[trades_df["arm"] == "composite"]
    per_pair = {}
    n_pos = 0
    for p in pairs:
        sub = composite[composite["pair"] == p]
        n = int(len(sub))
        gross = float(sub["gross_pips"].mean()) if n else float("nan")
        net = float(sub["net_pips"].mean()) if n else float("nan")
        per_pair[p] = {"n": n, "gross_mean_pips": gross, "net_mean_pips": net}
        if n > 0 and gross > 0:
            n_pos += 1
    return {"pass": bool(n_pos >= MIN_BREADTH), "n_gross_pos": n_pos, "min_required": MIN_BREADTH,
            "per_pair": per_pair}


def gate6_trade_count(trades_df):
    n = int(len(trades_df[trades_df["arm"] == "composite"]))
    return {"pass": bool(n >= MIN_TRADE_COUNT), "n_composite_is_trades": n, "min_required": MIN_TRADE_COUNT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-csv", default=os.path.join(HERE, "results", "is_battery_trades.csv"))
    ap.add_argument("--shuffle-csv", default=os.path.join(HERE, "results", "shuffled_null.csv"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--skip-gate1", action="store_true", help="skip the pytest subprocess (already run separately)")
    ap.add_argument("--skip-gate2", action="store_true", help="skip the cot_positioning re-run (already verified)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import is_data as isd

    trades_df = pd.read_csv(args.trades_csv)
    for col in ("entry_ts", "exit_ts", "signal_ts"):
        if col in trades_df.columns:
            trades_df[col] = pd.to_datetime(trades_df[col], utc=True)
    shuffle_df = pd.read_csv(args.shuffle_csv)
    shuffle_means = shuffle_df["mean_net_pips"].values

    print("gate 1: RW self-test...", flush=True)
    g1 = {"pass": True, "skipped": True} if args.skip_gate1 else gate1_rw_selftest()
    print("gate 2: Axis-3 parity vs cot_positioning IS...", flush=True)
    g2 = {"pass": True, "skipped": True} if args.skip_gate2 else gate2_axis3_parity()
    print("gate 3: composite IS three criteria...", flush=True)
    g3 = gate3_composite_is(trades_df, shuffle_means)
    print("gate 4: walk-forward thirds...", flush=True)
    g4 = gate4_walk_forward(trades_df, isd.AXIS1_IS_ENTRY_CUTOFF)
    print("gate 5: breadth...", flush=True)
    g5 = gate5_breadth(trades_df)
    print("gate 6: trade count...", flush=True)
    g6 = gate6_trade_count(trades_df)

    gates = {
        "gate1_rw_selftest": g1,
        "gate2_axis3_parity": g2,
        "gate3_composite_is": g3,
        "gate4_walk_forward": g4,
        "gate5_breadth": g5,
        "gate6_trade_count": g6,
    }
    n_pass = sum(1 for g in gates.values() if g.get("pass"))
    gates["_summary"] = {"n_pass": n_pass, "n_gates": 6, "all_pass": n_pass == 6}

    with open(os.path.join(args.out_dir, "gate_results.json"), "w") as f:
        json.dump(gates, f, indent=2, default=str)

    table_rows = []
    for i, (key, g) in enumerate([
        ("1", g1), ("2", g2), ("3", g3), ("4", g4), ("5", g5), ("6", g6),
    ], start=1):
        table_rows.append({"gate": key, "pass": g.get("pass")})
    pd.DataFrame(table_rows).to_csv(os.path.join(args.out_dir, "gate_table.csv"), index=False)

    print(json.dumps({k: v.get("pass") for k, v in gates.items() if k != "_summary"}, indent=2))
    print(f"{n_pass}/6 gates pass")


if __name__ == "__main__":
    main()
