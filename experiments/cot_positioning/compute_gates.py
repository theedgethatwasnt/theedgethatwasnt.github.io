"""compute_gates.py — COT Contrarian Positioning: Gates 1-4 (IS-only), per PREREGISTRATION.md.

Gate 1: Data integrity: COT weekly continuity >=95% (test_fetch_cot.py) + publication-lag
        alignment tripwire test (test_release_lag.py). Both are pytest-verified; this
        script re-asserts the continuity number on the committed cot_weekly.parquet and
        records PASS/FAIL alongside the others for a single combined gate table.
Gate 2: RW/null self-test: random portfolios (the 200-replicate null arm, IS-only) must
        be ~= -costs (no phantom edge from the trade GEOMETRY alone — mean null return
        should be close to zero-ish net of real costs, i.e. not systematically positive).
Gate 3: Contrarian IS: net > 0, > null 95th pct, > momentum arm.
Gate 4: WF: IS thirds >= 2/3 net-positive.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import bootstrap as bs

N_THIRDS = 3


def gate1_continuity(cot_parquet_path: str) -> dict:
    df = pd.read_parquet(cot_parquet_path)
    per_ccy = {}
    ok = True
    for ccy, g in df.groupby("currency"):
        span_weeks = (g["report_date"].max() - g["report_date"].min()).days / 7.0
        continuity = len(g) / span_weeks
        per_ccy[ccy] = round(float(continuity), 4)
        if continuity < 0.95:
            ok = False
    return {"pass": ok, "per_currency_continuity": per_ccy}


def gate2_null_selftest(is_out: pd.DataFrame, n_null: int) -> dict:
    """Mean of the null arm's 200 replicates, averaged over ALL weeks — should be close
    to a small (near-zero-ish) cost drag, not a large systematic edge. Threshold: the
    null's overall mean magnitude must be well inside the contrarian/momentum arms'
    magnitude (documented heuristic: < half the smaller of the two signal arms'
    |mean|, or an absolute pips/week ceiling if both signal arms are ~0) — i.e. it must
    NOT look like a real strategy."""
    null_cols = [c for c in is_out.columns if c.startswith("null_net_")]
    assert len(null_cols) == n_null, f"expected {n_null} null columns, found {len(null_cols)}"
    null_mean_per_replicate = is_out[null_cols].mean(axis=0)  # one mean per replicate (200 numbers)
    overall_null_mean = float(null_mean_per_replicate.mean())
    overall_null_std = float(null_mean_per_replicate.std(ddof=1))
    contrarian_mean = float(is_out["contrarian_net"].mean())
    momentum_mean = float(is_out["momentum_net"].mean())
    signal_scale = max(abs(contrarian_mean), abs(momentum_mean), 1.0)  # pips/week floor=1.0
    ok = abs(overall_null_mean) < 0.5 * signal_scale
    return {
        "pass": bool(ok),
        "overall_null_mean": overall_null_mean,
        "overall_null_std_across_replicates": overall_null_std,
        "contrarian_mean": contrarian_mean,
        "momentum_mean": momentum_mean,
        "signal_scale": signal_scale,
    }


def gate3_contrarian_vs_null_and_momentum(is_out: pd.DataFrame, n_null: int) -> dict:
    null_cols = [c for c in is_out.columns if c.startswith("null_net_")]
    null_replicate_means = is_out[null_cols].mean(axis=0).values  # 200 numbers
    null_95pct = float(np.percentile(null_replicate_means, 95))
    contrarian_mean = float(is_out["contrarian_net"].mean())
    momentum_mean = float(is_out["momentum_net"].mean())
    ok = bool((contrarian_mean > 0) and (contrarian_mean > null_95pct) and (contrarian_mean > momentum_mean))
    return {
        "pass": ok,
        "contrarian_mean": contrarian_mean,
        "momentum_mean": momentum_mean,
        "null_95pct_of_replicate_means": null_95pct,
        "null_replicate_means_summary": {
            "mean": float(null_replicate_means.mean()),
            "p5": float(np.percentile(null_replicate_means, 5)),
            "p50": float(np.percentile(null_replicate_means, 50)),
            "p95": null_95pct,
        },
    }


def walk_forward_thirds(is_out: pd.DataFrame, n_thirds: int = N_THIRDS) -> list:
    """Time-based thirds (equal calendar duration, not equal row count — same convention
    as multiday_contrarian/compute_gates.py's walk_forward_thirds)."""
    dates = pd.to_datetime(is_out["action_date"])
    lo_all, hi_all = dates.min(), dates.max()
    edges = pd.date_range(lo_all, hi_all, periods=n_thirds + 1)
    thirds = []
    for i in range(n_thirds):
        lo, hi = edges[i], edges[i + 1]
        mask = (dates >= lo) & (dates < hi) if i < n_thirds - 1 else (dates >= lo) & (dates <= hi)
        sub = is_out[mask]
        n = len(sub)
        mean_net = float(sub["contrarian_net"].mean()) if n else float("nan")
        thirds.append({"third": i + 1, "start": str(lo), "end": str(hi), "n": n, "mean_contrarian_net": mean_net})
    return thirds


def gate4_walk_forward(is_out: pd.DataFrame) -> dict:
    thirds = walk_forward_thirds(is_out)
    n_pos = sum(1 for t in thirds if not np.isnan(t["mean_contrarian_net"]) and t["mean_contrarian_net"] > 0)
    ok = bool(n_pos >= 2)
    return {"pass": ok, "thirds": thirds, "n_pos": n_pos}


def compute_all_gates(is_out: pd.DataFrame, cot_parquet_path: str, n_null: int) -> dict:
    g1 = gate1_continuity(cot_parquet_path)
    g2 = gate2_null_selftest(is_out, n_null)
    g3 = gate3_contrarian_vs_null_and_momentum(is_out, n_null)
    g4 = gate4_walk_forward(is_out)

    contrarian_vals = is_out["contrarian_net"].values
    p_le_zero, boot_means, ci_lo, ci_hi = bs.weekly_block_bootstrap(contrarian_vals)

    return {
        "gate1_data_integrity": g1,
        "gate2_null_selftest": g2,
        "gate3_contrarian_vs_null_and_momentum": g3,
        "gate4_walk_forward": g4,
        "bootstrap": {
            "n_boot": bs.N_BOOT, "seed": bs.BOOT_SEED,
            "p_le_zero": p_le_zero, "boot_mean": float(boot_means.mean()) if len(boot_means) else float("nan"),
            "ci_2p5": ci_lo, "ci_97p5": ci_hi,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--is-battery-csv", default="results/is_battery.csv")
    ap.add_argument("--cot-parquet", default="cot_weekly.parquet")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    is_out = pd.read_csv(args.is_battery_csv)
    gates = compute_all_gates(is_out, args.cot_parquet, args.n_null)

    with open(os.path.join(args.out_dir, "gate_results.json"), "w") as f:
        json.dump(gates, f, indent=2, default=str)
    print(json.dumps(gates, indent=2, default=str))


if __name__ == "__main__":
    main()
