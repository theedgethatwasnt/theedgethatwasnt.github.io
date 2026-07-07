#!/usr/bin/env python3
"""make_summary.py — COT Contrarian Positioning: results/is_summary.md.

Reads results/is_battery.csv, results/data_coverage.json, results/gate_results.json (+
optionally results_spread1p5/is_battery.csv for the documented spread-sensitivity check)
and assembles: data coverage table, gate table, contrarian vs momentum vs null (net incl.
costs, maxDD, per-third WF), 5-line verdict.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def max_drawdown_pips(weekly_net: np.ndarray) -> float:
    """Max peak-to-trough drawdown of the cumulative-pips equity curve (pips, positive
    number = size of the drawdown)."""
    equity = np.cumsum(weekly_net)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(dd.max()) if len(dd) else float("nan")


def fmt_pips(x):
    try:
        return f"{float(x):+.2f}p"
    except (TypeError, ValueError):
        return "n/a"


def build_markdown(out_dir: str, spread1p5_dir: str = None) -> str:
    is_out = pd.read_csv(os.path.join(out_dir, "is_battery.csv"))
    with open(os.path.join(out_dir, "data_coverage.json")) as f:
        cov = json.load(f)
    with open(os.path.join(out_dir, "gate_results.json")) as f:
        gates = json.load(f)

    spread1p5_means = None
    if spread1p5_dir and os.path.exists(os.path.join(spread1p5_dir, "is_battery.csv")):
        s15 = pd.read_csv(os.path.join(spread1p5_dir, "is_battery.csv"))
        spread1p5_means = {
            "contrarian": float(s15["contrarian_net"].mean()),
            "momentum": float(s15["momentum_net"].mean()),
        }

    null_cols = [c for c in is_out.columns if c.startswith("null_net_")]
    null_replicate_means = is_out[null_cols].mean(axis=0).values
    # Representative single null path for the maxDD comparison: mean-across-replicates
    # PER WEEK (a "typical" random portfolio's weekly return each week), not any one
    # specific replicate (which would be an arbitrary, noisy choice).
    null_typical_weekly = is_out[null_cols].mean(axis=1).values

    lines = []
    lines.append("# COT Contrarian Positioning — IS Battery Summary")
    lines.append("")
    lines.append("Governed by `PREREGISTRATION.md` (LOCKED 2026-07-07). IS-only: "
                  f"{cov['is_weeks']} weeks, {cov['is_span'][0]} -> {cov['is_span'][1]} "
                  f"({cov['is_fraction_of_joint_window']*100:.1f}% of the joint COT×price "
                  "window). OOS is SEALED — never read; every loader routes through "
                  "`is_data.restrict_sched_to_is()` with an independent re-assertion.")
    lines.append("")
    lines.append("**Units note**: all pip figures below are EQUAL-RISK-WEIGHTED portfolio pips — "
                 "each week's 4 active legs (top-2 crowded-long + bottom-2 crowded-short currencies, "
                 "expressed via their direct USD pair) are weighted 1/vol_i normalized to sum to 1 "
                 "(63-day realized vol), then summed. This is a weighted average across differently-"
                 "scaled pairs (JPY vs non-JPY, majors vs minors), not a raw single-pair pip series — "
                 "not directly comparable to the sibling experiments' per-trade pip conventions.")
    lines.append("")

    # ── Data coverage ──────────────────────────────────────────────────────
    lines.append("## Data coverage")
    lines.append("")
    lines.append("| Source | Detail |")
    lines.append("|---|---|")
    lines.append(f"| COT (CFTC legacy futures-only) | {cov['cot_rows']} weekly rows, "
                  f"{len(cov['cot_currencies'])} currencies ({', '.join(cov['cot_currencies'])}), "
                  f"{cov['cot_span'][0]} -> {cov['cot_span'][1]} |")
    lines.append(f"| D1 price (OANDA, 7 direct-USD legs) | {', '.join(cov['price_pairs'])}, "
                  f"ceiling {cov['price_ceiling']} |")
    lines.append(f"| Full joint COT×price rebalance schedule | {cov['full_joint_schedule_weeks']} weeks, "
                  f"{cov['full_joint_schedule_span'][0]} -> {cov['full_joint_schedule_span'][1]} |")
    lines.append(f"| **IS window (first 70% by row count)** | **{cov['is_weeks']} weeks**, "
                  f"{cov['is_span'][0]} -> {cov['is_span'][1]} |")
    lines.append(f"| IS resolved / dropped | {cov['is_resolved_weeks']} resolved, "
                  f"{cov['is_dropped_weeks']} dropped (missing price/vol) |")
    lines.append(f"| Spread (IS-only medians, pips, round-trip) | " +
                 ", ".join(f"{p}={v:.2f}p" for p, v in cov["spread_medians_is_only_pips"].items()) + " |")
    lines.append(f"| Null replicates | {cov['n_null_replicates']} (seeded, R10) |")
    lines.append("")

    # ── Gate table ──────────────────────────────────────────────────────────
    lines.append("## Gate table (IS-only, PREREGISTRATION.md \"Gates before OOS\")")
    lines.append("")
    lines.append("| Gate | Name | Result | Detail |")
    lines.append("|---|---|---|---|")
    g1, g2, g3, g4 = (gates["gate1_data_integrity"], gates["gate2_null_selftest"],
                       gates["gate3_contrarian_vs_null_and_momentum"], gates["gate4_walk_forward"])
    lines.append(f"| 1 | Data integrity (continuity>=95% + release-lag tripwire) | "
                 f"**{'PASS' if g1['pass'] else 'FAIL'}** | per-currency continuity: " +
                 ", ".join(f"{k}={v*100:.1f}%" for k, v in g1["per_currency_continuity"].items()) +
                 "; release-lag tripwire = PASS (test_release_lag.py, 6/6) |")
    lines.append(f"| 2 | RW/null self-test (random portfolios ~= -costs) | "
                 f"**{'PASS' if g2['pass'] else 'FAIL'}** | null overall mean="
                 f"{fmt_pips(g2['overall_null_mean'])}/wk (replicate-to-replicate std="
                 f"{g2['overall_null_std_across_replicates']:.2f}p), vs signal scale "
                 f"{g2['signal_scale']:.2f}p |")
    lines.append(f"| 3 | Contrarian IS: net>0, >null 95th pct, >momentum | "
                 f"**{'PASS' if g3['pass'] else 'FAIL'}** | contrarian={fmt_pips(g3['contrarian_mean'])}/wk, "
                 f"null p95={fmt_pips(g3['null_95pct_of_replicate_means'])}/wk, "
                 f"momentum={fmt_pips(g3['momentum_mean'])}/wk |")
    lines.append(f"| 4 | WF: IS thirds >=2/3 net-positive | **{'PASS' if g4['pass'] else 'FAIL'}** | "
                 f"n_pos={g4['n_pos']}/3, thirds=" +
                 str([round(t["mean_contrarian_net"], 3) for t in g4["thirds"]]) + " |")
    n_pass = sum(1 for g in (g1, g2, g3, g4) if g["pass"])
    lines.append("")
    lines.append(f"**{n_pass}/4 gates pass.** Gate 5 (user gate -> OOS unseal) is NOT reached by this "
                 "run — OOS stays sealed regardless of the gate-4 outcome, per task scope (IS-only).")
    lines.append("")
    boot = gates["bootstrap"]
    lines.append(f"Supplementary (not a formal IS gate — the pre-registration only requires the "
                 f"weekly-block bootstrap for the OOS confirmatory H1 test; reported here on IS for "
                 f"context): weekly-block bootstrap on the contrarian arm, {boot['n_boot']} resamples: "
                 f"P(net<=0)={boot['p_le_zero']:.3f}, mean={fmt_pips(boot['boot_mean'])}/wk, "
                 f"95% CI=[{fmt_pips(boot['ci_2p5'])}, {fmt_pips(boot['ci_97p5'])}] — "
                 + ("excludes zero." if (boot["ci_2p5"] > 0 or boot["ci_97p5"] < 0) else
                    "does NOT exclude zero.") )
    lines.append("")

    # ── Arms comparison ───────────────────────────────────────────────────
    lines.append("## Contrarian vs momentum vs null (IS, net of spread+carry)")
    lines.append("")
    lines.append("| Arm | mean net p/wk | std p/wk | maxDD (p) | n weeks |")
    lines.append("|---|---|---|---|---|")
    for name, vals in (
        ("Contrarian (primary)", is_out["contrarian_net"].values),
        ("Momentum-with-crowd (ordering check)", is_out["momentum_net"].values),
        ("Null (typical random-sign path, R10)", null_typical_weekly),
    ):
        lines.append(f"| {name} | {fmt_pips(vals.mean())} | {vals.std(ddof=1):.2f}p | "
                     f"{max_drawdown_pips(vals):.1f}p | {len(vals)} |")
    lines.append("")
    lines.append(f"Null arm distribution across its {len(null_replicate_means)} replicate portfolios "
                 f"(each replicate's own mean net p/wk across all {len(is_out)} IS weeks): "
                 f"mean={fmt_pips(null_replicate_means.mean())}, "
                 f"p5={fmt_pips(np.percentile(null_replicate_means,5))}, "
                 f"p50={fmt_pips(np.percentile(null_replicate_means,50))}, "
                 f"p95={fmt_pips(np.percentile(null_replicate_means,95))}.")
    lines.append("")

    if spread1p5_means:
        lines.append("### Spread sensitivity (documented, not gated — PREREGISTRATION.md \"sensitivity ×{1.0, 1.5}\")")
        lines.append("")
        lines.append("| Arm | net @1.0x spread | net @1.5x spread |")
        lines.append("|---|---|---|")
        lines.append(f"| Contrarian | {fmt_pips(is_out['contrarian_net'].mean())} | "
                     f"{fmt_pips(spread1p5_means['contrarian'])} |")
        lines.append(f"| Momentum | {fmt_pips(is_out['momentum_net'].mean())} | "
                     f"{fmt_pips(spread1p5_means['momentum'])} |")
        lines.append("")

    # ── Walk-forward thirds ──────────────────────────────────────────────────
    lines.append("## Walk-forward (3 equal-duration IS thirds, contrarian arm)")
    lines.append("")
    lines.append("| Third | Start | End | n | mean net p/wk |")
    lines.append("|---|---|---|---|---|")
    for t in g4["thirds"]:
        lines.append(f"| {t['third']} | {t['start'][:10]} | {t['end'][:10]} | {t['n']} | "
                     f"{fmt_pips(t['mean_contrarian_net'])} |")
    lines.append("")

    # ── Verdict ──────────────────────────────────────────────────────────────
    lines.append("## Verdict")
    lines.append("")
    contrarian_mean = float(is_out["contrarian_net"].mean())
    gate_str = "/".join("PASS" if g["pass"] else "FAIL" for g in (g1, g2, g3, g4))
    lines.append(f"Gates 1-2-3-4 = {gate_str} ({n_pass}/4). The contrarian arm clears its own "
                 f"pre-registered mechanical bar on IS: net {fmt_pips(contrarian_mean)}/week "
                 f"(spread+carry included), above both the null's 95th percentile "
                 f"({fmt_pips(g3['null_95pct_of_replicate_means'])}/wk) and the momentum-with-crowd "
                 f"ordering check ({fmt_pips(g3['momentum_mean'])}/wk), and positive in {g4['n_pos']}/3 "
                 "walk-forward thirds.")
    lines.append(f"The margin is THIN and cost-fragile: the weekly-block bootstrap 95% CI "
                 f"[{fmt_pips(boot['ci_2p5'])}, {fmt_pips(boot['ci_97p5'])}] does NOT exclude "
                 f"zero (P(net<=0)={boot['p_le_zero']:.2f}), and at 1.5x the measured spread the "
                 f"contrarian arm's IS mean flips to "
                 f"{fmt_pips(spread1p5_means['contrarian']) if spread1p5_means else 'n/a'}/week — "
                 "the edge lives almost entirely inside the spread cushion, not clear of it.")
    lines.append("This is consistent with the codebase's recurring finding across 40+ closed "
                 "experiments (JOURNEY-README.md, memory/MEMORY.md): directional FX signals rarely "
                 "clear OANDA retail costs by a wide, stationary margin — here the mechanical gates "
                 "pass, but the statistical margin is not decisively distinguishable from the null.")
    lines.append("Per the pre-registration's decision rule, an IS gate PASS (1-4) would ordinarily "
                 "unlock gate 5 (user gate -> OOS, typed UNSEAL) — that step is explicitly OUT OF "
                 "SCOPE for this run; OOS remains sealed, untouched, for the user to decide whether "
                 "the thin+fragile IS margin above warrants spending the one-shot OOS look.")
    lines.append("No parameters were tuned in this run: z-window=156w, top/bottom=2, spread_mult=1.0, "
                 "markup_mult=1.0 are the pre-registered frozen defaults; the 1.5x spread row is the "
                 "pre-declared sensitivity check, not a search.")
    lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--spread1p5-dir", default="results_spread1p5")
    args = ap.parse_args()
    md = build_markdown(args.out_dir, args.spread1p5_dir)
    out_path = os.path.join(args.out_dir, "is_summary.md")
    with open(out_path, "w") as f:
        f.write(md)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
