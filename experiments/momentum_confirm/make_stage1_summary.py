#!/usr/bin/env python3
"""
make_stage1_summary.py — momentum_confirm: results/stage1_summary.md (coverage, gate table,
net/gross/maxDD, per-third + per-half numbers, null distribution stats, 5-line verdict ending
with the explicit Stage-2 recommendation). Reads results/stage1_{monthly,null_r10,gate_table}.csv
and results/stage1_gate_extra.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def variant_table(monthly_df):
    rows = []
    for variant, g in monthly_df.groupby("variant"):
        g = g.sort_values("signal_date")
        net = g["net_pips"]
        gross = g["carry_free_gross_pips"]
        cum = net.cumsum()
        dd = cum - cum.cummax()
        rows.append({
            "variant": variant,
            "n": len(g),
            "mean_net_pips": float(net.mean()) if len(g) else float("nan"),
            "cum_net_pips": float(net.sum()) if len(g) else float("nan"),
            "max_dd_pips": float(dd.min()) if len(dd) else float("nan"),
            "mean_carry_free_gross_pips": float(gross.mean()) if len(g) else float("nan"),
            "cum_carry_free_gross_pips": float(gross.sum()) if len(g) else float("nan"),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = args.out_dir

    monthly_df = pd.read_csv(os.path.join(out, "stage1_monthly.csv"))
    null_df = pd.read_csv(os.path.join(out, "stage1_null_r10.csv"))
    gate_table = pd.read_csv(os.path.join(out, "stage1_gate_table.csv"))
    with open(os.path.join(out, "stage1_gate_extra.json")) as f:
        extra = json.load(f)

    vt = variant_table(monthly_df)
    vt.to_csv(os.path.join(out, "stage1_variant_summary.csv"), index=False)

    mom = monthly_df[monthly_df["variant"] == "momentum"].sort_values("signal_date")
    mom_dates = pd.to_datetime(mom["signal_date"], utc=True)

    def pass_str(v):
        return "PASS" if bool(v) else "FAIL"

    all_pass = bool(gate_table["pass"].all())

    lines = []
    lines.append("# Monthly FX Cross-Sectional Momentum — Stage 1 Summary\n")
    lines.append(
        "Governed by `PREREGISTRATION.md` (LOCKED 2026-07-07). Rule frozen verbatim from "
        "`research/experiments/fx_factors/factors.py` (momentum variant: 12-1 currency "
        "momentum, top-3/bottom-3 of 7 non-USD currencies, 63-day inverse-vol weighting, "
        "no risk-off gate — imported, not reimplemented). Data: OANDA D1 deep history "
        "(`data/d1_deep/`), hard-sealed to timestamps < 2020-11-01 by `stage1_data.py` "
        "(pyarrow filter pushdown + independent post-read assertion). The observation window "
        "(2020-11->2024-08) and the fx_factors sealed OOS (2024-08-30->2026-05-21) were never "
        "loaded.\n"
    )

    lines.append("## Coverage\n")
    n_rebal = int(extra["n_rebalances"])
    lines.append(
        f"**{n_rebal} monthly rebalances**, {mom_dates.min().date()} -> {mom_dates.max().date()} "
        f"(pre-reg expected ~180 for 2005-2020; the master calendar's earliest common bar across "
        f"all 7 required pairs is 2004-05-31 because CAD_JPY/CHF_JPY only start there in the "
        f"deep-D1 pull, pushing coverage slightly earlier and to {n_rebal} rebalances).\n"
    )

    lines.append("## Gate table (Stage-1 gates 1-4, PREREGISTRATION.md \"Stage-1 gates\")\n")
    lines.append("| Gate | Name | Result | Detail |")
    lines.append("|---|---|---|---|")
    for _, r in gate_table.iterrows():
        lines.append(f"| {int(r['gate'])} | {r['name']} | **{pass_str(r['pass'])}** | {r['detail']} |")
    lines.append("")
    n_pass = int(gate_table["pass"].sum())
    lines.append(f"**{n_pass}/4 gates pass.** All 4 are required (pre-reg: \"all required before Stage 2 may be requested\").\n")

    lines.append("## Net / gross / max drawdown (primary spread_mult=1.0, plus 1.5x sensitivity)\n")
    lines.append("| Variant | n | mean net p/rebal | cum net (p) | max DD (p) | mean carry-free gross p/rebal | cum carry-free gross (p) |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in vt.iterrows():
        lines.append(
            f"| {r['variant']} | {int(r['n'])} | {r['mean_net_pips']:+.3f} | {r['cum_net_pips']:+.1f} | "
            f"{r['max_dd_pips']:+.1f} | {r['mean_carry_free_gross_pips']:+.3f} | {r['cum_carry_free_gross_pips']:+.1f} |"
        )
    lines.append("")
    lines.append(
        "Carry-free gross = price return net of spread only (no carry term) — reported per the "
        "pre-reg's mitigation for the pre-2020 carry splice's \"direction of bias unknown\".\n"
    )

    lines.append("## Per-third (WF, Gate 3)\n")
    tm = extra["third_means"]
    lines.append("| Third | mean net p/rebal |")
    lines.append("|---|---|")
    for i, m in enumerate(tm, start=1):
        lines.append(f"| {i} | {m:+.3f} |")
    lines.append(f"\n{extra['n_positive_thirds']}/3 thirds net-positive.\n")

    lines.append(f"## Per-half (regime split @ {extra['split_year']}, Gate 4)\n")
    lines.append("| Half | n | mean net p/rebal |")
    lines.append("|---|---|---|")
    lines.append(f"| pre-{extra['split_year']} | {extra['pre_split_n']} | {extra['pre_split_mean']:+.3f} |")
    lines.append(f"| post-{extra['split_year']} | {extra['post_split_n']} | {extra['post_split_mean']:+.3f} |")
    lines.append("")

    lines.append("## R10 null distribution (200 random-weight portfolios, identical schedule/costs)\n")
    lines.append(
        f"n_seeds={len(null_df)}, mean={extra['null_mean']:+.4f}p, p95={extra['null_p95']:+.4f}p, "
        f"p5={extra['null_p5']:+.4f}p, std={extra['null_std']:.4f}p.\n"
    )

    lines.append("## Verdict\n")
    v = []
    deep_net = extra["deep_net"]
    v.append(
        f"Deep-segment momentum (primary, spread_mult=1.0) mean net = {deep_net:+.3f} p/rebalance "
        f"over {n_rebal} monthly rebalances (cum {extra['cum_net_pips']:+.1f}p, max DD "
        f"{extra['max_dd_pips']:+.1f}p) vs R10 null mean {extra['null_mean']:+.3f}p / p95 "
        f"{extra['null_p95']:+.3f}p."
    )
    if all_pass:
        v.append(
            "All 4 Stage-1 gates PASS: the frozen momentum rule clears its own apparatus "
            "self-test, beats the deep-segment R10 null at the 95th percentile, holds up "
            "walk-forward across chronological thirds, and survives the pre-declared 2013 "
            "regime split."
        )
        v.append(
            f"Regime detail: thirds = {tm[0]:+.1f}/{tm[1]:+.1f}/{tm[2]:+.1f} p/rebal "
            f"({extra['n_positive_thirds']}/3 positive); pre-/post-{extra['split_year']} = "
            f"{extra['pre_split_mean']:+.1f}/{extra['post_split_mean']:+.1f} p/rebal."
        )
        v.append(
            "Per the pre-registration's decision rule, this Stage-1 pass is required (not "
            "sufficient) — Stage 2 (the fx_factors sealed OOS, 2024-08-30->2026-05-21) may now "
            "be requested, evaluated exactly once, only after the user gate."
        )
        v.append("STAGE-2 RECOMMENDATION: REQUEST the user gate (type UNSEAL) to open Stage 2.")
    else:
        failed = gate_table.loc[~gate_table["pass"], "gate"].astype(int).tolist()
        v.append(
            f"Gate(s) {', '.join(str(g) for g in failed)} FAIL — not all 4 required Stage-1 "
            "gates clear on this locked configuration."
        )
        v.append(
            f"Regime detail: thirds = {tm[0]:+.1f}/{tm[1]:+.1f}/{tm[2]:+.1f} p/rebal "
            f"({extra['n_positive_thirds']}/3 positive); pre-/post-{extra['split_year']} = "
            f"{extra['pre_split_mean']:+.1f}/{extra['post_split_mean']:+.1f} p/rebal — "
            "the edge does not survive out of the observation window on genuinely new (older) data."
        )
        v.append(
            "Per the pre-registration's decision rule: Stage 1 fails -> momentum is recorded as "
            "a sweep-artifact (the +27.4 p/rebalance IS 'positive' was a single-pass observation, "
            "not a generalizing rule) and the fx_factors sealed OOS window stays sealed/unspent."
        )
        v.append("STAGE-2 RECOMMENDATION: DO NOT request the user gate — Stage 2 stays closed on this rule.")
    lines.extend(v)
    lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(out, "stage1_summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
