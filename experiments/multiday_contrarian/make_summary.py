#!/usr/bin/env python3
"""
make_summary.py — Task A5 final deliverable: results/is_summary.md.
Reads every artifact the battery/gates/secondary scripts wrote to --out-dir and
assembles the compact report: gate table, per-pair tables, secondaries, 5-line verdict.
"""
import argparse
import json
import os

import pandas as pd


def fmt_pips(x):
    try:
        return f"{float(x):+.2f}p"
    except (TypeError, ValueError):
        return "n/a"


def build_markdown(out_dir):
    gate_table = pd.read_csv(os.path.join(out_dir, "gate_table.csv"))
    with open(os.path.join(out_dir, "gate_extra.json")) as f:
        extra = json.load(f)
    per_pair = pd.read_csv(os.path.join(out_dir, "is_per_pair_summary.csv"))
    portfolio = pd.read_csv(os.path.join(out_dir, "is_portfolio_summary.csv"))

    ss_summary = {}
    ss_path = os.path.join(out_dir, "secondary_strengthspread_summary.json")
    if os.path.exists(ss_path):
        with open(ss_path) as f:
            ss_summary = json.load(f)

    rsi2_summary = {}
    rsi2_path = os.path.join(out_dir, "secondary_rsi2_summary.json")
    if os.path.exists(rsi2_path):
        with open(rsi2_path) as f:
            rsi2_summary = json.load(f)

    eqr = {}
    eqr_path = os.path.join(out_dir, "secondary_equal_risk.json")
    if os.path.exists(eqr_path):
        with open(eqr_path) as f:
            eqr = json.load(f)

    lines = []
    lines.append("# Task A5 — Multi-Day Contrarian Program: IS Battery Summary")
    lines.append("")
    lines.append("Governed by `PREREGISTRATION.md` (LOCKED 2026-07-06) + Amendment 1. "
                  "IS window only: 2020-11-11 -> 2024-09-25 (OOS sealed, never read — every "
                  "loader in this battery hard-filters via `is_data.load_pair_is()`).")
    lines.append("")

    # ── Gate table ──────────────────────────────────────────────────────────
    lines.append("## Gate table (gates 3-6; gate 1 = harness self-test PASS in test_harness.py, "
                 "gate 2 = SATISFIED-IN-PURPOSE/FAILED-IN-LETTER per Amendment 1)")
    lines.append("")
    lines.append("| Gate | Name | Result | Detail |")
    lines.append("|---|---|---|---|")
    for _, r in gate_table.iterrows():
        result = "PASS" if r["pass"] else "FAIL"
        lines.append(f"| {int(r['gate'])} | {r['name']} | **{result}** | {r['detail']} |")
    lines.append("")
    n_pass = int(gate_table["pass"].sum())
    lines.append(f"**{n_pass}/4 gates pass.** Portfolio (pooled, signal arm, base cost): "
                 f"{fmt_pips(extra['portfolio_signal_net_base'])} net vs coin arm "
                 f"{fmt_pips(extra['portfolio_coin_net_base'])} net vs continuation arm "
                 f"{fmt_pips(extra['portfolio_continuation_net_base'])} net.")
    lines.append("")
    lines.append(f"Walk-forward thirds (signal arm, pooled, net_base p/trade): " +
                 ", ".join(f"third {t['third']} ({t['n']}n)={fmt_pips(t['mean_net_base'])}"
                           for t in extra["wf_thirds"]))
    lines.append("")
    lines.append(f"Day-block bootstrap (2000 resamples by UTC day): "
                 f"P(net<=0)={extra['mc_p_le_zero']:.4f}, "
                 f"boot mean={fmt_pips(extra['mc_boot_mean'])}, "
                 f"95% CI=[{fmt_pips(extra['mc_boot_ci_2p5'])}, {fmt_pips(extra['mc_boot_ci_97p5'])}].")
    lines.append("")
    lines.append(f"Breadth: {extra['breadth_n_gross_pos']}/12 pairs gross-positive (signal arm).")
    lines.append("")

    # ── Per-pair tables (one per arm) ────────────────────────────────────────
    for arm in ("signal", "coin", "continuation"):
        lines.append(f"## Per-pair — arm={arm}")
        lines.append("")
        lines.append("| Pair | n | WR | gross | net@1.0x | net@spread1.5x | net@carry2.0x | timeout% |")
        lines.append("|---|---|---|---|---|---|---|---|")
        sub = per_pair[per_pair["arm"] == arm]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['pair']} | {int(r['n'])} | {r['wr']*100:.0f}% | {fmt_pips(r['gross'])} | "
                f"{fmt_pips(r['net_base'])} | {fmt_pips(r['net_spread1p5'])} | "
                f"{fmt_pips(r['net_carry2p0'])} | {r['timeout_share']*100:.0f}% |"
            )
        prow = portfolio[portfolio["arm"] == arm].iloc[0]
        lines.append(
            f"| **PORTFOLIO** | {int(prow['n'])} | {prow['wr']*100:.0f}% | {fmt_pips(prow['gross'])} | "
            f"{fmt_pips(prow['net_base'])} | {fmt_pips(prow['net_spread1p5'])} | "
            f"{fmt_pips(prow['net_carry2p0'])} | {prow['timeout_share']*100:.0f}% |"
        )
        lines.append("")

    # ── Secondaries ───────────────────────────────────────────────────────────
    lines.append("## Secondary analyses (exploratory, IS-only, never confirmatory)")
    lines.append("")
    lines.append("### (a) CSI StrengthSpread H4/64-bar port")
    if ss_summary:
        lines.append(
            f"- n_rebalances={ss_summary['n_rebalances']}, n_legs={ss_summary['n_legs']}, "
            f"mean gross/leg={fmt_pips(ss_summary['mean_gross_pips_per_leg'])}, "
            f"mean net/leg={fmt_pips(ss_summary['mean_net_pips_per_leg'])}, "
            f"frac legs net+={ss_summary['frac_legs_net_positive']*100:.0f}%. "
            "(H=64, N=3 taken verbatim from csi_factor_study's recorded prior; 12-pair/8-currency "
            "port, conservative no-turnover-netting cost model — see script docstring.)"
        )
    else:
        lines.append("- not run / summary missing.")
    lines.append("")
    lines.append("### (b) D1 RSI(2) mean-reversion (classic, Wilder smoothing)")
    if rsi2_summary:
        lines.append(
            f"- n={rsi2_summary['n']}, WR={rsi2_summary['wr']*100:.0f}%, "
            f"mean gross={fmt_pips(rsi2_summary['mean_gross_pips'])}, "
            f"mean net@1.0x={fmt_pips(rsi2_summary['mean_net_base'])}, "
            f"net@spread1.5x={fmt_pips(rsi2_summary['mean_net_spread1p5'])}, "
            f"net@carry2.0x={fmt_pips(rsi2_summary['mean_net_carry2p0'])}, "
            f"{rsi2_summary['n_pairs_gross_positive']}/12 pairs gross-positive."
        )
    else:
        lines.append("- not run / summary missing.")
    lines.append("")
    lines.append("### (c) Equal-risk portfolio of IS-positive signals")
    if eqr:
        means_str = ", ".join(f"{k}={fmt_pips(v)}" for k, v in eqr.get("candidate_means_net_pips", {}).items())
        lines.append(f"- candidate means (base cost, **per calendar day**, trades summed within a day "
                     "then averaged across days — NOT the same aggregation as the per-trade means "
                     "above/below; a day with several trades counts once, so this differs from the "
                     "flat per-trade portfolio mean, e.g. first_touch_signal here vs "
                     f"{fmt_pips(extra['portfolio_signal_net_base'])} per-trade in the gate table): {means_str}")
        if eqr.get("portfolio_built"):
            lines.append(
                f"- qualifying (IS-positive): {eqr['qualifying']}, weights={eqr['weights']}, "
                f"n_days={eqr['n_days']}, combined mean/day={fmt_pips(eqr['combined_mean_daily_pips'])}, "
                f"P(net<=0)={eqr['mc_p_le_zero']:.4f}, "
                f"95% CI=[{fmt_pips(eqr['mc_ci_2p5'])}, {fmt_pips(eqr['mc_ci_97p5'])}]."
            )
            if eqr.get("pairwise_correlation"):
                lines.append(f"- pairwise correlation: {eqr['pairwise_correlation']}")
        else:
            lines.append(f"- {eqr.get('note', 'no qualifying signals')}")
    else:
        lines.append("- not run / summary missing.")
    lines.append("")

    # ── Verdict ──────────────────────────────────────────────────────────────
    lines.append("## Verdict")
    lines.append("")
    gates_str = "/".join(("PASS" if bool(r) else "FAIL") for r in gate_table["pass"])
    lines.append(
        f"Gates 3-4-5-6 = {gates_str}. The portfolio (pooled, signal arm, base cost) IS net "
        f"expectancy is {fmt_pips(extra['portfolio_signal_net_base'])}, versus the coin-flip "
        f"control at {fmt_pips(extra['portfolio_coin_net_base'])} — "
        + ("the signal beats its own null." if extra['portfolio_signal_net_base'] > extra['portfolio_coin_net_base']
           else "the signal does NOT clear its own coin-flip null, consistent with Amendment 1's "
                "early uncontrolled read (~-6p IS net) on this corrected NY-17:00 bar grid.")
    )
    lines.append(
        "Per the pre-registration's decision rule: IS gates failing means the program stops here "
        "on this frozen-parameter configuration — OOS stays sealed for a future amended shot, not "
        "opened on this run."
    )
    lines.append(
        "Of the three secondaries, none is presented as confirmatory; each is reported exactly as "
        "computed above with its documented simplifications, and the equal-risk portfolio (c) only "
        "combines whatever subset was IS-positive at base cost, which may be zero, one, two, or three."
    )
    lines.append(
        "This reproduces the program's recurring pattern across the wider FX-Core research history: "
        "intraday/multi-day directional signals rarely clear the OANDA retail spread once carried "
        "through an honest walk-forward + bootstrap gate, net of realistic cost."
    )
    lines.append(
        "No parameters were tuned in this run; all values are the pre-registered frozen defaults or "
        "recorded classic/prior configurations, exactly as required."
    )
    lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    md = build_markdown(args.out_dir)
    out_path = os.path.join(args.out_dir, "is_summary.md")
    with open(out_path, "w") as f:
        f.write(md)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
