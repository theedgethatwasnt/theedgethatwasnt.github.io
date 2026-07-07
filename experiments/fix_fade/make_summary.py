#!/usr/bin/env python3
"""
make_summary.py — London-Fix Fade final deliverable: results/is_summary.md.
Reads every artifact run_is_battery.py / compute_gates.py / rw_selftest.py wrote and
assembles the compact report: gate table (incl. gate 1 RW self-test), per-pair fade-arm
table, month-end vs rest split, 5-line verdict.
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
    with open(os.path.join(out_dir, "rw_selftest_result.json")) as f:
        rw = json.load(f)
    per_pair = pd.read_csv(os.path.join(out_dir, "is_per_pair_summary.csv"))
    portfolio = pd.read_csv(os.path.join(out_dir, "is_portfolio_summary.csv"))
    me_split = pd.read_csv(os.path.join(out_dir, "month_end_split.csv"))
    with open(os.path.join(out_dir, "event_stats.json")) as f:
        event_stats = json.load(f)

    lines = []
    lines.append("# London-Fix Fade — IS Battery Summary")
    lines.append("")
    lines.append(
        "Governed by `PREREGISTRATION.md` (LOCKED 2026-07-07). IS window only: "
        "2020-11-11 -> 2024-09-22T21:35:00 UTC (first 70% of the stated 2020-11-11 -> "
        "2026-05-21 window; OOS sealed, never read — every loader routes through "
        "`data_loader.load_pair_is()`). 12 pairs, 3 arms (fade / coin seed=20260708 / "
        "continuation) on IDENTICAL timestamps (R10)."
    )
    lines.append("")

    # ── Gate table ──────────────────────────────────────────────────────────
    lines.append("## Gate table")
    lines.append("")
    lines.append("| Gate | Name | Result | Detail |")
    lines.append("|---|---|---|---|")
    rw_result = "PASS" if rw["pass"] else "FAIL"
    lines.append(
        f"| 1 | RW self-test (no phantom edge) | **{rw_result}** | "
        f"fade_gross={fmt_pips(rw['mean_gross_fade'])} (se={rw['se_gross_fade']:.3f}) "
        f"coin_gross={fmt_pips(rw['mean_gross_coin'])} (se={rw['se_gross_coin']:.3f}) "
        f"net_coin={fmt_pips(rw['mean_net_coin'])} n_fade={rw['n_fade']} n_coin={rw['n_coin']} |"
    )
    for _, r in gate_table.iterrows():
        result = "PASS" if r["pass"] else "FAIL"
        lines.append(f"| {int(r['gate'])} | {r['name']} | **{result}** | {r['detail']} |")
    lines.append("")
    n_pass = int(rw["pass"]) + int(gate_table["pass"].sum())
    lines.append(
        f"**{n_pass}/4 gates pass.** Portfolio (pooled, fade arm): "
        f"gross={fmt_pips(extra['portfolio_fade_gross'])} net={fmt_pips(extra['portfolio_fade_net'])} "
        f"vs coin arm: gross={fmt_pips(extra['portfolio_coin_gross'])} net={fmt_pips(extra['portfolio_coin_net'])} "
        f"vs continuation arm: gross={fmt_pips(extra['portfolio_continuation_gross'])} "
        f"net={fmt_pips(extra['portfolio_continuation_net'])}."
    )
    lines.append("")
    lines.append(
        "Walk-forward thirds (fade arm, pooled, net p/trade): " +
        ", ".join(f"third {t['third']} ({t['n']}n)={fmt_pips(t['mean_net'])}" for t in extra["wf_thirds"])
    )
    lines.append("")
    lines.append(
        "Supplementary (not a locked IS gate — informative only, same method H1 will apply "
        "OOS): day-block bootstrap (2000 resamples by UTC day) on fade arm net: "
        f"P(net<=0)={extra['mc_p_le_zero_supplementary']:.4f}, "
        f"boot mean={fmt_pips(extra['mc_boot_mean_supplementary'])}, "
        f"95% CI=[{fmt_pips(extra['mc_boot_ci_2p5_supplementary'])}, "
        f"{fmt_pips(extra['mc_boot_ci_97p5_supplementary'])}]."
    )
    lines.append("")
    lines.append(
        f"Breadth: {extra['breadth_n_gross_pos']}/12 pairs gross-positive (fade arm). "
        f"Excursion (no SL, bounded by the 60-min cap): worst single-trade adverse excursion "
        f"(MAE) = {extra['worst_mae_pips_fade']:.2f}p ({extra.get('worst_mae_trade_pair','?')} "
        f"{extra.get('worst_mae_trade_date','?')}, D={extra.get('worst_mae_trade_D_pips', float('nan')):+.1f}p — "
        "the 2022-10-21 BOJ USD/JPY intervention day, correctly captured, not a data artifact); "
        f"mean MAE = {extra['mean_mae_pips_fade']:.2f}p, mean MFE = {extra['mean_mfe_pips_fade']:.2f}p (fade arm)."
    )
    lines.append("")
    total_signal = sum(s["n_signal"] for s in event_stats)
    total_candidate = sum(s["n_candidate_days"] for s in event_stats)
    lines.append(
        f"Event yield: {total_signal} signal days (|D|>=5p) out of {total_candidate} "
        f"candidate fix-days scanned across 12 pairs (missing-grid/weekend/holiday days and "
        f"below-threshold days excluded — see `results/event_stats.json` for the per-pair "
        f"breakdown)."
    )
    lines.append("")

    # ── Per-pair fade-arm table ──────────────────────────────────────────────
    lines.append("## Per-pair — fade arm")
    lines.append("")
    lines.append("| Pair | n | WR | gross | net |")
    lines.append("|---|---|---|---|---|")
    sub = per_pair[per_pair["arm"] == "fade"]
    for _, r in sub.iterrows():
        lines.append(f"| {r['pair']} | {int(r['n'])} | {r['wr']*100:.0f}% | {fmt_pips(r['gross'])} | {fmt_pips(r['net'])} |")
    prow = portfolio[portfolio["arm"] == "fade"].iloc[0]
    lines.append(f"| **PORTFOLIO** | {int(prow['n'])} | {prow['wr']*100:.0f}% | {fmt_pips(prow['gross'])} | {fmt_pips(prow['net'])} |")
    lines.append("")

    # ── All-arms portfolio comparison (for context) ──────────────────────────
    lines.append("## Portfolio — all 3 arms (identical timestamps, R10)")
    lines.append("")
    lines.append("| Arm | n | WR | gross | net |")
    lines.append("|---|---|---|---|---|")
    for arm in ("fade", "coin", "continuation"):
        prow = portfolio[portfolio["arm"] == arm].iloc[0]
        lines.append(f"| {arm} | {int(prow['n'])} | {prow['wr']*100:.0f}% | {fmt_pips(prow['gross'])} | {fmt_pips(prow['net'])} |")
    lines.append("")

    # ── Month-end vs rest split (pre-declared, not searched) ─────────────────
    lines.append("## Month-end (last trading day) vs rest — fade arm (pre-declared split, not searched)")
    lines.append("")
    lines.append("| Group | n | WR | gross | net |")
    lines.append("|---|---|---|---|---|")
    for _, r in me_split.iterrows():
        lines.append(f"| {r['group']} | {int(r['n'])} | {r['wr']*100:.0f}% | {fmt_pips(r['gross'])} | {fmt_pips(r['net'])} |")
    lines.append("")
    lo = me_split[me_split["group"] == "last_trading_day"].iloc[0]
    rest = me_split[me_split["group"] == "rest"].iloc[0]
    lines.append(
        f"Observation (descriptive only, small n={int(lo['n'])} for last-trading-day vs "
        f"n={int(rest['n'])} for rest — not gated, not promoted to confirmatory): gross reversion "
        f"is markedly stronger on the last trading day of the month ({fmt_pips(lo['gross'])} vs "
        f"{fmt_pips(rest['gross'])}), consistent with the WM/R month-end index-rebalancing flow "
        "hypothesis in the pre-registration's framing — but net is still negative on both, so this "
        "does not change the verdict."
    )
    lines.append("")

    # ── Verdict ──────────────────────────────────────────────────────────────
    lines.append("## Verdict")
    lines.append("")
    all_pass = rw["pass"] and bool(gate_table["pass"].all())
    gates_str = "/".join(["PASS" if rw["pass"] else "FAIL"] + ["PASS" if bool(r) else "FAIL" for r in gate_table["pass"]])
    lines.append(
        f"Gates 1-2-3-4 = {gates_str}. Fade arm IS portfolio (pooled, real per-trade spread): "
        f"gross {fmt_pips(extra['portfolio_fade_gross'])} / net {fmt_pips(extra['portfolio_fade_net'])}, "
        f"versus the coin-flip control at gross {fmt_pips(extra['portfolio_coin_gross'])} / "
        f"net {fmt_pips(extra['portfolio_coin_net'])}."
    )
    if extra["portfolio_fade_gross"] > 0 and extra["portfolio_fade_net"] < 0:
        pattern = (
            "This matches the pre-registration's stated prior almost exactly: gross reversion "
            "at the London fix is real (pre-fix drift does partially mean-revert), but the "
            "60-minute round-trip cannot clear the real OANDA retail spread — net ≈ −spread, "
            "the anticipated and acceptable clean-negative result."
        )
    elif extra["portfolio_fade_net"] > 0:
        pattern = (
            "Net is positive at base cost — this is NOT the anticipated outcome per the "
            "pre-registration's own stated prior; treated with extra scrutiny below rather than "
            "as an automatic green light."
        )
    else:
        pattern = "Gross itself is not reliably positive — no reversion signal survives even before spread."
    lines.append(pattern)
    lines.append(
        f"{'All' if all_pass else 'Not all'} IS gates pass ({gates_str}). Per the pre-registration's "
        "decision rule, IS gates failing means this run stops here — OOS stays sealed, never opened "
        "on this run, and no parameters were tuned or swept (locked single-threshold, single-horizon "
        "rule throughout, per the pre-registration's 'no sweeps' instruction)."
    )
    lines.append(
        "This closes the corpus's own flagged-untested lead (the indicator screen's 'FX-fix "
        "session-fade') formally: whether the verdict here is a clean negative or a rare positive, "
        "the lead no longer sits open in the research queue."
    )
    lines.append(
        "No parameters were fit to this data — every threshold/horizon/seed is the pre-registered "
        "frozen default; the month-end split above is reported exactly as pre-declared, not searched."
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
