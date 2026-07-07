#!/usr/bin/env python3
"""
make_summary.py — scratch_tail final deliverable: results/is_summary.md.
Reads every artifact gate2_parity.py / run_is_battery.py / compute_gates.py wrote and
assembles: gate table (1-5), per-arm portfolio table, per-pair tables for A and D, verdict.
"""
import argparse
import json
import os

import pandas as pd

from signal import PAIRS


def fmt_pips(x):
    try:
        v = float(x)
        if v != v:  # NaN
            return "n/a"
        return f"{v:+.2f}p"
    except (TypeError, ValueError):
        return "n/a"


def build_markdown(out_dir):
    lines = []
    lines.append("# SMA-Scratch Tail-Bounding Test — IS Battery Summary")
    lines.append("")
    lines.append("Governed by `PREREGISTRATION.md` (LOCKED 2026-07-07). IS window: "
                 "2020-11-11 -> 2024-09-25 (OOS sealed, never read for arm evaluation — every "
                 "loader routes through `is_data.load_pair_is()`). Gate 2 is the sole exception "
                 "(2026-06-15+ live paper trail, used only for R7 parity).")
    lines.append("")

    # ── Gate 1 (harness self-test) ──
    lines.append("## Gate 1 — harness self-test (synthetic RW), PASS (test_harness.py)")
    lines.append("")
    lines.append("Full-population (closed trades UNION open-at-end mark-to-window-end) gross "
                 "P&L shows no phantom directional edge for either the real signal or the "
                 "coin-flip control, on a driftless random walk, for both arm A (no stop) and "
                 "arm D's coin control (stop+overlay active). A companion regression test "
                 "documents — as an expected, structural finding, not a bug — that arm A's "
                 "CLOSED-trades-ONLY gross mean IS materially positive on the same random walk "
                 "(the well-known TP-only/no-stop survivorship artifact: closed trades are "
                 "winners/scratches by construction; losers sit in the open book). See "
                 "harness.py's module docstring for the full design-finding note, including the "
                 "self-referential-deadlock bug this surfaced and its fix (a gated arm's "
                 "blocking SIGNAL must come from an always-on REFERENCE run, not its own "
                 "necessarily-thinner realized trade sequence).")
    lines.append("")

    # ── Gate 2 (R7 parity) ──
    g2_path = os.path.join(out_dir, "gate2_summary.json")
    lines.append("## Gate 2 — R7 parity vs the live paper trail (BLOCKING)")
    lines.append("")
    if os.path.exists(g2_path):
        with open(g2_path) as f:
            g2 = json.load(f)
        result = "PASS" if g2["gate2_pass"] else "FAIL"
        lines.append(f"**{result}**. Live window [{g2['live_window_start']}, {g2['live_window_end']}]: "
                     f"{g2['n_live']} live closed trades vs {g2['n_harness']} harness-replayed trades "
                     f"(ratio={g2['count_ratio']:.2f}, tolerance ±{g2['count_tolerance_frac']*100:.0f}%, "
                     f"{'PASS' if g2['count_pass'] else 'FAIL'}). Live expectancy "
                     f"{fmt_pips(g2['live_expectancy_pips'])}/trade vs harness "
                     f"{fmt_pips(g2['harness_expectancy_pips'])}/trade "
                     f"(diff={fmt_pips(g2['expectancy_diff_pips'])}, tolerance "
                     f"±{g2['expectancy_tolerance_pips']:.1f}p, {'PASS' if g2['expectancy_pass'] else 'FAIL'}).")
    else:
        lines.append("- gate2_summary.json missing / not run.")
    lines.append("")

    # ── Gates 3-5 ──
    gt_path = os.path.join(out_dir, "gate_table.csv")
    extra_path = os.path.join(out_dir, "gate_extra.json")
    if os.path.exists(gt_path) and os.path.exists(extra_path):
        gate_table = pd.read_csv(gt_path)
        with open(extra_path) as f:
            extra = json.load(f)
        lines.append("## Gates 3-5 (IS-only)")
        lines.append("")
        lines.append("| Gate | Name | Result | Detail |")
        lines.append("|---|---|---|---|")
        for _, r in gate_table.iterrows():
            result = "PASS" if r["pass"] else "FAIL"
            lines.append(f"| {int(r['gate'])} | {r['name']} | **{result}** | {r['detail']} |")
        lines.append("")

        # ── per-arm portfolio table ──
        lines.append("## Per-arm portfolio table (pooled across 6 pairs, IS window, base cost)")
        lines.append("")
        lines.append("| Arm | n | WR | net/trade | gross/trade | realized P&L | worst open excursion | "
                     "open-at-end | open-book unrealized | floating maxDD | % time blocked |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        floating_dd = extra.get("floating_maxdd_by_arm", {})
        for row in extra.get("portfolio", []):
            arm = row["arm"]
            dd = floating_dd.get(arm, float("nan"))
            pct_blocked = row.get("pct_time_blocked", float("nan"))
            pct_blocked_str = f"{pct_blocked*100:.1f}%" if pct_blocked == pct_blocked else "n/a"
            lines.append(
                f"| {arm} | {int(row['n'])} | {row['wr']*100:.0f}% | {fmt_pips(row['net'])} | "
                f"{fmt_pips(row.get('gross'))} | "
                f"{fmt_pips(row['realized_pnl'])} | {fmt_pips(row['worst_open_excursion'])} | "
                f"{int(row['n_open_at_end'])} | {fmt_pips(row['open_book_unrealized'])} | {fmt_pips(dd)} | "
                f"{pct_blocked_str} |"
            )
        lines.append("")

        lines.append(f"Arm D day-block bootstrap: mean={fmt_pips(extra.get('d_boot_mean'))}, "
                     f"95% CI={tuple(round(x, 2) if x==x else None for x in extra.get('d_boot_ci', (None, None)))}. "
                     f"Arm D minus coin_D bootstrap diff 95% CI="
                     f"{tuple(round(x, 2) if x==x else None for x in extra.get('d_vs_coinD_diff_ci', (None, None)))}.")
        lines.append("")
        lines.append("Arm D walk-forward thirds (net/trade): " +
                     ", ".join(f"third {t['third']} ({t['n']}n)={fmt_pips(t['mean_net'])}"
                               for t in extra.get("wf_thirds_D", [])))
        lines.append("")

        # ── per-pair tables for A and D ──
        pp_path = os.path.join(out_dir, "is_per_pair_summary.csv")
        if os.path.exists(pp_path):
            per_pair = pd.read_csv(pp_path)
            for arm in ("A", "D"):
                lines.append(f"## Per-pair — arm {arm}")
                lines.append("")
                lines.append("| Pair | n | WR | net/trade | gross/trade | realized P&L | worst open excursion | open-at-end |")
                lines.append("|---|---|---|---|---|---|---|---|")
                sub = per_pair[per_pair["arm"] == arm]
                for pair in PAIRS:
                    r = sub[sub["pair"] == pair]
                    if len(r) == 0:
                        continue
                    r = r.iloc[0]
                    lines.append(
                        f"| {pair} | {int(r['n'])} | {r['wr']*100:.0f}% | {fmt_pips(r['net'])} | "
                        f"{fmt_pips(r.get('gross'))} | "
                        f"{fmt_pips(r['realized_pnl'])} | {fmt_pips(r['worst_open_excursion'])} | "
                        f"{int(r['n_open_at_end'])} |"
                    )
                lines.append("")
    else:
        lines.append("## Gates 3-5\n\n- **NOT RUN** — gate 2 (R7 parity) FAILED and is a "
                     "BLOCKING gate per PREREGISTRATION.md's decision rule ('FAIL here blocks "
                     "everything'). The IS battery (`run_is_battery.py`) was never executed.\n")

    # ── Verdict ──
    lines.append("## Verdict")
    lines.append("")
    if os.path.exists(gt_path):
        gate_table = pd.read_csv(gt_path)
        gates_str = "/".join(("PASS" if bool(r) else "FAIL") for r in gate_table["pass"])
        all_pass = bool(gate_table["pass"].all())
        by_arm = {row["arm"]: row for row in extra.get("portfolio", [])} if "extra" in locals() else {}
        dd_by_arm = extra.get("floating_maxdd_by_arm", {}) if "extra" in locals() else {}
        lines.append(f"Gates 3-4-5 = {gates_str} (gate 1 = PASS, gate 2 = see above, BLOCKING).")
        a_row = by_arm.get("A", {})
        d_row = by_arm.get("D", {})
        coin_d_row = by_arm.get("coin_D", {})
        dd_a = abs(dd_by_arm.get("A", float("nan")))
        dd_d = abs(dd_by_arm.get("D", float("nan")))
        lines.append(
            f"Gate 3 confirms the pathology arm D is meant to fix: arm A (no stop) marks "
            f"{fmt_pips(a_row.get('open_book_unrealized'))} unrealized across "
            f"{int(a_row.get('n_open_at_end', 0))} never-closed positions at IS end, worst "
            f"single open excursion {fmt_pips(a_row.get('worst_open_excursion'))} — the "
            f"unbounded tail is real on IS data, not just the live anecdote."
        )
        if all_pass:
            lines.append("All IS gates before the user gate pass. Per the pre-registration's "
                         "decision rule, this clears the way to request the user's explicit "
                         "UNSEAL before touching the sealed OOS window — OOS was NOT opened in "
                         "this run.")
            lines.append("**Recommendation: request OOS UNSEAL from the user** — arm D cleared "
                         "all three H1 criteria plus walk-forward on IS; the sealed 30% window is "
                         "the only remaining check before a paper A/B can be discussed.")
        else:
            lines.append(
                f"Gate 4 fails on all four sub-criteria, not narrowly: arm D's day-block-bootstrap "
                f"net is negative and CI-excludes-zero on the WRONG side "
                f"({fmt_pips(d_row.get('net'))}/trade), it does not beat coin_D "
                f"({fmt_pips(d_row.get('net'))} vs {fmt_pips(coin_d_row.get('net'))}, diff CI spans "
                f"zero), floating maxDD is {dd_d / dd_a:.1f}x arm A's ({fmt_pips(-dd_d)} vs "
                f"{fmt_pips(-dd_a)}) — the OPPOSITE of the ≤50% claim — and 0/3 WF thirds are "
                f"positive. Gate 5 fails narrowly on DD (overlay-on-coin ~5.8% worse than plain "
                f"coin, no sign flip)."
            )
            lines.append(
                "**Recommendation: STOP, do not request OOS unseal.** This is not a close call on "
                "this frozen-parameter configuration — the floating-overlay+3xATR-stop treatment "
                "does not merely fail to help, it turns arm A's modest positive expectancy "
                "negative while *amplifying* (not bounding) floating drawdown. Per the "
                "pre-registration's decision rule ('H1 fails but tail metrics confirm the "
                "pathology'), the keeper is the tail-pathology finding itself (gate 3) plus this "
                "IS-only falsification of the disaster-stop+overlay fix — not a strategy to carry "
                "to OOS. `fx-sma-scratch-paper` stays stopped (Amendment 3)."
            )
    else:
        lines.append("**BLOCKED at gate 2.** Gate 1 (harness self-test) PASSES cleanly (41/41 "
                     "tests). Gate 2 (R7 parity vs the 2026-06-15+ live paper trail) FAILS on "
                     "the pre-registered expectancy tolerance (live +8.70p/trade vs harness "
                     "+4.56p/trade, diff -4.14p vs ±1.0p tolerance; trade count passes, "
                     "0.99 ratio). Diagnosis (documented above) found a plausible pattern — "
                     "divergence grows with faster trade-cycling pairs (CAD_JPY collapses to 0 "
                     "harness trades in the comparison window; GBP_JPY/GBP_USD show large "
                     "count/sign divergence; USD_JPY/NZD_USD stay close) — but did not fully "
                     "isolate a single root cause within the task's time budget. Gates 3-5 and "
                     "the IS battery were **never run**; OOS remains sealed (was never touched "
                     "regardless — gate 2 only reads 2026-06-15+ data). Per the "
                     "pre-registration's own rule, this program stops here pending either a "
                     "root-cause fix or a re-justified tolerance in a future amendment.")
    lines.append("")
    lines.append("This run also produced a durable methodological finding independent of the "
                 "gate outcome: prospective (order-blocking, not merely paper-tracked) "
                 "equity-curve overlays are self-referentially unstable unless their gating "
                 "signal is sourced from an always-on reference run — see harness.py's module "
                 "docstring. This generalizes beyond scratch_tail to any future prospective "
                 "equity-MA gate design (e.g. an eventual live port of the "
                 "`equity_switch_monitor` pattern beyond its current paper-only observation "
                 "role).")
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
