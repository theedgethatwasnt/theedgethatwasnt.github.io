#!/usr/bin/env python3
"""Analyze PINN-CMA experiment results.

Loads all *_summary.json files from results_pinn/, produces:
  1. Comparison table: OOS p/d by arm × pair
  2. Overfit diagnostic: IS vs OOS ratio
  3. Evolved hyperparameter summary
  4. Contrarian violation rates
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results_pinn"

ARMS_ORDER = ["baseline", "inputs", "hyper", "inputs_hyper", "inputs_fitness", "full"]
ARM_LABELS = {
    "baseline": "C0",
    "inputs": "T-A",
    "hyper": "T-B",
    "inputs_hyper": "T-AB",
    "inputs_fitness": "T-AC",
    "full": "T-ABC",
}


def load_results():
    results = []
    for p in sorted(RESULTS_DIR.glob("pinn_*_summary.json")):
        with open(p) as f:
            r = json.load(f)
            results.append(r)
    return results


def main():
    results = load_results()
    if not results:
        print("No results found in", RESULTS_DIR)
        sys.exit(1)

    print(f"\nLoaded {len(results)} results from {RESULTS_DIR}\n")

    # Group by (mode, pair)
    by_mode_pair = defaultdict(list)
    for r in results:
        key = (r["mode"], r["pair"])
        by_mode_pair[key].append(r)

    pairs = sorted(set(r["pair"] for r in results))

    # ── Table 1: OOS pips/day ──────────────────────────────────
    print("=" * 80)
    print("TABLE 1: OOS Pips/Day  (median across seeds)")
    print("=" * 80)
    header = f"{'Arm':<8}"
    for p in pairs:
        header += f"  {p:>10}"
    header += f"  {'ALL':>10}"
    print(header)
    print("-" * len(header))

    for mode in ARMS_ORDER:
        label = ARM_LABELS[mode]
        row = f"{label:<8}"
        all_oos = []
        for pair in pairs:
            key = (mode, pair)
            runs = by_mode_pair.get(key, [])
            oos_vals = [r["oos"]["pips_per_day"] for r in runs if r.get("oos")]
            if oos_vals:
                med = np.median(oos_vals)
                all_oos.extend(oos_vals)
                row += f"  {med:>+10.2f}"
            else:
                row += f"  {'—':>10}"
        if all_oos:
            row += f"  {np.median(all_oos):>+10.2f}"
        else:
            row += f"  {'—':>10}"
        print(row)

    # ── Table 2: Hard gate pass rate ───────────────────────────
    print(f"\n{'=' * 80}")
    print("TABLE 2: Hard Gate Pass Rate  (count passing / total runs)")
    print("=" * 80)
    header = f"{'Arm':<8}"
    for p in pairs:
        header += f"  {p:>10}"
    header += f"  {'ALL':>10}"
    print(header)
    print("-" * len(header))

    for mode in ARMS_ORDER:
        label = ARM_LABELS[mode]
        row = f"{label:<8}"
        total_pass = 0
        total_runs = 0
        for pair in pairs:
            key = (mode, pair)
            runs = by_mode_pair.get(key, [])
            n_pass = sum(1 for r in runs if r.get("passed_hard_gates"))
            n_total = len(runs)
            total_pass += n_pass
            total_runs += n_total
            if n_total:
                row += f"  {n_pass}/{n_total:>8}"
            else:
                row += f"  {'—':>10}"
        if total_runs:
            row += f"  {total_pass}/{total_runs:>8}"
        print(row)

    # ── Table 3: Overfit diagnostic ────────────────────────────
    print(f"\n{'=' * 80}")
    print("TABLE 3: OOS/IS Ratio  (median, higher=less overfit, >0.3 target)")
    print("=" * 80)
    header = f"{'Arm':<8}  {'OOS/IS':>10}  {'IS p/d':>10}  {'OOS p/d':>10}"
    print(header)
    print("-" * len(header))

    for mode in ARMS_ORDER:
        label = ARM_LABELS[mode]
        is_vals = []
        oos_vals = []
        for pair in pairs:
            key = (mode, pair)
            runs = by_mode_pair.get(key, [])
            for r in runs:
                if r.get("is_full") and r.get("oos"):
                    is_ppd = r["is_full"]["pips_per_day"]
                    oos_ppd = r["oos"]["pips_per_day"]
                    is_vals.append(is_ppd)
                    oos_vals.append(oos_ppd)
        if is_vals:
            med_is = np.median(is_vals)
            med_oos = np.median(oos_vals)
            ratio = med_oos / med_is if med_is != 0 else float('inf')
            print(f"{label:<8}  {ratio:>+10.2f}  {med_is:>+10.2f}  {med_oos:>+10.2f}")
        else:
            print(f"{label:<8}  {'—':>10}  {'—':>10}  {'—':>10}")

    # ── Table 4: Evolved hyperparameters ───────────────────────
    hyper_modes = ["hyper", "inputs_hyper", "full"]
    hyper_results = [r for r in results if r["mode"] in hyper_modes]
    if hyper_results:
        print(f"\n{'=' * 80}")
        print("TABLE 4: Evolved Hyperparameters")
        print("=" * 80)
        print(f"{'Arm':<8}  {'Pair':<10}  {'Seed':>5}  {'max_hold':>10}  {'threshold':>10}")
        print("-" * 55)
        for r in sorted(hyper_results, key=lambda x: (x["mode"], x["pair"], x["seed"])):
            mh = r.get("evolved_max_hold", "—")
            et = r.get("evolved_threshold")
            et_str = f"{et:.3f}" if et is not None else "—"
            print(f"{ARM_LABELS[r['mode']]:<8}  {r['pair']:<10}  {r['seed']:>5}  "
                  f"{str(mh):>10}  {et_str:>10}")

    # ── Table 5: Contrarian + quality diagnostics ──────────────
    ss_modes = ["inputs", "inputs_hyper", "inputs_fitness", "full"]
    ss_results = [r for r in results if r["mode"] in ss_modes]
    if ss_results:
        print(f"\n{'=' * 80}")
        print("TABLE 5: Contrarian & Quality Diagnostics (OOS)")
        print("=" * 80)
        print(f"{'Arm':<8}  {'Pair':<10}  {'Seed':>5}  {'Contr%':>7}  {'WinR%':>7}  "
              f"{'AvgHold':>8}  {'ShortH%':>8}")
        print("-" * 65)
        for r in sorted(ss_results, key=lambda x: (x["mode"], x["pair"], x["seed"])):
            oos = r.get("oos", {})
            cr = oos.get("contrarian_rate", 0)
            wr = oos.get("win_rate", 0)
            ah = oos.get("avg_hold", 0)
            sh = oos.get("short_hold_pct", 0)
            print(f"{ARM_LABELS[r['mode']]:<8}  {r['pair']:<10}  {r['seed']:>5}  "
                  f"{cr*100:>6.1f}%  {wr*100:>6.1f}%  {ah:>8.1f}  {sh*100:>7.1f}%")

    # ── Summary verdict ────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("VERDICT")
    print("=" * 80)

    # Compare each treatment to C0
    c0_oos = []
    for pair in pairs:
        runs = by_mode_pair.get(("baseline", pair), [])
        c0_oos.extend([r["oos"]["pips_per_day"] for r in runs if r.get("oos")])
    c0_med = np.median(c0_oos) if c0_oos else 0.0

    for mode in ARMS_ORDER[1:]:  # skip baseline
        label = ARM_LABELS[mode]
        t_oos = []
        for pair in pairs:
            runs = by_mode_pair.get((mode, pair), [])
            t_oos.extend([r["oos"]["pips_per_day"] for r in runs if r.get("oos")])
        if t_oos:
            t_med = np.median(t_oos)
            delta = t_med - c0_med
            verdict = "BETTER" if delta > 0.05 else ("WORSE" if delta < -0.05 else "NEUTRAL")
            print(f"  {label:>5} vs C0: median OOS {t_med:+.2f} vs {c0_med:+.2f} "
                  f"(delta={delta:+.2f}) → {verdict}")

    print(f"\nC0 baseline median OOS: {c0_med:+.2f} p/d")
    print("Success threshold: > C0 + 0.05 p/d")


if __name__ == "__main__":
    main()
