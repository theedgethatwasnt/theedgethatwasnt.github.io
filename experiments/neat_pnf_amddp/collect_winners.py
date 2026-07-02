#!/usr/bin/env python3
"""
collect_winners.py — scan all islands' all_time_best bundles, rank by validation
amddp5/day, and emit the SINGLE best for the sealed OOS/test evaluation + MC.
================================================================================
Design: research/experiments/neat_pnf_amddp/PLAN.md (3-way split, R8 OOS sealed).

The campaign runs many islands (4 seeds x 4 exponents = 16 + their surrogate twins).
Each island writes <out_root>/<tag>/bundles/all_time_best.pkl with a full trading-stats
bundle (IS + VAL computed, TEST untouched). This helper:

  1. Collects every all_time_best.pkl under a campaign root.
  2. Ranks the REAL (non-surrogate) islands by validation amddp/day.
  3. Compares the top real island to the BEST SURROGATE island (the equal-compute
     null): a real winner must beat the noise-evolved best.
  4. Emits the single selected genome's bundle for the ONE sealed test evaluation.
     (Touching OOS/test is done by a separate sealed-eval step — R8: exactly once.)

Usage:
  python3 collect_winners.py [--root campaign_runs/GBP_JPY] [--emit selected.pkl]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Bundles embed a CappedGenome (module research.experiments.neat_pnf_amddp.phase1_harness)
# and use custom activations — importing the harness registers them and makes the
# genome class importable so pickle.load succeeds.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from research.experiments.neat_pnf_amddp import phase1_harness  # noqa: E402,F401


def load_all_time_bests(root):
    """Find every <root>/**/bundles/all_time_best.pkl and load it."""
    found = []
    for path in glob.glob(os.path.join(root, "**", "bundles", "all_time_best.pkl"),
                          recursive=True):
        try:
            with open(path, "rb") as f:
                b = pickle.load(f)
            b["_path"] = path
            found.append(b)
        except Exception as e:
            print(f"  [warn] could not load {path}: {e}")
    return found


def _val_amddp_per_day(b):
    return b.get("stats", {}).get("val", {}).get("aggregates", {}).get("amddp_per_day", float("-inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str,
                    default=os.path.join(HERE, "campaign_runs", "GBP_JPY"))
    ap.add_argument("--emit", type=str, default=os.path.join(HERE, "selected_winner.pkl"),
                    help="path to write the single selected bundle for sealed OOS eval")
    args = ap.parse_args()

    bundles = load_all_time_bests(args.root)
    if not bundles:
        print(f"No all_time_best.pkl found under {args.root}")
        return

    real = [b for b in bundles if not b.get("is_surrogate", False)]
    surrogate = [b for b in bundles if b.get("is_surrogate", False)]

    real.sort(key=_val_amddp_per_day, reverse=True)
    surrogate.sort(key=_val_amddp_per_day, reverse=True)

    print("=" * 92)
    print(f"COLLECT WINNERS — {len(real)} real islands, {len(surrogate)} surrogate islands")
    print("=" * 92)
    print(f"{'rank':>4} {'island':>6} {'seed':>4} {'exp':>4} "
          f"{'val_amddp/d':>12} {'val_sharpe':>10} {'val_sqn':>8} {'mc_p':>7} {'wf+':>4} {'gen':>5}")
    for i, b in enumerate(real):
        va = b["stats"]["val"]["aggregates"]
        g = b["gates"]
        print(f"{i:>4} {b['island']:>6} {b['seed']:>4} {b['exponent']:>4} "
              f"{va['amddp_per_day']:>12.3f} {va['sharpe']:>10.3f} {va['sqn']:>8.3f} "
              f"{g['mc_pvalue_val']:>7.3f} {str(g['wf_all_positive']):>4} {b['generation']:>5}")

    best_sur_metric = _val_amddp_per_day(surrogate[0]) if surrogate else float("-inf")
    print(f"\n  best surrogate (null) val_amddp/d = "
          f"{best_sur_metric if surrogate else 'n/a'}")

    if not real:
        print("\n  No real islands to select from.")
        return

    winner = real[0]
    win_metric = _val_amddp_per_day(winner)
    beats_null = win_metric > best_sur_metric
    g = winner["gates"]

    # record the surrogate-null comparison into the winner's gates slot
    winner["gates"]["surrogate_null_amddp_per_day"] = (
        best_sur_metric if surrogate else None)
    winner["gates"]["beats_surrogate_null"] = bool(beats_null)

    print("\n" + "-" * 92)
    print("SELECTED WINNER (best validation amddp/day):")
    print(f"  island={winner['island']} seed={winner['seed']} exp={winner['exponent']} "
          f"gen={winner['generation']}")
    print(f"  val_amddp/d = {win_metric:.3f}  |  beats surrogate-null? "
          f"{'YES' if beats_null else 'NO'}")
    print(f"  gates: wf_all_positive={g['wf_all_positive']}  "
          f"mc_p={g['mc_pvalue_val']:.4f} (pass={g['mc_pass']})  "
          f"validated={g['validated']}")
    print(f"  source: {winner['_path']}")

    out = dict(winner)
    out.pop("_path", None)
    with open(args.emit, "wb") as f:
        pickle.dump(out, f)
    print(f"\n  Emitted selected bundle -> {args.emit}")
    print("  NEXT: run the ONE sealed OOS/test evaluation on this genome (R8: touch test once),"
          " then MC on the test trades. Do NOT re-select after seeing test.")


if __name__ == "__main__":
    main()
