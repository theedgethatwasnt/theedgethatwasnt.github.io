#!/usr/bin/env python3
"""Pick best genome across all seed pkl files for a given variant tag."""
import argparse
import glob
import pickle
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="glob pattern e.g. 'results/cma_mh200_s*_EUR_JPY.pkl'")
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files matched: {args.pattern}")
        return

    best_file = None
    best_fit  = -1e9
    print(f"{'File':<50} {'IS fitness':>12}")
    print("─" * 64)
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        fit = d.get("fitness", -1e9)
        marker = ""
        if fit > best_fit:
            best_fit  = fit
            best_file = f
            marker = " ← best"
        print(f"  {Path(f).name:<48} {fit:>12.4f}{marker}")

    print(f"\nBest: {best_file}  fitness={best_fit:.4f}")

    # Save winner as canonical file (strip seed from name)
    with open(best_file, "rb") as fh:
        best = pickle.load(fh)
    out = Path(best_file).parent / (Path(best_file).stem.rsplit("_s", 1)[0] + "_best.pkl")
    with open(out, "wb") as fh:
        pickle.dump(best, fh)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
