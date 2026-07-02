#!/usr/bin/env python3
"""
Test: SBA swing buffer fix.
Compares three modes on OOS data for all 12 pairs:
  A) FULL  — full-history SBA (how training was run)
  B) LIVE  — 500-bar rolling window (current broken live behavior)
  C) FIXED — 10,000-bar rolling window (proposed fix)

Optimisation: rolling SBA computed only for OOS bars (not full history)
→ O(n_oos × buffer_size) instead of O(n × buffer_size).
"""
import sys, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.swing_indicators import compute_swing_features
from research.experiments.sb_a_ironnet.train_sb_a_cma import (
    simulate_ironnet, PAIR_SPREAD, PAIR_PIP,
)

DATA_DIR   = ROOT / "data" / "range_bar_causal"
GENOME_DIR = ROOT / "research" / "experiments" / "sb_a_ironnet" / "results"
GATES_FILE = GENOME_DIR / "five_gates_ironnet_results.json"

IS_FRAC  = 0.70
MAX_HOLD = 500
ENTER_T  = 0.30
CLOSE_T  = 0.20

gates     = json.loads(GATES_FILE.read_text())
BEST_SEED = {g["pair"]: int(np.argmax(g.get("seed_ppds", [g["oos_ppd"]]))) for g in gates}
ALL_PAIRS = list(PAIR_SPREAD.keys())


def full_sba(close):
    state, _, _, _, _ = compute_swing_features(close, close, close)
    return (state / 2.0).astype(np.float32)


def rolling_sba_oos(close, n_is, buf_size):
    """Compute rolling SBA only for OOS bars — O(n_oos × buf_size)."""
    n     = len(close)
    n_oos = n - n_is
    sba   = np.zeros(n_oos, dtype=np.float32)
    for i in range(n_oos):
        idx   = n_is + i
        start = max(0, idx - buf_size + 1)
        buf   = close[start : idx + 1]
        if len(buf) >= 3:
            state, _, _, _, _ = compute_swing_features(buf, buf, buf)
            sba[i] = float(state[-1]) / 2.0
    return sba


def eval_pair(pair, verbose=True):
    rb_path = DATA_DIR / f"{pair}_range10_causal.parquet"
    if not rb_path.exists():
        return None
    df    = pd.read_parquet(rb_path)
    close = df["mid_close"].values.astype(np.float64)
    mc_d  = df["mc_d"].values.astype(np.float32)
    mc_dd = df["mc_dd"].values.astype(np.float32)
    n     = len(close)
    n_is  = int(n * IS_FRAC)
    n_oos = n - n_is
    pip   = PAIR_PIP[pair]
    spread= PAIR_SPREAD[pair]
    days  = n_oos / (n / (5 * 250))   # approx OOS trading days

    seed  = BEST_SEED.get(pair, 0)
    pkl   = GENOME_DIR / f"sb_a_ironnet_{pair}_s{seed}_best.pkl"
    if not pkl.exists():
        return None
    pkg   = pickle.loads(pkl.read_bytes())
    genes = np.array(pkg["genes"] if isinstance(pkg, dict) else pkg, dtype=np.float64)

    # ── SBA arrays for each mode ──────────────────────────────────────────────
    sba_full = full_sba(close)[-n_oos:]

    if verbose: print(f"  {pair} LIVE ...", end='\r')
    sba_live  = rolling_sba_oos(close, n_is, 500)

    if verbose: print(f"  {pair} FIXED...", end='\r')
    sba_fixed = rolling_sba_oos(close, n_is, 10_000)

    # SBA agreement stats
    agree_live  = float(np.mean(sba_full == sba_live))
    agree_fixed = float(np.mean(sba_full == sba_fixed))

    results = {}
    for mode, sba_oos in [("FULL", sba_full), ("LIVE", sba_live), ("FIXED", sba_fixed)]:
        tp, nt, nl, ns, avg_mae = simulate_ironnet(
            genes,
            np.ascontiguousarray(sba_oos),
            np.ascontiguousarray(mc_d[-n_oos:]),
            np.ascontiguousarray(mc_dd[-n_oos:]),
            np.ascontiguousarray(close[-n_oos:]),
            spread, pip, MAX_HOLD, ENTER_T, CLOSE_T,
        )
        results[mode] = {"ppd": tp / days if days > 0 else 0.0, "trades": nt}

    results["agree_live"]  = agree_live
    results["agree_fixed"] = agree_fixed
    return results


print("SBA fix test — FULL vs LIVE(500) vs FIXED(10K) on OOS")
print(f"\n{'Pair':<12} {'FULL':>8} {'LIVE':>8} {'FIXED':>8}  {'LIVE Δ':>7} {'FIXED Δ':>8}  {'agr-L':>6} {'agr-F':>6}")
print("─" * 74)

agg = {"FULL": 0.0, "LIVE": 0.0, "FIXED": 0.0}
n_pairs = 0

for pair in sorted(ALL_PAIRS):
    res = eval_pair(pair)
    if res is None:
        print(f"{pair:<12}  (no data/genome)"); continue
    full  = res["FULL"]["ppd"]
    live  = res["LIVE"]["ppd"]
    fixed = res["FIXED"]["ppd"]
    lf = "🟢" if live  >= full * 0.8 else "🔴"
    ff = "🟢" if fixed >= full * 0.8 else "🔴"
    print(f"{pair:<12} {full:>8.1f} {live:>8.1f} {fixed:>8.1f}  "
          f"{lf}{live-full:>+5.1f}  {ff}{fixed-full:>+6.1f}  "
          f"{res['agree_live']:>5.1%} {res['agree_fixed']:>5.1%}")
    for m in ["FULL","LIVE","FIXED"]: agg[m] += res[m]["ppd"]
    n_pairs += 1

print("─" * 74)
n = n_pairs
print(f"{'AVG':<12} {agg['FULL']/n:>8.1f} {agg['LIVE']/n:>8.1f} {agg['FIXED']/n:>8.1f}  "
      f"  {(agg['LIVE']-agg['FULL'])/n:>+5.1f}  {(agg['FIXED']-agg['FULL'])/n:>+6.1f}")
print()
print("agr-L = SBA agreement LIVE vs FULL  |  agr-F = SBA agreement FIXED vs FULL")
