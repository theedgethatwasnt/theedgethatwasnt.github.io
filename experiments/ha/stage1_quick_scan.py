#!/usr/bin/env python3
"""
Stage 1: Quick NEAT scan with HA direction input.
=================================================
50 generations, 2 islands, 1 pair (EUR_JPY), 4 variants.
Run locally, ~15 min each, 2 at a time → ~30 min total.

Variants:
  S1-1: Long  M5, 2-out (ENTER/CLOSE)
  S1-2: Short M5, 2-out (ENTER/CLOSE)
  S1-3: Both  M5, 3-out (BUY/SELL/CLOSE)
  S1-4: Both  H1, 3-out (BUY/SELL/CLOSE)

Pass criteria: at least 1 variant shows positive OOS P/L with > 20 trades.
"""

import sys
import os
import gc
import time
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import neat
from lib.fast_eval import (
    extract_network, compute_ha_dir,
    evaluate_ha_2out_jit, evaluate_ha_3out_jit,
)

DATA_DIR = PROJECT_ROOT / "data" / "scalper_parquet"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "stage1"
EXPERIMENT_DIR = Path(__file__).resolve().parent

# EUR_JPY config
PAIR = "EUR_JPY"
PIP = 0.01
SPREAD_PIPS = 2.3


def load_pair_data(pair: str, tf: str):
    """Load S5 data and resample to target timeframe. Returns ha_dir, mid_close arrays."""
    parquet = DATA_DIR / f"{pair.replace('_', '')}_S5_BA.parquet"
    print(f"  Loading {parquet.name}...", end=" ", flush=True)
    df = pd.read_parquet(parquet, engine="pyarrow")
    print(f"{len(df):,} S5 bars")

    if tf == "M5":
        ts = pd.to_datetime(df["timestamp"])
        ohlc = pd.DataFrame({
            "o": df["bid_o"].values, "h": df["bid_h"].values,
            "l": df["bid_l"].values, "c": df["bid_c"].values,
            "mid_c": (df["bid_c"].values + df["ask_c"].values) / 2.0,
        }, index=ts)
        r = ohlc.resample("5min").agg({"o": "first", "h": "max", "l": "min", "c": "last", "mid_c": "last"}).dropna()
    elif tf == "H1":
        ts = pd.to_datetime(df["timestamp"])
        ohlc = pd.DataFrame({
            "o": df["bid_o"].values, "h": df["bid_h"].values,
            "l": df["bid_l"].values, "c": df["bid_c"].values,
            "mid_c": (df["bid_c"].values + df["ask_c"].values) / 2.0,
        }, index=ts)
        r = ohlc.resample("1h").agg({"o": "first", "h": "max", "l": "min", "c": "last", "mid_c": "last"}).dropna()
    else:
        raise ValueError(f"Unknown TF: {tf}")

    o = r["o"].values.astype(np.float64)
    h = r["h"].values.astype(np.float64)
    l = r["l"].values.astype(np.float64)
    c = r["c"].values.astype(np.float64)
    mid_c = r["mid_c"].values.astype(np.float64)
    n = len(o)

    ha_dir = compute_ha_dir(o, h, l, c, n)
    print(f"  {tf}: {n:,} bars, HA bullish: {(ha_dir > 0).sum():,}, bearish: {(ha_dir < 0).sum():,}")

    del df
    gc.collect()
    return ha_dir, mid_c


# ── NEAT Training ──────────────────────────────────────────────────────────

class HAEvaluator:
    """NEAT fitness evaluator for HA experiments."""

    def __init__(self, ha_dir, mid_close, pip, spread_pips,
                 is_long=True, n_outputs=2,
                 is_split=0.7, max_hold=200):
        n = len(ha_dir)
        split = int(n * is_split)
        self.ha_dir_is = ha_dir[:split]
        self.mid_close_is = mid_close[:split]
        self.ha_dir_oos = ha_dir[split:]
        self.mid_close_oos = mid_close[split:]
        self.pip = pip
        self.spread = spread_pips
        self.is_long = is_long
        self.n_outputs = n_outputs
        self.max_hold = max_hold
        self.max_bars = split  # Use all IS data

    def evaluate(self, genomes, config):
        """NEAT fitness function — evaluate on IS data."""
        for genome_id, genome in genomes:
            genome.fitness = self._eval_genome(genome, config,
                                                self.ha_dir_is, self.mid_close_is)

    def _eval_genome(self, genome, config, ha_dir, mid_close):
        """Evaluate a single genome. Returns fitness (Sharpe-like)."""
        try:
            (n_inputs, n_out, n_eval, total_values,
             node_bias, node_response, node_act,
             conn_from, conn_to, conn_weight,
             output_indices) = extract_network(genome, config)
        except Exception:
            return -10.0

        if self.n_outputs == 2:
            result = evaluate_ha_2out_jit(
                ha_dir, mid_close, self.pip, self.spread,
                self.max_bars, self.max_hold, 0.01,
                n_inputs, n_eval, total_values,
                node_bias, node_response, node_act,
                conn_from, conn_to, conn_weight,
                output_indices, 0, self.is_long,
            )
        else:
            result = evaluate_ha_3out_jit(
                ha_dir, mid_close, self.pip, self.spread,
                self.max_bars, self.max_hold, 0.01,
                n_inputs, n_eval, total_values,
                node_bias, node_response, node_act,
                conn_from, conn_to, conn_weight,
                output_indices, 0,
            )

        n_trades, total_pnl, sharpe, win_rate = result[0], result[1], result[2], result[3]

        # Fitness: Sharpe with trade count bonus/penalty
        if n_trades < 5:
            return -10.0 + n_trades * 0.1  # Encourage some trading
        return float(sharpe)

    def eval_oos(self, genome, config):
        """Evaluate best genome on OOS data."""
        try:
            (n_inputs, n_out, n_eval, total_values,
             node_bias, node_response, node_act,
             conn_from, conn_to, conn_weight,
             output_indices) = extract_network(genome, config)
        except Exception:
            return {}

        if self.n_outputs == 2:
            result = evaluate_ha_2out_jit(
                self.ha_dir_oos, self.mid_close_oos, self.pip, self.spread,
                len(self.mid_close_oos), self.max_hold, 0.01,
                n_inputs, n_eval, total_values,
                node_bias, node_response, node_act,
                conn_from, conn_to, conn_weight,
                output_indices, 0, self.is_long,
            )
        else:
            result = evaluate_ha_3out_jit(
                self.ha_dir_oos, self.mid_close_oos, self.pip, self.spread,
                len(self.mid_close_oos), self.max_hold, 0.01,
                n_inputs, n_eval, total_values,
                node_bias, node_response, node_act,
                conn_from, conn_to, conn_weight,
                output_indices, 0,
            )

        (n_trades, total_pnl, sharpe, win_rate, mean_pnl,
         avg_mfe, avg_mae, avg_cap, max_dd, avg_bars) = result

        return {
            "n_trades": int(n_trades), "total_pnl": round(float(total_pnl), 1),
            "sharpe": round(float(sharpe), 4), "win_rate": round(float(win_rate), 1),
            "mean_pnl": round(float(mean_pnl), 2), "avg_mfe": round(float(avg_mfe), 1),
            "avg_mae": round(float(avg_mae), 1), "max_dd": round(float(max_dd), 1),
            "avg_bars": round(float(avg_bars), 1),
        }


def run_variant(variant_id: str, tf: str, direction: str, n_outputs: int,
                ha_dir: np.ndarray, mid_close: np.ndarray,
                generations: int = 50, seed: int = 42):
    """Run a single NEAT training variant."""
    print(f"\n{'='*60}")
    print(f"  {variant_id}: {direction} {tf} {n_outputs}-out, {generations} gens")
    print(f"{'='*60}")

    is_long = direction in ("long", "both")  # For 2-out: True=long, False=short
    if n_outputs == 3:
        is_long = True  # Not used for 3-out

    config_file = EXPERIMENT_DIR / f"neat_config_{n_outputs}out.ini"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_file))

    # Max hold: M5=200 bars (16h), H1=48 bars (2 days)
    max_hold = 200 if tf == "M5" else 48

    evaluator = HAEvaluator(ha_dir, mid_close, PIP, SPREAD_PIPS,
                            is_long=(direction == "long"), n_outputs=n_outputs,
                            max_hold=max_hold)

    # Run NEAT with 2 islands (subpopulations via species)
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(False))
    stats_reporter = neat.StatisticsReporter()
    pop.add_reporter(stats_reporter)

    t0 = time.time()
    winner = pop.run(evaluator.evaluate, generations)
    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed:.1f}s")

    # Evaluate on OOS
    oos = evaluator.eval_oos(winner, config)
    is_fitness = winner.fitness

    print(f"  IS fitness (Sharpe): {is_fitness:.4f}")
    print(f"  OOS: {oos}")

    # Save results
    result = {
        "variant_id": variant_id,
        "tf": tf,
        "direction": direction,
        "n_outputs": n_outputs,
        "pair": PAIR,
        "generations": generations,
        "seed": seed,
        "is_fitness": round(float(is_fitness), 4),
        "oos": oos,
        "elapsed_s": round(elapsed, 1),
    }

    # Save genome
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    genome_path = RESULTS_DIR / f"{variant_id}_best.pkl"
    with open(genome_path, "wb") as f:
        pickle.dump({"genome": winner, "config": config}, f)
    result["genome_file"] = str(genome_path)

    result_path = RESULTS_DIR / f"{variant_id}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Stage 1: HA NEAT Quick Scan")
    parser.add_argument("--variant", type=str, default="all",
                        help="Which variant to run: S1-1, S1-2, S1-3, S1-4, or 'all'")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Define variants (per PLAN.md, but M5+H1 only, S5 dropped after Stage 0)
    VARIANTS = {
        "S1-1": {"tf": "M5", "direction": "long",  "n_outputs": 2},
        "S1-2": {"tf": "M5", "direction": "short", "n_outputs": 2},
        "S1-3": {"tf": "M5", "direction": "both",  "n_outputs": 3},
        "S1-4": {"tf": "H1", "direction": "both",  "n_outputs": 3},
    }

    # Load data once per timeframe
    print(f"Loading {PAIR} data...")
    data_cache = {}
    tfs_needed = set()
    if args.variant == "all":
        for v in VARIANTS.values():
            tfs_needed.add(v["tf"])
    else:
        tfs_needed.add(VARIANTS[args.variant]["tf"])

    for tf in tfs_needed:
        data_cache[tf] = load_pair_data(PAIR, tf)

    # Run variants
    results = []
    if args.variant == "all":
        for vid, cfg in VARIANTS.items():
            ha_dir, mid_c = data_cache[cfg["tf"]]
            r = run_variant(vid, cfg["tf"], cfg["direction"], cfg["n_outputs"],
                            ha_dir, mid_c, args.generations, args.seed)
            results.append(r)
    else:
        vid = args.variant
        cfg = VARIANTS[vid]
        ha_dir, mid_c = data_cache[cfg["tf"]]
        r = run_variant(vid, cfg["tf"], cfg["direction"], cfg["n_outputs"],
                        ha_dir, mid_c, args.generations, args.seed)
        results.append(r)

    # Summary
    print("\n" + "=" * 80)
    print("STAGE 1 RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'ID':<6} {'TF':<4} {'Dir':<6} {'Out':<4} {'IS_Sharpe':>10} {'OOS_Trades':>10} "
          f"{'OOS_PnL':>10} {'OOS_Sharpe':>10} {'OOS_WR':>8}")
    print("-" * 80)

    any_pass = False
    for r in results:
        oos = r["oos"]
        nt = oos.get("n_trades", 0)
        pnl = oos.get("total_pnl", 0)
        sh = oos.get("sharpe", 0)
        wr = oos.get("win_rate", 0)
        passed = nt > 20 and pnl > 0
        marker = "PASS" if passed else "FAIL"
        if passed:
            any_pass = True

        print(f"{r['variant_id']:<6} {r['tf']:<4} {r['direction']:<6} {r['n_outputs']:<4} "
              f"{r['is_fitness']:>10.4f} {nt:>10} {pnl:>10.1f} {sh:>10.4f} {wr:>7.1f}%  {marker}")

    print()
    if any_pass:
        print("🟢 STAGE 1 PASSED — at least one variant shows positive OOS. Proceed to Stage 2.")
    else:
        print("🔴 STAGE 1 FAILED — no variant shows positive OOS with > 20 trades. STOP.")

    # Save combined results
    combined_path = RESULTS_DIR / "stage1_summary.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {combined_path}")


if __name__ == "__main__":
    main()
