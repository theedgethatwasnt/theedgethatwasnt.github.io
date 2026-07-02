#!/usr/bin/env python3
"""
Stage 2: Full HA NEAT training on Hetzner.
==========================================
Train winning variants from Stage 1 with full rigor:
- 4 training pairs: EUR_JPY, GBP_USD, USD_JPY, GBP_JPY
- 200 generations, 4 islands (pop_size=150)
- 70/30 IS/OOS split
- Evaluate OOS on all 12 pairs

Run on Hetzner cx53 (16 vCPU, 32GB RAM).
Each variant takes ~2-3 hours per server.

Usage:
  python3 stage2_training.py --variant S2-long --seed 42
  python3 stage2_training.py --variant S2-both --seed 137
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

# Add project root to path — works both locally (parents[3]) and on Hetzner (cwd)
SCRIPT_DIR = Path(__file__).resolve().parent
_local_root = SCRIPT_DIR.parents[2] if len(SCRIPT_DIR.parts) > 4 else SCRIPT_DIR
sys.path.insert(0, str(_local_root))
sys.path.insert(0, str(SCRIPT_DIR))  # For Hetzner flat layout

import neat
from lib.fast_eval import (
    extract_network, compute_ha_dir,
    evaluate_ha_2out_jit, evaluate_ha_3out_jit,
)

DATA_DIR = Path(os.environ.get("NEAT_DATA_DIR", str(_local_root / "data" / "scalper_parquet")))
RESULTS_DIR = SCRIPT_DIR / "results" / "stage2"
EXPERIMENT_DIR = SCRIPT_DIR

# Training pairs (most liquid / best historical results)
TRAIN_PAIRS = ["EUR_JPY", "GBP_USD", "USD_JPY", "GBP_JPY"]

# All 12 pairs for OOS evaluation
ALL_PAIRS = ["EUR_JPY", "USD_JPY", "GBP_JPY", "GBP_USD", "EUR_USD", "AUD_USD",
             "AUD_JPY", "CAD_JPY", "NZD_JPY", "CHF_JPY", "NZD_USD", "EUR_GBP"]

PAIR_PIP = {
    "EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
    "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
}

PAIR_SPREAD = {
    "EUR_JPY": 2.3, "USD_JPY": 1.7, "GBP_JPY": 3.3, "AUD_JPY": 2.1,
    "CAD_JPY": 2.3, "CHF_JPY": 3.5, "NZD_JPY": 2.7,
    "EUR_USD": 1.6, "GBP_USD": 1.9, "AUD_USD": 1.3,
    "NZD_USD": 1.5, "EUR_GBP": 1.4,
}

IS_SPLIT = 0.7
MAX_HOLD_M5 = 200  # ~16 hours


def load_pair_m5(pair: str):
    """Load S5 data, resample to M5, compute HA direction."""
    parquet = DATA_DIR / f"{pair.replace('_', '')}_S5_BA.parquet"
    print(f"  Loading {pair}...", end=" ", flush=True)
    df = pd.read_parquet(parquet, engine="pyarrow")
    print(f"{len(df):,} S5 bars → ", end="", flush=True)

    ts = pd.to_datetime(df["timestamp"])
    ohlc = pd.DataFrame({
        "o": df["bid_o"].values, "h": df["bid_h"].values,
        "l": df["bid_l"].values, "c": df["bid_c"].values,
        "mid_c": (df["bid_c"].values + df["ask_c"].values) / 2.0,
    }, index=ts)
    r = ohlc.resample("5min").agg({"o": "first", "h": "max", "l": "min", "c": "last", "mid_c": "last"}).dropna()

    o = r["o"].values.astype(np.float64)
    h = r["h"].values.astype(np.float64)
    l = r["l"].values.astype(np.float64)
    c = r["c"].values.astype(np.float64)
    mid_c = r["mid_c"].values.astype(np.float64)
    n = len(o)

    ha_dir = compute_ha_dir(o, h, l, c, n)
    print(f"{n:,} M5 bars")

    del df
    gc.collect()
    return ha_dir, mid_c


class MultiPairEvaluator:
    """NEAT fitness evaluator across multiple pairs."""

    def __init__(self, pair_data: dict, is_long: bool, n_outputs: int):
        """pair_data: {pair: (ha_dir_is, mid_close_is, ha_dir_oos, mid_close_oos, pip, spread)}"""
        self.pair_data = pair_data
        self.is_long = is_long
        self.n_outputs = n_outputs

    def evaluate(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return -10.0

        total_sharpe = 0.0
        total_trades = 0
        total_bars = 0
        n_pairs = 0

        for pair, (ha_is, mc_is, ha_oos, mc_oos, pip, spread) in self.pair_data.items():
            if self.n_outputs == 2:
                result = evaluate_ha_2out_jit(
                    ha_is, mc_is, pip, spread, len(mc_is), MAX_HOLD_M5, 0.01,
                    net[0], net[2], net[3], net[4], net[5], net[6],
                    net[7], net[8], net[9], net[10], 0, self.is_long,
                )
            else:
                result = evaluate_ha_3out_jit(
                    ha_is, mc_is, pip, spread, len(mc_is), MAX_HOLD_M5, 0.01,
                    net[0], net[2], net[3], net[4], net[5], net[6],
                    net[7], net[8], net[9], net[10], 0,
                )

            n_trades, total_pnl, sharpe = result[0], result[1], result[2]
            if n_trades >= 5:
                total_sharpe += sharpe
                n_pairs += 1
            total_trades += n_trades
            total_bars += len(mc_is)

        if n_pairs == 0 or total_trades < 20:
            return -10.0

        avg_sharpe = total_sharpe / n_pairs

        # Trade frequency bonus: reward more trades once Sharpe plateaus
        # M5 bars: 288/day. Compute trades/day across all pairs.
        data_days = total_bars / 288.0 if total_bars > 0 else 1.0
        trades_per_day = total_trades / data_days
        trade_bonus = min(0.1, trades_per_day * 0.01)  # cap at +0.1

        return avg_sharpe + trade_bonus

    def eval_oos_all_pairs(self, genome, config, all_pair_data: dict) -> dict:
        """Evaluate on OOS data across all 12 pairs."""
        try:
            net = extract_network(genome, config)
        except Exception:
            return {"error": "extract failed"}

        results = {}
        total_pnl = 0.0
        total_trades = 0

        for pair, (ha_oos, mc_oos, pip, spread) in all_pair_data.items():
            if self.n_outputs == 2:
                result = evaluate_ha_2out_jit(
                    ha_oos, mc_oos, pip, spread, len(mc_oos), MAX_HOLD_M5, 0.01,
                    net[0], net[2], net[3], net[4], net[5], net[6],
                    net[7], net[8], net[9], net[10], 0, self.is_long,
                )
            else:
                result = evaluate_ha_3out_jit(
                    ha_oos, mc_oos, pip, spread, len(mc_oos), MAX_HOLD_M5, 0.01,
                    net[0], net[2], net[3], net[4], net[5], net[6],
                    net[7], net[8], net[9], net[10], 0,
                )

            nt, pnl, sh, wr, mpnl = result[0], result[1], result[2], result[3], result[4]
            mfe, mae, cap, mdd, ab = result[5], result[6], result[7], result[8], result[9]
            results[pair] = {
                "n_trades": int(nt), "total_pnl": round(float(pnl), 1),
                "sharpe": round(float(sh), 4), "win_rate": round(float(wr), 1),
                "mean_pnl": round(float(mpnl), 2), "avg_mfe": round(float(mfe), 1),
                "avg_mae": round(float(mae), 1), "max_dd": round(float(mdd), 1),
            }
            total_pnl += pnl
            total_trades += nt

        results["_total"] = {"n_trades": total_trades, "total_pnl": round(float(total_pnl), 1)}
        return results


def main():
    parser = argparse.ArgumentParser(description="Stage 2: HA Full Training")
    parser.add_argument("--variant", required=True, choices=["S2-long", "S2-both"],
                        help="S2-long = 2-out long-only M5, S2-both = 3-out bidirectional M5")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop-size", type=int, default=150)
    args = parser.parse_args()

    np.random.seed(args.seed)
    is_long = args.variant == "S2-long"
    n_outputs = 2 if is_long else 3
    config_file = EXPERIMENT_DIR / f"neat_config_{n_outputs}out.ini"

    print(f"{'='*70}")
    print(f"  Stage 2: {args.variant} | {args.generations} gens | seed {args.seed}")
    print(f"  Config: {config_file.name} | pop_size={args.pop_size}")
    print(f"  Training pairs: {TRAIN_PAIRS}")
    print(f"{'='*70}")

    # Load all 12 pairs
    print("\nLoading data...")
    all_ha = {}
    all_mc = {}
    for pair in ALL_PAIRS:
        ha, mc = load_pair_m5(pair)
        all_ha[pair] = ha
        all_mc[pair] = mc

    # Split IS/OOS
    train_data = {}  # For fitness function (IS only, train pairs only)
    oos_data = {}    # For final evaluation (OOS, all 12 pairs)

    for pair in ALL_PAIRS:
        n = len(all_ha[pair])
        split = int(n * IS_SPLIT)
        pip = PAIR_PIP[pair]
        spread = PAIR_SPREAD[pair]

        if pair in TRAIN_PAIRS:
            train_data[pair] = (
                all_ha[pair][:split], all_mc[pair][:split],
                all_ha[pair][split:], all_mc[pair][split:],
                pip, spread,
            )
        oos_data[pair] = (all_ha[pair][split:], all_mc[pair][split:], pip, spread)

    del all_ha, all_mc
    gc.collect()

    # Update NEAT config with pop_size
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_file))
    config.pop_size = args.pop_size

    evaluator = MultiPairEvaluator(train_data, is_long, n_outputs)

    # Run NEAT
    print(f"\nStarting NEAT evolution...")
    t0 = time.time()
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Checkpoint every 25 gens
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_prefix = str(RESULTS_DIR / f"{args.variant}_s{args.seed}_ckpt_")
    pop.add_reporter(neat.Checkpointer(25, filename_prefix=ckpt_prefix))

    winner = pop.run(evaluator.evaluate, args.generations)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"IS fitness (avg Sharpe): {winner.fitness:.4f}")

    # OOS evaluation on all 12 pairs
    print(f"\nOOS evaluation on all 12 pairs...")
    oos_results = evaluator.eval_oos_all_pairs(winner, config, oos_data)

    print(f"\n{'='*70}")
    print(f"  OOS RESULTS — {args.variant} seed {args.seed}")
    print(f"{'='*70}")
    print(f"{'Pair':<10} {'Trades':>7} {'PnL':>10} {'Sharpe':>8} {'WR':>6} {'MFE':>6} {'MAE':>6} {'MxDD':>8}")
    print("-" * 70)
    for pair in ALL_PAIRS:
        r = oos_results.get(pair, {})
        if not r:
            continue
        print(f"{pair:<10} {r['n_trades']:>7} {r['total_pnl']:>+10.1f} "
              f"{r['sharpe']:>8.4f} {r['win_rate']:>5.1f}% {r['avg_mfe']:>6.1f} "
              f"{r['avg_mae']:>6.1f} {r['max_dd']:>8.1f}")
    tot = oos_results.get("_total", {})
    print(f"\n  TOTAL: {tot.get('n_trades', 0)} trades, {tot.get('total_pnl', 0):+.1f} pips")

    # Save genome + results
    genome_path = RESULTS_DIR / f"{args.variant}_s{args.seed}_best.pkl"
    with open(genome_path, "wb") as f:
        pickle.dump({"genome": winner, "config": config}, f)

    result = {
        "variant": args.variant,
        "seed": args.seed,
        "generations": args.generations,
        "pop_size": args.pop_size,
        "is_fitness": round(float(winner.fitness), 4),
        "oos": oos_results,
        "elapsed_s": round(elapsed, 1),
        "genome_file": str(genome_path),
        "train_pairs": TRAIN_PAIRS,
    }

    result_path = RESULTS_DIR / f"{args.variant}_s{args.seed}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {genome_path.name}, {result_path.name}")


if __name__ == "__main__":
    main()
