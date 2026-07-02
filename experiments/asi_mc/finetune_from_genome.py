#!/usr/bin/env python3
"""
Fine-tune the original ASI-MC genome on 4 training pairs.
==========================================================
Loads the existing trained genome into a NEAT population,
then continues evolution on multi-pair data.

This tests whether the genome can IMPROVE with more pair exposure,
vs the from-scratch approach that may find a different local optimum.
"""

import sys
import os
import gc
import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2] if len(SCRIPT_DIR.parts) > 4 else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import neat
from lib.fast_eval import extract_network
from asi_indicator import compute_asi_mc
from train_pretrain_continue import (
    evaluate_3out_jit, AsiMcEvaluator, load_pair_m5_ohlc, tg_send,
    PAIR_PIP, PAIR_SPREAD, ALL_PAIRS, DATA_DIR, RESULTS_DIR,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", default=str(RESULTS_DIR / "asi_mc_s42_best_ORIGINAL.pkl"))
    parser.add_argument("--pairs", default="EUR_JPY,USD_JPY,GBP_JPY,GBP_USD")
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--pop-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hold", type=int, default=200)
    args = parser.parse_args()

    np.random.seed(args.seed)
    train_pairs = [p.strip() for p in args.pairs.split(",")]

    print(f"{'='*60}")
    print(f"  Fine-tune from existing genome")
    print(f"  Genome: {args.genome}")
    print(f"  Pairs: {train_pairs}")
    print(f"  Gens: {args.gens}")
    print(f"{'='*60}")

    tg_send(f"🔧 Fine-tune starting\n"
            f"Pairs: {train_pairs}\n"
            f"Gens: {args.gens}")

    # Load existing genome
    with open(args.genome, "rb") as f:
        d = pickle.load(f)
    genome = d["genome"]
    config = d["config"]
    config.pop_size = args.pop_size

    print(f"Loaded genome: size={genome.size()}, fitness={genome.fitness:.4f}")

    # Create population seeded with the existing genome
    # NEAT doesn't natively support this, so we create a fresh population
    # and inject the genome into it
    pop = neat.Population(config)

    # Replace worst genome in population with our trained one
    for sid, species in pop.species.species.items():
        members = species.members
        gids = list(members.keys())
        for i, gid in enumerate(gids):
            if i == 0:
                # Inject the original genome
                genome.key = gid
                genome.fitness = None  # Will be evaluated in gen 0
                members[gid] = genome
            else:
                # Mutated copy of original
                child = config.genome_type(gid)
                child.configure_crossover(genome, genome, config.genome_config)
                child.mutate(config.genome_config)
                child.fitness = None
                members[gid] = child
        # Also inject into population dict
        pop.population = {g.key: g for g in members.values()}
        break

    pop.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Load training data
    print("\nLoading market data...")
    real_data = {}
    for pair in train_pairs:
        o, h, l, c, mid = load_pair_m5_ohlc(pair)
        n = len(o)
        print(f"  Computing ASI-MC for {pair}...", end=" ", flush=True)
        mc_d, mc_dd = compute_asi_mc(o, h, l, c, n)
        print("done")
        split = int(n * 0.7)
        real_data[pair] = (
            mc_d[:split], mc_dd[:split], mid[:split],
            mc_d[split:], mc_dd[split:], mid[split:],
        )
        del o, h, l, c
        gc.collect()

    evaluator = AsiMcEvaluator(real_data, max_hold=args.max_hold)

    # Fine-tune
    t0 = time.time()
    winner = pop.run(evaluator.evaluate, args.gens)
    elapsed = time.time() - t0

    print(f"\nFine-tune done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Fitness: {winner.fitness:.4f}")
    print(f"  Size: {winner.size()}")

    # OOS on training pairs
    oos = evaluator.eval_oos(winner, config)
    total_pnl = sum(r["total_pnl"] for r in oos.values())
    total_trades = sum(r["n_trades"] for r in oos.values())
    total_long = sum(r.get("n_long", 0) for r in oos.values())
    total_short = sum(r.get("n_short", 0) for r in oos.values())

    print(f"\nOOS (training pairs):")
    for pair in train_pairs:
        r = oos.get(pair, {})
        print(f"  {pair}: {r.get('n_trades',0)}T {r.get('total_pnl',0):+.1f}p "
              f"L={r.get('n_long',0)} S={r.get('n_short',0)}")
    print(f"  TOTAL: {total_trades}T {total_pnl:+.1f}p L={total_long} S={total_short}")

    # Full 12-pair OOS
    print("\nLoading remaining pairs for full OOS...")
    for pair in ALL_PAIRS:
        if pair in real_data:
            continue
        try:
            o, h, l, c, mid = load_pair_m5_ohlc(pair)
            n = len(o)
            mc_d, mc_dd = compute_asi_mc(o, h, l, c, n)
            split = int(n * 0.7)
            real_data[pair] = (
                mc_d[:split], mc_dd[:split], mid[:split],
                mc_d[split:], mc_dd[split:], mid[split:],
            )
            del o, h, l, c
            gc.collect()
        except Exception as e:
            print(f"  {pair}: SKIP ({e})")

    full_eval = AsiMcEvaluator(real_data, max_hold=args.max_hold)
    full_oos = full_eval.eval_oos(winner, config)

    print(f"\nFull 12-pair OOS:")
    full_total = full_trades = full_long = full_short = 0
    for pair in ALL_PAIRS:
        r = full_oos.get(pair, {})
        if not r:
            continue
        print(f"  {pair}: {r.get('n_trades',0)}T {r.get('total_pnl',0):+.1f}p "
              f"Sh={r.get('sharpe',0):.2f} L={r.get('n_long',0)} S={r.get('n_short',0)}")
        full_total += r.get("total_pnl", 0)
        full_trades += r.get("n_trades", 0)
        full_long += r.get("n_long", 0)
        full_short += r.get("n_short", 0)
    print(f"  ALL: {full_trades}T {full_total:+.1f}p L={full_long} S={full_short}")

    tg_send(f"🔧 Fine-tune done ({elapsed:.0f}s)\n"
            f"Fitness: {winner.fitness:.4f}\n"
            f"12-pair OOS: {full_trades}T {full_total:+.1f}p\n"
            f"L={full_long} S={full_short}\n"
            f"Size: {winner.size()}")

    # Save
    out_path = RESULTS_DIR / f"asi_mc_finetuned_s{args.seed}_best.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"genome": winner, "config": config}, f)

    result = {
        "seed": args.seed, "gens": args.gens,
        "train_pairs": train_pairs,
        "is_fitness": round(float(winner.fitness), 4),
        "network_size": list(winner.size()),
        "oos": {k: v for k, v in full_oos.items()},
    }
    with open(RESULTS_DIR / f"asi_mc_finetuned_s{args.seed}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
