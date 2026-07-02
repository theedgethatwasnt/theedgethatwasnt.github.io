#!/usr/bin/env python3
"""
Island NEAT Fine-tune: seed with original genome, 4 islands, migration.
======================================================================
- 4 independent NEAT populations (islands) seeded from the original genome
- Every 10 gens: migrate best genome from each island to the next
- Save best genome per generation (never overwritten)
- 3 inputs (MC_D, MC_dD, UPnL), 3 outputs (BUY/SELL/FLATTEN)
- Activations: tanh, sin, cos
"""

import sys
import os
import gc
import time
import json
import copy
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

N_ISLANDS = 4
MIGRATE_EVERY = 10
ISLAND_POP = 75  # Per island (total = 4 × 75 = 300)


class IslandRunner:
    """Runs N independent NEAT populations with periodic migration."""

    def __init__(self, config_path, n_islands, pop_per_island, seed_genome=None):
        self.n_islands = n_islands
        self.islands = []
        self.best_ever = None
        self.best_ever_fitness = -999

        for i in range(n_islands):
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 str(config_path))
            config.pop_size = pop_per_island

            pop = neat.Population(config)

            # Seed with original genome: deep copy + mutate each member
            if seed_genome is not None:
                for gid in list(pop.population.keys()):
                    clone = copy.deepcopy(seed_genome)
                    clone.key = gid
                    clone.fitness = None
                    clone.mutate(config.genome_config)
                    pop.population[gid] = clone

            self.islands.append({"pop": pop, "config": config, "best": None})

    def run(self, evaluator_fn, generations, save_dir, migrate_every=10):
        """Run all islands for N generations with migration.

        evaluator_fn: callable(genomes, config) — NEAT fitness function
        save_dir: directory for per-generation checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for gen in range(generations):
            # Evaluate all islands
            for i, island in enumerate(self.islands):
                pop = island["pop"]
                config = island["config"]

                # Run 1 generation
                # NEAT's pop.run() does full N gens. We need per-gen control.
                # Use internal methods instead.
                if gen == 0:
                    pop.generation = 0

                # Evaluate fitness
                evaluator_fn(list(pop.population.items()), config)

                # Find best in this island
                best = max(pop.population.values(), key=lambda g: g.fitness if g.fitness is not None else -999)
                island["best"] = best

                # Track global best
                if best.fitness is not None and best.fitness > self.best_ever_fitness:
                    self.best_ever = copy.deepcopy(best)
                    self.best_ever_fitness = best.fitness

                # Reproduce (create next generation)
                pop.species.speciate(config, pop.population, pop.generation)
                pop.population = pop.reproduction.reproduce(
                    config, pop.species, config.pop_size, pop.generation)
                pop.generation += 1

            # Migration: ring topology, send best from island i to island i+1
            if gen > 0 and gen % migrate_every == 0:
                for i in range(self.n_islands):
                    src = self.islands[i]
                    dst = self.islands[(i + 1) % self.n_islands]
                    if src["best"] is not None:
                        # Replace worst genome in destination with migrant
                        worst_gid = min(dst["pop"].population,
                                        key=lambda g: dst["pop"].population[g].fitness
                                        if dst["pop"].population[g].fitness is not None else 999)
                        migrant = copy.deepcopy(src["best"])
                        migrant.key = worst_gid
                        migrant.fitness = None  # Will be re-evaluated
                        dst["pop"].population[worst_gid] = migrant

            # Save best of this generation
            gen_best = max(
                (isl["best"] for isl in self.islands if isl["best"] is not None),
                key=lambda g: g.fitness if g.fitness is not None else -999,
                default=None
            )
            if gen_best is not None:
                ckpt_path = save_dir / f"gen_{gen:03d}_best.pkl"
                with open(ckpt_path, "wb") as f:
                    pickle.dump({
                        "genome": copy.deepcopy(gen_best),
                        "config": self.islands[0]["config"],
                        "generation": gen,
                        "fitness": gen_best.fitness,
                    }, f)

            # Log
            fitnesses = [isl["best"].fitness for isl in self.islands
                         if isl["best"] is not None and isl["best"].fitness is not None]
            if fitnesses:
                avg = sum(fitnesses) / len(fitnesses)
                best_f = max(fitnesses)
                print(f"  Gen {gen:>3}: best={best_f:.4f} avg={avg:.4f} "
                      f"global_best={self.best_ever_fitness:.4f}"
                      f"{' [MIGRATE]' if gen > 0 and gen % migrate_every == 0 else ''}")

        return self.best_ever


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", default=str(RESULTS_DIR / "asi_mc_s42_best_ORIGINAL.pkl"))
    parser.add_argument("--pairs", default="EUR_JPY,USD_JPY,GBP_JPY,GBP_USD")
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--pop-per-island", type=int, default=75)
    parser.add_argument("--migrate-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hold", type=int, default=200)
    args = parser.parse_args()

    np.random.seed(args.seed)
    train_pairs = [p.strip() for p in args.pairs.split(",")]

    print(f"{'='*60}")
    print(f"  Island NEAT Fine-tune")
    print(f"  {args.islands} islands × {args.pop_per_island} pop = {args.islands * args.pop_per_island} total")
    print(f"  Migrate every {args.migrate_every} gens")
    print(f"  Genome: {Path(args.genome).name}")
    print(f"  Pairs: {train_pairs}")
    print(f"  Gens: {args.gens}")
    print(f"{'='*60}")

    tg_send(f"🏝 Island NEAT starting\n"
            f"{args.islands} islands × {args.pop_per_island} pop\n"
            f"Pairs: {train_pairs}\n"
            f"Gens: {args.gens}, migrate every {args.migrate_every}")

    # Load seed genome
    with open(args.genome, "rb") as f:
        d = pickle.load(f)
    seed_genome = d["genome"]
    print(f"Seed genome: size={seed_genome.size()}")

    # Load data
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

    # Create island runner
    config_path = SCRIPT_DIR / "neat_config_3out.ini"
    runner = IslandRunner(config_path, args.islands, args.pop_per_island, seed_genome)

    # Run
    save_dir = RESULTS_DIR / "island_checkpoints"
    t0 = time.time()
    winner = runner.run(evaluator.evaluate, args.gens, save_dir, args.migrate_every)
    elapsed = time.time() - t0

    print(f"\nIsland NEAT done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Best fitness: {winner.fitness:.4f}")
    print(f"  Size: {winner.size()}")

    config = runner.islands[0]["config"]

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
              f"Sh={r.get('sharpe',0):.2f} L={r.get('n_long',0)} S={r.get('n_short',0)}")
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

    tg_send(f"🏝 Island NEAT done ({elapsed:.0f}s)\n"
            f"Fitness: {winner.fitness:.4f}\n"
            f"12-pair OOS: {full_trades}T {full_total:+.1f}p\n"
            f"L={full_long} S={full_short}\n"
            f"Size: {winner.size()}\n"
            f"Checkpoints: {args.gens} saved")

    # Save final best
    out_path = RESULTS_DIR / f"island_best_s{args.seed}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"genome": winner, "config": config}, f)

    result = {
        "seed": args.seed, "gens": args.gens,
        "islands": args.islands, "pop_per_island": args.pop_per_island,
        "migrate_every": args.migrate_every,
        "train_pairs": train_pairs,
        "is_fitness": round(float(winner.fitness), 4),
        "network_size": list(winner.size()),
        "oos": {k: v for k, v in full_oos.items()},
        "elapsed_s": round(elapsed, 1),
    }
    with open(RESULTS_DIR / f"island_best_s{args.seed}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"Checkpoints: {save_dir}/ ({args.gens} files)")


if __name__ == "__main__":
    main()
