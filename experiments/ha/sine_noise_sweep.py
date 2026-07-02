#!/usr/bin/env python3
"""
Sine Wave Noise Sweep — How much noise can NEAT handle?
========================================================
Runs the same sine wave test at increasing noise levels:
  0, 1, 2, 3, 5, 8, 10 pips std (amplitude = ±10 pips = 20 pip range)

Two configs:
  A) Standard activations: tanh, sigmoid, relu
  B) + sin/cos activations (Fourier-capable)

Compares convergence speed, network complexity, and OOS capture ratio.
"""

import sys
import os
import time
import json
import math
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import neat
from lib.fast_eval import extract_network
from research.experiments.ha.sine_wave_test import (
    compute_mc_ha, evaluate_sine_3out_jit, SineEvaluator,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent


def generate_noisy_sine(n_bars=50000, period_bars=500, amplitude_pips=10,
                        noise_std_pips=0, center=1.1244, pip=0.0001, seed=42):
    """Sine wave + Gaussian noise."""
    rng = np.random.RandomState(seed)
    t = np.arange(n_bars, dtype=np.float64)
    clean = center + amplitude_pips * pip * np.sin(2 * np.pi * t / period_bars)
    noise = rng.normal(0, noise_std_pips * pip, n_bars) if noise_std_pips > 0 else 0.0
    return clean + noise


def register_sincos():
    """Register sin/cos activations with NEAT-Python."""
    def sin_activation(x):
        return math.sin(x)
    def cos_activation(x):
        return math.cos(x)
    try:
        neat.genome.DefaultGenome.add_activation('sin', sin_activation)
        neat.genome.DefaultGenome.add_activation('cos', cos_activation)
    except Exception:
        pass  # Already registered


def run_one(noise_std, use_sincos, generations=100, pop_size=150, seed=42):
    """Run one training variant. Returns result dict."""
    np.random.seed(seed)

    # Generate data
    mid = generate_noisy_sine(50000, 500, 10, noise_std_pips=noise_std, seed=seed)
    mc_d, mc_dd = compute_mc_ha(mid)

    # Config
    if use_sincos:
        register_sincos()

    config_file = EXPERIMENT_DIR / "neat_config_3out.ini"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_file))
    config.genome_config.num_inputs = 3
    config.pop_size = pop_size

    if use_sincos:
        config.genome_config.activation_options = ['tanh', 'sigmoid', 'relu', 'sin', 'cos']

    evaluator = SineEvaluator(mc_d, mc_dd, mid, max_hold=500)

    # Train
    t0 = time.time()
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Track when plateau is first reached
    best_per_gen = []

    class FitnessTracker(neat.reporting.BaseReporter):
        def post_evaluate(self, config, population, species, best_genome):
            best_per_gen.append((len(best_per_gen), best_genome.fitness,
                                 best_genome.size()))

    pop.add_reporter(FitnessTracker())
    winner = pop.run(evaluator.evaluate, generations)
    elapsed = time.time() - t0

    # Find convergence gen (first gen reaching 95% of final fitness)
    final_fit = winner.fitness
    converge_gen = generations
    for gen, fit, size in best_per_gen:
        if fit >= final_fit * 0.95:
            converge_gen = gen
            break

    # Simplest network at plateau
    plateau_sizes = [(g, s) for g, f, s in best_per_gen if f >= final_fit * 0.95]
    simplest = min(plateau_sizes, key=lambda x: x[1][0] + x[1][1]) if plateau_sizes else (0, (0, 0))

    oos = evaluator.eval_oos(winner, config)

    # Theoretical optimum
    n_oos_cycles = (50000 * 0.3) / 500
    optimal_pnl = n_oos_cycles * 18  # 20 amp - 2 spread
    capture = (oos['total_pnl'] / optimal_pnl * 100) if optimal_pnl > 0 else 0

    return {
        "noise_std": noise_std,
        "sincos": use_sincos,
        "is_fitness": round(float(final_fit), 4),
        "converge_gen": converge_gen,
        "simplest_at_plateau": simplest[1],
        "final_size": winner.size(),
        "oos_trades": oos["n_trades"],
        "oos_pnl": oos["total_pnl"],
        "oos_sharpe": oos["sharpe"],
        "oos_wr": oos["win_rate"],
        "capture_pct": round(capture, 1),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    noise_levels = [0, 1, 2, 3, 5, 8, 10]
    results = []

    print(f"{'='*90}")
    print(f"  Sine Wave Noise Sweep — Standard vs Sin/Cos Activations")
    print(f"  Noise levels: {noise_levels} pips std (amplitude ±10 pips)")
    print(f"{'='*90}")

    for noise in noise_levels:
        for sincos in [False, True]:
            label = f"noise={noise:>2}p {'sin/cos' if sincos else 'standard':>8}"
            print(f"\n── {label} ──")
            r = run_one(noise, sincos, generations=100, pop_size=150)
            results.append(r)
            print(f"  Conv gen: {r['converge_gen']:>3} | "
                  f"Simplest: {r['simplest_at_plateau']} | "
                  f"OOS: {r['oos_pnl']:>+7.1f}p {r['oos_trades']}T "
                  f"Sharpe={r['oos_sharpe']:.2f} "
                  f"Capture={r['capture_pct']:.0f}%")

    # Summary table
    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}")
    print(f"{'Noise':>5} {'Type':>8} {'ConvGen':>7} {'Simplest':>10} {'OOS PnL':>9} {'Trades':>6} "
          f"{'Sharpe':>7} {'WR':>5} {'Capture':>7}")
    print("-" * 90)
    for r in results:
        typ = "sin/cos" if r["sincos"] else "std"
        sz = f"{r['simplest_at_plateau'][0]}n{r['simplest_at_plateau'][1]}c"
        print(f"{r['noise_std']:>4}p {typ:>8} {r['converge_gen']:>7} {sz:>10} "
              f"{r['oos_pnl']:>+9.1f} {r['oos_trades']:>6} "
              f"{r['oos_sharpe']:>7.2f} {r['oos_wr']:>4.0f}% {r['capture_pct']:>6.0f}%")

    # Save
    results_dir = EXPERIMENT_DIR / "results" / "sine_wave"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "noise_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_dir}/noise_sweep_results.json")


if __name__ == "__main__":
    main()
