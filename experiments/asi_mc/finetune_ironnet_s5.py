#!/usr/bin/env python3
"""
IronNet S5-Cadence Fine-Tuning
================================
Takes a pre-trained IronNet genome and continues WF P&L evolution
at S5 cadence (12× more evaluations per M5 bar).

Matches live curator behavior:
- MC_D, MC_dD, ER_norm update every M5 (held constant for 12 S5 bars)
- UPnL updates every S5 bar from S5 mid price
- Network evaluates every S5 bar

Input: S5-cadence parquet from export_s5_training_data.py
  Columns: s5_mid, mc_d, mc_dd, er_norm, m5_idx

Usage:
  python3 finetune_ironnet_s5.py --pair EUR_GBP --seed 42 --genome models/iron_v3_EUR_GBP.pkl
"""

import sys
import os
import gc
import time
import copy
import json
import pickle
import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2] if len(SCRIPT_DIR.parts) > 4 else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import neat
from lib.fast_eval import extract_network, _activate

S5_DATA_DIR = Path(os.environ.get("S5_DATA_DIR",
                   str(PROJECT_ROOT / "data" / "s5_ironnet")))
RESULTS_DIR = SCRIPT_DIR / "results" / "ironnet_s5"

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

N_CHUNKS = 3


def tg_send(text):
    try:
        import requests
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        if token and chat:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text, "parse_mode": "HTML"},
                          timeout=5)
    except Exception:
        pass


# Activations
def gauss_activation(x): return math.exp(-x * x)
def sin_activation(x): return math.sin(x)
def cos_activation(x): return math.cos(x)
def tanh_activation(x): return math.tanh(x)

def register_activations(config):
    for name, fn in [('gauss', gauss_activation), ('sin', sin_activation),
                     ('cos', cos_activation), ('tanh', tanh_activation)]:
        try: config.genome_config.add_activation(name, fn)
        except: pass


# ═══════════════════════════════════════════════════════════════════════════
# S5-Cadence JIT Evaluator
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def evaluate_s5_chunk_jit(
    s5_mid, mc_d, mc_dd, er_norm,
    pip, spread_pips, max_hold_s5,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices,
    chunk_start, chunk_end,
):
    """Evaluate at S5 cadence. Indicators held constant per M5 (12 S5 bars).
    max_hold_s5 is in S5 bars (e.g., 200 M5 bars × 12 = 2400 S5 bars).
    """
    values = np.zeros(total_values)
    start_bar = max(chunk_start + 120, 120)  # skip warmup (10 M5 = 120 S5)
    end_bar = min(chunk_end, len(s5_mid) - 1)
    max_trades = (end_bar - start_bar) // 6 + 1  # rough upper bound
    trade_pnls = np.zeros(max_trades)
    trade_maes = np.zeros(max_trades)
    n_trades = 0
    n_long = 0
    n_short = 0
    position = 0
    entry_price = 0.0
    entry_bar = 0
    worst_ae = 0.0

    for i in range(start_bar, end_bar):
        if position != 0:
            pnl_pips = (s5_mid[i] - entry_price) / pip * position - spread_pips
            ae = -(s5_mid[i] - entry_price) / pip * position + spread_pips
            if ae > worst_ae:
                worst_ae = ae
        else:
            pnl_pips = 0.0

        # Set inputs: indicators (held constant per M5) + UPnL (updates every S5)
        values[0] = mc_d[i]
        values[1] = mc_dd[i]
        values[2] = er_norm[i]
        values[3] = np.tanh(pnl_pips / 20.0)

        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)
        out_buy = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_flat = values[output_indices[2]]

        # Max hold
        if position != 0 and (i - entry_bar) >= max_hold_s5:
            pnl = (s5_mid[i] - entry_price) / pip * position - spread_pips
            if n_trades < max_trades:
                trade_pnls[n_trades] = pnl
                trade_maes[n_trades] = worst_ae
                if position > 0: n_long += 1
                else: n_short += 1
                n_trades += 1
            position = 0; worst_ae = 0.0

        if position == 0:
            if out_buy > out_sell and out_buy > out_flat:
                position = 1; entry_price = s5_mid[i]; entry_bar = i; worst_ae = 0.0
            elif out_sell > out_buy and out_sell > out_flat:
                position = -1; entry_price = s5_mid[i]; entry_bar = i; worst_ae = 0.0
        else:
            close = False; new_pos = 0
            if out_flat > out_buy and out_flat > out_sell:
                close = True
            elif position == 1 and out_sell > out_buy and out_sell > out_flat:
                close = True; new_pos = -1
            elif position == -1 and out_buy > out_sell and out_buy > out_flat:
                close = True; new_pos = 1
            if close:
                pnl = (s5_mid[i] - entry_price) / pip * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    trade_maes[n_trades] = worst_ae
                    if position > 0: n_long += 1
                    else: n_short += 1
                    n_trades += 1
                position = new_pos
                entry_price = s5_mid[i] if new_pos != 0 else 0.0
                entry_bar = i; worst_ae = 0.0

    if position != 0 and end_bar > start_bar:
        pnl = (s5_mid[end_bar - 1] - entry_price) / pip * position - spread_pips
        if n_trades < max_trades:
            trade_pnls[n_trades] = pnl
            trade_maes[n_trades] = worst_ae
            if position > 0: n_long += 1
            else: n_short += 1
            n_trades += 1

    if n_trades < 1:
        return 0, 0.0, 0.0, 0, 0
    total_pnl = 0.0
    total_mae = 0.0
    for j in range(n_trades):
        total_pnl += trade_pnls[j]
        total_mae += trade_maes[j]
    avg_mae = total_mae / n_trades
    return n_trades, total_pnl, avg_mae, n_long, n_short


class S5WFEvaluator:
    """Walk-Forward evaluator at S5 cadence."""

    def __init__(self, s5_mid, mc_d, mc_dd, er_norm, pip, spread,
                 max_hold_m5=200, n_chunks=3, min_dir_ratio=0.15):
        self.s5_mid = s5_mid
        self.mc_d = mc_d
        self.mc_dd = mc_dd
        self.er_norm = er_norm
        self.pip = pip
        self.spread = spread
        self.max_hold_s5 = max_hold_m5 * 12  # convert M5 bars to S5
        self.n_chunks = n_chunks
        self.min_dir_ratio = min_dir_ratio
        self.n_bars = len(s5_mid)

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return -10.0

        chunk_scores = []
        total_long = 0
        total_short = 0
        total_trades = 0

        for ci in range(self.n_chunks):
            c_start = int(self.n_bars * ci / self.n_chunks)
            c_end = int(self.n_bars * (ci + 1) / self.n_chunks)

            nt, pnl, mae, nl, ns = evaluate_s5_chunk_jit(
                self.s5_mid, self.mc_d, self.mc_dd, self.er_norm,
                self.pip, self.spread, self.max_hold_s5,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10],
                c_start, c_end)

            total_long += nl
            total_short += ns
            total_trades += nt

            min_trades = max(30, int(self.n_bars / self.n_chunks / 288 / 12 * 0.5))
            if nt < min_trades:
                return -10.0
            if pnl <= 0:
                return -10.0

            mean_pnl = pnl / nt
            score = mean_pnl / mae if mae > 0 else mean_pnl
            chunk_scores.append(score * (nt ** 0.5))

        if total_trades < 10:
            return -10.0
        dir_ratio = min(total_long, total_short) / total_trades
        if dir_ratio < self.min_dir_ratio:
            return -10.0

        min_score = min(chunk_scores)
        mean_score = sum(chunk_scores) / len(chunk_scores)
        if mean_score > 0:
            cv = (sum((s - mean_score)**2 for s in chunk_scores) / len(chunk_scores))**0.5 / mean_score
            consistency = 1.0 / (1.0 + cv)
        else:
            consistency = 0.5
        dir_bonus = 1.0 + 0.5 * (dir_ratio - self.min_dir_ratio) / (0.5 - self.min_dir_ratio)
        return min_score * (1.0 + consistency) * dir_bonus


# ═══════════════════════════════════════════════════════════════════════════
# IronNet mutation (same as train_ironnet_perpair.py)
# ═══════════════════════════════════════════════════════════════════════════

def mutate_ironnet(genome, config):
    act_choices = ['tanh', 'sin', 'cos', 'gauss']
    for key, conn in genome.connections.items():
        r = np.random.random()
        if r < 0.1:
            conn.weight = np.random.uniform(-5, 5)
        elif r < 0.8:
            conn.weight += np.random.normal(0, 0.5)
            conn.weight = max(-5.0, min(5.0, conn.weight))
    for nid, node in genome.nodes.items():
        r = np.random.random()
        if r < 0.1:
            node.bias = np.random.uniform(-5, 5)
        elif r < 0.7:
            node.bias += np.random.normal(0, 0.5)
            node.bias = max(-5.0, min(5.0, node.bias))
        if np.random.random() < 0.15:
            node.activation = act_choices[np.random.randint(4)]


def eval_oos_s5(genome, config, s5_mid, mc_d, mc_dd, er_norm, pip, spread, max_hold_m5=200):
    net = extract_network(genome, config)
    nt, pnl, mae, nl, ns = evaluate_s5_chunk_jit(
        s5_mid, mc_d, mc_dd, er_norm,
        pip, spread, max_hold_m5 * 12,
        net[0], net[2], net[3], net[4], net[5], net[6],
        net[7], net[8], net[9], net[10],
        0, len(s5_mid))
    n_days = len(s5_mid) / 288.0 / 12.0
    return {
        "n_trades": int(nt), "total_pnl": round(float(pnl), 1),
        "avg_mae": round(float(mae), 2),
        "n_long": int(nl), "n_short": int(ns),
        "pips_per_day": round(float(pnl) / max(n_days, 1), 1),
        "dir_ratio": round(min(nl, ns) / max(nt, 1), 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IronNet S5-cadence fine-tuning")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--genome", required=True, help="Path to pre-trained IronNet genome")
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--pop", type=int, default=150)
    parser.add_argument("--max-hold", type=int, default=200, help="Max hold in M5 bars")
    parser.add_argument("--stall-limit", type=int, default=40)
    parser.add_argument("--min-dir-ratio", type=float, default=0.15)
    args = parser.parse_args()

    np.random.seed(args.seed)
    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]

    print(f"{'='*65}")
    print(f"  IronNet S5-Cadence Fine-Tuning: {pair}")
    print(f"  Seed genome: {args.genome}")
    print(f"  Seed: {args.seed} | {args.islands}×{args.pop} pop | {args.gens} gens")
    print(f"  WF chunks: {N_CHUNKS} | Bidir ≥{args.min_dir_ratio*100:.0f}%")
    print(f"{'='*65}")
    tg_send(f"🔩S5 IronNet FT {pair} s{args.seed}\n{args.gens}g from {Path(args.genome).name}")

    # Load seed genome
    with open(args.genome, "rb") as f:
        data = pickle.load(f)
    seed_genome = data["genome"]
    print(f"  Seed: size={seed_genome.size()}, fitness={seed_genome.fitness:.4f}")

    # Load S5 training data
    s5_path = S5_DATA_DIR / f"{pair}_s5_ironnet.parquet"
    if not s5_path.exists():
        print(f"ERROR: {s5_path} not found. Run export_s5_training_data.py first.")
        return
    df = pd.read_parquet(s5_path, engine="pyarrow")
    s5_mid = df["s5_mid"].values.astype(np.float64)
    mc_d = df["mc_d"].values.astype(np.float64)
    mc_dd = df["mc_dd"].values.astype(np.float64)
    er_norm = df["er_norm"].values.astype(np.float64)
    n_s5 = len(s5_mid)
    split = int(n_s5 * 0.7)

    print(f"  S5 data: {n_s5:,} bars | IS: {split:,} | OOS: {n_s5 - split:,}")
    print(f"  ≈ {split // 12:,} M5 bars IS, {(n_s5 - split) // 12:,} M5 bars OOS")
    del df; gc.collect()

    # Setup
    config_path = SCRIPT_DIR / "neat_config_4in_3out.ini"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_path))
    register_activations(config)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"iron_s5_{pair}_s{args.seed}"

    # Evaluate seed genome at S5 cadence first (baseline)
    print(f"\n  Baseline (M5-trained genome at S5 cadence)...")
    baseline_is = eval_oos_s5(seed_genome, config,
                               s5_mid[:split], mc_d[:split], mc_dd[:split], er_norm[:split],
                               pip, spread, args.max_hold)
    baseline_oos = eval_oos_s5(seed_genome, config,
                                s5_mid[split:], mc_d[split:], mc_dd[split:], er_norm[split:],
                                pip, spread, args.max_hold)
    print(f"  Baseline IS:  {baseline_is['n_trades']}T {baseline_is['total_pnl']:+.1f}p "
          f"dir={baseline_is['dir_ratio']:.2f} ({baseline_is['pips_per_day']:.1f}p/day)")
    print(f"  Baseline OOS: {baseline_oos['n_trades']}T {baseline_oos['total_pnl']:+.1f}p "
          f"dir={baseline_oos['dir_ratio']:.2f} ({baseline_oos['pips_per_day']:.1f}p/day)")

    # WF evaluator on IS data at S5 cadence
    wf_eval = S5WFEvaluator(s5_mid[:split], mc_d[:split], mc_dd[:split], er_norm[:split],
                             pip, spread, max_hold_m5=args.max_hold,
                             n_chunks=N_CHUNKS, min_dir_ratio=args.min_dir_ratio)

    # Build seeded islands
    islands = []
    for i in range(args.islands):
        pop = {}
        for j in range(args.pop):
            clone = copy.deepcopy(seed_genome)
            clone.key = j; clone.fitness = None
            mutate_ironnet(clone, config)
            pop[j] = clone
        islands.append({"pop": pop, "best": None})

    save_dir = str(RESULTS_DIR / f"{tag}_ckpt")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    best_ever = None; best_ever_fitness = -999; stall = 0
    t0 = time.time()

    for gen in range(args.gens):
        for i, island in enumerate(islands):
            pop = island["pop"]
            wf_eval.evaluate(list(pop.items()), config)
            best = max(pop.values(), key=lambda g: g.fitness if g.fitness is not None else -999)
            island["best"] = best
            if best.fitness is not None and best.fitness > best_ever_fitness:
                best_ever = copy.deepcopy(best); best_ever_fitness = best.fitness; stall = 0

            sorted_g = sorted(pop.values(),
                               key=lambda g: g.fitness if g.fitness is not None else -999, reverse=True)
            new_pop = {}
            for j in range(min(3, len(sorted_g))):
                e = copy.deepcopy(sorted_g[j]); e.key = j; e.fitness = None; new_pop[j] = e
            for j in range(3, args.pop):
                cands = np.random.choice(len(sorted_g), size=min(3, len(sorted_g)), replace=False)
                p = copy.deepcopy(sorted_g[min(cands)]); p.key = j; p.fitness = None
                mutate_ironnet(p, config); new_pop[j] = p
            island["pop"] = new_pop

        if gen > 0 and gen % 10 == 0:
            for i in range(args.islands):
                src = islands[i]; dst = islands[(i + 1) % args.islands]
                if src["best"] is not None:
                    wg = min(dst["pop"], key=lambda g: dst["pop"][g].fitness if dst["pop"][g].fitness is not None else 999)
                    m = copy.deepcopy(src["best"]); m.key = wg; m.fitness = None; dst["pop"][wg] = m

        stall += 1

        gen_best = max((isl["best"] for isl in islands if isl["best"] is not None),
                       key=lambda g: g.fitness if g.fitness is not None else -999, default=None)
        if gen_best:
            with open(f"{save_dir}/gen_{gen:03d}_best.pkl", "wb") as f:
                pickle.dump({"genome": copy.deepcopy(gen_best), "config": config,
                             "generation": gen, "fitness": gen_best.fitness}, f)

        if gen % 10 == 0:
            fitnesses = [isl["best"].fitness for isl in islands
                         if isl["best"] is not None and isl["best"].fitness is not None]
            if fitnesses:
                print(f"  [{pair} S5] Gen {gen:>3}: best={max(fitnesses):.4f} "
                      f"global={best_ever_fitness:.4f} stall={stall}")

        if stall >= args.stall_limit:
            print(f"  [{pair} S5] Early stop gen {gen} (stalled {args.stall_limit})")
            break

    elapsed = time.time() - t0
    print(f"  Fine-tuning done: fitness={best_ever_fitness:.4f} ({elapsed:.0f}s)")

    # OOS evaluation
    is_res = eval_oos_s5(best_ever, config,
                          s5_mid[:split], mc_d[:split], mc_dd[:split], er_norm[:split],
                          pip, spread, args.max_hold)
    oos_res = eval_oos_s5(best_ever, config,
                           s5_mid[split:], mc_d[split:], mc_dd[split:], er_norm[split:],
                           pip, spread, args.max_hold)

    print(f"\n{'='*65}")
    print(f"  RESULTS: {pair} (IronNet S5 Fine-Tuned)")
    print(f"{'='*65}")
    print(f"  Baseline OOS: {baseline_oos['n_trades']}T {baseline_oos['total_pnl']:+.1f}p "
          f"dir={baseline_oos['dir_ratio']:.2f} ({baseline_oos['pips_per_day']:.1f}p/day)")
    print(f"  S5-FT   OOS:  {oos_res['n_trades']}T {oos_res['total_pnl']:+.1f}p "
          f"dir={oos_res['dir_ratio']:.2f} ({oos_res['pips_per_day']:.1f}p/day)")
    delta = oos_res['total_pnl'] - baseline_oos['total_pnl']
    print(f"  Delta: {delta:+.1f}p ({100*delta/max(abs(baseline_oos['total_pnl']),1):+.1f}%)")

    acts = [n.activation for n in best_ever.nodes.values()]
    print(f"  Fitness: {best_ever_fitness:.4f} | Activations: {dict((a, acts.count(a)) for a in set(acts))}")

    result_data = {
        "pair": pair, "variant": "ironnet_s5",
        "seed": args.seed, "gens": args.gens,
        "actual_gens": gen + 1,
        "fitness": round(best_ever_fitness, 4),
        "baseline_oos": baseline_oos,
        "s5ft_is": is_res,
        "s5ft_oos": oos_res,
        "delta_pips": round(delta, 1),
        "elapsed_s": round(elapsed, 1),
    }

    with open(RESULTS_DIR / f"{tag}_best.pkl", "wb") as f:
        pickle.dump({"genome": best_ever, "config": config, "pair": pair}, f)
    with open(RESULTS_DIR / f"{tag}_result.json", "w") as f:
        json.dump(result_data, f, indent=2)

    tg_send(f"🔩S5 IronNet {pair} s{args.seed} DONE\n"
            f"Baseline: {baseline_oos['total_pnl']:+.1f}p\n"
            f"S5-FT: {oos_res['total_pnl']:+.1f}p ({delta:+.1f}p)\n"
            f"dir={oos_res['dir_ratio']:.2f}")

    print(f"\nSaved: {tag}_best.pkl + {tag}_result.json")


if __name__ == "__main__":
    main()
