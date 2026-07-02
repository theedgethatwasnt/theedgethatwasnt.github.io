#!/usr/bin/env python3
"""
NEAT Training on 10-pip Range Bars with WF-in-Fitness.

Same proven framework as train_wf_pnlmae.py but on range bars instead of M5.
Inputs: MC(D) + MC(dD) + UPnL (3 inputs, same as Variant A)
Fitness: PnL/MAE × n_trades^exp with WF validation built in.

Usage:
  python3 train_range_bar_neat.py --gens 200 --seed 42
"""

import sys, os, time, pickle, copy, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).parent))

import neat
from lib.fast_eval import extract_network, _activate

DATA_DIR = PROJECT / "data" / "range_bar_indicators"
RESULTS_DIR = Path(__file__).parent / "results"

PAIR_PIP = {"EUR_JPY":0.01,"USD_JPY":0.01,"GBP_JPY":0.01,"AUD_JPY":0.01,"CAD_JPY":0.01,"CHF_JPY":0.01,"NZD_JPY":0.01,"EUR_USD":0.0001,"GBP_USD":0.0001,"AUD_USD":0.0001,"NZD_USD":0.0001,"EUR_GBP":0.0001}
PAIR_SPREAD = {"EUR_JPY":2.3,"USD_JPY":1.7,"GBP_JPY":3.3,"AUD_JPY":2.1,"CAD_JPY":2.3,"CHF_JPY":3.5,"NZD_JPY":2.7,"EUR_USD":1.6,"GBP_USD":1.9,"AUD_USD":1.3,"NZD_USD":1.5,"EUR_GBP":1.4}

GENS = 200; FREQ_EXP = 0.5; N_CHUNKS = 3

def tg_send(text):
    try:
        import requests
        requests.post("https://api.telegram.org/bot{os.environ.get("TELEGRAM_BOT_TOKEN","")}/sendMessage",
                      json={"chat_id":os.environ.get("TELEGRAM_CHAT_ID",""),"text":text,"parse_mode":"HTML"},timeout=10)
    except: pass


@njit(cache=True)
def eval_range_chunk(mc_d, mc_dd, mid_close, pip, spread_pips, max_hold,
                     n_inputs, n_eval, total_values, node_bias, node_response, node_act,
                     conn_from, conn_to, conn_weight, output_indices,
                     chunk_start, chunk_end):
    """Evaluate NEAT genome on a chunk of range bar data."""
    values = np.zeros(total_values)
    start_bar = max(chunk_start + 2, 2)
    end_bar = min(chunk_end, len(mid_close) - 1)
    max_t = end_bar - start_bar + 1
    pnls = np.zeros(max_t); maes = np.zeros(max_t)
    nt = 0; pos = 0; ep = 0.0; eb = 0; rmae = 0.0

    for i in range(start_bar, end_bar):
        if pos != 0:
            pp = (mid_close[i] - ep) / pip * pos - spread_pips
            if pp < rmae: rmae = pp
        else:
            pp = 0.0
        values[0] = mc_d[i]; values[1] = mc_dd[i]; values[2] = np.tanh(pp / 20.0)
        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)
        ob = values[output_indices[0]]; os_ = values[output_indices[1]]; of = values[output_indices[2]]

        if pos != 0 and (i - eb) >= max_hold:
            pnl = (mid_close[i] - ep) / pip * pos - spread_pips
            if nt < max_t: pnls[nt] = pnl; maes[nt] = rmae; nt += 1
            pos = 0
        if pos == 0:
            if ob > os_ and ob > of: pos = 1; ep = mid_close[i]; eb = i; rmae = 0.0
            elif os_ > ob and os_ > of: pos = -1; ep = mid_close[i]; eb = i; rmae = 0.0
        else:
            cl = False; np_ = 0
            if of > ob and of > os_: cl = True
            elif pos == 1 and os_ > ob and os_ > of: cl = True; np_ = -1
            elif pos == -1 and ob > os_ and ob > of: cl = True; np_ = 1
            if cl:
                pnl = (mid_close[i] - ep) / pip * pos - spread_pips
                if nt < max_t: pnls[nt] = pnl; maes[nt] = rmae; nt += 1
                pos = np_; ep = mid_close[i] if np_ != 0 else 0.0; eb = i; rmae = 0.0

    if pos != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar-1] - ep) / pip * pos - spread_pips
        if nt < max_t: pnls[nt] = pnl; maes[nt] = rmae; nt += 1

    if nt < 1: return 0, 0.0, 0.0
    tp = 0.0; tm = 0.0
    for j in range(nt): tp += pnls[j]; tm += abs(maes[j])
    return nt, tp, tm / nt


class RangeBarWFEvaluator:
    """WF-in-fitness evaluator for range bar data."""

    def __init__(self, pair_data, max_hold=30, n_chunks=3, freq_exp=0.5, gen_counter=None):
        self.pair_data = pair_data
        self.max_hold = max_hold
        self.n_chunks = n_chunks
        self.freq_exp = freq_exp
        self.current_gen = 0

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness_simple(self, genome, config):
        """Phase 1: fast single-pass."""
        try: net = extract_network(genome, config)
        except: return -10.0
        total_fitness = 0.0; total_trades = 0; n_pairs = 0
        for pair, (mc_d, mc_dd, mid) in self.pair_data.items():
            pip = PAIR_PIP.get(pair, 0.01); spread = PAIR_SPREAD.get(pair, 2.0)
            nt, pnl, mae = eval_range_chunk(mc_d, mc_dd, mid, pip, spread, self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6], net[7], net[8], net[9], net[10], 0, len(mid))
            total_trades += nt
            if nt >= 3:
                mp = pnl / nt
                if mae > 0: total_fitness += mp / mae
                else: total_fitness += mp
                n_pairs += 1
        if n_pairs == 0 or total_trades < 200: return -10.0
        return total_fitness / n_pairs * (total_trades ** self.freq_exp)

    def _fitness(self, genome, config):
        if self.current_gen < 100:
            return self._fitness_simple(genome, config)

        try: net = extract_network(genome, config)
        except: return -10.0
        chunk_scores = []
        for chunk_idx in range(self.n_chunks):
            chunk_trades = 0; chunk_fitness = 0.0; chunk_pairs = 0; chunk_pnl = 0.0
            for pair, (mc_d, mc_dd, mid) in self.pair_data.items():
                n = len(mid)
                c_start = int(n * chunk_idx / self.n_chunks)
                c_end = int(n * (chunk_idx + 1) / self.n_chunks)
                pip = PAIR_PIP.get(pair, 0.01); spread = PAIR_SPREAD.get(pair, 2.0)
                nt, pnl, mae = eval_range_chunk(mc_d, mc_dd, mid, pip, spread, self.max_hold,
                    net[0], net[2], net[3], net[4], net[5], net[6], net[7], net[8], net[9], net[10], c_start, c_end)
                chunk_trades += nt; chunk_pnl += pnl
                if nt >= 3:
                    mp = pnl / nt
                    if mae > 0: chunk_fitness += mp / mae
                    else: chunk_fitness += mp
                    chunk_pairs += 1
            if chunk_trades < 100 or chunk_pairs < 4 or chunk_pnl <= 0: return -10.0
            avg_score = chunk_fitness / max(chunk_pairs, 1)
            chunk_scores.append(avg_score * (chunk_trades ** self.freq_exp))

        min_score = min(chunk_scores)
        mean_score = sum(chunk_scores) / len(chunk_scores)
        std_score = (sum((s - mean_score)**2 for s in chunk_scores) / len(chunk_scores)) ** 0.5
        cv = std_score / mean_score if mean_score > 0 else 1.0
        return min_score * (1.0 / (1.0 + cv))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp", type=float, default=0.5)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--pop", type=int, default=150)
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"{'='*60}")
    print(f"  NEAT on 10-pip Range Bars — WF-in-Fitness")
    print(f"  {args.gens} gens, seed={args.seed}, exp={args.exp}")
    print(f"{'='*60}")
    tg_send(f"🧬 NEAT Range Bar training\n{args.gens}g seed={args.seed} exp={args.exp}")

    # Load range bar data — IS only (70%)
    pair_data = {}
    for pair in PAIR_PIP:
        path = DATA_DIR / f"{pair}_range10_asi_mc.parquet"
        if not path.exists():
            print(f"  {pair}: MISSING")
            continue
        df = pd.read_parquet(path)
        mc_d = df["mc_d"].values.astype(np.float64)
        mc_dd = df["mc_dd"].values.astype(np.float64)
        mid = df["mid_close"].values.astype(np.float64)
        n = len(mid)
        split = int(n * 0.7)
        pair_data[pair] = (mc_d[:split], mc_dd[:split], mid[:split])
        print(f"  {pair}: {n:,} bars, IS={split:,}")

    config_path = Path(__file__).parent / "neat_config_3out.ini"
    evaluator = RangeBarWFEvaluator(pair_data, max_hold=30, freq_exp=args.exp)

    from train_from_indicators import run_island_neat

    # Inline island NEAT with gen tracking
    t0 = time.time()
    islands = []
    best_ever = None; best_ever_fitness = -999
    for i in range(args.islands):
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path))
        cfg.pop_size = args.pop
        pop = neat.Population(cfg)
        islands.append({"pop": pop, "config": cfg, "best": None})

    save_dir = str(RESULTS_DIR / f"range_neat_s{args.seed}_ckpts")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for gen in range(args.gens):
        evaluator.current_gen = gen
        for i, island in enumerate(islands):
            pop = island["pop"]; cfg = island["config"]
            evaluator.evaluate(list(pop.population.items()), cfg)
            best = max(pop.population.values(), key=lambda g: g.fitness if g.fitness is not None else -999)
            island["best"] = best
            if best.fitness is not None and best.fitness > best_ever_fitness:
                best_ever = copy.deepcopy(best); best_ever_fitness = best.fitness
            pop.species.speciate(cfg, pop.population, pop.generation)
            pop.population = pop.reproduction.reproduce(cfg, pop.species, cfg.pop_size, pop.generation)
            pop.generation += 1

        if gen > 0 and gen % 10 == 0:
            for i in range(args.islands):
                src = islands[i]; dst = islands[(i+1) % args.islands]
                if src["best"] is not None:
                    worst_gid = min(dst["pop"].population,
                                    key=lambda g: dst["pop"].population[g].fitness
                                    if dst["pop"].population[g].fitness is not None else 999)
                    migrant = copy.deepcopy(src["best"])
                    migrant.key = worst_gid; migrant.fitness = None
                    dst["pop"].population[worst_gid] = migrant

        gen_best = max((isl["best"] for isl in islands if isl["best"] is not None),
                       key=lambda g: g.fitness if g.fitness is not None else -999, default=None)
        if gen_best and save_dir:
            with open(f"{save_dir}/gen_{gen:03d}_best.pkl", "wb") as f:
                pickle.dump({"genome": copy.deepcopy(gen_best), "config": islands[0]["config"],
                             "generation": gen, "fitness": gen_best.fitness}, f)

        if gen % 10 == 0:
            phase = "FAST" if gen < 100 else "WF"
            print(f"  Gen {gen:>3} [{phase}]: best={best_ever_fitness:.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s, fitness={best_ever_fitness:.4f}")

    # OOS evaluation
    oos_data = {}
    for pair in PAIR_PIP:
        path = DATA_DIR / f"{pair}_range10_asi_mc.parquet"
        if not path.exists(): continue
        df = pd.read_parquet(path)
        mc_d = df["mc_d"].values.astype(np.float64)
        mc_dd = df["mc_dd"].values.astype(np.float64)
        mid = df["mid_close"].values.astype(np.float64)
        n = len(mid); split = int(n * 0.7)
        oos_data[pair] = (mc_d[split:], mc_dd[split:], mid[split:])

    net = extract_network(best_ever, islands[0]["config"])
    print(f"\nOOS (range bars):")
    total_pnl = 0; total_trades = 0
    for pair, (mc_d, mc_dd, mid) in oos_data.items():
        pip = PAIR_PIP[pair]; spread = PAIR_SPREAD[pair]
        nt, pnl, mae = eval_range_chunk(mc_d, mc_dd, mid, pip, spread, 30,
            net[0], net[2], net[3], net[4], net[5], net[6], net[7], net[8], net[9], net[10], 0, len(mid))
        total_pnl += pnl; total_trades += nt
        print(f"  {pair:<10} {nt:>5}T {pnl:>+8.1f}p MAE={mae:.1f}p")
    print(f"\n  TOTAL: {total_trades}T {total_pnl:+.0f}p")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"range_neat_s{args.seed}_best.pkl", "wb") as f:
        pickle.dump({"genome": best_ever, "config": islands[0]["config"]}, f)
    with open(RESULTS_DIR / f"range_neat_s{args.seed}_result.json", "w") as f:
        json.dump({"gens": args.gens, "seed": args.seed, "fitness": round(float(best_ever_fitness), 4),
                    "oos_pnl": round(total_pnl, 1), "oos_trades": total_trades,
                    "elapsed_s": round(elapsed, 1)}, f, indent=2)

    tg_send(f"🏁 NEAT Range Bar DONE ({elapsed:.0f}s)\nFitness: {best_ever_fitness:.4f}\nOOS: {total_trades}T {total_pnl:+.0f}p")


if __name__ == "__main__":
    main()
