#!/usr/bin/env python3
"""
Per-Pair ASI-MC V3 Training with Zigzag Label Pretraining
==========================================================
Phase 1: Generate zigzag labels from M5 mid price (hindsight-optimal entry/exit)
Phase 2: Supervised pretrain — network learns to match zigzag labels (50 gens)
Phase 3: Full NEAT evolution on single pair P&L fitness (150 gens)
Phase 4: OOS evaluation + comparison with general model

4 inputs: MC(D), MC(dD), ER_norm, UPnL
3 outputs: BUY, SELL, FLATTEN

Usage:
  python3 train_zigzag_perpair.py --pair EUR_GBP --seed 42
  python3 train_zigzag_perpair.py --pair CAD_JPY --seed 42 --pretrain-gens 50 --gens 150
"""

import sys
import os
import gc
import time
import copy
import json
import pickle
import argparse
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

DATA_DIR = Path(os.environ.get("ASI_MC_DATA_DIR",
                str(PROJECT_ROOT / "data" / "asi_mc_indicators")))
RESULTS_DIR = SCRIPT_DIR / "results" / "zigzag_perpair"

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
ALL_PAIRS = list(PAIR_PIP.keys())

# Zigzag min_swing in pips — tuned per pair for ~2-5 swings/day on M5
PAIR_MIN_SWING = {
    "EUR_JPY": 30, "USD_JPY": 25, "GBP_JPY": 40, "AUD_JPY": 25,
    "CAD_JPY": 30, "CHF_JPY": 35, "NZD_JPY": 25,
    "EUR_USD": 20, "GBP_USD": 25, "AUD_USD": 18,
    "NZD_USD": 18, "EUR_GBP": 15,
}

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


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Zigzag Label Generation
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def generate_zigzag_labels(mid_close, pip, min_swing_pips, label_window=6, min_mfe_pips=3.0):
    """Generate BUY/SELL/FLATTEN labels from zigzag swing detection.

    Returns labels array: 0=FLATTEN, 1=BUY, 2=SELL for each bar.

    Logic:
    - Run zigzag: track running high/low, direction flips when swing >= min_swing
    - On direction flip UP (swing low confirmed): label next `label_window` bars as BUY
    - On direction flip DOWN (swing high confirmed): label next `label_window` bars as SELL
    - Bars near swing extremes (within flatten_proximity) → FLATTEN
    - Default → FLATTEN (no action)

    Also filters: only label swings where actual MFE > min_mfe_pips (profitable swings).
    """
    n = len(mid_close)
    labels = np.zeros(n, dtype=np.int64)  # 0=FLATTEN default
    min_swing = min_swing_pips * pip
    min_mfe = min_mfe_pips * pip

    # Zigzag state
    running_high = mid_close[0]
    running_low = mid_close[0]
    rh_bar = 0
    rl_bar = 0
    direction = 0  # 0=init, 1=up, -1=down

    for i in range(1, n):
        price = mid_close[i]

        if price > running_high:
            running_high = price
            rh_bar = i
        if price < running_low:
            running_low = price
            rl_bar = i

        if direction == 0:
            if running_high - price >= min_swing:
                direction = -1
                running_low = price
                rl_bar = i
            elif price - running_low >= min_swing:
                direction = 1
                running_high = price
                rh_bar = i
        elif direction == 1:
            if running_high - price >= min_swing:
                # Swing high confirmed — look ahead to check MFE for SHORT
                mfe = 0.0
                for j in range(i, min(i + 200, n)):
                    dd = running_high - mid_close[j]
                    if dd > mfe:
                        mfe = dd
                if mfe > min_mfe:
                    end = min(i + label_window, n)
                    for k in range(i, end):
                        labels[k] = 2  # SELL
                direction = -1
                running_low = price
                rl_bar = i
        else:  # direction == -1
            if price - running_low >= min_swing:
                # Swing low confirmed — look ahead to check MFE for LONG
                mfe = 0.0
                for j in range(i, min(i + 200, n)):
                    uu = mid_close[j] - running_low
                    if uu > mfe:
                        mfe = uu
                if mfe > min_mfe:
                    end = min(i + label_window, n)
                    for k in range(i, end):
                        labels[k] = 1  # BUY
                direction = 1
                running_high = price
                rh_bar = i

    return labels


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Supervised Pretrain Evaluator
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def evaluate_supervised_jit(
    inputs_2d, mid_close, labels,
    pip, spread_pips, max_bars,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices, start_offset,
):
    """Supervised fitness: reward network for matching zigzag labels.

    Fitness = correct_entries - wrong_direction_penalty + profitable_bonus
    """
    values = np.zeros(total_values)
    data_len = inputs_2d.shape[1]
    n_ind = inputs_2d.shape[0]
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 10, 10)

    correct = 0
    wrong_dir = 0
    profitable_correct = 0
    total_labeled = 0

    position = 0
    entry_price = 0.0

    for i in range(start_bar, end_bar):
        # Simulate UPnL for the last input
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position - spread_pips
        else:
            pnl_pips = 0.0

        # Set inputs
        for k in range(n_ind):
            values[k] = inputs_2d[k, i]
        values[n_ind] = np.tanh(pnl_pips / 20.0)

        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        out_buy = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_flat = values[output_indices[2]]

        # Determine network action
        if out_buy > out_sell and out_buy > out_flat:
            net_action = 1  # BUY
        elif out_sell > out_buy and out_sell > out_flat:
            net_action = 2  # SELL
        else:
            net_action = 0  # FLATTEN

        label = labels[i]
        if label > 0:  # Only score on labeled bars (BUY or SELL)
            total_labeled += 1
            if net_action == label:
                correct += 1
                # Bonus if this would have been profitable
                if label == 1:  # BUY label
                    future_pnl = 0.0
                    for j in range(i + 1, min(i + 50, end_bar)):
                        future_pnl = (mid_close[j] - mid_close[i]) / pip - spread_pips
                        if future_pnl > 3.0:
                            profitable_correct += 1
                            break
                elif label == 2:  # SELL label
                    future_pnl = 0.0
                    for j in range(i + 1, min(i + 50, end_bar)):
                        future_pnl = (mid_close[i] - mid_close[j]) / pip - spread_pips
                        if future_pnl > 3.0:
                            profitable_correct += 1
                            break
            elif net_action > 0 and net_action != label:
                wrong_dir += 1

        # Track simulated position for UPnL computation
        if net_action == 1 and position <= 0:
            position = 1
            entry_price = mid_close[i]
        elif net_action == 2 and position >= 0:
            position = -1
            entry_price = mid_close[i]
        elif net_action == 0 and position != 0:
            position = 0
            entry_price = 0.0

    if total_labeled < 10:
        return -10.0, 0, 0, 0, 0

    accuracy = correct / total_labeled
    wrong_rate = wrong_dir / total_labeled
    profit_rate = profitable_correct / max(correct, 1)

    # Fitness: accuracy + profitable bonus - wrong direction penalty
    fitness = accuracy * 10.0 + profit_rate * 5.0 - wrong_rate * 8.0

    return fitness, total_labeled, correct, wrong_dir, profitable_correct


class ZigzagSupervisedEvaluator:
    """NEAT evaluator for supervised zigzag label matching."""

    def __init__(self, inputs_2d, mid_close, labels, pip, spread, max_hold=200):
        self.inputs_2d = inputs_2d
        self.mid_close = mid_close
        self.labels = labels
        self.pip = pip
        self.spread = spread
        self.max_hold = max_hold

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return -10.0
        result = evaluate_supervised_jit(
            self.inputs_2d, self.mid_close, self.labels,
            self.pip, self.spread, len(self.mid_close),
            net[0], net[2], net[3], net[4], net[5], net[6],
            net[7], net[8], net[9], net[10], 0)
        return float(result[0])


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: P&L Evolution Evaluator (standard)
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def evaluate_gen_jit(
    inputs_2d, mid_close,
    pip, spread_pips, max_bars, max_hold,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices, start_offset,
):
    """Generalized evaluator: inputs_2d is (n_indicator_cols, n_bars).
    Last input slot is always UPnL (computed dynamically).
    """
    values = np.zeros(total_values)
    data_len = inputs_2d.shape[1]
    n_ind = inputs_2d.shape[0]
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 10, 10)
    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    trade_dirs = np.zeros(max_trades)
    trade_maes = np.zeros(max_trades)
    n_trades = 0
    position = 0
    entry_price = 0.0
    entry_bar = 0
    worst_ae = 0.0

    for i in range(start_bar, end_bar):
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position - spread_pips
            ae = -(mid_close[i] - entry_price) / pip * position + spread_pips
            if ae > worst_ae:
                worst_ae = ae
        else:
            pnl_pips = 0.0

        for k in range(n_ind):
            values[k] = inputs_2d[k, i]
        values[n_ind] = np.tanh(pnl_pips / 20.0)

        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)
        out_buy = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_flat = values[output_indices[2]]

        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
            if n_trades < max_trades:
                trade_pnls[n_trades] = pnl
                trade_dirs[n_trades] = position
                trade_maes[n_trades] = worst_ae
                n_trades += 1
            position = 0
            worst_ae = 0.0

        if position == 0:
            if out_buy > out_sell and out_buy > out_flat:
                position = 1; entry_price = mid_close[i]; entry_bar = i; worst_ae = 0.0
            elif out_sell > out_buy and out_sell > out_flat:
                position = -1; entry_price = mid_close[i]; entry_bar = i; worst_ae = 0.0
        else:
            close = False; new_pos = 0
            if out_flat > out_buy and out_flat > out_sell:
                close = True
            elif position == 1 and out_sell > out_buy and out_sell > out_flat:
                close = True; new_pos = -1
            elif position == -1 and out_buy > out_sell and out_buy > out_flat:
                close = True; new_pos = 1
            if close:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    trade_dirs[n_trades] = position
                    trade_maes[n_trades] = worst_ae
                    n_trades += 1
                position = new_pos
                entry_price = mid_close[i] if new_pos != 0 else 0.0
                entry_bar = i
                worst_ae = 0.0

    if position != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar-1] - entry_price) / pip * position - spread_pips
        if n_trades < max_trades:
            trade_pnls[n_trades] = pnl
            trade_dirs[n_trades] = position
            trade_maes[n_trades] = worst_ae
            n_trades += 1

    if n_trades < 1:
        return 0, 0.0, -10.0, 0.0, 0.0, 0, 0, 0.0
    pnls = trade_pnls[:n_trades]
    dirs = trade_dirs[:n_trades]
    maes = trade_maes[:n_trades]
    total_pnl = 0.0
    for j in range(n_trades):
        total_pnl += pnls[j]
    mean_pnl = total_pnl / n_trades
    var = 0.0
    for j in range(n_trades):
        var += (pnls[j] - mean_pnl)**2
    std = (var / n_trades)**0.5 if n_trades > 1 else 1.0
    sharpe = mean_pnl / std * (n_trades**0.5) if std > 0 else 0.0
    wins = 0
    for j in range(n_trades):
        if pnls[j] > 0:
            wins += 1
    wr = 100.0 * wins / n_trades
    nl = ns = 0
    for j in range(n_trades):
        if dirs[j] > 0:
            nl += 1
        else:
            ns += 1
    avg_mae = 0.0
    for j in range(n_trades):
        avg_mae += maes[j]
    avg_mae = avg_mae / n_trades if n_trades > 0 else 0.0

    return n_trades, total_pnl, sharpe, wr, mean_pnl, nl, ns, avg_mae


class PnLEvaluator:
    """Standard P&L fitness evaluator for single pair."""

    def __init__(self, inputs_2d_is, mid_is, pip, spread, max_hold=200):
        self.inputs_2d_is = inputs_2d_is
        self.mid_is = mid_is
        self.pip = pip
        self.spread = spread
        self.max_hold = max_hold

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return -10.0
        result = evaluate_gen_jit(
            self.inputs_2d_is, self.mid_is,
            self.pip, self.spread, len(self.mid_is), self.max_hold,
            net[0], net[2], net[3], net[4], net[5], net[6],
            net[7], net[8], net[9], net[10], 0)
        nt, pnl, sh, wr, mpnl, nl, ns, avg_mae = result
        if nt < 5:
            return -10.0
        # Fitness: Sharpe + trade frequency bonus + bidirectional penalty
        data_days = len(self.mid_is) / 288.0
        tpd = nt / max(data_days, 1)
        trade_bonus = min(0.1, tpd * 0.01)
        avg_sharpe = sh
        if nt > 0:
            dir_ratio = min(nl, ns) / max(nl + ns, 1)
            if dir_ratio < 0.2:
                avg_sharpe *= 0.5
        return avg_sharpe + trade_bonus


# ═══════════════════════════════════════════════════════════════════════════
# Island NEAT (from train_from_indicators.py)
# ═══════════════════════════════════════════════════════════════════════════

class _GlobalNodeIDFactory:
    """Centralized node ID counter shared across all islands."""
    def __init__(self, start: int):
        self._next = start

    def get_new_node_key(self, node_dict):
        while True:
            new_id = self._next
            self._next += 1
            if new_id not in node_dict:
                return new_id


def run_island_neat(config_path, evaluator, seed_genome, n_islands=4, pop_per_island=150,
                    generations=100, save_dir=None, label=""):
    islands = []
    best_ever = None
    best_ever_fitness = -999

    if seed_genome is not None and seed_genome.nodes:
        factory_start = max(seed_genome.nodes.keys()) + 1
    else:
        factory_start = 1
    global_node_factory = _GlobalNodeIDFactory(factory_start)

    for i in range(n_islands):
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          str(config_path))
        cfg.pop_size = pop_per_island
        cfg.genome_config.get_new_node_key = global_node_factory.get_new_node_key
        pop = neat.Population(cfg)
        if seed_genome is not None:
            for gid in list(pop.population.keys()):
                clone = copy.deepcopy(seed_genome)
                clone.key = gid
                clone.fitness = None
                try:
                    clone.mutate(cfg.genome_config)
                except (AssertionError, Exception):
                    pass
                pop.population[gid] = clone
        islands.append({"pop": pop, "config": cfg, "best": None})

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for gen in range(generations):
        for i, island in enumerate(islands):
            pop = island["pop"]
            cfg = island["config"]
            evaluator.evaluate(list(pop.population.items()), cfg)
            best = max(pop.population.values(),
                       key=lambda g: g.fitness if g.fitness is not None else -999)
            island["best"] = best
            if best.fitness is not None and best.fitness > best_ever_fitness:
                best_ever = copy.deepcopy(best)
                best_ever_fitness = best.fitness
            pop.species.speciate(cfg, pop.population, pop.generation)
            try:
                pop.population = pop.reproduction.reproduce(cfg, pop.species, cfg.pop_size, pop.generation)
            except (AssertionError, Exception):
                pass
            pop.generation += 1

        # Migration every 10 gens
        if gen > 0 and gen % 10 == 0:
            for i in range(n_islands):
                src = islands[i]
                dst = islands[(i + 1) % n_islands]
                if src["best"] is not None:
                    worst_gid = min(dst["pop"].population,
                                    key=lambda g: dst["pop"].population[g].fitness
                                    if dst["pop"].population[g].fitness is not None else 999)
                    migrant = copy.deepcopy(src["best"])
                    migrant.key = worst_gid
                    migrant.fitness = None
                    dst["pop"].population[worst_gid] = migrant

        # Save checkpoint every generation
        if save_dir:
            gen_best = max((isl["best"] for isl in islands if isl["best"] is not None),
                           key=lambda g: g.fitness if g.fitness is not None else -999,
                           default=None)
            if gen_best:
                with open(f"{save_dir}/gen_{gen:03d}_best.pkl", "wb") as f:
                    pickle.dump({"genome": copy.deepcopy(gen_best),
                                 "config": islands[0]["config"],
                                 "generation": gen,
                                 "fitness": gen_best.fitness}, f)

        fitnesses = [isl["best"].fitness for isl in islands
                     if isl["best"] is not None and isl["best"].fitness is not None]
        if fitnesses and gen % 10 == 0:
            print(f"  [{label}] Gen {gen:>3}: best={max(fitnesses):.4f} global={best_ever_fitness:.4f}")

    return best_ever, islands[0]["config"]


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Per-pair V3 training with zigzag label pretraining")
    parser.add_argument("--pair", required=True, help="Pair to train (e.g., EUR_GBP)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-gens", type=int, default=50)
    parser.add_argument("--gens", type=int, default=150)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--pop", type=int, default=150)
    parser.add_argument("--max-hold", type=int, default=200)
    parser.add_argument("--min-swing", type=int, default=0,
                        help="Zigzag min swing in pips (0=auto per pair)")
    parser.add_argument("--label-window", type=int, default=6,
                        help="Bars to label after zigzag confirmation")
    parser.add_argument("--genome", default="", help="Seed genome (skips zigzag pretrain)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]
    min_swing_pips = args.min_swing if args.min_swing > 0 else PAIR_MIN_SWING.get(pair, 20)

    print(f"{'='*60}")
    print(f"  Zigzag Per-Pair V3 Training: {pair}")
    print(f"  Seed: {args.seed} | {args.islands}×{args.pop} pop")
    print(f"  Pretrain: {args.pretrain_gens}g (zigzag) → Evolve: {args.gens}g (P&L)")
    print(f"  Min swing: {min_swing_pips}p | Label window: {args.label_window} bars")
    print(f"{'='*60}")
    tg_send(f"🎯 Zigzag V3 {pair} s{args.seed}\n"
            f"Pretrain {args.pretrain_gens}g + Evolve {args.gens}g\n"
            f"Min swing: {min_swing_pips}p")

    # Load data
    path = DATA_DIR / f"{pair}_asi_mc.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found")
        return
    df = pd.read_parquet(path, engine="pyarrow")
    mid = df["mid_close"].values.astype(np.float64)
    n = len(mid)
    split = int(n * 0.7)

    ind_cols = ["mc_d_a", "mc_dd_a", "er_norm"]
    for c in ind_cols:
        if c not in df.columns:
            print(f"ERROR: column {c} not in parquet")
            return

    inputs = np.stack([df[c].values.astype(np.float64) for c in ind_cols], axis=0)
    inputs_is = inputs[:, :split]
    mid_is = mid[:split]
    inputs_oos = inputs[:, split:]
    mid_oos = mid[split:]
    del df; gc.collect()

    print(f"\nData: {n:,} total M5 bars | IS: {split:,} | OOS: {n-split:,}")

    # ── Phase 1: Generate zigzag labels ───────────────────────────────
    print(f"\nPhase 1: Generating zigzag labels (min_swing={min_swing_pips}p)...")
    t0 = time.time()
    labels_is = generate_zigzag_labels(mid_is, pip, min_swing_pips,
                                        label_window=args.label_window,
                                        min_mfe_pips=spread + 2.0)
    n_buy = np.sum(labels_is == 1)
    n_sell = np.sum(labels_is == 2)
    n_flat = np.sum(labels_is == 0)
    n_labeled = n_buy + n_sell
    pct = 100.0 * n_labeled / len(labels_is)
    print(f"  Labels: BUY={n_buy:,} SELL={n_sell:,} FLAT={n_flat:,}")
    print(f"  Labeled bars: {n_labeled:,}/{len(labels_is):,} ({pct:.1f}%)")
    print(f"  Label generation: {time.time()-t0:.1f}s")

    if n_labeled < 100:
        print(f"  WARNING: Very few labels ({n_labeled}). Reducing min_swing...")
        reduced = max(min_swing_pips // 2, 5)
        labels_is = generate_zigzag_labels(mid_is, pip, reduced,
                                            label_window=args.label_window,
                                            min_mfe_pips=spread + 1.0)
        n_buy = np.sum(labels_is == 1)
        n_sell = np.sum(labels_is == 2)
        n_labeled = n_buy + n_sell
        print(f"  After reduction ({reduced}p): BUY={n_buy:,} SELL={n_sell:,} total={n_labeled:,}")
        min_swing_pips = reduced

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"zz_v3_{pair}_s{args.seed}"
    config_path = SCRIPT_DIR / "neat_config_4in_3out.ini"

    # ── Phase 2: Supervised pretrain on zigzag labels ─────────────────
    seed_genome = None
    if args.genome and os.path.exists(args.genome):
        with open(args.genome, "rb") as f:
            d = pickle.load(f)
        seed_genome = d["genome"]
        print(f"\nSkipping pretrain — using seed genome: {Path(args.genome).name}")
    else:
        print(f"\nPhase 2: Supervised pretrain ({args.pretrain_gens} gens)...")
        zz_eval = ZigzagSupervisedEvaluator(inputs_is, mid_is, labels_is, pip, spread)
        save_dir_pt = str(RESULTS_DIR / f"{tag}_pretrain_ckpt")

        t0 = time.time()
        seed_genome, _ = run_island_neat(
            config_path, zz_eval, None,
            n_islands=args.islands, pop_per_island=args.pop,
            generations=args.pretrain_gens,
            save_dir=save_dir_pt, label=f"{pair} pretrain")
        pt_elapsed = time.time() - t0
        print(f"  Pretrain done: fitness={seed_genome.fitness:.4f} ({pt_elapsed:.0f}s)")
        tg_send(f"✅ {pair} pretrain done: fit={seed_genome.fitness:.4f} ({pt_elapsed:.0f}s)")

        # Save pretrained genome
        with open(RESULTS_DIR / f"{tag}_pretrained.pkl", "wb") as f:
            pickle.dump({"genome": seed_genome, "pair": pair,
                         "pretrain_fitness": seed_genome.fitness,
                         "min_swing_pips": min_swing_pips,
                         "labels_buy": int(n_buy), "labels_sell": int(n_sell)}, f)

    # ── Phase 3: Full P&L evolution ───────────────────────────────────
    print(f"\nPhase 3: P&L evolution ({args.gens} gens)...")
    pnl_eval = PnLEvaluator(inputs_is, mid_is, pip, spread, max_hold=args.max_hold)
    save_dir_ev = str(RESULTS_DIR / f"{tag}_evolve_ckpt")

    t0 = time.time()
    winner, config = run_island_neat(
        config_path, pnl_eval, seed_genome,
        n_islands=args.islands, pop_per_island=args.pop,
        generations=args.gens,
        save_dir=save_dir_ev, label=f"{pair} evolve")
    ev_elapsed = time.time() - t0
    print(f"  Evolution done: fitness={winner.fitness:.4f} ({ev_elapsed:.0f}s)")

    # ── Phase 4: OOS evaluation ───────────────────────────────────────
    print(f"\nPhase 4: OOS evaluation...")
    net = extract_network(winner, config)
    oos_result = evaluate_gen_jit(
        inputs_oos, mid_oos,
        pip, spread, len(mid_oos), args.max_hold,
        net[0], net[2], net[3], net[4], net[5], net[6],
        net[7], net[8], net[9], net[10], 0)
    nt, pnl, sh, wr, mpnl, nl, ns, avg_mae = oos_result

    # Also eval IS for comparison
    is_result = evaluate_gen_jit(
        inputs_is, mid_is,
        pip, spread, len(mid_is), args.max_hold,
        net[0], net[2], net[3], net[4], net[5], net[6],
        net[7], net[8], net[9], net[10], 0)
    is_nt, is_pnl, is_sh, is_wr, is_mpnl, is_nl, is_ns, is_mae = is_result

    oos_days = (n - split) / 288.0
    is_days = split / 288.0

    print(f"\n{'='*60}")
    print(f"  RESULTS: {pair} (Zigzag V3 Per-Pair)")
    print(f"{'='*60}")
    print(f"  IS:  {int(is_nt):>5}T {float(is_pnl):>+9.1f}p Sh={float(is_sh):>6.2f} "
          f"WR={float(is_wr):>4.0f}% L={int(is_nl)} S={int(is_ns)} MAE={float(is_mae):.1f}p "
          f"({float(is_pnl)/is_days:.1f}p/day)")
    print(f"  OOS: {int(nt):>5}T {float(pnl):>+9.1f}p Sh={float(sh):>6.2f} "
          f"WR={float(wr):>4.0f}% L={int(nl)} S={int(ns)} MAE={float(avg_mae):.1f}p "
          f"({float(pnl)/oos_days:.1f}p/day)")
    print(f"  Fitness: {winner.fitness:.4f} | Size: {winner.size()}")

    result_data = {
        "pair": pair,
        "seed": args.seed,
        "pretrain_gens": args.pretrain_gens,
        "evolution_gens": args.gens,
        "min_swing_pips": min_swing_pips,
        "label_window": args.label_window,
        "labels_buy": int(n_buy),
        "labels_sell": int(n_sell),
        "fitness": round(float(winner.fitness), 4),
        "network_size": list(winner.size()),
        "is": {
            "n_trades": int(is_nt), "total_pnl": round(float(is_pnl), 1),
            "sharpe": round(float(is_sh), 4), "win_rate": round(float(is_wr), 1),
            "n_long": int(is_nl), "n_short": int(is_ns), "avg_mae": round(float(is_mae), 2),
            "pips_per_day": round(float(is_pnl) / is_days, 1),
        },
        "oos": {
            "n_trades": int(nt), "total_pnl": round(float(pnl), 1),
            "sharpe": round(float(sh), 4), "win_rate": round(float(wr), 1),
            "n_long": int(nl), "n_short": int(ns), "avg_mae": round(float(avg_mae), 2),
            "pips_per_day": round(float(pnl) / oos_days, 1),
        },
    }

    # Save
    with open(RESULTS_DIR / f"{tag}_best.pkl", "wb") as f:
        pickle.dump({"genome": winner, "config": config, "pair": pair,
                     "result": result_data}, f)
    with open(RESULTS_DIR / f"{tag}_result.json", "w") as f:
        json.dump(result_data, f, indent=2)

    tg_send(f"🏁 Zigzag V3 {pair} s{args.seed}\n"
            f"Fitness: {winner.fitness:.4f}\n"
            f"OOS: {int(nt)}T {float(pnl):+.1f}p Sh={float(sh):.2f} WR={float(wr):.0f}%\n"
            f"({float(pnl)/oos_days:.1f}p/day)")

    print(f"\nSaved: {tag}_best.pkl + {tag}_result.json")
    print(f"Total time: {(time.time()-t0+pt_elapsed if not args.genome else ev_elapsed):.0f}s")


if __name__ == "__main__":
    main()
