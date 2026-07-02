#!/usr/bin/env python3
"""
Train NEAT from pre-computed indicators.
=========================================
Reads exported indicator parquets (from export_indicators.py / export_d_indicators.py).
Zero indicator computation during training — just network evolution.

Supports:
  --variant A/B/C02/C05/C10  (ASI-MC variants, 3 inputs: mc_d, mc_dd, UPnL)
  --mode d1..d6              (D-experiment combos, 3-4 inputs from parquet columns)

Usage:
  python3 train_from_indicators.py --variant A --seed 42
  python3 train_from_indicators.py --mode d1 --seed 42
  python3 train_from_indicators.py --mode d6 --seed 42 --pairs EUR_JPY,USD_JPY
"""

import sys
import os
import gc
import time
import json
import copy
import pickle
import math
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
RESULTS_DIR = SCRIPT_DIR / "results"

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


# ── JIT evaluator (same as before) ────────────────────────────────────────

@njit(cache=True)
def evaluate_3out_jit(
    mc_d, mc_dd, mid_close,
    pip, spread_pips, max_bars, max_hold,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices, start_offset,
):
    values = np.zeros(total_values)
    data_len = len(mid_close)
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 10, 10)
    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    trade_dirs = np.zeros(max_trades)
    n_trades = 0
    position = 0
    entry_price = 0.0
    entry_bar = 0

    for i in range(start_bar, end_bar):
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position - spread_pips
        else:
            pnl_pips = 0.0
        values[0] = mc_d[i]
        values[1] = mc_dd[i]
        values[2] = np.tanh(pnl_pips / 20.0)
        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)
        out_buy = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_flat = values[output_indices[2]]

        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
            if n_trades < max_trades:
                trade_pnls[n_trades] = pnl; trade_dirs[n_trades] = position; n_trades += 1
            position = 0

        if position == 0:
            if out_buy > out_sell and out_buy > out_flat:
                position = 1; entry_price = mid_close[i]; entry_bar = i
            elif out_sell > out_buy and out_sell > out_flat:
                position = -1; entry_price = mid_close[i]; entry_bar = i
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
                    trade_pnls[n_trades] = pnl; trade_dirs[n_trades] = position; n_trades += 1
                position = new_pos
                entry_price = mid_close[i] if new_pos != 0 else 0.0
                entry_bar = i

    if position != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar-1] - entry_price) / pip * position - spread_pips
        if n_trades < max_trades:
            trade_pnls[n_trades] = pnl; trade_dirs[n_trades] = position; n_trades += 1

    if n_trades < 1:
        return 0, 0.0, -10.0, 0.0, 0.0, 0, 0
    pnls = trade_pnls[:n_trades]; dirs = trade_dirs[:n_trades]
    total_pnl = 0.0
    for j in range(n_trades): total_pnl += pnls[j]
    mean_pnl = total_pnl / n_trades
    var = 0.0
    for j in range(n_trades): var += (pnls[j] - mean_pnl)**2
    std = (var / n_trades)**0.5 if n_trades > 1 else 1.0
    sharpe = mean_pnl / std * (n_trades**0.5) if std > 0 else 0.0
    wins = 0
    for j in range(n_trades):
        if pnls[j] > 0: wins += 1
    wr = 100.0 * wins / n_trades
    nl = ns = 0
    for j in range(n_trades):
        if dirs[j] > 0: nl += 1
        else: ns += 1
    return n_trades, total_pnl, sharpe, wr, mean_pnl, nl, ns


# ── Generalized JIT evaluator (N inputs) ──────────────────────────────────

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
    n_ind = inputs_2d.shape[0]  # number of indicator columns (excl UPnL)
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 10, 10)
    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    trade_dirs = np.zeros(max_trades)
    n_trades = 0
    position = 0
    entry_price = 0.0
    entry_bar = 0

    for i in range(start_bar, end_bar):
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position - spread_pips
        else:
            pnl_pips = 0.0

        # Set indicator inputs
        for k in range(n_ind):
            values[k] = inputs_2d[k, i]
        # Last input = UPnL
        values[n_ind] = np.tanh(pnl_pips / 20.0)

        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)
        out_buy = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_flat = values[output_indices[2]]

        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
            if n_trades < max_trades:
                trade_pnls[n_trades] = pnl; trade_dirs[n_trades] = position; n_trades += 1
            position = 0

        if position == 0:
            if out_buy > out_sell and out_buy > out_flat:
                position = 1; entry_price = mid_close[i]; entry_bar = i
            elif out_sell > out_buy and out_sell > out_flat:
                position = -1; entry_price = mid_close[i]; entry_bar = i
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
                    trade_pnls[n_trades] = pnl; trade_dirs[n_trades] = position; n_trades += 1
                position = new_pos
                entry_price = mid_close[i] if new_pos != 0 else 0.0
                entry_bar = i

    if position != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar-1] - entry_price) / pip * position - spread_pips
        if n_trades < max_trades:
            trade_pnls[n_trades] = pnl; trade_dirs[n_trades] = position; n_trades += 1

    if n_trades < 1:
        return 0, 0.0, -10.0, 0.0, 0.0, 0, 0
    pnls = trade_pnls[:n_trades]; dirs = trade_dirs[:n_trades]
    total_pnl = 0.0
    for j in range(n_trades): total_pnl += pnls[j]
    mean_pnl = total_pnl / n_trades
    var = 0.0
    for j in range(n_trades): var += (pnls[j] - mean_pnl)**2
    std = (var / n_trades)**0.5 if n_trades > 1 else 1.0
    sharpe = mean_pnl / std * (n_trades**0.5) if std > 0 else 0.0
    wins = 0
    for j in range(n_trades):
        if pnls[j] > 0: wins += 1
    wr = 100.0 * wins / n_trades
    nl = ns = 0
    for j in range(n_trades):
        if dirs[j] > 0: nl += 1
        else: ns += 1
    return n_trades, total_pnl, sharpe, wr, mean_pnl, nl, ns


# ── D-experiment mode definitions ─────────────────────────────────────────

D_MODES = {
    # mode: (indicator_columns, n_inputs_total)
    # n_inputs_total = len(indicator_columns) + 1 (UPnL)
    "d1": (["sign_mc_d", "h1_sr_zone"], 3),
    "d2": (["sign_mc_d", "adx_regime"], 3),
    "d3": (["sign_mc_d", "str_diff_sign"], 3),
    "d4": (["sign_mc_d", "vol_regime"], 3),
    "d5": (["h1_sr_zone", "str_diff_sign"], 3),
    "d6": (["sign_mc_d", "sign_mc_dd", "h1_sr_zone"], 4),  # 4 inputs
    # V3: full-precision ASI-MC + ER regime filter (4 inputs)
    # MC(D) + MC(dD) + ER_norm + UPnL
    # ER_norm = Kaufman ER(60) arctan-normalized → 0=chop, 0.82=perfect trend
    # Network learns WHEN to trade (ER) + WHERE (MC direction) jointly
    "v3": (["mc_d_a", "mc_dd_a", "er_norm"], 4),
}


# ── Generalized Evaluator ─────────────────────────────────────────────────

class GenEvaluator:
    """Evaluator for D-experiment modes with N indicator inputs."""

    def __init__(self, pair_data, n_indicators, max_hold=200):
        """pair_data: {pair: (inputs_2d_is, mid_is, inputs_2d_oos, mid_oos)}
        inputs_2d: np.array shape (n_indicators, n_bars)
        """
        self.pair_data = pair_data
        self.n_indicators = n_indicators
        self.max_hold = max_hold

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return -10.0
        total_sharpe = 0.0; total_trades = 0; total_bars = 0
        total_longs = total_shorts = n_pairs = 0
        for pair, (inp_is, mid_is, inp_oos, mid_oos) in self.pair_data.items():
            pip = PAIR_PIP.get(pair, 0.01)
            spread = PAIR_SPREAD.get(pair, 2.0)
            result = evaluate_gen_jit(
                inp_is, mid_is, pip, spread, len(mid_is), self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10], 0)
            nt, pnl, sh, wr, mpnl, nl, ns = result
            if nt >= 5:
                total_sharpe += sh; n_pairs += 1
            total_trades += nt; total_bars += len(mid_is)
            total_longs += nl; total_shorts += ns
        if n_pairs == 0 or total_trades < 10:
            return -10.0
        avg_sharpe = total_sharpe / n_pairs
        data_days = total_bars / 288.0
        trades_per_day = total_trades / max(data_days, 1)
        trade_bonus = min(0.1, trades_per_day * 0.01)
        if total_trades > 0:
            dir_ratio = min(total_longs, total_shorts) / max(total_longs + total_shorts, 1)
            if dir_ratio < 0.2:
                avg_sharpe *= 0.5
        return avg_sharpe + trade_bonus

    def eval_oos(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return {}
        results = {}
        for pair, (inp_is, mid_is, inp_oos, mid_oos) in self.pair_data.items():
            pip = PAIR_PIP.get(pair, 0.01)
            spread = PAIR_SPREAD.get(pair, 2.0)
            result = evaluate_gen_jit(
                inp_oos, mid_oos, pip, spread, len(mid_oos), self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10], 0)
            nt, pnl, sh, wr, mpnl, nl, ns = result
            results[pair] = {
                "n_trades": int(nt), "total_pnl": round(float(pnl), 1),
                "sharpe": round(float(sh), 4), "win_rate": round(float(wr), 1),
                "n_long": int(nl), "n_short": int(ns),
            }
        return results


# ── Evaluator ──────────────────────────────────────────────────────────────

class IndicatorEvaluator:
    def __init__(self, pair_data, max_hold=200):
        """pair_data: {pair: (mc_d_is, mc_dd_is, mid_is, mc_d_oos, mc_dd_oos, mid_oos)}"""
        self.pair_data = pair_data
        self.max_hold = max_hold

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return -10.0
        total_sharpe = 0.0; total_trades = 0; total_bars = 0
        total_longs = total_shorts = n_pairs = 0
        for pair, (mc_d_is, mc_dd_is, mid_is, *_) in self.pair_data.items():
            pip = PAIR_PIP.get(pair, 0.01)
            spread = PAIR_SPREAD.get(pair, 2.0)
            result = evaluate_3out_jit(
                mc_d_is, mc_dd_is, mid_is, pip, spread, len(mid_is), self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10], 0)
            nt, pnl, sh, wr, mpnl, nl, ns = result
            if nt >= 5:
                total_sharpe += sh; n_pairs += 1
            total_trades += nt; total_bars += len(mid_is)
            total_longs += nl; total_shorts += ns
        if n_pairs == 0 or total_trades < 10:
            return -10.0
        avg_sharpe = total_sharpe / n_pairs
        data_days = total_bars / 288.0
        trades_per_day = total_trades / max(data_days, 1)
        trade_bonus = min(0.1, trades_per_day * 0.01)
        if total_trades > 0:
            dir_ratio = min(total_longs, total_shorts) / max(total_longs + total_shorts, 1)
            if dir_ratio < 0.2:
                avg_sharpe *= 0.5
        return avg_sharpe + trade_bonus

    def eval_oos(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return {}
        results = {}
        for pair, data in self.pair_data.items():
            mc_d_oos, mc_dd_oos, mid_oos = data[3], data[4], data[5]
            pip = PAIR_PIP.get(pair, 0.01)
            spread = PAIR_SPREAD.get(pair, 2.0)
            result = evaluate_3out_jit(
                mc_d_oos, mc_dd_oos, mid_oos, pip, spread, len(mid_oos), self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10], 0)
            nt, pnl, sh, wr, mpnl, nl, ns = result
            results[pair] = {
                "n_trades": int(nt), "total_pnl": round(float(pnl), 1),
                "sharpe": round(float(sh), 4), "win_rate": round(float(wr), 1),
                "n_long": int(nl), "n_short": int(ns),
            }
        return results


# ── Island NEAT ────────────────────────────────────────────────────────────

class _GlobalNodeIDFactory:
    """Centralized node ID counter shared across all islands.

    Each island has its own neat.Config (and thus its own node_indexer), which
    means they independently assign node IDs starting from the same seed max.
    When a genome migrates between islands the receiving island's counter may
    already have issued that same ID — causing neat-python's assert to fire.

    Fix: replace every island's genome_config.get_new_node_key with a single
    shared method backed by one monotonically-increasing counter.  All node IDs
    are globally unique, so migration is always safe.

    Plain integer counter (no threading.Lock) — training is single-threaded
    and this object must survive pickle for checkpoint saving.
    """
    def __init__(self, start: int):
        self._next = start

    def get_new_node_key(self, node_dict):
        while True:
            new_id = self._next
            self._next += 1
            if new_id not in node_dict:
                return new_id


def run_island_neat(config, evaluator, seed_genome, n_islands=4, pop_per_island=150,
                    generations=100, save_dir=None):
    islands = []
    best_ever = None; best_ever_fitness = -999

    # Seed the global factory above the highest node ID already in the genome
    if seed_genome is not None and seed_genome.nodes:
        factory_start = max(seed_genome.nodes.keys()) + 1
    else:
        factory_start = 1
    global_node_factory = _GlobalNodeIDFactory(factory_start)

    for i in range(n_islands):
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          str(config))
        cfg.pop_size = pop_per_island
        # Patch this island's genome_config to use the shared factory
        cfg.genome_config.get_new_node_key = global_node_factory.get_new_node_key
        pop = neat.Population(cfg)
        if seed_genome is not None:
            for gid in list(pop.population.keys()):
                clone = copy.deepcopy(seed_genome)
                clone.key = gid; clone.fitness = None
                try:
                    clone.mutate(cfg.genome_config)
                except AssertionError:
                    pass  # factory collision guard (belt-and-suspenders)
                pop.population[gid] = clone
        islands.append({"pop": pop, "config": cfg, "best": None})

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for gen in range(generations):
        for i, island in enumerate(islands):
            pop = island["pop"]; cfg = island["config"]
            evaluator.evaluate(list(pop.population.items()), cfg)
            best = max(pop.population.values(), key=lambda g: g.fitness if g.fitness is not None else -999)
            island["best"] = best
            if best.fitness is not None and best.fitness > best_ever_fitness:
                best_ever = copy.deepcopy(best); best_ever_fitness = best.fitness
            pop.species.speciate(cfg, pop.population, pop.generation)
            try:
                pop.population = pop.reproduction.reproduce(cfg, pop.species, cfg.pop_size, pop.generation)
            except AssertionError:
                pass  # node ID collision guard — skip this generation's reproduce, keep existing population
            pop.generation += 1

        # Migration every 10 gens
        if gen > 0 and gen % 10 == 0:
            for i in range(n_islands):
                src = islands[i]; dst = islands[(i+1) % n_islands]
                if src["best"] is not None:
                    worst_gid = min(dst["pop"].population,
                                    key=lambda g: dst["pop"].population[g].fitness
                                    if dst["pop"].population[g].fitness is not None else 999)
                    migrant = copy.deepcopy(src["best"])
                    migrant.key = worst_gid; migrant.fitness = None
                    dst["pop"].population[worst_gid] = migrant

        # Save checkpoint
        if save_dir:
            gen_best = max((isl["best"] for isl in islands if isl["best"] is not None),
                           key=lambda g: g.fitness if g.fitness is not None else -999, default=None)
            if gen_best:
                with open(f"{save_dir}/gen_{gen:03d}_best.pkl", "wb") as f:
                    pickle.dump({"genome": copy.deepcopy(gen_best), "config": islands[0]["config"],
                                 "generation": gen, "fitness": gen_best.fitness}, f)

        fitnesses = [isl["best"].fitness for isl in islands
                     if isl["best"] is not None and isl["best"].fitness is not None]
        if fitnesses and gen % 10 == 0:
            print(f"  Gen {gen:>3}: best={max(fitnesses):.4f} global={best_ever_fitness:.4f}")

    return best_ever, islands[0]["config"]


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["A", "B", "C02", "C05", "C10"],
                        help="ASI-MC variant (3 inputs: mc_d, mc_dd, UPnL)")
    parser.add_argument("--mode", choices=list(D_MODES.keys()),
                        help="D-experiment mode (d1-d6=quantized combos, v3=ASI-MC+ER full-precision)")
    parser.add_argument("--pairs", default="EUR_JPY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--pop", type=int, default=150)
    parser.add_argument("--genome", default="", help="Seed genome path (empty=from scratch with sine pretrain)")
    parser.add_argument("--max-hold", type=int, default=200)
    parser.add_argument("--per-pair", action="store_true",
                        help="Train a separate genome for each pair (loops over all 12)")
    args = parser.parse_args()

    if not args.variant and not args.mode:
        parser.error("Either --variant or --mode is required")

    if args.per_pair:
        _run_per_pair(args)
    else:
        _run_single(args)


def _load_all_data(args):
    """Load indicator data for all available pairs. Returns (d_all_data, all_data)."""
    use_d_mode = args.mode is not None
    variant = args.variant or args.mode
    d_all_data = {}
    all_data = {}

    for pair in ALL_PAIRS:
        path = DATA_DIR / f"{pair}_asi_mc.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, engine="pyarrow")
        mid = df["mid_close"].values.astype(np.float64)
        n = len(mid)
        split = int(n * 0.7)

        if use_d_mode:
            ind_cols, n_inputs = D_MODES[args.mode]
            missing = [c for c in ind_cols if c not in df.columns]
            if missing:
                print(f"  {pair}: MISSING columns {missing}")
                continue
            inputs = np.stack([df[c].values.astype(np.float64) for c in ind_cols], axis=0)
            d_all_data[pair] = (inputs[:, :split], mid[:split], inputs[:, split:], mid[split:])
        else:
            mc_d_col = f"mc_d_{variant.lower()}"
            mc_dd_col = f"mc_dd_{variant.lower()}"
            mc_d = df[mc_d_col].values.astype(np.float64)
            mc_dd = df[mc_dd_col].values.astype(np.float64)
            all_data[pair] = (mc_d[:split], mc_dd[:split], mid[:split],
                              mc_d[split:], mc_dd[split:], mid[split:])
        del df; gc.collect()

    return d_all_data, all_data


def _get_config_path(args):
    """Return NEAT config path and n_inputs for the given mode/variant."""
    use_d_mode = args.mode is not None
    if use_d_mode:
        _, n_inputs = D_MODES[args.mode]
        if n_inputs == 4:
            config_path = SCRIPT_DIR / "neat_config_4in_3out.ini"
        else:
            config_path = SCRIPT_DIR / "neat_config_3out.ini"
    else:
        config_path = SCRIPT_DIR / "neat_config_3out.ini"
        n_inputs = 3

    if n_inputs == 4 and not config_path.exists():
        src = SCRIPT_DIR / "neat_config_3out.ini"
        with open(src) as f:
            cfg_text = f.read()
        cfg_text = cfg_text.replace("num_inputs              = 3",
                                    "num_inputs              = 4")
        with open(config_path, "w") as f:
            f.write(cfg_text)
    return config_path, n_inputs


def _run_per_pair(args):
    """Train a separate genome for each pair."""
    use_d_mode = args.mode is not None
    variant = args.variant or args.mode
    label = f"D{args.mode.upper()}" if use_d_mode else f"V{variant}"

    print(f"{'='*60}")
    print(f"  PER-PAIR Training — {label}")
    print(f"  Seed: {args.seed} | {args.islands}×{args.pop} pop, {args.gens} gens")
    print(f"{'='*60}")
    tg_send(f"🏝 PER-PAIR {label} s{args.seed}\n{args.islands}×{args.pop} pop, {args.gens}g\n12 pairs")

    print("\nLoading indicator data...")
    d_all_data, all_data = _load_all_data(args)
    config_path, n_inputs = _get_config_path(args)

    # Seed genome (sine pretrain for ASI-MC variants only)
    base_seed_genome = None
    if args.genome and os.path.exists(args.genome):
        with open(args.genome, "rb") as f:
            d = pickle.load(f)
        base_seed_genome = d["genome"]
        print(f"  Seed genome: {Path(args.genome).name}")
    elif not use_d_mode:
        print("\nSine pretrain (50 gens)...")
        from train_pretrain_continue import generate_sine_ohlc, AsiMcEvaluator
        from lib.asi_indicator import compute_asi_mc
        o_s, h_s, l_s, c_s, mid_s = generate_sine_ohlc(50000, 500, 10)
        n_s = len(mid_s)
        mc_d_s, mc_dd_s = compute_asi_mc(o_s, h_s, l_s, c_s, n_s)
        sp = int(n_s * 0.7)
        sine_data = {"sine": (mc_d_s[:sp], mc_dd_s[:sp], mid_s[:sp],
                              mc_d_s[sp:], mc_dd_s[sp:], mid_s[sp:])}
        sine_eval = AsiMcEvaluator(sine_data, pip=0.0001, spread=2.0, max_hold=500)
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path))
        cfg.pop_size = args.pop
        pop = neat.Population(cfg)
        pop.run(sine_eval.evaluate, 50)
        base_seed_genome = max(pop.population.values(), key=lambda g: g.fitness if g.fitness else -999)
        print(f"  Sine pretrain done: fitness={base_seed_genome.fitness:.4f}")

    data_source = d_all_data if use_d_mode else all_data
    available_pairs = sorted(data_source.keys())
    print(f"\nTraining {len(available_pairs)} pairs: {available_pairs}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}" if use_d_mode else f"v{variant}"
    per_pair_results = {}
    t0_all = time.time()

    for pair in available_pairs:
        print(f"\n{'─'*50}")
        print(f"  Training {pair}...")
        print(f"{'─'*50}")

        # Single-pair training data
        if use_d_mode:
            ind_cols, _ = D_MODES[args.mode]
            evaluator = GenEvaluator({pair: d_all_data[pair]}, n_indicators=len(ind_cols),
                                     max_hold=args.max_hold)
        else:
            evaluator = IndicatorEvaluator({pair: all_data[pair]}, max_hold=args.max_hold)

        seed = copy.deepcopy(base_seed_genome) if base_seed_genome else None
        save_dir = str(RESULTS_DIR / f"{tag}_perpair_{pair}_s{args.seed}_ckpt")

        t0 = time.time()
        winner, config = run_island_neat(config_path, evaluator, seed,
                                          args.islands, args.pop, args.gens, save_dir)
        elapsed = time.time() - t0

        # OOS eval on this pair only
        if use_d_mode:
            oos_eval = GenEvaluator({pair: d_all_data[pair]}, n_indicators=len(ind_cols),
                                    max_hold=args.max_hold)
        else:
            oos_eval = IndicatorEvaluator({pair: all_data[pair]}, max_hold=args.max_hold)
        oos = oos_eval.eval_oos(winner, config)
        r = oos.get(pair, {})

        per_pair_results[pair] = {
            "fitness": round(float(winner.fitness), 4),
            "network_size": list(winner.size()),
            "elapsed_s": round(elapsed, 1),
            "oos": r,
        }

        # Save genome
        genome_file = f"{tag}_perpair_{pair}_s{args.seed}_best.pkl"
        with open(RESULTS_DIR / genome_file, "wb") as f:
            pickle.dump({"genome": winner, "config": config, "pair": pair}, f)

        pnl = r.get("total_pnl", 0)
        nt = r.get("n_trades", 0)
        sh = r.get("sharpe", 0)
        print(f"  {pair}: {nt}T {pnl:+.1f}p Sh={sh:.2f} fit={winner.fitness:.4f} ({elapsed:.0f}s)")

    total_elapsed = time.time() - t0_all
    total_pnl = sum(r["oos"].get("total_pnl", 0) for r in per_pair_results.values())
    total_trades = sum(r["oos"].get("n_trades", 0) for r in per_pair_results.values())

    print(f"\n{'='*60}")
    print(f"  PER-PAIR SUMMARY — {label}")
    print(f"{'='*60}")
    for pair in available_pairs:
        r = per_pair_results[pair]["oos"]
        if not r:
            continue
        print(f"  {pair:<10} {r.get('n_trades',0):>5}T {r.get('total_pnl',0):>+9.1f}p "
              f"Sh={r.get('sharpe',0):>6.2f} WR={r.get('win_rate',0):>4.0f}% "
              f"fit={per_pair_results[pair]['fitness']:.4f}")
    print(f"\n  AGGREGATE: {total_trades}T {total_pnl:+.1f}p ({total_elapsed:.0f}s total)")

    tg_send(f"🏁 PER-PAIR {label} s{args.seed} done ({total_elapsed:.0f}s)\n"
            f"{len(available_pairs)} pairs trained\n"
            f"Aggregate OOS: {total_trades}T {total_pnl:+.1f}p")

    # Save summary
    with open(RESULTS_DIR / f"{tag}_perpair_s{args.seed}_summary.json", "w") as f:
        json.dump({"mode": args.mode or variant, "seed": args.seed, "gens": args.gens,
                    "per_pair": True, "pairs": available_pairs,
                    "results": per_pair_results,
                    "aggregate_pnl": round(total_pnl, 1),
                    "aggregate_trades": total_trades,
                    "elapsed_s": round(total_elapsed, 1)}, f, indent=2)
    print(f"\nSaved: {tag}_perpair_s{args.seed}_summary.json")


def _run_single(args):
    """Original single-genome training."""
    np.random.seed(args.seed)
    train_pairs = [p.strip() for p in args.pairs.split(",")]
    use_d_mode = args.mode is not None
    variant = args.variant or args.mode
    label = f"D{args.mode.upper()}" if use_d_mode else f"V{variant}"

    print(f"{'='*60}")
    print(f"  NEAT Training from Pre-computed Indicators")
    print(f"  {'Mode' if use_d_mode else 'Variant'}: {variant} | Seed: {args.seed}")
    print(f"  Pairs: {train_pairs}")
    print(f"  {args.islands} islands × {args.pop} pop, {args.gens} gens")
    print(f"{'='*60}")

    tg_send(f"🏝 {label} training seed {args.seed}\n"
            f"Pairs: {train_pairs}\n{args.islands}×{args.pop} pop, {args.gens}g")

    # Load indicator data
    print("\nLoading pre-computed indicators...")
    train_data = {}
    all_data = {}
    d_train_data = {}
    d_all_data = {}

    for pair in ALL_PAIRS:
        path = DATA_DIR / f"{pair}_asi_mc.parquet"
        if not path.exists():
            if pair in train_pairs:
                print(f"  {pair}: MISSING — cannot train!")
                return
            continue
        df = pd.read_parquet(path, engine="pyarrow")
        mid = df["mid_close"].values.astype(np.float64)
        n = len(mid)
        split = int(n * 0.7)

        if use_d_mode:
            ind_cols, n_inputs = D_MODES[args.mode]
            missing = [c for c in ind_cols if c not in df.columns]
            if missing:
                print(f"  {pair}: MISSING columns {missing} — run export_d_indicators.py first!")
                if pair in train_pairs:
                    return
                continue
            inputs = np.stack([df[c].values.astype(np.float64) for c in ind_cols], axis=0)
            d_data = (inputs[:, :split], mid[:split], inputs[:, split:], mid[split:])
            d_all_data[pair] = d_data
            if pair in train_pairs:
                d_train_data[pair] = d_data
            print(f"  {pair}: {n:,} M5 bars, {len(ind_cols)} indicators: {ind_cols}")
        else:
            mc_d_col = f"mc_d_{variant.lower()}"
            mc_dd_col = f"mc_dd_{variant.lower()}"
            mc_d = df[mc_d_col].values.astype(np.float64)
            mc_dd = df[mc_dd_col].values.astype(np.float64)
            data_tuple = (mc_d[:split], mc_dd[:split], mid[:split],
                          mc_d[split:], mc_dd[split:], mid[split:])
            all_data[pair] = data_tuple
            if pair in train_pairs:
                train_data[pair] = data_tuple
            print(f"  {pair}: {n:,} M5 bars, MC(D) std={mc_d.std():.3f}")

    # Load seed genome
    seed_genome = None
    if args.genome and os.path.exists(args.genome):
        with open(args.genome, "rb") as f:
            d = pickle.load(f)
        seed_genome = d["genome"]
        print(f"\nSeed genome: {Path(args.genome).name}, size={seed_genome.size()}")

    config_path, n_inputs = _get_config_path(args)

    # Sine pretrain if no seed genome (only for ASI-MC variants, not D modes)
    if seed_genome is None and not use_d_mode:
        print("\nSine pretrain (50 gens)...")
        from train_pretrain_continue import generate_sine_ohlc, AsiMcEvaluator
        from lib.asi_indicator import compute_asi_mc
        o_s, h_s, l_s, c_s, mid_s = generate_sine_ohlc(50000, 500, 10)
        n_s = len(mid_s)
        mc_d_s, mc_dd_s = compute_asi_mc(o_s, h_s, l_s, c_s, n_s)
        split = int(n_s * 0.7)
        sine_data = {"sine": (mc_d_s[:split], mc_dd_s[:split], mid_s[:split],
                              mc_d_s[split:], mc_dd_s[split:], mid_s[split:])}
        sine_eval = AsiMcEvaluator(sine_data, pip=0.0001, spread=2.0, max_hold=500)
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path))
        cfg.pop_size = args.pop
        pop = neat.Population(cfg)
        pop.run(sine_eval.evaluate, 50)
        seed_genome = max(pop.population.values(), key=lambda g: g.fitness if g.fitness else -999)
        print(f"  Sine pretrain done: fitness={seed_genome.fitness:.4f}")

    # Train
    if use_d_mode:
        ind_cols, _ = D_MODES[args.mode]
        evaluator = GenEvaluator(d_train_data, n_indicators=len(ind_cols), max_hold=args.max_hold)
    else:
        evaluator = IndicatorEvaluator(train_data, max_hold=args.max_hold)
    tag = f"{args.mode}" if use_d_mode else f"v{variant}"
    save_dir = str(RESULTS_DIR / f"{tag}_s{args.seed}_checkpoints")

    t0 = time.time()
    winner, config = run_island_neat(config_path, evaluator, seed_genome,
                                      args.islands, args.pop, args.gens, save_dir)
    elapsed = time.time() - t0

    print(f"\nTraining done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Fitness: {winner.fitness:.4f}, Size: {winner.size()}")

    # OOS on all 12 pairs
    if use_d_mode:
        ind_cols, _ = D_MODES[args.mode]
        full_eval = GenEvaluator(d_all_data, n_indicators=len(ind_cols), max_hold=args.max_hold)
    else:
        full_eval = IndicatorEvaluator(all_data, max_hold=args.max_hold)
    oos = full_eval.eval_oos(winner, config)

    total_pnl = sum(r["total_pnl"] for r in oos.values())
    total_trades = sum(r["n_trades"] for r in oos.values())
    total_long = sum(r.get("n_long", 0) for r in oos.values())
    total_short = sum(r.get("n_short", 0) for r in oos.values())

    print(f"\n{'='*60}")
    print(f"  OOS — All {len(oos)} pairs ({label})")
    print(f"{'='*60}")
    for pair in ALL_PAIRS:
        r = oos.get(pair, {})
        if not r: continue
        print(f"  {pair:<10} {r['n_trades']:>5}T {r['total_pnl']:>+9.1f}p "
              f"Sh={r['sharpe']:>6.2f} WR={r['win_rate']:>4.0f}% "
              f"L={r['n_long']} S={r['n_short']}")
    print(f"\n  TOTAL: {total_trades}T {total_pnl:+.1f}p L={total_long} S={total_short}")

    tg_send(f"🏁 {label} s{args.seed} done ({elapsed:.0f}s)\n"
            f"Fitness: {winner.fitness:.4f}\n"
            f"12-pair OOS: {total_trades}T {total_pnl:+.1f}p\n"
            f"L={total_long} S={total_short}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_tag = f"{tag}_s{args.seed}"
    with open(RESULTS_DIR / f"{save_tag}_best.pkl", "wb") as f:
        pickle.dump({"genome": winner, "config": config}, f)
    with open(RESULTS_DIR / f"{save_tag}_result.json", "w") as f:
        json.dump({"mode": args.mode or variant, "seed": args.seed, "gens": args.gens,
                    "is_fitness": round(float(winner.fitness), 4),
                    "network_size": list(winner.size()),
                    "train_pairs": train_pairs, "oos": oos,
                    "elapsed_s": round(elapsed, 1)}, f, indent=2)
    print(f"\nSaved: {save_tag}_best.pkl")


if __name__ == "__main__":
    main()
