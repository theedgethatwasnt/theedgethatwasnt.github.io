#!/usr/bin/env python3
"""
IronNet V5: Fixed-Topology 6-Input Per-Pair Training
=====================================================
Extends IronNet V3 (4→4→3) with 2 additional inputs:
  - range_pos: price position within N-bar range, normalized to [-1, 1]
  - bb_width:  Bollinger Band width (upper-lower)/mid, normalized via tanh

Architecture: 6 inputs → 4 hidden → 3 outputs, fully connected + skip.
Total connections: 6×4 + 4×3 + 6×3 = 24 + 12 + 18 = 54 (vs 40 in V3)
NEAT mutates ONLY: weights, biases, activation functions {tanh, sin, cos, gauss}.
No topology mutations.

6 inputs: MC(D), MC(dD), ER_norm, range_pos, bb_width, UPnL
3 outputs: BUY, SELL, FLATTEN

Usage:
  python3 train_ironnet_v5.py --pair EUR_GBP --seed 42
  python3 train_ironnet_v5.py --pair CAD_JPY --seed 137 --gens 200
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

DATA_DIR = Path(os.environ.get("ASI_MC_DATA_DIR",
                str(PROJECT_ROOT / "data" / "asi_mc_indicators")))
RESULTS_DIR = SCRIPT_DIR / "results" / "ironnet_v5"

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

PAIR_MIN_SWING = {
    "EUR_JPY": 30, "USD_JPY": 25, "GBP_JPY": 40, "AUD_JPY": 25,
    "CAD_JPY": 30, "CHF_JPY": 35, "NZD_JPY": 25,
    "EUR_USD": 20, "GBP_USD": 25, "AUD_USD": 18,
    "NZD_USD": 18, "EUR_GBP": 15,
}

N_HIDDEN = 4
N_INPUTS = 6   # MC_D, MC_dD, ER_norm, range_pos, bb_width, UPnL
N_OUTPUTS = 3  # BUY, SELL, FLATTEN
N_CHUNKS = 3   # WF chunks within IS data


# ═══════════════════════════════════════════════════════════════════════════
# Indicator computations (JIT)
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def compute_er_norm(closes, window=60):
    """Kaufman Efficiency Ratio, arctan-normalized. Identical to V3/curator."""
    n = len(closes)
    result = np.zeros(n)
    half_pi = np.pi / 2.0
    for i in range(window, n):
        net_change = abs(closes[i] - closes[i - window])
        total_path = 0.0
        for j in range(i - window + 1, i + 1):
            total_path += abs(closes[j] - closes[j - 1])
        if total_path > 0.0:
            er = net_change / total_path
            result[i] = np.arctan(er / 0.3) / half_pi
    return result


@njit(cache=True)
def compute_range_pos(closes, window=30):
    """Price position within rolling N-bar range, normalized to [-1, 1].
    range_pos = 2 * (close - rolling_min) / (rolling_max - rolling_min) - 1
    Near +1 = at range top (overbought), near -1 = at range bottom (oversold).
    """
    n = len(closes)
    result = np.zeros(n)
    for i in range(window, n):
        lo = closes[i]
        hi = closes[i]
        for j in range(i - window, i + 1):
            if closes[j] < lo:
                lo = closes[j]
            if closes[j] > hi:
                hi = closes[j]
        rng = hi - lo
        if rng > 0.0:
            result[i] = 2.0 * (closes[i] - lo) / rng - 1.0
    return result


@njit(cache=True)
def compute_bb_width(closes, window=20):
    """Bollinger Band width = (upper - lower) / mid, tanh-scaled.
    Width > 0 always. tanh(width * 5) maps typical FX values to (0, 1).
    """
    n = len(closes)
    result = np.zeros(n)
    for i in range(window, n):
        s = 0.0
        for j in range(i - window + 1, i + 1):
            s += closes[j]
        mean = s / window
        var = 0.0
        for j in range(i - window + 1, i + 1):
            d = closes[j] - mean
            var += d * d
        std = (var / window) ** 0.5
        if mean > 0.0:
            width = (4.0 * std) / mean  # upper-lower = 4*std
            result[i] = np.tanh(width * 5.0)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# IronNet V5 architecture: 54 connections
# input→hidden: 6×4=24, hidden→output: 4×3=12, input→output: 6×3=18 skip
# ═══════════════════════════════════════════════════════════════════════════


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


def gauss_activation(x):
    return math.exp(-x * x)

def sin_activation(x):
    return math.sin(x)

def cos_activation(x):
    return math.cos(x)

def tanh_activation(x):
    return math.tanh(x)


def register_activations(config):
    for name, fn in [('gauss', gauss_activation), ('sin', sin_activation),
                     ('cos', cos_activation), ('tanh', tanh_activation)]:
        try:
            config.genome_config.add_activation(name, fn)
        except Exception:
            pass


def build_ironnet_v5_genome(config):
    """Create a fully-connected 6→4→3 genome with skip connections (54 total).

    Inputs:  -1 (MC_D), -2 (MC_dD), -3 (ER_norm), -4 (range_pos), -5 (bb_width), -6 (UPnL)
    Hidden:  100, 101, 102, 103
    Outputs: 0 (BUY), 1 (SELL), 2 (FLAT)
    """
    genome = config.genome_type(0)
    genome.fitness = None

    gc_cfg = config.genome_config
    input_ids = [-1, -2, -3, -4, -5, -6]
    output_ids = [0, 1, 2]
    hidden_ids = [100, 101, 102, 103]

    for nid in output_ids:
        node = gc_cfg.node_gene_type(nid)
        node.bias = np.random.uniform(-1, 1)
        node.response = 1.0
        node.activation = 'tanh'
        node.aggregation = 'sum'
        genome.nodes[nid] = node

    for nid in hidden_ids:
        node = gc_cfg.node_gene_type(nid)
        node.bias = np.random.uniform(-1, 1)
        node.response = 1.0
        node.activation = ['tanh', 'sin', 'cos', 'gauss'][np.random.randint(4)]
        node.aggregation = 'sum'
        genome.nodes[nid] = node

    all_connections = []
    for src in input_ids:
        for dst in hidden_ids:
            all_connections.append((src, dst))
    for src in hidden_ids:
        for dst in output_ids:
            all_connections.append((src, dst))
    for src in input_ids:
        for dst in output_ids:
            all_connections.append((src, dst))

    for innov, (src, dst) in enumerate(all_connections):
        key = (src, dst)
        conn = gc_cfg.connection_gene_type(key, innovation=innov)
        conn.weight = np.random.uniform(-2, 2)
        conn.enabled = True
        genome.connections[key] = conn

    return genome


def build_ironnet_v5_population(config, pop_size):
    population = {}
    for i in range(pop_size):
        g = build_ironnet_v5_genome(config)
        g.key = i
        population[i] = g
    return population


# ═══════════════════════════════════════════════════════════════════════════
# Zigzag labels (identical to V3)
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def generate_zigzag_labels(mid_close, pip, min_swing_pips, label_window=6, min_mfe_pips=3.0):
    n = len(mid_close)
    labels = np.zeros(n, dtype=np.int64)
    min_swing = min_swing_pips * pip
    min_mfe = min_mfe_pips * pip
    running_high = mid_close[0]
    running_low = mid_close[0]
    direction = 0
    for i in range(1, n):
        price = mid_close[i]
        if price > running_high:
            running_high = price
        if price < running_low:
            running_low = price
        if direction == 0:
            if running_high - price >= min_swing:
                direction = -1; running_low = price
            elif price - running_low >= min_swing:
                direction = 1; running_high = price
        elif direction == 1:
            if running_high - price >= min_swing:
                mfe = 0.0
                for j in range(i, min(i + 200, n)):
                    dd = running_high - mid_close[j]
                    if dd > mfe: mfe = dd
                if mfe > min_mfe:
                    end = min(i + label_window, n)
                    for k in range(i, end): labels[k] = 2
                direction = -1; running_low = price
        else:
            if price - running_low >= min_swing:
                mfe = 0.0
                for j in range(i, min(i + 200, n)):
                    uu = mid_close[j] - running_low
                    if uu > mfe: mfe = uu
                if mfe > min_mfe:
                    end = min(i + label_window, n)
                    for k in range(i, end): labels[k] = 1
                direction = 1; running_high = price
    return labels


# ═══════════════════════════════════════════════════════════════════════════
# JIT Evaluators (identical logic to V3, parameter n_inputs handles 6-input)
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
    values = np.zeros(total_values)
    data_len = inputs_2d.shape[1]
    n_ind = inputs_2d.shape[0]
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 10, 10)
    correct = 0; wrong_dir = 0; profitable_correct = 0; total_labeled = 0
    position = 0; entry_price = 0.0

    for i in range(start_bar, end_bar):
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position - spread_pips
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

        if out_buy > out_sell and out_buy > out_flat:
            net_action = 1
        elif out_sell > out_buy and out_sell > out_flat:
            net_action = 2
        else:
            net_action = 0

        label = labels[i]
        if label > 0:
            total_labeled += 1
            if net_action == label:
                correct += 1
                if label == 1:
                    for j in range(i + 1, min(i + 50, end_bar)):
                        if (mid_close[j] - mid_close[i]) / pip - spread_pips > 3.0:
                            profitable_correct += 1; break
                elif label == 2:
                    for j in range(i + 1, min(i + 50, end_bar)):
                        if (mid_close[i] - mid_close[j]) / pip - spread_pips > 3.0:
                            profitable_correct += 1; break
            elif net_action > 0 and net_action != label:
                wrong_dir += 1

        if net_action == 1 and position <= 0:
            position = 1; entry_price = mid_close[i]
        elif net_action == 2 and position >= 0:
            position = -1; entry_price = mid_close[i]
        elif net_action == 0 and position != 0:
            position = 0; entry_price = 0.0

    if total_labeled < 10:
        return -10.0, 0, 0, 0, 0
    accuracy = correct / total_labeled
    wrong_rate = wrong_dir / total_labeled
    profit_rate = profitable_correct / max(correct, 1)
    fitness = accuracy * 10.0 + profit_rate * 5.0 - wrong_rate * 8.0
    return fitness, total_labeled, correct, wrong_dir, profitable_correct


@njit(cache=True)
def evaluate_chunk_jit(
    inputs_2d, mid_close,
    pip, spread_pips, max_hold,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices,
    chunk_start, chunk_end,
):
    values = np.zeros(total_values)
    n_ind = inputs_2d.shape[0]
    start_bar = max(chunk_start + 10, 10)
    end_bar = min(chunk_end, inputs_2d.shape[1] - 1)
    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    trade_maes = np.zeros(max_trades)
    n_trades = 0; n_long = 0; n_short = 0
    position = 0; entry_price = 0.0; entry_bar = 0; worst_ae = 0.0

    for i in range(start_bar, end_bar):
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position - spread_pips
            ae = -(mid_close[i] - entry_price) / pip * position + spread_pips
            if ae > worst_ae: worst_ae = ae
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
                trade_pnls[n_trades] = pnl; trade_maes[n_trades] = worst_ae
                if position > 0: n_long += 1
                else: n_short += 1
                n_trades += 1
            position = 0; worst_ae = 0.0

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
                    trade_pnls[n_trades] = pnl; trade_maes[n_trades] = worst_ae
                    if position > 0: n_long += 1
                    else: n_short += 1
                    n_trades += 1
                position = new_pos
                entry_price = mid_close[i] if new_pos != 0 else 0.0
                entry_bar = i; worst_ae = 0.0

    if position != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar - 1] - entry_price) / pip * position - spread_pips
        if n_trades < max_trades:
            trade_pnls[n_trades] = pnl; trade_maes[n_trades] = worst_ae
            if position > 0: n_long += 1
            else: n_short += 1
            n_trades += 1

    if n_trades < 1:
        return 0, 0.0, 0.0, 0, 0
    total_pnl = 0.0; total_mae = 0.0
    for j in range(n_trades):
        total_pnl += trade_pnls[j]; total_mae += trade_maes[j]
    avg_mae = total_mae / n_trades
    return n_trades, total_pnl, avg_mae, n_long, n_short


# ═══════════════════════════════════════════════════════════════════════════
# Evaluators (identical to V3 — work on any N_INPUTS via extract_network)
# ═══════════════════════════════════════════════════════════════════════════

class IronNetWFEvaluator:
    def __init__(self, inputs_2d, mid_close, pip, spread,
                 max_hold=200, n_chunks=3, min_dir_ratio=0.15):
        self.inputs_2d = inputs_2d
        self.mid_close = mid_close
        self.pip = pip
        self.spread = spread
        self.max_hold = max_hold
        self.n_chunks = n_chunks
        self.min_dir_ratio = min_dir_ratio
        self.n_bars = inputs_2d.shape[1]

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return -10.0

        chunk_scores = []
        total_long = 0; total_short = 0; total_trades = 0

        for ci in range(self.n_chunks):
            c_start = int(self.n_bars * ci / self.n_chunks)
            c_end = int(self.n_bars * (ci + 1) / self.n_chunks)

            nt, pnl, mae, nl, ns = evaluate_chunk_jit(
                self.inputs_2d, self.mid_close,
                self.pip, self.spread, self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10],
                c_start, c_end)

            total_long += nl; total_short += ns; total_trades += nt

            min_trades_per_chunk = max(30, int(self.n_bars / self.n_chunks / 288 * 0.5))
            if nt < min_trades_per_chunk:
                return -10.0
            if pnl <= 0:
                return -10.0

            mean_pnl = pnl / nt
            score = (mean_pnl / mae) if mae > 0 else mean_pnl
            chunk_scores.append(score * (nt ** 0.5))

        if total_trades < 10:
            return -10.0
        dir_ratio = min(total_long, total_short) / total_trades
        if dir_ratio < self.min_dir_ratio:
            return -10.0

        min_score = min(chunk_scores)
        mean_score = sum(chunk_scores) / len(chunk_scores)
        if mean_score > 0:
            cv = (sum((s - mean_score) ** 2 for s in chunk_scores) / len(chunk_scores)) ** 0.5 / mean_score
            consistency = 1.0 / (1.0 + cv)
        else:
            consistency = 0.5
        dir_bonus = 1.0 + 0.5 * (dir_ratio - self.min_dir_ratio) / (0.5 - self.min_dir_ratio)
        return min_score * (1.0 + consistency) * dir_bonus


class IronNetSupervisedEvaluator:
    def __init__(self, inputs_2d, mid_close, labels, pip, spread):
        self.inputs_2d = inputs_2d; self.mid_close = mid_close
        self.labels = labels; self.pip = pip; self.spread = spread

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            try:
                net = extract_network(genome, config)
            except Exception:
                genome.fitness = -10.0; continue
            result = evaluate_supervised_jit(
                self.inputs_2d, self.mid_close, self.labels,
                self.pip, self.spread, len(self.mid_close),
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10], 0)
            genome.fitness = float(result[0])


def eval_oos(genome, config, inputs_2d, mid_close, pip, spread, max_hold=200):
    net = extract_network(genome, config)
    nt, pnl, mae, nl, ns = evaluate_chunk_jit(
        inputs_2d, mid_close, pip, spread, max_hold,
        net[0], net[2], net[3], net[4], net[5], net[6],
        net[7], net[8], net[9], net[10],
        0, inputs_2d.shape[1])
    n_days = inputs_2d.shape[1] / 288.0
    return {
        "n_trades": int(nt), "total_pnl": round(float(pnl), 1),
        "avg_mae": round(float(mae), 2),
        "n_long": int(nl), "n_short": int(ns),
        "pips_per_day": round(float(pnl) / max(n_days, 1), 1),
        "dir_ratio": round(min(nl, ns) / max(nt, 1), 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Island evolution (identical logic to V3)
# ═══════════════════════════════════════════════════════════════════════════

def mutate_ironnet(genome, config):
    """Mutate weights, biases, and activations only. No topology changes."""
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


def run_island_loop(config_path, evaluator, seed_genome=None,
                    n_islands=4, pop_per_island=150, generations=200,
                    save_dir=None, label="", stall_limit=40):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_path))
    register_activations(config)

    best_ever = None; best_ever_fitness = -999; stall_counter = 0
    islands = []
    for i in range(n_islands):
        pop = {}
        for j in range(pop_per_island):
            if seed_genome is not None:
                g = copy.deepcopy(seed_genome)
                g.key = j; g.fitness = None
                mutate_ironnet(g, config)
            else:
                g = build_ironnet_v5_genome(config)
                g.key = j
            pop[j] = g
        islands.append({"pop": pop, "config": config, "best": None})

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for gen in range(generations):
        for i, island in enumerate(islands):
            pop = island["pop"]
            evaluator.evaluate(list(pop.items()), config)
            best = max(pop.values(), key=lambda g: g.fitness if g.fitness is not None else -999)
            island["best"] = best
            if best.fitness is not None and best.fitness > best_ever_fitness:
                best_ever = copy.deepcopy(best)
                best_ever_fitness = best.fitness
                stall_counter = 0

            sorted_g = sorted(pop.values(),
                               key=lambda g: g.fitness if g.fitness is not None else -999,
                               reverse=True)
            new_pop = {}
            for j in range(min(3, len(sorted_g))):
                e = copy.deepcopy(sorted_g[j]); e.key = j; e.fitness = None; new_pop[j] = e
            for j in range(3, pop_per_island):
                cands = np.random.choice(len(sorted_g), size=min(3, len(sorted_g)), replace=False)
                p = copy.deepcopy(sorted_g[min(cands)]); p.key = j; p.fitness = None
                mutate_ironnet(p, config); new_pop[j] = p
            island["pop"] = new_pop

        if gen > 0 and gen % 10 == 0:
            for i in range(n_islands):
                src = islands[i]; dst = islands[(i + 1) % n_islands]
                if src["best"] is not None:
                    worst_gid = min(dst["pop"],
                                    key=lambda g: dst["pop"][g].fitness
                                    if dst["pop"][g].fitness is not None else 999)
                    migrant = copy.deepcopy(src["best"])
                    migrant.key = worst_gid; migrant.fitness = None
                    dst["pop"][worst_gid] = migrant

        stall_counter += 1

        if save_dir:
            gen_best = max((isl["best"] for isl in islands if isl["best"] is not None),
                           key=lambda g: g.fitness if g.fitness is not None else -999, default=None)
            if gen_best:
                with open(f"{save_dir}/gen_{gen:03d}_best.pkl", "wb") as f:
                    pickle.dump({"genome": copy.deepcopy(gen_best), "config": config,
                                 "generation": gen, "fitness": gen_best.fitness}, f)

        fitnesses = [isl["best"].fitness for isl in islands
                     if isl["best"] is not None and isl["best"].fitness is not None]
        if fitnesses and gen % 10 == 0:
            print(f"  [{label}] Gen {gen:>3}: best={max(fitnesses):.4f} "
                  f"global={best_ever_fitness:.4f} stall={stall_counter}")

        if stall_counter >= stall_limit:
            print(f"  [{label}] Early stop gen {gen} (stalled {stall_limit})")
            break

    return best_ever, config, gen


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IronNet V5: 6-input fixed-topology per-pair")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sine-gens", type=int, default=30)
    parser.add_argument("--pretrain-gens", type=int, default=50)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--pop", type=int, default=150)
    parser.add_argument("--max-hold", type=int, default=200)
    parser.add_argument("--min-swing", type=int, default=0)
    parser.add_argument("--label-window", type=int, default=6)
    parser.add_argument("--stall-limit", type=int, default=40)
    parser.add_argument("--min-dir-ratio", type=float, default=0.15)
    parser.add_argument("--range-window", type=int, default=30)
    parser.add_argument("--bb-window", type=int, default=20)
    args = parser.parse_args()

    np.random.seed(args.seed)
    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]
    min_swing_pips = args.min_swing if args.min_swing > 0 else PAIR_MIN_SWING.get(pair, 20)
    config_path = SCRIPT_DIR / "neat_config_6in_3out.ini"

    print(f"{'='*65}")
    print(f"  IronNet V5 Per-Pair: {pair}")
    print(f"  Fixed topology: {N_INPUTS}→{N_HIDDEN}→{N_OUTPUTS} + skip (54 conn)")
    print(f"  Inputs: MC_D, MC_dD, ER_norm, range_pos({args.range_window}), bb_width({args.bb_window}), UPnL")
    print(f"  Activations: tanh, sin, cos, gauss")
    print(f"  Seed: {args.seed} | {args.islands}×{args.pop} pop")
    print(f"  Sine: {args.sine_gens}g → Zigzag: {args.pretrain_gens}g → WF P&L: {args.gens}g")
    print(f"{'='*65}")
    tg_send(f"🔩 IronNet V5 {pair} s{args.seed}\n"
            f"6→4→3 fixed | 6 inputs (+range_pos, +bb_width)\n"
            f"PT {args.pretrain_gens}g + WF {args.gens}g")

    # Load data
    path = DATA_DIR / f"{pair}_asi_mc.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found"); return

    df = pd.read_parquet(path, engine="pyarrow")
    mid = df["mid_close"].values.astype(np.float64)
    n = len(mid)
    split = int(n * 0.7)

    # Compute 5 indicator columns (UPnL added dynamically in evaluator)
    mc_d = df["mc_d_a"].values.astype(np.float64)
    mc_dd = df["mc_dd_a"].values.astype(np.float64)
    er = compute_er_norm(mid, window=60)
    rp = compute_range_pos(mid, window=args.range_window)
    bbw = compute_bb_width(mid, window=args.bb_window)
    del df; gc.collect()

    inputs = np.stack([mc_d, mc_dd, er, rp, bbw], axis=0)  # shape (5, n)
    inputs_is = inputs[:, :split]
    mid_is = mid[:split]
    inputs_oos = inputs[:, split:]
    mid_oos = mid[split:]

    print(f"\nData: {n:,} M5 bars | IS: {split:,} | OOS: {n - split:,}")
    print(f"  MC_D    range: [{inputs_is[0].min():.4f}, {inputs_is[0].max():.4f}]")
    print(f"  MC_dD   range: [{inputs_is[1].min():.4f}, {inputs_is[1].max():.4f}]")
    print(f"  ER      range: [{inputs_is[2].min():.4f}, {inputs_is[2].max():.4f}]")
    print(f"  range_pos range: [{inputs_is[3].min():.4f}, {inputs_is[3].max():.4f}]")
    print(f"  bb_width range: [{inputs_is[4].min():.4f}, {inputs_is[4].max():.4f}]")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"iron_v5_{pair}_s{args.seed}"

    # ── Phase 1: Sine pretrain ────────────────────────────────────────────
    print(f"\nPhase 1: Sine wave pretrain ({args.sine_gens} gens)...")
    from lib.asi_indicator import compute_asi_mc

    pair_center = float(mid_is.mean())
    amp_pips = 15
    n_sine = 50000
    sine_period = 500
    rng = np.random.RandomState(args.seed)
    t_arr = np.arange(n_sine, dtype=np.float64)
    sine_mid = pair_center + amp_pips * pip * np.sin(2 * np.pi * t_arr / sine_period)
    sine_mid += rng.normal(0, 1.0 * pip, n_sine)

    sine_o = np.empty(n_sine, dtype=np.float64)
    sine_c = sine_mid.copy()
    sine_o[0] = sine_mid[0]
    for ii in range(1, n_sine):
        sine_o[ii] = sine_c[ii - 1]
    hl_noise = rng.uniform(0, 2 * pip, n_sine)
    sine_h = np.maximum(sine_o, sine_c) + hl_noise
    sine_l = np.minimum(sine_o, sine_c) - hl_noise

    sine_mc_d, sine_mc_dd = compute_asi_mc(sine_o, sine_h, sine_l, sine_c, n_sine)
    sine_er = compute_er_norm(sine_mid, window=60)
    sine_rp = compute_range_pos(sine_mid, window=args.range_window)
    sine_bbw = compute_bb_width(sine_mid, window=args.bb_window)

    sine_inputs = np.stack([sine_mc_d, sine_mc_dd, sine_er, sine_rp, sine_bbw], axis=0)
    sine_labels = generate_zigzag_labels(sine_mid, pip, amp_pips // 2,
                                          label_window=8, min_mfe_pips=2.0)
    print(f"  Sine BUY={int(np.sum(sine_labels==1))} SELL={int(np.sum(sine_labels==2))}")

    sine_eval = IronNetSupervisedEvaluator(sine_inputs, sine_mid, sine_labels, pip, spread)
    t0 = time.time()
    sine_genome, _, _ = run_island_loop(
        config_path, sine_eval, seed_genome=None,
        n_islands=args.islands, pop_per_island=args.pop,
        generations=args.sine_gens,
        save_dir=str(RESULTS_DIR / f"{tag}_sine_ckpt"),
        label=f"{pair} SINE", stall_limit=args.sine_gens)
    sine_elapsed = time.time() - t0
    print(f"  Sine done: fitness={sine_genome.fitness:.4f} ({sine_elapsed:.0f}s)")

    del sine_inputs, sine_mid, sine_mc_d, sine_mc_dd, sine_er, sine_rp, sine_bbw, sine_labels
    gc.collect()

    # ── Phase 2: Zigzag pretrain ──────────────────────────────────────────
    print(f"\nPhase 2: Zigzag pretrain ({args.pretrain_gens} gens)...")
    labels_is = generate_zigzag_labels(mid_is, pip, min_swing_pips,
                                        label_window=args.label_window,
                                        min_mfe_pips=spread + 2.0)
    n_buy = int(np.sum(labels_is == 1)); n_sell = int(np.sum(labels_is == 2))
    print(f"  BUY={n_buy:,} SELL={n_sell:,} ({100*(n_buy+n_sell)/len(labels_is):.1f}%)")

    if n_buy + n_sell < 100:
        reduced = max(min_swing_pips // 2, 5)
        labels_is = generate_zigzag_labels(mid_is, pip, reduced,
                                            label_window=args.label_window,
                                            min_mfe_pips=spread + 1.0)
        n_buy = int(np.sum(labels_is == 1)); n_sell = int(np.sum(labels_is == 2))
        print(f"  Reduced to {reduced}p: BUY={n_buy:,} SELL={n_sell:,}")
        min_swing_pips = reduced

    zz_eval = IronNetSupervisedEvaluator(inputs_is, mid_is, labels_is, pip, spread)
    t_pt = time.time()
    pretrained, _, _ = run_island_loop(
        config_path, zz_eval, seed_genome=sine_genome,
        n_islands=args.islands, pop_per_island=args.pop,
        generations=args.pretrain_gens,
        save_dir=str(RESULTS_DIR / f"{tag}_zigzag_ckpt"),
        label=f"{pair} ZZ", stall_limit=args.pretrain_gens)
    pt_elapsed = time.time() - t_pt
    print(f"  Zigzag done: fitness={pretrained.fitness:.4f} ({pt_elapsed:.0f}s)")
    tg_send(f"✅ IronNet V5 {pair} phases 1+2: sine={sine_genome.fitness:.4f} zz={pretrained.fitness:.4f}")

    # ── Phase 3: WF P&L evolution ─────────────────────────────────────────
    print(f"\nPhase 3: WF P&L evolution ({args.gens} gens, {N_CHUNKS} chunks)...")
    wf_eval = IronNetWFEvaluator(inputs_is, mid_is, pip, spread,
                                  max_hold=args.max_hold,
                                  n_chunks=N_CHUNKS,
                                  min_dir_ratio=args.min_dir_ratio)
    t1 = time.time()
    best_ever, config, last_gen = run_island_loop(
        config_path, wf_eval, seed_genome=pretrained,
        n_islands=args.islands, pop_per_island=args.pop,
        generations=args.gens,
        save_dir=str(RESULTS_DIR / f"{tag}_evolve_ckpt"),
        label=f"{pair} EV", stall_limit=args.stall_limit)
    ev_elapsed = time.time() - t1
    best_ever_fitness = best_ever.fitness if best_ever else -999
    print(f"  Evolution done: fitness={best_ever_fitness:.4f} ({ev_elapsed:.0f}s)")

    # ── Phase 4: OOS evaluation ───────────────────────────────────────────
    print(f"\nPhase 4: OOS evaluation...")
    is_res = eval_oos(best_ever, config, inputs_is, mid_is, pip, spread, args.max_hold)
    oos_res = eval_oos(best_ever, config, inputs_oos, mid_oos, pip, spread, args.max_hold)

    print(f"\n{'='*65}")
    print(f"  RESULTS: {pair} (IronNet V5 — 6 inputs)")
    print(f"{'='*65}")
    print(f"  IS:  {is_res['n_trades']:>5}T {is_res['total_pnl']:>+9.1f}p "
          f"L={is_res['n_long']} S={is_res['n_short']} "
          f"MAE={is_res['avg_mae']:.1f}p dir={is_res['dir_ratio']:.2f} "
          f"({is_res['pips_per_day']:.1f}p/day)")
    print(f"  OOS: {oos_res['n_trades']:>5}T {oos_res['total_pnl']:>+9.1f}p "
          f"L={oos_res['n_long']} S={oos_res['n_short']} "
          f"MAE={oos_res['avg_mae']:.1f}p dir={oos_res['dir_ratio']:.2f} "
          f"({oos_res['pips_per_day']:.1f}p/day)")
    print(f"  Fitness: {best_ever_fitness:.4f} | Size: {best_ever.size()}")
    acts = [n.activation for n in best_ever.nodes.values()]
    n_enabled = sum(1 for c in best_ever.connections.values() if c.enabled)
    print(f"  Connections: {n_enabled}/54 enabled | Activations: {dict((a, acts.count(a)) for a in set(acts))}")

    result_data = {
        "pair": pair, "variant": "ironnet_v5",
        "seed": args.seed,
        "n_inputs": N_INPUTS, "n_hidden": N_HIDDEN,
        "pretrain_gens": args.pretrain_gens,
        "evolution_gens": args.gens,
        "actual_gens": last_gen + 1,
        "min_swing_pips": min_swing_pips,
        "n_chunks": N_CHUNKS,
        "min_dir_ratio": args.min_dir_ratio,
        "range_window": args.range_window,
        "bb_window": args.bb_window,
        "fitness": round(best_ever_fitness, 4),
        "network_size": list(best_ever.size()),
        "activations": dict((a, acts.count(a)) for a in set(acts)),
        "is": is_res, "oos": oos_res,
        "pretrain_time_s": round(pt_elapsed, 1),
        "evolve_time_s": round(ev_elapsed, 1),
    }

    with open(RESULTS_DIR / f"{tag}_best.pkl", "wb") as f:
        pickle.dump({"genome": best_ever, "config": config, "pair": pair,
                     "result": result_data}, f)
    with open(RESULTS_DIR / f"{tag}_result.json", "w") as f:
        json.dump(result_data, f, indent=2)

    tg_send(f"🔩 IronNet V5 {pair} s{args.seed} DONE\n"
            f"6→4→3 fixed (54 conn)\n"
            f"Fitness: {best_ever_fitness:.4f}\n"
            f"OOS: {oos_res['n_trades']}T {oos_res['total_pnl']:+.1f}p "
            f"dir={oos_res['dir_ratio']:.2f} ({oos_res['pips_per_day']:.1f}p/day)")
    print(f"\nSaved: {tag}_best.pkl + {tag}_result.json")
    print(f"Total: {pt_elapsed + ev_elapsed:.0f}s")


if __name__ == "__main__":
    main()
