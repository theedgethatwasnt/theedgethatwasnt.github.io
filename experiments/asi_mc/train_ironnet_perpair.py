#!/usr/bin/env python3
"""
IronNet: Fixed-Topology Per-Pair ASI-MC V3 Training
=====================================================
"Iron" = rigid architecture, no dead outputs possible.

Architecture: 4 inputs → 4 hidden → 3 outputs, fully connected + skip.
NEAT mutates ONLY: weights, biases, activation functions {tanh, sin, cos, gauss}.
No topology mutations (no add/delete nodes or connections, no enable/disable).

Fitness: WF-in-fitness (3 chunks), PnL/MAE, bidirectional enforcement.
Phase 1: Zigzag supervised pretrain (50 gens)
Phase 2: WF P&L evolution (150 gens)
Phase 3: OOS evaluation

4 inputs: MC(D), MC(dD), ER_norm, UPnL
3 outputs: BUY, SELL, FLATTEN

Usage:
  python3 train_ironnet_perpair.py --pair EUR_GBP --seed 42
  python3 train_ironnet_perpair.py --pair CAD_JPY --seed 42 --gens 200
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
RESULTS_DIR = SCRIPT_DIR / "results" / "ironnet"

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

N_OUTPUTS = 3  # BUY, SELL, FLATTEN
N_CHUNKS = 3   # WF chunks within IS data
# V3: 4 inputs (mc_d, mc_dd, er_norm, UPnL) → 4 hidden → 3 outputs (40 conn)
# V7: 7 inputs (bb_width, stoch_d, macd_hist, range_pos_30, aroon_osc, mc_d_a, UPnL) → 5 hidden → 3 outputs (71 conn)
MODE_CONFIG = {
    "v3": {"n_inputs": 4, "n_hidden": 4, "ind_cols": ["mc_d_a", "mc_dd_a", "er_norm"],
           "data_dir_name": "asi_mc_indicators", "parquet_suffix": "_asi_mc",
           "normalize": {}},
    "v7": {"n_inputs": 7, "n_hidden": 5, "ind_cols": ["bb_width", "stoch_d", "macd_hist", "range_pos_30", "aroon_osc", "mc_d_a"],
           "data_dir_name": "v7_indicators", "parquet_suffix": "_v7",
           "normalize": {0: ("mul", 20.0), 2: ("div_clip", 2.0)}},
    # ── Feature set experiments A-F (SHAP-selected, 1-hidden-layer: N→4→3) ──
    # 3-input sets (+ UPnL = 4 total): 4→4→3 (40 conn) — same as proven V3
    "setA": {"n_inputs": 4, "n_hidden": 4, "ind_cols": ["tec_5", "bb_width", "h1_slope"],
             "data_dir_name": "unified_indicators", "parquet_suffix": "_unified",
             "normalize": {1: ("mul", 20.0)}},
    "setB": {"n_inputs": 4, "n_hidden": 4, "ind_cols": ["tec_5", "stoch_d", "range_pos_30"],
             "data_dir_name": "unified_indicators", "parquet_suffix": "_unified",
             "normalize": {}},
    "setC": {"n_inputs": 4, "n_hidden": 4, "ind_cols": ["tec_5", "bb_width", "gap_norm"],
             "data_dir_name": "unified_indicators", "parquet_suffix": "_unified",
             "normalize": {1: ("mul", 20.0), 2: ("div_clip", 3.0)}},
    # ── Recovered experiments (fell between chairs — shallow retry) ──
    # S1-S4: bounded subsets, original 5→5→10→3→3 failed from bugs, now 5→4→3
    "s1": {"n_inputs": 5, "n_hidden": 4, "ind_cols": ["mc_d_a", "mc_dd_a", "er_norm", "sb_a"],
           "data_dir_name": "asi_mc_indicators", "parquet_suffix": "_asi_mc",
           "normalize": {}},
    "s2": {"n_inputs": 5, "n_hidden": 4, "ind_cols": ["sb_a", "hh_asi", "hl_asi", "er_norm"],
           "data_dir_name": "asi_mc_indicators", "parquet_suffix": "_asi_mc",
           "normalize": {}},
    "s3": {"n_inputs": 5, "n_hidden": 4, "ind_cols": ["sb_a", "mc_d_a", "mc_dd_a", "hh_asi"],
           "data_dir_name": "asi_mc_indicators", "parquet_suffix": "_asi_mc",
           "normalize": {}},
    "s4": {"n_inputs": 5, "n_hidden": 4, "ind_cols": ["hh_asi", "hl_asi", "mc_d_a", "er_norm"],
           "data_dir_name": "asi_mc_indicators", "parquet_suffix": "_asi_mc",
           "normalize": {}},
    # E3 bounded: original 10→4→3 collapsed, remove unbounded erp_*, keep 6 bounded → 7→4→3
            "aroon_only": {"n_inputs": 2, "n_hidden": 4, "ind_cols": ["aroon_osc"],
           "data_dir_name": "unified_indicators", "parquet_suffix": "_unified",
           "normalize": {}},
    "er_trade": {"n_inputs": 4, "n_hidden": 4, "ind_cols": ["tec_5", "aroon_osc", "range_pos_30"],
           "data_dir_name": "unified_indicators", "parquet_suffix": "_unified",
           "normalize": {}},
    "e3b": {"n_inputs": 7, "n_hidden": 4, "ind_cols": ["mc_d_a", "mc_dd_a", "er_norm", "sb_a", "hh_asi", "hl_asi"],
            "data_dir_name": "asi_mc_indicators", "parquet_suffix": "_asi_mc",
            "normalize": {}},
    # 4-input sets (+ UPnL = 5 total): 5→4→3 (47 conn)
    "setD": {"n_inputs": 5, "n_hidden": 4, "ind_cols": ["tec_5", "bb_width", "h1_slope", "stoch_d"],
             "data_dir_name": "unified_indicators", "parquet_suffix": "_unified",
             "normalize": {1: ("mul", 20.0)}},
    "setE": {"n_inputs": 5, "n_hidden": 4, "ind_cols": ["tec_5", "bb_width", "range_pos_30", "stoch_d"],
             "data_dir_name": "unified_indicators", "parquet_suffix": "_unified",
             "normalize": {1: ("mul", 20.0)}},
    "setF": {"n_inputs": 5, "n_hidden": 4, "ind_cols": ["tec_5", "h1_slope", "hl_price", "stoch_d"],
             "data_dir_name": "unified_indicators", "parquet_suffix": "_unified",
             "normalize": {}},
}


@njit(cache=True)
def compute_er_norm(closes, window=60):
    """Kaufman Efficiency Ratio (window bars), arctan-normalized.
    Identical to export_d_indicators.py and curator live code.
    ER = |close[i] - close[i-window]| / sum(|diff(close)| over window)
    er_norm = arctan(ER / 0.3) / (pi/2)
    """
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

# IronNet architecture: 40 connections + 7 biases + 7 activations
# input→hidden: 4×4=16, hidden→output: 4×3=12, input→output: 4×3=12 skip = 40 total


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
# Register gauss activation with neat-python
# ═══════════════════════════════════════════════════════════════════════════

def gauss_activation(x): return math.exp(-x * x)
def sin_activation(x): return math.sin(x)
def cos_activation(x): return math.cos(x)
def tanh_activation(x): return math.tanh(x)
def sech_activation(x): return 1.0 / math.cosh(max(min(x, 50), -50))
def dog_activation(x): return math.exp(-x*x/2) - 0.5*math.exp(-x*x/8)
def gabor_activation(x): return math.exp(-2*x*x) * math.cos(2*math.pi*x)
def sinc_activation(x): return math.sin(math.pi*x)/(math.pi*x) if abs(x) > 1e-7 else 1.0
def morlet_activation(x): return math.sin(x) * math.exp(-x*x/2)


def register_activations(config):
    """Register all wavelet activations with a NEAT config."""
    activations = [
        ('tanh', tanh_activation), ('sin', sin_activation), ('cos', cos_activation),
        ('gauss', gauss_activation), ('sech', sech_activation), ('dog', dog_activation),
        ('gabor', gabor_activation), ('sinc', sinc_activation), ('morlet', morlet_activation),
    ]
    for name, fn in activations:
        try:
            config.genome_config.add_activation(name, fn)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Build fixed-topology NEAT genome
# ═══════════════════════════════════════════════════════════════════════════

def build_ironnet_genome(config, n_inputs=4, n_hidden=4):
    """Create a fully-connected genome with skip connections.

    n_hidden: int (1 layer) or list[int] (multi-layer, e.g. [7, 4])

    1-layer: input→h1→output + input→output (skip)
    2-layer: input→h1→h2→output + input→output (skip)
    """
    genome = config.genome_type(0)
    genome.fitness = None
    gc = config.genome_config

    # Handle single or multi hidden layer
    if isinstance(n_hidden, int):
        hidden_layers = [n_hidden]
    else:
        hidden_layers = list(n_hidden)

    input_ids = list(range(-n_inputs, 0))
    output_ids = [0, 1, 2]

    # Create hidden layer node IDs: layer 0 starts at 100, layer 1 at 200, etc.
    all_hidden_ids = []
    for li, layer_size in enumerate(hidden_layers):
        base = 100 + li * 100
        layer_ids = list(range(base, base + layer_size))
        all_hidden_ids.append(layer_ids)

    # Create output nodes
    for nid in output_ids:
        node = gc.node_gene_type(nid)
        node.bias = np.random.uniform(-1, 1)
        node.response = 1.0
        node.activation = 'tanh'
        node.aggregation = 'sum'
        genome.nodes[nid] = node

    # Create hidden nodes
    act_choices = ['tanh', 'sin', 'cos', 'gauss', 'sech', 'dog', 'gabor', 'sinc', 'morlet']
    for layer_ids in all_hidden_ids:
        for nid in layer_ids:
            node = gc.node_gene_type(nid)
            node.bias = np.random.uniform(-1, 1)
            node.response = 1.0
            node.activation = act_choices[np.random.randint(4)]
            node.aggregation = 'sum'
            genome.nodes[nid] = node

    # Create connections: sequential layers + input→output skip
    all_connections = []

    # input → first hidden layer
    for src in input_ids:
        for dst in all_hidden_ids[0]:
            all_connections.append((src, dst))

    # hidden layer i → hidden layer i+1
    for li in range(len(all_hidden_ids) - 1):
        for src in all_hidden_ids[li]:
            for dst in all_hidden_ids[li + 1]:
                all_connections.append((src, dst))

    # last hidden → output
    for src in all_hidden_ids[-1]:
        for dst in output_ids:
            all_connections.append((src, dst))

    # input → output (skip)
    for src in input_ids:
        for dst in output_ids:
            all_connections.append((src, dst))

    for innov, (src, dst) in enumerate(all_connections):
        key = (src, dst)
        conn = gc.connection_gene_type(key, innovation=innov)
        conn.weight = np.random.uniform(-2, 2)
        conn.enabled = True
        genome.connections[key] = conn

    return genome


def build_ironnet_population(config, pop_size, n_inputs=4, n_hidden=4):
    """Create a population of IronNet genomes."""
    population = {}
    for i in range(pop_size):
        g = build_ironnet_genome(config, n_inputs, n_hidden)
        g.key = i
        population[i] = g
    return population


# ═══════════════════════════════════════════════════════════════════════════
# Zigzag labels (reuse from train_zigzag_perpair.py)
# ═══════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def generate_zigzag_labels(mid_close, pip, min_swing_pips, label_window=6, min_mfe_pips=3.0):
    """Generate BUY/SELL/FLATTEN labels from zigzag swing detection."""
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
                direction = -1
                running_low = price
            elif price - running_low >= min_swing:
                direction = 1
                running_high = price
        elif direction == 1:
            if running_high - price >= min_swing:
                mfe = 0.0
                for j in range(i, min(i + 200, n)):
                    dd = running_high - mid_close[j]
                    if dd > mfe:
                        mfe = dd
                if mfe > min_mfe:
                    end = min(i + label_window, n)
                    for k in range(i, end):
                        labels[k] = 2
                direction = -1
                running_low = price
        else:
            if price - running_low >= min_swing:
                mfe = 0.0
                for j in range(i, min(i + 200, n)):
                    uu = mid_close[j] - running_low
                    if uu > mfe:
                        mfe = uu
                if mfe > min_mfe:
                    end = min(i + label_window, n)
                    for k in range(i, end):
                        labels[k] = 1
                direction = 1
                running_high = price

    return labels


# ═══════════════════════════════════════════════════════════════════════════
# JIT Evaluators
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
    """Supervised fitness: match zigzag labels. Returns (fitness, total_labeled, correct, wrong, profitable)."""
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
                            profitable_correct += 1
                            break
                elif label == 2:
                    for j in range(i + 1, min(i + 50, end_bar)):
                        if (mid_close[i] - mid_close[j]) / pip - spread_pips > 3.0:
                            profitable_correct += 1
                            break
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
    """Evaluate on a data chunk. Returns (n_trades, total_pnl, avg_mae, n_long, n_short)."""
    values = np.zeros(total_values)
    n_ind = inputs_2d.shape[0]
    start_bar = max(chunk_start + 10, 10)
    end_bar = min(chunk_end, inputs_2d.shape[1] - 1)
    max_trades = end_bar - start_bar + 1
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
                trade_maes[n_trades] = worst_ae
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
                    trade_pnls[n_trades] = pnl
                    trade_maes[n_trades] = worst_ae
                    if position > 0: n_long += 1
                    else: n_short += 1
                    n_trades += 1
                position = new_pos
                entry_price = mid_close[i] if new_pos != 0 else 0.0
                entry_bar = i; worst_ae = 0.0

    if position != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar - 1] - entry_price) / pip * position - spread_pips
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


# ═══════════════════════════════════════════════════════════════════════════
# WF P&L Evaluator with bidirectional enforcement
# ═══════════════════════════════════════════════════════════════════════════

class IronNetWFEvaluator:
    """Walk-Forward PnL/MAE evaluator for single pair.

    Hard gates:
    - Minimum trades per chunk
    - Must be profitable in EVERY chunk
    - Must trade both directions (bidirectional enforcement)
    """

    def __init__(self, inputs_2d, mid_close, pip, spread,
                 max_hold=200, n_chunks=3, min_dir_ratio=0.15):
        self.inputs_2d = inputs_2d
        self.mid_close = mid_close
        self.pip = pip
        self.spread = spread
        self.max_hold = max_hold
        self.n_chunks = n_chunks
        self.min_dir_ratio = min_dir_ratio  # minimum fraction of minority direction
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
        total_long = 0
        total_short = 0
        total_trades = 0

        for ci in range(self.n_chunks):
            c_start = int(self.n_bars * ci / self.n_chunks)
            c_end = int(self.n_bars * (ci + 1) / self.n_chunks)

            nt, pnl, mae, nl, ns = evaluate_chunk_jit(
                self.inputs_2d, self.mid_close,
                self.pip, self.spread, self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10],
                c_start, c_end)

            total_long += nl
            total_short += ns
            total_trades += nt

            # Hard gates per chunk
            min_trades_per_chunk = max(30, int(self.n_bars / self.n_chunks / 288 * 0.5))
            if nt < min_trades_per_chunk:
                return -10.0
            if pnl <= 0:
                return -10.0

            # PnL/MAE score
            mean_pnl = pnl / nt
            if mae > 0:
                score = mean_pnl / mae
            else:
                score = mean_pnl
            chunk_scores.append(score * (nt ** 0.5))

        # Bidirectional enforcement (hard gate)
        if total_trades < 10:
            return -10.0
        dir_ratio = min(total_long, total_short) / total_trades
        if dir_ratio < self.min_dir_ratio:
            return -10.0

        # Final: min(chunks) × consistency × direction bonus
        min_score = min(chunk_scores)
        mean_score = sum(chunk_scores) / len(chunk_scores)
        if mean_score > 0:
            cv = sum((s - mean_score) ** 2 for s in chunk_scores) / len(chunk_scores)
            cv = cv ** 0.5 / mean_score
            consistency = 1.0 / (1.0 + cv)
        else:
            consistency = 0.5

        # Direction balance bonus: max at 50/50, min at threshold
        dir_bonus = 1.0 + 0.5 * (dir_ratio - self.min_dir_ratio) / (0.5 - self.min_dir_ratio)

        return min_score * (1.0 + consistency) * dir_bonus


# ═══════════════════════════════════════════════════════════════════════════
# Island Evolution (fixed topology — no NEAT topology ops)
# ═══════════════════════════════════════════════════════════════════════════

def mutate_ironnet(genome, config):
    """Mutate weights, biases, and activations only. Never touch topology."""
    gc = config.genome_config
    act_choices = ['tanh', 'sin', 'cos', 'gauss', 'sech', 'dog', 'gabor', 'sinc', 'morlet']

    # Mutate connection weights
    for key, conn in genome.connections.items():
        r = np.random.random()
        if r < 0.1:  # replace weight
            conn.weight = np.random.uniform(-5, 5)
        elif r < 0.8:  # perturb weight
            conn.weight += np.random.normal(0, 0.5)
            conn.weight = max(-5.0, min(5.0, conn.weight))
        # Never disable: conn.enabled stays True

    # Mutate node biases and activations
    for nid, node in genome.nodes.items():
        r = np.random.random()
        if r < 0.1:  # replace bias
            node.bias = np.random.uniform(-5, 5)
        elif r < 0.7:  # perturb bias
            node.bias += np.random.normal(0, 0.5)
            node.bias = max(-5.0, min(5.0, node.bias))

        # Mutate activation (15% chance)
        if np.random.random() < 0.15:
            node.activation = act_choices[np.random.randint(4)]


def run_ironnet_islands(config_path, evaluator, n_islands=4, pop_per_island=150,
                        generations=200, save_dir=None, label="",
                        stall_limit=40, n_inputs=4, n_hidden=4):
    """Island model with fixed-topology IronNet genomes."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_path))
    register_activations(config)

    # Register gauss activation
    register_activations(config)

    islands = []
    best_ever = None
    best_ever_fitness = -999
    stall_counter = 0

    for i in range(n_islands):
        pop = build_ironnet_population(config, pop_per_island, n_inputs, n_hidden)
        islands.append({"pop": pop, "config": config, "best": None})

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for gen in range(generations):
        for i, island in enumerate(islands):
            pop = island["pop"]

            # Evaluate
            evaluator.evaluate(list(pop.items()), config)

            # Find best
            best = max(pop.values(), key=lambda g: g.fitness if g.fitness is not None else -999)
            island["best"] = best
            if best.fitness is not None and best.fitness > best_ever_fitness:
                best_ever = copy.deepcopy(best)
                best_ever_fitness = best.fitness
                stall_counter = 0

            # Selection + reproduction (tournament)
            new_pop = {}
            sorted_genomes = sorted(pop.values(),
                                     key=lambda g: g.fitness if g.fitness is not None else -999,
                                     reverse=True)
            # Elitism: keep top 3
            for j in range(min(3, len(sorted_genomes))):
                elite = copy.deepcopy(sorted_genomes[j])
                elite.key = j
                elite.fitness = None
                new_pop[j] = elite

            # Fill rest via tournament selection + mutation
            for j in range(3, pop_per_island):
                # Tournament of 3
                candidates = np.random.choice(len(sorted_genomes),
                                               size=min(3, len(sorted_genomes)),
                                               replace=False)
                winner_idx = min(candidates)  # lower index = higher fitness
                parent = copy.deepcopy(sorted_genomes[winner_idx])
                parent.key = j
                parent.fitness = None
                mutate_ironnet(parent, config)
                new_pop[j] = parent

            island["pop"] = new_pop

        # Migration every 10 gens
        if gen > 0 and gen % 10 == 0:
            for i in range(n_islands):
                src = islands[i]
                dst = islands[(i + 1) % n_islands]
                if src["best"] is not None:
                    worst_gid = min(dst["pop"],
                                    key=lambda g: dst["pop"][g].fitness
                                    if dst["pop"][g].fitness is not None else 999)
                    migrant = copy.deepcopy(src["best"])
                    migrant.key = worst_gid
                    migrant.fitness = None
                    dst["pop"][worst_gid] = migrant

        stall_counter += 1

        # Save checkpoint every generation
        if save_dir:
            gen_best = max((isl["best"] for isl in islands if isl["best"] is not None),
                           key=lambda g: g.fitness if g.fitness is not None else -999,
                           default=None)
            if gen_best:
                with open(f"{save_dir}/gen_{gen:03d}_best.pkl", "wb") as f:
                    pickle.dump({"genome": copy.deepcopy(gen_best),
                                 "config": config,
                                 "generation": gen,
                                 "fitness": gen_best.fitness}, f)

        fitnesses = [isl["best"].fitness for isl in islands
                     if isl["best"] is not None and isl["best"].fitness is not None]
        if fitnesses and gen % 10 == 0:
            print(f"  [{label}] Gen {gen:>3}: best={max(fitnesses):.4f} "
                  f"global={best_ever_fitness:.4f} stall={stall_counter}")

        # Early stopping on stall
        if stall_counter >= stall_limit:
            print(f"  [{label}] Early stop at gen {gen} (stalled {stall_limit} gens)")
            break

    return best_ever, config


# ═══════════════════════════════════════════════════════════════════════════
# Supervised pretrain evaluator
# ═══════════════════════════════════════════════════════════════════════════

class IronNetSupervisedEvaluator:
    def __init__(self, inputs_2d, mid_close, labels, pip, spread):
        self.inputs_2d = inputs_2d
        self.mid_close = mid_close
        self.labels = labels
        self.pip = pip
        self.spread = spread

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            try:
                net = extract_network(genome, config)
            except Exception:
                genome.fitness = -10.0
                continue
            result = evaluate_supervised_jit(
                self.inputs_2d, self.mid_close, self.labels,
                self.pip, self.spread, len(self.mid_close),
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10], 0)
            genome.fitness = float(result[0])


# ═══════════════════════════════════════════════════════════════════════════
# OOS evaluator (full pass, no chunks)
# ═══════════════════════════════════════════════════════════════════════════

def eval_oos(genome, config, inputs_2d, mid_close, pip, spread, max_hold=200):
    """Full OOS evaluation. Returns dict of metrics."""
    net = extract_network(genome, config)
    nt, pnl, mae, nl, ns = evaluate_chunk_jit(
        inputs_2d, mid_close,
        pip, spread, max_hold,
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
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IronNet fixed-topology per-pair training")
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
    parser.add_argument("--stall-limit", type=int, default=40,
                        help="Stop if no improvement for N gens")
    parser.add_argument("--min-dir-ratio", type=float, default=0.15,
                        help="Minimum minority direction fraction (0.15=15%%)")
    parser.add_argument("--tf", type=str, default="M5", choices=["M5", "H1"],
                        help="Timeframe: M5 (default) or H1")
    parser.add_argument("--mode", type=str, default="v3", choices=list(MODE_CONFIG.keys()),
                        help="Feature mode: v3, v7, setA-setF")
    args = parser.parse_args()

    np.random.seed(args.seed)
    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]
    min_swing_pips = args.min_swing if args.min_swing > 0 else PAIR_MIN_SWING.get(pair, 20)
    tf = args.tf
    mode = args.mode
    mcfg = MODE_CONFIG[mode]
    N_INPUTS = mcfg["n_inputs"]
    N_HIDDEN = mcfg["n_hidden"]
    ind_cols = mcfg["ind_cols"]

    # H1 adjustments: scale parameters from M5 (12 M5 bars = 1 H1 bar)
    if tf == "H1":
        min_swing_pips = int(min_swing_pips * 2.0)   # wider swings at H1
        if args.max_hold == 200:  # only override if default
            args.max_hold = 17  # 200 M5 bars = 16.7h → 17 H1 bars
        if args.label_window == 6:
            args.label_window = 3  # fewer bars at H1

    # Compute connection count for display
    if isinstance(N_HIDDEN, list):
        layers = [N_INPUTS] + N_HIDDEN + [N_OUTPUTS]
        n_conn = sum(layers[i] * layers[i+1] for i in range(len(layers)-1)) + N_INPUTS * N_OUTPUTS
        topo_str = "→".join(str(x) for x in layers)
    else:
        n_conn = N_INPUTS * N_HIDDEN + N_HIDDEN * N_OUTPUTS + N_INPUTS * N_OUTPUTS
        topo_str = f"{N_INPUTS}→{N_HIDDEN}→{N_OUTPUTS}"
    version_tag = mode.upper() if mode.startswith("set") else f"V{mode[1:]}" if mode != "v3" else "V3"
    print(f"{'='*65}")
    print(f"  IronNet {version_tag} Per-Pair: {pair} ({tf})")
    print(f"  Fixed topology: {topo_str} + skip ({n_conn} conn)")
    print(f"  Indicators: {', '.join(ind_cols)} + UPnL")
    print(f"  Activations: tanh, sin, cos, gauss")
    print(f"  Seed: {args.seed} | {args.islands}×{args.pop} pop")
    print(f"  Sine: {args.sine_gens}g → Zigzag: {args.pretrain_gens}g → WF P&L: {args.gens}g")
    print(f"  WF chunks: {N_CHUNKS} | Min dir ratio: {args.min_dir_ratio}")
    print(f"  Min swing: {min_swing_pips}p | Max hold: {args.max_hold} bars | Stall limit: {args.stall_limit}")
    print(f"{'='*65}")
    tg_send(f"🔩 IronNet {version_tag} {pair} {tf} s{args.seed}\n"
            f"{N_INPUTS}→{N_HIDDEN}→{N_OUTPUTS} fixed ({n_conn}c)\n"
            f"PT {args.pretrain_gens}g + WF {args.gens}g\n"
            f"Bidir ≥{args.min_dir_ratio*100:.0f}% | {N_CHUNKS} chunks")

    # Load data
    data_dir = Path(os.environ.get("ASI_MC_DATA_DIR",
                    str(PROJECT_ROOT / "data" / mcfg["data_dir_name"])))
    path = data_dir / f"{pair}{mcfg['parquet_suffix']}.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found")
        return
    df = pd.read_parquet(path, engine="pyarrow")

    # Resample to H1 if needed
    if tf == "H1":
        df = df.set_index("timestamp")
        agg = {"mid_close": "last"}
        for c in ind_cols:
            agg[c] = "last"
        df = df.resample("1h").agg(agg).dropna(subset=["mid_close"]).reset_index()
        print(f"  Resampled M5 → H1: {len(df):,} bars")

    mid = df["mid_close"].values.astype(np.float64)
    n = len(mid)
    split = int(n * 0.7)

    # Apply mode-specific normalization
    inputs = np.stack([df[c].values.astype(np.float64) for c in ind_cols], axis=0)
    for idx, (op, val) in mcfg.get("normalize", {}).items():
        if op == "mul":
            inputs[idx] = inputs[idx] * val
        elif op == "div_clip":
            inputs[idx] = np.clip(inputs[idx] / val, -1.0, 1.0)

    inputs_is = inputs[:, :split]
    mid_is = mid[:split]
    inputs_oos = inputs[:, split:]
    mid_oos = mid[split:]
    del df; gc.collect()

    print(f"\nData: {n:,} {tf} bars | IS: {split:,} | OOS: {n - split:,}")
    for i, col in enumerate(ind_cols):
        print(f"  {col:15s} range: [{inputs_is[i].min():.4f}, {inputs_is[i].max():.4f}]")
    print(f"  {'Price':15s} range: [{mid_is.min():.5f}, {mid_is.max():.5f}]")

    config_name = f"neat_config_{N_INPUTS}in_3out.ini"
    config_path = SCRIPT_DIR / config_name
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        return
    if mode == "v3":
        results_dir = RESULTS_DIR if tf == "M5" else SCRIPT_DIR / "results" / "ironnet_h1"
    else:
        results_dir = SCRIPT_DIR / "results" / f"ironnet_{mode}"
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"iron_{mode}_{tf}_{pair}_s{args.seed}"

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Sine wave pretrain (30 gens)
    # Uses IDENTICAL indicator code: compute_asi_mc + compute_er_norm
    # Sine center/amplitude/pip match the real pair's price scale
    # ══════════════════════════════════════════════════════════════
    print(f"\nPhase 1: Sine wave pretrain ({args.sine_gens} gens)...")

    # Import the exact same ASI-MC function used to build the parquets
    from lib.asi_indicator import compute_asi_mc
    # compute_er_norm is inlined above (identical to export_d_indicators.py + curator)

    # Generate sine at pair's price scale
    pair_center = float(mid_is.mean())
    if tf == "H1":
        amp_pips = 40   # ±40 pips — typical H1 swing
        n_sine = 5000   # ~208 days at H1
        sine_period = 42  # ~42 hours (same real time as M5 sine_period=500)
    else:
        amp_pips = 15   # ±15 pips — typical M5 swing
        n_sine = 50000
        sine_period = 500  # ~42 hours at M5

    rng = np.random.RandomState(args.seed)
    t_arr = np.arange(n_sine, dtype=np.float64)
    sine_mid = pair_center + amp_pips * pip * np.sin(2 * np.pi * t_arr / sine_period)
    sine_mid += rng.normal(0, 1.0 * pip, n_sine)  # 1 pip noise

    # Proper OHLC from mid
    sine_o = np.empty(n_sine, dtype=np.float64)
    sine_c = sine_mid.copy()
    sine_o[0] = sine_mid[0]
    for ii in range(1, n_sine):
        sine_o[ii] = sine_c[ii - 1]
    hl_noise = rng.uniform(0, 2 * pip, n_sine)
    sine_h = np.maximum(sine_o, sine_c) + hl_noise
    sine_l = np.minimum(sine_o, sine_c) - hl_noise

    if mode == "v3":
        # V3: ASI-MC indicators
        sine_mc_d, sine_mc_dd = compute_asi_mc(sine_o, sine_h, sine_l, sine_c, n_sine)
        sine_er = compute_er_norm(sine_mid, window=60)
        sine_inputs = np.stack([sine_mc_d, sine_mc_dd, sine_er], axis=0)
    else:
        # Generic: compute all unified indicators, select by ind_cols
        from export_unified_training_data import (compute_tec, compute_h1_slope,
            compute_bb_width, compute_stoch_d, compute_range_pos, compute_gap_norm,
            compute_macd_hist, compute_aroon_osc, compute_mc_d_a)
        from lib.swing_indicators import compute_all_swing_features
        from lib.asi_indicator import compute_asi as _compute_asi
        # Compute ASI + swing features for sine (needed by S1-S4 and E3b)
        from lib.asi_indicator import compute_asi as _compute_asi
        from lib.asi_indicator import compute_asi_mc as _compute_asi_mc
        from lib.swing_indicators import compute_all_swing_features
        _sine_mc_d, _sine_mc_dd = _compute_asi_mc(sine_o, sine_h, sine_l, sine_c, n_sine)
        _sine_er = compute_er_norm(sine_c, window=60)
        _sine_asi = _compute_asi(sine_o, sine_h, sine_l, sine_c, n_sine)
        _sine_sw = compute_all_swing_features(sine_o, sine_h, sine_l, sine_c, _sine_asi)

        all_sine_ind = {
            # ASI-MC indicators (S1-S4, E3b)
            "mc_d_a": _sine_mc_d,
            "mc_dd_a": _sine_mc_dd,
            "er_norm": _sine_er,
            "sb_a": _sine_sw["sb_a"].astype(np.float64),
            "hh_asi": _sine_sw["hh_asi"].astype(np.float64),
            "hl_asi": _sine_sw["hl_asi"].astype(np.float64),
            "hl_price": _sine_sw["hl_price"].astype(np.float64),
            # Technical indicators (setA-F)
            "tec_5": compute_tec(sine_c, 5),
            "bb_width": compute_bb_width(sine_c),
            "h1_slope": compute_h1_slope(sine_c, 12, 3),
            "stoch_d": compute_stoch_d(sine_h, sine_l, sine_c),
            "range_pos_30": compute_range_pos(sine_h, sine_l, 30),
            "gap_norm": compute_gap_norm(sine_o, sine_c, sine_h, sine_l),
            "macd_hist": compute_macd_hist(sine_c),
            "aroon_osc": compute_aroon_osc(sine_h, sine_l, 25),
        }
        sine_inputs = np.stack([all_sine_ind[c] for c in ind_cols], axis=0).astype(np.float64)
        # Apply normalization
        for idx, (op, val) in mcfg.get("normalize", {}).items():
            if op == "mul": sine_inputs[idx] = sine_inputs[idx] * val
            elif op == "div_clip": sine_inputs[idx] = np.clip(sine_inputs[idx] / val, -1.0, 1.0)

    print(f"  Sine: center={pair_center:.5f} amp=±{amp_pips}p period={sine_period}")
    for i, col in enumerate(ind_cols):
        print(f"  Sine {col:15s} range: [{sine_inputs[i, 200:].min():.4f}, {sine_inputs[i, 200:].max():.4f}]")

    # Sine labels: BUY near troughs, SELL near peaks (from zigzag on sine)
    sine_labels = generate_zigzag_labels(sine_mid, pip, amp_pips // 2,
                                          label_window=8, min_mfe_pips=2.0)
    s_buy = int(np.sum(sine_labels == 1))
    s_sell = int(np.sum(sine_labels == 2))
    print(f"  Sine labels: BUY={s_buy} SELL={s_sell}")

    sine_eval = IronNetSupervisedEvaluator(sine_inputs, sine_mid, sine_labels, pip, spread)
    save_s1 = str(results_dir / f"{tag}_sine_ckpt")

    t0 = time.time()
    sine_genome, _ = run_ironnet_islands(
        config_path, sine_eval,
        n_islands=args.islands, pop_per_island=args.pop,
        generations=args.sine_gens,
        save_dir=save_s1, label=f"{pair} SINE",
        stall_limit=args.sine_gens,
        n_inputs=N_INPUTS, n_hidden=N_HIDDEN)
    sine_elapsed = time.time() - t0
    print(f"  Sine pretrain done: fitness={sine_genome.fitness:.4f} ({sine_elapsed:.0f}s)")

    del sine_inputs, sine_mid, sine_labels
    gc.collect()

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Zigzag label pretrain on real data (50 gens)
    # Seeded from sine pretrained genome
    # ══════════════════════════════════════════════════════════════
    print(f"\nPhase 2: Zigzag pretrain ({args.pretrain_gens} gens)...")
    labels_is = generate_zigzag_labels(mid_is, pip, min_swing_pips,
                                        label_window=args.label_window,
                                        min_mfe_pips=spread + 2.0)
    n_buy = int(np.sum(labels_is == 1))
    n_sell = int(np.sum(labels_is == 2))
    print(f"  BUY={n_buy:,} SELL={n_sell:,} ({100 * (n_buy + n_sell) / len(labels_is):.1f}%)")

    if n_buy + n_sell < 100:
        reduced = max(min_swing_pips // 2, 5)
        labels_is = generate_zigzag_labels(mid_is, pip, reduced,
                                            label_window=args.label_window,
                                            min_mfe_pips=spread + 1.0)
        n_buy = int(np.sum(labels_is == 1))
        n_sell = int(np.sum(labels_is == 2))
        print(f"  Reduced to {reduced}p: BUY={n_buy:,} SELL={n_sell:,}")
        min_swing_pips = reduced

    zz_eval = IronNetSupervisedEvaluator(inputs_is, mid_is, labels_is, pip, spread)
    save_pt = str(results_dir / f"{tag}_zigzag_ckpt")

    # Seed from sine genome
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_path))
    register_activations(config)

    zz_islands = []
    for i in range(args.islands):
        pop = {}
        for j in range(args.pop):
            clone = copy.deepcopy(sine_genome)
            clone.key = j; clone.fitness = None
            mutate_ironnet(clone, config)
            pop[j] = clone
        zz_islands.append({"pop": pop, "config": config, "best": None})

    # Manual run loop for zigzag pretrain (seeded from sine)
    best_zz = None; best_zz_fit = -999
    Path(save_pt).mkdir(parents=True, exist_ok=True)
    t_pt = time.time()

    for gen in range(args.pretrain_gens):
        for i, island in enumerate(zz_islands):
            pop = island["pop"]
            zz_eval.evaluate(list(pop.items()), config)
            best = max(pop.values(), key=lambda g: g.fitness if g.fitness is not None else -999)
            island["best"] = best
            if best.fitness is not None and best.fitness > best_zz_fit:
                best_zz = copy.deepcopy(best); best_zz_fit = best.fitness

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
                src = zz_islands[i]; dst = zz_islands[(i + 1) % args.islands]
                if src["best"] is not None:
                    wg = min(dst["pop"], key=lambda g: dst["pop"][g].fitness if dst["pop"][g].fitness is not None else 999)
                    m = copy.deepcopy(src["best"]); m.key = wg; m.fitness = None; dst["pop"][wg] = m

        gen_best = max((isl["best"] for isl in zz_islands if isl["best"] is not None),
                       key=lambda g: g.fitness if g.fitness is not None else -999, default=None)
        if gen_best and save_pt:
            with open(f"{save_pt}/gen_{gen:03d}_best.pkl", "wb") as f:
                pickle.dump({"genome": copy.deepcopy(gen_best), "config": config,
                             "generation": gen, "fitness": gen_best.fitness}, f)
        if gen % 10 == 0:
            print(f"  [{pair} ZZ] Gen {gen:>3}: global={best_zz_fit:.4f}")

    pt_elapsed = time.time() - t_pt
    pretrained = best_zz
    print(f"  Zigzag pretrain done: fitness={best_zz_fit:.4f} ({pt_elapsed:.0f}s)")
    tg_send(f"✅ IronNet {pair} phases 1+2: sine={sine_genome.fitness:.4f} zz={best_zz_fit:.4f}")

    # ── Phase 3: WF P&L evolution ─────────────────────────────────
    print(f"\nPhase 3: WF P&L evolution ({args.gens} gens, {N_CHUNKS} chunks)...")
    wf_eval = IronNetWFEvaluator(inputs_is, mid_is, pip, spread,
                                  max_hold=args.max_hold,
                                  n_chunks=N_CHUNKS,
                                  min_dir_ratio=args.min_dir_ratio)
    save_ev = str(results_dir / f"{tag}_evolve_ckpt")

    # Seed all islands from pretrained genome
    t1 = time.time()

    # Build seeded islands manually
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_path))
    register_activations(config)

    islands_data = []
    for i in range(args.islands):
        pop = {}
        for j in range(args.pop):
            clone = copy.deepcopy(pretrained)
            clone.key = j
            clone.fitness = None
            mutate_ironnet(clone, config)
            pop[j] = clone
        islands_data.append({"pop": pop, "config": config, "best": None})

    # Run evolution loop
    best_ever = None
    best_ever_fitness = -999
    stall_counter = 0
    Path(save_ev).mkdir(parents=True, exist_ok=True)

    for gen in range(args.gens):
        for i, island in enumerate(islands_data):
            pop = island["pop"]
            wf_eval.evaluate(list(pop.items()), config)
            best = max(pop.values(), key=lambda g: g.fitness if g.fitness is not None else -999)
            island["best"] = best
            if best.fitness is not None and best.fitness > best_ever_fitness:
                best_ever = copy.deepcopy(best)
                best_ever_fitness = best.fitness
                stall_counter = 0

            # Selection + mutation
            sorted_g = sorted(pop.values(),
                               key=lambda g: g.fitness if g.fitness is not None else -999,
                               reverse=True)
            new_pop = {}
            for j in range(min(3, len(sorted_g))):
                e = copy.deepcopy(sorted_g[j])
                e.key = j; e.fitness = None
                new_pop[j] = e
            for j in range(3, args.pop):
                cands = np.random.choice(len(sorted_g), size=min(3, len(sorted_g)), replace=False)
                w = min(cands)
                p = copy.deepcopy(sorted_g[w])
                p.key = j; p.fitness = None
                mutate_ironnet(p, config)
                new_pop[j] = p
            island["pop"] = new_pop

        if gen > 0 and gen % 10 == 0:
            for i in range(args.islands):
                src = islands_data[i]
                dst = islands_data[(i + 1) % args.islands]
                if src["best"] is not None:
                    worst_gid = min(dst["pop"],
                                    key=lambda g: dst["pop"][g].fitness
                                    if dst["pop"][g].fitness is not None else 999)
                    migrant = copy.deepcopy(src["best"])
                    migrant.key = worst_gid; migrant.fitness = None
                    dst["pop"][worst_gid] = migrant

        stall_counter += 1

        if save_ev:
            gen_best = max((isl["best"] for isl in islands_data if isl["best"] is not None),
                           key=lambda g: g.fitness if g.fitness is not None else -999, default=None)
            if gen_best:
                with open(f"{save_ev}/gen_{gen:03d}_best.pkl", "wb") as f:
                    pickle.dump({"genome": copy.deepcopy(gen_best), "config": config,
                                 "generation": gen, "fitness": gen_best.fitness}, f)

        fitnesses = [isl["best"].fitness for isl in islands_data
                     if isl["best"] is not None and isl["best"].fitness is not None]
        if fitnesses and gen % 10 == 0:
            print(f"  [{pair} EV] Gen {gen:>3}: best={max(fitnesses):.4f} "
                  f"global={best_ever_fitness:.4f} stall={stall_counter}")

        if stall_counter >= args.stall_limit:
            print(f"  [{pair} EV] Early stop gen {gen} (stalled {args.stall_limit})")
            break

    ev_elapsed = time.time() - t1
    print(f"  Evolution done: fitness={best_ever_fitness:.4f} ({ev_elapsed:.0f}s)")

    # ── Phase 4: OOS evaluation ───────────────────────────────────
    print(f"\nPhase 4: OOS evaluation...")
    is_res = eval_oos(best_ever, config, inputs_is, mid_is, pip, spread, args.max_hold)
    oos_res = eval_oos(best_ever, config, inputs_oos, mid_oos, pip, spread, args.max_hold)

    print(f"\n{'='*65}")
    print(f"  RESULTS: {pair} (IronNet V3 Per-Pair)")
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

    # Network analysis
    n_enabled = sum(1 for c in best_ever.connections.values() if c.enabled)
    acts = [n.activation for n in best_ever.nodes.values()]
    print(f"  Connections: {n_enabled}/40 enabled | Activations: {dict((a, acts.count(a)) for a in set(acts))}")

    result_data = {
        "pair": pair, "variant": "ironnet",
        "seed": args.seed,
        "pretrain_gens": args.pretrain_gens,
        "evolution_gens": args.gens,
        "actual_gens": gen + 1 if stall_counter >= args.stall_limit else args.gens,
        "min_swing_pips": min_swing_pips,
        "n_chunks": N_CHUNKS,
        "min_dir_ratio": args.min_dir_ratio,
        "fitness": round(best_ever_fitness, 4),
        "network_size": list(best_ever.size()),
        "activations": dict((a, acts.count(a)) for a in set(acts)),
        "is": is_res,
        "oos": oos_res,
        "pretrain_time_s": round(pt_elapsed, 1),
        "evolve_time_s": round(ev_elapsed, 1),
    }

    with open(results_dir / f"{tag}_best.pkl", "wb") as f:
        pickle.dump({"genome": best_ever, "config": config, "pair": pair,
                     "result": result_data}, f)
    with open(results_dir / f"{tag}_result.json", "w") as f:
        json.dump(result_data, f, indent=2)

    tg_send(f"🔩 IronNet {pair} s{args.seed} DONE\n"
            f"Fitness: {best_ever_fitness:.4f}\n"
            f"OOS: {oos_res['n_trades']}T {oos_res['total_pnl']:+.1f}p "
            f"dir={oos_res['dir_ratio']:.2f}\n"
            f"({oos_res['pips_per_day']:.1f}p/day)")

    print(f"\nSaved: {tag}_best.pkl + {tag}_result.json")
    print(f"Total: {pt_elapsed + ev_elapsed:.0f}s")


if __name__ == "__main__":
    main()
# Added at EOF for reference — er_trade mode already in MODE_CONFIG
