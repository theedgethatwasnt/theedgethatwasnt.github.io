#!/usr/bin/env python3
"""
SwingDim IronNet Training
=========================
IronNet fixed-topology training with swing structure indicators.
Tests whether adding SB-A, ERP, and HHHL dimensions improves V3.

Experiment modes:
  E1  — 6 inputs: MC_D, MC_dD, ER_norm, SB_A, ERP_P, UPnL
        Tests: swing breakout context (ASI state + price range position)
  E2  — 7 inputs: MC_D, MC_dD, ER_norm, SB_A, HH_ASI, HL_ASI, UPnL
        Tests: swing breakout + HHHL structure (higher-high / higher-low)
  E3  — 10 inputs: MC_D, MC_dD, ER_norm, ERP_A, HH_ASI, HL_ASI, ERP_P, d_ERP_P, d_ERP_A, UPnL
        Tests: full swing superset + velocity features
        Topology: DEEP 10→10→20→12→3 fully connected (576 conn) — wider/deeper for 10 inputs

Topology:
  V3/E1/E2:  N→4→3 + skip (shallow, N inputs)
  E3:        10→10→20→12→3 fully connected, NO skip (deep, 576 connections)
    L1 hidden (10): IDs 100-109
    L2 hidden (20): IDs 200-219
    L3 hidden (12): IDs 300-311

V3 (control): 4→4→3 + skip = 40 connections (from train_ironnet_perpair.py)

Required parquet columns (from export_swing_indicators.py):
  sb_a, erp_a, hh_asi, hl_asi, erp_p, hh_price, hl_price

Usage:
  python3 train_swingdim.py --mode E1 --pair EUR_GBP --seed 42
  python3 train_swingdim.py --mode E2 --pair EUR_GBP --seed 137
  python3 train_swingdim.py --mode E3 --pair EUR_GBP --seed 42
  python3 train_swingdim.py --mode E1 --all-pairs --seed 42

Hetzner (deploy_swingdim.sh will run these):
  ASI_MC_DATA_DIR=/root/neat/data python3 train_swingdim.py --mode E1 --pair EUR_GBP --seed 42
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
                str(PROJECT_ROOT / "data" / "s5_training")))
RESULTS_DIR = SCRIPT_DIR / "results" / "swingdim"

# S5 cadence: 17,280 bars/day (24h × 3600s / 5s)
# M5 was 288 bars/day. Fitness pips/day calculation uses this.
BARS_PER_DAY = 17_280
S5_PER_M5    = 12     # 12 × 5s = 1 M5 bar

PAIR_PIP = {
    "EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
    "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
}
PAIR_SPREAD = {
    # Real mean spreads from S5 BA data (mean across all sessions)
    "EUR_JPY": 2.53, "USD_JPY": 1.77, "GBP_JPY": 3.69, "AUD_JPY": 2.26,
    "CAD_JPY": 2.54, "CHF_JPY": 3.89, "NZD_JPY": 2.94,
    "EUR_USD": 1.61, "GBP_USD": 1.99, "AUD_USD": 1.35,
    "NZD_USD": 1.53, "EUR_GBP": 1.49,
}
ALL_PAIRS = list(PAIR_PIP.keys())
PAIR_MIN_SWING = {
    "EUR_JPY": 30, "USD_JPY": 25, "GBP_JPY": 40, "AUD_JPY": 25,
    "CAD_JPY": 30, "CHF_JPY": 35, "NZD_JPY": 25,
    "EUR_USD": 20, "GBP_USD": 25, "AUD_USD": 18,
    "NZD_USD": 18, "EUR_GBP": 15,
}

N_HIDDEN = 4
N_OUTPUTS = 3  # BUY, SELL, FLATTEN
N_CHUNKS = 3   # WF chunks within IS

# Input column specifications per mode
# UPnL is added dynamically during evaluation — not stored in parquet
MODE_COLS = {
    "E1": ["mc_d_a", "mc_dd_a", "er_norm", "sb_a", "erp_p"],           # 5 + UPnL = 6
    "E2": ["mc_d_a", "mc_dd_a", "er_norm", "sb_a", "hh_asi", "hl_asi"],  # 6 + UPnL = 7
    # E3: full swing context + velocity (d_erp_p/d_erp_a = 1h rate-of-change)
    "E3": ["mc_d_a", "mc_dd_a", "er_norm", "erp_a", "hh_asi", "hl_asi",
           "erp_p", "d_erp_p", "d_erp_a"],                              # 9 + UPnL = 10
    "V3": ["mc_d_a", "mc_dd_a", "er_norm"],                              # 3 + UPnL = 4 (control)
    # E5: SB_A + str_diff_sign — two independently proven signals stacked on V3 core
    # SB_A: sensitivity confirmed dominant in E1/E2 (delta=-89p when zeroed)
    # str_diff_sign: D3 experiment +61K pips/12/12 pairs
    "E5": ["mc_d_a", "mc_dd_a", "er_norm", "sb_a", "str_diff_sign"],    # 5 + UPnL = 6
    # E6: SB_A + vol_regime — does regime filtering compound the swing signal?
    "E6": ["mc_d_a", "mc_dd_a", "er_norm", "sb_a", "vol_regime"],       # 5 + UPnL = 6
    # E7: SB_A + h1_sr_zone — multi-TF structure context on top of swing
    "E7": ["mc_d_a", "mc_dd_a", "er_norm", "sb_a", "h1_sr_zone"],       # 5 + UPnL = 6
    # Vmax: all validated candidates — train once, sensitivity finds winners
    "Vmax": ["mc_d_a", "mc_dd_a", "er_norm",          # core V3
             "sb_a", "hh_asi", "hl_asi",               # swing structure
             "erp_p", "erp_a", "d_erp_p", "d_erp_a",  # range position + velocity
             "str_diff_sign", "vol_regime", "h1_sr_zone"],  # 13 + UPnL = 14
    # ── Subset experiments (4 features + UPnL = 5 inputs, small/fast) ──────
    # All features bounded: mc_d_a/mc_dd_a ∈ [-1,+1], er_norm ∈ [0,~0.85],
    # sb_a ∈ {-1,-0.5,0,+0.5,+1}, hh_asi/hl_asi ∈ {0,1}
    # erp_p/erp_a/d_erp_p/d_erp_a DROPPED — unclamped, can reach ±62K (unusable)
    # S1: Momentum core — V3 + swing breakout
    "S1": ["mc_d_a", "mc_dd_a", "er_norm", "sb_a"],          # 4 + UPnL = 5
    # S2: Swing structure — HHHL pattern + regime
    "S2": ["sb_a", "hh_asi", "hl_asi", "er_norm"],            # 4 + UPnL = 5
    # S3: Direction + structure — momentum meets swing state
    "S3": ["sb_a", "mc_d_a", "mc_dd_a", "hh_asi"],           # 4 + UPnL = 5
    # S4: HHHL + regime — pure structural signals, no momentum
    "S4": ["hh_asi", "hl_asi", "mc_d_a", "er_norm"],          # 4 + UPnL = 5
}

# NEAT config files per N_INPUTS
CONFIG_FILES = {
    4:  "neat_config_4in_3out.ini",
    5:  "neat_config_5in_3out.ini",
    6:  "neat_config_6in_3out.ini",
    7:  "neat_config_7in_3out.ini",
    8:  "neat_config_8in_3out.ini",
    10: "neat_config_10in_3out.ini",
    14: "neat_config_14in_3out.ini",
}


# ─────────────────────────────────────────────────────────────────────────────
# Activation functions
# ─────────────────────────────────────────────────────────────────────────────

def gauss_activation(x):
    return math.exp(-x * x)

def sin_activation(x):
    return math.sin(x)

def cos_activation(x):
    return math.cos(x)

def tanh_activation(x):
    return math.tanh(x)

# Wavelet-family activations (proven by activation study)
def sech_activation(x):
    """1/cosh(x) — heavier-tailed bump than Gaussian, top performer in study."""
    c = math.cosh(x)
    return 1.0 / c if c != 0 else 0.0

def dog_activation(x):
    """Difference of Gaussians — bandpass filter, top performer in study."""
    return math.exp(-0.5 * x * x) - 0.5 * math.exp(-0.125 * x * x)

def gabor_activation(x):
    """Gabor wavelet — dominant in L2 with clean inputs."""
    return math.exp(-2.0 * x * x) * math.cos(2.0 * math.pi * x)

def sinc_activation(x):
    """Sinc function — best pure seed in activation study."""
    if abs(x) < 1e-6:
        return 1.0
    v = math.pi * x
    return math.sin(v) / v

def morlet_activation(x):
    """Morlet wavelet (imaginary part: sin × Gaussian envelope)."""
    return math.sin(x) * math.exp(-0.5 * x * x)


BASE_ACTIVATIONS = [
    ('gauss', gauss_activation), ('sin', sin_activation),
    ('cos', cos_activation), ('tanh', tanh_activation),
]
WAVELET_ACTIVATIONS = BASE_ACTIVATIONS + [
    ('sech', sech_activation), ('dog', dog_activation),
    ('gabor', gabor_activation), ('sinc', sinc_activation),
    ('morlet', morlet_activation),
]


def register_activations(config, wavelet=False):
    activations = WAVELET_ACTIVATIONS if wavelet else BASE_ACTIVATIONS
    for name, fn in activations:
        try:
            config.genome_config.add_activation(name, fn)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-topology genome builder (generalized for N inputs)
# ─────────────────────────────────────────────────────────────────────────────

def deep_topology(n_inputs):
    """Return (l1_size, l2_size, l3_size) for N→N→2N→(2N//3)→3 deep topology."""
    l1 = n_inputs
    l2 = 2 * n_inputs
    l3 = max(3, (2 * n_inputs) // 3)
    return l1, l2, l3


def topology_conn_count(n_inputs):
    l1, l2, l3 = deep_topology(n_inputs)
    return n_inputs*l1 + l1*l2 + l2*l3 + l3*3


def topology_str(n_inputs):
    l1, l2, l3 = deep_topology(n_inputs)
    return f"{n_inputs}→{l1}→{l2}→{l3}→3 ({topology_conn_count(n_inputs)} conn)"


def build_ironnet_genome(config, n_inputs):
    """Create deep fixed-topology genome: N→N→2N→(2N//3)→3 fully connected.

    Layer node IDs:
      L1 (N nodes):      100 … 100+N-1
      L2 (2N nodes):     200 … 200+2N-1
      L3 (2N//3 nodes):  300 … 300+(2N//3)-1
      Output:            0 (BUY), 1 (SELL), 2 (FLAT)
      Input:             -1 … -N (NEAT convention)

    Examples:
      N=4  (V3):   4→ 4→  8→3→3  (116 conn)
      N=6  (E1):   6→ 6→ 12→4→3  (228 conn)
      N=7  (E2):   7→ 7→ 14→5→3  (316 conn)
      N=10 (E3):  10→10→ 20→7→3  (561 conn)
      N=19 (Vmax):19→19→ 38→13→3 (2,068 conn)
    """
    genome = config.genome_type(0)
    genome.fitness = None
    gc = config.genome_config

    input_ids  = [-i - 1 for i in range(n_inputs)]
    output_ids = [0, 1, 2]
    act_choices = ['tanh', 'sin', 'cos', 'gauss', 'sech', 'dog', 'gabor', 'sinc', 'morlet']

    def _make_node(nid, act=None):
        node = gc.node_gene_type(nid)
        node.bias = np.random.uniform(-1, 1)
        node.response = 1.0
        node.activation = act if act else act_choices[np.random.randint(4)]
        node.aggregation = 'sum'
        return node

    l1_size, l2_size, l3_size = deep_topology(n_inputs)
    l1_ids = list(range(100, 100 + l1_size))
    l2_ids = list(range(200, 200 + l2_size))
    l3_ids = list(range(300, 300 + l3_size))

    for nid in output_ids:
        genome.nodes[nid] = _make_node(nid, 'tanh')
    for nid in l1_ids + l2_ids + l3_ids:
        genome.nodes[nid] = _make_node(nid)

    all_conn = []
    for src in input_ids:
        for dst in l1_ids: all_conn.append((src, dst))
    for src in l1_ids:
        for dst in l2_ids: all_conn.append((src, dst))
    for src in l2_ids:
        for dst in l3_ids: all_conn.append((src, dst))
    for src in l3_ids:
        for dst in output_ids: all_conn.append((src, dst))

    for innov, (src, dst) in enumerate(all_conn):
        key = (src, dst)
        conn = gc.connection_gene_type(key, innovation=innov)
        conn.weight = np.random.uniform(-2, 2)
        conn.enabled = True
        genome.connections[key] = conn

    return genome


def build_ironnet_population(config, pop_size, n_inputs):
    pop = {}
    for i in range(pop_size):
        g = build_ironnet_genome(config, n_inputs)
        g.key = i
        pop[i] = g
    return pop


# ─────────────────────────────────────────────────────────────────────────────
# IronNet mutation (weights + biases + activations only)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# JIT evaluator (reuse from train_ironnet_perpair.py — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def compute_er_norm(closes, window=60):
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
            # Spread already baked into entry_price — pnl/ae are clean from that basis
            pnl_pips = (mid_close[i] - entry_price) / pip * position
            ae = -pnl_pips   # positive when trade is underwater
            if ae > worst_ae:
                worst_ae = ae
        else:
            pnl_pips = 0.0

        for k in range(n_ind):
            values[k] = inputs_2d[k, i]
        values[n_ind] = np.tanh(pnl_pips / 20.0)

        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)
        out_buy  = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_flat = values[output_indices[2]]

        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid_close[i] - entry_price) / pip * position
            if n_trades < max_trades:
                trade_pnls[n_trades] = pnl
                trade_maes[n_trades] = worst_ae
                if position > 0: n_long += 1
                else: n_short += 1
                n_trades += 1
            position = 0; worst_ae = 0.0

        if position == 0:
            if out_buy > out_sell and out_buy > out_flat:
                # Long: enter at ask = mid + spread (immediately down by spread)
                position = 1; entry_price = mid_close[i] + spread_pips * pip
                entry_bar = i; worst_ae = spread_pips
            elif out_sell > out_buy and out_sell > out_flat:
                # Short: enter at bid = mid - spread (immediately down by spread)
                position = -1; entry_price = mid_close[i] - spread_pips * pip
                entry_bar = i; worst_ae = spread_pips
        else:
            close = False; new_pos = 0
            if out_flat > out_buy and out_flat > out_sell:
                close = True
            elif position == 1 and out_sell > out_buy and out_sell > out_flat:
                close = True; new_pos = -1
            elif position == -1 and out_buy > out_sell and out_buy > out_flat:
                close = True; new_pos = 1
            if close:
                pnl = (mid_close[i] - entry_price) / pip * position
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    trade_maes[n_trades] = worst_ae
                    if position > 0: n_long += 1
                    else: n_short += 1
                    n_trades += 1
                position = new_pos
                if new_pos == 1:
                    entry_price = mid_close[i] + spread_pips * pip
                elif new_pos == -1:
                    entry_price = mid_close[i] - spread_pips * pip
                else:
                    entry_price = 0.0
                entry_bar = i; worst_ae = spread_pips if new_pos != 0 else 0.0

    if position != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar - 1] - entry_price) / pip * position
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


@njit(cache=True)
def generate_zigzag_labels(mid_close, pip, min_swing_pips, label_window=6, min_mfe_pips=3.0):
    n = len(mid_close)
    labels = np.zeros(n, dtype=np.int64)
    min_swing = min_swing_pips * pip
    min_mfe = min_mfe_pips * pip
    running_high = mid_close[0]
    running_low  = mid_close[0]
    direction = 0
    for i in range(1, n):
        price = mid_close[i]
        if price > running_high: running_high = price
        if price < running_low:  running_low = price
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
    n_ind = inputs_2d.shape[0]
    end_bar = min(start_offset + max_bars, inputs_2d.shape[1] - 1)
    start_bar = max(start_offset + 10, 10)
    correct = 0; wrong_dir = 0; profitable_correct = 0; total_labeled = 0
    position = 0; entry_price = 0.0

    for i in range(start_bar, end_bar):
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position
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
        if out_buy > out_sell and out_buy > out_flat:   net_action = 1
        elif out_sell > out_buy and out_sell > out_flat: net_action = 2
        else:                                            net_action = 0
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
            position = 1; entry_price = mid_close[i] + spread_pips * pip
        elif net_action == 2 and position >= 0:
            position = -1; entry_price = mid_close[i] - spread_pips * pip
        elif net_action == 0 and position != 0:
            position = 0; entry_price = 0.0
    if total_labeled < 10:
        return 0.0, 0, 0, 0, 0   # too few labels — neutral score, don't penalise
    accuracy = correct / total_labeled
    wrong_rate = wrong_dir / total_labeled
    profit_rate = profitable_correct / max(correct, 1)
    fitness = accuracy * 10.0 + profit_rate * 5.0 - wrong_rate * 8.0
    return fitness, total_labeled, correct, wrong_dir, profitable_correct


# ─────────────────────────────────────────────────────────────────────────────
# WF P&L evaluator
# ─────────────────────────────────────────────────────────────────────────────

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
            return -999.0
        chunk_scores = []
        total_long = total_short = total_trades = 0
        for ci in range(self.n_chunks):
            c_start = int(self.n_bars * ci / self.n_chunks)
            c_end   = int(self.n_bars * (ci + 1) / self.n_chunks)
            nt, pnl, mae, nl, ns = evaluate_chunk_jit(
                self.inputs_2d, self.mid_close,
                self.pip, self.spread, self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10],
                c_start, c_end)
            total_long += nl; total_short += ns; total_trades += nt
            if nt < 3:
                return -999.0
            n_days = (c_end - c_start) / BARS_PER_DAY
            score = pnl / max(n_days, 1.0)   # pips/day this chunk (S5 cadence)
            chunk_scores.append(score)
        if total_trades < 6:
            return -999.0
        # Require bidirectional trading — all-one-direction = -999
        if total_long == 0 or total_short == 0:
            return -999.0
        # Fitness = worst chunk pips/day (WF: must be profitable in every period)
        return min(chunk_scores)


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


# ─────────────────────────────────────────────────────────────────────────────
# Island evolution loop (generalized)
# ─────────────────────────────────────────────────────────────────────────────

def run_islands(config_path, evaluator, n_inputs,
                n_islands=4, pop_per_island=150, generations=200,
                save_dir=None, label="", stall_limit=40,
                seed_genome=None, wavelet=False):
    """Island model evolution. Returns (best_genome, config)."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_path))
    register_activations(config, wavelet=wavelet)

    islands = []
    for i in range(n_islands):
        if seed_genome is not None:
            pop = {}
            for j in range(pop_per_island):
                clone = copy.deepcopy(seed_genome)
                clone.key = j; clone.fitness = None
                mutate_ironnet(clone, config)
                pop[j] = clone
        else:
            pop = build_ironnet_population(config, pop_per_island, n_inputs)
        islands.append({"pop": pop, "best": None})

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    best_ever = None
    best_ever_fitness = -999
    stall_counter = 0

    for gen in range(generations):
        for island in islands:
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
                    wg = min(dst["pop"],
                             key=lambda g: dst["pop"][g].fitness
                             if dst["pop"][g].fitness is not None else 999)
                    migrant = copy.deepcopy(src["best"])
                    migrant.key = wg; migrant.fitness = None
                    dst["pop"][wg] = migrant

        stall_counter += 1

        if save_dir:
            gen_best = max((isl["best"] for isl in islands if isl["best"] is not None),
                           key=lambda g: g.fitness if g.fitness is not None else -999,
                           default=None)
            if gen_best:
                with open(f"{save_dir}/gen_{gen:03d}_best.pkl", "wb") as f:
                    pickle.dump({"genome": copy.deepcopy(gen_best), "config": config,
                                 "generation": gen, "fitness": gen_best.fitness}, f)

        if gen % 10 == 0:
            fitnesses = [isl["best"].fitness for isl in islands
                         if isl["best"] is not None and isl["best"].fitness is not None]
            if fitnesses:
                # Show L/S split of best genome so we can see bidirectionality during training
                bidir = ""
                if best_ever is not None and hasattr(evaluator, 'inputs_2d'):
                    try:
                        _net = extract_network(best_ever, config)
                        _n = evaluator.inputs_2d.shape[1]
                        _nt, _pnl, _mae, _nl, _ns = evaluate_chunk_jit(
                            evaluator.inputs_2d, evaluator.mid_close,
                            evaluator.pip, evaluator.spread, evaluator.max_hold,
                            _net[0], _net[2], _net[3], _net[4], _net[5], _net[6],
                            _net[7], _net[8], _net[9], _net[10], 0, _n)
                        _ppd = _pnl / max(_n / BARS_PER_DAY, 1.0)
                        bidir = (f" | L={_nl} S={_ns} T={_nt} "
                                 f"{'✓bidir' if _nl>0 and _ns>0 else '✗UNIDIRECTIONAL'}"
                                 f" {_ppd:+.1f}p/d")
                    except Exception:
                        pass
                print(f"  [{label}] Gen {gen:>3}: best={max(fitnesses):.4f} "
                      f"global={best_ever_fitness:.4f} stall={stall_counter}{bidir}")

        # Only stall-stop when we've found something real (fitness > -5)
        # Prevents premature stop when entire population is degenerate
        if stall_counter >= stall_limit and best_ever_fitness > -5.0:
            print(f"  [{label}] Early stop at gen {gen} (stalled {stall_limit} gens, best={best_ever_fitness:.4f})")
            break

    return best_ever, config


# ─────────────────────────────────────────────────────────────────────────────
# OOS evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_oos(genome, config, inputs_2d, mid_close, pip, spread, max_hold=200):
    net = extract_network(genome, config)
    nt, pnl, mae, nl, ns = evaluate_chunk_jit(
        inputs_2d, mid_close, pip, spread, max_hold,
        net[0], net[2], net[3], net[4], net[5], net[6],
        net[7], net[8], net[9], net[10],
        0, inputs_2d.shape[1])
    n_days = inputs_2d.shape[1] / BARS_PER_DAY
    return {
        "n_trades": int(nt), "total_pnl": round(float(pnl), 1),
        "avg_mae": round(float(mae), 2),
        "n_long": int(nl), "n_short": int(ns),
        "pips_per_day": round(float(pnl) / max(n_days, 1), 1),
        "dir_ratio": round(min(nl, ns) / max(nt, 1), 3),
    }


def sensitivity_analysis(pkl_path, data_dir=None):
    """Freeze a trained genome and zero out each input in turn.
    Measures OOS P&L delta → trading-relevant feature importance.

    Usage:
        python3 train_swingdim.py --sensitivity results/swingdim/swingdim_E2_EUR_GBP_s42_best.pkl
    """
    import pandas as pd

    data = pickle.load(open(pkl_path, "rb"))
    genome  = data["genome"]
    config  = data["config"]
    pair    = data["pair"]
    mode    = data["mode"]
    ind_cols = data["ind_cols"]
    n_inputs = data["n_inputs"]
    labels  = ind_cols + ["UPnL"]

    _data_dir = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_parquet(_data_dir / f"{pair}_asi_mc.parquet", engine="pyarrow")

    pip    = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]
    n_bars = len(df)
    oos_start = int(n_bars * 0.7)

    df_oos = df.iloc[oos_start:].reset_index(drop=True)
    mid_oos = df_oos["mid_close"].values.astype(np.float64)

    # Build OOS input matrix (shape: n_inputs × n_bars_oos)
    upnl_col = np.zeros(len(df_oos), dtype=np.float64)  # UPnL=0 for static eval
    base_cols = [df_oos[c].fillna(0).values.astype(np.float64) for c in ind_cols] + [upnl_col]
    inputs_base = np.stack(base_cols, axis=0)  # (n_inputs, n_bars)

    # Baseline OOS
    baseline = eval_oos(genome, config, inputs_base, mid_oos, pip, spread)
    base_pnl = baseline["pips_per_day"]

    print(f"\nSensitivity Analysis: {mode} {pair}")
    print(f"Topology: {topology_str(n_inputs)}")
    print(f"Baseline OOS: {base_pnl:.1f} p/day  |  {baseline['n_trades']} trades  |  MAE={baseline['avg_mae']:.2f}")
    print(f"\n{'Feature':<14} {'Zeroed p/day':>13} {'Delta':>8} {'Impact':>8}  Bar")
    print("-" * 60)

    results = []
    for i, lbl in enumerate(labels):
        ablated = inputs_base.copy()
        ablated[i, :] = 0.0          # zero out this input
        res = eval_oos(genome, config, ablated, mid_oos, pip, spread)
        delta = res["pips_per_day"] - base_pnl
        results.append((lbl, res["pips_per_day"], delta))

    results.sort(key=lambda x: x[2])  # most damaging first
    for lbl, zeroed, delta in results:
        sign = "+" if delta >= 0 else ""
        bar_len = max(0, int(abs(delta) / max(abs(results[0][2]), 1) * 20))
        bar = ("▓" if delta < 0 else "░") * bar_len
        print(f"  {lbl:<12}  {zeroed:>10.1f}   {sign}{delta:>6.1f}   {sign}{delta:>6.1f}  {bar}")

    print(f"\n  Most critical: {results[0][0]} (delta={results[0][2]:.1f} p/day when zeroed)")
    print(f"  Least impact:  {results[-1][0]} (delta={results[-1][2]:+.1f} p/day when zeroed)")
    return results


def tg_send(text):
    try:
        import requests
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
        if token and chat:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text, "parse_mode": "HTML"},
                          timeout=5)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# NEAT config generation (builds .ini files for new N_INPUTS values)
# ─────────────────────────────────────────────────────────────────────────────

NEAT_CONFIG_TEMPLATE = """[NEAT]
fitness_criterion     = max
fitness_threshold     = 999
no_fitness_termination = False
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
num_inputs            = {n_inputs}
num_outputs           = 3
num_hidden            = 0
initial_connection    = full
feed_forward          = True
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob         = {conn_add_prob}
conn_delete_prob      = {conn_delete_prob}
node_add_prob         = {node_add_prob}
node_delete_prob      = {node_delete_prob}
activation_default    = tanh
activation_mutate_rate = 0.15
activation_options    = {activation_options}
aggregation_default   = sum
aggregation_mutate_rate = 0.0
aggregation_options   = sum
bias_init_mean        = 0.0
bias_init_stdev       = 1.0
bias_max_value        = 30.0
bias_min_value        = -30.0
bias_mutate_power     = 0.5
bias_mutate_rate      = 0.7
bias_replace_rate     = 0.1
response_init_mean    = 1.0
response_init_stdev   = 0.0
response_max_value    = 30.0
response_min_value    = -30.0
response_mutate_power = 0.0
response_mutate_rate  = 0.0
response_replace_rate = 0.0
weight_init_mean      = 0.0
weight_init_stdev     = 1.0
weight_max_value      = 30.0
weight_min_value      = -30.0
weight_mutate_power   = 0.5
weight_mutate_rate    = 0.8
weight_replace_rate   = 0.1
enabled_default       = True
enabled_mutate_rate   = 0.0

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""


def ensure_neat_config(n_inputs, wavelet=False, free_phase=False):
    """Create neat config for n_inputs if not present."""
    act_opts = "tanh sin cos gauss sech dog gabor sinc morlet" if wavelet else "tanh sin cos gauss"
    suffix = ("_wav" if wavelet else "") + ("_free" if free_phase else "")
    base = f"neat_config_{n_inputs}in_3out{suffix}.ini"
    path = SCRIPT_DIR / base
    if not path.exists():
        print(f"  Creating NEAT config: {base}")
        path.write_text(NEAT_CONFIG_TEMPLATE.format(
            n_inputs=n_inputs,
            activation_options=act_opts,
            conn_add_prob=0.05 if free_phase else 0.0,
            conn_delete_prob=0.05 if free_phase else 0.0,
            node_add_prob=0.03 if free_phase else 0.0,
            node_delete_prob=0.03 if free_phase else 0.0,
        ))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair training orchestration
# ─────────────────────────────────────────────────────────────────────────────

def train_pair(pair, mode, args):
    """Full training pipeline for one pair in one mode."""
    pip    = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]
    min_swing_pips = args.min_swing if args.min_swing > 0 else PAIR_MIN_SWING.get(pair, 20)

    ind_cols = MODE_COLS[mode]
    n_ind    = len(ind_cols)
    n_inputs = n_ind + 1   # +1 for UPnL

    tag = f"swingdim_{mode}_{pair}_s{args.seed}"
    results_pair_dir = RESULTS_DIR / tag
    results_pair_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SwingDim {mode}: {pair}")
    print(f"  Inputs ({n_inputs}): {ind_cols} + UPnL")
    print(f"  Topology: {topology_str(n_inputs)}")
    wavelet = getattr(args, 'wavelet', False)
    free_phase_gens = getattr(args, 'free_phase', 0)
    act_str = "tanh sin cos gauss sech dog gabor sinc morlet" if wavelet else "tanh sin cos gauss"
    print(f"  Activations: {act_str}")
    print(f"  Seed: {args.seed} | {args.islands}×{args.pop} pop")
    print(f"  Sine: {args.sine_gens}g → Zigzag: {args.pretrain_gens}g → WF P&L: {args.gens}g")
    print(f"{'='*65}")

    # Load data — S5-cadence parquet (export_s5_training_data.py)
    path = DATA_DIR / f"{pair}_s5_training.parquet"
    if not path.exists():
        print(f"  ERROR: {path} not found")
        print(f"  Run: python3 export_s5_training_data.py --pairs {pair}")
        return None

    df = pd.read_parquet(path, engine="pyarrow")

    # Verify required columns exist
    missing = [c for c in ind_cols if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns {missing}")
        print(f"  Run: python3 export_s5_training_data.py --pairs {pair}")
        return None

    mid = df["s5_mid"].values.astype(np.float64)   # S5 mid price for entry/exit
    n = len(mid)
    split = int(n * 0.7)

    inputs = np.stack([df[c].values.astype(np.float64) for c in ind_cols], axis=0)
    inputs_is  = inputs[:, :split]
    mid_is     = mid[:split]
    inputs_oos = inputs[:, split:]
    mid_oos    = mid[split:]
    del df; gc.collect()

    n_m5_total = n // S5_PER_M5
    print(f"\nData: {n:,} S5 bars ({n_m5_total:,} M5) | IS: {split:,} | OOS: {n - split:,}")
    for i, col in enumerate(ind_cols):
        print(f"  {col:<14}: [{inputs_is[i].min():.4f}, {inputs_is[i].max():.4f}]")

    config_path = ensure_neat_config(n_inputs, wavelet=wavelet)

    # ── Phase 1: Sine wave pretrain ────────────────────────────────
    print(f"\nPhase 1: Sine wave pretrain ({args.sine_gens} gens)...")
    from lib.asi_indicator import compute_asi_mc, compute_asi
    from lib.swing_indicators import compute_all_swing_features

    pair_center = float(mid_is.mean())
    amp_pips    = 15
    # Sine at S5 cadence: same physical duration as M5 pretrain
    # 500 M5 periods × 12 S5/M5 = 6000 S5 bars per sine cycle (~8.3h)
    sine_period_s5 = 6000
    n_sine_m5  = 1000           # ~2.9 days of M5 bars
    n_sine     = n_sine_m5 * S5_PER_M5   # equivalent S5 bars
    rng = np.random.RandomState(args.seed)

    # Generate S5 sine mid prices
    t_arr    = np.arange(n_sine, dtype=np.float64)
    sine_mid = pair_center + amp_pips * pip * np.sin(2 * np.pi * t_arr / sine_period_s5)
    sine_mid += rng.normal(0, 0.5 * pip, n_sine)

    # Build S5 OHLC: open = prev close, hl = open/close ± small noise
    sine_c = sine_mid.copy()
    sine_o = np.empty(n_sine, dtype=np.float64)
    sine_o[0] = sine_c[0]
    sine_o[1:] = sine_c[:-1]
    hl_noise = rng.uniform(0, pip, n_sine)
    sine_h = np.maximum(sine_o, sine_c) + hl_noise
    sine_l = np.minimum(sine_o, sine_c) - hl_noise

    # Aggregate S5 → M5 for indicator computation
    n_sine_m5_actual = n_sine // S5_PER_M5
    sm5_o = np.array([sine_o[i*S5_PER_M5] for i in range(n_sine_m5_actual)])
    sm5_h = np.array([sine_h[i*S5_PER_M5:(i+1)*S5_PER_M5].max() for i in range(n_sine_m5_actual)])
    sm5_l = np.array([sine_l[i*S5_PER_M5:(i+1)*S5_PER_M5].min() for i in range(n_sine_m5_actual)])
    sm5_c = np.array([sine_c[(i+1)*S5_PER_M5-1] for i in range(n_sine_m5_actual)])

    # Compute all M5-cadence indicators on synthetic data — same batch functions as export
    sine_mc_d_m5, sine_mc_dd_m5 = compute_asi_mc(sm5_o, sm5_h, sm5_l, sm5_c, n_sine_m5_actual)
    sine_er_m5  = compute_er_norm(sm5_c, window=60)
    sine_asi_m5 = compute_asi(sm5_o, sm5_h, sm5_l, sm5_c, n_sine_m5_actual)
    sine_sw     = compute_all_swing_features(sm5_o, sm5_h, sm5_l, sm5_c, sine_asi_m5)

    def _ffill_m5_to_s5(m5_arr):
        """Forward-fill M5 array to S5 cadence — identical to export pipeline."""
        out = np.empty(n_sine, dtype=np.float64)
        n = len(m5_arr)
        for i in range(n):
            v = m5_arr[i]
            out[i*S5_PER_M5:(i+1)*S5_PER_M5] = v if not np.isnan(v) else 0.0
        return out

    def _ffill_sw(key):
        arr = sine_sw.get(key, np.zeros(n_sine_m5_actual))
        last = 0.0
        out_m5 = np.empty(n_sine_m5_actual, dtype=np.float64)
        for i in range(n_sine_m5_actual):
            v = arr[i] if i < len(arr) else 0.0
            if not np.isnan(v):
                last = v
            out_m5[i] = last
        return _ffill_m5_to_s5(out_m5)

    col_to_sine = {
        "mc_d_a":  _ffill_m5_to_s5(sine_mc_d_m5),
        "mc_dd_a": _ffill_m5_to_s5(sine_mc_dd_m5),
        "er_norm": _ffill_m5_to_s5(sine_er_m5),
        "sb_a":    _ffill_sw("sb_a"),
        "hh_asi":  _ffill_sw("hh_asi"),
        "hl_asi":  _ffill_sw("hl_asi"),
    }
    sine_inputs = np.stack([col_to_sine[c] for c in ind_cols], axis=0)
    sine_labels = generate_zigzag_labels(sine_mid, pip, amp_pips // 2,
                                          label_window=8, min_mfe_pips=2.0)
    s_buy = int(np.sum(sine_labels == 1)); s_sell = int(np.sum(sine_labels == 2))
    print(f"  Sine labels: BUY={s_buy} SELL={s_sell}")

    sine_eval = IronNetSupervisedEvaluator(sine_inputs, sine_mid, sine_labels, pip, spread)
    t0 = time.time()
    sine_genome, _ = run_islands(
        config_path, sine_eval, n_inputs,
        n_islands=args.islands, pop_per_island=args.pop,
        generations=args.sine_gens,
        save_dir=str(results_pair_dir / "sine_ckpt"),
        label=f"{pair}_{mode} SINE", stall_limit=args.sine_gens,
        wavelet=wavelet)
    print(f"  Sine pretrain: fitness={sine_genome.fitness:.4f} ({time.time()-t0:.0f}s)")
    del sine_inputs, sine_mid, sine_mc_d_m5, sine_mc_dd_m5, sine_er_m5, sine_labels
    del sm5_o, sm5_h, sm5_l, sm5_c, sine_asi_m5, sine_sw, col_to_sine
    gc.collect()

    # ── Phase 2: Zigzag label pretrain on real IS ──────────────────
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

    # Cap ZZ pretrain to 200K S5 bars (~11 days) — enough label density, 23× faster per gen
    ZZ_MAX_BARS = 200_000
    if inputs_is.shape[1] > ZZ_MAX_BARS:
        inputs_zz  = inputs_is[:, :ZZ_MAX_BARS]
        mid_zz     = mid_is[:ZZ_MAX_BARS]
        labels_zz  = labels_is[:ZZ_MAX_BARS]
        print(f"  ZZ pretrain capped at {ZZ_MAX_BARS:,} bars (of {inputs_is.shape[1]:,})")
    else:
        inputs_zz, mid_zz, labels_zz = inputs_is, mid_is, labels_is
    zz_eval = IronNetSupervisedEvaluator(inputs_zz, mid_zz, labels_zz, pip, spread)
    t0 = time.time()
    pretrained, _ = run_islands(
        config_path, zz_eval, n_inputs,
        n_islands=args.islands, pop_per_island=args.pop,
        generations=args.pretrain_gens,
        save_dir=str(results_pair_dir / "zigzag_ckpt"),
        label=f"{pair}_{mode} ZZ", stall_limit=args.pretrain_gens,
        seed_genome=sine_genome, wavelet=wavelet)
    print(f"  Zigzag pretrain: fitness={pretrained.fitness:.4f} ({time.time()-t0:.0f}s)")
    del labels_is
    tg_send(f"[SwingDim] {mode} {pair} s{args.seed}: sine+zz done, zz={pretrained.fitness:.4f}")

    # ── Phase 3: WF P&L evolution ──────────────────────────────────
    print(f"\nPhase 3: WF P&L ({args.gens} gens, {N_CHUNKS} chunks)...")
    wf_eval = IronNetWFEvaluator(inputs_is, mid_is, pip, spread,
                                  max_hold=args.max_hold, n_chunks=N_CHUNKS,
                                  min_dir_ratio=args.min_dir_ratio)
    t0 = time.time()
    winner, config = run_islands(
        config_path, wf_eval, n_inputs,
        n_islands=args.islands, pop_per_island=args.pop,
        generations=args.gens,
        save_dir=str(results_pair_dir / "evolve_ckpt"),
        label=f"{pair}_{mode} WF", stall_limit=args.stall_limit,
        seed_genome=None, wavelet=wavelet)   # random init: avoid all-LONG ZZ seed poisoning
    evol_elapsed = time.time() - t0
    print(f"  WF evolution: fitness={winner.fitness:.4f} ({evol_elapsed:.0f}s)")

    # ── Phase 4 (optional): Free topology pruning ──────────────────
    if free_phase_gens > 0:
        print(f"\nPhase 4: Free topology pruning ({free_phase_gens} gens)...")
        free_config_path = ensure_neat_config(n_inputs, wavelet=wavelet, free_phase=True)
        t0 = time.time()
        winner, config = run_islands(
            free_config_path, wf_eval, n_inputs,
            n_islands=args.islands, pop_per_island=args.pop,
            generations=free_phase_gens,
            save_dir=str(results_pair_dir / "free_ckpt"),
            label=f"{pair}_{mode} FREE", stall_limit=free_phase_gens,
            seed_genome=winner, wavelet=wavelet)
        print(f"  Free phase: fitness={winner.fitness:.4f} ({time.time()-t0:.0f}s)")

    # ── OOS evaluation ─────────────────────────────────────────────
    print(f"\nOOS evaluation...")
    oos = eval_oos(winner, config, inputs_oos, mid_oos, pip, spread, max_hold=args.max_hold)
    print(f"  Trades: {oos['n_trades']} | Total: {oos['total_pnl']:.1f}p | "
          f"p/day: {oos['pips_per_day']:.1f} | MAE: {oos['avg_mae']:.2f}")
    print(f"  Long: {oos['n_long']} | Short: {oos['n_short']} | Dir: {oos['dir_ratio']:.3f}")

    # Save best genome
    save_path = RESULTS_DIR / f"{tag}_best.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "genome": winner, "config": config,
            "pair": pair, "mode": mode, "seed": args.seed,
            "fitness": winner.fitness,
            "ind_cols": ind_cols,
            "n_inputs": n_inputs,
            "oos": oos,
        }, f)
    print(f"  Saved: {save_path.name}")

    # Save JSON summary
    summary = {
        "pair": pair, "mode": mode, "seed": args.seed,
        "n_inputs": n_inputs, "ind_cols": ind_cols,
        "fitness": float(winner.fitness),
        "oos": oos,
    }
    with open(results_pair_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Auto sensitivity analysis ──────────────────────────────────
    try:
        sens = sensitivity_analysis(str(save_path))
        # Append to JSON summary
        summary["sensitivity"] = [
            {"feature": lbl, "zeroed_pday": round(zpd, 1), "delta": round(d, 1)}
            for lbl, zpd, d in sens
        ]
        with open(results_pair_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"  [sensitivity] skipped: {e}")

    tg_send(f"[SwingDim] {mode} {pair} s{args.seed} DONE\n"
            f"fitness={winner.fitness:.4f}\n"
            f"OOS {oos['pips_per_day']:.1f}p/day | {oos['n_trades']}T")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SwingDim IronNet training (E1/E2/E3)")
    parser.add_argument("--sensitivity", default=None, metavar="PKL",
                        help="Run sensitivity analysis on a saved genome pkl. Skips training.")
    parser.add_argument("--mode", default=None,
                        choices=["E1", "E2", "E3", "V3", "E5", "E6", "E7", "Vmax",
                                 "S1", "S2", "S3", "S4"],
                        help="Experiment mode. S1-S4: 4-feature subsets (5 inputs, fast)")
    parser.add_argument("--pair", default=None,
                        help="Single pair (e.g. EUR_GBP). Mutually exclusive with --all-pairs")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Train all 12 pairs sequentially")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--sine-gens",    type=int,   default=30)
    parser.add_argument("--pretrain-gens",type=int,   default=50)
    parser.add_argument("--gens",         type=int,   default=200)
    parser.add_argument("--islands",      type=int,   default=4)
    parser.add_argument("--pop",          type=int,   default=150)
    parser.add_argument("--max-hold",     type=int,   default=2400,
                        help="Max hold in S5 bars. 2400=3.3h, 720=1h, 144=12min")
    parser.add_argument("--min-swing",    type=int,   default=0)
    parser.add_argument("--label-window", type=int,   default=6)
    parser.add_argument("--stall-limit",  type=int,   default=40)
    parser.add_argument("--min-dir-ratio",type=float, default=0.15)
    parser.add_argument("--wavelet", action="store_true",
                        help="Enable wavelet activations (sech/dog/gabor/sinc/morlet) in addition to base set")
    parser.add_argument("--free-phase", type=int, default=0, metavar="GENS",
                        help="After WF convergence, run N gens with topology changes enabled (pruning phase)")
    args = parser.parse_args()

    # Sensitivity mode — no training needed
    if args.sensitivity:
        sensitivity_analysis(args.sensitivity)
        return

    if not args.mode:
        parser.error("--mode is required unless using --sensitivity")

    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = ALL_PAIRS if args.all_pairs else ([args.pair] if args.pair else [])
    if not pairs:
        parser.error("Specify --pair PAIR or --all-pairs")

    print(f"SwingDim {args.mode}: {len(pairs)} pairs, seed={args.seed}")
    print(f"Mode inputs: {MODE_COLS[args.mode]} + UPnL = {len(MODE_COLS[args.mode])+1} total")
    print(f"Data dir: {DATA_DIR}")
    print()

    all_results = []
    t_total = time.time()
    for pair in pairs:
        result = train_pair(pair, args.mode, args)
        if result:
            all_results.append(result)
        gc.collect()

    # Print comparison table
    if all_results:
        print(f"\n{'='*70}")
        print(f"SwingDim {args.mode} Results (seed={args.seed})")
        print(f"{'Pair':<12} {'p/day':>8} {'Trades':>7} {'Fitness':>10} {'MAE':>6}")
        print(f"{'-'*50}")
        for r in sorted(all_results, key=lambda x: x["oos"]["pips_per_day"], reverse=True):
            o = r["oos"]
            print(f"  {r['pair']:<10}  {o['pips_per_day']:>7.1f}  {o['n_trades']:>7}  "
                  f"{r['fitness']:>10.4f}  {o['avg_mae']:>5.2f}")
        avg = sum(r["oos"]["pips_per_day"] for r in all_results) / len(all_results)
        print(f"  {'AVG':<10}  {avg:>7.1f}")
        print(f"\nV3 baseline avg (reference): 42.0 p/day")
        delta = avg - 42.0
        sign = "+" if delta >= 0 else ""
        print(f"SwingDim {args.mode} delta:  {sign}{delta:.1f} p/day vs V3")

    total = time.time() - t_total
    print(f"\nTotal time: {total:.0f}s ({total/3600:.1f}h)")

    # Save combined results
    with open(RESULTS_DIR / f"swingdim_{args.mode}_s{args.seed}_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results: {RESULTS_DIR}/swingdim_{args.mode}_s{args.seed}_results.json")


if __name__ == "__main__":
    main()
