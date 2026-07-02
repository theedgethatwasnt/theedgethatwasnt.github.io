#!/usr/bin/env python3
"""
Free NEAT v2: Topology-Evolving NEAT with Full Activation Set + Seeding + Islands
==================================================================================
Fixes from v1:
  - All 13 activations (tanh/sin/cos/gauss/sech/dog/gabor/sinc/mex_hat/morlet_re/
    morlet_im/haar/sigmoid). No chirp/relu/elu/swish — never selected vs wavelets.
  - Seeded population: 1 pure-activation genome per activation type injected at gen 0
    (same technique as activation_study v3/v4 — accelerates early convergence ~2x)
  - 4 islands with migration every 10 gens (same as IronNet — prevents premature
    convergence and diversity collapse in single population)
  - Zigzag pretrain: 50 gens supervised before WF P&L evolution
    (gives topology mutations a structured starting point, not random noise)
  - Stall limit 150 (was 60 — topology innovations need time to develop before
    being killed; 60 gens was too aggressive for node-add mutations)
  - 500 gens default (was 300)

4 inputs: MC(D), MC(dD), ER_norm, UPnL
3 outputs: BUY, SELL, FLATTEN

Usage:
  python3 train_free_neat.py --pair EUR_GBP --seed 42
  python3 train_free_neat.py --pair CAD_JPY --seed 137
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
RESULTS_DIR = SCRIPT_DIR / "results" / "free_neat"

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
PAIR_MIN_SWING = {
    "EUR_JPY": 30, "USD_JPY": 25, "GBP_JPY": 40, "AUD_JPY": 25,
    "CAD_JPY": 30, "CHF_JPY": 35, "NZD_JPY": 25,
    "EUR_USD": 20, "GBP_USD": 25, "AUD_USD": 18,
    "NZD_USD": 18, "EUR_GBP": 15,
}

N_INPUTS  = 4   # MC_D, MC_dD, ER_norm, UPnL (UPnL added dynamically in evaluator)
N_OUTPUTS = 3   # BUY, SELL, FLATTEN
N_CHUNKS  = 3


# ═══════════════════════════════════════════════════════════════════════════
# All 13 activations — no chirp/relu/elu/swish (never selected vs wavelets)
# ═══════════════════════════════════════════════════════════════════════════

def _tanh_s(x):      return math.tanh(x)
def _sin_s(x):       return math.sin(x)
def _cos_s(x):       return math.cos(x)
def _gauss_s(x):     return math.exp(-x * x)
def _sech_s(x):      return 1.0 / math.cosh(x)
def _dog_s(x):       return math.exp(-x*x/2) - 0.5*math.exp(-x*x/8)
def _gabor_s(x):     return math.exp(-2*x*x) * math.cos(2*math.pi*x)
def _sinc_s(x):      return 1.0 if abs(x) < 1e-9 else math.sin(math.pi*x) / (math.pi*x)
def _mex_hat_s(x):   return (1.0 - x*x) * math.exp(-x*x/2.0)
def _morlet_re_s(x): return math.exp(-x*x/2) * math.cos(5*x)
def _morlet_im_s(x): return math.exp(-x*x/2) * math.sin(5*x)
def _sigmoid_s(x):   return 1.0 / (1.0 + math.exp(-x))
def _haar_s(x):
    if 0.0 <= x < 0.5: return 1.0
    if 0.5 <= x < 1.0: return -1.0
    return 0.0

# (name, scalar_fn) — order matters for seeding (one seed per activation)
ACTIVATIONS = [
    ('tanh',      _tanh_s),
    ('sin',       _sin_s),
    ('cos',       _cos_s),
    ('gauss',     _gauss_s),
    ('sech',      _sech_s),
    ('dog',       _dog_s),
    ('gabor',     _gabor_s),
    ('sinc',      _sinc_s),
    ('mex_hat',   _mex_hat_s),
    ('morlet_re', _morlet_re_s),
    ('morlet_im', _morlet_im_s),
    ('sigmoid',   _sigmoid_s),
    ('haar',      _haar_s),
]
ACT_NAMES   = [a[0] for a in ACTIVATIONS]
ACT_SCALAR  = {a[0]: a[1] for a in ACTIVATIONS}
N_ACTS      = len(ACTIVATIONS)   # 13


def register_activations(config):
    for name, fn in ACTIVATIONS:
        try:
            config.genome_config.add_activation(name, fn)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════

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
def compute_range_pos(closes, window=30):
    """Price position within rolling N-bar range, normalized to [-1, 1]."""
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
    """Bollinger Band width = (upper-lower)/mid, tanh-scaled."""
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
            width = (4.0 * std) / mean
            result[i] = np.tanh(width * 5.0)
    return result


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


# ═══════════════════════════════════════════════════════════════════════════
# Zigzag pretrain labels (identical to IronNet)
# ═══════════════════════════════════════════════════════════════════════════

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
        if price < running_low:  running_low  = price
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
# JIT Evaluators
# ═══════════════════════════════════════════════════════════════════════════

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
        out_buy  = values[output_indices[0]]
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
        pnl = (mid_close[end_bar-1] - entry_price) / pip * position - spread_pips
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
    return n_trades, total_pnl, total_mae / n_trades, n_long, n_short


@njit(cache=True)
def zigzag_fitness_jit(
    inputs_2d, labels,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices,
):
    values = np.zeros(total_values)
    n = inputs_2d.shape[1]
    correct = 0; wrong = 0; total_labeled = 0
    for i in range(10, n - 1):
        lbl = labels[i]
        if lbl == 0:
            continue
        for k in range(inputs_2d.shape[0]):
            values[k] = inputs_2d[k, i]
        values[inputs_2d.shape[0]] = 0.0
        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)
        out_buy  = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_flat = values[output_indices[2]]
        if lbl == 1:   # BUY
            if out_buy >= out_sell and out_buy >= out_flat: correct += 1
            else: wrong += 1
        elif lbl == 2:  # SELL
            if out_sell >= out_buy and out_sell >= out_flat: correct += 1
            else: wrong += 1
        total_labeled += 1
    if total_labeled == 0:
        return 0.0
    return float(correct) / float(total_labeled)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluator classes
# ═══════════════════════════════════════════════════════════════════════════

class SineEvaluator:
    """Supervised evaluator on synthetic sine wave data — same as IronNet phase 1."""
    def __init__(self, inputs_2d, labels):
        self.inputs_2d = inputs_2d
        self.labels = labels

    def evaluate(self, genomes, config):
        for _, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return 0.0
        return zigzag_fitness_jit(
            self.inputs_2d, self.labels,
            net[0], net[2], net[3], net[4], net[5], net[6],
            net[7], net[8], net[9], net[10])


class ZigzagEvaluator:
    def __init__(self, inputs_2d, labels):
        self.inputs_2d = inputs_2d
        self.labels = labels

    def evaluate(self, genomes, config):
        for _, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return 0.0
        return zigzag_fitness_jit(
            self.inputs_2d, self.labels,
            net[0], net[2], net[3], net[4], net[5], net[6],
            net[7], net[8], net[9], net[10])


class WFEvaluator:
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
        for _, genome in genomes:
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
            c_end   = int(self.n_bars * (ci + 1) / self.n_chunks)
            nt, pnl, mae, nl, ns = evaluate_chunk_jit(
                self.inputs_2d, self.mid_close,
                self.pip, self.spread, self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10],
                c_start, c_end)
            total_long += nl; total_short += ns; total_trades += nt
            min_trades = max(30, int(self.n_bars / self.n_chunks / 288 * 0.5))
            if nt < min_trades or pnl <= 0:
                return -10.0
            score = (pnl / nt / mae) if mae > 0 else (pnl / nt)
            chunk_scores.append(score * (nt ** 0.5))
        if total_trades < 10:
            return -10.0
        dir_ratio = min(total_long, total_short) / total_trades
        if dir_ratio < self.min_dir_ratio:
            return -10.0
        mean_score = sum(chunk_scores) / len(chunk_scores)
        if mean_score > 0:
            cv = (sum((s - mean_score)**2 for s in chunk_scores) / len(chunk_scores))**0.5 / mean_score
            consistency = 1.0 / (1.0 + cv)
        else:
            consistency = 0.5
        dir_bonus = 1.0 + 0.5 * (dir_ratio - self.min_dir_ratio) / (0.5 - self.min_dir_ratio)
        return min(chunk_scores) * (1.0 + consistency) * dir_bonus


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
# Population seeding — one genome per activation type
# Mirrors activation_study/run_v4.py seeding logic
# ═══════════════════════════════════════════════════════════════════════════

def build_seeded_genome(config, activation_name, genome_key):
    """Build a genome with a fixed activation on all output nodes.
    configure_new with partial_direct 0.5 already creates connections — no manual addition needed."""
    genome = neat.DefaultGenome(genome_key)
    genome.configure_new(config.genome_config)

    # Override all node activations to the target activation
    for node in genome.nodes.values():
        node.activation = activation_name

    return genome


def build_seeded_population(config, pop_size):
    """
    Inject 1 pure-activation seed per activation (13 seeds), fill remainder randomly.
    Same strategy as activation_study v3/v4 — accelerates early convergence ~2×.
    """
    population = {}
    # Inject activation seeds (13 genomes)
    for idx, (name, _) in enumerate(ACTIVATIONS):
        g = build_seeded_genome(config, name, idx)
        g.fitness = None
        population[idx] = g

    # Fill rest with neat default genomes
    for i in range(N_ACTS, pop_size):
        g = neat.DefaultGenome(i)
        g.configure_new(config.genome_config)
        g.fitness = None
        population[i] = g

    return population


# ═══════════════════════════════════════════════════════════════════════════
# Island management (4 islands with migration every 10 gens)
# ═══════════════════════════════════════════════════════════════════════════

class _IslandReporter(neat.reporting.BaseReporter):
    def __init__(self, label):
        self.label = label
        self.gen = 0

    def post_evaluate(self, config, population, species_set, best_genome):
        self.gen += 1
        f = best_genome.fitness if best_genome and best_genome.fitness else -999
        n_sp = len(species_set.species)
        if self.gen % 20 == 0:
            print(f"    [{self.label}] gen {self.gen:>3}: best={f:.4f} species={n_sp}")


def run_free_neat_islands(config_path, sine_evaluator, zz_evaluator, wf_evaluator,
                          n_islands=4, pop_per_island=150,
                          sine_gens=30, pretrain_gens=50, gens=500,
                          stall_limit=150, migrate_every=10,
                          pair="?", tag="free_neat"):
    """4-island free NEAT with sine pretrain + zigzag pretrain + WF evolution."""

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_path))
    config.pop_size = pop_per_island
    register_activations(config)

    # Build 4 islands, each with seeded population
    islands = []
    for i in range(n_islands):
        # Create population normally (builds random pop + species), then
        # replace population dict with our seeded genomes and re-speciate.
        p = neat.Population(config)
        p.population = build_seeded_population(config, pop_per_island)
        p.species.speciate(config, p.population, 0)
        p.generation = 0
        p.best_genome = None
        p.add_reporter(_IslandReporter(f"{pair} ISL{i+1}"))
        islands.append({"pop": p, "best": None, "best_fitness": -999})

    best_ever = None
    best_ever_fitness = -999
    stall_counter = 0

    # ── Phase 0: Sine wave pretrain ────────────────────────────────────────
    if sine_gens > 0 and sine_evaluator is not None:
        print(f"\nPhase 0: Sine wave pretrain ({sine_gens} gens, {n_islands} islands)...")
        for isl in islands:
            try:
                isl["pop"].run(sine_evaluator.evaluate, sine_gens)
            except Exception as e:
                print(f"  Sine pretrain error: {e}")
            best = max(isl["pop"].population.values(),
                       key=lambda g: g.fitness if g.fitness else -999)
            isl["best"] = copy.deepcopy(best)
            isl["best_fitness"] = best.fitness if best.fitness else -999
        fits = [f"{isl['best_fitness']:.4f}" for isl in islands]
        print(f"  Sine pretrain done. ISL fitnesses: {fits}")

    # ── Phase 1: Zigzag pretrain ──────────────────────────────────────────
    print(f"\nPhase 1: Zigzag pretrain ({pretrain_gens} gens, {n_islands} islands)...")
    for isl in islands:
        try:
            isl["pop"].run(zz_evaluator.evaluate, pretrain_gens)
        except Exception as e:
            print(f"  Pretrain error: {e}")
        best = max(isl["pop"].population.values(),
                   key=lambda g: g.fitness if g.fitness else -999)
        isl["best"] = copy.deepcopy(best)
        isl["best_fitness"] = best.fitness if best.fitness else -999
        print(f"  ISL {islands.index(isl)+1}: pretrain done, best_zz={isl['best_fitness']:.4f}")

    # ── Phase 2: WF P&L evolution + island migration ──────────────────────
    print(f"\nPhase 2: WF P&L evolution ({gens} gens, stall={stall_limit}, migrate every {migrate_every})...")

    for gen in range(gens):
        # Run 1 gen per island
        for isl in islands:
            try:
                isl["pop"].run(wf_evaluator.evaluate, 1)
            except Exception:
                pass
            gen_best = max(isl["pop"].population.values(),
                           key=lambda g: g.fitness if g.fitness is not None else -999)
            if gen_best.fitness is not None and gen_best.fitness > isl["best_fitness"]:
                isl["best"] = copy.deepcopy(gen_best)
                isl["best_fitness"] = gen_best.fitness

        # Global best tracking
        global_best = max(islands, key=lambda isl: isl["best_fitness"])
        if global_best["best_fitness"] > best_ever_fitness:
            best_ever = copy.deepcopy(global_best["best"])
            best_ever_fitness = global_best["best_fitness"]
            stall_counter = 0
        else:
            stall_counter += 1

        # Log every 10 gens
        if gen % 10 == 0:
            fitnesses = [isl["best_fitness"] for isl in islands]
            print(f"  [{pair} FREE] Gen {gen:>3}: best={best_ever_fitness:.4f} "
                  f"stall={stall_counter} islands={[f'{f:.2f}' for f in fitnesses]}")

        # Island migration (ring topology): each island donates best to next
        if (gen + 1) % migrate_every == 0 and gen > 0:
            for i in range(n_islands):
                src = islands[i]
                dst = islands[(i + 1) % n_islands]
                if src["best"] is not None:
                    migrant = copy.deepcopy(src["best"])
                    # Replace worst genome in destination
                    worst_key = min(dst["pop"].population,
                                    key=lambda k: dst["pop"].population[k].fitness
                                    if dst["pop"].population[k].fitness is not None else -999)
                    migrant.key = worst_key
                    dst["pop"].population[worst_key] = migrant

        if stall_counter >= stall_limit:
            print(f"  Stalled {stall_limit} gens — stopping at gen {gen}")
            break

    return best_ever, best_ever_fitness, gen + 1


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Free NEAT v2: full activations + seeding + islands")
    parser.add_argument("--pair",          required=True)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--gens",          type=int, default=500)
    parser.add_argument("--pop",           type=int, default=150)
    parser.add_argument("--islands",       type=int, default=4)
    parser.add_argument("--sine-gens",     type=int, default=30)
    parser.add_argument("--pretrain-gens", type=int, default=50)
    parser.add_argument("--migrate-every", type=int, default=10)
    parser.add_argument("--max-hold",      type=int, default=200)
    parser.add_argument("--min-dir-ratio", type=float, default=0.15)
    parser.add_argument("--stall-limit",   type=int,  default=150)
    parser.add_argument("--n-inputs",      type=int,  default=4,
                        choices=[4, 6],
                        help="4=V3 inputs (MC_D,MC_dD,ER,UPnL) | 6=V5 inputs (+range_pos,bb_width)")
    parser.add_argument("--range-window",  type=int,  default=30)
    parser.add_argument("--bb-window",     type=int,  default=20)
    args = parser.parse_args()

    np.random.seed(args.seed)
    pair    = args.pair
    pip     = PAIR_PIP[pair]
    spread  = PAIR_SPREAD[pair]
    min_sw  = PAIR_MIN_SWING[pair]
    if args.n_inputs == 6:
        config_path = SCRIPT_DIR / "neat_config_free_6in.ini"
    else:
        config_path = SCRIPT_DIR / "neat_config_free_4in.ini"

    input_desc = ("MC_D, MC_dD, ER_norm, UPnL" if args.n_inputs == 4
                  else f"MC_D, MC_dD, ER_norm, range_pos({args.range_window}), bb_width({args.bb_window}), UPnL")
    print(f"{'='*65}")
    print(f"  Free NEAT v2: {pair}")
    print(f"  Inputs: {args.n_inputs} ({input_desc})")
    print(f"  Topology: evolving (0 hidden start, node_add=0.2, conn_add=0.3)")
    print(f"  Activations: {N_ACTS} ({', '.join(ACT_NAMES)})")
    print(f"  Seeding: 1 genome/activation + {args.pop - N_ACTS} random = {args.pop} total")
    print(f"  Islands: {args.islands} × pop {args.pop}, migrate every {args.migrate_every}")
    print(f"  Pretrain: sine {args.sine_gens}g → zigzag {args.pretrain_gens}g → WF {args.gens}g")
    print(f"  Stall limit: {args.stall_limit} | Seed: {args.seed}")
    print(f"{'='*65}")

    tg_send(f"🧬 Free NEAT v2 {pair} s{args.seed}\n"
            f"{N_ACTS} acts + seeded + {args.islands} islands\n"
            f"Sine {args.sine_gens}g → Zigzag {args.pretrain_gens}g → WF {args.gens}g | stall={args.stall_limit}")

    # ── Load data ──────────────────────────────────────────────────────────
    path = DATA_DIR / f"{pair}_asi_mc.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found"); return

    df   = pd.read_parquet(path, engine="pyarrow")
    mid  = df["mid_close"].values.astype(np.float64)
    n    = len(mid)
    split = int(n * 0.7)

    mc_d  = df["mc_d_a"].values.astype(np.float64)
    mc_dd = df["mc_dd_a"].values.astype(np.float64)
    er    = compute_er_norm(mid, window=60)
    if args.n_inputs == 6:
        rp  = compute_range_pos(mid, window=args.range_window)
        bbw = compute_bb_width(mid, window=args.bb_window)
        raw_inputs = np.stack([mc_d, mc_dd, er, rp, bbw], axis=0)  # (5, n) — UPnL added in evaluator
    else:
        raw_inputs = np.stack([mc_d, mc_dd, er], axis=0)            # (3, n) — UPnL added in evaluator
    del df; gc.collect()

    inputs     = raw_inputs
    inputs_is  = inputs[:, :split]
    mid_is     = mid[:split]
    inputs_oos = inputs[:, split:]
    mid_oos    = mid[split:]

    print(f"\nData: {n:,} M5 bars | IS: {split:,} | OOS: {n-split:,}")

    # Zigzag labels for pretrain
    labels_is = generate_zigzag_labels(mid_is, pip, min_sw)
    n_buy  = int(np.sum(labels_is == 1))
    n_sell = int(np.sum(labels_is == 2))
    print(f"  Zigzag labels: BUY={n_buy:,} SELL={n_sell:,} ({(n_buy+n_sell)/split*100:.1f}%)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"free_neat_{pair}_s{args.seed}"

    # ── Sine wave evaluator (Phase 0) ─────────────────────────────────────
    sine_eval = None
    if args.sine_gens > 0:
        from lib.asi_indicator import compute_asi_mc
        pair_center = float(mid_is.mean())
        amp_pips    = 15
        n_sine      = 50000
        sine_period = 500
        rng = np.random.RandomState(args.seed)
        t_arr   = np.arange(n_sine, dtype=np.float64)
        sine_mid = pair_center + amp_pips * pip * np.sin(2 * np.pi * t_arr / sine_period)
        sine_mid += rng.normal(0, 1.0 * pip, n_sine)
        sine_o  = np.empty(n_sine, dtype=np.float64)
        sine_c  = sine_mid.copy()
        sine_o[0] = sine_mid[0]
        for ii in range(1, n_sine):
            sine_o[ii] = sine_c[ii - 1]
        hl_noise = rng.uniform(0, 2 * pip, n_sine)
        sine_h   = np.maximum(sine_o, sine_c) + hl_noise
        sine_l   = np.minimum(sine_o, sine_c) - hl_noise
        s_mc_d, s_mc_dd = compute_asi_mc(sine_o, sine_h, sine_l, sine_c, n_sine)
        s_er = compute_er_norm(sine_mid, window=60)
        if args.n_inputs == 6:
            s_rp  = compute_range_pos(sine_mid, window=args.range_window)
            s_bbw = compute_bb_width(sine_mid, window=args.bb_window)
            sine_inputs = np.stack([s_mc_d, s_mc_dd, s_er, s_rp, s_bbw], axis=0)
        else:
            sine_inputs = np.stack([s_mc_d, s_mc_dd, s_er], axis=0)
        sine_labels = generate_zigzag_labels(sine_mid, pip, amp_pips // 2,
                                             label_window=8, min_mfe_pips=2.0)
        s_buy  = int(np.sum(sine_labels == 1))
        s_sell = int(np.sum(sine_labels == 2))
        print(f"\nPhase 0 setup: Sine center={pair_center:.5f} amp=±{amp_pips}p  BUY={s_buy} SELL={s_sell}")
        print(f"  Sine MC_D  range: [{s_mc_d[200:].min():.4f}, {s_mc_d[200:].max():.4f}]")
        print(f"  Sine ER    range: [{s_er[200:].min():.4f}, {s_er[200:].max():.4f}]")
        sine_eval = SineEvaluator(sine_inputs, sine_labels)
        del s_mc_d, s_mc_dd, s_er, sine_o, sine_h, sine_l, sine_c, hl_noise, t_arr
        if args.n_inputs == 6:
            del s_rp, s_bbw
        gc.collect()

    zz_eval = ZigzagEvaluator(inputs_is, labels_is)
    wf_eval = WFEvaluator(inputs_is, mid_is, pip, spread,
                          max_hold=args.max_hold, n_chunks=N_CHUNKS,
                          min_dir_ratio=args.min_dir_ratio)

    t_start = time.time()

    best_ever, best_ever_fitness, actual_gens = run_free_neat_islands(
        config_path, sine_eval, zz_eval, wf_eval,
        n_islands=args.islands,
        pop_per_island=args.pop,
        sine_gens=args.sine_gens,
        pretrain_gens=args.pretrain_gens,
        gens=args.gens,
        stall_limit=args.stall_limit,
        migrate_every=args.migrate_every,
        pair=pair,
        tag=tag,
    )

    t_elapsed = time.time() - t_start

    if best_ever is None:
        print("ERROR: No valid genome found"); return

    # ── OOS evaluation ─────────────────────────────────────────────────────
    config_final = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation,
                               str(config_path))
    register_activations(config_final)

    print(f"\nOOS evaluation...")
    is_res  = eval_oos(best_ever, config_final, inputs_is,  mid_is,  pip, spread, args.max_hold)
    oos_res = eval_oos(best_ever, config_final, inputs_oos, mid_oos, pip, spread, args.max_hold)

    n_nodes = len(best_ever.nodes)
    n_conns = sum(1 for c in best_ever.connections.values() if c.enabled)
    acts    = [nd.activation for nd in best_ever.nodes.values()]
    act_counts = dict((a, acts.count(a)) for a in set(acts))

    print(f"\n{'='*65}")
    print(f"  RESULTS: {pair} (Free NEAT v2)")
    print(f"{'='*65}")
    print(f"  Architecture: 4 → {n_nodes - N_OUTPUTS} hidden → {N_OUTPUTS} (discovered)")
    print(f"  Connections: {n_conns} enabled | Generations: {actual_gens}")
    print(f"  Activations: {act_counts}")
    print(f"  IS:  {is_res['n_trades']:>5}T  {is_res['total_pnl']:>+9.1f}p  "
          f"L={is_res['n_long']} S={is_res['n_short']}  "
          f"MAE={is_res['avg_mae']:.1f}p  ({is_res['pips_per_day']:.1f}p/day)")
    print(f"  OOS: {oos_res['n_trades']:>5}T  {oos_res['total_pnl']:>+9.1f}p  "
          f"L={oos_res['n_long']} S={oos_res['n_short']}  "
          f"MAE={oos_res['avg_mae']:.1f}p  ({oos_res['pips_per_day']:.1f}p/day)")
    print(f"  Fitness: {best_ever_fitness:.4f} | Time: {t_elapsed:.0f}s")

    result_data = {
        "pair": pair, "variant": "free_neat_v2",
        "seed": args.seed, "actual_gens": actual_gens,
        "n_inputs": args.n_inputs, "n_outputs": N_OUTPUTS,
        "n_islands": args.islands,
        "n_activations": N_ACTS, "activation_names": ACT_NAMES,
        "pretrain_gens": args.pretrain_gens,
        "discovered_nodes": n_nodes, "discovered_conns": n_conns,
        "n_chunks": N_CHUNKS, "min_dir_ratio": args.min_dir_ratio,
        "fitness": round(best_ever_fitness, 4),
        "activations": act_counts,
        "is": is_res, "oos": oos_res,
        "elapsed_s": round(t_elapsed, 1),
    }

    with open(RESULTS_DIR / f"{tag}_v2_best.pkl", "wb") as f:
        pickle.dump({"genome": best_ever, "config": config_final, "pair": pair,
                     "result": result_data}, f)
    with open(RESULTS_DIR / f"{tag}_v2_result.json", "w") as f:
        json.dump(result_data, f, indent=2)

    tg_send(f"🧬 Free NEAT v2 {pair} s{args.seed} DONE\n"
            f"Arch: 4→{n_nodes-N_OUTPUTS}h→{N_OUTPUTS} ({n_conns} conn)\n"
            f"Acts: {act_counts}\n"
            f"Fitness: {best_ever_fitness:.4f} | {actual_gens} gens\n"
            f"OOS: {oos_res['n_trades']}T {oos_res['total_pnl']:+.1f}p "
            f"({oos_res['pips_per_day']:.1f}p/day)")
    print(f"\nSaved: {tag}_v2_best.pkl + {tag}_v2_result.json")


if __name__ == "__main__":
    main()
