#!/usr/bin/env python3
"""
NEAT 2-input Z-score momentum strategy — AMDDP1 reward (book §3.3.3).

Inputs (causal, computed at M5 bar close):
  f1[i]  = arctan_scaled( zscore_1000( (close[i] - close[i-1])  / pip        ) )
  f10[i] = arctan_scaled( zscore_1000( (close[i] - close[i-10]) / (10 * pip) ) )

  Sign-preserving Z-score: z = x / σ (no mean subtraction — preserves direction).
  Population = [i-1000, i-1], strictly causal. σ = std(x_pop).
  arctan scaling: arctan(z) * 2/π → maps to (-1, +1).
  close[i-1]  = end of previous 5-min bar (exactly 5 min ago, not bar open).
  close[i-10] = end of bar 10 steps ago (exactly 50 min ago).

Outputs (argmax of 3): 0=BUY  1=SELL  2=FLATTEN

Reward — AMDDP1 (§3.3.3):
  accumulated_drawdown[trade] = Σ max(0, hwm_pnl − pnl[bar])  for every bar
  score[trade] = (pnl − 0.01 × accumulated_drawdown) / max(mae, 0.1)
  fitness = mean(score) × n_trades   (linear trade count, SQN-like)
  Hard floor: n_trades < MIN_TRADES → fitness = −10

  TRUE AUDDC: drawdown accumulated at every bar (not just at new peaks).
  lambda=0.01 chosen per book: mild penalty, avoids "Hold-only" collapse.

No look-ahead bias:
  f1[i] uses close[i] (current close) and open[i-1] (prev bar open). ✓
  f10[i] uses close[i] and open[i-10]. ✓
  Z-score population excludes bar i. ✓
  Decision at bar close → executed at same bar close (standard). ✓

Pipeline: IS sweep (zigzag pretrain 30 gens → WF P&L 200 gens, 4 islands)
          → MC permutation (2000 shuffles, shuffle directions)
          → OOS final evaluation

Usage:
  python3 train.py                          # all 12 pairs, IS only
  python3 train.py --oos                    # include OOS evaluation
  python3 train.py --pair GBP_JPY --oos     # single pair
  python3 train.py --gens 300 --pop 150
"""

import sys
import os
import copy
import math
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import neat
from lib.fast_eval import extract_network, _activate

M5_DIR      = PROJECT_ROOT / "data" / "m5_ohlc"
S5_DIR      = PROJECT_ROOT / "data" / "s5_ohlc"
RESULTS_DIR = SCRIPT_DIR / "results"
CONFIG_PATH = SCRIPT_DIR / "neat_config_2in_3out.ini"

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
ALL_PAIRS   = list(PAIR_PIP.keys())
MIN_TRADES  = 50    # per chunk — below this → fitness = -10
POP_SIZE    = 1000  # rolling Z-score window
WARMUP      = POP_SIZE + 600  # bars to skip before features are valid (covers lb2=600 for S5)
IS_FRAC     = 0.70
N_CHUNKS    = 3
N_ISLANDS   = 4
LAMBDA      = 0.01  # AMDDP1 penalty factor (book §3.3.3)


# ── Feature computation ───────────────────────────────────────────────────

@njit(cache=True)
def compute_zscore_inputs(close, pip, pop_size=1000, lb1=1, lb2=10):
    """
    Causal rolling sign-preserving Z-score, arctan-scaled.

    f1[i]  = arctan( (close[i]-close[i-lb1]) / pip         / σ_pop ) * 2/π
    f10[i] = arctan( (close[i]-close[i-lb2]) / (lb2 * pip) / σ_pop ) * 2/π

    For M5 bars: lb1=1  (5 min),   lb2=10  (50 min)
    For S5 bars: lb1=12 (1 min),   lb2=600 (50 min)

    Sign-preserving: z = x / σ — no mean subtraction.
    Population = [i-pop_size, i-1], strictly causal.
    Warmup: first pop_size+lb2 bars are 0 (insufficient history).
    Uses sliding window sum/sum² for O(n) computation.
    """
    n = len(close)
    f1  = np.zeros(n)
    f10 = np.zeros(n)
    half_pi = np.pi / 2.0

    # Raw returns — close-to-close, exact fixed lookback
    r1  = np.zeros(n)
    r10 = np.zeros(n)
    for i in range(lb1, n):
        r1[i]  = (close[i] - close[i - lb1]) / pip
    for i in range(lb2, n):
        r10[i] = (close[i] - close[i - lb2]) / (lb2 * pip)

    # Sliding window: population = [i-pop_size, i-1] (strictly causal)
    sum1 = 0.0;  sum1_sq = 0.0
    sum10 = 0.0; sum10_sq = 0.0
    start = pop_size + lb2  # first bar with full history for both features

    # Initialise window at bar `start` (window covers [start-pop_size, start-1])
    for k in range(start - pop_size, start):
        sum1    += r1[k];   sum1_sq    += r1[k]  * r1[k]
        sum10   += r10[k];  sum10_sq   += r10[k] * r10[k]

    for i in range(start, n):
        if i > start:
            # Slide window: add r[i-1], remove r[i-1-pop_size]
            add1 = r1[i - 1];   rem1 = r1[i - 1 - pop_size]
            sum1    += add1 - rem1
            sum1_sq += add1 * add1 - rem1 * rem1

            add10 = r10[i - 1];  rem10 = r10[i - 1 - pop_size]
            sum10    += add10 - rem10
            sum10_sq += add10 * add10 - rem10 * rem10

        m1  = sum1  / pop_size
        m10 = sum10 / pop_size

        var1 = sum1_sq  / pop_size - m1  * m1
        var10= sum10_sq / pop_size - m10 * m10
        std1  = var1 ** 0.5  if var1  > 1e-20 else 1e-10
        std10 = var10 ** 0.5 if var10 > 1e-20 else 1e-10

        z1  = r1[i]  / std1
        z10 = r10[i] / std10

        f1[i]  = np.arctan(z1)  / half_pi
        f10[i] = np.arctan(z10) / half_pi

    return f1, f10


# ── Zigzag labels for pretrain ────────────────────────────────────────────

@njit(cache=True)
def make_zigzag_labels(close, pip, min_swing_pips=10.0):
    """Simple zigzag: label bars near confirmed swing lows (BUY=1) / highs (SELL=2)."""
    n          = len(close)
    labels     = np.zeros(n, dtype=np.int64)
    min_swing  = min_swing_pips * pip
    running_hi = close[0]; running_lo = close[0]; direction = 0

    for i in range(1, n):
        p = close[i]
        if p > running_hi: running_hi = p
        if p < running_lo: running_lo = p

        if direction == 0:
            if running_hi - p >= min_swing:
                direction = -1; running_lo = p
            elif p - running_lo >= min_swing:
                direction = 1;  running_hi = p
        elif direction == 1:
            if running_hi - p >= min_swing:
                end = min(i + 6, n)
                for k in range(i, end): labels[k] = 2   # SELL at top
                direction = -1; running_lo = p
        else:
            if p - running_lo >= min_swing:
                end = min(i + 6, n)
                for k in range(i, end): labels[k] = 1   # BUY at bottom
                direction = 1;  running_hi = p
    return labels


@njit(cache=True)
def zigzag_fitness_jit(f1, f10, labels,
                       n_inputs, n_eval, total_values,
                       node_bias, node_response, node_act,
                       conn_from, conn_to, conn_weight,
                       output_indices):
    values = np.zeros(total_values)
    n = len(f1)
    correct = 0; total_labeled = 0
    for i in range(WARMUP, n - 1):
        lbl = labels[i]
        if lbl == 0:
            continue
        values[0] = f1[i]
        values[1] = f10[i]
        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)
        out_buy  = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_flat = values[output_indices[2]]
        if lbl == 1 and out_buy >= out_sell and out_buy >= out_flat:
            correct += 1
        elif lbl == 2 and out_sell >= out_buy and out_sell >= out_flat:
            correct += 1
        total_labeled += 1
    if total_labeled == 0:
        return 0.0
    return float(correct) / float(total_labeled)


# ── AMDDP1 chunk evaluator ────────────────────────────────────────────────

@njit(cache=True)
def evaluate_amddp1_chunk(f1, f10, close,
                          pip, spread_pips, max_hold, lam,
                          n_inputs, n_eval, total_values,
                          node_bias, node_response, node_act,
                          conn_from, conn_to, conn_weight,
                          output_indices,
                          chunk_start, chunk_end):
    """
    AMDDP1 evaluator (book §3.3.3).

    AUDDC per trade: dd_sum += max(0, hwm_pnl - pnl_pips) at EVERY bar.
    score[trade] = (pnl - lam * dd_sum) / max(mae, 0.1)
    Returns (n_trades, mean_score, n_long, n_short).
    """
    values    = np.zeros(total_values)
    start_bar = max(chunk_start, WARMUP)
    end_bar   = chunk_end

    max_trades = end_bar - start_bar + 1
    scores     = np.zeros(max_trades)
    n_trades   = 0; n_long = 0; n_short = 0

    position    = 0
    entry_price = 0.0
    entry_bar   = 0
    mfe         = 0.0   # max favourable excursion (pips)
    mae         = 0.0   # max adverse excursion (pips)
    hwm_pnl     = 0.0   # high-water-mark of pnl within trade
    dd_sum      = 0.0   # AUDDC: Σ max(0, hwm - pnl) at every bar

    for i in range(start_bar, end_bar):
        # Current unrealised P&L
        if position != 0:
            pnl_pips = (close[i] - entry_price) * position / pip - spread_pips
            # MFE / MAE
            if pnl_pips > mfe:  mfe = pnl_pips
            if -pnl_pips > mae: mae = -pnl_pips
            # HWM and AUDDC: accumulate at EVERY bar (book §3.3.2 + §3.3.3)
            if pnl_pips > hwm_pnl: hwm_pnl = pnl_pips
            dd_sum += max(0.0, hwm_pnl - pnl_pips)
        else:
            pnl_pips = 0.0

        # Network inputs
        values[0] = f1[i]
        values[1] = f10[i]
        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        out_buy  = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_flat = values[output_indices[2]]

        # Determine action (argmax)
        if out_buy >= out_sell and out_buy >= out_flat:
            action = 1   # BUY
        elif out_sell > out_buy and out_sell >= out_flat:
            action = 2   # SELL
        else:
            action = 3   # FLATTEN

        # Force close at max hold
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (close[i] - entry_price) * position / pip - spread_pips
            sc  = (pnl - lam * dd_sum) / max(mae, 0.1)
            if n_trades < max_trades:
                scores[n_trades] = sc
                if position > 0: n_long += 1
                else:            n_short += 1
                n_trades += 1
            position = 0; mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0
            action = 1 if out_buy >= out_sell else 2   # re-evaluate entry

        # Execute action
        if action == 1:   # BUY
            if position == -1:   # close short
                pnl = (close[i] - entry_price) * position / pip - spread_pips
                sc  = (pnl - lam * dd_sum) / max(mae, 0.1)
                if n_trades < max_trades:
                    scores[n_trades] = sc; n_short += 1; n_trades += 1
                position = 0; mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0
            if position == 0:
                position = 1; entry_price = close[i]; entry_bar = i

        elif action == 2:  # SELL
            if position == 1:    # close long
                pnl = (close[i] - entry_price) * position / pip - spread_pips
                sc  = (pnl - lam * dd_sum) / max(mae, 0.1)
                if n_trades < max_trades:
                    scores[n_trades] = sc; n_long += 1; n_trades += 1
                position = 0; mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0
            if position == 0:
                position = -1; entry_price = close[i]; entry_bar = i

        elif action == 3:  # FLATTEN
            if position != 0:
                pnl = (close[i] - entry_price) * position / pip - spread_pips
                sc  = (pnl - lam * dd_sum) / max(mae, 0.1)
                if n_trades < max_trades:
                    scores[n_trades] = sc
                    if position > 0: n_long += 1
                    else:            n_short += 1
                    n_trades += 1
                position = 0; mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0

    # Close any remaining position at end of chunk
    if position != 0 and end_bar > start_bar:
        pnl = (close[end_bar - 1] - entry_price) * position / pip - spread_pips
        sc  = (pnl - lam * dd_sum) / max(mae, 0.1)
        if n_trades < max_trades:
            scores[n_trades] = sc
            if position > 0: n_long += 1
            else:            n_short += 1
            n_trades += 1

    if n_trades < 1:
        return 0, -10.0, 0, 0

    mean_sc = 0.0
    for j in range(n_trades):
        mean_sc += scores[j]
    mean_sc /= n_trades

    return n_trades, mean_sc, n_long, n_short


# ── MC permutation (shuffle trade directions) ─────────────────────────────

@njit(cache=True)
def evaluate_shuffled_amddp1(f1, f10, close, shuffle_seed,
                              pip, spread_pips, max_hold, lam,
                              n_inputs, n_eval, total_values,
                              node_bias, node_response, node_act,
                              conn_from, conn_to, conn_weight,
                              output_indices):
    """Run with shuffled entry directions — baseline for MC permutation test."""
    np.random.seed(shuffle_seed)
    values    = np.zeros(total_values)
    n         = len(close)
    start_bar = WARMUP

    max_trades = n - start_bar
    scores     = np.zeros(max_trades)
    n_trades   = 0

    position    = 0
    entry_price = 0.0
    entry_bar   = 0
    mfe         = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0

    for i in range(start_bar, n - 1):
        if position != 0:
            pnl_pips = (close[i] - entry_price) * position / pip - spread_pips
            if pnl_pips > mfe:  mfe = pnl_pips
            if -pnl_pips > mae: mae = -pnl_pips
            if pnl_pips > hwm_pnl: hwm_pnl = pnl_pips
            dd_sum += max(0.0, hwm_pnl - pnl_pips)
        else:
            pnl_pips = 0.0

        values[0] = f1[i];  values[1] = f10[i]
        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        out_buy  = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_flat = values[output_indices[2]]

        if out_buy >= out_sell and out_buy >= out_flat:
            net_action = 1
        elif out_sell > out_buy and out_sell >= out_flat:
            net_action = 2
        else:
            net_action = 3

        # Flip direction with 50% probability for BUY/SELL entries
        if net_action in (1, 2) and position == 0:
            if np.random.random() < 0.5:
                net_action = 2 if net_action == 1 else 1

        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (close[i] - entry_price) * position / pip - spread_pips
            sc  = (pnl - lam * dd_sum) / max(mae, 0.1)
            if n_trades < max_trades: scores[n_trades] = sc; n_trades += 1
            position = 0; mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0

        if net_action == 1:
            if position == -1:
                pnl = (close[i] - entry_price) * position / pip - spread_pips
                sc  = (pnl - lam * dd_sum) / max(mae, 0.1)
                if n_trades < max_trades: scores[n_trades] = sc; n_trades += 1
                position = 0; mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0
            if position == 0:
                position = 1; entry_price = close[i]; entry_bar = i
        elif net_action == 2:
            if position == 1:
                pnl = (close[i] - entry_price) * position / pip - spread_pips
                sc  = (pnl - lam * dd_sum) / max(mae, 0.1)
                if n_trades < max_trades: scores[n_trades] = sc; n_trades += 1
                position = 0; mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0
            if position == 0:
                position = -1; entry_price = close[i]; entry_bar = i
        elif net_action == 3:
            if position != 0:
                pnl = (close[i] - entry_price) * position / pip - spread_pips
                sc  = (pnl - lam * dd_sum) / max(mae, 0.1)
                if n_trades < max_trades: scores[n_trades] = sc; n_trades += 1
                position = 0; mfe = 0.0; mae = 0.0; hwm_pnl = 0.0; dd_sum = 0.0

    if n_trades < 1:
        return 0.0
    total = 0.0
    for j in range(n_trades): total += scores[j]
    return total / n_trades


# ── Data loading ──────────────────────────────────────────────────────────

def load_pair(pair: str):
    s5_path = S5_DIR / f"{pair}_S5_BA.parquet"
    if s5_path.exists():
        df  = pd.read_parquet(s5_path)
        df.columns = [c.lower() for c in df.columns]
        closes = df["bid_c"].values.astype(np.float64)
        lb1, lb2 = 12, 600          # 1 min and 50 min in S5 bars
        granularity = "S5"
        zz_swing = 2.0              # 2-pip zigzag for S5 (10-pip → only 1.7% labeled)
    else:
        path = M5_DIR / f"{pair}_M5.parquet"
        df   = pd.read_parquet(path).sort_index()
        df.columns = [c.lower() for c in df.columns]
        closes = df["close"].values.astype(np.float64)
        lb1, lb2 = 1, 10            # 5 min and 50 min in M5 bars
        granularity = "M5"
        zz_swing = 10.0

    n      = len(closes)
    n_is   = int(n * IS_FRAC)
    pip    = PAIR_PIP[pair]

    f1, f10 = compute_zscore_inputs(closes, pip, POP_SIZE, lb1, lb2)
    print(f"  {pair}: {granularity} {n:,} bars (lb1={lb1}, lb2={lb2})")

    return {
        "pair":      pair,
        "pip":       pip,
        "spread":    PAIR_SPREAD[pair],
        "zz_swing":  zz_swing,
        "f1_is":     f1[:n_is],
        "f10_is":    f10[:n_is],
        "close_is":  closes[:n_is],
        "f1_oos":    f1[n_is:],
        "f10_oos":   f10[n_is:],
        "close_oos": closes[n_is:],
        "n_is":      n_is,
        "n_oos":     n - n_is,
    }


# ── NEAT evaluators ───────────────────────────────────────────────────────

def _extract(genome, config):
    return extract_network(genome, config)


class ZigzagEvaluator:
    def __init__(self, pairs_data):
        self.pairs_data = pairs_data

    def evaluate(self, genomes, config):
        for _, genome in genomes:
            try:
                net = _extract(genome, config)
            except Exception:
                genome.fitness = 0.0
                continue
            total = 0.0; count = 0
            for pd_ in self.pairs_data:
                labels = make_zigzag_labels(pd_["close_is"], pd_["pip"], pd_["zz_swing"])
                s = zigzag_fitness_jit(
                    pd_["f1_is"], pd_["f10_is"], labels,
                    net[0], net[2], net[3], net[4], net[5], net[6],
                    net[7], net[8], net[9], net[10])
                total += s; count += 1
            genome.fitness = total / count if count else 0.0


class AMDDP1Evaluator:
    """WF evaluator: min-chunk score × n_trades scaling (§3.3.3 fitness)."""
    def __init__(self, pairs_data, max_hold=200, n_chunks=N_CHUNKS,
                 lam=LAMBDA):
        self.pairs_data = pairs_data
        self.max_hold   = max_hold
        self.n_chunks   = n_chunks
        self.lam        = lam

    def evaluate(self, genomes, config):
        for _, genome in genomes:
            try:
                genome.fitness = self._fitness(genome, config)
            except Exception:
                genome.fitness = -10.0

    def _fitness(self, genome, config):
        try:
            net = _extract(genome, config)
        except Exception:
            return -10.0

        pair_min_scores = []
        total_trades    = 0

        for pd_ in self.pairs_data:
            n_is  = pd_["n_is"]
            chunk_scores = []
            for ci in range(self.n_chunks):
                cs = int(n_is * ci / self.n_chunks)
                ce = int(n_is * (ci + 1) / self.n_chunks)
                nt, mean_sc, nl, ns = evaluate_amddp1_chunk(
                    pd_["f1_is"], pd_["f10_is"], pd_["close_is"],
                    pd_["pip"], pd_["spread"], self.max_hold, self.lam,
                    net[0], net[2], net[3], net[4], net[5], net[6],
                    net[7], net[8], net[9], net[10],
                    cs, ce)
                if nt < MIN_TRADES or mean_sc <= -5.0:
                    chunk_scores.append(-10.0)
                else:
                    # fitness = mean_score × n_trades (linear, §3.3.3)
                    chunk_scores.append(mean_sc * nt)
                total_trades += nt

            pair_min_scores.append(min(chunk_scores))

        if total_trades < MIN_TRADES * self.n_chunks * len(self.pairs_data):
            return -10.0
        # Bottleneck: worst pair × worst chunk
        return min(pair_min_scores)


# ── Island runner ─────────────────────────────────────────────────────────

class _Reporter(neat.reporting.BaseReporter):
    def __init__(self, tag): self.tag = tag; self.gen = 0
    def post_evaluate(self, config, population, species_set, best_genome):
        self.gen += 1
        f = best_genome.fitness if best_genome and best_genome.fitness else -999
        if self.gen % 20 == 0:
            print(f"    [{self.tag}] gen {self.gen:>3}: best={f:.4f} "
                  f"sp={len(species_set.species)}")


def run_islands(config, zz_eval, wf_eval,
                n_islands=N_ISLANDS, pop_per_island=150,
                pretrain_gens=30, gens=200,
                stall_limit=80, migrate_every=10, tag="zscore"):

    islands = []
    for i in range(n_islands):
        p = neat.Population(config)
        p.generation = 0; p.best_genome = None
        p.add_reporter(_Reporter(f"{tag} ISL{i+1}"))
        islands.append({"pop": p, "best": None, "best_fitness": -999})

    # Zigzag pretrain
    print(f"\n  Pretrain ({pretrain_gens} gens × {n_islands} islands)...")
    for isl in islands:
        isl["pop"].run(zz_eval.evaluate, pretrain_gens)
        best = max(isl["pop"].population.values(),
                   key=lambda g: g.fitness if g.fitness is not None else -999)
        isl["best"] = copy.deepcopy(best)
        isl["best_fitness"] = best.fitness or -999

    # WF P&L evolution — reset island bests so pretrain fitness doesn't pollute WF tracking
    for isl in islands:
        isl["best_fitness"] = -999
    print(f"  WF evolution ({gens} gens, stall={stall_limit})...")
    best_ever = None; best_ever_fitness = -999; stall = 0

    for gen in range(gens):
        for isl in islands:
            isl["pop"].run(wf_eval.evaluate, 1)
            gb = max(isl["pop"].population.values(),
                     key=lambda g: g.fitness if g.fitness is not None else -999)
            if gb.fitness is not None and gb.fitness > isl["best_fitness"]:
                isl["best"] = copy.deepcopy(gb)
                isl["best_fitness"] = gb.fitness

        glb = max(islands, key=lambda x: x["best_fitness"])
        if glb["best_fitness"] > best_ever_fitness:
            best_ever = copy.deepcopy(glb["best"])
            best_ever_fitness = glb["best_fitness"]
            stall = 0
        else:
            stall += 1

        if gen % 20 == 0:
            isl_str = ', '.join(f'{x["best_fitness"]:.2f}' for x in islands)
            print(f"    [{tag}] gen {gen:>3}: best={best_ever_fitness:.4f} "
                  f"stall={stall} isl=[{isl_str}]")

        # Ring migration
        if (gen + 1) % migrate_every == 0:
            for i in range(n_islands):
                src = islands[i]; dst = islands[(i + 1) % n_islands]
                if src["best"]:
                    migrant  = copy.deepcopy(src["best"])
                    worst_k  = min(dst["pop"].population,
                                   key=lambda k: dst["pop"].population[k].fitness
                                   if dst["pop"].population[k].fitness is not None else -999)
                    migrant.key = worst_k
                    dst["pop"].population[worst_k] = migrant

        if stall >= stall_limit:
            print(f"    Stalled — stopping at gen {gen}")
            break

    return best_ever, best_ever_fitness


# ── OOS evaluation ────────────────────────────────────────────────────────

def eval_oos(genome, config, pairs_data, max_hold=200):
    net = _extract(genome, config)
    results = []
    for pd_ in pairs_data:
        nt, mean_sc, nl, ns = evaluate_amddp1_chunk(
            pd_["f1_oos"], pd_["f10_oos"], pd_["close_oos"],
            pd_["pip"], pd_["spread"], max_hold, LAMBDA,
            net[0], net[2], net[3], net[4], net[5], net[6],
            net[7], net[8], net[9], net[10],
            0, pd_["n_oos"])
        n_days = pd_["n_oos"] / 288.0
        results.append({
            "pair": pd_["pair"],
            "n_trades": int(nt),
            "mean_score": round(float(mean_sc), 4),
            "fitness": round(float(mean_sc * nt), 2),
            "pips_day": round(float(mean_sc * nt / max(n_days, 1)), 2),
            "n_long": int(nl), "n_short": int(ns),
        })
    return results


def mc_permutation(genome, config, pairs_data, n_shuffles=2000, max_hold=200):
    """MC permutation test: what fraction of shuffled-direction runs beat real fitness?"""
    net = _extract(genome, config)
    real_scores = []
    for pd_ in pairs_data:
        nt, mean_sc, _, _ = evaluate_amddp1_chunk(
            pd_["f1_oos"], pd_["f10_oos"], pd_["close_oos"],
            pd_["pip"], pd_["spread"], max_hold, LAMBDA,
            net[0], net[2], net[3], net[4], net[5], net[6],
            net[7], net[8], net[9], net[10],
            0, pd_["n_oos"])
        real_scores.append(mean_sc * nt if nt > 0 else -10.0)
    real_fitness = float(np.mean(real_scores))

    beat = 0
    for s in range(n_shuffles):
        shuf_scores = []
        for pd_ in pairs_data:
            sc = evaluate_shuffled_amddp1(
                pd_["f1_oos"], pd_["f10_oos"], pd_["close_oos"],
                s * 31337 + hash(pd_["pair"]) % 10000,
                pd_["pip"], pd_["spread"], max_hold, LAMBDA,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10])
            # approximate n_trades for shuffled — use same count proxy
            shuf_scores.append(float(sc) * 100)   # rough scale
        if float(np.mean(shuf_scores)) >= real_fitness:
            beat += 1
    return beat / n_shuffles


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair",      default=None)
    parser.add_argument("--gens",      type=int, default=200)
    parser.add_argument("--pop",       type=int, default=150)
    parser.add_argument("--islands",   type=int, default=N_ISLANDS)
    parser.add_argument("--max-hold",  type=int, default=200)
    parser.add_argument("--pretrain",  type=int, default=30)
    parser.add_argument("--stall",     type=int, default=80)
    parser.add_argument("--oos",       action="store_true")
    parser.add_argument("--mc",        action="store_true")
    parser.add_argument("--mc-n",      type=int, default=2000)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    pairs = [args.pair] if args.pair else ALL_PAIRS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"NEAT 2-input Z-score + AMDDP1 (λ={LAMBDA})")
    print(f"Inputs: f1 = zscore(close[i]-close[i-1]/pip), f10 = zscore(close[i]-close[i-10]/(10×pip))")
    print(f"Pairs: {pairs}  |  IS={IS_FRAC*100:.0f}%  |  max_hold={args.max_hold}")
    print(f"Warmup: {WARMUP} bars  |  pop_size: {args.pop}×{args.islands} islands")
    print()

    print("Loading and computing Z-score features...")
    t0 = time.time()
    pairs_data = []
    for pair in pairs:
        pd_ = load_pair(pair)
        pairs_data.append(pd_)
        print(f"  {pair}: IS={pd_['n_is']:,} bars, OOS={pd_['n_oos']:,} bars")
    print(f"  Done in {time.time()-t0:.1f}s\n")

    # JIT warmup
    print("JIT warmup...")
    dummy_c = np.ones(WARMUP + 50, dtype=np.float64)
    compute_zscore_inputs(dummy_c, 0.01, POP_SIZE, 12, 600)
    print("  Done\n")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(CONFIG_PATH))
    config.pop_size = args.pop

    zz_eval = ZigzagEvaluator(pairs_data)
    wf_eval = AMDDP1Evaluator(pairs_data, max_hold=args.max_hold)

    print("Training...")
    t1 = time.time()
    best_genome, best_fitness = run_islands(
        config, zz_eval, wf_eval,
        n_islands=args.islands, pop_per_island=args.pop,
        pretrain_gens=args.pretrain, gens=args.gens,
        stall_limit=args.stall, tag="zscore")
    print(f"\nTraining done in {time.time()-t1:.1f}s  |  IS fitness = {best_fitness:.4f}")

    # Save genome
    genome_path = RESULTS_DIR / f"best_genome_amddp1.pkl"
    with open(genome_path, "wb") as f:
        pickle.dump((best_genome, config), f)
    print(f"Genome saved: {genome_path}")

    if args.oos:
        print("\nOOS evaluation:")
        oos_results = eval_oos(best_genome, config, pairs_data, args.max_hold)
        positive = sum(1 for r in oos_results if r["fitness"] > 0)
        total_fitness = sum(r["fitness"] for r in oos_results)
        avg_ppd = np.mean([r["pips_day"] for r in oos_results])
        print(f"  {'Pair':<12} {'N':>6} {'Score':>8} {'Fitness':>9} {'p/d':>7} {'L':>5} {'S':>5}")
        print("  " + "─" * 60)
        for r in oos_results:
            marker = "🟢" if r["fitness"] > 0 else "🔴"
            print(f"  {r['pair']:<12} {r['n_trades']:>6} {r['mean_score']:>8.4f} "
                  f"{r['fitness']:>9.1f} {r['pips_day']:>7.1f} "
                  f"{r['n_long']:>5} {r['n_short']:>5}  {marker}")
        print(f"\n  Positive: {positive}/{len(oos_results)} | "
              f"Total fitness: {total_fitness:.1f} | Avg p/d: {avg_ppd:.2f}")

    if args.mc:
        print(f"\nMC permutation test ({args.mc_n} shuffles)...")
        perm_p = mc_permutation(best_genome, config, pairs_data,
                                args.mc_n, args.max_hold)
        gate = "🟢 PASS" if perm_p < 0.05 else "🔴 FAIL"
        print(f"  perm_p = {perm_p:.4f}  {gate}")


if __name__ == "__main__":
    main()
