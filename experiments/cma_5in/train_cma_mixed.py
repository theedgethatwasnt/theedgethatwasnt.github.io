#!/usr/bin/env python3
"""
CMA-NN Mixed Activation Layers with Skip Connections to Output.
================================================================

Architecture:
    input(n_in) → L1(w1, sin) → L2(w2, mexican_hat) → L3(w3, tanh) → output(3)
                         ↘              ↘                              ↗
                          ──── skip connections (additive) ────────────

    out = W_out @ h3 + W_skip2 @ h2 + W_skip1 @ h1 + b_out

Each hidden layer has `width` nodes (default 3). All nodes in a layer share
the same activation function. Skip connections let CMA-ES learn to bypass
depth if it hurts — effectively discovering optimal depth automatically.

Supports incremental depth training via --warm-start:
    1. Train 1-layer: --layers 8 --activations sin  (baseline, matches train_cma_v2)
    2. Train 3-layer: --layers 3,3,3 --activations sin,mexican_hat,tanh --warm-start prev.pkl
       (loads L1 weights from prev, identity-inits new layers, small sigma)

Data loading and fitness function identical to train_cma_v2.py (V3+extras, IS pps).

Usage:
    # Experiment A: mixed activation with skip
    python3 train_cma_mixed.py --pair CHF_JPY --extras macd_hist \\
        --layers 3,3,3 --activations sin,mexican_hat,tanh --skip

    # Experiment C: incremental depth (after training shallow)
    python3 train_cma_mixed.py --pair CHF_JPY --extras macd_hist \\
        --layers 3,3,3 --activations sin,mexican_hat,tanh --skip \\
        --warm-start results/shallow_best.pkl --sigma0 0.1

    # Single wide layer (replicates train_cma_v2 architecture)
    python3 train_cma_mixed.py --pair CHF_JPY --extras macd_hist \\
        --layers 8 --activations sin
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import cma

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "unified_indicators"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_OUT = 3
N_POSITION_STATE = 3

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

EXTRA_NORMALIZE = {
    "bb_width":   ("mul", 20.0),
    "macd_hist":  ("div_clip", 2.0),
    "gap_norm":   ("div_clip", 1.0),
}

ACT_NAMES = ["sin", "tanh", "mexican_hat", "gauss", "sech", "dog", "morlet", "cos"]


# ── Activation functions (numba) ──────────────────────────────────────
@njit(cache=True, inline="always")
def act_sin(z):
    return np.sin(z)


@njit(cache=True, inline="always")
def act_tanh(z):
    return np.tanh(z)


@njit(cache=True, inline="always")
def act_mexican_hat(z):
    return (1.0 - z * z) * np.exp(-z * z / 2.0)


@njit(cache=True, inline="always")
def act_gauss(z):
    return np.exp(-z * z)


@njit(cache=True, inline="always")
def act_sech(z):
    zc = max(-50.0, min(50.0, z))
    return 1.0 / np.cosh(zc)


@njit(cache=True, inline="always")
def act_dog(z):
    return np.exp(-z * z / 2.0) - 0.5 * np.exp(-z * z / 8.0)


@njit(cache=True, inline="always")
def act_morlet(z):
    return np.sin(z) * np.exp(-z * z / 2.0)


@njit(cache=True, inline="always")
def apply_act(z, act_id):
    """Dispatch activation by integer ID matching ACT_NAMES order."""
    if act_id == 0:
        return np.sin(z)
    elif act_id == 1:
        return np.tanh(z)
    elif act_id == 2:
        return (1.0 - z * z) * np.exp(-z * z / 2.0)
    elif act_id == 3:
        return np.exp(-z * z)
    elif act_id == 4:
        zc = max(-50.0, min(50.0, z))
        return 1.0 / np.cosh(zc)
    elif act_id == 5:
        return np.exp(-z * z / 2.0) - 0.5 * np.exp(-z * z / 8.0)
    elif act_id == 6:  # morlet
        return np.sin(z) * np.exp(-z * z / 2.0)
    else:  # 7 = cos
        return np.cos(z)


# ── ER norm (curator-identical) ───────────────────────────────────────
@njit(cache=True)
def _compute_er_norm_v3(closes, window=60):
    n = len(closes)
    out = np.zeros(n)
    hp = np.pi / 2.0
    for i in range(window, n):
        net = abs(closes[i] - closes[i - window])
        path = 0.0
        for j in range(i - window + 1, i + 1):
            path += abs(closes[j] - closes[j - 1])
        if path > 0.0:
            er = net / path
            out[i] = np.arctan(er / 0.3) / hp
    return out


@njit(cache=True)
def compute_m5_slope(closes, lookback=12):
    n = len(closes)
    out = np.zeros(n)
    hp = np.pi / 2.0
    for i in range(lookback, n):
        x_mean = (lookback - 1) / 2.0
        y_mean = 0.0
        for k in range(lookback):
            y_mean += closes[i - lookback + 1 + k]
        y_mean /= lookback
        num = 0.0
        den = 0.0
        ymin = closes[i - lookback + 1]
        ymax = ymin
        for k in range(lookback):
            y = closes[i - lookback + 1 + k]
            xd = k - x_mean
            num += xd * (y - y_mean)
            den += xd * xd
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
        if den > 0:
            slope = num / den
            rng = ymax - ymin
            out[i] = np.arctan((slope / rng * 3.0) if rng > 0 else 0.0) / hp
    return out


# ── Genome layout ─────────────────────────────────────────────────────
#
# For layers = [w1, w2, w3] with skip connections:
#
#   Section 1: L1 weights + biases
#     W1: n_in × w1, b1: w1
#   Section 2: L2 weights + biases
#     W2: w1 × w2, b2: w2
#   Section 3: L3 weights + biases
#     W3: w2 × w3, b3: w3
#   Section 4: Output weights + biases (from last hidden)
#     W_out: w_last × 3, b_out: 3
#   Section 5 (if skip): Skip weights from each non-last hidden layer
#     W_skip1: w1 × 3
#     W_skip2: w2 × 3
#
# Total params = sum(layer_weights) + sum(layer_biases) + out_weights + out_biases + skip_weights

def compute_genome_layout(n_in, layer_widths, skip=False):
    """Compute genome layout: offsets and total size."""
    layout = {"layers": [], "n_layers": len(layer_widths)}
    offset = 0

    prev_width = n_in
    for i, w in enumerate(layer_widths):
        lw = prev_width * w  # weight count
        lb = w               # bias count
        layout["layers"].append({
            "w_start": offset,
            "w_end": offset + lw,
            "b_start": offset + lw,
            "b_end": offset + lw + lb,
            "in_width": prev_width,
            "out_width": w,
        })
        offset += lw + lb
        prev_width = w

    # Output layer
    last_w = layer_widths[-1]
    lw = last_w * N_OUT
    layout["out_w_start"] = offset
    layout["out_w_end"] = offset + lw
    layout["out_b_start"] = offset + lw
    layout["out_b_end"] = offset + lw + N_OUT
    offset += lw + N_OUT

    # Skip connections
    layout["skip"] = skip
    layout["skip_layers"] = []
    if skip and len(layer_widths) > 1:
        for i in range(len(layer_widths) - 1):
            w = layer_widths[i]
            sw = w * N_OUT
            layout["skip_layers"].append({
                "start": offset,
                "end": offset + sw,
                "width": w,
            })
            offset += sw

    layout["total_params"] = offset
    return layout


# ── JIT simulator (multi-layer, skip) ─────────────────────────────────
@njit(cache=True)
def simulate_chunk_mixed(
    market_features,   # (n_market, n_bars)
    mid_close,
    pip, spread_pips, max_hold,
    weights,
    n_in,
    # Layer config (flattened for numba)
    n_layers,
    layer_widths,      # int64 array
    layer_act_ids,     # int64 array (one per layer)
    layer_w_starts,    # int64 array
    layer_b_starts,    # int64 array
    # Output config
    out_w_start, out_b_start,
    # Skip config
    use_skip,
    skip_starts,       # int64 array (one per skip layer)
    skip_widths,       # int64 array
    n_skips,
    # Chunk bounds
    chunk_start, chunk_end,
):
    n_market = market_features.shape[0]
    n = market_features.shape[1]

    start_bar = max(chunk_start + 12, 12)
    end_bar = min(chunk_end, n - 1)
    n_capacity = end_bar - start_bar + 1
    if n_capacity <= 0:
        return 0, 0.0, 0, 0

    pnls = np.zeros(n_capacity)
    n_trades = 0
    n_long = 0
    n_short = 0

    position = 0
    entry_price = 0.0
    entry_bar = 0
    upnl_pips = 0.0
    mae_pips = 0.0
    mfe_pips = 0.0

    # Allocate layer activations (max width across layers)
    max_w = 0
    for li in range(n_layers):
        if layer_widths[li] > max_w:
            max_w = layer_widths[li]
    # Store all layer outputs for skip connections
    all_h = np.zeros((n_layers, max_w))

    inp = np.zeros(n_in)

    for i in range(start_bar, end_bar):
        # Position metrics
        if position != 0:
            upnl_pips = (mid_close[i] - entry_price) / pip * position
            adverse = -upnl_pips
            if adverse > mae_pips:
                mae_pips = adverse
            if upnl_pips > mfe_pips:
                mfe_pips = upnl_pips
        else:
            upnl_pips = 0.0
            mae_pips = 0.0
            mfe_pips = 0.0

        # Build input
        for k in range(n_market):
            inp[k] = market_features[k, i]
        inp[n_market] = np.tanh(upnl_pips / 20.0)
        inp[n_market + 1] = np.tanh(mae_pips / 20.0)
        inp[n_market + 2] = np.tanh(mfe_pips / 20.0)

        # Forward pass through hidden layers
        prev = inp
        prev_width = n_in
        for li in range(n_layers):
            w = layer_widths[li]
            ws = layer_w_starts[li]
            bs = layer_b_starts[li]
            act = layer_act_ids[li]
            for j in range(w):
                z = weights[bs + j]  # bias
                for k in range(prev_width):
                    z += weights[ws + j * prev_width + k] * prev[k]
                all_h[li, j] = apply_act(z, act)
            prev = all_h[li, :w]
            prev_width = w

        # Output: W_out @ h_last + b_out
        last_w = layer_widths[n_layers - 1]
        out = np.zeros(N_OUT)
        for j in range(N_OUT):
            z = weights[out_b_start + j]
            for k in range(last_w):
                z += weights[out_w_start + j * last_w + k] * all_h[n_layers - 1, k]
            out[j] = z

        # Add skip contributions
        if use_skip:
            for si in range(n_skips):
                sw = skip_widths[si]
                ss = skip_starts[si]
                for j in range(N_OUT):
                    for k in range(sw):
                        out[j] += weights[ss + j * sw + k] * all_h[si, k]

        # Argmax
        if out[0] >= out[1] and out[0] >= out[2]:
            action = 1   # buy
        elif out[1] >= out[0] and out[1] >= out[2]:
            action = 2   # sell
        else:
            action = 0   # flatten

        # Force-close on max_hold
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid_close[i] - entry_price) / pip * position
            if n_trades < n_capacity:
                pnls[n_trades] = pnl
                if position > 0:
                    n_long += 1
                else:
                    n_short += 1
                n_trades += 1
            position = 0
            entry_price = 0.0
            mae_pips = 0.0
            mfe_pips = 0.0

        # Apply action
        if position == 0:
            if action == 1:
                position = 1
                entry_price = mid_close[i] + spread_pips * pip
                entry_bar = i
                mae_pips = spread_pips
                mfe_pips = 0.0
            elif action == 2:
                position = -1
                entry_price = mid_close[i] - spread_pips * pip
                entry_bar = i
                mae_pips = spread_pips
                mfe_pips = 0.0
        else:
            close_now = False
            new_pos = 0
            if action == 0:
                close_now = True
            elif position == 1 and action == 2:
                close_now = True
                new_pos = -1
            elif position == -1 and action == 1:
                close_now = True
                new_pos = 1
            if close_now:
                pnl = (mid_close[i] - entry_price) / pip * position
                if n_trades < n_capacity:
                    pnls[n_trades] = pnl
                    if position > 0:
                        n_long += 1
                    else:
                        n_short += 1
                    n_trades += 1
                position = new_pos
                if new_pos == 1:
                    entry_price = mid_close[i] + spread_pips * pip
                    entry_bar = i
                    mae_pips = spread_pips
                    mfe_pips = 0.0
                elif new_pos == -1:
                    entry_price = mid_close[i] - spread_pips * pip
                    entry_bar = i
                    mae_pips = spread_pips
                    mfe_pips = 0.0
                else:
                    entry_price = 0.0

    # Close open position
    if position != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar - 1] - entry_price) / pip * position
        if n_trades < n_capacity:
            pnls[n_trades] = pnl
            if position > 0:
                n_long += 1
            else:
                n_short += 1
            n_trades += 1

    if n_trades < 1:
        return 0, 0.0, 0, 0
    total = 0.0
    for k in range(n_trades):
        total += pnls[k]
    return n_trades, total, n_long, n_short


# ── Fitness (identical to train_cma_v2) ──────────────────────────────
def fitness_neg(weights, market, mid_close, pip, spread, max_hold, n_in,
                layout, act_ids, n_chunks, min_dir_ratio):
    n_bars = len(mid_close)
    total_long = 0
    total_short = 0
    total_trades = 0
    total_pnl = 0.0
    chunk_pps = []
    losing_chunk_loss = 0.0

    # Extract numba-friendly arrays from layout
    n_layers = layout["n_layers"]
    layer_widths = np.array([l["out_width"] for l in layout["layers"]], dtype=np.int64)
    layer_w_starts = np.array([l["w_start"] for l in layout["layers"]], dtype=np.int64)
    layer_b_starts = np.array([l["b_start"] for l in layout["layers"]], dtype=np.int64)
    layer_act_ids = np.array(act_ids, dtype=np.int64)
    out_w_start = layout["out_w_start"]
    out_b_start = layout["out_b_start"]

    use_skip = layout["skip"]
    n_skips = len(layout["skip_layers"])
    if n_skips > 0:
        skip_starts = np.array([s["start"] for s in layout["skip_layers"]], dtype=np.int64)
        skip_widths = np.array([s["width"] for s in layout["skip_layers"]], dtype=np.int64)
    else:
        skip_starts = np.zeros(0, dtype=np.int64)
        skip_widths = np.zeros(0, dtype=np.int64)

    for ci in range(n_chunks):
        c_start = int(n_bars * ci / n_chunks)
        c_end = int(n_bars * (ci + 1) / n_chunks)

        nt, pnl, nl, ns = simulate_chunk_mixed(
            market, mid_close, pip, spread, max_hold,
            weights, n_in,
            n_layers, layer_widths, layer_act_ids,
            layer_w_starts, layer_b_starts,
            out_w_start, out_b_start,
            use_skip, skip_starts, skip_widths, n_skips,
            c_start, c_end,
        )
        total_long += nl
        total_short += ns
        total_trades += nt
        total_pnl += pnl

        n_days = (c_end - c_start) / 288.0
        pps = pnl / n_days if n_days > 0 else 0.0
        chunk_pps.append(pps)
        if pps < 0:
            losing_chunk_loss += -pps

    total_days = n_bars / 288.0
    base_pps = total_pnl / total_days if total_days > 0 else 0.0

    if total_trades == 0:
        return 500.0 - base_pps
    dir_ratio = min(total_long, total_short) / total_trades

    asym_penalty = (1.0 - 2.0 * dir_ratio) * 50.0
    activity_penalty = max(0.0, 30.0 - total_trades) * 2.0
    losing_pen = losing_chunk_loss * 2.0

    all_profitable = all(p > 0 for p in chunk_pps)
    if all_profitable and dir_ratio >= min_dir_ratio:
        score = min(chunk_pps) - asym_penalty
    else:
        score = base_pps - asym_penalty - activity_penalty - losing_pen

    return -score


def passes_hard_gates(weights, market, mid_close, pip, spread, max_hold, n_in,
                      layout, act_ids, n_chunks, min_dir_ratio):
    n_bars = len(mid_close)
    total_long = 0
    total_short = 0
    total_trades = 0
    chunk_pps = []

    n_layers = layout["n_layers"]
    layer_widths = np.array([l["out_width"] for l in layout["layers"]], dtype=np.int64)
    layer_w_starts = np.array([l["w_start"] for l in layout["layers"]], dtype=np.int64)
    layer_b_starts = np.array([l["b_start"] for l in layout["layers"]], dtype=np.int64)
    layer_act_ids = np.array(act_ids, dtype=np.int64)
    out_w_start = layout["out_w_start"]
    out_b_start = layout["out_b_start"]
    use_skip = layout["skip"]
    n_skips = len(layout["skip_layers"])
    if n_skips > 0:
        skip_starts = np.array([s["start"] for s in layout["skip_layers"]], dtype=np.int64)
        skip_widths = np.array([s["width"] for s in layout["skip_layers"]], dtype=np.int64)
    else:
        skip_starts = np.zeros(0, dtype=np.int64)
        skip_widths = np.zeros(0, dtype=np.int64)

    for ci in range(n_chunks):
        c_start = int(n_bars * ci / n_chunks)
        c_end = int(n_bars * (ci + 1) / n_chunks)
        nt, pnl, nl, ns = simulate_chunk_mixed(
            market, mid_close, pip, spread, max_hold,
            weights, n_in,
            n_layers, layer_widths, layer_act_ids,
            layer_w_starts, layer_b_starts,
            out_w_start, out_b_start,
            use_skip, skip_starts, skip_widths, n_skips,
            c_start, c_end,
        )
        total_long += nl
        total_short += ns
        total_trades += nt
        n_days = (c_end - c_start) / 288.0
        min_trades = max(20, int(n_days * 0.5))
        if nt < min_trades or pnl <= 0:
            return False, None
        chunk_pps.append(pnl / n_days)
    if total_trades < 30:
        return False, None
    if min(total_long, total_short) / total_trades < min_dir_ratio:
        return False, None
    return True, min(chunk_pps)


# ── Multiprocessing ───────────────────────────────────────────────────
_W = {}


def _worker_init(market, mid, pip, spread, max_hold, n_in,
                 layout, act_ids, n_chunks, min_dir):
    _W["market"] = market
    _W["mid"] = mid
    _W["pip"] = pip
    _W["spread"] = spread
    _W["max_hold"] = max_hold
    _W["n_in"] = n_in
    _W["layout"] = layout
    _W["act_ids"] = act_ids
    _W["n_chunks"] = n_chunks
    _W["min_dir"] = min_dir


def _worker_fit(vec):
    return fitness_neg(
        vec,
        _W["market"], _W["mid"], _W["pip"], _W["spread"], _W["max_hold"],
        _W["n_in"], _W["layout"], _W["act_ids"], _W["n_chunks"], _W["min_dir"],
    )


# ── OOS evaluator ─────────────────────────────────────────────────────
def eval_oos(weights, market, mid, pip, spread, max_hold, n_in, layout, act_ids):
    n_layers = layout["n_layers"]
    layer_widths = np.array([l["out_width"] for l in layout["layers"]], dtype=np.int64)
    layer_w_starts = np.array([l["w_start"] for l in layout["layers"]], dtype=np.int64)
    layer_b_starts = np.array([l["b_start"] for l in layout["layers"]], dtype=np.int64)
    layer_act_ids = np.array(act_ids, dtype=np.int64)
    out_w_start = layout["out_w_start"]
    out_b_start = layout["out_b_start"]
    use_skip = layout["skip"]
    n_skips = len(layout["skip_layers"])
    if n_skips > 0:
        skip_starts = np.array([s["start"] for s in layout["skip_layers"]], dtype=np.int64)
        skip_widths = np.array([s["width"] for s in layout["skip_layers"]], dtype=np.int64)
    else:
        skip_starts = np.zeros(0, dtype=np.int64)
        skip_widths = np.zeros(0, dtype=np.int64)

    nt, pnl, nl, ns = simulate_chunk_mixed(
        market, mid, pip, spread, max_hold,
        weights, n_in,
        n_layers, layer_widths, layer_act_ids,
        layer_w_starts, layer_b_starts,
        out_w_start, out_b_start,
        use_skip, skip_starts, skip_widths, n_skips,
        0, len(mid),
    )
    n_days = len(mid) / 288.0
    return {
        "n_trades": int(nt),
        "total_pnl": round(float(pnl), 1),
        "pips_per_day": round(float(pnl) / max(n_days, 1), 2),
        "n_long": int(nl),
        "n_short": int(ns),
        "dir_ratio": round(min(nl, ns) / max(nt, 1), 3),
    }


# ── CMA-ES runner ─────────────────────────────────────────────────────
def run_cma(x0, sigma0, popsize, gens, seed, pool, n_in, layout, act_ids,
            m_is, mid_is, pip, spread, max_hold, n_chunks, min_dir_ratio, t0,
            label="r0"):
    opts = {
        "popsize": popsize,
        "seed": seed,
        "verbose": -9,
        "tolx": 1e-9,
        "tolfun": 1e-3,
        "tolfunhist": 1e-3,
        "tolflatfitness": 50,
        "maxiter": gens,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_fit = 1e18
    best_vec = None
    best_valid_pps = None
    best_valid_vec = None
    gen = 0

    while not es.stop():
        candidates = es.ask()
        fitnesses = list(pool.map(_worker_fit, candidates))
        es.tell(candidates, fitnesses)

        gen_min = min(fitnesses)
        if gen_min < best_fit:
            best_fit = gen_min
            best_idx = fitnesses.index(gen_min)
            best_vec = np.array(candidates[best_idx])

        ok, min_pps = passes_hard_gates(
            best_vec, m_is, mid_is, pip, spread, max_hold, n_in,
            layout, act_ids, n_chunks, min_dir_ratio)
        if ok and (best_valid_pps is None or min_pps > best_valid_pps):
            best_valid_pps = min_pps
            best_valid_vec = np.array(best_vec)

        if gen % 10 == 0:
            valid_str = (f"{best_valid_pps:+.2f}p/d"
                         if best_valid_pps is not None else "---")
            print(f"  [{label}] Gen {gen:>3}: raw_fit={best_fit:>10.2f}  "
                  f"valid={valid_str}  sigma={es.sigma:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s")
        gen += 1
        if gen >= gens:
            break

    return best_vec, best_fit, best_valid_pps, best_valid_vec, gen


# ── Warm start: build expanded genome from shallow pkl ────────────────
def build_warm_start(warm_pkl_path, layout, act_ids):
    """Load a trained genome and map its weights into the new layout.

    For incremental depth: copies L1 weights from the trained shallow net,
    initializes new hidden layers as near-identity, and copies output weights
    where dimensions match.
    """
    with open(warm_pkl_path, "rb") as f:
        data = pickle.load(f)
    old_weights = data["weights"]

    n_params = layout["total_params"]
    new_weights = np.zeros(n_params)

    # Copy L1 weights (input→first hidden) from old genome
    old_l1 = layout["layers"][0]
    l1_size = old_l1["w_end"] - old_l1["w_start"] + old_l1["b_end"] - old_l1["b_start"]
    if len(old_weights) >= l1_size:
        # Copy W1 + b1
        w_len = old_l1["w_end"] - old_l1["w_start"]
        b_len = old_l1["b_end"] - old_l1["b_start"]
        if w_len <= len(old_weights):
            new_weights[old_l1["w_start"]:old_l1["w_end"]] = old_weights[:w_len]
            new_weights[old_l1["b_start"]:old_l1["b_end"]] = old_weights[w_len:w_len + b_len]

    # Initialize new hidden layers as near-identity + small noise
    rng = np.random.RandomState(42)
    for li in range(1, layout["n_layers"]):
        l = layout["layers"][li]
        w_in = l["in_width"]
        w_out = l["out_width"]
        # Near-identity: diagonal 1.0, off-diagonal ~0
        for j in range(w_out):
            for k in range(w_in):
                idx = l["w_start"] + j * w_in + k
                if j == k and j < min(w_in, w_out):
                    # Near zero for sin: sin(x) ≈ x near 0
                    # For mexican_hat: mhat(0) = 1, so small input → ~1 output
                    # For tanh: tanh(x) ≈ x near 0
                    new_weights[idx] = 1.0 if act_ids[li] == 1 else 0.3  # tanh/mhat need smaller
                else:
                    new_weights[idx] = rng.randn() * 0.01
            new_weights[l["b_start"] + j] = rng.randn() * 0.01

    # Output layer: small random init
    out_size = layout["out_w_end"] - layout["out_w_start"]
    new_weights[layout["out_w_start"]:layout["out_w_end"]] = rng.randn(out_size) * 0.3
    new_weights[layout["out_b_start"]:layout["out_b_end"]] = 0.0

    # Skip connections: zero init (let CMA-ES discover if they help)
    for sl in layout["skip_layers"]:
        new_weights[sl["start"]:sl["end"]] = 0.0

    return new_weights


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CMA-NN mixed-activation multi-layer trainer")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--max-hold", type=int, default=200)
    parser.add_argument("--n-chunks", type=int, default=3)
    parser.add_argument("--min-dir-ratio", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label", type=str, default="mixed")
    parser.add_argument("--extras", nargs="+", default=["macd_hist"],
                        help="Extra market indicators on top of V3")
    parser.add_argument("--layers", type=str, default="3,3,3",
                        help="Comma-separated hidden layer widths (e.g. '3,3,3' or '8')")
    parser.add_argument("--activations", type=str, default="sin,mexican_hat,tanh",
                        help=f"Comma-separated activations per layer. "
                             f"Available: {', '.join(ACT_NAMES)}")
    parser.add_argument("--skip", action="store_true",
                        help="Enable skip connections from all hidden layers to output")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Path to trained genome .pkl for incremental depth")
    parser.add_argument("--restarts", type=int, default=0,
                        help="IPOP restarts (0 = single run)")
    parser.add_argument("--popsize-mult", type=float, default=2.0)
    args = parser.parse_args()

    # Parse architecture
    layer_widths = [int(x) for x in args.layers.split(",")]
    act_names = args.activations.split(",")
    if len(act_names) == 1 and len(layer_widths) > 1:
        act_names = act_names * len(layer_widths)  # broadcast single act
    if len(act_names) != len(layer_widths):
        print(f"ERROR: --activations count ({len(act_names)}) != "
              f"--layers count ({len(layer_widths)})")
        sys.exit(1)
    act_ids = []
    for a in act_names:
        if a not in ACT_NAMES:
            print(f"ERROR: unknown activation '{a}'. Available: {ACT_NAMES}")
            sys.exit(1)
        act_ids.append(ACT_NAMES.index(a))

    extras_list = list(args.extras)
    n_in = 3 + len(extras_list) + N_POSITION_STATE  # V3(3) + extras + state(3)
    n_market = n_in - N_POSITION_STATE

    layout = compute_genome_layout(n_in, layer_widths, skip=args.skip)
    n_params = layout["total_params"]

    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]

    arch_str = "→".join(
        f"{w}({a})" for w, a in zip(layer_widths, act_names))
    skip_str = "+skip" if args.skip else ""

    print(f"{'='*65}")
    print(f"  CMA-NN Mixed: {pair}")
    print(f"  Architecture: {n_in}→{arch_str}→3{skip_str}")
    print(f"  Params: {n_params}")
    print(f"  Market: mc_d_a, mc_dd_a, er_norm + {extras_list}")
    print(f"  State:  upnl, mae, mfe")
    print(f"  Seed: {args.seed} | Pop: {args.popsize} | Sigma: {args.sigma0}")
    print(f"  Gens: {args.gens} | Workers: {args.workers}")
    if args.warm_start:
        print(f"  Warm start: {args.warm_start}")
    print(f"{'='*65}")

    # ── Load data (V3+extras, same as train_cma_v2) ───────────────
    ohlc_path = PROJECT_ROOT / "data" / "m5_ohlc" / f"{pair}_M5.parquet"
    uni_path = DATA_DIR / f"{pair}_unified.parquet"
    if not ohlc_path.exists():
        print(f"ERROR: {ohlc_path} not found")
        sys.exit(1)

    df_o = pd.read_parquet(ohlc_path, engine="pyarrow")

    # Build extras
    from extra_indicators import is_inline_computable, compute_inline
    extras_frame = pd.DataFrame({"timestamp": df_o["timestamp"]})
    o_arr = df_o["open"].values.astype(np.float64)
    h_arr = df_o["high"].values.astype(np.float64)
    l_arr = df_o["low"].values.astype(np.float64)
    c_arr = df_o["close"].values.astype(np.float64)

    need_unified = False
    for ex in extras_list:
        if ex == "m5_slope":
            extras_frame[ex] = compute_m5_slope(c_arr, lookback=12)
        elif is_inline_computable(ex):
            extras_frame[ex] = compute_inline(ex, o_arr, h_arr, l_arr, c_arr)
        else:
            need_unified = True
            break

    if need_unified:
        if not uni_path.exists():
            print(f"ERROR: {uni_path} not found (needed for non-inline extras)")
            sys.exit(1)
        df_u = pd.read_parquet(uni_path, engine="pyarrow")
        for ex in extras_list:
            if ex not in extras_frame.columns and ex != "m5_slope":
                if is_inline_computable(ex):
                    extras_frame[ex] = compute_inline(ex, o_arr, h_arr, l_arr, c_arr)
                elif ex in df_u.columns:
                    tmp = df_u[["timestamp", ex]]
                    extras_frame = extras_frame.merge(tmp, on="timestamp", how="left")
                else:
                    print(f"ERROR: '{ex}' not found")
                    sys.exit(1)
        del df_u

    del o_arr, h_arr, l_arr, c_arr
    df = df_o.merge(extras_frame, on="timestamp", how="inner")
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c_arr = df["close"].values.astype(np.float64)
    mid = c_arr.copy()
    n = len(mid)

    from lib.asi_indicator import compute_asi_mc as _compute_asi_mc
    mc_d_a, mc_dd_a = _compute_asi_mc(o, h, l, c_arr, n)
    er_norm_arr = _compute_er_norm_v3(c_arr, window=60)

    market_cols = ["mc_d_a", "mc_dd_a", "er_norm"] + extras_list
    cols = {"mc_d_a": mc_d_a, "mc_dd_a": mc_dd_a, "er_norm": er_norm_arr}
    for ex in extras_list:
        arr = df[ex].values.astype(np.float64)
        arr = np.nan_to_num(arr, nan=0.0)
        if ex in EXTRA_NORMALIZE:
            op, val = EXTRA_NORMALIZE[ex]
            if op == "mul":
                arr = arr * val
            elif op == "div_clip":
                arr = np.clip(arr / val, -1.0, 1.0)
        cols[ex] = arr

    del df, df_o, extras_frame, o, h, l, c_arr
    gc.collect()

    market_full = np.stack([cols[c] for c in market_cols], axis=0)
    split = int(n * 0.7)
    market_is = market_full[:, :split].copy()
    mid_is = mid[:split].copy()
    market_oos = market_full[:, split:].copy()
    mid_oos = mid[split:].copy()
    del market_full, mid, cols
    gc.collect()

    print(f"\nData: {n:,} M5 bars | IS: {split:,} | OOS: {n - split:,}")

    # ── JIT warmup ─────────────────────────────────────────────
    print("JIT warming up...")
    warm_w = np.zeros(n_params)
    lw = np.array(layer_widths, dtype=np.int64)
    la = np.array(act_ids, dtype=np.int64)
    lws = np.array([l["w_start"] for l in layout["layers"]], dtype=np.int64)
    lbs = np.array([l["b_start"] for l in layout["layers"]], dtype=np.int64)
    n_skips = len(layout["skip_layers"])
    if n_skips > 0:
        ss = np.array([s["start"] for s in layout["skip_layers"]], dtype=np.int64)
        sw = np.array([s["width"] for s in layout["skip_layers"]], dtype=np.int64)
    else:
        ss = np.zeros(0, dtype=np.int64)
        sw = np.zeros(0, dtype=np.int64)
    simulate_chunk_mixed(
        market_is[:, :200], mid_is[:200], pip, spread, 50,
        warm_w, n_in,
        len(layer_widths), lw, la, lws, lbs,
        layout["out_w_start"], layout["out_b_start"],
        args.skip, ss, sw, n_skips,
        0, 200,
    )
    print("  warm.")

    # ── Worker pool ────────────────────────────────────────────
    init_args = (market_is, mid_is, pip, spread, args.max_hold, n_in,
                 layout, act_ids, args.n_chunks, args.min_dir_ratio)
    pool = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=init_args,
    )

    tag_suffix = "_" + "+".join(extras_list) if extras_list else ""
    arch_tag = "_".join(f"{w}{a[0]}" for w, a in zip(layer_widths, act_names))
    skip_tag = "_skip" if args.skip else ""
    tag = f"{args.label}_{arch_tag}{skip_tag}{tag_suffix}_{pair}_s{args.seed}"

    t0 = time.time()

    overall_best_fit = 1e18
    overall_best_vec = None
    overall_best_valid_pps = None
    overall_best_valid_vec = None

    try:
        for ri in range(args.restarts + 1):
            popsize = int(args.popsize * (args.popsize_mult ** ri))
            sigma = args.sigma0 * (1.5 ** ri)
            seed = args.seed + ri * 1000

            if args.warm_start and ri == 0:
                x0 = build_warm_start(args.warm_start, layout, act_ids)
                sigma = args.sigma0  # typically 0.1 for warm start
                print(f"\n  Warm-started from {args.warm_start}")
            else:
                rng = np.random.RandomState(seed)
                x0 = rng.randn(n_params) * 0.3

            lbl = f"r{ri}"
            print(f"\n── Run {ri}/{args.restarts}: popsize={popsize} sigma0={sigma:.3f} seed={seed} ──")
            best_vec, best_fit, best_valid_pps, best_valid_vec, gens_done = run_cma(
                x0, sigma, popsize, args.gens, seed, pool, n_in, layout, act_ids,
                market_is, mid_is, pip, spread, args.max_hold,
                args.n_chunks, args.min_dir_ratio, t0, label=lbl,
            )

            if best_fit < overall_best_fit:
                overall_best_fit = best_fit
                overall_best_vec = best_vec
            if best_valid_pps is not None and (overall_best_valid_pps is None
                                                or best_valid_pps > overall_best_valid_pps):
                overall_best_valid_pps = best_valid_pps
                overall_best_valid_vec = best_valid_vec

    finally:
        pool.shutdown(wait=False)

    elapsed = time.time() - t0

    # Pick best genome
    final_vec = overall_best_valid_vec if overall_best_valid_vec is not None else overall_best_vec
    if final_vec is None:
        print("\nERROR: No genome found")
        sys.exit(1)

    # Eval IS + OOS
    is_full = eval_oos(final_vec, market_is, mid_is, pip, spread, args.max_hold,
                       n_in, layout, act_ids)
    oos = eval_oos(final_vec, market_oos, mid_oos, pip, spread, args.max_hold,
                   n_in, layout, act_ids)
    passed, min_pps = passes_hard_gates(
        final_vec, market_is, mid_is, pip, spread, args.max_hold, n_in,
        layout, act_ids, args.n_chunks, args.min_dir_ratio)

    print(f"\n{'='*65}")
    print(f"  RESULT: {pair} | {arch_str}{skip_str}")
    print(f"  IS:  {is_full['pips_per_day']:+.2f} p/day  "
          f"({is_full['n_trades']}T, L/S={is_full['n_long']}/{is_full['n_short']}, "
          f"dir={is_full['dir_ratio']:.2f})")
    print(f"  OOS: {oos['pips_per_day']:+.2f} p/day  "
          f"({oos['n_trades']}T, L/S={oos['n_long']}/{oos['n_short']}, "
          f"dir={oos['dir_ratio']:.2f})")
    print(f"  Hard gates: {'PASS' if passed else 'FAIL'}"
          + (f"  min_chunk={min_pps:+.2f}" if min_pps else ""))
    print(f"  Elapsed: {elapsed:.1f}s ({n_params} params)")
    print(f"{'='*65}")

    # Save
    result = {
        "pair": pair,
        "seed": args.seed,
        "architecture": f"{n_in}→{arch_str}→3{skip_str}",
        "layers": layer_widths,
        "activations": act_names,
        "skip": args.skip,
        "n_params": n_params,
        "elapsed_s": round(elapsed, 1),
        "extras": extras_list,
        "is": is_full,
        "oos": oos,
        "hard_gates": passed,
        "is_min_chunk_pps": round(float(min_pps), 2) if min_pps else None,
    }
    with open(RESULTS_DIR / f"{tag}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULTS_DIR / f"{tag}_best.pkl", "wb") as f:
        pickle.dump({
            "weights": final_vec,
            "layout": layout,
            "act_ids": act_ids,
            "is_full": is_full,
            "oos": oos,
            "passed_hard_gates": passed,
            "is_min_chunk_pps": min_pps,
            "architecture": result["architecture"],
            "extras": extras_list,
            "layer_widths": layer_widths,
            "act_names": act_names,
            "skip": args.skip,
        }, f)

    print(f"\nSaved: {tag}_result.json / {tag}_best.pkl")


if __name__ == "__main__":
    main()
