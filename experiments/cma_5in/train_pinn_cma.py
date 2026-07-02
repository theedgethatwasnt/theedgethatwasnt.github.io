#!/usr/bin/env python3
"""
PINN-CMA: Physics-Informed CMA-ES Trainer with 6 experimental arms.
====================================================================

Extends train_cma_5in.py with PINN-inspired techniques:
  - Adaptive activation scale per hidden node (all arms)
  - Expanded inputs: StrengthSpread H1 + d_close3 (T-A, T-AB, T-AC, T-ABC)
  - Learnable hyperparameters: max_hold + entry_threshold (T-B, T-AB, T-ABC)
  - Multi-term PINN fitness with constraint penalties (T-AC, T-ABC)

Arms (--mode flag):
  baseline       C0: 5 inputs, 91 params, original fitness
  inputs         T-A: 7 inputs, 107 params, original fitness
  hyper          T-B: 5 inputs, 93 params, original fitness
  inputs_hyper   T-AB: 7 inputs, 109 params, original fitness
  inputs_fitness T-AC: 7 inputs, 107 params, PINN fitness
  full           T-ABC: 7 inputs, 109 params, PINN fitness

Usage:
  python3 train_pinn_cma.py --pair EUR_JPY --mode baseline --seed 42
  python3 train_pinn_cma.py --pair CAD_JPY --mode full --seed 777 --gens 200
"""
import argparse
import gc
import json
import math
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path(os.environ.get(
    "CMA5IN_DATA_DIR", str(PROJECT_ROOT / "data" / "unified_indicators")))
SS_DIR = PROJECT_ROOT / "data" / "pinn_features"
RESULTS_DIR = SCRIPT_DIR / "results_pinn"

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

# ── Architecture constants ─────────────────────────────────────────────
N_HID = 8
N_OUT = 3
N_ACTS = 9   # tanh, sin, cos, gauss, sech, dog, gabor, sinc, morlet
ACT_NAMES = ["tanh", "sin", "cos", "gauss", "sech", "dog", "gabor", "sinc", "morlet"]

MODES = {
    "baseline":       {"n_in": 5, "use_ss": False, "learnable_hyper": False, "pinn_fitness": False},
    "inputs":         {"n_in": 7, "use_ss": True,  "learnable_hyper": False, "pinn_fitness": False},
    "hyper":          {"n_in": 5, "use_ss": False, "learnable_hyper": True,  "pinn_fitness": False},
    "inputs_hyper":   {"n_in": 7, "use_ss": True,  "learnable_hyper": True,  "pinn_fitness": False},
    "inputs_fitness": {"n_in": 7, "use_ss": True,  "learnable_hyper": False, "pinn_fitness": True},
    "full":           {"n_in": 7, "use_ss": True,  "learnable_hyper": True,  "pinn_fitness": True},
}


def compute_layout(n_in, learnable_hyper):
    """Compute gene layout offsets for given config."""
    w1_end = n_in * N_HID
    b1_end = w1_end + N_HID
    w2_end = b1_end + N_HID * N_OUT
    b2_end = w2_end + N_OUT
    act_end = b2_end + N_HID          # activation selection genes
    scale_end = act_end + N_HID       # activation scale genes
    if learnable_hyper:
        hyper_end = scale_end + 2     # max_hold_gene, threshold_gene
    else:
        hyper_end = scale_end
    return {
        "w1_end": w1_end, "b1_end": b1_end, "w2_end": w2_end,
        "b2_end": b2_end, "act_end": act_end, "scale_end": scale_end,
        "hyper_end": hyper_end, "n_params": hyper_end,
    }


# ── Wavelet activation bank ───────────────────────────────────────────
@njit(cache=True, inline="always")
def activate(z, act_id):
    if act_id == 0:
        return np.tanh(z)
    elif act_id == 1:
        return np.sin(z)
    elif act_id == 2:
        return np.cos(z)
    elif act_id == 3:
        return np.exp(-z * z)
    elif act_id == 4:
        zc = min(max(z, -50.0), 50.0)
        return 1.0 / np.cosh(zc)
    elif act_id == 5:
        return np.exp(-z * z / 2.0) - 0.5 * np.exp(-z * z / 8.0)
    elif act_id == 6:
        return np.exp(-2.0 * z * z) * np.cos(2.0 * np.pi * z)
    elif act_id == 7:
        if z > 1e-7 or z < -1e-7:
            return np.sin(np.pi * z) / (np.pi * z)
        return 1.0
    else:  # morlet
        return np.sin(z) * np.exp(-z * z / 2.0)


@njit(cache=True, inline="always")
def decode_act(gene):
    g = gene - np.floor(gene)
    aid = int(g * N_ACTS)
    if aid < 0:
        aid = 0
    if aid >= N_ACTS:
        aid = N_ACTS - 1
    return aid


@njit(cache=True, inline="always")
def decode_scale(gene):
    """Sigmoid(gene) mapped to [0.1, 5.0]."""
    s = 1.0 / (1.0 + np.exp(-gene))
    return 0.1 + s * 4.9


@njit(cache=True, inline="always")
def decode_max_hold(gene):
    """Gene → max_hold in [50, 500]."""
    g = 1.0 / (1.0 + np.exp(-gene))  # sigmoid → [0,1]
    return int(50 + g * 450)


@njit(cache=True, inline="always")
def decode_threshold(gene):
    """Gene → entry threshold in [0, 0.5]."""
    s = 1.0 / (1.0 + np.exp(-gene))
    return s * 0.5


# ── m5_slope computation ───────────────────────────────────────────────
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


@njit(cache=True)
def compute_d_close3(closes, pip):
    """3-bar momentum: tanh((close[i] - close[i-3]) / pip / 10)."""
    n = len(closes)
    out = np.zeros(n)
    for i in range(3, n):
        out[i] = np.tanh((closes[i] - closes[i - 3]) / pip / 10.0)
    return out


# ── Bar simulation (unified for all arms) ──────────────────────────────
@njit(cache=True)
def simulate_chunk(
    m5_slope, h1_slope, mid_close,
    ss_h1, d_close3,            # may be empty arrays if not used
    pip, spread_pips, max_hold, entry_threshold,
    weights,
    n_in, w1_end, b1_end, w2_end, b2_end, act_end, scale_end,
    chunk_start, chunk_end,
):
    """Walk bars, apply NN action. Returns extended stats tuple:
    (n_trades, total_pnl, n_long, n_short, n_contrarian, total_hold, n_short_hold, n_winning)
    """
    use_extra = n_in >= 7
    start_bar = max(chunk_start + 12, 12)
    end_bar = min(chunk_end, len(mid_close) - 1)
    n_capacity = end_bar - start_bar + 1
    if n_capacity <= 0:
        return 0, 0.0, 0, 0, 0, 0, 0, 0

    pnls = np.zeros(n_capacity)
    n_trades = 0
    n_long = 0
    n_short = 0
    n_contrarian = 0
    total_hold = 0
    n_short_hold = 0
    n_winning = 0

    position = 0
    entry_price = 0.0
    entry_bar = 0
    entry_ss = 0.0
    upnl_pips = 0.0
    mae_pips = 0.0
    mfe_pips = 0.0

    x = np.zeros(n_in)
    h = np.zeros(N_HID)

    # Decode per-hidden activation IDs and scales once
    act_ids = np.zeros(N_HID, dtype=np.int64)
    scales = np.zeros(N_HID)
    for j in range(N_HID):
        act_ids[j] = decode_act(weights[b2_end + j])
        scales[j] = decode_scale(weights[act_end + j])

    for i in range(start_bar, end_bar):
        # ── Update position metrics ──────────────────────────────
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

        # ── Build input vector ───────────────────────────────────
        x[0] = m5_slope[i]
        x[1] = h1_slope[i]
        x[2] = np.tanh(upnl_pips / 20.0)
        x[3] = np.tanh(mae_pips / 20.0)
        x[4] = np.tanh(mfe_pips / 20.0)
        if use_extra:
            x[5] = ss_h1[i]
            x[6] = d_close3[i]

        # ── Forward pass: N_IN → 8 (scaled wavelet) → 3 (linear) ─
        for j in range(N_HID):
            z = weights[w1_end + j]   # bias b1[j]
            for k in range(n_in):
                z += weights[j * n_in + k] * x[k]
            h[j] = activate(scales[j] * z, act_ids[j])

        out_buy = weights[w2_end + 0]
        out_sell = weights[w2_end + 1]
        out_flat = weights[w2_end + 2]
        for j in range(N_HID):
            out_buy += weights[b1_end + 0 * N_HID + j] * h[j]
            out_sell += weights[b1_end + 1 * N_HID + j] * h[j]
            out_flat += weights[b1_end + 2 * N_HID + j] * h[j]

        # ── Action with entry threshold ──────────────────────────
        if out_buy >= out_sell and out_buy >= out_flat:
            best = out_buy
            second = max(out_sell, out_flat)
            action = 1 if (best - second) >= entry_threshold else 0
        elif out_sell >= out_buy and out_sell >= out_flat:
            best = out_sell
            second = max(out_buy, out_flat)
            action = 2 if (best - second) >= entry_threshold else 0
        else:
            action = 0  # FLATTEN

        # ── Force-close on max_hold ──────────────────────────────
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid_close[i] - entry_price) / pip * position
            hold_len = i - entry_bar
            if n_trades < n_capacity:
                pnls[n_trades] = pnl
                if position > 0:
                    n_long += 1
                else:
                    n_short += 1
                if pnl > 0:
                    n_winning += 1
                # Contrarian check: trade went WITH SS direction?
                if use_extra and position * entry_ss > 0:
                    n_contrarian += 1
                total_hold += hold_len
                if hold_len < 12:
                    n_short_hold += 1
                n_trades += 1
            position = 0
            entry_price = 0.0
            mae_pips = 0.0
            mfe_pips = 0.0

        # ── Apply action ─────────────────────────────────────────
        if position == 0:
            if action == 1:
                position = 1
                entry_price = mid_close[i] + spread_pips * pip
                entry_bar = i
                entry_ss = ss_h1[i] if use_extra else 0.0
                mae_pips = spread_pips
                mfe_pips = 0.0
            elif action == 2:
                position = -1
                entry_price = mid_close[i] - spread_pips * pip
                entry_bar = i
                entry_ss = ss_h1[i] if use_extra else 0.0
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
                hold_len = i - entry_bar
                if n_trades < n_capacity:
                    pnls[n_trades] = pnl
                    if position > 0:
                        n_long += 1
                    else:
                        n_short += 1
                    if pnl > 0:
                        n_winning += 1
                    if use_extra and position * entry_ss > 0:
                        n_contrarian += 1
                    total_hold += hold_len
                    if hold_len < 12:
                        n_short_hold += 1
                    n_trades += 1
                position = new_pos
                if new_pos == 1:
                    entry_price = mid_close[i] + spread_pips * pip
                    entry_bar = i
                    entry_ss = ss_h1[i] if use_extra else 0.0
                    mae_pips = spread_pips
                    mfe_pips = 0.0
                elif new_pos == -1:
                    entry_price = mid_close[i] - spread_pips * pip
                    entry_bar = i
                    entry_ss = ss_h1[i] if use_extra else 0.0
                    mae_pips = spread_pips
                    mfe_pips = 0.0
                else:
                    entry_price = 0.0

    # Close open position at chunk end
    if position != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar - 1] - entry_price) / pip * position
        hold_len = (end_bar - 1) - entry_bar
        if n_trades < n_capacity:
            pnls[n_trades] = pnl
            if position > 0:
                n_long += 1
            else:
                n_short += 1
            if pnl > 0:
                n_winning += 1
            if use_extra and position * entry_ss > 0:
                n_contrarian += 1
            total_hold += hold_len
            if hold_len < 12:
                n_short_hold += 1
            n_trades += 1

    if n_trades < 1:
        return 0, 0.0, 0, 0, 0, 0, 0, 0
    total = 0.0
    for k in range(n_trades):
        total += pnls[k]
    return n_trades, total, n_long, n_short, n_contrarian, total_hold, n_short_hold, n_winning


# ── Fitness functions ──────────────────────────────────────────────────
def fitness_original(weights, m5_slope, h1_slope, mid_close, ss_h1, d_close3,
                     pip, spread, max_hold_default, entry_threshold_default,
                     n_chunks, min_dir_ratio, layout, mode_cfg):
    """Original fitness (same as train_cma_5in with adaptive scale)."""
    n_in = mode_cfg["n_in"]
    ly = layout

    # Decode learnable hypers if applicable
    if mode_cfg["learnable_hyper"]:
        max_hold = decode_max_hold(weights[ly["scale_end"]])
        entry_threshold = decode_threshold(weights[ly["scale_end"] + 1])
    else:
        max_hold = max_hold_default
        entry_threshold = entry_threshold_default

    n_bars = len(mid_close)
    total_long = 0
    total_short = 0
    total_trades = 0
    total_pnl = 0.0
    total_contrarian = 0
    total_hold_bars = 0
    total_short_holds = 0
    total_winning = 0
    chunk_pps = []
    losing_chunk_loss = 0.0

    for ci in range(n_chunks):
        c_start = int(n_bars * ci / n_chunks)
        c_end = int(n_bars * (ci + 1) / n_chunks)

        nt, pnl, nl, ns, nc, th, nsh, nw = simulate_chunk(
            m5_slope, h1_slope, mid_close, ss_h1, d_close3,
            pip, spread, max_hold, entry_threshold,
            weights,
            n_in, ly["w1_end"], ly["b1_end"], ly["w2_end"],
            ly["b2_end"], ly["act_end"], ly["scale_end"],
            c_start, c_end,
        )
        total_long += nl
        total_short += ns
        total_trades += nt
        total_pnl += pnl
        total_contrarian += nc
        total_hold_bars += th
        total_short_holds += nsh
        total_winning += nw

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


def fitness_pinn(weights, m5_slope, h1_slope, mid_close, ss_h1, d_close3,
                 pip, spread, max_hold_default, entry_threshold_default,
                 n_chunks, min_dir_ratio, layout, mode_cfg):
    """PINN multi-term fitness with constraint penalties."""
    n_in = mode_cfg["n_in"]
    ly = layout

    if mode_cfg["learnable_hyper"]:
        max_hold = decode_max_hold(weights[ly["scale_end"]])
        entry_threshold = decode_threshold(weights[ly["scale_end"] + 1])
    else:
        max_hold = max_hold_default
        entry_threshold = entry_threshold_default

    n_bars = len(mid_close)
    total_long = 0
    total_short = 0
    total_trades = 0
    total_pnl = 0.0
    total_contrarian = 0
    total_hold_bars = 0
    total_short_holds = 0
    total_winning = 0
    chunk_pps = []
    losing_chunk_loss = 0.0

    for ci in range(n_chunks):
        c_start = int(n_bars * ci / n_chunks)
        c_end = int(n_bars * (ci + 1) / n_chunks)

        nt, pnl, nl, ns, nc, th, nsh, nw = simulate_chunk(
            m5_slope, h1_slope, mid_close, ss_h1, d_close3,
            pip, spread, max_hold, entry_threshold,
            weights,
            n_in, ly["w1_end"], ly["b1_end"], ly["w2_end"],
            ly["b2_end"], ly["act_end"], ly["scale_end"],
            c_start, c_end,
        )
        total_long += nl
        total_short += ns
        total_trades += nt
        total_pnl += pnl
        total_contrarian += nc
        total_hold_bars += th
        total_short_holds += nsh
        total_winning += nw

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

    # ── Original penalties ───────────────────────────────────────
    asym_penalty = (1.0 - 2.0 * dir_ratio) * 50.0
    activity_penalty = max(0.0, 30.0 - total_trades) * 2.0
    losing_pen = losing_chunk_loss * 2.0

    all_profitable = all(p > 0 for p in chunk_pps)
    if all_profitable and dir_ratio >= min_dir_ratio:
        perf_score = min(chunk_pps) - asym_penalty
    else:
        perf_score = base_pps - asym_penalty - activity_penalty - losing_pen

    # ── PINN constraint penalties ────────────────────────────────
    avg_pnl = total_pnl / total_trades
    spread_violation = max(0.0, 2.0 * spread - avg_pnl)

    contrarian_rate = total_contrarian / total_trades
    contrarian_penalty = contrarian_rate * 5.0

    short_hold_rate = total_short_holds / total_trades
    short_hold_penalty = short_hold_rate * 3.0

    # ── Quality bonus ────────────────────────────────────────────
    win_rate = total_winning / total_trades
    quality_bonus = max(0.0, win_rate - 0.45) * 10.0

    score = (perf_score
             - spread_violation
             - contrarian_penalty
             - short_hold_penalty
             + 0.5 * quality_bonus)

    return -score


# ── Hard gates ─────────────────────────────────────────────────────────
def passes_hard_gates(weights, m5_slope, h1_slope, mid_close, ss_h1, d_close3,
                      pip, spread, max_hold, entry_threshold,
                      n_chunks, min_dir_ratio, layout, mode_cfg):
    n_in = mode_cfg["n_in"]
    ly = layout
    n_bars = len(mid_close)
    total_long = 0
    total_short = 0
    total_trades = 0
    chunk_pps = []
    for ci in range(n_chunks):
        c_start = int(n_bars * ci / n_chunks)
        c_end = int(n_bars * (ci + 1) / n_chunks)
        nt, pnl, nl, ns, nc, th, nsh, nw = simulate_chunk(
            m5_slope, h1_slope, mid_close, ss_h1, d_close3,
            pip, spread, max_hold, entry_threshold,
            weights,
            n_in, ly["w1_end"], ly["b1_end"], ly["w2_end"],
            ly["b2_end"], ly["act_end"], ly["scale_end"],
            c_start, c_end)
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


# ── OOS evaluator ──────────────────────────────────────────────────────
def eval_oos(weights, m5, h1, mid, ss, dc3, pip, spread, max_hold,
             entry_threshold, layout, mode_cfg):
    n_in = mode_cfg["n_in"]
    ly = layout
    nt, pnl, nl, ns, nc, th, nsh, nw = simulate_chunk(
        m5, h1, mid, ss, dc3,
        pip, spread, max_hold, entry_threshold,
        weights,
        n_in, ly["w1_end"], ly["b1_end"], ly["w2_end"],
        ly["b2_end"], ly["act_end"], ly["scale_end"],
        0, len(mid))
    n_days = len(mid) / 288.0
    return {
        "n_trades": int(nt),
        "total_pnl": round(float(pnl), 1),
        "pips_per_day": round(float(pnl) / max(n_days, 1), 2),
        "n_long": int(nl), "n_short": int(ns),
        "dir_ratio": round(min(nl, ns) / max(nt, 1), 3),
        "win_rate": round(nw / max(nt, 1), 3),
        "contrarian_rate": round(nc / max(nt, 1), 3),
        "avg_hold": round(th / max(nt, 1), 1),
        "short_hold_pct": round(nsh / max(nt, 1), 3),
    }


# ── Multiprocessing ───────────────────────────────────────────────────
_W = {}


def _worker_init(m5, h1, mid, ss, dc3, pip, spread, max_hold, entry_thresh,
                 n_chunks, min_dir, layout, mode_cfg, use_pinn):
    _W.update(m5=m5, h1=h1, mid=mid, ss=ss, dc3=dc3, pip=pip, spread=spread,
              max_hold=max_hold, entry_thresh=entry_thresh,
              n_chunks=n_chunks, min_dir=min_dir,
              layout=layout, mode_cfg=mode_cfg, use_pinn=use_pinn)


def _worker_fit(vec):
    fn = fitness_pinn if _W["use_pinn"] else fitness_original
    return fn(vec, _W["m5"], _W["h1"], _W["mid"], _W["ss"], _W["dc3"],
              _W["pip"], _W["spread"], _W["max_hold"], _W["entry_thresh"],
              _W["n_chunks"], _W["min_dir"], _W["layout"], _W["mode_cfg"])


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PINN-CMA experiment trainer")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--mode", required=True, choices=list(MODES.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--max-hold", type=int, default=200)
    parser.add_argument("--n-chunks", type=int, default=3)
    parser.add_argument("--min-dir-ratio", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    np.random.seed(args.seed)
    mode_cfg = MODES[args.mode]
    n_in = mode_cfg["n_in"]
    layout = compute_layout(n_in, mode_cfg["learnable_hyper"])
    n_params = layout["n_params"]

    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]

    print(f"{'='*70}")
    print(f"  PINN-CMA: {pair} | mode={args.mode}")
    print(f"  Topology: {n_in}→{N_HID}→{N_OUT}  ({n_params} params)")
    print(f"  Features: m5_slope, h1_slope, upnl, mae, mfe"
          + (", ss_h1, d_close3" if mode_cfg["use_ss"] else ""))
    print(f"  Adaptive scale: YES | Learnable hyper: {mode_cfg['learnable_hyper']}")
    print(f"  Fitness: {'PINN multi-term' if mode_cfg['pinn_fitness'] else 'original'}")
    print(f"  Seed={args.seed} pop={args.popsize} σ0={args.sigma0} gens={args.gens}")
    print(f"  Workers={args.workers} chunks={args.n_chunks} min_dir={args.min_dir_ratio}")
    print(f"{'='*70}")

    # ── Load data ──────────────────────────────────────────────
    path = DATA_DIR / f"{pair}_unified.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    df = pd.read_parquet(path, engine="pyarrow")
    mid = df["mid_close"].values.astype(np.float64)
    h1_slope_full = df["h1_slope"].values.astype(np.float64)
    m5_slope_full = compute_m5_slope(mid, lookback=12)
    del df; gc.collect()

    # StrengthSpread + d_close3
    if mode_cfg["use_ss"]:
        ss_path = SS_DIR / f"{pair}_ss_h1.npy"
        if not ss_path.exists():
            print(f"ERROR: {ss_path} not found — run precompute_strength_spread.py first")
            sys.exit(1)
        ss_full = np.load(ss_path).astype(np.float64)
        dc3_full = compute_d_close3(mid, pip)
    else:
        ss_full = np.zeros(1, dtype=np.float64)    # dummy
        dc3_full = np.zeros(1, dtype=np.float64)

    n = len(mid)
    split = int(n * 0.7)
    mid_is, mid_oos = mid[:split], mid[split:]
    h1_is, h1_oos = h1_slope_full[:split], h1_slope_full[split:]
    m5_is, m5_oos = m5_slope_full[:split], m5_slope_full[split:]
    if mode_cfg["use_ss"]:
        ss_is, ss_oos = ss_full[:split], ss_full[split:]
        dc3_is, dc3_oos = dc3_full[:split], dc3_full[split:]
    else:
        ss_is = ss_oos = ss_full
        dc3_is = dc3_oos = dc3_full

    print(f"\nData: {n:,} M5 bars | IS: {split:,} | OOS: {n - split:,}")

    # ── JIT warmup ─────────────────────────────────────────────
    print("JIT warming up...")
    warm = np.zeros(n_params)
    simulate_chunk(m5_is[:200], h1_is[:200], mid_is[:200],
                   ss_is[:200] if mode_cfg["use_ss"] else ss_is,
                   dc3_is[:200] if mode_cfg["use_ss"] else dc3_is,
                   pip, spread, 50, 0.0, warm,
                   n_in, layout["w1_end"], layout["b1_end"], layout["w2_end"],
                   layout["b2_end"], layout["act_end"], layout["scale_end"],
                   0, 200)
    print("  warm.")

    # ── CMA-ES init ────────────────────────────────────────────
    x0 = np.random.randn(n_params) * 0.3
    # Activation selection genes → uniform [0,1)
    x0[layout["b2_end"]:layout["act_end"]] = np.random.uniform(0.0, 1.0, N_HID)
    # Scale genes → 0.0 (sigmoid→0.5, scale≈2.6)
    x0[layout["act_end"]:layout["scale_end"]] = 0.0
    # Learnable hyper genes
    if mode_cfg["learnable_hyper"]:
        x0[layout["scale_end"]] = 0.0       # max_hold → sigmoid(0)=0.5 → 275
        x0[layout["scale_end"] + 1] = 0.0   # threshold → sigmoid(0)*0.5 = 0.25

    opts = {
        "popsize": args.popsize,
        "seed": args.seed,
        "verbose": -9,
        "tolx": 1e-9,
        "tolfun": 1e-3,
        "tolfunhist": 1e-3,
        "tolflatfitness": 50,
        "maxiter": args.gens,
    }
    es = cma.CMAEvolutionStrategy(x0, args.sigma0, opts)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"pinn_{args.mode}_{pair}_s{args.seed}"

    best_fit = 1e18
    best_vec = None
    best_valid_pps = None
    best_valid_vec = None
    t0 = time.time()

    entry_thresh_default = 0.0  # no threshold unless learnable_hyper

    init_args = (m5_is, h1_is, mid_is, ss_is, dc3_is,
                 pip, spread, args.max_hold, entry_thresh_default,
                 args.n_chunks, args.min_dir_ratio,
                 layout, mode_cfg, mode_cfg["pinn_fitness"])

    pool = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=init_args,
    )
    try:
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

            # Decode hypers for hard gate check
            if mode_cfg["learnable_hyper"]:
                mh = decode_max_hold(best_vec[layout["scale_end"]])
                et = decode_threshold(best_vec[layout["scale_end"] + 1])
            else:
                mh = args.max_hold
                et = entry_thresh_default

            ok, min_pps = passes_hard_gates(
                best_vec, m5_is, h1_is, mid_is, ss_is, dc3_is,
                pip, spread, mh, et,
                args.n_chunks, args.min_dir_ratio, layout, mode_cfg)
            if ok and (best_valid_pps is None or min_pps > best_valid_pps):
                best_valid_pps = min_pps
                best_valid_vec = np.array(best_vec)

            if gen % 10 == 0:
                valid_str = (f"{best_valid_pps:+.2f}p/d"
                             if best_valid_pps is not None else "—")
                hyper_str = ""
                if mode_cfg["learnable_hyper"] and best_vec is not None:
                    hyper_str = f"  mh={mh} et={et:.2f}"
                print(f"  Gen {gen:>3}: fit={best_fit:>10.2f}  "
                      f"valid={valid_str}  σ={es.sigma:.4f}{hyper_str}  "
                      f"{time.time()-t0:.0f}s")
            gen += 1
            if gen >= args.gens:
                break
    finally:
        pool.shutdown(wait=True)

    elapsed = time.time() - t0

    if best_vec is None:
        print(f"\nNo candidates in {gen} gens.")
        sys.exit(2)

    final_vec = best_valid_vec if best_valid_vec is not None else best_vec
    passed_gates = best_valid_vec is not None

    # Decode final hypers
    if mode_cfg["learnable_hyper"]:
        final_mh = decode_max_hold(final_vec[layout["scale_end"]])
        final_et = decode_threshold(final_vec[layout["scale_end"] + 1])
    else:
        final_mh = args.max_hold
        final_et = entry_thresh_default

    # ── OOS evaluation ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Training complete: {gen} gens, {elapsed:.0f}s")
    print(f"  Raw best fitness: {best_fit:.4f}")
    if passed_gates:
        print(f"  Hard-gate winner: min chunk = {best_valid_pps:+.2f} p/d")
    else:
        print(f"  Hard-gate winner: NONE")
    if mode_cfg["learnable_hyper"]:
        print(f"  Evolved max_hold={final_mh}  entry_threshold={final_et:.3f}")
    print(f"{'='*70}")

    is_full = eval_oos(final_vec, m5_is, h1_is, mid_is, ss_is, dc3_is,
                       pip, spread, final_mh, final_et, layout, mode_cfg)
    oos = eval_oos(final_vec, m5_oos, h1_oos, mid_oos, ss_oos, dc3_oos,
                   pip, spread, final_mh, final_et, layout, mode_cfg)
    print(f"\nIS  full:  {is_full}")
    print(f"OOS full:  {oos}")

    chosen_acts = [ACT_NAMES[decode_act(final_vec[layout["b2_end"] + j])]
                   for j in range(N_HID)]
    chosen_scales = [round(decode_scale(final_vec[layout["act_end"] + j]), 2)
                     for j in range(N_HID)]
    print(f"Hidden activations: {chosen_acts}")
    print(f"Hidden scales: {chosen_scales}")

    # ── Save ───────────────────────────────────────────────────
    result = {
        "weights": final_vec,
        "n_in": n_in, "n_hid": N_HID, "n_out": N_OUT,
        "n_params": n_params,
        "pair": pair, "seed": args.seed, "mode": args.mode,
        "passed_hard_gates": passed_gates,
        "is_min_chunk_pps": best_valid_pps,
        "is_full": is_full, "oos": oos,
        "hidden_activations": chosen_acts,
        "hidden_scales": chosen_scales,
        "evolved_max_hold": final_mh if mode_cfg["learnable_hyper"] else None,
        "evolved_threshold": final_et if mode_cfg["learnable_hyper"] else None,
        "elapsed_s": round(elapsed, 1),
        "args": vars(args),
    }

    final_path = RESULTS_DIR / f"{tag}_best.pkl"
    with open(final_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved: {final_path}")

    # Also save JSON summary for easy aggregation
    summary = {k: v for k, v in result.items() if k != "weights"}
    summary["is_min_chunk_pps"] = float(summary["is_min_chunk_pps"]) if summary["is_min_chunk_pps"] else None
    json_path = RESULTS_DIR / f"{tag}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
