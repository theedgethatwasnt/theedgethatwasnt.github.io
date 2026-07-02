#!/usr/bin/env python3
"""
CMA-NN v2: Parameterized fixed-topology NN trained with CMA-ES + IPOP restarts.
================================================================================

Differences vs train_cma_5in.py
-------------------------------
1. **Feature modes** via `--features` flag.
   * `slopes`       — 5 inputs:  m5_slope, h1_slope, upnl, mae, mfe
   * `slopes_shap`  — 7 inputs:  m5_slope, h1_slope, tec_5, bb_width, upnl, mae, mfe
                                 (top-2 SHAP M5 indicators added)
2. **IPOP-CMA-ES** restart wrapper via `--restarts` flag.
   After each CMA-ES instance stops, restart with doubled popsize and reset
   sigma. Track best across restarts.
3. **Parameterized n_in** — simulator takes (market_features, n_in) so the same
   numba-jit code handles both modes; position state (upnl/mae/mfe) is always
   appended as the last 3 inputs.

Architecture (fixed)
--------------------
    n_in → 8 hidden (per-node wavelet activation) → 3 output (linear, argmax)

Wavelet bank (9 activations, encoded as continuous gene → bucketed):
    tanh, sin, cos, gauss, sech, dog, gabor, sinc, morlet

Gene layout (computed from n_in):
    [           0 :     n_in*8 ]   W1   (8 hidden × n_in inputs, row-major)
    [        W1E  :   W1E +  8 ]   b1   (8 biases)
    [        B1E  :   B1E + 24 ]   W2   (3 outputs × 8 hidden, row-major)
    [        W2E  :   W2E +  3 ]   b2   (3 biases)
    [        B2E  :   B2E +  8 ]   act  (8 activation genes)
    Total: n_in*8 + 8 + 24 + 3 + 8 = 8*n_in + 43

For n_in=5 → 83 params. For n_in=7 → 99 params.

Fitness (smooth, always-defined; matches fx-core SOP)
-----------------------------------------------------
    base_pps = total_pnl / total_days
    asym_penalty     = (1 - 2·dir_ratio) · 50
    activity_penalty = max(0, 30 - total_trades) · 2
    losing_pen       = sum(max(0, -chunk_pps)) · 2

    if every chunk profitable AND dir_ratio ≥ 0.15:
        score = min(chunk_pps) - asym_penalty
    else:
        score = base_pps - asym_penalty - activity_penalty - losing_pen
    return -score    (CMA-ES minimizes)

Spread is charged at entry (mae starts at spread_pips). Closing P&L uses raw mid.

Usage
-----
    # 5-input baseline
    python3 train_cma_v2.py --pair CHF_JPY --seed 42 --gens 200 --features slopes

    # 7-input with SHAP-top market features
    python3 train_cma_v2.py --pair CHF_JPY --seed 42 --gens 200 --features slopes_shap

    # IPOP-CMA-ES: 4 restarts, each 100 gens, doubled popsize per restart
    python3 train_cma_v2.py --pair CHF_JPY --seed 42 --gens 100 --restarts 4 --features slopes_shap
"""
import argparse
import gc
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import cma  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path(os.environ.get(
    "CMA5IN_DATA_DIR", str(PROJECT_ROOT / "data" / "unified_indicators")))
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

# ── Fixed architecture constants (n_in is dynamic) ─────────────────────
N_HID = 8
N_OUT = 3
N_ACTS = 9
N_POSITION_STATE = 3  # upnl, mae, mfe always last 3 inputs

ACT_NAMES = ["tanh", "sin", "cos", "gauss", "sech", "dog", "gabor", "sinc", "morlet"]


def n_params_for(n_in: int, fixed_act: bool = False) -> int:
    """Total CMA-ES parameter count.
    With wavelet bank: 8*n_in + 43 (8 activation genes appended).
    With fixed activation: 8*n_in + 35 (no activation genes).
    """
    base = n_in * N_HID + N_HID + N_HID * N_OUT + N_OUT
    if fixed_act:
        return base
    return base + N_HID


# ── Feature modes ──────────────────────────────────────────────────────
# Each mode lists the MARKET feature column names. Position state appended.
# Source: "unified" = data/unified_indicators/*.parquet (5y, has tec/h1_slope/bb_width/etc)
#         "v3"      = computed inline from data/m5_ohlc/*.parquet via lib/asi_indicator
#                     (curator-identical mc_d_a, mc_dd_a, er_norm)
FEATURE_MODES = {
    "slopes":      {"market": ["m5_slope", "h1_slope"],
                    "n_in": 5, "source": "unified", "normalize": {}},
    "slopes_shap": {"market": ["m5_slope", "h1_slope", "tec_5", "bb_width"],
                    "n_in": 7, "source": "unified",
                    "normalize": {"bb_width": ("mul", 20.0)}},
    # V3: same indicators as IronNet V3 (proven 42 p/day avg, CHF_JPY 65.5 p/day)
    # 6 inputs: 3 market + 3 position state (vs IronNet V3 which has only upnl)
    "v3":          {"market": ["mc_d_a", "mc_dd_a", "er_norm"],
                    "n_in": 6, "source": "v3", "normalize": {}},
    # V3 + 1 extra indicator (specified via --extra-feature). Used by sweep loop.
    # 7 inputs: 4 market + 3 position state.
    "v3_plus":     {"market": ["mc_d_a", "mc_dd_a", "er_norm", "<extra>"],
                    "n_in": 7, "source": "v3_plus", "normalize": {}},
}

# Per-extra-feature normalization for v3_plus mode
EXTRA_NORMALIZE = {
    "bb_width":   ("mul", 20.0),       # 0..0.05 → 0..1
    "macd_hist":  ("div_clip", 2.0),   # variable → clip [-1,1]
    "gap_norm":   ("div_clip", 1.0),
    # tec_5, h1_slope, aroon_osc are already in [-1,1]
    # stoch_d, range_pos_30, hl_price are already in [0,1]
    # m5_slope is arctan-normalized in [-1,1]
}


# ── Wavelet activation dispatch ────────────────────────────────────────
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
        zc = z
        if zc > 50.0:
            zc = 50.0
        elif zc < -50.0:
            zc = -50.0
        return 1.0 / np.cosh(zc)
    elif act_id == 5:
        return np.exp(-z * z / 2.0) - 0.5 * np.exp(-z * z / 8.0)
    elif act_id == 6:
        return np.exp(-2.0 * z * z) * np.cos(2.0 * np.pi * z)
    elif act_id == 7:
        if z > 1e-7 or z < -1e-7:
            return np.sin(np.pi * z) / (np.pi * z)
        return 1.0
    else:  # 8 morlet
        return np.sin(z) * np.exp(-z * z / 2.0)


@njit(cache=True)
def decode_act(gene):
    g = gene - np.floor(gene)
    aid = int(g * N_ACTS)
    if aid < 0:
        aid = 0
    if aid >= N_ACTS:
        aid = N_ACTS - 1
    return aid


# ── Curator-identical er_norm (Kaufman ER, arctan-normalized) ──────────
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


# ── Fast ASI momentum: raw D, no sign counting (Option D) ─────────────
@njit(cache=True)
def _d_to_fast_mc(smooth, n):
    """Convert SMA5(ASI) to fast mc_d/mc_dd: raw D normalized by rolling std.
    Skips 5-lag sign counting. ~15 min lag vs ~30 min for standard mc_d_a."""
    alpha3 = 2.0 / 4.0
    alpha5 = 2.0 / 6.0
    e3 = smooth[0]
    e5 = smooth[0]
    d_vals = np.zeros(n)
    for i in range(n):
        e3 = alpha3 * smooth[i] + (1.0 - alpha3) * e3
        e5 = alpha5 * smooth[i] + (1.0 - alpha5) * e5
        d_vals[i] = e3 - e5

    mc_d_fast = np.zeros(n)
    mc_dd_fast = np.zeros(n)
    window = 60

    for i in range(window, n):
        m = 0.0
        for j in range(i - window, i):
            m += d_vals[j]
        m /= window
        s = 0.0
        for j in range(i - window, i):
            s += (d_vals[j] - m) ** 2
        std = (s / window) ** 0.5
        if std > 1e-10:
            mc_d_fast[i] = np.tanh(d_vals[i] / std)
        else:
            mc_d_fast[i] = 0.0
        if i >= 2:
            dd = d_vals[i] - 2.0 * d_vals[i - 1] + d_vals[i - 2]
            mc_dd_fast[i] = np.tanh(dd / max(std * 0.5, 1e-10))

    return mc_d_fast, mc_dd_fast


def compute_asi_mc_fast(o, h, l, c, n):
    """Fast ASI MC: ASI → SMA5 → raw D (no sign counting)."""
    from lib.asi_indicator import compute_asi, sma_jit
    asi = compute_asi(o, h, l, c, n)
    smooth = sma_jit(asi, 5, n)
    return _d_to_fast_mc(smooth, n)


# ── Indicator: m5_slope (mirrors compute_h1_slope at M5 cadence) ───────
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


# ── Bar walker (parameterized n_in) ────────────────────────────────────
@njit(cache=True)
def simulate_chunk(
    market_features,    # shape (n_market, n_bars)
    mid_close,
    pip, spread_pips, max_hold,
    weights, n_in, fixed_act_id,
    chunk_start, chunk_end,
):
    """Same logic as v1 simulate_chunk but n_in is a runtime parameter.

    fixed_act_id:
        -1  → decode per-node activation from weights[b2_end : b2_end + N_HID]
       0..8 → use this ID for ALL hidden nodes (no activation genes in weights)

    Layout for weights[] (fixed_act_id == -1):
        W1  : indices [           0 :  n_in*N_HID            ]
        b1  : indices [   w1_end    :  w1_end + N_HID        ]
        W2  : indices [   b1_end    :  b1_end + N_HID*N_OUT  ]
        b2  : indices [   w2_end    :  w2_end + N_OUT        ]
        act : indices [   b2_end    :  b2_end + N_HID        ]   ← omitted when fixed
    """
    n_market = n_in - N_POSITION_STATE   # n_in includes upnl/mae/mfe at the end
    w1_end = n_in * N_HID
    b1_end = w1_end + N_HID
    w2_end = b1_end + N_HID * N_OUT
    b2_end = w2_end + N_OUT

    start_bar = max(chunk_start + 12, 12)
    end_bar = min(chunk_end, len(mid_close) - 1)
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

    x = np.zeros(n_in)
    h = np.zeros(N_HID)

    act_ids = np.zeros(N_HID, dtype=np.int64)
    if fixed_act_id < 0:
        for j in range(N_HID):
            act_ids[j] = decode_act(weights[b2_end + j])
    else:
        for j in range(N_HID):
            act_ids[j] = fixed_act_id

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

        # Build input vector: market features then 3 position states
        for k in range(n_market):
            x[k] = market_features[k, i]
        x[n_market]     = np.tanh(upnl_pips / 20.0)
        x[n_market + 1] = np.tanh(mae_pips / 20.0)
        x[n_market + 2] = np.tanh(mfe_pips / 20.0)

        # Forward pass: n_in → 8 hidden (per-node wavelet) → 3 (linear)
        for j in range(N_HID):
            z = weights[w1_end + j]   # bias b1[j]
            for k in range(n_in):
                z += weights[j * n_in + k] * x[k]
            h[j] = activate(z, act_ids[j])

        out_buy = weights[w2_end + 0]
        out_sell = weights[w2_end + 1]
        out_flat = weights[w2_end + 2]
        for j in range(N_HID):
            out_buy += weights[b1_end + 0 * N_HID + j] * h[j]
            out_sell += weights[b1_end + 1 * N_HID + j] * h[j]
            out_flat += weights[b1_end + 2 * N_HID + j] * h[j]

        if out_buy >= out_sell and out_buy >= out_flat:
            action = 1
        elif out_sell >= out_buy and out_sell >= out_flat:
            action = 2
        else:
            action = 0

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


# ── Fitness wrapper (shaped, always-defined) ───────────────────────────
def fitness_neg(weights, market, mid_close, pip, spread, max_hold, n_in,
                fixed_act_id, n_chunks, min_dir_ratio, bars_per_day=288.0):
    n_bars = len(mid_close)
    total_long = 0
    total_short = 0
    total_trades = 0
    total_pnl = 0.0
    chunk_pps = []
    losing_chunk_loss = 0.0

    for ci in range(n_chunks):
        c_start = int(n_bars * ci / n_chunks)
        c_end = int(n_bars * (ci + 1) / n_chunks)

        nt, pnl, nl, ns = simulate_chunk(
            market, mid_close,
            pip, spread, max_hold,
            weights, n_in, fixed_act_id, c_start, c_end,
        )
        total_long += nl
        total_short += ns
        total_trades += nt
        total_pnl += pnl

        n_days = (c_end - c_start) / bars_per_day
        pps = pnl / n_days if n_days > 0 else 0.0
        chunk_pps.append(pps)
        if pps < 0:
            losing_chunk_loss += -pps

    total_days = n_bars / bars_per_day
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
                      fixed_act_id, n_chunks, min_dir_ratio, bars_per_day=288.0):
    n_bars = len(mid_close)
    total_long = 0
    total_short = 0
    total_trades = 0
    chunk_pps = []
    for ci in range(n_chunks):
        c_start = int(n_bars * ci / n_chunks)
        c_end = int(n_bars * (ci + 1) / n_chunks)
        nt, pnl, nl, ns = simulate_chunk(
            market, mid_close, pip, spread, max_hold,
            weights, n_in, fixed_act_id, c_start, c_end)
        total_long += nl
        total_short += ns
        total_trades += nt
        n_days = (c_end - c_start) / bars_per_day
        min_trades = max(20, int(n_days * 0.5))
        if nt < min_trades or pnl <= 0:
            return False, None
        chunk_pps.append(pnl / n_days)
    if total_trades < 30:
        return False, None
    if min(total_long, total_short) / total_trades < min_dir_ratio:
        return False, None
    return True, min(chunk_pps)


# ── Multiprocessing worker ─────────────────────────────────────────────
_W = {}


def _worker_init(market, mid, pip, spread, max_hold, n_in, fixed_act_id,
                 n_chunks, min_dir, bpd=288.0):
    _W["market"] = market
    _W["mid"] = mid
    _W["pip"] = pip
    _W["spread"] = spread
    _W["max_hold"] = max_hold
    _W["n_in"] = n_in
    _W["fixed_act_id"] = fixed_act_id
    _W["n_chunks"] = n_chunks
    _W["min_dir"] = min_dir
    _W["bpd"] = bpd


def _worker_fit(vec):
    return fitness_neg(
        vec,
        _W["market"], _W["mid"], _W["pip"], _W["spread"], _W["max_hold"],
        _W["n_in"], _W["fixed_act_id"], _W["n_chunks"], _W["min_dir"],
        _W["bpd"],
    )


# ── OOS evaluator ──────────────────────────────────────────────────────
def eval_oos(weights, market, mid, pip, spread, max_hold, n_in, fixed_act_id,
             bars_per_day=288.0):
    nt, pnl, nl, ns = simulate_chunk(
        market, mid, pip, spread, max_hold, weights, n_in, fixed_act_id, 0, len(mid))
    n_days = len(mid) / bars_per_day
    return {
        "n_trades": int(nt),
        "total_pnl": round(float(pnl), 1),
        "pips_per_day": round(float(pnl) / max(n_days, 1), 2),
        "n_long": int(nl),
        "n_short": int(ns),
        "dir_ratio": round(min(nl, ns) / max(nt, 1), 3),
    }


# ── Single CMA-ES instance (one run, for IPOP wrapper) ─────────────────
def run_cma_once(x0, sigma0, popsize, gens, seed, pool, args, n_in, b2_end,
                 fixed_act_id,
                 m_is, mid_is, pip, spread, max_hold,
                 n_chunks, min_dir_ratio, t0, label="r0",
                 bars_per_day=288.0):
    """Run one CMA-ES instance to convergence or maxiter. Returns (best_vec, best_fit, best_valid_pps, best_valid_vec)."""
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
            fixed_act_id, n_chunks, min_dir_ratio, bars_per_day)
        if ok and (best_valid_pps is None or min_pps > best_valid_pps):
            best_valid_pps = min_pps
            best_valid_vec = np.array(best_vec)

        if gen % 10 == 0:
            valid_str = (f"{best_valid_pps:+.2f}p/d"
                         if best_valid_pps is not None else "—")
            print(f"  [{label}] Gen {gen:>3}: raw_fit={best_fit:>10.2f}  "
                  f"valid={valid_str}  σ={es.sigma:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s")
        gen += 1
        if gen >= gens:
            break

    return best_vec, best_fit, best_valid_pps, best_valid_vec, gen


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CMA-NN v2 trainer (IPOP, multi-feature)")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", choices=list(FEATURE_MODES.keys()),
                        default="slopes_shap")
    parser.add_argument("--gens", type=int, default=200,
                        help="Generations PER CMA-ES instance")
    parser.add_argument("--restarts", type=int, default=0,
                        help="IPOP restarts (0 = single run, 4 = 5 total runs)")
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--popsize-mult", type=float, default=2.0,
                        help="popsize multiplier per IPOP restart")
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--max-hold", type=int, default=200)
    parser.add_argument("--n-chunks", type=int, default=3)
    parser.add_argument("--min-dir-ratio", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label", type=str, default="cma_v2")
    parser.add_argument("--fixed-activation", type=str, default=None,
                        choices=ACT_NAMES,
                        help="Use this single activation for all hidden nodes "
                             "and drop activation genes (8 fewer params)")
    parser.add_argument("--extra-feature", type=str, default=None,
                        help="[Deprecated] Single extra feature for v3_plus. "
                             "Prefer --extras for multiple.")
    parser.add_argument("--extras", nargs="+", default=None,
                        help="For --features v3_plus: list of extra market "
                             "indicators (columns from data/unified_indicators, "
                             "or 'm5_slope'). Stacks any number of features.")
    parser.add_argument("--tf", choices=["M5", "H1"], default="M5",
                        help="Timeframe. H1 resamples M5-computed indicators "
                             "to hourly via .last() (matches IronNet H1 approach)")
    parser.add_argument("--fast-mc", action="store_true",
                        help="Use fast MC variant: raw D (arctan-normalized), "
                             "no 5-lag sign counting. ~15 min lag vs ~30 min.")
    parser.add_argument("--causal-parquet", type=str, default=None,
                        help="Smoother name (e.g. 'kalman10', 'sma5') — loads "
                             "data/m5_ohlc/{PAIR}_M5_{smoother}_causal.parquet "
                             "instead of computing indicators inline. Ensures "
                             "train/live feature parity.")
    args = parser.parse_args()

    fixed_act_id = -1
    if args.fixed_activation is not None:
        fixed_act_id = ACT_NAMES.index(args.fixed_activation)

    # Resolve extras list (priority: --extras > --extra-feature)
    extras_list = []
    if args.extras:
        extras_list = list(args.extras)
    elif args.extra_feature:
        extras_list = [args.extra_feature]

    np.random.seed(args.seed)
    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]
    mode = FEATURE_MODES[args.features]
    # For v3_plus, n_in is dynamic based on # of extras: 3 V3 + N extras + 3 state
    if args.features == "v3_plus":
        if not extras_list:
            print("ERROR: --extras (or --extra-feature) required for v3_plus mode")
            sys.exit(1)
        n_in = 3 + len(extras_list) + N_POSITION_STATE
    else:
        n_in = mode["n_in"]
    n_market = n_in - N_POSITION_STATE
    market_cols = mode["market"]
    n_params = n_params_for(n_in, fixed_act=(fixed_act_id >= 0))
    b2_end = n_in * N_HID + N_HID + N_HID * N_OUT + N_OUT  # for activation gene index

    print(f"{'='*65}")
    print(f"  CMA-NN v2: {pair} | features={args.features} (n_in={n_in})")
    print(f"  Topology: {n_in}→{N_HID}→{N_OUT}  ({n_params} params)")
    print(f"  Market features: {market_cols}")
    print(f"  Position state:  upnl, mae, mfe (always last)")
    if fixed_act_id >= 0:
        print(f"  Hidden activation: FIXED to '{ACT_NAMES[fixed_act_id]}' "
              f"(no act genes, n_params={n_params})")
    else:
        print(f"  Hidden act bank ({N_ACTS}): {', '.join(ACT_NAMES)}")
    print(f"  Output: linear, argmax → BUY/SELL/FLATTEN")
    print(f"  Seed: {args.seed} | popsize {args.popsize} | sigma0 {args.sigma0}")
    print(f"  Workers: {args.workers} | Gens/run: {args.gens} | Restarts: {args.restarts}")
    print(f"  WF chunks: {args.n_chunks} | min_dir {args.min_dir_ratio}")
    print(f"  Max hold: {args.max_hold} M5 bars")
    print(f"{'='*65}")

    # ── Load data ──────────────────────────────────────────────
    source = mode.get("source", "unified")

    # CAUSAL PARQUET PATH (overrides source when --causal-parquet is set)
    if args.causal_parquet:
        parquet_path = PROJECT_ROOT / f"data/m5_ohlc/{pair}_M5_{args.causal_parquet}_causal.parquet"
        if not parquet_path.exists():
            print(f"ERROR: causal parquet not found: {parquet_path}")
            print(f"Build it first: python3 build_causal_parquets.py --smoother {args.causal_parquet}")
            sys.exit(1)
        print(f"  Loading causal parquet: {parquet_path.name}")
        df_causal = pd.read_parquet(parquet_path, engine="pyarrow")
        n = len(df_causal)
        mid = df_causal["close"].values.astype(np.float64)
        # For v3_plus: mc_d_a, mc_dd_a, er_norm + extras (only macd_hist supported for now)
        cols = {
            "mc_d_a": df_causal["mc_d_a"].values.astype(np.float64),
            "mc_dd_a": df_causal["mc_dd_a"].values.astype(np.float64),
            "er_norm": df_causal["er_norm"].values.astype(np.float64),
        }
        for ex in extras_list:
            if ex == "macd_hist" and "macd_hist" in df_causal.columns:
                # Normalize macd_hist to [-1,+1] via /2+clip (matches EXTRA_NORMALIZE)
                cols["macd_hist"] = np.clip(df_causal["macd_hist"].values.astype(np.float64) / 2.0, -1.0, 1.0)
            else:
                print(f"ERROR: extra '{ex}' not available in causal parquet yet")
                sys.exit(1)
        market_cols = ["mc_d_a", "mc_dd_a", "er_norm"] + extras_list
        del df_causal
        gc.collect()
        # Skip the usual feature computation paths below — jump to market_full stacking
        skip_unified = True
    else:
        skip_unified = False

    if skip_unified:
        pass  # already loaded from causal parquet
    elif source == "v3_plus":
        # V3 indicators (computed inline from M5 OHLC) + N extra columns from
        # unified_indicators, inner-merged on timestamp.
        ohlc_path = PROJECT_ROOT / "data" / "m5_ohlc" / f"{pair}_M5.parquet"
        uni_path = DATA_DIR / f"{pair}_unified.parquet"
        if not ohlc_path.exists() or not uni_path.exists():
            print(f"ERROR: missing data file ({ohlc_path} or {uni_path})")
            sys.exit(1)
        df_o = pd.read_parquet(ohlc_path, engine="pyarrow")
        df_u = pd.read_parquet(uni_path, engine="pyarrow")

        # Build a frame of extras keyed by timestamp.
        from extra_indicators import is_inline_computable, compute_inline
        extras_frame = pd.DataFrame({"timestamp": df_o["timestamp"]})
        o_arr = df_o["open"].values.astype(np.float64)
        h_arr = df_o["high"].values.astype(np.float64)
        l_arr = df_o["low"].values.astype(np.float64)
        c_arr_pre = df_o["close"].values.astype(np.float64)
        for ex in extras_list:
            if ex == "m5_slope":
                extras_frame[ex] = compute_m5_slope(c_arr_pre, lookback=12)
            elif is_inline_computable(ex):
                extras_frame[ex] = compute_inline(ex, o_arr, h_arr, l_arr, c_arr_pre)
            elif ex in df_u.columns:
                tmp = df_u[["timestamp", ex]]
                extras_frame = extras_frame.merge(tmp, on="timestamp", how="left")
            else:
                print(f"ERROR: '{ex}' is neither inline-computable nor in "
                      f"unified parquet. Available: {list(df_u.columns)}")
                sys.exit(1)
        del o_arr, h_arr, l_arr, c_arr_pre

        df = df_o.merge(extras_frame, on="timestamp", how="inner")
        if len(df) < 1000:
            print(f"ERROR: merge produced only {len(df)} rows — check timestamps")
            sys.exit(1)
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        c_arr = df["close"].values.astype(np.float64)
        mid = c_arr.copy()
        n = len(mid)
        if args.fast_mc:
            mc_d_a, mc_dd_a = compute_asi_mc_fast(o, h, l, c_arr, n)
            print("  Using FAST MC (raw D, no sign counting)")
        else:
            from lib.asi_indicator import compute_asi_mc as _compute_asi_mc
            mc_d_a, mc_dd_a = _compute_asi_mc(o, h, l, c_arr, n)
        er_norm_arr = _compute_er_norm_v3(c_arr, window=60)

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

        market_cols = ["mc_d_a", "mc_dd_a", "er_norm"] + extras_list
        del df, df_o, df_u, extras_frame, o, h, l, c_arr
        gc.collect()
    elif source == "v3":
        # Load M5 OHLC; compute asi_mc + er_norm inline using curator-identical code
        ohlc_path = PROJECT_ROOT / "data" / "m5_ohlc" / f"{pair}_M5.parquet"
        if not ohlc_path.exists():
            print(f"ERROR: {ohlc_path} not found")
            sys.exit(1)
        df = pd.read_parquet(ohlc_path, engine="pyarrow")
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        c_arr = df["close"].values.astype(np.float64)
        mid = c_arr.copy()   # use close as mid (matches curator behaviour)
        n = len(mid)
        del df
        gc.collect()

        # Curator-identical V3 indicators
        if args.fast_mc:
            mc_d_a, mc_dd_a = compute_asi_mc_fast(o, h, l, c_arr, n)
            print("  Using FAST MC (raw D, no sign counting)")
        else:
            from lib.asi_indicator import compute_asi_mc as _compute_asi_mc
            mc_d_a, mc_dd_a = _compute_asi_mc(o, h, l, c_arr, n)
        er_norm = _compute_er_norm_v3(c_arr, window=60)
        cols = {"mc_d_a": mc_d_a, "mc_dd_a": mc_dd_a, "er_norm": er_norm}
        del o, h, l, c_arr
        gc.collect()
    else:
        path = DATA_DIR / f"{pair}_unified.parquet"
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)
        df = pd.read_parquet(path, engine="pyarrow")
        mid = df["mid_close"].values.astype(np.float64)
        n = len(mid)

        cols = {}
        for c in market_cols:
            if c == "m5_slope":
                cols["m5_slope"] = compute_m5_slope(mid, lookback=12)
            elif c in df.columns:
                cols[c] = df[c].values.astype(np.float64)
            else:
                print(f"ERROR: feature '{c}' not found in {path}")
                sys.exit(1)
        del df
        gc.collect()

    # Apply normalization
    for c, (op, val) in mode.get("normalize", {}).items():
        if c not in cols:
            continue
        if op == "mul":
            cols[c] = cols[c] * val
        elif op == "div_clip":
            cols[c] = np.clip(cols[c] / val, -1.0, 1.0)

    market_full = np.stack([cols[c] for c in market_cols], axis=0)

    # ── H1 resampling (M5 indicators → hourly via .last()) ────
    if args.tf == "H1":
        # Build a DataFrame with timestamp + all indicators + mid, resample
        resample_df = pd.DataFrame({"timestamp": pd.to_datetime(
            pd.read_parquet(
                PROJECT_ROOT / "data" / "m5_ohlc" / f"{pair}_M5.parquet",
                columns=["timestamp"], engine="pyarrow"
            )["timestamp"][:n]
        )})
        resample_df["mid"] = mid
        for i, c in enumerate(market_cols):
            resample_df[c] = market_full[i]
        resample_df = resample_df.set_index("timestamp").resample("1h").last().dropna()
        mid = resample_df["mid"].values.astype(np.float64)
        market_full = np.stack(
            [resample_df[c].values.astype(np.float64) for c in market_cols], axis=0
        )
        n = len(mid)
        del resample_df
        gc.collect()
        bars_per_day = 24.0
        print(f"\nResampled to H1: {n:,} bars ({n/24:.0f} days)")
    else:
        bars_per_day = 288.0

    split = int(n * 0.7)
    market_is = market_full[:, :split].copy()
    mid_is = mid[:split].copy()
    market_oos = market_full[:, split:].copy()
    mid_oos = mid[split:].copy()
    del market_full, mid, cols
    gc.collect()

    tf_label = args.tf
    print(f"\nData: {n:,} {tf_label} bars | IS: {split:,} | OOS: {n - split:,}")
    for i, c in enumerate(market_cols):
        print(f"  {c:12s} range: [{market_is[i].min():+.3f}, "
              f"{market_is[i].max():+.3f}]  std={market_is[i].std():.3f}")

    # ── JIT warmup ─────────────────────────────────────────────
    print("\nJIT warming up...")
    warm_w = np.zeros(n_params)
    simulate_chunk(market_is[:, :200], mid_is[:200], pip, spread, 50,
                   warm_w, n_in, fixed_act_id, 0, 200)
    print("  warm.")

    # ── Worker pool ────────────────────────────────────────────
    init_args = (market_is, mid_is, pip, spread, args.max_hold, n_in,
                 fixed_act_id, args.n_chunks, args.min_dir_ratio, bars_per_day)
    pool = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=init_args,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag_suffix = ("_" + "+".join(extras_list)) if extras_list else ""
    tag = f"{args.label}_{args.features}{tag_suffix}_{pair}_s{args.seed}"
    t0 = time.time()

    overall_best_fit = 1e18
    overall_best_vec = None
    overall_best_valid_pps = None
    overall_best_valid_vec = None

    try:
        for ri in range(args.restarts + 1):
            popsize = int(args.popsize * (args.popsize_mult ** ri))
            sigma = args.sigma0 * (1.5 ** ri)   # mild sigma growth (less aggressive than IPOP standard)
            seed = args.seed + ri * 1000

            # Random fresh start each restart
            rng = np.random.RandomState(seed)
            x0 = rng.randn(n_params) * 0.3
            if fixed_act_id < 0:
                x0[b2_end:b2_end + N_HID] = rng.uniform(0.0, 1.0, N_HID)

            label = f"r{ri}"
            print(f"\n── Restart {ri}/{args.restarts}: popsize={popsize} sigma0={sigma:.3f} seed={seed} ──")
            best_vec, best_fit, best_valid_pps, best_valid_vec, gens_done = run_cma_once(
                x0, sigma, popsize, args.gens, seed, pool, args,
                n_in, b2_end, fixed_act_id,
                market_is, mid_is, pip, spread, args.max_hold,
                args.n_chunks, args.min_dir_ratio, t0, label=label,
                bars_per_day=bars_per_day,
            )

            if best_fit < overall_best_fit:
                overall_best_fit = best_fit
                overall_best_vec = best_vec
            if best_valid_pps is not None and (overall_best_valid_pps is None or best_valid_pps > overall_best_valid_pps):
                overall_best_valid_pps = best_valid_pps
                overall_best_valid_vec = best_valid_vec
                # Save valid checkpoint
                with open(RESULTS_DIR / f"{tag}_r{ri}_valid.pkl", "wb") as f:
                    pickle.dump({
                        "weights": overall_best_valid_vec,
                        "min_chunk_pps": overall_best_valid_pps,
                        "n_in": n_in, "n_hid": N_HID, "n_out": N_OUT,
                        "n_params": n_params, "features": args.features,
                        "restart": ri,
                    }, f)
            print(f"  [{label}] done {gens_done} gens. raw_fit={best_fit:.2f}  "
                  f"valid={best_valid_pps if best_valid_pps else '—'}")
    finally:
        pool.shutdown(wait=True)

    elapsed = time.time() - t0

    if overall_best_vec is None:
        print(f"\nNo candidates produced — check setup.")
        sys.exit(2)

    final_vec = overall_best_valid_vec if overall_best_valid_vec is not None else overall_best_vec
    passed_gates = overall_best_valid_vec is not None

    # ── Final evaluation ───────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Training complete: {args.restarts + 1} run(s), {elapsed:.0f}s total")
    print(f"  Overall raw fitness: {overall_best_fit:.4f}")
    if passed_gates:
        print(f"  Hard-gate winner:    min chunk = {overall_best_valid_pps:+.2f} pips/day")
    else:
        print(f"  Hard-gate winner:    NONE — no candidate satisfied all WF gates")
    print(f"{'='*65}")

    is_full = eval_oos(final_vec, market_is, mid_is, pip, spread, args.max_hold,
                       n_in, fixed_act_id, bars_per_day)
    oos = eval_oos(final_vec, market_oos, mid_oos, pip, spread, args.max_hold,
                   n_in, fixed_act_id, bars_per_day)
    print(f"\nIS  full:  {is_full}")
    print(f"OOS full:  {oos}")

    if fixed_act_id >= 0:
        chosen_acts = [ACT_NAMES[fixed_act_id]] * N_HID
        print(f"\nHidden activations (fixed): {ACT_NAMES[fixed_act_id]} × {N_HID}")
    else:
        chosen_acts = [ACT_NAMES[decode_act(final_vec[b2_end + j])] for j in range(N_HID)]
        print(f"\nHidden activations: {chosen_acts}")

    final_path = RESULTS_DIR / f"{tag}_best.pkl"
    with open(final_path, "wb") as f:
        pickle.dump({
            "weights": final_vec,
            "n_in": n_in, "n_hid": N_HID, "n_out": N_OUT, "n_params": n_params,
            "features": args.features, "market_cols": market_cols,
            "pair": pair, "seed": args.seed,
            "passed_hard_gates": passed_gates,
            "is_min_chunk_pps": overall_best_valid_pps,
            "is_full": is_full,
            "oos": oos,
            "hidden_activations": chosen_acts,
            "args": vars(args),
        }, f)
    print(f"\nSaved: {final_path}")


if __name__ == "__main__":
    main()
