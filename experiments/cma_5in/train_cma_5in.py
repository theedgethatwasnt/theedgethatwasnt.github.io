#!/usr/bin/env python3
"""
CMA-5in: Fixed-Topology 5→8→3 NN trained with CMA-ES (no NEAT, no crossover)
=============================================================================

Architecture
------------
Inputs (5):
    0  m5_slope        — 12-bar arctan-normalized slope at M5 cadence  ([-1, 1])
    1  h1_slope        — 3-bar arctan-normalized slope at H1 cadence   ([-1, 1])
                         (existing column in unified_indicators parquets)
    2  upnl_n          — tanh(upnl_pips / 20)                          ([-1, 1])
    3  mae_n           — tanh(mae_pips  / 20)   (mae stored as ≥0)     ([ 0, 1])
    4  mfe_n           — tanh(mfe_pips  / 20)   (mfe stored as ≥0)     ([ 0, 1])

Hidden: 8 neurons, per-node activation chosen from wavelet bank
        {tanh, sin, cos, gauss, sech, dog, gabor, sinc, morlet}
Output: 3 neurons (linear, argmax → {0:BUY, 1:SELL, 2:FLATTEN})

CMA-ES gene layout (83 params total)
------------------------------------
    [ 0:40]   W1   (8 hidden × 5 inputs, row-major)
    [40:48]   b1   (8 hidden biases)
    [48:72]   W2   (3 outputs × 8 hidden, row-major)
    [72:75]   b2   (3 output biases)
    [75:83]   act  (8 activation genes, continuous → bucketed via int(g×9))

Activation discretization
-------------------------
CMA-ES is continuous-only. We encode activation choice as a float in [0,1),
clipped at fitness time to one of 9 wavelet IDs. Boundary jitter is fine —
CMA-ES learns the smooth weight landscape inside each bucket and explores
neighbouring buckets via its sigma.

Training
--------
- CMA-ES (pycma) on flat 48-d weight vector. No topology mutation, no crossover.
- Per-pair training (matches IronNet pattern).
- Walk-forward in fitness: 3 chunks of IS data. Fitness = min(chunk pips/day).
- Hard gates: must be profitable in EVERY chunk + bidir ratio ≥ 15%.
- Parallel candidate evaluation via multiprocessing.Pool.
- 70/30 IS/OOS split.

Bar simulation
--------------
Numba JIT loop. State per bar:
    pip-based upnl, mae, mfe (mae/mfe stored ≥0, reset on entry).
    These are fed BACK into the network as inputs 2-4 → "self-aware" position state.

Spread charged at entry: entry_price = mid ± spread*pip, mae starts at spread_pips.
max_hold force-close on stale positions.

Usage
-----
    python3 train_cma_5in.py --pair EUR_GBP --seed 42 --gens 200
    python3 train_cma_5in.py --pair CAD_JPY --seed 42 --gens 200 --workers 8
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

# ── Architecture constants ─────────────────────────────────────────────
N_IN = 5
N_HID = 8
N_OUT = 3
N_ACTS = 9   # tanh, sin, cos, gauss, sech, dog, gabor, sinc, morlet

# Gene layout offsets:
W1_END = N_IN * N_HID                       # 40
B1_END = W1_END + N_HID                     # 48
W2_END = B1_END + N_HID * N_OUT             # 72
B2_END = W2_END + N_OUT                     # 75
ACT_END = B2_END + N_HID                    # 83
N_PARAMS = ACT_END                          # 83


# ── Wavelet activation bank (Numba-friendly dispatch) ─────────────────
# IDs: 0 tanh | 1 sin | 2 cos | 3 gauss | 4 sech | 5 dog | 6 gabor | 7 sinc | 8 morlet
ACT_NAMES = ["tanh", "sin", "cos", "gauss", "sech", "dog", "gabor", "sinc", "morlet"]


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
    """Continuous gene → activation ID. Genes can drift outside [0,1); we wrap."""
    g = gene - np.floor(gene)   # fractional part, in [0, 1)
    aid = int(g * N_ACTS)
    if aid < 0:
        aid = 0
    if aid >= N_ACTS:
        aid = N_ACTS - 1
    return aid


# ── Indicator: m5_slope (mirrors compute_h1_slope at M5 cadence) ───────
@njit(cache=True)
def compute_m5_slope(closes, lookback=12):
    """12-bar M5 linreg slope, arctan-normalized. Same shape as h1_slope."""
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


# ── Bar walker + fitness (Numba JIT) ───────────────────────────────────
@njit(cache=True)
def simulate_chunk(
    m5_slope, h1_slope, mid_close,
    pip, spread_pips, max_hold,
    weights,
    chunk_start, chunk_end,
):
    """Walk bars, apply NN action each bar. Returns (n_trades, total_pnl, n_long, n_short).

    weights layout (83 dims, see module docstring):
        [ 0:40]  W1   (8 hidden × 5 inputs, row-major)
        [40:48]  b1   (8 hidden biases)
        [48:72]  W2   (3 outputs × 8 hidden, row-major)
        [72:75]  b2   (3 output biases)
        [75:83]  act  (8 activation genes, decoded → IDs 0-8)

    Activations are decoded once per call (not per bar).
    """
    start_bar = max(chunk_start + 12, 12)
    end_bar = min(chunk_end, len(mid_close) - 1)
    n_capacity = end_bar - start_bar + 1
    if n_capacity <= 0:
        return 0, 0.0, 0, 0

    pnls = np.zeros(n_capacity)
    n_trades = 0
    n_long = 0
    n_short = 0

    position = 0          # +1 long, -1 short, 0 flat
    entry_price = 0.0
    entry_bar = 0
    upnl_pips = 0.0
    mae_pips = 0.0
    mfe_pips = 0.0

    x = np.zeros(N_IN)
    h = np.zeros(N_HID)

    # Decode per-hidden activation IDs once
    act_ids = np.zeros(N_HID, dtype=np.int64)
    for j in range(N_HID):
        act_ids[j] = decode_act(weights[B2_END + j])

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

        # ── Forward pass: 5 → 8 (per-node wavelet) → 3 (linear) ──
        for j in range(N_HID):
            z = weights[W1_END + j]   # bias b1[j]
            for k in range(N_IN):
                z += weights[j * N_IN + k] * x[k]
            h[j] = activate(z, act_ids[j])

        out_buy = weights[W2_END + 0]   # bias b2[0]
        out_sell = weights[W2_END + 1]  # bias b2[1]
        out_flat = weights[W2_END + 2]  # bias b2[2]
        for j in range(N_HID):
            out_buy += weights[B1_END + 0 * N_HID + j] * h[j]
            out_sell += weights[B1_END + 1 * N_HID + j] * h[j]
            out_flat += weights[B1_END + 2 * N_HID + j] * h[j]

        if out_buy >= out_sell and out_buy >= out_flat:
            action = 1   # BUY
        elif out_sell >= out_buy and out_sell >= out_flat:
            action = 2   # SELL
        else:
            action = 0   # FLATTEN

        # ── Force-close on max_hold ──────────────────────────────
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

        # ── Apply action ─────────────────────────────────────────
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

    # Close any open position at chunk end (no extra spread — exit at mid)
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


# ── Fitness wrapper ────────────────────────────────────────────────────
def fitness_neg(weights, m5_slope, h1_slope, mid_close, pip, spread, max_hold,
                n_chunks, min_dir_ratio):
    """CMA-ES minimizes. Smooth, always-defined objective so search has gradient
    everywhere (no flat plateaus that trigger CMA-ES tolfun stop).

    Components (all in pips/day units, all subtracted from a base):

      base_pps           = total_pnl / total_days      (raw productivity)
      asym_penalty       = (1 - 2·dir_ratio) · 50      0 at 50/50, 50 at all-one-side
      activity_penalty   = max(0, 30 - total_trades) · 2     under-trading penalty
      losing_chunk_pen   = sum(max(0, -chunk_pps)) · 2       per-chunk loss penalty

    If every chunk is profitable AND dir_ratio ≥ min_dir_ratio:
      use min(chunk_pps) instead of total — match the strict scoring.

    Returns -score (CMA-ES minimizes).
    """
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
            m5_slope, h1_slope, mid_close,
            pip, spread, max_hold,
            weights, c_start, c_end,
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
        # idle: monotonic gradient toward "do something"
        return 500.0 - base_pps   # base_pps == 0 here, so flat 500 — fine
    dir_ratio = min(total_long, total_short) / total_trades

    asym_penalty = (1.0 - 2.0 * dir_ratio) * 50.0
    activity_penalty = max(0.0, 30.0 - total_trades) * 2.0
    losing_pen = losing_chunk_loss * 2.0

    # Strict mode: every chunk profitable AND balanced → use min chunk
    all_profitable = all(p > 0 for p in chunk_pps)
    if all_profitable and dir_ratio >= min_dir_ratio:
        score = min(chunk_pps) - asym_penalty
    else:
        score = base_pps - asym_penalty - activity_penalty - losing_pen

    return -score


def passes_hard_gates(weights, m5_slope, h1_slope, mid_close, pip, spread,
                      max_hold, n_chunks, min_dir_ratio):
    """Strict post-search check: identical to the old hard gates."""
    n_bars = len(mid_close)
    total_long = 0
    total_short = 0
    total_trades = 0
    chunk_pps = []
    for ci in range(n_chunks):
        c_start = int(n_bars * ci / n_chunks)
        c_end = int(n_bars * (ci + 1) / n_chunks)
        nt, pnl, nl, ns = simulate_chunk(
            m5_slope, h1_slope, mid_close, pip, spread, max_hold,
            weights, c_start, c_end)
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


# ── Multiprocessing worker ─────────────────────────────────────────────
_WORKER_DATA = {}


def _worker_init(m5, h1, mid, pip, spread, max_hold, n_chunks, min_dir):
    _WORKER_DATA["m5"] = m5
    _WORKER_DATA["h1"] = h1
    _WORKER_DATA["mid"] = mid
    _WORKER_DATA["pip"] = pip
    _WORKER_DATA["spread"] = spread
    _WORKER_DATA["max_hold"] = max_hold
    _WORKER_DATA["n_chunks"] = n_chunks
    _WORKER_DATA["min_dir"] = min_dir


def _worker_fit(vec):
    return fitness_neg(
        vec,
        _WORKER_DATA["m5"], _WORKER_DATA["h1"], _WORKER_DATA["mid"],
        _WORKER_DATA["pip"], _WORKER_DATA["spread"], _WORKER_DATA["max_hold"],
        _WORKER_DATA["n_chunks"], _WORKER_DATA["min_dir"],
    )


# ── OOS evaluator ──────────────────────────────────────────────────────
def eval_oos(weights, m5, h1, mid, pip, spread, max_hold):
    nt, pnl, nl, ns = simulate_chunk(
        m5, h1, mid, pip, spread, max_hold, weights, 0, len(mid))
    n_days = len(mid) / 288.0
    return {
        "n_trades": int(nt),
        "total_pnl": round(float(pnl), 1),
        "pips_per_day": round(float(pnl) / max(n_days, 1), 2),
        "n_long": int(nl),
        "n_short": int(ns),
        "dir_ratio": round(min(nl, ns) / max(nt, 1), 3),
    }


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CMA-5in fixed-topology trainer")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=24,
                        help="CMA-ES popsize (default 24, ~1.5× the recommended 16)")
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--max-hold", type=int, default=200)
    parser.add_argument("--n-chunks", type=int, default=3)
    parser.add_argument("--min-dir-ratio", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label", type=str, default="cma5in")
    args = parser.parse_args()

    np.random.seed(args.seed)

    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]

    print(f"{'='*65}")
    print(f"  CMA-5in Per-Pair: {pair}")
    print(f"  Topology: {N_IN}→{N_HID}→{N_OUT}  ({N_PARAMS} params, "
          f"75 weights + {N_HID} act genes)")
    print(f"  Inputs: m5_slope, h1_slope, upnl, mae, mfe")
    print(f"  Hidden act bank ({N_ACTS}): {', '.join(ACT_NAMES)}")
    print(f"  Output: linear, argmax → BUY/SELL/FLATTEN")
    print(f"  Seed: {args.seed} | popsize {args.popsize} | sigma0 {args.sigma0}")
    print(f"  Workers: {args.workers} | Gens: {args.gens}")
    print(f"  WF chunks: {args.n_chunks} | min_dir {args.min_dir_ratio}")
    print(f"  Max hold: {args.max_hold} M5 bars")
    print(f"{'='*65}")

    # ── Load data ──────────────────────────────────────────────
    path = DATA_DIR / f"{pair}_unified.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    df = pd.read_parquet(path, engine="pyarrow")
    mid = df["mid_close"].values.astype(np.float64)
    h1_slope_full = df["h1_slope"].values.astype(np.float64)
    m5_slope_full = compute_m5_slope(mid, lookback=12)
    del df
    gc.collect()

    n = len(mid)
    split = int(n * 0.7)
    mid_is = mid[:split]
    h1_is = h1_slope_full[:split]
    m5_is = m5_slope_full[:split]
    mid_oos = mid[split:]
    h1_oos = h1_slope_full[split:]
    m5_oos = m5_slope_full[split:]

    print(f"\nData: {n:,} M5 bars | IS: {split:,} | OOS: {n - split:,}")
    print(f"  m5_slope range:  [{m5_is.min():+.3f}, {m5_is.max():+.3f}]  std={m5_is.std():.3f}")
    print(f"  h1_slope range:  [{h1_is.min():+.3f}, {h1_is.max():+.3f}]  std={h1_is.std():.3f}")
    print(f"  Price range:     [{mid_is.min():.5f}, {mid_is.max():.5f}]")

    # ── JIT warmup (one tiny call so compile cost isn't in worker init) ──
    print("\nJIT warming up...")
    warm = np.zeros(N_PARAMS)
    simulate_chunk(m5_is[:200], h1_is[:200], mid_is[:200], pip, spread, 50,
                   warm, 0, 200)
    print("  warm.")

    # ── CMA-ES ─────────────────────────────────────────────────
    # Weights: small Gaussian. Activation genes: uniform in [0,1) so the
    # initial population samples all 9 wavelets roughly uniformly per slot.
    x0 = np.random.randn(N_PARAMS) * 0.3
    x0[B2_END:ACT_END] = np.random.uniform(0.0, 1.0, N_HID)
    opts = {
        "popsize": args.popsize,
        "seed": args.seed,
        "verbose": -9,
        # Loose stop conditions: don't bail on flat plateaus, let maxiter govern.
        "tolx": 1e-9,
        "tolfun": 1e-3,
        "tolfunhist": 1e-3,
        "tolflatfitness": 50,
        "maxiter": args.gens,
    }
    es = cma.CMAEvolutionStrategy(x0, args.sigma0, opts)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.label}_{pair}_s{args.seed}"
    ckpt_dir = RESULTS_DIR / f"{tag}_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_fit = 1e18           # raw CMA-ES objective (lower is better)
    best_vec = None
    best_valid_pps = None     # min-chunk pips/day, only set when hard gates pass
    best_valid_vec = None
    t0 = time.time()

    init_args = (m5_is, h1_is, mid_is, pip, spread, args.max_hold,
                 args.n_chunks, args.min_dir_ratio)

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

            # Check this gen's best against the strict hard-gate criterion
            ok, min_pps = passes_hard_gates(
                best_vec, m5_is, h1_is, mid_is, pip, spread,
                args.max_hold, args.n_chunks, args.min_dir_ratio)
            if ok and (best_valid_pps is None or min_pps > best_valid_pps):
                best_valid_pps = min_pps
                best_valid_vec = np.array(best_vec)
                with open(ckpt_dir / f"gen_{gen:03d}_valid.pkl", "wb") as f:
                    pickle.dump({
                        "weights": best_valid_vec,
                        "min_chunk_pps": best_valid_pps,
                        "generation": gen,
                        "n_in": N_IN, "n_hid": N_HID, "n_out": N_OUT,
                        "n_params": N_PARAMS,
                    }, f)

            if gen % 5 == 0:
                valid_str = (f"{best_valid_pps:+.2f}p/d"
                             if best_valid_pps is not None else "—")
                print(f"  Gen {gen:>3}: raw_fit={best_fit:>12.2f}  "
                      f"valid_min_chunk={valid_str}  "
                      f"σ={es.sigma:.4f}  elapsed={time.time()-t0:.0f}s")
            gen += 1
            if gen >= args.gens:
                break
    finally:
        pool.shutdown(wait=True)

    elapsed = time.time() - t0

    if best_vec is None:
        print(f"\nCMA-ES produced no candidates in {gen} gens — should not happen.")
        sys.exit(2)

    # Prefer the strict-gate winner if any; otherwise fall back to raw best.
    final_vec = best_valid_vec if best_valid_vec is not None else best_vec
    passed_gates = best_valid_vec is not None

    # ── OOS evaluation ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Training complete: {gen} gens, {elapsed:.0f}s")
    print(f"  Raw best fitness:    {best_fit:.4f}")
    if passed_gates:
        print(f"  Hard-gate winner:    min chunk = {best_valid_pps:+.2f} pips/day")
    else:
        print(f"  Hard-gate winner:    NONE — no candidate satisfied all WF gates")
    print(f"{'='*65}")

    is_full = eval_oos(final_vec, m5_is, h1_is, mid_is, pip, spread, args.max_hold)
    oos = eval_oos(final_vec, m5_oos, h1_oos, mid_oos, pip, spread, args.max_hold)
    print(f"\nIS  full:  {is_full}")
    print(f"OOS full:  {oos}")

    chosen_acts = [ACT_NAMES[decode_act(final_vec[B2_END + j])] for j in range(N_HID)]
    print(f"\nHidden activations: {chosen_acts}")

    # ── Save final ─────────────────────────────────────────────
    final_path = RESULTS_DIR / f"{tag}_best.pkl"
    with open(final_path, "wb") as f:
        pickle.dump({
            "weights": final_vec,
            "n_in": N_IN, "n_hid": N_HID, "n_out": N_OUT, "n_params": N_PARAMS,
            "pair": pair, "seed": args.seed,
            "passed_hard_gates": passed_gates,
            "is_min_chunk_pps": best_valid_pps,
            "is_full": is_full,
            "oos": oos,
            "hidden_activations": chosen_acts,
            "args": vars(args),
        }, f)
    print(f"\nSaved: {final_path}")


if __name__ == "__main__":
    main()
