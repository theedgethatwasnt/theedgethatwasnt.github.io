#!/usr/bin/env python3
"""
Sweep adaptive MA variants replacing SMA5 in the ASI-MC pipeline.

Tests:
  1. SMA5 (baseline — current)
  2. KAMA (Kaufman Adaptive MA)
  3. VIDYA (Variable Index Dynamic Average)
  4. Kalman filter MA
  5. Wilder/RMA smoothing
  6. Simple D = ASI - AMA(ASI) (single adaptive MA)
  7. D = AMA_fast(ASI) - AMA_slow(ASI) (two adaptive MAs)

For each variant, computes mc_d_a / mc_dd_a using the standard 5-lag sign
counting (proven to work), then trains CMA-NN V3+macd_hist on CHF_JPY.

Usage:
    python3 adaptive_ma_sweep.py [--pair CHF_JPY] [--seeds 42]
"""
import argparse
import gc
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.asi_indicator import compute_asi, sma_jit, _ema_diff_mc, compute_mc_on_series, TF_BARS_S5, TF_WEIGHTS, N_TFS

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

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

EXTRA_NORMALIZE = {"macd_hist": ("div_clip", 2.0)}


# ══════════════════════════════════════════════════════════════════
# Adaptive MAs — all operate on ASI array, return smoothed series
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def kama(arr, n, er_period=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average."""
    out = np.zeros(n)
    out[0] = arr[0]
    fast_alpha = 2.0 / (fast_sc + 1.0)
    slow_alpha = 2.0 / (slow_sc + 1.0)
    for i in range(1, n):
        if i >= er_period:
            direction = abs(arr[i] - arr[i - er_period])
            volatility = 0.0
            for j in range(1, er_period + 1):
                volatility += abs(arr[i - j + 1] - arr[i - j])
            if volatility > 1e-10:
                er = direction / volatility
            else:
                er = 0.0
            sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        else:
            sc = fast_alpha
        out[i] = out[i - 1] + sc * (arr[i] - out[i - 1])
    return out


@njit(cache=True)
def vidya(arr, n, cmo_period=9, alpha=0.2):
    """Variable Index Dynamic Average (Chande)."""
    out = np.zeros(n)
    out[0] = arr[0]
    for i in range(1, n):
        if i >= cmo_period:
            up_sum = 0.0
            dn_sum = 0.0
            for j in range(cmo_period):
                diff = arr[i - j] - arr[i - j - 1]
                if diff > 0:
                    up_sum += diff
                else:
                    dn_sum += abs(diff)
            total = up_sum + dn_sum
            if total > 1e-10:
                cmo = abs(up_sum - dn_sum) / total
            else:
                cmo = 0.0
            sc = alpha * cmo
        else:
            sc = alpha
        out[i] = out[i - 1] + sc * (arr[i] - out[i - 1])
    return out


@njit(cache=True)
def kalman_ma(arr, n, process_noise=0.01, measurement_noise=1.0):
    """Simple 1D Kalman filter as adaptive MA."""
    out = np.zeros(n)
    x = arr[0]       # state estimate
    p = 1.0           # estimate covariance
    q = process_noise
    r = measurement_noise
    for i in range(n):
        # Predict
        p_pred = p + q
        # Update
        k = p_pred / (p_pred + r)
        x = x + k * (arr[i] - x)
        p = (1.0 - k) * p_pred
        out[i] = x
    return out


@njit(cache=True)
def wilder_rma(arr, n, period=5):
    """Wilder's smoothing (RMA/SMMA)."""
    out = np.zeros(n)
    out[0] = arr[0]
    alpha = 1.0 / period
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


@njit(cache=True)
def ema(arr, n, period=5):
    """Standard EMA."""
    out = np.zeros(n)
    out[0] = arr[0]
    alpha = 2.0 / (period + 1.0)
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


# ══════════════════════════════════════════════════════════════════
# MC computation variants
# ══════════════════════════════════════════════════════════════════

def compute_mc_variant(o, h, l, c, n, variant, params=None):
    """Compute mc_d, mc_dd using the given smoothing variant."""
    asi_arr = compute_asi(o, h, l, c, n)

    if variant == "sma5":
        smooth = sma_jit(asi_arr, 5, n)
    elif variant == "kama_10_2_30":
        smooth = kama(asi_arr, n, er_period=10, fast_sc=2, slow_sc=30)
    elif variant == "kama_5_2_15":
        smooth = kama(asi_arr, n, er_period=5, fast_sc=2, slow_sc=15)
    elif variant == "vidya_9":
        smooth = vidya(asi_arr, n, cmo_period=9, alpha=0.2)
    elif variant == "vidya_5":
        smooth = vidya(asi_arr, n, cmo_period=5, alpha=0.3)
    elif variant == "kalman_01":
        smooth = kalman_ma(asi_arr, n, process_noise=0.01, measurement_noise=1.0)
    elif variant == "kalman_1":
        smooth = kalman_ma(asi_arr, n, process_noise=0.1, measurement_noise=1.0)
    elif variant == "kalman_10":
        smooth = kalman_ma(asi_arr, n, process_noise=1.0, measurement_noise=1.0)
    elif variant == "rma5":
        smooth = wilder_rma(asi_arr, n, period=5)
    elif variant == "ema3":
        smooth = ema(asi_arr, n, period=3)
    elif variant == "ema5":
        smooth = ema(asi_arr, n, period=5)
    # D = ASI - AMA(ASI) variants (single MA, no crossover)
    elif variant == "d_kama":
        smooth_kama = kama(asi_arr, n, er_period=10, fast_sc=2, slow_sc=30)
        # Use (ASI - KAMA) as the "smooth" series, then MC on that
        # But MC expects a price-like series. Instead, directly compute D.
        return _compute_direct_d(asi_arr, smooth_kama, n)
    elif variant == "d_vidya":
        smooth_vidya = vidya(asi_arr, n, cmo_period=9, alpha=0.2)
        return _compute_direct_d(asi_arr, smooth_vidya, n)
    elif variant == "d_kalman":
        smooth_kal = kalman_ma(asi_arr, n, process_noise=0.1, measurement_noise=1.0)
        return _compute_direct_d(asi_arr, smooth_kal, n)
    # D = AMA_fast - AMA_slow variants
    elif variant == "dual_kama":
        fast = kama(asi_arr, n, er_period=5, fast_sc=2, slow_sc=10)
        slow = kama(asi_arr, n, er_period=10, fast_sc=2, slow_sc=30)
        return _compute_direct_d(fast, slow, n)
    elif variant == "dual_vidya":
        fast = vidya(asi_arr, n, cmo_period=5, alpha=0.3)
        slow = vidya(asi_arr, n, cmo_period=14, alpha=0.15)
        return _compute_direct_d(fast, slow, n)
    elif variant == "dual_kalman":
        fast = kalman_ma(asi_arr, n, process_noise=1.0, measurement_noise=1.0)
        slow = kalman_ma(asi_arr, n, process_noise=0.01, measurement_noise=1.0)
        return _compute_direct_d(fast, slow, n)
    # Two-stage: Kalman(ASI) first, then fast/slow crossover on top
    elif variant == "kalman10_then_dual_kalman":
        # Stage 1: Kalman(q=1.0) smooths ASI
        k_smooth = kalman_ma(asi_arr, n, process_noise=1.0, measurement_noise=1.0)
        # Stage 2: two Kalmans (fast/slow) on the smoothed series → D
        fast = kalman_ma(k_smooth, n, process_noise=1.0, measurement_noise=1.0)
        slow = kalman_ma(k_smooth, n, process_noise=0.01, measurement_noise=1.0)
        return _compute_direct_d(fast, slow, n)
    elif variant == "kalman10_then_emadiff":
        # Stage 1: Kalman(q=1.0) smooths ASI → Stage 2: standard EMA3-EMA5
        # This is exactly the winning variant but made explicit
        smooth = kalman_ma(asi_arr, n, process_noise=1.0, measurement_noise=1.0)
        mc_d, mc_dd = compute_mc_on_series(smooth, n, TF_BARS_S5, TF_WEIGHTS, N_TFS)
        return mc_d, mc_dd
    elif variant == "ema3_then_dual_kalman":
        # Stage 1: EMA3 smooths ASI (new winner smoother)
        smooth = ema(asi_arr, n, period=3)
        fast = kalman_ma(smooth, n, process_noise=1.0, measurement_noise=1.0)
        slow = kalman_ma(smooth, n, process_noise=0.01, measurement_noise=1.0)
        return _compute_direct_d(fast, slow, n)
    elif variant == "sma5_then_dual_kalman":
        # Baseline smoother + adaptive crossover
        smooth = sma_jit(asi_arr, 5, n)
        fast = kalman_ma(smooth, n, process_noise=1.0, measurement_noise=1.0)
        slow = kalman_ma(smooth, n, process_noise=0.01, measurement_noise=1.0)
        return _compute_direct_d(fast, slow, n)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Standard: smooth → multi-TF MC(D)/MC(dD) with 5-lag sign counting
    mc_d, mc_dd = compute_mc_on_series(smooth, n, TF_BARS_S5, TF_WEIGHTS, N_TFS)
    return mc_d, mc_dd


@njit(cache=True)
def _compute_direct_d(series_a, series_b, n, n_lags=5):
    """Compute MC(D)/MC(dD) where D = series_a - series_b.
    Uses standard 5-lag sign counting on the difference."""
    d_vals = np.zeros(n)
    for i in range(n):
        d_vals[i] = series_a[i] - series_b[i]

    mc_d = np.zeros(n)
    mc_dd = np.zeros(n)

    for i in range(n_lags + 1, n):
        pos = neg = 0
        for lag in range(n_lags):
            change = d_vals[i - lag] - d_vals[i - lag - 1]
            if change > 0:
                pos += 1
            elif change < 0:
                neg += 1
        mc_d[i] = (pos - neg) / n_lags

    for i in range(n_lags + 2, n):
        pos = neg = 0
        for lag in range(n_lags):
            j = i - lag
            if j >= 3:
                dd_now = d_vals[j] - 2.0 * d_vals[j - 1] + d_vals[j - 2]
                dd_prev = d_vals[j - 1] - 2.0 * d_vals[j - 2] + d_vals[j - 3]
                change = dd_now - dd_prev
                if change > 0:
                    pos += 1
                elif change < 0:
                    neg += 1
        mc_dd[i] = (pos - neg) / n_lags

    return mc_d, mc_dd


# ══════════════════════════════════════════════════════════════════
# Training wrapper
# ══════════════════════════════════════════════════════════════════

def train_variant(pair, seed, variant, gens=200, popsize=24, workers=4):
    """Load data, compute MC with given variant, train CMA-NN, return results."""
    from research.experiments.cma_5in.train_cma_v2 import (
        _compute_er_norm_v3, simulate_chunk, fitness_neg, passes_hard_gates,
        eval_oos, N_HID, N_OUT, N_POSITION_STATE, ACT_NAMES,
    )
    from concurrent.futures import ProcessPoolExecutor
    import cma

    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]

    # Load M5 OHLC
    ohlc_path = PROJECT_ROOT / "data" / "m5_ohlc" / f"{pair}_M5.parquet"
    df_o = pd.read_parquet(ohlc_path, engine="pyarrow")
    o = df_o["open"].values.astype(np.float64)
    h = df_o["high"].values.astype(np.float64)
    l = df_o["low"].values.astype(np.float64)
    c = df_o["close"].values.astype(np.float64)
    n = len(c)
    mid = c.copy()

    # Compute MC with variant
    mc_d, mc_dd = compute_mc_variant(o, h, l, c, n, variant)
    er_norm = _compute_er_norm_v3(c, window=60)

    # Load macd_hist
    uni_path = PROJECT_ROOT / "data" / "unified_indicators" / f"{pair}_unified.parquet"
    df_u = pd.read_parquet(uni_path, columns=["timestamp", "macd_hist"], engine="pyarrow")
    df_m = df_o[["timestamp"]].merge(df_u, on="timestamp", how="left")
    macd = np.clip(df_m["macd_hist"].fillna(0).values.astype(np.float64) / 2.0, -1, 1)

    market = np.stack([mc_d, mc_dd, er_norm, macd], axis=0)
    del df_o, df_u, df_m, o, h, l, c, mc_d, mc_dd, er_norm, macd
    gc.collect()

    n_in = 4 + N_POSITION_STATE  # 7
    n_params = n_in * N_HID + N_HID + N_HID * N_OUT + N_OUT  # 91 for fixed sin
    fixed_act_id = ACT_NAMES.index("sin")
    bars_per_day = 288.0

    split = int(n * 0.7)
    m_is = market[:, :split].copy()
    mid_is = mid[:split].copy()
    m_oos = market[:, split:].copy()
    mid_oos = mid[split:].copy()
    del market, mid
    gc.collect()

    # JIT warmup
    warm = np.zeros(n_params)
    simulate_chunk(m_is[:, :200], mid_is[:200], pip, spread, 50,
                   warm, n_in, fixed_act_id, 0, 200)

    # CMA-ES
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    x0 = rng.randn(n_params) * 0.3

    from research.experiments.cma_5in.train_cma_v2 import _worker_init, _worker_fit
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(m_is, mid_is, pip, spread, 200, n_in, fixed_act_id, 3, 0.15, bars_per_day),
    )

    opts = {"popsize": popsize, "seed": seed, "verbose": -9,
            "tolx": 1e-9, "tolfun": 1e-3, "maxiter": gens}
    es = cma.CMAEvolutionStrategy(x0, 0.5, opts)

    best_fit = 1e18
    best_vec = None
    best_valid_pps = None
    best_valid_vec = None

    while not es.stop():
        candidates = es.ask()
        fitnesses = list(pool.map(_worker_fit, candidates))
        es.tell(candidates, fitnesses)
        gen_min = min(fitnesses)
        if gen_min < best_fit:
            best_fit = gen_min
            best_vec = np.array(candidates[fitnesses.index(gen_min)])
        ok, min_pps = passes_hard_gates(
            best_vec, m_is, mid_is, pip, spread, 200, n_in,
            fixed_act_id, 3, 0.15, bars_per_day)
        if ok and (best_valid_pps is None or min_pps > best_valid_pps):
            best_valid_pps = min_pps
            best_valid_vec = np.array(best_vec)

    pool.shutdown(wait=False)

    final = best_valid_vec if best_valid_vec is not None else best_vec
    oos = eval_oos(final, m_oos, mid_oos, pip, spread, 200, n_in, fixed_act_id, bars_per_day)
    is_full = eval_oos(final, m_is, mid_is, pip, spread, 200, n_in, fixed_act_id, bars_per_day)

    return {
        "variant": variant,
        "pair": pair,
        "seed": seed,
        "is": is_full,
        "oos": oos,
        "hard_gates": best_valid_pps is not None,
        "min_chunk_pps": best_valid_pps,
    }


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

VARIANTS = [
    "sma5",           # baseline
    "ema3",           # fastest standard
    "ema5",           # match SMA5 period
    "rma5",           # Wilder smoothing
    "kama_10_2_30",   # KAMA standard params
    "kama_5_2_15",    # KAMA faster
    "vidya_9",        # VIDYA standard
    "vidya_5",        # VIDYA faster
    "kalman_01",      # Kalman tight (smooth)
    "kalman_1",       # Kalman medium
    "kalman_10",      # Kalman loose (responsive)
    "d_kama",         # D = ASI - KAMA(ASI)
    "d_vidya",        # D = ASI - VIDYA(ASI)
    "d_kalman",       # D = ASI - Kalman(ASI)
    "dual_kama",      # D = KAMA_fast - KAMA_slow
    "dual_vidya",     # D = VIDYA_fast - VIDYA_slow
    "dual_kalman",    # D = Kalman_fast - Kalman_slow
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="CHF_JPY")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--variants", nargs="+", default=None,
                        help=f"Subset of variants to test. Default: all. "
                             f"Available: {VARIANTS}")
    args = parser.parse_args()

    variants = args.variants or VARIANTS
    results = []

    print(f"{'='*65}")
    print(f"  Adaptive MA Sweep: {len(variants)} variants × {len(args.seeds)} seeds")
    print(f"  Pair: {args.pair} | Gens: {args.gens}")
    print(f"{'='*65}")

    for variant in variants:
        for seed in args.seeds:
            t0 = time.time()
            print(f"\n── {variant} (s{seed}) ──", flush=True)
            try:
                r = train_variant(args.pair, seed, variant, gens=args.gens)
                elapsed = time.time() - t0
                r["elapsed_s"] = round(elapsed, 1)
                results.append(r)
                oos = r["oos"]
                gate = "PASS" if r["hard_gates"] else "FAIL"
                print(f"  OOS: {oos['pips_per_day']:+.2f} p/d  "
                      f"({oos['n_trades']}T, dir={oos['dir_ratio']:.2f})  "
                      f"{gate}  {elapsed:.0f}s")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"variant": variant, "seed": seed, "error": str(e)})

    # Summary
    print(f"\n{'='*65}")
    print(f"  SUMMARY: {args.pair}")
    print(f"{'='*65}")
    print(f"{'Variant':>20} {'OOS p/d':>8} {'Trades':>7} {'WR%':>6} {'Dir':>5} {'Gate':>5}")
    print("-" * 55)
    for r in sorted(results, key=lambda x: x.get("oos", {}).get("pips_per_day", -999), reverse=True):
        if "error" in r:
            print(f"{r['variant']:>20} ERROR: {r['error'][:30]}")
            continue
        oos = r["oos"]
        wr = 0
        if oos["n_trades"] > 0:
            wr = (oos["n_long"] if oos["n_long"] < oos["n_short"] else oos["n_short"])
        gate = "PASS" if r["hard_gates"] else "FAIL"
        print(f"{r['variant']:>20} {oos['pips_per_day']:>+8.1f} {oos['n_trades']:>7} "
              f"{oos.get('dir_ratio', 0)*100:>5.1f}% {oos['dir_ratio']:>5.2f} {gate:>5}")

    # Save
    out_path = RESULTS_DIR / f"ama_sweep_{args.pair}_s{args.seeds[0]}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
