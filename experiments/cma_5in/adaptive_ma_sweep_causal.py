#!/usr/bin/env python3
"""Causal version of adaptive_ma_sweep.py.

Uses a causal multi-TF MC consensus (no lookahead) and tests the same AMA
variants as the original sweep. This is the "fixed data" retest.

If ANY variant produces positive OOS, that's the direction to retrain.
If all fail, the lookahead was the whole edge and we need a different approach.
"""
import argparse, gc, json, pickle, sys, time
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.asi_indicator import compute_asi, sma_jit
from lib.incremental_features import TFState, MC_SIGN_LAGS
import math

TF_BARS = np.array([1, 2, 6, 12, 24, 60, 120, 360, 720], dtype=np.int64)
TF_SEC = [5, 10, 30, 60, 120, 300, 600, 1800, 3600]
TF_WEIGHTS = np.array([math.log2(max(s / 5, 1)) + 1 for s in TF_SEC], dtype=np.float64)
N_TFS = len(TF_BARS)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

PAIR_PIP = {"EUR_JPY":0.01,"USD_JPY":0.01,"GBP_JPY":0.01,"AUD_JPY":0.01,
            "CAD_JPY":0.01,"CHF_JPY":0.01,"NZD_JPY":0.01,
            "EUR_USD":0.0001,"GBP_USD":0.0001,"AUD_USD":0.0001,
            "NZD_USD":0.0001,"EUR_GBP":0.0001}
PAIR_SPREAD = {"EUR_JPY":2.3,"USD_JPY":1.7,"GBP_JPY":3.3,"AUD_JPY":2.1,
               "CAD_JPY":2.3,"CHF_JPY":3.5,"NZD_JPY":2.7,
               "EUR_USD":1.6,"GBP_USD":1.9,"AUD_USD":1.3,
               "NZD_USD":1.5,"EUR_GBP":1.4}


# ── AMA variants (operate on arbitrary array) ─────────────────────
@njit(cache=True)
def kama(arr, n, er_period=10, fast_sc=2, slow_sc=30):
    out = np.zeros(n); out[0] = arr[0]
    fa = 2.0 / (fast_sc + 1.0); sa = 2.0 / (slow_sc + 1.0)
    for i in range(1, n):
        if i >= er_period:
            direction = abs(arr[i] - arr[i - er_period])
            volatility = 0.0
            for j in range(1, er_period + 1):
                volatility += abs(arr[i - j + 1] - arr[i - j])
            er = direction / volatility if volatility > 1e-10 else 0.0
            sc = (er * (fa - sa) + sa) ** 2
        else:
            sc = fa
        out[i] = out[i - 1] + sc * (arr[i] - out[i - 1])
    return out


@njit(cache=True)
def vidya(arr, n, cmo_period=9, alpha=0.2):
    out = np.zeros(n); out[0] = arr[0]
    for i in range(1, n):
        if i >= cmo_period:
            up = 0.0; dn = 0.0
            for j in range(cmo_period):
                d = arr[i-j] - arr[i-j-1]
                if d > 0: up += d
                else: dn += abs(d)
            tot = up + dn
            cmo = abs(up-dn)/tot if tot > 1e-10 else 0.0
            sc = alpha * cmo
        else:
            sc = alpha
        out[i] = out[i-1] + sc * (arr[i] - out[i-1])
    return out


@njit(cache=True)
def kalman_ma(arr, n, process_noise=0.01, measurement_noise=1.0):
    out = np.zeros(n)
    x = arr[0]; p = 1.0; q = process_noise; r = measurement_noise
    for i in range(n):
        pp = p + q; k = pp / (pp + r)
        x = x + k * (arr[i] - x)
        p = (1.0 - k) * pp
        out[i] = x
    return out


@njit(cache=True)
def wilder_rma(arr, n, period=5):
    out = np.zeros(n); out[0] = arr[0]
    a = 1.0/period
    for i in range(1, n):
        out[i] = a * arr[i] + (1.0 - a) * out[i-1]
    return out


@njit(cache=True)
def ema(arr, n, period=3):
    out = np.zeros(n); out[0] = arr[0]
    a = 2.0 / (period + 1.0)
    for i in range(1, n):
        out[i] = a * arr[i] + (1.0 - a) * out[i-1]
    return out


# ── Causal multi-TF MC consensus ──────────────────────────────────
def causal_multi_tf_mc(smoothed_asi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run causal multi-TF MC on smoothed ASI series. No lookahead.

    Each TF samples every bp bars, maintains its own EMA3/EMA5 state,
    caches last mc_d/mc_dd between samples. Weighted avg across TFs,
    normalized by sum of weights of warmed-up TFs.
    """
    n = len(smoothed_asi)
    mc_d = np.zeros(n)
    mc_dd = np.zeros(n)

    # Initialize TF states
    tfs = [TFState(bp=int(TF_BARS[i]), weight=float(TF_WEIGHTS[i]))
           for i in range(N_TFS)]

    for i in range(n):
        val = smoothed_asi[i]
        mc_d_sum = 0.0
        mc_dd_sum = 0.0
        tw = 0.0
        for tf in tfs:
            tf.counter += 1
            if tf.counter >= tf.bp:
                tf.counter = 0
                md, mdd = tf.step(val)
                tf.last_mc_d = md
                tf.last_mc_dd = mdd
            if tf.n_samples >= MC_SIGN_LAGS + 5:
                mc_d_sum += tf.weight * tf.last_mc_d
                mc_dd_sum += tf.weight * tf.last_mc_dd
                tw += tf.weight
        if tw > 0:
            mc_d[i] = mc_d_sum / tw
            mc_dd[i] = mc_dd_sum / tw
    return mc_d, mc_dd


# ── ER_norm (causal, same as builder) ─────────────────────────────
@njit(cache=True)
def er_norm_causal(closes, window=60):
    n = len(closes)
    out = np.zeros(n)
    hp = np.pi / 2.0
    for i in range(window, n):
        net = abs(closes[i] - closes[i - window])
        path = 0.0
        for j in range(i - window + 1, i + 1):
            path += abs(closes[j] - closes[j - 1])
        if path > 0.0:
            out[i] = np.arctan((net / path) / 0.3) / hp
    return out


# ── MACD (causal, proper EMAs, matches training) ─────────────────
def macd_hist_causal(close, high, low):
    """True MACD histogram / Wilder ATR14. Same as compute_macd_hist in training."""
    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    hist = ema_fast - ema_slow - pd.Series(ema_fast - ema_slow).ewm(span=9, adjust=False).mean().values
    tr = np.maximum(np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1)),
                    np.abs(close - np.roll(close, 1)))
    tr[0] = 0
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    return hist / np.where(atr > 0, atr, 1e-10)


# ── Variant dispatch ─────────────────────────────────────────────
def compute_features_causal(o, h, l, c, variant):
    """Return (mc_d, mc_dd, er_norm, macd_hist) arrays, all causal."""
    n = len(c)
    asi_arr = compute_asi(o, h, l, c, n)

    # Pick smoother
    if variant == "sma5":
        smooth = sma_jit(asi_arr, 5, n)
    elif variant == "ema3":
        smooth = ema(asi_arr, n, period=3)
    elif variant == "ema5":
        smooth = ema(asi_arr, n, period=5)
    elif variant == "rma5":
        smooth = wilder_rma(asi_arr, n, period=5)
    elif variant == "kama_10_2_30":
        smooth = kama(asi_arr, n, 10, 2, 30)
    elif variant == "kama_5_2_15":
        smooth = kama(asi_arr, n, 5, 2, 15)
    elif variant == "vidya_9":
        smooth = vidya(asi_arr, n, 9, 0.2)
    elif variant == "vidya_5":
        smooth = vidya(asi_arr, n, 5, 0.3)
    elif variant == "kalman_01":
        smooth = kalman_ma(asi_arr, n, 0.01, 1.0)
    elif variant == "kalman_1":
        smooth = kalman_ma(asi_arr, n, 0.1, 1.0)
    elif variant == "kalman_10":
        smooth = kalman_ma(asi_arr, n, 1.0, 1.0)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Causal multi-TF MC consensus
    mc_d, mc_dd = causal_multi_tf_mc(smooth)
    er = er_norm_causal(c, 60)
    macd = macd_hist_causal(c.astype(np.float64), h.astype(np.float64), l.astype(np.float64))
    return mc_d, mc_dd, er, macd


# ── Training wrapper ─────────────────────────────────────────────
def train_variant(pair, seed, variant, gens=200, popsize=24, workers=4):
    from research.experiments.cma_5in.train_cma_v2 import (
        simulate_chunk, passes_hard_gates, eval_oos,
        _worker_init, _worker_fit, N_HID, N_OUT, N_POSITION_STATE, ACT_NAMES,
    )
    from concurrent.futures import ProcessPoolExecutor
    import cma

    pip = PAIR_PIP[pair]; spread = PAIR_SPREAD[pair]
    df = pd.read_parquet(PROJECT_ROOT / f"data/m5_ohlc/{pair}_M5.parquet")
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)

    mc_d, mc_dd, er, macd = compute_features_causal(o, h, l, c, variant)
    macd_n = np.clip(macd / 2.0, -1, 1)

    market = np.stack([mc_d, mc_dd, er, macd_n], axis=0)
    mid = c.copy()
    n = len(mid)
    del o, h, l, c, df, mc_d, mc_dd, er, macd, macd_n
    gc.collect()

    n_in = 4 + N_POSITION_STATE
    fixed_act_id = ACT_NAMES.index("sin")
    n_params = n_in * N_HID + N_HID + N_HID * N_OUT + N_OUT
    split = int(n * 0.7)
    m_is = market[:, :split].copy(); mid_is = mid[:split].copy()
    m_oos = market[:, split:].copy(); mid_oos = mid[split:].copy()
    del market, mid
    gc.collect()

    warm = np.zeros(n_params)
    simulate_chunk(m_is[:, :200], mid_is[:200], pip, spread, 50, warm, n_in, fixed_act_id, 0, 200)

    pool = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
        initargs=(m_is, mid_is, pip, spread, 200, n_in, fixed_act_id, 3, 0.15, 288.0))

    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    x0 = rng.randn(n_params) * 0.3
    es = cma.CMAEvolutionStrategy(x0, 0.5, {'popsize': popsize, 'seed': seed, 'verbose': -9, 'maxiter': gens})

    best_fit = 1e18; best_vec = None; best_valid_pps = None; best_valid_vec = None
    while not es.stop():
        c_ = es.ask()
        f_ = list(pool.map(_worker_fit, c_))
        es.tell(c_, f_)
        gm = min(f_)
        if gm < best_fit:
            best_fit = gm
            best_vec = np.array(c_[f_.index(gm)])
        ok, mps = passes_hard_gates(best_vec, m_is, mid_is, pip, spread, 200,
                                     n_in, fixed_act_id, 3, 0.15, 288.0)
        if ok and (best_valid_pps is None or mps > best_valid_pps):
            best_valid_pps = mps; best_valid_vec = np.array(best_vec)
    pool.shutdown(wait=False)

    final = best_valid_vec if best_valid_vec is not None else best_vec
    oos = eval_oos(final, m_oos, mid_oos, pip, spread, 200, n_in, fixed_act_id, 288.0)
    is_full = eval_oos(final, m_is, mid_is, pip, spread, 200, n_in, fixed_act_id, 288.0)

    return {
        "variant": variant, "pair": pair, "seed": seed,
        "is": is_full, "oos": oos,
        "hard_gates": best_valid_pps is not None,
        "min_chunk_pps": best_valid_pps,
    }


VARIANTS = ["sma5", "ema3", "ema5", "rma5",
            "kama_10_2_30", "kama_5_2_15",
            "vidya_9", "vidya_5",
            "kalman_01", "kalman_1", "kalman_10"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="CHF_JPY")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--gens", type=int, default=200)
    p.add_argument("--variants", nargs="+", default=None)
    args = p.parse_args()

    variants = args.variants or VARIANTS
    results = []
    print(f"=== CAUSAL AMA sweep: {len(variants)} variants x {len(args.seeds)} seeds ===")
    print(f"Pair: {args.pair} | Gens: {args.gens}")

    for v in variants:
        for s in args.seeds:
            t0 = time.time()
            print(f"\n── {v} (s{s}) ──", flush=True)
            try:
                r = train_variant(args.pair, s, v, gens=args.gens)
                r["elapsed_s"] = round(time.time()-t0, 1)
                results.append(r)
                oos = r["oos"]
                gate = "PASS" if r["hard_gates"] else "FAIL"
                print(f"  OOS: {oos['pips_per_day']:+.2f} p/d  ({oos['n_trades']}T, "
                      f"dir={oos['dir_ratio']:.2f})  {gate}  {r['elapsed_s']:.0f}s")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"variant": v, "seed": s, "error": str(e)})

    print(f"\n=== CAUSAL SUMMARY: {args.pair} ===")
    print(f"{'Variant':>16} {'OOS p/d':>8} {'Trades':>7} {'Dir':>5} {'Gate':>5}")
    for r in sorted(results, key=lambda x: x.get("oos",{}).get("pips_per_day",-999), reverse=True):
        if "error" in r:
            print(f"{r['variant']:>16} ERROR")
            continue
        oos = r["oos"]
        gate = "PASS" if r["hard_gates"] else "FAIL"
        print(f"{r['variant']:>16} {oos['pips_per_day']:>+8.1f} {oos['n_trades']:>7} "
              f"{oos['dir_ratio']:>5.2f} {gate:>5}")

    out = RESULTS_DIR / f"ama_sweep_causal_{args.pair}_s{args.seeds[0]}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
