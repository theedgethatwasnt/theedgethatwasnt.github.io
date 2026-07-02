#!/usr/bin/env python3
"""
CMA-ES quick experiment: pure PnL reward, M5 cadence, 3 inputs.
No AUDDP penalty, no bidirectionality gate. Just mean_pnl × n_trades.

Inputs:
  f1   = arctan( zscore(1-bar return / pip) )          [5-min momentum]
  f10  = arctan( zscore(10-bar return / (10*pip)) )    [50-min momentum]
  upnl = arctan( running_pnl_pips / 10 )               [position state]
Outputs: argmax(BUY=0, SELL=1, FLATTEN=2)

Usage:
  python3 train_cma_quick.py
  python3 train_cma_quick.py --restarts 10 --workers 6
"""
import argparse
import pickle
import time
from pathlib import Path

import cma
import multiprocessing as mp
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR  = Path(__file__).resolve().parent
_dev_s5     = SCRIPT_DIR.parents[2] / "data" / "s5_ohlc"
S5_DIR      = _dev_s5 if _dev_s5.exists() else SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PAIR     = "EUR_JPY"
PIP      = 0.01
SPREAD   = 2.3   # pips
LB1      = 1     # M5: 1-bar = 5 min
LB2      = 10    # M5: 10-bar = 50 min
MAX_HOLD = 48    # M5: 48-bar = 4 hours
POP_SIZE = 1000  # rolling zscore window
IS_FRAC  = 0.70
N_CHUNKS = 3
MIN_TRADES = 10  # per chunk, any direction

# ── Architecture: 3→8→3 ───────────────────────────────────────────────
N_IN   = 3
N_HID  = 8
N_OUT  = 3
N_ACTS = 9
W1_END  = N_IN  * N_HID
B1_END  = W1_END + N_HID
W2_END  = B1_END + N_HID * N_OUT
B2_END  = W2_END + N_OUT
ACT_END = B2_END + N_HID
N_PARAMS = ACT_END   # 67


@njit(cache=True, inline="always")
def _act(z, act_id):
    if   act_id == 0: return np.tanh(z)
    elif act_id == 1: return np.sin(z)
    elif act_id == 2: return np.cos(z)
    elif act_id == 3: return np.exp(-z * z)
    elif act_id == 4: return 1.0 / np.cosh(z)
    elif act_id == 5: return -z * np.exp(-z * z)
    elif act_id == 6: return np.cos(z) * np.exp(-0.5 * z * z)
    elif act_id == 7: return (np.sinc(z / np.pi) if abs(z) > 1e-9 else 1.0)
    else:             return np.tanh(z * np.cos(z))


@njit(cache=True)
def forward(genes, f1, f10, upnl):
    h = np.empty(N_HID)
    for j in range(N_HID):
        z = genes[W1_END + j]
        z += genes[j * N_IN + 0] * f1
        z += genes[j * N_IN + 1] * f10
        z += genes[j * N_IN + 2] * upnl
        act_id = int(abs(genes[B2_END + j]) * N_ACTS) % N_ACTS
        h[j] = _act(z, act_id)
    out = np.empty(N_OUT)
    for i in range(N_OUT):
        s = genes[W2_END + i]
        for j in range(N_HID):
            s += genes[B1_END + i * N_HID + j] * h[j]
        out[i] = s
    return out


@njit(cache=True)
def compute_features(close, lb1, lb2, pop_size):
    n = len(close)
    r1  = np.zeros(n)
    r10 = np.zeros(n)
    for i in range(lb1, n):
        r1[i]  = (close[i] - close[i - lb1]) / PIP
    for i in range(lb2, n):
        r10[i] = (close[i] - close[i - lb2]) / (lb2 * PIP)

    f1  = np.zeros(n)
    f10 = np.zeros(n)
    hp  = np.pi / 2.0
    start = pop_size + lb2

    s1 = 0.0; s1q = 0.0
    s10 = 0.0; s10q = 0.0
    for k in range(start - pop_size, start):
        s1 += r1[k];  s1q  += r1[k]  * r1[k]
        s10 += r10[k]; s10q += r10[k] * r10[k]

    for i in range(start, n):
        if i > start:
            a1 = r1[i-1];  rem1 = r1[i-1-pop_size]
            s1  += a1 - rem1;  s1q  += a1*a1 - rem1*rem1
            a10 = r10[i-1]; rem10 = r10[i-1-pop_size]
            s10 += a10 - rem10; s10q += a10*a10 - rem10*rem10
        m1  = s1  / pop_size; m10 = s10 / pop_size
        v1  = s1q  / pop_size - m1  * m1
        v10 = s10q / pop_size - m10 * m10
        std1  = v1  ** 0.5 if v1  > 1e-20 else 1e-10
        std10 = v10 ** 0.5 if v10 > 1e-20 else 1e-10
        f1[i]  = np.arctan(r1[i]  / std1)  / hp
        f10[i] = np.arctan(r10[i] / std10) / hp
    return f1, f10, start


@njit(cache=True)
def simulate_chunk(genes, f1, f10, close, spread_pips, max_hold,
                   chunk_start, chunk_end):
    """Pure PnL simulator — no AUDDP, no direction tracking."""
    pos         = 0
    entry_price = 0.0
    entry_bar   = 0
    pnl_pips    = 0.0
    score_sum   = 0.0
    n_trades    = 0
    hp          = np.pi / 2.0

    for i in range(chunk_start, chunk_end):
        upnl_i = np.arctan(pnl_pips / 10.0) / hp
        out = forward(genes, f1[i], f10[i], upnl_i)

        if out[0] >= out[1] and out[0] >= out[2]:
            action = 0
        elif out[1] > out[0] and out[1] >= out[2]:
            action = 1
        else:
            action = 2

        if pos != 0:
            price = close[i]
            pnl_pips = (price - entry_price) / PIP if pos == 1 else (entry_price - price) / PIP
            force_close  = (i - entry_bar) >= max_hold
            signal_close = (pos == 1 and action == 1) or \
                           (pos == -1 and action == 0) or \
                           (action == 2)
            if force_close or signal_close:
                score_sum += pnl_pips
                n_trades  += 1
                pos = 0
                pnl_pips = 0.0

        if pos == 0 and action != 2:
            direction   = 1 if action == 0 else -1
            entry_price = close[i] + direction * (spread_pips * 0.5) * PIP
            entry_bar   = i
            pnl_pips    = -spread_pips
            pos         = direction

    if n_trades == 0:
        return 0, -10.0
    return n_trades, score_sum / n_trades


def fitness_fn(genes_arr, data, max_hold):
    genes = np.asarray(genes_arr, dtype=np.float64)
    chunk_fits = []
    n_is = data["n_is"]
    for ci in range(N_CHUNKS):
        cs = int(n_is * ci       / N_CHUNKS)
        ce = int(n_is * (ci + 1) / N_CHUNKS)
        nt, mean_pnl = simulate_chunk(
            genes, data["f1_is"], data["f10_is"], data["close_is"],
            SPREAD, max_hold, cs, ce)
        if nt < MIN_TRADES:
            chunk_fits.append(-10.0 + nt * 0.1)   # gradient toward trading
        else:
            chunk_fits.append(mean_pnl * nt)
    return -min(chunk_fits)   # CMA minimizes


def eval_oos(genes_arr, data, max_hold):
    genes = np.asarray(genes_arr, dtype=np.float64)
    n = len(data["close_oos"])
    nt, mean_pnl = simulate_chunk(
        genes, data["f1_oos"], data["f10_oos"], data["close_oos"],
        SPREAD, max_hold, 0, n)
    return nt, mean_pnl, mean_pnl * nt if nt else -10.0


def _worker(args):
    genes_arr, data, max_hold = args
    return fitness_fn(genes_arr, data, max_hold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restarts",  type=int,   default=5)
    parser.add_argument("--max-iter",  type=int,   default=300)
    parser.add_argument("--workers",   type=int,   default=6)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--sigma0",    type=float, default=0.5)
    args = parser.parse_args()

    print(f"CMA-ES quick: pure PnL, M5 cadence, 3 inputs")
    print(f"Arch: {N_IN}→{N_HID}→{N_OUT}  params={N_PARAMS}  wavelet activations")
    print(f"Pair: {PAIR}  LB1={LB1} LB2={LB2} M5 bars  max_hold={MAX_HOLD} bars (4h)")
    print(f"CMA: {args.restarts} restarts × {args.max_iter} iters  σ0={args.sigma0}  workers={args.workers}")

    # ── Load data ──────────────────────────────────────────────────────
    print("\nLoading data...")
    path = S5_DIR / f"{PAIR}_S5_BA.parquet"
    if not path.exists():
        path = S5_DIR / f"{PAIR}_S5.parquet"
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    col = "bid_c" if "bid_c" in df.columns else "close"
    closes_s5 = df[col].values.astype(np.float64)
    closes = closes_s5[::12]   # M5 stride
    n = len(closes)
    n_is  = int(n * IS_FRAC)
    n_oos = n - n_is
    print(f"  {PAIR}: S5={len(closes_s5):,}  M5={n:,}  IS={n_is:,}  OOS={n_oos:,}")

    # ── Features ──────────────────────────────────────────────────────
    print("Computing features + JIT warmup...")
    f1_all, f10_all, feat_start = compute_features(closes, LB1, LB2, POP_SIZE)
    warmup_start = max(feat_start, 0)

    close_is  = closes[:n_is]
    f1_is     = f1_all[:n_is]
    f10_is    = f10_all[:n_is]
    close_oos = closes[n_is:]
    f1_oos    = f1_all[n_is:]
    f10_oos   = f10_all[n_is:]

    data = dict(
        close_is=close_is, f1_is=f1_is, f10_is=f10_is, n_is=n_is,
        close_oos=close_oos, f1_oos=f1_oos, f10_oos=f10_oos,
    )

    # JIT warmup
    dummy = np.zeros(N_PARAMS)
    simulate_chunk(dummy, f1_is, f10_is, close_is, SPREAD, MAX_HOLD, warmup_start, warmup_start+200)
    print("  Done")

    # ── Training ──────────────────────────────────────────────────────
    print("\nTraining...\n")
    best_is  = -1e9
    best_oos = -1e9
    best_genes = None
    rng = np.random.default_rng(args.seed)

    pool = mp.Pool(args.workers,
                   initializer=lambda: None,
                   initargs=()) if args.workers > 1 else None

    opts = cma.CMAOptions()
    opts["maxiter"]        = args.max_iter
    opts["popsize"]        = 16
    opts["verbose"]        = -9
    opts["tolflatfitness"] = 1000
    opts["tolfun"]         = 1e-9
    opts["tolx"]           = 1e-10

    for restart in range(1, args.restarts + 1):
        t0   = time.time()
        seed = int(rng.integers(0, 2**31))
        x0   = rng.standard_normal(N_PARAMS) * 0.3

        es = cma.CMAEvolutionStrategy(x0, args.sigma0, {**opts, "seed": seed})

        while not es.stop():
            candidates = es.ask()
            if pool:
                fitvals = pool.map(_worker, [(c, data, MAX_HOLD) for c in candidates])
            else:
                fitvals = [fitness_fn(c, data, MAX_HOLD) for c in candidates]
            es.tell(candidates, fitvals)

        best = es.result.xbest
        is_fit = -es.result.fbest

        nt_oos, mpnl_oos, oos_fit = eval_oos(best, data, MAX_HOLD)
        elapsed = time.time() - t0

        marker = ""
        if is_fit > best_is:
            best_is = is_fit
            best_genes = best.copy()
            marker = " ★ best IS"
        if oos_fit > best_oos:
            best_oos = oos_fit
            marker += " ★ best OOS"

        print(f"  restart {restart:2d}/{args.restarts}: "
              f"IS={is_fit:+8.2f}  OOS={oos_fit:+8.2f}  "
              f"oos_trades={nt_oos:4d}  oos_mean={mpnl_oos:+.3f}p  "
              f"t={elapsed:.0f}s{marker}")

    if pool:
        pool.close()

    if best_genes is not None:
        tag = f"cma_quick_m5_s{args.seed}"
        pkl = RESULTS_DIR / f"{tag}.pkl"
        with open(pkl, "wb") as f:
            pickle.dump({"genes": best_genes, "is_fit": best_is, "oos_fit": best_oos,
                         "pair": PAIR, "lb1": LB1, "lb2": LB2, "max_hold": MAX_HOLD}, f)
        print(f"\nSaved: {pkl}")
        print(f"Best IS={best_is:+.2f}  Best OOS={best_oos:+.2f}")


if __name__ == "__main__":
    import os
    os.environ["OMP_NUM_THREADS"]     = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"]     = "1"
    main()
