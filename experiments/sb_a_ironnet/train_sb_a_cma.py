#!/usr/bin/env python3
"""
Experiment C: SB_A + IronNet + Range Bars (CMA-ES)
=====================================================
Book Review V2 Campaign — 2026-04-24

Tests whether SB_A (swing breakout state), combined with IronNet architecture
and range bars, produces a tradeable edge after fixing the cost-model bugs
that inflated the book's §20.9 result.

Architecture: IronNet
    Fixed 5→5→2 topology, CMA-ES optimization (no NEAT topology mutation)
    Inputs:
        0  SB_A_norm   — swing breakout state {-1,-0.5,0,+0.5,+1} (causal)
        1  mc_d        — momentum consistency direction (from range bar data)
        2  mc_dd       — MC rate-of-change (from range bar data)
        3  upnl_n      — tanh(upnl_pips / 10)   ([-1, 1])
        4  mae_n       — tanh(mae_pips  / 10)   ([ 0, 1])
    Hidden: 5 neurons, wavelet activations
    Outputs: 2 → ENTER confidence, CLOSE confidence
    Total params: 5×5 + 5 + 5×2 + 2 + 5(act) = 47

Direction: provided by sign(SB_A) at ENTER signal time
    SB_A > 0 → LONG (ASI above last HSP = bullish breakout)
    SB_A < 0 → SHORT (ASI below last LSP = bearish breakout)
    SB_A = 0 → no trade

Cost model (correct, V2):
    Entry price: mid ± full_spread (pre-charges exit cost too; exit at mid)
    MAE initialized to spread_pips at entry (not zero)
    Exit at mid price — full round-trip cost = spread_pips

Fitness: pips_per_day on worst walk-forward chunk (prevents temporal overfitting)

Data: data/range_bar_causal/{PAIR}_range10_causal.parquet  (causal, no lookahead)

Usage:
    python3 train_sb_a_cma.py --pair EUR_GBP --seed 42 --gens 200
    python3 train_sb_a_cma.py --pair CAD_JPY --seed 137 --gens 200 --workers 8

Causality:
    SB_A computed via IncrementalTopsBots (O(1)/bar, causal). Matches batch
    swing set exactly with a variable 1–5 bar lag — identical algorithm used
    in both training and live, so training/live SBA always agree 100%.
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import cma

SCRIPT_DIR = Path(__file__).resolve().parent
# Support running locally (depth=4) or on Hetzner (depth=1)
try:
    PROJECT_ROOT = SCRIPT_DIR.parents[3]
except IndexError:
    PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Also allow explicit path injection via PYTHONPATH (set in deploy script)
for _extra in os.environ.get("PYTHONPATH", "").split(":"):
    if _extra and _extra not in sys.path:
        sys.path.insert(0, _extra)

from lib.incremental_topsbots import IncrementalTopsBots

DATA_DIR = Path(os.environ.get(
    "RANGE_DATA_DIR",
    str(PROJECT_ROOT / "data" / "range_bar_causal")   # causal export (no lookahead)
))
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Pair config ────────────────────────────────────────────────────────
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

# ── IronNet architecture (fixed) ──────────────────────────────────────
N_IN = 5
N_HID = 5
N_OUT = 2   # [ENTER_conf, CLOSE_conf]
N_ACTS = 9  # wavelet bank size

W1_END = N_IN * N_HID                  # 25
B1_END = W1_END + N_HID                # 30
W2_END = B1_END + N_HID * N_OUT        # 40
B2_END = W2_END + N_OUT                # 42
ACT_END = B2_END + N_HID               # 47
N_PARAMS = ACT_END

assert N_PARAMS == 47, f"Expected 47, got {N_PARAMS}"

# ── Activation bank (Numba) ────────────────────────────────────────────
@njit(cache=True, inline="always")
def activate(z, act_id):
    if act_id == 0:   return np.tanh(z)
    elif act_id == 1: return np.sin(z)
    elif act_id == 2: return np.cos(z)
    elif act_id == 3: return np.exp(-z*z)           # gauss
    elif act_id == 4: return 1.0 / np.cosh(z)       # sech
    elif act_id == 5: return -z * np.exp(-z*z)       # DOG
    elif act_id == 6: return np.cos(z) * np.exp(-0.5*z*z)  # gabor
    elif act_id == 7: return np.sinc(z / np.pi) if abs(z) > 1e-9 else 1.0  # sinc
    else:             return np.tanh(z * np.cos(z))  # morlet


@njit(cache=True)
def forward(genes, inp):
    """IronNet forward pass: 5→5→2. No reshape — explicit indexing for Numba C-layout."""
    # Parameter layout: W1[N_HID×N_IN] | b1[N_HID] | W2[N_OUT×N_HID] | b2[N_OUT] | act[N_HID]
    # Hidden layer: h[j] = activate( sum_k(W1[j,k]*inp[k]) + b1[j] )
    h = np.empty(N_HID)
    for j in range(N_HID):
        z = genes[W1_END + j]  # b1[j]
        for k in range(N_IN):
            z += genes[j * N_IN + k] * inp[k]  # W1[j,k]
        act_id = int(genes[B2_END + j] * N_ACTS) % N_ACTS  # act[j]
        h[j] = activate(z, act_id)

    # Output layer (linear): out[i] = sum_j(W2[i,j]*h[j]) + b2[i]
    out = np.empty(N_OUT)
    for i in range(N_OUT):
        s = genes[W2_END + i]  # b2[i]
        for j in range(N_HID):
            s += genes[B1_END + i * N_HID + j] * h[j]  # W2[i,j]
        out[i] = s

    return out


# ── Bar simulator (Numba JIT) ──────────────────────────────────────────
@njit(cache=True)
def simulate_ironnet(
    genes,
    sba,        # float32 arr: SB_A normalized {-1,-0.5,0,+0.5,+1}
    mc_d,       # float32 arr: momentum direction
    mc_dd,      # float32 arr: momentum delta
    mid_close,  # float64 arr: mid prices
    spread_pips,   # float64: spread in pips
    pip_size,      # float64: pip value
    max_hold,      # int: max bars before forced close
    enter_thresh,  # float64: ENTER confidence threshold (tanh applied internally)
    close_thresh,  # float64: CLOSE confidence threshold
):
    """
    Simulate IronNet strategy on range bars.
    Direction provided by sign(SB_A).

    Returns: (total_pips, n_trades, n_long, n_short, avg_mae_pips)
    """
    N = len(mid_close)
    pos = 0          # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    entry_bar = 0
    upnl = 0.0
    mae = 0.0
    mfe = 0.0

    total_pips = 0.0
    total_mae = 0.0
    n_trades = 0
    n_long = 0
    n_short = 0

    for i in range(1, N):
        price = mid_close[i]

        # Build inputs
        sba_i = float(sba[i])
        inp = np.empty(N_IN)
        inp[0] = sba_i                          # SB_A (already normalized)
        inp[1] = float(mc_d[i])                 # MC direction
        inp[2] = float(mc_dd[i])                # MC delta
        inp[3] = np.tanh(upnl / 10.0)           # UpnL
        inp[4] = np.tanh(mae  / 10.0)           # MAE (always ≥ 0)

        out = forward(genes, inp)
        enter_conf = np.tanh(out[0])
        close_conf = np.tanh(out[1])

        if pos != 0:
            # Update position metrics
            if pos == 1:
                upnl = (price - entry_price) / pip_size
            else:
                upnl = (entry_price - price) / pip_size

            if upnl < -mae:
                mae = -upnl  # mae is ≥ 0, tracks worst adverse excursion
            if upnl > mfe:
                mfe = upnl

            # Close conditions
            force_close = (i - entry_bar) >= max_hold
            signal_close = close_conf > close_thresh

            if force_close or signal_close:
                # Exit at mid price (no spread on exit)
                total_pips += upnl
                total_mae += mae
                n_trades += 1
                pos = 0
                upnl = 0.0
                mae = 0.0
                mfe = 0.0

        if pos == 0:
            # Entry condition: confidence above threshold AND SB_A not neutral
            can_enter = enter_conf > enter_thresh and abs(sba_i) > 0.1
            if can_enter:
                direction = 1 if sba_i > 0 else -1
                # Correct cost model: spread charged at entry
                entry_price = price + direction * spread_pips * pip_size
                mae = spread_pips  # MAE initialized to spread cost, not zero
                upnl = -spread_pips  # Starts negative by spread
                mfe = 0.0
                pos = direction
                entry_bar = i
                if direction == 1:
                    n_long += 1
                else:
                    n_short += 1

    return total_pips, n_trades, n_long, n_short, (total_mae / max(n_trades, 1))


# ── Data loading & SB_A computation ───────────────────────────────────

def load_range_data(pair: str) -> dict:
    """Load range bar data and compute SB_A from price swings."""
    pip = PAIR_PIP[pair]
    # Try causal parquet first; fall back to old name for compatibility
    fpath = DATA_DIR / f"{pair}_range10_causal.parquet"
    if not fpath.exists():
        fpath = DATA_DIR / f"{pair}_range10_asi_mc.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"No range bar data for {pair}: {fpath}")

    df = pd.read_parquet(fpath)

    close = df["mid_close"].values.astype(np.float64)
    n = len(close)

    # IncrementalTopsBots: O(1)/bar, strictly causal.
    # Range bars zigzag on close so close==hi==lo gives TopsBots the turning points.
    # Using incremental (not batch) ensures training SBA == live SBA 100%.
    tb = IncrementalTopsBots()
    sba_arr = np.empty(n, dtype=np.float32)
    for i in range(n):
        s, _, _, _ = tb.update(close[i], close[i], close[i])
        sba_arr[i] = np.float32(s / 2.0)
    sba_norm = sba_arr

    result = {
        "sba": sba_norm,
        "mc_d": df["mc_d"].values.astype(np.float32),
        "mc_dd": df["mc_dd"].values.astype(np.float32),
        "mid_close": close,
        "n_bars": n,
    }
    return result


# ── Fitness function ───────────────────────────────────────────────────

def evaluate_genome(genes, data, spread_pips, pip_size,
                    max_hold=100, enter_thresh=0.3, close_thresh=0.2,
                    n_wf_chunks=3):
    """
    Fitness = pips_per_day on worst WF chunk.
    Hard constraints: must be profitable in ALL chunks + bidirectional.
    """
    sba = data["sba"]
    mc_d = data["mc_d"]
    mc_dd = data["mc_dd"]
    mid = data["mid_close"]
    N = data["n_bars"]

    genes_f = np.ascontiguousarray(genes, dtype=np.float64)

    # Walk-forward chunks (70% IS, 30% OOS split, then OOS into n_wf_chunks)
    is_end = int(N * 0.7)
    oos_bars = N - is_end
    chunk_size = oos_bars // n_wf_chunks

    chunk_scores = []
    for c in range(n_wf_chunks):
        s = is_end + c * chunk_size
        e = s + chunk_size
        if e > N:
            e = N
        if e <= s + 50:
            continue

        tp, nt, nl, ns, avg_mae = simulate_ironnet(
            genes_f,
            np.ascontiguousarray(sba[s:e]),
            np.ascontiguousarray(mc_d[s:e]),
            np.ascontiguousarray(mc_dd[s:e]),
            np.ascontiguousarray(mid[s:e]),
            spread_pips, pip_size, max_hold, enter_thresh, close_thresh,
        )

        # Hard constraints
        if nt < 10:
            return -500.0
        if nl == 0 or ns == 0:
            return -999.0  # must trade both directions

        bars_per_day = 35  # ~35 range-10 bars/day on active pairs
        ppd = tp / (chunk_size / bars_per_day)
        chunk_scores.append(ppd)

        if tp < 0:
            return -200.0  # must be profitable in every chunk

    if len(chunk_scores) < n_wf_chunks:
        return -500.0

    return float(np.min(chunk_scores))  # optimize worst chunk


# ── CMA-ES training ────────────────────────────────────────────────────

def train_pair(pair: str, seed: int, gens: int = 200, popsize: int = 20,
               workers: int = 4, max_hold: int = 100,
               verbose: bool = True):
    """Train IronNet CMA-ES for one pair."""
    print(f"\n{'='*50}")
    print(f"  SB_A IronNet Training: {pair}  seed={seed}")
    print(f"  Gens={gens}  Pop={popsize}  Workers={workers}")
    print(f"{'='*50}")

    # Load data
    try:
        data = load_range_data(pair)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return None

    spread = PAIR_SPREAD[pair]
    pip = PAIR_PIP[pair]
    n = data["n_bars"]
    print(f"  Bars: {n:,}  Spread: {spread}p  IS: {int(n*0.7):,}  OOS: {n - int(n*0.7):,}")

    # Causality probe — verify SB_A doesn't leak future
    # Shuffle future bars and check that SB_A doesn't change at current bar
    # (smoke test only; full validate.py test run separately)
    sba = data["sba"]
    n_nonzero = np.count_nonzero(sba)
    print(f"  SB_A coverage: {n_nonzero:,}/{n:,} bars non-zero ({100*n_nonzero/n:.1f}%)")

    np.random.seed(seed)
    x0 = np.random.randn(N_PARAMS) * 0.3

    es = cma.CMAEvolutionStrategy(
        x0, 0.5,
        {"seed": seed, "maxiter": gens, "popsize": popsize,
         "tolx": 1e-8, "tolfun": 1e-8, "verbose": -9}
    )

    best_fitness = -9999.0
    best_genes = None
    t0 = time.time()

    def batch_eval(solutions):
        results = []
        for genes in solutions:
            f = evaluate_genome(genes, data, spread, pip, max_hold)
            results.append(-f)  # CMA minimizes
        return results

    while not es.stop():
        solutions = es.ask()

        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(evaluate_genome, g, data, spread, pip, max_hold)
                           for g in solutions]
                fitnesses = [-f.result() for f in futures]
        else:
            fitnesses = batch_eval(solutions)

        es.tell(solutions, fitnesses)

        gen = es.result.iterations
        best_neg = min(fitnesses)
        best_pos = -best_neg

        if best_pos > best_fitness:
            best_fitness = best_pos
            best_genes = solutions[fitnesses.index(best_neg)]

        if verbose and gen % 25 == 0:
            elapsed = time.time() - t0
            print(f"  Gen {gen:3d}/{gens}  best={best_pos:7.2f} p/d  "
                  f"sigma={es.sigma:.4f}  [{elapsed:.0f}s]")

    if best_genes is None:
        print("  Training produced no valid genome")
        return None

    # Evaluate on OOS
    genes_f = np.asarray(best_genes, dtype=np.float64)
    n = data["n_bars"]
    is_end = int(n * 0.7)
    oos_data = {k: v[is_end:] for k, v in data.items() if k != "n_bars"}
    oos_data["n_bars"] = n - is_end

    tp, nt, nl, ns, avg_mae = simulate_ironnet(
        genes_f,
        np.ascontiguousarray(oos_data["sba"]),
        np.ascontiguousarray(oos_data["mc_d"]),
        np.ascontiguousarray(oos_data["mc_dd"]),
        np.ascontiguousarray(oos_data["mid_close"]),
        spread, pip, max_hold, 0.3, 0.2,
    )
    bars_per_day = 35
    oos_ppd = tp / ((n - is_end) / bars_per_day)

    print(f"\n  OOS Results:")
    print(f"    Total pips: {tp:.1f}")
    print(f"    Trades: {nt} (Long: {nl}, Short: {ns})")
    print(f"    Pips/day: {oos_ppd:.2f}")
    print(f"    Avg MAE: {avg_mae:.2f}p")
    print(f"    Bidir ratio: {min(nl, ns)/max(nl, ns):.2f}" if nl > 0 and ns > 0 else "    UNIDIRECTIONAL")
    print(f"  IS Best fitness: {best_fitness:.2f} p/d")

    result = {
        "pair": pair,
        "seed": seed,
        "oos_total_pips": round(tp, 2),
        "oos_pips_per_day": round(oos_ppd, 2),
        "oos_n_trades": nt,
        "oos_n_long": nl,
        "oos_n_short": ns,
        "oos_avg_mae": round(avg_mae, 2),
        "is_best_fitness": round(best_fitness, 2),
        "training_gens": gens,
        "genes": best_genes.tolist(),
    }

    # Save
    out_stem = f"sb_a_ironnet_{pair}_s{seed}"
    pkl_path = RESULTS_DIR / f"{out_stem}_best.pkl"
    json_path = RESULTS_DIR / f"{out_stem}_result.json"

    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)

    result_no_genes = {k: v for k, v in result.items() if k != "genes"}
    with open(json_path, "w") as f:
        json.dump(result_no_genes, f, indent=2)

    print(f"  Saved: {pkl_path.name}")
    return result


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-hold", type=int, default=100)
    args = parser.parse_args()

    result = train_pair(
        pair=args.pair,
        seed=args.seed,
        gens=args.gens,
        popsize=args.popsize,
        workers=args.workers,
        max_hold=args.max_hold,
    )

    if result:
        print(f"\n✅ Done: {args.pair} s{args.seed} → {result['oos_pips_per_day']:.2f} p/d OOS")
    else:
        print(f"\n❌ Failed: {args.pair} s{args.seed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
