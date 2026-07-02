#!/usr/bin/env python3
"""
CMA-ES Z-score strategy — AMDDP reward (PNL - λ × cumulative adverse excursion).

Architecture: fixed 2→8→3 network, wavelet activations per hidden node.
Inputs (causal, S5 bars):
  f1  = arctan( zscore( (bid_c[i] - bid_c[i-12])  / pip       ) )  [1-min return]
  f10 = arctan( zscore( (bid_c[i] - bid_c[i-600]) / (600*pip) ) )  [50-min return]
  Sign-preserving zscore: z = x / std(pop_1000), no mean subtraction.
Outputs: argmax(BUY=0, SELL=1, FLATTEN=2)

Reward per trade:
  auddp_sum = Σ max(0, -pnl_pips) sampled every bar while in trade
  score     = pnl_pips - λ × auddp_sum   (λ=0.01)
  fitness   = mean(scores) × n_trades    (WF bottleneck: min across 3 IS chunks)

Usage:
  python3 train_cma.py --pair EUR_JPY --max-hold 200
  python3 train_cma.py --pair EUR_JPY --max-hold 1200 --restarts 30
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import cma
import multiprocessing as mp
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR   = Path(__file__).resolve().parent
# On Hetzner: script lives at /root/neat/, data at /root/neat/data/
# On dev box: script at …/neat_zscore/, data at …/fx-core/data/s5_ohlc/
_dev_s5 = SCRIPT_DIR.parents[2] / "data" / "s5_ohlc" if len(SCRIPT_DIR.parents) > 2 else None
S5_DIR  = _dev_s5 if (_dev_s5 and _dev_s5.exists()) else SCRIPT_DIR / "data"
M5_DIR  = S5_DIR  # same directory
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

# ── Network architecture ───────────────────────────────────────────────
N_IN   = 2
N_HID  = 8
N_OUT  = 3   # BUY / SELL / FLATTEN
N_ACTS = 9   # wavelet bank size

# Parameter layout: W1[N_HID×N_IN] | b1[N_HID] | W2[N_OUT×N_HID] | b2[N_OUT] | act[N_HID]
W1_END  = N_IN * N_HID          # 16
B1_END  = W1_END + N_HID        # 24
W2_END  = B1_END + N_HID * N_OUT  # 48
B2_END  = W2_END + N_OUT        # 51
ACT_END = B2_END + N_HID        # 59
N_PARAMS = ACT_END

IS_FRAC          = 0.70
N_CHUNKS         = 3
POP_SIZE         = 1000   # rolling zscore window
MIN_TRADES_PER_DIR = 20   # min longs AND min shorts per WF chunk
WARMUP           = POP_SIZE + 600  # covers lb2=600 S5 bars


# ── Activation bank ────────────────────────────────────────────────────
@njit(cache=True, inline="always")
def _act(z, act_id):
    if act_id == 0:   return np.tanh(z)
    elif act_id == 1: return np.sin(z)
    elif act_id == 2: return np.cos(z)
    elif act_id == 3: return np.exp(-z * z)
    elif act_id == 4: return 1.0 / np.cosh(z)
    elif act_id == 5: return -z * np.exp(-z * z)
    elif act_id == 6: return np.cos(z) * np.exp(-0.5 * z * z)
    elif act_id == 7: return (np.sinc(z / np.pi) if abs(z) > 1e-9 else 1.0)
    else:             return np.tanh(z * np.cos(z))


@njit(cache=True)
def forward(genes, f1, f10):
    """2→8→3 forward pass. Returns raw output logits (no softmax)."""
    h = np.empty(N_HID)
    for j in range(N_HID):
        z = genes[W1_END + j]           # b1[j]
        z += genes[j * N_IN + 0] * f1
        z += genes[j * N_IN + 1] * f10
        act_id = int(abs(genes[B2_END + j]) * N_ACTS) % N_ACTS
        h[j] = _act(z, act_id)

    out = np.empty(N_OUT)
    for i in range(N_OUT):
        s = genes[W2_END + i]           # b2[i]
        for j in range(N_HID):
            s += genes[B1_END + i * N_HID + j] * h[j]
        out[i] = s
    return out


# ── Z-score features ───────────────────────────────────────────────────
@njit(cache=True)
def compute_zscore_inputs(close, pip, pop_size=1000, lb1=12, lb2=600):
    """
    f1[i]  = arctan( (close[i]-close[i-lb1]) / pip         / std_pop ) * 2/π
    f10[i] = arctan( (close[i]-close[i-lb2]) / (lb2 * pip) / std_pop ) * 2/π
    Sign-preserving: z = x / σ (no mean subtraction).
    """
    n = len(close)
    f1  = np.zeros(n)
    f10 = np.zeros(n)
    half_pi = np.pi / 2.0

    r1  = np.zeros(n)
    r10 = np.zeros(n)
    for i in range(lb1, n):
        r1[i]  = (close[i] - close[i - lb1]) / pip
    for i in range(lb2, n):
        r10[i] = (close[i] - close[i - lb2]) / (lb2 * pip)

    sum1 = 0.0; sum1_sq = 0.0
    sum10 = 0.0; sum10_sq = 0.0
    start = pop_size + lb2

    for k in range(start - pop_size, start):
        sum1    += r1[k];  sum1_sq    += r1[k]  * r1[k]
        sum10   += r10[k]; sum10_sq   += r10[k] * r10[k]

    for i in range(start, n):
        if i > start:
            add1 = r1[i-1];   rem1 = r1[i-1-pop_size]
            sum1    += add1 - rem1
            sum1_sq += add1*add1 - rem1*rem1
            add10 = r10[i-1]; rem10 = r10[i-1-pop_size]
            sum10    += add10 - rem10
            sum10_sq += add10*add10 - rem10*rem10

        m1  = sum1  / pop_size
        m10 = sum10 / pop_size
        var1  = sum1_sq  / pop_size - m1  * m1
        var10 = sum10_sq / pop_size - m10 * m10
        std1  = var1  ** 0.5 if var1  > 1e-20 else 1e-10
        std10 = var10 ** 0.5 if var10 > 1e-20 else 1e-10

        z1  = r1[i]  / std1
        z10 = r10[i] / std10
        f1[i]  = np.arctan(z1)  / half_pi
        f10[i] = np.arctan(z10) / half_pi

    return f1, f10


# ── Simulator ──────────────────────────────────────────────────────────
LAMBDA_AUDDP = 0.001  # penalty weight on cumulative adverse excursion

@njit(cache=True)
def simulate_chunk(genes, f1, f10, close, pip, spread_pips, max_hold,
                   chunk_start, chunk_end):
    """
    Simulate on bars [chunk_start, chunk_end).
    score_per_trade = pnl_pips - λ × auddp_sum
      auddp_sum = Σ max(0, -pnl_pips) sampled every bar while in trade.
    fitness = mean(scores) × n_trades
    Returns (n_trades, n_long, n_short, mean_score)
    """
    pos          = 0      # 0=flat, 1=long, -1=short
    entry_price  = 0.0
    entry_bar    = 0
    auddp_sum    = 0.0    # cumulative adverse excursion in pips

    score_sum    = 0.0
    n_trades     = 0
    n_long       = 0
    n_short      = 0

    for i in range(chunk_start, chunk_end):
        f1_i  = f1[i]
        f10_i = f10[i]
        out   = forward(genes, f1_i, f10_i)

        # argmax → action: 0=BUY, 1=SELL, 2=FLATTEN
        if out[0] >= out[1] and out[0] >= out[2]:
            action = 0
        elif out[1] > out[0] and out[1] >= out[2]:
            action = 1
        else:
            action = 2

        if pos != 0:
            price = close[i]
            if pos == 1:
                pnl_pips = (price - entry_price) / pip
            else:
                pnl_pips = (entry_price - price) / pip

            # accumulate adverse excursion every bar
            if pnl_pips < 0.0:
                auddp_sum -= pnl_pips   # add |loss| to running sum

            force_close  = (i - entry_bar) >= max_hold
            signal_close = (pos == 1 and action == 1) or \
                           (pos == -1 and action == 0) or \
                           (action == 2)

            if force_close or signal_close:
                score = pnl_pips - LAMBDA_AUDDP * auddp_sum

                score_sum += score
                n_trades  += 1
                if pos == 1:
                    n_long += 1
                else:
                    n_short += 1
                pos         = 0
                entry_price = 0.0
                auddp_sum   = 0.0

        if pos == 0 and action != 2:
            direction   = 1 if action == 0 else -1
            entry_price = close[i] + direction * (spread_pips * 0.5) * pip
            entry_bar   = i
            auddp_sum   = spread_pips   # seed with spread cost
            pos         = direction

    if n_trades == 0:
        return 0, 0, 0, -10.0
    return n_trades, n_long, n_short, score_sum / n_trades


# ── Fitness ────────────────────────────────────────────────────────────
def fitness(genes_arr, pairs_data, max_hold):
    """WF fitness: min( mean_score × n_trades ) across 3 IS chunks. Minimized by CMA.

    Three-tier ramp to enforce bidirectionality without stagnation:
      Tier 0 (n_min=0):   -10  + nt*1e-4   [-10 .. -9.8]  ← different n_trades = different fitness
      Tier 1 (0<n_min<T): -10  + n_min*(9/T) [-9.55 .. -1]  ← linear climb toward threshold T
      Tier 2 (n_min>=T):  mean_sc * nt                     ← actual PnL fitness

    Variance is always non-zero so tolfun never triggers prematurely.
    """
    genes = np.asarray(genes_arr, dtype=np.float64)
    chunk_fits = []

    for pd_ in pairs_data:
        n_is  = pd_["n_is"]
        for ci in range(N_CHUNKS):
            cs = int(n_is * ci       / N_CHUNKS)
            ce = int(n_is * (ci + 1) / N_CHUNKS)
            nt, n_long, n_short, mean_sc = simulate_chunk(
                genes, pd_["f1_is"], pd_["f10_is"], pd_["close_is"],
                pd_["pip"], pd_["spread"], max_hold, cs, ce)
            n_min = min(n_long, n_short)
            if nt == 0:
                chunk_fits.append(-10.0)
            elif n_min == 0:
                # tier 0: uncapped so candidates with different n_trades differ
                chunk_fits.append(-10.0 + nt * 1e-5)
            elif n_min < MIN_TRADES_PER_DIR:
                # tier 1: ignore PnL, linear climb as minority direction grows
                chunk_fits.append(-10.0 + n_min * (9.0 / MIN_TRADES_PER_DIR))
            else:
                # tier 2: both directions met — actual PnL fitness
                chunk_fits.append(mean_sc * nt)

    return -min(chunk_fits)   # CMA minimizes → negate


# ── OOS evaluation ─────────────────────────────────────────────────────
def eval_oos(genes_arr, pairs_data, max_hold):
    genes = np.asarray(genes_arr, dtype=np.float64)
    results = []
    for pd_ in pairs_data:
        n = len(pd_["close_oos"])
        nt, n_long, n_short, mean_sc = simulate_chunk(
            genes, pd_["f1_oos"], pd_["f10_oos"], pd_["close_oos"],
            pd_["pip"], pd_["spread"], max_hold, 0, n)
        results.append({"pair": pd_["pair"], "n_trades": nt,
                        "n_long": n_long, "n_short": n_short,
                        "mean_score": mean_sc, "fitness": mean_sc * nt if nt else -10.0})
    return results


# ── MC permutation ─────────────────────────────────────────────────────
def mc_permutation(genes_arr, pairs_data, max_hold, n_shuffles=500, seed=0):
    """Fraction of random direction-flipped baselines beating real fitness."""
    real_fit = -fitness(genes_arr, pairs_data, max_hold)
    rng = np.random.default_rng(seed)
    beat = 0
    for s in range(n_shuffles):
        perm_data = []
        for pd_ in pairs_data:
            flip = rng.choice([-1.0, 1.0], size=pd_["n_is"])
            perm_data.append({**pd_,
                "f1_is":  pd_["f1_is"]  * flip,
                "f10_is": pd_["f10_is"] * flip})
        perm_fit = -fitness(genes_arr, perm_data, max_hold)
        if perm_fit >= real_fit:
            beat += 1
    return beat / n_shuffles


# ── Data loading ───────────────────────────────────────────────────────
def load_pair(pair, gran="S5"):
    s5_path = S5_DIR / f"{pair}_S5_BA.parquet"
    if s5_path.exists():
        df     = pd.read_parquet(s5_path)
        df.columns = [c.lower() for c in df.columns]
        closes = df["bid_c"].values.astype(np.float64)
    else:
        path   = S5_DIR / f"{pair}_S5.parquet"
        df     = pd.read_parquet(path).sort_index()
        df.columns = [c.lower() for c in df.columns]
        closes = df["close"].values.astype(np.float64)

    if gran == "M5":
        closes = closes[::12]   # resample S5→M5 cadence, ~12x fewer bars
        lb1, lb2 = 1, 10        # 1-bar (5 min) and 10-bar (50 min) returns
    else:
        lb1, lb2 = 12, 600      # 1-min and 50-min returns on S5

    n    = len(closes)
    n_is = int(n * IS_FRAC)
    pip  = PAIR_PIP[pair]

    f1, f10 = compute_zscore_inputs(closes, pip, POP_SIZE, lb1, lb2)
    print(f"  {pair}: {gran} {n:,} bars  IS={n_is:,}  OOS={n-n_is:,}  (lb1={lb1},lb2={lb2},stride={'12' if gran=='M5' else '1'})")

    return {
        "pair":      pair,
        "pip":       pip,
        "spread":    PAIR_SPREAD[pair],
        "f1_is":     f1[:n_is],
        "f10_is":    f10[:n_is],
        "close_is":  closes[:n_is],
        "f1_oos":    f1[n_is:],
        "f10_oos":   f10[n_is:],
        "close_oos": closes[n_is:],
        "n_is":      n_is,
        "n_oos":     n - n_is,
    }


# ── Parallel worker (pairs_data loaded once per process via initializer) ──
_worker_pairs_data = None
_worker_max_hold   = None

def _worker_init(pairs_data, max_hold):
    global _worker_pairs_data, _worker_max_hold
    _worker_pairs_data = pairs_data
    _worker_max_hold   = max_hold

def _worker_eval(genes_arr):
    return fitness(genes_arr, _worker_pairs_data, _worker_max_hold)


# ── Training ───────────────────────────────────────────────────────────
def train(pairs_data, max_hold, n_restarts, max_iter, sigma0, seed, n_workers=1):
    best_genes   = None
    best_fitness = np.inf

    pool = None
    if n_workers > 1:
        pool = mp.Pool(n_workers, initializer=_worker_init,
                       initargs=(pairs_data, max_hold))

    for r in range(n_restarts):
        rng = np.random.default_rng(seed + r * 997)
        x0  = rng.standard_normal(N_PARAMS) * sigma0

        opts = {
            "maxiter":        max_iter,
            "popsize":        4 + int(3 * np.log(N_PARAMS)),  # ~16
            "seed":           int(seed + r),
            "verbose":        -9,
            "tolx":           1e-8,
            "tolfun":         1e-9,
            "tolflatfitness": 1000,  # keep exploring even when stuck at -10 floor
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        t0 = time.time()
        while not es.stop():
            sols = es.ask()
            if pool is not None:
                fits = pool.map(_worker_eval, sols)
            else:
                fits = [fitness(x, pairs_data, max_hold) for x in sols]
            es.tell(sols, fits)

        f_val   = -fitness(es.result.xbest, pairs_data, max_hold)
        elapsed = time.time() - t0
        marker  = " ✓" if f_val > -best_fitness else ""
        print(f"  restart {r+1:>3}/{n_restarts}: fit={f_val:.4f}  "
              f"iters={es.result.iterations}  t={elapsed:.1f}s{marker}")

        if es.result.fbest < best_fitness:
            best_fitness = es.result.fbest
            best_genes   = es.result.xbest.copy()

    if pool is not None:
        pool.close(); pool.join()

    return best_genes, -best_fitness


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair",      default="EUR_JPY")
    parser.add_argument("--gran",      default="S5", choices=["S5", "M5"])
    parser.add_argument("--max-hold",  type=int,   default=None)
    parser.add_argument("--restarts",  type=int,   default=20)
    parser.add_argument("--max-iter",  type=int,   default=500)
    parser.add_argument("--sigma",     type=float, default=0.5)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--oos",       action="store_true")
    parser.add_argument("--mc",        action="store_true")
    parser.add_argument("--mc-n",      type=int,   default=500)
    parser.add_argument("--workers",   type=int,   default=4)
    args = parser.parse_args()

    pairs = [p.strip() for p in args.pair.split(",")]

    # default max_hold: 48 M5 bars (4h) or 200 S5 bars (17 min)
    max_hold = args.max_hold if args.max_hold is not None else (48 if args.gran == "M5" else 200)
    bar_mins  = 5 if args.gran == "M5" else (5/60)
    print(f"CMA-ES Z-score — AMDDP reward (pnl - λ×auddp, λ={LAMBDA_AUDDP})")
    print(f"Architecture: {N_IN}→{N_HID}→{N_OUT}  params={N_PARAMS}  wavelet activations")
    print(f"Pairs: {pairs}  gran={args.gran}  max_hold={max_hold} bars "
          f"({max_hold*bar_mins:.0f} min)  IS={IS_FRAC*100:.0f}%")
    print(f"CMA: {args.restarts} restarts × {args.max_iter} iters  σ0={args.sigma}")
    print()

    print("Loading data...")
    pairs_data = [load_pair(p, args.gran) for p in pairs]
    print()

    # JIT warmup
    print("JIT warmup...")
    dummy = np.ones(WARMUP + 50, dtype=np.float64)
    compute_zscore_inputs(dummy, 0.01, POP_SIZE, 12, 600)
    dummy_genes = np.zeros(N_PARAMS)
    simulate_chunk(dummy_genes, dummy, dummy, dummy, 0.01, 1.0, 200,
                   WARMUP, WARMUP + 40)
    print("  Done\n")

    print("Training...")
    t_train = time.time()
    best_genes, best_fit = train(
        pairs_data, max_hold, args.restarts, args.max_iter,
        args.sigma, args.seed, args.workers)
    print(f"\nTraining done in {time.time()-t_train:.1f}s  IS fitness = {best_fit:.4f}")

    tag  = f"cma_{args.gran.lower()}_mh{max_hold}_s{args.seed}"
    path = RESULTS_DIR / f"{tag}_{'_'.join(pairs)}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"genes": best_genes, "fitness": best_fit,
                     "pairs": pairs, "max_hold": max_hold, "gran": args.gran}, f)
    print(f"Saved: {path}")

    if args.oos:
        print("\nOOS evaluation:")
        oos = eval_oos(best_genes, pairs_data, max_hold)
        print(f"  {'Pair':<12} {'N':>6} {'L/S':<9} {'Score':>9} {'Fitness':>10}")
        print("  " + "─" * 50)
        pos = 0
        for r in oos:
            ls = f"{r['n_long']}L/{r['n_short']}S"
            print(f"  {r['pair']:<12} {r['n_trades']:>6} {ls:<9} "
                  f"{r['mean_score']:>9.4f} {r['fitness']:>10.2f}")
            if r["fitness"] > 0:
                pos += 1
        print(f"\n  Positive: {pos}/{len(oos)}")

    if args.mc:
        print(f"\nMC permutation ({args.mc_n} shuffles)...")
        perm_p = mc_permutation(best_genes, pairs_data, args.max_hold,
                                args.mc_n, args.seed)
        gate = "✅ PASS" if perm_p < 0.05 else "❌ FAIL"
        print(f"  perm_p = {perm_p:.4f}  {gate}")


if __name__ == "__main__":
    main()
