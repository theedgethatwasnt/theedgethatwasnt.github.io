#!/usr/bin/env python3
"""
CMA-NN on S5 cadence: pure sin activations, weights+biases only.

Inputs: mc_d_a, mc_dd_a, er_norm, macd_hist + upnl, mae, mfe = 7
Architecture: 7 → 3 (sin) → 3 (linear, argmax → BUY/SELL/FLATTEN)
Data: S5 bid/ask OHLC → mid close, compute indicators inline.

Usage:
    python3 train_s5_sin.py --pair EUR_JPY
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

import cma

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

S5_DATA_DIR = PROJECT_ROOT / "data" / "s5_ohlc"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
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

# ── Architecture ──────────────────────────────────────────────────────
N_HID = 3
N_OUT = 3
N_POS_STATE = 3  # upnl, mae, mfe
N_MARKET = 4     # mc_d_a, mc_dd_a, er_norm, macd_hist
N_IN = N_MARKET + N_POS_STATE  # 7


def n_params():
    return N_IN * N_HID + N_HID + N_HID * N_OUT + N_OUT  # W1 + b1 + W2 + b2


# ── Indicators (all inline, S5 cadence) ──────────────────────────────

@njit(cache=True)
def compute_er_norm(closes, window=60):
    """Kaufman ER, arctan-normalized. Curator-identical."""
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
def compute_macd_hist(closes):
    """MACD histogram (EMA12 - EMA26 - Signal9) / ATR14, clipped to [-1,+1].

    Standard MACD normalized by ATR to make it scale-independent.
    """
    n = len(closes)
    out = np.zeros(n)
    if n < 35:
        return out

    # EMA12
    alpha12 = 2.0 / 13.0
    ema12 = closes[0]
    # EMA26
    alpha26 = 2.0 / 27.0
    ema26 = closes[0]
    # Signal line (EMA9 of MACD)
    alpha9 = 2.0 / 10.0
    macd_val = 0.0
    signal = 0.0
    # ATR14
    atr = 0.0
    atr_init = False

    for i in range(1, n):
        ema12 = alpha12 * closes[i] + (1.0 - alpha12) * ema12
        ema26 = alpha26 * closes[i] + (1.0 - alpha26) * ema26
        macd_val = ema12 - ema26

        if i == 1:
            signal = macd_val
        else:
            signal = alpha9 * macd_val + (1.0 - alpha9) * signal

        # ATR (simplified: |close - prev_close| since S5 has no real H/L range)
        tr = abs(closes[i] - closes[i - 1])
        if not atr_init and i >= 14:
            atr_init = True
            s = 0.0
            for k in range(i - 13, i + 1):
                s += abs(closes[k] - closes[k - 1])
            atr = s / 14.0
        elif atr_init:
            atr = (atr * 13.0 + tr) / 14.0

        if i >= 34 and atr > 0:
            hist = macd_val - signal
            normed = hist / (2.0 * atr)
            if normed > 1.0:
                normed = 1.0
            elif normed < -1.0:
                normed = -1.0
            out[i] = normed

    return out


# ── Bar-by-bar simulator ─────────────────────────────────────────────
@njit(cache=True)
def simulate_chunk(market, mid, pip, spread_pips, max_hold,
                   weights, chunk_start, chunk_end):
    """Walk bars with 7→3(sin)→3 network. Returns (n_trades, pnl, n_long, n_short)."""
    w1_end = N_IN * N_HID
    b1_end = w1_end + N_HID
    w2_end = b1_end + N_HID * N_OUT

    start_bar = max(chunk_start + 60, 60)  # skip warmup
    end_bar = min(chunk_end, len(mid) - 1)
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

    x = np.zeros(N_IN)
    h = np.zeros(N_HID)

    for i in range(start_bar, end_bar):
        if position != 0:
            upnl_pips = (mid[i] - entry_price) / pip * position
            adverse = -upnl_pips
            if adverse > mae_pips:
                mae_pips = adverse
            if upnl_pips > mfe_pips:
                mfe_pips = upnl_pips
        else:
            upnl_pips = 0.0
            mae_pips = 0.0
            mfe_pips = 0.0

        # Market features
        for k in range(N_MARKET):
            x[k] = market[k, i]
        x[N_MARKET]     = np.tanh(upnl_pips / 20.0)
        x[N_MARKET + 1] = np.tanh(mae_pips / 20.0)
        x[N_MARKET + 2] = np.tanh(mfe_pips / 20.0)

        # Forward: all sin
        for j in range(N_HID):
            z = weights[w1_end + j]
            for k in range(N_IN):
                z += weights[j * N_IN + k] * x[k]
            h[j] = np.sin(z)

        out_buy  = weights[w2_end + 0]
        out_sell = weights[w2_end + 1]
        out_flat = weights[w2_end + 2]
        for j in range(N_HID):
            out_buy  += weights[b1_end + 0 * N_HID + j] * h[j]
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
            pnl = (mid[i] - entry_price) / pip * position
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
                entry_price = mid[i] + spread_pips * pip
                entry_bar = i
                mae_pips = spread_pips
                mfe_pips = 0.0
            elif action == 2:
                position = -1
                entry_price = mid[i] - spread_pips * pip
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
                pnl = (mid[i] - entry_price) / pip * position
                if n_trades < n_capacity:
                    pnls[n_trades] = pnl
                    if position > 0:
                        n_long += 1
                    else:
                        n_short += 1
                    n_trades += 1
                position = new_pos
                if new_pos == 1:
                    entry_price = mid[i] + spread_pips * pip
                    entry_bar = i
                    mae_pips = spread_pips
                    mfe_pips = 0.0
                elif new_pos == -1:
                    entry_price = mid[i] - spread_pips * pip
                    entry_bar = i
                    mae_pips = spread_pips
                    mfe_pips = 0.0
                else:
                    entry_price = 0.0

    # Close remaining
    if position != 0 and end_bar > start_bar:
        pnl = (mid[end_bar - 1] - entry_price) / pip * position
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


# ── Fitness (shaped, same as train_cma_v2) ────────────────────────────
S5_BARS_PER_DAY = 17280  # 24h * 720 bars/h


def fitness_neg(weights, market, mid, pip, spread, max_hold, n_chunks, min_dir):
    n_bars = len(mid)
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
            market, mid, pip, spread, max_hold, weights, c_start, c_end)
        total_long += nl
        total_short += ns
        total_trades += nt
        total_pnl += pnl
        n_days = (c_end - c_start) / S5_BARS_PER_DAY
        pps = pnl / n_days if n_days > 0 else 0.0
        chunk_pps.append(pps)
        if pps < 0:
            losing_chunk_loss += -pps

    total_days = n_bars / S5_BARS_PER_DAY
    base_pps = total_pnl / total_days if total_days > 0 else 0.0

    if total_trades == 0:
        return 500.0 - base_pps
    dir_ratio = min(total_long, total_short) / total_trades

    asym_penalty = (1.0 - 2.0 * dir_ratio) * 50.0
    activity_penalty = max(0.0, 30.0 - total_trades) * 2.0
    losing_pen = losing_chunk_loss * 2.0

    all_profitable = all(p > 0 for p in chunk_pps)
    if all_profitable and dir_ratio >= min_dir:
        score = min(chunk_pps) - asym_penalty
    else:
        score = base_pps - asym_penalty - activity_penalty - losing_pen

    return -score


def passes_hard_gates(weights, market, mid, pip, spread, max_hold, n_chunks, min_dir):
    n_bars = len(mid)
    total_long = 0
    total_short = 0
    total_trades = 0
    chunk_pps = []
    for ci in range(n_chunks):
        c_start = int(n_bars * ci / n_chunks)
        c_end = int(n_bars * (ci + 1) / n_chunks)
        nt, pnl, nl, ns = simulate_chunk(
            market, mid, pip, spread, max_hold, weights, c_start, c_end)
        total_long += nl
        total_short += ns
        total_trades += nt
        n_days = (c_end - c_start) / S5_BARS_PER_DAY
        min_trades = max(20, int(n_days * 0.5))
        if nt < min_trades or pnl <= 0:
            return False, None
        chunk_pps.append(pnl / n_days)
    if total_trades < 30:
        return False, None
    if min(total_long, total_short) / total_trades < min_dir:
        return False, None
    return True, min(chunk_pps)


# ── Worker pool ───────────────────────────────────────────────────────
_W = {}


def _worker_init(market, mid, pip, spread, max_hold, n_chunks, min_dir):
    _W["market"] = market
    _W["mid"] = mid
    _W["pip"] = pip
    _W["spread"] = spread
    _W["max_hold"] = max_hold
    _W["n_chunks"] = n_chunks
    _W["min_dir"] = min_dir


def _worker_fit(vec):
    return fitness_neg(
        vec, _W["market"], _W["mid"], _W["pip"], _W["spread"],
        _W["max_hold"], _W["n_chunks"], _W["min_dir"])


# ── Main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="CMA-NN S5 trainer: 7→3(sin)→3")
    ap.add_argument("--pair", default="EUR_JPY")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--popsize", type=int, default=24)
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--max-hold", type=int, default=2400,
                    help="Max hold in S5 bars (2400 = 3.3h)")
    ap.add_argument("--n-chunks", type=int, default=3)
    ap.add_argument("--min-dir-ratio", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]
    np_total = n_params()

    print(f"{'='*65}")
    print(f"  CMA-NN S5: {pair}")
    print(f"  Network: {N_IN} → {N_HID} (sin) → {N_OUT}  ({np_total} params)")
    print(f"  Inputs: mc_d_a, mc_dd_a, er_norm, macd_hist + upnl, mae, mfe")
    print(f"  Activation: sin (fixed, all nodes)")
    print(f"  Spread: {spread} pips at entry")
    print(f"  Max hold: {args.max_hold} S5 bars ({args.max_hold/720:.1f}h)")
    print(f"  CMA-ES: pop={args.popsize}, gens={args.gens}, σ₀={args.sigma0}")
    print(f"  Seed: {args.seed} | WF chunks: {args.n_chunks}")
    print(f"{'='*65}")

    # ── Load S5 data ──────────────────────────────────────────
    s5_path = S5_DATA_DIR / f"{pair}_S5_BA.parquet"
    if not s5_path.exists():
        print(f"ERROR: {s5_path} not found")
        sys.exit(1)

    print(f"\nLoading {s5_path.name}...")
    df = pd.read_parquet(s5_path, engine="pyarrow")
    # Mid = (bid_c + ask_c) / 2
    bid_c = df["bid_c"].values.astype(np.float64)
    ask_c = df["ask_c"].values.astype(np.float64)
    mid = (bid_c + ask_c) / 2.0

    # For ASI we need OHLC — use mid of bid/ask
    mid_o = ((df["bid_o"].values.astype(np.float64) +
              df["ask_o"].values.astype(np.float64)) / 2.0)
    mid_h = ((df["bid_h"].values.astype(np.float64) +
              df["ask_h"].values.astype(np.float64)) / 2.0)
    mid_l = ((df["bid_l"].values.astype(np.float64) +
              df["ask_l"].values.astype(np.float64)) / 2.0)
    n = len(mid)
    n_days = n / S5_BARS_PER_DAY
    del df
    gc.collect()
    print(f"  {n:,} S5 bars ({n_days:.1f} days)")

    # ── Compute indicators ────────────────────────────────────
    print("Computing indicators...")
    t0_ind = time.time()

    # ASI-MC (curator-identical)
    from lib.asi_indicator import compute_asi_mc
    mc_d_a, mc_dd_a = compute_asi_mc(mid_o, mid_h, mid_l, mid, n)

    # ER norm
    er_norm = compute_er_norm(mid, window=60)

    # MACD histogram
    macd_hist = compute_macd_hist(mid)

    print(f"  done in {time.time()-t0_ind:.1f}s")
    del mid_o, mid_h, mid_l
    gc.collect()

    # Stack market features
    market = np.stack([mc_d_a, mc_dd_a, er_norm, macd_hist], axis=0)
    del mc_d_a, mc_dd_a, er_norm, macd_hist
    gc.collect()

    # ── IS / OOS split ────────────────────────────────────────
    split = int(n * 0.7)
    m_is = market[:, :split].copy()
    mid_is = mid[:split].copy()
    m_oos = market[:, split:].copy()
    mid_oos = mid[split:].copy()
    del market, mid
    gc.collect()

    is_days = split / S5_BARS_PER_DAY
    oos_days = (n - split) / S5_BARS_PER_DAY
    print(f"\n  IS: {split:,} bars ({is_days:.1f} days)")
    print(f"  OOS: {n - split:,} bars ({oos_days:.1f} days)")
    for i, name in enumerate(["mc_d_a", "mc_dd_a", "er_norm", "macd_hist"]):
        print(f"  {name:12s} IS range: [{m_is[i].min():+.4f}, {m_is[i].max():+.4f}]  "
              f"std={m_is[i].std():.4f}")

    # ── JIT warmup ────────────────────────────────────────────
    print("\nJIT warming up...")
    warm_w = np.zeros(np_total)
    simulate_chunk(m_is[:, :500], mid_is[:500], pip, spread, 100, warm_w, 0, 500)
    print("  done.")

    # ── CMA-ES with multiprocessing ──────────────────────────
    from concurrent.futures import ProcessPoolExecutor

    init_args = (m_is, mid_is, pip, spread, args.max_hold,
                 args.n_chunks, args.min_dir_ratio)
    pool = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=init_args)

    np.random.seed(args.seed)
    x0 = np.random.randn(np_total) * 0.1
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

    best_fit = 1e18
    best_vec = None
    best_valid_pps = None
    best_valid_vec = None
    gen = 0
    t0 = time.time()

    print(f"\nTraining...\n")
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
            best_vec, m_is, mid_is, pip, spread, args.max_hold,
            args.n_chunks, args.min_dir_ratio)
        if ok and (best_valid_pps is None or min_pps > best_valid_pps):
            best_valid_pps = min_pps
            best_valid_vec = np.array(best_vec)

        if gen % 10 == 0:
            valid_str = (f"{best_valid_pps:+.2f}p/d"
                         if best_valid_pps is not None else "—")
            print(f"  Gen {gen:>3}: raw_fit={best_fit:>10.2f}  "
                  f"valid={valid_str}  σ={es.sigma:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s")
        gen += 1

    pool.shutdown(wait=False)
    elapsed = time.time() - t0

    # ── Evaluate best on IS and OOS ──────────────────────────
    eval_vec = best_valid_vec if best_valid_vec is not None else best_vec

    def eval_full(m, mid_arr, label):
        nt, pnl, nl, ns = simulate_chunk(
            m, mid_arr, pip, spread, args.max_hold, eval_vec, 0, len(mid_arr))
        nd = len(mid_arr) / S5_BARS_PER_DAY
        dr = min(nl, ns) / nt if nt > 0 else 0
        pps = pnl / nd if nd > 0 else 0
        print(f"  {label:4s}: {pnl:>+10.1f}p  {pps:>+8.2f}p/day  "
              f"{nt} trades (L={nl} S={ns} dr={dr:.2f})  "
              f"{nd:.1f} days")
        return {"pnl": round(float(pnl), 1), "pps": round(float(pps), 2),
                "trades": int(nt), "n_long": int(nl), "n_short": int(ns),
                "dir_ratio": round(float(dr), 3), "days": round(float(nd), 1)}

    print(f"\n{'='*65}")
    print(f"  RESULTS  ({gen} gens, {elapsed:.0f}s)")
    print(f"{'='*65}")
    is_res = eval_full(m_is, mid_is, "IS")
    oos_res = eval_full(m_oos, mid_oos, "OOS")

    gate_ok, gate_pps = passes_hard_gates(
        eval_vec, m_is, mid_is, pip, spread, args.max_hold,
        args.n_chunks, args.min_dir_ratio)
    print(f"\n  Hard gates: {'PASS' if gate_ok else 'FAIL'}  "
          f"(min_chunk={gate_pps:+.2f}p/d)" if gate_ok else
          f"\n  Hard gates: FAIL")

    # ── Save ──────────────────────────────────────────────────
    import pickle, json
    label = f"s5_sin_{pair}_s{args.seed}"
    pkl_path = RESULTS_DIR / f"{label}_best.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"weights": eval_vec, "n_in": N_IN, "n_hid": N_HID,
                      "n_out": N_OUT, "activation": "sin",
                      "market_features": ["mc_d_a", "mc_dd_a", "er_norm", "macd_hist"],
                      "pair": pair, "seed": args.seed}, f)
    print(f"\n  Genome saved: {pkl_path}")

    res_path = RESULTS_DIR / f"{label}_result.json"
    result = {
        "pair": pair, "seed": args.seed, "gens": gen,
        "architecture": f"{N_IN}→{N_HID}(sin)→{N_OUT}",
        "n_params": np_total, "elapsed_s": round(elapsed, 1),
        "max_hold_bars": args.max_hold,
        "spread_pips": spread,
        "is": is_res, "oos": oos_res,
        "hard_gates": gate_ok,
    }
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Result saved: {res_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
