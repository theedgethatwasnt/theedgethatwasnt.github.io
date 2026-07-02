"""Parametric slot-4 CMA-ES runner.

Architecture (fixed across all candidates):
  Inputs (5): mc_d_a, mc_dd_a, er_norm, {candidate}, UPnL
  Hidden (4): per-node activation gene ∈ {tanh, sin, gauss}
  Output (3): BUY/SELL/FLATTEN
  Skip connections: input → output (always on)
  θ gene: latching entry threshold (sigmoid × 0.5)

Fitness: amddp1 = total_pnl − 0.01 × cum_mae, 3-chunk WF-in-fitness, hard gates.

Candidate sources:
  OHLC-derived     → loaded from causal parquet column
  State composite  → computed in simulator from open-trade mae/mfe each bar
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cma
import numpy as np
import pandas as pd
from numba import njit

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
N_IN = 5
N_OUT = 3
N_ACTS = 3  # tanh, sin, gauss
BARS_PER_DAY = 288.0
MAX_HOLD = 200
MIN_DIR = 0.15
N_CHUNKS = 3
AMDDP_COEF = 0.01

PAIR_PIP = {"EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
            "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
            "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
            "NZD_USD": 0.0001, "EUR_GBP": 0.0001}
PAIR_SPREAD = {"EUR_JPY": 2.3, "USD_JPY": 1.7, "GBP_JPY": 3.3, "AUD_JPY": 2.1,
               "CAD_JPY": 2.3, "CHF_JPY": 3.5, "NZD_JPY": 2.7,
               "EUR_USD": 1.6, "GBP_USD": 1.9, "AUD_USD": 1.3,
               "NZD_USD": 1.5, "EUR_GBP": 1.4}

STATE_CANDIDATES = {"THI": 1, "give_back": 2, "recovery": 3}


def _candidate_mode(name, df_columns):
    """Decide: 0 = OHLC-derived (read from parquet col), >0 = state composite."""
    if name in STATE_CANDIDATES:
        return STATE_CANDIDATES[name]
    if name in df_columns:
        return 0
    raise SystemExit(f"Candidate '{name}' not in parquet columns and not a state composite.\n"
                     f"Did you port it into FXFeatureBuilder + rebuild parquets?\n"
                     f"Available cols: {sorted(df_columns)}")


# ── Param count ────────────────────────────────────────────────────────
N_HID = 4
USE_SKIP = True
NP = (N_IN * N_HID + N_HID            # W1 + b1
      + N_HID * N_OUT                 # W2
      + N_IN * N_OUT                  # Wskip
      + N_OUT                         # b2
      + N_HID                         # act_genes
      + 1)                            # θ


# ── Simulator ──────────────────────────────────────────────────────────
@njit(cache=True, inline="always")
def _activate(z, aid):
    if aid == 0:
        return np.tanh(z)
    elif aid == 1:
        return np.sin(z)
    else:
        return np.exp(-z * z)


@njit(cache=True)
def _decode_theta(g):
    return 0.5 / (1.0 + np.exp(-g))


@njit(cache=True)
def _decode_act(g):
    f = g - np.floor(g)
    aid = int(f * N_ACTS)
    if aid < 0:
        aid = 0
    if aid >= N_ACTS:
        aid = N_ACTS - 1
    return aid


@njit(cache=True)
def _state_feature(state_mode, upnl, mae_signed, mfe_signed):
    """state_mode: 1=THI, 2=give_back, 3=recovery. Returns scalar in ~[0,1]."""
    eps = 1e-6
    if state_mode == 1:
        denom = (mfe_signed - mae_signed) + eps
        return (upnl - mae_signed) / denom
    elif state_mode == 2:
        return (max(mfe_signed, 0.0) - upnl) / (max(mfe_signed, 0.0) + eps)
    elif state_mode == 3:
        ma = -mae_signed  # positive magnitude of worst drawdown
        return (upnl - mae_signed) / (ma + eps)
    return 0.0


@njit(cache=True)
def simulate(genes, mc_d, mc_dd, er, candidate_arr, state_mode,
             mid, pip, spread_pips, max_hold, chunk_start, chunk_end, amddp_coef):
    """Returns (nt, total_score, total_pnl, cum_mae, nl, ns)."""
    n = len(mid)
    start = max(chunk_start + 20, 20)
    end = min(chunk_end, n - 1)
    if end <= start:
        return 0, 0.0, 0.0, 0.0, 0, 0

    # Gene offsets
    w1_end = N_IN * N_HID
    b1_end = w1_end + N_HID
    w2_end = b1_end + N_HID * N_OUT
    wskip_end = w2_end + N_IN * N_OUT
    b2_end = wskip_end + N_OUT
    # act_genes at [b2_end : b2_end + N_HID], θ at genes[-1]

    theta = _decode_theta(genes[-1])

    nt = 0; nl = 0; ns = 0
    total_pnl = 0.0; cum_mae = 0.0
    position = 0
    entry_price = 0.0; entry_bar = 0
    # Signed running extremes (relative to entry, in pips):
    upnl = 0.0; mae_s = 0.0; mfe_s = 0.0

    x = np.empty(N_IN)
    h = np.empty(N_HID)

    for i in range(start, end):
        if position != 0:
            upnl = (mid[i] - entry_price) / pip * position
            if upnl < mae_s:
                mae_s = upnl
            if upnl > mfe_s:
                mfe_s = upnl
            cum_mae += -mae_s if mae_s < 0 else 0.0
        else:
            upnl = 0.0; mae_s = 0.0; mfe_s = 0.0

        # Slot-4 value: either parquet column or state composite
        if state_mode == 0:
            slot4 = candidate_arr[i]
        else:
            slot4 = _state_feature(state_mode, upnl, mae_s, mfe_s)
            # state composites only defined when in-position; else 0.5 (neutral)
            if position == 0:
                slot4 = 0.5

        x[0] = mc_d[i]
        x[1] = mc_dd[i]
        x[2] = er[i]
        x[3] = slot4
        x[4] = np.tanh(upnl / 10.0)

        # Hidden layer
        for k in range(N_HID):
            z = genes[b1_end - N_HID + k]
            w1_row = k * N_IN
            for j in range(N_IN):
                z += genes[w1_row + j] * x[j]
            aid_k = _decode_act(genes[b2_end + k])
            h[k] = _activate(z, aid_k)

        # Output layer (skip always on)
        out = np.empty(N_OUT)
        for o in range(N_OUT):
            val = genes[wskip_end + o]  # b2
            w2_row = b1_end + o * N_HID
            for k in range(N_HID):
                val += genes[w2_row + k] * h[k]
            wskip_row = w2_end + o * N_IN
            for j in range(N_IN):
                val += genes[wskip_row + j] * x[j]
            out[o] = val

        ob = out[0]; os_ = out[1]; of = out[2]

        # Max-hold force close
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid[i] - entry_price) / pip * position
            total_pnl += pnl
            nt += 1
            if position > 0: nl += 1
            else: ns += 1
            position = 0

        if position == 0:
            if (ob - of) > theta and (ob - os_) > theta:
                position = 1
                entry_price = mid[i] + spread_pips * pip
                entry_bar = i
                mae_s = -spread_pips; mfe_s = 0.0
            elif (os_ - of) > theta and (os_ - ob) > theta:
                position = -1
                entry_price = mid[i] - spread_pips * pip
                entry_bar = i
                mae_s = -spread_pips; mfe_s = 0.0
        else:
            close_now = False; new_pos = 0
            if of > ob and of > os_:
                close_now = True
            elif position == 1 and os_ > ob and os_ > of:
                close_now = True; new_pos = -1
            elif position == -1 and ob > os_ and ob > of:
                close_now = True; new_pos = 1
            if close_now:
                pnl = (mid[i] - entry_price) / pip * position
                total_pnl += pnl
                nt += 1
                if position > 0: nl += 1
                else: ns += 1
                position = new_pos
                if new_pos != 0:
                    if new_pos == 1:
                        entry_price = mid[i] + spread_pips * pip
                    else:
                        entry_price = mid[i] - spread_pips * pip
                    entry_bar = i
                    mae_s = -spread_pips; mfe_s = 0.0
                else:
                    mae_s = 0.0; mfe_s = 0.0

    total_score = total_pnl - amddp_coef * cum_mae
    return nt, total_score, total_pnl, cum_mae, nl, ns


# ── Fitness (pool worker) ──────────────────────────────────────────────
_W = {}


def _winit(mc_d, mc_dd, er, cand, state_mode, mid, pip, spread):
    _W.update({"mc_d": mc_d, "mc_dd": mc_dd, "er": er, "cand": cand,
               "state_mode": state_mode, "mid": mid, "pip": pip, "spread": spread})


def _eval_one(genes):
    n = len(_W["mid"])
    tl = 0; ts = 0; tt = 0; tscore = 0.0
    chunk_sps = []; losing = 0.0
    chunk_trades_list = []
    for ci in range(N_CHUNKS):
        c_s = int(n * ci / N_CHUNKS)
        c_e = int(n * (ci + 1) / N_CHUNKS)
        nt, score, _p, _c, nl, ns = simulate(
            genes, _W["mc_d"], _W["mc_dd"], _W["er"], _W["cand"], _W["state_mode"],
            _W["mid"], _W["pip"], _W["spread"], MAX_HOLD, c_s, c_e, AMDDP_COEF)
        tl += nl; ts += ns; tt += nt; tscore += score
        days = (c_e - c_s) / BARS_PER_DAY
        sps = score / days if days > 0 else 0.0
        chunk_sps.append(sps)
        chunk_trades_list.append(nt)
        if sps < 0:
            losing += -sps
    total_days = n / BARS_PER_DAY
    base_sps = tscore / total_days if total_days > 0 else 0.0

    chunk_days = (n / N_CHUNKS) / BARS_PER_DAY
    min_per_chunk = int(chunk_days)
    min_chunk_trades = min(chunk_trades_list) if chunk_trades_list else 0
    trades_short = max(0, min_per_chunk - min_chunk_trades)
    dir_ratio = (min(tl, ts) / tt) if tt > 0 else 0.0
    dir_short = max(0.0, MIN_DIR - dir_ratio)

    if trades_short > 0 or dir_short > 0:
        return 500.0 + trades_short * 0.1 + dir_short * 100.0

    asym = (1.0 - 2.0 * dir_ratio) * 25.0
    activity = max(0.0, 100.0 - tt) * 2.0
    all_prof = all(s > 0 for s in chunk_sps)
    if all_prof:
        score = min(chunk_sps) - asym
    else:
        score = base_sps - asym - activity - losing * 2.0
    return -score


_POOL = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    help="Slot-4 candidate name (OHLC-derived: parquet col; state: THI/give_back/recovery)")
    ap.add_argument("--pair", default="EUR_JPY")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gens", type=int, default=50)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoother", default="kalman10")
    args = ap.parse_args()

    candidate = args.candidate
    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]

    parq = PROJECT / f"data/m5_ohlc/{pair}_M5_{args.smoother}_causal.parquet"
    df = pd.read_parquet(parq)
    print(f"[{candidate}][{pair}] Loaded {len(df):,} bars from {parq.name}", flush=True)

    state_mode = _candidate_mode(candidate, set(df.columns))

    mid = df["close"].values.astype(np.float64)
    mc_d = df["mc_d_a"].values.astype(np.float64)
    mc_dd = df["mc_dd_a"].values.astype(np.float64)
    er = df["er_norm"].values.astype(np.float64)

    if state_mode == 0:
        cand = df[candidate].values.astype(np.float64)
        print(f"[{candidate}] slot4 from parquet: range=[{cand[100:].min():+.3f}, {cand[100:].max():+.3f}] std={cand[100:].std():.4f}", flush=True)
    else:
        cand = np.zeros(len(df))  # unused when state_mode > 0
        print(f"[{candidate}] slot4 computed live (state_mode={state_mode})", flush=True)

    n_bars = len(df)
    split = int(n_bars * 0.7)
    mid_is, mid_oos = mid[:split], mid[split:]
    mc_d_is, mc_d_oos = mc_d[:split], mc_d[split:]
    mc_dd_is, mc_dd_oos = mc_dd[:split], mc_dd[split:]
    er_is, er_oos = er[:split], er[split:]
    cand_is, cand_oos = cand[:split], cand[split:]
    print(f"IS: {split:,} bars ({split/BARS_PER_DAY:.0f}d), OOS: {n_bars-split:,} ({(n_bars-split)/BARS_PER_DAY:.0f}d)")

    # JIT warm
    dummy = np.zeros(NP)
    simulate(dummy, mc_d_is[:300], mc_dd_is[:300], er_is[:300], cand_is[:300],
             state_mode, mid_is[:300], pip, spread, 50, 0, 300, AMDDP_COEF)

    # CMA
    np.random.seed(args.seed)
    x0 = np.zeros(NP)
    b2_end = N_IN * N_HID + N_HID + N_HID * N_OUT + N_IN * N_OUT + N_OUT
    for k in range(N_HID):
        x0[b2_end + k] = np.random.uniform(0, 1)
    x0[-1] = 0.0  # θ → 0.25

    global _POOL
    _POOL = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_winit,
        initargs=(mc_d_is, mc_dd_is, er_is, cand_is, state_mode, mid_is, pip, spread))

    es = cma.CMAEvolutionStrategy(x0, args.sigma, {
        "popsize": args.pop, "seed": args.seed, "verbose": -9,
        "maxiter": args.gens, "tolx": 1e-8,
    })

    print(f"CMA 5→{N_HID}→3+skip | {NP} params | pop {args.pop} | gens {args.gens}", flush=True)
    t0 = time.time()
    best_score = float("inf"); best_genes = x0.copy()
    gen = 0
    while not es.stop() and gen < args.gens:
        solutions = es.ask()
        losses = list(_POOL.map(_eval_one, solutions))
        es.tell(solutions, losses)
        gb = min(losses); gi = losses.index(gb)
        if gb < best_score:
            best_score = gb
            best_genes = solutions[gi].copy()
        if gen % 10 == 0:
            acts = []
            for k in range(N_HID):
                g = best_genes[b2_end + k]
                aid = int((g - np.floor(g)) * N_ACTS)
                aid = min(max(aid, 0), N_ACTS - 1)
                acts.append(["tanh", "sin", "gauss"][aid])
            theta_val = 0.5 / (1.0 + np.exp(-best_genes[-1]))
            print(f"  gen {gen:3d} | fit {-best_score:+.3f} | acts=[{'/'.join(acts)}] θ={theta_val:.3f}", flush=True)
        gen += 1
    elapsed = time.time() - t0

    # Evaluate winner
    is_nt, _, is_pnl, is_cm, is_nl, is_ns = simulate(
        best_genes, mc_d_is, mc_dd_is, er_is, cand_is, state_mode, mid_is,
        pip, spread, MAX_HOLD, 0, len(mid_is), AMDDP_COEF)
    oos_nt, _, oos_pnl, oos_cm, oos_nl, oos_ns = simulate(
        best_genes, mc_d_oos, mc_dd_oos, er_oos, cand_oos, state_mode, mid_oos,
        pip, spread, MAX_HOLD, 0, len(mid_oos), AMDDP_COEF)
    is_days = len(mid_is) / BARS_PER_DAY
    oos_days = len(mid_oos) / BARS_PER_DAY
    is_dir = min(is_nl, is_ns) / max(is_nt, 1)
    oos_dir = min(oos_nl, oos_ns) / max(oos_nt, 1)

    acts = []
    for k in range(N_HID):
        g = best_genes[b2_end + k]
        aid = int((g - np.floor(g)) * N_ACTS)
        aid = min(max(aid, 0), N_ACTS - 1)
        acts.append(["tanh", "sin", "gauss"][aid])
    theta_val = 0.5 / (1.0 + np.exp(-best_genes[-1]))

    print(f"\n{'='*72}")
    print(f"  SLOT4={candidate}  pair={pair}  seed={args.seed}")
    print(f"{'='*72}")
    print(f"  Winner: acts=[{'/'.join(acts)}] θ={theta_val:.3f} fit={-best_score:+.3f}")
    print(f"  IS : {is_nt}T L/S={is_nl}/{is_ns}  {is_pnl/is_days:+.2f} p/d  dir={is_dir:.2f}  cumMAE={is_cm:.0f}")
    print(f"  OOS: {oos_nt}T L/S={oos_nl}/{oos_ns}  {oos_pnl/oos_days:+.2f} p/d  dir={oos_dir:.2f}  cumMAE={oos_cm:.0f}")
    print(f"  Elapsed: {elapsed:.0f}s")

    out = OUT_DIR / f"slot4_{candidate}_{pair}_s{args.seed}.pkl"
    payload = {
        "candidate": candidate, "pair": pair, "seed": args.seed,
        "state_mode": state_mode, "n_params": int(NP),
        "topology": f"5→{N_HID}→3+skip",
        "fitness": float(-best_score),
        "activations": acts, "theta": float(theta_val),
        "genes": best_genes,
        "is": {"n_trades": int(is_nt), "pnl": float(is_pnl), "pd": float(is_pnl/is_days),
               "dir": float(is_dir), "cum_mae": float(is_cm)},
        "oos": {"n_trades": int(oos_nt), "pnl": float(oos_pnl), "pd": float(oos_pnl/oos_days),
                "dir": float(oos_dir), "cum_mae": float(oos_cm)},
        "elapsed_s": float(elapsed), "gens_run": int(gen),
    }
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    # Also drop JSON metrics (compact, loop reads these)
    jout = OUT_DIR / f"slot4_{candidate}_{pair}_s{args.seed}.json"
    payload_json = {k: v for k, v in payload.items() if k not in ("genes",)}
    with open(jout, "w") as f:
        json.dump(payload_json, f, indent=2, default=float)
    print(f"Saved: {out.name}, {jout.name}")


if __name__ == "__main__":
    main()
