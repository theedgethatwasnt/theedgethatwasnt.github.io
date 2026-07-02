"""Path B (new core) + Path C (H1 cadence) CMA runner.

Path B (--path B): 4 fixed core + 1 swap + UPnL = 6 inputs
  Core: cci, bb_width, ema21_ratio, atr_ratio
  Slot: --candidate X
  Arch: 6 → 4+skip → 3

Path C (--path C): V3 core + 1 swap + UPnL = 5 inputs, NN evaluated only at H1 boundaries
  Core: mc_d_a, mc_dd_a, er_norm
  Slot: --candidate X
  Arch: 5 → 4+skip → 3
  Cadence: NN runs only every 12 M5 bars; position / MAE / MFE update every bar
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

N_OUT = 3
N_ACTS = 3  # tanh, sin, gauss
BARS_PER_DAY = 288.0
MAX_HOLD = 200
MIN_DIR = 0.15
N_CHUNKS = 3
AMDDP_COEF = 0.01
H1_STRIDE = 12  # M5 bars per H1 bar

PAIR_PIP = {"EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
            "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
            "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
            "NZD_USD": 0.0001, "EUR_GBP": 0.0001}
PAIR_SPREAD = {"EUR_JPY": 2.3, "USD_JPY": 1.7, "GBP_JPY": 3.3, "AUD_JPY": 2.1,
               "CAD_JPY": 2.3, "CHF_JPY": 3.5, "NZD_JPY": 2.7,
               "EUR_USD": 1.6, "GBP_USD": 1.9, "AUD_USD": 1.3,
               "NZD_USD": 1.5, "EUR_GBP": 1.4}

PATH_B_CORE = ["cci", "bb_width", "ema21_ratio", "atr_ratio"]
PATH_C_CORE = ["mc_d_a", "mc_dd_a", "er_norm"]


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
    if aid < 0: aid = 0
    if aid >= N_ACTS: aid = N_ACTS - 1
    return aid


@njit(cache=True)
def simulate(genes, core_inputs, cand_arr, n_core, use_cand, n_hid, n_in,
             mid, pip, spread_pips, max_hold, chunk_start, chunk_end,
             amddp_coef, h1_stride):
    """Unified sim. core_inputs shape (n_core, n_bars). cand_arr shape (n_bars,).
    n_in = n_core + (1 if use_cand else 0) + 1 (UPnL).
    h1_stride=1 → decide every bar (M5). h1_stride=12 → H1 cadence."""
    n = len(mid)
    start = max(chunk_start + 20, 20)
    end = min(chunk_end, n - 1)
    if end <= start:
        return 0, 0.0, 0.0, 0.0, 0, 0

    # Gene layout
    w1_end = n_in * n_hid
    b1_end = w1_end + n_hid
    w2_end = b1_end + n_hid * N_OUT
    wskip_end = w2_end + n_in * N_OUT
    b2_end = wskip_end + N_OUT

    theta = _decode_theta(genes[-1])

    nt = 0; nl = 0; ns = 0
    total_pnl = 0.0; cum_mae = 0.0
    position = 0
    entry_price = 0.0; entry_bar = 0
    upnl = 0.0; mae_s = 0.0; mfe_s = 0.0

    x = np.empty(n_in)
    h = np.empty(n_hid)

    for i in range(start, end):
        # Always track MAE/MFE + cum_mae + max_hold (even on non-decision bars)
        if position != 0:
            upnl = (mid[i] - entry_price) / pip * position
            if upnl < mae_s:
                mae_s = upnl
            if upnl > mfe_s:
                mfe_s = upnl
            cum_mae += -mae_s if mae_s < 0 else 0.0
        else:
            upnl = 0.0; mae_s = 0.0; mfe_s = 0.0

        # Max-hold force close — always active
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid[i] - entry_price) / pip * position
            total_pnl += pnl
            nt += 1
            if position > 0: nl += 1
            else: ns += 1
            position = 0

        # NN evaluation — gated by cadence
        if h1_stride > 1 and i % h1_stride != 0:
            continue  # skip NN eval on non-H1 bars

        # Build input vector
        for k in range(n_core):
            x[k] = core_inputs[k, i]
        ix = n_core
        if use_cand:
            x[ix] = cand_arr[i]
            ix += 1
        x[ix] = np.tanh(upnl / 10.0)  # UPnL

        # Hidden layer
        for k in range(n_hid):
            z = genes[b1_end - n_hid + k]
            w1_row = k * n_in
            for j in range(n_in):
                z += genes[w1_row + j] * x[j]
            aid_k = _decode_act(genes[b2_end + k])
            h[k] = _activate(z, aid_k)

        # Output + skip
        out = np.empty(N_OUT)
        for o in range(N_OUT):
            val = genes[wskip_end + o]
            w2_row = b1_end + o * n_hid
            for k in range(n_hid):
                val += genes[w2_row + k] * h[k]
            wskip_row = w2_end + o * n_in
            for j in range(n_in):
                val += genes[wskip_row + j] * x[j]
            out[o] = val

        ob = out[0]; os_ = out[1]; of = out[2]

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


_W = {}


def _winit(core, cand, n_core, use_cand, n_hid, n_in, mid, pip, spread, h1_stride):
    _W.update({"core": core, "cand": cand, "n_core": n_core, "use_cand": use_cand,
               "n_hid": n_hid, "n_in": n_in, "mid": mid, "pip": pip,
               "spread": spread, "h1_stride": h1_stride})


def _eval_one(genes):
    n = len(_W["mid"])
    tl = 0; ts = 0; tt = 0; tscore = 0.0
    chunk_sps = []; losing = 0.0
    chunk_trades_list = []
    for ci in range(N_CHUNKS):
        c_s = int(n * ci / N_CHUNKS)
        c_e = int(n * (ci + 1) / N_CHUNKS)
        nt, score, _p, _c, nl, ns = simulate(
            genes, _W["core"], _W["cand"], _W["n_core"], _W["use_cand"],
            _W["n_hid"], _W["n_in"], _W["mid"], _W["pip"], _W["spread"],
            MAX_HOLD, c_s, c_e, AMDDP_COEF, _W["h1_stride"])
        tl += nl; ts += ns; tt += nt; tscore += score
        days = (c_e - c_s) / BARS_PER_DAY
        sps = score / days if days > 0 else 0.0
        chunk_sps.append(sps)
        chunk_trades_list.append(nt)
        if sps < 0:
            losing += -sps
    total_days = n / BARS_PER_DAY
    base_sps = tscore / total_days if total_days > 0 else 0.0

    # For H1 cadence, expect fewer trades — relax min-trade gate proportionally
    chunk_days = (n / N_CHUNKS) / BARS_PER_DAY
    min_per_chunk = max(1, int(chunk_days / max(1, _W["h1_stride"] / 12)))
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
    ap.add_argument("--path", choices=["B", "C"], required=True)
    ap.add_argument("--candidate", required=True,
                    help="Slot candidate name. Pass '_none_' for core-only (Path B only).")
    ap.add_argument("--pair", default="EUR_JPY")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoother", default="kalman10")
    args = ap.parse_args()

    if args.path == "B":
        core_cols = PATH_B_CORE
        h1_stride = 1  # M5 cadence
    else:
        core_cols = PATH_C_CORE
        h1_stride = H1_STRIDE

    use_cand = args.candidate != "_none_"
    n_core = len(core_cols)
    n_in = n_core + (1 if use_cand else 0) + 1  # +1 for UPnL
    n_hid = 4
    n_params = (n_in * n_hid + n_hid + n_hid * N_OUT + n_in * N_OUT + N_OUT + n_hid + 1)

    parq = PROJECT / f"data/m5_ohlc/{args.pair}_M5_{args.smoother}_causal.parquet"
    df = pd.read_parquet(parq)
    print(f"[path={args.path}][{args.candidate}][{args.pair}] {len(df):,} bars, n_in={n_in}, h1_stride={h1_stride}", flush=True)

    # Verify cols
    missing = [c for c in core_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing core cols: {missing}")
    if use_cand and args.candidate not in df.columns:
        raise SystemExit(f"Missing candidate col: {args.candidate}")

    mid = df["close"].values.astype(np.float64)
    core_arr = np.stack([df[c].values.astype(np.float64) for c in core_cols])
    cand_arr = df[args.candidate].values.astype(np.float64) if use_cand else np.zeros(len(df))

    n_bars = len(df)
    split = int(n_bars * 0.7)
    mid_is, mid_oos = mid[:split], mid[split:]
    core_is, core_oos = core_arr[:, :split], core_arr[:, split:]
    cand_is, cand_oos = cand_arr[:split], cand_arr[split:]

    pip = PAIR_PIP[args.pair]; spread = PAIR_SPREAD[args.pair]

    # JIT warm
    dummy = np.zeros(n_params)
    simulate(dummy, core_is[:, :300], cand_is[:300], n_core, use_cand, n_hid, n_in,
             mid_is[:300], pip, spread, 50, 0, 300, AMDDP_COEF, h1_stride)

    np.random.seed(args.seed)
    x0 = np.zeros(n_params)
    b2_end = n_in * n_hid + n_hid + n_hid * N_OUT + n_in * N_OUT + N_OUT
    for k in range(n_hid):
        x0[b2_end + k] = np.random.uniform(0, 1)

    global _POOL
    _POOL = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_winit,
        initargs=(core_is, cand_is, n_core, use_cand, n_hid, n_in,
                  mid_is, pip, spread, h1_stride))

    es = cma.CMAEvolutionStrategy(x0, args.sigma, {
        "popsize": args.pop, "seed": args.seed, "verbose": -9,
        "maxiter": args.gens, "tolx": 1e-8,
    })

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
        if gen % 20 == 0:
            print(f"  gen {gen:3d} | fit {-best_score:+.3f}", flush=True)
        gen += 1
    elapsed = time.time() - t0

    # OOS eval
    is_nt, _, is_pnl, is_cm, is_nl, is_ns = simulate(
        best_genes, core_is, cand_is, n_core, use_cand, n_hid, n_in, mid_is,
        pip, spread, MAX_HOLD, 0, len(mid_is), AMDDP_COEF, h1_stride)
    oos_nt, _, oos_pnl, oos_cm, oos_nl, oos_ns = simulate(
        best_genes, core_oos, cand_oos, n_core, use_cand, n_hid, n_in, mid_oos,
        pip, spread, MAX_HOLD, 0, len(mid_oos), AMDDP_COEF, h1_stride)
    is_days = len(mid_is) / BARS_PER_DAY
    oos_days = len(mid_oos) / BARS_PER_DAY
    is_dir = min(is_nl, is_ns) / max(is_nt, 1)
    oos_dir = min(oos_nl, oos_ns) / max(oos_nt, 1)

    print(f"\n{'='*72}")
    print(f"  PATH={args.path} CAND={args.candidate} PAIR={args.pair} SEED={args.seed}")
    print(f"{'='*72}")
    print(f"  Winner: fit={-best_score:+.3f}")
    print(f"  IS : {is_nt}T  {is_pnl/is_days:+.2f} p/d  dir={is_dir:.2f}")
    print(f"  OOS: {oos_nt}T  {oos_pnl/oos_days:+.2f} p/d  dir={oos_dir:.2f}")
    print(f"  Elapsed: {elapsed:.0f}s")

    out = OUT_DIR / f"path{args.path}_{args.candidate}_{args.pair}_s{args.seed}.pkl"
    payload = {
        "path": args.path, "candidate": args.candidate, "pair": args.pair, "seed": args.seed,
        "core": core_cols, "n_params": int(n_params),
        "h1_stride": h1_stride,
        "fitness": float(-best_score),
        "genes": best_genes,
        "is": {"n_trades": int(is_nt), "pnl": float(is_pnl), "pd": float(is_pnl/is_days),
               "dir": float(is_dir), "cum_mae": float(is_cm)},
        "oos": {"n_trades": int(oos_nt), "pnl": float(oos_pnl), "pd": float(oos_pnl/oos_days),
                "dir": float(oos_dir), "cum_mae": float(oos_cm)},
        "elapsed_s": float(elapsed), "gens_run": int(gen),
    }
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    jout = OUT_DIR / f"path{args.path}_{args.candidate}_{args.pair}_s{args.seed}.json"
    payload_json = {k: v for k, v in payload.items() if k != "genes"}
    with open(jout, "w") as f:
        json.dump(payload_json, f, indent=2, default=float)
    print(f"Saved: {out.name}")


if __name__ == "__main__":
    main()
