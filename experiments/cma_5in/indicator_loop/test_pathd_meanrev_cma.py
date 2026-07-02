"""Path D1: Mean-reversion CMA runner.

Philosophy shift from A/B/C: NN is a FILTER, not a signal generator.

Entry gate (explicit, not NN):
  - LONG when range_pos_30 < 0.10 AND flat
  - SHORT when range_pos_30 > 0.90 AND flat
  - NN sees setup, outputs confidence ∈ [-1,+1], take if |conf| > θ

Exit rules (explicit, not NN):
  - mean revert: range_pos_30 in [0.30, 0.70]
  - OR max_hold bars
  - OR emergency SL (20 pips)

Inputs (6): range_pos_30, atr_ratio, trending, high_vol, vol_regime, upnl
Topology: 6 → 3 hidden + skip → 1 output (confidence)
Activation gene per hidden node {tanh, sin, gauss}, θ latch gene.
Cadence: H1 (stride=12, same as Path C).
Fitness: amddp1 with hard gates.
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
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

N_IN = 6
N_OUT = 1  # single confidence output
N_HID = 3
N_ACTS = 3  # tanh, sin, gauss
BARS_PER_DAY = 288.0
MAX_HOLD = 144   # 12 H1 bars = 12h
MIN_DIR = 0.15
N_CHUNKS = 3
AMDDP_COEF = 0.01
H1_STRIDE = 12
EMERGENCY_SL = 20.0  # pips

# Entry gate thresholds (on range_pos_30)
RP_LONG_ENTRY = 0.10
RP_SHORT_ENTRY = 0.90
# Exit zone (mean revert)
RP_EXIT_LO = 0.30
RP_EXIT_HI = 0.70

PAIR_PIP = {"EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
            "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
            "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
            "NZD_USD": 0.0001, "EUR_GBP": 0.0001}
PAIR_SPREAD = {"EUR_JPY": 2.3, "USD_JPY": 1.7, "GBP_JPY": 3.3, "AUD_JPY": 2.1,
               "CAD_JPY": 2.3, "CHF_JPY": 3.5, "NZD_JPY": 2.7,
               "EUR_USD": 1.6, "GBP_USD": 1.9, "AUD_USD": 1.3,
               "NZD_USD": 1.5, "EUR_GBP": 1.4}

N_PARAMS = (N_IN * N_HID + N_HID + N_HID * N_OUT + N_IN * N_OUT + N_OUT + N_HID + 1)


@njit(cache=True, inline="always")
def _activate(z, aid):
    if aid == 0: return np.tanh(z)
    elif aid == 1: return np.sin(z)
    else: return np.exp(-z * z)


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
def simulate(genes, rp, atrr, trending, hv, vol_reg, mid,
             pip, spread_pips, max_hold, chunk_start, chunk_end,
             amddp_coef, h1_stride):
    """Mean-reversion simulator with explicit gates. NN only for take/skip confidence."""
    n = len(mid)
    start = max(chunk_start + 20, 20)
    end = min(chunk_end, n - 1)
    if end <= start:
        return 0, 0.0, 0.0, 0.0, 0, 0

    # Gene offsets
    w1_end = N_IN * N_HID                 # 18
    b1_end = w1_end + N_HID               # 21
    w2_end = b1_end + N_HID * N_OUT       # 24
    wskip_end = w2_end + N_IN * N_OUT     # 30
    b2_end = wskip_end + N_OUT            # 31
    # act_genes at [31:34], theta at [34]

    theta = _decode_theta(genes[-1])

    nt = 0; nl = 0; ns = 0
    total_pnl = 0.0; cum_mae = 0.0
    position = 0
    entry_price = 0.0; entry_bar = 0
    upnl = 0.0; mae_s = 0.0; mfe_s = 0.0

    x = np.empty(N_IN)
    h = np.empty(N_HID)

    for i in range(start, end):
        # Always track MAE/MFE
        if position != 0:
            upnl = (mid[i] - entry_price) / pip * position
            if upnl < mae_s:
                mae_s = upnl
            if upnl > mfe_s:
                mfe_s = upnl
            cum_mae += -mae_s if mae_s < 0 else 0.0
        else:
            upnl = 0.0; mae_s = 0.0; mfe_s = 0.0

        # ── Exits: rule-based, always active ──
        if position != 0:
            # Max hold
            if (i - entry_bar) >= max_hold:
                pnl = (mid[i] - entry_price) / pip * position
                total_pnl += pnl; nt += 1
                if position > 0: nl += 1
                else: ns += 1
                position = 0
                continue
            # Emergency SL
            if upnl <= -EMERGENCY_SL:
                pnl = upnl
                total_pnl += pnl; nt += 1
                if position > 0: nl += 1
                else: ns += 1
                position = 0
                continue
            # Mean-revert exit (price returned to normal zone)
            if RP_EXIT_LO <= rp[i] <= RP_EXIT_HI:
                pnl = (mid[i] - entry_price) / pip * position
                total_pnl += pnl; nt += 1
                if position > 0: nl += 1
                else: ns += 1
                position = 0
                continue

        # ── Entry gates: only check on H1 boundaries + at extremes ──
        if position != 0:
            continue
        if h1_stride > 1 and i % h1_stride != 0:
            continue

        # Is this an entry setup?
        long_setup = rp[i] < RP_LONG_ENTRY
        short_setup = rp[i] > RP_SHORT_ENTRY
        if not (long_setup or short_setup):
            continue

        # Build NN input
        x[0] = rp[i]
        x[1] = atrr[i]
        x[2] = trending[i]
        x[3] = hv[i]
        x[4] = vol_reg[i]
        x[5] = np.tanh(upnl / 10.0)  # always 0 here, for consistency

        # Hidden layer
        for k in range(N_HID):
            z = genes[b1_end - N_HID + k]
            w1_row = k * N_IN
            for j in range(N_IN):
                z += genes[w1_row + j] * x[j]
            aid_k = _decode_act(genes[b2_end + k])
            h[k] = _activate(z, aid_k)

        # Output (1 scalar confidence) + skip
        val = genes[wskip_end]  # b2[0]
        w2_row = b1_end
        for k in range(N_HID):
            val += genes[w2_row + k] * h[k]
        wskip_row = w2_end
        for j in range(N_IN):
            val += genes[wskip_row + j] * x[j]
        confidence = val

        # Take if |conf| above θ
        if abs(confidence) <= theta:
            continue

        # Open trade — direction set by gate, not NN sign
        if long_setup:
            position = 1
            entry_price = mid[i] + spread_pips * pip
        else:  # short_setup
            position = -1
            entry_price = mid[i] - spread_pips * pip
        entry_bar = i
        mae_s = -spread_pips; mfe_s = 0.0

    total_score = total_pnl - amddp_coef * cum_mae
    return nt, total_score, total_pnl, cum_mae, nl, ns


_W = {}


def _winit(rp, atrr, trending, hv, vol_reg, mid, pip, spread, h1_stride):
    _W.update({"rp": rp, "atrr": atrr, "trending": trending, "hv": hv,
               "vol_reg": vol_reg, "mid": mid, "pip": pip, "spread": spread,
               "h1_stride": h1_stride})


def _eval_one(genes):
    n = len(_W["mid"])
    tl = ts = tt = 0; tscore = 0.0
    chunk_sps = []; losing = 0.0; chunk_trades_list = []
    for ci in range(N_CHUNKS):
        c_s = int(n * ci / N_CHUNKS)
        c_e = int(n * (ci + 1) / N_CHUNKS)
        nt, score, _p, _c, nl, ns = simulate(
            genes, _W["rp"], _W["atrr"], _W["trending"], _W["hv"], _W["vol_reg"],
            _W["mid"], _W["pip"], _W["spread"], MAX_HOLD, c_s, c_e,
            AMDDP_COEF, _W["h1_stride"])
        tl += nl; ts += ns; tt += nt; tscore += score
        days = (c_e - c_s) / BARS_PER_DAY
        sps = score / days if days > 0 else 0.0
        chunk_sps.append(sps)
        chunk_trades_list.append(nt)
        if sps < 0: losing += -sps
    total_days = n / BARS_PER_DAY
    base_sps = tscore / total_days if total_days > 0 else 0.0

    # Mean-rev trades are rarer → relaxed trade minimum
    chunk_days = (n / N_CHUNKS) / BARS_PER_DAY
    min_per_chunk = max(1, int(chunk_days / 5))  # 1 trade per 5 days = very lenient
    min_chunk_trades = min(chunk_trades_list) if chunk_trades_list else 0
    trades_short = max(0, min_per_chunk - min_chunk_trades)
    dir_ratio = (min(tl, ts) / tt) if tt > 0 else 0.0
    dir_short = max(0.0, MIN_DIR - dir_ratio)

    if trades_short > 0 or dir_short > 0:
        return 500.0 + trades_short * 0.1 + dir_short * 100.0

    asym = (1.0 - 2.0 * dir_ratio) * 25.0
    all_prof = all(s > 0 for s in chunk_sps)
    if all_prof:
        score = min(chunk_sps) - asym
    else:
        score = base_sps - asym - losing * 2.0
    return -score


_POOL = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="EUR_JPY")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoother", default="kalman10")
    args = ap.parse_args()

    parq = PROJECT / f"data/m5_ohlc/{args.pair}_M5_{args.smoother}_causal.parquet"
    df = pd.read_parquet(parq)

    needed = ["range_pos_30", "atr_ratio", "trending", "high_vol", "vol_regime"]
    for c in needed:
        if c not in df.columns:
            raise SystemExit(f"Missing col: {c}")

    mid = df["close"].values.astype(np.float64)
    rp = df["range_pos_30"].values.astype(np.float64)
    atrr = df["atr_ratio"].values.astype(np.float64)
    trending = df["trending"].values.astype(np.float64)
    hv = df["high_vol"].values.astype(np.float64)
    vol_reg = df["vol_regime"].values.astype(np.float64)

    n_bars = len(df)
    split = int(n_bars * 0.7)
    print(f"[pathD][{args.pair}] {n_bars:,} bars, IS={split:,} OOS={n_bars-split:,}, h1_stride={H1_STRIDE}", flush=True)

    pip = PAIR_PIP[args.pair]; spread = PAIR_SPREAD[args.pair]

    dummy = np.zeros(N_PARAMS)
    simulate(dummy, rp[:300], atrr[:300], trending[:300], hv[:300], vol_reg[:300],
             mid[:300], pip, spread, 50, 0, 300, AMDDP_COEF, H1_STRIDE)

    np.random.seed(args.seed)
    x0 = np.zeros(N_PARAMS)
    b2_end = N_IN * N_HID + N_HID + N_HID * N_OUT + N_IN * N_OUT + N_OUT
    for k in range(N_HID):
        x0[b2_end + k] = np.random.uniform(0, 1)

    global _POOL
    _POOL = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_winit,
        initargs=(rp[:split], atrr[:split], trending[:split], hv[:split],
                  vol_reg[:split], mid[:split], pip, spread, H1_STRIDE))

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
            best_score = gb; best_genes = solutions[gi].copy()
        if gen % 20 == 0:
            print(f"  gen {gen:3d} | fit {-best_score:+.3f}", flush=True)
        gen += 1
    elapsed = time.time() - t0

    is_nt, _, is_pnl, is_cm, is_nl, is_ns = simulate(
        best_genes, rp[:split], atrr[:split], trending[:split], hv[:split], vol_reg[:split],
        mid[:split], pip, spread, MAX_HOLD, 0, split, AMDDP_COEF, H1_STRIDE)
    oos_nt, _, oos_pnl, oos_cm, oos_nl, oos_ns = simulate(
        best_genes, rp[split:], atrr[split:], trending[split:], hv[split:], vol_reg[split:],
        mid[split:], pip, spread, MAX_HOLD, 0, n_bars-split, AMDDP_COEF, H1_STRIDE)
    is_days = split / BARS_PER_DAY
    oos_days = (n_bars - split) / BARS_PER_DAY
    is_dir = min(is_nl, is_ns) / max(is_nt, 1)
    oos_dir = min(oos_nl, oos_ns) / max(oos_nt, 1)

    print(f"\n{'='*72}")
    print(f"  PATH D1 MEAN-REV  pair={args.pair}  seed={args.seed}")
    print(f"{'='*72}")
    print(f"  fit={-best_score:+.3f}")
    print(f"  IS : {is_nt}T  {is_pnl/is_days:+.2f} p/d  L/S={is_nl}/{is_ns}  dir={is_dir:.2f}")
    print(f"  OOS: {oos_nt}T  {oos_pnl/oos_days:+.2f} p/d  L/S={oos_nl}/{oos_ns}  dir={oos_dir:.2f}")
    print(f"  Elapsed: {elapsed:.0f}s")

    out = OUT_DIR / f"pathD_{args.pair}_s{args.seed}.json"
    payload = {
        "path": "D1", "pair": args.pair, "seed": args.seed,
        "arch": "meanrev_gate+NN_filter", "n_params": int(N_PARAMS),
        "inputs": ["range_pos_30","atr_ratio","trending","high_vol","vol_regime","upnl"],
        "h1_stride": H1_STRIDE, "fitness": float(-best_score),
        "is": {"n_trades": int(is_nt), "pd": float(is_pnl/is_days), "dir": float(is_dir)},
        "oos": {"n_trades": int(oos_nt), "pd": float(oos_pnl/oos_days), "dir": float(oos_dir),
                "pnl": float(oos_pnl), "cum_mae": float(oos_cm)},
        "elapsed_s": float(elapsed), "gens_run": int(gen),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    pkl = OUT_DIR / f"pathD_{args.pair}_s{args.seed}.pkl"
    payload["genes"] = best_genes
    with open(pkl, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved: {out.name}")


if __name__ == "__main__":
    main()
