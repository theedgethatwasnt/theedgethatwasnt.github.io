"""Path D2: Mean-reversion with TIGHTER entry gate (dual-extreme + counter-trend).

Same architecture as D1, different entry rule:
  LONG  only when range_pos_30 < 0.10 AND h1_slope < 0  (stretched DOWN against uptrend)
  SHORT only when range_pos_30 > 0.90 AND h1_slope > 0  (stretched UP against downtrend)

The extra h1_slope filter says: only fade moves that are *against* the larger trend.
Fewer entries but each has stronger mean-reversion prior.

Exits: same (mean-revert zone, max_hold, SL).
NN: same 6→3+skip→1 filter. Added h1_slope as 7th input since it's now part of gate.
Cadence: H1 (stride=12).
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

N_IN = 7  # +h1_slope
N_OUT = 1
N_HID = 3
N_ACTS = 3
BARS_PER_DAY = 288.0
MAX_HOLD = 144
MIN_DIR = 0.15
N_CHUNKS = 3
AMDDP_COEF = 0.01
H1_STRIDE = 12
EMERGENCY_SL = 20.0
RP_LONG_ENTRY = 0.10
RP_SHORT_ENTRY = 0.90
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
def simulate(genes, rp, atrr, trending, hv, vol_reg, h1s, mid,
             pip, spread_pips, max_hold, chunk_start, chunk_end,
             amddp_coef, h1_stride):
    n = len(mid)
    start = max(chunk_start + 20, 20)
    end = min(chunk_end, n - 1)
    if end <= start:
        return 0, 0.0, 0.0, 0.0, 0, 0

    w1_end = N_IN * N_HID
    b1_end = w1_end + N_HID
    w2_end = b1_end + N_HID * N_OUT
    wskip_end = w2_end + N_IN * N_OUT
    b2_end = wskip_end + N_OUT

    theta = _decode_theta(genes[-1])

    nt = 0; nl = 0; ns = 0
    total_pnl = 0.0; cum_mae = 0.0
    position = 0
    entry_price = 0.0; entry_bar = 0
    upnl = 0.0; mae_s = 0.0; mfe_s = 0.0

    x = np.empty(N_IN)
    h = np.empty(N_HID)

    for i in range(start, end):
        if position != 0:
            upnl = (mid[i] - entry_price) / pip * position
            if upnl < mae_s: mae_s = upnl
            if upnl > mfe_s: mfe_s = upnl
            cum_mae += -mae_s if mae_s < 0 else 0.0
        else:
            upnl = 0.0; mae_s = 0.0; mfe_s = 0.0

        # Exits
        if position != 0:
            if (i - entry_bar) >= max_hold:
                pnl = (mid[i] - entry_price) / pip * position
                total_pnl += pnl; nt += 1
                if position > 0: nl += 1
                else: ns += 1
                position = 0
                continue
            if upnl <= -EMERGENCY_SL:
                total_pnl += upnl; nt += 1
                if position > 0: nl += 1
                else: ns += 1
                position = 0
                continue
            if RP_EXIT_LO <= rp[i] <= RP_EXIT_HI:
                pnl = (mid[i] - entry_price) / pip * position
                total_pnl += pnl; nt += 1
                if position > 0: nl += 1
                else: ns += 1
                position = 0
                continue

        if position != 0:
            continue
        if h1_stride > 1 and i % h1_stride != 0:
            continue

        # D2 gate: dual-extreme (range extreme AND counter-trend)
        long_setup = (rp[i] < RP_LONG_ENTRY) and (h1s[i] < 0)
        short_setup = (rp[i] > RP_SHORT_ENTRY) and (h1s[i] > 0)
        if not (long_setup or short_setup):
            continue

        x[0] = rp[i]
        x[1] = atrr[i]
        x[2] = trending[i]
        x[3] = hv[i]
        x[4] = vol_reg[i]
        x[5] = h1s[i]
        x[6] = np.tanh(upnl / 10.0)

        for k in range(N_HID):
            z = genes[b1_end - N_HID + k]
            w1_row = k * N_IN
            for j in range(N_IN):
                z += genes[w1_row + j] * x[j]
            aid_k = _decode_act(genes[b2_end + k])
            h[k] = _activate(z, aid_k)

        val = genes[wskip_end]
        w2_row = b1_end
        for k in range(N_HID):
            val += genes[w2_row + k] * h[k]
        wskip_row = w2_end
        for j in range(N_IN):
            val += genes[wskip_row + j] * x[j]
        confidence = val

        if abs(confidence) <= theta:
            continue

        if long_setup:
            position = 1
            entry_price = mid[i] + spread_pips * pip
        else:
            position = -1
            entry_price = mid[i] - spread_pips * pip
        entry_bar = i
        mae_s = -spread_pips; mfe_s = 0.0

    total_score = total_pnl - amddp_coef * cum_mae
    return nt, total_score, total_pnl, cum_mae, nl, ns


_W = {}


def _winit(rp, atrr, trending, hv, vol_reg, h1s, mid, pip, spread, h1_stride):
    _W.update({"rp": rp, "atrr": atrr, "trending": trending, "hv": hv,
               "vol_reg": vol_reg, "h1s": h1s, "mid": mid, "pip": pip,
               "spread": spread, "h1_stride": h1_stride})


def _eval_one(genes):
    n = len(_W["mid"])
    tl = ts = tt = 0; tscore = 0.0
    chunk_sps = []; losing = 0.0; chunk_trades_list = []
    for ci in range(N_CHUNKS):
        c_s = int(n * ci / N_CHUNKS)
        c_e = int(n * (ci + 1) / N_CHUNKS)
        nt, score, _p, _c, nl, ns = simulate(
            genes, _W["rp"], _W["atrr"], _W["trending"], _W["hv"], _W["vol_reg"],
            _W["h1s"], _W["mid"], _W["pip"], _W["spread"], MAX_HOLD, c_s, c_e,
            AMDDP_COEF, _W["h1_stride"])
        tl += nl; ts += ns; tt += nt; tscore += score
        days = (c_e - c_s) / BARS_PER_DAY
        sps = score / days if days > 0 else 0.0
        chunk_sps.append(sps)
        chunk_trades_list.append(nt)
        if sps < 0: losing += -sps
    total_days = n / BARS_PER_DAY
    base_sps = tscore / total_days if total_days > 0 else 0.0

    chunk_days = (n / N_CHUNKS) / BARS_PER_DAY
    min_per_chunk = max(1, int(chunk_days / 7))  # D2 has even tighter gate, more lenient
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
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoother", default="kalman10")
    args = ap.parse_args()

    parq = PROJECT / f"data/m5_ohlc/{args.pair}_M5_{args.smoother}_causal.parquet"
    df = pd.read_parquet(parq)
    for c in ["range_pos_30","atr_ratio","trending","high_vol","vol_regime","h1_slope"]:
        if c not in df.columns: raise SystemExit(f"Missing {c}")

    mid = df["close"].values.astype(np.float64)
    rp = df["range_pos_30"].values.astype(np.float64)
    atrr = df["atr_ratio"].values.astype(np.float64)
    trending = df["trending"].values.astype(np.float64)
    hv = df["high_vol"].values.astype(np.float64)
    vol_reg = df["vol_regime"].values.astype(np.float64)
    h1s = df["h1_slope"].values.astype(np.float64)

    n_bars = len(df); split = int(n_bars * 0.7)
    pip = PAIR_PIP[args.pair]; spread = PAIR_SPREAD[args.pair]

    dummy = np.zeros(N_PARAMS)
    simulate(dummy, rp[:300], atrr[:300], trending[:300], hv[:300], vol_reg[:300],
             h1s[:300], mid[:300], pip, spread, 50, 0, 300, AMDDP_COEF, H1_STRIDE)

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
                  vol_reg[:split], h1s[:split], mid[:split], pip, spread, H1_STRIDE))

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
        gen += 1
    elapsed = time.time() - t0

    is_nt, _, is_pnl, is_cm, is_nl, is_ns = simulate(
        best_genes, rp[:split], atrr[:split], trending[:split], hv[:split], vol_reg[:split],
        h1s[:split], mid[:split], pip, spread, MAX_HOLD, 0, split, AMDDP_COEF, H1_STRIDE)
    oos_nt, _, oos_pnl, oos_cm, oos_nl, oos_ns = simulate(
        best_genes, rp[split:], atrr[split:], trending[split:], hv[split:], vol_reg[split:],
        h1s[split:], mid[split:], pip, spread, MAX_HOLD, 0, n_bars-split, AMDDP_COEF, H1_STRIDE)
    is_days = split / BARS_PER_DAY
    oos_days = (n_bars - split) / BARS_PER_DAY
    is_dir = min(is_nl, is_ns) / max(is_nt, 1)
    oos_dir = min(oos_nl, oos_ns) / max(oos_nt, 1)

    print(f"PATH D2  {args.pair} s{args.seed}  fit={-best_score:+.2f}  IS {is_nt}T {is_pnl/is_days:+.2f}p/d  OOS {oos_nt}T {oos_pnl/oos_days:+.2f}p/d  dir={oos_dir:.2f}  ({elapsed:.0f}s)", flush=True)

    out = OUT_DIR / f"pathD2_{args.pair}_s{args.seed}.json"
    payload = {
        "path": "D2", "pair": args.pair, "seed": args.seed,
        "arch": "dual_extreme+counter_trend gate + NN filter", "n_params": int(N_PARAMS),
        "h1_stride": H1_STRIDE, "fitness": float(-best_score),
        "is": {"n_trades": int(is_nt), "pd": float(is_pnl/is_days), "dir": float(is_dir)},
        "oos": {"n_trades": int(oos_nt), "pd": float(oos_pnl/oos_days), "dir": float(oos_dir),
                "pnl": float(oos_pnl), "cum_mae": float(oos_cm)},
        "elapsed_s": float(elapsed), "gens_run": int(gen),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=float)


if __name__ == "__main__":
    main()
