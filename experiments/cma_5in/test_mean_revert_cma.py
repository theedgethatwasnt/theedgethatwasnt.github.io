"""Mean reversion CMA-ES — fixed 4→4→3 topology, per-node activation as gene.

Same feature as NEAT version: percentile-scaled z of (close - SMA10), window=10000.

Genome (39 floats):
  [0:16]  W1 (4×4)
  [16:20] b1 (4)
  [20:24] activation gene per hidden node (bucketed to sin/tanh/gauss)
  [24:36] W2 (3×4, 3 outputs × 4 hidden)
  [36:39] b_out (3)

Output: linear, argmax → BUY/SELL/FLATTEN.
Activations: {sin, tanh, gauss} via `int((gene - floor(gene)) * 3) % 3`.
"""
import argparse, math, pickle, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from numba import njit
import cma

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from research.experiments.cma_5in.test_mean_revert_neat import compute_zscore_residual

POP = 24
GENS = 300
MAX_HOLD = 200
BARS_PER_DAY = 288.0
OUT_DIR = PROJECT / "research/experiments/cma_5in/results"
OUT_DIR.mkdir(exist_ok=True)

PAIR_PIP = {"EUR_JPY":0.01,"USD_JPY":0.01,"GBP_JPY":0.01,"AUD_JPY":0.01,
            "CAD_JPY":0.01,"CHF_JPY":0.01,"NZD_JPY":0.01,
            "EUR_USD":0.0001,"GBP_USD":0.0001,"AUD_USD":0.0001,
            "NZD_USD":0.0001,"EUR_GBP":0.0001}
PAIR_SPREAD = {"EUR_JPY":2.3,"USD_JPY":1.7,"GBP_JPY":3.3,"AUD_JPY":2.1,
               "CAD_JPY":2.3,"CHF_JPY":3.5,"NZD_JPY":2.7,
               "EUR_USD":1.6,"GBP_USD":1.9,"AUD_USD":1.3,
               "NZD_USD":1.5,"EUR_GBP":1.4}

N_IN = 4
N_H = 4
N_OUT = 3
N_PARAMS = N_IN * N_H + N_H + N_H + N_H * N_OUT + N_OUT  # 16+4+4+12+3 = 39


@njit(cache=True)
def decode_act(gene):
    """Map continuous gene → activation id in [0, 2] for {sin, tanh, gauss}."""
    g = gene - math.floor(gene)
    aid = int(g * 3)
    if aid < 0: aid = 0
    if aid >= 3: aid = 2
    return aid


@njit(cache=True)
def apply_act(z, aid):
    if aid == 0: return np.sin(z)
    elif aid == 1: return np.tanh(z)
    else: return np.exp(-z * z)


@njit(cache=True)
def cma_mr_simulate(market, mid, pip, spread_pips, max_hold, weights,
                    chunk_start, chunk_end):
    n = market.shape[1]
    start = max(chunk_start + 120, 120)
    end = min(chunk_end, n - 1)
    if end <= start:
        return 0, 0.0, 0, 0

    n_cap = end - start + 1
    pnls = np.zeros(n_cap)
    nt = 0; nl = 0; ns = 0
    position = 0; entry_price = 0.0; entry_bar = 0
    mae_pips = 0.0; mfe_pips = 0.0

    # Layout
    w1_e = N_IN * N_H       # 16
    b1_e = w1_e + N_H       # 20
    act_e = b1_e + N_H      # 24
    w2_e = act_e + N_H * N_OUT  # 36
    # b_out_e = w2_e + N_OUT = 39

    # Pre-decode activation ids (cheap, but keep inside loop for clarity)
    a0 = decode_act(weights[b1_e + 0])
    a1 = decode_act(weights[b1_e + 1])
    a2 = decode_act(weights[b1_e + 2])
    a3 = decode_act(weights[b1_e + 3])

    h = np.zeros(N_H)
    inp = np.zeros(N_IN)

    for i in range(start, end):
        if position != 0:
            upnl_pips = (mid[i] - entry_price) / pip * position
            adv = -upnl_pips
            if adv > mae_pips: mae_pips = adv
            if upnl_pips > mfe_pips: mfe_pips = upnl_pips
        else:
            upnl_pips = 0.0; mae_pips = 0.0; mfe_pips = 0.0

        inp[0] = market[0, i]
        inp[1] = np.tanh(upnl_pips / 20.0)
        inp[2] = np.tanh(mae_pips / 20.0)
        inp[3] = np.tanh(mfe_pips / 20.0)

        # L1
        for j in range(N_H):
            z = weights[w1_e + j]  # b1
            for k in range(N_IN):
                z += weights[j * N_IN + k] * inp[k]
            if j == 0: h[j] = apply_act(z, a0)
            elif j == 1: h[j] = apply_act(z, a1)
            elif j == 2: h[j] = apply_act(z, a2)
            else: h[j] = apply_act(z, a3)

        # Output
        ob = weights[w2_e + 0]
        os_ = weights[w2_e + 1]
        of = weights[w2_e + 2]
        for k in range(N_H):
            ob += weights[act_e + 0 * N_H + k] * h[k]
            os_ += weights[act_e + 1 * N_H + k] * h[k]
            of += weights[act_e + 2 * N_H + k] * h[k]

        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid[i] - entry_price) / pip * position
            pnls[nt] = pnl; nt += 1
            if position > 0: nl += 1
            else: ns += 1
            position = 0

        if position == 0:
            if ob > os_ and ob > of:
                position = 1; entry_price = mid[i] + spread_pips * pip
                entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0
            elif os_ > ob and os_ > of:
                position = -1; entry_price = mid[i] - spread_pips * pip
                entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0
        else:
            close_now = False; new_pos = 0
            if of > ob and of > os_: close_now = True
            elif position == 1 and os_ > ob and os_ > of: close_now = True; new_pos = -1
            elif position == -1 and ob > os_ and ob > of: close_now = True; new_pos = 1
            if close_now:
                pnl = (mid[i] - entry_price) / pip * position
                pnls[nt] = pnl; nt += 1
                if position > 0: nl += 1
                else: ns += 1
                position = new_pos
                if new_pos != 0:
                    if new_pos == 1: entry_price = mid[i] + spread_pips * pip
                    else: entry_price = mid[i] - spread_pips * pip
                    entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0

    total = 0.0
    for k in range(nt): total += pnls[k]
    return nt, total, nl, ns


def cma_fitness(weights, market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
    n_bars = len(mid)
    tl = 0; ts = 0; tt = 0; tp = 0.0
    chunk_pps = []; losing = 0.0
    for ci in range(n_chunks):
        c_s = int(n_bars * ci / n_chunks)
        c_e = int(n_bars * (ci + 1) / n_chunks)
        nt, pnl, nl, ns = cma_mr_simulate(market, mid, pip, spread, max_hold, weights, c_s, c_e)
        tl += nl; ts += ns; tt += nt; tp += pnl
        days = (c_e - c_s) / bpd
        pps = pnl / days if days > 0 else 0
        chunk_pps.append(pps)
        if pps < 0: losing += -pps
    total_days = n_bars / bpd
    base_pps = tp / total_days if total_days > 0 else 0
    if tt == 0:
        return 500.0 - base_pps
    dir_ratio = min(tl, ts) / tt
    asym = (1.0 - 2.0 * dir_ratio) * 50.0
    act = max(0.0, 30.0 - tt) * 2.0
    all_prof = all(p > 0 for p in chunk_pps)
    if all_prof and dir_ratio >= min_dir:
        return -(min(chunk_pps) - asym)
    return -(base_pps - asym - act - losing * 2.0)


def cma_passes_gates(weights, market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
    n_bars = len(mid)
    tl = 0; ts = 0; tt = 0; chunk_pps = []
    for ci in range(n_chunks):
        c_s = int(n_bars * ci / n_chunks)
        c_e = int(n_bars * (ci + 1) / n_chunks)
        nt, pnl, nl, ns = cma_mr_simulate(market, mid, pip, spread, max_hold, weights, c_s, c_e)
        tl += nl; ts += ns; tt += nt
        days = (c_e - c_s) / bpd
        min_tr = max(20, int(days * 0.5))
        if nt < min_tr or pnl <= 0:
            return False, None
        chunk_pps.append(pnl / days)
    if tt < 30: return False, None
    if min(tl, ts) / tt < min_dir: return False, None
    return True, min(chunk_pps)


_W = {}
def _winit(market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
    _W['market']=market; _W['mid']=mid; _W['pip']=pip; _W['spread']=spread
    _W['max_hold']=max_hold; _W['n_chunks']=n_chunks; _W['min_dir']=min_dir; _W['bpd']=bpd

def _wfit(vec):
    return cma_fitness(vec, _W['market'], _W['mid'], _W['pip'], _W['spread'],
                       _W['max_hold'], _W['n_chunks'], _W['min_dir'], _W['bpd'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="EUR_JPY")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]

    df = pd.read_parquet(PROJECT / f"data/m5_ohlc/{pair}_M5.parquet")
    mid = df['close'].values.astype(np.float64)
    n = len(mid)
    print(f"Loaded {n:,} M5 bars of {pair}", flush=True)

    # Feature
    t0 = time.time()
    z_raw = compute_zscore_residual(mid, n, 10, 10000)
    pctile = pd.Series(z_raw).rolling(10000, min_periods=10000).rank(pct=True).fillna(0.5).values
    z_feat = (2.0 * (pctile - 0.5)).astype(np.float64)
    print(f"Feature: {time.time()-t0:.1f}s, std={z_feat[10500:].std():.3f}", flush=True)

    market = np.stack([z_feat], axis=0)
    split = int(n * 0.7)
    m_is = market[:, :split].copy(); mid_is = mid[:split].copy()
    m_oos = market[:, split:].copy(); mid_oos = mid[split:].copy()
    print(f"IS: {split:,}d ({split/BARS_PER_DAY:.0f}d), OOS: {n-split:,} ({(n-split)/BARS_PER_DAY:.0f}d)", flush=True)

    # JIT warm
    warm = np.zeros(N_PARAMS)
    cma_mr_simulate(m_is[:, :300], mid_is[:300], pip, spread, 50, warm, 0, 300)

    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_winit,
        initargs=(m_is, mid_is, pip, spread, MAX_HOLD, 3, 0.15, BARS_PER_DAY))

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    x0 = rng.randn(N_PARAMS) * 0.3
    # Init activation genes uniformly in [0,1)
    x0[20:24] = rng.uniform(0.0, 1.0, 4)

    es = cma.CMAEvolutionStrategy(x0, 0.5,
        {'popsize': POP, 'seed': args.seed, 'verbose': -9, 'maxiter': GENS})

    best_fit = 1e18; best_vec = None
    best_valid_pps = None; best_valid_vec = None
    t0 = time.time()
    gen = 0
    print(f"CMA-ES {N_PARAMS} params, pop {POP}, {GENS} gens...", flush=True)
    while not es.stop():
        c_ = es.ask()
        f_ = list(pool.map(_wfit, c_))
        es.tell(c_, f_)
        gm = min(f_)
        if gm < best_fit:
            best_fit = gm; best_vec = np.array(c_[f_.index(gm)])
        ok, mps = cma_passes_gates(best_vec, m_is, mid_is, pip, spread, MAX_HOLD, 3, 0.15, BARS_PER_DAY)
        if ok and (best_valid_pps is None or mps > best_valid_pps):
            best_valid_pps = mps; best_valid_vec = np.array(best_vec)
        gen += 1
        if gen % 25 == 0:
            print(f"  Gen {gen}: fit={best_fit:.2f} valid={best_valid_pps} t={time.time()-t0:.0f}s", flush=True)
        if gen >= GENS: break

    pool.shutdown(wait=False)
    final = best_valid_vec if best_valid_vec is not None else best_vec

    # OOS + IS
    is_nt, is_pnl, is_nl, is_ns = cma_mr_simulate(m_is, mid_is, pip, spread, MAX_HOLD, final, 0, len(mid_is))
    oos_nt, oos_pnl, oos_nl, oos_ns = cma_mr_simulate(m_oos, mid_oos, pip, spread, MAX_HOLD, final, 0, len(mid_oos))

    is_days = len(mid_is) / BARS_PER_DAY
    oos_days = len(mid_oos) / BARS_PER_DAY
    is_dir = min(is_nl, is_ns) / max(is_nt, 1)
    oos_dir = min(oos_nl, oos_ns) / max(oos_nt, 1)

    # Decode chosen activations
    acts_chosen = [['sin','tanh','gauss'][decode_act(final[20+j])] for j in range(4)]

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  MEAN REVERT CMA-ES: {pair}")
    print(f"{'='*65}")
    print(f"  Activations (hidden): {acts_chosen}")
    print(f"  IS:  {is_nt}T L/S={is_nl}/{is_ns} {is_pnl:+.1f}p ({is_pnl/is_days:+.2f} p/d dir={is_dir:.2f})")
    print(f"  OOS: {oos_nt}T L/S={oos_nl}/{oos_ns} {oos_pnl:+.1f}p ({oos_pnl/oos_days:+.2f} p/d dir={oos_dir:.2f})")
    print(f"  Hard gates: {'PASS' if best_valid_pps else 'FAIL'}")
    print(f"  Elapsed: {elapsed:.0f}s")

    save = {
        "pair": pair, "seed": args.seed, "weights": final,
        "activations": acts_chosen,
        "is": {"n_trades": is_nt, "total_pnl": is_pnl, "pips_per_day": is_pnl/is_days,
               "n_long": is_nl, "n_short": is_ns, "dir_ratio": is_dir},
        "oos": {"n_trades": oos_nt, "total_pnl": oos_pnl, "pips_per_day": oos_pnl/oos_days,
                "n_long": oos_nl, "n_short": oos_ns, "dir_ratio": oos_dir},
        "hard_gates": best_valid_pps is not None,
        "elapsed_s": elapsed,
    }
    path = OUT_DIR / f"mean_revert_cma_{pair}_s{args.seed}.pkl"
    with open(path, "wb") as f:
        pickle.dump(save, f)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
