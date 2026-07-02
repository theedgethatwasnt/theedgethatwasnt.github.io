"""Minimal CMA-ES baseline — 2023 oanda_neat input set, CMA optimizer.

Identical feature set to test_minimal_neat_baseline.py:
  [0] pips_delta (causal)
  [1] tl_slope (causal, 12-bar)
  [2] unrealized_pnl (position state)
  [3] holding_bars (position state)

Architecture: 4 → 4 → 3 FIXED, per-node activation as gene
  Genome (39 floats):
    [0:16]  W1 (4 hidden × 4 input)
    [16:20] b1 (4 biases)
    [20:24] act genes (4 per-node activations, bucketed to sin/tanh/gauss)
    [24:36] W2 (3 out × 4 hidden)
    [36:39] b_out (3)

Pre-training causality check on features (same as NEAT baseline).

Multi-seed × many gens for robustness.
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

from research.experiments.cma_5in.test_minimal_neat_baseline import (
    compute_pips_delta, compute_tl_slope, validate_features_causal,
    PAIR_PIP, PAIR_SPREAD, MAX_HOLD
)

OUT_DIR = PROJECT / "research/experiments/cma_5in/results"
OUT_DIR.mkdir(exist_ok=True)

POP = 24

N_IN = 4
N_H = 4
N_OUT = 3
N_PARAMS = N_IN * N_H + N_H + N_H + N_H * N_OUT + N_OUT  # 16+4+4+12+3 = 39


@njit(cache=True)
def decode_act(gene):
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
def cma_sim(market, mid, pip, spread_pips, max_hold, weights, chunk_start, chunk_end):
    """CMA-NN 4→4→3 simulator on minimal-NEAT inputs."""
    n = market.shape[1]
    start = max(chunk_start + 20, 20)
    end = min(chunk_end, n - 1)
    if end <= start: return 0, 0.0, 0, 0

    n_cap = end - start + 1
    pnls = np.zeros(n_cap)
    nt = 0; nl = 0; ns = 0
    position = 0; entry_price = 0.0; entry_bar = 0

    w1_e = N_IN * N_H       # 16
    b1_e = w1_e + N_H       # 20
    act_e = b1_e + N_H      # 24
    w2_e = act_e + N_H * N_OUT  # 36
    # b_out_e = 39

    a0 = decode_act(weights[b1_e + 0])
    a1 = decode_act(weights[b1_e + 1])
    a2 = decode_act(weights[b1_e + 2])
    a3 = decode_act(weights[b1_e + 3])

    h = np.zeros(N_H)
    inp = np.zeros(N_IN)

    for i in range(start, end):
        if position != 0:
            upnl_pips = (mid[i] - entry_price) / pip * position
            holding = i - entry_bar
        else:
            upnl_pips = 0.0; holding = 0

        inp[0] = market[0, i]                       # pips_delta
        inp[1] = market[1, i]                       # tl_slope
        inp[2] = np.tanh(upnl_pips / 20.0)          # unrealized
        inp[3] = np.tanh(holding / 20.0)            # holding_bars

        # L1
        for j in range(N_H):
            z = weights[w1_e + j]
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
                position = 1; entry_price = mid[i] + spread_pips * pip; entry_bar = i
            elif os_ > ob and os_ > of:
                position = -1; entry_price = mid[i] - spread_pips * pip; entry_bar = i
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
                    entry_bar = i

    total = 0.0
    for k in range(nt): total += pnls[k]
    return nt, total, nl, ns


def fitness(weights, market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
    n_bars = len(mid)
    tl = 0; ts = 0; tt = 0; tp = 0.0
    chunk_pps = []; losing = 0.0
    for ci in range(n_chunks):
        c_s = int(n_bars * ci / n_chunks)
        c_e = int(n_bars * (ci + 1) / n_chunks)
        nt, pnl, nl, ns = cma_sim(market, mid, pip, spread, max_hold, weights, c_s, c_e)
        tl += nl; ts += ns; tt += nt; tp += pnl
        days = (c_e - c_s) / bpd
        pps = pnl / days if days > 0 else 0
        chunk_pps.append(pps)
        if pps < 0: losing += -pps
    total_days = n_bars / bpd
    base_pps = tp / total_days if total_days > 0 else 0
    if tt == 0: return 500.0 - base_pps
    dir_ratio = min(tl, ts) / tt
    asym = (1.0 - 2.0 * dir_ratio) * 50.0
    act = max(0.0, 30.0 - tt) * 2.0
    all_prof = all(p > 0 for p in chunk_pps)
    if all_prof and dir_ratio >= min_dir:
        return -(min(chunk_pps) - asym)
    return -(base_pps - asym - act - losing * 2.0)


def passes_gates(weights, market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
    n_bars = len(mid)
    tl = 0; ts = 0; tt = 0; chunk_pps = []
    for ci in range(n_chunks):
        c_s = int(n_bars * ci / n_chunks)
        c_e = int(n_bars * (ci + 1) / n_chunks)
        nt, pnl, nl, ns = cma_sim(market, mid, pip, spread, max_hold, weights, c_s, c_e)
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
    _W.update({'market':market,'mid':mid,'pip':pip,'spread':spread,
               'max_hold':max_hold,'n_chunks':n_chunks,'min_dir':min_dir,'bpd':bpd})

def _wfit(vec):
    return fitness(vec, _W['market'], _W['mid'], _W['pip'], _W['spread'],
                   _W['max_hold'], _W['n_chunks'], _W['min_dir'], _W['bpd'])


def run_single(pair, seed, gens, market_is, mid_is, market_oos, mid_oos, pip, spread, bpd, workers=4):
    # JIT warm
    warm = np.zeros(N_PARAMS)
    cma_sim(market_is[:, :300], mid_is[:300], pip, spread, 50, warm, 0, 300)

    pool = ProcessPoolExecutor(max_workers=workers, initializer=_winit,
        initargs=(market_is, mid_is, pip, spread, MAX_HOLD, 3, 0.15, bpd))

    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    x0 = rng.randn(N_PARAMS) * 0.3
    x0[20:24] = rng.uniform(0.0, 1.0, 4)

    es = cma.CMAEvolutionStrategy(x0, 0.5,
        {'popsize': POP, 'seed': seed, 'verbose': -9, 'maxiter': gens})

    best_fit = 1e18; best_vec = None
    best_valid_pps = None; best_valid_vec = None
    t0 = time.time()
    gen = 0
    while not es.stop():
        c_ = es.ask()
        f_ = list(pool.map(_wfit, c_))
        es.tell(c_, f_)
        gm = min(f_)
        if gm < best_fit:
            best_fit = gm; best_vec = np.array(c_[f_.index(gm)])
        ok, mps = passes_gates(best_vec, market_is, mid_is, pip, spread, MAX_HOLD, 3, 0.15, bpd)
        if ok and (best_valid_pps is None or mps > best_valid_pps):
            best_valid_pps = mps; best_valid_vec = np.array(best_vec)
        gen += 1
        if gen >= gens: break
    pool.shutdown(wait=False)

    final = best_valid_vec if best_valid_vec is not None else best_vec
    is_nt, is_pnl, is_nl, is_ns = cma_sim(market_is, mid_is, pip, spread, MAX_HOLD, final, 0, len(mid_is))
    oos_nt, oos_pnl, oos_nl, oos_ns = cma_sim(market_oos, mid_oos, pip, spread, MAX_HOLD, final, 0, len(mid_oos))
    is_days = len(mid_is) / bpd
    oos_days = len(mid_oos) / bpd
    acts = [['sin','tanh','gauss'][decode_act(final[20+j])] for j in range(4)]

    return {
        "pair": pair, "seed": seed, "weights": final, "activations": acts,
        "is": {"n_trades": is_nt, "pips_per_day": is_pnl/is_days,
               "n_long": is_nl, "n_short": is_ns,
               "dir_ratio": min(is_nl, is_ns)/max(is_nt, 1)},
        "oos": {"n_trades": oos_nt, "pips_per_day": oos_pnl/oos_days,
                "n_long": oos_nl, "n_short": oos_ns,
                "dir_ratio": min(oos_nl, oos_ns)/max(oos_nt, 1)},
        "hard_gates": best_valid_pps is not None,
        "elapsed_s": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="EUR_JPY")
    p.add_argument("--tf", choices=["M5", "H1"], default="M5")
    p.add_argument("--seeds", nargs='+', type=int, default=[42, 137, 23, 99])
    p.add_argument("--gens", type=int, default=500)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]

    df = pd.read_parquet(PROJECT / f"data/m5_ohlc/{pair}_M5.parquet")
    if args.tf == "H1":
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').resample('1h').last().dropna().reset_index()
    mid = df['close'].values.astype(np.float64)
    n = len(mid)
    bpd = 288.0 if args.tf == "M5" else 24.0
    print(f"Loaded {n:,} {args.tf} bars of {pair}", flush=True)

    # Pre-training causality check
    validate_features_causal(mid, pip)
    print()

    # Compute features
    t0 = time.time()
    pips_d = compute_pips_delta(mid, n, pip)
    slope = compute_tl_slope(mid, n, 12)
    print(f"Features in {time.time()-t0:.1f}s  pips_d std={pips_d[100:].std():.3f}  slope std={slope[100:].std():.3f}", flush=True)

    market = np.stack([pips_d, slope], axis=0)
    split = int(n * 0.7)
    m_is = market[:, :split].copy(); mid_is = mid[:split].copy()
    m_oos = market[:, split:].copy(); mid_oos = mid[split:].copy()
    print(f"IS: {split:,} ({split/bpd:.0f}d), OOS: {n-split:,} ({(n-split)/bpd:.0f}d)")

    print(f"\n{'='*65}")
    print(f"  Minimal CMA-ES: {pair} {args.tf} | Gens: {args.gens} | Seeds: {args.seeds}")
    print(f"{'='*65}\n", flush=True)

    results = []
    for seed in args.seeds:
        print(f"── Seed {seed} ──", flush=True)
        r = run_single(pair, seed, args.gens, m_is, mid_is, m_oos, mid_oos, pip, spread, bpd, args.workers)
        results.append(r)
        print(f"  IS:  {r['is']['n_trades']}T L/S={r['is']['n_long']}/{r['is']['n_short']} "
              f"{r['is']['pips_per_day']:+.2f} p/d dir={r['is']['dir_ratio']:.2f}")
        print(f"  OOS: {r['oos']['n_trades']}T L/S={r['oos']['n_long']}/{r['oos']['n_short']} "
              f"{r['oos']['pips_per_day']:+.2f} p/d dir={r['oos']['dir_ratio']:.2f}")
        print(f"  Acts: {r['activations']}  Gate: {'PASS' if r['hard_gates'] else 'FAIL'}  ({r['elapsed_s']:.0f}s)\n", flush=True)

    print(f"{'='*65}")
    print(f"  MULTI-SEED SUMMARY: {pair} {args.tf}")
    print(f"{'='*65}")
    print(f"{'Seed':>6} {'IS p/d':>10} {'OOS p/d':>10} {'OOS Tr':>8} {'OOS Dir':>8} {'Gate':>6}")
    for r in results:
        gate = 'PASS' if r['hard_gates'] else 'FAIL'
        print(f"{r['seed']:>6} {r['is']['pips_per_day']:>+10.2f} "
              f"{r['oos']['pips_per_day']:>+10.2f} "
              f"{r['oos']['n_trades']:>8} "
              f"{r['oos']['dir_ratio']:>8.2f} {gate:>6}")
    mean_oos = np.mean([r['oos']['pips_per_day'] for r in results])
    print(f"{'MEAN':>6} {'':>10} {mean_oos:>+10.2f}")

    out = OUT_DIR / f"minimal_cma_{pair}_{args.tf}.pkl"
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
