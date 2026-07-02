"""Mean revert CMA-ES variants — deeper topologies with act-as-gene.

Supports two architectures via --arch flag:
  - 'deep2':       4 → 4 → 3 → 3    (two hidden layers)
  - 'bottleneck':  4 → 4 → 1 (+ skip 4→3) → 3   (bottleneck + skip from input to output)

Same feature as test_mean_revert_neat.py: percentile-scaled z of (close - SMA10),
window=10000, sign-preserved.

Activations per hidden node = genes, bucketed to {sin, tanh, gauss}.
Runs multiple seeds, many gens for robustness.
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
BARS_PER_DAY = 288.0
MAX_HOLD = 200
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
N_OUT = 3


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


# ══════════════════════════════════════════════════════════════════
# Arch: deep2 = 4→4→3→3  (two hidden layers, no skip)
#   W1: 4×4=16, b1:4, act1:4  → 24
#   W2: 4×3=12, b2:3, act2:3  → 18
#   W_out: 3×3=9, b_out:3    → 12
#   Total: 54 params
# ══════════════════════════════════════════════════════════════════
N_H1_D2 = 4
N_H2_D2 = 3
N_PARAMS_D2 = N_IN*N_H1_D2 + N_H1_D2 + N_H1_D2 + N_H1_D2*N_H2_D2 + N_H2_D2 + N_H2_D2 + N_H2_D2*N_OUT + N_OUT


@njit(cache=True)
def sim_deep2(market, mid, pip, spread_pips, max_hold, weights, chunk_start, chunk_end):
    n = market.shape[1]
    start = max(chunk_start + 120, 120)
    end = min(chunk_end, n - 1)
    if end <= start: return 0, 0.0, 0, 0

    n_cap = end - start + 1
    pnls = np.zeros(n_cap)
    nt = 0; nl = 0; ns = 0
    position = 0; entry_price = 0.0; entry_bar = 0
    mae_pips = 0.0; mfe_pips = 0.0

    # Offsets
    w1_e = N_IN * N_H1_D2         # 16
    b1_e = w1_e + N_H1_D2          # 20
    a1_e = b1_e + N_H1_D2          # 24
    w2_e = a1_e + N_H1_D2 * N_H2_D2  # 36
    b2_e = w2_e + N_H2_D2          # 39
    a2_e = b2_e + N_H2_D2          # 42
    wout_e = a2_e + N_H2_D2 * N_OUT  # 51
    # bout_e = wout_e + N_OUT = 54

    # Pre-decode activations
    a1_ids = np.zeros(N_H1_D2, dtype=np.int64)
    a2_ids = np.zeros(N_H2_D2, dtype=np.int64)
    for j in range(N_H1_D2):
        a1_ids[j] = decode_act(weights[b1_e + j])
    for j in range(N_H2_D2):
        a2_ids[j] = decode_act(weights[b2_e + j])

    h1 = np.zeros(N_H1_D2)
    h2 = np.zeros(N_H2_D2)
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

        # L1: input → h1 (with act gene per node)
        for j in range(N_H1_D2):
            z = weights[w1_e + j]  # b1[j]
            for k in range(N_IN):
                z += weights[j * N_IN + k] * inp[k]
            h1[j] = apply_act(z, a1_ids[j])

        # L2: h1 → h2
        for j in range(N_H2_D2):
            z = weights[w2_e + j]  # b2[j]
            for k in range(N_H1_D2):
                z += weights[a1_e + j * N_H1_D2 + k] * h1[k]
            h2[j] = apply_act(z, a2_ids[j])

        # Output
        ob = weights[wout_e + 0]
        os_ = weights[wout_e + 1]
        of = weights[wout_e + 2]
        for k in range(N_H2_D2):
            ob += weights[a2_e + 0 * N_H2_D2 + k] * h2[k]
            os_ += weights[a2_e + 1 * N_H2_D2 + k] * h2[k]
            of += weights[a2_e + 2 * N_H2_D2 + k] * h2[k]

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


# ══════════════════════════════════════════════════════════════════
# Arch: bottleneck = 4→4→1 + skip(4→3)→3
#   L1:    W1 4×4=16, b1:4, act1:4  → 24
#   L2:    W2 4×1=4,  b2:1, act2:1  → 6   (1-node bottleneck)
#   Output: W_out 1×3=3, b_out:3    → 6   (from bottleneck)
#   Skip:   W_skip 4×3=12             → 12 (input direct → output)
#   Total: 48 params
# ══════════════════════════════════════════════════════════════════
N_H1_BN = 4
N_H2_BN = 1
N_PARAMS_BN = N_IN*N_H1_BN + N_H1_BN + N_H1_BN + N_H1_BN*N_H2_BN + N_H2_BN + N_H2_BN + N_H2_BN*N_OUT + N_OUT + N_IN*N_OUT


@njit(cache=True)
def sim_bottleneck(market, mid, pip, spread_pips, max_hold, weights, chunk_start, chunk_end):
    n = market.shape[1]
    start = max(chunk_start + 120, 120)
    end = min(chunk_end, n - 1)
    if end <= start: return 0, 0.0, 0, 0

    n_cap = end - start + 1
    pnls = np.zeros(n_cap)
    nt = 0; nl = 0; ns = 0
    position = 0; entry_price = 0.0; entry_bar = 0
    mae_pips = 0.0; mfe_pips = 0.0

    w1_e = N_IN * N_H1_BN          # 16
    b1_e = w1_e + N_H1_BN           # 20
    a1_e = b1_e + N_H1_BN           # 24
    w2_e = a1_e + N_H1_BN * N_H2_BN # 28  (4*1)
    b2_e = w2_e + N_H2_BN           # 29
    a2_e = b2_e + N_H2_BN           # 30
    wout_e = a2_e + N_H2_BN * N_OUT # 33  (1*3)
    bout_e = wout_e + N_OUT         # 36
    # wskip_e = bout_e + N_IN * N_OUT = 48

    a1_ids = np.zeros(N_H1_BN, dtype=np.int64)
    for j in range(N_H1_BN):
        a1_ids[j] = decode_act(weights[b1_e + j])
    a2_id = decode_act(weights[b2_e + 0])

    h1 = np.zeros(N_H1_BN)
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
        for j in range(N_H1_BN):
            z = weights[w1_e + j]
            for k in range(N_IN):
                z += weights[j * N_IN + k] * inp[k]
            h1[j] = apply_act(z, a1_ids[j])

        # L2 (single bottleneck node)
        z = weights[w2_e + 0]  # b2
        for k in range(N_H1_BN):
            z += weights[a1_e + k] * h1[k]
        h2 = apply_act(z, a2_id)

        # Output = bottleneck contribution + direct skip from input
        ob = weights[bout_e + 0] + weights[a2_e + 0] * h2
        os_ = weights[bout_e + 1] + weights[a2_e + 1] * h2
        of = weights[bout_e + 2] + weights[a2_e + 2] * h2
        # Skip
        for k in range(N_IN):
            ob += weights[bout_e + N_OUT + 0 * N_IN + k] * inp[k]
            os_ += weights[bout_e + N_OUT + 1 * N_IN + k] * inp[k]
            of += weights[bout_e + N_OUT + 2 * N_IN + k] * inp[k]

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


# Shared worker infra
def make_fitness_and_gates(simulate_fn):
    def fitness(weights, market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
        n_bars = len(mid)
        tl = 0; ts = 0; tt = 0; tp = 0.0
        chunk_pps = []; losing = 0.0
        for ci in range(n_chunks):
            c_s = int(n_bars * ci / n_chunks)
            c_e = int(n_bars * (ci + 1) / n_chunks)
            nt, pnl, nl, ns = simulate_fn(market, mid, pip, spread, max_hold, weights, c_s, c_e)
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
            nt, pnl, nl, ns = simulate_fn(market, mid, pip, spread, max_hold, weights, c_s, c_e)
            tl += nl; ts += ns; tt += nt
            days = (c_e - c_s) / bpd
            min_tr = max(20, int(days * 0.5))
            if nt < min_tr or pnl <= 0:
                return False, None
            chunk_pps.append(pnl / days)
        if tt < 30: return False, None
        if min(tl, ts) / tt < min_dir: return False, None
        return True, min(chunk_pps)
    return fitness, passes_gates


_W = {}

def _winit_d2(market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
    _W.update({'market':market,'mid':mid,'pip':pip,'spread':spread,
               'max_hold':max_hold,'n_chunks':n_chunks,'min_dir':min_dir,'bpd':bpd,
               'fn':sim_deep2})

def _winit_bn(market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
    _W.update({'market':market,'mid':mid,'pip':pip,'spread':spread,
               'max_hold':max_hold,'n_chunks':n_chunks,'min_dir':min_dir,'bpd':bpd,
               'fn':sim_bottleneck})

def _wfit(vec):
    fit_fn, _ = make_fitness_and_gates(_W['fn'])
    return fit_fn(vec, _W['market'], _W['mid'], _W['pip'], _W['spread'],
                  _W['max_hold'], _W['n_chunks'], _W['min_dir'], _W['bpd'])


def run_single(pair, seed, arch, gens, market_is, mid_is, market_oos, mid_oos, pip, spread, workers=4):
    if arch == 'deep2':
        n_params = N_PARAMS_D2
        simulate_fn = sim_deep2
        winit_fn = _winit_d2
    else:  # bottleneck
        n_params = N_PARAMS_BN
        simulate_fn = sim_bottleneck
        winit_fn = _winit_bn

    fit_fn, gates_fn = make_fitness_and_gates(simulate_fn)

    # JIT warm
    warm = np.zeros(n_params)
    simulate_fn(market_is[:, :300], mid_is[:300], pip, spread, 50, warm, 0, 300)

    pool = ProcessPoolExecutor(max_workers=workers, initializer=winit_fn,
        initargs=(market_is, mid_is, pip, spread, MAX_HOLD, 3, 0.15, BARS_PER_DAY))

    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    x0 = rng.randn(n_params) * 0.3
    # Init activation genes in [0,1) based on arch
    if arch == 'deep2':
        # 4 L1 acts at [20:24], 3 L2 acts at [39:42]
        x0[20:24] = rng.uniform(0.0, 1.0, 4)
        x0[39:42] = rng.uniform(0.0, 1.0, 3)
    else:
        # bottleneck: 4 L1 acts at [20:24], 1 L2 act at [29]
        x0[20:24] = rng.uniform(0.0, 1.0, 4)
        x0[29] = rng.uniform(0.0, 1.0)

    es = cma.CMAEvolutionStrategy(x0, 0.5,
        {'popsize': POP, 'seed': seed, 'verbose': -9, 'maxiter': gens})

    best_fit = 1e18; best_vec = None
    best_valid_pps = None; best_valid_vec = None
    t0 = time.time(); gen = 0
    while not es.stop():
        c_ = es.ask()
        f_ = list(pool.map(_wfit, c_))
        es.tell(c_, f_)
        gm = min(f_)
        if gm < best_fit:
            best_fit = gm; best_vec = np.array(c_[f_.index(gm)])
        ok, mps = gates_fn(best_vec, market_is, mid_is, pip, spread, MAX_HOLD, 3, 0.15, BARS_PER_DAY)
        if ok and (best_valid_pps is None or mps > best_valid_pps):
            best_valid_pps = mps; best_valid_vec = np.array(best_vec)
        gen += 1
        if gen >= gens: break
    pool.shutdown(wait=False)

    final = best_valid_vec if best_valid_vec is not None else best_vec
    is_nt, is_pnl, is_nl, is_ns = simulate_fn(market_is, mid_is, pip, spread, MAX_HOLD, final, 0, len(mid_is))
    oos_nt, oos_pnl, oos_nl, oos_ns = simulate_fn(market_oos, mid_oos, pip, spread, MAX_HOLD, final, 0, len(mid_oos))
    is_days = len(mid_is) / BARS_PER_DAY
    oos_days = len(mid_oos) / BARS_PER_DAY

    return {
        "pair": pair, "seed": seed, "arch": arch,
        "weights": final,
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
    p.add_argument("--arch", choices=['deep2', 'bottleneck'], default='deep2')
    p.add_argument("--seeds", nargs='+', type=int, default=[42, 137, 23, 99])
    p.add_argument("--gens", type=int, default=500)
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
    z_raw = compute_zscore_residual(mid, n, 10, 10000)
    pctile = pd.Series(z_raw).rolling(10000, min_periods=10000).rank(pct=True).fillna(0.5).values
    z_feat = (2.0 * (pctile - 0.5)).astype(np.float64)
    print(f"Feature ready, std={z_feat[10500:].std():.3f}", flush=True)

    market = np.stack([z_feat], axis=0)
    split = int(n * 0.7)
    m_is = market[:, :split].copy(); mid_is = mid[:split].copy()
    m_oos = market[:, split:].copy(); mid_oos = mid[split:].copy()

    print(f"\n{'='*65}")
    print(f"  Arch: {args.arch} | Gens: {args.gens} | Seeds: {args.seeds}")
    print(f"{'='*65}\n", flush=True)

    results = []
    for seed in args.seeds:
        print(f"── Seed {seed} ──", flush=True)
        r = run_single(pair, seed, args.arch, args.gens, m_is, mid_is, m_oos, mid_oos, pip, spread, args.workers)
        results.append(r)
        print(f"  IS:  {r['is']['n_trades']}T L/S={r['is']['n_long']}/{r['is']['n_short']} "
              f"{r['is']['pips_per_day']:+.2f} p/d dir={r['is']['dir_ratio']:.2f}")
        print(f"  OOS: {r['oos']['n_trades']}T L/S={r['oos']['n_long']}/{r['oos']['n_short']} "
              f"{r['oos']['pips_per_day']:+.2f} p/d dir={r['oos']['dir_ratio']:.2f}")
        print(f"  Hard gates: {'PASS' if r['hard_gates'] else 'FAIL'}  ({r['elapsed_s']:.0f}s)\n", flush=True)

    # Summary
    print(f"{'='*65}")
    print(f"  MULTI-SEED SUMMARY: {pair} {args.arch}")
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

    # Save
    out_path = OUT_DIR / f"mean_revert_cma_{args.arch}_{pair}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
