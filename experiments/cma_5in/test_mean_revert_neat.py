"""Mean reversion on M5 zscore of (close - SMA10).

Feature: zscore of residual (close - SMA10), sign preserved
  residual = close - SMA10(close)
  z = residual / rolling_std(residual, window=100)
  input = tanh(z / 3.0)  # bound, 3 std normalized

Inputs (4):
  [0] z-scored residual (causal by construction)
  [1] upnl
  [2] mae
  [3] mfe

Architecture: 4 → 4 → 3 FIXED topology.
NEAT only mutates: activations (sin, tanh, gauss), weights, biases.
No node/connection add/delete.
Bidirectional enforced by fitness.
"""
import argparse, math, pickle, sys, time, random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from numba import njit
import neat

# Module-level activation wrappers (picklable, have __code__)
def sin_act(x): return math.sin(x)
def tanh_act(x): return math.tanh(x)
def gauss_act(x): return math.exp(-x*x)

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

POP = 100
GENS = 150
MAX_HOLD = 200   # 200 M5 bars = ~16.7h
BARS_PER_DAY = 288.0   # M5
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


# ══════════════════════════════════════════════════════════════════
# Feature: signed z-score of (close - SMA10)
# ══════════════════════════════════════════════════════════════════
@njit(cache=True)
def compute_zscore_residual(closes, n, sma_period=10, z_window=10000):
    """Residual = close - SMA10. Z = residual / rolling_std(residual, 100).
    Sign preserved. Returns raw z (un-normalized)."""
    out = np.zeros(n)
    # Rolling SMA10
    for i in range(sma_period - 1, n):
        s = 0.0
        for k in range(sma_period):
            s += closes[i - k]
        sma = s / sma_period
        residual = closes[i] - sma
        out[i] = residual

    # Now compute rolling std of residual (out currently holds residuals)
    # Z = residual / rolling_std(residual, z_window)
    z = np.zeros(n)
    for i in range(z_window + sma_period, n):
        mean_r = 0.0
        for k in range(z_window):
            mean_r += out[i - k]
        mean_r /= z_window
        var_r = 0.0
        for k in range(z_window):
            d = out[i - k] - mean_r
            var_r += d * d
        std_r = (var_r / z_window) ** 0.5
        if std_r > 1e-10:
            z[i] = out[i] / std_r  # keeps sign of residual

    return z


# ══════════════════════════════════════════════════════════════════
# Numba-compatible NEAT simulator (fixed topology 4→4→3 + bias term)
# Topology arrays encoded at eval time
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def simulate_neat(market, mid, pip, spread_pips, max_hold,
                  n_in, n_nodes, n_conns,
                  node_bias, node_resp, node_act,
                  conn_from, conn_to, conn_weight,
                  output_indices, chunk_start, chunk_end):
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
    values = np.zeros(n_nodes)

    for i in range(start, end):
        if position != 0:
            upnl_pips = (mid[i] - entry_price) / pip * position
            adv = -upnl_pips
            if adv > mae_pips: mae_pips = adv
            if upnl_pips > mfe_pips: mfe_pips = upnl_pips
        else:
            upnl_pips = 0.0; mae_pips = 0.0; mfe_pips = 0.0

        # 4 inputs
        values[0] = market[0, i]  # z-scored residual
        values[1] = np.tanh(upnl_pips / 20.0)
        values[2] = np.tanh(mae_pips / 20.0)
        values[3] = np.tanh(mfe_pips / 20.0)

        # Non-input nodes in topological order (hidden first, then outputs)
        for node_idx in range(n_in, n_nodes):
            z = node_bias[node_idx] * node_resp[node_idx]
            for c in range(n_conns):
                if conn_to[c] == node_idx:
                    z += conn_weight[c] * values[conn_from[c]]
            act = node_act[node_idx]
            if act == 0:   values[node_idx] = np.sin(z)
            elif act == 1: values[node_idx] = np.tanh(z)
            else:          values[node_idx] = np.exp(-z * z)  # gauss

        ob = values[output_indices[0]]
        os_ = values[output_indices[1]]
        of = values[output_indices[2]]

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


def genome_to_arrays(genome, config):
    """Extract topology into flat arrays for numba."""
    input_keys = list(config.genome_config.input_keys)
    output_keys = list(config.genome_config.output_keys)
    hidden_keys = [k for k in genome.nodes.keys() if k not in output_keys]

    node_list = input_keys + hidden_keys + output_keys
    node_idx = {k: i for i, k in enumerate(node_list)}
    n_nodes = len(node_list)
    n_in = len(input_keys)

    node_bias = np.zeros(n_nodes)
    node_resp = np.ones(n_nodes)
    node_act = np.zeros(n_nodes, dtype=np.int64)

    ACT_IDS = {"sin": 0, "tanh": 1, "gauss": 2}
    for key in hidden_keys + list(output_keys):
        if key in genome.nodes:
            nd = genome.nodes[key]
            idx = node_idx[key]
            node_bias[idx] = nd.bias
            node_resp[idx] = nd.response
            node_act[idx] = ACT_IDS.get(nd.activation, 1)  # default tanh

    conns = [c for c in genome.connections.values() if c.enabled]
    n_conns = len(conns)
    conn_from = np.zeros(n_conns, dtype=np.int64)
    conn_to = np.zeros(n_conns, dtype=np.int64)
    conn_weight = np.zeros(n_conns)
    for i, c in enumerate(conns):
        a, b = c.key
        conn_from[i] = node_idx[a]
        conn_to[i] = node_idx[b]
        conn_weight[i] = c.weight

    output_indices = np.array([node_idx[k] for k in output_keys], dtype=np.int64)
    return (n_in, n_nodes, n_conns, node_bias, node_resp, node_act,
            conn_from, conn_to, conn_weight, output_indices)


# Worker globals
_W = {}

def _winit(market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
    _W['market'] = market; _W['mid'] = mid
    _W['pip'] = pip; _W['spread'] = spread; _W['max_hold'] = max_hold
    _W['n_chunks'] = n_chunks; _W['min_dir'] = min_dir; _W['bpd'] = bpd


def _eval_one(args):
    """Args: (n_in, n_nodes, n_conns, nb, nr, na, cf, ct, cw, oi)"""
    n_in, n_nodes, n_conns, nb, nr, na, cf, ct, cw, oi = args
    if n_conns == 0:
        return -500.0
    market = _W['market']; mid = _W['mid']
    n_bars = len(mid)
    tl = 0; ts = 0; tt = 0; tp = 0.0
    chunk_pps = []; losing = 0.0
    for ci in range(_W['n_chunks']):
        c_s = int(n_bars * ci / _W['n_chunks'])
        c_e = int(n_bars * (ci + 1) / _W['n_chunks'])
        nt, pnl, nl, ns = simulate_neat(
            market, mid, _W['pip'], _W['spread'], _W['max_hold'],
            n_in, n_nodes, n_conns, nb, nr, na, cf, ct, cw, oi, c_s, c_e)
        tl += nl; ts += ns; tt += nt; tp += pnl
        days = (c_e - c_s) / _W['bpd']
        pps = pnl / days if days > 0 else 0
        chunk_pps.append(pps)
        if pps < 0: losing += -pps
    total_days = n_bars / _W['bpd']
    base_pps = tp / total_days if total_days > 0 else 0
    if tt == 0:
        return -500.0
    dir_ratio = min(tl, ts) / tt
    asym = (1.0 - 2.0 * dir_ratio) * 50.0
    act = max(0.0, 30.0 - tt) * 2.0
    all_prof = all(p > 0 for p in chunk_pps)
    if all_prof and dir_ratio >= _W['min_dir']:
        return min(chunk_pps) - asym
    return base_pps - asym - act - losing * 2.0


_POOL = None


def eval_pop_parallel(genomes, config):
    """Parallel fitness eval via multiprocessing."""
    # Pre-extract topology arrays (can't pickle genome directly with lambdas)
    args_list = [genome_to_arrays(g, config) for _, g in genomes]
    results = list(_POOL.map(_eval_one, args_list))
    for (gid, genome), fit in zip(genomes, results):
        genome.fitness = fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="EUR_JPY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]

    # Load M5 OHLC
    df = pd.read_parquet(PROJECT / f"data/m5_ohlc/{pair}_M5.parquet")
    mid = df['close'].values.astype(np.float64)
    n = len(mid)
    print(f"Loaded {n:,} M5 bars of {pair}", flush=True)

    # Compute feature
    t0 = time.time()
    z_raw = compute_zscore_residual(mid, n, 10, 10000)

    # Percentile scaling — non-parametric, adapts to empirical distribution.
    # Maps z to rolling percentile rank within last 10000 bars, then to [-1, +1].
    # No saturation bias — extremes stay extreme, middle stays middle.
    # Equally distinguishes ±12σ vs ±20σ (unlike tanh which saturates both).
    z_series = pd.Series(z_raw)
    # rolling rank with pct=True gives [1/window, 1.0]; we subtract 0.5 and double to [-1, +1]
    pctile = z_series.rolling(10000, min_periods=10000).rank(pct=True).fillna(0.5).values
    z_feat = (2.0 * (pctile - 0.5)).astype(np.float64)
    print(f"Feature: {time.time()-t0:.1f}s, "
          f"post-warmup range=[{z_feat[200:].min():+.3f}, {z_feat[200:].max():+.3f}], "
          f"std={z_feat[200:].std():.3f}", flush=True)

    market = np.stack([z_feat], axis=0)
    split = int(n * 0.7)
    m_is = market[:, :split].copy(); mid_is = mid[:split].copy()
    m_oos = market[:, split:].copy(); mid_oos = mid[split:].copy()
    print(f"IS: {split:,} M5 ({split/BARS_PER_DAY:.0f}d), OOS: {n-split:,} M5 ({(n-split)/BARS_PER_DAY:.0f}d)", flush=True)

    # NEAT config: FIXED topology 4→4→3
    cfg_path = OUT_DIR / "neat_mean_revert_config.ini"
    cfg = f"""
[NEAT]
fitness_criterion       = max
fitness_threshold       = 1e9
pop_size                = {POP}
reset_on_extinction     = False
no_fitness_termination  = True

[DefaultGenome]
num_inputs              = 4
num_outputs             = 3
num_hidden              = 4
feed_forward            = True
initial_connection      = full_direct
activation_default      = tanh
activation_mutate_rate  = 0.15
activation_options      = sin tanh gauss
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
conn_add_prob           = 0.0
conn_delete_prob        = 0.0
node_add_prob           = 0.0
node_delete_prob        = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.5
weight_max_value        = 6.0
weight_min_value        = -6.0
weight_mutate_rate      = 0.8
weight_mutate_power     = 0.5
weight_replace_rate     = 0.1
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 6.0
bias_min_value          = -6.0
bias_mutate_rate        = 0.7
bias_mutate_power       = 0.4
bias_replace_rate       = 0.1
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 1.0
response_min_value      = 1.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
response_mutate_power   = 0.0
enabled_default         = True
enabled_mutate_rate     = 0.0
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 25
species_elitism      = 2

[DefaultReproduction]
elitism              = 2
survival_threshold   = 0.3
min_species_size     = 2
"""
    with open(cfg_path, "w") as f:
        f.write(cfg)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, str(cfg_path))
    config.genome_config.add_activation('sin', sin_act)
    config.genome_config.add_activation('tanh', tanh_act)
    config.genome_config.add_activation('gauss', gauss_act)

    random.seed(args.seed); np.random.seed(args.seed)

    # JIT warm (fitness worker)
    warm_args = (4, 11, 20,
                 np.zeros(11), np.ones(11), np.zeros(11, dtype=np.int64),
                 np.zeros(20, dtype=np.int64), np.zeros(20, dtype=np.int64), np.zeros(20),
                 np.array([8, 9, 10], dtype=np.int64))
    # Quick warm on small slice
    simulate_neat(m_is[:, :300], mid_is[:300], pip, spread, 50,
                  4, 11, 20, np.zeros(11), np.ones(11), np.zeros(11, dtype=np.int64),
                  np.zeros(20, dtype=np.int64), np.zeros(20, dtype=np.int64), np.zeros(20),
                  np.array([8, 9, 10], dtype=np.int64), 0, 300)

    global _POOL
    _POOL = ProcessPoolExecutor(max_workers=args.workers,
        initializer=_winit,
        initargs=(m_is, mid_is, pip, spread, MAX_HOLD, 3, 0.15, BARS_PER_DAY))

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    print(f"\nNEAT fixed 4→4→3 | pop {POP} | gens {GENS} | acts {{sin, tanh, gauss}}", flush=True)
    t0 = time.time()
    winner = pop.run(eval_pop_parallel, GENS)
    elapsed = time.time() - t0

    # OOS
    net_t = genome_to_arrays(winner, config)
    (n_in, nn, nc, nb, nr, na, cf, ct, cw, oi) = net_t

    is_nt, is_pnl, is_nl, is_ns = simulate_neat(
        m_is, mid_is, pip, spread, MAX_HOLD,
        n_in, nn, nc, nb, nr, na, cf, ct, cw, oi, 0, len(mid_is))
    oos_nt, oos_pnl, oos_nl, oos_ns = simulate_neat(
        m_oos, mid_oos, pip, spread, MAX_HOLD,
        n_in, nn, nc, nb, nr, na, cf, ct, cw, oi, 0, len(mid_oos))

    is_days = len(mid_is) / BARS_PER_DAY
    oos_days = len(mid_oos) / BARS_PER_DAY
    is_dir = min(is_nl, is_ns) / max(is_nt, 1)
    oos_dir = min(oos_nl, oos_ns) / max(oos_nt, 1)

    print(f"\n{'='*65}")
    print(f"  MEAN REVERT NEAT: {pair}")
    print(f"{'='*65}")
    print(f"  Winner: {nn-n_in} non-input nodes, {nc} conns")
    print(f"  IS:  {is_nt}T L/S={is_nl}/{is_ns} {is_pnl:+.1f}p ({is_pnl/is_days:+.2f} p/d dir={is_dir:.2f})")
    print(f"  OOS: {oos_nt}T L/S={oos_nl}/{oos_ns} {oos_pnl:+.1f}p ({oos_pnl/oos_days:+.2f} p/d dir={oos_dir:.2f})")
    print(f"  Elapsed: {elapsed:.0f}s")

    # Save with named activation keys (not lambdas)
    save = {
        "pair": pair, "seed": args.seed,
        "topology": "4->4->3 fixed",
        "activations": ["sin", "tanh", "gauss"],
        "winner_arrays": net_t,
        "is": {"n_trades": is_nt, "total_pnl": is_pnl, "pips_per_day": is_pnl/is_days,
               "n_long": is_nl, "n_short": is_ns, "dir_ratio": is_dir},
        "oos": {"n_trades": oos_nt, "total_pnl": oos_pnl, "pips_per_day": oos_pnl/oos_days,
                "n_long": oos_nl, "n_short": oos_ns, "dir_ratio": oos_dir},
        "elapsed_s": elapsed,
    }
    path = OUT_DIR / f"mean_revert_neat_{pair}_s{args.seed}.pkl"
    with open(path, "wb") as f:
        pickle.dump(save, f)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
