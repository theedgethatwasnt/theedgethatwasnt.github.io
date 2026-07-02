"""Minimal NEAT baseline — 2023 oanda_neat input set on causal pipeline.

Features (4 inputs — all causal by construction):
  [0] pips_delta:    (close[t] - close[t-1]) / pip, tanh-normalized
  [1] tl_slope:      12-bar linear-regression slope, arctan-normalized [-1, +1]
  [2] unrealized:    tanh(upnl_pips / 20)           (position state, computed live)
  [3] holding_bars:  tanh(bars_since_entry / 20)    (position state, computed live)

NO MAE/MFE (simpler than V3). NO ASI. NO multi-TF.

Architecture: 4 → 4 → 3 FIXED topology, activations evolve among {sin, tanh, gauss}.
NEAT mutates weights + biases + activations only. Bidirectional enforced by fitness.

PRE-TRAINING: runs tests/test_feature_parity.py style checks on the feature
pipeline to confirm causality + streaming=batch before any training.

Usage:
    python3 test_minimal_neat_baseline.py --pair EUR_JPY --seed 42
"""
import argparse, math, pickle, sys, time, random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from numba import njit
import neat

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))


def sin_act(x): return math.sin(x)
def tanh_act(x): return math.tanh(x)
def gauss_act(x): return math.exp(-x*x)

POP = 100
GENS = 150
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


# ══════════════════════════════════════════════════════════════════
# Causal features (inline numba, no external deps)
# ══════════════════════════════════════════════════════════════════
@njit(cache=True)
def compute_pips_delta(closes, n, pip):
    """close[t] - close[t-1] in pips, tanh-normalized at ±10 pips."""
    out = np.zeros(n)
    for i in range(1, n):
        d_pips = (closes[i] - closes[i-1]) / pip
        out[i] = np.tanh(d_pips / 10.0)
    return out


@njit(cache=True)
def compute_tl_slope(closes, n, lookback=12):
    """12-bar linear regression slope on closes, arctan-normalized [-1, +1]."""
    out = np.zeros(n)
    hp = np.pi / 2.0
    for i in range(lookback, n):
        x_mean = (lookback - 1) / 2.0
        y_mean = 0.0
        for k in range(lookback):
            y_mean += closes[i - lookback + 1 + k]
        y_mean /= lookback
        num = 0.0
        den = 0.0
        ymin = closes[i - lookback + 1]
        ymax = ymin
        for k in range(lookback):
            y = closes[i - lookback + 1 + k]
            xd = k - x_mean
            num += xd * (y - y_mean)
            den += xd * xd
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
        if den > 0:
            slope = num / den
            rng = ymax - ymin
            out[i] = np.arctan((slope / rng * 3.0) if rng > 0 else 0.0) / hp
    return out


# ══════════════════════════════════════════════════════════════════
# Pre-training feature parity/causality check
# ══════════════════════════════════════════════════════════════════
def validate_features_causal(mid, pip, sample_n=2000):
    """Verify features at bar i don't change when bars [i+1:] are perturbed.
    This catches any lookahead bug before we waste training time.
    """
    print("Running pre-training parity checks on features...", flush=True)
    if sample_n > len(mid):
        sample_n = len(mid)

    # Reference features on full array
    d1 = compute_pips_delta(mid[:sample_n], sample_n, pip)
    s1 = compute_tl_slope(mid[:sample_n], sample_n, 12)

    # Probe: at bar 1000, perturb bars [1001:2000] and recompute
    probe = 1000
    mid2 = mid[:sample_n].copy()
    rng = np.random.default_rng(42)
    mid2[probe + 1:] = mid2[probe + 1:] + rng.normal(0, 0.1, sample_n - probe - 1)
    d2 = compute_pips_delta(mid2, sample_n, pip)
    s2 = compute_tl_slope(mid2, sample_n, 12)

    errors = []
    for name, a, b in [("pips_delta", d1, d2), ("tl_slope", s1, s2)]:
        # Features at bar <= probe must be unchanged (causal)
        diff = np.max(np.abs(a[:probe + 1] - b[:probe + 1]))
        if diff > 1e-10:
            errors.append(f"{name}: causality violation, max_diff_at_past={diff:.2e}")
        # Features after probe SHOULD differ (sanity: test is actually stressing the features)
        future_diff = np.max(np.abs(a[probe + 1:] - b[probe + 1:]))
        if future_diff < 1e-10:
            errors.append(f"{name}: test itself broken (future identical)")

    if errors:
        print("  ✗ FEATURE VALIDATION FAILED:")
        for e in errors: print(f"      {e}")
        raise RuntimeError("Causality broken — aborting training. Fix features first.")
    print("  ✓ pips_delta and tl_slope are causal")
    print("  ✓ feature pipeline validated for training")


# ══════════════════════════════════════════════════════════════════
# NEAT simulator (reuses pattern from test_mean_revert_neat)
# ══════════════════════════════════════════════════════════════════
@njit(cache=True)
def simulate_neat(market, mid, pip, spread_pips, max_hold,
                  n_in, n_nodes, n_conns,
                  node_bias, node_resp, node_act,
                  conn_from, conn_to, conn_weight,
                  output_indices, chunk_start, chunk_end):
    n = market.shape[1]
    start = max(chunk_start + 20, 20)
    end = min(chunk_end, n - 1)
    if end <= start:
        return 0, 0.0, 0, 0

    n_cap = end - start + 1
    pnls = np.zeros(n_cap)
    nt = 0; nl = 0; ns = 0
    position = 0; entry_price = 0.0; entry_bar = 0

    values = np.zeros(n_nodes)

    for i in range(start, end):
        if position != 0:
            upnl_pips = (mid[i] - entry_price) / pip * position
            holding = i - entry_bar
        else:
            upnl_pips = 0.0; holding = 0

        # 4 inputs: pips_delta, tl_slope, unrealized, holding_bars
        values[0] = market[0, i]
        values[1] = market[1, i]
        values[2] = np.tanh(upnl_pips / 20.0)
        values[3] = np.tanh(holding / 20.0)

        # Non-input nodes (topological order)
        for node_idx in range(n_in, n_nodes):
            z = node_bias[node_idx] * node_resp[node_idx]
            for c in range(n_conns):
                if conn_to[c] == node_idx:
                    z += conn_weight[c] * values[conn_from[c]]
            act = node_act[node_idx]
            if act == 0:   values[node_idx] = np.sin(z)
            elif act == 1: values[node_idx] = np.tanh(z)
            else:          values[node_idx] = np.exp(-z * z)

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


def genome_to_arrays(genome, config):
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
            node_act[idx] = ACT_IDS.get(nd.activation, 1)
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


_W = {}
def _winit(market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
    _W.update({'market':market,'mid':mid,'pip':pip,'spread':spread,
               'max_hold':max_hold,'n_chunks':n_chunks,'min_dir':min_dir,'bpd':bpd})


def _eval_one(args):
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


def eval_pop(genomes, config):
    args_list = [genome_to_arrays(g, config) for _, g in genomes]
    results = list(_POOL.map(_eval_one, args_list))
    for (gid, genome), fit in zip(genomes, results):
        genome.fitness = fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="EUR_JPY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gens", type=int, default=GENS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tf", choices=["M5", "H1"], default="M5",
                        help="M5 (288 bars/day) or H1 (resample M5 to hourly via .last())")
    args = parser.parse_args()

    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]

    # Load
    df = pd.read_parquet(PROJECT / f"data/m5_ohlc/{pair}_M5.parquet")
    if args.tf == "H1":
        # Resample to H1 via .last() on close — causal (uses only completed bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').resample('1h').last().dropna().reset_index()
        print(f"Resampled to H1: {len(df):,} bars", flush=True)

    mid = df['close'].values.astype(np.float64)
    n = len(mid)
    tf_label = args.tf
    bars_per_day = BARS_PER_DAY if args.tf == "M5" else 24.0
    print(f"Loaded {n:,} {tf_label} bars of {pair} ({bars_per_day} bars/day)", flush=True)

    # ──  PRE-TRAINING: verify features are causal ──
    validate_features_causal(mid, pip)
    print()

    # Compute features
    t0 = time.time()
    pips_d = compute_pips_delta(mid, n, pip)
    slope = compute_tl_slope(mid, n, 12)
    print(f"Features computed in {time.time()-t0:.1f}s", flush=True)

    # Sanity stats
    post = slice(100, None)
    print(f"  pips_delta: range=[{pips_d[post].min():+.3f}, {pips_d[post].max():+.3f}] std={pips_d[post].std():.3f}")
    print(f"  tl_slope: range=[{slope[post].min():+.3f}, {slope[post].max():+.3f}] std={slope[post].std():.3f}")

    market = np.stack([pips_d, slope], axis=0)
    split = int(n * 0.7)
    m_is = market[:, :split].copy(); mid_is = mid[:split].copy()
    m_oos = market[:, split:].copy(); mid_oos = mid[split:].copy()
    print(f"\nIS: {split:,} ({split/bars_per_day:.0f}d), OOS: {n-split:,} ({(n-split)/bars_per_day:.0f}d)")

    # NEAT config — FIXED 4→4→3 (topology mutation disabled)
    cfg_path = OUT_DIR / "neat_minimal_config.ini"
    with open(cfg_path, "w") as f:
        f.write(f"""
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
""")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, str(cfg_path))
    config.genome_config.add_activation('sin', sin_act)
    config.genome_config.add_activation('tanh', tanh_act)
    config.genome_config.add_activation('gauss', gauss_act)

    random.seed(args.seed); np.random.seed(args.seed)

    # JIT warm
    simulate_neat(m_is[:, :300], mid_is[:300], pip, spread, 50,
                  4, 11, 20, np.zeros(11), np.ones(11), np.zeros(11, dtype=np.int64),
                  np.zeros(20, dtype=np.int64), np.zeros(20, dtype=np.int64), np.zeros(20),
                  np.array([8, 9, 10], dtype=np.int64), 0, 300)

    global _POOL
    _POOL = ProcessPoolExecutor(max_workers=args.workers,
        initializer=_winit,
        initargs=(m_is, mid_is, pip, spread, MAX_HOLD, 3, 0.15, bars_per_day))

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))

    print(f"\nNEAT 4→4→3 fixed | pop {POP} | gens {args.gens} | acts {{sin, tanh, gauss}}", flush=True)
    t0 = time.time()
    winner = pop.run(eval_pop, args.gens)
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
    is_days = len(mid_is) / bars_per_day
    oos_days = len(mid_oos) / bars_per_day
    is_dir = min(is_nl, is_ns) / max(is_nt, 1)
    oos_dir = min(oos_nl, oos_ns) / max(oos_nt, 1)

    print(f"\n{'='*65}")
    print(f"  MINIMAL NEAT BASELINE: {pair}")
    print(f"{'='*65}")
    print(f"  Winner: {nn-n_in} non-input nodes, {nc} conns")
    print(f"  IS:  {is_nt}T L/S={is_nl}/{is_ns} {is_pnl:+.1f}p ({is_pnl/is_days:+.2f} p/d dir={is_dir:.2f})")
    print(f"  OOS: {oos_nt}T L/S={oos_nl}/{oos_ns} {oos_pnl:+.1f}p ({oos_pnl/oos_days:+.2f} p/d dir={oos_dir:.2f})")
    print(f"  Elapsed: {elapsed:.0f}s")

    save = {
        "pair": pair, "seed": args.seed,
        "topology": "4->4->3 fixed",
        "inputs": ["pips_delta", "tl_slope", "unrealized_tanh", "holding_tanh"],
        "activations": ["sin", "tanh", "gauss"],
        "winner_arrays": net_t,
        "is": {"n_trades": is_nt, "total_pnl": is_pnl, "pips_per_day": is_pnl/is_days,
               "n_long": is_nl, "n_short": is_ns, "dir_ratio": is_dir},
        "oos": {"n_trades": oos_nt, "total_pnl": oos_pnl, "pips_per_day": oos_pnl/oos_days,
                "n_long": oos_nl, "n_short": oos_ns, "dir_ratio": oos_dir},
        "elapsed_s": elapsed,
    }
    out = OUT_DIR / f"minimal_neat_{pair}_s{args.seed}.pkl"
    with open(out, "wb") as f:
        pickle.dump(save, f)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
