"""NEAT discovery on S5 momentum features (causal by construction).

Same 7 inputs as CMA-NN test:
  A: close[t]-close[t-6], B: close[t]-close[t-60], C: close[t]-close[t-720],
  D: 2nd diff of close at 5m scale, + upnl, mae, mfe

But NEAT evolves topology AND activation per node, seeded with one
pure-activation genome per type across the full wavelet+classic bank.

Bidirectional via fitness (min_dir_ratio=0.15 + asymmetry penalty).
"""
import argparse, math, pickle, sys, time, random
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
import neat

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

PAIR = "EUR_JPY"
PIP = 0.01
SPREAD = 2.3
MAX_HOLD = 720  # 1h at S5
POP = 80
GENS = 150
BARS_PER_DAY = 17280.0
CONFIG_PATH = PROJECT / "research/experiments/cma_5in/neat_config_s5_momentum.ini"
OUT_DIR = PROJECT / "research/experiments/cma_5in/results"
OUT_DIR.mkdir(exist_ok=True)


# ── Activation bank (13 functions) ──
ACTIVATIONS = [
    ('tanh',    lambda x: math.tanh(x)),
    ('sin',     lambda x: math.sin(x)),
    ('cos',     lambda x: math.cos(x)),
    ('gauss',   lambda x: math.exp(-x*x)),
    ('sech',    lambda x: 1.0 / math.cosh(max(-50.0, min(50.0, x)))),
    ('dog',     lambda x: math.exp(-x*x/2.0) - 0.5*math.exp(-x*x/8.0)),
    ('gabor',   lambda x: math.exp(-2.0*x*x) * math.cos(2.0*math.pi*x)),
    ('sinc',    lambda x: math.sin(math.pi*x)/(math.pi*x) if abs(x) > 1e-7 else 1.0),
    ('morlet',  lambda x: math.sin(x) * math.exp(-x*x/2.0)),
    ('sigmoid', lambda x: 1.0/(1.0 + math.exp(-max(-50.0, min(50.0, x))))),
    ('relu',    lambda x: max(0.0, x)),
    ('elu',     lambda x: x if x > 0 else math.exp(max(-50.0, x)) - 1.0),
    ('swish',   lambda x: x / (1.0 + math.exp(-max(-50.0, min(50.0, x))))),
]
ACT_NAMES = [a[0] for a in ACTIVATIONS]

# NEAT config template with all activations
NEAT_CFG = f"""
[NEAT]
fitness_criterion       = max
fitness_threshold       = 1e9
pop_size                = {POP}
reset_on_extinction     = False
no_fitness_termination  = True

[DefaultGenome]
num_inputs              = 7
num_outputs             = 3
num_hidden              = 1
feed_forward            = True
initial_connection      = full_direct
activation_default      = tanh
activation_mutate_rate  = 0.15
activation_options      = {' '.join(ACT_NAMES)}
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
conn_add_prob           = 0.4
conn_delete_prob        = 0.1
node_add_prob           = 0.2
node_delete_prob        = 0.05
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
enabled_mutate_rate     = 0.02
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism              = 2
survival_threshold   = 0.3
min_species_size     = 2
"""


@njit(cache=True)
def compute_s5_features(closes, n):
    A = np.zeros(n); B = np.zeros(n); C = np.zeros(n); D = np.zeros(n)
    for i in range(n):
        if i >= 6: A[i] = closes[i] - closes[i-6]
        if i >= 60: B[i] = closes[i] - closes[i-60]
        if i >= 720: C[i] = closes[i] - closes[i-720]
        if i >= 120: D[i] = closes[i] - 2.0*closes[i-60] + closes[i-120]
    return A, B, C, D


@njit(cache=True)
def simulate_neat_py(market, mid_close, pip, spread_pips, max_hold,
                     weights_flat, n_inputs, n_nodes, n_conns,
                     node_bias, node_resp, node_act_id,
                     conn_from, conn_to, conn_weight,
                     output_indices,
                     chunk_start, chunk_end):
    """Simulate NEAT network (generic graph) over a chunk.
    Similar logic to CMA-NN simulator but with arbitrary topology."""
    n = market.shape[1]
    start = max(chunk_start + 120, 120)
    end = min(chunk_end, n - 1)
    if end <= start:
        return 0, 0.0, 0, 0

    pnls = np.zeros(end - start + 1)
    nt = 0; nl = 0; ns = 0
    position = 0; entry_price = 0.0; entry_bar = 0
    mae_pips = 0.0; mfe_pips = 0.0

    values = np.zeros(n_nodes)

    for i in range(start, end):
        if position != 0:
            upnl_pips = (mid_close[i] - entry_price) / pip * position
            adv = -upnl_pips
            if adv > mae_pips: mae_pips = adv
            if upnl_pips > mfe_pips: mfe_pips = upnl_pips
        else:
            upnl_pips = 0.0; mae_pips = 0.0; mfe_pips = 0.0

        # Set input values
        values[0] = market[0, i]  # A
        values[1] = market[1, i]  # B
        values[2] = market[2, i]  # C
        values[3] = market[3, i]  # D
        values[4] = np.tanh(upnl_pips / 20.0)
        values[5] = np.tanh(mae_pips / 20.0)
        values[6] = np.tanh(mfe_pips / 20.0)

        # Forward pass: for each non-input node, sum weighted inputs from conns
        # Then apply activation (by act_id)
        # Assumes topological order in node_bias/node_act_id (inputs first, then hidden, then outputs)
        for node_idx in range(n_inputs, n_nodes):
            z = node_bias[node_idx] * node_resp[node_idx]
            for c in range(n_conns):
                if conn_to[c] == node_idx:
                    z += conn_weight[c] * values[conn_from[c]]
            # activation
            act = node_act_id[node_idx]
            if act == 0:   values[node_idx] = np.tanh(z)
            elif act == 1: values[node_idx] = np.sin(z)
            elif act == 2: values[node_idx] = np.cos(z)
            elif act == 3: values[node_idx] = np.exp(-z*z)
            elif act == 4:
                zc = max(-50.0, min(50.0, z))
                values[node_idx] = 1.0 / np.cosh(zc)
            elif act == 5: values[node_idx] = np.exp(-z*z/2.0) - 0.5*np.exp(-z*z/8.0)
            elif act == 6: values[node_idx] = np.exp(-2.0*z*z) * np.cos(2.0*np.pi*z)
            elif act == 7:
                if z > 1e-7 or z < -1e-7: values[node_idx] = np.sin(np.pi*z)/(np.pi*z)
                else: values[node_idx] = 1.0
            elif act == 8: values[node_idx] = np.sin(z) * np.exp(-z*z/2.0)
            elif act == 9:
                zc = max(-50.0, min(50.0, z))
                values[node_idx] = 1.0/(1.0 + np.exp(-zc))
            elif act == 10: values[node_idx] = max(0.0, z)
            elif act == 11:
                if z > 0: values[node_idx] = z
                else: values[node_idx] = np.exp(max(-50.0, z)) - 1.0
            else:
                zc = max(-50.0, min(50.0, z))
                values[node_idx] = z / (1.0 + np.exp(-zc))

        out_buy = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_flat = values[output_indices[2]]

        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid_close[i] - entry_price) / pip * position
            pnls[nt] = pnl; nt += 1
            if position > 0: nl += 1
            else: ns += 1
            position = 0

        if position == 0:
            if out_buy > out_sell and out_buy > out_flat:
                position = 1; entry_price = mid_close[i] + spread_pips * pip
                entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0
            elif out_sell > out_buy and out_sell > out_flat:
                position = -1; entry_price = mid_close[i] - spread_pips * pip
                entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0
        else:
            close_now = False; new_pos = 0
            if out_flat > out_buy and out_flat > out_sell: close_now = True
            elif position == 1 and out_sell > out_buy and out_sell > out_flat:
                close_now = True; new_pos = -1
            elif position == -1 and out_buy > out_sell and out_buy > out_flat:
                close_now = True; new_pos = 1
            if close_now:
                pnl = (mid_close[i] - entry_price) / pip * position
                pnls[nt] = pnl; nt += 1
                if position > 0: nl += 1
                else: ns += 1
                position = new_pos
                if new_pos != 0:
                    if new_pos == 1: entry_price = mid_close[i] + spread_pips * pip
                    else: entry_price = mid_close[i] - spread_pips * pip
                    entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0

    total = 0.0
    for k in range(nt): total += pnls[k]
    return nt, total, nl, ns


def genome_to_arrays(genome, config):
    """Extract topology from NEAT genome into flat arrays for numba."""
    genome_config = config.genome_config
    input_keys = list(genome_config.input_keys)   # e.g. [-1, -2, ..., -7]
    output_keys = list(genome_config.output_keys)  # [0, 1, 2]

    # Build list of nodes: inputs first, then hidden, then outputs
    # NEAT inputs are negative, we map them to 0..n_inputs-1
    hidden_keys = [k for k in genome.nodes.keys() if k not in output_keys]

    node_list = input_keys + hidden_keys + output_keys
    node_idx = {k: i for i, k in enumerate(node_list)}
    n_nodes = len(node_list)
    n_inputs = len(input_keys)

    node_bias = np.zeros(n_nodes)
    node_resp = np.ones(n_nodes)
    node_act_id = np.zeros(n_nodes, dtype=np.int64)

    for key in hidden_keys + list(output_keys):
        if key in genome.nodes:
            node = genome.nodes[key]
            idx = node_idx[key]
            node_bias[idx] = node.bias
            node_resp[idx] = node.response
            try:
                node_act_id[idx] = ACT_NAMES.index(node.activation)
            except ValueError:
                node_act_id[idx] = 0  # tanh default

    # Connections (enabled only)
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
    return (n_inputs, n_nodes, n_conns, node_bias, node_resp, node_act_id,
            conn_from, conn_to, conn_weight, output_indices)


# Globals for workers
_G = {}


def init_worker(market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd, config):
    _G['market'] = market; _G['mid'] = mid
    _G['pip'] = pip; _G['spread'] = spread; _G['max_hold'] = max_hold
    _G['n_chunks'] = n_chunks; _G['min_dir'] = min_dir; _G['bpd'] = bpd
    _G['config'] = config


def eval_genome(genome_id, genome, config):
    """Evaluate NEAT genome via WF fitness (3 chunks)."""
    try:
        net_tuple = genome_to_arrays(genome, config)
        (n_inputs, n_nodes, n_conns, nb, nr, na, cf, ct, cw, oi) = net_tuple
    except Exception:
        return -1000.0

    if n_conns == 0:
        return -500.0

    market = _G['market']; mid = _G['mid']
    n_bars = len(mid)
    n_chunks = _G['n_chunks']; min_dir = _G['min_dir']; bpd = _G['bpd']
    total_long = 0; total_short = 0; total_trades = 0; total_pnl = 0.0
    chunk_pps = []
    losing = 0.0

    for ci in range(n_chunks):
        c_s = int(n_bars * ci / n_chunks)
        c_e = int(n_bars * (ci + 1) / n_chunks)
        nt, pnl, nl, ns = simulate_neat_py(
            market, mid, _G['pip'], _G['spread'], _G['max_hold'],
            np.zeros(0), n_inputs, n_nodes, n_conns,
            nb, nr, na, cf, ct, cw, oi, c_s, c_e)
        total_long += nl; total_short += ns; total_trades += nt
        total_pnl += pnl
        n_days = (c_e - c_s) / bpd
        pps = pnl / n_days if n_days > 0 else 0
        chunk_pps.append(pps)
        if pps < 0: losing += -pps

    total_days = n_bars / bpd
    base_pps = total_pnl / total_days if total_days > 0 else 0

    if total_trades == 0:
        return -500.0

    dir_ratio = min(total_long, total_short) / total_trades
    asym_pen = (1.0 - 2.0 * dir_ratio) * 50.0
    act_pen = max(0.0, 30.0 - total_trades) * 2.0

    all_prof = all(p > 0 for p in chunk_pps)
    if all_prof and dir_ratio >= min_dir:
        return min(chunk_pps) - asym_pen
    else:
        return base_pps - asym_pen - act_pen - losing * 2.0


def eval_pop(genomes, config):
    for gid, genome in genomes:
        genome.fitness = eval_genome(gid, genome, config)


def main():
    # Write NEAT config file
    with open(CONFIG_PATH, "w") as f:
        f.write(NEAT_CFG)

    # Load data
    df = pd.read_parquet(PROJECT / f"data/s5_ohlc/{PAIR}_S5_BA.parquet")
    print(f"Loaded {len(df):,} S5 bars")
    mid = ((df['bid_c'].values + df['ask_c'].values) / 2.0).astype(np.float64)
    n = len(mid)

    t0 = time.time()
    A, B, C, D = compute_s5_features(mid, n)
    print(f"Features in {time.time()-t0:.1f}s")

    A_n = np.tanh(A / PIP / 5.0)
    B_n = np.tanh(B / PIP / 15.0)
    C_n = np.tanh(C / PIP / 40.0)
    D_n = np.tanh(D / PIP / 10.0)
    market = np.stack([A_n, B_n, C_n, D_n], axis=0)

    split = int(n * 0.7)
    m_is = market[:, :split].copy(); mid_is = mid[:split].copy()
    m_oos = market[:, split:].copy(); mid_oos = mid[split:].copy()
    print(f"IS: {split:,} ({split/BARS_PER_DAY:.0f}d), OOS: {n-split:,} ({(n-split)/BARS_PER_DAY:.0f}d)")

    # Setup NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, str(CONFIG_PATH))
    for name, fn in ACTIVATIONS:
        try: config.genome_config.add_activation(name, fn)
        except Exception: pass

    random.seed(42); np.random.seed(42)
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Set globals (single-threaded eval — multiproc w/ neat is fiddly)
    init_worker(m_is, mid_is, PIP, SPREAD, MAX_HOLD, 3, 0.15, BARS_PER_DAY, config)

    print(f"\nStarting NEAT: pop={POP}, gens={GENS}, 7 inputs, 3 outputs, 13 activations")
    t0 = time.time()
    try:
        winner = pop.run(eval_pop, GENS)
    except KeyboardInterrupt:
        print("Interrupted")
        winner = None

    elapsed = time.time() - t0

    if winner is None:
        print("No winner")
        return

    # Evaluate on OOS
    net_tuple = genome_to_arrays(winner, config)
    (n_inputs, n_nodes, n_conns, nb, nr, na, cf, ct, cw, oi) = net_tuple

    print(f"\nWinner: {n_nodes-n_inputs} non-input nodes, {n_conns} connections")

    nt, pnl, nl, ns = simulate_neat_py(
        m_is, mid_is, PIP, SPREAD, MAX_HOLD,
        np.zeros(0), n_inputs, n_nodes, n_conns,
        nb, nr, na, cf, ct, cw, oi, 0, len(mid_is))
    is_days = len(mid_is) / BARS_PER_DAY
    print(f"IS:  {nt}T L/S={nl}/{ns} {pnl:+.1f}p ({pnl/is_days:+.2f} p/d)")

    nt, pnl, nl, ns = simulate_neat_py(
        m_oos, mid_oos, PIP, SPREAD, MAX_HOLD,
        np.zeros(0), n_inputs, n_nodes, n_conns,
        nb, nr, na, cf, ct, cw, oi, 0, len(mid_oos))
    oos_days = len(mid_oos) / BARS_PER_DAY
    dir_ratio = min(nl, ns) / max(nt, 1)
    print(f"OOS: {nt}T L/S={nl}/{ns} {pnl:+.1f}p ({pnl/oos_days:+.2f} p/d, dir={dir_ratio:.2f})")
    print(f"Elapsed: {elapsed:.0f}s")

    # Save
    out = {
        "genome": winner, "config": config,
        "is_trades": nt, "oos_pips_per_day": pnl/oos_days,
        "dir_ratio": dir_ratio,
        "elapsed_s": elapsed,
    }
    path = OUT_DIR / f"s5_momentum_neat_{PAIR}_s42_best.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
