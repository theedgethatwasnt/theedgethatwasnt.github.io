"""Minimal 4-input M5 test: (close[t] - close[t-3])/pip + UPnL + MAE + MFE.

Inputs (all causal):
  [0] d3:    tanh( (close[t] - close[t-3]) / pip / 10.0 )
  [1] upnl:  tanh(upnl_pips / 20)
  [2] mae:   tanh(mae_pips  / 20)          (mae ≥ 0, starts at spread on entry)
  [3] mfe:   tanh(mfe_pips  / 20)          (mfe ≥ 0, starts at 0 on entry)

Outputs: BUY / SELL / FLATTEN (argmax).
Topology: 4 → 4 → 3 FIXED, activations evolve among {sin, tanh, gauss}.

Usage:
    python3 test_dclose3_mfe_mae.py --pair EUR_JPY --seed 42
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
def gauss_act(x): return math.exp(-x * x)


POP = 100
GENS = 150
MAX_HOLD = 200
BARS_PER_DAY = 288.0
OUT_DIR = PROJECT / "research/experiments/cma_5in/results"
OUT_DIR.mkdir(exist_ok=True)

PAIR_PIP = {"EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
            "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
            "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
            "NZD_USD": 0.0001, "EUR_GBP": 0.0001}
PAIR_SPREAD = {"EUR_JPY": 2.3, "USD_JPY": 1.7, "GBP_JPY": 3.3, "AUD_JPY": 2.1,
               "CAD_JPY": 2.3, "CHF_JPY": 3.5, "NZD_JPY": 2.7,
               "EUR_USD": 1.6, "GBP_USD": 1.9, "AUD_USD": 1.3,
               "NZD_USD": 1.5, "EUR_GBP": 1.4}


@njit(cache=True)
def compute_d3(closes, n, pip):
    """(close[t] - close[t-3]) in pips, tanh-normalized at ±10 pips."""
    out = np.zeros(n)
    for i in range(3, n):
        d_pips = (closes[i] - closes[i - 3]) / pip
        out[i] = np.tanh(d_pips / 10.0)
    return out


def validate_causal(mid, pip, sample_n=2000):
    """Confirm d3 at bar i doesn't change when bars [i+1:] are perturbed."""
    print("Pre-training causality check on d3...", flush=True)
    if sample_n > len(mid):
        sample_n = len(mid)
    d1 = compute_d3(mid[:sample_n], sample_n, pip)
    probe = 1000
    mid2 = mid[:sample_n].copy()
    rng = np.random.default_rng(42)
    mid2[probe + 1:] = mid2[probe + 1:] + rng.normal(0, 0.1, sample_n - probe - 1)
    d2 = compute_d3(mid2, sample_n, pip)
    past = np.max(np.abs(d1[:probe + 1] - d2[:probe + 1]))
    future = np.max(np.abs(d1[probe + 1:] - d2[probe + 1:]))
    if past > 1e-10:
        raise RuntimeError(f"d3 causality violated: past max_diff={past:.2e}")
    if future < 1e-10:
        raise RuntimeError("d3 test broken: future identical under perturbation")
    print(f"  ✓ d3 causal (past_diff={past:.2e}, future_diff={future:.2e})")


@njit(cache=True)
def simulate_neat(d3, mid, pip, spread_pips, max_hold,
                  n_in, n_nodes, n_conns,
                  node_bias, node_resp, node_act,
                  conn_from, conn_to, conn_weight,
                  output_indices, chunk_start, chunk_end,
                  amddp_coef, theta):
    """Returns (nt, total_score, total_pnl, cum_mae, nl, ns).
    INTEGRATED drawdown: at every bar a position is open, add the running MAE
    (monotone non-decreasing per trade) to cum_mae. A trade that sits deep
    underwater for many bars is penalized more than one that briefly touches
    the same DD and exits.
    total_score = total_pnl - amddp_coef * cum_mae."""
    n = len(mid)
    start = max(chunk_start + 20, 20)
    end = min(chunk_end, n - 1)
    if end <= start:
        return 0, 0.0, 0.0, 0.0, 0, 0

    nt = 0; nl = 0; ns = 0
    total_pnl = 0.0; cum_mae = 0.0
    position = 0; entry_price = 0.0; entry_bar = 0
    mae_pips = 0.0; mfe_pips = 0.0
    values = np.zeros(n_nodes)

    for i in range(start, end):
        if position != 0:
            upnl_pips = (mid[i] - entry_price) / pip * position
            adv = -upnl_pips
            if adv > mae_pips: mae_pips = adv
            if upnl_pips > mfe_pips: mfe_pips = upnl_pips
            # Integrated drawdown: add current running MAE each bar in position
            cum_mae += mae_pips
        else:
            upnl_pips = 0.0; mae_pips = 0.0; mfe_pips = 0.0

        values[0] = d3[i]
        values[1] = np.tanh(upnl_pips / 20.0)
        values[2] = np.tanh(mae_pips / 20.0)
        values[3] = np.tanh(mfe_pips / 20.0)

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
            total_pnl += pnl
            nt += 1
            if position > 0: nl += 1
            else: ns += 1
            position = 0

        # Latching decision: challenger must beat the incumbent output by > theta
        # to switch state. Flat is output 2 (FLATTEN). Long=0 (BUY), Short=1 (SELL).
        if position == 0:
            # Incumbent = FLATTEN (of). Challengers = ob (long), os_ (short).
            best = of; best_id = -1  # stay flat by default
            if ob - best > theta and ob > os_: best = ob; best_id = 1
            if os_ - best > theta and os_ > ob: best = os_; best_id = -1
            # Only accept flip if challenger beats BOTH the incumbent by theta AND the other challenger
            if best_id == 1 and (ob - of) > theta and ob > os_:
                position = 1; entry_price = mid[i] + spread_pips * pip
                entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0
            elif best_id == -1 and (os_ - of) > theta and os_ > ob:
                position = -1; entry_price = mid[i] - spread_pips * pip
                entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0
        else:
            # Incumbent = current position's entry action
            # For position=+1 incumbent=ob (BUY); position=-1 incumbent=os_ (SELL)
            incumbent = ob if position == 1 else os_
            # FLATTEN wins only if of > incumbent + theta
            flat_challenges = (of - incumbent) > theta and of > (ob if position == -1 else os_) + theta
            # Reverse wins only if other-side beats incumbent by theta AND flat
            reverse_val = os_ if position == 1 else ob
            reverse_challenges = (reverse_val - incumbent) > theta and (reverse_val - of) > theta
            close_now = False; new_pos = 0
            if flat_challenges: close_now = True; new_pos = 0
            elif reverse_challenges: close_now = True; new_pos = -position
            if close_now:
                pnl = (mid[i] - entry_price) / pip * position
                total_pnl += pnl
                nt += 1
                if position > 0: nl += 1
                else: ns += 1
                position = new_pos
                if new_pos != 0:
                    if new_pos == 1: entry_price = mid[i] + spread_pips * pip
                    else: entry_price = mid[i] - spread_pips * pip
                    entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0

    total_score = total_pnl - amddp_coef * cum_mae
    return nt, total_score, total_pnl, cum_mae, nl, ns


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
def _winit(d3, mid, pip, spread, max_hold, n_chunks, min_dir, bpd, amddp_coef, theta):
    _W.update({'d3': d3, 'mid': mid, 'pip': pip, 'spread': spread,
               'max_hold': max_hold, 'n_chunks': n_chunks, 'min_dir': min_dir,
               'bpd': bpd, 'amddp': amddp_coef, 'theta': theta})


def _eval_one(args):
    n_in, n_nodes, n_conns, nb, nr, na, cf, ct, cw, oi = args
    if n_conns == 0:
        return -500.0
    d3 = _W['d3']; mid = _W['mid']
    n_bars = len(mid)
    tl = 0; ts = 0; tt = 0; tscore = 0.0
    chunk_sps = []; losing = 0.0
    for ci in range(_W['n_chunks']):
        c_s = int(n_bars * ci / _W['n_chunks'])
        c_e = int(n_bars * (ci + 1) / _W['n_chunks'])
        nt, score, _pnl, _cmae, nl, ns = simulate_neat(
            d3, mid, _W['pip'], _W['spread'], _W['max_hold'],
            n_in, n_nodes, n_conns, nb, nr, na, cf, ct, cw, oi, c_s, c_e,
            _W['amddp'], _W['theta'])
        tl += nl; ts += ns; tt += nt; tscore += score
        days = (c_e - c_s) / _W['bpd']
        sps = score / days if days > 0 else 0  # score-per-day
        chunk_sps.append(sps)
        if sps < 0: losing += -sps
    total_days = n_bars / _W['bpd']
    base_sps = tscore / total_days if total_days > 0 else 0
    if tt == 0:
        return -500.0
    dir_ratio = min(tl, ts) / tt
    asym = (1.0 - 2.0 * dir_ratio) * 50.0
    act = max(0.0, 30.0 - tt) * 2.0
    all_prof = all(s > 0 for s in chunk_sps)
    if all_prof and dir_ratio >= _W['min_dir']:
        return min(chunk_sps) - asym
    return base_sps - asym - act - losing * 2.0


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
    parser.add_argument("--fitness", choices=["pips", "amddp1", "amddp5"], default="pips",
                        help="pips=raw P&L/day. amddp1=pips - 0.01*cum_mae. amddp5=0.05.")
    parser.add_argument("--theta", type=float, default=0.0,
                        help="Latching hysteresis. 0=argmax (default). >0 requires challenger to beat "
                             "incumbent output by this margin before switching state. Try 0.1-0.3.")
    args = parser.parse_args()

    AMDDP_COEFS = {"pips": 0.0, "amddp1": 0.01, "amddp5": 0.05}
    amddp_coef = AMDDP_COEFS[args.fitness]

    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]

    df = pd.read_parquet(PROJECT / f"data/m5_ohlc/{pair}_M5.parquet")
    mid = df['close'].values.astype(np.float64)
    n = len(mid)
    print(f"Loaded {n:,} M5 bars of {pair}", flush=True)

    validate_causal(mid, pip)

    t0 = time.time()
    d3 = compute_d3(mid, n, pip)
    print(f"d3 computed in {time.time()-t0:.1f}s", flush=True)
    post = slice(100, None)
    print(f"  d3: range=[{d3[post].min():+.3f}, {d3[post].max():+.3f}] std={d3[post].std():.3f}")

    split = int(n * 0.7)
    d3_is = d3[:split].copy(); mid_is = mid[:split].copy()
    d3_oos = d3[split:].copy(); mid_oos = mid[split:].copy()
    print(f"\nIS: {split:,} ({split/BARS_PER_DAY:.0f}d), OOS: {n-split:,} ({(n-split)/BARS_PER_DAY:.0f}d)")

    cfg_path = OUT_DIR / "neat_dclose3_config.ini"
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
    simulate_neat(d3_is[:300], mid_is[:300], pip, spread, 50,
                  4, 11, 20, np.zeros(11), np.ones(11), np.zeros(11, dtype=np.int64),
                  np.zeros(20, dtype=np.int64), np.zeros(20, dtype=np.int64), np.zeros(20),
                  np.array([8, 9, 10], dtype=np.int64), 0, 300, amddp_coef, args.theta)

    global _POOL
    _POOL = ProcessPoolExecutor(max_workers=args.workers,
        initializer=_winit,
        initargs=(d3_is, mid_is, pip, spread, MAX_HOLD, 3, 0.15, BARS_PER_DAY, amddp_coef, args.theta))

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))

    print(f"\nNEAT 4→4→3 fixed | pop {POP} | gens {args.gens} | fitness={args.fitness} (amddp_coef={amddp_coef}) | θ={args.theta} | acts {{sin, tanh, gauss}}", flush=True)
    t0 = time.time()
    winner = pop.run(eval_pop, args.gens)
    elapsed = time.time() - t0

    net_t = genome_to_arrays(winner, config)
    (n_in, nn, nc, nb, nr, na, cf, ct, cw, oi) = net_t
    is_nt, _is_score, is_pnl, is_cmae, is_nl, is_ns = simulate_neat(
        d3_is, mid_is, pip, spread, MAX_HOLD,
        n_in, nn, nc, nb, nr, na, cf, ct, cw, oi, 0, len(mid_is), amddp_coef, args.theta)
    oos_nt, _oos_score, oos_pnl, oos_cmae, oos_nl, oos_ns = simulate_neat(
        d3_oos, mid_oos, pip, spread, MAX_HOLD,
        n_in, nn, nc, nb, nr, na, cf, ct, cw, oi, 0, len(mid_oos), amddp_coef, args.theta)
    is_days = len(mid_is) / BARS_PER_DAY
    oos_days = len(mid_oos) / BARS_PER_DAY
    is_dir = min(is_nl, is_ns) / max(is_nt, 1)
    oos_dir = min(oos_nl, oos_ns) / max(oos_nt, 1)

    print(f"\n{'='*65}")
    print(f"  DCLOSE3 + UPnL + MAE + MFE: {pair}")
    print(f"{'='*65}")
    print(f"  Winner: {nn-n_in} non-input nodes, {nc} conns")
    print(f"  IS:  {is_nt}T L/S={is_nl}/{is_ns} {is_pnl:+.1f}p ({is_pnl/is_days:+.2f} p/d dir={is_dir:.2f} cumMAE={is_cmae:.0f}p)")
    print(f"  OOS: {oos_nt}T L/S={oos_nl}/{oos_ns} {oos_pnl:+.1f}p ({oos_pnl/oos_days:+.2f} p/d dir={oos_dir:.2f} cumMAE={oos_cmae:.0f}p)")
    print(f"  Elapsed: {elapsed:.0f}s")

    save = {
        "pair": pair, "seed": args.seed,
        "topology": "4->4->3 fixed",
        "inputs": ["d3_tanh", "upnl_tanh", "mae_tanh", "mfe_tanh"],
        "activations": ["sin", "tanh", "gauss"],
        "fitness": args.fitness, "amddp_coef": amddp_coef, "theta": args.theta,
        "winner_arrays": net_t,
        "is": {"n_trades": is_nt, "total_pnl": is_pnl, "pips_per_day": is_pnl/is_days,
               "n_long": is_nl, "n_short": is_ns, "dir_ratio": is_dir},
        "oos": {"n_trades": oos_nt, "total_pnl": oos_pnl, "pips_per_day": oos_pnl/oos_days,
                "n_long": oos_nl, "n_short": oos_ns, "dir_ratio": oos_dir},
        "elapsed_s": elapsed,
    }
    theta_tag = f"_th{args.theta}".replace(".", "") if args.theta > 0 else ""
    out = OUT_DIR / f"dclose3_mfe_mae_{pair}_s{args.seed}_{args.fitness}{theta_tag}.pkl"
    with open(out, "wb") as f:
        pickle.dump(save, f)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
