"""PnF-based momentum features + CMA-NN / NEAT training.

Pipeline:
  S5 mid → PnF(box_size, reversal=2) → per-box close level
  At each S5 bar: which box is current, compute features

Features (5 inputs):
  box_close - box_close[-10]    (10-boxes-ago diff, normalized)
  box_close - box_close[-120]   (120-boxes-ago diff, normalized)
  upnl, mae, mfe                (position state)

Architecture 1 (CMA-NN fixed):
  5 → 6(sin) → 4(gauss) → 3 + skip from input → 3

Architecture 2 (NEAT):
  5 inputs, free topology, activations = {sin, gauss, tanh}

Runs --box-size (5 or 10 pips), --arch (cma | neat).
"""
import argparse, math, pickle, sys, time, random
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
from numba import njit

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

PAIR = "EUR_JPY"
PIP = 0.01
SPREAD = 2.3
MAX_HOLD_S5_BARS = 1440  # 2 hours at S5
POP_CMA = 24
GENS_CMA = 200
POP_NEAT = 80
GENS_NEAT = 150
BARS_PER_DAY = 17280.0
OUT_DIR = PROJECT / "research/experiments/cma_5in/results"
OUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# Incremental PnF builder — per-S5-bar box tracking
# ══════════════════════════════════════════════════════════════════

@njit(cache=True)
def build_pnf_features(mid, n, box_size, reversal, lookback_short=10, lookback_long=120):
    """For each S5 bar, compute PnF-based features.

    Returns arrays of size n:
      current_box_level[i]  (float): level of current PnF box
      box_count[i]          (int):   total boxes created so far
      feat_short[i]         (float): current_box - box_N_ago_short, normalized pip-scale
      feat_long[i]          (float): current_box - box_N_ago_long, normalized pip-scale
      is_new_box[i]         (bool):  did a new box just form at bar i

    Causal by construction — uses only past price data.
    """
    # State
    current_level = 0.0
    direction = 0          # 0 init, +1 up, -1 down
    box_count = 0
    # Ring buffer of last N box levels
    max_buf = lookback_long + 10
    box_levels = np.zeros(max_buf)  # circular buffer
    box_buf_head = 0

    out_current = np.zeros(n)
    out_count = np.zeros(n, dtype=np.int64)
    feat_short = np.zeros(n)
    feat_long = np.zeros(n)
    is_new = np.zeros(n, dtype=np.int64)

    for i in range(n):
        p = mid[i]
        had_new_box = False

        if direction == 0:
            # Initialize on first price
            current_level = math.floor(p / box_size) * box_size
            direction = 1
            box_levels[box_buf_head % max_buf] = current_level
            box_buf_head += 1
            box_count = 1
            had_new_box = True
        else:
            delta_boxes = int((p - current_level) / box_size)

            # Same-direction continuation
            if direction == 1 and delta_boxes >= 1:
                for _ in range(delta_boxes):
                    current_level += box_size
                    box_levels[box_buf_head % max_buf] = current_level
                    box_buf_head += 1
                    box_count += 1
                    had_new_box = True
            elif direction == -1 and delta_boxes <= -1:
                for _ in range(-delta_boxes):
                    current_level -= box_size
                    box_levels[box_buf_head % max_buf] = current_level
                    box_buf_head += 1
                    box_count += 1
                    had_new_box = True
            # Reversal conditions
            elif direction == 1 and delta_boxes <= -reversal:
                direction = -1
                current_level -= box_size
                for _ in range(-delta_boxes):
                    box_levels[box_buf_head % max_buf] = current_level
                    box_buf_head += 1
                    box_count += 1
                    current_level -= box_size
                    had_new_box = True
                current_level += box_size  # undo last extra decrement
            elif direction == -1 and delta_boxes >= reversal:
                direction = 1
                current_level += box_size
                for _ in range(delta_boxes):
                    box_levels[box_buf_head % max_buf] = current_level
                    box_buf_head += 1
                    box_count += 1
                    current_level += box_size
                    had_new_box = True
                current_level -= box_size

        out_current[i] = current_level
        out_count[i] = box_count
        is_new[i] = 1 if had_new_box else 0

        # Features: diff between current box level and N boxes ago
        if box_count > lookback_short:
            old_idx = (box_buf_head - 1 - lookback_short) % max_buf
            feat_short[i] = current_level - box_levels[old_idx]
        if box_count > lookback_long:
            old_idx = (box_buf_head - 1 - lookback_long) % max_buf
            feat_long[i] = current_level - box_levels[old_idx]

    return out_current, out_count, feat_short, feat_long, is_new


# ══════════════════════════════════════════════════════════════════
# CMA-NN: 5 → 6(sin) → 4(gauss) → 3 + skip(input→output)
# ══════════════════════════════════════════════════════════════════
# Params:
#  W1: 5*6=30, b1: 6      (input→L1)
#  W2: 6*4=24, b2: 4      (L1→L2)
#  W_out: 4*3=12, b_out: 3  (L2→output)
#  W_skip: 5*3=15         (input→output skip)
#  Total: 30+6+24+4+12+3+15 = 94

N_IN = 5
N_L1 = 6
N_L2 = 4
N_OUT = 3
N_PARAMS = N_IN*N_L1 + N_L1 + N_L1*N_L2 + N_L2 + N_L2*N_OUT + N_OUT + N_IN*N_OUT

@njit(cache=True)
def cma_simulate(market, mid_close, pip, spread_pips, max_hold, weights,
                 chunk_start, chunk_end):
    """Simulate CMA-NN 5→6(sin)→4(gauss)→3+skip on a chunk."""
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

    # Weight offsets
    w1_e = N_IN * N_L1
    b1_e = w1_e + N_L1
    w2_e = b1_e + N_L1 * N_L2
    b2_e = w2_e + N_L2
    w_out_e = b2_e + N_L2 * N_OUT
    b_out_e = w_out_e + N_OUT
    # w_skip_e = b_out_e + N_IN * N_OUT

    h1 = np.zeros(N_L1)
    h2 = np.zeros(N_L2)
    out = np.zeros(N_OUT)
    inp = np.zeros(N_IN)

    for i in range(start, end):
        if position != 0:
            upnl_pips = (mid_close[i] - entry_price) / pip * position
            adv = -upnl_pips
            if adv > mae_pips: mae_pips = adv
            if upnl_pips > mfe_pips: mfe_pips = upnl_pips
        else:
            upnl_pips = 0.0; mae_pips = 0.0; mfe_pips = 0.0

        inp[0] = market[0, i]
        inp[1] = market[1, i]
        inp[2] = np.tanh(upnl_pips / 20.0)
        inp[3] = np.tanh(mae_pips / 20.0)
        inp[4] = np.tanh(mfe_pips / 20.0)

        # L1: 5 → 6 sin
        for j in range(N_L1):
            z = weights[w1_e + j]  # b1[j]
            for k in range(N_IN):
                z += weights[j * N_IN + k] * inp[k]
            h1[j] = np.sin(z)

        # L2: 6 → 4 gauss
        for j in range(N_L2):
            z = weights[b1_e + N_L1 * N_L2 + j]  # b2[j]
            for k in range(N_L1):
                z += weights[b1_e + j * N_L1 + k] * h1[k]
            h2[j] = np.exp(-z * z)

        # Out: 4 → 3 linear + skip from input
        for j in range(N_OUT):
            z = weights[w_out_e + j]  # b_out[j]
            for k in range(N_L2):
                z += weights[b2_e + j * N_L2 + k] * h2[k]
            # Skip from input
            for k in range(N_IN):
                z += weights[b_out_e + j * N_IN + k] * inp[k]
            out[j] = z

        ob, os_, of = out[0], out[1], out[2]

        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid_close[i] - entry_price) / pip * position
            pnls[nt] = pnl; nt += 1
            if position > 0: nl += 1
            else: ns += 1
            position = 0

        if position == 0:
            if ob > os_ and ob > of:
                position = 1; entry_price = mid_close[i] + spread_pips * pip
                entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0
            elif os_ > ob and os_ > of:
                position = -1; entry_price = mid_close[i] - spread_pips * pip
                entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0
        else:
            close_now = False; new_pos = 0
            if of > ob and of > os_: close_now = True
            elif position == 1 and os_ > ob and os_ > of: close_now = True; new_pos = -1
            elif position == -1 and ob > os_ and ob > of: close_now = True; new_pos = 1
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


def cma_fitness(weights, market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
    n_bars = len(mid)
    tl = 0; ts = 0; tt = 0; tp = 0.0
    chunk_pps = []; losing = 0.0
    for ci in range(n_chunks):
        c_s = int(n_bars * ci / n_chunks)
        c_e = int(n_bars * (ci + 1) / n_chunks)
        nt, pnl, nl, ns = cma_simulate(market, mid, pip, spread, max_hold, weights, c_s, c_e)
        tl += nl; ts += ns; tt += nt; tp += pnl
        days = (c_e - c_s) / bpd
        pps = pnl / days if days > 0 else 0.0
        chunk_pps.append(pps)
        if pps < 0: losing += -pps
    total_days = n_bars / bpd
    base_pps = tp / total_days if total_days > 0 else 0.0
    if tt == 0:
        return 500.0 - base_pps
    dir_ratio = min(tl, ts) / tt
    asym = (1.0 - 2.0 * dir_ratio) * 50.0
    act = max(0.0, 30.0 - tt) * 2.0
    all_prof = all(p > 0 for p in chunk_pps)
    if all_prof and dir_ratio >= min_dir:
        score = min(chunk_pps) - asym
    else:
        score = base_pps - asym - act - losing * 2.0
    return -score


def cma_passes_gates(weights, market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
    n_bars = len(mid)
    tl = 0; ts = 0; tt = 0; chunk_pps = []
    for ci in range(n_chunks):
        c_s = int(n_bars * ci / n_chunks)
        c_e = int(n_bars * (ci + 1) / n_chunks)
        nt, pnl, nl, ns = cma_simulate(market, mid, pip, spread, max_hold, weights, c_s, c_e)
        tl += nl; ts += ns; tt += nt
        days = (c_e - c_s) / bpd
        min_tr = max(20, int(days * 0.5))
        if nt < min_tr or pnl <= 0:
            return False, None
        chunk_pps.append(pnl / days)
    if tt < 30: return False, None
    if min(tl, ts) / tt < min_dir: return False, None
    return True, min(chunk_pps)


# Worker globals
_W = {}
def _winit(market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
    _W['market']=market; _W['mid']=mid; _W['pip']=pip; _W['spread']=spread
    _W['max_hold']=max_hold; _W['n_chunks']=n_chunks; _W['min_dir']=min_dir; _W['bpd']=bpd

def _wfit(vec):
    return cma_fitness(vec, _W['market'], _W['mid'], _W['pip'], _W['spread'],
                       _W['max_hold'], _W['n_chunks'], _W['min_dir'], _W['bpd'])


def run_cma(market, mid, pip, spread, max_hold, bpd, seed=42):
    from concurrent.futures import ProcessPoolExecutor
    import cma as cma_mod

    n = len(mid)
    split = int(n * 0.7)
    m_is = market[:, :split].copy(); mid_is = mid[:split].copy()
    m_oos = market[:, split:].copy(); mid_oos = mid[split:].copy()

    # JIT warm
    warm = np.zeros(N_PARAMS)
    cma_simulate(m_is[:, :300], mid_is[:300], pip, spread, 50, warm, 0, 300)

    pool = ProcessPoolExecutor(max_workers=4, initializer=_winit,
        initargs=(m_is, mid_is, pip, spread, max_hold, 3, 0.15, bpd))

    np.random.seed(seed)
    x0 = np.random.RandomState(seed).randn(N_PARAMS) * 0.3
    es = cma_mod.CMAEvolutionStrategy(x0, 0.5, {'popsize': POP_CMA, 'seed': seed, 'verbose': -9, 'maxiter': GENS_CMA})

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
        ok, mps = cma_passes_gates(best_vec, m_is, mid_is, pip, spread, max_hold, 3, 0.15, bpd)
        if ok and (best_valid_pps is None or mps > best_valid_pps):
            best_valid_pps = mps; best_valid_vec = np.array(best_vec)
        gen += 1
        if gen % 25 == 0:
            print(f"  Gen {gen}: fit={best_fit:.2f}, valid={best_valid_pps}, t={time.time()-t0:.0f}s", flush=True)
        if gen >= GENS_CMA: break

    pool.shutdown(wait=False)
    final = best_valid_vec if best_valid_vec is not None else best_vec

    is_nt, is_pnl, is_nl, is_ns = cma_simulate(m_is, mid_is, pip, spread, max_hold, final, 0, len(mid_is))
    oos_nt, oos_pnl, oos_nl, oos_ns = cma_simulate(m_oos, mid_oos, pip, spread, max_hold, final, 0, len(mid_oos))
    is_days = len(mid_is) / bpd; oos_days = len(mid_oos) / bpd

    return {
        "arch": "cma_5_6sin_4gauss_skip",
        "weights": final,
        "is": {"n_trades": is_nt, "total_pnl": is_pnl, "pips_per_day": is_pnl/is_days,
               "n_long": is_nl, "n_short": is_ns,
               "dir_ratio": min(is_nl, is_ns)/max(is_nt, 1)},
        "oos": {"n_trades": oos_nt, "total_pnl": oos_pnl, "pips_per_day": oos_pnl/oos_days,
                "n_long": oos_nl, "n_short": oos_ns,
                "dir_ratio": min(oos_nl, oos_ns)/max(oos_nt, 1)},
        "hard_gates": best_valid_pps is not None,
        "min_chunk_pps": best_valid_pps,
    }


# ══════════════════════════════════════════════════════════════════
# NEAT version (only sin, gauss, tanh) — same simulator shape
# ══════════════════════════════════════════════════════════════════
def run_neat(market, mid, pip, spread, max_hold, bpd, seed=42):
    import neat
    cfg_path = OUT_DIR / "neat_pnf_config.ini"
    cfg_text = f"""
[NEAT]
fitness_criterion       = max
fitness_threshold       = 1e9
pop_size                = {POP_NEAT}
reset_on_extinction     = False
no_fitness_termination  = True

[DefaultGenome]
num_inputs              = 5
num_outputs             = 3
num_hidden              = 1
feed_forward            = True
initial_connection      = full_direct
activation_default      = tanh
activation_mutate_rate  = 0.15
activation_options      = sin gauss tanh
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
    with open(cfg_path, "w") as f:
        f.write(cfg_text)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, str(cfg_path))
    for name, fn in [('sin', math.sin), ('gauss', lambda x: math.exp(-x*x)), ('tanh', math.tanh)]:
        try: config.genome_config.add_activation(name, fn)
        except Exception: pass

    # Splits
    n = len(mid)
    split = int(n * 0.7)
    m_is = market[:, :split].copy(); mid_is = mid[:split].copy()
    m_oos = market[:, split:].copy(); mid_oos = mid[split:].copy()

    # Fitness using neat-python's FeedForwardNetwork (slow but correct)
    def eval_pop(genomes, config):
        for gid, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = -_neat_fitness(net, m_is, mid_is, pip, spread, max_hold, 3, 0.15, bpd)

    random.seed(seed); np.random.seed(seed)
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))

    t0 = time.time()
    try:
        winner = pop.run(eval_pop, GENS_NEAT)
    except KeyboardInterrupt:
        winner = None

    elapsed = time.time() - t0
    if winner is None:
        return {"arch": "neat_pnf", "error": "no winner"}

    # OOS eval
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    is_nt, is_pnl, is_nl, is_ns = _neat_sim(net, m_is, mid_is, pip, spread, max_hold, 0, len(mid_is))
    oos_nt, oos_pnl, oos_nl, oos_ns = _neat_sim(net, m_oos, mid_oos, pip, spread, max_hold, 0, len(mid_oos))
    is_days = len(mid_is) / bpd; oos_days = len(mid_oos) / bpd
    return {
        "arch": "neat_pnf",
        "genome": winner, "config": config,
        "is": {"n_trades": is_nt, "pips_per_day": is_pnl/is_days,
               "n_long": is_nl, "n_short": is_ns,
               "dir_ratio": min(is_nl, is_ns)/max(is_nt, 1)},
        "oos": {"n_trades": oos_nt, "pips_per_day": oos_pnl/oos_days,
                "n_long": oos_nl, "n_short": oos_ns,
                "dir_ratio": min(oos_nl, oos_ns)/max(oos_nt, 1)},
        "elapsed_s": elapsed,
    }


def _neat_sim(net, market, mid, pip, spread_pips, max_hold, start, end):
    """Slow Python NEAT simulator (per-genome eval)."""
    n = market.shape[1]
    s = max(start + 120, 120)
    e = min(end, n - 1)
    if e <= s:
        return 0, 0.0, 0, 0
    nt = 0; nl = 0; ns = 0; total = 0.0
    position = 0; entry_price = 0.0; entry_bar = 0
    mae_pips = 0.0; mfe_pips = 0.0
    for i in range(s, e):
        if position != 0:
            upnl_pips = (mid[i] - entry_price) / pip * position
            adv = -upnl_pips
            if adv > mae_pips: mae_pips = adv
            if upnl_pips > mfe_pips: mfe_pips = upnl_pips
        else:
            upnl_pips = 0.0; mae_pips = 0.0; mfe_pips = 0.0
        inp = [market[0, i], market[1, i],
               math.tanh(upnl_pips/20.0),
               math.tanh(mae_pips/20.0),
               math.tanh(mfe_pips/20.0)]
        ob, os_, of = net.activate(inp)
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid[i] - entry_price) / pip * position
            total += pnl; nt += 1
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
                total += pnl; nt += 1
                if position > 0: nl += 1
                else: ns += 1
                position = new_pos
                if new_pos != 0:
                    if new_pos == 1: entry_price = mid[i] + spread_pips * pip
                    else: entry_price = mid[i] - spread_pips * pip
                    entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0
    return nt, total, nl, ns


def _neat_fitness(net, market, mid, pip, spread, max_hold, n_chunks, min_dir, bpd):
    n_bars = len(mid)
    tl = 0; ts = 0; tt = 0; tp = 0.0
    chunk_pps = []; losing = 0.0
    for ci in range(n_chunks):
        c_s = int(n_bars * ci / n_chunks)
        c_e = int(n_bars * (ci + 1) / n_chunks)
        nt, pnl, nl, ns = _neat_sim(net, market, mid, pip, spread, max_hold, c_s, c_e)
        tl += nl; ts += ns; tt += nt; tp += pnl
        days = (c_e - c_s) / bpd
        pps = pnl / days if days > 0 else 0.0
        chunk_pps.append(pps)
        if pps < 0: losing += -pps
    total_days = n_bars / bpd
    base_pps = tp / total_days if total_days > 0 else 0.0
    if tt == 0:
        return 500.0 - base_pps
    dir_ratio = min(tl, ts) / tt
    asym = (1.0 - 2.0 * dir_ratio) * 50.0
    act = max(0.0, 30.0 - tt) * 2.0
    all_prof = all(p > 0 for p in chunk_pps)
    if all_prof and dir_ratio >= min_dir:
        return -(min(chunk_pps) - asym)
    return -(base_pps - asym - act - losing * 2.0)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--box-size", type=int, default=5, choices=[5, 10])
    parser.add_argument("--reversal", type=int, default=2)
    parser.add_argument("--arch", choices=["cma", "neat"], default="cma")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    label = f"pnf_b{args.box_size}r{args.reversal}_{args.arch}"
    print(f"{'='*65}")
    print(f"  PnF Momentum: box={args.box_size}pip, rev={args.reversal}, arch={args.arch}")
    print(f"{'='*65}", flush=True)

    # Load
    df = pd.read_parquet(PROJECT / f"data/s5_ohlc/{PAIR}_S5_BA.parquet")
    mid = ((df['bid_c'].values + df['ask_c'].values) / 2.0).astype(np.float64)
    n = len(mid)
    print(f"Loaded {n:,} S5 bars", flush=True)

    # PnF features
    box = args.box_size * PIP
    t0 = time.time()
    current_level, box_count, feat_short, feat_long, is_new = build_pnf_features(
        mid, n, box, args.reversal, 10, 120)
    print(f"PnF features: {time.time()-t0:.1f}s, total boxes: {box_count[-1]}", flush=True)

    # Normalize features to [-1,+1] pip-scale
    # feat_short: up to 10 boxes diff = 10*5=50 or 10*10=100 pips
    # feat_long: up to 120 boxes diff = 600 or 1200 pips
    feat_short_n = np.tanh(feat_short / PIP / (args.box_size * 5.0))   # scale: 5-box worth
    feat_long_n = np.tanh(feat_long / PIP / (args.box_size * 30.0))    # scale: 30-box worth

    print(f"Feature stats (post-warmup, bar {n//2}:):")
    for name, v in [("feat_short", feat_short_n[n//2:]), ("feat_long", feat_long_n[n//2:])]:
        print(f"  {name}: [{v.min():+.3f}, {v.max():+.3f}] mean={v.mean():+.4f} std={v.std():.3f}")

    market = np.stack([feat_short_n, feat_long_n], axis=0)
    print(f"\nTraining {args.arch} (popsize {POP_CMA if args.arch=='cma' else POP_NEAT}, "
          f"gens {GENS_CMA if args.arch=='cma' else GENS_NEAT})...", flush=True)

    if args.arch == "cma":
        result = run_cma(market, mid, PIP, SPREAD, MAX_HOLD_S5_BARS, BARS_PER_DAY, args.seed)
    else:
        result = run_neat(market, mid, PIP, SPREAD, MAX_HOLD_S5_BARS, BARS_PER_DAY, args.seed)

    print(f"\n{'='*65}")
    print(f"  RESULT: {label}")
    print(f"  IS:  {result['is']['pips_per_day']:+.2f} p/d  "
          f"({result['is']['n_trades']}T, dir={result['is']['dir_ratio']:.2f})")
    print(f"  OOS: {result['oos']['pips_per_day']:+.2f} p/d  "
          f"({result['oos']['n_trades']}T, dir={result['oos']['dir_ratio']:.2f})")
    if "hard_gates" in result:
        print(f"  Hard gates: {'PASS' if result['hard_gates'] else 'FAIL'}")
    print(f"{'='*65}")

    # Save
    out = OUT_DIR / f"{label}_s{args.seed}_best.pkl"
    with open(out, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
