#!/usr/bin/env python3
"""
Activation Function Experiment
================================
1. Generate synthetic EUR/USD-like sine wave price
2. Compute ASI-MC indicators (MC_D, MC_dD, ER_norm)
3. Visualise each activation {tanh, sin, cos, gauss, mex_hat} as shape curve
   AND as transform of the live MC_D signal
4. Train a fixed-topology NEAT (5→5→7→3) with all 5 activations
5. Plot fitness curve + best genome's trades on price

Fixed topology (2 hidden layers):
  Inputs:  -1=MC_D  -2=MC_dD  -3=ER_norm  -4=UPnL  -5=mex_hat(MC_D)
  Layer 1: nodes 100-104  (5 nodes)
  Layer 2: nodes 200-206  (7 nodes)
  Outputs: 0=BUY  1=SELL  2=EXIT
  Connections: in→L1(25) + L1→L2(35) + L2→out(21) + in→out skip(15) = 96
"""

import math
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # must be before pyplot import
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import neat
import numpy as np

# ─── Synthetic Price ──────────────────────────────────────────────────────────

def generate_price(n_bars=600, period=200, amplitude_pips=80,
                   center=1.1000, pip=0.0001, noise_pips=2.5, seed=42):
    rng = np.random.RandomState(seed)
    t = np.arange(n_bars, dtype=np.float64)
    clean = center + amplitude_pips * pip * np.sin(2 * np.pi * t / period)
    noise = rng.randn(n_bars) * noise_pips * pip
    close = clean + noise
    # Thin OHLC bars
    high  = close + rng.uniform(0.5, 3, n_bars) * pip
    low   = close - rng.uniform(0.5, 3, n_bars) * pip
    open_ = np.roll(close, 1); open_[0] = close[0]
    return open_, high, low, close, pip

# ─── Indicators ───────────────────────────────────────────────────────────────

def compute_asi(open_, high, low, close):
    """Simplified ASI (Wilder): cumulative normalised price change."""
    n = len(close)
    asi = np.zeros(n)
    for i in range(1, n):
        tr = max(high[i], close[i-1]) - min(low[i], close[i-1])
        if tr < 1e-12:
            continue
        asi[i] = asi[i-1] + (close[i] - close[i-1]) / tr
    return asi

def ema(x, period):
    alpha = 2.0 / (period + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i-1]
    return out

def compute_mc(asi, sma_p=5, e_s=3, e_l=5):
    """MC(D) = EMA3(SMA5(ASI)) − EMA5(SMA5(ASI)); MC(dD) = diff(MC_D)."""
    kernel = np.ones(sma_p) / sma_p
    sma = np.convolve(asi, kernel, mode='same')
    mc_d  = ema(sma, e_s) - ema(sma, e_l)
    mc_dd = np.diff(mc_d, prepend=mc_d[0])
    return mc_d, mc_dd

def compute_er(close, period=60):
    """Kaufman ER, arctan-normalised to (~0,1]."""
    n = len(close)
    er = np.zeros(n)
    for i in range(period, n):
        direction  = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period : i + 1])))
        er[i] = direction / volatility if volatility > 1e-12 else 0.0
    return np.arctan(er * 5.0) / (math.pi / 2.0)

def normalise(x, clip=3.0):
    std = np.std(x)
    return np.clip(x / (std + 1e-10), -clip, clip)

# ─── Activation Functions ─────────────────────────────────────────────────────

def act_tanh(x):    return np.tanh(x)
def act_sin(x):     return np.sin(x)
def act_cos(x):     return np.cos(x)
def act_gauss(x):   return np.exp(-x * x)
def act_mex_hat(x): return (1.0 - x**2) * np.exp(-x**2 / 2.0)

# scalar versions for neat-python
def _tanh(x):    return math.tanh(x)
def _sin(x):     return math.sin(x)
def _cos(x):     return math.cos(x)
def _gauss(x):   return math.exp(-x * x)
def _mex_hat(x): return (1.0 - x*x) * math.exp(-x*x / 2.0)

ACTIVATIONS = [
    ('tanh',    act_tanh,    _tanh,    'S-curve',            '#e74c3c'),
    ('sin',     act_sin,     _sin,     'Wave',               '#f39c12'),
    ('cos',     act_cos,     _cos,     'Wave (shifted)',     '#2ecc71'),
    ('gauss',   act_gauss,   _gauss,   'Bell curve',         '#3498db'),
    ('mex_hat', act_mex_hat, _mex_hat, 'Wavelet/edge detect','#9b59b6'),
]

ACT_NAMES = [a[0] for a in ACTIVATIONS]

def register_activations(config):
    for name, _, scalar_fn, _, _ in ACTIVATIONS:
        try:
            config.genome_config.add_activation(name, scalar_fn)
        except RuntimeError:
            pass

# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_indicators(close, mc_d, mc_dd, er_norm, pip):
    mc_d_n = normalise(mc_d)

    fig = plt.figure(figsize=(18, 15))
    gs  = gridspec.GridSpec(4, 5, hspace=0.55, wspace=0.38,
                            top=0.93, bottom=0.05)

    # ── Row 0: price ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ax.plot(close, color='#2c3e50', lw=0.9, label='Close')
    center = np.mean(close)
    ax.axhline(center, color='gray', lw=0.6, ls='--', label='Center')
    ax.set_title('Synthetic EUR/USD  (sine + noise)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Price'); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    # ── Row 1: indicators ─────────────────────────────────────────────────────
    for col, (signal, label, color) in enumerate([
        (mc_d_n,  'MC(D)  — norm',        '#3498db'),
        (normalise(mc_dd), 'MC(dD) — norm', '#e67e22'),
        (er_norm, 'ER_norm',               '#27ae60'),
        (act_mex_hat(mc_d_n), 'mex_hat(MC_D)', '#9b59b6'),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(signal, color=color, lw=0.8)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title(label, fontsize=8, fontweight='bold')
        ax.grid(True, alpha=0.25)

    # UPnL placeholder (shown as zero — it's dynamic at runtime)
    ax = fig.add_subplot(gs[1, 4])
    ax.plot(np.zeros(len(close)), color='#95a5a6', lw=0.8)
    ax.set_title('UPnL  (dynamic)', fontsize=8, fontweight='bold')
    ax.grid(True, alpha=0.25)

    # ── Row 2: activation shapes on x ∈ [-3, 3] ──────────────────────────────
    x = np.linspace(-3, 3, 500)
    for col, (name, vec_fn, _, desc, color) in enumerate(ACTIVATIONS):
        ax = fig.add_subplot(gs[2, col])
        ax.plot(x, vec_fn(x), color=color, lw=2.2)
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.set_ylim(-1.6, 1.6)
        ax.set_title(f'{name}\n{desc}', fontsize=8, fontweight='bold', color=color)
        ax.set_xlabel('x', fontsize=7); ax.grid(True, alpha=0.25)

    # ── Row 3: activation applied to live MC_D signal ─────────────────────────
    for col, (name, vec_fn, _, desc, color) in enumerate(ACTIVATIONS):
        ax = fig.add_subplot(gs[3, col])
        ax.plot(vec_fn(mc_d_n), color=color, lw=0.75, alpha=0.9)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title(f'{name}(MC_D)', fontsize=8, color=color)
        ax.grid(True, alpha=0.25)

    fig.suptitle('Activation Functions on Synthetic FX Data', fontsize=14,
                 fontweight='bold')
    out = Path(__file__).parent / 'activation_shapes.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f'  Saved: {out}')
    return fig

# ─── NEAT Config ──────────────────────────────────────────────────────────────

NEAT_CFG = """
[NEAT]
fitness_criterion       = max
fitness_threshold       = 10000
pop_size                = 100
reset_on_extinction     = False
no_fitness_termination  = True

[DefaultGenome]
num_inputs              = 5
num_outputs             = 3
num_hidden              = 0
feed_forward            = True
initial_connection      = full_direct
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh
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
max_stagnation       = 30
species_elitism      = 2

[DefaultReproduction]
elitism            = 3
survival_threshold = 0.25
"""

def make_config():
    tf = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
    tf.write(NEAT_CFG)
    tf.flush()
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      tf.name)
    register_activations(cfg)
    os.unlink(tf.name)
    return cfg

# ─── Fixed Genome Builder (5→5→7→3) ──────────────────────────────────────────

INPUT_IDS  = [-1, -2, -3, -4, -5]
OUTPUT_IDS = [0, 1, 2]
L1_IDS     = [100, 101, 102, 103, 104]
L2_IDS     = [200, 201, 202, 203, 204, 205, 206]

def _make_node(gc, nid, act):
    node = gc.node_gene_type(nid)
    node.bias       = np.random.uniform(-1.5, 1.5)
    node.response   = 1.0
    node.activation = act
    node.aggregation = 'sum'
    return node

def _make_conn(gc, src, dst, innov):
    key  = (src, dst)
    conn = gc.connection_gene_type(key, innovation=innov)
    conn.weight  = np.random.uniform(-2.0, 2.0)
    conn.enabled = True
    return key, conn

def build_genome(config, gid=0):
    genome = config.genome_type(gid)
    genome.fitness = None
    gc = config.genome_config

    # Output nodes (fixed tanh)
    for nid in OUTPUT_IDS:
        genome.nodes[nid] = _make_node(gc, nid, 'tanh')

    # L1 nodes — random activation from all 5
    for nid in L1_IDS:
        act = ACT_NAMES[np.random.randint(len(ACT_NAMES))]
        genome.nodes[nid] = _make_node(gc, nid, act)

    # L2 nodes — random activation from all 5
    for nid in L2_IDS:
        act = ACT_NAMES[np.random.randint(len(ACT_NAMES))]
        genome.nodes[nid] = _make_node(gc, nid, act)

    # Connections with unique innovation numbers
    innov = 0
    # input → L1
    for src in INPUT_IDS:
        for dst in L1_IDS:
            k, c = _make_conn(gc, src, dst, innov); genome.connections[k] = c; innov += 1
    # L1 → L2
    for src in L1_IDS:
        for dst in L2_IDS:
            k, c = _make_conn(gc, src, dst, innov); genome.connections[k] = c; innov += 1
    # L2 → output
    for src in L2_IDS:
        for dst in OUTPUT_IDS:
            k, c = _make_conn(gc, src, dst, innov); genome.connections[k] = c; innov += 1
    # skip: input → output
    for src in INPUT_IDS:
        for dst in OUTPUT_IDS:
            k, c = _make_conn(gc, src, dst, innov); genome.connections[k] = c; innov += 1

    return genome

def build_population(config, pop_size):
    pop = {}
    for i in range(pop_size):
        g = build_genome(config, i)
        g.key = i
        pop[i] = g
    return pop

# ─── Mutation ─────────────────────────────────────────────────────────────────

def mutate(genome):
    # Weights
    for conn in genome.connections.values():
        r = np.random.random()
        if r < 0.10:
            conn.weight = np.random.uniform(-5, 5)
        elif r < 0.80:
            conn.weight = np.clip(conn.weight + np.random.normal(0, 0.5), -6, 6)

    # Biases + activations (hidden only)
    for nid, node in genome.nodes.items():
        if nid in OUTPUT_IDS:
            continue
        r = np.random.random()
        if r < 0.10:
            node.bias = np.random.uniform(-5, 5)
        elif r < 0.70:
            node.bias = np.clip(node.bias + np.random.normal(0, 0.4), -6, 6)
        if np.random.random() < 0.12:
            node.activation = ACT_NAMES[np.random.randint(len(ACT_NAMES))]

# ─── Trading Simulation ───────────────────────────────────────────────────────

MAX_HOLD = 150   # bars before forced exit
SL_PIPS  = 20   # emergency stop loss

def simulate(genome, config, close, ind_arr, pip):
    """
    ind_arr: (n, 4) — [mc_d_n, mc_dd_n, er_norm, mex_hat_n]  (UPnL filled live)
    Returns: (total_pips, n_trades, trade_log)
    """
    net   = neat.nn.FeedForwardNetwork.create(genome, config)
    n     = len(close)
    pos   = 0      # 0=flat 1=long -1=short
    entry = 0.0
    bars_held = 0
    total_pips = 0.0
    n_trades   = 0
    trade_log  = []   # (bar, direction, entry, exit, pips)

    for i in range(n):
        upnl_norm = 0.0
        if pos != 0:
            raw = (close[i] - entry) * pos / pip
            upnl_norm = float(np.clip(raw / 50.0, -1.0, 1.0))
            bars_held += 1

        inp = [ind_arr[i, 0], ind_arr[i, 1], ind_arr[i, 2],
               upnl_norm,     ind_arr[i, 3]]

        out    = net.activate(inp)
        action = int(np.argmax(out))   # 0=BUY 1=SELL 2=EXIT

        # Emergency SL/TP forced exit
        if pos != 0 and (bars_held >= MAX_HOLD or
                         abs((close[i] - entry) / pip) >= SL_PIPS * 2.5):
            pips = (close[i] - entry) * pos / pip
            total_pips += pips
            trade_log.append((i, pos, entry, close[i], pips))
            n_trades += 1
            pos = 0; bars_held = 0

        if pos == 0:
            if action == 0:
                pos = 1; entry = close[i]; bars_held = 0
            elif action == 1:
                pos = -1; entry = close[i]; bars_held = 0
        elif pos == 1:
            if action in (1, 2):
                pips = (close[i] - entry) / pip
                total_pips += pips
                trade_log.append((i, 1, entry, close[i], pips))
                n_trades += 1; pos = 0; bars_held = 0
                if action == 1:   # flip to short
                    pos = -1; entry = close[i]; bars_held = 0
        elif pos == -1:
            if action in (0, 2):
                pips = (entry - close[i]) / pip
                total_pips += pips
                trade_log.append((i, -1, entry, close[i], pips))
                n_trades += 1; pos = 0; bars_held = 0
                if action == 0:   # flip to long
                    pos = 1; entry = close[i]; bars_held = 0

    # Close open position at end
    if pos != 0:
        pips = (close[-1] - entry) * pos / pip
        total_pips += pips
        trade_log.append((n - 1, pos, entry, close[-1], pips))
        n_trades += 1

    return total_pips, n_trades, trade_log

# ─── Fitness ──────────────────────────────────────────────────────────────────

def fitness(genome, config, close, ind_arr, pip):
    pips, n_trades, _ = simulate(genome, config, close, ind_arr, pip)
    if n_trades < 8:
        return -200.0
    return pips / (n_trades ** 0.3)   # reward pips/trade quality

# ─── Island Evolution ─────────────────────────────────────────────────────────

def crossover(g1, g2, config, new_id):
    """Uniform crossover of weights/biases/activations."""
    child = build_genome(config, new_id)
    # Weights
    for key in child.connections:
        if key in g1.connections and key in g2.connections:
            parent = g1 if np.random.random() < 0.5 else g2
            child.connections[key].weight = parent.connections[key].weight
        elif key in g1.connections:
            child.connections[key].weight = g1.connections[key].weight
        elif key in g2.connections:
            child.connections[key].weight = g2.connections[key].weight
    # Nodes
    for nid in child.nodes:
        if nid in g1.nodes and nid in g2.nodes:
            parent = g1 if np.random.random() < 0.5 else g2
            child.nodes[nid].bias       = parent.nodes[nid].bias
            child.nodes[nid].activation = parent.nodes[nid].activation
    return child

def run_evolution(close, ind_arr, pip, generations=80, pop_size=100, n_islands=2,
                  migrate_every=10):
    config = make_config()

    # Init islands
    islands = [build_population(config, pop_size) for _ in range(n_islands)]
    best_ever   = None
    best_fitness = -9999
    history      = []   # (gen, best_fitness, mean_fitness)

    next_id = pop_size * n_islands

    for gen in range(generations):
        gen_best_f = -9999

        for isl_idx, pop in enumerate(islands):
            # Evaluate
            scores = {}
            for gid, g in pop.items():
                f = fitness(g, config, close, ind_arr, pip)
                g.fitness = f
                scores[gid] = f

            sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
            best_f = scores[sorted_ids[0]]

            if best_f > best_fitness:
                best_fitness = best_f
                best_ever    = pop[sorted_ids[0]]

            gen_best_f = max(gen_best_f, best_f)

            # Selection + reproduction
            elite_n   = max(2, pop_size // 10)
            survive_n = max(elite_n, int(pop_size * 0.25))
            survivors = sorted_ids[:survive_n]

            new_pop = {}
            # Elites survive unchanged
            for i, gid in enumerate(sorted_ids[:elite_n]):
                new_pop[next_id] = pop[gid]
                new_pop[next_id].key = next_id
                next_id += 1

            # Fill with offspring
            while len(new_pop) < pop_size:
                p1_id = survivors[np.random.randint(survive_n)]
                p2_id = survivors[np.random.randint(survive_n)]
                child = crossover(pop[p1_id], pop[p2_id], config, next_id)
                mutate(child)
                new_pop[next_id] = child
                next_id += 1

            islands[isl_idx] = new_pop

        # Migration: swap top 5 from each island
        if n_islands > 1 and (gen + 1) % migrate_every == 0:
            for i in range(n_islands):
                j   = (i + 1) % n_islands
                src = islands[i]
                dst = islands[j]
                top5_src = sorted(src, key=lambda k: src[k].fitness or -9999,
                                  reverse=True)[:5]
                for gid in top5_src:
                    g = src[gid]
                    g_copy = build_genome(config, next_id)
                    for k in g_copy.connections:
                        if k in g.connections:
                            g_copy.connections[k].weight = g.connections[k].weight
                    for nid in g_copy.nodes:
                        if nid in g.nodes:
                            g_copy.nodes[nid].bias       = g.nodes[nid].bias
                            g_copy.nodes[nid].activation = g.nodes[nid].activation
                    dst[next_id] = g_copy
                    next_id += 1

        # Compute mean across all islands for logging
        all_fitnesses = [g.fitness for isl in islands for g in isl.values()
                         if g.fitness is not None]
        mean_f = np.mean(all_fitnesses) if all_fitnesses else 0.0
        history.append((gen, gen_best_f, mean_f))

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f'  Gen {gen+1:3d}/{generations}  best={gen_best_f:7.1f}  mean={mean_f:7.1f}')

    return best_ever, config, history

# ─── Results Plot ─────────────────────────────────────────────────────────────

def plot_results(close, ind_arr, pip, best_genome, config, history):
    gens   = [h[0] for h in history]
    bests  = [h[1] for h in history]
    means  = [h[2] for h in history]

    _, _, trade_log = simulate(best_genome, config, close, ind_arr, pip)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('NEAT Training Results  (5→5→7→3, activations: tanh/sin/cos/gauss/mex_hat)',
                 fontsize=12, fontweight='bold')

    # ── Price + trades ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(close, color='#2c3e50', lw=0.8, label='Price', zorder=1)
    ax.axhline(np.mean(close), color='gray', lw=0.5, ls='--', alpha=0.6)

    long_wins, long_loss, short_wins, short_loss = [], [], [], []
    for (bar, direction, entry_p, exit_p, pips) in trade_log:
        # Find entry bar (approximate via price proximity)
        # We stored exit bar; search backward for entry
        color = '#27ae60' if pips > 0 else '#e74c3c'
        marker = '^' if direction == 1 else 'v'
        ax.scatter(bar, exit_p, marker=marker, color=color, s=40, zorder=3,
                   alpha=0.85)

    # Legend proxies
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color='#27ae60', marker='^', ls='None', ms=8, label='Long exit (profit)'),
        Line2D([0],[0], color='#e74c3c', marker='^', ls='None', ms=8, label='Long exit (loss)'),
        Line2D([0],[0], color='#27ae60', marker='v', ls='None', ms=8, label='Short exit (profit)'),
        Line2D([0],[0], color='#e74c3c', marker='v', ls='None', ms=8, label='Short exit (loss)'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='upper left')
    ax.set_title(f'Best genome trades  ({len(trade_log)} trades)', fontsize=10)
    ax.set_ylabel('Price'); ax.grid(True, alpha=0.2)

    # ── Fitness curve ─────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(gens, bests,  color='#27ae60', lw=2.0, label='Best fitness')
    ax2.plot(gens, means,  color='#3498db', lw=1.2, ls='--', alpha=0.8, label='Mean fitness')
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_title('Fitness over generations', fontsize=10)
    ax2.set_xlabel('Generation'); ax2.set_ylabel('Fitness (pips/trade^0.7)')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    out = Path(__file__).parent / 'training_results.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f'  Saved: {out}')

    # Print activation usage in best genome
    print('\n  Best genome hidden node activations:')
    for nid, node in sorted(best_genome.nodes.items()):
        if nid not in OUTPUT_IDS:
            layer = 'L1' if nid in L1_IDS else 'L2'
            print(f'    node {nid} ({layer}): {node.activation}')

    # P&L summary
    total = sum(t[4] for t in trade_log)
    wins  = sum(1 for t in trade_log if t[4] > 0)
    print(f'\n  Trades: {len(trade_log)}  |  Total: {total:.1f}p  |  Win rate: {wins/max(1,len(trade_log))*100:.0f}%')

    return fig

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('── Generating synthetic price ────────────────────────────────────')
    open_, high, low, close, pip = generate_price()
    print(f'  {len(close)} bars, range {close.min():.4f}–{close.max():.4f}')

    print('── Computing indicators ──────────────────────────────────────────')
    asi           = compute_asi(open_, high, low, close)
    mc_d, mc_dd   = compute_mc(asi)
    er_norm       = compute_er(close)

    mc_d_n    = normalise(mc_d)
    mc_dd_n   = normalise(mc_dd)
    mex_hat_n = act_mex_hat(mc_d_n)

    # Indicator array for NEAT (UPnL = col 3 is filled live; here init to 0)
    ind_arr = np.column_stack([mc_d_n, mc_dd_n, er_norm, mex_hat_n]).astype(np.float32)

    print('── Plotting indicator & activation shapes ────────────────────────')
    plot_indicators(close, mc_d, mc_dd, er_norm, pip)

    print('── Training NEAT ─────────────────────────────────────────────────')
    print('   topology: 5 inputs → 5(L1) → 7(L2) → 3 outputs')
    print('   activations: tanh / sin / cos / gauss / mex_hat')
    best_genome, config, history = run_evolution(
        close, ind_arr, pip,
        generations=40,
        pop_size=40,
        n_islands=2,
        migrate_every=8,
    )

    print('\n── Plotting training results ─────────────────────────────────────')
    plot_results(close, ind_arr, pip, best_genome, config, history)

    # Save best genome
    import pickle
    out_pkl = Path(__file__).parent / 'best_genome.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(best_genome, f)
    print(f'\n  Best genome saved: {out_pkl}')

if __name__ == '__main__':
    main()
