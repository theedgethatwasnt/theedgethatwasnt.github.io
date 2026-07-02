#!/usr/bin/env python3
"""
Activation Function Experiment v3 — Seeded Population + 200 Generations
=========================================================================
Strategy: seed the initial population with one "pure" genome per activation
(all 12 hidden nodes set to that activation), then let NEAT cross-breed them.

  Gen 1 already contains:  pure-tanh, pure-morlet_im, pure-sech, pure-haar ...
  Crossover then builds:   L1=morlet_im + L2=sech, L1=dog + L2=gauss, etc.
  200 gens gives time to:  explore combinations, not just individual activations

Activations tested (17 total, grouped by concept):

  BASELINE       tanh, sin, cos, gauss
  RICKER         mex_hat   — 2nd derivative of Gaussian (no oscillation)
  COMPLEX MORLET morlet_re — exp(-x²/2)·cos(5x)  [real part]
                 morlet_im — exp(-x²/2)·sin(5x)  [imaginary part, 90° phase]
  GABOR          gabor     — exp(-2x²)·cos(2πx)  [tighter envelope than Morlet]
  DOG            dog       — exp(-x²/2) − 0.5·exp(-x²/8)  [bandpass, Diff of Gaussians]
  SINC/SHANNON   sinc      — sin(πx)/(πx)  [ideal lowpass wavelet]
  CHIRP          chirp     — sin(x²)  [increasing frequency with magnitude]
  SECH           sech      — 1/cosh(x)  [smooth bump, heavier tails than Gauss]
  HAAR           haar      — step-based edge detector (discontinuous OK for NEAT)

LEARNABLE WAVELET NOTE:
  Any wavelet activation f becomes learnable via NEAT's weight/bias:
    node output = f(w·input + b)
  → w controls SCALE (1/frequency): large |w| = high freq, narrow support
  → b controls TRANSLATION: shift of the wavelet centre in input space
  NEAT discovers optimal scale & translation through weight evolution —
  no extra learnable parameters needed.

COMPLEX DECOMPOSITION:
  morlet_re + morlet_im together span the complex Morlet:
    Ψ(x) = morlet_re + i·morlet_im = exp(-x²/2)·exp(i·5x)
  Two separate nodes using each activation recover both amplitude and phase:
    amplitude = sqrt(morlet_re² + morlet_im²)  [envelope]
    phase     = atan2(morlet_im, morlet_re)     [instantaneous phase]

Fixed topology: 5 inputs → 5 hidden (L1) → 7 hidden (L2) → 3 outputs
Inputs: MC_D, MC_dD, ER_norm, UPnL, mex_hat(MC_D)
"""

import math
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import neat
import numpy as np

# ─── Synthetic Price ──────────────────────────────────────────────────────────

def generate_price(n_bars=600, period=200, amplitude_pips=80,
                   center=1.1000, pip=0.0001, noise_pips=2.5, seed=42):
    rng = np.random.RandomState(seed)
    t   = np.arange(n_bars, dtype=np.float64)
    clean = center + amplitude_pips * pip * np.sin(2 * np.pi * t / period)
    noise = rng.randn(n_bars) * noise_pips * pip
    close = clean + noise
    high  = close + rng.uniform(0.5, 3, n_bars) * pip
    low   = close - rng.uniform(0.5, 3, n_bars) * pip
    open_ = np.roll(close, 1); open_[0] = close[0]
    return open_, high, low, close, pip

# ─── Indicators ───────────────────────────────────────────────────────────────

def compute_asi(open_, high, low, close):
    n = len(close); asi = np.zeros(n)
    for i in range(1, n):
        tr = max(high[i], close[i-1]) - min(low[i], close[i-1])
        if tr > 1e-12:
            asi[i] = asi[i-1] + (close[i] - close[i-1]) / tr
        else:
            asi[i] = asi[i-1]
    return asi

def ema(x, p):
    a = 2.0 / (p + 1); out = np.empty_like(x); out[0] = x[0]
    for i in range(1, len(x)): out[i] = a * x[i] + (1-a) * out[i-1]
    return out

def compute_mc(asi):
    sma = np.convolve(asi, np.ones(5)/5, mode='same')
    mc_d  = ema(sma, 3) - ema(sma, 5)
    mc_dd = np.diff(mc_d, prepend=mc_d[0])
    return mc_d, mc_dd

def compute_er(close, period=60):
    n = len(close); er = np.zeros(n)
    for i in range(period, n):
        d = abs(close[i] - close[i-period])
        v = np.sum(np.abs(np.diff(close[i-period:i+1])))
        er[i] = d/v if v > 1e-12 else 0.0
    return np.arctan(er * 5.0) / (math.pi / 2.0)

def normalise(x, clip=3.0):
    return np.clip(x / (np.std(x) + 1e-10), -clip, clip)

# ─── Activation Functions ─────────────────────────────────────────────────────
# Each entry: (name, numpy_vec_fn, scalar_fn, group, description, hex_color)

def _morlet_re_np(x):  return np.exp(-x**2/2) * np.cos(5*x)
def _morlet_im_np(x):  return np.exp(-x**2/2) * np.sin(5*x)
def _gabor_np(x):      return np.exp(-2*x**2) * np.cos(2*math.pi*x)
def _dog_np(x):        return np.exp(-x**2/2) - 0.5*np.exp(-x**2/8)
def _sinc_np(x):       return np.where(np.abs(x)<1e-9, 1.0, np.sin(math.pi*x)/(math.pi*x))
def _chirp_np(x):      return np.sin(x**2)
def _sech_np(x):       return 1.0 / np.cosh(x)
def _haar_np(x):       return np.where((x>=0)&(x<0.5), 1.0,
                                np.where((x>=0.5)&(x<1.0), -1.0, 0.0))

def _tanh_s(x):     return math.tanh(x)
def _sin_s(x):      return math.sin(x)
def _cos_s(x):      return math.cos(x)
def _gauss_s(x):    return math.exp(-x*x)
def _mex_hat_s(x):  return (1.0-x*x)*math.exp(-x*x/2.0)
def _morlet_re_s(x):return math.exp(-x*x/2)*math.cos(5*x)
def _morlet_im_s(x):return math.exp(-x*x/2)*math.sin(5*x)
def _gabor_s(x):    return math.exp(-2*x*x)*math.cos(2*math.pi*x)
def _dog_s(x):      return math.exp(-x*x/2) - 0.5*math.exp(-x*x/8)
def _sinc_s(x):     return 1.0 if abs(x)<1e-9 else math.sin(math.pi*x)/(math.pi*x)
def _chirp_s(x):    return math.sin(x*x)
def _sech_s(x):     return 1.0/math.cosh(x)
def _haar_s(x):
    if 0.0 <= x < 0.5: return 1.0
    if 0.5 <= x < 1.0: return -1.0
    return 0.0

def _sigmoid_np(x):  return 1.0 / (1.0 + np.exp(-x))
def _relu_np(x):     return np.maximum(0.0, x)
def _elu_np(x):      return np.where(x >= 0, x, np.exp(x) - 1.0)
def _swish_np(x):    return x / (1.0 + np.exp(-x))
def _sigmoid_s(x):   return 1.0 / (1.0 + math.exp(-x))
def _relu_s(x):      return max(0.0, x)
def _elu_s(x):       return x if x >= 0.0 else math.exp(x) - 1.0
def _swish_s(x):     return x / (1.0 + math.exp(-x))

# (name, vec_fn, scalar_fn, group, description, color)
ACTIVATIONS = [
    # ── Baseline ──────────────────────────────────────────────────────────────
    ('tanh',      np.tanh,       _tanh_s,      'Baseline',      'S-curve (monotone)',            '#e74c3c'),
    ('sin',       np.sin,        _sin_s,        'Baseline',      'Pure frequency',                '#f39c12'),
    ('cos',       np.cos,        _cos_s,        'Baseline',      'Pure freq (phase shifted)',      '#e67e22'),
    ('gauss',     lambda x: np.exp(-x**2), _gauss_s, 'Baseline', 'Pure localization',            '#3498db'),
    # ── Mexican Hat / Ricker ──────────────────────────────────────────────────
    ('mex_hat',   lambda x:(1-x**2)*np.exp(-x**2/2),  _mex_hat_s,  'Ricker',     'Transition detector',   '#9b59b6'),
    # ── Complex Morlet (real + imaginary parts) ───────────────────────────────
    ('morlet_re', _morlet_re_np, _morlet_re_s, 'Morlet',        'Freq+loc (even/cosine)',        '#1abc9c'),
    ('morlet_im', _morlet_im_np, _morlet_im_s, 'Morlet',        'Freq+loc (odd/sine, 90° phase)','#16a085'),
    # ── Gabor (tighter envelope) ──────────────────────────────────────────────
    ('gabor',     _gabor_np,     _gabor_s,     'Gabor',         'exp(-2x²)·cos(2πx) tight',      '#2ecc71'),
    # ── Difference of Gaussians (bandpass) ────────────────────────────────────
    ('dog',       _dog_np,       _dog_s,        'DoG',           'Bandpass (Mexican hat family)', '#27ae60'),
    # ── Sinc / Shannon wavelet ────────────────────────────────────────────────
    ('sinc',      _sinc_np,      _sinc_s,       'Sinc',          'Ideal lowpass / Shannon',       '#8e44ad'),
    # ── Chirp (freq scales with magnitude) ────────────────────────────────────
    ('chirp',     _chirp_np,     _chirp_s,      'Chirp',         'sin(x²): freq grows with |x|', '#d35400'),
    # ── Sech (heavier-tailed bump) ────────────────────────────────────────────
    ('sech',      _sech_np,      _sech_s,       'Sech',          '1/cosh(x): soft bump',          '#c0392b'),
    # ── Haar (edge / step detector) ───────────────────────────────────────────
    ('haar',      _haar_np,      _haar_s,       'Haar',          'Step edge detector',            '#7f8c8d'),
    # ── Classic / common ──────────────────────────────────────────────────────
    ('sigmoid',   _sigmoid_np,   _sigmoid_s,    'Classic',       'Logistic (0→1)',                '#e74c3c'),
    ('relu',      _relu_np,      _relu_s,       'Classic',       'Rectified linear',              '#c0392b'),
    ('elu',       _elu_np,       _elu_s,        'Classic',       'Exp linear (neg lobe)',         '#e91e63'),
    ('swish',     _swish_np,     _swish_s,      'Classic',       'x·sigmoid(x) smooth',          '#ad1457'),
]

ACT_NAMES   = [a[0] for a in ACTIVATIONS]
ACT_SCALAR  = {a[0]: a[2] for a in ACTIVATIONS}
ACT_VEC     = {a[0]: a[1] for a in ACTIVATIONS}
ACT_GROUP   = {a[0]: a[3] for a in ACTIVATIONS}
ACT_DESC    = {a[0]: a[4] for a in ACTIVATIONS}
ACT_COLOR   = {a[0]: a[5] for a in ACTIVATIONS}

def register_activations(config):
    for name, _, scalar_fn, *_ in ACTIVATIONS:
        try: config.genome_config.add_activation(name, scalar_fn)
        except RuntimeError: pass

# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_activations(close, mc_d, mc_dd, er_norm, pip, out_dir):
    n_act = len(ACTIVATIONS)
    cols  = 5
    rows_shapes = math.ceil(n_act / cols)   # 3 rows for 13 activations

    # Total rows: price(1) + indicators(1) + shapes(rows_shapes) + applied(rows_shapes)
    total_rows = 1 + 1 + rows_shapes + rows_shapes
    fig = plt.figure(figsize=(cols * 3.4, total_rows * 2.6))
    gs  = gridspec.GridSpec(total_rows, cols, hspace=0.60, wspace=0.38,
                            top=0.95, bottom=0.03)

    # ── Row 0: price ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ax.plot(close, color='#2c3e50', lw=0.9)
    ax.axhline(np.mean(close), color='gray', lw=0.5, ls='--', alpha=0.6)
    ax.set_title('Synthetic EUR/USD  (sine + Gaussian noise)', fontweight='bold')
    ax.set_ylabel('Price'); ax.grid(True, alpha=0.25)

    # ── Row 1: indicators ─────────────────────────────────────────────────────
    mc_d_n   = normalise(mc_d)
    mc_dd_n  = normalise(mc_dd)
    mex_n    = (lambda x: (1-x**2)*np.exp(-x**2/2))(mc_d_n)

    for col, (sig, label, color) in enumerate([
        (mc_d_n,           'MC(D) [input 1]',     '#3498db'),
        (normalise(mc_dd), 'MC(dD) [input 2]',    '#e67e22'),
        (er_norm,          'ER_norm [input 3]',   '#27ae60'),
        (mex_n,            'mex_hat(MC_D) [in 5]','#9b59b6'),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(sig, color=color, lw=0.8); ax.axhline(0, color='gray', lw=0.4)
        ax.set_title(label, fontsize=8, fontweight='bold'); ax.grid(True, alpha=0.25)
    ax = fig.add_subplot(gs[1, 4])
    ax.plot(np.zeros(len(close)), color='#95a5a6', lw=0.8)
    ax.set_title('UPnL [input 4]\n(dynamic)', fontsize=8); ax.grid(True, alpha=0.25)

    # ── Rows 2…: activation shapes ────────────────────────────────────────────
    x = np.linspace(-3, 3, 600)
    base_row = 2
    for i, (name, vec_fn, _, group, desc, color) in enumerate(ACTIVATIONS):
        r = base_row + i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c])
        ax.plot(x, vec_fn(x), color=color, lw=2.0)
        ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
        ax.set_ylim(-1.7, 1.7)
        ax.set_title(f'{name}\n[{group}] {desc}', fontsize=6.5, fontweight='bold', color=color)
        ax.set_xlabel('x', fontsize=7); ax.grid(True, alpha=0.25)

    # ── Rows …: activations applied to MC_D ───────────────────────────────────
    applied_base = base_row + rows_shapes
    for i, (name, vec_fn, _, group, desc, color) in enumerate(ACTIVATIONS):
        r = applied_base + i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c])
        ax.plot(vec_fn(mc_d_n), color=color, lw=0.75, alpha=0.9)
        ax.axhline(0, color='gray', lw=0.4)
        ax.set_title(f'{name}(MC_D)', fontsize=7.5, color=color)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        'Expanded Wavelet Activation Suite  —  shape curves + applied to MC_D signal',
        fontsize=12, fontweight='bold'
    )
    out = Path(out_dir) / 'activation_shapes_v3.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f'  Saved: {out}')
    plt.close(fig)

# ─── NEAT Config ──────────────────────────────────────────────────────────────

NEAT_CFG = """
[NEAT]
fitness_criterion       = max
fitness_threshold       = 10000
pop_size                = 40
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
                      neat.DefaultSpeciesSet, neat.DefaultStagnation, tf.name)
    register_activations(cfg)
    os.unlink(tf.name)
    return cfg

# ─── Fixed Topology Genome (5→5→7→3) ─────────────────────────────────────────

INPUT_IDS  = [-1, -2, -3, -4, -5]
OUTPUT_IDS = [0, 1, 2]
L1_IDS     = [100, 101, 102, 103, 104]
L2_IDS     = [200, 201, 202, 203, 204, 205, 206]

def _node(gc, nid, act):
    n = gc.node_gene_type(nid)
    n.bias = np.random.uniform(-1.5, 1.5)
    n.response = 1.0; n.activation = act; n.aggregation = 'sum'
    return n

def _conn(gc, src, dst, innov):
    key = (src, dst)
    c = gc.connection_gene_type(key, innovation=innov)
    c.weight = np.random.uniform(-2.0, 2.0); c.enabled = True
    return key, c

def build_genome(config, gid=0):
    g = config.genome_type(gid); g.fitness = None
    gc = config.genome_config

    for nid in OUTPUT_IDS:
        g.nodes[nid] = _node(gc, nid, 'tanh')
    for nid in L1_IDS:
        g.nodes[nid] = _node(gc, nid, ACT_NAMES[np.random.randint(len(ACT_NAMES))])
    for nid in L2_IDS:
        g.nodes[nid] = _node(gc, nid, ACT_NAMES[np.random.randint(len(ACT_NAMES))])

    innov = 0
    for src in INPUT_IDS:
        for dst in L1_IDS:
            k, c = _conn(gc, src, dst, innov); g.connections[k] = c; innov += 1
    for src in L1_IDS:
        for dst in L2_IDS:
            k, c = _conn(gc, src, dst, innov); g.connections[k] = c; innov += 1
    for src in L2_IDS:
        for dst in OUTPUT_IDS:
            k, c = _conn(gc, src, dst, innov); g.connections[k] = c; innov += 1
    for src in INPUT_IDS:     # skip: input → output
        for dst in OUTPUT_IDS:
            k, c = _conn(gc, src, dst, innov); g.connections[k] = c; innov += 1
    return g

def build_pure_seed(config, activation_name, gid):
    """All hidden nodes use the same activation — one seed per activation type."""
    g = build_genome(config, gid)
    for nid in L1_IDS + L2_IDS:
        g.nodes[nid].activation = activation_name
    return g

def build_seeded_pop(config, pop_size, act_names, island_idx=0, verbose=False):
    """
    Seed population: one pure-activation genome per act type, rest random.
    Each island gets different random weights so seeds diverge independently.
    """
    pop  = {}
    gid  = island_idx * pop_size   # non-overlapping IDs across islands

    # One seed per activation
    for name in act_names:
        g = build_pure_seed(config, name, gid)
        g.key = gid
        pop[gid] = g
        gid += 1

    # Fill remainder with random genomes
    while len(pop) < pop_size:
        g = build_genome(config, gid); g.key = gid
        pop[gid] = g; gid += 1

    if verbose:
        print(f'   Island {island_idx}: {len(act_names)} pure seeds + '
              f'{pop_size - len(act_names)} random  (total {pop_size})')
    return pop

# ─── Mutation ─────────────────────────────────────────────────────────────────

def mutate(genome):
    for conn in genome.connections.values():
        r = np.random.random()
        if r < 0.10: conn.weight = np.random.uniform(-5, 5)
        elif r < 0.80: conn.weight = np.clip(conn.weight + np.random.normal(0, 0.5), -6, 6)
    for nid, node in genome.nodes.items():
        if nid in OUTPUT_IDS: continue
        r = np.random.random()
        if r < 0.10: node.bias = np.random.uniform(-5, 5)
        elif r < 0.70: node.bias = np.clip(node.bias + np.random.normal(0, 0.4), -6, 6)
        if np.random.random() < 0.12:
            node.activation = ACT_NAMES[np.random.randint(len(ACT_NAMES))]

# ─── Trading Simulation ───────────────────────────────────────────────────────

MAX_HOLD = 150
SL_PIPS  = 50

def simulate(genome, config, close, ind_arr, pip):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    n = len(close); pos = 0; entry = 0.0; bars_held = 0
    total_pips = 0.0; n_trades = 0; trade_log = []

    for i in range(n):
        upnl = float(np.clip((close[i]-entry)*pos/pip/50.0, -1.0, 1.0)) if pos else 0.0
        if pos: bars_held += 1

        inp = [float(ind_arr[i,0]), float(ind_arr[i,1]), float(ind_arr[i,2]),
               upnl,               float(ind_arr[i,3])]
        action = int(np.argmax(net.activate(inp)))

        # Forced exit
        if pos and (bars_held >= MAX_HOLD or abs((close[i]-entry)/pip) >= SL_PIPS*2):
            pips = (close[i]-entry)*pos/pip
            total_pips += pips; trade_log.append((i, pos, entry, close[i], pips))
            n_trades += 1; pos = 0; bars_held = 0

        if pos == 0:
            if action == 0: pos = 1; entry = close[i]; bars_held = 0
            elif action == 1: pos = -1; entry = close[i]; bars_held = 0
        elif pos == 1:
            if action in (1, 2):
                pips = (close[i]-entry)/pip; total_pips += pips
                trade_log.append((i, 1, entry, close[i], pips))
                n_trades += 1; pos = 0; bars_held = 0
                if action == 1: pos = -1; entry = close[i]; bars_held = 0
        elif pos == -1:
            if action in (0, 2):
                pips = (entry-close[i])/pip; total_pips += pips
                trade_log.append((i, -1, entry, close[i], pips))
                n_trades += 1; pos = 0; bars_held = 0
                if action == 0: pos = 1; entry = close[i]; bars_held = 0

    if pos:
        pips = (close[-1]-entry)*pos/pip; total_pips += pips
        trade_log.append((n-1, pos, entry, close[-1], pips)); n_trades += 1

    return total_pips, n_trades, trade_log

def fitness(genome, config, close, ind_arr, pip):
    pips, nt, _ = simulate(genome, config, close, ind_arr, pip)
    if nt < 8: return -200.0
    return pips / (nt ** 0.3)

# ─── Island Evolution ─────────────────────────────────────────────────────────

def crossover(g1, g2, config, new_id):
    child = build_genome(config, new_id)
    for key in child.connections:
        p = g1 if np.random.random() < 0.5 else g2
        if key in p.connections:
            child.connections[key].weight = p.connections[key].weight
    for nid in child.nodes:
        if nid in OUTPUT_IDS: continue
        p = g1 if np.random.random() < 0.5 else g2
        if nid in p.nodes:
            child.nodes[nid].bias       = p.nodes[nid].bias
            child.nodes[nid].activation = p.nodes[nid].activation
    return child

def run_evolution(close, ind_arr, pip, generations=200, pop_size=40,
                  n_islands=2, migrate_every=20):
    config   = make_config()
    islands  = [
        build_seeded_pop(config, pop_size, ACT_NAMES,
                         island_idx=i, verbose=True)
        for i in range(n_islands)
    ]
    best_ever = None; best_f = -9999
    history   = []; next_id = pop_size * n_islands

    # ── Log seed fitness (gen 0 snapshot) ─────────────────────────────────────
    print('\n  Seed fitness (island 0, pure-activation genomes):')
    seed_scores = []
    for gid, g in list(islands[0].items())[:len(ACT_NAMES)]:
        act = g.nodes[L1_IDS[0]].activation   # all nodes same activation
        f   = fitness(g, config, close, ind_arr, pip)
        g.fitness = f
        seed_scores.append((act, f))
    seed_scores.sort(key=lambda x: x[1], reverse=True)
    for act, f in seed_scores:
        bar = '█' * max(0, int((f + 200) / 20))
        print(f'    {act:<12s} {f:7.1f}  {bar}')
    print()

    for gen in range(generations):
        gen_best = -9999

        for idx, pop in enumerate(islands):
            for g in pop.values():
                g.fitness = fitness(g, config, close, ind_arr, pip)

            sorted_ids = sorted(pop, key=lambda k: pop[k].fitness, reverse=True)
            top_f = pop[sorted_ids[0]].fitness

            if top_f > best_f:
                best_f = top_f; best_ever = pop[sorted_ids[0]]
            gen_best = max(gen_best, top_f)

            elite_n  = max(2, pop_size // 10)
            survive_n = max(elite_n, int(pop_size * 0.25))
            survivors = sorted_ids[:survive_n]

            new_pop = {}
            for gid in sorted_ids[:elite_n]:
                new_pop[next_id] = pop[gid]; new_pop[next_id].key = next_id; next_id += 1
            while len(new_pop) < pop_size:
                p1 = pop[survivors[np.random.randint(survive_n)]]
                p2 = pop[survivors[np.random.randint(survive_n)]]
                child = crossover(p1, p2, config, next_id)
                mutate(child); new_pop[next_id] = child; next_id += 1
            islands[idx] = new_pop

        # Migration
        if n_islands > 1 and (gen + 1) % migrate_every == 0:
            for i in range(n_islands):
                j = (i+1) % n_islands; src = islands[i]; dst = islands[j]
                top5 = sorted(src, key=lambda k: src[k].fitness or -9999, reverse=True)[:5]
                for gid in top5:
                    g = src[gid]; clone = build_genome(config, next_id)
                    for k in clone.connections:
                        if k in g.connections: clone.connections[k].weight = g.connections[k].weight
                    for nid in clone.nodes:
                        if nid in g.nodes:
                            clone.nodes[nid].bias       = g.nodes[nid].bias
                            clone.nodes[nid].activation = g.nodes[nid].activation
                    dst[next_id] = clone; next_id += 1

        all_f = [g.fitness for isl in islands for g in isl.values() if g.fitness is not None]
        mean_f = float(np.mean(all_f)) if all_f else 0.0
        history.append((gen, gen_best, mean_f))

        if (gen+1) % 10 == 0 or gen == 0:
            print(f'  Gen {gen+1:3d}/{generations}  best={gen_best:7.1f}  mean={mean_f:7.1f}')

    return best_ever, config, history

# ─── Results Plot ─────────────────────────────────────────────────────────────

def plot_results(close, ind_arr, pip, best_genome, config, history, out_dir):
    gens  = [h[0] for h in history]
    bests = [h[1] for h in history]
    means = [h[2] for h in history]
    _, _, trade_log = simulate(best_genome, config, close, ind_arr, pip)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('NEAT v3 — 17 activations, seeded pop, 200 gens (5→5→7→3)',
                 fontsize=9, fontweight='bold')

    # Price + trades
    ax = axes[0]
    ax.plot(close, color='#2c3e50', lw=0.8, zorder=1)
    ax.axhline(np.mean(close), color='gray', lw=0.5, ls='--', alpha=0.5)
    for (bar, direction, ep, xp, pips) in trade_log:
        color  = '#27ae60' if pips > 0 else '#e74c3c'
        marker = '^' if direction == 1 else 'v'
        ax.scatter(bar, xp, marker=marker, color=color, s=45, zorder=3, alpha=0.85)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color='#27ae60', marker='^', ls='None', ms=8, label='Long exit ✓'),
        Line2D([0],[0], color='#e74c3c', marker='^', ls='None', ms=8, label='Long exit ✗'),
        Line2D([0],[0], color='#27ae60', marker='v', ls='None', ms=8, label='Short exit ✓'),
        Line2D([0],[0], color='#e74c3c', marker='v', ls='None', ms=8, label='Short exit ✗'),
    ], fontsize=8, loc='upper left')
    total = sum(t[4] for t in trade_log)
    wins  = sum(1 for t in trade_log if t[4] > 0)
    ax.set_title(f'Best genome  |  {len(trade_log)} trades  |  {total:.0f}p total  |  '
                 f'{wins/max(1,len(trade_log))*100:.0f}% win rate', fontsize=10)
    ax.set_ylabel('Price'); ax.grid(True, alpha=0.2)

    # Fitness curve
    ax2 = axes[1]
    ax2.plot(gens, bests, color='#27ae60', lw=2.0, label='Best')
    ax2.plot(gens, means, color='#3498db', lw=1.2, ls='--', alpha=0.8, label='Mean')
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_title('Fitness over generations', fontsize=10)
    ax2.set_xlabel('Generation'); ax2.set_ylabel('Fitness')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    out = Path(out_dir) / 'training_results_v3.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f'  Saved: {out}')
    plt.close(fig)

    # ── Activation usage heatmap ───────────────────────────────────────────────
    l1_acts = [best_genome.nodes[nid].activation for nid in L1_IDS]
    l2_acts = [best_genome.nodes[nid].activation for nid in L2_IDS]

    fig2, (ax_l1, ax_l2) = plt.subplots(1, 2, figsize=(14, 4))
    fig2.suptitle('Best Genome — Activation Distribution per Layer', fontweight='bold')

    for ax, acts, title in [(ax_l1, l1_acts, f'L1 ({len(L1_IDS)} nodes)'),
                             (ax_l2, l2_acts, f'L2 ({len(L2_IDS)} nodes)')]:
        from collections import Counter
        counts = Counter(acts)
        names  = [n for n in ACT_NAMES if counts.get(n, 0) > 0]
        vals   = [counts[n] for n in names]
        colors = [ACT_COLOR[n] for n in names]
        bars = ax.bar(names, vals, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Count'); ax.set_ylim(0, max(vals)+1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.05, str(v),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=35, labelsize=8)
        ax.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    out2 = Path(out_dir) / 'activation_usage_v3.png'
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    print(f'  Saved: {out2}')
    plt.close(fig2)

    # Print table
    print('\n  ┌──────────────┬──────────────┬───────────────────────────────────┐')
    print(  '  │  Node        │  Layer       │  Activation                       │')
    print(  '  ├──────────────┼──────────────┼───────────────────────────────────┤')
    for nid in L1_IDS + L2_IDS:
        node = best_genome.nodes[nid]
        layer = 'L1' if nid in L1_IDS else 'L2'
        group = ACT_GROUP.get(node.activation, '')
        desc  = ACT_DESC.get(node.activation, '')
        print(f'  │  node {nid:<6d} │  {layer:<10}  │  {node.activation:<10} [{group}] {desc[:20]:<20}│')
    print(  '  └──────────────┴──────────────┴───────────────────────────────────┘')

# ─── Main ─────────────────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).parent

def main():
    print('── Generating synthetic price ──────────────────────────────────────')
    open_, high, low, close, pip = generate_price()
    print(f'  {len(close)} bars  |  {len(ACTIVATIONS)} activations: {ACT_NAMES}')

    print('── Computing indicators ────────────────────────────────────────────')
    asi          = compute_asi(open_, high, low, close)
    mc_d, mc_dd  = compute_mc(asi)
    er_norm      = compute_er(close)
    mc_d_n       = normalise(mc_d)
    mc_dd_n      = normalise(mc_dd)
    mex_hat_n    = (lambda x: (1-x**2)*np.exp(-x**2/2))(mc_d_n)
    ind_arr      = np.column_stack([mc_d_n, mc_dd_n, er_norm, mex_hat_n]).astype(np.float32)

    print('── Plotting activation shapes ──────────────────────────────────────')
    plot_activations(close, mc_d, mc_dd, er_norm, pip, OUT_DIR)

    print('── Training NEAT ───────────────────────────────────────────────────')
    print(f'   topology: 5→5→7→3  |  {len(ACTIVATIONS)} activation choices')
    best_genome, config, history = run_evolution(
        close, ind_arr, pip,
        generations=200, pop_size=40, n_islands=2, migrate_every=20
    )

    print('\n── Plotting results ────────────────────────────────────────────────')
    plot_results(close, ind_arr, pip, best_genome, config, history, OUT_DIR)

    import pickle
    pkl = OUT_DIR / 'best_genome_v3.pkl'
    with open(pkl, 'wb') as f: pickle.dump(best_genome, f)
    print(f'\n  Genome saved: {pkl}')

if __name__ == '__main__':
    main()
