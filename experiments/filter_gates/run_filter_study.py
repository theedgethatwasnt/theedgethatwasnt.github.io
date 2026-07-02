#!/usr/bin/env python3
"""
Filter Gate Study
=================
Apply autoresearch-discovered entry filters as gates on existing trained genomes.
Tests all 8 combinations (2^3) of computable filters on M5 parquet data.

Filters (computable from mid_close only):
  F1  BB squeeze      — BB bandwidth < threshold (volatility compression)
  F2  ER14 ranging    — Kaufman ER < 0.35 (low-trending / ranging market)
  F3  RangePosition   — price at top/bottom 15% of 30-bar range (contrarian setup)

  Note: Spread/ATR filter requires tick/OHLC data — not available in M5 parquets.

Genomes tested:
  iron_s5ft_EUR_GBP, iron_s5ft_CAD_JPY   (4 inputs, account 001)
  iron_v3_EUR_GBP,   iron_v3_CAD_JPY     (4 inputs, account 009)
  asi_mc_v2_best                          (3 inputs, account 008, 12 pairs)
"""

import math
import os
import pickle
import tempfile
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import neat
import numpy as np
import pandas as pd

ROOT    = Path(__file__).parents[3]          # fx-core/
DATA    = ROOT / 'data' / 'curator_identical'
MODELS  = ROOT / 'models'
OUT     = Path(__file__).parent / 'results'
OUT.mkdir(exist_ok=True)

PIP = {'EUR_GBP': 0.0001, 'CAD_JPY': 0.01, 'EUR_USD': 0.0001,
       'GBP_USD': 0.0001, 'USD_JPY': 0.01, 'AUD_USD': 0.0001,
       'EUR_JPY': 0.01,   'NZD_USD': 0.0001, 'GBP_JPY': 0.01}

# ─── Activations ──────────────────────────────────────────────────────────────

def _gauss(x):    return math.exp(-x * x)
def _sin(x):      return math.sin(x)
def _cos(x):      return math.cos(x)
def _tanh(x):     return math.tanh(x)
def _sigmoid(x):  return 1.0 / (1.0 + math.exp(-x))
def _relu(x):     return max(0.0, x)
def _sech(x):     return 1.0 / math.cosh(x)
def _mex_hat(x):  return (1.0 - x*x) * math.exp(-x*x / 2.0)
def _morlet_re(x):return math.exp(-x*x/2) * math.cos(5*x)
def _morlet_im(x):return math.exp(-x*x/2) * math.sin(5*x)
def _gabor(x):    return math.exp(-2*x*x) * math.cos(2*math.pi*x)
def _dog(x):      return math.exp(-x*x/2) - 0.5*math.exp(-x*x/8)
def _sinc(x):     return 1.0 if abs(x)<1e-9 else math.sin(math.pi*x)/(math.pi*x)
def _haar(x):
    if 0.0 <= x < 0.5: return 1.0
    if 0.5 <= x < 1.0: return -1.0
    return 0.0

CUSTOM_ACTS = [('gauss',_gauss),('sin',_sin),('cos',_cos),('tanh',_tanh),
               ('sigmoid',_sigmoid),('relu',_relu),('sech',_sech),
               ('mex_hat',_mex_hat),('morlet_re',_morlet_re),('morlet_im',_morlet_im),
               ('gabor',_gabor),('dog',_dog),('sinc',_sinc),('haar',_haar)]

def register_acts(config):
    for name, fn in CUSTOM_ACTS:
        try: config.genome_config.add_activation(name, fn)
        except RuntimeError: pass

# ─── NEAT Config ──────────────────────────────────────────────────────────────

_CFG_CACHE = {}

def load_config(n_inputs, n_outputs=3):
    key = (n_inputs, n_outputs)
    if key in _CFG_CACHE:
        return _CFG_CACHE[key]
    if n_inputs == 4:
        cfg_file = MODELS / 'neat_config_4in.ini'
    else:
        # Write a minimal inline config for 3-in 3-out
        import textwrap, tempfile
        body = textwrap.dedent(f"""
        [NEAT]
        fitness_criterion     = max
        fitness_threshold     = 10000
        pop_size              = 10
        reset_on_extinction   = False
        no_fitness_termination = True
        [DefaultGenome]
        num_inputs              = {n_inputs}
        num_outputs             = {n_outputs}
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
        weight_init_stdev       = 1.0
        weight_max_value        = 5.0
        weight_min_value        = -5.0
        weight_mutate_rate      = 0.8
        weight_mutate_power     = 0.5
        weight_replace_rate     = 0.1
        bias_init_mean          = 0.0
        bias_init_stdev         = 1.0
        bias_max_value          = 5.0
        bias_min_value          = -5.0
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
        max_stagnation       = 20
        species_elitism      = 2
        [DefaultReproduction]
        elitism            = 2
        survival_threshold = 0.2
        """)
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
        tf.write(body); tf.flush()
        cfg_file = tf.name
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation, str(cfg_file))
    register_acts(cfg)
    _CFG_CACHE[key] = cfg
    return cfg

def load_genome(pkl_path):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    return d['genome'] if isinstance(d, dict) and 'genome' in d else d

# ─── Indicators ───────────────────────────────────────────────────────────────

def compute_er(close, period=14):
    """Kaufman ER, arctan-normalised."""
    s = pd.Series(close)
    direction  = s.diff(period).abs()
    volatility = s.diff().abs().rolling(period).sum()
    er = direction / (volatility + 1e-10)
    return np.arctan(er.values * 5.0) / (math.pi / 2.0)

def compute_bb_squeeze(close, period=20, threshold=0.015):
    """True when BB bandwidth < threshold (volatility compression)."""
    s   = pd.Series(close)
    sma = s.rolling(period).mean()
    std = s.rolling(period).std()
    bw  = (2 * std) / (sma + 1e-10)
    return (bw < threshold).values

def compute_range_position(close, period=30, extremes_pct=0.15):
    """True when price is in top/bottom extremes_pct of N-bar range."""
    s  = pd.Series(close)
    hi = s.rolling(period).max()
    lo = s.rolling(period).min()
    pos = (s - lo) / (hi - lo + 1e-10)
    return ((pos < extremes_pct) | (pos > 1.0 - extremes_pct)).values

# ─── Simulation ───────────────────────────────────────────────────────────────

MAX_HOLD = 150
SL_PIPS  = 20

def simulate(genome, config, close, mc_d, mc_dd, er_norm, pip,
             n_inputs, gate_mask=None):
    """
    gate_mask: boolean array, True = entry allowed at this bar.
    n_inputs: 3 (ASI-MC v2) or 4 (IronNet).
    Input order matches training:
      4-input: [-1=MC_D, -2=MC_dD, -3=ER_norm, -4=UPnL]
      3-input: [-1=MC_D, -2=MC_dD, -3=UPnL]
    """
    net   = neat.nn.FeedForwardNetwork.create(genome, config)
    n     = len(close)
    pos   = 0; entry = 0.0; bars_held = 0
    total = 0.0; n_tr = 0; trade_pips = []

    for i in range(n):
        upnl = float(np.clip((close[i]-entry)*pos/pip/50.0,-1.0,1.0)) if pos else 0.0
        if pos: bars_held += 1

        if n_inputs == 4:
            inp = [float(mc_d[i]), float(mc_dd[i]), float(er_norm[i]), upnl]
        else:
            inp = [float(mc_d[i]), float(mc_dd[i]), upnl]

        out    = net.activate(inp)
        action = int(np.argmax(out))   # 0=BUY 1=SELL 2=EXIT/FLAT

        # Forced exit (SL or max hold)
        if pos and (bars_held >= MAX_HOLD or
                    abs((close[i]-entry)/pip) >= SL_PIPS*2.5):
            p = (close[i]-entry)*pos/pip
            total += p; trade_pips.append(p); n_tr += 1; pos = 0; bars_held = 0

        entry_allowed = (gate_mask is None) or bool(gate_mask[i])

        if pos == 0:
            if action == 0 and entry_allowed:
                pos = 1; entry = close[i]; bars_held = 0
            elif action == 1 and entry_allowed:
                pos = -1; entry = close[i]; bars_held = 0
        elif pos == 1:
            if action in (1, 2):
                p = (close[i]-entry)/pip
                total += p; trade_pips.append(p); n_tr += 1; pos = 0; bars_held = 0
                if action == 1 and entry_allowed:
                    pos = -1; entry = close[i]; bars_held = 0
        elif pos == -1:
            if action in (0, 2):
                p = (entry-close[i])/pip
                total += p; trade_pips.append(p); n_tr += 1; pos = 0; bars_held = 0
                if action == 0 and entry_allowed:
                    pos = 1; entry = close[i]; bars_held = 0

    if pos:
        p = (close[-1]-entry)*pos/pip; total += p; trade_pips.append(p); n_tr += 1

    wr     = sum(1 for p in trade_pips if p > 0) / max(1, n_tr)
    ppt    = total / max(1, n_tr)
    std_tp = float(np.std(trade_pips)) if len(trade_pips) > 1 else 1.0
    sharpe = ppt / (std_tp + 1e-10) * math.sqrt(max(1, n_tr))
    peak = 0.0; maxdd = 0.0; equity = 0.0
    for p in trade_pips:
        equity += p
        if equity > peak: peak = equity
        dd = peak - equity
        if dd > maxdd: maxdd = dd

    return {'pips': total, 'trades': n_tr, 'wr': wr,
            'ppt': ppt, 'sharpe': sharpe, 'maxdd': maxdd}

# ─── Run Experiment ───────────────────────────────────────────────────────────

FILTER_NAMES = ['BB_squeeze', 'ER_ranging', 'RangePos']
N_FILTERS    = len(FILTER_NAMES)

GENOME_CONFIGS = [
    # (pkl stem, [pairs], n_inputs, label)
    ('iron_s5ft_EUR_GBP', ['EUR_GBP'], 4, 'IronNet S5-FT EUR_GBP'),
    ('iron_s5ft_CAD_JPY', ['CAD_JPY'], 4, 'IronNet S5-FT CAD_JPY'),
    ('iron_v3_EUR_GBP',   ['EUR_GBP'], 4, 'IronNet V3 EUR_GBP'),
    ('iron_v3_CAD_JPY',   ['CAD_JPY'], 4, 'IronNet V3 CAD_JPY'),
    ('asi_mc_v2_best',    ['EUR_GBP','CAD_JPY','EUR_USD','GBP_USD',
                           'USD_JPY','EUR_JPY','AUD_USD','NZD_USD'], 3, 'ASI-MC v2 (8 pairs)'),
]

def run_genome_study(pkl_stem, pairs, n_inputs, label):
    print(f'\n  ── {label} ──')
    genome = load_genome(MODELS / f'{pkl_stem}.pkl')
    config = load_config(n_inputs)

    # Aggregate across pairs
    combo_results = {}   # combo_idx -> list of per-pair metric dicts

    for pair in pairs:
        pq_path = DATA / f'{pair}_curator.parquet'
        if not pq_path.exists():
            print(f'    [skip] {pair}: parquet not found')
            continue

        df      = pd.read_parquet(pq_path)
        close   = df['mid_close'].values.astype(np.float64)
        mc_d    = df['mc_d_curator'].values.astype(np.float64)
        mc_dd   = df['mc_dd_curator'].values.astype(np.float64)
        pip     = PIP.get(pair, 0.0001)

        # OOS: last 30% of data
        n_oos   = int(len(close) * 0.30)
        close   = close[-n_oos:]
        mc_d    = mc_d[-n_oos:]
        mc_dd   = mc_dd[-n_oos:]

        # Normalise indicators
        def norm(x): return np.clip(x / (np.std(x) + 1e-10), -3, 3)
        mc_d_n  = norm(mc_d)
        mc_dd_n = norm(mc_dd)
        er_norm = compute_er(close)[-n_oos:] if n_inputs == 4 else np.zeros(n_oos)

        # Compute filter arrays
        bb_sq  = compute_bb_squeeze(close)
        er_rng = er_norm < 0.35
        rng_ex = compute_range_position(close)

        filters = [bb_sq, er_rng, rng_ex]

        # Handle NaN at start (rolling windows)
        warmup = 30
        start  = warmup

        for combo_idx in range(2 ** N_FILTERS):
            bits    = [(combo_idx >> b) & 1 for b in range(N_FILTERS)]
            active  = [f for f, b in zip(filters, bits) if b]

            if active:
                gate = np.ones(len(close), dtype=bool)
                for f in active: gate &= f
                gate[:start] = False
            else:
                gate = None   # no filter = baseline

            res = simulate(genome, config, close, mc_d_n, mc_dd_n, er_norm,
                           pip, n_inputs, gate_mask=gate)

            if combo_idx not in combo_results:
                combo_results[combo_idx] = []
            combo_results[combo_idx].append(res)

        print(f'    {pair}: {n_oos} bars processed')

    # Aggregate across pairs
    rows = []
    for combo_idx in range(2 ** N_FILTERS):
        bits    = [(combo_idx >> b) & 1 for b in range(N_FILTERS)]
        combo_label = '+'.join(FILTER_NAMES[b] for b, bit in enumerate(bits) if bit) or 'BASELINE'
        results = combo_results.get(combo_idx, [])
        if not results: continue

        agg = {
            'combo': combo_label,
            'bits':  ''.join(str(b) for b in bits),
            'pips':  sum(r['pips']   for r in results),
            'trades':sum(r['trades'] for r in results),
            'wr':    np.mean([r['wr']     for r in results]),
            'ppt':   np.mean([r['ppt']    for r in results]),
            'sharpe':np.mean([r['sharpe'] for r in results]),
            'maxdd': np.mean([r['maxdd']  for r in results]),
        }
        rows.append(agg)

    # Print table
    baseline = next((r for r in rows if r['combo'] == 'BASELINE'), None)
    print(f'\n    {"Combo":<30} {"Pips":>8} {"Trades":>7} {"WR%":>6} {"PPT":>7} {"Sharpe":>8} {"MaxDD":>8}')
    print(f'    {"-"*78}')
    for r in sorted(rows, key=lambda x: x['pips'], reverse=True):
        flag = ' ◄ BEST' if r == sorted(rows, key=lambda x: x['pips'], reverse=True)[0] else ''
        delta_pips = f' ({r["pips"]-baseline["pips"]:+.0f})' if baseline and r['combo'] != 'BASELINE' else ''
        print(f'    {r["combo"]:<30} {r["pips"]:>8.1f}{delta_pips:<8} '
              f'{r["trades"]:>7} {r["wr"]*100:>5.1f}% {r["ppt"]:>7.1f} '
              f'{r["sharpe"]:>8.2f} {r["maxdd"]:>8.1f}{flag}')

    return rows

def plot_heatmap(all_results, title, out_path):
    """Heatmap: filter combos (rows) vs metrics (cols), colour = delta vs baseline."""
    metrics = ['pips', 'wr', 'ppt', 'sharpe', 'maxdd']
    m_labels = ['Total\nPips', 'Win\nRate%', 'Pips/\nTrade', 'Sharpe', 'Max\nDD']

    baseline = next((r for r in all_results if r['combo'] == 'BASELINE'), None)
    if not baseline: return

    others = [r for r in all_results if r['combo'] != 'BASELINE']
    others.sort(key=lambda x: x['pips'], reverse=True)

    rows    = [baseline] + others
    row_lbl = [r['combo'] for r in rows]

    data = np.array([[r[m] for m in metrics] for r in rows], dtype=float)
    # Normalise each col to z-score for colour (except maxdd which is bad when high)
    data_norm = np.zeros_like(data)
    for j, m in enumerate(metrics):
        col  = data[:, j]
        std_ = col.std() + 1e-10
        data_norm[:, j] = (col - col.mean()) / std_
        if m == 'maxdd': data_norm[:, j] *= -1  # lower DD = better

    fig, ax = plt.subplots(figsize=(10, max(6, len(rows)*0.6+2)))
    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)

    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(m_labels, fontsize=9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(row_lbl, fontsize=8)

    for i in range(len(rows)):
        for j, m in enumerate(metrics):
            val = data[i, j]
            fmt = f'{val:.1f}' if m != 'wr' else f'{val*100:.1f}%'
            ax.text(j, i, fmt, ha='center', va='center', fontsize=7.5,
                    color='black' if abs(data_norm[i,j]) < 1.2 else 'white')

    ax.set_title(title, fontweight='bold', fontsize=11, pad=10)
    plt.colorbar(im, ax=ax, label='Normalised score (green=better)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'    Heatmap saved: {out_path}')

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('Filter Gate Study')
    print(f'Filters: {FILTER_NAMES}')
    print(f'Combos:  2^{N_FILTERS} = {2**N_FILTERS} (including baseline)')
    print(f'Data:    M5 OOS parquets (last 30% = ~240 days)')

    all_genome_results = {}

    for pkl_stem, pairs, n_inputs, label in GENOME_CONFIGS:
        if not (MODELS / f'{pkl_stem}.pkl').exists():
            print(f'\n  [SKIP] {pkl_stem}.pkl not found')
            continue

        rows = run_genome_study(pkl_stem, pairs, n_inputs, label)
        all_genome_results[pkl_stem] = rows

        # Save CSV
        import csv
        csv_path = OUT / f'{pkl_stem}_filters.csv'
        with open(csv_path, 'w', newline='') as fp:
            w = csv.DictWriter(fp, fieldnames=['combo','bits','pips','trades','wr','ppt','sharpe','maxdd'])
            w.writeheader(); w.writerows(rows)

        # Save heatmap
        plot_heatmap(rows, f'{label}\nFilter Gate Study', OUT / f'{pkl_stem}_heatmap.png')

    # Cross-genome summary
    print('\n\n  ══ Cross-Genome Summary: Best Filter Per Genome ══')
    print(f'  {"Genome":<30} {"Best Filter":<30} {"Pips":>8} {"vs Baseline":>12}')
    print(f'  {"-"*84}')
    for pkl_stem, rows in all_genome_results.items():
        baseline  = next((r for r in rows if r['combo'] == 'BASELINE'), None)
        best      = max(rows, key=lambda r: r['pips'])
        delta     = best['pips'] - baseline['pips'] if baseline else 0
        sign      = '▲' if delta >= 0 else '▼'
        print(f'  {pkl_stem:<30} {best["combo"]:<30} {best["pips"]:>8.1f} {sign}{abs(delta):>10.1f}')

if __name__ == '__main__':
    main()
