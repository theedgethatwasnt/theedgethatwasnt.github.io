#!/usr/bin/env python3
"""
Generate genome architecture PNGs for all winner genomes.

Handles three topology classes automatically:
  - Fixed 2-layer  (activation study v1-v4): input → L1(100s) → L2(200s) → output
  - Fixed 1-layer  (IronNet):                input → L1(100s) → output
  - Free topology  (ASI-MC v2):              auto-layout via BFS depth
"""

import math
import pickle
import sys
from pathlib import Path
from collections import defaultdict, deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

ROOT = Path(__file__).parent.parent.parent.parent   # fx-core root
ACT_DIR = Path(__file__).parent
MODELS   = ROOT / 'models'
OUT_DIR  = ACT_DIR / 'results'
OUT_DIR.mkdir(exist_ok=True)

# ─── Activation → color ───────────────────────────────────────────────────────
ACT_COLOR = {
    'tanh':      '#e74c3c',
    'sin':       '#f39c12',
    'cos':       '#e67e22',
    'gauss':     '#3498db',
    'mex_hat':   '#9b59b6',
    'morlet_re': '#1abc9c',
    'morlet_im': '#16a085',
    'gabor':     '#2ecc71',
    'dog':       '#27ae60',
    'sinc':      '#8e44ad',
    'chirp':     '#d35400',
    'sech':      '#c0392b',
    'haar':      '#7f8c8d',
    'sigmoid':   '#ff7675',
    'relu':      '#fd79a8',
    'elu':       '#e91e63',
    'swish':     '#ad1457',
}
DEFAULT_COLOR = '#546e7a'

ACT_GROUP = {
    'tanh': 'Baseline', 'sin': 'Baseline', 'cos': 'Baseline', 'gauss': 'Baseline',
    'mex_hat': 'Ricker', 'morlet_re': 'Morlet', 'morlet_im': 'Morlet',
    'gabor': 'Gabor', 'dog': 'DoG', 'sinc': 'Sinc', 'chirp': 'Chirp',
    'sech': 'Sech', 'haar': 'Haar',
    'sigmoid': 'Classic', 'relu': 'Classic', 'elu': 'Classic', 'swish': 'Classic',
}

# ─── Genome catalogue ─────────────────────────────────────────────────────────
GENOMES = [
    dict(
        path=ACT_DIR / 'best_genome.pkl',
        label='Activation Study v1',
        subtitle='5 inputs · 40 gens · 5 activations · 96 connections',
        input_labels=['MC(D)', 'MC(dD)', 'ER_norm', 'UPnL', 'mex_hat\n(MC_D)'],
        output_labels=['BUY', 'SELL', 'EXIT'],
        out_name='genome_architecture_v1.png',
    ),
    dict(
        path=ACT_DIR / 'best_genome_v2.pkl',
        label='Activation Study v2',
        subtitle='5 inputs · 40 gens · 17 activations (incl. wavelets) · 96 connections',
        input_labels=['MC(D)', 'MC(dD)', 'ER_norm', 'UPnL', 'mex_hat\n(MC_D)'],
        output_labels=['BUY', 'SELL', 'EXIT'],
        out_name='genome_architecture_v2.png',
    ),
    dict(
        path=ACT_DIR / 'best_genome_v3.pkl',
        label='Activation Study v3',
        subtitle='5 inputs · 200 gens · seeded pop · 17 activations · 96 connections',
        input_labels=['MC(D)', 'MC(dD)', 'ER_norm', 'UPnL', 'mex_hat\n(MC_D)'],
        output_labels=['BUY', 'SELL', 'EXIT'],
        out_name='genome_architecture_v3.png',
    ),
    dict(
        path=ACT_DIR / 'best_genome_v4.pkl',
        label='Activation Study v4',
        subtitle='4 clean inputs · 200 gens · seeded pop · 17 activations · 88 connections',
        input_labels=['MC(D)', 'MC(dD)', 'ER_norm', 'UPnL'],
        output_labels=['BUY', 'SELL', 'EXIT'],
        out_name='genome_architecture_v4.png',
    ),
    dict(
        path=MODELS / 'iron_s5ft_EUR_GBP.pkl',
        label='IronNet S5-FT  EUR/GBP',
        subtitle='4 inputs · 1 hidden layer · 40 connections · deployed acct 001',
        input_labels=['MC(D)', 'MC(dD)', 'ER_norm', 'UPnL'],
        output_labels=['BUY', 'SELL', 'EXIT'],
        out_name='genome_architecture_iron_s5ft_EUR_GBP.png',
    ),
    dict(
        path=MODELS / 'iron_s5ft_CAD_JPY.pkl',
        label='IronNet S5-FT  CAD/JPY',
        subtitle='4 inputs · 1 hidden layer · 40 connections · deployed acct 001',
        input_labels=['MC(D)', 'MC(dD)', 'ER_norm', 'UPnL'],
        output_labels=['BUY', 'SELL', 'EXIT'],
        out_name='genome_architecture_iron_s5ft_CAD_JPY.png',
    ),
    dict(
        path=MODELS / 'iron_v3_EUR_GBP.pkl',
        label='IronNet V3  EUR/GBP',
        subtitle='4 inputs · 1 hidden layer · 40 connections · deployed acct 009',
        input_labels=['MC(D)', 'MC(dD)', 'ER_norm', 'UPnL'],
        output_labels=['BUY', 'SELL', 'EXIT'],
        out_name='genome_architecture_iron_v3_EUR_GBP.png',
    ),
    dict(
        path=MODELS / 'iron_v3_CAD_JPY.pkl',
        label='IronNet V3  CAD/JPY',
        subtitle='4 inputs · 1 hidden layer · 40 connections · deployed acct 009',
        input_labels=['MC(D)', 'MC(dD)', 'ER_norm', 'UPnL'],
        output_labels=['BUY', 'SELL', 'EXIT'],
        out_name='genome_architecture_iron_v3_CAD_JPY.png',
    ),
    dict(
        path=MODELS / 'asi_mc_v2_best.pkl',
        label='ASI-MC v2  (general, 12 pairs)',
        subtitle='3 inputs · free topology · 9 hidden nodes · 27 connections · deployed acct 008',
        input_labels=['MC(D)', 'MC(dD)', 'UPnL'],
        output_labels=['BUY', 'SELL', 'EXIT'],
        out_name='genome_architecture_asi_mc_v2.png',
    ),
]

# ─── Topology detection ───────────────────────────────────────────────────────

def load_genome(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['genome'] if isinstance(d, dict) and 'genome' in d else d


def detect_topology(g):
    """
    Returns layers: list of sorted node-ID lists, left to right.
    [input_ids, ...hidden_layers..., output_ids]
    """
    output_ids = sorted([k for k in g.nodes if k in (0, 1, 2)])
    input_ids  = sorted([k for k in g.connections if k[0] < 0], key=lambda x: x[0])
    # collect unique negative source keys
    neg_ids = sorted({k for c in g.connections for k in [c[0]] if k < 0})
    input_ids = neg_ids

    hidden_ids = [k for k in g.nodes if k not in output_ids and k >= 0]

    # check if all hidden in 100-199 range (IronNet style: 1 hidden layer)
    in_l1 = [k for k in hidden_ids if 100 <= k <= 199]
    in_l2 = [k for k in hidden_ids if 200 <= k <= 299]
    in_other = [k for k in hidden_ids if k not in in_l1 + in_l2]

    if in_other:
        # Free topology — BFS-assign depths
        depths = _bfs_depths(g, neg_ids, output_ids)
        max_d = max(depths.values()) if depths else 1
        # group by depth
        by_depth = defaultdict(list)
        for nid, d in depths.items():
            if nid in hidden_ids:
                by_depth[d].append(nid)
        layers = [sorted(neg_ids)]
        for d in sorted(by_depth.keys()):
            layers.append(sorted(by_depth[d]))
        layers.append(output_ids)
    elif in_l2:
        layers = [sorted(neg_ids), sorted(in_l1), sorted(in_l2), output_ids]
    else:
        layers = [sorted(neg_ids), sorted(in_l1), output_ids]

    return layers


def _bfs_depths(g, input_ids, output_ids):
    """BFS from inputs to assign depth to each node."""
    # build adjacency: src → [dst]
    adj = defaultdict(list)
    for (src, dst) in g.connections:
        adj[src].append(dst)

    depth = {}
    queue = deque()
    for nid in input_ids:
        depth[nid] = 0
        queue.append(nid)

    while queue:
        curr = queue.popleft()
        for nxt in adj[curr]:
            if nxt not in depth and nxt not in output_ids:
                depth[nxt] = depth[curr] + 1
                queue.append(nxt)

    # assign outputs the max depth + 1
    max_d = max(depth.values(), default=0)
    for nid in output_ids:
        depth[nid] = max_d + 1

    return depth


# ─── Drawing ──────────────────────────────────────────────────────────────────

def y_positions(n, lo=0.05, hi=0.95):
    if n == 1:
        return [0.5]
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


def weight_style(w, w_max):
    norm  = abs(w) / (w_max + 1e-9)
    alpha = 0.08 + 0.72 * norm
    lw    = 0.3  + 2.5  * norm
    color = '#e74c3c' if w < 0 else '#3498db'
    return color, alpha, lw


def draw_node(ax, x, y, label, sublabel, color, radius=0.045, fontsize=9):
    circle = plt.Circle((x, y), radius, color=color, zorder=3,
                         linewidth=1.5, edgecolor='white', alpha=0.92)
    ax.add_patch(circle)
    ax.text(x, y + 0.012, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white', zorder=4,
            path_effects=[pe.withStroke(linewidth=1, foreground='black')])
    if sublabel:
        ax.text(x, y - 0.02, sublabel, ha='center', va='center',
                fontsize=6.0, color='white', alpha=0.85, zorder=4)


def draw_genome(cfg):
    path = cfg['path']
    if not Path(path).exists():
        print(f'  SKIP (not found): {path}')
        return

    g = load_genome(path)
    layers = detect_topology(g)
    n_layers = len(layers)

    input_labels  = cfg['input_labels']
    output_labels = cfg['output_labels']
    layer_names   = (
        ['INPUTS'] +
        [f'LAYER {i}' for i in range(1, n_layers - 1)] +
        ['OUTPUTS']
    )

    # X positions spread evenly
    x_vals = [i / (n_layers - 1) * 3.2 for i in range(n_layers)]

    # Build position map
    pos = {}
    for li, layer in enumerate(layers):
        ys = y_positions(len(layer))
        for ni, nid in enumerate(layer):
            pos[nid] = (x_vals[li], ys[ni])

    all_weights = [c.weight for c in g.connections.values()]
    w_max = max(abs(w) for w in all_weights) if all_weights else 1.0

    fig, ax = plt.subplots(figsize=(4 + n_layers * 3.5, 10))
    x_range = x_vals[-1]
    ax.set_xlim(-0.35, x_range + 0.45)
    ax.set_ylim(-0.10, 1.12)
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Draw connections
    for (src, dst), conn in g.connections.items():
        if src not in pos or dst not in pos:
            continue
        color, alpha, lw = weight_style(conn.weight, w_max)
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, lw=lw, zorder=1)

    # Draw input nodes
    for i, nid in enumerate(layers[0]):
        x, y = pos[nid]
        lbl = input_labels[i] if i < len(input_labels) else f'IN{i}'
        draw_node(ax, x, y, lbl, '', DEFAULT_COLOR, fontsize=7.5)

    # Draw output nodes
    out_colors = ['#27ae60', '#e74c3c', '#f39c12']
    for i, nid in enumerate(layers[-1]):
        x, y = pos[nid]
        act  = g.nodes[nid].activation if nid in g.nodes else 'sigmoid'
        bias = g.nodes[nid].bias       if nid in g.nodes else 0.0
        lbl  = output_labels[i] if i < len(output_labels) else f'OUT{i}'
        draw_node(ax, x, y, lbl, f'{act}\nb={bias:.2f}',
                  out_colors[i % len(out_colors)], radius=0.055, fontsize=10)

    # Draw hidden nodes
    for layer in layers[1:-1]:
        for nid in layer:
            if nid not in g.nodes:
                continue
            x, y = pos[nid]
            act  = g.nodes[nid].activation
            bias = g.nodes[nid].bias
            color = ACT_COLOR.get(act, '#888')
            draw_node(ax, x, y, act, f'b={bias:.2f}', color)

    # Layer labels
    for li, (xv, lname, layer) in enumerate(zip(x_vals, layer_names, layers)):
        ax.text(xv, 1.05, f'{lname}\n({len(layer)})',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='#ecf0f1')

    # Strong weight annotations
    strong = [(k, c) for k, c in g.connections.items() if abs(c.weight) > 3.5]
    strong.sort(key=lambda x: abs(x[1].weight), reverse=True)
    for (src, dst), conn in strong[:10]:
        if src not in pos or dst not in pos:
            continue
        x0, y0 = pos[src]; x1, y1 = pos[dst]
        xm, ym = (x0+x1)/2, (y0+y1)/2
        ax.text(xm, ym, f'{conn.weight:+.1f}', fontsize=5.5, color='white',
                ha='center', va='center', alpha=0.8, zorder=5,
                path_effects=[pe.withStroke(linewidth=1, foreground='black')])

    # Legend — weight colors
    legend_handles = [
        mpatches.Patch(color='#3498db', alpha=0.8, label='Positive weight'),
        mpatches.Patch(color='#e74c3c', alpha=0.8, label='Negative weight'),
        mpatches.Patch(color='white',   alpha=0.3, label='Thickness ∝ |weight|'),
    ]
    # Activation groups present
    seen_groups = {}
    for layer in layers[1:-1]:
        for nid in layer:
            if nid not in g.nodes:
                continue
            act = g.nodes[nid].activation
            grp = ACT_GROUP.get(act, act)
            if grp not in seen_groups:
                seen_groups[grp] = ACT_COLOR.get(act, '#888')
    for grp, col in seen_groups.items():
        legend_handles.append(mpatches.Patch(color=col, label=grp))

    ax.legend(handles=legend_handles, loc='lower center',
              ncol=min(len(legend_handles), 6), fontsize=8,
              facecolor='#16213e', edgecolor='#ecf0f1',
              labelcolor='white', framealpha=0.85,
              bbox_to_anchor=(0.5, -0.09))

    # Weight histogram inset
    ax_ins = fig.add_axes([0.82, 0.09, 0.12, 0.18])
    ax_ins.set_facecolor('#16213e')
    ax_ins.tick_params(colors='white', labelsize=6)
    for sp in ax_ins.spines.values(): sp.set_color('#546e7a')
    ws = np.array(all_weights)
    pos_w = ws[ws > 0]; neg_w = ws[ws < 0]
    if len(pos_w): ax_ins.hist(pos_w, bins=12, color='#3498db', alpha=0.75)
    if len(neg_w): ax_ins.hist(neg_w, bins=12, color='#e74c3c', alpha=0.75)
    ax_ins.axvline(0, color='white', lw=0.8)
    ax_ins.set_title('Weights', fontsize=7, color='white', pad=2)
    ax_ins.set_xlabel('w', fontsize=6, color='white')

    # Title
    topo_str = '→'.join(str(len(l)) for l in layers)
    ax.set_title(
        f"{cfg['label']}  —  Fixed-topology NEAT  ({topo_str})\n{cfg['subtitle']}",
        fontsize=12, fontweight='bold', color='#ecf0f1', pad=10,
    )

    plt.tight_layout()
    out = OUT_DIR / cfg['out_name']
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved: {out}')


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Drawing {len(GENOMES)} genome architectures → {OUT_DIR}')
    for cfg in GENOMES:
        print(f'\n[{cfg["label"]}]')
        draw_genome(cfg)
    print('\nAll done.')
