#!/usr/bin/env python3
"""Draw the best genome v3 as an annotated network diagram."""

import math
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

DIR = Path(__file__).parent

# ─── Load genome ──────────────────────────────────────────────────────────────

with open(DIR / 'best_genome_v3.pkl', 'rb') as f:
    g = pickle.load(f)

# ─── Layout constants ─────────────────────────────────────────────────────────

INPUT_IDS  = [-1, -2, -3, -4, -5]
L1_IDS     = [100, 101, 102, 103, 104]
L2_IDS     = [200, 201, 202, 203, 204, 205, 206]
OUTPUT_IDS = [0, 1, 2]

INPUT_LABELS  = ['MC(D)', 'MC(dD)', 'ER_norm', 'UPnL', 'mex_hat\n(MC_D)']
OUTPUT_LABELS = ['BUY', 'SELL', 'EXIT']

# Activation → color (matches run_v3 palette)
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
    'sigmoid':   '#e74c3c',
    'relu':      '#c0392b',
    'elu':       '#e91e63',
    'swish':     '#ad1457',
}

ACT_GROUP = {
    'tanh': 'Baseline', 'sin': 'Baseline', 'cos': 'Baseline', 'gauss': 'Baseline',
    'mex_hat': 'Ricker', 'morlet_re': 'Morlet', 'morlet_im': 'Morlet',
    'gabor': 'Gabor', 'dog': 'DoG', 'sinc': 'Sinc', 'chirp': 'Chirp',
    'sech': 'Sech', 'haar': 'Haar',
    'sigmoid': 'Classic', 'relu': 'Classic', 'elu': 'Classic', 'swish': 'Classic',
}

# X positions for each layer
X = {'input': 0.0, 'L1': 1.0, 'L2': 2.2, 'output': 3.2}

def y_positions(n, y_range=(0.05, 0.95)):
    """Evenly spaced Y positions for n nodes."""
    lo, hi = y_range
    if n == 1:
        return [0.5]
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]

# Build node position map
pos = {}
for i, nid in enumerate(INPUT_IDS):
    pos[nid] = (X['input'], y_positions(len(INPUT_IDS))[i])
for i, nid in enumerate(L1_IDS):
    pos[nid] = (X['L1'], y_positions(len(L1_IDS))[i])
for i, nid in enumerate(L2_IDS):
    pos[nid] = (X['L2'], y_positions(len(L2_IDS))[i])
for i, nid in enumerate(OUTPUT_IDS):
    pos[nid] = (X['output'], y_positions(len(OUTPUT_IDS))[i])

# ─── Draw ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(-0.25, 3.55)
ax.set_ylim(-0.08, 1.08)
ax.axis('off')
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Weight → visual encoding
all_weights = [c.weight for c in g.connections.values()]
w_max = max(abs(w) for w in all_weights)

def weight_style(w):
    norm = abs(w) / w_max          # 0→1
    alpha = 0.08 + 0.72 * norm     # 0.08 (faint) → 0.80 (strong)
    lw    = 0.3  + 2.5  * norm     # thin → thick
    color = '#e74c3c' if w < 0 else '#3498db'   # red=negative, blue=positive
    return color, alpha, lw

# ── Draw connections (back layer, behind nodes) ────────────────────────────────
for (src, dst), conn in g.connections.items():
    if src not in pos or dst not in pos:
        continue
    color, alpha, lw = weight_style(conn.weight)
    x0, y0 = pos[src]
    x1, y1 = pos[dst]
    ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, lw=lw, zorder=1)

# ── Draw nodes ─────────────────────────────────────────────────────────────────
NODE_R = 0.045   # radius in axes fraction

def draw_node(ax, x, y, label, sublabel, color, radius=NODE_R, fontsize=9):
    circle = plt.Circle((x, y), radius, color=color, zorder=3,
                         transform=ax.transData, linewidth=1.5,
                         edgecolor='white', alpha=0.92)
    ax.add_patch(circle)
    # Main label
    ax.text(x, y + 0.012, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white', zorder=4,
            path_effects=[pe.withStroke(linewidth=1, foreground='black')])
    # Sub-label (activation name)
    if sublabel:
        ax.text(x, y - 0.018, sublabel, ha='center', va='center',
                fontsize=6.5, color='white', alpha=0.85, zorder=4)

# Input nodes
for i, nid in enumerate(INPUT_IDS):
    x, y = pos[nid]
    draw_node(ax, x, y, INPUT_LABELS[i], '', '#546e7a', fontsize=7.5)

# L1 nodes
for nid in L1_IDS:
    x, y = pos[nid]
    act = g.nodes[nid].activation
    bias = g.nodes[nid].bias
    color = ACT_COLOR.get(act, '#888')
    draw_node(ax, x, y, act, f'b={bias:.2f}', color)

# L2 nodes
for nid in L2_IDS:
    x, y = pos[nid]
    act = g.nodes[nid].activation
    bias = g.nodes[nid].bias
    color = ACT_COLOR.get(act, '#888')
    draw_node(ax, x, y, act, f'b={bias:.2f}', color)

# Output nodes
out_colors = ['#27ae60', '#e74c3c', '#f39c12']
for i, nid in enumerate(OUTPUT_IDS):
    x, y = pos[nid]
    act = g.nodes[nid].activation
    bias = g.nodes[nid].bias
    draw_node(ax, x, y, OUTPUT_LABELS[i], f'{act}\nb={bias:.2f}',
              out_colors[i], radius=NODE_R * 1.25, fontsize=10)

# ── Layer labels ───────────────────────────────────────────────────────────────
label_kw = dict(ha='center', va='bottom', fontsize=11, fontweight='bold',
                color='#ecf0f1', transform=ax.transData)
ax.text(X['input'],  1.04, 'INPUTS\n(5)',  **label_kw)
ax.text(X['L1'],     1.04, 'LAYER 1\n(5 nodes)', **label_kw)
ax.text(X['L2'],     1.04, 'LAYER 2\n(7 nodes)', **label_kw)
ax.text(X['output'], 1.04, 'OUTPUTS\n(3)', **label_kw)

# ── Connection legend (weight color) ──────────────────────────────────────────
for x_pos, color, label in [
    (0.68, '#3498db', 'Positive weight'),
    (0.68, '#e74c3c', 'Negative weight'),
]:
    pass  # done via legend patches below

legend_handles = [
    mpatches.Patch(color='#3498db', alpha=0.8, label='Positive weight'),
    mpatches.Patch(color='#e74c3c', alpha=0.8, label='Negative weight'),
    mpatches.Patch(color='white',   alpha=0.3, label='Thickness ∝ |weight|'),
]

# Activation group legend
seen_groups = {}
for nid in L1_IDS + L2_IDS:
    act = g.nodes[nid].activation
    grp = ACT_GROUP.get(act, act)
    if grp not in seen_groups:
        seen_groups[grp] = ACT_COLOR.get(act, '#888')

for grp, col in seen_groups.items():
    legend_handles.append(mpatches.Patch(color=col, label=f'{grp}'))

leg = ax.legend(handles=legend_handles, loc='lower center',
                ncol=len(legend_handles), fontsize=8,
                facecolor='#16213e', edgecolor='#ecf0f1',
                labelcolor='white', framealpha=0.85,
                bbox_to_anchor=(0.5, -0.07))

# ── Strong weight annotations (|w| > 3.5) ─────────────────────────────────────
strong = [(k, c) for k, c in g.connections.items() if abs(c.weight) > 3.8]
strong.sort(key=lambda x: abs(x[1].weight), reverse=True)
for (src, dst), conn in strong[:12]:   # top 12 strongest
    if src not in pos or dst not in pos: continue
    x0, y0 = pos[src]; x1, y1 = pos[dst]
    xm, ym = (x0+x1)/2, (y0+y1)/2
    ax.text(xm, ym, f'{conn.weight:+.1f}', fontsize=5.5, color='white',
            ha='center', va='center', alpha=0.75, zorder=5,
            path_effects=[pe.withStroke(linewidth=1, foreground='black')])

# ── Skip connection bracket ────────────────────────────────────────────────────
ax.annotate('', xy=(X['output']-0.02, 0.5),
            xytext=(X['input']+0.02, 0.5),
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.2,
                            connectionstyle='arc3,rad=-0.35', alpha=0.4))
ax.text((X['input']+X['output'])/2, -0.06,
        'skip: inputs → outputs (15 connections)',
        ha='center', fontsize=8, color='#f39c12', alpha=0.6)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title('Best Genome v3  —  Fixed-topology NEAT  (5→5→7→3)\n'
             '200 gens, seeded population, 17 activation choices',
             fontsize=13, fontweight='bold', color='#ecf0f1', pad=12)

# ── Weight histogram inset ────────────────────────────────────────────────────
ax_inset = fig.add_axes([0.80, 0.10, 0.13, 0.20])
ax_inset.set_facecolor('#16213e')
ax_inset.tick_params(colors='white', labelsize=6)
for spine in ax_inset.spines.values(): spine.set_color('#546e7a')
weights = np.array(all_weights)
pos_w = weights[weights > 0]
neg_w = weights[weights < 0]
ax_inset.hist(pos_w, bins=15, color='#3498db', alpha=0.75, label='+')
ax_inset.hist(neg_w, bins=15, color='#e74c3c', alpha=0.75, label='−')
ax_inset.axvline(0, color='white', lw=0.8)
ax_inset.set_title('Weight dist.', fontsize=7, color='white', pad=3)
ax_inset.set_xlabel('w', fontsize=6, color='white')

plt.tight_layout()
out = DIR / 'genome_architecture_v3.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved: {out}')
