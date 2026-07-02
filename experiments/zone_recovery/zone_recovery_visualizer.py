"""
Visual simulation of zone recovery + trailing hybrid.

Generates synthetic GBP_JPY-like price bouncing in a zone, marks each leg
entry, and plots unrealized P&L (UPnL) in a subplot sharing the x-axis.

Usage:
  python3 zone_recovery_visualizer.py [--seed N] [--bars N] [--trail]
"""

import argparse
import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Params ────────────────────────────────────────────────────────────────
PIP      = 0.01
ZW       = 56          # zone width pips
TGT      = 28          # target beyond zone pips
MAX_LEGS = 10
PF       = 1.19
SPREAD   = 1.4
ACT_PIPS = 14          # trail activation (hybrid mode)
TRAIL_PIP= 7           # trail distance


def net_basket(legs, price):
    gross = sum(l['vol'] * l['dir'] * (price - l['price']) / PIP for l in legs)
    cost  = sum(l['vol'] for l in legs) * SPREAD
    return gross - cost


def breakeven_vol(legs, target):
    net = net_basket(legs, target)
    if net >= 0:
        return 0.0
    return max(1.0, math.ceil(-net / TGT * PF))


def make_synthetic_price(n_bars, seed=7, start=200.0, noise_pip=8.0):
    rng = np.random.RandomState(seed)
    price = [start]
    # Use mean-reverting random walk (Ornstein-Uhlenbeck-like)
    mu = start
    theta = 0.03
    sigma = noise_pip * PIP
    for _ in range(n_bars - 1):
        dp = theta * (mu - price[-1]) + sigma * rng.randn()
        price.append(max(price[-1] + dp, start - 200 * PIP))
    return np.array(price)


def run_simulation(close, trail_mode_on=False):
    """Run one zone recovery cycle starting at close[0]."""
    n = len(close)
    entry     = close[0]
    direction = 1   # always LONG for clarity

    upper_zone   = entry
    lower_zone   = entry - ZW * PIP
    upper_target = entry + TGT * PIP
    lower_target = lower_zone - TGT * PIP

    legs = [{'dir': direction, 'price': entry, 'vol': 1.0, 'bar': 0}]
    events = [{'bar': 0, 'type': 'entry', 'price': entry, 'vol': 1.0, 'dir': direction}]

    last_crossed = last_crossed_bar = None
    closed = False
    exit_bar = 0
    exit_price = entry
    exit_reason = 'eod'

    mfe_pips    = 0.0
    trail_stop  = None
    trail_active = False
    zone_crossed = False

    upnl_series = [0.0]
    bar_prices  = [entry]

    for i in range(1, n):
        hi = close[max(0, i-1):i+1].max()  # approximate hi/lo from close
        lo = close[max(0, i-1):i+1].min()
        cl = close[i]
        # Simulate bar extremes with ±30% of move
        move = abs(cl - close[i-1])
        hi = max(cl, close[i-1]) + move * 0.3
        lo = min(cl, close[i-1]) - move * 0.3

        bar_prices.append(cl)

        if trail_mode_on and not zone_crossed:
            bar_mfe = (hi - entry) / PIP
            if bar_mfe > mfe_pips:
                mfe_pips = bar_mfe
            if not trail_active and mfe_pips >= ACT_PIPS:
                trail_active = True
            if trail_active:
                new_stop = entry + (mfe_pips - TRAIL_PIP) * PIP
                if trail_stop is None or new_stop > trail_stop:
                    trail_stop = new_stop
                if lo <= trail_stop:
                    exit_price = trail_stop
                    exit_reason = 'trail_stop'
                    exit_bar = i
                    closed = True
                    events.append({'bar': i, 'type': 'trail_exit',
                                   'price': trail_stop, 'vol': 0, 'dir': 0})
                    break

        # Target check
        if not (trail_active and not zone_crossed):
            if hi >= upper_target:
                exit_price = upper_target
                exit_reason = 'target_up'
                exit_bar = i
                closed = True
                events.append({'bar': i, 'type': 'target', 'price': upper_target, 'vol': 0, 'dir': 0})
                break
            if lo <= lower_target:
                exit_price = lower_target
                exit_reason = 'target_dn'
                exit_bar = i
                closed = True
                events.append({'bar': i, 'type': 'target', 'price': lower_target, 'vol': 0, 'dir': 0})
                break

        # Zone crossings
        if hi >= upper_zone:
            if not (last_crossed == 'upper' and last_crossed_bar == i):
                last_crossed, last_crossed_bar = 'upper', i
                vol = breakeven_vol(legs, upper_target)
                if vol > 0 and len(legs) < MAX_LEGS:
                    legs.append({'dir': 1, 'price': upper_zone, 'vol': vol, 'bar': i})
                    events.append({'bar': i, 'type': 'leg_long', 'price': upper_zone,
                                   'vol': vol, 'dir': 1})
                    zone_crossed = True

        if lo <= lower_zone:
            if not (last_crossed == 'lower' and last_crossed_bar == i):
                last_crossed, last_crossed_bar = 'lower', i
                vol = breakeven_vol(legs, lower_target)
                if vol > 0 and len(legs) < MAX_LEGS:
                    legs.append({'dir': -1, 'price': lower_zone, 'vol': vol, 'bar': i})
                    events.append({'bar': i, 'type': 'leg_short', 'price': lower_zone,
                                   'vol': vol, 'dir': -1})
                    zone_crossed = True

        upnl_series.append(net_basket(legs, cl))

    if not closed:
        # Mark EOD
        exit_price = bar_prices[-1]
        exit_bar = len(bar_prices) - 1

    # Extend upnl to exit bar
    while len(upnl_series) < len(bar_prices):
        upnl_series.append(net_basket(legs, bar_prices[len(upnl_series)]))

    return {
        'bars': bar_prices,
        'upnl': upnl_series,
        'legs': legs,
        'events': events,
        'exit_bar': exit_bar,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'upper_zone': upper_zone,
        'lower_zone': lower_zone,
        'upper_target': upper_target,
        'lower_target': lower_target,
        'entry': entry,
        'trail_stop_final': trail_stop,
    }


def plot_simulation(result, title, filename):
    bars  = np.array(result['bars'])
    upnl  = np.array(result['upnl'])
    n     = len(bars)
    x     = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#0d1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#cccccc')
        for spine in ax.spines.values():
            spine.set_color('#333333')

    # ── Price chart ───────────────────────────────────────────────────────
    ax1.plot(x, bars, color='#4fc3f7', linewidth=1.2, zorder=3, label='Price')

    uz = result['upper_zone']
    lz = result['lower_zone']
    ut = result['upper_target']
    lt = result['lower_target']

    ax1.axhline(uz, color='#66bb6a', linewidth=1.0, linestyle='--', alpha=0.7, label='Zone boundary (upper)')
    ax1.axhline(lz, color='#ef5350', linewidth=1.0, linestyle='--', alpha=0.7, label='Zone boundary (lower)')
    ax1.axhline(ut, color='#66bb6a', linewidth=0.7, linestyle=':', alpha=0.5, label='Target (upper)')
    ax1.axhline(lt, color='#ef5350', linewidth=0.7, linestyle=':', alpha=0.5, label='Target (lower)')

    # Zone shading
    ax1.axhspan(lz, uz, alpha=0.06, color='#ffffff', zorder=0)

    # Trail stop line (if any)
    if result.get('trail_stop_final') is not None:
        ax1.axhline(result['trail_stop_final'], color='#ffa726', linewidth=0.8,
                    linestyle='-.', alpha=0.7, label='Trail stop')

    # Event markers
    for ev in result['events']:
        b = ev['bar']
        p = ev['price']
        etype = ev['type']
        if etype == 'entry':
            ax1.scatter(b, p, marker='^', s=120, color='#66bb6a', zorder=5)
            ax1.annotate(f"  LONG\n  vol={ev['vol']:.0f}", (b, p),
                         color='#66bb6a', fontsize=7, va='center')
        elif etype == 'leg_long':
            ax1.scatter(b, p, marker='^', s=90, color='#29b6f6', zorder=5)
            ax1.annotate(f"  +LONG\n  vol={ev['vol']:.0f}", (b, p),
                         color='#29b6f6', fontsize=7, va='center')
        elif etype == 'leg_short':
            ax1.scatter(b, p, marker='v', s=90, color='#ff7043', zorder=5)
            ax1.annotate(f"  +SHORT\n  vol={ev['vol']:.0f}", (b, p),
                         color='#ff7043', fontsize=7, va='center')
        elif etype == 'target':
            ax1.scatter(b, p, marker='*', s=180, color='#ffd54f', zorder=6)
            ax1.annotate(f"  EXIT\n  target", (b, p),
                         color='#ffd54f', fontsize=7, va='center')
        elif etype == 'trail_exit':
            ax1.scatter(b, p, marker='D', s=120, color='#ffa726', zorder=6)
            ax1.annotate(f"  EXIT\n  trail", (b, p),
                         color='#ffa726', fontsize=7, va='center')

    # Exit bar vertical line
    ax1.axvline(result['exit_bar'], color='#ffffff', linewidth=0.5, alpha=0.3)

    ax1.set_ylabel('Price', color='#cccccc', fontsize=10)
    ax1.set_title(title, color='#ffffff', fontsize=11, pad=10)
    ax1.legend(loc='upper left', fontsize=7, facecolor='#1a1a2e',
               edgecolor='#333333', labelcolor='#cccccc', ncol=2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))

    # ── UPnL subplot ──────────────────────────────────────────────────────
    colors = np.where(upnl >= 0, '#66bb6a', '#ef5350')
    ax2.fill_between(x, upnl, 0, where=upnl >= 0, color='#66bb6a', alpha=0.4)
    ax2.fill_between(x, upnl, 0, where=upnl <  0, color='#ef5350', alpha=0.4)
    ax2.plot(x, upnl, color='#b0bec5', linewidth=0.8)
    ax2.axhline(0, color='#555555', linewidth=0.6)
    ax2.axvline(result['exit_bar'], color='#ffffff', linewidth=0.5, alpha=0.3)

    final_upnl = upnl[result['exit_bar']] if result['exit_bar'] < len(upnl) else upnl[-1]
    ax2.set_ylabel('UPnL (pips)', color='#cccccc', fontsize=9)
    ax2.set_xlabel('Bar', color='#cccccc', fontsize=9)

    n_legs = len(result['legs'])
    reason = result['exit_reason']
    ax2.set_title(f"Unrealized P&L  |  n_legs={n_legs}  exit={reason}  final={final_upnl:+.0f}p",
                  color='#aaaaaa', fontsize=8)

    plt.tight_layout(h_pad=0.5)
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',  type=int, default=7,   help='Price seed')
    parser.add_argument('--bars',  type=int, default=300, help='Bars to simulate')
    parser.add_argument('--trail', action='store_true',   help='Enable trail hybrid')
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    price = make_synthetic_price(args.bars, seed=args.seed)

    print("="*60)
    print("  ZONE RECOVERY VISUAL SIMULATION")
    print("="*60)

    # Baseline (fixed target)
    r_base = run_simulation(price, trail_mode_on=False)
    plot_simulation(r_base,
        f"Zone Recovery — Baseline (fixed {TGT}p target)  seed={args.seed}",
        f"{out_dir}/zr_visual_baseline_s{args.seed}.png")
    print(f"  Baseline: exit={r_base['exit_reason']}  legs={len(r_base['legs'])}")

    # Trailing hybrid
    r_trail = run_simulation(price, trail_mode_on=True)
    plot_simulation(r_trail,
        f"Zone Recovery — Trailing Hybrid (act={ACT_PIPS}p trail={TRAIL_PIP}p)  seed={args.seed}",
        f"{out_dir}/zr_visual_trail_s{args.seed}.png")
    print(f"  Hybrid  : exit={r_trail['exit_reason']}  legs={len(r_trail['legs'])}")
