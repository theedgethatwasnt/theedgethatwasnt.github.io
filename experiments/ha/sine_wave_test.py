#!/usr/bin/env python3
"""
Sine Wave NEAT Training Sanity Check
=====================================
Validates the training pipeline on a PERFECTLY PREDICTABLE signal before
running on real market data. If NEAT can't learn to trade a sine wave,
the training setup has a fundamental issue.

Dataset: Synthetic EURUSD-like price oscillating between 1.1234 and 1.1254
         (20 pip amplitude, 0.0001 pip increment).

Inputs (3):
  - MC_HA(D):  HA color momentum consensus across simulated TFs (S5,S30,M1,M5,H1,H4)
  - MC_HA(dD): Acceleration of MC_HA(D) — second derivative
  - UPnL:     Unrealized P/L of current position, tanh-scaled

Outputs (3): BUY, SELL, FLATTEN
  - Highest output wins. One position at a time.
  - 2 pips spread cost on every entry.

Fitness: Sharpe + trade frequency bonus (same as Stage 2)

Expected result: NEAT should learn buy-at-bottom, sell-at-top within ~50 gens.
If it can't: training pipeline is broken.
"""

import sys
import os
import time
import json
import pickle
import math
import argparse
import numpy as np
from pathlib import Path
from numba import njit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import neat
from lib.fast_eval import extract_network, _activate


# ── Sine Wave Data Generation ──────────────────────────────────────────────

def generate_sine_wave(n_bars=50000, period_bars=500, amplitude_pips=10,
                       center=1.1244, pip=0.0001):
    """Generate synthetic EURUSD-like price as a sine wave.

    Args:
        n_bars: Total S5-equivalent bars
        period_bars: Bars per full cycle (500 S5 bars ≈ 42 min)
        amplitude_pips: Half-range in pips (10 = oscillates ±10 pips)
        center: Center price
        pip: Pip size (0.0001 for EURUSD)

    Returns:
        mid_close: float64 array of prices
    """
    t = np.arange(n_bars, dtype=np.float64)
    mid_close = center + amplitude_pips * pip * np.sin(2 * np.pi * t / period_bars)
    return mid_close


# ── HA Color Momentum Consensus (MC_HA) ───────────────────────────────────

@njit(cache=True)
def _compute_ha_color(o, h, l, c, n):
    """Compute HA direction: +1 bullish, -1 bearish."""
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty(n, dtype=np.float64)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    colors = np.empty(n, dtype=np.float64)
    for i in range(n):
        colors[i] = 1.0 if ha_c[i] >= ha_o[i] else -1.0
    return colors


@njit(cache=True)
def _resample_and_ha(prices, n, bars_per_tf):
    """Resample prices to lower TF, compute HA colors, return (colors, n_tf)."""
    n_tf = n // bars_per_tf
    if n_tf < 8:
        return np.zeros(1, dtype=np.float64), 0

    o = np.empty(n_tf, dtype=np.float64)
    h = np.empty(n_tf, dtype=np.float64)
    l = np.empty(n_tf, dtype=np.float64)
    c = np.empty(n_tf, dtype=np.float64)
    for j in range(n_tf):
        start = j * bars_per_tf
        end = start + bars_per_tf
        o[j] = prices[start]
        c[j] = prices[end - 1]
        hh = prices[start]
        ll = prices[start]
        for k in range(start + 1, end):
            if prices[k] > hh:
                hh = prices[k]
            if prices[k] < ll:
                ll = prices[k]
        h[j] = hh
        l[j] = ll

    colors = _compute_ha_color(o, h, l, c, n_tf)
    return colors, n_tf


@njit(cache=True)
def _mc_ha_inner(all_colors, all_n_tf, all_bars_per, weights, n_tfs, n, n_lags):
    """JIT inner loop: compute MC_HA(D) and MC_HA(dD) per S5 bar."""
    mc_d = np.zeros(n, dtype=np.float64)
    mc_dd = np.zeros(n, dtype=np.float64)

    for i in range(n):
        wa_d = 0.0
        wa_dd = 0.0
        tw = 0.0

        for tf_idx in range(n_tfs):
            n_tf = all_n_tf[tf_idx]
            if n_tf < n_lags + 3:
                continue

            bp = all_bars_per[tf_idx]
            tf_i = min(i // bp, n_tf - 1)
            w = weights[tf_idx]

            if tf_i < n_lags + 1:
                continue

            # Offset into flat colors array
            offset = 0
            for k in range(tf_idx):
                offset += all_n_tf[k]

            # MC(D): consensus of color changes over last n_lags
            pos = 0
            neg = 0
            for lag in range(n_lags):
                idx = tf_i - lag
                prev = idx - 1
                change = all_colors[offset + idx] - all_colors[offset + prev]
                if change > 0.5:
                    pos += 1
                elif change < -0.5:
                    neg += 1
            mc_d_tf = (pos - neg) / n_lags
            wa_d += w * mc_d_tf

            # MC(dD): acceleration — changes of color changes
            if tf_i >= n_lags + 2:
                # Build d_vals: color[j] - color[j-1] for recent bars
                pos2 = 0
                neg2 = 0
                for lag in range(n_lags):
                    j1 = tf_i - lag
                    j0 = j1 - 1
                    j_1 = j0 - 1
                    d_now = all_colors[offset + j1] - all_colors[offset + j0]
                    d_prev = all_colors[offset + j0] - all_colors[offset + j_1]
                    dd = d_now - d_prev
                    if dd > 0.5:
                        pos2 += 1
                    elif dd < -0.5:
                        neg2 += 1
                mc_dd_tf = (pos2 - neg2) / n_lags
                wa_dd += w * mc_dd_tf

            tw += w

        if tw > 0.0:
            mc_d[i] = wa_d / tw
            mc_dd[i] = wa_dd / tw

    return mc_d, mc_dd


def compute_mc_ha(prices, n_lags=5):
    """Compute MC_HA(D) and MC_HA(dD) across simulated timeframes. JIT-accelerated.

    TFs (in S5 bars): S5=1, S30=6, M1=12, M5=60, H1=720, H4=2880
    Weights: log2(tf_sec/5) + 1 (same formula as MTFMC)

    Returns: mc_d[n], mc_dd[n] arrays (one value per S5 bar)
    """
    n = len(prices)
    tf_bars = np.array([1, 6, 12, 60, 720, 2880], dtype=np.int64)
    tf_sec = [5, 30, 60, 300, 3600, 14400]
    weights = np.array([math.log2(max(s / 5, 1)) + 1 for s in tf_sec], dtype=np.float64)
    n_tfs = len(tf_bars)

    # Precompute HA colors per TF (vectorized/JIT)
    color_arrays = []
    n_tf_list = []
    for bp in tf_bars:
        if bp == 1:
            colors = _compute_ha_color(prices, prices, prices, prices, n)
            color_arrays.append(colors)
            n_tf_list.append(n)
        else:
            colors, n_tf = _resample_and_ha(prices, n, bp)
            color_arrays.append(colors)
            n_tf_list.append(n_tf)

    # Flatten into single array for numba
    all_colors = np.concatenate(color_arrays)
    all_n_tf = np.array(n_tf_list, dtype=np.int64)

    return _mc_ha_inner(all_colors, all_n_tf, tf_bars, weights, n_tfs, n, n_lags)


# ── JIT Evaluator: 3 inputs, 3 outputs (BUY/SELL/FLATTEN) ─────────────────

@njit(cache=True)
def evaluate_sine_3out_jit(
    mc_d, mc_dd, mid_close,
    pip, spread_pips, max_bars, max_hold,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices, start_offset,
):
    """3 inputs [MC_HA(D), MC_HA(dD), UPnL], 3 outputs [BUY, SELL, FLATTEN].
    One position at a time. Spread cost on entry."""
    values = np.zeros(total_values)
    data_len = len(mid_close)
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 10, 10)

    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    n_trades = 0

    position = 0  # -1, 0, +1
    entry_price = 0.0
    entry_bar = 0

    for i in range(start_bar, end_bar):
        # UPnL
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position - spread_pips
        else:
            pnl_pips = 0.0

        # Inputs
        values[0] = mc_d[i]
        values[1] = mc_dd[i]
        values[2] = np.tanh(pnl_pips / 20.0)

        _activate(values, n_inputs, n_eval,
                  node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        out_buy = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_flat = values[output_indices[2]]

        # Force close on max hold
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
            if n_trades < max_trades:
                trade_pnls[n_trades] = pnl
                n_trades += 1
            position = 0

        # Decision: highest output wins
        if position == 0:
            if out_buy > out_sell and out_buy > out_flat:
                position = 1
                entry_price = mid_close[i]
                entry_bar = i
            elif out_sell > out_buy and out_sell > out_flat:
                position = -1
                entry_price = mid_close[i]
                entry_bar = i
        else:
            # Flatten
            if out_flat > out_buy and out_flat > out_sell:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    n_trades += 1
                position = 0
            # Direction flip
            elif position == 1 and out_sell > out_buy and out_sell > out_flat:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    n_trades += 1
                position = -1
                entry_price = mid_close[i]
                entry_bar = i
            elif position == -1 and out_buy > out_sell and out_buy > out_flat:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    n_trades += 1
                position = 1
                entry_price = mid_close[i]
                entry_bar = i

    # Close open position at end
    if position != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar - 1] - entry_price) / pip * position - spread_pips
        if n_trades < max_trades:
            trade_pnls[n_trades] = pnl
            n_trades += 1

    if n_trades < 1:
        return 0, 0.0, -10.0, 0.0, 0.0

    pnls = trade_pnls[:n_trades]
    total_pnl = 0.0
    for j in range(n_trades):
        total_pnl += pnls[j]
    mean_pnl = total_pnl / n_trades
    var = 0.0
    for j in range(n_trades):
        var += (pnls[j] - mean_pnl) ** 2
    std = (var / n_trades) ** 0.5 if n_trades > 1 else 1.0
    sharpe = mean_pnl / std * (n_trades ** 0.5) if std > 0 else 0.0
    wins = 0
    for j in range(n_trades):
        if pnls[j] > 0:
            wins += 1
    win_rate = 100.0 * wins / n_trades

    return n_trades, total_pnl, sharpe, win_rate, mean_pnl


# ── NEAT Evaluator ─────────────────────────────────────────────────────────

class SineEvaluator:
    def __init__(self, mc_d, mc_dd, mid_close, pip=0.0001, spread=2.0,
                 is_split=0.7, max_hold=300):
        n = len(mc_d)
        split = int(n * is_split)
        self.mc_d_is = mc_d[:split]
        self.mc_dd_is = mc_dd[:split]
        self.mid_is = mid_close[:split]
        self.mc_d_oos = mc_d[split:]
        self.mc_dd_oos = mc_dd[split:]
        self.mid_oos = mid_close[split:]
        self.pip = pip
        self.spread = spread
        self.max_hold = max_hold
        self.is_bars = split

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return -10.0

        result = evaluate_sine_3out_jit(
            self.mc_d_is, self.mc_dd_is, self.mid_is,
            self.pip, self.spread, self.is_bars, self.max_hold,
            net[0], net[2], net[3], net[4], net[5], net[6],
            net[7], net[8], net[9], net[10], 0,
        )

        n_trades, total_pnl, sharpe, win_rate, mean_pnl = result

        if n_trades < 5:
            return -10.0 + n_trades * 0.1

        # Sharpe + trade frequency bonus
        data_days = self.is_bars / 288.0  # S5 bars per day
        trades_per_day = n_trades / max(data_days, 1)
        trade_bonus = min(0.1, trades_per_day * 0.01)

        return float(sharpe) + trade_bonus

    def eval_oos(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return {}

        result = evaluate_sine_3out_jit(
            self.mc_d_oos, self.mc_dd_oos, self.mid_oos,
            self.pip, self.spread, len(self.mid_oos), self.max_hold,
            net[0], net[2], net[3], net[4], net[5], net[6],
            net[7], net[8], net[9], net[10], 0,
        )

        n_trades, total_pnl, sharpe, win_rate, mean_pnl = result
        return {
            "n_trades": int(n_trades), "total_pnl": round(float(total_pnl), 1),
            "sharpe": round(float(sharpe), 4), "win_rate": round(float(win_rate), 1),
            "mean_pnl": round(float(mean_pnl), 2),
        }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sine Wave NEAT Sanity Check")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--pop-size", type=int, default=150)
    parser.add_argument("--n-bars", type=int, default=50000)
    parser.add_argument("--period", type=int, default=500,
                        help="Bars per sine cycle (500 ≈ 42 min at S5)")
    parser.add_argument("--amplitude", type=int, default=10,
                        help="Half-amplitude in pips (10 = 20 pip range)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"{'='*60}")
    print(f"  Sine Wave NEAT Sanity Check")
    print(f"  {args.n_bars} bars, period={args.period}, amp=±{args.amplitude} pips")
    print(f"  Spread: 2 pips, {args.generations} gens, pop={args.pop_size}")
    print(f"{'='*60}")

    # Generate data
    print("\nGenerating sine wave...")
    mid_close = generate_sine_wave(
        n_bars=args.n_bars, period_bars=args.period,
        amplitude_pips=args.amplitude,
    )
    print(f"  Price range: {mid_close.min():.4f} - {mid_close.max():.4f}")
    print(f"  Cycles: {args.n_bars / args.period:.1f}")

    # Compute MC_HA features
    print("Computing MC_HA(D) and MC_HA(dD)...")
    t0 = time.time()
    mc_d, mc_dd = compute_mc_ha(mid_close)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  MC_HA(D)  range: [{mc_d.min():.3f}, {mc_d.max():.3f}]")
    print(f"  MC_HA(dD) range: [{mc_dd.min():.3f}, {mc_dd.max():.3f}]")

    # NEAT config — 3 inputs, 3 outputs
    config_file = Path(__file__).parent / "neat_config_3out.ini"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_file))
    # Override: 3 inputs (not 2)
    config.genome_config.num_inputs = 3
    config.pop_size = args.pop_size

    evaluator = SineEvaluator(mc_d, mc_dd, mid_close, max_hold=args.period)

    # Train
    print(f"\nStarting NEAT evolution...")
    t0 = time.time()
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(evaluator.evaluate, args.generations)
    elapsed = time.time() - t0
    print(f"\nTraining: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"IS fitness: {winner.fitness:.4f}")

    # OOS
    oos = evaluator.eval_oos(winner, config)
    print(f"\nOOS Results:")
    print(f"  Trades: {oos['n_trades']}")
    print(f"  P/L: {oos['total_pnl']:+.1f} pips")
    print(f"  Sharpe: {oos['sharpe']:.4f}")
    print(f"  Win Rate: {oos['win_rate']:.1f}%")
    print(f"  Avg P/L: {oos['mean_pnl']:+.2f} pips/trade")

    # Theoretical optimum: buy at bottom, sell at top = ~18 pips/cycle (20 - 2 spread)
    # OOS has ~30% of data = ~30 cycles
    n_oos_cycles = (args.n_bars * 0.3) / args.period
    optimal_pnl = n_oos_cycles * (args.amplitude * 2 - 2)  # 18 pips per cycle
    print(f"\n  Theoretical optimum: ~{optimal_pnl:.0f} pips ({n_oos_cycles:.0f} cycles × {args.amplitude*2-2}p)")
    capture = (oos['total_pnl'] / optimal_pnl * 100) if optimal_pnl > 0 else 0
    print(f"  Capture ratio: {capture:.1f}%")

    if oos['total_pnl'] > 0 and oos['n_trades'] > 10:
        print(f"\n🟢 SANITY CHECK PASSED — NEAT can learn from MC_HA signals")
    else:
        print(f"\n🔴 SANITY CHECK FAILED — training pipeline issue")

    # Save
    results_dir = Path(__file__).parent / "results" / "sine_wave"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "sine_best.pkl", "wb") as f:
        pickle.dump({"genome": winner, "config": config}, f)
    with open(results_dir / "sine_result.json", "w") as f:
        json.dump({"is_fitness": round(float(winner.fitness), 4), "oos": oos,
                    "elapsed_s": round(elapsed, 1), "args": vars(args)}, f, indent=2)
    print(f"\nSaved to {results_dir}/")


if __name__ == "__main__":
    main()
