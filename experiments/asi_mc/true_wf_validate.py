#!/usr/bin/env python3
"""
TRUE Walk-Forward Validation for ASI-MC
========================================
For each WF window:
  1. Train a FRESH genome on IS portion (pretrain sine + continue on IS)
  2. Evaluate on OOS portion (never seen during training)
  3. Collect detailed per-trade metrics

WF Windows (on each pair's M5 data):
  Split 1: IS 0-60%, OOS 60-100%
  Split 2: IS 0-50%, OOS 50-100%
  Split 3: IS 0-70%, OOS 70-100%

Per-trade metrics collected:
  - P/L pips, direction, hold bars, MFE, MAE, entry/exit bar

Aggregated per pair per split:
  - trades/day, pips/day, avg pnl, avg hold, avg DD, max DD, WR, Sharpe
"""

import sys
import os
import gc
import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2] if len(SCRIPT_DIR.parts) > 4 else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import neat
from lib.fast_eval import extract_network, _activate
from asi_indicator import compute_asi_mc
from train_pretrain_continue import (
    generate_sine_ohlc, AsiMcEvaluator, load_pair_m5_ohlc, tg_send,
    PAIR_PIP, PAIR_SPREAD, ALL_PAIRS, DATA_DIR,
)

RESULTS_DIR = SCRIPT_DIR / "results" / "true_wf"


# ── Detailed trade collector (JIT) ────────────────────────────────────────

@njit(cache=True)
def collect_trades_jit(
    mc_d, mc_dd, mid_close,
    pip, spread_pips, max_bars, max_hold,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices, start_offset,
):
    """Run genome, return per-trade arrays: pnl, direction, hold_bars, mfe, mae."""
    values = np.zeros(total_values)
    data_len = len(mid_close)
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 10, 10)

    cap = end_bar - start_bar + 1
    t_pnl = np.zeros(cap, dtype=np.float64)
    t_dir = np.zeros(cap, dtype=np.float64)
    t_hold = np.zeros(cap, dtype=np.float64)
    t_mfe = np.zeros(cap, dtype=np.float64)
    t_mae = np.zeros(cap, dtype=np.float64)
    n_trades = 0

    position = 0
    entry_price = 0.0
    entry_bar = 0
    mfe = 0.0
    mae = 0.0

    for i in range(start_bar, end_bar):
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position - spread_pips
        else:
            pnl_pips = 0.0

        # Track MFE/MAE
        if position != 0:
            raw_pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
            if raw_pnl > mfe:
                mfe = raw_pnl
            if -raw_pnl > mae:
                mae = -raw_pnl

        values[0] = mc_d[i]
        values[1] = mc_dd[i]
        values[2] = np.tanh(pnl_pips / 20.0)

        _activate(values, n_inputs, n_eval,
                  node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        ob = values[output_indices[0]]
        os_ = values[output_indices[1]]
        of = values[output_indices[2]]

        # Max hold
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
            if n_trades < cap:
                t_pnl[n_trades] = pnl
                t_dir[n_trades] = position
                t_hold[n_trades] = i - entry_bar
                t_mfe[n_trades] = mfe
                t_mae[n_trades] = mae
                n_trades += 1
            position = 0
            mfe = 0.0
            mae = 0.0

        if position == 0:
            if ob > os_ and ob > of:
                position = 1; entry_price = mid_close[i]; entry_bar = i; mfe = 0.0; mae = 0.0
            elif os_ > ob and os_ > of:
                position = -1; entry_price = mid_close[i]; entry_bar = i; mfe = 0.0; mae = 0.0
        else:
            close = False
            new_pos = 0
            if of > ob and of > os_:
                close = True
            elif position == 1 and os_ > ob and os_ > of:
                close = True; new_pos = -1
            elif position == -1 and ob > os_ and ob > of:
                close = True; new_pos = 1

            if close:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                if n_trades < cap:
                    t_pnl[n_trades] = pnl
                    t_dir[n_trades] = position
                    t_hold[n_trades] = i - entry_bar
                    t_mfe[n_trades] = mfe
                    t_mae[n_trades] = mae
                    n_trades += 1
                if new_pos != 0:
                    position = new_pos; entry_price = mid_close[i]; entry_bar = i
                    mfe = 0.0; mae = 0.0
                else:
                    position = 0; mfe = 0.0; mae = 0.0

    # Close open
    if position != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar - 1] - entry_price) / pip * position - spread_pips
        if n_trades < cap:
            t_pnl[n_trades] = pnl
            t_dir[n_trades] = position
            t_hold[n_trades] = (end_bar - 1) - entry_bar
            t_mfe[n_trades] = mfe
            t_mae[n_trades] = mae
            n_trades += 1

    return t_pnl[:n_trades], t_dir[:n_trades], t_hold[:n_trades], t_mfe[:n_trades], t_mae[:n_trades]


def compute_detailed_metrics(pnls, dirs, holds, mfes, maes, n_bars, bars_per_day=288.0):
    """Compute full metrics from trade arrays."""
    n = len(pnls)
    if n == 0:
        return {"n_trades": 0, "error": "no trades"}

    days = n_bars / bars_per_day
    total_pnl = float(pnls.sum())
    wins = int((pnls > 0).sum())
    losses = n - wins
    wr = wins / n * 100

    avg_pnl = float(pnls.mean())
    std_pnl = float(pnls.std()) if n > 1 else 1.0
    sharpe = avg_pnl / std_pnl * (n ** 0.5) if std_pnl > 0 else 0.0

    avg_win = float(pnls[pnls > 0].mean()) if wins > 0 else 0.0
    avg_loss = float(pnls[pnls <= 0].mean()) if losses > 0 else 0.0

    avg_hold_bars = float(holds.mean())
    avg_hold_hrs = avg_hold_bars * 5 / 60  # M5 bars to hours

    avg_mfe = float(mfes.mean())
    avg_mae = float(maes.mean())

    # Drawdown
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max())
    avg_dd = float(dd.mean())

    n_long = int((dirs > 0).sum())
    n_short = int((dirs < 0).sum())

    return {
        "n_trades": n, "total_pnl": round(total_pnl, 1),
        "trades_per_day": round(n / max(days, 0.1), 2),
        "pips_per_day": round(total_pnl / max(days, 0.1), 1),
        "sharpe": round(sharpe, 4),
        "win_rate": round(wr, 1),
        "avg_pnl": round(avg_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_hold_hrs": round(avg_hold_hrs, 1),
        "avg_mfe": round(avg_mfe, 1),
        "avg_mae": round(avg_mae, 1),
        "avg_dd": round(avg_dd, 1),
        "max_dd": round(max_dd, 1),
        "n_long": n_long, "n_short": n_short,
        "days": round(days, 1),
    }


def train_fresh(config, pair_data_is, pretrain_gens=50, real_gens=100, pop_size=150, seed=42):
    """Train a fresh genome: pretrain on sine, continue on IS data."""
    np.random.seed(seed)

    # Phase 1: Sine pretrain
    o_s, h_s, l_s, c_s, mid_s = generate_sine_ohlc(50000, 500, 10, seed=seed)
    n_s = len(mid_s)
    mc_d_s, mc_dd_s = compute_asi_mc(o_s, h_s, l_s, c_s, n_s)
    split = int(n_s * 0.7)
    sine_data = {"sine": (mc_d_s[:split], mc_dd_s[:split], mid_s[:split],
                          mc_d_s[split:], mc_dd_s[split:], mid_s[split:])}
    sine_eval = AsiMcEvaluator(sine_data, pip=0.0001, spread=2.0, max_hold=500)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.run(sine_eval.evaluate, pretrain_gens)

    del o_s, h_s, l_s, c_s, mid_s, mc_d_s, mc_dd_s
    gc.collect()

    # Phase 2: Continue on real IS data
    real_eval = AsiMcEvaluator(pair_data_is, max_hold=200)
    winner = pop.run(real_eval.evaluate, real_gens)

    return winner


def main():
    print(f"{'='*70}")
    print(f"  TRUE Walk-Forward Validation (retrain per window)")
    print(f"{'='*70}")

    tg_send("🔬 TRUE WF starting — retrains per window, full trade metrics")

    config_file = SCRIPT_DIR / "neat_config_3out.ini"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_file))
    config.pop_size = 150

    WF_SPLITS = [
        ("60/40", 0.6),
        ("50/50", 0.5),
        ("70/30", 0.7),
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ── Precompute indicators for all 12 pairs (once) ──
    print("\nPrecomputing ASI-MC indicators for all 12 pairs...")
    pair_cache = {}  # pair -> (o, h, l, c, mid, mc_d_full, mc_dd_full)
    for pair in ALL_PAIRS:
        o, h, l, c, mid = load_pair_m5_ohlc(pair)
        n = len(o)
        print(f"  Computing ASI-MC for {pair}...", end=" ", flush=True)
        mc_d_full, mc_dd_full = compute_asi_mc(o, h, l, c, n)
        print("done")
        pair_cache[pair] = (o, h, l, c, mid, mc_d_full, mc_dd_full, n)
        del o, h, l, c  # keep mid, mc_d, mc_dd
        gc.collect()

    tg_send(f"📦 All 12 pairs precomputed. Starting 3 WF splits...")

    for split_name, is_pct in WF_SPLITS:
        print(f"\n{'='*70}")
        print(f"  WF SPLIT: {split_name} (IS={is_pct*100:.0f}%)")
        print(f"{'='*70}")

        tg_send(f"🔄 WF {split_name}: training fresh genome...")

        # Get EUR_JPY from cache for training
        _, _, _, _, mid_ej, mc_d_ej, mc_dd_ej, n_ej = pair_cache["EUR_JPY"]
        split_idx = int(n_ej * is_pct)

        # Use cached full-array indicators, slice IS portion for training
        train_split = int(split_idx * 0.8)
        is_data = {"EUR_JPY": (
            mc_d_ej[:train_split], mc_dd_ej[:train_split], mid_ej[:train_split],
            mc_d_ej[train_split:split_idx], mc_dd_ej[train_split:split_idx],
            mid_ej[train_split:split_idx],
        )}

        t0 = time.time()
        winner = train_fresh(config, is_data, pretrain_gens=50, real_gens=100,
                             pop_size=150, seed=42)
        train_time = time.time() - t0
        print(f"  Trained in {train_time:.0f}s, fitness={winner.fitness:.4f}")

        gpath = RESULTS_DIR / f"wf_{split_name.replace('/', '_')}_genome.pkl"
        with open(gpath, "wb") as f:
            pickle.dump({"genome": winner, "config": config}, f)

        net = extract_network(winner, config)
        split_results = {}

        # Evaluate on ALL 12 pairs OOS (from cache)
        for pair in ALL_PAIRS:
            _, _, _, _, mid_p, mc_d_p, mc_dd_p, n_p = pair_cache[pair]
            split_p = int(n_p * is_pct)
            pip = PAIR_PIP[pair]
            spread = PAIR_SPREAD[pair]

            pnls, dirs, holds, mfes, maes = collect_trades_jit(
                mc_d_p[split_p:], mc_dd_p[split_p:], mid_p[split_p:],
                pip, spread, n_p - split_p, 200,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10], 0,
            )

            metrics = compute_detailed_metrics(pnls, dirs, holds, mfes, maes,
                                                n_p - split_p)
            split_results[pair] = metrics

            print(f"  {pair:<10} {metrics['n_trades']:>5}T {metrics['total_pnl']:>+9.1f}p "
                  f"{metrics['pips_per_day']:>+6.1f}p/d "
                  f"Sh={metrics['sharpe']:>5.2f} WR={metrics['win_rate']:>4.0f}% "
                  f"Hold={metrics['avg_hold_hrs']:.1f}h "
                  f"MFE={metrics['avg_mfe']:.1f} MAE={metrics['avg_mae']:.1f} "
                  f"MxDD={metrics['max_dd']:.0f} "
                  f"L={metrics['n_long']} S={metrics['n_short']}")

        all_results[split_name] = split_results

        total_pnl = sum(r["total_pnl"] for r in split_results.values())
        total_trades = sum(r["n_trades"] for r in split_results.values())
        n_profitable = sum(1 for r in split_results.values() if r.get("total_pnl", 0) > 0)
        tg_send(f"✅ WF {split_name} done ({train_time:.0f}s)\n"
                f"{total_trades}T {total_pnl:+.0f}p\n"
                f"{n_profitable}/12 pairs profitable")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  TRUE WF SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Pair':<10}", end="")
    for sn, _ in WF_SPLITS:
        print(f" {'PnL_'+sn:>10} {'T/d':>5} {'p/d':>7} {'MxDD':>6}", end="")
    print()
    print("-" * 100)

    pair_pass_count = {}
    for pair in ALL_PAIRS:
        print(f"{pair:<10}", end="")
        passes = 0
        for sn, _ in WF_SPLITS:
            r = all_results.get(sn, {}).get(pair, {})
            pnl = r.get("total_pnl", 0)
            tpd = r.get("trades_per_day", 0)
            ppd = r.get("pips_per_day", 0)
            mdd = r.get("max_dd", 0)
            print(f" {pnl:>+10.0f} {tpd:>5.1f} {ppd:>+7.1f} {mdd:>6.0f}", end="")
            if pnl > 0:
                passes += 1
        pair_pass_count[pair] = passes
        mark = " ✓" if passes == 3 else ""
        print(mark)

    full_pass = sum(1 for v in pair_pass_count.values() if v == 3)
    print(f"\n  {full_pass}/12 pairs profitable in ALL 3 splits")

    if full_pass >= 8:
        verdict = "🟢 TRUE WF VALIDATED"
    elif full_pass >= 5:
        verdict = "🟡 MARGINAL"
    else:
        verdict = "🔴 FAILED"
    print(f"  {verdict}")

    tg_send(f"🏁 TRUE WF Complete\n\n"
            f"{full_pass}/12 pairs pass all 3 splits\n{verdict}")

    # Save
    with open(RESULTS_DIR / "true_wf_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
