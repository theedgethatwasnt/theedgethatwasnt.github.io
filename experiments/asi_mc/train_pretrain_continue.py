#!/usr/bin/env python3
"""
ASI-MC Training: Pretrain on sine wave, continue on real data.
==============================================================
Phase 1: 50 gens on sine wave — learn basic buy-low/sell-high
Phase 2: 150 gens on real EUR_JPY M5 — adapt to real market

3 inputs: MC(D), MC(dD), UPnL
3 outputs: BUY, SELL, FLATTEN
Activations: tanh, sin, cos
Spread: 2 pips

Usage:
  python3 train_pretrain_continue.py                    # Local, EUR_JPY
  python3 train_pretrain_continue.py --pairs EUR_JPY,GBP_USD  # Multi-pair
  python3 train_pretrain_continue.py --skip-pretrain    # Real data only
"""

import sys
import os
import gc
import time
import json
import pickle
import argparse
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
from asi_indicator import compute_asi_mc, compute_asi, sma_jit

DATA_DIR = Path(os.environ.get("NEAT_DATA_DIR", str(PROJECT_ROOT / "data" / "scalper_parquet")))
RESULTS_DIR = SCRIPT_DIR / "results"

PAIR_PIP = {
    "EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
    "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
}
PAIR_SPREAD = {
    "EUR_JPY": 2.3, "USD_JPY": 1.7, "GBP_JPY": 3.3, "AUD_JPY": 2.1,
    "CAD_JPY": 2.3, "CHF_JPY": 3.5, "NZD_JPY": 2.7,
    "EUR_USD": 1.6, "GBP_USD": 1.9, "AUD_USD": 1.3,
    "NZD_USD": 1.5, "EUR_GBP": 1.4,
}
ALL_PAIRS = list(PAIR_PIP.keys())


# ── Telegram notifications ─────────────────────────────────────────────────

def tg_send(text):
    try:
        import requests
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        if token and chat:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text, "parse_mode": "HTML"},
                          timeout=5)
    except Exception:
        pass


# ── JIT Evaluator ──────────────────────────────────────────────────────────

@njit(cache=True)
def evaluate_3out_jit(
    mc_d, mc_dd, mid_close,
    pip, spread_pips, max_bars, max_hold,
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight,
    output_indices, start_offset,
):
    """3 inputs [MC(D), MC(dD), UPnL], 3 outputs [BUY, SELL, FLATTEN]."""
    values = np.zeros(total_values)
    data_len = len(mid_close)
    end_bar = min(start_offset + max_bars, data_len - 1)
    start_bar = max(start_offset + 10, 10)

    max_trades = end_bar - start_bar + 1
    trade_pnls = np.zeros(max_trades)
    trade_dirs = np.zeros(max_trades)  # +1 long, -1 short
    n_trades = 0

    position = 0  # -1, 0, +1
    entry_price = 0.0
    entry_bar = 0

    for i in range(start_bar, end_bar):
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position - spread_pips
        else:
            pnl_pips = 0.0

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
                trade_dirs[n_trades] = position
                n_trades += 1
            position = 0

        # Decision
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
            if out_flat > out_buy and out_flat > out_sell:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    trade_dirs[n_trades] = position
                    n_trades += 1
                position = 0
            elif position == 1 and out_sell > out_buy and out_sell > out_flat:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    trade_dirs[n_trades] = position
                    n_trades += 1
                position = -1
                entry_price = mid_close[i]
                entry_bar = i
            elif position == -1 and out_buy > out_sell and out_buy > out_flat:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                if n_trades < max_trades:
                    trade_pnls[n_trades] = pnl
                    trade_dirs[n_trades] = position
                    n_trades += 1
                position = 1
                entry_price = mid_close[i]
                entry_bar = i

    if position != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar - 1] - entry_price) / pip * position - spread_pips
        if n_trades < max_trades:
            trade_pnls[n_trades] = pnl
            trade_dirs[n_trades] = position
            n_trades += 1

    if n_trades < 1:
        return 0, 0.0, -10.0, 0.0, 0.0, 0, 0

    pnls = trade_pnls[:n_trades]
    dirs = trade_dirs[:n_trades]
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
    n_long = 0
    n_short = 0
    for j in range(n_trades):
        if dirs[j] > 0:
            n_long += 1
        else:
            n_short += 1

    return n_trades, total_pnl, sharpe, win_rate, mean_pnl, n_long, n_short


# ── Sine Wave Data ─────────────────────────────────────────────────────────

def generate_sine_ohlc(n_bars=50000, period=500, amp_pips=10,
                       center=1.1244, pip=0.0001, noise_pips=0, seed=42):
    """Generate sine wave with proper OHLC (not just close)."""
    rng = np.random.RandomState(seed)
    t = np.arange(n_bars, dtype=np.float64)
    mid = center + amp_pips * pip * np.sin(2 * np.pi * t / period)
    if noise_pips > 0:
        mid += rng.normal(0, noise_pips * pip, n_bars)

    # Simulate OHLC from mid: open=prev close, high/low = mid ± random
    spread = 0.5 * pip
    o = np.empty(n_bars, dtype=np.float64)
    h = np.empty(n_bars, dtype=np.float64)
    l = np.empty(n_bars, dtype=np.float64)
    c = mid.copy()
    o[0] = mid[0]
    for i in range(1, n_bars):
        o[i] = c[i - 1]
    noise_hl = rng.uniform(0, 2 * pip, n_bars)
    h = np.maximum(o, c) + noise_hl
    l = np.minimum(o, c) - noise_hl

    return o, h, l, c, mid


# ── NEAT Evaluator ─────────────────────────────────────────────────────────

class AsiMcEvaluator:
    def __init__(self, pair_data, pip=0.0001, spread=2.0, max_hold=200):
        """pair_data: dict of {name: (mc_d_is, mc_dd_is, mid_is, mc_d_oos, mc_dd_oos, mid_oos)}"""
        self.pair_data = pair_data
        self.pip = pip
        self.spread = spread
        self.max_hold = max_hold

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return -10.0

        total_sharpe = 0.0
        total_trades = 0
        total_bars = 0
        total_longs = 0
        total_shorts = 0
        n_pairs = 0

        for name, (mc_d_is, mc_dd_is, mid_is, *_) in self.pair_data.items():
            pip = PAIR_PIP.get(name, self.pip)
            spread = PAIR_SPREAD.get(name, self.spread)

            result = evaluate_3out_jit(
                mc_d_is, mc_dd_is, mid_is,
                pip, spread, len(mid_is), self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10], 0,
            )

            nt, pnl, sh, wr, mpnl, nl, ns = result
            if nt >= 5:
                total_sharpe += sh
                n_pairs += 1
            total_trades += nt
            total_bars += len(mid_is)
            total_longs += nl
            total_shorts += ns

        if n_pairs == 0 or total_trades < 10:
            return -10.0

        avg_sharpe = total_sharpe / n_pairs

        # Trade frequency bonus
        data_days = total_bars / 288.0
        trades_per_day = total_trades / max(data_days, 1)
        trade_bonus = min(0.1, trades_per_day * 0.01)

        # Bidirectional bonus: penalize if >80% one direction
        if total_trades > 0:
            dir_ratio = min(total_longs, total_shorts) / max(total_longs + total_shorts, 1)
            if dir_ratio < 0.2:
                avg_sharpe *= 0.5  # Heavy penalty for unidirectional

        return avg_sharpe + trade_bonus

    def eval_oos(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return {}

        results = {}
        for name, data in self.pair_data.items():
            mc_d_oos, mc_dd_oos, mid_oos = data[3], data[4], data[5]
            pip = PAIR_PIP.get(name, self.pip)
            spread = PAIR_SPREAD.get(name, self.spread)

            result = evaluate_3out_jit(
                mc_d_oos, mc_dd_oos, mid_oos,
                pip, spread, len(mid_oos), self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10], 0,
            )
            nt, pnl, sh, wr, mpnl, nl, ns = result
            results[name] = {
                "n_trades": int(nt), "total_pnl": round(float(pnl), 1),
                "sharpe": round(float(sh), 4), "win_rate": round(float(wr), 1),
                "n_long": int(nl), "n_short": int(ns),
            }
        return results


# ── Data Loading ───────────────────────────────────────────────────────────

def load_pair_m5_ohlc(pair):
    """Load S5 parquet, resample to M5, return OHLC + mid arrays."""
    parquet = DATA_DIR / f"{pair.replace('_', '')}_S5_BA.parquet"
    print(f"  Loading {pair}...", end=" ", flush=True)
    df = pd.read_parquet(parquet, engine="pyarrow")
    print(f"{len(df):,} S5 → ", end="", flush=True)

    ts = pd.to_datetime(df["timestamp"])
    # Use mid prices (matches curator which uses OANDA mid OHLC)
    half_spread = (df["ask_c"].values - df["bid_c"].values) / 2.0
    ohlc = pd.DataFrame({
        "o": df["bid_o"].values + half_spread,
        "h": df["bid_h"].values + half_spread,
        "l": df["bid_l"].values + half_spread,
        "c": df["bid_c"].values + half_spread,
        "mid": df["bid_c"].values + half_spread,
    }, index=ts)
    r = ohlc.resample("5min").agg({
        "o": "first", "h": "max", "l": "min", "c": "last", "mid": "last"
    }).dropna()

    o = r["o"].values.astype(np.float64)
    h = r["h"].values.astype(np.float64)
    l = r["l"].values.astype(np.float64)
    c = r["c"].values.astype(np.float64)
    mid = r["mid"].values.astype(np.float64)
    print(f"{len(o):,} M5 bars")

    del df
    gc.collect()
    return o, h, l, c, mid


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="EUR_JPY",
                        help="Comma-separated pairs for training")
    parser.add_argument("--pretrain-gens", type=int, default=50)
    parser.add_argument("--real-gens", type=int, default=150)
    parser.add_argument("--pop-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--max-hold", type=int, default=200)
    args = parser.parse_args()

    np.random.seed(args.seed)
    train_pairs = [p.strip() for p in args.pairs.split(",")]
    total_gens = (0 if args.skip_pretrain else args.pretrain_gens) + args.real_gens

    print(f"{'='*60}")
    print(f"  ASI-MC: Pretrain→Continue")
    print(f"  Pairs: {train_pairs}")
    print(f"  Pretrain: {args.pretrain_gens} gens on sine")
    print(f"  Real: {args.real_gens} gens on market data")
    print(f"  Activations: tanh, sin, cos")
    print(f"{'='*60}")

    tg_send(f"🔬 ASI-MC starting\nPairs: {train_pairs}\n"
            f"Pretrain: {args.pretrain_gens}g sine → {args.real_gens}g real")

    # ── NEAT config ──
    config_file = SCRIPT_DIR / "neat_config_3out.ini"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_file))
    config.pop_size = args.pop_size

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Pretrain on sine wave
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_pretrain:
        print(f"\n── Phase 1: Sine pretrain ({args.pretrain_gens} gens) ──")
        o_s, h_s, l_s, c_s, mid_s = generate_sine_ohlc(50000, 500, 10)
        n_s = len(mid_s)
        mc_d_s, mc_dd_s = compute_asi_mc(o_s, h_s, l_s, c_s, n_s)
        split = int(n_s * 0.7)

        sine_data = {"sine": (
            mc_d_s[:split], mc_dd_s[:split], mid_s[:split],
            mc_d_s[split:], mc_dd_s[split:], mid_s[split:],
        )}

        sine_eval = AsiMcEvaluator(sine_data, pip=0.0001, spread=2.0,
                                    max_hold=500)

        pop = neat.Population(config)
        pop.add_reporter(neat.StdOutReporter(False))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)

        t0 = time.time()
        winner = pop.run(sine_eval.evaluate, args.pretrain_gens)
        elapsed = time.time() - t0

        # Check sine OOS
        sine_oos = sine_eval.eval_oos(winner, config)
        sr = sine_oos.get("sine", {})
        print(f"\n  Sine pretrain done in {elapsed:.0f}s")
        print(f"  IS fitness: {winner.fitness:.4f}")
        print(f"  OOS: {sr.get('n_trades',0)}T {sr.get('total_pnl',0):+.1f}p "
              f"L={sr.get('n_long',0)} S={sr.get('n_short',0)}")

        tg_send(f"✅ Sine pretrain done ({elapsed:.0f}s)\n"
                f"Fitness: {winner.fitness:.4f}\n"
                f"OOS: {sr.get('n_trades',0)}T {sr.get('total_pnl',0):+.1f}p "
                f"L={sr.get('n_long',0)} S={sr.get('n_short',0)}")

        del o_s, h_s, l_s, c_s, mid_s, mc_d_s, mc_dd_s
        gc.collect()
    else:
        print("\n── Skipping pretrain ──")
        pop = neat.Population(config)
        pop.add_reporter(neat.StdOutReporter(False))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Continue on real market data
    # ══════════════════════════════════════════════════════════════════
    print(f"\n── Phase 2: Real data ({args.real_gens} gens) ──")
    print("Loading market data...")

    real_data = {}
    for pair in train_pairs:
        o, h, l, c, mid = load_pair_m5_ohlc(pair)
        n = len(o)
        print(f"  Computing ASI-MC for {pair}...", end=" ", flush=True)
        mc_d, mc_dd = compute_asi_mc(o, h, l, c, n)
        print(f"done. MC_D [{mc_d.min():.3f},{mc_d.max():.3f}]")

        split = int(n * 0.7)
        real_data[pair] = (
            mc_d[:split], mc_dd[:split], mid[:split],
            mc_d[split:], mc_dd[split:], mid[split:],
        )
        del o, h, l, c
        gc.collect()

    real_eval = AsiMcEvaluator(real_data, max_hold=args.max_hold)

    t0 = time.time()
    winner = pop.run(real_eval.evaluate, args.real_gens)
    elapsed = time.time() - t0

    print(f"\n  Real training done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  IS fitness: {winner.fitness:.4f}")
    print(f"  Network size: {winner.size()}")

    # ── OOS on training pairs ──
    oos = real_eval.eval_oos(winner, config)
    total_pnl = sum(r["total_pnl"] for r in oos.values())
    total_trades = sum(r["n_trades"] for r in oos.values())
    total_long = sum(r["n_long"] for r in oos.values())
    total_short = sum(r["n_short"] for r in oos.values())

    print(f"\n{'='*60}")
    print(f"  OOS RESULTS (training pairs)")
    print(f"{'='*60}")
    print(f"{'Pair':<10} {'Trades':>7} {'PnL':>9} {'Sharpe':>8} {'WR':>6} {'Long':>5} {'Short':>5}")
    print("-" * 60)
    for pair in train_pairs:
        r = oos.get(pair, {})
        print(f"{pair:<10} {r.get('n_trades',0):>7} {r.get('total_pnl',0):>+9.1f} "
              f"{r.get('sharpe',0):>8.4f} {r.get('win_rate',0):>5.1f}% "
              f"{r.get('n_long',0):>5} {r.get('n_short',0):>5}")
    print(f"\n  TOTAL: {total_trades}T {total_pnl:+.1f}p L={total_long} S={total_short}")

    tg_send(f"🏁 ASI-MC training done ({elapsed:.0f}s)\n"
            f"Fitness: {winner.fitness:.4f}\n"
            f"OOS: {total_trades}T {total_pnl:+.1f}p\n"
            f"Long={total_long} Short={total_short}\n"
            f"Size: {winner.size()}")

    # ── OOS on ALL 12 pairs (if we have more data) ──
    if len(train_pairs) < len(ALL_PAIRS):
        print(f"\n  Loading remaining pairs for full OOS...")
        all_oos_data = dict(real_data)  # Start with training pairs
        for pair in ALL_PAIRS:
            if pair in all_oos_data:
                continue
            try:
                o, h, l, c, mid = load_pair_m5_ohlc(pair)
                n = len(o)
                mc_d, mc_dd = compute_asi_mc(o, h, l, c, n)
                split = int(n * 0.7)
                all_oos_data[pair] = (
                    mc_d[:split], mc_dd[:split], mid[:split],
                    mc_d[split:], mc_dd[split:], mid[split:],
                )
                del o, h, l, c
                gc.collect()
            except Exception as e:
                print(f"    {pair}: SKIP ({e})")

        full_eval = AsiMcEvaluator(all_oos_data, max_hold=args.max_hold)
        full_oos = full_eval.eval_oos(winner, config)

        print(f"\n{'='*60}")
        print(f"  FULL OOS (all pairs)")
        print(f"{'='*60}")
        full_total = 0.0
        full_trades = 0
        full_long = full_short = 0
        for pair in ALL_PAIRS:
            r = full_oos.get(pair, {})
            if not r:
                continue
            print(f"{pair:<10} {r.get('n_trades',0):>7} {r.get('total_pnl',0):>+9.1f} "
                  f"{r.get('sharpe',0):>8.4f} {r.get('win_rate',0):>5.1f}% "
                  f"{r.get('n_long',0):>5} {r.get('n_short',0):>5}")
            full_total += r.get("total_pnl", 0)
            full_trades += r.get("n_trades", 0)
            full_long += r.get("n_long", 0)
            full_short += r.get("n_short", 0)
        print(f"\n  ALL 12: {full_trades}T {full_total:+.1f}p L={full_long} S={full_short}")

        tg_send(f"📊 Full 12-pair OOS\n"
                f"{full_trades}T {full_total:+.1f}p\n"
                f"L={full_long} S={full_short}")
        oos = full_oos

    # Save
    genome_path = RESULTS_DIR / f"asi_mc_s{args.seed}_best.pkl"
    with open(genome_path, "wb") as f:
        pickle.dump({"genome": winner, "config": config}, f)

    result = {
        "seed": args.seed, "pretrain_gens": args.pretrain_gens,
        "real_gens": args.real_gens, "pop_size": args.pop_size,
        "train_pairs": train_pairs,
        "is_fitness": round(float(winner.fitness), 4),
        "network_size": list(winner.size()),
        "oos": {k: v for k, v in oos.items()},
        "elapsed_s": round(elapsed, 1),
    }
    with open(RESULTS_DIR / f"asi_mc_s{args.seed}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}/")
    tg_send(f"💾 Saved asi_mc_s{args.seed}_best.pkl")


if __name__ == "__main__":
    main()
