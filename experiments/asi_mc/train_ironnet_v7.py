#!/usr/bin/env python3
"""
IronNet V7: 7-Input SHAP-Selected Per-Pair Training
=====================================================
Architecture: 7 inputs → 5 hidden → 3 outputs, fully connected + skip (71 conn).
Activations: {tanh, sin, cos, gauss}. Fixed topology — no add/delete nodes.

Inputs (SHAP rank):
  [0] bb_width      (#1, volatility)     [0, ~0.05] → ×20 → [0, ~1]
  [1] stoch_d       (#2, oscillator)     [0, 1]
  [2] macd_hist     (#3, momentum)       [-2, +2] → /2 → [-1, +1]
  [3] range_pos_30  (#4, swing)          [0, 1]
  [4] aroon_osc     (#6, trend)          [-1, +1]
  [5] mc_d_a        (#13, ASI momentum)  [-1, +1]
  [6] UPnL          (trade state)        (-1, +1)

Fitness: WF-in-fitness (3 chunks), PnL/MAE, bidirectional enforcement (≥15%).
Training on H1 cadence (M5 indicators resampled via .last()).

Usage:
  python3 train_ironnet_v7.py --pair EUR_GBP --seed 42
"""

import sys, os, gc, time, copy, json, pickle, math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import neat
from lib.fast_eval import extract_network, _activate

V7_DATA_DIR = Path(os.environ.get("V7_DATA_DIR",
                str(PROJECT_ROOT / "data" / "v7_indicators")))
RESULTS_DIR = SCRIPT_DIR / "results" / "ironnet_v7"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
PAIR_MIN_SWING = {
    "EUR_JPY": 30, "USD_JPY": 25, "GBP_JPY": 40, "AUD_JPY": 25,
    "CAD_JPY": 30, "CHF_JPY": 35, "NZD_JPY": 25,
    "EUR_USD": 20, "GBP_USD": 25, "AUD_USD": 18,
    "NZD_USD": 18, "EUR_GBP": 15,
}

N_INPUTS = 7   # 6 indicators + UPnL
N_HIDDEN = 5
N_OUTPUTS = 3  # BUY, SELL, FLATTEN
N_CHUNKS = 3

IND_COLS = ["bb_width", "stoch_d", "macd_hist", "range_pos_30", "aroon_osc", "mc_d_a"]


# ── Telegram notification (optional) ──
def tg_send(msg):
    try:
        import requests
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        if token and chat:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": msg}, timeout=5)
    except Exception:
        pass


# ── Input normalization ──
def normalize_inputs(inputs_2d):
    """Scale raw indicators to NEAT-friendly ranges.
    Row 0: bb_width ×20 → [0, ~1]
    Row 2: macd_hist /2 → [-1, +1]
    Others: already in [0,1] or [-1,+1]
    """
    out = inputs_2d.copy()
    out[0] = out[0] * 20.0          # bb_width: [0, 0.05] → [0, 1]
    out[2] = np.clip(out[2] / 2.0, -1.0, 1.0)  # macd_hist: [-2,+2] → [-1,+1]
    return out


# ── Zigzag label generation (from train_ironnet_perpair.py) ──
@njit(cache=True)
def generate_zigzag_labels(mid_close, pip, min_swing_pips, label_window=6, min_mfe_pips=3.0):
    n = len(mid_close)
    labels = np.zeros(n, dtype=np.int64)
    min_swing = min_swing_pips * pip
    min_mfe = min_mfe_pips * pip
    running_high = mid_close[0]
    running_low = mid_close[0]
    direction = 0
    for i in range(1, n):
        price = mid_close[i]
        if price > running_high: running_high = price
        if price < running_low: running_low = price
        if direction == 0:
            if running_high - price >= min_swing: direction = -1; running_low = price
            elif price - running_low >= min_swing: direction = 1; running_high = price
        elif direction == 1:
            if running_high - price >= min_swing:
                mfe = 0.0
                for j in range(i, min(i + 200, n)):
                    dd = running_high - mid_close[j]
                    if dd > mfe: mfe = dd
                if mfe > min_mfe:
                    for k in range(i, min(i + label_window, n)): labels[k] = 2
                direction = -1; running_low = price
        else:
            if price - running_low >= min_swing:
                mfe = 0.0
                for j in range(i, min(i + 200, n)):
                    uu = mid_close[j] - running_low
                    if uu > mfe: mfe = uu
                if mfe > min_mfe:
                    for k in range(i, min(i + label_window, n)): labels[k] = 1
                direction = 1; running_high = price
    return labels


# ── JIT evaluators (reused from train_ironnet_perpair.py) ──

@njit(cache=True)
def evaluate_supervised_v7(inputs_2d, mid_close, labels, pip, spread_pips,
                            n_inputs, n_eval, total_values,
                            node_bias, node_response, node_act,
                            conn_from, conn_to, conn_weight,
                            output_indices,
                            start_bar=10, end_bar=-1):
    n_ind = inputs_2d.shape[0]
    n = inputs_2d.shape[1]
    if end_bar < 0: end_bar = n - 1
    values = np.zeros(total_values)
    correct = 0
    total = 0
    for i in range(start_bar, end_bar):
        lab = labels[i]
        for k in range(n_ind):
            values[k] = inputs_2d[k, i]
        values[n_ind] = 0.0  # UPnL = 0 for supervised
        _activate(values, n_eval, total_values, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)
        best_out = -1; best_val = -1e30
        for oi in range(len(output_indices)):
            v = values[output_indices[oi]]
            if v > best_val: best_val = v; best_out = oi
        # BUY=0→label1, SELL=1→label2, FLATTEN=2→label0
        pred_label = 1 if best_out == 0 else (2 if best_out == 1 else 0)
        if pred_label == lab: correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


@njit(cache=True)
def evaluate_chunk_v7(inputs_2d, mid_close, pip, spread_pips, max_hold,
                       n_inputs, n_eval, total_values,
                       node_bias, node_response, node_act,
                       conn_from, conn_to, conn_weight,
                       output_indices,
                       c_start, c_end):
    n_ind = inputs_2d.shape[0]
    values = np.zeros(total_values)
    position = 0
    entry_price = 0.0
    entry_bar = 0
    total_pnl = 0.0
    total_mae = 0.0
    n_trades = 0
    n_long = 0
    n_short = 0
    worst_ae = 0.0
    max_t = c_end - c_start
    trade_pnls = np.zeros(max_t)
    trade_maes = np.zeros(max_t)

    for i in range(c_start + 10, c_end - 1):
        mid = mid_close[i]
        for k in range(n_ind):
            values[k] = inputs_2d[k, i]
        # UPnL
        if position != 0 and entry_price > 0:
            pnl_pips = (mid - entry_price) * position / pip - spread_pips
            values[n_ind] = np.tanh(pnl_pips / 20.0)
        else:
            values[n_ind] = 0.0

        _activate(values, n_eval, total_values, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        best_out = -1; best_val = -1e30
        for oi in range(len(output_indices)):
            v = values[output_indices[oi]]
            if v > best_val: best_val = v; best_out = oi

        # Max hold
        if position != 0 and (i - entry_bar) >= max_hold:
            best_out = 2  # force FLATTEN

        if best_out == 0 and position <= 0:  # BUY
            if position == -1:
                pnl = (entry_price - mid) / pip - spread_pips
                trade_pnls[n_trades] = pnl
                trade_maes[n_trades] = worst_ae
                n_trades += 1
            position = 1
            entry_price = mid + spread_pips * pip  # spread at entry
            entry_bar = i
            worst_ae = spread_pips
            n_long += 1
        elif best_out == 1 and position >= 0:  # SELL
            if position == 1:
                pnl = (mid - entry_price) / pip - spread_pips
                trade_pnls[n_trades] = pnl
                trade_maes[n_trades] = worst_ae
                n_trades += 1
            position = -1
            entry_price = mid - spread_pips * pip
            entry_bar = i
            worst_ae = spread_pips
            n_short += 1
        elif best_out == 2 and position != 0:  # FLATTEN
            if position == 1:
                pnl = (mid - entry_price) / pip - spread_pips
            else:
                pnl = (entry_price - mid) / pip - spread_pips
            trade_pnls[n_trades] = pnl
            trade_maes[n_trades] = worst_ae
            n_trades += 1
            position = 0
            entry_price = 0.0

        # Track MAE
        if position != 0 and entry_price > 0:
            if position == 1:
                ae = (entry_price - mid) / pip
            else:
                ae = (mid - entry_price) / pip
            if ae > worst_ae:
                worst_ae = ae

    if n_trades == 0:
        return 0, 0.0, 0.0, 0, 0

    total_pnl = 0.0; total_mae = 0.0
    for j in range(n_trades):
        total_pnl += trade_pnls[j]
        total_mae += trade_maes[j]
    avg_mae = total_mae / n_trades
    return n_trades, total_pnl, avg_mae, n_long, n_short


# ── WF Evaluator ──

class V7WFEvaluator:
    def __init__(self, inputs_2d, mid_close, pip, spread,
                 max_hold=17, n_chunks=3, min_dir_ratio=0.15):
        self.inputs_2d = inputs_2d
        self.mid_close = mid_close
        self.pip = pip
        self.spread = spread
        self.max_hold = max_hold
        self.n_chunks = n_chunks
        self.min_dir_ratio = min_dir_ratio
        self.n_bars = inputs_2d.shape[1]

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness(self, genome, config):
        try:
            net = extract_network(genome, config)
        except Exception:
            return -10.0

        chunk_scores = []
        total_long = 0; total_short = 0; total_trades = 0

        for ci in range(self.n_chunks):
            c_start = int(self.n_bars * ci / self.n_chunks)
            c_end = int(self.n_bars * (ci + 1) / self.n_chunks)

            nt, pnl, mae, nl, ns = evaluate_chunk_v7(
                self.inputs_2d, self.mid_close,
                self.pip, self.spread, self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10],
                c_start, c_end)

            total_long += nl; total_short += ns; total_trades += nt

            min_t = max(10, int(self.n_bars / self.n_chunks / 288 * 0.3))
            if nt < min_t: return -10.0
            if pnl <= 0: return -10.0

            mean_pnl = pnl / nt
            score = mean_pnl / mae if mae > 0 else mean_pnl
            chunk_scores.append(score * (nt ** 0.5))

        if total_trades < 10: return -10.0
        dir_ratio = min(total_long, total_short) / total_trades
        if dir_ratio < self.min_dir_ratio: return -10.0

        min_score = min(chunk_scores)
        mean_score = sum(chunk_scores) / len(chunk_scores)
        cv = (sum((s - mean_score)**2 for s in chunk_scores) / len(chunk_scores))**0.5 / mean_score if mean_score > 0 else 1.0
        consistency = 1.0 / (1.0 + cv)
        dir_bonus = 1.0 + 0.5 * (dir_ratio - self.min_dir_ratio) / (0.5 - self.min_dir_ratio)
        return min_score * (1.0 + consistency) * dir_bonus


# ── Supervised evaluator ──

class V7SupervisedEvaluator:
    def __init__(self, inputs_2d, mid_close, labels, pip, spread):
        self.inputs_2d = inputs_2d
        self.mid_close = mid_close
        self.labels = labels
        self.pip = pip
        self.spread = spread

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            try:
                net = extract_network(genome, config)
                acc = evaluate_supervised_v7(
                    self.inputs_2d, self.mid_close, self.labels,
                    self.pip, self.spread,
                    net[0], net[2], net[3], net[4], net[5], net[6],
                    net[7], net[8], net[9], net[10])
                genome.fitness = acc * 100.0
            except Exception:
                genome.fitness = 0.0


# ── Fixed topology builder ──

def build_fixed_topology(config, n_inputs, n_hidden, n_outputs):
    """Build N_IN → N_HID → N_OUT fully connected + skip.
    Inputs: -1..-N_INPUTS, Outputs: 0..N_OUTPUTS-1, Hidden: 100..100+N_HIDDEN-1
    """
    gc = config.genome_config
    genome = config.genome_type(0)
    genome.fitness = None

    in_ids = list(range(-n_inputs, 0))     # [-7, -6, ..., -1]
    out_ids = list(range(0, n_outputs))     # [0, 1, 2]
    hid_ids = list(range(100, 100 + n_hidden))  # [100, 101, ..., 104]

    # Output nodes
    for nid in out_ids:
        node = gc.node_gene_type(nid)
        node.bias = np.random.uniform(-1, 1)
        node.response = 1.0
        node.activation = 'tanh'
        node.aggregation = 'sum'
        genome.nodes[nid] = node

    # Hidden nodes
    act_choices = ['tanh', 'sin', 'cos', 'gauss']
    for nid in hid_ids:
        node = gc.node_gene_type(nid)
        node.bias = np.random.uniform(-1, 1)
        node.response = 1.0
        node.activation = act_choices[np.random.randint(4)]
        node.aggregation = 'sum'
        genome.nodes[nid] = node

    # All connections: in→hid, hid→out, in→out (skip)
    all_conns = []
    for src in in_ids:
        for dst in hid_ids:
            all_conns.append((src, dst))
    for src in hid_ids:
        for dst in out_ids:
            all_conns.append((src, dst))
    for src in in_ids:
        for dst in out_ids:
            all_conns.append((src, dst))

    for innov, (src, dst) in enumerate(all_conns):
        conn = gc.connection_gene_type((src, dst), innovation=innov)
        conn.weight = np.random.uniform(-2, 2)
        conn.enabled = True
        genome.connections[(src, dst)] = conn

    return genome


def mutate_v7(genome, config):
    """Mutate weights, biases, activations. Never touch topology."""
    for cg in genome.connections.values():
        if np.random.random() < config.genome_config.weight_mutate_rate:
            cg.weight += np.random.normal(0, config.genome_config.weight_mutate_power)
            cg.weight = max(min(cg.weight, 5.0), -5.0)
    for ng in genome.nodes.values():
        if np.random.random() < config.genome_config.bias_mutate_rate:
            ng.bias += np.random.normal(0, config.genome_config.bias_mutate_power)
            ng.bias = max(min(ng.bias, 5.0), -5.0)
        if np.random.random() < config.genome_config.activation_mutate_rate:
            ng.activation = np.random.choice(config.genome_config.activation_options)


# ── Island evolution ──

def island_evolution(evaluator, config, n_islands, pop_size, n_gens, stall_limit,
                     pair, tag, phase_name, results_dir, seed_genome=None):
    islands = []
    for isl in range(n_islands):
        pop = []
        for i in range(pop_size):
            g = build_fixed_topology(config, N_INPUTS, N_HIDDEN, N_OUTPUTS)
            if seed_genome is not None and i < pop_size // 4:
                g = copy.deepcopy(seed_genome)
                mutate_v7(g, config)
            g.key = isl * pop_size + i
            pop.append(g)
        islands.append(pop)

    global_best = None; global_fitness = -1e30; stall = 0

    for gen in range(n_gens):
        for isl_idx, pop in enumerate(islands):
            evaluator.evaluate([(g.key, g) for g in pop], config)

            pop.sort(key=lambda g: g.fitness, reverse=True)
            best = pop[0]

            if best.fitness > global_fitness:
                global_fitness = best.fitness
                global_best = copy.deepcopy(best)
                stall = 0

            # Selection + mutation
            elite = max(2, pop_size // 10)
            new_pop = pop[:elite]
            while len(new_pop) < pop_size:
                parent = pop[np.random.randint(0, len(pop) // 3)]
                child = copy.deepcopy(parent)
                child.key = len(new_pop) + isl_idx * pop_size
                mutate_v7(child, config)
                new_pop.append(child)
            islands[isl_idx] = new_pop

        stall += 1
        if gen % 10 == 0:
            print(f"  [{pair} {phase_name}] Gen {gen:3d}: best={global_fitness:.4f} stall={stall}", flush=True)

        # Migration every 10 gens
        if gen > 0 and gen % 10 == 0 and len(islands) > 1:
            for i in range(len(islands)):
                j = (i + 1) % len(islands)
                migrant = copy.deepcopy(islands[i][0])
                migrant.key = len(islands[j])
                islands[j][-1] = migrant

        if stall >= stall_limit:
            print(f"  [{pair} {phase_name}] Stalled at gen {gen}, stopping.")
            break

    return global_best, global_fitness


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="IronNet V7 (7-input SHAP) per-pair training")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sine-gens", type=int, default=30)
    parser.add_argument("--pretrain-gens", type=int, default=50)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--pop", type=int, default=150)
    parser.add_argument("--max-hold", type=int, default=17)
    parser.add_argument("--stall-limit", type=int, default=60)
    parser.add_argument("--min-dir-ratio", type=float, default=0.15)
    args = parser.parse_args()

    np.random.seed(args.seed)
    pair = args.pair
    pip = PAIR_PIP[pair]
    spread = PAIR_SPREAD[pair]
    min_swing_pips = int(PAIR_MIN_SWING.get(pair, 20) * 2.0)  # H1 scale

    n_conn = N_INPUTS * N_HIDDEN + N_HIDDEN * N_OUTPUTS + N_INPUTS * N_OUTPUTS
    print(f"{'='*65}")
    print(f"  IronNet V7: {pair} (H1)")
    print(f"  Fixed topology: {N_INPUTS}→{N_HIDDEN}→{N_OUTPUTS} + skip ({n_conn} conn)")
    print(f"  Inputs: bb_width, stoch_d, macd_hist, range_pos_30, aroon_osc, mc_d_a, UPnL")
    print(f"  Activations: tanh, sin, cos, gauss")
    print(f"  Seed: {args.seed} | {args.islands}×{args.pop} pop")
    print(f"  Sine: {args.sine_gens}g → Zigzag: {args.pretrain_gens}g → WF P&L: {args.gens}g")
    print(f"  Min swing: {min_swing_pips}p | Max hold: {args.max_hold} bars | Stall: {args.stall_limit}")
    print(f"{'='*65}")
    tg_send(f"🔩 V7 {pair} H1 s{args.seed}\n{N_INPUTS}→{N_HIDDEN}→{N_OUTPUTS} ({n_conn}c)\n"
            f"PT {args.pretrain_gens}g + WF {args.gens}g")

    # Load data
    path = V7_DATA_DIR / f"{pair}_v7.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found"); return
    df = pd.read_parquet(path, engine="pyarrow")

    # Resample M5 → H1 via .last()
    df = df.set_index("timestamp")
    agg = {"mid_close": "last"}
    for c in IND_COLS:
        agg[c] = "last"
    df = df.resample("1h").agg(agg).dropna(subset=["mid_close"]).reset_index()
    print(f"  Resampled M5 → H1: {len(df):,} bars")

    mid = df["mid_close"].values.astype(np.float64)
    n = len(mid)
    split = int(n * 0.7)

    inputs = np.stack([df[c].values.astype(np.float64) for c in IND_COLS], axis=0)
    inputs = normalize_inputs(inputs)  # Scale bb_width, macd_hist

    inputs_is = inputs[:, :split]
    mid_is = mid[:split]
    inputs_oos = inputs[:, split:]
    mid_oos = mid[split:]
    del df; gc.collect()

    print(f"\nData: {n:,} H1 bars | IS: {split:,} | OOS: {n - split:,}")
    for i, name in enumerate(IND_COLS):
        print(f"  {name:15s} range: [{inputs_is[i].min():.4f}, {inputs_is[i].max():.4f}]")

    config_path = SCRIPT_DIR / "neat_config_7in_3out.ini"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_path))
    for name, fn in [('gauss', lambda x: math.exp(-x*x)),
                     ('sin', math.sin), ('cos', math.cos), ('tanh', math.tanh)]:
        try: config.genome_config.add_activation(name, fn)
        except: pass

    tag = f"iron_v7_H1_{pair}_s{args.seed}"

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Sine wave pretrain
    # ═══════════════════════════════════════════════════════════════
    print(f"\nPhase 1: Sine wave pretrain ({args.sine_gens} gens)...")
    center = float(mid_is.mean())
    amp = 40 * pip
    period = 42
    n_sine = 5000

    # Generate synthetic sine with oscillating indicators
    t = np.arange(n_sine, dtype=np.float64)
    sine_close = center + amp * np.sin(2 * np.pi * t / period)
    noise = np.random.normal(0, amp * 0.05, n_sine)
    sine_close += noise
    phase = 2 * np.pi * t / period

    # Synthetic indicators that track sine phase (no need for full ASI pipeline)
    sine_ind = np.stack([
        np.clip(np.abs(np.cos(phase)) * 0.5 + 0.1, 0, 1),      # bb_width proxy
        (np.sin(phase) + 1) / 2,                                  # stoch_d proxy [0,1]
        np.clip(np.cos(phase) * 0.8, -1, 1),                     # macd_hist proxy
        (np.sin(phase + 0.5) + 1) / 2,                           # range_pos proxy [0,1]
        np.clip(np.sin(phase * 1.1) * 0.9, -1, 1),               # aroon proxy
        np.clip(np.sin(phase) * 0.7, -1, 1),                     # mc_d proxy
    ], axis=0).astype(np.float64)
    sine_ind = np.ascontiguousarray(sine_ind)
    sine_close = np.ascontiguousarray(sine_close)

    sine_labels = generate_zigzag_labels(sine_close, pip, 20, label_window=8)

    sine_eval = V7SupervisedEvaluator(sine_ind, sine_close, sine_labels, pip, spread)
    sine_best, sine_fit = island_evolution(
        sine_eval, config, args.islands, args.pop, args.sine_gens, 30,
        pair, tag, "SINE", RESULTS_DIR)
    print(f"  Sine best fitness: {sine_fit:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Zigzag supervised pretrain on real data
    # ═══════════════════════════════════════════════════════════════
    print(f"\nPhase 2: Zigzag pretrain ({args.pretrain_gens} gens)...")
    zz_labels = generate_zigzag_labels(mid_is, pip, min_swing_pips, label_window=3)
    buy_pct = (zz_labels == 1).mean() * 100
    sell_pct = (zz_labels == 2).mean() * 100
    print(f"  Labels: BUY={buy_pct:.1f}% SELL={sell_pct:.1f}%")

    zz_eval = V7SupervisedEvaluator(inputs_is, mid_is, zz_labels, pip, spread)
    zz_best, zz_fit = island_evolution(
        zz_eval, config, args.islands, args.pop, args.pretrain_gens, 40,
        pair, tag, "ZZ", RESULTS_DIR, seed_genome=sine_best)
    print(f"  Zigzag best fitness: {zz_fit:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: WF P&L evolution
    # ═══════════════════════════════════════════════════════════════
    print(f"\nPhase 3: WF P&L evolution ({args.gens} gens, {N_CHUNKS} chunks)...")
    wf_eval = V7WFEvaluator(inputs_is, mid_is, pip, spread,
                             max_hold=args.max_hold, n_chunks=N_CHUNKS,
                             min_dir_ratio=args.min_dir_ratio)
    ev_best, ev_fit = island_evolution(
        wf_eval, config, args.islands, args.pop, args.gens, args.stall_limit,
        pair, tag, "EV", RESULTS_DIR, seed_genome=zz_best)
    print(f"  WF best fitness: {ev_fit:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: OOS evaluation
    # ═══════════════════════════════════════════════════════════════
    print(f"\nPhase 4: OOS evaluation...")
    net = extract_network(ev_best, config)
    nt, pnl, mae, nl, ns = evaluate_chunk_v7(
        inputs_oos, mid_oos, pip, spread, args.max_hold,
        net[0], net[2], net[3], net[4], net[5], net[6],
        net[7], net[8], net[9], net[10],
        0, inputs_oos.shape[1])

    oos_bars = inputs_oos.shape[1]
    oos_days = oos_bars / 24.0
    ppd = pnl / oos_days if oos_days > 0 else 0
    dir_ratio = min(nl, ns) / nt if nt > 0 else 0

    print(f"  RESULTS: {pair} (IronNet V7)")
    print(f"  OOS:   {nt}T   {pnl:+.1f}p L={nl} S={ns} MAE={mae:.1f}p dir={dir_ratio:.2f} ({ppd:.1f}p/day)")
    print(f"  Fitness: {ev_fit:.4f} | Size: ({len(ev_best.nodes)}, {len(ev_best.connections)})")

    tg_send(f"✅ V7 {pair} H1 s{args.seed}\n"
            f"OOS: {nt}T {pnl:+.1f}p ({ppd:.1f}p/day)\n"
            f"L={nl} S={ns} MAE={mae:.1f}p")

    # Save
    result = {
        "pair": pair, "seed": args.seed, "tf": "H1",
        "n_inputs": N_INPUTS, "n_hidden": N_HIDDEN, "n_outputs": N_OUTPUTS,
        "oos_trades": nt, "oos_pnl": round(pnl, 1), "oos_ppd": round(ppd, 1),
        "oos_long": nl, "oos_short": ns, "oos_mae": round(mae, 1),
        "fitness": round(ev_fit, 4), "dir_ratio": round(dir_ratio, 2),
        "min_dir_ratio": args.min_dir_ratio,
    }

    pkl_path = RESULTS_DIR / f"{tag}_best.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"genome": ev_best, "config": config, "result": result}, f)
    print(f"  Saved: {pkl_path.name}")

    json_path = RESULTS_DIR / f"{tag}_result.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {json_path.name}")


if __name__ == "__main__":
    main()
