#!/usr/bin/env python3
"""
MTF Slope + Trade State Experiment — 8→11→3 on S5 cadence
============================================================
8 inputs:
  [0] slope_s5    — LinReg slope last 10 S5 closes, arctan
  [1] slope_m1    — LinReg slope last 10 M1 closes (sampled every 12 S5 bars)
  [2] slope_m5    — LinReg slope last 10 M5 closes (sampled every 60 S5 bars)
  [3] slope_h1    — LinReg slope last 10 H1 closes (sampled every 720 S5 bars)
  [4] upnl        — tanh((mid - entry ± spread*pip) / pip / 20), 0 flat
  [5] mae         — tanh(worst_adverse / 20), 0 flat
  [6] mfe         — tanh(best_favorable / 20), 0 flat
  [7] delta_5m    — arctan((mid - close_60bars_ago) / pip / 20)

Architecture: 8→11→3 fixed (11 hidden, BUY/SELL/FLATTEN)
Cadence: S5 (every 5 seconds)
Spread: real from bid/ask data, charged at entry

Usage:
  python3 train_mtf_slope.py --pair EUR_JPY --seed 42
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

S5_DIR = PROJECT_ROOT / "data" / "s5_ohlc"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PAIR_PIP = {
    "EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
    "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
}

N_INPUTS = 8
N_HIDDEN = 5
N_OUTPUTS = 3
N_CHUNKS = 3


# ── Activations ──
def gauss_activation(x): return math.exp(-x*x)
def sin_activation(x): return math.sin(x)
def cos_activation(x): return math.cos(x)
def tanh_activation(x): return math.tanh(x)
def sech_activation(x): return 1.0/math.cosh(max(min(x,50),-50))
def dog_activation(x): return math.exp(-x*x/2)-0.5*math.exp(-x*x/8)
def gabor_activation(x): return math.exp(-2*x*x)*math.cos(2*math.pi*x)
def sinc_activation(x): return math.sin(math.pi*x)/(math.pi*x) if abs(x)>1e-7 else 1.0
def morlet_activation(x): return math.sin(x)*math.exp(-x*x/2)

ACT_NAMES = ['tanh','sin','cos','gauss','sech','dog','gabor','sinc','morlet']
ACT_FUNCS = [tanh_activation,sin_activation,cos_activation,gauss_activation,
             sech_activation,dog_activation,gabor_activation,sinc_activation,morlet_activation]


# ── Compute indicators from S5 BA data ──

@njit(cache=True)
def compute_linreg_slope_multi(mid, n, periods_s5):
    """Compute arctan-normalized linreg slopes at multiple timeframe sample rates.
    periods_s5: array of S5 bar spacings [1, 12, 60, 720] for S5/M1/M5/H1
    Returns (n_tf, n) array of slopes.
    """
    n_tf = len(periods_s5)
    slopes = np.zeros((n_tf, n))
    hp = np.pi / 2.0
    n_points = 10  # 10 samples per slope

    for tf in range(n_tf):
        spacing = periods_s5[tf]
        lookback = n_points * spacing
        for i in range(lookback, n):
            # Sample 10 points at this TF spacing
            vals = np.zeros(n_points)
            for k in range(n_points):
                vals[n_points - 1 - k] = mid[i - k * spacing]
            # LinReg slope
            x_mean = (n_points - 1) / 2.0
            y_mean = 0.0
            for k in range(n_points): y_mean += vals[k]
            y_mean /= n_points
            num = 0.0; den = 0.0
            for k in range(n_points):
                xd = k - x_mean
                num += xd * (vals[k] - y_mean)
                den += xd * xd
            if den > 0:
                slope = num / den
                rng = vals.max() - vals.min()
                if rng > 0:
                    slopes[tf, i] = np.arctan(slope / rng * 3.0) / hp
    return slopes


@njit(cache=True)
def compute_delta_5m(mid, n, pip):
    """(mid[i] - mid[i-60]) / pip / 20, arctan-normalized."""
    out = np.zeros(n)
    hp = np.pi / 2.0
    for i in range(60, n):
        raw = (mid[i] - mid[i - 60]) / pip / 20.0
        out[i] = np.arctan(raw) / hp
    return out


@njit(cache=True)
def evaluate_s5_chunk(slopes, delta_5m, mid, pip, spread_arr, max_hold,
                      n_inputs, n_eval, total_values,
                      node_bias, node_response, node_act,
                      conn_from, conn_to, conn_weight, output_indices,
                      c_start, c_end):
    """Evaluate genome on S5 data with trade state inputs."""
    values = np.zeros(total_values)
    position = 0
    entry_price = 0.0
    entry_bar = 0
    running_mfe = 0.0
    running_mae = 0.0
    n_trades = 0
    n_long = 0; n_short = 0
    total_pnl = 0.0
    total_mae_sum = 0.0
    max_t = c_end - c_start
    trade_pnls = np.zeros(max_t)

    warmup = max(800, max_hold * 2)  # ensure enough bars for slope computation
    for i in range(c_start + warmup, c_end - 1):
        mid_i = mid[i]
        spread_pips = spread_arr[i]

        # 4 slope inputs
        for tf in range(4):
            values[tf] = slopes[tf, i]

        # Trade state inputs
        if position != 0 and entry_price > 0:
            if position == 1:
                pnl_pips = (mid_i - entry_price) / pip - spread_pips
                ae = (entry_price - mid_i) / pip + spread_pips
            else:
                pnl_pips = (entry_price - mid_i) / pip - spread_pips
                ae = (mid_i - entry_price) / pip + spread_pips
            fe = max(0.0, -ae + spread_pips)  # favorable excursion
            if ae > running_mae: running_mae = ae
            if fe > running_mfe: running_mfe = fe
            values[4] = np.tanh(pnl_pips / 20.0)  # upnl
            values[5] = np.tanh(running_mae / 20.0)  # mae
            values[6] = np.tanh(running_mfe / 20.0)  # mfe
        else:
            values[4] = 0.0
            values[5] = 0.0
            values[6] = 0.0

        # delta_5m
        values[7] = delta_5m[i]

        _activate(values, n_eval, total_values, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)

        out_buy = values[output_indices[0]]
        out_sell = values[output_indices[1]]
        out_flat = values[output_indices[2]]

        # Max hold (in S5 bars): 7h = 5040 S5 bars
        if position != 0 and (i - entry_bar) >= max_hold:
            out_flat = max(out_buy, out_sell) + 0.01

        if position == 0:
            if out_buy > out_sell and out_buy > out_flat:
                position = 1
                entry_price = mid_i + spread_pips * pip  # spread at entry
                entry_bar = i; running_mfe = 0.0; running_mae = spread_pips
                n_long += 1
            elif out_sell > out_buy and out_sell > out_flat:
                position = -1
                entry_price = mid_i - spread_pips * pip
                entry_bar = i; running_mfe = 0.0; running_mae = spread_pips
                n_short += 1
        else:
            close = False; new_pos = 0
            if out_flat > out_buy and out_flat > out_sell:
                close = True
            elif position == 1 and out_sell > out_buy and out_sell > out_flat:
                close = True; new_pos = -1
            elif position == -1 and out_buy > out_sell and out_buy > out_flat:
                close = True; new_pos = 1
            if close:
                if position == 1:
                    pnl = (mid_i - entry_price) / pip
                else:
                    pnl = (entry_price - mid_i) / pip
                trade_pnls[n_trades] = pnl
                total_pnl += pnl
                total_mae_sum += running_mae
                n_trades += 1
                position = new_pos
                if new_pos != 0:
                    entry_price = mid_i + (spread_pips * pip * new_pos)
                    entry_bar = i; running_mfe = 0.0; running_mae = spread_pips
                    if new_pos > 0: n_long += 1
                    else: n_short += 1
                else:
                    entry_price = 0.0

    if n_trades == 0:
        return 0, 0.0, 0.0, 0, 0
    return n_trades, total_pnl, total_mae_sum / n_trades, n_long, n_short


# ── WF Evaluator ──

class MTFSlopeWFEvaluator:
    def __init__(self, slopes, delta_5m, mid, pip, spread_arr,
                 max_hold=5040, n_chunks=3, min_dir_ratio=0.15):
        self.slopes = slopes
        self.delta_5m = delta_5m
        self.mid = mid
        self.pip = pip
        self.spread_arr = spread_arr
        self.max_hold = max_hold
        self.n_chunks = n_chunks
        self.min_dir_ratio = min_dir_ratio
        self.n_bars = len(mid)

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
            cs = int(self.n_bars * ci / self.n_chunks)
            ce = int(self.n_bars * (ci + 1) / self.n_chunks)

            nt, pnl, mae, nl, ns = evaluate_s5_chunk(
                self.slopes, self.delta_5m, self.mid, self.pip, self.spread_arr, self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10],
                cs, ce)

            total_long += nl; total_short += ns; total_trades += nt
            min_t = max(50, int(self.n_bars / self.n_chunks / 17280 * 2))
            if nt < min_t: return -10.0
            if pnl <= 0: return -10.0
            mean_pnl = pnl / nt
            score = mean_pnl / mae if mae > 0 else mean_pnl
            chunk_scores.append(score * (nt ** 0.5))

        if total_trades < 50: return -10.0
        dir_ratio = min(total_long, total_short) / total_trades
        if dir_ratio < self.min_dir_ratio: return -10.0

        min_score = min(chunk_scores)
        mean_score = sum(chunk_scores) / len(chunk_scores)
        cv = (sum((s - mean_score)**2 for s in chunk_scores) / len(chunk_scores))**0.5 / mean_score if mean_score > 0 else 1.0
        consistency = 1.0 / (1.0 + cv)
        dir_bonus = 1.0 + 0.5 * (dir_ratio - self.min_dir_ratio) / (0.5 - self.min_dir_ratio)
        return min_score * (1.0 + consistency) * dir_bonus


# ── Genome builder ──

def build_genome(config):
    genome = config.genome_type(0)
    genome.fitness = None
    gc = config.genome_config
    in_ids = list(range(-N_INPUTS, 0))
    out_ids = [0, 1, 2]
    hid_ids = list(range(100, 100 + N_HIDDEN))

    for nid in out_ids:
        node = gc.node_gene_type(nid)
        node.bias = np.random.uniform(-1, 1); node.response = 1.0
        node.activation = 'tanh'; node.aggregation = 'sum'
        genome.nodes[nid] = node

    for nid in hid_ids:
        node = gc.node_gene_type(nid)
        node.bias = np.random.uniform(-1, 1); node.response = 1.0
        node.activation = ACT_NAMES[np.random.randint(len(ACT_NAMES))]
        node.aggregation = 'sum'
        genome.nodes[nid] = node

    all_conns = []
    for s in in_ids:
        for d in hid_ids: all_conns.append((s, d))
    for s in hid_ids:
        for d in out_ids: all_conns.append((s, d))
    for s in in_ids:
        for d in out_ids: all_conns.append((s, d))

    for innov, (s, d) in enumerate(all_conns):
        conn = gc.connection_gene_type((s, d), innovation=innov)
        conn.weight = np.random.uniform(-2, 2); conn.enabled = True
        genome.connections[(s, d)] = conn
    return genome


def mutate(genome, config):
    gc = config.genome_config
    for cg in genome.connections.values():
        if np.random.random() < gc.weight_mutate_rate:
            cg.weight += np.random.normal(0, gc.weight_mutate_power)
            cg.weight = max(min(cg.weight, 5.0), -5.0)
    for ng in genome.nodes.values():
        if np.random.random() < gc.bias_mutate_rate:
            ng.bias += np.random.normal(0, gc.bias_mutate_power)
            ng.bias = max(min(ng.bias, 5.0), -5.0)
        if np.random.random() < gc.activation_mutate_rate:
            ng.activation = ACT_NAMES[np.random.randint(len(ACT_NAMES))]


# ── Island evolution ──

def evolve(evaluator, config, n_islands, pop_size, n_gens, stall_limit, label):
    islands = []
    for isl in range(n_islands):
        pop = {}
        for j in range(pop_size):
            g = build_genome(config); g.key = j; pop[j] = g
        islands.append({"pop": pop, "best": None})

    global_best = None; global_fit = -999; stall = 0

    for gen in range(n_gens):
        for i, island in enumerate(islands):
            pop = island["pop"]
            evaluator.evaluate(list(pop.items()), config)
            best = max(pop.values(), key=lambda g: g.fitness if g.fitness else -999)
            island["best"] = best
            if best.fitness and best.fitness > global_fit:
                global_fit = best.fitness; global_best = copy.deepcopy(best); stall = 0

            sorted_g = sorted(pop.values(), key=lambda g: g.fitness if g.fitness else -999, reverse=True)
            new_pop = {}
            for j in range(min(3, len(sorted_g))):
                e = copy.deepcopy(sorted_g[j]); e.key = j; e.fitness = None; new_pop[j] = e
            for j in range(3, pop_size):
                cands = np.random.choice(len(sorted_g), size=min(3, len(sorted_g)), replace=False)
                p = copy.deepcopy(sorted_g[min(cands)]); p.key = j; p.fitness = None
                mutate(p, config); new_pop[j] = p
            island["pop"] = new_pop

        stall += 1
        if gen > 0 and gen % 10 == 0 and len(islands) > 1:
            for i in range(len(islands)):
                j = (i + 1) % len(islands)
                if islands[i]["best"]:
                    wk = min(islands[j]["pop"], key=lambda k: islands[j]["pop"][k].fitness if islands[j]["pop"][k].fitness else 999)
                    m = copy.deepcopy(islands[i]["best"]); m.key = wk; m.fitness = None
                    islands[j]["pop"][wk] = m

        if gen % 10 == 0:
            print(f"  [{label}] Gen {gen:3d}: best={global_fit:.4f} stall={stall}", flush=True)
        if stall >= stall_limit:
            print(f"  [{label}] Stalled at gen {gen}"); break

    return global_best, global_fit


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--pop", type=int, default=150)
    parser.add_argument("--stall-limit", type=int, default=60)
    parser.add_argument("--max-hold", type=int, default=5040, help="Max hold in S5 bars (5040=7h)")
    parser.add_argument("--cadence", type=int, default=12, help="Subsample rate (1=S5, 12=M1, 60=M5)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    pair = args.pair
    pip = PAIR_PIP[pair]

    cadence = args.cadence
    cadence_name = {1: "S5", 12: "M1", 60: "M5"}.get(cadence, f"{cadence}xS5")
    max_hold_bars = args.max_hold // cadence  # convert S5 max_hold to cadence bars

    n_conn = N_INPUTS * N_HIDDEN + N_HIDDEN * N_OUTPUTS + N_INPUTS * N_OUTPUTS
    print(f"{'='*65}")
    print(f"  MTF Slope: {pair} ({cadence_name} cadence, subsample={cadence})")
    print(f"  Topology: {N_INPUTS}→{N_HIDDEN}→{N_OUTPUTS} + skip ({n_conn} conn)")
    print(f"  Inputs: slope_s5, slope_m1, slope_m5, slope_h1, upnl, mae, mfe, delta_5m")
    print(f"  Max hold: {max_hold_bars} bars ({max_hold_bars*cadence/720:.1f}h)")
    print(f"{'='*65}")

    # Load S5 BA data
    path = S5_DIR / f"{pair}_S5_BA.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found"); return
    df = pd.read_parquet(path)
    bid_c = df["bid_c"].values.astype(np.float64)
    ask_c = df["ask_c"].values.astype(np.float64)
    mid = (bid_c + ask_c) / 2.0
    spread_arr = (ask_c - bid_c) / pip  # spread in pips per bar
    n = len(mid)
    split = int(n * 0.7)
    print(f"  Data: {n:,} S5 bars | IS: {split:,} | OOS: {n-split:,}")
    print(f"  Spread: mean={spread_arr[1000:].mean():.2f}p median={np.median(spread_arr[1000:]):.2f}p")

    # Compute indicators
    print("  Computing MTF slopes...", flush=True)
    tf_spacings = np.array([1, 12, 60, 720], dtype=np.int64)  # S5, M1, M5, H1
    slopes = compute_linreg_slope_multi(mid, n, tf_spacings)
    delta_5m = compute_delta_5m(mid, n, pip)

    # Subsample to cadence (e.g., M1 = every 12th bar)
    if cadence > 1:
        idx = np.arange(0, n, cadence)
        slopes = slopes[:, idx]
        mid = mid[idx]
        spread_arr = spread_arr[idx]
        delta_5m = delta_5m[idx]
        n = len(mid)
        split = int(n * 0.7)
        print(f"  Subsampled S5→{cadence_name}: {n:,} bars | IS: {split:,} | OOS: {n-split:,}")

    slopes_is = slopes[:, :split]
    mid_is = mid[:split]
    spread_is = spread_arr[:split]
    delta_5m_is = delta_5m[:split]

    for i, tf_name in enumerate(["S5", "M1", "M5", "H1"]):
        print(f"  slope_{tf_name:2s} range: [{slopes_is[i, 7200//cadence:].min():.4f}, {slopes_is[i, 7200//cadence:].max():.4f}]")
    print(f"  delta_5m range: [{delta_5m_is[60//cadence:].min():.4f}, {delta_5m_is[60//cadence:].max():.4f}]")

    # NEAT config
    config_path = SCRIPT_DIR / "neat_config_8in_3out.ini"
    if not config_path.exists():
        # Create from template
        template = PROJECT_ROOT / "research/experiments/asi_mc/neat_config_4in_3out.ini"
        text = template.read_text()
        text = text.replace("num_inputs              = 4", "num_inputs              = 8")
        text = text.replace("activation_options      = tanh sin cos",
                           "activation_options      = tanh sin cos gauss sech dog gabor sinc morlet")
        config_path.write_text(text)
        print(f"  Created {config_path.name}")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, str(config_path))
    for name, fn in zip(ACT_NAMES, ACT_FUNCS):
        try: config.genome_config.add_activation(name, fn)
        except: pass

    # Train — direct WF P&L (no sine/zigzag pretrain — S5 data is rich enough)
    print(f"\n  WF P&L evolution ({args.gens} gens, {N_CHUNKS} chunks)...")
    wf_eval = MTFSlopeWFEvaluator(slopes_is, delta_5m_is, mid_is, pip, spread_is,
                                   max_hold=max_hold_bars)
    t0 = time.time()
    best, fitness = evolve(wf_eval, config, args.islands, args.pop, args.gens,
                           args.stall_limit, f"{pair} EV")
    elapsed = time.time() - t0
    print(f"  Done: fitness={fitness:.4f} ({elapsed:.0f}s)")

    # OOS eval
    slopes_oos = slopes[:, split:]
    mid_oos = mid[split:]
    spread_oos = spread_arr[split:]
    delta_5m_oos = delta_5m[split:]

    net = extract_network(best, config)
    nt, pnl, mae, nl, ns = evaluate_s5_chunk(
        slopes_oos, delta_5m_oos, mid_oos, pip, spread_oos, max_hold_bars,
        net[0], net[2], net[3], net[4], net[5], net[6],
        net[7], net[8], net[9], net[10],
        0, len(mid_oos))

    oos_days = len(mid_oos) / 17280.0
    ppd = pnl / oos_days if oos_days > 0 else 0
    dr = min(nl, ns) / nt if nt > 0 else 0

    print(f"\n  RESULTS: {pair} (MTF Slope S5)")
    print(f"  OOS: {nt}T {pnl:+.1f}p L={nl} S={ns} MAE={mae:.1f}p dir={dr:.2f} ({ppd:.1f}p/day)")

    # Save
    tag = f"mtf_slope_{pair}_s{args.seed}"
    result = {"pair": pair, "seed": args.seed, "oos_trades": nt,
              "oos_pnl": round(pnl, 1), "oos_ppd": round(ppd, 1),
              "oos_long": nl, "oos_short": ns, "oos_mae": round(mae, 1),
              "fitness": round(fitness, 4), "dir_ratio": round(dr, 2)}

    pkl_path = RESULTS_DIR / f"{tag}_best.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"genome": best, "config": config, "result": result}, f)
    json_path = RESULTS_DIR / f"{tag}_result.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {pkl_path.name}")


if __name__ == "__main__":
    main()
