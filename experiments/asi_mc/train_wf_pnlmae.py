#!/usr/bin/env python3
"""PnL/MAE training WITH Walk-Forward baked into fitness function."""
import sys, os, time, pickle, copy, json
import numpy as np
from pathlib import Path
from numba import njit

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "research" / "experiments" / "asi_mc"))

import neat
from lib.fast_eval import extract_network, _activate

DATA_DIR = PROJECT / "data" / "asi_mc_indicators"
RESULTS_DIR = PROJECT / "research" / "experiments" / "asi_mc" / "results"

PAIR_PIP = {"EUR_JPY":0.01,"USD_JPY":0.01,"GBP_JPY":0.01,"AUD_JPY":0.01,"CAD_JPY":0.01,"CHF_JPY":0.01,"NZD_JPY":0.01,"EUR_USD":0.0001,"GBP_USD":0.0001,"AUD_USD":0.0001,"NZD_USD":0.0001,"EUR_GBP":0.0001}
PAIR_SPREAD = {"EUR_JPY":2.3,"USD_JPY":1.7,"GBP_JPY":3.3,"AUD_JPY":2.1,"CAD_JPY":2.3,"CHF_JPY":3.5,"NZD_JPY":2.7,"EUR_USD":1.6,"GBP_USD":1.9,"AUD_USD":1.3,"NZD_USD":1.5,"EUR_GBP":1.4}

GENS = 200; SEED = 137; FREQ_EXP = 0.5
N_CHUNKS = 3  # Walk-forward chunks within IS data

def tg_send(text):
    try:
        import requests
        requests.post("https://api.telegram.org/bot{os.environ.get("TELEGRAM_BOT_TOKEN","")}/sendMessage",
                      json={"chat_id":os.environ.get("TELEGRAM_CHAT_ID",""),"text":text,"parse_mode":"HTML"},timeout=10)
    except: pass

@njit
def eval_chunk(mc_d, mc_dd, mid_close, pip, spread_pips, max_hold,
               n_inputs, n_eval, total_values, node_bias, node_response, node_act,
               conn_from, conn_to, conn_weight, output_indices,
               chunk_start, chunk_end):
    """Evaluate on a specific chunk of data. Returns (n_trades, total_pnl, avg_mae)."""
    values = np.zeros(total_values)
    start_bar = max(chunk_start + 10, 10)
    end_bar = min(chunk_end, len(mid_close) - 1)
    max_t = end_bar - start_bar + 1
    pnls = np.zeros(max_t); maes = np.zeros(max_t)
    nt = 0; pos = 0; ep = 0.0; eb = 0; rmae = 0.0

    for i in range(start_bar, end_bar):
        if pos != 0:
            pp = (mid_close[i] - ep) / pip * pos - spread_pips
            if pp < rmae: rmae = pp
        else:
            pp = 0.0
        values[0] = mc_d[i]; values[1] = mc_dd[i]; values[2] = np.tanh(pp / 20.0)
        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)
        ob = values[output_indices[0]]; os_ = values[output_indices[1]]; of = values[output_indices[2]]

        if pos != 0 and (i - eb) >= max_hold:
            pnl = (mid_close[i] - ep) / pip * pos - spread_pips
            if nt < max_t: pnls[nt] = pnl; maes[nt] = rmae; nt += 1
            pos = 0
        if pos == 0:
            if ob > os_ and ob > of: pos = 1; ep = mid_close[i]; eb = i; rmae = 0.0
            elif os_ > ob and os_ > of: pos = -1; ep = mid_close[i]; eb = i; rmae = 0.0
        else:
            cl = False; np_ = 0
            if of > ob and of > os_: cl = True
            elif pos == 1 and os_ > ob and os_ > of: cl = True; np_ = -1
            elif pos == -1 and ob > os_ and ob > of: cl = True; np_ = 1
            if cl:
                pnl = (mid_close[i] - ep) / pip * pos - spread_pips
                if nt < max_t: pnls[nt] = pnl; maes[nt] = rmae; nt += 1
                pos = np_; ep = mid_close[i] if np_ != 0 else 0.0; eb = i; rmae = 0.0

    if pos != 0 and end_bar > start_bar:
        pnl = (mid_close[end_bar-1] - ep) / pip * pos - spread_pips
        if nt < max_t: pnls[nt] = pnl; maes[nt] = rmae; nt += 1

    if nt < 1:
        return 0, 0.0, 0.0
    tp = 0.0; tm = 0.0
    for j in range(nt): tp += pnls[j]; tm += abs(maes[j])
    return nt, tp, tm / nt


class WFPnlMaeEvaluator:
    """Walk-Forward PnL/MAE evaluator — genome must be profitable on ALL chunks."""

    def __init__(self, pair_data, max_hold=200, n_chunks=3, freq_exp=0.5):
        self.pair_data = pair_data  # {pair: (mc_d_is, mc_dd_is, mid_is, mc_d_oos, mc_dd_oos, mid_oos)}
        self.max_hold = max_hold
        self.n_chunks = n_chunks
        self.freq_exp = freq_exp

    def evaluate(self, genomes, config):
        for gid, genome in genomes:
            genome.fitness = self._fitness(genome, config)

    def _fitness(self, genome, config):
        try:
            net = extract_network(genome, config)
        except:
            return -10.0

        chunk_scores = []

        for chunk_idx in range(self.n_chunks):
            chunk_trades = 0; chunk_fitness = 0.0; chunk_pairs = 0
            chunk_pnl = 0.0

            for pair, (mc_d, mc_dd, mid, *_) in self.pair_data.items():
                n = len(mid)
                # Split IS data into N chunks
                c_start = int(n * chunk_idx / self.n_chunks)
                c_end = int(n * (chunk_idx + 1) / self.n_chunks)

                pip = PAIR_PIP.get(pair, 0.01)
                spread = PAIR_SPREAD.get(pair, 2.0)

                nt, pnl, mae = eval_chunk(
                    mc_d, mc_dd, mid, pip, spread, self.max_hold,
                    net[0], net[2], net[3], net[4], net[5], net[6],
                    net[7], net[8], net[9], net[10],
                    c_start, c_end)

                chunk_trades += nt
                chunk_pnl += pnl
                if nt >= 3:
                    mp = pnl / nt
                    if mae > 0:
                        chunk_fitness += mp / mae
                    else:
                        chunk_fitness += mp
                    chunk_pairs += 1

            # Hard gates per chunk
            if chunk_trades < 200:  # ~1 trade/day/pair minimum per chunk
                return -10.0
            if chunk_pairs < 4:  # must trade 4+ pairs per chunk
                return -10.0
            if chunk_pnl <= 0:  # must be profitable in EVERY chunk
                return -10.0

            # Chunk score = avg PnL/MAE × frequency bonus
            avg_score = chunk_fitness / max(chunk_pairs, 1)
            freq_bonus = chunk_trades ** self.freq_exp
            chunk_scores.append(avg_score * freq_bonus)

        # Final fitness = min(chunk_scores) × consistency bonus
        min_score = min(chunk_scores)
        mean_score = sum(chunk_scores) / len(chunk_scores)
        std_score = (sum((s - mean_score)**2 for s in chunk_scores) / len(chunk_scores)) ** 0.5

        # Consistency: bonus for similar performance across chunks (low std/mean ratio)
        if mean_score > 0:
            cv = std_score / mean_score  # coefficient of variation
            consistency = 1.0 / (1.0 + cv)  # 1.0 if perfectly consistent, lower if variable
        else:
            consistency = 0.5

        return min_score * (1.0 + consistency)

    def eval_oos(self, genome, config):
        """Evaluate on held-out OOS data (70-100%). NOT used during training."""
        try:
            net = extract_network(genome, config)
        except:
            return {}
        results = {}
        for pair, data in self.pair_data.items():
            mc_d_oos, mc_dd_oos, mid_oos = data[3], data[4], data[5]
            pip = PAIR_PIP.get(pair, 0.01)
            spread = PAIR_SPREAD.get(pair, 2.0)
            nt, pnl, mae = eval_chunk(
                mc_d_oos, mc_dd_oos, mid_oos, pip, spread, self.max_hold,
                net[0], net[2], net[3], net[4], net[5], net[6],
                net[7], net[8], net[9], net[10],
                0, len(mid_oos))
            results[pair] = {
                "n_trades": int(nt), "total_pnl": round(float(pnl), 1),
                "avg_mae": round(float(mae), 2),
                "avg_pnl": round(float(pnl/max(nt,1)), 2),
                "win_rate": 0,  # not tracked in chunk eval for speed
                "n_long": 0, "n_short": 0,
            }
        return results


if __name__ == "__main__":
    from train_from_indicators import run_island_neat
    import pandas as pd

    np.random.seed(SEED)
    print("=" * 60)
    print(f"  WF-IN-FITNESS PnL/MAE Training")
    print(f"  {N_CHUNKS} WF chunks, exp={FREQ_EXP}, {GENS} gens, seed={SEED}")
    print(f"  Genome MUST be profitable in ALL {N_CHUNKS} time periods")
    print("=" * 60)
    tg_send(f"🧬 WF-in-fitness training launched on S2\n{N_CHUNKS} chunks, exp={FREQ_EXP}, {GENS}g\nGenome must pass WF during evolution — no post-hoc needed")

    # Load seed genome
    with open(str(RESULTS_DIR / "pnl_mae_best.pkl"), "rb") as f:
        d = pickle.load(f)
    seed_genome = d["genome"]
    print(f"Seed: fitness={seed_genome.fitness}, size={seed_genome.size()}")

    # Load ALL pairs
    all_data = {}
    for pair in PAIR_PIP:
        path = DATA_DIR / f"{pair}_asi_mc.parquet"
        if not path.exists(): continue
        df = pd.read_parquet(path)
        mc_d = df["mc_d_a"].values.astype(np.float64)
        mc_dd = df["mc_dd_a"].values.astype(np.float64)
        mid = df["mid_close"].values.astype(np.float64)
        n = len(mid); sp = int(n * 0.7)
        all_data[pair] = (mc_d[:sp], mc_dd[:sp], mid[:sp], mc_d[sp:], mc_dd[sp:], mid[sp:])
        print(f"  {pair}: {n:,} bars (IS={sp:,}, OOS={n-sp:,})")

    config_path = PROJECT / "research" / "experiments" / "asi_mc" / "neat_config_3out.ini"

    # WF evaluator — trains on ALL pairs, evaluates on 3 IS chunks
    evaluator = WFPnlMaeEvaluator(all_data, max_hold=200,
                                   n_chunks=N_CHUNKS, freq_exp=FREQ_EXP)

    t0 = time.time()
    winner, config = run_island_neat(config_path, evaluator, seed_genome,
                                      n_islands=4, pop_per_island=150,
                                      generations=GENS,
                                      save_dir=str(RESULTS_DIR / "wf_pnlmae_ckpts"))
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.0f}s, fitness={winner.fitness:.4f}")

    # OOS eval (completely unseen data)
    oos = evaluator.eval_oos(winner, config)
    tp = sum(r["total_pnl"] for r in oos.values())
    tt = sum(r["n_trades"] for r in oos.values())
    pp = sum(1 for r in oos.values() if r["total_pnl"] > 0)
    avg_mae = np.mean([r["avg_mae"] for r in oos.values() if r["avg_mae"] > 0])
    avg_pnl = np.mean([r["avg_pnl"] for r in oos.values()])
    days = len(list(all_data.values())[0][5]) / 288

    print(f"\nOOS: {tt}T {tp:+.0f}p {pp}/12+ MAE={avg_mae:.2f}p")
    print(f"  {tt/days:.1f} trades/day, {tp/days:+.1f} pips/day")
    for pair in sorted(oos):
        r = oos[pair]
        print(f"  {pair:<10} {r['n_trades']:>5}T {r['total_pnl']:>+8.1f}p MAE={r['avg_mae']:>6.2f}p")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "wf_pnlmae_best.pkl", "wb") as f:
        pickle.dump({"genome": winner, "config": config}, f)
    with open(RESULTS_DIR / "wf_pnlmae_result.json", "w") as f:
        json.dump({"fitness_mode": "wf_pnl_mae", "n_chunks": N_CHUNKS,
                    "freq_exp": FREQ_EXP, "gens": GENS, "seed": SEED,
                    "fitness": round(float(winner.fitness), 4),
                    "network_size": list(winner.size()),
                    "oos_total_pnl": round(tp, 1), "oos_total_trades": tt,
                    "oos_pairs_profitable": pp,
                    "avg_mae": round(float(avg_mae), 2),
                    "avg_pnl": round(float(avg_pnl), 2),
                    "trades_per_day": round(tt/days, 1),
                    "pips_per_day": round(tp/days, 1),
                    "elapsed_s": round(elapsed, 1), "oos": oos}, f, indent=2)

    tg_send(f"🏁 WF-in-fitness DONE ({elapsed:.0f}s)\n\n"
            f"Fitness: {winner.fitness:.4f}\n"
            f"OOS: {tt}T {tp:+.0f}p {pp}/12+\n"
            f"{tt/days:.1f} trades/day, {tp/days:+.1f} pips/day\n"
            f"MAE: {avg_mae:.2f}p\n\n"
            f"This genome passed WF DURING training.\n"
            f"Compare vs S3-S5 (no WF) to validate approach.")
