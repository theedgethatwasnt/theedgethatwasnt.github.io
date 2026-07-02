#!/usr/bin/env python3
"""Batch MC sign-shuffle validation for all per-pair IronNet V3 genomes.

Uses same evaluation logic as training (collect_pnls_jit from mc_validate_ironnet.py).
Saves results to results/ironnet/mc_validation_perpair.json (crash-safe, resumes).
"""
import sys, pickle, math, json
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.fast_eval import extract_network, _activate
import neat

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

ALL_PAIRS = [
    "EUR_GBP", "CAD_JPY", "AUD_JPY", "CHF_JPY", "EUR_JPY",
    "USD_JPY", "AUD_USD", "EUR_USD", "GBP_JPY", "GBP_USD",
    "NZD_JPY", "NZD_USD",
]


def gauss_activation(x): return math.exp(-x * x)
def sin_activation(x): return math.sin(x)
def cos_activation(x): return math.cos(x)
def tanh_activation(x): return math.tanh(x)


@njit(cache=True)
def collect_pnls_jit(inputs_2d, mid_close, pip, spread_pips, max_hold,
                     n_inputs, n_eval, total_values,
                     node_bias, node_response, node_act,
                     conn_from, conn_to, conn_weight, output_indices):
    values = np.zeros(total_values)
    n_ind = inputs_2d.shape[0]
    n = inputs_2d.shape[1]
    start_bar = 10
    end_bar = n - 1
    max_t = end_bar - start_bar + 1
    pnls = np.zeros(max_t)
    nt = 0
    position = 0
    entry_price = 0.0
    entry_bar = 0

    for i in range(start_bar, end_bar):
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position - spread_pips
        else:
            pnl_pips = 0.0
        for k in range(n_ind):
            values[k] = inputs_2d[k, i]
        values[n_ind] = np.tanh(pnl_pips / 20.0)
        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)
        ob = values[output_indices[0]]
        os_ = values[output_indices[1]]
        of = values[output_indices[2]]

        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
            pnls[nt] = pnl; nt += 1; position = 0

        if position == 0:
            if ob > os_ and ob > of:
                position = 1; entry_price = mid_close[i]; entry_bar = i
            elif os_ > ob and os_ > of:
                position = -1; entry_price = mid_close[i]; entry_bar = i
        else:
            close = False; new_pos = 0
            if of > ob and of > os_:
                close = True
            elif position == 1 and os_ > ob and os_ > of:
                close = True; new_pos = -1
            elif position == -1 and ob > os_ and ob > of:
                close = True; new_pos = 1
            if close:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                pnls[nt] = pnl; nt += 1
                position = new_pos
                entry_price = mid_close[i] if new_pos != 0 else 0.0
                entry_bar = i

    if position != 0:
        pnl = (mid_close[end_bar] - entry_price) / pip * position - spread_pips
        pnls[nt] = pnl; nt += 1
    return pnls[:nt]


def mc_sign_test(pnls, n_shuffles=10000, seed=42):
    actual = pnls.sum()
    rng = np.random.RandomState(seed)
    better = 0
    for _ in range(n_shuffles):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(pnls))
        if (pnls * signs).sum() >= actual:
            better += 1
    return better / n_shuffles


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=None)
    parser.add_argument("--shuffles", type=int, default=10000)
    args = parser.parse_args()

    config_path = SCRIPT_DIR / "neat_config_4in_3out.ini"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_path))
    for name, fn in [('gauss', gauss_activation), ('sin', sin_activation),
                     ('cos', cos_activation), ('tanh', tanh_activation)]:
        try: config.genome_config.add_activation(name, fn)
        except: pass

    DATA_DIR = PROJECT_ROOT / "data" / "asi_mc_indicators"
    GENOME_DIR = SCRIPT_DIR / "results" / "ironnet"
    OUT_PATH = GENOME_DIR / "mc_validation_perpair.json"

    existing = {}
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text())

    pairs_to_run = args.pairs or ALL_PAIRS
    available = [p for p in pairs_to_run
                 if (GENOME_DIR / f"iron_v3_{p}_s42_best.pkl").exists()]
    missing = [p for p in pairs_to_run if p not in available]

    print("=" * 65)
    print(f"  MC Validation — Per-Pair IronNet V3 ({args.shuffles:,} shuffles)")
    print("=" * 65)
    if missing:
        print(f"  Skipping (no genome yet): {missing}")

    results = dict(existing)

    # JIT warmup
    print("  JIT warmup...", end=" ", flush=True)
    dummy_inp = np.zeros((4, 20))
    dummy_mid = np.zeros(20)
    dummy_bias = np.zeros(7); dummy_resp = np.ones(7); dummy_act = np.zeros(7, dtype=np.int64)
    dummy_cf = np.zeros(40, dtype=np.int64); dummy_ct = np.zeros(40, dtype=np.int64)
    dummy_cw = np.zeros(40); dummy_oi = np.array([4, 5, 6], dtype=np.int64)
    try:
        collect_pnls_jit(dummy_inp, dummy_mid, 0.0001, 1.5, 200,
                         4, 7, 7, dummy_bias, dummy_resp, dummy_act,
                         dummy_cf, dummy_ct, dummy_cw, dummy_oi)
    except Exception:
        pass
    print("done")

    for pair in available:
        if pair in existing:
            r = existing[pair]
            print(f"  {pair}: CACHED — {r['n_trades']}T, sign_p={r['sign_p']:.4f} {r['verdict']}")
            continue

        print(f"\n  {pair}", flush=True)
        print(f"  {'─'*50}", flush=True)

        pkl_path = GENOME_DIR / f"iron_v3_{pair}_s42_best.pkl"
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        genome = data["genome"]

        df = pd.read_parquet(DATA_DIR / f"{pair}_asi_mc.parquet")
        mid = df["mid_close"].values.astype(np.float64)
        n = len(mid)
        split = int(n * 0.7)

        # 4 inputs: mc_d_a, mc_dd_a, er_norm, upnl (upnl handled inside JIT as running pnl)
        inputs_oos = np.stack([
            df["mc_d_a"].values[split:].astype(np.float64),
            df["mc_dd_a"].values[split:].astype(np.float64),
            df["er_norm"].values[split:].astype(np.float64),
        ], axis=0)
        mid_oos = mid[split:].copy()

        net = extract_network(genome, config)
        n_inputs, n_outputs, n_eval, total_values, \
            node_bias, node_response, node_act, \
            conn_from, conn_to, conn_weight, output_indices = net

        pip = PAIR_PIP[pair]
        spread = PAIR_SPREAD[pair]

        pnls = collect_pnls_jit(
            inputs_oos, mid_oos, pip, spread, 200,
            n_inputs, n_eval, total_values,
            node_bias, node_response, node_act,
            conn_from, conn_to, conn_weight, output_indices)

        total = float(pnls.sum())
        nt = len(pnls)
        wins = int((pnls > 0).sum())
        n_oos_bars = len(mid_oos)
        days = n_oos_bars / (12 * 24)  # M5 bars per day = 288, but parquet may be M5
        # Use actual bar count from parquet
        try:
            bar_secs = int((df.index[1] - df.index[0]).total_seconds()) if hasattr(df.index[0], 'total_seconds') else 300
        except Exception:
            bar_secs = 300
        days = n_oos_bars * bar_secs / 86400
        avg_pips_day = total / days if days > 0 else 0.0

        print(f"  OOS: {nt}T, {total:+.1f}p ({avg_pips_day:.1f}p/d), WR={100*wins/nt:.0f}%", flush=True)
        print(f"  MC sign test ({args.shuffles:,} shuffles)...", end=" ", flush=True)
        sign_p = mc_sign_test(pnls, n_shuffles=args.shuffles)
        verdict = "PASS" if sign_p < 0.05 else "FAIL"
        print(f"sign_p={sign_p:.4f} {'✅' if sign_p < 0.05 else '❌'} {verdict}", flush=True)

        results[pair] = {
            "n_trades": nt,
            "total_pnl": round(total, 1),
            "pips_per_day": round(avg_pips_day, 1),
            "win_rate": round(100 * wins / nt, 1),
            "sign_p": round(sign_p, 4),
            "verdict": verdict,
        }
        OUT_PATH.write_text(json.dumps(results, indent=2))
        del df, inputs_oos, pnls

    print(f"\n{'='*65}")
    n_pass = sum(1 for r in results.values() if r["verdict"] == "PASS")
    print(f"  SUMMARY ({n_pass}/{len(results)} PASS)")
    print(f"{'='*65}")
    for pair in ALL_PAIRS:
        if pair in results:
            r = results[pair]
            sym = "✅" if r["verdict"] == "PASS" else "❌"
            print(f"  {sym} {pair:8s}: {r['n_trades']:4d}T  {r['pips_per_day']:5.1f}p/d  "
                  f"sign_p={r['sign_p']:.4f}  {r['verdict']}")
        else:
            print(f"  ⏳ {pair:8s}: pending genome")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
