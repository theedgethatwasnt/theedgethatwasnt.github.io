#!/usr/bin/env python3
"""Monte Carlo sign-shuffle validation for IronNet genomes on OOS data."""
import sys, os, pickle, math, numpy as np, pandas as pd
from pathlib import Path
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2] if len(SCRIPT_DIR.parts) > 4 else SCRIPT_DIR
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

def gauss_activation(x): return math.exp(-x * x)
def sin_activation(x): return math.sin(x)
def cos_activation(x): return math.cos(x)
def tanh_activation(x): return math.tanh(x)
def sech_activation(x): return 1.0 / math.cosh(max(min(x, 50), -50))
def dog_activation(x): return math.exp(-x*x/2) - 0.5*math.exp(-x*x/8)
def gabor_activation(x): return math.exp(-2*x*x) * math.cos(2*math.pi*x)
def sinc_activation(x): return math.sin(math.pi*x)/(math.pi*x) if abs(x) > 1e-7 else 1.0
def morlet_activation(x): return math.sin(x) * math.exp(-x*x/2)


@njit(cache=True)
def collect_pnls_jit(inputs_2d, mid_close, pip, spread_pips, max_hold,
                     n_inputs, n_eval, total_values,
                     node_bias, node_response, node_act,
                     conn_from, conn_to, conn_weight, output_indices):
    """Collect individual trade P&Ls from OOS evaluation."""
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
        signs = rng.choice(np.array([-1, 1]), size=len(pnls))
        if (pnls * signs).sum() >= actual:
            better += 1
    return better / n_shuffles


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", default="M5", choices=["M5", "H1"])
    parser.add_argument("--mode", default="v3", choices=["v3", "v7", "s1"])
    args = parser.parse_args()
    tf = args.tf
    mode = args.mode

    config_name = {"v3": "neat_config_4in_3out.ini", "v7": "neat_config_7in_3out.ini", "s1": "neat_config_5in_3out.ini"}[mode]
    config_path = SCRIPT_DIR / config_name
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(config_path))
    for name, fn in [('gauss', gauss_activation), ('sin', sin_activation),
                     ('cos', cos_activation), ('tanh', tanh_activation),
                     ('sech', sech_activation), ('dog', dog_activation),
                     ('gabor', gabor_activation), ('sinc', sinc_activation),
                     ('morlet', morlet_activation)]:
        try: config.genome_config.add_activation(name, fn)
        except: pass

    results = {}
    ALL_PAIRS = list(PAIR_PIP.keys())

    if mode == "s1":
        max_hold = 17
        pairs_pkls = [(p, f"iron_s1_H1_{p}_s42_best") for p in ALL_PAIRS]
        pkl_dir = SCRIPT_DIR / "results" / "ironnet_s1"
        version = "S1"
    elif mode == "v7":
        V7_DATA_DIR = PROJECT_ROOT / "data" / "v7_indicators"
        V7_IND_COLS = ["bb_width", "stoch_d", "macd_hist", "range_pos_30", "aroon_osc", "mc_d_a"]
        max_hold = 17
        pairs_pkls = [(p, f"iron_v7_H1_{p}_s42_best") for p in ALL_PAIRS]
        pkl_dir = SCRIPT_DIR / "results" / "ironnet_v7"
        version = "V7"
    elif tf == "M5":
        max_hold = 200
        pairs_pkls = [(p, f"iron_v3_{p}") for p in ALL_PAIRS]
        pkl_dir = PROJECT_ROOT / "models"
        version = "V3"
    else:
        max_hold = 17
        pairs_pkls = [(p, f"iron_v3_H1_{p}_s42_best") for p in ALL_PAIRS]
        pkl_dir = SCRIPT_DIR / "results" / "ironnet_h1"
        version = "V3"

    print("=" * 65)
    print(f"  Monte Carlo Sign-Shuffle Validation — IronNet {version} {tf}")
    print("  10,000 shuffles per pair on OOS (30% holdout)")
    print("=" * 65)

    for pair, pkl_name in pairs_pkls:
        print(f"\n  {pair}")
        print(f"  {'─'*50}")

        pkl_path = pkl_dir / f"{pkl_name}.pkl"
        if not pkl_path.exists():
            print(f"  SKIP: {pkl_path} not found")
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        genome = data["genome"]

        if mode == "s1":
            DATA_DIR = PROJECT_ROOT / "data" / "asi_mc_indicators"
            df = pd.read_parquet(DATA_DIR / f"{pair}_asi_mc.parquet")
            if tf == "H1":
                df = df.set_index("timestamp")
                agg = {"mid_close": "last", "mc_d_a": "last", "mc_dd_a": "last", "er_norm": "last", "sb_a": "last"}
                df = df.resample("1h").agg(agg).dropna(subset=["mid_close"]).reset_index()
            mid = df["mid_close"].values.astype(np.float64)
            n = len(mid); split = int(n * 0.7)
            inputs_oos = np.stack([
                df["mc_d_a"].values[split:].astype(np.float64),
                df["mc_dd_a"].values[split:].astype(np.float64),
                df["er_norm"].values[split:].astype(np.float64),
                df["sb_a"].values[split:].astype(np.float64),
            ], axis=0)
        elif mode == "v7":
            df = pd.read_parquet(V7_DATA_DIR / f"{pair}_v7.parquet")
            df = df.set_index("timestamp")
            agg = {"mid_close": "last"}
            for c in V7_IND_COLS:
                agg[c] = "last"
            df = df.resample("1h").agg(agg).dropna(subset=["mid_close"]).reset_index()

            mid = df["mid_close"].values.astype(np.float64)
            n = len(mid)
            split = int(n * 0.7)

            inputs_raw = np.stack([df[c].values[split:].astype(np.float64) for c in V7_IND_COLS], axis=0)
            # V7 normalization
            inputs_raw[0] = inputs_raw[0] * 20.0
            inputs_raw[2] = np.clip(inputs_raw[2] / 2.0, -1.0, 1.0)
            inputs_oos = inputs_raw
        else:
            DATA_DIR = PROJECT_ROOT / "data" / "asi_mc_indicators"
            df = pd.read_parquet(DATA_DIR / f"{pair}_asi_mc.parquet")
            if tf == "H1":
                df = df.set_index("timestamp")
                agg = {"mid_close": "last", "mc_d_a": "last", "mc_dd_a": "last", "er_norm": "last"}
                df = df.resample("1h").agg(agg).dropna(subset=["mid_close"]).reset_index()

            mid = df["mid_close"].values.astype(np.float64)
            n = len(mid)
            split = int(n * 0.7)

            inputs_oos = np.stack([
                df["mc_d_a"].values[split:].astype(np.float64),
                df["mc_dd_a"].values[split:].astype(np.float64),
                df["er_norm"].values[split:].astype(np.float64),
            ], axis=0)

        mid_oos = mid[split:]

        net = extract_network(genome, config)
        pip = PAIR_PIP[pair]
        spread = PAIR_SPREAD[pair]

        pnls = collect_pnls_jit(
            inputs_oos, mid_oos, pip, spread, max_hold,
            net[0], net[2], net[3], net[4], net[5], net[6],
            net[7], net[8], net[9], net[10])

        total = float(pnls.sum())
        nt = len(pnls)
        wins = int((pnls > 0).sum())
        print(f"  OOS: {nt} trades, {total:+.1f} pips, WR={100*wins/nt:.0f}%")

        print(f"  MC sign test (10,000 shuffles)...", end=" ", flush=True)
        sign_p = mc_sign_test(pnls, n_shuffles=10000)
        verdict = "PASS" if sign_p < 0.05 else "FAIL"
        print(f"sign_p = {sign_p:.4f} {'✅' if sign_p < 0.05 else '❌'} {verdict}")

        results[pair] = {
            "n_trades": nt, "total_pnl": round(total, 1),
            "win_rate": round(100 * wins / nt, 1),
            "sign_p": round(sign_p, 4), "verdict": verdict,
        }

        del df, inputs_oos

    print(f"\n{'='*65}")
    print(f"  SUMMARY — {version} {tf}")
    print(f"{'='*65}")
    for pair, r in results.items():
        print(f"  {pair}: {r['n_trades']}T {r['total_pnl']:+.1f}p "
              f"sign_p={r['sign_p']:.4f} {r['verdict']}")

    # Save results
    import json
    if mode == "v7":
        out_dir = SCRIPT_DIR / "results" / "ironnet_v7"
    elif tf == "M5":
        out_dir = SCRIPT_DIR / "results" / "ironnet"
    else:
        out_dir = SCRIPT_DIR / "results" / "ironnet_h1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mc_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
