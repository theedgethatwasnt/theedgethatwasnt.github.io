#!/usr/bin/env python3
"""
5-Gate Validation for Experiment C: SB_A IronNet
==================================================
Range bars have no timestamps — we use bar-segment equivalents.

Gates:
  1. OOS pips/day > 2× spread (meaningful edge net of costs)
  2. Permutation p < 0.05  (shuffle SB_A signal, check pips collapse)
  3. Bootstrap CI lower > 0  (per-trade pips bootstrap)
  4. All 5 WF OOS segments positive pips/day
  5. Drop-one of 4 segments: all positive pips/day

Robustness bonus (reported, not gated):
  - Seed stability: std(OOS p/d across seeds) / mean < 0.05
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parent
# Probe for project root by looking for lib/ dir
for _depth in range(6):
    if _depth < len(SCRIPT_DIR.parents):
        candidate = SCRIPT_DIR.parents[_depth]
    else:
        break
    if (candidate / "lib").is_dir():
        PROJECT_ROOT = candidate
        break
else:
    PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
for _extra in os.environ.get("PYTHONPATH", "").split(":"):
    if _extra and _extra not in sys.path:
        sys.path.insert(0, _extra)

from lib.incremental_topsbots import IncrementalTopsBots

DATA_DIR = Path(os.environ.get(
    "RANGE_DATA_DIR",
    str(PROJECT_ROOT / "data" / "range_bar_causal")
))
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
BARS_PER_DAY = 35.0  # approximate for range-10 bars

N_IN = 5; N_HID = 5; N_OUT = 2; N_ACTS = 9
W1_END = N_IN * N_HID; B1_END = W1_END + N_HID
W2_END = B1_END + N_HID * N_OUT; B2_END = W2_END + N_OUT
ACT_END = B2_END + N_HID

@njit(cache=True, inline="always")
def activate(z, act_id):
    if act_id == 0:   return np.tanh(z)
    elif act_id == 1: return np.sin(z)
    elif act_id == 2: return np.cos(z)
    elif act_id == 3: return np.exp(-z*z)
    elif act_id == 4: return 1.0 / np.cosh(z)
    elif act_id == 5: return -z * np.exp(-z*z)
    elif act_id == 6: return np.cos(z) * np.exp(-0.5*z*z)
    elif act_id == 7: return np.sinc(z / np.pi) if abs(z) > 1e-9 else 1.0
    else:             return np.tanh(z * np.cos(z))

@njit(cache=True)
def forward(genes, inp):
    h = np.empty(N_HID)
    for j in range(N_HID):
        z = genes[W1_END + j]
        for k in range(N_IN):
            z += genes[j * N_IN + k] * inp[k]
        act_id = int(genes[B2_END + j] * N_ACTS) % N_ACTS
        h[j] = activate(z, act_id)
    out = np.empty(N_OUT)
    for i in range(N_OUT):
        s = genes[W2_END + i]
        for j in range(N_HID):
            s += genes[B1_END + i * N_HID + j] * h[j]
        out[i] = s
    return out


@njit(cache=True)
def simulate_with_trades(
    genes, sba, mc_d, mc_dd, mid_close,
    spread_pips, pip_size, max_hold,
    enter_thresh=0.3, close_thresh=0.2,
):
    """Returns (entry_bars, exit_bars, trade_pips)."""
    N = len(mid_close)
    pos = 0; entry_price = 0.0; entry_bar = 0
    upnl = 0.0; mae = 0.0; mfe = 0.0

    entry_bars = np.empty(N, dtype=np.int64)
    exit_bars  = np.empty(N, dtype=np.int64)
    trade_pips = np.empty(N, dtype=np.float64)
    n_trades = 0

    for i in range(1, N):
        price = mid_close[i]
        sba_i = float(sba[i])
        inp = np.empty(N_IN)
        inp[0] = sba_i
        inp[1] = float(mc_d[i])
        inp[2] = float(mc_dd[i])
        inp[3] = np.tanh(upnl / 10.0)
        inp[4] = np.tanh(mae  / 10.0)
        out = forward(genes, inp)
        enter_conf = np.tanh(out[0])
        close_conf = np.tanh(out[1])

        if pos != 0:
            if pos == 1:
                upnl = (price - entry_price) / pip_size
            else:
                upnl = (entry_price - price) / pip_size
            if upnl < -mae:
                mae = -upnl
            if upnl > mfe:
                mfe = upnl
            force_close = (i - entry_bar) >= max_hold
            signal_close = close_conf > close_thresh
            if force_close or signal_close:
                entry_bars[n_trades] = entry_bar
                exit_bars[n_trades]  = i
                trade_pips[n_trades] = upnl
                n_trades += 1
                pos = 0; upnl = 0.0; mae = 0.0; mfe = 0.0

        if pos == 0:
            if enter_conf > enter_thresh and abs(sba_i) > 0.1:
                direction = 1 if sba_i > 0 else -1
                entry_price = price + direction * spread_pips * pip_size
                mae = spread_pips; upnl = -spread_pips; mfe = 0.0
                pos = direction; entry_bar = i

    return entry_bars[:n_trades], exit_bars[:n_trades], trade_pips[:n_trades]


def load_oos_data(pair: str) -> dict:
    pip  = PAIR_PIP[pair]
    fpath = DATA_DIR / f"{pair}_range10_causal.parquet"
    if not fpath.exists():
        fpath = DATA_DIR / f"{pair}_range10_asi_mc.parquet"
    df = pd.read_parquet(fpath)
    df = df.sort_index()

    close = df["mid_close"].values.astype(np.float64)
    tb = IncrementalTopsBots()
    sba_arr = np.empty(len(close), dtype=np.float32)
    for i in range(len(close)):
        s, _, _, _ = tb.update(close[i], close[i], close[i])
        sba_arr[i] = np.float32(s / 2.0)
    sba_norm = sba_arr

    mc_d  = df["mc_d"].values.astype(np.float32)
    mc_dd = df["mc_dd"].values.astype(np.float32)

    n = len(close)
    is_end = int(n * 0.7)

    return {
        "sba":       sba_norm[is_end:],
        "mc_d":      mc_d[is_end:],
        "mc_dd":     mc_dd[is_end:],
        "mid_close": close[is_end:],
        "n_bars":    n - is_end,
        "pip": pip,
        "spread": PAIR_SPREAD[pair],
    }


def run_five_gates(pair: str, genes: np.ndarray, all_seeds_data: list,
                   n_perms: int = 200, rng_seed: int = 42):
    oos = load_oos_data(pair)
    spread = oos["spread"]; pip = oos["pip"]
    sba    = np.ascontiguousarray(oos["sba"])
    mc_d   = np.ascontiguousarray(oos["mc_d"])
    mc_dd  = np.ascontiguousarray(oos["mc_dd"])
    mid    = np.ascontiguousarray(oos["mid_close"])
    n_oos  = oos["n_bars"]
    max_hold = 100

    eb, xb, tp = simulate_with_trades(genes, sba, mc_d, mc_dd, mid, spread, pip, max_hold)
    n_trades = len(tp)
    if n_trades < 20:
        return {"pair": pair, "gates_passed": 0, "note": "too few trades",
                "oos_ppd": 0.0, "oos_sharpe": None}

    oos_ppd = tp.sum() / (n_oos / BARS_PER_DAY)

    # Per-trade Sharpe (annualized by trades_per_year)
    trades_per_day = n_trades / (n_oos / BARS_PER_DAY)
    trades_per_year = trades_per_day * 252
    per_trade_sharpe = float(tp.mean() / (tp.std(ddof=1) + 1e-9) * np.sqrt(trades_per_year))

    # Gate 1: OOS pips/day > 2× spread (meaningful after costs)
    g1 = oos_ppd > 2.0 * spread

    # Gate 2: Permutation p < 0.05
    # Shuffle SB_A signal → measure pips/day → p = frac(perm_ppd >= real_ppd)
    rng = np.random.default_rng(rng_seed)
    perm_ppds = []
    sba_copy = sba.copy()
    for _ in range(n_perms):
        perm_sba = rng.permutation(sba_copy).astype(np.float32)
        perm_sba = np.ascontiguousarray(perm_sba)
        _, _, ptp = simulate_with_trades(genes, perm_sba, mc_d, mc_dd, mid, spread, pip, max_hold)
        perm_ppds.append(ptp.sum() / (n_oos / BARS_PER_DAY) if len(ptp) > 0 else -1e9)
    p_value = float(np.mean(np.array(perm_ppds) >= oos_ppd))
    g2 = p_value < 0.05

    # Gate 3: Bootstrap CI of per-trade pips, lower bound > 0
    boot_means = []
    for _ in range(1000):
        idx = rng.integers(0, n_trades, size=n_trades)
        boot_means.append(float(tp[idx].mean()))
    ci_lo = float(np.nanpercentile(boot_means, 2.5))
    ci_hi = float(np.nanpercentile(boot_means, 97.5))
    g3 = ci_lo > 0.0

    # Gate 4: All 5 WF segments positive pips/day
    chunk = n_oos // 5
    fold_ppds = []
    for k in range(5):
        s_bar = k * chunk
        e_bar = (k + 1) * chunk if k < 4 else n_oos
        mask = (eb >= s_bar) & (xb < e_bar)
        fold_pips = tp[mask].sum()
        fold_bars = e_bar - s_bar
        fold_ppds.append(fold_pips / (fold_bars / BARS_PER_DAY))
    n_pos_folds = sum(1 for x in fold_ppds if x > 0)
    g4 = n_pos_folds == 5

    # Gate 5: Drop-one of 4 equal segments, all positive
    seg = n_oos // 4
    seg_ppds = {}
    all_pos_drop = True
    for k in range(4):
        # Drop segment k, evaluate on the remaining 3
        mask_drop = (xb >= k * seg) & (xb < (k + 1) * seg)
        kept_pips = tp[~mask_drop].sum()
        kept_bars = n_oos - seg
        ppd = kept_pips / (kept_bars / BARS_PER_DAY)
        seg_ppds[f"seg{k}"] = round(ppd, 2)
        if ppd <= 0:
            all_pos_drop = False
    g5 = all_pos_drop

    # Seed stability (bonus)
    seed_ppds = [d["oos_pips_per_day"] for d in all_seeds_data]
    seed_std_pct = float(np.std(seed_ppds) / (np.mean(seed_ppds) + 1e-9)) * 100

    gates = [g1, g2, g3, g4, g5]
    n_gates = sum(gates)
    gate_str = "".join(["✅" if g else "❌" for g in gates])

    perm_95th = float(np.percentile(perm_ppds, 95))
    print(f"\n  {pair}: {gate_str} ({n_gates}/5)")
    print(f"    OOS: {oos_ppd:.1f} p/d, per-trade Sharpe={per_trade_sharpe:.3f}")
    print(f"    trades={n_trades} ({trades_per_day:.0f}/day), spread={spread}p")
    print(f"    perm_p={p_value:.4f}  (real={oos_ppd:.1f} vs 95th={perm_95th:.1f})")
    print(f"    Bootstrap CI: [{ci_lo:.3f}, {ci_hi:.3f}] pips/trade")
    print(f"    WF folds: {[f'{x:.1f}' for x in fold_ppds]} → {n_pos_folds}/5 positive")
    print(f"    Drop-one segs: {seg_ppds}")
    print(f"    Seed stability: {seed_std_pct:.1f}% CV across {len(seed_ppds)} seeds")

    return {
        "pair": pair,
        "oos_ppd": round(oos_ppd, 2),
        "per_trade_sharpe": round(per_trade_sharpe, 4),
        "n_trades": n_trades,
        "trades_per_day": round(trades_per_day, 1),
        "perm_p": round(p_value, 4),
        "perm_95th": round(perm_95th, 2),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "wf_folds": [round(x, 2) for x in fold_ppds],
        "n_pos_folds": n_pos_folds,
        "drop_one_segs": seg_ppds,
        "seed_std_pct": round(seed_std_pct, 2),
        "seed_ppds": seed_ppds,
        "gate_str": gate_str,
        "gates_passed": n_gates,
        "g1_ppd": g1,
        "g2_perm": g2,
        "g3_ci": g3,
        "g4_wf": g4,
        "g5_drop": g5,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="*", default=None)
    parser.add_argument("--perms", type=int, default=200)
    args = parser.parse_args()

    all_pkls = sorted(RESULTS_DIR.rglob("sb_a_ironnet_*_s*_best.pkl"))
    pair_seeds: dict[str, list] = {}
    for p in all_pkls:
        with open(p, "rb") as f:
            d = pickle.load(f)
        pair = d["pair"]
        if args.pairs and pair not in args.pairs:
            continue
        if pair not in pair_seeds:
            pair_seeds[pair] = []
        pair_seeds[pair].append(d)

    if not pair_seeds:
        print("No results found.")
        return

    print(f"\n{'='*70}")
    print(f"5-Gate Evaluation: SB_A IronNet ({len(pair_seeds)} pairs)")
    print(f"{'='*70}")

    all_results = []
    for pair in sorted(pair_seeds):
        best = max(pair_seeds[pair], key=lambda d: d["oos_pips_per_day"])
        genes = np.ascontiguousarray(best["genes"], dtype=np.float64)
        print(f"\n{'─'*60}")
        seeds_str = ", ".join(f"s{d['seed']}={d['oos_pips_per_day']:.0f}" for d in sorted(pair_seeds[pair], key=lambda x: x["seed"]))
        print(f"  {pair}  → [{seeds_str}]  best seed={best['seed']}")
        result = run_five_gates(pair, genes, pair_seeds[pair], n_perms=args.perms)
        result["best_seed"] = best["seed"]
        all_results.append(result)

    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY — SB_A IronNet 5-Gate Results")
    print(f"{'='*70}")
    sorted_res = sorted(all_results, key=lambda r: (-r["gates_passed"], -(r["oos_ppd"] or 0)))
    deployable = [r for r in all_results if r["gates_passed"] == 5]

    print(f"\n{'Pair':10s}  {'ppd':>7s}  {'Sharpe':>7s}  {'p':>6s}  {'SeedCV':>6s}  Gates")
    print("─" * 65)
    for r in sorted_res:
        sh = f"{r['per_trade_sharpe']:.3f}" if r.get("per_trade_sharpe") is not None else "  nan"
        cv = f"{r.get('seed_std_pct', 0):.1f}%"
        print(f"  {r['pair']:10s}  {r['oos_ppd']:7.1f}  {sh:>7s}  "
              f"{r['perm_p']:6.4f}  {cv:>6s}  {r['gate_str']} ({r['gates_passed']}/5)")

    print(f"\n🏆 Deployable (5/5 gates): {len(deployable)}")
    for r in deployable:
        print(f"    → {r['pair']:10s}  {r['oos_ppd']:.1f} p/d  Sharpe={r['per_trade_sharpe']:.3f}  "
              f"SeedCV={r['seed_std_pct']:.1f}%")

    out_path = RESULTS_DIR / "five_gates_ironnet_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else str(x))
    print(f"\nSaved → {out_path}")
    return all_results


if __name__ == "__main__":
    main()
