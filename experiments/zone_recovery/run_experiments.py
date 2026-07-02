"""
Zone Recovery Experiment Runner
Executes all experiment phases and generates report.
Sends Telegram updates at each milestone.

Usage:
  python run_experiments.py [--phase 1|2|3|4|all]
"""

import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from engine import ZoneRecoveryEngine
from data_utils import load_m5, prepare_features, train_test_split_temporal
from backtest import (run_backtest, compute_metrics, walk_forward_test,
                      permutation_test, seed_robustness_test, run_5gate_validation)
from calibration import classic_param_grid_search, calibration_grid_search, analyze_ez_ratio_distribution
from notify import send_telegram

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PAIR = "EUR_USD"
SPREAD_PIPS = 1.4  # OANDA EUR/USD typical spread


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 0: Data EDA + E/Z ratio analysis
# ══════════════════════════════════════════════════════════════════════════════

def phase0_eda():
    print(f"\n{'='*60}")
    print(f"PHASE 0: EDA + E/Z Ratio Analysis  [{ts()}]")
    print(f"{'='*60}")

    df = load_m5(PAIR)
    feats = prepare_features(df)

    print(f"EUR_USD M5: {len(df):,} bars | {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    # ATR distribution
    atr_s = feats["atr_short"]
    atr_l = feats["atr_long"]
    valid = ~(np.isnan(atr_s) | np.isnan(atr_l))
    print(f"\nATR_short (H1×8) median: {np.nanmedian(atr_s)/0.0001:.1f} pips")
    print(f"ATR_long  (D1×20) median: {np.nanmedian(atr_l)/0.0001:.1f} pips")

    # Original cBot parameters (from Roni_cBot_Prod.cs)
    orig_hz = 10.25  # pips (1025 ticks * 0.01 pip/tick)
    orig_tgt = 6.0   # pips (600 ticks * 0.01 pip/tick)
    orig_ez = orig_tgt / (2 * orig_hz)
    print(f"\nOriginal cBot params:")
    print(f"  halfZone = {orig_hz} pips | target = {orig_tgt} pips | E/Z = {orig_ez:.3f}")

    # Analyze ATR-calibrated E/Z ratios
    for hz_m, tgt_m in [(0.3, 1.0), (0.5, 1.5), (0.7, 2.0)]:
        stats = analyze_ez_ratio_distribution(PAIR, hz_m, tgt_m)
        print(f"\n  ATR mult hz={hz_m:.1f}×S tgt={tgt_m:.1f}×L:")
        print(f"    Zone half = {stats['half_zone_median_pips']:.1f}p | Target = {stats['target_median_pips']:.1f}p")
        print(f"    E/Z median = {stats['ez_ratio_median']:.2f} | in [6,15]: {stats['ez_in_range_6_15_pct']*100:.0f}%")

    msg = (f"🟡 Zone Recovery Phase 0 done [{ts()}]\n"
           f"EUR_USD M5: {len(df):,} bars\n"
           f"ATR_S: {np.nanmedian(atr_s)/0.0001:.1f}p | ATR_L: {np.nanmedian(atr_l)/0.0001:.1f}p\n"
           f"Original cBot E/Z: {orig_ez:.3f}")
    send_telegram(msg)
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Classic cBot parameter sweep
# ══════════════════════════════════════════════════════════════════════════════

def phase1_classic_sweep():
    print(f"\n{'='*60}")
    print(f"PHASE 1: Classic cBot Grid Search  [{ts()}]")
    print(f"{'='*60}")

    send_telegram(f"🔍 Phase 1 starting: classic cBot grid search on EUR_USD M5\nParams: halfZone 5-25p × target 3-15p × PF 1.0-1.3")

    grid = classic_param_grid_search(
        pair=PAIR,
        half_zone_pips_list=[5, 8, 10, 10.25, 12, 15, 20, 25],
        target_beyond_pips_list=[3, 5, 6, 8, 10, 12, 15, 20],
        profit_factor_list=[1.0, 1.1, 1.19, 1.3, 1.5],
        use_is_data_frac=0.7,
    )

    grid = grid.sort_values("sharpe", ascending=False)
    path = RESULTS_DIR / "phase1_classic_grid.csv"
    grid.to_csv(path, index=False)
    print(f"\nTop 10 by Sharpe:")
    print(grid.head(10)[["half_zone_pips", "target_beyond_pips", "profit_factor",
                           "n_cycles", "net_pnl_pips", "sharpe", "win_rate",
                           "avg_legs", "max_legs_hit_pct"]].to_string())

    best = grid.iloc[0]
    msg = (f"🟡 Phase 1 done: {len(grid)} configs tested\n"
           f"🏆 Best: hz={best['half_zone_pips']}p tgt={best['target_beyond_pips']}p pf={best['profit_factor']:.2f}\n"
           f"   pnl={best['net_pnl_pips']:+.1f}p sharpe={best['sharpe']:.3f} n={best['n_cycles']}")
    send_telegram(msg)
    return grid


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: ATR-calibrated grid search
# ══════════════════════════════════════════════════════════════════════════════

def phase2_atr_calibration():
    print(f"\n{'='*60}")
    print(f"PHASE 2: ATR-Calibrated Grid Search  [{ts()}]")
    print(f"{'='*60}")

    send_telegram(f"🔍 Phase 2 starting: ATR-calibrated grid search\nhz_mult 0.3-1.0 × target_mult 1.0-3.0")

    grid = calibration_grid_search(
        pair=PAIR,
        half_zone_mults=np.arange(0.3, 1.1, 0.1),
        target_mults=np.arange(1.0, 3.5, 0.25),
        max_legs=10,
        use_is_data_frac=0.7,
    )

    grid = grid.sort_values("sharpe", ascending=False)
    path = RESULTS_DIR / "phase2_atr_grid.csv"
    grid.to_csv(path, index=False)
    print(f"\nTop 10 by Sharpe:")
    print(grid.head(10)[["half_zone_mult", "target_mult", "median_ez_ratio",
                           "n_cycles", "net_pnl_pips", "sharpe", "sqn",
                           "avg_legs", "max_legs_hit_pct"]].to_string())

    best = grid.iloc[0]
    msg = (f"🟡 Phase 2 done: ATR calibration\n"
           f"🏆 Best: hz_mult={best['half_zone_mult']:.1f} tgt_mult={best['target_mult']:.2f}\n"
           f"   E/Z={best['median_ez_ratio']:.1f} | pnl={best['net_pnl_pips']:+.1f}p sharpe={best['sharpe']:.3f}")
    send_telegram(msg)
    return grid


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Walk-forward + 5-gate validation on best configs
# ══════════════════════════════════════════════════════════════════════════════

def phase3_validation(classic_grid: pd.DataFrame, atr_grid: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"PHASE 3: Walk-Forward + 5-Gate Validation  [{ts()}]")
    print(f"{'='*60}")

    send_telegram("🔍 Phase 3: Walk-forward validation on top candidates")

    df = load_m5(PAIR)
    feats = prepare_features(df)
    n = len(feats["close"])
    train_n = int(n * 0.7)
    train_f = {k: v[:train_n] for k, v in feats.items()}
    test_f  = {k: v[train_n:] for k, v in feats.items()}

    results_all = []

    # Test top 5 classic configs
    print("\n--- Classic cBot top candidates ---")
    for _, row in classic_grid.head(5).iterrows():
        engine = ZoneRecoveryEngine(
            mode="classic",
            half_zone_pips=row["half_zone_pips"],
            target_beyond_pips=row["target_beyond_pips"],
            profit_factor=row["profit_factor"],
            base_unit=1000, max_legs=10, sizing_mode="dynamic", spread_pips=SPREAD_PIPS,
        )
        vr = run_5gate_validation(engine, feats, train_f, test_f)
        vr["config"] = f"classic_hz{row['half_zone_pips']}_tgt{row['target_beyond_pips']}_pf{row['profit_factor']:.2f}"
        vr["mode"] = "classic"
        vr["gates_passed"] = vr["gates_passed"]
        results_all.append(vr)
        gates = vr["gates"]
        print(f"  {vr['config']}: {vr['gates_passed']}/5 gates | "
              f"OOS pnl={vr['oos_metrics']['net_pnl_pips']:+.1f}p | {vr['verdict']}")
        for gk, gv in gates.items():
            print(f"    {'✅' if gv else '❌'} {gk}")

    # Test top 5 ATR configs
    print("\n--- ATR-calibrated top candidates ---")
    for _, row in atr_grid.head(5).iterrows():
        engine = ZoneRecoveryEngine(
            mode="atr",
            atr_zone_mult=row["half_zone_mult"],
            atr_target_mult=row["target_mult"],
            base_unit=1000, max_legs=10, sizing_mode="dynamic", spread_pips=SPREAD_PIPS,
        )
        vr = run_5gate_validation(engine, feats, train_f, test_f)
        vr["config"] = f"atr_hz{row['half_zone_mult']:.1f}_tgt{row['target_mult']:.2f}"
        vr["mode"] = "atr"
        results_all.append(vr)
        print(f"  {vr['config']}: {vr['gates_passed']}/5 gates | "
              f"OOS pnl={vr['oos_metrics']['net_pnl_pips']:+.1f}p | {vr['verdict']}")
        for gk, gv in vr["gates"].items():
            print(f"    {'✅' if gv else '❌'} {gk}")

    # Save
    path = RESULTS_DIR / "phase3_validation.json"
    with open(path, "w") as f:
        # Convert non-serializable
        def make_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj
        json.dump([make_serializable(r) for r in results_all], f, indent=2)

    passes = [r for r in results_all if r["verdict"] == "PASS"]
    msg = (f"🟡 Phase 3 done\n"
           f"Configs tested: {len(results_all)}\n"
           f"✅ PASS: {len(passes)} / {len(results_all)}\n")
    if passes:
        best = max(passes, key=lambda r: r["oos_metrics"]["net_pnl_pips"])
        msg += (f"🏆 Best: {best['config']}\n"
                f"   OOS pnl={best['oos_metrics']['net_pnl_pips']:+.1f}p "
                f"SQN={best['oos_metrics']['sqn']:.2f}")
    send_telegram(msg)
    return results_all


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Sizing mode comparison
# ══════════════════════════════════════════════════════════════════════════════

def phase4_sizing_comparison(best_config: dict):
    """Compare dynamic vs convex vs linear sizing on the best classical config."""
    print(f"\n{'='*60}")
    print(f"PHASE 4: Sizing Mode Comparison  [{ts()}]")
    print(f"{'='*60}")

    send_telegram("🔍 Phase 4: Sizing mode comparison (dynamic vs convex vs linear)")

    df = load_m5(PAIR)
    feats = prepare_features(df)
    n = len(feats["close"])
    test_f = {k: v[int(n*0.7):] for k, v in feats.items()}

    hz = best_config.get("half_zone_pips", 10.25)
    tgt = best_config.get("target_beyond_pips", 6.0)
    pf = best_config.get("profit_factor", 1.19)

    results = {}
    for mode in ["dynamic", "convex", "linear"]:
        engine = ZoneRecoveryEngine(
            mode="classic",
            half_zone_pips=hz,
            target_beyond_pips=tgt,
            profit_factor=pf,
            base_unit=1000, max_legs=10,
            sizing_mode=mode,
            convex_exponent=1.5,
            spread_pips=SPREAD_PIPS,
        )
        res = run_backtest(engine, test_f)
        m = compute_metrics(res)
        results[mode] = m
        print(f"  {mode:10s}: pnl={m['net_pnl_pips']:+8.1f}p sharpe={m['sharpe']:.3f} "
              f"avgLegs={m['avg_legs']:.1f} maxLegsPct={m['max_legs_hit_pct']*100:.1f}%")

    msg = (f"🟡 Phase 4: Sizing comparison\n"
           f"Config: hz={hz}p tgt={tgt}p pf={pf:.2f}\n")
    for mode, m in results.items():
        msg += f"  {mode}: {m['net_pnl_pips']:+.1f}p Sharpe={m['sharpe']:.3f}\n"
    send_telegram(msg)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["0", "1", "2", "3", "4", "all"])
    args = parser.parse_args()

    send_telegram(
        f"🚀 Zone Recovery Experiment START [{ts()}]\n"
        f"Pair: EUR_USD | Data: M5 | Mode: {args.phase}\n"
        f"Faithfully porting Roni_cBot_Prod.cs + ATR-calibrated extension\n"
        f"Post-RCA: all features causal, 5-gate validation"
    )

    try:
        feats = None
        classic_grid = None
        atr_grid = None
        validation_results = None

        if args.phase in ("0", "all"):
            feats = phase0_eda()

        if args.phase in ("1", "all"):
            classic_grid = phase1_classic_sweep()

        if args.phase in ("2", "all"):
            atr_grid = phase2_atr_calibration()

        if args.phase in ("3", "all"):
            if classic_grid is None:
                classic_grid = pd.read_csv(RESULTS_DIR / "phase1_classic_grid.csv")
            if atr_grid is None:
                atr_grid = pd.read_csv(RESULTS_DIR / "phase2_atr_grid.csv")
            validation_results = phase3_validation(classic_grid, atr_grid)

        if args.phase in ("4", "all"):
            if classic_grid is None:
                classic_grid = pd.read_csv(RESULTS_DIR / "phase1_classic_grid.csv")
            best_row = classic_grid.sort_values("sharpe", ascending=False).iloc[0]
            best_config = {
                "half_zone_pips": float(best_row["half_zone_pips"]),
                "target_beyond_pips": float(best_row["target_beyond_pips"]),
                "profit_factor": float(best_row["profit_factor"]),
            }
            phase4_sizing_comparison(best_config)

        send_telegram(f"✅ Zone Recovery Experiment COMPLETE [{ts()}]")

    except Exception as e:
        tb = traceback.format_exc()
        send_telegram(f"🔴 Zone Recovery CRASHED: {e}\n{tb[:500]}")
        raise


if __name__ == "__main__":
    main()
