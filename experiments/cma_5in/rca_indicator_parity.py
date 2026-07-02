#!/usr/bin/env python3
"""Indicator parity test: training pipeline vs live curator class.

Simulates the curator's streaming buffer and checks if it produces identical
mc_d_a/mc_dd_a/er_norm values to the training pipeline on the same OHLC.

If diff > epsilon, live curator is producing different inputs than training,
explaining why backtest predicts +30 p/d but live loses -96 p/d.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.asi_indicator import compute_asi_mc, compute_asi, sma_jit, compute_mc_on_series, TF_BARS_S5, TF_WEIGHTS, N_TFS
from lib.indicators import ASIMC
from numba import njit


@njit(cache=True)
def er_norm_batch(closes, window=60):
    """Training pipeline er_norm (matches train_cma_v2._compute_er_norm_v3)."""
    n = len(closes)
    out = np.zeros(n)
    hp = np.pi / 2.0
    for i in range(window, n):
        net = abs(closes[i] - closes[i - window])
        path = 0.0
        for j in range(i - window + 1, i + 1):
            path += abs(closes[j] - closes[j - 1])
        if path > 0.0:
            out[i] = np.arctan((net / path) / 0.3) / hp
    return out


def er_norm_curator(closes_list):
    """Exactly mirror the curator's er_norm code from services/curator/main.py."""
    if len(closes_list) < 61:
        return 0.0
    _c = np.array(closes_list[-61:], dtype=np.float64)
    _net = abs(float(_c[-1]) - float(_c[0]))
    _path = float(np.sum(np.abs(np.diff(_c))))
    _er = _net / _path if _path > 0 else 0.0
    return float(np.arctan(_er / 0.3) / (np.pi / 2))


def main():
    pair = "CHF_JPY"
    print(f"Loading {pair} M5 OHLC...")
    df = pd.read_parquet(PROJECT_ROOT / "data" / "m5_ohlc" / f"{pair}_M5.parquet")
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    n = len(c)
    print(f"  {n} M5 bars")

    # ══ TRAINING PIPELINE (batch, full array) ══
    print("\n=== TRAINING PIPELINE ===")
    mc_d_train, mc_dd_train = compute_asi_mc(o, h, l, c, n)
    er_train = er_norm_batch(c, 60)
    print(f"  mc_d last 5: {mc_d_train[-5:]}")
    print(f"  mc_dd last 5: {mc_dd_train[-5:]}")
    print(f"  er_norm last 5: {er_train[-5:]}")

    # ══ CURATOR CLASS (streaming) ══
    print("\n=== CURATOR CLASS (streaming simulation) ===")
    print("Simulating curator: warmup 1000 bars, then append one-at-a-time")
    print("Capturing last-bar values after each append for last 200 bars...")

    asi_mc_instance = ASIMC()

    # Warmup with first 1000 bars (matches curator.warmup behavior)
    for i in range(min(1000, n)):
        asi_mc_instance.append_m5(o[i], h[i], l[i], c[i])

    # Now simulate live streaming — append bars starting from 1000
    # Capture mc_d, mc_dd, er_norm after each append
    mc_d_curator = np.zeros(n)
    mc_dd_curator = np.zeros(n)
    er_curator = np.zeros(n)

    # Fill in warmup period with what compute() returns at that point
    mc_d_last, mc_dd_last = asi_mc_instance.compute()
    er_last = er_norm_curator(asi_mc_instance.m5_c)
    for i in range(min(1000, n)):
        mc_d_curator[i] = mc_d_last
        mc_dd_curator[i] = mc_dd_last
        er_curator[i] = er_last

    # Streaming simulation — append remaining bars
    # For speed, only capture last 200 bars' worth
    capture_start = max(1000, n - 200)
    print(f"  Capturing bars {capture_start}..{n}")
    for i in range(1000, n):
        asi_mc_instance.append_m5(o[i], h[i], l[i], c[i])
        if i >= capture_start:
            mc_d_now, mc_dd_now = asi_mc_instance.compute()
            er_now = er_norm_curator(asi_mc_instance.m5_c)
            mc_d_curator[i] = mc_d_now
            mc_dd_curator[i] = mc_dd_now
            er_curator[i] = er_now

    print(f"  mc_d last 5: {mc_d_curator[-5:]}")
    print(f"  mc_dd last 5: {mc_dd_curator[-5:]}")
    print(f"  er_norm last 5: {er_curator[-5:]}")

    # ══ DIFF ══
    print("\n=== PARITY CHECK (last 200 bars) ===")
    window = slice(-200, None)

    for name, train, curator in [
        ("mc_d_a", mc_d_train[window], mc_d_curator[window]),
        ("mc_dd_a", mc_dd_train[window], mc_dd_curator[window]),
        ("er_norm", er_train[window], er_curator[window]),
    ]:
        diff = np.abs(train - curator)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        match_count = np.sum(diff < 0.001)
        print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
              f"match<0.001: {match_count}/{len(diff)}")

        if max_diff > 0.001:
            # Show worst mismatches
            worst_idx = np.argsort(diff)[-5:]
            print(f"    Worst 5 mismatches (idx from end):")
            for idx in worst_idx:
                absolute_idx = n - 200 + idx
                print(f"      bar {absolute_idx}: train={train[idx]:+.6f}, "
                      f"curator={curator[idx]:+.6f}, diff={diff[idx]:+.6f}")

    # Side-by-side last 10 bars
    print("\n=== Side-by-side: last 10 bars ===")
    print(f"{'Bar':>6} {'train_mc_d':>12} {'curator_mc_d':>14} {'train_mc_dd':>14} "
          f"{'curator_mc_dd':>16} {'train_er':>10} {'curator_er':>12}")
    for i in range(n - 10, n):
        print(f"{i:>6} {mc_d_train[i]:>+12.6f} {mc_d_curator[i]:>+14.6f} "
              f"{mc_dd_train[i]:>+14.6f} {mc_dd_curator[i]:>+16.6f} "
              f"{er_train[i]:>10.6f} {er_curator[i]:>12.6f}")


if __name__ == "__main__":
    main()
