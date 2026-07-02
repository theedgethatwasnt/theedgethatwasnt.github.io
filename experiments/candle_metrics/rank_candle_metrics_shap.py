#!/usr/bin/env python3
"""
SHAP Ranking of Candle Metrics — M5 OHLC
==========================================
Vectorized computation of all candle metrics (single-bar + two-bar + log deltas)
from M5 OHLC data. Ranks features against zigzag labels via LightGBM + SHAP.

Features (~22 per bar):
  Single-bar: body_ratio, upper_wick_ratio, lower_wick_ratio, wick_ratio, close_position
  Two-bar: range_expansion, body_overlap, two_bar_momentum, gap (normalized)
  Log deltas: dlog_body_ratio, dlog_upper_wick_ratio, dlog_lower_wick_ratio, dlog_wick_ratio,
              dlog_close_position, dlog_body_size, dlog_range_size
  Categorical (as float): direction, reversal_score, is_engulfing, is_inside_bar,
                           prior_close_breach, prior_high_breach, prior_low_breach

Usage:
  python3 rank_candle_metrics_shap.py              # all 12 pairs
  python3 rank_candle_metrics_shap.py --pair EUR_JPY
"""

import sys, os, gc, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OHLC_DIR = PROJECT_ROOT / "data" / "m5_ohlc"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PAIR_PIP = {
    "EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
    "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
}
PAIR_MIN_SWING = {
    "EUR_JPY": 30, "USD_JPY": 25, "GBP_JPY": 40, "AUD_JPY": 25,
    "CAD_JPY": 30, "CHF_JPY": 35, "NZD_JPY": 25,
    "EUR_USD": 20, "GBP_USD": 25, "AUD_USD": 18,
    "NZD_USD": 18, "EUR_GBP": 15,
}
ALL_PAIRS = list(PAIR_PIP.keys())

EPS = 1e-6
MIN_RANGE = 1e-8


# ── Vectorized candle metrics ──

def compute_candle_features(o, h, l, c):
    """Compute all candle metrics vectorized. Returns dict of arrays."""
    n = len(c)
    rng = np.maximum(h - l, MIN_RANGE)

    # Single-bar metrics
    body_size = np.abs(c - o)
    body_ratio = body_size / rng
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    upper_wick_ratio = upper_wick / rng
    lower_wick_ratio = lower_wick / rng
    wick_ratio = (upper_wick + lower_wick) / rng
    close_position = (c - l) / rng
    direction = np.sign(c - o)

    # Two-bar metrics (shifted by 1)
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    prev_o = np.roll(o, 1); prev_o[0] = o[0]
    prev_h = np.roll(h, 1); prev_h[0] = h[0]
    prev_l = np.roll(l, 1); prev_l[0] = l[0]
    prev_rng = np.maximum(prev_h - prev_l, MIN_RANGE)
    prev_direction = np.sign(prev_c - prev_o)

    # Range expansion
    range_expansion = rng / prev_rng

    # Two-bar momentum
    two_bar_momentum = (c - prev_o) / prev_rng

    # Gap (normalized by prev range)
    gap_norm = (o - prev_c) / prev_rng

    # Body overlap
    curr_body_lo = np.minimum(o, c)
    curr_body_hi = np.maximum(o, c)
    prev_body_lo = np.minimum(prev_o, prev_c)
    prev_body_hi = np.maximum(prev_o, prev_c)
    overlap = np.maximum(0, np.minimum(curr_body_hi, prev_body_hi) - np.maximum(curr_body_lo, prev_body_lo))
    curr_body_rng = np.maximum(curr_body_hi - curr_body_lo, MIN_RANGE)
    body_overlap = overlap / curr_body_rng

    # Categorical (as float for LightGBM)
    reversal_score = -(prev_direction * direction)
    is_engulfing = ((h > prev_h) & (l < prev_l)).astype(np.float64)
    is_inside_bar = ((h < prev_h) & (l > prev_l)).astype(np.float64)
    prior_close_breach = np.sign(c - prev_c)
    prior_high_breach = (h > prev_h).astype(np.float64)
    prior_low_breach = (l < prev_l).astype(np.float64)

    # Log deltas
    prev_body_size = np.abs(prev_c - prev_o)
    prev_body_ratio = prev_body_size / prev_rng
    prev_upper_wick_ratio = (prev_h - np.maximum(prev_o, prev_c)) / prev_rng
    prev_lower_wick_ratio = (np.minimum(prev_o, prev_c) - prev_l) / prev_rng
    prev_wick_ratio = (prev_upper_wick_ratio + prev_lower_wick_ratio)  # already ratios
    prev_close_position = (prev_c - prev_l) / prev_rng

    dlog_body_size = np.log(body_size + EPS) - np.log(prev_body_size + EPS)
    dlog_range_size = np.log(rng + EPS) - np.log(prev_rng + EPS)
    dlog_body_ratio = np.log(body_ratio + EPS) - np.log(prev_body_ratio + EPS)
    dlog_upper_wick_ratio = np.log(upper_wick_ratio + EPS) - np.log(prev_upper_wick_ratio + EPS)
    dlog_lower_wick_ratio = np.log(lower_wick_ratio + EPS) - np.log(prev_lower_wick_ratio + EPS)
    dlog_wick_ratio = np.log(wick_ratio + EPS) - np.log(prev_wick_ratio + EPS)
    dlog_close_position = np.log(close_position + EPS) - np.log(prev_close_position + EPS)

    features = {
        # Single-bar (5)
        "body_ratio": body_ratio,
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "wick_ratio": wick_ratio,
        "close_position": close_position,
        # Two-bar continuous (4)
        "range_expansion": range_expansion,
        "body_overlap": body_overlap,
        "two_bar_momentum": two_bar_momentum,
        "gap_norm": gap_norm,
        # Log deltas (7)
        "dlog_body_size": dlog_body_size,
        "dlog_range_size": dlog_range_size,
        "dlog_body_ratio": dlog_body_ratio,
        "dlog_upper_wick_ratio": dlog_upper_wick_ratio,
        "dlog_lower_wick_ratio": dlog_lower_wick_ratio,
        "dlog_wick_ratio": dlog_wick_ratio,
        "dlog_close_position": dlog_close_position,
        # Categorical as float (6)
        "direction": direction,
        "reversal_score": reversal_score,
        "is_engulfing": is_engulfing,
        "is_inside_bar": is_inside_bar,
        "prior_high_breach": prior_high_breach,
        "prior_low_breach": prior_low_breach,
    }
    return features


# ── Zigzag labels ──

@njit(cache=True)
def generate_zigzag_labels(mid_close, pip, min_swing_pips, label_window=6, min_mfe_pips=3.0):
    n = len(mid_close)
    labels = np.zeros(n, dtype=np.int64)
    min_swing = min_swing_pips * pip
    min_mfe = min_mfe_pips * pip
    running_high = mid_close[0]; running_low = mid_close[0]; direction = 0
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
                for j in range(i, min(i+200, n)):
                    dd = running_high - mid_close[j]
                    if dd > mfe: mfe = dd
                if mfe > min_mfe:
                    for k in range(i, min(i+label_window, n)): labels[k] = 2
                direction = -1; running_low = price
        else:
            if price - running_low >= min_swing:
                mfe = 0.0
                for j in range(i, min(i+200, n)):
                    uu = mid_close[j] - running_low
                    if uu > mfe: mfe = uu
                if mfe > min_mfe:
                    for k in range(i, min(i+label_window, n)): labels[k] = 1
                direction = 1; running_high = price
    return labels


# ── LightGBM + SHAP ──

def train_and_shap(X, y, feat_names):
    import lightgbm as lgb
    import shap

    s1 = int(len(X) * 0.6); s2 = int(len(X) * 0.8)
    X_train, X_val, X_test = X[:s1], X[s1:s2], X[s2:]
    y_train, y_val = y[:s1], y[s1:s2]

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feat_names, free_raw_data=False)
    valid_data = lgb.Dataset(X_val, label=y_val, feature_name=feat_names, free_raw_data=False)

    params = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
              "max_depth": 6, "learning_rate": 0.05, "num_leaves": 31,
              "min_child_samples": 50, "n_jobs": -1, "verbose": -1, "seed": 42}

    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[valid_data],
                      callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)])

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:5000])
    if isinstance(shap_values, list):
        shap_abs = np.zeros(len(feat_names))
        for sv in shap_values:
            shap_abs += np.abs(sv).mean(axis=0)
        shap_abs /= len(shap_values)
    else:
        shap_abs = np.abs(shap_values).mean(axis=(0, 2))

    return {name: float(shap_abs[i]) for i, name in enumerate(feat_names)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default=None)
    args = parser.parse_args()

    pairs = [args.pair] if args.pair else ALL_PAIRS
    all_results = {}

    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"  {pair} — Candle Metrics SHAP")
        print(f"{'='*60}")

        df = pd.read_parquet(OHLC_DIR / f"{pair}_M5.parquet")
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        c = df["close"].values.astype(np.float64)

        features = compute_candle_features(o, h, l, c)
        feat_names = sorted(features.keys())

        X = np.column_stack([features[f] for f in feat_names]).astype(np.float32)
        labels = generate_zigzag_labels(c, PAIR_PIP[pair], PAIR_MIN_SWING[pair], label_window=6)

        # Drop first 50 warmup bars + NaN/inf
        warmup = 50
        X = X[warmup:]; labels = labels[warmup:]
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]; labels = labels[mask]

        print(f"  Samples: {len(X):,}  Features: {len(feat_names)}")
        buy_pct = (labels == 1).mean() * 100
        sell_pct = (labels == 2).mean() * 100
        print(f"  Labels: FLAT={100-buy_pct-sell_pct:.1f}%  BUY={buy_pct:.1f}%  SELL={sell_pct:.1f}%")

        print("  Training LightGBM + SHAP...")
        shap_imp = train_and_shap(X, labels, feat_names)
        sorted_shap = sorted(shap_imp.items(), key=lambda x: -x[1])
        print("\n  ── SHAP Importance ──")
        for name, val in sorted_shap:
            print(f"    {name:25s}  {val:.6f}")

        all_results[pair] = shap_imp
        del X, labels, features; gc.collect()

    # Aggregate
    if len(pairs) > 1:
        print(f"\n{'='*60}")
        print(f"  AGGREGATE ({len(pairs)} pairs)")
        print(f"{'='*60}")

        all_feats = set()
        for r in all_results.values():
            all_feats.update(r.keys())

        agg = {}
        for f in sorted(all_feats):
            vals = [r.get(f, 0) for r in all_results.values()]
            agg[f] = np.mean(vals)

        sorted_agg = sorted(agg.items(), key=lambda x: -x[1])
        print("\n  ── Aggregate SHAP ──")
        for i, (name, val) in enumerate(sorted_agg):
            print(f"    {i+1:2d}. {name:25s}  {val:.6f}")

        out = {"per_pair": all_results, "aggregate": dict(sorted_agg)}
        out_path = RESULTS_DIR / "candle_metrics_shap.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
