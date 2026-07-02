#!/usr/bin/env python3
"""
SHAP Ranking — Remaining Untested Indicators
=============================================
Tests indicators from IronNet Training Roadmap NOT covered by the comprehensive
SHAP study (33 features) or candle metrics study (22 features).

New indicators (7):
  TEC       — Signed Kaufman Efficiency (5-bar), arctan-normalized
  dTEC      — Trend acceleration (TEC_13 - TEC_3), arctan-normalized
  SB_P      — Price breakout position from TopsBots {-1,-0.5,0,+0.5,+1}
  hh_price  — Higher-highs on price (binary)
  hl_price  — Higher-lows on price (binary)
  H1_slope  — H1 linear regression slope (3-bar), arctan-normalized
  MTF_align — H1×M5 swing alignment {-2,-1,+1,+2}

Also re-tests existing indicators for completeness in combined ranking.
"""

import sys, os, gc, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.asi_indicator import compute_asi, sma_jit
from lib.swing_indicators import compute_all_swing_features

OHLC_DIR = PROJECT_ROOT / "data" / "m5_ohlc"
RESULTS_DIR = SCRIPT_DIR / "results" / "shap_remaining"
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


# ── New indicator computations ──

@njit(cache=True)
def compute_tec(closes, window=5):
    """Signed Kaufman Efficiency Ratio. Range: arctan[-1,+1]."""
    n = len(closes)
    out = np.zeros(n)
    half_pi = np.pi / 2.0
    for i in range(window, n):
        net = closes[i] - closes[i - window]  # signed
        path = 0.0
        for j in range(i - window + 1, i + 1):
            path += abs(closes[j] - closes[j - 1])
        if path > 0:
            raw = net / path  # [-1, +1]
            out[i] = np.arctan(raw / 0.3) / half_pi
    return out


@njit(cache=True)
def compute_dtec(closes, long_w=13, short_w=3):
    """Trend acceleration: TEC_long - TEC_short, arctan-normalized."""
    n = len(closes)
    tec_long = np.zeros(n)
    tec_short = np.zeros(n)
    half_pi = np.pi / 2.0

    for i in range(long_w, n):
        # Long TEC
        net_l = closes[i] - closes[i - long_w]
        path_l = 0.0
        for j in range(i - long_w + 1, i + 1):
            path_l += abs(closes[j] - closes[j - 1])
        if path_l > 0:
            tec_long[i] = net_l / path_l

        # Short TEC
        if i >= short_w:
            net_s = closes[i] - closes[i - short_w]
            path_s = 0.0
            for j in range(i - short_w + 1, i + 1):
                path_s += abs(closes[j] - closes[j - 1])
            if path_s > 0:
                tec_short[i] = net_s / path_s

    dtec = tec_long - tec_short
    out = np.zeros(n)
    for i in range(n):
        out[i] = np.arctan(dtec[i] * 5.0) / half_pi
    return out


@njit(cache=True)
def compute_h1_slope(closes, bar_period=12, slope_bars=3):
    """H1 slope from M5 data: linreg slope on last 3 H1 bars, arctan-normalized.
    bar_period=12 means 12 M5 bars = 1 H1 bar.
    """
    n = len(closes)
    out = np.zeros(n)
    half_pi = np.pi / 2.0
    lookback = slope_bars * bar_period  # 36 M5 bars = 3 H1 bars

    for i in range(lookback, n):
        # Sample 3 H1 closes: i, i-12, i-24
        vals = np.zeros(slope_bars)
        for k in range(slope_bars):
            idx = i - k * bar_period
            vals[slope_bars - 1 - k] = closes[idx]

        # Linear regression slope
        x_mean = (slope_bars - 1) / 2.0
        y_mean = 0.0
        for k in range(slope_bars):
            y_mean += vals[k]
        y_mean /= slope_bars

        num = 0.0
        den = 0.0
        for k in range(slope_bars):
            xd = k - x_mean
            num += xd * (vals[k] - y_mean)
            den += xd * xd

        if den > 0:
            slope = num / den
            # Normalize by local ATR proxy (range of the 3 H1 closes)
            rng = vals.max() - vals.min()
            if rng > 0:
                norm_slope = slope / rng
            else:
                norm_slope = 0.0
            out[i] = np.arctan(norm_slope * 3.0) / half_pi
    return out


def compute_mtf_align(sb_a_m5, h1_slope):
    """MTF alignment: combine M5 swing state with H1 slope direction.
    +2 = STRONG_BULL (both bullish)
    +1 = RALLY (H1 up, M5 not confirming)
    -1 = PULLBACK (H1 down, M5 not confirming)
    -2 = STRONG_BEAR (both bearish)
    """
    n = len(sb_a_m5)
    out = np.zeros(n)
    for i in range(n):
        h1_dir = 1 if h1_slope[i] > 0.1 else (-1 if h1_slope[i] < -0.1 else 0)
        m5_dir = 1 if sb_a_m5[i] > 0 else (-1 if sb_a_m5[i] < 0 else 0)

        if h1_dir > 0 and m5_dir > 0:
            out[i] = 2.0
        elif h1_dir < 0 and m5_dir < 0:
            out[i] = -2.0
        elif h1_dir > 0:
            out[i] = 1.0
        elif h1_dir < 0:
            out[i] = -1.0
    return out


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
                    if running_high - mid_close[j] > mfe: mfe = running_high - mid_close[j]
                if mfe > min_mfe:
                    for k in range(i, min(i+label_window, n)): labels[k] = 2
                direction = -1; running_low = price
        else:
            if price - running_low >= min_swing:
                mfe = 0.0
                for j in range(i, min(i+200, n)):
                    if mid_close[j] - running_low > mfe: mfe = mid_close[j] - running_low
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


def load_pair(pair):
    df = pd.read_parquet(OHLC_DIR / f"{pair}_M5.parquet")
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    n = len(c)

    print(f"  Computing TEC/dTEC...", flush=True)
    tec = compute_tec(c, 5)
    dtec = compute_dtec(c, 13, 3)

    print(f"  Computing H1_slope...", flush=True)
    h1_slope = compute_h1_slope(c, 12, 3)

    print(f"  Computing swing features (TopsBots)...", flush=True)
    asi = compute_asi(o, h, l, c, n)
    swing = compute_all_swing_features(o, h, l, c, asi)

    sb_p = swing["sb_p"].astype(np.float64)
    hh_price = swing["hh_price"].astype(np.float64)
    hl_price = swing["hl_price"].astype(np.float64)
    sb_a = swing["sb_a"].astype(np.float64)

    print(f"  Computing MTF_align...", flush=True)
    mtf_align = compute_mtf_align(sb_a, h1_slope)

    features = {
        "tec_5": tec,
        "dtec_13_3": dtec,
        "sb_p": sb_p,
        "hh_price": hh_price,
        "hl_price": hl_price,
        "h1_slope": h1_slope,
        "mtf_align": mtf_align,
    }

    feat_names = sorted(features.keys())
    X = np.column_stack([features[f] for f in feat_names]).astype(np.float32)
    labels = generate_zigzag_labels(c, PAIR_PIP[pair], PAIR_MIN_SWING[pair], label_window=6)

    warmup = 500
    X = X[warmup:]; labels = labels[warmup:]
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]; labels = labels[mask]

    return X, labels, feat_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default=None)
    args = parser.parse_args()

    pairs = [args.pair] if args.pair else ALL_PAIRS
    all_results = {}

    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"  {pair} — Remaining Indicators SHAP")
        print(f"{'='*60}")

        X, y, feat_names = load_pair(pair)
        print(f"  Samples: {len(X):,}  Features: {len(feat_names)}")
        print(f"  Training LightGBM + SHAP...")
        shap_imp = train_and_shap(X, y, feat_names)

        sorted_shap = sorted(shap_imp.items(), key=lambda x: -x[1])
        print(f"\n  ── SHAP Importance ──")
        for name, val in sorted_shap:
            print(f"    {name:15s}  {val:.6f}")

        all_results[pair] = shap_imp
        del X, y; gc.collect()

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
        print(f"\n  ── Aggregate SHAP ──")
        for i, (name, val) in enumerate(sorted_agg):
            print(f"    {i+1:2d}. {name:15s}  {val:.6f}")

        out = {"per_pair": all_results, "aggregate": dict(sorted_agg)}
        out_path = RESULTS_DIR / "remaining_indicators_shap.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
