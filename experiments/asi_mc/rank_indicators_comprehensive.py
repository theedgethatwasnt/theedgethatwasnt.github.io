#!/usr/bin/env python3
"""
Comprehensive SHAP Indicator Ranking — M5 OHLC
================================================
Computes ~30 indicators from proper M5 OHLC data + merges existing ASI-MC
parquet indicators. Ranks via LightGBM + SHAP on zigzag labels.

Data: 5 years M5 OHLC (2021-2026), ~393K bars per pair
Split: Train 60% | Validate 20% | Test 20%

Usage:
  python3 rank_indicators_comprehensive.py                # all 12 pairs
  python3 rank_indicators_comprehensive.py --pair EUR_JPY  # single pair
"""

import sys, os, gc, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OHLC_DIR = PROJECT_ROOT / "data" / "m5_ohlc"
ASI_DIR = PROJECT_ROOT / "data" / "asi_mc_indicators"
RESULTS_DIR = SCRIPT_DIR / "results" / "shap_comprehensive"
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

# ══════════════════════════════════════════════════════════════════
# Vectorized indicator computation from OHLC
# ══════════════════════════════════════════════════════════════════

def compute_rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
    rsi = 100 - 100 / (1 + rs)
    return rsi / 100.0  # normalize to [0, 1]


def compute_atr(high, low, close, period=14):
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    return atr


def compute_bb(close, period=20, std_mult=2.0):
    sma = pd.Series(close).rolling(period, min_periods=1).mean().values
    std = pd.Series(close).rolling(period, min_periods=1).std().values
    std = np.where(std > 0, std, 1e-10)
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_pos = (close - sma) / (std_mult * std)  # [-1, +1] roughly
    bb_width = (upper - lower) / np.where(sma > 0, sma, 1e-10)
    return bb_pos, bb_width


def compute_stochastic(high, low, close, k_period=14, d_period=3):
    lowest = pd.Series(low).rolling(k_period, min_periods=1).min().values
    highest = pd.Series(high).rolling(k_period, min_periods=1).max().values
    rng = highest - lowest
    rng = np.where(rng > 0, rng, 1e-10)
    k = (close - lowest) / rng  # [0, 1]
    d = pd.Series(k).rolling(d_period, min_periods=1).mean().values
    return k, d


def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    # Normalize by ATR-scale
    atr = compute_atr(np.maximum(close, np.roll(close, 1)),
                      np.minimum(close, np.roll(close, 1)), close, 14)
    atr = np.where(atr > 0, atr, 1e-10)
    return histogram / atr  # roughly [-3, +3]


def compute_adx(high, low, close, period=14):
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    atr = compute_atr(high, low, close, period)
    atr = np.where(atr > 0, atr, 1e-10)
    plus_di = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr * 100
    minus_di = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr * 100
    dx_sum = plus_di + minus_di
    dx_sum = np.where(dx_sum > 0, dx_sum, 1e-10)
    dx = np.abs(plus_di - minus_di) / dx_sum * 100
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    return adx / 100.0, (plus_di - minus_di) / 100.0  # adx [0,1], dmi_diff [-1,+1]


@njit(cache=True)
def compute_supertrend(high, low, close, period=10, multiplier=3.0):
    n = len(close)
    atr = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    tr[0] = high[0] - low[0]
    atr[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]

    direction = np.ones(n)  # +1 = bull, -1 = bear
    up_band = np.zeros(n)
    dn_band = np.zeros(n)
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2.0
        up_band[i] = mid + multiplier * atr[i]
        dn_band[i] = mid - multiplier * atr[i]
        if i > period:
            if dn_band[i] < dn_band[i-1] and close[i-1] > dn_band[i-1]:
                dn_band[i] = dn_band[i-1]
            if up_band[i] > up_band[i-1] and close[i-1] < up_band[i-1]:
                up_band[i] = up_band[i-1]
            if close[i] > up_band[i]:
                direction[i] = 1.0
            elif close[i] < dn_band[i]:
                direction[i] = -1.0
            else:
                direction[i] = direction[i-1]
    return direction


@njit(cache=True)
def compute_aroon_osc(high, low, period=25):
    n = len(high)
    out = np.zeros(n)
    for i in range(period, n):
        hh_idx = 0; ll_idx = 0
        hh_val = high[i - period]; ll_val = low[i - period]
        for j in range(1, period + 1):
            if high[i - period + j] >= hh_val:
                hh_val = high[i - period + j]; hh_idx = j
            if low[i - period + j] <= ll_val:
                ll_val = low[i - period + j]; ll_idx = j
        bars_hh = period - hh_idx; bars_ll = period - ll_idx
        out[i] = ((period - bars_hh) - (period - bars_ll)) / period
    return out


@njit(cache=True)
def compute_er_norm(closes, window=60):
    n = len(closes)
    out = np.zeros(n)
    half_pi = np.pi / 2.0
    for i in range(window, n):
        net = abs(closes[i] - closes[i - window])
        path = 0.0
        for j in range(i - window + 1, i + 1):
            path += abs(closes[j] - closes[j - 1])
        if path > 0:
            out[i] = np.arctan((net / path) / 0.3) / half_pi
    return out


@njit(cache=True)
def compute_range_position(high, low, period=30):
    n = len(high)
    out = np.zeros(n)
    for i in range(period, n):
        hh = high[i]; ll = low[i]
        for j in range(i - period, i):
            if high[j] > hh: hh = high[j]
            if low[j] < ll: ll = low[j]
        rng = hh - ll
        if rng > 0:
            out[i] = ((high[i] + low[i]) / 2.0 - ll) / rng  # [0, 1]
    return out


def compute_donchian_pos(high, low, close, period=20):
    hh = pd.Series(high).rolling(period, min_periods=1).max().values
    ll = pd.Series(low).rolling(period, min_periods=1).min().values
    rng = hh - ll
    rng = np.where(rng > 0, rng, 1e-10)
    return (close - ll) / rng * 2 - 1  # [-1, +1]


def compute_ema_ratio(close, period):
    ema = pd.Series(close).ewm(span=period, adjust=False).mean().values
    return close / np.where(ema > 0, ema, 1e-10) - 1.0


def compute_roc(close, period=10):
    prev = np.roll(close, period)
    prev[:period] = close[:period]
    return (close - prev) / np.where(prev > 0, prev, 1e-10)


def compute_vol_regime(atr):
    """ATR z-score as volatility regime: -1=low, 0=normal, +1=high."""
    mean = pd.Series(atr).rolling(200, min_periods=50).mean().values
    std = pd.Series(atr).rolling(200, min_periods=50).std().values
    std = np.where(std > 0, std, 1e-10)
    z = (atr - mean) / std
    return np.clip(z / 3.0, -1.0, 1.0)


def compute_squeeze(close, high, low, bb_period=20, kc_period=20, kc_mult=1.5):
    """Squeeze momentum: 1 = BB inside KC (squeeze on), -1 = BB outside."""
    sma = pd.Series(close).rolling(bb_period, min_periods=1).mean().values
    bb_std = pd.Series(close).rolling(bb_period, min_periods=1).std().values
    bb_upper = sma + 2 * bb_std
    bb_lower = sma - 2 * bb_std
    atr = compute_atr(high, low, close, kc_period)
    kc_upper = sma + kc_mult * atr
    kc_lower = sma - kc_mult * atr
    squeeze = np.where((bb_lower > kc_lower) & (bb_upper < kc_upper), 1.0, -1.0)
    return squeeze


def compute_hour_features(timestamps):
    hours = pd.to_datetime(timestamps).dt.hour
    hour_sin = np.sin(2 * np.pi * hours / 24).values
    hour_cos = np.cos(2 * np.pi * hours / 24).values
    return hour_sin, hour_cos


def compute_dow_features(timestamps):
    dow = pd.to_datetime(timestamps).dt.dayofweek
    dow_sin = np.sin(2 * np.pi * dow / 5).values
    dow_cos = np.cos(2 * np.pi * dow / 5).values
    return dow_sin, dow_cos


# ── Zigzag label generation ──

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
            if running_high - price >= min_swing:
                direction = -1; running_low = price
            elif price - running_low >= min_swing:
                direction = 1; running_high = price
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


# ══════════════════════════════════════════════════════════════════
# Data loading + feature computation
# ══════════════════════════════════════════════════════════════════

def load_pair(pair):
    """Load M5 OHLC + compute all indicators + merge ASI-MC indicators."""
    ohlc_path = OHLC_DIR / f"{pair}_M5.parquet"
    df = pd.read_parquet(ohlc_path)
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)

    features = {}

    # ── Momentum ──
    features["rsi_14"] = compute_rsi(c, 14)
    features["macd_hist"] = compute_macd(c)
    features["roc_10"] = compute_roc(c, 10)
    features["ema8_ratio"] = compute_ema_ratio(c, 8)
    features["ema21_ratio"] = compute_ema_ratio(c, 21)
    features["supertrend"] = compute_supertrend(h, l, c, 10, 3.0)

    # ── Volatility / Regime ──
    atr = compute_atr(h, l, c, 14)
    features["bb_pos"], features["bb_width"] = compute_bb(c, 20, 2.0)
    features["vol_regime"] = compute_vol_regime(atr)
    features["squeeze"] = compute_squeeze(c, h, l)
    features["atr_ratio"] = atr / np.where(pd.Series(atr).rolling(100, min_periods=1).mean().values > 0,
                                            pd.Series(atr).rolling(100, min_periods=1).mean().values, 1e-10)

    # ── Oscillators ──
    features["stoch_k"], features["stoch_d"] = compute_stochastic(h, l, c, 14, 3)
    features["aroon_osc"] = compute_aroon_osc(h, l, 25)

    # ── Trend ──
    features["adx"], features["dmi_diff"] = compute_adx(h, l, c, 14)
    features["er_norm"] = compute_er_norm(c, 60)
    features["donchian_pos"] = compute_donchian_pos(h, l, c, 20)
    features["range_pos_30"] = compute_range_position(h, l, 30)

    # ── Time ──
    features["hour_sin"], features["hour_cos"] = compute_hour_features(df["timestamp"])
    features["dow_sin"], features["dow_cos"] = compute_dow_features(df["timestamp"])

    # ── Body / structure ──
    pip = PAIR_PIP[pair]
    features["body_pips"] = np.abs(c - o) / pip
    features["candle_range"] = (h - l) / pip

    # ── Merge ASI-MC indicators from existing parquets ──
    asi_path = ASI_DIR / f"{pair}_asi_mc.parquet"
    if asi_path.exists():
        asi_df = pd.read_parquet(asi_path)
        # Align by timestamp
        df_ts = df["timestamp"].values
        asi_ts = asi_df["timestamp"].values
        idx = np.searchsorted(asi_ts, df_ts, side="right") - 1
        idx = np.clip(idx, 0, len(asi_ts) - 1)

        for col in ["mc_d_a", "mc_dd_a", "sb_a", "hh_asi", "hl_asi",
                     "h1_sr_zone", "str_diff_sign", "vol_regime"]:
            if col in asi_df.columns:
                vals = asi_df[col].values.astype(np.float64)
                mapped = vals[idx]
                # Only use where timestamps actually align (within 5 min)
                ts_diff = np.abs((pd.to_datetime(df_ts) - pd.to_datetime(asi_ts[idx])).total_seconds())
                mapped[ts_diff > 300] = 0.0
                feat_name = f"asi_{col}" if col in features else col
                features[feat_name] = mapped

    # Build feature matrix
    feat_names = sorted(features.keys())
    n = len(c)
    X = np.column_stack([features[f] for f in feat_names]).astype(np.float32)

    # Labels
    labels = generate_zigzag_labels(c, pip, PAIR_MIN_SWING[pair], label_window=6)

    # Drop warmup (500 bars)
    warmup = 500
    X = X[warmup:]
    labels = labels[warmup:]

    # Drop NaN rows
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    labels = labels[mask]

    return X, labels, feat_names


# ══════════════════════════════════════════════════════════════════
# LightGBM + SHAP
# ══════════════════════════════════════════════════════════════════

def train_lgbm(X, y, feature_names):
    import lightgbm as lgb

    # 60/20/20 split
    n = len(X)
    s1 = int(n * 0.6)
    s2 = int(n * 0.8)
    X_train, X_val, X_test = X[:s1], X[s1:s2], X[s2:]
    y_train, y_val, y_test = y[:s1], y[s1:s2], y[s2:]

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    valid_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, free_raw_data=False)

    params = {
        "objective": "multiclass", "num_class": 3,
        "metric": "multi_logloss", "max_depth": 6,
        "learning_rate": 0.05, "num_leaves": 31,
        "min_child_samples": 50, "n_jobs": -1,
        "verbose": -1, "seed": 42,
    }

    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[valid_data],
                      callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)])

    gain = model.feature_importance(importance_type="gain")
    total = gain.sum()
    gain_norm = gain / total if total > 0 else gain
    importance = {name: {"gain_pct": float(gain_norm[i] * 100)} for i, name in enumerate(feature_names)}

    return model, importance, X_test, y_test


def compute_shap(model, X_test, feature_names):
    import shap
    explainer = shap.TreeExplainer(model)
    sample = X_test[:5000]
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_abs = np.zeros(len(feature_names))
        for sv in shap_values:
            shap_abs += np.abs(sv).mean(axis=0)
        shap_abs /= len(shap_values)
    else:
        shap_abs = np.abs(shap_values).mean(axis=(0, 2))

    return {name: float(shap_abs[i]) for i, name in enumerate(feature_names)}


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default=None)
    args = parser.parse_args()

    pairs = [args.pair] if args.pair else ALL_PAIRS
    all_results = {}

    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"  {pair}")
        print(f"{'='*60}")

        X, y, feat_names = load_pair(pair)
        print(f"  Samples: {len(X):,}  Features: {len(feat_names)}")
        buy_pct = (y == 1).mean() * 100
        sell_pct = (y == 2).mean() * 100
        print(f"  Labels: FLAT={100-buy_pct-sell_pct:.1f}%  BUY={buy_pct:.1f}%  SELL={sell_pct:.1f}%")

        print("  Training LightGBM...")
        model, gain_imp, X_test, y_test = train_lgbm(X, y, feat_names)

        # Gain importance (top 15)
        sorted_gain = sorted(gain_imp.items(), key=lambda x: -x[1]["gain_pct"])
        print("\n  ── Gain Importance (top 15) ──")
        for name, vals in sorted_gain[:15]:
            print(f"    {name:20s}  {vals['gain_pct']:6.2f}%")

        # SHAP
        print("\n  Computing SHAP values...")
        shap_imp = compute_shap(model, X_test, feat_names)
        sorted_shap = sorted(shap_imp.items(), key=lambda x: -x[1])
        print("\n  ── SHAP Importance (top 15) ──")
        for name, val in sorted_shap[:15]:
            print(f"    {name:20s}  {val:.6f}")

        all_results[pair] = {
            "n_samples": len(X),
            "n_features": len(feat_names),
            "gain_importance": gain_imp,
            "shap_importance": shap_imp,
        }
        del X, y, X_test, model; gc.collect()

    # ── Aggregate ──
    if len(pairs) > 1:
        print(f"\n{'='*60}")
        print(f"  AGGREGATE (mean across {len(pairs)} pairs)")
        print(f"{'='*60}")

        all_feats = set()
        for r in all_results.values():
            all_feats.update(r["shap_importance"].keys())

        agg_shap = {}
        for feat in sorted(all_feats):
            vals = [r["shap_importance"].get(feat, 0) for r in all_results.values()]
            agg_shap[feat] = np.mean(vals)

        sorted_agg = sorted(agg_shap.items(), key=lambda x: -x[1])
        print("\n  ── Aggregate SHAP Importance ──")
        for i, (name, val) in enumerate(sorted_agg):
            rank = i + 1
            print(f"    {rank:2d}. {name:20s}  {val:.6f}")

        # Save
        out = {
            "per_pair": all_results,
            "aggregate_shap": dict(sorted_agg),
            "feature_list": sorted(list(all_feats)),
        }
        out_path = RESULTS_DIR / "comprehensive_ranking.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
