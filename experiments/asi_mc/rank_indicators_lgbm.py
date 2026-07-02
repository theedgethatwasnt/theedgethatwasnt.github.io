#!/usr/bin/env python3
"""
LightGBM Feature Importance Ranker for IronNet Indicators
==========================================================
Trains LightGBM multiclass on zigzag labels (BUY/SELL/FLATTEN) using all 11
candidate indicators. Outputs:
  1. Individual feature importance (gain + SHAP)
  2. SHAP interaction matrix → ranked pairs (55 combos)
  3. Top triplet recommendations (from pair interactions)

Uses existing M5 indicator parquets + zigzag label generation from
train_zigzag_perpair.py.

Usage:
  python3 rank_indicators_lgbm.py              # all 12 pairs
  python3 rank_indicators_lgbm.py --pair EUR_GBP  # single pair
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path(os.environ.get("ASI_MC_DATA_DIR",
                str(PROJECT_ROOT / "data" / "asi_mc_indicators")))
RESULTS_DIR = SCRIPT_DIR / "results" / "indicator_ranking"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── The 12 indicators (7 active + 4 dropped + aroon_osc) ──
FEATURE_COLS = [
    "mc_d_a",     # א — ASI momentum direction
    "mc_dd_a",    # ב — ASI momentum acceleration
    "er_norm",    # ג — Kaufman ER
    "sb_a",       # ד — ASI swing breakout
    "hh_asi",     # ה — ASI higher-high
    "hl_asi",     # ו — ASI higher-low
    # UPnL (ז) is trade-state, not a market indicator — excluded from ranking
    # (always included as input to NEAT, not a feature to select)
    "erp_p",      # ח — price range position (unbounded)
    "erp_a",      # ט — ASI range position (unbounded)
    "d_erp_p",    # י — 1h delta of erp_p (unbounded)
    "d_erp_a",    # כ — 1h delta of erp_a (unbounded)
    "aroon_osc",  # ל — Aroon Oscillator (Up - Down) / 100
]

# Hebrew labels for display
HEBREW = {
    "mc_d_a": "א", "mc_dd_a": "ב", "er_norm": "ג", "sb_a": "ד",
    "hh_asi": "ה", "hl_asi": "ו",
    "erp_p": "ח", "erp_a": "ט", "d_erp_p": "י", "d_erp_a": "כ",
    "aroon_osc": "ל",
}

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

# ── Zigzag label generation (copied from train_zigzag_perpair.py) ──

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
        if price > running_high:
            running_high = price
        if price < running_low:
            running_low = price

        if direction == 0:
            if running_high - price >= min_swing:
                direction = -1
                running_low = price
            elif price - running_low >= min_swing:
                direction = 1
                running_high = price
        elif direction == 1:
            if running_high - price >= min_swing:
                mfe = 0.0
                for j in range(i, min(i + 200, n)):
                    dd = running_high - mid_close[j]
                    if dd > mfe:
                        mfe = dd
                if mfe > min_mfe:
                    end = min(i + label_window, n)
                    for k in range(i, end):
                        labels[k] = 2
                direction = -1
                running_low = price
        else:
            if price - running_low >= min_swing:
                mfe = 0.0
                for j in range(i, min(i + 200, n)):
                    uu = mid_close[j] - running_low
                    if uu > mfe:
                        mfe = uu
                if mfe > min_mfe:
                    end = min(i + label_window, n)
                    for k in range(i, end):
                        labels[k] = 1
                direction = 1
                running_high = price

    return labels


@njit(cache=True)
def compute_aroon_osc(highs, lows, period=25):
    """Aroon Oscillator = (Aroon_Up - Aroon_Down) / 100.
    Aroon_Up = (period - bars_since_HH) / period * 100
    Aroon_Down = (period - bars_since_LL) / period * 100
    Result in [-1, +1].
    """
    n = len(highs)
    out = np.full(n, np.nan)
    for i in range(period, n):
        hh_idx = 0
        ll_idx = 0
        hh_val = highs[i - period]
        ll_val = lows[i - period]
        for j in range(1, period + 1):
            if highs[i - period + j] >= hh_val:
                hh_val = highs[i - period + j]
                hh_idx = j
            if lows[i - period + j] <= ll_val:
                ll_val = lows[i - period + j]
                ll_idx = j
        bars_since_hh = period - hh_idx
        bars_since_ll = period - ll_idx
        aroon_up = (period - bars_since_hh) / period * 100.0
        aroon_down = (period - bars_since_ll) / period * 100.0
        out[i] = (aroon_up - aroon_down) / 100.0
    return out


@njit(cache=True)
def _compute_er_norm_simple(closes, window=60):
    """Kaufman ER, arctan-normalized. Inline for S5 loading."""
    n = len(closes)
    out = np.full(n, np.nan)
    for i in range(window, n):
        net_chg = abs(closes[i] - closes[i - window])
        sum_chg = 0.0
        for j in range(i - window + 1, i + 1):
            sum_chg += abs(closes[j] - closes[j - 1])
        raw = net_chg / sum_chg if sum_chg > 0 else 0.0
        out[i] = np.arctan(raw / 0.3) / (np.pi / 2)
    return out


S5_DATA_DIR = Path(os.environ.get("S5_DATA_DIR", ""))


def resample_to_tf(df, tf):
    """Resample M5 indicator dataframe to H1 or keep as M5."""
    if tf == "M5":
        return df
    elif tf == "H1":
        df = df.set_index("timestamp")
        # mid_close: take last value per hour
        agg = {"mid_close": "last"}
        # For indicator columns: take last (forward-filled values)
        for col in FEATURE_COLS:
            if col in df.columns:
                agg[col] = "last"
        # Need high/low for Aroon at H1
        if "mid_high" in df.columns:
            agg["mid_high"] = "max"
            agg["mid_low"] = "min"
        resampled = df.resample("1h").agg(agg).dropna(subset=["mid_close"])
        resampled = resampled.reset_index()
        return resampled
    else:
        raise ValueError(f"Unknown timeframe: {tf}")


# Zigzag swing sizes scale with timeframe
# H1 bars are 12× M5 bars, so swings need to be larger
H1_MIN_SWING_MULT = 2.0  # 2× M5 swing size for H1


def load_s5_pair(pair):
    """Load S5 BA parquet, compute M5 indicators, return at S5 cadence."""
    from lib.asi_indicator import compute_asi_mc

    # Find S5 file
    s5_dir = S5_DATA_DIR if S5_DATA_DIR and S5_DATA_DIR.exists() else DATA_DIR.parent / "scalper_parquet"
    candidates = [
        s5_dir / f"{pair.replace('_','')}_S5_BA.parquet",
        s5_dir / f"{pair}_S5_BA.parquet",
        s5_dir / f"{pair.replace('_','')}_S5.parquet",
    ]
    s5_path = None
    for c in candidates:
        if c.exists():
            s5_path = c; break
    if s5_path is None:
        raise FileNotFoundError(f"No S5 parquet for {pair} in {s5_dir}")

    print(f"  Loading S5: {s5_path.name}")
    s5 = pd.read_parquet(s5_path, engine="pyarrow")

    # Need bid/ask → mid
    if "bid_c" in s5.columns and "ask_c" in s5.columns:
        s5["mid_close"] = (s5["bid_c"] + s5["ask_c"]) / 2
        s5["mid_high"] = (s5["bid_h"] + s5["ask_h"]) / 2
        s5["mid_low"] = (s5["bid_l"] + s5["ask_l"]) / 2
        s5["mid_open"] = (s5["bid_o"] + s5["ask_o"]) / 2
    elif "close" in s5.columns:
        s5["mid_close"] = s5["close"]
        s5["mid_high"] = s5["high"]
        s5["mid_low"] = s5["low"]
        s5["mid_open"] = s5["open"]

    # Resample to M5 for indicator computation
    if "time" in s5.columns:
        s5 = s5.rename(columns={"time": "timestamp"})
    s5 = s5.set_index("timestamp")
    m5 = s5.resample("5min").agg({
        "mid_open": "first", "mid_high": "max", "mid_low": "min", "mid_close": "last"
    }).dropna()

    # Compute indicators at M5
    o = m5["mid_open"].values.astype(np.float64)
    h = m5["mid_high"].values.astype(np.float64)
    l = m5["mid_low"].values.astype(np.float64)
    c = m5["mid_close"].values.astype(np.float64)
    n_m5 = len(c)

    mc_d, mc_dd = compute_asi_mc(o, h, l, c, n_m5)

    er = _compute_er_norm_simple(c, 60)

    # Aroon on M5 highs/lows
    aroon = compute_aroon_osc(h, l, period=25)

    # Forward-fill M5 indicators to S5
    # Create M5 bar index for each S5 bar
    s5_reset = s5.reset_index()
    m5_reset = m5.reset_index()
    m5_ts = m5_reset["timestamp"].values
    s5_ts = s5_reset["timestamp"].values

    # Map each S5 bar to its M5 bar via searchsorted
    m5_idx = np.searchsorted(m5_ts, s5_ts, side="right") - 1
    m5_idx = np.clip(m5_idx, 0, n_m5 - 1)

    # Build S5-cadence dataframe
    result = pd.DataFrame({
        "timestamp": s5_reset["timestamp"],
        "mid_close": s5_reset["mid_close"].values,
        "mid_high": s5_reset["mid_high"].values,
        "mid_low": s5_reset["mid_low"].values,
        "mc_d_a": mc_d[m5_idx],
        "mc_dd_a": mc_dd[m5_idx],
        "er_norm": er[m5_idx],
        "aroon_osc": aroon[m5_idx],
    })

    # SwingStructure indicators need the full ASI pipeline — load from M5 parquet if available
    m5_path = DATA_DIR / f"{pair}_asi_mc.parquet"
    if m5_path.exists():
        m5_ind = pd.read_parquet(m5_path, engine="pyarrow")
        for col in ["sb_a", "hh_asi", "hl_asi", "erp_p", "erp_a", "d_erp_p", "d_erp_a"]:
            if col in m5_ind.columns:
                vals = m5_ind[col].values
                # Map via timestamp alignment
                m5_ind_ts = m5_ind["timestamp"].values
                idx = np.searchsorted(m5_ind_ts, s5_ts, side="right") - 1
                idx = np.clip(idx, 0, len(vals) - 1)
                result[col] = vals[idx]

    return result


def load_pair(pair, tf="M5"):
    """Load indicator parquet + generate zigzag labels."""
    if tf == "S5":
        df = load_s5_pair(pair)
    else:
        path = DATA_DIR / f"{pair}_asi_mc.parquet"
        df = pd.read_parquet(path)

        # Resample if needed
        df = resample_to_tf(df, tf)

    # Compute Aroon if not present (M5/H1 from existing OHLC-proxy)
    if "aroon_osc" not in df.columns:
        if "mid_high" not in df.columns:
            # Approximate high/low from mid_close ± half-spread
            df["mid_high"] = df["mid_close"]
            df["mid_low"] = df["mid_close"]
        aroon = compute_aroon_osc(
            df["mid_high"].values.astype(np.float64),
            df["mid_low"].values.astype(np.float64),
            period=25
        )
        df["aroon_osc"] = aroon

    # Check all feature cols exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: {pair} missing columns: {missing}")

    # Generate labels — adjust swing size for timeframe
    mid = df["mid_close"].values.astype(np.float64)
    pip = PAIR_PIP[pair]
    min_swing = PAIR_MIN_SWING[pair]
    if tf == "H1":
        min_swing = int(min_swing * H1_MIN_SWING_MULT)
        label_window = 3
    elif tf == "S5":
        # Same swing as M5 but label_window scales: 6 M5 bars = 72 S5 bars
        label_window = 72
    else:
        label_window = 6
    labels = generate_zigzag_labels(mid, pip, min_swing, label_window=label_window)

    # Drop warmup
    warmup = {"S5": 6000, "M5": 500, "H1": 50}[tf]
    df = df.iloc[warmup:].reset_index(drop=True)
    labels = labels[warmup:]

    # Extract features
    avail_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[avail_cols].values.astype(np.float32)
    y = labels.astype(np.int32)

    # Drop NaN rows
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    return X, y, avail_cols


def train_lgbm(X, y, feature_names):
    """Train LightGBM multiclass on zigzag labels. Return model + importance."""
    import lightgbm as lgb

    # Chronological 70/30 split
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                             free_raw_data=False)
    valid_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_names,
                             free_raw_data=False)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "n_jobs": -1,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[valid_data],
        callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(30, verbose=False)],
    )

    # Feature importance (gain)
    gain = model.feature_importance(importance_type="gain")
    total = gain.sum()
    gain_norm = gain / total if total > 0 else gain

    importance = {}
    for i, name in enumerate(feature_names):
        importance[name] = {
            "gain_raw": float(gain[i]),
            "gain_pct": float(gain_norm[i] * 100),
        }

    # Label distribution
    from collections import Counter
    train_dist = Counter(y_train.tolist())
    test_dist = Counter(y_test.tolist())

    return model, importance, {
        "n_train": len(X_train), "n_test": len(X_test),
        "best_iteration": model.best_iteration,
        "train_labels": {str(k): v for k, v in train_dist.items()},
        "test_labels": {str(k): v for k, v in test_dist.items()},
    }


def compute_shap(model, X_test, feature_names):
    """Compute SHAP values + interaction values."""
    import shap

    explainer = shap.TreeExplainer(model)

    # SHAP values — shape (n_samples, n_features, n_classes) for multiclass
    shap_values = explainer.shap_values(X_test[:5000])  # cap at 5K for speed

    # Mean absolute SHAP per feature (averaged over classes)
    if isinstance(shap_values, list):
        # list of (n_samples, n_features) per class
        shap_abs = np.zeros(len(feature_names))
        for sv in shap_values:
            shap_abs += np.abs(sv).mean(axis=0)
        shap_abs /= len(shap_values)
    else:
        shap_abs = np.abs(shap_values).mean(axis=(0, 2))

    shap_importance = {}
    for i, name in enumerate(feature_names):
        shap_importance[name] = float(shap_abs[i])

    # SHAP interaction values (expensive but gives pair importance)
    print("  Computing SHAP interactions (this takes ~1 min)...")
    try:
        shap_interaction = explainer.shap_interaction_values(X_test[:2000])

        # Average absolute interaction across samples and classes
        if isinstance(shap_interaction, list):
            inter_abs = np.zeros((len(feature_names), len(feature_names)))
            for si in shap_interaction:
                inter_abs += np.abs(si).mean(axis=0)
            inter_abs /= len(shap_interaction)
        else:
            inter_abs = np.abs(shap_interaction).mean(axis=0)

        # Build pair interaction dict
        pair_interactions = {}
        for i, j in combinations(range(len(feature_names)), 2):
            key = f"{feature_names[i]}+{feature_names[j]}"
            # Symmetric: average both directions
            val = float((inter_abs[i, j] + inter_abs[j, i]) / 2)
            pair_interactions[key] = val

    except Exception as e:
        print(f"  SHAP interactions failed: {e}")
        pair_interactions = {}

    return shap_importance, pair_interactions


def rank_triplets(pair_interactions, feature_names):
    """Rank 120 triplets by sum of their 3 pairwise interactions."""
    triplet_scores = {}
    for a, b, c in combinations(feature_names, 3):
        ab = pair_interactions.get(f"{a}+{b}", 0) or pair_interactions.get(f"{b}+{a}", 0)
        ac = pair_interactions.get(f"{a}+{c}", 0) or pair_interactions.get(f"{c}+{a}", 0)
        bc = pair_interactions.get(f"{b}+{c}", 0) or pair_interactions.get(f"{c}+{b}", 0)
        triplet_scores[f"{a}+{b}+{c}"] = ab + ac + bc
    return dict(sorted(triplet_scores.items(), key=lambda x: -x[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default=None, help="Single pair (default: all 12)")
    parser.add_argument("--tf", type=str, default="M5", choices=["M5", "H1", "S5"],
                        help="Timeframe: M5 (default), H1, or S5")
    args = parser.parse_args()

    pairs = [args.pair] if args.pair else ALL_PAIRS
    tf = args.tf
    print(f"Timeframe: {tf}")
    RESULTS_DIR_TF = RESULTS_DIR / tf
    RESULTS_DIR_TF.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"  {pair}")
        print(f"{'='*60}")

        X, y, feat_names = load_pair(pair, tf=tf)
        print(f"  Samples: {len(X):,}  Features: {len(feat_names)}")
        buy_pct = (y == 1).mean() * 100
        sell_pct = (y == 2).mean() * 100
        flat_pct = (y == 0).mean() * 100
        print(f"  Labels: FLATTEN={flat_pct:.1f}%  BUY={buy_pct:.1f}%  SELL={sell_pct:.1f}%")

        # Train LightGBM
        print("  Training LightGBM...")
        model, gain_importance, meta = train_lgbm(X, y, feat_names)
        print(f"  Best iteration: {meta['best_iteration']}")

        # Gain importance
        sorted_gain = sorted(gain_importance.items(), key=lambda x: -x[1]["gain_pct"])
        print("\n  ── Gain Importance ──")
        for name, vals in sorted_gain:
            heb = HEBREW.get(name, "?")
            print(f"    {heb} {name:12s}  {vals['gain_pct']:6.1f}%")

        # SHAP importance + interactions
        split = int(len(X) * 0.7)
        X_test = X[split:]
        print("\n  Computing SHAP values...")
        shap_imp, pair_inter = compute_shap(model, X_test, feat_names)

        sorted_shap = sorted(shap_imp.items(), key=lambda x: -x[1])
        print("\n  ── SHAP Importance ──")
        for name, val in sorted_shap:
            heb = HEBREW.get(name, "?")
            print(f"    {heb} {name:12s}  {val:.6f}")

        # Pair interactions
        if pair_inter:
            sorted_pairs = sorted(pair_inter.items(), key=lambda x: -x[1])
            print(f"\n  ── Top 15 Pairs (of {len(sorted_pairs)}) ──")
            for i, (pname, val) in enumerate(sorted_pairs[:15]):
                parts = pname.split("+")
                heb_label = "".join(HEBREW.get(p, "?") for p in parts)
                print(f"    {i+1:2d}. {heb_label}  {pname:30s}  {val:.6f}")

            # Triplet ranking
            triplets = rank_triplets(pair_inter, feat_names)
            sorted_triplets = list(triplets.items())
            print(f"\n  ── Top 20 Triplets (of {len(sorted_triplets)}) ──")
            for i, (tname, val) in enumerate(sorted_triplets[:20]):
                parts = tname.split("+")
                heb_label = "".join(HEBREW.get(p, "?") for p in parts)
                print(f"    {i+1:2d}. {heb_label}  {tname:45s}  {val:.6f}")

        all_results[pair] = {
            "meta": meta,
            "gain_importance": gain_importance,
            "shap_importance": shap_imp,
            "pair_interactions": pair_inter,
            "triplet_ranking": rank_triplets(pair_inter, feat_names) if pair_inter else {},
        }

    # ── Aggregate across all pairs ──
    if len(pairs) > 1:
        print(f"\n{'='*60}")
        print(f"  AGGREGATE (mean across {len(pairs)} pairs)")
        print(f"{'='*60}")

        # Aggregate SHAP importance
        agg_shap = {}
        for feat in FEATURE_COLS:
            vals = [r["shap_importance"].get(feat, 0) for r in all_results.values()]
            agg_shap[feat] = np.mean(vals)
        sorted_agg = sorted(agg_shap.items(), key=lambda x: -x[1])
        print("\n  ── Aggregate SHAP Importance ──")
        for name, val in sorted_agg:
            heb = HEBREW.get(name, "?")
            print(f"    {heb} {name:12s}  {val:.6f}")

        # Aggregate pair interactions
        agg_pairs = {}
        all_pair_keys = set()
        for r in all_results.values():
            all_pair_keys.update(r["pair_interactions"].keys())
        for pk in all_pair_keys:
            vals = [r["pair_interactions"].get(pk, 0) for r in all_results.values()]
            agg_pairs[pk] = np.mean(vals)
        sorted_agg_pairs = sorted(agg_pairs.items(), key=lambda x: -x[1])
        print(f"\n  ── Top 15 Aggregate Pairs ──")
        for i, (pname, val) in enumerate(sorted_agg_pairs[:15]):
            parts = pname.split("+")
            heb_label = "".join(HEBREW.get(p, "?") for p in parts)
            print(f"    {i+1:2d}. {heb_label}  {pname:30s}  {val:.6f}")

        # Aggregate triplets
        agg_triplets = {}
        all_trip_keys = set()
        for r in all_results.values():
            all_trip_keys.update(r["triplet_ranking"].keys())
        for tk in all_trip_keys:
            vals = [r["triplet_ranking"].get(tk, 0) for r in all_results.values()]
            agg_triplets[tk] = np.mean(vals)
        sorted_agg_trips = sorted(agg_triplets.items(), key=lambda x: -x[1])
        print(f"\n  ── Top 20 Aggregate Triplets ──")
        for i, (tname, val) in enumerate(sorted_agg_trips[:20]):
            parts = tname.split("+")
            heb_label = "".join(HEBREW.get(p, "?") for p in parts)
            print(f"    {i+1:2d}. {heb_label}  {tname:45s}  {val:.6f}")

        all_results["aggregate"] = {
            "shap_importance": agg_shap,
            "pair_interactions": agg_pairs,
            "triplet_ranking": agg_triplets,
        }

    # Save results
    out_path = RESULTS_DIR_TF / "indicator_ranking.json"
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
