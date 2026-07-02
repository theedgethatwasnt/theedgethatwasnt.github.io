#!/usr/bin/env python3
"""
LightGBM feature study for SB_A IronNet range-bar inputs.

Tests the 3 bar-level IronNet features on 10-pip range bars:
  1. sba     — IncrementalTopsBots swing state, normalized to {-1,-0.5,0,+0.5,+1}
  2. mc_d    — ASIMC momentum consistency direction
  3. mc_dd   — ASIMC momentum consistency 2nd derivative

Note: upnl_n / mae_n are position-state signals (computed live inside a trade).
They have no meaning as standalone bar features and are excluded here.

Target: sign(forward_pip_return) at horizons [5, 10, 20, 50] range bars.
Protocol: 70/30 IS/OOS, OOS AUC + Spearman IC, permutation p-value.
"""

import sys
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.incremental_topsbots import IncrementalTopsBots

DATA_DIR = Path(os.environ.get(
    "RANGE_DATA_DIR",
    str(PROJECT_ROOT / "data" / "range_bar_causal")
))
RESULTS_DIR = Path(__file__).resolve().parent / "results"

PAIRS = [
    "EUR_JPY", "USD_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY", "CHF_JPY", "NZD_JPY",
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "EUR_GBP",
]
PIP = {
    "EUR_JPY": 0.01, "USD_JPY": 0.01, "GBP_JPY": 0.01, "AUD_JPY": 0.01,
    "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
}
SPREAD = {
    "EUR_JPY": 2.3, "USD_JPY": 1.7, "GBP_JPY": 3.3, "AUD_JPY": 2.1,
    "CAD_JPY": 2.3, "CHF_JPY": 3.5, "NZD_JPY": 2.7,
    "EUR_USD": 1.6, "GBP_USD": 1.9, "AUD_USD": 1.3,
    "NZD_USD": 1.5, "EUR_GBP": 1.4,
}
HORIZONS = [5, 10, 20, 50]
LAGS = [0, 1, 2, 3, 4, 5]   # current bar + 5 lagged bars
TRAIN_FRAC = 0.70
N_PERMS = 200

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 300,
    "max_depth": 4,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 40,
    "verbose": -1,
    "n_jobs": 4,
}


def build_features(pair: str) -> pd.DataFrame:
    fpath = DATA_DIR / f"{pair}_range10_causal.parquet"
    df = pd.read_parquet(fpath).sort_index()

    close = df["mid_close"].values.astype(np.float64)
    pip = PIP[pair]

    # Compute SBA incrementally
    tb = IncrementalTopsBots()
    sba = np.empty(len(close), dtype=np.float32)
    for i, c in enumerate(close):
        s, _, _, _ = tb.update(c, c, c)
        sba[i] = np.float32(s / 2.0)

    df["sba"]   = sba
    df["mc_d"]  = df["mc_d"].astype(np.float32)
    df["mc_dd"] = df["mc_dd"].astype(np.float32)

    # Lag features: current + lag 1-5 for each base feature
    for base in ["sba", "mc_d", "mc_dd"]:
        for lag in LAGS:
            col = f"{base}_lag{lag}" if lag > 0 else base
            if lag > 0:
                df[col] = df[base].shift(lag)

    # Forward pip returns and binary targets
    for h in HORIZONS:
        fwd = (df["mid_close"].shift(-h) - df["mid_close"]) / pip
        df[f"fwd_{h}"] = fwd
        df[f"tgt_{h}"] = (fwd > 0).astype(int)

    df = df.dropna()
    return df


def run_pair(pair: str, rng: np.random.Generator) -> dict:
    df = build_features(pair)
    base_features = ["sba", "mc_d", "mc_dd"]
    features = []
    for base in base_features:
        for lag in LAGS:
            features.append(f"{base}_lag{lag}" if lag > 0 else base)
    n = len(df)
    split = int(n * TRAIN_FRAC)

    X_tr = df[features].values[:split]
    X_oo = df[features].values[split:]

    pair_results = {"pair": pair, "n_bars": n, "n_oos": n - split, "horizons": {}}

    for h in HORIZONS:
        y_tr = df[f"tgt_{h}"].values[:split]
        y_oo = df[f"tgt_{h}"].values[split:]
        fwd_oo = df[f"fwd_{h}"].values[split:]

        # Guard: need at least both classes in OOS
        if y_oo.sum() < 10 or (len(y_oo) - y_oo.sum()) < 10:
            continue

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_oo)[:, 1]

        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_oo, proba))
        ic, ic_p = stats.spearmanr(proba, fwd_oo)

        # Feature importances (gain)
        imp = dict(zip(features, model.booster_.feature_importance(importance_type="gain")))
        total_gain = sum(imp.values()) + 1e-9
        imp_pct = {k: round(v / total_gain * 100, 1) for k, v in imp.items()}

        # Permutation p-value: shuffle all sba lag columns together (preserve row structure)
        n_sba_cols = len(LAGS)  # columns 0..len(LAGS)-1 are sba+lags
        perm_aucs = []
        for _ in range(N_PERMS):
            X_perm = X_oo.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[:, :n_sba_cols] = X_perm[perm_idx, :n_sba_cols]
            p_proba = model.predict_proba(X_perm)[:, 1]
            perm_aucs.append(float(roc_auc_score(y_oo, p_proba)))
        perm_p = float(np.mean(np.array(perm_aucs) >= auc))

        spread = SPREAD[pair]
        net_fwd = fwd_oo - spread  # net of spread cost
        long_mean = float(np.mean(net_fwd[proba > 0.55]))  if (proba > 0.55).sum() > 0 else 0.0
        short_mean = float(np.mean(-net_fwd[proba < 0.45])) if (proba < 0.45).sum() > 0 else 0.0
        n_long_sigs  = int((proba > 0.55).sum())
        n_short_sigs = int((proba < 0.45).sum())

        pair_results["horizons"][str(h)] = {
            "auc": round(auc, 4),
            "ic": round(float(ic), 4),
            "ic_p": round(float(ic_p), 4),
            "perm_p_sba": round(perm_p, 4),
            "importance_pct": imp_pct,
            "net_long_mean_pips": round(long_mean, 3),
            "net_short_mean_pips": round(short_mean, 3),
            "n_long_signals": n_long_sigs,
            "n_short_signals": n_short_sigs,
        }

    return pair_results


def main():
    rng = np.random.default_rng(42)

    print(f"\n{'='*70}")
    print("LightGBM Feature Study — SB_A IronNet Range-Bar Inputs")
    print(f"Features: sba, mc_d, mc_dd × lags {LAGS}  |  Horizons: {HORIZONS} bars")
    print(f"Note: upnl_n/mae_n are position-state signals, excluded (not bar features)")
    print(f"{'='*70}\n")

    all_results = []
    for pair in PAIRS:
        fpath = DATA_DIR / f"{pair}_range10_causal.parquet"
        if not fpath.exists():
            print(f"  {pair}: parquet not found, skip")
            continue
        print(f"  {pair} ...", end="", flush=True)
        res = run_pair(pair, rng)
        all_results.append(res)
        best_auc = max((v["auc"] for v in res["horizons"].values()), default=0.5)
        best_h   = max(res["horizons"], key=lambda k: res["horizons"][k]["auc"], default="?")
        print(f" n={res['n_bars']}  best AUC={best_auc:.4f} @ h={best_h}")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("OOS AUC by pair × horizon (>0.52 = meaningful)")
    print(f"{'─'*70}")
    hdr = f"{'Pair':10s}" + "".join(f"  h={h:>3d}" for h in HORIZONS)
    print(hdr)
    print("─" * len(hdr))
    for res in all_results:
        row = f"{res['pair']:10s}"
        for h in HORIZONS:
            auc = res["horizons"].get(str(h), {}).get("auc", 0.5)
            flag = "🟢" if auc > 0.52 else ("🟡" if auc > 0.505 else "🔴")
            row += f"  {auc:.4f}{flag}"
        print(row)

    # ── Permutation p-value summary (sba shuffle) ──
    print(f"\n{'='*70}")
    print("Permutation p-value (sba shuffle) — significant = <0.05 = sba matters")
    print(f"{'─'*70}")
    print(f"{'Pair':10s}" + "".join(f"  h={h:>3d}" for h in HORIZONS))
    print("─" * len(hdr))
    for res in all_results:
        row = f"{res['pair']:10s}"
        for h in HORIZONS:
            p = res["horizons"].get(str(h), {}).get("perm_p_sba", 1.0)
            flag = "🟢" if p < 0.05 else ("🟡" if p < 0.10 else "🔴")
            row += f"  {p:.3f}{flag}"
        print(row)

    # ── Feature importance summary (aggregate by base feature across lags) ──
    print(f"\n{'='*70}")
    print("Average feature importance % across pairs × horizons (summed by base feature)")
    print(f"{'─'*70}")
    base_features = ["sba", "mc_d", "mc_dd"]
    imp_agg = {b: [] for b in base_features}
    lag_imp_agg = {}  # per-lag breakdown
    for res in all_results:
        for h in HORIZONS:
            h_data = res["horizons"].get(str(h), {})
            imp_pct = h_data.get("importance_pct", {})
            for base in base_features:
                total = 0.0
                for lag in LAGS:
                    col = f"{base}_lag{lag}" if lag > 0 else base
                    v = imp_pct.get(col, 0.0)
                    total += v
                    lag_imp_agg.setdefault(col, []).append(v)
                imp_agg[base].append(total)
    for b in base_features:
        avg = np.mean(imp_agg[b])
        lag_line = "  ".join(
            f"lag{lag}={np.mean(lag_imp_agg.get(f'{b}_lag{lag}' if lag>0 else b, [0])):.1f}%"
            for lag in LAGS
        )
        print(f"  {b:8s}: {avg:.1f}%  [{lag_line}]")

    # ── Net edge summary ──
    print(f"\n{'='*70}")
    print("Net pip edge after spread (threshold 0.55 long / 0.45 short)")
    print(f"{'─'*70}")
    for res in all_results:
        for h in HORIZONS:
            h_data = res["horizons"].get(str(h), {})
            nl = h_data.get("net_long_mean_pips", 0)
            ns = h_data.get("net_short_mean_pips", 0)
            n_l = h_data.get("n_long_signals", 0)
            n_s = h_data.get("n_short_signals", 0)
            if abs(nl) > 0.5 or abs(ns) > 0.5:
                print(f"  {res['pair']:10s} h={h:>3d}: long={nl:+.2f}p (n={n_l})  short={ns:+.2f}p (n={n_s})")

    # ── Save ──
    out = RESULTS_DIR / "gbm_feature_study.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
