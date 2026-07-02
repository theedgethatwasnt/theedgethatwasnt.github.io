#!/usr/bin/env python3
"""
LightGBM study: bar > λ×ATR signal on S5/M1/M5 (EUR_JPY only).

Feature: sign-preserving z-score of (bar_range / ATR14) over rolling window=1000.
  signed_z = sign(close - open) × z_1000(bar_range / ATR14)

Lambda sweep: threshold signed_z at λ ∈ [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
Combinations at M5 resolution:
  - s5_last, m1_last, m5 raw signed_z
  - agree_all: all 3 TFs fire same direction
  - agree_2: any 2 of 3 fire same direction

Training: expanding-window WF — monthly retrains (incremental live train).
Target: sign(forward pip return) at H = [1, 3, 12] M5 bars (5 / 15 / 60 min).
"""

import sys
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

S5_PATH  = PROJECT_ROOT / "data" / "s5_ohlc" / "EUR_JPY_S5_BA.parquet"
OUT_DIR  = PROJECT_ROOT / "research" / "experiments" / "results"
OUT_DIR.mkdir(exist_ok=True)

PIP      = 0.01        # EUR_JPY
SPREAD   = 2.3         # pips
ATR_PER  = 14
Z_WIN    = 1000        # rolling z-score population
LAMBDAS  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
HORIZONS = [1, 3, 12]  # M5 bars forward

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


# ─── Feature computation ──────────────────────────────────────────────────────

def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_c).abs(),
                    (low - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def signed_zscore(high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series,
                  atr_n: int = ATR_PER, z_win: int = Z_WIN) -> pd.DataFrame:
    """Returns DataFrame with columns: bar_range, atr, ratio, z, bar_dir, signed_z."""
    atr     = wilder_atr(high, low, close, atr_n)
    rng     = high - low
    ratio   = rng / (atr + 1e-12)
    z_mu    = ratio.rolling(z_win, min_periods=z_win // 2).mean()
    z_sigma = ratio.rolling(z_win, min_periods=z_win // 2).std().clip(lower=1e-12)
    z       = (ratio - z_mu) / z_sigma
    bar_dir = np.sign(close - open_).replace(0, 1).astype(float)
    return pd.DataFrame({
        "bar_range": rng,
        "atr":       atr,
        "ratio":     ratio,
        "z":         z,
        "bar_dir":   bar_dir,
        "signed_z":  bar_dir * z,
    }, index=high.index)


def resample_ohlc(df_s5: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample S5 mid-OHLC to given freq."""
    mid = pd.DataFrame({
        "open":  (df_s5["bid_o"] + df_s5["ask_o"]) / 2,
        "high":  (df_s5["bid_h"] + df_s5["ask_h"]) / 2,
        "low":   (df_s5["bid_l"] + df_s5["ask_l"]) / 2,
        "close": (df_s5["bid_c"] + df_s5["ask_c"]) / 2,
    }, index=df_s5.index)
    return mid.resample(freq).agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()


# ─── Load and build feature matrix at M5 resolution ──────────────────────────

def build_feature_matrix() -> pd.DataFrame:
    print("Loading S5 data...")
    raw = pd.read_parquet(S5_PATH)
    raw["ts"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.set_index("ts").sort_index()

    print("Computing S5 features...")
    s5_feat = signed_zscore(
        (raw["bid_h"] + raw["ask_h"]) / 2,
        (raw["bid_l"] + raw["ask_l"]) / 2,
        (raw["bid_c"] + raw["ask_c"]) / 2,
        (raw["bid_o"] + raw["ask_o"]) / 2,
    )
    # Resample last s5 signed_z into M5 buckets
    s5_m5 = s5_feat["signed_z"].resample("5min").last().rename("s5_signed_z")

    print("Computing M1 features...")
    m1 = resample_ohlc(raw, "1min")
    m1_feat = signed_zscore(m1["high"], m1["low"], m1["close"], m1["open"])
    m1_m5 = m1_feat["signed_z"].resample("5min").last().rename("m1_signed_z")

    print("Computing M5 features...")
    m5 = resample_ohlc(raw, "5min")
    m5_feat = signed_zscore(m5["high"], m5["low"], m5["close"], m5["open"])
    m5_feat = m5_feat.rename(columns={"signed_z": "m5_signed_z"})
    m5_z    = m5_feat["m5_signed_z"]

    # Forward pip returns (target)
    mid_close_m5 = m5["close"]
    df = pd.DataFrame({"m5_signed_z": m5_z}, index=m5_z.index)
    df = df.join(s5_m5, how="left").join(m1_m5, how="left")

    for h in HORIZONS:
        fwd_pip = (mid_close_m5.shift(-h) - mid_close_m5) / PIP
        df[f"fwd_{h}"] = fwd_pip
        df[f"tgt_{h}"] = (fwd_pip > 0).astype(int)

    # Lambda threshold features + combination features
    for lam in LAMBDAS:
        tag = str(lam).replace(".", "p")
        s5z = df["s5_signed_z"].fillna(0)
        m1z = df["m1_signed_z"].fillna(0)
        m5z = df["m5_signed_z"].fillna(0)

        # Per-TF fire signal: direction × (z > lambda)
        df[f"s5_fire_{tag}"] = np.sign(s5z) * (s5z.abs() > lam).astype(float)
        df[f"m1_fire_{tag}"] = np.sign(m1z) * (m1z.abs() > lam).astype(float)
        df[f"m5_fire_{tag}"] = np.sign(m5z) * (m5z.abs() > lam).astype(float)

        # Combination: agreement score (-3 to +3, counts TFs in same direction)
        df[f"agree_{tag}"] = (
            np.sign(s5z) * (s5z.abs() > lam) +
            np.sign(m1z) * (m1z.abs() > lam) +
            np.sign(m5z) * (m5z.abs() > lam)
        ).astype(float)

        # Binary: all 3 agree same direction
        df[f"agree_all_{tag}"] = (df[f"agree_{tag}"].abs() == 3).astype(float) * np.sign(df[f"agree_{tag}"])
        # Binary: any 2 of 3 agree
        df[f"agree_2plus_{tag}"] = (df[f"agree_{tag}"].abs() >= 2).astype(float) * np.sign(df[f"agree_{tag}"])

    df = df.dropna(subset=["s5_signed_z", "m1_signed_z", "m5_signed_z"])
    print(f"Feature matrix: {len(df)} M5 bars, {len(df.columns)} columns")
    print(f"Date range: {df.index[0]} → {df.index[-1]}")
    return df


# ─── Walk-forward LightGBM ────────────────────────────────────────────────────

def wf_train_eval(df: pd.DataFrame, feature_cols: list[str], h: int,
                  rng: np.random.Generator, n_perms: int = 100) -> dict:
    """Expanding-window WF: retrain monthly, test on each subsequent month."""
    target_col = f"tgt_{h}"
    fwd_col    = f"fwd_{h}"

    # Build monthly fold indices
    months = df.index.to_period("M").unique()
    n_months = len(months)
    min_train_months = max(2, n_months // 3)  # first 1/3 as seed train

    fold_results = []
    all_proba = []
    all_y     = []
    all_fwd   = []

    model = None
    for i in range(min_train_months, n_months):
        train_mask = df.index.to_period("M") < months[i]
        test_mask  = df.index.to_period("M") == months[i]

        X_tr = df.loc[train_mask, feature_cols].values
        y_tr = df.loc[train_mask, target_col].values
        X_te = df.loc[test_mask, feature_cols].values
        y_te = df.loc[test_mask, target_col].values
        f_te = df.loc[test_mask, fwd_col].values

        if len(X_tr) < 500 or len(X_te) < 50:
            continue
        if y_te.sum() < 10 or (len(y_te) - y_te.sum()) < 10:
            continue

        # Incremental: continue training from previous model if available
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_te)[:, 1]
        fold_auc = float(roc_auc_score(y_te, proba))
        fold_ic, _  = stats.spearmanr(proba, f_te)
        fold_results.append({
            "month": str(months[i]),
            "n_train": int(train_mask.sum()),
            "n_test":  int(test_mask.sum()),
            "auc": round(fold_auc, 4),
            "ic":  round(float(fold_ic), 4),
        })
        all_proba.extend(proba.tolist())
        all_y.extend(y_te.tolist())
        all_fwd.extend(f_te.tolist())

    if not all_y:
        return {"n_folds": 0}

    all_proba = np.array(all_proba)
    all_y     = np.array(all_y, dtype=int)
    all_fwd   = np.array(all_fwd)

    agg_auc = float(roc_auc_score(all_y, all_proba))
    agg_ic, ic_p = stats.spearmanr(all_proba, all_fwd)

    # Net pip edge at 0.55/0.45 thresholds (after spread)
    long_mask  = all_proba > 0.55
    short_mask = all_proba < 0.45
    net_long  = float(np.mean(all_fwd[long_mask]  - SPREAD)) if long_mask.sum() > 0 else 0.0
    net_short = float(np.mean(-all_fwd[short_mask] - SPREAD)) if short_mask.sum() > 0 else 0.0

    # Permutation p-value (shuffle first feature — the primary z-score)
    perm_aucs = []
    for _ in range(n_perms):
        # retrain on full IS with shuffled first feature
        X_tr_full = df.iloc[:int(len(df) * 0.7)][feature_cols].values.copy()
        y_tr_full = df.iloc[:int(len(df) * 0.7)][target_col].values
        X_te_full = df.iloc[int(len(df) * 0.7):][feature_cols].values.copy()
        y_te_full = df.iloc[int(len(df) * 0.7):][target_col].values
        if len(X_te_full) < 50 or y_te_full.sum() < 5:
            break
        X_tr_full[:, 0] = rng.permutation(X_tr_full[:, 0])
        X_te_full[:, 0] = rng.permutation(X_te_full[:, 0])
        pm = lgb.LGBMClassifier(**LGB_PARAMS)
        pm.fit(X_tr_full, y_tr_full)
        p_proba = pm.predict_proba(X_te_full)[:, 1]
        if y_te_full.sum() >= 5 and (len(y_te_full) - y_te_full.sum()) >= 5:
            perm_aucs.append(float(roc_auc_score(y_te_full, p_proba)))
    perm_p = float(np.mean(np.array(perm_aucs) >= agg_auc)) if perm_aucs else 1.0

    return {
        "n_folds": len(fold_results),
        "agg_auc": round(agg_auc, 4),
        "agg_ic":  round(float(agg_ic), 4),
        "ic_p":    round(float(ic_p), 4),
        "perm_p":  round(perm_p, 4),
        "net_long_pips":  round(net_long, 3),
        "net_short_pips": round(net_short, 3),
        "n_long_signals":  int(long_mask.sum()),
        "n_short_signals": int(short_mask.sum()),
        "fold_aucs": [f["auc"] for f in fold_results],
        "folds": fold_results,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)
    df  = build_feature_matrix()

    print(f"\n{'='*70}")
    print("EUR/JPY — Bar-ATR Sign-Preserving Z-Score Feature Study")
    print(f"Z-score window={Z_WIN}, ATR period={ATR_PER}, TFs: S5/M1/M5")
    print(f"Lambdas: {LAMBDAS}, Horizons: {HORIZONS} M5 bars")
    print(f"WF: expanding-window monthly retrains (incremental live train)")
    print(f"{'='*70}\n")

    all_results = []

    # ── Study A: raw continuous signed_z per TF (no lambda threshold) ──
    print("── A: Raw signed_z per TF ──")
    for feat_cols, label in [
        (["s5_signed_z"], "S5_raw"),
        (["m1_signed_z"], "M1_raw"),
        (["m5_signed_z"], "M5_raw"),
        (["s5_signed_z", "m1_signed_z", "m5_signed_z"], "All3_raw"),
    ]:
        for h in HORIZONS:
            res = wf_train_eval(df, feat_cols, h, rng, n_perms=100)
            res.update({"label": label, "h": h, "type": "raw_z"})
            all_results.append(res)
            flag = "🟢" if res["agg_auc"] > 0.52 else ("🟡" if res["agg_auc"] > 0.505 else "🔴")
            print(f"  {label:15s} h={h:2d}: AUC={res['agg_auc']:.4f}{flag}  "
                  f"IC={res['agg_ic']:+.4f}  perm_p={res['perm_p']:.3f}  "
                  f"folds={res['fold_aucs']}")

    # ── Study B: lambda sweep per TF ──
    print("\n── B: Lambda sweep (per TF fire signal) ──")
    for lam in LAMBDAS:
        tag = str(lam).replace(".", "p")
        for feat_cols, label in [
            ([f"s5_fire_{tag}"], f"S5_L{lam}"),
            ([f"m1_fire_{tag}"], f"M1_L{lam}"),
            ([f"m5_fire_{tag}"], f"M5_L{lam}"),
        ]:
            for h in HORIZONS:
                res = wf_train_eval(df, feat_cols, h, rng, n_perms=50)
                res.update({"label": label, "h": h, "lam": lam, "type": "lambda_single"})
                all_results.append(res)
            # Print summary for this lambda/TF (best horizon only)
            best = max([r for r in all_results if r["label"] == label], key=lambda r: r["agg_auc"])
            flag = "🟢" if best["agg_auc"] > 0.52 else ("🟡" if best["agg_auc"] > 0.505 else "🔴")
            print(f"  {label:12s}: best AUC={best['agg_auc']:.4f}{flag} @ h={best['h']}")

    # ── Study C: combination features (all agree / 2+ agree) ──
    print("\n── C: Combination features (agreement across TFs) ──")
    for lam in LAMBDAS:
        tag = str(lam).replace(".", "p")
        for feat_cols, label in [
            ([f"agree_{tag}"],        f"Agree_score_L{lam}"),
            ([f"agree_all_{tag}"],    f"Agree_all_L{lam}"),
            ([f"agree_2plus_{tag}"],  f"Agree_2p_L{lam}"),
        ]:
            for h in HORIZONS:
                res = wf_train_eval(df, feat_cols, h, rng, n_perms=50)
                res.update({"label": label, "h": h, "lam": lam, "type": "combination"})
                all_results.append(res)
            best = max([r for r in all_results if r["label"] == label], key=lambda r: r["agg_auc"])
            flag = "🟢" if best["agg_auc"] > 0.52 else ("🟡" if best["agg_auc"] > 0.505 else "🔴")
            print(f"  {label:25s}: best AUC={best['agg_auc']:.4f}{flag} @ h={best['h']}")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("TOP RESULTS (AUC > 0.51, sorted)")
    print(f"{'─'*70}")
    top = sorted([r for r in all_results if r.get("agg_auc", 0) > 0.51],
                 key=lambda r: -r["agg_auc"])
    print(f"{'Label':30s}  {'h':>3s}  {'AUC':>6s}  {'IC':>7s}  {'perm_p':>6s}  {'net_L':>7s}  {'net_S':>7s}")
    print("─" * 80)
    for r in top[:25]:
        flag = "🟢" if r["agg_auc"] > 0.52 else "🟡"
        print(f"  {r['label']:30s}  {r['h']:3d}  {r['agg_auc']:.4f}{flag}  "
              f"{r['agg_ic']:+.4f}  {r['perm_p']:.3f}  "
              f"{r['net_long_pips']:+.2f}p  {r['net_short_pips']:+.2f}p")

    # ── Save ──
    out = OUT_DIR / "lgbm_bar_atr_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
