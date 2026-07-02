"""
S5 Rolling-Window Momentum Feature Study — EUR/JPY
====================================================
Instead of resampling S5 → fixed TFs, compute rolling windows of N S5 bars
to get a continuum of horizons. Finds which horizon has predictive power.

Feature types at N = [12, 60, 120, 360, 720, 2880] S5 bars (1m,5m,10m,30m,1h,4h):
  mom_N   : N-bar momentum (pips)             — 1st finite difference
  vel_N   : momentum / N (pips/bar)           — velocity
  acc_N   : mom[t] - mom[t-N] (pips)          — 2nd finite difference (acceleration)
  vol_N   : rolling ATR at horizon N (pips)   — volatility context
  mom_z_N : mom_N / vol_N                     — normalized momentum

Targets at H = [6, 12, 60, 120] S5 bars (30s, 1m, 5m, 10m):
  dir_H   : sign(mid[t+H] - mid[t])           — binary direction
  sadj_H  : forward_ret > median_spread        — spread-adjusted profitability

Study:
  1. IC (Spearman rank correlation) per feature x target (IS only)
  2. LightGBM walk-forward AUC per target horizon (IS=70%, 3 folds, subsampled)
  3. Feature importance by horizon to find which TF has the most signal

EUR_JPY S5: Oct 2025 → Apr 2026, 2.17M bars.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
PIP  = 0.0001  # EUR/USD

# Rolling horizons in S5 bars → equivalent TF
FEAT_HORIZONS = [12,   60,  120,  360,  720, 2880]
FEAT_LABELS   = ["1m","5m","10m","30m","1h","4h"]

# Target horizons in S5 bars → prediction window
TGT_HORIZONS = [6,    12,   60,  120]
TGT_LABELS   = ["30s","1m","5m","10m"]

IS_FRAC    = 0.70
N_WF_FOLDS = 3


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    # EUR_USD S5: timestamp, open, high, low, close (mid), bid_c, ask_c, volume
    path = ROOT / "data" / "s5_ohlc" / "EUR_USD_S5_BA.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["mid_c"] = df["close"].astype(float)
    df["mid_h"] = df["high"].astype(float)
    df["mid_l"] = df["low"].astype(float)
    df["spread_pips"] = (df["ask_c"].astype(float) - df["bid_c"].astype(float)) / PIP
    return df


# ── Feature computation ───────────────────────────────────────────────────────

def rolling_atr(hi, lo, cl, n):
    """Rolling mean of True Range over N bars (causal)."""
    pc = pd.Series(cl).shift(1)
    pc.iloc[0] = cl[0]
    tr = np.maximum(hi - lo,
         np.maximum(np.abs(hi - pc.values),
                    np.abs(lo - pc.values)))
    return pd.Series(tr).rolling(n, min_periods=n).mean().values


def build_features(df):
    mid = pd.Series(df["mid_c"].values)
    hi  = df["mid_h"].values
    lo  = df["mid_l"].values
    cl  = df["mid_c"].values

    feat = {}
    for N, lbl in zip(FEAT_HORIZONS, FEAT_LABELS):
        # Momentum: N-bar return in pips
        mom = (mid - mid.shift(N)) / PIP

        # Velocity: pips per S5 bar
        vel = mom / N

        # Acceleration: change in N-bar momentum over another N bars
        # = (close[t] - close[t-N]) - (close[t-N] - close[t-2N])
        # = close[t] - 2*close[t-N] + close[t-2N]
        acc = mom - mom.shift(N)

        # Volatility: rolling ATR at this horizon
        atr_raw = rolling_atr(hi, lo, cl, N)
        vol = pd.Series(atr_raw) / PIP

        # Normalized momentum (z-score by local volatility)
        mom_z = mom / vol.replace(0, np.nan)

        feat[f"mom_{lbl}"]   = mom.values
        feat[f"vel_{lbl}"]   = vel.values
        feat[f"acc_{lbl}"]   = acc.values
        feat[f"vol_{lbl}"]   = vol.values
        feat[f"mom_z_{lbl}"] = mom_z.values

    return pd.DataFrame(feat, index=df.index)


# ── Target computation ────────────────────────────────────────────────────────

def build_targets(df):
    mid = pd.Series(df["mid_c"].values)
    sp50 = float(df["spread_pips"].quantile(0.50))
    tgt = {}
    for H, lbl in zip(TGT_HORIZONS, TGT_LABELS):
        fwd_pip = (mid.shift(-H) - mid) / PIP
        tgt[f"ret_{lbl}"]  = fwd_pip.values
        tgt[f"dir_{lbl}"]  = (fwd_pip > 0).astype(int).values
        # Spread-adjusted: long trade must beat median spread cost
        tgt[f"sadj_{lbl}"] = (fwd_pip > sp50).astype(int).values
    return pd.DataFrame(tgt, index=df.index), sp50


# ── IC study ─────────────────────────────────────────────────────────────────

def ic_study(feat_df, tgt_df, is_mask):
    """Spearman rank-IC of each feature vs forward return, IS only."""
    rows = []
    for fcol in feat_df.columns:
        for lbl in TGT_LABELS:
            tcol = f"ret_{lbl}"
            fv = feat_df.loc[is_mask, fcol].values.astype(float)
            tv = tgt_df.loc[is_mask, tcol].values.astype(float)
            ok = np.isfinite(fv) & np.isfinite(tv)
            if ok.sum() < 500:
                continue
            rho, pval = stats.spearmanr(fv[ok], tv[ok])
            denom = max(1 - rho**2, 1e-12)
            t_stat = rho * np.sqrt(ok.sum() - 2) / np.sqrt(denom)
            rows.append({
                "feature": fcol, "target": lbl,
                "IC": round(rho, 6),
                "t_stat": round(t_stat, 2),
                "pval": round(pval, 5),
                "n": int(ok.sum()),
            })
    return pd.DataFrame(rows).sort_values("t_stat", key=abs, ascending=False)


# ── LightGBM walk-forward ─────────────────────────────────────────────────────

def lgbm_wf(X, y, n_folds=3, step=1):
    """Walk-forward AUC. Step=H for non-overlapping target windows."""
    # Subsample at target resolution to remove serial autocorrelation
    X = X[::step]
    y = y[::step]
    n = len(X)
    fold_sz = n // (n_folds + 1)
    if fold_sz < 200:
        return np.nan, []
    aucs = []
    for k in range(n_folds):
        tr_end   = fold_sz * (k + 1)
        te_start = tr_end
        te_end   = tr_end + fold_sz
        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_te, y_te = X[te_start:te_end], y[te_start:te_end]
        if y_tr.sum() < 30 or (y_tr == 0).sum() < 30:
            continue
        m = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=31,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, verbose=-1, random_state=42,
        )
        m.fit(X_tr, y_tr)
        prob = m.predict_proba(X_te)[:, 1]
        if y_te.sum() > 0 and (y_te == 0).sum() > 0:
            aucs.append(roc_auc_score(y_te, prob))
    return (np.mean(aucs) if aucs else np.nan), aucs


def run_lgbm_study(feat_df, tgt_df, is_mask, feat_names):
    X_all = feat_df.loc[is_mask].values.astype(np.float32)
    # Replace inf/nan with 0 (LightGBM handles missing but cleaner this way)
    X_all = np.where(np.isfinite(X_all), X_all, 0.0)

    auc_results  = {}
    importances  = {}

    for H, lbl in zip(TGT_HORIZONS, TGT_LABELS):
        y_all = tgt_df.loc[is_mask, f"dir_{lbl}"].values.astype(float)
        ok    = np.isfinite(y_all)
        X_ok  = X_all[ok]
        y_ok  = y_all[ok].astype(int)
        n_ok  = ok.sum()
        print(f"  [{lbl}] {n_ok:,} IS bars (step={H}), base rate {y_ok.mean():.3f}")

        auc_mean, folds = lgbm_wf(X_ok, y_ok, n_folds=N_WF_FOLDS, step=H)
        auc_results[lbl] = {
            "auc": round(auc_mean, 4) if np.isfinite(auc_mean) else None,
            "folds": [round(a, 4) for a in folds],
        }
        marker = ("🟢" if auc_mean > 0.52 else
                  "🟡" if auc_mean > 0.505 else "🔴")
        print(f"    AUC = {auc_mean:.4f} {marker}  folds={[round(a,4) for a in folds]}")

        # Fit on full IS for importance
        X_sub = X_ok[::H]
        y_sub = y_ok[::H]
        m_full = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=31,
            min_child_samples=100, n_jobs=-1, verbose=-1, random_state=42,
        )
        m_full.fit(X_sub, y_sub)
        importances[lbl] = dict(zip(feat_names, m_full.feature_importances_))

    return auc_results, importances


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  S5 Momentum Feature Study — EUR/USD")
    print("=" * 62)

    print("\nLoading data...")
    df = load_data()
    n  = len(df)
    print(f"  {n:,} bars | {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")
    sp50 = float(df["spread_pips"].quantile(0.50))
    sp90 = float(df["spread_pips"].quantile(0.90))
    print(f"  Spread: P50={sp50:.2f}p  P90={sp90:.2f}p")

    print("\nBuilding features...")
    feat_df = build_features(df)
    feat_names = feat_df.columns.tolist()
    print(f"  {len(feat_names)} features: {feat_names[:5]} ...")

    print("Building targets...")
    tgt_df, sp50 = build_targets(df)

    # Warmup: need 2*max_horizon for acceleration feature
    warmup   = 2 * max(FEAT_HORIZONS) + 1   # 5761 bars ≈ 8h
    max_tgt  = max(TGT_HORIZONS)            # drop last rows where target is NaN
    # Valid index range
    valid    = feat_df.iloc[warmup:-max_tgt].index
    n_valid  = len(valid)
    is_end   = int(n_valid * IS_FRAC)
    is_idx   = valid[:is_end]
    oos_idx  = valid[is_end:]

    is_mask  = pd.Series(False, index=df.index)
    oos_mask = pd.Series(False, index=df.index)
    is_mask.loc[is_idx]   = True
    oos_mask.loc[oos_idx] = True

    print(f"\n  IS:  {is_mask.sum():>8,} bars → through {df['timestamp'][is_mask].iloc[-1].date()}")
    print(f"  OOS: {oos_mask.sum():>8,} bars → through {df['timestamp'][oos_mask].iloc[-1].date()}")

    # ── 1. IC Study ───────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  IC Study (Spearman, IS only)")
    print("=" * 62)
    ic_df = ic_study(feat_df, tgt_df, is_mask)

    # Top 20 by |t_stat|
    print(f"\nTop 20 feature-target pairs by |t-stat|:")
    print(f"  {'Feature':<20} {'Target':>6} {'IC':>9} {'t-stat':>8} {'pval':>8}")
    print("  " + "-" * 56)
    for _, row in ic_df.head(20).iterrows():
        sig = "**" if abs(row["t_stat"]) > 3.0 else "  "
        print(f"  {row['feature']:<20} {row['target']:>6} {row['IC']:>9.5f} "
              f"{row['t_stat']:>8.2f} {row['pval']:>8.5f} {sig}")

    # Best horizon per feature type
    print(f"\nBest horizon per feature type (max |IC| across targets):")
    ic_df["ftype"] = ic_df["feature"].str.split("_").str[0]
    ic_df["fhz"]   = ic_df["feature"].str.split("_").str[1]
    best_hz = (ic_df.groupby(["ftype", "fhz"])["t_stat"]
               .apply(lambda x: x.abs().max())
               .reset_index()
               .sort_values(["ftype", "t_stat"], ascending=[True, False]))
    print(f"  {'Type':<8} {'Horizon':>8} {'Max|t|':>8}")
    print("  " + "-" * 28)
    shown = set()
    for _, row in best_hz.iterrows():
        key = row["ftype"]
        if key not in shown:
            shown.add(key)
            print(f"  {row['ftype']:<8} {row['fhz']:>8} {row['t_stat']:>8.2f}")

    # ── 2. LightGBM Walk-Forward ──────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  LightGBM Walk-Forward AUC (IS only, subsampled at target step)")
    print("=" * 62)
    auc_results, importances = run_lgbm_study(feat_df, tgt_df, is_mask, feat_names)

    # ── 3. Horizon importance breakdown ──────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Feature Importance by Horizon")
    print("=" * 62)
    for tgt_lbl, imp in importances.items():
        print(f"\n  Target {tgt_lbl}:")
        hz_scores = {lbl: 0 for lbl in FEAT_LABELS}
        for feat, score in imp.items():
            for lbl in FEAT_LABELS:
                if feat.endswith(f"_{lbl}"):
                    hz_scores[lbl] += score
        total = sum(hz_scores.values()) or 1
        for hz, sc in sorted(hz_scores.items(), key=lambda x: -x[1]):
            pct = 100 * sc / total
            bar = "█" * int(pct / 4)
            print(f"    {hz:>4}  {pct:5.1f}%  {bar}")

    # Feature type breakdown
    print("\n  Feature Type Importance (avg across targets):")
    type_total = {ft: 0 for ft in ["mom", "vel", "acc", "vol", "mom_z"]}
    count = 0
    for imp in importances.values():
        count += 1
        for feat, score in imp.items():
            ftype = feat.split("_")[0]
            if ftype in type_total:
                type_total[ftype] += score
    grand = sum(type_total.values()) or 1
    for ftype, sc in sorted(type_total.items(), key=lambda x: -x[1]):
        pct = 100 * sc / grand
        bar = "█" * int(pct / 4)
        print(f"    {ftype:>6}  {pct:5.1f}%  {bar}")

    # ── Top 10 features per target ────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Top 10 Features per Target")
    print("=" * 62)
    for tgt_lbl, imp in importances.items():
        top = sorted(imp.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Target {tgt_lbl}:")
        for feat, score in top:
            print(f"    {feat:<22} {score:6.0f}")

    # ── Final AUC table ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  AUC Summary")
    print("=" * 62)
    print(f"  {'Target':>6} | {'AUC':>6} | Folds")
    print("  " + "-" * 40)
    for lbl, res in auc_results.items():
        auc = res["auc"]
        if auc is None:
            auc = float("nan")
        marker = ("🟢" if auc > 0.52 else "🟡" if auc > 0.505 else "🔴")
        print(f"  {lbl:>6} | {auc:.4f} | {res['folds']}  {marker}")

    print("\n  Baseline (random): 0.5000")
    print("  Threshold (marginal): 0.505 | (meaningful): 0.52")
    print("\nDone.")


if __name__ == "__main__":
    main()
