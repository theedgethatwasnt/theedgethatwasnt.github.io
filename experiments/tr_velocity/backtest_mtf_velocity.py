"""
MTF TR Velocity — LightGBM feature study at native S5 resolution.

No resampling. Every feature is a rolling window over S5 bars.
Window size determines the implied timeframe:
  N=6    →  30s  (S5 short)
  N=12   →  1m   (M1 equiv)
  N=60   →  5m   (M5 equiv)
  N=360  →  30m
  N=720  →  1h   (H1 equiv)
  N=1440 →  2h

All features normalised to pips/minute so they're directly comparable.

  velocity_N  = sum(TR_pips, last N bars) / (N × 5/60)   [pips/min, unsigned]
  direction_N = (close[-1] − close[-N]) / pip / (N × 5/60)  [pips/min, signed]

Weighted average: vel_weighted = Σ (1/N × vel_N) / Σ (1/N)
  Short windows get higher weight (more responsive to recent momentum).

Instrument: EUR_USD  (8.26M S5 bars, 2024-02 → 2026-05)
Forward targets: 1m (N=12), 5m (N=60), 10m (N=120) ahead returns in pips.

Run:
    cd /path/to/projects/fx-core
    python3 research/experiments/tr_velocity/backtest_mtf_velocity.py
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import roc_auc_score

BASE   = Path(__file__).resolve().parents[3]
S5_DIR = BASE / "data/s5_ohlc"

PAIR   = "EUR_USD"
PIP    = 0.0001
BAR_S  = 5
BAR_M  = BAR_S / 60.0   # minutes per S5 bar = 0.0833

IS_FRAC = 0.70

# (window_bars, human_label)  — bars × BAR_M = window minutes
WINDOWS = [
    (6,    "s5_30s"),
    (12,   "m1_1m"),
    (36,   "m1_3m"),
    (60,   "m5_5m"),
    (180,  "m5_15m"),
    (360,  "h1_30m"),
    (720,  "h1_1h"),
    (1440, "h1_2h"),
]

# Forward horizons (bars) for IC and target
FWD_BARS = [12, 60, 120]   # 1m, 5m, 10m


# ── Data loader ───────────────────────────────────────────────────────────────
def load_s5(pair: str) -> pd.DataFrame:
    df = pd.read_parquet(S5_DIR / f"{pair}_S5_BA.parquet")
    # EUR_USD S5 has mid OHLC directly + bid_c/ask_c for spread
    df['sp'] = (df['ask_c'] - df['bid_c']) / PIP
    prev_c   = df['close'].shift(1)
    tr_hi    = np.maximum(df['high'].values,  prev_c.values)
    tr_lo    = np.minimum(df['low'].values,   prev_c.values)
    df['tr'] = (tr_hi - tr_lo) / PIP
    df = df.rename(columns={'open':'o','high':'h','low':'l','close':'c'})
    return df[['timestamp', 'o', 'h', 'l', 'c', 'sp', 'tr']].reset_index(drop=True)


# ── Feature builder ───────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    max_win = max(N for N, _ in WINDOWS)
    feats   = {}

    inv_N = np.array([1.0 / N for N, _ in WINDOWS])
    inv_N /= inv_N.sum()   # normalised weights for weighted average

    # Per-bar directional movement (S5 resolution) — ADX building blocks
    prev_h    = df['h'].shift(1)
    prev_l    = df['l'].shift(1)
    up_move   = (df['h'] - prev_h).clip(lower=0)
    down_move = (prev_l - df['l']).clip(lower=0)
    # +DM fires when up move > down move; -DM vice versa
    dm_plus  = np.where(up_move   > down_move, up_move,   0.0)
    dm_minus = np.where(down_move > up_move,   down_move, 0.0)
    dm_plus_s  = pd.Series(dm_plus,  index=df.index) / PIP   # pips
    dm_minus_s = pd.Series(dm_minus, index=df.index) / PIP

    vel_series = []
    acc_series = []
    csi_series = []
    for (N, label), w in zip(WINDOWS, inv_N):
        win_min = N * BAR_M

        # TR velocity: unsigned pips/minute
        vel = df['tr'].rolling(N).sum() / win_min
        feats[f"vel_{label}"] = vel
        vel_series.append(vel * w)

        # Acceleration: Δvelocity / Δtime  [pips/min²]
        acc = vel.diff(N) / win_min
        feats[f"acc_{label}"] = acc
        acc_series.append(acc * w)

        # Signed drift: pips/minute in price direction
        drift = (df['c'] - df['c'].shift(N)) / PIP / win_min
        feats[f"dir_{label}"] = drift

        # ── CSI components at window N ─────────────────────────────────
        atr_N   = df['tr'].rolling(N).mean()                      # pips
        di_plus  = dm_plus_s.rolling(N).sum()  / (atr_N * N + 1e-10) * 100
        di_minus = dm_minus_s.rolling(N).sum() / (atr_N * N + 1e-10) * 100
        net_di   = di_plus - di_minus                              # signed directional bias
        dx       = (net_di.abs() / (di_plus + di_minus + 1e-10)) * 100
        adx_N    = dx.rolling(N).mean()                           # trend strength 0-100
        csi_N    = adx_N * atr_N                                  # simplified CSI: strength × volatility

        feats[f"net_di_{label}"] = net_di   # signed: + = uptrend, − = downtrend
        feats[f"adx_{label}"]    = adx_N    # directionless trend strength
        feats[f"csi_{label}"]    = csi_N    # simplified CSI (pips, unsigned)
        csi_series.append(csi_N * w)

    feats["vel_weighted"] = sum(vel_series)   # weighted average velocity
    feats["acc_weighted"] = sum(acc_series)   # weighted average acceleration
    feats["csi_weighted"] = sum(csi_series)   # weighted average CSI
    feats["spread"]       = df['sp']

    feat_df = pd.DataFrame(feats, index=df.index)

    # Forward returns in pips (targets)
    for fwd in FWD_BARS:
        feat_df[f"y_{fwd}"] = (df['c'].shift(-fwd) - df['c']) / PIP

    feat_df["ts"] = df["timestamp"]
    return feat_df.iloc[max_win * 2:].dropna()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"MTF TR Velocity study — {PAIR} at S5 resolution")
    print(f"Windows (bars → minutes): "
          + "  ".join(f"{N}→{N*BAR_M:.1f}m" for N,_ in WINDOWS))
    print("="*70)

    t0 = time.time()
    print("Loading S5...", end=" ", flush=True)
    df = load_s5(PAIR)
    print(f"{len(df):,} bars  ({time.time()-t0:.1f}s)")

    print("Building features...", end=" ", flush=True)
    t0 = time.time()
    feat_df = build_features(df)
    print(f"{len(feat_df):,} rows  ({time.time()-t0:.1f}s)")

    feat_cols = [c for c in feat_df.columns
                 if c.startswith(("vel_", "acc_", "dir_",
                                   "net_di_", "adx_", "csi_", "spread"))]
    print(f"Features: {feat_cols}\n")

    is_end   = int(len(feat_df) * IS_FRAC)
    df_is    = feat_df.iloc[:is_end]
    df_oos   = feat_df.iloc[is_end:]

    # ── IC table (Spearman corr vs each forward horizon) ─────────────────
    print(f"  {'Feature':<24} {'IC_1m':>8} {'IC_5m':>8} {'IC_10m':>8}  sig")
    print(f"  {'─'*60}")
    for col in feat_cols:
        ics = []
        for fwd in FWD_BARS:
            r, p = stats.spearmanr(feat_df[col], feat_df[f"y_{fwd}"],
                                   nan_policy='omit')
            ics.append((r, p))
        sig = "★" if abs(ics[1][0]) > 0.01 and ics[1][1] < 0.001 else " "
        print(f"  {sig} {col:<23} "
              f"{ics[0][0]:>+8.4f} {ics[1][0]:>+8.4f} {ics[2][0]:>+8.4f}")

    # ── LightGBM: predict sign of 5m forward return ───────────────────────
    PRIMARY = f"y_{FWD_BARS[1]}"   # 5m
    print(f"\nLightGBM  target=sign({PRIMARY})  "
          f"IS={is_end:,}  OOS={len(df_oos):,}")

    X_is    = df_is[feat_cols].values.astype(np.float32)
    X_oos   = df_oos[feat_cols].values.astype(np.float32)
    y_is    = (df_is[PRIMARY].values  > 0).astype(int)
    y_oos   = (df_oos[PRIMARY].values > 0).astype(int)

    params = dict(
        objective='binary', metric='auc',
        num_leaves=31, learning_rate=0.05, n_estimators=500,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=100, verbose=-1,
    )
    t0 = time.time()
    model = lgb.LGBMClassifier(**params)
    model.fit(X_is, y_is,
              eval_set=[(X_oos, y_oos)],
              callbacks=[lgb.early_stopping(40, verbose=False),
                         lgb.log_evaluation(period=-1)])
    print(f"Trained {time.time()-t0:.1f}s  best_iter={model.best_iteration_}")

    auc_is  = roc_auc_score(y_is,  model.predict_proba(X_is)[:,  1])
    auc_oos = roc_auc_score(y_oos, model.predict_proba(X_oos)[:, 1])
    print(f"AUC  IS={auc_is:.4f}   OOS={auc_oos:.4f}  "
          + ("🟢 signal" if auc_oos > 0.52 else
             "🟡 marginal" if auc_oos > 0.505 else "🔴 no signal"))

    print("\nFeature importance (gain, top 12):")
    max_gain = max(model.feature_importances_)
    imp = sorted(zip(feat_cols, model.feature_importances_),
                 key=lambda x: -x[1])
    for name, gain in imp[:12]:
        bar = "█" * int(gain / max_gain * 25)
        print(f"  {name:<24} {gain:>8.0f}  {bar}")

    # ── Threshold sweep on vel_weighted (IS only) ────────────────────────
    print("\nvel_weighted threshold sweep (IS, 5m target):")
    vel  = df_is["vel_weighted"].values
    ret5 = df_is[PRIMARY].values
    print(f"  {'Threshold':>12} {'N_above':>8} {'mean_ret':>10} {'t_stat':>8}")
    for thr in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        mask = vel >= thr
        if mask.sum() < 50:
            continue
        above = ret5[mask]
        t, p  = stats.ttest_1samp(above, 0)
        print(f"  vel≥{thr:>6.1f}p/m  {mask.sum():>8,}  "
              f"{above.mean():>+10.4f}p  t={t:>+7.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
