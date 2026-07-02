#!/usr/bin/env python3
"""LightGBM walk-forward: 14 trimmed features, 5-year M5 data, 3 OOS folds.

Features (current + l5 lag each):
  mc_d, mc_dd      — multi-TF ASI momentum (FXFeatureBuilder kalman10)
  s5_s50, p_s5, p_s50 — SMA5/SMA50 spread & deviation (pips)
  f1               — 5-min return z-score (arctan-normalized)
  f3               — 20-min efficiency ratio on M5 OHLC

Walk-forward: 3 expanding-IS folds
  Fold 1: IS 0-60%  | OOS 60-73%
  Fold 2: IS 0-73%  | OOS 73-87%
  Fold 3: IS 0-87%  | OOS 87-100%
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

M5_PATH     = ROOT / "data" / "m5_ohlc" / "EUR_JPY_M5.parquet"
M5_FEAT_PATH = ROOT / "data" / "m5_ohlc" / "EUR_JPY_M5_kalman10_causal.parquet"
PIP      = 0.01
SPREAD   = 2.3
HORIZONS = [5, 10, 20]

# Walk-forward split points (fraction of total bars)
WF_SPLITS = [0.60, 0.73, 0.87, 1.00]   # IS ends / OOS windows


# ── Numba helpers ─────────────────────────────────────────────────────────────

@njit(cache=True)
def compute_f1(close, pop=1000):
    n = len(close); r = np.zeros(n)
    for i in range(1, n): r[i] = (close[i] - close[i-1]) / PIP
    f = np.zeros(n); hp = np.pi / 2.; start = pop + 1
    s = 0.; sq = 0.
    for k in range(start - pop, start): s += r[k]; sq += r[k]*r[k]
    for i in range(start, n):
        if i > start:
            a = r[i-1]; e = r[i-1-pop]; s += a-e; sq += a*a-e*e
        v = sq/pop - (s/pop)**2
        std = v**0.5 if v > 1e-20 else 1e-10
        f[i] = np.arctan(r[i]/std) / hp
    return f, start


@njit(cache=True)
def compute_f3_m5(close, high, low, window=4):
    # 20-min efficiency ratio on M5 OHLC (window=4 M5 bars = 20 min)
    n = len(close); f = np.zeros(n)
    for i in range(window, n):
        net  = close[i] - close[i - window]
        path = 0.0
        for j in range(i - window, i + 1):
            path += high[j] - low[j]
        if path > 1e-10:
            f[i] = net / path
    return f


@njit(cache=True)
def simulate_pred(pred_signal, close, max_hold, thresh):
    pos = 0; ep = 0.; eb = 0; pnl = 0.; sc = 0.; nt = 0
    n = len(close)
    for i in range(n):
        s = pred_signal[i]
        if pos != 0:
            pnl = (close[i]-ep)/PIP if pos == 1 else (ep-close[i])/PIP
            if (i-eb) >= max_hold or (pos==1 and s<-thresh) or (pos==-1 and s>thresh):
                sc += pnl; nt += 1; pos = 0; pnl = 0.
        if pos == 0:
            if s > thresh:
                ep = close[i]+SPREAD*0.5*PIP; eb = i; pos = 1
            elif s < -thresh:
                ep = close[i]-SPREAD*0.5*PIP; eb = i; pos = -1
    return nt, sc/nt if nt > 0 else 0., sc


# ── Load M5 data ──────────────────────────────────────────────────────────────

print("Loading M5 data...")
df = pd.read_parquet(M5_PATH).sort_values("timestamp").reset_index(drop=True)
m5_o = df["open"].values.astype(np.float64)
m5_h = df["high"].values.astype(np.float64)
m5_l = df["low"].values.astype(np.float64)
m5_c = df["close"].values.astype(np.float64)
n = len(m5_c)
print(f"  {n:,} bars  {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")


# ── Compute features ──────────────────────────────────────────────────────────

print("Computing f1, f3...")
f1, start = compute_f1(m5_c)
f3 = compute_f3_m5(m5_c, m5_h, m5_l)
simulate_pred(np.zeros(200), m5_c[:200], 10, 0.01)  # JIT warmup

print("Computing SMA features...")
s = pd.Series(m5_c)
sma5    = s.rolling(5).mean().values
sma50   = s.rolling(50).mean().values
p_s5    = (m5_c - sma5)  / PIP
p_s50   = (m5_c - sma50) / PIP
s5_s50  = (sma5  - sma50) / PIP

print("Loading mc_d, mc_dd from pre-built kalman10 parquet...")
df_feat   = pd.read_parquet(M5_FEAT_PATH).sort_values("timestamp").reset_index(drop=True)
mc_d_arr  = df_feat["mc_d_a"].values.astype(np.float64)
mc_dd_arr = df_feat["mc_dd_a"].values.astype(np.float64)

print("Building feature matrix (14 features)...")
BASE = {
    'f1':     f1,    'f3':     f3,
    'mc_d':   mc_d_arr,  'mc_dd':  mc_dd_arr,
    'p_s5':   p_s5,  'p_s50':  p_s50,  's5_s50': s5_s50,
}
cols = {}
for nm, arr in BASE.items():
    cols[nm] = arr
    cols[f"{nm}_l5"] = np.roll(arr, 5)
feat_names = list(cols.keys())
X_all = np.column_stack(list(cols.values()))
print(f"  {len(feat_names)} features: {feat_names}")


# ── Walk-forward ──────────────────────────────────────────────────────────────

fold_edges = [int(n * f) for f in WF_SPLITS]   # [235881, 286889, 342050, 393135]
folds = []
for i in range(len(fold_edges) - 1):
    folds.append((fold_edges[i], fold_edges[i+1]))   # (is_end, oos_end)

print(f"\n{'='*72}")
print(f"3-Fold Walk-Forward  |  {n:,} bars  |  {len(HORIZONS)} horizons")
print(f"{'='*72}")

all_fold_imps = {nm: [] for nm in feat_names}

for fold_idx, (is_end, oos_end) in enumerate(folds):
    oos_start = is_end
    is_bars   = is_end
    oos_bars  = oos_end - oos_start
    is_date   = df['timestamp'].iloc[is_end-1].date()
    oos_s_dt  = df['timestamp'].iloc[oos_start].date()
    oos_e_dt  = df['timestamp'].iloc[oos_end-1].date()
    print(f"\n── Fold {fold_idx+1}  IS=0..{is_end:,} ({is_date})  OOS={oos_start:,}..{oos_end:,} ({oos_s_dt}→{oos_e_dt}, {oos_bars:,} bars) ──")

    for H in HORIZONS:
        fwd = np.empty(n); fwd[:] = np.nan
        fwd[:n-H] = (m5_c[H:] - m5_c[:n-H]) / PIP

        valid_is  = (np.arange(n) >= start+10) & (np.arange(n) < is_end)   & np.isfinite(fwd) & np.isfinite(X_all).all(axis=1)
        valid_oos = (np.arange(n) >= oos_start) & (np.arange(n) < oos_end) & np.isfinite(fwd) & np.isfinite(X_all).all(axis=1)

        X_is  = X_all[valid_is];  y_is  = fwd[valid_is]
        X_oos = X_all[valid_oos]; y_oos = fwd[valid_oos]
        cl_oos = m5_c[valid_oos]

        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=100, verbose=-1, n_jobs=4)
        model.fit(X_is, y_is)

        pred_is  = model.predict(X_is)
        pred_oos = model.predict(X_oos)
        r_is,  _ = stats.spearmanr(pred_is,  y_is)
        r_oos, _ = stats.spearmanr(pred_oos, y_oos)

        ic_flag = "🟢" if r_oos > 0.02 else ("🟡" if r_oos > 0 else "🔴")
        print(f"  H={H:2d} ({H*5:3d}min): IC_IS={r_is:+.3f}  {ic_flag}IC_OOS={r_oos:+.3f}", end="")

        best = None
        for thresh in [0.1, 0.2, 0.3, 0.5, 0.8, 1.2]:
            for mh in [H, H*2, 48]:
                if mh > 48: continue
                nt, mp, pnl = simulate_pred(pred_oos, cl_oos, mh, thresh)
                if nt >= 20 and (best is None or pnl > best[0]):
                    best = (pnl, nt, mp, thresh, mh)
        if best:
            pnl, nt, mp, thresh, mh = best
            sim_flag = "🟢" if pnl > 0 else "🔴"
            print(f"  {sim_flag}MOM={pnl:+.0f}p ({nt}t, {mp:+.2f}p/t)")
        else:
            print()

        for nm, imp in zip(feat_names, model.feature_importances_):
            all_fold_imps[nm].append(int(imp))


# ── Aggregate importance across all folds × horizons ─────────────────────────

print(f"\n{'='*72}")
print(f"Feature importance — avg across all folds × horizons ({len(folds)*len(HORIZONS)} runs):")
print(f"{'Rank':>4}  {'Feature':<14}  {'Avg':>6}  per-run: ", end="")
print("  ".join(f"F{f+1}H{H}" for f in range(len(folds)) for H in HORIZONS))
print("-" * 80)

avg_imps = [(nm, np.mean(vals)) for nm, vals in all_fold_imps.items()]
avg_imps.sort(key=lambda x: -x[1])
for rank, (nm, avg) in enumerate(avg_imps, 1):
    vals = all_fold_imps[nm]
    runs = "  ".join(f"{v:>4}" for v in vals)
    print(f"{rank:>4}  {nm:<14}  {avg:>6.0f}  {runs}")
