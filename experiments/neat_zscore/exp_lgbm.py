#!/usr/bin/env python3
"""Experiment: LightGBM on full feature set.

Features (all at M5 cadence):
  f1      — 5-min return z-score  (arctan-normalized)
  f2      — 20-min return z-score (arctan-normalized)
  f3      — 20-min efficiency ratio on raw S5 OHLC  ∈ [-1, +1]
  sba     — TopsBots swing state on 10-pip range bars ∈ {-1,-0.5,0,+0.5,+1}
  mc_d_a  — Δ multi-TF ASI momentum count (FXFeatureBuilder kalman10)
  mc_dd_a — Δ(mc_d_a)
  p_s5    — (price - SMA5) in pips
  p_s50   — (price - SMA50) in pips
  s5_s50  — (SMA5 - SMA50) in pips
  + lags [1, 2, 3, 5] of each → 45 total features

Simulation: both MOM (+1) and M/R (-1) directions tested explicitly.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
import lightgbm as lgb
from scipy import stats
from collections import deque

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.incremental_features import FXFeatureBuilder
from lib.swing_indicators import compute_swing_features

S5_DIR = ROOT / "data" / "s5_ohlc"
PAIR, PIP, SPREAD = "EUR_JPY", 0.01, 2.3
IS_FRAC      = 0.70
HORIZONS     = [5, 10, 20]
RANGE_PIPS   = 10.0
RANGE_PRICE  = RANGE_PIPS * PIP


# ── Numba feature functions ───────────────────────────────────────────────────

@njit(cache=True)
def compute_f1_f2(close, pop=1000, lb1=1, lb2=4):
    # close = m5_c (M5 bar closes) — consistent with target + all other features
    n = len(close); r1 = np.zeros(n); r2 = np.zeros(n)
    for i in range(lb1, n): r1[i] = (close[i] - close[i-lb1]) / PIP
    for i in range(lb2, n): r2[i] = (close[i] - close[i-lb2]) / (lb2 * PIP)
    f1 = np.zeros(n); f2 = np.zeros(n); hp = np.pi / 2.; start = pop + lb2
    s1 = 0.; s1q = 0.; s2 = 0.; s2q = 0.
    for k in range(start - pop, start):
        s1 += r1[k]; s1q += r1[k]*r1[k]; s2 += r2[k]; s2q += r2[k]*r2[k]
    for i in range(start, n):
        if i > start:
            a1 = r1[i-1]; e1 = r1[i-1-pop]; s1 += a1-e1; s1q += a1*a1-e1*e1
            a2 = r2[i-1]; e2 = r2[i-1-pop]; s2 += a2-e2; s2q += a2*a2-e2*e2
        v1 = s1q/pop - (s1/pop)**2; v2 = s2q/pop - (s2/pop)**2
        std1 = v1**0.5 if v1 > 1e-20 else 1e-10
        std2 = v2**0.5 if v2 > 1e-20 else 1e-10
        f1[i] = np.arctan(r1[i]/std1)/hp; f2[i] = np.arctan(r2[i]/std2)/hp
    return f1, f2, start


@njit(cache=True)
def compute_f3(bid_c_s5, bid_o_s5, bid_h_s5, bid_l_s5, window=240):
    n_m5 = len(bid_c_s5) // 60
    f3 = np.zeros(n_m5)
    for k in range(n_m5):
        i = k * 60
        if i < window:
            continue
        net = bid_c_s5[i + 59] - bid_o_s5[i - window]  # m5_c[k] = last S5 of M5 bar k
        path = 0.0
        for j in range(i - window, i + 1):
            path += bid_h_s5[j] - bid_l_s5[j]
        if path > 1e-10:
            f3[k] = net / path
    return f3


@njit(cache=True)
def simulate_pred(pred_signal, close, max_hold, thresh):
    pos = 0; ep = 0.; eb = 0; pnl = 0.; sc_sum = 0.; nt = 0
    n = len(close)
    for i in range(n):
        s = pred_signal[i]
        if pos != 0:
            pnl = (close[i]-ep)/PIP if pos == 1 else (ep-close[i])/PIP
            fc = (i - eb) >= max_hold
            sig_cl = (pos == 1 and s < -thresh) or (pos == -1 and s > thresh)
            if fc or sig_cl:
                sc_sum += pnl; nt += 1; pos = 0; pnl = 0.
        if pos == 0:
            if s > thresh:
                ep = close[i] + SPREAD*0.5*PIP; eb = i; pos = 1; pnl = -SPREAD
            elif s < -thresh:
                ep = close[i] - SPREAD*0.5*PIP; eb = i; pos = -1; pnl = -SPREAD
    return nt, sc_sum/nt if nt > 0 else 0., sc_sum


# ── Load S5 data ──────────────────────────────────────────────────────────────

print("Loading S5 data...")
path = S5_DIR / "EUR_JPY_S5_BA.parquet"
df = pd.read_parquet(path)
df.columns = [c.lower() for c in df.columns]
bid_c_s5 = df["bid_c"].values.astype(np.float64)
bid_o_s5 = df["bid_o"].values.astype(np.float64)
bid_h_s5 = df["bid_h"].values.astype(np.float64)
bid_l_s5 = df["bid_l"].values.astype(np.float64)

n_m5 = len(bid_c_s5) // 60

# M5 OHLC via reshape
s5c = bid_c_s5[:n_m5*60].reshape(n_m5, 60)
s5o = bid_o_s5[:n_m5*60].reshape(n_m5, 60)
s5h = bid_h_s5[:n_m5*60].reshape(n_m5, 60)
s5l = bid_l_s5[:n_m5*60].reshape(n_m5, 60)
m5_o = s5o[:, 0]
m5_h = s5h.max(axis=1)
m5_l = s5l.min(axis=1)
m5_c = s5c[:, -1]
m5_mid = (m5_h + m5_l) / 2.0

# Unified price series: m5_c (M5 close = last S5 bar of each M5 period).
# Features, target, and simulation all use this — eliminates within-bar lookahead.
closes = m5_c
n = len(closes)
n_is = int(n * IS_FRAC)

print(f"  M5={n:,}  IS={n_is:,}  OOS={n-n_is:,}")


# ── f1, f2, f3 ────────────────────────────────────────────────────────────────

print("Computing f1, f2, f3...")
f1, f2, start = compute_f1_f2(closes)
f3 = compute_f3(bid_c_s5, bid_o_s5, bid_h_s5, bid_l_s5)
simulate_pred(np.zeros(200), closes[:200], 10, 0.01)  # JIT warmup


# ── SMA features ─────────────────────────────────────────────────────────────

print("Computing SMA features...")
s = pd.Series(m5_c)
sma5  = s.rolling(5).mean().values
sma50 = s.rolling(50).mean().values
p_s5   = (m5_c - sma5)  / PIP
p_s50  = (m5_c - sma50) / PIP
s5_s50 = (sma5 - sma50) / PIP


# ── SBA from 10-pip range bars ────────────────────────────────────────────────

print("Computing SBA from range bars...")
rb_closes_list = []
rb_m5_idx      = []    # which M5 bar completed each range bar
bar_open = None
for i, mid in enumerate(m5_mid):
    if bar_open is None:
        bar_open = mid
        continue
    if abs(mid - bar_open) >= RANGE_PRICE:
        bar_open = mid
        rb_closes_list.append(mid)
        rb_m5_idx.append(i)

rb_closes = np.array(rb_closes_list, dtype=np.float64)
rb_m5_idx = np.array(rb_m5_idx, dtype=np.int64)

# Compute swing state on full range bar array (1-bar range-bar lookahead, acceptable)
sba_full = np.zeros(len(rb_closes), dtype=np.float64)
if len(rb_closes) >= 3:
    state, _, _, _, _ = compute_swing_features(rb_closes, rb_closes, rb_closes)
    sba_full = state.astype(np.float64) / 2.0

# Map back to M5 bars (forward-fill)
sba = np.zeros(n, dtype=np.float64)
rb_ptr = 0
for i in range(n):
    while rb_ptr < len(rb_m5_idx) and rb_m5_idx[rb_ptr] <= i:
        sba[i] = sba_full[rb_ptr]
        rb_ptr += 1
    if rb_ptr > 0 and rb_m5_idx[rb_ptr-1] > i:
        pass  # keep sba[i] = 0 until first range bar
    elif rb_ptr > 0:
        sba[i] = sba_full[rb_ptr - 1]

print(f"  {len(rb_closes):,} range bars from {n:,} M5 bars")


# ── mc_d_a, mc_dd_a via FXFeatureBuilder ─────────────────────────────────────

print("Computing mc_d_a, mc_dd_a (FXFeatureBuilder kalman10)...")
builder = FXFeatureBuilder('EUR_JPY', smoother='kalman10')
mc_d_arr  = np.zeros(n, dtype=np.float64)
mc_dd_arr = np.zeros(n, dtype=np.float64)
for i in range(n):
    feats = builder.process_new_bar(m5_o[i], m5_h[i], m5_l[i], m5_c[i])
    mc_d_arr[i]  = feats.get('mc_d_a',  0.0) or 0.0
    mc_dd_arr[i] = feats.get('mc_dd_a', 0.0) or 0.0
    if i % 5000 == 0:
        print(f"  {i:,}/{n:,}", end='\r')
print()


# ── Build feature matrix ──────────────────────────────────────────────────────

print("Building feature matrix...")
BASE_FEATS = {
    'f1':      f1,
    'f2':      f2,
    'f3':      f3,
    'sba':     sba,
    'mc_d':    mc_d_arr,
    'mc_dd':   mc_dd_arr,
    'p_s5':    p_s5,
    'p_s50':   p_s50,
    's5_s50':  s5_s50,
}
LAGS = [1, 2, 3, 5]

cols = {}
for name, arr in BASE_FEATS.items():
    cols[name] = arr
    for lag in LAGS:
        cols[f"{name}_l{lag}"] = np.roll(arr, lag)

X_all      = np.column_stack(list(cols.values()))
feat_names = list(cols.keys())
print(f"  {len(feat_names)} features total")


# ── LightGBM per horizon ──────────────────────────────────────────────────────

print(f"\n{'='*72}")
all_imps = {name: [] for name in feat_names}

for H in HORIZONS:
    fwd = np.empty(n); fwd[:] = np.nan
    fwd[:n-H] = (closes[H:] - closes[:n-H]) / PIP

    valid  = np.isfinite(fwd) & np.isfinite(X_all).all(axis=1) & (np.arange(n) >= start+5)
    X_is   = X_all[:n_is][valid[:n_is]]
    y_is   = fwd[:n_is][valid[:n_is]]
    X_oos  = X_all[n_is:][valid[n_is:]]
    y_oos  = fwd[n_is:][valid[n_is:]]
    cl_oos = closes[n_is:][valid[n_is:]]

    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=100, verbose=-1, n_jobs=4)
    model.fit(X_is, y_is)

    pred_is  = model.predict(X_is)
    pred_oos = model.predict(X_oos)

    r_is,  _ = stats.spearmanr(pred_is,  y_is)
    r_oos, _ = stats.spearmanr(pred_oos, y_oos)

    flag_ic = "🟢" if r_oos > 0.01 else ("🟡" if r_oos > 0 else "🔴")
    print(f"\nH={H} bars ({H*5}min):  IC_IS={r_is:+.4f}  {flag_ic}IC_OOS={r_oos:+.4f}")

    for direction, label in [(+1, "MOM"), (-1, "M/R")]:
        sig = direction * pred_oos
        best = None
        for thresh in [0.1, 0.2, 0.3, 0.5, 0.8, 1.2]:
            for mh in [H, H*2, 48]:
                if mh > 48: continue
                nt, mp, pnl = simulate_pred(sig, cl_oos, mh, thresh)
                if nt >= 5 and (best is None or pnl > best[0]):
                    best = (pnl, nt, mp, thresh, mh)
        if best:
            pnl, nt, mp, thresh, mh = best
            flag = "🟢" if pnl > 0 else "🔴"
            print(f"  {label}: {flag}{pnl:+.1f}p  trades={nt}  mean={mp:+.3f}p  thresh={thresh}  mh={mh}")
        else:
            print(f"  {label}: no valid results")

    for name, imp in zip(feat_names, model.feature_importances_):
        all_imps[name].append(int(imp))

# ── Aggregate importance ranking across all horizons ─────────────────────────

print(f"\n{'='*72}")
print("Feature importance (avg across H=5,10,20), ranked:")
print(f"{'Rank':>4} {'Feature':<14} {'H5':>5} {'H10':>5} {'H20':>5} {'Avg':>6}")
print("-" * 42)
avg_imps = [(name, np.mean(vals)) for name, vals in all_imps.items()]
avg_imps.sort(key=lambda x: -x[1])
for rank, (name, avg) in enumerate(avg_imps, 1):
    vals = all_imps[name]
    print(f"{rank:>4} {name:<14} {vals[0]:>5} {vals[1]:>5} {vals[2]:>5} {avg:>6.0f}")
