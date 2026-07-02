#!/usr/bin/env python3
"""
MSP Phase 7+9 Combined Signal Backtest — Volatility Harvesting

Strategy:
  - Entry signal: D3 shock detected (high-energy fine-scale event)
  - Direction:    Joint GBM model (P9+P7 features, trained walk-forward)
  - Filter:       Top-decile model confidence (P90) only
  - Exit:         TP=N pips (mid-price trigger) OR timeout (M S5 bars)

Simulation on EUR_USD S5 BA data:
  - Long entry at ask_c, exit at bid_c (for timeout) or mid (for TP)
  - Short entry at bid_c, exit at ask_c (for timeout) or mid (for TP)
  - No SL — pure timeout exit

Walk-forward: 5 folds, 20% embargo between IS and OOS.
"""
import warnings; warnings.filterwarnings("ignore")
import gc, sys
import numpy as np
import pandas as pd
import pywt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from itertools import product

BASE     = Path(__file__).resolve().parents[3]
S5_PATH  = BASE / "data" / "s5_ohlc" / "EUR_USD_S5_BA.parquet"
OUT_DIR  = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PIP       = 0.0001
WAVELET   = "db4"
DWT_LEVEL = 8
MAD_WIN   = 1024
SHOCK_Z   = 2.5
FWD_LAG   = 22   # S5 bars — Phase 7 direction horizon

TIME_WINDOWS = [1, 2, 6, 12, 24, 60, 180, 720]
TIME_LABELS  = ["5s","10s","30s","1m","2m","5m","15m","1h"]

N_FOLDS  = 5
EMBARGO  = 0.20   # fraction of fold size between IS end and OOS start

TP_SWEEP      = [5, 10, 15, 20]   # pips
TIMEOUT_SWEEP = [12, 36, 60, 120] # S5 bars  (1min, 3min, 5min, 10min)
CONF_PCT      = 90                 # model confidence percentile gate


# ── Feature computation (identical to joint_timing_direction.py) ──────────────

def mad_zscore(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x.astype(np.float64))
    rm  = s.rolling(w, center=True, min_periods=max(10, w // 4)).median()
    ad  = (s - rm).abs()
    rm2 = ad.rolling(w, center=True, min_periods=max(10, w // 4)).median()
    return ((s - rm) / (1.4826 * rm2.clip(lower=1e-12))).fillna(0).values


def compute_dwt_bands(close: np.ndarray) -> dict:
    n = len(close)
    coeffs = pywt.wavedec(close, WAVELET, level=DWT_LEVEL, mode="periodization")
    out = {}
    for k in range(1, DWT_LEVEL + 1):
        name = f"D{DWT_LEVEL + 1 - k}"
        cd = coeffs[k]
        factor = (n + len(cd) - 1) // len(cd)
        out[name] = np.repeat(cd, factor)[:n]
    return out


def compute_band_z_all(bands: dict) -> dict:
    return {name: mad_zscore(np.log(coef ** 2 + 1e-12), MAD_WIN)
            for name, coef in bands.items()}


def compute_time_shock_z(df: pd.DataFrame) -> dict:
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    out   = {}
    for w, lbl in zip(TIME_WINDOWS, TIME_LABELS):
        roll_hi = pd.Series(high).rolling(w, min_periods=w).max()
        roll_lo = pd.Series(low).rolling(w, min_periods=w).min()
        prev_c  = pd.Series(close).shift(w)
        hl  = roll_hi - roll_lo
        hpc = (roll_hi - prev_c).abs()
        lpc = (roll_lo - prev_c).abs()
        tr  = pd.Series(np.maximum(hl.values,
              np.maximum(hpc.fillna(hl).values, lpc.fillna(hl).values)))
        dur_min = w * (5.0 / 60.0)
        tr_rate = tr / PIP / dur_min
        out[lbl] = mad_zscore(tr_rate.values, MAD_WIN)
    return out


def session_bucket(ts: pd.Series) -> np.ndarray:
    h = ts.dt.hour.values
    b = np.zeros(len(h), dtype=np.int8)
    b[(h >= 6) & (h < 9)]   = 1
    b[(h >= 9) & (h < 12)]  = 2
    b[(h >= 12) & (h < 17)] = 3
    b[(h >= 17) & (h < 22)] = 4
    return b


def build_features(df: pd.DataFrame, bands: dict, band_z: dict,
                   time_z: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X (joint P9+P7 features), y (continuation label), shock_idx, and d3_sign.
    Returns arrays indexed over D3 shock events only.
    """
    close  = df["close"].values
    spread = ((df["ask_c"] - df["bid_c"]) / PIP).values
    ts     = df["timestamp"]

    d3 = bands["D3"]; d4 = bands["D4"]; d2 = bands["D2"]
    z3 = band_z["D3"]; z4 = band_z["D4"]; z2 = band_z["D2"]

    ret = np.diff(close, prepend=close[0]) / PIP
    vel = pd.Series(ret).ewm(alpha=0.10, adjust=False).mean().values
    hi20  = pd.Series(close).rolling(20, min_periods=5).max()
    lo20  = pd.Series(close).rolling(20, min_periods=5).min()
    atr20 = ((hi20 - lo20) / PIP).values
    sess  = session_bucket(ts)
    sp_med   = pd.Series(spread).rolling(200, min_periods=20).median()
    sp_ratio = (pd.Series(spread) / sp_med.clip(lower=1e-6)).values

    p7_arrays = [band_z[f"D{j}"] for j in range(1, DWT_LEVEL + 1)]
    p7_time   = [time_z[lbl] for lbl in TIME_LABELS]

    shock_idx = np.where(z3 > SHOCK_Z)[0]
    shock_idx = shock_idx[shock_idx + FWD_LAG < len(close)]

    X_rows, y_rows, signs = [], [], []
    for si in shock_idx:
        d3s  = np.sign(d3[si])
        fwd  = (close[si + FWD_LAG] - close[si]) * d3s / PIP
        cont = 1 if fwd > 0 else 0

        vel_al = vel[si] * d3s
        d4_al  = d4[si] * d3s / (atr20[si] + 1e-9)
        sess_oh = [int(sess[si] == k) for k in range(5)]
        p9 = [vel_al, float(z3[si]), float(z4[si]), float(z2[si]),
              float(d4_al), float(sp_ratio[si])] + sess_oh
        p7 = ([float(arr[si]) for arr in p7_arrays] +
              [float(arr[si]) for arr in p7_time])
        X_rows.append(p9 + p7)
        y_rows.append(cont)
        signs.append(d3s)

    X = np.clip(np.nan_to_num(np.array(X_rows, dtype=np.float64), nan=0.0), -10, 10)
    y = np.array(y_rows, dtype=np.int8)
    signs = np.array(signs, dtype=np.float32)
    return X, y, shock_idx, signs


# ── Trade simulation on S5 bars ───────────────────────────────────────────────

def simulate_trades(df: pd.DataFrame, shock_idx_oos: np.ndarray,
                    directions: np.ndarray, tp_pips: int, timeout_bars: int) -> dict:
    """
    For each qualifying shock event, simulate a trade:
      - Enter at next bar's ask_c (long) or bid_c (short)
      - TP: mid high/low reaches entry ± tp_pips
      - Timeout: exit at ask/bid close after timeout_bars
    Returns dict with trade-level stats.
    """
    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    bid_c  = df["bid_c"].values
    ask_c  = df["ask_c"].values
    n      = len(close)

    tp_level = tp_pips * PIP
    pnls = []

    for idx, si in enumerate(shock_idx_oos):
        entry_bar = si + 1   # enter at next S5 bar close
        if entry_bar >= n:
            continue
        direction = directions[idx]  # +1 long, -1 short
        if direction == 1:
            entry_px = ask_c[entry_bar]
            tp_px    = entry_px + tp_level
        else:
            entry_px = bid_c[entry_bar]
            tp_px    = entry_px - tp_level

        hit = False
        for j in range(1, timeout_bars + 1):
            bar = entry_bar + j
            if bar >= n:
                break
            if direction == 1 and high[bar] >= tp_px:
                pnl = tp_pips - (ask_c[entry_bar] - bid_c[entry_bar]) / PIP
                pnls.append(pnl)
                hit = True
                break
            elif direction == -1 and low[bar] <= tp_px:
                pnl = tp_pips - (ask_c[entry_bar] - bid_c[entry_bar]) / PIP
                pnls.append(pnl)
                hit = True
                break

        if not hit:
            bar = min(entry_bar + timeout_bars, n - 1)
            if direction == 1:
                exit_px = bid_c[bar]
                pnl = (exit_px - entry_px) / PIP
            else:
                exit_px = ask_c[bar]
                pnl = (entry_px - exit_px) / PIP
            pnls.append(pnl)

    if not pnls:
        return {"n": 0, "wr": 0.0, "mean_pnl": 0.0, "pd": 0.0}

    pnls = np.array(pnls)
    n_days = len(df) / (12 * 288)   # S5 bars per trading day (M5/5×12×288)
    n_days_oos = timeout_bars      # rough — will be overridden
    return {
        "n":        len(pnls),
        "wr":       (pnls > 0).mean(),
        "mean_pnl": pnls.mean(),
        "pd":       pnls.sum(),  # total pips (normalize later)
    }


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
                 shock_idx: np.ndarray, d3_signs: np.ndarray) -> pd.DataFrame:
    n_events  = len(shock_idx)
    fold_size = n_events // (N_FOLDS + 1)
    embargo   = int(fold_size * EMBARGO)

    n_s5_total = len(df)
    trading_days_total = n_s5_total / (12 * 288)

    rows = []

    for fold in range(N_FOLDS):
        tr_end = fold_size * (fold + 1)
        te_st  = tr_end + embargo
        te_end = te_st + fold_size
        if te_end > n_events:
            break

        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_te, y_te = X[te_st:te_end], y[te_st:te_end]
        shock_te   = shock_idx[te_st:te_end]
        signs_te   = d3_signs[te_st:te_end]

        sc  = StandardScaler().fit(X_tr)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                         learning_rate=0.05, random_state=42)
        clf.fit(sc.transform(X_tr), y_tr)
        prob = clf.predict_proba(sc.transform(X_te))[:, 1]

        threshold = np.percentile(prob, CONF_PCT)
        mask = prob >= threshold
        shock_sel = shock_te[mask]
        signs_sel = signs_te[mask]
        n_sel = mask.sum()

        # OOS window S5 bars: shock_idx[te_st] to shock_idx[te_end-1] + buffer
        if len(shock_sel) == 0:
            continue
        oos_s5_start = int(shock_te[0])
        oos_s5_end   = int(shock_te[-1]) + max(TIMEOUT_SWEEP) + 2
        oos_s5_end   = min(oos_s5_end, n_s5_total)
        df_oos = df.iloc[oos_s5_start:oos_s5_end].reset_index(drop=True)

        # Adjust shock indices relative to oos slice
        shock_sel_rel = shock_sel - oos_s5_start
        valid = (shock_sel_rel >= 0) & (shock_sel_rel < len(df_oos))
        shock_sel_rel = shock_sel_rel[valid]
        signs_sel_adj = signs_sel[valid]

        oos_days = (oos_s5_end - oos_s5_start) / (12 * 288)

        for tp, timeout in product(TP_SWEEP, TIMEOUT_SWEEP):
            res = simulate_trades(df_oos, shock_sel_rel, signs_sel_adj, tp, timeout)
            rows.append({
                "fold":        fold,
                "tp_pips":     tp,
                "timeout_s5":  timeout,
                "timeout_min": round(timeout * 5 / 60, 1),
                "n_trades":    res["n"],
                "wr":          round(res["wr"] * 100, 1),
                "mean_pnl":    round(res["mean_pnl"], 2),
                "total_pips":  round(res["pd"], 1),
                "pd_pips":     round(res["pd"] / oos_days, 2) if oos_days > 0 else 0.0,
                "oos_days":    round(oos_days, 1),
                "n_sel_events": n_sel,
            })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading EUR_USD S5 BA data...")
    df = pd.read_parquet(S5_PATH).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "close" not in df.columns:
        df["close"] = (df["bid_c"] + df["ask_c"]) / 2
    n = len(df)
    sp_med = ((df["ask_c"] - df["bid_c"]) / PIP).median()
    print(f"  {n:,} bars  spread_med={sp_med:.2f}p  "
          f"dates: {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")

    close = df["close"].values.astype(np.float64)

    print("\nComputing DWT bands + MAD z-scores (all 8 bands)...")
    bands  = compute_dwt_bands(close)
    band_z = compute_band_z_all(bands)

    print("Computing time-window TR-rate shock_z (8 windows)...")
    time_z = compute_time_shock_z(df)

    print("\nBuilding feature matrix at D3 shock events...")
    X, y, shock_idx, d3_signs = build_features(df, bands, band_z, time_z)
    baseline = y.mean()
    print(f"  D3 shocks: {len(shock_idx):,}  baseline_cont: {baseline*100:.1f}%  "
          f"X.shape={X.shape}")

    del bands, band_z, time_z; gc.collect()

    print(f"\nWalk-forward backtest ({N_FOLDS} folds, {int(EMBARGO*100)}% embargo)...")
    print(f"  TP sweep: {TP_SWEEP}p  Timeout sweep: {TIMEOUT_SWEEP} S5bars  "
          f"Confidence gate: P{CONF_PCT}\n")

    results = run_backtest(df, X, y, shock_idx, d3_signs)

    out_path = OUT_DIR / "msp_p7p9_backtest.csv"
    results.to_csv(out_path, index=False)

    print("\n=== SUMMARY (mean across folds) ===")
    summary = (results.groupby(["tp_pips", "timeout_s5"])[
        ["n_trades", "wr", "mean_pnl", "pd_pips", "oos_days"]]
        .mean().round(2))
    summary["timeout_min"] = (summary.index.get_level_values("timeout_s5") * 5 / 60).round(1)
    print(summary.to_string())

    print(f"\n=== BEST CONFIGS (pd_pips > 0, sorted) ===")
    best = (results.groupby(["tp_pips", "timeout_s5"])[
        ["n_trades", "wr", "mean_pnl", "pd_pips"]]
        .mean().round(2)
        .query("pd_pips > 0")
        .sort_values("pd_pips", ascending=False))
    if len(best):
        best["timeout_min"] = (best.index.get_level_values("timeout_s5") * 5 / 60).round(1)
        print(best.to_string())
    else:
        print("  No positive p/d configs found.")

    print(f"\nFull results saved to {out_path}")

    # Per-fold detail for best config
    if len(best):
        tp_b, tmo_b = best.index[0]
        print(f"\n=== PER-FOLD DETAIL: tp={tp_b}p timeout={tmo_b}bars "
              f"({tmo_b*5/60:.1f}min) ===")
        fold_detail = results[(results.tp_pips == tp_b) & (results.timeout_s5 == tmo_b)]
        for _, r in fold_detail.iterrows():
            flag = "🟢" if r.pd_pips > 0 else "🔴"
            print(f"  {flag} fold {int(r.fold)}: n={int(r.n_trades)}  "
                  f"WR={r.wr:.1f}%  pnl/t={r.mean_pnl:+.2f}p  "
                  f"p/d={r.pd_pips:+.2f}  oos={r.oos_days:.0f}d")


if __name__ == "__main__":
    main()
