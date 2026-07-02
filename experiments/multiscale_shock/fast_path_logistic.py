"""
Phase 9 Fast Path: Kalman velocity sign + session bucket →
logistic regression for D3 shock continuation at 110s.

Baseline from run_s5_msp.py: EUR_USD D3 continuation = 58.1% @ 22bars.
Goal: AUC > 0.60 and conditional continuation rate > 65%.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pywt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE     = Path(__file__).resolve().parents[3]
DATA_DIR = BASE / "data" / "s5_ohlc"
PIP_MAP  = {"EUR_USD": 0.0001, "EUR_JPY": 0.01, "GBP_JPY": 0.01}
WAVELET  = "db4"
DWT_LEVEL = 8
MAD_WIN   = 1024
SHOCK_Z   = 2.5
FWD_LAG   = 22          # bars → 110 seconds

def load_s5(path: Path, pip: float) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    if "close" not in df.columns:
        df["close"] = (df["bid_c"] + df["ask_c"]) / 2
        df["open"]  = (df["bid_o"] + df["ask_o"]) / 2
    df["spread_p"] = (df["ask_c"] - df["bid_c"]) / pip
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

def mad_zscore(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x.astype(np.float64))
    rm = s.rolling(w, center=True, min_periods=max(10, w // 4)).median()
    ad = (s - rm).abs()
    rm2 = ad.rolling(w, center=True, min_periods=max(10, w // 4)).median()
    return ((s - rm) / (1.4826 * rm2.clip(lower=1e-12))).fillna(0).values

def dwt_bands(signal: np.ndarray) -> dict:
    n = len(signal)
    coeffs = pywt.wavedec(signal, WAVELET, level=DWT_LEVEL, mode="periodization")
    out = {}
    for k in range(1, DWT_LEVEL + 1):
        name = f"D{DWT_LEVEL + 1 - k}"
        cd = coeffs[k]
        factor = (n + len(cd) - 1) // len(cd)
        out[name] = np.repeat(cd, factor)[:n]
    return out

def session_bucket(ts_series: pd.Series) -> np.ndarray:
    """
    0=Asian(22-06 UTC), 1=LondonOpen(06-09), 2=London(09-12),
    3=Overlap(12-17), 4=NYafternoon(17-22)
    """
    h = ts_series.dt.hour.values
    bucket = np.zeros(len(h), dtype=np.int8)
    bucket[(h >= 6)  & (h < 9)]  = 1
    bucket[(h >= 9)  & (h < 12)] = 2
    bucket[(h >= 12) & (h < 17)] = 3
    bucket[(h >= 17) & (h < 22)] = 4
    return bucket

def ema_velocity(close: np.ndarray, pip: float, alpha: float = 0.10) -> np.ndarray:
    """EMA of pip-returns — proxy for Kalman filter velocity."""
    ret = np.diff(close, prepend=close[0]) / pip   # pip changes
    return pd.Series(ret).ewm(alpha=alpha, adjust=False).mean().values

def walk_forward_logistic(X: np.ndarray, y: np.ndarray,
                          n_folds: int = 5) -> pd.DataFrame:
    n = len(X)
    fold_size = n // (n_folds + 1)
    rows = []
    for fold in range(n_folds):
        train_end  = fold_size * (fold + 1)
        test_start = train_end + fold_size // 5   # 20% embargo
        test_end   = test_start + fold_size
        if test_end > n:
            break
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        if y_tr.mean() == 0 or y_tr.mean() == 1:
            continue
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(scaler.transform(X_tr), y_tr)
        prob = clf.predict_proba(scaler.transform(X_te))[:, 1]
        auc  = roc_auc_score(y_te, prob)
        # Top-decile lift
        thr  = np.percentile(prob, 90)
        top  = y_te[prob >= thr]
        lift = top.mean() / y_te.mean() if y_te.mean() > 0 else 1.0
        # Continuation rate in top-decile
        cont = top.mean()
        rows.append({"fold": fold, "train_n": train_end,
                     "auc": round(auc, 4), "lift_p90": round(lift, 3),
                     "cont_top_decile": round(cont, 4)})
        print(f"  fold {fold}: AUC={auc:.3f}  lift@P90={lift:.2f}  "
              f"cont%={cont*100:.1f}%  train={train_end:,}")
    return pd.DataFrame(rows)

def run_pair(pair: str, pip: float) -> pd.DataFrame | None:
    path = BASE / "data" / "s5_ohlc" / f"{pair}_S5_BA.parquet"
    if not path.exists():
        path = BASE / "data" / "s5_ba" / f"{pair}_S5_BA.parquet"
    if not path.exists():
        print(f"  {pair}: no data, skip"); return None

    print(f"\n{'='*60}\n  {pair}")
    df = load_s5(path, pip)
    close = df["close"].values.astype(np.float64)
    n = len(close)
    spread_med = df["spread_p"].median()
    rt_cost    = 2 * spread_med
    barrier    = 3 * rt_cost
    print(f"  n={n:,}  spread={spread_med:.2f}p  RT={rt_cost:.2f}p  barrier={barrier:.2f}p")

    # DWT D3 band
    print(f"  Computing DWT...")
    bands  = dwt_bands(close)
    d3     = bands["D3"]
    log_e3 = np.log(d3 ** 2 + 1e-12)
    print(f"  Computing MAD z-score...")
    z3     = mad_zscore(log_e3, MAD_WIN)
    shock  = z3 > SHOCK_Z

    # Features at each shock bar
    vel    = ema_velocity(close, pip)
    sess   = session_bucket(df["timestamp"])
    z_raw  = z3                              # shock magnitude

    # Shock direction: D3 coef sign
    d3_sign = np.sign(d3)

    # Target: does price move in shock direction at FWD_LAG bars?
    shock_idx = np.where(shock)[0]
    shock_idx = shock_idx[shock_idx + FWD_LAG < n]
    n_sh = len(shock_idx)
    print(f"  D3 shocks: {n_sh:,}  ({n_sh/n*100:.3f}%)")

    # Build feature matrix
    y = np.zeros(n_sh, dtype=np.int8)
    X_rows = []
    for i, si in enumerate(shock_idx):
        fwd_move = (close[si + FWD_LAG] - close[si]) * d3_sign[si] / pip
        y[i] = 1 if fwd_move > 0 else 0
        vel_sign  = np.sign(vel[si])
        sess_val  = sess[si]
        z_val     = z_raw[si]
        # One-hot sessions 0-4
        sess_ohe  = [int(sess_val == k) for k in range(5)]
        # vel_sign aligned to shock direction
        vel_aligned = vel_sign * d3_sign[si]   # +1 = vel in shock direction
        X_rows.append([vel_aligned, z_val, vel[si] * d3_sign[si]] + sess_ohe)

    X = np.array(X_rows, dtype=np.float64)
    feature_names = ["vel_aligned", "shock_z", "vel_magnitude",
                     "sess_0_asian", "sess_1_lonopen", "sess_2_london",
                     "sess_3_overlap", "sess_4_nyafternoon"]
    print(f"  Baseline continuation: {y.mean()*100:.1f}%")
    print(f"  Features: {feature_names}")

    print(f"  Walk-forward logistic (5-fold temporal):")
    results = walk_forward_logistic(X, y, n_folds=5)
    print(f"\n  Summary:")
    print(results.to_string(index=False))
    auc_all = results["auc"].values
    print(f"\n  AUC: mean={auc_all.mean():.4f}  min={auc_all.min():.4f}")
    print(f"  All folds AUC>0.60: {(auc_all > 0.60).all()}")
    print(f"  All folds lift@P90>1.5: {(results['lift_p90'] > 1.5).all()}")

    # Save results
    OUT = Path(__file__).parent / "results"
    OUT.mkdir(exist_ok=True)
    results["pair"] = pair
    results.to_csv(OUT / f"fast_path_{pair}.csv", index=False)
    print(f"  Saved → results/fast_path_{pair}.csv")
    return results

if __name__ == "__main__":
    all_results = []
    for pair, pip in PIP_MAP.items():
        r = run_pair(pair, pip)
        if r is not None:
            all_results.append(r)
    print("\n" + "="*60)
    print("COMBINED RESULTS:")
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(combined.to_string(index=False))
