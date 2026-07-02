"""
MSP Phase 9 Full: Continuation vs Reversal direction model.
Features: EMA velocity (aligned to shock), session one-hot, shock z-score magnitude,
          adjacent band shock_z (D2, D4), spread ratio, ATR-normalized shock.
Target: price moves in D3 shock direction at 22 bars (110s).
WF: 5 temporal folds with embargo.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pywt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

BASE     = Path(__file__).resolve().parents[3]
S5_DIRS  = [BASE / "data" / "s5_ohlc", BASE / "data" / "s5_ba"]
OUT      = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
PIP_MAP  = {"EUR_USD": 0.0001, "EUR_JPY": 0.01, "GBP_JPY": 0.01}
WAVELET  = "db4"; DWT_LEVEL = 8; MAD_WIN = 1024; SHOCK_Z = 2.5; FWD_LAG = 22


def find_parquet(pair: str):
    for d in S5_DIRS:
        p = d / f"{pair}_S5_BA.parquet"
        if p.exists(): return p
    return None


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
    rm = s.rolling(w, center=True, min_periods=max(10, w//4)).median()
    ad = (s - rm).abs()
    rm2 = ad.rolling(w, center=True, min_periods=max(10, w//4)).median()
    return ((s - rm) / (1.4826 * rm2.clip(lower=1e-12))).fillna(0).values


def compute_dwt_bands(close: np.ndarray) -> dict:
    n = len(close)
    coeffs = pywt.wavedec(close, WAVELET, level=DWT_LEVEL, mode="periodization")
    out = {}
    for k in range(1, DWT_LEVEL + 1):
        name = f"D{DWT_LEVEL+1-k}"
        cd = coeffs[k]
        factor = (n + len(cd) - 1) // len(cd)
        out[name] = np.repeat(cd, factor)[:n]
    return out


def session_bucket(ts: pd.Series) -> np.ndarray:
    h = ts.dt.hour.values
    b = np.zeros(len(h), dtype=np.int8)
    b[(h >= 6) & (h < 9)]  = 1
    b[(h >= 9) & (h < 12)] = 2
    b[(h >= 12) & (h < 17)] = 3
    b[(h >= 17) & (h < 22)] = 4
    return b


def build_direction_features(df: pd.DataFrame, bands: dict,
                              d3_shock_idx: np.ndarray, pip: float):
    close = df["close"].values
    spread = df["spread_p"].values
    ts    = df["timestamp"]

    d3, d4, d2 = bands["D3"], bands["D4"], bands["D2"]
    z3 = mad_zscore(np.log(d3**2 + 1e-12), MAD_WIN)
    z4 = mad_zscore(np.log(d4**2 + 1e-12), MAD_WIN)
    z2 = mad_zscore(np.log(d2**2 + 1e-12), MAD_WIN)
    # EMA velocity (Kalman proxy)
    ret = np.diff(close, prepend=close[0]) / pip
    vel = pd.Series(ret).ewm(alpha=0.10, adjust=False).mean().values
    # ATR(20 bars)
    hi20 = pd.Series(close).rolling(20, min_periods=5).max()
    lo20 = pd.Series(close).rolling(20, min_periods=5).min()
    atr20 = ((hi20 - lo20) / pip).values
    # Session buckets
    sess = session_bucket(ts)
    # Spread ratio vs 200-bar rolling median
    sp_med = pd.Series(spread).rolling(200, min_periods=20).median()
    sp_ratio = (pd.Series(spread) / sp_med.clip(lower=1e-6)).values

    n_sh = len(d3_shock_idx)
    X_rows = []
    y = np.zeros(n_sh, dtype=np.int8)
    valid = 0
    for i, si in enumerate(d3_shock_idx):
        if si + FWD_LAG >= len(close): continue
        d3_sign = np.sign(d3[si])
        fwd_pip = (close[si + FWD_LAG] - close[si]) * d3_sign / pip
        y[valid] = 1 if fwd_pip > 0 else 0
        vel_al  = vel[si] * d3_sign           # positive = vel in shock dir
        d4_al   = d4[si] * d3_sign / (atr20[si] + 1e-9)  # D4 aligned + ATR-norm
        sess_oh = [int(sess[si] == k) for k in range(5)]
        X_rows.append([
            vel_al,                             # EMA vel aligned
            float(z3[si]),                      # D3 shock magnitude
            float(z4[si]),                      # D4 z-score
            float(z2[si]),                      # D2 z-score (finer)
            float(d4_al),                       # D4 coef aligned + ATR-norm
            float(sp_ratio[si]),                # spread wideness
        ] + sess_oh)
        valid += 1
    X = np.array(X_rows[:valid], dtype=np.float64)
    y = y[:valid]
    X = np.clip(X, -10, 10)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def walk_forward_direction(X, y, n_folds=5, model_type="logistic"):
    n = len(X)
    fold_size = n // (n_folds + 1)
    rows = []
    for fold in range(n_folds):
        tr_end  = fold_size * (fold + 1)
        te_st   = tr_end + fold_size // 5
        te_end  = te_st + fold_size
        if te_end > n: break
        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_te, y_te = X[te_st:te_end], y[te_st:te_end]
        if y_tr.sum() < 10 or y_te.sum() < 5: continue
        sc = StandardScaler().fit(X_tr)
        if model_type == "logistic":
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
        else:
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                             learning_rate=0.05, random_state=42)
        clf.fit(sc.transform(X_tr), y_tr)
        prob = clf.predict_proba(sc.transform(X_te))[:, 1]
        auc  = roc_auc_score(y_te, prob)
        thr  = np.percentile(prob, 90)
        top  = y_te[prob >= thr]
        lift = top.mean() / y_te.mean() if y_te.mean() > 0 else 1.0
        cont = top.mean()
        rows.append({"fold": fold, "model": model_type,
                     "auc": round(auc,4), "lift_p90": round(lift,3),
                     "cont_top_decile": round(cont,4),
                     "baseline": round(y_te.mean(),4)})
        print(f"  fold {fold}: AUC={auc:.3f} lift@P90={lift:.2f} "
              f"cont_top={cont*100:.1f}% baseline={y_te.mean()*100:.1f}%")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_results = []
    for pair, pip in PIP_MAP.items():
        path = find_parquet(pair)
        if path is None:
            print(f"  {pair}: no data, skip"); continue
        print(f"\n{'='*60}\n{pair}")
        df    = load_s5(path, pip)
        close = df["close"].values.astype(np.float64)
        print(f"  Computing DWT bands...")
        bands = compute_dwt_bands(close)
        d3    = bands["D3"]
        print(f"  Computing MAD z-scores...")
        log_e3 = np.log(d3**2 + 1e-12)
        z3    = mad_zscore(log_e3, MAD_WIN)
        shock_idx = np.where(z3 > SHOCK_Z)[0]
        shock_idx = shock_idx[shock_idx + FWD_LAG < len(close)]
        baseline_cont = (np.sign(d3[shock_idx]) *
                         (close[shock_idx + FWD_LAG] - close[shock_idx]) > 0).mean()
        print(f"  D3 shocks: {len(shock_idx):,}  baseline cont: {baseline_cont*100:.1f}%")
        X, y = build_direction_features(df, bands, shock_idx, pip)
        print(f"  X shape: {X.shape}  y_mean={y.mean():.3f}")
        for mt in ["logistic", "gbm"]:
            print(f"  [{mt}]")
            res = walk_forward_direction(X, y, model_type=mt)
            if res.empty: continue
            res["pair"] = pair
            all_results.append(res)
            auc_v = res["auc"].values
            lift_v = res["lift_p90"].values
            print(f"  AUC: mean={auc_v.mean():.4f} min={auc_v.min():.4f}")
            print(f"  Gate AUC>=0.60 all: {(auc_v>=0.60).all()}")
            print(f"  Gate lift@P90>=1.5 in 4/5: {(lift_v>=1.5).sum()>=4}")
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUT / "direction_model_results.csv", index=False)
    print(f"\nSaved: results/direction_model_results.csv")
    print("\n=== COMBINED SUMMARY ===")
    print(combined.groupby(["pair","model"])[["auc","lift_p90","cont_top_decile"]].mean().round(4))
