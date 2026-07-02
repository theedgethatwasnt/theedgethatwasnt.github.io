"""
MSP Joint Timing+Direction Model — inline feature computation (no DuckDB join).

Computes all features from S5 parquet in memory:
  Phase 9 (11): vel_aligned, D2/D3/D4 z-scores, D4 ATR-aligned, spread ratio, session OHE
  Phase 7 (21): D1..D8 shock_z  + 5s/10s/30s/1m/2m/5m/15m/1h TR-rate shock_z

At each D3 shock event: does the FULL multi-scale profile (coarser band energies +
time-window activity) predict continuation better than Phase 9 features alone?

Key hypothesis: if D4/D5 also elevated (multi-scale shock) → continuation.
If only D1-D3 elevated (isolated fine-scale event) → reversal.
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

BASE    = Path(__file__).resolve().parents[3]
S5_DIRS = [BASE / "data" / "s5_ohlc", BASE / "data" / "s5_ba"]
OUT     = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

PIP_MAP   = {"EUR_USD": 0.0001, "EUR_JPY": 0.01, "GBP_JPY": 0.01}
WAVELET   = "db4"
DWT_LEVEL = 8
MAD_WIN   = 1024
SHOCK_Z   = 2.5
FWD_LAG   = 22   # bars → 110s

TIME_WINDOWS = [1, 2, 6, 12, 24, 60, 180, 720]
TIME_LABELS  = ["5s","10s","30s","1m","2m","5m","15m","1h"]


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
        df["high"]  = df.get("ask_h", df["ask_c"])
        df["low"]   = df.get("bid_l", df["bid_c"])
    df["spread_p"] = (df["ask_c"] - df["bid_c"]) / pip
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def mad_zscore(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x.astype(np.float64))
    rm  = s.rolling(w, center=True, min_periods=max(10, w//4)).median()
    ad  = (s - rm).abs()
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


def compute_band_z_all(bands: dict) -> dict:
    """MAD z-score for all DWT bands (log-energy)."""
    return {name: mad_zscore(np.log(coef**2 + 1e-12), MAD_WIN)
            for name, coef in bands.items()}


def compute_time_shock_z(df: pd.DataFrame, pip: float) -> dict:
    """
    Rolling TR-rate MAD z-score for each time window.
    Returns dict {label: z_array} length=n each.
    """
    close = df["close"].values
    high  = df["high"].values if "high" in df.columns else close
    low   = df["low"].values  if "low"  in df.columns else close
    open_ = df["open"].values
    out   = {}
    for w, lbl in zip(TIME_WINDOWS, TIME_LABELS):
        roll_hi  = pd.Series(high).rolling(w, min_periods=w).max()
        roll_lo  = pd.Series(low).rolling(w, min_periods=w).min()
        prev_c   = pd.Series(close).shift(w)
        hl   = roll_hi - roll_lo
        hpc  = (roll_hi - prev_c).abs()
        lpc  = (roll_lo - prev_c).abs()
        tr   = pd.Series(np.maximum(hl.values,
               np.maximum(hpc.fillna(hl).values, lpc.fillna(hl).values)))
        dur_min = w * (5.0 / 60.0)
        tr_rate = tr / pip / dur_min
        z = mad_zscore(tr_rate.values, MAD_WIN)
        out[lbl] = z
    return out


def session_bucket(ts: pd.Series) -> np.ndarray:
    h = ts.dt.hour.values
    b = np.zeros(len(h), dtype=np.int8)
    b[(h >= 6) & (h < 9)]   = 1
    b[(h >= 9) & (h < 12)]  = 2
    b[(h >= 12) & (h < 17)] = 3
    b[(h >= 17) & (h < 22)] = 4
    return b


def build_joint_matrix(df: pd.DataFrame, bands: dict, band_z: dict,
                        time_z: dict, shock_idx: np.ndarray, pip: float):
    """
    Phase 9 + Phase 7 features at each D3 shock event.
    Returns X_all (26 feat), X_p9 (11 feat), y, meta (list of [si, fwd_abs, cont]).
    """
    close  = df["close"].values
    spread = df["spread_p"].values
    ts     = df["timestamp"]

    d3, d4, d2 = bands["D3"], bands["D4"], bands["D2"]
    z3 = band_z["D3"]; z4 = band_z["D4"]; z2 = band_z["D2"]

    ret = np.diff(close, prepend=close[0]) / pip
    vel = pd.Series(ret).ewm(alpha=0.10, adjust=False).mean().values
    hi20  = pd.Series(close).rolling(20, min_periods=5).max()
    lo20  = pd.Series(close).rolling(20, min_periods=5).min()
    atr20 = ((hi20 - lo20) / pip).values
    sess  = session_bucket(ts)
    sp_med = pd.Series(spread).rolling(200, min_periods=20).median()
    sp_ratio = (pd.Series(spread) / sp_med.clip(lower=1e-6)).values

    # Pre-extract Phase 7 timing arrays
    p7_band_arrays   = [band_z[f"D{j}"] for j in range(1, DWT_LEVEL+1)]  # D1..D8
    p7_time_arrays   = [time_z[lbl] for lbl in TIME_LABELS]               # 8 windows

    X_p9, X_p7, y_arr, meta = [], [], [], []

    for si in shock_idx:
        if si + FWD_LAG >= len(close): continue
        d3s    = np.sign(d3[si])
        fwd    = (close[si + FWD_LAG] - close[si]) * d3s / pip
        cont   = 1 if fwd > 0 else 0

        vel_al  = vel[si] * d3s
        d4_al   = d4[si] * d3s / (atr20[si] + 1e-9)
        sess_oh = [int(sess[si] == k) for k in range(5)]

        p9 = [vel_al, float(z3[si]), float(z4[si]), float(z2[si]),
              float(d4_al), float(sp_ratio[si])] + sess_oh

        # Phase 7: D1..D8 shock_z + 8 time-window TR-rate shock_z at bar si
        p7 = ([float(arr[si]) for arr in p7_band_arrays] +
              [float(arr[si]) for arr in p7_time_arrays])

        X_p9.append(p9)
        X_p7.append(p7)
        y_arr.append(cont)
        meta.append([si, abs(fwd), cont])

    X_p9_ = np.clip(np.nan_to_num(np.array(X_p9, dtype=np.float64), nan=0.0), -10, 10)
    X_p7_ = np.clip(np.nan_to_num(np.array(X_p7, dtype=np.float64), nan=0.0), -10, 10)
    X_all = np.hstack([X_p9_, X_p7_])
    y     = np.array(y_arr, dtype=np.int8)
    return X_all, X_p9_, y, np.array(meta)


def walk_forward(X, y, meta, n_folds, model_type, rt_cost):
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
        top_mask = prob >= thr
        top_y    = y_te[top_mask]
        lift     = top_y.mean() / y_te.mean() if y_te.mean() > 0 else 1.0
        cont_top = top_y.mean()
        fwd_top  = meta[te_st:te_end, 1][top_mask]
        mean_abs = fwd_top.mean() if len(fwd_top) > 0 else 0.0
        ev_pre   = (2 * cont_top - 1) * mean_abs
        ev_post  = ev_pre - rt_cost
        rows.append({"fold": fold, "model": model_type,
                     "auc": round(auc, 4), "lift_p90": round(lift, 3),
                     "cont_top": round(cont_top, 4), "baseline": round(y_te.mean(), 4),
                     "mean_abs_top": round(mean_abs, 2),
                     "ev_pre": round(ev_pre, 3), "ev_post": round(ev_post, 3),
                     "n_top": int(top_mask.sum())})
        print(f"  fold {fold}: AUC={auc:.3f}  lift={lift:.2f}  "
              f"cont={cont_top*100:.1f}%  EV_pre={ev_pre:+.2f}p  "
              f"EV_post={ev_post:+.2f}p  n_top={top_mask.sum()}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_results = []

    for pair, pip in PIP_MAP.items():
        path = find_parquet(pair)
        if path is None: print(f"  {pair}: skip"); continue
        print(f"\n{'='*60}\n{pair}")
        df    = load_s5(path, pip)
        close = df["close"].values.astype(np.float64)
        n     = len(close)
        sp_med  = df["spread_p"].median()
        rt_cost = 2 * sp_med
        print(f"  n={n:,}  spread={sp_med:.2f}p  RT={rt_cost:.2f}p")

        print(f"  DWT all bands + MAD z-scores...")
        bands  = compute_dwt_bands(close)
        band_z = compute_band_z_all(bands)

        print(f"  Time-window TR-rate shock_z ({len(TIME_WINDOWS)} windows)...")
        time_z = compute_time_shock_z(df, pip)

        d3 = bands["D3"]
        z3 = band_z["D3"]
        shock_idx = np.where(z3 > SHOCK_Z)[0]
        shock_idx = shock_idx[shock_idx + FWD_LAG < n]
        baseline  = (np.sign(d3[shock_idx]) *
                     (close[shock_idx + FWD_LAG] - close[shock_idx]) > 0).mean()
        print(f"  D3 shocks: {len(shock_idx):,}  baseline cont: {baseline*100:.1f}%")

        X_all, X_p9, y, meta = build_joint_matrix(
            df, bands, band_z, time_z, shock_idx, pip)
        print(f"  X_all={X_all.shape}  X_p9={X_p9.shape}  y_rate={y.mean()*100:.1f}%")

        for model_type in ["logistic", "gbm"]:
            print(f"\n  ── P9 only [{model_type}] ──")
            res_p9 = walk_forward(X_p9, y, meta, 5, model_type, rt_cost)
            res_p9["feature_set"] = "P9_only"; res_p9["pair"] = pair

            print(f"\n  ── Joint P9+P7 [{model_type}] ──")
            res_jt = walk_forward(X_all, y, meta, 5, model_type, rt_cost)
            res_jt["feature_set"] = "Joint"; res_jt["pair"] = pair

            for res, tag in [(res_p9, "P9_only"), (res_jt, "Joint ")]:
                av = res["auc"].values; lv = res["lift_p90"].values
                cv = res["cont_top"].values; ev = res["ev_post"].values
                print(f"  [{tag}] AUC {av.mean():.4f}(min {av.min():.4f})  "
                      f"lift {lv.mean():.2f}  cont_top {cv.mean()*100:.1f}%  "
                      f"EV_post_spread {ev.mean():+.2f}p  "
                      f"gate_AUC60: {'✅' if (av>=0.60).all() else '❌'}  "
                      f"gate_EV: {'✅' if (ev>=0).all() else '❌'}")
            all_results.extend([res_p9, res_jt])
        print()

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUT / "joint_model_results.csv", index=False)

    print("\n=== SUMMARY TABLE ===")
    summary = combined.groupby(["pair","feature_set","model"])[
        ["auc","lift_p90","cont_top","ev_pre","ev_post"]
    ].mean().round(4)
    print(summary.to_string())
