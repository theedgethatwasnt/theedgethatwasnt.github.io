"""
MSP Phase 7: Predictive model + walk-forward validation.
Feature set: wavelet D1..D6 shock_z + time-window 5s..15m shock_z.
Target: D5_shock at t+22bars (110s).
WF: 5 temporal folds, 20% embargo.
Gates: AUC >= 0.55 all folds, lift@P90 >= 2.0 in 4/5 folds.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import duckdb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

DB_PATH = Path(__file__).parent / "results" / "msp_features.duckdb"
OUT     = Path(__file__).parent / "results"
FWD_LAG = 22   # predict D5_shock at t+22bars (110s)
N_FOLDS = 5


def get_pairs(con) -> list:
    return [r[0] for r in con.execute(
        "SELECT DISTINCT pair FROM wavelet_features").fetchall()]


def build_features(con, pair: str):
    wt_cols = ", ".join([f'w."D{j}_shock_z"' for j in range(1, 7)])
    tm_cols = ", ".join([f't."{lbl}_shock_z"' for lbl in
                         ["5s","10s","30s","1m","2m","5m","15m"]])
    q = f"""
        SELECT w.timestamp,
               {wt_cols},
               {tm_cols},
               LEAD(w."D5_shock", {FWD_LAG}) OVER (ORDER BY w.timestamp::TIMESTAMPTZ) AS y
        FROM wavelet_features w
        LEFT JOIN time_features t
          ON t.pair = w.pair
          AND t.timestamp::TIMESTAMPTZ = w.timestamp::TIMESTAMPTZ
        WHERE w.pair = '{pair}'
        ORDER BY w.timestamp::TIMESTAMPTZ
    """
    df = con.execute(q).fetchdf().dropna()
    feat_cols = [c for c in df.columns if c not in ("timestamp", "y")]
    X = df[feat_cols].values.astype(np.float64)
    X = np.clip(np.nan_to_num(X, nan=0.0), -10, 10)
    y = df["y"].values.astype(np.int8)
    return X, y, df["timestamp"]


def walk_forward(X, y, n_folds=N_FOLDS):
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
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
        clf.fit(sc.transform(X_tr), y_tr)
        prob = clf.predict_proba(sc.transform(X_te))[:, 1]
        auc  = roc_auc_score(y_te, prob)
        thr  = np.percentile(prob, 90)
        top  = y_te[prob >= thr]
        lift = top.mean() / y_te.mean() if y_te.mean() > 0 else 1.0
        rows.append({"fold": fold,
                     "n_train": tr_end, "n_test": te_end - te_st,
                     "auc": round(auc, 4), "lift_p90": round(lift, 3),
                     "base_rate_pct": round(y_te.mean() * 100, 4)})
        print(f"  fold {fold}: AUC={auc:.3f} lift@P90={lift:.2f} "
              f"base%={y_te.mean()*100:.3f}%  n_train={tr_end:,}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH), read_only=True)
    pairs = get_pairs(con)
    all_results = []
    for pair in pairs:
        print(f"\n{'='*50}\n{pair}")
        X, y, ts = build_features(con, pair)
        print(f"  n={len(X):,}  y_rate={y.mean()*100:.3f}%  features={X.shape[1]}")
        res = walk_forward(X, y)
        res["pair"] = pair
        all_results.append(res)
        auc_v  = res["auc"].values
        lift_v = res["lift_p90"].values
        print(f"  → AUC mean={auc_v.mean():.4f} min={auc_v.min():.4f}")
        print(f"  → lift mean={lift_v.mean():.3f} min={lift_v.min():.3f}")
        gates = {
            "AUC>=0.55 all folds":   bool((auc_v >= 0.55).all()),
            "lift@P90>=2.0 in 4/5":  bool((lift_v >= 2.0).sum() >= 4),
        }
        print(f"  Gates: {gates}")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUT / "wf_prediction_results.csv", index=False)
    print(f"\nSaved: results/wf_prediction_results.csv")
    print("\n=== COMBINED AUC SUMMARY ===")
    print(combined.groupby("pair")[["auc","lift_p90"]].agg(["mean","min"]).round(4))
    con.close()
