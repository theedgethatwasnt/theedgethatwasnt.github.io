"""
MSP Phase 6: Granger causality + Transfer Entropy between wavelet bands.
Tests: does D3_shock Granger-cause D5_shock? D4→D6? (fine → coarse propagation)
Downsamples 10x for speed (binary series, stationarity preserved).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import duckdb
from statsmodels.tsa.stattools import grangercausalitytests
from pathlib import Path
from collections import Counter

DB_PATH  = Path(__file__).parent / "results" / "msp_features.duckdb"
OUT      = Path(__file__).parent / "results"
MAX_LAG  = 22   # bars (5..110s)
TEST_PAIRS = [("D3","D5"), ("D4","D6"), ("D3","D4"), ("D1","D3"), ("D2","D4")]


def get_pairs(con) -> list:
    return [r[0] for r in con.execute(
        "SELECT DISTINCT pair FROM wavelet_features ORDER BY pair").fetchall()]


def granger_test(con, pair: str, src: str, tgt: str) -> dict:
    q = f"""
        SELECT "{src}_shock" AS src, "{tgt}_shock" AS tgt
        FROM wavelet_features WHERE pair='{pair}'
        ORDER BY timestamp::TIMESTAMPTZ
    """
    df = con.execute(q).fetchdf().dropna().astype(float)
    df = df.iloc[::10].reset_index(drop=True)   # downsample 10x
    if df["src"].sum() < 20 or df["tgt"].sum() < 20:
        return {"pair": pair, "src": src, "tgt": tgt, "min_p": 1.0, "best_lag": 0}
    data = df[["tgt", "src"]].values
    try:
        res = grangercausalitytests(data, maxlag=MAX_LAG // 10, verbose=False)
        p_vals = {lag: res[lag][0]["ssr_ftest"][1] for lag in res}
        min_p   = min(p_vals.values())
        best_lag = min(p_vals, key=p_vals.get) * 10
    except Exception:
        return {"pair": pair, "src": src, "tgt": tgt, "min_p": 1.0, "best_lag": 0}
    return {"pair": pair, "src": src, "tgt": tgt,
            "min_p": round(min_p, 6), "best_lag": best_lag}


def transfer_entropy(x: np.ndarray, y: np.ndarray, lag: int = 1) -> float:
    """TE(X→Y) for binary series: H(Y_future|Y_past) - H(Y_future|Y_past,X_past)."""
    n = len(x) - lag
    yp = y[:n].astype(int); yf = y[lag:].astype(int); xp = x[:n].astype(int)

    def joint_h(*arrs):
        total = len(arrs[0])
        keys = list(zip(*arrs))
        counts = Counter(keys)
        return -sum((c / total) * np.log2(c / total + 1e-15) for c in counts.values())

    h_yf_yp    = joint_h(yp, yf) - joint_h(yp,)
    h_yf_yp_xp = joint_h(yp, xp, yf) - joint_h(yp, xp)
    return max(0.0, h_yf_yp - h_yf_yp_xp)


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH), read_only=True)
    pairs = get_pairs(con)
    granger_rows = []
    te_rows = []

    for pair in pairs:
        print(f"\n{'='*50}\n{pair}")
        for src, tgt in TEST_PAIRS:
            r = granger_test(con, pair, src, tgt)
            granger_rows.append(r)
            sig = "✅" if r["min_p"] < 0.01 else ("⚠️" if r["min_p"] < 0.05 else "❌")
            print(f"  Granger {src}→{tgt}: p={r['min_p']:.4f} lag={r['best_lag']}bars {sig}")

            # Transfer Entropy at lag=11 bars
            q = f"""SELECT "{src}_shock", "{tgt}_shock"
                    FROM wavelet_features WHERE pair='{pair}'
                    ORDER BY timestamp::TIMESTAMPTZ"""
            df = con.execute(q).fetchdf().dropna().astype(float)
            if len(df) > 1000:
                te = transfer_entropy(df[f"{src}_shock"].values,
                                      df[f"{tgt}_shock"].values, lag=11)
                te_rows.append({"pair": pair, "src": src, "tgt": tgt,
                                 "te_lag11": round(te, 6)})
                print(f"  TE {src}→{tgt} @11bars(55s): {te:.6f}")

    pd.DataFrame(granger_rows).to_csv(OUT / "granger_results.csv", index=False)
    pd.DataFrame(te_rows).to_csv(OUT / "transfer_entropy.csv", index=False)

    print("\n=== GRANGER SUMMARY (all pairs) ===")
    gr = pd.DataFrame(granger_rows)
    print(gr.to_string(index=False))

    print("\n=== TRANSFER ENTROPY SUMMARY ===")
    te = pd.DataFrame(te_rows)
    if not te.empty:
        print(te.to_string(index=False))

    print(f"\nSignificant Granger (p<0.01): {(gr['min_p'] < 0.01).sum()}/{len(gr)}")
    con.close()
