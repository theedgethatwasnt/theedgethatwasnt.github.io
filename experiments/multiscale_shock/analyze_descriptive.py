"""
MSP Phase 5: Descriptive analysis — lift tables + cross-scale xcorr from DuckDB.
Reads time_features and wavelet_features, computes:
 1. Shock rates per window / wavelet band
 2. Lift P(wavelet_shock | time_shock) / P(wavelet_shock) at lag 0,2,11,22 bars
 3. Cross-scale xcorr energy D3→D5, D4→D6, D1→D4
 4. Session-conditional D3/D4 shock rates
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DB_PATH = Path(__file__).parent / "results" / "msp_features.duckdb"
OUT     = Path(__file__).parent / "results"

TIME_LABELS  = ["5s","10s","30s","1m","2m","5m","15m","1h"]
BAND_LABELS  = [f"D{j}" for j in range(1, 9)]
FOCUS_BANDS  = ["D3","D4","D5","D6"]
SRC_TIME     = ["5s","10s","30s","1m","2m","5m"]   # fine time windows as sources


def get_pairs(con) -> list:
    return [r[0] for r in con.execute(
        "SELECT DISTINCT pair FROM wavelet_features ORDER BY pair").fetchall()]


def shock_rates(con, pair: str) -> pd.DataFrame:
    rows = []
    for lbl in TIME_LABELS:
        r = con.execute(
            f'SELECT avg("{lbl}_shock")*100 FROM time_features WHERE pair=?', [pair]
        ).fetchone()[0]
        rows.append({"source": "time", "window": lbl, "shock_rate_pct": round(r or 0, 4)})
    for b in BAND_LABELS:
        r = con.execute(
            f'SELECT avg("{b}_shock")*100 FROM wavelet_features WHERE pair=?', [pair]
        ).fetchone()[0]
        rows.append({"source": "wavelet", "window": b, "shock_rate_pct": round(r or 0, 4)})
    return pd.DataFrame(rows)


def lift_table(con, pair: str, lag: int) -> pd.DataFrame:
    """P(D3_shock_j+lag | time_shock_j) / P(D3_shock_j+lag)."""
    rows = []
    for src_t in SRC_TIME:
        for tgt_b in FOCUS_BANDS:
            q = f"""
            WITH j AS (
                SELECT t."{src_t}_shock" AS src,
                       LEAD(w."{tgt_b}_shock", {lag}) OVER (
                           PARTITION BY t.pair ORDER BY t.timestamp
                       ) AS tgt
                FROM time_features t
                JOIN wavelet_features w
                  ON t.pair = w.pair
                  AND t.timestamp::TIMESTAMPTZ = w.timestamp::TIMESTAMPTZ
                WHERE t.pair = '{pair}'
            )
            SELECT avg(tgt) FILTER (WHERE src=1) AS cond,
                   avg(tgt) AS base
            FROM j
            """
            r = con.execute(q).fetchone()
            cond, base = r
            if base and base > 0 and cond is not None:
                rows.append({
                    "pair": pair, "src_time": src_t, "tgt_wt": tgt_b,
                    "lag": lag, "lift": round(cond / base, 3),
                    "cond_rate": round(cond * 100, 4), "base_rate": round(base * 100, 4)
                })
    return pd.DataFrame(rows)


def wavelet_xcorr(con, pair: str, src_b: str, tgt_b: str, max_lag: int = 88) -> np.ndarray:
    q = f"""
        SELECT "{src_b}_log_e", "{tgt_b}_log_e"
        FROM wavelet_features WHERE pair='{pair}' ORDER BY timestamp::TIMESTAMPTZ
    """
    df = con.execute(q).fetchdf().fillna(0)
    a = df[f"{src_b}_log_e"].values.astype(np.float64)
    b = df[f"{tgt_b}_log_e"].values.astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.zeros(max_lag + 1)
    n = min(len(a), len(b))
    out = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            out[0] = np.dot(a[:n], b[:n]) / (na * nb)
        else:
            out[lag] = np.dot(a[:n-lag], b[lag:n]) / (na * nb)
    return out


def plot_xcorr(pair: str, xcorr_dict: dict) -> None:
    fig, axes = plt.subplots(len(xcorr_dict), 1,
                             figsize=(12, 3 * len(xcorr_dict)))
    if len(xcorr_dict) == 1:
        axes = [axes]
    for ax, ((sa, sb), corrs) in zip(axes, xcorr_dict.items()):
        lags = np.arange(len(corrs))
        ax.plot(lags * 5, corrs, color="royalblue")
        pk = np.argmax(np.abs(corrs))
        ax.axvline(lags[pk] * 5, color="red", ls="--", lw=1,
                   label=f"peak={corrs[pk]:+.3f}@{lags[pk]*5}s")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"{pair}: log_E({sa}) → log_E({sb})")
        ax.set_xlabel("Lag (seconds)"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / f"descriptive_{pair}_xcorr.png", dpi=100)
    plt.close()
    print(f"  Saved: descriptive_{pair}_xcorr.png")


def session_shock_rates(con, pair: str) -> pd.DataFrame:
    q = f"""
    SELECT
        CASE
            WHEN hour(w.timestamp) BETWEEN 6 AND 8   THEN 'lonopen'
            WHEN hour(w.timestamp) BETWEEN 9 AND 11  THEN 'london'
            WHEN hour(w.timestamp) BETWEEN 12 AND 16 THEN 'overlap'
            WHEN hour(w.timestamp) BETWEEN 17 AND 21 THEN 'nyafternoon'
            ELSE 'asian'
        END AS session,
        avg(D3_shock)*100 AS d3_pct,
        avg(D4_shock)*100 AS d4_pct,
        count(*) AS n
    FROM wavelet_features w
    WHERE pair='{pair}'
    GROUP BY session ORDER BY d3_pct DESC
    """
    return con.execute(q).fetchdf()


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH), read_only=True)
    pairs = get_pairs(con)
    all_lifts = []
    for pair in pairs:
        print(f"\n{'='*50}\n{pair}")

        # Shock rates
        sr = shock_rates(con, pair)
        print("Shock rates:")
        print(sr.to_string(index=False))
        sr.to_csv(OUT / f"descriptive_{pair}_shock_rates.csv", index=False)

        # Session shock rates
        sess_sr = session_shock_rates(con, pair)
        print("\nD3/D4 shock rate by session:")
        print(sess_sr.to_string(index=False))

        # Lift tables
        for lag in [0, 2, 11, 22]:
            lt = lift_table(con, pair, lag)
            if not lt.empty:
                top = lt.nlargest(5, "lift")
                print(f"\nTop-5 lifts at lag={lag}bars ({lag*5}s):")
                print(top[["src_time","tgt_wt","lift","cond_rate","base_rate"]].to_string(index=False))
                all_lifts.append(lt)

        # XCorr
        xcorr_results = {}
        for sa, sb in [("D3","D5"), ("D4","D6"), ("D1","D4")]:
            print(f"  xcorr {sa}→{sb}...", end="", flush=True)
            corrs = wavelet_xcorr(con, pair, sa, sb, max_lag=88)
            xcorr_results[(sa, sb)] = corrs
            pk = np.argmax(np.abs(corrs))
            print(f"  peak={corrs[pk]:+.3f}@{pk*5}s")
        plot_xcorr(pair, xcorr_results)

    if all_lifts:
        combined = pd.concat(all_lifts, ignore_index=True)
        combined.to_csv(OUT / "descriptive_all_lifts.csv", index=False)
        print(f"\nSaved: descriptive_all_lifts.csv ({len(combined)} rows)")

        # Print overall top lifts
        print("\n=== TOP-10 LIFTS (all pairs, all lags) ===")
        print(combined.nlargest(10, "lift")[
            ["pair","src_time","tgt_wt","lag","lift","cond_rate","base_rate"]
        ].to_string(index=False))
    con.close()
