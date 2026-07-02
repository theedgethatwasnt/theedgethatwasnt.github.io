"""
MSP Phase 1: Rolling time-window features from S5 BA parquets → DuckDB.
Computes: TR-rate, body, shock_z, shock flag per window (5s..1h).
Writes to msp_features.duckdb table `time_features`.
"""
import gc, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

BASE     = Path(__file__).resolve().parents[3]
S5_DIRS  = [BASE / "data" / "s5_ohlc", BASE / "data" / "s5_ba"]
DB_PATH  = Path(__file__).parent / "results" / "msp_features.duckdb"
DB_PATH.parent.mkdir(exist_ok=True)

PIP_MAP  = {"EUR_USD": 0.0001, "EUR_JPY": 0.01, "GBP_JPY": 0.01}

# Time windows in S5 bars
TIME_WINDOWS = [1, 2, 6, 12, 24, 60, 180, 720]
TIME_LABELS  = ["5s","10s","30s","1m","2m","5m","15m","1h"]
MAD_WIN = 1024


def find_parquet(pair: str):
    for d in S5_DIRS:
        p = d / f"{pair}_S5_BA.parquet"
        if p.exists():
            return p
    return None


def load_s5(path: Path, pip: float) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    if "close" not in df.columns:
        df["close"] = (df["bid_c"] + df["ask_c"]) / 2
        df["open"]  = (df["bid_o"] + df["ask_o"]) / 2
        df["high"]  = df["ask_h"] if "ask_h" in df.columns else df["ask_c"]
        df["low"]   = df["bid_l"] if "bid_l" in df.columns else df["bid_c"]
    df["spread_p"] = (df["ask_c"] - df["bid_c"]) / pip
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def mad_zscore_series(s: pd.Series, w: int) -> pd.Series:
    rm  = s.rolling(w, center=True, min_periods=max(10, w // 4)).median()
    ad  = (s - rm).abs()
    rm2 = ad.rolling(w, center=True, min_periods=max(10, w // 4)).median()
    return ((s - rm) / (1.4826 * rm2.clip(lower=1e-12))).fillna(0)


def compute_time_features(df: pd.DataFrame, pip: float) -> pd.DataFrame:
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    open_ = df["open"].values
    out   = {"timestamp": df["timestamp"].values, "spread_p": df["spread_p"].values}

    for w, lbl in zip(TIME_WINDOWS, TIME_LABELS):
        s_close = pd.Series(close)
        roll_hi  = pd.Series(high).rolling(w, min_periods=w).max()
        roll_lo  = pd.Series(low).rolling(w, min_periods=w).min()
        roll_open = pd.Series(open_).shift(w - 1)
        prev_c   = s_close.shift(w)

        hl   = roll_hi - roll_lo
        hpc  = (roll_hi - prev_c).abs()
        lpc  = (roll_lo - prev_c).abs()
        tr   = np.maximum(hl.values, np.maximum(hpc.fillna(hl).values, lpc.fillna(hl).values))

        dur_min  = w * (5.0 / 60.0)
        tr_rate  = pd.Series(tr) / pip / dur_min
        body_p   = (s_close - roll_open) / pip
        z = mad_zscore_series(tr_rate, MAD_WIN)

        out[f"{lbl}_tr_rate"]   = tr_rate.values.astype(np.float32)
        out[f"{lbl}_body_pips"] = body_p.values.astype(np.float32)
        out[f"{lbl}_shock_z"]   = z.values.astype(np.float32)
        out[f"{lbl}_shock"]     = (z > 2.5).astype(np.int8).values

    return pd.DataFrame(out)


def write_to_db(feat: pd.DataFrame, pair: str) -> None:
    con = duckdb.connect(str(DB_PATH))
    feat = feat.copy()
    feat["pair"] = pair
    feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True)
    try:
        con.execute("DELETE FROM time_features WHERE pair = ?", [pair])
        con.execute("INSERT INTO time_features SELECT * FROM feat")
    except Exception:
        con.execute("CREATE TABLE time_features AS SELECT * FROM feat")
    n = con.execute("SELECT count(*) FROM time_features WHERE pair=?", [pair]).fetchone()[0]
    con.close()
    print(f"  {pair}: {n:,} rows in time_features")


if __name__ == "__main__":
    for pair, pip in PIP_MAP.items():
        path = find_parquet(pair)
        if path is None:
            print(f"  {pair}: no parquet found, skip")
            continue
        print(f"Processing {pair} ({path.name})...")
        df   = load_s5(path, pip)
        print(f"  n={len(df):,}  Computing time-window features...")
        feat = compute_time_features(df, pip)
        write_to_db(feat, pair)
        del df, feat; gc.collect()
    print("Done.")
