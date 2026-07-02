"""
MSP Phase 3: DWT wavelet features → msp_features.duckdb table `wavelet_features`.
Writes: D1..D8 coefficient, log-energy, shock_z, shock flag per bar.
"""
import gc, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pywt
import duckdb
from pathlib import Path

BASE     = Path(__file__).resolve().parents[3]
S5_DIRS  = [BASE / "data" / "s5_ohlc", BASE / "data" / "s5_ba"]
DB_PATH  = Path(__file__).parent / "results" / "msp_features.duckdb"
WAVELET  = "db4"
DWT_LEVEL = 8
MAD_WIN   = 1024
PIP_MAP   = {"EUR_USD": 0.0001, "EUR_JPY": 0.01, "GBP_JPY": 0.01}
BAND_NAMES = [f"D{j}" for j in range(1, DWT_LEVEL + 1)]


def find_parquet(pair: str):
    for d in S5_DIRS:
        p = d / f"{pair}_S5_BA.parquet"
        if p.exists():
            return p
    return None


def load_close(path: Path, pip: float):
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    if "close" not in df.columns:
        df["close"] = (df["bid_c"] + df["ask_c"]) / 2
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return df["close"].values.astype(np.float64), ts


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


def mad_zscore(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x.astype(np.float64))
    rm  = s.rolling(w, center=True, min_periods=max(10, w // 4)).median()
    ad  = (s - rm).abs()
    rm2 = ad.rolling(w, center=True, min_periods=max(10, w // 4)).median()
    return ((s - rm) / (1.4826 * rm2.clip(lower=1e-12))).fillna(0).values


def compute_wavelet_features(close: np.ndarray, ts: pd.Series) -> pd.DataFrame:
    print(f"  Computing DWT bands...")
    bands = compute_dwt_bands(close)
    rows  = {"timestamp": ts.values}
    for name, coef in bands.items():
        print(f"  MAD z-score for {name}...")
        log_e = np.log(coef ** 2 + 1e-12)
        z     = mad_zscore(log_e, MAD_WIN)
        rows[f"{name}_coef"]    = coef.astype(np.float32)
        rows[f"{name}_log_e"]   = log_e.astype(np.float32)
        rows[f"{name}_shock_z"] = z.astype(np.float32)
        rows[f"{name}_shock"]   = (z > 2.5).astype(np.int8)
    return pd.DataFrame(rows)


def write_to_db(feat: pd.DataFrame, pair: str) -> None:
    con = duckdb.connect(str(DB_PATH))
    feat = feat.copy()
    feat["pair"] = pair
    try:
        con.execute("DELETE FROM wavelet_features WHERE pair = ?", [pair])
        con.execute("INSERT INTO wavelet_features SELECT * FROM feat")
    except Exception:
        con.execute("CREATE TABLE wavelet_features AS SELECT * FROM feat")
    n = con.execute("SELECT count(*) FROM wavelet_features WHERE pair=?", [pair]).fetchone()[0]
    con.close()
    print(f"  {pair}: {n:,} rows in wavelet_features")


if __name__ == "__main__":
    for pair, pip in PIP_MAP.items():
        path = find_parquet(pair)
        if path is None:
            print(f"  {pair}: no parquet, skip"); continue
        print(f"\nProcessing {pair}...")
        close, ts = load_close(path, pip)
        feat = compute_wavelet_features(close, ts)
        write_to_db(feat, pair)
        del close, feat; gc.collect()
    print("\nDone.")
