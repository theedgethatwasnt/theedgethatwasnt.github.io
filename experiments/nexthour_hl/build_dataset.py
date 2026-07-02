"""Assemble the causal supervised table for the next-hour high/low forecaster."""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from lib.mtf_metrics import rolling_window_ohlc  # noqa: E402
from research.experiments.nexthour_hl.targets import forward_excursions  # noqa: E402
from research.experiments.nexthour_hl.features import hourly_atr, clock_features  # noqa: E402

METRICS = [f"{tf}_{m}" for tf in ["m5", "h1", "h4", "h8", "d1", "w1"] for m in ["pm", "eff", "loc"]]
FEATURES = METRICS + ["how_sin", "how_cos", "dow", "atr_pips", "trail_range_pips"]

SRC_MTF = os.path.join(REPO, "data/mtf_metrics/EUR_USD_mtf.parquet")
SRC_S5 = os.path.join(REPO, "data/s5_ohlc/EUR_USD_S5_BA.parquet")
OUT = os.path.join(REPO, "data/nexthour_hl/EUR_USD_supervised.parquet")


def build(mtf_path, s5_path, pip=0.0001):
    mtf = pd.read_parquet(mtf_path)
    s5 = pd.read_parquet(s5_path, columns=["timestamp", "open", "high", "low", "close"])
    s5 = s5.rename(columns={"timestamp": "ts"})
    df = mtf.merge(s5, on="ts", how="inner").sort_values("ts").reset_index(drop=True)

    ts = df["ts"]
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)

    atr = hourly_atr(ts, h, l, c)                 # price units
    df["atr_pips"] = atr / pip
    _, hw, lw, _ = rolling_window_ohlc(ts, o, h, l, c, 60)   # trailing 60-min range
    df["trail_range_pips"] = (hw - lw) / pip

    cf = clock_features(ts)
    df["how"] = cf["how"]; df["how_sin"] = cf["how_sin"]
    df["how_cos"] = cf["how_cos"]; df["dow"] = cf["dow"]

    up_pips, dn_pips, valid = forward_excursions(ts, h, l, c, pip, 60, 30)
    # Clamp to zero: MFE/MAE are always non-negative (negative = price never moved that direction)
    up_pips = np.where(valid, np.maximum(0.0, up_pips), np.nan)
    dn_pips = np.where(valid, np.maximum(0.0, dn_pips), np.nan)
    atr_pips = df["atr_pips"].to_numpy()
    good = valid & np.isfinite(atr_pips) & (atr_pips > 0)
    df["up"] = np.where(good, up_pips / atr_pips, np.nan)
    df["dn"] = np.where(good, dn_pips / atr_pips, np.nan)
    df["valid"] = good

    hour_id = ts.dt.floor("h")
    minute_id = ts.dt.floor("min")
    df["is_hour_anchor"] = hour_id.ne(hour_id.shift(1)).to_numpy()
    df["is_minute_anchor"] = minute_id.ne(minute_id.shift(1)).to_numpy()

    keep = ["ts"] + METRICS + ["how", "how_sin", "how_cos", "dow", "atr_pips",
            "trail_range_pips", "up", "dn", "valid", "is_minute_anchor", "is_hour_anchor"]
    return df[keep]


def main():
    df = build(SRC_MTF, SRC_S5, 0.0001)
    n_valid = int(df["valid"].sum())
    print(f"rows={len(df):,}  valid={n_valid:,}  dropped(invalid)={len(df)-n_valid:,}")
    print(f"minute-anchors={int(df['is_minute_anchor'].sum()):,}  hour-anchors={int(df['is_hour_anchor'].sum()):,}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print("wrote", OUT, df.shape)


if __name__ == "__main__":
    main()
