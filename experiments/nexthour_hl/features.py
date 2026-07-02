"""Causal hourly ATR (the normalizer) + clock features for the next-hour forecaster."""
import numpy as np
import pandas as pd


def hourly_atr(ts, high, low, close, period=14):
    """Wilder ATR(period) on H1 bars, mapped causally to each S5 row (value = last
    COMPLETED hour's ATR). Price units; NaN until `period` hours have completed."""
    idx = pd.DatetimeIndex(ts).as_unit("ns")
    df = pd.DataFrame({"high": np.asarray(high, float),
                       "low": np.asarray(low, float),
                       "close": np.asarray(close, float)}, index=idx)
    h1 = df.resample("1h").agg(high=("high", "max"), low=("low", "min"),
                               close=("close", "last")).dropna()
    m = len(h1)
    out = np.full(len(idx), np.nan)
    if m == 0:
        return out
    hi = h1["high"].to_numpy(); lo = h1["low"].to_numpy(); cl = h1["close"].to_numpy()
    tr = np.empty(m)
    tr[0] = hi[0] - lo[0]
    for i in range(1, m):
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i - 1]), abs(lo[i] - cl[i - 1]))
    atr = np.full(m, np.nan)
    if m >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, m):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    # map each S5 ts to the last hour whose end (index + 1h) is <= ts  (completed)
    end_ns = (h1.index + pd.Timedelta(hours=1)).asi8
    ts_ns = idx.asi8
    j = np.searchsorted(end_ns, ts_ns, side="right") - 1     # last completed hour, or -1
    ok = j >= 0
    out[ok] = atr[j[ok]]
    return out


def clock_features(ts):
    idx = pd.DatetimeIndex(ts)
    how = (idx.dayofweek * 24 + idx.hour).to_numpy().astype(np.int64)   # 0..167
    ang = 2.0 * np.pi * how / 168.0
    return {"how": how,
            "how_sin": np.sin(ang),
            "how_cos": np.cos(ang),
            "dow": idx.dayofweek.to_numpy().astype(float)}
