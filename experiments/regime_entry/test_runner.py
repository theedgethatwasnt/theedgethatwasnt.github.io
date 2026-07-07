import numpy as np, pandas as pd, os
from runner import build_signals_for_slice, PIP_SIZE

def test_synthetic_slice_produces_signals(tmp_path):
    # 3 hours of dense synthetic S5 data on a perfect grid
    n = 2160
    ts = pd.date_range("2025-01-06 09:00:00", periods=n, freq="5s", tz="UTC")
    rng = np.random.default_rng(1)
    c = 1.1 + np.cumsum(rng.normal(0, 1e-4 * 0.5, n))
    df = pd.DataFrame({"timestamp": ts, "open": c, "high": c + 2e-5,
                       "low": c - 2e-5, "close": c,
                       "bid_c": c - 8e-5, "ask_c": c + 8e-5})
    out = build_signals_for_slice(df, "EUR_USD")
    assert len(out) > 20                              # ~36 M5 boundaries − warmup
    assert out["n_real_bars"].max() == 60
    assert abs(out["spread_pips"].iloc[0] - 1.6) < 0.01
    assert {"drift", "er", "vr4"}.issubset(out.columns)
    assert "t32_h2_long_label" in out.columns
    assert set(out["t32_h2_long_label"].unique()) <= {-1, 0, 1, -9}
    # causality: features at signal ts must not change if future data is altered
    df2 = df.copy()
    cut = df2["timestamp"] > out["ts"].iloc[5]
    df2.loc[cut, ["open", "high", "low", "close"]] += 0.01
    out2 = build_signals_for_slice(df2, "EUR_USD")
    row, row2 = out.iloc[5], out2.iloc[5]
    for f in ["drift", "er", "vr4", "peak_slope"]:
        assert row[f] == row2[f] or (np.isnan(row[f]) and np.isnan(row2[f]))
