# research/experiments/mtf_metrics/build_eurusd.py
"""Build the causal MTF metric dataset for EUR_USD from S5.

Run from the repo root:  python3 research/experiments/mtf_metrics/build_eurusd.py
Writes data/mtf_metrics/EUR_USD_mtf.parquet (19 cols: ts + 6 TF x pm/eff/loc).
"""
import gc
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from lib.mtf_metrics import build_mtf, metrics_from_ohlc, position_in_range, TF_MINUTES  # noqa: E402

SRC = os.path.join(REPO, "data/s5_ohlc/EUR_USD_S5_BA.parquet")
OUT = os.path.join(REPO, "data/mtf_metrics/EUR_USD_mtf.parquet")
PIP = 0.0001


def spot_check(ts, o, h, l, c, res, n=5, seed=7):
    """Recompute each TF window with a naive boolean mask and compare to the
    builder's output at n random indices (independent of the vectorized path)."""
    ts_ns = pd.DatetimeIndex(ts).asi8
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, len(ts), size=n)
    for i in idxs:
        for tf, minutes in TF_MINUTES.items():
            ns = np.int64(minutes) * 60 * 1_000_000_000
            # CURRENT window (t-N, t] → pm/eff (and the price c[i])
            mask = (ts_ns > ts_ns[i] - ns) & (ts_ns <= ts_ns[i])
            ow, hw, lw, cw = o[mask][0], h[mask].max(), l[mask].min(), c[i]
            pm, eff, _ = metrics_from_ohlc(ow, hw, lw, cw, PIP, minutes)
            assert np.isclose(float(pm), res[f"{tf}_pm"][i], atol=1e-3), (i, tf, "pm")
            assert np.isclose(float(eff), res[f"{tf}_eff"][i], atol=1e-3), (i, tf, "eff")
            # PRIOR window for loc = the trailing window at the last bar <= t-N
            # (most recent completed window before now; robust across gaps). Independently:
            prior_idx = np.where(ts_ns <= ts_ns[i] - ns)[0]
            ref = res[f"{tf}_loc"][i]
            if prior_idx.size:
                jp = prior_idx[-1]
                pmask = (ts_ns > ts_ns[jp] - ns) & (ts_ns <= ts_ns[jp])
                loc = position_in_range(c[i], l[pmask].min(), h[pmask].max())
                assert np.isclose(float(loc), ref, atol=1e-3), (i, tf, "loc")
            else:
                assert np.isnan(ref), (i, tf, "loc should be NaN in warmup")
    print(f"spot-check OK ({n} timestamps x {len(TF_MINUTES)} TFs)")


def main():
    df = pd.read_parquet(SRC, columns=["timestamp", "open", "high", "low", "close"])
    assert df["timestamp"].is_monotonic_increasing, "S5 must be sorted ascending"
    ts = pd.DatetimeIndex(df["timestamp"])
    o = df["open"].to_numpy(np.float64); h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64);  c = df["close"].to_numpy(np.float64)
    print(f"loaded {len(df):,} S5 rows  {ts[0]} .. {ts[-1]}")

    out = build_mtf(ts, o, h, l, c, PIP)

    res = {"ts": df["timestamp"].to_numpy()}
    res.update(out)
    res = pd.DataFrame(res)

    # range validation
    for tf in TF_MINUTES:
        assert res[f"{tf}_eff"].between(-1e-6, 1 + 1e-6).all(), f"{tf}_eff out of [0,1]"
        assert np.isfinite(res[f"{tf}_pm"]).all(), f"{tf}_pm not finite"
        # loc references the PRIOR window → unbounded (penetration) + NaN in warmup
        loc = res[f"{tf}_loc"].to_numpy()
        assert np.isfinite(loc).sum() > 0, f"{tf}_loc all NaN"
    print("range validation OK")

    spot_check(ts, o, h, l, c, {k: out[k] for k in out}, n=5)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  shape={res.shape}")
    print(res[[f"{tf}_pm" for tf in TF_MINUTES]].describe().T.to_string())
    del df, out, res
    gc.collect()


if __name__ == "__main__":
    main()
