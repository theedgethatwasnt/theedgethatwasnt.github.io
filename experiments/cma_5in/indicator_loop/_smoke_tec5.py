"""Smoke test: validate tec5 in FXFeatureBuilder against reference compute_tec5."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from validate import validate_candidate, print_result


def tec5_reference(df: pd.DataFrame) -> np.ndarray:
    """Original vectorized reference from test_dclose3_regime_cma.py."""
    closes = df["close"].values.astype(np.float64)
    n = len(closes)
    out = np.zeros(n)
    for i in range(5, n):
        net = closes[i] - closes[i - 5]
        path = 0.0
        for k in range(i - 4, i + 1):
            path += abs(closes[k] - closes[k - 1])
        if path > 1e-12:
            er = abs(net) / path
            if net > 0:
                out[i] = er
            elif net < 0:
                out[i] = -er
    return out


def main():
    pair = "EUR_JPY"
    df = pd.read_parquet(PROJECT / f"data/m5_ohlc/{pair}_M5.parquet")
    print(f"Loaded {len(df)} M5 bars for {pair}")
    r = validate_candidate(
        name="tec5",
        reference_fn=tec5_reference,
        builder_key="tec5",
        df=df,
        expected_range=(-1.0, 1.0),
        smoother="kalman10",
        probe_bar=5000,
        n_bars_limit=20000,
    )
    print_result(r)
    print("\n✅ tec5 validation PASSED" if r.passed else "\n❌ tec5 validation FAILED")


if __name__ == "__main__":
    main()
