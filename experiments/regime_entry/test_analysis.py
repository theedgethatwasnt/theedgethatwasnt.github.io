import numpy as np, pandas as pd
from analysis import p_star, assign_arms, fifo_realize, day_block_bootstrap, er_bucket_edges, bucket_table

def test_p_star_matches_doc_worked_example():
    assert abs(p_star(3.0, 6.0, 1.4) - (7.4 / (1.6 + 7.4))) < 1e-12   # ~0.822
    assert abs(p_star(3.0, 6.0, 0.7) - (6.7 / (2.3 + 6.7))) < 1e-12   # ~0.744
    assert abs(p_star(3.2, 6.4, 0.0) - 2/3) < 1e-12                   # zero-cost = baseline

def test_assign_arms_directions():
    df = pd.DataFrame({"drift": [2.0, -1.0, 0.0]})
    out = assign_arms(df.copy(), seed=1)
    assert list(out["dir_with"]) == [1, -1]           # zero-drift row dropped
    assert list(out["dir_against"]) == [-1, 1]
    assert set(out["dir_coin"]) <= {1, -1}

def test_fifo_blocks_overlap():
    ts = pd.date_range("2025-01-06 09:00", periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame({"ts": ts, "held": [30, 10, 10, 10, 10]})  # first holds 30 S5 bars(=150s<300s? use bars*5s)
    # held is in S5 bars; trade 0 spans 30*5=150s -> next boundary free
    mask = fifo_realize(df, held_col="held")
    assert mask.tolist() == [True, True, True, True, True]
    df2 = pd.DataFrame({"ts": ts, "held": [130, 10, 10, 10, 10]})  # 650s -> blocks 2 bars
    mask2 = fifo_realize(df2, held_col="held")
    assert mask2.tolist() == [True, False, False, True, True]

def test_day_block_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(3)
    ts = pd.date_range("2025-01-01", periods=2000, freq="30min", tz="UTC")
    df = pd.DataFrame({"ts": ts, "x": rng.normal(1.0, 1.0, 2000)})
    lo, hi = day_block_bootstrap(df, lambda d: d["x"].mean(), n=500, seed=4)
    assert lo < 1.0 < hi

def test_er_bucket_edges_from_is_only():
    df = pd.DataFrame({"er": np.linspace(0, 1, 100)})
    edges = er_bucket_edges(df["er"])
    assert len(edges) == 2 and 0.2 < edges[0] < 0.45 < edges[1] < 0.8

def test_nan_er_excluded_from_buckets():
    n = 40
    ts = pd.date_range("2025-01-06", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "ts": ts,
        "er": [np.nan] * n,                      # all NaN -> nothing should bucket
        "spread_pips": 1.6,
        "dir_against": 1,
        "t32_h2_long_label": 1, "t32_h2_long_exit_pips": 3.2, "t32_h2_long_bars_held": 5,
        "t32_h2_short_label": -1, "t32_h2_short_exit_pips": -6.4, "t32_h2_short_bars_held": 5,
    })
    out = bucket_table(df, "against", "t32", "h2", (0.3, 0.5))
    assert len(out) == 0
