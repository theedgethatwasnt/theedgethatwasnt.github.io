"""Tests for bootstrap.py."""
import numpy as np
import pytest

import bootstrap as bs


def test_clearly_positive_series_gives_low_p_le_zero():
    rng = np.random.default_rng(0)
    vals = rng.normal(5.0, 1.0, 500)  # mean=5, std=1 -> essentially never <=0 in the mean
    p, boot_means, lo, hi = bs.weekly_block_bootstrap(vals, n_boot=1000, seed=1)
    assert p < 0.01
    assert lo > 0


def test_clearly_negative_series_gives_high_p_le_zero():
    rng = np.random.default_rng(0)
    vals = rng.normal(-5.0, 1.0, 500)
    p, boot_means, lo, hi = bs.weekly_block_bootstrap(vals, n_boot=1000, seed=1)
    assert p > 0.99
    assert hi < 0


def test_bootstrap_mean_is_calibrated_to_sample_mean():
    """The bootstrap distribution's own mean must track the input sample's mean closely
    (basic calibration check) — not a claim about where zero falls, which depends on the
    realized sample mean's sampling noise, not the generator's true mean (a flaky-test
    trap the first version of this test fell into)."""
    rng = np.random.default_rng(0)
    vals = rng.normal(0.0, 1.0, 2000)
    p, boot_means, lo, hi = bs.weekly_block_bootstrap(vals, n_boot=2000, seed=2)
    assert boot_means.mean() == pytest.approx(vals.mean(), abs=0.05)
    assert lo < boot_means.mean() < hi
    # p_le_zero must be internally consistent with the CI: if the CI excludes zero on one
    # side, p must be near that side's tail probability, not the opposite.
    if lo > 0:
        assert p < 0.05
    elif hi < 0:
        assert p > 0.95
    else:
        assert 0.0 <= p <= 1.0


def test_deterministic_given_seed():
    vals = np.arange(100, dtype=float) - 50
    p1, m1, _, _ = bs.weekly_block_bootstrap(vals, n_boot=500, seed=42)
    p2, m2, _, _ = bs.weekly_block_bootstrap(vals, n_boot=500, seed=42)
    assert p1 == p2
    assert np.array_equal(m1, m2)


def test_empty_input_returns_nan():
    p, boot_means, lo, hi = bs.weekly_block_bootstrap([], n_boot=100)
    assert np.isnan(p)
    assert len(boot_means) == 0


def test_nan_values_filtered_out():
    vals = [1.0, 2.0, np.nan, 3.0, np.nan, -1.0]
    p, boot_means, lo, hi = bs.weekly_block_bootstrap(vals, n_boot=200, seed=3)
    assert np.isfinite(p)
