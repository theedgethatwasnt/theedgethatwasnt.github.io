"""bootstrap.py — COT Contrarian Positioning: weekly-block bootstrap.

PREREGISTRATION.md task brief: "the 3 arms, weekly-block bootstrap." Since the underlying
observation unit in this experiment IS already weekly (one rebalance per COT release —
unlike the sibling M5/D1-bar experiments where a "day block" groups many intraday
observations), a weekly-block bootstrap here reduces to standard resample-with-replacement
of the weekly return series: each "block" is exactly one week, so resampling blocks IS
resampling individual observations. Documented explicitly rather than silently
reimplementing plain bootstrap under a different name.
"""
import numpy as np

N_BOOT = 2000
BOOT_SEED = 20260707


def weekly_block_bootstrap(weekly_returns, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """weekly_returns: 1D array-like of per-week portfolio net returns (pips).
    Returns (p_le_zero, boot_means, ci_2p5, ci_97p5)."""
    vals = np.asarray(weekly_returns, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return float("nan"), np.array([]), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = vals[idx].mean()
    p_le_zero = float(np.mean(boot_means <= 0))
    ci_2p5 = float(np.percentile(boot_means, 2.5))
    ci_97p5 = float(np.percentile(boot_means, 97.5))
    return p_le_zero, boot_means, ci_2p5, ci_97p5
