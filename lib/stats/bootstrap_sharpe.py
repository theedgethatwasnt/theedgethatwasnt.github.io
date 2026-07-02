"""Circular block-bootstrap for Sharpe confidence intervals.

Pure numpy. Preserves autocorrelation up to `block_len`. Designed to be fast
enough to attach to every backtest run (< 1 sec for 10k resamples on 5k bars).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SharpeCI:
    """Block-bootstrap Sharpe summary."""
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n_boot: int
    block_len: int
    annualization: float
    lower_positive: bool  # True iff ci_low > 0 (the deploy gate)

    def as_dict(self) -> dict:
        return {
            "sharpe_mean": self.mean,
            "sharpe_std": self.std,
            "sharpe_ci_low": self.ci_low,
            "sharpe_ci_high": self.ci_high,
            "sharpe_ci_lower_positive": self.lower_positive,
            "bootstrap_n": self.n_boot,
            "bootstrap_block_len": self.block_len,
            "annualization": self.annualization,
        }


def _circular_block_indices(n: int, block_len: int, n_blocks: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Draw starting indices and wrap with modulo for circular bootstrap."""
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_len)
    return (starts[:, None] + offsets[None, :]) % n


def block_bootstrap_sharpe(
    net_returns: Iterable[float],
    block_len: int = 20,
    n_boot: int = 10_000,
    annualization: float = 252.0,
    ci: float = 0.95,
    seed: int = 0,
) -> SharpeCI:
    """Circular block-bootstrap Sharpe CI.

    Args:
        net_returns: per-bar net returns (any units — CI is scale-invariant up
            to `annualization`).
        block_len: bars per block; rough choice = holding period in bars.
        n_boot: number of bootstrap resamples (10k is a good default).
        annualization: sqrt multiplier. 252 for daily, 252*24 for hourly, etc.
        ci: confidence level; 0.95 → [2.5, 97.5] percentiles.
        seed: RNG seed.

    Returns:
        SharpeCI with mean/std/CI bounds and a ``lower_positive`` deploy flag.
    """
    r = np.asarray(list(net_returns), dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2 * block_len:
        return SharpeCI(
            mean=float("nan"), std=float("nan"),
            ci_low=float("nan"), ci_high=float("nan"),
            n_boot=0, block_len=block_len,
            annualization=annualization, lower_positive=False,
        )

    rng = np.random.default_rng(seed)
    n_blocks = n // block_len + 1
    idx = _circular_block_indices(n, block_len, n_blocks, rng)
    # flatten to shape (n_boot, n_blocks*block_len) via resampling
    sharpes = np.empty(n_boot, dtype=np.float64)
    sqrt_ann = np.sqrt(annualization)
    for b in range(n_boot):
        pick = rng.integers(0, n_blocks, size=n_blocks)
        sample_idx = idx[pick].ravel()[:n]
        sample = r[sample_idx]
        s = sample.std(ddof=1)
        sharpes[b] = (sample.mean() / s * sqrt_ann) if s > 0 else 0.0

    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = np.quantile(sharpes, [alpha, 1 - alpha])
    return SharpeCI(
        mean=float(sharpes.mean()),
        std=float(sharpes.std(ddof=1)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_boot=int(n_boot),
        block_len=int(block_len),
        annualization=float(annualization),
        lower_positive=bool(ci_low > 0),
    )


def sharpe_ci_report(net_returns, **kwargs) -> dict:
    """Convenience wrapper returning a dict suitable for JSON serialization."""
    return block_bootstrap_sharpe(net_returns, **kwargs).as_dict()
