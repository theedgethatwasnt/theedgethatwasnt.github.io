"""Permutation p-values for strategy gating.

Three shuffle modes:
  - "pairs_per_bar": shuffle factor values across pairs per time bar.
      Correct null for cross-sectional factor alpha (e.g. CSI StrengthSpread).
  - "time": shuffle factor values across time, per pair.
      Correct null for mean-reversion vs. trend claims.
  - "block": circular block shuffle of length ``block_size``.
      Preserves autocorrelation; destroys long-range structure.

See research/MCMC_INTEGRATION_PLAN.md §3.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd


ShuffleMode = Literal["pairs_per_bar", "time", "block"]


@dataclass
class PermutationResult:
    real_score: float
    p_value: float
    perm_mean: float
    perm_std: float
    perm_95th: float
    n_perms: int
    mode: str

    @property
    def significant(self) -> bool:
        """Convention: reject null at p < 0.05."""
        return self.p_value < 0.05

    def as_dict(self) -> dict:
        return {
            "real_score": self.real_score,
            "p_value": self.p_value,
            "perm_mean": self.perm_mean,
            "perm_std": self.perm_std,
            "perm_95th": self.perm_95th,
            "n_perms": self.n_perms,
            "shuffle_mode": self.mode,
            "significant": self.significant,
        }


def _shuffle_pairs_per_bar(factor_df: pd.DataFrame,
                            rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle columns (pairs) within each row (time bar)."""
    vals = factor_df.values.copy()
    for i in range(vals.shape[0]):
        rng.shuffle(vals[i])
    return pd.DataFrame(vals, index=factor_df.index, columns=factor_df.columns)


def _shuffle_time(factor_df: pd.DataFrame,
                   rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle rows (time bars) per column (pair)."""
    vals = factor_df.values.copy()
    for j in range(vals.shape[1]):
        rng.shuffle(vals[:, j])
    return pd.DataFrame(vals, index=factor_df.index, columns=factor_df.columns)


def _shuffle_block(factor_df: pd.DataFrame, block_size: int,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Circular block shuffle: partition into contiguous blocks, permute blocks."""
    n = factor_df.shape[0]
    n_blocks = (n + block_size - 1) // block_size
    block_order = rng.permutation(n_blocks)
    vals = factor_df.values
    out = np.empty_like(vals)
    write = 0
    for b_idx in block_order:
        start = b_idx * block_size
        end = min(start + block_size, n)
        length = end - start
        out[write:write + length] = vals[start:end]
        write += length
    return pd.DataFrame(out, index=factor_df.index, columns=factor_df.columns)


def permutation_pvalue(
    score_fn: Callable[[pd.DataFrame], float],
    factor_df: pd.DataFrame,
    n_perms: int = 1_000,
    mode: ShuffleMode = "pairs_per_bar",
    block_size: int | None = None,
    seed: int = 0,
    tail: Literal["upper", "two"] = "upper",
) -> PermutationResult:
    """Generic permutation test.

    Args:
        score_fn: callable that takes a permuted ``factor_df`` and returns a
            scalar score (Sharpe, return, whatever). Must be pure.
        factor_df: original factor panel (time x pairs).
        n_perms: number of permutations.
        mode: shuffle scheme.
        block_size: required if mode == "block".
        seed: base RNG seed.
        tail: "upper" for one-sided (real >= perm) or "two" for absolute.

    Returns:
        PermutationResult with real_score, p_value, and null distribution
        summary statistics.
    """
    rng = np.random.default_rng(seed)
    real = float(score_fn(factor_df))

    if mode == "block" and not block_size:
        raise ValueError("mode='block' requires block_size")

    perm_scores = np.empty(n_perms, dtype=np.float64)
    for k in range(n_perms):
        perm_seed = rng.integers(0, 2**31 - 1)
        sub_rng = np.random.default_rng(perm_seed)
        if mode == "pairs_per_bar":
            permuted = _shuffle_pairs_per_bar(factor_df, sub_rng)
        elif mode == "time":
            permuted = _shuffle_time(factor_df, sub_rng)
        elif mode == "block":
            permuted = _shuffle_block(factor_df, block_size, sub_rng)
        else:
            raise ValueError(f"unknown mode: {mode}")
        perm_scores[k] = float(score_fn(permuted))

    if tail == "upper":
        p = float((perm_scores >= real).mean())
    else:
        p = float((np.abs(perm_scores) >= abs(real)).mean())

    return PermutationResult(
        real_score=real,
        p_value=p,
        perm_mean=float(perm_scores.mean()),
        perm_std=float(perm_scores.std(ddof=1)),
        perm_95th=float(np.percentile(perm_scores, 95)),
        n_perms=int(n_perms),
        mode=str(mode),
    )
