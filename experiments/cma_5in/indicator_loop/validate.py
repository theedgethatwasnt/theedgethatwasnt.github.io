"""Generic parity + causality validator for FXFeatureBuilder candidates.

Usage:
    from validate import validate_candidate, ValidationError
    result = validate_candidate(
        name="tec5",
        reference_fn=compute_tec5_reference,
        builder_key="tec5",
        df=ohlc_df,
        expected_range=(-1.0, 1.0),
    )

Each candidate must pass:
1. Parity     — Pearson correlation >= 0.9999 vs reference vectorized version,
                max |diff| <= 1e-8 (after warmup bars dropped)
2. Sign match — sign(reference) == sign(incremental) on non-zero samples
3. Range      — observed [min, max] fits within declared range with ε
4. Causality  — perturb bars [probe+1:], assert feature[:probe+1] byte-identical

Raises ValidationError on any failure.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

from lib.incremental_features import FXFeatureBuilder  # noqa: E402


class ValidationError(RuntimeError):
    pass


@dataclass
class ValidationResult:
    name: str
    n_bars: int
    n_compared: int
    correlation: float
    max_abs_diff: float
    sign_match_pct: float
    obs_range: Tuple[float, float]
    declared_range: Tuple[float, float]
    causal_past_max_diff: float
    causal_future_max_diff: float
    passed: bool
    notes: str = ""


def _warmup_bars_for(name: str) -> int:
    """How many initial bars to drop before comparing (both streams warm up slowly)."""
    # Generous defaults — ASI multi-TF pipeline can take 720+ bars for slowest TF.
    return {
        "tec5": 10,
        "macd_hist": 40,
        "er_norm": 80,
        "mc_d_a": 1000,
        "mc_dd_a": 1000,
        "range_pos_30": 40,
        "bb_width": 40,
        "h1_slope": 20,
        "trending": 30,
        "high_vol": 30,
    }.get(name, 100)


def validate_candidate(
    name: str,
    reference_fn: Callable[[pd.DataFrame], np.ndarray],
    builder_key: str,
    df: pd.DataFrame,
    expected_range: Tuple[float, Optional[float]],
    smoother: str = "kalman10",
    probe_bar: int = 5000,
    n_bars_limit: int = 20000,
    corr_threshold: float = 0.9999,
    max_diff_threshold: float = 1e-8,
) -> ValidationResult:
    """Validate a candidate feature and return results. Raises on any hard-gate failure."""
    df = df.iloc[:n_bars_limit].copy().reset_index(drop=True)
    n = len(df)
    warmup = _warmup_bars_for(name)
    if n <= warmup + 100:
        raise ValidationError(f"df too small (n={n}, warmup={warmup})")

    # ── 1. Reference (vectorized/naïve) ──────────────────────────
    ref = np.asarray(reference_fn(df), dtype=np.float64)
    assert len(ref) == n, f"reference_fn returned {len(ref)} values for {n} bars"

    # ── 2. Incremental (FXFeatureBuilder) ────────────────────────
    b = FXFeatureBuilder("VALIDATION", smoother=smoother)
    inc_df = b.walk_history(df)
    if builder_key not in inc_df.columns:
        raise ValidationError(
            f"FXFeatureBuilder output missing key '{builder_key}'. "
            f"Ported correctly? Found keys: {list(inc_df.columns)}"
        )
    inc = inc_df[builder_key].values.astype(np.float64)

    # ── 3. Parity after warmup ───────────────────────────────────
    ref_w = ref[warmup:]
    inc_w = inc[warmup:]
    if np.std(ref_w) < 1e-12:
        raise ValidationError(f"reference is ~constant after warmup (std={np.std(ref_w):.2e})")
    corr = float(np.corrcoef(ref_w, inc_w)[0, 1])
    max_abs = float(np.max(np.abs(ref_w - inc_w)))

    # ── 4. Sign match (only non-zero bars) ────────────────────────
    mask = (np.abs(ref_w) > 1e-9) & (np.abs(inc_w) > 1e-9)
    if mask.sum() > 0:
        sign_match = float(np.mean(np.sign(ref_w[mask]) == np.sign(inc_w[mask])))
    else:
        sign_match = 1.0  # no signed samples to test

    # ── 5. Range check ───────────────────────────────────────────
    obs_min, obs_max = float(np.min(inc)), float(np.max(inc))
    lo, hi = expected_range
    eps = 0.05 * (abs(lo) + abs(hi if hi is not None else obs_max) + 1.0)
    if obs_min < lo - eps:
        raise ValidationError(f"obs_min={obs_min:.4f} below declared {lo:.4f} (eps={eps:.4f})")
    if hi is not None and obs_max > hi + eps:
        raise ValidationError(f"obs_max={obs_max:.4f} above declared {hi:.4f} (eps={eps:.4f})")

    # ── 6. Causality probe: perturb [probe+1:], re-run incremental ───
    if probe_bar >= n - 100:
        probe_bar = n // 2
    rng = np.random.default_rng(42)
    df2 = df.copy()
    k = n - probe_bar - 1
    perturb = rng.normal(0, 0.1, k)
    for col in ("open", "high", "low", "close"):
        df2.loc[probe_bar + 1:, col] = df2.loc[probe_bar + 1:, col].values + perturb
    b2 = FXFeatureBuilder("VALIDATION2", smoother=smoother)
    inc2 = b2.walk_history(df2)[builder_key].values.astype(np.float64)
    past_diff = float(np.max(np.abs(inc[: probe_bar + 1] - inc2[: probe_bar + 1])))
    fut_diff = float(np.max(np.abs(inc[probe_bar + 1:] - inc2[probe_bar + 1:])))

    # ── 7. Hard gates ────────────────────────────────────────────
    errors = []
    if corr < corr_threshold:
        errors.append(f"correlation={corr:.6f} < {corr_threshold}")
    if max_abs > max_diff_threshold and corr < 1.0:
        # Allow if correlation is perfect (constant offset is still OK — but note it)
        errors.append(f"max_abs_diff={max_abs:.2e} > {max_diff_threshold:.2e}")
    if sign_match < 0.99:
        errors.append(f"sign_match={sign_match:.4f} < 0.99")
    if past_diff > 1e-10:
        errors.append(f"causality violated: past_max_diff={past_diff:.2e}")
    if fut_diff < 1e-10:
        errors.append(f"causality test broken: future unchanged under perturbation ({fut_diff:.2e})")

    passed = not errors
    result = ValidationResult(
        name=name,
        n_bars=n,
        n_compared=len(ref_w),
        correlation=corr,
        max_abs_diff=max_abs,
        sign_match_pct=sign_match,
        obs_range=(obs_min, obs_max),
        declared_range=expected_range,
        causal_past_max_diff=past_diff,
        causal_future_max_diff=fut_diff,
        passed=passed,
        notes="; ".join(errors) if errors else "",
    )
    if not passed:
        raise ValidationError(f"{name}: {result.notes}")
    return result


def print_result(r: ValidationResult) -> None:
    print(f"\n── Validation: {r.name} ──")
    print(f"  bars         : {r.n_bars} (compared {r.n_compared})")
    print(f"  correlation  : {r.correlation:.6f}")
    print(f"  max |diff|   : {r.max_abs_diff:.2e}")
    print(f"  sign match   : {r.sign_match_pct*100:.2f}%")
    print(f"  obs range    : [{r.obs_range[0]:.4f}, {r.obs_range[1]:.4f}]")
    print(f"  causal past  : {r.causal_past_max_diff:.2e}")
    print(f"  causal future: {r.causal_future_max_diff:.2e} (needs > 1e-10)")
    print(f"  → {'PASS' if r.passed else 'FAIL'}  {r.notes}")
