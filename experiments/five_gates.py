#!/usr/bin/env python3
"""
5-Gate Statistical Validation Protocol
=======================================
Used by all experiments in the Book Review V2 campaign.

Gates (must ALL pass for a configuration to be considered deployable):
  1. Point estimate : OOS Sharpe > threshold (default 0.5)
  2. Permutation    : shuffle signal 1,000×; real result exceeds ≥95% of shuffles
  3. Bootstrap CI   : 1,000 bootstrap resamples; 95% CI lower bound > 0
  4. Walk-forward   : every temporal fold positive independently
  5. Drop-one       : remove each year of OOS; strategy remains positive

Reference: CLAUDE.md §MCMC Statistical Gates
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

# ── Helpers ───────────────────────────────────────────────────────────

def sharpe_from_returns(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Annualised Sharpe from a series of per-period returns."""
    if len(returns) < 4:
        return float('nan')
    mu = np.nanmean(returns)
    sd = np.nanstd(returns, ddof=1)
    if sd < 1e-12:
        return float('nan')
    return float(mu / sd * np.sqrt(periods_per_year))


def equity_sharpe(equity: pd.Series, periods_per_year: float = 252.0) -> float:
    """Sharpe from an equity (cumulative pips) curve — uses period-to-period changes."""
    rets = equity.diff().dropna()
    return sharpe_from_returns(rets.values, periods_per_year)


# ── Gate implementations ──────────────────────────────────────────────

def gate1_point_estimate(oos_sharpe: float, threshold: float = 0.5) -> dict:
    passed = oos_sharpe > threshold
    return {"gate": 1, "name": "point_estimate", "passed": passed,
            "oos_sharpe": round(oos_sharpe, 4), "threshold": threshold}


def gate2_permutation(
    real_score: float,
    score_fn: Callable[[np.ndarray], float],
    signal: np.ndarray,
    n_perms: int = 1000,
    alpha: float = 0.05,
    mode: str = "time",
) -> dict:
    """
    Permutation test.

    mode = 'time'         : shuffle signal across time (for temporal strategies)
    mode = 'pairs_per_bar': shuffle across pairs at each time step (for cross-sectional)
    """
    rng = np.random.default_rng(seed=42)
    perm_scores = np.empty(n_perms)

    if mode == "time":
        for i in range(n_perms):
            shuffled = rng.permutation(signal)
            perm_scores[i] = score_fn(shuffled)
    elif mode == "pairs_per_bar":
        # signal shape: (T, N_pairs); shuffle columns within each row
        sig2d = signal.copy()
        for i in range(n_perms):
            shuf = sig2d.copy()
            for row in range(shuf.shape[0]):
                rng.shuffle(shuf[row])
            perm_scores[i] = score_fn(shuf)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    p_value = float(np.mean(perm_scores >= real_score))
    perm_95th = float(np.percentile(perm_scores, 95))
    passed = p_value < alpha
    return {"gate": 2, "name": "permutation", "passed": passed,
            "p_value": round(p_value, 4), "n_perms": n_perms,
            "perm_95th": round(perm_95th, 4), "real_score": round(real_score, 4)}


def gate3_bootstrap_ci(
    returns: np.ndarray,
    n_boot: int = 1000,
    confidence: float = 0.95,
    periods_per_year: float = 252.0,
) -> dict:
    """Bootstrap 95% CI on OOS Sharpe. Pass if lower bound > 0."""
    rng = np.random.default_rng(seed=42)
    n = len(returns)
    boot_sharpes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_sharpes[i] = sharpe_from_returns(returns[idx], periods_per_year)
    lo = float(np.nanpercentile(boot_sharpes, 100 * (1 - confidence) / 2))
    hi = float(np.nanpercentile(boot_sharpes, 100 * (1 - (1 - confidence) / 2)))
    passed = lo > 0.0
    return {"gate": 3, "name": "bootstrap_ci", "passed": passed,
            "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            "n_boot": n_boot}


def gate4_walk_forward(
    fold_sharpes: List[float],
    require_all_positive: bool = True,
) -> dict:
    """All walk-forward folds must be positive (or fraction specified)."""
    n_total = len(fold_sharpes)
    n_positive = sum(1 for s in fold_sharpes if s > 0)
    passed = (n_positive == n_total) if require_all_positive else (n_positive >= n_total * 0.75)
    return {"gate": 4, "name": "walk_forward", "passed": passed,
            "fold_sharpes": [round(s, 4) for s in fold_sharpes],
            "n_folds": n_total, "n_positive": n_positive}


def gate5_drop_one(
    annual_returns: dict,  # {year: returns_array}
    periods_per_year: float = 252.0,
) -> dict:
    """Drop each year; remaining data must still produce positive Sharpe."""
    years = sorted(annual_returns.keys())
    all_returns = np.concatenate([annual_returns[y] for y in years])
    fold_results = {}
    all_pass = True
    for drop_year in years:
        kept = np.concatenate([annual_returns[y] for y in years if y != drop_year])
        s = sharpe_from_returns(kept, periods_per_year)
        fold_results[str(drop_year)] = round(s, 4)
        if s <= 0:
            all_pass = False
    passed = all_pass
    return {"gate": 5, "name": "drop_one", "passed": passed,
            "year_sharpes": fold_results}


# ── Composite runner ─────────────────────────────────────────────────

@dataclass
class GateResult:
    config: dict
    gates: List[dict] = field(default_factory=list)
    passed_all: bool = False
    n_gates_passed: int = 0

    def to_dict(self):
        return {
            "config": self.config,
            "gates": self.gates,
            "passed_all": self.passed_all,
            "n_gates_passed": self.n_gates_passed,
        }


def run_all_gates(
    config: dict,
    oos_sharpe: float,
    oos_returns: np.ndarray,
    signal_for_perm: np.ndarray,
    perm_score_fn: Callable,
    fold_sharpes: List[float],
    annual_returns: dict,
    perm_mode: str = "time",
    sharpe_threshold: float = 0.5,
    periods_per_year: float = 252.0,
    n_perms: int = 1000,
    verbose: bool = True,
) -> GateResult:
    result = GateResult(config=config)
    gates = []

    g1 = gate1_point_estimate(oos_sharpe, sharpe_threshold)
    gates.append(g1)
    if verbose:
        status = "✅" if g1["passed"] else "❌"
        print(f"  {status} Gate 1 (point estimate): Sharpe={g1['oos_sharpe']:.3f} > {sharpe_threshold}")

    g2 = gate2_permutation(oos_sharpe, perm_score_fn, signal_for_perm, n_perms, mode=perm_mode)
    gates.append(g2)
    if verbose:
        status = "✅" if g2["passed"] else "❌"
        print(f"  {status} Gate 2 (permutation):    p={g2['p_value']:.4f}")

    g3 = gate3_bootstrap_ci(oos_returns, periods_per_year=periods_per_year)
    gates.append(g3)
    if verbose:
        status = "✅" if g3["passed"] else "❌"
        print(f"  {status} Gate 3 (bootstrap CI):   [{g3['ci_lower']:.3f}, {g3['ci_upper']:.3f}]")

    g4 = gate4_walk_forward(fold_sharpes)
    gates.append(g4)
    if verbose:
        status = "✅" if g4["passed"] else "❌"
        print(f"  {status} Gate 4 (walk-forward):   {g4['n_positive']}/{g4['n_folds']} folds positive")

    g5 = gate5_drop_one(annual_returns, periods_per_year)
    gates.append(g5)
    if verbose:
        status = "✅" if g5["passed"] else "❌"
        print(f"  {status} Gate 5 (drop-one):       {g5['year_sharpes']}")

    result.gates = gates
    result.n_gates_passed = sum(1 for g in gates if g["passed"])
    result.passed_all = all(g["passed"] for g in gates)
    return result


if __name__ == "__main__":
    # Smoke test with synthetic data
    rng = np.random.default_rng(42)
    n = 500
    returns = rng.normal(0.05, 1.0, n)  # Slightly positive

    fold_sharpes = [0.3, 0.5, 0.7, 0.4, 0.6]
    annual_returns = {
        2021: rng.normal(0.05, 1.0, 100),
        2022: rng.normal(0.05, 1.0, 100),
        2023: rng.normal(0.05, 1.0, 100),
        2024: rng.normal(0.05, 1.0, 100),
        2025: rng.normal(0.05, 1.0, 100),
    }

    signal = rng.normal(0, 1, n)
    def score_fn(sig):
        return sharpe_from_returns(sig * np.sign(returns), 252.0)

    print("Five-gates smoke test:")
    result = run_all_gates(
        config={"test": True},
        oos_sharpe=sharpe_from_returns(returns, 52.0),
        oos_returns=returns,
        signal_for_perm=signal,
        perm_score_fn=score_fn,
        fold_sharpes=fold_sharpes,
        annual_returns=annual_returns,
        periods_per_year=52.0,
        n_perms=200,
        verbose=True,
    )
    print(f"\n  Passed {result.n_gates_passed}/5 gates. All passed: {result.passed_all}")
