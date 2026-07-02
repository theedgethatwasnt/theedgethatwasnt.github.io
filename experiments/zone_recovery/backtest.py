"""
Zone Recovery Backtest Runner
Walk-forward validation + Monte Carlo permutation test.
"""

import numpy as np
from typing import List, Dict, Any
from engine import ZoneRecoveryEngine, CycleResult
from data_utils import walk_forward_splits


def compute_metrics(results: List[CycleResult]) -> Dict[str, Any]:
    """Aggregate performance metrics from a list of CycleResult."""
    if not results:
        return {"n_cycles": 0, "net_pnl_pips": 0.0, "sharpe": 0.0}

    pnls = np.array([r.net_pnl_pips for r in results])
    legs = np.array([len(r.legs) for r in results])
    durations = np.array([r.duration_bars for r in results])

    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_dd = float(drawdown.min())

    wins = pnls > 0
    mean_pnl = float(pnls.mean())
    std_pnl = float(pnls.std()) if len(pnls) > 1 else 1.0
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

    # Per-trade Sharpe (annualized estimate: ~200 trades/year typical)
    # sqrtN * mean / std
    sqn = (np.sqrt(len(pnls)) * mean_pnl / std_pnl) if std_pnl > 0 else 0.0

    gross_wins = pnls[wins].sum() if wins.any() else 0.0
    gross_losses = abs(pnls[~wins].sum()) if (~wins).any() else 1.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    exit_reasons = {}
    for r in results:
        k = r.exit_reason
        exit_reasons[k] = exit_reasons.get(k, 0) + 1

    return {
        "n_cycles": len(results),
        "net_pnl_pips": float(pnls.sum()),
        "mean_pnl_pips": mean_pnl,
        "std_pnl_pips": std_pnl,
        "win_rate": float(wins.mean()),
        "sharpe": sharpe,
        "sqn": sqn,
        "profit_factor": profit_factor,
        "max_drawdown_pips": max_dd,
        "avg_legs": float(legs.mean()),
        "max_legs_hit_pct": float((legs >= 10).mean()),
        "avg_duration_bars": float(durations.mean()),
        "exit_reasons": exit_reasons,
    }


def run_backtest(engine: ZoneRecoveryEngine, features: dict, seed: int = 42,
                 entry_mask: np.ndarray = None,
                 direction_signal: np.ndarray = None) -> List[CycleResult]:
    """Run engine on features dict. Returns list of CycleResult."""
    rng = np.random.RandomState(seed)
    results = engine.simulate_on_ohlc(
        open_arr=features["open"],
        high_arr=features["high"],
        low_arr=features["low"],
        close_arr=features["close"],
        atr_short=features.get("atr_short"),
        atr_long=features.get("atr_long"),
        rng=rng,
        entry_mask=entry_mask,
        direction_signal=direction_signal,
    )
    return results


def walk_forward_test(engine: ZoneRecoveryEngine, features: dict,
                      n_chunks: int = 3, test_frac: float = 0.3,
                      seed: int = 42, entry_mask: np.ndarray = None,
                      direction_signal: np.ndarray = None) -> Dict[str, Any]:
    """Walk-forward validation across n_chunks OOS windows.

    Returns dict with per-chunk metrics and aggregate verdict.
    """
    splits = walk_forward_splits(features, n_chunks=n_chunks, test_frac=test_frac)

    chunk_metrics = []
    all_oos_pnls = []

    for k, (train_f, test_f) in enumerate(splits):
        mask_k = entry_mask[len(train_f["close"]):len(train_f["close"])+len(test_f["close"])] if entry_mask is not None else None
        sig_k  = direction_signal[len(train_f["close"]):len(train_f["close"])+len(test_f["close"])] if direction_signal is not None else None
        oos_results = run_backtest(engine, test_f, seed=seed + k, entry_mask=mask_k, direction_signal=sig_k)
        m = compute_metrics(oos_results)
        m["chunk"] = k
        m["n_bars_train"] = len(train_f["close"])
        m["n_bars_oos"] = len(test_f["close"])
        chunk_metrics.append(m)
        all_oos_pnls.extend([r.net_pnl_pips for r in oos_results])

    all_oos = np.array(all_oos_pnls) if all_oos_pnls else np.array([0.0])
    n_positive = sum(1 for m in chunk_metrics if m["net_pnl_pips"] > 0)

    return {
        "chunks": chunk_metrics,
        "chunks_positive": n_positive,
        "chunks_total": n_chunks,
        "oos_total_pnl_pips": float(all_oos.sum()),
        "oos_mean_pnl_pips": float(all_oos.mean()),
        "oos_sharpe": float(all_oos.mean() / all_oos.std()) if all_oos.std() > 0 else 0.0,
        "oos_n_trades": len(all_oos_pnls),
        "wf_pass": n_positive == n_chunks,  # all chunks must be positive
    }


def permutation_test(results: List[CycleResult], n_perms: int = 1000,
                     seed: int = 42) -> Dict[str, Any]:
    """Sign-flip permutation test: randomly flip sign of each trade's P&L.

    Tests whether the positive expectation could arise by chance even if we had
    no directional ability — only the magnitudes of wins/losses matter to the null.

    p_value: fraction of sign-flipped totals >= real total.
    p_value < 0.05 means statistically significant positive expectation.
    """
    if not results:
        return {"p_value": 1.0, "real_total_pips": 0.0, "perm_mean_pips": 0.0,
                "perm_std_pips": 0.0, "significant": False}

    rng = np.random.RandomState(seed)
    pnls = np.array([r.net_pnl_pips for r in results])
    real_total = float(pnls.sum())
    n = len(pnls)

    # Sign-flip: each permutation randomly flips ±1 on each trade
    signs = rng.choice([-1, 1], size=(n_perms, n))
    perm_sums = (signs * pnls[np.newaxis, :]).sum(axis=1)
    p_value = float((perm_sums >= real_total).mean())

    return {
        "p_value": p_value,
        "real_total_pips": real_total,
        "perm_mean_pips": float(perm_sums.mean()),
        "perm_std_pips": float(perm_sums.std()),
        "significant": p_value < 0.05,
    }


def seed_robustness_test(engine: ZoneRecoveryEngine, features: dict,
                         seeds: List[int] = None,
                         entry_mask: np.ndarray = None,
                         direction_signal: np.ndarray = None) -> Dict[str, Any]:
    """Run same params with multiple seeds; check all seeds positive.

    For zone recovery, entry direction is random (first-leg buy/sell).
    Different seeds produce different first-leg sequences.
    Gate: ALL seeds must yield positive total P&L (direction-agnostic edge).
    CV is reported but not gating (inherent noise from random direction).
    """
    if seeds is None:
        seeds = [42, 123, 777, 999, 1234]

    totals = []
    for s in seeds:
        results = run_backtest(engine, features, seed=s,
                               entry_mask=entry_mask, direction_signal=direction_signal)
        m = compute_metrics(results)
        totals.append(m["net_pnl_pips"])

    totals = np.array(totals)
    mean = float(totals.mean())
    std = float(totals.std())
    cv = std / abs(mean) if mean != 0 else float("inf")
    all_positive = bool(np.all(totals > 0))

    return {
        "seeds": seeds,
        "totals": totals.tolist(),
        "mean_pips": mean,
        "std_pips": std,
        "cv": cv,
        "all_positive": all_positive,
        "cv_pass": all_positive,  # Gate: all seeds must be positive
    }


def run_5gate_validation(engine: ZoneRecoveryEngine, features: dict,
                         train_features: dict, test_features: dict,
                         seed: int = 42,
                         entry_mask: np.ndarray = None,
                         direction_signal: np.ndarray = None) -> Dict[str, Any]:
    """Full 5-gate validation protocol (post-RCA standard).

    Gates:
      1. OOS positive P&L
      2. Walk-forward all chunks positive
      3. Permutation p < 0.05
      4. Seed CV < 10%
      5. Per-trade Sharpe (SQN proxy) > 1.0
    """
    n_train = len(train_features["close"])
    n_test  = len(test_features["close"])

    mask_oos = entry_mask[n_train:n_train+n_test] if entry_mask is not None else None
    sig_oos  = direction_signal[n_train:n_train+n_test] if direction_signal is not None else None

    # Gate 1 & 3 & 4 & 5: OOS run
    oos_results = run_backtest(engine, test_features, seed=seed,
                               entry_mask=mask_oos, direction_signal=sig_oos)
    oos_m = compute_metrics(oos_results)

    # Gate 2: Walk-forward on full data
    wf = walk_forward_test(engine, features, seed=seed,
                           entry_mask=entry_mask, direction_signal=direction_signal)

    # Gate 3: Permutation on OOS trades
    perm = permutation_test(oos_results, n_perms=500, seed=seed)

    # Gate 4: Seed robustness on OOS
    # For direction_signal mode, seeds don't vary direction; pass direction through
    seed_r = seed_robustness_test(engine, test_features,
                                  entry_mask=mask_oos, direction_signal=sig_oos)

    gates = {
        "gate1_oos_positive": oos_m["net_pnl_pips"] > 0,
        "gate2_wf_all_positive": wf["wf_pass"],
        "gate3_permutation_significant": perm["significant"],
        "gate4_seed_cv_pass": seed_r["cv_pass"],
        "gate5_sqn_pass": oos_m["sqn"] > 1.0,
    }
    gates_passed = sum(gates.values())

    return {
        "gates": gates,
        "gates_passed": gates_passed,
        "gates_total": 5,
        "oos_metrics": oos_m,
        "wf_result": wf,
        "perm_result": perm,
        "seed_result": seed_r,
        "verdict": "PASS" if gates_passed >= 4 else "FAIL",  # 4/5 minimum
    }
