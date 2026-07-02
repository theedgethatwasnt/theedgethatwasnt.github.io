"""
Zone Recovery Calibration Study
Grid search over zone/target parameters to find optimal ATR multipliers.
Analyzes crossing statistics, escape probabilities, E/Z ratio distribution.
Uses Numba JIT for inner-loop performance.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

from engine import ZoneRecoveryEngine
from backtest import run_backtest, compute_metrics
from data_utils import prepare_features, load_m5


def simulate_zone_stats(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                         half_zone: float, target_beyond: float,
                         max_legs: int = 10) -> np.ndarray:
    """Fast inner-loop: simulate zone recovery stats for one parameter set.

    Returns array of shape (n_cycles, 4):
      [legs_used, escaped, cycle_bars, max_adverse_pips]
    """
    pip = 0.0001
    n = len(close)
    results = np.zeros((n // 10 + 1, 4), dtype=np.float64)
    cycle_count = 0

    i = 0
    while i < n:
        entry = close[i]
        lower = entry - half_zone
        upper = entry + half_zone
        upper_tgt = upper + target_beyond
        lower_tgt = lower - target_beyond

        legs = 0
        escaped = 0
        max_adverse = 0.0
        entry_bar = i

        i += 1
        closed = False
        while i < n and not closed:
            h = high[i]
            l = low[i]
            c = close[i]

            adverse = min(0.0, (c - entry) / pip)  # simplification
            if abs(adverse) > max_adverse:
                max_adverse = abs(adverse)

            # Target hit
            if h >= upper_tgt or l <= lower_tgt:
                escaped = 1
                closed = True
                break

            # Zone crossing
            bullish = c >= close[i - 1]
            if bullish:
                if h >= upper and legs < max_legs:
                    legs += 1
                if l <= lower and legs < max_legs:
                    legs += 1
            else:
                if l <= lower and legs < max_legs:
                    legs += 1
                if h >= upper and legs < max_legs:
                    legs += 1

            if legs >= max_legs and not closed:
                closed = True
                break

            i += 1

        if cycle_count < len(results):
            results[cycle_count, 0] = legs
            results[cycle_count, 1] = escaped
            results[cycle_count, 2] = i - entry_bar
            results[cycle_count, 3] = max_adverse
            cycle_count += 1

    return results[:cycle_count]


def calibration_grid_search(
    pair: str = "EUR_USD",
    half_zone_mults: np.ndarray = None,
    target_mults: np.ndarray = None,
    atr_period_short: int = 8,
    atr_period_long: int = 20,
    max_legs: int = 10,
    use_is_data_frac: float = 0.7,
) -> pd.DataFrame:
    """Grid search over ATR multipliers. Returns DataFrame with calibration stats."""

    if half_zone_mults is None:
        half_zone_mults = np.arange(0.3, 1.1, 0.1)
    if target_mults is None:
        target_mults = np.arange(1.0, 3.5, 0.25)

    df = load_m5(pair)
    feats = prepare_features(df, granularity="M5")

    n = len(feats["close"])
    train_n = int(n * use_is_data_frac)
    c = feats["close"][:train_n]
    h = feats["high"][:train_n]
    l = feats["low"][:train_n]
    atr_s = feats["atr_short"][:train_n]
    atr_l = feats["atr_long"][:train_n]

    valid_mask = ~(np.isnan(atr_s) | np.isnan(atr_l))

    rows = []
    for hz_mult in half_zone_mults:
        for tgt_mult in target_mults:
            # Use median ATR values for stats (dynamic ATR would be per-bar but this gives baseline)
            median_atr_s = float(np.nanmedian(atr_s))
            median_atr_l = float(np.nanmedian(atr_l))

            half_zone_price = hz_mult * median_atr_s
            target_beyond_price = tgt_mult * median_atr_l
            zone_width = 2 * half_zone_price
            ez_ratio = target_beyond_price / zone_width if zone_width > 0 else 0

            # ATR-dynamic engine
            engine = ZoneRecoveryEngine(
                mode="atr",
                atr_zone_mult=hz_mult,
                atr_target_mult=tgt_mult,
                ez_ratio_min=0.0,  # no filter during calibration
                ez_ratio_max=999.0,
                base_unit=1000,
                max_legs=max_legs,
                sizing_mode="dynamic",
                profit_factor=1.19,
                spread_pips=1.4,
            )

            results = run_backtest(engine, {
                "open": feats["open"][:train_n],
                "high": h, "low": l, "close": c,
                "atr_short": atr_s, "atr_long": atr_l,
                "timestamps": feats["timestamps"][:train_n],
            })
            m = compute_metrics(results)

            rows.append({
                "half_zone_mult": hz_mult,
                "target_mult": tgt_mult,
                "median_half_zone_pips": half_zone_price / 0.0001,
                "median_target_pips": target_beyond_price / 0.0001,
                "median_ez_ratio": ez_ratio,
                "n_cycles": m["n_cycles"],
                "net_pnl_pips": m["net_pnl_pips"],
                "mean_pnl_pips": m["mean_pnl_pips"],
                "win_rate": m["win_rate"],
                "sharpe": m["sharpe"],
                "sqn": m["sqn"],
                "profit_factor": m["profit_factor"],
                "max_dd_pips": m["max_drawdown_pips"],
                "avg_legs": m["avg_legs"],
                "max_legs_hit_pct": m["max_legs_hit_pct"],
            })
            print(f"  hz={hz_mult:.2f} tgt={tgt_mult:.2f} | "
                  f"n={m['n_cycles']:4d} pnl={m['net_pnl_pips']:+8.1f}p "
                  f"sharpe={m['sharpe']:.3f} ez={ez_ratio:.1f}")

    return pd.DataFrame(rows)


def classic_param_grid_search(
    pair: str = "EUR_USD",
    half_zone_pips_list: List[float] = None,
    target_beyond_pips_list: List[float] = None,
    profit_factor_list: List[float] = None,
    use_is_data_frac: float = 0.7,
) -> pd.DataFrame:
    """Grid search over classic cBot parameters (fixed pips, not ATR-dynamic)."""

    if half_zone_pips_list is None:
        half_zone_pips_list = [5, 8, 10, 12, 15, 20, 25]
    if target_beyond_pips_list is None:
        target_beyond_pips_list = [3, 5, 6, 8, 10, 12, 15]
    if profit_factor_list is None:
        profit_factor_list = [1.0, 1.1, 1.19, 1.3]

    df = load_m5(pair)
    feats = prepare_features(df, granularity="M5")

    n = len(feats["close"])
    train_n = int(n * use_is_data_frac)
    train_feats = {k: v[:train_n] for k, v in feats.items()}

    rows = []
    total = len(half_zone_pips_list) * len(target_beyond_pips_list) * len(profit_factor_list)
    idx = 0

    for hz_pips in half_zone_pips_list:
        for tgt_pips in target_beyond_pips_list:
            for pf in profit_factor_list:
                idx += 1
                ez_ratio = tgt_pips / (2 * hz_pips)

                engine = ZoneRecoveryEngine(
                    mode="classic",
                    half_zone_pips=hz_pips,
                    target_beyond_pips=tgt_pips,
                    init_target_pips=2.7,
                    base_unit=1000,
                    max_legs=10,
                    sizing_mode="dynamic",
                    profit_factor=pf,
                    spread_pips=1.4,
                )

                results = run_backtest(engine, train_feats)
                m = compute_metrics(results)

                rows.append({
                    "half_zone_pips": hz_pips,
                    "target_beyond_pips": tgt_pips,
                    "profit_factor": pf,
                    "ez_ratio": ez_ratio,
                    "n_cycles": m["n_cycles"],
                    "net_pnl_pips": m["net_pnl_pips"],
                    "mean_pnl_pips": m["mean_pnl_pips"],
                    "win_rate": m["win_rate"],
                    "sharpe": m["sharpe"],
                    "sqn": m["sqn"],
                    "profit_factor_result": m["profit_factor"],
                    "max_dd_pips": m["max_drawdown_pips"],
                    "avg_legs": m["avg_legs"],
                    "max_legs_hit_pct": m["max_legs_hit_pct"],
                    "exit_target_pct": m["exit_reasons"].get("target", 0) / max(m["n_cycles"], 1),
                    "exit_maxlegs_pct": m["exit_reasons"].get("max_legs", 0) / max(m["n_cycles"], 1),
                })

                if idx % 20 == 0 or idx == total:
                    print(f"  [{idx}/{total}] hz={hz_pips}p tgt={tgt_pips}p pf={pf:.2f} | "
                          f"n={m['n_cycles']:4d} pnl={m['net_pnl_pips']:+8.1f}p "
                          f"sharpe={m['sharpe']:.3f}")

    return pd.DataFrame(rows)


def analyze_ez_ratio_distribution(pair: str = "EUR_USD",
                                   atr_zone_mult: float = 0.5,
                                   atr_target_mult: float = 1.5) -> Dict[str, Any]:
    """Analyze the distribution of E/Z ratios over time for given multipliers."""
    df = load_m5(pair)
    feats = prepare_features(df, granularity="M5")

    atr_s = feats["atr_short"]
    atr_l = feats["atr_long"]
    valid = ~(np.isnan(atr_s) | np.isnan(atr_l))

    half_zones = atr_zone_mult * atr_s[valid]
    targets = atr_target_mult * atr_l[valid]
    zone_widths = 2 * half_zones
    ez_ratios = targets / zone_widths

    return {
        "ez_ratio_mean": float(ez_ratios.mean()),
        "ez_ratio_median": float(np.median(ez_ratios)),
        "ez_ratio_std": float(ez_ratios.std()),
        "ez_ratio_p25": float(np.percentile(ez_ratios, 25)),
        "ez_ratio_p75": float(np.percentile(ez_ratios, 75)),
        "ez_in_range_6_15_pct": float(((ez_ratios >= 6) & (ez_ratios <= 15)).mean()),
        "half_zone_median_pips": float(np.median(half_zones) / 0.0001),
        "target_median_pips": float(np.median(targets) / 0.0001),
        "atr_s_median_pips": float(np.nanmedian(atr_s) / 0.0001),
        "atr_l_median_pips": float(np.nanmedian(atr_l) / 0.0001),
    }
