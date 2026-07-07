#!/usr/bin/env python3
"""
equal_risk_portfolio.py — Task A5 secondary (c): equal-risk portfolio of whichever of the
three signals (first-touch H1 "signal" arm, CSI StrengthSpread, D1 RSI(2)) are IS-positive
at base cost (spread_mult=1.0, markup_mult=1.0). Same cost model as the rest of A5 (already
baked into each input file's net_base column). Exploratory — reported regardless of outcome,
never promoted to confirmatory (PREREGISTRATION.md).

Method (documented, one pass, no tuning):
  1. "IS-positive" = mean net_base pips > 0 over the whole IS window, for that signal's own
     unit of account (first-touch: per trade; StrengthSpread: per leg; RSI2: per trade).
  2. Only qualifying (IS-positive) signals are combined; if zero qualify, no portfolio is
     built (reported honestly, not forced).
  3. Each qualifying signal's trade/leg net_base pips are attributed to their EXIT date
     (UTC calendar date) and summed within that date -> one daily pip series per signal
     (zero-filled on non-trading days over its own active date range).
  4. Equal-RISK weights: w_i = (1/sigma_i) / sum_j(1/sigma_j), sigma_i = std of signal i's
     daily pip series over the union of all qualifying signals' trading days. Combined daily
     series = sum_i w_i * daily_i (documented simplification: this weights by volatility of
     PIPS, not of dollar P&L — pip units differ across pair mixes, a known limitation shared
     with the rest of this program's pip-pooling convention).
  5. Pairwise correlation of the qualifying signals' daily pip series (on the common/union
     date index, zero-filled) + a day-block bootstrap (2000 resamples) on the COMBINED daily
     series, reported for completeness (not gated).

Writes results/secondary_equal_risk.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

N_BOOT = 2000
BOOT_SEED = 20260706


def daily_series_from_trades(df, ts_col, val_col):
    if len(df) == 0:
        return pd.Series(dtype=float)
    d = df.copy()
    d["date"] = pd.to_datetime(d[ts_col]).dt.date
    return d.groupby("date")[val_col].sum()


def day_block_bootstrap_series(daily, n_boot=N_BOOT, seed=BOOT_SEED):
    vals = daily.values
    n = len(vals)
    if n == 0:
        return float("nan"), np.array([])
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = vals[idx].mean()
    p_le_zero = float(np.mean(boot_means <= 0))
    return p_le_zero, boot_means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out_dir = args.out_dir

    candidates = {}

    def _read_utc(path, cols):
        df = pd.read_csv(path)
        for c in cols:
            df[c] = pd.to_datetime(df[c], utc=True)
        return df

    ft_path = os.path.join(out_dir, "is_battery_trades.csv")
    if os.path.exists(ft_path):
        ft = _read_utc(ft_path, ["entry_ts", "exit_ts"])
        ft_signal = ft[ft["arm"] == "signal"]
        candidates["first_touch_signal"] = daily_series_from_trades(ft_signal, "exit_ts", "net_base")

    ss_path = os.path.join(out_dir, "secondary_strengthspread_legs.csv")
    if os.path.exists(ss_path):
        ss = _read_utc(ss_path, ["entry_ts", "exit_ts"])
        candidates["strengthspread"] = daily_series_from_trades(ss, "exit_ts", "net_pips")

    rsi_path = os.path.join(out_dir, "secondary_rsi2_trades.csv")
    if os.path.exists(rsi_path):
        rsi = _read_utc(rsi_path, ["entry_ts", "exit_ts"])
        candidates["rsi2_d1"] = daily_series_from_trades(rsi, "exit_ts", "net_base")

    means = {name: float(s.mean()) if len(s) else float("nan") for name, s in candidates.items()}
    qualifying = {name: s for name, s in candidates.items() if len(s) and s.mean() > 0}

    result = {
        "candidate_means_net_pips": means,
        "qualifying": list(qualifying.keys()),
    }

    if len(qualifying) == 0:
        result["portfolio_built"] = False
        result["note"] = "No signal was IS-positive at base cost; equal-risk portfolio not constructed."
    else:
        union_idx = sorted(set().union(*[set(s.index) for s in qualifying.values()]))
        aligned = {name: s.reindex(union_idx, fill_value=0.0) for name, s in qualifying.items()}
        aligned_df = pd.DataFrame(aligned)

        sigmas = aligned_df.std(ddof=1)
        inv_sigma = 1.0 / sigmas.replace(0, np.nan)
        weights = (inv_sigma / inv_sigma.sum()).fillna(0.0)

        combined = (aligned_df * weights).sum(axis=1)
        combined.index = pd.to_datetime(union_idx)

        corr = aligned_df.corr().round(4).to_dict() if aligned_df.shape[1] > 1 else {}

        p_le_zero, boot_means = day_block_bootstrap_series(combined)

        result["portfolio_built"] = True
        result["weights"] = weights.to_dict()
        result["n_days"] = int(len(combined))
        result["combined_mean_daily_pips"] = float(combined.mean())
        result["combined_std_daily_pips"] = float(combined.std(ddof=1))
        result["pairwise_correlation"] = corr
        result["mc_p_le_zero"] = p_le_zero
        result["mc_ci_2p5"] = float(np.percentile(boot_means, 2.5)) if len(boot_means) else float("nan")
        result["mc_ci_97p5"] = float(np.percentile(boot_means, 97.5)) if len(boot_means) else float("nan")

    with open(os.path.join(out_dir, "secondary_equal_risk.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
