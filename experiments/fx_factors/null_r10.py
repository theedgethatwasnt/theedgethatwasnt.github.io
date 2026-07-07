#!/usr/bin/env python3
"""
null_r10.py — R10 null: 200 random-weight portfolios on the IDENTICAL rebalance schedule and
gross exposure as the primary (gated-carry) factor (PREREGISTRATION.md H1 + Gates 1/3).

Construction: at every rebalance date, draw a random permutation of the fixed pattern
[+1,+1,+1,-1,-1,-1,0] across the 7 non-USD currencies (3 long / 3 short / 1 flat — the
carry factor's modal book size) via a per-seed, per-date deterministic RNG stream
(`np.random.default_rng((seed, date_key))`). Runs through the EXACT SAME run_portfolio()
engine (R6) — spread + carry costs, equal-risk 63d-vol weighting — as every factor variant,
so the null's expected net return is "pure cost drag with no directional edge," which is
exactly what Gate 1 (harness self-test) and Gate 3 (gated carry vs null 95th pct) need.
"""
import numpy as np
import pandas as pd

from currency_index import NON_USD_CURRENCIES
from rebalance_engine import run_portfolio

N_SEEDS = 200
BASE_PATTERN = [1, 1, 1, -1, -1, -1, 0]


def _date_key(d):
    return int(pd.Timestamp(d).value % (2**31 - 1))


def _random_direction_fn(seed):
    def fn(sig_d):
        rng = np.random.default_rng((seed, _date_key(sig_d)))
        perm = rng.permutation(BASE_PATTERN)
        return dict(zip(NON_USD_CURRENCIES, perm.tolist()))
    return fn


def run_null(pair_d1, schedule, n_seeds=N_SEEDS, spread_mult=1.0, markup_mult=1.0, seed_offset=0):
    """Returns a DataFrame with one row per seed: [seed, mean_net_pips, n_rebalances]."""
    rows = []
    for s in range(seed_offset, seed_offset + n_seeds):
        direction_fn = _random_direction_fn(s)
        monthly_df, _ = run_portfolio(pair_d1, schedule, direction_fn,
                                       spread_mult=spread_mult, markup_mult=markup_mult)
        rows.append({
            "seed": s,
            "mean_net_pips": float(monthly_df["net_pips"].mean()) if len(monthly_df) else float("nan"),
            "n_rebalances": len(monthly_df),
        })
    return pd.DataFrame(rows)
