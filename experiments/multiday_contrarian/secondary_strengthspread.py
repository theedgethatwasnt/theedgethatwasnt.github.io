#!/usr/bin/env python3
"""
secondary_strengthspread.py — Task A5 secondary (a): CSI StrengthSpread H4/64-bar port.

Source: ~/projects/csi_factor_study (accessible on this machine). Recorded prior:
"H4 H=64 N=3: IS Net Sharpe 0.41, OOS Net Sharpe 0.59" (README.md Experiment 2).
Implementation ported from src/factors/currency_strength.py + strength_spread.py +
src/strategy/contrarian_backtest.py, adapted to our 12-pair universe / cost model /
IS-only window. Documented deviations from the original (exploratory secondary, not
a strict replication — no parameter search performed, H=64/N=3 taken verbatim as the
recorded "winner" config, not re-tuned):

  1. Universe: our 12 pairs (8 currencies: USD EUR GBP JPY CHF AUD CAD NZD), not the
     original's 24 G10 crosses. Some currencies are thin in our universe (CHF and CAD
     each appear in only 1 pair: CHF_JPY, CAD_JPY) — their "strength" estimate is a
     single pair's return, not an aggregate. Flagged, not fixed (would require adding
     pairs outside our data footprint).
  2. Currency strength = cross-sectional z-score (at each H4 bar) of
     sum_{pairs containing ccy}(sign * 1-bar log return), sign=+1 if ccy is base else -1
     (verbatim currency_strength.py formula). Pair spread = strength[base] - strength[quote].
  3. Rebalance: non-overlapping every H=64 H4 bars. At each rebalance, long the bottom
     N=3 pairs by spread (lowest = weakest-base-relative, contrarian per the original's
     negative-IC finding: "go long the lowest-ranked pairs, short the highest-ranked"),
     short the top N=3. Equal-weighted within each side.
  4. Return/cost convention DELIBERATELY SIMPLER than the primary harness (documented,
     not an oversight): entry/exit price = the H4 bar's own MID CLOSE at the rebalance
     bar and at bar+64 (not harness's next-M5-open-after-close convention) — this
     matches how the original factor study computes forward returns. Spread cost uses
     the H4 bar's own last-M5-bar bid_c/ask_c (from bars.py's H4 aggregation) at both
     ends. Carry uses carry_model.carry_pips at the same (entry_ts, exit_ts) pair.
  5. CONSERVATIVE cost simplification: every leg pays a full round-trip spread+carry at
     EVERY rebalance, even when a pair happens to stay on the same side across two
     consecutive rebalances (no turnover netting, unlike the original project's
     `compute_realistic_costs`). This overstates costs versus a live implementation that
     would only pay on actual position changes — a conservative bias, not favorable.
  6. Units: net/gross reported in PIPS per leg (matching this program's house convention
     elsewhere), NOT the original project's Sharpe/annualized-return units — pips
     aggregated across JPY and non-JPY pairs without price-normalization, the same
     simplification this codebase's live dashboards already use for "portfolio pips/day".

Writes results/secondary_strengthspread_legs.csv (one row per leg per rebalance) and
prints/returns a summary dict.
"""
import argparse
import gc
import json
import os

import numpy as np
import pandas as pd

from bars import m5_to_h4
from carry_model import carry_pips, pip_of
from is_data import IS_END, PAIRS, load_pair_is

H = 64      # holding period, H4 bars (verbatim recorded "winner": H4 H=64 N=3)
N = 3       # pairs per side


def build_currency_pair_map(pairs):
    m = {}
    for pair in pairs:
        base, quote = pair.split("_")
        m.setdefault(base, []).append((pair, +1))
        m.setdefault(quote, []).append((pair, -1))
    return m


def load_h4_panel(data_dir):
    """One pair at a time (memory-safety default); returns dict[pair] -> H4 df with
    columns timestamp/open/high/low/close/volume/bid_c/ask_c, IS-only."""
    h4 = {}
    for pair in PAIRS:
        df = load_pair_is(pair, data_dir)
        h4[pair] = m5_to_h4(df)
        del df
        gc.collect()
    return h4


def build_strength_and_spread(h4_panel):
    """Returns (spread_df, close_df, bidc_df, askc_df) all aligned on the INNER-JOIN
    timestamp index common to all 12 pairs' H4 bars."""
    common_idx = None
    for pair, df in h4_panel.items():
        idx = pd.DatetimeIndex(df["timestamp"])
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()

    close = {}
    bidc = {}
    askc = {}
    ret = {}
    for pair, df in h4_panel.items():
        d = df.set_index("timestamp").reindex(common_idx)
        close[pair] = d["close"]
        bidc[pair] = d["bid_c"]
        askc[pair] = d["ask_c"]
        ret[pair] = np.log(d["close"]).diff()

    close_df = pd.DataFrame(close)
    bidc_df = pd.DataFrame(bidc)
    askc_df = pd.DataFrame(askc)
    ret_df = pd.DataFrame(ret)

    ccy_map = build_currency_pair_map(PAIRS)
    strength = {}
    for ccy, pairs_signs in ccy_map.items():
        s = pd.Series(0.0, index=common_idx)
        for pair, sign in pairs_signs:
            s = s + sign * ret_df[pair]
        strength[ccy] = s
    strength_df = pd.DataFrame(strength)

    mean = strength_df.mean(axis=1)
    std = strength_df.std(axis=1).replace(0, np.nan)
    z = strength_df.sub(mean, axis=0).div(std, axis=0)

    spread = {}
    for pair in PAIRS:
        base, quote = pair.split("_")
        spread[pair] = z[base] - z[quote]
    spread_df = pd.DataFrame(spread)

    return spread_df, close_df, bidc_df, askc_df


def run_backtest(spread_df, close_df, bidc_df, askc_df):
    n = len(spread_df)
    starts = [p for p in range(0, n, H) if p + H < n]
    legs = []
    for p in starts:
        row = spread_df.iloc[p].dropna()
        if len(row) < 2 * N:
            continue
        longs = row.nsmallest(N).index.tolist()   # contrarian: long the lowest spread
        shorts = row.nlargest(N).index.tolist()   # short the highest spread
        entry_ts = spread_df.index[p]
        exit_ts = spread_df.index[p + H]
        for pair in longs + shorts:
            direction = 1 if pair in longs else -1
            pip = pip_of(pair)
            entry_px = close_df[pair].iloc[p]
            exit_px = close_df[pair].iloc[p + H]
            if pd.isna(entry_px) or pd.isna(exit_px):
                continue
            gross_pips = direction * (exit_px - entry_px) / pip
            entry_spread = (askc_df[pair].iloc[p] - bidc_df[pair].iloc[p]) / pip
            exit_spread = (askc_df[pair].iloc[p + H] - bidc_df[pair].iloc[p + H]) / pip
            spread_rt = np.nanmean([entry_spread, exit_spread])
            carry = carry_pips(pair, direction, entry_ts, exit_ts, markup_mult=1.0)
            net_pips = gross_pips - spread_rt + carry
            legs.append({
                "rebal_idx": p, "entry_ts": entry_ts, "exit_ts": exit_ts,
                "pair": pair, "direction": direction, "side": "long" if direction > 0 else "short",
                "gross_pips": gross_pips, "spread_rt_pips": spread_rt, "carry_pips": carry,
                "net_pips": net_pips,
            })
    return pd.DataFrame(legs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    h4_panel = load_h4_panel(args.data_dir)
    assert all(pd.Timestamp(df["timestamp"].max()) < IS_END for df in h4_panel.values()), "OOS LEAK"
    spread_df, close_df, bidc_df, askc_df = build_strength_and_spread(h4_panel)
    del h4_panel
    gc.collect()

    legs_df = run_backtest(spread_df, close_df, bidc_df, askc_df)
    legs_df.to_csv(os.path.join(args.out_dir, "secondary_strengthspread_legs.csv"), index=False)

    n_rebal = legs_df["rebal_idx"].nunique() if len(legs_df) else 0
    summary = {
        "n_legs": int(len(legs_df)),
        "n_rebalances": int(n_rebal),
        "mean_gross_pips_per_leg": float(legs_df["gross_pips"].mean()) if len(legs_df) else float("nan"),
        "mean_net_pips_per_leg": float(legs_df["net_pips"].mean()) if len(legs_df) else float("nan"),
        "frac_legs_net_positive": float((legs_df["net_pips"] > 0).mean()) if len(legs_df) else float("nan"),
        "H": H, "N": N,
    }
    with open(os.path.join(args.out_dir, "secondary_strengthspread_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
