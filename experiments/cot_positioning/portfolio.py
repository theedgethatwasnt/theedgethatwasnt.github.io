"""portfolio.py — COT Contrarian Positioning: weekly rebalance schedule + 3-arm portfolio
construction (contrarian / momentum-with-crowd / R10 null of 200 random-sign replicates).

PREREGISTRATION.md: "equal risk (63-day realized-vol scaling), held one week to the next
release." "Arms (identical rebalance dates): Contrarian (primary) · momentum-with-crowd
(ordering check) · R10 null = 200 random-sign portfolios, identical dates and gross
exposure."

Design (documented, no sweeps):
  - Entry/exit fill = D1 bar OPEN at the resolved action_date (release_lag.py) — "act no
    earlier than the following Monday's open".
  - Equal-risk weights: w_i = (1/vol_i) / sum_j(1/vol_j) over that week's 4 active legs,
    vol_i = 63-day trailing realized stdev of daily pip returns (d1_data.realized_vol_pips,
    shift(1)-safe, no lookahead into the entry week itself).
  - Cost model: spread = per-pair median D1 spread (IS-only, computed by is_data.py) x
    spread_mult, round-trip, in pips; carry = carry_splice.carry_pips_spliced(...),
    markup_mult sensitivity hook. net_pips_i = gross_pips_i - spread_pips_i*spread_mult
    + carry_pips_i(markup_mult).
  - The null arm reuses the SAME 4 legs/weights every replicate/week (R10: "identical
    dates and gross exposure") — only the per-leg SIGN is randomized (seeded). Both
    directions' net_pips are computed ONCE per leg per week (cheap: gross/spread/carry are
    NOT symmetric in direction, since long/short carry rates differ), then the 200
    replicates are built by CHEAP lookup+sum, not 200x re-simulation.
"""
import numpy as np
import pandas as pd

import carry_splice as cs
import cot_signal as sig
import d1_data as d1
import release_lag as rl

N_NULL = 200
NULL_SEED = 20260707
VOL_WINDOW = 63
DIRECT_PAIRS = sorted({pair for pair, _ in sig.DIRECT_PAIR.values()})


def _price_lookup(df: pd.DataFrame, date, col: str, tolerance_days: int = 4):
    """First index timestamp >= `date` (bfill — the next available trading day at/after a
    requested action date), within `tolerance_days` of `date` — tolerant of the rare
    pair-specific holiday gap without silently reaching arbitrarily far into the future.
    Returns (found_timestamp, value) or (None, None) if nothing resolves in range."""
    idx = df.index
    pos = idx.searchsorted(date, side="left")
    if pos >= len(idx):
        return None, None
    found = idx[pos]
    if (found - date).days > tolerance_days:
        return None, None
    return found, df.loc[found, col]


def load_price_panel(pairs=DIRECT_PAIRS, data_dir=d1.DATA_DIR):
    """{pair: (df, vol_series)} — df has open/raw_ts, vol_series is the shift(1)-safe
    63-day realized vol (pips), same index as df."""
    panel = {}
    for pair in pairs:
        df = d1.load_pair(pair, data_dir)
        vol = d1.realized_vol_pips(df, window=VOL_WINDOW)
        panel[pair] = (df, vol)
    return panel


def build_rebalance_schedule(cot_df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Weekly rebalance table: index=report_date (all 7 currencies z-valid), columns =
    7 currency z-scores + action_date (entry) + exit_action_date (next rebalance's entry,
    i.e. 'held one week to the next release')."""
    z = sig.compute_zscore_panel(cot_df)
    panel_z = sig.build_rebalance_panel(z)  # report_date-indexed, all-7-valid only
    sched = pd.DataFrame({"report_date": panel_z.index})
    aligned = rl.align_cot_to_action_dates(sched, calendar).set_index("report_date")
    out = panel_z.join(aligned[["action_date"]], how="inner").sort_index()
    out["exit_action_date"] = out["action_date"].shift(-1)
    out = out.dropna(subset=["exit_action_date"]).copy()
    return out


def _leg_net_pips(pair, direction, entry_row, exit_row, spread_pips, spread_mult, markup_mult):
    pip = d1.pip_size(pair)
    gross = (exit_row["open"] - entry_row["open"]) / pip * direction
    carry = cs.carry_pips_spliced(pair, direction, entry_row["raw_ts"], exit_row["raw_ts"],
                                   markup_mult=markup_mult)
    net = gross - spread_pips * spread_mult + carry
    return net, gross, carry


def build_weekly_returns_from_schedule(sched: pd.DataFrame, price_panel: dict,
                                        spread_medians: dict, spread_mult: float = 1.0,
                                        markup_mult: float = 1.0, n_null: int = N_NULL,
                                        null_seed: int = NULL_SEED):
    """Core battery: one row per rebalance week (from build_rebalance_schedule), columns:
      report_date, action_date, exit_action_date, legs/top/bottom (currencies),
      contrarian_net, momentum_net, null_net_0..null_net_{n_null-1}  (all in EQUAL-RISK-
      WEIGHTED pips — a weighted average across the week's 4 legs, weights summing to 1).
    Rows where any active leg's price or 63-day vol can't be resolved are dropped; the
    count is returned as `n_dropped` (second return value) for the caller to report."""
    rng = np.random.default_rng(null_seed)
    rows = []
    n_dropped = 0
    for report_date, row in sched.iterrows():
        z_row = row[sig.CURRENCIES]
        top, bottom = sig.select_legs(z_row)
        legs = top + bottom  # 4 currencies

        entry_date, exit_date = row["action_date"], row["exit_action_date"]

        leg_ok = True
        weight_raw = {}
        net_by_dir = {}  # currency -> {+1: net_pips, -1: net_pips}
        for ccy in legs:
            pair, base_sign = sig.DIRECT_PAIR[ccy]
            df, vol_series = price_panel[pair]
            entry_ts, entry_open = _price_lookup(df, entry_date, "open")
            exit_ts, exit_open = _price_lookup(df, exit_date, "open")
            if entry_ts is None or exit_ts is None:
                leg_ok = False
                break
            vpos = vol_series.index.searchsorted(entry_ts, side="right") - 1
            vol = vol_series.iloc[vpos] if vpos >= 0 else np.nan
            if not np.isfinite(vol) or vol <= 0:
                leg_ok = False
                break
            weight_raw[ccy] = 1.0 / vol

            entry_row = df.loc[entry_ts]
            exit_row = df.loc[exit_ts]
            spread_pips = spread_medians[pair]
            net_plus, _, _ = _leg_net_pips(pair, base_sign, entry_row, exit_row, spread_pips,
                                            spread_mult, markup_mult)
            net_minus, _, _ = _leg_net_pips(pair, -base_sign, entry_row, exit_row, spread_pips,
                                             spread_mult, markup_mult)
            net_by_dir[ccy] = {+1: net_plus, -1: net_minus}
        if not leg_ok:
            n_dropped += 1
            continue

        wsum = sum(weight_raw.values())
        weights = {ccy: w / wsum for ccy, w in weight_raw.items()}

        contrarian_dir = dict(sig.legs_for_arm(top, bottom, "contrarian"))
        momentum_dir = dict(sig.legs_for_arm(top, bottom, "momentum"))

        contrarian_net = sum(weights[c] * net_by_dir[c][contrarian_dir[c]] for c in legs)
        momentum_net = sum(weights[c] * net_by_dir[c][momentum_dir[c]] for c in legs)

        null_nets = np.empty(n_null)
        for r in range(n_null):
            signs = rng.choice([-1, 1], size=len(legs))
            null_nets[r] = sum(weights[legs[i]] * net_by_dir[legs[i]][signs[i]] for i in range(len(legs)))

        out_row = {
            "report_date": report_date, "action_date": entry_date, "exit_action_date": exit_date,
            "legs": ",".join(legs), "top": ",".join(top), "bottom": ",".join(bottom),
            "contrarian_net": contrarian_net, "momentum_net": momentum_net,
        }
        for r in range(n_null):
            out_row[f"null_net_{r}"] = null_nets[r]
        rows.append(out_row)

    out = pd.DataFrame(rows)
    return out, n_dropped
