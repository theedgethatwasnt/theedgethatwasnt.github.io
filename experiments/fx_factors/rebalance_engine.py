#!/usr/bin/env python3
"""
rebalance_engine.py — fx_factors: monthly rebalance engine (locked construction,
PREREGISTRATION.md "Portfolio construction"). ONE code path (R6) for every factor variant
(carry gated/ungated, momentum, value, composite) AND for the R10 random-weight null —
`run_portfolio()` takes a `direction_fn(signal_date) -> {currency: +1/-1/0}` callable; the
null (null_r10.py) just passes a random one.

Mechanics (locked):
  - Monthly rebalance: signal at the LAST D1 bar of each calendar month (using the
    intersection-of-7-expression-pairs master calendar, `build_master_calendar`), position
    EXECUTED at the NEXT D1 bar's OPEN (mid) — R1 (only a bar that has closed can be
    signalled on) + R3 (mid price, spread deducted explicitly and separately).
  - Position held from execution_date to the NEXT rebalance's execution_date (positions are
    marked-to-market / fully closed-and-reopened at each monthly boundary — see cost-model
    note below).
  - Equal-risk sizing: each ACTIVE leg (currency with direction != 0) gets an inverse-63-D1-
    day-realized-vol weight, normalized across all active legs so gross weight sums to 1.0
    per rebalance — identical convention to multiday_contrarian/equal_risk_portfolio.py's
    "Equal-RISK weights: w_i = (1/sigma_i) / sum_j(1/sigma_j)".
  - Cost model: spread cost = half round-trip logged spread at the ENTRY D1 bar + half at the
    EXIT D1 bar (that bar's own bid_c/ask_c — the D1 bar's closing spread per bars.py; a
    proxy for the bar's spread level, not the exact opening-tick spread — documented, R9),
    scaled by `spread_mult`. Carry accrues via carry_model.carry_pips() (broker-truth,
    weekend/triple-swap-aware), called once per leg per holding month on its actual
    entry_ts/exit_ts — a single bulk call that internally sums per-rollover-day, i.e.
    literally "carry accrued daily per position" (R6: this IS the same function
    test_rebalance_engine.py's Gate-2 parity test independently re-derives day-by-day).
  - Documented simplification (R9, conservative/upper-bound on realistic cost): EVERY
    rebalance is treated as a full close-and-reopen of the target book, even for a currency
    that happens to stay in the book two months running (e.g. long EUR again) — no turnover
    netting. This charges a full round-trip spread every month on every active leg regardless
    of persistence, which OVERSTATES real trading cost — the same conservative convention
    multiday_contrarian/equal_risk_portfolio.py documents for StrengthSpread ("conservative
    no-turnover-netting cost model").
  - Pip-pooling: portfolio monthly return = sum_i w_i * net_pips_i, net_pips_i in THAT leg's
    own pair's pip units. Mixing pip units of a JPY-cross with a USD-cross when weighting is a
    known, documented limitation shared with the rest of this program's pip-pooling
    convention (multiday_contrarian/equal_risk_portfolio.py, same docstring language).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multiday_contrarian"))
from carry_model import carry_pips, pip_of  # noqa: E402

from currency_index import EXPRESSION, REQUIRED_PAIRS_FOR_INDEX

VOL_WINDOW = 63


def build_master_calendar(pair_d1, pairs=None):
    """Intersection of D1 timestamps across `pairs` (default: the 7 pairs used to trade every
    currency, currency_index.REQUIRED_PAIRS_FOR_INDEX) — the tradeable calendar."""
    if pairs is None:
        pairs = REQUIRED_PAIRS_FOR_INDEX
    common = None
    for p in pairs:
        idx = set(pair_d1[p]["timestamp"])
        common = idx if common is None else (common & idx)
    return pd.DatetimeIndex(sorted(common))


def month_end_signal_dates(calendar):
    """Last calendar bar of each (UTC year, month) present in `calendar` — the monthly signal
    dates. Grouped on plain UTC (year, month) tuples, not `to_period` (avoids a tz-drop
    warning/footgun). This is safe for real D1 data: bars.py anchors D1 bars at NY 17:00,
    which lands at 21:00-22:00 UTC (DST-dependent) — comfortably mid-day, nowhere near the
    00:00-05:00 UTC band where a UTC-vs-NY-calendar-month classification could ever disagree
    — so UTC-month and NY-trading-month grouping always coincide for actual bars.py output."""
    s = pd.Series(calendar, index=calendar)
    key = s.index.year * 100 + s.index.month  # e.g. 202301 -- single sortable int key
    return s.groupby(key).max().sort_values().tolist()


def build_rebalance_schedule(calendar):
    """[(signal_date, execution_date), ...] — execution_date is the NEXT calendar bar strictly
    after signal_date (R1: only a closed bar can be signalled on; execution is the following
    bar's open). A signal date with no following bar (nothing left to execute) is dropped."""
    signals = month_end_signal_dates(calendar)
    cal_list = list(calendar)
    pos_of = {d: i for i, d in enumerate(cal_list)}
    out = []
    for sd in signals:
        i = pos_of[sd]
        if i + 1 >= len(cal_list):
            continue
        out.append((sd, cal_list[i + 1]))
    return out


def _pair_close_series(pair_d1, pair):
    return pair_d1[pair].set_index("timestamp")["close"].sort_index()


def _pair_bar_at_or_after(pair_d1, pair, ts):
    d = pair_d1[pair].set_index("timestamp").sort_index()
    pos = d.index.searchsorted(pd.Timestamp(ts), side="left")
    if pos >= len(d):
        return None
    return d.iloc[pos]


def realized_vol_63d(pair_d1, pair, asof_date, window=VOL_WINDOW):
    """Rolling stdev of log returns over the trailing `window` D1 bars up to & including
    asof_date (causal — no lookahead). None if fewer than 2 usable bars."""
    s = _pair_close_series(pair_d1, pair)
    s = s[s.index <= pd.Timestamp(asof_date)]
    if len(s) < 2:
        return None
    logret = np.log(s / s.shift(1)).dropna()
    win = logret.iloc[-window:]
    if len(win) < 2:
        return None
    v = float(win.std(ddof=1))
    return v if v > 0 else None


def run_portfolio(pair_d1, schedule, direction_fn, spread_mult=1.0, markup_mult=1.0, gate_fn=None):
    """Core monthly rebalance loop. Returns (monthly_df, legs_df):
      monthly_df: one row per COMPLETED rebalance [signal_date, execution_date,
                  next_execution_date, n_long, n_short, gated_flat, net_pips (pip-pooled)]
      legs_df: one row per active leg per rebalance (full cost breakdown).
    The final scheduled rebalance is dropped (no next execution date to mark the exit to —
    an incomplete/open position, not a completed trade)."""
    monthly_rows, leg_rows = [], []
    for k, (sig_d, exec_d) in enumerate(schedule):
        if k + 1 >= len(schedule):
            continue
        next_exec_d = schedule[k + 1][1]

        directions = direction_fn(sig_d)
        gated_flat = gate_fn is not None and not gate_fn(sig_d)

        active = {c: d for c, d in directions.items() if d != 0}
        vols = {}
        for c in active:
            pair, _sign = EXPRESSION[c]
            v = realized_vol_63d(pair_d1, pair, sig_d)
            if v is not None:
                vols[c] = v
        active = {c: d for c, d in active.items() if c in vols}

        inv_vol = {c: 1.0 / vols[c] for c in active}
        tot = sum(inv_vol.values())
        weights = {c: (inv_vol[c] / tot if tot > 0 else 0.0) for c in active}

        n_long = sum(1 for d in active.values() if d > 0)
        n_short = sum(1 for d in active.values() if d < 0)
        port_net = 0.0

        for c, d in active.items():
            pair, sign = EXPRESSION[c]
            trade_dir = d * sign
            pip = pip_of(pair)

            entry_bar = _pair_bar_at_or_after(pair_d1, pair, exec_d)
            exit_bar = _pair_bar_at_or_after(pair_d1, pair, next_exec_d)
            if entry_bar is None or exit_bar is None:
                continue
            entry_px, exit_px = float(entry_bar["open"]), float(exit_bar["open"])
            entry_ts, exit_ts = entry_bar.name, exit_bar.name

            entry_spread_pips = float(entry_bar["ask_c"] - entry_bar["bid_c"]) / pip
            exit_spread_pips = float(exit_bar["ask_c"] - exit_bar["bid_c"]) / pip
            spread_rt_pips = (entry_spread_pips + exit_spread_pips) / 2.0 * spread_mult

            gross_pips = trade_dir * (exit_px - entry_px) / pip
            carry = carry_pips(pair, trade_dir, entry_ts, exit_ts, markup_mult=markup_mult)
            net_pips = gross_pips - spread_rt_pips + carry

            w = 0.0 if gated_flat else weights[c]
            port_net += w * net_pips

            leg_rows.append({
                "signal_date": sig_d, "execution_date": exec_d, "next_execution_date": next_exec_d,
                "currency": c, "pair": pair, "currency_direction": d, "trade_direction": trade_dir,
                "weight": w, "vol_63d": vols[c],
                "entry_ts": entry_ts, "exit_ts": exit_ts, "entry_px": entry_px, "exit_px": exit_px,
                "gross_pips": gross_pips, "spread_rt_pips": spread_rt_pips, "carry_pips": carry,
                "net_pips": net_pips, "gated_flat": gated_flat,
            })

        monthly_rows.append({
            "signal_date": sig_d, "execution_date": exec_d, "next_execution_date": next_exec_d,
            "n_long": n_long, "n_short": n_short, "gated_flat": gated_flat, "net_pips": port_net,
        })

    return pd.DataFrame(monthly_rows), pd.DataFrame(leg_rows)
