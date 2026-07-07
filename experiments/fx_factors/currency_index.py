#!/usr/bin/env python3
"""
currency_index.py — fx_factors: derive per-currency (vs USD) synthetic price/rate panels
from the 12 available pairs (triangulation through JPY for CAD/CHF, which have no direct USD
pair in this 12-pair universe), plus the SPX500 D1 SMA(200) risk-off gate signal.

Numeraire: USD. For every currency X we define two aligned daily series:
  usd_per_x[X]  = USD amount per 1 unit of X ("currency strength index"; higher = X stronger
                  vs USD). USD itself is the trivial constant 1.0.
  xusd[X]       = 1 / usd_per_x[X] = X units per 1 USD — the SAME convention World Bank's
                  PA.NUS.PPP series uses (LCU per international $), so xusd[X] is directly
                  comparable to ppp_fetch.ppp_asof(X, d) for the value factor.

Triangulation (no CAD_USD/USD_CAD or CHF_USD/USD_CHF pair exists in our 12-pair universe —
both currencies are ONLY expressed via their JPY cross):
  EUR/GBP/AUD/NZD: direct *_USD pairs, usd_per_x = pair close (already USD per 1 unit).
  JPY:  usd_per_x = 1 / USD_JPY_close     (USD_JPY = JPY per 1 USD).
  CAD:  usd_per_x = CAD_JPY_close / USD_JPY_close   (JPY/CAD divided by JPY/USD = USD/CAD).
  CHF:  usd_per_x = CHF_JPY_close / USD_JPY_close   (same derivation).
  Documented limitation (R9): CAD's and CHF's usd_per_x therefore embed JPY as the vehicle
  currency. Since JPY is itself independently ranked/traded in this same universe, the CAD/CHF
  legs are not free of JPY co-movement the way a genuine CAD_USD quote would be — this is the
  "expressed through the most liquid pairs" tradeoff the pre-registration explicitly allows,
  not an oversight. EUR_GBP (the 12th pair) is redundant for currency-index purposes (EUR and
  GBP are both already covered via their own *_USD legs) and is not used here.

Carry-rate panel (per-currency broker-truth annualized rate vs USD, decimal; USD=0 by
construction): built by chaining carry_model._rate_dir() (FRED differential + measured-pinch
model — the "broker-truth" primitive) through the SAME triangulation as the price panel
above, so the carry RANK signal and the P&L carry ACCRUAL (carry_model.carry_pips, called
directly by rebalance_engine.py on the real expression pairs) share one underlying formula
(R6 — one code path for the rate model, evaluated at two different directions/dates).

Expression pair + sign (the pair/direction that realizes "go long currency X vs USD"; USD
itself has no expression pair — see rebalance_engine.py's non-USD-only selection rule):
  EUR->(EUR_USD,+1) GBP->(GBP_USD,+1) AUD->(AUD_USD,+1) NZD->(NZD_USD,+1)
  JPY->(USD_JPY,-1)  [long JPY = short USD_JPY]
  CAD->(CAD_JPY,+1)  CHF->(CHF_JPY,+1)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multiday_contrarian"))
from carry_model import _rate_dir  # noqa: E402

from is_data import CURRENCIES

EXPRESSION = {
    "EUR": ("EUR_USD", +1),
    "GBP": ("GBP_USD", +1),
    "AUD": ("AUD_USD", +1),
    "NZD": ("NZD_USD", +1),
    "JPY": ("USD_JPY", -1),
    "CAD": ("CAD_JPY", +1),
    "CHF": ("CHF_JPY", +1),
}
NON_USD_CURRENCIES = [c for c in CURRENCIES if c != "USD"]

REQUIRED_PAIRS_FOR_INDEX = [
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "CAD_JPY", "CHF_JPY",
]


def build_panels(pair_d1):
    """pair_d1: dict {pair: D1 DataFrame with columns timestamp,open,high,low,close,bid_c,
    ask_c,volume}, must cover at least REQUIRED_PAIRS_FOR_INDEX. Returns (usd_per_x, xusd) —
    two DataFrames indexed by the INTERSECTION of those 7 pairs' D1 timestamps (conservative:
    every currency column is defined on every row, no partial-coverage dates invented),
    columns = CURRENCIES. Built from each pair's D1 CLOSE (mid) — R3."""
    closes = {}
    common_idx = None
    for pair in REQUIRED_PAIRS_FOR_INDEX:
        s = pair_d1[pair].set_index("timestamp")["close"].sort_index()
        closes[pair] = s
        idx = set(s.index)
        common_idx = idx if common_idx is None else (common_idx & idx)
    common_idx = pd.DatetimeIndex(sorted(common_idx))
    for pair in closes:
        closes[pair] = closes[pair].reindex(common_idx)

    usd_per_x = pd.DataFrame(index=common_idx)
    usd_per_x["USD"] = 1.0
    usd_per_x["EUR"] = closes["EUR_USD"]
    usd_per_x["GBP"] = closes["GBP_USD"]
    usd_per_x["AUD"] = closes["AUD_USD"]
    usd_per_x["NZD"] = closes["NZD_USD"]
    usd_per_x["JPY"] = 1.0 / closes["USD_JPY"]
    usd_per_x["CAD"] = closes["CAD_JPY"] / closes["USD_JPY"]
    usd_per_x["CHF"] = closes["CHF_JPY"] / closes["USD_JPY"]
    usd_per_x = usd_per_x[CURRENCIES]

    xusd = 1.0 / usd_per_x
    return usd_per_x, xusd


def carry_rate_panel(dates, markup_mult=1.0):
    """Per-currency broker-truth annualized carry rate vs USD (decimal), one row per date in
    `dates`. USD column is exactly 0.0 by construction (numeraire)."""
    rows = []
    idx = []
    for d in dates:
        ts = pd.Timestamp(d)
        dd = ts.date()
        jpy_vs_usd = _rate_dir("USD_JPY", -1, dd, markup_mult)
        r = {
            "USD": 0.0,
            "EUR": _rate_dir("EUR_USD", +1, dd, markup_mult),
            "GBP": _rate_dir("GBP_USD", +1, dd, markup_mult),
            "AUD": _rate_dir("AUD_USD", +1, dd, markup_mult),
            "NZD": _rate_dir("NZD_USD", +1, dd, markup_mult),
            "JPY": jpy_vs_usd,
            "CAD": _rate_dir("CAD_JPY", +1, dd, markup_mult) + jpy_vs_usd,
            "CHF": _rate_dir("CHF_JPY", +1, dd, markup_mult) + jpy_vs_usd,
        }
        rows.append(r)
        idx.append(ts)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))[CURRENCIES]


def spx_gate_signal(spx_df, dates, sma_window=200):
    """For each date in `dates` (rebalance signal dates), True (risk_on) iff the most recent
    SPX500 D1 close at-or-before that date is >= its trailing SMA(sma_window) computed AT that
    same bar (causal: SMA at bar i uses only bars <= i, no lookahead). SPX history starts
    2003 (data/cross_asset), so every 2020+ rebalance date has a fully-warmed-up SMA(200)."""
    s = spx_df.set_index("time")["close"].sort_index()
    sma = s.rolling(sma_window, min_periods=sma_window).mean()
    out = {}
    for d in dates:
        ts = pd.Timestamp(d)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        pos = s.index.searchsorted(ts, side="right") - 1
        if pos < 0 or pd.isna(sma.iloc[pos]):
            out[d] = True  # no warmup case; not expected to fire given 2003+ SPX history
            continue
        out[d] = bool(s.iloc[pos] >= sma.iloc[pos])
    return pd.Series(out)
