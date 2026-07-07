"""axis3_vote.py — Composite 1: Axis-3 COT-crowding vote.

PREREGISTRATION.md "Axis-3 vote (inherited verbatim from cot_positioning)": currency-level
contrarian crowding, 156-week z-score of net non-commercial position scaled by open
interest, as of the most recent RELEASED report. Reuses cot_positioning's cot_signal.py
(z-score) and release_lag.py (publication-lag alignment) VERBATIM via import (see
_paths.py) — literally the same code object, not a copy, which is what makes gate 2's
"same code" parity claim true by construction.

Composite gate direction mapping — documented judgment call (R9-style disclosure)
-----------------------------------------------------------------------------------
PREREGISTRATION.md's prose: "fading a swing HIGH of pair base (going short base) requires
base-currency crowding z >= +1.0 (crowd long the thing being faded); fading a LOW (long
base) requires z <= -1.0." This sentence is written using the EUR_USD-style exemplar, where
the pair's literal FX base (EUR) IS the COT-tracked currency and
cot_signal.DIRECT_PAIR[ccy] has sign=+1. For USD_JPY/USD_CHF/USD_CAD the literal FX base is
USD, which has NO COT z-score at all (USD is the implicit residual leg — see
cot_positioning/PREREGISTRATION.md "Data": "CME currency futures EUR, JPY, GBP, CHF, AUD,
CAD, NZD (USD implicit as the base leg)"). The literal "base-currency z-score" reading is
therefore not even implementable for 3 of the 7 pairs, which proves the prose must
generalize to "the pair's COT-TRACKED (non-USD) currency", not the literal FX base — the
only internally-consistent reading, and the only one that keeps gate 2's verbatim-code
guarantee meaningful (cot_signal.DIRECT_PAIR is the single source of truth for which
currency's z-score governs which pair, and with what sign).

This module implements that generalization. For an Axis-1 trade `direction_pair` on `pair`
(+1 = long pair / fade a LOW, -1 = short pair / fade a HIGH), find (ccy, sign) via
cot_signal.DIRECT_PAIR (pair -> (ccy, sign) inverted). The implied view on the COT-tracked
currency is `view_ccy = direction_pair * sign` — the exact algebraic inverse of
cot_signal.pair_direction(ccy, view) = (pair, view*sign). The gate then applies the same
contrarian logic cot_signal.legs_for_arm("contrarian") encodes at the ranking level (short
a crowded-long currency, long a crowded-short currency), thresholded at |z|>=1.0 instead of
top/bottom-2 ranking:
    view_ccy == -1 (short the currency) requires z(ccy) >= +Z_THRESH  (crowded long -> fade)
    view_ccy == +1 (long the currency)  requires z(ccy) <= -Z_THRESH (crowded short -> fade)
This reproduces the prose EXACTLY for EUR/GBP/AUD/NZD-style pairs (sign=+1: e.g. fading a
HIGH of EUR_USD -> direction_pair=-1 -> view_ccy = -1*1 = -1 -> requires z(EUR)>=+1.0,
literally "base-currency z>=+1.0") and is the unique internally-consistent extension for
JPY/CHF/CAD-style pairs (sign=-1: e.g. fading a HIGH of USD_JPY -> direction_pair=-1 ->
view_ccy = -1*(-1) = +1 (going long JPY, i.e. short the literal FX base USD) -> requires
z(JPY)<=-1.0, i.e. JPY must be crowded SHORT to justify going long it).
"""
import numpy as np
import pandas as pd

import _paths  # noqa: F401  (sys.path setup for verbatim cot_positioning imports)
import cot_signal as sig
import release_lag as rl

Z_THRESH = 1.0

# pair -> (ccy, sign), the exact inverse of cot_signal.DIRECT_PAIR: {ccy: (pair, sign)}
PAIR_TO_CCY_SIGN = {pair: (ccy, sign) for ccy, (pair, sign) in sig.DIRECT_PAIR.items()}


def build_currency_z_series(cot_df: pd.DataFrame, trading_calendar) -> dict:
    """{currency: DataFrame(action_date, z)}, sorted by action_date, RELEASED-only (every
    row's action_date already respects release_lag's no-lookahead alignment). Built from
    cot_signal.compute_zscore_panel (verbatim) + release_lag.align_cot_to_action_dates
    (verbatim) — no code of this experiment's own touches the z-score or release-lag math."""
    z = sig.compute_zscore_panel(cot_df)
    z = z.dropna(subset=["z"])
    aligned = rl.align_cot_to_action_dates(z[["currency", "report_date", "z"]], trading_calendar)
    out = {}
    for ccy, g in aligned.groupby("currency"):
        out[ccy] = g[["action_date", "z"]].sort_values("action_date").reset_index(drop=True)
    return out


def make_zlookup(ccy_series: dict):
    """Returns lookup(ccy, asof_ts) -> latest RELEASED z with action_date <= asof_ts, else
    nan. Backward-asof (np.searchsorted 'right' - 1): only ever reads a report whose
    action_date has already passed relative to asof_ts — the no-lookahead property
    release_lag.py's own tripwire test (test_release_lag.py, reused verbatim by
    cot_positioning) already proves for action_date construction itself; this lookup adds
    no further lookahead on top of it."""
    arrs = {}
    for ccy, g in ccy_series.items():
        dates = pd.DatetimeIndex(g["action_date"]).values  # tz-aware -> UTC-naive datetime64[ns]
        arrs[ccy] = (dates, g["z"].values.astype(float))

    def lookup(ccy, asof_ts):
        if ccy not in arrs:
            return float("nan")
        dates, zs = arrs[ccy]
        if len(dates) == 0:
            return float("nan")
        asof = pd.Timestamp(asof_ts)
        asof_np = np.datetime64(asof.tz_localize(None) if asof.tzinfo is None else asof.tz_convert("UTC").tz_localize(None), "ns")
        pos = np.searchsorted(dates, asof_np, side="right") - 1
        if pos < 0:
            return float("nan")
        return float(zs[pos])

    return lookup


def view_direction_for_ccy(direction_pair: int, pair: str):
    ccy, sign = PAIR_TO_CCY_SIGN[pair]
    return ccy, int(direction_pair) * sign


def composite_gate(direction_pair: int, pair: str, asof_ts, zlookup) -> bool:
    """True iff the Axis-3 vote AGREES with the Axis-1 trade direction, per the mapping
    documented in this module's docstring."""
    ccy, view_ccy = view_direction_for_ccy(direction_pair, pair)
    z = zlookup(ccy, asof_ts)
    if not np.isfinite(z):
        return False
    if view_ccy == -1:
        return z >= Z_THRESH
    return z <= -Z_THRESH


def permute_currency_z_series(ccy_series: dict, seed: int) -> dict:
    """Block-preserving-by-currency permutation (PREREGISTRATION.md arm 5): shuffle each
    currency's OWN z values among its OWN action_dates — NEVER swap a value across
    currencies. This destroys the temporal alignment between an Axis-1 signal's date and
    the Axis-3 vote (the thing being tested: does that alignment carry information?) while
    exactly preserving each currency's own marginal z distribution (identical multiset of
    values, identical count, identical unconditional crowding-extremity mix)."""
    rng = np.random.default_rng(seed)
    out = {}
    for ccy, g in ccy_series.items():
        gg = g.copy()
        gg["z"] = rng.permutation(gg["z"].values)
        out[ccy] = gg
    return out
