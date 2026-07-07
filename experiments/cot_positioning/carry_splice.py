"""carry_splice.py — COT Contrarian Positioning: pre-2020 / 2020+ carry model splice.

Task brief: "Reuse carry_model from multiday_contrarian for 2020+; pre-2020 use
FRED-differential flat approximation (its fred cache covers history) — document the
splice." This module is that splice, built on TOP of the (copied, unmodified) carry_model
module in this directory.

WHY A SPLICE IS NEEDED
-----------------------
carry_model.carry_pips(..., markup_mult) computes:
    rate_dir(t) = fred_diff_dir(t) + markup_mult * pinch
where `pinch` is the OANDA retail markup MEASURED ONCE, from the 2026-07-06/07 snapshot,
and then held CONSTANT and extrapolated across the whole requested date range (module
docstring's approximation #1). multiday_contrarian's own pre-registration only ever
applies this back to 2020-11-11 (its IS start) — the constant-pinch assumption is already
a documented stretch over that 5.5-year span, and this experiment's COT joint window
starts far earlier (~2008, once the 156-week z-score lookback is satisfied). Extrapolating
a single 2026 retail markup back 15-20+ years is a materially weaker assumption than
extrapolating it 5.5 years, so the pre-registration deliberately does NOT do that:

  - date >= SPLICE_DATE (2020-11-11, matching multiday_contrarian's own IS start — the
    latest date the constant-pinch assumption has already been used/accepted in this
    codebase): full carry_model, i.e. carry_pips(..., markup_mult=markup_mult) — FRED
    differential + markup_mult * measured pinch.
  - date <  SPLICE_DATE: FLAT FRED-DIFFERENTIAL approximation, i.e.
    carry_pips(..., markup_mult=0.0) — the pinch term drops out entirely (multiplied by
    zero), leaving exactly the interbank policy-rate differential with no retail markup.
    This is deliberately the SAME carry_pips() code path (not a separate formula) so the
    two regimes share 100% of the rollover-day-counting / financingDaysOfWeek /
    price-per-pip machinery — only the markup_mult differs. `fred_rate()` already covers
    every currency back to at least 1999 (EUR) and considerably further for the rest — see
    module-level FRED coverage check in test_carry_splice.py.

Direction of bias (documented, per R9): pre-2020 carry is likely UNDERSTATED in magnitude
(a real retail account would have paid/received some non-zero markup even in 2008-2019),
but the SIGN is preserved (still the correct differential direction) and this is the
conservative choice for a "does the edge survive costs" gate — understating a COST make
the strategy look BETTER than reality on the pre-2020 leg, not worse, so any IS-gate
PASS achieved with this splice should be read with that caution; an IS-gate FAIL is
unaffected (real costs would only be higher).

A rebalance week's [entry_ts, exit_ts) can itself straddle the splice boundary (only
possible for the one week nearest 2020-11-11); that single week's carry is split into two
sub-intervals at the boundary and each priced under its own regime, then summed — exactly
equivalent to calling carry_pips() on each sub-interval, because carry_pips() is additive
over disjoint time sub-intervals of the same [entry, exit) span (see run_is_battery-style
derivation note in multiday_contrarian/run_is_battery.py for the same additivity property
used there for the spread_mult/markup_mult sensitivity variants).
"""
import pandas as pd

import carry_model as cm

SPLICE_DATE = pd.Timestamp("2020-11-11", tz="UTC")


def _as_utc(ts):
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def carry_pips_spliced(pair, direction, entry_ts, exit_ts, notional=1.0, markup_mult=1.0,
                        splice_date=SPLICE_DATE):
    entry = _as_utc(entry_ts)
    exit_ = _as_utc(exit_ts)
    splice = _as_utc(splice_date)

    if exit_ <= splice:
        return cm.carry_pips(pair, direction, entry, exit_, notional=notional, markup_mult=0.0)
    if entry >= splice:
        return cm.carry_pips(pair, direction, entry, exit_, notional=notional, markup_mult=markup_mult)

    pre = cm.carry_pips(pair, direction, entry, splice, notional=notional, markup_mult=0.0)
    post = cm.carry_pips(pair, direction, splice, exit_, notional=notional, markup_mult=markup_mult)
    return pre + post
