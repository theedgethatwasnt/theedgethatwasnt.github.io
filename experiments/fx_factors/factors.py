#!/usr/bin/env python3
"""
factors.py — fx_factors: the three pre-registered rank functions (carry primary; momentum,
value secondary) + equal-weight composite (secondary), and the shared top-3/bottom-3
selection rule.

Selection-rule note (documented, not a deviation): PREREGISTRATION.md says "Cross-sectional
rank over the 8 currencies ... long top-3 / short bottom-3 currencies." USD has no dedicated
expression pair (currency_index.py), so it can never itself be traded. Ranking all 8
currencies (USD's score is always exactly 0 by construction of every factor below) and then
DROPPING USD from a top-3/bottom-3 slot it happens to occupy, backfilling from the next
non-USD candidate, is mathematically IDENTICAL to ranking the 7 non-USD currencies directly
and taking their own top-3/bottom-3 — removing one element from an already-sorted list never
changes the relative order of what remains. `rank_select()` below therefore ranks the 7
non-USD currencies directly; USD's score is still computed and reported (informative context:
"how do currencies compare to the numeraire") but is inert for trade selection.

Momentum (12-1, Jegadeesh-Titman skip-month convention): log return of the currency's USD-
value index from (rebalance_date - 12mo) to (rebalance_date - 1mo) — the most recent month is
excluded to avoid short-term reversal contamination, the standard academic construction.

Value: log real exchange rate = log(xusd_market) - log(ppp_asof) — positive means the market
gives you MORE currency units per USD than PPP predicts, i.e. the currency is CHEAPER than its
PPP fair value (undervalued) -> long (matches pre-reg: "cheap = long").

Composite: equal-weight average of the three factors' cross-sectional RANKS (not raw scores —
carry is a small decimal rate, momentum/value are log-ratios; ranks are scale-free and avoid
one factor mechanically dominating the sum). A currency missing from some factors that month
(insufficient history) is averaged over whichever factors ARE available for it that month.
"""
import numpy as np
import pandas as pd

from currency_index import NON_USD_CURRENCIES, carry_rate_panel
from is_data import CURRENCIES
from ppp_fetch import ppp_asof


def _asof(series, target_date):
    """Last available value in `series` (DatetimeIndex, sorted) at or before target_date."""
    ts = pd.Timestamp(target_date)
    if series.index.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize(series.index.tz)
    pos = series.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    v = series.iloc[pos]
    return None if pd.isna(v) else float(v)


def carry_score(rebal_date, markup_mult=1.0):
    """Broker-truth annualized carry rate vs USD (decimal), one row -> pd.Series over
    CURRENCIES. Thin wrapper around currency_index.carry_rate_panel for a single date."""
    row = carry_rate_panel([rebal_date], markup_mult=markup_mult).iloc[0]
    return row


def momentum_score(usd_per_x, rebal_date):
    t_1mo = pd.Timestamp(rebal_date) - pd.DateOffset(months=1)
    t_12mo = pd.Timestamp(rebal_date) - pd.DateOffset(months=12)
    scores = {}
    for ccy in CURRENCIES:
        v1 = _asof(usd_per_x[ccy], t_1mo)
        v12 = _asof(usd_per_x[ccy], t_12mo)
        if v1 is None or v12 is None or v1 <= 0 or v12 <= 0:
            scores[ccy] = np.nan
        else:
            scores[ccy] = float(np.log(v1) - np.log(v12))
    return pd.Series(scores)[CURRENCIES]


def value_score(xusd, rebal_date):
    scores = {}
    for ccy in CURRENCIES:
        market = _asof(xusd[ccy], rebal_date)
        ppp = ppp_asof(ccy, rebal_date)
        if market is None or ppp is None or np.isnan(ppp) or market <= 0 or ppp <= 0:
            scores[ccy] = np.nan
        else:
            scores[ccy] = float(np.log(market) - np.log(ppp))
    return pd.Series(scores)[CURRENCIES]


def composite_score(carry_s, mom_s, val_s):
    ranks = []
    for s in (carry_s, mom_s, val_s):
        ranks.append(s.reindex(NON_USD_CURRENCIES).rank())
    avg_rank = pd.concat(ranks, axis=1).mean(axis=1, skipna=True)
    out = pd.Series(np.nan, index=CURRENCIES)
    out.loc[NON_USD_CURRENCIES] = avg_rank
    out.loc["USD"] = 0.0
    return out


def rank_select(scores):
    """scores: pd.Series over CURRENCIES (NaN allowed for currencies lacking history yet).
    Returns {currency: direction in {-1,0,+1}} over NON_USD_CURRENCIES. If fewer than 6
    non-NaN candidates are available that month, the long/short book shrinks symmetrically
    (documented degrade path — happens only in the first ~12 months of the momentum factor's
    IS window, before a full 12-month lookback exists)."""
    non_usd = scores.reindex(NON_USD_CURRENCIES).dropna()
    ranked_desc = non_usd.sort_values(ascending=False).index.tolist()
    directions = {c: 0 for c in NON_USD_CURRENCIES}
    n = len(ranked_desc)
    k = min(3, n // 2)
    for c in ranked_desc[:k]:
        directions[c] = +1
    if k > 0:
        for c in ranked_desc[-k:]:
            directions[c] = -1
    return directions
