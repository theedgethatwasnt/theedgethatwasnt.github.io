"""signal.py — COT Contrarian Positioning: z-score, ranking, currency->pair expression.

PREREGISTRATION.md "Signal (locked, no sweeps)": weekly, on the first trading day after
each COT release, per currency, z-score of net non-commercial position (scaled by open
interest) over a trailing 156-week window. Contrarian: short the top-2 most crowded-long
currencies, long the bottom-2 most crowded-short currencies, expressed via the most
liquid USD pairs. Single construction: z-window 156w, top/bottom-2 — no sweeps.
"""
import numpy as np
import pandas as pd

Z_WINDOW = 156          # weeks
Z_MIN_PERIODS = 156     # require a FULL window (strict reading of "trailing 156-week
                         # window" — no partial-window z-scores; a currency simply isn't
                         # in the tradable universe until it has 3 full years of history)
N_TOP = 2
N_BOTTOM = 2

CURRENCIES = ["EUR", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD"]

# Most liquid USD pair per currency + the sign that converts "go LONG the currency" into
# "go +1/-1 on that pair". EUR/GBP/AUD/NZD are the BASE currency in their pair (long
# currency = long pair, sign +1). JPY/CHF/CAD are the QUOTE currency (USD is base; long
# the currency = SHORT the pair, sign -1).
DIRECT_PAIR = {
    "EUR": ("EUR_USD", +1),
    "GBP": ("GBP_USD", +1),
    "AUD": ("AUD_USD", +1),
    "NZD": ("NZD_USD", +1),
    "JPY": ("USD_JPY", -1),
    "CHF": ("USD_CHF", -1),
    "CAD": ("USD_CAD", -1),
}


def compute_zscore_panel(cot_df: pd.DataFrame, window: int = Z_WINDOW,
                          min_periods: int = Z_MIN_PERIODS) -> pd.DataFrame:
    """Per currency, rolling z-score of net_noncomm_frac_oi over a trailing `window`-week
    window (min_periods enforces a FULL window before any z is emitted). ddof=1 (sample
    std), matching the rest of this codebase's bootstrap/vol conventions."""
    out = []
    for ccy, g in cot_df.groupby("currency"):
        g = g.sort_values("report_date").reset_index(drop=True).copy()
        roll = g["net_noncomm_frac_oi"].rolling(window=window, min_periods=min_periods)
        mean = roll.mean()
        std = roll.std(ddof=1)
        g["z"] = (g["net_noncomm_frac_oi"] - mean) / std
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_rebalance_panel(z_df: pd.DataFrame, currencies=CURRENCIES) -> pd.DataFrame:
    """Pivot to report_date x currency z-score matrix, keeping ONLY weeks where ALL
    `currencies` have a valid (non-NaN) z — required for a well-defined top-2/bottom-2
    selection across the full 7-currency universe every week."""
    wide = z_df.pivot(index="report_date", columns="currency", values="z")
    missing = [c for c in currencies if c not in wide.columns]
    if missing:
        raise ValueError(f"currencies missing from z_df: {missing}")
    wide = wide[list(currencies)]
    wide = wide.dropna(how="any")
    return wide.sort_index()


def select_legs(z_row: pd.Series, n_top: int = N_TOP, n_bottom: int = N_BOTTOM):
    """Returns (top_currencies, bottom_currencies): top = highest z (most crowded LONG),
    bottom = lowest z (most crowded SHORT). Deterministic tie-break: pandas' stable sort
    preserves the fixed CURRENCIES column order on exact ties (never occurs in practice
    with continuous z-scores, but kept deterministic regardless)."""
    ranked = z_row.sort_values(ascending=False, kind="mergesort")
    top = list(ranked.index[:n_top])
    bottom = list(ranked.index[-n_bottom:])
    assert len(set(top) & set(bottom)) == 0, "top/bottom overlap — n_top+n_bottom > n_currencies?"
    return top, bottom


def legs_for_arm(top, bottom, arm: str):
    """Returns list of (currency, view_direction) for the requested arm.
    view_direction=+1 means 'go long the currency', -1 means 'go short the currency'.
      contrarian:  short crowded-long (top), long crowded-short (bottom)   [primary, H1]
      momentum:    long crowded-long (top), short crowded-short (bottom)  [ordering check]
    (the 'null' arm's random signs are generated separately in portfolio.py, per replicate)
    """
    if arm == "contrarian":
        return [(c, -1) for c in top] + [(c, +1) for c in bottom]
    if arm == "momentum":
        return [(c, +1) for c in top] + [(c, -1) for c in bottom]
    raise ValueError(f"unknown arm: {arm!r}")


def pair_direction(currency: str, view_direction: int) -> tuple:
    """(pair, pair_direction) for 'go view_direction on `currency`' expressed via its
    DIRECT_PAIR leg."""
    pair, sign = DIRECT_PAIR[currency]
    return pair, int(view_direction) * sign
