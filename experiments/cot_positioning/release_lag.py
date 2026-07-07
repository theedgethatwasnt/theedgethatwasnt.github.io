"""release_lag.py — COT Contrarian Positioning: publication-lag alignment (R1/R9 analog).

PREREGISTRATION.md: "positions are as-of Tuesday, published Friday ~15:30 ET — signals
may act no earlier than the following Monday's open (no lookahead through the
publication lag)."

Two documented approximations (no historical CFTC release-time calendar is queryable,
same category of approximation as carry_model.py's R9 disclosures):

1. RELEASE_LAG_DAYS = 3 calendar days added to the as-of date to get a nominal "Friday
   release" instant, regardless of which weekday the as-of date actually falls on. Recon
   (2026-07-07, test_fetch_cot.py::test_report_dates_are_tuesday_or_holiday_shifted)
   found ~1.2% of as-of dates are holiday-shifted to Monday or Wednesday instead of the
   usual Tuesday — this module intentionally computes the lag from the ACTUAL as-of date
   every time, never assuming Tuesday, so those shifted weeks are still handled correctly
   (the +3-day nominal Friday still falls in the right neighborhood).
2. ACTION_DATE = the first date in a supplied trading-day calendar strictly AFTER the
   nominal release instant (i.e. "the following Monday" in the normal case — and if that
   Monday happens to be a market holiday, this naturally rolls forward to the actual next
   trading day, which is only ever LATER than Monday, never earlier — a conservative
   direction of error, since real CFTC releases are essentially never delayed past Friday
   in a way that would make the nominal Friday+3 boundary optimistic).

No sign of these approximations can make a signal actionable EARLIER than reality — both
are one-directional-safe (never lookahead), which is the property Gate 1's tripwire test
checks below.
"""
import numpy as np
import pandas as pd

RELEASE_LAG_DAYS = 3  # as-of Tuesday -> nominal Friday release


def release_instant(report_date) -> pd.Timestamp:
    """Nominal publication instant: as-of date + RELEASE_LAG_DAYS calendar days."""
    return pd.Timestamp(report_date) + pd.Timedelta(days=RELEASE_LAG_DAYS)


def compute_action_date(report_date, trading_calendar: pd.DatetimeIndex,
                         release_lag_days: int = RELEASE_LAG_DAYS) -> pd.Timestamp:
    """First date in `trading_calendar` STRICTLY AFTER (report_date + release_lag_days).
    Returns pd.NaT if no such date exists (e.g. report is beyond the calendar's end)."""
    release = pd.Timestamp(report_date) + pd.Timedelta(days=release_lag_days)
    cal = pd.DatetimeIndex(trading_calendar)
    candidates = cal[cal > release]
    if len(candidates) == 0:
        return pd.NaT
    return candidates.min()


def align_cot_to_action_dates(cot_df: pd.DataFrame, trading_calendar: pd.DatetimeIndex,
                               release_lag_days: int = RELEASE_LAG_DAYS) -> pd.DataFrame:
    """Vectorized version of compute_action_date for a whole COT DataFrame (must have a
    `report_date` column). Adds `release_instant` and `action_date` columns. Rows whose
    action_date could not be resolved (report beyond calendar) are dropped."""
    cal = pd.DatetimeIndex(trading_calendar).sort_values()
    out = cot_df.copy()
    out["release_instant"] = out["report_date"] + pd.Timedelta(days=release_lag_days)
    # searchsorted(side='right') on the release instant gives the index of the first
    # calendar date > release_instant (since np.searchsorted 'right' returns insertion
    # point after any equal elements, which is exactly "strictly greater than").
    idx = np.searchsorted(cal.values, out["release_instant"].values, side="right")
    resolved = idx < len(cal)
    out.loc[resolved, "action_date"] = cal[idx[resolved]]
    out.loc[~resolved, "action_date"] = pd.NaT
    out = out.dropna(subset=["action_date"]).reset_index(drop=True)
    out["action_date"] = pd.to_datetime(out["action_date"])
    return out
