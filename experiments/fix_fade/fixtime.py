#!/usr/bin/env python3
"""
fixtime.py — London-Fix Fade (PREREGISTRATION.md, LOCKED 2026-07-07): DST-aware fix-time
utilities, shared by every script in this directory (R6 — one code path for time math).

WM/R 4pm fix = 16:00 **Europe/London local wall-clock time**, every calendar day, all year
round (the fix itself does not move with DST — it is always "4pm London"). What DOES move is
its UTC representation: 16:00 GMT (winter, UTC+0) -> 16:00 UTC; 16:00 BST (summer, UTC+1) ->
15:00 UTC. `london_fix_close_utc()` below is the single source of truth for that conversion.

Bar timestamp convention (confirmed against research/experiments/zone_recovery/fetch_m5_ba.py
and research/experiments/cma_5in/precompute_strength_spread.py, and matching the H4 pattern
in multiday_contrarian/harness.py): the `timestamp` column of data/m5_ba/*.parquet is the
OANDA candle's OPEN (start) time — a row with `timestamp = T` spans `[T, T+5min)` and CLOSES
at `T+5min`. So the M5 bar that CLOSES at a target instant `X` is the row with
`timestamp == X - 5min`, NOT the row with `timestamp == X`.

DST-safety note (unlike multiday_contrarian/bars.py's NY-17:00 H4 anchor, which resamples
every M5 bar and therefore must handle the ~2/year US fall-back ambiguous-local-time fold):
16:00 is never inside the UK's DST fold. The UK's spring-forward/fall-back transitions both
happen at 01:00/02:00 **local** clock time (last Sunday of March / October), nowhere near
16:00 — so `pd.Timestamp(..., tz=LONDON)` at 16:00 local is *always* well-defined (verified
empirically across 2021's transition dates in test_fixtime.py; no ambiguous/nonexistent
handling is needed here, unlike the H4 module).
"""
import calendar as _calendar
from zoneinfo import ZoneInfo

import pandas as pd

LONDON = ZoneInfo("Europe/London")


def london_fix_close_utc(date):
    """`date`: anything pd.Timestamp can parse (str 'YYYY-MM-DD', date, Timestamp — time-of-day
    component, if any, is ignored). Returns a tz-aware UTC pd.Timestamp for 16:00 Europe/London
    local wall-clock time on that calendar date — the instant the WM/R 4pm fix CLOSES."""
    d = pd.Timestamp(date)
    local = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=16, minute=0, second=0,
                          tz=LONDON)
    return local.tz_convert("UTC")


def last_business_day_of_month(date):
    """Last Mon-Fri (weekday() < 5) calendar date of `date`'s month — a simple, deterministic,
    non-fitted 'month-end' rule (does NOT account for UK/US bank holidays; documented
    simplification, R9). `date`: anything pd.Timestamp can parse; returns a python `date`."""
    d = pd.Timestamp(date)
    last_day_num = _calendar.monthrange(d.year, d.month)[1]
    cand = pd.Timestamp(year=d.year, month=d.month, day=last_day_num)
    while cand.weekday() >= 5:  # 5=Sat, 6=Sun
        cand -= pd.Timedelta(days=1)
    return cand.date()


def is_last_business_day_of_month(date):
    """True iff `date`'s calendar date (London-local, per the fix event's own reference frame)
    equals `last_business_day_of_month(date)` — the pre-declared 'last trading day' split of
    the PREREGISTRATION.md month-end amplification report (not a search, not a gate)."""
    d = pd.Timestamp(date)
    return d.date() == last_business_day_of_month(d)
