"""Tests for fixtime.py — DST-correct 16:00 Europe/London fix-time identification, incl.
across the UK's spring-forward / fall-back transitions (2021-2024)."""
import pandas as pd
import pytest

from fixtime import is_last_business_day_of_month, last_business_day_of_month, london_fix_close_utc

# UK DST transition dates (last Sunday of March / October), verified against known calendar:
#   2021: spring 03-28, fall 10-31   2022: spring 03-27, fall 10-30
#   2023: spring 03-26, fall 10-29   2024: spring 03-31, fall 10-27
TRANSITIONS = [
    ("2021-03-28", "2021-10-31"),
    ("2022-03-27", "2022-10-30"),
    ("2023-03-26", "2023-10-29"),
    ("2024-03-31", "2024-10-27"),
]


@pytest.mark.parametrize("spring,fall", TRANSITIONS)
def test_spring_forward_flips_utc_hour_16_to_15(spring, fall):
    """The day BEFORE spring-forward: 16:00 London = 16:00 UTC (GMT, UTC+0).
    The day OF/AFTER spring-forward: 16:00 London = 15:00 UTC (BST, UTC+1)."""
    day_before = pd.Timestamp(spring) - pd.Timedelta(days=1)
    before = london_fix_close_utc(day_before)
    on_day = london_fix_close_utc(spring)
    after = london_fix_close_utc(pd.Timestamp(spring) + pd.Timedelta(days=1))

    assert before.hour == 16 and before.tzinfo is not None
    assert on_day.hour == 15
    assert after.hour == 15


@pytest.mark.parametrize("spring,fall", TRANSITIONS)
def test_fall_back_flips_utc_hour_15_to_16(spring, fall):
    """The day BEFORE fall-back: 16:00 London = 15:00 UTC (still BST).
    The day OF/AFTER fall-back: 16:00 London = 16:00 UTC (back to GMT)."""
    day_before = pd.Timestamp(fall) - pd.Timedelta(days=1)
    before = london_fix_close_utc(day_before)
    on_day = london_fix_close_utc(fall)
    after = london_fix_close_utc(pd.Timestamp(fall) + pd.Timedelta(days=1))

    assert before.hour == 15
    assert on_day.hour == 16
    assert after.hour == 16


def test_deep_winter_and_deep_summer_sanity():
    assert london_fix_close_utc("2022-01-15").hour == 16   # deep GMT
    assert london_fix_close_utc("2022-07-15").hour == 15   # deep BST


def test_returns_tz_aware_utc():
    ts = london_fix_close_utc("2023-05-01")
    assert str(ts.tzinfo) == "UTC"
    assert ts.minute == 0 and ts.second == 0


def test_16_00_local_never_ambiguous_or_nonexistent_across_many_years():
    """16:00 London local time is never inside the DST fold (UK transitions happen at
    01:00/02:00 local, nowhere near 16:00) — every single calendar day in a multi-year span
    must resolve to a valid, non-NaT UTC instant."""
    dates = pd.date_range("2020-01-01", "2026-12-31", freq="D")
    for d in dates[::17]:  # every 17th day - representative sample, keeps the test fast
        ts = london_fix_close_utc(d)
        assert pd.notna(ts)
        assert ts.hour in (15, 16)
        assert ts.minute == 0


# ── month-end (last trading day) split helper ────────────────────────────────
def test_last_business_day_of_month_weekday_end():
    # 2024-05-31 is a Friday
    assert last_business_day_of_month("2024-05-15").isoformat() == "2024-05-31"
    assert is_last_business_day_of_month("2024-05-31") is True
    assert is_last_business_day_of_month("2024-05-30") is False


def test_last_business_day_of_month_weekend_end_rolls_back_to_friday():
    # 2025-08-31 is a Sunday -> last business day is Friday 2025-08-29
    assert last_business_day_of_month("2025-08-10").isoformat() == "2025-08-29"
    assert is_last_business_day_of_month("2025-08-29") is True
    assert is_last_business_day_of_month("2025-08-31") is False
    assert is_last_business_day_of_month("2025-08-30") is False


def test_last_business_day_of_month_saturday_end_rolls_back_to_friday():
    # 2026-08-31 is a Monday actually — pick a real Saturday month-end: 2024-08-31 is a Saturday
    assert last_business_day_of_month("2024-08-01").isoformat() == "2024-08-30"
    assert is_last_business_day_of_month("2024-08-30") is True
    assert is_last_business_day_of_month("2024-08-31") is False
