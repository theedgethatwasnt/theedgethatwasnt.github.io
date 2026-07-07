"""test_release_lag.py — Gate 1's "publication-lag alignment test (synthetic future-leak
tripwire)". Proves signals can never act before the Friday+1-trading-day boundary, using a
synthetic COT series + synthetic trading calendar (no real data dependency)."""
import numpy as np
import pandas as pd
import pytest

import release_lag as rl


def _synthetic_calendar(start="2020-01-01", end="2020-12-31", weekmask="Mon Tue Wed Thu Fri"):
    return pd.bdate_range(start, end)


def test_normal_tuesday_report_actions_next_monday():
    """As-of Tuesday 2020-01-07 -> nominal release Friday 2020-01-10 -> first trading day
    strictly after that Friday is Monday 2020-01-13 (no holiday in the way)."""
    cal = _synthetic_calendar()
    action = rl.compute_action_date(pd.Timestamp("2020-01-07"), cal)
    assert action == pd.Timestamp("2020-01-13")


def test_action_date_is_always_after_release_instant():
    """Core tripwire: for a whole synthetic COT panel (weekly Tuesdays across 5 years,
    occasionally holiday-shifted to Monday/Wednesday exactly like the real CFTC data),
    every resolved action_date must be STRICTLY LATER than report_date + 3 days. This is
    the literal 'signals may act no earlier than the Friday+1-trading-day boundary' rule."""
    cal = pd.bdate_range("2015-01-01", "2025-12-31")
    rng = np.random.default_rng(42)
    report_dates = pd.date_range("2015-01-06", "2025-12-30", freq="7D")  # weekly Tuesdays
    # Inject holiday-shift noise identical in spirit to the real data (±1 day on ~5% of weeks)
    shift_days = rng.choice([-1, 0, 0, 0, 0, 1], size=len(report_dates))
    report_dates = report_dates + pd.to_timedelta(shift_days, unit="D")

    cot_df = pd.DataFrame({"currency": "EUR", "report_date": report_dates})
    aligned = rl.align_cot_to_action_dates(cot_df, cal)

    assert len(aligned) > 0
    violation = aligned["action_date"] <= aligned["release_instant"]
    assert violation.sum() == 0, (
        f"{violation.sum()} rows have action_date <= release_instant — LOOKAHEAD TRIPWIRE TRIGGERED"
    )
    # And every action_date must be a real trading-calendar date (not e.g. a weekend).
    assert aligned["action_date"].isin(cal).all()


def test_holiday_shifted_release_never_rolls_backward():
    """If the nominal 'following Monday' is itself a market holiday, action_date must roll
    FORWARD (later), never fall back to Friday or earlier. Simulate: calendar with a Monday
    holiday removed."""
    cal = _synthetic_calendar().drop(pd.Timestamp("2020-01-13"))  # remove the Monday
    action = rl.compute_action_date(pd.Timestamp("2020-01-07"), cal)  # same Tuesday report
    assert action > pd.Timestamp("2020-01-13")
    assert action == pd.Timestamp("2020-01-14")  # rolls to Tuesday, not back to Friday


def test_mutation_zero_lag_would_leak_and_this_harness_catches_it():
    """Negative control: proves the tripwire test actually HAS POWER. If someone
    accidentally set release_lag_days=0 (act on the raw as-of date itself, a real
    lookahead bug), the resulting action_date would land at or before the true Friday
    release instant computed with the correct lag — this test asserts that broken
    configuration IS detectably earlier than the correct one, i.e. our gate would fire."""
    cal = _synthetic_calendar()
    report_date = pd.Timestamp("2020-01-07")  # Tuesday
    correct_action = rl.compute_action_date(report_date, cal, release_lag_days=rl.RELEASE_LAG_DAYS)
    buggy_action = rl.compute_action_date(report_date, cal, release_lag_days=0)
    assert buggy_action < correct_action, (
        "mutation test is broken: the injected zero-lag bug should produce a strictly "
        "earlier (leakier) action date than the correct pipeline"
    )
    # And the buggy action date would fall on/before the TRUE release instant (Friday),
    # i.e. it would actually be trading on data published later that same week — exactly
    # the leak this gate exists to catch.
    true_release = rl.release_instant(report_date)
    assert buggy_action <= true_release


def test_report_beyond_calendar_returns_nat():
    cal = _synthetic_calendar(end="2020-06-30")
    action = rl.compute_action_date(pd.Timestamp("2020-12-29"), cal)
    assert pd.isna(action)


def test_align_drops_unresolvable_rows():
    cal = _synthetic_calendar(end="2020-06-30")
    cot_df = pd.DataFrame({
        "currency": ["EUR", "EUR"],
        "report_date": [pd.Timestamp("2020-01-07"), pd.Timestamp("2020-12-29")],
    })
    aligned = rl.align_cot_to_action_dates(cot_df, cal)
    assert len(aligned) == 1
    assert aligned.iloc[0]["report_date"] == pd.Timestamp("2020-01-07")
