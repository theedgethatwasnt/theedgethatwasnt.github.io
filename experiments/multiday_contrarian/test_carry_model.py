"""Tests for carry_model.py (Workstream A1). Run on the Hetzner box per CLAUDE.md directive:
    rsync -az research/experiments/multiday_contrarian/ root@HETZNER:/root/multiday/code/
    ssh root@HETZNER 'cd /root/multiday/code && /root/venv/bin/python -m pytest test_carry_model.py -x -q'
"""
import pytest

import carry_model as cm


def test_snapshot_and_fred_cache_load():
    """Data artifacts committed alongside the module must be present and parse."""
    snap = cm._load_snapshot()
    assert snap["source_account"] == "010"
    assert len(snap["pairs"]) == 12
    assert "EUR_USD" in snap["pairs"] and "USD_JPY" in snap["pairs"]
    fred = cm._load_fred()
    assert set(fred.keys()) == set(cm.FRED_SERIES.keys())
    for ccy, (dates, rates) in fred.items():
        assert len(dates) > 0
        assert dates.min() <= __import__("numpy").datetime64("2020-11-11")


def test_zero_intraday_roundtrip():
    """No 17:00-NY rollover crossed within the same day -> zero carry, any pair/direction."""
    p = cm.carry_pips("EUR_USD", +1, "2024-06-10T12:00:00Z", "2024-06-10T16:00:00Z")
    assert p == 0.0
    p2 = cm.carry_pips("USD_JPY", -1, "2024-06-10T00:30:00Z", "2024-06-10T05:00:00Z")
    assert p2 == 0.0


def test_sign_short_usdjpy_pays_2024():
    """2024: US policy rate (~5.25-5.5%) far above BOJ's (near-zero, hiking from March 2024).
    Shorting USD_JPY (short high-yield USD / long low-yield JPY) should be a carry COST."""
    p = cm.carry_pips("USD_JPY", -1, "2024-06-03T15:00:00Z", "2024-06-05T15:00:00Z")
    assert p < 0.0


def test_sign_long_usdjpy_receives_2024():
    """Symmetric check: long USD_JPY in 2024 (long high-yield USD) should be carry-positive."""
    p = cm.carry_pips("USD_JPY", +1, "2024-06-03T15:00:00Z", "2024-06-05T15:00:00Z")
    assert p > 0.0


def test_triple_swap_wednesday():
    """Crossing exactly the Wednesday rollover (financingDaysOfWeek WED=3 for EUR_USD) must be
    3x the pips of crossing exactly a Monday rollover (MON=1), same month (same FRED rate)."""
    snap = cm._load_snapshot()["pairs"]["EUR_USD"]["financingDaysOfWeek"]
    assert snap["WEDNESDAY"] == 3 and snap["MONDAY"] == 1

    # 2025-01-06 = Monday, 2025-01-08 = Wednesday, both within January 2025 (same FRED obs).
    mon = cm.carry_pips("EUR_USD", +1, "2025-01-06T15:00:00Z", "2025-01-06T23:00:00Z")
    wed = cm.carry_pips("EUR_USD", +1, "2025-01-08T15:00:00Z", "2025-01-08T23:00:00Z")
    assert mon != 0.0
    assert wed == pytest.approx(3.0 * mon, rel=1e-9)


def test_triple_swap_nzd_tuesday_variant():
    """NZD pairs in the live snapshot charge 4x on TUESDAY (not the usual Wed=3) — verified
    live via the OANDA API 2026-07-06, not assumed. Confirms the model reads per-pair
    financingDaysOfWeek rather than hardcoding a global convention."""
    dow = cm._load_snapshot()["pairs"]["NZD_USD"]["financingDaysOfWeek"]
    assert dow["TUESDAY"] == 4 and dow["WEDNESDAY"] == 1

    tue = cm.carry_pips("NZD_USD", +1, "2025-01-07T15:00:00Z", "2025-01-07T23:00:00Z")  # Tue Jan 7 2025
    wed = cm.carry_pips("NZD_USD", +1, "2025-01-08T15:00:00Z", "2025-01-08T23:00:00Z")  # Wed Jan 8 2025
    assert tue != 0.0
    assert wed == pytest.approx((1.0 / 4.0) * tue, rel=1e-9)


def test_markup_sensitivity_hook_is_affine():
    """rate_dir(t) = fred_diff_dir(t) + markup_mult * pinch is affine in markup_mult, and every
    other factor (price, pip, days_charged) is markup_mult-independent, so carry_pips(...) must
    be affine in markup_mult too: total(2.0)-total(1.0) == 2 * (total(1.0)-total(0.5))."""
    entry, exit_ = "2024-06-03T15:00:00Z", "2024-06-05T15:00:00Z"
    p05 = cm.carry_pips("USD_JPY", -1, entry, exit_, markup_mult=0.5)
    p10 = cm.carry_pips("USD_JPY", -1, entry, exit_, markup_mult=1.0)
    p20 = cm.carry_pips("USD_JPY", -1, entry, exit_, markup_mult=2.0)
    assert (p20 - p10) == pytest.approx(2.0 * (p10 - p05), rel=1e-9)


def test_markup_sensitivity_pessimistic_is_worse_for_realistic_pinch():
    """Empirically (measured 2026-07-06) OANDA's pinch is a cost drag in both directions on
    every pair in the snapshot (matches carry_financing.py's finding: 'you always receive
    less / pay more than the symmetric differential implies'). So the pre-registration's
    pessimistic 2.0x markup sensitivity must make a paying position pay MORE."""
    entry, exit_ = "2024-06-03T15:00:00Z", "2024-06-05T15:00:00Z"
    p10 = cm.carry_pips("USD_JPY", -1, entry, exit_, markup_mult=1.0)
    p20 = cm.carry_pips("USD_JPY", -1, entry, exit_, markup_mult=2.0)
    assert p10 < 0.0
    assert p20 < p10  # more negative = costs more


def test_reproduces_snapshot_rate_at_snapshot_date():
    """White-box regression: at t=SNAPSHOT_DATE and markup_mult=1.0, _rate_dir must exactly
    equal the raw OANDA snapshot rate (the additive-pinch construction is defined to make
    this true by construction; a broken formula would silently drift)."""
    snap = cm._load_snapshot()["pairs"]["EUR_USD"]
    r_long = cm._rate_dir("EUR_USD", +1, cm.SNAPSHOT_DATE, markup_mult=1.0)
    r_short = cm._rate_dir("EUR_USD", -1, cm.SNAPSHOT_DATE, markup_mult=1.0)
    assert r_long == pytest.approx(snap["longRate"], abs=1e-12)
    assert r_short == pytest.approx(snap["shortRate"], abs=1e-12)


def test_notional_scales_linearly():
    p1 = cm.carry_pips("USD_JPY", -1, "2024-06-03T15:00:00Z", "2024-06-05T15:00:00Z", notional=1.0)
    p10 = cm.carry_pips("USD_JPY", -1, "2024-06-03T15:00:00Z", "2024-06-05T15:00:00Z", notional=10.0)
    assert p10 == pytest.approx(10.0 * p1, rel=1e-9)


def test_bad_direction_raises():
    with pytest.raises(ValueError):
        cm.carry_pips("EUR_USD", 0, "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")


def test_weekend_hold_only_charges_the_settlement_day():
    """A Friday-to-Monday weekend hold crosses Fri/Sat/Sun rollovers; Sat/Sun have
    daysCharged=0 for EUR_USD so only the Friday charge (weighted 1x) should apply — the
    weekend cost is bundled into Wednesday's triple, not spread across Sat/Sun."""
    # 2025-01-10 = Friday, 2025-01-13 = Monday.
    fri_only = cm.carry_pips("EUR_USD", +1, "2025-01-10T10:00:00Z", "2025-01-10T23:00:00Z")
    weekend = cm.carry_pips("EUR_USD", +1, "2025-01-10T10:00:00Z", "2025-01-13T10:00:00Z")
    assert fri_only != 0.0
    assert weekend == pytest.approx(fri_only, rel=1e-9)
