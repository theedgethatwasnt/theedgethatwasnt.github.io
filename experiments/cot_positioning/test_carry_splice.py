"""Tests for carry_splice.py — the pre-2020 flat-FRED / 2020+ full-model splice."""
import pandas as pd
import pytest

import carry_model as cm
import carry_splice as cs


def test_fred_cache_covers_2008_onward_for_all_currencies():
    """The COT joint window starts ~2008 (156-week z-score warmup) — every FRED series
    must have data at or before 2008-01-01."""
    fred = cm._load_fred()
    for ccy in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"):
        dates, _ = fred[ccy]
        assert dates.min() <= __import__("numpy").datetime64("2008-01-01"), ccy


def test_post_splice_matches_full_carry_model_exactly():
    """On/after SPLICE_DATE, carry_pips_spliced must be bit-identical to calling
    carry_model.carry_pips directly with the requested markup_mult."""
    entry, exit_ = "2024-06-03T15:00:00Z", "2024-06-05T15:00:00Z"
    spliced = cs.carry_pips_spliced("USD_JPY", -1, entry, exit_, markup_mult=1.0)
    direct = cm.carry_pips("USD_JPY", -1, entry, exit_, markup_mult=1.0)
    assert spliced == pytest.approx(direct, rel=1e-9)


def test_pre_splice_uses_zero_markup_ie_flat_fred_diff():
    """Before SPLICE_DATE, carry_pips_spliced must equal carry_model.carry_pips called
    with markup_mult=0.0 (the pinch term vanishes -> flat FRED-differential only),
    REGARDLESS of what markup_mult the caller requested."""
    entry, exit_ = "2015-06-03T15:00:00Z", "2015-06-05T15:00:00Z"
    spliced_req1 = cs.carry_pips_spliced("USD_JPY", -1, entry, exit_, markup_mult=1.0)
    spliced_req2 = cs.carry_pips_spliced("USD_JPY", -1, entry, exit_, markup_mult=2.0)
    flat_fred = cm.carry_pips("USD_JPY", -1, entry, exit_, markup_mult=0.0)
    assert spliced_req1 == pytest.approx(flat_fred, rel=1e-9)
    assert spliced_req2 == pytest.approx(flat_fred, rel=1e-9)
    assert flat_fred != 0.0  # sanity: the flat differential itself is non-trivial


def test_boundary_week_splits_and_sums_correctly():
    """A hold spanning the splice boundary must equal the sum of the pre-boundary leg
    (markup_mult=0.0) and post-boundary leg (markup_mult=requested)."""
    entry = "2020-11-09T15:00:00Z"   # before splice (2020-11-11)
    exit_ = "2020-11-13T15:00:00Z"   # after splice
    spliced = cs.carry_pips_spliced("EUR_USD", +1, entry, exit_, markup_mult=1.0)
    pre = cm.carry_pips("EUR_USD", +1, entry, cs.SPLICE_DATE, markup_mult=0.0)
    post = cm.carry_pips("EUR_USD", +1, cs.SPLICE_DATE, exit_, markup_mult=1.0)
    assert spliced == pytest.approx(pre + post, rel=1e-9)
    assert spliced != pytest.approx(cm.carry_pips("EUR_USD", +1, entry, exit_, markup_mult=1.0), rel=1e-9), (
        "boundary week must actually differ from the naive full-markup calculation, "
        "otherwise the splice isn't doing anything on this test"
    )


def test_usd_cad_usd_chf_snapshot_present():
    """These two pairs were added to financing_snapshot_2026-07-07.json specifically for
    this experiment (multiday_contrarian's snapshot didn't have them) — must be loadable."""
    snap = cm._load_snapshot()["pairs"]
    assert "USD_CAD" in snap and "USD_CHF" in snap
    for pair in ("USD_CAD", "USD_CHF"):
        p = cs.carry_pips_spliced(pair, +1, "2024-01-08T15:00:00Z", "2024-01-10T15:00:00Z")
        assert isinstance(p, float)
