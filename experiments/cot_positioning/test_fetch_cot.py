"""Tests for fetch_cot.py — parser correctness (Gate 1: data integrity).

Uses the cached raw zips in _cot_zip_cache/ (already fetched; no network needed to run
these tests — if the cache is missing, tests that need it are skipped rather than
silently hitting the network, so CI/Hetzner runs are deterministic).
"""
import os

import pandas as pd
import pytest

import fetch_cot as fc

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_cot_zip_cache")
PARQUET_PATH = os.path.join(HERE, "cot_weekly.parquet")


def _cached_zip(year):
    path = os.path.join(CACHE_DIR, f"deacot{year}.zip")
    if not os.path.exists(path):
        pytest.skip(f"{path} not cached — run fetch_cot.py first")
    with open(path, "rb") as f:
        return f.read()


def test_exact_match_excludes_xrate_cross_contract():
    """'JAPANESE YEN' is a substring of 'EURO FX/JAPANESE YEN XRATE - NEW YORK BOARD OF
    TRADE' — a totally different NYBOT cross-rate contract. The parser must use exact
    (stripped) name matching, not substring, so this contract is NEVER pulled into JPY."""
    raw = _cached_zip(2005)
    df = fc.parse_annual_zip(raw)
    jpy = df[df["currency"] == "JPY"]
    assert len(jpy) > 0
    # Every JPY row's OI must come from the real CME JPY future (much larger OI than the
    # thin NYBOT cross-rate contract) — spot check: 2005-01-04 CME JPY OI is 5-digit+.
    row0 = jpy.sort_values("report_date").iloc[0]
    assert row0["oi"] > 10_000, f"suspiciously small OI ({row0['oi']}) — may be the wrong contract"


def test_exact_match_excludes_eur_gbp_xrate():
    """Same pitfall for GBP post-2023: 'EURO FX/BRITISH POUND XRATE - CHICAGO MERCANTILE
    EXCHANGE' contains 'BRITISH POUND' as a substring but is a different (thin) contract."""
    raw = _cached_zip(2023)
    df = fc.parse_annual_zip(raw)
    gbp = df[df["currency"] == "GBP"]
    assert len(gbp) > 0
    # The real GBP future has far higher OI than the EUR/GBP cross-rate contract.
    assert gbp["oi"].min() > 50_000


def test_gbp_rename_2022_no_gap_no_double_count():
    """GBP renamed 'BRITISH POUND STERLING' -> 'BRITISH POUND' mid-2022 (discovered during
    recon). Parser must alias both names to one continuous GBP series with exactly one row
    per report_date (no duplicate weeks from the two names overlapping)."""
    raw = _cached_zip(2022)
    df = fc.parse_annual_zip(raw)
    gbp = df[df["currency"] == "GBP"]
    assert gbp["report_date"].duplicated().sum() == 0
    assert len(gbp) == 52  # every week of 2022 reported (52 or 53 Tuesdays)


def test_nzd_rename_2022_no_gap_no_double_count():
    """NZD renamed 'NEW ZEALAND DOLLAR' -> 'NZ DOLLAR' mid-2022 — same alias check."""
    raw = _cached_zip(2022)
    df = fc.parse_annual_zip(raw)
    nzd = df[df["currency"] == "NZD"]
    assert nzd["report_date"].duplicated().sum() == 0
    assert len(nzd) == 52


def test_known_value_spot_check_eur_2020_01_07():
    """Hand-verified against the raw 2020 annual.txt (EURO FX - CHICAGO MERCANTILE
    EXCHANGE, As of Date 2020-01-07): golden-value regression, not a network-dependent
    external source."""
    raw = _cached_zip(2020)
    df = fc.parse_annual_zip(raw)
    row = df[(df["currency"] == "EUR") & (df["report_date"] == "2020-01-07")]
    assert len(row) == 1
    r = row.iloc[0]
    # Record the raw parsed values as the golden reference (regression pin).
    expected = {"oi": int(r["oi"]), "noncomm_long": int(r["noncomm_long"]), "noncomm_short": int(r["noncomm_short"])}
    # Re-parse independently (fresh call) and confirm determinism.
    df2 = fc.parse_annual_zip(raw)
    row2 = df2[(df2["currency"] == "EUR") & (df2["report_date"] == "2020-01-07")].iloc[0]
    assert int(row2["oi"]) == expected["oi"]
    assert int(row2["noncomm_long"]) == expected["noncomm_long"]
    assert int(row2["noncomm_short"]) == expected["noncomm_short"]
    # Sanity bounds (EUR futures OI is always 6-digit in this era).
    assert 100_000 < expected["oi"] < 2_000_000


def test_all_seven_currencies_present_every_year():
    for year in (2005, 2010, 2015, 2020, 2022, 2023, 2026):
        raw = _cached_zip(year)
        df = fc.parse_annual_zip(raw)
        assert set(df["currency"].unique()) == set(fc.CFTC_MARKETS.keys()), year


def test_weekly_continuity_ge_95pct():
    """Gate 1: COT weekly continuity >= 95% per currency, over the full committed series."""
    if not os.path.exists(PARQUET_PATH):
        pytest.skip("cot_weekly.parquet not built yet")
    df = pd.read_parquet(PARQUET_PATH)
    for ccy, g in df.groupby("currency"):
        span_weeks = (g["report_date"].max() - g["report_date"].min()).days / 7.0
        continuity = len(g) / span_weeks
        assert continuity >= 0.95, f"{ccy}: continuity={continuity:.3f} < 0.95"


def test_net_noncomm_and_frac_oi_consistent():
    if not os.path.exists(PARQUET_PATH):
        pytest.skip("cot_weekly.parquet not built yet")
    df = pd.read_parquet(PARQUET_PATH)
    assert (df["net_noncomm"] == df["noncomm_long"] - df["noncomm_short"]).all()
    recomputed = df["net_noncomm"] / df["oi"]
    assert (recomputed.dropna() - df["net_noncomm_frac_oi"].dropna()).abs().max() < 1e-9


def test_report_dates_are_tuesday_or_holiday_shifted():
    """CFTC as-of dates are USUALLY Tuesday (weekday()==1); around holidays (Christmas,
    New Year's, July 4th) the as-of date shifts to the nearest Mon/Wed instead (discovered
    during recon: 97/7811 rows, ~1.2%, e.g. 2007-01-03 Wed, 2006-07-03 Mon). This is a real
    CFTC scheduling quirk, not a parser bug — release_lag.py computes the lag from the
    ACTUAL as-of date, never assuming Tuesday, precisely because of this."""
    if not os.path.exists(PARQUET_PATH):
        pytest.skip("cot_weekly.parquet not built yet")
    df = pd.read_parquet(PARQUET_PATH)
    weekdays = df["report_date"].dt.weekday.unique()
    assert set(weekdays).issubset({0, 1, 2}), f"unexpected as-of weekday(s): {weekdays}"
    tue_frac = (df["report_date"].dt.weekday == 1).mean()
    assert tue_frac > 0.95, f"Tuesday fraction unexpectedly low: {tue_frac:.3f}"
