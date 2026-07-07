#!/usr/bin/env python3
"""
carry_model.py — Multi-day contrarian program (Workstream A1): carry / financing cost model.

Governed by PREREGISTRATION.md § "Cost model (locked)". Inherits the broker-truth PIPS
formula from research/experiments/carry/carry_financing.py:

    carry_pips_per_charged_day = (rate / 365) * (price / pip)

where `rate` is OANDA's published annualized instrument financing rate (already includes
the retail "pinch" markup) and `price` is the pair's spot mid price (base->quote pip
conversion). OANDA charges a multi-day "triple swap" on one weekday per instrument
(`financingDaysOfWeek`, e.g. most pairs charge 3x on Wednesday to cover the weekend;
NZD pairs in our snapshot charge 4x on Tuesday instead — verified live, not assumed).

WHY AN APPROXIMATION IS NEEDED (documented per SOP rule R9)
------------------------------------------------------------
OANDA's `GET /v3/accounts/{aid}/instruments` endpoint returns only the CURRENT financing
rate — there is no historical financing-rate endpoint. To backtest carry over 2020-11 ->
2026-05 we need a rate for every historical date. We approximate with two explicit,
documented assumptions:

1. RATE LEVEL is scaled through time by the FRED policy-rate differential, holding the
   currently-measured "pinch" (OANDA's markup over the raw differential) CONSTANT
   (additive) through time:

       pinch(pair, dir)   = snapshot_rate(pair, dir) - fred_diff_dir(pair, dir, snapshot_date)
       rate_dir(pair, dir, t) = fred_diff_dir(pair, dir, t) + markup_mult * pinch(pair, dir)

   At t = snapshot_date and markup_mult = 1.0 this reproduces the snapshot rate exactly
   (by construction). `markup_mult` in {0.5, 1.0, 2.0} is the pre-registered sensitivity
   hook — 1.0 is the measured markup, 2.0 is the pessimistic case reported alongside the
   money criterion. Direction of bias from this approximation: uncertain (rates float with
   policy in ways the constant-pinch model cannot capture), mitigated by the 2.0x check.

2. PRICE/PIP CONVERSION uses the FROZEN SNAPSHOT mid price for every historical date, not
   the historical spot price (inherited directly from carry_financing.py, which was
   itself snapshot-only — there was no historical-price carry tool to "extend"). This is a
   SECOND documented approximation beyond #1: JPY-cross pip conversion drifts most since
   spot moved ~110 (2020) -> ~162 (2026), a ~47% swing. Direction of bias: uncertain in
   net (partially self-cancelling since the carry direction on JPY crosses, long-the-cross,
   is held constant across the whole sample — it is not regime-dependent flip-flopping).
   Flagged, not resolved; the 2.0x markup_mult sensitivity is the mitigation the
   pre-registration specifies.

FRED SERIES CHOICES (one policy-rate proxy per currency, all verified live 2026-07-06 to
have data covering the IS+OOS window 2020-11-11 -> 2026-05-21 AND to be reasonably current)
------------------------------------------------------------------------------------------
  USD -> DFF              Effective Federal Funds Rate, daily.
  EUR -> ECBDFR           ECB Deposit Facility Rate, daily.
  GBP -> IRSTCI01GBM156N  OECD MEI "Immediate Rates: Less than 24 Hours", monthly.
  JPY -> IRSTCI01JPM156N  OECD MEI "Immediate Rates: Less than 24 Hours", monthly.
  AUD -> IRSTCI01AUM156N  OECD MEI "Immediate Rates: Less than 24 Hours", monthly.
  CAD -> IRSTCI01CAM156N  OECD MEI "Immediate Rates: Less than 24 Hours", monthly.
  NZD -> IR3TIB01NZM156N  OECD MEI 3-Month Interbank Rate, monthly. (NZ's own
                          IRSTCI01NZM156N overnight series stopped updating 2024-12 —
                          verified via probe on 2026-07-06 — so we use the 3-month
                          interbank series instead, which is current through 2026-05.)
  CHF -> IR3TIB01CHM156N  OECD MEI 3-Month Interbank Rate, monthly. (CH's own
                          IRSTCI01CHM156N overnight series went stale 2024-03 — verified
                          via probe — so we use the 3-month interbank series instead,
                          current through 2026-05.)

All series values are ANNUAL PERCENT (e.g. "3.63" = 3.63%/yr); `fred_rate()` returns the
decimal (0.0363) to match OANDA's rate convention. Monthly series are treated as
piecewise-constant (as-of / forward-filled): the rate observed on the 1st of a month
applies to every day of that month until the next observation.

Data artifacts (committed):
  financing_snapshot_2026-07-06.json — one-time OANDA pull (account 010, 2026-07-06) of
    longRate/shortRate/financingDaysOfWeek/mid spot price for the 12 traded pairs.
  fred_rates.parquet — cached FRED pulls for the 8 series above, built by
    `python3 carry_model.py --refresh-fred` (network required; run once, commit the
    parquet, do not require network at test/backtest time).
"""
import argparse
import io
import json
import os
import urllib.request
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.path.join(HERE, "financing_snapshot_2026-07-06.json")
FRED_CACHE_PATH = os.path.join(HERE, "fred_rates.parquet")
SNAPSHOT_DATE = date(2026, 7, 6)

NY = ZoneInfo("America/New_York")

FRED_SERIES = {
    "USD": "DFF",
    "EUR": "ECBDFR",
    "GBP": "IRSTCI01GBM156N",
    "JPY": "IRSTCI01JPM156N",
    "AUD": "IRSTCI01AUM156N",
    "CAD": "IRSTCI01CAM156N",
    "NZD": "IR3TIB01NZM156N",
    "CHF": "IR3TIB01CHM156N",
}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"

_WEEKDAY_NAMES = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]


def pip_of(pair):
    return 0.01 if pair.endswith("_JPY") else 0.0001


def _base_quote(pair):
    base, quote = pair.split("_")
    return base, quote


# ── snapshot + FRED cache loaders (lazy, memoized) ───────────────────────────
_snapshot_cache = None


def _load_snapshot():
    global _snapshot_cache
    if _snapshot_cache is None:
        with open(SNAPSHOT_PATH) as f:
            _snapshot_cache = json.load(f)
    return _snapshot_cache


_fred_cache = None


def _load_fred():
    """Returns {currency: (sorted_dates[datetime64[D]], rates[float, decimal])}."""
    global _fred_cache
    if _fred_cache is None:
        df = pd.read_parquet(FRED_CACHE_PATH)
        out = {}
        for ccy, g in df.groupby("currency"):
            g = g.sort_values("date")
            out[ccy] = (
                g["date"].values.astype("datetime64[D]"),
                g["rate_pct"].values.astype(float) / 100.0,
            )
        _fred_cache = out
    return _fred_cache


def fred_rate(ccy, d):
    """As-of (piecewise-constant, forward-filled) FRED policy rate, decimal annual, for date d."""
    dates, rates = _load_fred()[ccy]
    d64 = np.datetime64(d, "D")
    idx = np.searchsorted(dates, d64, side="right") - 1
    if idx < 0:
        idx = 0  # before first observation: use earliest known rate (documented approx)
    return float(rates[idx])


# ── rate-scaling model (§1 of the module docstring) ──────────────────────────
def _fred_diff_dir(pair, direction, d):
    base, quote = _base_quote(pair)
    raw = fred_rate(base, d) - fred_rate(quote, d)
    return raw if direction > 0 else -raw


def _pinch(pair, direction):
    snap = _load_snapshot()["pairs"][pair]
    rate_applied = snap["longRate"] if direction > 0 else snap["shortRate"]
    fred_snap = _fred_diff_dir(pair, direction, SNAPSHOT_DATE)
    return rate_applied - fred_snap


def _rate_dir(pair, direction, d, markup_mult=1.0):
    return _fred_diff_dir(pair, direction, d) + markup_mult * _pinch(pair, direction)


# ── rollover-day counting (§ position-holding accounting) ────────────────────
def _to_ny(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.tz_convert(NY)


def _rollover_dates_crossed(entry_ts, exit_ts):
    """NY calendar dates whose 17:00-NY rollover falls in (entry_ts, exit_ts] — the position
    must be open AT the rollover moment to be charged that day's financing (R1: closed-bar-only
    analog — you must actually hold through the instant, not merely overlap the calendar day)."""
    e = _to_ny(entry_ts)
    x = _to_ny(exit_ts)
    if x <= e:
        return []
    out = []
    d = e.date()
    end = x.date()
    while d <= end:
        rollover = datetime(d.year, d.month, d.day, 17, 0, 0, tzinfo=NY)
        if e < rollover <= x:
            out.append(d)
        d = d + timedelta(days=1)
    return out


def carry_pips(pair, direction, entry_ts, exit_ts, notional=1.0, markup_mult=1.0):
    """Net carry cost/credit in PIPS for `notional` units, `direction` (+1 long / -1 short),
    held from entry_ts to exit_ts (tz-aware or naive-UTC timestamp-likes: str/datetime/
    pandas.Timestamp all accepted). Charges once per OANDA-style 17:00-New-York rollover
    crossed, weighted by that instrument's financingDaysOfWeek (captures the triple-swap
    day, which is Wednesday for most pairs but Tuesday for the NZD pairs in our snapshot).
    Rate is scaled through time per the FRED-differential + constant-measured-pinch model
    (module docstring §1); price/pip conversion uses the frozen snapshot spot (§2).
    `markup_mult` is the pre-registered sensitivity hook (0.5 / 1.0 / 2.0)."""
    if direction not in (1, -1, 1.0, -1.0):
        raise ValueError(f"direction must be +1 or -1, got {direction!r}")
    direction = 1 if direction > 0 else -1

    snap = _load_snapshot()["pairs"][pair]
    price = snap["mid_price_snapshot"]
    pip = pip_of(pair)
    dow_map = snap["financingDaysOfWeek"]

    total = 0.0
    for d in _rollover_dates_crossed(entry_ts, exit_ts):
        wname = _WEEKDAY_NAMES[d.weekday()]
        days_charged = dow_map.get(wname, 1)
        if days_charged == 0:
            continue
        rate = _rate_dir(pair, direction, d, markup_mult)
        total += (rate / 365.0) * (price / pip) * days_charged
    return total * notional


# ── one-time FRED cache build (network required; run once, commit the parquet) ──
def _fetch_fred_series(series_id):
    url = FRED_CSV_URL.format(id=series_id)
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode()
    df = pd.read_csv(io.StringIO(raw))
    df.columns = ["date", "value"]
    df = df[df["value"].astype(str).str.strip() != "."]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["value"] = df["value"].astype(float)
    return df


def build_fred_cache(out_path=FRED_CACHE_PATH):
    frames = []
    for ccy, sid in FRED_SERIES.items():
        df = _fetch_fred_series(sid)
        df = df.rename(columns={"value": "rate_pct"})
        df["currency"] = ccy
        frames.append(df[["date", "currency", "rate_pct"]])
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(out_path, index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-fred", action="store_true", help="re-fetch fred_rates.parquet from FRED (network)")
    args = ap.parse_args()
    if args.refresh_fred:
        df = build_fred_cache()
        print(f"wrote {FRED_CACHE_PATH}: {len(df)} rows, {df['currency'].nunique()} currencies")
        for ccy, g in df.groupby("currency"):
            print(f"  {ccy}: {g['date'].min()} -> {g['date'].max()}  ({len(g)} obs)")
