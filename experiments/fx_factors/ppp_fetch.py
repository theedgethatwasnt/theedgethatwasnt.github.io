#!/usr/bin/env python3
"""
ppp_fetch.py — Value factor (fx_factors program): World Bank PPP conversion-rate fetch.

Source: World Bank Open Data, World Development Indicators, indicator PA.NUS.PPP
("PPP conversion factor, GDP (LCU per international $)"), bulk CSV export endpoint
https://api.worldbank.org/v2/en/indicator/PA.NUS.PPP?downloadformat=csv — free, no API key,
documented at https://data.worldbank.org/indicator/PA.NUS.PPP.

R9 note: the row-level query API (.../v2/country/{iso3}/indicator/PA.NUS.PPP?format=json)
returned HTTP 502 (Microsoft-Azure-Application-Gateway) on every param combination tried from
this box on 2026-07-07 (country filter, mrv, source=2, http/1.1 downgrade, repeated retries),
while OTHER indicators (e.g. NY.GDP.MKTP.CD) succeeded on the identical query path and the
BULK CSV export for this SAME indicator succeeded immediately — an upstream World Bank
backend issue specific to that one query path, not a network/auth/rate-limit problem here.
The bulk CSV export is the documented, equally-official workaround (same underlying data,
same WDI source note) and is what this script uses.

Country/currency mapping (one representative country per currency; PPP is a national-accounts
concept, so a currency union needs a proxy country):
  USD->USA  GBP->GBR  JPY->JPN  AUD->AUS  NZD->NZL  CAD->CAN  CHF->CHE
  EUR->DEU (Germany): World Bank does NOT publish a PA.NUS.PPP value for the "Euro area"
    (EMU) aggregate row — confirmed EMPTY for every year in the fetched CSV (2026-07-07).
    DEU (Germany, the largest Eurozone economy by GDP) is used as the single-country proxy.
    Germany's LCU has been EUR since 1999, fully covering this program's 2020+ backtest
    window, so there is no currency-redenomination discontinuity to handle.
    Documented approximation (R9): a GDP-weighted multi-country Euro basket (DEU+FRA+ITA+...)
    would reduce single-country idiosyncrasy but needs a second indicator fetch + a weighting
    scheme, for a factor that is SECONDARY (never gated) in this pre-registration. Not built.

Publication-lag rule (locked, PREREGISTRATION.md "6-month publication lag — no lookahead"):
  The annual PPP value for calendar year Y becomes usable in the backtest starting
  (Y+1)-07-01 (6 months after year-end Dec-31-Y). Applied identically to every
  country/year — no per-country customization, no lookahead.

Usage (network required; run once, commit the parquet):
  /root/venv/bin/python3 ppp_fetch.py --refresh
  /root/venv/bin/python3 ppp_fetch.py --selftest    # prints the as-of table, no network
"""
import argparse
import io
import os
import urllib.request
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "ppp_rates.parquet")

BULK_CSV_URL = "https://api.worldbank.org/v2/en/indicator/PA.NUS.PPP?downloadformat=csv"
INDICATOR_CSV_PREFIX = "API_PA.NUS.PPP_DS2_en_csv_v2"

CCY_TO_ISO3 = {
    "USD": "USA",
    "EUR": "DEU",  # Euro-area proxy — see module docstring
    "GBP": "GBR",
    "JPY": "JPN",
    "AUD": "AUS",
    "NZD": "NZL",
    "CAD": "CAN",
    "CHF": "CHE",
}

PUBLICATION_LAG_MONTHS = 6


def fetch_bulk_csv(url=BULK_CSV_URL, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "fx-core-fx_factors/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    csv_name = next(n for n in zf.namelist() if n.startswith(INDICATOR_CSV_PREFIX))
    with zf.open(csv_name) as f:
        # World Bank's bulk CSV has 4 junk header lines before the real header row.
        df = pd.read_csv(f, skiprows=4)
    return df


def parse_ppp(df):
    """Long-format frame: columns [currency, year, ppp_rate] (ppp_rate = LCU per intl $)."""
    year_cols = [c for c in df.columns if c.isdigit()]
    rows = []
    for ccy, iso3 in CCY_TO_ISO3.items():
        sub = df[df["Country Code"] == iso3]
        if len(sub) == 0:
            raise RuntimeError(f"{ccy} ({iso3}): no row found in World Bank PPP CSV")
        rec = sub.iloc[0]
        for yc in year_cols:
            val = rec[yc]
            if pd.notna(val) and val != "":
                rows.append({"currency": ccy, "year": int(yc), "ppp_rate": float(val)})
    out = pd.DataFrame(rows).sort_values(["currency", "year"]).reset_index(drop=True)
    if len(out) == 0:
        raise RuntimeError("parsed 0 PPP rows — check World Bank CSV format")
    return out


def build_cache(out_path=OUT_PATH):
    raw = fetch_bulk_csv()
    long_df = parse_ppp(raw)
    long_df.to_parquet(out_path, index=False)
    return long_df


# ── as-of lookup (used by currency_index.py) ─────────────────────────────────
_ppp_cache = None


def _load_ppp(path=OUT_PATH):
    global _ppp_cache
    if _ppp_cache is None:
        df = pd.read_parquet(path)
        out = {}
        for ccy, g in df.groupby("currency"):
            g = g.sort_values("year")
            # usable_from: the first UTC date on which year Y's figure is allowed to be read
            # (6-month publication lag from year-end).
            usable_from = pd.to_datetime((g["year"] + 1).astype(str) + "-07-01", utc=True)
            out[ccy] = (usable_from.values, g["ppp_rate"].values.astype(float))
        _ppp_cache = out
    return _ppp_cache


def ppp_asof(ccy, d, path=OUT_PATH):
    """As-of (piecewise-constant, publication-lag-respecting) PPP conversion rate
    (LCU per international $) for currency `ccy` on date `d`. Returns NaN if `d` is before
    the first publishable observation (documented: no PPP signal that early)."""
    dates, rates = _load_ppp(path)[ccy]
    d64 = pd.Timestamp(d)
    if d64.tzinfo is None:
        d64 = d64.tz_localize("UTC")
    else:
        d64 = d64.tz_convert("UTC")
    d64 = d64.to_datetime64()
    import numpy as np
    idx = np.searchsorted(dates, d64, side="right") - 1
    if idx < 0:
        return float("nan")
    return float(rates[idx])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-fetch ppp_rates.parquet from World Bank (network)")
    ap.add_argument("--selftest", action="store_true", help="print the as-of table (no network)")
    args = ap.parse_args()

    if args.refresh:
        df = build_cache()
        print(f"wrote {OUT_PATH}: {len(df)} rows, {df['currency'].nunique()} currencies")
        for ccy, g in df.groupby("currency"):
            print(f"  {ccy} ({CCY_TO_ISO3[ccy]}): {g['year'].min()}->{g['year'].max()} ({len(g)} obs) latest={g.sort_values('year').iloc[-1]['ppp_rate']:.4f}")

    if args.selftest:
        for ccy in CCY_TO_ISO3:
            v = ppp_asof(ccy, "2024-01-01")
            print(f"{ccy}: ppp_asof(2024-01-01) = {v}")
