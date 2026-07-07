#!/usr/bin/env python3
"""
fetch_cot.py — COT Contrarian Positioning: CFTC Commitments of Traders fetcher/parser.

Governed by PREREGISTRATION.md "Data (fixed)": CFTC Commitments of Traders, Legacy
report, FUTURES-ONLY, weekly net non-commercial (speculator) positions for the CME
currency futures EUR, JPY, GBP, CHF, AUD, CAD, NZD, from the free annual archives.

Source (verified live 2026-07-07, HTTP 200 for every year 2005-2026):
    https://www.cftc.gov/files/dea/history/deacot<YEAR>.zip
Each zip contains one file, `annual.txt`, a CSV with a header row. Column layout is
stable by NAME across the whole 2005-2026 range (verified: the 5 columns this parser
reads are byte-identical strings in the 2005 and 2026 headers; only one unrelated
column, "CFTC Commodity Code (Quotes)", gained/lost a trailing space across years,
which is why this parser selects columns BY NAME, never by position).

Columns used:
  "Market and Exchange Names"          -> currency identification (exact-string match)
  "As of Date in Form YYYY-MM-DD"      -> report date (as-of Tuesday)
  "Open Interest (All)"                -> OI
  "Noncommercial Positions-Long (All)" -> speculator long
  "Noncommercial Positions-Short (All)"-> speculator short

Market-name matching pitfall (caught during recon, 2026-07-07): "JAPANESE YEN" is a
SUBSTRING of a totally different contract, "EURO FX/JAPANESE YEN XRATE - NEW YORK BOARD
OF TRADE" (a NYBOT cross-rate contract, not the CME JPY future we want). A naive
substring match on "JAPANESE YEN" would silently pull in that unrelated series. This
parser therefore requires an EXACT match (after strip()) against the full market name
string, e.g. "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", not a substring test.

Usage:
  python3 fetch_cot.py --years 2005-2026 --out cot_weekly.parquet
"""
import argparse
import csv
import io
import os
import urllib.request
import zipfile
from datetime import date

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "cot_weekly.parquet")

URL_TMPL = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"

# Exact (stripped) "Market and Exchange Names" strings for the 7 CME FX futures.
# USD is implicit as the base leg on every one of these (CME convention: long the
# listed currency = long <currency>/USD).
#
# Two currencies were RENAMED by CFTC mid-2022 (discovered during recon, 2026-07-07):
#   GBP: "BRITISH POUND STERLING - ..." (thru 2022-02-01) -> "BRITISH POUND - ..." (from 2022-02-08)
#   NZD: "NEW ZEALAND DOLLAR - ..."     (thru 2022-02-01) -> "NZ DOLLAR - ..."     (from 2022-02-08)
# Verified clean hard cutover (no overlapping report_date between old/new name for either
# currency — spot-checked the full 2022 annual.txt), so both aliases safely map to one
# currency series with no double-counting risk. Each value below is a LIST of exact
# (stripped) names to match; "exact" (not substring) matching is deliberate — see the
# module docstring's JAPANESE YEN / XRATE pitfall.
CFTC_MARKETS = {
    "EUR": ["EURO FX - CHICAGO MERCANTILE EXCHANGE"],
    "JPY": ["JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE"],
    "GBP": ["BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE",
            "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE"],
    "CHF": ["SWISS FRANC - CHICAGO MERCANTILE EXCHANGE"],
    "AUD": ["AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE"],
    "CAD": ["CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE"],
    "NZD": ["NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE",
            "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE"],
}

COLS = {
    "name": "Market and Exchange Names",
    "date": "As of Date in Form YYYY-MM-DD",
    "oi": "Open Interest (All)",
    "long": "Noncommercial Positions-Long (All)",
    "short": "Noncommercial Positions-Short (All)",
}


def fetch_year_bytes(year: int, timeout: int = 30) -> bytes:
    url = URL_TMPL.format(year=year)
    req = urllib.request.Request(url, headers={"User-Agent": "fx-core-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_annual_zip(raw_zip_bytes: bytes) -> pd.DataFrame:
    """Parse one year's deacot<YEAR>.zip bytes -> long DataFrame filtered to the 7 CME FX
    futures, columns: currency, report_date, oi, noncomm_long, noncomm_short."""
    zf = zipfile.ZipFile(io.BytesIO(raw_zip_bytes))
    inner_name = "annual.txt" if "annual.txt" in zf.namelist() else zf.namelist()[0]
    with zf.open(inner_name) as f:
        text = io.TextIOWrapper(f, encoding="latin-1")
        reader = csv.DictReader(text)
        rows_by_currency = {ccy: [] for ccy in CFTC_MARKETS}
        name_to_ccy = {name: ccy for ccy, names in CFTC_MARKETS.items() for name in names}
        for row in reader:
            name = row[COLS["name"]].strip()
            ccy = name_to_ccy.get(name)
            if ccy is None:
                continue
            rows_by_currency[ccy].append({
                "currency": ccy,
                "report_date": row[COLS["date"]],
                "oi": int(row[COLS["oi"]]),
                "noncomm_long": int(row[COLS["long"]]),
                "noncomm_short": int(row[COLS["short"]]),
            })
    all_rows = [r for rows in rows_by_currency.values() for r in rows]
    return pd.DataFrame(all_rows)


def build_cot_weekly(years, cache_dir=None, verbose=True) -> pd.DataFrame:
    frames = []
    for year in years:
        cache_path = os.path.join(cache_dir, f"deacot{year}.zip") if cache_dir else None
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                raw = f.read()
        else:
            raw = fetch_year_bytes(year)
            if cache_path:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(raw)
        df = parse_annual_zip(raw)
        if verbose:
            print(f"{year}: {len(df)} rows, {df['currency'].nunique()} currencies", flush=True)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["report_date"] = pd.to_datetime(out["report_date"])
    out = out.drop_duplicates(subset=["currency", "report_date"]).sort_values(
        ["currency", "report_date"]
    ).reset_index(drop=True)
    out["net_noncomm"] = out["noncomm_long"] - out["noncomm_short"]
    out["net_noncomm_frac_oi"] = out["net_noncomm"] / out["oi"].replace(0, pd.NA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2005-2026", help="e.g. 2005-2026")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "_cot_zip_cache"))
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.years.split("-"))
    years = list(range(lo, hi + 1))

    df = build_cot_weekly(years, cache_dir=args.cache_dir)
    df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}: {len(df)} rows")
    for ccy, g in df.groupby("currency"):
        print(f"  {ccy}: {len(g)} weeks, {g['report_date'].min().date()} -> {g['report_date'].max().date()}")


if __name__ == "__main__":
    main()
