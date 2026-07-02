#!/usr/bin/env python3
"""
carry_firstpass.py — FX-Core's FIRST carry / long-horizon experiment.

WHY THIS EXISTS
---------------
~60 prior experiments (see research/SYNTHESIS_trend_vs_meanreversion.md and the
auto-memory index) proved that intraday DIRECTION is dead net of OANDA spread.
The only edges that survived were CONTRARIAN-at-long-horizon (retrace 009, CSI H4,
first-touch H4, BB-fade). CARRY is the untested *structural* long-horizon edge:
you earn the interest-rate differential by holding the higher-yielder vs the
lower-yielder for WEEKS. Because the holding period is long, the per-trade OANDA
spread is amortized to near-zero — carry is essentially spread-insensitive.
That makes it categorically different from everything we have tried.

This is a FIRST PASS. It splits the return into:
  (a) SPOT component  — what our M5 mid data actually gives us (price drift in
      the chosen carry direction). This is the RISK side of carry.
  (b) ESTIMATED CARRY overlay — hardcoded annual interest-rate differentials for
      the data era, accrued daily (annual/252 per day held). This is the
      *yield* side. We do NOT have OANDA swap/financing rates in-repo, so this
      is an ESTIMATE, clearly labelled as such.

DATA SPAN (verified at runtime): 2021-01 .. 2026-04. This window contains BOTH
canonical carry stress events:
  - 2022 H2: aggressive Fed hikes -> USD/JPY ramp (carry ON, the slow grind UP
    for JPY-cross longs).
  - 2024-08-05: the yen carry-unwind crash (BoJ hike + soft US payrolls ->
    violent JPY-cross sell-off). This is the canonical "carry crash" tail.
Both are IN-SAMPLE, so the drawdown profile here is real, not hypothetical.

CARRY DIRECTION ASSUMPTIONS (static for the era, documented per pair)
---------------------------------------------------------------------
Carry = LONG the higher-yielding currency, SHORT the lower-yielding one.
Sign convention: signed_return = direction * daily_return_of_base_per_quote.
A pair "EUR_USD" means base=EUR, quote=USD; a LONG (+1) earns EUR rate, pays USD.

JPY crosses (XXX_JPY): JPY policy rate was ~ -0.1% .. 0% for nearly the whole
era (BoJ only lifted to +0.25% in 2024-07). Every other currency yielded more,
so the classic positive-carry trade is LONG the cross (long XXX, short JPY).
  -> USD_JPY, EUR_JPY, GBP_JPY, AUD_JPY, CAD_JPY, NZD_JPY, CHF_JPY = +1.
  (CHF_JPY is the weakest of these — SNB also near zero early — but CHF still
   > JPY for most of the window, so +1.)

Commodity dollars vs USD:
  AUD_USD / NZD_USD — era-dependent. For 2021-01..2026-04 the USD spent the
  *majority* of the window at a HIGHER policy rate than AUD/NZD (Fed hiked to
  5.25-5.50% by 2023; RBA peaked 4.35%, RBNZ 5.50%). Net over the full span the
  USD yield >= AUD/NZD yield more often than not, so the positive-carry trade is
  SHORT the cross => direction = -1 for AUD_USD and NZD_USD. (Pre-2022 it would
  have been +1; we pick the sign that matches the dominant regime of THIS span
  and accrue an era-blended differential below. This is the known weak spot of
  a STATIC-direction first pass and is exactly what the dynamic full version
  fixes.)

European majors (small/variable differential):
  EUR_USD — USD out-yielded EUR for most of the span (Fed > ECB). Positive carry
    = SHORT EUR_USD => -1.
  GBP_USD — BoE and Fed were close; Fed generally >= BoE. SHORT => -1.
  EUR_GBP — BoE > ECB for most of the window. base=EUR is the LOWER yielder, so
    positive carry = SHORT EUR_GBP => -1.

These European signs carry a small/uncertain differential; included for
completeness with best-guess sign, flagged in the carry-estimate table.

ESTIMATED ANNUAL DIFFERENTIALS (carry yield, % per year, era-blended 2021-2026)
-------------------------------------------------------------------------------
These are deliberately conservative whole-span averages (the differential was
small in 2021, peaked 2023-24, compressed late 2024-2025). Applied in the
direction above (so the number is the yield earned by holding the position in
its carry direction; already sign-aligned to be POSITIVE for a genuine carry).
Source: approximate central-bank policy-rate paths for the era; NOT OANDA swaps.

LIMITATIONS / SCOPE OF FULL VERSION — see bottom of file.
"""

import os
import sys
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "m5_ohlc")
DATA_DIR = os.path.abspath(DATA_DIR)

PAIRS = [
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "EUR_GBP",
    "USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY", "NZD_JPY", "CHF_JPY",
]

# direction: +1 = long the cross (long base, short quote); -1 = short the cross
CARRY_DIR = {
    "USD_JPY": +1, "EUR_JPY": +1, "GBP_JPY": +1, "AUD_JPY": +1,
    "CAD_JPY": +1, "NZD_JPY": +1, "CHF_JPY": +1,
    "AUD_USD": -1, "NZD_USD": -1,
    "EUR_USD": -1, "GBP_USD": -1, "EUR_GBP": -1,
}

# Estimated annual carry YIELD (% per yr) earned in the carry direction above.
# Era-blended 2021-2026 average; conservative. NOT OANDA swap rates.
CARRY_ANNUAL_PCT = {
    # JPY crosses: foreign rate minus ~0% JPY, blended over the era
    "USD_JPY": 3.0,   # Fed 0->5.5->4.5; JPY ~0. Big chunk of era at 4-5%.
    "GBP_JPY": 3.0,   # BoE 0.1->5.25
    "EUR_JPY": 1.8,   # ECB -0.5->4.0->late cuts
    "AUD_JPY": 2.5,   # RBA 0.1->4.35
    "NZD_JPY": 2.8,   # RBNZ 0.25->5.5
    "CAD_JPY": 2.7,   # BoC 0.25->5.0
    "CHF_JPY": 0.7,   # SNB -0.75->1.75; weakest carry, still > JPY
    # commodity $ vs USD: SHORT cross earns (USD rate - AUD/NZD rate); era-blended
    # USD slightly out-yielded AUD/NZD on average -> small POSITIVE carry to the short
    "AUD_USD": 0.5,
    "NZD_USD": 0.3,
    # European majors vs the higher-yielding leg: small
    "EUR_USD": 1.2,   # short EUR_USD earns USD-EUR differential (Fed > ECB)
    "GBP_USD": 0.4,   # Fed ~>= BoE, small
    "EUR_GBP": 0.8,   # short EUR_GBP earns GBP-EUR (BoE > ECB)
}

ANN = 252  # trading-day annualization


def load_daily(pair):
    fp = os.path.join(DATA_DIR, f"{pair}_M5.parquet")
    df = pd.read_parquet(fp, columns=["timestamp", "close"])
    df = df.set_index("timestamp").sort_index()
    # daily close = last M5 close of the UTC day
    daily = df["close"].resample("1D").last().dropna()
    return daily


def perf_stats(daily_signed_ret):
    """daily_signed_ret: pd.Series of daily P&L (fraction of notional)."""
    r = daily_signed_ret.dropna()
    n = len(r)
    if n < 2:
        return None
    equity = (1.0 + r).cumprod()
    total = equity.iloc[-1] - 1.0
    years = n / ANN
    ann_ret = (equity.iloc[-1]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    mu, sd = r.mean(), r.std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(ANN) if sd > 0 else np.nan
    # drawdown
    peak = equity.cummax()
    dd = equity / peak - 1.0
    maxdd = dd.min()
    maxdd_date = dd.idxmin()
    underwater = (dd < -1e-9).mean()
    return {
        "n_days": n, "total": total, "ann_ret": ann_ret, "sharpe": sharpe,
        "maxdd": maxdd, "maxdd_date": maxdd_date, "underwater": underwater,
        "equity": equity, "dd": dd,
    }


def main():
    print("=" * 78)
    print("CARRY FIRST-PASS — FX-Core")
    print("=" * 78)

    daily = {}
    for p in PAIRS:
        daily[p] = load_daily(p)

    spans = [(p, d.index.min(), d.index.max(), len(d)) for p, d in daily.items()]
    gmin = min(s[1] for s in spans)
    gmax = max(s[2] for s in spans)
    print(f"\nDATA SPAN: {gmin.date()} .. {gmax.date()}")
    for p, mn, mx, n in spans:
        print(f"  {p}: {mn.date()} .. {mx.date()}  ({n} daily bars)")

    # canonical carry-stress windows (both should be in-sample)
    yen_surge = (pd.Timestamp("2022-08-01", tz="UTC"), pd.Timestamp("2022-11-01", tz="UTC"))
    carry_unwind = (pd.Timestamp("2024-07-25", tz="UTC"), pd.Timestamp("2024-08-10", tz="UTC"))
    print(f"\n  2022 yen-surge window {yen_surge[0].date()}..{yen_surge[1].date()} "
          f"in-sample: {gmin <= yen_surge[0] and gmax >= yen_surge[1]}")
    print(f"  2024 carry-unwind window {carry_unwind[0].date()}..{carry_unwind[1].date()} "
          f"in-sample: {gmin <= carry_unwind[0] and gmax >= carry_unwind[1]}")

    # ---- per-pair signed daily returns (SPOT) ----
    spot_rets = {}
    for p in PAIRS:
        ret = daily[p].pct_change()  # daily return of the cross (base per quote)
        spot_rets[p] = CARRY_DIR[p] * ret  # signed into carry direction

    # carry overlay: annual/252 per day, sign already baked into CARRY_ANNUAL_PCT (positive)
    carry_daily = {p: (CARRY_ANNUAL_PCT[p] / 100.0) / ANN for p in PAIRS}

    print("\n" + "-" * 78)
    print("PER-PAIR RESULTS  (SPOT-only  vs  SPOT+CARRY)")
    print("-" * 78)
    hdr = f"{'pair':9s} {'dir':>3s} {'carry%/y':>8s} | " \
          f"{'SPOT tot':>9s} {'ann':>7s} {'Shrp':>6s} {'maxDD':>7s} {'uw%':>5s} | " \
          f"{'+C tot':>9s} {'ann':>7s} {'Shrp':>6s} {'maxDD':>7s}"
    print(hdr)
    per_pair = {}
    for p in PAIRS:
        s = spot_rets[p]
        sc = s + carry_daily[p]  # spot + carry accrual (carry earned every day held)
        st = perf_stats(s)
        sct = perf_stats(sc)
        per_pair[p] = {"spot": st, "spotcarry": sct}
        print(f"{p:9s} {CARRY_DIR[p]:+3d} {CARRY_ANNUAL_PCT[p]:8.1f} | "
              f"{st['total']*100:8.1f}% {st['ann_ret']*100:6.1f}% {st['sharpe']:6.2f} "
              f"{st['maxdd']*100:6.1f}% {st['underwater']*100:4.0f}% | "
              f"{sct['total']*100:8.1f}% {sct['ann_ret']*100:6.1f}% {sct['sharpe']:6.2f} "
              f"{sct['maxdd']*100:6.1f}%")

    # ---- equal-weight basket ----
    spot_df = pd.DataFrame(spot_rets).dropna(how="all")
    basket_spot = spot_df.mean(axis=1)  # equal weight, daily rebalance
    carry_vec = pd.Series(carry_daily)
    basket_carry_accrual = (spot_df.notna() * carry_vec).sum(axis=1) / spot_df.notna().sum(axis=1)
    basket_spotcarry = basket_spot + basket_carry_accrual

    bs = perf_stats(basket_spot)
    bsc = perf_stats(basket_spotcarry)

    print("\n" + "=" * 78)
    print("EQUAL-WEIGHT BASKET (12 pairs, daily rebalance)")
    print("=" * 78)
    for name, st in [("SPOT-only ", bs), ("SPOT+CARRY", bsc)]:
        print(f"  {name}: total={st['total']*100:7.1f}%  "
              f"ann={st['ann_ret']*100:6.2f}%  sharpe={st['sharpe']:5.2f}  "
              f"maxDD={st['maxdd']*100:6.1f}%  (worst {st['maxdd_date'].date()})  "
              f"underwater={st['underwater']*100:4.1f}%")

    # ---- locate the worst drawdown & overlap with carry-stress windows ----
    print("\n" + "-" * 78)
    print("DRAWDOWN FORENSICS (basket SPOT-only)")
    print("-" * 78)
    dd = bs["dd"]
    print(f"  Worst drawdown {bs['maxdd']*100:.1f}% bottomed {bs['maxdd_date'].date()}")
    # worst single-day basket loss
    worst_day = basket_spot.idxmin()
    print(f"  Worst single DAY: {worst_day.date()} = {basket_spot.min()*100:.2f}%")
    print(f"  Top-8 worst basket days:")
    for d, v in basket_spot.nsmallest(8).items():
        tag = ""
        if carry_unwind[0] <= d <= carry_unwind[1]:
            tag = "  <-- 2024 CARRY UNWIND"
        elif yen_surge[0] <= d <= yen_surge[1]:
            tag = "  <-- 2022 yen surge"
        print(f"    {d.date()}  {v*100:6.2f}%{tag}")

    # behaviour during the two stress windows (JPY-cross sub-basket, all +1)
    jpy_pairs = [p for p in PAIRS if p.endswith("_JPY")]
    jpy_spot = spot_df[jpy_pairs].mean(axis=1)
    for label, (a, b) in [("2022 yen surge", yen_surge), ("2024 carry unwind", carry_unwind)]:
        win = jpy_spot.loc[a:b]
        if len(win):
            cum = (1 + win).prod() - 1
            print(f"  JPY-cross spot return over {label} "
                  f"({a.date()}..{b.date()}): {cum*100:+.2f}%  "
                  f"(min day {win.min()*100:+.2f}%)")

    print("\n" + "-" * 78)
    print("LIMITATIONS & FULL-VERSION SCOPE")
    print("-" * 78)
    print("""  This first pass is intentionally crude:
  * Carry yield is HARDCODED era-blended estimate, NOT real OANDA financing.
    The full version must pull v20 financing/swap rates per instrument
    (long & short legs differ; OANDA charges asymmetric swap).
  * STATIC direction per pair. AUD_USD/NZD_USD/EUR_USD flipped sign mid-era;
    a static sign mis-times those. Full version = DYNAMIC yield-ranked basket
    (rank pairs by live differential each rebalance, long top-K short bottom-K).
  * Equal-weight, daily rebalance, no vol-targeting. Real carry books risk-weight
    (inverse-vol) and rebalance weekly/monthly to cut turnover.
  * No tail/crash modelling. Carry's signature is exactly the left tail seen here
    (2024-08). Full version must size for the crash (vol-target, crash hedge,
    or regime de-risk on rate-vol / risk-off spikes).
  * Transaction cost ~ negligible for multi-week holds but NOT zero on rebalance;
    full version should net real OANDA spread on each rebalance trade.
  * Spot here uses daily-close mid; financing on OANDA accrues on position at
    5pm NY (triple Wednesday). Full version must accrue on the real swap schedule.
""")


if __name__ == "__main__":
    main()
