# COT Contrarian Positioning — IS Battery Summary

Governed by `PREREGISTRATION.md` (LOCKED 2026-07-07). IS-only: 641 weeks, 2008-10-25 00:00:00+00:00 -> 2021-01-31 00:00:00+00:00 (70.0% of the joint COT×price window). OOS is SEALED — never read; every loader routes through `is_data.restrict_sched_to_is()` with an independent re-assertion.

**Units note**: all pip figures below are EQUAL-RISK-WEIGHTED portfolio pips — each week's 4 active legs (top-2 crowded-long + bottom-2 crowded-short currencies, expressed via their direct USD pair) are weighted 1/vol_i normalized to sum to 1 (63-day realized vol), then summed. This is a weighted average across differently-scaled pairs (JPY vs non-JPY, majors vs minors), not a raw single-pair pip series — not directly comparable to the sibling experiments' per-trade pip conventions.

## Data coverage

| Source | Detail |
|---|---|
| COT (CFTC legacy futures-only) | 7811 weekly rows, 7 currencies (AUD, CAD, CHF, EUR, GBP, JPY, NZD), 2005-01-04 -> 2026-06-30 |
| D1 price (OANDA, 7 direct-USD legs) | AUD_USD, EUR_USD, GBP_USD, NZD_USD, USD_CAD, USD_CHF, USD_JPY, ceiling 2026-05-21 |
| Full joint COT×price rebalance schedule | 916 weeks, 2008-10-25 00:00:00+00:00 -> 2026-05-10 00:00:00+00:00 |
| **IS window (first 70% by row count)** | **641 weeks**, 2008-10-25 00:00:00+00:00 -> 2021-01-31 00:00:00+00:00 |
| IS resolved / dropped | 641 resolved, 0 dropped (missing price/vol) |
| Spread (IS-only medians, pips, round-trip) | AUD_USD=3.80p, EUR_USD=2.70p, GBP_USD=5.00p, NZD_USD=5.20p, USD_CAD=5.00p, USD_CHF=4.00p, USD_JPY=3.20p |
| Null replicates | 200 (seeded, R10) |

## Gate table (IS-only, PREREGISTRATION.md "Gates before OOS")

| Gate | Name | Result | Detail |
|---|---|---|---|
| 1 | Data integrity (continuity>=95% + release-lag tripwire) | **PASS** | per-currency continuity: AUD=100.1%, CAD=100.1%, CHF=100.1%, EUR=100.1%, GBP=100.1%, JPY=100.1%, NZD=97.0%; release-lag tripwire = PASS (test_release_lag.py, 6/6) |
| 2 | RW/null self-test (random portfolios ~= -costs) | **PASS** | null overall mean=-4.14p/wk (replicate-to-replicate std=2.99p), vs signal scale 9.32p |
| 3 | Contrarian IS: net>0, >null 95th pct, >momentum | **PASS** | contrarian=+1.01p/wk, null p95=+0.41p/wk, momentum=-9.32p/wk |
| 4 | WF: IS thirds >=2/3 net-positive | **PASS** | n_pos=2/3, thirds=[3.831, -2.924, 2.093] |

**4/4 gates pass.** Gate 5 (user gate -> OOS unseal) is NOT reached by this run — OOS stays sealed regardless of the gate-4 outcome, per task scope (IS-only).

Supplementary (not a formal IS gate — the pre-registration only requires the weekly-block bootstrap for the OOS confirmatory H1 test; reported here on IS for context): weekly-block bootstrap on the contrarian arm, 2000 resamples: P(net<=0)=0.354, mean=+0.92p/wk, 95% CI=[-3.70p, +5.48p] — does NOT exclude zero.

## Contrarian vs momentum vs null (IS, net of spread+carry)

| Arm | mean net p/wk | std p/wk | maxDD (p) | n weeks |
|---|---|---|---|---|
| Contrarian (primary) | +1.01p | 58.57p | 1391.7p | 641 |
| Momentum-with-crowd (ordering check) | -9.32p | 58.58p | 6076.9p | 641 |
| Null (typical random-sign path, R10) | -4.14p | 4.99p | 2651.0p | 641 |

Null arm distribution across its 200 replicate portfolios (each replicate's own mean net p/wk across all 641 IS weeks): mean=-4.14p, p5=-9.37p, p50=-4.02p, p95=+0.41p.

### Spread sensitivity (documented, not gated — PREREGISTRATION.md "sensitivity ×{1.0, 1.5}")

| Arm | net @1.0x spread | net @1.5x spread |
|---|---|---|
| Contrarian | +1.01p | -1.05p |
| Momentum | -9.32p | -11.38p |

## Walk-forward (3 equal-duration IS thirds, contrarian arm)

| Third | Start | End | n | mean net p/wk |
|---|---|---|---|---|
| 1 | 2008-10-25 | 2012-11-26 | 214 | +3.83p |
| 2 | 2012-11-26 | 2016-12-29 | 213 | -2.92p |
| 3 | 2016-12-29 | 2021-01-31 | 214 | +2.09p |

## Verdict

Gates 1-2-3-4 = PASS/PASS/PASS/PASS (4/4). The contrarian arm clears its own pre-registered mechanical bar on IS: net +1.01p/week (spread+carry included), above both the null's 95th percentile (+0.41p/wk) and the momentum-with-crowd ordering check (-9.32p/wk), and positive in 2/3 walk-forward thirds.
The margin is THIN and cost-fragile: the weekly-block bootstrap 95% CI [-3.70p, +5.48p] does NOT exclude zero (P(net<=0)=0.35), and at 1.5x the measured spread the contrarian arm's IS mean flips to -1.05p/week — the edge lives almost entirely inside the spread cushion, not clear of it.
This is consistent with the codebase's recurring finding across 40+ closed experiments (JOURNEY-README.md, memory/MEMORY.md): directional FX signals rarely clear OANDA retail costs by a wide, stationary margin — here the mechanical gates pass, but the statistical margin is not decisively distinguishable from the null.
Per the pre-registration's decision rule, an IS gate PASS (1-4) would ordinarily unlock gate 5 (user gate -> OOS, typed UNSEAL) — that step is explicitly OUT OF SCOPE for this run; OOS remains sealed, untouched, for the user to decide whether the thin+fragile IS margin above warrants spending the one-shot OOS look.
No parameters were tuned in this run: z-window=156w, top/bottom=2, spread_mult=1.0, markup_mult=1.0 are the pre-registered frozen defaults; the 1.5x spread row is the pre-declared sensitivity check, not a search.
