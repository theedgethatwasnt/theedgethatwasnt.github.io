# Composite 1 — Axis-1 Displacement Fade × Axis-3 COT Positioning — IS Battery Summary

Governed by `PREREGISTRATION.md` (LOCKED 2026-07-07). Inherited seal from `cot_positioning`:
IS = joint window through 2021-01-31 (COT report_date < 2021-02-02; Axis-1/composite trade
**entries** strictly before 2021-02-01 UTC). OOS (2021-02-01 → 2026-05-21) is SEALED — never
loaded; every loader routes through `is_data.load_pair_is()` / `is_data.restrict_cot_to_is()`
with an independent post-hoc re-assertion (`is_data.assert_trade_is_is`).

## Data coverage

| Source | Detail |
|---|---|
| COT (CFTC legacy futures-only), IS-restricted | 5,830 weekly rows, 7 currencies, 2005-01-04 → 2021-01-26 |
| D1 price (OANDA, 7 direct-USD pairs), IS-loader-truncated | EUR_USD, GBP_USD, AUD_USD, NZD_USD, USD_JPY, USD_CHF, USD_CAD, load ceiling 2021-03-03 (entry cutoff 2021-02-01 + 30d buffer) |
| Axis-3 released z observations | 684 per currency (641 for NZD, later COT start), 2007-12-28 / 2008-10-25 → 2021-01-31 |
| Axis-1 raw signals (all 7 pairs, all-time incl. buffer) | 268 → 256 after FIFO → 254 IS (entry < 2021-02-01) |
| Composite (gated) IS trades | **54** (21.3% of the 254 Axis-1-alone IS trades pass the ±1.0 z-gate) |
| Shuffled-positioning replicates | 200 (seeded 20260709, block-preserving by currency) |

## Gate table (IS-only, task-brief ordering)

| Gate | Name | Result | Detail |
|---|---|---|---|
| 1 | RW self-test (coin ≈ −costs, no phantom edge) | **PASS** | pooled synthetic random-walk battery (test_rw_selftest.py, 3/3): coin arm not significantly positive at 2×SE; natural-direction arm indistinguishable from coin at 4×SE |
| 2 | Axis-3 parity vs cot_positioning IS (same window, same code) | **PASS** | fresh re-run of `cot_positioning/run_is_battery.py` (verbatim, unmodified) reproduces contrarian_mean = **+1.0063196573… p/wk EXACTLY** (abs diff = 0.0000, tolerance ±0.1p/wk), n=641 weeks |
| 3 | Composite IS: 3 H1 criteria (money / vs coin / vs shuffled-null p95) | **FAIL** | all three sub-criteria fail — see below |
| 4 | WF: IS thirds ≥2/3 net-positive | PASS | thirds = [+83.22p (n=18), +1.43p (n=20), **−16.92p** (n=16)] — 2/3 positive, but the most recent third is negative and n-per-third is small |
| 5 | Breadth: ≥4/7 pairs gross-positive | PASS | 4/7 (GBP_USD, NZD_USD, USD_JPY, USD_CHF positive; EUR_USD, AUD_USD, USD_CAD negative) — barely clears the minimum, per-pair n ranges 3–14 |
| 6 | Trade count: composite ≥150 IS trades | **FAIL** | **54 / 150 required (36% of the pre-registered minimum) — underpowered by the pre-registration's own explicit test** |

**4/6 gates pass, 2/6 fail** (gates 3 and 6 — including the primary substantive gate and the
power gate). Gate 7 (user gate → OOS unseal) is reached only if all prior gates pass; it is
**not** reached here.

### Gate 3 detail — composite IS three criteria (all FAIL)

| Criterion | Result | Detail |
|---|---|---|
| 1. Money: net>0, day-block bootstrap 95% CI excl. 0 | **FAIL** | mean = **+23.26p/trade**, CI = **[−43.29p, +87.66p]** — does not exclude zero |
| 2. vs coin-flip (same composite timestamps): net>coin, CI excl. 0 | **FAIL** | composite (+23.26p) is **worse** than its own coin-flip null (+41.90p) on the identical 54 signals — mean diff = **−18.64p**, CI = [−100.41p, +58.94p] |
| 3. vs shuffled-positioning null's 95th pct | **FAIL** | composite mean (+23.26p) sits **below** the null's own 95th percentile (+58.68p) — and only barely above the null's own **mean** (+21.25p) |

## 5-arm table (IS, net of real per-bar spread + carry)

| Arm | n | net mean (p/trade) | gross mean (p/trade) | net std | WR | maxDD (p) | Units |
|---|---|---|---|---|---|---|---|
| **Composite** (primary) | 54 | **+23.26** | +22.88 | 231.5 | 57.4% | 895.1 | pips/trade |
| Axis-1 alone (all first-touches, no gate) | 254 | +14.24 | +14.91 | 172.6 | 55.1% | 2128.7 | pips/trade |
| Axis-3 alone (cot_positioning contrarian, same shared window, reference) | 641 weeks | +1.01 | n/a | 58.6 | n/a | 1391.7 | **pips/WEEK, portfolio-weighted — not comparable unit-for-unit to the per-trade rows above** |
| Coin-flip (composite timestamps, seed 20260709) | 54 | +41.90 | +42.83 | 215.5 | 57.4% | 834.7 | pips/trade |
| Shuffled-positioning null (200 reps, mean/p5/p50/p95) | avg n=41.4/rep | **mean +21.25 / p5 −18.17 / p50 +21.04 / p95 +58.68** | — | — | — | — | pips/trade (per-replicate mean) |

Composite exit-reason mix: 28 timecap (52%), 21 TP (39%), 5 SL (9%) — over half of composite
trades never resolve on the 2×ATR/4×ATR barriers at all within the 10-D1-bar cap, consistent
with the very high per-trade variance (net std ≈230p against a mean of +23p — a mean/SE ratio
that is not statistically distinguishable from noise, matching the CI-spans-zero result above).

## Walk-forward (3 equal-duration IS thirds, composite arm)

| Third | Start | End | n | mean net (p) |
|---|---|---|---|---|
| 1 | 2008-09-01 | 2012-10-22 | 18 | +83.22 |
| 2 | 2012-10-22 | 2016-12-12 | 20 | +1.43 |
| 3 | 2016-12-12 | 2021-02-01 | 16 | −16.92 |

## Per-pair composite (IS, gross mean pips/trade, n)

| Pair | n | gross mean | net mean |
|---|---|---|---|
| EUR_USD | 8 | −121.28 | −121.77 |
| GBP_USD | 8 | **+232.29** | +233.15 |
| AUD_USD | 8 | −39.30 | −35.92 |
| NZD_USD | 8 | +10.89 | +9.44 |
| USD_JPY | 14 | +16.04 | +15.92 |
| USD_CHF | 5 | +97.65 | +98.16 |
| USD_CAD | 3 | −45.98 | −45.59 |

Breadth passes 4/7 only because two pairs (GBP_USD n=8, USD_CHF n=5) carry very large
per-trade swings on very small samples — this is the same high-variance/low-power pattern
that fails gates 3 and 6, not independent confirmation.

## Shuffled-positioning-null distribution (the decisive control, 200 block-preserving-by-currency permutations)

mean = +21.25p · p5 = −18.17p · p50 = +21.04p · **p95 = +58.68p** · avg n/replicate = 41.4

The composite's actual mean (+23.26p, n=54) is statistically indistinguishable from this
null's own central tendency (mean +21.25p / median +21.04p) and sits well inside its bulk,
far short of the null's 95th percentile. A randomly-permuted COT vote — which by
construction carries **zero true information** about the currency's actual crowding at the
time of each Axis-1 touch — produces composite-style subsets that perform, on average, about
as well as the real gate does.

## Verdict

**Composite 1 fails on IS.** Two of six gates fail outright — gate 6 (trade count: 54 vs the
pre-registered minimum of 150, only 36% power) and gate 3 (all three H1 criteria: the money
CI spans zero, the composite arm is **worse** than a coin flip on its own 54 timestamps, and
it sits below the shuffled-positioning null's 95th percentile). Gate 3's third criterion is
the pre-registration's own named decisive test, and it fails in the specific way the
pre-registration flagged as the worst-case interpretation ("selectivity in disguise") — except
here the composite doesn't even clear the null's mean by a meaningful margin, and it actively
underperforms its own coin-flip control, so this reads as noise rather than "selectivity
without information." Gates 4 and 5 pass only mechanically, on tiny per-third/per-pair samples
(16–20 trades/third, 3–14 trades/pair) dominated by a few large-payout outliers (GBP_USD
+232p/trade on n=8), not evidence of a robust, generalizing effect. Gate 2 (Axis-3 parity)
reproduces cot_positioning's IS number **exactly** (diff=0.0000p/wk), confirming the harness
correctly reuses the verbatim COT code — the failure is not a plumbing/parity bug, it is the
composite construction itself. **Recommendation: do NOT type UNSEAL.** Per the
pre-registration's decision rule, an IS gate failure closes Composite 1 with the seal intact
and unspent, available for one future amended shot (different Axis-1 parameters or a coarser
gate) rather than being spent on a battery this underpowered and this clearly non-selective;
the book's Appendix-E recommendation has been executed to its end on this construction and the
answer — negative — is recorded here.
