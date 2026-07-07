# Composite 1 — Axis-1 Displacement Fade × Axis-3 COT Positioning (LOCKED 2026-07-07)

> The book's Appendix-E recommendation, step 2 — executed after step 1 (COT isolated: 4/4 IS
> gates, +1.01p/wk > null p95, but CI spans zero; seal HELD by user ruling for this test).
> The composite thesis: two individually-thin INDEPENDENT directional votes, stacked at a
> horizon where the toll is rounding error, clear what neither clears alone. 10-rule SOP.

## Inherited seal (binding)
This experiment inherits cot_positioning's IS/OOS boundary so the held seal is spent exactly
once, here: **IS = joint window through 2021-01-31; OOS = 2021-02-01 → 2026-05-21, SEALED,
one shot after the user gate.** No component may be evaluated on the OOS segment beforehand.

## Data (fixed)
- 7 direct-USD pairs matching the COT currencies: EUR_USD, GBP_USD, AUD_USD, NZD_USD,
  USD_JPY, USD_CHF, USD_CAD — D1 bid/ask `data/d1_deep_ba/` (2002→2026-05-21 ceiling).
- COT weekly `research/experiments/cot_positioning/cot_weekly.parquet`, release-lag rule
  inherited verbatim (act no earlier than the first trading day after Friday publication).
- Carry: carry_model 2020+, FRED flat-differential earlier (R9 splice as before).

## Axis-1 signal (structure-transfer, declared — parameters translated from the deployed
## first-touch config to the D1 grid, scale-free; NOT fitted to any data)
Fresh rolling swing extreme: highest high / lowest low of the prior L=25 completed D1 bars.
Touch tolerance EPS = 0.25 × ATR(14,D1) at the touch. First touch only. Volume gate: touch-bar
tick-volume < mean of the prior VW=20 D1 bars (low-volume only). Direction: fade the extreme.
Note for the record: the H4-grid version of this family was REFUTED (Entry 72); the D1 grid is
untested; Axis-1 alone is expected thin-to-null and serves as a measured arm, not a hope.

## Axis-3 vote (inherited verbatim from cot_positioning)
Currency-level contrarian crowding: 156-week z-score of net non-commercial position scaled by
open interest, as of the most recent RELEASED report. The vote for a pair trade = the crowding
of the faded currency: fading a swing HIGH of pair base (going short base) requires base-currency
crowding z ≥ +1.0 (crowd long the thing being faded); fading a LOW (long base) requires z ≤ −1.0.
Single threshold ±1.0, declared, no sweep.

## Composite rule
Take the Axis-1 fade ONLY when the Axis-3 vote agrees. Exits (frozen, from the book's Composite-1
prescription, structure-transfer): TP = 2 × ATR(14,D1) at entry; broker-side stop = 4 × ATR
(wide, bounded); time cap = 10 D1 bars, exit at close. One position per pair, FIFO. Entry next
D1 open after the signal bar (mid ± half real spread). Sizing: fixed 1-unit risk-parity
(fractional-Kelly is a later sizing layer, out of scope — declared).

## Arms (identical machinery; R10)
1. **Composite** (primary)
2. Axis-1 alone (all first-touches, no positioning gate)
3. Axis-3 alone (the cot_positioning weekly portfolio, for reference on this window)
4. Coin-flip direction on composite timestamps (seed 20260709)
5. **Shuffled-positioning control** (the decisive one): Axis-1 gated by a randomly permuted
   COT vote series (200 permutations, block-preserving by currency) — the composite must beat
   this distribution's 95th percentile, proving the COT vote adds information, not just
   trade-count reduction.

## H1 (confirmatory, one cell — spends the held seal)
Composite arm OOS: (1) net expectancy (real spread + carry) > 0, day-block bootstrap 95% CI
excluding 0; (2) > coin arm, CI excluding 0; (3) > the shuffled-positioning null's 95th pct.
All three. One shot.

## Gates before OOS (IS-only)
1. RW self-test (coin ≈ −costs, no phantom edge).
2. Component sanity: Axis-3-alone IS reproduces cot_positioning's IS numbers on the shared
   window (same code, parity ±0.1p/wk).
3. Composite IS: all three H1 criteria on IS.
4. WF: IS thirds net-positive ≥ 2/3.
5. Breadth: ≥ 4/7 pairs gross-positive IS.
6. Trade count: composite ≥ 150 IS trades (else underpowered — stop, report, seal intact).
7. **User gate** → typed UNSEAL, once.

## Decision rule
- H1 passes → paper program pre-registration (tiny size, ≥ 6 months) — never straight to live.
- Any IS gate fails → composite closed; seal stays intact (unspent) for one future amended shot;
  Appendix E updated: the book's recommendation was executed to its end and the answer recorded.
- The shuffled-positioning control failing while raw net passes = the vote adds nothing:
  recorded as "selectivity in disguise" (the book's own Axis-4 lesson) — closed.
