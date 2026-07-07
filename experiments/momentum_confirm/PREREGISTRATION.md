# Monthly FX Cross-Sectional Momentum — Confirmation (LOCKED 2026-07-07)

> Origin: the fx_factors sweep's lone positive SECONDARY (+27.4 p/rebalance IS vs null p95 +15.5,
> single pass, 2020-11→2024-08). Per that pre-registration, secondaries prove nothing without
> their own confirmation. This is that confirmation — on data the observation never touched.
> Philosophical stake: monthly momentum surviving where all intraday momentum died would be
> the toll-amortization law's cleanest positive expression.

## The self-grading trap, and the design that avoids it
The +27.4 was observed on 2020-11→2024-08. Testing the same rule on the same window proves
nothing. Confirmation therefore uses two DISJOINT segments:
- **Stage 1 (backward out-of-sample):** OANDA D1 deep history ≈2005→2020-10 — never seen by the
  factor sweep or any prior experiment here. Full gate battery runs HERE.
- **Stage 2 (the sealed one-shot):** the fx_factors sealed OOS (2024-08-30→2026-05-21), used as
  that pre-registration's allowed "one amended shot" — opened ONLY if Stage 1 passes all gates,
  ONLY after the user gate, evaluated once.
The observation window (2020-11→2024-08) is quarantined: reported for continuity, never gated on.

## Rule (FROZEN verbatim from research/experiments/fx_factors/factors.py as run — copied, not refitted)
Monthly rebalance (last trading day close → next D1 open, mid ± half spread): per currency,
12-month return minus 1-month return (via the currency index construction of fx_factors);
long top-3 / short bottom-3 currencies, 63-day realized-vol scaling, most-liquid-pair expression.
No parameter may differ from the sweep's implementation — same code, new data.

## Costs
Per-pair spread as fx_factors (measured medians; ×{1.0, 1.5}); carry: carry_model 2020+,
FRED-differential flat approximation pre-2020 (R9: documented splice, direction of bias unknown,
mitigated by reporting carry-free gross alongside).

## Null (R10)
200 random-weight portfolios, identical rebalance dates and gross exposure, per segment.

## H1 (confirmatory)
Stage 2 (sealed window, one shot): net expectancy > 0 with monthly-block bootstrap 95% CI
excluding 0 AND > null 95th percentile.

## Stage-1 gates (all required before Stage 2 may be requested)
1. Apparatus self-test on the deep segment (null ≈ −costs).
2. Deep-segment net > 0 AND > null 95th pct (~180 rebalances 2005-2020 — real statistical power
   for once).
3. WF: deep segment thirds, net-positive ≥ 2/3, none < −40 p/rebalance.
4. Regime robustness (pre-declared split, no search): positive in at least one of the two halves
   split at 2013 AND not catastrophic (< −40) in the other.
5. **User gate** reviewing Stage-1 results → typed UNSEAL for Stage 2.

## Decision rule
- Stage 1 fails → momentum recorded as sweep-artifact; fx_factors OOS stays sealed (unspent).
- Stage 1 passes + Stage 2 passes → first confirmed positive of the post-book era; next step is
  a paper program pre-registration (monthly, tiny size), NOT live.
- Stage 1 passes + Stage 2 fails → recorded as regime-bound (worked 2005-2020, gone now); closed.
