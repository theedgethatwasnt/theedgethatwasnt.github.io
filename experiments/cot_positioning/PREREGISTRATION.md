# COT Contrarian Positioning (Axis 3, Isolated) — Pre-Registration (LOCKED 2026-07-07)

> The book's own Composite-1 roadmap, step 1 (Appendix E): "Prototype Axis 3 first, in isolation,
> on free data with deep history... the decisive, cheapest test." Extra-urgent since the OANDA
> position book is entitlement-blocked (2026-07-07 probe) — COT is the only positioning leg
> testable today. 10-rule SOP; R10 null on identical rebalances.

## Data (fixed)
- **CFTC Commitments of Traders**, Legacy report, futures-only: weekly net non-commercial
  (speculator) positions for the CME currency futures EUR, JPY, GBP, CHF, AUD, CAD, NZD (USD
  implicit as the base leg), from the free annual archives (deacot files), full available history.
  **Release-lag alignment: positions are as-of Tuesday, published Friday ~15:30 ET — signals may
  act no earlier than the following Monday's open (no lookahead through the publication lag).**
- Prices: OANDA D1 mid candles, 12 pairs, deep history (max fetch, ≈2005→2026-05-21 CEILING —
  nothing after 2026-05-21 is loaded anywhere, matching the sibling experiments' data edge),
  fetched via the curator path. Spread: per-pair typical D1 spread from our measured medians
  (documented; sensitivity ×{1.0, 1.5}). Carry: carry_model where valid (2020→), flat FRED
  differential earlier (R9-documented approximation).
- **IS = first 70% of the joint COT×price window. OOS = final 30%, SEALED, one shot after user gate.**

## Signal (locked, no sweeps)
Weekly, on the first trading day after each COT release: per currency, z-score of net
non-commercial position (scaled by open interest) over a trailing 156-week window.
**Contrarian: short the top-2 most crowded-long currencies, long the bottom-2 most crowded-short**,
expressed via the most liquid USD pairs, equal risk (63-day realized-vol scaling), held one week
to the next release. Single construction: z-window 156w, top/bottom-2 — no sweeps.

## Arms (identical rebalance dates)
Contrarian (primary) · momentum-with-crowd (ordering check) · R10 null = 200 random-sign
portfolios, identical dates and gross exposure.

## H1 (confirmatory)
Contrarian arm OOS: net expectancy (spread+carry) > 0, weekly-block bootstrap 95% CI excluding 0,
AND beats the null's 95th percentile.

## Gates before OOS (IS-only)
1. Data integrity: COT weekly continuity ≥95%; publication-lag alignment test (synthetic future-leak tripwire).
2. RW/null self-test (random portfolios ≈ −costs).
3. Contrarian IS: net > 0, > null 95th pct, > momentum arm.
4. WF: IS thirds ≥ 2/3 net-positive.
5. User gate → OOS once (typed UNSEAL).

## Decision rule
Pass → Composite 1 gains a validated second vote; next step per the book = combine with the
Axis-1 fade under a fresh pre-registration. Fail → Composite 1 loses its only testable
independent leg (reduces toward Composite 3); recorded in Appendix E as run-and-closed either way.
