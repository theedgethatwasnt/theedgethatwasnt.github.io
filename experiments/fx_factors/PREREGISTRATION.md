# FX Factor Suite (Monthly Horizon) — Pre-Registration (LOCKED 2026-07-07)

> The academic FX risk premia — never tested in this project (Appendix D §5 carry-as-primary,
> §2 cross-asset, §7 long horizons). Long-horizon (the only surviving regime), not price-derived
> (the exhausted zone). 10-rule SOP; R10 null = random portfolio weights on identical rebalances.

## Data (fixed)
- 12 pairs, D1 bars aggregated from data/m5_ba (2020-11-11 → 2026-05-21) on the broker grid;
  extended-history context from data/cross_asset (2003→) is GATE data only, never signal.
- Carry: research/experiments/multiday_contrarian/carry_model.py (broker-truth pinch + FRED scaling).
- Value: OECD/World-Bank PPP annual rates (fetch, cache, commit); staleness-lagged 6 months (publication lag — no lookahead).
- **IS = first 70% (→ ≈ 2024-08-30). OOS = final 30%, SEALED, one shot after user gate.**
- Disclosed limitation: ~66 monthly rebalances total, ~46 IS — small-N is inherent to the horizon;
  CIs will be wide; the literature prior carries part of the burden (stated, not hidden).

## Portfolio construction (locked)
Monthly rebalance (last trading day close, execute next D1 open, mid ± half logged spread; carry
accrued daily per position; spread ×{1.0, 1.5} sensitivity). Cross-sectional rank over the 8
currencies (via the 12 pairs, net currency exposures), long top-3 / short bottom-3 currencies,
equal risk (positions scaled by 63-day realized vol), expressed through the most liquid pairs.

## Factors
1. **CARRY (primary):** rank by broker-truth net carry (long high-yield, short low-yield).
2. Momentum (secondary): rank by 12-month minus 1-month currency return.
3. Value (secondary): rank by real-exchange-rate deviation from PPP (cheap = long).
4. Equal-weight composite (secondary).

## Risk-off gate (single pre-declared rule, no search)
Flat the carry portfolio when SPX500 D1 close < its SMA(200) at rebalance (data/cross_asset).
Carry reported both gated and ungated; the GATED version is the confirmatory cell.

## H1 (confirmatory, one cell)
Gated carry, OOS: (1) net expectancy (spread+carry-inclusive) > 0 with monthly-block bootstrap
95% CI excluding 0; (2) beats the R10 null (200 random-weight portfolios, identical rebalance
dates and gross exposure) at the 95th percentile of the null's net distribution.

## Gates before OOS (IS-only)
1. Harness self-test (random-weight portfolios ≈ −costs on IS).
2. Carry-accrual parity: portfolio carry vs carry_model per-position sums (±5%).
3. Gated carry IS: net > 0 AND > null 95th pct.
4. WF: IS halves both net-positive (thirds too thin at this N — halves, disclosed).
5. User gate; then OOS once (UNSEAL).

## Decision rule
Pass → paper program (monthly, tiny size) ≥ 6 months before live discussion. Fail → closed;
factors recorded; OOS stays sealed for one amended shot. Momentum/value/composite never promoted
without their own confirmation.
