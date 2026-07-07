# London-Fix Fade — Pre-Registration (LOCKED 2026-07-07)

> The corpus's own flagged-untested lead (indicator screen memory: "FX-fix session-fade").
> WM/R 4pm-London fix: documented institutional rebalancing flow — pre-fix pressure,
> post-fix reversion. Intraday, so the spread wall is the expected killer: stated up front.

## Data / window
12 pairs, M5 BA (2020-11-11 → 2026-05-21). Fix time = 16:00 London (DST-aware via Europe/London),
i.e. the 15:55-16:00 UTC-variable M5 bar. IS first 70%, OOS final 30% SEALED, one shot.

## Rule (locked, no sweeps)
Signal day-filter: month-end amplification reported as a pre-declared split (last trading day vs rest), not searched.
Pre-fix drift D = mid(16:00 Ldn) − mid(15:00 Ldn). Enter AGAINST D at the first M5 open after the
fix bar closes, |D| ≥ 5 pips required (single threshold). Exit: 60 minutes later at close (single
horizon) — no TP/SL (bounded by the 1-hour cap; per-trade excursion reported).
Arms: fade / coin (seed 20260708) / with-drift continuation — identical timestamps (R10).

## H1 (confirmatory)
Fade arm OOS: net (real per-trade spread) > 0, day-block bootstrap CI excluding 0, AND beats coin
arm CI. Expected outcome per priors: gross reversion real, net ≈ −spread — a clean negative is
the anticipated and acceptable result; it formally closes the corpus's last flagged lead.

## Gates
RW self-test; IS net>0 & >coin; WF thirds ≥2/3; breadth ≥6/12 gross; user gate; OOS once.
