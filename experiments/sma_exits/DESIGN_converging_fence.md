# 010 Converging-Fence Robustness — Experiment Design (2026-06-18)

**Goal.** Close the risk gap between entry and the +20p PSAR arm (and, for the SL pairs,
the trade's whole pre-TP life) by replacing the flat 200p catastrophe fence with a stop
that tightens progressively. Success criterion (user): **show the p/d-vs-MaxDD frontier
first**, then pick a point.

**Scope.** All 4 live pairs, two regimes:
- PSAR pairs (EUR_USD, GBP_USD): converging fence → smooth handoff to PSAR at +20p.
- SL pairs (EUR_JPY, USD_JPY): converging fence that locks toward breakeven by +20p
  (progressive protection they completely lack today).

**Mechanism families on the frontier (all four + baseline):**
1. **flat** (baseline = current 200p) and tighter flat fences {120,80,50} = the *step* /
   early-SL family.
2. **psar-from-entry**: lower the PSAR arm threshold `act ∈ {0,5,10}` (PSAR pairs only).
3. **converge-MFE**: `d(mfe) = H + (F0−H)·(1 − clamp(mfe/20,0,1))^γ`, γ∈{0.5,1,2,4}, H∈{5,30}.
4. **converge-time**: same curve but driven by bars-in-trade vs a horizon T∈{8h,24h}.

Stop is entry-anchored and ratchets (mfe and age are monotonic). Once armed, PSAR coexists
(usually tighter). One kernel, mechanism selected by code — matches the live exit path (R6/R7).

**Harness.** Extend the H17 backtest: `prep()` (signals cached once per pair) + a unified
`kernx`. ~9.6mo S5, net of per-pair spread (IS-only gate, R5).

**Frontier axes.** y = OOS p/d, x = OOS MaxDD (and worst single-trade net), annotated with
WR, trades, tail%, and the realizable-under-margin worst-case $ at current NAV
(validation-gap lesson). Per-config portfolio of the 4 pairs; baseline marked.

**Validation.** Frontier is exploratory. The chosen point then gets the full
IS→OOS→WF→MC gate + a live-vs-backtest consistency check before any deploy (R7/R8 — OOS
sealed until then).
