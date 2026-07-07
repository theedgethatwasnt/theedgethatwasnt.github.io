# Phase A Pre-Registration (LOCKED 2026-07-06)

## Global Constraints

- **Pre-registration (AMENDED 2026-07-06, locked before any data analysis — user decisions):**
  - **Primary (confirmatory) arm: AGAINST-DRIFT** (fade prior-bar drift), amended from the doc's with-drift per 7 negative priors (`quiet_drift`, `gbpusd_regime`, `rolling_amddp`, `stage1_momentum_efficiency`, S5-burst, `oracle_traits`, `microstructure_study`) and all gross-positive precedent being contrarian (`structural_fade`, 010 exit sweep). With-drift and coin-flip run as controls on identical timestamps.
  - **TP levels: {1.5, 1.8, 3.2, 4.0} pips** (ECN-anchored 2.0×/2.5× of 0.75p; OANDA-measured 2.0×/2.5× of 1.6p). **Confirmatory test at TP = 3.2p**, SL = 2×TP = 6.4p, horizon = 2 M5 bars. All other combos pre-declared secondary/exploratory.
  - **Zigzag legs: plain alternating extrema** (TopsBots Stage 1+2 pattern, no Stage-3 gate) on S5 mid-closes within the closed window; window endpoints included as leg boundaries so legs tile the whole path.
  - **ER buckets: terciles fixed on the IS (first 70%) portion only.** Primary bucket = top tercile.
  - **VR: k = 4** (20-second aggregation); k ∈ {2, 8} exploratory.
  - **Primary pair: EUR_USD** (cost analysis anchor). All 12 pairs run identically as breadth robustness (per-pair pip size; JPY pairs pip = 0.01).
  - **OOS: final 30% by time per pair, evaluated ONCE, only after IS/WF review gate with the user (R8).**
- Success criterion (amended §15): against-drift entries in the top-ER tercile clear cost-adjusted p\* OOS with block-bootstrap CI excluding p\*.
- Every result table reports win rates relative to 66.7% and to each broker's p\* — never absolute-only (§8).
- Labels on **mid-price, absolute pips** (§24.1). Spreads logged per signal, never consumed into labels (§24.2).
- Same-S5-bar barrier ambiguity resolved conservatively: **SL checked before TP**.
- Memory rule: load ONE pair at a time, `del` + `gc.collect()` between pairs; never two heavy jobs concurrently (`feedback_no_concurrent_backtests`).
- S5 gaps: feature windows forward-fill closes onto the 60-slot grid; require ≥30 real bars in the window else skip signal (skip reason logged). Bar density logged per signal.
- Entry fill: close of the first S5 bar with timestamp in [M5close+5s, M5close+30s]; skip signal if none. Log close→entry drift (fill − prior M5 close) per trade (§9).
- Commit + push at every task boundary (`feedback_commit_push_often`).
- Timeout (label 0) trades are their own P&L bucket everywhere (§11).
- Sessions (locked): Asia 22:00–07:00 UTC, London 07:00–12:00, NY 12:00–21:00, Other 21:00–22:00 (by entry timestamp).

## Amendment record (before any data analysis)
- Primary arm amended with-drift → AGAINST-DRIFT. Rationale: 7 direct negative
  priors on with-drift/high-ER continuation (quiet_drift; gbpusd_regime 46-48%
  continuation; rolling_amddp signed-eff contrarian at 10min; stage1_momentum_
  efficiency all-cells-negative with high-eff slight reversal; s5_burst -4.06
  p/trade WR 11.6%; oracle_traits optimal legs are messy ER~0.57; microstructure
  ER-family = magnitude only). All gross-positive precedent is contrarian
  (structural_fade OOS gross +1.089 12/12 MC p=0; 010 TP100/SL200 fade
  break-even at OANDA). Amended BEFORE any Phase A data was touched.
- TP pip values frozen: {1.5, 1.8, 3.2, 4.0}; confirmatory 3.2 (=2.0x measured
  OANDA EUR_USD median spread 1.6p; doc §9 worked example ~matches). ECN-anchored
  {1.5, 1.8} per §24.1 retained as secondary.
- Zigzag leg extraction (doc gap): plain alternating extrema on S5 mid-closes,
  endpoints as leg boundaries. TopsBots Stage 1+2 lineage, no Stage-3 gate.
- Measured planning inputs: OANDA EUR_USD S5 spread p10/p50/p90 = 1.4/1.6/1.7p.

## Hypothesis (confirmatory, one test)
H1: against-drift entries in the top-ER tercile (IS-fixed thresholds), TP=3.2p
SL=6.4p horizon=2 bars, on EUR_USD, achieve OOS win rate whose day-block-
bootstrap 95% CI lies above the ECN cost-adjusted break-even p* (flat 0.7p,
pessimistic 1.5x sensitivity reported).
Decision rule per doc §22: gross null => stop everything; fails ECN p* => stop
or redesign exit; clears ECN only => Phase C nomination, NOT confirmation.

## Amendment 2 (2026-07-06, after Task 4 smoke run, BEFORE any conditioned cell was viewed)

**Trigger.** The Task 4 real-data smoke run (unconditioned aggregates only) + synthetic
random-walk controls showed that at the 2-bar vertical barrier only ~20% of trades
reach a price barrier, and conditioning on barrier-resolution inflates decided-trade
win share to ~82-85% ON PURE RANDOM WALKS (verified by simulation, shuffle control,
and direction symmetry). The 66.7% gambler's-ruin baseline holds only for unbounded
horizons; at a fixed vertical it is NOT the correct null.

**Amended test statistic (H1 restated).** The confirmatory cell (against-drift,
top-ER tercile, TP=3.2p SL=6.4p h=2, EUR_USD) passes iff BOTH, evaluated once OOS:
1. **Net expectancy at ECN cost (0.7p, and reported at 1.5x) > 0**, day-block
   bootstrap 95% CI excluding 0 — the money criterion, timeouts included.
2. **Decided-WR effect vs the coin-flip arm on identical timestamps > 0**, day-block
   bootstrap 95% CI excluding 0 — the information criterion. The coin arm is the
   operative null; p* and 66.7% are still reported for continuity but are no longer
   sufficient alone (at 80% timeout share the decided subset does not carry P&L).

**What was seen before this amendment:** unconditioned smoke aggregates (signal count,
feature ranges, label mix, pooled decided win-share both directions) and synthetic
simulations. No ER-bucket, arm-contrast, or session-conditioned cell was computed.

## R9 divergence log (documented, from Task 4 review)

- Feature-window grid seeding backward-fills empty slots BEFORE the first real bar in
  the window from that (past) bar's close — no future data, but flattens early slots
  on sparse windows, biasing drift/rv toward 0 in quiet periods. n_real_bars logged
  per signal; robustness slice by density available.
- Labeler horizon end includes a bar stamped exactly at t+h*300 (bar covers up to 5s
  past the nominal horizon). Direction of bias: negligible, symmetric.
- Entry fill = close of the first S5 bar in [t+5s, t+30s] (~t=10s typical), ~4s later
  than the doc's t=6s: pessimistic (worse fill), conservative.

## Phase A VERDICT (2026-07-06) — CLOSED IS-NEGATIVE on the money criterion; OOS NEVER UNSEALED

IS results (first 70%, 12 pairs, confirmatory cell against-drift/hiER/t32/h2):
- **Information criterion: PASS (gross).** against > coin gross expectancy on 12/12
  pairs (+0.01..+0.20 p/trade delta); with-drift worst 12/12. Arm ordering exactly as
  the amended pre-registration predicted. The regime vector carries real, contrarian,
  broad directional information at the 10-minute horizon.
- **Money criterion: FAIL. 0/12 pairs net-positive at ECN 1.0x** (best −0.57 p/trade);
  gross tilt is 5-10x below the 0.7p ECN floor, ~20x below OANDA. Doc §22 weak-positive
  branch → STOP. Decided-WR deltas mixed (7/12); the signal lives mostly in timeout-
  bucket drift, per Amendment 2's expectation.
- **User decision: OOS (final 30%) remains SEALED** — preserved for one future
  confirmatory shot at a redesigned exit structure (wider TP / passive ECN entry,
  doc §16/Phase C route). H1 was unfalsifiable-in-the-positive given the IS net gap;
  unsealing would have spent the seal on a foregone failure.
- Settles the two previously-open ECN branches by triangulation: signals of this
  gross size (structural_fade +1.09 is the ceiling of the family) do not clear even
  raw-ECN costs at short horizons. Phase B re-pricing of harvester/010-fade remains
  possible on their own data but expectations are now firmly negative.
- Artifacts: signals/*.parquet (12 pairs, ~1.2M signals), report_is_*.md x12,
  synth_is_confirmatory.csv, synth_is.py.
