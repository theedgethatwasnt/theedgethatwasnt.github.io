# TradingView Technical Ratings as a tradeable metric — campaign plan

**Goal:** treat TradingView's indicator/timeframe assembly (the "Technicals" gauges) as a
*metric*, recreate it causally, and test whether it carries an exploitable edge net of spread —
without brute-forcing an astronomical search space.

## 1. What the metric actually is (the public recipe)

TradingView "Technical Ratings" = 26 indicators → each casts Buy(+1)/Sell(−1)/Neutral(0) by a
fixed rule → averaged into three scores in [−1,+1], bucketed Strong-Sell…Strong-Buy:

- **Oscillators (11):** RSI(14), Stoch %K(14,3,3), CCI(20), ADX(14)+DI, Awesome Osc, Momentum(10),
  MACD(12,26,9), StochRSI(3,3,14,14), Williams %R(14), Bull-Bear Power, Ultimate Osc(7,14,28).
- **Moving Averages (15):** EMA/SMA at 10/20/30/50/100/200, Ichimoku Base(9,26,52,26), VWMA(20), HullMA(9).
- **Three outputs:** `osc_rating`, `ma_rating`, `summary_rating` (= mean of the 26 votes).

Vote rules are public (e.g. MA: price>MA→Buy / <→Sell; RSI<30 & rising→Buy, >70 & falling→Sell, else
Neutral; MACD: line>signal→Buy; etc.) — reimplemented from the documented spec, **not** the live-fetching
`tradingview-ta` lib (which can't be made causal/offline).

**Timeframes (from our M5_BA, resampled):** 5m, 15m, 30m, 1h, 2h, 4h, 1d, 1w, 1mo (9 TFs; 1m deferred —
needs M1 fetch). 12 pairs, 2020-11→2026-05.

## 2. The space — and why we DON'T search it directly

Naïve: 2²⁶ indicator subsets × 9 TFs × {follow,fade} × thresholds × horizons × 12 pairs ≈ 10¹⁰⁺.
**We never enumerate subsets.** TV *already pre-combined* the indicators into 3 scores — so the first
object to test is those **~27 pre-aggregated metrics** (3 ratings × 9 TFs), not the power set. The funnel
below collapses everything to a few hundred backtests total.

## 3. The funnel (staged, each stage kills ≥80% before the next)

**Phase 0 — Rating engine (build once, cache).** Causal numpy/pandas reimplementation of the 26 votes +
3 ratings, per (pair, TF). Validate a handful of values against live TradingView (sanity, ±tolerance).
Output: per-pair parquet with columns `{tf}_osc, {tf}_ma, {tf}_summary` + 26 raw votes per TF, aligned to
the M5 grid (higher-TF value = last CLOSED higher-TF bar at each M5 timestamp — see §5). **Compute-once:**
all later phases read these cached series; indicators are never recomputed per config.

**Phase 1 — IC screen of the 3 aggregate ratings** (≈27 metrics × 12 pairs). Each rating vs forward
mid-return at horizons matched to its TF, **net of spread**, tested **follow AND fade**. Gate: IS |t|>2,
OOS same-sign IC, decile-spread > spread cost. Cheap, decisive — directly answers "do the meters work?"

**Phase 2 — IC screen of the 26 raw votes** (same pipeline) → which individual components carry signal,
at which TF. Identifies the survivor shortlist (expect ≤10 of 234 indicator×TF cells).

**Phase 3 — combination search on survivors ONLY** (small set, not 2²⁶): rating-threshold entries
(follow extreme / fade extreme), and **multi-TF confluence** (e.g. 1d rating + 1h rating agree). WF
(3 IS chunks) + MC sign-shuffle. See §3.5 for the search method.

**Phase 4 — full strategy validation** on best combos: SOP backtest (bid/ask fills, IS-only P90 spread
gate, OOS sealed, WF, MC), then paper before any live.

### 3.5 Search method — how to traverse the space

**Binding constraint is NOT search speed — it is overfitting under multiple testing.** Every config tested
inflates false-discovery. So: shrink first (Phases 1-2), then choose the *least* overfit-prone search that
covers the reduced space, and always run a **surrogate-data null baseline** (same search on sign-shuffled /
block-bootstrapped returns) — the real winner must beat the noise-search "best" by a wide margin.

Method ladder (use the cheapest that fits the reduced space):
1. **Exhaustive grid, chunked** — DEFAULT once reduced (~10³-10⁴ configs/pair). Deterministic, auditable,
   no stochastic curve-fit; chunk the config list, numba `prange`, checkpoint per chunk. Preferred.
2. **Coarse-to-fine (multi-resolution grid)** — coarse grid first, refine only around positive regions.
   Cuts evaluations ~10× when params are continuous (thresholds, horizons).
3. **Random / Sobol (quasi-random) search** — if the grid is still too big; better coverage per eval than
   grid for high-dim, fewer evals, still non-adaptive (lower overfit than adaptive optimizers).
4. **Bayesian optimization** — sample-efficient for expensive evals; adaptive ⇒ higher overfit risk; only
   if eval cost dominates (not our case — cached ratings make evals cheap).
5. **Genetic / evolutionary** — ONLY if the reduced space is still combinatorially large (continuous
   weights + free TF assignment). Most overfit-prone (adaptive + many evals). Mandatory guards: fitness =
   IS min(WF-chunk p/d) or WF-pass×MC-sig (never raw return), sealed hold-out touched once, surrogate-null,
   small population × few generations, regularize toward fewer indicators. (`auto_research` has a harness.)

**Objective = AMDDP5 / AMDDP10, not raw return.** We want trades that go the right way from entry to exit
(clean, high-efficiency, short). The project's `research/experiments/amddp5/scorer.py` already defines it
(causal, MC-ready): `AMDDP_trade = pnl − K × cum_dd`, where `cum_dd` = accumulated *underwater pip-minutes*
(depth × time spent below water) and **K=0.05 (AMDDP5) / 0.10 (AMDDP10)**. A trade that's barely underwater
scores ≈ its pnl; one that bleeds underwater before recovering is penalized hard. This *is* the
"high-efficiency / right-direction-throughout" selector. Reuse `amddp5_from_arrays` + `mc_pvalue_amddp`.

**Anti-degeneracy guard (lesson from `project_escma_exit`):** maximizing AMDDP alone collapses to
"exit on first red tick / barely trade" — near-zero drawdown ⇒ high AMDDP, but near-zero pnl and tiny WR
(escma_exit: first-negative-tick exit scored amddp5 +437 while a +1459-pnl 20p-TP scored −73217). So the
fitness must be **AMDDP5 gated by a pnl floor + minimum trade count + min hold**, and ranked by `min`
across WF chunks AND pairs. Complexity-penalize (a config needing all 26 indicators is fit to noise).

### 3.6 Reversal-level / order-concentration conditioning (enters Phase 3, not as a search axis)

Hypothesis: momentum / TV-rating / shock signals behave differently near historical reversal levels
(where orders concentrate) than in open space — likely **fade/stall near a level, run in open space**.
This is the mechanism behind the retrace edge, and aligns with the prior RACS finding (H1 reversals
IC counter-trend, |t|>2 on 69/96 cells — too weak standalone, flagged as a *filter*).

**Causal level map (no future levels — SOP R1/R4):**
- **Swing pivots:** fractal highs/lows over N bars, clustered into levels (merge within ε·ATR).
- **Order concentration:** volume-by-price histogram (VPVR) over a trailing window → high-volume shelves;
  and touch-count (how often a level was revisited) → high-touch = likely resting orders.
- **Known liquidity pools:** prior day/week/session H-L, round numbers (00/50).
- Reuse what exists: curator's H1 S/R, P&F box history, VWMA/volume already in the data.

**Feature:** `dist_to_level = (price − nearest_level)/ATR` (signed: approaching from below/above) +
`level_strength` (touch count / volume share). **Test as a conditioner:** split each survivor's trades by
near-level vs open-space and compare AMDDP5/WR/efficiency; if proximity flips or strengthens the edge, add
it as an entry filter (e.g. "fade only when momentum stalls into a high-strength level"). Scored by AMDDP5,
WF+MC. It does NOT multiply the Phase-1/2 screen — it conditions Phase-3 survivors only.

## 4. A-priori hypotheses (bake into the tests, don't assume)

- TV ratings are **trend-following by construction** (price>MA→Buy). Project history: intraday trend
  never generalizes net of spread → expect **follow fails at low TFs**. So we test **fade** and **higher
  TFs (1d/1w/1mo)** explicitly, and the rating as a **regime FILTER** on an existing edge, not just a
  standalone follow-entry.
- Most likely positive outcomes (rank order to watch): (a) fade extreme ratings at low TF (contrarian),
  (b) follow at D/W, (c) multi-TF confluence as a filter. Treat a flat result as the likely base case.

## 5. SOP / lookahead guardrails (NON-NEGOTIABLE — this project lost 55k pips to lookahead)

- **R1 closed bars only.** Each indicator uses only bars ≤ t.
- **Multi-TF alignment = last CLOSED higher-TF bar**, propagated forward to M5 timestamps. NO `merge_asof`
  to the *current* (forming) higher-TF bar — that was the StrengthSpread 55-min-leak RCA. Shift higher-TF
  series by one bar before broadcast.
- **R3 mid for signals, explicit spread cost** at entry/exit; **R5 spread gate IS-only**; **R8 OOS sealed**
  (touched once, after IS/WF/MC).
- Validate the engine: replay N bars, assert causal == reference within tolerance before any backtest.

## 6. Resource management

- **Memory:** one (pair) at a time; resample all TFs from that pair's M5_BA in-memory; float32 storage,
  float64 only for compute; `del`+`gc.collect()` between pairs. Never hold all 12 pairs' raw TF data.
- **Disk:** cache only the *rating series* (a few floats/bar), not duplicate OHLC per TF — tiny vs raw.
  One parquet per pair under `research/experiments/tv_ratings/cache/`. Estimated < 200 MB total.
  (Disk is at 80%; the cache is small, but we prune intermediate CSVs and avoid storing per-config output.)
- **Efficiency:** compute-once rating cache → Phase 1-3 are vectorized reads; numba only for the
  per-bar backtest sim. Stage-gate funnel avoids the combinatorial blow-up entirely.
- **Crash-resilience (today's lesson):** each phase checkpoints results to CSV as it goes; the rating
  cache persists so Phase 0 never re-runs. Long runs go to background with output to a file.

## 7. Deliverables & go/no-go gates

| Phase | Output | Gate to proceed |
|-------|--------|-----------------|
| 0 | rating cache (12 pairs × 9 TFs) + engine sanity check | causal==reference within tol |
| 1 | aggregate-rating IC table (follow & fade) | any metric: IS\|t\|>2 + OOS same-sign + decile>spread |
| 2 | raw-vote IC table | ≥1 survivor cell |
| 3 | combination WF+MC table | ≥1 config WF-pass + mc_p<0.05 |
| 4 | SOP backtest + paper | OOS p/d>0 net, WF, MC, then paper A/B |

**Kill criteria:** if Phase 1 + Phase 2 yield zero survivors across all pairs/TFs/follow/fade, the TV
ratings carry no net-of-spread edge — log the negative and stop (do not force Phase 3).

## 8. First step
Build Phase 0 (the causal rating engine + cache) for 2-3 pairs, sanity-check, then run Phase 1. ~1 script,
bounded runtime. Everything downstream reads the cache.
