# Reading Guide

**How to read the code alongside the book**

The book is divided into a Prelude, nine Parts, and an Epilogue. This guide maps each Part to the experiment directories most relevant to it, and suggests a reading order through the code that mirrors the book's argument.

---

## Narrative Arc

The book moves through four phases: belief (Prelude, Parts I–II), collapse (Parts III–IV), investigation of why the collapses happened (Parts V–VII), and verdict (Parts VIII–IX, Epilogue). Read the code in the same order — starting with the machine that survived, then working backwards through the graveyard.

---

## Prelude — The Fifteen Years Before

No code directories correspond to the Prelude's narrative of the COMEX floor, the blown accounts, and the Zone Recovery platform failures. The Prelude is the frame; the code begins in Part II.

---

## Part I — The Two Questions

*Is there a repeatable edge in retail spot FX? Is it large enough to beat an index fund?*

No experiments yet — this Part sets the terms of the inquiry. The two questions structure everything that follows.

---

## Part II — Building a Machine That Could Not Lie

*The nine validation rules, and where each came from.*

Start here with the infrastructure experiments. These are the skeleton before the flesh:

- `experiments/project_genesis/` — the 16-container Docker architecture, ZMQ IPC, DuckDB storage, 49 unit tests (#1)
- `experiments/central_trades_db/` — the single-source-of-truth trades database (#41)
- `experiments/jpy_exposure_fix/` — a notional-math bug that suppressed live position sizes by 100–160× (#2)
- `lib/incremental_features.py` — the causal feature builder that the post-RCA apparatus is built on

The funnel the book describes — ~7,000 backtests in, one live strategy at the bottom — is the shape of this entire repository.

---

## Part III — The Lookahead Bug: How a Clean Backtest Can Be Pure Fiction

*The 55,000-pip loss that came from data the strategies were never supposed to see.*

This is the book's central crisis. Read these experiments in order:

- `experiments/asi_mc_v3_er/` — the first sign that +84,000 OOS pips was not real (#3)
- `experiments/per_pair_ironnet_v3/` — 12/12 pairs MC-validated; all contaminated (#5)
- `experiments/ironnet_v3_h1/` — V7 at +747 p/d; same fingerprint (#9)
- `experiments/rca_lookahead/` — the root-cause analysis: two bugs, ≈−55,000 pips (#12)
- `experiments/causal_retrain/` — what happened after the bugs were fixed: +0.11 p/d (#13)
- `experiments/strengthspread_rca/` — the second lookahead discovery, caught before deployment: +203.15 → −14.18 p/d (#20)

The causal fix lives in `lib/incremental_features.py`. The tripwire that caught the StrengthSpread leak lives in `experiments/strengthspread_rca/`.

---

## Part IV — The Negative-Result Library: A Map of Where Edge Is Not

*Sixty-odd experiments on trend, momentum, mean-reversion, indicator screens, and machine learning. The answer is no.*

The negative-result library is the longest chapter and the heart of the book's argument. Browse by sub-theme:

**Trend and momentum (the most heavily tested family):**
- `experiments/sma16_momentum/` — 10/12 WF, +29.8 p/d OOS, mc_p=0.0000 — then failed finite-margin test (#49)
- `experiments/price_momentum/` — 12/12 WF, +30.4 p/d OOS — same fate (#50)
- `experiments/h4_donchian/` — the one breakout strategy that worked live, briefly (#26)
- `experiments/h4_donchian_live_check/` — what happened to it (#53)
- `experiments/tr_momentum/` — 12/12 WF with optimistic fills; 0/12 with corrected fills (#33)

**Machine learning (three families, zero new signal):**
- `experiments/slot4_indicator_sweep/` — 43 indicators, 0 crossed OOS zero (#17)
- `experiments/lgbm_shap_ranking/` — SHAP rank and trading edge are orthogonal (#7)
- `experiments/cma_nn_sin/` / `experiments/cma_nn_12pair/` — CMA-ES on fixed neural nets, pre-RCA (#10, #11)

**Mean-reversion against the spread floor:**
- `experiments/racs_reversal/` — 72% of IC cells significant; net still below spread (#30)
- `experiments/deep2_mean_reversion/` — all seeds OOS > IS; still below spread (#15)
- `experiments/daily_range_regime/` — PDH/PDL breakout and reversion, both negative (#43)
- `experiments/grid_trail/` — 480 grid configurations, none positive (#42)

**Lead-lag and stat-arb (the structural ideas):**
- `experiments/lead_lag/` — confirmed major→cross IC +0.07..+0.12 but catch-up takes one bar (#65)
- `experiments/fx_statarb/` — eigen-residual reversion non-stationary across regimes (#64)

---

## Part V — The Hardest Category: Edges That Passed Every Gate and Were Still Not Real

*The strategies that cleared in-sample, walk-forward, OOS, and Monte-Carlo — and then revealed their flaw.*

These experiments illustrate the book's most unsettling finding: passing every standard gate is not sufficient.

- `experiments/fifo_trends_pnf_sweep/` — 71.6/68.5 p/d OOS, passed all gates; live: 7% WR (#23)
- `experiments/fifo_barclose_fill/` — why: bar-close fills vs. trail-level fills, a 200+ p/d gap (#36)
- `experiments/fifo_live_sim_v2/` — the corrected simulation: 135/139 p/d, genuine (#37)
- `experiments/portfolio_variation/` — 130/130 MC-validated variants; all failed finite-margin test (the MC gate has no power when every draw passes) (#56)
- `experiments/finite_margin/` — the gate the standard suite misses: closeout under realistic margin (#57)
- `experiments/gbpusd_regime/` — three measurement failures that each inverted a verdict (#63)

---

## Part VI — Zone Recovery: Where the Edge and the Tail Are the Same Thing

*A strategy where the profit mechanism and the catastrophic loss mechanism are the same mathematical object.*

- `experiments/zr_trail_lock/` — the redesigned exit: P5=162.7 vp/d, P(+)=0.997 (#27)
- `experiments/zr_ml4_car25/` — capping legs at 4 to limit tail: cuts 96% of the edge (#28)
- `experiments/weekend_gap_fill/` — a structurally related tail: 92.2% fill rate, 450–660p adverse gaps (#22)

The lesson is that a martingale's tail and its edge are not separable features — they are the same feature seen from different angles.

---

## Part VII — The Three Walls

*Spread floor. Execution realism. Finite-margin risk. The three structural reasons intraday retail FX is harder than it appears.*

**The spread floor — signals that were real but below the toll:**
- `experiments/oracle_traits/` — perfect-foresight ceiling 2,426 p/day; 49% of optimal legs don't clear one spread (#59, #60, #61)
- `experiments/microstructure_tickpace/` — tick momentum predicts big bars (4× lift), not direction (#25)
- `experiments/msp_propagation/` / `experiments/msp_joint/` — timing AUC 0.73–0.88 across TFs; direction AUC 0.60–0.65, too weak (#40, #48)
- `experiments/spread_band_random/` — structural S/R fade: gross +1.09p, net −0.89p; the spread takes the profit (#67)

**Execution realism — where optimistic fills create phantom edges:**
- `experiments/fifo_s5_trail/` — S5 monitoring changed the verdict from +45 to −69 p/d (#35)
- `experiments/spread_at_entry_fix/` — charging real spread at entry changes the optimizer's behavior (#6)

**Finite-margin risk — the tail the standard gate misses:**
- `experiments/random_vs_trend/` — money management arithmetic: a 2:1 R:R coin-flip still loses the spread (#62)
- `experiments/dynamic_sizing/` — dynamic sizing scales hidden risk proportionally when the pipeline leaks (#55)
- `experiments/conservative_010/` — the SMA-Stack robustness study: real-spread, real-fill, real-margin; net −3,616p; CAR25 negative at every position size (#66)

---

## Part VIII — The Verdict, Kept Separate From the Feeling

*The answer the apparatus gave, separated from what was wanted.*

No new code directories correspond to Part VIII — the verdict is assembled from the evidence already examined. The cross-cutting themes:

- Three machine-learning families (LightGBM, NEAT, CMA-ES) found zero new directional signal.
- Every apparently large positive result was either lookahead, fill-model fantasy, or a martingale tail.
- Volatility and volume are genuinely forecastable; direction is not.
- The only surviving structural positive: contrarian entries at long horizons, consistent with existing literature (carry, mean-reversion), and constrained by the same spread toll.

---

## Part IX — The Sine Wave: The One Test That Could Have Overturned the Verdict

*A control experiment: a synthetic signal with known structure, to verify the apparatus can find an edge when one is there.*

- `experiments/causal_retrain/` — the sine-control run is documented here; it proved the apparatus could find the planted edge, and confirmed the deficit was entirely in the entry, not the machine (#13)

---

## Epilogue — What the Curiosity Was Really For

No code directories correspond to the Epilogue's reflection. The Epilogue looks outward at what was not tested (carry, ECN venues, longer timeframes, order-flow data) and at why a negative result, documented carefully, is itself a kind of contribution.

---

## Survivors and Open Leads

The book ends with the few strategies that were not definitively closed:

- `experiments/post_shock_retrace/` + `experiments/markov_d1/` — the only fully validated contrarian edge (#51, #52)
- `experiments/zr_trail_lock/` — Zone Recovery at large leg counts; positive but tail-bounded (#27)
- `experiments/conservative_010/` — the SMA-Stack's final honest accounting; the negative result the book closes on (#66)

Two leads the book names as open, with code here:
- `experiments/bb_reentry_fade/` — first robust intraday OOS edge found (post-book; EUR/USD fade on band re-entry after outside bar)
- `experiments/first_touch_h4/` — H4 first-touch + low-volume reversion; OOS +9.49p/trade, MC p=0.018, WF positive all thirds

Neither was live when the book closed. They are the blank on the map the project itself points at.
