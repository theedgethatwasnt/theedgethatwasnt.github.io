# Regime-Conditioned Entry Research Program — Master Document

**Project:** fx-core
**Version:** 2.0 — Consolidated (supersedes v1.0 experiment spec)
**Date:** 2026-07-06
**Scope:** Complete record of the regime-characterization theory, experiment design, cost/broker analysis, and the OANDA→ECN transferability and staging plan.

---

# Part I — Theory: Characterizing Intra-Bar Market Regime

## 1. The Problem

Within a single M5 bar observed at S5 resolution (60 samples), price "wiggles": it zigzags through a sequence of up-legs and down-legs. The goal is to compress the character of that motion into a minimal set of numbers that adequately (not fully) describes the regime within the bar:

- Is there drift, and in which direction?
- Are the up-legs systematically larger than the down-legs?
- Is the swing structure HH/HL (uptrend), LH/LL (downtrend), diverging (higher highs *and* lower lows — megaphone), or converging (lower highs *and* higher lows — triangle)?
- Are the legs growing or shrinking in size?
- Is the motion a disciplined trend or erratic chop?
- How much total activity is there?

## 2. Why Standard Deviation and ATR Cannot Do This

Standard deviation and plain ATR are **order-blind**. Shuffle the 60 S5 returns into any sequence and the standard deviation is identical — yet one ordering is a clean trend and another is pure chop. Variance measures the *size* of the wiggles; it says nothing about their *organization*. Regime detection therefore requires statistics that are **sensitive to ordering and direction**. This is the central distinction the feature set is built on.

## 3. The Regime Feature Vector

All features are computed from the S5 series within the window, plus the zigzag legs extracted from S5 peaks and troughs.

| # | Feature | Definition | What it captures |
|---|---------|------------|------------------|
| 1 | **Drift** | Net signed move (last − first), or OLS slope of price vs time. Equivalent to (Σ up-legs − Σ down-legs). | Direction and strength |
| 2 | **Efficiency Ratio (Kaufman)** | \|net move\| ÷ Σ\|all legs\|. Range 0→1. | Disciplined trend (→1) vs erratic chop (→0). The single most useful scalar. |
| 3 | **Realized volatility / activity** | Std of S5 returns, or total path length Σ\|legs\|. | Activity magnitude, kept separate from direction |
| 4 | **Peak-envelope slope** | OLS line through zigzag peaks. | HH vs LH structure |
| 5 | **Trough-envelope slope** | OLS line through zigzag troughs. | HL vs LL structure |
| 6 | **Leg expansion** | Regression slope of \|leg size\| against leg index. | Wiggles growing (+) vs dying out (−) |
| 7 | **Variance Ratio VR(k)** (Lo–MacKinlay) | Var(k-step return) / (k · Var(1-step return)). | Persistence: VR > 1 trending, VR < 1 mean-reverting, ≈ 1 random walk |

The two envelope slopes are the elegant compression of the swing-pattern taxonomy: there is no need to classify letter-patterns (HHHL, LHLL, …) — the two slopes contain them:

- Both slopes up → uptrend
- Both slopes down → downtrend
- Peaks up, troughs down → **diverging** (megaphone / expanding)
- Peaks down, troughs up → **converging** (triangle / contracting)

## 4. Regime Mapping (interpretive, not hard classification)

- **Disciplined trend:** high ER, VR > 1, both envelope slopes same sign, drift large.
- **Erratic / choppy:** low ER, VR ≤ 1, drift ≈ 0, many direction changes.
- **Diverging:** positive leg expansion, envelope slopes spreading apart, rising volatility.
- **Converging:** negative leg expansion, envelope slopes closing.

## 5. Why Wavelets Are Excluded

Wavelets answer "at which timescale does the energy live" — a different question than regime. Over ~60 samples they are noisy, hard to reduce to a stable scalar, and add nothing beyond ER + VR for the trend/chop axis. Excluded unless scale-decomposition specifically becomes a goal.

## 6. Minimal Vector

If maximum compression is required: **{drift, efficiency ratio, realized vol, peak-slope, trough-slope}** — five numbers separately encoding direction, trend quality, scale, and expansion/contraction structure. Add VR for mean-reversion texture. The experiment uses all seven.

---

# Part II — The Experiment

## 7. Objective and Trade Structure

Determine whether the prior-bar (or prior rolling 5-minute window) regime vector predicts the outcome of an otherwise-random entry with:

- **Small take-profit:** 2.0–2.5× spread (gross)
- **Generous stop-loss:** 2× the TP distance (TP:SL = 1:2, i.e. risk 2 to make 1)
- **Maximum hold:** 1–4 M5 bars (5–20 minutes), fixed per run
- **One entry per M5 bar maximum; one open position at a time (FIFO)**

## 8. The Theoretical Baseline — the Most Important Number

For a driftless random walk with barriers at TP and SL:

> **P(hit TP first) = SL / (TP + SL)**

At TP:SL = 1:2 → **P(win) = 2/3 ≈ 66.7%**, expectancy = 0 before costs.

A naive backtest of this structure will show a seductive ~66% win rate and feel like an edge. It is not one. **All results must be reported relative to 66.7%, never in absolute terms.**

## 9. Cost-Adjusted Break-Even p\*

Let all-in round-trip cost = C (spread + commission + slippage). With market entry:

- TP_net ≈ TP − C, SL_net ≈ SL + C
- **Break-even win rate p\* = SL_net / (TP_net + SL_net)**

The gap **(p\* − 0.667) is the exact size of edge the regime filter must deliver.** Compute p\* per broker/cost scenario before touching data; it is the success criterion. Additionally: entry occurs at second 6 of the new bar, not at the prior close — the close→entry drift is part of the cost and must be logged per trade. If favored regimes are fast-moving, this gap systematically eats the small TP.

Worked example at TP = 3 pips, SL = 6 pips (see Part IV for cost derivation):

| Broker scenario | All-in cost C | Net winner | Net loser | p\* | Required edge over 66.7% |
|---|---|---|---|---|---|
| OANDA US (retail) | ~1.4 pips | 1.6 | 7.4 | **≈ 82%** | ~15.5 points |
| Offshore ECN (raw + commission) | ~0.7 pips | 2.3 | 6.7 | **≈ 74%** | ~7.7 points |

A 15-point win-rate lift from a prior-bar regime feature is almost certainly fantasy; an 8-point lift is ambitious but plausible. **Cost structure alone can move the strategy from dead-on-arrival to worth testing.**

## 10. Entry Protocol (primary, clock-aligned)

1. M5 bar closes at t = 0.
2. **t = 0–5 s:** compute the 7-feature regime vector on the just-closed bar. Strictly causal — no data from the entry bar, no future ticks. Lookahead leakage is the single most common way such studies produce fake edges; here it is prevented by construction, since the feature window and the entry are separated by a hard bar boundary.
3. **t = 6 s:** fire order (or abstain) at market.
4. One entry per bar max; one position at a time (FIFO). While a position is open, signals are logged but skipped.
5. Log per signal: full feature vector, arm decisions, entry price, prior-bar close, close→entry slippage, live spread at t = 6 s, session tag, skipped-due-to-open-position flag.

This timing is realistic to implement live and leak-free by design.

## 11. Labeling — Triple-Barrier Method (López de Prado)

For each entry, three barriers:

- **Upper:** TP at 2.0× spread and 2.5× spread (two separate runs; TP multiple is a design parameter with exactly these two values, not a free knob)
- **Lower:** SL at 2× TP distance
- **Vertical:** fixed time limit — primary at **2 bars (10 min)**, secondary at **4 bars (20 min)**. Do not let the horizon float within a run; a floating exit rule multiplies the hypothesis count.

Label by first barrier touched: **+1** (TP), **−1** (SL), **0** (timeout → close at market). Timeout trades tracked as their own P&L bucket — with a tight TP a large fraction of trades will time out, and whether timeouts bleed or break even can decide the whole system. Net P&L recorded in pips after all costs.

## 12. Direction Arms — Defining "Random Entry"

"Random" has two axes: random *time* (entry timestamps on a grid) and *direction*. Direction is the real design choice. Three arms evaluated on identical signal timestamps:

1. **With-drift:** direction = sign of prior-bar drift
2. **Against-drift:** opposite
3. **Coin-flip:** random direction (control)

The (with-drift) vs (against-drift) contrast is the cleanest test that regime carries directional information: if arm 1 clears p\* while arm 2 falls below the 66.7% baseline, the features are doing real work.

## 13. Sample Size

- 288 M5 bars/day; after regime filtering, session stratification, and FIFO blocking: expect ~10–40 realized trades/day.
- Detecting a ~5-point win-rate lift over a ~70% baseline requires on the order of **≥1,000 trades per regime bucket**.
- Therefore: several months of S5 data minimum; **at most 2–3 ER buckets** for the primary test to avoid starving cells.

## 14. Threats to Validity and Controls

| Threat | Control |
|---|---|
| Lookahead / leakage | Hard bar boundary between feature window and entry; prior-bar data only; enforced in code by construction. |
| Trade dependence | One-per-bar + FIFO prevents overlapping horizons natively. Residual volatility clustering: **block bootstrap** (resample by day or session) for all confidence intervals — never trade-level bootstrap, which inflates effective N. |
| FIFO selection bias | While a trade is open, signals are skipped, so realized trades are not a random sample of signals. Log every signal; analyze both populations: *all signals* (does the information exist?) and *FIFO-realized* (what does the system as traded earn?). |
| Cost sensitivity | All results at 1.0× / 1.5× / 2.0× assumed cost. An edge that dies at realistic cost is not an edge. |
| Multiple testing | Pre-register the primary hypothesis (one feature, one threshold, one TP multiple, one horizon) before looking. Everything else is exploratory. Each extra knob (thresholds, TP/SL distances, lookbacks) is a coin flip at spurious significance. |
| Session confounds | Volatility clusters and time-of-day both correlate with regime and with win rate. Stratify by Asia/London/NY; the regime effect must survive within a fixed session — otherwise the study has merely rediscovered "London open is volatile." |
| Overfitting | Walk-forward: buckets/thresholds fit on rolling past windows only; final ~30% of data held out and evaluated **once**. |

## 15. Analysis Plan

1. Compute p\* per cost scenario (§9).
2. Per arm × ER bucket × TP multiple × horizon: conditional win rate, mean R, expectancy in pips.
3. Binomial test of bucket win rate vs p\*; block-bootstrapped 95% CIs; report **effect size** (points above 66.7% and above p\*), not just p-values.
4. Timeout-bucket P&L reported separately.
5. Walk-forward out-of-sample confirmation of the primary hypothesis only.

**Success criterion:** with-drift entries in the high-ER bucket clear the cost-adjusted break-even p\* out-of-sample, with the confidence interval excluding p\*.

**Anticipated most-likely outcome (stated for honesty):** regime shifts win rate (arm 1 > arm 3 > arm 2), but the tight TP means spread consumes the edge and expectancy hovers near zero at retail costs. This is still a valuable finding — it localizes the problem to the entry/exit cost structure, which motivates passive entry (§16) and the ECN cost base (Part IV).

## 16. Secondary Study A — Passive (Limit) Entry

**Mechanism.** A market order crosses the spread: buy at the ask, half a spread above mid — every trade starts half a spread underwater, a huge fraction of a 2–2.5× spread TP. A limit order posted at the bid waits for a seller to come to you; if filled, the half-spread is *earned* rather than paid. A fully passive round trip reclaims the entire spread. With this trade geometry, a winner can go from ~1.5 cost-units net to ~2.5 — enough to move the required win rate **below** the random-walk baseline, which changes everything.

**The catch — adverse selection and fill uncertainty.** A buy limit fills precisely when price moves down against you, so fills systematically arrive at the start of adverse moves. Worse, in the best regimes (strong drift with you) price runs away and the order never fills: the best trades are missed and the mediocre ones are filled. Passive entry trades explicit cost (spread) for implicit cost (adverse selection + missed trades).

**Protocol.** Fills become part of what must be simulated, and a limit-fill model on S5 data is approximate at best. Honest compromise: passive entry with a short patience window (rest the limit 10–15 s, cancel if unfilled); log the miss rate per regime bucket; compare net expectancy vs market entry on matched signals.

## 17. Secondary Study B — Rolling Window vs Clock Bars

A rolling 5-minute window updated every S5 step yields the same regime vector refreshed every 5 seconds. The M5 clock boundary is arbitrary — the market doesn't know where the bar starts; a trend beginning at minute 3 of a clock bar is split across two bars and diluted in both, while a rolling window sees it whole. For *feature quality*, rolling is strictly better, and it models a live system more faithfully.

Three consequences:

1. **Entry discipline is lost.** The clock design gives a clean pre-registered decision point (6th second, once per bar). Rolling stats produce a signal every 5 s, so a trigger rule must be defined (e.g. ER crossing a threshold) — a new degree of freedom. Subtlety: a rolling stat crossing a threshold is often driven by an old bar *leaving* the window, not by new information arriving.
2. **Massive overlap.** Windows 5 s apart share 59/60 of their data; the signal series is extremely autocorrelated and signals cluster. FIFO handles the trading side, but inference must count **regime episodes, not signal instances** — a 4-minute stretch of high ER is one observation, not 48.
3. **Design placement.** Keep the clock-bar experiment as the primary, pre-registered test (cleaner, fewer knobs, dependence controlled by construction). Run the rolling version as a secondary A/B answering one question: *does entering immediately when the regime turns favorable beat waiting up to 5 minutes for the next bar boundary?* If regime information decays over minutes, entering earlier could matter a lot; this measures the decay directly.

---

# Part III — Pre-Registered Parameters (locked before data analysis)

| Parameter | Value |
|---|---|
| Feature for primary test | Efficiency Ratio (prior M5 bar) |
| ER buckets | 2–3, thresholds fixed on training portion only |
| Direction rule (primary) | With-drift |
| TP (primary) | 2.0× spread |
| SL | 2× TP |
| Max hold (primary) | 2 bars (10 min); secondary 4 bars (20 min) |
| Entry timing | 6th second of new M5 bar |
| Position rules | 1 entry/bar max, 1 position at a time, FIFO |
| Out-of-sample | Final ~30%, walk-forward, evaluated once |
| Cost scenarios | OANDA ~1.4 pips and ECN ~0.7 pips, each at 1.0× / 1.5× / 2.0× |

---

# Part IV — Broker Cost Analysis: OANDA US Retail vs Offshore ECN

## 18. Cost Structures (as researched July 2026)

**OANDA US (retail, CFTC/NFA-regulated):**
- Standard (spread-only) account: EUR/USD averages ~1.4 pips; independent testing measured ~0.94–1.54 pips (~$14+ per standard-lot round turn).
- Core pricing model: ~0.4 pips average + $10/lot round-turn commission → all-in ~1.4 pips equivalent.
- **All-in cost: ~1.0–1.5 pips; use 1.4 as the planning number.**
- No commission on standard; costs built into spread; market-maker execution model.

**Offshore ECN (IC Markets Raw as benchmark; Pepperstone Razor similar):**
- Raw spread EUR/USD averages ~0.1 pips (from 0.0) + $3.50/lot/side commission (cTrader $3.00/side) → **all-in ~0.62–0.72 pips**. Pepperstone Razor all-in ~0.80 pips.
- **All-in cost: ~0.7 pips; roughly half of OANDA US.**
- True order book: Level II depth of market, sub-35ms execution, no scalping/EA restrictions.

## 19. Impact on This Strategy

1. **Break-even hurdle halves.** From §9: required edge over the random-walk baseline is ~15.5 points at OANDA cost vs ~7.7 points at ECN cost. The first is almost certainly unattainable from a prior-bar regime feature; the second is ambitious but plausible.
2. **TP geometry.** Since TP is defined as a multiple of spread, OANDA's wider spread forces a physically larger TP (~3 pips vs ~1.5–1.8 on ECN), meaning slower barrier resolution, more timeout trades, and a longer effective horizon per trade.
3. **Passive entry is only real on ECN.** On an ECN, a limit order rests on actual interbank liquidity in a visible book. At OANDA US (market maker), a limit order fills against the dealer's own quote — "reclaiming the spread" is not available in the same sense, and the adverse-selection dynamics of §16 cannot be measured meaningfully there.
4. **US regulatory constraints.** OANDA US: 50:1 leverage cap, NFA FIFO close rule, no hedging. The strategy's one-position-at-a-time design already complies with FIFO, so this costs nothing. Leverage is immaterial to the experiment.
5. **Foreign-account caveats (flagged, not advised).** IC Markets and most offshore ECNs do not accept US residents (CFTC rule, not broker preference). For a US person, routes around this carry real legal/tax obligations — FBAR/FATCA reporting on foreign accounts — and reduced recourse outside CFTC/NFA protection. For a non-US person or non-US residency, none of this applies and the ECN route is straightforwardly better for this strategy.

---

# Part V — Transferability: Can the OANDA Study Validate for ECN?

## 20. The Governing Principle

Separate what is a property of the **market** from what is a property of the **broker**:

- **Market properties (broker-independent, transfer fully):** the price path of EUR/USD, the regime feature vector, whether prior-bar regime predicts the direction/organization of the next 5–20 minutes, gross barrier-hit probabilities measured on mid-price. Every broker is quoting essentially the same underlying market; feed differences at S5 resolution are second-order.
- **Broker properties (do NOT transfer):** all-in cost, spread *dynamics* (how the spread widens by session, at news, and — critically — possibly in exactly the regimes the filter selects), execution/slippage behavior, limit-order fill mechanics, and therefore net expectancy and the entire passive-entry study.

## 21. What the OANDA Study Can and Cannot Establish

**Can establish (fully valid for ECN):**
- Whether the regime signal *exists*: conditional gross win rates per bucket per arm, the with-drift vs against-drift contrast, effect sizes over the 66.7% baseline. This is the scientific heart of the experiment and it is broker-agnostic — **provided barriers are defined on mid-price in absolute pips**, not on OANDA's bid/ask or as multiples of OANDA's live spread. (If barriers are tied to OANDA's spread, the geometry itself becomes broker-specific and transfer degrades. Design accordingly: label trades on mid-price barriers; apply costs as a separate layer.)
- A *conditional* ECN viability estimate: take the OANDA-data trade set, re-price it under the ECN cost model (~0.7 pips, at 1.0×/1.5×/2.0×), and test against ECN p\* ≈ 74%. This is arithmetic on already-labeled trades — no ECN account needed.

**Cannot establish:**
- True ECN cost per trade at signal times. OANDA's recorded spread at t = 6 s is OANDA's spread; ECN raw spread at the same instants correlates but is not identical, and the difference is regime-dependent (raw ECN spreads widen sharply at news/thin liquidity, exactly when some regimes fire). A flat 0.7-pip assumption is a first approximation only — obtaining or recording an ECN spread series is the fix.
- Execution reality: fill quality, slippage distribution, close→entry gap on an ECN.
- The passive-entry study (§16) in its entirety: fill rates, adverse selection, and miss rates per regime bucket require a real book. This is structurally impossible to study at OANDA US.

## 22. Decision Logic — Can OANDA Preclude the Need for ECN?

**Yes, in the negative direction; no, in the positive direction.** The OANDA study is a valid *filter* but not a valid *confirmation*:

- **Negative result is decisive.** If the regime signal does not exist on gross (mid-price) outcomes — with-drift ≉ against-drift, no bucket separates from 66.7% — then no cost structure can save it. **Stop. ECN is not needed. This is the cheapest and most likely exit.**
- **Weak-positive result is also decisive.** If the signal exists but the re-priced edge fails even the ECN p\* (~74%) with pessimistic costs — **stop, or redesign the exit structure** (wider TP, passive entry concept, different horizon). ECN account still not needed.
- **Strong-positive at OANDA costs (edge clears p\* ≈ 82%).** Then it clears ECN costs a fortiori and the strategy is viable even retail — but this outcome is improbable given the size of lift required.
- **The realistic interesting zone: edge in the ~8–15 point band.** Signal is real; viable only at ECN costs. Here the OANDA study *cannot* confirm — it can only nominate. ECN validation becomes mandatory before believing the result.

**Bottom line: both are needed for a positive answer; OANDA alone suffices for a negative one.** The OANDA phase answers "is there information?"; the ECN phase answers "is there money?"

## 23. What Must Be Conducted on ECN, and at What Stage

**Phase A — Signal discovery (OANDA data; no ECN account needed).**
Full primary experiment of Part II on existing fx-core / OANDA S5 data. Barriers on mid-price in absolute pips. Deliverable: does the regime signal exist, with what effect size, out-of-sample? *Gate: stop here on a null result.*

**Phase B — Cost re-parameterization (offline; no ECN account needed).**
Re-price Phase A trades under the ECN cost model; test against ECN p\*. Improve the cost model by obtaining an ECN historical spread series (many ECNs publish average-spread data; demo feeds can record live raw spreads) so cost per trade reflects the actual spread *at signal times in the selected regimes* rather than a flat 0.7. *Gate: stop or redesign if the edge fails ECN p\* under pessimistic costs.*

**Phase C — ECN validation (ECN account or demo feed required).**
Only entered if A and B pass. Contents, in order of increasing commitment:
1. **Record raw ECN spreads and depth at signal times** (demo/data feed) — validates the Phase B cost assumptions per regime bucket.
2. **Execution study:** small-size live trades replicating the t = 6 s protocol; measure fill quality, slippage, close→entry gap; compare realized costs to modeled.
3. **Passive-entry study (§16):** ECN-only by nature — limit fill rates, miss rates per regime bucket, adverse-selection cost, patience-window tuning. If Phase B showed the edge is marginal at market-order costs, this study decides whether spread reclamation rescues expectancy.
4. **Confirmation run:** the pre-registered primary hypothesis re-evaluated on live/forward ECN data, once.

**Phase D — Rolling-window A/B (§17)** can run on OANDA data (it is a signal-timing question, broker-agnostic), but its live implications are only settled in Phase C where execution latency is real.

## 24. Design Adjustments to Maximize Transferability (apply now, in Phase A)

1. **Label on mid-price, absolute-pip barriers.** Keep the 2.0×/2.5× spread definition only for *choosing* the pip values (anchored to the ECN spread, e.g. TP ≈ 1.5–1.8 pips), then freeze them in pips. This makes the label set broker-independent and lets any cost model be layered on afterward.
2. **Log spreads, don't consume them.** Record OANDA's live spread per signal as data; never bake it into labels.
3. **Record an ECN spread series in parallel from now** (demo feed costs nothing) so Phase B uses real regime-conditional ECN costs.
4. **Keep the timeout bucket and slippage log granular** — these are the quantities that differ most across brokers and will be compared directly in Phase C.

---

# Part VI — Implementation Pipeline

1. S5 ingestion → M5 bar-close regime vectors for all history (7 features, strictly causal).
2. Signal generator at t = 6 s per bar: features, live spread (logged), three arm directions, FIFO state, skip flags.
3. Triple-barrier labeler on **mid-price**: TP ∈ {fixed pip values anchored to 2.0× and 2.5× ECN spread}, SL = 2× TP, vertical ∈ {2, 4} bars; labels +1/−1/0.
4. Cost engine as a separate layer: OANDA model (~1.4) and ECN model (~0.7, upgraded to recorded series when available), each at 1.0×/1.5×/2.0×; plus close→entry slippage.
5. Analysis harness: buckets, arms, block bootstrap (by day/session), binomial tests vs each p\*, effect sizes, walk-forward split (final ~30%, run once).
6. Reports: per-arm/bucket tables, all relative to 66.7% and to each broker's p\*; timeout bucket separate; all-signals vs FIFO-realized populations both reported.
7. Parallel task from day one: ECN demo-feed spread recorder.

# Appendix — Key Numbers at a Glance

| Quantity | Value |
|---|---|
| Random-walk baseline win rate (TP:SL = 1:2) | 66.7% |
| OANDA US all-in cost (EUR/USD) | ~1.4 pips |
| ECN all-in cost (EUR/USD) | ~0.7 pips |
| p\* at OANDA cost (TP 3 / SL 6) | ≈ 82% (edge required ≈ 15.5 pts) |
| p\* at ECN cost (TP 3 / SL 6) | ≈ 74% (edge required ≈ 7.7 pts) |
| Trades needed per bucket | ≥ 1,000 |
| Realized trades/day (est.) | 10–40 |
| Feature window | prior M5 bar (60 × S5) |
| Entry | 6th second of new bar, market order |
| Horizons | 2 bars primary, 4 bars secondary |

---

# Addendum 1 (2026-07-06) — Pre-registration amendments, locked before analysis

1. **Primary arm flipped to against-drift.** Repo priors (7 independent negative
   with-drift results; all gross-positive precedent contrarian) make with-drift a
   near-certain negative control. §12 arms unchanged; only the confirmatory
   designation moves. See research/experiments/regime_entry/PREREGISTRATION.md.
2. **TP pip values frozen: {1.5, 1.8, 3.2, 4.0}p; confirmatory 3.2p** (2.0× the
   measured OANDA EUR_USD median spread of 1.6p — resolves the §9-vs-§24.1
   anchoring ambiguity by testing both, one confirmatory).
3. **Zigzag legs (spec gap): plain alternating extrema** on S5 mid-closes within
   the closed window, endpoints included as leg boundaries.
4. **VR k=4 primary** (20 s), k∈{2,8} exploratory. **ER terciles, IS-fixed.**
5. Prior-work correction to §21/§24: the barrier baseline and the gross-vs-cost
   separation are established in-repo (random_vs_trend 66.7% note; harvester
   break-even-WR table; structural_fade gross/net/ECN split) — cited, not novel.
   Two previously-open ECN branches (harvester, 010 TP100/SL200 fade) are
   adjudicated by Phase B here.
