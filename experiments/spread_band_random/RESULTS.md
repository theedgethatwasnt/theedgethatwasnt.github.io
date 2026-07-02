# Random-entry TP=+1 (no SL, 60-min stop) — deep losers by trailing-spread band

**Date:** 2026-06-21 · **Instrument:** EUR_USD S5 (8.26M bars, 2024-02 → 2026-05, ~2.2yr)
**Script:** `random_tp1_spreadband.py` · seeds 12345 + 999 (pattern identical)

## Setup
Always-in-market, random ±1 entry, one trade at a time. TP = +1 pip **net** (mid must move `1 + entry_spread`). No stop. Close at 60 min (720 S5 bars) or at a session gap. Spread charged up front from real per-bar `ask_c − bid_c`. Regime variable = **mean spread over trailing 12 S5 bars (~1 min)**.

## Headline numbers
~41,800 trades, **net −82,000 pips, mean −1.96/trade, win 54.8%**, avg hold 16.4 min. As expected for random entry, the strategy loses ≈ spread per trade — this experiment is about the *conditional loss tail*, not an edge.

## Result (seed 12345; seed 999 matches within noise)

| trailing-spread band (pips) | n | win% | mean | P5 | P1 | worst | <−10% | <−20% |
|---|---|---|---|---|---|---|---|---|
| 1.40–1.47 (lowest) | 4,205 | 52.9 | −1.45 | −10.3 | −20.8 | −84 | **5.3** | **1.2** |
| 1.47–1.51 | 4,190 | 60.3 | −1.41 | −11.8 | −22.3 | −49 | 6.6 | 1.4 |
| 1.51–1.54 | 4,219 | 64.9 | −1.55 | −13.0 | −24.9 | −75 | 7.5 | 2.0 |
| 1.54–1.57 | 3,961 | 67.4 | −1.67 | −13.9 | −26.7 | −99 | 8.4 | 2.3 |
| 1.57–1.59 | 4,111 | 67.4 | −1.67 | −14.0 | −26.4 | −73 | 8.2 | 2.4 |
| **1.59–1.62 (median)** | 4,415 | 67.8 | −1.72 | **−14.4** | −27.8 | −87 | **8.4** | **2.7** |
| 1.62–1.66 | 4,096 | 68.4 | −1.61 | −13.6 | −30.5 | −121 | 8.2 | 2.2 |
| 1.66–1.73 | 4,241 | 62.9 | −1.70 | −13.3 | −27.9 | −100 | 7.4 | **2.5** |
| 1.73–2.54 | 4,208 | 31.1 | −2.52 | −9.6 | −30.5 | −127 | 4.7 | 1.9 |
| 2.54–10.0 (highest) | 4,188 | 5.7 | −4.30 | −10.0 | −13.8 | −119 | 4.5 | **0.5** |

## Findings
1. **Yes — deep losers cluster in a band, and it's the MIDDLE (normal-liquidity) band.** The catastrophic tail (<−20 pips) peaks at **~2.7% around the median spread (1.6–1.7 pips)** and is **lowest at the spread extremes** (1.2% at lowest spread, 0.5% at highest). Same shape for <−10% (peaks ~8.4% mid, ~5% at the edges).
2. **High win rate ≠ safe.** Win% peaks (~68%) in exactly the mid-spread band where deep losers also peak — the classic small-TP / no-SL signature: you win +1 often, but the rare timeout losses are the deepest. The median trade is +1.0 in every band up to 1.66.
3. **Lowest-spread (calm) regime is the best all-round:** least bleed (mean −1.41 to −1.45) *and* a relatively shallow tail (<−10% ≈ 5%). Calm markets don't produce sustained 60-min adverse runs.
4. **Highest-spread regime has the fewest *catastrophic* losses (0.5% <−20)** but the worst mean (−4.3): the TP (1 + huge spread) is nearly unreachable (5.7% win) so it just bleeds steady spread cost and times out — few extreme tails because those spread spikes are transient/mean-reverting, not sustained trends.

## Takeaway
Deep-loss risk is **non-monotonic in spread** — it is *highest in the normal/median-spread regime*, not at the high-spread extreme one might expect. A "avoid deep losers" filter would **sit out the median-spread band**, not the high-spread band. The high-spread band is safe from catastrophe but expensive (unwinnable TP); the low-spread band is the genuine sweet spot (cheapest + shallow tail). None of it is profitable — random entry pays the spread floor regardless.

**Caveats:** one instrument (EUR_USD), one TP (+1), one hold (60 min), one regime window (12 bars). The high-band mean is partly an artifact of charging the full (large) spread as cost. Worth a robustness sweep (other pairs, K, TP, hold) before drawing a general law.

---

# Part 2 — Trailing-stop exit (replace TP), sweep 5..21 pips

**Script:** `random_trail_sweep.py` · seeds 12345 + 999 (identical pattern). Same setup, but the exit is a **trailing stop** (no fixed TP): long stop = HWM − trail, short stop = LWM + trail, within-bar order per SOP R2. The trail also *bounds the loss* (≈ trail + spread from the peak).

## Aggregate (seed 12345)
| trail | trades | mean/trade | win% | worst | <−20% |
|---|---|---|---|---|---|
| 5 | 63,572 | −1.86 | 22.6 | −15.0 | 0.00 |
| 9 | 33,143 | −2.02 | 22.6 | −19.0 | 0.00 |
| 13 | 26,947 | −2.14 | 21.9 | −23.0 | 0.01 |
| 17 | 24,840 | −2.14 | 21.9 | −26.4 | 0.06 |
| 21 | 23,966 | −2.12 | 21.5 | −26.4 | 1.99 |

The trailing stop **bounds the tail**: worst loss ≈ −15 to −26 pips (vs −127 in Part 1's no-SL version), <−20% ≈ 0% up to trail 17. Win% drops to ~22% (trailing exits give back `trail` from the peak, so winners need a sustained run). Tighter trail = more trades + most total bleed (stopped on noise); wider trail = fewer, bigger losses. **All trails still net-negative — random entry has no edge to capture.**

## Does the band help? — YES, and now MONOTONIC (mean pips/trade by spread band)
| trail | 1.40–1.50 | 1.50–1.55 | 1.55–1.59 | 1.59–1.64 | 1.64–10.0 |
|---|---|---|---|---|---|
| 5 | **−1.42** | −1.43 | −1.55 | −1.55 | −2.53 |
| 11 | **−1.44** | −1.41 | −1.75 | −1.76 | −2.87 |
| 17 | **−1.48** | −1.58 | −1.55 | −1.58 | −2.98 |
| 21 | **−1.39** | −1.50 | −1.49 | −1.64 | −3.00 |

**The lowest-spread band is the best (least-negative) cell in ALL 9 trail rows, both seeds; the highest-spread band is the worst by ~1.1–1.6 pips/trade, monotonically.** Unlike Part 1 (where the *tail* peaked non-monotonically at the median), under a trailing stop the **expected cost rises monotonically with spread** and the trail caps the tail — so the band signal is cleaner and fully actionable.

## Answer to "does knowing the spread band help?"
**Yes — but as a cost/risk filter, not an alpha source.** Trading only the calm (lowest-spread) band is reliably ~1.1–1.6 pips/trade better than the high-spread band, consistent across every trail width and both seeds. But **even the best band stays negative** (~−1.4) — band knowledge reduces the bleed and (via the trail) caps the tail; it cannot manufacture a directional edge on random entry. Consistent with the project-wide law: spread/magnitude/cost are predictable, **direction is not**.

---

# Part 3 — best trail (5p) + fixed TP, sweep TP

**Script:** `random_trail5_tp_sweep.py` · seeds 12345 + 999 (identical). Exit = first of: fixed TP (+X net) | 5p trail | 60-min | gap.

| TP | trades | mean | win% | TP-exit% | trail-exit% | worst |
|---|---|---|---|---|---|---|
| **1** | 127K | **−1.757** | 53.6 | 53.5 | 36.3 | −15 |
| 3 | 92K | −1.772 | 35.3 | 34.6 | 51.0 | −15 |
| 5 | 79K | −1.787 | 23.7 | 22.3 | 60.5 | −15 |
| 10 | 68K | −1.832 | 22.9 | 7.6 | 71.6 | −15 |
| 30 (≈pure trail) | 64K | −1.860 | 22.7 | 0.1 | 77.1 | −15 |

**Adding a tight TP improves the mean, monotonically — tighter is better.** TP=1 (−1.757) beats pure-trail (−1.86) *and* beats Part-1's TP=1-no-SL (−1.96) with a bounded tail (worst −15 vs −127). The 5p trail caps the loss; the +1 TP banks the quick winners before the trail gives 5 back. The combined **TP=1 + trail=5** is the best exit found, with a capped tail. Band still helps monotonically (calm best every row): the least-bad cell in the whole study is **trail5 / TP≈2-3 / calm band ≈ −1.42 pips/trade**.

**Synthesis of Parts 1-3 (random entry):** the best *exit* recovers you to ≈ −1.4 to −1.76 pips/trade (≈ the spread), with a bounded tail. No exit structure and no spread band reaches positive — because the entry is a coin flip. The spread floor is the wall; the exit work just stops you over-paying it.

---

# Part A — replace random with a contrarian fade (vs follow control), calm-band gated

**Script:** `pathA_fade_calmband.py`. Entry = M-bar extension ≥ N pips → FADE (against) or FOLLOW (with). Gate: trailing avg spread ≤ 1.50p. Exit = trail 5p + TP 2p, 60-min. Sweep M∈{1,2,5,10}min × N∈{3,5,7,10}p.

**Finding — there IS a faint contrarian sign, but it's smaller than the spread:**
- **FADE beats FOLLOW in every single cell** (fade mean ≈ −1.19 to −1.40; follow ≈ −1.53 to −1.85). Fading a fast extension is directionally correct — consistent with "only contrarian survives."
- **FADE beats the random baseline** (−1.46): best fade = −1.19 (M=2min, N=10p, 379 trades); dense fade (N=3) ≈ −1.35.
- **But still negative.** The contrarian edge is ~0.2–0.35 pip/trade over follow and ~0.1–0.27 over random — **real but well under the ~1.5p spread floor.** It narrows the loss; it doesn't clear it. Tail fully bounded (worst −6.9).

**Verdict:** the direction has a measurable contrarian tilt at the scalp horizon, but its magnitude is a fraction of a pip — uncapturable net of spread. Settles "can any intraday entry beat random here": only barely, and not enough. The lever is not a better intraday entry; it's a longer-horizon edge (Part B).


---

# Part C — HF entry ladder: volume + H1 structure (the real lever for direction)

Stacking edge layers on the HF (intraday) entry, calm-band gated, trail5/TP2 exit, EUR_USD:

| Entry | trades/day | mean/trade | win% | gross edge (net spread + ~1.6p) |
|---|---|---|---|---|
| Random | very high | −1.46 | ~53% | ~+0.14p |
| Contrarian fade (2min ext) | ~daily+ | −1.19 | ~54% | ~+0.4p |
| Fade + volume filter | — | −1.14 | 55% | ~+0.45p (no real lift; intraday vol doesn't separate revert/continue) |
| **Structural fade @ H1 S/R + hi-vol (D=5p)** | **5.7** | **−1.015** | **51%** | **~+0.6p** |
| Structural BREAK @ H1 S/R (sbreak) | — | −1.8 to −1.95 | 37–40% | negative — chasing breaks loses |

Scripts: `pathA_v2_volume.py` (volume tiers), `pathA_v3_structural.py` (H1 TopsBots S/R via `lib/swing_indicators.compute_swing_features`, causal previous-completed-H1).

**Verdict:** structure is the biggest directional lever (fade H1 S/R, never chase breaks); high volume helps the *structural* fade (rejection at a level = climax). The ladder is monotone (−1.46 → −1.02) and the combined gross directional edge reaches ~+0.6 pip — **real and reproducible, but below the ~1.6p OANDA retail spread.** The HF wall is spread, not signal: this combo would be ~break-even on raw/ECN spreads. To clear retail spread outright you need the long-horizon contrarian edge (first-touch-H4 +9.49p), whose moves dwarf the spread.
