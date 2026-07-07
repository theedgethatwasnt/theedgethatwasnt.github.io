# Multi-Day Contrarian Program — Pre-Registration (LOCKED 2026-07-06)

> Fills Appendix D §7 ("Longer Holding Horizons — SIMPLY-UNREACHED") + §5 (carry accounting).
> Governed by the 10-rule Backtest–Live Consistency SOP (CLAUDE.md), especially **R10**:
> the null shares the trade geometry — coin-flip control arm on identical timestamps.

## Data (fixed)

- **12-pair M5 bid/ask parquets** `data/m5_ba/<PAIR>_M5_BA.parquet`, 2020-11-11 → 2026-05-21 (~5.5 yr, ~412k bars/pair).
- H4 and D1 bars aggregated from M5 **mid**; spread from `ask_c − bid_c` at the M5 entry bar (R3/R3a).
- H4 anchor: OANDA convention (NY 17:00). Aggregation code must be shared by any later live port (R6).
- **IS = first 70%** by time (≈ 2020-11-11 → 2024-09-25). **OOS = final 30% (≈ 2024-09-25 → 2026-05-21), SEALED — evaluated exactly once, after the user gate (R8).**

## Primary hypothesis (confirmatory, one test)

**H1:** The first-touch H4 low-volume fade, at the parameters frozen below, traded as a
multi-day program on all 12 pairs **net of spread AND carry**, achieves OOS:
1. **Money criterion:** mean net expectancy > 0 with day-block-bootstrap 95% CI excluding 0
   (timeouts included; carry at 1.0× measured markup; also reported at pessimistic 2.0× markup).
2. **Information criterion (R10):** net expectancy minus the coin-flip arm's net expectancy > 0,
   day-block-bootstrap 95% CI excluding 0, identical timestamps and exit machinery.

Both must hold. One shot.

## Frozen entry/exit parameters (verbatim from the deployed paper service, `services/strategy_first_touch_paper/main.py`, in place since 2026-06-18 — copied, not fitted)

| Param | Value |
|---|---|
| Swing-level lookback L | 25 H4 bars |
| Touch tolerance EPS | 12 pips |
| Volume gate | tick-volume of touch bar < mean of prior VW=20 H4 bars (LOW-volume only) |
| Touch count | FIRST touch only (TOUCH_MAX=1) |
| Direction | fade the level (short at swing high touch, long at swing low touch) |
| Target | 2.0 × ATR(14, H4) at entry |
| Stop | 2.0 × ATR(14, H4) at entry (broker-side in any live port) |
| Time cap | 12 H4 bars (≈ 48 h), exit at close |
| Position rules | one position per pair; FIFO; weekend holds allowed (carry accounted) |
| Entry fill | next M5 open after the signal H4 bar closes, mid ± half logged spread |

## Cost model (locked)

- **Spread:** per-trade logged M5 `ask_c − bid_c` at entry and exit bars (full round-trip), sensitivity ×{1.0, 1.5}.
- **Carry:** per-instrument long/short **published financing rates from the OANDA API**
  (`accounts/{aid}/instruments → financing.longRate/shortRate`, which already embed the retail
  markup "pinch"), held-days × annualized rate / 365, weekend/triple-swap per
  `financingDaysOfWeek`. Historical rates are not queryable; the **rate level is scaled through
  time by the FRED policy-rate differential series** with the currently measured pinch held
  constant — an approximation documented under R9 (direction of bias: uncertain; mitigated by the
  2.0× pessimistic sensitivity). Method inherits `research/experiments/carry/carry_financing.py`.

## Control arms (identical timestamps, identical exits)

1. **Coin-flip direction** at every signal timestamp (seeded, seed=20260706) — **the operative null (R10)**.
2. **With-touch continuation** (trade toward the break) — expected negative, ordering check.

## Gates before the sealed OOS may be opened (all IS-only)

1. Harness self-test: on synthetic random walks the coin arm's net expectancy ≈ −(spread+carry) and signal≈coin (no phantom edge).
2. **Parity check (R7-analog):** IS trade list reproduces the 2026-06-18 study's IS results on overlapping windows (tolerance: expectancy ±0.5p, trade count ±5%). Divergences documented (R9) before proceeding.
3. IS net expectancy > 0 at 1.0× costs, and > coin arm.
4. **Walk-forward:** 3 IS thirds, net-positive in ≥2 of 3, none catastrophically negative (< −2p/trade).
5. **MC:** day-block bootstrap P(net ≤ 0) < 0.05 on IS.
6. Breadth: ≥ 6/12 pairs gross-positive IS.
7. **User gate:** IS summary reviewed by the user; OOS unsealed only on explicit approval (typed UNSEAL).

## Pre-declared secondary analyses (exploratory; reported, never promoted to confirmatory)

- CSI StrengthSpread H4/64-bar hold (prior: OOS Sharpe 0.59, csi_factor_study) on the same split/cost model.
- D1 RSI(2) mean-reversion (prior: +4.7 p/trade, weak) same treatment.
- Equal-risk portfolio of the three signals (correlation + combined bootstrap).
- Gate columns (IS-only): risk-on/off (SPX500/XAU/WTICO CFD D1 trend), FRED yield-differential sign,
  OANDA positioning-book crowding, calendar-event proximity. Each gate = one pre-declared split,
  no threshold search; any promising gate needs its own future pre-registration.

## Decision rule

- H1 passes both criteria OOS → paper-trade the program ≥ 50 trades (fx-first-touch-paper exists) before any live-money discussion.
- Money passes but information fails (≈ coin) → edge is structural/carry, not signal — report and stop.
- IS gates fail → stop; OOS stays sealed for one future amended shot.

## Threats & mitigations

- Carry-history approximation (above, R9). — Multiple-testing: one confirmatory cell; everything else labeled exploratory. — Volume gate uses OANDA tick volume (venue-specific; live parity holds since live also reads OANDA). — Weekend gaps through stops: modeled fill at first M5 open past the stop (gap slippage realized, not idealized) — matches broker behavior direction (R9: conservative). — Regime: 5.5 yr spans hike + cut cycles; WF thirds gate covers it.

## A4 parity record (2026-07-06) — **Gate 2 = FAIL, root cause diagnosed, not a harness bug**

**Original study located:** `research/experiments/touch_ladder/first_touch_v2.py` (2026-06-18),
cited verbatim by `services/strategy_first_touch_paper/main.py`'s docstring and by
JOURNEY-README.md's 2026-06-18 entries. Data: `data/m5_ohlc/{PAIR}_M5.parquet` (MID-only OHLC
+ tick volume, no bid/ask), tz-aware `America/New_York`→UTC, range 2021-01-03→2026-04-09.
`IS_FRAC=0.6`. Grid-searches (tgt,sl,Hcap) on IS portfolio expectancy → picks (2.0,2.0,12)
(matches the frozen params). Splits by volume at the IS-median `vrel` (=1.16, matches
`VREL_MAX`). Re-ran the script verbatim and reproduced the recorded headline exactly:
`loVol: IS +4.00p | OOS +9.49p (n303, WR54%) MC P(<=0)=0.018`, 7/12 pairs OOS+ — confirms
memory (`project_first_touch_h4.md`) is accurate.

**Data-source check (ruled out as a cause):** `data/m5_ohlc/*` mid OHLC+volume is
bit-for-bit identical to `data/m5_ba/*` mid OHLC+volume on every overlapping M5 timestamp
(diff = 0.0 across open/high/low/close/volume, 393,090 rows checked for EUR_USD). Same feed;
`m5_ba` just extends earlier (2020-11-11) and later (2026-05-21) and adds bid/ask.

**Window used for parity (R8 safety):** the study's headline (+9.49p) is its OOS (last 40%,
2024-02-29→2026-04-09), which oversteps our sealed 70/30 boundary (2024-09-25). Per the task
brief, parity was run **only on the study's IS portion** (2021-01-03 → 2024-02-29 08:00 UTC,
computed as `int(n_h4*0.6)` — verified identical cutoff for USD_JPY and EUR_USD), entirely
inside our sealed IS window. The comparison target is therefore the study's **IS-loVol**
number (`+4.00p`, n=434, per-pair table extracted by re-running the script), not the OOS
headline.

**Parity run (new harness, as-built, `arm="signal"` — VREL_MAX=1.16 is baked into the frozen
params so "signal" already *is* the loVol-only subset; gross and net-of-real-BA-spread,
carry excluded, `spread_mult=1.0`, same IS window, one pair at a time on Hetzner):**

| Pair | orig n | orig net | new n | new net | new WR | Δn | Δnet |
|---|---|---|---|---|---|---|---|
| USD_JPY | 36 | +0.14 | 36 | +10.75 | 58% | 0% | +10.6 |
| EUR_JPY | 36 | -3.97 | 45 | -24.07 | 38% | +25% | -20.1 |
| GBP_JPY | 61 | +8.28 | 56 | -12.37 | 50% | -8% | -20.7 |
| AUD_JPY | 39 | -10.18 | 39 | -9.15 | 49% | 0% | +1.0 |
| CAD_JPY | 33 | -8.67 | 39 | -2.52 | 49% | +18% | +6.2 |
| CHF_JPY | 42 | -2.94 | 40 | -18.20 | 48% | -5% | -15.3 |
| NZD_JPY | 40 | -2.58 | 35 | -10.67 | 40% | -12% | -8.1 |
| EUR_USD | 33 | +17.11 | 25 | +3.64 | 60% | -24% | -13.5 |
| GBP_USD | 40 | +22.43 | 37 | -9.17 | 46% | -8% | -31.6 |
| AUD_USD | 34 | +2.96 | 32 | -3.59 | 44% | -6% | -6.6 |
| EUR_GBP | 9 | -1.74 | 14 | -0.81 | 43% | +56% | +0.9 |
| NZD_USD | 31 | +23.55 | 27 | +28.09 | 74% | -13% | +4.5 |
| **Portfolio** | **434** | **+4.00** | **425** | **-6.03** | **49.2%** | **-2.1%** | **-10.03** |

Tolerance (expectancy ±0.5p, count ±5%): **FAILS** on the portfolio aggregate and on most
individual pairs (several sign flips: GBP_JPY, GBP_USD; count deltas >15% on EUR_JPY, EUR_GBP,
EUR_USD).

**Diagnosis.** Isolated one variable at a time. Swapped ONLY the H4 bar-aggregation function
(all other harness logic — entry condition, ATR, position management, cost model — held
fixed) from `bars.m5_to_h4` (NY-17:00 OANDA anchor) to a naive UTC-midnight
`pandas.resample("4h")` — i.e. exactly what `first_touch_v2.py`'s `load()` does (it
`pd.to_datetime(..., utc=True)`s the timestamps and calls `.resample("4h")`, which bins at
00/04/08/12/16/20 **UTC**, not NY 17:00). Result: portfolio net_ex_carry **+2.61p (n=506)** —
closes ~86% of the gap toward the original's +4.00p (vs the NY-anchor run's -6.03p). **The H4
bar-boundary convention is the dominant cause of the divergence, not a harness bug.**

**Bigger finding, surfaced only by this exercise:** which bar grid does the *live* paper
service actually consume? `services/strategy_first_touch_paper/main.py:223` calls
`adapter.get_candles(pair, count=BUF+1, granularity="H4")` — i.e. it reads OANDA's own
native H4 candles, which OANDA anchors at **NY 17:00** (the same convention `bars.py`
deliberately implements, per R6, and documents at length). **The 2026-06-18 backtest study
that validated the frozen parameters (`first_touch_v2.py`) used a bar grid — UTC-midnight —
that matches neither OANDA's real H4 candles nor the live paper service it was meant to
validate.** This is a previously-undetected R6/R9 gap in the *original* study, not in the new
harness. The new harness's bars.py is *more* live-consistent than the reference it's being
checked against. Regressing it to match the original's ad hoc UTC anchor would reintroduce a
real bug purely to pass a numeric parity check — not done.

**Residual gap** (even after matching the anchor: +2.61p/n=506 vs +4.00p/n=434) is consistent
with three other deliberate, documented, pre-registered harness refinements absent from the
quick research script, not further chased down to the decimal: (1) entry fill at the next M5
bar's open after the H4 close vs. the original's same-bar H4-close entry (no live-achievable
same-bar fill); (2) FIFO position management that reopens the signal search immediately after
an early TP/SL exit vs. the original's blanket `Hcap`-bars-from-*entry* block regardless of
actual exit timing (this alone plausibly explains the new harness's higher trade count under
the UTC-anchor diagnostic, 506 vs 434); (3) ATR computed on the harness's bounded ~55-bar
rolling buffer vs. the original's unbounded full-history pandas EWM.

**Verdict: FAIL** on the literal tolerance, against the harness as actually built (NY-17:00
anchor, the correct/live-matching choice). Root cause is fully diagnosed and is **not a
harness defect** — no code fix applied, none warranted. **Blocking: A5 does not proceed.**
The frozen parameters (L/EPS/VW/**VREL_MAX=1.16**/grid-selected TGT=SL=2.0/HCAP=12) were all
fit on the original study's non-OANDA UTC-anchored H4 series; porting them verbatim onto the
harness's OANDA-correct NY-17:00 series is not guaranteed to preserve their statistical
properties (touch counts, ATR scale, and especially the volume-ratio distribution that
`VREL_MAX` was calibrated against all shift with the bar grid). Two options for whoever picks
this up next: (a) re-run the original study's own IS-selection (incl. re-deriving
`VREL_MAX` as the IS-median `vrel`) from scratch on OANDA-correct NY-17:00-anchored H4 bars
before trusting the frozen params at all; or (b) treat `fx-first-touch-paper`'s own
accumulating forward trades (it already reads correct OANDA H4 candles) as the reference to
validate the harness against, instead of the flawed backtest. Either way, the pre-registered
frozen-parameter table above should be treated as **unconfirmed** pending a redo, not as a
settled prior.

## Amendment 1 (2026-07-06, after A4, BEFORE any IS gate-3+ analysis or OOS exposure)

**Gate-2 disposition.** Strict tolerance FAIL, but the divergence is fully diagnosed and is a
defect of the ORIGINAL study, not the new harness: `first_touch_v2.py` binned H4 bars with a
naive UTC-midnight `resample("4h")`, while the deployed paper service (and this harness) use
OANDA's true NY-17:00 anchor. Replicating the wrong anchor closes ~86% of the gap; the original
script itself was re-run verbatim and reproduced exactly. The harness is therefore
**consistency-proven** (the R7 purpose of gate 2), and the gate is recorded as
SATISFIED-IN-PURPOSE / FAILED-IN-LETTER.

**Consequences, locked before proceeding:**
1. The +9.49p/MC-0.018 prior is **downgraded**: it was measured on a bar grid the live service
   never traded. The first-touch edge on the correct grid is OPEN, not established.
2. The frozen parameters are UNCHANGED — they are the deployed service's parameters and the
   correct grid is the one that service trades; A5 tests exactly the deployed configuration.
3. Ops note: `fx-first-touch-paper` (VPS) has been running on a validation basis now known to be
   grid-inconsistent; its paper results should be read as primary evidence, not confirmation.
4. A5 proceeds on IS only (gates 1,3-6). Early uncontrolled read on the correct grid is negative
   (portfolio ≈ −6.0p IS net) — recorded here to pre-empt any temptation to re-frame later.

## A5 result (2026-07-06) — Gates 3-6 ALL FAIL, on Hetzner, controlled battery

Full IS battery run: 12 pairs × 3 arms (signal / coin seed=20260706 / continuation), 1,518
trades, IS window only (hard-guarded, `is_data.load_pair_is()`). Runner scripts +
`results/is_summary.md` + per-pair/per-arm CSVs committed under this directory.

| Gate | Result | Detail |
|---|---|---|
| 3 (net>0 & >coin) | **FAIL** | signal −7.31p/trade vs coin **+3.52p/trade** — signal is worse than its own null |
| 4 (WF 3 thirds) | **FAIL** | only 1/3 positive: +1.50, −10.79, −9.87 |
| 5 (day-block bootstrap) | **FAIL** | P(net≤0)=0.965 (2000 resamples) |
| 6 (breadth ≥6/12) | **FAIL** | 4/12 pairs gross-positive |

Secondaries (exploratory, IS-only): CSI StrengthSpread H4/64-bar port −23.84p/leg (93
rebalances); D1 RSI(2) classic −16.31p/trade (n=1,631, 5/12 pairs gross-positive); equal-risk
portfolio (c) not constructed — none of the three candidates was IS-positive at base cost.

**Verdict: confirms Amendment 1's early uncontrolled read with the full controlled gate
battery.** Per the pre-registration's decision rule, IS gates failing stops the program on
this frozen-parameter configuration. **OOS remains sealed** — not opened on this run, per
R8/gate 7 (user gate never reached because gates 3-6 already failed). Any future attempt on
this hypothesis needs a fresh pre-registration (new parameters or new signal), not a re-run
of this one under a relaxed gate.
