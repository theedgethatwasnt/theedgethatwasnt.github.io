# SMA-Scratch Tail-Bounding Test — Pre-Registration (LOCKED 2026-07-07)

> Question (user, 2026-07-07): is the sma-scratch paper survivor real, or the condemned momentum
> book wearing a scratch-exit costume? The closed ledger shows +975p/3wk; the open book, marked to
> market, shows ≈ −209p unrealized including a 3-day USD_JPY short at −126p that the scratch exit
> structurally cannot close. Prior art: the finite-margin study (project_validation_gap) — the
> SMA16 base flips +27 → −206 p/d when a stop is added; the equity-MA overlay (bb_equity_switch,
> WF-validated DD-cut 43-86%) has never been tried on this strategy, and its deployed form reads
> CLOSED equity only, which is blind to exactly this pathology.
> Governed by the 10-rule SOP, especially **R7** (parity vs the live paper trail) and **R10**
> (coin-flip null on identical timestamps).

## Data (fixed)

- 6 pairs (the deployed set): USD_JPY, NZD_USD, GBP_USD, CAD_JPY, AUD_USD, GBP_JPY.
- `data/m5_ba/<PAIR>_M5_BA.parquet`, 2020-11-11 → 2026-05-21. H1 and M30 aggregated from M5 mid
  on the **broker grid** (H1 = top-of-hour UTC as OANDA serves it; the live service polls
  granularity H1/M30 directly — aggregation parity with the broker grid is gate 2, per the
  first-touch lesson of 2026-07-06).
- **IS = first 70% (→ ≈ 2024-09-25). OOS = final 30%, SEALED, evaluated once after the user gate.**
- **Disclosed contamination:** the frozen strategy parameters were selected by the H7 sweep
  (2026-06-01) using data inside this range; the *level* of IS results is therefore
  selection-flattered. The primary question is differential — the EFFECT of overlay/stop
  treatments relative to the baseline on identical entries — which is robust to that flattery.
  The OOS one-shot remains the only clean level estimate.

## Frozen strategy parameters (verbatim from services/strategy_sma_scratch_paper/main.py — copied, not fitted)

SMA(16), lags (8,10,15), 6-of-6 agreement on H1+M30; entry at next M5 open; TP +20p;
scratch exit after T_s if |price−entry| ≤ W = k·ATR(14,H1)@entry; USD_JPY quality exit (T_q=2h,
X=3p). Per-pair (T_s bars@M5, k): USD_JPY (576, 0.5) · NZD_USD (288, 1.5) · GBP_USD (144, 1.5) ·
CAD_JPY (96, 0.5) · AUD_USD (144, 2.0) · GBP_JPY (72, 1.5). No other exits in the baseline.

## Treatment arms (pre-declared, exactly these; identical entries across arms)

| Arm | Definition |
|---|---|
| A baseline | scratch as deployed (no overlay, no stop) |
| B closed-overlay | block new entries while closed-trade equity < SMA(10) of itself (deployed monitor convention, trade-sequence sampling) |
| C floating-overlay | block new entries while floating equity (closed + open mark-to-market, sampled at H1 closes) < SMA(24 H1 samples) of itself |
| D floating-overlay + disaster stop | C plus a hard per-trade stop at 3.0 × ATR(14,H1)@entry (broker-side in any live port) |
| E disaster stop only | 3.0×ATR stop, no overlay (attribution control) |

Single pre-declared values everywhere: SMA(10) trades / SMA(24) H1 / 3.0×ATR. **No sweeps.**

## Cost model

Spread: per-trade logged `ask_c − bid_c` at entry and exit M5 bars. Carry: per
`research/experiments/multiday_contrarian/carry_model.py` (positions can be held days/weeks;
the baseline's indefinite carries make carry material). Sensitivity: spread ×{1.0, 1.5}.

## Nulls (R10)

- Coin-flip direction arm (seed 20260707) on identical entry timestamps, for arm A and for the
  primary arm D.
- Overlay-on-coin control: arm C's overlay applied to the coin-arm equity — the overlay must NOT
  manufacture positive expectancy from random trades (it may reduce their drawdown; it must not
  flip their sign).

## Metrics (pre-declared)

Per arm: net expectancy (incl. carry), realized P&L, **worst per-trade open excursion** (MAE
including never-closed positions, marked to window end), **floating-equity max drawdown**,
end-of-window open-book unrealized P&L, trades, % time blocked by overlay, WR.

## Primary hypothesis (confirmatory, one cell)

**H1: Arm D** (floating-NAV overlay + 3×ATR disaster stop — the only treatment that bounds the
per-trade tail *by construction* while the overlay handles regime) achieves, on the sealed OOS:
1. Net expectancy (incl. carry) > 0, day-block bootstrap 95% CI excluding 0; and
2. beats its coin arm, CI excluding 0; and
3. floating-equity max DD ≤ 50% of arm A's on the same window.

Prior to beat: validation_gap says the stop alone (arm E) flips the base hard negative; H1 is the
claim that the overlay rescues what the stop breaks. Arms B/C/E are pre-declared secondaries —
reported, never promoted without a fresh confirmation.

## Gates before OOS (IS-only)

1. Harness self-test on synthetic RW (coin ≈ −(spread+carry); no phantom edge).
2. **Bar-grid + code parity (R7):** replay the live paper window (2026-06-15 → present, from the
   VPS trades.duckdb trail, 132+ closed trades) through the harness; trade-count ±10%, per-trade
   expectancy ±1.0p. Divergences documented before proceeding. FAIL here blocks everything.
3. Arm A IS reproduces the H7 backtest's sign and rough magnitude (documented, given the
   selection-flattery disclosure) AND its open-book pathology (unbounded excursions visible).
4. Arm D IS: all three H1 criteria on IS, plus walk-forward thirds net-positive ≥ 2/3.
5. Overlay-on-coin control clean (no sign-flip on random trades).
6. **User gate**: IS table reviewed; OOS unsealed only on explicit UNSEAL.

## Decision rule

- H1 passes OOS → arm D graduates to a paper A/B against the current deployed scratch (same
  container, second label), minimum 50 trades, before any live discussion.
- H1 fails but the tail metrics confirm the pathology → the paper service is stopped and the
  strategy recorded as the momentum book in costume; keeper = the floating-NAV overlay finding
  (feeds the 010/001 monitor upgrade).
- Either way the open USD_JPY/NZD_USD paper positions get a verdict note in JOURNEY.

## Execution

All computation on an ephemeral Hetzner box (user rule; create → run → collect → DELETE).
Carry/bars/harness reuse from research/experiments/multiday_contrarian/ where applicable
(H1/M30 aggregation added to bars.py with the same closed-bar tests).

## Amendment 1 (2026-07-07, before OOS exposure) — implementation findings

**Discrepancy in the frozen-params prose (harmless, documented while porting).** The live
module's own docstring (`services/strategy_sma_scratch_paper/main.py`, top of file) describes
"USD_JPY quality exit (T_q=2h, X=3p)" — but the actual deployed `CONFIGS` list leaves
`T_q_bars`/`X_pips` unset (`None`) for ALL 6 pairs, including USD_JPY, so the quality-filter
branch is a no-op on the live service today (verified by reading the CONFIGS list directly,
not the docstring prose). The harness (`signal.py`) ports the quality-filter CODE PATH
faithfully (never exercised by these CONFIGS, exactly matching live) and the frozen T_s/k_atr
table in this document was cross-checked against CONFIGS directly (not the docstring) —
matches exactly.

**Design finding: prospective overlay gating is self-referentially unstable (found via gate 1,
fixed before any IS/OOS numbers were touched).** The pre-registration's "block new entries"
language for arms B/C/D is a genuine PROSPECTIVE gate (unlike the currently-deployed
`equity_switch_monitor`/`bb_equity_switch.py`, which only ever paper-tracks a hypothetical
switch and never blocks real orders — a deliberate, necessary widening to actually test H1's
tail-bounding claim, since a purely-retrospective P&L zero-out cannot prevent the specific
pathology, an unbounded open position, this experiment exists to test). Implementing this
literally — computing the overlay's blocked/unblocked state from the SAME (gated) trade
sequence it controls — is self-referentially unstable: the moment it blocks, the gated
sequence stops producing new trades, so its own equity curve freezes, and (using the deployed
monitor's own tie-break convention: ties resolve to blocked) a frozen series is never again
strictly above its own moving average. This is a PERMANENT deadlock, reproduced deterministically
on 100% of gate-1 synthetic-RW runs (arms D/coin_D collapsed to exactly 0 trades before the fix,
regardless of the random walk's realized path or scale).

**Fix.** Each gated arm's blocking SIGNAL is now sourced from a separate, ALWAYS-ON REFERENCE
run (same entry logic, same cost model, same stop configuration, but `overlay='none'`), which
keeps generating fresh trades/equity regardless of whether the gated arm itself is currently
blocked. Declared reference mapping (mechanical, not tuned): B←A (closed-trade equity), C←A
(floating equity), D←E (floating equity — D is "C's overlay on top of E's stop"), coin_D←coin_E
(a new, internal-only ungated coin+stop reference; never itself pre-registered or reported as a
standalone arm), coin_overlay←coin_A (floating equity). The tie-break on the reference lookup
is also flipped from monitor.py's `blocked if value <= MA` to `blocked if value < MA` (strict) —
ties now resolve to UNBLOCKED — since a reference run's own necessarily-episodic equity curve
(H1-sampled, can be genuinely flat for stretches with no closes) needs this to avoid a
secondary, shorter-lived version of the same deadlock. See `harness.py`'s module docstring and
`test_harness.py::test_run_battery_no_arm_deadlocks_to_zero_trades` (regression test) for the
full detail. This is judged a truer reading of "the deployed monitor convention" than a literal
self-referential replica would have been: `monitor.py`'s own `eq`/`ma` are computed from the
REAL, ever-flowing (ungated) live trade sequence, precisely BECAUSE the strategy it observes is
never itself gated by it — the self-reference only becomes possible (and only breaks) once the
overlay is upgraded from observation to genuine gating, which this pre-registration requires.

**Design finding: gate 1's "no phantom edge" check must use full-population (closed + open,
mark-to-window-end) accounting for a no-stop arm, not closed-trades-only.** Arm A has no stop;
its only exits are TP(+20p fixed) and scratch (reachable only once price has wandered back
within W of entry, after T_s_bars). A losing position that drifts away and stays away never
closes on its own. On a FINITE-window random walk this makes the CLOSED-trades-only sample a
biased subsample (skewed toward TP hits + flat scratches; the still-drifting, typically
underwater tail sits in `open_at_end`) — exactly the pathology this pre-registration exists to
characterize (`project_validation_gap`'s "closed = winners by construction"), not a harness
defect. A closed-trades-only gross-mean check FAILS gate 1 on arm A even with zero phantom
directional edge in the harness. The martingale optional-stopping invariant (E[gross]=0 for any
non-anticipating stopping rule) only holds over the FULL population — closed trades' realized
gross-pips UNION open-at-end positions' unrealized (mark-to-window-end) gross-pips — which is
what gate 1 now checks (`test_harness.py::test_gate1_synthetic_rw_no_phantom_edge_arm_a_vs_coin_a`).
A companion regression test documents the closed-only positive bias as an expected, permanent
property of this exit structure, not a result to chase toward zero.

## Gate 2 result (2026-07-07) — **FAIL, on Hetzner, root cause partially diagnosed, BLOCKING**

Replayed the harness (arm A: identical config to the deployed
`services/strategy_sma_scratch_paper/main.py` — no stop, no overlay) over a freshly-fetched
M5-BA window (`fetch_recent_m5_ba.py`, LOCAL fetch since not "heavy": 2026-06-10 → 2026-07-07,
6 pairs, ~5,619-5,620 M5 bars/pair, 8 days of warmup before the live trail's own start),
against the `sma_scratch_%` paper trades in the VPS `trades.duckdb` dump
(`trades_2026-07-06.duckdb`, 112 closed trades — not "132+" as informally estimated in the task
brief; the live window actually observed is 2026-06-18 22:02:38 → 2026-07-06 16:05:54, i.e. the
paper service's own live_since across all 6 pairs, not 2026-06-15 — no trades exist before
06-18 in the dump).

**Portfolio comparison** (harness trades restricted to `entry_ts` inside the live trail's own
observed window, `gate2_parity.py`):

| | live | harness (arm A) |
|---|---|---|
| n closed trades | 112 | 111 |
| count ratio | — | 0.991 (**PASS**, tolerance ±10%) |
| mean net/trade | +8.70p | +4.56p |
| expectancy diff | — | **-4.14p (FAIL, tolerance ±1.0p)** |

**Per-pair** (n live / n harness, mean live / mean harness):

| Pair | n live | n harness | mean live | mean harness | T_s_bars (fast→slow) |
|---|---|---|---|---|---|
| GBP_JPY | 30 | 50 | +11.60p | +3.49p | 72 (6h) |
| GBP_USD | 17 | 24 | +5.13p | **-1.45p (sign flip)** | 144 (12h) |
| CAD_JPY | 20 | **0** | +5.79p | n/a | 96 (8h) |
| AUD_USD | 21 | 16 | +6.08p | +9.16p | 144 (12h) |
| NZD_USD | 15 | 14 | +10.51p | +10.02p | 288 (24h) |
| USD_JPY | 9 | 7 | +15.39p | +11.43p | 576 (48h) |

**Diagnosis attempt.** CAD_JPY's zero is not a harness crash/bug: run standalone over the full
parity window (incl. the 8-day warmup), CAD_JPY produces 8 closed trades total, but the one
active exactly across the live-comparison window (entered 2026-06-16 21:00, held 2,867 M5 bars
≈ 239h, closed 2026-06-30 19:55 on `scratch`) straddles almost the ENTIRE window before the
next position opens and is still open (`open_at_end`) at the fetch cutoff — i.e. the offline
replay gets "stuck" in one long-held position (driven by a very tight ATR-scaled scratch band,
observed ~3.6-4.5p, `k_atr=0.5`) for most of the window, while the live service recorded 20
separate, materially profitable trades in the same span. This means the two runs' POSITION
STATE genuinely diverged well before the comparison window even starts (the harness's single
long hold vs. the live service's many short ones), not merely a same-trade fill-price quibble.

**Pattern across pairs:** divergence magnitude anti-correlates with `T_s_bars` — the
fastest-cycling pairs (GBP_JPY 6h, GBP_USD 12h, CAD_JPY 8h) show the largest count/expectancy
divergence (including one outright sign flip and one collapse to zero), while the
slowest-cycling pairs (NZD_USD 24h, USD_JPY 48h) are close (14 vs 15, 7 vs 9; same sign,
expectancy within a few pips). This is consistent with a SMALL, systematic timing/state
perturbation between the live service and the offline replay compounding over ~2.5 weeks and
~20-60 position-cycles for the fast pairs, while barely affecting the slow pairs that only
cycle a handful of times in the same window. Two candidate mechanisms, NEITHER confirmed nor
ruled out given the task's time budget (documented per R9, not silently guessed):
1. The live service's own `h1[-2]`/`m30[-2]`/`m5[-2]` "one bar of safety margin" indexing in
   `process_pair()`/`warmup()` (`services/strategy_sma_scratch_paper/main.py`) may introduce an
   effective one-poll-cycle lag relative to the harness's "most recently complete bar in the
   historical record" interpretation, shifting entry/exit timing by roughly one bar throughout
   — plausible but not instrumented/proven here.
2. The historical M5-BA candle endpoint (`price="MBA"`, used by both `fetch_recent_m5_ba.py`
   and the original training data) vs. the live service's own real-time polled candles could
   carry small bid/ask/close differences at the bar-boundary microstructure level that this
   task did not cross-check tick-for-tick.

**Verdict: Gate 2 = FAIL** on the literal pre-registered tolerance (count PASSES at 0.99 ratio;
expectancy FAILS at -4.14p vs ±1.0p). Unlike the multiday_contrarian precedent's own gate-2
(A4/Amendment 1), where the root cause was fully isolated to a single, correctable convention
(UTC-midnight vs NY-17:00 H4 anchoring) and the harness was shown to be MORE correct than the
reference it was checked against, this gate-2 divergence is only PARTIALLY diagnosed — a
plausible pattern (fast-cycling pairs diverge more) and two candidate mechanisms are recorded,
but neither is confirmed as the root cause, and CAD_JPY's near-total trade-count collapse in
the offline replay is not adequately explained by either candidate alone. **Per
PREREGISTRATION.md's own decision rule ("FAIL here blocks everything"), the program STOPS
here.** Gates 3-5 / the IS battery were NOT run. OOS remains sealed (was never at risk — gate 2
uses only 2026-06-15+ data, never IS/OOS). The open USD_JPY/NZD_USD/CAD_JPY/GBP_JPY paper
positions currently running on `fx-sma-scratch-paper` get a verdict note in JOURNEY-README
pointing back to this section, per the pre-registration's decision rule ("Either way the open
... paper positions get a verdict note in JOURNEY").

**What would need to happen to unblock:** either (a) fully isolate and fix the root cause
(instrument the live service's actual bar-selection behavior directly, e.g. a short live A/B
replay logging both the live service's internal deque state and this harness's equivalent
state bar-by-bar), reaching the same "SATISFIED-IN-PURPOSE" standard multiday_contrarian's gate
2 reached; or (b) relax/re-derive the tolerance with a fresh justification; either path requires
a new amendment before any IS/OOS analysis is trusted on this harness.

## Amendment 2 (2026-07-07) — parity-failure root cause

**Verdict: CONFIRMED — restart censoring of the live paper trail.** The gate-2 expectancy
divergence is not a harness defect. `services/strategy_sma_scratch_paper/main.py` holds all
position state in an in-memory `PairState` dataclass with **no persistence of any kind**: no
state save on shutdown, no state load on startup. The SIGTERM handler only sets a `_shutdown`
flag; `main()` exits without writing close rows for open positions. On every container
(re)start, `pos_dir` resets to 0 and the service re-enters on the next signal. A trade reaches
`trades.duckdb` **only if it opens AND closes within a single container lifetime** — a
survivorship mechanism: quick winners (TP +20p, flat scratches) always record; long-held
losers are silently abandoned at restarts.

**Restart history across the live trail window (docker logs, laptop `trader` + VPS):**
4 container lifetimes — start 2026-06-18 22:02 (laptop); restart 2026-06-21 15:20; restart
2026-06-30 19:36; migration to the VPS 2026-07-02 18:37 (laptop container shut down 18:43).
Three full state-reset events sit INSIDE the ledger window. (The VPS-only
`docker inspect` shows `RestartCount=0` — misleading, because the pre-07-02 history lives in
the laptop's stopped container.)

**Orphan census (OPEN log lines with no subsequent close line before lifetime end, marked to
the M5-BA mid close at the abandonment timestamp):** **17 orphaned positions** across the 3
resets — 5 at 06-21, 6 at 06-30, 6 at 07-02 — **16 of 17 negative** at abandonment, total
**≈ −591.6p**. Worst: USD_JPY +1 @162.826 (opened 07-01) abandoned at −176p; GBP_JPY +1
abandoned at −102p; GBP_JPY −1 at −95p; GBP_USD −1 at −62p. Every pair contributed orphans.

**This explains all three gate-2 anomalies:**
1. **Gap direction (+8.70 live vs +4.56 harness):** the live ledger drops precisely the
   losing tail. Adding orphans back: (974.4p closed − 591.6p orphaned) / (112+17) ≈
   **+3.0 p/trade** full-population — on the harness's side of the live number, not vice
   versa. The harness's full-population accounting (Amendment 1, gate 1) was already the
   correct frame; the live trail is the biased sample.
2. **CAD_JPY (harness 1 stuck never-exiting position vs 20 live trades):** the live service's
   own stuck CAD_JPY positions were repeatedly ORPHANED at restarts (−20.1p at 06-21, then
   again at 06-30 and 07-02), each time freeing it to re-enter and bank quick wins. The
   harness, with no restarts, shows the strategy's true behavior: one 239-hour hold.
3. **T_s anti-correlation:** fast-cycling pairs re-enter sooner after each reset and are also
   likelier to be holding at any given reset — divergence compounds with cycle count.
   Amendment 1's two candidate mechanisms (h1[-2] indexing lag, MBA endpoint microstructure)
   are hereby demoted to at-most-second-order; neither is needed to explain the gap.

**Disposition:**
- The live paper trail is **NOT a valid R7 parity target** — it is restart-censored. R7's
  premise (the live trail faithfully records the deployed logic's outcomes) is violated by the
  service itself, not by the harness.
- The harness — which passed the gate-1 synthetic-RW self-test (41/41, no phantom edge, full
  population accounting) — is the better description of the strategy. The pre-registered
  program may proceed to gates 3-5 with gate 2 re-scoped to **"count parity only (PASSED,
  0.99) + documented censoring (this amendment)"** — pending controller/user decision; gates
  3-5 were NOT run under this amendment.
- **The famous "+975p closed / 3wk" is doubly flattered**: (a) restart-censoring removed
  ≈ −592p of abandoned losers (full-population ≈ +383p realized-equivalent, ≈ +3.0 p/trade),
  and (b) the surviving open book (4 positions on the VPS as of 2026-07-07) marks to ≈ −190p
  unrealized on top. The sma-scratch keep-running decision must not cite +975p; the honest
  ledger-to-date is ≈ +190p including orphans and open marks — i.e. the strategy is roughly
  an order of magnitude less profitable than its closed ledger claims, exactly the
  `project_validation_gap` "closed = winners by construction" pathology this pre-registration
  set out to test.
- Any future paper service that can hold positions across days MUST persist position state
  (or write an `abandoned` close row on SIGTERM) — otherwise its ledger is unusable for R7.

## Amendment 3 (2026-07-07, before any gate-3-5 numbers were computed) — gate-2 re-scope + service stop

Per Amendment 2 the live paper trail is restart-censored (17 orphaned opens, 16/17 negative,
≈−592p erased across 3 in-window state resets) and is therefore NOT a valid R7 expectancy target.
Gate 2 is re-scoped, before any arm results exist, to: trade-count parity (PASSED, 111 vs 112,
±10%) + documented censoring mechanism (this amendment). The harness — validated by the RW
self-test and count parity, and free of restart censoring — is adopted as the authoritative
description of the strategy. Gates 3-5 proceed on it. Separately, fx-sma-scratch-paper was
STOPPED (2026-07-07): its recorded ledger is structurally censored and its future output would
be equally so; the user's standing keep-if-worth-keeping condition fails on corrected numbers
(≈+3.0 p/trade full-population, unbounded tail, vs the flattered +8.70).

## Gates 3-5 result (2026-07-07) — **H1 FALSIFIED on IS. Program STOPS. OOS not requested.**

Full IS battery run on Hetzner (6 pairs, IS = 2020-11-11 → 2024-09-25, `run_is_battery.py` →
`compute_gates.py` → `make_summary.py`; full detail/tables in `results/is_summary.md`).

- **Gate 3 PASS.** Arm A reproduces both the sign (net +3.71p/trade, WR 64%, 849 trades) and the
  open-book pathology this whole test exists to characterize: 6 never-closed positions marked
  **-12,420p unrealized** at IS end, worst single open excursion **-7,100p** (GBP_JPY). The
  unbounded tail is real and large on 4 years of IS data, not a live-anecdote artifact.
- **Gate 4 FAIL, on all four sub-criteria, not narrowly.** Arm D (floating-overlay + 3×ATR stop):
  day-block-bootstrap net **-2.84p/trade, 95% CI (-3.67, -2.03) — negative and CI-excludes-zero
  on the wrong side**; does **not** beat coin_D (diff CI (-1.34, +0.80), spans zero — D is
  statistically indistinguishable from, and numerically worse than, its own coin-flip control);
  floating maxDD **2.3× arm A's** (-33,320p vs -14,259p) — the *opposite* of the ≤50% claim; WF
  thirds **0/3 positive** (-1.98p / -3.16p / -3.23p, monotonically worsening). The stop does not
  bound the tail — it converts one rare catastrophic hold into ~14× more trades (11,666 vs 849)
  that individually lose ~3p net each, net negative and MORE volatile in floating terms, not less.
- **Gate 5 FAIL, narrowly.** Overlay-on-coin control: no sign flip (coin_A +2.54p vs coin_overlay
  +2.17p, both positive — attribution clean, the overlay is not manufacturing edge from noise),
  but floating maxDD is ~5.8% worse than plain coin_A (-18,941p vs -17,905p, tolerance was ≤5%) —
  a mild, not material, DD violation.
- **Arm E (stop-only, no overlay)** replicates the validation_gap prior exactly: adding the bare
  3×ATR stop alone flips arm A's +3.71p/trade to **-2.47p/trade** (n jumps 849→20,290) — the
  overlay in arm D does not rescue this; it fails to even fully re-close the gap.

**Verdict: STOP. OOS is not requested.** Per the decision rule's second branch ("H1 fails but the
tail metrics confirm the pathology → the paper service is stopped and the strategy recorded as
the momentum book in costume"): gate 3 confirms the pathology is real, gate 4 falsifies the
proposed fix outright (not a close call — every one of the 3 H1 criteria fails, several by a
large margin, one in the wrong direction), and gate 5's mild DD miss adds no reason to reconsider.
`fx-sma-scratch-paper` remains stopped (Amendment 3). Keeper findings: (1) the tail-bounding
question itself — TP-only/no-stop scratch exits accumulate an unbounded, eventually-dominant
open-book tail, reproduced cleanly on IS data; (2) a bare disaster stop does not rescue it (confirms
`project_validation_gap`'s no-SL-book finding on a second, independent strategy); (3) the
floating-NAV overlay does not rescue the stop either — on this configuration it makes the
floating-drawdown outcome worse, not better, while failing to restore positive expectancy;
(4) the self-referential-overlay-gating fix (Amendment 1) and the full-population no-phantom-edge
accounting (gate 1) are durable, reusable harness-design findings independent of this verdict.
