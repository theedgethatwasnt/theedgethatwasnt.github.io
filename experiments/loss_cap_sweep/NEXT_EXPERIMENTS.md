# Loss-cap study — brainstorm of next experiments

The scratch-exit rule + composite (scratch + early-MFE quality filter)
established a real IS+OOS-confirmed edge on 3 of the 10 SMA pairs.
Several user-suggested mechanisms remain untested. Each is a focused
backtest with the same IS/OOS protocol on the 3 winning pairs (where
we know the data has signal) AND the 7 currently-bad pairs (to see
if any can be rescued).

---

## H2 — MFE-stagnation exit

**Idea.** "If a trade has been making progress but the progress has
*stalled* — MFE hasn't grown in N hours — get out, the move is over."

**Rule.** Track `bars_since_mfe_updated`. If it exceeds threshold S
(in bars), exit at next M5 close.

**Backtest design.**
- Grid: S = {12, 24, 48, 72, 144} bars (1h, 2h, 4h, 6h, 12h of stagnation)
- Apply on top of scratch overlay; compare to scratch-alone
- IS/OOS split, all 10 pairs

**Hypothesis to confirm/reject.** Stagnation indicates the move's
momentum has dissipated; exiting locks in current MFE rather than
waiting for either TP (which won't come) or scratch (which might
take much longer if price drifts further adverse first).

---

## H3 — MAE / MFE ratio gate at checkpoint

**Idea.** "If by hour T the adverse excursion is bigger than the
favorable, the trade is structurally bad — exit."

**Rule.** At T_check bars, if `abs(MAE) > MFE * ratio`, exit at close.

**Backtest design.**
- Grid: T_check = {0.5h, 1h, 2h, 4h, 8h}, ratio = {0.8, 1.0, 1.5, 2.0}
- ratio < 1 = exit if MAE is small relative to MFE (lenient)
- ratio > 1 = exit only if MAE dwarfs MFE (strict)
- Compare to scratch-alone and composite

---

## H4 — Initial velocity / direction-check

**Idea.** "In the first hour, price must move *some* pips favorably,
or the signal didn't really fire — exit immediately."

**Rule.** At t = 60 minutes (12 M5 bars), if MFE < V pips, exit.

**Backtest design.**
- Grid: V = {1, 2, 3, 5, 8} pips
- This is a *strict* version of the existing T_q quality filter
  (T_q=1h with X=V); already partially covered by composite backtest.
- Test on the 7 currently-bad pairs to see if any can be rescued.

---

## H5 — Anti-pyramiding: skip new signals when prior didn't develop

**Idea.** "If the last signal on this pair didn't develop in the first
few hours, the regime is noisy — wait before re-entering."

**Rule.** After a quality-filter exit (early MFE < X), set a cooldown
of N hours during which new same-pair signals are ignored.

**Backtest design.**
- Grid: cooldown = {2h, 6h, 12h, 24h}
- Pair this with composite rule

---

## H6 — Pair-specific signal calibration

**Idea.** The strategy uses identical lags (8, 10, 15) on all pairs.
Maybe per-pair lag optimization improves edge enough that the
quality filter becomes additive on more pairs.

**Backtest design.**
- For each pair, sweep (lag1, lag2, lag3) where lags in {3, 5, 8, 10,
  15, 20, 30}, choose IS-best, validate OOS
- Per-pair best lags + per-pair scratch + per-pair quality filter

**Caution.** This is high-degrees-of-freedom and likely curve-fits.
Run with caution and stricter OOS gate (e.g., require IS+OOS+ on
*both* halves of OOS if we split it into two further sub-windows).

---

## H7 — Use ATR to scale scratch window

**Idea.** Current scratch W is fixed pips. But pairs and regimes have
different volatility — a 10-pip window means different things in low
vs high vol.

**Rule.** W = k × ATR(14, H1) at entry. Vary k.

**Backtest design.**
- Grid: k = {0.3, 0.5, 0.7, 1.0, 1.5}
- Compare to fixed-W scratch on the 3 winning pairs
- Try on the 7 currently-bad pairs (maybe their failure was wrong-W
  not no-edge)

---

## H8 — Closeout-aware portfolio (real-money safety net)

**Idea.** The 3-pair scratch overlay is bounded per pair, but
correlated drawdowns across the 3 could still threaten a small
research account.

**Test.** Run the per-pair scratch overlay through a closeout-aware
simulator with realistic margin requirements at $50 NAV. Does the
portfolio survive 2026's worst weeks?

---

## H9 — Multi-strategy ensemble vote

**Idea.** Only enter when SMA16 signal + a different signal (e.g.,
ADX trend strength > X) agree. The hypothesis: agreement reduces
false signals enough to make quality filter unnecessary.

**Backtest design.**
- Pre-compute ADX(14) on H1. Filter SMA16 signal to fire only when
  ADX > threshold.
- Threshold grid: {15, 20, 25, 30}
- Compare to scratch-alone

---

## Priority order (best research:cost ratio)

1. **H2 (stagnation exit)** — directly addresses the user's "let it
   ride while moving, get out when stalled" intent.
2. **H7 (ATR-scaled scratch)** — might rescue the 7 currently-bad
   pairs by adjusting scratch W to local volatility.
3. **H4 (initial velocity filter)** — tighter version of what we
   already tested; quick check on the 7 bad pairs.
4. **H9 (ADX filter)** — entry filter rather than exit; might be
   complementary.
5. Lower priority: H3, H5, H6, H8.
