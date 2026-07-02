# ESCMA Revival — Experiment Roadmap (2026-06-12)

User-directed reopening of the ESCMA exit-learner after the v2 NEGATIVE verdict.
Execute phases **in order**; each builds on the previous.

## Motivation
v2 result (see `v2_verdict.txt`): every config converges to "bail within 1–5 bars",
best raw PnL −3881 vs −3997 naive, **never net-positive** after full spread. Two
explanations to disambiguate:
- **(a) Real no-edge** — post-shock reversion magnitude < round-trip spread (1.71p).
- **(b) Optimization/init artifact** — CMA stuck in the "bail" basin from a flat/zero
  init; a better starting point might find a real ride-and-exit policy.

Plus two new directions: a *different entry* (momentum continuation, not shock-fade),
and eventually *learning entries too* (not just exits).

## Frozen apparatus (all phases)
- **6 inputs:** 5× `tanh(mn_TF / (4.45·MAD_pooled))` (shared IS-frozen pooled MAD,
  sign-exact, bounded, robust) + 1× position = **live AMDDP5 reward** `tanh((u−λ·cum_dd)/20)`.
- **Net** 6→3→1; **activation bank** {sin, cos, tanh, gauss, morlet, ricker} searched
  two ways: **sweep** (one CMA per activation) + **evolved** (per-node softmax logits).
- **Reward** AMDDP5 (λ=5%); rails (disaster SL 100p / runaway TP 200p / time cap).
- **Spread charged UP FRONT** (entry at ask/bid, full round-trip). Seeds 42/7/123.
- **Engine** `train_cma_exit_v2.py` + `@njit(parallel=True) eval_population` (~9× / 9-min runs).

## Phases

### Phase 0 — Sine positive-control (CLEAN) + warm-start  ← START HERE
- **Data:** single clean sine + spread, NO noise. `price = A·sin(2π t/P) + baseline`,
  bid/ask = mid ± spread/2. Same 6 features via the same kernel (R6). Non-JPY scale
  (pip 1e-4): A≈50p amplitude, P≈720 bars, spread≈1.7p.
- **Entries (momentum continuation):** rising zero-cross → LONG, falling zero-cross →
  SHORT; t_timeout = +P/2. **Optimal exit = the next extreme at +P/4** (capture full A).
- **Goal A (positive control):** the exit ESCMA MUST turn a large profit (swings ≈50p ≫
  spread 1.7p). If it *can't*, the harness/reward/init is broken — and every prior
  negative is suspect.
- **Goal B (warm-start):** use sine-best weights as CMA `x0` for the REAL shock-fade run.
  Does it escape the bail basin? **If it still bails on real → v2 negative is real
  no-edge, not optimization.** Decisive either way.

### Phase 0b — Sine + spread + NOISE
- Same sine, add noise calibrated toward real USD_JPY bar-to-bar σ. Tests whether the
  exit policy survives realistic signal-to-noise. Sweep noise level to find **at what
  SNR the rideable edge drops below spread** — an "edge vs noise" curve that bridges the
  clean control (Phase 0) and the real-data negative (v2).

### Phase 1 — Momentum-continuation entries (real data, medium)
- Re-chop REAL entries on aligned positive momentum (`c−c5>0 & c−c12>0 & c−c120>0` → LONG;
  mirror for SHORT) instead of shock-fade. Run the exit ESCMA **warm-started from Phase 0**.
- Question: does momentum-continuation have an edge > spread where shock-fade did not?

### Phase 2 — Two-net entry + exit (continuous sim, big)
- **One ESCMA learns entries** (every bar: open long / short / flat from the 5 mn channels),
  **one learns exits** (incl. SL/TP at the late extreme). Same feature set. Requires a
  continuous walk-forward simulation with position management (heavier than the current
  event-sliced batch engine). Both nets sine-pretrained.

## Success bar (real-data phases)
Net-positive raw PnL after a full round-trip spread, AND beat naive first-neg-tick on OOS
AMDDP5, on a majority of seeds (42/7/123), IS+OOS. (The sine phases use profit-on-clean-
signal as the pass.)
