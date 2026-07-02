# NEAT on P&F box-series with AMDDP5 reward — experiment plan

**One line:** evolve a tiny NEAT net that trades the 5-pip/1-box P&F series using two inputs
(signed trend-age, in-trade running AMDDP5), 3 outputs (long/flat/short), reward = AMDDP5.

## Motivation (why this isn't just another NEAT)
This is **"FIFO-Trends entry + a *learned* exit."** FIFO-Trends on exactly these boxes (P&F b=5, r=1,
`research/experiments/fifo_trends/`) had real OOS edge (GBP_JPY +71.6, USD_JPY +68.5 p/d) but **failed
live because the fixed 1-box trail exit was too tight** — the entry edge was real, the exit was wrong.
Letting NEAT manage the exit on the same box series targets that exact failure. Box=5p/rev=1 also gives
moderate trade frequency (boxes form every ~5 pips) — the "action" band the user wants, not daily.

## Data / engine (causal — SOP R1/R4/R4a)
- Build the P&F series with `lib/pnf_engine.py` (canonical), **box=5 pips, reversal=1 box**, on mid OHLC.
- **Uses HIGH and LOW (not just open/close)** — boxes form on bar extremes (up-col: high extends / low
  reverses; down-col: low extends / high reverses). Open seeds the first level; close sets bar direction.
- **Paint ALL boxes a bar traverses.** If one S5 bar's move spans multiple boxes (e.g. a 12-pip bar with
  5-pip boxes = 2+ boxes), paint every box it crosses — trend-age advances by the full count (+N), not +1.
  The engine must LOOP to fill all boxes between the prior level and the bar's extreme (verify
  `pnf_engine._update_pnf` does this; fix if it only steps one box).
- **R2 within-bar sequencing** (the lookahead guard): OHLC can't say whether high or low came first
  intrabar, so apply a fixed order — bull bar → high then low; bear bar → low then high — identical in
  backtest and live.
- **Decisions & fills at BAR CLOSE only — never at an intrabar box level.** Multi-box painting updates
  STATE (trend-age, levels) at bar close; the network sees the new state once, at close, and any
  entry/exit fills at the bar's close price (+ spread). Filling at an intermediate box level that occurred
  mid-bar would assume we knew the intrabar path = lookahead (the FIFO-Trends / TR-momentum bar-close-fill
  lesson). So: paint all boxes for state, act once per bar at close.
- **Source = S5, not M5** (the reason: finer bars have a tiny intrabar range, so the high/low ordering
  assumption barely matters → far more faithful boxes than M5 would give).
- **SINGLE PAIR for the probe: GBP_JPY** (`data/s5_ba/GBP_JPY_S5_BA.parquet`, 5.5y). Chosen because it was
  FIFO-Trends' strongest P&F pair (+71.6 p/d OOS) and the one whose live failure was the 1-box-trail exit —
  the exact thing a learned exit targets; wide ~3.5p spread = stiff test. (USD_JPY avoided — drift flatters.)
  ⚠️ Single-pair = most overfit-prone setup in the project. A winner is a HYPOTHESIS, not an edge — must
  still pass OOS/WF/MC/surrogate-null within GBP_JPY, AND confirm cross-pair before belief or deploy.
- **R4a guard:** features use `prev_col` / `col_hist` (completed columns) only — NEVER `col_count`
  (in-progress column = lookahead). The trend-age input is read AFTER a box completes.
- Spread taken **up front** in unrealized P&L (buy ask / sell bid), SOP R3.

## ⛔ Lookahead elimination — TOP priority, non-negotiable (the project lost ~55k pips to it)
Everything else is secondary to this. Hard gates, all required before any result counts:
- **R1 closed bars only** — the net acts at bar close on completed boxes; never reads bar[i+1].
- **R4a P&F rule** — features use `prev_col` / `col_hist` (completed columns) ONLY; never `col_count`
  (the in-progress column, whose completion is the very bar we'd be acting on = pure lookahead).
- **Paint-all-boxes updates state at close; fills at close price** (above) — no intrabar-path assumption.
- **R6 one code path** — the SAME `process_box(state, bar)→action` function runs in backtest and live;
  zero divergence permitted.
- **R7 consistency test = HARD GATE** — replay N historical S5 bars through the live warmup and assert
  live state == backtest state bit-for-bit (boxes, trend-age, running-AMDDP5, net output) within
  tolerance < 0.001. If it fails, DO NOT TRAIN — find the divergence first. (IronNet died at 47%
  batch-vs-rolling agreement; we gate at ~100%.)
- **R3 mid signals + explicit spread up front**; **R8 OOS sealed** (touched once); **surrogate-null**
  (evolve on shuffled boxes — a real winner must beat noise-evolution).
- Running-AMDDP5 input (#2) is computed only from realized path up to the current closed bar — never the
  trade's eventual outcome.

## Inputs (exactly 2)
1. **Signed trend-age** = `dir × boxes_in_current_direction`. +3 = 3 boxes up; −1 = first box down.
   Scale into NN range: `tanh(signed_age / A)` (A≈5) so it saturates gracefully, keeping sign + early
   magnitude resolution.
2. **In-trade running AMDDP5** = `tanh(running_amddp5 / S)` **clipped to [−0.9, +0.9]** while in a
   position (S = scale const). **Flat sentinel** (no position) = **−1.0**, giving a clean 0.1 margin so
   −1.0 is unambiguously "flat" vs even a deeply-underwater open trade (revises the user's −999, keeps it
   in activation range). AMDDP-flavored analog of v6's UPnL input.

**(Phase 3, CONDITIONAL — only if the 2-input net beats the controls) 3rd input — reversal-level
proximity (ternary, sign = where WE are vs the level):** `0` = not near a historical reversal level;
`+1` = we are ABOVE the nearest level (level is below us → support); `−1` = we are BELOW the nearest level
(level is above us → resistance). Already in [−1,1] (categorical), no scaling. **P&F-native + causal:**
reversal levels = box levels where *completed* columns topped/bottomed (`col_hist`, never the in-progress
column — R4a); proximity = current box within K boxes (K≈1–2) of such a level; sign = current price
above(+1)/below(−1) that level. (Sign convention is arbitrary to the net — chosen to match the user's
"we are above/below" framing; consistency is what matters.) This is
the reversal/order-concentration conditioner (cf. TV-plan §3.6, RACS) tested *inside* the net. Network
grows to 3 inputs → ≤5 hidden → 3 outputs; keep ≤5 hidden (revisit only if it limits).

**Input-range rationale:** both inputs are squashed to ~[−1,1] on the SAME scale so neither dominates and
nothing saturates on entry. Bounding is *required* here (not just good practice) because the activation
suite includes periodic sin/cos + ricker — unbounded inputs alias/wrap through those. One scalar (input 2)
encodes both "in a trade?" and "how is it doing?"; the reserved-extreme sentinel + clip gap is the clean
resolution under the hard 2-input constraint (a 3rd in-trade flag would remove the ambiguity but is ruled
out).

## Network (tiny, capped)
- **2 inputs → ≤5 hidden neurons (hard cap) → 3 outputs** (long / flat / short), argmax (or softmax).
- Hard-cap hidden nodes at 5: reject `add_node` mutations beyond 5 in the genome (neat-python has no
  native cap — enforce in a genome-config subclass / mutation guard). Topology/connections evolve freely.
- **Activations — full spectrum** (genome evolves the mix): **tanh** (squash) · **sin, cos** (Fourier) ·
  **gauss, ricker, morlet, dog, sech, sinc** (wavelet family — localized/transient detectors). Added to
  `lib/fast_eval.py` (additive IDs 7=ricker,8=morlet,9=dog,10=sech,11=sinc). **Exclude sigmoid/relu** (SOP).
  **Why this matters HERE (review of prior work):** the ESCMA exit-learner found **bump wavelets
  {gauss, morlet, ricker} are NECESSARY for "exit-at-the-extreme" decisions — sin/cos/tanh-only FAILED
  (held to timeout, OOS −380).** Since this experiment IS a learned exit, the wavelet bank is required,
  not optional. (MSP: wavelets nailed shock TIMING/magnitude AUC 0.73–0.88, not direction AUC 0.61 → they
  serve the EXIT-timing job; directional entry must come from the trend-age signal.)

## Session bound / max-hold (bounds time-in-trade; prevents loss-deferral)
- **Force-close any open trade at session end** (session ≈ 4–6h, or aligned to FX liquidity so we don't
  hold into thin Asian hours / across Friday close). Caps time-in-trade, matches the action preference,
  and blocks the ESCMA "hold-underwater-forever to defer the loss" degenerate.
- **The boundary close is a REAL scored exit** (market = close − spread), AMDDP5 counted normally.
  **Do NOT null the reward** — nulling would let evolution park losing trades into the boundary to wipe
  the AMDDP5 penalty (a reward-hacking exploit that inflates fitness by hiding losses).

## Reward / fitness
- Per-trade score = **AMDDP5** = `pnl − 0.05·cum_dd` (cum_dd = accumulated underwater pip-bars;
  `research/experiments/amddp5/scorer.py`). Spread deducted up front.
- NEAT is evolutionary: the genome's fitness already *is* the trajectory outcome — "reward every timestep
  equally with the trade's final AMDDP5" = the trade's AMDDP5 contributing to aggregate fitness. No
  per-step RL credit assignment needed.
- **Fitness = min over WF chunks of (mean trade AMDDP5 × n_trades^exp)**, multi-pair, with hard minimum
  trades/chunk (per NEAT SOP). exp∈{0.4..0.7} swept across islands.

## THE key control (ESCMA lesson)
`project_escma_exit` proved **exit-learning alone creates no alpha — entry edge is necessary.** Here the
net learns entry too (trend-age → long/short), so we MUST isolate whether the trend-age input carries
entry edge:
- **Control A — random entry, learned exit:** net only manages exits; entries random. If full-net ≈
  control A, the "edge" is just exit-deferral (the escma trap) → reject.
- **Control B — sine positive-control:** confirm the harness *can* learn when edge is planted (escma
  Phase 0). If it can't learn a known signal, the harness is broken.
- Full net must beat BOTH controls on OOS AMDDP5 to be real.

## Validation (NEAT SOP)
**3-way split (fixes select-on-test with 16 islands):**
- **Train** (~60%) — evolve; 3 walk-forward chunks inside → fitness = min-chunk (robustness).
- **Validation** (~20%, held-out) — used ONLY to rank the 16 island×seed winners, pick the SINGLE
  deployable genome, and drive early-stop. (Selecting best-of-many on OOS would contaminate it.)
- **Test/OOS** (~20%, SEALED) — the one selected genome touched EXACTLY once.
Then MC sign-shuffle (p<0.05) + surrogate-null (same evolution on shuffled boxes; real winner must beat
the noise-evolved best). Single-pair (GBP_JPY) probe; a winner is a hypothesis → **cross-pair confirmation
mandatory before belief or deploy** (then paper). Never deploy a single-pair genome.

## A-priori
Trend-age is a momentum feature → project canon (intraday trend dies to spread) says the *follow* use
likely fails; the net may instead learn a **contrarian/fade use of extended trends** or, more likely, a
**better exit** that rescues the FIFO-Trends entry edge. The AMDDP5 reward + running-AMDDP5 input bias it
toward clean, low-drawdown holds — which is the point.

## Checkpointing — hermetic (resume AND live), no external deps
Lesson from the review: the old NEAT `.pkl` (genome+config+fitness only) is **neither resumable nor
live-reconstructable** — population/species/RNG missing, and the live bot relied on **hardcoded** P&F
constants + an external `neat_config.ini` + implicit custom activations. That hardcoding is a train↔live
drift (and lookahead) vector. MuZero is the model: save everything, resume exactly. Two artifacts:

**(1) Resume checkpoint — every generation (cheap; tiny nets/pop), rolling keep-last-K + best:**
`population` (all genomes), `species_set`, innovation DB (`next_node_id`, `next_conn_id`), `rng_state`,
`generation`, per-island state (island id, migration counters, **exponent** for that island), embedded
`neat_config`, `best_fitness` history.

**(2) Best-genome deploy bundle — the self-contained "NEAT bot". Saved as a NEW versioned file the best
improves AND once per generation (`best_gen{NNNN}.pkl`) — NEVER overwritten** — plus an `all_time_best`
pointer always kept. Each bundle contains:
- `genome` + embedded `neat_config` (num inputs/outputs) — never an external .ini.
- **activation registry**: names + impls for tanh/sin/cos/gauss/ricker (NOT default-serialized — embed the
  name→impl map, or names + a hash of the shared activation module so live verifies identical code).
- **input spec**: ordered input names; normalization constants — trend-age scale `A`, AMDDP5 scale `S`,
  flat sentinel −1.0, clip bounds; (Phase-3) reversal-proximity `K` + level rule.
- **P&F params**: box=5p, reversal=1, paint-all-boxes, R2 convention — embedded, never hardcoded live.
- `amddp_K=0.05`, **IS-only frozen spread gate scalar**, `pip`, `pair=GBP_JPY`, output names (long/flat/short).
- **`code_version`: git commit + hash of `pnf_engine.py` + `process_box` source** (R6); **`r7_pass_hash`**
  (bit-for-bit live==backtest gate — refuse to deploy without it).
- **TRADING-STATS BUNDLE** (attached to every saved best genome):
  - *Per-trade record* (arrays, from amddp5 scorer): AMDDP5, pnl, accumulated_dd, **time-in-trade**
    (hold bars), MFE, MAE, capture_ratio, entry/exit ts, direction. Enables full distribution analysis.
  - *Population-of-trades aggregates*: **Sharpe, Calmar, expectancy, SQN (system quality), profit factor,
    win rate, p/d, trades/day, mean/median AMDDP5, mean dd, mean hold** — IS and OOS separately.
  - *Validation gates*: per-WF-chunk scores, MC p-value, surrogate-null comparison, and a pass/fail flag
    per gate — so any saved genome carries its full provenance for later ranking/deploy decisions.

**(3) Retention / purge — disk-aware (mirror the tick-capture discipline):**
keep `all_time_best` + every WF/MC-validated bundle FOREVER; for the per-gen bests, keep last K gens +
every Mth gen historically; if the checkpoint dir exceeds a size cap, purge oldest non-validated per-gen
bundles first (never the all-time best or any validated bundle). Resume checkpoints: keep last K + best.

## Real-time monitoring — graphs → Telegram (Hetzner is headless)
No GUI on Hetzner, so render with **matplotlib Agg backend** (headless, no display) → PNG → push via the
existing Telegram bot (`notify.py`, `TELEGRAM_BOT_TOKEN`/`CHAT_ID`, `sendPhoto`). The monitor reads the
latest checkpoint's trading-stats bundle (decoupled from training; reconstructable if a server dies).
**Cadence:** every ~10–25 gens (configurable) + a final summary; throttled so it doesn't spam. Each
Hetzner server reports its islands (tagged), or a small orchestrator aggregates.
**Graphs sent:**
1. **Fitness vs generation** — best + mean per island (convergence / plateau / which island leads).
2. **Running-best OOS cumulative AMDDP5** (equity curve) + IS overlay (the overfit gap at a glance).
3. **Per-trade distributions** — AMDDP5, drawdown, time-in-trade histograms.
4. **Aggregate-metric trend** — Sharpe / Calmar / expectancy / SQN of the running best vs generation.
5. **Real-best vs surrogate-null** fitness band — live read on whether we're beating noise-evolution.
6. **Final**: deploy-bundle summary card (IS/OOS metrics, WF chunks, MC p, gates passed) as an image.

## Resources / efficiency
- P&F series is sparse (one step per completed box ≪ M5 bars) → fast eval; numba sim kernel.
- **Generous run (single pair GBP_JPY): 16 islands × 400 pop × 400 gens** + stagnation early-stop (~60
  gens no global improvement), migration every 10–25 gens, diversity = 4 seeds × 4 exponents
  (n_trades^exp ∈ {0.4,0.5,0.6,0.7}). For a 2–3-input/≤5-hidden net the genome space is small, so islands
  ×seed/exponent diversity (robustness) matters far more than raw pop/gens — DON'T exceed ~500 pop / ~500
  gens (diminishing returns; just re-samples). ~2.6M evals × ~0.35s ≈ 250 core-h → ~3–4 h across 4–5
  Hetzner cx53 (~$2–4). Spend extra compute on MORE SEEDS + the **equal-compute surrogate-null** (same
  16×400×400 on shuffled boxes), not bigger pop. Dev/sanity locally first (1 island × 50 × 30, minutes).
  float32 storage, checkpoint every gen (hermetic, above).
- Crash-resilience: checkpoint best genome + generation to disk each gen (today's lesson).

## Build order
0. P&F box engine → causal box-series + signed trend-age, per pair (cache, tiny). Validate R7 consistency.
1. NEAT harness: tiny-net config (≤5 hidden, activation suite), AMDDP5 fitness, WF+MC, the two controls.
2. Run 2-3 pairs IS; if full-net beats both controls on OOS → expand 12 pairs + surrogate-null → paper.
3. **(Conditional on step-2 success) add the ternary reversal-proximity input**, re-run WF+MC+controls,
   KEEP it only if 3-input OOS AMDDP5 beats the 2-input net (don't add complexity for free).
