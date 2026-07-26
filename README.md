# The Edge That Wasn't
### A Machine Built to Tell Me Whether My Edge Was Real, and the Answer It Gave

**Companion repository to the book by Aharon Zbaida (2026)**

> **Companion website: [theedgethatwasnt.com](https://theedgethatwasnt.com)** — the interactive 81-experiment explorer, glossary, indicator viewer, the frozen trade ledger, the broker-reconciled account table, and the retractions page. The book is [available in paperback on Amazon](https://www.amazon.com/dp/B0H8MNYTRF).

---

## About the Book

For fifteen years one developer set out to build a rigorous, professional system for making money in the foreign-exchange market. This book is the record of that attempt, and of the machine built, in the end, to tell the truth about whether an edge had been found.

The subject is retail foreign exchange — currency trading from a laptop, via the OANDA v20 API — but the harder problem is universal. The difficulty is not coming up with ideas; it is knowing whether the thing you have built works, or only looks as though it does. A backtest can be made to show almost anything. A winning streak feels exactly like skill. A smooth, rising equity curve looks like proof — and it is also exactly what a subtle lookahead-bias bug produces. The apparatus in this book was built to keep human judgment out of the scoring.

Eighty-one experiments. Multi-pair walk-forward validation. Monte-Carlo gates. Causal-consistency checks. Realistic fill models. Finite-margin closeout simulation. Three families of machine learning (LightGBM, NEAT neuroevolution, CMA-ES). The answer the apparatus gave is this: intraday directional retail spot FX, across every technique the project tried, does not produce a spread-net edge that survives rigorous validation — and the few strategies that looked large were either statistical illusions, fill-model artifacts, or martingales with unbounded tails. The code in this repository is the evidence base behind that conclusion.

> **Not financial advice.** This repository and the book it accompanies are a record of research and a memoir of method. Nothing here is financial, investment, or trading advice, or a solicitation to trade. Trading foreign exchange carries substantial risk of loss; past results — including every figure shown — do not indicate future performance.

An interactive drill-down of all 81 experiments (search, verdict filters, per-experiment code links and figures) is published as the companion website — see `docs/index.html` and `docs/experiments/index.html`, or `COMPANION_URL.md`.

---

## Code Availability

> The code behind this book's experiments is being released publicly as it is reviewed and prepared. This companion repository contains the curated experiment scripts, library code, and reading guides that illuminate the book's argument, and additional project repositories are copied in as they are cleaned for release. The only permanent exclusions are broker credentials and account configuration.

---

## How to Use This Repository

Each row in the Experiment Map below corresponds to one entry in the book's Experiment Audit Table. The `Path in this repo` column points to the subdirectory that contains the generating script(s) for that experiment. Infrastructure and engineering milestones (entries marked ✅ without a trading edge claim) contain the code artefacts relevant to the text.

You do not need to run the experiments to follow the book's argument. The scripts are here so the numbers can be checked.

---

## Experiment Map

*81 experiments, one row each, verdicts re-derived from the book's Experiment Audit Table (the source of truth). Legend: ✅ positive on its own terms · 🟡 mixed · ⛔ negative · 🔴 one-time survivor, retracted. Key results are rounded to the precision used in the book.*

### Theme — Lookahead bias (the apparatus saw the future)

| # | Experiment | Verdict | Key result | Path in this repo |
|---|------------|---------|------------|-------------------|
| 3 | ASI-MC V3 efficiency-ratio gate | ⛔ | +84,861p OOS / 499 p/d — entirely lookahead artifact | `experiments/asi_mc_v3_er/` |
| 4 | IronNet fixed-topology NEAT | ⛔ | EUR_GBP +3,421p, CAD_JPY +12,287p, MC sign_p=0 — lookahead | `experiments/ironnet_fixed_topology/` |
| 5 | Per-pair IronNet V3, 12 pairs | ⛔ | 12/12 MC-validated, avg 42.0 p/d OOS — all contaminated | `experiments/per_pair_ironnet_v3/` |
| 8 | H1 V3 (EUR_GBP + CAD_JPY) | ⛔ | EUR_GBP +41.2, CAD_JPY +218.1 p/d — lookahead-contaminated | `experiments/h1_v3/` |
| 9 | IronNet V3 H1 12-pair + V7 | ⛔ | V3 178.9, V7 371, S1 747.1 p/d — "entirely artifact" | `experiments/ironnet_v3_h1/` |
| 10 | CMA-NN with sin basis | ⛔ | +73 p/d CHF_JPY — lookahead; clean causal retrain +0.11 p/d | `experiments/cma_nn_sin/` |
| 11 | CMA-NN 12-pair grid + mixed activations | ⛔ | M5 +54.6 / H1 +40.2 p/d, seed-stable — invalidated by RCA | `experiments/cma_nn_12pair/` |
| 12 | RCA — lookahead + MACD bug | ⛔ | 16/16 strategies net-negative live; ≈−55,000 pips; two bugs confirmed | `experiments/rca_lookahead/` |
| 13 | Causal retrain verdict | ⛔ | Best of 16 causal runs +0.11 p/d (13 trades) — zero causal edge | `experiments/causal_retrain/` |
| 20 | StrengthSpread lookahead RCA | ⛔ | merge_asof leaked ≤55min future; +203.15 → −14.18 p/d causal | `experiments/strengthspread_rca/` |

### Theme — Feature-class exhaustion (the indicators are tapped out)

| # | Experiment | Verdict | Key result | Path in this repo |
|---|------------|---------|------------|-------------------|
| 17 | Slot-4 indicator sweep (43 indicators) | ⛔ | 0 of 43 crossed OOS 0 p/d; best (cci) −1.62 p/d | `experiments/slot4_indicator_sweep/` |
| 24 | FIFO-Trends body filter | ⛔ | Zero OOS winners at any threshold, all 12 pairs | `experiments/fifo_body_filter/` |
| 44 | ATR-adaptive P&F box sweep | ⛔ | Converges to ~5 pips; all 4 pairs worse — 5p = noise floor | `experiments/atr_adaptive_pnf/` |

### Theme — Execution realism (the fill model was the edge)

| # | Experiment | Verdict | Key result | Path in this repo |
|---|------------|---------|------------|-------------------|
| 6 | Spread-at-entry fix + Vmax | 🟡 | Corrected V3 s42 failed to converge — realistic spread changes dynamics | `experiments/spread_at_entry_fix/` |
| 23 | FIFO-Trends 12-pair P&F sweep | ✅ | GBP_JPY 71.6, USD_JPY 68.5 p/d OOS — but 1-box trail failed live | `experiments/fifo_trends_pnf_sweep/` |
| 26 | H4 Donchian breakout (backtest) | ✅ | Backtest OOS 16–25 p/d/pair, 84% WR, +41p win / −12p loss — refuted live (see 53: 3 trades, −61.6p, 0% WR, stopped) | `experiments/h4_donchian/` |
| 33 | TR Momentum strategy | ⛔ | Phase-1 12/12 WF, but corrected bar-close fill = 0/12 — fill artifact | `experiments/tr_momentum/` |
| 35 | FIFO-Trends S5 trail resolution | ⛔ | EUR_USD +45.2 → −69.2 p/d; box quantization IS the edge | `experiments/fifo_s5_trail/` |
| 36 | FIFO-Trends bar-close fill correction | ⛔ | WR 70→34%, ΔOOS −54 to −230 p/d, 0/8 configs survive | `experiments/fifo_barclose_fill/` |
| 37 | FIFO-Trends proper live sim (v2) | ✅ | OOS GBP_JPY 135, USD_JPY 139 p/d (70–71% WR) | `experiments/fifo_live_sim_v2/` |
| 38 | FIFO-Live S5 exit monitoring redesign | 🟡 | First deploy −96.5p/3 days (S5 not yet built); redeployed | `experiments/fifo_live_s5/` |
| 45 | GBP_USD small-box P&F sweep | ✅ | b2_r3 OOS 54.52 p/d (vs 15.4 at b=5); 2p trail vs 1.8p spread | `experiments/gbpusd_small_box/` |
| 46 | FIFO 2-box trail sweep (X3c_2_5) | 🟡 | GBP_JPY 35.3, USD_JPY 32.7 p/d — lower p/d but execution-robust | `experiments/fifo_2box_trail/` |
| 49 | No-VPS migration + SMA16 momentum | ✅ | lags=(8,10,15) TP=20p, 10/12 pass, +29.8 p/d OOS, mc_p=0.0000 | `experiments/sma16_momentum/` |
| 50 | Price momentum M15+M5 live | ✅ | lags=(1,3,8) TP=10p, 12/12 pass, +30.4 p/d OOS, mc_p=0.0000 | `experiments/price_momentum/` |
| 53 | H4 Donchian + FIFO stopped (live check) | ⛔ | H4 −61.6p 0% WR; FIFO 15t −168.5p 7% WR — both stopped | `experiments/h4_donchian_live_check/` |

### Theme — Realizable-under-finite-margin risk (the tail the gate missed)

| # | Experiment | Verdict | Key result | Path in this repo |
|---|------------|---------|------------|-------------------|
| 22 | Weekend gap-fill backtest | 🟡 | 92.2% fill, +3.51 p/trade, 12/12 pairs — 450–660p adverse tails | `experiments/weekend_gap_fill/` |
| 27 | Zone Recovery trail-lock redesign | ✅ | P5=162.7 vp/d, P(+)=0.997; live GBP_ABS_125 +381.5p | `experiments/zr_trail_lock/` |
| 28 | ZR ML=4 force-close + CAR-25 | ⛔ | ML=4 −111.9 vp/d vs ML=10 +7,158.6; 96% of edge in 5+ leg cycles | `experiments/zr_ml4_car25/` |
| 55 | Dynamic position sizing | 🟡 | units = max(1, round(bal × 1.25)) — scales hidden risk on a leaky pipeline | `experiments/dynamic_sizing/` |
| 56 | Portfolio variation campaign (8 variants) | 🟡 | Combined +250.4 p/d, mc_p=0.0000 — but 130/130 passed MC (the tell) | `experiments/portfolio_variation/` |
| 57 | Finite-margin re-validation | ⛔ | SL flips +27.3 → −206 p/d (30p stop); 6/6 accounts wiped in simulation | `experiments/finite_margin/` |
| 62 | Money management cannot manufacture edge | ⛔ | 2:1 R:R = 66% WR netting ≈ −spread; all 3 trend framings −spread EV | `experiments/random_vs_trend/` |
| 66 | SMA-Stack robustness (exit + gate) | 🔴 | Conservative 1.5yr backtest: live-freq-matched net −3,616p; no fence setting positive; CAR25 negative at every position size | `experiments/conservative_010/` |

### Theme — Spread floor (real signal, below the toll)

| # | Experiment | Verdict | Key result | Path in this repo |
|---|------------|---------|------------|-------------------|
| 15 | Deep2 mean-reversion multi-seed | 🟡 | OOS +1.15 p/d (all seeds OOS>IS) — below 2.3p spread, not deployable | `experiments/deep2_mean_reversion/` |
| 16 | CMA-ES + regime gates | ⛔ | Best OOS −2.38 p/d; M5 momentum + binary regime can't beat spread | `experiments/cma_regime_gates/` |
| 18 | Path B (new core) + Path C (H1 cadence) | 🟡 | H1 ceiling −2.2 → −1.82 p/d (18% improvement); all tier-2 still sub-zero | `experiments/path_b_path_c/` |
| 29 | H1 Donchian sweep | ⛔ | 0/2400 H1 configs OOS+ (best −6.37 p/d); spread dominates | `experiments/h1_donchian/` |
| 30 | RACS reversal-accumulator IC study | 🟡 | 72%/59% cells \|t\|>2 (significant) but decile spread ≈1p < spread | `experiments/racs_reversal/` |
| 31 | Currency-strength lower-TF viability | ⛔ | All holds ≤512 bars negative (5min −365.9 p/d); only 10-day+ survives | `experiments/csi_lower_tf/` |
| 34 | S5 velocity/acceleration sweep | ⛔ | IC −0.026 counter-trend, 0/168 WF — indistinguishable from random | `experiments/s5_velocity/` |
| 42 | Grid-trail strategy | ⛔ | v1 0/20 WF (21–34% WR), v2 0/480; FX not random-walk at M5 | `experiments/grid_trail/` |
| 43 | Daily range regime (PDH/PDL) | ⛔ | 0/128 WF, best IS −15 p/d, avg trade −2.87p; MR and breakout both fail | `experiments/daily_range_regime/` |
| 47 | PantasticSMA indicator screen | ⛔ | +1.4 p/d M5 only (below significance); 0/36 P&F configs OOS+ | `experiments/pantastic_sma/` |
| 54 | MTF/wave/wavelet confluence + Harvester | ⛔ | S5 confluence 0 WF; wavelet 0 WF; Harvester 89% WR only at spread<1p | `experiments/mtf_wavelet/` |
| 59 | Oracle optimal-trade ceiling | ⛔ | Perfect-foresight ceiling 2,426 p/day but 49% of optimal trades net < 1 spread | `experiments/oracle_traits/` |
| 65 | Majors lead crosses (lead-lag) | ⛔ | Confirmed major→cross IC +0.07..+0.12 (t≈109–189) but catch-up ≈ −spread | `experiments/lead_lag/` |
| 67 | Structural fade at hourly S/R | ⛔ | Gross +1.09p, p=0, 12/12+, all WF thirds+ — net −0.89p, 12/12 negative | `experiments/spread_band_random/` |

### Theme — Measurement (the apparatus itself can lie)

| # | Experiment | Verdict | Key result | Path in this repo |
|---|------------|---------|------------|-------------------|
| 1 | Project genesis + architecture | ✅ | 49 unit tests passing, curator smoke test green, 4 strategies complete | `experiments/project_genesis/` |
| 2 | JPY exposure bug fix | ✅ | JPY notionals 100–160× overstated; fix raised live positions 13→32 | `experiments/jpy_exposure_fix/` |
| 7 | LightGBM/SHAP indicator ranking | 🟡 | TEC_5 SHAP #1 but CMA #36/43 — SHAP rank and trading edge orthogonal | `experiments/lgbm_shap_ranking/` |
| 14 | S5 momentum NEAT + P&F | 🟡 | OOS +2.74 p/d, +5.75 p/trade vs 2.3p spread — only 18 OOS trades | `experiments/s5_momentum_pnf/` |
| 25 | EUR/USD microstructure / tick-pace | 🟡 | tpm predicts big bars (4× lift) NOT direction — not standalone | `experiments/microstructure_tickpace/` |
| 39 | Dashboard momentum signals tab | ✅ | (engineering) multi-TF momentum/acceleration tab, DuckDB history | `experiments/dashboard_momentum/` |
| 40 | Multiscale Price Shock (MSP) propagation | 🟡 | Timing AUC 0.73–0.88, direction AUC 0.60–0.65; Granger 15/15 p<0.0001 | `experiments/msp_propagation/` |
| 41 | Central trades DB + infrastructure | ✅ | (engineering) write_trade_direct() trade_id dedup; dashboard reads DB only | `experiments/central_trades_db/` |
| 48 | Multiscale shock Phase 10 + joint model | 🟡 | Direction AUC 0.60–0.65 weak; standalone +0.40 p/d not deployable | `experiments/msp_joint/` |
| 60 | Indicator-vs-optimal-direction screen | ⛔ | None of ~70 indicators beat ~53% agreement with oracle direction | `experiments/oracle_traits/` |
| 61 | Continuation vs exhaustion | 🟡 | Signal corr up to 0.18 but uniformly exhaustion — exit signal, not entry | `experiments/oracle_traits/` |
| 63 | Three measurement failures inverting verdicts | 🟡 | Ledger fix GBP_USD +256.8→+286.8p; TP-proxy faked +2.0 vs PSAR −3.4; yen pip 110× too large | `experiments/gbpusd_regime/` |

### Theme — Contrarian-at-structure (the only family that survived)

| # | Experiment | Verdict | Key result | Path in this repo |
|---|------------|---------|------------|-------------------|
| 19 | Path D mean-reversion filter | 🟡 | EUR_JPY +0.21, CAD_JPY +0.17 p/d (all seeds+) — real but sub-tradeable | `experiments/path_d/` |
| 21 | CSI StrengthSpread campaign (stages 1–2) | 🟡 | Stage-2 7/30 all-folds+; top SignedCSI H8 Sharpe 0.752 — Stage 3 incomplete | `experiments/csi_strengthspread/` |
| 32 | P&F momentum vs counter-trend | 🟡 | Counter-trend wins 8/12 pairs but ~+5 p/d, 10–15× weaker than FIFO | `experiments/pnf_counter_trend/` |
| 51 | Post-shock counter-trend retrace live | ✅ | 72 OOS+ configs, IS WF 40/40, MC all mc_p=0.0000, +56 p/d OOS | `experiments/post_shock_retrace/` |
| 52 | Markov D1 regime filter | ✅ | All ICs counter-trend; Phase-3 filter +70.4 p/d, WF 9/12, mc_p=0.0000 | `experiments/markov_d1/` |
| 58 | Indicator screen → momentum×eff → regime-gated MR | ⛔ | Regime-gated +13.0p t=2.25, p=0.025 — retracted at the premise: the Coda's re-run found the daily-MR signal +1.1p gross, negative net of rollover | `experiments/regime_mr/` |
| 64 | FX stat-arb (Avellaneda-Lee eigen-residual) | ⛔ | Residual reversion non-stationary (2022 +0.19 → 2024-25 −0.13); naked t=−1.24 | `experiments/fx_statarb/` |

### Theme — Cost floor, factor premia & the composite (entries 68–81)

*The final program: pre-registered against-drift entries, classic FX factor premia, positioning data, the spread clock, and the book's own recommended composite — each run to its sealed conclusion at real retail cost.*

| # | Experiment | Verdict | Key result | Path in this repo |
|---|------------|---------|------------|-------------------|
| 68 | First-touch H4 low-volume reversion | ⛔ | Recorded OOS +9.49 p/trade (MC p=0.018) — refuted as a bar-grid artifact: on the true NY-17:00 H4 grid, −7.31 p/trade vs coin +3.52 | `experiments/touch_ladder/` |
| 69 | ESCMA exit-learner (shock-entry + CMA-NN exit) | ⛔ | CMA-NN exit converges to "bail in 1–3 bars"; loses to trivial first-red-tick baseline — no exit alpha | `experiments/escma_exit/` |
| 70 | Trailing-stop bake-off (SuperTrend vs PSAR et al.) | 🟡 | SuperTrend per-trade ATR-band ratchet wins IS+OOS (H4 +17.5) and beats PSAR by ~20 p/trade | `experiments/trailing_bakeoff/` |
| 71 | Regime-conditioned entry Phase A (triple-barrier, against-drift) | ⛔ | Gross signal real (12/12 pairs +) but 5–10× below the ECN floor; information PASS / money FAIL | `experiments/regime_entry/` |
| 72 | Multi-day contrarian program + first-touch H4 bar-grid refutation | ⛔ | 0/4 IS gates; the earlier +9.49p first-touch edge was a bar-grid artifact | `experiments/multiday_contrarian/` |
| 73 | SMA-scratch tail-bounding test + restart-censoring discovery | ⛔ | Equity overlay + 3×ATR stop AMPLIFIES drawdown; restart-censoring in the ledger uncovered | `experiments/scratch_tail/` |
| 74 | London-fix fade (16:00 pre-fix drift reversion) | ⛔ | Gross reversion real (fade −1.63p vs coin −1.89p) but net dead at spread | `experiments/fix_fade/` |
| 75 | FX factor suite (carry / value / momentum / composite) | ⛔ | Classic FX risk premia do not survive OANDA retail cost; carry IS net −34.1 p/rebalance | `experiments/fx_factors/` |
| 76 | Momentum confirmation Stage 1 (deep-history 12-1) | ⛔ | The lone +27.4 was a window artifact — FX momentum decayed pre-2013; net −8.25 p/rebalance | `experiments/momentum_confirm/` |
| 77 | CFTC COT positioning-extremes signal | ⛔ | 4/4 IS gates but bootstrap CI spans zero; one-shot OOS −3.2 p/wk over 275 weeks, below the null p95 — refuted, seal spent | `experiments/cot_positioning/` |
| 78 | Spread clock — hour-of-week toll profiles + guard | 🟡 | Spread flat 00–20 UTC, spikes 2–8× at rollover; a live blackout guard would COST, not save | `experiments/spread_clock/` |
| 79 | Core-pricing arithmetic (OANDA core tier re-pricing) | ⛔ | Core (commission+razor) tier is +0.3p WORSE all-in; the real unlock is a ~0.7p ECN venue | `experiments/core_pricing/` |
| 80 | Meta-allocation — equity-MA strategy switching | 🟡 | Causal equity-MA switch nets −1,949p vs −3,792p equal-weight — a defensive risk-reducer, not alpha | `experiments/meta_allocation/` |
| 81 | Composite 1 — D1 displacement fade × COT vote | ⛔ | The book's own Appendix-E composite: conjunction too rare, vote no better than a random gate | `experiments/composite1/` |

---

## Results Summary

*Counted row-by-row from the book's Experiment Audit Table. Legend: ✅ positive on its own terms · 🟡 mixed · ⛔ negative · 🔴 one-time survivor, retracted.*

| Outcome | Count |
|---------|-------|
| ✅ Positive on its own terms (incl. engineering milestones) | 13 |
| 🟡 Mixed / inconclusive / sub-spread | 22 |
| ⛔ Negative — no deployable edge | 45 |
| 🔴 One-time survivor, later retracted | 1 |

The positives are not victories in the usual sense. Several (entries 49, 50, and the four portfolio variants in 56) did not survive finite-margin re-validation once a realistic stop-loss was added; two (23, 37) worked in simulation but failed live execution; and the late-program successes are cost-accounting and risk-reduction findings, not edges (78–80). The once-deployable positives — post-shock retrace + Markov gate (51/52), Zone Recovery at large leg counts (27), selected P&F variants — were all contrarian, all at longer holding horizons, all constrained by the same retail spread toll, and all ultimately wound down. The one account that ended lifetime-positive (+171 pips, account 010) ran the strategy entry 66 records as 🔴 — refuted at real spread and stopped. Nothing on this map is live; the three retracted headline results are published at [theedgethatwasnt.com/retractions.html](https://theedgethatwasnt.com/retractions.html).

---

## How to Run

**Requirements:** Python 3.11

```bash
pip install numpy pandas numba pyarrow lightgbm duckdb python-dotenv v20
```

The complete trade ledger IS included: [`docs/data/trades_snapshot_2026-05-31.csv`](docs/data/trades_snapshot_2026-05-31.csv) — every closed trade (5,556 rows) behind the book's tables.

**Price history not included.** The generating scripts expect OANDA S5 OHLC Parquet files with bid/ask columns. Bring your own data via the OANDA v20 API:

```python
import v20, os
ctx = v20.Context(
    hostname="api-fxtrade.oanda.com",
    port="443",
    token=os.environ["OANDA_API_KEY"]
)
```

Each `experiments/<name>/` directory contains a `README.md` with the specific data requirements and run instructions for that experiment.

**Causal feature builder:** `lib/incremental_features.py` — `FXFeatureBuilder` is the single source of truth for all training exports. Use it; do not reimplement rolling features from scratch.

---

## License

MIT — see `LICENSE`.
