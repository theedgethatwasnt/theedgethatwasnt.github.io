# Experiment: Heiken Ashi as NEAT Input Signal

## Hypothesis

Closed Heiken Ashi (HA) bars encode trend direction in a noise-filtered representation.
A NEAT network receiving HA direction (+1/-1) may learn profitable entry/exit timing.

---

## Staged Approach: Simple to Complex

### Stage 0: Quick Feasibility Scan (before committing resources)

**Goal**: Determine if HA direction has ANY predictive value. Kill the idea fast if not.

**Method**: No NEAT training. Pure statistical test on historical data.
- Compute HA bars on S5/M5/H1 for all 12 pairs
- Measure: after HA color flip, what happens in the next N bars?
  - `avg_return(next 10 bars | bullish flip)` vs `avg_return(next 10 bars | bearish flip)`
  - `avg_return(during streak | bullish)` vs `avg_return(during streak | bearish)`
- If no separation > 0.5 pips after spread: STOP. HA has no edge.

| Item | Count | Time | Where |
|------|-------|------|-------|
| Script: `stage0_feasibility.py` | 1 | 30 min to write | Local |
| Runs: 3 TFs x 12 pairs | 36 | 5 min total | Local |
| **Total** | | **~35 min** | |

**Pass criteria**: Conditional return spread > 0.5 pips on at least 2 timeframes.

---

### Stage 1: Minimal NEAT — Quick Directional Scan (4 experiments)

**Goal**: Test if NEAT can learn anything from HA direction alone. Use SHORT training
(50 generations, 2 islands) on 1 pair to get directional signal fast.

**Setup**:
- Inputs: 2 — `[ha_dir, UPnL]`
- Training: **1 pair only** (EUR_JPY — most liquid JPY cross, best historical results)
- Training data: 70% IS
- Generations: 50 (not 200 — just enough to see if fitness climbs)
- Islands: 2 (not 4)
- **No Hetzner** — run locally, ~15 min each

| ID | Variant | Direction | TF | Outputs |
|----|---------|-----------|-----|---------|
| S1-1 | V1 | Long | M5 | 2 (BUY, CLOSE) |
| S1-2 | V2 | Short | M5 | 2 (SELL, CLOSE) |
| S1-3 | V3 | Both | M5 | 3 (BUY, SELL, CLOSE) |
| S1-4 | V3 | Both | H1 | 3 (BUY, SELL, CLOSE) |

| Item | Count | Time | Where |
|------|-------|------|-------|
| Experiments | 4 | ~15 min each | Local |
| **Parallelizable**: 2 at a time (8 cores) | | ~30 min | |
| **Total** | 4 | **~30 min** | Local |

**Pass criteria**: At least 1 variant shows positive OOS P/L on EUR_JPY with > 20 trades.
**Decision**: If all 4 show flat/negative fitness curves — STOP. HA direction alone is insufficient.

---

### Stage 2: Full Training — Winning Variants Only (up to 6 experiments)

**Goal**: Train winning variant(s) from Stage 1 with full rigor on 4 training pairs.

Only variants that passed Stage 1 proceed. Worst case: 0 (stop). Best case: all 3 variant
types x 2 best TFs = 6 runs.

**Setup**:
- Training pairs: EUR_JPY, GBP_USD, USD_JPY, GBP_JPY
- Generations: 200
- Islands: 4 per server
- Hetzner: 4x cx53

| Item | Count | Time | Where |
|------|-------|------|-------|
| Experiments | up to 6 | ~2.5 hrs each | Hetzner |
| **Parallelizable**: 4 servers, so 4 at a time | | ~1 batch of 4 + 1 batch of 2 | |
| **Total** | up to 6 | **~5 hrs Hetzner** (~$2) | Cloud |

---

### Stage 3: Validation — OOS + WF + MC (per winning genome)

**Goal**: Full validation pipeline on each genome from Stage 2.

- OOS: last 30%, all 12 pairs
- Walk-Forward: 3 splits (60/40, 50/50, 70/30)
- Monte Carlo: 10,000 shuffles
- Safe-f computation
- Per-pair profitability check

| Item | Count | Time | Where |
|------|-------|------|-------|
| Validation per genome | up to 6 | ~20 min each | Local |
| **Total** | up to 6 | **~2 hrs** | Local |

**Pass criteria**: See `research/VALIDATION.md` (Sharpe >= 0.8, WF 3/3, MC p < 0.05, all pairs profitable).

---

### Stage 4: Enhanced Inputs (only if Stage 3 validates at least 1 genome)

**Goal**: Test if adding more HA features improves the validated base.

Candidate additional inputs (add one at a time, not all at once):

| Feature | Description | Why it might help |
|---------|-------------|-------------------|
| `ha_streak` | Consecutive same-color bars, `tanh(count/10)` | Trend strength |
| `ha_body_ratio` | Body / (high-low) range | Trend quality |
| `MC(D)` | Existing MTF momentum consensus | Combine HA timing with momentum |
| `strength_diff` | Existing Kalman strength | Combine HA timing with fundamentals |

Each addition = 1 training run (same winning TF + direction from Stage 2).

| Item | Count | Time | Where |
|------|-------|------|-------|
| Experiments | up to 4 | ~2.5 hrs each | Hetzner |
| **Parallelizable**: all 4 at once | | 1 batch | |
| **Total** | up to 4 | **~2.5 hrs** (~$1) | Cloud |

Then validate each, compare to Stage 3 baseline.

---

### Stage 5: Integration as Ensemble Expert (only if Stage 4 improves on Stage 3)

**Goal**: Add validated HA genome as 8th expert in the ensemble gate.

- No new training needed for the gate (it already handles N experts)
- Add HA expert to the gate pkl or as a separate feed
- Shadow test on FX-Core for 5 trading days

| Item | Count | Time | Where |
|------|-------|------|-------|
| Integration code | 1 | ~1 hr | Local |
| Shadow validation | 1 | 5 trading days | FX-Core |

---

## Total Resource Budget (worst case, all stages)

| Stage | Experiments | Local Time | Hetzner Time | Hetzner Cost |
|-------|-------------|------------|--------------|-------------|
| 0: Feasibility | 1 script | 35 min | 0 | $0 |
| 1: Quick scan | 4 | 30 min | 0 | $0 |
| 2: Full training | 6 | 0 | 5 hrs | ~$2 |
| 3: Validation | 6 | 2 hrs | 0 | $0 |
| 4: Enhanced | 4 | 30 min | 2.5 hrs | ~$1 |
| 5: Integration | 1 | 1 hr + 5d shadow | 0 | $0 |
| **TOTAL** | **up to 22** | **~4.5 hrs local** | **~7.5 hrs cloud** | **~$3** |

## Early Kill Points

The experiment can be killed at 3 points, saving all downstream resources:

| Kill Point | Condition | Savings |
|------------|-----------|---------|
| After Stage 0 | No return separation > 0.5 pips | Save Stages 1-5 (~$3 + 4 hrs) |
| After Stage 1 | No variant shows positive OOS | Save Stages 2-5 (~$3 + 3 hrs) |
| After Stage 3 | No genome passes full validation | Save Stages 4-5 (~$1 + 1 hr) |

---

## Parallelization Summary

| Stage | Sequential Time | Parallel Time | Speedup |
|-------|----------------|---------------|---------|
| 0 | 35 min | 35 min (1 script) | 1x |
| 1 | 60 min (4x15) | 30 min (2 at a time, 8 cores) | 2x |
| 2 | 15 hrs (6x2.5) | 5 hrs (4 servers, 2 batches) | 3x |
| 3 | 2 hrs (6x20min) | 40 min (3 at a time) | 3x |
| 4 | 10 hrs (4x2.5) | 2.5 hrs (4 servers, 1 batch) | 4x |

**Critical path**: Stage 0 (35 min) -> Stage 1 (30 min) -> Stage 2 (5 hrs) -> Stage 3 (40 min).
**Best case to first validated genome: ~7 hours from start.**
**Best case to kill a bad idea: 35 minutes (Stage 0).**

---

## HA Computation

```python
def heiken_ashi_incremental(o, h, l, c, prev_ha_o, prev_ha_c):
    ha_c = (o + h + l + c) / 4
    ha_o = (prev_ha_o + prev_ha_c) / 2
    ha_h = max(h, ha_o, ha_c)
    ha_l = min(l, ha_o, ha_c)
    return ha_o, ha_h, ha_l, ha_c

ha_dir = +1.0 if ha_c >= ha_o else -1.0
```

For S5: HA computed directly on S5 OHLC.
For M5: Resample S5 to M5, then compute HA.
For H1: Resample S5 to H1, then compute HA.

---

## Files

```
research/experiments/ha/
  PLAN.md                      # This file
  stage0_feasibility.py        # Stage 0: statistical test
  stage1_quick_scan.py         # Stage 1: local NEAT quick runs
  stage2_training.py           # Stage 2: full Hetzner training
  stage3_validate.py           # Stage 3: OOS + WF + MC
  neat_config_2out.ini         # NEAT config: 2 inputs, 2 outputs
  neat_config_3out.ini         # NEAT config: 2 inputs, 3 outputs
  results/                     # All outputs
    stage0_results.csv
    stage1/
    stage2/
    stage3_validation/
```
