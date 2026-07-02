# Filter Gate Experiment Plan
**Date**: 2026-04-05  
**Goal**: Determine whether autoresearch-discovered entry filters improve or degrade performance of existing trained NEAT genomes.

---

## Background

Karpathy autoresearch (1,272 experiments) found these entry gates consistently improve ASI-MC base strategy:

| Filter | Sharpe | Trades | Mechanism |
|--------|--------|--------|-----------|
| BB squeeze | 5.47 | 51 | Only trade during volatility compression |
| Spread/ATR < 0.5 | 4.48 | 656 | Skip low-liquidity / wide-spread bars |
| ER14 < 0.5 | 4.14 | 161 | Only enter in ranging markets |
| RangePosition extreme | ~4.0 | ~600 | Only enter at price extremes of recent range |

Feature IC study (12 pairs, 5 horizons) confirms RangePosition is the #1 contrarian signal (mean abs IC = 0.022 across all pairs).

---

## Target Genomes

| Genome | Account | Pairs | Type |
|--------|---------|-------|------|
| `models/asi_mc_v2_best.pkl` | 008 | 12 (all) | General bidirectional |
| `models/iron_s5ft_EUR_GBP.pkl` | 001 | EUR_GBP | Pair-specific S5 fine-tuned |
| `models/iron_s5ft_CAD_JPY.pkl` | 001 | CAD_JPY | Pair-specific S5 fine-tuned |
| `models/iron_v3_EUR_GBP.pkl` | 009 | EUR_GBP | Pair-specific V3 fixed-topology |
| `models/iron_v3_CAD_JPY.pkl` | 009 | CAD_JPY | Pair-specific V3 fixed-topology |

---

## Filter Definitions

```python
# 1. BB Squeeze: Bollinger Band width below threshold
def filter_bb_squeeze(close, period=20, squeeze_pct=0.015):
    sma = rolling_mean(close, period)
    std = rolling_std(close, period)
    bb_width = (2 * std) / sma          # normalised bandwidth
    return bb_width < squeeze_pct       # True = in squeeze = allow entry

# 2. ER14 Ranging: Kaufman ER below threshold (already computed as ER_norm)
def filter_er_ranging(er_norm, threshold=0.35):
    return er_norm < threshold          # True = ranging = allow entry

# 3. RangePosition Extreme: price near top/bottom of N-bar range
def filter_range_extreme(close, period=30, threshold=0.15):
    hi = rolling_max(close, period)
    lo = rolling_min(close, period)
    pos = (close - lo) / (hi - lo + 1e-10)  # 0=bottom, 1=top
    return (pos < threshold) | (pos > (1 - threshold))  # extremes only

# 4. Spread/ATR (use synthetic proxy — high-low range as spread proxy)
def filter_spread_atr(high, low, close, atr_period=14, ratio=0.5):
    atr = rolling_atr(high, low, close, atr_period)
    bar_range = high - low              # proxy for spread
    return bar_range < ratio * atr      # True = tight spread = allow entry
```

---

## Experiment Matrix

For each genome × each filter combination (2^4 = 16 combos + baseline = 17 runs per genome):

```
Combo 0000: no filter (baseline)
Combo 0001: spread/ATR only
Combo 0010: range_extreme only
Combo 0011: range_extreme + spread/ATR
Combo 0100: ER14_ranging only
...
Combo 1111: all 4 filters
```

**Applied as**: if filter active AND network signals BUY/SELL → execute. If filter blocks → skip (stay flat or hold existing position).

---

## Metrics Per Run

- Total pips (OOS period)
- Trades executed
- Win rate %
- Pips per trade
- Max drawdown (pips)
- Sharpe proxy: (total_pips / n_trades) / std(trade_pips)

---

## Expected Hypotheses

1. **BB squeeze + ER ranging** should be complementary — both select low-volatility periods, but from different angles. Overlap may be high → fewer trades but higher quality.
2. **RangePosition** should flip signal direction for IronNet (which is bidirectional) — the filter selects entry points but doesn't change direction, so a BUY at price bottom is fine; a BUY at price top should be blocked.
3. **General model (v2) vs pair-specific**: filters may help general model more, since pair-specific genomes were already trained on tighter conditions.
4. **Spread/ATR**: least interesting on synthetic data (no real spread variation); most useful on live S5 data.

---

## Implementation Plan

### Step 1: Export OOS indicator data
Use `export_curator_identical.py` for the 2 EUR_GBP + CAD_JPY pairs (needed for IronNet genomes).
For v2 general, use existing exported parquets.

### Step 2: Build filter_study.py
- Load genome + config
- For each filter combo (17 runs):
  - Simulate trading with filter gate applied
  - Record metrics
- Output: results table + heatmap (filter combo vs metric)

### Step 3: Run on all 5 genomes
- Run locally (no Hetzner needed — inference only, fast)

### Step 4: Analyse
- Identify which filters help/hurt general vs pair-specific
- Identify best single filter and best combination
- If a combination consistently improves: consider adding as live gate

### Step 5: If promising — promote to live
- Add filter as pre-entry check in strategy container
- Monitor 1 week before making permanent

---

## Files

```
research/experiments/filter_gates/
├── PLAN.md                  ← this file
├── run_filter_study.py      ← main experiment script (to build)
├── results/
│   ├── asi_mc_v2_filters.csv
│   ├── iron_s5ft_EUR_GBP_filters.csv
│   ├── iron_s5ft_CAD_JPY_filters.csv
│   ├── iron_v3_EUR_GBP_filters.csv
│   └── iron_v3_CAD_JPY_filters.csv
└── plots/
    ├── heatmap_asi_mc_v2.png
    └── heatmap_ironnet.png
```

---

## Data Requirements

- OOS S5 parquets for EUR_GBP and CAD_JPY (from `neat-data` volume or local export)
- `lib/asi_indicator.py` for indicator computation
- `lib/fast_eval.py` for genome inference
- NEAT configs: `models/neat_config_4in.ini` (IronNet) + `models/neat_config.ini` (v2)

---

## After Filter Study

If filters validated → next experiments:
1. **Add RangePosition as NEAT input** (5th input) — let the network learn to use it rather than hard-gating
2. **BB squeeze state as input** — continuous bandwidth value, not binary gate
3. **Re-train v5** with expanded 6-input set: MC_D, MC_dD, ER_norm, UPnL, RangePosition, BB_width
