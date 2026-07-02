# Experiment Plan: Autoencoder → NEAT Latent Input
**Date**: 2026-04-05  
**Status**: Design (not started)

---

## Motivation

We have 13 candidate features:
- 4 original NEAT inputs: MC_D, MC_dD, ER_norm, UPnL
- 2 new from filter study: RangePosition, BB_width
- 7 top IC features from feature_statistics study:
  RSIExtreme, ZScore, BBPosition, Stoch_K, WilliamsR, BreakoutChannel, HeikenAshi, SuperTrend

Problem: feeding all 13 directly to NEAT explodes the search space and likely introduces redundancy (many are correlated: BB/ZScore, Stoch_K/WilliamsR). An autoencoder can:
1. Compress correlated features into orthogonal latent dims
2. Validate which features carry signal (reconstruction matters)
3. Produce a compact, dense representation for NEAT

---

## Experiment A — Train Autoencoder

### Data
- 12 pairs × M5 OHLC parquets (curator_identical + compute new indicators)
- IS: first 70% of data per pair
- OOS: last 30% (same split as filter study)

### Feature Set (13 inputs to AE)

```python
features = {
    # Existing NEAT inputs
    'mc_d':          curator parquet col
    'mc_dd':         curator parquet col
    'er_norm':       compute_er(close, 14)
    'upnl':          0.0 at training time (positional — exclude from AE, add back at NEAT)

    # New from filter study
    'range_pos':     (close - rolling_min(30)) / (rolling_max(30) - rolling_min(30))
    'bb_width':      2 * rolling_std(20) / rolling_mean(20)

    # Top IC features (all contrarian, M5)
    'rsi_extreme':   RSI(14), rescaled to [-1,1] from 0-100
    'zscore':        (close - rolling_mean(20)) / rolling_std(20)
    'bb_position':   (close - lower_band) / (upper_band - lower_band)
    'stoch_k':       Stochastic %K(14,3), rescaled [-1,1]
    'williams_r':    Williams %R(14), rescaled [-1,1]
    'breakout_ch':   (close - rolling_max(20)) / ATR  — channel breakout signal
    'ha_direction':  Heiken-Ashi candle direction (sign of HA_close - HA_open)
}
# UPnL excluded: it's a runtime positional state, not a market signal
# → 12 features into AE
```

### AE Architecture

```
Encoder:  12 → 8 → latent_dim
Decoder:  latent_dim → 8 → 12

Activation: tanh (bounded, consistent with NEAT inputs)
Loss: MSE reconstruction + L1 sparsity penalty on latent (encourage few active dims)
Optimizer: Adam lr=1e-3, 100 epochs, batch=256
```

### Latent Dim Search
Test latent_dim ∈ {2, 3, 4, 6} — pick smallest with reconstruction R² > 0.85 across all features.

### Validation Criteria
- Reconstruction R² > 0.85 per feature on OOS data
- Latent dimensions show low inter-correlation (< 0.3 pairwise) — confirms orthogonality
- PCA of latent space should show no dominant direction (even distribution of variance)

### If Compression Fails
If R² < 0.85 with latent_dim=6: features are too noisy / not compressible.
→ Fall back to manual selection: keep top 3 IC features + 3 existing = 6 NEAT inputs directly.

---

## Experiment B — NEAT with AE Latent Inputs

### Only run if Experiment A succeeds (R² > 0.85)

### NEAT Inputs
```
latent_dim × AE outputs  (e.g. 3 or 4 continuous values)
+ UPnL                   (runtime positional state — always added back)
= latent_dim + 1 total inputs
```

### Active Activations (from experiments v1-v4)

Activations that appeared at least once across all 4 experiments:
```python
ACTIVE_ACTS = [
    'tanh',       # S-curve — every experiment
    'sin',        # appeared in v3, v4
    'cos',        # appeared in v3, v4
    'gauss',      # appeared in v1, v3, v4
    'sech',       # appeared in every experiment — most consistent
    'dog',        # appeared in every experiment — bandpass
    'gabor',      # dominated v4 L2
    'sinc',       # appeared in v3, v4 — best pure seed
    'morlet_re',  # appeared in v3
    'morlet_im',  # appeared in v1, v2
    'sigmoid',    # appeared in v3, v4
    'haar',       # appeared in v2, v4
    'mex_hat',    # appeared in v1, v2
]
# NOT included: chirp, relu, elu, swish — never selected when competing with wavelets
```

13 active activations.

### Search Space Analysis

**Fixed topology (matching IronNet pattern):**
```
latent+1 inputs → 5 L1 nodes → 7 L2 nodes → 3 outputs
Activation combos: 13^(5+7) = 13^12 ≈ 23 trillion
```

**Free topology (full NEAT):**
```
Adds: number of hidden nodes (0→∞), connection pattern
Search space: effectively unbounded
```

**Seeded population approach (same as v3/v4):**
```
13 activation seeds × 2 islands × 40 pop = 1,040 genomes in gen 1
Each seed tests pure-activation strategy
Crossover builds combinations
200 gens → convergence at ~gen 90 (as observed)
Effective search: ~16,000 evaluated genomes
Still tiny vs 23T but directional via EA
```

### Topology Question (see separate note)

Should we keep fixed topology or allow NEAT topology mutations?
- **Fixed**: reproducible, faster convergence, known failure modes
- **Free**: potentially finds better depth/width, but much larger search space, harder to analyse
- **Recommendation**: test both — fixed 200 gens as baseline, then one free run to compare

### Training Config

Same as v4 but:
- `num_inputs = latent_dim + 1`
- `ACTIVATIONS = ACTIVE_ACTS` (13, not 17)
- Seeded: one pure-activation genome per activation type
- 200 gens, 2 islands, migrate every 20

### Expected Benefit

If AE compresses 12 features → 3-4 orthogonal latent dims:
- NEAT sees a **denser, less redundant** input space
- Each latent dim encodes a genuine independent axis of market state
- NEAT can learn combination rules on meaningful abstractions rather than correlated raw features
- Comparison: NEAT on 12 raw features vs NEAT on 4 latent dims — which finds better edge?

---

## Implementation Steps

1. **Build AE** (`research/experiments/ae_neat/train_ae.py`)
   - Compute 12 features from M5 parquets
   - Train AE, validate R², plot latent space
   - Save encoder weights: `models/ae_encoder_12to{k}.pkl`

2. **Build NEAT experiment** (`research/experiments/ae_neat/run_neat_ae.py`)
   - Load encoder, transform live indicators → latent
   - Same NEAT/simulation infrastructure as run_v4.py
   - Compare vs v4 baseline

3. **Hetzner run** (if local validation promising)
   - 4 servers × 4 islands × 200 gens
   - Multiple latent_dim options in parallel

---

## Files

```
research/experiments/ae_neat/
├── PLAN.md               ← this file
├── train_ae.py           ← autoencoder training
├── run_neat_ae.py        ← NEAT with latent inputs
└── results/
    ├── ae_reconstruction.png
    ├── ae_latent_pca.png
    └── neat_ae_training.png
```

---

## Related: Fixed vs Free Topology Note

See discussion in `research/experiments/ae_neat/TOPOLOGY_NOTE.md`
