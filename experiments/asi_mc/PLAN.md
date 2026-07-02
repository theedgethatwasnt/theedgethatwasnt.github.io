# Experiment: SMA5(ASI) as Price Series for MC(D)/MC(dD)

## Hypothesis

The Accumulative Swing Index (ASI, Wilder 1978) is a cumulative momentum indicator
that trends like price but with less noise. SMA5 smoothing further reduces chatter.
Feeding SMA5(ASI) as a "synthetic price series" into the existing MC(D)/MC(dD) multi-
timeframe consensus should produce a cleaner momentum signal than raw price or HA color.

Key insight: the 1-3 pip noise dead zone that killed HA-based MC in the sine wave test
may be solved because ASI integrates swing information across bars, amplifying real moves
and dampening random noise.

## Inputs (3)

| Input | Description | Range |
|-------|-------------|-------|
| MC(D) | Multi-TF EMA3-EMA5 momentum consensus on SMA5(ASI) | [-1, +1] |
| MC(dD) | Acceleration of MC(D) — second derivative | [-1, +1] |
| UPnL | Unrealized P/L of current position, `tanh(pnl/20)` | [-1, +1] |

## Outputs (3)

BUY / SELL / FLATTEN — highest wins. One position at a time. 2 pips spread.

## ASI Computation (Wilder)

```
SI[i] = 50 * N / R * K / L

Where:
  N = (C - C₋₁) + 0.5*(C - O) + 0.25*(C₋₁ - O₋₁)
  R = max(|H - C₋₁| - 0.5*|L - C₋₁| + 0.25*|C₋₁ - O₋₁|,
          |L - C₋₁| - 0.5*|H - C₋₁| + 0.25*|C₋₁ - O₋₁|,
          (H - L) + 0.25*|C₋₁ - O₋₁|)
  K = max(|H - C₋₁|, |L - C₋₁|)
  L = 3 * ATR(14)  (dynamic limit move)

ASI[i] = ASI[i-1] + SI[i]
```

Then: `synthetic_price = SMA(5, ASI)`

## MC Computation on Synthetic Price

Same as existing MTFMC but operating on SMA5(ASI) instead of raw mid price:
- 9 timeframes: S5, 10s, 30s, M1, 2m, M5, 10m, 30m, H1
- EMA(3) - EMA(5) difference at each TF
- Weighted consensus of last 5 changes
- MC(dD) = second derivative of MC(D)

## Staged Approach

### Stage 0: Sine Wave Validation (local, ~5 min)
- Generate sine wave, compute ASI → SMA5(ASI) → MC(D)/MC(dD)
- Train NEAT with 3 inputs, 3 outputs, tanh/sin/cos activations
- **Pass**: positive OOS P/L, bidirectional trading
- Also test in the 1-3 pip noise zone where HA-MC failed

### Stage 1: Quick NEAT on 1 pair (local, ~15 min)
- EUR_JPY M5, 50 gens, 150 pop
- Compute ASI on M5 OHLC, SMA5 smooth, feed to MC
- **Pass**: positive OOS P/L with >20 trades

### Stage 2: Full Training (Hetzner 5 servers, ~3 hrs, ~$2)
- 4 training pairs, 200 gens, 5 seeds/variants
- **Pass**: positive OOS on majority of 12 pairs

### Stage 3: Validation (local, ~40 min)
- OOS + WF + MC on all 12 pairs

## Files

```
research/experiments/asi_mc/
  PLAN.md                  # This file
  asi_indicator.py         # ASI + SMA5 + MC computation (JIT)
  stage0_sine_test.py      # Sine wave validation
  stage1_quick_scan.py     # Quick NEAT on EUR_JPY
  neat_config_3out.ini     # 3 inputs, 3 outputs, tanh/sin/cos
  results/
```
