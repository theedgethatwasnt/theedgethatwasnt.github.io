# Experiment: Variant C — Quantized ASI-MC Inputs

## Hypothesis
Continuous MC(D)/MC(dD) values may cause overfitting to specific magnitudes.
Quantizing to {-1, 0, +1} based on sign forces the network to learn from
direction only, not magnitude. Simpler inputs = more robust generalization.

## Inputs (3)
| Input | Computation | Values |
|-------|-------------|--------|
| MC_D_sign | `sign(mc_d) if abs(mc_d) > threshold else 0` | {-1, 0, +1} |
| MC_DD_sign | `sign(mc_dd) if abs(mc_dd) > threshold else 0` | {-1, 0, +1} |
| UPnL | `tanh(pnl_pips / 20)` (continuous, unchanged) | [-1, +1] |

## Threshold parameter
Dead zone around zero — values with `abs < threshold` are mapped to 0.
Test thresholds: 0.02, 0.05, 0.10

## Implementation
1. Add quantized columns to `export_indicators.py`:
   ```python
   threshold = 0.05
   df["mc_d_c"] = np.where(np.abs(mc_d) > threshold, np.sign(mc_d), 0.0)
   df["mc_dd_c"] = np.where(np.abs(mc_dd) > threshold, np.sign(mc_dd), 0.0)
   ```
2. Add `--variant C` to `train_from_indicators.py` (reads `mc_d_c`/`mc_dd_c` columns)
3. Train on Hetzner: 3 thresholds × 2 seeds = 6 runs (fits on 5 servers + 1 local)

## Comparison
- A (continuous) vs C (quantized) on same 12-pair OOS
- Metrics: pips/day, WR, Sharpe, max DD, trade count
- If C wins: simpler indicator, easier to deploy, less sensitive to indicator precision

## Estimated effort
- Export: 10 min (add columns to existing pipeline)
- Training: 30 min on Hetzner (5 servers)
- Total: ~1 hour including validation
