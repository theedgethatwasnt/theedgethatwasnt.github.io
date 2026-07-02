# Zone Recovery Experiment Report
Generated: 2026-04-28 18:24
Pair: EUR_USD | Data: M5 2021-2026 | Post-RCA causal standards

## Phase 1: Classic cBot Parameter Grid
Configs tested: 320

### Top 10 by Sharpe
|   half_zone_pips |   target_beyond_pips |   profit_factor |   n_cycles |   net_pnl_pips |     sharpe |   win_rate |   avg_legs |   max_legs_hit_pct |
|-----------------:|---------------------:|----------------:|-----------:|---------------:|-----------:|-----------:|-----------:|-------------------:|
|            10.25 |                   20 |            1    |      17886 |       -24278.5 | -0.0220452 |   0.844459 |    3.58426 |           0.263614 |
|            25    |                   20 |            1.19 |      10633 |       -16869   | -0.0226206 |   0.913571 |    2.35155 |           0.135333 |
|            25    |                   20 |            1.3  |      10633 |       -17616.8 | -0.0226558 |   0.913947 |    2.35155 |           0.135333 |
|            25    |                   20 |            1.1  |      10633 |       -16508.2 | -0.0228845 |   0.913101 |    2.35155 |           0.135333 |
|            25    |                   20 |            1    |      10633 |       -16140.1 | -0.0231746 |   0.911972 |    2.35155 |           0.135333 |
|            10.25 |                   20 |            1.1  |      17886 |       -29946.9 | -0.0250192 |   0.846975 |    3.58426 |           0.263614 |
|            25    |                   20 |            1.5  |      10633 |       -23962.2 | -0.0251229 |   0.914888 |    2.35155 |           0.135333 |
|            10.25 |                   20 |            1.19 |      17886 |       -34631.8 | -0.0273012 |   0.847646 |    3.58426 |           0.263614 |
|            10.25 |                   20 |            1.3  |      17886 |       -38056   | -0.0287485 |   0.848429 |    3.58426 |           0.263614 |
|            25    |                   15 |            1.1  |      11094 |       -26059.7 | -0.0292002 |   0.912385 |    2.19894 |           0.111321 |

**Best**: hz=10.25p tgt=20.0p pf=1.00 → pnl=-24278.5p sharpe=-0.022

## Phase 2: ATR-Calibrated Parameter Grid
Configs tested: 80

### Top 10 by Sharpe
|   half_zone_mult |   target_mult |   median_ez_ratio |   n_cycles |   net_pnl_pips |    sharpe |     sqn |   avg_legs |
|-----------------:|--------------:|------------------:|-----------:|---------------:|----------:|--------:|-----------:|
|              0.7 |          1.5  |           6.09036 |      17494 |        57624.7 | 0.0554897 | 7.33935 |    3.80873 |
|              0.7 |          1.25 |           5.0753  |      17494 |        55544.4 | 0.0539071 | 7.13001 |    3.80839 |
|              0.7 |          1    |           4.06024 |      17433 |        50896.2 | 0.0507527 | 6.70109 |    3.79872 |
|              0.9 |          3.25 |          10.2634  |      14951 |        42494   | 0.0478807 | 5.85458 |    3.43355 |
|              0.9 |          3    |           9.47389 |      14951 |        42328.3 | 0.0476539 | 5.82685 |    3.43355 |
|              0.9 |          2.75 |           8.6844  |      14951 |        42313.1 | 0.0476238 | 5.82316 |    3.43355 |
|              0.9 |          2.5  |           7.89491 |      14951 |        42164   | 0.0473565 | 5.79048 |    3.43355 |
|              0.9 |          2.25 |           7.10542 |      14951 |        42067.1 | 0.047189  | 5.77    |    3.43355 |
|              0.9 |          2    |           6.31593 |      14951 |        41874.1 | 0.0468434 | 5.72775 |    3.43355 |
|              0.7 |          3.25 |          13.1958  |      17203 |        46216.3 | 0.0459748 | 6.03007 |    3.84741 |

## Phase 3: 5-Gate Walk-Forward Validation
Configs validated: 10 | PASS: 2/10

### 🔴 classic_hz10.25_tgt20.0_pf1.00 [FAIL]
Gates passed: 0/5

- ❌ gate1_oos_positive
- ❌ gate2_wf_all_positive
- ❌ gate3_permutation_significant
- ❌ gate4_seed_cv_pass
- ❌ gate5_sqn_pass

OOS: pnl=-12963.9p | sharpe=-0.026 | sqn=-2.40 | n=8434 trades
Win rate: 84.1% | MaxDD: -17560.1p

### 🔴 classic_hz25.0_tgt20.0_pf1.19 [FAIL]
Gates passed: 0/5

- ❌ gate1_oos_positive
- ❌ gate2_wf_all_positive
- ❌ gate3_permutation_significant
- ❌ gate4_seed_cv_pass
- ❌ gate5_sqn_pass

OOS: pnl=-15516.6p | sharpe=-0.034 | sqn=-2.37 | n=4855 trades
Win rate: 91.6% | MaxDD: -18092.2p

### 🔴 classic_hz25.0_tgt20.0_pf1.30 [FAIL]
Gates passed: 0/5

- ❌ gate1_oos_positive
- ❌ gate2_wf_all_positive
- ❌ gate3_permutation_significant
- ❌ gate4_seed_cv_pass
- ❌ gate5_sqn_pass

OOS: pnl=-16041.7p | sharpe=-0.033 | sqn=-2.32 | n=4855 trades
Win rate: 91.6% | MaxDD: -18738.9p

### 🔴 classic_hz25.0_tgt20.0_pf1.10 [FAIL]
Gates passed: 0/5

- ❌ gate1_oos_positive
- ❌ gate2_wf_all_positive
- ❌ gate3_permutation_significant
- ❌ gate4_seed_cv_pass
- ❌ gate5_sqn_pass

OOS: pnl=-14861.0p | sharpe=-0.035 | sqn=-2.41 | n=4855 trades
Win rate: 91.5% | MaxDD: -17372.5p

### 🔴 classic_hz25.0_tgt20.0_pf1.00 [FAIL]
Gates passed: 0/5

- ❌ gate1_oos_positive
- ❌ gate2_wf_all_positive
- ❌ gate3_permutation_significant
- ❌ gate4_seed_cv_pass
- ❌ gate5_sqn_pass

OOS: pnl=-14488.2p | sharpe=-0.036 | sqn=-2.48 | n=4855 trades
Win rate: 91.4% | MaxDD: -16898.8p

### 🔴 atr_hz0.7_tgt1.50 [FAIL]
Gates passed: 2/5

- ✅ gate1_oos_positive
- ❌ gate2_wf_all_positive
- ❌ gate3_permutation_significant
- ✅ gate4_seed_cv_pass
- ❌ gate5_sqn_pass

OOS: pnl=+1580.6p | sharpe=0.008 | sqn=0.51 | n=4078 trades
Win rate: 81.6% | MaxDD: -2304.7p

### 🔴 atr_hz0.7_tgt1.25 [FAIL]
Gates passed: 0/5

- ❌ gate1_oos_positive
- ❌ gate2_wf_all_positive
- ❌ gate3_permutation_significant
- ❌ gate4_seed_cv_pass
- ❌ gate5_sqn_pass

OOS: pnl=-790.7p | sharpe=-0.009 | sqn=-0.39 | n=2064 trades
Win rate: 81.4% | MaxDD: -2388.6p

### 🔴 atr_hz0.7_tgt1.00 [FAIL]
Gates passed: 2/5

- ✅ gate1_oos_positive
- ✅ gate2_wf_all_positive
- ❌ gate3_permutation_significant
- ❌ gate4_seed_cv_pass
- ❌ gate5_sqn_pass

OOS: pnl=+555.8p | sharpe=0.023 | sqn=0.49 | n=448 trades
Win rate: 79.2% | MaxDD: -1048.0p

### 🟢 atr_hz0.9_tgt3.25 [PASS]
Gates passed: 5/5

- ✅ gate1_oos_positive
- ✅ gate2_wf_all_positive
- ✅ gate3_permutation_significant
- ✅ gate4_seed_cv_pass
- ✅ gate5_sqn_pass

OOS: pnl=+10975.3p | sharpe=0.031 | sqn=2.52 | n=6392 trades
Win rate: 86.4% | MaxDD: -2140.0p

### 🟢 atr_hz0.9_tgt3.00 [PASS]
Gates passed: 5/5

- ✅ gate1_oos_positive
- ✅ gate2_wf_all_positive
- ✅ gate3_permutation_significant
- ✅ gate4_seed_cv_pass
- ✅ gate5_sqn_pass

OOS: pnl=+10144.4p | sharpe=0.031 | sqn=2.48 | n=6292 trades
Win rate: 85.9% | MaxDD: -2285.7p

## Verdict

🟢 **DEPLOYABLE** candidate found: `atr_hz0.9_tgt3.25`
- OOS pnl: +10975.3 pips
- OOS Sharpe: 0.031
- SQN: 2.52
- Gates: 5/5

**Next step**: Deploy to OANDA accounts 011 (long) + 012 (short) at 0.001 lot, 1-week shadow validation.