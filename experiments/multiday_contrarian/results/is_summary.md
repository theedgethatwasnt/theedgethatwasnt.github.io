# Task A5 — Multi-Day Contrarian Program: IS Battery Summary

Governed by `PREREGISTRATION.md` (LOCKED 2026-07-06) + Amendment 1. IS window only: 2020-11-11 -> 2024-09-25 (OOS sealed, never read — every loader in this battery hard-filters via `is_data.load_pair_is()`).

## Gate table (gates 3-6; gate 1 = harness self-test PASS in test_harness.py, gate 2 = SATISFIED-IN-PURPOSE/FAILED-IN-LETTER per Amendment 1)

| Gate | Name | Result | Detail |
|---|---|---|---|
| 3 | IS net>0 @1.0x AND > coin | **FAIL** | signal_net=-7.307p coin_net=+3.525p |
| 4 | WF 3 thirds >=2/3 net-positive, none<-2p | **FAIL** | n_pos=1/3 thirds=[1.502, -10.788, -9.87] |
| 5 | day-block bootstrap P(net<=0)<0.05 | **FAIL** | P(net<=0)=0.9650 boot_mean=-7.345p n_boot=2000 |
| 6 | breadth >=6/12 pairs gross-positive | **FAIL** | n_gross_pos=4/12 |

**0/4 gates pass.** Portfolio (pooled, signal arm, base cost): -7.31p net vs coin arm +3.52p net vs continuation arm -0.44p net.

Walk-forward thirds (signal arm, pooled, net_base p/trade): third 1 (131n)=+1.50p, third 2 (210n)=-10.79p, third 3 (165n)=-9.87p

Day-block bootstrap (2000 resamples by UTC day): P(net<=0)=0.9650, boot mean=-7.35p, 95% CI=[-15.07p, +0.61p].

Breadth: 4/12 pairs gross-positive (signal arm).

## Per-pair — arm=signal

| Pair | n | WR | gross | net@1.0x | net@spread1.5x | net@carry2.0x | timeout% |
|---|---|---|---|---|---|---|---|
| AUD_JPY | 47 | 45% | -13.40p | -17.07p | -18.51p | -17.60p | 34% |
| AUD_USD | 34 | 44% | -1.77p | -3.72p | -4.54p | -4.01p | 24% |
| CAD_JPY | 47 | 47% | -5.40p | -9.25p | -10.96p | -9.88p | 34% |
| CHF_JPY | 48 | 50% | -7.34p | -12.48p | -14.56p | -13.08p | 23% |
| EUR_GBP | 15 | 40% | +0.56p | -2.71p | -4.21p | -2.99p | 20% |
| EUR_JPY | 53 | 38% | -17.67p | -21.78p | -23.63p | -22.64p | 28% |
| EUR_USD | 29 | 52% | +1.38p | -1.20p | -2.21p | -1.47p | 28% |
| GBP_JPY | 66 | 47% | -5.70p | -11.36p | -13.64p | -12.28p | 32% |
| GBP_USD | 44 | 45% | -4.42p | -7.55p | -8.82p | -8.08p | 23% |
| NZD_JPY | 44 | 43% | -3.68p | -9.28p | -11.44p | -9.84p | 25% |
| NZD_USD | 30 | 70% | +25.69p | +23.04p | +21.86p | +22.78p | 30% |
| USD_JPY | 49 | 55% | +9.12p | +6.00p | +4.86p | +5.40p | 31% |
| **PORTFOLIO** | 506 | 48% | -3.36p | -7.31p | -8.93p | -7.89p | 28% |

## Per-pair — arm=coin

| Pair | n | WR | gross | net@1.0x | net@spread1.5x | net@carry2.0x | timeout% |
|---|---|---|---|---|---|---|---|
| AUD_JPY | 47 | 60% | +16.98p | +14.00p | +12.55p | +13.43p | 34% |
| AUD_USD | 34 | 41% | -4.34p | -6.24p | -7.06p | -6.53p | 24% |
| CAD_JPY | 47 | 53% | +8.04p | +4.39p | +2.69p | +3.74p | 34% |
| CHF_JPY | 48 | 56% | +17.27p | +12.34p | +10.25p | +11.66p | 23% |
| EUR_GBP | 15 | 53% | +1.10p | -2.22p | -3.72p | -2.50p | 20% |
| EUR_JPY | 53 | 47% | -7.68p | -12.05p | -13.90p | -12.89p | 28% |
| EUR_USD | 29 | 45% | +7.79p | +5.31p | +4.30p | +5.05p | 28% |
| GBP_JPY | 66 | 53% | +9.44p | +4.36p | +2.08p | +3.44p | 32% |
| GBP_USD | 44 | 61% | +23.68p | +20.63p | +19.36p | +20.12p | 23% |
| NZD_JPY | 44 | 36% | -19.35p | -24.23p | -26.38p | -24.81p | 25% |
| NZD_USD | 30 | 60% | +10.44p | +7.84p | +6.66p | +7.55p | 30% |
| USD_JPY | 49 | 55% | +17.07p | +14.12p | +12.97p | +13.52p | 31% |
| **PORTFOLIO** | 506 | 52% | +7.23p | +3.52p | +1.91p | +2.93p | 28% |

## Per-pair — arm=continuation

| Pair | n | WR | gross | net@1.0x | net@spread1.5x | net@carry2.0x | timeout% |
|---|---|---|---|---|---|---|---|
| AUD_JPY | 47 | 55% | +13.40p | +10.19p | +8.75p | +9.64p | 34% |
| AUD_USD | 34 | 56% | +1.77p | -0.15p | -0.97p | -0.45p | 24% |
| CAD_JPY | 47 | 49% | +5.40p | +1.22p | -0.49p | +0.63p | 34% |
| CHF_JPY | 48 | 48% | +7.34p | +2.73p | +0.65p | +1.94p | 23% |
| EUR_GBP | 15 | 60% | -0.56p | -3.87p | -5.38p | -4.16p | 20% |
| EUR_JPY | 53 | 55% | +17.40p | +12.45p | +10.59p | +11.67p | 28% |
| EUR_USD | 29 | 41% | -1.60p | -3.57p | -4.58p | -3.81p | 28% |
| GBP_JPY | 66 | 50% | +5.70p | +0.43p | -1.85p | -0.48p | 32% |
| GBP_USD | 44 | 52% | +4.42p | +1.44p | +0.17p | +0.95p | 23% |
| NZD_JPY | 44 | 52% | +3.10p | -1.12p | -3.28p | -1.75p | 25% |
| NZD_USD | 30 | 23% | -25.69p | -28.33p | -29.52p | -28.62p | 30% |
| USD_JPY | 49 | 43% | -9.12p | -11.80p | -12.94p | -12.40p | 31% |
| **PORTFOLIO** | 506 | 49% | +3.27p | -0.44p | -2.06p | -1.04p | 28% |

## Secondary analyses (exploratory, IS-only, never confirmatory)

### (a) CSI StrengthSpread H4/64-bar port
- n_rebalances=93, n_legs=558, mean gross/leg=-15.33p, mean net/leg=-23.84p, frac legs net+=44%. (H=64, N=3 taken verbatim from csi_factor_study's recorded prior; 12-pair/8-currency port, conservative no-turnover-netting cost model — see script docstring.)

### (b) D1 RSI(2) mean-reversion (classic, Wilder smoothing)
- n=1631, WR=52%, mean gross=-2.99p, mean net@1.0x=-16.31p, net@spread1.5x=-22.30p, net@carry2.0x=-17.28p, 5/12 pairs gross-positive.

### (c) Equal-risk portfolio of IS-positive signals
- candidate means (base cost, **per calendar day**, trades summed within a day then averaged across days — NOT the same aggregation as the per-trade means above/below; a day with several trades counts once, so this differs from the flat per-trade portfolio mean, e.g. first_touch_signal here vs -7.31p per-trade in the gate table): first_touch_signal=-9.97p, strengthspread=-143.01p, rsi2_d1=-43.39p
- No signal was IS-positive at base cost; equal-risk portfolio not constructed.

## Verdict

Gates 3-4-5-6 = FAIL/FAIL/FAIL/FAIL. The portfolio (pooled, signal arm, base cost) IS net expectancy is -7.31p, versus the coin-flip control at +3.52p — the signal does NOT clear its own coin-flip null, consistent with Amendment 1's early uncontrolled read (~-6p IS net) on this corrected NY-17:00 bar grid.
Per the pre-registration's decision rule: IS gates failing means the program stops here on this frozen-parameter configuration — OOS stays sealed for a future amended shot, not opened on this run.
Of the three secondaries, none is presented as confirmatory; each is reported exactly as computed above with its documented simplifications, and the equal-risk portfolio (c) only combines whatever subset was IS-positive at base cost, which may be zero, one, two, or three.
This reproduces the program's recurring pattern across the wider FX-Core research history: intraday/multi-day directional signals rarely clear the OANDA retail spread once carried through an honest walk-forward + bootstrap gate, net of realistic cost.
No parameters were tuned in this run; all values are the pre-registered frozen defaults or recorded classic/prior configurations, exactly as required.
