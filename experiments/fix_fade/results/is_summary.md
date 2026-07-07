# London-Fix Fade — IS Battery Summary

Governed by `PREREGISTRATION.md` (LOCKED 2026-07-07). IS window only: 2020-11-11 -> 2024-09-22T21:35:00 UTC (first 70% of the stated 2020-11-11 -> 2026-05-21 window; OOS sealed, never read — every loader routes through `data_loader.load_pair_is()`). 12 pairs, 3 arms (fade / coin seed=20260708 / continuation) on IDENTICAL timestamps (R10).

## Gate table

| Gate | Name | Result | Detail |
|---|---|---|---|
| 1 | RW self-test (no phantom edge) | **PASS** | fade_gross=+0.13p (se=0.254) coin_gross=-0.34p (se=0.254) net_coin=-1.84p n_fade=360 n_coin=360 |
| 2 | IS net>0 AND > coin | **FAIL** | fade_net=-1.627p coin_net=-1.895p |
| 3 | WF 3 thirds >=2/3 net-positive | **FAIL** | n_pos=0/3 thirds=[-0.914, -2.676, -1.155] |
| 4 | breadth >=6/12 pairs gross-positive | **PASS** | n_gross_pos=8/12 |

**2/4 gates pass.** Portfolio (pooled, fade arm): gross=+0.20p net=-1.63p vs coin arm: gross=-0.06p net=-1.89p vs continuation arm: gross=-0.20p net=-2.03p.

Walk-forward thirds (fade arm, pooled, net p/trade): third 1 (2820n)=-0.91p, third 2 (3171n)=-2.68p, third 3 (2789n)=-1.16p

Supplementary (not a locked IS gate — informative only, same method H1 will apply OOS): day-block bootstrap (2000 resamples by UTC day) on fade arm net: P(net<=0)=1.0000, boot mean=-1.63p, 95% CI=[-2.39p, -0.89p].

Breadth: 8/12 pairs gross-positive (fade arm). Excursion (no SL, bounded by the 60-min cap): worst single-trade adverse excursion (MAE) = 380.60p (USD_JPY 2022-10-21, D=-125.1p — the 2022-10-21 BOJ USD/JPY intervention day, correctly captured, not a data artifact); mean MAE = 10.77p, mean MFE = 11.06p (fade arm).

Event yield: 8780 signal days (|D|>=5p) out of 16944 candidate fix-days scanned across 12 pairs (missing-grid/weekend/holiday days and below-threshold days excluded — see `results/event_stats.json` for the per-pair breakdown).

## Per-pair — fade arm

| Pair | n | WR | gross | net |
|---|---|---|---|---|
| AUD_JPY | 737 | 47% | +0.41p | -1.30p |
| AUD_USD | 675 | 46% | -0.17p | -1.48p |
| CAD_JPY | 753 | 45% | -0.07p | -1.97p |
| CHF_JPY | 792 | 44% | +0.48p | -1.85p |
| EUR_GBP | 562 | 46% | +0.95p | -0.48p |
| EUR_JPY | 791 | 45% | +0.33p | -1.49p |
| EUR_USD | 719 | 46% | +0.13p | -1.29p |
| GBP_JPY | 846 | 48% | +0.93p | -1.79p |
| GBP_USD | 800 | 47% | +0.08p | -1.66p |
| NZD_JPY | 706 | 42% | +0.24p | -1.91p |
| NZD_USD | 655 | 41% | -0.67p | -2.23p |
| USD_JPY | 744 | 46% | -0.28p | -1.83p |
| **PORTFOLIO** | 8780 | 45% | +0.20p | -1.63p |

## Portfolio — all 3 arms (identical timestamps, R10)

| Arm | n | WR | gross | net |
|---|---|---|---|---|
| fade | 8780 | 45% | +0.20p | -1.63p |
| coin | 8780 | 44% | -0.06p | -1.89p |
| continuation | 8780 | 42% | -0.20p | -2.03p |

## Month-end (last trading day) vs rest — fade arm (pre-declared split, not searched)

| Group | n | WR | gross | net |
|---|---|---|---|---|
| last_trading_day | 452 | 46% | +1.72p | -0.13p |
| rest | 8328 | 45% | +0.12p | -1.71p |

Observation (descriptive only, small n=452 for last-trading-day vs n=8328 for rest — not gated, not promoted to confirmatory): gross reversion is markedly stronger on the last trading day of the month (+1.72p vs +0.12p), consistent with the WM/R month-end index-rebalancing flow hypothesis in the pre-registration's framing — but net is still negative on both, so this does not change the verdict.

## Verdict

Gates 1-2-3-4 = PASS/FAIL/FAIL/PASS. Fade arm IS portfolio (pooled, real per-trade spread): gross +0.20p / net -1.63p, versus the coin-flip control at gross -0.06p / net -1.89p.
This matches the pre-registration's stated prior almost exactly: gross reversion at the London fix is real (pre-fix drift does partially mean-revert), but the 60-minute round-trip cannot clear the real OANDA retail spread — net ≈ −spread, the anticipated and acceptable clean-negative result.
Not all IS gates pass (PASS/FAIL/FAIL/PASS). Per the pre-registration's decision rule, IS gates failing means this run stops here — OOS stays sealed, never opened on this run, and no parameters were tuned or swept (locked single-threshold, single-horizon rule throughout, per the pre-registration's 'no sweeps' instruction).
This closes the corpus's own flagged-untested lead (the indicator screen's 'FX-fix session-fade') formally: whether the verdict here is a clean negative or a rare positive, the lead no longer sits open in the research queue.
No parameters were fit to this data — every threshold/horizon/seed is the pre-registered frozen default; the month-end split above is reported exactly as pre-declared, not searched.
