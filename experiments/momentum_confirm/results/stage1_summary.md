# Monthly FX Cross-Sectional Momentum — Stage 1 Summary

Governed by `PREREGISTRATION.md` (LOCKED 2026-07-07). Rule frozen verbatim from `research/experiments/fx_factors/factors.py` (momentum variant: 12-1 currency momentum, top-3/bottom-3 of 7 non-USD currencies, 63-day inverse-vol weighting, no risk-off gate — imported, not reimplemented). Data: OANDA D1 deep history (`data/d1_deep/`), hard-sealed to timestamps < 2020-11-01 by `stage1_data.py` (pyarrow filter pushdown + independent post-read assertion). The observation window (2020-11->2024-08) and the fx_factors sealed OOS (2024-08-30->2026-05-21) were never loaded.

## Coverage

**196 monthly rebalances**, 2004-05-31 -> 2020-08-31 (pre-reg expected ~180 for 2005-2020; the master calendar's earliest common bar across all 7 required pairs is 2004-05-31 because CAD_JPY/CHF_JPY only start there in the deep-D1 pull, pushing coverage slightly earlier and to 196 rebalances).

## Gate table (Stage-1 gates 1-4, PREREGISTRATION.md "Stage-1 gates")

| Gate | Name | Result | Detail |
|---|---|---|---|
| 1 | apparatus self-test: null mean approx -costs (<0, |.|<50p) | **PASS** | null_mean=-13.1300p/rebal (n_seeds=200) |
| 2 | deep-segment momentum net>0 AND > null p95 | **FAIL** | deep_net=-8.2450p null_p95=-0.2785p null_mean=-13.1300p n=196 |
| 3 | WF thirds: >=2/3 positive, none < -40p/rebal | **FAIL** | thirds_mean=[+9.151, -1.846, -31.679] n_positive=1/3 |
| 4 | regime split @ 2013: >=1 half positive, other not < -40p | **PASS** | pre_2013(n=104)=+3.7217p post_2013(n=92)=-21.7725p |

**2/4 gates pass.** All 4 are required (pre-reg: "all required before Stage 2 may be requested").

## Net / gross / max drawdown (primary spread_mult=1.0, plus 1.5x sensitivity)

| Variant | n | mean net p/rebal | cum net (p) | max DD (p) | mean carry-free gross p/rebal | cum carry-free gross (p) |
|---|---|---|---|---|---|---|
| momentum | 196 | -8.245 | -1616.0 | -3047.1 | +1.272 | +234.0 |
| momentum_spread1.5x | 196 | -9.621 | -1885.7 | -3247.5 | -0.194 | -35.7 |

Carry-free gross = price return net of spread only (no carry term) — reported per the pre-reg's mitigation for the pre-2020 carry splice's "direction of bias unknown".

## Per-third (WF, Gate 3)

| Third | mean net p/rebal |
|---|---|
| 1 | +9.151 |
| 2 | -1.846 |
| 3 | -31.679 |

1/3 thirds net-positive.

## Per-half (regime split @ 2013, Gate 4)

| Half | n | mean net p/rebal |
|---|---|---|
| pre-2013 | 104 | +3.722 |
| post-2013 | 92 | -21.773 |

## R10 null distribution (200 random-weight portfolios, identical schedule/costs)

n_seeds=200, mean=-13.1300p, p95=-0.2785p, p5=-26.1953p, std=8.2224p.

## Verdict

Deep-segment momentum (primary, spread_mult=1.0) mean net = -8.245 p/rebalance over 196 monthly rebalances (cum -1616.0p, max DD -3047.1p) vs R10 null mean -13.130p / p95 -0.278p.
Gate(s) 2, 3 FAIL — not all 4 required Stage-1 gates clear on this locked configuration.
Regime detail: thirds = +9.2/-1.8/-31.7 p/rebal (1/3 positive); pre-/post-2013 = +3.7/-21.8 p/rebal — the edge does not survive out of the observation window on genuinely new (older) data.
Per the pre-registration's decision rule: Stage 1 fails -> momentum is recorded as a sweep-artifact (the +27.4 p/rebalance IS 'positive' was a single-pass observation, not a generalizing rule) and the fx_factors sealed OOS window stays sealed/unspent.
STAGE-2 RECOMMENDATION: DO NOT request the user gate — Stage 2 stays closed on this rule.
