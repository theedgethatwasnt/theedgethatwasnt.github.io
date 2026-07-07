# FX Factor Suite — IS Summary

Governed by `PREREGISTRATION.md` (LOCKED 2026-07-07). IS window only: 2020-11-11 -> 2024-08-30 (OOS sealed, never read — `is_data.load_pair_is_d1()` hard-filters every load, pushed down to pyarrow at read time + a second independent post-read assertion).

## Gate table (gates 1-4, PREREGISTRATION.md "Gates before OOS")

| Gate | Name | Result | Detail |
|---|---|---|---|
| 1 | harness self-test: null mean approx -costs (<0, |.|<50p) | **PASS** | null_mean=-15.9926p/rebal (n_seeds=200); rigorous check in test_rebalance_engine.py |
| 2 | carry-accrual parity vs carry_model (+-5%) | **SEE PYTEST** | proven in test_rebalance_engine.py::test_carry_accrual_parity (see pytest output, not re-derived here) |
| 3 | gated carry IS net>0 AND > null p95 | **FAIL** | carry_gated_net=-34.1364p null_p95=+15.4705p null_mean=-15.9926p |
| 4 | WF halves both net-positive | **FAIL** | first_half(n=22)=-39.3742p second_half(n=22)=-28.8986p |

**1/3 numerically-checked gates pass** (gate 2 is proven separately by test_rebalance_engine.py::test_carry_accrual_parity).

## Per-factor table (IS, pooled across all monthly rebalances)

| Variant | n | mean net p/rebal | cum net (p) | max DD (p) | vs null mean | vs null p95 (pctile proxy) |
|---|---|---|---|---|---|---|
| carry_gated | 44 | -34.136 | -1502.00 | -1640.30 | -18.144 | -57.7% |
| carry_ungated | 44 | -53.418 | -2350.41 | -2488.71 | -37.426 | -119.0% |
| composite | 44 | -54.226 | -2385.96 | -2534.99 | -38.234 | -121.5% |
| momentum | 44 | +27.448 | +1207.72 | -874.07 | +43.441 | 138.1% |
| value | 44 | -59.958 | -2638.13 | -2987.85 | -43.965 | -139.7% |

R10 null: n_seeds=200, mean=-15.9926p, p95=+15.4705p, p5=-48.4097p, std=17.8869p.

## Verdict

Gate 3=FAIL, Gate 4=FAIL — the two IS gates do not both clear.
Per the pre-registration's decision rule, the program stops here on this locked configuration: OOS stays sealed for a future amended shot, not opened on this run.
Gated carry (primary) IS mean net = -34.136 p/rebalance over 44 monthly rebalances (cum -1502.0p, max DD -1640.3p).
Momentum, value and the equal-weight composite are reported for completeness only and are never promoted without their own separate confirmation (pre-reg).
Small-N caveat (pre-reg, disclosed): ~45 IS monthly rebalances is an inherently wide-CI regime for a factor-investing horizon; the academic prior for FX carry/momentum/value carries part of the burden here, stated not hidden.
