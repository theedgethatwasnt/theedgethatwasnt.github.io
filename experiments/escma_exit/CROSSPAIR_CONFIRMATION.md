# Position-Aware Sighted-Shock-Fade ESCMA — Cross-Pair Confirmation

**Validation gate before paper deployment.** Re-run the established USD_JPY
sighted wavelet-z2_k2 fade at full convergence (100 gens vs 60), then replicate
the FULL pipeline out-of-instrument on EUR_JPY, GBP_JPY, AUD_JPY.

Network/features frozen: `N_POS=3` (signed P/L, hold-frac, accumulated pain),
`N_PARAMS=273` (the docstring header is stale; code is N_IN=26→8→1).
Exit-learner trained with CMA-ES, mode=fade, z_thr=2 k_bands=2 wavelet gate.

## Established baseline (60 gens, posaware, t_max_alloc=2880 default)

| seed | OOS AMDDP5 | raw PnL | WR |
|------|-----------|---------|-----|
| 42 | +291.45 | +410.33 | 47.0% |
| 7 | +108.08 | +177.80 | 47.6% |
| 123 | +19.91 | +566.09 | 51.9% |
| **mean** | **+139.8** | — | — |

All 3 seeds positive; beat the best naive baseline (first-negative-tick = −56.59
AMDDP5 at the 2880 cap). Random-entry control: mean AMDDP5 ≈ −3269 (spread floor).

## ⚠️ Methodology note — t_max_alloc changed mid-run

The 100-gen confirmation runs were launched with `--t-max-alloc 1440` (a 2× speed
optimization, capping max-hold at 2h instead of 4h), NOT the 2880 default used by
the established 60-gen result. This is **not a clean apples-to-apples comparison**:
the shorter time cap changes the OOS baselines materially (e.g. first-negative-tick
naive baseline jumps from −56.59 at 2880 to **+437.17** at 1440 for USD_JPY s42,
because the time-cap fallback resolves differently). The learned exit (mean_hold≈22
bars) is barely affected by the cap, but the **bar it must clear (the naive
baseline) moved up sharply.**

## Part A — USD_JPY 100-gen confirmation (t_max_alloc=1440)

| seed | OOS AMDDP5 | raw PnL | WR | best naive baseline (firstneg) | beats naive? |
|------|-----------|---------|-----|--------------------------------|--------------|
| 42 | **+22.64** | +318.42 | 50.5% | **+437.17** | ❌ NO |
| 7 | _pending_ | | | | |
| 123 | _pending_ | | | | |

**Interim read (s42 only):** at 100 gens the +291 (60g) collapsed to +22.64, and
the naive first-negative-tick baseline (+437 at the 1440 cap) now BEATS the learned
exit. The +140 mean did NOT hold or improve at convergence — it regressed toward
zero. Whether this is convergence-regression (overfit at 60g, real edge ≈0) or the
t_max_alloc=1440 change inflating the baseline needs the other two seeds + ideally
a 2880-cap rerun to disentangle. **Flagged.**

## Cross-pair confirmation table (PENDING — run in progress)

| pair | n_events (σ-gate) | wav_z2_k2 events | OOS n | sighted-fade OOS AMDDP5: s42/s7/s123 | mean | all-seeds-positive? | beats naive? |
|------|------|------|------|------|------|------|------|
| USD_JPY (100g, tma=1440) | 9,751 | _see survival_ | ~2316 | +22.6 / _pend_ / _pend_ | _pend_ | _pend_ | s42 ❌ |
| EUR_JPY | _pend_ | _pend_ | _pend_ | _pend_ | _pend_ | _pend_ | _pend_ |
| GBP_JPY | _pend_ | _pend_ | _pend_ | _pend_ | _pend_ | _pend_ | _pend_ |
| AUD_JPY | _pend_ | _pend_ | _pend_ | _pend_ | _pend_ | _pend_ | _pend_ |

_Verdict pending full run completion._
