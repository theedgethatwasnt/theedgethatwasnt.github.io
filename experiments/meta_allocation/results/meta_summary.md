# Workstream F — Meta-Allocation Ensemble (Regime-Switched Allocation Across Strategies)

**Question:** can regime-switched ALLOCATION across the project's own strategies be an edge
layer, generalizing the WF-validated equity-MA overlay
(`research/experiments/conservative_010/bb_equity_switch.py` / `bb_equity_switch_validate.py`)
from a single strategy's own trade stream to a portfolio of strategy-columns?

Data: `trades.duckdb` dump on the Hetzner R&D box (`root@<rnd-box>:/root/work/trades_2026-07-06.duckdb`),
8,426 rows, live + paper, 2026-03-25 → 2026-07-06. All computation ran on that box (`build_matrix.py`,
`meta_alloc.py` in this directory, rsynced to `/root/work/code_meta/`).

## 1. Matrix coverage

- 8,426 closed trades in the DB; 8,382 kept (99.5%) after the grouping filter.
- Grouping: live trades (`is_paper=False`) pooled by `strategy` column, any size. Paper trades
  pooled by `strategy`, kept only if group size > 50 trades. Live/paper groups sharing a strategy
  name (e.g. `post_shock_retrace`) are separate columns (`*__paper` suffix).
- 3 paper groups dropped for size: `first_touch_reversion` (n=2), `portfolio_paper` (n=9),
  `tr_paper` (n=33).
- Result: **38 strategy-groups × 104 calendar days (2026-03-25 → 2026-07-06), 5 calendar months**
  (Mar partial, Apr, May, Jun, Jul partial → only **4 causal monthly rebalance events**).
- Per-group active windows vary wildly: from 1 day (`sma_fade`, `asi_mc_v3`) to the full 104 days
  (`sma_stack`). Full breakdown: `results/group_meta.csv`.

## 2. Method

- Daily net pips per group = sum of `pnl_pips` over trades closing that calendar date. Cell is 0
  inside a group's `[first_trade, last_trade]` window (running, no fill that day), NaN outside
  (group doesn't exist yet / already stopped).
- **Baselines:** (a) equal-weight always-on — daily mean pnl across all groups active that day;
  (b) each strategy alone (full table: `results/solo_strategy_summary.csv`).
- **Treatment:** monthly-rebalanced equity-MA switch, generalizing the overlay convention exactly —
  causal, `state_on = (own_cumulative_equity[t-1] > own_rolling_MA(W)[t-1])`, decided once at each
  month's start using **only data strictly before that month** (R1-style: a strategy can never see
  its own future). A group with fewer than `W` days of its own trading history yet defaults to ON
  (benefit of the doubt, same convention as `bb_equity_switch.py`'s `state_live=True` default).
  Swept W ∈ {10, 20, 40} calendar days.
- **R10 null:** for the W=20 switch, at each month's rebalance the switch selected `k_m` groups
  out of that month's eligible pool. The null draws `k_m` groups **uniformly at random** from the
  same eligible pool on the same rebalance dates, 100 seeds (seed 20260706+i).
- **Walk-forward:** month-by-month is inherently the honesty gate here — every decision is causal
  by construction (no separate IS/OOS split; only 4 rebalance points exist, too few to hold out a
  further split meaningfully).

## 3. Results

| Strategy | Net (p) | MaxDD (p) | Sharpe-ish |
|---|---:|---:|---:|
| Equal-weight always-on | −3,792 | −5,174 | −2.26 |
| Switch W=10 | **−1,869** | **−4,560** | −1.13 |
| Switch W=20 | −1,949 | −4,560 | −1.18 |
| Switch W=40 | −3,630 | −5,174 | −2.16 |
| Random null (mean of 100, k matched to W=20) | −3,799 (σ=502) | −5,237 (σ=419) | −2.24 |

**Switch W=20 vs the 100-seed random null:** net beats **100%** of random draws (random 5–95pct
range: [−4,549, −2,822], switch at −1,949 is clear of the whole band); maxDD beats **86%** of
random draws. This is a genuine, non-random allocation effect — not just "fewer slots happen to
be better."

Solo-strategy spread is enormous: best single strategy alone (`zr_paper__paper`, 33 days) nets
+7,615p; worst four (`range_neat`, `asi_mc_008`, `wf_pnlmae`, `pnl_mae_a` — the pre-RCA
lookahead-trained strategies, all launched together in April and stopped within ~9-10 days) each
net **−11,800 to −12,300p**. These four alone account for essentially all of the portfolio's
total drawdown.

**Why the switch can't fix April:** all 19 of April's strategy-groups (including the four
disasters) launched fresh that month with zero prior trading history, so the switch's own default
rule ("< W days of own history ⇒ ON") puts every one of them in the on-set for the whole month —
identical to equal-weight. Monthly-cadence gating is structurally blind to a blowup that completes
*within* its first rebalance period. The switch's entire measured benefit (−1,949 vs −3,792 ≈
+1,843p, and the maxDD cut) comes from **May onward**, where it correctly drops still-active
underperformers (`zr_random`, plain `retrace`) that equal-weight keeps in the average because they
haven't formally stopped trading yet. W=40 collapses back to equal-weight because most groups never
accumulate 40 days of own history before either stopping or the data ending — the window is too
long for the survival lengths actually observed here.

## 4. Verdict

There is a **real, distinguishable allocation-alpha layer** — the monthly equity-MA switch beats
100% of matched-slot-count random draws on net P&L and beats 86% on max drawdown, using nothing but
each strategy's own trailing equity vs. its own trailing MA (no cross-strategy information, exactly
the validated single-strategy overlay applied per-column). It roughly **halves the portfolio's net
loss and cuts max drawdown by ~12%** versus equal-weight always-on over this window. But it is a
**risk-shedding layer, not an alpha-manufacturing layer**: it cannot rescue a strategy that blows up
faster than the rebalance cadence (the four −12k lookahead-poisoned strategies dominated April
before the switch ever got a vote), and the whole portfolio — switched or not — is still net
negative because those four strategies overwhelm everything else combined. The honest reading:
*generalizes the single-strategy result* (turn off things whose trailing equity has rolled over) but
does **not** turn a loss-dominated strategy roster into a profitable ensemble; it just loses less
while losing.

## 5. Caveats

- **Survivorship in the trade ledger** (must-name per plan): the ledger records what actually ran,
  not what could have run. Strategies that were killed for cause (RCA, kill-switches, manual stops)
  simply stop emitting trades and drop out of the eligible pool from that point on — for *every*
  arm (equal-weight, switch, random null) equally, since pool membership is defined the same way
  in all three. This means the comparison is fair (apples-to-apples pool), but it also means none
  of the arms had a chance to be tested on "what if a bad strategy had kept running" — real-time
  operator intervention (not the switch) removed most of the worst-case tail from the *pool*
  itself. The 86%/100% beat-rates above are the switch's contribution *on top of* that
  already-curated pool, not a claim that the switch alone would have caught the RCA disasters had
  the operator not also killed them.
- **Extremely thin sample for a monthly-rebalanced strategy:** only 4 causal rebalance events
  (Apr/May/Jun/Jul boundaries) over ~3.5 months of data. The W-sweep and random-null give some
  confidence the direction of the effect is real, but this is nowhere near enough rebalance events
  to trust the magnitude, and no true out-of-sample seal exists for this workstream (exploratory,
  IS-only mindset per the plan).
- Many groups have very short lifespans (single-day to few-week), so "own equity vs own MA" is
  undefined (defaults ON) for most of a group's early life — the switch is only doing real work for
  the handful of groups that lived long enough to accumulate ≥10-20 days of history while still
  being run.
- Pip P&L is treated as directly poolable/comparable across strategies (no capital-weighting,
  margin, or correlation adjustment) — consistent with how this repo already reports "combined
  p/d" elsewhere, but it means a strategy on 500 units and one on 5 units contribute equally to the
  equal-weight/switch average.

## Files

- `build_matrix.py` — daily P&L matrix builder (grouping rule, active windows)
- `meta_alloc.py` — baselines, monthly equity-MA switch (W sweep), R10 random null
- `results/daily_pnl_matrix.csv`, `results/group_meta.csv` — the matrix + per-group metadata
- `results/solo_strategy_summary.csv` — every strategy alone (net/maxDD/Sharpe-ish)
- `results/switch_log_W{10,20,40}.csv` — monthly on/off decisions per W
- `results/baseline_vs_treatment.csv`, `results/random_null_dist.npz` — headline numbers + full null distribution
