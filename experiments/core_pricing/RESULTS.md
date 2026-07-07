# Core-Pricing Arithmetic — Does OANDA US "core spreads + commission" change any verdict?

**Date:** 2026-07-07 · **Type:** cost measurement / arithmetic on recorded results (no backtests re-run; not an edge claim)
**Companion:** measured spread-only medians come from `research/experiments/spread_clock/` (12-pair M5-BA, 2020-11→2026-05, spread = `(ask_c−bid_c)/pip`).

## 1. Verified terms (all sources accessed 2026-07-07)

| Item | Current published (OANDA US) | Source |
|---|---|---|
| Commission | **$0.70 per 10,000 units per leg** = $7.00/100k/side = **$14.00/100k round-turn**, prorated to any size | [OANDA US help: core spreads + commission](https://help.oanda.com/us/en/faqs/spreads-commission.htm) — "A commission of 0.70 USD per 10,000 units is applied to each 'leg' of a trade" |
| Legacy / widely-cited rate | $5.00/100k **per side** = $10 round-turn ("commission-equivalent of 1 pip") | [OANDA 2017 US price sheet PDF](https://www.oanda.com/assets/documents/566/OANDA-CC-Pricing.pdf); still cited by [ForexBrokers.com](https://www.forexbrokers.com/reviews/oanda) (Aug 2025 data) and [BestBrokers](https://www.bestbrokers.com/reviews/oanda/spreads-fees-and-commissions/) (Feb 2026) |
| Core raw spread | "as low as 0.0" on selected majors (USD/CAD, EUR/USD, USD/JPY, AUD/USD); global page "from 0.1"; **measured avg EUR/USD 0.4p** (ForexBrokers.com, Aug 2025, US entity) | same help page; [OANDA BVI pricing](https://www.oanda.com/bvi-en/cfds/our-pricing/); ForexBrokers.com |
| **Eligibility** | **Initial deposit or maintained balance ≥ $10,000 USD** | help page + [US account comparison](https://www.oanda.com/us-en/trading/account-comparison/) |
| Volume rebates | Elite Trader: $5→$17/M rebate, needs ~10M+/month volume | [US pricing page](https://www.oanda.com/us-en/trading/our-pricing/) |
| Spread-only default (comparison) | EUR/USD 1.4p snapshot on US pricing page; measured avg 1.69p (Aug 2025) | US pricing page; ForexBrokers.com |

Entity caveat: other OANDA entities differ (SG $3/100k/side, AU A$3.50/side). All arithmetic below uses the **US** current-published rate, with the legacy $5/side shown as sensitivity.

## 2. Commission in pips + all-in round-trip vs our measured spread-only medians

Commission is fixed in USD per 100k units, so its pip-equivalent depends on the pip value of 100k units (FX rates assumed: USD/JPY≈147 → JPY-quote pip value ≈ $6.80/100k; GBP/USD≈1.34 → EUR_GBP pip ≈ $13.40; USD-quote pairs $10.00). Core raw spread per pair is **estimated** as `max(0.1, measured spread-only median − 1.1p)` — calibrated on EUR/USD (1.5 median vs 0.4 measured core), flagged as an assumption.

| Pair class | Measured spread-only median (ours) | Core raw (est.) | Commission RT @$7/side | **All-in RT (current)** | All-in @legacy $5/side | Δ vs spread-only |
|---|---|---|---|---|---|---|
| EUR_USD | 1.5 | 0.4 (measured) | 1.40 | **1.80** | 1.40 | **+0.30 worse** (−0.10 at legacy) |
| AUD_USD | 1.3 | 0.2 | 1.40 | **1.60** | 1.20 | +0.30 worse |
| GBP_USD | 1.8 | 0.7 | 1.40 | **2.10** | 1.70 | +0.30 worse |
| EUR_GBP | 1.4 | 0.3 | 1.04 | **1.34** | 1.05 | ~flat |
| USD_JPY | 1.6 | 0.5 | 2.06 | **2.56** | 1.97 | **+0.96 worse** |
| EUR_JPY | 2.0 | 0.9 | 2.06 | **2.96** | 2.37 | +0.96 worse |
| GBP_JPY | 3.0 | 1.9 | 2.06 | **3.96** | 3.37 | +0.96 worse |
| CHF_JPY | 2.7 | 1.6 | 2.06 | **3.66** | 3.07 | +0.96 worse |

The structural reason: the ~1.1p spread markup OANDA removes is worth $11/100k on USD-quote pairs, but the flat $14 round-turn commission it adds is worth **2.06p on JPY-quote pairs** (pip = 1000 JPY ≈ $6.80). Core pricing at the current published rate is **more expensive than spread-only on every pair we trade**, and dramatically so on the JPY book. Even at the legacy $5/side rate it is at best break-even on EUR_USD/EUR_GBP and still worse on all JPY pairs.

## 3. Re-pricing the recorded gross-positive-but-spread-blocked results

Arithmetic on recorded per-trade figures only; no backtests re-run.

| Result (recorded) | Recorded cost basis | Net at core all-in (current $7/side) | Net at legacy $5/side | Verdict change? |
|---|---|---|---|---|
| **Structural fade** (H1 swing S/R + hi-vol, 12 pairs): gross **+1.089p/trade**, net **−0.890** → implied recorded RT cost 1.979p | real per-bar spread | 12-pair avg core all-in ≈ 0.88 raw + 1.70 comm (6 JPY @2.06 + 5 USD-quote @1.40 + EUR_GBP @1.04) = **2.58p** → net **−1.49** | all-in ≈ 2.09p → net **−1.00** | **No — gets worse.** The JPY commission conversion more than eats the spread-markup refund. |
| **BB re-entry fade**: net margin **+0.5–1.1p/trade** at measured spread (EUR/USD-anchored M5/M15/H1) | real per-bar spread | EUR_USD leg: cost 1.5→1.8 → margin **−0.3p** (e.g. +0.5 → +0.2); JPY legs −0.96p, several flip negative | ≈ +0.1p on EUR_USD, JPY legs still worse | **No — direction of change is negative** at current terms; ≈unchanged at legacy on EUR_USD only. |
| **Regime-entry contrarian tilt**: gross **+0.01–0.20p/trade** | pre-cost | any tier's floor ≥ ~1.3p RT (even 0-spread + commission) | ≥ ~1.0p | **Hopeless at any tier** — cost floor is 7–130× the gross signal. Commission-free AND spread-free execution would be required. |

## 4. Verdict

The domestic core-pricing tier **changes no verdict — it points the wrong way**. At OANDA US's current published commission ($0.70/10k/leg = $14/100k round-turn), all-in cost is *higher* than our measured spread-only medians on every pair in the book: +0.3p on USD-quote majors and +≈1.0p on JPY crosses, because the flat USD commission converts to ~2.06 pips on JPY-quote pairs. Structural fade goes from −0.89 to ≈−1.49p/trade; BB re-entry fade's thin +0.5–1.1p margin shrinks (and flips negative on JPY legs); the contrarian tilt remains 1–2 orders of magnitude below any achievable cost floor. Even under the legacy $5/side rate the best case is ~break-even on EUR_USD. Separately, eligibility requires a **$10,000 maintained balance** — our live accounts run $8–$100, so the tier is not even accessible at current sizing. The only realistic cost levers remain the ones in `spread_clock/RESULTS.md`: entry-window blackouts (0.1–0.8p/RT) and, longer-term, a genuinely cheaper venue (raw-spread + low-commission ECN) rather than OANDA's core tier.
