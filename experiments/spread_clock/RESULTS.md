# Spread Clock — Hour-of-Week Execution-Cost Study

**Date:** 2026-07-07 · **Type:** cost measurement (not an edge claim; no pre-registration)
**Data:** 12-pair OANDA M5 bid/ask parquets, 2020-11-11 → 2026-05-21 (~412K bars/pair), spread = `(ask_c − bid_c)/pip` at bar close — the same definition used by the live traders (R3a) and all recorded backtests.
**Code:** `spread_clock.py` (run on Hetzner, `/root/work/code_clock/`). Outputs: `profiles.parquet` (168 hour-of-week buckets × 12 pairs, median/p25/p75/n), `summary.csv`, `toll.csv`.

## Method

- Bucket = UTC day-of-week × 24 + hour (Mon=0). Per bucket: median, p25, p75, n over full history. Bars with spread ≤0 or ≥50p dropped (0 dropped in practice).
- **Liquid bucket** = n ≥ 25% of the pair's max bucket count (removes weekend-gap buckets). 121/168 buckets are liquid for every pair (the market is closed ~Fri 22:00 → Sun 20:59 UTC).
- **8h windows**: circular scan; eligible only if all 8 buckets liquid; score = mean of bucket medians.
- **Toll model**: one round trip costs one full spread (buy at ask, sell at bid). (a) uniform-random liquid hour = mean of liquid-bucket medians; (b) cheapest liquid hour = min; (c) worst liquid hour = max.

## 1. Per-pair profile (summary.csv)

| Pair | Cheapest 8h (UTC) | med | Most expensive 8h (UTC) | med | Ratio | Overall med |
|---|---|---|---|---|---|---|
| AUD_JPY | Mon 06–14 | 1.80 | Sun 21–Mon 05 | 3.43 | 1.90 | 1.9 |
| AUD_USD | Mon 07–15 | 1.30 | Sun 21–Mon 05 | 1.93 | 1.48 | 1.3 |
| CAD_JPY | Mon 12–20 | 1.95 | Sun 21–Mon 05 | 4.06 | 2.08 | 2.1 |
| CHF_JPY | Tue 10–18 | 2.55 | Sun 21–Mon 05 | 5.54 | 2.17 | 2.7 |
| EUR_GBP | Mon 07–15 | 1.40 | Sun 21–Mon 05 | 2.59 | 1.85 | 1.4 |
| EUR_JPY | Tue 11–19 | 1.90 | Sun 21–Mon 05 | 3.88 | 2.04 | 2.0 |
| EUR_USD | Mon 10–18 | 1.50 | Sun 21–Mon 05 | 2.06 | 1.38 | 1.5 |
| GBP_JPY | Wed 09–17 | 2.87 | Sun 21–Mon 05 | 4.96 | 1.73 | 3.0 |
| GBP_USD | Tue 11–19 | 1.79 | Sun 21–Mon 05 | 3.34 | 1.87 | 1.8 |
| NZD_JPY | Mon 09–17 | 2.30 | Sun 21–Mon 05 | 4.53 | 1.97 | 2.4 |
| NZD_USD | Mon 06–14 | 1.50 | Sun 21–Mon 05 | 2.51 | 1.67 | 1.5 |
| USD_JPY | Mon 12–20 | 1.60 | Sun 21–Mon 05 | 2.64 | 1.65 | 1.6 |

**Pattern.** The *most expensive* 8h window is **Sun 21:00–Mon 05:00 UTC for every pair** (weekend re-open + thin Monday Asia). The "cheapest" window's day-of-week label is mostly noise — the hour-of-day pattern dominates: Tue–Thu hour-of-day medians are essentially **flat from 00:00–20:00 UTC** (EUR_USD: 1.5p at every hour 0–20) and spike only at **21:00–23:00 UTC daily** (the 5pm-NY rollover: CHF_JPY 11.5p, GBP_JPY 10.5p median at 21:00 even midweek, vs 2.5–2.9p intraday). So the actionable clock is two blackouts, not a golden hour:

1. **Daily 21:00–23:00 UTC** (rollover) — spreads 2–5× intraday medians on JPY crosses.
2. **Sun 21:00–Mon ~05:00 UTC** (weekend open) — worst single liquid bucket is Sun 21:00 on all 12 pairs (5.2–20p medians).

## 2. Toll saving — trade once/day, choose the hour (toll.csv)

Round-trip toll in pips; savings vs the 1.6p EUR_USD reference:

| Pair | (a) uniform | (b) cheapest | (c) worst | Save (a−b) | % of 1.6p |
|---|---|---|---|---|---|
| AUD_JPY | 2.11 | 1.80 | 12.7 | 0.31 | 19% |
| AUD_USD | 1.41 | 1.30 | 5.5 | 0.11 | 7% |
| CAD_JPY | 2.39 | 1.90 | 15.7 | 0.49 | 31% |
| CHF_JPY | 3.30 | 2.50 | 20.0 | 0.80 | 50% |
| EUR_GBP | 1.63 | 1.40 | 9.1 | 0.23 | 14% |
| EUR_JPY | 2.30 | 1.90 | 15.0 | 0.40 | 25% |
| EUR_USD | 1.59 | 1.50 | 5.2 | 0.09 | 6% |
| GBP_JPY | 3.44 | 2.80 | 15.0 | 0.64 | 40% |
| GBP_USD | 2.07 | 1.70 | 11.9 | 0.37 | 23% |
| NZD_JPY | 2.73 | 2.30 | 17.4 | 0.43 | 27% |
| NZD_USD | 1.67 | 1.50 | 8.1 | 0.17 | 10% |
| USD_JPY | 1.78 | 1.60 | 8.5 | 0.18 | 11% |

- Uniform → cheapest saves **0.09–0.80 p/round-trip** (6–50% of the 1.6p reference). Biggest on JPY crosses (CHF_JPY 0.80p, GBP_JPY 0.64p), near-nil on EUR_USD/AUD_USD.
- Most of the "uniform" penalty comes from the rollover/Sunday buckets — a blackout (avoid 21:00–23:00 + Sun open) captures the bulk of the saving without needing a specific golden hour, since intraday spreads are flat.
- The asymmetric tail matters more: worst-hour toll is **3–8× uniform** (Sun 21:00 = 5.2–20p). Not avoiding it is far costlier than optimizing within the flat intraday band.

## 3. Reality check — what our recorded books actually paid for hour choice

From `docs/data/trades_snapshot_2026-05-31.csv` (live trades, `is_paper=false`, n=4,932; the CSV has no per-trade spread, so toll is **estimated** = the pair's hour-of-week median at `entry_time`). Excess = mean est. toll − pair's cheapest liquid hour.

| Strategy | n | Mean est toll | Excess/trade | Total excess (p) |
|---|---|---|---|---|
| range_neat (pre-RCA) | 819 | 10.29 | 8.09 | 6,628 |
| asi_mc_008 (pre-RCA) | 794 | 10.52 | 8.31 | 6,602 |
| wf_pnlmae (pre-RCA) | 784 | 10.59 | 8.39 | 6,578 |
| pnl_mae_a (pre-RCA) | 766 | 10.78 | 8.57 | 6,568 |
| ensemble_short | 357 | 4.46 | 2.43 | 867 |
| perpair_a | 242 | 4.82 | 2.88 | 697 |
| ironnet_v3_er_003 | 165 | 3.81 | 1.78 | 293 |
| ironnet_v3_perpair | 165 | 3.76 | 1.73 | 285 |
| (all others ≤106 trades) | — | 1.8–3.0 | 0.03–1.3 | ≤45 each |
| **TOTAL live** | **4,932** | — | — | **≈28,800** |

**The headline is historical, and it's big:** 37% of all recorded live entries (1,830) fired in the 21:00 UTC bucket at ~17.6p estimated spread — **1,544 of them Sunday 21:00**, i.e. the pre-RCA NEAT book re-entering at the weekend open. That book (range_neat/asi_mc_008/wf_pnlmae/pnl_mae_a, 63% of entries at 21:00–23:59) paid an estimated **~26,400p of avoidable toll** — a material slice of the −55K RCA loss was execution timing, on top of the lookahead. Everything deployed since is already near-optimal: post_shock_retrace excess 0.16p/trade, tr_live 0.05, sma_momentum 0.22; the paper book totals only ~176p excess (zr_paper worst at 0.81p/trade). The current live strategies (sma_stack 010, sma_fade 001, both live after this snapshot) are M5-signal-driven around the clock and have no blackout.

## Actionable summary

1. **Add a two-window entry blackout to all live/paper strategy containers: no new entries 21:00–23:00 UTC daily, and Sun 21:00–Mon 05:00 UTC.** Cost: near zero signal loss (spreads are flat 00–20 UTC, so the blocked hours are exactly the expensive ones). Benefit: caps the 3–8× worst-hour toll and captures most of the 0.1–0.8 p/RT scheduling saving.
2. **Per-strategy harvest at current books:** small — 010/001 trade ~1–3 round-trips/day/pair on majors+EUR_JPY, so scheduling is worth ~0.1–0.4 p/RT ≈ low single-digit pips/day. This does not rescue any spread-blocked edge (structural fade needs ~0.9p; only CHF_JPY/GBP_JPY scheduling approaches that, and we don't trade a CHF_JPY strategy).
3. **For any future once-a-day/multi-day book** (first-touch H4, CSI H4): schedule entries inside London/NY hours on JPY crosses for a free 0.3–0.8 p/RT — a real fraction of typical H4 edges (~5–10p/trade).
4. **Never let anything re-enter at Sunday open again** — that single bucket cost the pre-RCA book thousands of pips.

**Caveats.** Trade tolls in §3 are profile estimates at the entry hour, not measured per-trade spreads (the snapshot lacks them); exit-side timing is not scored separately (toll model assumes entry-hour spread class for the round trip); Sunday-open buckets are heavy-tailed, so medians understate the tail there.
