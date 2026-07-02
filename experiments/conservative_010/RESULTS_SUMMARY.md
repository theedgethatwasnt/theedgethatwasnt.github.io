# Conservative 010 — Entry/Exit Study Compilation (2026-06-30 → 07-01)

Conservative engine: real per-bar bid/ask spread, mid signals + worse-side fills + 2p stop
slippage, sealed IS/OOS (4/6 split), 6-fold walk-forward (WF), Monte-Carlo bootstrap (MC,
n=300). 4 pairs: EUR_JPY / EUR_USD / GBP_USD / USD_JPY (base data S5, ~1.2–1.5 yr/pair).
Memory-safe: one pair at a time. "Confirmed" gate: net>0 + WF≥4/6 + **MC p_net<0.05** + broad
across pairs + bounded per-trade risk. Every number below is a measured backtest output.

## A. Does 010's own entry have an edge? — NO (three controls)

**A1. Random DIRECTION** (same signal timing, coin-flip direction), 500 seeds, no_flip:
- Real signal net **−3,616p**. Random-direction: mean **−1,903p**, std 1,904; pct 5/50/95 =
  −4,858 / −1,850 / +1,304; min/max −6,682 / +3,729. **P(random≥real)=0.802, z=−0.90.**
- Per-pair real vs random-mean: EUR_JPY −878/+305 · EUR_USD −742/−595 · GBP_USD −300/−793 ·
  USD_JPY −1,696/−821. (3/4 pairs: real WORSE than random.)

**A2. Random MARKET ENTRY** (random time + random direction, rate matched), 500 seeds:
- Portfolio mean **−1,901p**, std 1,884; pct 5/50/95 = −5,237 / −1,849 / +1,025; min/max
  −8,424 / +3,938. **P(random≥real)=0.824, z=−0.91.**
- Per-pair random means: EUR_JPY +110 · EUR_USD −67 · GBP_USD −1,045 · USD_JPY −898.
  Avg trades: 283/188/236/490.

**Verdict:** random-timing (−1,901) == random-direction (−1,903) to within 2p → neither WHEN
nor WHICH-WAY the entry fires carries information. Both = a **~−1,900p cost floor**; real is below it.

## B. Exit sweep on RANDOM entries (portfolio net, 100 seeds) — fade > trend, none survives

| exit | net | ~trades | | exit | net | ~trades |
|---|---|---|---|---|---|---|
| TP50+SL_BIG | +2,833 | 57 | | TP20+PSAR20+SL200 | −526 | 1,113 |
| TP20+SL_BIG | +2,833 | 142 | | TP50+SL100 | −1,035 | 933 |
| TP100+SL_BIG | +2,807 | 28 | | TP20+SL100 | −1,859 | 1,853 |
| TP20+SL500 | +2,092 | 538 | | **CURRENT (live)** | **−1,872** | 1,198 |
| TP10+SL500 | +1,480 | 950 | | PSAR_act40+SL500 | −2,134 | 101 |
| TP100+SL200 | +94 | 271 | | TP200+SL50 | −2,320 | 529 |
| TP50+PSAR20+SL300 | +85 | 516 | | PSAR_act20+SL500 | −2,509 | 184 |
| TP50+SL200 | −79 | 513 | | PSAR_act10+SL500 | −2,763 | 350 |
| | | | | SL100/300/500/150/50/200 | −3,014…−3,186 | 6–61 |
| | | | | TP100+SL50 | −3,431 | 949 |

**Risk-reality check (worst SINGLE-trade drawdown):**
- TP50+**SL_BIG** (the "+2,833"): worst trade EUR_JPY −2,937 · EUR_USD −1,869 · GBP_USD −1,159 ·
  USD_JPY −2,021 → **finite-margin trap** (liquidation). Per-pair net +964/+297/+174/+1,466.
- TP50+**SL200** (bounded): worst trade −281/−281/−225/−279. Per-pair net +435/+119/−385/−356
  (portfolio ≈ break-even).

## C. Frozen fade harness — REAL 010 entry + fixed TP + hard SL (no PSAR/flip)

| TP/SL | net | n | worst | eqMaxDD |
|---|---|---|---|---|
| **TP150/SL150** | **−242** | 242 | −152p | −2,718 |
| TP50/SL150 | −344 | 688 | −152p | −1,802 |
| TP150/SL200 | −2,776 | 188 | −202p | −2,770 |
| TP50/SL200 | −2,926 | 511 | −202p | −3,408 |
| TP100/SL150 | −3,072 | 375 | −152p | −3,424 |
| TP100/SL200 | −3,304 | 275 | −202p | −3,398 |

TP150/SL150 validation: IS −614 / OOS +372 (WR 52%), WF **2/6**, **MC p_net=0.530**, p_maxdd 0.390.
Margin sim: $100→$109.08, $500→$544.86, $1000→$1044.86 (all survive). Bleed halted (−3,616→−242),
risk bounded (−152p), but **no edge** (MC 0.53). This is the fixed exit for the entry comparisons.

## D. New entries on SMA50 ±1σ (M5/M30/H1) @ TP50/SL150 (vs SMA-stack baseline −344p)

**D1. PULLBACK (idea #1):** slope↑ + low≤lower → long; mirror short.

| TF | net | n | IS/OOS | WR | WF | MC p_net | eqMaxDD | per-pair EJ/EU/GU/UJ |
|---|---|---|---|---|---|---|---|---|
| **M5** | **+1,792** | 658 | +1094/+698 | 77% | 4/6 | **0.200** | −1,334 | −1000/+908/+838/+1046 |
| M30 | −3,702 | 435 | −3074/−628 | 73% | 2/6 | 0.973 | −5,258 | −1016/+410/−1204/−1892 |
| H1 | −1,890 | 346 | −2640/+750 | 78% | 3/6 | 0.910 | −3,164 | +606/−136/+616/−2976 |

M5 grid (net): TP30/60 −908 · TP50/100 −354 · **TP50/150 +1,792** · TP150/150 +64 (config standout).

**D2. BREAKOUT (idea #2):** slope just turned ↓ + close<lower → short; mirror long.

| TF | net | n | IS/OOS | WR | WF | MC p_net | eqMaxDD | per-pair EJ/EU/GU/UJ |
|---|---|---|---|---|---|---|---|---|
| **H1** | **+2,756** | 245 | +1686/+1070 | 82% | **5/6** | **0.010** ✅ | −884 | +116/+1890/−24/+774 |
| M30 | +530 | 354 | −320/+850 | 79% | 3/6 | 0.367 | −1,402 | +1010/−642/+416/−254 |
| M5 | −2,122 | 608 | −2172/+50 | 75% | 2/6 | 0.823 | −3,452 | +1174/−960/−2884/+548 |

H1 grid (net, ALL positive): TP30/60 +1,040 · TP50/100 +3,158 · **TP50/150 +2,756** · TP150/150 +1,978.

## E. Pullback + H1 filter (robustness test) — REFUTES idea #1

| variant | net | n | IS/OOS | WR | WF | MC p_net | per-pair EJ/EU/GU/UJ |
|---|---|---|---|---|---|---|---|
| ungated | +1,792 | 658 | +1094/+698 | 77% | 4/6 | 0.200 | −1000/+908/+838/+1046 |
| **H1-trend filter** | **−2,320** | 600 | −806/−1514 | 72% | 2/6 | 0.837 | −1400/+760/−1272/−408 |
| H1-strict confluence | −376 | 146 | −806/+430 | 80% | 3/6 | 0.633 | −224/+288/+690/−1130 |

A real "dip-in-uptrend" edge would improve when the H1 trend must agree; it collapses instead
→ the +1,792 was not trend-aligned; with MC 0.20 + config-standout it is most likely noise.

## F. Rarity & multi-TF confluence (entries/yr/pair)

**Single-TF signal rate:** Pullback M5 8.21% of bars (6,128/yr) · M30 7.63% (947) · H1 8.27% (511).
Breakout M5 1.72% (1,282) · M30 1.54% (191) · H1 1.54% (95).

**Confluence entries/yr/pair (S30/M1/M5/M30/H1):**
- Pullback 2-TF: S30+M1 1153 · S30+M5 631 · S30+H1 949 · M1+H1 449 · M1+M5 425 · M1+M30 290 ·
  **M5+M30 69 · M5+H1 64 · M30+H1 31**. 3-TF: M5+M30+H1 **4.9** (too rare). Singles/yr:
  S30 22,482 · M1 11,483 · M5 2,442 · M30 371 · H1 182.
- Breakout 2-TF: S30+M1 181 · S30+H1 93 · S30+M30 75 · S30+M5 62 · M1+M5 53 · **M5+M30 8.5 ·
  M30+H1 4.9 · M5+H1 3.4**. 3-TF ≈ 0. → breakout confluence untradeable; use H1 alone / trend filter.

## Standings

- **Entry (010 SMA-stack): no edge** (A). **Exit: no edge, best bounded ≈ break-even** (B/C).
- **Pullback (idea #1): REFUTED** (E). **Breakout H1 (idea #2): sole surviving lead** — MC 0.010,
  grid-robust, WF 5/6, bounded −884 DD — caveats: EUR_USD = +1,890 of +2,756 (69%); momentum
  contradicts priors; OOS spent (R8); ~1.2-1.3yr/pair.

## G. H1 breakout — band Kσ × exit sweep + drop-EUR_USD test — SURVIVES concentration

4-pair vs 3-pair (EUR_USD dropped). WF/6 + MC p_net. (OOS spent per R8 — WF/MC are the gates.)

| K | TP/SL | 4p net | n | WF | MCp | ex-EU net | WF | MCp |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 30/60 | +1,040 | 372 | 3/6 | 0.097 | +238 | 2/6 | 0.380 |
| **1.0** | **50/100** | **+3,158** | 279 | **6/6** | **0.000** | **+1,884** | **5/6** | **0.013** ✅ |
| 1.0 | 50/150 | +2,756 | 245 | 5/6 | 0.010 | +866 | 4/6 | 0.220 |
| 1.0 | 100/150 | +2,528 | 179 | 4/6 | 0.060 | +340 | 3/6 | 0.420 |
| 1.0 | 150/150 | +1,978 | 136 | 4/6 | 0.147 | +942 | 3/6 | 0.260 |
| 1.5 | 50/150 | +2,632 | 190 | 5/6 | 0.010 | +1,188 | 5/6 | 0.120 |
| 1.5 | 100/150 | +2,504 | 146 | 4/6 | 0.073 | +1,164 | 2/6 | 0.163 |
| 2.0 | 50/150 | +1,354 | 120 | 5/6 | 0.087 | +658 | 4/6 | 0.167 |
| 2.5 | (all) | −400…+626 | 49–75 | ≤5/6 | ≥0.13 | mostly neg | — | — |

**Findings:** (1) **Exit revisit matters more than the band** — TP50/**SL100** (tighter 1:2 stop),
not the inherited SL150, is best: 4p +3,158, WF **6/6**, MC **0.000**, and — critically — it
**survives dropping EUR_USD** (ex-EU +1,884, WF 5/6, MC **0.013**). TP50/SL150 does NOT survive
(ex-EU p=0.22). (2) **Band K=1 (or 1.5) is right; K≥2 degrades** (fewer trades, loses
significance). (3) Wider bands don't help — the edge is in the fresh slope-flip + a modest break,
not extreme extension.

**Upgrade:** H1 breakout @ K=1 **TP50/SL100** is the first config positive+MC<0.05 on all 4 pairs
AND still significant without the pair that carried most of the P&L. Concentration caveat largely
resolved. Remaining: OOS spent (R8) → needs fresh data / out-of-sample confirmation; momentum
contradicts priors; verify ex-EU sign spread across EJ/GU/UJ.

## H. SMA50 vs EMA50 band center (H1 breakout, K=1) — EMA is strictly WORSE

| MA | TP/SL | 4p net | n | WF | MCp | ex-EU net | MCp | per-pair EJ/EU/GU/UJ |
|---|---|---|---|---|---|---|---|---|
| **SMA** | **50/100** | **+3,158** | 279 | **6/6** | **0.000** | **+1,884** | **0.013** | **+354/+1274/+416/+1114** |
| SMA | 50/150 | +2,756 | 245 | 5/6 | 0.010 | +866 | 0.220 | +116/+1890/−24/+774 |
| SMA | 30/60 | +1,040 | 372 | 3/6 | 0.097 | +238 | 0.380 | −98/+802/−430/+766 |
| EMA | 50/100 | −368 | 172 | 3/6 | 0.630 | −644 | 0.813 | −790/+276/−136/+282 |
| EMA | 50/150 | −1,188 | 154 | 2/6 | 0.833 | −1,420 | 0.950 | −1130/+232/−578/+288 |
| EMA | 30/60 | +110 | 203 | 4/6 | 0.410 | −42 | 0.550 | −128/+152/−72/+158 |

**EMA is strictly worse at every exit** (+3,158→−368 at the winner). Keep SMA. Reason: EMA50's
reactivity makes the slope flip early/often (noisy crosses) and the band chase price; the slow
SMA only fires on a *sustained* turn — consistent with the Oracle "mild/meandering" reward.

**Refinement (from the per-pair column):** at the winner **TP50/SL100**, ALL FOUR pairs are
positive (EJ +354 / EU +1274 / GU +416 / UJ +1114) and EUR_USD is only **40%** of net — not the
69% seen at SL150. Tightening the stop broadened the edge; the concentration worry is largely resolved.

## I. OUT-OF-SAMPLE confirmation on 8 UNSEEN pairs — REFUTES the H1 breakout

Frozen config (K=1σ/SMA50/TP50/SL100 H1 breakout) run once on the 8 pairs never in the config
search. Nothing tuned.

| | pairs + | net | n | WF | MC p_net |
|---|---|---|---|---|---|
| IS (4 tuned pairs) | 4/4 | +3,158 | 279 | 6/6 | 0.000 |
| **OOS (8 unseen pairs)** | **3/8** | **−1,116** | 534 | 2/6 | **0.783** |

OOS per-pair: AUD_JPY +4 · AUD_USD −388 · CAD_JPY +502 · CHF_JPY −724 · EUR_GBP −370 ·
GBP_JPY −308 · NZD_JPY −458 · NZD_USD +626. (IS block reproduced +3,158 exactly → loader faithful.)

**The edge does NOT generalize.** It lived entirely in the 4 pairs it was selected on. The
+3,158/MC p=0.000 was in-sample selection bias (config chosen from ~dozens of grid cells). OOS =
3/8 positive, net −1,116, MC p=0.78 — noise. High per-pair WR (62–76%) but net-negative = the
project's signature (small wins, SL100 tail, no edge net of spread).

## FINAL STANDING (2026-07-01) — no deployable edge; H1 breakout REFUTED out-of-sample

- Entry (SMA-stack): no edge (A). Exits: no edge, bounded best ≈ break-even (B/C).
- Pullback (#1): refuted by H1-filter (E). **Breakout (#2): looked best in-sample (4/4, MC 0.000)
  but REFUTED out-of-sample (3/8, MC 0.78) — overfit to the 4-pair search.**
- **Nothing here is deployable.** The BB50 exploration confirms the project wall. Discipline held:
  no deployment on the in-sample result; the pair-dimension OOS test caught the overfit before any
  money moved. Broker time-dimension fetch NOT warranted (failed the cheaper pair test).
- Reusable: the whole harness (random-entry controls, fade harness, Kσ×exit sweep, SMA/EMA,
  confluence, **frozen-config OOS-on-unseen-pairs = the key anti-overfit test**), all in this dir.

## J. Catalogue of ALL entry conditions tried (2026-07-01)

A. Direction/timing controls: (1) SMA-stack novelty [no edge −3,616]; (2) random direction
[worthless]; (3) random timing+direction [worthless; ~−1,900 cost floor].
B. Distance-from-MA / band (SMA50±1σ): (4) pullback slope-up+touch-lower→long [M5 +1,792 IS,
MC0.20, REFUTED H1-filter & OOS 3/8]; (5) breakout slope-flip+close-beyond-band [H1 +2,756
MC0.010 IS, REFUTED OOS 3/8 & EUR_USD time-OOS +22].
C. Band width: (6) K∈{1,1.5,2,2.5}σ [K=1-1.5 best, K≥2 worse].
D. Multi-TF: (7) all-3 confluence [too rare ~5/yr]; (8) pairwise confluence M5+H1/M5+M30/S30-M5/
M1-M30 [breakout confluence untradeable]; (9) H1-trend filter on M5 pullback [WORSE −2,320];
(10) H1-strict confluence [−376].
E. Band-width: (11) width vs recent-min (squeeze) [no rescue]; (12) width slope expand/contract
[one weak IS bucket].
F. Prior-run quality: (13) rolling-std returns smooth/choppy [monotone IS +1,914 vs +394, EU-dep];
(14) velocity shallow/sharp [monotone, milder better, IS]; (15) duration/sustained rlen [NOT
monotone — 'mild' not 'long']; (16) path efficiency [metric buggy]; (17) AMDDP5-into-entry
momentum-quality [earlier journey, IC+0.13 sub-spread].
G. Overshoot/mean-reversion (SMA200/S5): (18) S5 pullback [dead −14,142 1/12]; (19) S5 breakout
[dead −2,404 5/12]; (20) deep-overshoot fade→mean target [shelved, not nimble]; (21) spread-unit
reversion ≥N-spreads/10s [dead, WR 45-52% coin flip].
H. Move-origin conditioning (on #21): (22) origin distance-from-SMA200 near/far [near-mean WORST
41% WR — refutes reversion]; (23) side above/below [both neg]; (24) extending-away vs toward [both neg].

VERDICT: all negative or OOS-refuted. Only real-but-IS signals were contrarian (pullback,
prior-run smoothness); none survived OOS. Carry-forward candidate = prior-run smoothness
conditioner, to be built into a fresh entry and run through the OOS-on-unseen-pairs gate.
