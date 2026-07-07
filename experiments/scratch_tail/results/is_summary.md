# SMA-Scratch Tail-Bounding Test — IS Battery Summary

Governed by `PREREGISTRATION.md` (LOCKED 2026-07-07). IS window: 2020-11-11 -> 2024-09-25 (OOS sealed, never read for arm evaluation — every loader routes through `is_data.load_pair_is()`). Gate 2 is the sole exception (2026-06-15+ live paper trail, used only for R7 parity).

## Gate 1 — harness self-test (synthetic RW), PASS (test_harness.py)

Full-population (closed trades UNION open-at-end mark-to-window-end) gross P&L shows no phantom directional edge for either the real signal or the coin-flip control, on a driftless random walk, for both arm A (no stop) and arm D's coin control (stop+overlay active). A companion regression test documents — as an expected, structural finding, not a bug — that arm A's CLOSED-trades-ONLY gross mean IS materially positive on the same random walk (the well-known TP-only/no-stop survivorship artifact: closed trades are winners/scratches by construction; losers sit in the open book). See harness.py's module docstring for the full design-finding note, including the self-referential-deadlock bug this surfaced and its fix (a gated arm's blocking SIGNAL must come from an always-on REFERENCE run, not its own necessarily-thinner realized trade sequence).

## Gate 2 — R7 parity vs the live paper trail (BLOCKING)

**FAIL**. Live window [2026-06-18 22:02:38+00:00, 2026-07-06 16:05:54+00:00]: 112 live closed trades vs 111 harness-replayed trades (ratio=0.99, tolerance ±10%, PASS). Live expectancy +8.70p/trade vs harness +4.56p/trade (diff=-4.14p, tolerance ±1.0p, FAIL).

## Gates 3-5 (IS-only)

| Gate | Name | Result | Detail |
|---|---|---|---|
| 3 | Arm A reproduces sign/pathology (open-book excursions visible) | **PASS** | A: n=849 net=+3.709p WR=64% worst_excursion=-7100.3p open_at_end=6 open_book_unrealized=-12420.1p |
| 4 | Arm D: 3 H1 criteria + WF thirds >=2/3 | **FAIL** | crit1(net>0,CI excl 0)=False [boot_mean=-2.842p CI=(-3.675,-2.033)] | crit2(beats coin_D,CI excl 0)=False [diff_CI=(-1.336,+0.797)] | crit3(floatDD<=50% of A)=False [ddA=14258.6p ddD=33319.6p] | WF=False [0/3, thirds=[-1.975, -3.156, -3.234]] |
| 5 | Overlay-on-coin control clean (no sign-flip, no worse DD) | **FAIL** | coin_A net=+2.545p (n=704) vs coin_overlay net=+2.167p (n=445) | ddCoin=17904.5p ddOverlay=18941.4p | no_sign_flip=True dd_not_worse=False |

## Per-arm portfolio table (pooled across 6 pairs, IS window, base cost)

| Arm | n | WR | net/trade | gross/trade | realized P&L | worst open excursion | open-at-end | open-book unrealized | floating maxDD | % time blocked |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 849 | 64% | +3.71p | +6.83p | +3149.11p | -7100.30p | 6 | -12420.10p | -14258.57p | 0.0% |
| E | 20290 | 65% | -2.47p | +0.03p | -50139.35p | -367.20p | 4 | -32.50p | -50339.94p | 0.0% |
| coin_A | 704 | 59% | +2.54p | +5.08p | +1791.48p | -6550.80p | 6 | -13095.50p | -17904.53p | 0.0% |
| coin_E | 19967 | 65% | -2.81p | -0.20p | -56083.24p | -334.60p | 3 | -10.60p | -56350.99p | 0.0% |
| B | 431 | 63% | +4.13p | +7.10p | +1781.21p | -7090.20p | 6 | -12358.90p | -16381.25p | 39.2% |
| C | 801 | 60% | +2.79p | +5.32p | +2237.41p | -7100.30p | 6 | -11138.40p | -14951.99p | 56.2% |
| D | 11666 | 65% | -2.85p | -0.39p | -33190.54p | -334.60p | 1 | -51.90p | -33319.64p | 61.3% |
| coin_D | 9468 | 64% | -2.59p | -0.09p | -24486.67p | -270.60p | 2 | -77.50p | -24645.13p | 61.7% |
| coin_overlay | 445 | 61% | +2.17p | +4.86p | +964.22p | -6902.70p | 6 | -13637.90p | -18941.41p | 55.4% |

Arm D day-block bootstrap: mean=-2.84p, 95% CI=(-3.67, -2.03). Arm D minus coin_D bootstrap diff 95% CI=(-1.34, 0.8).

Arm D walk-forward thirds (net/trade): third 1 (3330n)=-1.98p, third 2 (4420n)=-3.16p, third 3 (3916n)=-3.23p

## Per-pair — arm A

| Pair | n | WR | net/trade | gross/trade | realized P&L | worst open excursion | open-at-end |
|---|---|---|---|---|---|---|---|
| USD_JPY | 124 | 77% | +11.63p | +14.35p | +1442.72p | -4906.20p | 1 |
| NZD_USD | 201 | 70% | +5.76p | +8.21p | +1158.55p | -925.30p | 1 |
| GBP_USD | 231 | 61% | -0.05p | +4.73p | -10.67p | -2780.30p | 1 |
| CAD_JPY | 67 | 57% | +5.01p | +7.20p | +335.60p | -3683.20p | 1 |
| AUD_USD | 159 | 55% | +1.18p | +3.33p | +188.00p | -1477.80p | 1 |
| GBP_JPY | 67 | 58% | +0.52p | +3.90p | +34.91p | -7100.30p | 1 |

## Per-pair — arm D

| Pair | n | WR | net/trade | gross/trade | realized P&L | worst open excursion | open-at-end |
|---|---|---|---|---|---|---|---|
| USD_JPY | 1916 | 75% | -2.18p | -0.30p | -4179.61p | -252.00p | 0 |
| NZD_USD | 1230 | 60% | -2.84p | -0.93p | -3497.69p | -103.80p | 0 |
| GBP_USD | 1878 | 64% | -2.64p | -0.48p | -4961.52p | -231.40p | 0 |
| CAD_JPY | 1997 | 60% | -2.54p | -0.11p | -5068.39p | -182.10p | 1 |
| AUD_USD | 1445 | 56% | -2.24p | -0.69p | -3231.08p | -110.20p | 0 |
| GBP_JPY | 3200 | 67% | -3.83p | -0.23p | -12252.25p | -334.60p | 0 |

## Verdict

Gates 3-4-5 = PASS/FAIL/FAIL (gate 1 = PASS, gate 2 = see above, BLOCKING).
Gate 3 confirms the pathology arm D is meant to fix: arm A (no stop) marks -12420.10p unrealized across 6 never-closed positions at IS end, worst single open excursion -7100.30p — the unbounded tail is real on IS data, not just the live anecdote.
Gate 4 fails on all four sub-criteria, not narrowly: arm D's day-block-bootstrap net is negative and CI-excludes-zero on the WRONG side (-2.85p/trade), it does not beat coin_D (-2.85p vs -2.59p, diff CI spans zero), floating maxDD is 2.3x arm A's (-33319.64p vs -14258.57p) — the OPPOSITE of the ≤50% claim — and 0/3 WF thirds are positive. Gate 5 fails narrowly on DD (overlay-on-coin ~5.8% worse than plain coin, no sign flip).
**Recommendation: STOP, do not request OOS unseal.** This is not a close call on this frozen-parameter configuration — the floating-overlay+3xATR-stop treatment does not merely fail to help, it turns arm A's modest positive expectancy negative while *amplifying* (not bounding) floating drawdown. Per the pre-registration's decision rule ('H1 fails but tail metrics confirm the pathology'), the keeper is the tail-pathology finding itself (gate 3) plus this IS-only falsification of the disaster-stop+overlay fix — not a strategy to carry to OOS. `fx-sma-scratch-paper` stays stopped (Amendment 3).

This run also produced a durable methodological finding independent of the gate outcome: prospective (order-blocking, not merely paper-tracked) equity-curve overlays are self-referentially unstable unless their gating signal is sourced from an always-on reference run — see harness.py's module docstring. This generalizes beyond scratch_tail to any future prospective equity-MA gate design (e.g. an eventual live port of the `equity_switch_monitor` pattern beyond its current paper-only observation role).
