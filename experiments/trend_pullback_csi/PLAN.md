# Trend-Pullback + Currency-Strength (TPCS) — MTF strategy plan

**One line:** in a higher-TF trend, enter near the *bottom of a correction* (pullback to a causal
trendline / support) — but ONLY when multi-timeframe currency strength backs the trade; mirror for
downtrends; trendline break = invalidation/exit. Package the user's GBP_JPY scenario into one testable
system.

## Honest prior (read before building)
The BARE version is a known negative here: `project_asi_trendline_h1` (HH/HL + trendline third-touch entry,
stop below prior low, exit on break, 12-pair H1 WF) = **−11.8 to −53.9 p/d**, "entry not selective enough,
same wall as `trend_vs_contrarian`" (intraday trend/pullback never generalizes cross-pair net of spread =
USD_JPY drift). **So the experiment's REAL question is not "does trendline-pullback work" (answered: no) —
it is "does an MTF CURRENCY-STRENGTH GATE rescue it?"** That gate is the one ingredient the prior test
lacked, and currency-strength has shown real edge at longer horizons (csi_factor_study StrengthSpread H4).
**Decisive design = ablation: bare pullback vs strength-gated, same everything else.**

## R:R stance (per user)
No R:R dogma. Sub-1 reward:risk is fine if win-rate carries it — our live 009 retrace (TP20/SL30 = 0.67:1)
proves it. **TP/SL is a SWEPT parameter (incl. ≤1:1)**, judged by realized expectancy + AMDDP5 + WR + WF/MC,
not by the ratio. (AMDDP5 already rewards clean high-WR low-drawdown trades.)

## The 4 ingredients, packaged
1. **Trend context (HTF — 4h or D):** uptrend = rising confirmed swing structure (HH/HL) OR price above a
   positive-slope MA. Downtrend = mirror.
2. **Causal trendline (THE lookahead risk):** fit through the last 2–3 **confirmed** swing lows (uptrend) /
   highs (downtrend). Pivots confirmed only N bars after the fact (fractal). NEVER use a future pivot — the
   line at bar t uses only pivots ≤ t−N. (This is where hand-drawn trendlines cheat; we cannot.)
3. **Pullback entry (LTF — 1h/15m):** in an HTF-uptrend, price corrects toward the trendline/support;
   enter long when the correction **exhausts and turns up near the line** (bullish reversal bar / fast-MA
   turn / momentum stops falling) — "near the bottom of the correction." Mirror short in downtrend.
4. **Currency-strength gate (the matrix — the hypothesis):** take X_Y long only if MTF strength backs it,
   e.g. quote Y weak AND/OR base X strong on the entry TF and/or HTF (rank or threshold on `lib/csi.py` /
   fx_signals strength). The GBP_JPY scenario: JPY weakest ⇒ yen-cross long favoured. Tunable; ablated.
- **Trendline break = invalidation/exit** (uptrend line breaks down → close long). Optional Mode-B: a
  break + strength flip = reversal entry. Primary is Mode-A continuation; break is the natural stop.

## Exits / risk (swept, not dogmatic)
- Stop: below the correction low / trendline (structure-based — sane, not the inverted 0.43:1 manual case).
- Target: sweep {fixed R-multiple incl. 0.5R/1R/1.5R, prior swing high (structure), ATR-trail, trendline-
  break exit}. Let expectancy/AMDDP5 pick.

## Causality / SOP (non-negotiable — trendlines are a lookahead magnet)
- Swing pivots confirmed N bars late; trendline from past-confirmed pivots only (R1/R4).
- MTF alignment = last CLOSED higher-TF bar shifted forward (no merge_asof to the forming bar — the
  StrengthSpread 55-min-leak RCA). Strength from closed bars.
- Mid signals, spread deducted up front (R3); IS-only spread/strength gates (R5); OOS sealed (R8).
- One code path backtest≡live (R6); R7 consistency before any deploy.

## Test pipeline
- **Multi-pair always (12)** — never single-pair (the trendline test died partly on USD_JPY drift; the
  strength gate must prove cross-pair).
- Data: M5/S5 BA. HTF 4h/D from resample; LTF 1h/15m; strength from csi.
- Stage 0: causal swing+trendline+strength features (+ R7 check).
- Stage 1 — **ABLATION**: bare pullback vs strength-gated, IS, net p/d + AMDDP5 + WR. Does the gate flip
  it positive on multiple pairs? If not → confirms the wall, stop.
- Stage 2: param sweep (HTF def, pivot N, trendline pivots/age/tol, pullback depth/trigger, strength TF/thr,
  TP/SL incl ≤1:1) — grid-chunked on survivors, WF (3 IS chunks) + MC + surrogate-null.
- Stage 3: full validation → paper. Gate: OOS net>0 + AMDDP5>0 + WF + MC<0.05 + beats surrogate-null,
  on ≥ several pairs.

## Build order
0. Causal swing/trendline + strength-aligned feature builder (12 pairs), R7 check.
1. Ablation screen (bare vs strength-gated) — the make-or-break, cheap. If the gate doesn't rescue it, log
   the negative and stop before the full sweep.
