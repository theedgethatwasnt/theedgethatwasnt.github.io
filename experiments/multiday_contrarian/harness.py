#!/usr/bin/env python3
"""
harness.py — Multi-day contrarian program (Workstream A3): first-touch signal (ported
VERBATIM from services/strategy_first_touch_paper/main.py, live since 2026-06-18) + the
pre-registered 3-arm backtest engine.

R6 ("one code path"): `process_h4_bar(state, bar)` is the exact entry-condition logic from
the live paper service's `_on_new_bar` (L/EPS/VW/TOUCH_MAX/VREL_MAX, Wilder ATR), stripped of
its own position-management (the live service manages TP/SL/timecap at H4 resolution inline;
this backtest re-simulates those same 2xATR/12-bar-cap PARAMETERS at M5 resolution — see
"Deliberate resolution divergence from the live service" below, an R9-documented choice, not
an accidental one).

Frozen parameters (verbatim, PREREGISTRATION.md "Frozen entry/exit parameters"):
  L=25 H4 bars, EPS=12 pips, VW=20 H4 bars, VREL_MAX=1.16 (live service's IS-median
  threshold — this IS the "low-volume gate" the pre-reg's table describes in prose;
  TOUCH_MAX=1 (first touch only), TGT_ATR=SL_ATR=2.0 (2x Wilder ATR(14,H4)), HCAP=12 H4 bars
  (~48h), fade direction (short at resistance touch, long at support touch).

Deliberate resolution divergence from the live service (documented per R9):
  The live paper service checks its own TP/SL/timecap only once per completed H4 bar (it
  polls every 5 min but only ACTS on newly-completed H4 candles), so its realized exit
  resolution is H4 (4h). The pre-registration explicitly upgrades backtest exit-barrier
  scanning to M5 resolution ("barriers ... scanned on M5 highs/lows") for more realistic
  fill/gap modeling — this is a documented, PRE-REGISTERED refinement of the backtest over
  the live service's own coarser exit checks, not a live/backtest inconsistency to fix.

Entry fill (R3a): next M5 bar's OPEN after the signal H4 bar's close, on MID price (R3: signal
and P&L both built on mid; spread deducted explicitly and separately, never mixed into the
price series). TP/SL are computed off that same mid entry price using the ATR measured AT the
H4 signal bar (`atr_e`), matching the live service's own `entry ± TGT/SL_ATR*atr` construction.

Barrier scanning (R2 — SL-before-TP same-bar; gap-through-stop realistic fill):
  Within each M5 bar (mid OHLC), SL is checked before TP (conservative: if both thresholds
  are technically crossed in one bar we cannot know the true intrabar path without tick data,
  so we assume the worse outcome). If the bar's own OPEN has already gapped through the stop
  level, the fill is the bar's OPEN (gap slippage realized, matching R9: this is the direction
  a real stop order would actually fill, not the idealized stop price). Target gaps are NOT
  given the same favorable treatment — a bar that gaps THROUGH the target still fills at the
  nominal TP level (conservative; the pre-reg only authorizes realistic/adverse gap-fill for
  stops, not favorable gap-fill for targets).

Time cap: exit at the CLOSE of the H4 bar that is exactly HCAP=12 bars after the signal bar
(the live service's own `pos_bars>=HCAP` bar-count convention — a bar-count cap, not a fixed
48 calendar hours; a weekend gap still only counts as one bar toward the cap, exactly
matching the live service).

Cost model (locked, PREREGISTRATION.md "Cost model"):
  spread_rt_pips = (spread at the entry M5 bar + spread at the exit M5 bar) / 2, in pips,
    scaled by `spread_mult` (sensitivity {1.0, 1.5}) — one full round-trip charged as half at
    each leg (R3).
  carry_pips_result = carry_model.carry_pips(pair, direction, entry_ts, exit_ts, markup_mult=
    markup_mult) — SIGNED (positive = the trader receives carry, negative = pays).
  net_pips = gross_pips - spread_rt_pips + carry_pips_result
    (the pre-registration's shorthand "net = gross - spread_rt - carry" refers to a signed
    carry COST; since carry_pips() already returns the signed net EFFECT with positive =
    credit, the arithmetically-correct combination is addition — a negative carry_pips_result
    (paying) still reduces net exactly as "- carry" intends.)

Control arms (identical signal-detection pass + identical exit machinery, PREREGISTRATION.md):
  "signal"       — fade the touched level (the H1 hypothesis direction).
  "coin"         — direction chosen by a coin flip at every signal timestamp, seeded
                   (seed=20260706, one `np.random.default_rng(seed)` stream per pair — see
                   the note in `simulate_pair`).
  "continuation" — trade TOWARD the break (opposite of "signal"); expected negative,
                   ordering/sanity check only.
"""
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from bars import m5_to_h4
from carry_model import carry_pips, pip_of

# ── frozen parameters (verbatim from services/strategy_first_touch_paper/main.py) ──────────
L = 25
EPS_PIPS = 12.0
VW = 20
ATR_N = 14
TGT_ATR = 2.0
SL_ATR = 2.0
HCAP = 12
TOUCH_MAX = 1
VREL_MAX = 1.16
BUF = max(L, VW, ATR_N) + 30

ARMS = ("signal", "coin", "continuation")
DEFAULT_SEED = 20260706


@dataclass
class H4State:
    pair: str
    bars: list = field(default_factory=list)


def _atr(bars, n=ATR_N):
    """Wilder ATR over the buffer (mid OHLC) — verbatim port of _atr() in the paper service."""
    if len(bars) < n + 1:
        return 0.0
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    a = trs[0]
    alpha = 1.0 / n
    for tr in trs[1:]:
        a = a + alpha * (tr - a)
    return a


def process_h4_bar(state, bar, pip):
    """Port of the entry-condition branch of `_on_new_bar` in strategy_first_touch_paper/
    main.py (verbatim L/EPS/VW/TOUCH_MAX/VREL_MAX logic; Wilder ATR). Appends `bar` (dict with
    keys timestamp/open/high/low/close/volume) to state.bars (trimmed to BUF) and returns a
    signal dict {direction, atr, r, s, touches, vrel} if a first-touch low-volume fade fires
    on THIS bar, else None. Position management (skip while a trade is open) is the caller's
    responsibility (R6: this function only ever looks for entries, so backtest and any future
    live port share the identical decision function)."""
    state.bars.append(bar)
    if len(state.bars) > BUF:
        state.bars.pop(0)

    if len(state.bars) < max(L, VW) + 2:
        return None
    win = state.bars[-(L + 1):-1]
    prev = state.bars[-2]
    cur = state.bars[-1]
    R = max(b["high"] for b in win)
    S = min(b["low"] for b in win)
    eps = EPS_PIPS * pip
    atr = _atr(state.bars)
    if atr <= 0:
        return None
    volwin = state.bars[-(VW + 1):-1]
    vmean = float(np.mean([b["volume"] for b in volwin])) if volwin else 0.0
    if vmean <= 0:
        return None
    vrel = cur["volume"] / vmean
    if vrel > VREL_MAX:
        return None

    up = cur["high"] >= R - eps and prev["high"] < R - eps and cur["close"] <= R
    dn = cur["low"] <= S + eps and prev["low"] > S + eps and cur["close"] >= S
    if up:
        touches = sum(1 for b in win if b["high"] >= R - eps)
        if touches > TOUCH_MAX:
            return None
        d = -1
    elif dn:
        touches = sum(1 for b in win if b["low"] <= S + eps)
        if touches > TOUCH_MAX:
            return None
        d = +1
    else:
        return None

    return {"direction": d, "atr": atr, "r": R, "s": S, "touches": touches, "vrel": vrel}


def _scan_barriers(m5_open, m5_high, m5_low, m5_ts, entry_pos, direction, tp, sl, cap_ts):
    """Scan M5 bars from entry_pos (INCLUSIVE — the entry bar's own high/low after the open is
    still live price action that can hit a barrier before the bar closes) for the first of
    SL/TP, up to (not including) cap_ts. SL checked before TP in each bar (R2-conservative).
    Gap-through-stop: if the bar's own open is already past the stop, fill at that open
    (realistic slippage) — this can never fire on the entry bar itself, since entry_px IS that
    bar's open and tp/sl are defined relative to it. Returns (exit_px, exit_ts, exit_reason) or
    None if no barrier fires before cap_ts."""
    n = len(m5_ts)
    for j in range(entry_pos, n):
        if m5_ts[j] >= cap_ts:
            break
        o, h, l = m5_open[j], m5_high[j], m5_low[j]
        if direction > 0:
            if o <= sl:
                return o, m5_ts[j], "sl_gap"
            if l <= sl:
                return sl, m5_ts[j], "sl"
            if h >= tp:
                return tp, m5_ts[j], "tp"
        else:
            if o >= sl:
                return o, m5_ts[j], "sl_gap"
            if h >= sl:
                return sl, m5_ts[j], "sl"
            if l <= tp:
                return tp, m5_ts[j], "tp"
    return None


def simulate_pair(pair, m5_df, arm="signal", seed=DEFAULT_SEED, spread_mult=1.0, markup_mult=1.0):
    """Backtest one pair, one arm, over the full m5_df history. Returns a list of trade dicts.
    `seed`: the coin-flip arm draws from `np.random.default_rng(seed)` — ONE stream per call,
    so the i-th signal for every pair (run with the same seed) draws the i-th random number
    from an identically-seeded generator (simple, reproducible; PREREGISTRATION.md specifies
    a single seed=20260706 without a per-pair variant, so this is the literal reading)."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")

    pip = pip_of(pair)
    h4 = m5_to_h4(m5_df)
    h4_records = h4.to_dict("records")
    n_h4 = len(h4_records)

    m5 = m5_df.sort_values("timestamp").reset_index(drop=True)
    m5_ts = m5["timestamp"].values
    m5_open = m5["open"].values
    m5_high = m5["high"].values
    m5_low = m5["low"].values
    m5_bid = m5["bid_c"].values
    m5_ask = m5["ask_c"].values

    h4_ts_arr = h4["timestamp"].values

    rng = np.random.default_rng(seed)
    state = H4State(pair=pair)
    trades = []

    idx = 0
    in_position_until_idx = -1  # H4 index up to & incl. which we're still in a trade (skip signals)
    while idx < n_h4:
        bar = h4_records[idx]
        # Always feed the bar through (keeps the L/VW/ATR rolling buffers correct, exactly
        # like the live service which appends every bar regardless of position state) —
        # but discard any signal found while a trade is still open (one position per pair).
        sig = process_h4_bar(state, bar, pip)
        if idx <= in_position_until_idx:
            idx += 1
            continue

        if sig is not None:
            raw_dir = sig["direction"]
            if arm == "signal":
                direction = raw_dir
            elif arm == "continuation":
                direction = -raw_dir
            else:  # coin
                direction = 1 if rng.random() < 0.5 else -1

            h4_close_ts = bar["timestamp"] + timedelta(hours=4)
            entry_pos = int(np.searchsorted(m5_ts, h4_close_ts.to_datetime64(), side="left"))
            if entry_pos >= len(m5_ts):
                idx += 1
                continue  # data ends before we can even enter — drop, not a trade

            entry_ts = m5_ts[entry_pos]
            entry_px = float(m5_open[entry_pos])
            entry_spread_pips = float(m5_ask[entry_pos] - m5_bid[entry_pos]) / pip

            atr_e = sig["atr"]
            tp = entry_px + direction * TGT_ATR * atr_e
            sl = entry_px - direction * SL_ATR * atr_e

            cap_h4_idx = idx + HCAP
            if cap_h4_idx < n_h4:
                cap_bar = h4_records[cap_h4_idx]
                cap_ts = (cap_bar["timestamp"] + timedelta(hours=4)).to_datetime64()
                cap_price = cap_bar["close"]
                cap_reached = True
            else:
                # data truncated before the cap: fall back to the last available H4 close
                cap_bar = h4_records[-1]
                cap_ts = (cap_bar["timestamp"] + timedelta(hours=4)).to_datetime64()
                cap_price = cap_bar["close"]
                cap_reached = False

            hit = _scan_barriers(m5_open, m5_high, m5_low, m5_ts, entry_pos, direction, tp, sl, cap_ts)
            if hit is not None:
                exit_px, exit_ts, exit_reason = hit
                exit_pos = int(np.searchsorted(m5_ts, exit_ts, side="left"))
                exit_spread_pips = float(m5_ask[exit_pos] - m5_bid[exit_pos]) / pip
            else:
                exit_px, exit_ts, exit_reason = cap_price, cap_ts, ("timecap" if cap_reached else "data_end")
                exit_pos = int(np.searchsorted(m5_ts, exit_ts, side="right")) - 1
                exit_pos = max(0, min(exit_pos, len(m5_ts) - 1))
                exit_spread_pips = float(m5_ask[exit_pos] - m5_bid[exit_pos]) / pip

            gross_pips = direction * (exit_px - entry_px) / pip
            spread_rt_pips = (entry_spread_pips + exit_spread_pips) / 2.0 * spread_mult
            carry = carry_pips(pair, direction, entry_ts, exit_ts, markup_mult=markup_mult)
            net_pips = gross_pips - spread_rt_pips + carry
            hours_held = (pd.Timestamp(exit_ts) - pd.Timestamp(entry_ts)).total_seconds() / 3600.0

            trades.append({
                "pair": pair, "arm": arm, "direction": direction,
                "signal_ts": bar["timestamp"], "entry_ts": entry_ts, "entry_px": entry_px,
                "exit_ts": exit_ts, "exit_px": exit_px, "exit_reason": exit_reason,
                "atr_e": atr_e, "gross_pips": gross_pips, "spread_rt_pips": spread_rt_pips,
                "carry_pips": carry, "net_pips": net_pips, "hours_held": hours_held,
                "touches": sig["touches"], "vrel": sig["vrel"],
            })

            # One position per pair (FIFO): block new signal search on every H4 bar up to and
            # including whichever bar the exit fell in (matches the live service, which only
            # resumes signal-search on the bar AFTER the one that recorded the close) — NOT
            # always the nominal time-cap bar, so an early TP/SL exit correctly frees up the
            # remaining H4 bars for new signals instead of being skipped wastefully.
            if hit is not None:
                exit_h4_idx = int(np.searchsorted(h4_ts_arr, exit_ts, side="right")) - 1
            else:
                exit_h4_idx = cap_h4_idx if cap_reached else n_h4 - 1
            in_position_until_idx = max(in_position_until_idx, exit_h4_idx)

        idx += 1

    return trades


def trades_to_frame(trades):
    return pd.DataFrame(trades)


def expectancy(trades, col="net_pips"):
    """Simple mean + standard-error summary, handy for tests / quick sanity checks."""
    if not trades:
        return {"n": 0, "mean": float("nan"), "se": float("nan")}
    vals = np.array([t[col] for t in trades], dtype=float)
    n = len(vals)
    se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": float(vals.mean()), "se": float(se)}
