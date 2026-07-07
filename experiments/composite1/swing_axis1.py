"""swing_axis1.py — Composite 1: Axis-1 D1 swing-extreme fade detector.

PREREGISTRATION.md "Axis-1 signal (structure-transfer, declared ... parameters translated
from the deployed first-touch config to the D1 grid, scale-free; NOT fitted to any data)":

  Fresh rolling swing extreme: highest high / lowest low of the prior L=25 completed D1
  bars. Touch tolerance EPS = 0.25 x ATR(14,D1) at the touch. First touch only. Volume
  gate: touch-bar tick-volume < mean of the prior VW=20 D1 bars (low-volume only).
  Direction: fade the extreme.

Structurally this is the same family as multiday_contrarian/harness.py's
`process_h4_bar` (H4-grid version, REFUTED per PREREGISTRATION.md's note — that program's
own A5 gates 3-6 all failed, see multiday_contrarian/PREREGISTRATION.md). This module is a
FRESH, D1-native implementation (not a verbatim port — the D1 grid is explicitly declared
"untested"), differing from the H4 version in two declared, scale-free ways:
  1. EPS is ATR-relative (0.25xATR14) instead of a fixed pip constant (12 pips on H4) —
     "structure-transfer" / scale-free translation across timeframes, not a re-fit.
  2. The volume gate is a strict "< mean" (vrel < 1.0), not an IS-fitted ratio threshold
     (VREL_MAX=1.16 on the H4 version was the live service's IS-median — a fitted number
     this program deliberately avoids).
R1 (closed-bars-only): every bar fed to `process_d1_bar` must already be a closed D1 bar —
trivially satisfied in an offline backtest over historical parquet rows (all closed), and
the live-port equivalent (out of scope here) would only ever call this once a new D1 bar
has actually closed, exactly like harness.py's H4 analog.
"""
from dataclasses import dataclass, field

import numpy as np

L = 25
VW = 20
ATR_N = 14
EPS_ATR_FRAC = 0.25
TOUCH_MAX = 1
TGT_ATR = 2.0
SL_ATR = 4.0
HCAP = 10  # D1 bars, counted from the SIGNAL bar (matches harness.py's HCAP convention)
BUF = max(L, VW, ATR_N) + 30


@dataclass
class D1State:
    pair: str
    bars: list = field(default_factory=list)


def _atr(bars, n: int = ATR_N) -> float:
    """Wilder ATR over the buffer (mid OHLC) — same formula as
    multiday_contrarian/harness.py's `_atr` (a generic, universal indicator, not COT/Axis-3
    code, so no verbatim-reuse requirement applies; re-derived here for this module's own
    bar-dict schema)."""
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


def process_d1_bar(state: D1State, bar: dict):
    """Appends `bar` (dict: timestamp/open/high/low/close/volume, MID prices) to
    state.bars (trimmed to BUF) and returns a signal dict
    {direction, atr, r, s, touches, vrel} if a first-touch low-volume fade fires on THIS
    bar, else None. Position management (FIFO, one-per-pair) is the caller's
    responsibility (R6: this function only ever looks for entries)."""
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

    atr = _atr(state.bars)
    if atr <= 0:
        return None
    eps = EPS_ATR_FRAC * atr

    volwin = state.bars[-(VW + 1):-1]
    vmean = float(np.mean([b["volume"] for b in volwin])) if volwin else 0.0
    if vmean <= 0:
        return None
    vrel = cur["volume"] / vmean
    if vrel >= 1.0:  # strict "touch-bar volume < mean of prior VW bars" (low-volume only)
        return None

    up = cur["high"] >= R - eps and prev["high"] < R - eps and cur["close"] <= R
    dn = cur["low"] <= S + eps and prev["low"] > S + eps and cur["close"] >= S

    if up:
        touches = sum(1 for b in win if b["high"] >= R - eps)
        if touches > TOUCH_MAX:
            return None
        direction = -1  # fade the swing HIGH: short the pair
    elif dn:
        touches = sum(1 for b in win if b["low"] <= S + eps)
        if touches > TOUCH_MAX:
            return None
        direction = +1  # fade the swing LOW: long the pair
    else:
        return None

    return {"direction": direction, "atr": atr, "r": R, "s": S, "touches": touches, "vrel": vrel}
