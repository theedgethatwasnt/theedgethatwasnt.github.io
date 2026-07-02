#!/usr/bin/env python3
"""
Conservative 010 backtest engine.

Ports the stack010_equity.py `kern` verbatim with exactly THREE conservative changes:
  1. Real per-bar spread: entry fills use bid_c/ask_c (not mid open). No SPREAD_FRAC.
  2. Worse-side fills: long buys at ask_c, exits at bid_c; short vice versa.
  3. Stop slippage: fence/PSAR exits slip `slippage_pips` beyond the fence/trigger
     level (against you). TP fills at the limit level exactly (no slip). Flip exits
     fill at bid_c/ask_c bar close with 0 slip (discretionary).

Exit reason codes: 0=fence, 1=psar, 2=tp, 3=flip.

Ambiguities resolved vs stack010:
  - Entry fill: ask_c[i]/bid_c[i] (bar close bid/ask) — NOT stack010's o[i] (mid open).
    This is a deliberate, systematically-conservative divergence: only close bid/ask is
    available; using the close vs the open slightly overstates entry cost, which is the
    conservative direction.
  - Fence level = entry_fill ± fence*pip (from actual fill price, not mid).
    Fence EXIT fills at the fence level ± slip (not at bar close bid/ask).
  - MFE tracks h[i]/l[i] (mid) vs entry_fill — slightly underestimates MFE for PSAR
    arming but is causal and consistent (no bid/ask OHLC in the data).
  - TP level = entry_fill ± tp_pips*pip (from fill, same logic as fence).
    TP EXIT fills at the TP level exactly (limit order, no slip).
  - PSAR trigger = c[i] (mid close) vs sar_b[i], verbatim from stack010.
  - Flip exit = opposing TF1+TF2 novelty at bar i → exit at bid_c[i]/ask_c[i], no slip.
    Confirmed real live 010 behavior; disable with no_flip=True for comparison runs.
  - Exit priority: fence → TP → PSAR → flip (unchanged from stack010 fence→TP→PSAR).
"""
import numpy as np
import numba as nb


# ---------------------------------------------------------------------------
# Pure-Python helper (used in tests; logic inlined inside numba kernel)
# ---------------------------------------------------------------------------

def _net_pips_worse_side(direction, entry_ask, entry_bid, exit_ask, exit_bid, pip, slip):
    """Net pips with worse-side fills + stop slippage (slip applied against you)."""
    if direction == 1:           # long: enter at ask, exit at bid
        entry = entry_ask
        exit_ = exit_bid - slip * pip
        return (exit_ - entry) / pip
    else:                        # short: enter at bid, exit at ask
        entry = entry_bid
        exit_ = exit_ask + slip * pip
        return (entry - exit_) / pip


# ---------------------------------------------------------------------------
# Numba JIT kernel
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _kern(o, h, l, c, bid_c, ask_c, t1l, t1s, t2l, t2s, sar_b,
          pip, tp_pips, use_psar, act, fence, slip, no_flip):
    """
    Conservative 010 kernel (Numba JIT).

    Inputs mirror the stack010_equity.py kern signature, extended with bid_c/ask_c
    and a slip parameter. All trigger logic is verbatim from stack010; only fill
    prices and P&L computation differ.
    """
    n = len(o)
    pos = 0; entry_fill = 0.0; ebar = -1; mfe = 0.0; armed = False

    _ebar = np.empty(n, np.int64)
    _xbar = np.empty(n, np.int64)
    _dir  = np.empty(n, np.int64)
    _epx  = np.empty(n, np.float64)
    _xpx  = np.empty(n, np.float64)
    _pnl  = np.empty(n, np.float64)
    _rsn  = np.empty(n, np.int64)
    nt = 0

    for i in range(1, n):

        # ── ENTRY (verbatim stack010 novelty conditions; fill at worse side) ──
        if pos == 0:
            if t1l[i] == 1 and t2l[i] == 1:
                pos = 1; entry_fill = ask_c[i]; ebar = i; mfe = 0.0; armed = False
                continue
            if t1s[i] == 1 and t2s[i] == 1:
                pos = -1; entry_fill = bid_c[i]; ebar = i; mfe = 0.0; armed = False
                continue

        # ── IN TRADE ──
        if pos != 0:
            # MFE: mid high/low vs actual fill (causal; slightly underestimates for longs)
            fav = (h[i] - entry_fill) / pip if pos == 1 else (entry_fill - l[i]) / pip
            if fav > mfe: mfe = fav
            if use_psar and (not armed) and mfe >= act: armed = True

            ex_fill = 0.0; rsn = -1

            # --- 1. Fence (r=0): worst-side trigger, fill at FENCE LEVEL + slip ---
            # Fence level based on actual entry fill (identical to stack010's logic
            # but using fill price so the 200p measures our true loss, not mid loss).
            # Fill is at fc (not bar-close bid/ask) — the bar may recover after touching
            # the fence, so filling at bar close would understate the realized loss.
            fc = entry_fill - pos * fence * pip
            if pos == 1 and l[i] <= fc:
                ex_fill = fc - slip * pip; rsn = 0
            elif pos == -1 and h[i] >= fc:
                ex_fill = fc + slip * pip; rsn = 0

            # --- 2. TP (r=2): verbatim trigger, fill at TP LEVEL, no slip ---
            # TP is a limit order: fills at the limit level (no slip, no better).
            if rsn < 0 and tp_pips > 0.0:
                tp = entry_fill + pos * tp_pips * pip
                if pos == 1 and h[i] >= tp:
                    ex_fill = tp; rsn = 2
                elif pos == -1 and l[i] <= tp:
                    ex_fill = tp; rsn = 2

            # --- 3. PSAR (r=1): verbatim trigger, fill at bid/ask + slip ---
            if rsn < 0 and use_psar and armed and not np.isnan(sar_b[i]):
                if pos == 1 and c[i] < sar_b[i]:
                    ex_fill = bid_c[i] - slip * pip; rsn = 1
                elif pos == -1 and c[i] > sar_b[i]:
                    ex_fill = ask_c[i] + slip * pip; rsn = 1

            # --- 4. Flip (r=3): opposing novelty exits the trade, no slip ---
            # Default ON (confirmed live 010 behavior). Disable via no_flip=True
            # for stack010-comparison runs that want to isolate other effects.
            if rsn < 0 and not no_flip:
                if pos == 1 and t1s[i] == 1 and t2s[i] == 1:
                    ex_fill = bid_c[i]; rsn = 3
                elif pos == -1 and t1l[i] == 1 and t2l[i] == 1:
                    ex_fill = ask_c[i]; rsn = 3

            if rsn >= 0:
                pnl = ((ex_fill - entry_fill) / pip if pos == 1
                       else (entry_fill - ex_fill) / pip)
                _ebar[nt] = ebar; _xbar[nt] = i; _dir[nt] = pos
                _epx[nt] = entry_fill; _xpx[nt] = ex_fill
                _pnl[nt] = pnl; _rsn[nt] = rsn
                nt += 1; pos = 0

    return (_ebar[:nt], _xbar[:nt], _dir[:nt],
            _epx[:nt], _xpx[:nt], _pnl[:nt], _rsn[:nt])


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_TRADE_DTYPE = np.dtype([
    ('entry_bar',        np.int64),
    ('exit_bar',         np.int64),
    ('dir',              np.int64),
    ('entry_px',         np.float64),
    ('exit_px',          np.float64),
    ('pnl_pips_net',     np.float64),
    ('exit_reason_code', np.int64),
])


def backtest_pair(d: dict, cfg: tuple, slippage_pips: float = 2.0,
                  no_flip: bool = False) -> np.ndarray:
    """Run the conservative 010 kernel on one pair.

    Parameters
    ----------
    d : dict
        Output of data.load_pair_ba().
    cfg : tuple
        (pip, tf1_min, tf2_min, sma_tuple, tp_pips, use_psar, af_start, act, fence)
        Verbatim stack010_equity.py CFG format.  Passed separately so callers can
        sweep parameters without reloading data.
    slippage_pips : float
        Extra slippage applied against the trader on fence/PSAR stop exits.
        Fence and PSAR exits slip beyond the trigger level; TP fills at the
        limit level exactly (no slip); flip exits fill at bid_c/ask_c (no slip).
    no_flip : bool, default False
        When True, disables the flip exit (rsn=3). Default is False (flip ON),
        which matches confirmed live 010 behavior. Set True for stack010-comparison
        runs that want to isolate the effect of other conservatism changes.

    Returns
    -------
    np.ndarray with dtype _TRADE_DTYPE
        Fields: entry_bar, exit_bar, dir, entry_px, exit_px,
                pnl_pips_net, exit_reason_code
        Reason codes: 0=fence, 1=psar, 2=tp, 3=flip.
    """
    pip, _t1m, _t2m, _sma, tp_pips, use_psar, _af, act, fence = cfg

    eb, xb, di, ep, xp, pn, rs = _kern(
        d['m5_o'], d['m5_h'], d['m5_l'], d['m5_c'],
        d['bid_c'], d['ask_c'],
        d['t1l'], d['t1s'], d['t2l'], d['t2s'],
        d['sar'],
        pip, tp_pips, use_psar, act, fence, float(slippage_pips), no_flip,
    )

    out = np.empty(len(eb), dtype=_TRADE_DTYPE)
    out['entry_bar']        = eb
    out['exit_bar']         = xb
    out['dir']              = di
    out['entry_px']         = ep
    out['exit_px']          = xp
    out['pnl_pips_net']     = pn
    out['exit_reason_code'] = rs
    return out
