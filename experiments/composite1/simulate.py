"""simulate.py — Composite 1: per-pair Axis-1 signal detection + both-direction trade
simulation (R6, "one code path"): every arm (composite / axis1-alone / coin-flip /
shuffled-positioning-null) reads from the SAME precomputed signal+trade table, differing
only in WHICH subset of signals is kept and, for two arms, which of the two precomputed
directions is used. The underlying opportunity set (FIFO trade calendar) is fixed and
arm-independent — a COT gate or a coin flip decides whether a fixed opportunity is taken,
never when the next opportunity can occur (see fifo_filter_natural's docstring).

Exits (PREREGISTRATION.md "Composite rule", frozen): TP = 2xATR(14,D1) at entry; stop =
4xATR SL-first (wide, bounded); time cap = 10 D1 bars (HCAP, counted from the signal bar),
exit at close. One position per pair, FIFO.

Entry / cost model (documented interpretation, R9-style)
----------------------------------------------------------
PREREGISTRATION.md: "Entry next D1 open after the signal bar (mid ± half real spread)."
Signal/level construction (R/S, ATR, touch geometry) is done ENTIRELY on mid OHLC (R3: never
mix mid-built charts with bid/ask P&L). The entry/exit FILL price is then built literally as
"mid ± half of that bar's own REAL logged spread" (bid_c/ask_c-derived, never a fixed/default
value — R3b): entry_fill = mid_open +- 0.5*spread_entry*direction (paying/crossing the
spread going in), and — the natural, symmetric completion of the same rule, applied at the
other leg — exit_fill = mid_exit_trigger -+ 0.5*spread_exit*direction (crossing again going
out). TP/SL levels are computed relative to the entry FILL price (matching
multiday_contrarian/harness.py's own `tp = entry_px + direction*TGT_ATR*atr_e` convention,
where entry_px there is likewise the fill price), and the barrier scan itself still compares
against bars' raw MID high/low (R3-clean: the chart geometry stays pure mid throughout, only
the fill/P&L construction touches bid/ask). The net effect: one full round-trip spread is
charged per trade (half at entry + half at exit), using that specific bar's own real spread
at each leg, not a portfolio-level median (contrast with cot_positioning/portfolio.py, which
uses a fixed IS-only per-pair median — this experiment's own pre-registration explicitly
says "real spread", so per-bar actual spread is used instead here).

SL-first same-bar barrier scan (task brief: "stop 4xATR SL-first"): if both a stop and a
target are technically crossable within the same bar, SL is checked first (conservative
worse-outcome assumption — the same posture R2's bull/bear sequencing exists to enforce,
simplified here to an unconditional SL-first check, exactly matching
multiday_contrarian/harness.py's own `_scan_barriers`). Gap-through-stop: if the bar's own
OPEN has already crossed the stop, the fill is that OPEN (realistic slippage, R9).
"""
import numpy as np
import pandas as pd

import _paths  # noqa: F401
from carry_splice import carry_pips_spliced
import d1_data as d1
from swing_axis1 import D1State, HCAP, SL_ATR, TGT_ATR, process_d1_bar


def pair_records(df: pd.DataFrame):
    """df: an is_data.load_pair_is()-style DataFrame (DatetimeIndex, tz-aware UTC,
    open/high/low/close/bid_c/ask_c/volume columns). Returns aligned numpy arrays."""
    ts = df.index.values
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    bid = df["bid_c"].values.astype(float)
    ask = df["ask_c"].values.astype(float)
    return ts, o, h, lo, c, v, bid, ask


def _bar_dict(i, ts, o, h, lo, c, v):
    return {"timestamp": ts[i], "open": o[i], "high": h[i], "low": lo[i], "close": c[i], "volume": v[i]}


def _as_utc_ts(np_ts):
    t = pd.Timestamp(np_ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _trade_for_direction(pair, ts, o, h, lo, c, bid, ask, pip, signal_idx, direction, atr_e,
                          markup_mult=1.0):
    """Full entry->exit simulation for one signal, ONE assumed direction. Returns a trade
    dict or None if the data ends before the cap window can resolve (dropped, never a
    partial/leaking trade)."""
    n = len(ts)
    entry_idx = signal_idx + 1
    if entry_idx >= n:
        return None

    entry_mid = o[entry_idx]
    entry_spread_pips = (ask[entry_idx] - bid[entry_idx]) / pip
    entry_fill = entry_mid + direction * 0.5 * entry_spread_pips * pip

    tp = entry_fill + direction * TGT_ATR * atr_e
    sl = entry_fill - direction * SL_ATR * atr_e

    cap_idx = signal_idx + HCAP
    scan_end = min(cap_idx, n)

    exit_idx = None
    exit_mid = None
    exit_reason = None
    for j in range(entry_idx, scan_end):
        oj, hj, lj = o[j], h[j], lo[j]
        if direction > 0:
            if oj <= sl:
                exit_idx, exit_mid, exit_reason = j, oj, "sl_gap"
                break
            if lj <= sl:
                exit_idx, exit_mid, exit_reason = j, sl, "sl"
                break
            if hj >= tp:
                exit_idx, exit_mid, exit_reason = j, tp, "tp"
                break
        else:
            if oj >= sl:
                exit_idx, exit_mid, exit_reason = j, oj, "sl_gap"
                break
            if hj >= sl:
                exit_idx, exit_mid, exit_reason = j, sl, "sl"
                break
            if lj <= tp:
                exit_idx, exit_mid, exit_reason = j, tp, "tp"
                break

    if exit_idx is None:
        if cap_idx < n:
            exit_idx, exit_mid, exit_reason = cap_idx, c[cap_idx], "timecap"
        else:
            return None  # cap window runs past loaded data — dropped, not faked

    exit_spread_pips = (ask[exit_idx] - bid[exit_idx]) / pip
    exit_fill = exit_mid - direction * 0.5 * exit_spread_pips * pip

    gross_pips = direction * (exit_fill - entry_fill) / pip
    entry_ts = _as_utc_ts(ts[entry_idx])
    exit_ts = _as_utc_ts(ts[exit_idx])
    carry = carry_pips_spliced(pair, direction, entry_ts, exit_ts, markup_mult=markup_mult)
    net_pips = gross_pips + carry

    return {
        "pair": pair, "direction": int(direction),
        "signal_idx": int(signal_idx), "entry_idx": int(entry_idx), "exit_idx": int(exit_idx),
        "entry_ts": entry_ts, "exit_ts": exit_ts,
        "entry_fill": float(entry_fill), "exit_fill": float(exit_fill),
        "atr_e": float(atr_e), "gross_pips": float(gross_pips),
        "spread_rt_pips": float((entry_spread_pips + exit_spread_pips) / 2.0),
        "carry_pips": float(carry), "net_pips": float(net_pips), "exit_reason": exit_reason,
    }


def detect_signals_and_trades(pair, df, markup_mult=1.0):
    """Runs the Axis-1 detector bar-by-bar (R1 closed-bars-only) over `df` and, for every
    fired first-touch signal, precomputes BOTH directions' full trade outcome (the natural
    fade direction and its opposite). Returns a list of dicts:
      {pair, signal_idx, signal_ts, natural_direction, touches, vrel, atr_e,
       trade_natural, trade_opposite}
    A signal is DROPPED (excluded from the returned list) if either direction's trade
    cannot resolve within the loaded data (cap window runs past the buffer)."""
    ts, o, h, lo, c, v, bid, ask = pair_records(df)
    pip = d1.pip_size(pair)
    state = D1State(pair=pair)
    n = len(ts)
    out = []
    for i in range(n):
        bar = _bar_dict(i, ts, o, h, lo, c, v)
        sig = process_d1_bar(state, bar)
        if sig is None:
            continue
        nat_dir = sig["direction"]
        t_nat = _trade_for_direction(pair, ts, o, h, lo, c, bid, ask, pip, i, nat_dir, sig["atr"], markup_mult)
        t_opp = _trade_for_direction(pair, ts, o, h, lo, c, bid, ask, pip, i, -nat_dir, sig["atr"], markup_mult)
        if t_nat is None or t_opp is None:
            continue
        out.append({
            "pair": pair, "signal_idx": i, "signal_ts": _as_utc_ts(ts[i]),
            "natural_direction": nat_dir, "touches": sig["touches"], "vrel": sig["vrel"],
            "atr_e": sig["atr"],
            "trade_natural": t_nat, "trade_opposite": t_opp,
        })
    return out


def fifo_filter_natural(pair_signals):
    """One-position-per-pair FIFO, applied ONCE using the NATURAL fade direction's exit —
    this defines the pair's fixed Axis-1 trade calendar. Every arm (composite gate /
    coin-flip / shuffled-positioning permutation) is a SUBSET or a per-trade direction
    relabeling of exactly this list; none of them may alter the underlying opportunity
    timing (a signal that fires while a position is open is unavailable to ANY arm, gated or
    not — the entry engine itself is single-position, regardless of which overlay decides to
    act on a given fixed opportunity). `pair_signals` must already be in ascending
    signal_idx order (guaranteed by detect_signals_and_trades's single forward pass)."""
    kept = []
    in_position_until = -1
    for rec in pair_signals:
        if rec["signal_idx"] <= in_position_until:
            continue
        kept.append(rec)
        in_position_until = rec["trade_natural"]["exit_idx"]
    return kept
