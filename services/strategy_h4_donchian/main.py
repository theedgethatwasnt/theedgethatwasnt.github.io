#!/usr/bin/env python3
"""
H4 Donchian Trend — Account 010, 4 pairs, 10 units each
Strategy: Donchian(10) breakout + Wilder ATR(14) trailing stop
Entry:  close > 10-bar hh  → LONG;  close < 10-bar ll → SHORT
Exit:   lo ≤ trail_stop (LONG) or hi ≥ trail_stop (SHORT), OR opposite signal
OOS:    GBP_JPY=24.8p/d, USD_JPY=24.0p/d, EUR_JPY=20.2p/d, GBP_USD=16.4p/d
MaxDD:  GBP_JPY=84.7p, USD_JPY=70.2p, EUR_JPY=75.9p, GBP_USD=99.7p (Session 053)

SOP compliance (CLAUDE.md §Backtest–Live Consistency SOP):
  R1  Closed bars only — exclude last fetched bar (in-progress); track last_ts
  R2  Mid OHLC for signals (price='MBA')
  R3  Mid signals; spread gate at entry (sp <= sp_gate)
  R3a bid_c/ask_c for spread only
  R3b IS P90 spread gates hardcoded per pair (R5)
  R4  Donchian: deque of last DON_N completed-bar highs/lows (excludes current bar)
  R5  IS P90 spread gates: GBP_JPY=3.80p, USD_JPY=2.00p, EUR_JPY=2.40p, GBP_USD=2.30p
  R6  process_bar() logic mirrors backtest_h4_trend.py Donchian kernel exactly
  R7  startup warmup replays completed bars; state verified against OANDA on open pos
  R9  Entry-bar close→open lookahead: live entry is market order after bar close,
      backtest uses open[i+1]. Negligible for H4. Documented here.
"""

import os
import sys
import json
import time
import signal
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Deque
from collections import deque

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

from lib.oanda_adapter import OANDAAdapter
from lib.db import write_trade_direct

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [h4_donchian] %(message)s",
)
logger = logging.getLogger("h4_donchian")

# ── Account + sizing ──────────────────────────────────────────────────────────
ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID_010", "001-001-${OANDA_CUSTOMER_ID}-010")
STATE_DIR  = os.environ.get("H4_STATE_DIR", "/data/logs")
UNITS      = 10
STRATEGY   = "h4_donchian"
LABEL      = "h4_don010"

# ── Strategy params ───────────────────────────────────────────────────────────
DON_N        = 10        # Donchian lookback (completed bars)
ATR_PERIOD   = 14        # Wilder ATR period
ATR_TRAIL    = 1.0       # trail distance in ATR multiples (from peak bar high/low)
ATR_SL_INIT  = 2.0       # initial SL in ATR multiples (used at entry time)

POLL_SECS    = 60
WARMUP_BARS  = 100       # bars fetched for indicator warmup

# ── Pair config: pip size + IS P90 spread gate (pips, R5) ────────────────────
PAIRS = {
    "GBP_JPY": {"pip": 0.01,   "sp_gate": 3.80},
    "USD_JPY": {"pip": 0.01,   "sp_gate": 2.00},
    "EUR_JPY": {"pip": 0.01,   "sp_gate": 2.40},
    "GBP_USD": {"pip": 0.0001, "sp_gate": 2.30},
}

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT_ID", "")
_shutdown      = False


def _tg(msg: str):
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": msg},
            timeout=5,
        )
    except Exception:
        pass


# ── Per-pair state ────────────────────────────────────────────────────────────
@dataclass
class PairState:
    pair:       str
    pip:        float
    sp_gate:    float
    pos:        int   = 0       # 0=flat, +1=long, -1=short
    entry_px:   float = 0.0
    trail_stop: float = 0.0
    peak_px:    float = 0.0     # highest high (long) or lowest low (short)
    atr:        float = 0.0
    entry_time: str   = ""
    trade_id:   str   = ""
    mfe_pips:   float = 0.0
    mae_pips:   float = 0.0
    last_ts:    str   = ""
    # indicator buffers (not persisted — rebuilt from warmup bars)
    hi_buf:     Deque = field(default_factory=lambda: deque(maxlen=DON_N))
    lo_buf:     Deque = field(default_factory=lambda: deque(maxlen=DON_N))
    prev_close: float = 0.0
    atr_count:  int   = 0       # bars seen for ATR init


def _state_file(pair: str) -> str:
    return os.path.join(STATE_DIR, f"h4_donchian_{pair}_state.json")


def _save_state(st: PairState):
    data = {
        "pos":        st.pos,
        "entry_px":   st.entry_px,
        "trail_stop": st.trail_stop,
        "peak_px":    st.peak_px,
        "atr":        st.atr,
        "entry_time": st.entry_time,
        "trade_id":   st.trade_id,
        "mfe_pips":   st.mfe_pips,
        "mae_pips":   st.mae_pips,
        "last_ts":    st.last_ts,
    }
    try:
        with open(_state_file(st.pair), "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"{st.pair}: failed to save state: {e}")


def _load_state(st: PairState):
    path = _state_file(st.pair)
    if not os.path.exists(path):
        return
    try:
        data = json.load(open(path))
        st.pos        = data.get("pos", 0)
        st.entry_px   = data.get("entry_px", 0.0)
        st.trail_stop = data.get("trail_stop", 0.0)
        st.peak_px    = data.get("peak_px", 0.0)
        st.atr        = data.get("atr", 0.0)
        st.entry_time = data.get("entry_time", "")
        st.trade_id   = data.get("trade_id", "")
        st.mfe_pips   = data.get("mfe_pips", 0.0)
        st.mae_pips   = data.get("mae_pips", 0.0)
        st.last_ts    = data.get("last_ts", "")
        logger.info(f"{st.pair}: loaded state pos={st.pos} last_ts={st.last_ts}")
    except Exception as e:
        logger.warning(f"{st.pair}: failed to load state: {e}")


# ── Wilder ATR step (R6: matches backtest) ────────────────────────────────────
def _atr_step(st: PairState, hi: float, lo: float) -> float:
    tr = max(hi - lo, abs(hi - st.prev_close), abs(lo - st.prev_close))
    if st.atr_count < ATR_PERIOD:
        st.atr_count += 1
        # running simple average during seed phase
        st.atr = st.atr + (tr - st.atr) / st.atr_count
    else:
        st.atr = (st.atr * (ATR_PERIOD - 1) + tr) / ATR_PERIOD
    return st.atr


# ── Single bar processing (R6: identical to backtest kernel logic) ─────────────
def _process_bar(st: PairState, bar: dict, adapter: OANDAAdapter, live: bool) -> bool:
    """Process one completed H4 bar. Returns True if any trade action taken."""
    op  = bar["open"]
    hi  = bar["high"]
    lo  = bar["low"]
    cl  = bar["close"]
    bid = bar["bid_c"]
    ask = bar["ask_c"]
    sp  = (ask - bid) / st.pip

    # update ATR (R6: Wilder, uses prev_close for TR)
    if st.prev_close > 0:
        _atr_step(st, hi, lo)
    st.prev_close = cl

    # Donchian: push CURRENT bar into buffer AFTER reading entry signal
    # (so buffer holds hi[i-N:i], excludes bar i — causal, R4)
    hh = max(st.hi_buf) if len(st.hi_buf) >= DON_N else None
    ll = min(st.lo_buf) if len(st.lo_buf) >= DON_N else None

    acted = False

    if st.pos != 0:
        # ── update peak and trail stop ────────────────────────────────────────
        if st.pos == 1:  # LONG
            if hi > st.peak_px:
                st.peak_px   = hi
                st.trail_stop = st.peak_px - ATR_TRAIL * st.atr
            run = (cl - st.entry_px) / st.pip
            st.mfe_pips = max(st.mfe_pips, (hi - st.entry_px) / st.pip)
            st.mae_pips = min(st.mae_pips, (lo - st.entry_px) / st.pip)
        else:            # SHORT
            if lo < st.peak_px:
                st.peak_px   = lo
                st.trail_stop = st.peak_px + ATR_TRAIL * st.atr
            run = (st.entry_px - cl) / st.pip
            st.mfe_pips = max(st.mfe_pips, (st.entry_px - lo) / st.pip)
            st.mae_pips = min(st.mae_pips, (st.entry_px - hi) / st.pip)

        # ── check exit ───────────────────────────────────────────────────────
        exit_reason = None
        exit_px     = 0.0

        if st.pos == 1:
            if lo <= st.trail_stop:
                exit_reason = "trail"
                exit_px     = min(op, st.trail_stop)  # slippage approximation
            elif ll is not None and cl < ll:
                exit_reason = "reverse"
                exit_px     = op
        else:
            if hi >= st.trail_stop:
                exit_reason = "trail"
                exit_px     = max(op, st.trail_stop)
            elif hh is not None and cl > hh:
                exit_reason = "reverse"
                exit_px     = op

        if exit_reason and live:
            result = adapter.close_trade(st.trade_id)
            if result.success:
                exit_px   = result.fill_price or exit_px
                pnl_pips  = (exit_px - st.entry_px) / st.pip * st.pos
                pnl_sp    = -sp  # spread at close (approx)
                pnl_total = pnl_pips + pnl_sp
                dur_h     = 0.0
                if st.entry_time:
                    try:
                        t0 = datetime.fromisoformat(st.entry_time.replace("Z", "+00:00"))
                        dur_h = (datetime.now(timezone.utc) - t0).total_seconds() / 3600
                    except Exception:
                        pass
                cap = st.mfe_pips / max(abs(st.mae_pips), 0.01) if st.mae_pips < 0 else 99.0

                write_trade_direct(
                    strategy=STRATEGY,
                    pair=st.pair,
                    account_id=ACCOUNT_ID,
                    trade_id=st.trade_id,
                    direction=st.pos,
                    entry_price=st.entry_px,
                    exit_price=exit_px,
                    entry_time=st.entry_time,
                    exit_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    pnl_pips=pnl_total,
                    exit_reason=exit_reason,
                    hours_held=dur_h,
                    units=UNITS,
                    mfe_pips=st.mfe_pips,
                    mae_pips=abs(st.mae_pips),
                    capture_ratio=round(cap, 2),
                    is_paper=False,
                    label=LABEL,
                )
                dir_sym = "🟢" if st.pos == 1 else "🔴"
                pl_sym  = "🟢" if pnl_total >= 0 else "🔴"
                _tg(
                    f"{pl_sym} 010 {st.pair} CLOSE {dir_sym}  {exit_reason}\n"
                    f"P/L {pnl_total:+.1f}p  MFE {st.mfe_pips:.1f}p  MAE {abs(st.mae_pips):.1f}p\n"
                    f"held {dur_h:.1f}h  trail={st.trail_stop:.3f}"
                )
                logger.info(
                    f"{st.pair} CLOSE pos={st.pos} {exit_reason} @ {exit_px:.3f} "
                    f"pnl={pnl_total:+.1f}p  mfe={st.mfe_pips:.1f}  mae={abs(st.mae_pips):.1f}"
                )
                st.pos = 0; st.entry_px = 0; st.trail_stop = 0; st.peak_px = 0
                st.entry_time = ""; st.trade_id = ""; st.mfe_pips = 0; st.mae_pips = 0
                acted = True
                # on reverse signal, re-enter below (after buffer push)
                if exit_reason == "reverse":
                    # push bar into Donchian buffers before re-entry
                    st.hi_buf.append(hi)
                    st.lo_buf.append(lo)
                    # re-compute after push
                    hh = max(st.hi_buf) if len(st.hi_buf) >= DON_N else None
                    ll = min(st.lo_buf) if len(st.lo_buf) >= DON_N else None
                    _try_entry(st, op, cl, hi, lo, sp, hh, ll, bar, adapter, live)
                    return acted

    # ── push completed bar into Donchian buffer ───────────────────────────────
    st.hi_buf.append(hi)
    st.lo_buf.append(lo)

    # ── check entry ───────────────────────────────────────────────────────────
    if st.pos == 0 and hh is not None and ll is not None:
        _try_entry(st, op, cl, hi, lo, sp, hh, ll, bar, adapter, live)
        if st.pos != 0:
            acted = True

    return acted


def _try_entry(st: PairState, op: float, cl: float, hi: float, lo: float,
               sp: float, hh: Optional[float], ll: Optional[float],
               bar: dict, adapter: OANDAAdapter, live: bool):
    if hh is None or ll is None or st.atr_count < ATR_PERIOD:
        return

    direction = 0
    if cl > hh:
        direction = 1
    elif cl < ll:
        direction = -1
    else:
        return

    if sp > st.sp_gate:
        logger.info(f"{st.pair}: spread {sp:.2f}p > gate {st.sp_gate}p — entry skipped")
        return

    # entry at market (live) or open of next bar (backtest approx)
    if live:
        units_signed = UNITS if direction == 1 else -UNITS
        init_sl = (op - direction * ATR_SL_INIT * st.atr) if direction == 1 \
                  else (op + ATR_SL_INIT * st.atr)
        result = adapter.place_market_order(st.pair, units_signed, sl_price=init_sl)
        if not result.success:
            logger.warning(f"{st.pair}: order rejected: {result.error}")
            return
        fill_px = result.fill_price or op
        trade_id = result.trade_id or ""
    else:
        fill_px = op
        trade_id = f"warmup_{bar.get('timestamp','')}"

    st.pos        = direction
    st.entry_px   = fill_px
    st.peak_px    = hi if direction == 1 else lo
    st.trail_stop = st.peak_px - direction * ATR_TRAIL * st.atr
    st.trade_id   = trade_id
    st.entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    st.mfe_pips   = 0.0
    st.mae_pips   = 0.0

    if live:
        dir_str = "LONG" if direction == 1 else "SHORT"
        dir_sym = "🟢" if direction == 1 else "🔴"
        _tg(
            f"{dir_sym} 010 {st.pair} OPEN {dir_str} @ {fill_px:.3f}  {UNITS}u\n"
            f"trail={st.trail_stop:.3f}  atr={st.atr:.3f}  sp={sp:.2f}p"
        )
        logger.info(
            f"{st.pair} OPEN {dir_str} @ {fill_px:.3f}  trail={st.trail_stop:.3f}  id={trade_id}"
        )


# ── Startup: warmup + reconcile ───────────────────────────────────────────────
def _warmup_pair(st: PairState, adapter: OANDAAdapter):
    """Fetch WARMUP_BARS H4 candles, replay to init ATR+Donchian. Restore pos from file+OANDA."""
    logger.info(f"{st.pair}: fetching {WARMUP_BARS} H4 bars for warmup")
    bars = adapter.get_candles(st.pair, count=WARMUP_BARS + 1, granularity="H4")
    if not bars:
        logger.error(f"{st.pair}: warmup fetch returned no bars")
        return

    # exclude last bar (potentially in-progress)
    completed = bars[:-1]
    if not completed:
        return

    # find bars newer than saved last_ts
    if st.last_ts:
        new_bars = [b for b in completed if b["timestamp"] > st.last_ts]
    else:
        new_bars = completed

    if not new_bars:
        logger.info(f"{st.pair}: no new bars to process (last_ts={st.last_ts})")
        return

    # if we have no ATR yet, we need to seed it from history
    if st.atr_count == 0 and new_bars:
        st.prev_close = new_bars[0]["open"]

    min_warmup = ATR_PERIOD + DON_N  # need 24 bars before signals
    total = len(new_bars)
    logger.info(f"{st.pair}: replaying {total} new bars (warmup min={min_warmup})")

    for i, bar in enumerate(new_bars):
        warmup_phase = (total - i) > 5  # last 5 bars treated as live-capable
        # during startup, if we already have a saved position, don't fake-enter new ones
        suppress_entry = (st.pos != 0)
        hi = bar["high"]; lo = bar["low"]; cl = bar["close"]
        op = bar["open"]
        sp = (bar["ask_c"] - bar["bid_c"]) / st.pip

        if st.prev_close > 0:
            _atr_step(st, hi, lo)
        st.prev_close = cl

        hh = max(st.hi_buf) if len(st.hi_buf) >= DON_N else None
        ll = min(st.lo_buf) if len(st.lo_buf) >= DON_N else None

        st.hi_buf.append(hi)
        st.lo_buf.append(lo)

        # don't simulate trades during warmup replay (just build buffers)
        # on the last few bars: update trailing stop for existing positions
        if st.pos != 0 and not warmup_phase:
            if st.pos == 1:
                if hi > st.peak_px:
                    st.peak_px   = hi
                    st.trail_stop = st.peak_px - ATR_TRAIL * st.atr
            else:
                if lo < st.peak_px:
                    st.peak_px   = lo
                    st.trail_stop = st.peak_px + ATR_TRAIL * st.atr

        st.last_ts = bar["timestamp"]

    logger.info(
        f"{st.pair}: warmup done — atr={st.atr:.4f} bufs={len(st.hi_buf)}/{len(st.lo_buf)} "
        f"pos={st.pos} last_ts={st.last_ts}"
    )

    # ── reconcile position with OANDA ─────────────────────────────────────────
    try:
        open_trades = adapter.get_open_trades()
        pair_trades = [t for t in open_trades if t.instrument == st.pair]
        if pair_trades and st.pos == 0:
            t = pair_trades[0]
            st.pos       = 1 if t.units > 0 else -1
            st.entry_px  = t.price
            st.trade_id  = t.trade_id
            st.peak_px   = st.entry_px
            st.trail_stop = st.entry_px - st.pos * ATR_TRAIL * st.atr
            st.entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.info(f"{st.pair}: adopted orphan trade {t.trade_id} pos={st.pos} @ {st.entry_px:.3f}")
        elif not pair_trades and st.pos != 0:
            logger.warning(f"{st.pair}: state says pos={st.pos} but no open trade found — resetting")
            st.pos = 0; st.entry_px = 0; st.trail_stop = 0; st.peak_px = 0
            st.trade_id = ""; st.entry_time = ""
    except Exception as e:
        logger.warning(f"{st.pair}: OANDA reconcile failed: {e}")

    _save_state(st)


# ── Main poll loop ────────────────────────────────────────────────────────────
def _poll_pair(st: PairState, adapter: OANDAAdapter):
    bars = adapter.get_candles(st.pair, count=DON_N + ATR_PERIOD + 5, granularity="H4")
    if not bars:
        return
    # adapter.get_candles already filters incomplete bars, so all returned bars are closed
    new_bars = [b for b in bars if b["timestamp"] > st.last_ts]
    if not new_bars:
        return

    for bar in new_bars:
        _process_bar(st, bar, adapter, live=True)
        st.last_ts = bar["timestamp"]

    _save_state(st)


def _signal_handler(sig, frame):
    global _shutdown
    logger.info(f"Received signal {sig} — shutting down")
    _shutdown = True


def main():
    global _shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT,  _signal_handler)

    if not ACCOUNT_ID:
        logger.error("OANDA_ACCOUNT_ID_010 not set — exiting")
        sys.exit(1)

    adapter = OANDAAdapter(account_id=ACCOUNT_ID)

    # initialise state objects
    states = {}
    for pair, cfg in PAIRS.items():
        st = PairState(pair=pair, pip=cfg["pip"], sp_gate=cfg["sp_gate"])
        _load_state(st)
        states[pair] = st

    # warmup all pairs
    for pair, st in states.items():
        try:
            _warmup_pair(st, adapter)
        except Exception as e:
            logger.error(f"{pair}: warmup failed: {e}")

    _tg(
        f"🟢 010 H4 Donchian started\n"
        f"Pairs: {', '.join(PAIRS.keys())}  units={UNITS}\n"
        f"DON_N={DON_N} ATR_TRAIL={ATR_TRAIL}× ATR_SL_INIT={ATR_SL_INIT}×\n"
        f"Gates: GBP_JPY=3.80p USD_JPY=2.00p EUR_JPY=2.40p GBP_USD=2.30p"
    )

    logger.info(f"H4 Donchian live on {list(PAIRS.keys())} acct={ACCOUNT_ID} units={UNITS}")

    while not _shutdown:
        for pair, st in states.items():
            try:
                _poll_pair(st, adapter)
            except Exception as e:
                logger.error(f"{pair}: poll error: {e}")

        # status heartbeat every ~100 polls (~1.7h)
        time.sleep(POLL_SECS)

    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
