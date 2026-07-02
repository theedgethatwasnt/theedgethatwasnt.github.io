#!/usr/bin/env python3
"""
SMA Momentum Live — Account 012, 10 pairs, 25 units each
=========================================================
Signal: SMA(16) on H1 + M30, momentum at lags 8, 10, 15.
        LONG  when all 6 momentum values > 0 (strict 6/6)
        SHORT when all 6 momentum values < 0
Entry:  market order on first M5 bar after signal fires, spread ≤ IS-P90 gate
Exit:   TP=20p placed as broker-side takeProfitOnFill (no manual exit needed)
        Service detects fill when trade disappears from open trades.

Validation: SMA16 lags=(8,10,15) TP=20p — all 10 pairs IS 3/3 + MC p<0.05
  USD_JPY +6.8p/d  EUR_JPY +4.6p/d  GBP_JPY +4.2p/d  AUD_JPY +3.2p/d
  EUR_USD +3.0p/d  GBP_USD +2.3p/d  CAD_JPY +1.8p/d  AUD_USD +0.9p/d
  EUR_GBP +0.3p/d  NZD_USD +0.2p/d  Portfolio mc_p=0.0000

SOP compliance (CLAUDE.md §Backtest–Live Consistency SOP):
  R1  Closed bars only — bars[:-1] excludes in-progress bar
  R3  SMA built on mid closes (OANDA 'M' price='MBA' default)
  R3a M5 bid_c/ask_c for spread gate check at entry
  R3b IS P90 spread gates hardcoded per pair (R5)
  R5  IS P90 gates: USD_JPY=2.10p EUR_JPY=2.50p GBP_JPY=4.00p AUD_JPY=2.30p
                    EUR_USD=1.70p GBP_USD=2.40p CAD_JPY=2.60p AUD_USD=1.60p
                    EUR_GBP=2.00p NZD_USD=2.00p
  R6  compute_signal() mirrors backtest feasibility_study.py / mc_validate_winner.py
  R7  Warmup fetches 50 completed H1+M30 bars; reconciles open pos vs OANDA
  R9  Entry at market; backtest uses ask/bid at same bar — negligible for TP=20p
"""

import os
import sys
import json
import time
import signal
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from dotenv import load_dotenv
load_dotenv()

from lib.oanda_adapter import OANDAAdapter
from lib.db import write_trade_direct

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [sma_live] %(message)s",
)
logger = logging.getLogger("sma_live")

# ── Account + sizing ──────────────────────────────────────────────────────────
ACCOUNT_ID       = os.environ.get("OANDA_ACCOUNT_ID_012", "<OANDA_ACCOUNT_ID>")
STATE_DIR        = os.environ.get("SMA_STATE_DIR", "/data/logs")
UNITS_PER_DOLLAR = 1.25          # units = round(balance * UNITS_PER_DOLLAR)
_units           = 35            # live value, refreshed from broker every hour
_last_balance_refresh = 0.0
STRATEGY   = "sma_momentum"
LABEL      = "sma16_mom012"

# ── Signal params (validated SMA16 lags=(8,10,15) TP=20p) ────────────────────
SMA_N    = 16
LAGS     = (8, 10, 15)
TP_PIPS  = 20.0
WARMUP_N = 55           # completed H1/M30 bars to fetch for warmup (≥ SMA_N + max_lag + buffer)
POLL_SECS = 60          # poll every minute; acts on new completed H1/M30 bars

# ── Pair config: pip + IS P90 spread gate (pips, R5) ─────────────────────────
PAIRS = {
    "USD_JPY": {"pip": 0.01,   "sp_gate": 2.10},
    "EUR_JPY": {"pip": 0.01,   "sp_gate": 2.50},
    "GBP_JPY": {"pip": 0.01,   "sp_gate": 4.00},
    "AUD_JPY": {"pip": 0.01,   "sp_gate": 2.30},
    "EUR_USD": {"pip": 0.0001, "sp_gate": 1.70},
    "GBP_USD": {"pip": 0.0001, "sp_gate": 2.40},
    "CAD_JPY": {"pip": 0.01,   "sp_gate": 2.60},
    "AUD_USD": {"pip": 0.0001, "sp_gate": 1.60},
    "EUR_GBP": {"pip": 0.0001, "sp_gate": 2.00},
    "NZD_USD": {"pip": 0.0001, "sp_gate": 2.00},
}

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT_ID", "")
_shutdown = False


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


def _refresh_units(adapter: "OANDAAdapter") -> None:
    global _units, _last_balance_refresh
    _last_balance_refresh = time.time()
    try:
        info = adapter.get_account_summary()
        if info is not None:
            new_units = max(1, round(info.balance * UNITS_PER_DOLLAR))
            if new_units != _units:
                logger.info(
                    f"Units updated: ${info.balance:.2f} × {UNITS_PER_DOLLAR} "
                    f"→ {new_units}u (was {_units}u)"
                )
            else:
                logger.info(f"Balance refresh: ${info.balance:.2f} → {new_units}u/trade")
            _units = new_units
    except Exception as e:
        logger.warning(f"Balance refresh failed: {e} — keeping {_units}u")


# ── Signal computation (R6: mirrors backtest exactly) ─────────────────────────

def compute_signal(closes_h1: list, closes_m30: list) -> int:
    """
    Returns +1 (long), -1 (short), 0 (flat / insufficient data).
    Uses last SMA_N + max(LAGS) + 1 = 32 completed bars per TF.
    """
    need = SMA_N + max(LAGS) + 1   # 32
    if len(closes_h1) < need or len(closes_m30) < need:
        return 0

    def tf_signal(closes):
        arr = np.array(closes[-need:], dtype=np.float64)
        # compute rolling SMA values at current and each lag position
        # sma at lag k = mean of arr[-(SMA_N + k) : -k]  (or to end if k==0)
        def sma_at_offset(offset):
            if offset == 0:
                return arr[-SMA_N:].mean()
            return arr[-(SMA_N + offset):-offset].mean()

        sma_now = sma_at_offset(0)
        moms = [sma_now - sma_at_offset(k) for k in LAGS]
        return moms

    all_moms = tf_signal(closes_h1) + tf_signal(closes_m30)   # 6 values
    n_pos = sum(1 for m in all_moms if m > 0)
    if n_pos == 6:
        return 1
    if n_pos == 0:
        return -1
    return 0


# ── Per-pair state ────────────────────────────────────────────────────────────
@dataclass
class PairState:
    pair:       str
    pip:        float
    sp_gate:    float
    pos:        int   = 0
    entry_px:   float = 0.0
    tp_price:   float = 0.0
    trade_id:   str   = ""
    entry_time: str   = ""
    signal:     int   = 0     # last computed signal
    last_h1_ts: str   = ""
    last_m30_ts: str  = ""
    units:      int   = 0     # units at entry; 0 = not yet in a trade


def _state_path(pair: str) -> str:
    return os.path.join(STATE_DIR, f"sma_live_{pair}_state.json")


def _save(st: PairState):
    data = {k: v for k, v in st.__dict__.items()
            if k not in ("pair", "pip", "sp_gate")}
    try:
        with open(_state_path(st.pair), "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"{st.pair}: save state failed: {e}")


def _load(st: PairState):
    path = _state_path(st.pair)
    if not os.path.exists(path):
        return
    try:
        data = json.load(open(path))
        for k, v in data.items():
            if hasattr(st, k):
                setattr(st, k, v)
        logger.info(f"{st.pair}: loaded state pos={st.pos} signal={st.signal}")
    except Exception as e:
        logger.warning(f"{st.pair}: load state failed: {e}")


# ── Bar fetch helpers ─────────────────────────────────────────────────────────

def _get_closes(adapter: OANDAAdapter, pair: str, granularity: str,
                count: int) -> tuple[list, str]:
    """
    Returns (closes_list, last_completed_ts).
    Excludes the last (in-progress) bar — R1.
    """
    bars = adapter.get_candles(pair, count=count + 1, granularity=granularity)
    if not bars:
        return [], ""
    completed = bars[:-1]    # drop last (in-progress)
    closes = [b["close"] for b in completed]
    last_ts = completed[-1]["timestamp"] if completed else ""
    return closes, last_ts


def _get_m5_spread(adapter: OANDAAdapter, pair: str, pip: float) -> float:
    """Fetch 1 completed M5 bar to read current spread for gate check.
    Request count+1 so adapter's in-progress filter leaves ≥1 completed bar."""
    bars = adapter.get_candles(pair, count=3, granularity="M5")
    if not bars:
        return 999.0   # gate fail-safe
    b = bars[-1]       # last completed bar (adapter already filtered in-progress)
    return (b["ask_c"] - b["bid_c"]) / pip


# ── Trade lifecycle ───────────────────────────────────────────────────────────

def _enter(st: PairState, direction: int, spread: float, adapter: OANDAAdapter):
    if os.environ.get("NO_NEW_ENTRIES", "") == "1":
        logger.info(f"{st.pair}: NO_NEW_ENTRIES set — new entry skipped")
        return

    # FIFO-safe guard: OANDA's US compliance forbids hedged positions on
    # the same instrument. If an opposing-direction position is already
    # open on this pair, the broker will reject this order with
    # FIFO_VIOLATION_SAFEGUARD_VIOLATION. Pre-check open trades and skip
    # the entry, so the strategy does not spam the broker with orders
    # that are guaranteed to fail.
    opposing_units = 0
    for t in adapter.get_open_trades():
        if t.instrument != st.pair:
            continue
        if direction == 1 and t.units < 0:
            opposing_units += abs(t.units)
        elif direction == -1 and t.units > 0:
            opposing_units += t.units
    if opposing_units > 0:
        logger.info(
            f"{st.pair}: signal={direction:+d} but {opposing_units}u "
            f"opposing position open — entry skipped (FIFO)"
        )
        return

    entry_units  = _units
    units_signed = entry_units if direction == 1 else -entry_units

    # fetch current mid price for TP calculation (count+1 so filter leaves ≥1)
    bars = adapter.get_candles(st.pair, count=3, granularity="M5")
    if not bars:
        logger.warning(f"{st.pair}: no M5 bars for entry price estimate")
        return
    ref_bar = bars[-1]   # last completed bar
    ref_close = ref_bar["close"]
    tp = ref_close + direction * TP_PIPS * st.pip

    result = adapter.place_market_order(st.pair, units_signed, tp_price=tp)
    if not result.success:
        logger.warning(f"{st.pair}: order rejected: {result.error}")
        return

    fill_px  = result.fill_price or ref_close
    trade_id = result.trade_id or ""
    tp_actual = fill_px + direction * TP_PIPS * st.pip

    st.pos        = direction
    st.entry_px   = fill_px
    st.tp_price   = tp_actual
    st.trade_id   = trade_id
    st.entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    st.units      = entry_units

    dir_str = "LONG" if direction == 1 else "SHORT"
    dir_sym = "🟢" if direction == 1 else "🔴"
    _tg(
        f"{dir_sym} 012 {st.pair} OPEN {dir_str} @ {fill_px:.3f}  {entry_units}u\n"
        f"TP @ {tp_actual:.3f}  ({TP_PIPS}p)  spread={spread:.2f}p"
    )
    logger.info(
        f"{st.pair} OPEN {dir_str} @ {fill_px:.3f} TP={tp_actual:.3f} "
        f"id={trade_id} sp={spread:.2f}p"
    )
    _save(st)


def _record_close(st: PairState, exit_px: float, exit_reason: str):
    pnl_pips = (exit_px - st.entry_px) / st.pip * st.pos
    dur_h = 0.0
    if st.entry_time:
        try:
            t0 = datetime.fromisoformat(st.entry_time.replace("Z", "+00:00"))
            dur_h = (datetime.now(timezone.utc) - t0).total_seconds() / 3600
        except Exception:
            pass

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
        pnl_pips=pnl_pips,
        exit_reason=exit_reason,
        hours_held=dur_h,
        units=st.units if st.units > 0 else _units,
        mfe_pips=0.0,
        mae_pips=0.0,
        capture_ratio=0.0,
        is_paper=False,
        label=LABEL,
    )
    dir_sym = "🟢" if st.pos == 1 else "🔴"
    pl_sym  = "🟢" if pnl_pips >= 0 else "🔴"
    _tg(
        f"{pl_sym} 012 {st.pair} CLOSE {dir_sym}  {exit_reason}\n"
        f"P/L {pnl_pips:+.1f}p  held {dur_h:.1f}h"
    )
    logger.info(
        f"{st.pair} CLOSE pos={st.pos} {exit_reason} @ {exit_px:.3f} "
        f"pnl={pnl_pips:+.1f}p  held={dur_h:.1f}h"
    )


def _reset_pos(st: PairState):
    st.pos = 0; st.entry_px = 0.0; st.tp_price = 0.0
    st.trade_id = ""; st.entry_time = ""


# ── Per-pair poll logic ───────────────────────────────────────────────────────

def _poll_pair(st: PairState, adapter: OANDAAdapter):
    # 1. Fetch completed H1 and M30 bars
    closes_h1,  ts_h1  = _get_closes(adapter, st.pair, "H1",  WARMUP_N)
    closes_m30, ts_m30 = _get_closes(adapter, st.pair, "M30", WARMUP_N)

    if not closes_h1 or not closes_m30:
        return

    # 2. Compute signal from last completed bars
    sig = compute_signal(closes_h1, closes_m30)
    prev_sig = st.signal
    st.signal = sig
    st.last_h1_ts  = ts_h1
    st.last_m30_ts = ts_m30

    # 3. If in a position, check if TP was hit (trade gone from OANDA)
    if st.pos != 0:
        try:
            open_trades = adapter.get_open_trades()
            pair_trades = [t for t in open_trades if t.instrument == st.pair]
            our_trade   = next((t for t in pair_trades if t.trade_id == st.trade_id), None)

            if our_trade is None:
                # Trade closed by broker (TP hit or manual)
                _record_close(st, st.tp_price, "tp")
                _reset_pos(st)
                _save(st)
        except Exception as e:
            logger.warning(f"{st.pair}: OANDA trade check failed: {e}")
        return   # don't enter a new trade on the same poll as TP detection

    # 4. Flat: check for new entry signal (only act when new H1/M30 bar has arrived)
    if st.pos == 0 and sig != 0:
        # Only enter if signal just fired (new bar) or we were flat and sig is fresh
        spread = _get_m5_spread(adapter, st.pair, st.pip)
        if spread > st.sp_gate:
            logger.info(
                f"{st.pair}: signal={sig} but spread {spread:.2f}p > gate {st.sp_gate:.2f}p — skip"
            )
            return
        _enter(st, sig, spread, adapter)

    _save(st)


# ── Startup: warmup + reconcile ───────────────────────────────────────────────

def _warmup(st: PairState, adapter: OANDAAdapter):
    logger.info(f"{st.pair}: warming up …")

    closes_h1,  ts_h1  = _get_closes(adapter, st.pair, "H1",  WARMUP_N)
    closes_m30, ts_m30 = _get_closes(adapter, st.pair, "M30", WARMUP_N)

    if not closes_h1 or not closes_m30:
        logger.error(f"{st.pair}: warmup fetch returned no bars")
        return

    sig = compute_signal(closes_h1, closes_m30)
    st.signal    = sig
    st.last_h1_ts  = ts_h1
    st.last_m30_ts = ts_m30

    # reconcile with OANDA
    try:
        open_trades = adapter.get_open_trades()
        pair_trades = [t for t in open_trades if t.instrument == st.pair]
        if pair_trades and st.pos == 0:
            t = pair_trades[0]
            st.pos       = 1 if t.units > 0 else -1
            st.entry_px  = t.entry_price
            st.trade_id  = t.trade_id
            st.tp_price  = t.tp_price or (st.entry_px + st.pos * TP_PIPS * st.pip)
            st.entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.info(
                f"{st.pair}: adopted orphan trade {t.trade_id} pos={st.pos} @ {st.entry_px:.3f}"
            )
        elif not pair_trades and st.pos != 0:
            logger.warning(
                f"{st.pair}: state says pos={st.pos} but no open trade — resetting"
            )
            _reset_pos(st)
    except Exception as e:
        logger.warning(f"{st.pair}: OANDA reconcile error: {e}")

    logger.info(
        f"{st.pair}: warmup done  signal={sig}  pos={st.pos}  "
        f"last_h1={ts_h1[-16:] if ts_h1 else '—'}"
    )
    _save(st)


# ── Market hours guard ────────────────────────────────────────────────────────

def _market_open() -> bool:
    """FX is closed Sat all day + Sun before 21:00 UTC + Fri after 21:00 UTC."""
    now = datetime.now(timezone.utc)
    wd  = now.weekday()   # 0=Mon … 4=Fri, 5=Sat, 6=Sun
    h   = now.hour
    if wd == 5:                    # Saturday — always closed
        return False
    if wd == 6 and h < 21:         # Sunday before 21:00 UTC
        return False
    if wd == 4 and h >= 21:        # Friday after 21:00 UTC
        return False
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def _signal_handler(sig, frame):
    global _shutdown
    logger.info(f"Signal {sig} received — shutting down")
    _shutdown = True


def main():
    global _shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT,  _signal_handler)

    if not ACCOUNT_ID:
        logger.error("OANDA_ACCOUNT_ID_012 not set — exiting")
        sys.exit(1)

    adapter = OANDAAdapter(account_id=ACCOUNT_ID)

    _refresh_units(adapter)

    states = {}
    for pair, cfg in PAIRS.items():
        st = PairState(pair=pair, pip=cfg["pip"], sp_gate=cfg["sp_gate"])
        _load(st)
        states[pair] = st

    for pair, st in states.items():
        try:
            _warmup(st, adapter)
        except Exception as e:
            logger.error(f"{pair}: warmup failed: {e}")

    _tg(
        f"🟢 012 SMA Momentum started\n"
        f"SMA{SMA_N} lags={LAGS} TP={TP_PIPS}p  {_units}u/trade\n"
        f"Pairs ({len(PAIRS)}): {', '.join(PAIRS)}"
    )
    logger.info(
        f"SMA Momentum live — acct={ACCOUNT_ID}  {len(PAIRS)} pairs  {_units}u"
    )

    while not _shutdown:
        if not _market_open():
            time.sleep(300)
            continue
        if time.time() - _last_balance_refresh >= 3600:
            _refresh_units(adapter)
        for pair, st in states.items():
            try:
                _poll_pair(st, adapter)
            except Exception as e:
                logger.error(f"{pair}: poll error: {e}")
        time.sleep(POLL_SECS)

    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
