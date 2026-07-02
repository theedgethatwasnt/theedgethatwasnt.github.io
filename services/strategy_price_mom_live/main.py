#!/usr/bin/env python3
"""
Price Momentum Confluence Live — Account 011, 12 pairs, 25 units each
======================================================================
Signal: raw close momentum at lags (1,3,8) on M15 + M5 bars.
        mom_k = close[t] - close[t-k]  (no SMA smoothing)
        LONG  when all 6 momentum values > 0 (strict 6/6)
        SHORT when all 6 momentum values < 0
Entry:  market order on signal fire, spread ≤ IS-P90 gate
Exit:   TP=10p placed as broker-side takeProfitOnFill

Validation: M15+M5 lags=(1,3,8) TP=10p — 12/12 pairs IS 3/3 + MC p<0.05
  GBP_JPY +4.5  CAD_JPY +4.4  EUR_JPY +4.0  AUD_JPY +3.2  USD_JPY +3.1
  NZD_USD +2.6  NZD_JPY +2.6  EUR_USD +2.1  AUD_USD +1.7  GBP_USD +1.2
  CHF_JPY +0.6  EUR_GBP +0.3  Portfolio mc_p=0.0000  WR=99.1%

SOP compliance (CLAUDE.md §Backtest–Live Consistency SOP):
  R1  Closed bars only — bars[:-1] excludes in-progress bar
  R3  Built on mid closes (OANDA 'M' price)
  R3a M5 bid_c/ask_c for spread gate check at entry
  R5  IS P90 spread gates hardcoded per pair (from IS M5 BA data)
  R6  compute_signal() mirrors mc_validate_price_mom.py exactly
  R7  Warmup fetches 20 completed M15+M5 bars; reconciles open pos vs OANDA
"""

import os
import sys
import json
import time
import signal
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from dotenv import load_dotenv
load_dotenv()

from lib.oanda_adapter import OANDAAdapter
from lib.db import write_trade_direct

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [pm_live] %(message)s",
)
logger = logging.getLogger("pm_live")

# ── Account + sizing ──────────────────────────────────────────────────────────
ACCOUNT_ID       = os.environ.get("OANDA_ACCOUNT_ID_011", "<OANDA_ACCOUNT_ID>")
STATE_DIR        = os.environ.get("PM_STATE_DIR", "/data/logs")
UNITS_PER_DOLLAR = 1.25          # units = round(balance * UNITS_PER_DOLLAR)
_units           = 25            # live value, refreshed from broker every hour
_last_balance_refresh = 0.0
STRATEGY   = "price_mom_confluence"
LABEL      = "pm_m15m5_011"

# ── Signal params (validated M15+M5 lags=(1,3,8) TP=10p) ─────────────────────
LAGS     = (1, 3, 8)
TP_PIPS  = 10.0
WARMUP_N = 20          # completed M15/M5 bars to fetch (≥ max_lag+1 = 9)
POLL_SECS = 60

# ── 12-pair config: pip + IS P90 spread gate (pips, R5) ──────────────────────
PAIRS = {
    "GBP_JPY": {"pip": 0.01,   "sp_gate": 4.00},
    "CAD_JPY": {"pip": 0.01,   "sp_gate": 2.60},
    "EUR_JPY": {"pip": 0.01,   "sp_gate": 2.50},
    "AUD_JPY": {"pip": 0.01,   "sp_gate": 2.30},
    "USD_JPY": {"pip": 0.01,   "sp_gate": 2.10},
    "NZD_JPY": {"pip": 0.01,   "sp_gate": 3.10},
    "CHF_JPY": {"pip": 0.01,   "sp_gate": 3.70},
    "NZD_USD": {"pip": 0.0001, "sp_gate": 2.00},
    "EUR_USD": {"pip": 0.0001, "sp_gate": 1.70},
    "AUD_USD": {"pip": 0.0001, "sp_gate": 1.60},
    "GBP_USD": {"pip": 0.0001, "sp_gate": 2.40},
    "EUR_GBP": {"pip": 0.0001, "sp_gate": 2.00},
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


# ── Signal computation (R6: mirrors mc_validate_price_mom.py exactly) ─────────

def compute_signal(closes_m15: list, closes_m5: list) -> int:
    """
    Returns +1 (long), -1 (short), 0 (flat).
    Raw price momentum: mom_k = close[-1] - close[-1-k] on each TF.
    Needs max(LAGS)+1 = 9 completed bars per TF.
    """
    need = max(LAGS) + 1  # 9
    if len(closes_m15) < need or len(closes_m5) < need:
        return 0

    def tf_moms(closes):
        arr = np.array(closes[-need:], dtype=np.float64)
        cur = arr[-1]
        return [cur - arr[-(k + 1)] for k in LAGS]

    all_moms = tf_moms(closes_m15) + tf_moms(closes_m5)   # 6 values
    n_pos = sum(1 for m in all_moms if m > 0)
    if n_pos == 6:
        return 1
    if n_pos == 0:
        return -1
    return 0


# ── Per-pair state ────────────────────────────────────────────────────────────
@dataclass
class PairState:
    pair:        str
    pip:         float
    sp_gate:     float
    pos:         int   = 0
    entry_px:    float = 0.0
    tp_price:    float = 0.0
    trade_id:    str   = ""
    entry_time:  str   = ""
    signal:      int   = 0
    last_m15_ts: str   = ""
    last_m5_ts:  str   = ""
    units:       int   = 0    # units at entry; 0 = not yet in a trade


def _state_path(pair: str) -> str:
    return os.path.join(STATE_DIR, f"pm_live_{pair}_state.json")


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
    """Returns (closes_list, last_completed_ts). Excludes in-progress bar (R1)."""
    bars = adapter.get_candles(pair, count=count + 1, granularity=granularity)
    if not bars:
        return [], ""
    completed = bars[:-1]
    closes    = [b["close"] for b in completed]
    last_ts   = completed[-1]["timestamp"] if completed else ""
    return closes, last_ts


def _get_m5_spread(adapter: OANDAAdapter, pair: str, pip: float) -> float:
    bars = adapter.get_candles(pair, count=3, granularity="M5")
    if not bars:
        return 999.0
    b = bars[-1]
    return (b["ask_c"] - b["bid_c"]) / pip


# ── Trade lifecycle ───────────────────────────────────────────────────────────

def _enter(st: PairState, direction: int, spread: float, adapter: OANDAAdapter):
    if os.environ.get("NO_NEW_ENTRIES", "") == "1":
        logger.info(f"{st.pair}: NO_NEW_ENTRIES set — new entry skipped")
        return
    entry_units  = _units
    units_signed = entry_units if direction == 1 else -entry_units

    bars = adapter.get_candles(st.pair, count=3, granularity="M5")
    if not bars:
        logger.warning(f"{st.pair}: no M5 bars for entry price estimate")
        return
    ref_close = bars[-1]["close"]
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
        f"{dir_sym} 011 {st.pair} OPEN {dir_str} @ {fill_px:.3f}  {entry_units}u\n"
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
        f"{pl_sym} 011 {st.pair} CLOSE {dir_sym}  {exit_reason}\n"
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
    closes_m15, ts_m15 = _get_closes(adapter, st.pair, "M15", WARMUP_N)
    closes_m5,  ts_m5  = _get_closes(adapter, st.pair, "M5",  WARMUP_N)

    if not closes_m15 or not closes_m5:
        return

    sig = compute_signal(closes_m15, closes_m5)
    st.signal      = sig
    st.last_m15_ts = ts_m15
    st.last_m5_ts  = ts_m5

    if st.pos != 0:
        try:
            open_trades = adapter.get_open_trades()
            our_trade   = next(
                (t for t in open_trades
                 if t.instrument == st.pair and t.trade_id == st.trade_id),
                None
            )
            if our_trade is None:
                _record_close(st, st.tp_price, "tp")
                _reset_pos(st)
                _save(st)
        except Exception as e:
            logger.warning(f"{st.pair}: OANDA trade check failed: {e}")
        return

    if sig != 0:
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

    closes_m15, ts_m15 = _get_closes(adapter, st.pair, "M15", WARMUP_N)
    closes_m5,  ts_m5  = _get_closes(adapter, st.pair, "M5",  WARMUP_N)

    if not closes_m15 or not closes_m5:
        logger.error(f"{st.pair}: warmup fetch returned no bars")
        return

    sig = compute_signal(closes_m15, closes_m5)
    st.signal      = sig
    st.last_m15_ts = ts_m15
    st.last_m5_ts  = ts_m5

    try:
        open_trades = adapter.get_open_trades()
        pair_trades = [t for t in open_trades if t.instrument == st.pair]
        if pair_trades and st.pos == 0:
            t = pair_trades[0]
            st.pos        = 1 if t.units > 0 else -1
            st.entry_px   = t.entry_price
            st.trade_id   = t.trade_id
            st.tp_price   = t.tp_price or (st.entry_px + st.pos * TP_PIPS * st.pip)
            st.entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.info(
                f"{st.pair}: adopted orphan trade {t.trade_id} pos={st.pos} @ {st.entry_px:.3f}"
            )
        elif not pair_trades and st.pos != 0:
            logger.warning(f"{st.pair}: state says pos={st.pos} but no open trade — resetting")
            _reset_pos(st)
    except Exception as e:
        logger.warning(f"{st.pair}: OANDA reconcile error: {e}")

    logger.info(
        f"{st.pair}: warmup done  signal={sig}  pos={st.pos}  "
        f"last_m15={ts_m15[-16:] if ts_m15 else '—'}"
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
        logger.error("OANDA_ACCOUNT_ID_011 not set — exiting")
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
        f"🟢 011 Price Momentum started\n"
        f"M15+M5 lags={LAGS} TP={TP_PIPS}p  {_units}u/trade\n"
        f"Pairs ({len(PAIRS)}): {', '.join(PAIRS)}"
    )
    logger.info(
        f"Price Momentum live — acct={ACCOUNT_ID}  {len(PAIRS)} pairs  {_units}u"
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
