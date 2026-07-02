#!/usr/bin/env python3
"""
Exhaustion Continuation B Live — Account 004, 12 pairs
=======================================================
Signal (M5): n_consec=4 consecutive same-direction bars + SMA14 distance ≥ dist_mult × sp_gate
  LONG  when last 4 M5 bars all bull (close>open) AND (close - SMA14) / pip ≥ 2.0 × sp_gate
  SHORT when last 4 M5 bars all bear (close<open) AND (SMA14 - close) / pip ≥ 2.0 × sp_gate
Enter IN the exhaustion direction (momentum continuation, not fade).
Exit:   TP=15p broker-side takeProfitOnFill

Validation: n_consec=2, dist_mult=1.0, TP=10p — 11/12 pairs IS 3/3 + MC p<0.05
  Portfolio mc_p=0.0005  portf_pd=+32.0 p/d

SOP compliance (CLAUDE.md §Backtest–Live Consistency SOP):
  R1  Closed bars only — bars[:-1] excludes in-progress bar
  R3  Built on mid closes
  R3a M5 bid_c/ask_c for spread gate check at entry
  R5  IS P90 spread gates hardcoded per pair
  R6  compute_signal() mirrors backtest_exhaust_cont.py exactly
  R7  Warmup fetches completed M5 bars; reconciles open pos vs OANDA
"""

import os, sys, json, time, signal, logging
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
    format="%(asctime)s %(levelname)s [exhaust_b] %(message)s",
)
logger = logging.getLogger("exhaust_b")

ACCOUNT_ID       = os.environ.get("OANDA_ACCOUNT_ID_004", "<OANDA_ACCOUNT_ID>")
STATE_DIR        = os.environ.get("EXHAUST_B_STATE_DIR", "/data/logs")
UNITS_PER_DOLLAR = 1.25
_units           = 25
_last_balance_refresh = 0.0
STRATEGY  = "exhaust_cont_b"
LABEL     = "exhaust_b_004"

N_CONSEC  = 2
DIST_MULT = 1.0
SMA_N     = 14
TP_PIPS   = 10.0
WARMUP_N  = 30   # ≥ SMA_N + N_CONSEC + buffer = 18+
POLL_SECS = 60

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
            json={"chat_id": TELEGRAM_CHAT, "text": msg}, timeout=5,
        )
    except Exception:
        pass


def _refresh_units(adapter):
    global _units, _last_balance_refresh
    _last_balance_refresh = time.time()
    try:
        info = adapter.get_account_summary()
        if info is not None:
            new_units = max(1, round(info.balance * UNITS_PER_DOLLAR))
            if new_units != _units:
                logger.info(f"Units updated: ${info.balance:.2f} × {UNITS_PER_DOLLAR} → {new_units}u (was {_units}u)")
            else:
                logger.info(f"Balance refresh: ${info.balance:.2f} → {new_units}u/trade")
            _units = new_units
    except Exception as e:
        logger.warning(f"Balance refresh failed: {e} — keeping {_units}u")


def compute_signal(closes: list, opens: list, pip: float, sp_gate: float) -> int:
    """
    Returns +1 (long), -1 (short), 0 (flat).
    Checks last N_CONSEC completed M5 bars for same-direction + SMA14 distance.
    Mirrors build_exhaust_sig in backtest_exhaust_cont.py exactly (R6).
    """
    need = SMA_N + N_CONSEC  # enough for SMA14 at the last bar + N_CONSEC bars
    if len(closes) < need or len(opens) < need:
        return 0

    arr_c = np.array(closes[-need:], dtype=np.float64)
    arr_o = np.array(opens[-need:],  dtype=np.float64)
    sma14 = arr_c[-SMA_N:].mean()  # SMA14 at the last completed bar

    # Check last N_CONSEC bars
    last_closes = arr_c[-N_CONSEC:]
    last_opens  = arr_o[-N_CONSEC:]
    all_bull = bool(np.all(last_closes > last_opens))
    all_bear = bool(np.all(last_closes < last_opens))

    cur_close = arr_c[-1]
    dist = (cur_close - sma14) / pip
    if all_bull and dist >= DIST_MULT * sp_gate:
        return 1
    if all_bear and (-dist) >= DIST_MULT * sp_gate:
        return -1
    return 0


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
    signal:     int   = 0
    units:      int   = 0


def _state_path(pair: str) -> str:
    return os.path.join(STATE_DIR, f"exhaust_b_{pair}_state.json")


def _save(st: PairState):
    data = {k: v for k, v in st.__dict__.items() if k not in ("pair", "pip", "sp_gate")}
    try:
        with open(_state_path(st.pair), "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"{st.pair}: save state failed: {e}")


def _load(st: PairState):
    path = _state_path(st.pair)
    if not os.path.exists(path): return
    try:
        data = json.load(open(path))
        for k, v in data.items():
            if hasattr(st, k): setattr(st, k, v)
        logger.info(f"{st.pair}: loaded state pos={st.pos}")
    except Exception as e:
        logger.warning(f"{st.pair}: load state failed: {e}")


def _get_m5_bars(adapter, pair: str, count: int):
    """Returns (closes, opens, last_ts). Excludes in-progress bar (R1)."""
    bars = adapter.get_candles(pair, count=count + 1, granularity="M5")
    if not bars: return [], [], ""
    completed = bars[:-1]
    closes = [b["close"] for b in completed]
    opens  = [b.get("open", b["close"]) for b in completed]
    last_ts = completed[-1]["timestamp"] if completed else ""
    return closes, opens, last_ts


def _get_m5_spread(adapter, pair: str, pip: float) -> float:
    bars = adapter.get_candles(pair, count=3, granularity="M5")
    if not bars: return 999.0
    b = bars[-1]
    return (b["ask_c"] - b["bid_c"]) / pip


def _enter(st: PairState, direction: int, spread: float, adapter):
    if os.environ.get("NO_NEW_ENTRIES", "") == "1":
        logger.info(f"{st.pair}: NO_NEW_ENTRIES set — new entry skipped")
        return
    entry_units  = _units
    units_signed = entry_units if direction == 1 else -entry_units
    bars = adapter.get_candles(st.pair, count=3, granularity="M5")
    if not bars:
        logger.warning(f"{st.pair}: no M5 bars for entry price estimate"); return
    ref_close = bars[-1]["close"]
    tp = ref_close + direction * TP_PIPS * st.pip
    result = adapter.place_market_order(st.pair, units_signed, tp_price=tp)
    if not result.success:
        logger.warning(f"{st.pair}: order rejected: {result.error}"); return
    fill_px  = result.fill_price or ref_close
    trade_id = result.trade_id or ""
    tp_actual = fill_px + direction * TP_PIPS * st.pip
    st.pos = direction; st.entry_px = fill_px; st.tp_price = tp_actual
    st.trade_id = trade_id; st.units = entry_units
    st.entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dir_str = "LONG" if direction == 1 else "SHORT"
    dir_sym = "🟢" if direction == 1 else "🔴"
    _tg(f"{dir_sym} 004 {st.pair} OPEN {dir_str} @ {fill_px:.3f}  {entry_units}u\n"
        f"TP @ {tp_actual:.3f}  ({TP_PIPS}p)  spread={spread:.2f}p")
    logger.info(f"{st.pair} OPEN {dir_str} @ {fill_px:.3f} TP={tp_actual:.3f} id={trade_id} sp={spread:.2f}p")
    _save(st)


def _record_close(st: PairState, exit_px: float, exit_reason: str):
    pnl_pips = (exit_px - st.entry_px) / st.pip * st.pos
    dur_h = 0.0
    if st.entry_time:
        try:
            t0 = datetime.fromisoformat(st.entry_time.replace("Z", "+00:00"))
            dur_h = (datetime.now(timezone.utc) - t0).total_seconds() / 3600
        except Exception: pass
    write_trade_direct(
        strategy=STRATEGY, pair=st.pair, account_id=ACCOUNT_ID, trade_id=st.trade_id,
        direction=st.pos, entry_price=st.entry_px, exit_price=exit_px,
        entry_time=st.entry_time, exit_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        pnl_pips=pnl_pips, exit_reason=exit_reason, hours_held=dur_h,
        units=st.units if st.units > 0 else _units,
        mfe_pips=0.0, mae_pips=0.0, capture_ratio=0.0, is_paper=False, label=LABEL,
    )
    dir_sym = "🟢" if st.pos == 1 else "🔴"
    pl_sym  = "🟢" if pnl_pips >= 0 else "🔴"
    _tg(f"{pl_sym} 004 {st.pair} CLOSE {dir_sym}  {exit_reason}\nP/L {pnl_pips:+.1f}p  held {dur_h:.1f}h")
    logger.info(f"{st.pair} CLOSE pos={st.pos} {exit_reason} @ {exit_px:.3f} pnl={pnl_pips:+.1f}p held={dur_h:.1f}h")


def _reset_pos(st: PairState):
    st.pos = 0; st.entry_px = 0.0; st.tp_price = 0.0; st.trade_id = ""; st.entry_time = ""


def _poll_pair(st: PairState, adapter):
    closes, opens, _ = _get_m5_bars(adapter, st.pair, WARMUP_N)
    if not closes: return
    sig = compute_signal(closes, opens, st.pip, st.sp_gate)
    st.signal = sig
    if st.pos != 0:
        try:
            open_trades = adapter.get_open_trades()
            our_trade   = next((t for t in open_trades
                                if t.instrument == st.pair and t.trade_id == st.trade_id), None)
            if our_trade is None:
                _record_close(st, st.tp_price, "tp")
                _reset_pos(st); _save(st)
        except Exception as e:
            logger.warning(f"{st.pair}: OANDA trade check failed: {e}")
        return
    if sig != 0:
        spread = _get_m5_spread(adapter, st.pair, st.pip)
        if spread > st.sp_gate:
            logger.info(f"{st.pair}: signal={sig} but spread {spread:.2f}p > gate {st.sp_gate:.2f}p — skip")
            return
        _enter(st, sig, spread, adapter)
    _save(st)


def _warmup(st: PairState, adapter):
    logger.info(f"{st.pair}: warming up …")
    closes, opens, _ = _get_m5_bars(adapter, st.pair, WARMUP_N)
    if not closes:
        logger.error(f"{st.pair}: warmup fetch returned no bars"); return
    st.signal = compute_signal(closes, opens, st.pip, st.sp_gate)
    try:
        open_trades = adapter.get_open_trades()
        pair_trades = [t for t in open_trades if t.instrument == st.pair]
        if pair_trades and st.pos == 0:
            t = pair_trades[0]
            st.pos = 1 if t.units > 0 else -1
            st.entry_px = t.entry_price; st.trade_id = t.trade_id
            st.tp_price = t.tp_price or (st.entry_px + st.pos * TP_PIPS * st.pip)
            st.entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.info(f"{st.pair}: adopted orphan trade {t.trade_id} pos={st.pos} @ {st.entry_px:.3f}")
        elif not pair_trades and st.pos != 0:
            logger.warning(f"{st.pair}: state says pos={st.pos} but no open trade — resetting")
            _reset_pos(st)
    except Exception as e:
        logger.warning(f"{st.pair}: OANDA reconcile error: {e}")
    logger.info(f"{st.pair}: warmup done signal={st.signal} pos={st.pos}")
    _save(st)


def _market_open() -> bool:
    now = datetime.now(timezone.utc); wd = now.weekday(); h = now.hour
    if wd == 5: return False
    if wd == 6 and h < 21: return False
    if wd == 4 and h >= 21: return False
    return True


def _signal_handler(sig, frame):
    global _shutdown
    logger.info(f"Signal {sig} received — shutting down")
    _shutdown = True


def main():
    global _shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT,  _signal_handler)
    if not ACCOUNT_ID:
        logger.error("OANDA_ACCOUNT_ID_004 not set — exiting"); sys.exit(1)
    adapter = OANDAAdapter(account_id=ACCOUNT_ID)
    _refresh_units(adapter)
    states = {pair: PairState(pair=pair, pip=cfg["pip"], sp_gate=cfg["sp_gate"])
              for pair, cfg in PAIRS.items()}
    for pair, st in states.items():
        _load(st)
    for pair, st in states.items():
        try:
            _warmup(st, adapter)
        except Exception as e:
            logger.error(f"{pair}: warmup failed: {e}")
    _tg(f"🟢 004 Exhaust-Cont B started\nn_consec={N_CONSEC} dist_mult={DIST_MULT} TP={TP_PIPS}p  {_units}u/trade\n"
        f"Pairs ({len(PAIRS)}): {', '.join(PAIRS)}")
    logger.info(f"Exhaust-Cont B live — acct={ACCOUNT_ID}  {len(PAIRS)} pairs  {_units}u")
    while not _shutdown:
        if not _market_open():
            time.sleep(300); continue
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
