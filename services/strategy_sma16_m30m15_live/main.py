#!/usr/bin/env python3
"""
SMA16 Momentum M30+M15 Live — Account 002, 12 pairs
=====================================================
Signal: SMA(16) on M30 + M15, momentum at lags (1,10,20).
        LONG  when all 6 SMA16 momentum values > 0 (strict 6/6)
        SHORT when all 6 SMA16 momentum values < 0
Entry:  market order on signal fire, spread ≤ IS-P90 gate
Exit:   TP=20p placed as broker-side takeProfitOnFill

Validation: SMA16 M30+M15 lags=(1,10,20) TP=20p — 10/12 pairs IS 3/3 + MC p<0.05
  Portfolio mc_p=0.0140  portf_pd=+33.0 p/d

SOP compliance (CLAUDE.md §Backtest–Live Consistency SOP):
  R1  Closed bars only — bars[:-1] excludes in-progress bar
  R3  Built on mid closes
  R3a M5 bid_c/ask_c for spread gate check at entry
  R5  IS P90 spread gates hardcoded per pair
  R6  compute_signal() mirrors backtest_variants.py exactly
  R7  Warmup fetches completed M30+M15 bars; reconciles open pos vs OANDA
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
    format="%(asctime)s %(levelname)s [sma16_m30m15] %(message)s",
)
logger = logging.getLogger("sma16_m30m15")

ACCOUNT_ID       = os.environ.get("OANDA_ACCOUNT_ID_002", "<OANDA_ACCOUNT_ID>")
STATE_DIR        = os.environ.get("SMA16_M30M15_STATE_DIR", "/data/logs")
UNITS_PER_DOLLAR = 1.25
_units           = 25
_last_balance_refresh = 0.0
STRATEGY = "sma16_m30m15"
LABEL    = "sma16_m30m15_002"

SMA_N    = 16
LAGS     = (1, 10, 20)
TP_PIPS  = 20.0
WARMUP_N = 42   # ≥ SMA_N + max(LAGS) + 1 = 37, with buffer
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


def compute_signal(closes_m30: list, closes_m15: list) -> int:
    """
    SMA16 momentum on M30 + M15. LONG if all 6 diffs > 0.
    Need SMA_N + max(LAGS) + 1 = 37 completed bars per TF.
    sma_at_offset(k): SMA16 ending k bars ago
      k=0 → mean of last SMA_N closes
      k>0 → mean of closes[-(SMA_N+k):-k]
    """
    need = SMA_N + max(LAGS) + 1  # 37
    if len(closes_m30) < need or len(closes_m15) < need:
        return 0

    def tf_signal(closes):
        arr = np.array(closes[-need:], dtype=np.float64)
        def sma_at_offset(offset):
            if offset == 0:
                return arr[-SMA_N:].mean()
            return arr[-(SMA_N + offset):-offset].mean()
        sma_now = sma_at_offset(0)
        return [sma_now - sma_at_offset(k) for k in LAGS]

    all_moms = tf_signal(closes_m30) + tf_signal(closes_m15)
    n_pos = sum(1 for m in all_moms if m > 0)
    if n_pos == 6: return 1
    if n_pos == 0: return -1
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
    return os.path.join(STATE_DIR, f"sma16_m30m15_{pair}_state.json")


def _save(st: PairState):
    data = {k: v for k, v in st.__dict__.items() if k not in ("pair", "pip", "sp_gate")}
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
            if hasattr(st, k): setattr(st, k, v)
        logger.info(f"{st.pair}: loaded state pos={st.pos}")
    except Exception as e:
        logger.warning(f"{st.pair}: load state failed: {e}")


def _get_closes(adapter, pair: str, granularity: str, count: int):
    bars = adapter.get_candles(pair, count=count + 1, granularity=granularity)
    if not bars: return [], ""
    completed = bars[:-1]
    return [b["close"] for b in completed], (completed[-1]["timestamp"] if completed else "")


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
    _tg(f"{dir_sym} 002 {st.pair} OPEN {dir_str} @ {fill_px:.3f}  {entry_units}u\n"
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
    _tg(f"{pl_sym} 002 {st.pair} CLOSE {dir_sym}  {exit_reason}\nP/L {pnl_pips:+.1f}p  held {dur_h:.1f}h")
    logger.info(f"{st.pair} CLOSE pos={st.pos} {exit_reason} @ {exit_px:.3f} pnl={pnl_pips:+.1f}p held={dur_h:.1f}h")


def _reset_pos(st: PairState):
    st.pos = 0; st.entry_px = 0.0; st.tp_price = 0.0; st.trade_id = ""; st.entry_time = ""


def _poll_pair(st: PairState, adapter):
    closes_m30, _ = _get_closes(adapter, st.pair, "M30", WARMUP_N)
    closes_m15, _ = _get_closes(adapter, st.pair, "M15", WARMUP_N)
    if not closes_m30 or not closes_m15: return
    sig = compute_signal(closes_m30, closes_m15)
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
    closes_m30, _ = _get_closes(adapter, st.pair, "M30", WARMUP_N)
    closes_m15, _ = _get_closes(adapter, st.pair, "M15", WARMUP_N)
    if not closes_m30 or not closes_m15:
        logger.error(f"{st.pair}: warmup fetch returned no bars"); return
    st.signal = compute_signal(closes_m30, closes_m15)
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
        logger.error("OANDA_ACCOUNT_ID_002 not set — exiting"); sys.exit(1)
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
    _tg(f"🟢 002 SMA16 M30+M15 started\nM30+M15 SMA16 lags={LAGS} TP={TP_PIPS}p  {_units}u/trade\n"
        f"Pairs ({len(PAIRS)}): {', '.join(PAIRS)}")
    logger.info(f"SMA16 M30+M15 live — acct={ACCOUNT_ID}  {len(PAIRS)} pairs  {_units}u")
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
