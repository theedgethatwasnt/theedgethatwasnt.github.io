#!/usr/bin/env python3
"""
FIFO-Trends Live Strategy — GBP_JPY_ft on Account 013
Config: b5_r1_n4_E2_X3c_1_5  sp_gate=4.0p  OOS_ref=135.3p/d (S5-monitor fills)

FILL MODEL (S5 exit monitoring):
  Entry:      M5 bar close (market order at M5 boundary)
  Trail exit: S5 bar close (≤5s detection lag → fill ≈ trail ± 0.3p)
  X7 exit:    M5 bar close (manual col_count check → bar-close fill correct)

Session 068 RCA: M5 manual management caused 5-min detection lag → 3–10p below
trail. S5 monitoring cuts lag to ≤5s; S5 bars are 0.1–0.5p wide → fills match
trail-fill backtest assumption.

SOP compliance (CLAUDE.md §Backtest–Live Consistency SOP):
  R1  Closed bars only — process_bar() called only when timestamp advances
  R2  Within-bar sequence — bull=(close≥open) → HIGH then LOW; bear → LOW then HIGH
  R3  Mid OHLC for signals; spread gated at entry (sp <= sp_gate)
  R3a bid_c/ask_c for spread only
  R3b BA spread used for gate; IS P90=4.0p hardcoded (R5)
  R4  Incremental-only features — ring buffer for col_hist
  R4a col_count (in-progress) never used in completed-column SMA
  R5  IS P90 spread gate = 4.0p hardcoded
  R6  process_bar() logic identical to backtest kernel and paper service
  R7  See tests/test_fifo_r7.py
"""

import os
import sys
import json
import logging
import asyncio
import signal
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

from lib.oanda_adapter import OANDAAdapter
from lib.db import write_trade_direct
from lib.pnf_engine import (
    PnFState, PnFConfig, BarResult,
    process_bar as _pnf_process_bar,
)

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [fifo_live] %(message)s",
)
logger = logging.getLogger("fifo_live")

ACCOUNT_ID   = os.environ.get("OANDA_ACCOUNT_ID_013", "")
STATE_DIR    = os.environ.get("FIFO_LIVE_STATE_DIR", "/tmp")
UNITS        = 10

# ── Config (GBP_JPY_ft) ────────────────────────────────────────────────────────
PAIR       = "GBP_JPY"
PIP        = 0.01
BOX_PIPS   = 5
REV        = 1
N_MIN      = 4
TRAIL_D    = 1
X7_K       = 5
SP_GATE    = 4.0          # IS P90 hardcoded (R5)
OOS_PD_REF = 135.3        # updated: S5-monitor fill model OOS
LABEL      = "GBP_JPY_ft"
FALLBACK_SP = 3.5         # GBP_JPY fallback spread (pips)

_CFG = PnFConfig(
    pip      = PIP,
    box_pips = BOX_PIPS,
    rev      = REV,
    n_min    = N_MIN,
    trail_d  = TRAIL_D,
    x7_k     = X7_K,
    sp_gate  = SP_GATE,
)


# ── Global state ───────────────────────────────────────────────────────────────
st           = PnFState()
bar_count    = 0
bars_in_pos  = 0
entry_bar    = 0
entry_time   = ""
n_trades     = 0
n_wins       = 0
total_pips   = 0.0
oanda_tid: Optional[str] = None
last_ts:   Optional[str] = None
last_s5_ts: Optional[str] = None   # last processed S5 bar timestamp


# ── S5 trail exit (≤5s detection lag → fill ≈ trail ± 0.3p) ──────────────────
def check_s5_trail(s5_bars: list, adapter: OANDAAdapter) -> bool:
    """
    Check new S5 bars for trail exit condition.
    When triggered: submit market order immediately (fill ≈ S5 close ≈ trail level).
    Returns True if position was closed.

    S5 bars are 0.1–0.5 pip wide → detection within 5s of trail trigger →
    fill within ~0.3p of trail price.  This matches the trail-fill backtest assumption.
    """
    global st, oanda_tid, n_trades, n_wins, total_pips, entry_time, last_s5_ts

    if st.pos == 0 or oanda_tid is None:
        return False

    bs    = BOX_PIPS * PIP
    trail = (st.hw_level - TRAIL_D * bs) if st.pos == 1 \
            else (st.hw_level + TRAIL_D * bs)

    new_bars = sorted(
        [b for b in s5_bars if b['timestamp'] > (last_s5_ts or '')],
        key=lambda x: x['timestamp'],
    )

    for bar in new_bars:
        lo = float(bar['low'])
        hi = float(bar['high'])
        cl = float(bar['close'])
        try:
            sp = max(0.0, (float(bar['ask_c']) - float(bar['bid_c'])) / PIP)
            if sp <= 0 or sp > 25:
                sp = FALLBACK_SP
        except Exception:
            sp = FALLBACK_SP
        last_s5_ts = bar['timestamp']

        triggered = (st.pos == 1 and lo <= trail) or (st.pos == -1 and hi >= trail)
        if not triggered:
            continue

        # Submit market order immediately (fill ≈ trail since S5 bars are tiny)
        broker_result = adapter.close_trade(oanda_tid)
        fill_px = broker_result.fill_price if broker_result.success else cl
        if not broker_result.success:
            logger.error(f"S5 close_trade failed: {broker_result.error} — using S5 close")

        pos_closed  = st.pos
        entry_px_r  = st.entry_px
        n_bars_held = bar_count - entry_bar
        pnl_pips    = (fill_px - entry_px_r) * pos_closed / PIP - sp

        logger.info(
            f"S5-TRAIL EXIT: fill={fill_px:.3f} trail={trail:.3f} "
            f"gap={abs(fill_px - trail)/PIP:.2f}p "
            f"pnl={pnl_pips:+.2f}p s5_ts={bar['timestamp']}"
        )

        acct_id_str = ACCOUNT_ID or "001-001-${OANDA_CUSTOMER_ID}-013"
        write_trade_direct(
            strategy      = "fifo_live",
            pair          = PAIR,
            account_id    = acct_id_str,
            trade_id      = f"fifo_live_{LABEL}_{entry_time}",
            direction     = pos_closed,
            entry_price   = entry_px_r,
            exit_price    = fill_px,
            entry_time    = entry_time,
            exit_time     = bar['timestamp'],
            pnl_pips      = pnl_pips,
            exit_reason   = "trail_s5",
            hours_held    = n_bars_held * 5.0 / 60.0,
            units         = UNITS,
            mfe_pips      = 0.0,
            mae_pips      = 0.0,
            capture_ratio = 0.0,
            is_paper      = False,
            label         = LABEL,
        )

        total_pips += pnl_pips
        n_trades   += 1
        if pnl_pips > 0:
            n_wins += 1

        dir_str = "LONG" if pos_closed == 1 else "SHORT"
        emoji   = "🟢" if pnl_pips > 0 else "🔴"
        _tg(
            f"{emoji} [FIFO LIVE] {LABEL}\n"
            f"013 {dir_str} trail_s5: {pnl_pips:+.2f}p ({n_bars_held}bar)\n"
            f"fill={fill_px:.3f}  trail={trail:.3f}  gap={abs(fill_px-trail)/PIP:.2f}p\n"
            f"Running: {total_pips:+.1f}p ({n_wins}/{n_trades} = "
            f"{n_wins/max(n_trades,1)*100:.0f}% WR)"
        )

        # Reset position state — engine won't re-exit on next M5 bar (pos=0 guard)
        st.pos = 0; st.entry_px = 0.0; st.hw_level = 0.0
        oanda_tid = None
        return True

    return False


# ── M5 bar processing (R6: delegates all P&F logic to lib.pnf_engine) ─────────
def process_bar(bar: dict, sp: float, ts: str, adapter: OANDAAdapter):
    global st, bar_count, bars_in_pos, entry_bar, entry_time
    global n_trades, n_wins, total_pips, oanda_tid, last_s5_ts

    cl = float(bar['close'])

    sig = _pnf_process_bar(
        st, _CFG,
        float(bar['open']), float(bar['high']),
        float(bar['low']),  cl,
        sp,
    )

    # ── EXIT (M5 — only X7 reaches here; trail exits handled by S5 loop) ──────
    if sig.exit_triggered and oanda_tid is not None:
        broker_result = adapter.close_trade(oanda_tid)
        fill_px = broker_result.fill_price if broker_result.success else cl
        if not broker_result.success:
            logger.error(f"M5 close_trade failed: {broker_result.error} — using bar close")

        pnl_pips    = (fill_px - sig.exited_entry_px) * sig.exited_pos / PIP - sp
        n_bars_held = bar_count - entry_bar

        logger.info(
            f"M5-EXIT {sig.exit_reason}: fill={fill_px:.3f} "
            f"pnl={pnl_pips:+.2f}p ({n_bars_held}bar)"
        )

        acct_id_str = ACCOUNT_ID or "001-001-${OANDA_CUSTOMER_ID}-013"
        write_trade_direct(
            strategy      = "fifo_live",
            pair          = PAIR,
            account_id    = acct_id_str,
            trade_id      = f"fifo_live_{LABEL}_{entry_time}",
            direction     = sig.exited_pos,
            entry_price   = sig.exited_entry_px,
            exit_price    = fill_px,
            entry_time    = entry_time,
            exit_time     = ts,
            pnl_pips      = pnl_pips,
            exit_reason   = sig.exit_reason,
            hours_held    = n_bars_held * 5.0 / 60.0,
            units         = UNITS,
            mfe_pips      = 0.0,
            mae_pips      = 0.0,
            capture_ratio = 0.0,
            is_paper      = False,
            label         = LABEL,
        )

        total_pips += pnl_pips
        n_trades   += 1
        if pnl_pips > 0:
            n_wins += 1

        dir_str = "LONG" if sig.exited_pos == 1 else "SHORT"
        emoji   = "🟢" if pnl_pips > 0 else "🔴"
        _tg(
            f"{emoji} [FIFO LIVE] {LABEL}\n"
            f"013 {dir_str} {sig.exit_reason}: {pnl_pips:+.2f}p ({n_bars_held}bar)\n"
            f"Running: {total_pips:+.1f}p ({n_wins}/{n_trades} = "
            f"{n_wins/max(n_trades,1)*100:.0f}% WR)"
        )
        oanda_tid = None

    # ── ENTRY ─────────────────────────────────────────────────────────────────
    if sig.entry_signal != 0:
        direction   = sig.entry_signal
        oanda_units = UNITS if direction == 1 else -UNITS
        broker_result = adapter.place_market_order(PAIR, oanda_units)
        if broker_result.success:
            fill_px      = broker_result.fill_price
            oanda_tid    = broker_result.trade_id
            st.entry_px  = fill_px
            entry_bar    = bar_count
            entry_time   = ts
            # Anchor S5 timestamp to entry wall-clock time so pre-entry S5 bars
            # (which may already violate the trail due to P&F level vs fill divergence)
            # are not processed by check_s5_trail on the next 5s poll.
            last_s5_ts   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
            dir_str = "LONG" if direction == 1 else "SHORT"
            logger.info(f"ENTRY {dir_str}: fill={fill_px:.3f} tid={oanda_tid}")
            _tg(
                f"{'🟢' if direction==1 else '🔴'} [FIFO LIVE] {LABEL}\n"
                f"013 {dir_str} ENTRY @ {fill_px:.3f}  {UNITS}u\n"
                f"PnF idx={st.pnf_idx} dir={'↑' if st.pnf_dir==1 else '↓'}"
            )
        else:
            logger.error(f"Entry order failed: {broker_result.error}")
            st.pos = 0; st.entry_px = 0.0; st.hw_level = 0.0; st.pending = 0

    bar_count += 1
    if st.pos != 0:
        bars_in_pos += 1


# ── Status writer ──────────────────────────────────────────────────────────────
def write_status(path: str) -> None:
    days = bar_count / 288
    pd   = total_pips / days if days > 0 else 0.0
    wr   = n_wins / n_trades if n_trades > 0 else 0.0
    bs   = BOX_PIPS * PIP
    trail_level = (st.hw_level - TRAIL_D * bs) if st.pos == 1 \
                  else (st.hw_level + TRAIL_D * bs) if st.pos == -1 else None
    data = {
        "updated":     datetime.now(timezone.utc).isoformat(),
        "label":       LABEL,
        "pair":        PAIR,
        "units":       UNITS,
        "account_id":  ACCOUNT_ID,
        "oos_pd_ref":  OOS_PD_REF,
        "exit_mode":   "S5_monitor",
        "bar_count":   bar_count,
        "days_elapsed": round(days, 1),
        "n_trades":    n_trades,
        "n_wins":      n_wins,
        "win_rate":    round(wr * 100, 1),
        "total_pips":  round(total_pips, 2),
        "pips_per_day": round(pd, 2),
        "pos":         st.pos,
        "entry_px":    round(st.entry_px, 3) if st.pos != 0 else None,
        "trail_level": round(trail_level, 3) if trail_level is not None else None,
        "hw_level":    round(st.hw_level, 3) if st.pos != 0 else None,
        "oanda_tid":   oanda_tid,
        "pnf_idx":     st.pnf_idx,
        "pnf_level":   round(st.pnf_level, 3),
        "pnf_dir":     st.pnf_dir,
        "col_count":   st.col_count,
        "last_s5_ts":  last_s5_ts,
    }
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ── Telegram ───────────────────────────────────────────────────────────────────
import requests as _requests
_TG_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
_TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def _tg(msg: str) -> None:
    if not _TG_TOKEN or not _TG_CHAT_ID:
        return
    try:
        _requests.post(
            f"https://api.telegram.org/bot{_TG_TOKEN}/sendMessage",
            json={"chat_id": _TG_CHAT_ID, "text": msg},
            timeout=8,
        )
    except Exception:
        pass


# ── Startup reconciliation ─────────────────────────────────────────────────────
def reconcile_open_position(adapter: OANDAAdapter) -> None:
    global oanda_tid, st
    trades = adapter.get_open_trades()
    for t in trades:
        if t.instrument == PAIR:
            direction = 1 if t.units > 0 else -1
            oanda_tid   = t.trade_id
            st.pos      = direction
            st.entry_px = t.entry_price
            st.hw_level = t.entry_price
            logger.warning(
                f"RECONCILE: found open {PAIR} pos {t.units}u @ {t.entry_price:.3f} "
                f"tid={oanda_tid} — adopted"
            )
            _tg(
                f"⚠️ [FIFO LIVE] {LABEL} restarted with open position\n"
                f"013 {'LONG' if direction==1 else 'SHORT'} @ {t.entry_price:.3f}  "
                f"tid={oanda_tid}\n"
                f"S5 trail monitoring active (poll every 5s)"
            )
            return
    logger.info(f"Reconcile: no open {PAIR} position on 013 — starting flat")


# ── Main loop ──────────────────────────────────────────────────────────────────
async def run_live(shutdown_event: asyncio.Event) -> None:
    global last_ts

    adapter      = OANDAAdapter(account_id=ACCOUNT_ID)
    read_adapter = OANDAAdapter(account_id=os.environ.get("OANDA_ACCOUNT_ID_011", ACCOUNT_ID))
    status_path  = os.path.join(STATE_DIR, "fifo_live_status.json")

    reconcile_open_position(adapter)

    _tg(
        f"🚀 [FIFO LIVE] {LABEL} started\n"
        f"013 {PAIR}  {UNITS}u  b{BOX_PIPS}_r{REV}_n{N_MIN}_E2_X3c_1_{X7_K}\n"
        f"sp_gate={SP_GATE}p  OOS_ref={OOS_PD_REF}p/d\n"
        f"Exit mode: S5 monitoring (5s trail detection)"
    )
    logger.info(
        f"FIFO-Trends live started — {LABEL} on {ACCOUNT_ID}  "
        f"exit_mode=S5_monitor  poll=5s"
    )

    loop = asyncio.get_event_loop()

    while not shutdown_event.is_set():
        try:
            # ── S5 trail exit monitoring (when in position) ───────────────────
            # Polls every 5s → detection lag ≤5s → fill ≈ trail ± 0.3p
            if st.pos != 0 and oanda_tid is not None:
                try:
                    s5_bars = await loop.run_in_executor(
                        None,
                        lambda: read_adapter.get_candles(PAIR, count=6, granularity="S5"),
                    )
                    if s5_bars:
                        exited = check_s5_trail(s5_bars, adapter)
                        if exited:
                            write_status(status_path)
                except Exception as e:
                    logger.warning(f"S5 trail check error: {e}")

            # ── M5 P&F chart update + entry detection ─────────────────────────
            # Trail exits that reach here are only X7 (col-SMA exits, rare ~1%)
            candles = await loop.run_in_executor(
                None, lambda: read_adapter.get_candles(PAIR, count=3, granularity="M5")
            )
            if candles:
                bar = candles[-1]
                ts  = bar['timestamp']
                if ts != last_ts:
                    last_ts = ts
                    try:
                        sp = max(0.0, (float(bar['ask_c']) - float(bar['bid_c'])) / PIP)
                        if sp <= 0 or sp > 25:
                            raise ValueError
                    except Exception:
                        sp = FALLBACK_SP

                    process_bar(bar, sp, ts, adapter)
                    write_status(status_path)

                    if bar_count > 0 and bar_count % 12 == 0:
                        days = bar_count / 288
                        pd   = total_pips / days if days > 0 else 0.0
                        logger.info(
                            f"[{LABEL}] bars={bar_count} trades={n_trades} "
                            f"wr={n_wins/max(n_trades,1)*100:.0f}% "
                            f"pips={total_pips:+.1f} ({pd:.1f}p/d)"
                        )

                    if bar_count > 0 and bar_count % 288 == 0:
                        days = bar_count / 288
                        pd   = total_pips / days if days > 0 else 0.0
                        wr   = n_wins / max(n_trades, 1) * 100
                        flag = "🟢" if pd >= OOS_PD_REF * 0.5 else ("🟡" if pd >= 0 else "🔴")
                        _tg(
                            f"{flag} [FIFO LIVE] {LABEL} Day {days:.0f}\n"
                            f"013 {total_pips:+.0f}p total ({pd:.1f}p/d vs {OOS_PD_REF}p/d ref)\n"
                            f"{n_trades}tr  {wr:.0f}%WR"
                        )

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)

        await asyncio.sleep(5)   # 5s: S5 bar granularity (was 15s)


async def main() -> None:
    shutdown = asyncio.Event()

    def _handle(sig, frame):
        logger.info(f"Signal {sig} — shutting down")
        shutdown.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT,  _handle)

    await run_live(shutdown)


if __name__ == "__main__":
    asyncio.run(main())
