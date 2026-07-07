#!/usr/bin/env python3
"""
TR Momentum Paper Trading Service.

Signal: if True Range > threshold_pips → enter at bar close in bar direction.
Exit:   trail stop (fixed pips from high-water mark).
One trade at a time per config. No real orders placed.

Configs (from backtest_tr_entry_v2.py — v2 sweep with IS P90 spread gate + MC, Session 067):
  USD_JPY_tr : TR12 trail2  OOS 94.4 p/d  sp_gate=2.1p
  GBP_JPY_tr : TR15 trail2  OOS 85.1 p/d  sp_gate=4.0p
  EUR_JPY_tr : TR12 trail2  OOS 58.4 p/d  sp_gate=2.5p
  EUR_USD_tr : TR8  trail2  OOS 29.1 p/d  sp_gate=1.7p
  GBP_USD_tr : TR10 trail2  OOS 26.5 p/d  sp_gate=2.4p
  AUD_JPY_tr : TR10 trail2  OOS 25.3 p/d  sp_gate=2.3p
  CAD_JPY_tr : TR10 trail2  OOS 15.5 p/d  sp_gate=2.6p
  CHF_JPY_tr : TR12 trail2  OOS 15.4 p/d  sp_gate=3.7p
  AUD_USD_tr : TR8  trail2  OOS 13.7 p/d  sp_gate=1.6p
  NZD_JPY_tr : TR12 trail2  OOS  8.8 p/d  sp_gate=3.1p
  NZD_USD_tr : TR8  trail2  OOS  5.2 p/d  sp_gate=2.0p
  EUR_GBP_tr : TR8  trail2  OOS  2.5 p/d  sp_gate=2.0p

SOP compliance (CLAUDE.md §Backtest-Live Consistency SOP):
  R1  Closed bars only — process_bar() called only when timestamp advances
  R2  No within-bar sequencing needed (no P&F chart)
  R3  Mid OHLC for signals; spread deducted at P&L close
  R3a bid_c/ask_c for spread only; never bid_h/l ask_h/l
  R3b SP gate hardcoded per pair (IS P90 conservative estimate)
  R5  IS P90 spread gate hardcoded — never recomputed from live
  R6  TRState + process_bar() logic identical to backtest kernel

Output: /data/logs/tr_paper_status.json
        /data/logs/tr_paper_trades.jsonl
"""

import os
import sys
import json
import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

from lib.oanda_adapter import OANDAAdapter
from lib.db import write_trade_direct

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [tr_paper] %(message)s",
)
logger = logging.getLogger("tr_paper")

STATE_DIR = os.environ.get("TR_STATE_DIR", "/tmp")


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class TRConfig:
    label:      str
    pair:       str
    pip:        float
    tr_thresh:  float   # TR threshold in pips
    trail_pips: float   # trail distance in pips
    sp_gate:    float   # IS P90 spread gate (pips) — hardcoded, never recomputed
    oos_pd:     float   # reference OOS p/d from sweep


CONFIGS = [
    TRConfig("USD_JPY_tr", "USD_JPY", 0.01,    12, 2, 2.1,  94.4),
    TRConfig("GBP_JPY_tr", "GBP_JPY", 0.01,    15, 2, 4.0,  85.1),
    TRConfig("EUR_JPY_tr", "EUR_JPY", 0.01,    12, 2, 2.5,  58.4),
    TRConfig("EUR_USD_tr", "EUR_USD", 0.0001,   8, 2, 1.7,  29.1),
    TRConfig("GBP_USD_tr", "GBP_USD", 0.0001,  10, 2, 2.4,  26.5),
    TRConfig("AUD_JPY_tr", "AUD_JPY", 0.01,    10, 2, 2.3,  25.3),
    TRConfig("CAD_JPY_tr", "CAD_JPY", 0.01,    10, 2, 2.6,  15.5),
    TRConfig("CHF_JPY_tr", "CHF_JPY", 0.01,    12, 2, 3.7,  15.4),
    TRConfig("AUD_USD_tr", "AUD_USD", 0.0001,   8, 2, 1.6,  13.7),
    TRConfig("NZD_JPY_tr", "NZD_JPY", 0.01,    12, 2, 3.1,   8.8),
    TRConfig("NZD_USD_tr", "NZD_USD", 0.0001,   8, 2, 2.0,   5.2),
    TRConfig("EUR_GBP_tr", "EUR_GBP", 0.0001,   8, 2, 2.0,   2.5),
]

PAIRS_NEEDED = list(dict.fromkeys(c.pair for c in CONFIGS))


# ── State + simulator ─────────────────────────────────────────────────────────
@dataclass
class TRState:
    initialized: bool  = False
    prev_close:  float = 0.0
    pos:         int   = 0      # 0=flat, +1=long, -1=short
    entry_px:    float = 0.0
    hw:          float = 0.0    # high-water for trail: max high (long) / min low (short)


@dataclass
class TradeRecord:
    label:       str
    pair:        str
    direction:   int
    entry_time:  str
    exit_time:   str
    entry_px:    float
    exit_px:     float
    pnl_pips:    float
    spread_pips: float
    n_bars:      int
    exit_reason: str
    tr_at_entry: float


class TRPaperSim:
    """
    Simulates one TR-momentum config on live M5 bars.
    R6: identical logic to backtest Numba kernel — pure Python.
    """

    def __init__(self, cfg: TRConfig):
        self.cfg           = cfg
        self.st            = TRState()
        self.records:      List[TradeRecord] = []
        self.total_pips    = 0.0
        self.n_trades      = 0
        self.n_wins        = 0
        self.bar_count     = 0
        self.bars_in_trade = 0
        self.entry_bar     = 0
        self.entry_time    = ""
        self.tr_at_entry   = 0.0
        self._started      = datetime.now(timezone.utc).isoformat()

    def process_bar(self, bar: dict, sp: float, ts: str) -> Optional[TradeRecord]:
        cfg     = self.cfg
        st      = self.st
        pip     = cfg.pip
        o       = float(bar['open'])
        h       = float(bar['high'])
        l       = float(bar['low'])
        c       = float(bar['close'])
        trail_p = cfg.trail_pips * pip

        # First bar: initialise prev_close only
        if not st.initialized:
            st.prev_close  = c
            st.initialized = True
            self.bar_count += 1
            return None

        prev_cl = st.prev_close
        tr_price = max(h, prev_cl) - min(l, prev_cl)   # TR in price units

        new_rec = None

        # ── EXIT ──────────────────────────────────────────────────────────────
        if st.pos != 0:
            if st.pos == 1:
                if h > st.hw:
                    st.hw = h
                trail = st.hw - trail_p
                if l <= trail:
                    pnl_pips = (c - st.entry_px) / pip - sp   # bar-close fill matches live
                    new_rec  = self._close(pnl_pips, c, sp, ts, "trail")
            else:
                if l < st.hw:
                    st.hw = l
                trail = st.hw + trail_p
                if h >= trail:
                    pnl_pips = (st.entry_px - c) / pip - sp   # bar-close fill matches live
                    new_rec  = self._close(pnl_pips, c, sp, ts, "trail")

        # ── ENTRY ─────────────────────────────────────────────────────────────
        if st.pos == 0 and tr_price >= cfg.tr_thresh * pip and sp <= cfg.sp_gate:
            direction  = 1 if c >= o else -1
            st.pos     = direction
            st.entry_px = c
            st.hw      = h if direction == 1 else l
            self.entry_bar   = self.bar_count
            self.entry_time  = ts
            self.tr_at_entry = tr_price / pip

        st.prev_close = c
        self.bar_count += 1
        if st.pos != 0:
            self.bars_in_trade += 1

        return new_rec

    def _close(self, pnl_pips: float, exit_px: float, sp: float,
                ts: str, reason: str) -> TradeRecord:
        st  = self.st
        cfg = self.cfg
        rec = TradeRecord(
            label       = cfg.label,
            pair        = cfg.pair,
            direction   = st.pos,
            entry_time  = self.entry_time,
            exit_time   = ts,
            entry_px    = round(st.entry_px, 6),
            exit_px     = round(exit_px, 6),
            pnl_pips    = round(pnl_pips, 3),
            spread_pips = round(sp, 3),
            n_bars      = self.bar_count - self.entry_bar,
            exit_reason = reason,
            tr_at_entry = round(self.tr_at_entry, 2),
        )
        self.records.append(rec)
        self.total_pips += pnl_pips
        self.n_trades   += 1
        if pnl_pips > 0:
            self.n_wins += 1
        st.pos      = 0
        st.entry_px = 0.0
        st.hw       = 0.0
        write_trade_direct(
            strategy   = "tr_paper",
            pair       = cfg.pair,
            account_id = f"paper_{cfg.label}",
            trade_id   = f"paper_{cfg.label}_{self.entry_time}",
            direction  = rec.direction,
            entry_price = rec.entry_px,
            exit_price  = rec.exit_px,
            entry_time  = self.entry_time,
            exit_time   = ts,
            pnl_pips    = pnl_pips,
            exit_reason = reason,
            hours_held  = rec.n_bars * 5.0 / 60.0,
            units=1, mfe_pips=0.0, mae_pips=0.0, capture_ratio=0.0,
            is_paper=True, label=cfg.label,
        )
        return rec

    def status_dict(self) -> dict:
        st   = self.st
        cfg  = self.cfg
        wr   = self.n_wins / self.n_trades if self.n_trades > 0 else 0.0
        days = self.bar_count / 288
        pd   = self.total_pips / days if days > 0 else 0.0
        return {
            "label":       cfg.label,
            "pair":        cfg.pair,
            "tr_thresh":   cfg.tr_thresh,
            "trail_pips":  cfg.trail_pips,
            "oos_pd_ref":  cfg.oos_pd,
            "bar_count":   self.bar_count,
            "days_elapsed": round(days, 1),
            "n_trades":    self.n_trades,
            "n_wins":      self.n_wins,
            "win_rate":    round(wr * 100, 1),
            "total_pips":  round(self.total_pips, 2),
            "pips_per_day": round(pd, 2),
            "pos":         st.pos,
            "entry_px":    round(st.entry_px, 6) if st.pos != 0 else None,
            "hw":          round(st.hw, 6) if st.pos != 0 else None,
        }


# ── OANDA bar fetcher ─────────────────────────────────────────────────────────
_FALLBACK_SPREAD = {
    "USD_JPY": 1.5, "EUR_JPY": 2.3, "GBP_JPY": 3.5,
    "AUD_JPY": 2.5, "CAD_JPY": 2.5, "EUR_USD": 1.0,
    "GBP_USD": 1.8, "AUD_USD": 1.2,
}


class BarFetcher:
    def __init__(self, pair: str, adapter: OANDAAdapter):
        self.pair    = pair
        self.adapter = adapter
        self.last_ts: Optional[str] = None

    def fetch_new(self):
        """Returns (bar_dict, spread_pips, timestamp) or (None, 0, None) if no new bar."""
        candles = self.adapter.get_candles(self.pair, count=3, granularity="M5")
        if not candles:
            return None, 0.0, None
        bar = candles[-1]
        ts  = bar['timestamp']
        if ts == self.last_ts:
            return None, 0.0, None
        self.last_ts = ts
        pip = 0.01 if 'JPY' in self.pair else 0.0001
        try:
            sp = max(0.0, (float(bar['ask_c']) - float(bar['bid_c'])) / pip)
            if sp <= 0 or sp > 30:
                raise ValueError
        except Exception:
            sp = _FALLBACK_SPREAD.get(self.pair, 2.0)
        return bar, sp, ts


# ── Telegram ──────────────────────────────────────────────────────────────────
import requests as _requests

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def _tg(msg: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=8,
        )
    except Exception:
        pass


# ── Status writer ─────────────────────────────────────────────────────────────
def write_status(sims: List[TRPaperSim], path: str) -> None:
    data = {
        "updated":    datetime.now(timezone.utc).isoformat(),
        "strategies": [s.status_dict() for s in sims],
        "totals": {
            "n_trades":   sum(s.n_trades for s in sims),
            "total_pips": round(sum(s.total_pips for s in sims), 2),
        },
    }
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def append_trade(rec: TradeRecord, path: str) -> None:
    with open(path, 'a') as f:
        f.write(json.dumps(rec.__dict__) + "\n")


# ── Main loop ─────────────────────────────────────────────────────────────────
async def run_paper(shutdown_event: asyncio.Event) -> None:
    acct_id  = os.environ.get("OANDA_ACCOUNT_TR_PAPER",
               os.environ.get("OANDA_ACCOUNT_ID_011", ""))
    adapter  = OANDAAdapter(account_id=acct_id)
    fetchers = {pair: BarFetcher(pair, adapter) for pair in PAIRS_NEEDED}
    sims     = [TRPaperSim(cfg) for cfg in CONFIGS]
    pair_sims: Dict[str, List[TRPaperSim]] = {p: [] for p in PAIRS_NEEDED}
    for s in sims:
        pair_sims[s.cfg.pair].append(s)

    status_path = os.path.join(STATE_DIR, "tr_paper_status.json")
    trades_path = os.path.join(STATE_DIR, "tr_paper_trades.jsonl")

    logger.info(f"TR Momentum paper trading started — {len(sims)} configs on {PAIRS_NEEDED}")
    for s in sims:
        logger.info(f"  [{s.cfg.label:12s}] TR≥{s.cfg.tr_thresh:.0f}p trail={s.cfg.trail_pips:.0f}p "
                    f"sp_gate={s.cfg.sp_gate}p  OOS_ref={s.cfg.oos_pd}p/d")

    loop = asyncio.get_event_loop()

    while not shutdown_event.is_set():
        any_new = False
        for pair, fetcher in fetchers.items():
            try:
                bar, sp, ts = await loop.run_in_executor(None, fetcher.fetch_new)
                if bar is None:
                    continue
                any_new = True
                logger.debug(f"[{pair}] bar {ts} cl={bar['close']} sp={sp:.2f}p")

                for sim in pair_sims[pair]:
                    rec = sim.process_bar(bar, sp, ts)
                    if rec is not None:
                        append_trade(rec, trades_path)
                        dir_str = "LONG" if rec.direction == 1 else "SHORT"
                        emoji   = "🟢" if rec.pnl_pips > 0 else "🔴"
                        logger.info(
                            f"[{rec.label}] {dir_str} {rec.exit_reason}: "
                            f"{rec.pnl_pips:+.2f}p ({rec.n_bars}bar "
                            f"TR={rec.tr_at_entry:.1f}p) "
                            f"running={sim.total_pips:+.1f}p "
                            f"wr={sim.n_wins/max(sim.n_trades,1)*100:.0f}%"
                        )
                        _tg(
                            f"{emoji} [TR PAPER] {rec.label}\n"
                            f"{dir_str} {rec.exit_reason}: {rec.pnl_pips:+.2f}p "
                            f"({rec.n_bars}bar  TR_entry={rec.tr_at_entry:.1f}p)\n"
                            f"Running: {sim.total_pips:+.1f}p "
                            f"({sim.n_wins}/{sim.n_trades} = "
                            f"{sim.n_wins/max(sim.n_trades,1)*100:.0f}% WR)"
                        )

            except Exception as e:
                logger.error(f"[{pair}] error: {e}", exc_info=True)

        if any_new:
            try:
                write_status(sims, status_path)
            except Exception as e:
                logger.warning(f"Status write failed: {e}")

            # Hourly summary (~12 bars)
            if sims[0].bar_count > 1 and sims[0].bar_count % 12 == 0:
                logger.info("─── TR Paper Summary ─────────────────────────────")
                for s in sims:
                    st = s.st
                    pos_str = ""
                    if st.pos != 0:
                        pos_str = (f" | {'L' if st.pos==1 else 'S'} "
                                   f"@ {st.entry_px:.5f} hw={st.hw:.5f}")
                    logger.info(
                        f"  [{s.cfg.label:12s}] trades={s.n_trades:4d} "
                        f"wr={s.n_wins/max(s.n_trades,1)*100:.0f}% "
                        f"pips={s.total_pips:+8.1f} "
                        f"({s.total_pips/max(s.bar_count/288,0.01):.1f}p/d){pos_str}"
                    )

            # Daily Telegram summary (~288 bars = 1 trading day)
            if sims[0].bar_count > 1 and sims[0].bar_count % 288 == 0:
                days  = sims[0].bar_count / 288
                lines = [f"📊 TR Paper Day {days:.0f} summary:"]
                for s in sims:
                    pd  = s.total_pips / days if days > 0 else 0.0
                    wr  = s.n_wins / max(s.n_trades, 1) * 100
                    ref = s.cfg.oos_pd
                    flag = "🟢" if pd >= ref * 0.5 else ("🟡" if pd >= 0 else "🔴")
                    lines.append(
                        f"{flag} {s.cfg.label}: {s.total_pips:+.0f}p total "
                        f"({pd:.1f}p/d vs {ref}p/d ref) "
                        f"{s.n_trades}tr {wr:.0f}%WR"
                    )
                _tg("\n".join(lines))

        await asyncio.sleep(15)


async def main() -> None:
    shutdown = asyncio.Event()

    import signal as _signal
    def _handle_signal(sig, frame):
        logger.info(f"Signal {sig} received — shutting down")
        shutdown.set()
    _signal.signal(_signal.SIGTERM, _handle_signal)
    _signal.signal(_signal.SIGINT,  _handle_signal)

    await run_paper(shutdown)


if __name__ == "__main__":
    asyncio.run(main())
