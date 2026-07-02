#!/usr/bin/env python3
"""
FIFO-Trends Paper Trading Service — S5 exit monitoring for accurate simulation.

FILL MODEL (matching S5 live design):
  Entry:      M5 bar close (market order at M5 boundary)
  Trail exit: S5 bar close (≤5s detection → fill ≈ trail ± 0.3p) [exit_reason="trail_s5"]
  X7 exit:    M5 bar close (manual col_count check)              [exit_reason="x7"]

Paper fills now match what the redesigned live service achieves with S5 monitoring.
Old M5 trail exits (exit_reason="trail") no longer generated — all trail exits
are caught by S5 monitoring first.

Configs running (Session 068 proper-sim validated):
  GBP_JPY_ft  : b5_r1_n4_E2_X3c_1_5  OOS_S5=135.3 p/d  sp_gate=4.0p
  USD_JPY_ft  : b5_r1_n3_E2_X3c_1_5  OOS_S5=138.9 p/d  sp_gate=2.1p
  EUR_JPY_ft  : b5_r1_n3_E2_X3c_1_5  OOS_S5= 73.7 p/d  sp_gate=2.5p
  GBP_USD_ft  : b5_r1_n3_E2_X3b_1    OOS_S5= 53.9 p/d  sp_gate=2.4p  (d=1)
  GBP_USD_b2  : b2_r3_n8_E2_X3c_1_5  OOS_S5= 53.9 p/d  sp_gate=2.4p
  GBP_JPY_ft2 : b5_r1_n4_E2_X3c_2_5  OOS_S5= 33.9 p/d  sp_gate=4.0p
  USD_JPY_ft2 : b5_r1_n3_E2_X3c_2_5  OOS_S5= 31.5 p/d  sp_gate=2.1p

SOP compliance:
  R1–R7  Identical to previous version (P&F logic unchanged)
  S5 monitoring only changes WHEN we submit the exit order, not the signal logic.
"""

import os
import sys
import json
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

from lib.oanda_adapter import OANDAAdapter
from lib.db import write_trade_direct
from lib.pnf_engine import (
    PnFState, PnFConfig, BarResult,
    process_bar as _pnf_process_bar,
    MAX_COL_HIST as _MAX_COL_HIST,
)

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [fifo_paper] %(message)s",
)
logger = logging.getLogger("fifo_paper")

STATE_DIR    = os.environ.get("FIFO_STATE_DIR", "/tmp")
MAX_COL_HIST = _MAX_COL_HIST

_FALLBACK_SPREAD = {
    "GBP_JPY": 3.5, "USD_JPY": 1.5, "EUR_JPY": 2.3, "GBP_USD": 1.9,
}


# ── Config registry ────────────────────────────────────────────────────────────
@dataclass
class FIFOConfig:
    label:    str
    pair:     str
    pip:      float
    box_pips: int
    rev:      int
    n_min:    int
    trail_d:  int
    x7_k:     int
    sp_gate:  float
    oos_pd:   float   # S5-monitor OOS reference
    note:     str = ""

CONFIGS = [
    FIFOConfig("GBP_JPY_ft",  "GBP_JPY", 0.01,   5, 1, 4, 1, 5, 4.0, 135.3, "S5mon d=1"),
    FIFOConfig("USD_JPY_ft",  "USD_JPY", 0.01,   5, 1, 3, 1, 5, 2.1, 138.9, "S5mon d=1"),
    FIFOConfig("EUR_JPY_ft",  "EUR_JPY", 0.01,   5, 1, 3, 1, 5, 2.5,  73.7, "S5mon d=1"),
    FIFOConfig("GBP_USD_ft",  "GBP_USD", 0.0001, 5, 1, 3, 1, 0, 2.4,  53.9, "S5mon d=1"),
    FIFOConfig("GBP_USD_b2",  "GBP_USD", 0.0001, 2, 3, 8, 1, 5, 2.4,  53.9, "small-box"),
    FIFOConfig("GBP_JPY_ft2", "GBP_JPY", 0.01,   5, 1, 4, 2, 5, 4.0,  33.9, "S5mon d=2"),
    FIFOConfig("USD_JPY_ft2", "USD_JPY", 0.01,   5, 1, 3, 2, 5, 2.1,  31.5, "S5mon d=2"),
]

PAIRS_NEEDED = list(dict.fromkeys(c.pair for c in CONFIGS))


# ── Trade record ───────────────────────────────────────────────────────────────
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


# ── Per-config simulator ───────────────────────────────────────────────────────
class FIFOPaperSim:
    """
    Simulates one FIFO-Trends config on live M5 bars with S5 trail exit monitoring.

    M5 process_bar(): builds P&F chart, detects entries and X7 exits.
    check_s5_trail_exit(): called from S5 loop — detects trail condition at S5
    resolution and records simulated fill at S5 bar close (≈ trail level).
    """

    def __init__(self, cfg: FIFOConfig):
        self.cfg            = cfg
        self.st             = PnFState()
        self.records: List[TradeRecord] = []
        self.total_pips     = 0.0
        self.n_trades       = 0
        self.n_wins         = 0
        self.bar_count      = 0
        self.bars_in_trade  = 0
        self.entry_bar      = 0
        self.entry_time     = ""
        self._started       = datetime.now(timezone.utc).isoformat()

    # ── S5 trail exit ──────────────────────────────────────────────────────────
    def check_s5_trail_exit(self, s5_bar: dict, s5_ts: str) -> bool:
        """
        Check S5 bar for trail condition.
        Records fill at S5 close (≈ trail ± 0.3p — S5 bars are 0.1–0.5 pip wide).
        Returns True if position closed.
        """
        if self.st.pos == 0:
            return False

        cfg = self.cfg; st = self.st; pip = cfg.pip
        bs    = cfg.box_pips * pip
        trail = (st.hw_level - cfg.trail_d * bs) if st.pos == 1 \
                else (st.hw_level + cfg.trail_d * bs)

        lo = float(s5_bar['low']); hi = float(s5_bar['high']); cl = float(s5_bar['close'])
        try:
            sp = max(0.0, (float(s5_bar['ask_c']) - float(s5_bar['bid_c'])) / pip)
            if sp <= 0 or sp > 25:
                sp = _FALLBACK_SPREAD.get(cfg.pair, 2.0)
        except Exception:
            sp = _FALLBACK_SPREAD.get(cfg.pair, 2.0)

        triggered = (st.pos == 1 and lo <= trail) or (st.pos == -1 and hi >= trail)
        if not triggered:
            return False

        # Simulated fill at S5 bar close (≈ trail price for paper tracking purposes)
        exit_px     = cl
        pos_closed  = st.pos
        entry_px_r  = st.entry_px
        n_bars_held = self.bar_count - self.entry_bar
        pnl_pips    = (exit_px - entry_px_r) * pos_closed / pip - sp

        rec = TradeRecord(
            label       = cfg.label,
            pair        = cfg.pair,
            direction   = pos_closed,
            entry_time  = self.entry_time,
            exit_time   = s5_ts,
            entry_px    = round(entry_px_r, 6),
            exit_px     = round(exit_px, 6),
            pnl_pips    = round(pnl_pips, 3),
            spread_pips = round(sp, 3),
            n_bars      = n_bars_held,
            exit_reason = "trail_s5",
        )
        self.records.append(rec)
        write_trade_direct(
            strategy="fifo_paper", pair=cfg.pair,
            account_id=f"paper_{cfg.label}",
            trade_id=f"paper_{cfg.label}_{self.entry_time}",
            direction=pos_closed, entry_price=entry_px_r, exit_price=exit_px,
            entry_time=self.entry_time, exit_time=s5_ts, pnl_pips=pnl_pips,
            exit_reason="trail_s5", hours_held=n_bars_held * 5.0 / 60.0,
            units=1, mfe_pips=0.0, mae_pips=0.0, capture_ratio=0.0,
            is_paper=True, label=cfg.label,
        )
        self.total_pips += pnl_pips
        self.n_trades   += 1
        if pnl_pips > 0:
            self.n_wins += 1

        # Reset state — M5 process_bar() won't re-exit (pos=0 guard in engine)
        st.pos = 0; st.entry_px = 0.0; st.hw_level = 0.0
        return True

    # ── M5 bar processing (chart update + entry + X7 exit) ────────────────────
    def process_bar(self, bar: dict, sp: float, ts: str) -> None:
        cfg = self.cfg; st = self.st; pip = cfg.pip

        eng_cfg = PnFConfig(
            pip      = cfg.pip,
            box_pips = cfg.box_pips,
            rev      = cfg.rev,
            n_min    = cfg.n_min,
            trail_d  = cfg.trail_d,
            x7_k     = cfg.x7_k,
            sp_gate  = cfg.sp_gate,
        )

        sig = _pnf_process_bar(
            st, eng_cfg,
            float(bar['open']), float(bar['high']),
            float(bar['low']),  float(bar['close']),
            sp,
        )

        # Trail exits are handled by S5 loop. Anything here is X7 (col-SMA exit).
        if sig.exit_triggered:
            pnl_pips    = (sig.exit_px - sig.exited_entry_px) * sig.exited_pos / pip - sp
            n_bars_held = self.bar_count - self.entry_bar
            rec = TradeRecord(
                label       = cfg.label,
                pair        = cfg.pair,
                direction   = sig.exited_pos,
                entry_time  = self.entry_time,
                exit_time   = ts,
                entry_px    = round(sig.exited_entry_px, 6),
                exit_px     = round(sig.exit_px, 6),
                pnl_pips    = round(pnl_pips, 3),
                spread_pips = round(sp, 3),
                n_bars      = n_bars_held,
                exit_reason = sig.exit_reason,
            )
            self.records.append(rec)
            write_trade_direct(
                strategy="fifo_paper", pair=cfg.pair,
                account_id=f"paper_{cfg.label}",
                trade_id=f"paper_{cfg.label}_{self.entry_time}",
                direction=sig.exited_pos, entry_price=rec.entry_px, exit_price=rec.exit_px,
                entry_time=self.entry_time, exit_time=ts, pnl_pips=pnl_pips,
                exit_reason=sig.exit_reason, hours_held=n_bars_held * 5.0 / 60.0,
                units=1, mfe_pips=0.0, mae_pips=0.0, capture_ratio=0.0,
                is_paper=True, label=cfg.label,
            )
            self.total_pips += pnl_pips
            self.n_trades   += 1
            if pnl_pips > 0:
                self.n_wins += 1

        if sig.entry_signal != 0:
            self.entry_bar  = self.bar_count
            self.entry_time = ts

        self.bar_count += 1
        if st.pos != 0:
            self.bars_in_trade += 1

    def status_dict(self) -> dict:
        st  = self.st; cfg = self.cfg
        wr  = self.n_wins / self.n_trades if self.n_trades > 0 else 0.0
        avg = self.total_pips / self.n_trades if self.n_trades > 0 else 0.0
        days = self.bar_count / 288
        pd   = self.total_pips / days if days > 0 else 0.0
        bs   = cfg.box_pips * cfg.pip
        trail = (st.hw_level - cfg.trail_d * bs) if st.pos == 1 \
                else (st.hw_level + cfg.trail_d * bs) if st.pos == -1 else None
        return {
            "label":       cfg.label,
            "pair":        cfg.pair,
            "oos_pd_ref":  cfg.oos_pd,
            "note":        cfg.note,
            "exit_mode":   "S5_monitor",
            "bar_count":   self.bar_count,
            "days_elapsed": round(days, 1),
            "n_trades":    self.n_trades,
            "n_wins":      self.n_wins,
            "win_rate":    round(wr * 100, 1),
            "total_pips":  round(self.total_pips, 2),
            "pips_per_day": round(pd, 2),
            "avg_pips_per_trade": round(avg, 2),
            "pos":         st.pos,
            "entry_px":    round(st.entry_px, 6) if st.pos != 0 else None,
            "trail_level": round(trail, 6) if trail is not None else None,
            "hw_level":    round(st.hw_level, 6) if st.pos != 0 else None,
            "pnf_idx":     st.pnf_idx,
            "pnf_level":   round(st.pnf_level, 6),
            "pnf_dir":     st.pnf_dir,
            "col_count":   st.col_count,
        }


# ── M5 bar fetcher ─────────────────────────────────────────────────────────────
class M5BarFetcher:
    def __init__(self, pair: str, adapter: OANDAAdapter):
        self.pair    = pair
        self.adapter = adapter
        self.last_ts: Optional[str] = None

    def fetch_new(self):
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
            if sp <= 0 or sp > 25:
                raise ValueError
        except Exception:
            sp = _FALLBACK_SPREAD.get(self.pair, 2.0)
        return bar, sp, ts


# ── S5 bar fetcher ─────────────────────────────────────────────────────────────
class S5BarFetcher:
    """Fetches S5 bars for trail exit monitoring. One instance per pair."""

    def __init__(self, pair: str, adapter: OANDAAdapter):
        self.pair    = pair
        self.adapter = adapter
        self.last_ts: Optional[str] = None

    def fetch_new(self) -> list:
        """Returns new completed S5 bars since last_ts, sorted by timestamp."""
        bars = self.adapter.get_candles(self.pair, count=6, granularity="S5")
        if not bars:
            return []
        new_bars = sorted(
            [b for b in bars if b['timestamp'] > (self.last_ts or '')],
            key=lambda x: x['timestamp'],
        )
        return new_bars


# ── Telegram ───────────────────────────────────────────────────────────────────
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


# ── Status writer ──────────────────────────────────────────────────────────────
def write_status(sims: List[FIFOPaperSim], path: str) -> None:
    data = {
        "updated":    datetime.now(timezone.utc).isoformat(),
        "exit_mode":  "S5_monitor",
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


# ── Main asyncio loop ──────────────────────────────────────────────────────────
async def run_paper(shutdown_event: asyncio.Event) -> None:
    acct_id = os.environ.get("OANDA_ACCOUNT_FIFO_PAPER",
              os.environ.get("OANDA_ACCOUNT_ID_011", ""))
    adapter    = OANDAAdapter(account_id=acct_id)
    m5_fetchers = {pair: M5BarFetcher(pair, adapter) for pair in PAIRS_NEEDED}
    s5_fetchers = {pair: S5BarFetcher(pair, adapter) for pair in PAIRS_NEEDED}
    sims        = [FIFOPaperSim(cfg) for cfg in CONFIGS]
    pair_sims: Dict[str, List[FIFOPaperSim]] = {p: [] for p in PAIRS_NEEDED}
    for s in sims:
        pair_sims[s.cfg.pair].append(s)

    status_path = os.path.join(STATE_DIR, "fifo_paper_status.json")
    trades_path = os.path.join(STATE_DIR, "fifo_paper_trades.jsonl")

    logger.info(
        f"FIFO-Trends paper started — {len(sims)} configs, exit_mode=S5_monitor, "
        f"poll=5s  pairs={PAIRS_NEEDED}"
    )
    for s in sims:
        logger.info(
            f"  [{s.cfg.label:12s}] b{s.cfg.box_pips}_r{s.cfg.rev}_n{s.cfg.n_min}_"
            f"E2_X3{'c_' + str(s.cfg.trail_d) + '_' + str(s.cfg.x7_k) if s.cfg.x7_k else 'b_' + str(s.cfg.trail_d)}"
            f"  sp_gate={s.cfg.sp_gate}p  OOS_S5={s.cfg.oos_pd}p/d ({s.cfg.note})"
        )

    loop = asyncio.get_event_loop()

    while not shutdown_event.is_set():
        try:
            # ── S5 trail exit monitoring (pairs with active positions) ─────────
            for pair in PAIRS_NEEDED:
                active = [s for s in pair_sims[pair] if s.st.pos != 0]
                if not active:
                    continue
                try:
                    new_s5 = await loop.run_in_executor(None, s5_fetchers[pair].fetch_new)
                    for s5_bar in new_s5:
                        for sim in active:
                            if sim.st.pos == 0:
                                continue
                            if sim.check_s5_trail_exit(s5_bar, s5_bar['timestamp']):
                                rec = sim.records[-1]
                                dir_str = "LONG" if rec.direction == 1 else "SHORT"
                                emoji   = "🟢" if rec.pnl_pips > 0 else "🔴"
                                log_line = (
                                    f"[{rec.label}] {dir_str} trail_s5: "
                                    f"{rec.pnl_pips:+.2f}p ({rec.n_bars}bar) "
                                    f"running={sim.total_pips:+.1f}p "
                                    f"wr={sim.n_wins/max(sim.n_trades,1)*100:.0f}%"
                                )
                                logger.info(log_line)
                                with open(trades_path, 'a') as f:
                                    f.write(json.dumps(rec.__dict__) + "\n")
                                _tg(
                                    f"{emoji} [FIFO PAPER] {rec.label}\n"
                                    f"{dir_str} trail_s5: {rec.pnl_pips:+.2f}p "
                                    f"({rec.n_bars}bar)\n"
                                    f"Running: {sim.total_pips:+.1f}p "
                                    f"({sim.n_wins}/{sim.n_trades} = "
                                    f"{sim.n_wins/max(sim.n_trades,1)*100:.0f}% WR)"
                                )
                        s5_fetchers[pair].last_ts = s5_bar['timestamp']
                except Exception as e:
                    logger.warning(f"[{pair}] S5 fetch error: {e}")

            # ── M5 chart update + entry + X7 exit ─────────────────────────────
            any_new = False
            for pair, fetcher in m5_fetchers.items():
                try:
                    bar, sp, ts = await loop.run_in_executor(None, fetcher.fetch_new)
                    if bar is None:
                        continue
                    any_new = True
                    logger.debug(f"[{pair}] M5 bar {ts} cl={bar['close']} sp={sp:.2f}p")

                    prev_counts = {s.cfg.label: s.n_trades for s in pair_sims[pair]}
                    prev_pos    = {s.cfg.label: s.st.pos  for s in pair_sims[pair]}
                    for sim in pair_sims[pair]:
                        sim.process_bar(bar, sp, ts)
                    # If any sim just entered a new position, anchor S5 last_ts to NOW
                    # so pre-entry S5 bars (which may already violate the trail) are skipped.
                    any_new_entry = any(
                        prev_pos[s.cfg.label] == 0 and s.st.pos != 0
                        for s in pair_sims[pair]
                    )
                    if any_new_entry:
                        s5_fetchers[pair].last_ts = datetime.now(timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%S.000000000Z"
                        )

                    for sim in pair_sims[pair]:
                        new_n = sim.n_trades
                        prev_n = prev_counts[sim.cfg.label]
                        if new_n > prev_n:
                            for rec in sim.records[-(new_n - prev_n):]:
                                with open(trades_path, 'a') as f:
                                    f.write(json.dumps(rec.__dict__) + "\n")
                                dir_str = "LONG" if rec.direction == 1 else "SHORT"
                                emoji   = "🟢" if rec.pnl_pips > 0 else "🔴"
                                logger.info(
                                    f"[{rec.label}] {dir_str} {rec.exit_reason}: "
                                    f"{rec.pnl_pips:+.2f}p ({rec.n_bars}bar) "
                                    f"running={sim.total_pips:+.1f}p "
                                    f"wr={sim.n_wins/max(sim.n_trades,1)*100:.0f}%"
                                )
                                _tg(
                                    f"{emoji} [FIFO PAPER] {rec.label}\n"
                                    f"{dir_str} {rec.exit_reason}: {rec.pnl_pips:+.2f}p "
                                    f"({rec.n_bars}bar)\n"
                                    f"Running: {sim.total_pips:+.1f}p "
                                    f"({sim.n_wins}/{sim.n_trades} = "
                                    f"{sim.n_wins/max(sim.n_trades,1)*100:.0f}% WR)"
                                )

                except Exception as e:
                    logger.error(f"[{pair}] M5 error: {e}", exc_info=True)

            if any_new:
                try:
                    write_status(sims, status_path)
                except Exception as e:
                    logger.warning(f"Status write failed: {e}")

                if sims[0].bar_count > 0 and sims[0].bar_count % 12 == 0:
                    logger.info("─── FIFO Paper Summary ───────────────────────────")
                    for s in sims:
                        st = s.st
                        pos_str = ""
                        if st.pos != 0:
                            bs = s.cfg.box_pips * s.cfg.pip
                            trail = (st.hw_level - s.cfg.trail_d * bs) if st.pos == 1 \
                                    else (st.hw_level + s.cfg.trail_d * bs)
                            pos_str = (f" | {'L' if st.pos==1 else 'S'} "
                                       f"@ {st.entry_px:.5f} trail={trail:.5f}")
                        logger.info(
                            f"  [{s.cfg.label:12s}] trades={s.n_trades:4d} "
                            f"wr={s.n_wins/max(s.n_trades,1)*100:.0f}% "
                            f"pips={s.total_pips:+8.1f} "
                            f"({s.total_pips/max(s.bar_count/288,0.01):.1f}p/d){pos_str}"
                        )

                if sims[0].bar_count > 0 and sims[0].bar_count % 288 == 0:
                    days = sims[0].bar_count / 288
                    lines = [f"📊 FIFO Paper S5-monitor Day {days:.0f} summary:"]
                    for s in sims:
                        pd   = s.total_pips / days if days > 0 else 0.0
                        wr   = s.n_wins / max(s.n_trades, 1) * 100
                        ref  = s.cfg.oos_pd
                        flag = "🟢" if pd >= ref * 0.5 else ("🟡" if pd >= 0 else "🔴")
                        lines.append(
                            f"{flag} {s.cfg.label}: {s.total_pips:+.0f}p "
                            f"({pd:.1f}p/d vs {ref}p/d ref) "
                            f"{s.n_trades}tr {wr:.0f}%WR"
                        )
                    _tg("\n".join(lines))

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)

        await asyncio.sleep(5)   # 5s: matches S5 bar granularity (was 15s)


async def main() -> None:
    shutdown = asyncio.Event()

    import signal as _signal
    def _handle(sig, frame):
        logger.info(f"Signal {sig} — shutting down")
        shutdown.set()
    _signal.signal(_signal.SIGTERM, _handle)
    _signal.signal(_signal.SIGINT, _handle)

    await run_paper(shutdown)


if __name__ == "__main__":
    asyncio.run(main())
