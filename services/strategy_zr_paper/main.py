#!/usr/bin/env python3
"""
ZR Paper Trading Service — all validated ZR strategies on live OANDA data.
No actual orders placed. Fills simulated at bar close / stop/target price.

Runs all strategies that passed IS+OOS+MC validation gates:
  GBP_ABS_125 : GBP_USD body=0.5 PF=1.25 ta=6  B=2  [rank 1, best risk_adj_p5]
  GBP_ABS_110 : GBP_USD body=0.5 PF=1.10 ta=6  B=2  [rank 2, LIVE on 011/012]
  EUR_JPY_125 : EUR_JPY body=0.0 PF=1.25 ta=6  B=1  [rank 3]
  EUR_USD_125 : EUR_USD body=0.0 PF=1.25 ta=10 B=8  [rank 4]
  EUR_USD_110 : EUR_USD body=0.0 PF=1.10 ta=10 B=8  [rank 5]
  EUR_USD_105 : EUR_USD body=0.0 PF=1.05 ta=10 B=8  [rank 6]

Data: polls OANDA M5 bars using account 011 credentials (read-only).
Output: /data/logs/zr_paper_status.json (refreshed every bar)
        /data/logs/zr_paper_trades.jsonl (one line per closed cycle)
"""

import os
import sys
import asyncio
import math
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

from lib.oanda_adapter import OANDAAdapter
from lib.db import write_trade_direct

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [zr_paper] %(message)s",
)
logger = logging.getLogger("zr_paper")

STATE_DIR = os.environ.get("ZR_STATE_DIR", "/tmp")

# ── Validated strategy configs ────────────────────────────────────────────────
@dataclass
class ZRConfig:
    label: str
    pair: str
    pip: float
    pip_usd: float
    zw: float
    tgt: float
    ta: float
    td: float
    pf: float
    body_thresh: float
    base_units: int
    rank: int
    note: str = ""

CONFIGS = [
    ZRConfig("GBP_ABS_125", "GBP_USD", 0.0001, 0.0001, 30.0, 21.0, 6.0,  1.0, 1.25, 0.5, 2, 1, "rank1 best radp5"),
    ZRConfig("GBP_ABS_110", "GBP_USD", 0.0001, 0.0001, 30.0, 21.0, 6.0,  1.0, 1.10, 0.5, 2, 2, "rank2 LIVE 011/012"),
    ZRConfig("EUR_JPY_125", "EUR_JPY", 0.01,   0.000069, 50.0, 35.0, 6.0, 1.0, 1.25, 0.0, 1, 3, "rank3"),
    ZRConfig("EUR_USD_125", "EUR_USD", 0.0001, 0.0001, 30.0, 21.0, 10.0, 1.0, 1.25, 0.0, 8, 4, "rank4"),
    ZRConfig("EUR_USD_110", "EUR_USD", 0.0001, 0.0001, 30.0, 21.0, 10.0, 1.0, 1.10, 0.0, 8, 5, "rank5"),
    ZRConfig("EUR_USD_105", "EUR_USD", 0.0001, 0.0001, 30.0, 21.0, 10.0, 1.0, 1.05, 0.0, 8, 6, "rank6"),
]

PAIRS_NEEDED = list(dict.fromkeys(c.pair for c in CONFIGS))  # unique pairs, insertion order

MAX_LEGS   = 10
PSAR_AF0   = 0.01
PSAR_STEP  = 0.01
PSAR_AFMAX = 0.20
ATD_THRESH = 3.0   # adaptive td: ATR5/ATR20 ratio threshold
ATD_FRAC   = 0.75
ATD_CAP    = 5.0


# ── Pure ZR simulation state (no broker) ─────────────────────────────────────
@dataclass
class SimCycle:
    entry_price: float
    direction: int
    upper_zone: float
    lower_zone: float
    upper_target: float
    lower_target: float
    tgt_pips: float
    zone_width: float
    entry_time: str
    legs: List[dict] = field(default_factory=list)  # {vol, dir, price}
    long_count: int = 0
    short_count: int = 0
    last_zone_bar: str = ""
    last_zone_side: str = ""
    peak_mfe: float = 0.0
    trail_on: bool = False
    psar_on: bool = False
    psar_val: float = 0.0
    psar_ep:  float = 0.0
    psar_af:  float = PSAR_AF0
    psar_net_dir: int = 0


@dataclass
class SimRecord:
    label: str
    pair: str
    entry_time: str
    exit_time: str
    direction: int
    n_legs: int
    exit_type: str   # trail / psar / target / max_legs
    pnl_pips: float
    pnl_usd: float


def _net_at(legs, target, pip):
    return sum(l['vol'] * l['dir'] * (target - l['price']) / pip for l in legs)

def _spread_cost(legs, sp):
    return sum(l['vol'] for l in legs) * sp

def _hedge_vol(legs, target, tgt_pips, pip, sp, pf):
    net = _net_at(legs, target, pip) - _spread_cost(legs, sp)
    if net >= 0:
        return 0.0
    npu = tgt_pips - sp
    if npu <= 0:
        return float(MAX_LEGS)
    return max(1.0, float(math.ceil(-net / npu * pf)))


class ZRPaperSim:
    """
    Simulates one ZR config on M5 bars. Call process_bar() per closed bar.
    Fills are at bar close (entries) or stop/target price (exits).
    """

    def __init__(self, cfg: ZRConfig):
        self.cfg        = cfg
        self.cycle: Optional[SimCycle] = None
        self.next_dir   = 1
        self.records: List[SimRecord] = []
        self.total_pips = 0.0
        self.total_usd  = 0.0
        self.n_cycles   = 0
        self.n_skipped  = 0
        self.bar_count  = 0
        self.hl_pips: deque = deque(maxlen=20)   # for adaptive td ATR
        self._started = datetime.now(timezone.utc).isoformat()

    # ── Adaptive td ──────────────────────────────────────────────────────────
    def _effective_td(self, bar: dict) -> float:
        hi = float(bar['high']); lo = float(bar['low'])
        p = self.cfg.pip
        self.hl_pips.append((hi - lo) / p)
        td = self.cfg.td
        if len(self.hl_pips) >= 20:
            buf = list(self.hl_pips)
            atr5  = sum(buf[-5:]) / 5
            atr20 = sum(buf) / 20
            if atr20 > 0 and atr5 / atr20 >= ATD_THRESH:
                td = min(atr5 * ATD_FRAC, ATD_CAP)
        return td

    # ── Absorption filter ────────────────────────────────────────────────────
    def _body_ok(self, bar: dict, direction: int) -> bool:
        thresh = self.cfg.body_thresh
        if thresh <= 0.0:
            return True
        hi = float(bar['high']); lo = float(bar['low'])
        op = float(bar['open']); cl = float(bar['close'])
        rng = hi - lo
        if rng < 1e-10:
            return True
        if direction == 1:
            adv = max(0.0, op - cl) / rng
        else:
            adv = max(0.0, cl - op) / rng
        return adv <= thresh

    # ── Cycle helpers ────────────────────────────────────────────────────────
    def _open_cycle(self, bar: dict, direction: int, sp: float) -> None:
        cfg = self.cfg
        p = cfg.pip
        e = float(bar['close'])
        if direction == 1:
            uz = e; lz = e - cfg.zw * p
            ut = e + cfg.tgt * p; lt = lz - cfg.tgt * p
        else:
            lz = e; uz = e + cfg.zw * p
            lt = e - cfg.tgt * p; ut = uz + cfg.tgt * p
        self.cycle = SimCycle(
            entry_price=e, direction=direction,
            upper_zone=uz, lower_zone=lz,
            upper_target=ut, lower_target=lt,
            tgt_pips=cfg.tgt, zone_width=cfg.zw,
            entry_time=bar['timestamp'],
            legs=[{'vol': 1.0, 'dir': float(direction), 'price': e}],
        )
        if direction == 1:
            self.cycle.long_count = 1
        else:
            self.cycle.short_count = 1

    def _record_exit(self, bar: dict, exit_price: float,
                     exit_type: str, sp: float) -> None:
        c = self.cycle
        cfg = self.cfg
        p = cfg.pip
        net_pips = _net_at(c.legs, exit_price, p) - _spread_cost(c.legs, sp)
        net_usd = net_pips * cfg.base_units * cfg.pip_usd
        self.total_pips += net_pips
        self.total_usd  += net_usd
        self.n_cycles   += 1
        rec = SimRecord(
            label=cfg.label, pair=cfg.pair,
            entry_time=c.entry_time,
            exit_time=bar['timestamp'],
            direction=c.direction,
            n_legs=len(c.legs),
            exit_type=exit_type,
            pnl_pips=round(net_pips, 2),
            pnl_usd=round(net_usd, 6),
        )
        self.records.append(rec)
        logger.info(f"[{cfg.label}] EXIT {exit_type} legs={len(c.legs)} "
                    f"P&L={net_pips:+.1f}p ${net_usd:+.4f} | "
                    f"cum_pips={self.total_pips:+.1f} "
                    f"cum_usd=${self.total_usd:+.4f}")
        entry_dt = datetime.fromisoformat(c.entry_time.replace('Z', '+00:00'))
        exit_dt  = datetime.fromisoformat(bar['timestamp'].replace('Z', '+00:00'))
        write_trade_direct(
            strategy="zr_paper",
            pair=cfg.pair,
            account_id=f"paper_{cfg.label}",
            trade_id=f"paper_{cfg.label}_{c.entry_time}",
            direction=c.direction,
            entry_price=c.entry_price,
            exit_price=exit_price,
            entry_time=c.entry_time,
            exit_time=bar['timestamp'],
            pnl_pips=net_pips,
            exit_reason=exit_type,
            hours_held=(exit_dt - entry_dt).total_seconds() / 3600,
            units=len(c.legs),
            mfe_pips=c.peak_mfe,
            mae_pips=0.0,
            capture_ratio=round(net_pips / c.peak_mfe, 2) if c.peak_mfe > 0 else 0.0,
            is_paper=True,
            label=cfg.label,
        )
        self.cycle = None
        self.next_dir = -c.direction

    # ── Main bar processor ────────────────────────────────────────────────────
    def process_bar(self, bar: dict, sp: float) -> None:
        """
        sp: live spread in pips for this bar.
        bar: dict with 'open','high','low','close','timestamp' keys.
        """
        self.bar_count += 1
        cfg = self.cfg
        p = cfg.pip
        hi = float(bar['high']); lo = float(bar['low'])
        cl = float(bar['close']); op = float(bar['open'])
        ts = bar['timestamp']
        td_eff = self._effective_td(bar)

        # ── No active cycle: check for entry ─────────────────────────────────
        if self.cycle is None:
            d = self.next_dir
            if self._body_ok(bar, d):
                self._open_cycle(bar, d, sp)
            else:
                self.n_skipped += 1
            return

        c = self.cycle
        bull = (cl >= op)

        # ── PSAR escape trail ─────────────────────────────────────────────────
        if c.psar_on:
            if c.psar_net_dir > 0:
                if hi > c.psar_ep:
                    c.psar_ep = hi
                    c.psar_af = min(c.psar_af + PSAR_STEP, PSAR_AFMAX)
                c.psar_val = c.psar_ep - (c.psar_ep - c.psar_val) * c.psar_af
                if lo <= c.psar_val:
                    self._record_exit(bar, c.psar_val, "psar", sp)
                    return
            else:
                if lo < c.psar_ep:
                    c.psar_ep = lo
                    c.psar_af = min(c.psar_af + PSAR_STEP, PSAR_AFMAX)
                c.psar_val = c.psar_ep + (c.psar_val - c.psar_ep) * c.psar_af
                if hi >= c.psar_val:
                    self._record_exit(bar, c.psar_val, "psar", sp)
                    return
            return  # PSAR active — skip trail + zone checks

        # ── 1-leg trailing stop ───────────────────────────────────────────────
        if len(c.legs) == 1:
            mfe = (hi - c.entry_price) / p if c.direction == 1 else (c.entry_price - lo) / p
            if mfe > c.peak_mfe:
                c.peak_mfe = mfe
            if c.peak_mfe >= cfg.ta:
                c.trail_on = True
            if c.trail_on:
                if c.direction == 1:
                    be = c.entry_price + sp * p
                    ts_price = max(be, c.entry_price + (c.peak_mfe - td_eff) * p)
                    if lo <= ts_price:
                        self._record_exit(bar, ts_price, "trail", sp)
                        return
                else:
                    be = c.entry_price - sp * p
                    ts_price = min(be, c.entry_price - (c.peak_mfe - td_eff) * p)
                    if hi >= ts_price:
                        self._record_exit(bar, ts_price, "trail", sp)
                        return

        # ── Zone crossings + escape targets ───────────────────────────────────
        # Process high before low for bullish bars (same logic as backtest)
        sequence = [(hi, True), (lo, False)] if bull else [(lo, False), (hi, True)]
        exited = False

        for extreme, is_hi in sequence:
            if exited:
                break

            if is_hi and hi >= c.upper_zone and c.last_zone_side != "upper":
                c.last_zone_side = "upper"; c.last_zone_bar = ts
                net_vol = sum(leg['vol'] * leg['dir'] for leg in c.legs)
                vol = _hedge_vol(c.legs, c.upper_target, cfg.tgt, p, sp, cfg.pf)
                if vol > 0 and net_vol < 0:  # net short → long recovery valid
                    if len(c.legs) >= MAX_LEGS:
                        self._record_exit(bar, cl, "max_legs", sp)
                        exited = True; break
                    c.legs.append({'vol': vol, 'dir': 1.0, 'price': c.upper_zone})
                    c.long_count += 1

            if not is_hi and lo <= c.lower_zone and c.last_zone_side != "lower":
                c.last_zone_side = "lower"; c.last_zone_bar = ts
                net_vol = sum(leg['vol'] * leg['dir'] for leg in c.legs)
                vol = _hedge_vol(c.legs, c.lower_target, cfg.tgt, p, sp, cfg.pf)
                if vol > 0 and net_vol > 0:  # net long → short recovery valid
                    if len(c.legs) >= MAX_LEGS:
                        self._record_exit(bar, cl, "max_legs", sp)
                        exited = True; break
                    c.legs.append({'vol': vol, 'dir': -1.0, 'price': c.lower_zone})
                    c.short_count += 1

            if exited:
                break

            # Check if either escape target is touched → activate PSAR
            if lo <= c.upper_target <= hi:
                net_v = sum(l['vol'] * l['dir'] for l in c.legs)
                c.psar_net_dir = 1 if net_v >= 0 else -1
                c.psar_on = True; c.psar_af = PSAR_AF0
                c.psar_ep  = c.upper_target
                c.psar_val = (c.upper_target - cfg.tgt * p if c.psar_net_dir > 0
                              else c.upper_target + cfg.tgt * p)
                break
            if lo <= c.lower_target <= hi:
                net_v = sum(l['vol'] * l['dir'] for l in c.legs)
                c.psar_net_dir = 1 if net_v >= 0 else -1
                c.psar_on = True; c.psar_af = PSAR_AF0
                c.psar_ep  = c.lower_target
                c.psar_val = (c.lower_target - cfg.tgt * p if c.psar_net_dir > 0
                              else c.lower_target + cfg.tgt * p)
                break

    def status_dict(self) -> dict:
        cfg = self.cfg
        c = self.cycle
        return {
            "label": cfg.label,
            "pair": cfg.pair,
            "rank": cfg.rank,
            "note": cfg.note,
            "pf": cfg.pf,
            "body_thresh": cfg.body_thresh,
            "base_units": cfg.base_units,
            "n_cycles": self.n_cycles,
            "n_skipped": self.n_skipped,
            "total_pips": round(self.total_pips, 2),
            "total_usd": round(self.total_usd, 6),
            "bar_count": self.bar_count,
            "cycle_open": c is not None,
            "cycle_legs": len(c.legs) if c else 0,
            "cycle_dir": c.direction if c else 0,
            "cycle_mfe": round(c.peak_mfe, 1) if c else 0.0,
            "cycle_psar": c.psar_on if c else False,
        }


# ── OANDA bar fetcher ─────────────────────────────────────────────────────────
class BarFetcher:
    """Polls OANDA M5 bars for a single pair, caches last seen timestamp."""

    def __init__(self, pair: str, adapter: OANDAAdapter):
        self.pair    = pair
        self.adapter = adapter
        self.last_ts: Optional[str] = None

    _DEFAULT_SPREAD = {"GBP_USD": 1.9, "EUR_JPY": 2.3, "EUR_USD": 1.4,
                        "CHF_JPY": 3.5, "USD_JPY": 1.5}

    def fetch_new(self) -> tuple:
        """
        Returns (bar_dict, spread_pips) for the latest CLOSED M5 bar, or (None, 0).
        bar_dict keys: timestamp, open, high, low, close, bid_c, ask_c.
        get_candles price='MBA' already includes bid_c/ask_c.
        """
        candles = self.adapter.get_candles(self.pair, count=3, granularity="M5")
        if not candles:
            return None, 0.0
        bar = candles[-1]
        if bar['timestamp'] == self.last_ts:
            return None, 0.0
        self.last_ts = bar['timestamp']
        pip = 0.01 if 'JPY' in self.pair else 0.0001
        try:
            spread = max(0.0, (float(bar['ask_c']) - float(bar['bid_c'])) / pip)
            if spread <= 0 or spread > 20:
                raise ValueError
        except Exception:
            spread = self._DEFAULT_SPREAD.get(self.pair, 1.5)
        return bar, spread


# ── Status + trade log writers ────────────────────────────────────────────────
def write_status(sims: List[ZRPaperSim], path: str) -> None:
    data = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "strategies": [s.status_dict() for s in sims],
        "totals": {
            "n_cycles": sum(s.n_cycles for s in sims),
            "total_usd": round(sum(s.total_usd for s in sims), 4),
            "total_pips": round(sum(s.total_pips for s in sims), 2),
        }
    }
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def append_trade(rec: SimRecord, path: str) -> None:
    with open(path, 'a') as f:
        f.write(json.dumps(rec.__dict__) + "\n")


# ── Main asyncio loop ─────────────────────────────────────────────────────────
async def run_paper(shutdown_event: asyncio.Event) -> None:
    acct_id = os.environ.get("OANDA_ACCOUNT_LONG",
              os.environ.get("OANDA_ACCOUNT_ID_011", ""))
    adapter = OANDAAdapter(account_id=acct_id)

    fetchers = {pair: BarFetcher(pair, adapter) for pair in PAIRS_NEEDED}
    sims     = [ZRPaperSim(cfg) for cfg in CONFIGS]

    # Map each pair → list of sims that need it
    pair_sims: Dict[str, List[ZRPaperSim]] = {p: [] for p in PAIRS_NEEDED}
    for s in sims:
        pair_sims[s.cfg.pair].append(s)

    status_path = os.path.join(STATE_DIR, "zr_paper_status.json")
    trades_path = os.path.join(STATE_DIR, "zr_paper_trades.jsonl")

    logger.info(f"Paper trading started — {len(sims)} strategies on {PAIRS_NEEDED}")
    for s in sims:
        logger.info(f"  [{s.cfg.label:15s}] {s.cfg.pair} ZW={s.cfg.zw} "
                    f"TGT={s.cfg.tgt} PF={s.cfg.pf} body={s.cfg.body_thresh} "
                    f"ta={s.cfg.ta} B={s.cfg.base_units}  ({s.cfg.note})")

    loop = asyncio.get_event_loop()

    while not shutdown_event.is_set():
        any_new = False
        for pair, fetcher in fetchers.items():
            try:
                bar, sp = await loop.run_in_executor(None, fetcher.fetch_new)
                if bar is None:
                    continue
                any_new = True
                ts = bar['timestamp']
                logger.debug(f"[{pair}] New bar {ts} close={bar['close']} sp={sp:.2f}p")

                prev_counts = {s.cfg.label: s.n_cycles for s in pair_sims[pair]}
                for sim in pair_sims[pair]:
                    sim.process_bar(bar, sp)

                # Log any newly closed cycles to trade file
                for sim in pair_sims[pair]:
                    new_count = sim.n_cycles
                    if new_count > prev_counts[sim.cfg.label]:
                        for rec in sim.records[-(new_count - prev_counts[sim.cfg.label]):]:
                            append_trade(rec, trades_path)

            except Exception as e:
                logger.error(f"[{pair}] fetch/process error: {e}", exc_info=True)

        if any_new:
            try:
                write_status(sims, status_path)
            except Exception as e:
                logger.warning(f"Status write failed: {e}")

            # Console summary every 12 bars (~1 hour)
            if sims[0].bar_count % 12 == 0:
                logger.info("─── Paper Summary ───────────────────────────────")
                for s in sims:
                    live_flag = " ← LIVE" if "LIVE" in s.cfg.note else ""
                    cyc_info = ""
                    if s.cycle:
                        cyc_info = (f" | open L{len(s.cycle.legs)} "
                                    f"{'L' if s.cycle.direction==1 else 'S'} "
                                    f"mfe={s.cycle.peak_mfe:.1f}p")
                    logger.info(f"  [{s.cfg.label:15s}]{live_flag} "
                                f"cycles={s.n_cycles:4d} skip={s.n_skipped:4d} "
                                f"pips={s.total_pips:+8.1f} "
                                f"usd=${s.total_usd:+7.4f}{cyc_info}")

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
    logger.info("Paper trading service stopped.")


if __name__ == "__main__":
    asyncio.run(main())
