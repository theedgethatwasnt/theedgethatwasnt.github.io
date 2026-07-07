#!/usr/bin/env python3
"""
Zone Recovery Trader — EUR_USD (primary), GBP_USD, EUR_JPY, USD_JPY M5.

Design (redesigned Session 052):
  - Net LONG when price is above zone, net SHORT when below zone.
  - Zone crossings fire recovery legs once per boundary per traverse.
    'last_zone_crossed' guard: re-fires only after price has crossed the FULL
    zone to the opposite boundary first.
  - Exit: aggregate P/L trailing stop only.
    Activates once agg P/L >= MIN_LOCK_PIPS; trail distance shrinks from
    TRAIL_BASE_PIPS to MIN_LOCK_PIPS as peak grows to TRAIL_MAX_PNL_PIPS.
    When stop hit: close ALL legs simultaneously, flat.
  - Margin gate blocks NEW CYCLE STARTS only, not recovery legs.

OANDA_ACCOUNT_LONG  (011): receives all LONG legs
OANDA_ACCOUNT_SHORT (012): receives all SHORT legs
ZR_PAIRS env var: comma-separated subset (default: all configured pairs)
BASE_UNITS env var: override per-pair base units for all pairs
"""

import os
import sys
import asyncio
import math
import json
import signal
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

from lib.oanda_adapter import OANDAAdapter
from lib.notify import _send as send_telegram
from lib.db import TradeDBSender, write_trade_direct

_trade_sender: Optional[TradeDBSender] = None

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [zr_random] %(message)s",
)
logger = logging.getLogger("zr_random")

# ── Per-pair validated config (IS/OOS backtest, WF=3/3) ─────────────────────
_BASE_UNITS_OVERRIDE = int(os.environ.get("BASE_UNITS", "0"))

PAIR_CONFIGS = {
    "EUR_JPY": dict(
        pip=0.01, pip_usd=0.000069,
        N=1, zw=50.0, tgt=25.0, ta=6.0, td=1.0, pf=1.25, body_thresh=0.0,
        signal="none",
        base_units=_BASE_UNITS_OVERRIDE or 1,
        pnf_box=0,
        max_entry_spread=0,
    ),
    "CHF_JPY": dict(
        pip=0.01, pip_usd=0.000107,
        N=1, zw=40.0, tgt=20.0, ta=5.0, td=1.0, pf=1.25, body_thresh=0.0,
        signal="none",
        base_units=_BASE_UNITS_OVERRIDE or 20,
        pnf_box=0,
        max_entry_spread=2.5,
    ),
    "EUR_USD": dict(
        pip=0.0001, pip_usd=0.0001,
        N=1, zw=30.0, tgt=21.0, ta=10.0, td=1.0, pf=1.25, body_thresh=0.0,
        signal="none",
        base_units=_BASE_UNITS_OVERRIDE or 8,  # ta=6 FAILS OOS=2/3 — only ta=10 valid
        pnf_box=0,   # 1608 p/d IS=3/3 OOS=3/3 P5=896 CumL5=203
        max_entry_spread=0,
    ),
    "GBP_USD": dict(
        pip=0.0001, pip_usd=0.0001,
        N=1, zw=30.0, tgt=21.0, ta=6.0, td=1.0, pf=1.25, body_thresh=0.5,
        signal="none",
        base_units=_BASE_UNITS_OVERRIDE or 1,  # B=1: pf=1.25 max_margin≈$4.05 min_NAV≈$9.00/acct (CumL5≈211u×1.31 vs pf=1.10)
        pnf_box=0,
        max_entry_spread=2.5,
    ),
    "USD_JPY": dict(
        pip=0.01, pip_usd=0.000064,
        N=1, zw=40.0, tgt=20.0, ta=10.0, td=5.0, pf=1.25, body_thresh=0.0,
        signal="h1_sma20_only",
        base_units=_BASE_UNITS_OVERRIDE or 20,
        pnf_box=0,
        max_entry_spread=2.5,
    ),
}

_pairs_env = os.environ.get("ZR_PAIRS", "")
ACTIVE_PAIRS = (
    [p.strip() for p in _pairs_env.split(",") if p.strip() in PAIR_CONFIGS]
    if _pairs_env else list(PAIR_CONFIGS.keys())
)

PROFIT_FACTOR = 1.25  # legacy default; per-pair pf in PAIR_CONFIGS takes precedence
SPREAD_PIPS   = 1.4
MAX_ENTRY_SPREAD = 2.5  # skip new cycle if live bid/ask > this many pips
MARGIN_CAP    = 0.40   # block NEW CYCLE starts if either account > 40% utilization
STATE_DIR     = os.environ.get("ZR_STATE_DIR", "/tmp")
HEARTBEAT_BARS = 100   # ~25 min at M5 per pair

# Profit-lock trailing stop — activates once aggregate P/L >= MIN_LOCK_PIPS.
# Trail distance shrinks linearly from TRAIL_BASE_PIPS to MIN_LOCK_PIPS as
# peak P/L grows from MIN_LOCK_PIPS to TRAIL_MAX_PNL_PIPS (optimize these three).
MIN_LOCK_PIPS      = 2.0   # minimum pips to lock in before trailing  [validated v2: P5=162.7 P(+)=0.997]
TRAIL_BASE_PIPS    = 3.0   # initial trail distance at activation       [validated v2: best P5 of 168 combos]
TRAIL_MAX_PNL_PIPS = 30.0  # peak at which trail tightens to MIN_LOCK   [validated v2]


# ── Dataclass for active cycle ────────────────────────────────────────────────
@dataclass
class CycleState:
    pair: str
    entry_bar_time: str
    entry_price: float
    direction: int         # +1=LONG, -1=SHORT
    lower_zone: float      # entry − ZW
    upper_zone: float      # entry + ZW (or entry for SHORT direction)
    upper_target: float
    lower_target: float
    tgt_pips: float
    zone_width: float
    all_legs: List[dict] = field(default_factory=list)
    long_count: int = 0
    short_count: int = 0
    last_zone_crossed: Optional[str] = None
    last_zone_crossed_bar: Optional[str] = None
    # Profit-lock trailing stop
    locked: bool = False          # True once aggregate P/L >= MIN_LOCK_PIPS
    peak_pnl: float = 0.0         # highest aggregate P/L seen since locked
    stop_pnl: float = 0.0         # current stop level in pips


# ── State persistence ─────────────────────────────────────────────────────────
def _state_file(pair: str) -> str:
    return os.path.join(STATE_DIR, f"zr_random_{pair}_state.json")

def save_state(cycle: Optional[CycleState], pair: str) -> None:
    try:
        with open(_state_file(pair), 'w') as f:
            if cycle is None:
                json.dump({"active": False}, f)
            else:
                d = cycle.__dict__.copy()
                json.dump({"active": True, "cycle": d}, f, default=str)
    except Exception as e:
        logger.warning(f"[{pair}] save_state failed: {e}")

def load_state(pair: str) -> Optional[CycleState]:
    path = _state_file(pair)
    try:
        if not os.path.exists(path):
            return None
        with open(path) as f:
            d = json.load(f)
        if not d.get("active"):
            return None
        c = d["cycle"]
        cs = CycleState(
            pair=c.get("pair", pair),
            entry_bar_time=c["entry_bar_time"],
            entry_price=c["entry_price"],
            direction=c["direction"],
            lower_zone=c["lower_zone"],
            upper_zone=c["upper_zone"],
            upper_target=c["upper_target"],
            lower_target=c["lower_target"],
            tgt_pips=c["tgt_pips"],
            zone_width=c["zone_width"],
            all_legs=c.get("all_legs", []),
            long_count=c.get("long_count", 0),
            short_count=c.get("short_count", 0),
            last_zone_crossed=c.get("last_zone_crossed"),
            last_zone_crossed_bar=c.get("last_zone_crossed_bar"),
            locked=c.get("locked", False),
            peak_pnl=c.get("peak_pnl", 0.0),
            stop_pnl=c.get("stop_pnl", 0.0),
        )
        logger.info(f"[{pair}] Resumed: entry={cs.entry_price:.5f} "
                    f"legs={len(cs.all_legs)} locked={cs.locked} "
                    f"peak_pnl={cs.peak_pnl:.1f}p stop={cs.stop_pnl:.1f}p")
        return cs
    except Exception as e:
        logger.warning(f"[{pair}] load_state failed: {e}")
        return None


# ── Shared margin gate ────────────────────────────────────────────────────────
class MarginGate:
    """Synchronous check — call via run_in_executor from asyncio context."""

    def __init__(self, broker_long: OANDAAdapter, broker_short: OANDAAdapter):
        self._bl = broker_long
        self._bs = broker_short

    def check_ok(self) -> bool:
        """Returns True only if BOTH accounts are below MARGIN_CAP utilization."""
        for broker, label in [(self._bl, "LONG-011"), (self._bs, "SHORT-012")]:
            info = broker.get_account_summary()
            if info is None:
                logger.warning(f"MarginGate: {label} summary unavailable — blocking hedge")
                return False
            total = info.margin_used + info.margin_available
            if total <= 0:
                continue
            util = info.margin_used / total
            if util > MARGIN_CAP:
                logger.info(f"MarginGate: {label} util={util:.1%} > {MARGIN_CAP:.0%} — blocked")
                return False
        return True


# ── ZR math helpers (pair-aware) ──────────────────────────────────────────────
def _get_pl_at_target(legs: List[dict], target: float, pip: float) -> float:
    return sum(l['vol'] * l['dir'] * (target - l['price']) / pip for l in legs)

def _spread_cost(legs: List[dict], spread: float) -> float:
    """Total spread cost = sum of volumes × spread pips. Use actual live spread."""
    return sum(l['vol'] for l in legs) * spread

def _breakeven_volume(legs: List[dict], target: float, tgt_pips: float,
                      pip: float, spread: float, pf: float = PROFIT_FACTOR) -> float:
    """
    Units needed so that net P&L at target > 0 after paying the new leg's spread.
    Denominator is (tgt_pips - spread): each new unit earns tgt_pips but pays
    spread upfront, so net clearance per unit is (tgt_pips - spread).
    pf: profit factor multiplier (per-pair; default fallback is global PROFIT_FACTOR).
    """
    net = _get_pl_at_target(legs, target, pip) - _spread_cost(legs, spread)
    if net >= 0:
        return 0.0
    net_per_unit = tgt_pips - spread
    if net_per_unit <= 0:
        return 0.0  # spread >= target — degenerate, skip recovery leg
    return max(1.0, float(math.ceil(-net / net_per_unit * pf)))

def _pnl_dollars(legs: List[dict], exit_price: float, pip: float, pip_usd: float,
                 base_units: int, spread: float = SPREAD_PIPS) -> float:
    net_pips = _get_pl_at_target(legs, exit_price, pip) - _spread_cost(legs, spread)
    return net_pips * base_units * pip_usd

def _agg_pnl_pips(legs: List[dict], mid_price: float, pip: float, spread: float) -> float:
    """Aggregate P/L in pips across all legs (mid price, minus total spread costs)."""
    gross = sum(l['vol'] * l['dir'] * (mid_price - l['price']) / pip for l in legs)
    cost  = sum(l['vol'] for l in legs) * spread
    return gross - cost

def _trail_dist(peak_pips: float) -> float:
    """Trail distance in pips. Shrinks linearly from TRAIL_BASE to MIN_LOCK as peak grows."""
    if peak_pips >= TRAIL_MAX_PNL_PIPS:
        return MIN_LOCK_PIPS
    span = max(TRAIL_MAX_PNL_PIPS - MIN_LOCK_PIPS, 1e-9)
    t = max(0.0, min(1.0, (peak_pips - MIN_LOCK_PIPS) / span))
    return TRAIL_BASE_PIPS + t * (MIN_LOCK_PIPS - TRAIL_BASE_PIPS)


# ── Per-pair trader ────────────────────────────────────────────────────────────
class ZRPairTrader:

    def __init__(self, pair: str, cfg: dict,
                 broker_long: OANDAAdapter, broker_short: OANDAAdapter,
                 margin_gate: MarginGate,
                 acct_long: str = "", acct_short: str = ""):
        self.pair        = pair
        self.acct_long   = acct_long
        self.acct_short  = acct_short
        self.pip         = cfg['pip']
        self.pip_usd     = cfg['pip_usd']
        self.N           = cfg['N']
        self.zw          = cfg['zw']
        self.tgt         = cfg['tgt']
        self.ta          = cfg['ta']
        self.td          = cfg['td']
        self.pf          = cfg.get('pf', PROFIT_FACTOR)
        self.body_thresh = cfg.get('body_thresh', 0.0)
        self.sig_mode         = cfg['signal']
        self.base_units       = cfg['base_units']
        self.max_entry_spread = cfg.get('max_entry_spread', MAX_ENTRY_SPREAD)

        self.broker_long  = broker_long
        self.broker_short = broker_short
        self.margin_gate  = margin_gate

        # alternating direction state (unused in h1_sma20_only mode)
        self.next_dir    = 1      # +1=LONG first
        self.bars_to_skip = 0     # N-1 cooldown after cycle exit

        # P&F entry clock (pnf_box=0 → random N-bar mode)
        self.pnf_box   = cfg.get('pnf_box', 0)
        self.pnf_rev   = cfg.get('pnf_rev', 3)
        self._pnf_col  = 0    # +1=X column (up), -1=O column (down), 0=uninit
        self._pnf_ext  = 0.0  # extreme of current P&F column
        self._pnf_seeded = False

        # rolling candle buffers for signal computation
        self.m5_closes: deque = deque(maxlen=25)
        self.h1_closes: deque = deque(maxlen=25)
        self._last_h1_fetch_ts: Optional[str] = None
        # ATR buffer for adaptive td (maxlen=20 gives ATR(5) and ATR(20) from same deque)
        self.hl_pips: deque = deque(maxlen=20)

        self.cycle: Optional[CycleState] = load_state(pair)
        self.bar_count = 0

    # ── P&F chart tracking ────────────────────────────────────────────────────
    def _pnf_state_file(self) -> str:
        return os.path.join(STATE_DIR, f"zr_pnf_{self.pair}_state.json")

    def _save_pnf_state(self) -> None:
        try:
            with open(self._pnf_state_file(), 'w') as f:
                json.dump({'col': self._pnf_col, 'ext': self._pnf_ext,
                           'seeded': self._pnf_seeded}, f)
        except Exception as e:
            logger.warning(f"[{self.pair}] save_pnf_state failed: {e}")

    def _load_pnf_state(self) -> bool:
        path = self._pnf_state_file()
        try:
            if not os.path.exists(path):
                return False
            with open(path) as f:
                d = json.load(f)
            self._pnf_col    = d.get('col', 0)
            self._pnf_ext    = d.get('ext', 0.0)
            self._pnf_seeded = d.get('seeded', False)
            col_name = 'X' if self._pnf_col > 0 else ('O' if self._pnf_col < 0 else '?')
            logger.info(f"[{self.pair}] P&F state loaded: col={col_name} ext={self._pnf_ext:.5f}")
            return self._pnf_seeded
        except Exception as e:
            logger.warning(f"[{self.pair}] load_pnf_state failed: {e}")
            return False

    def _seed_pnf(self, candles: list) -> None:
        """Build P&F chart from historical bars (all except last incomplete bar)."""
        p   = self.pip
        box = self.pnf_box
        rev = self.pnf_rev
        self._pnf_col = 0
        self._pnf_ext = 0.0
        for c in candles:
            hi = float(c['high']); lo = float(c['low'])
            if self._pnf_col == 0:
                self._pnf_col = 1
                self._pnf_ext = hi
                continue
            if self._pnf_col == 1:   # X column
                if hi >= self._pnf_ext + box * p:
                    self._pnf_ext = hi
                elif lo <= self._pnf_ext - rev * box * p:
                    self._pnf_col = -1
                    self._pnf_ext = lo
            else:                     # O column
                if lo <= self._pnf_ext - box * p:
                    self._pnf_ext = lo
                elif hi >= self._pnf_ext + rev * box * p:
                    self._pnf_col = 1
                    self._pnf_ext = hi
        self._pnf_seeded = True
        col_name = 'X' if self._pnf_col > 0 else 'O'
        logger.info(f"[{self.pair}] P&F seeded from {len(candles)} bars: "
                    f"col={col_name} ext={self._pnf_ext:.5f} box={box}p rev={rev}")
        self._save_pnf_state()

    def _update_pnf(self, bar: dict) -> bool:
        """Update P&F chart with new bar. Returns True if a column reversal occurred."""
        if not self._pnf_seeded:
            return False
        p   = self.pip
        box = self.pnf_box
        rev = self.pnf_rev
        hi = float(bar['high']); lo = float(bar['low'])
        reversed_ = False
        if self._pnf_col == 1:   # X column
            if hi >= self._pnf_ext + box * p:
                self._pnf_ext = hi
            elif lo <= self._pnf_ext - rev * box * p:
                self._pnf_col = -1
                self._pnf_ext = lo
                reversed_ = True
        else:                     # O column
            if lo <= self._pnf_ext - box * p:
                self._pnf_ext = lo
            elif hi >= self._pnf_ext + rev * box * p:
                self._pnf_col = 1
                self._pnf_ext = hi
                reversed_ = True
        if reversed_:
            col_name = 'X' if self._pnf_col > 0 else 'O'
            logger.info(f"[{self.pair}] P&F reversal → {col_name} col ext={self._pnf_ext:.5f}")
            self._save_pnf_state()
        return reversed_

    # ── Body absorption filter ────────────────────────────────────────────────
    def _check_body_absorption(self, bar: dict, direction: int) -> bool:
        """
        Returns True (entry allowed) unless adverse candle body > body_thresh × range.
        LONG entry: block if bearish body (open > close) fraction > body_thresh.
        SHORT entry: block if bullish body (close > open) fraction > body_thresh.
        Direction is NOT flipped on a skipped bar — it stays ready for next bar.
        """
        if self.body_thresh <= 0.0:
            return True
        hi  = float(bar['high'])
        lo  = float(bar['low'])
        op  = float(bar['open'])
        cl  = float(bar['close'])
        rng = hi - lo
        if rng < 1e-10:
            return True
        if direction == 1:
            adverse_body = max(0.0, op - cl) / rng
        else:
            adverse_body = max(0.0, cl - op) / rng
        if adverse_body > self.body_thresh:
            logger.debug(f"[{self.pair}] Absorption skip: adverse_body={adverse_body:.2f} "
                         f"> thresh={self.body_thresh} dir={'L' if direction==1 else 'S'}")
            return False
        return True

    # ── Signal computation ─────────────────────────────────────────────────────
    def _signal_direction(self) -> Optional[int]:
        """
        Returns intended entry direction (+1/-1) or None to skip this entry.
        m5_sma10_filter : keep alternating, skip if M5 SMA10 slope contradicts
        h1_sma20_only   : always enter in H1 SMA20 direction (ignore alternating)
        none            : always use next_dir (no filter)
        """
        if self.sig_mode == "none":
            return self.next_dir

        elif self.sig_mode == "m5_sma10_filter":
            if len(self.m5_closes) < 11:
                return None  # warmup
            buf = list(self.m5_closes)
            sma_now  = float(np.mean(buf[-10:]))
            sma_prev = float(np.mean(buf[-11:-1]))
            slope = 1 if sma_now > sma_prev else -1
            if slope == self.next_dir:
                return self.next_dir
            return None  # contradicts — skip, don't flip next_dir

        elif self.sig_mode == "h1_sma20_only":
            if len(self.h1_closes) < 21:
                return None  # warmup
            buf = list(self.h1_closes)
            sma20 = float(np.mean(buf[-20:]))
            d = 1 if buf[-1] > sma20 else -1
            return d  # replaces alternating

        return self.next_dir

    # ── H1 candle update ──────────────────────────────────────────────────────
    def _update_h1_buffer(self, candles_h1: list) -> None:
        """Add any new H1 closes to the buffer."""
        for c in candles_h1:
            ts = c['timestamp']
            if ts != self._last_h1_fetch_ts:
                self.h1_closes.append(float(c['close']))
                self._last_h1_fetch_ts = ts

    # ── Order helpers ─────────────────────────────────────────────────────────
    def _place_leg(self, direction: int, volume: float,
                   price_label: str) -> tuple:
        """Place one leg. Returns (tid_long, tid_short, fill_price)."""
        units = int(round(volume * self.base_units))
        tid_long = tid_short = fill_price = None
        if direction == 1:
            r = self.broker_long.place_market_order(self.pair, units)
            if r.success:
                tid_long = r.trade_id; fill_price = r.fill_price
                logger.info(f"[{self.pair}] LONG leg @{price_label}: {units}u "
                            f"fill={fill_price:.5f} id={tid_long}")
            else:
                logger.error(f"[{self.pair}] LONG leg FAILED: {r.error or r.cancel_reason}")
        else:
            r = self.broker_short.place_market_order(self.pair, -units)
            if r.success:
                tid_short = r.trade_id; fill_price = r.fill_price
                logger.info(f"[{self.pair}] SHORT leg @{price_label}: {units}u "
                            f"fill={fill_price:.5f} id={tid_short}")
            else:
                logger.error(f"[{self.pair}] SHORT leg FAILED: {r.error or r.cancel_reason}")
        return tid_long, tid_short, fill_price

    def _close_all_legs(self, reason: str, exit_price: float,
                        spread: float = SPREAD_PIPS) -> None:
        """Close all legs by trade_id (safe in multi-pair context)."""
        c = self.cycle
        pnl = _pnl_dollars(c.all_legs, exit_price, self.pip, self.pip_usd,
                            self.base_units, spread)
        logger.info(f"[{self.pair}] EXIT [{reason}] @ {exit_price:.5f} "
                    f"legs={len(c.all_legs)} P&L=${pnl:+.4f}")
        closed_long = closed_short = 0
        for leg in c.all_legs:
            if leg.get('tid_long'):
                self.broker_long.close_trade(leg['tid_long'])
                closed_long += 1
            if leg.get('tid_short'):
                self.broker_short.close_trade(leg['tid_short'])
                closed_short += 1
        logger.info(f"[{self.pair}]   Closed {closed_long}L + {closed_short}S trades")
        send_telegram(
            f"{'🟢' if pnl >= 0 else '🔴'} ZR {self.pair} [{reason}]\n"
            f"Acct 011/012 | Dir: {'L' if c.direction==1 else 'S'} | "
            f"Entry {c.entry_price:.5f} → {exit_price:.5f}\n"
            f"Legs: {len(c.all_legs)} ({c.long_count}L/{c.short_count}S) | "
            f"ZW: {c.zone_width:.1f}p tgt: {c.tgt_pips:.1f}p | "
            f"Est. P&L: ${pnl:+.4f}"
        )
        # Log closed cycle to DuckDB
        try:
            net_pips = (_get_pl_at_target(c.all_legs, exit_price, self.pip)
                        - _spread_cost(c.all_legs, spread))
            entry_dt = datetime.fromisoformat(c.entry_bar_time.replace('Z', '+00:00'))
            exit_dt  = datetime.now(timezone.utc)
            hours_held = (exit_dt - entry_dt).total_seconds() / 3600
            acct_id = self.acct_long if c.direction == 1 else self.acct_short
            pf_tag = int(self.pf * 100)
            body_tag = "ABS" if self.body_thresh > 0 else "ZR"
            lbl = f"{self.pair}_{body_tag}_{pf_tag}"
            trade_kwargs = dict(
                strategy="zr_random",
                pair=self.pair,
                account_id=acct_id,
                trade_id=f"{self.pair}_{c.entry_bar_time}",
                direction=c.direction,
                entry_price=c.entry_price,
                exit_price=exit_price,
                entry_time=c.entry_bar_time,
                exit_time=exit_dt.isoformat(),
                pnl_pips=net_pips,
                exit_reason=reason,
                hours_held=hours_held,
                units=int(len(c.all_legs) * self.base_units),
                mfe_pips=c.peak_pnl,
                mae_pips=0.0,
                capture_ratio=0.0,
                is_paper=False,
                label=lbl,
            )
            # Primary: ZMQ → portfolio_mgr. Fallback: direct write if sender unavailable.
            if _trade_sender is not None:
                _trade_sender.send_trade(**trade_kwargs)
            else:
                write_trade_direct(**trade_kwargs)
        except Exception as _e:
            logger.warning(f"[{self.pair}] TradeDB log failed: {_e}")
        self.cycle = None
        save_state(None, self.pair)
        # Post-exit cooldown: P&F mode uses reversal clock; N-bar uses N-1 skip
        self.bars_to_skip = 0 if self.pnf_box > 0 else self.N - 1

    # ── Cycle start ───────────────────────────────────────────────────────────
    def _start_cycle(self, bar: dict, direction: int) -> None:
        entry = float(bar['close'])
        p = self.pip; zw = self.zw; tgt = self.tgt

        # Matches backtest: LONG enters at uz (upper boundary), zone extends down ZW pips.
        # SHORT enters at lz (lower boundary), zone extends up ZW pips.
        # Backtest: d=1 → uz=e, lz=e-zw*pip, ut=e+tgt*pip, lt=lz-tgt*pip
        if direction == 1:
            upper_zone   = entry              # uz = entry (upper boundary)
            lower_zone   = entry - zw * p     # lz = entry - ZW
            upper_target = entry + tgt * p    # ut = entry + tgt
            lower_target = lower_zone - tgt * p  # lt = entry - ZW - tgt
        else:
            lower_zone   = entry              # lz = entry (lower boundary)
            upper_zone   = entry + zw * p     # uz = entry + ZW
            lower_target = entry - tgt * p    # lt = entry - tgt
            upper_target = upper_zone + tgt * p  # ut = entry + ZW + tgt

        self.cycle = CycleState(
            pair=self.pair,
            entry_bar_time=bar['timestamp'],
            entry_price=entry,
            direction=direction,
            lower_zone=lower_zone,
            upper_zone=upper_zone,
            upper_target=upper_target,
            lower_target=lower_target,
            tgt_pips=tgt,
            zone_width=zw,
        )
        c = self.cycle

        tid_long, tid_short, fill_price = self._place_leg(direction, 1.0, f"{entry:.5f}")
        leg_price = fill_price if fill_price is not None else entry
        c.all_legs.append({
            'dir': direction, 'price': leg_price, 'vol': 1.0,
            'tid_long': tid_long, 'tid_short': tid_short,
        })
        if direction == 1: c.long_count  = 1
        else:              c.short_count = 1

        save_state(c, self.pair)
        send_telegram(
            f"{'🟢' if direction==1 else '🔴'} ZR {self.pair} OPEN | Accts 011/012\n"
            f"Entry: {entry:.5f} | {'LONG' if direction==1 else 'SHORT'}\n"
            f"Zone: [{lower_zone:.5f}, {upper_zone:.5f}] ZW={zw:.1f}p\n"
            f"Targets: ↓{lower_target:.5f} ↑{upper_target:.5f} (tgt={tgt:.1f}p)"
        )
        logger.info(f"[{self.pair}] Cycle OPEN: entry={entry:.5f} "
                    f"dir={'L' if direction==1 else 'S'} "
                    f"zone=[{lower_zone:.5f},{upper_zone:.5f}] zw={zw:.1f}p tgt={tgt:.1f}p")

        # Update alternating direction for next cycle (only used in non-h1_sma20_only modes)
        if self.sig_mode != "h1_sma20_only":
            self.next_dir = -direction

    # ── Bar processing ────────────────────────────────────────────────────────
    def process_bar(self, bar: dict, margin_gate_ok: bool) -> None:
        hi = float(bar['high']); lo = float(bar['low'])
        cl = float(bar['close']); op = float(bar['open'])
        ts = bar['timestamp']
        bullish = cl >= op
        bid_c = float(bar.get('bid_c', cl))
        ask_c = float(bar.get('ask_c', cl))
        live_spread = (ask_c - bid_c) / self.pip if ask_c > bid_c else SPREAD_PIPS
        mid = (bid_c + ask_c) / 2

        self.m5_closes.append(cl)
        self.hl_pips.append((hi - lo) / self.pip)
        self.bar_count += 1

        pnf_reversed = self._update_pnf(bar) if self.pnf_box > 0 else False

        # ── No active cycle: check for new entry (margin gate here only) ─────
        if self.cycle is None:
            if self.pnf_box > 0:
                if pnf_reversed:
                    direction = self.next_dir
                    self.next_dir = -direction
                    if margin_gate_ok:
                        self._start_cycle(bar, direction)
                    else:
                        logger.info(f"[{self.pair}] MarginGate blocked new cycle")
                return
            else:
                if self.bars_to_skip > 0:
                    self.bars_to_skip -= 1
                    return
                direction = self._signal_direction()
                if direction is not None:
                    mes = self.max_entry_spread
                    if mes > 0 and live_spread > mes:
                        logger.info(f"[{self.pair}] Entry blocked: spread={live_spread:.1f}p > max={mes:.1f}p")
                    elif not margin_gate_ok:
                        logger.info(f"[{self.pair}] MarginGate blocked new cycle")
                    elif self._check_body_absorption(bar, direction):
                        self._start_cycle(bar, direction)
                return

        c = self.cycle
        p = self.pip

        # ── Aggregate P/L trailing stop ────────────────────────────────────────
        # Check BEFORE zone crossings: exit takes priority over adding new legs.
        agg_pnl = _agg_pnl_pips(c.all_legs, mid, p, live_spread)

        if not c.locked and agg_pnl >= MIN_LOCK_PIPS:
            c.locked = True
            c.peak_pnl = agg_pnl
            c.stop_pnl = MIN_LOCK_PIPS
            logger.info(f"[{self.pair}] Profit LOCKED: agg={agg_pnl:.2f}p stop={c.stop_pnl:.2f}p")

        if c.locked:
            if agg_pnl > c.peak_pnl:
                c.peak_pnl = agg_pnl
                c.stop_pnl = max(c.stop_pnl, c.peak_pnl - _trail_dist(c.peak_pnl))
            if agg_pnl <= c.stop_pnl:
                self._close_all_legs("trail_lock", mid, live_spread)
                return

        # ── Zone crossings: add recovery legs (no margin gate) ────────────────
        # Bull bar: high reached before low. Bear bar: low before high.
        sequence = [(hi, True), (lo, False)] if bullish else [(lo, False), (hi, True)]

        for _extreme, is_high in sequence:
            if self.cycle is None:
                break

            if is_high and hi >= c.upper_zone:
                if c.last_zone_crossed != "upper":
                    c.last_zone_crossed = "upper"; c.last_zone_crossed_bar = ts
                    net_vol = sum(leg['vol'] * leg['dir'] for leg in c.all_legs)
                    vol = _breakeven_volume(c.all_legs, c.upper_target, c.tgt_pips,
                                           p, live_spread, self.pf)
                    if vol > 0 and net_vol < 0:  # net short → LONG recovery
                        net_at = (_get_pl_at_target(c.all_legs, c.upper_target, p)
                                  - _spread_cost(c.all_legs, live_spread))
                        tid_long, tid_short, fill_price = self._place_leg(
                            1, vol, f"{c.upper_zone:.5f}")
                        leg_price = fill_price if fill_price is not None else c.upper_zone
                        c.all_legs.append({'dir': 1, 'price': leg_price, 'vol': vol,
                                           'tid_long': tid_long, 'tid_short': tid_short})
                        c.long_count += 1
                        save_state(c, self.pair)
                        logger.info(f"[{self.pair}] Upper cross: +LONG {vol:.0f}u "
                                    f"sp={live_spread:.2f}p net@tgt={net_at:+.1f}p "
                                    f"legs={len(c.all_legs)}")

            if not is_high and lo <= c.lower_zone:
                if c.last_zone_crossed != "lower":
                    c.last_zone_crossed = "lower"; c.last_zone_crossed_bar = ts
                    net_vol = sum(leg['vol'] * leg['dir'] for leg in c.all_legs)
                    vol = _breakeven_volume(c.all_legs, c.lower_target, c.tgt_pips,
                                           p, live_spread, self.pf)
                    if vol > 0 and net_vol > 0:  # net long → SHORT recovery
                        net_at = (_get_pl_at_target(c.all_legs, c.lower_target, p)
                                  - _spread_cost(c.all_legs, live_spread))
                        tid_long, tid_short, fill_price = self._place_leg(
                            -1, vol, f"{c.lower_zone:.5f}")
                        leg_price = fill_price if fill_price is not None else c.lower_zone
                        c.all_legs.append({'dir': -1, 'price': leg_price, 'vol': vol,
                                           'tid_long': tid_long, 'tid_short': tid_short})
                        c.short_count += 1
                        save_state(c, self.pair)
                        logger.info(f"[{self.pair}] Lower cross: +SHORT {vol:.0f}u "
                                    f"sp={live_spread:.2f}p net@tgt={net_at:+.1f}p "
                                    f"legs={len(c.all_legs)}")

        if self.cycle is not None:
            save_state(self.cycle, self.pair)

    # ── Startup orphan check ──────────────────────────────────────────────────
    def close_orphaned_trades(self) -> None:
        """Close open trades for this pair that are not in saved state."""
        state_tids: set = set()
        if self.cycle:
            for leg in self.cycle.all_legs:
                if leg.get('tid_long'):  state_tids.add(leg['tid_long'])
                if leg.get('tid_short'): state_tids.add(leg['tid_short'])

        for broker, label in [(self.broker_long, 'long'), (self.broker_short, 'short')]:
            try:
                open_trades = broker.get_open_trades()
                orphans = [t for t in open_trades
                           if t.instrument == self.pair and t.trade_id not in state_tids]
                if orphans:
                    logger.warning(f"[{self.pair}] STARTUP: {len(orphans)} orphaned "
                                   f"{label} trade(s) — closing: {[t.trade_id for t in orphans]}")
                    for t in orphans:
                        broker.close_trade(t.trade_id)
            except Exception as e:
                logger.warning(f"[{self.pair}] Orphan check {label} failed: {e}")


# ── Asyncio per-pair loop ──────────────────────────────────────────────────────
async def run_pair_loop(pair: str, trader: ZRPairTrader,
                        margin_gate: MarginGate,
                        shutdown_event: asyncio.Event) -> None:
    loop = asyncio.get_event_loop()
    last_bar_ts: Optional[str] = None

    # Seed P&F chart from historical bars (300 M5 bars ≈ 25h of history)
    if trader.pnf_box > 0:
        if not trader._load_pnf_state():
            try:
                pnf_seed = await loop.run_in_executor(
                    None, lambda: trader.broker_long.get_candles(pair, count=301, granularity="M5"))
                trader._seed_pnf(pnf_seed[:-1])   # exclude current incomplete bar
            except Exception as e:
                logger.warning(f"[{pair}] P&F seed failed: {e}")
        logger.info(f"[{pair}] P&F entry clock: box={trader.pnf_box}p rev={trader.pnf_rev} "
                    f"seeded={trader._pnf_seeded}")

    # Fetch initial H1 buffer if needed
    if trader.sig_mode == "h1_sma20_only":
        try:
            h1_candles = await loop.run_in_executor(
                None, lambda: trader.broker_long.get_candles(pair, count=25, granularity="H1"))
            trader._update_h1_buffer(h1_candles)
            logger.info(f"[{pair}] H1 buffer seeded: {len(trader.h1_closes)} bars")
        except Exception as e:
            logger.warning(f"[{pair}] H1 seed failed: {e}")

    # Seed M5 buffer for SMA warmup
    if trader.sig_mode == "m5_sma10_filter":
        try:
            seed = await loop.run_in_executor(
                None, lambda: trader.broker_long.get_candles(pair, count=15, granularity="M5"))
            for c in seed[:-1]:  # exclude current (incomplete) bar
                trader.m5_closes.append(float(c['close']))
            logger.info(f"[{pair}] M5 buffer seeded: {len(trader.m5_closes)} bars")
        except Exception as e:
            logger.warning(f"[{pair}] M5 seed failed: {e}")

    entry_clock = (f"P&F b{trader.pnf_box}r{trader.pnf_rev}" if trader.pnf_box > 0
                   else f"random N={trader.N}")
    logger.info(f"[{pair}] Pair loop started | "
                f"entry={entry_clock} ZW={trader.zw} tgt={trader.tgt} "
                f"ta={trader.ta} td={trader.td} signal={trader.sig_mode} "
                f"base_units={trader.base_units}")

    while not shutdown_event.is_set():
        try:
            candles = await loop.run_in_executor(
                None, lambda: trader.broker_long.get_candles(pair, count=3, granularity="M5"))
            if not candles:
                await asyncio.sleep(10)
                continue

            latest = candles[-1]
            if latest['timestamp'] == last_bar_ts:
                await asyncio.sleep(15)
                continue

            last_bar_ts = latest['timestamp']

            # Update H1 buffer once per hour for USD_JPY
            if trader.sig_mode == "h1_sma20_only" and trader.bar_count % 12 == 0:
                try:
                    h1_candles = await loop.run_in_executor(
                        None, lambda: trader.broker_long.get_candles(
                            pair, count=25, granularity="H1"))
                    trader._update_h1_buffer(h1_candles)
                except Exception as e:
                    logger.warning(f"[{pair}] H1 refresh failed: {e}")

            # Check margin gate before processing (needed for potential hedge decisions)
            margin_ok = await loop.run_in_executor(None, margin_gate.check_ok)

            trader.process_bar(latest, margin_ok)

            if trader.bar_count % HEARTBEAT_BARS == 0:
                cl = float(latest['close'])
                cycle_info = (
                    f"cycle legs={len(trader.cycle.all_legs)} "
                    f"{'L' if trader.cycle.direction==1 else 'S'} "
                    f"entry={trader.cycle.entry_price:.5f} "
                    f"locked={trader.cycle.locked} "
                    f"peak={trader.cycle.peak_pnl:.1f}p stop={trader.cycle.stop_pnl:.1f}p"
                    if trader.cycle else "no cycle"
                )
                logger.info(f"[{pair}] Heartbeat #{trader.bar_count} | "
                            f"close={cl:.5f} | {cycle_info}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[{pair}] Loop error: {e}", exc_info=True)
            await asyncio.sleep(30)

    logger.info(f"[{pair}] Pair loop stopped")


# ── Main orchestrator ──────────────────────────────────────────────────────────
async def main_async() -> None:
    acct_long  = os.environ.get("OANDA_ACCOUNT_LONG",  os.environ.get("OANDA_ACCOUNT_ID_011"))
    acct_short = os.environ.get("OANDA_ACCOUNT_SHORT", os.environ.get("OANDA_ACCOUNT_ID_012"))
    if not acct_long or not acct_short:
        raise ValueError("OANDA_ACCOUNT_LONG and OANDA_ACCOUNT_SHORT must be set")

    broker_long  = OANDAAdapter(account_id=acct_long)
    broker_short = OANDAAdapter(account_id=acct_short)
    logger.info(f"Long  account: {acct_long}")
    logger.info(f"Short account: {acct_short}")
    logger.info(f"Active pairs: {ACTIVE_PAIRS}")

    margin_gate = MarginGate(broker_long, broker_short)
    shutdown_event = asyncio.Event()

    def _handle_signal(*_):
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, _handle_signal)
    loop.add_signal_handler(signal.SIGINT,  _handle_signal)

    global _trade_sender
    try:
        _trade_sender = TradeDBSender()
        logger.info("TradeDBSender initialized — cycles will be logged to DuckDB")
    except Exception as e:
        logger.warning(f"TradeDBSender init failed (no DB logging): {e}")

    # Create traders
    traders: dict[str, ZRPairTrader] = {}
    for pair in ACTIVE_PAIRS:
        cfg = PAIR_CONFIGS[pair]
        t = ZRPairTrader(pair, cfg, broker_long, broker_short, margin_gate,
                         acct_long=acct_long, acct_short=acct_short)
        t.close_orphaned_trades()
        traders[pair] = t
        logger.info(f"[{pair}] Initialized | base_units={cfg['base_units']} | "
                    f"state={'resumed' if t.cycle else 'fresh'}")

    # Launch concurrent pair loops
    tasks = [
        asyncio.create_task(run_pair_loop(pair, traders[pair], margin_gate, shutdown_event))
        for pair in ACTIVE_PAIRS
    ]
    await asyncio.gather(*tasks)
    logger.info("ZR Random multi-pair trader stopped")


if __name__ == "__main__":
    asyncio.run(main_async())
