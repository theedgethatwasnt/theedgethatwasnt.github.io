"""
Unified Capital Allocation & Position Sizing Framework
======================================================
Shared module imported by all strategy traders.

Each trader instantiates a StrategyAllocator with its strategy name,
and calls compute_units() at entry time. No IPC — each trader is
an independent systemd process.

Sizing formula (first-come-first-served):
  units = min(risk_cap, safe_f_ceiling, currency_cap) × perf_gate × dd_scale

Portfolio-level drawdown controls (prop-firm style):
  - Daily DD limit: 25.0% of start-of-day equity
  - Overall DD limit: 50.0% of peak equity
  - Includes unrealized P/L — checked on every equity update
  - Throttle: as DD deepens, sizing shrinks proportionally
  - At limit: new entries blocked (units=0), but existing positions NOT force-closed
  - Each strategy declares expected_max_unreal_dd_pct — worst-case concurrent unrealized DD
"""

import json
import math
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sizing_config.json")


# ─── Weekend Guard ─────────────────────────────────────────────────────────
# FX markets close Friday 5:00 PM New York time. In UTC:
#   - EST (Nov-Mar): 22:00 UTC
#   - EDT (Mar-Nov): 21:00 UTC
# We detect the current US Eastern offset dynamically to handle DST correctly.
# Close positions 15 min before market close. Block entries 1 hour before.
# Markets reopen Sunday 5:00 PM New York time (same DST logic).

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    import pytz
    _ET = pytz.timezone("America/New_York")


def _fx_close_utc_hour() -> int:
    """Return the FX market close hour in UTC for today (21 in EDT, 22 in EST)."""
    now_et = datetime.now(timezone.utc).astimezone(_ET)
    # ET offset: -4 (EDT) or -5 (EST)
    offset_hours = now_et.utcoffset().total_seconds() / 3600
    # 5 PM ET in UTC = 17 - offset (offset is negative)
    return int(17 - offset_hours)  # 21 (EDT) or 22 (EST)


def is_weekend_entry_blocked() -> bool:
    """Return True if new entries should be blocked (approaching/during weekend).
    Blocks from 1 hour before FX close Friday through Sunday reopen."""
    now = datetime.now(timezone.utc)
    dow = now.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    close_hour = _fx_close_utc_hour()
    if dow == 4 and now.hour >= close_hour - 1:  # Friday, 1h before close
        return True
    if dow == 5:  # Saturday — market closed
        return True
    if dow == 6 and now.hour < close_hour:  # Sunday before reopen
        return True
    return False


def is_weekend_close_time() -> bool:
    """Return True if positions should be closed NOW (15 min before FX market close Friday)."""
    now = datetime.now(timezone.utc)
    if now.weekday() != 4:  # Only on Friday
        return False
    close_hour = _fx_close_utc_hour()
    # Close at close_hour - 1 hour, 45 minutes (= 15 min before close)
    close_minute_utc = (close_hour - 1) * 60 + 45  # e.g., 20:45 UTC (EDT) or 21:45 UTC (EST)
    current_minute_utc = now.hour * 60 + now.minute
    return current_minute_utc >= close_minute_utc


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load sizing config from JSON file."""
    with open(path) as f:
        return json.load(f)


# ─── Performance Gate (Expectancy-R based) ──────────────────────────────────

def compute_expectancy_r(trades: List[dict]) -> Optional[float]:
    """
    Compute Expectancy R from a list of trades.
    ExpR = (WR × avg_win - (1-WR) × avg_loss) / avg_loss
    Each trade dict must have 'pnl_pips' key.
    Returns None if no trades.
    """
    if not trades:
        return None

    wins = [t["pnl_pips"] for t in trades if t["pnl_pips"] > 0]
    losses = [abs(t["pnl_pips"]) for t in trades if t["pnl_pips"] <= 0]

    if not losses:
        return 10.0  # All winners — cap at high value
    if not wins:
        return -10.0  # All losers

    wr = len(wins) / len(trades)
    avg_win = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)

    return (wr * avg_win - (1 - wr) * avg_loss) / avg_loss


def compute_perf_gate(trades: List[dict], config: dict) -> Tuple[float, Optional[float], int]:
    """
    Compute performance scaling factor (0.0 to 1.0) based on rolling ExpR.

    Returns (scale_factor, expectancy_r, n_trades)
    - ExpR > full_expr → 1.0 (full allocation)
    - ExpR < kill_expr → 0.0 (pair shut off)
    - Between → linear interpolation
    - Fewer than min_trades → 1.0 (trust backtest)
    """
    window = config.get("perf_gate_window", 20)
    min_trades = config.get("perf_gate_min_trades", 5)
    kill_expr = config.get("perf_gate_kill_expr", -0.1)
    full_expr = config.get("perf_gate_full_expr", 0.0)

    recent = trades[-window:]
    n = len(recent)

    if n < min_trades:
        return 1.0, None, n  # Not enough data, trust backtest

    expr = compute_expectancy_r(recent)
    if expr is None:
        return 1.0, None, n

    if expr >= full_expr:
        return 1.0, expr, n
    elif expr <= kill_expr:
        return 0.0, expr, n
    else:
        # Linear interpolation between kill and full
        scale = (expr - kill_expr) / (full_expr - kill_expr)
        return scale, expr, n


# ─── Bayesian Sharpe Blending ───────────────────────────────────────────────

def bayesian_blend(backtest_sharpe: float, live_sharpe: Optional[float],
                   n_live_trades: int, halflife: int = 50) -> float:
    """
    Blend backtest and live Sharpe ratios using exponential decay on the prior.

    At 0 live trades: 100% backtest
    At halflife trades: 50/50
    At 2×halflife: 25% backtest, 75% live
    """
    if live_sharpe is None or n_live_trades == 0:
        return backtest_sharpe

    # Weight of backtest decays exponentially
    backtest_weight = 0.5 ** (n_live_trades / halflife)
    live_weight = 1.0 - backtest_weight

    return backtest_weight * backtest_sharpe + live_weight * live_sharpe


# ─── Portfolio-Level Drawdown Guard (Prop-Firm Style) ───────────────────────

class DrawdownGuard:
    """
    Portfolio-level drawdown THROTTLE (not a kill switch).

    Tracks two high-water marks and adjusts sizing based on DD depth:
      1. Daily DD: % loss from start-of-day equity (resets at daily_reset_hour UTC)
      2. Overall DD: % loss from all-time peak equity (never resets)

    Unrealized P/L counts — call update() on every tick/bar with current NAV.

    Behavior:
      - DD < 50% of limit → full sizing (dd_scale = 1.0)
      - DD 50%-100% of limit → linear taper (dd_scale shrinks toward 0)
      - DD >= limit → dd_scale = 0.0 (no new entries, but existing positions ride out)
      - NEVER force-closes existing positions

    Each strategy declares expected_max_unreal_dd_pct in config — the worst-case
    concurrent unrealized DD from backtesting. This feeds into how aggressively
    the throttle engages: if you KNOW a strategy can have 5% unrealized DD
    intra-trade, the throttle starts tapering earlier to leave room.

    Usage:
        guard = DrawdownGuard(max_daily_dd_pct=25.0, max_overall_dd_pct=50.0)
        guard.update(nav)  # call on every equity change (including unrealized)

        dd_scale = guard.dd_scale()  # 0.0–1.0 multiplier for sizing
        # Apply: units = units * dd_scale

        if not guard.can_trade():
            # dd_scale == 0 — don't open new positions
    """

    def __init__(
        self,
        max_daily_dd_pct: float = 25.0,
        max_overall_dd_pct: float = 50.0,
        daily_reset_hour: int = 0,
        taper_start_frac: float = 0.50,  # Start tapering at 50% of limit
    ):
        self.max_daily_dd_pct = max_daily_dd_pct
        self.max_overall_dd_pct = max_overall_dd_pct
        self.daily_reset_hour = daily_reset_hour
        self.taper_start_frac = taper_start_frac  # fraction of limit where taper begins

        # High-water marks
        self.daily_hwm: Optional[float] = None
        self.overall_hwm: Optional[float] = None
        self.current_nav: Optional[float] = None

        # State
        self._last_reset_date: Optional[str] = None
        self._breach_log: List[dict] = []

    def update(self, nav: float) -> dict:
        """
        Update with current NAV (balance + unrealized P/L).
        Call on every tick, bar close, or equity refresh.

        Returns status dict with current DD metrics and dd_scale.
        """
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        self.current_nav = nav

        # Daily reset: new trading day
        if self._last_reset_date != today and now.hour >= self.daily_reset_hour:
            self.daily_hwm = nav
            self._last_reset_date = today
            logger.info(f"[DD Guard] Daily reset — HWM set to ${nav:.2f} for {today}")

        # First call ever
        if self.daily_hwm is None:
            self.daily_hwm = nav
            self._last_reset_date = today

        if self.overall_hwm is None:
            self.overall_hwm = nav

        # Update overall HWM (only goes up)
        if nav > self.overall_hwm:
            self.overall_hwm = nav

        # Compute drawdowns
        daily_dd_pct = self._dd_pct(self.daily_hwm, nav)
        overall_dd_pct = self._dd_pct(self.overall_hwm, nav)

        # Log breaches (for audit trail, but no force-close)
        if daily_dd_pct >= self.max_daily_dd_pct:
            breach = {
                "type": "daily", "time": now.isoformat(),
                "hwm": self.daily_hwm, "nav": nav, "dd_pct": daily_dd_pct,
            }
            if not self._breach_log or self._breach_log[-1].get("type") != "daily" or \
               self._breach_log[-1].get("time", "")[:10] != today:
                self._breach_log.append(breach)
                logger.warning(
                    f"[DD Guard] DAILY DD LIMIT HIT: {daily_dd_pct:.2f}% "
                    f"(limit {self.max_daily_dd_pct}%) — new entries blocked, "
                    f"existing positions untouched"
                )

        if overall_dd_pct >= self.max_overall_dd_pct:
            breach = {
                "type": "overall", "time": now.isoformat(),
                "hwm": self.overall_hwm, "nav": nav, "dd_pct": overall_dd_pct,
            }
            if not self._breach_log or self._breach_log[-1].get("type") != "overall":
                self._breach_log.append(breach)
                logger.warning(
                    f"[DD Guard] OVERALL DD LIMIT HIT: {overall_dd_pct:.2f}% "
                    f"(limit {self.max_overall_dd_pct}%) — new entries blocked, "
                    f"existing positions untouched"
                )

        return self.get_status()

    def update_from_positions(self, balance: float, open_positions: List[dict]) -> dict:
        """
        Compute NAV from balance + sum of unrealized P/L and update guard.
        Call this on every tick or bar — no API call needed.

        Each position dict: {"pair": str, "direction": int, "units": int,
                             "entry_price": float, "current_price": float, "pip": float}

        This is the REAL-TIME path — use instead of update() for tick-level monitoring.
        The periodic refresh_equity() via REST API serves as a sanity cross-check.
        """
        unrealized_pl = 0.0
        for pos in open_positions:
            price_diff = pos["current_price"] - pos["entry_price"]
            # P/L per unit = price_diff (for USD-quoted) or price_diff/current for JPY
            if pos["pip"] >= 0.01:  # JPY pair
                pl_per_unit = price_diff / pos["current_price"]
            else:
                pl_per_unit = price_diff
            unrealized_pl += pl_per_unit * pos["units"] * pos["direction"]

        nav = balance + unrealized_pl
        return self.update(nav)

    def dd_scale(self) -> float:
        """
        Compute DD-aware sizing multiplier (0.0–1.0).

        Uses the TIGHTER of daily and overall constraints.
        Linear taper from taper_start_frac of limit down to 0 at limit.

        Example with daily limit=2.75%, taper_start=50%:
          DD=0.00% → scale=1.0  (full sizing)
          DD=1.00% → scale=1.0  (below taper zone: 50% of 2.75 = 1.375%)
          DD=1.375% → scale=1.0 (taper starts here)
          DD=2.06% → scale=0.5  (halfway through taper)
          DD=2.75% → scale=0.0  (at limit, no new entries)
        """
        daily_scale = self._taper(
            self.daily_hwm, self.current_nav, self.max_daily_dd_pct
        )
        overall_scale = self._taper(
            self.overall_hwm, self.current_nav, self.max_overall_dd_pct
        )
        return min(daily_scale, overall_scale)

    def _taper(self, hwm: Optional[float], nav: Optional[float], limit_pct: float) -> float:
        """Compute taper scale for one DD dimension."""
        if hwm is None or nav is None or hwm <= 0:
            return 1.0
        dd_pct = self._dd_pct(hwm, nav)
        taper_start = limit_pct * self.taper_start_frac
        if dd_pct <= taper_start:
            return 1.0
        elif dd_pct >= limit_pct:
            return 0.0
        else:
            return (limit_pct - dd_pct) / (limit_pct - taper_start)

    def can_trade(self) -> bool:
        """Return False if DD scale is 0 — no new entries (but no force-close)."""
        return self.dd_scale() > 0.0

    def get_daily_dd_pct(self) -> float:
        """Current daily DD as % of start-of-day equity."""
        if self.daily_hwm and self.current_nav:
            return self._dd_pct(self.daily_hwm, self.current_nav)
        return 0.0

    def get_overall_dd_pct(self) -> float:
        """Current overall DD as % of peak equity."""
        if self.overall_hwm and self.current_nav:
            return self._dd_pct(self.overall_hwm, self.current_nav)
        return 0.0

    def get_daily_remaining_pct(self) -> float:
        """How much DD room is left today (% of daily HWM)."""
        return max(0.0, self.max_daily_dd_pct - self.get_daily_dd_pct())

    def get_overall_remaining_pct(self) -> float:
        """How much DD room is left overall (% of peak)."""
        return max(0.0, self.max_overall_dd_pct - self.get_overall_dd_pct())

    def get_status(self) -> dict:
        """Full status dict for dashboards and logging."""
        daily_dd = self.get_daily_dd_pct()
        overall_dd = self.get_overall_dd_pct()
        scale = self.dd_scale()
        return {
            "daily_dd_pct": round(daily_dd, 3),
            "daily_dd_limit_pct": self.max_daily_dd_pct,
            "daily_remaining_pct": round(self.get_daily_remaining_pct(), 3),
            "daily_hwm": self.daily_hwm,
            "overall_dd_pct": round(overall_dd, 3),
            "overall_dd_limit_pct": self.max_overall_dd_pct,
            "overall_remaining_pct": round(self.get_overall_remaining_pct(), 3),
            "overall_hwm": self.overall_hwm,
            "nav": self.current_nav,
            "dd_scale": round(scale, 4),
            "can_trade": scale > 0.0,
            "breach_count": len(self._breach_log),
        }

    @staticmethod
    def _dd_pct(hwm: float, nav: float) -> float:
        """Compute drawdown as percentage of HWM."""
        if hwm <= 0:
            return 0.0
        return max(0.0, (hwm - nav) / hwm * 100.0)

    def save_state(self) -> dict:
        """Serialize state for persistence across restarts."""
        return {
            "daily_hwm": self.daily_hwm,
            "overall_hwm": self.overall_hwm,
            "current_nav": self.current_nav,
            "last_reset_date": self._last_reset_date,
            "breach_log": self._breach_log[-20:],  # Keep last 20 breaches
        }

    def load_state(self, state: dict):
        """Restore state from persistence."""
        self.daily_hwm = state.get("daily_hwm")
        self.overall_hwm = state.get("overall_hwm")
        self.current_nav = state.get("current_nav")
        self._last_reset_date = state.get("last_reset_date")
        self._breach_log = state.get("breach_log", [])


# ─── Pip Value Calculation ──────────────────────────────────────────────────

def pip_value_usd(pip: float, entry_price: float) -> float:
    """
    Approximate dollar value per pip per unit.
    - XXX/USD pairs (pip=0.0001): pip_value ≈ 0.0001
    - XXX/JPY pairs (pip=0.01):   pip_value ≈ 0.01 / entry_price
    """
    if pip >= 0.01:  # JPY pairs
        return pip / entry_price if entry_price > 1 else 0.0001
    else:  # USD quote pairs
        return pip


# ─── Core Sizing Engine ────────────────────────────────────────────────────

class StrategyAllocator:
    """
    Position sizing engine for a single strategy.

    Usage:
        allocator = StrategyAllocator("impulse")
        units, debug = allocator.compute_units(
            pair="GBP_JPY",
            entry_price=188.50,
            risk_pips=12.5,
            direction=1,
            completed_trades=[...],
            open_positions={"EUR_JPY": 15, "USD_JPY": -10},
        )
    """

    def __init__(self, strategy_name: str, config_path: str = CONFIG_PATH):
        self.strategy_name = strategy_name
        self.config = load_config(config_path)
        self.global_cfg = self.config["global"]
        self.strategy_cfg = self.config["strategies"][strategy_name]
        self.equity = self.strategy_cfg.get("reference_equity", 10.0)
        self._reference_equity_from_broker = False  # Set on first refresh_equity()

        # Pre-compute Sharpe weights
        self._recompute_sharpe_weights()

        # Portfolio-level drawdown guard (throttle, not kill switch)
        self.dd_guard = DrawdownGuard(
            max_daily_dd_pct=self.global_cfg.get("max_daily_dd_pct", 2.75),
            max_overall_dd_pct=self.global_cfg.get("max_overall_dd_pct", 8.0),
            taper_start_frac=self.global_cfg.get("dd_taper_start_frac", 0.50),
        )

    def _recompute_sharpe_weights(self):
        """Compute normalized Sharpe weights across all pairs in this strategy."""
        pairs = self.strategy_cfg["pairs"]
        total = sum(p.get("backtest_sharpe", 1.0) for p in pairs.values())
        self._sharpe_weights = {
            pair: cfg.get("backtest_sharpe", 1.0) / total
            for pair, cfg in pairs.items()
        } if total > 0 else {pair: 1.0 / len(pairs) for pair in pairs}

    def refresh_equity(self, oanda_client) -> Optional[float]:
        """
        Probe OANDA sub-account NAV, margin, leverage via REST API.
        Updates stored equity + DD guard + margin metrics.

        Two refresh paths (use both):
          1. refresh_equity() — REST API call, authoritative, every 15 min + after closes
          2. update_nav_realtime() — tick-derived, no API call, call on every tick/bar
        """
        # Cache client reference so compute_units() can call refresh_margin()
        self._oanda_client = oanda_client
        try:
            info = oanda_client.get_account_summary()
            if info and hasattr(info, 'nav') and info.nav > 0:
                self.equity = info.nav
                self.balance = info.balance

                # Anchor reference_equity from broker on first successful call
                if not self._reference_equity_from_broker:
                    old_ref = self.strategy_cfg.get("reference_equity", 10.0)
                    self.strategy_cfg["reference_equity"] = self.equity
                    self._reference_equity_from_broker = True
                    logger.info(
                        f"[{self.strategy_name}] reference_equity set from broker: "
                        f"${self.equity:.2f} (was ${old_ref:.2f} in config)"
                    )

                # Margin & leverage metrics from OANDA
                self._update_margin_from_info(info)

                dd_status = self.dd_guard.update(self.equity)
                logger.info(
                    f"[{self.strategy_name}] Equity=${self.equity:.2f} "
                    f"margin_used=${self.margin_used:.2f} "
                    f"margin_util={self.margin_utilization_pct:.1f}% "
                    f"daily_dd={dd_status['daily_dd_pct']:.2f}%/{self.dd_guard.max_daily_dd_pct}% "
                    f"overall_dd={dd_status['overall_dd_pct']:.2f}%/{self.dd_guard.max_overall_dd_pct}% "
                    f"dd_scale={dd_status['dd_scale']:.2f}"
                )
                return self.equity
        except Exception as e:
            logger.warning(f"[{self.strategy_name}] Equity refresh failed: {e}")
        return None

    def _update_margin_from_info(self, info):
        """Update margin metrics from AccountInfo dataclass."""
        self.margin_used = getattr(info, 'margin_used', 0.0)
        self.margin_available = getattr(info, 'margin_available', self.equity)
        self.margin_rate = 0.0  # Not available from AccountInfo
        self.open_trade_count = getattr(info, 'open_trade_count', 0)

        if self.equity > 0:
            self.margin_utilization_pct = (self.margin_used / self.equity) * 100.0
        else:
            self.margin_utilization_pct = 0.0

    def refresh_margin(self, oanda_client) -> Optional[float]:
        """
        Real-time margin check — call immediately before opening a trade.

        Lighter than refresh_equity(): only updates margin_used/available,
        does NOT update equity, DD guard, or reference_equity.
        Solves the thundering-herd race: 12 services sharing one account
        each read CURRENT margin right before entry, not a 15-min-old snapshot.
        """
        try:
            info = oanda_client.get_account_summary()
            if info and hasattr(info, 'nav') and info.nav > 0:
                self._update_margin_from_info(info)
                logger.info(
                    f"[{self.strategy_name}] Pre-trade margin refresh: "
                    f"used=${self.margin_used:.2f} "
                    f"util={self.margin_utilization_pct:.1f}%"
                )
                return self.margin_used
        except Exception as e:
            logger.warning(f"[{self.strategy_name}] Pre-trade margin refresh failed: {e}")
        return None

    def update_nav_realtime(self, open_positions: List[dict]) -> dict:
        """
        Real-time NAV update from tick prices — no API call.
        Call this from the tick stream thread on every price update.

        Each position dict: {"pair": str, "direction": int, "units": int,
                             "entry_price": float, "current_price": float, "pip": float}

        Uses stored balance (from last refresh_equity) + computed unrealized P/L.
        Returns DD guard status dict.
        """
        balance = getattr(self, "balance", self.equity)
        dd_status = self.dd_guard.update_from_positions(balance, open_positions)

        # Update equity estimate (balance + unrealized)
        self.equity = dd_status.get("nav", balance)

        return dd_status

    def get_risk_pips(self, pair: str, atr: Optional[float] = None) -> float:
        """
        Get risk measure in pips based on strategy's risk_source.
        - "sl": caller must provide ATR, risk = SL_ATR_MULT × ATR / pip
        - "p95_mae": uses static value from config
        """
        risk_source = self.strategy_cfg.get("risk_source", "sl")
        pair_cfg = self.strategy_cfg["pairs"].get(pair, {})

        if risk_source == "p95_mae":
            return self.strategy_cfg.get("p95_mae_pips", 35.0)
        elif risk_source == "sl":
            if atr is None:
                raise ValueError(f"ATR required for risk_source='sl' (strategy={self.strategy_name})")
            pip = pair_cfg.get("pip", 0.0001)
            sl_atr_mult = self.strategy_cfg.get("sl_atr_mult", 2.5)
            return sl_atr_mult * atr / pip
        else:
            raise ValueError(f"Unknown risk_source: {risk_source}")

    def compute_units(
        self,
        pair: str,
        entry_price: float,
        risk_pips: float,
        direction: int,
        completed_trades: Optional[List[dict]] = None,
        open_positions: Optional[Dict[str, float]] = None,
        live_sharpe: Optional[float] = None,
        n_live_trades: int = 0,
        active_pair_weights: Optional[Dict[str, float]] = None,
        regime_rvol: Optional[float] = None,
    ) -> Tuple[int, dict]:
        """
        Compute position size using Bandy risk-fraction model.

        Core formula (scales identically from $1 to $100K):
            units = (equity × risk_pct) / (risk_pips × pip_value_usd)

        Then constrained by:
            1. Performance gate (throttle on recent losing streak)
            2. Margin gate (OANDA 50:1 leverage, cap at 60% utilization)
            3. Currency exposure cap (limit correlated positions)
            4. DD guard (throttle on daily/overall drawdown)

        Args:
            pair: Instrument name (e.g., "GBP_JPY")
            entry_price: Current price for pip value calculation
            risk_pips: Risk in pips (SL distance or P95 MAE)
            direction: +1 long, -1 short
            completed_trades: List of trade dicts with 'pnl_pips' key (for perf gate)
            open_positions: Dict of {pair: units} for currency exposure cap

        Returns:
            (units, debug_dict) — units=0 means skip trade
        """
        pair_cfg = self.strategy_cfg["pairs"].get(pair, {})
        pip = pair_cfg.get("pip", 0.0001)

        equity = self.equity
        # risk_pct: max risk per trade as % of equity (Bandy fraction).
        risk_pct = self.strategy_cfg.get("risk_pct",
                       self.global_cfg.get("risk_pct", 2.0)) / 100.0
        debug = {
            "pair": pair, "equity": equity, "risk_pips": risk_pips,
            "direction": direction, "strategy": self.strategy_name,
            "risk_pct": risk_pct * 100,
        }

        # --- Step 1: Performance Gate (ExpR-based) ---
        perf_scale, expr_val, perf_n = compute_perf_gate(
            completed_trades or [], self.global_cfg
        )
        debug["perf_gate"] = {"scale": perf_scale, "expr": expr_val, "n_trades": perf_n}

        if perf_scale <= 0.0 and perf_n >= self.global_cfg.get("perf_gate_min_trades", 5):
            debug["units"] = 0
            debug["reason"] = "perf_gate_killed"
            return 0, debug

        # --- Step 2: Bandy risk-fraction sizing ---
        # units = (equity × risk_pct) / (risk_pips × pip_value_usd_per_unit)
        pv = pip_value_usd(pip, entry_price)
        risk_per_unit = risk_pips * pv
        if risk_per_unit > 0:
            units = (equity * risk_pct) / risk_per_unit
        else:
            units = 0
        debug["bandy"] = {
            "risk_dollars": round(equity * risk_pct, 4),
            "risk_per_unit": round(risk_per_unit, 8),
            "units_raw": round(units, 1),
        }

        # Apply performance gate scale
        units = units * perf_scale
        debug["post_perf_gate"] = round(units, 1)

        # --- Step 3: Margin Gate (FCFS with real-time margin) ---
        # Before checking headroom, refresh margin from OANDA so we see
        # positions opened by OTHER services on the same account.
        # This eliminates the thundering-herd race condition where 12
        # services all read a stale 15-min-old margin_used snapshot.
        if hasattr(self, '_oanda_client') and self._oanda_client is not None:
            self.refresh_margin(self._oanda_client)

        max_margin_util = self.global_cfg.get("max_margin_util_pct", 80.0) / 100.0
        margin_share = self.global_cfg.get("margin_share_pct", 33.0) / 100.0
        margin_rate = self.global_cfg.get("margin_rate", 0.02)
        margin_used = getattr(self, "margin_used", 0.0)

        # Margin per unit: notional_usd × margin_rate
        if pip >= 0.01:  # JPY pair
            margin_per_unit = margin_rate  # ~$0.02/unit (notional ≈ 1 USD)
        else:  # USD-quoted pairs
            margin_per_unit = entry_price * margin_rate

        max_margin_dollars = equity * max_margin_util
        margin_headroom = max(0.0, max_margin_dollars - margin_used)

        # Tiered FCFS: larger initial positions, floor for later ones.
        # margin_share of remaining headroom, but at least margin_floor_pct of max_margin.
        # This ensures the 12th pair still gets a tradeable size (~5u on $5 account)
        # while the 1st pair gets a meaningful position for visible P/L.
        margin_floor_pct = self.global_cfg.get("margin_floor_pct", 3.0) / 100.0
        margin_floor = max_margin_dollars * margin_floor_pct
        margin_for_this_trade = max(margin_headroom * margin_share, margin_floor)
        # But never exceed actual headroom
        margin_for_this_trade = min(margin_for_this_trade, margin_headroom)
        units_from_margin = margin_for_this_trade / margin_per_unit if margin_per_unit > 0 else float("inf")

        units = min(units, units_from_margin)
        debug["margin_gate"] = {
            "margin_used": round(margin_used, 4),
            "max_margin": round(max_margin_dollars, 4),
            "headroom_units": round(margin_headroom / margin_per_unit if margin_per_unit > 0 else 0, 1),
            "share_units": round(units_from_margin, 1),
            "binding": units_from_margin < debug["post_perf_gate"],
        }

        if units_from_margin <= 0:
            debug["units"] = 0
            debug["reason"] = "margin_limit_reached"
            logger.warning(
                f"[{self.strategy_name}] MARGIN GATE: {pair} blocked — "
                f"margin_used=${margin_used:.2f}/{max_margin_dollars:.2f}"
            )
            return 0, debug

        # --- Step 4: Currency Exposure Cap ---
        # Limit total RISK (not notional) to any single currency.
        # currency_risk_cap_pct of equity = max risk-dollars across all positions in that currency.
        # E.g., 5% cap at 1% risk/trade → max 5 concurrent trades per currency.
        currency_risk_cap = self.global_cfg.get("currency_risk_cap_pct", 5.0) / 100.0
        open_pos = open_positions or {}
        pair_currencies = self._get_pair_currencies(pair)
        if pair_currencies and currency_risk_cap > 0:
            max_cur_risk = equity * currency_risk_cap
            # This trade's risk in dollars
            this_trade_risk = units * risk_per_unit  # risk_per_unit from Step 2

            for cur in pair_currencies:
                # Sum risk-dollars already committed to this currency
                cur_risk = 0.0
                for p, p_units in open_pos.items():
                    p_curs = self._get_pair_currencies(p)
                    if p_curs and cur in p_curs:
                        # Estimate risk for existing position (same risk_pips assumption)
                        p_pip = self.strategy_cfg["pairs"].get(p, {}).get("pip", 0.0001)
                        p_pv = pip_value_usd(p_pip, entry_price)  # approx, uses current price
                        cur_risk += abs(p_units) * risk_pips * p_pv

                remaining_risk = max(0, max_cur_risk - cur_risk)
                if risk_per_unit > 0:
                    max_units_for_cur = remaining_risk / risk_per_unit
                    units = min(units, max_units_for_cur)

            debug["currency_cap"] = {
                "max_risk_per_currency": round(max_cur_risk, 4),
                "cap_pct": currency_risk_cap * 100,
            }

        # --- Step 5: DD Guard Throttle ---
        dd_scale = self.dd_guard.dd_scale()
        units = units * dd_scale
        debug["dd_guard"] = {
            "dd_scale": dd_scale,
            "daily_dd_pct": self.dd_guard.get_daily_dd_pct(),
            "overall_dd_pct": self.dd_guard.get_overall_dd_pct(),
        }

        if dd_scale <= 0.0:
            debug["units"] = 0
            debug["reason"] = "dd_limit_reached"
            return 0, debug

        # --- Step 5b: Regime-Aware Volatility Scaler ---
        # Scale position size based on realized volatility regime.
        # Research shows: volatile markets = +0.93p/bar edge, quiet = +0.50p/bar.
        # Scale up in volatile (more edge to capture), scale down in quiet (less edge, more noise).
        # regime_rvol: annualized realized vol from rolling 60-bar window (passed by strategy)
        regime_scale = 1.0
        if regime_rvol is not None and regime_rvol > 0:
            # Calibrated from backtest: median rvol ~0.08, 25th ~0.05, 75th ~0.12
            rvol_low = self.global_cfg.get("regime_rvol_low", 0.05)
            rvol_high = self.global_cfg.get("regime_rvol_high", 0.12)
            scale_min = self.global_cfg.get("regime_scale_min", 0.5)
            scale_max = self.global_cfg.get("regime_scale_max", 1.5)

            if regime_rvol <= rvol_low:
                regime_scale = scale_min  # Quiet regime: half size
            elif regime_rvol >= rvol_high:
                regime_scale = scale_max  # Volatile regime: 1.5x size
            else:
                # Linear interpolation between low and high
                frac = (regime_rvol - rvol_low) / (rvol_high - rvol_low)
                regime_scale = scale_min + frac * (scale_max - scale_min)

            units = units * regime_scale

        debug["regime_scaler"] = {
            "rvol": round(regime_rvol, 4) if regime_rvol else None,
            "scale": round(regime_scale, 2),
        }

        # --- Step 6: Floor & Round ---
        units = max(1, int(units)) if perf_scale > 0 else 0
        debug["units"] = units
        return units, debug

    def _get_pair_currencies(self, pair: str) -> Optional[Tuple[str, str]]:
        """Extract base and quote currencies from pair name (e.g., 'GBP_JPY' → ('GBP', 'JPY'))."""
        parts = pair.split("_")
        if len(parts) == 2:
            return (parts[0], parts[1])
        return None

    def get_all_pair_weights(
        self,
        pair_trades: Optional[Dict[str, List[dict]]] = None,
    ) -> Dict[str, float]:
        """
        Compute redistributed Sharpe weights accounting for gated pairs.
        Gated pairs' weight is redistributed proportionally to active pairs.

        Args:
            pair_trades: Dict of {pair: [trade_dicts]} for perf gate evaluation

        Returns:
            Dict of {pair: adjusted_weight}
        """
        pair_trades = pair_trades or {}
        active_weights = {}
        total_active = 0.0

        for pair in self.strategy_cfg["pairs"]:
            base_weight = self._sharpe_weights.get(pair, 0.0)
            trades = pair_trades.get(pair, [])
            scale, _, _ = compute_perf_gate(trades, self.global_cfg)

            if scale > 0:
                active_weights[pair] = base_weight * scale
                total_active += base_weight * scale

        # Redistribute to sum to 1.0
        if total_active > 0:
            return {p: w / total_active for p, w in active_weights.items()}
        return self._sharpe_weights.copy()
