#!/usr/bin/env python3
"""
FX-Core Portfolio Manager — centralized allocation + account monitoring.

Responsibilities:
  1. Poll all 10 OANDA accounts every 15 seconds
  2. Read closed trades from DuckDB for Sharpe computation
  3. Compute Sharpe-weighted allocation per strategy per pair
  4. Enforce cross-account currency exposure limits
  5. Publish allocation updates via ZMQ
  6. Write account summaries + allocation weights to DuckDB
"""

import os
import sys
import time
import math
import signal
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from lib.pair_config import ALL_PAIR_NAMES, CURRENCIES
from lib.oanda_adapter import OANDAAdapter
from lib.broker_adapter import AccountInfo
from lib.zmq_protocol import Publisher, ALLOCATION_PUB, make_topic, MSG_ALLOCATION, MSG_RISK_OVERLAY, MSG_PORTFOLIO_STATE
from lib.db import get_trades_db, init_trades_schema, TradeDBWriter

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [portfolio] %(message)s",
)
logger = logging.getLogger("portfolio")


# Account configuration — from centralized registry
from lib.strategy_registry import STRATEGIES

# Build ACCOUNTS list in portfolio_mgr's expected format
ACCOUNTS = [
    {"label": s["account"], "env": s["env_var"], "name": s["label"],
     "strategy": s["strategy_name"]}
    for s in STRATEGIES
]

# Sharpe computation parameters
SHARPE_LOOKBACK_DAYS = 30       # Rolling window for live Sharpe
SHARPE_MIN_TRADES = 5           # Minimum trades to compute Sharpe (else use backtest)
MAX_CURRENCY_EXPOSURE_PCT = 30  # Max % of total NAV exposed to any single currency

# Global risk overlay parameters
RISK_MAX_CURRENCY_PCT = int(os.environ.get("RISK_MAX_CURRENCY_PCT", "60"))
RISK_MAX_CONCENTRATION = int(os.environ.get("RISK_MAX_CONCENTRATION", "4"))
RISK_MAX_POSITIONS = int(os.environ.get("RISK_MAX_POSITIONS", "60"))
RISK_GLOBAL_DD_DAILY_PCT = float(os.environ.get("RISK_GLOBAL_DD_DAILY", "5"))
RISK_PUBLISH_INTERVAL = int(os.environ.get("RISK_PUBLISH_INTERVAL", "15"))

# Per-currency exposure overrides (tighter caps for correlated currencies)
# JPY is in 7/12 pairs — needs tighter cap to prevent correlated drawdowns
_CURRENCY_CAPS_RAW = os.environ.get("RISK_CURRENCY_CAPS", "")  # e.g. "JPY=25,GBP=50"
RISK_CURRENCY_CAPS: Dict[str, int] = {}
for _entry in _CURRENCY_CAPS_RAW.split(","):
    if "=" in _entry:
        _cur, _val = _entry.split("=", 1)
        RISK_CURRENCY_CAPS[_cur.strip().upper()] = int(_val.strip())
# Default JPY cap if not specified
if "JPY" not in RISK_CURRENCY_CAPS:
    RISK_CURRENCY_CAPS["JPY"] = int(os.environ.get("RISK_JPY_CAP_PCT", "25"))


class PortfolioManager:
    """Centralized portfolio allocation and risk management."""

    def __init__(self):
        self._shutdown = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # OANDA adapters per account
        self.adapters: Dict[str, OANDAAdapter] = {}
        for acct in ACCOUNTS:
            acct_id = os.environ.get(acct["env"])
            if acct_id:
                self.adapters[acct["label"]] = OANDAAdapter(account_id=acct_id)

        if not self.adapters:
            logger.warning("No OANDA accounts configured — check env vars")

        # ZMQ publisher
        self.publisher = Publisher(ALLOCATION_PUB)

        # DuckDB — portfolio_mgr is the SOLE writer to trades.duckdb.
        # Strategies send trade records via ZMQ PUSH → our PULL socket.
        db = get_trades_db(read_only=False)
        init_trades_schema(db)
        db.close()

        # Start ZMQ trade-write proxy (receives trades from strategies)
        self.trade_writer = TradeDBWriter()
        self.trade_writer.start()

        # State
        # USD/JPY price tracking for exposure calculation
        self._usd_jpy_price = 158.0  # Default fallback
        self._last_usd_jpy_update = 0
        self.account_state: Dict[str, AccountInfo] = {}

        # Global risk overlay state
        self._nav_hwm_daily = 0.0  # Daily high-water mark for global DD
        self._nav_hwm_daily_reset = 0  # UTC hour of last reset

        # Cached state for portfolio_state publishing
        self._cached_positions = {}  # {label: [TradeInfo, ...]}
        self._cached_recent_trades = []  # Last 20 trades from DuckDB
        self._last_trade_cache = 0  # timestamp of last trade cache refresh

        logger.info(f"Portfolio Manager: {len(self.adapters)} accounts configured")

    def _signal_handler(self, signum, frame):
        logger.info(f"Signal {signum}, shutting down...")
        self._shutdown = True

    def _poll_accounts(self):
        """Poll all OANDA accounts for NAV/margin/positions."""
        total_nav = 0.0
        total_margin = 0.0
        now = str(datetime.now(timezone.utc))

        rows = []
        for acct in ACCOUNTS:
            label = acct["label"]
            adapter = self.adapters.get(label)
            if not adapter:
                continue

            info = adapter.get_account_summary()
            if info:
                self.account_state[label] = info
                total_nav += info.nav
                total_margin += info.margin_used
                rows.append([info.account_id, now, info.nav, info.balance,
                            info.unrealized_pl, info.margin_used,
                            info.margin_available, info.open_trade_count])

        if rows:
            try:
                db = get_trades_db(read_only=False)
                db.begin()
                for row in rows:
                    db.execute(
                        """INSERT OR REPLACE INTO account_summary
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", row)
                db.commit()
                db.close()
            except Exception as e:
                try:
                    db.rollback()
                    db.close()
                except Exception:
                    pass
                logger.warning(f"DB write failed: {e}")

        logger.info(f"Accounts polled: total NAV=${total_nav:.2f}, "
                   f"margin=${total_margin:.2f}, "
                   f"{sum(1 for a in self.account_state.values() if a.open_trade_count > 0)} "
                   f"accounts with positions")

    def _compute_live_sharpe(self, strategy: str, pair: Optional[str] = None) -> Optional[float]:
        """Compute rolling Sharpe ratio from closed trades in DuckDB.

        Returns Sharpe (annualized, daily basis) or None if insufficient data.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=SHARPE_LOOKBACK_DAYS)
        db = None
        try:
            db = get_trades_db(read_only=True)
            if pair:
                rows = db.execute(
                    "SELECT pnl_pips FROM trades WHERE strategy = ? AND pair = ? "
                    "AND exit_time >= ? ORDER BY exit_time",
                    [strategy, pair, str(cutoff)]
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT pnl_pips FROM trades WHERE strategy = ? "
                    "AND exit_time >= ? ORDER BY exit_time",
                    [strategy, str(cutoff)]
                ).fetchall()
            db.close()

            pnls = [r[0] for r in rows if r[0] is not None]
            if len(pnls) < SHARPE_MIN_TRADES:
                return None

            mean = sum(pnls) / len(pnls)
            variance = sum((p - mean) ** 2 for p in pnls) / len(pnls)
            std = math.sqrt(variance) if variance > 0 else 1e-6

            # Annualize: assume ~1 trade/day avg, sqrt(252) for daily→annual
            sharpe = (mean / std) * math.sqrt(252)
            return round(sharpe, 2)

        except Exception as e:
            if db:
                try:
                    db.close()
                except Exception:
                    pass
            logger.debug(f"Sharpe computation failed for {strategy}/{pair}: {e}")
            return None


    def _update_usd_jpy_price(self):
        """Update cached USD/JPY price every 5 minutes."""
        import time
        now = time.time()
        if now - self._last_usd_jpy_update < 300:  # 5 minutes
            return
        
        try:
            adapter = list(self.adapters.values())[0]
            candles = adapter.get_candles("USD_JPY", count=1, granularity="S5")
            if candles:
                self._usd_jpy_price = candles[-1]["close"]
                self._last_usd_jpy_update = now
                logger.info(f"Updated USD/JPY price: {self._usd_jpy_price}")
        except Exception as e:
            logger.warning(f"Failed to update USD/JPY price: {e}, using {self._usd_jpy_price}")

    def _get_currency_exposure(self) -> Dict[str, float]:
        """Compute net currency exposure across all accounts.

        Returns dict of {currency: net_units_exposure}.
        """
        exposure: Dict[str, float] = {c: 0.0 for c in CURRENCIES}

        for label, adapter in self.adapters.items():
            try:
                trades = adapter.get_open_trades()
                for t in trades:
                    pair = t.instrument
                    parts = pair.split("_")
                    if len(parts) != 2:
                        continue
                    base, quote = parts
                    notional = abs(t.units) * t.entry_price
                    
                    # Convert JPY notionals to USD
                    if quote == "JPY" and self._usd_jpy_price:
                        notional = notional / self._usd_jpy_price
                    if base in exposure:
                        exposure[base] += notional if t.units > 0 else -notional
                    if quote in exposure:
                        exposure[quote] += -notional if t.units > 0 else notional
            except Exception as e:
                logger.debug(f"Exposure fetch failed for {label}: {e}")

        return exposure

    def _compute_risk_overlay(self):
        """Compute global risk overlay and publish via ZMQ.

        Checks:
          1. Net currency exposure (% of total NAV per currency)
          2. Pair concentration (how many accounts hold same pair+direction)
          3. Total open position count
          4. Portfolio-wide daily drawdown
        """
        total_nav = sum(info.nav for info in self.account_state.values())
        if total_nav <= 0:
            return

        # ── 1. Currency exposure ──
        currency_exposure = self._get_currency_exposure()
        currency_pcts = {}
        for cur, notional in currency_exposure.items():
            currency_pcts[cur] = round(notional / total_nav * 100, 1)

        # ── 2. Pair concentration + total positions ──
        pair_direction_counts: Dict[str, Dict[str, int]] = {}  # {pair: {"long": N, "short": N}}
        total_positions = 0

        for label, adapter in self.adapters.items():
            try:
                trades = adapter.get_open_trades()
                self._cached_positions[label] = trades  # Cache for portfolio_state
                for t in trades:
                    total_positions += 1
                    pair = t.instrument
                    direction = "long" if t.units > 0 else "short"
                    if pair not in pair_direction_counts:
                        pair_direction_counts[pair] = {"long": 0, "short": 0}
                    pair_direction_counts[pair][direction] += 1
            except Exception:
                pass

        # Determine blocked pairs (concentration too high)
        blocked_pairs = {}
        for pair, counts in pair_direction_counts.items():
            for direction, count in counts.items():
                if count >= RISK_MAX_CONCENTRATION:
                    blocked_pairs[f"{pair}_{direction}"] = f"concentration_{count}_accounts"

        # Block pairs where currency exposure exceeds limit
        # Use per-currency caps (e.g., JPY=25%) falling back to global default
        for pair in ALL_PAIR_NAMES:
            parts = pair.split("_")
            if len(parts) == 2:
                base, quote = parts
                for cur in (base, quote):
                    pct = abs(currency_pcts.get(cur, 0))
                    cap = RISK_CURRENCY_CAPS.get(cur, RISK_MAX_CURRENCY_PCT)
                    if pct > cap:
                        # Block the direction that would INCREASE exposure
                        if currency_pcts.get(cur, 0) > 0:
                            key = f"{pair}_long" if cur == base else f"{pair}_short"
                        else:
                            key = f"{pair}_short" if cur == base else f"{pair}_long"
                        if key not in blocked_pairs:
                            blocked_pairs[key] = f"{cur}_exposure_{pct:.0f}%>{cap}%"

        # ── 3. Global DD ──
        now_hour = datetime.now(timezone.utc).hour
        if now_hour != self._nav_hwm_daily_reset:
            # Reset daily HWM at start of new UTC hour (approximate daily reset)
            if now_hour == 0:
                self._nav_hwm_daily = total_nav
            self._nav_hwm_daily_reset = now_hour

        if total_nav > self._nav_hwm_daily:
            self._nav_hwm_daily = total_nav

        daily_dd_pct = (self._nav_hwm_daily - total_nav) / self._nav_hwm_daily * 100 if self._nav_hwm_daily > 0 else 0
        # Taper: 0-50% of limit = full, 50-100% = linear taper, >100% = blocked
        dd_ratio = daily_dd_pct / RISK_GLOBAL_DD_DAILY_PCT if RISK_GLOBAL_DD_DAILY_PCT > 0 else 0
        if dd_ratio <= 0.5:
            global_dd_scale = 1.0
        elif dd_ratio >= 1.0:
            global_dd_scale = 0.0
        else:
            global_dd_scale = round(1.0 - (dd_ratio - 0.5) * 2.0, 2)

        # ── Publish ──
        overlay = {
            "type": MSG_RISK_OVERLAY,
            "ts": str(datetime.now(timezone.utc)),
            "blocked_pairs": blocked_pairs,
            "global_dd_scale": global_dd_scale,
            "daily_dd_pct": round(daily_dd_pct, 2),
            "total_positions": total_positions,
            "max_positions": RISK_MAX_POSITIONS,
            "total_nav": round(total_nav, 2),
            "currency_exposure_pct": currency_pcts,
        }
        self.publisher.publish(make_topic(MSG_RISK_OVERLAY), overlay)

        # Log if anything is blocked
        if blocked_pairs or total_positions >= RISK_MAX_POSITIONS or global_dd_scale < 1.0:
            logger.info(f"RISK: pos={total_positions}/{RISK_MAX_POSITIONS} "
                       f"dd_scale={global_dd_scale} blocks={len(blocked_pairs)} "
                       f"dd={daily_dd_pct:.1f}%/{RISK_GLOBAL_DD_DAILY_PCT}%")

    def _compute_allocation(self):
        """Compute Sharpe-weighted allocation and publish via ZMQ.

        Weight formula per strategy per pair:
          1. Compute live Sharpe (30-day rolling) for each (strategy, pair)
          2. If insufficient trades, fall back to backtest Sharpe from sizing_config.json
          3. Normalize: weight = max(sharpe, 0) / sum(all positive sharpes)
          4. Apply currency exposure cap: block pair if currency > 30% of total NAV
          5. Publish weights for strategies to read via ZMQ
        """
        total_nav = sum(info.nav for info in self.account_state.values())
        currency_exposure = self._get_currency_exposure()

        # Load backtest Sharpe from config as fallback
        import json
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "lib", "sizing_config.json")
        backtest_sharpes = {}
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            for strategy_name, scfg in cfg.get("strategies", {}).items():
                for pair, pcfg in scfg.get("pairs", {}).items():
                    backtest_sharpes[(strategy_name, pair)] = pcfg.get("backtest_sharpe", 1.0)
        except Exception as e:
            logger.debug(f"Config load failed: {e}")

        weights = {}
        now_str = str(datetime.now(timezone.utc))

        for acct in ACCOUNTS:
            strategy = acct["strategy"]
            label = acct["label"]
            info = self.account_state.get(label)
            acct_nav = info.nav if info else 0

            pair_weights = {}
            raw_sharpes = {}

            for pair in ALL_PAIR_NAMES:
                # Try live Sharpe first, fall back to backtest
                live_sharpe = self._compute_live_sharpe(strategy, pair)
                if live_sharpe is not None:
                    raw_sharpes[pair] = live_sharpe
                else:
                    raw_sharpes[pair] = backtest_sharpes.get((strategy, pair), 1.0)

            # Normalize: positive Sharpes only, proportional weights
            positive_sharpes = {p: max(s, 0.01) for p, s in raw_sharpes.items()}
            total_sharpe = sum(positive_sharpes.values())

            for pair in ALL_PAIR_NAMES:
                weight = positive_sharpes.get(pair, 0) / total_sharpe if total_sharpe > 0 else 1.0 / len(ALL_PAIR_NAMES)

                # Currency exposure cap
                parts = pair.split("_")
                blocked = False
                block_reason = ""
                if len(parts) == 2 and total_nav > 0:
                    base, quote = parts
                    for cur in (base, quote):
                        cur_pct = abs(currency_exposure.get(cur, 0)) / total_nav * 100 if total_nav > 0 else 0
                        if cur_pct > MAX_CURRENCY_EXPOSURE_PCT:
                            blocked = True
                            block_reason = f"{cur} exposure {cur_pct:.0f}% > {MAX_CURRENCY_EXPOSURE_PCT}%"

                # Margin budget: account's margin_available * weight
                margin_avail = info.margin_available if info else 0
                margin_budget = margin_avail * weight

                pair_weights[pair] = {
                    "weight": round(weight, 4),
                    "sharpe": round(raw_sharpes.get(pair, 0), 2),
                    "margin_budget": round(margin_budget, 4),
                    "blocked": blocked,
                    "block_reason": block_reason,
                }

                pair_weights[pair]["_db_row"] = [
                    strategy, pair, now_str, weight, margin_budget,
                    1.0, 1.0, blocked, block_reason
                ]

            weights[strategy] = pair_weights

        # Batch-write allocation weights to DuckDB (brief connection)
        try:
            db = get_trades_db(read_only=False)
            db.begin()
            for strategy_weights in weights.values():
                for pw in strategy_weights.values():
                    row = pw.pop("_db_row", None)
                    if row:
                        db.execute(
                            """INSERT OR REPLACE INTO allocation_weights
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", row)
            db.commit()
            db.close()
        except Exception as e:
            try:
                db.rollback()
                db.close()
            except Exception:
                pass
            logger.warning(f"Allocation DB write failed: {e}")

        # Publish via ZMQ
        self.publisher.publish(make_topic(MSG_ALLOCATION), {
            "type": MSG_ALLOCATION,
            "ts": now_str,
            "weights": weights,
        })

        # Log summary
        for acct in ACCOUNTS:
            strategy = acct["strategy"]
            pw = weights.get(strategy, {})
            top_pair = max(pw.items(), key=lambda x: x[1]["weight"], default=("?", {"weight": 0}))
            blocked_count = sum(1 for p in pw.values() if p["blocked"])
            logger.info(f"  {acct['label']} {acct['name']}: top={top_pair[0]} "
                       f"w={top_pair[1]['weight']:.3f} blocked={blocked_count}")

    def _publish_portfolio_state(self):
        """Publish complete portfolio snapshot via ZMQ for dashboard/telegram."""
        now_str = str(datetime.now(timezone.utc))
        total_nav = 0.0
        total_upl = 0.0
        total_positions = 0
        accounts = {}

        # Build registry lookup
        registry = {s["account"]: s for s in STRATEGIES}

        for acct in ACCOUNTS:
            label = acct["label"]
            info = self.account_state.get(label)
            if not info:
                continue
            total_nav += info.nav
            total_upl += info.unrealized_pl
            total_positions += info.open_trade_count

            reg = registry.get(label, {})
            positions = []
            for t in self._cached_positions.get(label, []):
                positions.append({
                    "pair": t.instrument, "direction": 1 if t.units > 0 else -1,
                    "units": abs(t.units), "upl": t.unrealized_pl,
                    "entry": t.entry_price,
                })

            accounts[label] = {
                "label": reg.get("label", label),
                "strategy": reg.get("strategy_name", ""),
                "enabled": reg.get("enabled", False),
                "nav": round(info.nav, 2),
                "balance": round(info.balance, 2),
                "upl": round(info.unrealized_pl, 4),
                "margin_pct": round(info.margin_used / info.nav * 100, 1) if info.nav > 0 else 0,
                "open_count": info.open_trade_count,
                "positions": positions,
            }

        # Cache recent trades from DuckDB (refresh every 60s)
        now = time.time()
        if now - self._last_trade_cache >= 60:
            try:
                db = get_trades_db(read_only=True)
                rows = db.execute(
                    "SELECT strategy, pair, account_id, pnl_pips, exit_reason, "
                    "mfe_pips, mae_pips, exit_time FROM trades "
                    "ORDER BY exit_time DESC LIMIT 20"
                ).fetchall()
                db.close()
                self._cached_recent_trades = [
                    {"strategy": r[0], "pair": r[1], "account": r[2],
                     "pnl_pips": round(r[3], 1) if r[3] else 0,
                     "exit_reason": r[4], "mfe": round(r[5], 1) if r[5] else 0,
                     "mae": round(r[6], 1) if r[6] else 0, "ts": str(r[7]) if r[7] else ""}
                    for r in rows
                ]
            except Exception:
                pass
            self._last_trade_cache = now

        state = {
            "type": MSG_PORTFOLIO_STATE,
            "ts": now_str,
            "total_nav": round(total_nav, 2),
            "total_upl": round(total_upl, 4),
            "total_positions": total_positions,
            "accounts": accounts,
            "recent_trades": self._cached_recent_trades,
        }
        self.publisher.publish(make_topic(MSG_PORTFOLIO_STATE), state)

    def run(self):
        """Main loop — poll accounts, compute allocation, publish."""
        logger.info("Portfolio Manager starting...")

        poll_interval = 15  # seconds
        alloc_interval = 60  # seconds
        last_alloc = 0

        while not self._shutdown:
            try:
                self._update_usd_jpy_price()
                self._poll_accounts()
                self._compute_risk_overlay()
                self._publish_portfolio_state()

                now = time.time()
                if now - last_alloc >= alloc_interval:
                    self._compute_allocation()
                    last_alloc = now

            except Exception as e:
                logger.error(f"Poll error: {e}", exc_info=True)

            # Sleep with shutdown check
            for _ in range(poll_interval):
                if self._shutdown:
                    break
                time.sleep(1)

        self.trade_writer.stop()
        self.publisher.close()
        logger.info("Portfolio Manager shutdown complete.")


def main():
    mgr = PortfolioManager()
    mgr.run()


if __name__ == "__main__":
    main()
