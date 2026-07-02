"""
DuckDB helpers — schema creation, read/write, connection management.

Two database files:
  - fx.duckdb:     Market data (candles, indicators, P&F boxes, Kalman).
                   Written exclusively by the curator. Read by all.
  - trades.duckdb: Trading state (positions, trades, accounts, allocations).
                   Primary writer: portfolio-mgr via ZMQ PUSH/PULL.
                   Paper services write directly via write_trade_direct() (infrequent,
                   serialized by DuckDB's internal lock — safe for a few writes/hour).

DuckDB supports concurrent readers in WAL mode. Single writer per file at a time.
"""

import duckdb
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import msgpack
import zmq

logger = logging.getLogger(__name__)

# Default database directory (Docker volume mount)
DB_DIR = os.environ.get("FX_DB_DIR", "/data/db")

FX_DB_PATH = os.path.join(DB_DIR, "fx.duckdb")
TRADES_DB_PATH = os.path.join(DB_DIR, "trades.duckdb")


def get_fx_db(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Get connection to the market data database.

    Args:
        read_only: True for strategies/dashboard/telegram. False only for curator.
    """
    Path(DB_DIR).mkdir(parents=True, exist_ok=True)
    return duckdb.connect(FX_DB_PATH, read_only=read_only)


def get_trades_db(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Get connection to the trades database.

    Args:
        read_only: True for dashboard/telegram. False for strategies/portfolio-mgr.
    """
    Path(DB_DIR).mkdir(parents=True, exist_ok=True)
    return duckdb.connect(TRADES_DB_PATH, read_only=read_only)


def init_fx_schema(conn: Optional[duckdb.DuckDBPyConnection] = None):
    """Create market data tables if they don't exist. Called by curator on startup."""
    own_conn = conn is None
    if own_conn:
        conn = get_fx_db(read_only=False)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            pair           VARCHAR NOT NULL,
            granularity    VARCHAR NOT NULL,
            ts             TIMESTAMP NOT NULL,
            open           DOUBLE,
            high           DOUBLE,
            low            DOUBLE,
            close          DOUBLE,
            bid_c          DOUBLE,
            ask_c          DOUBLE,
            volume         INTEGER,
            PRIMARY KEY (pair, granularity, ts)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS pnf_boxes (
            pair           VARCHAR NOT NULL,
            box_config     VARCHAR NOT NULL,
            box_id         INTEGER NOT NULL,
            column_id      INTEGER NOT NULL,
            direction      TINYINT NOT NULL,
            level          DOUBLE NOT NULL,
            mid_price      DOUBLE,
            ts             TIMESTAMP NOT NULL,
            PRIMARY KEY (pair, box_config, box_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS indicators (
            pair           VARCHAR NOT NULL,
            ts             TIMESTAMP NOT NULL,
            mc_5pip_rev2   DOUBLE,
            mc_5pip_rev3   DOUBLE,
            mc_15pip_rev2  DOUBLE,
            mc_15pip_rev3  DOUBLE,
            h1_support     DOUBLE,
            h1_resistance  DOUBLE,
            h1_zz_dir      TINYINT,
            atr14_m5       DOUBLE,
            atr14_h1       DOUBLE,
            mtf_mc_d       DOUBLE,
            mtf_mc_dd      DOUBLE,
            asian_high     DOUBLE,
            asian_low      DOUBLE,
            asian_mid      DOUBLE,
            PRIMARY KEY (pair, ts)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS kalman_strength (
            ts             TIMESTAMP NOT NULL,
            granularity    VARCHAR NOT NULL,
            currency       VARCHAR NOT NULL,
            strength       DOUBLE NOT NULL,
            rank           TINYINT,
            PRIMARY KEY (ts, granularity, currency)
        )
    """)

    logger.info("FX database schema initialized")
    if own_conn:
        conn.close()


def init_trades_schema(conn: Optional[duckdb.DuckDBPyConnection] = None):
    """Create trading state tables if they don't exist. Called by portfolio-mgr on startup."""
    own_conn = conn is None
    if own_conn:
        conn = get_trades_db(read_only=False)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS account_summary (
            account_id     VARCHAR NOT NULL,
            ts             TIMESTAMP NOT NULL,
            nav            DOUBLE,
            balance        DOUBLE,
            unrealized_pl  DOUBLE,
            margin_used    DOUBLE,
            margin_avail   DOUBLE,
            open_positions INTEGER,
            PRIMARY KEY (account_id, ts)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS allocation_weights (
            strategy       VARCHAR NOT NULL,
            pair           VARCHAR NOT NULL,
            ts             TIMESTAMP NOT NULL,
            weight         DOUBLE,
            margin_budget  DOUBLE,
            perf_gate      DOUBLE,
            dd_scale       DOUBLE,
            blocked        BOOLEAN DEFAULT FALSE,
            block_reason   VARCHAR,
            PRIMARY KEY (strategy, pair, ts)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            strategy       VARCHAR NOT NULL,
            pair           VARCHAR NOT NULL,
            account_id     VARCHAR NOT NULL,
            trade_id       VARCHAR NOT NULL,
            direction      TINYINT NOT NULL,
            entry_price    DOUBLE NOT NULL,
            entry_time     TIMESTAMP NOT NULL,
            units          INTEGER NOT NULL,
            sl_price       DOUBLE,
            tp_price       DOUBLE,
            running_mfe    DOUBLE DEFAULT 0,
            running_mae    DOUBLE DEFAULT 0,
            status         VARCHAR DEFAULT 'OPEN',
            PRIMARY KEY (strategy, pair, trade_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id             INTEGER,
            strategy       VARCHAR NOT NULL,
            pair           VARCHAR NOT NULL,
            account_id     VARCHAR NOT NULL,
            trade_id       VARCHAR NOT NULL,
            direction      TINYINT NOT NULL,
            entry_price    DOUBLE NOT NULL,
            exit_price     DOUBLE,
            entry_time     TIMESTAMP NOT NULL,
            exit_time      TIMESTAMP,
            pnl_pips       DOUBLE,
            exit_reason    VARCHAR,
            hours_held     DOUBLE,
            units          INTEGER,
            mfe_pips       DOUBLE,
            mae_pips       DOUBLE,
            capture_ratio  DOUBLE,
            amddp1         DOUBLE,
            pnl_over_mae   DOUBLE,
            is_paper       BOOLEAN DEFAULT FALSE,
            label          VARCHAR DEFAULT ''
        )
    """)

    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS trades_seq START 1
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS account_ledger (
            account_id     VARCHAR NOT NULL,
            ts             TIMESTAMP NOT NULL,
            nav            DOUBLE,
            balance        DOUBLE,
            margin_used    DOUBLE,
            margin_util_pct DOUBLE,
            dd_daily_pct   DOUBLE,
            dd_overall_pct DOUBLE,
            open_count     INTEGER,
            positions_json VARCHAR,
            PRIMARY KEY (account_id)
        )
    """)

    logger.info("Trades database schema initialized")
    if own_conn:
        conn.close()


def migrate_trades_schema():
    """Add is_paper / label columns to an existing trades table that lacks them."""
    import time
    db = get_trades_db(read_only=False)
    try:
        cols = {row[0] for row in db.execute("DESCRIBE trades").fetchall()}
        if "is_paper" not in cols:
            db.execute("ALTER TABLE trades ADD COLUMN is_paper BOOLEAN DEFAULT FALSE")
            logger.info("migrate_trades_schema: added is_paper column")
        if "label" not in cols:
            db.execute("ALTER TABLE trades ADD COLUMN label VARCHAR DEFAULT ''")
            logger.info("migrate_trades_schema: added label column")
    finally:
        db.close()


def write_trade_direct(strategy: str, pair: str, account_id: str,
                       trade_id: str, direction: int, entry_price: float,
                       exit_price: float, entry_time: str, exit_time: str,
                       pnl_pips: float, exit_reason: str, hours_held: float,
                       units: int, mfe_pips: float, mae_pips: float,
                       capture_ratio: float, is_paper: bool = False,
                       label: str = "") -> bool:
    """Write a trade record directly to DuckDB with dedup and retry on lock contention.

    Safe for infrequent writes (paper services, fallback path). Skips silently if
    trade_id already exists, so restart-replays are idempotent.
    """
    # DuckDB is single-writer (exclusive file lock). With ~15 concurrent writers, two writes
    # can collide; the loser raises "Conflicting lock is held". Retry with exponential backoff
    # + jitter over ~4.5s — far longer than any single INSERT — so collisions clear instead of
    # dropping the trade. (Architectural alternative: route all writes through portfolio_mgr's
    # ZMQ single-writer, below; this retry is the low-risk fix for the direct-write path.)
    import time as _time, random as _random
    ATTEMPTS = 8
    last_err = None
    for attempt in range(ATTEMPTS):
        db = None
        try:
            db = get_trades_db(read_only=False)
            db.execute(
                """INSERT INTO trades
                   (id, strategy, pair, account_id, trade_id, direction,
                    entry_price, exit_price, entry_time, exit_time, pnl_pips,
                    exit_reason, hours_held, units, mfe_pips, mae_pips,
                    capture_ratio, is_paper, label)
                   SELECT nextval('trades_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, ?
                   WHERE NOT EXISTS (SELECT 1 FROM trades WHERE trade_id = ?)""",
                [strategy, pair, account_id, trade_id, direction,
                 entry_price, exit_price, entry_time, exit_time,
                 round(pnl_pips, 1), exit_reason, round(hours_held, 2),
                 units, round(mfe_pips, 1), round(mae_pips, 1),
                 round(capture_ratio, 2), is_paper, label,
                 trade_id]
            )
            db.close(); db = None
            logger.info(f"write_trade_direct: {strategy} {pair} {pnl_pips:+.1f}p "
                        f"({exit_reason}) paper={is_paper} label={label}")
            return True
        except Exception as e:
            last_err = e
            if db is not None:
                try: db.close()
                except Exception: pass
            if attempt < ATTEMPTS - 1:
                _time.sleep(min(0.1 * (2 ** attempt), 1.0) + _random.uniform(0, 0.1))
    logger.error(f"write_trade_direct failed after {ATTEMPTS} attempts: {last_err}")
    return False


# ─── ZMQ Trade Write Proxy ────────────────────────────────────────────────
#
# Strategies send trade records via ZMQ PUSH to portfolio_mgr's PULL socket.
# This eliminates DuckDB lock contention — only portfolio_mgr writes.

from lib.zmq_protocol import TRADES_DB_PULL


class TradeDBSender:
    """ZMQ PUSH socket — strategies use this to send trade records to the DB writer."""

    def __init__(self, context: Optional[zmq.Context] = None):
        self.ctx = context or zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.SNDHWM, 1000)
        self.socket.setsockopt(zmq.LINGER, 5000)  # wait up to 5s on close
        ipc_path = TRADES_DB_PULL.replace("ipc://", "")
        os.makedirs(os.path.dirname(ipc_path), exist_ok=True)
        self.socket.connect(TRADES_DB_PULL)

    def send_trade(self, strategy: str, pair: str, account_id: str,
                   trade_id: str, direction: int, entry_price: float,
                   exit_price: float, entry_time: str, exit_time: str,
                   pnl_pips: float, exit_reason: str, hours_held: float,
                   units: int, mfe_pips: float, mae_pips: float,
                   capture_ratio: float, is_paper: bool = False,
                   label: str = ""):
        """Send a closed trade record to the DB writer."""
        msg = {
            "cmd": "insert_trade",
            "strategy": strategy, "pair": pair, "account_id": account_id,
            "trade_id": trade_id, "direction": direction,
            "entry_price": entry_price, "exit_price": exit_price,
            "entry_time": entry_time, "exit_time": exit_time,
            "pnl_pips": round(pnl_pips, 1), "exit_reason": exit_reason,
            "hours_held": round(hours_held, 2), "units": units,
            "mfe_pips": round(mfe_pips, 1), "mae_pips": round(mae_pips, 1),
            "capture_ratio": round(capture_ratio, 2),
            "is_paper": is_paper, "label": label,
        }
        self.socket.send(msgpack.packb(msg, use_bin_type=True))

    def close(self):
        self.socket.close()


class TradeDBWriter:
    """ZMQ PULL socket + DuckDB writer — runs in portfolio_mgr as a background thread.

    Receives trade records from all strategy containers and writes them
    to trades.duckdb as the single writer process.
    """

    def __init__(self, context: Optional[zmq.Context] = None):
        self.ctx = context or zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, 1000)
        ipc_path = TRADES_DB_PULL.replace("ipc://", "")
        os.makedirs(os.path.dirname(ipc_path), exist_ok=True)
        if os.path.exists(ipc_path):
            os.unlink(ipc_path)
        self.socket.bind(TRADES_DB_PULL)
        self._shutdown = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the writer thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="trade-db-writer")
        self._thread.start()
        logger.info("TradeDBWriter thread started")

    def stop(self):
        """Signal shutdown and wait for thread to finish."""
        self._shutdown = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self):
        """Poll for trade records and write to DuckDB."""
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        while not self._shutdown:
            events = dict(poller.poll(1000))  # 1s timeout
            if self.socket not in events:
                continue

            try:
                raw = self.socket.recv(zmq.NOBLOCK)
                msg = msgpack.unpackb(raw, raw=False)
            except zmq.Again:
                continue
            except Exception as e:
                logger.error(f"TradeDBWriter recv error: {e}")
                continue

            cmd = msg.get("cmd")
            if cmd == "insert_trade":
                self._write_trade(msg)
            else:
                logger.warning(f"TradeDBWriter unknown cmd: {cmd}")

    def _write_trade(self, msg: dict):
        """Write a single trade record to DuckDB (deduped by trade_id)."""
        try:
            db = get_trades_db(read_only=False)
            db.execute(
                """INSERT INTO trades
                   (id, strategy, pair, account_id, trade_id, direction,
                    entry_price, exit_price, entry_time, exit_time, pnl_pips,
                    exit_reason, hours_held, units, mfe_pips, mae_pips,
                    capture_ratio, is_paper, label)
                   SELECT nextval('trades_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, ?
                   WHERE NOT EXISTS (SELECT 1 FROM trades WHERE trade_id = ?)""",
                [msg["strategy"], msg["pair"], msg["account_id"], msg["trade_id"],
                 msg["direction"], msg["entry_price"], msg["exit_price"],
                 msg["entry_time"], msg["exit_time"], msg["pnl_pips"],
                 msg["exit_reason"], msg["hours_held"], msg["units"],
                 msg["mfe_pips"], msg["mae_pips"], msg["capture_ratio"],
                 msg.get("is_paper", False), msg.get("label", ""),
                 msg["trade_id"]]
            )
            db.close()
            logger.info(f"TradeDBWriter: wrote {msg['strategy']} {msg['pair']} "
                        f"{msg['pnl_pips']:+.1f}p ({msg['exit_reason']}) "
                        f"paper={msg.get('is_paper', False)}")
        except Exception as e:
            logger.error(f"TradeDBWriter DB write failed: {e}", exc_info=True)
