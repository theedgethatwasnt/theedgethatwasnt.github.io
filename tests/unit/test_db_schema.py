"""Tests for DuckDB schema — tables must exist with correct columns."""

import pytest
import tempfile
import os
import duckdb


@pytest.fixture
def fx_db(tmp_path):
    """Create a temporary FX database with schema."""
    os.environ["FX_DB_DIR"] = str(tmp_path)
    from lib.db import init_fx_schema, get_fx_db
    conn = get_fx_db(read_only=False)
    init_fx_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def trades_db(tmp_path):
    """Create a temporary trades database with schema."""
    os.environ["FX_DB_DIR"] = str(tmp_path)
    from lib.db import init_trades_schema, get_trades_db
    conn = get_trades_db(read_only=False)
    init_trades_schema(conn)
    yield conn
    conn.close()


class TestFXSchema:
    """Market data database schema tests."""

    def test_candles_table_exists(self, fx_db):
        result = fx_db.execute("SELECT * FROM candles LIMIT 0").fetchall()
        assert result == []

    def test_candles_insert_and_query(self, fx_db):
        fx_db.execute("""
            INSERT INTO candles VALUES
            ('EUR_JPY', 'S5', '2026-03-25 14:00:00', 184.321, 184.350, 184.300, 184.340, 184.338, 184.342, 127)
        """)
        rows = fx_db.execute("SELECT * FROM candles WHERE pair='EUR_JPY'").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "EUR_JPY"

    def test_pnf_boxes_table_exists(self, fx_db):
        fx_db.execute("SELECT * FROM pnf_boxes LIMIT 0")

    def test_indicators_table_exists(self, fx_db):
        fx_db.execute("SELECT * FROM indicators LIMIT 0")

    def test_kalman_strength_table_exists(self, fx_db):
        fx_db.execute("SELECT * FROM kalman_strength LIMIT 0")

    def test_candles_primary_key_enforced(self, fx_db):
        fx_db.execute("""
            INSERT INTO candles VALUES
            ('GBP_USD', 'M5', '2026-03-25 15:00:00', 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1)
        """)
        with pytest.raises(Exception):
            fx_db.execute("""
                INSERT INTO candles VALUES
                ('GBP_USD', 'M5', '2026-03-25 15:00:00', 1.27, 1.27, 1.27, 1.27, 1.27, 1.27, 2)
            """)


class TestTradesSchema:
    """Trading state database schema tests."""

    def test_positions_table_exists(self, trades_db):
        trades_db.execute("SELECT * FROM positions LIMIT 0")

    def test_trades_table_exists(self, trades_db):
        trades_db.execute("SELECT * FROM trades LIMIT 0")

    def test_account_summary_table_exists(self, trades_db):
        trades_db.execute("SELECT * FROM account_summary LIMIT 0")

    def test_allocation_weights_table_exists(self, trades_db):
        trades_db.execute("SELECT * FROM allocation_weights LIMIT 0")

    def test_position_insert_and_query(self, trades_db):
        trades_db.execute("""
            INSERT INTO positions VALUES
            ('label_long', 'EUR_JPY', '001-001-${OANDA_CUSTOMER_ID}-004', '123', 1,
             184.321, '2026-03-25 14:00:00', 15, 183.321, 185.071, 5.2, 1.3, 'OPEN')
        """)
        rows = trades_db.execute(
            "SELECT * FROM positions WHERE strategy='label_long' AND pair='EUR_JPY'"
        ).fetchall()
        assert len(rows) == 1

    def test_trade_with_reward_metrics(self, trades_db):
        trades_db.execute("""
            INSERT INTO trades VALUES
            (1, 'label_long', 'EUR_JPY', '001-001-${OANDA_CUSTOMER_ID}-004', '123', 1,
             184.321, 184.450, '2026-03-25 14:00:00', '2026-03-25 15:00:00',
             12.9, 'NETWORK_CLOSE', 1.0, 15, 15.2, 3.1, 0.849, 0.05, 4.16)
        """)
        row = trades_db.execute("SELECT amddp1, pnl_over_mae FROM trades WHERE id=1").fetchone()
        assert row[0] == pytest.approx(0.05)
        assert row[1] == pytest.approx(4.16)
