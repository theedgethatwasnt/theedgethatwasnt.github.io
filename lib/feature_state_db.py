"""DuckDB persistence layer for incremental indicator state.

WHY THIS EXISTS
---------------
Incremental indicator classes (IncrementalTopsBots, IncrementalZscore, FXFeatureBuilder)
accumulate path-dependent state by walking bars chronologically. On live service restart
they would normally need to replay N bars of history from the REST API just to warm up
indicators (e.g., 2016 bars for regime, 200 bars for channels, arbitrarily long for TopsBots).

This module persists the serialized state to DuckDB after every bar. On restart the service
loads the last saved state and resumes O(1) processing from the next new bar — no warmup
replay needed at all.

SCHEMA
------
Table: indicator_state
  pair        VARCHAR  — e.g. 'EUR_JPY'
  series_key  VARCHAR  — e.g. 'range_bars_10pip', 'm5', 'sba_h4'
                         Allows multiple independent state types per pair.
  last_bar_ts VARCHAR  — ISO-8601 timestamp of the last bar fed into this state
  state_json  TEXT     — JSON-serialized state dict from the indicator's to_dict()
  updated_at  VARCHAR  — wall-clock ISO timestamp of the last DB write

  PRIMARY KEY: (pair, series_key)

USAGE
-----
    import duckdb
    from lib.feature_state_db import ensure_table, save_state, load_state
    from lib.incremental_topsbots import IncrementalTopsBots

    conn = duckdb.connect('/data/feature_state.duckdb')
    ensure_table(conn)

    # Restore on startup:
    result = load_state(conn, 'EUR_JPY', 'range_bars_10pip')
    if result:
        last_ts, state_dict = result
        tb = IncrementalTopsBots.from_dict(state_dict)
    else:
        tb = IncrementalTopsBots()

    # After each bar:
    state, erp, act_h, act_l = tb.update(h, l, mid)
    save_state(conn, 'EUR_JPY', 'range_bars_10pip', bar_ts, tb.to_dict())

THREAD SAFETY
-------------
DuckDB connections are not thread-safe. Each thread/process should use its own
connection object. Passing `conn` between threads is not supported.

PERFORMANCE
-----------
save_state() does an UPSERT (INSERT OR REPLACE) — single row write, fast enough
to call on every bar without measurable overhead (~50 µs on SSD).
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Optional, Tuple


def ensure_table(conn) -> None:
    """Create the indicator_state table if it does not exist.

    Safe to call on every startup — idempotent.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS indicator_state (
            pair        VARCHAR NOT NULL,
            series_key  VARCHAR NOT NULL,
            last_bar_ts VARCHAR NOT NULL,
            state_json  TEXT    NOT NULL,
            updated_at  VARCHAR NOT NULL,
            PRIMARY KEY (pair, series_key)
        )
    """)


def save_state(
    conn,
    pair: str,
    series_key: str,
    last_bar_ts,
    state_dict: dict,
) -> None:
    """Persist indicator state after processing a bar.

    Args:
        conn        : active DuckDB connection
        pair        : instrument identifier, e.g. 'EUR_JPY'
        series_key  : logical name for this state, e.g. 'range_bars_10pip'
        last_bar_ts : timestamp of the bar just processed (str, datetime, or
                      pandas Timestamp — all converted to ISO-8601 string)
        state_dict  : JSON-serializable dict from indicator.to_dict()

    The call does an UPSERT — (pair, series_key) is the primary key, so multiple
    calls for the same pair/series_key simply overwrite the previous state.
    """
    ts_str = _to_iso(last_bar_ts)
    now_str = datetime.now(tz=timezone.utc).isoformat()
    json_str = json.dumps(state_dict)

    conn.execute("""
        INSERT INTO indicator_state (pair, series_key, last_bar_ts, state_json, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (pair, series_key) DO UPDATE SET
            last_bar_ts = excluded.last_bar_ts,
            state_json  = excluded.state_json,
            updated_at  = excluded.updated_at
    """, [pair, series_key, ts_str, json_str, now_str])


def load_state(
    conn,
    pair: str,
    series_key: str,
) -> Optional[Tuple[str, dict]]:
    """Load the last saved state for a (pair, series_key).

    Returns:
        (last_bar_ts_str, state_dict)  if a record exists
        None                            if no record has been saved yet

    Example::

        result = load_state(conn, 'EUR_JPY', 'range_bars_10pip')
        if result:
            last_ts, d = result
            tb = IncrementalTopsBots.from_dict(d)
        else:
            tb = IncrementalTopsBots()   # cold start
    """
    rows = conn.execute("""
        SELECT last_bar_ts, state_json
        FROM indicator_state
        WHERE pair = ? AND series_key = ?
    """, [pair, series_key]).fetchall()

    if not rows:
        return None
    last_bar_ts_str, json_str = rows[0]
    return last_bar_ts_str, json.loads(json_str)


def list_states(conn) -> list:
    """Return all stored (pair, series_key, last_bar_ts, updated_at) rows.

    Useful for diagnostics and monitoring startup state coverage.
    """
    rows = conn.execute("""
        SELECT pair, series_key, last_bar_ts, updated_at
        FROM indicator_state
        ORDER BY pair, series_key
    """).fetchall()
    return [{'pair': r[0], 'series_key': r[1], 'last_bar_ts': r[2], 'updated_at': r[3]}
            for r in rows]


def delete_state(conn, pair: str, series_key: str) -> bool:
    """Delete a stored state row. Returns True if a row was deleted."""
    result = conn.execute("""
        DELETE FROM indicator_state WHERE pair = ? AND series_key = ?
    """, [pair, series_key])
    return result.rowcount > 0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_iso(ts) -> str:
    """Convert various timestamp types to ISO-8601 string."""
    if ts is None:
        return datetime.now(tz=timezone.utc).isoformat()
    if isinstance(ts, str):
        return ts
    if isinstance(ts, datetime):
        return ts.isoformat()
    # pandas Timestamp, numpy datetime64, etc.
    try:
        return str(ts)
    except Exception:
        return datetime.now(tz=timezone.utc).isoformat()
