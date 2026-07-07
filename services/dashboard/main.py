#!/usr/bin/env python3
"""
FX-Core Dashboard — unified drill-down UI reading from DuckDB.

Top level: 10 account cards (NAV, uPL, margin%, open count)
Drill-down: click account → strategy details, per-pair P/L, open positions
Trade history: per-pair trade list with MFE/MAE

Runs on port 5558 (exposed via nginx reverse proxy).
"""

import os
import sys
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [dashboard] %(message)s")
logger = logging.getLogger("dashboard")

app = Flask(__name__)

# ── Active paper strategies shown on the dashboard ──
# Others are cut (Zone Recovery, Portfolio Paper, TR paper, SMA Scratch) — their
# trade history stays in trades.duckdb but is hidden from the live paper view.
# FIFO-Trends: %_ft, %_ft2, %_b2 · Retrace no-filter: retrace_nofilter · SMA-PSAR: sma_psar_%
# First-touch low-volume reversion: first_touch_lv (paper, 2026-06-18)
# FIFO paper (%_ft/%_ft2/%_b2) stopped 2026-06-19 — bled −627p, removed from active view.
# retrace_atr (2026-06-19): retrace + M5-ATR>=5 entry gate, A/B vs retrace_nofilter.
ACTIVE_PAPER_LABEL_PATTERNS = ["retrace_nofilter", "retrace_atr", "sma_psar_%", "first_touch_lv", "sma_scratch_%"]

def _active_paper_sql(col="label"):
    """SQL fragment (with no params) restricting `col` to active paper labels.
    Uses backslash ESCAPE so `\\_` matches a literal underscore (not any char)."""
    clauses = []
    for pat in ACTIVE_PAPER_LABEL_PATTERNS:
        if "%" in pat or "\\_" in pat:
            clauses.append(f"{col} LIKE '{pat}' ESCAPE '\\'")
        else:
            clauses.append(f"{col} = '{pat}'")
    return "(" + " OR ".join(clauses) + ")"

# ── ZMQ State Subscriber (real-time portfolio state from portfolio_mgr) ──
_portfolio_state = {}  # Latest snapshot from portfolio_mgr
_portfolio_state_lock = threading.Lock()

def _zmq_subscriber_thread():
    """Background thread: subscribe to portfolio_state from portfolio_mgr."""
    from lib.zmq_protocol import Subscriber, ALLOCATION_PUB, MSG_PORTFOLIO_STATE, make_topic
    sub = Subscriber(ALLOCATION_PUB, topics=[make_topic(MSG_PORTFOLIO_STATE)])
    logger.info("ZMQ subscriber started for portfolio_state")
    while True:
        try:
            result = sub.receive(timeout_ms=5000)
            if result:
                _, payload = result
                if payload.get("type") == MSG_PORTFOLIO_STATE:
                    with _portfolio_state_lock:
                        global _portfolio_state
                        _portfolio_state = payload
        except Exception as e:
            logger.warning(f"ZMQ subscriber error: {e}")
            import time; time.sleep(1)

_zmq_thread = threading.Thread(target=_zmq_subscriber_thread, daemon=True)
_zmq_thread.start()

# Account configuration — ALL accounts from env + registry
from lib.strategy_registry import STRATEGIES
import os

# Build from registry first
_registry_accounts = {s["account"]: s for s in STRATEGIES}

# All OANDA accounts from env (001-013)
ALL_ACCOUNT_IDS = {}
for key, val in os.environ.items():
    if key.startswith("OANDA_ACCOUNT_ID"):
        short = val.split("-")[-1]  # "001-001-${OANDA_CUSTOMER_ID}-008" -> "008"
        ALL_ACCOUNT_IDS[short] = val

ACCOUNTS = []
for short, full_id in sorted(ALL_ACCOUNT_IDS.items()):
    if short in _registry_accounts:
        s = _registry_accounts[short]
        ACCOUNTS.append({"label": s["account"], "name": s["label"],
                         "strategy": s["strategy_name"], "active": s.get("enabled", True),
                         "paused": bool(s.get("paused", False)),
                         "paused_reason": s.get("paused_reason", ""),
                         "live_since": s.get("live_since", ""),
                         "account_id": full_id})
    else:
        ACCOUNTS.append({"label": short, "name": f"Acct {short}",
                         "strategy": "none", "active": False,
                         "paused": False, "paused_reason": "", "live_since": "",
                         "account_id": full_id})


def _active_account_ids():
    """Full OANDA account_ids of currently-active (enabled) accounts — used to keep
    the live equity aggregate from resurrecting stopped/drained accounts' history."""
    return [a["account_id"] for a in ACCOUNTS if a.get("active") and a.get("account_id")]


def _get_db(db_type="trades"):
    """Get read-only DuckDB connection."""
    try:
        from lib.db import get_trades_db, get_fx_db
        return get_trades_db(read_only=True) if db_type == "trades" else get_fx_db(read_only=True)
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        return None


def _get_oanda_trade_stats(account_id):
    """Fallback: Get trade stats directly from OANDA API."""
    try:
        import v20
        api_key = os.environ.get('OANDA_API_KEY')
        if not api_key:
            return {}
            
        ctx = v20.Context(
            hostname='api-fxtrade.oanda.com',
            port='443',
            token=api_key
        )
        
        # Get recent closed trades
        resp = ctx.transaction.list(account_id, type='ORDER_FILL')
        if not resp or not resp.body:
            return {}
            
        fills = [t for t in resp.body.get('transactions', []) if t.type == 'ORDER_FILL']
        if not fills:
            return {}
            
        total_pnl_usd = sum(float(getattr(f, 'pl', 0)) for f in fills)
        wins = sum(1 for f in fills if float(getattr(f, 'pl', 0)) > 0)
        pnl_pips = total_pnl_usd * 10  # Rough USD to pips conversion
        
        return {
            'count': len(fills),
            'pnl_pips': round(pnl_pips, 1), 
            'wins': wins
        }
        
    except Exception as e:
        logger.error(f"OANDA API fallback error: {e}")
        return {}


# ── Today panel: per-account broker-realized USD + DB pips (cached) ──
_today_cache = {"ts": 0.0, "data": None, "day": None}
_today_lock = threading.Lock()
_TODAY_TTL = 60.0  # seconds — bounds OANDA transaction.list calls

def _today_window_utc():
    """UTC start/end ISO timestamps for the current calendar day."""
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start, end

def _today_oanda_pl(ctx, account_id: str, from_iso: str, to_iso: str) -> tuple:
    """Sum broker-realized USD pl and ORDER_FILL count for one account, today."""
    import re
    r = ctx.transaction.list(account_id, fromTime=from_iso, toTime=to_iso,
                             pageSize=500, type='ORDER_FILL')
    if r.status != 200:
        return 0.0, 0, f"list status={r.status}"
    pages = r.body.get('pages', []) or []
    pl_total, fills_with_pl = 0.0, 0
    for url in pages:
        m_from = re.search(r'from=(\d+)', url)
        m_to = re.search(r'to=(\d+)', url)
        if not m_from or not m_to:
            continue
        rr = ctx.transaction.range(account_id, fromID=m_from.group(1),
                                   toID=m_to.group(1), type='ORDER_FILL')
        if rr.status != 200:
            continue
        for t in rr.body.get('transactions', []):
            v = float(getattr(t, 'pl', '0') or 0)
            if v != 0:
                pl_total += v
                fills_with_pl += 1
    return pl_total, fills_with_pl, None

def _get_today_panel():
    """Compose today's per-account snapshot. Cached for _TODAY_TTL seconds."""
    import time
    now_ts = time.time()
    start, end = _today_window_utc()
    day = start.date().isoformat()

    with _today_lock:
        cached_ok = (_today_cache["data"] is not None
                     and _today_cache["day"] == day
                     and (now_ts - _today_cache["ts"]) < _TODAY_TTL)
        if cached_ok:
            return _today_cache["data"]

    # ── DB pips (cheap) ──
    db_pips = {}  # label -> {"trades":int, "pips":float}
    db = _get_db()
    if db:
        try:
            rows = db.execute(
                "SELECT account_id, COUNT(*), COALESCE(SUM(pnl_pips), 0) "
                "FROM trades "
                "WHERE is_paper = FALSE AND exit_time >= ? AND exit_time <= ? "
                "GROUP BY account_id",
                [start, end]
            ).fetchall()
            for acct_id, n, pips in rows:
                short = (acct_id or "").split("-")[-1]
                db_pips[short] = {"trades": int(n), "pips": float(pips)}
        except Exception as e:
            logger.error(f"today: db query failed: {e}")
        finally:
            db.close()

    # ── OANDA realized USD (expensive — only every _TODAY_TTL seconds) ──
    usd_by_label = {}
    try:
        import v20
        api_key = os.environ.get("OANDA_API_KEY")
        if api_key:
            ctx = v20.Context(hostname="api-fxtrade.oanda.com", port="443", token=api_key)
            from_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
            to_iso = end.strftime("%Y-%m-%dT%H:%M:%SZ")
            for acct in ACCOUNTS:
                if not acct.get("active", False):
                    continue   # skip stopped/drained accounts (also saves dead OANDA calls)
                acct_id = ALL_ACCOUNT_IDS.get(acct["label"])
                if not acct_id:
                    continue
                try:
                    pl, n, err = _today_oanda_pl(ctx, acct_id, from_iso, to_iso)
                    if err:
                        logger.warning(f"today {acct['label']}: {err}")
                        continue
                    usd_by_label[acct["label"]] = {"usd": pl, "fills": n}
                except Exception as e:
                    logger.warning(f"today {acct['label']}: {e}")
    except Exception as e:
        logger.error(f"today: oanda init failed: {e}")

    # ── Compose rows (NAV + open come from ZMQ state, kept warm) ──
    with _portfolio_state_lock:
        state = _portfolio_state.copy()
    zmq_accounts = state.get("accounts", {})

    rows_out = []
    tot_trades = tot_pips = tot_usd = 0
    for acct in ACCOUNTS:
        if not acct.get("active", False):
            continue   # only actively-trading accounts in today's panel
        lbl = acct["label"]
        z = zmq_accounts.get(lbl, {})
        d = db_pips.get(lbl, {"trades": 0, "pips": 0.0})
        u = usd_by_label.get(lbl, {"usd": 0.0, "fills": 0})
        rows_out.append({
            "label": lbl,
            "strategy": acct["name"],
            "active": acct.get("active", False),
            "paused": acct.get("paused", False),
            "paused_reason": acct.get("paused_reason", ""),
            "live_since": acct.get("live_since", ""),
            "trades": d["trades"],
            "pips": round(d["pips"], 1),
            "usd": round(u["usd"], 4),
            "fills": u["fills"],
            "open": int(z.get("open_count", 0) or 0),
            "nav": round(float(z.get("nav", 0) or 0), 2),
        })
        tot_trades += d["trades"]
        tot_pips += d["pips"]
        tot_usd += u["usd"]

    payload = {
        "day_utc": day,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows_out,
        "totals": {"trades": tot_trades, "pips": round(tot_pips, 1),
                   "usd": round(tot_usd, 4)},
        "cache_age_s": 0,
    }

    with _today_lock:
        _today_cache.update({"ts": now_ts, "data": payload, "day": day})
    return payload


# ─── API Endpoints ─────────────────────────────────────────────────────────

@app.route("/health")
def health():
    db = _get_db()
    if db:
        db.close()
        return "OK", 200
    return "DB unavailable", 503


@app.route("/api/portfolio_paper")
def api_portfolio_paper():
    """DISCONTINUED — Portfolio Paper strategy cut; tab removed from UI.
    History stays in trades.duckdb (label='portfolio_paper_%') but is hidden from view."""
    return jsonify({"discontinued": True})


def _api_portfolio_paper_legacy():
    """SMA-Stack portfolio: backtest reps (from portfolio.csv) + live paper P&L
    from trades.duckdb (label='portfolio_paper_<candidate_id>')."""
    PORT_DIR = "/app/portfolio"
    csv_path = os.path.join(PORT_DIR, "portfolio.csv")
    md_path  = os.path.join(PORT_DIR, "portfolio_report.md")
    if not os.path.exists(csv_path):
        return jsonify({"error": "portfolio.csv not found — run framework first"})

    import csv as _csv
    reps_csv = []
    try:
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                reps_csv.append(row)
    except Exception as e:
        return jsonify({"error": f"read portfolio.csv: {e}"})

    # Parse headline from report.md (cheap, no need to recompute)
    headline = {}
    if os.path.exists(md_path):
        try:
            with open(md_path) as f:
                txt = f.read()
            # Pull numbers via regex
            import re
            m = re.search(r"Total OOS pips:\s+\*\*([+-]?[\d\.]+)\*\*", txt)
            if m: headline["total_pips"] = float(m.group(1))
            m = re.search(r"Avg pips/trading-day:\s+\*\*([+-]?[\d\.]+)\*\*", txt)
            if m: headline["avg_pd"] = float(m.group(1))
            m = re.search(r"Combined trades/day:\s+\*\*([+-]?[\d\.]+)\*\*", txt)
            if m: headline["combined_freq"] = float(m.group(1))
            m = re.search(r"Sharpe.*?\*\*([+-]?[\d\.]+)\*\*", txt)
            if m: headline["sharpe"] = float(m.group(1))
            m = re.search(r"Max OOS drawdown:\s+\*\*([+-]?[\d\.]+).*?\*\*", txt)
            if m: headline["max_dd"] = float(m.group(1))
            m = re.search(r"p = \*\*([\d\.]+)\*\*", txt)
            if m: headline["mc_p"] = float(m.group(1))
        except Exception:
            pass

    # Pull live paper trade stats per candidate_id
    live_per = {}
    live_total = {"total_pips": 0.0, "n_trades": 0}
    db = _get_db()
    if db:
        try:
            rows = db.execute(
                "SELECT label, COUNT(*), COALESCE(SUM(pnl_pips), 0), "
                "COALESCE(SUM(CASE WHEN pnl_pips>0 THEN 1 ELSE 0 END), 0) "
                "FROM trades WHERE is_paper=TRUE AND label LIKE 'portfolio_paper_%' "
                "GROUP BY label"
            ).fetchall()
            for lbl, cnt, total_p, wins in rows:
                cid = lbl.replace("portfolio_paper_", "")
                live_per[cid] = {
                    "n": int(cnt),
                    "pips": float(total_p),
                    "wr":   round(wins / cnt * 100, 0) if cnt else 0,
                }
                live_total["n_trades"]  += int(cnt)
                live_total["total_pips"] += float(total_p)
        except Exception:
            pass
        finally:
            db.close()

    # Merge backtest + live
    reps = []
    for r in reps_csv:
        cid = r.get("candidate_id", "")
        live = live_per.get(cid, {"n": 0, "pips": 0.0, "wr": None})
        reps.append({
            "candidate_id": cid,
            "tf_label":     r.get("tf_label"),
            "pair":         r.get("pair"),
            "sma":          r.get("sma"),
            "M_exit":       int(float(r.get("M_exit", 0))),
            "tp_pips":      int(float(r.get("tp_pips", 0))),
            "oos_pd":       float(r.get("oos_pd", 0)),
            "oos_n":        int(float(r.get("oos_n", 0))),
            "live_n":       live["n"],
            "live_pips":    live["pips"],
            "live_wr":      live["wr"],
        })

    return jsonify({
        "headline": headline,
        "reps":     reps,
        "live":     live_total,
    })


@app.route("/api/paper")
def api_paper():
    """Paper trading status: active paper strategies only (FIFO-Trends; SMA-PSAR has no status file yet).
    Cut strategies (Zone Recovery, TR paper) are hidden — history stays in trades.duckdb."""
    DATA_DIR = "/data/logs"
    result = {}
    # FIFO paper + FIFO live both STOPPED — their stale *_status.json files are no longer served
    # (2026-06-19, Session 087; FIFO paper bled −627p, do not redeploy). first_touch is the only
    # active paper status file.
    for fname, key in [("first_touch_status.json", "first_touch")]:
        path = os.path.join(DATA_DIR, fname)
        try:
            with open(path) as f:
                result[key] = json.load(f)
        except FileNotFoundError:
            result[key] = {"error": "not running"}
        except Exception as e:
            result[key] = {"error": str(e)}
    return jsonify(result)


@app.route("/api/paper/stats")
def api_paper_stats():
    """Paper trade DB stats: per-label counts, pips, win-rate + current service status."""
    DATA_DIR = "/data/logs"

    # Live service status from JSON status files (FIFO paper+live STOPPED 2026-06-19, omitted)
    service_status = {}
    for fname, key in [("first_touch_status.json", "first_touch")]:
        path = os.path.join(DATA_DIR, fname)
        try:
            with open(path) as f:
                service_status[key] = json.load(f)
        except Exception:
            service_status[key] = {}

    # DB stats per paper label
    db = _get_db()
    label_stats = {}
    if db:
        try:
            rows = db.execute(
                "SELECT label, COUNT(*), COALESCE(SUM(pnl_pips),0), "
                "COALESCE(SUM(CASE WHEN pnl_pips>0 THEN 1 ELSE 0 END),0) "
                "FROM trades WHERE is_paper=TRUE AND label IS NOT NULL AND label != '' "
                f"AND {_active_paper_sql()} "
                "GROUP BY label ORDER BY label"
            ).fetchall()
            for lbl, cnt, total_pips, wins in rows:
                label_stats[lbl] = {
                    "trades": cnt,
                    "total_pips": round(total_pips, 1),
                    "win_rate": round(wins / cnt * 100, 0) if cnt > 0 else 0,
                }
        except Exception:
            pass
        finally:
            db.close()

    return jsonify({
        "label_stats": label_stats,
        "service_status": service_status,
    })


@app.route("/api/accounts")
def api_accounts():
    """All accounts: real-time from ZMQ portfolio_state, trade stats from DuckDB."""
    with _portfolio_state_lock:
        state = _portfolio_state.copy()

    # Get trade stats from DuckDB (still the best source for historical stats)
    trade_stats = {}
    db = _get_db()
    if db:
        try:
            for acct in ACCOUNTS:
                strategy = acct["strategy"]
                acct_like = f"%{acct['label']}"
                row = db.execute(
                    "SELECT COUNT(*), COALESCE(SUM(pnl_pips), 0), "
                    "COALESCE(SUM(CASE WHEN pnl_pips > 0 THEN 1 ELSE 0 END), 0) "
                    "FROM trades WHERE strategy = ? AND is_paper = FALSE "
                    "AND account_id LIKE ?",
                    [strategy, acct_like]
                ).fetchone()
                if row:
                    trade_stats[acct["label"]] = {
                        "n_trades": row[0], "total_pnl": row[1], "wins": row[2]}
        except Exception:
            pass
        finally:
            db.close()

    results = []
    zmq_accounts = state.get("accounts", {})
    use_zmq = len(zmq_accounts) > 0

    # DuckDB fallback for account data when ZMQ state unavailable
    db_accounts = {}
    if not use_zmq and db:
        try:
            db2 = _get_db()
            if db2:
                for acct in ACCOUNTS:
                    acct_id = f"001-001-${OANDA_CUSTOMER_ID}-{acct['label']}"
                    row = db2.execute(
                        "SELECT nav, balance, unrealized_pl, margin_used, margin_avail, open_positions "
                        "FROM account_summary WHERE account_id = ? ORDER BY ts DESC LIMIT 1",
                        [acct_id]
                    ).fetchone()
                    if row:
                        nav = row[0] or 0
                        db_accounts[acct["label"]] = {
                            "nav": round(nav, 2), "balance": round(row[1] or 0, 2),
                            "upl": round(row[2] or 0, 4),
                            "margin_pct": round((row[3] or 0) / nav * 100, 1) if nav > 0 else 0,
                            "open_count": row[5] or 0,
                        }
                db2.close()
        except Exception:
            pass

    for acct in ACCOUNTS:
        if not acct.get("active", False):
            continue   # only actively-trading accounts on the dashboard (no stopped/drained)
        label = acct["label"]
        zmq = zmq_accounts.get(label, db_accounts.get(label, {}))
        ts = trade_stats.get(label, {})
        n_trades = ts.get("n_trades", 0)
        total_pnl = ts.get("total_pnl", 0)
        wins = ts.get("wins", 0)
        wr = (wins / n_trades * 100) if n_trades > 0 else 0

        results.append({
            "label": label,
            "name": acct["name"],
            "strategy": acct["strategy"],
            "active": acct.get("active", False),
            "nav": zmq.get("nav", 0),
            "balance": zmq.get("balance", 0),
            "unrealized_pl": zmq.get("upl", 0),
            "margin_pct": zmq.get("margin_pct", 0),
            "open_positions": zmq.get("open_count", 0),
            "total_trades": n_trades,
            "total_pnl": round(total_pnl, 1),
            "win_rate": round(wr, 0),
        })

    total_nav = state.get("total_nav") or sum(r["nav"] for r in results)
    total_upl = state.get("total_upl") or sum(r["unrealized_pl"] for r in results)
    total_pos = state.get("total_positions") or sum(r["open_positions"] for r in results)

    return jsonify({
        "accounts": results,
        "ts": state.get("ts", str(datetime.now(timezone.utc))),
        "total_nav": round(total_nav, 2),
        "total_upl": round(total_upl, 4),
        "total_positions": total_pos,
        "source": "zmq" if use_zmq else "duckdb",
    })


@app.route("/api/accounts/today")
def api_accounts_today():
    """Per-account broker-realized snapshot for today UTC.
    DB pips/trade-count refresh every poll; OANDA USD cached for _TODAY_TTL s."""
    return jsonify(_get_today_panel())


@app.route("/api/account/<label>/trades")
def api_account_trades(label):
    """Trade history for one account's strategy."""
    acct = next((a for a in ACCOUNTS if a["label"] == label), None)
    if not acct:
        return jsonify({"error": f"Unknown account {label}"}), 404

    db = _get_db()
    if not db:
        return jsonify({"error": "DB unavailable"}), 503

    limit = request.args.get("limit", 50, type=int)
    # Live accounts: filter by account_id + is_paper=FALSE so per-pair stats
    # don't mash paper variant trades into the live drilldown.
    acct_id = acct.get("account_id", "")
    try:
        rows = db.execute(
            "SELECT pair, direction, entry_price, exit_price, entry_time, exit_time, "
            "pnl_pips, exit_reason, mfe_pips, mae_pips, capture_ratio, units "
            "FROM trades "
            "WHERE strategy = ? AND account_id = ? AND is_paper = FALSE "
            "ORDER BY exit_time DESC LIMIT ?",
            [acct["strategy"], acct_id, limit]
        ).fetchall()

        trades = [{
            "pair": r[0], "direction": r[1], "entry_price": r[2], "exit_price": r[3],
            "entry_time": str(r[4]), "exit_time": str(r[5]),
            "pnl_pips": round(r[6] or 0, 1), "exit_reason": r[7],
            "mfe_pips": round(r[8] or 0, 1), "mae_pips": round(r[9] or 0, 1),
            "capture_ratio": round(r[10] or 0, 2), "units": r[11],
        } for r in rows]

        # Per-pair stats — same live-only filter
        pair_rows = db.execute(
            "SELECT pair, COUNT(*), SUM(pnl_pips), "
            "SUM(CASE WHEN pnl_pips > 0 THEN 1 ELSE 0 END) "
            "FROM trades "
            "WHERE strategy = ? AND account_id = ? AND is_paper = FALSE "
            "GROUP BY pair ORDER BY SUM(pnl_pips) DESC",
            [acct["strategy"], acct_id]
        ).fetchall()
        pair_stats = [{
            "pair": r[0], "trades": r[1], "pnl": round(r[2] or 0, 1),
            "win_rate": round((r[3] / r[1] * 100) if r[1] > 0 else 0, 0),
        } for r in pair_rows]

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

    return jsonify({
        "account": label, "strategy": acct["strategy"], "name": acct["name"],
        "trades": trades, "pair_stats": pair_stats,
    })


def _time_clause(hours: int, since_str: str) -> tuple:
    """Return (sql_fragment, params) for the time filter in equity queries."""
    if since_str:
        return "ts >= CAST(? AS TIMESTAMP)", [since_str]
    return f"ts >= NOW() - INTERVAL '{int(hours)} hours'", []


def _time_clause_trades(hours: int, since_str: str) -> tuple:
    """Return (sql_fragment, params) for the time filter on trades table."""
    if since_str:
        return "exit_time >= CAST(? AS TIMESTAMP)", [since_str]
    return f"exit_time >= NOW() - INTERVAL '{int(hours)} hours'", []


@app.route("/api/equity")
def api_equity():
    """Equity curve: NAV snapshots over time. Optional account filter."""
    db = _get_db()
    if not db:
        return jsonify({"error": "DB unavailable"}), 503

    hours = request.args.get("hours", 168, type=int)
    since_str = request.args.get("since", "").strip()
    acct_filter = request.args.get("account", "")
    time_sql, time_params = _time_clause(hours, since_str)
    try:
        if acct_filter:
            rows = db.execute(
                f"SELECT ts, account_id, nav FROM account_summary "
                f"WHERE {time_sql} AND account_id LIKE ? ORDER BY ts",
                time_params + [f"%{acct_filter}"]
            ).fetchall()

            from collections import OrderedDict
            snapshots = OrderedDict()
            for ts, acct_id, nav in rows:
                key = str(ts)[:16]
                snapshots[key] = round(nav, 4)

            total_series = [{"t": t, "v": v} for t, v in snapshots.items()]
        else:
            rows = db.execute(
                f"SELECT ts, account_id, nav FROM account_summary "
                f"WHERE {time_sql} ORDER BY ts",
                time_params
            ).fetchall()

            from collections import OrderedDict
            last_nav = {}
            snapshots = OrderedDict()
            for ts, acct_id, nav in rows:
                key = str(ts)[:16]
                last_nav[acct_id] = nav
                if len(last_nav) >= 5:
                    snapshots[key] = round(sum(last_nav.values()), 2)

            total_series = [{"t": t, "v": v} for t, v in snapshots.items()]

        return jsonify({"equity": total_series, "hours": hours})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route("/api/equity/pips")
def api_equity_pips():
    """Cumulative pips curve from closed trades. Optional account filter."""
    db = _get_db()
    if not db:
        return jsonify({"error": "DB unavailable"}), 503

    hours = request.args.get("hours", 168, type=int)
    since_str = request.args.get("since", "").strip()
    acct_filter = request.args.get("account", "")
    time_sql, time_params = _time_clause_trades(hours, since_str)
    try:
        if acct_filter:
            acct = next((a for a in ACCOUNTS if a["label"] == acct_filter), None)
            if not acct:
                return jsonify({"equity": [], "hours": hours})
            rows = db.execute(
                f"SELECT exit_time, pnl_pips FROM trades "
                f"WHERE strategy = ? AND account_id LIKE ? "
                f"AND {time_sql} AND exit_time IS NOT NULL ORDER BY exit_time",
                [acct["strategy"], f"%{acct_filter}"] + time_params
            ).fetchall()
            # No DuckDB data — fall back to OANDA transaction history
            if not rows:
                db.close()
                acct_id = f"001-001-${OANDA_CUSTOMER_ID}-{acct_filter}"
                series = _oanda_pnl_series(acct_id, since_str, hours)
                return jsonify({"equity": series, "hours": hours,
                                "source": "oanda_usd", "account": acct_filter})
        else:
            rows = db.execute(
                f"SELECT exit_time, pnl_pips FROM trades "
                f"WHERE {time_sql} AND exit_time IS NOT NULL ORDER BY exit_time",
                time_params
            ).fetchall()

        cum = 0.0
        series = []
        for exit_time, pnl in rows:
            cum += (pnl or 0)
            series.append({"t": str(exit_time)[:16], "v": round(cum, 1)})

        return jsonify({"equity": series, "hours": hours, "source": "duckdb"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route("/api/equity/pips/split")
def api_equity_pips_split():
    """Cumulative pips split into live series + per-label paper series."""
    db = _get_db()
    if not db:
        return jsonify({"error": "DB unavailable"}), 503

    hours = request.args.get("hours", 168, type=int)
    since_str = request.args.get("since", "").strip()
    time_sql, time_params = _time_clause_trades(hours, since_str)

    def to_cum_series(rows):
        cum = 0.0
        series = []
        for exit_time, pnl in rows:
            cum += (pnl or 0)
            series.append({"t": str(exit_time)[:16], "v": round(cum, 1)})
        return series

    # Restrict the live aggregate to active accounts only (no stopped/drained history).
    active_ids = _active_account_ids()
    acct_sql = ""
    acct_params = []
    if active_ids:
        acct_sql = " AND account_id IN (" + ",".join(["?"] * len(active_ids)) + ")"
        acct_params = active_ids

    try:
        live_rows = db.execute(
            f"SELECT exit_time, pnl_pips FROM trades "
            f"WHERE is_paper=FALSE AND {time_sql}{acct_sql} AND exit_time IS NOT NULL ORDER BY exit_time",
            time_params + acct_params
        ).fetchall()

        label_rows = db.execute(
            f"SELECT DISTINCT label FROM trades "
            f"WHERE is_paper=TRUE AND label IS NOT NULL AND label != '' "
            f"AND {_active_paper_sql()} "
            f"AND {time_sql} AND exit_time IS NOT NULL ORDER BY label",
            time_params
        ).fetchall()

        paper = {}
        for (lbl,) in label_rows:
            rows = db.execute(
                f"SELECT exit_time, pnl_pips FROM trades "
                f"WHERE is_paper=TRUE AND label=? AND {time_sql} "
                f"AND exit_time IS NOT NULL ORDER BY exit_time",
                [lbl] + time_params
            ).fetchall()
            paper[lbl] = to_cum_series(rows)

        return jsonify({
            "live": to_cum_series(live_rows),
            "paper": paper,
            "hours": hours,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route("/api/equity/pips/accounts")
def api_equity_pips_accounts():
    """Cumulative pips per account, for selected accounts in pip mode."""
    raw = request.args.get("accounts", "")
    accts = [a.strip() for a in raw.split(",") if a.strip()]
    if not accts:
        return jsonify({"accounts": {}, "hours": 168})
    db = _get_db()
    if not db:
        return jsonify({"error": "DB unavailable"}), 503
    hours     = request.args.get("hours", 168, type=int)
    since_str = request.args.get("since", "").strip()

    # Window bounds (UTC, naive — matches DB exit_time). We compute the FULL running
    # cumulative since live_since, then anchor a point at the window start and extend a
    # point to "now", so a short/quiet window (e.g. 1h with no recent trades) still
    # renders a continuous line at the real running level instead of going blank.
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    now = _dt.now(_tz.utc).replace(tzinfo=None)
    win_start = None
    if since_str:
        s = since_str.replace("T", " ").replace("Z", "").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try: win_start = _dt.strptime(s, fmt); break
            except ValueError: pass
    if win_start is None:
        win_start = now - _td(hours=max(hours, 1))
    now_str = now.strftime("%Y-%m-%d %H:%M")
    ws_str  = win_start.strftime("%Y-%m-%d %H:%M")

    try:
        # Build label -> live_since lookup so we can clip series at deployment date.
        live_since_by_label = {a["label"]: a.get("live_since", "") for a in ACCOUNTS}
        result = {}
        for acct in accts:
            acct_id = f"001-001-${OANDA_CUSTOMER_ID}-{acct}"
            ls = live_since_by_label.get(acct, "") or ""
            extra_sql, extra_params = "", []
            if ls:
                extra_sql = " AND exit_time >= ?"
                extra_params = [ls]
            rows = db.execute(
                f"SELECT exit_time, pnl_pips FROM trades "
                f"WHERE is_paper=FALSE AND account_id=? {extra_sql} "
                f"AND exit_time IS NOT NULL ORDER BY exit_time",
                [acct_id] + extra_params
            ).fetchall()
            cum, anchor, inwin = 0.0, 0.0, []
            for exit_time, pnl in rows:
                cum += (pnl or 0)
                if exit_time < win_start:
                    anchor = cum
                else:
                    inwin.append({"t": str(exit_time)[:16], "v": round(cum, 1)})
            series = [{"t": ws_str, "v": round(anchor, 1)}] + inwin
            if not inwin or series[-1]["t"] < now_str:
                series.append({"t": now_str, "v": round(cum, 1)})
            result[acct] = series
        return jsonify({"accounts": result, "hours": hours})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route("/api/equity/flips")
def api_equity_flips():
    """Exit timestamps of 'flip' trades (position FIFO-netted by an opposing order —
    the 010 state-desync bug) per account, for red-X markers on the equity chart."""
    raw = request.args.get("accounts", "")
    accts = [a.strip() for a in raw.split(",") if a.strip()]
    if not accts:
        return jsonify({"flips": {}})
    db = _get_db()
    if not db:
        return jsonify({"error": "DB unavailable"}), 503
    try:
        out = {}
        for acct in accts:
            acct_id = f"001-001-${OANDA_CUSTOMER_ID}-{acct}"
            rows = db.execute(
                "SELECT exit_time, pnl_pips FROM trades WHERE is_paper=FALSE AND account_id=? "
                "AND exit_reason='flip' AND exit_time IS NOT NULL ORDER BY exit_time",
                [acct_id]).fetchall()
            if rows:
                out[acct] = [{"t": str(r[0])[:16], "pnl": round(r[1], 1)} for r in rows]
        return jsonify({"flips": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# Curated per-account change-log: strategy deploys, sizing changes, code bug fixes.
# Times are UTC (match trades.duckdb exit_time convention). kind drives marker
# colour/shape on the equity chart. Extend as new live changes ship.
EQUITY_EVENTS = {
    "010": [
        {"t": "2026-06-03 07:11", "kind": "deploy",   "label": "H17 stack-alignment v2 deployed on 010"},
        {"t": "2026-06-04 23:45", "kind": "bugfix",   "label": "fix phantom CLOSE loop + missing DB writes + orphan adoption"},
        {"t": "2026-06-09 16:17", "kind": "sizing",   "label": "dynamic sizing ×1.25, cap 50u"},
        {"t": "2026-06-11 11:30", "kind": "sizing",   "label": "remove 50u cap — sizing = balance ×1.25 straight"},
        {"t": "2026-06-11 11:48", "kind": "sizing",   "label": "bump UNITS_PER_DOLLAR 1.25 → 1.5"},
        {"t": "2026-06-14 04:41", "kind": "sizing",   "label": "re-add MAX_UNITS=30 cap (risk fix)"},
        {"t": "2026-06-17 21:44", "kind": "bugfix",   "label": "record real PSAR-exit fill price + true hours_held"},
        {"t": "2026-06-17 22:28", "kind": "sizing",   "label": "sizing → 0.5×balance + correct 110× pip-value error"},
        {"t": "2026-06-18 01:46", "kind": "strategy", "label": "DROP GBP_JPY — 4-pair stack"},
        {"t": "2026-06-18 11:55", "kind": "bugfix",   "label": "close handler: stop fabricating TP wins for losers (dashboard-lie fix)"},
        {"t": "2026-06-18 13:13", "kind": "bugfix",   "label": "flip bug: broker-truth guard (re-adopt instead of FIFO-net)"},
    ],
    # ── Paper strategies — keyed by paper LABEL (same kind→shape encoding). Markers
    #    snap onto the paper trace just like the live ones. Extend as paper changes ship.
    "retrace_nofilter": [
        {"t": "2026-05-26 14:50", "kind": "deploy", "label": "paper A/B launched — no-Markov retrace baseline (4 JPY pairs)"},
    ],
    "retrace_atr": [
        {"t": "2026-06-19 11:40", "kind": "deploy", "label": "deployed — retrace + M5-ATR(14)≥5 entry gate (A/B vs nofilter)"},
    ],
    "first_touch_lv": [
        {"t": "2026-06-18 00:00", "kind": "deploy", "label": "deployed — first-touch H4 low-volume reversion (12 pairs)"},
    ],
}


@app.route("/api/equity/events")
def api_equity_events():
    """Annotated change-log per account (deploys, sizing changes, bug fixes) for
    marker overlays on the equity chart. Returns {acct: [{t, kind, label}]}."""
    raw = request.args.get("accounts", "")
    accts = [a.strip() for a in raw.split(",") if a.strip()]
    out = {a: EQUITY_EVENTS[a] for a in accts if a in EQUITY_EVENTS}
    return jsonify({"events": out})


@app.route("/api/equity/mfe_mae")
def api_equity_mfe_mae():
    """Per-trade MFE/MAE for the visible window+accounts — feeds the side scatter next to the
    equity chart. Only live trades with recorded excursion (mfe/mae != 0)."""
    raw = request.args.get("accounts", "")
    accts = [a.strip() for a in raw.split(",") if a.strip() and a.strip() != "sum"]
    db = _get_db()
    if not db or not accts:
        return jsonify({"trades": []})
    hours = request.args.get("hours", 168, type=int)
    since_str = request.args.get("since", "").strip()
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    now = _dt.now(_tz.utc).replace(tzinfo=None)
    ws = None
    if since_str:
        s = since_str.replace("T", " ").replace("Z", "").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try: ws = _dt.strptime(s, fmt); break
            except ValueError: pass
    if ws is None:
        ws = now - _td(hours=max(hours, 1)) if hours else _dt(2000, 1, 1)
    ws_str = ws.strftime("%Y-%m-%d %H:%M:%S")
    try:
        live_since_by_label = {a["label"]: a.get("live_since", "") for a in ACCOUNTS}
        out = []
        for acct in accts:
            acct_id = f"001-001-${OANDA_CUSTOMER_ID}-{acct}"
            ls = live_since_by_label.get(acct, "") or ""
            extra = " AND exit_time >= ?" if ls else ""
            params = [acct_id, ws_str] + ([ls] if ls else [])
            rows = db.execute(
                f"SELECT mfe_pips, mae_pips, pnl_pips, direction, pair FROM trades "
                f"WHERE is_paper=FALSE AND account_id=? AND exit_time >= ? {extra} "
                f"AND exit_time IS NOT NULL AND (COALESCE(mfe_pips,0)<>0 OR COALESCE(mae_pips,0)<>0)",
                params).fetchall()
            for mfe, mae, pnl, d, pair in rows:
                out.append({"mfe": round(abs(mfe or 0), 1), "mae": round(abs(mae or 0), 1),
                            "pnl": round(pnl or 0, 1), "dir": int(d or 0),
                            "pair": pair, "acct": acct})
        return jsonify({"trades": out})
    except Exception as e:
        return jsonify({"error": str(e), "trades": []})
    finally:
        db.close()


# ── Floating-equity reconstruction (MT4-style continuous NAV) ────────────────
# A closed trade's intra-trade equity path is a deterministic function of the price
# bars over its life: floating(t) = dir*(mid(t)-entry)/pip - spread. We fetch the bars
# once per trade at a duration-adaptive granularity (S5 for short trades, coarser for
# long), cache the path (a closed trade's bars never change), and merge all trades into
# one portfolio NAV curve.
_PAIR_PIP_SP = {
    "USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
    "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
    "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0),
}
_GRAN = [("S5",5),("S10",10),("S30",30),("M1",60),("M2",120),("M4",240),("M5",300),
         ("M10",600),("M15",900),("M30",1800),("H1",3600)]
_PATHS_FILE = "/data/logs/trade_paths.json"
_path_cache = None


def _adaptive_gran(dur_sec):
    target = max(5.0, min(3600.0, dur_sec / 30.0))   # aim for ~30 samples
    name, secs = _GRAN[0]
    for n, s in _GRAN:
        if s <= target: name, secs = n, s
    return name, secs


def _load_path_cache():
    global _path_cache
    if _path_cache is None:
        try:
            with open(_PATHS_FILE) as f:
                _path_cache = json.load(f)
        except Exception:
            _path_cache = {}
    return _path_cache


def _save_path_cache():
    try:
        tmp = _PATHS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_path_cache, f)
        os.replace(tmp, _PATHS_FILE)
    except Exception as e:
        logger.error(f"path cache save failed: {e}")


def _parse_dt(v):
    """DB datetime or OANDA RFC3339 -> naive-UTC epoch seconds."""
    from datetime import datetime as _dt, timezone as _tz
    if hasattr(v, "timestamp"):
        return v.replace(tzinfo=None).timestamp() if v.tzinfo is None else v.astimezone(_tz.utc).replace(tzinfo=None).timestamp()
    s = str(v).replace("T", " ").replace("Z", "").split("+")[0].split(".")[0].strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try: return _dt.strptime(s, fmt).timestamp()
        except ValueError: pass
    return 0.0


def _trade_floating(ctx, tr):
    """Return [[epoch, floating_pips], ...] for one trade. Cached for closed trades."""
    import bisect
    tid = tr["trade_id"]; closed = bool(tr["exit_time"])
    cache = _load_path_cache()
    if closed and tid in cache:
        return cache[tid]
    pair = tr["pair"]; pip, sp = _PAIR_PIP_SP.get(pair, (0.0001, 2.0))
    d = tr["direction"]; entry = float(tr["entry_price"])
    e0 = _parse_dt(tr["entry_time"]); e1 = _parse_dt(tr["exit_time"]) if closed else None
    from datetime import datetime as _dt, timezone as _tz
    end_dt = tr["exit_time"] if closed else _dt.now(_tz.utc).replace(tzinfo=None)
    dur = (e1 if e1 else _dt.now(_tz.utc).timestamp()) - e0
    gran, _gs = _adaptive_gran(max(dur, 5))
    fiso = _dt.utcfromtimestamp(e0).strftime("%Y-%m-%dT%H:%M:%SZ")
    tiso = (_dt.utcfromtimestamp(e1) if e1 else _dt.utcnow()).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Bid/ask valuation (real, time-varying spread): a long is marked at the bid (what you'd
    # get selling), a short at the ask (what you'd pay covering). The close anchors to the
    # recorded, full-spread-net realized P&L so the curve lands exactly on each close.
    samples = []
    try:
        resp = ctx.instrument.candles(pair, granularity=gran, fromTime=fiso, toTime=tiso, price="BA")
        for c in resp.body.get("candles", []):
            tep = _parse_dt(c.time)
            fp = (float(c.bid.c) - entry) / pip if d > 0 else (entry - float(c.ask.c)) / pip
            samples.append([tep, round(fp, 1)])
    except Exception as e:
        logger.error(f"candles {pair} {tid}: {e}")
    if not samples:
        samples = [[e0, round(-sp, 1)]]
    elif samples[0][0] > e0 + 1:
        samples.insert(0, [e0, samples[0][1]])
    if closed:
        samples.append([e1, round(float(tr["pnl_pips"]), 1)])
        cache[tid] = samples; _save_path_cache()
    return samples


@app.route("/api/equity/floating")
def api_equity_floating():
    """Continuous portfolio NAV (realized + open floating), reconstructed from price bars."""
    import bisect
    raw = request.args.get("accounts", "")
    accts = [a.strip() for a in raw.split(",") if a.strip() and a.strip() != "sum"]
    api_key = os.environ.get("OANDA_API_KEY")
    db = _get_db()
    if not db or not accts or not api_key:
        return jsonify({"series": [], "n": 0})
    hours = request.args.get("hours", 168, type=int)
    since_str = request.args.get("since", "").strip()
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    now = _dt.now(_tz.utc).replace(tzinfo=None)
    ws = None
    if since_str:
        s = since_str.replace("T", " ").replace("Z", "").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try: ws = _dt.strptime(s, fmt); break
            except ValueError: pass
    if ws is None:
        ws = now - _td(hours=max(hours, 1)) if hours else _dt(2000, 1, 1)
    try:
        live_since = {a["label"]: a.get("live_since", "") for a in ACCOUNTS}
        trades = []
        for acct in accts:
            acct_id = f"001-001-${OANDA_CUSTOMER_ID}-{acct}"
            ls = live_since.get(acct, "") or ""
            extra = " AND (entry_time >= ? OR exit_time >= ?)"
            params = [acct_id, str(ws), str(ws)]
            if ls: extra += " AND exit_time >= ?"; params.append(ls)
            rows = db.execute(
                f"SELECT trade_id, pair, direction, entry_price, exit_price, entry_time, exit_time, pnl_pips "
                f"FROM trades WHERE is_paper=FALSE AND account_id=? {extra} ORDER BY entry_time DESC LIMIT 80",
                params).fetchall()
            for r in rows:
                trades.append(dict(trade_id=r[0], pair=r[1], direction=int(r[2] or 0),
                    entry_price=r[3], exit_price=r[4], entry_time=r[5], exit_time=r[6],
                    pnl_pips=float(r[7] or 0)))
        db.close(); db = None
        if not trades:
            return jsonify({"series": [], "n": 0})
        import v20
        ctx = v20.Context("api-fxtrade.oanda.com", 443, token=api_key)
        paths = []
        for tr in trades:
            s = _trade_floating(ctx, tr)
            if s: paths.append((_parse_dt(tr["entry_time"]),
                                _parse_dt(tr["exit_time"]) if tr["exit_time"] else 9e18,
                                tr["pnl_pips"], s))
        # master timeline = union of all sample epochs, downsampled to <=700
        allts = sorted({e for _, _, _, s in paths for e, _ in s})
        if len(allts) > 700:
            step = len(allts) / 700.0
            allts = [allts[int(i * step)] for i in range(700)]
        series = []
        for t in allts:
            nav = 0.0
            for e0, e1, pnl, s in paths:
                if t < e0: continue
                if t >= e1: nav += pnl; continue
                idx = bisect.bisect_right([x[0] for x in s], t) - 1
                nav += s[max(idx, 0)][1]
            series.append({"t": _dt.utcfromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"), "v": round(nav, 1)})
        return jsonify({"series": series, "n": len(trades)})
    except Exception as e:
        return jsonify({"error": str(e), "series": []})
    finally:
        if db: db.close()


@app.route("/api/strategy_starts")
def api_strategy_starts():
    """Return first closed trade timestamp per enabled strategy (for equity chart annotations)."""
    enabled = [s for s in STRATEGIES if s.get("enabled")]
    db = _get_db()
    starts = []
    for s in enabled:
        acct_id = f"001-001-${OANDA_CUSTOMER_ID}-{s['account']}"
        first_ts = None
        live_since = s.get("live_since", "")
        if db:
            try:
                if live_since:
                    row = db.execute(
                        "SELECT MIN(exit_time) FROM trades "
                        "WHERE account_id=? AND is_paper=FALSE AND exit_time IS NOT NULL "
                        "AND exit_time >= ?",
                        [acct_id, live_since]
                    ).fetchone()
                else:
                    row = db.execute(
                        "SELECT MIN(exit_time) FROM trades "
                        "WHERE account_id=? AND is_paper=FALSE AND exit_time IS NOT NULL",
                        [acct_id]
                    ).fetchone()
                if row and row[0]:
                    first_ts = str(row[0])[:16]
            except Exception:
                pass
        if not first_ts:
            first_ts = live_since
        if first_ts:
            starts.append({"account": s["account"], "name": s["label"], "first_trade_ts": first_ts})
    if db:
        db.close()
    return jsonify({"starts": starts})


def _oanda_pnl_series(account_id: str, since_str: str = "", hours: int = 168) -> list:
    """Fetch closed ORDER_FILL transactions from OANDA and return USD P&L time-series."""
    try:
        import v20
        from datetime import datetime, timezone, timedelta
        api_key = os.environ.get('OANDA_API_KEY')
        if not api_key:
            return []
        ctx = v20.Context(hostname='api-fxtrade.oanda.com', port='443', token=api_key)
        # Determine earliest timestamp we care about
        if since_str:
            cutoff = datetime.fromisoformat(since_str.replace('Z', '+00:00'))
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
        else:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        # Fetch last 900 transactions with fromTime if supported, else filter client-side
        try:
            resp = ctx.transaction.list(account_id, fromTime=cutoff_str, pageSize=900)
        except Exception:
            resp = ctx.transaction.list(account_id, pageSize=900)
        if not resp or not resp.body:
            return []
        txns = resp.body.get('transactions', [])
        fills = [t for t in txns
                 if getattr(t, 'type', '') == 'ORDER_FILL'
                 and str(getattr(t, 'time', '')) >= cutoff_str]
        fills.sort(key=lambda x: getattr(x, 'time', ''))
        cum = 0.0
        series = []
        for f in fills:
            pl = float(getattr(f, 'pl', 0) or 0)
            cum += pl
            t_raw = str(getattr(f, 'time', ''))
            t_str = t_raw[:16].replace('T', ' ')
            series.append({"t": t_str, "v": round(cum, 4)})
        return series
    except Exception as e:
        logger.error(f"OANDA reconcile error {account_id}: {e}")
        return []


@app.route("/api/reconcile/<label>")
def api_reconcile(label):
    """OANDA transaction-based P&L curve for an account. Falls back when DuckDB is empty."""
    acct = next((a for a in ACCOUNTS if a["label"] == label), None)
    if not acct:
        return jsonify({"error": f"Unknown account {label}"}), 404
    hours    = request.args.get("hours", 168, type=int)
    since_str = request.args.get("since", "").strip()
    acct_id  = f"001-001-${OANDA_CUSTOMER_ID}-{label}"
    series   = _oanda_pnl_series(acct_id, since_str, hours)
    return jsonify({"equity": series, "hours": hours, "source": "oanda_usd",
                    "account": label, "n": len(series)})


@app.route("/api/indicators/<pair>")
def api_indicators(pair):
    """Latest indicator snapshot for a pair."""
    db = _get_db("fx")
    if not db:
        return jsonify({"error": "DB unavailable"}), 503

    try:
        row = db.execute(
            "SELECT ts, mc_5pip_rev2, mc_5pip_rev3, mc_15pip_rev2, mc_15pip_rev3, "
            "h1_support, h1_resistance, h1_zz_dir, atr14_m5, atr14_h1, "
            "mtf_mc_d, mtf_mc_dd, asian_high, asian_low, asian_mid "
            "FROM indicators WHERE pair = ? ORDER BY ts DESC LIMIT 1",
            [pair]
        ).fetchone()

        if not row:
            return jsonify({"pair": pair, "data": None})

        return jsonify({"pair": pair, "data": {
            "ts": str(row[0]),
            "mc_5pip_rev2": row[1], "mc_5pip_rev3": row[2],
            "mc_15pip_rev2": row[3], "mc_15pip_rev3": row[4],
            "h1_support": row[5], "h1_resistance": row[6], "h1_zz_dir": row[7],
            "atr14_m5": row[8], "atr14_h1": row[9],
            "mtf_mc_d": row[10], "mtf_mc_dd": row[11],
            "asian_high": row[12], "asian_low": row[13], "asian_mid": row[14],
        }})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route("/api/signals")
def api_signals():
    """Rolling-window pips/min metrics + CSI, computed by fx-signals service."""
    state_file = "/data/logs/signals_state.json"
    try:
        with open(state_file) as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "signals_state.json not found — is fx-signals running?"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tick_momentum")
def api_tick_momentum():
    """Live tick-resolution momentum + spread, computed by fx-tick-mom service."""
    state_file = "/data/logs/tick_momentum_state.json"
    try:
        with open(state_file) as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "tick_momentum_state.json not found — is fx-tick-mom running?"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Streaming tab: live bid/ask buffers written by fx-tick-mom ─────────────
STREAMING_DIR = "/data/logs/streaming"
STREAMING_LATEST = "/data/logs/streaming_latest.json"
_STREAMING_FALLBACK = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY",
                       "AUD_JPY", "CAD_JPY", "NZD_JPY", "CHF_JPY", "NZD_USD", "EUR_GBP"]


@app.route("/api/streaming/pairs")
def api_streaming_pairs():
    """Pairs with a live buffer (reflects exactly what fx-tick-mom is streaming)."""
    try:
        pairs = sorted(f[:-5] for f in os.listdir(STREAMING_DIR) if f.endswith(".json"))
    except FileNotFoundError:
        pairs = []
    return jsonify(pairs or _STREAMING_FALLBACK)


@app.route("/api/streaming/latest")
def api_streaming_latest():
    """{pair: [ts_ms, bid, ask]} newest tick per pair — drives live chart append."""
    try:
        with open(STREAMING_LATEST) as f:
            return jsonify(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({})


@app.route("/api/streaming/context/<pair>")
def api_streaming_context(pair):
    """Last CLOSED bar of each higher timeframe (mid OHLC) — the chart's left context strip."""
    pair = os.path.basename(pair)
    from lib.oanda_adapter import OANDAAdapter
    tfs = [("1W", "W"), ("1D", "D"), ("8H", "H8"), ("4H", "H4"), ("1H", "H1"), ("5M", "M5")]
    try:
        adapter = OANDAAdapter()
    except Exception as e:
        return jsonify({"bars": [], "error": str(e)}), 503
    bars = []
    for label, gran in tfs:
        try:
            c = adapter.get_candles(pair, count=2, granularity=gran, price="MBA")
            if not c:
                continue
            b = c[-1]   # get_candles returns COMPLETE candles only → last = last closed bar
            bars.append({"tf": label, "o": b["open"], "h": b["high"], "l": b["low"],
                         "c": b["close"], "t": b["timestamp"]})   # t = bar START time (OANDA convention)
        except Exception:
            continue
    return jsonify({"bars": bars})


@app.route("/api/streaming/<pair>")
def api_streaming_pair(pair):
    """Full 1h buffer for one pair: {pair, stream_start_ms, points:[[ts,bid,ask]...]}."""
    path = os.path.join(STREAMING_DIR, f"{os.path.basename(pair)}.json")
    try:
        with open(path) as f:
            return jsonify(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({"pair": pair, "stream_start_ms": None, "points": []}), 404


@app.route("/api/signals/history")
def api_signals_history():
    """Return last N hours of signals history from DuckDB for charts."""
    import duckdb
    pair = request.args.get("pair", "GBP_JPY")
    hours = int(request.args.get("hours", 4))
    db_path = "/data/logs/signals_history.duckdb"
    if not os.path.exists(db_path):
        return jsonify({"error": "signals_history.duckdb not found"}), 503
    try:
        con = duckdb.connect(db_path, read_only=True)
        rows = con.execute("""
            SELECT ts, tf, momentum, accel, weighted_sum, momentum_norm
            FROM pair_signals
            WHERE pair = ? AND ts >= now() - INTERVAL (? || ' hours')
            ORDER BY ts, tf
        """, [pair, str(hours)]).fetchall()
        csi_rows = con.execute("""
            SELECT ts, currency, tf, value, accel, csi_weighted_sum
            FROM csi_signals
            WHERE ts >= now() - INTERVAL (? || ' hours')
            ORDER BY ts, currency, tf
        """, [str(hours)]).fetchall()
        # Price series straight out of xbreak_signals (same fx-signals 30s tick
        # cadence as the momentum rows — so all subgraphs share one timeline).
        price_rows = con.execute("""
            SELECT ts, c
            FROM xbreak_signals
            WHERE pair = ? AND ts >= now() - INTERVAL (? || ' hours')
            ORDER BY ts
        """, [pair, str(hours)]).fetchall()
        con.close()

        # Pivot pair rows: {tf: [{ts, momentum, accel, ws, mn}]}
        # mn = σ-scaled momentum (momentum_norm): dimensionless "sigma units"
        pair_data = {}
        for ts, tf, momentum, accel, ws, mn in rows:
            if tf not in pair_data:
                pair_data[tf] = []
            pair_data[tf].append({"ts": ts.isoformat(), "m": momentum, "a": accel,
                                  "ws": ws, "mn": mn})

        # Pivot CSI rows: {currency: {tf: [{ts, value, accel}]}}
        csi_data = {}
        for ts, currency, tf, value, accel, csi_ws in csi_rows:
            if currency not in csi_data:
                csi_data[currency] = {}
            if tf not in csi_data[currency]:
                csi_data[currency][tf] = []
            csi_data[currency][tf].append({"ts": ts.isoformat(), "v": value, "a": accel, "ws": csi_ws})

        price = [{"ts": ts.isoformat(), "c": c} for ts, c in price_rows]

        return jsonify({"pair": pair, "hours": hours,
                        "momentum": pair_data, "csi": csi_data, "price": price})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals/xbreak")
def api_signals_xbreak():
    """H1 cross-breakout indicators per pair (M5-emulated from S5 stream).
    Reads `xbreak` block out of signals_state.json (computed by fx-signals).
    """
    state_file = "/data/logs/signals_state.json"
    try:
        with open(state_file) as f:
            state = json.load(f)
        return jsonify({"ts": state.get("ts"), "xbreak": state.get("xbreak", {})})
    except FileNotFoundError:
        return jsonify({"error": "signals_state.json not found"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals/xbreak/history")
def api_signals_xbreak_history():
    """Last N hours of xbreak history for one pair."""
    import duckdb
    pair = request.args.get("pair", "GBP_JPY")
    hours = int(request.args.get("hours", 4))
    db_path = "/data/logs/signals_history.duckdb"
    if not os.path.exists(db_path):
        return jsonify({"error": "signals_history.duckdb not found"}), 503
    try:
        con = duckdb.connect(db_path, read_only=True)
        rows = con.execute("""
            SELECT ts, c, sma7, xover_4ago_p, xover_3ago_p,
                   small_move_p, large_move_p, current_mv_p,
                   accel, gap_shrink_1, gap_shrink_2,
                   long_armed, short_armed
            FROM xbreak_signals
            WHERE pair = ? AND ts >= now() - INTERVAL (? || ' hours')
            ORDER BY ts
        """, [pair, str(hours)]).fetchall()
        con.close()
        return jsonify({
            "pair": pair, "hours": hours,
            "rows": [{
                "ts": r[0].isoformat(),
                "c": r[1], "sma7": r[2],
                "xover_4ago_p": r[3], "xover_3ago_p": r[4],
                "small_move_p": r[5], "large_move_p": r[6],
                "current_mv_p": r[7],
                "accel": r[8], "gap_shrink_1": r[9], "gap_shrink_2": r[10],
                "long_armed": r[11], "short_armed": r[12],
            } for r in rows],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals/price")
def api_signals_price():
    """Return OHLC candles for the selected pair at S5 or M1 granularity."""
    import v20
    pair        = request.args.get("pair", "GBP_JPY")
    granularity = request.args.get("granularity", "M1")
    count       = min(int(request.args.get("count", 240)), 2880)
    api_key     = os.environ.get("OANDA_API_KEY")
    if not api_key:
        return jsonify({"error": "no OANDA_API_KEY"}), 503
    try:
        ctx  = v20.Context("api-fxtrade.oanda.com", 443, token=api_key)
        resp = ctx.instrument.candles(pair, granularity=granularity, count=count + 1, price="M")
        bars = []
        for c in resp.body["candles"]:
            if not c.complete:
                continue
            bars.append({
                "ts": str(c.time),
                "o":  float(c.mid.o),
                "h":  float(c.mid.h),
                "l":  float(c.mid.l),
                "c":  float(c.mid.c),
            })
        return jsonify({"pair": pair, "granularity": granularity, "bars": bars[-count:]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/bbchart/<pair>")
def api_bbchart(pair):
    """Price + SMA9 ±1σ Bollinger bands + BB-fade re-entry signal markers, for the live BB-fade book."""
    import v20, numpy as np
    tf = request.args.get("tf", "M15")
    gran = {"M5":"M5","M15":"M15","H1":"H1","H4":"H4"}.get(tf, "M15")
    count = 150
    api_key = os.environ.get("OANDA_API_KEY")
    if not api_key:
        return jsonify({"error": "no OANDA_API_KEY"}), 503
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    meat = {"M5":4.0,"M15":6.0,"H1":6.0,"H4":10.0}.get(tf, 6.0)
    spread = 2.5 if pair.endswith("JPY") else 1.5
    try:
        ctx = v20.Context("api-fxtrade.oanda.com", 443, token=api_key)
        resp = ctx.instrument.candles(pair, granularity=gran, count=count+12, price="M")
        h=[];l=[];c=[];ts=[]
        for cd in resp.body["candles"]:
            if not cd.complete: continue
            h.append(float(cd.mid.h)); l.append(float(cd.mid.l)); c.append(float(cd.mid.c)); ts.append(str(cd.time)[5:16])
        c=np.array(c); h=np.array(h); l=np.array(l); n=len(c)
        basis=np.full(n,np.nan); sd=np.full(n,np.nan)
        for i in range(8,n): w=c[i-8:i+1]; basis[i]=w.mean(); sd[i]=w.std()
        up=basis+sd; lo=basis-sd; sigs=[]
        for i in range(9,n):
            if l[i-1]>up[i-1] and l[i]<=up[i] and 0.5*(c[i]-basis[i])/pip-spread>=meat: sigs.append((i,-1,float(c[i])))
            elif h[i-1]<lo[i-1] and h[i]>=lo[i] and 0.5*(basis[i]-c[i])/pip-spread>=meat: sigs.append((i,1,float(c[i])))
        k=max(0,n-count); R=lambda a:[None if np.isnan(x) else round(float(x),5) for x in a[k:]]
        return jsonify({"pair":pair,"tf":tf,"ts":ts[k:],"c":[round(float(x),5) for x in c[k:]],
                        "upper":R(up),"lower":R(lo),"basis":R(basis),
                        "signals":[{"i":i-k,"dir":d,"p":round(p,5)} for (i,d,p) in sigs if i>=k]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


BB_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>BB-Fade</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>body{background:#0b0e14;color:#cdd6f4;font:13px system-ui;margin:0;padding:10px}
h2{margin:6px 0}select{background:#1a1f2b;color:#cdd6f4;border:1px solid #333;padding:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:10px}
.cell{background:#11151d;border:1px solid #222;border-radius:6px;padding:6px}
.pp{font-weight:600;margin-bottom:2px}.sig{color:#f9e2af}</style></head>
<body>
<h2>BB-Fade &#9889; &mdash; price + SMA9 &plusmn;1&sigma; + re-entry signals</h2>
<div style="margin:8px 0">Timeframe:
<select id="tf" onchange="reload()"><option>M5</option><option selected>M15</option><option>H1</option><option>H4</option></select>
&nbsp;<span id="status">loading...</span></div>
<div class="grid" id="grid"></div>
<script>
const PAIRS=["EUR_USD","GBP_USD","AUD_USD","NZD_USD","EUR_GBP","USD_JPY","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","CHF_JPY"];
const charts={};
function cell(p){const d=document.createElement('div');d.className='cell';
 d.innerHTML='<div class="pp">'+p+' <span class="sig" id="s-'+p+'"></span></div><canvas id="c-'+p+'" height="150"></canvas>';
 document.getElementById('grid').appendChild(d);}
async function load(p){const tf=document.getElementById('tf').value;
 let d; try{d=await(await fetch('/api/bbchart/'+p+'?tf='+tf)).json();}catch(e){return;}
 if(d.error){document.getElementById('s-'+p).textContent='('+d.error+')';return;}
 const x=d.ts, sd=d.signals||[];
 document.getElementById('s-'+p).textContent = sd.length? (sd.length+' signal(s) in view') : 'no signal in view';
 const sigPts=x.map((_,i)=>{const s=sd.find(z=>z.i===i);return s?s.p:null;});
 const cfg={type:'line',data:{labels:x,datasets:[
   {label:'close',data:d.c,borderColor:'#89b4fa',borderWidth:1.2,pointRadius:0,tension:0},
   {label:'upper',data:d.upper,borderColor:'#585b70',borderWidth:0.8,pointRadius:0},
   {label:'lower',data:d.lower,borderColor:'#585b70',borderWidth:0.8,pointRadius:0,fill:'-1',backgroundColor:'rgba(137,180,250,0.06)'},
   {label:'basis',data:d.basis,borderColor:'#f38ba8',borderWidth:0.7,pointRadius:0,borderDash:[3,3]},
   {label:'signal',data:sigPts,borderColor:'transparent',backgroundColor:'#f9e2af',pointRadius:5,pointStyle:'triangle',showLine:false}
 ]},options:{animation:false,plugins:{legend:{display:false}},scales:{x:{ticks:{maxTicksLimit:6,color:'#666'}},y:{ticks:{color:'#666'}}}}};
 if(charts[p]){const C=charts[p];C.data.labels=x;C.data.datasets[0].data=d.c;C.data.datasets[1].data=d.upper;
   C.data.datasets[2].data=d.lower;C.data.datasets[3].data=d.basis;C.data.datasets[4].data=sigPts;C.update('none');}
 else charts[p]=new Chart(document.getElementById('c-'+p),cfg);}
function reload(){PAIRS.forEach(load);document.getElementById('status').textContent='updated '+new Date().toLocaleTimeString();}
PAIRS.forEach(cell);reload();setInterval(reload,30000);
</script></body></html>"""

@app.route("/bb")
def bb_page():
    return BB_PAGE


TICK_MOM_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Tick Momentum ⚡</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body{background:#0a0a0a;color:#e0e0e0;font-family:'SF Mono','Menlo','Consolas',monospace;margin:0;padding:0;}
  .nav{background:#0f0f0f;border-bottom:1px solid #2a2a2a;padding:10px 20px;}
  .nav a{color:#888;text-decoration:none;margin-right:18px;font-size:13px;}
  .nav a:hover{color:#fff;}
  .hdr{padding:12px 20px 0;display:flex;justify-content:space-between;align-items:baseline;}
  .hdr h1{font-size:16px;margin:0;font-weight:600;}
  .ts{font-size:11px;color:#555;}
  .grid{display:grid;grid-template-columns:1fr;gap:14px;padding:16px 20px;}
  .card{background:#111;border:1px solid #222;border-radius:8px;padding:12px 14px;}
  .card h2{font-size:14px;margin:0 0 8px;display:flex;justify-content:space-between;align-items:baseline;}
  .px{font-size:13px;color:#bbb;font-weight:400;}
  .row{display:flex;gap:14px;font-size:12px;margin-bottom:8px;flex-wrap:wrap;}
  .metric{display:flex;flex-direction:column;}
  .metric .lbl{color:#666;font-size:10px;text-transform:uppercase;letter-spacing:.5px;}
  .metric .val{font-size:15px;font-weight:600;}
  .up{color:#4ade80;} .down{color:#f87171;} .flat{color:#888;}
  .warn{color:#fbbf24;}
  canvas{max-height:120px;}
</style></head>
<body>
<div class="nav"><a href="/">← Dashboard</a><a href="/tick_mom">Tick Mom ⚡</a></div>
<div class="hdr"><h1>Live Tick Momentum ⚡ <span style="color:#555;font-size:12px;">(5s window · OANDA stream ~250ms)</span></h1>
  <span class="ts" id="ts">connecting…</span></div>
<div class="grid" id="grid"></div>
<script>
const charts = {};
function dirBadge(d){ if(d>0) return '<span class="up">▲</span>'; if(d<0) return '<span class="down">▼</span>'; return '<span class="flat">─</span>'; }
function ensureCard(pair){
  let el = document.getElementById('card-'+pair);
  if(el) return el;
  el = document.createElement('div'); el.className='card'; el.id='card-'+pair;
  el.innerHTML =
    '<h2>'+pair+' <span class="px" id="px-'+pair+'">—</span></h2>'+
    '<div class="row">'+
      '<div class="metric"><span class="lbl">dir</span><span class="val" id="dir-'+pair+'">─</span></div>'+
      '<div class="metric"><span class="lbl">pips/min</span><span class="val" id="mpm-'+pair+'">—</span></div>'+
      '<div class="metric"><span class="lbl">ticks (5s)</span><span class="val" id="n-'+pair+'">—</span></div>'+
      '<div class="metric"><span class="lbl">rate /s</span><span class="val" id="r-'+pair+'">—</span></div>'+
      '<div class="metric"><span class="lbl">spread p</span><span class="val" id="sp-'+pair+'">—</span></div>'+
      '<div class="metric"><span class="lbl">sp x̄ / rel</span><span class="val" id="spr-'+pair+'">—</span></div>'+
    '</div>'+
    '<canvas id="cm-'+pair+'"></canvas>'+
    '<canvas id="cs-'+pair+'"></canvas>';
  document.getElementById('grid').appendChild(el);
  return el;
}
function mkChart(id, labels, cfgs){
  const ctx = document.getElementById(id);
  return new Chart(ctx, {type:'line', data:{labels:labels, datasets:cfgs},
    options:{animation:false, responsive:true, plugins:{legend:{labels:{color:'#888',boxWidth:10,font:{size:9}}}},
      scales:{x:{display:false}, y:{ticks:{color:'#666',font:{size:9}}, grid:{color:'#1a1a1a'}},
              y1:{position:'right',ticks:{color:'#666',font:{size:9}},grid:{display:false}}},
      elements:{point:{radius:0}, line:{borderWidth:1.4}}}});
}
async function tick(){
  let d;
  try { d = await (await fetch('/api/tick_momentum')).json(); }
  catch(e){ document.getElementById('ts').textContent='fetch error'; return; }
  if(d.error){ document.getElementById('ts').textContent=d.error; return; }
  document.getElementById('ts').textContent = 'updated '+(d.updated||'').replace('T',' ').slice(0,19)+' UTC';
  const _wk=Object.keys(d.windows||{}); const _w=_wk.length?d.windows[_wk[0]]:{pairs:{}};
  for(const pair of Object.keys(_w.pairs||{})){
    const p = _w.pairs[pair];
    ensureCard(pair);
    document.getElementById('px-'+pair).textContent = p.mid!=null ? p.mid : '—';
    const de = document.getElementById('dir-'+pair); de.innerHTML = dirBadge(p.dir);
    const me = document.getElementById('mpm-'+pair);
    me.textContent = (p.mpm>0?'+':'')+p.mpm; me.className = 'val '+(p.mpm>0?'up':(p.mpm<0?'down':'flat'));
    document.getElementById('n-'+pair).textContent = p.ticks;
    document.getElementById('r-'+pair).textContent = p.rate;
    const spe = document.getElementById('sp-'+pair);
    spe.textContent = p.spread; spe.className = 'val '+(p.spread_rel>1.5?'warn':'');
    document.getElementById('spr-'+pair).textContent = p.spread_avg+' / '+p.spread_rel+'×';
    const tr = p.trace||[];
    const labels = tr.map(x=>x.t.slice(11,19));
    const mpm = tr.map(x=>x.m), rate = tr.map(x=>x.n), sp = tr.map(x=>x.sp);
    if(!charts['m-'+pair]){
      charts['m-'+pair] = mkChart('cm-'+pair, labels, [
        {label:'pips/min', data:mpm, borderColor:'#60a5fa', yAxisID:'y'},
        {label:'ticks/win', data:rate, borderColor:'#a78bfa', yAxisID:'y1'}]);
      charts['s-'+pair] = mkChart('cs-'+pair, labels, [
        {label:'spread (p)', data:sp, borderColor:'#fbbf24', yAxisID:'y'}]);
    } else {
      const cm = charts['m-'+pair]; cm.data.labels=labels;
      cm.data.datasets[0].data=mpm; cm.data.datasets[1].data=rate; cm.update('none');
      const cs = charts['s-'+pair]; cs.data.labels=labels; cs.data.datasets[0].data=sp; cs.update('none');
    }
  }
}
tick(); setInterval(tick, 750);
</script>
</body></html>"""


@app.route("/tick_mom")
def tick_mom_page():
    return TICK_MOM_PAGE


# ─── Main HTML Page ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FX-Core Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<script>
// Inline plugin: draw vertical dashed lines at strategy start timestamps
Chart.register({
  id: 'strategyStartLines',
  afterDraw(chart) {
    const anns = chart.options._startAnnotations;
    if (anns === undefined) return;   // only the equity chart sets this
    const { ctx, chartArea: area, scales: { x } } = chart;
    if (!area) return;
    ctx.save();
    (anns || []).forEach(ann => {
      const xPos = x.getPixelForValue(ann.idx);
      if (xPos === undefined || xPos < area.left || xPos > area.right) return;
      ctx.globalAlpha = 0.55;
      ctx.strokeStyle = ann.color;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(xPos, area.top);
      ctx.lineTo(xPos, area.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = ann.color;
      ctx.font = 'bold 9px monospace';
      ctx.textAlign = 'center';
      const label = ann.name.length > 6 ? ann.name.slice(0, 6) : ann.name;
      ctx.fillText(label, xPos, area.top + 9);
    });
    // End-of-line labels: tie each live curve to its name + current value in
    // ITS OWN color, so attribution is unambiguous regardless of where lines cross.
    ctx.setLineDash([]);
    ctx.textAlign = 'left';
    ctx.font = 'bold 10px monospace';
    const placed = [];
    (chart.data.datasets || []).forEach((ds, di) => {
      if (ds._paper) return;
      const meta = chart.getDatasetMeta(di);
      if (!meta || meta.hidden || !meta.data || !meta.data.length) return;
      const last = meta.data[meta.data.length - 1];
      const val = ds.data[ds.data.length - 1];
      if (val === undefined || last.x === undefined) return;
      let yy = Math.max(area.top + 8, Math.min(area.bottom - 2, last.y));
      while (placed.some(p => Math.abs(p - yy) < 11)) yy += 11;   // avoid overlap
      placed.push(yy);
      ctx.globalAlpha = 0.95;
      ctx.fillStyle = ds.borderColor;
      const txt = ds.label + ' ' + (val >= 0 ? '+' : '') + Math.round(val) + 'p';
      ctx.fillText(txt, Math.min(last.x + 5, area.right - 78), yy);
    });
    ctx.restore();
  }
});
</script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; background: #0a0a0a; color: #e0e0e0; font-size: 13px; }
.header { background: #111; padding: 12px 20px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 16px; color: #fff; }
.header .ts { color: #888; font-size: 11px; }
.pulse { display: inline-block; width: 8px; height: 8px; background: #0f0; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

.summary-bar { display: flex; gap: 20px; padding: 12px 20px; background: #151515; border-bottom: 1px solid #222; flex-wrap: wrap; }
.summary-item { text-align: center; min-width: 80px; }
.summary-item .label { color: #888; font-size: 10px; text-transform: uppercase; }
.summary-item .value { font-size: 18px; font-weight: bold; }
.green { color: #4caf50; } .red { color: #f44336; } .white { color: #fff; }

/* Page-level tabs */
.page-tab-nav { display: flex; gap: 0; background: #0f0f0f; border-bottom: 1px solid #2a2a2a; padding: 0 20px; }
.page-tab-btn { padding: 8px 20px; background: transparent; border: none; border-bottom: 2px solid transparent; color: #666; font-family: inherit; font-size: 12px; cursor: pointer; transition: color 0.15s, border-color 0.15s; }
.page-tab-btn:hover { color: #aaa; }
.page-tab-btn.active { color: #fff; border-bottom-color: #4caf50; }
.page-tab-content { display: none; }
.page-tab-content.active { display: block; }

/* Equity chart section */
.equity-section { padding: 16px 20px; }
.equity-section canvas { width: 100% !important; height: 220px !important; }
.section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.section-header h2 { font-size: 13px; color: #aaa; }
.section-header select { background: #222; color: #ccc; border: 1px solid #444; padding: 3px 8px; border-radius: 3px; font-size: 11px; }
.equity-controls { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.pip-toggle-btn { background: #222; border: 1px solid #444; color: #888; padding: 3px 10px; border-radius: 3px; cursor: pointer; font-size: 11px; font-family: inherit; transition: all 0.2s; }
.pip-toggle-btn.active { background: #1a2e1a; border-color: #4caf50; color: #4caf50; }
.preset-btn { background: #1a1a1a; border: 1px solid #333; color: #666; padding: 3px 9px; border-radius: 3px; cursor: pointer; font-size: 11px; font-family: inherit; transition: all 0.2s; }
.preset-btn:hover { border-color: #555; color: #aaa; }
.preset-btn.active { background: #1a1e2e; border-color: #2196f3; color: #2196f3; }
.since-input { background: #111; border: 1px solid #444; color: #ccc; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-family: inherit; margin-left: 4px; }
.since-sep { color: #444; font-size: 11px; }
.acct-checkboxes { display: flex; flex-wrap: wrap; gap: 4px 10px; padding: 6px 20px 2px; align-items: center; border-bottom: 1px solid #1a1a1a; }
.chk-label { display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 11px; color: #888; white-space: nowrap; user-select: none; }
.chk-label input[type=checkbox] { accent-color: #4caf50; cursor: pointer; width: 11px; height: 11px; }
.chk-label:hover { color: #ccc; }
.chk-label.sum-label { color: #fff; font-weight: bold; padding-right: 10px; border-right: 1px solid #333; margin-right: 4px; }

.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; padding: 16px 20px; }
.card { background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 14px; cursor: pointer; transition: border-color 0.2s; position: relative; }
.card:hover { border-color: #555; }
.card.selected { border-color: #2196f3; box-shadow: 0 0 8px rgba(33,150,243,0.3); }
.card .accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 6px 6px 0 0; }
.card .acct-label { font-size: 11px; color: #888; }
.card .acct-name { font-size: 15px; font-weight: bold; color: #fff; margin: 4px 0; }
.card .row { display: flex; justify-content: space-between; margin: 3px 0; }
.card .k { color: #888; } .card .v { color: #ccc; }

.detail-panel { display: none; background: #111; border: 1px solid #333; margin: 0 20px 16px; border-radius: 6px; padding: 16px; }
.detail-panel.active { display: block; }
.detail-panel h2 { font-size: 14px; color: #fff; margin-bottom: 12px; }
.detail-close { float: right; cursor: pointer; color: #888; font-size: 18px; padding: 0 4px; }
.detail-close:hover { color: #fff; }

.chart-row { display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
.chart-box { flex: 1; min-width: 280px; background: #0d0d0d; border: 1px solid #222; border-radius: 4px; padding: 12px; }
.chart-box h3 { font-size: 11px; color: #888; text-transform: uppercase; margin-bottom: 8px; }
.chart-box canvas { width: 100% !important; height: 200px !important; }

table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { text-align: left; color: #888; padding: 6px 8px; border-bottom: 1px solid #333; font-weight: normal; position: sticky; top: 0; background: #111; }
td { padding: 6px 8px; border-bottom: 1px solid #1a1a1a; }
tr:hover td { background: #1a1a1a; }
.dir-long { color: #4caf50; } .dir-short { color: #f44336; }
.trade-table-wrap { max-height: 400px; overflow-y: auto; }

.tabs { display: flex; gap: 0; margin-bottom: 12px; }
.tab { padding: 6px 16px; cursor: pointer; background: #1a1a1a; border: 1px solid #333; color: #888; font-size: 12px; }
.tab:first-child { border-radius: 4px 0 0 4px; }
.tab:last-child { border-radius: 0 4px 4px 0; }
.tab.active { background: #333; color: #fff; }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* Mobile responsive */
@media (max-width: 768px) {
  .header h1 { font-size: 14px; }
  .summary-bar { gap: 12px; padding: 10px 12px; }
  .summary-item .value { font-size: 14px; }
  .grid { grid-template-columns: 1fr 1fr; gap: 8px; padding: 10px 12px; }
  .card { padding: 10px; }
  .card .acct-name { font-size: 13px; }
  .detail-panel { margin: 0 12px 12px; padding: 12px; }
  .chart-row { flex-direction: column; }
  .chart-box { min-width: auto; }
  table { font-size: 11px; }
  th, td { padding: 4px 6px; }
  .equity-section { padding: 12px; }
}
@media (max-width: 480px) {
  .grid { grid-template-columns: 1fr; gap: 8px; padding: 8px; }
  .summary-bar { justify-content: space-around; }
  .summary-item { min-width: 60px; }
  .summary-item .value { font-size: 13px; }
  .tabs { flex-wrap: wrap; }
  .tab { font-size: 11px; padding: 5px 10px; }
}

/* ── Markets ⚡ combined tab ───────────────────────────────────────── */
.mk-wrap { padding: 14px 20px; }
.mk-head { display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:6px; margin-bottom:10px; }
.mk-head h2 { font-size:13px; color:#aaa; }
.mk-sub { color:#555; font-weight:400; }
.mk-matrix-wrap { overflow-x:auto; -webkit-overflow-scrolling:touch; margin-bottom:14px; border:1px solid #222; border-radius:8px; background:#0d0d0d; }
table.mk-matrix { width:100%; border-collapse:collapse; font-size:12px; min-width:340px; }
table.mk-matrix th { font-weight:600; color:#999; padding:6px 4px; text-align:center; border-bottom:1px solid #222; position:static; background:transparent; }
table.mk-matrix th.mk-tfh { text-align:left; color:#777; padding-left:10px; }
table.mk-matrix td { padding:0; text-align:center; border:none; }
.mk-tfrow { cursor:pointer; }
.mk-tfrow:hover td .mk-cell { filter:brightness(1.18); }
.mk-tfrow.sel td.mk-tflabel { color:#fff; }
.mk-tfrow.sel { outline:1px solid #2a3a2a; }
td.mk-tflabel { text-align:left; padding:5px 10px !important; color:#888; font-weight:600; white-space:nowrap; }
.mk-cell { margin:2px; border-radius:4px; padding:5px 2px; font-size:11px; font-weight:600; line-height:1.1; color:#0a0a0a; }
.mk-cell small { display:block; font-size:8px; font-weight:400; opacity:.75; }
.mk-pills { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:14px; }
.mk-pill { background:#161616; border:1px solid #2a2a2a; color:#888; padding:4px 12px; border-radius:14px; font-size:12px; font-family:inherit; cursor:pointer; transition:all .15s; }
.mk-pill:hover { border-color:#444; color:#bbb; }
.mk-pill.sel { background:#16241a; border-color:#4caf50; color:#7fdca0; }
.mk-pill .mk-pdot { font-size:8px; opacity:.7; }
.mk-ladder { display:flex; flex-wrap:wrap; align-items:center; gap:4px; font-size:16px; font-weight:600; margin-bottom:12px; }
.mk-panel { background:#0d0d0d; border:1px solid #222; border-radius:8px; padding:12px 14px; margin-bottom:14px; }
.mk-panel-lbl { font-size:10px; color:#666; text-transform:uppercase; letter-spacing:.5px; margin-bottom:6px; }
.mk-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.mk-card { background:#111; border:1px solid #222; border-radius:8px; padding:10px 12px; }
.mk-bar-row { display:flex; align-items:center; gap:8px; font-size:12px; margin:3px 0; }
.mk-bar-row .mk-bar-name { width:64px; color:#bbb; flex:none; }
.mk-bar-track { flex:1; height:12px; background:#161616; border-radius:3px; position:relative; overflow:hidden; }
.mk-bar-fill { position:absolute; top:0; bottom:0; }
.mk-bar-val { width:54px; text-align:right; flex:none; font-size:11px; }
@media (max-width: 768px) {
  .mk-wrap { padding:10px 12px; }
  .mk-grid { grid-template-columns:1fr; }
  .mk-ladder { font-size:14px; }
  .mk-cell { font-size:10px; }
}
</style>
</head>
<body>

<div class="header">
  <h1><span class="pulse"></span>FX-Core Dashboard</h1>
  <div class="ts" id="last-update">Loading...</div>
</div>

<div class="summary-bar" id="summary-bar"></div>

<!-- Page-level tab nav -->
<div class="page-tab-nav">
  <button class="page-tab-btn active" id="ptab-dashboard" onclick="switchPageTab('dashboard')">Dashboard</button>
  <button class="page-tab-btn" id="ptab-markets" onclick="switchPageTab('markets')">Markets ⚡</button>
  <button class="page-tab-btn" id="ptab-bb" onclick="switchPageTab('bb')">BB ⚡</button>
  <button class="page-tab-btn" id="ptab-streaming" onclick="switchPageTab('streaming')">Streaming</button>
</div>

<!-- BB-Fade ⚡ tab (price + SMA9 ±1σ + re-entry signal markers) -->
<div id="ptab-content-bb" class="page-tab-content" style="padding:12px 20px;">
  <div style="margin-bottom:8px;font-size:12px;">BB-Fade ⚡ — price + SMA9 ±1σ envelope + re-entry signals (▲). Timeframe:
    <select id="bb-tf" onchange="loadBBall()" style="background:#1a1a1a;color:#ddd;border:1px solid #333;padding:3px;">
      <option>M5</option><option selected>M15</option><option>H1</option><option>H4</option></select>
    <span id="bb-status" style="color:#666;font-size:11px;margin-left:8px;"></span></div>
  <div id="bb-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:10px;"></div>
</div>

<!-- Streaming tab (live bid/ask, 1h buffer @1/s history, ~250ms live edge, S5 backfill + divider) -->
<div id="ptab-content-streaming" class="page-tab-content" style="padding:12px 20px;">
  <div style="margin-bottom:8px;font-size:12px;">Streaming — live bid/ask, last hour. S5 backfill left of the dashed line; live stream right of it. Under each context bar's TF label: pips/minute ((close−open)÷pip÷minutes, green up / red down) η = efficiency (|close−open|÷(high−low), 0=chop … 1=clean one-way bar); and ↕ = current price location in the bar range (0=low, 1=high, 0.5=mid; >1 = above the bar, <0 = below — fraction is % of bar range).
    <select id="strm-pair" onchange="strmLoad(this.value)" style="background:#1a1a1a;color:#ddd;border:1px solid #333;padding:3px;margin-left:6px;"></select>
    <span id="strm-stale" style="font-size:11px;margin-left:8px;">—</span></div>
  <div style="background:#11151d;border:1px solid #222;border-radius:6px;padding:8px;height:440px;">
    <canvas id="strmChart"></canvas></div>
</div>

<!-- Markets ⚡ tab (combines momentum signals + livestream across timeframes) -->
<div id="ptab-content-markets" class="page-tab-content">
  <div class="mk-wrap">
    <div class="mk-head">
      <h2>Markets ⚡ <span class="mk-sub">currency strength across timeframes · 5s · 30s (live stream) · 5m–24h (candles)</span></h2>
      <span class="ts" id="mk-ts" style="font-size:11px;color:#555;">connecting…</span>
    </div>
    <div class="mk-panel-lbl">Currency strength matrix — rows = timeframe, columns = currency · green strong / red weak (per-row) · tap a row to focus</div>
    <div class="mk-matrix-wrap"><table class="mk-matrix" id="mk-matrix"></table></div>
    <div id="mk-pills" class="mk-pills"></div>
    <div id="mk-focus"></div>
  </div>
</div>

<!-- Dashboard tab -->
<div id="ptab-content-dashboard" class="page-tab-content active">
  <div class="equity-section">
    <div class="section-header">
      <h2 id="equity-title">Realized Pips</h2>
      <div class="equity-controls">
        <button class="preset-btn" data-hours="1" onclick="setPreset(this)">1h</button>
        <button class="preset-btn active" data-hours="24" onclick="setPreset(this)">24h</button>
        <button class="preset-btn" data-hours="72" onclick="setPreset(this)">3d</button>
        <button class="preset-btn" data-hours="168" onclick="setPreset(this)">7d</button>
        <button class="preset-btn" data-hours="720" onclick="setPreset(this)">30d</button>
        <button class="preset-btn" data-hours="0" onclick="setPreset(this)">All</button>
        <span class="since-sep">│</span>
        <input type="datetime-local" id="since-input" class="since-input"
               title="Custom start date/time" onchange="onSinceChange()">
      </div>
    </div>
    <div class="acct-checkboxes" id="acct-checkboxes">
      <label class="chk-label sum-label"><input type="checkbox" id="show-live" checked onchange="fetchEquity()"> 🟢 Live</label>
      <label class="chk-label sum-label" style="margin-left:8px;"><input type="checkbox" id="show-paper" onchange="fetchEquity()"> 🟡 Paper</label>
      <span style="color:#444;margin:0 6px;">│</span>
      <label class="chk-label sum-label"><input type="checkbox" class="acct-chk" value="sum" onchange="fetchEquity()"> ∑ Acct Sum</label>
      <span style="color:#444;margin:0 6px;">│</span>
      <label class="chk-label sum-label" title="Continuous floating NAV reconstructed from price bars (shows intra-trade drawdowns)"><input type="checkbox" id="show-floating" onchange="fetchEquity()"> 〰 Floating NAV</label>
    </div>
    <div style="display:flex; gap:14px; align-items:flex-start;">
      <div style="flex:1 1 72%; min-width:0;"><canvas id="equity-chart"></canvas></div>
      <div style="flex:1 1 28%; min-width:0;">
        <div style="font-size:11px;color:#888;text-align:center;margin-bottom:1px;">MFE ↑ vs MAE → · visible trades (▲long ◆short, green=win)</div>
        <canvas id="mfemae-chart"></canvas>
      </div>
    </div>
  </div>
  <div style="padding:8px 20px 0;">
    <div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid #2a2a2a;padding-top:10px;margin-bottom:6px;">
      <h2 style="font-size:13px;color:#aaa;">Today — Broker-Realized <span id="today-day" style="color:#555;font-size:11px;font-weight:normal;"></span></h2>
      <span id="today-meta" style="font-size:11px;color:#555;"></span>
    </div>
    <div id="today-section"></div>
  </div>
  <div class="grid" id="account-grid"></div>
  <div id="detail-container"></div>
  <div style="padding:0 20px 8px;">
    <div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid #2a2a2a;padding-top:12px;margin-bottom:8px;">
      <h2 style="font-size:13px;color:#aaa;">Paper Strategies</h2>
      <span style="font-size:11px;color:#555;">DB trades + live positions (no real orders)</span>
    </div>
    <div id="paper-section"></div>
  </div>
</div>

<!-- Signals tab -->
<div id="ptab-content-signals" class="page-tab-content" style="padding:16px 20px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <h2 style="font-size:13px;color:#aaa;">Rolling Momentum Signals</h2>
    <span style="font-size:11px;color:#555;" id="signals-ts">pips/min · updated by fx-signals</span>
  </div>
  <div id="signals-section" style="font-family:'SF Mono','Menlo','Consolas',monospace;font-size:12px;overflow-x:auto;"></div>

  <div id="signals-chart-section" style="margin-top:24px;">
    <div style="margin-bottom:8px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
      <span style="color:#888;font-size:11px;">History chart:</span>
      <select id="chart-pair-select" style="background:#1a1a1a;color:#ccc;border:1px solid #333;padding:3px 6px;font-size:11px;border-radius:3px;" onchange="loadSignalChart()">
      </select>
      <select id="chart-hours-select" style="background:#1a1a1a;color:#ccc;border:1px solid #333;padding:3px 6px;font-size:11px;border-radius:3px;" onchange="loadSignalChart()">
        <option value="1">1h</option>
        <option value="4" selected>4h</option>
        <option value="12">12h</option>
        <option value="24">24h</option>
      </select>
      <span id="chart-status" style="font-size:10px;color:#555;"></span>
    </div>
    <canvas id="signals-price-chart" style="max-height:180px;margin-bottom:12px;"></canvas>
    <canvas id="signals-momentum-chart" style="max-height:180px;margin-bottom:12px;"></canvas>
    <canvas id="signals-accel-chart" style="max-height:130px;margin-bottom:12px;"></canvas>
    <canvas id="signals-ws-chart" style="max-height:130px;"></canvas>
  </div>
</div>

<!-- Tick Momentum tab -->
<div id="ptab-content-tickmom" class="page-tab-content" style="padding:14px 20px;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;">
    <h2 style="font-size:13px;color:#aaa;">Live Tick Momentum ⚡ <span style="color:#555;font-weight:400;">OANDA stream ~250ms · per-window momentum (pips/min) + spread + currency strength</span></h2>
    <span style="font-size:11px;color:#555;" id="tm-ts">connecting…</span>
  </div>
  <div id="tm-cols" style="display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start;"></div>
</div>

<!-- Portfolio Paper tab removed (strategy cut — history retained in trades.duckdb) -->

<script>
const REFRESH_MS = 5000;
let selectedAcct = null;
let activeTab = 'pairs';  // Track active tab across refreshes
let equityChart = null;
let mfemaeChart = null;
let mfeChart = null;
let pnlChart = null;

let equityCheckboxesPopulated = false;
let activePresetHours = 24;   // tracks current preset; 0 = All
let acctNames = {};

// Signals history charts
let priceChart = null, momentumChart = null, accelChart = null, wsChart = null;
const CHART_WINDOWS = ["5m", "1h", "4h", "24h"];
const CHART_COLORS = {"5m":"#4fc3f7","1h":"#81c784","4h":"#ffb74d","24h":"#ce93d8"};
const PAIRS_LIST = [
  "GBP_JPY","USD_JPY","EUR_JPY","EUR_USD","GBP_USD",
  "AUD_JPY","CAD_JPY","CHF_JPY","AUD_USD","NZD_JPY","NZD_USD","EUR_GBP"
];

function accelArrow(a) {
  if (a === null || a === undefined) return '';
  if (a > 0.001) return '<span style="color:#4caf50;font-size:9px;">&#9650;</span>';
  if (a < -0.001) return '<span style="color:#e57373;font-size:9px;">&#9660;</span>';
  return '<span style="color:#555;font-size:9px;">&#9644;</span>';
}

function initChartPairSelect() {
  const sel = document.getElementById('chart-pair-select');
  if (!sel || sel.options.length > 0) return;
  PAIRS_LIST.forEach(p => {
    const o = document.createElement('option');
    o.value = p; o.text = p;
    sel.appendChild(o);
  });
}

async function loadSignalChart() {
  const pair  = document.getElementById('chart-pair-select')?.value || 'GBP_JPY';
  const hours = document.getElementById('chart-hours-select')?.value || '4';
  document.getElementById('chart-status').textContent = 'loading…';

  // ONE fetch — both price and momentum come out of signals_history.duckdb at
  // the same 30s fx-signals poll cadence, so every subgraph shares the same
  // timeline by construction.
  let histResp;
  try {
    histResp = await fetch('/api/signals/history?pair=' + pair + '&hours=' + hours);
  } catch(e) {
    document.getElementById('chart-status').textContent = 'network error';
    return;
  }
  document.getElementById('chart-status').textContent = '';

  if (!histResp.ok) return;
  let data = null;
  try { data = await histResp.json(); } catch(e) { return; }
  if (!data || data.error) return;

  // Shared x-range pinned to the signals history span.
  let xMin = null, xMax = null;
  CHART_WINDOWS.forEach(w => {
    const pts = data.momentum[w] || [];
    pts.forEach(p => {
      const t = new Date(p.ts).getTime();
      if (xMin === null || t < xMin) xMin = t;
      if (xMax === null || t > xMax) xMax = t;
    });
  });

  // Single y-axis. Click any legend entry to toggle that trace on/off — the
  // chart auto-rescales to fit the remaining visible traces.
  const chartOpts = (title, yLabel) => ({
    responsive: true,
    animation: false,
    plugins: {
      legend: {
        labels: { color: '#888', font: { size: 10 } },
        onClick: (e, legendItem, legend) => {
          const ci = legend.chart;
          const idx = legendItem.datasetIndex;
          const m = ci.getDatasetMeta(idx);
          m.hidden = m.hidden === null ? !ci.data.datasets[idx].hidden : null;
          ci.update();
        },
      },
      title: { display: true, text: title, color: '#888', font: { size: 11 } }
    },
    scales: {
      x: {
        type: 'time',
        min: xMin, max: xMax,
        time: { unit: 'minute', displayFormats: { minute: 'HH:mm', hour: 'HH:mm' } },
        ticks: { color: '#555', maxTicksLimit: 10, font: { size: 9 } },
        grid: { color: '#1a1a1a' }
      },
      y: { ticks: { color: '#555', font: { size: 9 } }, grid: { color: '#222' }, title: { display: true, text: yLabel, color: '#555', font: { size: 9 } } }
    }
  });

  // ── Price chart (same source as everything else: xbreak_signals.c) ──
  if (priceChart) { priceChart.destroy(); priceChart = null; }
  const pCtx = document.getElementById('signals-price-chart');
  if (pCtx && data.price && data.price.length) {
    const closePts = data.price.map(p => ({ x: new Date(p.ts), y: p.c }));
    const pDatasets = [
      { label: 'close', borderColor: '#81c784', data: closePts,
        pointRadius: 0, borderWidth: 1.5, fill: false, tension: 0.1 },
    ];
    priceChart = new Chart(pCtx, { type: 'line', data: { datasets: pDatasets },
      options: chartOpts(pair + ' Price (close, 30s ticks)', 'price') });
  }

  // ── Momentum / Accel / WSum charts ───────────────────────────
  if (momentumChart) momentumChart.destroy();
  if (accelChart) accelChart.destroy();
  if (wsChart) wsChart.destroy();

  const mDatasets = [], aDatasets = [];
  const wsDataset = { label: 'weighted_sum', borderColor: '#fff', data: [], pointRadius: 0, borderWidth: 1.5, tension: 0.3 };

  CHART_WINDOWS.forEach(w => {
    const pts = data.momentum[w] || [];
    mDatasets.push({ label: w, borderColor: CHART_COLORS[w], data: pts.map(p => ({x: new Date(p.ts), y: p.m})), pointRadius: 0, borderWidth: 1.5, tension: 0.3 });
    aDatasets.push({ label: w, borderColor: CHART_COLORS[w], data: pts.map(p => ({x: new Date(p.ts), y: p.a})), pointRadius: 0, borderWidth: 1, tension: 0.3, borderDash: [2, 2] });
  });
  const wspts = data.momentum[CHART_WINDOWS[0]] || [];
  wsDataset.data = wspts.map(p => ({x: new Date(p.ts), y: p.ws}));

  const mCtx = document.getElementById('signals-momentum-chart');
  const aCtx = document.getElementById('signals-accel-chart');
  const wCtx = document.getElementById('signals-ws-chart');
  if (mCtx) momentumChart = new Chart(mCtx, { type: 'line', data: { datasets: mDatasets }, options: chartOpts(pair + ' Momentum (pips/min) — click legend to toggle', 'pips/min') });
  if (aCtx) accelChart   = new Chart(aCtx, { type: 'line', data: { datasets: aDatasets }, options: chartOpts(pair + ' Acceleration (Δpips/min²) — click legend to toggle', 'Δpips/min²') });
  if (wCtx) wsChart      = new Chart(wCtx, { type: 'line', data: { datasets: [wsDataset] }, options: chartOpts(pair + ' Weighted Sum', 'pips/min') });
}

const PALETTE = ['#4caf50','#2196f3','#ff9800','#e91e63','#9c27b0','#00bcd4','#ff5722','#8bc34a','#ffc107','#3f51b5','#009688','#f44336','#673ab7'];

function switchPageTab(name) {
  document.querySelectorAll('.page-tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.page-tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('ptab-' + name).classList.add('active');
  document.getElementById('ptab-content-' + name).classList.add('active');
  if (name === 'signals') fetchSignals();
  if (name === 'tickmom') startTickMom(); else stopTickMom();
  if (name === 'markets') startMarkets(); else stopMarkets();
  if (name === 'bb') startBB(); else stopBB();
  if (name === 'streaming') startStreaming(); else stopStreaming();
}

// ─── BB-Fade tab (price + SMA9 ±1σ + re-entry signal markers) ──────────────
const BB_PAIRS=["EUR_USD","GBP_USD","AUD_USD","NZD_USD","EUR_GBP","USD_JPY","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","CHF_JPY"];
const bbCharts={}; let bbTimer=null;
function bbCells(){const g=document.getElementById('bb-grid'); if(g.children.length) return;
  BB_PAIRS.forEach(p=>{const d=document.createElement('div');
    d.style='background:#11151d;border:1px solid #222;border-radius:6px;padding:6px';
    d.innerHTML='<div style="font-weight:600;font-size:12px">'+p+' <span style="color:#f9e2af" id="bbs-'+p+'"></span></div><canvas id="bbc-'+p+'" height="140"></canvas>';
    g.appendChild(d);});}
async function loadBB(p){const tf=document.getElementById('bb-tf').value;
  let d; try{d=await(await fetch('/api/bbchart/'+p+'?tf='+tf)).json();}catch(e){return;}
  if(d.error){const e=document.getElementById('bbs-'+p); if(e)e.textContent='('+d.error+')';return;}
  const x=d.ts, sd=d.signals||[]; const lbl=document.getElementById('bbs-'+p);
  if(lbl) lbl.textContent = sd.length? (sd.length+' signal'+(sd.length>1?'s':'')) : '—';
  const sp=x.map((_,i)=>{const s=sd.find(z=>z.i===i);return s?s.p:null;});
  const ds=[{data:d.c,borderColor:'#89b4fa',borderWidth:1.2,pointRadius:0},
    {data:d.upper,borderColor:'#585b70',borderWidth:.8,pointRadius:0},
    {data:d.lower,borderColor:'#585b70',borderWidth:.8,pointRadius:0,fill:'-1',backgroundColor:'rgba(137,180,250,0.06)'},
    {data:d.basis,borderColor:'#f38ba8',borderWidth:.7,pointRadius:0,borderDash:[3,3]},
    {data:sp,borderColor:'transparent',backgroundColor:'#f9e2af',pointRadius:5,pointStyle:'triangle',showLine:false}];
  if(bbCharts[p]){const C=bbCharts[p];C.data.labels=x;ds.forEach((z,i)=>C.data.datasets[i].data=z.data);C.update('none');}
  else bbCharts[p]=new Chart(document.getElementById('bbc-'+p),{type:'line',data:{labels:x,datasets:ds},
    options:{animation:false,plugins:{legend:{display:false}},scales:{x:{ticks:{maxTicksLimit:6,color:'#666'}},y:{ticks:{color:'#666'}}}}});}
function loadBBall(){BB_PAIRS.forEach(loadBB);const s=document.getElementById('bb-status');if(s)s.textContent='updated '+new Date().toLocaleTimeString();}
function startBB(){bbCells();loadBBall(); if(!bbTimer) bbTimer=setInterval(loadBBall,30000);}
function stopBB(){if(bbTimer){clearInterval(bbTimer);bbTimer=null;}}


// ─── Streaming tab (live bid/ask, 1h buffer @1/s history, ~250ms live edge) ──
let strmChart=null, strmTimer=null, strmCtxTimer=null, strmPair=null, strmStart=null, strmInit=false;
let strmCtxBars=[], strmCtxTicks=[];   // higher-TF context bars + x-tick labels (each label = TF + pips/minute)
let strmH1=null;   // {x,c} of the 1H context bar — anchor for the connector into the live window
let strmM5=null;   // {x,c,tclose} of the 5M context bar — its close maps to a real point in the stream
// Connectors (drawn in pixel space because each end is on a different y-scale, so diagonal by design):
//   blue   = H1 close (context axis) → first live tick (live axis): where the strip ends / stream begins.
//   yellow = M5 close (context axis) → its actual time+price on the stream (live axis): where they meet.
const strmConnector={id:'strmConnector',afterDraw(chart){
  const d0=chart.data.datasets[0].data; if(!d0.length) return;
  const xs=chart.scales.x, ys=chart.scales.y, y2=chart.scales.y2; if(!ys||!y2) return;
  const c=chart.ctx; c.save(); c.lineWidth=1; c.setLineDash([3,3]);
  if(strmH1){ c.strokeStyle='#7aa2f7'; c.beginPath();
    c.moveTo(xs.getPixelForValue(strmH1.x), ys.getPixelForValue(strmH1.c));
    c.lineTo(xs.getPixelForValue(d0[0].x), y2.getPixelForValue(d0[0].y)); c.stroke(); }
  // only draw the M5 connector while its stream-side endpoint is still inside the live window;
  // once tclose scrolls off the left edge the line would otherwise reverse direction
  if(strmM5 && strmM5.tclose && strmM5.tclose>=d0[0].x && strmM5.tclose<=d0[d0.length-1].x){
    c.strokeStyle='#f9e2af'; c.beginPath();
    c.moveTo(xs.getPixelForValue(strmM5.x), ys.getPixelForValue(strmM5.c));
    c.lineTo(xs.getPixelForValue(strmM5.tclose), y2.getPixelForValue(strmM5.c)); c.stroke(); }
  c.restore();
}};
async function startStreaming(){
  if(!strmInit){
    strmInit=true;
    let pairs=[]; try{pairs=await(await fetch('/api/streaming/pairs')).json();}catch(e){}
    const sel=document.getElementById('strm-pair');
    sel.innerHTML=pairs.map(p=>'<option>'+p+'</option>').join('');
    const saved=localStorage.strmPair;
    if(saved && pairs.indexOf(saved)>=0) sel.value=saved;
    if(sel.value) strmLoad(sel.value);
  } else if(strmPair){ strmLoad(strmPair); }
}
function stopStreaming(){ if(strmTimer){clearInterval(strmTimer);strmTimer=null;} if(strmCtxTimer){clearInterval(strmCtxTimer);strmCtxTimer=null;} }
function strmDivider(){
  if(strmStart===null||!strmChart) return [];
  const d0=strmChart.data.datasets[0].data;
  if(!d0.length) return [];
  // Clamp to the live-window left edge: while the backfill is still visible the divider sits at the
  // real backfill→live boundary (strmStart); once strmStart scrolls out of the 1h window it tracks the
  // window edge instead of stranding far to the left and falling behind the rest of the chart.
  const x=Math.max(strmStart, d0[0].x);
  let lo=Infinity,hi=-Infinity;
  strmChart.data.datasets.slice(0,2).forEach(d=>d.data.forEach(p=>{if(p.y<lo)lo=p.y;if(p.y>hi)hi=p.y;}));
  return isFinite(lo)? [{x:x,y:lo},{x:x,y:hi}] : [];
}
function strmLevels(){   // thin horizontal lines at the CURRENT bid/ask, spanning the whole chart
  if(!strmChart) return [[],[]];
  const d0=strmChart.data.datasets[0].data, d1=strmChart.data.datasets[1].data;
  if(!d0.length) return [[],[]];
  let xmin=d0[0].x; const xmax=d0[d0.length-1].x;
  if(strmCtxTicks.length) xmin=Math.min(xmin, strmCtxTicks[0].value);   // extend across context strip
  const yb=d0[d0.length-1].y, ya=d1[d1.length-1].y;
  return [[{x:xmin,y:yb},{x:xmax,y:yb}], [{x:xmin,y:ya},{x:xmax,y:ya}]];
}
function strmContextSegments(){   // OHLC bar segments for the cached TF bars, glued just left of live
  strmCtxTicks=[]; strmH1=null; strmM5=null;
  if(!strmChart) return [];
  const d0=strmChart.data.datasets[0].data;
  if(!d0.length || !strmCtxBars.length) return [];
  const liveXmin=d0[0].x, liveXmax=d0[d0.length-1].x;
  const liveSpan=Math.max(liveXmax-liveXmin, 60000);
  const stripW=0.6*liveSpan, slotW=stripW/strmCtxBars.length, tick=slotW*0.28;
  const MIN={'1W':10080,'1D':1440,'8H':480,'4H':240,'1H':60,'5M':5};   // minutes per bar
  const pip=(strmPair && strmPair.indexOf('JPY')>=0)?0.01:0.0001;
  const d1=strmChart.data.datasets[1].data;                            // current mid = where price sits now
  const curPrice=d1.length?(d0[d0.length-1].y+d1[d1.length-1].y)/2:d0[d0.length-1].y;
  const pts=[];
  strmCtxBars.forEach((b,i)=>{
    const cx=liveXmin - stripW + (i+0.5)*slotW;
    const m=MIN[b.tf]||1, pm=(b.c-b.o)/pip/m;                 // pm = pips/minute = (close-open)/pip/minutes
    const rng=b.h-b.l, eff=rng>0?Math.abs(b.c-b.o)/rng:0;     // efficiency = |close-open|/(high-low), 0..1
    const loc=rng>0?(curPrice-b.l)/rng:0.5; // price location in bar range: 0=low,1=high,0.5=mid; >1 above bar, <0 below (fraction=% of range)
    strmCtxTicks.push({value:cx,label:b.tf,pm:pm,eff:eff,loc:loc});
    if(b.tf==='1H') strmH1={x:cx,c:b.c};                      // anchor for the H1 connector line
    if(b.tf==='5M') strmM5={x:cx,c:b.c,tclose:(Date.parse(b.t)||0)+300000};  // M5 close → its time in the stream
    pts.push({x:cx,y:b.l},{x:cx,y:b.h},{x:cx,y:null});        // high-low wick
    pts.push({x:cx-tick,y:b.o},{x:cx,y:b.o},{x:cx,y:null});   // open tick (left)
    pts.push({x:cx,y:b.c},{x:cx+tick,y:b.c},{x:cx,y:null});   // close tick (right)
  });
  strmCtxTicks.unshift({value:liveXmin - stripW - 0.5*slotW, header:true});  // row-name key, half a slot left of the strip
  return pts;
}
function strmDrawContext(){ if(strmChart) strmChart.data.datasets[5].data=strmContextSegments(); }
async function strmLoad(pair){
  strmPair=pair; localStorage.strmPair=pair;
  if(strmTimer){clearInterval(strmTimer);strmTimer=null;}
  if(strmCtxTimer){clearInterval(strmCtxTimer);strmCtxTimer=null;}
  let d; try{d=await(await fetch('/api/streaming/'+pair)).json();}catch(e){return;}
  strmStart=(d.stream_start_ms===undefined)?null:d.stream_start_ms;
  const bid=d.points.map(p=>({x:p[0],y:p[1]})), ask=d.points.map(p=>({x:p[0],y:p[2]}));
  if(strmChart) strmChart.destroy();
  strmChart=new Chart(document.getElementById('strmChart'),{type:'line',
    data:{datasets:[
      {label:'bid',data:bid,borderColor:'#f38ba8',borderWidth:1,pointRadius:0,tension:0,yAxisID:'y2'},
      {label:'ask',data:ask,borderColor:'#a6e3a1',borderWidth:1,pointRadius:0,tension:0,yAxisID:'y2'},
      {label:'stream start',data:[],borderColor:'#888',borderWidth:1,borderDash:[4,4],pointRadius:0,yAxisID:'y2'},
      {label:'bid level',data:[],borderColor:'#f38ba8',borderWidth:0.6,borderDash:[2,3],pointRadius:0,yAxisID:'y2'},
      {label:'ask level',data:[],borderColor:'#a6e3a1',borderWidth:0.6,borderDash:[2,3],pointRadius:0,yAxisID:'y2'},
      {label:'context 1W·1D·8H·4H·1H·5M',data:[],borderColor:'#7aa2f7',borderWidth:1,pointRadius:0,spanGaps:false,yAxisID:'y'}]},
    options:{animation:false,parsing:false,maintainAspectRatio:false,
      interaction:{mode:'nearest',intersect:false},
      plugins:{legend:{display:true,labels:{color:'#888',boxWidth:12,font:{size:10},filter:i=>i.datasetIndex!==3&&i.datasetIndex!==4}}},
      scales:{x:{type:'linear',
        afterBuildTicks:axis=>{ if(!strmCtxTicks.length||!strmChart) return;
          const d0=strmChart.data.datasets[0].data, lt=[];
          if(d0.length){const a=d0[0].x,b=d0[d0.length-1].x; for(let k=0;k<=4;k++) lt.push({value:a+(b-a)*k/4});}
          axis.ticks=strmCtxTicks.concat(lt); },
        ticks:{color:function(c){ const t=c.tick&&strmCtxTicks.find(z=>z.value===c.tick.value); if(!t) return '#666'; if(t.header) return '#cdd6f4'; return t.pm>=0?'#a6e3a1':'#f38ba8'; },autoSkip:false,maxRotation:0,
          callback:function(v){ const t=strmCtxTicks.find(z=>z.value===v); if(!t) return new Date(v).toISOString().substr(11,8); return t.header?['TF','p/m','eff','loc']:[t.label,(t.pm>=0?'+':'')+t.pm.toFixed(2),'η'+t.eff.toFixed(2),'↕'+(t.loc>=0?'+':'')+t.loc.toFixed(2)]; }}},
        y:{position:'left',grid:{color:'#1a1a1a'},ticks:{color:'#7aa2f7'},title:{display:true,text:'context',color:'#7aa2f7',font:{size:9}}},
        y2:{position:'right',grid:{drawOnChartArea:false},ticks:{color:'#a6e3a1'},title:{display:true,text:'live',color:'#a6e3a1',font:{size:9}}}}},
    plugins:[strmConnector]});
  strmChart.data.datasets[2].data=strmDivider();
  const lv0=strmLevels(); strmChart.data.datasets[3].data=lv0[0]; strmChart.data.datasets[4].data=lv0[1];
  strmChart.update('none');
  try{ const cd=await(await fetch('/api/streaming/context/'+pair)).json(); strmCtxBars=cd.bars||[]; }
  catch(e){ strmCtxBars=[]; }
  strmDrawContext();
  const lv1=strmLevels(); strmChart.data.datasets[3].data=lv1[0]; strmChart.data.datasets[4].data=lv1[1];
  strmChart.update('none');
  strmTimer=setInterval(strmTick,300);   // ~300ms live poll; live edge updates at tick speed
  // refresh the higher-TF context bars every 30s so they (and the pips/min + connector anchors) don't go stale
  strmCtxTimer=setInterval(async()=>{ if(!strmPair) return;
    try{ const cd=await(await fetch('/api/streaming/context/'+strmPair)).json(); strmCtxBars=cd.bars||[]; }catch(e){} },30000);
}
async function strmTick(){
  if(!strmPair||!strmChart) return;
  let all; try{all=await(await fetch('/api/streaming/latest')).json();}catch(e){return;}
  const p=all[strmPair]; if(!p) return;
  const ds=strmChart.data.datasets, d0=ds[0].data, d1=ds[1].data;
  const lastX=d0.length?d0[d0.length-1].x:0;
  if(d0.length && Math.floor(p[0]/1000)===Math.floor(lastX/1000)){
    d0[d0.length-1]={x:p[0],y:p[1]}; d1[d1.length-1]={x:p[0],y:p[2]};   // live edge: update current second
  } else if(p[0]>lastX){
    d0.push({x:p[0],y:p[1]}); d1.push({x:p[0],y:p[2]});                 // new second: lock prev, append
    if(strmStart===null) strmStart=p[0];
    const cut=p[0]-3600000;
    ds[0].data=d0.filter(q=>q.x>=cut); ds[1].data=d1.filter(q=>q.x>=cut);
  }
  ds[2].data=strmDivider();
  strmDrawContext();
  const lv=strmLevels(); ds[3].data=lv[0]; ds[4].data=lv[1];
  strmChart.update('none');
  const age=(Date.now()-p[0])/1000, badge=document.getElementById('strm-stale');
  badge.textContent = age>10? ('⚠ stale '+age.toFixed(0)+'s'):'● live';
  badge.style.color = age>10? '#f9e2af':'#a6e3a1';
}


// ─── Tick Momentum tab (live, ~250ms OANDA stream → multi-window) ──────────
let tmTimer = null;
const tmCharts = {};       // key: 'm-'+W+'-'+pair / 's-'+W+'-'+pair
const tmStrCharts = {};    // key: W
const TM_CCY = ['USD','EUR','GBP','JPY','CHF','AUD','NZD','CAD'];
const TM_CCY_COLOR = {USD:'#4ade80',EUR:'#60a5fa',GBP:'#a78bfa',JPY:'#f87171',
                      CHF:'#fb923c',AUD:'#2dd4bf',NZD:'#f472b6',CAD:'#facc15'};
function tmDir(d){ if(d>0) return '<span style="color:#4ade80;">▲</span>';
                  if(d<0) return '<span style="color:#f87171;">▼</span>';
                  return '<span style="color:#888;">─</span>'; }
function tmStrengthColor(score, mx){
  if(mx<=0) return '#888';
  const f=Math.max(-1,Math.min(1,score/mx));
  if(f>=0){ const g=Math.round(120+135*f), o=Math.round(60+60*f); return 'rgb('+o+','+g+','+o+')'; }
  const r=Math.round(120+135*(-f)), o=Math.round(60+60*(-f)); return 'rgb('+r+','+o+','+o+')';
}
// build the two (or N) columns once, one per window
function tmEnsureCols(wins){
  const host=document.getElementById('tm-cols'); if(!host) return;
  if(host.dataset.wins===wins.join(',')) return;
  host.dataset.wins=wins.join(','); host.innerHTML='';
  for(const w of wins){
    const col=document.createElement('div');
    col.innerHTML =
      '<div style="font-size:13px;font-weight:600;color:#ddd;margin-bottom:8px;border-bottom:1px solid #222;padding-bottom:4px;">'+w+'s window</div>'+
      '<div style="font-size:9px;color:#666;text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px;">Currency strength (bp/min · strongest → weakest)</div>'+
      '<div id="tm-ladder-'+w+'" style="display:flex;flex-wrap:wrap;align-items:center;gap:3px;font-size:14px;font-weight:600;margin-bottom:8px;">…</div>'+
      '<canvas id="tm-strchart-'+w+'" style="max-height:150px;margin-bottom:14px;"></canvas>'+
      '<div id="tm-grid-'+w+'" style="display:grid;grid-template-columns:1fr;gap:12px;"></div>';
    host.appendChild(col);
  }
}
function tmRenderLadder(w, arr){
  const el=document.getElementById('tm-ladder-'+w); if(!el) return;
  if(!arr.length){ el.innerHTML='<span style="color:#555;font-weight:400;font-size:12px;">waiting for ticks…</span>'; return; }
  const mx=Math.max(...arr.map(x=>Math.abs(x.score)), 1e-9);
  el.innerHTML = arr.map((x,i)=>{
    const c=tmStrengthColor(x.score,mx);
    const chip='<span style="color:'+c+';">'+x.ccy+'<span style="font-size:9px;color:#777;font-weight:400;"> '+(x.score>0?'+':'')+x.score+'</span></span>';
    return i? '<span style="color:#444;font-weight:400;">▸</span>'+chip : chip;
  }).join('');
}
function tmRenderStrChart(w, tr){
  const labels=tr.map(x=>x.t.slice(11,19));
  const datasets=TM_CCY.map(c=>({label:c, data:tr.map(p=>(c in p)?p[c]:null),
    borderColor:TM_CCY_COLOR[c], spanGaps:true, borderWidth:1.3}));
  if(!tmStrCharts[w]){
    tmStrCharts[w]=new Chart(document.getElementById('tm-strchart-'+w),{type:'line',
      data:{labels:labels,datasets:datasets},
      options:{animation:false,responsive:true,
        plugins:{legend:{labels:{color:'#999',boxWidth:8,font:{size:9},padding:5}}},
        scales:{x:{display:false},y:{ticks:{color:'#666',font:{size:9}},grid:{color:'#181818'}}},
        elements:{point:{radius:0}}}});
  } else {
    const ch=tmStrCharts[w]; ch.data.labels=labels;
    datasets.forEach((ds,i)=>{ ch.data.datasets[i].data=ds.data; }); ch.update('none');
  }
}
function tmMetric(key,w,pair,lbl){
  return '<div style="display:flex;flex-direction:column;">'+
    '<span style="color:#666;font-size:10px;text-transform:uppercase;letter-spacing:.5px;">'+(lbl||key)+'</span>'+
    '<span style="font-size:14px;font-weight:600;" id="tm-'+key+'-'+w+'-'+pair+'">—</span></div>';
}
function tmEnsureCard(w, pair){
  let el=document.getElementById('tm-card-'+w+'-'+pair);
  if(el) return el;
  el=document.createElement('div'); el.id='tm-card-'+w+'-'+pair;
  el.style.cssText='background:#111;border:1px solid #222;border-radius:8px;padding:10px 12px;';
  el.innerHTML =
    '<div style="font-size:13px;font-weight:600;display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">'+
      pair+' <span style="font-size:12px;color:#bbb;font-weight:400;" id="tm-px-'+w+'-'+pair+'">—</span></div>'+
    '<div style="display:flex;gap:12px;font-size:12px;margin-bottom:6px;flex-wrap:wrap;">'+
      tmMetric('dir',w,pair)+tmMetric('mpm',w,pair,'pips/min')+tmMetric('n',w,pair,'ticks')+
      tmMetric('r',w,pair,'rate /s')+tmMetric('sp',w,pair,'spread p')+tmMetric('spr',w,pair,'sp x̄/rel')+
    '</div>'+
    '<canvas id="tm-cm-'+w+'-'+pair+'" style="max-height:100px;"></canvas>'+
    '<canvas id="tm-cs-'+w+'-'+pair+'" style="max-height:100px;"></canvas>';
  document.getElementById('tm-grid-'+w).appendChild(el);
  return el;
}
function tmMkChart(id,labels,cfgs){
  return new Chart(document.getElementById(id),{type:'line',data:{labels:labels,datasets:cfgs},
    options:{animation:false,responsive:true,plugins:{legend:{labels:{color:'#888',boxWidth:10,font:{size:9}}}},
      scales:{x:{display:false},y:{ticks:{color:'#666',font:{size:9}},grid:{color:'#1a1a1a'}},
              y1:{position:'right',ticks:{color:'#666',font:{size:9}},grid:{display:false}}},
      elements:{point:{radius:0},line:{borderWidth:1.4}}}});
}
async function fetchTickMom(){
  let d;
  try { d = await (await fetch('/api/tick_momentum')).json(); }
  catch(e){ document.getElementById('tm-ts').textContent='fetch error'; return; }
  if(d.error){ document.getElementById('tm-ts').textContent=d.error; return; }
  const wins=Object.keys(d.windows||{}).sort((a,b)=>(+a)-(+b));
  if(!wins.length){ document.getElementById('tm-ts').textContent='waiting for stream…'; return; }
  document.getElementById('tm-ts').textContent='updated '+(d.updated||'').replace('T',' ').slice(0,19)+' UTC';
  tmEnsureCols(wins);
  for(const w of wins){
    const W=d.windows[w];
    tmRenderLadder(w, W.strength||[]);
    tmRenderStrChart(w, W.strength_trace||[]);
    for(const pair of Object.keys(W.pairs||{})){
      const p=W.pairs[pair]; tmEnsureCard(w,pair);
      document.getElementById('tm-px-'+w+'-'+pair).textContent = p.mid!=null ? p.mid : '—';
      document.getElementById('tm-dir-'+w+'-'+pair).innerHTML = tmDir(p.dir);
      const me=document.getElementById('tm-mpm-'+w+'-'+pair);
      me.textContent=(p.mpm>0?'+':'')+p.mpm; me.style.color = p.mpm>0?'#4ade80':(p.mpm<0?'#f87171':'#888');
      document.getElementById('tm-n-'+w+'-'+pair).textContent=p.ticks;
      document.getElementById('tm-r-'+w+'-'+pair).textContent=p.rate;
      const spe=document.getElementById('tm-sp-'+w+'-'+pair);
      spe.textContent=p.spread; spe.style.color = p.spread_rel>1.5 ? '#fbbf24' : '#e0e0e0';
      document.getElementById('tm-spr-'+w+'-'+pair).textContent=p.spread_avg+' / '+p.spread_rel+'×';
      const tr=p.trace||[]; const labels=tr.map(x=>x.t.slice(11,19));
      const mpm=tr.map(x=>x.m), rate=tr.map(x=>x.n), sp=tr.map(x=>x.sp);
      const km='m-'+w+'-'+pair, ks='s-'+w+'-'+pair;
      if(!tmCharts[km]){
        tmCharts[km]=tmMkChart('tm-cm-'+w+'-'+pair,labels,[
          {label:'pips/min',data:mpm,borderColor:'#60a5fa',yAxisID:'y'},
          {label:'ticks/win',data:rate,borderColor:'#a78bfa',yAxisID:'y1'}]);
        tmCharts[ks]=tmMkChart('tm-cs-'+w+'-'+pair,labels,[
          {label:'spread (p)',data:sp,borderColor:'#fbbf24',yAxisID:'y'}]);
      } else {
        const cm=tmCharts[km]; cm.data.labels=labels;
        cm.data.datasets[0].data=mpm; cm.data.datasets[1].data=rate; cm.update('none');
        const cs=tmCharts[ks]; cs.data.labels=labels; cs.data.datasets[0].data=sp; cs.update('none');
      }
    }
  }
}
function startTickMom(){ if(tmTimer) return; fetchTickMom(); tmTimer=setInterval(fetchTickMom,750); }
function stopTickMom(){ if(tmTimer){ clearInterval(tmTimer); tmTimer=null; } }


// ─── Markets ⚡ : combined momentum-signals + livestream across timeframes ──
const MK_TFS = [
  {k:'5',  label:'5s',  src:'tick', tk:'5'},
  {k:'30', label:'30s', src:'tick', tk:'30'},
  {k:'5m', label:'5m',  src:'sig'},
  {k:'15m',label:'15m', src:'sig'},
  {k:'1h', label:'1h',  src:'sig'},
  {k:'4h', label:'4h',  src:'sig'},
  {k:'24h',label:'24h', src:'sig'},
];
const MK_CCY = ['USD','EUR','GBP','JPY','CHF','AUD','NZD','CAD'];
let mkTimer=null, mkTick=null, mkSig=null, mkSel='1h', mkPoll=0;
const mkCharts={}; let mkStrChart=null;
function mkTF(k){ return MK_TFS.find(t=>t.k===k) || MK_TFS[0]; }
function mkStrengthFor(tf){
  if(tf.src==='tick'){
    const w = mkTick && mkTick.windows && mkTick.windows[tf.tk];
    return (w && w.strength) ? w.strength.map(x=>({ccy:x.ccy,score:x.score})) : [];
  }
  if(!mkSig || !mkSig.csi) return [];
  const out=[];
  for(const c of MK_CCY){ const v=mkSig.csi[c] && mkSig.csi[c][tf.k]; if(v!==null&&v!==undefined) out.push({ccy:c,score:v}); }
  return out.sort((a,b)=>b.score-a.score);
}
function mkRenderMatrix(){
  const el=document.getElementById('mk-matrix'); if(!el) return;
  let h='<thead><tr><th class="mk-tfh">TF</th>'+MK_CCY.map(c=>'<th style="color:'+TM_CCY_COLOR[c]+'">'+c+'</th>').join('')+'</tr></thead><tbody>';
  for(const tf of MK_TFS){
    const arr=mkStrengthFor(tf); const m={}; arr.forEach(x=>m[x.ccy]=x.score);
    const mx=Math.max(...arr.map(x=>Math.abs(x.score)),1e-9);
    const sel = tf.k===mkSel ? ' sel' : '';
    h+='<tr class="mk-tfrow'+sel+'" data-tf="'+tf.k+'"><td class="mk-tflabel">'+tf.label+'</td>';
    for(const c of MK_CCY){
      if(c in m){ const col=tmStrengthColor(m[c],mx);
        h+='<td><div class="mk-cell" style="background:'+col+'">'+c+'<small>'+(m[c]>0?'+':'')+(Math.abs(m[c])>=10?Math.round(m[c]):m[c].toFixed(1))+'</small></div></td>';
      } else { h+='<td><div class="mk-cell" style="background:#1a1a1a;color:#444">·</div></td>'; }
    }
    h+='</tr>';
  }
  el.innerHTML=h+'</tbody>';
}
function mkRenderPills(){
  const el=document.getElementById('mk-pills'); if(!el) return;
  el.innerHTML=MK_TFS.map(tf=>{
    const live=tf.src==='tick'?'<span class="mk-pdot"> ●</span>':'';
    return '<button class="mk-pill'+(tf.k===mkSel?' sel':'')+'" data-tf="'+tf.k+'">'+tf.label+live+'</button>';
  }).join('');
}
function mkLadderHTML(arr){
  if(!arr.length) return '<span style="color:#555;font-weight:400;font-size:13px;">waiting for data…</span>';
  const mx=Math.max(...arr.map(x=>Math.abs(x.score)),1e-9);
  return arr.map((x,i)=>{ const c=tmStrengthColor(x.score,mx);
    const chip='<span style="color:'+c+'">'+x.ccy+'<span style="font-size:9px;color:#777;font-weight:400"> '+(x.score>0?'+':'')+x.score+'</span></span>';
    return i?'<span style="color:#444;font-weight:400">▸</span>'+chip:chip; }).join('');
}
function mkBarsHTML(rows){
  const mx=Math.max(...rows.map(r=>Math.abs(r.v)),1e-9);
  return rows.map(r=>{ const w=Math.abs(r.v)/mx*50; const pos=r.v>=0;
    const col=pos?'#4ade80':'#f87171';
    const fill='<div class="mk-bar-fill" style="background:'+col+';'+(pos?'left:50%':'right:50%')+';width:'+w+'%"></div>';
    return '<div class="mk-bar-row"><span class="mk-bar-name">'+r.name+'</span>'+
      '<span class="mk-bar-track"><span style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#333"></span>'+fill+'</span>'+
      '<span class="mk-bar-val" style="color:'+col+'">'+(pos?'+':'')+r.v.toFixed(2)+'</span></div>';
  }).join('');
}
function mkCsiHTML(arr){
  if(!arr || !arr.length) return '<span style="color:#555;font-weight:400;font-size:12px;">n/a for this timeframe</span>';
  const mx=Math.max(...arr.map(x=>x.csi),1e-9);
  return arr.slice(0,10).map((x,i)=>{
    const f=Math.max(0,Math.min(1,x.csi/mx)); const g=Math.round(120+135*f), o=Math.round(45+55*f);
    const col='rgb('+o+','+g+','+o+')';
    const chip='<span style="color:'+col+'">'+x.pair+'<span style="font-size:9px;color:#777;font-weight:400"> '+x.csi+'</span></span>';
    return i?'<span style="color:#444;font-weight:400">▸</span>'+chip:chip;
  }).join('');
}
function mkDestroyFocusCharts(){
  for(const k in mkCharts){ try{mkCharts[k].destroy();}catch(e){} delete mkCharts[k]; }
  if(mkStrChart){ try{mkStrChart.destroy();}catch(e){} mkStrChart=null; }
}
function mkSelect(k){
  if(k===mkSel){ return; }
  mkSel=k; mkDestroyFocusCharts();
  document.getElementById('mk-focus').dataset.tf='';
  mkRenderMatrix(); mkRenderPills(); mkRenderFocus();
}
function mkEnsureFocusSkeleton(tf){
  const host=document.getElementById('mk-focus');
  if(host.dataset.tf===tf.k) return;
  mkDestroyFocusCharts(); host.dataset.tf=tf.k;
  let h='<div class="mk-panel"><div class="mk-panel-lbl">'+tf.label+' strength (strongest → weakest)</div>'+
        '<div class="mk-ladder" id="mk-ladder"></div>';
  if(tf.src==='tick'){
    h+='<div class="mk-panel-lbl" style="margin-top:8px;">strength over time (bp/min per currency)</div>'+
       '<canvas id="mk-strchart" style="max-height:160px;"></canvas>';
  }
  h+='</div>';
  // CSI panel (most tradeable pairs) — tick analog for 5s/30s, Wilder for candles
  h+='<div class="mk-panel"><div class="mk-panel-lbl">CSI — most tradeable pairs '+
     (tf.src==='tick' ? '· tick analog (efficiency × range × margin-eff)'
                      : '· Wilder ADXR × ATR × (V/√M) ÷ (150+C)')+'</div>'+
     '<div class="mk-ladder" id="mk-csi" style="font-size:14px;">…</div></div>';
  if(tf.src==='tick'){
    h+='<div class="mk-panel-lbl">Featured pairs — momentum (pips/min) + spread</div>'+
       '<div class="mk-grid" id="mk-pairgrid"></div>';
  } else {
    h+='<div class="mk-panel"><div class="mk-panel-lbl">Pair momentum @ '+tf.label+' (pips/min · sorted)</div>'+
       '<div id="mk-bars"></div></div>';
  }
  host.innerHTML=h;
}
function mkRenderFocus(){
  const tf=mkTF(mkSel); mkEnsureFocusSkeleton(tf);
  document.getElementById('mk-ladder').innerHTML=mkLadderHTML(mkStrengthFor(tf));
  const csiArr = tf.src==='tick'
    ? ((mkTick&&mkTick.windows&&mkTick.windows[tf.tk]&&mkTick.windows[tf.tk].csi)||[])
    : ((mkSig&&mkSig.csi_pairs&&mkSig.csi_pairs[tf.k])||[]);
  const csiEl=document.getElementById('mk-csi'); if(csiEl) csiEl.innerHTML=mkCsiHTML(csiArr);
  if(tf.src==='tick'){
    const w=mkTick && mkTick.windows && mkTick.windows[tf.tk]; if(!w) return;
    const tr=w.strength_trace||[]; const labels=tr.map(x=>x.t.slice(11,19));
    const ds=MK_CCY.map(c=>({label:c,data:tr.map(p=>(c in p)?p[c]:null),borderColor:TM_CCY_COLOR[c],spanGaps:true,borderWidth:1.3}));
    if(!mkStrChart){
      mkStrChart=new Chart(document.getElementById('mk-strchart'),{type:'line',data:{labels:labels,datasets:ds},
        options:{animation:false,responsive:true,plugins:{legend:{labels:{color:'#999',boxWidth:8,font:{size:9},padding:5}}},
          scales:{x:{display:false},y:{ticks:{color:'#666',font:{size:9}},grid:{color:'#181818'}}},elements:{point:{radius:0}}}});
    } else { mkStrChart.data.labels=labels; ds.forEach((d,i)=>{mkStrChart.data.datasets[i].data=d.data;}); mkStrChart.update('none'); }
    const grid=document.getElementById('mk-pairgrid');
    for(const pair of Object.keys(w.pairs||{})){
      const p=w.pairs[pair];
      let card=document.getElementById('mk-card-'+pair);
      if(!card){ card=document.createElement('div'); card.className='mk-card'; card.id='mk-card-'+pair;
        card.innerHTML='<div style="font-size:13px;font-weight:600;display:flex;justify-content:space-between;margin-bottom:4px;">'+pair+
          ' <span style="font-weight:400;color:#bbb" id="mk-px-'+pair+'">—</span></div>'+
          '<div style="display:flex;gap:12px;font-size:12px;margin-bottom:6px;">'+
          '<span>mom <b id="mk-mom-'+pair+'">—</b></span><span style="color:#888">spr <b id="mk-spr-'+pair+'" style="color:#e0e0e0">—</b></span></div>'+
          '<canvas id="mk-cm-'+pair+'" style="max-height:90px;"></canvas>';
        grid.appendChild(card); }
      document.getElementById('mk-px-'+pair).textContent=p.mid!=null?p.mid:'—';
      const mo=document.getElementById('mk-mom-'+pair); mo.textContent=(p.mpm>0?'+':'')+p.mpm; mo.style.color=p.mpm>0?'#4ade80':(p.mpm<0?'#f87171':'#888');
      const se=document.getElementById('mk-spr-'+pair); se.textContent=p.spread; se.style.color=p.spread_rel>1.5?'#fbbf24':'#e0e0e0';
      const trp=p.trace||[]; const ml=trp.map(x=>x.t.slice(11,19)); const mv=trp.map(x=>x.m); const sv=trp.map(x=>x.sp);
      const km='mc-'+pair;
      if(!mkCharts[km]){ mkCharts[km]=tmMkChart('mk-cm-'+pair,ml,[
        {label:'pips/min',data:mv,borderColor:'#60a5fa',yAxisID:'y'},
        {label:'spread',data:sv,borderColor:'#fbbf24',yAxisID:'y1'}]);
      } else { const ch=mkCharts[km]; ch.data.labels=ml; ch.data.datasets[0].data=mv; ch.data.datasets[1].data=sv; ch.update('none'); }
    }
  } else {
    if(!mkSig||!mkSig.pairs) return;
    const rows=[];
    for(const pair of Object.keys(mkSig.pairs)){ const v=mkSig.pairs[pair][tf.k]; if(v!==null&&v!==undefined) rows.push({name:pair,v:v}); }
    rows.sort((a,b)=>b.v-a.v);
    document.getElementById('mk-bars').innerHTML=mkBarsHTML(rows);
  }
}
async function fetchMarkets(){
  try { mkTick = await (await fetch('/api/tick_momentum')).json(); if(mkTick.error) mkTick=null; }
  catch(e){ mkTick=null; }
  if(mkPoll%7===0){ try{ const s=await (await fetch('/api/signals')).json(); if(!s.error) mkSig=s; }catch(e){} }
  mkPoll++;
  const t=(mkTick&&mkTick.updated)||(mkSig&&mkSig.ts)||'';
  document.getElementById('mk-ts').textContent = t ? 'updated '+t.replace('T',' ').slice(0,19)+' UTC' : 'waiting for data…';
  mkRenderMatrix(); mkRenderPills(); mkRenderFocus();
}
function mkBindOnce(){
  if(window._mkBound) return; window._mkBound=true;
  const handler=e=>{ const t=e.target.closest('[data-tf]'); if(t) mkSelect(t.dataset.tf); };
  const mx=document.getElementById('mk-matrix'); if(mx) mx.addEventListener('click',handler);
  const pl=document.getElementById('mk-pills'); if(pl) pl.addEventListener('click',handler);
}
function startMarkets(){ if(mkTimer) return; mkBindOnce(); const f=document.getElementById('mk-focus'); if(f) f.dataset.tf=''; fetchMarkets(); mkTimer=setInterval(fetchMarkets,750); }
function stopMarkets(){ if(mkTimer){ clearInterval(mkTimer); mkTimer=null; } mkDestroyFocusCharts(); }


// fetchPortfolio() removed — Portfolio Paper tab cut (history retained in trades.duckdb)


function setPreset(btn) {
  // Clear custom since-input so preset takes effect
  document.getElementById('since-input').value = '';
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  activePresetHours = parseInt(btn.dataset.hours);
  fetchEquity();
}

function onSinceChange() {
  // Custom datetime picked — deactivate preset buttons
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  fetchEquity();
}

function getCheckedAccounts() {
  return [...document.querySelectorAll('.acct-chk:checked')].map(el => el.value);
}

function computeClientSum(seriesMap) {
  const lists = Object.values(seriesMap);
  if (!lists.length) return [];
  const maps = lists.map(s => { const m = {}; s.forEach(p => m[p.t] = p.v); return m; });
  const allTs = [...new Set(lists.flatMap(s => s.map(p => p.t)))].sort();
  const lastVals = lists.map(() => 0);
  return allTs.map(t => {
    let total = 0;
    maps.forEach((m, i) => { if (m[t] !== undefined) lastVals[i] = m[t]; total += lastVals[i]; });
    return { t, v: Math.round(total * 10) / 10 };
  });
}

async function fetchAccounts() {
  try {
    const resp = await fetch('/api/accounts');
    if (!resp.ok) return;   // tolerate nginx HTML error page during container restarts
    const data = await resp.json();
    renderSummary(data.accounts);
    renderGrid(data.accounts);
    if (!equityCheckboxesPopulated) {
      const container = document.getElementById('acct-checkboxes');
      data.accounts.forEach((a, i) => {
        acctNames[a.label] = a.name;
        const color = acctColor(a.label) || PALETTE[i % PALETTE.length];
        const lbl = document.createElement('label');
        lbl.className = 'chk-label';
        const defChecked = a.label === '010' ? 'checked' : '';   // default view: account 010 only
        lbl.innerHTML = `<input type="checkbox" class="acct-chk" value="${a.label}" ${defChecked} onchange="fetchEquity()"> <span style="color:${color}">${a.label}</span>`;
        container.appendChild(lbl);
      });
      equityCheckboxesPopulated = true;
      fetchEquity();   // redraw once 010 is checked (checkboxes populate after first equity fetch)
    }
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
    if (selectedAcct) selectAccount(selectedAcct, true);
  } catch(e) { console.error('Fetch failed:', e); }
}

async function fetchPaper() {
  try {
    const resp = await fetch('/api/paper/stats');
    if (!resp.ok) return;
    const data = await resp.json();
    renderPaper(data);
  } catch(e) { console.error('Paper fetch failed:', e); }
}

async function fetchToday() {
  try {
    const resp = await fetch('/api/accounts/today');
    if (!resp.ok) return;
    const data = await resp.json();
    renderToday(data);
  } catch(e) { console.error('Today fetch failed:', e); }
}

function renderToday(data) {
  const el = document.getElementById('today-section');
  if (!el || !data || !data.rows) return;
  document.getElementById('today-day').textContent = data.day_utc + ' UTC';
  document.getElementById('today-meta').textContent = 'updated ' + (data.ts || '').replace('T',' ').slice(11,19);

  const fmtPips = (n, trades) => {
    if (!trades) return '<span style="color:#555;">—</span>';
    const cls = n >= 0 ? 'green' : 'red';
    return `<span class="${cls}">${n >= 0 ? '+' : ''}${n.toFixed(1)}</span>`;
  };
  const fmtUsd = (n) => {
    if (Math.abs(n) < 0.00005) return '<span style="color:#555;">$0.00</span>';
    const cls = n >= 0 ? 'green' : 'red';
    const abs = Math.abs(n);
    const txt = abs < 0.01 ? abs.toFixed(4) : abs.toFixed(2);
    return `<span class="${cls}">${n >= 0 ? '+' : '-'}$${txt}</span>`;
  };

  let html = '<table style="width:100%;border-collapse:collapse;font-size:12px;">';
  html += '<tr style="border-bottom:1px solid #333;color:#888;text-align:left;">';
  html += '<th style="padding:4px 8px;">Acct</th><th>Strategy</th>';
  html += '<th style="text-align:right;">Trades</th>';
  html += '<th style="text-align:right;">Pips</th>';
  html += '<th style="text-align:right;">USD</th>';
  html += '<th style="text-align:right;">Open</th>';
  html += '<th style="text-align:right;">NAV</th></tr>';
  data.rows.forEach(r => {
    let rowStyle = '';
    if (r.active === false) rowStyle = 'opacity:0.55;';
    else if (r.paused) rowStyle = 'background:#2a1d05;';
    const pauseBadge = r.paused
      ? ` <span style="color:#ffb74d;font-size:10px;font-weight:normal;" title="${(r.paused_reason||'').replace(/"/g,'&quot;')}">⏸ PAUSED</span>`
      : '';
    const labelColor = r.paused ? '#ffb74d' : '#ccc';
    const stratColor = r.paused ? '#d8a85f' : '#aaa';
    html += `<tr style="border-bottom:1px solid #1a1a1a;${rowStyle}">
      <td style="padding:4px 8px;font-weight:bold;color:${labelColor};">${r.label}</td>
      <td style="color:${stratColor};">${r.strategy}${pauseBadge}</td>
      <td style="text-align:right;color:#ccc;">${r.trades || '—'}</td>
      <td style="text-align:right;">${fmtPips(r.pips, r.trades)}</td>
      <td style="text-align:right;">${fmtUsd(r.usd)}</td>
      <td style="text-align:right;color:#ccc;">${r.open || 0}</td>
      <td style="text-align:right;color:#ccc;">$${(r.nav || 0).toFixed(2)}</td>
    </tr>`;
  });
  const t = data.totals || {};
  html += `<tr style="border-top:2px solid #333;font-weight:bold;background:#0e0e0e;">
      <td style="padding:6px 8px;color:#fff;">Live total</td>
      <td></td>
      <td style="text-align:right;color:#fff;">${t.trades || 0}</td>
      <td style="text-align:right;">${fmtPips(t.pips || 0, t.trades || 0)}</td>
      <td style="text-align:right;">${fmtUsd(t.usd || 0)}</td>
      <td></td><td></td></tr>`;
  html += '</table>';
  el.innerHTML = html;
}

function renderPaper(data) {
  const el = document.getElementById('paper-section');
  if (!el) return;

  const stats = data.label_stats || {};
  const svc = data.service_status || {};

  // Collect configs from ZR paper status (uses "strategies" key)
  const zrConfigs = [];
  const zrSvc = svc.zr || {};
  (zrSvc.strategies || []).forEach(c => {
    const lbl = c.label || '?';
    let posStr;
    if (c.cycle_open) {
      const dir = c.cycle_dir > 0 ? 'LONG' : 'SHORT';
      posStr = `${dir} ${c.cycle_legs}L${c.cycle_psar ? ' PSAR' : ''}`;
    } else { posStr = 'flat'; }
    zrConfigs.push({
      label: lbl, pair: c.pair || 'GBP_USD', type: 'ZR',
      pos: posStr, pnf: '', bars: c.bar_count || 0,
      svc_trades: c.n_cycles || 0, svc_pips: c.total_pips || 0,
    });
  });

  // Collect configs from FIFO paper status (also "strategies" key)
  const fifoConfigs = [];
  const fifoSvc = svc.fifo || {};
  (fifoSvc.strategies || []).forEach(c => {
    const lbl = c.label || '?';
    const pos = c.pos > 0 ? 'LONG' : c.pos < 0 ? 'SHORT' : 'flat';
    const pnfStr = `P&F ${c.pnf_dir > 0 ? '+' : '-'}${c.col_count} @ ${(c.pnf_level||0).toFixed(1)}`;
    fifoConfigs.push({
      label: lbl, pair: c.pair || '?', type: 'FIFO',
      pos, pnf: pnfStr, bars: c.bar_count || 0,
      svc_trades: c.n_trades || 0, svc_pips: c.total_pips || 0,
      oos_ref: c.oos_pd_ref ? c.oos_pd_ref + 'p/d' : '',
    });
  });

  // FIFO Live (account 013 — real orders, flat status object)
  const liveFifoConfigs = [];
  const fifoLiveSvc = svc.fifo_live || {};
  if (fifoLiveSvc.label) {
    const c = fifoLiveSvc;
    const pos = c.pos > 0 ? 'LONG' : c.pos < 0 ? 'SHORT' : 'flat';
    const pnfStr = `P&F ${c.pnf_dir > 0 ? '+' : '-'}${c.col_count || 0} @ ${(c.pnf_level||0).toFixed(3)}`;
    const tidStr = c.oanda_tid ? ` tid=${c.oanda_tid}` : '';
    liveFifoConfigs.push({
      label: c.label, pair: c.pair || 'USD_JPY', type: '🟢LIVE',
      pos: pos + (c.pos !== 0 && c.entry_px ? ` @${c.entry_px}` : ''),
      pnf: pnfStr + tidStr, bars: c.bar_count || 0,
      svc_trades: c.n_trades || 0, svc_pips: c.total_pips || 0,
      oos_ref: c.oos_pd_ref ? c.oos_pd_ref + 'p/d' : '',
      is_live: true,
    });
  }

  // Collect configs from TR Momentum paper status
  const trConfigs = [];
  const trSvc = svc.tr || {};
  (trSvc.strategies || []).forEach(c => {
    const lbl = c.label || '?';
    const pos = c.pos > 0 ? 'LONG' : c.pos < 0 ? 'SHORT' : 'flat';
    const posDetail = c.pos !== 0 && c.entry_px
      ? `${pos} @${c.entry_px} hw=${c.hw}` : pos;
    trConfigs.push({
      label: lbl, pair: c.pair || '?', type: 'TR',
      pos: posDetail, pnf: '', bars: c.bar_count || 0,
      svc_trades: c.n_trades || 0, svc_pips: c.total_pips || 0,
      oos_ref: c.oos_pd_ref ? c.oos_pd_ref + 'p/d' : '',
    });
  });

  // TR Live configs (account 011 — real orders, 50u/trade)
  const trLiveConfigs = [];
  const trLiveSvc = svc.tr_live || {};
  (trLiveSvc.strategies || []).forEach(c => {
    const pos = c.pos > 0 ? 'LONG' : c.pos < 0 ? 'SHORT' : 'flat';
    const posDetail = c.pos !== 0 && c.entry_px ? `${pos} @${c.entry_px} hw=${c.hw}` : pos;
    trLiveConfigs.push({
      label: c.label, pair: c.pair || '?', type: '🟢LIVE',
      pos: posDetail, pnf: '', bars: c.bar_count || 0,
      svc_trades: c.n_trades || 0, svc_pips: c.total_pips || 0,
      oos_ref: c.oos_pd_ref ? c.oos_pd_ref + 'p/d' : '',
      is_live: true,
    });
  });

  // First-touch low-volume reversion paper (H4, 12 pairs) — one label in DB, so render an
  // aggregate row (maps to DB stats) summarising how many of the 12 pairs are in position.
  const ftConfigs = [];
  const ftStrats = (svc.first_touch || {}).strategies || [];
  if (ftStrats.length) {
    const openLong  = ftStrats.filter(c => c.pos > 0);
    const openShort = ftStrats.filter(c => c.pos < 0);
    const nOpen = openLong.length + openShort.length;
    const posStr = nOpen ? `${nOpen}/12 open (${openLong.length}L ${openShort.length}S: ` +
                   ftStrats.filter(c=>c.pos!==0).map(c=>c.pair.split('_').join('')).join(',') + ')'
                   : 'flat (0/12)';
    ftConfigs.push({
      label: 'first_touch_lv', pair: '12 pairs', type: 'FT-rev',
      pos: posStr, pnf: '', bars: Math.max(...ftStrats.map(c => c.bar_count || 0)),
      svc_trades: ftStrats.reduce((s,c)=>s+(c.n_trades||0),0),
      svc_pips: ftStrats.reduce((s,c)=>s+(c.total_pips||0),0),
      oos_ref: ftStrats[0].oos_pd_ref || '',
    });
  }

  const allConfigs = [...trLiveConfigs, ...liveFifoConfigs, ...zrConfigs, ...fifoConfigs, ...trConfigs, ...ftConfigs];

  if (!allConfigs.length && !Object.keys(stats).length) {
    el.innerHTML = '<div style="color:#555;font-size:12px;padding:8px 0;">No paper services running or no trades yet.</div>';
    return;
  }

  let html = '<table style="width:100%;border-collapse:collapse;font-size:12px;">';
  html += '<tr style="border-bottom:1px solid #333;color:#888;text-align:left;">';
  html += '<th style="padding:5px 8px;">Label</th><th>Type</th><th>Pair</th>';
  html += '<th style="text-align:right;">Bars</th><th>Position / State</th>';
  html += '<th style="text-align:right;">Cycles</th><th style="text-align:right;">Pips (svc)</th>';
  html += '<th style="text-align:right;">DB Pips</th><th style="text-align:right;">OOS ref</th></tr>';

  // Labels from service status
  const renderedLabels = new Set();
  allConfigs.forEach(c => {
    const s = stats[c.label] || {};
    const pnlCls = (c.svc_pips || 0) >= 0 ? 'green' : 'red';
    const dbPnlCls = (s.total_pips || 0) >= 0 ? 'green' : 'red';
    const posFlat = c.pos === 'flat' || c.pos.startsWith('flat');
    const posCol = posFlat ? `<span style="color:#555;">flat</span>` :
                   `<span style="color:#ffc107;">${c.pos}</span>${c.pnf ? ' <span style="color:#555;font-size:11px;">' + c.pnf + '</span>' : ''}`;
    const rowBg = c.is_live ? 'background:#0d1a0d;' : '';
    html += `<tr style="border-bottom:1px solid #1a1a1a;${rowBg}">
      <td style="padding:5px 8px;color:${c.is_live ? '#7fff7f' : '#ccc'};">${c.label}</td>
      <td style="color:${c.is_live ? '#4caf50' : '#888'};">${c.type}</td>
      <td>${c.pair}</td>
      <td style="text-align:right;color:#666;">${c.bars.toLocaleString()}</td>
      <td>${posCol}</td>
      <td style="text-align:right;">${c.svc_trades}</td>
      <td style="text-align:right;" class="${pnlCls}">${c.svc_pips >= 0 ? '+' : ''}${c.svc_pips.toFixed(1)}p</td>
      <td style="text-align:right;" class="${dbPnlCls}">${s.total_pips !== undefined ? (s.total_pips >= 0 ? '+' : '') + s.total_pips + 'p' : '—'}</td>
      <td style="text-align:right;color:#555;">${c.oos_ref || ''}</td>
    </tr>`;
    renderedLabels.add(c.label);
  });

  // DB-only paper labels (no live status file, e.g. retrace_nofilter / retrace_atr /
  // sma_psar / sma_scratch). These are gated to ACTIVE_PAPER_LABEL_PATTERNS server-side,
  // so every label here is a CURRENTLY-RUNNING paper experiment — show it as running,
  // not "stopped" (stopped services are filtered out of the active patterns entirely).
  Object.entries(stats).forEach(([lbl, s]) => {
    if (renderedLabels.has(lbl)) return;
    const pnlCls = s.total_pips >= 0 ? 'green' : 'red';
    html += `<tr style="border-bottom:1px solid #1a1a1a;">
      <td style="padding:5px 8px;color:#ccc;">${lbl}</td>
      <td style="color:#26c6da;">paper</td><td>—</td><td style="text-align:right;">—</td>
      <td><span style="color:#4caf50;">● running</span></td>
      <td style="text-align:right;">${s.trades}</td>
      <td style="text-align:right;" class="${pnlCls}">${s.total_pips >= 0 ? '+' : ''}${s.total_pips}p</td>
      <td style="text-align:right;">${s.win_rate}%</td>
    </tr>`;
  });

  html += '</table>';
  el.innerHTML = html;
}


async function fetchEquity() {
  const sinceVal = document.getElementById('since-input').value;
  let timeParam;
  if (sinceVal) {
    timeParam = `since=${encodeURIComponent(sinceVal)}`;
  } else if (activePresetHours === 0) {
    timeParam = `hours=87600`;
  } else {
    timeParam = `hours=${activePresetHours}`;
  }
  const showLive  = document.getElementById('show-live')?.checked ?? true;
  const showPaper = document.getElementById('show-paper')?.checked ?? true;
  const showFloating = document.getElementById('show-floating')?.checked ?? false;

  try {
    const checked    = getCheckedAccounts();
    const indiv      = checked.filter(c => c !== 'sum');
    const sumChecked = checked.includes('sum');
    const seriesMap  = {};
    let flipsMap = {};   // acct -> [{t, pnl}] of FIFO-flip closes (010 desync bug)
    let eventsMap = {};  // acct -> [{t, kind, label}] of deploys / sizing / bugfix changes

    if (showLive) {
      if (indiv.length > 0) {
        const resp = await fetch(`/api/equity/pips/accounts?accounts=${indiv.join(',')}&${timeParam}`);
        if (!resp.ok) return;   // tolerate nginx HTML during restarts (no JSON-parse spam)
        const data = await resp.json();
        const acctSeriesMap = {};
        Object.entries(data.accounts || {}).forEach(([acct, s]) => {
          if (s && s.length) {
            seriesMap[acct] = { series: s, paper: false };
            acctSeriesMap[acct] = s;
          }
        });
        try {
          const fr = await fetch(`/api/equity/flips?accounts=${indiv.join(',')}`);
          flipsMap = (await fr.json()).flips || {};
        } catch (e) { flipsMap = {}; }
        try {
          const er = await fetch(`/api/equity/events?accounts=${indiv.join(',')}`);
          eventsMap = (await er.json()).events || {};
        } catch (e) { eventsMap = {}; }
        if (sumChecked && Object.keys(acctSeriesMap).length > 0) {
          const sumSeries = computeClientSum(acctSeriesMap);
          if (sumSeries.length) seriesMap['Σ'] = { series: sumSeries, paper: false };
        }
        if (showFloating) {
          // Continuous floating NAV reconstructed from price bars, anchored onto the same
          // running baseline as the realized line so it dips below it during open drawdowns.
          try {
            let anchor = 0;
            Object.values(acctSeriesMap).forEach(s => { if (s && s.length) anchor += s[0].v; });
            const fr = await fetch(`/api/equity/floating?accounts=${indiv.join(',')}&${timeParam}`);
            const fd = await fr.json();
            if (fd.series && fd.series.length) {
              seriesMap['〰 NAV'] = { _floating: true, paper: false,
                series: fd.series.map(p => ({ t: p.t, v: Math.round((p.v + anchor) * 10) / 10 })) };
            }
          } catch (e) {}
        }
      } else {
        const resp = await fetch(`/api/equity/pips/split?${timeParam}`);
        const data = await resp.json();
        if (data.live && data.live.length) seriesMap['🟢 Live'] = { series: data.live, paper: false };
      }
    }

    if (showPaper) {
      const resp = await fetch(`/api/equity/pips/split?${timeParam}`);
      const data = await resp.json();
      const paperLabels = [];
      Object.entries(data.paper || {}).forEach(([lbl, s]) => {
        if (s && s.length) { seriesMap[lbl] = { series: s, paper: true }; paperLabels.push(lbl); }
      });
      // Change-log markers for paper strategies too (deploy/modify), same as live.
      if (paperLabels.length) {
        try {
          const er = await fetch(`/api/equity/events?accounts=${paperLabels.join(',')}`);
          const pe = (await er.json()).events || {};
          Object.assign(eventsMap, pe);   // merge alongside any live account events
        } catch (e) {}
      }
    }

    let strategyStarts = [];
    try {
      const sr = await fetch('/api/strategy_starts');
      const sd = await sr.json();
      strategyStarts = sd.starts || [];
    } catch(e) {}

    renderEquityChart(seriesMap, strategyStarts, flipsMap, eventsMap);
    try {
      if (indiv.length) {
        const mr = await fetch(`/api/equity/mfe_mae?accounts=${indiv.join(',')}&${timeParam}`);
        renderMfeMae((await mr.json()).trades || []);
      } else { renderMfeMae([]); }
    } catch(e) { renderMfeMae([]); }
  } catch(e) { console.error('Equity fetch failed:', e); }
}

const PAPER_PALETTE = ['#ff9800','#e91e63','#9c27b0','#00bcd4','#ff5722','#ffc107','#3f51b5','#673ab7'];

// ONE canonical color per account label (e.g. '009' → PALETTE[8]), used by BOTH
// the checkbox legend AND the chart trace, so they always agree. Deterministic
// from the account number, independent of fetch order / selection.
function acctColor(label) {
  const n = parseInt(label, 10);
  return isNaN(n) ? null : PALETTE[(n - 1 + PALETTE.length) % PALETTE.length];
}
// Stable paper color per label (hash → palette), so a label keeps its color and
// the legend entry matches its trace.
function paperColorFor(label) {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) | 0;
  return PAPER_PALETTE[Math.abs(h) % PAPER_PALETTE.length];
}

function renderEquityChart(seriesMap, strategyStarts = [], flipsMap = {}, eventsMap = {}) {
  // seriesMap: { label: {series: [{t,v},...], paper: bool} }
  const ctx = document.getElementById('equity-chart').getContext('2d');
  if (equityChart) equityChart.destroy();

  const allEntries = Object.entries(seriesMap);
  if (!allEntries.length) return;

  const allTs = [...new Set(allEntries.flatMap(([,{series}]) => series.map(p => p.t)))].sort();
  if (!allTs.length) return;

  // Build start-line annotations:
  //   - If individual accounts are checked, only render markers for those accounts.
  //   - Drop any marker whose start_ts is before the chart's left edge (no clamping to idx=0).
  const START_COLORS = ['#4caf50','#2196f3','#ff9800','#e91e63','#9c27b0','#00bcd4','#ffeb3b','#f44336'];
  const startAnnotations = [];
  const selectedAccts = Array.from(document.querySelectorAll('.acct-chk:checked'))
                           .map(el => el.value).filter(v => v !== 'sum');
  const restrictToSelected = selectedAccts.length > 0;
  const minTs = allTs[0];
  strategyStarts.forEach((s, i) => {
    if (restrictToSelected && !selectedAccts.includes(s.account)) return;
    const startTs = (s.first_trade_ts || '').replace('T', ' ').substring(0, 16);
    if (!startTs) return;
    if (startTs < minTs) return;   // before visible window — skip rather than clamp
    let idx = allTs.findIndex(t => t >= startTs);
    if (idx < 0) return;
    // Use the SAME canonical acctColor as the equity line (NOT a position-based
    // palette) so the start marker + label match their curve. START_COLORS is a
    // fallback only for non-numeric labels.
    startAnnotations.push({ idx, name: s.account,
      color: acctColor(s.account) || START_COLORS[i % START_COLORS.length] });
  });

  const yCallback = v => (v >= 0 ? '+' : '') + v.toFixed(0) + 'p';

  // Colors: Σ = white, '🟢 Live' aggregate = green, per-account = canonical
  // acctColor() (same as the checkbox legend), paper = stable per-label hash.
  let liveIdx = 0;
  const datasets = allEntries.map(([key, {series, paper, _floating}]) => {
    let color, isSumTrace = false;
    if (_floating) {
      color = '#26c6da';                       // floating NAV: cyan, solid continuous line
    } else if (!paper) {
      if (key === 'Σ') {
        color = '#ffffff'; isSumTrace = true;
      } else {
        color = acctColor(key) || PALETTE[liveIdx % PALETTE.length];
        liveIdx++;
      }
    } else {
      color = paperColorFor(key);
    }
    const tsMap = {}; series.forEach(p => tsMap[p.t] = p.v);
    let last = 0;
    const data = allTs.map(t => { if (tsMap[t] !== undefined) last = tsMap[t]; return last; });
    return {
      label: key,
      data,
      borderColor: color,
      backgroundColor: _floating ? '#26c6da14' : (paper || isSumTrace) ? 'transparent' : color + '12',
      fill: _floating ? true : (!paper && !isSumTrace),
      borderWidth: _floating ? 1.4 : isSumTrace ? 2.5 : paper ? 1.5 : 2,
      borderDash: _floating ? [] : paper ? [6, 3] : isSumTrace ? [] : [3, 2],
      pointRadius: 0,
      tension: _floating ? 0 : 0.3,             // floating is raw (already fine-grained)
      _paper: paper,
      _floating: !!_floating,
    };
  });

  // Flip markers: red ✕ on an account's curve where a position was FIFO-netted by an
  // opposing order (the 010 state-desync bug, now guarded). Overlaid as a no-line
  // point dataset; tooltip shows the flip's realised pips.
  Object.entries(flipsMap).forEach(([acct, flips]) => {
    const base = datasets.find(d => d.label === acct && !d._paper);
    if (!base || !flips || !flips.length) return;
    const mdata = allTs.map(() => null);
    const pnls = {};
    flips.forEach(f => {
      let idx = allTs.indexOf(f.t);
      if (idx < 0) idx = allTs.findIndex(t => t >= f.t);
      if (idx >= 0) { mdata[idx] = base.data[idx]; pnls[idx] = f.pnl; }
    });
    datasets.push({
      label: acct + ' flip', data: mdata, showLine: false,
      pointRadius: 6, pointStyle: 'crossRot', pointBorderWidth: 2,
      pointBorderColor: '#f44336', pointBackgroundColor: '#f44336', borderColor: '#f44336',
      _flip: true, _pnls: pnls,
    });
  });

  // Change-log markers: deploys / sizing changes / code bug fixes, snapped to the
  // account's curve at the change time. Per-point colour+shape encode the kind;
  // full description shows in the tooltip. One legend entry per account.
  const EVT_STYLE = {
    deploy:   { color: '#4caf50', style: 'rectRot'  },  // green diamond
    bugfix:   { color: '#ff9800', style: 'triangle' },  // orange triangle
    sizing:   { color: '#2196f3', style: 'rect'     },  // blue square
    strategy: { color: '#9c27b0', style: 'star'     },  // purple star
  };
  Object.entries(eventsMap).forEach(([acct, events]) => {
    // Match the trace by label whether it's live or paper (paper events keyed by label);
    // exclude the marker datasets themselves.
    const base = datasets.find(d => d.label === acct && !d._flip && !d._evt);
    if (!base || !events || !events.length) return;
    const minTs = allTs[0];
    const mdata = allTs.map(() => null);
    const colors = allTs.map(() => '#888');
    const styles = allTs.map(() => 'circle');
    const meta = {};
    events.forEach(e => {
      if (e.t < minTs) return;                 // before visible window — skip
      let idx = allTs.indexOf(e.t);
      if (idx < 0) idx = allTs.findIndex(t => t >= e.t);
      if (idx < 0) return;
      const s = EVT_STYLE[e.kind] || { color: '#888', style: 'circle' };
      mdata[idx] = base.data[idx]; colors[idx] = s.color; styles[idx] = s.style;
      meta[idx] = { label: e.label, kind: e.kind };
    });
    datasets.push({
      label: acct + ' changes', data: mdata, showLine: false,
      pointRadius: 7, pointStyle: styles, pointBorderWidth: 1.5,
      pointBorderColor: '#fff', pointBackgroundColor: colors, borderColor: '#bbb',
      _evt: true, _meta: meta,
    });
  });

  // When the floating NAV is shown, shade the band between it and the realized line:
  // green where floating > realized (open profit), red where below (open drawdown). Make
  // the realized line solid + on top so it reads as the "closes" reference.
  const fIdx = datasets.findIndex(d => d._floating);
  if (fIdx >= 0) {
    const rIdx = datasets.findIndex((d, i) => i !== fIdx && !d._paper && !d._flip && !d._evt &&
                                              d.label !== 'Σ' && d.label !== '🟢 Live');
    if (rIdx >= 0) {
      datasets[rIdx].fill = false;
      datasets[rIdx].borderDash = [];
      datasets[rIdx].borderWidth = 2.2;
      datasets[fIdx].fill = { target: rIdx,
        above: 'rgba(38,198,218,0.18)',   // floating above realized → open profit
        below: 'rgba(244,67,54,0.30)' };  // floating below realized → drawdown
      datasets[fIdx].backgroundColor = 'rgba(244,67,54,0.30)';
    }
  }

  equityChart = new Chart(ctx, {
    type: 'line',
    data: { labels: allTs.map(t => t.substring(5, 16)), datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      _startAnnotations: startAnnotations,
      plugins: {
        tooltip: {
          callbacks: {
            // Title = the timestamp of the hovered point (each body row now names its
            // own trace, so the title no longer needs to single out one dataset).
            title: items => items[0]?.label || '',
            label: item => {
              const name = item.dataset.label || '';
              if (item.dataset._flip) {
                const p = item.dataset._pnls?.[item.dataIndex];
                return `${name}: ` + (p != null ? `flip close ${p >= 0 ? '+' : ''}${p}p` : 'flip close');
              }
              if (item.dataset._evt) {
                const m = item.dataset._meta?.[item.dataIndex];
                return m ? `${name}: [${m.kind}] ${m.label}` : `${name}: change`;
              }
              const v = item.raw;
              const sign = v >= 0 ? '+' : '';
              // Prepend the strategy/account name so each colored row is identifiable.
              return `${name}: ${sign}${v.toFixed(0)}p`;
            },
          },
          titleColor: '#eee',
          bodyColor: '#aaa',
          backgroundColor: '#1a1a1a',
          borderColor: '#333',
          borderWidth: 1,
        },
        legend: {
          display: true,
          labels: { color: '#aaa', font: { size: 10 }, boxWidth: 14, padding: 8,
                    generateLabels: chart => chart.data.datasets.map((ds, i) => ({
                      text: ds._paper ? `◌ ${ds.label}` : ds.label,
                      fillStyle: ds._paper ? 'transparent' : ds.borderColor,
                      strokeStyle: ds.borderColor,
                      lineWidth: 2, hidden: !chart.isDatasetVisible(i), datasetIndex: i,
                      lineDash: ds.borderDash || [],
                    }))
                  }
        }
      },
      scales: {
        x: { display: true, ticks: { color: '#555', maxTicksLimit: 10, font: { size: 10 } }, grid: { color: '#1a1a1a' } },
        y: { display: true, ticks: { color: '#555', font: { size: 10 }, callback: yCallback }, grid: { color: '#222' } }
      }
    }
  });
}

// MAE/MFE scatter for the trades visible in the equity chart's window. x=MAE (drawdown the
// trade took), y=MFE (run-up it reached). Color = win/loss; shape = long/short. Points above
// the dashed MFE=MAE diagonal ran further in profit than they drew down.
function renderMfeMae(trades) {
  const el = document.getElementById('mfemae-chart');
  if (!el) return;
  const ctx = el.getContext('2d');
  if (mfemaeChart) { mfemaeChart.destroy(); mfemaeChart = null; }
  if (!trades.length) {
    mfemaeChart = new Chart(ctx, { type: 'scatter', data: { datasets: [] },
      options: { responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false },
          title: { display: true, text: 'no excursion data in window', color: '#555', font: { size: 11 } } },
        scales: { x: { ticks: { display: false }, grid: { color: '#1a1a1a' } },
                  y: { ticks: { display: false }, grid: { color: '#222' } } } } });
    return;
  }
  // Bound axes to ~the 90th percentile of per-trade max(MAE,MFE) so 90%+ of trades fill the
  // plot; clamp the rarer extremes to the edge (marked white-ringed, real value in tooltip).
  const pctl = (arr, p) => { if (!arr.length) return 0; const s = [...arr].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
  const perMax = trades.map(t => Math.max(t.mae, t.mfe));
  const maxv = Math.max(10, pctl(perMax, 0.90) * 1.12);
  let nOut = 0;
  const pts = trades.map(t => {
    const out = t.mae > maxv || t.mfe > maxv; if (out) nOut++;
    return { x: Math.min(t.mae, maxv), y: Math.min(t.mfe, maxv),
             _pnl: t.pnl, _dir: t.dir, _pair: t.pair, _mae: t.mae, _mfe: t.mfe, _out: out };
  });
  const colors  = trades.map(t => t.pnl >= 0 ? 'rgba(76,175,80,0.85)' : 'rgba(244,67,54,0.85)');
  const styles  = trades.map(t => t.dir > 0 ? 'triangle' : 'rectRot');   // long ▲ / short ◆
  const borders = pts.map((p, i) => p._out ? '#fff' : colors[i]);
  const bwidth  = pts.map(p => p._out ? 2 : 0);
  mfemaeChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets: [
      { data: pts, pointBackgroundColor: colors, pointBorderColor: borders,
        pointStyle: styles, pointRadius: 5, pointBorderWidth: bwidth },
      { type: 'line', data: [{ x: 0, y: 0 }, { x: maxv, y: maxv }], borderColor: '#555',
        borderDash: [4, 4], borderWidth: 1, pointRadius: 0, fill: false },
    ] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: { display: nOut > 0, text: `${nOut} outlier${nOut > 1 ? 's' : ''} clamped to edge ⌐`,
                 color: '#888', font: { size: 9 }, padding: 2 },
        tooltip: { callbacks: { label: i => {
          const t = i.raw;
          return `${t._pair} ${t._dir > 0 ? 'L' : 'S'}  MAE ${t._mae}p  MFE ${t._mfe}p  P/L ${t._pnl >= 0 ? '+' : ''}${t._pnl}p${t._out ? '  ⚠ outlier' : ''}`;
        } } },
      },
      scales: {
        x: { min: 0, max: maxv, title: { display: true, text: 'MAE — drawdown (pips)', color: '#777', font: { size: 9 } },
             ticks: { color: '#666', font: { size: 9 }, maxTicksLimit: 5 }, grid: { color: '#1a1a1a' } },
        y: { min: 0, max: maxv, title: { display: true, text: 'MFE — run-up (pips)', color: '#777', font: { size: 9 } },
             ticks: { color: '#666', font: { size: 9 }, maxTicksLimit: 5 }, grid: { color: '#222' } },
      },
    },
  });
}

function renderSummary(accounts) {
  const totalNav = accounts.reduce((s,a) => s + a.nav, 0);
  const totalUpl = accounts.reduce((s,a) => s + a.unrealized_pl, 0);
  const totalTrades = accounts.reduce((s,a) => s + a.total_trades, 0);
  const totalPnl = accounts.reduce((s,a) => s + a.total_pnl, 0);
  const totalOpen = accounts.reduce((s,a) => s + a.open_positions, 0);

  document.getElementById('summary-bar').innerHTML = `
    <div class="summary-item"><div class="label">Total NAV</div><div class="value white">$${totalNav.toFixed(2)}</div></div>
    <div class="summary-item"><div class="label">Unrealized</div><div class="value ${totalUpl >= 0 ? 'green' : 'red'}">$${totalUpl.toFixed(2)}</div></div>
    <div class="summary-item"><div class="label">Total P/L</div><div class="value ${totalPnl >= 0 ? 'green' : 'red'}">${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(1)}p</div></div>
    <div class="summary-item"><div class="label">Trades</div><div class="value white">${totalTrades}</div></div>
    <div class="summary-item"><div class="label">Open</div><div class="value white">${totalOpen}</div></div>
    <div class="summary-item"><div class="label">Accounts</div><div class="value white">${accounts.length}</div></div>
  `;
}

function renderGrid(accounts) {
  let html = '<table style="width:100%;border-collapse:collapse;font-size:13px;margin:10px 20px;">';
  html += '<tr style="border-bottom:1px solid #444;color:#888;text-align:left;">';
  html += '<th style="padding:6px;">Acct</th><th>Strategy</th><th>Status</th><th style="text-align:right;">NAV</th><th style="text-align:right;">uPL</th><th style="text-align:right;">Margin</th><th style="text-align:right;">Open</th><th style="text-align:right;">Trades</th><th style="text-align:right;">P/L</th><th style="text-align:right;">WR</th></tr>';
  accounts.forEach(a => {
    const pnlClass = a.total_pnl >= 0 ? 'green' : 'red';
    const uplClass = a.unrealized_pl >= 0 ? 'green' : 'red';
    const status = a.active !== false ? '<span style="color:#4caf50;">●</span> Live' : '<span style="color:#666;">○</span> Off';
    const rowStyle = a.active === false ? 'opacity:0.5;' : '';
    const clickable = `onclick="selectAccount('${a.label}')" style="cursor:pointer;${rowStyle}"`;
    html += `<tr ${clickable}>
      <td style="padding:6px;font-weight:bold;">${a.label}</td>
      <td>${a.name}</td>
      <td>${status}</td>
      <td style="text-align:right;">$${a.nav.toFixed(2)}</td>
      <td style="text-align:right;" class="${uplClass}">${a.unrealized_pl ? '$' + a.unrealized_pl.toFixed(4) : '-'}</td>
      <td style="text-align:right;">${a.margin_pct ? a.margin_pct.toFixed(0) + '%' : '-'}</td>
      <td style="text-align:right;">${a.open_positions || 0}</td>
      <td style="text-align:right;">${a.total_trades || 0}</td>
      <td style="text-align:right;" class="${pnlClass}">${a.total_pnl ? (a.total_pnl >= 0 ? '+' : '') + a.total_pnl.toFixed(1) + 'p' : '-'}</td>
      <td style="text-align:right;">${a.win_rate ? a.win_rate + '%' : '-'}</td>
    </tr>`;
  });
  html += '</table>';
  document.getElementById('account-grid').innerHTML = html;
}

async function selectAccount(label, silent) {
  if (selectedAcct === label && !silent) { selectedAcct = null; document.getElementById('detail-container').innerHTML = ''; return; }
  selectedAcct = label;
  try {
    const resp = await fetch(`/api/account/${label}/trades?limit=200`);
    const data = await resp.json();
    renderDetail(data);
  } catch(e) { console.error('Fetch detail failed:', e); }
}

function renderDetail(data) {
  const container = document.getElementById('detail-container');

  // Per-pair stats table
  let pairHtml = '<table><tr><th>Pair</th><th>Trades</th><th>P/L (pips)</th><th>WR</th></tr>';
  (data.pair_stats || []).forEach(ps => {
    const cls = ps.pnl >= 0 ? 'green' : 'red';
    pairHtml += `<tr><td>${ps.pair}</td><td>${ps.trades}</td><td class="${cls}">${ps.pnl >= 0 ? '+' : ''}${ps.pnl.toFixed(1)}</td><td>${ps.win_rate}%</td></tr>`;
  });
  pairHtml += '</table>';

  // Trade history table
  let tradeHtml = '<div class="trade-table-wrap"><table><tr><th>Time</th><th>Pair</th><th>Dir</th><th>Units</th><th>Entry</th><th>Exit</th><th>P/L</th><th>MFE</th><th>MAE</th><th>Cap</th><th>Reason</th></tr>';
  (data.trades || []).forEach(t => {
    const dir = t.direction > 0 ? 'LONG' : 'SHORT';
    const dirCls = t.direction > 0 ? 'dir-long' : 'dir-short';
    const pnlCls = t.pnl_pips >= 0 ? 'green' : 'red';
    const time = t.exit_time ? t.exit_time.substring(5, 16) : '';
    tradeHtml += `<tr>
      <td>${time}</td><td>${t.pair}</td><td class="${dirCls}">${dir}</td><td>${t.units||''}</td>
      <td>${t.entry_price ? t.entry_price.toFixed(5) : ''}</td>
      <td>${t.exit_price ? t.exit_price.toFixed(5) : ''}</td>
      <td class="${pnlCls}">${t.pnl_pips >= 0 ? '+' : ''}${t.pnl_pips.toFixed(1)}</td>
      <td>${t.mfe_pips.toFixed(1)}</td><td>${t.mae_pips.toFixed(1)}</td>
      <td>${t.capture_ratio.toFixed(2)}</td><td>${t.exit_reason || ''}</td>
    </tr>`;
  });
  tradeHtml += '</table></div>';

  container.innerHTML = `
    <div class="detail-panel active">
      <span class="detail-close" onclick="selectedAcct=null;this.parentElement.remove();">&times;</span>
      <h2>${data.name} (${data.account}) — ${data.strategy}</h2>
      <div class="chart-row">
        <div class="chart-box"><h3>MFE vs MAE (pips)</h3><canvas id="mfe-chart"></canvas></div>
        <div class="chart-box"><h3>P/L Distribution</h3><canvas id="pnl-chart"></canvas></div>
      </div>
      <div class="tabs">
        <div class="tab ${activeTab==='pairs'?'active':''}" onclick="switchTab(this, 'pairs')">Per-Pair</div>
        <div class="tab ${activeTab==='trades'?'active':''}" onclick="switchTab(this, 'trades')">Trade History</div>
      </div>
      <div class="tab-content ${activeTab==='pairs'?'active':''}" id="tab-pairs">${pairHtml}</div>
      <div class="tab-content ${activeTab==='trades'?'active':''}" id="tab-trades">${tradeHtml}</div>
    </div>
  `;

  renderMfeChart(data.trades || []);
  renderPnlChart(data.trades || []);
}

function renderMfeChart(trades) {
  const ctx = document.getElementById('mfe-chart');
  if (!ctx) return;
  if (mfeChart) mfeChart.destroy();
  if (!trades.length) return;

  const winners = trades.filter(t => t.pnl_pips >= 0);
  const losers = trades.filter(t => t.pnl_pips < 0);

  mfeChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'Winners', data: winners.map(t => ({x: t.mfe_pips, y: t.mae_pips})), backgroundColor: '#4caf5080', borderColor: '#4caf50', pointRadius: 4 },
        { label: 'Losers', data: losers.map(t => ({x: t.mfe_pips, y: t.mae_pips})), backgroundColor: '#f4433680', borderColor: '#f44336', pointRadius: 4 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#888', font: { size: 10 } } } },
      scales: {
        x: { title: { display: true, text: 'MFE (pips)', color: '#666', font: { size: 10 } }, ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1a1a1a' } },
        y: { title: { display: true, text: 'MAE (pips)', color: '#666', font: { size: 10 } }, ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1a1a1a' } }
      }
    }
  });
}

function renderPnlChart(trades) {
  const ctx = document.getElementById('pnl-chart');
  if (!ctx) return;
  if (pnlChart) pnlChart.destroy();
  if (!trades.length) return;

  // Cumulative P/L
  const sorted = [...trades].reverse();
  let cum = 0;
  const cumData = sorted.map((t, i) => { cum += t.pnl_pips; return cum; });
  const labels = sorted.map((t, i) => i + 1);
  const borderColor = cum >= 0 ? '#4caf50' : '#f44336';

  pnlChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Cumulative P/L', data: cumData,
        borderColor, backgroundColor: borderColor + '20',
        fill: true, borderWidth: 1.5, pointRadius: 0, tension: 0.2,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Trade #', color: '#666', font: { size: 10 } }, ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1a1a1a' } },
        y: { title: { display: true, text: 'Pips', color: '#666', font: { size: 10 } }, ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1a1a1a' } }
      }
    }
  });
}

function switchTab(el, id) {
  activeTab = id;  // Remember for refreshes
  el.parentElement.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  el.closest('.detail-panel').querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
}

// Initial load
fetchAccounts();
fetchToday();
fetchEquity();
fetchPaper();
fetchSignals();
setInterval(fetchAccounts, REFRESH_MS);
setInterval(fetchEquity, 60000);  // Equity chart every 60s
setInterval(fetchPaper, 30000);   // Paper section every 30s
setInterval(fetchSignals, 30000); // Signals every 30s
setInterval(fetchXbreak, 30000);  // X-Break tab every 30s
setInterval(fetchToday, 30000);   // Today panel every 30s (backend caches OANDA call 60s)

async function fetchSignals() {
  try {
    const resp = await fetch('/api/signals');
    if (!resp.ok) { document.getElementById('signals-section').innerHTML = '<span style="color:#555;">fx-signals not running</span>'; return; }
    const data = await resp.json();
    if (data.error) { document.getElementById('signals-section').innerHTML = `<span style="color:#555;">${data.error}</span>`; return; }
    renderSignals(data);
    initChartPairSelect();
    loadSignalChart();
  } catch(e) { console.error('Signals fetch failed:', e); }
}

function signalBg(v, maxV) {
  if (v === null || v === undefined) return '#111';
  const t = Math.max(-1, Math.min(1, v / maxV));
  if (t > 0) { const g = Math.round(30 + t * 120); return `rgb(10,${g},10)`; }
  else { const r = Math.round(30 + (-t) * 120); return `rgb(${r},10,10)`; }
}

function signalFg(v) {
  if (v === null || v === undefined) return '#444';
  return v >= 0 ? '#6fdc6f' : '#e05555';
}

function fmtV(v) {
  if (v === null || v === undefined) return '   —  ';
  const s = v >= 0 ? '+' : '';
  return `${s}${v.toFixed(3)}`;
}

async function fetchXbreak() {
  try {
    const r = await fetch('/api/signals/xbreak');
    const data = await r.json();
    if (data.error) {
      document.getElementById('xbreak-table').innerHTML =
        '<div style="color:#a55;">' + data.error + '</div>';
      return;
    }
    renderXbreak(data);
  } catch(e) { console.error('xbreak fetch failed:', e); }
}

function _fmtPip(v) {
  if (v === null || v === undefined) return '<span style="color:#444;">—</span>';
  const s = (v >= 0 ? '+' : '') + v.toFixed(1);
  const color = v > 0 ? '#5a5' : (v < 0 ? '#c55' : '#888');
  return '<span style="color:' + color + ';">' + s + '</span>';
}
function _fmtBool(v, color_true='#5a5', color_false='#666') {
  if (v === null || v === undefined) return '<span style="color:#444;">—</span>';
  return v ? '<span style="color:' + color_true + ';">✓</span>'
           : '<span style="color:' + color_false + ';">·</span>';
}
function _fmtArmed(long_armed, short_armed) {
  if (long_armed) return '<span style="background:#1a3a1a;color:#7d7;padding:1px 5px;border-radius:3px;">▲ LONG</span>';
  if (short_armed) return '<span style="background:#3a1a1a;color:#d77;padding:1px 5px;border-radius:3px;">▼ SHORT</span>';
  return '<span style="color:#444;">—</span>';
}

function renderXbreak(data) {
  const _xts = document.getElementById('xbreak-ts');
  if (!_xts) return;   // X-Break tab removed — backend dormant, DOM gone
  _xts.textContent =
    'updated ' + (data.ts ? new Date(data.ts).toLocaleTimeString() : '—');
  const xb = data.xbreak || {};
  const pairs = Object.keys(xb).sort();
  if (!pairs.length) {
    document.getElementById('xbreak-table').innerHTML =
      '<div style="color:#888;">no xbreak data yet (warming up?)</div>';
    return;
  }
  let html = '<table style="border-collapse:collapse;width:100%;">';
  html += '<thead><tr style="border-bottom:1px solid #333;color:#888;text-align:right;">';
  const cols = [
    ['pair', 'left', 'pair'],
    ['c', 'right', 'close'],
    ['sma7', 'right', 'sma7'],
    ['xover_4ago_p', 'right', 'c−sma7 [t−240] (long: ▼)'],
    ['xover_3ago_p', 'right', 'c−sma7 [t−180] (long: ▲)'],
    ['small_move_p', 'right', 'small Δ [t−120]−[t−180]'],
    ['large_move_p', 'right', 'large Δ [t−60]−[t−120]'],
    ['accel', 'center', '|L|>|S|'],
    ['current_mv_p', 'right', 'cur Δ [t]−[t−60]'],
    ['gap_shrink_1', 'center', 'gap↓₁'],
    ['gap_shrink_2', 'center', 'gap↓₂'],
    ['armed', 'center', 'signal'],
  ];
  cols.forEach(c => { html += '<th style="padding:4px 6px;text-align:' + c[1] + ';font-weight:normal;">' + c[2] + '</th>'; });
  html += '</tr></thead><tbody>';
  pairs.forEach(p => {
    const x = xb[p] || {};
    const armed = (x.long_armed || x.short_armed);
    const rowBg = armed ? 'background:#1c1c0a;' : '';
    html += '<tr style="border-bottom:1px solid #1f1f1f;' + rowBg + '">';
    html += '<td style="padding:3px 6px;color:#bbb;">' + p + '</td>';
    html += '<td style="padding:3px 6px;text-align:right;color:#bbb;">' + (x.c !== null && x.c !== undefined ? x.c.toFixed(p.includes('JPY') ? 3 : 5) : '—') + '</td>';
    html += '<td style="padding:3px 6px;text-align:right;color:#888;">' + (x.sma7 !== null && x.sma7 !== undefined ? x.sma7.toFixed(p.includes('JPY') ? 3 : 5) : '—') + '</td>';
    html += '<td style="padding:3px 6px;text-align:right;">' + _fmtPip(x.xover_4ago_p) + '</td>';
    html += '<td style="padding:3px 6px;text-align:right;">' + _fmtPip(x.xover_3ago_p) + '</td>';
    html += '<td style="padding:3px 6px;text-align:right;">' + _fmtPip(x.small_move_p) + '</td>';
    html += '<td style="padding:3px 6px;text-align:right;">' + _fmtPip(x.large_move_p) + '</td>';
    html += '<td style="padding:3px 6px;text-align:center;">' + _fmtBool(x.accel) + '</td>';
    html += '<td style="padding:3px 6px;text-align:right;">' + _fmtPip(x.current_mv_p) + '</td>';
    html += '<td style="padding:3px 6px;text-align:center;">' + _fmtBool(x.gap_shrink_1, '#d77', '#666') + '</td>';
    html += '<td style="padding:3px 6px;text-align:center;">' + _fmtBool(x.gap_shrink_2, '#d77', '#666') + '</td>';
    html += '<td style="padding:3px 6px;text-align:center;">' + _fmtArmed(x.long_armed, x.short_armed) + '</td>';
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('xbreak-table').innerHTML = html;
}

function renderSignals(data) {
  const { windows, pairs, csi, ts, accel, weighted_sum, csi_accel, csi_weighted_sum } = data;
  const tsEl = document.getElementById('signals-ts');
  if (ts) tsEl.textContent = `pips/min · ${ts.substring(11,19)} UTC`;

  // Per-column max for color scale (S5/M1 are noisier than M5 windows)
  const colMax = {};
  windows.forEach(w => colMax[w] = 0.05);
  Object.values(pairs || {}).forEach(row => {
    windows.forEach(w => {
      const v = row[w];
      if (v !== null && v !== undefined) colMax[w] = Math.max(colMax[w], Math.abs(v));
    });
  });

  const PAIRS_ORDER = [
    'GBP_JPY','USD_JPY','EUR_JPY','EUR_USD','GBP_USD',
    'AUD_JPY','CAD_JPY','CHF_JPY','AUD_USD','NZD_JPY','NZD_USD','EUR_GBP'
  ];
  const CSI_ORDER = ['GBP','JPY','USD','EUR','AUD','CAD','CHF','NZD'];

  // Max for weighted_sum color scale
  let wsMax = 0.05;
  Object.values(weighted_sum || {}).forEach(v => { if (v !== null && v !== undefined) wsMax = Math.max(wsMax, Math.abs(v)); });

  const colW = '68px';
  let html = `<table style="border-collapse:collapse;width:auto;">`;
  // Header
  html += `<tr><th style="text-align:left;color:#555;padding:2px 8px 2px 2px;min-width:80px;">Pair</th>`;
  windows.forEach(w => {
    html += `<th style="color:#555;padding:2px 6px;text-align:right;min-width:${colW};">${w}</th>`;
  });
  html += `<th style="color:#4fc3f7;padding:2px 6px;text-align:right;min-width:${colW};border-left:1px solid #2a2a2a;">WSum</th>`;
  html += '</tr>';

  // Pair rows
  PAIRS_ORDER.forEach(pair => {
    const row = (pairs || {})[pair];
    if (!row) return;
    const pairAccel = (accel || {})[pair] || {};
    const ws = weighted_sum ? weighted_sum[pair] : undefined;
    html += `<tr><td style="color:#888;padding:2px 8px 2px 2px;">${pair.replace('_',' ')}</td>`;
    windows.forEach(w => {
      const v = row[w];
      const a = pairAccel[w];
      const bg = signalBg(v, colMax[w]);
      const fg = signalFg(v);
      html += `<td style="background:${bg};color:${fg};padding:2px 6px;text-align:right;font-variant-numeric:tabular-nums;">${fmtV(v)}${accelArrow(a)}</td>`;
    });
    // Weighted sum column
    const wsBg = signalBg(ws, wsMax);
    const wsFg = signalFg(ws);
    html += `<td style="background:${wsBg};color:${wsFg};padding:2px 6px;text-align:right;font-variant-numeric:tabular-nums;border-left:1px solid #2a2a2a;">${fmtV(ws)}</td>`;
    html += '</tr>';
  });

  // Divider + CSI rows
  const csiColMax = {};
  windows.forEach(w => csiColMax[w] = 0.05);
  Object.values(csi || {}).forEach(row => {
    windows.forEach(w => {
      const v = row[w];
      if (v !== null && v !== undefined) csiColMax[w] = Math.max(csiColMax[w], Math.abs(v));
    });
  });
  let csiWsMax = 0.05;
  Object.values(csi_weighted_sum || {}).forEach(v => { if (v !== null && v !== undefined) csiWsMax = Math.max(csiWsMax, Math.abs(v)); });

  html += `<tr><td colspan="${windows.length+2}" style="height:6px;"></td></tr>`;
  html += `<tr><td style="color:#555;padding:2px 8px 2px 2px;font-size:10px;">CSI</td>`;
  windows.forEach(w => html += `<th style="color:#555;padding:2px 6px;text-align:right;font-size:10px;">${w}</th>`);
  html += `<th style="color:#4fc3f7;padding:2px 6px;text-align:right;font-size:10px;border-left:1px solid #2a2a2a;">WSum</th>`;
  html += '</tr>';

  CSI_ORDER.forEach(cur => {
    const row = (csi || {})[cur];
    if (!row) return;
    const curAccel = (csi_accel || {})[cur] || {};
    const csiWs = csi_weighted_sum ? csi_weighted_sum[cur] : undefined;
    html += `<tr><td style="color:#888;padding:2px 8px 2px 2px;font-size:10px;">${cur}</td>`;
    windows.forEach(w => {
      const v = row[w];
      const a = curAccel[w];
      const bg = signalBg(v, csiColMax[w]);
      const fg = signalFg(v);
      html += `<td style="background:${bg};color:${fg};padding:2px 6px;text-align:right;font-size:10px;font-variant-numeric:tabular-nums;">${fmtV(v)}${accelArrow(a)}</td>`;
    });
    // CSI weighted sum column
    const cwsBg = signalBg(csiWs, csiWsMax);
    const cwsFg = signalFg(csiWs);
    html += `<td style="background:${cwsBg};color:${cwsFg};padding:2px 6px;text-align:right;font-size:10px;font-variant-numeric:tabular-nums;border-left:1px solid #2a2a2a;">${fmtV(csiWs)}</td>`;
    html += '</tr>';
  });

  html += '</table>';
  document.getElementById('signals-section').innerHTML = html;
}
</script>
</body>
</html>"""


@app.after_request
def add_no_cache_headers(response):
    """Prevent browser caching of API and HTML responses."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def main():
    port = int(os.environ.get("DASHBOARD_PORT", "5558"))
    logger.info(f"FX-Core Dashboard starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
