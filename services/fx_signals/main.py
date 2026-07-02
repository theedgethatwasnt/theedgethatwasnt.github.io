"""
fx-signals: Rolling-window momentum metrics for 12 FX pairs.

Granularities tracked: S5 (5-sec), M1 (1-min), M5 (5-min).
On startup: fetch warm-up history from OANDA per granularity per pair.
Every POLL_INTERVAL seconds: fetch latest bars, append new completed ones.
Compute pips/min for each granularity:
  S5  → 1-bar  window:  5-sec rate  → pips/min
  M1  → 1-bar  window:  1-min rate  → pips/min
  M5  → [1,3,12,48,288]-bar windows → 5m/15m/1h/4h/24h pips/min
Compute CSI (currency strength index) across all windows.
Write atomic JSON to SIGNALS_STATE_DIR/signals_state.json.
Dashboard reads via /api/signals — no calculation in dashboard.
"""

import os, json, time, logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import v20

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from lib import csi as csilib  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fx_signals")

PAIRS = [
    "GBP_JPY", "USD_JPY", "EUR_JPY", "EUR_USD", "GBP_USD",
    "AUD_JPY", "CAD_JPY", "CHF_JPY", "AUD_USD", "NZD_JPY",
    "NZD_USD", "EUR_GBP",
]

# sign: +1 = currency is base (pair goes up → currency strong)
#        -1 = currency is quote
CURRENCY_PAIRS: dict[str, list[tuple[str, int]]] = {
    "GBP": [("GBP_JPY", +1), ("GBP_USD", +1), ("EUR_GBP", -1)],
    "JPY": [("GBP_JPY", -1), ("USD_JPY", -1), ("EUR_JPY", -1),
             ("AUD_JPY", -1), ("CAD_JPY", -1), ("CHF_JPY", -1), ("NZD_JPY", -1)],
    "USD": [("USD_JPY", +1), ("EUR_USD", -1), ("GBP_USD", -1),
             ("AUD_USD", -1), ("NZD_USD", -1)],
    "EUR": [("EUR_JPY", +1), ("EUR_USD", +1), ("EUR_GBP", +1)],
    "AUD": [("AUD_JPY", +1), ("AUD_USD", +1)],
    "CAD": [("CAD_JPY", +1)],
    "CHF": [("CHF_JPY", +1)],
    "NZD": [("NZD_JPY", +1), ("NZD_USD", +1)],
}

# Per-granularity config:
#   max_bars    — deque size
#   warmup      — bars to fetch on startup
#   poll_count  — bars to fetch each poll
#   min_per_bar — minutes per bar (for pips/min calculation)
#   windows     — list of (n_bars, label) for pips/min computation
GRAN_CONFIG = {
    "S5": {
        "max_bars":    700,       # >=601 needed for xbreak sma7_lag240 oldest sample
        "warmup":      700,
        "poll_count":  8,         # 30s poll / 5s per bar = 6 new bars max
        "min_per_bar": 5.0 / 60,  # 5 seconds = 1/12 minute
        "windows":     [(1, "S5")],
    },
    "M1": {
        "max_bars":    20,
        "warmup":      10,
        "poll_count":  3,
        "min_per_bar": 1.0,
        "windows":     [(1, "M1")],
    },
    "M5": {
        "max_bars":    295,
        "warmup":      295,
        "poll_count":  2,
        "min_per_bar": 5.0,
        "windows":     [(1, "5m"), (3, "15m"), (12, "1h"), (48, "4h"), (288, "24h")],
    },
}

# Column order for display (and CSI computation)
ALL_WINDOW_LABELS = ["S5", "M1", "5m", "15m", "1h", "4h", "24h"]

OANDA_KEY = os.environ.get("OANDA_API_KEY", "")
ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID") or os.environ.get("OANDA_ACCOUNT_ID_001", "")
STATE_DIR = Path(os.environ.get("SIGNALS_STATE_DIR", "/data/logs"))
STATE_FILE = STATE_DIR / "signals_state.json"
HISTORY_DB = STATE_DIR / "signals_history.duckdb"
POLL_INTERVAL = 30
# Bound the append-only history DB: keep N days of rows, prune hourly.
HISTORY_RETENTION_DAYS = int(os.environ.get("SIGNALS_HISTORY_RETENTION_DAYS", "14"))
HISTORY_PRUNE_EVERY_SEC = 3600.0
_last_hist_prune: float = 0.0

# Candle-based Wilder CSI (per-pair tradeability) — slow, refreshed every 5 min.
# OANDA granularity → Markets timeframe label.
CSI_GRANS = [("M5", "5m"), ("M15", "15m"), ("H1", "1h"), ("H4", "4h"), ("D", "24h")]
CSI_REFRESH_SEC = 300
ECON: dict = {}        # {pair: {"V":.., "M":..}}
SPREAD_PIPS: dict = {} # {pair: spread_pips at startup snapshot}
_csi_cache: dict = {}
_csi_last: float = 0.0

WINDOW_WEIGHTS = {
    "S5": 0.05, "M1": 0.05, "5m": 0.10,
    "15m": 0.15, "1h": 0.20, "4h": 0.20, "24h": 0.25,
}

# Module-level previous-cycle state for acceleration computation
prev_pair_signals: dict = {}
prev_csi: dict = {}

# Buffers: gran -> pair -> deque[{t, c}]
bufs: dict[str, dict[str, deque]] = {
    gran: {pair: deque(maxlen=cfg["max_bars"]) for pair in PAIRS}
    for gran, cfg in GRAN_CONFIG.items()
}


def _pip(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def _ctx() -> v20.Context:
    return v20.Context("api-fxtrade.oanda.com", 443, token=OANDA_KEY)


def fetch_bars(pair: str, granularity: str, count: int) -> list[dict]:
    """Fetch the last `count` completed mid candles from OANDA."""
    ctx = _ctx()
    resp = ctx.instrument.candles(
        pair, granularity=granularity, count=count + 1, price="M"
    )
    bars = []
    for c in resp.body["candles"]:
        if not c.complete:
            continue
        bars.append({"t": str(c.time), "c": float(c.mid.c)})
    return bars[-count:]


def fetch_econ() -> None:
    """Load V/M + startup spread per instrument (for the Wilder CSI factor)."""
    try:
        ctx = _ctx()
        insts = ctx.account.instruments(ACCOUNT_ID).get("instruments")
        names = [i.name for i in insts]
        pr = ctx.pricing.get(ACCOUNT_ID, instruments=",".join(names),
                             includeUnitsAvailable=False)
        prices = {}
        for p in pr.get("prices"):
            try:
                bid = float(p.closeoutBid); ask = float(p.closeoutAsk)
                prices[p.instrument] = bid
                pl = -2 if "JPY" in p.instrument else -4
                SPREAD_PIPS[p.instrument] = (ask - bid) / (10 ** pl)
            except Exception:
                pass
        ECON.update(csilib.compute_econ(insts, prices))
        logger.info(f"econ loaded: {len(ECON)} instruments · candle CSI for "
                    f"{sum(1 for p in PAIRS if p in ECON)}/{len(PAIRS)} pairs")
    except Exception as exc:
        logger.warning(f"econ fetch failed — candle CSI disabled: {exc}")


def fetch_ohlc(pair: str, gran: str, count: int):
    """Fetch the last `count` completed mid OHLC candles → (o, h, l, c) lists."""
    ctx = _ctx()
    resp = ctx.instrument.candles(pair, granularity=gran, count=count + 1, price="M")
    o, h, l, c = [], [], [], []
    for cd in resp.body["candles"]:
        if not cd.complete:
            continue
        o.append(float(cd.mid.o)); h.append(float(cd.mid.h))
        l.append(float(cd.mid.l)); c.append(float(cd.mid.c))
    return o[-count:], h[-count:], l[-count:], c[-count:]


def compute_csi_pairs() -> dict:
    """Wilder CSI per pair per timeframe → {label: [{pair, csi}] sorted desc}."""
    out = {}
    for gran, label in CSI_GRANS:
        rank = []
        for pair in PAIRS:
            e = ECON.get(pair)
            if not e:
                continue
            try:
                o, h, l, c = fetch_ohlc(pair, gran, 120)
                if len(c) < 40:
                    continue
                pip_loc = -2 if "JPY" in pair else -4
                C = SPREAD_PIPS.get(pair, 1.5) * e["V"]
                val = csilib.wilder_csi_from_ohlc(o, h, l, c, pip_loc, e["V"], e["M"], C)
                if val == val:   # not nan
                    rank.append((pair, round(val, 1)))
            except Exception:
                pass
        rank.sort(key=lambda x: -x[1])
        out[label] = [{"pair": p, "csi": v} for p, v in rank]
    return out


def warmup() -> None:
    logger.info("Warming up from OANDA history…")
    for gran, cfg in GRAN_CONFIG.items():
        logger.info(f"  {gran}: fetching {cfg['warmup']} bars per pair")
        for pair in PAIRS:
            for attempt in range(3):
                try:
                    bars = fetch_bars(pair, gran, cfg["warmup"])
                    bufs[gran][pair].extend(bars)
                    break
                except Exception as exc:
                    logger.warning(f"    {pair}/{gran} attempt {attempt+1}: {exc}")
                    time.sleep(5 * (attempt + 1))
        logger.info(f"  {gran}: done")


def update_latest() -> None:
    for gran, cfg in GRAN_CONFIG.items():
        for pair in PAIRS:
            try:
                bars = fetch_bars(pair, gran, cfg["poll_count"])
                if not bars:
                    continue
                buf = bufs[gran][pair]
                for bar in bars:
                    if not buf or buf[-1]["t"] < bar["t"]:
                        buf.append(bar)
            except Exception as exc:
                logger.warning(f"Update {pair}/{gran}: {exc}")


def compute_pair_signals() -> tuple[dict, dict]:
    """Return (momentum, momentum_norm) where:
      momentum[pair][lbl]      = pips/min over the window (raw)
      momentum_norm[pair][lbl] = (pips/min) / σ_per_min where σ is the
                                  per-bar standard deviation of pip-changes
                                  over the same window (Sharpe-like).
    The σ-scaled version is dimensionless and answers "is the trend large
    relative to its own noise?" — values > 2 mark unusual momentum, near 0
    marks noisy chop. Useful as a shock-detection / wavelet feature.
    """
    import statistics as _stats
    raw: dict[str, dict] = {}
    norm: dict[str, dict] = {}
    for pair in PAIRS:
        entry_raw: dict[str, float | None] = {}
        entry_norm: dict[str, float | None] = {}
        pip = _pip(pair)
        for gran, cfg in GRAN_CONFIG.items():
            buf = list(bufs[gran][pair])
            closes = [b["c"] for b in buf]
            mpb = cfg["min_per_bar"]
            for w, lbl in cfg["windows"]:
                if len(closes) < w + 1:
                    entry_raw[lbl] = None
                    entry_norm[lbl] = None
                    continue
                pips_moved = (closes[-1] - closes[-(w + 1)]) / pip
                minutes = w * mpb
                mom = pips_moved / minutes
                entry_raw[lbl] = round(mom, 4)
                # σ-scaled: stddev of per-bar pip changes over the window.
                # Window of 1 has no internal bar-deltas → leave norm None.
                if w >= 2:
                    bar_deltas = [
                        (closes[-i] - closes[-i-1]) / pip for i in range(1, w + 1)
                    ]
                    try:
                        sigma_bar = _stats.stdev(bar_deltas)         # pips/bar
                        sigma_per_min = sigma_bar / mpb              # pips/min
                        entry_norm[lbl] = (
                            round(mom / sigma_per_min, 4) if sigma_per_min > 0 else None
                        )
                    except _stats.StatisticsError:
                        entry_norm[lbl] = None
                else:
                    entry_norm[lbl] = None
        raw[pair] = entry_raw
        norm[pair] = entry_norm
    return raw, norm


def compute_csi(pair_signals: dict) -> dict:
    csi: dict[str, dict] = {}
    for currency, pairs in CURRENCY_PAIRS.items():
        csi[currency] = {}
        for lbl in ALL_WINDOW_LABELS:
            vals = []
            for pair, sign in pairs:
                v = pair_signals.get(pair, {}).get(lbl)
                if v is not None:
                    vals.append(sign * v)
            csi[currency][lbl] = round(sum(vals) / len(vals), 4) if vals else None
    return csi


def compute_accel(pair_signals: dict, prev: dict) -> dict:
    """Acceleration = change in momentum per poll minute (0.5 min interval)."""
    result: dict[str, dict] = {}
    for pair in PAIRS:
        entry: dict[str, float | None] = {}
        for lbl in ALL_WINDOW_LABELS:
            cur = pair_signals.get(pair, {}).get(lbl)
            prv = prev.get(pair, {}).get(lbl)
            if cur is not None and prv is not None:
                entry[lbl] = round((cur - prv) / 0.5, 4)
            else:
                entry[lbl] = None
        result[pair] = entry
    return result


def compute_csi_accel(csi: dict, prev: dict) -> dict:
    """Acceleration for CSI values per currency per window."""
    result: dict[str, dict] = {}
    for currency in CURRENCY_PAIRS:
        entry: dict[str, float | None] = {}
        for lbl in ALL_WINDOW_LABELS:
            cur = csi.get(currency, {}).get(lbl)
            prv = prev.get(currency, {}).get(lbl)
            if cur is not None and prv is not None:
                entry[lbl] = round((cur - prv) / 0.5, 4)
            else:
                entry[lbl] = None
        result[currency] = entry
    return result


def compute_weighted_sum(signals: dict) -> dict:
    """Weighted average of momentum across all 7 windows.

    signals: dict[key, dict[window, float|None]]
    Returns: dict[key, float|None]
    """
    result: dict[str, float | None] = {}
    for key, windows in signals.items():
        total_w = 0.0
        total_wv = 0.0
        for lbl, w in WINDOW_WEIGHTS.items():
            v = windows.get(lbl)
            if v is not None:
                total_wv += w * v
                total_w += w
        result[key] = round(total_wv / total_w, 4) if total_w > 0 else None
    return result


def _sma7(closes: list[float], anchor: int) -> float | None:
    """SMA7 sampled at 60-bar spacing.
    sma7[t] = mean(c[t-1], c[t-61], c[t-121], c[t-181], c[t-241], c[t-301], c[t-361]).
    `anchor` is the lag-offset (0 = current; 60 = 1 M5-equivalent ago).
    Requires `closes` indexed from oldest to newest; we read c[-1-anchor-...]."""
    samples = []
    n = len(closes)
    # Offsets back from current: 1, 61, 121, 181, 241, 301, 361 — all shifted by `anchor`.
    for off in (1, 61, 121, 181, 241, 301, 361):
        idx = n - 1 - (off + anchor)
        if idx < 0:
            return None
        samples.append(closes[idx])
    return sum(samples) / 7.0


def compute_xbreak(pair_closes_s5: dict[str, list[float]],
                   pip_fn) -> dict[str, dict]:
    """For each pair, compute the H1-cross-breakout indicators on S5 stream
    using 60-S5-bar spacing (60 S5 bars = M5-equivalent).

    Returns dict[pair] = {
       'c': current close,
       'sma7': sma7 at current,
       'sma7_60', 'sma7_120', 'sma7_180', 'sma7_240': sma7 at lags,
       'xover_4ago_p', 'xover_3ago_p': (c - sma7) at lag 240 / 180, in pips,
       'small_move_p': c[t-120] - c[t-180] in pips,
       'large_move_p': c[t-60]  - c[t-120] in pips,
       'accel': bool (|large| > |small|),
       'current_mv_p': c[t] - c[t-60] in pips,
       'gap_shrink_1': bool, |c60-sma7_60| < |c120-sma7_120|,
       'gap_shrink_2': bool, |c-sma7|     < |c60-sma7_60|,
       'long_armed':  bool,
       'short_armed': bool,
    }
    """
    out: dict[str, dict] = {}
    for pair, closes in pair_closes_s5.items():
        n = len(closes)
        if n < 602:    # need c[t-601] for sma7_lag240 oldest sample
            out[pair] = {k: None for k in (
                "c","sma7","sma7_60","sma7_120","sma7_180","sma7_240",
                "xover_4ago_p","xover_3ago_p","small_move_p","large_move_p",
                "accel","current_mv_p","gap_shrink_1","gap_shrink_2",
                "long_armed","short_armed")}
            continue
        pip = pip_fn(pair)
        c       = closes[-1]
        c_60    = closes[-1 - 60]
        c_120   = closes[-1 - 120]
        c_180   = closes[-1 - 180]
        c_240   = closes[-1 - 240]
        sma7_now = _sma7(closes, anchor=0)
        sma7_60  = _sma7(closes, anchor=60)
        sma7_120 = _sma7(closes, anchor=120)
        sma7_180 = _sma7(closes, anchor=180)
        sma7_240 = _sma7(closes, anchor=240)

        xover_4_p = (c_240 - sma7_240) / pip if sma7_240 is not None else None
        xover_3_p = (c_180 - sma7_180) / pip if sma7_180 is not None else None
        small_p   = (c_120 - c_180) / pip
        large_p   = (c_60  - c_120) / pip
        cur_p     = (c     - c_60)  / pip
        accel     = abs(large_p) > abs(small_p)

        gap_60   = abs(c_60  - sma7_60)  / pip if sma7_60  is not None else None
        gap_120  = abs(c_120 - sma7_120) / pip if sma7_120 is not None else None
        gap_now  = abs(c     - sma7_now) / pip if sma7_now is not None else None
        gap_shrink_1 = (gap_60 < gap_120) if (gap_60 is not None and gap_120 is not None) else None
        gap_shrink_2 = (gap_now < gap_60) if (gap_now is not None and gap_60  is not None) else None

        long_armed = (
            xover_4_p is not None and xover_3_p is not None and
            xover_4_p < 0 and xover_3_p > 0 and
            small_p > 0 and large_p > small_p and
            accel and cur_p > 0
        )
        short_armed = (
            xover_4_p is not None and xover_3_p is not None and
            xover_4_p > 0 and xover_3_p < 0 and
            small_p < 0 and large_p < small_p and
            accel and cur_p < 0
        )

        out[pair] = {
            "c":           round(c, 5),
            "sma7":        round(sma7_now, 5) if sma7_now is not None else None,
            "sma7_60":     round(sma7_60, 5) if sma7_60 is not None else None,
            "sma7_120":    round(sma7_120, 5) if sma7_120 is not None else None,
            "sma7_180":    round(sma7_180, 5) if sma7_180 is not None else None,
            "sma7_240":    round(sma7_240, 5) if sma7_240 is not None else None,
            "xover_4ago_p": round(xover_4_p, 2) if xover_4_p is not None else None,
            "xover_3ago_p": round(xover_3_p, 2) if xover_3_p is not None else None,
            "small_move_p": round(small_p, 2),
            "large_move_p": round(large_p, 2),
            "accel":        bool(accel),
            "current_mv_p": round(cur_p, 2),
            "gap_shrink_1": gap_shrink_1,
            "gap_shrink_2": gap_shrink_2,
            "long_armed":   bool(long_armed),
            "short_armed":  bool(short_armed),
        }
    return out


def init_history_db() -> None:
    con = duckdb.connect(str(HISTORY_DB))
    con.execute("""
        CREATE TABLE IF NOT EXISTS pair_signals (
            ts TIMESTAMP NOT NULL,
            pair VARCHAR NOT NULL,
            tf VARCHAR NOT NULL,
            momentum DOUBLE,
            accel DOUBLE,
            weighted_sum DOUBLE,
            momentum_norm DOUBLE         -- σ-scaled momentum (Sharpe-like)
        )
    """)
    # Backfill column for existing DBs (DuckDB IF NOT EXISTS is a no-op if
    # the column is already there).
    try:
        con.execute("ALTER TABLE pair_signals ADD COLUMN IF NOT EXISTS momentum_norm DOUBLE")
    except Exception:
        pass
    con.execute("""
        CREATE TABLE IF NOT EXISTS csi_signals (
            ts TIMESTAMP NOT NULL,
            currency VARCHAR NOT NULL,
            tf VARCHAR NOT NULL,
            value DOUBLE,
            accel DOUBLE,
            csi_weighted_sum DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS xbreak_signals (
            ts TIMESTAMP NOT NULL,
            pair VARCHAR NOT NULL,
            c DOUBLE,
            sma7 DOUBLE,
            xover_4ago_p DOUBLE,
            xover_3ago_p DOUBLE,
            small_move_p DOUBLE,
            large_move_p DOUBLE,
            current_mv_p DOUBLE,
            accel BOOLEAN,
            gap_shrink_1 BOOLEAN,
            gap_shrink_2 BOOLEAN,
            long_armed BOOLEAN,
            short_armed BOOLEAN
        )
    """)
    con.close()


def append_xbreak_history(ts_str: str, xbreak: dict) -> None:
    try:
        con = duckdb.connect(str(HISTORY_DB))
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        rows = []
        for pair in PAIRS:
            x = xbreak.get(pair) or {}
            rows.append((
                ts, pair,
                x.get("c"), x.get("sma7"),
                x.get("xover_4ago_p"), x.get("xover_3ago_p"),
                x.get("small_move_p"), x.get("large_move_p"),
                x.get("current_mv_p"),
                x.get("accel"),
                x.get("gap_shrink_1"), x.get("gap_shrink_2"),
                x.get("long_armed"), x.get("short_armed"),
            ))
        con.executemany(
            "INSERT INTO xbreak_signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        con.close()
    except Exception as exc:
        logger.warning(f"append_xbreak_history failed (non-fatal): {exc}")


def append_history(ts_str: str, pair_signals: dict, pair_accel: dict,
                   pair_ws: dict, csi: dict, csi_accel: dict, csi_ws: dict,
                   pair_signals_norm: dict | None = None) -> None:
    try:
        con = duckdb.connect(str(HISTORY_DB))
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

        pair_rows = []
        for pair in PAIRS:
            for lbl in ALL_WINDOW_LABELS:
                pair_rows.append((
                    ts, pair, lbl,
                    pair_signals.get(pair, {}).get(lbl),
                    pair_accel.get(pair, {}).get(lbl),
                    pair_ws.get(pair),
                    (pair_signals_norm or {}).get(pair, {}).get(lbl),
                ))
        con.executemany(
            "INSERT INTO pair_signals VALUES (?, ?, ?, ?, ?, ?, ?)", pair_rows
        )

        csi_rows = []
        for currency in CURRENCY_PAIRS:
            for lbl in ALL_WINDOW_LABELS:
                csi_rows.append((
                    ts, currency, lbl,
                    csi.get(currency, {}).get(lbl),
                    csi_accel.get(currency, {}).get(lbl),
                    csi_ws.get(currency),
                ))
        con.executemany(
            "INSERT INTO csi_signals VALUES (?, ?, ?, ?, ?, ?)", csi_rows
        )
        con.close()
    except Exception as exc:
        logger.warning(f"append_history failed (non-fatal): {exc}")


def prune_history() -> None:
    """Delete history rows older than HISTORY_RETENTION_DAYS (keeps the DB bounded;
    live signals in signals_state.json are unaffected)."""
    try:
        con = duckdb.connect(str(HISTORY_DB))
        cut = f"now() - INTERVAL {HISTORY_RETENTION_DAYS} DAY"
        for tbl in ("pair_signals", "csi_signals", "xbreak_signals"):
            try:
                con.execute(f"DELETE FROM {tbl} WHERE ts < {cut}")
            except Exception:
                pass
        con.execute("CHECKPOINT")
        con.close()
        logger.info(f"history pruned: rows older than {HISTORY_RETENTION_DAYS}d removed")
    except Exception as exc:
        logger.warning(f"history prune failed (non-fatal): {exc}")


def write_state(
    pair_signals: dict,
    pair_accel: dict,
    pair_ws: dict,
    csi: dict,
    csi_accel: dict,
    csi_ws: dict,
    xbreak: dict,
    pair_signals_norm: dict | None = None,
    csi_pairs: dict | None = None,
) -> None:
    state = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "windows": ALL_WINDOW_LABELS,
        "pairs": pair_signals,
        "pairs_norm": pair_signals_norm or {},   # σ-scaled momentum
        "accel": pair_accel,
        "weighted_sum": pair_ws,
        "csi": csi,                                # currency strength index
        "csi_accel": csi_accel,
        "csi_weighted_sum": csi_ws,
        "csi_pairs": csi_pairs or {},              # Wilder CSI per-pair tradeability
        "xbreak": xbreak,
    }
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, separators=(",", ":")))
    tmp.rename(STATE_FILE)


def run() -> None:
    global prev_pair_signals, prev_csi

    global _csi_cache, _csi_last, _last_hist_prune
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    init_history_db()
    warmup()
    fetch_econ()
    logger.info("Warm-up complete. Entering main loop.")

    while True:
        try:
            pair_signals, pair_signals_norm = compute_pair_signals()
            csi = compute_csi(pair_signals)

            pair_accel = compute_accel(pair_signals, prev_pair_signals)
            pair_ws = compute_weighted_sum(pair_signals)
            csi_accel = compute_csi_accel(csi, prev_csi)
            csi_ws = compute_weighted_sum(csi)

            # H1 cross-breakout indicators from the S5 close buffers.
            s5_closes = {p: [b["c"] for b in bufs["S5"][p]] for p in PAIRS}
            xbreak = compute_xbreak(s5_closes, _pip)

            # Refresh slow per-pair Wilder CSI every CSI_REFRESH_SEC.
            now_t = time.time()
            if ECON and (now_t - _csi_last >= CSI_REFRESH_SEC or not _csi_cache):
                _csi_cache = compute_csi_pairs()
                _csi_last = now_t

            # Bound the history DB hourly.
            if now_t - _last_hist_prune >= HISTORY_PRUNE_EVERY_SEC:
                _last_hist_prune = now_t
                prune_history()

            ts_str = datetime.now(timezone.utc).isoformat()
            append_history(ts_str, pair_signals, pair_accel, pair_ws,
                           csi, csi_accel, csi_ws,
                           pair_signals_norm=pair_signals_norm)
            append_xbreak_history(ts_str, xbreak)
            write_state(pair_signals, pair_accel, pair_ws,
                        csi, csi_accel, csi_ws, xbreak,
                        pair_signals_norm=pair_signals_norm,
                        csi_pairs=_csi_cache)

            prev_pair_signals = pair_signals
            prev_csi = csi

            logger.info(f"Signals written: {len(pair_signals)} pairs")
        except Exception as exc:
            logger.error(f"Compute error: {exc}")

        time.sleep(POLL_INTERVAL)
        update_latest()


if __name__ == "__main__":
    run()
