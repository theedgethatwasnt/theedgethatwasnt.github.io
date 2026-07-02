#!/usr/bin/env python3
"""
Post-Shock Counter-Trend Retrace Live — Account 009, 4 JPY pairs, 1 unit
===========================================================================
Strategy (validated thr=2.5, peak=44b, tp=20p, OOS +56p/d WF=12/12 mc_p=0.0000):
  1. Detect S5 velocity shock: |z| > 2.5  (30s velocity, rolling MAD, 2048-bar window)
  2. Monitor 44 S5 bars for retrace peak to develop (~220s)
     — confirmed by MSP wavelet study D3→D5 xcorr peak at 110-220s
  3. Markov regime filter (mw=10 mt=0.002 sig_thr=0.20, +70.4p/d WF=9/12):
     upshock→SHORT requires Bull regime (signal>0.20); downshock→LONG requires Bear (<-0.20)
  4. Enter counter-trend at market: upshock→SHORT, downshock→LONG
  5. TP=20p  SL=30p (broker-side; avg non-TP loss=5.4p, SL guards extreme tails)
  6. Force-close after HORIZON=600 S5 bars (~50 min) if neither TP nor SL hit

SOP compliance (CLAUDE.md §Backtest–Live Consistency SOP):
  R1  Closed bars only — last_s5_ts guards; D1 adapter returns c.complete only
  R3  Mid close used for z-score; bid_c/ask_c for spread gate + fill price
  R4  Causal z-score: VelZScore deque (no pandas rolling, MAD_WIN=2048)
      Causal Markov: T built from transitions up to (but not including) day D-1→D
  R5  IS P90 spread gates hardcoded per pair
  R6  State machine mirrors sim_retrace() + sim_retrace_filtered() in backtest
  R9  Known divergence: live enters at monitor_bars==PEAK_BARS (bar t+44) vs
      backtest watch_start=t+45. Delta=1×5s=immaterial at 220s peak window.
"""

import os, sys, json, time, signal, logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from dotenv import load_dotenv; load_dotenv()

from lib.oanda_adapter import OANDAAdapter
from lib.db import write_trade_direct

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [retrace] %(message)s",
)
logger = logging.getLogger("retrace")

# ── Account + sizing ──────────────────────────────────────────────────────────
ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID_009", "<OANDA_ACCOUNT_ID>")
STATE_DIR  = os.environ.get("RETRACE_STATE_DIR", "/data/logs")
UNITS_PER_DOLLAR     = 1.5   # 2026-06-11: bumped from 1.25 (was hardcoded 1u → 1.25 on 2026-06-09)
# 2026-06-11: MAX_UNITS cap removed — sizing is uncapped balance × 1.5.
BALANCE_REFRESH_SECS = 3600
_units                = 1     # live cached unit count
_last_balance_refresh = 0.0
STRATEGY   = "post_shock_retrace"
LABEL      = "retrace_009"

# ── Strategy params (validated thr=2.5, peak=44b, TP=20p) ────────────────────
THR        = 2.5    # shock z-score threshold
PEAK_BARS  = 44     # S5 bars to monitor after shock before entry (~220s)
TP_PIPS    = 20.0   # take profit
SL_PIPS    = 30.0   # stop loss
HORIZON    = 600    # max S5 bars in position (~50 min) before force-close
COOLDOWN   = (PEAK_BARS + HORIZON) // 2   # 322 bars between shocks per pair
# ATR entry gate (validated 2026-06-17 backtest_retrace_atr.py: gate>=5 + fix20p TP
# → +79.4 p/d OOS, WF 11/12, mc_p=0.0000 vs no-gate +55.9 WF 7/12). Skip shocks in
# low-volatility regimes that can't produce a 20-pip retrace (the dead-timeout cause).
# Set RETRACE_ATR_GATE=0 to disable. Fail-OPEN on fetch error (preserve old behavior).
ATR_GATE   = float(os.environ.get("RETRACE_ATR_GATE", "5.0"))   # M5-ATR(14) pips
Z_WIN      = 6      # velocity window: vel = close[t] - close[t-Z_WIN]
MAD_WIN    = 2048   # rolling MAD window for z-score normalisation
WARMUP_N   = MAD_WIN + Z_WIN + 20   # S5 bars to fetch on startup (~2.9h)
POLL_SECS  = 5      # S5 poll interval

# ── Markov regime filter (validated Phase 3: +70.4p/d WF=9/12 mc_p=0.0000) ───
MARKOV_MW      = 10     # D1 rolling window for regime detection
MARKOV_MT      = 0.002  # cumret threshold: |cumret|>MT → Bull/Bear, else Sideways
MARKOV_SIG_THR = 0.20   # filter gate: shock_dir * signal > SIG_THR → allow entry
MARKOV_D1_BARS = 250    # D1 bars to fetch for signal computation (~1 year)
MARKOV_MIN_PRIME = 30   # min T observations before generating signal

# ── 3 JPY pairs (EUR_JPY removed 2026-06-09: -54p/9t 11%WR live) ────────────
PAIRS = {
    "GBP_JPY": {"pip": 0.01, "sp_gate": 4.00},
    "USD_JPY": {"pip": 0.01, "sp_gate": 2.10},
    "AUD_JPY": {"pip": 0.01, "sp_gate": 2.30},
}

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT_ID", "")
_shutdown = False

# ── Per-pair tripwire ─────────────────────────────────────────────────────────
# At N>=TRIPWIRE_MIN_N closed trades AND cumulative pips < 0 since LIVE_SINCE,
# auto-disable that pair. Re-enables if cum recovers to >= 0.
TRIPWIRE_MIN_N           = 30
TRIPWIRE_LIVE_SINCE      = "2026-05-25"   # 4-pair config start date
TRIPWIRE_CHECK_INTERVAL  = 300.0          # 5 min — DuckDB is cheap, this is plenty
TRIPWIRE_DB_PATH         = "/data/db/trades.duckdb"
_disabled_pairs: set[str] = set()
_last_tripwire_check: float = 0.0


def _refresh_tripwire():
    """Read per-pair lifetime cum_pips since LIVE_SINCE from trades.duckdb.
    Toggle membership in _disabled_pairs accordingly. Non-blocking — failures
    leave the prior state intact.
    """
    global _last_tripwire_check, _disabled_pairs
    now = time.time()
    if now - _last_tripwire_check < TRIPWIRE_CHECK_INTERVAL:
        return
    _last_tripwire_check = now
    try:
        import duckdb
        con = duckdb.connect(TRIPWIRE_DB_PATH, read_only=True)
        rows = con.execute(
            """
            SELECT pair,
                   COUNT(*)            AS n,
                   COALESCE(SUM(pnl_pips), 0.0) AS pips
            FROM trades
            WHERE is_paper = FALSE
              AND account_id = ?
              AND exit_time >= ?
            GROUP BY pair
            """,
            [ACCOUNT_ID, TRIPWIRE_LIVE_SINCE],
        ).fetchall()
        con.close()
    except Exception as e:
        logger.warning(f"tripwire refresh failed: {e}")
        return

    new_disabled: set[str] = set()
    for pair, n, pips in rows:
        if pair not in PAIRS:
            continue
        if int(n) >= TRIPWIRE_MIN_N and float(pips) < 0.0:
            new_disabled.add(pair)

    added   = new_disabled - _disabled_pairs
    removed = _disabled_pairs - new_disabled
    for p in sorted(added):
        msg = (f"🛑 009 TRIPWIRE: {p} DISABLED "
               f"(N≥{TRIPWIRE_MIN_N} closes since {TRIPWIRE_LIVE_SINCE}, cum negative)")
        logger.warning(msg)
        _tg(msg)
    for p in sorted(removed):
        msg = f"✅ 009 TRIPWIRE: {p} RE-ENABLED (cum recovered to ≥ 0)"
        logger.info(msg)
        _tg(msg)
    _disabled_pairs = new_disabled


def _refresh_units(adapter):
    """Refresh _units from OANDA balance × UNITS_PER_DOLLAR. Called hourly."""
    global _units, _last_balance_refresh
    _last_balance_refresh = time.time()
    try:
        info = adapter.get_account_summary()
        if info is not None:
            new_units = max(1, round(info.balance * UNITS_PER_DOLLAR))
            if new_units != _units:
                logger.info(
                    f"Units updated: ${info.balance:.2f} × {UNITS_PER_DOLLAR} "
                    f"→ {new_units}u/trade (was {_units}u)"
                )
            else:
                logger.info(f"Balance refresh: ${info.balance:.2f} → {new_units}u/trade")
            _units = new_units
    except Exception as e:
        logger.warning(f"Balance refresh failed: {e} — keeping {_units}u")


def _tg(msg: str):
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": msg},
            timeout=5,
        )
    except Exception:
        pass


# ── Markov regime filter (R4/R9: causal D1 transition matrix) ────────────────

class MarkovFilter:
    """
    Causal D1 Markov regime transition matrix.
    Signal = P(Bull|state_yesterday) - P(Bear|state_yesterday).
    Filter: shock_dir * signal > MARKOV_SIG_THR to allow entry.
    Refreshes once per UTC calendar day.
    """
    BULL, SIDE, BEAR = 0, 1, 2

    def __init__(self):
        self._signals: dict[str, dict[str, float]] = {}  # pair -> {date_str -> signal}
        self._last_date: str = ""

    def _compute(self, bars: list) -> dict[str, float]:
        """bars: list of completed D1 bars. Returns {date_str: signal}."""
        if len(bars) < MARKOV_MW + MARKOV_MIN_PRIME + 5:
            return {}

        closes = pd.Series(
            [b["close"] for b in bars],
            index=pd.to_datetime([b["timestamp"] for b in bars], utc=True),
        ).sort_index()
        lr = np.log(closes / closes.shift(1)).dropna()

        roll = lr.rolling(MARKOV_MW).sum()
        states = roll.map(
            lambda r: (self.BULL if r > MARKOV_MT else
                       self.BEAR if r < -MARKOV_MT else
                       self.SIDE) if not np.isnan(r) else np.nan
        ).dropna().astype(int)

        T = np.zeros((3, 3), dtype=np.float64)
        signals: dict[str, float] = {}
        for i in range(len(states) - 1):
            s     = int(states.iloc[i])
            s_nxt = int(states.iloc[i + 1])
            row_sum = T[s].sum()
            if row_sum >= MARKOV_MIN_PRIME:
                sig = (T[s, self.BULL] - T[s, self.BEAR]) / row_sum
                nxt_day = states.index[i + 1].strftime("%Y-%m-%d")
                signals[nxt_day] = float(sig)
            T[s, s_nxt] += 1.0

        return signals

    def refresh(self, adapter, pairs: list, force: bool = False):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not force and self._last_date == today:
            return
        for pair in pairs:
            try:
                # adapter returns only c.complete bars, so in-progress D1 is excluded (R1)
                bars = adapter.get_candles(pair, count=MARKOV_D1_BARS, granularity="D")
                if not bars:
                    logger.warning(f"Markov {pair}: no D1 bars returned")
                    continue
                sigs = self._compute(bars)
                self._signals[pair] = sigs
                latest = max(sigs.keys()) if sigs else "none"
                logger.info(f"Markov {pair}: {len(sigs)} signal-days  latest={latest}")
            except Exception as e:
                logger.warning(f"Markov {pair}: refresh error: {e}")
        self._last_date = today

    def allows_entry(self, pair: str, shock_dir: int) -> tuple[bool, float]:
        """
        Returns (allowed, signal_value).
        signal for today D = P(Bull|state_{D-1}) − P(Bear|state_{D-1})
        stored under key D in signals dict (see compute_signals in backtest).
        """
        sigs = self._signals.get(pair, {})
        if not sigs:
            logger.warning(f"{pair}: Markov no signals → allow through")
            return True, float("nan")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sig = sigs.get(today)
        if sig is None:
            latest = max(sigs.keys())
            sig = sigs[latest]
            logger.debug(f"{pair}: Markov {today} not in dict, using {latest}")

        passes = float(shock_dir * sig) > MARKOV_SIG_THR
        return passes, float(sig)


# ── Causal velocity z-score (R4: mirrors compute_shock_z in backtest) ─────────

class VelZScore:
    """
    Rolling MAD z-score of S5 velocity. Causal: deque-based, no lookahead.
    vel[t] = (close[t] - close[t-Z_WIN]) / pip  (30s momentum)
    z[t] = (vel[t] - median(vel_window)) / (1.4826 * MAD(vel_window))
    """
    def __init__(self, pip: float):
        self._pip    = pip
        self._closes = deque(maxlen=Z_WIN + 1)   # 7 closes → Z_WIN=6 bar velocity
        self._vels   = deque(maxlen=MAD_WIN)

    def update(self, close: float) -> Optional[float]:
        """Feed one completed S5 close. Returns z-score or None if not warmed up."""
        self._closes.append(close)
        if len(self._closes) < Z_WIN + 1:
            return None
        vel = (self._closes[-1] - self._closes[0]) / self._pip
        self._vels.append(vel)
        if len(self._vels) < 50:    # min_periods matches backtest
            return None
        arr = np.array(self._vels, dtype=np.float64)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad < 1e-6:
            return 0.0
        return float((vel - med) / (1.4826 * mad))

    def ready(self) -> bool:
        return len(self._vels) >= 50


# ── Per-pair persistent state ─────────────────────────────────────────────────

@dataclass
class PairState:
    pair:          str
    pip:           float
    sp_gate:       float
    # State machine: IDLE | MONITORING | IN_POSITION
    status:        str   = "IDLE"
    shock_dir:     int   = 0    # +1=upshock→SHORT entry, -1=downshock→LONG entry
    monitor_bars:  int   = 0    # bars elapsed since shock onset (during MONITORING)
    # Position
    pos:           int   = 0    # +1=LONG, -1=SHORT
    entry_px:      float = 0.0
    tp_price:      float = 0.0
    sl_price:      float = 0.0
    trade_id:      str   = ""
    entry_time:    str   = ""
    position_bars: int   = 0    # bars elapsed in position (for HORIZON exit)
    mfe_pips:      float = 0.0  # max favourable excursion (pips, always ≥ 0)
    mae_pips:      float = 0.0  # max adverse excursion (pips, always ≤ 0)
    # Cooldown
    cooldown_bars: int   = 0
    # Last processed S5 bar timestamp (persisted for replay on restart)
    last_s5_ts:    str   = ""


_SKIP = {"pair", "pip", "sp_gate"}


def _state_path(pair: str) -> str:
    return os.path.join(STATE_DIR, f"retrace_{pair}_state.json")


def _save(st: PairState):
    data = {k: v for k, v in st.__dict__.items() if k not in _SKIP}
    try:
        with open(_state_path(st.pair), "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"{st.pair}: save failed: {e}")


def _load(st: PairState):
    path = _state_path(st.pair)
    if not os.path.exists(path):
        return
    try:
        data = json.load(open(path))
        for k, v in data.items():
            if hasattr(st, k) and k not in _SKIP:
                setattr(st, k, v)
        logger.info(
            f"{st.pair}: loaded state={st.status}  pos={st.pos}  "
            f"cooldown={st.cooldown_bars}  "
            f"last_s5={st.last_s5_ts[-16:] if st.last_s5_ts else '—'}"
        )
    except Exception as e:
        logger.warning(f"{st.pair}: load failed: {e}")


# ── Trade lifecycle ───────────────────────────────────────────────────────────

def _m5_atr14_pips(adapter: OANDAAdapter, pair: str, pip: float) -> Optional[float]:
    """Causal M5-ATR(14) in pips from completed M5 candles (matches the backtest's
    M5-ATR computed on S5). Returns None on fetch failure (caller fails open)."""
    try:
        bars = adapter.get_candles(pair, count=16, granularity="M5")
        completed = [b for b in (bars or []) if b.get("complete", True)]
        if len(completed) < 15:
            completed = (bars or [])[:-1]      # drop in-progress bar (R1)
        if len(completed) < 15:
            return None
        h = [b["high"] for b in completed]
        l = [b["low"] for b in completed]
        c = [b["close"] for b in completed]
        trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
               for i in range(1, len(c))]
        return (sum(trs[-14:]) / 14) / pip
    except Exception as e:
        logger.warning(f"{pair}: M5-ATR fetch failed ({e})")
        return None


def _enter(st: PairState, adapter: OANDAAdapter, bid_c: float, ask_c: float):
    if os.environ.get("NO_NEW_ENTRIES", "") == "1":
        logger.info(f"{st.pair}: NO_NEW_ENTRIES set — new entry skipped")
        return
    _refresh_tripwire()
    if st.pair in _disabled_pairs:
        logger.info(
            f"{st.pair}: tripwire active "
            f"(N≥{TRIPWIRE_MIN_N} closes, cum<0) — entry skipped"
        )
        return
    """Place counter-trend market order after PEAK_BARS monitoring window."""
    direction    = -st.shock_dir          # upshock (z>0) → SHORT (-1)
    units_signed = _units * direction
    fill_ref     = bid_c if direction == -1 else ask_c   # sell at bid, buy at ask
    spread_p     = (ask_c - bid_c) / st.pip

    # direction=+1 LONG:  tp above entry (+), sl below entry (-)
    # direction=-1 SHORT: tp below entry (-), sl above entry (+)
    tp = fill_ref + direction * TP_PIPS * st.pip
    sl = fill_ref - direction * SL_PIPS * st.pip

    result = adapter.place_market_order(st.pair, units_signed, sl_price=sl, tp_price=tp)
    if not result.success:
        logger.warning(f"{st.pair}: order rejected ({getattr(result,'error','')}) → IDLE cooldown")
        _reset_to_idle(st)
        _save(st)
        return

    fill_px   = result.fill_price or fill_ref
    tp_actual = fill_px + direction * TP_PIPS * st.pip
    sl_actual = fill_px - direction * SL_PIPS * st.pip
    dir_str   = "LONG" if direction == 1 else "SHORT"
    dir_sym   = "🟢" if direction == 1 else "🔴"
    shock_str = "↑shock" if st.shock_dir == 1 else "↓shock"

    st.status        = "IN_POSITION"
    st.pos           = direction
    st.entry_px      = fill_px
    st.tp_price      = tp_actual
    st.sl_price      = sl_actual
    st.trade_id      = result.trade_id or ""
    st.entry_time    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    st.position_bars = 0

    _tg(
        f"{dir_sym} 009 {st.pair} OPEN {dir_str} @ {fill_px:.3f}  {_units}u\n"
        f"[{shock_str}→retrace]  TP={tp_actual:.3f} (+{TP_PIPS:.0f}p)  "
        f"SL={sl_actual:.3f} (-{SL_PIPS:.0f}p)  spread={spread_p:.2f}p"
    )
    logger.info(
        f"{st.pair} OPEN {dir_str} @ {fill_px:.3f} TP={tp_actual:.3f} "
        f"SL={sl_actual:.3f} id={st.trade_id} spread={spread_p:.2f}p"
    )
    _save(st)


def _record_close(st: PairState, exit_px: float, reason: str):
    pnl   = (exit_px - st.entry_px) / st.pip * st.pos
    dur_h = 0.0
    if st.entry_time:
        try:
            t0    = datetime.fromisoformat(st.entry_time.replace("Z", "+00:00"))
            dur_h = (datetime.now(timezone.utc) - t0).total_seconds() / 3600
        except Exception:
            pass
    cap = pnl / st.mfe_pips if st.mfe_pips > 0 else 0.0
    write_trade_direct(
        strategy=STRATEGY, pair=st.pair, account_id=ACCOUNT_ID,
        trade_id=st.trade_id, direction=st.pos,
        entry_price=st.entry_px, exit_price=exit_px,
        entry_time=st.entry_time,
        exit_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        pnl_pips=pnl, exit_reason=reason, hours_held=dur_h, units=_units,
        mfe_pips=st.mfe_pips, mae_pips=st.mae_pips, capture_ratio=cap,
        is_paper=False, label=LABEL,
    )
    dir_sym = "🟢" if st.pos == 1 else "🔴"
    pl_sym  = "🟢" if pnl >= 0 else "🔴"
    _tg(
        f"{pl_sym} 009 {st.pair} CLOSE {dir_sym}  [{reason}]\n"
        f"P/L {pnl:+.1f}p  held {dur_h:.1f}h"
    )
    logger.info(
        f"{st.pair} CLOSE pos={st.pos} [{reason}] @ {exit_px:.3f} "
        f"pnl={pnl:+.1f}p held={dur_h:.1f}h"
    )


def _reset_to_idle(st: PairState):
    st.status        = "IDLE"
    st.shock_dir     = 0
    st.monitor_bars  = 0
    st.pos           = 0
    st.entry_px      = 0.0
    st.tp_price      = 0.0
    st.sl_price      = 0.0
    st.trade_id      = ""
    st.entry_time    = ""
    st.position_bars = 0
    st.mfe_pips      = 0.0
    st.mae_pips      = 0.0
    st.cooldown_bars = COOLDOWN


# ── Per-bar state machine ─────────────────────────────────────────────────────

def _process_bar(st: PairState, adapter: OANDAAdapter, zs: VelZScore,
                 markov: MarkovFilter,
                 close: float, bid_c: float, ask_c: float):
    """Process one completed S5 bar through the state machine."""

    # Always update z-score deque (keeps history current across all states)
    z = zs.update(close)

    # Cooldown tick (always decrement regardless of state)
    if st.cooldown_bars > 0:
        st.cooldown_bars -= 1

    # ── IN_POSITION ──────────────────────────────────────────────────────────
    if st.status == "IN_POSITION":
        st.position_bars += 1
        try:
            open_trades = adapter.get_open_trades()
            our = next(
                (t for t in open_trades
                 if t.instrument == st.pair and t.trade_id == st.trade_id),
                None,
            )
        except Exception as e:
            logger.warning(f"{st.pair}: OANDA check error: {e}")
            return

        if our is not None:
            unreal = (close - st.entry_px) / st.pip * st.pos
            if unreal > st.mfe_pips:
                st.mfe_pips = unreal
            if unreal < st.mae_pips:
                st.mae_pips = unreal

        if our is None:
            # Trade no longer open — infer TP vs SL from price proximity
            reason  = "tp" if abs(close - st.tp_price) < abs(close - st.sl_price) else "sl"
            exit_px = st.tp_price if reason == "tp" else st.sl_price
            _record_close(st, exit_px, reason)
            _reset_to_idle(st)
            _save(st)
            return

        if st.position_bars >= HORIZON:
            logger.info(f"{st.pair}: HORIZON {HORIZON} bars → force-close at market")
            try:
                adapter.close_trade(st.trade_id)
            except Exception as e:
                logger.error(f"{st.pair}: close_trade error: {e}")
            exit_px = bid_c if st.pos == -1 else ask_c
            _record_close(st, exit_px, "timeout")
            _reset_to_idle(st)
            _save(st)
        return

    # ── MONITORING ───────────────────────────────────────────────────────────
    if st.status == "MONITORING":
        st.monitor_bars += 1
        if st.monitor_bars >= PEAK_BARS:
            spread = (ask_c - bid_c) / st.pip
            if spread > st.sp_gate:
                logger.info(
                    f"{st.pair}: peak window done, spread {spread:.2f}p > "
                    f"gate {st.sp_gate:.2f}p → skip, cooldown"
                )
                _reset_to_idle(st)
                _save(st)
                return

            allowed, m_sig = markov.allows_entry(st.pair, st.shock_dir)
            if not allowed:
                logger.info(
                    f"{st.pair}: Markov BLOCKED  shock_dir={st.shock_dir:+d}  "
                    f"signal={m_sig:+.3f}  gate={MARKOV_SIG_THR}"
                )
                _reset_to_idle(st)
                _save(st)
                return

            logger.info(
                f"{st.pair}: Markov PASS  shock_dir={st.shock_dir:+d}  "
                f"signal={m_sig:+.3f}"
            )

            # ATR entry gate — skip low-vol regimes (validated +79.4 p/d OOS)
            if ATR_GATE > 0:
                atr = _m5_atr14_pips(adapter, st.pair, st.pip)
                if atr is None:
                    logger.warning(
                        f"{st.pair}: M5-ATR unavailable — fail-open, allowing entry"
                    )
                elif atr < ATR_GATE:
                    logger.info(
                        f"{st.pair}: ATR gate BLOCKED  M5-ATR14={atr:.2f}p "
                        f"< {ATR_GATE:.1f}p → skip, cooldown"
                    )
                    _reset_to_idle(st)
                    _save(st)
                    return
                else:
                    logger.info(
                        f"{st.pair}: ATR gate PASS  M5-ATR14={atr:.2f}p >= {ATR_GATE:.1f}p"
                    )

            _enter(st, adapter, bid_c, ask_c)
        return

    # ── IDLE ─────────────────────────────────────────────────────────────────
    if st.cooldown_bars > 0 or z is None or not zs.ready():
        return
    if abs(z) > THR:
        st.status       = "MONITORING"
        st.shock_dir    = 1 if z > 0 else -1
        st.monitor_bars = 0
        logger.info(
            f"{st.pair}: shock z={z:.2f} dir={st.shock_dir} → MONITORING "
            f"({PEAK_BARS} bars until entry)"
        )


# ── Warmup + reconcile ───────────────────────────────────────────────────────

def _warmup_pair(st: PairState, adapter: OANDAAdapter, zs: VelZScore,
                 markov: MarkovFilter):
    logger.info(f"{st.pair}: fetching {WARMUP_N} S5 bars for warmup …")
    bars = adapter.get_candles(st.pair, count=WARMUP_N + 1, granularity="S5")
    if not bars:
        logger.error(f"{st.pair}: warmup fetch returned no bars")
        return

    completed = bars[:-1]   # exclude in-progress bar (R1)

    # Seed z-scorer with historical bars (bars already processed before last_s5_ts)
    # Then replay state machine only for missed bars (newer than last_s5_ts)
    old_bars = [b for b in completed
                if not st.last_s5_ts or b["timestamp"] <= st.last_s5_ts]
    new_bars = [b for b in completed
                if st.last_s5_ts and b["timestamp"] > st.last_s5_ts]

    for b in old_bars:
        zs.update(b["close"])   # deque-only update, no state machine

    if new_bars:
        logger.info(
            f"{st.pair}: replaying {len(new_bars)} missed bars "
            f"(status={st.status})"
        )
        for b in new_bars:
            _process_bar(st, adapter, zs, markov, b["close"], b["bid_c"], b["ask_c"])
            st.last_s5_ts = b["timestamp"]
    elif old_bars:
        st.last_s5_ts = old_bars[-1]["timestamp"]

    # Reconcile open trades with OANDA
    try:
        open_trades = adapter.get_open_trades()
        pair_trades = [t for t in open_trades if t.instrument == st.pair]

        if pair_trades and st.status != "IN_POSITION":
            t = pair_trades[0]
            st.pos           = 1 if t.units > 0 else -1
            st.entry_px      = t.entry_price
            st.trade_id      = t.trade_id
            st.tp_price      = t.tp_price or (st.entry_px + st.pos * TP_PIPS * st.pip)
            st.sl_price      = st.entry_px - st.pos * SL_PIPS * st.pip
            st.entry_time    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            st.status        = "IN_POSITION"
            st.position_bars = 0
            logger.info(
                f"{st.pair}: adopted orphan trade {t.trade_id} "
                f"pos={st.pos} @ {st.entry_px:.3f}"
            )
        elif not pair_trades and st.status == "IN_POSITION":
            logger.warning(
                f"{st.pair}: state=IN_POSITION but no open trade — "
                f"recording as tp_or_sl"
            )
            reason  = "tp" if abs(st.entry_px - st.tp_price) < abs(st.entry_px - st.sl_price) else "sl"
            exit_px = st.tp_price if reason == "tp" else st.sl_price
            _record_close(st, exit_px, "tp_or_sl_on_restart")
            _reset_to_idle(st)
    except Exception as e:
        logger.warning(f"{st.pair}: reconcile error: {e}")

    logger.info(
        f"{st.pair}: warmup done  status={st.status}  pos={st.pos}  "
        f"z_ready={zs.ready()}  "
        f"last_s5={st.last_s5_ts[-16:] if st.last_s5_ts else '—'}"
    )
    _save(st)


# ── Market hours guard ────────────────────────────────────────────────────────

def _market_open() -> bool:
    """FX closed Sat all day, Sun before 21:00 UTC, Fri after 21:00 UTC."""
    now = datetime.now(timezone.utc)
    wd  = now.weekday()
    h   = now.hour
    if wd == 5:
        return False
    if wd == 6 and h < 21:
        return False
    if wd == 4 and h >= 21:
        return False
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def _signal_handler(sig, frame):
    global _shutdown
    logger.info(f"Signal {sig} → shutting down")
    _shutdown = True


def main():
    global _shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT,  _signal_handler)

    if not ACCOUNT_ID:
        logger.error("OANDA_ACCOUNT_ID_009 not set")
        sys.exit(1)

    adapter  = OANDAAdapter(account_id=ACCOUNT_ID)
    states   = {}
    zscorers = {}
    markov   = MarkovFilter()

    for pair, cfg in PAIRS.items():
        st = PairState(pair=pair, pip=cfg["pip"], sp_gate=cfg["sp_gate"])
        _load(st)
        states[pair]   = st
        zscorers[pair] = VelZScore(pip=cfg["pip"])

    # Dynamic sizing (balance × 1.25, hourly refresh, uncapped)
    _refresh_units(adapter)

    # Fetch D1 Markov signals before warmup so filter is ready on first missed bar
    logger.info("Refreshing Markov D1 signals …")
    markov.refresh(adapter, list(PAIRS.keys()), force=True)

    # Seed the per-pair tripwire on startup so the first entry honours it.
    _refresh_tripwire()
    if _disabled_pairs:
        logger.warning(
            f"Tripwire active on boot: {sorted(_disabled_pairs)} "
            f"(N≥{TRIPWIRE_MIN_N} since {TRIPWIRE_LIVE_SINCE}, cum<0)"
        )

    for pair, st in states.items():
        try:
            _warmup_pair(st, adapter, zscorers[pair], markov)
        except Exception as e:
            logger.error(f"{pair}: warmup failed: {e}")

    _tg(
        f"🟢 009 Post-Shock Retrace started\n"
        f"thr={THR}  peak={PEAK_BARS}b (~220s)  TP={TP_PIPS:.0f}p  "
        f"SL={SL_PIPS:.0f}p  {_units}u/trade (balance×{UNITS_PER_DOLLAR}, uncapped)\n"
        f"Markov filter: mw={MARKOV_MW} mt={MARKOV_MT} sig_thr={MARKOV_SIG_THR}\n"
        f"ATR gate: M5-ATR(14) >= {ATR_GATE:.1f}p (0=off)\n"
        f"Pairs: {', '.join(PAIRS)}"
    )
    logger.info(
        f"Retrace live — acct={ACCOUNT_ID}  {len(PAIRS)} pairs  {_units}u  "
        f"thr={THR}  peak={PEAK_BARS}  TP={TP_PIPS}  SL={SL_PIPS}  "
        f"horizon={HORIZON}  cooldown={COOLDOWN}  "
        f"markov mw={MARKOV_MW} mt={MARKOV_MT} sig_thr={MARKOV_SIG_THR}  "
        f"atr_gate={ATR_GATE:.1f}p"
    )

    while not _shutdown:
        if not _market_open():
            time.sleep(300)
            continue

        # Hourly sizing refresh
        if time.time() - _last_balance_refresh >= BALANCE_REFRESH_SECS:
            _refresh_units(adapter)

        # Refresh Markov signals once per UTC calendar day
        markov.refresh(adapter, list(PAIRS.keys()))

        for pair, st in states.items():
            try:
                bars = adapter.get_candles(pair, count=15, granularity="S5")
                if not bars:
                    continue
                completed = bars[:-1]   # exclude in-progress bar (R1)
                new_bars  = [b for b in completed if b["timestamp"] > st.last_s5_ts]
                for b in new_bars:
                    _process_bar(
                        st, adapter, zscorers[pair], markov,
                        b["close"], b["bid_c"], b["ask_c"],
                    )
                    st.last_s5_ts = b["timestamp"]
                if new_bars:
                    _save(st)
            except Exception as e:
                logger.error(f"{pair}: poll error: {e}")

        time.sleep(POLL_SECS)

    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
