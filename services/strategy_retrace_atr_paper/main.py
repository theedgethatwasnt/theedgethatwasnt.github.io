#!/usr/bin/env python3
"""
Post-Shock Retrace Paper — ATR≥5 entry gate, NO Markov, 4 JPY pairs
====================================================================
Identical to strategy_retrace_paper (thr=2.5, peak=44b, TP=20p, SL=30p,
HORIZON=600, NO Markov filter) with ONE addition: an M5-ATR(14) entry gate.
A shock is only taken when M5-ATR(14) at entry >= ATR_GATE pips (skip low-vol
regimes). This is the validated +79.4 p/d OOS / WF 11/12 / mc_p=0 config from
`research/experiments/shock_straddle/backtest_retrace_atr.py` (gate=5, fixed TP=20p).

Purpose: clean live A/B vs the existing label='retrace_nofilter' paper (which is
identical EXCEPT it has no ATR gate). The only difference between the two labels is
the gate, so the head-to-head isolates the gate's value live. 009 itself stays
OFFLINE — it was pulled for vol-compression decay; this paper-tests the proposed fix
before any live restart.

ATR computation mirrors the backtest EXACTLY: causal M5-ATR(14) broadcast from the
last COMPLETED M5 block, simple 14-period mean of true range, expressed in pips
(compute_atr_m5_on_s5 in backtest_retrace_atr.py). No lookahead.

- Simulates fills using live bid/ask (same fill model as live service / nofilter paper).
- No real OANDA orders. TP/SL/timeout tracked internally.
- Writes: is_paper=True, label='retrace_atr', account_id='paper-009-atr'
"""

import os, sys, time, signal, logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from dotenv import load_dotenv; load_dotenv()

from lib.oanda_adapter import OANDAAdapter
from lib.db import write_trade_direct

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [retrace-atr-paper] %(message)s",
)
logger = logging.getLogger("retrace-atr-paper")

# ── Strategy params (identical to nofilter paper + ATR gate) ──────────────────
THR        = 2.5
PEAK_BARS  = 44
TP_PIPS    = 20.0
SL_PIPS    = 30.0
HORIZON    = 600
COOLDOWN   = (PEAK_BARS + HORIZON) // 2
Z_WIN      = 6
MAD_WIN    = 2048
ATR_GATE   = 5.0            # M5-ATR(14) pips — the validated entry gate (skip low vol)
ATR_M5     = 60             # 60 S5 bars per M5 block
ATR_PERIOD = 14
WARMUP_N   = MAD_WIN + Z_WIN + 20
POLL_SECS  = 5

STRATEGY   = "post_shock_retrace"
LABEL      = "retrace_atr"
ACCOUNT_ID = "paper-009-atr"

PAIRS = {
    "GBP_JPY": {"pip": 0.01, "sp_gate": 4.00},
    "USD_JPY": {"pip": 0.01, "sp_gate": 2.10},
    "EUR_JPY": {"pip": 0.01, "sp_gate": 2.50},
    "AUD_JPY": {"pip": 0.01, "sp_gate": 2.30},
}

_shutdown = False
_trade_seq = 0  # monotonic counter for paper trade IDs


def _next_trade_id(pair: str) -> str:
    global _trade_seq
    _trade_seq += 1
    return f"paper_atr_{pair}_{_trade_seq}"


# ── Causal velocity z-score (mirrors live exactly) ───────────────────────────

class VelZScore:
    def __init__(self, pip: float):
        self._pip    = pip
        self._closes = deque(maxlen=Z_WIN + 1)
        self._vels   = deque(maxlen=MAD_WIN)

    def update(self, close: float) -> Optional[float]:
        self._closes.append(close)
        if len(self._closes) < Z_WIN + 1:
            return None
        vel = (self._closes[-1] - self._closes[0]) / self._pip
        self._vels.append(vel)
        if len(self._vels) < 50:
            return None
        arr = np.array(self._vels, dtype=np.float64)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad < 1e-6:
            return 0.0
        return float((vel - med) / (1.4826 * mad))

    def ready(self) -> bool:
        return len(self._vels) >= 50


# ── Causal M5-ATR(14), built incrementally from S5 bars ──────────────────────
# Mirrors compute_atr_m5_on_s5: aggregate S5 → M5 blocks (H=max,L=min,C=last),
# TR = max(H-L, |H-prevC|, |L-prevC|), ATR = simple mean of last 14 TRs, in pips.
# `value()` returns the ATR of the most recent COMPLETED M5 block (no lookahead).

class ATRM5:
    def __init__(self, pip: float):
        self._pip   = pip
        self._bucket = None          # current M5 bucket key (epoch // 300)
        self._h = self._l = self._c = None
        self._prev_close = None      # close of previous COMPLETED M5 block
        self._trs = deque(maxlen=ATR_PERIOD)
        self._atr_pips = None        # ATR of last completed block

    @staticmethod
    def _bucket_key(ts: str) -> int:
        # ts like '2026-06-19T07:55:00.000000000Z' → epoch seconds // 300
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return int(t.timestamp()) // (ATR_M5 * 5)

    def _close_block(self):
        if self._h is None:
            return
        pc = self._prev_close if self._prev_close is not None else self._c
        tr = max(self._h - self._l, abs(self._h - pc), abs(self._l - pc))
        self._trs.append(tr)
        if len(self._trs) >= ATR_PERIOD:
            self._atr_pips = float(np.mean(self._trs)) / self._pip
        self._prev_close = self._c

    def update(self, ts: str, high: float, low: float, close: float):
        try:
            key = self._bucket_key(ts)
        except Exception:
            return
        if self._bucket is None:
            self._bucket = key
            self._h, self._l, self._c = high, low, close
            return
        if key != self._bucket:
            # new M5 block started → finalize the previous one
            self._close_block()
            self._bucket = key
            self._h, self._l, self._c = high, low, close
        else:
            self._h = max(self._h, high)
            self._l = min(self._l, low)
            self._c = close

    def value(self) -> Optional[float]:
        return self._atr_pips


# ── Per-pair state ────────────────────────────────────────────────────────────

@dataclass
class PairState:
    pair:          str
    pip:           float
    sp_gate:       float
    status:        str   = "IDLE"
    shock_dir:     int   = 0
    monitor_bars:  int   = 0
    pos:           int   = 0
    entry_px:      float = 0.0
    tp_price:      float = 0.0
    sl_price:      float = 0.0
    trade_id:      str   = ""
    entry_time:    str   = ""
    position_bars: int   = 0
    mfe_pips:      float = 0.0
    mae_pips:      float = 0.0
    cooldown_bars: int   = 0
    last_s5_ts:    str   = ""
    n_atr_skips:   int   = 0


# ── Trade recording ───────────────────────────────────────────────────────────

def _record_close(st: PairState, exit_px: float, reason: str):
    pnl = (exit_px - st.entry_px) / st.pip * st.pos
    dur_h = 0.0
    if st.entry_time:
        try:
            t0 = datetime.fromisoformat(st.entry_time.replace("Z", "+00:00"))
            dur_h = (datetime.now(timezone.utc) - t0).total_seconds() / 3600
        except Exception:
            pass
    cap = pnl / st.mfe_pips if st.mfe_pips > 0 else 0.0
    exit_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sym = "🟢" if pnl >= 0 else "🔴"
    logger.info(
        f"{sym} {st.pair} CLOSE {reason}  pnl={pnl:+.1f}p  "
        f"mfe={st.mfe_pips:+.1f}  mae={st.mae_pips:+.1f}  id={st.trade_id}"
    )
    write_trade_direct(
        strategy=STRATEGY, pair=st.pair, account_id=ACCOUNT_ID,
        trade_id=st.trade_id, direction=st.pos,
        entry_price=st.entry_px, exit_price=exit_px,
        entry_time=st.entry_time, exit_time=exit_time,
        pnl_pips=pnl, exit_reason=reason, hours_held=dur_h,
        units=1, mfe_pips=st.mfe_pips, mae_pips=st.mae_pips,
        capture_ratio=cap, is_paper=True, label=LABEL,
    )


def _reset(st: PairState):
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


# ── Bar processor ─────────────────────────────────────────────────────────────

def _process_bar(st: PairState, close: float, bid_c: float, ask_c: float, atr_m5: Optional[float]):
    # Cooldown
    if st.cooldown_bars > 0:
        st.cooldown_bars -= 1
        return

    # ── IN_POSITION ──────────────────────────────────────────────────────────
    if st.status == "IN_POSITION":
        st.position_bars += 1

        unreal = (close - st.entry_px) / st.pip * st.pos
        if unreal > st.mfe_pips:
            st.mfe_pips = unreal
        if unreal < st.mae_pips:
            st.mae_pips = unreal

        if st.pos == -1 and bid_c <= st.tp_price:
            _record_close(st, st.tp_price, "tp"); _reset(st); return
        if st.pos == 1 and ask_c >= st.tp_price:
            _record_close(st, st.tp_price, "tp"); _reset(st); return

        if st.pos == -1 and ask_c >= st.sl_price:
            _record_close(st, st.sl_price, "sl"); _reset(st); return
        if st.pos == 1 and bid_c <= st.sl_price:
            _record_close(st, st.sl_price, "sl"); _reset(st); return

        if st.position_bars >= HORIZON:
            exit_px = bid_c if st.pos == -1 else ask_c
            _record_close(st, exit_px, "timeout"); _reset(st)
        return

    # ── MONITORING ───────────────────────────────────────────────────────────
    if st.status == "MONITORING":
        st.monitor_bars += 1
        if st.monitor_bars < PEAK_BARS:
            return

        # Spread gate
        spread = (ask_c - bid_c) / st.pip
        if spread > st.sp_gate:
            logger.info(f"{st.pair}: spread {spread:.2f}p > gate {st.sp_gate:.2f}p → skip")
            _reset(st)
            return

        # ── ATR ENTRY GATE — the only difference vs retrace_nofilter ──────────
        if atr_m5 is None or atr_m5 < ATR_GATE:
            st.n_atr_skips += 1
            logger.info(
                f"{st.pair}: ATR gate skip — M5-ATR(14)="
                f"{atr_m5 if atr_m5 is not None else float('nan'):.2f}p < {ATR_GATE:.1f}p "
                f"(skips={st.n_atr_skips})"
            )
            _reset(st)
            return

        # Enter (no Markov check)
        direction = -st.shock_dir
        fill_px   = bid_c if direction == -1 else ask_c
        tp        = fill_px + direction * TP_PIPS * st.pip
        sl        = fill_px - direction * SL_PIPS * st.pip
        tid       = _next_trade_id(st.pair)
        dir_str   = "LONG" if direction == 1 else "SHORT"
        logger.info(
            f"{st.pair} OPEN {dir_str} @ {fill_px:.3f}  "
            f"TP={tp:.3f} (+{TP_PIPS:.0f}p)  SL={sl:.3f} (-{SL_PIPS:.0f}p)  "
            f"spread={spread:.2f}p  ATR={atr_m5:.2f}p  id={tid}"
        )
        st.status        = "IN_POSITION"
        st.pos           = direction
        st.entry_px      = fill_px
        st.tp_price      = tp
        st.sl_price      = sl
        st.trade_id      = tid
        st.entry_time    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        st.position_bars = 0
        st.mfe_pips      = 0.0
        st.mae_pips      = 0.0
        return


# ── Market hours guard ────────────────────────────────────────────────────────

def _market_open() -> bool:
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


# ── Shock detection wrapper (called per bar in main loop) ─────────────────────

def _tick(st: PairState, zs: VelZScore, atr: ATRM5,
          ts: str, high: float, low: float, close: float, bid_c: float, ask_c: float):
    atr.update(ts, high, low, close)
    z = zs.update(close)

    if st.cooldown_bars > 0:
        st.cooldown_bars -= 1
        return

    if st.status == "IDLE":
        if z is not None and abs(z) > THR:
            st.shock_dir    = 1 if z > 0 else -1
            st.status       = "MONITORING"
            st.monitor_bars = 0
            logger.info(
                f"{st.pair}: shock detected z={z:+.2f}  dir={st.shock_dir:+d} → MONITORING"
            )
        return

    _process_bar(st, close, bid_c, ask_c, atr.value())


# ── Main ─────────────────────────────────────────────────────────────────────

def _signal_handler(sig, frame):
    global _shutdown
    logger.info(f"Signal {sig} → shutting down")
    _shutdown = True


def main():
    global _shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT,  _signal_handler)

    adapter  = OANDAAdapter(account_id=os.environ.get("OANDA_ACCOUNT_ID_009", ""))
    states   = {}
    zscorers = {}
    atrs     = {}

    for pair, cfg in PAIRS.items():
        states[pair]   = PairState(pair=pair, pip=cfg["pip"], sp_gate=cfg["sp_gate"])
        zscorers[pair] = VelZScore(pip=cfg["pip"])
        atrs[pair]     = ATRM5(pip=cfg["pip"])

    # Warmup: seed z-scorers AND ATR with WARMUP_N historical S5 bars
    logger.info(f"Warming up z-scorers + ATR ({WARMUP_N} S5 bars per pair) …")
    for pair, st in states.items():
        try:
            bars = adapter.get_candles(pair, count=WARMUP_N + 1, granularity="S5")
            completed = bars[:-1]
            for b in completed:
                zscorers[pair].update(b["close"])
                atrs[pair].update(b["timestamp"], b["high"], b["low"], b["close"])
            if completed:
                st.last_s5_ts = completed[-1]["timestamp"]
            logger.info(
                f"{pair}: warmup done  z_ready={zscorers[pair].ready()}  "
                f"atr_m5={atrs[pair].value()}  "
                f"last_s5={st.last_s5_ts[-16:] if st.last_s5_ts else '—'}"
            )
        except Exception as e:
            logger.error(f"{pair}: warmup failed: {e}")

    logger.info(
        f"Retrace ATR-gate paper (no Markov, ATR>={ATR_GATE}p) — {len(PAIRS)} pairs  "
        f"thr={THR}  peak={PEAK_BARS}  TP={TP_PIPS}  SL={SL_PIPS}  "
        f"horizon={HORIZON}  label={LABEL}"
    )

    while not _shutdown:
        if not _market_open():
            time.sleep(300)
            continue

        for pair, st in states.items():
            try:
                bars = adapter.get_candles(pair, count=15, granularity="S5")
                if not bars:
                    continue
                completed = bars[:-1]
                new_bars  = [b for b in completed if b["timestamp"] > st.last_s5_ts]
                for b in new_bars:
                    _tick(st, zscorers[pair], atrs[pair],
                          b["timestamp"], b["high"], b["low"], b["close"],
                          b["bid_c"], b["ask_c"])
                    st.last_s5_ts = b["timestamp"]
            except Exception as e:
                logger.error(f"{pair}: poll error: {e}")

        time.sleep(POLL_SECS)

    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
