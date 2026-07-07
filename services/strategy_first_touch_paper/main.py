#!/usr/bin/env python3
"""
First-Touch Low-Volume Reversion — PAPER
=========================================
Fade the FIRST touch (touches<=1) of a rolling H4 swing level, but ONLY when tick-volume at
the touch is LOW (vrel <= VREL_MAX). Exit at the first of: ATR target / ATR stop / time cap.
Contrarian, ~48h hold. Validated 2026-06-18 (research/experiments/touch_ladder/first_touch_v2.py):
loVol OOS +9.49p/trade, WR 54%, MC P(<=0)=0.018, temporal WF +ve all 3 calendar thirds, 7/12
pairs. hiVol fails (genuine break). Paper-only — no real orders, writes to trades.duckdb.

Exact-live-consistency (SOP): mid OHLC for signal + target/SL triggers, fixed per-pair spread
deducted at close (matches the backtest). One position per pair. All 12 pairs (no OOS-selection).
  Writes: is_paper=True, label='first_touch_lv', account_id='paper-ftlv'
"""
import os, sys, time, signal, logging, json
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from dotenv import load_dotenv; load_dotenv()

from lib.oanda_adapter import OANDAAdapter
from lib.db import write_trade_direct

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s [first-touch-paper] %(message)s",
)
logger = logging.getLogger("first-touch-paper")

# ── Validated config (IS-best, first_touch_v2.py) ────────────────────────────
L          = int(os.environ.get("FT_L", 25))        # swing-level lookback (H4 bars)
EPS_PIPS   = float(os.environ.get("FT_EPS", 12))    # touch tolerance (pips)
VW         = int(os.environ.get("FT_VW", 20))       # volume mean window
ATR_N      = 14
TGT_ATR    = float(os.environ.get("FT_TGT", 2.0))   # target = TGT_ATR * ATR(entry)
SL_ATR     = float(os.environ.get("FT_SL", 2.0))    # stop   = SL_ATR  * ATR(entry)
HCAP       = int(os.environ.get("FT_HCAP", 12))     # time cap (H4 bars ≈ 48h)
TOUCH_MAX  = 1                                       # first touch only
VREL_MAX   = float(os.environ.get("FT_VREL", 1.16)) # low-volume gate (IS-median)
GRAN       = "H4"
POLL_SECS  = int(os.environ.get("FT_POLL", 300))
BUF        = max(L, VW, ATR_N) + 30

STRATEGY   = "first_touch_reversion"
LABEL      = "first_touch_lv"
ACCOUNT_ID = "paper-ftlv"

PAIRS = {
    "USD_JPY":(0.01,2.1), "EUR_JPY":(0.01,2.5), "GBP_JPY":(0.01,4.0), "AUD_JPY":(0.01,2.3),
    "CAD_JPY":(0.01,2.6), "CHF_JPY":(0.01,3.0), "NZD_JPY":(0.01,3.0), "EUR_USD":(0.0001,1.7),
    "GBP_USD":(0.0001,2.4), "AUD_USD":(0.0001,1.6), "EUR_GBP":(0.0001,2.0), "NZD_USD":(0.0001,2.0),
}

_shutdown = False
_seq = 0


def _tid(pair):
    global _seq; _seq += 1
    return f"paper_ft_{pair}_{_seq}"


@dataclass
class PairState:
    pair: str; pip: float; spread: float
    bars: list                      # rolling completed H4 bars (dicts)
    last_ts: str = ""
    pos: int = 0; entry_px: float = 0.0; atr_e: float = 0.0
    tp: float = 0.0; sl: float = 0.0
    trade_id: str = ""; entry_time: str = ""
    pos_bars: int = 0; mfe: float = 0.0; mae: float = 0.0
    bars_seen: int = 0; n_trades: int = 0; total_pips: float = 0.0


def _atr(bars, n=ATR_N):
    """Wilder ATR over the buffer (mid OHLC)."""
    if len(bars) < n + 1: return 0.0
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]["high"], bars[i]["low"], bars[i-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    a = trs[0]
    alpha = 1.0 / n
    for tr in trs[1:]:
        a = a + alpha * (tr - a)
    return a


def _record_close(st, exit_px, reason):
    pnl = (exit_px - st.entry_px) / st.pip * st.pos - st.spread
    dur_h = 0.0
    if st.entry_time:
        try:
            t0 = datetime.fromisoformat(st.entry_time.replace("Z", "+00:00"))
            dur_h = (datetime.now(timezone.utc) - t0).total_seconds() / 3600
        except Exception:
            pass
    cap = pnl / st.mfe if st.mfe > 0 else 0.0
    exit_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sym = "🟢" if pnl >= 0 else "🔴"
    logger.info(f"{sym} {st.pair} CLOSE {reason} pnl={pnl:+.1f}p mfe={st.mfe:+.1f} mae={st.mae:+.1f} id={st.trade_id}")
    write_trade_direct(
        strategy=STRATEGY, pair=st.pair, account_id=ACCOUNT_ID, trade_id=st.trade_id,
        direction=st.pos, entry_price=st.entry_px, exit_price=exit_px,
        entry_time=st.entry_time, exit_time=exit_time, pnl_pips=pnl, exit_reason=reason,
        hours_held=dur_h, units=1, mfe_pips=st.mfe, mae_pips=st.mae,
        capture_ratio=cap, is_paper=True, label=LABEL,
    )
    st.n_trades += 1; st.total_pips += pnl
    st.pos = 0; st.trade_id = ""; st.pos_bars = 0; st.mfe = 0.0; st.mae = 0.0


def _on_new_bar(st, cur):
    """Process one freshly-completed H4 bar (mid OHLC + volume)."""
    st.bars.append(cur); st.bars_seen += 1
    if len(st.bars) > BUF: st.bars.pop(0)
    pip = st.pip

    # ── manage open position ────────────────────────────────────────────────
    if st.pos != 0:
        st.pos_bars += 1
        unreal = (cur["close"] - st.entry_px) / pip * st.pos
        st.mfe = max(st.mfe, unreal); st.mae = min(st.mae, unreal)
        if st.pos == -1:                       # short fade off resistance
            if cur["high"] >= st.sl: _record_close(st, st.sl, "sl"); return
            if cur["low"]  <= st.tp: _record_close(st, st.tp, "tp"); return
        else:                                  # long fade off support
            if cur["low"]  <= st.sl: _record_close(st, st.sl, "sl"); return
            if cur["high"] >= st.tp: _record_close(st, st.tp, "tp"); return
        if st.pos_bars >= HCAP:
            _record_close(st, cur["close"], "timecap"); return
        return

    # ── look for a fresh low-volume first touch ───────────────────────────────
    if len(st.bars) < max(L, VW) + 2:
        return
    win = st.bars[-(L+1):-1]                    # L bars before current
    prev = st.bars[-2]
    R = max(b["high"] for b in win); S = min(b["low"] for b in win)
    eps = EPS_PIPS * pip
    atr = _atr(st.bars)
    if atr <= 0: return
    volwin = st.bars[-(VW+1):-1]
    vmean = np.mean([b["volume"] for b in volwin]) if volwin else 0.0
    if vmean <= 0: return
    vrel = cur["volume"] / vmean
    if vrel > VREL_MAX:                          # low-volume gate
        return

    up = cur["high"] >= R - eps and prev["high"] < R - eps and cur["close"] <= R
    dn = cur["low"]  <= S + eps and prev["low"]  > S + eps and cur["close"] >= S
    if up:
        touches = sum(1 for b in win if b["high"] >= R - eps)
        if touches > TOUCH_MAX: return
        d = -1
    elif dn:
        touches = sum(1 for b in win if b["low"] <= S + eps)
        if touches > TOUCH_MAX: return
        d = +1
    else:
        return

    entry = cur["close"]
    tp = entry + d * TGT_ATR * atr               # toward reversion
    sl = entry - d * SL_ATR * atr                # beyond entry
    st.pos = d; st.entry_px = entry; st.atr_e = atr; st.tp = tp; st.sl = sl
    st.trade_id = _tid(st.pair)
    st.entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    st.pos_bars = 0; st.mfe = 0.0; st.mae = 0.0
    dstr = "SHORT" if d == -1 else "LONG"
    logger.info(f"{st.pair} OPEN {dstr} @ {entry:.5f} tp={tp:.5f} sl={sl:.5f} "
                f"atr={atr/pip:.1f}p vrel={vrel:.2f} touches={touches} id={st.trade_id}")


STATUS_PATH = "/data/logs/first_touch_status.json"
OOS_REF = "+9.5p/tr"


def _write_status(states):
    """Publish per-pair state for the dashboard Paper tab (mirrors fifo/tr/zr paper)."""
    strategies = []
    for pair, st in states.items():
        strategies.append({
            "label": LABEL, "pair": pair, "pos": st.pos,
            "entry_px": round(st.entry_px, 5) if st.pos else 0,
            "bar_count": st.bars_seen, "n_trades": st.n_trades,
            "total_pips": round(st.total_pips, 1), "oos_pd_ref": OOS_REF,
        })
    try:
        tmp = STATUS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"strategies": strategies,
                       "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, f)
        os.replace(tmp, STATUS_PATH)
    except Exception as e:
        logger.error(f"status write failed: {e}")


def _market_open():
    now = datetime.now(timezone.utc); wd = now.weekday(); h = now.hour
    if wd == 5: return False
    if wd == 6 and h < 21: return False
    if wd == 4 and h >= 21: return False
    return True


def _signal_handler(sig, frame):
    global _shutdown
    logger.info(f"Signal {sig} → shutting down"); _shutdown = True


def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    adapter = OANDAAdapter(account_id=os.environ.get("OANDA_ACCOUNT_ID_009", ""))
    states = {}
    logger.info(f"Warmup ({BUF} H4 bars/pair) …")
    for pair, (pip, sp) in PAIRS.items():
        st = PairState(pair=pair, pip=pip, spread=sp, bars=[])
        try:
            bars = adapter.get_candles(pair, count=BUF + 1, granularity=GRAN)
            completed = bars[:-1]
            st.bars = [dict(timestamp=b["timestamp"], open=b["open"], high=b["high"],
                            low=b["low"], close=b["close"], volume=b["volume"]) for b in completed]
            if st.bars: st.last_ts = st.bars[-1]["timestamp"]
            logger.info(f"{pair}: warmup {len(st.bars)} bars  last={st.last_ts[-16:] if st.last_ts else '—'}")
        except Exception as e:
            logger.error(f"{pair}: warmup failed: {e}")
        states[pair] = st

    logger.info(f"First-touch LV paper — {len(PAIRS)} pairs  L={L} eps={EPS_PIPS} vrel≤{VREL_MAX} "
                f"tgt={TGT_ATR}·ATR sl={SL_ATR}·ATR cap={HCAP}bars label={LABEL}")
    _write_status(states)

    while not _shutdown:
        if not _market_open():
            _write_status(states); time.sleep(POLL_SECS); continue
        for pair, st in states.items():
            try:
                bars = adapter.get_candles(pair, count=5, granularity=GRAN)
                if not bars: continue
                for b in bars[:-1]:               # completed bars only
                    if b["timestamp"] > st.last_ts:
                        _on_new_bar(st, dict(timestamp=b["timestamp"], open=b["open"], high=b["high"],
                                             low=b["low"], close=b["close"], volume=b["volume"]))
                        st.last_ts = b["timestamp"]
            except Exception as e:
                logger.error(f"{pair}: poll error: {e}")
        _write_status(states)
        time.sleep(POLL_SECS)

    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
