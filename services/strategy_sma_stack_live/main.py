#!/usr/bin/env python3
"""SMA stack-alignment + novelty live service — H17 mixed-exit deployment on acct 010.

ARCHITECTURE.  Single source = OANDA S5 feed. All higher TFs are constructed
in-strategy by aggregating the S5 stream (open=first, high=max, low=min,
close=last of N consecutive S5 bars). Matches H17d's `bin_resample` exactly.

ENTRY (same for all pairs).  Long fires when on BOTH TF1 and TF2:
  (a) last 3 completed bar closes monotonically rise
  (b) c[-1] > SMA_sm > SMA_md > SMA_lg
  (c) each SMA rose over last 2 steps
  AND the alignment is NEWLY FORMED (previous TF bar not aligned).
Short is the mirror.

EXIT (per-pair — H17f/H17g sweep winners):

  EUR_JPY  S5/M2/M10  SMA 5/15/35  TP=20p  + hard SL=200p at broker
           OOS +18.31 p/d  DD -202  WR 98% (118 trades / 299d)

  EUR_USD  S5/M1/M5   SMA 5/15/35  TP=30p (kept) + PSAR(TF1) trail
           af_start=0.020 af_max=0.10  activate=20p MFE
           OOS +10.92 p/d  DD -13   WR 98% (85 trades / 478d)

  GBP_JPY  S5/M30/H1  SMA 5/15/35  no TP  + PSAR(TF1) trail
           af_start=0.010 af_max=0.10  activate=20p MFE
           OOS +10.71 p/d  DD -82   WR 77% (39 trades / 303d)

  GBP_USD  S5/S30/M1  SMA 7/22/50  no TP   + PSAR(S30) trail + 200p outer SL
           K=0 baseline OOS +11.63 p/d  DD -118 (SL never fired in 286d backtest)

  USD_JPY  S5/M1/M5   SMA 5/10/22  TP=15p  + 200p outer-fence SL (free insurance)
           K=0 baseline OOS +10.18 p/d  DD -72  (SL never fired in 294d backtest)

SIZING.  units = max(1, round(balance × 1.25)), refreshed hourly, cap MAX_UNITS=500. Self-scaling: the
200p fence then risks a CONSTANT ~1.6% NAV/trade (JPY) / ~2.6% (USD) at any balance,
so no hard unit cap is needed (MAX_UNITS is just a sanity ceiling). See the pip-value
note below — earlier docs overstated fence risk ~110×.
"""
import os
import sys
import time
import logging
import signal as _signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lib.oanda_adapter import OANDAAdapter
from lib.db import write_trade_direct

logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO')),
    format='%(asctime)s %(levelname)s [stack_live] %(message)s',
)
log = logging.getLogger('stack_live')

def tg(msg):
    """Best-effort Telegram alert; never raises (must not break trading)."""
    try:
        import requests
        tok = os.getenv("TELEGRAM_BOT_TOKEN", ""); chat = os.getenv("TELEGRAM_CHAT_ID", "")
        if tok and chat:
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id": chat, "text": msg}, timeout=5)
    except Exception:
        pass

STRATEGY      = os.environ.get('LABEL_PREFIX', 'sma_stack')
ACCOUNT_ENV   = os.environ.get('ACCOUNT_ENV', 'OANDA_ACCOUNT_ID_010')
EXIT_PROFILE  = os.environ.get('EXIT_PROFILE', 'default')  # 'fade' = TP150/SL150 hard-SL all pairs, no PSAR
ACCT_LABEL    = ACCOUNT_ENV.rsplit('_', 1)[-1]  # '010' / '001' — for logs + telegram
STRAT_DISPLAY = 'SMA-Stack' if STRATEGY == 'sma_stack' else STRATEGY  # keep 010's telegram text byte-identical
UNITS_PER_DOLLAR     = 1.25  # 2026-07-01: user-directed 1.25×balance for the VPS move. Prior: 1.3 (2026-06-21), 0.5 (2026-06-17).
# 2026-06-17 PIP-VALUE CORRECTION: earlier docs used ~$0.0069/unit/pip for JPY
# pairs — that is 110× TOO HIGH. Ground truth from a live 009 fill is
# $0.0000626/unit/pip (1 pip = 0.01 JPY ÷ USDJPY≈160). So the real 200p-fence
# risk is TINY: at 0.5×balance (≈38u @ $75 NAV) ≈ $0.45/trade ≈ 0.6% NAV;
# even uncapped at 113u it was ≈ $1.4 ≈ 1.9% NAV — never the "$39/$147/2× NAV"
# the old comment claimed. With units = k×balance the fence risk is a CONSTANT
# fraction of NAV at every balance, so the hard cap is redundant — MAX_UNITS is
# now just a far-away sanity ceiling. Keep the 200p fence WIDE: h17f shows it
# almost never fires (TP/PSAR exits dominate) and tightening it chops the edge
# on the PSAR trend pairs (see h17f_catastrophe_sl / RISK_REVIEW_2026-06-11).
MAX_UNITS            = 500
BALANCE_REFRESH_SECS = 3600
POLL_SECS     = 5
LB_CLOSE      = 3       # close-monotonicity lookback
LB_SLOPE      = 2       # slope lookback
TF_HISTORY    = 80      # TF closes kept (enough for SMA(50))

# Live cached unit count
_units = 5
_last_balance_refresh = 0.0


@dataclass
class PairCfg:
    pair: str
    pip: float
    # TF aggregation
    n1: int      # base bars per TF1 bar
    n2: int      # base bars per TF2 bar
    sma_sm: int
    sma_md: int
    sma_lg: int
    tp_pips: float           # broker TP at this distance (0.0 = no broker TP)
    # Exit policy
    exit_mode: str           # 'sl' or 'psar'
    sl_pips: float = 0.0          # hard SL distance (for both 'sl' mode and outer-fence)
    psar_af_start: float = 0.0    # for 'psar' mode
    psar_af_max: float = 0.0
    psar_activate_pips: float = 0.0
    psar_replaces_tp: bool = False  # if True, no broker TP, PSAR is sole exit


CONFIGS = [
    # EUR_JPY  S5/M2/M10  SMA 5/15/35  TP=20p + hard SL=200p
    PairCfg('EUR_JPY', 0.01,   n1=24, n2=120, sma_sm=5,  sma_md=15, sma_lg=35,
            tp_pips=20.0, exit_mode='sl', sl_pips=200.0),
    # EUR_USD  S5/M1/M5   SMA 5/15/35  TP=30p + PSAR(M1) trail af=0.020 act=20p
    PairCfg('EUR_USD', 0.0001, n1=12, n2=60,  sma_sm=5,  sma_md=15, sma_lg=35,
            tp_pips=30.0, exit_mode='psar',
            psar_af_start=0.020, psar_af_max=0.10, psar_activate_pips=20.0,
            psar_replaces_tp=False,
            # Also place a wide 200p outer-fence SL at broker to cap catastrophes
            sl_pips=200.0),
    # GBP_JPY  DROPPED 2026-06-17: no edge at ANY TF speed (M30/H1..M1/M5) or exit
    # (PSAR/TP20) — every config IN-SAMPLE negative (IS -0.5 to -14 p/d) over 5.8mo+WF.
    # 1 closed live trade (0p). gbpjpy_h1h4_psar.py / gbpjpy_fast.py. Open pos 804
    # closed at market on drop. Keep the 4 working pairs (EUR_JPY/EUR_USD/USD_JPY/GBP_USD).
    # GBP_USD  S5/S30/M1  SMA 7/22/50  no TP + PSAR(S30) trail af=0.020 act=20p
    # Switched to PSAR-only on 2026-06-07: h17f_catastrophe_sl sweep showed
    # SL=200p costs ~7 p/d OOS vs no-SL on this pair (best non-SL OOS=+11.6 p/d
    # vs SL=200p OOS=+4.5 p/d). Following the GBP_JPY pattern: 200p outer fence
    # only, PSAR is the active exit. n1 was 6 bars (S30); kept for psar update.
    PairCfg('GBP_USD', 0.0001, n1=6, n2=12,  sma_sm=7,  sma_md=22, sma_lg=50,
            tp_pips=0.0, exit_mode='psar',
            psar_af_start=0.020, psar_af_max=0.10, psar_activate_pips=20.0,
            psar_replaces_tp=True,
            sl_pips=200.0),
    # USD_JPY  S5/M1/M5   SMA 5/10/22  TP=15p + outer-fence SL=200p
    PairCfg('USD_JPY', 0.01,   n1=12, n2=60,  sma_sm=5,  sma_md=10, sma_lg=22,
            tp_pips=15.0, exit_mode='sl', sl_pips=200.0),
]

if EXIT_PROFILE == 'fade':
    # Fade-exit live variant — research/experiments/conservative_010/baseline_fade_exit.py.
    # The market mean-reverts at these horizons, so a fixed TP + a hard bounded SL beats the
    # PSAR trail / wide 200p fence: it halts the bleed and eliminates the -2000p catastrophic
    # tail (worst single trade ~= SL). It does NOT manufacture an edge (entry is the binding
    # constraint); this is a risk-capped live A/B on account 001 vs 010's PSAR/fence exit.
    for _c in CONFIGS:
        _c.tp_pips = 150.0
        _c.sl_pips = 150.0
        _c.exit_mode = 'sl'
        _c.psar_replaces_tp = False


@dataclass
class TFBar:
    open_:  Optional[float] = None
    high:   Optional[float] = None
    low:    Optional[float] = None
    close:  Optional[float] = None

    def update(self, s5_o, s5_h, s5_l, s5_c):
        if self.open_ is None:
            self.open_ = s5_o; self.high = s5_h; self.low = s5_l
        else:
            if s5_h > self.high: self.high = s5_h
            if s5_l < self.low:  self.low  = s5_l
        self.close = s5_c

    def reset(self):
        self.open_ = self.high = self.low = self.close = None


@dataclass
class PSARState:
    """Wilder PSAR on TF1 completed bars."""
    initialized: bool = False
    direction:   int  = 0
    sar:    float = 0.0
    ep:     float = 0.0
    af:     float = 0.0
    prev_high1: float = 0.0
    prev_low1:  float = 0.0
    prev_high2: float = 0.0
    prev_low2:  float = 0.0


@dataclass
class TFState:
    n_per_bar: int
    history: deque = field(default_factory=lambda: deque(maxlen=TF_HISTORY))
    high_hist: deque = field(default_factory=lambda: deque(maxlen=TF_HISTORY))
    low_hist:  deque = field(default_factory=lambda: deque(maxlen=TF_HISTORY))
    in_progress: TFBar = field(default_factory=TFBar)
    s5_count: int = 0

    was_aligned_long:  bool = False
    was_aligned_short: bool = False
    is_aligned_long:   bool = False
    is_aligned_short:  bool = False
    novelty_long:      bool = False
    novelty_short:     bool = False


@dataclass
class PairState:
    pair: str
    cfg:  PairCfg
    tf1:  TFState = None
    tf2:  TFState = None
    psar: PSARState = field(default_factory=PSARState)
    last_s5_ts:    str = ''
    pos_dir:       int = 0
    entry_px:      float = 0.0
    entry_time_iso:str = ''
    open_trade_id: Optional[str] = None
    bar_count:     int = 0
    mfe_pips:      float = 0.0
    psar_armed:    bool  = False
    # Real fill price captured when WE close the trade (PSAR exit). The broker's
    # close_trade() returns the actual fill; the reconciler prefers this over the
    # frequently-None averageClosePrice, so PSAR exits no longer collapse to
    # exit_px=entry / 'closed_unknown' / pnl=0.
    pending_exit_px:     Optional[float] = None
    pending_exit_reason: str = ''


def make_state(cfg: PairCfg) -> PairState:
    st = PairState(pair=cfg.pair, cfg=cfg)
    st.tf1 = TFState(n_per_bar=cfg.n1)
    st.tf2 = TFState(n_per_bar=cfg.n2)
    return st


def sma_n(deq, n):
    if len(deq) < n: return None
    arr = list(deq)[-n:]
    return sum(arr) / n


def check_stack_alignment(deq, sma_sm_n, sma_md_n, sma_lg_n):
    if len(deq) < max(sma_lg_n + LB_SLOPE + 1, LB_CLOSE + 1):
        return False, False
    closes = list(deq)

    def sma_at(arr, n, offset):
        idx_end = len(arr) - offset
        idx_start = idx_end - n
        if idx_start < 0: return None
        return sum(arr[idx_start:idx_end]) / n

    cur_sm = sma_at(closes, sma_sm_n, 0)
    cur_md = sma_at(closes, sma_md_n, 0)
    cur_lg = sma_at(closes, sma_lg_n, 0)
    if cur_sm is None or cur_md is None or cur_lg is None:
        return False, False

    c1 = closes[-1]; c2 = closes[-2]; c3 = closes[-3]

    mono_up = (c1 > c2 > c3)
    stacked_up = (c1 > cur_sm > cur_md > cur_lg)
    slopes_up = True
    for n_sma in (sma_sm_n, sma_md_n, sma_lg_n):
        for k in range(LB_SLOPE):
            cur_v  = sma_at(closes, n_sma, k)
            prev_v = sma_at(closes, n_sma, k+1)
            if cur_v is None or prev_v is None or not (cur_v > prev_v):
                slopes_up = False; break
        if not slopes_up: break
    aligned_long = mono_up and stacked_up and slopes_up

    mono_dn = (c1 < c2 < c3)
    stacked_dn = (c1 < cur_sm < cur_md < cur_lg)
    slopes_dn = True
    for n_sma in (sma_sm_n, sma_md_n, sma_lg_n):
        for k in range(LB_SLOPE):
            cur_v  = sma_at(closes, n_sma, k)
            prev_v = sma_at(closes, n_sma, k+1)
            if cur_v is None or prev_v is None or not (cur_v < prev_v):
                slopes_dn = False; break
        if not slopes_dn: break
    aligned_short = mono_dn and stacked_dn and slopes_dn

    return aligned_long, aligned_short


def on_tf_bar_emit(tfs: TFState, cfg: PairCfg):
    n_sm = cfg.sma_sm; n_md = cfg.sma_md; n_lg = cfg.sma_lg
    if len(tfs.history) < max(n_lg + LB_SLOPE + 1, LB_CLOSE + 1):
        tfs.is_aligned_long = False; tfs.is_aligned_short = False
        tfs.novelty_long = False; tfs.novelty_short = False
        return False
    aligned_long, aligned_short = check_stack_alignment(tfs.history, n_sm, n_md, n_lg)
    tfs.novelty_long  = aligned_long  and not tfs.was_aligned_long
    tfs.novelty_short = aligned_short and not tfs.was_aligned_short
    tfs.is_aligned_long  = aligned_long
    tfs.is_aligned_short = aligned_short
    tfs.was_aligned_long  = aligned_long
    tfs.was_aligned_short = aligned_short
    return True


def update_psar(state: PSARState, h: float, l: float, cfg: PairCfg):
    """Wilder PSAR — incremental update on a new completed TF1 bar."""
    if not state.initialized:
        if state.prev_high1 == 0 and state.prev_low1 == 0:
            state.prev_high1 = h; state.prev_low1 = l; return
        if h > state.prev_high1:
            state.direction = 1; state.ep = h; state.sar = state.prev_low1
        else:
            state.direction = -1; state.ep = l; state.sar = state.prev_high1
        state.af = cfg.psar_af_start
        state.prev_high2 = state.prev_high1; state.prev_low2 = state.prev_low1
        state.prev_high1 = h;                state.prev_low1  = l
        state.initialized = True
        return
    new_sar = state.sar + state.af * (state.ep - state.sar)
    if state.direction == 1:
        new_sar = min(new_sar, state.prev_low1, state.prev_low2)
        if l < new_sar:
            state.direction = -1
            state.sar = state.ep; state.ep = l
            state.af = cfg.psar_af_start
        else:
            state.sar = new_sar
            if h > state.ep:
                state.ep = h
                state.af = min(state.af + cfg.psar_af_start, cfg.psar_af_max)
    else:
        new_sar = max(new_sar, state.prev_high1, state.prev_high2)
        if h > new_sar:
            state.direction = 1
            state.sar = state.ep; state.ep = h
            state.af = cfg.psar_af_start
        else:
            state.sar = new_sar
            if l < state.ep:
                state.ep = l
                state.af = min(state.af + cfg.psar_af_start, cfg.psar_af_max)
    state.prev_high2 = state.prev_high1; state.prev_low2 = state.prev_low1
    state.prev_high1 = h;                state.prev_low1  = l


def _refresh_units(adapter: OANDAAdapter):
    global _units, _last_balance_refresh
    _last_balance_refresh = time.time()
    try:
        info = adapter.get_account_summary()
        if info is not None:
            flat = int(os.environ.get("FLAT_UNITS", "0"))   # >0 = hard flat cap (de-risk override)
            new_units = flat if flat > 0 else min(MAX_UNITS, max(1, round(info.balance * UNITS_PER_DOLLAR)))
            if new_units != _units:
                log.info(f"Units updated: ${info.balance:.2f} × {UNITS_PER_DOLLAR} "
                         f"→ {new_units}u/trade (was {_units}u)")
            else:
                log.info(f"Balance refresh: ${info.balance:.2f} → {new_units}u/trade")
            _units = new_units
    except Exception as e:
        log.warning(f"Balance refresh failed: {e} — keeping {_units}u")


_shutdown = False
def _on_sig(_sig, _frame):
    global _shutdown
    _shutdown = True
    log.info('SIGTERM received')

_signal.signal(_signal.SIGTERM, _on_sig)
_signal.signal(_signal.SIGINT,  _on_sig)


def process_s5_bar(adapter: OANDAAdapter, st: PairState, s5: dict, replay: bool = False):
    """Process one completed S5 bar.
    If replay=True (warmup), updates state but does NOT place orders."""
    cfg = st.cfg
    o = s5['open']; h = s5['high']; l = s5['low']; c = s5['close']
    st.bar_count += 1

    # Update both in-progress TF bars
    for tfs in (st.tf1, st.tf2):
        tfs.in_progress.update(o, h, l, c)
        tfs.s5_count += 1

    emitted_tf1 = False
    if st.tf1.s5_count >= cfg.n1:
        ip = st.tf1.in_progress
        st.tf1.history.append(ip.close)
        st.tf1.high_hist.append(ip.high)
        st.tf1.low_hist.append(ip.low)
        # Update PSAR on TF1 bar emit (only relevant for psar exit_mode)
        if cfg.exit_mode == 'psar':
            update_psar(st.psar, ip.high, ip.low, cfg)
        st.tf1.in_progress.reset()
        st.tf1.s5_count = 0
        emitted_tf1 = True
    emitted_tf2 = False
    if st.tf2.s5_count >= cfg.n2:
        ip = st.tf2.in_progress
        st.tf2.history.append(ip.close)
        st.tf2.high_hist.append(ip.high)
        st.tf2.low_hist.append(ip.low)
        st.tf2.in_progress.reset()
        st.tf2.s5_count = 0
        emitted_tf2 = True

    if emitted_tf1:
        on_tf_bar_emit(st.tf1, cfg)
    if emitted_tf2:
        on_tf_bar_emit(st.tf2, cfg)

    # Position management — only when not in warmup replay
    if replay:
        return

    # Track MFE on open trade; check PSAR exit if applicable
    if st.pos_dir != 0:
        cur_pip = (c - st.entry_px) / cfg.pip * st.pos_dir
        if cur_pip > st.mfe_pips:
            st.mfe_pips = cur_pip
        if not st.psar_armed and cfg.exit_mode == 'psar' and st.mfe_pips >= cfg.psar_activate_pips:
            st.psar_armed = True
            log.info(f"{cfg.pair} PSAR armed (MFE={st.mfe_pips:+.1f}p)")
        # PSAR exit check (if armed)
        if cfg.exit_mode == 'psar' and st.psar_armed and st.psar.initialized:
            psar_val = st.psar.sar
            close_now = c
            trigger = (st.pos_dir == 1 and close_now < psar_val) or \
                      (st.pos_dir == -1 and close_now > psar_val)
            if trigger:
                log.info(f"{cfg.pair} PSAR EXIT pos={st.pos_dir:+d} close={close_now:.5f} psar={psar_val:.5f}")
                if st.open_trade_id:
                    res = adapter.close_trade(st.open_trade_id)
                    # Capture the AUTHORITATIVE close fill from the broker so the
                    # reconciler records real P&L (not the entry-price fallback).
                    if res.success and res.fill_price:
                        st.pending_exit_px = float(res.fill_price)
                        st.pending_exit_reason = 'psar_trail'
                    else:
                        # Close fill unavailable → use the trigger price (the SAR
                        # cross), still far better than entry-price/0-pnl.
                        st.pending_exit_px = close_now
                        st.pending_exit_reason = 'psar_trail_est'
                # Reconciler will write the trade record on next cycle

    # Entry check: only when flat AND both TFs in joint novelty
    if st.pos_dir != 0:
        return
    if not (emitted_tf1 or emitted_tf2):
        return
    new_dir = 0
    if st.tf1.novelty_long and st.tf2.novelty_long:
        new_dir = 1
    elif st.tf1.novelty_short and st.tf2.novelty_short:
        new_dir = -1
    if new_dir == 0:
        return

    # ── BROKER-TRUTH FLIP GUARD (fixes the ~-366p MARKET_ORDER flip bleed) ──
    # The strategy enters only when it believes it is flat (st.pos_dir==0). But a
    # state desync (mis-recorded/missed close, restart race) can leave the broker
    # holding a position while state reads flat. Placing this entry would FIFO-net
    # the live position at a loss (the position-flip bug). The broker is the source
    # of truth: if ANY position is open on this pair, re-adopt it and skip the entry.
    try:
        open_now = [t for t in adapter.get_open_trades() if t.instrument == cfg.pair]
    except Exception as e:
        log.warning(f"{cfg.pair}: open-trade check failed ({e}) — skipping entry this bar")
        return
    if open_now:
        t = open_now[0]
        st.pos_dir = 1 if t.units > 0 else -1
        st.open_trade_id = t.trade_id
        if st.entry_px == 0.0:
            st.entry_px = float(t.entry_price)
        log.warning(f"{cfg.pair}: broker has OPEN pos {t.units:+}u (id={t.trade_id}) but state was flat "
                    f"— FLIP GUARD re-adopted, skipped {new_dir:+d} entry")
        return

    # Build broker-side brackets
    entry_px = c
    tp_price = None
    if cfg.tp_pips > 0:
        tp_price = entry_px + new_dir * cfg.tp_pips * cfg.pip
    sl_price = None
    if cfg.sl_pips > 0:
        sl_price = entry_px - new_dir * cfg.sl_pips * cfg.pip   # opposite side

    log.info(f"{cfg.pair} ENTRY {new_dir:+d} @ {entry_px:.5f}  "
             f"TP={tp_price}  SL={sl_price}  exit_mode={cfg.exit_mode}  units={_units}")

    res = adapter.place_market_order(
        cfg.pair,
        units=new_dir * _units,
        tp_price=tp_price,
        sl_price=sl_price,
    )
    if not res.success:
        log.warning(f"{cfg.pair} order failed: {res.error or res.cancel_reason}")
        return

    st.pos_dir = new_dir
    st.entry_px = float(res.fill_price)
    st.entry_time_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    st.open_trade_id = res.trade_id
    st.mfe_pips = 0.0
    st.psar_armed = False
    log.info(f"{cfg.pair} FILLED {new_dir:+d} @ {st.entry_px:.5f}  trade_id={st.open_trade_id}")
    tg(f"{'🟢 LONG' if new_dir>0 else '🔴 SHORT'} {STRAT_DISPLAY} {ACCT_LABEL} {cfg.pair} {_units}u @ {st.entry_px:.5f}")


def reconcile_open_trade(adapter: OANDAAdapter, st: PairState):
    if st.pos_dir == 0:
        return
    trades = adapter.get_open_trades()
    open_ids = {t.trade_id for t in trades if t.instrument == st.cfg.pair}
    if st.open_trade_id and st.open_trade_id in open_ids:
        return
    # Trade closed at broker — try to read the real close price / reason.
    # TradeInfo exposes close_price + realizedPL (not exit_price). For a
    # closed trade, t.price is sometimes None on OANDA → swallow that
    # quietly and fall back to TP estimate.
    # Read the REAL broker close fill. averageClosePrice can lag a beat after the
    # close, so retry a few times. NEVER assume a TP — that fabricated +TP winners
    # for every loser whose first read hiccupped (the 100%-WR / dashboard-lie bug).
    info = adapter.get_trade_details(st.open_trade_id) if st.open_trade_id else None
    for _attempt in range(4):
        if info is not None and getattr(info, 'close_price', None) is not None:
            break
        time.sleep(1.0)
        info = adapter.get_trade_details(st.open_trade_id) if st.open_trade_id else None
    exit_px = None
    exit_reason = 'closed'
    if st.pending_exit_px is not None:
        # We closed it ourselves (PSAR exit) and captured the real fill — trust it.
        exit_px = st.pending_exit_px
        exit_reason = st.pending_exit_reason or 'closed'
    elif info is not None and getattr(info, 'close_price', None) is not None:
        exit_px = float(info.close_price)          # real broker fill: TP, SL, or trail
        exit_reason = 'closed_broker'
    else:
        # Could not read the fill. RE-VERIFY before assuming a close: a transient incomplete
        # read of get_open_trades() can momentarily drop the trade, making us think it closed
        # while it is still open — that produced false "closed" Telegram alerts + phantom DB rows
        # while the position stayed open (the FLIP GUARD then re-adopted it). Re-query a few
        # times; if it reappears, it never closed — bail with NO state change, alert, or DB write.
        for _rv in range(3):
            time.sleep(1.0)
            still = {t.trade_id for t in adapter.get_open_trades() if t.instrument == st.cfg.pair}
            if st.open_trade_id in still:
                log.info(f"{st.cfg.pair}: transient read — trade {st.open_trade_id} still OPEN on "
                         f"re-verify; NOT closing (false-close suppressed)")
                return
        # Confirmed gone, but fill price unreadable — it really closed; record reason only with no
        # fabricated pnl, and (per the gate below) no alert/DB-write. Backfill from OANDA later.
        exit_px = st.entry_px
        exit_reason = 'closed_unknown'
        log.warning(f"{st.cfg.pair}: close CONFIRMED (trade {st.open_trade_id} gone after re-verify) but "
                    f"fill price unavailable — closed_unknown, NO alert/DB-write (backfill later)")
    pnl_pips = (exit_px - st.entry_px) / st.cfg.pip * st.pos_dir
    # Real hold time (was hardcoded 0.0h).
    hours_held = 0.0
    if st.entry_time_iso:
        try:
            t_in = datetime.strptime(st.entry_time_iso, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
            hours_held = max(0.0, (datetime.now(timezone.utc) - t_in).total_seconds() / 3600.0)
        except Exception:
            hours_held = 0.0
    log.info(f"{st.cfg.pair} CLOSED {st.pos_dir:+d}  pnl={pnl_pips:+.1f}p  "
             f"reason={exit_reason}  exit_px={exit_px:.5f}")
    # Only alert + record a CONFIRMED close (real fill known). closed_unknown means the trade is
    # gone but the fill was unreadable — no Telegram alert and no DB write (avoids the false alerts
    # and the phantom pnl=0 rows that also dedup-block the real backfill). Backfill from OANDA later.
    if exit_reason != 'closed_unknown':
        tg(f"{'🟢' if pnl_pips>=0 else '🔴'} SMA-Stack 010 {st.cfg.pair} CLOSE {pnl_pips:+.1f}p ({exit_reason})")
        try:
            write_trade_direct(
                strategy=STRATEGY,
                pair=st.cfg.pair,
                account_id=os.environ.get(ACCOUNT_ENV, '010'),
                # Account-prefix the trade_id so it can't collide with the same OANDA
                # short-id used on a different account (write_trade_direct dedups on trade_id alone).
                trade_id=(f"010_{st.open_trade_id}" if st.open_trade_id
                          else f"010_{st.cfg.pair}_{st.entry_time_iso}"),
                direction=st.pos_dir,
                entry_price=st.entry_px,
                exit_price=exit_px,
                entry_time=st.entry_time_iso,
                exit_time=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                pnl_pips=pnl_pips,
                exit_reason=exit_reason,
                hours_held=hours_held,
                units=_units,
                mfe_pips=st.mfe_pips, mae_pips=0.0,
                capture_ratio=1.0,
                is_paper=False,
                label=f'{STRATEGY}_{st.cfg.pair}',
            )
        except Exception as e:
            log.error(f"{st.cfg.pair}: DB write failed ({e}); resetting state anyway")
    # Always reset state — the position is confirmed gone at the broker (reconcile + re-verify).
    st.pos_dir = 0
    st.entry_px = 0.0
    st.open_trade_id = None
    st.mfe_pips = 0.0
    st.psar_armed = False
    st.pending_exit_px = None
    st.pending_exit_reason = ''


def _tf_minutes_to_granularity(minutes: float) -> str:
    """Map a TF in minutes to the closest OANDA-supported granularity for warmup."""
    if minutes < 1:    return 'S30'      # 0.5 min
    if minutes == 1:   return 'M1'
    if minutes == 2:   return 'M2'
    if minutes == 5:   return 'M5'
    if minutes == 10:  return 'M10'
    if minutes == 15:  return 'M15'
    if minutes == 30:  return 'M30'
    if minutes == 60:  return 'H1'
    # Fallback to nearest known
    for v, g in [(1,'M1'),(2,'M2'),(5,'M5'),(10,'M10'),(15,'M15'),(30,'M30'),(60,'H1')]:
        if minutes <= v: return g
    return 'H1'


def warmup(adapter: OANDAAdapter, st: PairState):
    """Two-phase warmup:
      1. Pull native TF1 and TF2 candles (clock-aligned OANDA bars) to seed
         the history deques. Enough bars to satisfy SMA(sma_lg) + lookback margin.
      2. PSAR seeded from TF1 high/low history during phase 1.
      3. last_s5_ts initialised from the latest S5 candle so the polling loop
         picks up from "now" (in-strategy aggregation starts fresh at runtime).
    """
    cfg = st.cfg
    tf1_min = cfg.n1 * 5 / 60.0    # minutes per TF1 bar
    tf2_min = cfg.n2 * 5 / 60.0    # minutes per TF2 bar
    g1 = _tf_minutes_to_granularity(tf1_min)
    g2 = _tf_minutes_to_granularity(tf2_min)
    need = cfg.sma_lg + LB_SLOPE + 30   # SMA window + lookback + margin

    bars1 = adapter.get_candles(cfg.pair, count=need, granularity=g1)
    bars2 = adapter.get_candles(cfg.pair, count=need, granularity=g2)
    if not bars1 or not bars2:
        log.warning(f"{cfg.pair}: warmup native TF fetch failed (g1={g1} g2={g2})")
        return

    # Seed TF1 history + PSAR
    for b in bars1[:-1]:
        st.tf1.history.append(b['close'])
        st.tf1.high_hist.append(b['high'])
        st.tf1.low_hist.append(b['low'])
        if cfg.exit_mode == 'psar':
            update_psar(st.psar, b['high'], b['low'], cfg)
    # Seed TF2 history
    for b in bars2[:-1]:
        st.tf2.history.append(b['close'])
        st.tf2.high_hist.append(b['high'])
        st.tf2.low_hist.append(b['low'])

    # Compute current alignment state from seeded history
    on_tf_bar_emit(st.tf1, cfg)
    on_tf_bar_emit(st.tf2, cfg)

    # Initialise last_s5_ts so the polling loop only picks up bars from now-on
    recent_s5 = adapter.get_candles(cfg.pair, count=3, granularity='S5')
    if recent_s5 and len(recent_s5) >= 2:
        st.last_s5_ts = recent_s5[-2]['timestamp']

    # Adopt any broker position left over from a previous container instance.
    # Without this, an orphan position will TP/SL on the broker but never get
    # recorded in our DB (reconcile gates on pos_dir != 0, which stays 0 forever
    # if we never adopt). Pattern copied from strategy_sma_live.
    try:
        open_trades = adapter.get_open_trades()
        pair_trades = [t for t in open_trades if t.instrument == cfg.pair]
        if pair_trades:
            t = pair_trades[0]
            st.pos_dir = 1 if t.units > 0 else -1
            st.entry_px = float(t.entry_price)
            st.open_trade_id = t.trade_id
            # We don't know the original entry_time; use "now" as the conservative
            # default — affects only the DB record's entry_time field, not P/L.
            st.entry_time_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            st.mfe_pips = 0.0
            st.psar_armed = False
            log.info(f"{cfg.pair}: adopted orphan trade {t.trade_id} pos={st.pos_dir:+d} "
                     f"@ {st.entry_px:.5f} units={int(t.units)}")
    except Exception as e:
        log.warning(f"{cfg.pair}: orphan-adoption failed: {e}")

    log.info(f"{cfg.pair} warmed (native {g1}+{g2}): "
             f"TF1_hist={len(st.tf1.history)}  TF2_hist={len(st.tf2.history)}  "
             f"PSAR_init={st.psar.initialized}  "
             f"aligned_long(TF1)={st.tf1.is_aligned_long} (TF2)={st.tf2.is_aligned_long}  "
             f"adopted_pos={st.pos_dir:+d}")


def main():
    account_id = os.environ.get(ACCOUNT_ENV)
    if not account_id:
        log.error(f"{ACCOUNT_ENV} not set — refusing to start")
        sys.exit(1)
    adapter = OANDAAdapter(account_id=account_id)

    states = []
    for cfg in CONFIGS:
        st = make_state(cfg)
        log.info(f"Warming up {cfg.pair} (TF1={cfg.n1}*S5  TF2={cfg.n2}*S5  "
                 f"SMA {cfg.sma_sm}/{cfg.sma_md}/{cfg.sma_lg}  TP={cfg.tp_pips:.0f}p  "
                 f"exit={cfg.exit_mode}  SL={cfg.sl_pips:.0f}p)...")
        warmup(adapter, st)
        st.tf1.novelty_long = False; st.tf1.novelty_short = False
        st.tf2.novelty_long = False; st.tf2.novelty_short = False
        states.append(st)

    _refresh_units(adapter)

    # Startup adoption (restart / VPS-cutover safety): re-adopt any position the broker
    # already holds so state is never falsely flat at boot. Mirrors the in-loop FLIP-GUARD
    # re-adoption; makes takeover of open trades immediate and log-verifiable.
    for st in states:
        try:
            open_now = [t for t in adapter.get_open_trades() if t.instrument == st.cfg.pair]
        except Exception as e:
            log.warning(f"{st.cfg.pair}: startup open-trade check failed ({e})")
            continue
        if open_now:
            t = open_now[0]
            st.pos_dir = 1 if t.units > 0 else -1
            st.open_trade_id = t.trade_id
            st.entry_px = float(t.entry_price)
            # Re-seed MFE from the position's CURRENT favorable excursion so a PSAR-trail
            # pair re-arms immediately if already in profit (0 if losing → stays disarmed,
            # broker ±200p stop covers it). st.psar.sar is rebuilt live during warmup.
            try:
                _b = adapter.get_candles(st.cfg.pair, count=2, granularity='S5')
                if _b:
                    _cur = float(_b[-1]['close'])
                    st.mfe_pips = max(0.0, (_cur - st.entry_px) / st.cfg.pip * st.pos_dir)
            except Exception:
                pass
            log.warning(f"{st.cfg.pair}: STARTUP ADOPTED broker pos {t.units:+g}u "
                        f"id={t.trade_id} @ {st.entry_px:.5f}  seed MFE={st.mfe_pips:+.1f}p"
                        f"{' (PSAR will arm)' if st.cfg.exit_mode=='psar' and st.mfe_pips>=st.cfg.psar_activate_pips else ''}")

    log.info(f"Stack-live v2 started on {account_id}: {', '.join(s.cfg.pair for s in states)}")
    log.info(f"  poll={POLL_SECS}s  units={_units} (balance×{UNITS_PER_DOLLAR}, cap MAX_UNITS={MAX_UNITS}, hourly refresh)")
    n_sl = sum(1 for c in CONFIGS if c.exit_mode == 'sl')
    n_psar = sum(1 for c in CONFIGS if c.exit_mode == 'psar')
    log.info(f"  per-pair exits configured ({n_sl} SL-bracket, {n_psar} PSAR-trail)")

    while not _shutdown:
        if time.time() - _last_balance_refresh >= BALANCE_REFRESH_SECS:
            _refresh_units(adapter)
        for st in states:
            try:
                reconcile_open_trade(adapter, st)
                s5_bars = adapter.get_candles(st.cfg.pair, count=10, granularity='S5')
                if not s5_bars or len(s5_bars) < 2:
                    continue
                new_bars = [b for b in s5_bars[:-1] if b['timestamp'] > st.last_s5_ts]
                for b in new_bars:
                    process_s5_bar(adapter, st, b)
                    st.last_s5_ts = b['timestamp']
            except Exception as e:
                log.warning(f"{st.cfg.pair} cycle error: {e}")
        for _ in range(POLL_SECS):
            if _shutdown: break
            time.sleep(1)
    log.info('Shutdown complete')


if __name__ == '__main__':
    main()
