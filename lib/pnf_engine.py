"""
Canonical P&F signal engine — single source of truth for all FIFO-Trends logic.

R6: both strategy_fifo_paper and strategy_fifo_live import from here.
    The Numba backtest kernel is a separate speed-optimised reimplementation;
    tests/test_fifo_r7.py asserts byte-for-byte state equality after N bars.

Versioning note: any change here MUST re-run tests/test_fifo_r7.py before deploy.
"""

from dataclasses import dataclass, field
from typing import List

MAX_COL_HIST = 10  # ring-buffer depth for completed column heights (R4a)


@dataclass
class PnFConfig:
    pip:             float
    box_pips:        float  # box size in pips (float to support non-integer boxes)
    rev:             float  # reversal — boxes for integer mode, fractional for price-based
    n_min:           int    # min prev-column length to qualify E2 entry
    trail_d:         int    # trail distance in boxes from high-water
    x7_k:            int    # X7 col-SMA lookback; 0 = X3b only (no X7)
    sp_gate:         float  # IS P90 spread gate in pips (hardcoded per pair, R5)
    price_based_rev: bool = False  # True → px<=pnf_level-rev*bs check (fractional rev support)


@dataclass
class PnFState:
    pnf_idx:      int   = 0
    pnf_level:    float = 0.0
    pnf_dir:      int   = 0       # 0=uninit, +1=up column, -1=down column
    col_count:    int   = 0
    prev_col:     int   = 0
    col_hist:     List[float] = field(default_factory=lambda: [0.0] * MAX_COL_HIST)
    col_hist_ptr: int   = 0
    col_hist_n:   int   = 0
    pos:          int   = 0       # 0=flat, +1=long, -1=short
    entry_px:     float = 0.0     # mid close at entry (caller patches with real fill)
    hw_level:     float = 0.0     # high-water P&F level for trailing stop
    pending:      int   = 0       # E2 pending direction


@dataclass
class BarResult:
    """Pure signal output — no I/O. Caller does broker/DB work."""
    exit_triggered:  bool  = False
    exit_reason:     str   = ""
    exit_px:         float = 0.0  # trail or X7 price
    exited_pos:      int   = 0    # direction of the trade that closed (+1/-1)
    exited_entry_px: float = 0.0  # mid entry price of closed trade (for P&L)
    entry_signal:    int   = 0    # +1=go long, -1=go short, 0=no entry
    entry_px:        float = 0.0  # mid bar close (caller adjusts for real fill)


def _update_pnf(st: PnFState, price: float, bs: float, rev: float,
                price_based_rev: bool = False):
    """
    Update P&F chart with one price tick.
    Returns (did_reverse, prev_col_at_rev).

    Integer mode (price_based_rev=False, default):
      Reversal when int(price/bs) has moved rev boxes from column extreme.
      Classic P&F — no float drift.

    Price-based mode (price_based_rev=True):
      Reversal when price <= pnf_level - rev*bs (up col) or >= pnf_level + rev*bs (down col).
      Supports fractional rev values (e.g. 1.5 boxes). Stricter filter — requires price
      to reach the box boundary, not just enter the box.
    """
    if st.pnf_dir == 0:
        st.pnf_idx   = int(price / bs)
        st.pnf_level = st.pnf_idx * bs
        st.pnf_dir   = 1
        st.col_count = 1
        return False, 0

    new_idx = int(price / bs)
    delta   = new_idx - st.pnf_idx

    if st.pnf_dir == 1:
        if delta >= 1:                              # extension up (same both modes)
            st.pnf_idx    = new_idx
            st.pnf_level  = st.pnf_idx * bs
            st.col_count += delta
        else:
            rev_triggered = (price <= st.pnf_level - rev * bs) if price_based_rev \
                            else (delta <= -int(rev))
            if rev_triggered:
                old_idx      = st.pnf_idx
                st.prev_col  = st.col_count
                st.col_hist[st.col_hist_ptr % MAX_COL_HIST] = st.prev_col
                st.col_hist_ptr += 1
                if st.col_hist_n < MAX_COL_HIST:
                    st.col_hist_n += 1
                st.pnf_dir   = -1
                st.pnf_idx   = new_idx
                st.pnf_level = st.pnf_idx * bs
                st.col_count = max(1, old_idx - new_idx)
                return True, st.prev_col
    else:  # pnf_dir == -1
        if delta <= -1:                             # extension down
            st.pnf_idx    = new_idx
            st.pnf_level  = st.pnf_idx * bs
            st.col_count += (-delta)
        else:
            rev_triggered = (price >= st.pnf_level + rev * bs) if price_based_rev \
                            else (delta >= int(rev))
            if rev_triggered:
                old_idx      = st.pnf_idx
                st.prev_col  = st.col_count
                st.col_hist[st.col_hist_ptr % MAX_COL_HIST] = st.prev_col
                st.col_hist_ptr += 1
                if st.col_hist_n < MAX_COL_HIST:
                    st.col_hist_n += 1
                st.pnf_dir   = 1
                st.pnf_idx   = new_idx
                st.pnf_level = st.pnf_idx * bs
                st.col_count = max(1, new_idx - old_idx)
                return True, st.prev_col

    return False, 0


def _col_sma(st: PnFState, k: int) -> float:
    """Mean of last min(k, col_hist_n) completed column heights (R4a)."""
    count = min(k, st.col_hist_n)
    if count == 0:
        return 0.0
    total = 0.0
    for j in range(count):
        idx = (st.col_hist_ptr - 1 - j) % MAX_COL_HIST
        total += st.col_hist[idx]
    return total / count


def process_bar(st: PnFState, cfg: PnFConfig,
                bar_open: float, bar_high: float, bar_low: float, bar_close: float,
                sp_pips: float) -> BarResult:
    """
    Process one closed M5 bar. Updates st in place. Returns BarResult (no I/O).

    R2: bull=(close>=open) → high first, low second.
    R3: entry_px = bar_close (mid); caller overrides with real broker fill.
    R5: sp_pips vs cfg.sp_gate gates entry.
    """
    bs  = cfg.box_pips * cfg.pip
    rev = cfg.rev
    pbr = cfg.price_based_rev

    bull = (bar_close >= bar_open)
    p1   = bar_high if bull else bar_low
    p2   = bar_low  if bull else bar_high

    did_rev_p1, prev_p1 = _update_pnf(st, p1, bs, rev, pbr)
    did_rev_p2, prev_p2 = _update_pnf(st, p2, bs, rev, pbr)
    did_rev         = did_rev_p1 or did_rev_p2
    prev_col_at_rev = prev_p1 if did_rev_p1 else prev_p2

    # Update high-water level
    if st.pos == 1:
        if st.pnf_dir == 1 and st.pnf_level > st.hw_level:
            st.hw_level = st.pnf_level
    elif st.pos == -1:
        if st.pnf_dir == -1 and st.pnf_level < st.hw_level:
            st.hw_level = st.pnf_level

    result = BarResult()

    # ── EXIT ──────────────────────────────────────────────────────────────────
    if st.pos != 0:
        d_val = float(cfg.trail_d)

        if st.pos == 1:
            trail = st.hw_level - d_val * bs
            if bar_low <= trail:
                result.exit_triggered  = True
                result.exit_reason     = "trail"
                result.exit_px         = trail
        else:
            trail = st.hw_level + d_val * bs
            if bar_high >= trail:
                result.exit_triggered  = True
                result.exit_reason     = "trail"
                result.exit_px         = trail

        if not result.exit_triggered and cfg.x7_k > 0 and st.pnf_dir != st.pos:
            sma_k = _col_sma(st, cfg.x7_k)
            if sma_k > 0.0 and st.col_count >= sma_k:
                result.exit_triggered  = True
                result.exit_reason     = "x7"
                result.exit_px         = bar_close

    if result.exit_triggered:
        result.exited_pos      = st.pos
        result.exited_entry_px = st.entry_px
        st.pos      = 0
        st.entry_px = 0.0
        st.hw_level = 0.0

    # ── ENTRY ─────────────────────────────────────────────────────────────────
    if st.pos == 0:
        if sp_pips <= cfg.sp_gate:
            if did_rev and prev_col_at_rev >= cfg.n_min:
                st.pending = st.pnf_dir
            if did_rev and st.pending != 0 and st.pnf_dir != st.pending:
                st.pending = 0
            if st.pending != 0 and st.pnf_dir == st.pending and st.col_count > rev:
                result.entry_signal = st.pending
                result.entry_px     = bar_close
                st.pos      = st.pending
                st.entry_px = bar_close  # placeholder; live caller overrides with fill_px
                st.hw_level = st.pnf_level
                st.pending  = 0
        else:
            if did_rev and st.pending != 0 and st.pnf_dir != st.pending:
                st.pending = 0

    return result
