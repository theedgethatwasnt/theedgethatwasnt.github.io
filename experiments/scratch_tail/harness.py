#!/usr/bin/env python3
"""
harness.py — SMA-Scratch Tail-Bounding Test: the 5-arm (+coin, +overlay-on-coin) portfolio
simulation engine. Governed by PREREGISTRATION.md.

R6 ("one code path"): every arm (A baseline / B closed-overlay / C floating-overlay / D
floating-overlay+stop / E stop-only, plus the coin and overlay-on-coin controls) is driven by
the SAME `simulate_portfolio()` function, parameterized by one `ArmCfg`.

Entry/exit logic (`_step_exit` + the entry branch inside `simulate_portfolio`) is a
verbatim-preserving port of `services/strategy_sma_scratch_paper/main.py::process_pair()`'s
position-management branch, extended with two NEW mechanisms not present in the live service
(both pre-registered, neither retrofit into "no code change" claims):
  1. a disaster stop (3.0xATR(14,H1)@entry, arms D/E), scanned with the same
     SL-before-TP-conservative + gap-through-fill convention as
     research/experiments/multiday_contrarian/harness.py's `_scan_barriers` (R2/R9);
  2. an entry-gating overlay (arms B/C/D + coin_D + coin_overlay) that can block ONLY new
     entries — never manages/forces an exit — using the same "equity vs its own causal SMA"
     rule as the deployed `services/equity_switch_monitor/monitor.py` (closed-trade case) /
     `research/experiments/conservative_010/bb_equity_switch.py` (the floating-equity variant).

DESIGN FINDING, addendum to PREREGISTRATION.md (discovered via gate 1, see test_harness.py):
a genuinely PROSPECTIVE overlay (one that actually skips entries, unlike the deployed
monitor.py which only ever paper-tracks a hypothetical switch and never gates real orders) is
self-referentially unstable if the gating signal is computed from the SAME (gated) trade
sequence it controls — the moment it blocks, no new trades close, so its own equity curve
freezes, and a frozen series is (using monitor.py's own tie convention, ties -> blocked) never
again strictly ABOVE its own moving average: a PERMANENT deadlock, reproduced on 100% of
synthetic-RW gate-1 runs (arms D/coin_D always converged to 0 trades before the fix below).
FIX (documented, not a silent patch): the overlay's gating SIGNAL is computed from a separate,
ALWAYS-ON REFERENCE run — the same entry logic and cost model, same stop configuration, but
overlay='none' — which keeps generating fresh trades/equity regardless of whether the GATED
arm itself is currently blocked. This is a truer reading of "the deployed monitor convention"
than a literal self-referential replica: monitor.py's own `eq`/`ma` are computed from the
REAL, ever-flowing (ungated) live trade sequence, precisely because the strategy it observes
is never itself gated by it. Reference mapping (pre-declared, not tuned): B <- A (closed-eq),
C <- A (floating-eq), D <- E (floating-eq, since D = "C's overlay on top of E's stop"), coin_D
<- coin_E (a new, internal-only ungated coin+stop reference), coin_overlay <- coin_A
(floating-eq). `run_battery()` below sequences phase 1 (ungated references: A, E, coin_A,
coin_E) before phase 2 (gated arms), each gated arm using its declared reference's precomputed
blocked-state timeline via a causal as-of lookup (`_BlockedLookup`).
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "multiday_contrarian"))
from carry_model import carry_pips, pip_of  # noqa: E402,F401

from signal import CONFIG_BY_PAIR, PAIRS, TP_PIPS, build_pair_signal  # noqa: E402

STOP_ATR_MULT = 3.0
CLOSED_OVERLAY_W = 10   # trades
FLOATING_OVERLAY_W = 24  # H1 samples
COIN_SEED = 20260707


@dataclass
class ArmCfg:
    name: str
    direction_source: str = "signal"   # "signal" | "coin"
    stop_mult: float = None            # None or STOP_ATR_MULT
    overlay: str = "none"              # "none" | "closed" | "floating"
    overlay_window: int = None
    seed: int = None                   # required if direction_source == "coin"
    ref_arm: str = None                # which ARMS[...] entry supplies the gating signal

    def __post_init__(self):
        if self.direction_source == "coin" and self.seed is None:
            self.seed = COIN_SEED
        if self.overlay == "closed" and self.overlay_window is None:
            self.overlay_window = CLOSED_OVERLAY_W
        if self.overlay == "floating" and self.overlay_window is None:
            self.overlay_window = FLOATING_OVERLAY_W


ARMS = {
    "A": ArmCfg("A", "signal", None, "none"),
    "B": ArmCfg("B", "signal", None, "closed", ref_arm="A"),
    "C": ArmCfg("C", "signal", None, "floating", ref_arm="A"),
    "D": ArmCfg("D", "signal", STOP_ATR_MULT, "floating", ref_arm="E"),
    "E": ArmCfg("E", "signal", STOP_ATR_MULT, "none"),
    "coin_A": ArmCfg("coin_A", "coin", None, "none"),
    # coin_E: internal-only ungated reference for coin_D (never itself pre-registered/reported
    # as a standalone arm, but a natural, cheap byproduct of the same code path).
    "coin_E": ArmCfg("coin_E", "coin", STOP_ATR_MULT, "none"),
    "coin_D": ArmCfg("coin_D", "coin", STOP_ATR_MULT, "floating", ref_arm="coin_E"),
    # "overlay-on-coin" control (PREREGISTRATION.md "Nulls (R10)"): arm C's floating overlay
    # applied to coin-flip entries, no stop — attribution check that the overlay alone cannot
    # manufacture positive expectancy from random trades.
    "coin_overlay": ArmCfg("coin_overlay", "coin", None, "floating", ref_arm="coin_A"),
}

# Battery execution order: phase-1 ungated references before phase-2 arms that consume them.
PHASE1 = ["A", "E", "coin_A", "coin_E"]
PHASE2 = ["B", "C", "D", "coin_D", "coin_overlay"]


@dataclass
class PairState:
    pos_dir: int = 0
    entry_px: float = 0.0
    entry_time: object = None
    entry_bar_count: int = -1
    entry_W_pips: float = 0.0
    entry_stop_level: float = None
    entry_spread_pips: float = 0.0
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    bar_count: int = 0
    last_close: float = np.nan


class BlockedLookup:
    """Causal as-of lookup: state AFTER the most recent reference update at or before a query
    timestamp. Tie-break is intentionally `blocked = value < ma` (STRICT less-than, i.e. ties
    resolve to UNBLOCKED) — see module docstring's design-finding note: this is what allows a
    reference run's necessarily-episodic equity curve to naturally re-open the gate after a
    long flat stretch, rather than a literal port of monitor.py's `>` (ties->blocked), which
    only avoids deadlock there because monitor.py's own reference is never itself gated."""

    def __init__(self, times, blocked):
        self.times = np.asarray(times)
        self.blocked = np.asarray(blocked, dtype=bool)

    def __call__(self, ts):
        if len(self.times) == 0:
            return False
        idx = np.searchsorted(self.times, ts, side="right") - 1
        if idx < 0:
            return False
        return bool(self.blocked[idx])


def _closed_blocked_lookup(trades, window=CLOSED_OVERLAY_W):
    """Build a BlockedLookup from a REFERENCE arm's own closed-trade equity curve: state
    AFTER each trade's close, using an SMA(window) of the closed-equity LEVEL itself (matches
    services/equity_switch_monitor/monitor.py's convention: `ma = rolling(W).mean() of eq`,
    a moving average of the running equity level, not of per-trade returns)."""
    trades_sorted = sorted(trades, key=lambda t: t["exit_ts"])
    eq = np.cumsum([t["net_pips"] for t in trades_sorted])
    times = np.array([t["exit_ts"] for t in trades_sorted])
    blocked = np.zeros(len(eq), dtype=bool)
    for k in range(len(eq)):
        if k + 1 >= window:
            ma = eq[k - window + 1: k + 1].mean()
            blocked[k] = eq[k] < ma
        else:
            blocked[k] = False
    return BlockedLookup(times, blocked)


def _floating_blocked_lookup(floating_eq_trace, window=FLOATING_OVERLAY_W):
    """Same idea as `_closed_blocked_lookup` but sampled on the reference's H1-tick floating
    (closed+open mark-to-market) equity trace."""
    if not floating_eq_trace:
        return BlockedLookup([], [])
    times = np.array([x[0] for x in floating_eq_trace])
    vals = np.array([x[1] for x in floating_eq_trace])
    blocked = np.zeros(len(vals), dtype=bool)
    for k in range(len(vals)):
        if k + 1 >= window:
            ma = vals[k - window + 1: k + 1].mean()
            blocked[k] = vals[k] < ma
        else:
            blocked[k] = False
    return BlockedLookup(times, blocked)


def _build_master_grid(pairs_m5, pairs_signal):
    """pairs_m5: {pair: m5_df (IS-filtered)}. pairs_signal: {pair: (dir_signal, atr_h1)}.
    Returns (master_ts: np.ndarray[datetime64], arrays: {pair: {field: np.ndarray}})."""
    all_ts = pd.concat([pd.Series(df["timestamp"].to_numpy()) for df in pairs_m5.values()])
    master_index = pd.DatetimeIndex(all_ts.unique()).sort_values()

    arrays = {}
    for pair, df in pairs_m5.items():
        dir_signal, atr_h1 = pairs_signal[pair]
        idx = pd.DatetimeIndex(df["timestamp"].to_numpy())
        cols = {}
        for c in ("open", "high", "low", "close", "bid_c", "ask_c"):
            cols[c] = pd.Series(df[c].to_numpy(), index=idx).reindex(master_index).to_numpy()
        dir_s = pd.Series(dir_signal, index=idx).reindex(master_index)
        atr_s = pd.Series(atr_h1, index=idx).reindex(master_index)
        present = ~np.isnan(cols["close"])
        cols["dir_signal"] = np.nan_to_num(dir_s.to_numpy(), nan=0).astype(np.int8)
        cols["atr_h1"] = atr_s.to_numpy()
        cols["present"] = present
        arrays[pair] = cols

    return master_index.to_numpy(), arrays


def _spread_pips(bid, ask, pip):
    return (ask - bid) / pip


def _step_exit(pst, cfg, pip, o, h, l, c, arm):
    """Check exits for an open position, in R2-conservative priority order:
    disaster stop (if arm.stop_mult set) -> TP -> quality (verbatim, no-op for current CONFIGS)
    -> scratch. Returns (exit_px, reason) or (None, None)."""
    d = pst.pos_dir
    held = pst.bar_count - pst.entry_bar_count

    high_pip = (h - pst.entry_px) / pip * d
    low_pip = (l - pst.entry_px) / pip * d
    if high_pip > pst.mfe_pips:
        pst.mfe_pips = high_pip
    if low_pip < pst.mae_pips:
        pst.mae_pips = low_pip

    # 0) disaster stop (arms D/E) — worst-outcome-first, gap-through-stop realistic fill (R2/R9)
    if arm.stop_mult is not None and pst.entry_stop_level is not None:
        sl = pst.entry_stop_level
        if d == 1:
            if o <= sl:
                return o, "stop_gap"
            if l <= sl:
                return sl, "stop"
        else:
            if o >= sl:
                return o, "stop_gap"
            if h >= sl:
                return sl, "stop"

    # 1) TP (verbatim: nominal fill, no gap favor)
    if d == 1:
        tp_lvl = pst.entry_px + TP_PIPS * pip
        if h >= tp_lvl:
            return tp_lvl, "tp"
    else:
        tp_lvl = pst.entry_px - TP_PIPS * pip
        if l <= tp_lvl:
            return tp_lvl, "tp"

    # 2) quality filter (verbatim; T_q_bars is None for all 6 deployed CONFIGS -> always no-op)
    if cfg.T_q_bars is not None and held == cfg.T_q_bars and pst.mfe_pips < cfg.X_pips:
        return c, "quality"

    # 3) scratch (verbatim)
    if held >= cfg.T_s_bars:
        dist = abs(c - pst.entry_px) / pip
        if dist <= pst.entry_W_pips:
            return c, "scratch"

    return None, None


def precompute_grid(pairs_m5):
    """Signal precompute (`build_pair_signal`, per pair — H1/M30 aggregation + six_of_six/ATR
    replay) and the master merged M5 grid are both ARM-INDEPENDENT (pure functions of the price
    data). `run_battery()` computes this ONCE and reuses it across all 9 arm-runs — a 9x
    reduction in redundant work vs calling `simulate_portfolio()` fresh per arm (each of which,
    before this reuse, was independently re-running the same H1/M30 signal replay AND grid
    union/reindex): this was the dominant cost, discovered via a runaway pytest wall-clock time
    (>280s for a handful of single-pair, multi-year run_battery() calls) — see JOURNEY-README /
    is_summary.md for the concrete timing. Returns (master_ts, arrays) as consumed by
    `simulate_portfolio(..., master_ts=..., arrays=...)`."""
    pairs_signal = {p: build_pair_signal(p, df) for p, df in pairs_m5.items()}
    return _build_master_grid(pairs_m5, pairs_signal)


def simulate_portfolio(pairs_m5, arm, spread_mult=1.0, markup_mult=1.0, ref_blocked_fn=None,
                        master_ts=None, arrays=None):
    """Run one arm over all pairs in `pairs_m5` ({pair: IS-filtered m5_df}) jointly.
    `ref_blocked_fn`: None for ungated arms (overlay='none'), else a `BlockedLookup` (or any
    callable ts->bool) built from a REFERENCE (ungated) arm's own equity — see module
    docstring's design-finding note for why the gating signal must NOT be self-referential.
    `master_ts`/`arrays`: precomputed via `precompute_grid(pairs_m5)` — pass these (as
    `run_battery()` does) to avoid re-deriving the arm-independent signal/grid on every call;
    omit for a one-off/standalone call (e.g. a single test), which computes them internally.
    Returns dict with: trades, open_at_end, floating_eq_trace, closed_eq_trace, blocked_trace.
    """
    if arm.overlay != "none" and ref_blocked_fn is None:
        raise ValueError(f"arm {arm.name!r} requires ref_blocked_fn (overlay={arm.overlay!r})")

    if master_ts is None or arrays is None:
        master_ts, arrays = precompute_grid(pairs_m5)
    n = len(master_ts)

    pst = {p: PairState() for p in pairs_m5}
    rng = {p: np.random.default_rng(arm.seed) for p in pairs_m5} if arm.direction_source == "coin" else {}

    trades = []
    closed_eq_running = 0.0
    closed_eq_trace = []
    floating_eq_trace = []
    blocked_trace = []

    ts_pd = pd.DatetimeIndex(master_ts)
    is_hour_tick = (ts_pd.minute == 0) & (ts_pd.second == 0)

    for t in range(n):
        for pair in pairs_m5:
            arr = arrays[pair]
            if not arr["present"][t]:
                continue
            cfg = CONFIG_BY_PAIR[pair]
            pip = cfg.pip
            st = pst[pair]
            st.bar_count += 1
            o, h, l, c = arr["open"][t], arr["high"][t], arr["low"][t], arr["close"][t]
            bid, ask = arr["bid_c"][t], arr["ask_c"][t]
            st.last_close = c

            if st.pos_dir != 0:
                exit_px, reason = _step_exit(st, cfg, pip, o, h, l, c, arm)
                if exit_px is not None:
                    direction = st.pos_dir
                    entry_ts = st.entry_time
                    exit_ts = master_ts[t]
                    gross_pips = (exit_px - st.entry_px) / pip * direction
                    exit_spread = _spread_pips(bid, ask, pip)
                    spread_rt_pips = (st.entry_spread_pips + exit_spread) / 2.0 * spread_mult
                    carry = carry_pips(pair, direction, entry_ts, exit_ts, markup_mult=markup_mult)
                    net_pips = gross_pips - spread_rt_pips + carry
                    trades.append({
                        "pair": pair, "arm": arm.name, "direction": direction,
                        "entry_ts": entry_ts, "entry_px": st.entry_px,
                        "exit_ts": exit_ts, "exit_px": exit_px, "exit_reason": reason,
                        "held_bars": st.bar_count - st.entry_bar_count,
                        "gross_pips": gross_pips, "spread_rt_pips": spread_rt_pips,
                        "carry_pips": carry, "net_pips": net_pips,
                        "mfe_pips": st.mfe_pips, "mae_pips": st.mae_pips,
                        "entry_W_pips": st.entry_W_pips,
                    })
                    st.pos_dir = 0
                    st.entry_px = 0.0
                    st.mfe_pips = 0.0
                    st.mae_pips = 0.0
                    st.entry_stop_level = None

                    closed_eq_running += net_pips
                    closed_eq_trace.append((exit_ts, closed_eq_running))

            if st.pos_dir == 0:
                raw_dir = int(arr["dir_signal"][t])
                if raw_dir != 0:
                    if arm.direction_source == "coin":
                        direction = 1 if rng[pair].random() < 0.5 else -1
                    else:
                        direction = raw_dir
                    blocked_now = ref_blocked_fn(master_ts[t]) if (arm.overlay != "none") else False
                    do_enter = not blocked_now
                    if do_enter:
                        atr_val = arr["atr_h1"][t]
                        if cfg.k_atr is not None:
                            if np.isnan(atr_val):
                                do_enter = False
                            else:
                                W = cfg.k_atr * (atr_val / pip)
                        else:
                            W = cfg.W_pips or 0.0
                        if do_enter:
                            st.pos_dir = direction
                            st.entry_px = c
                            st.entry_time = master_ts[t]
                            st.entry_bar_count = st.bar_count
                            st.entry_W_pips = W
                            st.entry_spread_pips = _spread_pips(bid, ask, pip)
                            st.mfe_pips = 0.0
                            st.mae_pips = 0.0
                            if arm.stop_mult is not None:
                                st.entry_stop_level = c - direction * arm.stop_mult * atr_val

        if is_hour_tick[t]:
            unrealized = 0.0
            for pair, st in pst.items():
                if st.pos_dir != 0 and not np.isnan(st.last_close):
                    pip = CONFIG_BY_PAIR[pair].pip
                    unrealized += st.pos_dir * (st.last_close - st.entry_px) / pip
            floating_eq_now = closed_eq_running + unrealized
            floating_eq_trace.append((master_ts[t], floating_eq_now))
            blocked_here = ref_blocked_fn(master_ts[t]) if (arm.overlay != "none") else False
            blocked_trace.append((master_ts[t], blocked_here))

    open_at_end = []
    for pair, st in pst.items():
        if st.pos_dir != 0:
            pip = CONFIG_BY_PAIR[pair].pip
            unrealized_pnl = st.pos_dir * (st.last_close - st.entry_px) / pip if not np.isnan(st.last_close) else float("nan")
            open_at_end.append({
                "pair": pair, "arm": arm.name, "direction": st.pos_dir,
                "entry_ts": st.entry_time, "entry_px": st.entry_px,
                "held_bars": st.bar_count - st.entry_bar_count,
                "mfe_pips": st.mfe_pips, "mae_pips": st.mae_pips,
                "mark_price": st.last_close, "unrealized_pnl_pips": unrealized_pnl,
            })

    return {
        "trades": trades,
        "open_at_end": open_at_end,
        "floating_eq_trace": floating_eq_trace,
        "closed_eq_trace": closed_eq_trace,
        "blocked_trace": blocked_trace,
    }


def run_battery(pairs_m5, spread_mult=1.0, markup_mult=1.0):
    """Run the full pre-declared arm set (PHASE1 ungated references, then PHASE2 gated arms
    consuming their declared reference's blocked-state timeline). Returns {arm_name: result}.
    Computes the arm-independent signal/grid ONCE (`precompute_grid`) and reuses it across all
    9 arm-runs — see that function's docstring for why (a 9x redundant-work reduction)."""
    master_ts, arrays = precompute_grid(pairs_m5)

    results = {}
    for name in PHASE1:
        results[name] = simulate_portfolio(pairs_m5, ARMS[name], spread_mult, markup_mult,
                                            master_ts=master_ts, arrays=arrays)

    for name in PHASE2:
        arm = ARMS[name]
        ref = results[arm.ref_arm]
        if arm.overlay == "closed":
            ref_fn = _closed_blocked_lookup(ref["trades"], arm.overlay_window)
        else:
            ref_fn = _floating_blocked_lookup(ref["floating_eq_trace"], arm.overlay_window)
        results[name] = simulate_portfolio(pairs_m5, arm, spread_mult, markup_mult,
                                            ref_blocked_fn=ref_fn, master_ts=master_ts, arrays=arrays)

    return results
