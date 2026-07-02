"""
Grid-Trail v2 (Non-Reversal) Sweep — Session 064

v1 finding: reversal mode (close+reverse on opposite crossing) is structurally
negative EV — 66-79% of exits are costly reversals that overwhelm trail wins.

v2 fix: NON-REVERSAL mode.
  - Enter LONG or SHORT on grid crossing when FLAT.
  - While in position: IGNORE grid crossings. Only two exits:
      (a) Latching trail: activates when unrealized PnL ≥ tgt_pips,
          trails at tgt_pips from peak. Minimum gross = 0p.
      (b) Absolute stop: exit at entry ∓ stop_pips when trail not yet latched.
  - After exit: wait for next grid crossing to re-enter.

Theoretical improvement (random walk, box=24p, tgt=4.8p, stop=2×24=48p):
  P(win) = 48/(48+4.8) = 91.5%
  Expected trade: +2-7p net (depending on trend)

Parameters swept:
  box_mult  [5, 7, 10, 15, 20]  — box_pips = round(mult × sp_p90)
  tgt_mult  [2, 3, 4, 5]        — tgt_pips = mult × sp_p90
  stop_mult [2, 3]               — stop_pips = stop_mult × box_pips
  12 pairs × 5 × 4 × 2 = 480 configs total

SOP: R1, R2 (bull→hi first / bear→lo first), R3 (mid OHLC, spread deducted),
     R3b (BA parquets), R5 (IS P90 gate), R8 (OOS sealed once).

Run:
  cd /path/to/projects/fx-core
  python3 research/experiments/grid_trail/backtest_grid_trail_v2.py [PAIR ...]
"""

import math, sys, time, gc
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from pathlib import Path

BASE    = Path(__file__).resolve().parents[3]
BA_DIR  = BASE / "data" / "m5_ba"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

IS_FRAC  = 0.70
MC_P_THR = 0.05

PAIRS = [
    ("GBP_JPY", 0.01),   ("USD_JPY", 0.01),   ("EUR_JPY", 0.01),
    ("GBP_USD", 0.0001), ("EUR_USD", 0.0001),  ("AUD_USD", 0.0001),
    ("AUD_JPY", 0.01),   ("CAD_JPY", 0.01),   ("CHF_JPY", 0.01),
    ("NZD_USD", 0.0001), ("NZD_JPY", 0.01),   ("EUR_GBP", 0.0001),
]

BOX_MULTS  = [5, 7, 10, 15, 20]
TGT_MULTS  = [2, 3, 4, 5]
STOP_MULTS = [2, 3]
N_CFGS     = len(BOX_MULTS) * len(TGT_MULTS) * len(STOP_MULTS)   # 40 per pair


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(pair, pip):
    path = BA_DIR / f"{pair}_M5_BA.parquet"
    assert path.exists(), f"Missing BA parquet: {path}"
    df  = pd.read_parquet(path)
    op  = df["open"].values.astype(np.float64)
    hi  = df["high"].values.astype(np.float64)
    lo  = df["low"].values.astype(np.float64)
    cl  = df["close"].values.astype(np.float64)
    sp  = ((df["ask_c"] - df["bid_c"]) / pip).values.astype(np.float64)
    n      = len(df)
    is_end = int(n * IS_FRAC)
    sp_p90 = float(np.percentile(sp[:is_end], 90))
    sp_p50 = float(np.percentile(sp[:is_end], 50))
    c0 = is_end // 3;  c1 = 2 * (is_end // 3)
    chunks = np.zeros(n, dtype=np.int8)
    chunks[c0:c1]     = 1
    chunks[c1:is_end] = 2
    chunks[is_end:]   = 3
    is_days  = is_end / 288.0
    oos_days = (n - is_end) / 288.0
    return op, hi, lo, cl, sp, chunks, is_end, n, sp_p90, sp_p50, is_days, oos_days


def build_configs(sp_p90):
    """
    configs[:, 0] = box_pips_int (float64)
    configs[:, 1] = tgt_pips     (float64)
    configs[:, 2] = stop_pips    (float64)  — absolute stop distance in pips
    """
    rows, names = [], []
    for bm in BOX_MULTS:
        for tm in TGT_MULTS:
            for sm in STOP_MULTS:
                box_pips  = max(1, int(round(bm * sp_p90)))
                tgt_pips  = float(tm * sp_p90)
                stop_pips = float(sm * box_pips)
                rows.append((float(box_pips), tgt_pips, stop_pips))
                names.append(f"bm{bm}_tm{tm}_sm{sm}")
    return np.array(rows, dtype=np.float64), names


# ── Numba kernels ─────────────────────────────────────────────────────────────

@nb.njit
def _run_agg(op, hi, lo, cl, sp, chunks, box_pips_f, tgt_pips, stop_pips, sp_gate, pip):
    """
    Non-reversal grid-trail. Aggregate stats only (safe for prange).

    Exits: latching trail (when latched) OR absolute stop (when not latched).
    Grid crossings only trigger entry when flat — ignored while in position.
    """
    box_pips   = int(box_pips_f)
    pip_inv    = 1.0 / pip
    n          = len(cl)
    chunk_pnl  = np.zeros(4, dtype=np.float64)
    chunk_ntrd = np.zeros(4, dtype=np.int64)

    pos      = 0; entry_px = 0.0; entry_sp = 0.0
    hw_pnl   = 0.0; latched  = False
    prev_box = int(cl[0] * pip_inv + 0.5) // box_pips

    for i in range(1, n):
        cl_i = cl[i]; hi_i = hi[i]; lo_i = lo[i]; sp_i = sp[i]
        bull = cl_i >= op[i]
        ck   = chunks[i]
        cl_pips    = int(cl_i * pip_inv + 0.5)
        new_box    = cl_pips // box_pips
        cross_up   = new_box > prev_box
        cross_down = new_box < prev_box
        exited     = False

        # ── Within-bar: trail (if latched) or stop (if not latched) ──────────
        if pos == 1:
            if bull:  # hi first (R2)
                hw_pnl = max(hw_pnl, (hi_i - entry_px) / pip)
                if hw_pnl >= tgt_pips:
                    latched = True
                if latched:
                    ts = entry_px + (hw_pnl - tgt_pips) * pip
                    if lo_i <= ts:
                        net = (ts - entry_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                else:
                    stop_px = entry_px - stop_pips * pip
                    if lo_i <= stop_px:
                        net = (stop_px - entry_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
            else:  # bear bar: lo first (R2)
                if latched:
                    ts = entry_px + (hw_pnl - tgt_pips) * pip
                    if lo_i <= ts:
                        net = (ts - entry_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                else:
                    stop_px = entry_px - stop_pips * pip
                    if lo_i <= stop_px:
                        net = (stop_px - entry_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                if not exited:
                    hw_pnl = max(hw_pnl, (hi_i - entry_px) / pip)
                    if hw_pnl >= tgt_pips:
                        latched = True

        elif pos == -1:
            if not bull:  # bear bar: lo first (R2) — favorable for short
                hw_pnl = max(hw_pnl, (entry_px - lo_i) / pip)
                if hw_pnl >= tgt_pips:
                    latched = True
                if latched:
                    ts = entry_px - (hw_pnl - tgt_pips) * pip
                    if hi_i >= ts:
                        net = (entry_px - ts) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                else:
                    stop_px = entry_px + stop_pips * pip
                    if hi_i >= stop_px:
                        net = (entry_px - stop_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
            else:  # bull bar: hi first (adverse for short)
                if latched:
                    ts = entry_px - (hw_pnl - tgt_pips) * pip
                    if hi_i >= ts:
                        net = (entry_px - ts) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                else:
                    stop_px = entry_px + stop_pips * pip
                    if hi_i >= stop_px:
                        net = (entry_px - stop_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                if not exited:
                    hw_pnl = max(hw_pnl, (entry_px - lo_i) / pip)
                    if hw_pnl >= tgt_pips:
                        latched = True

        # ── Bar close: enter on crossing only when flat ───────────────────────
        if not exited and pos == 0 and (cross_up or cross_down) and sp_i <= sp_gate:
            pos = 1 if cross_up else -1
            entry_px = cl_i; entry_sp = sp_i
            hw_pnl = 0.0; latched = False

        prev_box = new_box

    return chunk_pnl, chunk_ntrd


@nb.njit(parallel=True)
def run_batch(op, hi, lo, cl, sp, chunks, configs, pip, sp_gate):
    n_cfg    = configs.shape[0]
    out_pnl  = np.zeros((n_cfg, 4), dtype=np.float64)
    out_ntrd = np.zeros((n_cfg, 4), dtype=np.int64)
    for c in prange(n_cfg):
        cp, cn = _run_agg(op, hi, lo, cl, sp, chunks,
                          configs[c, 0], configs[c, 1], configs[c, 2], sp_gate, pip)
        out_pnl[c]  = cp
        out_ntrd[c] = cn
    return out_pnl, out_ntrd


@nb.njit
def _run_full(op, hi, lo, cl, sp, chunks, box_pips_f, tgt_pips, stop_pips, sp_gate, pip):
    """Same as _run_agg but also collects IS per-trade PnL (for MC)."""
    box_pips   = int(box_pips_f)
    pip_inv    = 1.0 / pip
    n          = len(cl)
    chunk_pnl  = np.zeros(4, dtype=np.float64)
    chunk_ntrd = np.zeros(4, dtype=np.int64)
    pnl_buf    = np.empty(n // 4 + 10, dtype=np.float64)  # fewer trades than reversal mode
    buf_n      = 0

    pos = 0; entry_px = 0.0; entry_sp = 0.0
    hw_pnl = 0.0; latched = False
    prev_box = int(cl[0] * pip_inv + 0.5) // box_pips

    for i in range(1, n):
        cl_i = cl[i]; hi_i = hi[i]; lo_i = lo[i]; sp_i = sp[i]
        bull = cl_i >= op[i]
        ck   = chunks[i]
        cl_pips    = int(cl_i * pip_inv + 0.5)
        new_box    = cl_pips // box_pips
        cross_up   = new_box > prev_box
        cross_down = new_box < prev_box
        exited     = False

        if pos == 1:
            if bull:
                hw_pnl = max(hw_pnl, (hi_i - entry_px) / pip)
                if hw_pnl >= tgt_pips:
                    latched = True
                if latched:
                    ts = entry_px + (hw_pnl - tgt_pips) * pip
                    if lo_i <= ts:
                        net = (ts - entry_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        if ck < 3 and buf_n < len(pnl_buf):
                            pnl_buf[buf_n] = net; buf_n += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                else:
                    stop_px = entry_px - stop_pips * pip
                    if lo_i <= stop_px:
                        net = (stop_px - entry_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        if ck < 3 and buf_n < len(pnl_buf):
                            pnl_buf[buf_n] = net; buf_n += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
            else:
                if latched:
                    ts = entry_px + (hw_pnl - tgt_pips) * pip
                    if lo_i <= ts:
                        net = (ts - entry_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        if ck < 3 and buf_n < len(pnl_buf):
                            pnl_buf[buf_n] = net; buf_n += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                else:
                    stop_px = entry_px - stop_pips * pip
                    if lo_i <= stop_px:
                        net = (stop_px - entry_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        if ck < 3 and buf_n < len(pnl_buf):
                            pnl_buf[buf_n] = net; buf_n += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                if not exited:
                    hw_pnl = max(hw_pnl, (hi_i - entry_px) / pip)
                    if hw_pnl >= tgt_pips:
                        latched = True

        elif pos == -1:
            if not bull:
                hw_pnl = max(hw_pnl, (entry_px - lo_i) / pip)
                if hw_pnl >= tgt_pips:
                    latched = True
                if latched:
                    ts = entry_px - (hw_pnl - tgt_pips) * pip
                    if hi_i >= ts:
                        net = (entry_px - ts) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        if ck < 3 and buf_n < len(pnl_buf):
                            pnl_buf[buf_n] = net; buf_n += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                else:
                    stop_px = entry_px + stop_pips * pip
                    if hi_i >= stop_px:
                        net = (entry_px - stop_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        if ck < 3 and buf_n < len(pnl_buf):
                            pnl_buf[buf_n] = net; buf_n += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
            else:
                if latched:
                    ts = entry_px - (hw_pnl - tgt_pips) * pip
                    if hi_i >= ts:
                        net = (entry_px - ts) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        if ck < 3 and buf_n < len(pnl_buf):
                            pnl_buf[buf_n] = net; buf_n += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                else:
                    stop_px = entry_px + stop_pips * pip
                    if hi_i >= stop_px:
                        net = (entry_px - stop_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        if ck < 3 and buf_n < len(pnl_buf):
                            pnl_buf[buf_n] = net; buf_n += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                if not exited:
                    hw_pnl = max(hw_pnl, (entry_px - lo_i) / pip)
                    if hw_pnl >= tgt_pips:
                        latched = True

        if not exited and pos == 0 and (cross_up or cross_down) and sp_i <= sp_gate:
            pos = 1 if cross_up else -1
            entry_px = cl_i; entry_sp = sp_i
            hw_pnl = 0.0; latched = False

        prev_box = new_box

    return chunk_pnl, chunk_ntrd, pnl_buf[:buf_n].copy()


# ── Statistical tests ─────────────────────────────────────────────────────────

def mc_pvalue(pnl_arr):
    if len(pnl_arr) < 10:
        return 1.0
    n   = len(pnl_arr)
    mu  = pnl_arr.mean()
    sig = pnl_arr.std(ddof=1)
    if sig == 0.0:
        return 0.0 if mu > 0 else 1.0
    t = mu / (sig / math.sqrt(n))
    return 0.5 * math.erfc(t / math.sqrt(2))


def bootstrap_p5(pnl_arr, is_days, n_boot=500):
    if len(pnl_arr) == 0:
        return 0.0
    rng  = np.random.default_rng(42)
    sums = np.array([
        rng.choice(pnl_arr, len(pnl_arr), replace=True).sum()
        for _ in range(n_boot)
    ])
    return float(np.percentile(sums, 5)) / is_days


# ── Per-pair sweep ────────────────────────────────────────────────────────────

_jit_warmed = False

def run_pair(pair, pip):
    global _jit_warmed
    print(f"\n{'='*60}")
    print(f"  {pair}  (pip={pip})")
    print(f"{'='*60}")

    op, hi, lo, cl, sp, chunks, is_end, n, sp_p90, sp_p50, \
        is_days, oos_days = load_data(pair, pip)

    print(f"  Bars={n:,}  IS={is_end:,}  OOS={n-is_end:,}")
    print(f"  IS P90={sp_p90:.2f}p  P50={sp_p50:.2f}p")

    configs, names = build_configs(sp_p90)

    bm_labels = "  ".join(f"bm{bm}={int(round(bm*sp_p90))}p" for bm in BOX_MULTS)
    tm_labels = "  ".join(f"tm{tm}={tm*sp_p90:.1f}p" for tm in TGT_MULTS)
    print(f"  Box sizes: {bm_labels}")
    print(f"  Targets:   {tm_labels}")
    print(f"  Stops:     {['sm'+str(s)+': x'+str(s)+' box' for s in STOP_MULTS]}")

    if not _jit_warmed:
        print(f"\n  Warming up Numba JIT (500 bars, 1 config)...")
        t0 = time.perf_counter()
        run_batch(op[:500], hi[:500], lo[:500], cl[:500], sp[:500],
                  chunks[:500], configs[:1], pip, sp_p90)
        _run_full(op[:500], hi[:500], lo[:500], cl[:500], sp[:500],
                  chunks[:500], configs[0, 0], configs[0, 1], configs[0, 2], sp_p90, pip)
        print(f"  Compiled in {time.perf_counter()-t0:.1f}s")
        _jit_warmed = True

    print(f"\n  Running {N_CFGS} configs × {n:,} bars...")
    t0 = time.perf_counter()
    out_pnl, out_ntrd = run_batch(op, hi, lo, cl, sp, chunks, configs, pip, sp_p90)
    ntrd_total = int(out_ntrd[:, :3].sum())
    print(f"  Done in {time.perf_counter()-t0:.1f}s  |  {ntrd_total:,} IS trades total")
    print(f"  Avg trades/day (best config): "
          f"{out_ntrd[:, :3].sum(axis=1).max() / is_days:.1f}")

    is_pnl   = out_pnl[:, :3].sum(axis=1)
    is_ntrd  = out_ntrd[:, :3].sum(axis=1)
    oos_pnl  = out_pnl[:, 3]
    oos_ntrd = out_ntrd[:, 3]
    is_pd    = is_pnl / is_days

    # Quick IS summary before WF gate
    best_ci = int(np.argmax(is_pd))
    print(f"  Best IS p/d: {is_pd[best_ci]:.2f}  ({names[best_ci]}, "
          f"{int(is_ntrd[best_ci])} trades)")

    # Stage 1: WF — all 3 chunks positive + min trades
    min_per_chunk = 5
    chunk_ok = (np.all(out_pnl[:, :3] > 0, axis=1) &
                np.all(out_ntrd[:, :3] >= min_per_chunk, axis=1))
    s1_pass = np.where(chunk_ok)[0]
    print(f"\n  Stage 1: IS walk-forward screen...")
    print(f"    {len(s1_pass)}/{N_CFGS} passed IS WF")

    if len(s1_pass) == 0:
        print(f"  🔴 No configs passed WF — {pair} no edge found")
        return None

    # Stage 2: MC on top IS configs
    top_k   = min(N_CFGS, len(s1_pass))
    top_idx = s1_pass[np.argsort(is_pd[s1_pass])[::-1][:top_k]]
    print(f"\n  Stage 2: MC t-test (top {len(top_idx)} configs)...")

    mc_pass = []
    for ci in top_idx:
        _, _, pnl_arr = _run_full(op, hi, lo, cl, sp, chunks,
                                   configs[ci, 0], configs[ci, 1], configs[ci, 2], sp_p90, pip)
        pv = mc_pvalue(pnl_arr)
        if pv < MC_P_THR:
            mc_pass.append(ci)
    mc_pass = np.array(mc_pass, dtype=np.int64)
    print(f"    {len(mc_pass)}/{len(top_idx)} passed MC (p<{MC_P_THR})")

    if len(mc_pass) == 0:
        print(f"  🔴 No configs passed MC — {pair} no edge found")
        return None

    # Stage 3: OOS (sealed)
    print(f"\n  Stage 3: OOS evaluation (sealed)...")
    oos_pd   = oos_pnl / oos_days
    oos_pass = mc_pass[oos_pd[mc_pass] > 0]
    print(f"    {len(oos_pass)}/{len(mc_pass)} configs OOS p/d > 0")

    if len(oos_pass) == 0:
        print(f"  🔴 No OOS winners — {pair}")
        return None

    # Print top 8
    top8 = oos_pass[np.argsort(oos_pd[oos_pass])[::-1][:8]]
    print(f"\n  🟢 Top OOS configs for {pair}:")
    print(f"  {'name':>20s}  {'oos_pd':>8s}  {'oos_ntrd':>8s}  {'is_pd':>8s}  "
          f"{'p5':>8s}  {'box':>4s}  {'tgt':>5s}  {'stp':>5s}")

    best_row = None
    for rank, ci in enumerate(top8):
        _, _, pnl_arr = _run_full(op, hi, lo, cl, sp, chunks,
                                   configs[ci, 0], configs[ci, 1], configs[ci, 2], sp_p90, pip)
        p5 = bootstrap_p5(pnl_arr, is_days)
        bm_idx = ci // (len(TGT_MULTS) * len(STOP_MULTS))
        rem    = ci % (len(TGT_MULTS) * len(STOP_MULTS))
        tm_idx = rem // len(STOP_MULTS)
        sm_idx = rem % len(STOP_MULTS)
        bm = BOX_MULTS[bm_idx]; tm = TGT_MULTS[tm_idx]; sm = STOP_MULTS[sm_idx]
        box_p = int(round(bm * sp_p90)); tgt_p = round(tm * sp_p90, 1)
        stp_p = sm * box_p
        line = (f"  {names[ci]:>20s}  {oos_pd[ci]:>8.2f}  {int(oos_ntrd[ci]):>8d}"
                f"  {is_pd[ci]:>8.2f}  {p5:>8.2f}  {box_p:>4d}p  {tgt_p:>4.1f}p  {stp_p:>4d}p")
        print(line)
        if rank == 0:
            best_row = {
                "pair": pair, "name": names[ci],
                "oos_pd": round(oos_pd[ci], 2), "oos_ntrd": int(oos_ntrd[ci]),
                "is_pd": round(is_pd[ci], 2), "p5": round(p5, 2),
                "box_pips": box_p, "tgt_pips": tgt_p, "stop_pips": stp_p,
                "bm": bm, "tm": tm, "sm": sm,
                "sp_p90": round(sp_p90, 2), "s1_pass": len(s1_pass),
                "mc_pass": len(mc_pass), "oos_winners": len(oos_pass),
            }
    return best_row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pair_map = {p: pip for p, pip in PAIRS}
    targets  = sys.argv[1:] if len(sys.argv) > 1 else [p for p, _ in PAIRS]

    print("Grid-Trail v2 (Non-Reversal) Sweep")
    print(f"Box multiples:    {BOX_MULTS}")
    print(f"Target multiples: {TGT_MULTS}")
    print(f"Stop multiples:   {STOP_MULTS}  (× box pips)")
    print(f"Pairs:            {targets}")

    summaries = []
    for pair in targets:
        if pair not in pair_map:
            print(f"Unknown pair: {pair}"); continue
        result = run_pair(pair, pair_map[pair])
        if result:
            summaries.append(result)
        gc.collect()

    if summaries:
        print(f"\n{'='*60}")
        print(f"SUMMARY — Grid-Trail v2 sweep")
        print(f"{'='*60}")
        df = pd.DataFrame(summaries)
        df["best_oos_pd"] = df["oos_pd"]
        df["best_config"] = df["name"]
        print(df[["pair", "oos_winners", "best_oos_pd", "best_config",
                  "box_pips", "tgt_pips", "stop_pips", "sp_p90"]].to_string(index=False))

        print(f"\nOOS p/d ranking:")
        for r in sorted(summaries, key=lambda x: x["oos_pd"], reverse=True):
            icon = "🟢" if r["oos_pd"] > 5 else ("🟡" if r["oos_pd"] > 0 else "🔴")
            print(f"  {icon} {r['pair']:10s}: {r['oos_pd']:6.2f} p/d"
                  f"  box={r['box_pips']}p  tgt={r['tgt_pips']}p  stop={r['stop_pips']}p"
                  f"  p5={r['p5']:.2f}  n={r['oos_ntrd']:,}")


if __name__ == "__main__":
    main()
