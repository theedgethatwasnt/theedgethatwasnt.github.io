"""
Grid-Trail Sweep — Session 064

Strategy:
  - Divide price into a fixed grid: box_pips = round(box_mult × IS_P90_spread)
  - Track close-to-close grid crossings (integer pip arithmetic — no float drift)
  - Enter LONG on upward crossing, SHORT on downward (reversal system —
    opposite crossing closes current trade and enters new direction)
  - Exit: latching trail
      latch activates when unrealized gross PnL ≥ tgt_pips
      trail fires when gross PnL ≤ hw_pnl - tgt_pips  (locks in hw_pnl−tgt)

Parameters swept:
  box_mult  [5, 7, 10, 15, 20]  — box_pips = round(mult × sp_p90)
  tgt_mult  [2, 3, 4, 5]        — tgt_pips = mult × sp_p90
  12 pairs × 5 × 4 = 240 configs total

Grid arithmetic: prices converted to nearest-pip integers before box index
computation (cl_pips = int(cl / pip + 0.5), new_box = cl_pips // box_pips_int).
This eliminates sub-pip float noise at boundaries that would otherwise cause
hundreds of spurious crossings per day.

SOP compliance (CLAUDE.md Backtest–Live Consistency SOP):
  R1  Closed bars only — bar[i] consumed after it closes
  R2  Within-bar: bull=(close≥open) → HIGH first then LOW; bear → LOW first
  R3  Mid OHLC for signals; spread deducted at entry+exit (half each side)
  R3b BA parquets required; no fallback spread
  R5  IS P90 spread gate — hardcoded IS scalar, not full-data percentile
  R8  OOS touched exactly once, after all IS/MC gates pass

Run:
  cd /path/to/projects/fx-core
  python3 research/experiments/grid_trail/backtest_grid_trail.py [PAIR ...]
  # omit PAIR to run all 12 pairs
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

BOX_MULTS = [5, 7, 10, 15, 20]
TGT_MULTS = [2, 3, 4, 5]
N_CFGS    = len(BOX_MULTS) * len(TGT_MULTS)   # 20 per pair


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
    sp_p90 = float(np.percentile(sp[:is_end], 90))   # R5: IS-only gate
    sp_p50 = float(np.percentile(sp[:is_end], 50))

    # Chunk labels: 0/1/2 = three IS walk-forward chunks, 3 = OOS
    c0 = is_end // 3
    c1 = 2 * (is_end // 3)
    chunks = np.zeros(n, dtype=np.int8)
    chunks[c0:c1]     = 1
    chunks[c1:is_end] = 2
    chunks[is_end:]   = 3

    is_days  = is_end / 288.0
    oos_days = (n - is_end) / 288.0
    ck_days  = [c0 / 288.0, (c1 - c0) / 288.0, (is_end - c1) / 288.0]
    return op, hi, lo, cl, sp, chunks, is_end, n, sp_p90, sp_p50, is_days, oos_days, ck_days


def build_configs(sp_p90):
    """
    Returns configs array with shape (N, 2):
      col 0: box_pips_int  (int64 stored as float64 — cast inside kernel)
      col 1: tgt_pips      (float64)
    """
    rows, names = [], []
    for bm in BOX_MULTS:
        for tm in TGT_MULTS:
            box_pips_int = max(1, int(round(bm * sp_p90)))  # whole pips
            tgt_pips     = float(tm * sp_p90)
            rows.append((float(box_pips_int), tgt_pips))
            names.append(f"bm{bm}_tm{tm}")
    return np.array(rows, dtype=np.float64), names


# ── Numba kernels ─────────────────────────────────────────────────────────────

@nb.njit
def _run_agg(op, hi, lo, cl, sp, chunks, box_pips_f, tgt_pips, sp_gate, pip):
    """
    Core grid-trail loop — aggregate stats only (safe for prange).

    box_pips_f: integer box width in pips, passed as float64 (cast to int64 here).
    Grid computed in integer pip space: cl_pips = round(cl/pip), new_box = cl_pips // box_pips.
    This eliminates sub-pip float oscillations at boundaries.

    chunk_pnl[0..2] = IS walk-forward chunk PnL; chunk_pnl[3] = OOS PnL.
    chunk_ntrd[k]   = trade count per chunk.
    """
    box_pips   = int(box_pips_f)
    pip_inv    = 1.0 / pip
    n          = len(cl)
    chunk_pnl  = np.zeros(4, dtype=np.float64)
    chunk_ntrd = np.zeros(4, dtype=np.int64)

    pos      = 0       # 0=flat, 1=long, -1=short
    entry_px = 0.0
    entry_sp = 0.0
    hw_pnl   = 0.0     # high-water gross pips from entry
    latched  = False
    prev_box = int(cl[0] * pip_inv + 0.5) // box_pips

    for i in range(1, n):
        cl_i = cl[i]; hi_i = hi[i]; lo_i = lo[i]; sp_i = sp[i]
        bull = cl_i >= op[i]
        ck   = chunks[i]

        cl_pips    = int(cl_i * pip_inv + 0.5)  # round to nearest pip
        new_box    = cl_pips // box_pips
        cross_up   = new_box > prev_box
        cross_down = new_box < prev_box
        exited     = False

        # ── Within-bar trail stop (R2) ────────────────────────────────────
        if pos == 1:  # long
            if bull:  # R2: bull bar → HIGH first
                hw_pnl = max(hw_pnl, (hi_i - entry_px) / pip)
                if hw_pnl >= tgt_pips:
                    latched = True
                if latched:
                    ts = entry_px + (hw_pnl - tgt_pips) * pip
                    if lo_i <= ts:
                        net = (ts - entry_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
            else:     # bear bar → LOW first: check exit before updating hw
                if latched:
                    ts = entry_px + (hw_pnl - tgt_pips) * pip
                    if lo_i <= ts:
                        net = (ts - entry_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                if not exited:
                    hw_pnl = max(hw_pnl, (hi_i - entry_px) / pip)
                    if hw_pnl >= tgt_pips:
                        latched = True

        elif pos == -1:  # short
            if not bull:  # R2: bear bar → LOW first (favorable for short)
                hw_pnl = max(hw_pnl, (entry_px - lo_i) / pip)
                if hw_pnl >= tgt_pips:
                    latched = True
                if latched:
                    ts = entry_px - (hw_pnl - tgt_pips) * pip
                    if hi_i >= ts:
                        net = (entry_px - ts) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
            else:         # bull bar → HIGH first (adverse for short): check exit first
                if latched:
                    ts = entry_px - (hw_pnl - tgt_pips) * pip
                    if hi_i >= ts:
                        net = (entry_px - ts) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                if not exited:
                    hw_pnl = max(hw_pnl, (entry_px - lo_i) / pip)
                    if hw_pnl >= tgt_pips:
                        latched = True

        # ── Bar close: grid crossing reversal ─────────────────────────────
        if not exited:
            if pos == 1 and cross_down:
                net = (cl_i - entry_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                pos = 0; latched = False; hw_pnl = 0.0
                if sp_i <= sp_gate:
                    pos = -1; entry_px = cl_i; entry_sp = sp_i
            elif pos == -1 and cross_up:
                net = (entry_px - cl_i) / pip - entry_sp * 0.5 - sp_i * 0.5
                chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                pos = 0; latched = False; hw_pnl = 0.0
                if sp_i <= sp_gate:
                    pos = 1; entry_px = cl_i; entry_sp = sp_i
            elif pos == 0 and (cross_up or cross_down) and sp_i <= sp_gate:
                pos = 1 if cross_up else -1
                entry_px = cl_i; entry_sp = sp_i

        prev_box = new_box

    return chunk_pnl, chunk_ntrd


@nb.njit(parallel=True)
def run_batch(op, hi, lo, cl, sp, chunks, configs, pip, sp_gate):
    """configs[:, 0] = box_pips_int (as float64), configs[:, 1] = tgt_pips."""
    n_cfg    = configs.shape[0]
    out_pnl  = np.zeros((n_cfg, 4), dtype=np.float64)
    out_ntrd = np.zeros((n_cfg, 4), dtype=np.int64)
    for c in prange(n_cfg):
        cp, cn = _run_agg(op, hi, lo, cl, sp, chunks,
                          configs[c, 0], configs[c, 1], sp_gate, pip)
        out_pnl[c]  = cp
        out_ntrd[c] = cn
    return out_pnl, out_ntrd


@nb.njit
def _run_full(op, hi, lo, cl, sp, chunks, box_pips_f, tgt_pips, sp_gate, pip):
    """Same as _run_agg but also collects IS per-trade PnL array (for MC)."""
    box_pips   = int(box_pips_f)
    pip_inv    = 1.0 / pip
    n          = len(cl)
    chunk_pnl  = np.zeros(4, dtype=np.float64)
    chunk_ntrd = np.zeros(4, dtype=np.int64)
    pnl_buf    = np.empty(n // 2 + 10, dtype=np.float64)
    buf_n      = 0

    pos = 0; entry_px = 0.0; entry_sp = 0.0
    hw_pnl = 0.0; latched = False
    prev_box = int(cl[0] * pip_inv + 0.5) // box_pips

    for i in range(1, n):
        cl_i = cl[i]; hi_i = hi[i]; lo_i = lo[i]; sp_i = sp[i]
        bull = cl_i >= op[i]
        ck   = chunks[i]

        cl_pips    = int(cl_i * pip_inv + 0.5)   # round to nearest pip
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
                if latched:
                    ts = entry_px + (hw_pnl - tgt_pips) * pip
                    if lo_i <= ts:
                        net = (ts - entry_px) / pip - entry_sp * 0.5 - sp_i * 0.5
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
                if latched:
                    ts = entry_px - (hw_pnl - tgt_pips) * pip
                    if hi_i >= ts:
                        net = (entry_px - ts) / pip - entry_sp * 0.5 - sp_i * 0.5
                        chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                        if ck < 3 and buf_n < len(pnl_buf):
                            pnl_buf[buf_n] = net; buf_n += 1
                        pos = 0; latched = False; hw_pnl = 0.0; exited = True
                if not exited:
                    hw_pnl = max(hw_pnl, (entry_px - lo_i) / pip)
                    if hw_pnl >= tgt_pips:
                        latched = True

        if not exited:
            if pos == 1 and cross_down:
                net = (cl_i - entry_px) / pip - entry_sp * 0.5 - sp_i * 0.5
                chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                if ck < 3 and buf_n < len(pnl_buf):
                    pnl_buf[buf_n] = net; buf_n += 1
                pos = 0; latched = False; hw_pnl = 0.0
                if sp_i <= sp_gate:
                    pos = -1; entry_px = cl_i; entry_sp = sp_i
            elif pos == -1 and cross_up:
                net = (entry_px - cl_i) / pip - entry_sp * 0.5 - sp_i * 0.5
                chunk_pnl[ck] += net; chunk_ntrd[ck] += 1
                if ck < 3 and buf_n < len(pnl_buf):
                    pnl_buf[buf_n] = net; buf_n += 1
                pos = 0; latched = False; hw_pnl = 0.0
                if sp_i <= sp_gate:
                    pos = 1; entry_px = cl_i; entry_sp = sp_i
            elif pos == 0 and (cross_up or cross_down) and sp_i <= sp_gate:
                pos = 1 if cross_up else -1
                entry_px = cl_i; entry_sp = sp_i

        prev_box = new_box

    return chunk_pnl, chunk_ntrd, pnl_buf[:buf_n].copy()


# ── Statistical tests ─────────────────────────────────────────────────────────

def mc_pvalue(pnl_arr):
    """One-sample one-tailed t-test: H0 = mean ≤ 0. Normal approx (valid n≥30)."""
    if len(pnl_arr) < 10:
        return 1.0
    n   = len(pnl_arr)
    mu  = pnl_arr.mean()
    sig = pnl_arr.std(ddof=1)
    if sig == 0.0:
        return 0.0 if mu > 0 else 1.0
    t = mu / (sig / math.sqrt(n))
    return 0.5 * math.erfc(t / math.sqrt(2))  # P(Z > t) standard normal


def bootstrap_p5(pnl_arr, is_days, n_boot=500):
    """5th-percentile bootstrap IS p/d — lower confidence bound on the edge."""
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
        is_days, oos_days, ck_days = load_data(pair, pip)

    print(f"  Bars={n:,}  IS={is_end:,}  OOS={n-is_end:,}")
    print(f"  IS P90={sp_p90:.2f}p  P50={sp_p50:.2f}p")

    configs, names = build_configs(sp_p90)

    bm_labels = "  ".join(f"bm{bm}={int(round(bm*sp_p90))}p" for bm in BOX_MULTS)
    tm_labels = "  ".join(f"tm{tm}={tm*sp_p90:.1f}p" for tm in TGT_MULTS)
    print(f"  Box sizes: {bm_labels}")
    print(f"  Targets:   {tm_labels}")

    if not _jit_warmed:
        print(f"\n  Warming up Numba JIT (500 bars, 1 config)...")
        t0 = time.perf_counter()
        run_batch(op[:500], hi[:500], lo[:500], cl[:500], sp[:500],
                  chunks[:500], configs[:1], pip, sp_p90)
        _run_full(op[:500], hi[:500], lo[:500], cl[:500], sp[:500],
                  chunks[:500], configs[0, 0], configs[0, 1], sp_p90, pip)
        print(f"  Compiled in {time.perf_counter()-t0:.1f}s")
        _jit_warmed = True

    print(f"\n  Running {N_CFGS} configs × {n:,} bars...")
    t0 = time.perf_counter()
    out_pnl, out_ntrd = run_batch(op, hi, lo, cl, sp, chunks, configs, pip, sp_p90)
    print(f"  Done in {time.perf_counter()-t0:.1f}s  |  "
          f"{int(out_ntrd[:, :3].sum()):,} IS trades total")

    is_pnl   = out_pnl[:, :3].sum(axis=1)
    is_ntrd  = out_ntrd[:, :3].sum(axis=1)
    oos_pnl  = out_pnl[:, 3]
    oos_ntrd = out_ntrd[:, 3]
    is_pd    = is_pnl / is_days

    # Stage 1: IS walk-forward screen — all 3 chunks positive + min trades
    min_per_chunk = 5
    chunk_ok = (np.all(out_pnl[:, :3] > 0, axis=1) &
                np.all(out_ntrd[:, :3] >= min_per_chunk, axis=1))
    s1_pass = np.where(chunk_ok)[0]
    print(f"\n  Stage 1: IS walk-forward screen...")
    print(f"    {len(s1_pass)}/{N_CFGS} passed IS WF (all 3 chunks >0, ≥{min_per_chunk} trades each)")

    if len(s1_pass) == 0:
        print(f"  ⚠️  No configs passed WF — {pair} skipped")
        return None

    # Stage 2: MC on top IS configs
    top_k   = min(N_CFGS, len(s1_pass))   # all 20 if few enough
    top_idx = s1_pass[np.argsort(is_pd[s1_pass])[::-1][:top_k]]
    print(f"\n  Stage 2: MC t-test (top {len(top_idx)} configs)...")

    mc_pass = []
    for ci in top_idx:
        _, _, pnl_arr = _run_full(op, hi, lo, cl, sp, chunks,
                                   configs[ci, 0], configs[ci, 1], sp_p90, pip)
        pv = mc_pvalue(pnl_arr)
        if pv < MC_P_THR:
            mc_pass.append(ci)
    mc_pass = np.array(mc_pass, dtype=np.int64)
    print(f"    {len(mc_pass)}/{len(top_idx)} passed MC (p<{MC_P_THR})")

    if len(mc_pass) == 0:
        print(f"  ⚠️  No configs passed MC — {pair} skipped")
        return None

    # Stage 3: OOS (sealed — one-time evaluation)
    print(f"\n  Stage 3: OOS evaluation (sealed)...")
    oos_pd   = oos_pnl / oos_days
    oos_pass = mc_pass[oos_pd[mc_pass] > 0]
    print(f"    {len(oos_pass)}/{len(mc_pass)} configs OOS p/d > 0")

    if len(oos_pass) == 0:
        print(f"  ⚠️  No OOS winners — {pair} skipped")
        return None

    # Print top OOS results with bootstrap_p5
    top8 = oos_pass[np.argsort(oos_pd[oos_pass])[::-1][:8]]
    print(f"\n  🟢 Top OOS configs for {pair}:")
    print(f"  {'name':>16s}  {'oos_pd':>8s}  {'oos_ntrd':>8s}  {'is_pd':>8s}  {'p5':>8s}  {'box_p':>6s}  {'tgt_p':>6s}")

    best_row = None
    for rank, ci in enumerate(top8):
        _, _, pnl_arr = _run_full(op, hi, lo, cl, sp, chunks,
                                   configs[ci, 0], configs[ci, 1], sp_p90, pip)
        p5 = bootstrap_p5(pnl_arr, is_days)
        bm = BOX_MULTS[ci // len(TGT_MULTS)]
        tm = TGT_MULTS[ci % len(TGT_MULTS)]
        box_p = int(round(bm * sp_p90))   # integer pips (matches kernel)
        tgt_p = round(tm * sp_p90, 1)
        line = (f"  {names[ci]:>16s}  {oos_pd[ci]:>8.2f}  {int(oos_ntrd[ci]):>8d}"
                f"  {is_pd[ci]:>8.2f}  {p5:>8.2f}  {box_p:>5d}p  {tgt_p:>6.1f}p")
        print(line)
        if rank == 0:
            best_row = {"pair": pair, "name": names[ci],
                        "oos_pd": round(oos_pd[ci], 2),
                        "oos_ntrd": int(oos_ntrd[ci]),
                        "is_pd": round(is_pd[ci], 2),
                        "p5": round(p5, 2),
                        "box_pips": int(box_p), "tgt_pips": tgt_p,
                        "bm": bm, "tm": tm,
                        "sp_p90": round(sp_p90, 2), "sp_p50": round(sp_p50, 2),
                        "s1_pass": len(s1_pass), "mc_pass": len(mc_pass),
                        "oos_winners": len(oos_pass)}
    return best_row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pair_map = {p: pip for p, pip in PAIRS}
    targets  = sys.argv[1:] if len(sys.argv) > 1 else [p for p, _ in PAIRS]

    print("Grid-Trail Sweep")
    print(f"Box multiples:    {BOX_MULTS}")
    print(f"Target multiples: {TGT_MULTS}")
    print(f"Pairs:            {targets}")

    summaries = []
    for pair in targets:
        if pair not in pair_map:
            print(f"Unknown pair: {pair}")
            continue
        result = run_pair(pair, pair_map[pair])
        if result:
            summaries.append(result)
        gc.collect()

    if summaries:
        print(f"\n{'='*60}")
        print(f"SUMMARY — Grid-Trail sweep")
        print(f"{'='*60}")
        df = pd.DataFrame(summaries)
        cols = ["pair", "s1_pass", "mc_pass", "oos_winners",
                "best_oos_pd", "best_config", "box_pips", "tgt_pips", "sp_p90"]
        # Rename for summary
        df["best_oos_pd"] = df["oos_pd"]
        df["best_config"] = df["name"]
        print(df[["pair", "s1_pass", "mc_pass", "oos_winners",
                  "best_oos_pd", "best_config", "box_pips", "tgt_pips", "sp_p90"
                  ]].to_string(index=False))

        if len(summaries) > 1:
            print(f"\nOOS p/d ranking:")
            ranked = sorted(summaries, key=lambda x: x["oos_pd"], reverse=True)
            for r in ranked:
                icon = "🟢" if r["oos_pd"] > 5 else ("🟡" if r["oos_pd"] > 0 else "🔴")
                print(f"  {icon} {r['pair']:10s}: {r['oos_pd']:6.2f} p/d"
                      f"  box={r['box_pips']:.1f}p  tgt={r['tgt_pips']:.1f}p"
                      f"  n={r['oos_ntrd']:,}  p5={r['p5']:.2f}")


if __name__ == "__main__":
    main()
