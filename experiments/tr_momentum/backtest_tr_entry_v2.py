"""
TR momentum entry sweep v2 — spread-gated + Monte Carlo validated.

Changes from v1:
  Fix 1: Spread gate at entry (SOP R5/R6).
          sp_gate = IS P90 spread per pair. Passed as scalar into kernel.
          Only enter a signal bar when sp <= sp_gate.
  Fix 2: Monte Carlo sign-shuffle on OOS P&L after WF+OOS>0 filter.
          Gate: p_value < 0.05 (fraction of sign-shuffled sums >= real sum).
  Fix 3 (2026-05-21): Trail exit fill = bar close, not trail level.
          Live service detects trail trigger at bar close then places a market
          order — fill lands at bar-close price, not the exact trail level.
          Losses on large-range bars were 3-7p worse than backtest assumed.
          Only trail mode (ex_mode=0) affected. TP/SL exits unchanged (those
          are broker-side conditional orders that fill at the exact level).

Three-gate filter applied in order:
  1. WF: all 3 IS chunks positive, >=5 trades each
  2. OOS p/d > 0
  3. MC sign shuffle p < 0.05

Signal: if True Range > threshold_pips → enter at bar close.
Direction: FOLLOW (bar direction) or FADE (mean reversion, opposite bar).
Exits:
  mode=0  trail stop (fixed pips from high-water mark)
  mode=1  fixed TP/SL as multiples of IS-median spread

Pairs: all 12.  Data: M5 BA (5.5yr).

Run:
    cd /path/to/projects/fx-core
    python3 research/experiments/tr_momentum/backtest_tr_entry_v2.py
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import numba as nb
from numba import prange

BASE   = Path(__file__).resolve().parents[3]
BA_DIR = BASE / "data/m5_ba"

IS_FRAC    = 0.70
MAX_TRADES = 30000

PAIRS = [
    ("GBP_JPY", 0.01),
    ("USD_JPY", 0.01),
    ("EUR_JPY", 0.01),
    ("GBP_USD", 0.0001),
    ("AUD_JPY", 0.01),
    ("EUR_USD", 0.0001),
    ("CHF_JPY", 0.01),
    ("CAD_JPY", 0.01),
    ("AUD_USD", 0.0001),
    ("NZD_JPY", 0.01),
    ("NZD_USD", 0.0001),
    ("EUR_GBP", 0.0001),
]

# ── Parameter grid ────────────────────────────────────────────────────────
TR_THRESHOLDS = np.array([5, 8, 10, 12, 15, 20, 25, 30], dtype=np.float64)  # pips
TRAIL_PIPS    = np.array([2, 3, 5, 8, 10],               dtype=np.float64)  # pips (trail exit)
TP_MULTIPLES  = np.array([1.0, 1.5, 2.0, 3.0],           dtype=np.float64)  # × spread for TP
SL_MULTIPLES  = np.array([1.0, 1.5, 2.0],                dtype=np.float64)  # × spread for SL

# configs: [tr_threshold_pips, exit_mode, param_a, param_b, direction]
#   exit_mode=0: trail (param_a=trail_pips, param_b unused)
#   exit_mode=1: fixed TP/SL as multiples of spread (param_a=tp_mult, param_b=sl_mult)
#   direction=+1: follow bar, direction=-1: fade (mean reversion)
def build_configs():
    rows = []
    for tr in TR_THRESHOLDS:
        for tl in TRAIL_PIPS:
            for d in [1, -1]:
                rows.append([tr, 0, tl, 0.0, float(d)])   # trail
        for tp in TP_MULTIPLES:
            for sl in SL_MULTIPLES:
                for d in [1, -1]:
                    rows.append([tr, 1, tp, sl, float(d)]) # fixed TP/SL
    return np.array(rows, dtype=np.float64)

CONFIGS = build_configs()
N_CONFIGS = len(CONFIGS)

def config_name(row):
    tr = int(row[0]); mode = int(row[1]); pa = row[2]; pb = row[3]
    d  = "FOLLOW" if row[4] > 0 else "FADE"
    if mode == 0:
        return f"TR{tr}_trail{int(pa)}_{d}"
    else:
        return f"TR{tr}_TP{pa:.1f}x_SL{pb:.1f}x_{d}"

CONFIG_NAMES = [config_name(CONFIGS[ci]) for ci in range(N_CONFIGS)]


# ── Monte Carlo sign shuffle ──────────────────────────────────────────────
def mc_sign_test(oos_pnl: np.ndarray, n: int = 300, seed: int = 0) -> float:
    """Returns p-value: fraction of sign-shuffle sums >= real sum. Gate: p < 0.05"""
    rng = np.random.default_rng(seed)
    abs_pnl  = np.abs(oos_pnl)
    real_sum = float(oos_pnl.sum())
    count_ge = 0
    for _ in range(n):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(abs_pnl))
        if float((signs * abs_pnl).sum()) >= real_sum:
            count_ge += 1
    return count_ge / n


# ── Numba kernel ──────────────────────────────────────────────────────────
# configs cols: [tr_thresh_pips, exit_mode, param_a, param_b, d_sign]
#   exit_mode=0: trail  (param_a=trail_pips, param_b unused)
#   exit_mode=1: TP/SL  (param_a=tp_mult×med_sp, param_b=sl_mult×med_sp)
#   d_sign: +1=FOLLOW, -1=FADE
#
# v2 change: sp_gate scalar passed in. Entry blocked when sp > sp_gate.
@nb.njit(parallel=True)
def run_kernel(opens, highs, lows, closes, spreads, bar_chunks,
               configs, pip, med_sp_pip, sp_gate, is_end,
               trade_pnl, trade_chunk, trade_cnt):
    N_BARS    = len(opens)
    N_CONFIGS = configs.shape[0]

    for ci in prange(N_CONFIGS):
        tr_thr  = configs[ci, 0] * pip
        ex_mode = int(configs[ci, 1])
        param_a = configs[ci, 2]
        param_b = configs[ci, 3]
        d_sign  = configs[ci, 4]

        trail_p = param_a * pip                  # trail mode
        tp_px_d = param_a * med_sp_pip * pip     # TP mode: distance to TP
        sl_px_d = param_b * med_sp_pip * pip     # TP mode: distance to SL

        pos      = 0
        entry_px = 0.0
        hw       = 0.0
        tp_level = 0.0
        sl_level = 0.0
        t_cnt    = 0
        prev_cl  = opens[0]

        for i in range(N_BARS):
            opn = opens[i]; hi = highs[i]; lo = lows[i]; cl = closes[i]
            sp  = spreads[i]; ck = bar_chunks[i]

            tr = max(hi, prev_cl) - min(lo, prev_cl)

            # ── EXIT ─────────────────────────────────────────────────────
            if pos != 0:
                if ex_mode == 0:
                    if pos == 1:
                        if hi > hw: hw = hi
                        trail = hw - trail_p
                        if lo <= trail:
                            pnl = (cl - entry_px) / pip - sp   # bar-close fill: live places market order at bar close
                            if t_cnt < MAX_TRADES:
                                trade_pnl[ci, t_cnt]   = np.float32(pnl)
                                trade_chunk[ci, t_cnt] = ck
                                t_cnt += 1
                            pos = 0; hw = 0.0
                    else:
                        if lo < hw: hw = lo
                        trail = hw + trail_p
                        if hi >= trail:
                            pnl = (entry_px - cl) / pip - sp   # bar-close fill: live places market order at bar close
                            if t_cnt < MAX_TRADES:
                                trade_pnl[ci, t_cnt]   = np.float32(pnl)
                                trade_chunk[ci, t_cnt] = ck
                                t_cnt += 1
                            pos = 0; hw = 0.0
                else:  # fixed TP/SL
                    bull_bar = cl >= opn
                    if pos == 1:
                        hit_tp = hi >= tp_level
                        hit_sl = lo <= sl_level
                        # bull bar → assume price rose first → TP hit first
                        if hit_tp and (bull_bar or not hit_sl):
                            pnl = (tp_level - entry_px) / pip - sp
                            if t_cnt < MAX_TRADES:
                                trade_pnl[ci, t_cnt]   = np.float32(pnl)
                                trade_chunk[ci, t_cnt] = ck
                                t_cnt += 1
                            pos = 0
                        elif hit_sl:
                            pnl = (sl_level - entry_px) / pip - sp
                            if t_cnt < MAX_TRADES:
                                trade_pnl[ci, t_cnt]   = np.float32(pnl)
                                trade_chunk[ci, t_cnt] = ck
                                t_cnt += 1
                            pos = 0
                    else:  # pos == -1, TP is below entry
                        hit_tp = lo <= tp_level
                        hit_sl = hi >= sl_level
                        # bear bar → price fell → TP hit first
                        if hit_tp and (not bull_bar or not hit_sl):
                            pnl = (entry_px - tp_level) / pip - sp
                            if t_cnt < MAX_TRADES:
                                trade_pnl[ci, t_cnt]   = np.float32(pnl)
                                trade_chunk[ci, t_cnt] = ck
                                t_cnt += 1
                            pos = 0
                        elif hit_sl:
                            pnl = (entry_px - sl_level) / pip - sp
                            if t_cnt < MAX_TRADES:
                                trade_pnl[ci, t_cnt]   = np.float32(pnl)
                                trade_chunk[ci, t_cnt] = ck
                                t_cnt += 1
                            pos = 0

            # ── ENTRY ─────────────────────────────────────────────────────
            # SOP R5/R6: only enter if spread <= IS P90 gate
            if pos == 0 and tr >= tr_thr and sp <= sp_gate:
                bar_dir   = 1 if cl >= opn else -1
                direction = bar_dir if d_sign > 0 else -bar_dir
                pos       = direction
                entry_px  = cl
                if ex_mode == 0:
                    # FOLLOW: trail from bar extreme (built-in breathing room = close-to-extreme)
                    # FADE: trail from entry close (tight — fading a move, not chasing it)
                    if d_sign > 0:
                        hw = hi if direction == 1 else lo
                    else:
                        hw = entry_px
                else:
                    if direction == 1:
                        tp_level = entry_px + tp_px_d
                        sl_level = entry_px - sl_px_d
                    else:
                        tp_level = entry_px - tp_px_d
                        sl_level = entry_px + sl_px_d

            prev_cl = cl

        trade_cnt[ci] = t_cnt


# ── Per-pair evaluation ───────────────────────────────────────────────────
def run_pair(pair, pip):
    ba_path = BA_DIR / f"{pair}_M5_BA.parquet"
    if not ba_path.exists():
        return None

    ba = pd.read_parquet(ba_path)
    n  = len(ba)
    is_end   = int(n * IS_FRAC)
    oos_days = (n - is_end) / 288.0
    is_days  = is_end / 288.0

    opens   = ba["open"].values.astype(np.float64)
    highs   = ba["high"].values.astype(np.float64)
    lows    = ba["low"].values.astype(np.float64)
    closes  = ba["close"].values.astype(np.float64)
    # Spread in pips — mid OHLC for signals, explicit spread cost at entry/exit (SOP R3)
    spreads = ((ba["ask_c"] - ba["bid_c"]) / pip).values.astype(np.float64)

    # SOP R5: sp_gate computed on IS data only — never touch OOS spread distribution
    sp_gate = float(np.percentile(spreads[:is_end], 90))

    c0e = is_end // 3; c1e = 2 * (is_end // 3)
    bar_chunks = np.zeros(n, dtype=np.int8)
    bar_chunks[c0e:c1e] = 1; bar_chunks[c1e:is_end] = 2; bar_chunks[is_end:] = 3

    trade_pnl   = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.float32)
    trade_chunk = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.int8)
    trade_cnt   = np.zeros(N_CONFIGS, dtype=np.int32)

    med_sp_pip = float(np.median(spreads[:is_end]))   # IS median spread (pips)

    t0 = time.time()
    run_kernel(opens, highs, lows, closes, spreads, bar_chunks,
               CONFIGS, pip, med_sp_pip, sp_gate, is_end,
               trade_pnl, trade_chunk, trade_cnt)
    elapsed = time.time() - t0

    n_configs_tested = N_CONFIGS

    # ── Gate 1: Walk-forward — all 3 IS chunks positive, >=5 trades each ─
    wf_survivors = []
    for ci in range(N_CONFIGS):
        tc = trade_cnt[ci]
        if tc == 0: continue
        pnl = trade_pnl[ci, :tc].astype(np.float64)
        ck  = trade_chunk[ci, :tc].astype(np.int32)

        wf_pass = True
        for chunk in range(3):
            cp = pnl[ck == chunk]
            if len(cp) < 5 or cp.sum() <= 0:
                wf_pass = False; break
        if not wf_pass: continue
        wf_survivors.append(ci)

    # ── Gate 2: OOS p/d > 0 ──────────────────────────────────────────────
    oos_survivors = []
    for ci in wf_survivors:
        tc   = trade_cnt[ci]
        pnl  = trade_pnl[ci, :tc].astype(np.float64)
        ck   = trade_chunk[ci, :tc].astype(np.int32)
        oos_mask = (ck == 3)
        oos_pnl  = pnl[oos_mask]
        oos_pd   = float(oos_pnl.sum()) / oos_days if oos_days > 0 else 0.0
        if oos_pd > 0:
            oos_survivors.append(ci)

    # ── Gate 3: MC sign shuffle p < 0.05 ────────────────────────────────
    winners = []
    for idx, ci in enumerate(oos_survivors):
        tc   = trade_cnt[ci]
        pnl  = trade_pnl[ci, :tc].astype(np.float64)
        ck   = trade_chunk[ci, :tc].astype(np.int32)

        oos_mask = (ck == 3)
        oos_pnl  = pnl[oos_mask]
        oos_ntrd = int(oos_mask.sum())
        oos_pd   = float(oos_pnl.sum()) / oos_days if oos_days > 0 else 0.0
        is_pd    = float(pnl[ck < 3].sum()) / is_days
        oos_wr   = float((oos_pnl > 0).sum()) / oos_ntrd * 100 if oos_ntrd > 0 else 0.0

        # MC uses config index as seed for reproducibility
        p_val = mc_sign_test(oos_pnl, n=300, seed=ci)
        if p_val >= 0.05:
            continue

        winners.append({
            "ci":        ci,
            "name":      CONFIG_NAMES[ci],
            "direction": "FOLLOW" if CONFIGS[ci, 4] > 0 else "FADE",
            "is_pd":     round(is_pd, 1),
            "oos_pd":    round(oos_pd, 1),
            "oos_ntrd":  oos_ntrd,
            "oos_wr":    round(oos_wr, 1),
            "mc_p":      round(p_val, 4),
            "sp_gate":   round(sp_gate, 4),
        })

    winners.sort(key=lambda x: x["oos_pd"], reverse=True)
    return {
        "winners":       winners,
        "n_tested":      n_configs_tested,
        "n_wf":          len(wf_survivors),
        "n_oos":         len(oos_survivors),
        "n_mc":          len(winners),
        "elapsed":       elapsed,
        "oos_days":      oos_days,
        "sp_gate":       round(sp_gate, 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────
n_trail = len(TR_THRESHOLDS) * len(TRAIL_PIPS) * 2
n_tpsl  = len(TR_THRESHOLDS) * len(TP_MULTIPLES) * len(SL_MULTIPLES) * 2
print("TR momentum entry sweep v2 — spread-gated + Monte Carlo validated")
print("FOLLOW + FADE × trail + TP/SL exits")
print(f"Configs: {N_CONFIGS}  (trail={n_trail}, TP/SL={n_tpsl})")
print(f"TR thresh: {list(TR_THRESHOLDS.astype(int))}p")
print(f"Trail pips: {list(TRAIL_PIPS.astype(int))}p  |  "
      f"TP×: {list(TP_MULTIPLES)}  SL×: {list(SL_MULTIPLES)}")
print(f"Gates: (1) WF all-chunks-positive >=5t  (2) OOS p/d > 0  (3) MC p < 0.05")
print("="*75)

# JIT warmup
dummy    = np.ones(500, np.float64)
dummy_sp = np.zeros(500, np.float64)
dummy_ck = np.zeros(500, np.int8)
_tp  = np.zeros((1, MAX_TRADES), np.float32)
_tc  = np.zeros((1, MAX_TRADES), np.int8)
_tn  = np.zeros(1, np.int32)
_cfg = np.array([[10.0, 0.0, 5.0, 0.0, 1.0]], np.float64)
print("JIT warmup...", end=" ", flush=True)
t0 = time.time()
run_kernel(dummy, dummy, dummy, dummy, dummy_sp, dummy_ck,
           _cfg, 0.01, 0.0, 1.0, 350, _tp, _tc, _tn)
print(f"done in {time.time()-t0:.1f}s\n")

summary = []
for pair, pip in PAIRS:
    print(f"── {pair} {'─'*(60-len(pair))}")
    result = run_pair(pair, pip)
    if result is None:
        print(f"  no data\n")
        summary.append((pair, 0, 0, 0, 0.0, "—", None))
        continue

    n_tested = result["n_tested"]
    n_wf     = result["n_wf"]
    n_oos    = result["n_oos"]
    n_mc     = result["n_mc"]
    elapsed  = result["elapsed"]
    sp_gate  = result["sp_gate"]
    winners  = result["winners"]

    print(f"  Configs tested: {n_tested}  |  WF survivors: {n_wf}"
          f"  |  OOS>0: {n_oos}  |  MC pass: {n_mc}"
          f"  ({elapsed:.1f}s)")
    print(f"  sp_gate (IS P90): {sp_gate:.4f} pips")

    if not winners:
        print(f"  No MC-passing configs.\n")
        summary.append((pair, n_wf, n_oos, 0, 0.0, "—", sp_gate))
        continue

    # Top 10 MC-passing configs
    top10 = winners[:10]
    hdr = f"  {'Config':<35} {'Dir':<7} {'IS p/d':>7} {'OOS p/d':>8} {'WR%':>5} {'OOS t':>6} {'MC p':>7} {'sp_gate':>8}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for w in top10:
        flag = "🟢" if w["mc_p"] < 0.01 else "🟡"
        print(f"  {flag} {w['name']:<33} {w['direction']:<7}"
              f" {w['is_pd']:>7.1f} {w['oos_pd']:>8.1f}"
              f" {w['oos_wr']:>5.1f} {w['oos_ntrd']:>6}"
              f" {w['mc_p']:>7.4f} {w['sp_gate']:>8.4f}")

    best = winners[0]
    summary.append((pair, n_wf, n_oos, n_mc, best["oos_pd"], best["name"], sp_gate))
    print()

print("\n\n── Summary (all pairs) " + "─"*52)
print(f"  {'Pair':<10} {'WF_pass':>8} {'OOS_pos':>8} {'MC_pass':>8}"
      f" {'best_oos_pd':>12} {'best_mc_p':>9} {'sp_gate':>8}  Best config")
print("  " + "─"*110)
for pair, n_wf, n_oos, n_mc, best_pd, best_cfg, sp_gate in summary:
    flag = "🟢" if n_mc > 0 else "🔴"
    sp_str  = f"{sp_gate:.4f}" if sp_gate is not None else "—"
    pd_str  = f"{best_pd:+.1f}" if isinstance(best_pd, float) else "—"
    print(f"  {flag} {pair:<10} {n_wf:>8} {n_oos:>8} {n_mc:>8}"
          f" {pd_str:>12} {'—':>9} {sp_str:>8}  {best_cfg}")

# sp_gate reference table for live service hardcoding
print("\n\n── sp_gate values for live service config " + "─"*32)
print("  (Computed from IS P90 spread — hardcode these into paper/live services)")
print(f"  {'Pair':<12} {'pip':>8} {'sp_gate (pips)':>16}")
print("  " + "─"*40)
for pair, n_wf, n_oos, n_mc, best_pd, best_cfg, sp_gate in summary:
    pip_val = dict(PAIRS).get(pair, None)
    pip_str = f"{pip_val}" if pip_val is not None else "?"
    sp_str  = f"{sp_gate:.4f}" if sp_gate is not None else "no data"
    print(f"  {pair:<12} {pip_str:>8} {sp_str:>16}")

print("\nDone.")
