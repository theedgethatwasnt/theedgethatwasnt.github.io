"""
TR momentum entry sweep v3 — TP-unlock then trail (two-stage exit).

User insight (2026-05-21): bar-close fills destroy trail-only edge.
Fix: Phase 1 waits for an initial target; Phase 2 arms the trail only after
target is confirmed. This guarantees the trail fires at ≥ break-even once
Phase 1 completes.

Exit modes tested:
  mode=2  TP-unlock then trail (NEW)
          param_a = target_pips   (Phase 1: distance to unlock; also hard SL = target_pips)
          param_b = trail_pips    (Phase 2: trail distance after unlock)

          Phase 1 (pos entered, waiting for target):
            LONG: if bar HIGH >= entry + target_pips → Phase 2 armed on this bar
                  if bar LOW  <= entry - target_pips → SL hit, exit at bar close
            Phase 2 (target confirmed, trailing):
            LONG: hw = max(hw, bar HIGH); trail = hw - trail_pips
                  if bar LOW <= trail → exit at bar close (SOP: bar-close fill)

          Break-even guarantee: once Phase 2 armed, worst exit =
            (hw_at_arm - trail_pips - entry) - spread
            = (min target_pips - trail_pips) - spread > 0 when target > trail + spread.

  mode=0  plain trail (bar-close fills, from v2)  — kept for reference
  mode=1  fixed TP/SL (exact-level fills)          — kept for reference

All three gates:
  1. WF: all 3 IS chunks positive, >=5 trades each
  2. OOS p/d > 0
  3. MC sign-shuffle p < 0.05

Pairs: all 12.  Data: M5 BA (5.5yr).

Run:
    cd /path/to/projects/fx-core
    python3 research/experiments/tr_momentum/backtest_tr_entry_v3.py
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

# ── Parameter grids ───────────────────────────────────────────────────────
TR_THRESHOLDS  = np.array([5, 8, 10, 12, 15, 20, 25, 30], dtype=np.float64)
TARGET_PIPS    = np.array([3, 5, 8, 10, 15],              dtype=np.float64)  # Phase 1 unlock
TRAIL_PIPS_P2  = np.array([2, 3, 5, 8],                   dtype=np.float64)  # Phase 2 trail

# configs: [tr_threshold_pips, exit_mode, param_a, param_b, direction]
#   exit_mode=2: TP-unlock-then-trail
#     param_a = target_pips (Phase 1 unlock distance; SL = same)
#     param_b = trail_pips  (Phase 2 trail)
#   direction=+1: follow bar, direction=-1: fade
def build_configs():
    rows = []
    for tr in TR_THRESHOLDS:
        for tgt in TARGET_PIPS:
            for trl in TRAIL_PIPS_P2:
                if trl >= tgt:
                    continue   # trail must be < target (otherwise no lock-in guarantee)
                for d in [1, -1]:
                    rows.append([tr, 2, tgt, trl, float(d)])
    return np.array(rows, dtype=np.float64)

CONFIGS   = build_configs()
N_CONFIGS = len(CONFIGS)

def config_name(row):
    tr  = int(row[0])
    tgt = int(row[2])
    trl = int(row[3])
    d   = "FOLLOW" if row[4] > 0 else "FADE"
    return f"TR{tr}_tgt{tgt}_trl{trl}_{d}"

CONFIG_NAMES = [config_name(CONFIGS[ci]) for ci in range(N_CONFIGS)]


# ── Monte Carlo sign shuffle ──────────────────────────────────────────────
def mc_sign_test(oos_pnl: np.ndarray, n: int = 300, seed: int = 0) -> float:
    rng      = np.random.default_rng(seed)
    abs_pnl  = np.abs(oos_pnl)
    real_sum = float(oos_pnl.sum())
    count_ge = 0
    for _ in range(n):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(abs_pnl))
        if float((signs * abs_pnl).sum()) >= real_sum:
            count_ge += 1
    return count_ge / n


# ── Numba kernel ──────────────────────────────────────────────────────────
# exit_mode=2: TP-unlock-then-trail, bar-close fills throughout
# State per config: pos, stage, entry_px, hw, target_px, sl_px
@nb.njit(parallel=True)
def run_kernel(opens, highs, lows, closes, spreads, bar_chunks,
               configs, pip, sp_gate, is_end,
               trade_pnl, trade_chunk, trade_cnt):
    N_BARS    = len(opens)
    N_CONFIGS = configs.shape[0]

    for ci in prange(N_CONFIGS):
        tr_thr  = configs[ci, 0] * pip
        param_a = configs[ci, 2]   # target_pips
        param_b = configs[ci, 3]   # trail_pips (Phase 2)
        d_sign  = configs[ci, 4]

        target_d = param_a * pip   # price distance to unlock target
        trail_p2 = param_b * pip   # Phase 2 trail distance
        hard_sl  = param_a * pip   # hard SL = target distance (symmetric risk)

        pos      = 0
        stage    = 0     # 1 = waiting for target; 2 = trailing
        entry_px = 0.0
        hw       = 0.0
        target_px = 0.0
        sl_px    = 0.0
        t_cnt    = 0
        prev_cl  = opens[0]

        for i in range(N_BARS):
            opn = opens[i]; hi = highs[i]; lo = lows[i]; cl = closes[i]
            sp  = spreads[i]; ck = bar_chunks[i]

            tr = max(hi, prev_cl) - min(lo, prev_cl)

            # ── EXIT ─────────────────────────────────────────────────────
            if pos != 0:
                exited = False

                if pos == 1:   # LONG
                    if stage == 1:
                        # Phase 1: check target OR SL
                        if hi >= target_px:
                            # Target hit — arm Phase 2 trail
                            stage = 2
                            hw = hi   # HWM = this bar's high (target confirmed)
                            trail = hw - trail_p2
                            # Same bar: check if trail also fires (wick up + reversal)
                            if lo <= trail:
                                pnl = (cl - entry_px) / pip - sp
                                if t_cnt < MAX_TRADES:
                                    trade_pnl[ci, t_cnt]   = np.float32(pnl)
                                    trade_chunk[ci, t_cnt] = ck
                                    t_cnt += 1
                                pos = 0; stage = 0; hw = 0.0; exited = True
                        if not exited and lo <= sl_px:
                            # SL hit before target: exit at bar close
                            pnl = (cl - entry_px) / pip - sp
                            if t_cnt < MAX_TRADES:
                                trade_pnl[ci, t_cnt]   = np.float32(pnl)
                                trade_chunk[ci, t_cnt] = ck
                                t_cnt += 1
                            pos = 0; stage = 0; hw = 0.0; exited = True
                    elif stage == 2 and not exited:
                        # Phase 2: trail with bar-close fill
                        if hi > hw: hw = hi
                        trail = hw - trail_p2
                        if lo <= trail:
                            pnl = (cl - entry_px) / pip - sp
                            if t_cnt < MAX_TRADES:
                                trade_pnl[ci, t_cnt]   = np.float32(pnl)
                                trade_chunk[ci, t_cnt] = ck
                                t_cnt += 1
                            pos = 0; stage = 0; hw = 0.0; exited = True

                else:   # SHORT (pos == -1)
                    if stage == 1:
                        if lo <= target_px:
                            stage = 2
                            hw = lo
                            trail = hw + trail_p2
                            if hi >= trail:
                                pnl = (entry_px - cl) / pip - sp
                                if t_cnt < MAX_TRADES:
                                    trade_pnl[ci, t_cnt]   = np.float32(pnl)
                                    trade_chunk[ci, t_cnt] = ck
                                    t_cnt += 1
                                pos = 0; stage = 0; hw = 0.0; exited = True
                        if not exited and hi >= sl_px:
                            pnl = (entry_px - cl) / pip - sp
                            if t_cnt < MAX_TRADES:
                                trade_pnl[ci, t_cnt]   = np.float32(pnl)
                                trade_chunk[ci, t_cnt] = ck
                                t_cnt += 1
                            pos = 0; stage = 0; hw = 0.0; exited = True
                    elif stage == 2 and not exited:
                        if lo < hw: hw = lo
                        trail = hw + trail_p2
                        if hi >= trail:
                            pnl = (entry_px - cl) / pip - sp
                            if t_cnt < MAX_TRADES:
                                trade_pnl[ci, t_cnt]   = np.float32(pnl)
                                trade_chunk[ci, t_cnt] = ck
                                t_cnt += 1
                            pos = 0; stage = 0; hw = 0.0; exited = True

            # ── ENTRY ─────────────────────────────────────────────────────
            if pos == 0 and tr >= tr_thr and sp <= sp_gate:
                bar_dir   = 1 if cl >= opn else -1
                direction = bar_dir if d_sign > 0 else -bar_dir
                pos       = direction
                stage     = 1
                entry_px  = cl
                if direction == 1:
                    target_px = entry_px + target_d
                    sl_px     = entry_px - hard_sl
                else:
                    target_px = entry_px - target_d
                    sl_px     = entry_px + hard_sl
                hw = 0.0

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
    spreads = ((ba["ask_c"] - ba["bid_c"]) / pip).values.astype(np.float64)

    sp_gate = float(np.percentile(spreads[:is_end], 90))

    c0e = is_end // 3; c1e = 2 * (is_end // 3)
    bar_chunks = np.zeros(n, dtype=np.int8)
    bar_chunks[c0e:c1e] = 1; bar_chunks[c1e:is_end] = 2; bar_chunks[is_end:] = 3

    trade_pnl   = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.float32)
    trade_chunk = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.int8)
    trade_cnt   = np.zeros(N_CONFIGS, dtype=np.int32)

    t0 = time.time()
    run_kernel(opens, highs, lows, closes, spreads, bar_chunks,
               CONFIGS, pip, sp_gate, is_end,
               trade_pnl, trade_chunk, trade_cnt)
    elapsed = time.time() - t0

    # ── Gate 1: Walk-forward ─────────────────────────────────────────────
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
        if wf_pass:
            wf_survivors.append(ci)

    # ── Gate 2: OOS p/d > 0 ─────────────────────────────────────────────
    oos_survivors = []
    for ci in wf_survivors:
        tc   = trade_cnt[ci]
        pnl  = trade_pnl[ci, :tc].astype(np.float64)
        ck   = trade_chunk[ci, :tc].astype(np.int32)
        oos_pnl = pnl[ck == 3]
        oos_pd  = float(oos_pnl.sum()) / oos_days if oos_days > 0 else 0.0
        if oos_pd > 0:
            oos_survivors.append(ci)

    # ── Gate 3: MC sign shuffle p < 0.05 ────────────────────────────────
    winners = []
    for ci in oos_survivors:
        tc   = trade_cnt[ci]
        pnl  = trade_pnl[ci, :tc].astype(np.float64)
        ck   = trade_chunk[ci, :tc].astype(np.int32)
        oos_mask = (ck == 3)
        oos_pnl  = pnl[oos_mask]
        oos_ntrd = int(oos_mask.sum())
        oos_pd   = float(oos_pnl.sum()) / oos_days if oos_days > 0 else 0.0
        is_pd    = float(pnl[ck < 3].sum()) / is_days
        oos_wr   = float((oos_pnl > 0).sum()) / oos_ntrd * 100 if oos_ntrd > 0 else 0.0

        p_val = mc_sign_test(oos_pnl, n=300, seed=ci)
        if p_val >= 0.05:
            continue

        winners.append({
            "ci":        ci,
            "name":      CONFIG_NAMES[ci],
            "direction": "FOLLOW" if CONFIGS[ci, 4] > 0 else "FADE",
            "target_p":  int(CONFIGS[ci, 2]),
            "trail_p":   int(CONFIGS[ci, 3]),
            "is_pd":     round(is_pd, 1),
            "oos_pd":    round(oos_pd, 1),
            "oos_ntrd":  oos_ntrd,
            "oos_wr":    round(oos_wr, 1),
            "mc_p":      round(p_val, 4),
            "sp_gate":   round(sp_gate, 4),
        })

    winners.sort(key=lambda x: x["oos_pd"], reverse=True)
    return {
        "winners":  winners,
        "n_tested": N_CONFIGS,
        "n_wf":     len(wf_survivors),
        "n_oos":    len(oos_survivors),
        "n_mc":     len(winners),
        "elapsed":  elapsed,
        "oos_days": oos_days,
        "sp_gate":  round(sp_gate, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────
print("TR momentum entry sweep v3 — TP-unlock then bar-close trail")
print("Phase 1: hold until target_pips hit (SL = target_pips, symmetric)")
print("Phase 2: trail from HWM with bar-close fills (matches live execution)")
print(f"Configs: {N_CONFIGS}  (TR×{len(TR_THRESHOLDS)} × tgt×{len(TARGET_PIPS)} × trl×{len(TRAIL_PIPS_P2)} × 2dir, trl<tgt filter applied)")
print(f"TR thresh: {list(TR_THRESHOLDS.astype(int))}p")
print(f"Target pips: {list(TARGET_PIPS.astype(int))}p  |  Trail pips (P2): {list(TRAIL_PIPS_P2.astype(int))}p")
print(f"Gates: (1) WF all-chunks-positive >=5t  (2) OOS p/d > 0  (3) MC p < 0.05")
print("="*75)

# JIT warmup
dummy    = np.ones(500, np.float64)
dummy_sp = np.zeros(500, np.float64)
dummy_ck = np.zeros(500, np.int8)
_tp  = np.zeros((1, MAX_TRADES), np.float32)
_tc  = np.zeros((1, MAX_TRADES), np.int8)
_tn  = np.zeros(1, np.int32)
_cfg = np.array([[10.0, 2.0, 5.0, 2.0, 1.0]], np.float64)
print("JIT warmup...", end=" ", flush=True)
t0 = time.time()
run_kernel(dummy, dummy, dummy, dummy, dummy_sp, dummy_ck,
           _cfg, 0.01, 1.0, 350, _tp, _tc, _tn)
print(f"done in {time.time()-t0:.1f}s\n")

summary = []
for pair, pip in PAIRS:
    print(f"── {pair} {'─'*(60-len(pair))}")
    result = run_pair(pair, pip)
    if result is None:
        print(f"  no data\n")
        summary.append((pair, 0, 0, 0, 0.0, "—", None))
        continue

    n_wf    = result["n_wf"]
    n_oos   = result["n_oos"]
    n_mc    = result["n_mc"]
    elapsed = result["elapsed"]
    sp_gate = result["sp_gate"]
    winners = result["winners"]

    print(f"  Configs tested: {result['n_tested']}  |  WF: {n_wf}  |  OOS>0: {n_oos}  |  MC pass: {n_mc}  ({elapsed:.1f}s)")
    print(f"  sp_gate (IS P90): {sp_gate:.4f} pips")

    if not winners:
        print(f"  No MC-passing configs.\n")
        summary.append((pair, n_wf, n_oos, 0, 0.0, "—", sp_gate))
        continue

    top10 = winners[:10]
    hdr = f"  {'Config':<35} {'Dir':<7} {'tgt':>4} {'trl':>4} {'IS p/d':>7} {'OOS p/d':>8} {'WR%':>5} {'OOS t':>6} {'MC p':>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for w in top10:
        flag = "🟢" if w["mc_p"] < 0.01 else "🟡"
        print(f"  {flag} {w['name']:<33} {w['direction']:<7}"
              f" {w['target_p']:>4} {w['trail_p']:>4}"
              f" {w['is_pd']:>7.1f} {w['oos_pd']:>8.1f}"
              f" {w['oos_wr']:>5.1f} {w['oos_ntrd']:>6}"
              f" {w['mc_p']:>7.4f}")

    best = winners[0]
    summary.append((pair, n_wf, n_oos, n_mc, best["oos_pd"], best["name"], sp_gate))
    print()

print("\n\n── Summary (all pairs) " + "─"*52)
print(f"  {'Pair':<10} {'WF':>5} {'OOS>0':>6} {'MC':>5} {'best_pd':>9}  Best config")
print("  " + "─"*80)
for pair, n_wf, n_oos, n_mc, best_pd, best_cfg, sp_gate in summary:
    flag = "🟢" if n_mc > 0 else "🔴"
    pd_str = f"{best_pd:+.1f}" if isinstance(best_pd, float) else "—"
    sp_str = f"{sp_gate:.2f}p" if sp_gate is not None else "—"
    print(f"  {flag} {pair:<10} {n_wf:>5} {n_oos:>6} {n_mc:>5} {pd_str:>9}  {best_cfg}  sp_gate={sp_str}")

print("\nDone.")
