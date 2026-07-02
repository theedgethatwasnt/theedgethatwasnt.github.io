#!/usr/bin/env python3
"""Validate IncrementalTopsBots on real EUR_JPY range bar data.

CORRECT INVARIANT
-----------------
IncrementalTopsBots is NOT a simple 1-bar shift of the batch algorithm.
In Stage 2, act_h only updates when an H run ENDS (first opposing L arrives).
Batch updates act_h at the H swing's own index (using bar i+1 as lookahead).

The correct thing to verify:
  1. SWING LIST IDENTITY: the set of confirmed swings (type, value) produced
     by incremental == produced by batch — same values, same order, no misses
  2. VALUE CORRECTNESS: when incremental has a non-NaN act_h, it equals the
     most recently batch-confirmed HSP up to that point
  3. LAG DISTRIBUTION: how many bars behind is incremental vs batch
     (characterizes how often the network sees stale vs current S/R)
  4. CONVERGENCE: after finalize(), inc act_h/act_l == batch final act_h/act_l
  5. IncrementalZscore matches batch sliding-window formula on real M5 data

The key property for V2 is training/live PARITY, not batch match.
If train + live both use IncrementalTopsBots, they always agree exactly.

Run from repo root:
    python3 research/experiments/validate_incremental_realdata.py
"""
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.incremental_topsbots import IncrementalTopsBots
from lib.incremental_features import IncrementalZscore
from lib.swing_indicators import compute_swing_features, topsbots_swings

RANGE_DIR = ROOT / "data" / "range_bar_causal"
S5_DIR    = ROOT / "data" / "s5_ohlc"
PAIRS = ["EUR_JPY", "GBP_JPY", "USD_JPY", "EUR_USD", "GBP_USD",
         "AUD_JPY", "CAD_JPY", "NZD_USD", "AUD_USD", "CHF_JPY",
         "NZD_JPY", "EUR_GBP"]


def validate_sba(pair: str):
    fpath = RANGE_DIR / f"{pair}_range10_causal.parquet"
    if not fpath.exists():
        print(f"  🟡 SKIP {pair}: no parquet")
        return None

    df = pd.read_parquet(fpath)
    close = df["mid_close"].values.astype(np.float64)
    N = len(close)

    # ── Batch: full history ──────────────────────────────────────────────
    sig_batch = topsbots_swings(close, close)
    batch_swings_vals = [(t, v) for _, t, v in sig_batch]   # (type, value) ordered

    # act_h/act_l at every bar from batch
    state_b, erp_b, act_h_b, act_l_b, _ = compute_swing_features(close, close, close)

    # ── Incremental ──────────────────────────────────────────────────────
    tb = IncrementalTopsBots()
    inc_act_h = np.full(N, np.nan)
    inc_act_l = np.full(N, np.nan)
    inc_swing_vals = []   # (type, value) in order of first appearance in act_h/act_l
    prev_ah = prev_al = None

    for i in range(N):
        _, _, ah, al = tb.update(close[i], close[i], close[i])
        ah_f = ah if ah is not None else np.nan
        al_f = al if al is not None else np.nan
        inc_act_h[i] = ah_f
        inc_act_l[i] = al_f
        # Track new HSP/LSP changes
        if ah is not None and ah != prev_ah:
            inc_swing_vals.append(('H', ah))
            prev_ah = ah
        if al is not None and al != prev_al:
            inc_swing_vals.append(('L', al))
            prev_al = al

    tb.finalize()
    final_ah = tb.act_h
    final_al = tb.act_l

    # ── Test 1: swing list identity ──────────────────────────────────────
    # The incremental may include extra entries because when a H run is flushed
    # as a batch unit, the incremental emits each individual step.
    # What we really want: every value in batch_swings_vals appears in inc_swing_vals
    # in the SAME ORDER. The incremental may have MORE entries (redundant same-type
    # updates if pending gets updated), but should not MISS any batch swing.
    # For a correct Stage2+3 implementation with no ambiguity, the lists should match.
    lists_match = batch_swings_vals == inc_swing_vals

    # ── Test 2: convergence after finalize() ─────────────────────────────
    batch_final_ah = float(act_h_b[-1]) if not np.isnan(act_h_b[-1]) else None
    batch_final_al = float(act_l_b[-1]) if not np.isnan(act_l_b[-1]) else None
    conv_h = (final_ah == batch_final_ah) or (final_ah is None and batch_final_ah is None)
    conv_l = (final_al == batch_final_al) or (final_al is None and batch_final_al is None)

    # ── Test 3: value correctness when inc has a value ────────────────────
    # For every bar where inc_act_h is not NaN, it should equal the LAST
    # batch-confirmed HSP up to (but not past) that bar.
    # Build "batch_running_ah": the most recently accepted HSP in batch at each bar
    # (same as act_h_b but derived from sig_batch for clarity)
    batch_running_ah = np.full(N, np.nan)
    batch_running_al = np.full(N, np.nan)
    cur_bah = cur_bal = np.nan
    sw_map = {idx: (t, v) for idx, t, v in sig_batch}
    for i in range(N):
        if i in sw_map:
            t, v = sw_map[i]
            if t == 'H': cur_bah = v
            else:        cur_bal = v
        batch_running_ah[i] = cur_bah
        batch_running_al[i] = cur_bal

    # Where incremental has a value, it must match the batch running value
    inc_has_h = ~np.isnan(inc_act_h)
    value_err_h = np.where(inc_has_h, np.abs(inc_act_h - batch_running_ah), 0.0)
    max_val_err_h = float(np.max(value_err_h[inc_has_h])) if inc_has_h.any() else 0.0

    inc_has_l = ~np.isnan(inc_act_l)
    value_err_l = np.where(inc_has_l, np.abs(inc_act_l - batch_running_al), 0.0)
    max_val_err_l = float(np.max(value_err_l[inc_has_l])) if inc_has_l.any() else 0.0

    # ── Test 4: lag distribution ──────────────────────────────────────────
    # For each bar in batch where act_h changes, find how many bars later
    # the incremental catches up.
    lags = []
    for idx, t, v in sig_batch:
        if t == 'H':
            # Find first i >= idx where inc_act_h[i] == v
            found = np.where((inc_act_h[idx:] == v))[0]
            if len(found):
                lags.append(found[0])
    median_lag = float(np.median(lags)) if lags else float('nan')
    p90_lag    = float(np.percentile(lags, 90)) if lags else float('nan')
    max_lag    = float(np.max(lags)) if lags else float('nan')

    # ── Summary ───────────────────────────────────────────────────────────
    pass_flag = (lists_match and conv_h and conv_l
                 and max_val_err_h < 1e-9 and max_val_err_l < 1e-9)
    icon = "🟢" if pass_flag else "🔴"

    print(f"  {icon} {pair:10s}  N={N:6,}  "
          f"lists={'✓' if lists_match else '✗'}  "
          f"conv={'✓' if (conv_h and conv_l) else '✗'}  "
          f"val_err_h={max_val_err_h:.1e}  "
          f"lag_med={median_lag:.0f}  lag_p90={p90_lag:.0f}  lag_max={max_lag:.0f}  "
          f"n_swings={len(batch_swings_vals)}")

    if not lists_match:
        # Show first difference
        for i, (b, inc_v) in enumerate(zip(batch_swings_vals[:20],
                                           inc_swing_vals[:20])):
            if b != inc_v:
                print(f"    First diff at position {i}: batch={b}  inc={inc_v}")
                break

    return bool(pass_flag)


def validate_zscore(pair: str = "EUR_JPY", pop: int = 1000):
    fpath = S5_DIR / f"{pair}_S5_BA.parquet"
    if not fpath.exists():
        print(f"  🟡 SKIP zscore: no S5 data for {pair}")
        return None

    df = pd.read_parquet(fpath)
    df.columns = [c.lower() for c in df.columns]
    bid_c = df["bid_c"].values.astype(np.float64)
    n_m5 = len(bid_c) // 60
    m5_c = bid_c[:n_m5 * 60].reshape(n_m5, 60)[:, -1]
    N = len(m5_c)

    PIP = 0.01
    r = np.zeros(N)
    r[1:] = (m5_c[1:] - m5_c[:-1]) / PIP

    HP = math.pi / 2.0
    batch_f = np.zeros(N)
    s = ss = 0.0
    for k in range(pop):
        s += r[k]; ss += r[k] ** 2
    for i in range(pop, N):
        v   = ss / pop - (s / pop) ** 2
        std = v ** 0.5 if v > 1e-20 else 1e-10
        batch_f[i] = math.atan(r[i] / std) / HP
        s  += r[i]       - r[i - pop]
        ss += r[i] ** 2  - r[i - pop] ** 2

    iz = IncrementalZscore(pop=pop)
    inc_f = np.array([iz.update(r[i]) for i in range(N)])

    max_err = float(np.max(np.abs(batch_f[pop:] - inc_f[pop:])))
    corr    = float(np.corrcoef(batch_f[pop:], inc_f[pop:])[0, 1])
    sign_ag = float(np.mean(np.sign(batch_f[pop:]) == np.sign(inc_f[pop:])) * 100)
    ok = max_err < 1e-9

    icon = "🟢" if ok else "🔴"
    print(f"  {icon} zscore ({pair}, pop={pop}):  "
          f"max_err={max_err:.2e}  corr={corr:.8f}  sign_agree={sign_ag:.2f}%  N={N:,}")
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────

print("=" * 78)
print("SBA IncrementalTopsBots correctness (real range bar data)")
print()
print("  lists=✓  : confirmed swing (type, value) list matches batch")
print("  conv=✓   : final act_h/act_l after finalize() == batch final")
print("  val_err  : max |inc_act_h - batch_running_ah| when inc has a value")
print("  lag_*    : bars behind batch when each new HSP appears in inc_act_h")
print("=" * 78)

results = {}
for pair in PAIRS:
    fpath = RANGE_DIR / f"{pair}_range10_causal.parquet"
    if fpath.exists():
        results[pair] = validate_sba(pair)

n_pass = sum(1 for v in results.values() if v is True)
n_fail = sum(1 for v in results.values() if v is False)
print(f"\nSBA summary: {n_pass} pass / {n_fail} fail / {len(PAIRS)-len(results)} skipped")

print("\n" + "=" * 78)
print("IncrementalZscore on real M5 data (identical to batch formula)")
print("=" * 78)
zok = validate_zscore("EUR_JPY", pop=1000)

print("\n" + "=" * 78)
overall = (n_fail == 0) and (zok is not False)
if overall:
    print("🟢 ALL PASS — IncrementalTopsBots is correct.")
    print("   Lag of 1–50 bars is expected and correct (Stage 2 requires run to end).")
    print("   For V2: use IncrementalTopsBots in BOTH train and live → perfect parity.")
else:
    print("🔴 ISSUES FOUND — fix before retraining.")
print("=" * 78)
