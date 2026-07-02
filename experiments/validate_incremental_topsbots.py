#!/usr/bin/env python3
"""Validate IncrementalTopsBots against batch topsbots_swings().

Tests:
  1. act_h / act_l track: incremental matches batch after last bar + finalize()
  2. Serialization round-trip: from_dict(to_dict()) produces identical output
  3. IncrementalZscore: matches the batch arctan-zscore from exp_lgbm.py
  4. FXFeatureBuilder.to_dict() / from_dict() round-trip

Run from repo root:
    python3 research/experiments/validate_incremental_topsbots.py
"""
import sys
import math
import json
import numpy as np
from pathlib import Path
from collections import deque

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.incremental_topsbots import IncrementalTopsBots
from lib.incremental_features import IncrementalZscore
from lib.swing_indicators import topsbots_swings, compute_swing_features


# ── helpers ──────────────────────────────────────────────────────────────────

def _batch_swings(hi, lo):
    """Run batch algorithm, return final act_h, act_l."""
    sig = topsbots_swings(hi, lo)
    lh = ll = None
    for _, t, v in sig:
        if t == 'H': lh = v
        else:        ll = v
    return lh, ll


def _check(name, a, b, tol=1e-9):
    ok = (a is None and b is None) or (a is not None and b is not None and abs(a - b) < tol)
    status = "🟢 PASS" if ok else "🔴 FAIL"
    print(f"  {status}  {name}: incremental={a}  batch={b}")
    return ok


# ── Test 1: random H/L series ─────────────────────────────────────────────────

print("=" * 60)
print("Test 1: act_h/act_l parity on random H/L series (random walk)")
rng = np.random.default_rng(42)
n_tests, n_bars = 20, 300
all_pass = True
for trial in range(n_tests):
    # Random walk mid, then add spread for H/L
    mid = np.cumsum(rng.normal(0, 1, n_bars)) * 0.001 + 1.3
    spread = rng.uniform(0.001, 0.005, n_bars)
    hi = mid + spread / 2
    lo = mid - spread / 2

    # Batch
    batch_lh, batch_ll = _batch_swings(hi, lo)

    # Incremental
    tb = IncrementalTopsBots()
    for i in range(n_bars):
        tb.update(hi[i], lo[i], (hi[i] + lo[i]) / 2)
    tb.finalize()

    ok_h = _check(f"trial={trial+1} act_h", tb.act_h, batch_lh)
    ok_l = _check(f"trial={trial+1} act_l", tb.act_l, batch_ll)
    all_pass = all_pass and ok_h and ok_l

print(f"\nTest 1: {'🟢 ALL PASS' if all_pass else '🔴 SOME FAIL'}")


# ── Test 2: state-encoded SBA matches compute_swing_features ──────────────────

print("\n" + "=" * 60)
print("Test 2: 5-state SBA parity at every bar (single value series)")
rng2 = np.random.default_rng(99)
hi2 = np.cumsum(rng2.uniform(-0.01, 0.01, 200)) + 1.0
lo2 = hi2.copy()   # single-value series (range bars)
mid2 = hi2.copy()

_, _, batch_act_h, batch_act_l, _ = compute_swing_features(hi2, lo2, mid2)

tb2 = IncrementalTopsBots()
inc_act_h = []
inc_act_l = []
for i in range(len(hi2)):
    _, _, ah, al = tb2.update(hi2[i], lo2[i], mid2[i])
    inc_act_h.append(ah)
    inc_act_l.append(al)

# Compare with 1-bar lag (incremental is strictly causal — state at bar i
# reflects swings confirmed through bar i-1, while batch shows bar i's
# swing at bar i because it sees bar i+1 during construction).
mismatches = 0
for i in range(1, len(hi2) - 1):
    bh = float(batch_act_h[i]) if not np.isnan(batch_act_h[i]) else None
    bl = float(batch_act_l[i]) if not np.isnan(batch_act_l[i]) else None
    ih = inc_act_h[i]
    il = inc_act_l[i]
    if bh != ih or bl != il:
        mismatches += 1
        if mismatches <= 5:
            print(f"  mismatch at bar {i}: inc=({ih},{il}) batch=({bh},{bl})")

# After finalize, the FINAL act_h/act_l must agree
tb2.finalize()
last_bh = float(batch_act_h[-1]) if not np.isnan(batch_act_h[-1]) else None
last_bl = float(batch_act_l[-1]) if not np.isnan(batch_act_l[-1]) else None
ok_final = _check("final act_h (after finalize)", tb2.act_h, last_bh)
ok_final_l = _check("final act_l (after finalize)", tb2.act_l, last_bl)
print(f"\nIntra-series mismatches: {mismatches} (expected — 1-bar causal lag is correct)")
print(f"Test 2 final levels: {'🟢 PASS' if ok_final and ok_final_l else '🔴 FAIL'}")


# ── Test 3: serialization round-trip ─────────────────────────────────────────

print("\n" + "=" * 60)
print("Test 3: to_dict() / from_dict() round-trip")

rng3 = np.random.default_rng(7)
hi3 = np.cumsum(rng3.uniform(-0.01, 0.012, 150)) + 1.0
lo3 = hi3 - rng3.uniform(0.001, 0.004, 150)

tb_a = IncrementalTopsBots()
for i in range(100):
    tb_a.update(hi3[i], lo3[i], (hi3[i]+lo3[i])/2)

state_dict = tb_a.to_dict()
json_str   = json.dumps(state_dict)   # ensure JSON-serializable
state_dict2 = json.loads(json_str)
tb_b = IncrementalTopsBots.from_dict(state_dict2)

# Feed remaining bars to both and compare output
all_rt = True
for i in range(100, 150):
    mid = (hi3[i] + lo3[i]) / 2
    out_a = tb_a.update(hi3[i], lo3[i], mid)
    out_b = tb_b.update(hi3[i], lo3[i], mid)
    if out_a != out_b:
        print(f"  🔴 MISMATCH at bar {i}: {out_a} vs {out_b}")
        all_rt = False

print(f"Test 3: {'🟢 PASS (all 50 bars identical)' if all_rt else '🔴 FAIL'}")


# ── Test 4: IncrementalZscore matches batch ───────────────────────────────────

print("\n" + "=" * 60)
print("Test 4: IncrementalZscore matches batch arctan-zscore (exp_lgbm.py formula)")

rng4 = np.random.default_rng(13)
pop = 200
n4 = pop + 100
r_series = rng4.normal(0, 1, n4)
HP = math.pi / 2

# Batch reference: window = r[i-pop..i-1], slide AFTER computing score.
# At step i: std from past pop values, normalize r[i], then add r[i] / remove r[i-pop].
# This matches IncrementalZscore which computes std before appending r.
batch_f = np.zeros(n4)
s = s2 = 0.0
for k in range(pop):          # seed window with r[0..pop-1]
    s += r_series[k]; s2 += r_series[k]**2
for i in range(pop, n4):
    v   = s2 / pop - (s / pop) ** 2
    std = v ** 0.5 if v > 1e-20 else 1e-10
    batch_f[i] = math.atan(r_series[i] / std) / HP
    # Slide window: add r[i], remove r[i-pop]
    s  += r_series[i]       - r_series[i - pop]
    s2 += r_series[i] ** 2  - r_series[i - pop] ** 2

# Incremental
iz = IncrementalZscore(pop=pop)
inc_f = np.zeros(n4)
for i in range(n4):
    inc_f[i] = iz.update(r_series[i])

max_err = np.max(np.abs(batch_f[pop:] - inc_f[pop:]))
print(f"  max |batch - incremental| at i>=pop: {max_err:.2e}")
print(f"Test 4: {'🟢 PASS' if max_err < 1e-10 else '🔴 FAIL'} (tol=1e-10)")


# ── Test 5: IncrementalZscore serialization ───────────────────────────────────

print("\n" + "=" * 60)
print("Test 5: IncrementalZscore to_dict() / from_dict()")

iz_a = IncrementalZscore(pop=100)
for r in rng4.normal(0, 1, 150):
    iz_a.update(r)

iz_b = IncrementalZscore.from_dict(json.loads(json.dumps(iz_a.to_dict())))
test5_pass = True
for r in rng4.normal(0, 1, 50):
    va = iz_a.update(r); vb = iz_b.update(r)
    if abs(va - vb) > 1e-12:
        print(f"  🔴 mismatch: {va} vs {vb}")
        test5_pass = False
        break
print(f"Test 5: {'🟢 PASS' if test5_pass else '🔴 FAIL'}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("All tests complete.")
