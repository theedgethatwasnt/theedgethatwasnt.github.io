"""
make_sine_dataset.py — Phase 0 positive control for ESCMA (2026-06-12).

Generate a CLEAN sine price series + spread, compute the SAME 5 mn_* features via
the SAME kernel (entry_chopper._compute_momentum_stack, R6), and chop
momentum-continuation entries with a known optimal exit. Writes:
  features_SINE.parquet   (same schema as features_<PAIR>.parquet)
  meta3_SINE.parquet      (same schema as meta3_<PAIR>.parquet)

Sine: price = BASE + A·sin(2π t/P).  Non-JPY scale (pip 1e-4) so _pip_for("SINE")
returns 1e-4. A≈50 pips, P=720 bars (1h @ S5), spread≈1.7 pips.

Entries (momentum continuation):
  rising zero-cross  (t = k·P)        → LONG,  optimal exit at +P/4 (the peak)
  falling zero-cross (t = k·P + P/2)  → SHORT, optimal exit at +P/4 (the trough)
  t_timeout = t_event + P/2  (enough to reach the extreme AND start giving it back,
              so the net is FORCED to exit at the extreme, not hold to flat).

If the exit ESCMA cannot turn a big profit on this (swings ≈50p ≫ spread 1.7p), the
harness/reward/init is broken. --noise adds Phase-0b calibrated noise.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from entry_chopper import _compute_momentum_stack

SCRIPT_DIR = Path(__file__).resolve().parent
FEAT_NAMES = ["mom_S5", "mom_M1", "mom_5m", "mom_15m", "mom_1h",
              "mn_S5", "mn_M1", "mn_5m", "mn_15m", "mn_1h"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pip", type=float, default=1e-4)
    ap.add_argument("--base", type=float, default=1.0)
    ap.add_argument("--amp-pips", type=float, default=50.0)
    ap.add_argument("--period", type=int, default=720)
    ap.add_argument("--spread-pips", type=float, default=1.7)
    ap.add_argument("--cycles", type=int, default=500)
    ap.add_argument("--noise-pips", type=float, default=0.0,
                    help="Phase 0b: Gaussian noise σ per bar in pips (0 = clean).")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tag", default="SINE")
    args = ap.parse_args()

    PIP = args.pip
    P = args.period
    A = args.amp_pips * PIP
    SPREAD = args.spread_pips * PIP
    n = args.cycles * P
    t = np.arange(n, dtype=np.float64)

    mid = args.base + A * np.sin(2.0 * np.pi * t / P)
    if args.noise_pips > 0.0:
        rng = np.random.default_rng(args.seed)
        mid = mid + rng.normal(0.0, args.noise_pips * PIP, size=n)
    mid = mid.astype(np.float64)

    feats = _compute_momentum_stack(mid.copy(), PIP)   # 10 arrays, same kernel
    bid = (mid - SPREAD / 2.0).astype(np.float32)
    ask = (mid + SPREAD / 2.0).astype(np.float32)

    cols = {
        "bar_idx": np.arange(n, dtype=np.int64),
        "timestamp": np.arange(n, dtype=np.int64),       # dummy monotone
        "open": mid.astype(np.float32),
        "high": ask, "low": bid,
        "close": mid.astype(np.float32),
        "bid_c": bid, "ask_c": ask,
        "spread_pips": np.full(n, SPREAD / PIP, dtype=np.float32),
    }
    for nm, arr in zip(FEAT_NAMES, feats):
        cols[nm] = arr.astype(np.float32)
    feat_path = SCRIPT_DIR / f"features_{args.tag}.parquet"
    pq.write_table(pa.table(cols), feat_path)

    # ── entries: zero-crossings, momentum continuation ──
    warmup = P + 70          # skip mn_1h warmup (W_1h=720) + σ baseline (60)
    half, quarter = P // 2, P // 4
    events = []
    sid = 0
    k = 1
    while True:
        te_long = k * P                      # rising zero-cross → LONG
        if te_long + half + 10 >= n:
            break
        if te_long > warmup:
            events.append((sid, te_long - 60, te_long, te_long + half, 1)); sid += 1
        te_short = k * P + half              # falling zero-cross → SHORT
        if te_short + half + 10 < n and te_short > warmup:
            events.append((sid, te_short - 60, te_short, te_short + half, -1)); sid += 1
        k += 1

    m = pd.DataFrame(events, columns=["sample_id", "t_pre", "t_event",
                                      "t_timeout", "direction"])
    cut = int(len(m) * 0.70)
    m["split"] = ["IS"] * cut + ["OOS"] * (len(m) - cut)
    meta_path = SCRIPT_DIR / f"meta3_{args.tag}.parquet"
    m.to_parquet(meta_path)

    # sanity readout
    mn_s5 = feats[5]
    finite = np.isfinite(mn_s5)
    print(f"[sine] n={n:,} bars  amp={args.amp_pips}p  P={P}  spread={args.spread_pips}p"
          f"  noise={args.noise_pips}p")
    print(f"[sine] mn_S5 range [{np.nanmin(mn_s5):+.2f},{np.nanmax(mn_s5):+.2f}]  "
          f"finite={finite.mean()*100:.1f}%")
    print(f"[sine] entries={len(m)}  (IS={cut} OOS={len(m)-cut})  "
          f"long={(m.direction==1).sum()} short={(m.direction==-1).sum()}")
    print(f"[sine] optimal hold to extreme = +{quarter} bars; timeout = +{half} bars")
    print(f"[wrote] {feat_path.name}  {meta_path.name}")


if __name__ == "__main__":
    main()
