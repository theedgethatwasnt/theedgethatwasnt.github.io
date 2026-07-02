"""
precompute_features.py — Compute the full feature stack ONCE over the entire
S5 source array, store it aligned by bar index, and let downstream loaders
slice by index. Single source of truth for features (SOP R6).

==========================================================================
WHY
==========================================================================
The old pipeline computed momentum/σ features in TWO places:
  1. entry_chopper.py (for event detection + pre-context cache)
  2. train_cma_exit.py load_real() (recomputed post-entry features from source)
Two code paths = divergence risk + slow (recompute on every training run).

This script computes the 10 momentum/σ arrays ONCE via the EXISTING
`_compute_momentum_stack` kernel imported from entry_chopper (NEVER
reimplemented — SOP R6), aligns them with OHLC + bid/ask + spread, and writes
`features_<PAIR>.parquet` with an explicit integer `bar_idx`. A sample's three
indices (t_pre, t_event, t_timeout) then map directly to row slices.

==========================================================================
R7 LOOK-AHEAD PROOF (--verify)
==========================================================================
The whole "compute over the full array then slice" approach is only valid if
features[t] depend ONLY on close[0:t+1]. We PROVE this empirically: pick 5
random interior bars t, recompute the stack on the truncated array close[0:t+1],
and assert the last-bar features bit-match the full-array computation. If this
fails, a precomputed parquet would leak future data and MUST NOT be used.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# SOP R6: import the feature kernel — NEVER reimplement.
from entry_chopper import _compute_momentum_stack, _pip  # noqa: E402

PROJECT_ROOT = Path("/path/to/projects/fx-core")

# Float feature columns (order is the on-disk parquet order; the 15-vector
# order used by the RFF/sim is assembled in the loader, not here).
FEAT_COLS = [
    "mom_S5", "mom_M1", "mom_5m", "mom_15m", "mom_1h",
    "mn_S5", "mn_M1", "mn_5m", "mn_15m", "mn_1h",
]


def _load_source_close(data_path: Path):
    """Read OHLC + bid/ask + timestamp from the S5 BA source parquet."""
    tbl = pq.read_table(
        data_path,
        columns=["timestamp", "open", "high", "low", "close", "bid_c", "ask_c"],
    )
    f32 = lambda c: tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    out = {
        "timestamp": tbl.column("timestamp").to_numpy(zero_copy_only=False),
        "open": f32("open"),
        "high": f32("high"),
        "low": f32("low"),
        "close": f32("close"),
        "bid_c": f32("bid_c"),
        "ask_c": f32("ask_c"),
    }
    return out


def _compute_stack_f64(close_f32: np.ndarray, pip: float):
    """Run the chopper kernel on a float64 copy of close, return 10 f64 arrays.

    Identical entry point used everywhere (R6). float64 input matches what the
    chopper does internally (close_f64 = close.astype(np.float64)).
    """
    close_f64 = np.asarray(close_f32, dtype=np.float64)
    return _compute_momentum_stack(close_f64, pip)


def precompute(pair: str, data_path: Path | None = None,
               out_dir: Path | None = None) -> Path:
    if data_path is None:
        data_path = PROJECT_ROOT / f"data/s5_ba/{pair}_S5_BA.parquet"
    if out_dir is None:
        out_dir = SCRIPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pip = _pip(pair)

    t0 = time.time()
    print(f"[load] reading {data_path} ...")
    src = _load_source_close(data_path)
    n = src["close"].shape[0]
    print(f"[load] {n:,} rows in {time.time()-t0:.1f}s")

    print(f"[features] computing momentum stack on {n:,} bars (chopper kernel) ...")
    t0 = time.time()
    feats = _compute_stack_f64(src["close"], pip)
    print(f"[features] done in {time.time()-t0:.1f}s")
    # feats order matches FEAT_COLS exactly:
    #   (mom_s5, mom_m1, mom_5m, mom_15m, mom_1h, mn_s5, mn_m1, mn_5m, mn_15m, mn_1h)

    bar_idx = np.arange(n, dtype=np.int64)
    spread_pips = ((src["ask_c"] - src["bid_c"]) / np.float32(pip)).astype(np.float32)

    cols = {
        "bar_idx": bar_idx,
        "timestamp": src["timestamp"],
        "open": src["open"],
        "high": src["high"],
        "low": src["low"],
        "close": src["close"],
        "bid_c": src["bid_c"],
        "ask_c": src["ask_c"],
        "spread_pips": spread_pips,
    }
    for name, arr in zip(FEAT_COLS, feats):
        cols[name] = arr.astype(np.float32)

    # NaN counts (warmup NaNs expected at the start of each window)
    print("[nan] warmup NaN counts per feature column:")
    for name in FEAT_COLS:
        nan_ct = int(np.isnan(cols[name]).sum())
        print(f"        {name:<8} : {nan_ct:>8,} NaN ({100.0*nan_ct/n:.4f}%)")

    out_path = out_dir / f"features_{pair}.parquet"
    schema = pa.schema(
        [("bar_idx", pa.int64()),
         ("timestamp", pa.timestamp("ns", tz="UTC")),
         ("open", pa.float32()),
         ("high", pa.float32()),
         ("low", pa.float32()),
         ("close", pa.float32()),
         ("bid_c", pa.float32()),
         ("ask_c", pa.float32()),
         ("spread_pips", pa.float32())]
        + [(c, pa.float32()) for c in FEAT_COLS]
    )
    tbl = pa.table(cols, schema=schema)
    print(f"[write] writing {out_path} (zstd) ...")
    t0 = time.time()
    pq.write_table(tbl, out_path, compression="zstd")
    sz = out_path.stat().st_size
    print(f"[write] {out_path} ({sz/1e9:.3f} GB) in {time.time()-t0:.1f}s")
    print(f"[done] rows={n:,}")
    return out_path


# ── R7 LOOK-AHEAD PROOF ───────────────────────────────────────────────────
def verify(pair: str, data_path: Path | None = None, n_bars: int = 5,
           seed: int = 0) -> bool:
    """Prove features[t] depend ONLY on close[0:t+1].

    For 5 random interior bars t (>=1500, warm windows), recompute the stack
    on close[0:t+1] and compare the LAST-bar features to the full-array
    computation at index t. Bit-identical (atol=1e-9) ⇒ no look-ahead.
    """
    if data_path is None:
        data_path = PROJECT_ROOT / f"data/s5_ba/{pair}_S5_BA.parquet"
    pip = _pip(pair)

    print(f"[verify] loading close from {data_path} ...")
    src = _load_source_close(data_path)
    close = src["close"]
    n = close.shape[0]

    # Full-array computation
    print("[verify] computing full-array feature stack ...")
    full = _compute_stack_f64(close, pip)  # 10 arrays length n

    rng = np.random.default_rng(seed)
    lo, hi = 1500, n - 1
    ts = np.sort(rng.choice(np.arange(lo, hi), size=n_bars, replace=False))

    print(f"[verify] checking {n_bars} bars: {ts.tolist()}")
    all_ok = True
    atol = 1e-9
    for t in ts:
        t = int(t)
        trunc = _compute_stack_f64(close[: t + 1], pip)  # only [0:t+1]
        worst = 0.0
        worst_name = ""
        for name, full_arr, trunc_arr in zip(FEAT_COLS, full, trunc):
            v_full = full_arr[t]            # full-array value at bar t
            v_trunc = trunc_arr[-1]         # truncated: last bar == t
            # NaN must match NaN exactly
            if np.isnan(v_full) and np.isnan(v_trunc):
                continue
            if np.isnan(v_full) != np.isnan(v_trunc):
                print(f"  FAIL bar {t} {name}: NaN mismatch "
                      f"(full={v_full} trunc={v_trunc})")
                all_ok = False
                continue
            diff = abs(float(v_full) - float(v_trunc))
            if diff > worst:
                worst, worst_name = diff, name
        if worst > atol:
            print(f"  FAIL bar {t}: max abs diff = {worst:.3e} "
                  f"on {worst_name} (atol={atol:.0e})")
            all_ok = False
        else:
            print(f"  ok   bar {t}: max abs diff = {worst:.3e} (<= {atol:.0e})")

    print()
    if all_ok:
        print(f"R7 PASS: features at bar t computed from [0:t+1] match "
              f"full-array computation ({n_bars}/{n_bars} bars)")
    else:
        print("R7 FAIL: feature[t] depends on bars > t — DO NOT use a "
              "precomputed parquet (the full-array approach leaks).")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="USD_JPY")
    ap.add_argument("--data-path", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--verify", action="store_true",
                    help="Run the R7 look-ahead proof and exit (no write).")
    ap.add_argument("--verify-bars", type=int, default=5)
    ap.add_argument("--verify-seed", type=int, default=0)
    args = ap.parse_args()

    if args.verify:
        ok = verify(args.pair, data_path=args.data_path,
                    n_bars=args.verify_bars, seed=args.verify_seed)
        sys.exit(0 if ok else 1)
    else:
        precompute(args.pair, data_path=args.data_path, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
