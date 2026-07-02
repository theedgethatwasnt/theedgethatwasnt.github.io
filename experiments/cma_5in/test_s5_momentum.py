"""Test proposed S5 momentum feature set:
  A: close[t] - close[t-6]     (30s change)
  B: close[t] - close[t-60]    (5m change, "slope/velocity")
  C: close[t] - close[t-720]   (1h change)
  D: B[t] - B[t-60]            (5m-slope acceleration = 2nd diff of close at 5m scale)
  + upnl, mae, mfe = 7 inputs

Architecture: 7 → 8(sin) → 3
Cadence: S5 (17,280 bars/day)
Fully causal by construction.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import cma
from numba import njit

from research.experiments.cma_5in.train_cma_v2 import (
    simulate_chunk, passes_hard_gates, eval_oos,
    _worker_init, _worker_fit, N_HID, N_OUT, N_POSITION_STATE, ACT_NAMES
)

PAIR = "EUR_JPY"
PIP = 0.01
SPREAD = 2.3  # EUR_JPY
MAX_HOLD = 720   # 1 hour at S5 cadence
GENS = 200
POPSIZE = 24
BARS_PER_DAY = 17280  # S5
PROJECT = Path(__file__).resolve().parents[3]


@njit(cache=True)
def compute_s5_momentum_features(closes, n):
    """Compute 4 S5 momentum features. All causal."""
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n)
    for i in range(n):
        if i >= 6:   A[i] = closes[i] - closes[i-6]
        if i >= 60:  B[i] = closes[i] - closes[i-60]
        if i >= 720: C[i] = closes[i] - closes[i-720]
        # D = B[i] - B[i-60]
        # = (close[i]-close[i-60]) - (close[i-60]-close[i-120])
        # = close[i] - 2*close[i-60] + close[i-120]
        if i >= 120: D[i] = closes[i] - 2.0 * closes[i-60] + closes[i-120]
    return A, B, C, D


def main():
    # Load EUR_JPY S5
    path = PROJECT / "data/s5_ohlc/EUR_JPY_S5_BA.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} S5 bars of {PAIR}")
    print(f"  Range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    # Use mid close
    mid = ((df['bid_c'].values + df['ask_c'].values) / 2.0).astype(np.float64)
    n = len(mid)

    # Compute features
    t0 = time.time()
    A, B, C, D = compute_s5_momentum_features(mid, n)
    print(f"Features computed in {time.time()-t0:.1f}s")

    # Normalize via tanh with pip-scale
    # Typical S5 30s move: 1-3 pips. 5m: 3-10 pips. 1h: 10-40 pips. Accel: 2-8 pips.
    A_norm = np.tanh(A / PIP / 5.0)      # 5 pip scale
    B_norm = np.tanh(B / PIP / 15.0)     # 15 pip scale
    C_norm = np.tanh(C / PIP / 40.0)     # 40 pip scale
    D_norm = np.tanh(D / PIP / 10.0)     # 10 pip scale

    # Sanity: post-warmup feature distributions
    print("\nPost-warmup feature stats:")
    warm = slice(1000, None)
    for name, v in [("A (30s)", A_norm[warm]), ("B (5m)", B_norm[warm]),
                    ("C (1h)", C_norm[warm]), ("D (accel)", D_norm[warm])]:
        print(f"  {name:10s}: [{v.min():+.3f}, {v.max():+.3f}]  "
              f"mean={v.mean():+.4f}  std={v.std():.3f}")

    market = np.stack([A_norm, B_norm, C_norm, D_norm], axis=0)

    split = int(n * 0.7)
    m_is = market[:, :split].copy()
    mid_is = mid[:split].copy()
    m_oos = market[:, split:].copy()
    mid_oos = mid[split:].copy()
    print(f"\nIS: {split:,} bars ({split/BARS_PER_DAY:.1f} days)")
    print(f"OOS: {n-split:,} bars ({(n-split)/BARS_PER_DAY:.1f} days)")

    n_in = 4 + N_POSITION_STATE  # 7
    fixed_act_id = ACT_NAMES.index("sin")
    n_params = n_in * N_HID + N_HID + N_HID * N_OUT + N_OUT

    # JIT warm
    warm_w = np.zeros(n_params)
    simulate_chunk(m_is[:, :200], mid_is[:200], PIP, SPREAD, 50,
                   warm_w, n_in, fixed_act_id, 0, 200)

    pool = ProcessPoolExecutor(max_workers=4, initializer=_worker_init,
        initargs=(m_is, mid_is, PIP, SPREAD, MAX_HOLD, n_in, fixed_act_id, 3, 0.15, BARS_PER_DAY))

    print(f"\nTraining CMA-ES {GENS} gens (popsize {POPSIZE})...")
    t0 = time.time()
    np.random.seed(42)
    x0 = np.random.RandomState(42).randn(n_params) * 0.3
    es = cma.CMAEvolutionStrategy(x0, 0.5, {'popsize': POPSIZE, 'seed': 42, 'verbose': -9, 'maxiter': GENS})

    best_fit = 1e18; best_vec = None
    best_valid_pps = None; best_valid_vec = None
    gen = 0
    while not es.stop():
        c_ = es.ask()
        f_ = list(pool.map(_worker_fit, c_))
        es.tell(c_, f_)
        gm = min(f_)
        if gm < best_fit:
            best_fit = gm; best_vec = np.array(c_[f_.index(gm)])
        ok, mps = passes_hard_gates(best_vec, m_is, mid_is, PIP, SPREAD, MAX_HOLD,
                                     n_in, fixed_act_id, 3, 0.15, BARS_PER_DAY)
        if ok and (best_valid_pps is None or mps > best_valid_pps):
            best_valid_pps = mps; best_valid_vec = np.array(best_vec)
        gen += 1
        if gen % 25 == 0:
            print(f"  Gen {gen}: fit={best_fit:.2f}, valid={best_valid_pps}, t={time.time()-t0:.0f}s")
        if gen >= GENS: break

    pool.shutdown(wait=False)
    final = best_valid_vec if best_valid_vec is not None else best_vec
    is_full = eval_oos(final, m_is, mid_is, PIP, SPREAD, MAX_HOLD, n_in, fixed_act_id, BARS_PER_DAY)
    oos = eval_oos(final, m_oos, mid_oos, PIP, SPREAD, MAX_HOLD, n_in, fixed_act_id, BARS_PER_DAY)

    print(f"\n{'='*65}")
    print(f"  S5 MOMENTUM RESULT: {PAIR}")
    print(f"{'='*65}")
    print(f"  IS:  {is_full['pips_per_day']:+.2f} p/d  ({is_full['n_trades']}T, dir={is_full['dir_ratio']})")
    print(f"  OOS: {oos['pips_per_day']:+.2f} p/d  ({oos['n_trades']}T, dir={oos['dir_ratio']})")
    print(f"  Hard gate: {'PASS' if best_valid_pps else 'FAIL'}")
    print(f"  Training time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
