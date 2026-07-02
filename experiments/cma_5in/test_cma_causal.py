"""Train CMA-NN on CHF_JPY with CAUSAL features (new incremental builder).
If this works, we've eliminated the train/live disparity.
Compare to known non-causal result: +73 p/d OOS on CHF_JPY."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import cma
from research.experiments.cma_5in.train_cma_v2 import (
    simulate_chunk, fitness_neg, passes_hard_gates, eval_oos,
    _worker_init, _worker_fit, N_HID, N_OUT, N_POSITION_STATE, ACT_NAMES
)

PAIR = "CHF_JPY"
PIP = 0.01
SPREAD = 3.5
MAX_HOLD = 200
GENS = 200
POPSIZE = 24
PROJECT = Path(__file__).resolve().parents[3]

# Load CAUSAL features
df = pd.read_parquet(PROJECT / f'data/m5_ohlc/{PAIR}_M5_causal_features.parquet')
print(f'Loaded {len(df)} bars from causal parquet')

# Match train_cma_v2 normalization (macd_hist /2, clip to [-1,1])
macd = np.clip(df['macd_hist'].values / 2.0, -1, 1)
market = np.stack([
    df['mc_d_a'].values.astype(np.float64),
    df['mc_dd_a'].values.astype(np.float64),
    df['er_norm'].values.astype(np.float64),
    macd.astype(np.float64),
], axis=0)
mid = df['close'].values.astype(np.float64)
n = len(mid)
split = int(n * 0.7)
market_is = market[:, :split].copy()
mid_is = mid[:split].copy()
market_oos = market[:, split:].copy()
mid_oos = mid[split:].copy()
print(f'IS: {split}, OOS: {n-split}')

n_in = 4 + N_POSITION_STATE
fixed_act_id = ACT_NAMES.index("sin")
n_params = n_in * N_HID + N_HID + N_HID * N_OUT + N_OUT

# Feature sanity
print(f'CAUSAL feature ranges (post-warmup):')
for i, name in enumerate(['mc_d_a','mc_dd_a','er_norm','macd_hist']):
    v = market_is[i, 1000:]
    print(f'  {name}: [{v.min():+.3f}, {v.max():+.3f}] std={v.std():.3f}')
print()

warm = np.zeros(n_params)
simulate_chunk(market_is[:, :200], mid_is[:200], PIP, SPREAD, 50,
               warm, n_in, fixed_act_id, 0, 200)

pool = ProcessPoolExecutor(max_workers=4, initializer=_worker_init,
    initargs=(market_is, mid_is, PIP, SPREAD, MAX_HOLD, n_in, fixed_act_id, 3, 0.15, 288.0))

print('Training CMA-ES 200 gens...')
np.random.seed(42)
x0 = np.random.randn(n_params) * 0.3
es = cma.CMAEvolutionStrategy(x0, 0.5, {'popsize': POPSIZE, 'seed': 42, 'verbose': -9, 'maxiter': GENS})

best_fit = 1e18
best_vec = None
best_valid_pps = None
best_valid_vec = None
t0 = time.time()
gen = 0
while not es.stop():
    cands = es.ask()
    fits = list(pool.map(_worker_fit, cands))
    es.tell(cands, fits)
    gm = min(fits)
    if gm < best_fit:
        best_fit = gm
        best_vec = np.array(cands[fits.index(gm)])
    ok, mps = passes_hard_gates(best_vec, market_is, mid_is, PIP, SPREAD, MAX_HOLD,
                                 n_in, fixed_act_id, 3, 0.15, 288.0)
    if ok and (best_valid_pps is None or mps > best_valid_pps):
        best_valid_pps = mps; best_valid_vec = np.array(best_vec)
    gen += 1
    if gen % 25 == 0:
        print(f'  Gen {gen}: fit={best_fit:.2f}, valid={best_valid_pps}, t={time.time()-t0:.0f}s')
    if gen >= GENS: break

pool.shutdown(wait=False)
final = best_valid_vec if best_valid_vec is not None else best_vec
is_full = eval_oos(final, market_is, mid_is, PIP, SPREAD, MAX_HOLD, n_in, fixed_act_id, 288.0)
oos = eval_oos(final, market_oos, mid_oos, PIP, SPREAD, MAX_HOLD, n_in, fixed_act_id, 288.0)
print()
print(f'=== {PAIR} CAUSAL features result ===')
print(f'IS:  {is_full["pips_per_day"]:+.2f} p/d ({is_full["n_trades"]}T, dir={is_full["dir_ratio"]:.2f})')
print(f'OOS: {oos["pips_per_day"]:+.2f} p/d ({oos["n_trades"]}T, dir={oos["dir_ratio"]:.2f})')
print(f'Hard gate: {"PASS" if best_valid_pps else "FAIL"}')
print()
print(f'COMPARISON: Non-causal training result was +73.0 p/d OOS.')
print(f'If CAUSAL OOS >> 0, the edge survives without lookahead.')
print(f'If CAUSAL OOS <= 0, the lookahead was most of the edge.')
