#!/usr/bin/env python3
"""OUT-OF-SAMPLE confirmation of the FROZEN H1 breakout config on UNSEEN pairs.

The config (H1, SMA50, K=1σ, TP50/SL100, slope-flip breakout, bounded fade exit) was SELECTED
by a grid search on EUR_JPY/EUR_USD/GBP_USD/USD_JPY -> in-sample, OOS spent (R8). These 8 pairs
were NEVER used in that search. Running the frozen config on them once = a true out-of-sample
test in the PAIR dimension. NOTHING is tuned here.

Confirms if: broadly positive across the unseen pairs + aggregate MC p<0.05 + WF>=4/6.
Refutes if: collapses / only 1-2 pairs positive (the project's 'reduces to one pair' failure).
MEMORY-SAFE: one pair at a time.
"""
import sys, gc
import numpy as np, pandas as pd, numba as nb, pyarrow.parquet as pq
PKG = "/path/to/projects/fx-core/research/experiments/conservative_010"
sys.path.insert(0, PKG)
from validate import walk_forward, monte_carlo, equity_drawdown

S5_DIR = "/path/to/projects/fx-core/data/s5_ohlc"
MAX_ROWS = 5_000_000   # match the IS study's tail window (same period as the tuned pairs)


def load_min(pair):
    """Minimal direct S5-BA loader (no stack010 CFG dependency): mid H/L/C + bid_c/ask_c.
    R3b: assert finite bid/ask, no fallback. pip inferred from quote currency."""
    df = pq.read_table(f"{S5_DIR}/{pair}_S5_BA.parquet",
                       columns=['timestamp', 'high', 'low', 'close', 'bid_c', 'ask_c']).to_pandas()
    df = df.sort_values('timestamp').reset_index(drop=True)
    if len(df) > MAX_ROWS:
        df = df.iloc[-MAX_ROWS:].reset_index(drop=True)
    assert np.isfinite(df['bid_c']).all() and np.isfinite(df['ask_c']).all(), f"{pair}: non-finite BA"
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    ts = df['timestamp'].to_numpy().astype('datetime64[ns]').astype(np.int64)
    return df, ts, pip

# 8 pairs NOT used in the config search:
OOS_PAIRS = ["AUD_JPY", "AUD_USD", "CAD_JPY", "CHF_JPY", "EUR_GBP", "GBP_JPY", "NZD_JPY", "NZD_USD"]
IS_PAIRS = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]   # for reference recomputation
SLIP = 2.0; SLOPE_LB = 3; H1_NS = 3_600_000_000_000
K = 1.0; TP = 50.0; SL = 100.0     # FROZEN


@nb.njit(cache=True)
def _kern(h, l, c, bid, ask, lower, upper, tneg, tpos, pip, tp, sl, slip):
    n = len(h); pos = 0; entry = 0.0
    _eb = np.empty(n, np.int64); _pnl = np.empty(n, np.float64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if tneg[i] and c[i] < lower[i]:
                pos = -1; entry = bid[i]; _eb[nt] = i; continue
            if tpos[i] and c[i] > upper[i]:
                pos = 1; entry = ask[i]; _eb[nt] = i; continue
        if pos != 0:
            rsn = -1; exf = 0.0
            fc = entry - pos * sl * pip
            if pos == 1 and l[i] <= fc: exf = fc - slip*pip; rsn = 0
            elif pos == -1 and h[i] >= fc: exf = fc + slip*pip; rsn = 0
            if rsn < 0:
                tpl = entry + pos*tp*pip
                if pos == 1 and h[i] >= tpl: exf = tpl; rsn = 2
                elif pos == -1 and l[i] <= tpl: exf = tpl; rsn = 2
            if rsn >= 0:
                _pnl[nt] = (exf-entry)/pip if pos == 1 else (entry-exf)/pip
                nt += 1; pos = 0
    return _eb[:nt], _pnl[:nt]


def resample(df, ts, tf):
    b = ts // tf; g = df.groupby(b, sort=True)
    return (g['h'].max().to_numpy(), g['l'].min().to_numpy(), g['c'].last().to_numpy(),
            g['bid'].last().to_numpy(), g['ask'].last().to_numpy(),
            g['c'].last().index.to_numpy() * tf)


def run_pair(p):
    raw, ts, pip = load_min(p)
    df = pd.DataFrame({'h': raw['high'].to_numpy(), 'l': raw['low'].to_numpy(),
                       'c': raw['close'].to_numpy(), 'bid': raw['bid_c'].to_numpy(),
                       'ask': raw['ask_c'].to_numpy()}); del raw; gc.collect()
    h, l, c, bid, ask, tsh = resample(df, ts, H1_NS); n = len(c); s = pd.Series(c)
    sma = s.rolling(50).mean().to_numpy(); sd = s.rolling(50).std(ddof=0).to_numpy()
    v = ~np.isnan(sma) & ~np.isnan(sd)
    slope = np.zeros(n); slope[SLOPE_LB:] = sma[SLOPE_LB:] - sma[:-SLOPE_LB]
    slope = np.where(v, slope, 0.0)
    sp = np.empty(n); sp[0] = 0.0; sp[1:] = slope[:-1]
    tneg = (slope < 0) & (sp >= 0) & v; tpos = (slope > 0) & (sp <= 0) & v
    lower = np.where(v, sma - K*sd, -1e18); upper = np.where(v, sma + K*sd, 1e18)
    eb, pnl = _kern(h, l, c, bid, ask, lower, upper, tneg, tpos, pip, TP, SL, SLIP)
    et = tsh[eb]
    rows = [{'exit_time': int(et[i]), 'pnl_pips': float(pnl[i])} for i in range(len(pnl))]
    del df, ts, h, l, c, bid, ask, sma, sd, slope, sp, tneg, tpos; gc.collect()
    return rows


def report(name, pairs):
    per = {}; allrows = []
    for p in pairs:
        r = run_pair(p); per[p] = r; allrows += r
        net = sum(x['pnl_pips'] for x in r)
        wr = 100*np.mean([x['pnl_pips'] > 0 for x in r]) if r else 0
        print(f"  {p:>8}: net={net:>7.0f}p  n={len(r):>4}  WR={wr:.0f}%", flush=True)
    tl = sorted(allrows, key=lambda x: x['exit_time']); net = np.array([x['pnl_pips'] for x in tl]); N = len(net)
    eb = np.arange(N)
    wf = sum(1 for f in walk_forward(eb, net, N, 6) if f['net']>0)
    mc = monte_carlo(net, 300)['p_net']; dd = equity_drawdown(np.cumsum(net))['max_dd']
    npos = sum(1 for p in pairs if sum(x['pnl_pips'] for x in per[p]) > 0)
    print(f"  === {name}: {len(pairs)} pairs, {npos} positive | net={net.sum():.0f}p n={N} "
          f"WF {wf}/6 MC p_net={mc:.3f} eqMaxDD={dd:.0f}p ===")
    return net.sum(), npos, wf, mc


print("FROZEN config: H1 breakout, SMA50, K=1σ, TP50/SL100, no PSAR/flip. NOTHING tuned here.\n")
print(">>> OUT-OF-SAMPLE — 8 pairs never used in the config search:")
report("OOS (8 unseen pairs)", OOS_PAIRS)
print("\n>>> IS reference — the 4 pairs the config was tuned on:")
report("IS (4 tuned pairs)", IS_PAIRS)
print("\nVERDICT: OOS broadly positive (>=5/8 pairs) + MC p<0.05 => the H1 breakout GENERALIZES "
      "(real edge). OOS collapses / 0-2 pairs => it was overfit to the 4-pair search.")
