"""H12 — Adaptive / smoothed MA cross-back exit on H1.

Tests two adaptive MA variants vs the H11 plain-SMA baseline:
  HMA(N)        Hull MA — lag-reduced
  Kalman(q,r)   1D constant-level Kalman filter on H1 close

Same exit semantics as H11: exit long when H1 close < MA on last completed H1.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (PAIRS, IS_FRAC, BARS_PER_H1, TP_PIPS_BASE,
                  load_pair, sma, project_to_m5, trade_stats_from_arrays)

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
HMA_N    = [8, 16, 32, 64]
KAL_Q    = [0.001, 0.01, 0.1, 1.0]      # process noise — smaller = smoother
KAL_R    = [1.0]                         # observation noise (fixed for sweep simplicity)


def wma(arr, n):
    """Weighted moving average, weights 1..n."""
    w = np.arange(1, n+1, dtype=np.float64)
    w_sum = w.sum()
    out = np.full(len(arr), np.nan)
    for i in range(n-1, len(arr)):
        out[i] = (arr[i-n+1:i+1] * w).sum() / w_sum
    return out


def hma(arr, n):
    """HMA(N) = WMA(2*WMA(p, n/2) - WMA(p, n), sqrt(n))."""
    n_half  = max(1, n // 2)
    n_sqrt  = max(1, int(round(np.sqrt(n))))
    w1 = wma(arr, n_half)
    w2 = wma(arr, n)
    raw = 2.0 * w1 - w2
    return wma(raw, n_sqrt)


def kalman_1d(arr, q, r):
    """Simple 1D Kalman filter on price level (constant-level model).
    state: x  variance: p
    Returns the filtered level series."""
    n = len(arr)
    out = np.full(n, np.nan)
    if n == 0: return out
    x = arr[0]; p = 1.0
    out[0] = x
    for i in range(1, n):
        # predict (constant-level)
        p = p + q
        # update
        k = p / (p + r)
        x = x + k * (arr[i] - x)
        p = (1 - k) * p
        out[i] = x
    return out


@nb.njit(cache=True)
def kernel(opens, highs, lows, closes, sig_m5,
           h1_c_m5, ma_m5, pip, with_tp):
    n = len(opens)
    pos = 0; entry_px = 0.0; entry_bar = -1
    pnls = np.empty(n, np.float64); ents = np.empty(n, np.int64); nt = 0
    for i in range(1, n):
        sig = sig_m5[i-1]
        if pos != 0:
            exit_px = 0.0; reason = -1
            if with_tp == 1:
                tp_lvl = entry_px + pos * TP_PIPS_BASE * pip
                if pos == 1 and highs[i] >= tp_lvl:
                    exit_px = tp_lvl; reason = 0
                elif pos == -1 and lows[i] <= tp_lvl:
                    exit_px = tp_lvl; reason = 0
            if reason < 0:
                hc = h1_c_m5[i-1]; ma = ma_m5[i-1]
                if not (np.isnan(hc) or np.isnan(ma)):
                    if pos == 1 and hc < ma:
                        exit_px = closes[i]; reason = 1
                    elif pos == -1 and hc > ma:
                        exit_px = closes[i]; reason = 1
            if reason >= 0:
                pnls[nt] = (exit_px - entry_px) / pip * pos
                ents[nt] = entry_bar; nt += 1
                pos = 0; continue
        if pos == 0 and sig != 0:
            pos = sig; entry_px = opens[i]; entry_bar = i
    if pos != 0:
        pnls[nt] = (closes[-1] - entry_px) / pip * pos
        ents[nt] = entry_bar; nt += 1
    return pnls[:nt], ents[:nt]


def main():
    print("="*92)
    print(f"  H12 — Adaptive MA cross-back.  HMA N={HMA_N}  Kalman q={KAL_Q} r={KAL_R}")
    print("="*92)
    _o = np.zeros(50); _s = np.zeros(50, np.int8); _h = np.full(50,1.0)
    kernel(_o,_o,_o,_o,_s,_h,_h,0.0001,1)

    rows = []; t0 = time.time()
    for pair in PAIRS:
        b = load_pair(pair); sp = b['spread_cost']; pip = b['pip']
        h1_ts = b['h1_ts']; h1_c = b['h1_c']
        prev_ts = np.empty_like(b['m5_ts']); prev_ts[0]=b['m5_ts'][0]; prev_ts[1:]=b['m5_ts'][:-1]
        # HMA family
        for N in HMA_N:
            ma_arr = hma(h1_c, N)
            ma_m5  = project_to_m5(prev_ts, h1_ts, ma_arr)
            for tp in (1, 0):
                p, e = kernel(b['opens'], b['highs'], b['lows'], b['closes'],
                              b['sig_m5'], b['h1_c_m5'], ma_m5, pip, tp)
                s = trade_stats_from_arrays(p, e, b['is_end'], b['n'], sp)
                rows.append({'pair':pair,'type':'HMA','param':N,'with_TP':bool(tp),**s})
        # Kalman family
        for q in KAL_Q:
            for r in KAL_R:
                ma_arr = kalman_1d(h1_c, q, r)
                ma_m5  = project_to_m5(prev_ts, h1_ts, ma_arr)
                for tp in (1, 0):
                    p_, e_ = kernel(b['opens'], b['highs'], b['lows'], b['closes'],
                                    b['sig_m5'], b['h1_c_m5'], ma_m5, pip, tp)
                    s = trade_stats_from_arrays(p_, e_, b['is_end'], b['n'], sp)
                    rows.append({'pair':pair,'type':'KAL','param':q,'with_TP':bool(tp),**s})

    rdf = pd.DataFrame(rows); rdf.to_csv(OUT/'h12_adaptive_ma.csv', index=False)
    print(f"  Runtime: {time.time()-t0:.1f}s  rows: {len(rdf)}")
    for typ in ['HMA','KAL']:
        sub = rdf[rdf['type']==typ]
        cand = sub[(sub.is_net>0)&(sub.oos_net>0)].sort_values(['pair','oos_pd'],
                                                                ascending=[True,False])
        bp = cand.groupby('pair').head(1)
        print(f"\n  {typ}  Best IS+OOS+ per pair:")
        if len(bp)==0:
            print("    none.")
        else:
            for _, r in bp.iterrows():
                tp = "+TP" if r['with_TP'] else "no-TP"
                print(f"    {r['pair']:<9} {typ} p={r['param']:>5}  {tp:<6} "
                      f"IS={r['is_pd']:+6.2f}  OOS={r['oos_pd']:+6.2f}  "
                      f"DD={r['oos_dd']:+6.0f}  N={int(r['oos_n'])}  WR={r['oos_wr']:5.1f}%")
            print(f"  Pairs IS+OOS+: {len(bp)}/10   Σ OOS p/d: {bp['oos_pd'].sum():+.2f}")


if __name__ == '__main__':
    main()
