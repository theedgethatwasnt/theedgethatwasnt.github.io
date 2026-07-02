"""H14 — Higher-TF distance band.

Exit long when (price − SMA(N, H1)) / ATR_H1 exceeds +X (TP-equiv)
            OR drops below −X (SL-equiv).
Replaces fixed TP/SL with regime-scaled band.

Grid:
  N ∈ {8, 16, 32}        SMA window on H1
  X ∈ {1, 1.5, 2, 3}     band half-width in ATR units
  mode ∈ {'replace', 'overlay'}
    replace: this band fully replaces fixed +20p TP and no-SL
    overlay: keep +20p TP, add SL-side of band only (downside protection only)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (PAIRS, IS_FRAC, BARS_PER_H1, TP_PIPS_BASE,
                  load_pair, sma, project_to_m5, trade_stats_from_arrays)

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
N_GRID = [8, 16, 32]
X_GRID = [1.0, 1.5, 2.0, 3.0]


@nb.njit(cache=True)
def kernel(opens, highs, lows, closes, sig_m5,
           h1_sma_m5, h1_atr_m5, pip, X, mode):
    """mode: 0=replace (TP+SL both band), 1=overlay (keep fixed TP, SL via band only)."""
    n = len(opens)
    pos = 0; entry_px = 0.0; entry_bar = -1
    pnls = np.empty(n, np.float64); ents = np.empty(n, np.int64); nt = 0
    for i in range(1, n):
        sig = sig_m5[i-1]
        if pos != 0:
            exit_px = 0.0; reason = -1
            sm = h1_sma_m5[i-1]; at = h1_atr_m5[i-1]
            # Fixed TP (overlay only)
            if mode == 1:
                tp_lvl = entry_px + pos * TP_PIPS_BASE * pip
                if pos == 1 and highs[i] >= tp_lvl:
                    exit_px = tp_lvl; reason = 0
                elif pos == -1 and lows[i] <= tp_lvl:
                    exit_px = tp_lvl; reason = 0
            # Band edges (both modes use band for SL-side; replace also for TP)
            if reason < 0 and not (np.isnan(sm) or np.isnan(at) or at <= 0):
                upper = sm + X * at
                lower = sm - X * at
                cl = closes[i]
                if pos == 1:
                    # SL when price falls X-ATR below H1 SMA
                    if cl <= lower:
                        exit_px = cl; reason = 2     # band-sl
                    # TP when price rises X-ATR above H1 SMA (replace mode only)
                    elif mode == 0 and cl >= upper:
                        exit_px = cl; reason = 3     # band-tp
                else:
                    if cl >= upper:
                        exit_px = cl; reason = 2
                    elif mode == 0 and cl <= lower:
                        exit_px = cl; reason = 3
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
    print(f"  H14 — Higher-TF distance band.  N={N_GRID}  X={X_GRID} ATR  modes=[replace,overlay]")
    print("="*92)
    _o = np.zeros(50); _s = np.zeros(50, np.int8); _h = np.full(50,1.0)
    kernel(_o,_o,_o,_o,_s,_h,_h,0.0001,1.0,0)

    rows = []; t0 = time.time()
    for pair in PAIRS:
        b = load_pair(pair); sp = b['spread_cost']; pip = b['pip']
        h1_ts = b['h1_ts']; h1_c = b['h1_c']
        prev_ts = np.empty_like(b['m5_ts']); prev_ts[0]=b['m5_ts'][0]; prev_ts[1:]=b['m5_ts'][:-1]
        for N in N_GRID:
            sma_arr = sma(h1_c, N)
            sma_m5  = project_to_m5(prev_ts, h1_ts, sma_arr)
            for X in X_GRID:
                for mode_id, mode_name in [(0,'replace'),(1,'overlay')]:
                    p, e = kernel(b['opens'], b['highs'], b['lows'], b['closes'],
                                  b['sig_m5'], sma_m5, b['h1_atr_m5'], pip, X, mode_id)
                    s = trade_stats_from_arrays(p, e, b['is_end'], b['n'], sp)
                    rows.append({'pair':pair,'N':N,'X':X,'mode':mode_name,**s})

    rdf = pd.DataFrame(rows); rdf.to_csv(OUT/'h14_distance_band.csv', index=False)
    print(f"  Runtime: {time.time()-t0:.1f}s  rows: {len(rdf)}")
    for mode in ['replace','overlay']:
        sub = rdf[rdf['mode']==mode]
        cand = sub[(sub.is_net>0)&(sub.oos_net>0)].sort_values(['pair','oos_pd'],
                                                                ascending=[True,False])
        bp = cand.groupby('pair').head(1)
        print(f"\n  mode={mode}   Best IS+OOS+ per pair:")
        if len(bp)==0:
            print("    none.")
        else:
            for _, r in bp.iterrows():
                print(f"    {r['pair']:<9} N={int(r['N']):>2d} X={r['X']:.1f}  "
                      f"IS={r['is_pd']:+6.2f}  OOS={r['oos_pd']:+6.2f}  "
                      f"DD={r['oos_dd']:+6.0f}  N={int(r['oos_n'])}  WR={r['oos_wr']:5.1f}%")
            print(f"  Pairs IS+OOS+: {len(bp)}/10   Σ OOS p/d: {bp['oos_pd'].sum():+.2f}")


if __name__ == '__main__':
    main()
