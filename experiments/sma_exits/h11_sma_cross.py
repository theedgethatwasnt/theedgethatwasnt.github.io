"""H11 — Exit long when H1 close crosses below SMA(N, H1).

Grid:
  N ∈ {8, 16, 32, 64}
  with_TP ∈ {True, False}      (keep +20p TP alongside?)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (PAIRS, IS_FRAC, BARS_PER_H1, TP_PIPS_BASE,
                  load_pair, sma, resample_tf, project_to_m5,
                  trade_stats_from_arrays)

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
N_GRID = [8, 16, 32, 64]


@nb.njit(cache=True)
def kernel(opens, highs, lows, closes, sig_m5, h1_close_m5, h1_sma_m5,
           pip, with_tp):
    """Exit on H1 close vs SMA(N, H1) cross at last completed H1 bar."""
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
                hc = h1_close_m5[i-1]; sm = h1_sma_m5[i-1]
                if not (np.isnan(hc) or np.isnan(sm)):
                    if pos == 1 and hc < sm:
                        exit_px = closes[i]; reason = 1
                    elif pos == -1 and hc > sm:
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
    print(f"  H11 — SMA(N, H1) cross-back exit.  N={N_GRID}  with_TP={{True,False}}")
    print("="*92)
    _o = np.zeros(50); _s = np.zeros(50, np.int8); _h = np.full(50,1.0)
    kernel(_o,_o,_o,_o,_s,_h,_h,0.0001,1)

    rows = []; t0 = time.time()
    for pair in PAIRS:
        b = load_pair(pair); sp = b['spread_cost']; pip = b['pip']
        h1_ts = b['h1_ts']; h1_c = b['h1_c']
        for N in N_GRID:
            h1_sma_arr = sma(h1_c, N)
            prev_ts = np.empty_like(b['m5_ts']); prev_ts[0]=b['m5_ts'][0]; prev_ts[1:]=b['m5_ts'][:-1]
            h1_sma_m5 = project_to_m5(prev_ts, h1_ts, h1_sma_arr)
            for tp in (1, 0):
                p, e = kernel(b['opens'], b['highs'], b['lows'], b['closes'],
                              b['sig_m5'], b['h1_c_m5'], h1_sma_m5, pip, tp)
                s = trade_stats_from_arrays(p, e, b['is_end'], b['n'], sp)
                rows.append({'pair':pair,'N':N,'with_TP':bool(tp),**s})

    rdf = pd.DataFrame(rows); rdf.to_csv(OUT/'h11_sma_cross.csv', index=False)
    print(f"  Runtime: {time.time()-t0:.1f}s  rows: {len(rdf)}")
    cand = rdf[(rdf.is_net>0)&(rdf.oos_net>0)].sort_values(['pair','oos_pd'],
                                                            ascending=[True,False])
    bp = cand.groupby('pair').head(1)
    print(f"\n  Best IS+OOS+ per pair:")
    if len(bp)==0:
        print("    none.")
    else:
        for _, r in bp.iterrows():
            tp = "+TP" if r['with_TP'] else "no-TP"
            print(f"    {r['pair']:<9} N={int(r['N']):>2d} {tp:<6} "
                  f"IS={r['is_pd']:+6.2f}  OOS={r['oos_pd']:+6.2f}  "
                  f"DD={r['oos_dd']:+6.0f}  N={int(r['oos_n'])}  WR={r['oos_wr']:5.1f}%")
        print(f"\n  Pairs IS+OOS+: {len(bp)}/10")
        print(f"  Σ OOS p/d:     {bp['oos_pd'].sum():+.2f}")


if __name__ == '__main__':
    main()
