"""H10 — Entry-signal-flip exit.

Exit when the entry rule would now fire the OPPOSITE direction.
  K=0 → exit on first flip;  K>0 → flip must hold K+1 bars.
Optionally keep the +20p TP alongside the flip exit.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (PAIRS, WINDOW_BARS, IS_FRAC, BARS_PER_H1, TP_PIPS_BASE,
                  load_pair, trade_stats_from_arrays)

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
K_BARS  = [0, 1, 3, 6, 12, 24]


@nb.njit(cache=True)
def kernel(opens, highs, lows, closes, sig_m5, pip, K, tp_keep):
    n = len(opens)
    pos = 0; entry_px = 0.0; entry_bar = -1; flip_streak = 0
    pnls = np.empty(n, np.float64); ents = np.empty(n, np.int64); nt = 0
    for i in range(1, n):
        sig = sig_m5[i-1]
        if pos != 0:
            exit_px = 0.0; reason = -1
            if tp_keep == 1:
                tp_lvl = entry_px + pos * TP_PIPS_BASE * pip
                if pos == 1 and highs[i] >= tp_lvl:
                    exit_px = tp_lvl; reason = 0
                elif pos == -1 and lows[i] <= tp_lvl:
                    exit_px = tp_lvl; reason = 0
            if reason < 0:
                if sig == -pos:
                    flip_streak += 1
                else:
                    flip_streak = 0
                if flip_streak >= K + 1:
                    exit_px = closes[i]; reason = 1
            if reason >= 0:
                pnls[nt] = (exit_px - entry_px) / pip * pos
                ents[nt] = entry_bar; nt += 1
                pos = 0; flip_streak = 0; continue
        if pos == 0 and sig != 0:
            pos = sig; entry_px = opens[i]; entry_bar = i; flip_streak = 0
    if pos != 0:
        pnls[nt] = (closes[-1] - entry_px) / pip * pos
        ents[nt] = entry_bar; nt += 1
    return pnls[:nt], ents[:nt]


def main():
    print("="*92)
    print("  H10 — ENTRY-SIGNAL-FLIP EXIT")
    print(f"  K (confirmation bars): {K_BARS}    TP_keep: [True, False]")
    print("="*92)
    _o = np.zeros(50); _s = np.zeros(50, np.int8)
    kernel(_o, _o, _o, _o, _s, 0.0001, 0, 1)

    rows = []; t0 = time.time()
    for pair in PAIRS:
        b = load_pair(pair); sp = b['spread_cost']; pip = b['pip']
        for K in K_BARS:
            for tp_keep in (1, 0):
                p, e = kernel(b['opens'], b['highs'], b['lows'], b['closes'],
                              b['sig_m5'], pip, K, tp_keep)
                s = trade_stats_from_arrays(p, e, b['is_end'], b['n'], sp)
                rows.append({'pair':pair,'K':K,'tp_keep':bool(tp_keep),**s})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT/'h10_signal_flip.csv', index=False)
    print(f"  Runtime: {time.time()-t0:.1f}s  rows: {len(rdf)}")

    cand = rdf[(rdf.is_net>0)&(rdf.oos_net>0)].sort_values(['pair','oos_pd'],
                                                            ascending=[True,False])
    bp = cand.groupby('pair').head(1)
    print(f"\n  Best IS+OOS+ per pair:")
    if len(bp)==0:
        print("    none.")
    else:
        for _, r in bp.iterrows():
            tp = "+TP" if r['tp_keep'] else "no-TP"
            print(f"    {r['pair']:<9} K={int(r['K']):>2d} {tp:<6} "
                  f"IS={r['is_pd']:+6.2f}  OOS={r['oos_pd']:+6.2f}  "
                  f"DD={r['oos_dd']:+6.0f}  N={int(r['oos_n'])}  WR={r['oos_wr']:5.1f}%")
        print(f"\n  Pairs IS+OOS+: {len(bp)}/10")
        print(f"  Σ OOS p/d:     {bp['oos_pd'].sum():+.2f}")


if __name__ == '__main__':
    main()
