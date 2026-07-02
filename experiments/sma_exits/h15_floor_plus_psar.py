"""H15 — H14 ATR-band floor + H13 PSAR trail (combined overlay).

Long trade exit (priority order):
  (a) Optional fixed TP at +20p
  (b) H14 floor: close[H1] ≤ SMA(N, H1) − X·ATR_H1     (active from bar 1)
  (c) H13 trail: close[M5] ≤ PSAR[H1]  AND  MFE ≥ activate_pips

Hypothesis. H14 closes the "no-stop-from-entry" gap that the bare H13
            rule leaves open. They don't conflict because the ATR-band
            stop sits much further below price than the activated PSAR.

Grid (full Cartesian).
  N         ∈ {8, 16, 32}                 SMA(N, H1) for floor
  X         ∈ {1.0, 2.0, 3.0}              ATR mult for floor
  af_start  ∈ {0.005, 0.010, 0.020}        PSAR step
  af_max    = 0.10  (all H13 winners used this)
  activate  ∈ {0, 10, 20} pips
  with_TP   ∈ {True, False}

3 × 3 × 3 × 1 × 3 × 2 = 162 configs × 10 pairs.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (PAIRS, IS_FRAC, BARS_PER_H1, TP_PIPS_BASE,
                  load_pair, sma, project_to_m5, trade_stats_from_arrays)
from h13_psar import psar_h1

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
N_GRID    = [8, 16, 32]
X_GRID    = [1.0, 2.0, 3.0]
AF_STARTS = [0.005, 0.010, 0.020]
AF_MAX    = 0.10
ACT_PIPS  = [0.0, 10.0, 20.0]


@nb.njit(cache=True)
def kernel(opens, highs, lows, closes, sig_m5,
           h1_sma_m5, h1_atr_m5, psar_m5,
           pip, X, activate_pips, with_tp):
    """Combined H14 floor + H13 PSAR trail."""
    n = len(opens)
    pos = 0; entry_px = 0.0; entry_bar = -1
    mfe_pips = 0.0
    pnls = np.empty(n, np.float64); ents = np.empty(n, np.int64)
    reasons = np.empty(n, np.int8); nt = 0
    for i in range(1, n):
        sig = sig_m5[i-1]
        if pos != 0:
            exit_px = 0.0; reason = -1
            # Track MFE in pips for activation gate
            cur_pips = (closes[i] - entry_px) / pip * pos
            if cur_pips > mfe_pips:
                mfe_pips = cur_pips
            # (a) fixed TP intrabar
            if with_tp == 1:
                tp_lvl = entry_px + pos * TP_PIPS_BASE * pip
                if pos == 1 and highs[i] >= tp_lvl:
                    exit_px = tp_lvl; reason = 0
                elif pos == -1 and lows[i] <= tp_lvl:
                    exit_px = tp_lvl; reason = 0
            # (b) H14 floor — always active
            if reason < 0:
                sm = h1_sma_m5[i-1]; at = h1_atr_m5[i-1]
                if not (np.isnan(sm) or np.isnan(at) or at <= 0):
                    if pos == 1:
                        lower = sm - X * at
                        if closes[i] <= lower:
                            exit_px = closes[i]; reason = 1     # floor
                    else:
                        upper = sm + X * at
                        if closes[i] >= upper:
                            exit_px = closes[i]; reason = 1
            # (c) PSAR trail — only after MFE ≥ activate
            if reason < 0 and mfe_pips >= activate_pips:
                p = psar_m5[i-1]
                if not np.isnan(p):
                    if pos == 1 and closes[i] < p:
                        exit_px = closes[i]; reason = 2     # trail
                    elif pos == -1 and closes[i] > p:
                        exit_px = closes[i]; reason = 2
            if reason >= 0:
                pnls[nt] = (exit_px - entry_px) / pip * pos
                ents[nt] = entry_bar; reasons[nt] = reason
                nt += 1
                pos = 0; mfe_pips = 0.0
                continue
        if pos == 0 and sig != 0:
            pos = sig; entry_px = opens[i]; entry_bar = i
            mfe_pips = 0.0
    if pos != 0:
        pnls[nt] = (closes[-1] - entry_px) / pip * pos
        ents[nt] = entry_bar; reasons[nt] = 3  # end
        nt += 1
    return pnls[:nt], ents[:nt], reasons[:nt]


def main():
    print("="*100)
    print(f"  H15 — H14 ATR floor + H13 PSAR trail (combined)")
    print(f"  N={N_GRID}  X={X_GRID} ATR  af_start={AF_STARTS}  af_max={AF_MAX}  "
          f"activate={ACT_PIPS}p  with_TP={{True,False}}")
    print("="*100)
    _o = np.zeros(50); _s = np.zeros(50, np.int8); _h = np.full(50,1.0)
    kernel(_o,_o,_o,_o,_s,_h,_h,_h,0.0001,1.0,0.0,1)

    rows = []; t0 = time.time()
    for pair in PAIRS:
        b = load_pair(pair); sp = b['spread_cost']; pip = b['pip']
        h1_ts = b['h1_ts']; h1_c = b['h1_c']; h1_h = b['h1_h']; h1_l = b['h1_l']
        prev_ts = np.empty_like(b['m5_ts']); prev_ts[0]=b['m5_ts'][0]; prev_ts[1:]=b['m5_ts'][:-1]
        # Cache per-(af_start) PSAR projections
        psar_cache = {}
        for af_s in AF_STARTS:
            psar_arr, _ = psar_h1(h1_h, h1_l, af_s, af_s, AF_MAX)
            psar_cache[af_s] = project_to_m5(prev_ts, h1_ts, psar_arr)
        # Cache per-N SMA projections
        sma_cache = {}
        for N in N_GRID:
            sma_arr = sma(h1_c, N)
            sma_cache[N] = project_to_m5(prev_ts, h1_ts, sma_arr)

        for N in N_GRID:
            for X in X_GRID:
                for af_s in AF_STARTS:
                    for act in ACT_PIPS:
                        for tp in (1, 0):
                            p, e, r = kernel(b['opens'], b['highs'], b['lows'], b['closes'],
                                             b['sig_m5'], sma_cache[N], b['h1_atr_m5'],
                                             psar_cache[af_s], pip, X, act, tp)
                            s = trade_stats_from_arrays(p, e, b['is_end'], b['n'], sp)
                            # Reason breakdown
                            r_tp    = int((r==0).sum())
                            r_floor = int((r==1).sum())
                            r_trail = int((r==2).sum())
                            r_end   = int((r==3).sum())
                            rows.append({'pair':pair,'N':N,'X':X,'af_s':af_s,
                                         'activate':act,'with_TP':bool(tp),
                                         'r_tp':r_tp,'r_floor':r_floor,
                                         'r_trail':r_trail,'r_end':r_end, **s})

    rdf = pd.DataFrame(rows); rdf.to_csv(OUT/'h15_floor_plus_psar.csv', index=False)
    print(f"  Runtime: {time.time()-t0:.1f}s  rows: {len(rdf)}")

    cand = rdf[(rdf.is_net>0)&(rdf.oos_net>0)].sort_values(['pair','oos_pd'],
                                                            ascending=[True,False])
    bp = cand.groupby('pair').head(1)
    print(f"\n  Best IS+OOS+ per pair:")
    print(f"  {'Pair':<9} {'N':>2} {'X':>4} {'af_s':>6} {'act':>4} {'TP':<3} "
          f"{'IS pd':>7} {'OOS pd':>7} {'DD':>7} {'N':>4} {'WR%':>5}  "
          f"{'tp/floor/trail/end'}")
    if len(bp)==0:
        print("    none.")
    else:
        for _, r in bp.iterrows():
            tp = "+TP" if r['with_TP'] else "off"
            print(f"  {r['pair']:<9} {int(r['N']):>2d} {r['X']:>4.1f} "
                  f"{r['af_s']:>6.3f} {int(r['activate']):>3d}p {tp:<3} "
                  f"{r['is_pd']:>+7.2f} {r['oos_pd']:>+7.2f} {r['oos_dd']:>+7.0f} "
                  f"{int(r['oos_n']):>4d} {r['oos_wr']:>5.1f}  "
                  f"{int(r['r_tp']):>2d}/{int(r['r_floor']):>2d}/{int(r['r_trail']):>2d}/{int(r['r_end']):>2d}")
        print(f"\n  Pairs IS+OOS+: {len(bp)}/10   Σ OOS p/d: {bp['oos_pd'].sum():+.2f}")

    # Compare to H7, H13, H14 baselines
    print()
    print("  ── Comparison to prior winners ──")
    print(f"    H7  symmetric scratch       : 7/10 pairs, Σ OOS +64.4 p/d")
    print(f"    H13 PSAR trail (best per pair): 7/10 pairs, Σ OOS +52.7 p/d")
    print(f"    H14 ATR-band overlay        : 5/10 pairs, Σ OOS +22.2 p/d")
    print(f"    H15 combined (this run)     : {len(bp)}/10 pairs, Σ OOS {bp['oos_pd'].sum():+.2f} p/d")


if __name__ == '__main__':
    main()
