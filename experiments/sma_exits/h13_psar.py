"""H13 — Parabolic SAR trail on H1.

Place a trailing stop using H1 Parabolic SAR. Activate only after the trade
has accumulated at least `activate_pips` of MFE (avoid premature stop in
chop), then exit when price closes through PSAR.

Grid:
  AF_start  ∈ {0.005, 0.01, 0.02}        SAR step
  AF_max    ∈ {0.10, 0.15, 0.20}         SAR ceiling
  activate  ∈ {0, 10, 20} pips           MFE before PSAR is armed
  with_TP   ∈ {True, False}              keep +20p TP?

Same FX-Strength scalper template (AF_start=0.01, AF_max=0.15, activate=20p)
is included.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (PAIRS, IS_FRAC, BARS_PER_H1, TP_PIPS_BASE,
                  load_pair, project_to_m5, trade_stats_from_arrays)

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
AF_STARTS = [0.005, 0.01, 0.02]
AF_MAXES  = [0.10, 0.15, 0.20]
ACT_PIPS  = [0.0, 10.0, 20.0]


@nb.njit(cache=True)
def psar_h1(highs, lows, af_start, af_step, af_max):
    """Wilder's classic Parabolic SAR. Returns (psar, dir) arrays.
    dir: +1 if SAR below price (long-side stop), -1 if above (short-side)."""
    n = len(highs)
    psar = np.zeros(n); psdir = np.zeros(n, np.int8)
    if n < 2:
        return psar, psdir
    # Init: trend = up if H[1] > H[0]
    if highs[1] > highs[0]:
        psdir[0] = 1; psdir[1] = 1
        ep = highs[1]; sar = lows[0]
    else:
        psdir[0] = -1; psdir[1] = -1
        ep = lows[1]; sar = highs[0]
    af = af_start
    psar[0] = sar; psar[1] = sar
    for i in range(2, n):
        # Step SAR
        new_sar = sar + af * (ep - sar)
        if psdir[i-1] == 1:
            # Long stop: SAR can't exceed prior 2 lows
            new_sar = min(new_sar, lows[i-1], lows[i-2])
            if lows[i] < new_sar:
                # flip to short
                psdir[i] = -1
                sar = ep                # new SAR = previous EP
                ep = lows[i]
                af = af_start
            else:
                psdir[i] = 1; sar = new_sar
                if highs[i] > ep:
                    ep = highs[i]; af = min(af + af_step, af_max)
        else:
            new_sar = max(new_sar, highs[i-1], highs[i-2])
            if highs[i] > new_sar:
                psdir[i] = 1
                sar = ep; ep = highs[i]; af = af_start
            else:
                psdir[i] = -1; sar = new_sar
                if lows[i] < ep:
                    ep = lows[i]; af = min(af + af_step, af_max)
        psar[i] = sar
    return psar, psdir


@nb.njit(cache=True)
def kernel(opens, highs, lows, closes, sig_m5,
           psar_m5, pip, activate_pips, with_tp):
    """Trade-side PSAR trail: exit when price closes through PSAR after activation."""
    n = len(opens)
    pos = 0; entry_px = 0.0; entry_bar = -1
    armed = 0; mfe_pips = 0.0
    pnls = np.empty(n, np.float64); ents = np.empty(n, np.int64); nt = 0
    for i in range(1, n):
        sig = sig_m5[i-1]
        if pos != 0:
            exit_px = 0.0; reason = -1
            # Track MFE
            cur_pips = (closes[i] - entry_px) / pip * pos
            if cur_pips > mfe_pips:
                mfe_pips = cur_pips
            if mfe_pips >= activate_pips:
                armed = 1
            # TP intrabar
            if with_tp == 1:
                tp_lvl = entry_px + pos * TP_PIPS_BASE * pip
                if pos == 1 and highs[i] >= tp_lvl:
                    exit_px = tp_lvl; reason = 0
                elif pos == -1 and lows[i] <= tp_lvl:
                    exit_px = tp_lvl; reason = 0
            # PSAR trail (only after armed)
            if reason < 0 and armed == 1:
                p = psar_m5[i-1]
                if not np.isnan(p):
                    if pos == 1 and closes[i] < p:
                        exit_px = closes[i]; reason = 1
                    elif pos == -1 and closes[i] > p:
                        exit_px = closes[i]; reason = 1
            if reason >= 0:
                pnls[nt] = (exit_px - entry_px) / pip * pos
                ents[nt] = entry_bar; nt += 1
                pos = 0; armed = 0; mfe_pips = 0.0; continue
        if pos == 0 and sig != 0:
            pos = sig; entry_px = opens[i]; entry_bar = i
            armed = 0; mfe_pips = 0.0
    if pos != 0:
        pnls[nt] = (closes[-1] - entry_px) / pip * pos
        ents[nt] = entry_bar; nt += 1
    return pnls[:nt], ents[:nt]


def main():
    print("="*92)
    print(f"  H13 — PSAR(H1) trail.  AF_start={AF_STARTS}  AF_max={AF_MAXES}  "
          f"activate={ACT_PIPS} p  with_TP={{True,False}}")
    print("="*92)
    _o = np.zeros(50); _s = np.zeros(50, np.int8); _h = np.full(50,1.0)
    kernel(_o,_o,_o,_o,_s,_h,0.0001,0.0,1)

    rows = []; t0 = time.time()
    for pair in PAIRS:
        b = load_pair(pair); sp = b['spread_cost']; pip = b['pip']
        h1_ts = b['h1_ts']; h1_h = b['h1_h']; h1_l = b['h1_l']
        prev_ts = np.empty_like(b['m5_ts']); prev_ts[0]=b['m5_ts'][0]; prev_ts[1:]=b['m5_ts'][:-1]
        for af_s in AF_STARTS:
            for af_m in AF_MAXES:
                psar_arr, _ = psar_h1(h1_h, h1_l, af_s, af_s, af_m)
                psar_m5 = project_to_m5(prev_ts, h1_ts, psar_arr)
                for act in ACT_PIPS:
                    for tp in (1, 0):
                        p, e = kernel(b['opens'], b['highs'], b['lows'], b['closes'],
                                      b['sig_m5'], psar_m5, pip, act, tp)
                        s = trade_stats_from_arrays(p, e, b['is_end'], b['n'], sp)
                        rows.append({'pair':pair,'af_start':af_s,'af_max':af_m,
                                     'activate':act,'with_TP':bool(tp),**s})

    rdf = pd.DataFrame(rows); rdf.to_csv(OUT/'h13_psar.csv', index=False)
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
            print(f"    {r['pair']:<9} af={r['af_start']:.3f}/{r['af_max']:.2f} "
                  f"act={int(r['activate']):>2d}p {tp:<6} "
                  f"IS={r['is_pd']:+6.2f}  OOS={r['oos_pd']:+6.2f}  "
                  f"DD={r['oos_dd']:+6.0f}  N={int(r['oos_n'])}  WR={r['oos_wr']:5.1f}%")
        print(f"\n  Pairs IS+OOS+: {len(bp)}/10   Σ OOS p/d: {bp['oos_pd'].sum():+.2f}")


if __name__ == '__main__':
    main()
