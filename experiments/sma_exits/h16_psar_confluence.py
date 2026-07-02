"""H16 — Multi-TF PSAR confluence exit.

Idea. The entry rule uses H1+M30 SMA-momentum agreement. The natural
      symmetry for exit: require PSAR on BOTH timeframes to flip
      against the position before exiting (vs single-TF H1 PSAR in H13).

Two confluence modes tested:
  AND  — exit only when M5 close has crossed PSAR on BOTH timeframes
  OR   — exit when M5 close crosses PSAR on EITHER timeframe (faster)

Also tests two TF pairings:
  H1+M30   — same as entry rule
  H1+M15   — wider separation, M15 PSAR is much faster

Grid.
  TF pair    ∈ {H1+M30, H1+M15}
  conf mode  ∈ {AND, OR}
  af_start   ∈ {0.005, 0.010, 0.020}   (shared between TFs)
  af_max     = 0.10
  activate   ∈ {0, 10, 20} pips
  with_TP    ∈ {True, False}

Compared head-to-head against H13 (single-TF H1 PSAR best per pair).
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (PAIRS, IS_FRAC, BARS_PER_H1, TP_PIPS_BASE,
                  load_pair, resample_tf, project_to_m5, trade_stats_from_arrays)
from h13_psar import psar_h1

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
AF_STARTS = [0.005, 0.010, 0.020]
AF_MAX    = 0.10
ACT_PIPS  = [0.0, 10.0, 20.0]


@nb.njit(cache=True)
def kernel(opens, highs, lows, closes, sig_m5,
           psar_a_m5, psar_b_m5, pip, activate_pips, with_tp, conf_and):
    """conf_and: 1 = require BOTH TFs to cross; 0 = exit when EITHER crosses."""
    n = len(opens)
    pos = 0; entry_px = 0.0; entry_bar = -1; mfe_pips = 0.0
    pnls = np.empty(n, np.float64); ents = np.empty(n, np.int64); nt = 0
    for i in range(1, n):
        sig = sig_m5[i-1]
        if pos != 0:
            exit_px = 0.0; reason = -1
            cur_pips = (closes[i] - entry_px) / pip * pos
            if cur_pips > mfe_pips:
                mfe_pips = cur_pips
            if with_tp == 1:
                tp_lvl = entry_px + pos * TP_PIPS_BASE * pip
                if pos == 1 and highs[i] >= tp_lvl:
                    exit_px = tp_lvl; reason = 0
                elif pos == -1 and lows[i] <= tp_lvl:
                    exit_px = tp_lvl; reason = 0
            if reason < 0 and mfe_pips >= activate_pips:
                pa = psar_a_m5[i-1]; pb = psar_b_m5[i-1]
                if not (np.isnan(pa) or np.isnan(pb)):
                    if pos == 1:
                        a_cross = closes[i] < pa
                        b_cross = closes[i] < pb
                        cross = (a_cross and b_cross) if conf_and == 1 else (a_cross or b_cross)
                        if cross:
                            exit_px = closes[i]; reason = 1
                    else:
                        a_cross = closes[i] > pa
                        b_cross = closes[i] > pb
                        cross = (a_cross and b_cross) if conf_and == 1 else (a_cross or b_cross)
                        if cross:
                            exit_px = closes[i]; reason = 1
            if reason >= 0:
                pnls[nt] = (exit_px - entry_px) / pip * pos
                ents[nt] = entry_bar; nt += 1
                pos = 0; mfe_pips = 0.0; continue
        if pos == 0 and sig != 0:
            pos = sig; entry_px = opens[i]; entry_bar = i; mfe_pips = 0.0
    if pos != 0:
        pnls[nt] = (closes[-1] - entry_px) / pip * pos
        ents[nt] = entry_bar; nt += 1
    return pnls[:nt], ents[:nt]


def main():
    print("="*100)
    print(f"  H16 — Multi-TF PSAR confluence exit")
    print(f"  TF pairings: H1+M30, H1+M15  | modes: AND, OR")
    print(f"  af_start={AF_STARTS}  af_max={AF_MAX}  activate={ACT_PIPS}p  with_TP={{T,F}}")
    print("="*100)
    _o = np.zeros(50); _s = np.zeros(50, np.int8); _h = np.full(50,1.0)
    kernel(_o,_o,_o,_o,_s,_h,_h,0.0001,0.0,1,1)

    rows = []; t0 = time.time()
    for pair in PAIRS:
        b = load_pair(pair); sp = b['spread_cost']; pip = b['pip']
        # Build raw H1 PSAR for each af_start
        prev_ts = np.empty_like(b['m5_ts']); prev_ts[0]=b['m5_ts'][0]; prev_ts[1:]=b['m5_ts'][:-1]
        # H1 PSAR cache
        h1_psar_proj = {}
        for af_s in AF_STARTS:
            arr,_ = psar_h1(b['h1_h'], b['h1_l'], af_s, af_s, AF_MAX)
            h1_psar_proj[af_s] = project_to_m5(prev_ts, b['h1_ts'], arr)
        # Build M30 + M15 PSAR (need to resample M5 → M30/M15)
        # We already have m5 OHLC; resample using pandas.
        df_m5 = pd.DataFrame({'timestamp': b['m5_ts'],
                              'open': b['opens'], 'high': b['highs'],
                              'low': b['lows'], 'close': b['closes']})
        m30 = resample_tf(df_m5, 30); m15 = resample_tf(df_m5, 15)
        m30_h = m30['high'].to_numpy(); m30_l = m30['low'].to_numpy()
        m15_h = m15['high'].to_numpy(); m15_l = m15['low'].to_numpy()
        m30_ts = m30['timestamp'].to_numpy(); m15_ts = m15['timestamp'].to_numpy()
        m30_psar_proj = {}; m15_psar_proj = {}
        for af_s in AF_STARTS:
            arr,_ = psar_h1(m30_h, m30_l, af_s, af_s, AF_MAX)
            m30_psar_proj[af_s] = project_to_m5(prev_ts, m30_ts, arr)
            arr,_ = psar_h1(m15_h, m15_l, af_s, af_s, AF_MAX)
            m15_psar_proj[af_s] = project_to_m5(prev_ts, m15_ts, arr)

        for tf_pair, b_proj in [('H1+M30', m30_psar_proj), ('H1+M15', m15_psar_proj)]:
            for conf in ('AND', 'OR'):
                conf_and = 1 if conf == 'AND' else 0
                for af_s in AF_STARTS:
                    for act in ACT_PIPS:
                        for tp in (1, 0):
                            p, e = kernel(b['opens'], b['highs'], b['lows'], b['closes'],
                                          b['sig_m5'], h1_psar_proj[af_s], b_proj[af_s],
                                          pip, act, tp, conf_and)
                            s = trade_stats_from_arrays(p, e, b['is_end'], b['n'], sp)
                            rows.append({'pair':pair,'tf_pair':tf_pair,'conf':conf,
                                         'af_s':af_s,'activate':act,'with_TP':bool(tp),**s})

    rdf = pd.DataFrame(rows); rdf.to_csv(OUT/'h16_psar_confluence.csv', index=False)
    print(f"  Runtime: {time.time()-t0:.1f}s  rows: {len(rdf)}")

    # Best per pair across all configs
    cand = rdf[(rdf.is_net>0)&(rdf.oos_net>0)].sort_values(['pair','oos_pd'],
                                                            ascending=[True,False])
    bp = cand.groupby('pair').head(1)
    print(f"\n  Best IS+OOS+ per pair (any tf_pair, any conf):")
    print(f"  {'Pair':<9} {'TF':<7} {'conf':<4} {'af_s':>6} {'act':>4} {'TP':<3} "
          f"{'IS pd':>7} {'OOS pd':>7} {'DD':>7} {'N':>4} {'WR%':>5}")
    if len(bp)==0:
        print("    none.")
    else:
        for _, r in bp.iterrows():
            tp = "+TP" if r['with_TP'] else "off"
            print(f"  {r['pair']:<9} {r['tf_pair']:<7} {r['conf']:<4} "
                  f"{r['af_s']:>6.3f} {int(r['activate']):>3d}p {tp:<3} "
                  f"{r['is_pd']:>+7.2f} {r['oos_pd']:>+7.2f} {r['oos_dd']:>+7.0f} "
                  f"{int(r['oos_n']):>4d} {r['oos_wr']:>5.1f}")
        print(f"\n  Pairs IS+OOS+: {len(bp)}/10   Σ OOS p/d: {bp['oos_pd'].sum():+.2f}")

    # Mode-by-mode comparison
    print()
    print("  ── By TF-pairing + confluence mode (best per pair, summed across pairs) ──")
    for tf_pair in ['H1+M30','H1+M15']:
        for conf in ['AND','OR']:
            sub = rdf[(rdf.tf_pair==tf_pair) & (rdf.conf==conf)
                      & (rdf.is_net>0) & (rdf.oos_net>0)].sort_values(['pair','oos_pd'],
                                                                      ascending=[True,False])
            sbp = sub.groupby('pair').head(1)
            print(f"    {tf_pair:<7} {conf:<4} : {len(sbp)}/10 pairs   Σ OOS p/d = {sbp['oos_pd'].sum():+.2f}")
    print()
    print(f"  Baselines:")
    print(f"    H13 single-TF H1 PSAR        : 7/10 pairs, Σ OOS +52.7 p/d")
    print(f"    H7  symmetric scratch (best) : 7/10 pairs, Σ OOS +64.4 p/d")


if __name__ == '__main__':
    main()
