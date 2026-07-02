"""P3 — 4-TF deep stacks (full-history rerun).

Extends the H17 skeleton from base + 2 upper TFs to base + 3 upper TFs.

ENTRY (long).  All conditions must hold on EVERY one of TF1, TF2, TF3
(stacked-alignment + slopes-rising + monotone closes on each upper TF),
AND the alignment must be newly formed on all three.

EXIT.  Count alignment aspects per upper TF. Exit when fewer than
M_exit_min aspects hold on ALL three upper TFs (any one TF still holding
keeps the trade open). Optional fixed TP at TP_PIPS in parallel.

Combos (4), all on S5 base, full history:
  S5/S30/M5/H1     S5/M1/M5/H1
  S5/M5/H1/H4      S5/S30/M5/H4

5 pairs (S5 BA available).  Uses h17d's numpy-bin resample (fast).
"""
import sys, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from _lib import IS_FRAC, sma, SPREAD_FRAC, PAIRS
from h17_stack_alignment import (
    SMA_COMBOS, M_EXIT, TP_GRID, LB_CLOSE, LB_SLOPE,
    S5_DIR, S5_PAIRS, tf_signal, novelty,
)
from h17d_full_history import fast_full_read, bin_resample, project_via_index

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)


# (label, tf1_min, tf2_min, tf3_min)
TF_TRIPLES = [
    ("S5/S30/M5/H1",  0.5,   5,  60),
    ("S5/M1/M5/H1",     1,   5,  60),
    ("S5/M5/H1/H4",     5,  60, 240),
    ("S5/S30/M5/H4",  0.5,   5, 240),
]


@nb.njit(cache=True)
def kernel_4tf(opens, highs, lows, closes,
               t1_long_nov, t1_shrt_nov,
               t2_long_nov, t2_shrt_nov,
               t3_long_nov, t3_shrt_nov,
               t1_sm, t1_md, t1_lg, t1_cp,
               t2_sm, t2_md, t2_lg, t2_cp,
               t3_sm, t3_md, t3_lg, t3_cp,
               pip, M_exit_min, tp_pips, with_tp):
    n = len(opens)
    pos = 0; entry_px = 0.0; entry_bar = -1
    pnls = np.empty(n, np.float64); ents = np.empty(n, np.int64); nt = 0
    for i in range(1, n):
        if pos == 0:
            if t1_long_nov[i] == 1 and t2_long_nov[i] == 1 and t3_long_nov[i] == 1:
                pos = 1; entry_px = opens[i]; entry_bar = i; continue
            if t1_shrt_nov[i] == 1 and t2_shrt_nov[i] == 1 and t3_shrt_nov[i] == 1:
                pos = -1; entry_px = opens[i]; entry_bar = i; continue
        if pos != 0:
            exit_px = 0.0; reason = -1
            if with_tp == 1:
                tp_lvl = entry_px + pos * tp_pips * pip
                if pos == 1 and highs[i] >= tp_lvl:
                    exit_px = tp_lvl; reason = 0
                elif pos == -1 and lows[i] <= tp_lvl:
                    exit_px = tp_lvl; reason = 0
            if reason < 0 and M_exit_min > 0:
                cl = closes[i]
                cnt1 = 0; cnt2 = 0; cnt3 = 0
                if not (np.isnan(t1_sm[i]) or np.isnan(t1_md[i]) or
                        np.isnan(t1_lg[i]) or np.isnan(t1_cp[i])):
                    if pos == 1:
                        if cl > t1_cp[i]: cnt1 += 1
                        if cl > t1_sm[i] and t1_sm[i] > t1_md[i] and t1_md[i] > t1_lg[i]: cnt1 += 1
                        if t1_sm[i] > t1_md[i] and t1_md[i] > t1_lg[i]: cnt1 += 1
                    else:
                        if cl < t1_cp[i]: cnt1 += 1
                        if cl < t1_sm[i] and t1_sm[i] < t1_md[i] and t1_md[i] < t1_lg[i]: cnt1 += 1
                        if t1_sm[i] < t1_md[i] and t1_md[i] < t1_lg[i]: cnt1 += 1
                if not (np.isnan(t2_sm[i]) or np.isnan(t2_md[i]) or
                        np.isnan(t2_lg[i]) or np.isnan(t2_cp[i])):
                    if pos == 1:
                        if cl > t2_cp[i]: cnt2 += 1
                        if cl > t2_sm[i] and t2_sm[i] > t2_md[i] and t2_md[i] > t2_lg[i]: cnt2 += 1
                        if t2_sm[i] > t2_md[i] and t2_md[i] > t2_lg[i]: cnt2 += 1
                    else:
                        if cl < t2_cp[i]: cnt2 += 1
                        if cl < t2_sm[i] and t2_sm[i] < t2_md[i] and t2_md[i] < t2_lg[i]: cnt2 += 1
                        if t2_sm[i] < t2_md[i] and t2_md[i] < t2_lg[i]: cnt2 += 1
                if not (np.isnan(t3_sm[i]) or np.isnan(t3_md[i]) or
                        np.isnan(t3_lg[i]) or np.isnan(t3_cp[i])):
                    if pos == 1:
                        if cl > t3_cp[i]: cnt3 += 1
                        if cl > t3_sm[i] and t3_sm[i] > t3_md[i] and t3_md[i] > t3_lg[i]: cnt3 += 1
                        if t3_sm[i] > t3_md[i] and t3_md[i] > t3_lg[i]: cnt3 += 1
                    else:
                        if cl < t3_cp[i]: cnt3 += 1
                        if cl < t3_sm[i] and t3_sm[i] < t3_md[i] and t3_md[i] < t3_lg[i]: cnt3 += 1
                        if t3_sm[i] < t3_md[i] and t3_md[i] < t3_lg[i]: cnt3 += 1
                if cnt1 < M_exit_min and cnt2 < M_exit_min and cnt3 < M_exit_min:
                    exit_px = cl; reason = 1
            if reason >= 0:
                pnls[nt] = (exit_px - entry_px) / pip * pos
                ents[nt] = entry_bar; nt += 1
                pos = 0
    if pos != 0:
        pnls[nt] = (closes[-1] - entry_px) / pip * pos
        ents[nt] = entry_bar; nt += 1
    return pnls[:nt], ents[:nt]


def run_one_4tf(pair, label, tf1_min, tf2_min, tf3_min, n_base_per_min=12.0,
                base_min_per_bar=5/60):
    path = S5_DIR / f"{pair}_S5_BA.parquet"
    if not path.exists():
        return [], 0.0
    t_load = time.time()
    df = fast_full_read(path)
    if len(df) < 5000:
        return [], 0.0
    opens  = df['open'].to_numpy(np.float64)
    highs  = df['high'].to_numpy(np.float64)
    lows   = df['low'].to_numpy(np.float64)
    closes = df['close'].to_numpy(np.float64)
    n_base = len(df)
    is_end = int(n_base * IS_FRAC)
    days = n_base * base_min_per_bar / 1440
    print(f"  {pair} loaded {n_base:,} bars ({days:.0f}d, {time.time()-t_load:.1f}s)", flush=True)

    pip, sp_proxy = PAIRS[pair]
    sp_cost = sp_proxy * SPREAD_FRAC

    n1 = int(round(tf1_min * n_base_per_min))
    n2 = int(round(tf2_min * n_base_per_min))
    n3 = int(round(tf3_min * n_base_per_min))
    if n1 < 2 or n2 < 2 or n3 < 2:
        return [], days

    r1 = bin_resample(opens, highs, lows, closes, n1)
    r2 = bin_resample(opens, highs, lows, closes, n2)
    r3 = bin_resample(opens, highs, lows, closes, n3)
    if r1 is None or r2 is None or r3 is None:
        return [], days
    _,_,_, t1_c = r1
    _,_,_, t2_c = r2
    _,_,_, t3_c = r3

    base_to_t1 = np.arange(n_base) // n1 - 1
    base_to_t2 = np.arange(n_base) // n2 - 1
    base_to_t3 = np.arange(n_base) // n3 - 1

    rows = []
    for (n_sm, n_md, n_lg) in SMA_COMBOS:
        t1_sm = sma(t1_c, n_sm); t1_md = sma(t1_c, n_md); t1_lg = sma(t1_c, n_lg)
        t2_sm = sma(t2_c, n_sm); t2_md = sma(t2_c, n_md); t2_lg = sma(t2_c, n_lg)
        t3_sm = sma(t3_c, n_sm); t3_md = sma(t3_c, n_md); t3_lg = sma(t3_c, n_lg)
        t1_long = tf_signal(t1_c, t1_sm, t1_md, t1_lg, 1)
        t1_shrt = tf_signal(t1_c, t1_sm, t1_md, t1_lg, 0)
        t2_long = tf_signal(t2_c, t2_sm, t2_md, t2_lg, 1)
        t2_shrt = tf_signal(t2_c, t2_sm, t2_md, t2_lg, 0)
        t3_long = tf_signal(t3_c, t3_sm, t3_md, t3_lg, 1)
        t3_shrt = tf_signal(t3_c, t3_sm, t3_md, t3_lg, 0)
        t1_lnov = novelty(t1_long); t1_snov = novelty(t1_shrt)
        t2_lnov = novelty(t2_long); t2_snov = novelty(t2_shrt)
        t3_lnov = novelty(t3_long); t3_snov = novelty(t3_shrt)

        t1_lnov_b = project_via_index(base_to_t1, t1_lnov).astype(np.int8)
        t1_snov_b = project_via_index(base_to_t1, t1_snov).astype(np.int8)
        t2_lnov_b = project_via_index(base_to_t2, t2_lnov).astype(np.int8)
        t2_snov_b = project_via_index(base_to_t2, t2_snov).astype(np.int8)
        t3_lnov_b = project_via_index(base_to_t3, t3_lnov).astype(np.int8)
        t3_snov_b = project_via_index(base_to_t3, t3_snov).astype(np.int8)
        t1_sm_b = project_via_index(base_to_t1, t1_sm); t1_md_b = project_via_index(base_to_t1, t1_md)
        t1_lg_b = project_via_index(base_to_t1, t1_lg); t1_c_b  = project_via_index(base_to_t1, t1_c)
        t2_sm_b = project_via_index(base_to_t2, t2_sm); t2_md_b = project_via_index(base_to_t2, t2_md)
        t2_lg_b = project_via_index(base_to_t2, t2_lg); t2_c_b  = project_via_index(base_to_t2, t2_c)
        t3_sm_b = project_via_index(base_to_t3, t3_sm); t3_md_b = project_via_index(base_to_t3, t3_md)
        t3_lg_b = project_via_index(base_to_t3, t3_lg); t3_c_b  = project_via_index(base_to_t3, t3_c)

        for M_min in M_EXIT:
            for tp_p in TP_GRID:
                p, e = kernel_4tf(opens, highs, lows, closes,
                                  t1_lnov_b, t1_snov_b, t2_lnov_b, t2_snov_b, t3_lnov_b, t3_snov_b,
                                  t1_sm_b, t1_md_b, t1_lg_b, t1_c_b,
                                  t2_sm_b, t2_md_b, t2_lg_b, t2_c_b,
                                  t3_sm_b, t3_md_b, t3_lg_b, t3_c_b,
                                  pip, M_min, tp_p, 1)
                if len(p) == 0:
                    rows.append({'tf_label':label,'pair':pair,
                                 'sma':f"{n_sm}/{n_md}/{n_lg}",
                                 'M_exit':M_min,'tp_pips':tp_p,
                                 'trades':0,'is_n':0,'oos_n':0,
                                 'is_pd':0,'oos_pd':0,'oos_dd':0,'oos_wr':0,
                                 'is_net':0,'oos_net':0,'days':round(days,1)})
                    continue
                net = p - sp_cost
                is_mask = e < is_end; oos_mask = ~is_mask
                is_days  = (is_end / n_base) * days
                oos_days = days - is_days
                is_net  = float(net[is_mask].sum())
                oos_net = float(net[oos_mask].sum())
                if oos_mask.sum() > 0:
                    cum = net[oos_mask].cumsum()
                    oos_dd = float((cum - np.maximum.accumulate(cum)).min())
                    oos_wr = float((net[oos_mask] > 0).mean() * 100)
                else:
                    oos_dd = 0.0; oos_wr = 0.0
                rows.append({'tf_label':label,'pair':pair,
                             'sma':f"{n_sm}/{n_md}/{n_lg}",
                             'M_exit':M_min,'tp_pips':tp_p,
                             'trades':int(len(p)),
                             'is_n':int(is_mask.sum()),'oos_n':int(oos_mask.sum()),
                             'is_net':round(is_net,1),'oos_net':round(oos_net,1),
                             'is_pd':round(is_net/max(is_days,1),2),
                             'oos_pd':round(oos_net/max(oos_days,1),2),
                             'oos_dd':round(oos_dd,1),'oos_wr':round(oos_wr,1),
                             'days':round(days,1)})
    return rows, days


def main():
    print("="*100)
    print(f"  P3 — 4-TF deep stacks on S5 base (FULL HISTORY)")
    print(f"  SMA combos: {len(SMA_COMBOS)}   M_exit: {M_EXIT}   TP: {TP_GRID}")
    print(f"  TF triples: {len(TF_TRIPLES)}   pairs: {sorted(S5_PAIRS)}")
    print("="*100, flush=True)

    # JIT warmup
    _c = np.zeros(100); _s = np.zeros(100, np.int8)
    tf_signal(_c, _c, _c, _c, 1)
    kernel_4tf(_c, _c, _c, _c,
               _s, _s, _s, _s, _s, _s,
               _c, _c, _c, _c, _c, _c, _c, _c, _c, _c, _c, _c,
               0.0001, 1, 20.0, 1)

    all_rows = []; t0 = time.time()
    for label, tf1, tf2, tf3 in TF_TRIPLES:
        print(f"\n=== {label} ===", flush=True)
        t_tf = time.time()
        for pair in sorted(S5_PAIRS):
            rows, days = run_one_4tf(pair, label, tf1, tf2, tf3)
            all_rows.extend(rows)
            gc.collect()
        df = pd.DataFrame([r for r in all_rows if r['tf_label']==label])
        if len(df):
            c = df[(df.is_net>0) & (df.oos_net>0)]
            bp = c.sort_values(['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(1)
            print(f"  {label}: {time.time()-t_tf:.1f}s  "
                  f"{len(bp)}/{df['pair'].nunique()} IS+OOS+  "
                  f"ΣOOS={bp['oos_pd'].sum() if len(bp) else 0:+.2f}", flush=True)
            for _, r in bp.iterrows():
                print(f"    {r['pair']:<9} {r['sma']:<10} M={int(r['M_exit'])} TP={int(r['tp_pips'])}  "
                      f"OOSpd={r['oos_pd']:+6.2f}  DD={r['oos_dd']:+7.0f}  N={int(r['oos_n']):>4}  "
                      f"WR={r['oos_wr']:>5.1f}", flush=True)

    df_all = pd.DataFrame(all_rows)
    out_path = OUT / 'p3_4tf_stacks.csv'
    df_all.to_csv(out_path, index=False)
    print(f"\n  Total runtime: {time.time()-t0:.1f}s   rows: {len(df_all)}")
    print(f"  → {out_path}")

    if df_all.empty or 'is_net' not in df_all.columns:
        print("\n  (no rows)"); return

    cand_all = df_all[(df_all.is_net>0)&(df_all.oos_net>0)].sort_values(
        ['pair','oos_pd'], ascending=[True,False])
    bp_all = cand_all.groupby('pair').head(1)
    print()
    print("="*100)
    print("  BEST IS+OOS+ ACROSS P3 4-TF STACKS, per pair")
    print("="*100)
    print(f"  {'Pair':<9} {'TF':<18} {'SMA':<10} {'M':>2} {'TP':>4} "
          f"{'IS pd':>7} {'OOS pd':>7} {'DD':>7} {'N':>4} {'WR%':>5}  {'days':>6}")
    for _, r in bp_all.iterrows():
        print(f"  {r['pair']:<9} {r['tf_label']:<18} {r['sma']:<10} "
              f"{int(r['M_exit']):>2d} {r['tp_pips']:>4.0f} "
              f"{r['is_pd']:>+7.2f} {r['oos_pd']:>+7.2f} {r['oos_dd']:>+7.0f} "
              f"{int(r['oos_n']):>4d} {r['oos_wr']:>5.1f}  {r['days']:>6.1f}")
    print(f"\n  Total: {len(bp_all)} pairs IS+OOS+  Σ OOS pd: {bp_all['oos_pd'].sum():+.2f}")


if __name__ == '__main__':
    main()
