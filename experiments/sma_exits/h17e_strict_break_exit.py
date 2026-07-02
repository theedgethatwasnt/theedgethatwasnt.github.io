"""H17e — Correct three-axis alignment-break exit, swept properly.

FIX vs H17d.  My earlier exit check (h17_stack_alignment.kernel) was sloppy:
  - "Monotone" axis used only `c > prev_c` (2-bar) instead of `c1>c2>c3` (3-bar).
  - "Slope" axis was just `sma_sm > sma_md > sma_lg`, which is stacking restated.

This file checks the proper three axes at exit time, the same way the
entry check does — but on the in-progress current base close c0 plus
the last completed TF bars for c1, c2 and the SMA history.

EXIT RULE (long; short mirror).  At each base bar t, count BROKEN aspects:
  ¬A: NOT (c0 > c1 > c2)             — monotone closes broken
  ¬B: NOT (c0 > SMA_sm > SMA_md > SMA_lg)  — stacking broken
  ¬C: NOT (SMA_sm slope rising over last 2 steps
           AND SMA_md slope rising over last 2 steps
           AND SMA_lg slope rising over last 2 steps)  — slopes broken
  broken_count = count of ¬A, ¬B, ¬C

  Per-TF "broken" = broken_count ≥ K_break.  Trade closes when both TFs
  have at least K_break broken axes (M_exit interpretation):
    K_break = 3 → strict: all axes broken on both TFs   (user's exact spec)
    K_break = 2 → at least 2 of 3 axes broken on both TFs
    K_break = 1 → at least 1 axis broken on both TFs    (loosest "both-TF" gate)

  Also tested: K_break with "either TF satisfies" instead of "both"
  (one_tf_mode), which trips exits faster.

GRID.
  Same SMA combos as H17d, M_exit (K_break × tf_mode), TP_PIPS, S5 TF combos.
"""
import sys, time, gc
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from h17_stack_alignment import tf_signal, novelty, fast_tail_read
from h17d_full_history import fast_full_read, bin_resample
from _lib import PAIRS, IS_FRAC, sma, SPREAD_FRAC

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
PROJECT = Path("/path/to/projects/fx-core")

SMA_COMBOS = [
    (5, 10, 22), (5, 15, 35), (5, 22, 50),
    (7, 15, 35), (7, 22, 50),
    (10, 22, 50),
]
# K_break × tf_mode: K_break ∈ {1,2,3}, tf_mode ∈ {"both","either"}
# Plus K_break=0 as the TP-only baseline (no alignment-break exit at all)
EXIT_MODES = [
    (0, "none"),    # M=0 baseline (TP-only, no alignment exit)
    (3, "both"),    # K=3 + both TFs (USER'S SPEC)
    (2, "both"),    # K=2 + both TFs (≥2 axes broken)
    (1, "both"),    # K=1 + both TFs (≥1 axis broken — loosest 'both' gate)
    (3, "either"),  # K=3 + either TF (faster)
    (2, "either"),  # K=2 + either TF
]
TP_GRID = [15.0, 20.0, 30.0]

S5_DIR = PROJECT / "data/s5_ohlc"
S5_PAIRS = ["EUR_JPY","EUR_USD","GBP_JPY","GBP_USD","USD_JPY"]
TF_COMBOS = [
    ("S5/S30/M1",   0.5,  1),
    ("S5/M1/M5",      1,  5),
    ("S5/M2/M10",     2, 10),
    ("S5/M5/M15",     5, 15),
    ("S5/M15/M30",   15, 30),
    ("S5/M15/H1",    15, 60),
    ("S5/M30/H1",    30, 60),
]


@nb.njit(cache=True)
def kernel_strict_break(opens, highs, lows, closes,
                         # TF1: novelty entry streams + per-bar projection of TF1 quantities
                         t1_long_nov, t1_shrt_nov,
                         t1_sm_now, t1_md_now, t1_lg_now,   # SMAs at most recent completed TF1 bar
                         t1_c1, t1_c2,                       # last 2 completed TF1 closes
                         t1_slope_up, t1_slope_dn,           # per-TF1 slope-rising / slope-falling flags
                         # TF2: same
                         t2_long_nov, t2_shrt_nov,
                         t2_sm_now, t2_md_now, t2_lg_now,
                         t2_c1, t2_c2,
                         t2_slope_up, t2_slope_dn,
                         pip, tp_pips, k_break, both_tfs, exit_active):
    """
    For each base bar i:
      Aspect A (monotone): c0=closes[i], c1=t1_c1[i], c2=t1_c2[i]
                           long_A = (c0 > c1 > c2)
      Aspect B (stacked):  long_B = (c0 > t1_sm_now > t1_md_now > t1_lg_now)
      Aspect C (slopes):   long_C = t1_slope_up[i] (already encodes whether ALL three SMAs rose
                           over last 2 completed-bar steps)
      ¬A_long = NOT long_A, etc.
      broken_count_TF1 = ¬A + ¬B + ¬C, similarly for TF2

    Exit (long) fires when broken_count ≥ k_break on the required number of TFs
      (both_tfs=1: both TFs; both_tfs=0: either TF).

    If exit_active == 0, alignment-break check is skipped (TP-only baseline).
    """
    n = len(opens)
    pos = 0; entry_px = 0.0; entry_bar = -1
    pnls = np.empty(n, np.float64); ents = np.empty(n, np.int64); nt = 0
    for i in range(1, n):
        # Entry
        if pos == 0:
            if t1_long_nov[i] == 1 and t2_long_nov[i] == 1:
                pos = 1; entry_px = opens[i]; entry_bar = i; continue
            if t1_shrt_nov[i] == 1 and t2_shrt_nov[i] == 1:
                pos = -1; entry_px = opens[i]; entry_bar = i; continue
        if pos != 0:
            exit_px = 0.0; reason = -1
            # TP intrabar
            tp_lvl = entry_px + pos * tp_pips * pip
            if pos == 1 and highs[i] >= tp_lvl:
                exit_px = tp_lvl; reason = 0
            elif pos == -1 and lows[i] <= tp_lvl:
                exit_px = tp_lvl; reason = 0
            # Alignment-break exit
            if reason < 0 and exit_active == 1:
                c0 = closes[i]
                # TF1 broken aspects (in the direction of the trade)
                bt1 = 0
                t1_c1_v = t1_c1[i]; t1_c2_v = t1_c2[i]
                sm1 = t1_sm_now[i]; md1 = t1_md_now[i]; lg1 = t1_lg_now[i]
                if pos == 1:
                    mono_ok = (not np.isnan(t1_c1_v)) and (not np.isnan(t1_c2_v)) \
                               and (c0 > t1_c1_v) and (t1_c1_v > t1_c2_v)
                    stack_ok = (not np.isnan(sm1)) and (not np.isnan(md1)) and (not np.isnan(lg1)) \
                                and (c0 > sm1) and (sm1 > md1) and (md1 > lg1)
                    slope_ok = (t1_slope_up[i] == 1)
                else:
                    mono_ok = (not np.isnan(t1_c1_v)) and (not np.isnan(t1_c2_v)) \
                               and (c0 < t1_c1_v) and (t1_c1_v < t1_c2_v)
                    stack_ok = (not np.isnan(sm1)) and (not np.isnan(md1)) and (not np.isnan(lg1)) \
                                and (c0 < sm1) and (sm1 < md1) and (md1 < lg1)
                    slope_ok = (t1_slope_dn[i] == 1)
                if not mono_ok:  bt1 += 1
                if not stack_ok: bt1 += 1
                if not slope_ok: bt1 += 1
                # TF2 broken aspects
                bt2 = 0
                t2_c1_v = t2_c1[i]; t2_c2_v = t2_c2[i]
                sm2 = t2_sm_now[i]; md2 = t2_md_now[i]; lg2 = t2_lg_now[i]
                if pos == 1:
                    mono_ok2 = (not np.isnan(t2_c1_v)) and (not np.isnan(t2_c2_v)) \
                                and (c0 > t2_c1_v) and (t2_c1_v > t2_c2_v)
                    stack_ok2 = (not np.isnan(sm2)) and (not np.isnan(md2)) and (not np.isnan(lg2)) \
                                 and (c0 > sm2) and (sm2 > md2) and (md2 > lg2)
                    slope_ok2 = (t2_slope_up[i] == 1)
                else:
                    mono_ok2 = (not np.isnan(t2_c1_v)) and (not np.isnan(t2_c2_v)) \
                                and (c0 < t2_c1_v) and (t2_c1_v < t2_c2_v)
                    stack_ok2 = (not np.isnan(sm2)) and (not np.isnan(md2)) and (not np.isnan(lg2)) \
                                 and (c0 < sm2) and (sm2 < md2) and (md2 < lg2)
                    slope_ok2 = (t2_slope_dn[i] == 1)
                if not mono_ok2:  bt2 += 1
                if not stack_ok2: bt2 += 1
                if not slope_ok2: bt2 += 1
                # Trigger
                tf1_trip = (bt1 >= k_break)
                tf2_trip = (bt2 >= k_break)
                trip = (tf1_trip and tf2_trip) if both_tfs == 1 else (tf1_trip or tf2_trip)
                if trip:
                    exit_px = c0; reason = 1
            if reason >= 0:
                pnls[nt] = (exit_px - entry_px) / pip * pos
                ents[nt] = entry_bar; nt += 1
                pos = 0
    if pos != 0:
        pnls[nt] = (closes[-1] - entry_px) / pip * pos
        ents[nt] = entry_bar; nt += 1
    return pnls[:nt], ents[:nt]


def slopes_rising(sma_arr, lb=2):
    """Per-TF-bar boolean: at TF bar i, is sma_arr rising over last lb steps?
    Returns int8 array same length as sma_arr."""
    n = len(sma_arr); out = np.zeros(n, np.int8)
    for i in range(lb + 1, n):
        ok = True
        for k in range(lb):
            a, b = sma_arr[i-k], sma_arr[i-k-1]
            if np.isnan(a) or np.isnan(b) or not (a > b):
                ok = False; break
        out[i] = 1 if ok else 0
    return out


def slopes_falling(sma_arr, lb=2):
    n = len(sma_arr); out = np.zeros(n, np.int8)
    for i in range(lb + 1, n):
        ok = True
        for k in range(lb):
            a, b = sma_arr[i-k], sma_arr[i-k-1]
            if np.isnan(a) or np.isnan(b) or not (a < b):
                ok = False; break
        out[i] = 1 if ok else 0
    return out


def project_by_index(base_idx_into_upper, upper_arr):
    out_typed = np.full(len(base_idx_into_upper), np.nan, np.float64)
    valid = base_idx_into_upper >= 0
    idx_clipped = np.clip(base_idx_into_upper, 0, len(upper_arr) - 1)
    vals = upper_arr[idx_clipped].astype(np.float64)
    out_typed[valid] = vals[valid]
    return out_typed


def project_int8_by_index(base_idx_into_upper, upper_arr):
    out = np.zeros(len(base_idx_into_upper), np.int8)
    valid = base_idx_into_upper >= 0
    idx_clipped = np.clip(base_idx_into_upper, 0, len(upper_arr) - 1)
    out[valid] = upper_arr[idx_clipped][valid].astype(np.int8)
    return out


def run_one(pair, label, tf1_min, tf2_min, base_min_per_bar=5/60):
    path = S5_DIR / f"{pair}_S5_BA.parquet"
    if not path.exists():
        return []
    df = fast_full_read(path, columns=('timestamp','open','high','low','close'),
                         max_rows=None)
    opens  = df['open'].to_numpy(np.float64)
    highs  = df['high'].to_numpy(np.float64)
    lows   = df['low'].to_numpy(np.float64)
    closes = df['close'].to_numpy(np.float64)
    n_base = len(df)
    is_end = int(n_base * IS_FRAC)
    days = n_base * base_min_per_bar / 1440
    print(f"  {pair} n={n_base:,} ({days:.0f}d)", flush=True)

    n_base_per_min = 12.0   # S5 → 12 bars per minute
    n1 = int(round(tf1_min * n_base_per_min))
    n2 = int(round(tf2_min * n_base_per_min))
    r1 = bin_resample(opens, highs, lows, closes, n1)
    r2 = bin_resample(opens, highs, lows, closes, n2)
    if r1 is None or r2 is None:
        return []
    t1_o, t1_h, t1_l, t1_c = r1
    t2_o, t2_h, t2_l, t2_c = r2
    base_to_t1 = np.arange(n_base) // n1 - 1
    base_to_t2 = np.arange(n_base) // n2 - 1

    pip, sp_proxy = PAIRS[pair]
    sp_cost = sp_proxy * SPREAD_FRAC

    # Pre-compute t1_c lagged (c1 = TF1 close 1 bar back, c2 = 2 bars back)
    # At base bar i, "most recent completed TF1 bar" = (i//n1 - 1). c1 = that bar's close.
    # c2 = (i//n1 - 2)'s close. We need projections of both.
    rows = []
    for (n_sm, n_md, n_lg) in SMA_COMBOS:
        t1_sm = sma(t1_c, n_sm); t1_md = sma(t1_c, n_md); t1_lg = sma(t1_c, n_lg)
        t2_sm = sma(t2_c, n_sm); t2_md = sma(t2_c, n_md); t2_lg = sma(t2_c, n_lg)
        t1_long = tf_signal(t1_c, t1_sm, t1_md, t1_lg, 1)
        t1_shrt = tf_signal(t1_c, t1_sm, t1_md, t1_lg, 0)
        t2_long = tf_signal(t2_c, t2_sm, t2_md, t2_lg, 1)
        t2_shrt = tf_signal(t2_c, t2_sm, t2_md, t2_lg, 0)
        t1_long_nov = novelty(t1_long); t1_shrt_nov = novelty(t1_shrt)
        t2_long_nov = novelty(t2_long); t2_shrt_nov = novelty(t2_shrt)

        # All-three-SMAs-rising / all-three-falling booleans per TF bar
        t1_sup = (slopes_rising(t1_sm) & slopes_rising(t1_md) & slopes_rising(t1_lg)).astype(np.int8)
        t1_sdn = (slopes_falling(t1_sm) & slopes_falling(t1_md) & slopes_falling(t1_lg)).astype(np.int8)
        t2_sup = (slopes_rising(t2_sm) & slopes_rising(t2_md) & slopes_rising(t2_lg)).astype(np.int8)
        t2_sdn = (slopes_falling(t2_sm) & slopes_falling(t2_md) & slopes_falling(t2_lg)).astype(np.int8)

        # Lagged TF closes
        t1_c1 = np.empty_like(t1_c); t1_c1[:] = np.nan
        if len(t1_c) >= 1: t1_c1[1:] = t1_c[:-1]   # t1_c[i-1]
        t1_c2 = np.empty_like(t1_c); t1_c2[:] = np.nan
        if len(t1_c) >= 2: t1_c2[2:] = t1_c[:-2]   # t1_c[i-2]
        t2_c1 = np.empty_like(t2_c); t2_c1[:] = np.nan
        if len(t2_c) >= 1: t2_c1[1:] = t2_c[:-1]
        t2_c2 = np.empty_like(t2_c); t2_c2[:] = np.nan
        if len(t2_c) >= 2: t2_c2[2:] = t2_c[:-2]

        # Project everything to base timeline
        t1_long_nov_b = project_int8_by_index(base_to_t1, t1_long_nov)
        t1_shrt_nov_b = project_int8_by_index(base_to_t1, t1_shrt_nov)
        t2_long_nov_b = project_int8_by_index(base_to_t2, t2_long_nov)
        t2_shrt_nov_b = project_int8_by_index(base_to_t2, t2_shrt_nov)
        t1_sm_b = project_by_index(base_to_t1, t1_sm)
        t1_md_b = project_by_index(base_to_t1, t1_md)
        t1_lg_b = project_by_index(base_to_t1, t1_lg)
        t1_c1_b = project_by_index(base_to_t1, t1_c1)
        t1_c2_b = project_by_index(base_to_t1, t1_c2)
        t1_sup_b = project_int8_by_index(base_to_t1, t1_sup)
        t1_sdn_b = project_int8_by_index(base_to_t1, t1_sdn)
        t2_sm_b = project_by_index(base_to_t2, t2_sm)
        t2_md_b = project_by_index(base_to_t2, t2_md)
        t2_lg_b = project_by_index(base_to_t2, t2_lg)
        t2_c1_b = project_by_index(base_to_t2, t2_c1)
        t2_c2_b = project_by_index(base_to_t2, t2_c2)
        t2_sup_b = project_int8_by_index(base_to_t2, t2_sup)
        t2_sdn_b = project_int8_by_index(base_to_t2, t2_sdn)

        for k_break, tf_mode in EXIT_MODES:
            both_tfs = 1 if tf_mode == "both" else 0
            exit_active = 0 if tf_mode == "none" else 1
            for tp_p in TP_GRID:
                p, e = kernel_strict_break(
                    opens, highs, lows, closes,
                    t1_long_nov_b, t1_shrt_nov_b,
                    t1_sm_b, t1_md_b, t1_lg_b, t1_c1_b, t1_c2_b,
                    t1_sup_b, t1_sdn_b,
                    t2_long_nov_b, t2_shrt_nov_b,
                    t2_sm_b, t2_md_b, t2_lg_b, t2_c1_b, t2_c2_b,
                    t2_sup_b, t2_sdn_b,
                    pip, tp_p, k_break, both_tfs, exit_active,
                )
                if len(p) == 0:
                    rows.append({'tf_label':label,'pair':pair,
                                 'sma':f"{n_sm}/{n_md}/{n_lg}",
                                 'k_break':k_break,'tf_mode':tf_mode,'tp_pips':tp_p,
                                 'trades':0,'is_n':0,'oos_n':0,'is_pd':0,'oos_pd':0,
                                 'oos_dd':0,'oos_wr':0,'is_net':0,'oos_net':0,
                                 'days':round(days,1)})
                    continue
                net = p - sp_cost
                is_mask = e < is_end; oos_mask = ~is_mask
                is_days = (is_end / n_base) * days; oos_days = days - is_days
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
                             'k_break':k_break,'tf_mode':tf_mode,'tp_pips':tp_p,
                             'trades':int(len(p)),
                             'is_n':int(is_mask.sum()),'oos_n':int(oos_mask.sum()),
                             'is_net':round(is_net,1),'oos_net':round(oos_net,1),
                             'is_pd':round(is_net/max(is_days,1),2),
                             'oos_pd':round(oos_net/max(oos_days,1),2),
                             'oos_dd':round(oos_dd,1),'oos_wr':round(oos_wr,1),
                             'days':round(days,1)})
    return rows


def main():
    print(f"H17e — corrected 3-axis alignment-break exit, full sweep")
    print(f"  Exit modes: {EXIT_MODES}")
    print(f"  TP: {TP_GRID}   SMA combos: {len(SMA_COMBOS)}   TF combos: {len(TF_COMBOS)}")
    # JIT warmup
    _c = np.zeros(200); _s = np.zeros(200, np.int8)
    kernel_strict_break(_c, _c, _c, _c, _s, _s, _c, _c, _c, _c, _c, _s, _s,
                         _s, _s, _c, _c, _c, _c, _c, _s, _s,
                         0.0001, 20.0, 3, 1, 1)

    all_rows = []; t0 = time.time()
    for label, tf1_min, tf2_min in TF_COMBOS:
        print(f"\n=== {label} ===", flush=True)
        t_tf = time.time()
        for pair in S5_PAIRS:
            rows = run_one(pair, label, tf1_min, tf2_min)
            all_rows.extend(rows)
            gc.collect()
        df = pd.DataFrame([r for r in all_rows if r['tf_label']==label])
        if len(df):
            c = df[(df.is_net>0)&(df.oos_net>0)]
            bp = c.sort_values(['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(1)
            print(f"  {label}: {time.time()-t_tf:.1f}s  "
                  f"{len(bp)}/{df['pair'].nunique()} IS+OOS+  "
                  f"ΣOOS={bp['oos_pd'].sum() if len(bp) else 0:+.2f}", flush=True)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT/'h17e_strict_break_exit.csv', index=False)
    print(f"\nTotal: {time.time()-t0:.1f}s  rows: {len(df_all)}", flush=True)

    # Group by (k_break, tf_mode) — does the user's K=3-both work anywhere?
    print()
    print(f"=== By exit mode: IS+OOS+ count and Σ OOS p/d (best per pair) ===")
    for (k_break, tf_mode) in EXIT_MODES:
        sub = df_all[(df_all.k_break==k_break)&(df_all.tf_mode==tf_mode)]
        c = sub[(sub.is_net>0)&(sub.oos_net>0)]
        bp = c.sort_values(['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(1)
        n_pairs = sub['pair'].nunique()
        label = f"K={k_break} {tf_mode}"
        print(f"  {label:<15}  pairs IS+OOS+: {len(bp)}/{n_pairs}   ΣOOS={bp['oos_pd'].sum() if len(bp) else 0:+.2f}")

    # User's exact spec: K=3 both
    user_spec = df_all[(df_all.k_break==3)&(df_all.tf_mode=='both')]
    print()
    print(f"=== USER'S EXACT SPEC (K=3 strict, both TFs): top results per pair ===")
    c = user_spec[(user_spec.is_net>0)&(user_spec.oos_net>0)]
    if len(c) == 0:
        print("  no IS+OOS+ configs.")
    else:
        bp = c.sort_values(['pair','oos_pd'], ascending=[True,False]).groupby('pair').head(3)
        for _, r in bp.iterrows():
            print(f"  {r['pair']:<9}{r['tf_label']:<13}{r['sma']:<10}"
                  f" TP={int(r['tp_pips']):>2}p  "
                  f"OOSpd={r['oos_pd']:+6.2f}  DD={r['oos_dd']:+7.0f}  "
                  f"N={int(r['oos_n']):>4}  WR={r['oos_wr']:.0f}%")


if __name__ == '__main__':
    main()
