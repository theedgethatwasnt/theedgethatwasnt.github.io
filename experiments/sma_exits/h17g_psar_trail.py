"""H17g — PSAR(TF1) trail-exit sweep on H17 stack-alignment K=0 entries.

Same H17 entry rule (stack-alignment + novelty on both TFs), but the
exit is a PSAR trail computed on TF1 bars (bin-resampled from S5).
Activates after `activate_pips` of MFE so it doesn't kick in during
the noisy first few bars of a trade.

This is the H13 template ported to H17 entries. Same +TP behaviour
(broker-side TP runs alongside PSAR trail).
"""
import sys, time, gc
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from h17_stack_alignment import tf_signal, novelty
from h17d_full_history import fast_full_read, bin_resample
from h13_psar import psar_h1   # reuse PSAR kernel
from _lib import PAIRS, IS_FRAC, sma, SPREAD_FRAC

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
PROJECT = Path("/path/to/projects/fx-core")

SMA_COMBOS = [
    (5, 10, 22), (5, 15, 35), (5, 22, 50),
    (7, 15, 35), (7, 22, 50),
    (10, 22, 50),
]
TP_GRID = [15.0, 20.0, 30.0]
AF_STARTS = [0.005, 0.010, 0.020]
AF_MAX    = 0.10
ACTIVATE  = [0.0, 10.0, 20.0]

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
def kernel_psar(opens, highs, lows, closes,
                 t1_long_nov, t1_shrt_nov, t2_long_nov, t2_shrt_nov,
                 t1_psar,        # per-base-bar projection of TF1 PSAR price
                 pip, tp_pips, activate_pips, with_tp):
    n = len(opens)
    pos = 0; entry_px = 0.0; entry_bar = -1; mfe_pips = 0.0; armed = 0
    pnls = np.empty(n, np.float64); ents = np.empty(n, np.int64); nt = 0
    for i in range(1, n):
        if pos == 0:
            new_dir = 0
            if t1_long_nov[i] == 1 and t2_long_nov[i] == 1: new_dir = 1
            elif t1_shrt_nov[i] == 1 and t2_shrt_nov[i] == 1: new_dir = -1
            if new_dir != 0:
                pos = new_dir; entry_px = opens[i]; entry_bar = i
                mfe_pips = 0.0; armed = 0
                continue
        if pos != 0:
            exit_px = 0.0; reason = -1
            cur_pips = (closes[i] - entry_px) / pip * pos
            if cur_pips > mfe_pips:
                mfe_pips = cur_pips
            if mfe_pips >= activate_pips:
                armed = 1
            if with_tp == 1:
                tp_lvl = entry_px + pos * tp_pips * pip
                if pos == 1 and highs[i] >= tp_lvl:
                    exit_px = tp_lvl; reason = 0
                elif pos == -1 and lows[i] <= tp_lvl:
                    exit_px = tp_lvl; reason = 0
            if reason < 0 and armed == 1:
                p = t1_psar[i-1]
                if not np.isnan(p):
                    if pos == 1 and closes[i] < p:
                        exit_px = closes[i]; reason = 1
                    elif pos == -1 and closes[i] > p:
                        exit_px = closes[i]; reason = 1
            if reason >= 0:
                pnls[nt] = (exit_px - entry_px) / pip * pos
                ents[nt] = entry_bar; nt += 1
                pos = 0
    if pos != 0:
        pnls[nt] = (closes[-1] - entry_px) / pip * pos
        ents[nt] = entry_bar; nt += 1
    return pnls[:nt], ents[:nt]


def project_by_index(base_idx_into_upper, upper_arr):
    out = np.full(len(base_idx_into_upper), np.nan, np.float64)
    valid = base_idx_into_upper >= 0
    idx_clipped = np.clip(base_idx_into_upper, 0, len(upper_arr) - 1)
    vals = upper_arr[idx_clipped].astype(np.float64)
    out[valid] = vals[valid]
    return out


def project_int8_by_index(base_idx_into_upper, upper_arr):
    out = np.zeros(len(base_idx_into_upper), np.int8)
    valid = base_idx_into_upper >= 0
    idx_clipped = np.clip(base_idx_into_upper, 0, len(upper_arr) - 1)
    out[valid] = upper_arr[idx_clipped][valid].astype(np.int8)
    return out


def run_one(pair, label, tf1_min, tf2_min, base_min_per_bar=5/60):
    path = S5_DIR / f"{pair}_S5_BA.parquet"
    if not path.exists(): return []
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

    n_base_per_min = 12.0
    n1 = int(round(tf1_min * n_base_per_min))
    n2 = int(round(tf2_min * n_base_per_min))
    r1 = bin_resample(opens, highs, lows, closes, n1)
    r2 = bin_resample(opens, highs, lows, closes, n2)
    if r1 is None or r2 is None: return []
    t1_o, t1_h, t1_l, t1_c = r1
    t2_o, t2_h, t2_l, t2_c = r2

    base_to_t1 = np.arange(n_base) // n1 - 1
    base_to_t2 = np.arange(n_base) // n2 - 1

    pip, sp_proxy = PAIRS[pair]
    sp_cost = sp_proxy * SPREAD_FRAC

    rows = []
    # PSAR cache by af_start
    psar_proj_cache = {}
    for af_s in AF_STARTS:
        psar_arr, _ = psar_h1(t1_h, t1_l, af_s, af_s, AF_MAX)
        psar_proj_cache[af_s] = project_by_index(base_to_t1, psar_arr)

    for (n_sm, n_md, n_lg) in SMA_COMBOS:
        t1_sm = sma(t1_c, n_sm); t1_md = sma(t1_c, n_md); t1_lg = sma(t1_c, n_lg)
        t2_sm = sma(t2_c, n_sm); t2_md = sma(t2_c, n_md); t2_lg = sma(t2_c, n_lg)
        t1_long = tf_signal(t1_c, t1_sm, t1_md, t1_lg, 1)
        t1_shrt = tf_signal(t1_c, t1_sm, t1_md, t1_lg, 0)
        t2_long = tf_signal(t2_c, t2_sm, t2_md, t2_lg, 1)
        t2_shrt = tf_signal(t2_c, t2_sm, t2_md, t2_lg, 0)
        t1_long_nov = novelty(t1_long); t1_shrt_nov = novelty(t1_shrt)
        t2_long_nov = novelty(t2_long); t2_shrt_nov = novelty(t2_shrt)
        t1_long_nov_b = project_int8_by_index(base_to_t1, t1_long_nov)
        t1_shrt_nov_b = project_int8_by_index(base_to_t1, t1_shrt_nov)
        t2_long_nov_b = project_int8_by_index(base_to_t2, t2_long_nov)
        t2_shrt_nov_b = project_int8_by_index(base_to_t2, t2_shrt_nov)

        for af_s in AF_STARTS:
            psar_b = psar_proj_cache[af_s]
            for act in ACTIVATE:
                for tp_p in TP_GRID:
                    for with_tp in (1, 0):
                        p, e = kernel_psar(
                            opens, highs, lows, closes,
                            t1_long_nov_b, t1_shrt_nov_b,
                            t2_long_nov_b, t2_shrt_nov_b,
                            psar_b, pip, tp_p, act, with_tp,
                        )
                        if len(p) == 0:
                            rows.append({'tf_label':label,'pair':pair,
                                         'sma':f"{n_sm}/{n_md}/{n_lg}",
                                         'af_s':af_s,'activate':act,
                                         'tp_pips':tp_p,'with_tp':bool(with_tp),
                                         'trades':0,'is_n':0,'oos_n':0,
                                         'is_pd':0,'oos_pd':0,'oos_dd':0,'oos_wr':0,
                                         'is_net':0,'oos_net':0,'days':round(days,1)})
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
                                     'af_s':af_s,'activate':act,
                                     'tp_pips':tp_p,'with_tp':bool(with_tp),
                                     'trades':int(len(p)),
                                     'is_n':int(is_mask.sum()),'oos_n':int(oos_mask.sum()),
                                     'is_net':round(is_net,1),'oos_net':round(oos_net,1),
                                     'is_pd':round(is_net/max(is_days,1),2),
                                     'oos_pd':round(oos_net/max(oos_days,1),2),
                                     'oos_dd':round(oos_dd,1),'oos_wr':round(oos_wr,1),
                                     'days':round(days,1)})
    return rows


def main():
    print(f"H17g — PSAR(TF1) trail sweep on H17 stack-alignment K=0 entries")
    print(f"  AF_start={AF_STARTS}  AF_max={AF_MAX}  activate={ACTIVATE}")
    print(f"  TP: {TP_GRID}  with_tp: [True,False]  SMA: {len(SMA_COMBOS)}  TF: {len(TF_COMBOS)}")
    _c = np.zeros(200); _s = np.zeros(200, np.int8)
    kernel_psar(_c,_c,_c,_c,_s,_s,_s,_s,_c, 0.0001, 20.0, 10.0, 1)

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
    df_all.to_csv(OUT/'h17g_psar_trail.csv', index=False)
    print(f"\nTotal: {time.time()-t0:.1f}s  rows: {len(df_all)}", flush=True)

    # Deploy candidate per pair
    cand = df_all[(df_all.is_net>0)&(df_all.oos_net>0)].sort_values(
        ['pair','oos_pd'], ascending=[True,False])
    bp = cand.groupby('pair').head(1)
    print(f"\n=== Best IS+OOS+ per pair ===")
    print(f"  Pair    TF           SMA       af_s   act  TP  TPkeep  OOSpd   OOS_DD   N   WR%")
    for _, r in bp.iterrows():
        tp = "+TP" if r['with_tp'] else "off"
        print(f"  {r['pair']:<7} {r['tf_label']:<12} {r['sma']:<9} "
              f"{r['af_s']:.3f} {int(r['activate']):>2}p  {int(r['tp_pips']):>2}p "
              f"{tp:<6} {r['oos_pd']:>+6.2f}  {r['oos_dd']:>+7.0f}  "
              f"{int(r['oos_n']):>4} {r['oos_wr']:>4.0f}%")
    print(f"\n  Σ OOS pd: {bp['oos_pd'].sum():+.2f}  on {len(bp)} pairs")


if __name__ == '__main__':
    main()
