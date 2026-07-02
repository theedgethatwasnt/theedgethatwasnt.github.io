"""H17g — Risk-bounding SL sweep on the 6 SMA-Stack PORTFOLIO REPS.

The portfolio-paper service currently runs the 6 reps from
research/portfolio/portfolio.csv TP-only (no stop). The user requires every
paper strategy to have a definable capped downside. This sweep answers, per
rep: can a broker-side SL be added that bounds the tail WITHOUT destroying the
positive OOS edge?

REUSE (do NOT reimplement the entry):
  - h17f_catastrophe_sl.kernel_sl       — 3-TF (TF1/TF2) novelty entry + TP + SL
  - h17f_catastrophe_sl.run_one         — full-history loader / resample / SL sweep
    (but h17f.run_one sweeps a fixed SMA/TP grid; here we pin each rep's exact
     sma + tp, so we call the lower-level kernel_sl directly via run_rep_3tf)
  - h17d_full_history.fast_full_read / bin_resample / project_via_index
  - h17f.atr_bin_resampled / project_by_index / project_int8_by_index
  - h17_stack_alignment.tf_signal / novelty  (the H17 stack-alignment entry)

For the two 4-TF reps the h17f 2-TF kernel cannot represent the entry
(it needs alignment on THREE upper TFs). We add kernel_sl_4tf — identical SL
fill model as h17f.kernel_sl, but with p3_4tf_stacks' 3-upper-TF entry.

The reps are all M_exit=0 (TP-only) entries, so the H17 novelty entry + TP is
exactly the portfolio-paper signal. We only add the SL leg.

SL modes (same as h17f): none, fixed {30,50,80,100,150,200}p, ATR {1,1.5,2,3}×ATR_TF1.
6 reps × 11 SL modes = 66 backtests.

SOP: closed bars (R1), mid OHLC for signal + explicit spread cost (R3),
IS-only window for the IS/OOS split inherited from h17f/h17d (IS_FRAC=4/6).
Spread cost = PAIRS[pair] proxy × SPREAD_FRAC, identical to the sweep that
produced the portfolio.csv numbers. The SL is broker-side, placed at entry,
filled intra-bar at the SL level (h17f's pessimistic TP-before-SL on
favourable bars, SL-before-TP on adverse bars).
"""
import sys, time, gc
from pathlib import Path
import numpy as np, pandas as pd
import numba as nb

sys.path.insert(0, str(Path(__file__).parent))
from h17_stack_alignment import tf_signal, novelty
from h17d_full_history import fast_full_read, bin_resample
from h17f_catastrophe_sl import (
    kernel_sl, atr_bin_resampled, project_by_index, project_int8_by_index,
)
from _lib import PAIRS, IS_FRAC, sma, SPREAD_FRAC

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
PROJECT = Path("/path/to/projects/fx-core")
S5_DIR = PROJECT / "data/s5_ohlc"

# SL modes — identical to h17f.
#   (mode_id, param): 0=none, 1=fixed pips, 2=ATR×k on ATR_TF1 at entry
SL_MODES = [
    (0,   0.0),
    (1,  30.0),  (1,  50.0),  (1,  80.0),
    (1, 100.0),  (1, 150.0),  (1, 200.0),
    (2,   1.0),  (2,   1.5),  (2,   2.0),  (2,   3.0),
]

# The 6 portfolio reps (from research/portfolio/portfolio.csv).
#   id, tf_label, pair, sma(sm,md,lg), tp_pips, n_tf, (tf1_min, tf2_min[, tf3_min])
# S5 base, n_base_per_min = 12, base_min_per_bar = 5/60.
REPS = [
    dict(id="30bb3bff76bf", tf="S5/M10/M30",    pair="USD_JPY", sma=(5,10,22), tp=15.0, ntf=3, mins=(10, 30)),
    dict(id="a237cb171d85", tf="S5/S30/M5",     pair="USD_JPY", sma=(7,22,50), tp=20.0, ntf=3, mins=(0.5, 5)),
    dict(id="ba2988590380", tf="S5/M1/M15",     pair="EUR_USD", sma=(7,15,35), tp=20.0, ntf=3, mins=(1, 15)),
    dict(id="f5ea18b318f2", tf="S5/S30/M5",     pair="GBP_USD", sma=(5,15,35), tp=15.0, ntf=3, mins=(0.5, 5)),
    dict(id="268d48dadd94", tf="S5/S30/M5/H1",  pair="EUR_JPY", sma=(7,10,50), tp=15.0, ntf=4, mins=(0.5, 5, 60)),
    dict(id="9eda87cef382", tf="S5/M1/M5/H1",   pair="EUR_JPY", sma=(7,10,35), tp=15.0, ntf=4, mins=(1, 5, 60)),
]

N_BASE_PER_MIN = 12.0
BASE_MIN_PER_BAR = 5/60


@nb.njit(cache=True)
def kernel_sl_4tf(opens, highs, lows, closes,
                  t1_long_nov, t1_shrt_nov,
                  t2_long_nov, t2_shrt_nov,
                  t3_long_nov, t3_shrt_nov,
                  t1_atr, pip, tp_pips, sl_mode, sl_param):
    """4-TF stack-alignment novelty entry (all 3 upper TFs) + broker-side TP + SL.

    Entry = p3_4tf_stacks.kernel_4tf entry (alignment newly formed on all 3).
    Exit fill model = h17f.kernel_sl (TP priority on favourable bar, SL priority
    on adverse bar; pessimistic when both within the bar). M_exit=0 (the reps
    are all TP-only), so no stack-degrade exit — only TP + SL.
    sl_mode: 0=none, 1=fixed_pips, 2=atr_scaled (sl_param=k_atr on ATR_TF1)."""
    n = len(opens)
    pos = 0; entry_px = 0.0; entry_bar = -1
    sl_price = 0.0
    pnls = np.empty(n, np.float64); ents = np.empty(n, np.int64)
    reasons = np.empty(n, np.int8); nt = 0
    for i in range(1, n):
        if pos == 0:
            new_dir = 0
            if (t1_long_nov[i] == 1 and t2_long_nov[i] == 1 and t3_long_nov[i] == 1):
                new_dir = 1
            elif (t1_shrt_nov[i] == 1 and t2_shrt_nov[i] == 1 and t3_shrt_nov[i] == 1):
                new_dir = -1
            if new_dir != 0:
                pos = new_dir
                entry_px = opens[i]
                entry_bar = i
                if sl_mode == 0:
                    sl_price = 0.0
                elif sl_mode == 1:
                    sl_price = entry_px - pos * sl_param * pip
                else:
                    a = t1_atr[i]
                    if np.isnan(a) or a <= 0:
                        sl_price = entry_px - pos * 100.0 * pip
                    else:
                        sl_price = entry_px - pos * sl_param * a
                continue
        if pos != 0:
            exit_px = 0.0; reason = -1
            tp_lvl = entry_px + pos * tp_pips * pip
            if pos == 1:
                bull = closes[i] >= opens[i]
                if bull:
                    if highs[i] >= tp_lvl:
                        exit_px = tp_lvl; reason = 0
                    elif sl_mode > 0 and lows[i] <= sl_price:
                        exit_px = sl_price; reason = 1
                else:
                    if sl_mode > 0 and lows[i] <= sl_price:
                        exit_px = sl_price; reason = 1
                    elif highs[i] >= tp_lvl:
                        exit_px = tp_lvl; reason = 0
            else:
                bear = closes[i] < opens[i]
                if bear:
                    if lows[i] <= tp_lvl:
                        exit_px = tp_lvl; reason = 0
                    elif sl_mode > 0 and highs[i] >= sl_price:
                        exit_px = sl_price; reason = 1
                else:
                    if sl_mode > 0 and highs[i] >= sl_price:
                        exit_px = sl_price; reason = 1
                    elif lows[i] <= tp_lvl:
                        exit_px = tp_lvl; reason = 0
            if reason >= 0:
                pnls[nt] = (exit_px - entry_px) / pip * pos
                ents[nt] = entry_bar; reasons[nt] = reason
                nt += 1
                pos = 0
    if pos != 0:
        pnls[nt] = (closes[-1] - entry_px) / pip * pos
        ents[nt] = entry_bar; reasons[nt] = 2
        nt += 1
    return pnls[:nt], ents[:nt], reasons[:nt]


def _stats(p, e, r, sp_cost, is_end, n_base, days):
    """Identical stat computation to h17f.run_one (IS/OOS split, OOS DD/WR)."""
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
    return dict(
        trades=int(len(p)),
        is_n=int(is_mask.sum()), oos_n=int(oos_mask.sum()),
        is_net=round(is_net,1), oos_net=round(oos_net,1),
        is_pd=round(is_net/max(is_days,1),2),
        oos_pd=round(oos_net/max(oos_days,1),2),
        oos_dd=round(oos_dd,1), oos_wr=round(oos_wr,1),
        r_tp=int((r==0).sum()), r_sl=int((r==1).sum()),
    )


def load_base(pair):
    path = S5_DIR / f"{pair}_S5_BA.parquet"
    df = fast_full_read(path, columns=('timestamp','open','high','low','close'),
                        max_rows=None)
    opens  = df['open'].to_numpy(np.float64)
    highs  = df['high'].to_numpy(np.float64)
    lows   = df['low'].to_numpy(np.float64)
    closes = df['close'].to_numpy(np.float64)
    n_base = len(df)
    return opens, highs, lows, closes, n_base


def upper_nov(closes_base, opens, highs, lows, n_per_bar, sma_combo, n_base):
    """Build base-projected long/short novelty for one upper TF + its ATR-ready
    arrays. Mirrors h17d/p3 projection (i//n - 1 = last completed upper bar)."""
    r = bin_resample(opens, highs, lows, closes_base, n_per_bar)
    if r is None:
        return None
    _, _, _, t_c = r
    n_sm, n_md, n_lg = sma_combo
    t_sm = sma(t_c, n_sm); t_md = sma(t_c, n_md); t_lg = sma(t_c, n_lg)
    t_long = tf_signal(t_c, t_sm, t_md, t_lg, 1)
    t_shrt = tf_signal(t_c, t_sm, t_md, t_lg, 0)
    t_lnov = novelty(t_long); t_snov = novelty(t_shrt)
    base_to_t = np.arange(n_base) // n_per_bar - 1
    lnov_b = project_int8_by_index(base_to_t, t_lnov)
    snov_b = project_int8_by_index(base_to_t, t_snov)
    return lnov_b, snov_b


def run_rep(rep):
    pair = rep["pair"]
    opens, highs, lows, closes, n_base = load_base(pair)
    is_end = int(n_base * IS_FRAC)
    days = n_base * BASE_MIN_PER_BAR / 1440
    pip, sp_proxy = PAIRS[pair]
    sp_cost = sp_proxy * SPREAD_FRAC
    print(f"  [{rep['id']}] {pair} {rep['tf']} sma={rep['sma']} tp={rep['tp']} "
          f"n={n_base:,} ({days:.0f}d)", flush=True)

    # TF1 bin size (for ATR-scaled SL — ATR on TF1, same as h17f)
    n1 = int(round(rep["mins"][0] * N_BASE_PER_MIN))
    t1_atr_series, _ = atr_bin_resampled(opens, highs, lows, closes, n1, period=14)
    base_to_t1 = np.arange(n_base) // n1 - 1
    if t1_atr_series is None:
        t1_atr_b = np.full(n_base, np.nan)
    else:
        t1_atr_b = project_by_index(base_to_t1, t1_atr_series)

    # Build per-TF novelty arrays
    novs = []
    for m in rep["mins"]:
        npb = int(round(m * N_BASE_PER_MIN))
        u = upper_nov(closes, opens, highs, lows, npb, rep["sma"], n_base)
        if u is None:
            print(f"    [skip] {rep['id']} — resample failed for {m}min")
            return []
        novs.append(u)

    rows = []
    for (sl_mode, sl_param) in SL_MODES:
        if rep["ntf"] == 3:
            (t1_l, t1_s), (t2_l, t2_s) = novs
            p, e, r = kernel_sl(
                opens, highs, lows, closes,
                t1_l, t1_s, t2_l, t2_s,
                t1_atr_b, pip, rep["tp"], sl_mode, sl_param,
            )
        else:  # 4-TF
            (t1_l, t1_s), (t2_l, t2_s), (t3_l, t3_s) = novs
            p, e, r = kernel_sl_4tf(
                opens, highs, lows, closes,
                t1_l, t1_s, t2_l, t2_s, t3_l, t3_s,
                t1_atr_b, pip, rep["tp"], sl_mode, sl_param,
            )
        mode_label = ("none" if sl_mode == 0 else
                      (f"sl_{int(sl_param)}p" if sl_mode == 1 else f"sl_{sl_param}xATR"))
        base = dict(rep_id=rep["id"], tf_label=rep["tf"], pair=pair,
                    sma=f"{rep['sma'][0]}/{rep['sma'][1]}/{rep['sma'][2]}",
                    tp_pips=rep["tp"], sl_mode=mode_label, sl_param=sl_param,
                    days=round(days,1))
        if len(p) == 0:
            base.update(dict(trades=0, is_n=0, oos_n=0, is_net=0, oos_net=0,
                             is_pd=0, oos_pd=0, oos_dd=0, oos_wr=0, r_tp=0, r_sl=0,
                             r_sl_frac=0.0))
            rows.append(base); continue
        st = _stats(p, e, r, sp_cost, is_end, n_base, days)
        st["r_sl_frac"] = round(st["r_sl"] / max(len(p), 1), 4)
        base.update(st)
        rows.append(base)
    del opens, highs, lows, closes; gc.collect()
    return rows


def _sl_distance_pips(sl_mode, sl_param, pair):
    """The actual per-trade max-loss cap implied by an SL mode, in pips.
    For fixed: the pip count. For ATR: not a fixed pip distance, so return inf
    (ATR stops do not give a constant definable per-trade cap — they vary by
    entry-time volatility; we rank fixed stops ahead of ATR for 'definable cap')."""
    if sl_mode.endswith("p"):
        return float(sl_mode.split("_")[1].rstrip("p"))
    return float("inf")  # ATR-scaled: no constant pip cap


def verdict_table(df):
    """Per-rep: TP-only baseline, then the TIGHTEST DEFINABLE per-trade cap
    (smallest fixed-pip SL distance) that keeps oos_pd >= 80% of the TP-only
    baseline AND stays positive. This is what actually bounds the downside —
    a too-wide stop (e.g. 200p that never fires) is not a real bound even if
    its realized oos_dd is small. KEEP or CULL."""
    EDGE_KEEP_FRAC = 0.80   # SL oos_pd must be >= 80% of TP-only oos_pd
    lines = []
    lines.append("| rep | pair | tf | TP-only oos_pd | TP-only oos_dd | best-SL | SL oos_pd | SL oos_dd | edge % | r_sl | verdict |")
    lines.append("|-----|------|----|---------------:|---------------:|---------|----------:|----------:|-------:|-----:|---------|")
    summary = []
    for rep in REPS:
        sub = df[df.rep_id == rep["id"]]
        base = sub[sub.sl_mode == "none"].iloc[0]
        base_pd = base["oos_pd"]; base_dd = base["oos_dd"]
        cands = sub[sub.sl_mode != "none"].copy()
        # Keep SL configs that preserve the edge (and stay positive)
        keep = cands[(cands.oos_pd >= EDGE_KEEP_FRAC * base_pd) & (cands.oos_pd > 0)].copy()
        if len(keep) == 0:
            best = None
        else:
            # Tightest DEFINABLE per-trade cap = smallest fixed-pip stop distance.
            # (ATR stops rank last — no constant pip cap.)
            keep["cap_pips"] = keep.apply(
                lambda r: _sl_distance_pips(r["sl_mode"], r["sl_param"], r["pair"]), axis=1)
            best = keep.sort_values(["cap_pips", "oos_dd"], ascending=[True, False]).iloc[0]
        if best is None:
            lines.append(f"| {rep['id']} | {rep['pair']} | {rep['tf']} | "
                         f"{base_pd:+.2f} | {base_dd:+.1f} | — | — | — | — | — | "
                         f"**CULL** (no SL keeps edge) |")
            summary.append((rep, base_pd, base_dd, None))
        else:
            edge_pct = 100 * best["oos_pd"] / base_pd if base_pd != 0 else 0
            lines.append(f"| {rep['id']} | {rep['pair']} | {rep['tf']} | "
                         f"{base_pd:+.2f} | {base_dd:+.1f} | {best['sl_mode']} | "
                         f"{best['oos_pd']:+.2f} | {best['oos_dd']:+.1f} | {edge_pct:.0f}% | "
                         f"{best['r_sl_frac']:.2f} | **KEEP** (SL={best['sl_mode']}) |")
            summary.append((rep, base_pd, base_dd, best))
    return "\n".join(lines), summary


def main():
    print("="*100)
    print("  H17g — risk-bounding SL sweep on the 6 SMA-Stack PORTFOLIO REPS")
    print(f"  SL modes: {[m for m in SL_MODES]}")
    print(f"  Reps: {[r['id'] for r in REPS]}")
    print("="*100, flush=True)

    # JIT warmup (both kernels)
    _c = np.zeros(200); _s = np.zeros(200, np.int8)
    kernel_sl(_c,_c,_c,_c,_s,_s,_s,_s,_c, 0.0001, 15.0, 0, 0.0)
    kernel_sl(_c,_c,_c,_c,_s,_s,_s,_s,_c, 0.0001, 15.0, 1, 50.0)
    kernel_sl(_c,_c,_c,_c,_s,_s,_s,_s,_c, 0.0001, 15.0, 2, 2.0)
    kernel_sl_4tf(_c,_c,_c,_c,_s,_s,_s,_s,_s,_s,_c, 0.0001, 15.0, 0, 0.0)
    kernel_sl_4tf(_c,_c,_c,_c,_s,_s,_s,_s,_s,_s,_c, 0.0001, 15.0, 1, 50.0)
    kernel_sl_4tf(_c,_c,_c,_c,_s,_s,_s,_s,_s,_s,_c, 0.0001, 15.0, 2, 2.0)

    all_rows = []; t0 = time.time()
    for rep in REPS:
        all_rows.extend(run_rep(rep))

    df = pd.DataFrame(all_rows)
    out_csv = OUT / "h17g_portfolio_rep_sl.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Total: {time.time()-t0:.1f}s   rows: {len(df)}   → {out_csv}", flush=True)

    # Per-rep detail
    print("\n" + "="*100)
    print("  PER-REP SL SWEEP DETAIL")
    print("="*100)
    for rep in REPS:
        sub = df[df.rep_id == rep["id"]].sort_values(
            ["sl_mode"], key=lambda s: s.map(_sl_sort_key))
        base = sub[sub.sl_mode == "none"].iloc[0]
        print(f"\n  [{rep['id']}] {rep['pair']} {rep['tf']} sma={sub.iloc[0]['sma']} "
              f"tp={int(rep['tp'])}p   "
              f"(TP-only baseline: oos_pd={base['oos_pd']:+.2f} oos_dd={base['oos_dd']:+.1f} "
              f"n={int(base['oos_n'])})")
        print(f"    {'SL mode':<13} {'oos_pd':>8} {'oos_dd':>9} {'oos_wr':>7} {'oos_n':>6} {'r_sl':>6}")
        for _, r in sub.iterrows():
            print(f"    {r['sl_mode']:<13} {r['oos_pd']:>+8.2f} {r['oos_dd']:>+9.1f} "
                  f"{r['oos_wr']:>6.1f}% {int(r['oos_n']):>6} {r['r_sl_frac']:>6.2f}")

    # Verdict table
    table, summary = verdict_table(df)
    print("\n" + "="*100)
    print("  PER-REP VERDICT  (best SL = tightest |oos_dd| that keeps oos_pd >= 80% of TP-only)")
    print("="*100)
    print(table)

    keep = [s for s in summary if s[3] is not None]
    cull = [s for s in summary if s[3] is None]
    print(f"\n  SURVIVE with bounding SL: {len(keep)}/6")
    print(f"  CULL (edge needs unbounded risk): {len(cull)}/6")
    if cull:
        print("    " + ", ".join(f"{s[0]['id']}({s[0]['pair']})" for s in cull))

    # Is there a single SL level across survivors?
    if keep:
        modes = {}
        for rep, bpd, bdd, best in keep:
            modes.setdefault(best["sl_mode"], []).append(rep["id"])
        print("\n  SL-mode consensus among survivors:")
        for m, ids in sorted(modes.items(), key=lambda kv: -len(kv[1])):
            print(f"    {m:<13} {len(ids)} rep(s): {', '.join(ids)}")

    with open(OUT / "h17g_verdict.md", "w") as f:
        f.write("# H17g — Portfolio rep risk-bounding SL verdict\n\n")
        f.write(table + "\n\n")
        f.write(f"Survive with bounding SL: {len(keep)}/6\n")


def _sl_sort_key(label):
    if label == "none": return -1
    if label.endswith("p"): return float(label.split("_")[1].rstrip("p"))
    if "xATR" in label: return 1000 + float(label.split("_")[1].rstrip("xATR"))
    return 9999


if __name__ == "__main__":
    main()
