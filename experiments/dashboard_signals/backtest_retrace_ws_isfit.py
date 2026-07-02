#!/usr/bin/env python3
"""
Honest IS-fit of WS threshold, then sealed OOS confirm.

The earlier +89 p/d for WS-only was on the same OOS slice that all the
ws_thr ∈ {0.0, 0.5, 1.0, 1.5, 2.0} were evaluated against (in
backtest_retrace_ws_filter.py). WS=0.5 was *picked because* it had the best
OOS Σ p/d — i.e., the OOS sample was used in selection. That's not OOS, it's
in-sample-by-the-time-we-pick-it.

This script does it honestly:
  1. On IS (first 70%), sweep ws_thr ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}.
     Pick the threshold with highest IS Σ p/d.
  2. Apply that single threshold to OOS (last 30%). Report once.
  3. Do this for both:
       • WS-only (no Markov)
       • Markov + WS (stacked)

Same retrace machinery as the factorial. Same 4 JPY pairs.
"""
import gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
S5_DIR  = PROJECT / "data" / "s5_ba"
M5_DIR  = PROJECT / "data" / "m5_ba"
OUT     = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

PAIRS = ["GBP_JPY", "USD_JPY", "EUR_JPY", "AUD_JPY"]
PIP   = 0.01

THR_Z         = 2.5
PEAK_BARS_S5  = 44
STOP_DIST_PIP = 3.0
TP_PIP        = 20.0
HORIZON_S5    = 600
Z_WIN         = 6
MAD_WIN       = 2048
IS_FRAC       = 0.70

MARKOV_MW       = 10
MARKOV_MT       = 0.002
MARKOV_SIG_THR  = 0.20
MARKOV_MIN_PRIME = 30
BULL, SIDE, BEAR = 0, 1, 2

WIN_LAGS_M5 = [1, 3, 12, 48, 288]
WIN_MINUTES = [5, 15, 60, 240, 1440]
WIN_WEIGHTS = [0.10, 0.15, 0.20, 0.20, 0.25]

WS_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def compute_shock_z(close, pip, w=Z_WIN, mad_win=MAD_WIN):
    n = len(close)
    vel = np.empty(n, dtype=np.float64)
    vel[:w] = 0.0
    vel[w:] = (close[w:] - close[:n-w]) / pip
    vs  = pd.Series(vel)
    rm  = vs.rolling(mad_win, min_periods=50).median()
    ad  = (vs - rm).abs()
    rmd = ad.rolling(mad_win, min_periods=50).median()
    z = ((vs - rm) / (1.4826 * rmd.clip(lower=1e-6))).fillna(0).values
    return z.astype(np.float64), vel.astype(np.float64)


def compute_m5_ws(m5_close, pip):
    n = len(m5_close)
    moms = np.full((len(WIN_LAGS_M5), n), np.nan, dtype=np.float64)
    for j, (lag, mins) in enumerate(zip(WIN_LAGS_M5, WIN_MINUTES)):
        if lag >= n: continue
        moms[j, :lag] = np.nan
        moms[j, lag:] = (m5_close[lag:] - m5_close[:-lag]) / pip / float(mins)
    weights = np.asarray(WIN_WEIGHTS, dtype=np.float64)
    valid   = ~np.isnan(moms)
    w_avail = np.where(valid, weights[:, None], 0.0).sum(axis=0)
    wv      = np.where(valid, moms * weights[:, None], 0.0).sum(axis=0)
    return np.where(w_avail > 0, wv / w_avail, np.nan)


def align_to_s5(s5_ts, m5_ts, vals):
    m5_arr = m5_ts.values.astype("datetime64[ns]")
    s5_arr = s5_ts.values.astype("datetime64[ns]")
    pos = np.searchsorted(m5_arr, s5_arr, side="right") - 1
    out = np.full(len(s5_arr), np.nan, dtype=np.float64)
    mask = pos >= 0
    out[mask] = vals[pos[mask]]
    return out


def build_markov_signals(close_s5, ts_s5):
    s5 = pd.Series(close_s5, index=ts_s5)
    d1 = s5.resample("1D").last().dropna()
    lr = np.log(d1 / d1.shift(1)).dropna()
    roll = lr.rolling(MARKOV_MW).sum()
    def _state(r):
        if np.isnan(r): return np.nan
        if r >  MARKOV_MT: return BULL
        if r < -MARKOV_MT: return BEAR
        return SIDE
    states = roll.map(_state).dropna().astype(int)
    T = np.zeros((3, 3), dtype=np.float64)
    signals = {}
    for i in range(len(states) - 1):
        day_i = states.index[i].date()
        s = int(states.iloc[i])
        row_sum = T[s].sum()
        if row_sum >= MARKOV_MIN_PRIME:
            signals[day_i] = (T[s, BULL] - T[s, BEAR]) / row_sum
        else:
            signals[day_i] = 0.0
        T[s, int(states.iloc[i + 1])] += 1.0
    return signals


def map_markov_to_s5(s5_ts, daily_signals):
    if not daily_signals:
        return np.zeros(len(s5_ts), dtype=np.float64)
    sorted_days = sorted(daily_signals.keys())
    prev = {}
    for i, d in enumerate(sorted_days):
        prev[d] = daily_signals[sorted_days[i - 1]] if i > 0 else 0.0
    out = np.zeros(len(s5_ts), dtype=np.float64)
    for i, ts in enumerate(s5_ts):
        out[i] = prev.get(ts.date(), 0.0)
    return out


@njit(cache=True)
def sim(bid, ask, close, shock_flag, vel,
        markov_sig, ws_aligned, pip,
        peak_bars, stop_pips, tp_pips, horizon,
        markov_thr, ws_thr,
        slice_start, slice_end):
    """Same as factorial sim, restricted to bar range [slice_start, slice_end)."""
    n = len(close)
    pb = int(peak_bars); hor = int(horizon)
    max_ev = n // 10
    filled  = np.zeros(max_ev, dtype=np.int8)
    tp_hit  = np.zeros(max_ev, dtype=np.int8)
    pnl_out = np.zeros(max_ev, dtype=np.float64)
    skipped_m = 0; skipped_w = 0
    ev_count = 0
    cooldown = 0

    start = max(Z_WIN, slice_start)
    end   = min(n - pb - hor - 2, slice_end)
    for t in range(start, end):
        if cooldown > 0:
            cooldown -= 1
            continue
        if shock_flag[t] != 1:
            continue

        d = 1 if vel[t] > 0 else -1
        planned = -d

        if markov_thr > -900.0:
            ms = markov_sig[t]
            if float(d) * ms <= markov_thr:
                skipped_m += 1; continue

        if ws_thr >= 0.0:
            w = ws_aligned[t]
            if np.isnan(w):
                skipped_w += 1; continue
            if abs(w) < ws_thr:
                skipped_w += 1; continue
            if (w > 0 and planned != 1) or (w < 0 and planned != -1):
                skipped_w += 1; continue

        peak_ask = ask[t]; peak_bid = bid[t]
        for k in range(1, pb + 1):
            j = t + k
            if ask[j] > peak_ask: peak_ask = ask[j]
            if bid[j] < peak_bid: peak_bid = bid[j]

        sp = (ask[t] - bid[t]) / pip
        watch_start = t + pb + 1
        watch_end = t + pb + hor
        if watch_start >= n or watch_end >= n:
            continue

        fld = 0; tp = 0; fill_price = 0.0; pnl = 0.0; tp_level = 0.0; entry = 0.0

        if stop_pips == 0.0:
            fld = 1
            fill_price = bid[watch_start] if d == 1 else ask[watch_start]
            tp_level = fill_price - tp_pips * pip * d
            if d == 1 and bid[watch_start] <= tp_level:
                tp = 1; pnl = tp_pips - sp
            elif d == -1 and ask[watch_start] >= tp_level:
                tp = 1; pnl = tp_pips - sp
            loop_start = watch_start + 1
        else:
            entry = peak_ask - stop_pips * pip if d == 1 else peak_bid + stop_pips * pip
            tp_level = entry - tp_pips * pip * d
            loop_start = watch_start

        for j in range(loop_start, min(watch_end + 1, n - 1)):
            lo = bid[j]; hi = ask[j]
            if stop_pips > 0.0 and fld == 0:
                if d == 1 and lo <= entry:
                    fld = 1; fill_price = entry
                    if lo <= tp_level:
                        tp = 1; pnl = tp_pips - sp
                elif d == -1 and hi >= entry:
                    fld = 1; fill_price = entry
                    if hi >= tp_level:
                        tp = 1; pnl = tp_pips - sp
            if fld == 1 and tp == 0:
                if d == 1 and lo <= tp_level:
                    tp = 1; pnl = tp_pips - sp
                elif d == -1 and hi >= tp_level:
                    tp = 1; pnl = tp_pips - sp
            if fld == 1 and tp == 1:
                break

        if fld == 1 and tp == 0:
            end_j = min(watch_end, n - 1)
            if d == 1:
                pnl = (fill_price - bid[end_j]) / pip - sp
            else:
                pnl = (ask[end_j] - fill_price) / pip - sp
        elif fld == 0:
            pnl = 0.0

        if ev_count < max_ev:
            filled[ev_count]  = fld
            tp_hit[ev_count]  = tp
            pnl_out[ev_count] = pnl
            ev_count += 1

        cooldown = (pb + hor) // 2

    return filled[:ev_count], tp_hit[:ev_count], pnl_out[:ev_count], skipped_m, skipped_w


def warmup_jit():
    n = 3000
    b = np.ones(n) * 214.0; a = b + 0.03; c = b + 0.015
    v = np.zeros(n); v[100]=1.2; v[500]=-0.8
    sf = np.zeros(n, dtype=np.int8); sf[100]=1; sf[500]=1
    ms = np.zeros(n); ws = np.zeros(n)
    sim(b, a, c, sf, v, ms, ws, PIP, PEAK_BARS_S5, STOP_DIST_PIP, TP_PIP, 120, -999.0, -1.0, 0, n)
    sim(b, a, c, sf, v, ms, ws, PIP, PEAK_BARS_S5, STOP_DIST_PIP, TP_PIP, 120, 0.20, 0.5, 0, n)


def run_pair(pair):
    t0 = time.time()
    s5 = (pd.read_parquet(S5_DIR / f"{pair}_S5_BA.parquet")
            .set_index("timestamp").sort_index())
    s5 = s5.astype({c:"float64" for c in s5.select_dtypes("float32").columns})
    m5 = (pd.read_parquet(M5_DIR / f"{pair}_M5_BA.parquet")
            .set_index("timestamp").sort_index())
    m5 = m5.astype({c:"float64" for c in m5.select_dtypes("float32").columns})

    bid   = s5["bid_c"].values
    ask   = s5["ask_c"].values
    close = s5["close"].values
    z, vel = compute_shock_z(close, PIP)
    shock_flag = (np.abs(z) >= THR_Z).astype(np.int8)

    ws_m5 = compute_m5_ws(m5["close"].values, PIP)
    ws_aligned = align_to_s5(s5.index, m5.index, ws_m5)

    daily_markov = build_markov_signals(close, s5.index)
    markov_aligned = map_markov_to_s5(s5.index, daily_markov)

    n_total = len(s5)
    n_is    = int(n_total * IS_FRAC)
    is_days  = n_is / 17280.0
    oos_days = (n_total - n_is) / 17280.0

    print(f"  {pair}: {time.time()-t0:.1f}s data prep, IS={is_days:.1f}d OOS={oos_days:.1f}d", flush=True)
    return dict(pair=pair, bid=bid, ask=ask, close=close, vel=vel,
                shock_flag=shock_flag, ws=ws_aligned, markov=markov_aligned,
                n_is=n_is, n_total=n_total,
                is_days=is_days, oos_days=oos_days)


def eval_cell(pair_data, slice_start, slice_end, markov_thr, ws_thr):
    """Run sim on [slice_start, slice_end) for each pair; return sum_pnl, n_fill."""
    sum_pnl = 0.0
    sum_n   = 0
    per_pair = {}
    for d in pair_data:
        fld, _, pnl, sk_m, sk_w = sim(
            d["bid"], d["ask"], d["close"], d["shock_flag"], d["vel"],
            d["markov"], d["ws"], PIP,
            PEAK_BARS_S5, STOP_DIST_PIP, TP_PIP, HORIZON_S5,
            float(markov_thr), float(ws_thr),
            int(slice_start), int(slice_end))
        n = int(fld.sum())
        net = float(pnl.sum())
        sum_pnl += net
        sum_n += n
        per_pair[d["pair"]] = (n, net, sk_m, sk_w)
    return sum_pnl, sum_n, per_pair


def main():
    warmup_jit()
    print("IS-fit then sealed OOS confirm — WS threshold + Markov×WS stack")
    print(f"  WS grid: {WS_GRID}")
    print(f"  IS = first 70%, OOS = last 30%")
    pair_data = [run_pair(p) for p in PAIRS]

    # Same IS-days / OOS-days for all 4 pairs ~ identical, average
    is_days_avg  = np.mean([d["is_days"]  for d in pair_data])
    oos_days_avg = np.mean([d["oos_days"] for d in pair_data])
    is_total_days  = sum(d["is_days"]  for d in pair_data) / 4   # avg, since all ~ same
    oos_total_days = sum(d["oos_days"] for d in pair_data) / 4

    print(f"\nIS days (avg): {is_total_days:.1f}   OOS days (avg): {oos_total_days:.1f}")
    print(f"Σ p/d below is summed over 4 pairs.")

    # IS slice = [Z_WIN, n_is) for each pair (we use n_is from first pair's grid;
    # they should be approximately equal across pairs but we use per-pair n_is
    # inside eval_cell to keep coherent. Pass uniform slice_end = first n_is
    # and trust slicing via shock_flag[t] checks. Cleanest: pass two boundaries
    # to sim — already does via slice_start/slice_end.)
    # Simpler: run IS = [0, n_is), OOS = [n_is, n_total) per-pair.
    # eval_cell accepts global ints; but each pair has its own n_is. Refactor:

    # Rerun per-pair eval to use per-pair n_is properly.
    def eval_cell_perpair(slice_label, markov_thr, ws_thr):
        sum_pnl = 0.0; sum_n = 0
        per_pair = {}
        for d in pair_data:
            if slice_label == "IS":
                start = 0; end = d["n_is"]
                days  = d["is_days"]
            else:
                start = d["n_is"]; end = d["n_total"]
                days  = d["oos_days"]
            fld, _, pnl, sk_m, sk_w = sim(
                d["bid"], d["ask"], d["close"], d["shock_flag"], d["vel"],
                d["markov"], d["ws"], PIP,
                PEAK_BARS_S5, STOP_DIST_PIP, TP_PIP, HORIZON_S5,
                float(markov_thr), float(ws_thr),
                int(start), int(end))
            n = int(fld.sum())
            net = float(pnl.sum())
            sum_pnl += net; sum_n += n
            per_pair[d["pair"]] = dict(n=n, net=net, ppd=net/days, sk_m=sk_m, sk_w=sk_w)
        return sum_pnl, sum_n, per_pair

    # ── Phase 1: IS sweep for WS-only ─────────────────────────────────────────
    print("\n" + "="*100)
    print("  PHASE 1 — IS sweep (WS-only, no Markov)")
    print("="*100)
    is_results_ws = []
    for ws_thr in WS_GRID:
        s, n, _ = eval_cell_perpair("IS", -999.0, ws_thr)
        ppd = s / is_total_days
        is_results_ws.append((ws_thr, ppd, n, s))
        print(f"  ws_thr={ws_thr:<5}  IS Σ p/d = {ppd:>+7.2f}   n_fills = {n:>6}")
    best_ws_only = max(is_results_ws, key=lambda x: x[1])
    print(f"\n  IS winner (WS-only): ws_thr = {best_ws_only[0]}  IS Σ p/d = {best_ws_only[1]:+.2f}")

    # ── Phase 2: IS sweep for Markov+WS ───────────────────────────────────────
    print("\n" + "="*100)
    print("  PHASE 2 — IS sweep (Markov + WS)")
    print("="*100)
    is_results_mw = []
    for ws_thr in WS_GRID:
        s, n, _ = eval_cell_perpair("IS", MARKOV_SIG_THR, ws_thr)
        ppd = s / is_total_days
        is_results_mw.append((ws_thr, ppd, n, s))
        print(f"  ws_thr={ws_thr:<5}  IS Σ p/d = {ppd:>+7.2f}   n_fills = {n:>6}")
    best_mw = max(is_results_mw, key=lambda x: x[1])
    print(f"\n  IS winner (Markov+WS): ws_thr = {best_mw[0]}  IS Σ p/d = {best_mw[1]:+.2f}")

    # Baseline & Markov-only IS reference
    s_base_is, n_base_is, _ = eval_cell_perpair("IS", -999.0, -1.0)
    s_m_is, n_m_is, _ = eval_cell_perpair("IS", MARKOV_SIG_THR, -1.0)
    print(f"\n  Reference IS: baseline Σ p/d = {s_base_is/is_total_days:+.2f}   "
          f"markov-only Σ p/d = {s_m_is/is_total_days:+.2f}")

    # ── Phase 3: SEALED OOS evaluation at IS-picked thresholds ───────────────
    print("\n" + "="*100)
    print(f"  PHASE 3 — SEALED OOS evaluation (apply IS-picked thresholds once)")
    print("="*100)

    cells = [
        ("baseline",       -999.0, -1.0,                "—"),
        ("markov_only",    MARKOV_SIG_THR, -1.0,        "—"),
        ("ws_isfit",       -999.0, float(best_ws_only[0]),  f"IS-pick ws_thr={best_ws_only[0]}"),
        ("markov+ws_isfit", MARKOV_SIG_THR, float(best_mw[0]), f"IS-pick ws_thr={best_mw[0]}"),
    ]
    rows = []
    for cell_name, mthr, wthr, note in cells:
        s, n, pp = eval_cell_perpair("OOS", mthr, wthr)
        ppd = s / oos_total_days
        rows.append((cell_name, ppd, s, n, note, pp))

    print(f"\n  {'cell':<22}  {'Σ p/d':>8}  {'Σ pips':>9}  {'fills':>6}  threshold")
    for name, ppd, s, n, note, _ in rows:
        print(f"  {name:<22}  {ppd:>+8.2f}  {s:>+9.1f}  {n:>6}  {note}")

    # Per-pair table
    print("\n  Per-pair OOS p/d at each cell:")
    print(f"  {'pair':<10}", end="")
    for name, _, _, _, _, _ in rows:
        print(f"{name:>22}", end="")
    print()
    for pair in PAIRS:
        print(f"  {pair:<10}", end="")
        for _, _, _, _, _, pp in rows:
            v = pp[pair]
            print(f"{v['ppd']:>+22.2f}", end="")
        print()
    print(f"\n  Per-pair OOS n_fills at each cell:")
    print(f"  {'pair':<10}", end="")
    for name, _, _, _, _, _ in rows:
        print(f"{name:>22}", end="")
    print()
    for pair in PAIRS:
        print(f"  {pair:<10}", end="")
        for _, _, _, _, _, pp in rows:
            v = pp[pair]
            print(f"{v['n']:>22}", end="")
        print()

    # Save CSV
    import csv
    out_csv = OUT / "retrace_ws_isfit_low.csv"
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["phase","cell","ws_thr","sum_ppd","sum_pips","n_fills","note"])
        for ws_thr, ppd, n, s in is_results_ws:
            w.writerow(["IS","ws_only",ws_thr,round(ppd,2),round(s,1),n,""])
        for ws_thr, ppd, n, s in is_results_mw:
            w.writerow(["IS","markov+ws",ws_thr,round(ppd,2),round(s,1),n,""])
        w.writerow(["IS","baseline","",round(s_base_is/is_total_days,2),round(s_base_is,1),n_base_is,""])
        w.writerow(["IS","markov_only","",round(s_m_is/is_total_days,2),round(s_m_is,1),n_m_is,""])
        for name, ppd, s, n, note, _ in rows:
            w.writerow(["OOS",name,"",round(ppd,2),round(s,1),n,note])
    print(f"\n  → {out_csv}")


if __name__ == "__main__":
    main()
