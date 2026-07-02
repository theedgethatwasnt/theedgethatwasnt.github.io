#!/usr/bin/env python3
"""
Experiment B — Tighter M5+WS entry: accel filter + pair-specific WS percentile.

Two refinements stacked on the existing M5+WS standalone:
  R1: require accel of the 5m window > 0 for LONG (< 0 for SHORT). Accel is
      Δ(5m momentum) / 5min — same definition as the dashboard.
  R2: replace the global WS=1.5 threshold with a pair-specific percentile of
      |WS| computed on IS history only. Pair-specific because pairs with
      higher typical volatility (JPY) have larger natural |WS| values.

Split discipline (the OOS we touched in m5_ws_sweep.csv is now soiled):
  - IS    = first 70% of bars         (build WS percentile thresholds)
  - OOS-A = bars 70%-85%              (parameter discovery; touchable)
  - OOS-B = bars 85%-100%             (sealed; reported once at the end)

Entry rule (after both refinements):
  LONG  if m5[i-1] <= 0 < m5[i]    AND accel[i] > 0
                                 AND ws[i] > pair_thr_pos[pair]
                                 AND sp[i] <= sp_gate
  SHORT symmetric.

Exit: honest TP_TIME24h (TP at TP_pips OR close at 288 M5 bars). NO PIP STOP
(we know any pip stop kills the family). Time cap is the honest constraint.

Sweep
-----
TP_pips          ∈ {10, 15, 20, 30}
ws_percentile    ∈ {90, 95, 98}   (pair-specific quantile of |WS| from IS)
accel_required   ∈ {False, True}

Output
------
results/m5_ws_tight_oos_a.csv     (discovery)
results/m5_ws_tight_oos_b.csv     (confirmation)
"""
import gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
OUT     = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

ALL_PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
             "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}

SP_GATES = {
    "GBP_JPY":4.00,"USD_JPY":2.10,"EUR_JPY":2.50,"GBP_USD":2.40,
    "AUD_JPY":2.30,"EUR_USD":1.70,"AUD_USD":1.60,"NZD_JPY":3.10,
    "CHF_JPY":3.70,"NZD_USD":2.00,"CAD_JPY":2.60,"EUR_GBP":2.00,
}

WIN_LAGS_M5 = [1, 3, 12, 48, 288]
WIN_MINUTES = [5, 15, 60, 240, 1440]
WIN_WEIGHTS = [0.10, 0.15, 0.20, 0.20, 0.25]

# Splits (in fractions of bar count)
IS_FRAC   = 0.70
OOS_A_END = 0.85   # OOS-A = 70-85%  ; OOS-B = 85-100%

TPS         = [10.0, 15.0, 20.0, 30.0]
PCT_LIST    = [90, 95, 98]
TIME_CAP_M5 = 288


def pip_sz(p): return 0.01 if p in JPY else 0.0001


def build_signals(close, pip):
    n = len(close)
    moms = np.full((len(WIN_LAGS_M5), n), np.nan, dtype=np.float64)
    for j, (lag, mins) in enumerate(zip(WIN_LAGS_M5, WIN_MINUTES)):
        if lag >= n: continue
        moms[j, :lag] = np.nan
        moms[j, lag:] = (close[lag:] - close[:-lag]) / pip / float(mins)
    weights = np.asarray(WIN_WEIGHTS, dtype=np.float64)
    valid   = ~np.isnan(moms)
    w_avail = np.where(valid, weights[:, None], 0.0).sum(axis=0)
    wv      = np.where(valid, moms * weights[:, None], 0.0).sum(axis=0)
    ws      = np.where(w_avail > 0, wv / w_avail, np.nan)
    m5 = moms[0]
    # accel = Δ5m / 5min (per dashboard: (cur-prv)/0.5min, but here we move by
    # one M5 bar, which is 5 min — so divide by 5 for pips/min/min units;
    # sign is what matters, so factor doesn't affect the >0 check).
    accel = np.empty(n, dtype=np.float64); accel[:] = np.nan
    accel[1:] = m5[1:] - m5[:-1]
    return m5, ws, accel


@njit(cache=True)
def _sim(bid, ask, mid, sp, m5, ws, accel, pip, tp_pips, time_cap, sp_gate,
         ws_thr_pos, ws_thr_neg, require_accel, start_i, end_i):
    n = end_i
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)
    count = 0
    in_trade = False; wait_zero = False
    dir_ = 0; ep = 0.0; ei = 0
    prev_m = 0.0; have_prev = False

    for i in range(start_i, end_i):
        cur_m  = m5[i]
        cur_ws = ws[i]
        cur_a  = accel[i]
        if np.isnan(cur_m) or np.isnan(cur_ws) or np.isnan(cur_a):
            have_prev = False
            continue

        if in_trade:
            excur = (mid[i] - ep) / pip * dir_
            if excur >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 0
                count += 1; in_trade = False; wait_zero = True
            elif (i - ei) >= time_cap:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 2
                count += 1; in_trade = False; wait_zero = True
        else:
            if not have_prev:
                prev_m = cur_m; have_prev = True; continue
            if wait_zero:
                if (prev_m * cur_m) < 0.0 or cur_m == 0.0:
                    wait_zero = False
            if (not wait_zero) and sp[i] <= sp_gate:
                if prev_m <= 0.0 < cur_m and cur_ws > ws_thr_pos:
                    if (not require_accel) or cur_a > 0.0:
                        ep = ask[i]; dir_ = 1; ei = i; in_trade = True
                elif prev_m >= 0.0 > cur_m and cur_ws < ws_thr_neg:
                    if (not require_accel) or cur_a < 0.0:
                        ep = bid[i]; dir_ = -1; ei = i; in_trade = True
        prev_m = cur_m

    return pnl_out[:count], hold_out[:count], type_out[:count]


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def warmup_jit():
    n = 500
    z = np.zeros(n); m = np.zeros(n); w = np.zeros(n); a = np.zeros(n)
    _sim(z,z,z,z,m,w,a, 0.0001, 10.0, 288, 2.0, 1.5, -1.5, True, 0, n)


def run_pair(pair, all_rows, label_split, start_i, end_i):
    df = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
          .set_index("timestamp").sort_index())
    df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    pip = pip_sz(pair); sg = SP_GATES[pair]
    m5, ws, accel = build_signals(df["close"].values.astype(np.float64), pip)
    n_total = len(df)
    n_is    = int(n_total * IS_FRAC)

    # IS-only quantiles of |WS|
    is_ws = ws[:n_is]
    is_ws = is_ws[~np.isnan(is_ws)]

    bid = df["bid_c"].values.astype(np.float64)
    ask = df["ask_c"].values.astype(np.float64)
    mid = df["close"].values.astype(np.float64)
    sp  = ((ask - bid) / pip).astype(np.float64)

    days = (end_i - start_i) / 288.0

    for pct in PCT_LIST:
        if len(is_ws) == 0:
            thr_pos = float("nan"); thr_neg = float("nan")
        else:
            thr_abs = float(np.quantile(np.abs(is_ws), pct / 100.0))
            thr_pos = +thr_abs
            thr_neg = -thr_abs
        for tp in TPS:
            for req_accel in (False, True):
                p, h, t = _sim(bid, ask, mid, sp, m5, ws, accel, pip,
                               tp, TIME_CAP_M5, sg,
                               thr_pos, thr_neg, req_accel,
                               start_i, end_i)
                n = len(p)
                if n == 0:
                    all_rows.append(dict(split=label_split, pair=pair, pct=pct,
                                         thr_abs=round(thr_pos,3),
                                         tp=tp, accel=req_accel,
                                         n=0, ppd=0.0, wr=0.0, mdd=0.0,
                                         calmar=0.0, mean_hold=0.0, days=days))
                    continue
                wr  = (p>0).sum() / n * 100
                ppd = p.sum() / days
                mdd = max_dd(p)
                cal = ppd / mdd if mdd > 0 else 0.0
                all_rows.append(dict(split=label_split, pair=pair, pct=pct,
                                     thr_abs=round(thr_pos,3),
                                     tp=tp, accel=req_accel,
                                     n=n, ppd=round(ppd,2), wr=round(wr,1),
                                     mdd=round(mdd,1), calmar=round(cal,2),
                                     mean_hold=round(float(h.mean()),1),
                                     days=round(days,1)))


def main():
    warmup_jit()
    print("Experiment B — tighter M5+WS (accel + pair-percentile)")
    print(f"  PCTs={PCT_LIST}  TPs={TPS}  accel={{False,True}}")
    print(f"  splits: IS=[0, 70%) → percentile build")
    print(f"          OOS-A=[70, 85%)  discovery")
    print(f"          OOS-B=[85, 100%) sealed confirmation")
    all_rows = []
    t0 = time.time()

    # Two passes: OOS-A and OOS-B
    for pair in ALL_PAIRS:
        df = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
              .set_index("timestamp").sort_index())
        n_total = len(df)
        n_is    = int(n_total * IS_FRAC)
        n_oosA  = int(n_total * OOS_A_END)
        del df
        t1 = time.time()
        run_pair(pair, all_rows, "OOS_A", n_is, n_oosA)
        run_pair(pair, all_rows, "OOS_B", n_oosA, n_total)
        gc.collect()
        print(f"  {pair}: {time.time()-t1:.1f}s", flush=True)

    df = pd.DataFrame(all_rows)
    out_csv = OUT / "m5_ws_tight.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    # ── Summary: best config on OOS-A, then evaluate same config on OOS-B ────
    a = df[df.split == "OOS_A"]
    b = df[df.split == "OOS_B"]
    print("\n" + "="*100)
    print("  Top configs on OOS-A (discovery)")
    print("="*100)
    g = (a.groupby(["pct","tp","accel"])
            .agg(sum_ppd=("ppd","sum"),
                 n_pos=("ppd", lambda x: int((x>0).sum())),
                 total_n=("n","sum"),
                 mean_wr=("wr","mean"))
            .reset_index().sort_values("sum_ppd", ascending=False))
    print(f"  {'pct':>4}  {'tp':>5}  {'accel':>6}  {'Σ ppd':>7}  {'pos':>3}/12  {'Σn':>6}  {'WR%':>5}")
    for _, r in g.head(10).iterrows():
        print(f"  {int(r['pct']):>4d}  {r['tp']:>5.1f}  {str(bool(r['accel'])):>6}  "
              f"{r['sum_ppd']:>+7.2f}  {int(r['n_pos']):>3d}/12  "
              f"{int(r['total_n']):>6d}  {r['mean_wr']:>5.1f}")

    if not g.empty:
        top = g.iloc[0]
        print("\n" + "="*100)
        print(f"  Top OOS-A config replayed on OOS-B (sealed):  "
              f"pct={int(top['pct'])}  tp={top['tp']}  accel={bool(top['accel'])}")
        print("="*100)
        sub = b[(b.pct==top['pct'])&(b.tp==top['tp'])&(b.accel==bool(top['accel']))]
        sub = sub.sort_values("ppd", ascending=False)
        print(f"  {'pair':<10}{'n':>5}{'ppd':>8}{'WR%':>6}{'MDD':>7}{'Calmar':>7}{'hold':>6}")
        for _, r in sub.iterrows():
            print(f"  {r['pair']:<10}{int(r['n']):>5}{r['ppd']:>+8.2f}{r['wr']:>6.1f}"
                  f"{r['mdd']:>7.1f}{r['calmar']:>7.2f}{r['mean_hold']:>6.1f}")
        sum_b = sub['ppd'].sum()
        n_pos_b = int((sub['ppd']>0).sum())
        print(f"\n  OOS-B portfolio: Σ ppd = {sum_b:+.2f}   pairs+ = {n_pos_b}/12")


if __name__ == "__main__":
    main()
