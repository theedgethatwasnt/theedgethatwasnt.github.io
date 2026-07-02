#!/usr/bin/env python3
"""
Experiment C — S5 momentum entries using the full 7-window WS gate.

Uses S5 BA data (4 JPY pairs available) to:
  - Compute momentum at ALL 7 dashboard windows (S5 + M1 + 5m + 15m + 1h + 4h + 24h)
    against the dashboard's exact weights summing to 1.0:
        S5=0.05  M1=0.05  5m=0.10  15m=0.15  1h=0.20  4h=0.20  24h=0.25
  - Entry on M1-bar boundaries (M1 momentum crosses zero) gated on WS direction +
    magnitude. We trigger off M1 cross (not S5 cross — too noisy) so that the
    entry rate is sane.
  - Exits at TP or time-cap, sampled at native S5 resolution.

Why M1 cross (and not S5 cross)?
  S5 momentum = velocity over 5 seconds. Crosses zero ~once per minute on
  average — far too noisy. M1 cross fires ~1-2× per hour and matches the
  cadence of trade decisions a human trader could meaningfully reason about.

Window-bar counts (in S5 bars, 12 S5 bars per minute):
    S5  : 1     (5s)
    M1  : 12    (1min)
    5m  : 60
    15m : 180
    1h  : 720
    4h  : 2880
    24h : 17280

Sweep
-----
ws_thr ∈ {0.5, 1.0, 1.5, 2.0}
tp     ∈ {5, 10, 20, 30}
time   ∈ {S5 bars: 60 (5min), 720 (1h), 2880 (4h), 17280 (24h)}
Pairs  : GBP_JPY, USD_JPY, EUR_JPY, AUD_JPY

Output
------
results/s5_ws_sweep.csv (pair × ws_thr × tp × time_cap)
"""
import gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "s5_ba"
OUT     = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True, parents=True)

PAIRS = ["GBP_JPY", "USD_JPY", "EUR_JPY", "AUD_JPY"]
PIP   = 0.01

# Full 7-window dashboard config (S5 bar count + minutes + weight)
WIN_S5_LAGS = [1, 12, 60, 180, 720, 2880, 17280]
WIN_MINUTES = [5/60.0, 1.0, 5.0, 15.0, 60.0, 240.0, 1440.0]
WIN_WEIGHTS = [0.05, 0.05, 0.10, 0.15, 0.20, 0.20, 0.25]

# Per-pair IS-P90 spread gates (JPY pairs only here)
SP_GATES = {"GBP_JPY":4.00,"USD_JPY":2.10,"EUR_JPY":2.50,"AUD_JPY":2.30}

IS_FRAC = 0.70

WS_THRS    = [0.5, 1.0, 1.5, 2.0]
TPS        = [5.0, 10.0, 20.0, 30.0]
TIME_CAPS  = [60, 720, 2880, 17280]   # S5 bars: 5min/1h/4h/24h


def build_s5_signals(close: np.ndarray, pip: float):
    """7-window pips/min momentum + weighted_sum on S5 closes.

    Returns:
      m1_mom : M1-window momentum (pips/min), array of length n
      ws     : weighted_sum across 7 windows (mean over available)
    """
    n = len(close)
    moms = np.full((len(WIN_S5_LAGS), n), np.nan, dtype=np.float64)
    for j, (lag, mins) in enumerate(zip(WIN_S5_LAGS, WIN_MINUTES)):
        if lag >= n: continue
        moms[j, :lag] = np.nan
        moms[j, lag:] = (close[lag:] - close[:-lag]) / pip / float(mins)
    weights = np.asarray(WIN_WEIGHTS, dtype=np.float64)
    valid   = ~np.isnan(moms)
    w_avail = np.where(valid, weights[:, None], 0.0).sum(axis=0)
    wv      = np.where(valid, moms * weights[:, None], 0.0).sum(axis=0)
    ws      = np.where(w_avail > 0, wv / w_avail, np.nan)
    return moms[1], ws    # M1 momentum, weighted_sum


@njit(cache=True)
def _sim_s5(bid, ask, mid, m1, ws, pip,
            ws_thr, tp_pips, time_cap, sp_gate):
    """Trigger entry at M1 boundary (every 12 S5 bars), check M1 momentum cross
    + WS gate. Exits sampled bar-by-bar.
    """
    n = len(mid)
    max_ev = n // 12 // 4   # liberal upper bound
    pnl_out  = np.empty(max_ev, dtype=np.float64)
    hold_out = np.empty(max_ev, dtype=np.int32)
    type_out = np.empty(max_ev, dtype=np.int8)
    count = 0
    in_trade = False; wait_zero = False
    dir_ = 0; ep = 0.0; ei = 0
    prev_m1 = 0.0; have_prev = False

    for i in range(n):
        if in_trade:
            excur = (mid[i] - ep) / pip * dir_
            if excur >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                sp_now = (ask[i] - bid[i]) / pip
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp_now
                hold_out[count] = i - ei
                type_out[count] = 0
                count += 1; in_trade = False; wait_zero = True
            elif (i - ei) >= time_cap:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                sp_now = (ask[i] - bid[i]) / pip
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp_now
                hold_out[count] = i - ei
                type_out[count] = 2
                count += 1; in_trade = False; wait_zero = True
            continue

        # Only consider M1 boundaries (12 S5 bars apart) — keeps signal cadence
        # at minute-level. Use i % 12 == 0 as proxy.
        if i % 12 != 0:
            continue

        m1_cur = m1[i]
        ws_cur = ws[i]
        if np.isnan(m1_cur) or np.isnan(ws_cur):
            have_prev = False
            continue

        if not have_prev:
            prev_m1 = m1_cur
            have_prev = True
            continue

        if wait_zero:
            if (prev_m1 * m1_cur) < 0.0 or m1_cur == 0.0:
                wait_zero = False

        if not wait_zero:
            sp_now = (ask[i] - bid[i]) / pip
            if sp_now <= sp_gate:
                if prev_m1 <= 0.0 < m1_cur and ws_cur > ws_thr:
                    ep = ask[i]; dir_ = 1; ei = i; in_trade = True
                elif prev_m1 >= 0.0 > m1_cur and ws_cur < -ws_thr:
                    ep = bid[i]; dir_ = -1; ei = i; in_trade = True

        prev_m1 = m1_cur

    return pnl_out[:count], hold_out[:count], type_out[:count]


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def warmup_jit():
    n = 100
    z = np.zeros(n); m = np.zeros(n); w = np.zeros(n)
    _sim_s5(z, z, z, m, w, PIP, 1.5, 10.0, 60, 2.0)


def run_pair(pair, all_rows):
    t0 = time.time()
    df = (pd.read_parquet(DATA / f"{pair}_S5_BA.parquet")
          .set_index("timestamp").sort_index())
    df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    n_total = len(df)
    n_is    = int(n_total * IS_FRAC)
    print(f"  {pair}: loaded {n_total:,} S5 rows; OOS slice = {n_total-n_is:,}", flush=True)

    close = df["close"].values.astype(np.float64)
    m1, ws = build_s5_signals(close, PIP)

    bid = df["bid_c"].values.astype(np.float64)[n_is:]
    ask = df["ask_c"].values.astype(np.float64)[n_is:]
    mid = close[n_is:]
    m1_o = m1[n_is:]
    ws_o = ws[n_is:]
    days = len(mid) / 17280.0   # 17280 S5 bars per 24h
    sg = SP_GATES[pair]

    for ws_thr in WS_THRS:
        for tp in TPS:
            for tc in TIME_CAPS:
                p, h, t = _sim_s5(bid, ask, mid, m1_o, ws_o, PIP,
                                  ws_thr, tp, tc, sg)
                n = len(p)
                if n == 0:
                    all_rows.append(dict(pair=pair, ws_thr=ws_thr, tp=tp,
                                         tc_s5=tc, n=0, ppd=0.0, wr=0.0,
                                         mdd=0.0, calmar=0.0, mean_hold_s5=0.0,
                                         days=days))
                    continue
                wr  = (p>0).sum() / n * 100
                ppd = p.sum() / days
                mdd = max_dd(p)
                cal = ppd / mdd if mdd > 0 else 0.0
                all_rows.append(dict(pair=pair, ws_thr=ws_thr, tp=tp,
                                     tc_s5=tc, n=n, ppd=round(ppd,2),
                                     wr=round(wr,1), mdd=round(mdd,1),
                                     calmar=round(cal,2),
                                     mean_hold_s5=round(float(h.mean()),1),
                                     days=round(days,1)))
    del df, close, m1, ws, bid, ask, mid, m1_o, ws_o
    gc.collect()
    print(f"  {pair} done ({time.time()-t0:.1f}s)", flush=True)


def main():
    warmup_jit()
    print("Experiment C — S5 momentum entries (7-window WS gate)")
    print(f"  pairs={PAIRS}  ws_thrs={WS_THRS}  TPs={TPS}")
    print(f"  time caps (S5 bars): {TIME_CAPS}  (5min, 1h, 4h, 24h)")
    rows = []
    t0 = time.time()
    for pair in PAIRS:
        run_pair(pair, rows)
    df = pd.DataFrame(rows)
    out_csv = OUT / "s5_ws_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    print("\n" + "="*100)
    print("  Σ p/d across 4 JPY pairs, top 15 configs")
    print("="*100)
    g = (df.groupby(["ws_thr","tp","tc_s5"])
            .agg(sum_ppd=("ppd","sum"),
                 n_pos=("ppd", lambda x: int((x>0).sum())),
                 total_n=("n","sum"),
                 mean_wr=("wr","mean"))
            .reset_index()
            .sort_values("sum_ppd", ascending=False))
    print(f"  {'ws_thr':>7}  {'tp':>5}  {'tc(S5)':>7}  {'Σ ppd':>7}  {'pos':>3}/4 {'Σn':>7}  {'WR%':>5}")
    for _, r in g.head(15).iterrows():
        print(f"  {r['ws_thr']:>7.1f}  {r['tp']:>5.1f}  {int(r['tc_s5']):>7d}  "
              f"{r['sum_ppd']:>+7.2f}  {int(r['n_pos']):>3d}/4 {int(r['total_n']):>7d}  {r['mean_wr']:>5.1f}")

    if not g.empty:
        top = g.iloc[0]
        print("\n" + "="*100)
        print(f"  Per-pair at top config: ws_thr={top['ws_thr']}  tp={top['tp']}  tc_s5={int(top['tc_s5'])}")
        print("="*100)
        sub = df[(df.ws_thr==top['ws_thr'])&(df.tp==top['tp'])&(df.tc_s5==top['tc_s5'])]
        sub = sub.sort_values("ppd", ascending=False)
        print(f"  {'pair':<10}{'n':>5}{'ppd':>8}{'WR%':>6}{'MDD':>7}{'Calmar':>7}{'hold(S5)':>10}")
        for _, r in sub.iterrows():
            print(f"  {r['pair']:<10}{int(r['n']):>5}{r['ppd']:>+8.2f}{r['wr']:>6.1f}"
                  f"{r['mdd']:>7.1f}{r['calmar']:>7.2f}{r['mean_hold_s5']:>10.1f}")


if __name__ == "__main__":
    main()
