#!/usr/bin/env python3
"""
Dashboard M5-momentum + weighted_sum gate — backtest sweep.

Origin
------
The user looked at the live dashboard (services/fx_signals/main.py) and asked:
"Build the experiment around what you see — positive forcing of the M5
momentum, and weighted_sum greater than 1.5, responding to price increase."

Signal definitions (R6: byte-equivalent to fx_signals/main.py)
-------------------------------------------------------------
At every M5 close, for each pair, compute pips-per-minute momentum at 5
horizons against M5 closes:

  5m  = (close[t] - close[t-1])   / pip /    5
  15m = (close[t] - close[t-3])   / pip /   15
  1h  = (close[t] - close[t-12])  / pip /   60
  4h  = (close[t] - close[t-48])  / pip /  240
  24h = (close[t] - close[t-288]) / pip / 1440

weighted_sum = Σ(w_i × momentum_i) / Σ(w_i)   (a weighted MEAN, not raw sum)

Dashboard weights, the 5 windows that derive from M5 closes:
  5m=0.10  15m=0.15  1h=0.20  4h=0.20  24h=0.25   (Σ=0.90)

The dashboard also tracks S5(0.05) and M1(0.05), which we drop in backtest —
their sum is 0.10 / 0.90 of the total weight basis. The threshold is unaffected
because `compute_weighted_sum` is a mean over available windows, not a sum.

Entry rule
----------
LONG  : m5[i-1] <= 0 < m5[i]   AND  ws[i] >  +WS_THR  AND  spread <= sp_gate
SHORT : m5[i-1] >= 0 > m5[i]   AND  ws[i] <  -WS_THR  AND  spread <= sp_gate

Wait-for-zero between exits (no re-entry until either momentum returns to
opposite sign of last position or the WS threshold gate disarms).

Exit
----
Modes tested per (pair, WS_THR, TP):
  TP_ONLY      : exit when MFE reaches TP
  TP_SL50      : TP=+TP, fixed SL=-50p
  TP_TIME288   : TP or close at 24h (288 M5 bars)
  TP_SIG_DIE   : TP or close when ws crosses back through 0

R-compliance
------------
R1 closed bars only (sig built from completed bars).
R3 mid OHLC for signal + (ask_c-bid_c)/pip explicit cost.
R5 spread gate IS-only (per-pair IS-P90, hardcoded from live).
R6 single signal builder — same array logic the live curator uses.
R8 OOS reported once.

Output
------
results/m5_ws_sweep.csv      — full grid (pair × ws_thr × tp × mode)
Console: per-strategy ΣOOS p/d + #pairs-positive summary.

Run:
  python3 backtest_m5_ws.py
"""
import gc
import time
import warnings
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
IS_FRAC = 0.70

# Per-pair IS-P90 spread gates (copied from live services).
SP_GATES = {
    "GBP_JPY":4.00,"USD_JPY":2.10,"EUR_JPY":2.50,"GBP_USD":2.40,
    "AUD_JPY":2.30,"EUR_USD":1.70,"AUD_USD":1.60,"NZD_JPY":3.10,
    "CHF_JPY":3.70,"NZD_USD":2.00,"CAD_JPY":2.60,"EUR_GBP":2.00,
}

# Dashboard window weights, restricted to the M5-derivable horizons.
# (S5 and M1 are dropped — they live in fx_signals' 5-sec/1-min buffers, which
#  we don't replay here. weighted_sum is a mean, so dropping them is unbiased.)
WIN_LAGS_M5 = [1, 3, 12, 48, 288]              # bars
WIN_MINUTES = [5, 15, 60, 240, 1440]
WIN_WEIGHTS = [0.10, 0.15, 0.20, 0.20, 0.25]
W_SUM       = sum(WIN_WEIGHTS)                  # 0.90

# Sweep
WS_THRS    = [1.0, 1.5, 2.0]
TPS        = [10.0, 15.0, 20.0, 30.0]
TIME_CAP   = 288   # M5 bars = 24h
SL_FIXED   = 50.0  # pips


def pip_sz(p): return 0.01 if p in JPY else 0.0001


# ── Signal builder ────────────────────────────────────────────────────────────

def build_signals(df, pip):
    """Per-bar pips/min momentum at 5 horizons + weighted mean."""
    close = df["close"].values.astype(np.float64)
    n = len(close)
    # Pre-compute momentum series for each window. NaN where lag insufficient.
    moms = np.full((len(WIN_LAGS_M5), n), np.nan, dtype=np.float64)
    for j, (lag, mins) in enumerate(zip(WIN_LAGS_M5, WIN_MINUTES)):
        if lag >= n:
            continue
        m = np.empty(n, dtype=np.float64)
        m[:lag] = np.nan
        m[lag:] = (close[lag:] - close[:-lag]) / pip / float(mins)
        moms[j] = m
    # Weighted mean across windows that are not nan at each timestep.
    weights = np.asarray(WIN_WEIGHTS, dtype=np.float64)
    valid   = ~np.isnan(moms)
    w_avail = np.where(valid, weights[:, None], 0.0).sum(axis=0)
    wv      = np.where(valid, moms * weights[:, None], 0.0).sum(axis=0)
    ws      = np.where(w_avail > 0, wv / w_avail, np.nan)
    return moms[0], ws   # 5m momentum, weighted_sum


# ── Numba simulators ──────────────────────────────────────────────────────────

@njit(cache=True)
def _simulate(bid, ask, mid, sp, m5, ws, pip,
              ws_thr, tp_pips, sl_pips, time_cap, sig_die_exit, sp_gate):
    """Single backtest pass.

    Entry:
      LONG  if  m5[i-1] <= 0 < m5[i]   AND ws[i] >  +ws_thr AND sp[i] <= sp_gate
      SHORT if  m5[i-1] >= 0 > m5[i]   AND ws[i] <  -ws_thr AND sp[i] <= sp_gate
    Wait-for-zero before re-entry: after any exit, require m5 to cross through 0
      in the opposite direction before next entry.
    Exit precedence: TP → SL (if enabled) → time-cap (if enabled) → sig_die (if enabled).
    """
    n = len(mid)
    pnl_out  = np.empty(n, dtype=np.float64)
    hold_out = np.empty(n, dtype=np.int32)
    type_out = np.empty(n, dtype=np.int8)   # 0=TP 1=SL 2=TIME 3=SIG_DIE
    count = 0
    use_sl   = sl_pips > 0.0
    use_time = time_cap > 0
    in_trade = False; wait_zero = False
    dir_ = 0; ep = 0.0; ei = 0
    prev_m = 0.0
    have_prev = False

    for i in range(n):
        cur_m  = m5[i]
        cur_ws = ws[i]
        # Skip warm-up rows where signals not ready
        if np.isnan(cur_m) or np.isnan(cur_ws):
            have_prev = False
            continue

        if in_trade:
            excur = (mid[i] - ep) / pip * dir_
            exited = False
            if excur >= tp_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 0
                count += 1; in_trade = False; wait_zero = True; exited = True
            elif use_sl and excur <= -sl_pips:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 1
                count += 1; in_trade = False; wait_zero = True; exited = True
            elif use_time and (i - ei) >= time_cap:
                exit_px = bid[i] if dir_ == 1 else ask[i]
                pnl_out[count]  = (exit_px - ep) / pip * dir_ - sp[i]
                hold_out[count] = i - ei
                type_out[count] = 2
                count += 1; in_trade = False; wait_zero = True; exited = True
            elif sig_die_exit:
                # Close when ws magnitude collapses past zero
                if dir_ == 1 and cur_ws <= 0.0:
                    exit_px = bid[i]
                    pnl_out[count]  = (exit_px - ep) / pip - sp[i]
                    hold_out[count] = i - ei
                    type_out[count] = 3
                    count += 1; in_trade = False; wait_zero = True; exited = True
                elif dir_ == -1 and cur_ws >= 0.0:
                    exit_px = ask[i]
                    pnl_out[count]  = (ep - exit_px) / pip - sp[i]
                    hold_out[count] = i - ei
                    type_out[count] = 3
                    count += 1; in_trade = False; wait_zero = True; exited = True
        else:
            if not have_prev:
                prev_m = cur_m
                have_prev = True
                continue
            if wait_zero:
                # require momentum to revert through 0 before next entry
                if (prev_m * cur_m) < 0.0 or cur_m == 0.0:
                    wait_zero = False
            if (not wait_zero) and sp[i] <= sp_gate:
                # Cross-zero detect
                if prev_m <= 0.0 < cur_m and cur_ws > ws_thr:
                    ep = ask[i]
                    dir_ = 1; ei = i; in_trade = True
                elif prev_m >= 0.0 > cur_m and cur_ws < -ws_thr:
                    ep = bid[i]
                    dir_ = -1; ei = i; in_trade = True
        prev_m = cur_m

    return pnl_out[:count], hold_out[:count], type_out[:count]


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def warmup_jit():
    n = 500
    z = np.zeros(n);  s = np.zeros(n);  m = np.zeros(n);  w = np.zeros(n)
    _simulate(z,z,z,z,m,w,0.0001, 1.5, 10.0, -1.0, 0, 0, 2.0)
    _simulate(z,z,z,z,m,w,0.0001, 1.5, 10.0, 50.0, 288, 1, 2.0)


def run_one_pair(pair, all_rows):
    df = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
          .set_index("timestamp").sort_index())
    df = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    pip = pip_sz(pair); sg = SP_GATES[pair]
    m5, ws = build_signals(df, pip)
    n_is = int(len(df) * IS_FRAC)
    oos = df.iloc[n_is:]
    m5_o = m5[n_is:]
    ws_o = ws[n_is:]
    bid  = oos["bid_c"].values.astype(np.float64)
    ask  = oos["ask_c"].values.astype(np.float64)
    mid  = oos["close"].values.astype(np.float64)
    sp   = ((ask - bid) / pip).astype(np.float64)
    days = len(oos) / 288.0

    for ws_thr in WS_THRS:
        for tp in TPS:
            cases = [
                ("TP_ONLY",     -1.0,  0,         False),
                (f"TP_SL{int(SL_FIXED)}", SL_FIXED, 0,    False),
                ("TP_TIME24h",  -1.0,  TIME_CAP,  False),
                ("TP_SIG_DIE",  -1.0,  0,         True),
            ]
            for label, sl, t_cap, sig_die in cases:
                p,h,t = _simulate(bid, ask, mid, sp, m5_o, ws_o, pip,
                                  ws_thr, tp, sl, t_cap, 1 if sig_die else 0, sg)
                n = len(p)
                if n == 0:
                    all_rows.append(dict(pair=pair, ws_thr=ws_thr, tp=tp, mode=label,
                                         n=0, ppd=0.0, wr=0.0, tp_pct=0.0,
                                         mdd=0.0, calmar=0.0, mean_hold=0.0,
                                         days=days))
                    continue
                wr = (p>0).sum() / n * 100
                tp_pct = (t==0).sum() / n * 100
                ppd = p.sum() / days
                mdd = max_dd(p)
                cal = ppd / mdd if mdd > 0 else 0.0
                all_rows.append(dict(pair=pair, ws_thr=ws_thr, tp=tp, mode=label,
                                     n=n, ppd=round(ppd,2), wr=round(wr,1),
                                     tp_pct=round(tp_pct,1),
                                     mdd=round(mdd,1), calmar=round(cal,2),
                                     mean_hold=round(float(h.mean()),1),
                                     days=round(days,1)))


def main():
    warmup_jit()
    print("M5-momentum + weighted_sum gate sweep")
    print(f"  WS thresholds: {WS_THRS}")
    print(f"  TPs:           {TPS}")
    print(f"  Exit modes:    TP_ONLY, TP_SL{int(SL_FIXED)}, TP_TIME24h, TP_SIG_DIE")
    print(f"  Pairs:         {len(ALL_PAIRS)}")
    all_rows = []; t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_one_pair(pair, all_rows)
        gc.collect()
        print(f"  {pair}: {time.time()-ts:.1f}s", flush=True)

    df = pd.DataFrame(all_rows)
    out_csv = OUT / "m5_ws_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    # ── Per-config (ws_thr × tp × mode): Σ p/d, #pairs ppd>0, total n ─────
    print("\n" + "="*100)
    print("  Σ OOS p/d across all 12 pairs (sorted)")
    print("="*100)
    g = (df.groupby(["ws_thr","tp","mode"])
            .agg(sum_ppd=("ppd","sum"),
                 n_pos=("ppd", lambda x: int((x>0).sum())),
                 total_n=("n","sum"),
                 mean_wr=("wr","mean"))
            .reset_index()
            .sort_values("sum_ppd", ascending=False))
    print(f"  {'ws_thr':>7}  {'tp':>5}  {'mode':<12} {'Σ p/d':>8}  {'pos':>3}/12 {'n':>7}  {'WR%':>5}")
    for _, r in g.head(20).iterrows():
        print(f"  {r['ws_thr']:>7.1f}  {r['tp']:>5.1f}  {r['mode']:<12} "
              f"{r['sum_ppd']:>+8.1f}  {int(r['n_pos']):>3d}/12 {int(r['total_n']):>7d}  {r['mean_wr']:>5.1f}")
    print("  ...")
    print("  worst 5:")
    for _, r in g.tail(5).iterrows():
        print(f"  {r['ws_thr']:>7.1f}  {r['tp']:>5.1f}  {r['mode']:<12} "
              f"{r['sum_ppd']:>+8.1f}  {int(r['n_pos']):>3d}/12 {int(r['total_n']):>7d}  {r['mean_wr']:>5.1f}")

    # ── Best per-pair at the top portfolio config ────────────────────────
    if not g.empty:
        top = g.iloc[0]
        print("\n" + "="*100)
        print(f"  Per-pair detail at top config: ws_thr={top['ws_thr']}  TP={top['tp']}  mode={top['mode']}")
        print("="*100)
        sub = df[(df.ws_thr==top['ws_thr'])&(df.tp==top['tp'])&(df['mode']==top['mode'])]
        sub = sub.sort_values("ppd", ascending=False)
        print(f"  {'pair':<10}{'n':>5}{'ppd':>8}{'WR%':>6}{'tp%':>6}{'MDD':>7}{'Calmar':>7}{'hold':>6}")
        for _, r in sub.iterrows():
            print(f"  {r['pair']:<10}{int(r['n']):>5}{r['ppd']:>+8.1f}{r['wr']:>6.1f}"
                  f"{r['tp_pct']:>6.1f}{r['mdd']:>7.1f}{r['calmar']:>7.2f}{r['mean_hold']:>6.1f}")


if __name__ == "__main__":
    main()
