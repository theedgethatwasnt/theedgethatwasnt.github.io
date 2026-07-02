#!/usr/bin/env python3
"""
Shock Straddle Experiment
==========================
Hypothesis (user insight, 2026-05-25):
  When a momentum shock is detected, price oscillates through both
  directions before settling. Placing symmetric stop-entry orders
  (BUY STOP above + SELL STOP below) at the moment of shock onset
  lets us capture the shock wave AND the rebound oscillation —
  on separate accounts, both long and short.

Design:
  1. Detect shock events: 30s velocity z-score > threshold on S5 data
  2. At each shock, place:
       BUY  STOP  @ ask + N*pip  (fills if price goes up N from current)
       SELL STOP  @ bid - N*pip  (fills if price goes down N from current)
  3. Each side has TP = M pips from fill price
  4. Track the next HORIZON bars (default 600 bars = 50 min @ S5)
  5. Record: long_filled, long_tp, short_filled, short_tp, dual_win

Sweep: N (stop distance) × M (TP) × threshold
Reports: fill_rate, tp_rate|fill, dual_win_rate, net_pips/event, p/d

S5 data available: GBP_JPY, USD_JPY, EUR_JPY, AUD_JPY (BA format)
"""
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit, prange
import warnings; warnings.filterwarnings("ignore")

PROJECT  = Path(__file__).resolve().parents[3]
S5_DIR   = PROJECT / "data" / "s5_ba"
RESULTS  = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PAIRS = ["GBP_JPY", "USD_JPY", "EUR_JPY", "AUD_JPY"]
PIP   = {"GBP_JPY":0.01,"USD_JPY":0.01,"EUR_JPY":0.01,"AUD_JPY":0.01}

# Sweep parameters
STOP_DISTS = [2, 3, 5, 7, 10]      # N pips: how far stop is placed from current price
TP_DISTS   = [5, 8, 10, 15, 20]    # M pips: profit target from fill
THRESHOLDS = [2.0, 2.5, 3.0]       # shock z-score threshold
HORIZON    = 600                    # S5 bars to watch after shock (~50 min)
Z_WINDOW   = 6                      # bars for shock velocity = 30s
MAD_WIN    = 2048                   # bars for rolling MAD normalisation
IS_FRAC    = 0.70                   # use OOS only


# ── Shock detection ────────────────────────────────────────────────────────────

def compute_shock_z(close: np.ndarray, pip: float, w: int = 6,
                    mad_win: int = 2048) -> np.ndarray:
    """
    Velocity = (close[t] - close[t-w]) / pip  (pips per w*5s window)
    Z-score via rolling MAD (robust to outliers).
    """
    n   = len(close)
    vel = np.empty(n, dtype=np.float32)
    vel[:w] = 0.0
    vel[w:] = (close[w:] - close[:n-w]) / pip

    # rolling median + MAD
    vel_s  = pd.Series(vel)
    rm     = vel_s.rolling(mad_win, min_periods=50, center=False).median()
    ad     = (vel_s - rm).abs()
    rmad   = ad.rolling(mad_win, min_periods=50, center=False).median()
    z      = ((vel_s - rm) / (1.4826 * rmad.clip(lower=1e-6))).fillna(0).values
    return z.astype(np.float32)


# ── Core straddle simulation (Numba) ──────────────────────────────────────────

@njit
def sim_straddle(bid, ask, close, shock_flag, pip,
                 stop_pips, tp_pips, horizon):
    """
    For every shock event:
      - BUY STOP  at ask[t] + stop_pips*pip
      - SELL STOP at bid[t] - stop_pips*pip
    Each side has TP = tp_pips from fill.
    No stop-loss: if TP not hit within horizon, close at market bid/ask.

    P&L per side (in pips, net of spread):
      Filled + TP hit   → +tp_pips - spread
      Filled + TP miss  → (close_price - fill_price) / pip * dir - spread
      Not filled        → 0
    """
    n = len(close)
    max_ev = n // 10
    long_fill  = np.zeros(max_ev, dtype=np.int8)
    long_tp    = np.zeros(max_ev, dtype=np.int8)
    short_fill = np.zeros(max_ev, dtype=np.int8)
    short_tp   = np.zeros(max_ev, dtype=np.int8)
    long_pnl   = np.zeros(max_ev, dtype=np.float32)
    short_pnl  = np.zeros(max_ev, dtype=np.float32)
    ev_count   = 0
    cooldown   = 0

    for t in range(Z_WINDOW, n - horizon - 1):
        if cooldown > 0:
            cooldown -= 1
            continue
        if shock_flag[t] != 1:
            continue

        sp = (ask[t] - bid[t]) / pip

        long_entry     = ask[t] + stop_pips * pip
        short_entry    = bid[t] - stop_pips * pip
        long_tp_price  = long_entry  + tp_pips * pip
        short_tp_price = short_entry - tp_pips * pip

        lf = 0; lt = 0; sf = 0; st_ = 0
        lpnl = 0.0; spnl = 0.0
        l_fill_price = 0.0; s_fill_price = 0.0

        for k in range(1, horizon + 1):
            j = t + k
            hi = ask[j]    # actual ask high approximation
            lo = bid[j]    # actual bid low approximation

            if lf == 0 and hi >= long_entry:
                lf = 1
                l_fill_price = long_entry   # filled at stop price
                if hi >= long_tp_price:
                    lt = 1
                    lpnl = tp_pips - sp

            if lf == 1 and lt == 0 and hi >= long_tp_price:
                lt = 1
                lpnl = tp_pips - sp

            if sf == 0 and lo <= short_entry:
                sf = 1
                s_fill_price = short_entry
                if lo <= short_tp_price:
                    st_ = 1
                    spnl = tp_pips - sp

            if sf == 1 and st_ == 0 and lo <= short_tp_price:
                st_ = 1
                spnl = tp_pips - sp

            if lf == 1 and lt == 1 and sf == 1 and st_ == 1:
                break

        # horizon expired — close any unfilled TP at market
        end = t + horizon
        if lf == 1 and lt == 0:
            # exit long at bid[end] (market exit for long position)
            exit_px = bid[end]
            lpnl = (exit_px - l_fill_price) / pip - sp
        if sf == 1 and st_ == 0:
            # exit short at ask[end]
            exit_px = ask[end]
            spnl = (s_fill_price - exit_px) / pip - sp

        if ev_count < max_ev:
            long_fill[ev_count]  = lf
            long_tp[ev_count]    = lt
            short_fill[ev_count] = sf
            short_tp[ev_count]   = st_
            long_pnl[ev_count]   = lpnl
            short_pnl[ev_count]  = spnl
            ev_count += 1

        cooldown = horizon // 2

    return (long_fill[:ev_count], long_tp[:ev_count],
            short_fill[:ev_count], short_tp[:ev_count],
            long_pnl[:ev_count], short_pnl[:ev_count])


# ── Load, run, report ──────────────────────────────────────────────────────────

def run_pair(pair: str, pip: float):
    path = S5_DIR / f"{pair}_S5_BA.parquet"
    print(f"\nLoading {pair} …")
    df   = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    df   = df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
    n_is = int(len(df) * IS_FRAC)
    df   = df.iloc[n_is:].reset_index(drop=True)     # OOS only
    print(f"  OOS bars: {len(df):,}  ({len(df)/17280:.0f} trading days @ S5)")

    close = df["close"].values.astype(np.float64)
    bid   = df["bid_c"].values.astype(np.float64)
    ask   = df["ask_c"].values.astype(np.float64)

    z = compute_shock_z(close, pip, w=Z_WINDOW, mad_win=MAD_WIN)

    results = []
    for thr in THRESHOLDS:
        shock_flag = (np.abs(z) > thr).astype(np.int8)
        n_events = shock_flag.sum()
        print(f"  thr={thr}: {n_events} shock events "
              f"({n_events/len(df)*100:.2f}% of bars)")

        for sd in STOP_DISTS:
            for tp in TP_DISTS:
                lf, lt, sf, st_, lpnl, spnl = sim_straddle(
                    bid, ask, close, shock_flag, pip,
                    float(sd), float(tp), HORIZON)
                n = len(lf)
                if n == 0:
                    continue

                n_any_fill   = ((lf+sf) > 0).sum()
                n_long_tp    = lt.sum()
                n_short_tp   = st_.sum()
                n_dual_tp    = ((lt == 1) & (st_ == 1)).sum()
                fill_rate    = n_any_fill / n * 100
                long_tp_rate = n_long_tp / (lf.sum() + 1e-9) * 100
                short_tp_rate= n_short_tp / (sf.sum() + 1e-9) * 100
                dual_rate    = n_dual_tp / n * 100
                total_pnl    = lpnl.sum() + spnl.sum()
                # trading days in OOS
                oos_days     = len(df) / 17280
                ppd          = total_pnl / oos_days
                ev_per_day   = n / oos_days

                results.append(dict(
                    pair=pair, thr=thr, sd=sd, tp=tp, n_events=n,
                    fill_rate=fill_rate,
                    long_tp_pct=long_tp_rate, short_tp_pct=short_tp_rate,
                    dual_tp_pct=dual_rate, total_pnl=total_pnl,
                    ppd=ppd, ev_per_day=ev_per_day
                ))

    del df, close, bid, ask, z; gc.collect()
    return results


# ── Warmup ─────────────────────────────────────────────────────────────────────

print("Warming up Numba …")
_b = np.ones(2000, dtype=np.float64) * 214.0
_a = _b + 0.03
_c = _b + 0.015
_sf= np.zeros(2000, dtype=np.int8); _sf[100]=1; _sf[500]=1; _sf[1000]=1
sim_straddle(_b, _a, _c, _sf, 0.01, 3.0, 10.0, 120)
print("Done.\n")

all_results = []
for pair in PAIRS:
    all_results.extend(run_pair(pair, PIP[pair]))

df_res = pd.DataFrame(all_results)
df_res.to_csv(RESULTS / "shock_straddle_results.csv", index=False)

# ── Portfolio summary ──────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("PORTFOLIO SUMMARY (all pairs combined)")
print(f"{'='*80}")

agg = (df_res.groupby(["thr","sd","tp"])
       .agg(n_events=("n_events","sum"),
            ev_per_day=("ev_per_day","sum"),
            ppd=("ppd","sum"),
            fill_rate=("fill_rate","mean"),
            long_tp_pct=("long_tp_pct","mean"),
            short_tp_pct=("short_tp_pct","mean"),
            dual_tp_pct=("dual_tp_pct","mean"))
       .reset_index())

for thr, grp in agg.groupby("thr"):
    print(f"\n  Threshold z>{thr}:")
    print(f"  {'Stop':>4}  {'TP':>4}  {'ev/d':>5}  {'fill%':>5}  "
          f"{'L_tp%':>6}  {'S_tp%':>6}  {'dual%':>5}  {'ppd':>8}")
    print(f"  {'-'*60}")
    best_row = grp.loc[grp["ppd"].idxmax()]
    for _, r in grp.sort_values("ppd", ascending=False).iterrows():
        marker = " ◄" if r["ppd"] == best_row["ppd"] else ""
        print(f"  {r['sd']:>4.0f}p  {r['tp']:>4.0f}p  "
              f"{r['ev_per_day']:>5.1f}  {r['fill_rate']:>5.1f}%  "
              f"{r['long_tp_pct']:>6.1f}%  {r['short_tp_pct']:>6.1f}%  "
              f"{r['dual_tp_pct']:>5.1f}%  {r['ppd']:>+8.1f}p{marker}")

print(f"\n{'='*80}")
print("TOP 10 CONFIGS BY p/d (portfolio)")
print(f"{'='*80}")
top = agg.nlargest(10, "ppd")
print(f"  {'thr':>4}  {'Stop':>4}  {'TP':>4}  {'ev/d':>5}  "
      f"{'fill%':>5}  {'dual%':>5}  {'ppd':>8}")
for _, r in top.iterrows():
    print(f"  {r['thr']:>4.1f}  {r['sd']:>4.0f}p  {r['tp']:>4.0f}p  "
          f"{r['ev_per_day']:>5.1f}  {r['fill_rate']:>5.1f}%  "
          f"{r['dual_tp_pct']:>5.1f}%  {r['ppd']:>+8.1f}p")
