#!/usr/bin/env python3
"""
Experiment A — M5 momentum + weighted_sum as an entry FILTER on the post-shock
retrace strategy.

Hypothesis
----------
The standalone M5+WS strategy showed positive autocorrelation but its
deployable edge collapses under bounded exits (loss-deferral pattern).
However it might still add signal as a *filter* on a strategy that already
has honest exits — namely fx-retrace-live (TP=20p, SL=30p, 50-min timeout,
Markov D1 filter, +70.4 p/d OOS portfolio).

Filter rule
-----------
At each shock event detected by the retrace strategy:
  - Counter-trend direction d = -sign(vel[shock])   (planned trade direction)
  - Look up the most recent COMPLETED M5 bar's WS at that S5 timestamp.
  - SKIP the entry unless:
        |ws| >= ws_thr   AND   sign(ws) == d
  - Rationale: WS pointing in the planned counter-trend direction means
    longer-horizon momentum is already retreating — the M5+WS lead-lag
    signal should reduce false positives where the shock continues.

R6 compliance
-------------
M5 momentum + weighted_sum computed identically to fx_signals/main.py:
  5m  = (close[t]-close[t-1])  / pip /    5    weight 0.10
  15m = (close[t]-close[t-3])  / pip /   15    weight 0.15
  1h  = (close[t]-close[t-12]) / pip /   60    weight 0.20
  4h  = (close[t]-close[t-48]) / pip /  240    weight 0.20
  24h = (close[t]-close[t-288])/ pip / 1440    weight 0.25
  ws  = Σ(w*m)/Σ(w)   (weighted mean; matches compute_weighted_sum)

S5→M5 alignment: for each S5 bar at time t_s5, the most recently CLOSED M5
bar has timestamp floor(t_s5 / 5min) - if t_s5 is exactly on a 5-min
boundary, the bar that just closed is at t_s5 itself. R1: only completed
bars — we use ws of the M5 bar whose closing timestamp ≤ t_s5.

Sweep
-----
ws_thr in {0.0 (no filter / baseline), 0.5, 1.0, 1.5, 2.0}
Other retrace params fixed at the live deployment:
  thr=2.5  peak_bars=44  sd=3  tp=20p
  4 pairs: GBP_JPY, USD_JPY, EUR_JPY, AUD_JPY
  IS=70% OOS=30% — OOS reported only.

Output
------
results/retrace_ws_filter.csv (one row per pair × ws_thr)
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

# Retrace strategy params (locked at live values)
PAIRS         = ["GBP_JPY", "USD_JPY", "EUR_JPY", "AUD_JPY"]
PIP           = 0.01
THR_Z         = 2.5
PEAK_BARS_S5  = 44
STOP_DIST_PIP = 3
TP_PIP        = 20.0
HORIZON_S5    = 600
Z_WINDOW      = 6
MAD_WIN       = 2048
IS_FRAC       = 0.70

# Dashboard signal weights (M5-only subset — S5/M1 dropped, mean is invariant)
WIN_LAGS_M5   = [1, 3, 12, 48, 288]
WIN_MINUTES   = [5, 15, 60, 240, 1440]
WIN_WEIGHTS   = [0.10, 0.15, 0.20, 0.20, 0.25]

WS_THRS = [0.0, 0.5, 1.0, 1.5, 2.0]


# ── Shock detection (mirrors backtest_post_shock_retrace) ─────────────────────

def compute_shock_z(close: np.ndarray, pip: float, w: int = Z_WINDOW,
                    mad_win: int = MAD_WIN):
    n = len(close)
    vel = np.empty(n, dtype=np.float64)
    vel[:w] = 0.0
    vel[w:] = (close[w:] - close[:n - w]) / pip
    vel_s = pd.Series(vel)
    rm    = vel_s.rolling(mad_win, min_periods=50, center=False).median()
    ad    = (vel_s - rm).abs()
    rmad  = ad.rolling(mad_win, min_periods=50, center=False).median()
    z     = ((vel_s - rm) / (1.4826 * rmad.clip(lower=1e-6))).fillna(0).values
    return z.astype(np.float64), vel.astype(np.float64)


# ── M5 weighted_sum signal ────────────────────────────────────────────────────

def compute_m5_ws(m5_close: np.ndarray, pip: float) -> np.ndarray:
    """Per-M5-bar weighted_sum (weighted mean over the 5 horizons that derive
    from M5 closes). Returns array of length len(m5_close)."""
    n = len(m5_close)
    moms = np.full((len(WIN_LAGS_M5), n), np.nan, dtype=np.float64)
    for j, (lag, mins) in enumerate(zip(WIN_LAGS_M5, WIN_MINUTES)):
        if lag >= n:
            continue
        moms[j, :lag] = np.nan
        moms[j, lag:] = (m5_close[lag:] - m5_close[:-lag]) / pip / float(mins)
    weights = np.asarray(WIN_WEIGHTS, dtype=np.float64)
    valid   = ~np.isnan(moms)
    w_avail = np.where(valid, weights[:, None], 0.0).sum(axis=0)
    wv      = np.where(valid, moms * weights[:, None], 0.0).sum(axis=0)
    ws      = np.where(w_avail > 0, wv / w_avail, np.nan)
    return ws


def align_m5_ws_to_s5(s5_ts: pd.DatetimeIndex, m5_ts: pd.DatetimeIndex,
                       ws_m5: np.ndarray) -> np.ndarray:
    """For each S5 timestamp, look up the WS of the most recent M5 bar
    whose close-timestamp ≤ s5_ts. searchsorted with side='right' gives the
    insertion point; index = pos-1 is the latest completed M5 bar.
    Returns ws aligned to s5 length; NaN where no prior M5 bar.
    """
    m5_arr = m5_ts.values.astype("datetime64[ns]")
    s5_arr = s5_ts.values.astype("datetime64[ns]")
    pos = np.searchsorted(m5_arr, s5_arr, side="right") - 1
    out = np.full(len(s5_arr), np.nan, dtype=np.float64)
    mask = pos >= 0
    out[mask] = ws_m5[pos[mask]]
    return out


# ── Numba sim with WS-filter ──────────────────────────────────────────────────

@njit(cache=True)
def sim_retrace_ws(bid, ask, close, shock_flag, vel, ws_aligned, pip,
                    peak_bars, stop_pips, tp_pips, horizon, ws_thr):
    """Same direction conventions as backtest_post_shock_retrace.sim_retrace:
        d   = sign(vel[t])       SHOCK direction
        d==1 (upshock)   → SELL STOP, entry = peak_ask - sd*pip   (short)
        d==-1 (downshock)→ BUY STOP,  entry = peak_bid + sd*pip   (long)
        sd==0            → market at watch_start: short=bid, long=ask
        tp_level = entry - tp_pips * pip * d
        pnl on horizon-close:
          d=1  short: (entry - exit_ask)/pip - sp
          d=-1 long:  (exit_bid - entry)/pip - sp

    WS filter (applied BEFORE entry):
        planned counter-trend direction = -d
        require sign(ws[t]) == (-d)  and  |ws[t]| >= ws_thr.
    """
    n      = len(close)
    pb_int = int(peak_bars)
    max_ev = n // 10
    filled  = np.zeros(max_ev, dtype=np.int8)
    tp_hit  = np.zeros(max_ev, dtype=np.int8)
    pnl_out = np.zeros(max_ev, dtype=np.float64)
    dir_out = np.zeros(max_ev, dtype=np.int8)
    skipped_total = 0
    skipped_ws    = 0
    ev_count = 0
    cooldown = 0

    for t in range(Z_WINDOW, n - pb_int - int(horizon) - 2):
        if cooldown > 0:
            cooldown -= 1
            continue
        if shock_flag[t] != 1:
            continue

        d = 1 if vel[t] > 0 else -1     # SHOCK direction (matches original)
        planned = -d                     # planned counter-trend direction

        # ── WS filter ────────────────────────────────────────────────────────
        if ws_thr > 0.0:
            ws_val = ws_aligned[t]
            if np.isnan(ws_val):
                skipped_total += 1
                skipped_ws    += 1
                continue
            if abs(ws_val) < ws_thr:
                skipped_total += 1
                skipped_ws    += 1
                continue
            if (ws_val > 0 and planned != 1) or (ws_val < 0 and planned != -1):
                skipped_total += 1
                skipped_ws    += 1
                continue

        # ── Find peak in [t, t+pb_int] ──────────────────────────────────────
        peak_ask = ask[t]
        peak_bid = bid[t]
        for k in range(1, pb_int + 1):
            j = t + k
            if ask[j] > peak_ask: peak_ask = ask[j]
            if bid[j] < peak_bid: peak_bid = bid[j]

        sp = (ask[t] - bid[t]) / pip
        watch_start = t + pb_int + 1
        watch_end   = t + pb_int + int(horizon)
        if watch_start >= n or watch_end >= n:
            continue

        fld = 0; tp = 0; fill_price = 0.0; pnl = 0.0; tp_level = 0.0; entry = 0.0

        if stop_pips == 0.0:
            # Market entry at watch_start (honest live fill)
            fld = 1
            if d == 1:                              # upshock → short
                fill_price = bid[watch_start]       # sell at bid
            else:                                   # downshock → long
                fill_price = ask[watch_start]       # buy at ask
            tp_level = fill_price - tp_pips * pip * d
            # same-bar TP check
            if d == 1 and bid[watch_start] <= tp_level:
                tp = 1; pnl = tp_pips - sp
            elif d == -1 and ask[watch_start] >= tp_level:
                tp = 1; pnl = tp_pips - sp
            loop_start = watch_start + 1
        else:
            # Stop entry
            if d == 1:
                entry = peak_ask - stop_pips * pip
            else:
                entry = peak_bid + stop_pips * pip
            tp_level = entry - tp_pips * pip * d
            loop_start = watch_start

        for j in range(loop_start, min(watch_end + 1, n - 1)):
            lo = bid[j]; hi = ask[j]

            if stop_pips > 0.0 and fld == 0:
                if d == 1 and lo <= entry:          # short stop fills when bid drops
                    fld = 1; fill_price = entry
                    if lo <= tp_level:               # short TP same bar
                        tp = 1; pnl = tp_pips - sp
                elif d == -1 and hi >= entry:        # long stop fills when ask rises
                    fld = 1; fill_price = entry
                    if hi >= tp_level:               # long TP same bar
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
            # Match backtest_post_shock_retrace.sim_retrace exactly: uses bid
            # for short close and ask for long close. Spread cost (sp) was
            # already deducted on entry; using the favorable side here avoids
            # double-charging spread. Same convention is the official baseline.
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
            dir_out[ev_count] = planned
            ev_count += 1

        cooldown = (pb_int + int(horizon)) // 2

    return (filled[:ev_count], tp_hit[:ev_count],
            pnl_out[:ev_count], dir_out[:ev_count],
            skipped_total, skipped_ws)


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def warmup_jit():
    n = 3000
    b = np.ones(n) * 214.0; a = b + 0.03; c = b + 0.015
    v = np.zeros(n); v[100]=1.2; v[500]=-0.8
    sf = np.zeros(n, dtype=np.int8); sf[100]=1; sf[500]=1
    ws = np.zeros(n)
    sim_retrace_ws(b, a, c, sf, v, ws, PIP, PEAK_BARS_S5, STOP_DIST_PIP, TP_PIP, 120, 0.0)
    sim_retrace_ws(b, a, c, sf, v, ws, PIP, PEAK_BARS_S5, STOP_DIST_PIP, TP_PIP, 120, 1.5)


def run_pair(pair, all_rows):
    t0 = time.time()
    s5  = (pd.read_parquet(S5_DIR / f"{pair}_S5_BA.parquet")
             .set_index("timestamp").sort_index())
    s5 = s5.astype({c:"float64" for c in s5.select_dtypes("float32").columns})
    m5  = (pd.read_parquet(M5_DIR / f"{pair}_M5_BA.parquet")
             .set_index("timestamp").sort_index())
    m5 = m5.astype({c:"float64" for c in m5.select_dtypes("float32").columns})

    bid   = s5["bid_c"].values
    ask   = s5["ask_c"].values
    close = s5["close"].values
    z, vel = compute_shock_z(close, PIP)
    shock_flag = (np.abs(z) >= THR_Z).astype(np.int8)

    ws_m5 = compute_m5_ws(m5["close"].values, PIP)
    ws_aligned = align_m5_ws_to_s5(s5.index, m5.index, ws_m5)

    n_is = int(len(s5) * IS_FRAC)
    sf_o  = shock_flag.copy(); sf_o[:n_is] = 0   # zero IS shocks → OOS-only
    bid_o = bid; ask_o = ask; close_o = close; vel_o = vel
    ws_o  = ws_aligned
    oos_days = (len(s5) - n_is) / (288.0 * 12.0)   # S5 bars/day = 17280

    for ws_thr in WS_THRS:
        fld, tph, pnl, drs, skipped_total, skipped_ws = sim_retrace_ws(
            bid_o, ask_o, close_o, sf_o, vel_o, ws_o, PIP,
            PEAK_BARS_S5, STOP_DIST_PIP, TP_PIP, HORIZON_S5, float(ws_thr))
        n_ev   = len(fld)
        n_fill = int(fld.sum())
        n_tp   = int(tph.sum())
        # Net pnl only counts filled trades; horizon-close pnls are
        # already in pnl[]. n_fill is the trade count.
        net    = float(pnl.sum())
        wr     = (pnl[fld==1] > 0).sum() / max(n_fill, 1) * 100
        ppd    = net / oos_days
        mdd    = max_dd(pnl[fld==1])
        all_rows.append(dict(
            pair=pair, ws_thr=ws_thr, n_events=n_ev, n_filled=n_fill,
            n_skipped=skipped_total, n_tp=n_tp,
            wr=round(wr,1), tp_pct=round(n_tp/max(n_fill,1)*100,1),
            net_pnl=round(net,1), ppd=round(ppd,2),
            mdd=round(mdd,1), days=round(oos_days,1)))

    del s5, m5
    gc.collect()
    print(f"  {pair}: {time.time()-t0:.1f}s", flush=True)


def main():
    warmup_jit()
    print(f"Experiment A — retrace ⊗ M5+WS filter (OOS-only)")
    print(f"  pairs={PAIRS}  ws_thrs={WS_THRS}")
    print(f"  fixed: thr_z={THR_Z}  peak_bars={PEAK_BARS_S5}  sd={STOP_DIST_PIP}p  tp={TP_PIP}p")
    rows = []
    t0 = time.time()
    for pair in PAIRS:
        run_pair(pair, rows)
    df = pd.DataFrame(rows)
    out_csv = OUT / "retrace_ws_filter.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    print("\n" + "="*100)
    print("  Per-pair: how the filter changes things (vs ws_thr=0 baseline)")
    print("="*100)
    for pair in PAIRS:
        print(f"\n  --- {pair} ---")
        sub = df[df.pair == pair].sort_values("ws_thr")
        base = sub[sub.ws_thr == 0.0].iloc[0]
        print(f"  {'ws_thr':>7}  {'n_ev':>6}  {'n_fill':>6}  {'WR%':>5}  "
              f"{'net':>7}  {'ppd':>6}  {'MDD':>6}  {'Δppd':>7}")
        for _, r in sub.iterrows():
            d_ppd = r['ppd'] - base['ppd']
            print(f"  {r['ws_thr']:>7.1f}  {int(r['n_events']):>6d}  {int(r['n_filled']):>6d}  "
                  f"{r['wr']:>5.1f}  {r['net_pnl']:>+7.1f}  {r['ppd']:>+6.2f}  "
                  f"{r['mdd']:>6.1f}  {d_ppd:>+7.2f}")

    print("\n" + "="*100)
    print("  Portfolio totals across 4 pairs by ws_thr")
    print("="*100)
    g = (df.groupby("ws_thr")
            .agg(sum_ppd=("ppd","sum"), tot_fill=("n_filled","sum"),
                 mean_wr=("wr","mean"), tot_skipped=("n_skipped","sum"))
            .reset_index())
    print(f"  {'ws_thr':>7}  {'Σ ppd':>7}  {'Σ fills':>8}  {'mean WR%':>9}  {'Σ skipped':>10}")
    for _, r in g.iterrows():
        print(f"  {r['ws_thr']:>7.1f}  {r['sum_ppd']:>+7.2f}  {int(r['tot_fill']):>8d}  "
              f"{r['mean_wr']:>9.1f}  {int(r['tot_skipped']):>10d}")


if __name__ == "__main__":
    main()
