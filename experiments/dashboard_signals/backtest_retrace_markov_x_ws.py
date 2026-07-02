#!/usr/bin/env python3
"""
Markov × WS 2×2 factorial on the retrace strategy.

Cells
-----
   ws_off, markov_off : baseline (no filter)
   ws_off, markov_on  : Markov-only  (mirrors current live deployment)
   ws_on,  markov_off : WS-only
   ws_on,  markov_on  : BOTH stacked (the proposed deployment)

Decomposes the +6.66 p/d backtest delta we saw for WS-only into:
  • Markov independent contribution
  • WS independent contribution
  • Overlap (joint already filtered by either gate)

Constants locked at the live service values
-------------------------------------------
THR_Z=2.5  PEAK_BARS=44  STOP_DIST=3p  TP=20p  HORIZON=600
MARKOV_MW=10  MARKOV_MT=0.002  MARKOV_SIG_THR=0.20  MIN_PRIME=30
WS gate: weighted-mean across 5 M5-derived windows, threshold |ws| ≥ 0.5
         AND sign(ws) == counter-trend direction.

Pairs: GBP_JPY, USD_JPY, EUR_JPY, AUD_JPY (the live 4-pair set).
Split: IS 70% / OOS 30%; OOS-only reported.
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

# Retrace parameters (locked at live)
THR_Z         = 2.5
PEAK_BARS_S5  = 44
STOP_DIST_PIP = 3.0
TP_PIP        = 20.0
HORIZON_S5    = 600
Z_WIN         = 6
MAD_WIN       = 2048
IS_FRAC       = 0.70

# Markov filter
MARKOV_MW       = 10
MARKOV_MT       = 0.002
MARKOV_SIG_THR  = 0.20
MARKOV_MIN_PRIME = 30
BULL, SIDE, BEAR = 0, 1, 2

# WS filter
WIN_LAGS_M5 = [1, 3, 12, 48, 288]
WIN_MINUTES = [5, 15, 60, 240, 1440]
WIN_WEIGHTS = [0.10, 0.15, 0.20, 0.20, 0.25]
WS_THR      = 0.5


# ── Shock detection ──────────────────────────────────────────────────────────

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


# ── M5 weighted_sum ──────────────────────────────────────────────────────────

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


# ── Markov D1 signals (mirrors live MarkovFilter._compute) ───────────────────

def build_markov_signals(close_s5: np.ndarray, ts_s5: pd.DatetimeIndex) -> dict:
    """Per-D1-date Markov signal. signal usable at OPEN of day D = key 'D'."""
    s5 = pd.Series(close_s5, index=ts_s5)
    d1 = s5.resample("1D").last().dropna()
    lr = np.log(d1 / d1.shift(1)).dropna()
    roll = lr.rolling(MARKOV_MW).sum()
    def _state(r):
        if np.isnan(r):       return np.nan
        if r >  MARKOV_MT:    return BULL
        if r < -MARKOV_MT:    return BEAR
        return SIDE
    states = roll.map(_state).dropna().astype(int)
    T = np.zeros((3, 3), dtype=np.float64)
    signals: dict = {}
    for i in range(len(states) - 1):
        day_i = states.index[i].date()
        s     = int(states.iloc[i])
        row_sum = T[s].sum()
        if row_sum >= MARKOV_MIN_PRIME:
            sig = (T[s, BULL] - T[s, BEAR]) / row_sum
            signals[day_i] = float(sig)
        else:
            signals[day_i] = 0.0
        T[s, int(states.iloc[i + 1])] += 1.0
    return signals


def map_markov_to_s5(s5_ts: pd.DatetimeIndex, daily_signals: dict) -> np.ndarray:
    """Causal: signal at S5 timestamp t = signal computed on day(t-1)."""
    if not daily_signals:
        return np.zeros(len(s5_ts), dtype=np.float64)
    sorted_days = sorted(daily_signals.keys())
    prev = {}
    for i, d in enumerate(sorted_days):
        prev[d] = daily_signals[sorted_days[i - 1]] if i > 0 else 0.0
    out = np.zeros(len(s5_ts), dtype=np.float64)
    for i, ts in enumerate(s5_ts):
        d = ts.date()
        out[i] = prev.get(d, 0.0)
    return out


# ── Numba simulator (Markov + WS gates, either may be disabled) ──────────────

@njit(cache=True)
def sim(bid, ask, close, shock_flag, vel,
        markov_sig, ws_aligned, pip,
        peak_bars, stop_pips, tp_pips, horizon,
        markov_thr, ws_thr):
    """
    markov_thr < -900 → Markov off; else require shock_dir * markov_sig > markov_thr
    ws_thr     <  0   → WS off;     else require |ws|≥ws_thr AND sign(ws)==planned_dir
    """
    n      = len(close)
    pb     = int(peak_bars)
    hor    = int(horizon)
    max_ev = n // 10
    filled  = np.zeros(max_ev, dtype=np.int8)
    tp_hit  = np.zeros(max_ev, dtype=np.int8)
    pnl_out = np.zeros(max_ev, dtype=np.float64)
    dir_out = np.zeros(max_ev, dtype=np.int8)
    skipped_markov = 0
    skipped_ws     = 0
    ev_count = 0
    cooldown = 0

    for t in range(Z_WIN, n - pb - hor - 2):
        if cooldown > 0:
            cooldown -= 1
            continue
        if shock_flag[t] != 1:
            continue

        d = 1 if vel[t] > 0 else -1
        planned = -d   # counter-trend direction

        # ── Markov filter ────────────────────────────────────────────────────
        if markov_thr > -900.0:
            ms = markov_sig[t]
            if float(d) * ms <= markov_thr:
                skipped_markov += 1
                continue

        # ── WS filter ────────────────────────────────────────────────────────
        if ws_thr >= 0.0:
            w = ws_aligned[t]
            if np.isnan(w):
                skipped_ws += 1; continue
            if abs(w) < ws_thr:
                skipped_ws += 1; continue
            if (w > 0 and planned != 1) or (w < 0 and planned != -1):
                skipped_ws += 1; continue

        # ── Find peak window ────────────────────────────────────────────────
        peak_ask = ask[t]; peak_bid = bid[t]
        for k in range(1, pb + 1):
            j = t + k
            if ask[j] > peak_ask: peak_ask = ask[j]
            if bid[j] < peak_bid: peak_bid = bid[j]

        sp = (ask[t] - bid[t]) / pip
        watch_start = t + pb + 1
        watch_end   = t + pb + hor
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
            dir_out[ev_count] = planned
            ev_count += 1

        cooldown = (pb + hor) // 2

    return (filled[:ev_count], tp_hit[:ev_count],
            pnl_out[:ev_count], dir_out[:ev_count],
            skipped_markov, skipped_ws)


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def warmup_jit():
    n = 3000
    b = np.ones(n) * 214.0; a = b + 0.03; c = b + 0.015
    v = np.zeros(n); v[100]=1.2; v[500]=-0.8
    sf = np.zeros(n, dtype=np.int8); sf[100]=1; sf[500]=1
    ms = np.zeros(n); ws = np.zeros(n)
    sim(b, a, c, sf, v, ms, ws, PIP, PEAK_BARS_S5, STOP_DIST_PIP, TP_PIP, 120, -999.0, -1.0)
    sim(b, a, c, sf, v, ms, ws, PIP, PEAK_BARS_S5, STOP_DIST_PIP, TP_PIP, 120, 0.20, 0.5)


CELLS = [
    ("baseline",   -999.0, -1.0),
    ("markov",      0.20,  -1.0),
    ("ws",         -999.0,  0.5),
    ("markov+ws",   0.20,   0.5),
]


def run_pair(pair, all_rows):
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

    n_is = int(len(s5) * IS_FRAC)
    sf_o = shock_flag.copy(); sf_o[:n_is] = 0   # OOS-only
    bid_o, ask_o, close_o, vel_o = bid, ask, close, vel
    ms_o, ws_o = markov_aligned, ws_aligned
    oos_days = (len(s5) - n_is) / 17280.0

    for cell_name, mthr, wthr in CELLS:
        fld, tph, pnl, drs, sk_m, sk_w = sim(
            bid_o, ask_o, close_o, sf_o, vel_o, ms_o, ws_o, PIP,
            PEAK_BARS_S5, STOP_DIST_PIP, TP_PIP, HORIZON_S5,
            float(mthr), float(wthr))
        n_ev   = len(fld)
        n_fill = int(fld.sum())
        net    = float(pnl.sum())
        wr     = (pnl[fld==1] > 0).sum() / max(n_fill, 1) * 100
        ppd    = net / oos_days
        mdd    = max_dd(pnl[fld==1])
        all_rows.append(dict(
            pair=pair, cell=cell_name,
            n_events=n_ev, n_filled=n_fill,
            skipped_markov=sk_m, skipped_ws=sk_w,
            wr=round(wr,1), net=round(net,1), ppd=round(ppd,2),
            mdd=round(mdd,1), days=round(oos_days,1)))

    del s5, m5
    gc.collect()
    print(f"  {pair}: {time.time()-t0:.1f}s", flush=True)


def main():
    warmup_jit()
    print("Markov × WS 2×2 factorial on retrace strategy")
    print(f"  pairs={PAIRS}")
    print(f"  Markov: mw={MARKOV_MW} mt={MARKOV_MT} sig_thr={MARKOV_SIG_THR}")
    print(f"  WS: ws_thr={WS_THR}, direction-match required")
    print(f"  Fixed: thr_z={THR_Z} peak_bars={PEAK_BARS_S5} sd={STOP_DIST_PIP}p tp={TP_PIP}p horizon={HORIZON_S5}")
    rows = []
    t0 = time.time()
    for pair in PAIRS:
        run_pair(pair, rows)
    df = pd.DataFrame(rows)
    out_csv = OUT / "retrace_markov_x_ws.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows → {out_csv}  ({time.time()-t0:.1f}s)")

    # ── 2x2 summary ──────────────────────────────────────────────────────────
    print("\n" + "="*110)
    print("  Portfolio totals across 4 pairs by cell")
    print("="*110)
    g = (df.groupby("cell")
            .agg(sum_ppd=("ppd","sum"),
                 sum_net=("net","sum"),
                 tot_fill=("n_filled","sum"),
                 tot_skip_m=("skipped_markov","sum"),
                 tot_skip_w=("skipped_ws","sum"),
                 mean_wr=("wr","mean"))
            .reindex([c[0] for c in CELLS])
            .reset_index())
    print(f"  {'cell':<14}  {'Σ p/d':>8}  {'Σ pips':>9}  {'Σ fills':>8}  "
          f"{'skip_M':>8}  {'skip_W':>8}  {'mean WR%':>9}")
    for _, r in g.iterrows():
        print(f"  {r['cell']:<14}  {r['sum_ppd']:>+8.2f}  {r['sum_net']:>+9.1f}  "
              f"{int(r['tot_fill']):>8d}  {int(r['tot_skip_m']):>8d}  "
              f"{int(r['tot_skip_w']):>8d}  {r['mean_wr']:>9.1f}")

    # ── Decomposition ────────────────────────────────────────────────────────
    base = float(g[g.cell=='baseline']['sum_ppd'].iloc[0])
    m    = float(g[g.cell=='markov']['sum_ppd'].iloc[0])
    w    = float(g[g.cell=='ws']['sum_ppd'].iloc[0])
    mw   = float(g[g.cell=='markov+ws']['sum_ppd'].iloc[0])
    print("\n" + "="*110)
    print("  Contribution decomposition (vs baseline)")
    print("="*110)
    print(f"  baseline                       {base:>+7.2f} p/d")
    print(f"  +Markov   (markov_only - base) {m - base:>+7.2f} p/d   ← Markov independent")
    print(f"  +WS       (ws_only     - base) {w - base:>+7.2f} p/d   ← WS independent")
    print(f"  +both     (markov+ws   - base) {mw - base:>+7.2f} p/d   ← stacked lift")
    overlap = (m - base) + (w - base) - (mw - base)
    print(f"  overlap   (sum-of-indiv − stacked) {overlap:>+7.2f} p/d   ← double-counted lift")
    print(f"\n  Live runs the Markov cell. Adding WS on top is expected to add:")
    print(f"     (markov+ws) − (markov) = {mw - m:>+7.2f} p/d  (incremental over current live)")

    # ── Per-pair table for all 4 cells ──────────────────────────────────────
    print("\n" + "="*110)
    print("  Per-pair p/d by cell")
    print("="*110)
    piv = df.pivot(index="pair", columns="cell", values="ppd")[
        ["baseline","markov","ws","markov+ws"]]
    print(piv.to_string(float_format=lambda x: f"{x:+7.2f}"))
    print("\n  Per-pair n_filled by cell")
    pivn = df.pivot(index="pair", columns="cell", values="n_filled")[
        ["baseline","markov","ws","markov+ws"]]
    print(pivn.to_string())


if __name__ == "__main__":
    main()
