#!/usr/bin/env python3
"""
MTF state-agreement baseline — stay long while M5 momentum > 0 AND H1 > 0;
stay short while both < 0; flat otherwise.

This is the simplest possible "MTF agreement" strategy. State-based, not
event-based. No fixed TP, no fixed SL — position flips with the agreement.

Per M5 bar:
   m5_mom = close[t] - close[t-1]      (just the sign matters)
   h1_mom = close[t] - close[t-12]     (60 minutes back)
   state  = +1 if m5>0 AND h1>0
            -1 if m5<0 AND h1<0
             0 otherwise (disagreement)

When state changes:
   if prior state != 0:  close at bid (long) / ask (short)
   if new state != 0:    open at ask (long) / bid (short)
   spread paid on every transition into/out of a non-flat state.

All 12 pairs, OOS-only (last 30%). Report:
   trade count, Σ pips, Σ p/d, mean hold bars, WR, max single-trade loss/win.
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

IS_FRAC = 0.70

# H1 = 12 M5 bars; M5 = 1 M5 bar. The two windows the user pointed at.
H1_LAG = 12
M5_LAG = 1


def pip_sz(p): return 0.01 if p in JPY else 0.0001


@njit(cache=True)
def sim_mtf_state(close, bid, ask, sp, pip, sp_gate,
                  m5_lag, h1_lag):
    """State-based: position = sign agreement of (M5 mom, H1 mom)."""
    n = len(close)
    pnl_arr   = np.empty(n, dtype=np.float64)
    hold_arr  = np.empty(n, dtype=np.int32)
    dir_arr   = np.empty(n, dtype=np.int8)
    count = 0

    state_prev = 0
    entry_px   = 0.0
    entry_i    = 0

    start = max(h1_lag, m5_lag) + 1
    for i in range(start, n):
        # Determine state at bar i
        m5_dir = 0
        h1_dir = 0
        m5_diff = close[i] - close[i - m5_lag]
        h1_diff = close[i] - close[i - h1_lag]
        if m5_diff > 0.0: m5_dir =  1
        elif m5_diff < 0.0: m5_dir = -1
        if h1_diff > 0.0: h1_dir =  1
        elif h1_diff < 0.0: h1_dir = -1

        s = 0
        if m5_dir == 1 and h1_dir == 1:
            s = 1
        elif m5_dir == -1 and h1_dir == -1:
            s = -1

        if s != state_prev:
            # Close prior position
            if state_prev != 0:
                exit_px = bid[i] if state_prev == 1 else ask[i]
                trade_pnl = (exit_px - entry_px) / pip * state_prev - sp[i]
                pnl_arr[count]  = trade_pnl
                hold_arr[count] = i - entry_i
                dir_arr[count]  = state_prev
                count += 1
            # Open new (only if spread acceptable; flat otherwise this bar)
            if s != 0 and sp[i] <= sp_gate:
                entry_px = ask[i] if s == 1 else bid[i]
                entry_i  = i
            else:
                s = 0  # don't open under high spread
            state_prev = s

    # Close any open at last bar
    if state_prev != 0:
        i = n - 1
        exit_px = bid[i] if state_prev == 1 else ask[i]
        trade_pnl = (exit_px - entry_px) / pip * state_prev - sp[i]
        pnl_arr[count]  = trade_pnl
        hold_arr[count] = i - entry_i
        dir_arr[count]  = state_prev
        count += 1

    return pnl_arr[:count], hold_arr[:count], dir_arr[:count]


def max_dd(pnl):
    if len(pnl) == 0: return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(eq) - eq).max())


def warmup_jit():
    n = 200
    c = np.linspace(1.0, 1.05, n).astype(np.float64)
    b = c - 0.0002; a = c + 0.0002
    sp = np.ones(n)
    sim_mtf_state(c, b, a, sp, 0.0001, 2.0, 1, 12)


def run_pair(pair, all_rows):
    df_m5 = (pd.read_parquet(DATA / f"{pair}_M5_BA.parquet")
                .set_index("timestamp").sort_index())
    df_m5 = df_m5.astype({c:"float64" for c in df_m5.select_dtypes("float32").columns})
    pip = pip_sz(pair); sg = SP_GATES[pair]

    close = df_m5["close"].values.astype(np.float64)
    bid   = df_m5["bid_c"].values.astype(np.float64)
    ask   = df_m5["ask_c"].values.astype(np.float64)
    sp    = ((ask - bid) / pip).astype(np.float64)
    n_total = len(close); n_is = int(n_total * IS_FRAC)
    oos_days = (n_total - n_is) / 288.0

    p, h, d = sim_mtf_state(
        close[n_is:], bid[n_is:], ask[n_is:], sp[n_is:],
        pip, float(sg), M5_LAG, H1_LAG)
    n = len(p)
    if n == 0:
        all_rows.append(dict(pair=pair, n=0, ppd=0.0, wr=0.0,
                             mdd=0.0, mean_hold_m5=0.0, mean_pnl=0.0,
                             max_loss=0.0, max_win=0.0,
                             pct_long=0.0, days=round(oos_days,1)))
        return
    wr      = (p > 0).sum() / n * 100
    ppd     = p.sum() / oos_days
    mdd     = max_dd(p)
    mean_h  = float(h.mean())
    mean_p  = float(p.mean())
    pct_long= (d == 1).sum() / n * 100
    all_rows.append(dict(
        pair=pair, n=n,
        ppd=round(ppd, 2),
        wr=round(wr, 1),
        mdd=round(mdd, 1),
        mean_hold_m5=round(mean_h, 1),
        mean_pnl=round(mean_p, 2),
        max_loss=round(float(p.min()), 1),
        max_win=round(float(p.max()), 1),
        pct_long=round(pct_long, 1),
        days=round(oos_days, 1),
        sum_pips=round(float(p.sum()), 1),
    ))
    del df_m5, close, bid, ask, sp
    gc.collect()


def main():
    warmup_jit()
    print("MTF momentum agreement state-test")
    print(f"  rule: long while c[t]>c[t-1] AND c[t]>c[t-12]")
    print(f"        short while opposite; flat otherwise")
    print(f"  data: M5 BA, 12 pairs, OOS-only (last 30%)")
    rows = []
    t0 = time.time()
    for pair in ALL_PAIRS:
        ts = time.time()
        run_pair(pair, rows)
        print(f"  {pair}: {time.time()-ts:.1f}s  trades={rows[-1]['n']}", flush=True)
    df = pd.DataFrame(rows)
    out_csv = OUT / "mtf_state_agreement.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows -> {out_csv}  ({time.time()-t0:.1f}s)")

    # ── Per-pair detail ──────────────────────────────────────────────────
    df = df.sort_values("ppd", ascending=False)
    print("\n" + "="*100)
    print("  Per-pair OOS (sorted by p/d):")
    print("="*100)
    print(f"  {'pair':<10}{'n':>6}{'Σ pips':>10}{'ppd':>8}{'WR%':>6}"
          f"{'mean pnl':>10}{'hold(M5)':>10}{'%long':>7}{'MDD':>8}"
          f"{'max_loss':>10}{'max_win':>10}")
    for _, r in df.iterrows():
        print(f"  {r['pair']:<10}{int(r['n']):>6}{r['sum_pips']:>+10.1f}"
              f"{r['ppd']:>+8.2f}{r['wr']:>6.1f}{r['mean_pnl']:>+10.2f}"
              f"{r['mean_hold_m5']:>10.1f}{r['pct_long']:>6.1f}%"
              f"{r['mdd']:>8.1f}{r['max_loss']:>+10.1f}{r['max_win']:>+10.1f}")

    # ── Portfolio totals ─────────────────────────────────────────────────
    print("\n" + "="*100)
    total_pips = df.sum_pips.sum()
    total_trades = df.n.sum()
    avg_days = df.days.mean()
    portfolio_ppd = total_pips / avg_days   # avg-pair-day ppd (12 pairs ~same OOS)
    print(f"  Portfolio totals (12 pairs):")
    print(f"    Σ pips       = {total_pips:+.1f}")
    print(f"    Σ p/d        = {portfolio_ppd:+.2f} (per pair-day basis)")
    print(f"    Σ trades     = {total_trades}")
    print(f"    pairs+       = {int((df.ppd > 0).sum())} / 12")
    print(f"    mean WR      = {df.wr.mean():.1f}%")
    print("="*100)


if __name__ == "__main__":
    main()
