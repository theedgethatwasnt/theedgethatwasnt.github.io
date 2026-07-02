"""
backtest_trail_channel.py — HTF trend-ride with a trailing channel exit (2026-06-12).

User's chart idea (hourly USD/JPY): ride an established trend, exit when price breaks a
channel that follows price like a trailing stop (lower band in an uptrend, upper in a
downtrend). The "average uptrend" = an SMA; the channel = a trailing peak − k·ATR
(Chandelier). This is the ONE family the project found real edge in (H4 Donchian
+16–25 p/d) — test it on the hourly, net of round-trip spread, IS/OOS.

H1 bars resampled from S5 close. Entry: SMA(P) rising AND close>SMA → long (mirror short).
Exit: Chandelier — long exits when close < max(high since entry) − k·ATR(14); short mirror.
Also reports the SMA-cross exit variant. Spread deducted round-trip per trade.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent


def resample_h1(close_s5, bars_per=720):
    n = close_s5.shape[0]
    nb = n // bars_per
    c = close_s5[:nb * bars_per].reshape(nb, bars_per)
    o = c[:, 0]; cl = c[:, -1]; hi = c.max(axis=1); lo = c.min(axis=1)
    return o, hi, lo, cl


def atr(hi, lo, cl, p=14):
    n = len(cl); tr = np.empty(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
    a = np.full(n, np.nan); a[p-1] = tr[:p].mean()
    for i in range(p, n):
        a[i] = (a[i-1]*(p-1) + tr[i]) / p
    return a


def rolling_reg(cl, N):
    """Rolling linear-regression endpoint value + slope over the last N bars.
    Returns reg_now[i] (line value at bar i) and slope[i] (per-bar)."""
    n = len(cl)
    x = np.arange(N, dtype=np.float64)
    Sx = x.sum(); Sxx = (x*x).sum(); denom = N*Sxx - Sx*Sx
    w = x[::-1].copy()                      # convolution weights for Σ x_j y_j
    Sxy = np.full(n, np.nan)
    Sxy[N-1:] = np.convolve(cl, w, mode="valid")
    csum = np.concatenate([[0.0], np.cumsum(cl)])
    Sy = np.full(n, np.nan); Sy[N-1:] = csum[N:] - csum[:-N]
    slope = (N*Sxy - Sx*Sy) / denom
    intercept = (Sy - slope*Sx) / N
    reg_now = intercept + slope*(N-1)       # value at the current (last) bar
    return reg_now, slope


def backtest(o, hi, lo, cl, sma_p, k_atr, pip, spread_pips, is_frac, exit_mode,
             entry_style="trend"):
    n = len(cl)
    sma = np.full(n, np.nan)
    csum = np.cumsum(cl)
    sma[sma_p-1:] = (csum[sma_p-1:] - np.concatenate([[0], csum[:-sma_p]])) / sma_p
    a = atr(hi, lo, cl, 14)
    sp = spread_pips * pip
    reg, slope = rolling_reg(cl, sma_p)     # sloped trend line over the SMA window

    pos = 0; entry = 0.0; peak = 0.0; trough = 0.0
    trades = []  # (exit_idx, pnl_pips, dir)
    start = max(sma_p, 15)
    for i in range(start, n):
        if np.isnan(sma[i]) or np.isnan(a[i]) or np.isnan(reg[i]):
            continue
        if entry_style == "pullback":
            # established trend (rising/falling line) but price has PULLED BACK
            # through its average — buy the dip in an uptrend, sell the pop in a downtrend
            up = slope[i] > 0 and cl[i] < sma[i]
            dn = slope[i] < 0 and cl[i] > sma[i]
        elif exit_mode == "regchan":         # SLOPED line, enter while above it
            up = slope[i] > 0 and cl[i] > reg[i]
            dn = slope[i] < 0 and cl[i] < reg[i]
        else:
            up = cl[i] > sma[i] and sma[i] > sma[i-1]
            dn = cl[i] < sma[i] and sma[i] < sma[i-1]
        if pos == 0:
            if up:
                pos = 1; entry = cl[i] + sp/2; peak = hi[i]
            elif dn:
                pos = -1; entry = cl[i] - sp/2; trough = lo[i]
        elif pos == 1:
            peak = max(peak, hi[i])
            if exit_mode == "chandelier":
                brk = cl[i] < peak - k_atr * a[i]
            elif exit_mode == "regchan":     # break below the rising line − k·ATR
                brk = cl[i] < reg[i] - k_atr * a[i]
            else:
                brk = cl[i] < sma[i]
            if brk:
                ex = cl[i] - sp/2
                trades.append((i, (ex - entry)/pip, 1)); pos = 0
        elif pos == -1:
            trough = min(trough, lo[i])
            if exit_mode == "chandelier":
                brk = cl[i] > trough + k_atr * a[i]
            elif exit_mode == "regchan":
                brk = cl[i] > reg[i] + k_atr * a[i]
            else:
                brk = cl[i] > sma[i]
            if brk:
                ex = cl[i] + sp/2
                trades.append((i, (entry - ex)/pip, -1)); pos = 0

    if not trades:
        return None
    idx = np.array([t[0] for t in trades]); pnl = np.array([t[1] for t in trades])
    cut_bar = int(n * is_frac)
    is_m = idx < cut_bar; oos_m = ~is_m
    days = n / 24.0  # H1 bars → hours → days
    def summ(m):
        if m.sum() == 0:
            return (0, 0.0, 0.0, 0.0)
        return (int(m.sum()), float(pnl[m].sum()), float((pnl[m] > 0).mean()),
                float(pnl[m].sum() / (days * (m.sum()/len(pnl)) + 1e-9)))
    return {"n": len(pnl), "is": summ(is_m), "oos": summ(oos_m),
            "tot_pnl": float(pnl.sum()), "wr": float((pnl > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="USD_JPY")
    ap.add_argument("--spread-pips", type=float, default=1.7)
    ap.add_argument("--is-frac", type=float, default=0.70)
    ap.add_argument("--hours", type=float, default=1.0, help="bar size in hours (1=H1, 4=H4)")
    args = ap.parse_args()
    pip = 0.01 if "JPY" in args.pair else 0.0001
    close = pq.read_table(SCRIPT_DIR / f"features_{args.pair}.parquet",
                          columns=["close"]).column("close").to_numpy().astype(np.float64)
    bars_per = int(720 * args.hours)
    o, hi, lo, cl = resample_h1(close, bars_per)
    days = len(cl) * args.hours / 24.0
    print(f"\n{'='*78}\nTRAILING-CHANNEL TREND-FOLLOW — {args.pair} H{args.hours:g}  "
          f"{len(cl):,} bars (~{days:.0f}d)  spread={args.spread_pips}p\n{'='*78}")
    print(f"{'entry':9s} {'exit':10s} {'SMA':>4s} {'k':>4s} | {'n':>5s} {'WR':>5s} {'IS p/d':>8s} {'OOS_n':>6s} {'OOS p/d':>8s} {'OOS_pnl':>9s}")
    combos = [("pullback", "regchan"), ("trend", "regchan"), ("pullback", "chandelier")]
    for entry_style, exit_mode in combos:
        for sma_p in (14, 24, 50):
            ks = (2.0, 3.0, 4.0) if exit_mode in ("chandelier", "regchan") else (0.0,)
            for k in ks:
                r = backtest(o, hi, lo, cl, sma_p, k, pip, args.spread_pips, args.is_frac,
                             exit_mode, entry_style)
                if not r:
                    continue
                isn, ispnl, iswr, ispd = r["is"]; on, opnl, owr, opd = r["oos"]
                flag = "🟢" if opd > 0 else "🔴"
                print(f"{entry_style:9s} {exit_mode:10s} {sma_p:>4d} {k:>4.1f} | {r['n']:>5d} {r['wr']*100:>4.0f}% "
                      f"{ispd:>8.2f} {on:>6d} {opd:>8.2f} {opnl:>9.1f} {flag}")
    print("="*78)
    print("p/d = pips/day net of spread. (Validate multi-pair + WF before trusting any 🟢.)")


if __name__ == "__main__":
    main()
