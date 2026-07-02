"""
sweep_sma_pullback.py — sharpen the SMA-pullback edge (2026-06-12).

The clean (no-lookahead) "buy the dip in an uptrend" that gave +1.2 p/d OOS on USD_JPY
H1 used only "close < SMA". Sweep the three filters the user asked for:
  - SMA period
  - DEPTH threshold: enter only if |SMA − close| > depth_thr × spread  (significant pullback)
  - ANGLE threshold: SMA slope (over a few bars) must exceed angle_thr pips/bar (real trend)
Exit = chandelier (peak − k·ATR). Net of round-trip spread. IS/OOS split.

SOP discipline (R8): rank configs by IS p/d and report each one's OOS beside it. Real
edges are robust neighborhoods where OOS ≈ IS, not the single max-OOS cherry pick.
All causal: SMA / slope / ATR use only past bars; no swing pivots (no lookahead).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from backtest_trail_channel import resample_h1, atr

SCRIPT_DIR = Path(__file__).resolve().parent


def backtest_pb(cl, hi, lo, sma_p, k_atr, depth_thr, angle_thr, slope_lb,
                pip, spread_pips, is_frac):
    n = len(cl)
    sma = np.full(n, np.nan)
    csum = np.cumsum(cl)
    sma[sma_p-1:] = (csum[sma_p-1:] - np.concatenate([[0], csum[:-sma_p]])) / sma_p
    a = atr(hi, lo, cl, 14)
    sp = spread_pips * pip
    depth = depth_thr * sp                          # min |SMA−close| in price units

    pos = 0; entry = 0.0; peak = 0.0; trough = 0.0
    pnl = []; eidx = []
    start = max(sma_p + slope_lb, 15)
    for i in range(start, n):
        if np.isnan(sma[i]) or np.isnan(a[i]):
            continue
        slope = (sma[i] - sma[i-slope_lb]) / slope_lb / pip   # pips/bar
        if pos == 0:
            if slope > angle_thr and (sma[i] - cl[i]) > depth:        # uptrend + deep dip
                pos = 1; entry = cl[i] + sp/2; peak = hi[i]
            elif slope < -angle_thr and (cl[i] - sma[i]) > depth:     # downtrend + deep pop
                pos = -1; entry = cl[i] - sp/2; trough = lo[i]
        elif pos == 1:
            peak = max(peak, hi[i])
            if cl[i] < peak - k_atr * a[i]:
                pnl.append((cl[i]-sp/2 - entry)/pip); eidx.append(i); pos = 0
        else:
            trough = min(trough, lo[i])
            if cl[i] > trough + k_atr * a[i]:
                pnl.append((entry - (cl[i]+sp/2))/pip); eidx.append(i); pos = 0

    if len(pnl) < 1:
        return None
    pnl = np.array(pnl); eidx = np.array(eidx)
    cut = int(n * is_frac); days = n / 24.0
    ism = eidx < cut; oosm = ~ism
    def pd_(m):
        return float(pnl[m].sum() / (days * (m.sum()/len(pnl)) + 1e-9)) if m.sum() else 0.0
    return {"n": len(pnl), "wr": float((pnl > 0).mean()),
            "is_n": int(ism.sum()), "is_pd": pd_(ism),
            "oos_n": int(oosm.sum()), "oos_pd": pd_(oosm),
            "oos_pnl": float(pnl[oosm].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="USD_JPY")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--spread-pips", type=float, default=1.7)
    ap.add_argument("--is-frac", type=float, default=0.70)
    ap.add_argument("--slope-lb", type=int, default=5)
    ap.add_argument("--min-oos-n", type=int, default=60)
    args = ap.parse_args()
    pip = 0.01 if "JPY" in args.pair else 0.0001
    close = pq.read_table(SCRIPT_DIR / f"features_{args.pair}.parquet",
                          columns=["close"]).column("close").to_numpy().astype(np.float64)
    o, hi, lo, cl = resample_h1(close, int(720*args.hours))

    rows = []
    for sma_p in (8, 10, 14, 20, 30, 50):
        for k in (2.0, 3.0, 4.0):
            for depth in (0.0, 1.0, 2.0, 3.0, 5.0):
                for angle in (0.0, 0.05, 0.1, 0.2):
                    r = backtest_pb(cl, hi, lo, sma_p, k, depth, angle, args.slope_lb,
                                    pip, args.spread_pips, args.is_frac)
                    if r and r["is_n"] >= 40:
                        r.update(sma=sma_p, k=k, depth=depth, angle=angle)
                        rows.append(r)

    print(f"\n{'='*92}\nSMA-PULLBACK SWEEP — {args.pair} H{args.hours:g}  spread={args.spread_pips}p  "
          f"slope_lb={args.slope_lb}  (ranked by IS p/d; OOS beside)\n{'='*92}")
    print(f"{'SMA':>3s} {'k':>3s} {'dep':>4s} {'ang':>5s} | {'n':>5s} {'WR':>4s} "
          f"{'IS p/d':>7s} {'OOSn':>5s} {'OOS p/d':>8s} {'OOSpnl':>8s}  robust?")
    rows.sort(key=lambda z: -z["is_pd"])
    shown = 0
    for r in rows:
        if r["oos_n"] < args.min_oos_n:
            continue
        rob = "🟢 OOS≈IS" if (r["oos_pd"] > 0 and r["is_pd"] > 0 and
                              r["oos_pd"] > 0.4*r["is_pd"]) else ("🟡" if r["oos_pd"] > 0 else "🔴")
        print(f"{r['sma']:>3d} {r['k']:>3.0f} {r['depth']:>4.1f} {r['angle']:>5.2f} | "
              f"{r['n']:>5d} {r['wr']*100:>3.0f}% {r['is_pd']:>7.2f} {r['oos_n']:>5d} "
              f"{r['oos_pd']:>8.2f} {r['oos_pnl']:>8.1f}  {rob}")
        shown += 1
        if shown >= 25:
            break
    print("="*92)
    pos = [r for r in rows if r["oos_n"] >= args.min_oos_n and r["is_pd"] > 0 and r["oos_pd"] > 0]
    print(f"configs with IS>0 AND OOS>0 (n_oos≥{args.min_oos_n}): {len(pos)} / {len([r for r in rows if r['oos_n']>=args.min_oos_n])}")


if __name__ == "__main__":
    main()
