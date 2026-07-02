"""
multipair_wf_pullback.py — the deployment gate for the SMA-pullback edge (2026-06-12).

Decisive test: does "buy the deep dip in a strong uptrend + chandelier trail" survive
across PAIRS and WALK-FORWARD folds, or is it USD_JPY drift? Causal (SMA/slope/ATR only,
no swing pivots), real per-pair spread (median ask−bid), 4 contiguous WF folds.

A config is a real edge iff it's net-positive on a MAJORITY of pairs AND walk-forward
consistent (≥3/4 folds positive) on them. USD_JPY-only → it's drift, like the 12h test.
"""
from __future__ import annotations
import gc
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from backtest_trail_channel import resample_h1, atr

ROOT = Path("/path/to/projects/fx-core")
S5_DIR = ROOT / "data/s5_ohlc"
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY",
         "AUD_JPY", "CAD_JPY", "NZD_JPY", "CHF_JPY", "NZD_USD", "EUR_GBP"]
NFOLD = 4

# sensible configs (NOT cherry-picked): (sma, k, depth_spreads, angle_pips_per_bar)
CONFIGS = [
    ("C1 mod",       10, 3.0, 3.0, 0.10),
    ("C2 deep/strong",10, 3.0, 5.0, 0.20),
    ("C3 sma14",      14, 3.0, 3.0, 0.10),
    ("C4 orig-ish",   14, 2.0, 2.0, 0.05),
]


def run_pair(cl, hi, lo, sma_p, k_atr, depth_thr, angle_thr, pip, spread_pips, slope_lb=5):
    n = len(cl)
    sma = np.full(n, np.nan)
    csum = np.cumsum(cl)
    sma[sma_p-1:] = (csum[sma_p-1:] - np.concatenate([[0], csum[:-sma_p]])) / sma_p
    a = atr(hi, lo, cl, 14)
    sp = spread_pips * pip
    depth = depth_thr * sp
    pos = 0; entry = 0.0; peak = 0.0; trough = 0.0
    eidx = []; pnl = []
    start = max(sma_p + slope_lb, 15)
    for i in range(start, n):
        if np.isnan(sma[i]) or np.isnan(a[i]):
            continue
        slope = (sma[i] - sma[i-slope_lb]) / slope_lb / pip
        if pos == 0:
            if slope > angle_thr and (sma[i]-cl[i]) > depth:
                pos = 1; entry = cl[i]+sp/2; peak = hi[i]
            elif slope < -angle_thr and (cl[i]-sma[i]) > depth:
                pos = -1; entry = cl[i]-sp/2; trough = lo[i]
        elif pos == 1:
            peak = max(peak, hi[i])
            if cl[i] < peak - k_atr*a[i]:
                pnl.append((cl[i]-sp/2-entry)/pip); eidx.append(i); pos = 0
        else:
            trough = min(trough, lo[i])
            if cl[i] > trough + k_atr*a[i]:
                pnl.append((entry-(cl[i]+sp/2))/pip); eidx.append(i); pos = 0
    return np.array(eidx), np.array(pnl), n


def main():
    # preload per-pair H1 + spread once
    pdata = {}
    for p in PAIRS:
        f = S5_DIR / f"{p}_S5_BA.parquet"
        if not f.exists():
            print(f"  [skip] {p} (no data)"); continue
        t = pq.read_table(f, columns=["close", "bid_c", "ask_c"])
        close = t.column("close").to_numpy().astype(np.float64)
        pip = 0.01 if "JPY" in p else 0.0001
        sp = float(np.median((t.column("ask_c").to_numpy() - t.column("bid_c").to_numpy()) / pip))
        o, hi, lo, cl = resample_h1(close, 720)
        pdata[p] = (o, hi, lo, cl, pip, sp, len(cl)/24.0)
        del t, close; gc.collect()
    print("pair spreads (pips):", {p: round(v[5], 2) for p, v in pdata.items()})

    for name, sma, k, depth, angle in CONFIGS:
        print(f"\n{'='*88}\n{name}: SMA{sma} k{k:.0f} depth{depth:.0f}sp angle{angle}  "
              f"(net of real spread, {NFOLD} WF folds)\n{'='*88}")
        print(f"{'pair':8s} {'sp':>4s} {'n':>5s} {'WR':>4s} {'tot p/d':>8s} | "
              f"{'fold p/d (4 contiguous)':>34s}  verdict")
        npos = nwf = 0
        for p, (o, hi, lo, cl, pip, sp, days) in pdata.items():
            eidx, pnl, nbar = run_pair(cl, hi, lo, sma, k, depth, angle, pip, sp)
            if len(pnl) < NFOLD*5:
                print(f"{p:8s} {sp:>4.1f} {len(pnl):>5d}  (too few trades)"); continue
            tot_pd = pnl.sum() / days
            edges = np.linspace(0, nbar, NFOLD+1).astype(int)
            fold_pd = []
            for fi in range(NFOLD):
                m = (eidx >= edges[fi]) & (eidx < edges[fi+1])
                fdays = nbar/24.0/NFOLD
                fold_pd.append(pnl[m].sum()/fdays if m.any() else 0.0)
            fold_pos = sum(1 for x in fold_pd if x > 0)
            net_pos = tot_pd > 0
            wf_ok = fold_pos >= 3
            npos += net_pos; nwf += (net_pos and wf_ok)
            v = "🟢 WF" if (net_pos and wf_ok) else ("🟡" if net_pos else "🔴")
            fp = " ".join(f"{x:+6.1f}" for x in fold_pd)
            print(f"{p:8s} {sp:>4.1f} {len(pnl):>5d} {(pnl>0).mean()*100:>3.0f}% "
                  f"{tot_pd:>8.2f} | {fp}  {v}")
        print(f"  → net-positive: {npos}/{len(pdata)} pairs   WF-consistent (3/4 folds): {nwf}/{len(pdata)}")
    print("\nGate: real edge if net-positive on a MAJORITY of pairs AND WF-consistent on them.")


if __name__ == "__main__":
    main()
