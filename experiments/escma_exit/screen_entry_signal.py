"""
screen_entry_signal.py — Phase 2 cheap screen (2026-06-12).

Before building the heavy two-net continuous sim, ask the decisive question directly:
does ANY (nonlinear) combination of the 5 mn channels predict a forward move LARGER
than the round-trip spread? If not, a learned entry ESCMA cannot help and Phase 2 is
pointless.

Method (no position engine, just predictability):
  X = the 5 mn_* channels at bar t (causal).
  y_K = forward mid return over K bars in pips = (mid[t+K] − mid[t]) / pip.   K ∈ {12,60,180}.
  Temporal 70/30 split. LightGBM regressor (nonlinear, captures channel interactions).
Metrics on OOS:
  • IC = corr(pred, realised y).
  • Decile spread: mean realised move of the top/bottom predicted deciles (pips).
  • Net-of-spread expectancy: top decile → go long, mean(y) − spread; bottom decile →
    go short, mean(−y) − spread. Edge exists iff either is > 0.
  • Univariate IC of each single channel (sanity).
Verdict: if no decile's |mean move| clears the spread, no learned entry can beat cost.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
MN = ["mn_S5", "mn_M1", "mn_5m", "mn_15m", "mn_1h"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="USD_JPY")
    ap.add_argument("--horizons", default="12,60,180")
    ap.add_argument("--spread-pips", type=float, default=1.7)
    ap.add_argument("--stride", type=int, default=12, help="subsample every N bars")
    args = ap.parse_args()

    import lightgbm as lgb
    pip = 0.01 if "JPY" in args.pair else 0.0001
    cols = ["close"] + MN
    tbl = pq.read_table(SCRIPT_DIR / f"features_{args.pair}.parquet", columns=cols)
    close = tbl.column("close").to_numpy().astype(np.float64)
    X_full = np.column_stack([tbl.column(c).to_numpy().astype(np.float64) for c in MN])
    n = close.shape[0]
    horizons = [int(h) for h in args.horizons.split(",")]
    sp = args.spread_pips

    print(f"\n{'='*70}\nENTRY-SIGNAL SCREEN — {args.pair}  spread={sp}p  stride={args.stride}\n"
          f"Can the 5 mn channels predict a forward move > spread?\n{'='*70}")

    # univariate IC vs the longest horizon (quick sanity)
    Kref = horizons[-1]
    yref = np.full(n, np.nan); yref[:n-Kref] = (close[Kref:] - close[:n-Kref]) / pip
    print(f"\nUnivariate IC vs forward {Kref}-bar return:")
    for j, name in enumerate(MN):
        m = np.isfinite(X_full[:, j]) & np.isfinite(yref)
        ic = np.corrcoef(X_full[m, j], yref[m])[0, 1]
        print(f"  {name:7s}  IC={ic:+.4f}")

    for K in horizons:
        y = np.full(n, np.nan); y[:n-K] = (close[K:] - close[:n-K]) / pip
        valid = np.isfinite(y) & np.all(np.isfinite(X_full), axis=1)
        idx = np.where(valid)[0]
        idx = idx[idx % args.stride == 0]          # subsample for independence/size
        warm = 720
        idx = idx[idx > warm]
        X, yv = X_full[idx], y[idx]
        cut = int(len(idx) * 0.70)                 # temporal split
        Xtr, ytr, Xte, yte = X[:cut], yv[:cut], X[cut:], yv[cut:]

        model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, min_child_samples=200,
                                  verbosity=-1)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        ic = np.corrcoef(pred, yte)[0, 1]

        order = np.argsort(pred)
        d = len(order) // 10
        bot = yte[order[:d]]; top = yte[order[-d:]]
        top_mean, bot_mean = top.mean(), bot.mean()
        long_net = top_mean - sp           # go long on top decile
        short_net = (-bot_mean) - sp       # go short on bottom decile
        verdict = "🟢 EDGE > spread" if (long_net > 0 or short_net > 0) else "🔴 < spread"
        print(f"\nHorizon K={K} bars ({K*5/60:.1f} min)  n_oos={len(yte):,}  IC={ic:+.4f}")
        print(f"  top decile mean move = {top_mean:+.3f}p  -> long net of spread = {long_net:+.3f}p")
        print(f"  bot decile mean move = {bot_mean:+.3f}p  -> short net of spread = {short_net:+.3f}p")
        print(f"  decile spread = {top_mean-bot_mean:.3f}p  (need > {2*sp:.1f}p to trade both sides)")
        print(f"  VERDICT: {verdict}")
    print("="*70)


if __name__ == "__main__":
    main()
