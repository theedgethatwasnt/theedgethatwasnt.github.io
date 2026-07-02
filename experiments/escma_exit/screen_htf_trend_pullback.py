"""
screen_htf_trend_pullback.py — HTF trend + pullback entry screen (2026-06-12).

User idea: spot an ongoing HIGHER-timeframe trend, then trade WITH the trend on a
significant PULLBACK swing (buy the dip in an uptrend) — using HTF momentum lags
5m / 1h / 4h. Test the entry signal directly (LightGBM + conditional expectancy)
BEFORE building the ESCMA exit, at hourly-to-daily holds where the 1.7p spread is a
small hurdle.

Features (causal, raw pip momentum — LightGBM is scale-invariant):
  mom_5m = c[t]−c[t−60]    mom_1h = c[t]−c[t−720]    mom_4h = c[t]−c[t−2880]   (S5 bars)
Forward target: signed mid return over H bars (pips). H ∈ {720, 2880, 8640} = 1h/4h/12h.

Two tests:
  A) LightGBM(3 HTF mom) → forward return: IC + decile spread vs spread (does HTF
     momentum predict direction at all?).
  B) Conditional expectancy of the SPECIFIC setups, net of round-trip spread:
       trend_up = mom_1h>0 & mom_4h>0 ;  pullback = mom_5m<0
       - WITH-trend pullback long:   E[fwd | trend_up & pullback]      − spread
       - WITH-trend continuation:    E[fwd | trend_up & mom_5m>0]      − spread   (compare)
       - mirror shorts for trend_down.
     Edge iff the pullback-with-trend expectancy clears the spread.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent


def cond_stat(y, mask, sp, label, side):
    """side +1 long (net = mean(y)-sp), -1 short (net = mean(-y)-sp)."""
    m = mask & np.isfinite(y)
    if m.sum() < 50:
        print(f"    {label:32s} n={int(m.sum()):>7}  (too few)")
        return
    mean = y[m].mean()
    net = side * mean - sp
    flag = "🟢 > spread" if net > 0 else "🔴 < spread"
    print(f"    {label:32s} n={int(m.sum()):>7}  E[fwd]={mean:+.2f}p  net={net:+.2f}p  {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="USD_JPY")
    ap.add_argument("--spread-pips", type=float, default=1.7)
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--horizons", default="720,2880,8640")  # 1h/4h/12h in S5 bars
    args = ap.parse_args()

    import lightgbm as lgb
    pip = 0.01 if "JPY" in args.pair else 0.0001
    sp = args.spread_pips
    close = pq.read_table(SCRIPT_DIR / f"features_{args.pair}.parquet",
                          columns=["close"]).column("close").to_numpy().astype(np.float64)
    n = close.shape[0]

    def mom(lag):
        m = np.full(n, np.nan); m[lag:] = (close[lag:] - close[:-lag]) / pip; return m
    m5, m1h, m4h = mom(60), mom(720), mom(2880)
    X_full = np.column_stack([m5, m1h, m4h])

    trend_up = (m1h > 0) & (m4h > 0)
    trend_dn = (m1h < 0) & (m4h < 0)
    pull_dn = m5 < 0     # pullback within (would-be) uptrend
    pull_up = m5 > 0

    print(f"\n{'='*72}\nHTF TREND + PULLBACK SCREEN — {args.pair}  spread={sp}p  stride={args.stride}\n"
          f"momentum lags 5m/1h/4h ; forward holds 1h/4h/12h\n{'='*72}")

    for H in [int(h) for h in args.horizons.split(",")]:
        y = np.full(n, np.nan); y[:n-H] = (close[H:] - close[:n-H]) / pip
        valid = np.all(np.isfinite(X_full), axis=1) & np.isfinite(y)
        idx = np.where(valid)[0]; idx = idx[(idx % args.stride == 0) & (idx > 2880)]
        X, yv = X_full[idx], y[idx]
        cut = int(len(idx) * 0.70)
        # ---- A) LightGBM predictability ----
        model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  min_child_samples=200, verbosity=-1)
        model.fit(X[:cut], yv[:cut])
        pred = model.predict(X[cut:]); yte = yv[cut:]
        ic = np.corrcoef(pred, yte)[0, 1]
        order = np.argsort(pred); d = len(order)//10
        top, bot = yte[order[-d:]].mean(), yte[order[:d]].mean()
        print(f"\n── Hold H={H} bars ({H*5/60/60:.1f}h) ── n_oos={len(yte):,}  spread bar={sp}p")
        print(f"  [A] LightGBM IC={ic:+.4f}  top decile {top:+.2f}p (long net {top-sp:+.2f})  "
              f"bot decile {bot:+.2f}p (short net {-bot-sp:+.2f})")
        # ---- B) conditional setups (OOS only) ----
        oos = np.zeros(n, bool); oos[idx[cut:]] = True
        print(f"  [B] conditional expectancy (OOS):")
        cond_stat(y, oos & trend_up & pull_dn, sp, "LONG: uptrend + pullback (dip)", +1)
        cond_stat(y, oos & trend_up & pull_up, sp, "LONG: uptrend + continuation", +1)
        cond_stat(y, oos & trend_dn & pull_up, sp, "SHORT: downtrend + pullback (pop)", -1)
        cond_stat(y, oos & trend_dn & pull_dn, sp, "SHORT: downtrend + continuation", -1)
    print("="*72)


if __name__ == "__main__":
    main()
