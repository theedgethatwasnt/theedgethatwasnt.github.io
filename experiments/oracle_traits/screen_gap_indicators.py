#!/usr/bin/env python3
"""
Exhaustive screen: every directional indicator class in lib/indicators.py that
wasn't in the causal parquet, run CAUSALLY (bar-at-a-time .update) over M5 OHLC,
tested against the OPTIMAL-trade direction. Same metric as indicator_vs_oracle.py:
  r = corr(value, optimal_dir)   (location/scale-invariant -> no centering needed)
  agree% = best-sign directional agreement (allows sign-flip, per user)
Lookahead guard: each oracle entry uses the last CLOSED M5 bar (m5_open+5min <= entry_ts).

usage: screen_gap_indicators.py [PAIR=AUD_JPY]
"""
import sys, inspect
from pathlib import Path
import numpy as np
import pandas as pd
import lib.indicators as I

PROJECT = Path(__file__).resolve().parents[3]
PAIR = sys.argv[1] if len(sys.argv) > 1 else "AUD_JPY"
HERE = Path(__file__).parent

# gap classes (from encyclopedia inventory) — all M5-OHLC computable
GAP = ["Vortex", "SchaffTrendCycle", "Repulse", "CoppockCurve", "PriceOscillator",
       "DMISignal", "DeltaPrice", "TrendQuality", "MomStrength", "PSARDelta",
       "DPO", "DEMA", "TEMA", "MACDZeroLag", "AdaptiveMovingAverage", "StochasticRSI",
       "SMI", "BreakoutChannel", "LinearRegressionSlope", "MACDDivergence",
       "RSIDivergence", "CCIDivergence", "DynamicZoneRSI", "DynamicZoneStochastic",
       "CoppockCurve", "SchaffTrendCycle"]
GAP = sorted(set(GAP))


def scalar_from(ret, obj):
    """Best directional scalar from an update() return or the object's .value."""
    if isinstance(ret, (tuple, list)):
        a = [float(x) for x in ret if isinstance(x, (int, float))]
        if len(a) >= 2:   # e.g. Vortex (VI+,VI-), DMI (+DI,-DI) -> difference is directional
            return a[0] - a[1]
        if a:
            return a[0]
    if isinstance(ret, (int, float)):
        return float(ret)
    v = getattr(obj, "value", None)
    return float(v) if isinstance(v, (int, float)) else np.nan


def run_indicator(cls, bars):
    obj = cls()
    out = np.full(len(bars), np.nan)
    for i, b in enumerate(bars):
        try:
            ret = obj.update(b)
            out[i] = scalar_from(ret, obj)
        except Exception:
            try:
                out[i] = float(getattr(obj, "value", np.nan))
            except Exception:
                out[i] = np.nan
    return out


def main():
    m5 = pd.read_parquet(PROJECT / "data" / "m5_ohlc" / f"{PAIR}_M5.parquet").sort_values("timestamp").reset_index(drop=True)
    if "volume" not in m5: m5["volume"] = 0.0
    if m5["timestamp"].dt.tz is None:
        m5["timestamp"] = m5["timestamp"].dt.tz_localize("UTC")
    bars = [{"open": float(o), "high": float(h), "low": float(l), "close": float(c),
             "volume": float(v), "timestamp": t}
            for o, h, l, c, v, t in zip(m5["open"], m5["high"], m5["low"], m5["close"],
                                        m5["volume"], m5["timestamp"])]
    avail = m5["timestamp"] + pd.Timedelta(minutes=5)            # closed-bar time

    ot = pd.read_parquet(HERE / f"{PAIR}_oracle_trades.parquet")
    ts = pd.read_parquet(PROJECT / "data" / "s5_ba" / f"{PAIR}_S5_BA.parquet", columns=["timestamp"])["timestamp"].values
    entry_ts = pd.to_datetime(ts[ot["entry_idx"].values.astype(int)], utc=True)
    trades = pd.DataFrame({"timestamp": entry_ts, "odir": ot["dir"].values.astype(float),
                           "net": ot["net_pips"].values}).sort_values("timestamp")

    print(f"{PAIR}: screening {len(GAP)} gap indicators over {len(m5):,} M5 bars vs {len(trades):,} optimal trades\n")
    print(f"  {'indicator':<22}{'r':>8}{'r_big':>8}{'agree%':>8}  verdict")
    results = []
    for name in GAP:
        cls = getattr(I, name, None)
        if cls is None or not inspect.isclass(cls):
            print(f"  {name:<22}{'—':>8}{'—':>8}{'—':>8}  (not found)"); continue
        series = run_indicator(cls, bars)
        if np.isfinite(series).sum() < 1000 or np.nanstd(series) == 0:
            print(f"  {name:<22}{'—':>8}{'—':>8}{'—':>8}  (no/constant output)"); continue
        ind = pd.DataFrame({"timestamp": avail, "val": series}).dropna()
        df = pd.merge_asof(trades, ind, on="timestamp", direction="backward").dropna(subset=["val"])
        x, od, net = df["val"].values, df["odir"].values, df["net"].values
        if np.std(x) == 0:
            print(f"  {name:<22}{'—':>8}{'—':>8}{'—':>8}  (constant after align)"); continue
        r = np.corrcoef(x, od)[0, 1]
        big = net >= 10.0
        rb = np.corrcoef(x[big], od[big])[0, 1] if big.sum() > 200 and np.std(x[big]) > 0 else np.nan
        s = np.sign(x - np.median(x)); s[s == 0] = 1
        ag = max((s == od).mean(), (-s == od).mean()) * 100
        v = "SIGNAL" if abs(r) >= 0.05 else ("weak" if abs(r) >= 0.03 else "noise")
        rbs = f"{rb:+.3f}" if np.isfinite(rb) else "   n/a"
        print(f"  {name:<22}{r:>+8.3f}{rbs:>8}{ag:>7.1f}%  {v}")
        results.append((name, r, rb, ag))
    if results:
        best = max(results, key=lambda t: abs(t[1]))
        print(f"\n  strongest gap indicator: {best[0]} r={best[1]:+.3f} ({best[3]:.1f}% agree). "
              f"|r|>=0.05? {'YES' if any(abs(t[1])>=0.05 for t in results) else 'NO'}")


if __name__ == "__main__":
    main()
