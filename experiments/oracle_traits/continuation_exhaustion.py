#!/usr/bin/env python3
"""
CONTINUATION vs EXHAUSTION screen (per user): not 'which way from a standstill at one
bar', but 'given price has an angle/move underway, is it CONTINUING or FINISHED' — with
the outcome measured many bars ahead.

Ground-truth moves = oracle optimal trades (maximal clean directional segments). For
each bar INSIDE a move we measure, against each causal indicator (oriented by the move
direction so + = 'reads strong in the move's favour'):
  remaining = favourable pips left to the move's end (the multi-bar outcome, many bars away)
  frac      = how far through the move we are (0=just started .. 1=about to finish)
Per indicator:
  r_cont = corr(oriented_indicator, remaining)   (+ = CONTINUATION gauge; - = EXHAUSTION gauge)
  r_age  = corr(oriented_indicator, frac)         (- = fades as the move ages = exhaustion detector)
Restricted to moves long enough to resolve at M5 (median optimal hold is ~85s < 1 M5 bar,
so single-bar tests were doomed). Closed-M5 lookahead guard throughout.

usage: continuation_exhaustion.py [PAIR=AUD_JPY] [MIN_HOLD_MIN=10]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
PAIR = sys.argv[1] if len(sys.argv) > 1 else "AUD_JPY"
MIN_HOLD_MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
PIP = 0.01 if "JPY" in PAIR else 0.0001
HERE = Path(__file__).parent
CENTER = {"rsi_14": 50.0, "stoch_k": 50.0, "stoch_d": 50.0, "cci": 0.0,
          "range_pos_30": 0.5, "donchian_pos": 0.5, "aroon_osc": 0.0,
          "williams_r": -50.0, "ema8_ratio": 1.0, "ema21_ratio": 1.0}
MAG = {"atr_ratio", "bb_width", "candle_range", "body_pips", "body_ratio", "bar_count", "vol_ratio_20"}


def main():
    ot = pd.read_parquet(HERE / f"{PAIR}_oracle_trades.parquet")   # entry_idx, exit_idx, dir, hold_bars
    min_hold_s5 = MIN_HOLD_MIN * 60 / 5
    keep = ot["hold_bars"].values >= min_hold_s5
    ot = ot[keep].reset_index(drop=True)
    s5 = pd.read_parquet(PROJECT / "data" / "s5_ba" / f"{PAIR}_S5_BA.parquet", columns=["timestamp", "close"])
    s5_ts = s5["timestamp"].values; s5_mid = s5["close"].values
    ei = ot["entry_idx"].values.astype(int); xi = ot["exit_idx"].values.astype(int); d = ot["dir"].values.astype(float)
    print(f"{PAIR}: {len(ot):,} optimal moves with hold>= {MIN_HOLD_MIN:.0f}min "
          f"({keep.mean()*100:.1f}% of all optimal trades resolve at M5)")

    m5 = pd.read_parquet(PROJECT / "data" / "m5_ohlc" / f"{PAIR}_M5_kalman10_causal.parquet")
    if m5["timestamp"].dt.tz is None:
        m5["timestamp"] = m5["timestamp"].dt.tz_localize("UTC")
    m5 = m5.sort_values("timestamp").reset_index(drop=True)
    m5_avail = (m5["timestamp"] + pd.Timedelta(minutes=5)).values     # closed-bar time
    feat = [c for c in m5.columns if c not in ("timestamp", "open", "high", "low", "close")
            and np.issubdtype(m5[c].dtype, np.number)]
    M = {c: m5[c].values.astype(float) for c in feat}

    # collect in-move M5 samples
    oriented = {c: [] for c in feat}
    REM = []; FRAC = []
    s5_ts_i64 = s5_ts.astype("datetime64[ns]")
    for k in range(len(ot)):
        t0 = s5_ts_i64[ei[k]]; t1 = s5_ts_i64[xi[k]]
        # M5 bars whose CLOSE time is within (t0, t1]
        a = np.searchsorted(m5_avail, t0, side="left"); b = np.searchsorted(m5_avail, t1, side="right")
        if b - a < 2:
            continue
        exit_mid = s5_mid[xi[k]]
        for j in range(a, b):
            # mid at this M5 close ~ nearest s5; use m5 close
            mb = m5["close"].values[j]
            rem = d[k] * (exit_mid - mb) / PIP
            frac = (m5_avail[j] - t0) / (t1 - t0)
            REM.append(rem); FRAC.append(float(frac))
            for c in feat:
                v = M[c][j] - CENTER.get(c, 0.0)
                oriented[c].append(v * d[k])
    REM = np.asarray(REM); FRAC = np.asarray(FRAC)
    print(f"  in-move M5 samples: {len(REM):,}   mean remaining fav pips {np.nanmean(REM):.1f}\n")
    if len(REM) < 500:
        print("  too few resolvable samples — raise data or lower MIN_HOLD"); return

    rows = []
    for c in feat:
        x = np.asarray(oriented[c]); m = np.isfinite(x) & np.isfinite(REM)
        if m.sum() < 500 or np.std(x[m]) == 0:
            continue
        rc = np.corrcoef(x[m], REM[m])[0, 1]
        ma = np.isfinite(x) & np.isfinite(FRAC)
        ra = np.corrcoef(x[ma], FRAC[ma])[0, 1] if np.std(x[ma]) > 0 else np.nan
        rows.append((c, rc, ra))
    res = pd.DataFrame(rows, columns=["indicator", "r_cont", "r_age"])
    res = res.reindex(res["r_cont"].abs().sort_values(ascending=False).index)
    print(f"  {'indicator':<18}{'r_cont':>9}{'r_age':>9}  reads")
    for _, r in res.head(20).iterrows():
        kind = "size" if r["indicator"] in MAG else ("CONTINUATION" if r["r_cont"] > 0 else "exhaustion")
        v = "  <-- signal" if abs(r["r_cont"]) >= 0.08 else ("  (weak)" if abs(r["r_cont"]) >= 0.05 else "")
        print(f"  {r['indicator']:<18}{r['r_cont']:>+9.3f}{r['r_age']:>+9.3f}  {kind}{v}")
    best = res.iloc[0]
    print(f"\n  strongest: {best['indicator']} r_cont={best['r_cont']:+.3f}. "
          f"|r_cont|>=0.08 anywhere? {'YES' if (res['r_cont'].abs()>=0.08).any() else 'NO'}")


if __name__ == "__main__":
    main()
