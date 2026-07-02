#!/usr/bin/env python3
"""
Do ANY of our causal indicators agree with the OPTIMAL-trade direction at a
meaningful degree?

For each causal indicator (M5, from {pair}_M5_kalman10_causal.parquet) we align its
last-CLOSED-M5 value to every oracle optimal-trade ENTRY bar (lookahead guard:
m5_open_ts+5min <= entry_ts), then measure how its sign relates to the optimal
trade's direction:
  - r  = point-biserial corr(indicator value, optimal_dir in {-1,+1})  (sign tells
         momentum[+] vs contrarian[-]; |r| tells strength; 0 = noise)
  - agree% = directional agreement using the BEST sign (max of mom/contra)
  - bigR = same r but on the reachable band only (optimal net >= 10p) — the trades
           that actually clear spread and carry the reward.
Sorted by |r|. A meaningful signal must clear |r| ~0.05+ AND hold on the big-trade cut.

usage: indicator_vs_oracle.py [PAIR=AUD_JPY]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
PAIR = sys.argv[1] if len(sys.argv) > 1 else "AUD_JPY"
HERE = Path(__file__).parent

# centered/oscillator indicators get their neutral level subtracted so sign = direction
CENTER = {"rsi_14": 50.0, "stoch_k": 50.0, "stoch_d": 50.0, "cci": 0.0,
          "range_pos_30": 0.5, "donchian_pos": 0.5, "aroon_osc": 0.0,
          "ema8_ratio": 1.0, "ema21_ratio": 1.0}
# magnitude-only (predict SIZE not direction per prior findings) — report but flag
MAG_ONLY = {"atr_ratio", "bb_width", "candle_range", "body_pips", "body_ratio", "bar_count"}


def main():
    ot = pd.read_parquet(HERE / f"{PAIR}_oracle_trades.parquet")   # entry_idx, dir, net_pips
    ts = pd.read_parquet(PROJECT / "data" / "s5_ba" / f"{PAIR}_S5_BA.parquet",
                         columns=["timestamp"])["timestamp"].values
    entry_ts = pd.to_datetime(ts[ot["entry_idx"].values.astype(int)], utc=True)
    trades = pd.DataFrame({"timestamp": entry_ts, "odir": ot["dir"].values.astype(float),
                           "net": ot["net_pips"].values}).sort_values("timestamp")

    m5 = pd.read_parquet(PROJECT / "data" / "m5_ohlc" / f"{PAIR}_M5_kalman10_causal.parquet")
    if m5["timestamp"].dt.tz is None:
        m5["timestamp"] = m5["timestamp"].dt.tz_localize("UTC")
    m5 = m5.sort_values("timestamp")
    m5["timestamp"] = m5["timestamp"] + pd.Timedelta(minutes=5)     # closed-bar avail time

    feat_cols = [c for c in m5.columns if c not in ("timestamp", "open", "high", "low", "close")
                 and np.issubdtype(m5[c].dtype, np.number)]
    df = pd.merge_asof(trades, m5[["timestamp"] + feat_cols], on="timestamp", direction="backward")
    df = df.dropna(subset=feat_cols, how="all")
    n = len(df); big = df["net"].values >= 10.0
    print(f"{PAIR}: {n:,} optimal trades aligned to closed-M5 indicators "
          f"(reachable band net>=10p: {big.sum():,} = {big.mean()*100:.0f}%)\n")

    rows = []
    odir = df["odir"].values
    for c in feat_cols:
        x = df[c].values.astype(float)
        if c in CENTER:
            x = x - CENTER[c]
        m = np.isfinite(x)
        if m.sum() < 500 or np.std(x[m]) == 0:
            continue
        r = np.corrcoef(x[m], odir[m])[0, 1]
        mb = m & big
        rb = np.corrcoef(x[mb], odir[mb])[0, 1] if mb.sum() > 200 and np.std(x[mb]) > 0 else np.nan
        # directional agreement using best sign
        s = np.sign(x[m]); s[s == 0] = 1
        ag = max((s == odir[m]).mean(), (-s == odir[m]).mean()) * 100
        rows.append((c, r, rb))
    res = pd.DataFrame(rows, columns=["indicator", "r", "r_big"])
    ag = []
    for c in res["indicator"]:
        x = df[c].values.astype(float)
        if c in CENTER: x = x - CENTER[c]
        m = np.isfinite(x); s = np.sign(x[m]); s[s == 0] = 1
        ag.append(max((s == odir[m]).mean(), (-s == odir[m]).mean()) * 100)
    res["agree%"] = ag
    res["kind"] = ["MAG" if c in MAG_ONLY else ("mom" if r >= 0 else "contra")
                   for c, r in zip(res["indicator"], res["r"])]
    res = res.reindex(res["r"].abs().sort_values(ascending=False).index)

    print(f"  {'indicator':<16}{'r':>8}{'r_big':>8}{'agree%':>8}  kind   verdict")
    for _, row in res.iterrows():
        meaningful = abs(row["r"]) >= 0.05 and abs(row["r_big"]) >= 0.04 if np.isfinite(row["r_big"]) else abs(row["r"]) >= 0.05
        v = "SIGNAL" if meaningful else ("weak" if abs(row["r"]) >= 0.03 else "noise")
        rb = f"{row['r_big']:+.3f}" if np.isfinite(row["r_big"]) else "   n/a"
        flag = "  (size, not dir)" if row["kind"] == "MAG" else ""
        print(f"  {row['indicator']:<16}{row['r']:>+8.3f}{rb:>8}{row['agree%']:>7.1f}%  {row['kind']:<6} {v}{flag}")
    best = res.iloc[0]
    print(f"\n  strongest: {best['indicator']} r={best['r']:+.3f} ({best['kind']}). "
          f"|r|>=0.05 anywhere? {'YES' if (res['r'].abs()>=0.05).any() else 'NO — nothing clears the meaningful bar'}.")


if __name__ == "__main__":
    main()
