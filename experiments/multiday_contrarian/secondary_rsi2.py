#!/usr/bin/env python3
"""
secondary_rsi2.py — Task A5 secondary (b): D1 RSI(2) mean-reversion (recorded prior:
"+4.7 p/trade, weak", project_indicator_screen.md memory entry).

Classic rule (documented, one config, no tuning):
  RSI(2) < 10 at a D1 close  -> go LONG.
  RSI(2) > 90 at a D1 close  -> go SHORT.
  Exit: the first subsequent D1 bar whose close moves in the trader's favor-direction
    ("next up close", i.e. close[j] > close[j-1], for a long; "next down close",
    close[j] < close[j-1], for a short), OR a 5-D1-bar time cap, whichever comes first.
  One position per pair at a time (FIFO, matches the primary harness's convention).

Documented implementation choice: RSI uses WILDER smoothing (alpha=1/period), period=2,
for consistency with this codebase's other indicators (harness.py's own ATR is Wilder).
The classic Connors RSI(2) as originally popularized sometimes uses a plain running
average of the last 2 gains/losses instead of Wilder smoothing — that variant is NOT
implemented here; this is a documented deviation, not a bug, per the task's "document"
instruction.

Entry/exit fill (R3a, matches primary harness convention): next M5 bar's OPEN after the
triggering D1 bar's close, mid price, spread deducted explicitly and separately.
D1 bars from bars.m5_to_d1 (NY 17:00 anchor, IS-only via is_data.load_pair_is).

Writes results/secondary_rsi2_trades.csv + secondary_rsi2_summary.json.
"""
import argparse
import gc
import json
import os

import numpy as np
import pandas as pd

from bars import m5_to_d1
from carry_model import carry_pips, pip_of
from is_data import IS_END, PAIRS, load_pair_is, to_utc

RSI_N = 2
RSI_LOW = 10.0
RSI_HIGH = 90.0
CAP_DAYS = 5


def wilder_rsi(closes, n=RSI_N):
    """Wilder-smoothed RSI over a 1-D array of closes. Returns array same length, NaN for
    the first n+1 bars (insufficient history)."""
    closes = np.asarray(closes, dtype=float)
    diffs = np.diff(closes)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    rsi = np.full(len(closes), np.nan)
    if len(diffs) < n:
        return rsi
    alpha = 1.0 / n
    avg_gain = np.mean(gains[:n])
    avg_loss = np.mean(losses[:n])
    for i in range(n, len(diffs)):
        if i > n:
            avg_gain = avg_gain + alpha * (gains[i] - avg_gain)
            avg_loss = avg_loss + alpha * (losses[i] - avg_loss)
        rs = avg_gain / avg_loss if avg_loss > 0 else np.inf
        rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def simulate_rsi2(pair, m5_df):
    pip = pip_of(pair)
    d1 = m5_to_d1(m5_df).reset_index(drop=True)
    n = len(d1)
    if n < RSI_N + 3:
        return []
    rsi = wilder_rsi(d1["close"].values, n=RSI_N)

    m5 = m5_df.sort_values("timestamp").reset_index(drop=True)
    m5_ts = m5["timestamp"].values
    m5_open = m5["open"].values
    m5_bid = m5["bid_c"].values
    m5_ask = m5["ask_c"].values

    d1_ts = d1["timestamp"].values
    d1_close = d1["close"].values

    def next_m5_open_after(d1_close_ts):
        pos = int(np.searchsorted(m5_ts, d1_close_ts, side="left"))
        if pos >= len(m5_ts):
            return None
        return pos

    trades = []
    i = 0
    while i < n:
        if np.isnan(rsi[i]):
            i += 1
            continue
        direction = 0
        if rsi[i] < RSI_LOW:
            direction = 1
        elif rsi[i] > RSI_HIGH:
            direction = -1
        if direction == 0:
            i += 1
            continue

        signal_close_ts = pd.Timestamp(d1_ts[i]) + pd.Timedelta(hours=24)
        entry_pos = next_m5_open_after(signal_close_ts.to_datetime64())
        if entry_pos is None:
            break  # data ends before we can enter
        entry_ts = m5_ts[entry_pos]
        entry_px = float(m5_open[entry_pos])
        entry_spread_pips = float(m5_ask[entry_pos] - m5_bid[entry_pos]) / pip

        exit_j = None
        exit_reason = None
        cap_j = min(i + CAP_DAYS, n - 1)
        for j in range(i + 1, min(i + CAP_DAYS, n - 1) + 1):
            up = d1_close[j] > d1_close[j - 1]
            down = d1_close[j] < d1_close[j - 1]
            if direction > 0 and up:
                exit_j = j
                exit_reason = "up_close"
                break
            if direction < 0 and down:
                exit_j = j
                exit_reason = "down_close"
                break
        if exit_j is None:
            exit_j = cap_j
            exit_reason = "timecap" if (i + CAP_DAYS) <= n - 1 else "data_end"

        exit_close_ts = pd.Timestamp(d1_ts[exit_j]) + pd.Timedelta(hours=24)
        exit_pos = next_m5_open_after(exit_close_ts.to_datetime64())
        if exit_pos is None:
            # data truncated before exit fill is available: fall back to last M5 bar
            exit_pos = len(m5_ts) - 1
            exit_reason = "data_end"
        exit_ts = m5_ts[exit_pos]
        exit_px = float(m5_open[exit_pos])
        exit_spread_pips = float(m5_ask[exit_pos] - m5_bid[exit_pos]) / pip

        gross_pips = direction * (exit_px - entry_px) / pip
        spread_rt_pips = (entry_spread_pips + exit_spread_pips) / 2.0
        carry = carry_pips(pair, direction, entry_ts, exit_ts, markup_mult=1.0)
        net_base = gross_pips - spread_rt_pips + carry
        carry2 = carry_pips(pair, direction, entry_ts, exit_ts, markup_mult=2.0)
        net_spread1p5 = gross_pips - spread_rt_pips * 1.5 + carry
        net_carry2p0 = gross_pips - spread_rt_pips + carry2

        trades.append({
            "pair": pair, "direction": direction, "rsi_at_signal": float(rsi[i]),
            "signal_ts": to_utc(d1_ts[i]), "entry_ts": to_utc(entry_ts), "entry_px": entry_px,
            "exit_ts": to_utc(exit_ts), "exit_px": exit_px, "exit_reason": exit_reason,
            "gross_pips": gross_pips, "spread_rt_pips": spread_rt_pips, "carry_pips": carry,
            "net_base": net_base, "net_spread1p5": net_spread1p5, "net_carry2p0": net_carry2p0,
        })

        # FIFO: resume signal search on the D1 bar after the exit bar.
        i = exit_j + 1

    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_trades = []
    for pair in PAIRS:
        df = load_pair_is(pair, args.data_dir)
        trades = simulate_rsi2(pair, df)
        all_trades.extend(trades)
        n = len(trades)
        mean_net = np.mean([t["net_base"] for t in trades]) if n else float("nan")
        print(f"[{pair}] rsi2: n={n} mean_net_base={mean_net:+.3f}p", flush=True)
        del df
        gc.collect()

    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(os.path.join(args.out_dir, "secondary_rsi2_trades.csv"), index=False)

    assert pd.Timestamp(trades_df["entry_ts"].max()) < IS_END, "OOS LEAK in rsi2 entry_ts"

    summary = {
        "n": int(len(trades_df)),
        "wr": float((trades_df["net_base"] > 0).mean()) if len(trades_df) else float("nan"),
        "mean_gross_pips": float(trades_df["gross_pips"].mean()) if len(trades_df) else float("nan"),
        "mean_net_base": float(trades_df["net_base"].mean()) if len(trades_df) else float("nan"),
        "mean_net_spread1p5": float(trades_df["net_spread1p5"].mean()) if len(trades_df) else float("nan"),
        "mean_net_carry2p0": float(trades_df["net_carry2p0"].mean()) if len(trades_df) else float("nan"),
        "n_pairs_gross_positive": int(
            trades_df.groupby("pair")["gross_pips"].mean().gt(0).sum()
        ) if len(trades_df) else 0,
    }
    with open(os.path.join(args.out_dir, "secondary_rsi2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
