"""Per-pair signal/label export (master doc Part VI steps 1-3).

Per M5 boundary t: feature window = the 60 S5 slots of [t-300s, t), closes
forward-filled onto the grid (no-tick = unchanged price), >=30 real bars
required. Entry = close of first S5 bar in [t+5s, t+30s]. Labels for all
TP x horizon x direction combos on the mid S5 path strictly after entry.
"""
import gc
import numpy as np
import pandas as pd
from features import compute_regime_features, FEATURE_NAMES
from labeler import label_trade

PIP_SIZE = {p: (0.01 if p.endswith("_JPY") else 0.0001) for p in
            ["AUD_JPY", "AUD_USD", "CAD_JPY", "CHF_JPY", "EUR_GBP", "EUR_JPY",
             "EUR_USD", "GBP_JPY", "GBP_USD", "NZD_JPY", "NZD_USD", "USD_JPY"]}
TP_LEVELS = {"t15": 1.5, "t18": 1.8, "t32": 3.2, "t40": 4.0}   # PREREGISTRATION
HORIZONS = {"h2": 2, "h4": 4}                                   # M5 bars
MIN_REAL_BARS = 30


def _session(hour):
    if 22 <= hour or hour < 7:   return "asia"
    if 7 <= hour < 12:           return "london"
    if 12 <= hour < 21:          return "ny"
    return "other"


def build_signals_for_slice(df, pair):
    pip = PIP_SIZE[pair]
    ts = df["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    spread = ((df["ask_c"].values - df["bid_c"].values) / pip).astype(np.float64)

    # M5 boundaries with at least one prior-window bar and one entry bar
    t0 = (ts[0] // 300 + 2) * 300
    t1 = (ts[-1] // 300 - 1) * 300
    boundaries = np.arange(t0, t1, 300, dtype=np.int64)
    pos = np.searchsorted(ts, boundaries)              # first bar >= boundary

    rows = []
    for b_i in range(len(boundaries)):
        t = boundaries[b_i]
        hi_idx = pos[b_i]
        lo_idx = np.searchsorted(ts, t - 300)
        w = slice(lo_idx, hi_idx)                      # bars in [t-300, t)
        n_real = hi_idx - lo_idx
        if n_real < MIN_REAL_BARS:
            continue
        # forward-fill closes onto the 60-slot grid
        grid = np.empty(60)
        wi = lo_idx
        last = close[lo_idx]                            # first real bar seeds grid
        for s in range(60):
            slot_end = t - 300 + (s + 1) * 5
            while wi < hi_idx and ts[wi] < slot_end:
                last = close[wi]; wi += 1
            grid[s] = last
        feats = compute_regime_features(grid, pip)

        # entry: first S5 bar stamped in [t+5, t+30]
        e_idx = np.searchsorted(ts, t + 5)
        if e_idx >= len(ts) or ts[e_idx] > t + 30:
            continue
        entry = close[e_idx]
        prior_close = grid[-1]
        row = {
            "ts": df["timestamp"].iloc[e_idx], "pair": pair,
            "n_real_bars": n_real, "spread_pips": spread[e_idx],
            "close_to_entry_pips": (entry - prior_close) / pip,
            "session": _session((t // 3600) % 24),
        }
        for name, val in zip(FEATURE_NAMES, feats):
            row[name] = val
        # labels: scan bars strictly after entry, within each horizon
        for hk, hbars in HORIZONS.items():
            end = np.searchsorted(ts, t + hbars * 300, side="right")
            for tk, tp in TP_LEVELS.items():
                for dk, d in (("long", 1), ("short", -1)):
                    lab, ex, held, mfe, mae = label_trade(
                        high, low, close, e_idx + 1, end, entry, d, tp, 2 * tp, pip)
                    pre = f"{tk}_{hk}_{dk}"
                    row[f"{pre}_label"] = lab
                    row[f"{pre}_exit_pips"] = ex
                    row[f"{pre}_bars_held"] = held
                    row[f"{pre}_mfe"] = mfe
                    row[f"{pre}_mae"] = mae
        rows.append(row)
    return pd.DataFrame(rows)


def run_pair(pair, data_dir="../../../data/s5_ohlc", out_dir="signals"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(f"{data_dir}/{pair}_S5_BA.parquet",
                         columns=["timestamp", "open", "high", "low", "close",
                                  "bid_c", "ask_c"])
    out = build_signals_for_slice(df, pair)
    path = f"{out_dir}/{pair}_signals.parquet"
    out.to_parquet(path, index=False)
    print(f"{pair}: {len(out)} signals -> {path}", flush=True)
    del df, out
    gc.collect()
    return path


if __name__ == "__main__":
    import sys
    for p in (sys.argv[1:] or list(PIP_SIZE)):
        run_pair(p)
