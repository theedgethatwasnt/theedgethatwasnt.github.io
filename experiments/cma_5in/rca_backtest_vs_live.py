#!/usr/bin/env python3
"""RCA: run IronNet V3 H1 CHF_JPY genome on Apr 9-10 backtest window.

If backtest agrees with live (-144 pips), the problem is real regime change.
If backtest is positive, problem is live execution/data.
"""
import sys, pickle, math
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.fast_eval import extract_network, _activate
from lib.asi_indicator import compute_asi_mc
import neat

# V3 H1 CHF_JPY genome
GENOME_PATH = PROJECT_ROOT / "research/experiments/asi_mc/results/ironnet_h1/iron_v3_H1_CHF_JPY_s42_best.pkl"
CONFIG_PATH = PROJECT_ROOT / "research/experiments/asi_mc/neat_config_4in_3out.ini"

# NEAT activation functions (both naming conventions for pickle compat)
def gauss_activation(x): return math.exp(-x * x)
def sin_activation(x): return math.sin(x)
def cos_activation(x): return math.cos(x)
def tanh_activation(x): return math.tanh(x)
# Aliases (the training script used these names)
def gauss(x): return math.exp(-x * x)
def sin(x): return math.sin(x)
def cos(x): return math.cos(x)
def tanh(x): return math.tanh(x)
_gauss_activation = gauss_activation
_sin_activation = sin_activation
_cos_activation = cos_activation
_tanh_activation = tanh_activation


@njit(cache=True)
def _compute_er_norm(closes, window=60):
    n = len(closes)
    out = np.zeros(n)
    hp = np.pi / 2.0
    for i in range(window, n):
        net = abs(closes[i] - closes[i - window])
        path = 0.0
        for j in range(i - window + 1, i + 1):
            path += abs(closes[j] - closes[j - 1])
        if path > 0.0:
            out[i] = np.arctan((net / path) / 0.3) / hp
    return out


@njit(cache=True)
def run_v3_h1(inputs_2d, mid_close, pip, spread_pips, max_hold_h,
              n_inputs, n_eval, total_values,
              node_bias, node_response, node_act,
              conn_from, conn_to, conn_weight, output_indices):
    values = np.zeros(total_values)
    n = inputs_2d.shape[1]
    max_hold_bars = max_hold_h  # H1 = 1 bar per hour
    trades = np.zeros(n)
    nt = 0
    nl = 0
    ns = 0
    position = 0
    entry_price = 0.0
    entry_bar = 0
    for i in range(10, n - 1):
        if position != 0:
            pnl_pips = (mid_close[i] - entry_price) / pip * position - spread_pips
        else:
            pnl_pips = 0.0
        values[0] = inputs_2d[0, i]  # asi_mc_d
        values[1] = inputs_2d[1, i]  # asi_mc_dd
        values[2] = inputs_2d[2, i]  # er_norm
        values[3] = np.tanh(pnl_pips / 20.0)  # upnl
        _activate(values, n_inputs, n_eval, node_bias, node_response, node_act,
                  conn_from, conn_to, conn_weight)
        ob = values[output_indices[0]]
        os_ = values[output_indices[1]]
        of = values[output_indices[2]]
        if position != 0 and (i - entry_bar) >= max_hold_bars:
            pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
            trades[nt] = pnl; nt += 1
            position = 0
            continue
        if position == 0:
            if ob > os_ and ob > of:
                position = 1; entry_price = mid_close[i]; entry_bar = i; nl += 1
            elif os_ > ob and os_ > of:
                position = -1; entry_price = mid_close[i]; entry_bar = i; ns += 1
        else:
            close = False; new_pos = 0
            if of > ob and of > os_: close = True
            elif position == 1 and os_ > ob and os_ > of: close = True; new_pos = -1
            elif position == -1 and ob > os_ and ob > of: close = True; new_pos = 1
            if close:
                pnl = (mid_close[i] - entry_price) / pip * position - spread_pips
                trades[nt] = pnl; nt += 1
                position = new_pos
                if new_pos != 0:
                    entry_price = mid_close[i]; entry_bar = i
                    if new_pos == 1: nl += 1
                    else: ns += 1
    return trades[:nt], nl, ns


def load_v3_h1_data_range(pair, start_ts, end_ts):
    """Load M5 OHLC, compute indicators at M5, resample to H1, slice to date range."""
    path = PROJECT_ROOT / "data" / "m5_ohlc" / f"{pair}_M5.parquet"
    df = pd.read_parquet(path)
    o, h, l, c = [df[x].values.astype(np.float64) for x in ["open","high","low","close"]]
    mc_d, mc_dd = compute_asi_mc(o, h, l, c, len(c))
    er = _compute_er_norm(c, window=60)

    ts = pd.to_datetime(df["timestamp"])
    frame = pd.DataFrame({
        "timestamp": ts,
        "mid": c,
        "asi_mc_d": mc_d,
        "asi_mc_dd": mc_dd,
        "er_norm": er,
    }).set_index("timestamp")

    # Resample to H1 via .last() (matches live curator)
    h1 = frame.resample("1h").last().dropna()

    # Filter date range (make timezone match)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tz is None else start_ts
    end_ts = end_ts.tz_localize("UTC") if end_ts.tz is None else end_ts
    h1 = h1[(h1.index >= start_ts) & (h1.index <= end_ts)]

    return h1


def main():
    # Load V3 H1 CHF_JPY genome
    print(f"Loading genome: {GENOME_PATH}")
    with open(GENOME_PATH, "rb") as f:
        saved = pickle.load(f)
    if isinstance(saved, dict) and "genome" in saved:
        genome = saved["genome"]
        config_from_pkl = saved.get("config")
    else:
        genome = saved
        config_from_pkl = None

    if config_from_pkl is not None:
        config = config_from_pkl
    else:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             str(CONFIG_PATH))
        for name, fn in [('gauss', gauss_activation), ('sin', sin_activation),
                         ('cos', cos_activation), ('tanh', tanh_activation)]:
            try: config.genome_config.add_activation(name, fn)
            except: pass

    net_tuple = extract_network(genome, config)
    # extract_network returns: n_inputs, n_outputs, n_eval, total_values,
    #                          node_bias, node_response, node_act,
    #                          conn_from, conn_to, conn_weight, output_indices
    n_inputs, n_outputs, n_eval, total_values, node_bias, node_response, node_act, \
        conn_from, conn_to, conn_weight, output_indices = net_tuple

    print(f"Network: {n_inputs} inputs, {total_values-n_inputs} non-input nodes")

    # ── Test 1: Full OOS (what backtest said was +73 p/day) ──
    print("\n=== TEST 1: Full OOS (last 30% of data) ===")
    data_full = load_v3_h1_data_range("CHF_JPY",
                                       pd.Timestamp("2020-01-01"),
                                       pd.Timestamp("2030-01-01"))
    split_idx = int(len(data_full) * 0.7)
    oos = data_full.iloc[split_idx:]
    print(f"OOS bars: {len(oos)}")
    inputs = np.stack([oos["asi_mc_d"].values, oos["asi_mc_dd"].values,
                       oos["er_norm"].values], axis=0)
    mid = oos["mid"].values
    pip = 0.01
    spread = 3.5
    trades, nl, ns = run_v3_h1(
        inputs, mid, pip, spread, 17,
        n_inputs, n_eval, total_values,
        node_bias, node_response, node_act,
        conn_from, conn_to, conn_weight, output_indices)
    days = len(oos) / 24.0
    print(f"OOS: {len(trades)} trades, L/S={nl}/{ns}, total={trades.sum():.1f} pips, "
          f"avg={trades.mean():.2f}, p/d={trades.sum()/days:.1f}")

    # ── Test 2: Regime check — rolling recent windows ──
    print("\n=== TEST 2: Recent backtest windows — is the genome still profitable? ===")
    for days in [7, 14, 30, 90]:
        hours = days * 24
        if len(data_full) < hours:
            continue
        window = data_full.iloc[-hours:]
        inputs_w = np.stack([window["asi_mc_d"].values, window["asi_mc_dd"].values,
                             window["er_norm"].values], axis=0)
        mid_w = window["mid"].values
        trades_w, nl_w, ns_w = run_v3_h1(
            inputs_w, mid_w, 0.01, 3.5, 17,
            n_inputs, n_eval, total_values,
            node_bias, node_response, node_act,
            conn_from, conn_to, conn_weight, output_indices)
        days_actual = len(window) / 24.0
        if len(trades_w) > 0:
            print(f"  Last {days:>3} days: {len(trades_w)} trades, "
                  f"L/S={nl_w}/{ns_w}, total={trades_w.sum():+.1f}p, "
                  f"avg={trades_w.mean():+.2f}, p/d={trades_w.sum()/days_actual:+.1f}")
        else:
            print(f"  Last {days:>3} days: 0 trades")

    print("\n=== TEST 2b: Last 60 H1 bars (for comparison to live ~30 H1 bars) ===")
    live_start = data_full.index[-60]
    live_end = data_full.index[-1]
    print(f"  Backtest last bar: {live_end}")
    print(f"  Live actually started: 2026-04-09 13:00 UTC (later — no direct overlap)")
    live = data_full[(data_full.index >= live_start) & (data_full.index <= live_end)]
    print(f"Live window bars: {len(live)}")
    if len(live) > 10:
        inputs2 = np.stack([live["asi_mc_d"].values, live["asi_mc_dd"].values,
                           live["er_norm"].values], axis=0)
        mid2 = live["mid"].values
        trades2, nl2, ns2 = run_v3_h1(
            inputs2, mid2, pip, spread, 17,
            n_inputs, n_eval, total_values,
            node_bias, node_response, node_act,
            conn_from, conn_to, conn_weight, output_indices)
        days2 = len(live) / 24.0
        print(f"Live window backtest: {len(trades2)} trades, L/S={nl2}/{ns2}, "
              f"total={trades2.sum():.1f} pips, avg={trades2.mean() if len(trades2)>0 else 0:.2f}")
        print(f"  Per day: {trades2.sum()/max(days2,0.1):.1f} pips/day")

    # ── Test 3: Check indicator values at live trade timestamps ──
    print("\n=== TEST 3: Indicator values at trade entry timestamps ===")
    live_timestamps = [
        ("2026-04-09 13:19:36", 201.221, "long", 14.0),   # +14p winner
        ("2026-04-09 13:53:27", 201.478, "long", -9.9),
        ("2026-04-09 14:16:53", 201.571, "long", -18.4),
        ("2026-04-09 21:14:59", 201.392, "long", -20.9),  # MAE=0 bug trade
        ("2026-04-09 21:21:05", 201.276, "long", -26.5),
    ]
    for ts_str, live_entry, direction, live_pnl in live_timestamps:
        ts = pd.Timestamp(ts_str)
        # Find nearest H1 bar (rounded down)
        h1_bar = ts.floor("h")
        if h1_bar in data_full.index:
            row = data_full.loc[h1_bar]
            print(f"  {ts_str} ({direction}, live_pnl={live_pnl:+.1f}): H1 bar={h1_bar}, "
                  f"mid={row['mid']:.3f} (live entry={live_entry:.3f}, "
                  f"diff={(row['mid']-live_entry)*100:+.1f}p), "
                  f"mc_d={row['asi_mc_d']:+.3f}, mc_dd={row['asi_mc_dd']:+.3f}, "
                  f"er={row['er_norm']:.3f}")
        else:
            print(f"  {ts_str}: H1 bar not found")


if __name__ == "__main__":
    main()
