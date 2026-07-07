#!/usr/bin/env python3
"""
run_is_battery.py — Task A5 primary IS battery (multi-day contrarian program).

12 pairs x 3 arms (signal / coin seed=20260706 / continuation), IS window only
(load_pair_is() enforces the hard OOS guard — see is_data.py). One pair fully
processed (all 3 arms) before moving to the next; `del` + `gc.collect()` between
pairs (CLAUDE.md memory-safety default).

Cost variants: base (spread_mult=1.0, markup_mult=1.0), spread1.5 (spread_mult=1.5),
carry2.0 (markup_mult=2.0). The two sensitivity variants are DERIVED analytically from
the base trade, not re-simulated: harness.simulate_pair computes
    net_pips = gross_pips - spread_rt_pips + carry_pips
with spread_rt_pips exactly linear in spread_mult (see harness.py's
`spread_rt_pips = (...)/2.0 * spread_mult`) and carry_pips a pure function of
(pair, direction, entry_ts, exit_ts, markup_mult) with no dependence on spread_mult or
on any other simulation state — so re-deriving net at a different mult from the stored
gross_pips/spread_rt_pips/entry_ts/exit_ts is bit-for-bit identical to re-running
simulate_pair with that mult, at 1/3 the compute. (Verified by spot re-run comparison
below `if __name__` for one pair before trusting this for the full battery.)

Usage (on Hetzner):
  /root/venv/bin/python3 run_is_battery.py \
      --data-dir /root/multiday/data/m5_ba --out-dir results
"""
import argparse
import gc
import os

import numpy as np
import pandas as pd

from carry_model import carry_pips
from harness import ARMS, DEFAULT_SEED, simulate_pair
from is_data import IS_END, PAIRS, load_pair_is, to_utc


def derive_cost_variants(trades):
    out = []
    for t in trades:
        t = dict(t)
        # Re-attach UTC tz (harness's M5 arrays strip it to naive datetime64 — see is_data.to_utc)
        # BEFORE any downstream comparison against IS_END or CSV round-trip.
        t["signal_ts"] = to_utc(t["signal_ts"])
        t["entry_ts"] = to_utc(t["entry_ts"])
        t["exit_ts"] = to_utc(t["exit_ts"])
        t["net_base"] = t["net_pips"]
        t["net_spread1p5"] = t["gross_pips"] - t["spread_rt_pips"] * 1.5 + t["carry_pips"]
        carry2 = carry_pips(t["pair"], t["direction"], t["entry_ts"], t["exit_ts"], markup_mult=2.0)
        t["net_carry2p0"] = t["gross_pips"] - t["spread_rt_pips"] + carry2
        out.append(t)
    return out


def run_battery(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for pair in PAIRS:
        print(f"[{pair}] loading IS data...", flush=True)
        df = load_pair_is(pair, data_dir)
        print(f"[{pair}] {len(df)} M5 IS rows, {df['timestamp'].min()} -> {df['timestamp'].max()}", flush=True)
        for arm in ARMS:
            trades = simulate_pair(pair, df, arm=arm, seed=DEFAULT_SEED, spread_mult=1.0, markup_mult=1.0)
            trades = derive_cost_variants(trades)
            for t in trades:
                t["arm"] = arm
            all_rows.extend(trades)
            n = len(trades)
            mean_net = np.mean([t["net_base"] for t in trades]) if n else float("nan")
            print(f"[{pair}] arm={arm}: n={n} mean_net_base={mean_net:+.3f}p", flush=True)
        del df
        gc.collect()

    trades_df = pd.DataFrame(all_rows)
    out_path = os.path.join(out_dir, "is_battery_trades.csv")
    trades_df.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(trades_df)} rows", flush=True)

    # Belt-and-suspenders OOS guard on the assembled output (should be trivially true by
    # construction, since every timestamp used descends from load_pair_is()'s filtered arrays).
    assert pd.Timestamp(trades_df["entry_ts"].max()) < IS_END, "OOS LEAK in entry_ts"
    assert pd.Timestamp(trades_df["exit_ts"].max()) < IS_END + pd.Timedelta(minutes=10), (
        "exit_ts unexpectedly far past IS_END — investigate before trusting downstream gates"
    )
    return trades_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--verify-derivation", action="store_true",
                     help="spot-check derive_cost_variants against a real re-simulation for one pair")
    args = ap.parse_args()

    if args.verify_derivation:
        df = load_pair_is("EUR_USD", args.data_dir)
        base = simulate_pair("EUR_USD", df, arm="signal", seed=DEFAULT_SEED, spread_mult=1.0, markup_mult=1.0)
        derived = derive_cost_variants(base)
        resim_spread = simulate_pair("EUR_USD", df, arm="signal", seed=DEFAULT_SEED, spread_mult=1.5, markup_mult=1.0)
        resim_carry = simulate_pair("EUR_USD", df, arm="signal", seed=DEFAULT_SEED, spread_mult=1.0, markup_mult=2.0)
        assert len(derived) == len(resim_spread) == len(resim_carry)
        for d, rs, rc in zip(derived, resim_spread, resim_carry):
            assert abs(d["net_spread1p5"] - rs["net_pips"]) < 1e-9, (d["net_spread1p5"], rs["net_pips"])
            assert abs(d["net_carry2p0"] - rc["net_pips"]) < 1e-9, (d["net_carry2p0"], rc["net_pips"])
        print(f"derivation verified bit-identical on {len(derived)} EUR_USD signal trades")

    run_battery(args.data_dir, args.out_dir)
