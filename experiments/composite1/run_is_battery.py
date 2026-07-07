#!/usr/bin/env python3
"""run_is_battery.py — Composite 1: primary IS battery (IS ONLY -- OOS never touched, per
PREREGISTRATION.md's inherited seal / R8). Loads the 7 direct-USD pairs' D1 price panel
(is_data.load_pair_is -- hard-truncated at load time) and the COT weekly panel
(is_data.restrict_cot_to_is), detects Axis-1 signals per pair, builds the FIFO trade
calendar, evaluates the Axis-3 vote (real + 200 block-preserving-by-currency permutations),
and writes the 5 arms to results/is_battery_trades.csv + results/shuffled_null.csv +
results/data_coverage.json.

Usage (on Hetzner):
  COT_CODE_DIR=/root/work/code_cot COT_D1_DATA_DIR=/root/work/data/d1_deep_ba \\
    /root/venv/bin/python3 run_is_battery.py --out-dir results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import _paths
import axis3_vote as av
import is_data as isd
from simulate import detect_signals_and_trades, fifo_filter_natural

HERE = os.path.dirname(os.path.abspath(__file__))

PAIRS = ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "USD_CHF", "USD_CAD"]
COIN_SEED = 20260709
N_SHUFFLE = 200
SHUFFLE_SEED_BASE = 20260709


def trades_frame(trades, arm):
    rows = []
    for t in trades:
        row = dict(t)
        row["arm"] = arm
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-parquet", default=os.path.join(_paths.COT_CODE_DIR, "cot_weekly.parquet"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--markup-mult", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/7] loading COT weekly, restricting to IS...", flush=True)
    cot_full = pd.read_parquet(args.cot_parquet)
    cot_is = isd.restrict_cot_to_is(cot_full)
    print(f"      IS COT rows: {len(cot_is)} ({cot_is['report_date'].min().date()} -> "
          f"{cot_is['report_date'].max().date()})", flush=True)

    print("[2/7] loading 7 pairs' D1 (IS-bounded loader)...", flush=True)
    price_panel = {p: isd.load_pair_is(p) for p in PAIRS}
    for p, df in price_panel.items():
        print(f"      {p}: {len(df)} bars, {df.index.min().date()} -> {df.index.max().date()}", flush=True)

    calendar = None
    for df in price_panel.values():
        calendar = df.index if calendar is None else calendar.union(df.index)
    calendar = calendar.sort_values()

    print("[3/7] building Axis-3 z panel (verbatim cot_signal/release_lag, IS-only)...", flush=True)
    ccy_series = av.build_currency_z_series(cot_is, calendar)
    zlookup = av.make_zlookup(ccy_series)
    for ccy, g in ccy_series.items():
        print(f"      {ccy}: {len(g)} released z obs, {g['action_date'].min()} -> {g['action_date'].max()}",
              flush=True)

    print("[4/7] Axis-1 detection + FIFO trade calendar per pair...", flush=True)
    per_pair_kept = {}
    n_all_signals = 0
    n_fifo = 0
    for p, df in price_panel.items():
        raw = detect_signals_and_trades(p, df, markup_mult=args.markup_mult)
        kept = fifo_filter_natural(raw)
        kept_is = [r for r in kept if r["trade_natural"]["entry_ts"] < isd.AXIS1_IS_ENTRY_CUTOFF]
        for r in kept_is:
            isd.assert_trade_is_is(r["trade_natural"]["entry_ts"], r["trade_natural"]["exit_ts"])
            isd.assert_trade_is_is(r["trade_opposite"]["entry_ts"], r["trade_opposite"]["exit_ts"])
        per_pair_kept[p] = kept_is
        n_all_signals += len(raw)
        n_fifo += len(kept)
        print(f"      {p}: {len(raw)} raw signals -> {len(kept)} FIFO -> {len(kept_is)} IS", flush=True)

    print("[5/7] building composite / axis1-alone / coin arms...", flush=True)
    composite_trades, axis1_trades, coin_trades = [], [], []
    coin_rng = np.random.default_rng(COIN_SEED)
    all_kept_sorted = sorted(
        [(p, r) for p, recs in per_pair_kept.items() for r in recs],
        key=lambda pr: (pr[1]["trade_natural"]["entry_ts"], pr[0]),
    )
    for p, r in all_kept_sorted:
        axis1_trades.append(r["trade_natural"])
        gate = av.composite_gate(r["natural_direction"], p, r["trade_natural"]["entry_ts"], zlookup)
        r["composite_gate"] = gate
        if gate:
            composite_trades.append(r["trade_natural"])
            coin_dir = 1 if coin_rng.random() < 0.5 else -1
            coin_trades.append(r["trade_natural"] if coin_dir == r["natural_direction"] else r["trade_opposite"])

    print(f"      axis1_alone n={len(axis1_trades)}  composite n={len(composite_trades)}  "
          f"coin n={len(coin_trades)}", flush=True)

    print(f"[6/7] shuffled-positioning control ({N_SHUFFLE} block-preserving permutations)...", flush=True)
    shuffle_means = np.empty(N_SHUFFLE)
    shuffle_ns = np.empty(N_SHUFFLE, dtype=int)
    for rep in range(N_SHUFFLE):
        perm_series = av.permute_currency_z_series(ccy_series, seed=SHUFFLE_SEED_BASE + rep)
        perm_lookup = av.make_zlookup(perm_series)
        vals = []
        for p, r in all_kept_sorted:
            if av.composite_gate(r["natural_direction"], p, r["trade_natural"]["entry_ts"], perm_lookup):
                vals.append(r["trade_natural"]["net_pips"])
        shuffle_means[rep] = np.mean(vals) if vals else np.nan
        shuffle_ns[rep] = len(vals)
    print(f"      shuffled null: mean={np.nanmean(shuffle_means):+.3f}p "
          f"p95={np.nanpercentile(shuffle_means, 95):+.3f}p (avg n={shuffle_ns.mean():.1f})", flush=True)

    print("[7/7] writing results...", flush=True)
    all_trades = pd.concat([
        trades_frame(composite_trades, "composite"),
        trades_frame(axis1_trades, "axis1_alone"),
        trades_frame(coin_trades, "coin"),
    ], ignore_index=True)
    all_trades.to_csv(os.path.join(args.out_dir, "is_battery_trades.csv"), index=False)

    pd.DataFrame({"replicate": range(N_SHUFFLE), "mean_net_pips": shuffle_means, "n_trades": shuffle_ns}) \
        .to_csv(os.path.join(args.out_dir, "shuffled_null.csv"), index=False)

    # belt-and-suspenders re-check on the assembled output
    assert pd.to_datetime(all_trades["entry_ts"]).max() < isd.AXIS1_IS_ENTRY_CUTOFF, "OOS LEAK in assembled output"

    coverage = {
        "pairs": PAIRS,
        "is_cot_rows": int(len(cot_is)),
        "is_cot_span": [str(cot_is["report_date"].min().date()), str(cot_is["report_date"].max().date())],
        "axis1_is_entry_cutoff": str(isd.AXIS1_IS_ENTRY_CUTOFF),
        "data_load_ceiling": str(isd.DATA_LOAD_CEILING),
        "n_raw_signals_all_pairs": n_all_signals,
        "n_fifo_all_pairs_all_time": n_fifo,
        "n_axis1_alone_is_trades": len(axis1_trades),
        "n_composite_is_trades": len(composite_trades),
        "n_coin_is_trades": len(coin_trades),
        "n_shuffle_replicates": N_SHUFFLE,
        "shuffle_null_mean": float(np.nanmean(shuffle_means)),
        "shuffle_null_p95": float(np.nanpercentile(shuffle_means, 95)),
        "shuffle_null_avg_n": float(shuffle_ns.mean()),
        "markup_mult": args.markup_mult,
    }
    with open(os.path.join(args.out_dir, "data_coverage.json"), "w") as f:
        json.dump(coverage, f, indent=2, default=str)
    print(json.dumps(coverage, indent=2, default=str))


if __name__ == "__main__":
    main()
