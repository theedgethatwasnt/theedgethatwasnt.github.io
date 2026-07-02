#!/usr/bin/env python3
"""
Indicator swap-in sweep for CMA-NN v3_plus.

For each candidate market indicator, runs train_cma_v2.py with:
    --features v3_plus --extra-feature <candidate>
holding everything else fixed (V3 base + sin activation).

Reads OOS pps + IS pps from each result pickle and prints a ranked table.
Sends Telegram updates after each candidate.

Usage
-----
    python3 sweep_indicators.py --pair CHF_JPY --gens 200 --seed 42
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PROJECT_ROOT = SCRIPT_DIR.parents[2]

# Candidate indicators to swap into the V3 base. All exist in unified_indicators
# (or m5_slope is computed inline).
CANDIDATES = [
    "m5_slope",       # 12-bar M5 regression slope, arctan-norm
    "h1_slope",       # 3-bar H1 regression slope, arctan-norm  (SHAP #3)
    "tec_5",          # signed Kaufman ER 5-bar                  (SHAP #1)
    "bb_width",       # Bollinger band width / close             (SHAP #2)
    "stoch_d",        # Stochastic %D 14/3                       (SHAP #4)
    "macd_hist",      # MACD histogram / ATR                     (SHAP #5)
    "gap_norm",       # M5 gap / prev range                      (SHAP #6)
    "range_pos_30",   # 30-bar range position                    (SHAP #7)
    "aroon_osc",      # 25-bar Aroon oscillator                  (SHAP #9)
    "hl_price",       # higher-low price                         (SHAP #11)
]


def tg_send(text: str):
    try:
        import requests
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        if token and chat:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": text, "parse_mode": "HTML"},
                timeout=5,
            )
    except Exception:
        pass


def run_one(pair: str, extra: str, gens: int, seed: int, popsize: int,
            workers: int) -> dict:
    """Run train_cma_v2.py for one candidate, return parsed result dict."""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "train_cma_v2.py"),
        "--pair", pair,
        "--seed", str(seed),
        "--gens", str(gens),
        "--features", "v3_plus",
        "--extra-feature", extra,
        "--fixed-activation", "sin",
        "--popsize", str(popsize),
        "--workers", str(workers),
        "--label", "sweep",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                          text=True, timeout=900)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        return {
            "extra": extra, "ok": False, "elapsed": elapsed,
            "error": proc.stderr.splitlines()[-1] if proc.stderr else "unknown",
        }

    pkl_path = RESULTS_DIR / f"sweep_v3_plus_{extra}_{pair}_s{seed}_best.pkl"
    if not pkl_path.exists():
        return {"extra": extra, "ok": False, "elapsed": elapsed,
                "error": f"missing {pkl_path}"}
    with open(pkl_path, "rb") as f:
        result = pickle.load(f)
    return {
        "extra": extra,
        "ok": True,
        "elapsed": elapsed,
        "is_pps": result["is_full"]["pips_per_day"],
        "oos_pps": result["oos"]["pips_per_day"],
        "is_trades": result["is_full"]["n_trades"],
        "oos_trades": result["oos"]["n_trades"],
        "dir_oos": result["oos"]["dir_ratio"],
        "passed_gates": result.get("passed_hard_gates", False),
        "min_chunk_pps": result.get("is_min_chunk_pps"),
    }


def fmt_row(r: dict) -> str:
    if not r["ok"]:
        return f"  {r['extra']:14s}  ERROR  {r.get('error','')}"
    gate = "✓" if r["passed_gates"] else "·"
    mc = f"{r['min_chunk_pps']:+.1f}" if r.get("min_chunk_pps") else "—"
    return (f"  {r['extra']:14s}  IS={r['is_pps']:+7.2f}  OOS={r['oos_pps']:+7.2f}  "
            f"min_ch={mc:>6}  trades={r['oos_trades']:5d}  "
            f"dir={r['dir_oos']:.2f}  gate={gate}  ({r['elapsed']:.0f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="CHF_JPY")
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidates", nargs="*", default=None,
                        help="Override the candidate list")
    args = parser.parse_args()

    candidates = args.candidates if args.candidates else CANDIDATES
    n = len(candidates)

    print(f"\n{'='*80}")
    print(f"  CMA-NN v3_plus indicator sweep — {args.pair}")
    print(f"  Base V3: mc_d_a + mc_dd_a + er_norm + <swap> + upnl/mae/mfe")
    print(f"  Activation: sin (fixed) | popsize={args.popsize} | gens={args.gens}")
    print(f"  Candidates ({n}): {', '.join(candidates)}")
    print(f"{'='*80}\n")

    tg_send(f"🔍 <b>CMA-NN sweep started</b>\n"
            f"Pair: {args.pair}\n"
            f"Candidates: {n}\n"
            f"Estimated time: ~{n * 3} min")

    results = []
    t_global = time.time()
    for i, extra in enumerate(candidates, 1):
        print(f"\n── [{i}/{n}] {extra} ──")
        r = run_one(args.pair, extra, args.gens, args.seed,
                    args.popsize, args.workers)
        results.append(r)
        print(fmt_row(r))

        if r["ok"]:
            tg_send(
                f"[{i}/{n}] <b>{extra}</b>\n"
                f"OOS: <b>{r['oos_pps']:+.2f}</b> p/d  |  "
                f"IS: {r['is_pps']:+.2f} p/d\n"
                f"min_chunk: {r.get('min_chunk_pps') or '—'}  "
                f"gate: {'✓' if r['passed_gates'] else '✗'}\n"
                f"trades: {r['oos_trades']}  dir: {r['dir_oos']:.2f}  "
                f"({r['elapsed']:.0f}s)"
            )
        else:
            tg_send(f"[{i}/{n}] <b>{extra}</b> ❌ ERROR: {r.get('error','')}")

    # ── Final ranked table ─────────────────────────────────────
    elapsed = time.time() - t_global
    print(f"\n{'='*80}")
    print(f"  Sweep complete — {elapsed:.0f}s total")
    print(f"{'='*80}\n")

    valid = [r for r in results if r["ok"]]
    valid.sort(key=lambda r: r["oos_pps"], reverse=True)

    print(f"Ranked by OOS pips/day:\n")
    print(f"  {'extra':14s}  {'IS pps':>9}  {'OOS pps':>9}  {'min_ch':>7}  "
          f"{'trades':>6}  {'dir':>4}  gate")
    print(f"  {'─'*14}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*6}  {'─'*4}  ────")
    for r in valid:
        gate = "✓" if r["passed_gates"] else "·"
        mc = f"{r['min_chunk_pps']:+.1f}" if r.get("min_chunk_pps") else "—"
        print(f"  {r['extra']:14s}  {r['is_pps']:+9.2f}  {r['oos_pps']:+9.2f}  "
              f"{mc:>7}  {r['oos_trades']:6d}  {r['dir_oos']:.2f}  {gate}")

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  {r['extra']:14s}  {r.get('error','')}")

    # Save results JSON
    out = RESULTS_DIR / f"sweep_v3_plus_{args.pair}_s{args.seed}_results.json"
    with open(out, "w") as f:
        json.dump({"pair": args.pair, "seed": args.seed,
                   "gens": args.gens, "elapsed": elapsed,
                   "results": results}, f, indent=2, default=str)
    print(f"\nSaved: {out}")

    # Telegram summary
    top3 = "\n".join(
        f"{i+1}. <b>{r['extra']}</b>: {r['oos_pps']:+.2f} p/d "
        f"(IS {r['is_pps']:+.2f})"
        for i, r in enumerate(valid[:3]))
    tg_send(
        f"✅ <b>Sweep complete</b> ({elapsed:.0f}s)\n"
        f"Pair: {args.pair}\n\n"
        f"<b>Top 3 by OOS p/day:</b>\n{top3}"
    )


if __name__ == "__main__":
    main()
