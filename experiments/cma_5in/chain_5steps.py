#!/usr/bin/env python3
"""
Chain orchestrator: runs experiments 1→5 in sequence with Telegram updates.

Steps
-----
1. macd_hist on EUR_GBP (single 200-gen run, V3 + macd_hist + sin)
2. Stack top-3 (V3 + macd_hist + range_pos_30 + m5_slope) on CHF_JPY
3. Full sweep on EUR_GBP (10 candidates)
4. macd_hist on all 12 pairs (apples-to-apples vs IronNet V3 grid)
5. Multi-seed robustness: 4 seeds × CHF_JPY V3+macd_hist

All sub-runs use --features v3_plus --fixed-activation sin --gens 200 --popsize 24.

Total expected runtime: ~85-100 minutes.

Results land in research/experiments/cma_5in/results/. Final summary printed
at the end and sent to Telegram.
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
TRAIN = SCRIPT_DIR / "train_cma_v2.py"
SWEEP = SCRIPT_DIR / "sweep_indicators.py"

ALL_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY",
    "CHF_JPY", "NZD_JPY", "NZD_USD", "EUR_GBP",
]

# IronNet V3 NEAT baselines (per CLAUDE.md, 2026-04-05) for comparison
IRONNET_V3 = {
    "CHF_JPY": 65.5, "EUR_JPY": 59.4, "GBP_USD": 47.9, "AUD_JPY": 47.8,
    "GBP_JPY": 41.3, "EUR_USD": 41.4, "CAD_JPY": 41.6, "USD_JPY": 37.6,
    "NZD_JPY": 25.2, "AUD_USD": 23.5, "NZD_USD": 18.3, "EUR_GBP": 14.3,
}


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


def run_train(pair: str, extras: list, seed: int, gens: int, popsize: int,
              workers: int, label: str) -> dict:
    """Run train_cma_v2.py with v3_plus + given extras. Returns parsed result."""
    cmd = [
        sys.executable, str(TRAIN),
        "--pair", pair,
        "--seed", str(seed),
        "--gens", str(gens),
        "--features", "v3_plus",
        "--extras", *extras,
        "--fixed-activation", "sin",
        "--popsize", str(popsize),
        "--workers", str(workers),
        "--label", label,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                          text=True, timeout=900)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"ok": False, "elapsed": elapsed,
                "error": (proc.stderr.splitlines()[-1] if proc.stderr
                          else "rc!=0")}
    pkl = RESULTS_DIR / f"{label}_v3_plus_{'+'.join(extras)}_{pair}_s{seed}_best.pkl"
    if not pkl.exists():
        return {"ok": False, "elapsed": elapsed, "error": f"missing {pkl.name}"}
    with open(pkl, "rb") as f:
        r = pickle.load(f)
    return {
        "ok": True, "elapsed": elapsed, "pair": pair, "extras": extras, "seed": seed,
        "is_pps": r["is_full"]["pips_per_day"],
        "oos_pps": r["oos"]["pips_per_day"],
        "is_trades": r["is_full"]["n_trades"],
        "oos_trades": r["oos"]["n_trades"],
        "dir_oos": r["oos"]["dir_ratio"],
        "passed_gates": r.get("passed_hard_gates", False),
        "min_chunk_pps": r.get("is_min_chunk_pps"),
    }


def run_sweep(pair: str, gens: int, seed: int, popsize: int, workers: int) -> str:
    """Run sweep_indicators.py and return path to results JSON."""
    cmd = [
        sys.executable, str(SWEEP),
        "--pair", pair,
        "--gens", str(gens),
        "--seed", str(seed),
        "--popsize", str(popsize),
        "--workers", str(workers),
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                          text=True, timeout=3600)
    if proc.returncode != 0:
        return None
    out = RESULTS_DIR / f"sweep_v3_plus_{pair}_s{seed}_results.json"
    return str(out) if out.exists() else None


def fmt_run(label: str, r: dict) -> str:
    if not r["ok"]:
        return f"❌ {label}: {r.get('error','unknown')}"
    gate = "✓" if r["passed_gates"] else "·"
    mc = f"{r['min_chunk_pps']:+.1f}" if r.get("min_chunk_pps") else "—"
    return (f"{label}: IS={r['is_pps']:+.2f}  OOS={r['oos_pps']:+.2f}  "
            f"min_ch={mc}  trades={r['oos_trades']}  dir={r['dir_oos']:.2f}  "
            f"gate={gate}  ({r['elapsed']:.0f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Skip steps by number, e.g. --skip 3 4")
    args = parser.parse_args()
    skip = {int(s) for s in args.skip}

    summary = {}
    t_global = time.time()
    tg_send(f"⛓ <b>Chain start (5 steps)</b>\n"
            f"gens={args.gens} pop={args.popsize} workers={args.workers}\n"
            f"Estimated: ~85 min")

    # ── Step 1: macd_hist on EUR_GBP ───────────────────────────
    if 1 not in skip:
        print("\n══ STEP 1: macd_hist on EUR_GBP ══")
        tg_send("⛓ <b>Step 1/5</b>: macd_hist on EUR_GBP")
        r = run_train("EUR_GBP", ["macd_hist"], args.seed, args.gens,
                      args.popsize, args.workers, "step1")
        summary["step1"] = r
        line = fmt_run("EUR_GBP+macd_hist", r)
        print(line)
        if r["ok"]:
            iron = IRONNET_V3["EUR_GBP"]
            delta = r["oos_pps"] - iron
            tg_send(f"✅ <b>Step 1</b>\n{line}\n"
                    f"vs IronNet V3 ({iron}): {delta:+.2f}")
        else:
            tg_send(f"❌ <b>Step 1</b>\n{line}")

    # ── Step 2: Stack top 3 on CHF_JPY ─────────────────────────
    if 2 not in skip:
        print("\n══ STEP 2: stack top-3 on CHF_JPY ══")
        tg_send("⛓ <b>Step 2/5</b>: V3 + macd_hist + range_pos_30 + m5_slope on CHF_JPY")
        r = run_train("CHF_JPY",
                      ["macd_hist", "range_pos_30", "m5_slope"],
                      args.seed, args.gens, args.popsize, args.workers, "step2")
        summary["step2"] = r
        line = fmt_run("CHF_JPY+top3", r)
        print(line)
        if r["ok"]:
            single = 73.03  # macd_hist alone OOS from sweep
            delta = r["oos_pps"] - single
            tg_send(f"✅ <b>Step 2</b>\n{line}\n"
                    f"vs macd_hist alone (+73.03): {delta:+.2f}")
        else:
            tg_send(f"❌ <b>Step 2</b>\n{line}")

    # ── Step 3: Sweep on EUR_GBP ───────────────────────────────
    if 3 not in skip:
        print("\n══ STEP 3: sweep on EUR_GBP (10 candidates) ══")
        tg_send("⛓ <b>Step 3/5</b>: full sweep on EUR_GBP (~30 min)")
        out = run_sweep("EUR_GBP", args.gens, args.seed, args.popsize, args.workers)
        summary["step3"] = {"results_json": out}
        if out:
            with open(out) as f:
                sweep_data = json.load(f)
            valid = [x for x in sweep_data["results"] if x["ok"]]
            valid.sort(key=lambda x: x["oos_pps"], reverse=True)
            top3 = valid[:3]
            top3_str = "\n".join(
                f"{i+1}. {x['extra']}: {x['oos_pps']:+.2f}"
                for i, x in enumerate(top3))
            tg_send(f"✅ <b>Step 3 sweep done</b>\n<b>Top 3 EUR_GBP:</b>\n{top3_str}")
        else:
            tg_send("❌ <b>Step 3</b>: sweep failed")

    # ── Step 4: macd_hist all 12 pairs ─────────────────────────
    if 4 not in skip:
        print("\n══ STEP 4: macd_hist on all 12 pairs ══")
        tg_send("⛓ <b>Step 4/5</b>: macd_hist on all 12 pairs (~36 min)")
        per_pair = []
        for i, pair in enumerate(ALL_PAIRS, 1):
            print(f"\n── [{i}/12] {pair} ──")
            r = run_train(pair, ["macd_hist"], args.seed, args.gens,
                          args.popsize, args.workers, "step4")
            per_pair.append(r)
            line = fmt_run(f"{pair}+macd_hist", r)
            print(line)
            if r["ok"]:
                iron = IRONNET_V3.get(pair, 0)
                delta = r["oos_pps"] - iron
                tg_send(f"[{i}/12] <b>{pair}</b>: OOS {r['oos_pps']:+.2f}  "
                        f"(IronNet {iron}, Δ{delta:+.2f})")
            else:
                tg_send(f"[{i}/12] ❌ {pair}: {r.get('error','')}")
        summary["step4"] = per_pair
        # Step-4 summary
        ok = [r for r in per_pair if r["ok"]]
        if ok:
            mean_oos = sum(r["oos_pps"] for r in ok) / len(ok)
            iron_mean = sum(IRONNET_V3.get(r["pair"], 0) for r in ok) / len(ok)
            tg_send(f"✅ <b>Step 4 grid done</b>\n"
                    f"Mean OOS: <b>{mean_oos:+.2f}</b> p/day\n"
                    f"IronNet V3 mean: {iron_mean:.2f}\n"
                    f"Δ vs IronNet: {mean_oos - iron_mean:+.2f}")

    # ── Step 5: 4-seed robustness on CHF_JPY+macd_hist ─────────
    if 5 not in skip:
        print("\n══ STEP 5: 4-seed robustness CHF_JPY+macd_hist ══")
        tg_send("⛓ <b>Step 5/5</b>: 4-seed robustness on CHF_JPY+macd_hist")
        seeds = [42, 137, 271, 314]
        per_seed = []
        for s in seeds:
            r = run_train("CHF_JPY", ["macd_hist"], s, args.gens,
                          args.popsize, args.workers, f"step5_s{s}")
            per_seed.append(r)
            line = fmt_run(f"seed={s}", r)
            print(line)
            if r["ok"]:
                tg_send(f"seed {s}: OOS {r['oos_pps']:+.2f}  "
                        f"gate={'✓' if r['passed_gates'] else '·'}")
        summary["step5"] = per_seed
        ok = [r for r in per_seed if r["ok"]]
        if ok:
            oos_vals = [r["oos_pps"] for r in ok]
            mean = sum(oos_vals) / len(oos_vals)
            std = (sum((v - mean) ** 2 for v in oos_vals) / len(oos_vals)) ** 0.5
            tg_send(f"✅ <b>Step 5 robustness done</b>\n"
                    f"4-seed OOS: mean={mean:+.2f} std={std:.2f}\n"
                    f"range: [{min(oos_vals):+.2f}, {max(oos_vals):+.2f}]")

    # ── Final summary ─────────────────────────────────────────
    elapsed = time.time() - t_global
    print(f"\n══ CHAIN COMPLETE — {elapsed/60:.1f} min ══")
    out_json = RESULTS_DIR / f"chain_5steps_s{args.seed}.json"
    with open(out_json, "w") as f:
        json.dump({"elapsed": elapsed, "summary": summary},
                  f, indent=2, default=str)
    print(f"Saved: {out_json}")
    tg_send(f"🏁 <b>Chain complete</b>\n"
            f"Total: {elapsed/60:.1f} min\n"
            f"Saved: chain_5steps_s{args.seed}.json")


if __name__ == "__main__":
    main()
