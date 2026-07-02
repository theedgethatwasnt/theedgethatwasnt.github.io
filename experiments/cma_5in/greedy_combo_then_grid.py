#!/usr/bin/env python3
"""
Greedy forward + backward feature selection on top of V3, then 12-pair grid.

Phase A — Greedy forward selection on CHF_JPY
---------------------------------------------
- Start: V3 + macd_hist (best single from sweep, IS=+55.68 OOS=+73.03)
- Round k:
    For each remaining candidate c not in stack:
        train V3+stack+c on CHF_JPY (200 gens, sin, popsize 24)
    Pick the c with the highest IS pips/day.
    If improvement over previous round's best IS > MIN_IMPROVE_PPS, accept
    and continue to round k+1. Else stop.
- Selection driven by IS pips/day (NEVER OOS) → no leakage.

Phase B — Backward elimination
------------------------------
- For each indicator in the final stack, train without it. If removing it
  doesn't drop IS pps by more than MIN_IMPROVE_PPS, keep it removed.

Phase C — 12-pair grid
----------------------
- Train the final winning stack on every pair.
- Compare against IronNet V3 baseline mean (42.0 p/day).

All settings: --features v3_plus --fixed-activation sin --gens 200 --popsize 24
Telegram updates per sub-run, per round, and at the end.
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

ALL_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY",
    "CHF_JPY", "NZD_JPY", "NZD_USD", "EUR_GBP",
]

# IronNet V3 NEAT baselines (per CLAUDE.md, 2026-04-05)
IRONNET_V3 = {
    "CHF_JPY": 65.5, "EUR_JPY": 59.4, "GBP_USD": 47.9, "AUD_JPY": 47.8,
    "GBP_JPY": 41.3, "EUR_USD": 41.4, "CAD_JPY": 41.6, "USD_JPY": 37.6,
    "NZD_JPY": 25.2, "AUD_USD": 23.5, "NZD_USD": 18.3, "EUR_GBP": 14.3,
}

CANDIDATES = [
    "macd_hist", "range_pos_30", "m5_slope", "bb_width",
    "aroon_osc", "gap_norm", "stoch_d", "h1_slope", "tec_5", "hl_price",
]

# Phase D — encyclopedia extension (inline-computable indicators not in CANDIDATES).
# aroon_osc_h1 first per user intuition.
ENCYCLOPEDIA = [
    "aroon_osc_h1",       # H1-cadence Aroon, prioritized per user hunch
    "rsi_14",
    "ema8_ratio",
    "ema21_ratio",
    "atr_ratio",
    "two_bar_momentum",
    "roc_10",
    "donchian_pos",
    "bb_pos",
    "stoch_k",
    "range_expansion",
    "body_ratio",
    "adx_14",
]

# Pre-loaded sweep round-1 results from CHF_JPY (already done, no need to re-run)
ROUND1_KNOWN = {
    "macd_hist":     {"is_pps": 55.68, "oos_pps": 73.03, "passed": True,  "min_chunk": 25.5},
    "range_pos_30":  {"is_pps": 56.78, "oos_pps": 67.36, "passed": True,  "min_chunk": 30.7},
    "m5_slope":      {"is_pps": 53.77, "oos_pps": 65.20, "passed": True,  "min_chunk": 29.5},
    "bb_width":      {"is_pps": 48.31, "oos_pps": 57.66, "passed": True,  "min_chunk": 26.1},
    "aroon_osc":     {"is_pps": 35.24, "oos_pps": 43.45, "passed": False, "min_chunk": None},
    "gap_norm":      {"is_pps": 28.96, "oos_pps": 34.13, "passed": True,  "min_chunk": 14.7},
    "stoch_d":       {"is_pps": 19.77, "oos_pps": 21.51, "passed": True,  "min_chunk": 11.1},
    "h1_slope":      {"is_pps": 14.47, "oos_pps": 15.86, "passed": True,  "min_chunk": 2.3},
    "tec_5":         {"is_pps": -0.69, "oos_pps":  0.21, "passed": False, "min_chunk": None},
    "hl_price":      {"is_pps": -0.15, "oos_pps":  0.05, "passed": False, "min_chunk": None},
}

MIN_IMPROVE_PPS = 1.0   # IS pps improvement needed to accept a new feature


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
                          else "rc!=0"), "extras": extras, "pair": pair}
    pkl = RESULTS_DIR / f"{label}_v3_plus_{'+'.join(extras)}_{pair}_s{seed}_best.pkl"
    if not pkl.exists():
        return {"ok": False, "elapsed": elapsed, "error": f"missing {pkl.name}",
                "extras": extras, "pair": pair}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    history = {"phase_A_rounds": [], "phase_B_removals": [], "phase_C_grid": []}
    t_global = time.time()

    # ── Phase A — Greedy forward selection ─────────────────────
    # Round 1: already done (sweep). Pick best by IS pps.
    round1_sorted = sorted(ROUND1_KNOWN.items(),
                           key=lambda kv: kv[1]["is_pps"], reverse=True)
    best_extra, best_data = round1_sorted[0]
    stack = [best_extra]
    best_is = best_data["is_pps"]
    history["phase_A_rounds"].append({
        "round": 1, "from_known": True, "winner": best_extra,
        "is_pps": best_is, "oos_pps": best_data["oos_pps"],
    })
    print(f"\n══ PHASE A — Greedy forward selection ══")
    print(f"\nRound 1 (from sweep): winner = {best_extra}  IS={best_is:+.2f}")
    tg_send(f"⛓ <b>Greedy combo search start</b>\n"
            f"Round 1 (from prior sweep): <b>{best_extra}</b>  IS={best_is:+.2f}")

    round_idx = 2
    while True:
        remaining = [c for c in CANDIDATES if c not in stack]
        if not remaining:
            print("All candidates exhausted.")
            break
        print(f"\n── Round {round_idx}: try adding to {stack} ──")
        tg_send(f"⛓ <b>Round {round_idx}</b>: trying +1 to {'+'.join(stack)}")

        round_results = []
        for c in remaining:
            test_stack = stack + [c]
            label = f"greedyR{round_idx}"
            print(f"   testing +{c} → {test_stack}")
            r = run_train("CHF_JPY", test_stack, args.seed, args.gens,
                          args.popsize, args.workers, label)
            round_results.append((c, r))
            if r["ok"]:
                gate = "✓" if r["passed_gates"] else "·"
                tg_send(f"R{round_idx}  +{c}: IS={r['is_pps']:+.2f}  "
                        f"OOS={r['oos_pps']:+.2f}  gate={gate}  ({r['elapsed']:.0f}s)")
            else:
                tg_send(f"R{round_idx}  +{c} ❌ {r.get('error','')}")

        ok_results = [(c, r) for c, r in round_results if r["ok"]]
        if not ok_results:
            print("All runs in this round failed; stopping.")
            tg_send(f"⚠️ Round {round_idx}: all runs failed, stopping greedy.")
            break

        # Pick winner by IS pps (the rigorous selection metric)
        ok_results.sort(key=lambda x: x[1]["is_pps"], reverse=True)
        winner_c, winner_r = ok_results[0]
        winner_is = winner_r["is_pps"]
        improvement = winner_is - best_is

        history["phase_A_rounds"].append({
            "round": round_idx, "winner": winner_c,
            "winner_is_pps": winner_is, "winner_oos_pps": winner_r["oos_pps"],
            "improvement_vs_prev": improvement,
            "all_results": [
                {"extra": c, "is_pps": r["is_pps"] if r["ok"] else None,
                 "oos_pps": r["oos_pps"] if r["ok"] else None,
                 "passed": r.get("passed_gates", False) if r["ok"] else False,
                 "ok": r["ok"]}
                for c, r in round_results
            ],
        })

        if improvement > MIN_IMPROVE_PPS:
            stack.append(winner_c)
            best_is = winner_is
            print(f"   ACCEPT +{winner_c}  IS {best_is - improvement:+.2f} → {best_is:+.2f}  (+{improvement:.2f})")
            tg_send(f"✅ R{round_idx} <b>ACCEPT</b> +{winner_c}\n"
                    f"Stack: {'+'.join(stack)}\n"
                    f"IS: {best_is:+.2f} (+{improvement:.2f})  "
                    f"OOS: {winner_r['oos_pps']:+.2f}")
            round_idx += 1
        else:
            print(f"   REJECT +{winner_c}  IS gain only +{improvement:.2f} (< {MIN_IMPROVE_PPS}), stopping.")
            tg_send(f"🛑 R{round_idx} <b>STOP</b> — best gain only +{improvement:.2f}")
            break

    print(f"\n══ Phase A done. Stack after greedy: {stack} ══")

    # ── Phase B — Backward elimination ─────────────────────────
    print(f"\n══ PHASE B — Backward elimination ══")
    tg_send(f"⛓ <b>Phase B</b>: backward elim from {'+'.join(stack)}")
    if len(stack) <= 1:
        print("Stack has only 1 element, skipping backward elimination.")
    else:
        i = 0
        while i < len(stack):
            test_stack = stack[:i] + stack[i+1:]
            removed = stack[i]
            print(f"   try removing {removed} → {test_stack}")
            r = run_train("CHF_JPY", test_stack, args.seed, args.gens,
                          args.popsize, args.workers, "greedyB")
            if not r["ok"]:
                tg_send(f"B  -{removed} ❌")
                i += 1
                continue
            new_is = r["is_pps"]
            drop = best_is - new_is
            tg_send(f"B  -{removed}: IS={new_is:+.2f}  "
                    f"(drop {drop:+.2f})  OOS={r['oos_pps']:+.2f}")
            history["phase_B_removals"].append({
                "removed": removed, "new_stack": test_stack,
                "new_is_pps": new_is, "new_oos_pps": r["oos_pps"],
                "is_drop": drop, "kept": drop > MIN_IMPROVE_PPS,
            })
            if drop <= MIN_IMPROVE_PPS:
                # Removal didn't hurt → eliminate
                print(f"   REMOVE {removed} (IS drop only {drop:.2f})")
                tg_send(f"✅ B <b>REMOVE</b> {removed} (drop {drop:.2f})")
                stack = test_stack
                best_is = new_is
                # Don't increment i — same index now points to next element
            else:
                print(f"   KEEP {removed} (IS drop {drop:.2f} > {MIN_IMPROVE_PPS})")
                i += 1

    print(f"\n══ Phase B done. Stack after backward elim: {stack} ══")
    tg_send(f"🏁 <b>Greedy + backward done</b>\n"
            f"Stack: <b>{'+'.join(stack)}</b>\n"
            f"CHF_JPY IS: {best_is:+.2f}")

    # ── Phase D — Encyclopedia extension ───────────────────────
    # Try inline-computable indicators NOT in the original sweep candidates.
    # Single pass: try each, accept if IS improves > MIN_IMPROVE_PPS.
    print(f"\n══ PHASE D — Encyclopedia extension ══")
    tg_send(f"⛓ <b>Phase D</b>: encyclopedia extension "
            f"({len(ENCYCLOPEDIA)} indicators not in sweep)")
    history["phase_D_attempts"] = []
    for i, ex in enumerate(ENCYCLOPEDIA, 1):
        if ex in stack:
            continue
        test_stack = stack + [ex]
        print(f"\n── D[{i}/{len(ENCYCLOPEDIA)}] try +{ex} ──")
        r = run_train("CHF_JPY", test_stack, args.seed, args.gens,
                      args.popsize, args.workers, "greedyD")
        if not r["ok"]:
            tg_send(f"D[{i}/{len(ENCYCLOPEDIA)}] +{ex} ❌ {r.get('error','')}")
            history["phase_D_attempts"].append({"extra": ex, "ok": False,
                                                 "error": r.get("error")})
            continue
        new_is = r["is_pps"]
        gain = new_is - best_is
        gate = "✓" if r["passed_gates"] else "·"
        history["phase_D_attempts"].append({
            "extra": ex, "ok": True, "is_pps": new_is, "oos_pps": r["oos_pps"],
            "gain_vs_prev": gain, "passed": r["passed_gates"],
            "accepted": gain > MIN_IMPROVE_PPS,
        })
        if gain > MIN_IMPROVE_PPS:
            stack = test_stack
            best_is = new_is
            print(f"   ACCEPT +{ex}  IS gain +{gain:.2f}")
            tg_send(f"✅ D[{i}] <b>ACCEPT</b> +{ex}\n"
                    f"Stack: {'+'.join(stack)}\n"
                    f"IS: {best_is:+.2f} (+{gain:.2f})  "
                    f"OOS: {r['oos_pps']:+.2f}  gate={gate}")
        else:
            print(f"   reject +{ex}  IS gain {gain:+.2f}")
            tg_send(f"D[{i}] +{ex}: IS={new_is:+.2f}  gain={gain:+.2f}  "
                    f"OOS={r['oos_pps']:+.2f}  gate={gate} (reject)")

    print(f"\n══ Phase D done. Final stack: {stack} ══")
    tg_send(f"🏁 <b>Phase D done</b>\n"
            f"Final stack ({len(stack)}): <b>{'+'.join(stack)}</b>\n"
            f"CHF_JPY IS: {best_is:+.2f}")

    # ── Phase C — 12-pair grid with the winning stack ──────────
    print(f"\n══ PHASE C — 12-pair grid with stack {stack} ══")
    tg_send(f"⛓ <b>Phase C</b>: 12-pair grid with {'+'.join(stack)}")
    grid_results = []
    for i, pair in enumerate(ALL_PAIRS, 1):
        print(f"\n── [{i}/12] {pair} ──")
        r = run_train(pair, stack, args.seed, args.gens,
                      args.popsize, args.workers, "greedyC")
        grid_results.append(r)
        if r["ok"]:
            iron = IRONNET_V3.get(pair, 0)
            delta = r["oos_pps"] - iron
            gate = "✓" if r["passed_gates"] else "·"
            tg_send(f"[{i}/12] <b>{pair}</b>: OOS {r['oos_pps']:+.2f}  "
                    f"(IronNet {iron}, Δ{delta:+.2f})  gate={gate}")
        else:
            tg_send(f"[{i}/12] ❌ {pair}")
    history["phase_C_grid"] = grid_results

    # Final summary
    elapsed = time.time() - t_global
    ok = [r for r in grid_results if r["ok"]]
    if ok:
        mean_oos = sum(r["oos_pps"] for r in ok) / len(ok)
        iron_mean = sum(IRONNET_V3.get(r["pair"], 0) for r in ok) / len(ok)
        passed = sum(1 for r in ok if r["passed_gates"])
        ranked = sorted(ok, key=lambda r: r["oos_pps"], reverse=True)
        top3 = "\n".join(
            f"{i+1}. {r['pair']}: {r['oos_pps']:+.2f} "
            f"(iron {IRONNET_V3.get(r['pair'],0)})"
            for i, r in enumerate(ranked[:3]))
        print(f"\n══ COMPLETE — {elapsed/60:.1f} min ══")
        print(f"Final stack: {stack}")
        print(f"Mean OOS over 12 pairs: {mean_oos:+.2f}  (IronNet {iron_mean:.2f})")
        print(f"Hard-gate pass: {passed}/{len(ok)}")
        tg_send(
            f"🏁 <b>CHAIN COMPLETE</b> ({elapsed/60:.1f} min)\n\n"
            f"<b>Final stack ({len(stack)} extras)</b>:\n{'+'.join(stack)}\n\n"
            f"<b>12-pair OOS</b>:\n"
            f"  Mean: <b>{mean_oos:+.2f}</b> p/day\n"
            f"  IronNet V3 mean: {iron_mean:.2f}\n"
            f"  Δ: <b>{mean_oos - iron_mean:+.2f}</b>\n"
            f"  Gates passed: {passed}/{len(ok)}\n\n"
            f"<b>Top 3 pairs:</b>\n{top3}"
        )

    out = RESULTS_DIR / f"greedy_combo_grid_s{args.seed}.json"
    with open(out, "w") as f:
        json.dump({"elapsed": elapsed, "final_stack": stack,
                   "history": history}, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
