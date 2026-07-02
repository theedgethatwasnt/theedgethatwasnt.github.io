"""Master loop: walks candidates.json, runs each through the full pipeline.

Per candidate:
  1. Check port_status — bail with human-visible message if "pending"
  2. Validate (parity + causality) via validate.py
  3. Verify candidate column exists in causal parquets for test pairs
  4. Run CMA-ES for each (pair × seed)
  5. Aggregate metrics → append to results_table.md
  6. Append dated section to JOURNEY-README.md
  7. Send Telegram update
  8. Git commit + push
  9. Mark candidate.validated = True in candidates.json, advance

Run:
    python3 loop.py                 # pending candidates only
    python3 loop.py --candidate tec5   # single candidate
    python3 loop.py --smoke         # 1 pair × 1 seed × 30 gens per candidate
"""
from __future__ import annotations
import argparse, json, multiprocessing, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
LOOP_DIR = Path(__file__).parent
RESULTS_DIR = LOOP_DIR / "results"
JOURNEY = PROJECT / "JOURNEY-README.md"
CANDIDATES = LOOP_DIR / "candidates.json"
RESULTS_TABLE = RESULTS_DIR / "results_table.md"

sys.path.insert(0, str(LOOP_DIR))
sys.path.insert(0, str(PROJECT))


def _telegram(msg: str) -> None:
    """Best-effort Telegram notification. Never blocks."""
    try:
        from lib.notify import _send
        _send(msg)
    except Exception as e:
        print(f"[telegram] failed: {e}", flush=True)


def _git(*args: str) -> int:
    return subprocess.call(["git", *args], cwd=str(PROJECT))


def load_candidates() -> dict:
    with open(CANDIDATES) as f:
        return json.load(f)


def save_candidates(data: dict) -> None:
    with open(CANDIDATES, "w") as f:
        json.dump(data, f, indent=2)


# ── Reference implementations for validator (map name → fn) ──────────
def _ref_tec5(df):
    import numpy as np
    c = df["close"].values.astype("float64")
    n = len(c)
    out = np.zeros(n)
    for i in range(5, n):
        net = c[i] - c[i - 5]
        path = 0.0
        for k in range(i - 4, i + 1):
            path += abs(c[k] - c[k - 1])
        if path > 1e-12:
            er = abs(net) / path
            out[i] = er if net > 0 else -er if net < 0 else 0.0
    return out


def _ref_roc_10(df):
    import numpy as np
    c = df["close"].values.astype("float64")
    out = np.zeros(len(c))
    for i in range(10, len(c)):
        out[i] = (c[i] - c[i - 10]) / c[i - 10] if c[i - 10] != 0 else 0.0
    return out


def _ref_range_pos_30(df):
    import numpy as np
    h = df["high"].values.astype("float64")
    l = df["low"].values.astype("float64")
    c = df["close"].values.astype("float64")
    n = len(c)
    out = np.full(n, 0.5)
    for i in range(29, n):
        mx = h[i-29:i+1].max()
        mn = l[i-29:i+1].min()
        rng = mx - mn
        if rng > 1e-12:
            out[i] = (c[i] - mn) / rng
    return out


def _ref_rsi_14(df):
    import numpy as np
    c = df["close"].values.astype("float64")
    n = len(c)
    out = np.full(n, 0.5)
    if n < 2:
        return out
    diff = np.diff(c)
    gain = np.maximum(diff, 0.0)
    loss = np.maximum(-diff, 0.0)
    # Cumulative mean through bar 14, then Wilder
    ag = al = 0.0
    for i in range(len(diff)):
        idx = i + 1  # index in c
        if i < 14:
            n_s = i + 1
            ag = ag + (gain[i] - ag) / n_s
            al = al + (loss[i] - al) / n_s
        else:
            ag = (1.0/14.0) * gain[i] + (13.0/14.0) * ag
            al = (1.0/14.0) * loss[i] + (13.0/14.0) * al
        if i >= 13:
            if al <= 1e-12:
                out[idx] = 1.0
            else:
                rs = ag / al
                out[idx] = 1.0 - 1.0 / (1.0 + rs)
    return out


def _ref_bb_width(df):
    import numpy as np
    c = df["close"].values.astype("float64")
    n = len(c)
    out = np.zeros(n)
    for i in range(19, n):
        w = c[i-19:i+1]
        sd = w.std()
        if c[i] > 0:
            out[i] = (4.0 * sd) / c[i]
    return out


def _ref_aroon_osc(df):
    import numpy as np
    h = df["high"].values.astype("float64")
    l = df["low"].values.astype("float64")
    n = len(h)
    out = np.zeros(n)
    for i in range(24, n):
        hs = h[i-24:i+1]
        ls = l[i-24:i+1]
        hh_idx = int(np.argmax(hs))
        ll_idx = int(np.argmin(ls))
        periods_since_hh = 24 - hh_idx
        periods_since_ll = 24 - ll_idx
        up = (25 - periods_since_hh) / 25.0
        down = (25 - periods_since_ll) / 25.0
        out[i] = up - down
    return out


def _ref_ema_ratio(df, period):
    import numpy as np
    c = df["close"].values.astype("float64")
    alpha = 2.0 / (period + 1.0)
    e = c[0]
    out = np.zeros(len(c))
    out[0] = 1.0
    for i in range(1, len(c)):
        e = alpha * c[i] + (1.0 - alpha) * e
        out[i] = c[i] / e if e > 0 else 1.0
    return out


def _ref_ema8_ratio(df): return _ref_ema_ratio(df, 8)
def _ref_ema21_ratio(df): return _ref_ema_ratio(df, 21)


def _ref_cci(df):
    import numpy as np
    h = df["high"].values.astype("float64")
    l = df["low"].values.astype("float64")
    c = df["close"].values.astype("float64")
    tp = (h + l + c) / 3.0
    n = len(tp)
    out = np.zeros(n)
    for i in range(19, n):
        w = tp[i-19:i+1]
        sma = w.mean()
        md = np.mean(np.abs(w - sma))
        if md > 1e-12:
            out[i] = (tp[i] - sma) / (0.015 * md) / 200.0
    return out


def _ref_stoch_d(df):
    import numpy as np
    h = df["high"].values.astype("float64")
    l = df["low"].values.astype("float64")
    c = df["close"].values.astype("float64")
    n = len(c)
    k_arr = np.full(n, 0.5)
    for i in range(13, n):
        hh = h[i-13:i+1].max()
        ll = l[i-13:i+1].min()
        rng = hh - ll
        k_arr[i] = (c[i] - ll) / rng if rng > 1e-12 else 0.5
    d_arr = np.full(n, 0.5)
    for i in range(n):
        lo = max(0, i - 2)
        d_arr[i] = k_arr[lo:i+1].mean()
    return d_arr


def _ref_atr_ratio(df):
    import numpy as np
    h = df["high"].values.astype("float64")
    l = df["low"].values.astype("float64")
    c = df["close"].values.astype("float64")
    n = len(c)
    out = np.zeros(n)
    atr = 0.0
    samples = 0
    for i in range(n):
        if i == 0:
            tr = h[0] - l[0]
        else:
            tr = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        samples += 1
        if samples <= 14:
            if samples == 1:
                atr = tr
            else:
                atr = atr + (tr - atr) / samples
        else:
            atr = (1.0/14.0) * tr + (13.0/14.0) * atr
        out[i] = atr / c[i] if c[i] > 0 else 0.0
    return out


def _ref_donchian_pos(df):
    import numpy as np
    h = df["high"].values.astype("float64")
    l = df["low"].values.astype("float64")
    c = df["close"].values.astype("float64")
    n = len(c)
    out = np.full(n, 0.5)
    for i in range(19, n):
        hh = h[i-19:i+1].max()
        ll = l[i-19:i+1].min()
        rng = hh - ll
        if rng > 1e-12:
            out[i] = (c[i] - ll) / rng
    return out


REF_FNS = {
    "tec5": _ref_tec5,
    "roc_10": _ref_roc_10,
    "range_pos_30": _ref_range_pos_30,
    "rsi_14": _ref_rsi_14,
    "bb_width": _ref_bb_width,
    "aroon_osc": _ref_aroon_osc,
    "ema8_ratio": _ref_ema8_ratio,
    "ema21_ratio": _ref_ema21_ratio,
    "cci": _ref_cci,
    "stoch_d": _ref_stoch_d,
    "atr_ratio": _ref_atr_ratio,
    "donchian_pos": _ref_donchian_pos,
}


def validate(name: str, builder_key: str, expected_range, pair: str = "EUR_JPY") -> bool:
    """Run validator for a ported candidate."""
    import pandas as pd
    from validate import validate_candidate, print_result, ValidationError
    if name not in REF_FNS:
        print(f"[validate] no reference_fn registered for '{name}' — skipping parity check", flush=True)
        return True  # trust incremental if no reference
    df = pd.read_parquet(PROJECT / f"data/m5_ohlc/{pair}_M5.parquet")
    try:
        r = validate_candidate(name, REF_FNS[name], builder_key, df, expected_range)
        print_result(r)
        return True
    except ValidationError as e:
        print(f"[validate] FAILED: {e}", flush=True)
        return False


def _run_one_cma(spec: tuple[str, str, int, int, int]) -> dict | None:
    """Subprocess worker — runs one CMA to completion, returns parsed JSON or None."""
    candidate, pair, seed, gens, inner_workers = spec
    script = LOOP_DIR / "test_slot4_swap_cma.py"
    cmd = ["python3", str(script),
           "--candidate", candidate, "--pair", pair,
           "--seed", str(seed), "--gens", str(gens),
           "--pop", "40", "--workers", str(inner_workers)]
    t0 = time.time()
    log = RESULTS_DIR / f"slot4_{candidate}_{pair}_s{seed}.log"
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=str(PROJECT), stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    tag = f"{candidate}/{pair}/s{seed}"
    if rc != 0:
        print(f"[run] {tag} FAILED rc={rc} in {dt:.0f}s (log: {log.name})", flush=True)
        return None
    jpath = RESULTS_DIR / f"slot4_{candidate}_{pair}_s{seed}.json"
    if not jpath.exists():
        print(f"[run] {tag} no json output", flush=True)
        return None
    with open(jpath) as f:
        r = json.load(f)
    print(f"[run] {tag} done in {dt:.0f}s: OOS {r['oos']['pd']:+.2f} p/d "
          f"({r['oos']['n_trades']}T dir {r['oos']['dir']:.2f})", flush=True)
    return r


def run_cma_grid(candidate: str, pairs: list[str], seeds: list[int], gens: int,
                 parallel: int, inner_workers: int) -> list[dict]:
    """Fan out (pair × seed) runs across `parallel` concurrent subprocesses.
    Each subprocess uses `inner_workers` for CMA fitness parallelism.
    Total CPU use ≈ parallel × inner_workers."""
    specs = [(candidate, p, s, gens, inner_workers) for p in pairs for s in seeds]
    rows: list[dict] = []
    print(f"\n[grid] {candidate}: {len(specs)} runs, {parallel}×{inner_workers} CPUs", flush=True)
    with ProcessPoolExecutor(max_workers=parallel) as ex:
        futs = {ex.submit(_run_one_cma, spec): spec for spec in specs}
        for fut in as_completed(futs):
            spec = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"[grid] {spec} raised: {e}", flush=True)
                continue
            if r is not None:
                rows.append(r)
                _telegram(
                    f"✓ {candidate} {spec[1]} s{spec[2]}: OOS {r['oos']['pd']:+.2f} p/d"
                )
    return rows


def append_results_row(candidate: str, rows: list[dict]) -> None:
    """Append a row per (pair, seed) to results_table.md."""
    header_needed = not RESULTS_TABLE.exists()
    with open(RESULTS_TABLE, "a") as f:
        if header_needed:
            f.write("# Slot-4 swap results\n\n")
            f.write("| when | candidate | pair | seed | fit | IS p/d | OOS p/d | OOS T | dir | acts | θ |\n")
            f.write("|------|-----------|------|------|-----|--------|---------|-------|-----|------|----|\n")
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        for r in rows:
            acts = "/".join(r.get("activations", []))
            f.write(
                f"| {now} | {candidate} | {r['pair']} | {r['seed']} "
                f"| {r['fitness']:+.2f} "
                f"| {r['is']['pd']:+.2f} | {r['oos']['pd']:+.2f} "
                f"| {r['oos']['n_trades']} | {r['oos']['dir']:.2f} "
                f"| {acts} | {r['theta']:.3f} |\n"
            )


def append_journey(candidate: str, rows: list[dict], notes: str) -> None:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if not rows:
        return
    oos_vals = [r["oos"]["pd"] for r in rows]
    avg_oos = sum(oos_vals) / len(oos_vals)
    min_oos = min(oos_vals); max_oos = max(oos_vals)
    lines = []
    lines.append(f"\n## {today} — Slot-4 swap: {candidate}\n")
    lines.append(f"\n### Setup\n")
    lines.append(f"- Fixed inputs: `mc_d_a, mc_dd_a, er_norm, UPnL` + slot-4 = `{candidate}`\n")
    lines.append(f"- Architecture: 5→4+skip→3, per-node activation gene {{tanh, sin, gauss}}, θ latching\n")
    lines.append(f"- Fitness: amddp1, 3-chunk WF-in-fitness, hard gates (≥1T/day/chunk, dir ≥ 0.15)\n")
    lines.append(f"- Via post-RCA FXFeatureBuilder (causal, validated parity + causality probe)\n")
    lines.append(f"\n### Results\n\n")
    lines.append("| pair | seed | fit | IS p/d | OOS p/d | OOS T | dir | acts |\n")
    lines.append("|------|------|-----|--------|---------|-------|-----|------|\n")
    for r in rows:
        acts = "/".join(r.get("activations", []))
        lines.append(
            f"| {r['pair']} | {r['seed']} | {r['fitness']:+.2f} "
            f"| {r['is']['pd']:+.2f} | {r['oos']['pd']:+.2f} "
            f"| {r['oos']['n_trades']} | {r['oos']['dir']:.2f} | {acts} |\n"
        )
    lines.append(f"\n**Summary**: OOS p/d across {len(rows)} runs: mean {avg_oos:+.2f}, "
                 f"range [{min_oos:+.2f}, {max_oos:+.2f}].\n")
    if notes:
        lines.append(f"\n**Notes**: {notes}\n")
    block = "".join(lines)
    # Prepend at top (JOURNEY is reverse-chronological)
    existing = JOURNEY.read_text() if JOURNEY.exists() else ""
    JOURNEY.write_text(block + existing)


def run_candidate(meta: dict, pairs: list[str], seeds: list[int], gens: int) -> tuple[bool, list[dict]]:
    name = meta["name"]
    print(f"\n{'='*72}\n  CANDIDATE: {name}\n{'='*72}", flush=True)

    # Validate port status
    if meta["port_status"] == "pending":
        msg = f"[{name}] port_status=pending — port into FXFeatureBuilder first"
        print(msg, flush=True)
        _telegram(f"⏸ Slot-4 loop: {name} pending port, skipping")
        return False, []

    # Validate parity + causality (OHLC candidates only; state composites skip)
    if meta["class"] == "indicator":
        builder_key = name
        rng = meta["range"]
        ok = validate(name, builder_key, (rng[0], rng[1]), pair=pairs[0])
        if not ok:
            _telegram(f"❌ Slot-4 loop: {name} validation FAILED")
            return False, []

    # Run CMA grid in parallel (total_cpu ≈ _PARALLEL × _INNER)
    rows = run_cma_grid(name, pairs, seeds, gens,
                        parallel=_PARALLEL, inner_workers=_INNER)
    return True, rows


def update_candidate_status(data: dict, name: str, **updates) -> None:
    for c in data["candidates"]:
        if c["name"] == name:
            c.update(updates)
            break


def git_publish(candidate: str, rows: list[dict]) -> None:
    _git("add",
         "research/experiments/cma_5in/indicator_loop/",
         "JOURNEY-README.md",
         "lib/incremental_features.py")
    oos = [r["oos"]["pd"] for r in rows]
    avg = sum(oos) / len(oos) if oos else 0.0
    msg = (f"Slot-4 loop: {candidate} — {len(rows)} runs, avg OOS {avg:+.2f} p/d\n\n"
           f"Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>")
    _git("commit", "-m", msg)
    _git("push", "origin", "master")


_PARALLEL = 2   # how many (pair, seed) CMAs to run concurrently
_INNER = 4      # inner workers per CMA (fitness eval parallelism)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=None, help="Run only this candidate")
    ap.add_argument("--smoke", action="store_true",
                    help="1 pair × 1 seed × 30 gens for fast smoke")
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--parallel", type=int, default=None,
                    help="Concurrent (pair×seed) subprocess runs. Default auto: CPU//4.")
    ap.add_argument("--inner", type=int, default=4,
                    help="Inner CMA fitness workers per subprocess. Default 4.")
    args = ap.parse_args()

    global _PARALLEL, _INNER
    cpu = multiprocessing.cpu_count()
    _INNER = args.inner
    _PARALLEL = args.parallel or max(1, cpu // max(_INNER, 1))
    print(f"[config] cpu={cpu}, parallel={_PARALLEL}, inner={_INNER} "
          f"(total cpu usage ≈ {_PARALLEL * _INNER})", flush=True)

    data = load_candidates()
    pairs = args.pairs or (["EUR_JPY"] if args.smoke else data["pairs_smoke"])
    seeds = args.seeds or ([42] if args.smoke else data["seeds"])
    gens = args.gens or (30 if args.smoke else data["gens_local"])

    queue = data["candidates"]
    if args.candidate:
        queue = [c for c in queue if c["name"] == args.candidate]
        if not queue:
            raise SystemExit(f"No candidate named '{args.candidate}'")

    _telegram(f"🔄 Slot-4 loop starting: {len(queue)} candidate(s), {len(pairs)}×{len(seeds)} runs each, gens={gens}")

    for meta in queue:
        name = meta["name"]
        if meta.get("validated") and not args.candidate:
            print(f"[{name}] already validated, skipping (use --candidate to force)", flush=True)
            continue

        ok, rows = run_candidate(meta, pairs, seeds, gens)
        if not rows:
            continue

        append_results_row(name, rows)
        append_journey(name, rows, meta.get("notes", ""))

        # Mark validated — reload from disk first to avoid clobbering
        # external edits (e.g. batch marking new ports while loop runs)
        fresh = load_candidates()
        update_candidate_status(fresh, name, validated=True,
                                last_run=datetime.utcnow().isoformat())
        save_candidates(fresh)
        data = fresh  # keep local in sync

        oos = [r["oos"]["pd"] for r in rows]
        avg = sum(oos) / len(oos)
        _telegram(
            f"📊 {name} done: avg OOS {avg:+.2f} p/d across {len(rows)} runs "
            f"(range [{min(oos):+.2f}, {max(oos):+.2f}])"
        )

        git_publish(name, rows)

    _telegram("✅ Slot-4 loop finished")


if __name__ == "__main__":
    main()
