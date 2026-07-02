#!/usr/bin/env python3
"""
monitor.py — HEADLESS real-time graphs → Telegram for the NEAT P&F + AMDDP5 campaign.
=====================================================================================
Design: research/experiments/neat_pnf_amddp/PLAN.md  § "Real-time monitoring".

Hetzner has NO GUI, so this renders with the matplotlib **Agg** backend (no display),
writes PNGs to disk, and pushes them to Telegram via the existing bot
(TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID, sendPhoto — same pattern as
services/telegram_bot/main.py + lib/notify.py).

It is **fully decoupled from training** (crash-safe): it only READS the artifacts
campaign.py writes — per-island resume checkpoints (best_fitness/best_val history)
and the versioned deploy bundles (best_gen*.pkl + all_time_best.pkl, each carrying a
full trading-stats bundle). If a training server dies, the monitor still renders from
whatever is on disk; when training resumes the monitor picks up the new gens.

Six figures are rendered + sent, throttled (every ~N gens or ~M minutes, configurable):
  (1) fitness vs generation     — best per island (one line per island), real vs surrogate.
  (2) running-best OOS/val cumulative AMDDP5 with IS overlay (the overfit gap).
  (3) per-trade distributions   — AMDDP5 / drawdown / hold-time histograms (best island).
  (4) aggregate-metric trend     — Sharpe / Calmar / expectancy / SQN vs gen (best island).
  (5) real-best vs surrogate-null fitness band — are we beating noise-evolution?
  (6) deploy-bundle summary card — IS/VAL metrics + WF chunks + MC p + gates (best island).

Run:
  python3 monitor.py --runs campaign_runs --interval-min 10        # loop forever
  python3 monitor.py --runs campaign_runs --once                   # render once + exit
  python3 monitor.py --runs campaign_runs --pair GBP_JPY --once

If Telegram creds are absent, PNGs are saved under <runs>/_monitor/ and we log
"would send" instead of failing.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")  # headless: no display required (Hetzner)  ── MUST precede pyplot
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# Deploy bundles embed a CappedGenome (research.experiments.neat_pnf_amddp.phase1_harness)
# + custom activations — importing the harness registers them so pickle.load succeeds.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
try:
    from research.experiments.neat_pnf_amddp import phase1_harness  # noqa: E402,F401
except Exception as _e:  # pragma: no cover — monitor must not die if harness import slips
    print(f"[monitor] WARN: could not import phase1_harness ({_e}); "
          "genome unpickling may fail, history-only plots still work.")

# ── Telegram (reuse the project's notify pattern; load .env if python-dotenv present) ──
try:
    from dotenv import load_dotenv
    for _env in (os.path.join(HERE, "..", "..", "..", "lib", ".env"),
                 os.path.join(HERE, "..", "..", "..", ".env")):
        if os.path.exists(_env):
            load_dotenv(_env)
            break
except Exception:
    pass

import requests  # noqa: E402

_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def send_photo(path, caption=""):
    """Send a PNG to Telegram via sendPhoto. If creds are absent, log 'would send'.

    Returns True if actually sent. Never raises (monitoring must not crash training-watch).
    """
    if not (_BOT_TOKEN and _CHAT_ID):
        print(f"[monitor] (no Telegram creds) would send {os.path.basename(path)} :: {caption}")
        return False
    try:
        with open(path, "rb") as fh:
            requests.post(
                f"https://api.telegram.org/bot{_BOT_TOKEN}/sendPhoto",
                data={"chat_id": _CHAT_ID, "caption": caption[:1024]},
                files={"photo": (os.path.basename(path), fh, "image/png")},
                timeout=20,
            )
        print(f"[monitor] sent {os.path.basename(path)}")
        return True
    except Exception as e:
        print(f"[monitor] send failed for {os.path.basename(path)}: {e}")
        return False


def send_text(text):
    if not (_BOT_TOKEN and _CHAT_ID):
        print(f"[monitor] (no Telegram creds) would send text :: {text}")
        return False
    try:
        requests.post(
            f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage",
            json={"chat_id": _CHAT_ID, "text": text[:4000], "parse_mode": "HTML"},
            timeout=10,
        )
        return True
    except Exception as e:
        print(f"[monitor] send_text failed: {e}")
        return False


# ── Artifact discovery / loading ────────────────────────────────────────────────
def find_island_dirs(runs_root, pair=None):
    """Every <runs_root>/<pair>/<tag>/ that has a bundles/ subdir."""
    pattern = os.path.join(runs_root, pair if pair else "*", "*")
    out = []
    for d in sorted(glob.glob(pattern)):
        if os.path.isdir(os.path.join(d, "bundles")):
            out.append(d)
    return out


def load_latest_resume(island_dir):
    """Latest resume checkpoint → its histories (best_fitness/best_val) + context.
    History-only (no genome unpickle needed for the fitness/val plots).
    """
    rdir = os.path.join(island_dir, "resume")
    files = sorted(glob.glob(os.path.join(rdir, "resume_gen*.pkl")),
                   key=lambda p: int(os.path.basename(p)[len("resume_gen"):-4])) \
        if os.path.isdir(rdir) else []
    if not files:
        return None
    try:
        with open(files[-1], "rb") as fh:
            st = pickle.load(fh)
        # drop the heavy population blob; the monitor only needs the histories/context
        st.pop("population_pickle", None)
        return st
    except Exception as e:
        print(f"[monitor] could not read resume in {island_dir}: {e}")
        return None


def load_all_time_best(island_dir):
    p = os.path.join(island_dir, "bundles", "all_time_best.pkl")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except Exception as e:
        print(f"[monitor] could not read all_time_best in {island_dir}: {e}")
        return None


def load_gen_bundles(island_dir):
    """Load every best_gen*.pkl (sorted by gen) for aggregate-metric-vs-gen trends.

    Returns a list of (gen, bundle) — only the lightweight aggregate dicts are kept
    (we DON'T retain the per-trade arrays for all gens, just for the latest = all_time_best).
    """
    bdir = os.path.join(island_dir, "bundles")
    files = sorted(glob.glob(os.path.join(bdir, "best_gen*.pkl")),
                   key=lambda p: int(os.path.basename(p)[len("best_gen"):-4]))
    out = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                b = pickle.load(fh)
            g = int(os.path.basename(f)[len("best_gen"):-4])
            # keep only what the trend plot needs (avoid holding all per-trade arrays)
            light = {
                "generation": g,
                "is_agg": b.get("stats", {}).get("is", {}).get("aggregates", {}),
                "val_agg": b.get("stats", {}).get("val", {}).get("aggregates", {}),
            }
            out.append((g, light))
        except Exception:
            continue
    return out


def island_label(st_or_bundle):
    """Short tag for an island from resume state or bundle."""
    isl = st_or_bundle.get("island", "?")
    seed = st_or_bundle.get("seed", "?")
    exp = st_or_bundle.get("exponent", st_or_bundle.get("exp", "?"))
    sur = st_or_bundle.get("surrogate", st_or_bundle.get("is_surrogate", False))
    s = f"isl{isl}·s{seed}·e{exp}"
    return s + ("·SUR" if sur else "")


def is_surrogate(island_dir, st, atb):
    if st is not None and "surrogate" in st:
        return bool(st["surrogate"])
    if atb is not None and "is_surrogate" in atb:
        return bool(atb["is_surrogate"])
    return island_dir.rstrip("/").endswith("_surrogate")


# ── Per-island gather ────────────────────────────────────────────────────────────
def gather(runs_root, pair):
    islands = []
    for d in find_island_dirs(runs_root, pair):
        st = load_latest_resume(d)
        atb = load_all_time_best(d)
        if st is None and atb is None:
            continue
        islands.append({
            "dir": d,
            "resume": st,
            "atb": atb,
            "surrogate": is_surrogate(d, st, atb),
            "label": island_label(st or atb or {}),
            "gens": (load_gen_bundles(d) if atb is not None else []),
        })
    return islands


# ── Figure renderers (each returns the PNG path) ─────────────────────────────────
def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fig_fitness_vs_gen(islands, out_dir, pair):
    """(1) best fitness per generation, one line per island; real solid / surrogate dashed."""
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for isl in islands:
        st = isl["resume"]
        if not st or not st.get("best_fitness_history"):
            continue
        h = st["best_fitness_history"]
        gens = [g for g, _ in h]
        fit = [f for _, f in h]
        ax.plot(gens, fit, ls="--" if isl["surrogate"] else "-",
                marker="", lw=1.6, alpha=0.85, label=isl["label"])
        plotted += 1
    ax.set_title(f"(1) Best fitness vs generation — {pair}\n(solid=real island, dashed=surrogate null)")
    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness  (min-WF-chunk · n_trades^exp)")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=7, ncol=2, loc="best")
    else:
        ax.text(0.5, 0.5, "no fitness history yet", ha="center", transform=ax.transAxes)
    return _save(fig, out_dir, "01_fitness_vs_gen.png")


def _best_real_island(islands):
    """The real (non-surrogate) island with the highest validation amddp/day."""
    cands = [i for i in islands if not i["surrogate"] and i["atb"] is not None]
    if not cands:
        return None

    def key(i):
        return i["atb"].get("stats", {}).get("val", {}).get("aggregates", {}).get(
            "amddp_per_day", float("-inf"))
    return max(cands, key=key)


def fig_cumulative_amddp(islands, out_dir, pair):
    """(2) running-best VAL cumulative AMDDP5 equity curve + IS overlay (overfit gap)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    best = _best_real_island(islands)
    if best is None:
        ax.text(0.5, 0.5, "no real-island bundle yet", ha="center", transform=ax.transAxes)
        ax.set_title(f"(2) Running-best cumulative AMDDP5 — {pair}")
        return _save(fig, out_dir, "02_cum_amddp.png")
    b = best["atb"]
    is_arr = b["stats"]["is"]["arrays"]["amddp5"]
    val_arr = b["stats"]["val"]["arrays"]["amddp5"]
    if len(is_arr):
        ax.plot(np.arange(1, len(is_arr) + 1), np.cumsum(is_arr),
                lw=1.8, color="tab:blue", label=f"IS (n={len(is_arr)})")
    if len(val_arr):
        ax.plot(np.arange(1, len(val_arr) + 1), np.cumsum(val_arr),
                lw=1.8, color="tab:orange", label=f"VAL/OOS (n={len(val_arr)})")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    g = best["atb"]["generation"]
    ax.set_title(f"(2) Running-best cumulative AMDDP5 — {pair}  [{best['label']} gen{g}]\n"
                 "IS vs VAL — the gap is the overfit gap")
    ax.set_xlabel("trade #")
    ax.set_ylabel("cumulative AMDDP5 (pips)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "02_cum_amddp.png")


def fig_distributions(islands, out_dir, pair):
    """(3) per-trade AMDDP5 / drawdown / hold-time histograms for the running-best island."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    best = _best_real_island(islands)
    if best is None:
        for ax in axes:
            ax.text(0.5, 0.5, "no bundle yet", ha="center", transform=ax.transAxes)
        fig.suptitle(f"(3) Per-trade distributions — {pair}")
        return _save(fig, out_dir, "03_distributions.png")
    arr = best["atb"]["stats"]["val"]["arrays"]
    specs = [
        ("amddp5", "AMDDP5 per trade (pips)", "tab:orange"),
        ("cum_dd", "accumulated drawdown (pip·bars)", "tab:red"),
        ("hold_min", "time-in-trade (minutes)", "tab:green"),
    ]
    for ax, (key, xlabel, color) in zip(axes, specs):
        data = np.asarray(arr.get(key, []), dtype=float)
        if data.size:
            ax.hist(data, bins=40, color=color, alpha=0.8)
            ax.axvline(float(np.median(data)), color="k", ls="--", lw=1,
                       label=f"med={np.median(data):.2f}")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no trades", ha="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
    fig.suptitle(f"(3) Per-trade VAL distributions — {pair}  [{best['label']}]")
    return _save(fig, out_dir, "03_distributions.png")


def fig_metric_trend(islands, out_dir, pair):
    """(4) Sharpe / Calmar / expectancy / SQN of the running-best island vs generation."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    best = _best_real_island(islands)
    metrics = [("sharpe", "Sharpe"), ("calmar", "Calmar"),
               ("expectancy", "expectancy (pips/trade)"), ("sqn", "SQN")]
    if best is None or not best["gens"]:
        for ax, (_, title) in zip(axes.ravel(), metrics):
            ax.text(0.5, 0.5, "no gen history yet", ha="center", transform=ax.transAxes)
            ax.set_title(title)
        fig.suptitle(f"(4) Aggregate-metric trend — {pair}")
        return _save(fig, out_dir, "04_metric_trend.png")
    gens = best["gens"]
    x = [g for g, _ in gens]

    def clean(vals):
        # guard against inf (profit_factor with no losses etc.)
        return [v if np.isfinite(v) else np.nan for v in vals]
    for ax, (key, title) in zip(axes.ravel(), metrics):
        is_y = clean([lt["is_agg"].get(key, np.nan) for _, lt in gens])
        val_y = clean([lt["val_agg"].get(key, np.nan) for _, lt in gens])
        ax.plot(x, is_y, color="tab:blue", lw=1.5, label="IS")
        ax.plot(x, val_y, color="tab:orange", lw=1.5, label="VAL")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("generation")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"(4) Running-best aggregate-metric trend — {pair}  [{best['label']}]")
    return _save(fig, out_dir, "04_metric_trend.png")


def fig_real_vs_surrogate(islands, out_dir, pair):
    """(5) real-best vs surrogate-null fitness band — are we beating noise-evolution?"""
    fig, ax = plt.subplots(figsize=(9, 5))

    def fit_curve(isl):
        st = isl["resume"]
        if not st or not st.get("best_fitness_history"):
            return None, None
        h = st["best_fitness_history"]
        return [g for g, _ in h], [f for _, f in h]

    real = [i for i in islands if not i["surrogate"]]
    sur = [i for i in islands if i["surrogate"]]

    # real island fitness envelope (max over islands per gen) — the "real best" line
    def envelope(group):
        by_gen = {}
        for isl in group:
            gx, gy = fit_curve(isl)
            if gx is None:
                continue
            for g, f in zip(gx, gy):
                by_gen[g] = max(by_gen.get(g, float("-inf")), f)
        if not by_gen:
            return None, None
        xs = sorted(by_gen)
        return xs, [by_gen[g] for g in xs]

    rx, ry = envelope(real)
    sx, sy = envelope(sur)
    if rx:
        ax.plot(rx, ry, color="tab:green", lw=2.2, label="REAL best (envelope)")
    if sx:
        ax.plot(sx, sy, color="tab:red", lw=2.0, ls="--", label="SURROGATE null (envelope)")
        # shade the band between them where both exist
        common = sorted(set(rx or []) & set(sx or []))
        if common and rx:
            rmap, smap = dict(zip(rx, ry)), dict(zip(sx, sy))
            ax.fill_between(common, [smap[g] for g in common], [rmap[g] for g in common],
                            where=[rmap[g] >= smap[g] for g in common],
                            color="tab:green", alpha=0.15, label="real beats null")
    ax.set_title(f"(5) Real-best vs surrogate-null fitness — {pair}\n"
                 "a real edge must clear the noise-evolution band")
    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness")
    ax.grid(alpha=0.3)
    if rx or sx:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no fitness history yet", ha="center", transform=ax.transAxes)
    return _save(fig, out_dir, "05_real_vs_surrogate.png")


def fig_summary_card(islands, out_dir, pair):
    """(6) deploy-bundle summary card for the running-best real island (text-as-image)."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis("off")
    best = _best_real_island(islands)
    if best is None:
        ax.text(0.5, 0.5, "no real-island deploy bundle yet", ha="center")
        return _save(fig, out_dir, "06_summary_card.png")
    b = best["atb"]
    isa = b["stats"]["is"]["aggregates"]
    va = b["stats"]["val"]["aggregates"]
    g = b["gates"]

    def fmt(d, k, p=3):
        v = d.get(k, float("nan"))
        if not np.isfinite(v):
            return "inf"
        return f"{v:.{p}f}"

    lines = []
    lines.append(f"DEPLOY-BUNDLE SUMMARY  —  {pair}")
    lines.append(f"island={b.get('island')}  seed={b.get('seed')}  "
                 f"exp={b.get('exponent')}  gen={b.get('generation')}")
    lines.append(f"schema={b.get('schema', '?')}   git={b.get('code_version', {}).get('git_commit', '?')}")
    lines.append("")
    lines.append(f"{'metric':<18}{'IS':>14}{'VAL/OOS':>14}")
    lines.append("-" * 46)
    rows = [("amddp/day", "amddp_per_day"), ("pnl/day", "pnl_per_day"),
            ("trades/day", "trades_per_day"), ("Sharpe", "sharpe"),
            ("Calmar", "calmar"), ("expectancy", "expectancy"),
            ("SQN", "sqn"), ("profit factor", "profit_factor"),
            ("win rate %", "win_rate"), ("mean dd", "mean_dd"),
            ("mean hold min", "mean_hold_min"), ("n trades", "n")]
    for label, key in rows:
        p = 0 if key == "n" else 3
        lines.append(f"{label:<18}{fmt(isa, key, p):>14}{fmt(va, key, p):>14}")
    lines.append("")
    lines.append("GATES")
    lines.append("-" * 46)
    lines.append(f"  WF chunks ({len(g.get('wf_chunks', []))}): "
                 + ", ".join(f"{c['score']:.1f}(n={c['n']})" for c in g.get("wf_chunks", [])))
    lines.append(f"  WF all positive : {g.get('wf_all_positive')}")
    lines.append(f"  MC p-value (val): {g.get('mc_pvalue_val'):.4f}  pass={g.get('mc_pass')}")
    snull = g.get("surrogate_null_amddp_per_day")
    lines.append(f"  surrogate-null  : {snull if snull is not None else 'n/a (run collect_winners)'}")
    lines.append(f"  VALIDATED       : {g.get('validated')}")
    lines.append("")
    lines.append("NOTE: single-pair probe — a winner is a HYPOTHESIS.")
    lines.append("Cross-pair + sealed-OOS + R7 consistency required before deploy.")

    ax.text(0.02, 0.98, "\n".join(lines), family="monospace", fontsize=10,
            va="top", ha="left", transform=ax.transAxes)
    return _save(fig, out_dir, "06_summary_card.png")


# ── Render + (throttled) send cycle ──────────────────────────────────────────────
def latest_max_gen(islands):
    g = -1
    for isl in islands:
        st = isl["resume"]
        if st and "generation" in st:
            g = max(g, int(st["generation"]))
    return g


def render_all(runs_root, pair, out_dir):
    islands = gather(runs_root, pair)
    if not islands:
        print(f"[monitor] no island artifacts under {runs_root} (pair={pair}) yet")
        return [], -1
    figs = [
        fig_fitness_vs_gen(islands, out_dir, pair),
        fig_cumulative_amddp(islands, out_dir, pair),
        fig_distributions(islands, out_dir, pair),
        fig_metric_trend(islands, out_dir, pair),
        fig_real_vs_surrogate(islands, out_dir, pair),
        fig_summary_card(islands, out_dir, pair),
    ]
    return figs, latest_max_gen(islands)


def send_all(figs, pair, gen):
    captions = [
        f"(1) Fitness vs gen — {pair} @ gen {gen}",
        f"(2) Running-best cumulative AMDDP5 (IS vs VAL) — {pair}",
        f"(3) Per-trade distributions — {pair}",
        f"(4) Aggregate-metric trend — {pair}",
        f"(5) Real-best vs surrogate-null — {pair}",
        f"(6) Deploy-bundle summary card — {pair}",
    ]
    for path, cap in zip(figs, captions):
        send_photo(path, cap)


def main():
    ap = argparse.ArgumentParser(description="Headless campaign monitor → Telegram graphs.")
    ap.add_argument("--runs", default=os.path.join(HERE, "campaign_runs"),
                    help="campaign_runs root (contains <pair>/<island_tag>/)")
    ap.add_argument("--pair", default="GBP_JPY")
    ap.add_argument("--interval-min", type=float, default=10.0,
                    help="minimum minutes between Telegram pushes")
    ap.add_argument("--interval-gens", type=int, default=10,
                    help="also push when the max generation advances by this many")
    ap.add_argument("--out", default=None, help="PNG output dir (default <runs>/_monitor)")
    ap.add_argument("--once", action="store_true", help="render once and exit (no loop)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.runs, "_monitor")
    os.makedirs(out_dir, exist_ok=True)

    if args.once:
        figs, gen = render_all(args.runs, args.pair, out_dir)
        if figs:
            print(f"[monitor] rendered {len(figs)} PNGs (max gen {gen}):")
            for f in figs:
                print(f"    {f}")
            send_all(figs, args.pair, gen)
        return

    last_send = 0.0
    last_gen = -1
    print(f"[monitor] loop: runs={args.runs} pair={args.pair} "
          f"interval={args.interval_min}min or +{args.interval_gens} gens")
    while True:
        figs, gen = render_all(args.runs, args.pair, out_dir)
        now = time.time()
        due_time = (now - last_send) >= args.interval_min * 60
        due_gens = gen >= 0 and (gen - last_gen) >= args.interval_gens
        if figs and (due_time or due_gens):
            send_all(figs, args.pair, gen)
            last_send = now
            last_gen = gen
        time.sleep(max(15.0, args.interval_min * 60 / 4))


if __name__ == "__main__":
    main()
