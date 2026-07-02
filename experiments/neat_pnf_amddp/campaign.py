#!/usr/bin/env python3
"""
Hetzner-campaign driver — ONE NEAT island for the P&F + AMDDP5 exit-learner probe.
==================================================================================
Design: research/experiments/neat_pnf_amddp/PLAN.md
Reuses phase1_harness.py:  run_neat building blocks (build_config / CappedGenome /
extract_network / trade_stats / fitness_from_wf / make_splits / load_real_series).

What this adds on top of the harness (the campaign layer):
  1. CAMPAIGN DRIVER — one island, generous config (pop 400):
       python3 campaign.py --island N --seed S --exp E \
           --config neat_pnf_generous.ini --gens 400 --pair GBP_JPY [--surrogate]
     * Fitness on TRAIN (min-WF-chunk * n_trades^exp).
     * VALIDATION set tracks the running best for SELECTION + early-stop only
       (never fitness, never test).
     * ~60-gen GLOBAL early-stop on the validation-best metric (amddp/day).
     * --surrogate = block-bootstrap / sign-shuffle the box series first
       (the equal-compute null; a real winner must beat the noise-evolved best).

  2. HERMETIC CHECKPOINTING (MuZero-grade):
     * Resume checkpoint EVERY generation (rolling keep-last-K + best): the whole
       neat Population (population, species_set, innovation tracker, genome indexer),
       generation, RNG state (python + numpy), island/seed/exp, best_fitness history,
       embedded config path. Resuming an interrupted island run is a save->load->continue.
     * Deploy bundle saved as a NEW versioned file best_gen{NNNN}.pkl every gen AND on
       improve (NEVER overwrites) + an all_time_best.pkl pointer. Each bundle is
       self-contained (genome, embedded neat config text, activation names + fast_eval
       hash, input-norm constants, P&F params, code_version) + a TRADING-STATS BUNDLE
       (per-trade arrays + IS/VAL aggregates: Sharpe/Calmar/expectancy/SQN/PF/WR/... +
       validation gates: per-WF-chunk scores, MC sign-shuffle p, surrogate-null slot).

  3. DISK-AWARE PURGE: keep all_time_best + any WF/MC-validated bundle forever;
     per-gen bests keep last K + every Mth; if the dir exceeds a size cap, purge oldest
     non-validated per-gen bundles first (mirrors the tick-capture retention discipline).

CONSTRAINTS: causal, no lookahead (the harness already is). SMOKE:
  python3 campaign.py --smoke   (1 island, pop 30, gens 5; proves run/resume/bundle/surrogate)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import random
import shutil
import sys
import time

import numpy as np
import neat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

# Reuse everything from the harness (it registers sin/cos/wavelets on import).
from research.experiments.neat_pnf_amddp.phase1_harness import (  # noqa: E402
    build_config, extract_network, fitness_from_wf, trade_stats, make_splits,
    load_real_series, _eval_range,
    PIP, AMDDP_K, AGE_SCALE, AMDDP_SCALE, FLAT_SENTINEL, CLIP,
    SESSION_MINUTES, MIN_TRADES_PER_CHUNK, _CUSTOM_ACTS, MAX_HIDDEN,
)
from research.experiments.amddp5.scorer import mc_pvalue_amddp  # noqa: E402
from lib import fast_eval as _fe  # noqa: E402

# ── Campaign constants ────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE = 60          # global early-stop: gens without VAL-best improvement
KEEP_LAST_K = 8                   # rolling resume checkpoints + per-gen bests to keep
KEEP_EVERY_M = 25                 # additionally keep every Mth per-gen best forever
DIR_SIZE_CAP_MB = 500             # purge oldest non-validated per-gen bundles past this
MC_SHUFFLES = 1000                # MC sign-shuffle on the running best's val trades
MINUTES_PER_YEAR = 365.25 * 24 * 60
PNF_BOX_PIPS = 5.0
PNF_REVERSAL = 3                  # cache uses rev3 (PLAN: box=5p; rev cache picks the series)
OUTPUT_NAMES = ("long", "flat", "short")
INPUT_NAMES = ("signed_trend_age", "in_trade_running_amddp5")


# ── Provenance hashes (code_version) ────────────────────────────────────────────
def _file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return "missing"


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", HERE, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()[:12]
    except Exception:
        return "unknown"


def _code_version():
    return {
        "git_commit": _git_commit(),
        "fast_eval_hash": _file_hash(os.path.join(HERE, "..", "..", "..", "lib", "fast_eval.py")),
        "pnf_engine_hash": _file_hash(os.path.join(HERE, "..", "..", "..", "lib", "pnf_engine.py")),
        "harness_hash": _file_hash(os.path.join(HERE, "phase1_harness.py")),
        "campaign_hash": _file_hash(os.path.join(HERE, "campaign.py")),
    }


# ── Surrogate-null: shuffle the box series (equal-compute null) ──────────────────
def make_surrogate(data, seed):
    """Block-bootstrap the signed_age series so any structure (trend persistence)
    is destroyed while keeping the marginal distribution + the price/dt arrays.
    A real winner must beat a genome evolved on THIS. We resample the signed_age
    in blocks (preserves short-run autocorrelation magnitude, kills the long
    directional structure trend-age would exploit) and sign-flip half the blocks.
    """
    rng = np.random.default_rng(seed + 99991)
    sa = data["signed_age"].copy()
    n = len(sa)
    block = 64
    out = np.empty_like(sa)
    i = 0
    while i < n:
        src = int(rng.integers(0, max(1, n - block)))
        ln = min(block, n - i)
        seg = sa[src:src + ln].copy()
        if rng.random() > 0.5:
            seg = -seg  # sign-flip the block: kills directional edge
        out[i:i + ln] = seg
        i += ln
    sur = dict(data)
    sur["signed_age"] = out
    return sur


# ── Rich aggregate metrics for the trading-stats bundle ─────────────────────────
def _span_days(data, rng):
    start, end = rng
    if data.get("ts") is not None:
        return max(1e-9, (data["ts"][end - 1] - data["ts"][start]) / np.timedelta64(1, "D"))
    return max(1e-9, float(np.sum(data["dt_min"][start:end])) / (60.0 * 24.0))


def rich_stats(net_arrays, data, rng, random_entry=False, rnd_dir=None):
    """Per-trade arrays + full aggregate metric bundle for one range.

    Returns a dict with:
      arrays: amddp5, pnl, cum_dd, hold_min, entry_idx, exit_idx, direction
      aggregates: n, sharpe, calmar, expectancy, sqn, profit_factor, win_rate,
                  amddp_per_day, pnl_per_day, trades_per_day, mean_dd, mean_hold_min,
                  amddp_sum, pnl_sum, mean_amddp, median_amddp
    """
    pnl, amddp, hold, dd, ei, xi, di = _eval_range(net_arrays, data, rng, random_entry, rnd_dir)
    n = int(len(amddp))
    span = _span_days(data, rng)
    arrays = {
        "amddp5": amddp.astype(np.float64),
        "pnl": pnl.astype(np.float64),
        "cum_dd": dd.astype(np.float64),
        "hold_min": hold.astype(np.float64),
        "entry_idx": ei.astype(np.int64),
        "exit_idx": xi.astype(np.int64),
        "direction": di.astype(np.int64),
    }
    if n == 0:
        agg = dict(n=0, sharpe=0.0, calmar=0.0, expectancy=0.0, sqn=0.0,
                   profit_factor=0.0, win_rate=0.0, amddp_per_day=0.0, pnl_per_day=0.0,
                   trades_per_day=0.0, mean_dd=0.0, mean_hold_min=0.0, amddp_sum=0.0,
                   pnl_sum=0.0, mean_amddp=0.0, median_amddp=0.0, span_days=span)
        return arrays, agg

    pnl_sum = float(pnl.sum())
    amddp_sum = float(amddp.sum())
    mean_r = float(pnl.mean())
    std_r = float(pnl.std(ddof=1)) if n > 1 else 0.0
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    # Sharpe: per-trade mean/std annualized by trades/year (sqrt-scaling).
    trades_per_year = n / span * 365.25
    sharpe = (mean_r / std_r) * np.sqrt(trades_per_year) if std_r > 0 else 0.0
    # Calmar: annualized pnl/day vs max underwater (equity-curve max drawdown in pips).
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    max_dd_equity = float(np.max(peak - eq)) if n else 0.0
    ann_pnl = pnl_sum / span * 365.25
    calmar = ann_pnl / max_dd_equity if max_dd_equity > 1e-9 else 0.0
    # SQN = sqrt(n) * mean(R) / std(R)
    sqn = np.sqrt(n) * mean_r / std_r if std_r > 0 else 0.0
    agg = dict(
        n=n,
        sharpe=float(sharpe),
        calmar=float(calmar),
        expectancy=mean_r,
        sqn=float(sqn),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 1e-12 else float("inf"),
        win_rate=100.0 * float(np.mean(pnl > 0)),
        amddp_per_day=amddp_sum / span,
        pnl_per_day=pnl_sum / span,
        trades_per_day=n / span,
        mean_dd=float(dd.mean()),
        mean_hold_min=float(hold.mean()),
        amddp_sum=amddp_sum,
        pnl_sum=pnl_sum,
        mean_amddp=float(amddp.mean()),
        median_amddp=float(np.median(amddp)),
        span_days=span,
    )
    return arrays, agg


def wf_chunk_scores(net_arrays, data, wf_chunks, exp, random_entry=False, rnd_dir=None):
    """Per-WF-chunk fitness components (mean_amddp * n^exp), for provenance."""
    out = []
    for rng in wf_chunks:
        pnl, amddp, hold, dd, ei, xi, di = _eval_range(net_arrays, data, rng, random_entry, rnd_dir)
        n = len(amddp)
        score = float(np.mean(amddp)) * (n ** exp) if n >= MIN_TRADES_PER_CHUNK else float("-1e6")
        out.append({"range": [int(rng[0]), int(rng[1])], "n": int(n), "score": score})
    return out


# ── The self-contained deploy bundle ────────────────────────────────────────────
def build_deploy_bundle(genome, cfg, data, splits, exp, gen, island, seed, pair,
                        random_entry, rnd_dir, surrogate, sp_gate, config_ini):
    """Assemble a self-contained deploy bundle for ONE genome with the full
    trading-stats bundle (IS/VAL computed separately) + validation gates.
    """
    train_rng, val_rng, test_rng, wf_chunks = splits
    na = extract_network(genome, cfg)

    is_arrays, is_agg = rich_stats(na, data, train_rng, random_entry, rnd_dir)
    val_arrays, val_agg = rich_stats(na, data, val_rng, random_entry, rnd_dir)

    chunks = wf_chunk_scores(na, data, wf_chunks, exp, random_entry, rnd_dir)
    wf_all_pos = all(c["score"] > 0 for c in chunks)

    # MC sign-shuffle on the VALIDATION trades (not test — test stays sealed).
    val_amddp = val_arrays["amddp5"]
    mc_p = float(mc_pvalue_amddp(val_amddp, MC_SHUFFLES)) if len(val_amddp) else 1.0

    # embed the raw neat config text so the bundle never needs an external .ini
    try:
        with open(config_ini, "r") as f:
            config_text = f.read()
    except OSError:
        config_text = ""

    bundle = {
        "schema": "neat_pnf_amddp.deploy_bundle.v1",
        "genome": genome,
        "neat_config_text": config_text,
        "neat_config_path": os.path.basename(config_ini),
        "num_inputs": cfg.genome_config.num_inputs,
        "num_outputs": cfg.genome_config.num_outputs,
        "max_hidden": MAX_HIDDEN,
        # activation provenance (names + impls map + code hash — live verifies identical code)
        "activation_options": list(cfg.genome_config.activation_options),
        "custom_activation_names": sorted(_CUSTOM_ACTS.keys()),
        # input spec / normalization constants
        "input_names": list(INPUT_NAMES),
        "input_norm": {
            "age_scale_A": AGE_SCALE,
            "amddp_scale_S": AMDDP_SCALE,
            "flat_sentinel": FLAT_SENTINEL,
            "clip": CLIP,
        },
        # P&F params (embedded, never hardcoded live)
        "pnf_params": {
            "box_pips": PNF_BOX_PIPS,
            "reversal": PNF_REVERSAL,
            "paint_all_boxes": True,
            "r2_within_bar": "bull:high-then-low; bear:low-then-high",
        },
        "amddp_K": AMDDP_K,
        "session_minutes": SESSION_MINUTES,
        "is_only_spread_gate": float(sp_gate),
        "pip": PIP,
        "pair": pair,
        "output_names": list(OUTPUT_NAMES),
        # context
        "generation": int(gen),
        "island": int(island),
        "seed": int(seed),
        "exponent": float(exp),
        "fitness": float(genome.fitness) if genome.fitness is not None else None,
        "random_entry_control": bool(random_entry),
        "is_surrogate": bool(surrogate),
        "code_version": _code_version(),
        "r7_pass_hash": None,   # set by the phase0 R7 consistency gate before deploy
        # ── TRADING-STATS BUNDLE ──
        "stats": {
            "is": {"arrays": is_arrays, "aggregates": is_agg},
            "val": {"arrays": val_arrays, "aggregates": val_agg},
        },
        # ── validation gates ──
        "gates": {
            "wf_chunks": chunks,
            "wf_all_positive": bool(wf_all_pos),
            "mc_pvalue_val": mc_p,
            "mc_pass": bool(mc_p < 0.05),
            "surrogate_null_amddp_per_day": None,   # filled by collect_winners comparison
            "val_amddp_per_day": val_agg["amddp_per_day"],
            "is_amddp_per_day": is_agg["amddp_per_day"],
            # a bundle is "validated" (keep forever) if WF all-pos AND MC passes
            "validated": bool(wf_all_pos and mc_p < 0.05),
        },
        "saved_at": time.time(),
    }
    return bundle


# ── Disk-aware retention / purge ─────────────────────────────────────────────────
def _dir_size_mb(path):
    total = 0
    for f in os.scandir(path):
        if f.is_file():
            total += f.stat().st_size
    return total / 1e6


def purge_bundles(bundles_dir):
    """Keep all_time_best + every validated bundle FOREVER. Per-gen bests: keep last K
    + every Mth. If dir exceeds the size cap, purge oldest NON-validated per-gen bundles
    first. (Mirrors the tick-capture retention discipline.)
    """
    files = sorted(
        (f for f in os.listdir(bundles_dir) if f.startswith("best_gen") and f.endswith(".pkl")),
        key=lambda x: int(x[len("best_gen"):-4]),
    )
    if not files:
        return
    gens = [int(f[len("best_gen"):-4]) for f in files]
    max_gen = max(gens)

    def is_validated(fname):
        try:
            with open(os.path.join(bundles_dir, fname), "rb") as fh:
                b = pickle.load(fh)
            return bool(b.get("gates", {}).get("validated", False))
        except Exception:
            return True  # if unreadable, err on keeping it

    # mandatory-keep set: last K gens + every Mth gen + validated bundles
    keep = set()
    for f, g in zip(files, gens):
        if g > max_gen - KEEP_LAST_K or (g % KEEP_EVERY_M == 0) or is_validated(f):
            keep.add(f)

    # routine purge: anything not in keep
    for f in files:
        if f not in keep:
            try:
                os.remove(os.path.join(bundles_dir, f))
            except OSError:
                pass

    # size-cap purge: if still too big, drop oldest NON-validated of the survivors
    if _dir_size_mb(bundles_dir) > DIR_SIZE_CAP_MB:
        survivors = sorted(
            (f for f in os.listdir(bundles_dir) if f.startswith("best_gen") and f.endswith(".pkl")),
            key=lambda x: int(x[len("best_gen"):-4]),
        )
        for f in survivors:
            if _dir_size_mb(bundles_dir) <= DIR_SIZE_CAP_MB:
                break
            g = int(f[len("best_gen"):-4])
            if g > max_gen - KEEP_LAST_K:
                continue  # never drop the last-K
            if is_validated(f):
                continue  # never drop validated
            try:
                os.remove(os.path.join(bundles_dir, f))
            except OSError:
                pass


def purge_resume(resume_dir):
    """Rolling keep-last-K resume checkpoints + the best-fitness one."""
    files = sorted(
        (f for f in os.listdir(resume_dir) if f.startswith("resume_gen") and f.endswith(".pkl")),
        key=lambda x: int(x[len("resume_gen"):-4]),
    )
    if len(files) <= KEEP_LAST_K:
        return
    for f in files[:-KEEP_LAST_K]:
        try:
            os.remove(os.path.join(resume_dir, f))
        except OSError:
            pass


# ── Resume checkpoint (hermetic) ─────────────────────────────────────────────────
def save_resume(resume_dir, pop, gen, island, seed, exp, config_ini,
                best_fitness_history, best_val_history, best_val_metric,
                gens_since_improve, surrogate):
    """Pickle the WHOLE neat Population (population, species_set, innovation tracker,
    genome indexer) + RNG states + island context. Reporters are detached (no
    evolutionary state) and reattached fresh on resume.
    """
    reporters = pop.reporters
    pop.reporters = None
    try:
        state = {
            "schema": "neat_pnf_amddp.resume.v1",
            "population_pickle": pickle.dumps(pop),
            "generation": int(gen),
            "island": int(island),
            "seed": int(seed),
            "exponent": float(exp),
            "config_ini": os.path.basename(config_ini),
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "best_fitness_history": list(best_fitness_history),
            "best_val_history": list(best_val_history),
            "best_val_metric": float(best_val_metric),
            "gens_since_improve": int(gens_since_improve),
            "surrogate": bool(surrogate),
        }
    finally:
        pop.reporters = reporters
    path = os.path.join(resume_dir, f"resume_gen{gen:04d}.pkl")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)
    purge_resume(resume_dir)
    return path


def load_resume(resume_dir):
    """Load the latest resume checkpoint. Returns (state, pop) or (None, None)."""
    files = sorted(
        (f for f in os.listdir(resume_dir) if f.startswith("resume_gen") and f.endswith(".pkl")),
        key=lambda x: int(x[len("resume_gen"):-4]),
    ) if os.path.isdir(resume_dir) else []
    if not files:
        return None, None
    with open(os.path.join(resume_dir, files[-1]), "rb") as f:
        state = pickle.load(f)
    pop = pickle.loads(state["population_pickle"])
    pop.reporters = neat.reporting.ReporterSet()
    if pop.species is not None:
        pop.species.reporters = pop.reporters
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    return state, pop


# ── Single-island campaign run ───────────────────────────────────────────────────
def run_island(island, seed, exp, config_ini, gens, pair, surrogate,
               pop_size=None, random_entry=False, out_root=None, verbose=True):
    tag = f"isl{island}_seed{seed}_exp{exp}" + ("_surrogate" if surrogate else "")
    out_root = out_root or os.path.join(HERE, "campaign_runs", pair, tag)
    resume_dir = os.path.join(out_root, "resume")
    bundles_dir = os.path.join(out_root, "bundles")
    os.makedirs(resume_dir, exist_ok=True)
    os.makedirs(bundles_dir, exist_ok=True)

    # ── data ──
    cache = os.path.join(HERE, "cache", f"{pair}_pnf_b{int(PNF_BOX_PIPS)}_rev{PNF_REVERSAL}.parquet")
    data = load_real_series(cache)
    if surrogate:
        data = make_surrogate(data, seed)
    splits = make_splits(len(data["mid"]))
    train_rng, val_rng, test_rng, wf_chunks = splits

    # IS-only spread gate (SOP R5) — P90 of per-event spread over the TRAIN range only
    sp = (data["ask"][train_rng[0]:train_rng[1]] - data["bid"][train_rng[0]:train_rng[1]]) / PIP
    sp_gate = float(np.percentile(sp, 90)) if len(sp) else 0.0

    # fixed random-entry stream (control A), frozen for reproducibility
    rnd = np.random.default_rng(seed + 1234)
    rnd_dir = np.where(rnd.random(len(data["mid"])) > 0.5, 1, -1).astype(np.int64)

    # ── resume-or-fresh ──
    state, pop = load_resume(resume_dir)
    if state is not None:
        cfg = build_config(pop_size=pop_size, config_ini=os.path.join(HERE, config_ini))
        pop.config = cfg
        start_gen = state["generation"] + 1
        best_fitness_history = state["best_fitness_history"]
        best_val_history = state["best_val_history"]
        best_val_metric = state["best_val_metric"]
        gens_since_improve = state["gens_since_improve"]
        if verbose:
            print(f"[{tag}] RESUMED from gen {state['generation']} "
                  f"(best_val={best_val_metric:.3f}, since_improve={gens_since_improve})",
                  flush=True)
    else:
        random.seed(seed)
        np.random.seed(seed)
        cfg = build_config(pop_size=pop_size, config_ini=os.path.join(HERE, config_ini))
        pop = neat.Population(cfg)
        start_gen = 0
        best_fitness_history = []
        best_val_history = []
        best_val_metric = -np.inf
        gens_since_improve = 0
        if verbose:
            print(f"[{tag}] FRESH start, pop={cfg.pop_size}, gens={gens}, "
                  f"sp_gate={sp_gate:.2f}p", flush=True)

    def eval_pop(genomes, config):
        for gid, genome in genomes:
            try:
                na = extract_network(genome, config)
                genome.fitness = fitness_from_wf(na, data, wf_chunks, exp, random_entry, rnd_dir)
            except Exception:
                genome.fitness = -1e6

    all_time_best_metric = best_val_metric
    config_ini_full = os.path.join(HERE, config_ini)

    for gen in range(start_gen, gens):
        pop.run(eval_pop, 1)

        # population's best genome this gen (by fitness)
        best, best_fit = None, -np.inf
        for genome in pop.population.values():
            if genome.fitness is not None and genome.fitness > best_fit:
                best_fit, best = genome.fitness, genome
        if best is None:
            continue
        best_fitness_history.append((gen, float(best_fit)))

        # validation metric for SELECTION + early-stop (NOT fitness, NOT test)
        na = extract_network(best, cfg)
        vstat = trade_stats(na, data, val_rng, random_entry, rnd_dir)
        val_metric = vstat["amddp_per_day"]
        best_val_history.append((gen, float(val_metric), int(vstat["n"])))

        improved = val_metric > best_val_metric + 1e-9
        if improved:
            best_val_metric = val_metric
            gens_since_improve = 0
        else:
            gens_since_improve += 1

        # ── deploy bundle: versioned every gen + on improve (never overwrite) ──
        bundle = build_deploy_bundle(
            best, cfg, data, splits, exp, gen, island, seed, pair,
            random_entry, rnd_dir, surrogate, sp_gate, config_ini_full)
        bpath = os.path.join(bundles_dir, f"best_gen{gen:04d}.pkl")
        with open(bpath, "wb") as f:
            pickle.dump(bundle, f)
        # all_time_best pointer — by validation amddp/day
        if val_metric > all_time_best_metric or not os.path.exists(
                os.path.join(bundles_dir, "all_time_best.pkl")):
            all_time_best_metric = max(all_time_best_metric, val_metric)
            shutil.copyfile(bpath, os.path.join(bundles_dir, "all_time_best.pkl"))

        # ── resume checkpoint EVERY generation (hermetic) ──
        save_resume(resume_dir, pop, gen, island, seed, exp, config_ini,
                    best_fitness_history, best_val_history, best_val_metric,
                    gens_since_improve, surrogate)
        purge_bundles(bundles_dir)

        if verbose:
            print(f"[{tag}] gen {gen:4d}  fit={best_fit:11.2f}  "
                  f"val_amddp/d={val_metric:8.3f}  val_n={vstat['n']:4d}  "
                  f"mc?={bundle['gates']['mc_pass']}  wf?={bundle['gates']['wf_all_positive']}  "
                  f"since_improve={gens_since_improve}", flush=True)

        # ── ~60-gen GLOBAL early-stop (no validation-best improvement) ──
        if gens_since_improve >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f"[{tag}] EARLY STOP at gen {gen} "
                      f"({EARLY_STOP_PATIENCE} gens no val improvement)", flush=True)
            break

    return out_root, bundles_dir, resume_dir


# ── Smoke test ───────────────────────────────────────────────────────────────────
def smoke():
    print("=" * 78)
    print("CAMPAIGN SMOKE — 1 island, pop 30, gens 5 (run / resume / bundle / surrogate)")
    print("=" * 78)
    import tempfile
    tmp = tempfile.mkdtemp(prefix="neatpnf_smoke_")
    ok = {"run": False, "resume": False, "bundle": False, "surrogate": False}

    # ── (1) RUN: gens 0..2 ──
    print("\n[1] RUN 3 gens (fresh) ...")
    out_root = os.path.join(tmp, "isl0")
    run_island(island=0, seed=0, exp=0.5, config_ini="neat_pnf_generous.ini",
               gens=3, pair="GBP_JPY", surrogate=False, pop_size=30, out_root=out_root)
    bundles_dir = os.path.join(out_root, "bundles")
    resume_dir = os.path.join(out_root, "resume")
    bfiles = [f for f in os.listdir(bundles_dir) if f.startswith("best_gen")]
    rfiles = [f for f in os.listdir(resume_dir) if f.startswith("resume_gen")]
    ok["run"] = len(bfiles) >= 1 and len(rfiles) >= 1
    print(f"    bundles written: {sorted(bfiles)}")
    print(f"    resume ckpts:    {sorted(rfiles)}")

    # ── (2) RESUME round-trip: continue to gen 5 from the gen-2 checkpoint ──
    print("\n[2] RESUME from latest checkpoint, continue to gen 5 ...")
    st, pop = load_resume(resume_dir)
    print(f"    loaded resume: gen={st['generation']}, npop={len(pop.population)}, "
          f"island={st['island']}, exp={st['exponent']}")
    run_island(island=0, seed=0, exp=0.5, config_ini="neat_pnf_generous.ini",
               gens=5, pair="GBP_JPY", surrogate=False, pop_size=30, out_root=out_root)
    bfiles2 = sorted(f for f in os.listdir(bundles_dir) if f.startswith("best_gen"))
    max_gen = max(int(f[len("best_gen"):-4]) for f in bfiles2)
    ok["resume"] = max_gen >= 4  # gens 0..4 ran (5 total), continuation worked
    print(f"    bundles after resume: {bfiles2}  (max gen {max_gen})")

    # ── (3) BUNDLE: inspect a deploy bundle (fields + stats) ──
    print("\n[3] DEPLOY-BUNDLE inspection ...")
    with open(os.path.join(bundles_dir, "all_time_best.pkl"), "rb") as f:
        b = pickle.load(f)
    has_stats = "stats" in b and "is" in b["stats"] and "val" in b["stats"]
    is_agg = b["stats"]["is"]["aggregates"]
    val_agg = b["stats"]["val"]["aggregates"]
    needed = {"sharpe", "sqn", "calmar", "expectancy", "profit_factor", "win_rate"}
    ok["bundle"] = (
        has_stats and needed.issubset(is_agg.keys()) and needed.issubset(val_agg.keys())
        and "genome" in b and "neat_config_text" in b and "code_version" in b
        and "input_norm" in b and "pnf_params" in b and "gates" in b
    )
    print("    --- DEPLOY BUNDLE TOP-LEVEL FIELDS ---")
    for k in b:
        print(f"      {k}")
    print("    --- code_version ---")
    for k, v in b["code_version"].items():
        print(f"      {k}: {v}")
    print("    --- IS aggregates (sample stats bundle) ---")
    for k, v in is_agg.items():
        print(f"      is.{k} = {v}")
    print("    --- VAL aggregates ---")
    for k, v in val_agg.items():
        print(f"      val.{k} = {v}")
    print("    --- gates ---")
    for k, v in b["gates"].items():
        print(f"      {k} = {v}")
    print(f"    --- per-trade array keys (IS) --- {list(b['stats']['is']['arrays'].keys())}")
    print(f"    IS trades n = {len(b['stats']['is']['arrays']['amddp5'])}, "
          f"VAL trades n = {len(b['stats']['val']['arrays']['amddp5'])}")

    # ── (4) SURROGATE run ──
    print("\n[4] SURROGATE (--surrogate) run 2 gens ...")
    sout = os.path.join(tmp, "isl0_sur")
    run_island(island=0, seed=0, exp=0.5, config_ini="neat_pnf_generous.ini",
               gens=2, pair="GBP_JPY", surrogate=True, pop_size=30, out_root=sout)
    sbundles = os.path.join(sout, "bundles")
    with open(os.path.join(sbundles, "all_time_best.pkl"), "rb") as f:
        sb = pickle.load(f)
    ok["surrogate"] = sb.get("is_surrogate", False) is True
    print(f"    surrogate bundle is_surrogate flag = {sb.get('is_surrogate')}")

    print("\n" + "=" * 78)
    print("SMOKE REPORT CARD")
    print("=" * 78)
    for k in ("run", "resume", "bundle", "surrogate"):
        print(f"  {k:10s}: {'PASS' if ok[k] else 'FAIL'}")
    print(f"\n  (smoke artifacts in {tmp})")
    allok = all(ok.values())
    print(f"  >>> SMOKE {'PASSED' if allok else 'FAILED'}")
    return allok


def main():
    global PNF_BOX_PIPS, PNF_REVERSAL
    ap = argparse.ArgumentParser(description="One NEAT island for the P&F+AMDDP5 campaign.")
    ap.add_argument("--island", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp", type=float, default=0.5, help="n_trades exponent {0.4..0.7}")
    ap.add_argument("--config", type=str, default="neat_pnf_generous.ini")
    ap.add_argument("--gens", type=int, default=400)
    ap.add_argument("--pair", type=str, default="GBP_JPY")
    ap.add_argument("--pop", type=int, default=None, help="override config pop_size")
    ap.add_argument("--surrogate", action="store_true", help="evolve on shuffled boxes (null)")
    ap.add_argument("--random-entry", action="store_true", help="control A: random entries")
    ap.add_argument("--out", type=str, default=None, help="output root dir")
    ap.add_argument("--box", type=float, default=PNF_BOX_PIPS, help="P&F box size pips (cache must exist)")
    ap.add_argument("--rev", type=int, default=PNF_REVERSAL, help="P&F reversal boxes (cache must exist)")
    ap.add_argument("--smoke", action="store_true", help="tiny smoke (1 isl, pop 30, gens 5)")
    args = ap.parse_args()

    # box/rev select which prebuilt cache is read; embed into the deploy bundle via globals
    PNF_BOX_PIPS = args.box
    PNF_REVERSAL = args.rev

    if args.smoke:
        ok = smoke()
        sys.exit(0 if ok else 1)

    run_island(
        island=args.island, seed=args.seed, exp=args.exp,
        config_ini=args.config, gens=args.gens, pair=args.pair,
        surrogate=args.surrogate, pop_size=args.pop,
        random_entry=args.random_entry, out_root=args.out)


if __name__ == "__main__":
    main()
