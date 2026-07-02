#!/usr/bin/env python3
"""
Phase 1 NEAT training harness — P&F + AMDDP5 exit-learner probe (SINGLE PAIR: GBP_JPY).
=======================================================================================
Design: research/experiments/neat_pnf_amddp/PLAN.md

A tiny NEAT net (2 inputs -> <=5 hidden -> 3 outputs) decides a target position at every
P&F box-change event. Reward = AMDDP5 (pnl - 0.05 * cum_dd) per trade. The net learns the
exit (and entry) on the same box series that FIFO-Trends had real edge on.

INPUTS (per box event, causal):
  in1 = tanh(signed_age / 5.0)                 # P&F signed trend-age
  in2 = -1.0 if FLAT, else clip(tanh(running_amddp5 / 20.0), -0.9, 0.9)   # in-trade health

OUTPUTS (argmax -> target position): out0=long(+1), out1=flat(0), out2=short(-1)

SIM (one @njit kernel, inlined forward pass):
  - iterate box events; build inputs; forward pass; argmax -> target_pos.
  - on position change: close any open trade (realized pnl = dir*(mid_exit-mid_entry)/pip
    - spread_entry, spread paid UP FRONT at entry, SOP R3), then open new at event mid.
  - cum_dd = accumulated underwater pip-MINUTES of the open trade (uses dt between events).
  - amddp5 = pnl - 0.05 * cum_dd.
  - SESSION BOUND ~6h: force-close any open trade; the close is REAL and scored normally.

FITNESS (on TRAIN): min over 3 WF chunks of (mean_trade_amddp5 * n_trades^exp), exp default
0.5; hard minimum trades/chunk (30) else large negative.

CONTROLS:
  (B) SINE positive-control: synthetic signed_age driven by a sine so "long when rising,
      short when falling" is profitable net of a small spread. NEAT must learn it (fitness
      climbs). If it can't, the harness is broken.
  (A) RANDOM-ENTRY control: entries random, net only manages exits.

Run:  python3 phase1_harness.py [--sanity]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import neat
from numba import njit

# Reuse the activation registry + network extractor (registers sin/cos/wavelets on import).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from lib.fast_eval import (  # noqa: E402
    extract_network,
    _sin_activation, _cos_activation, _ricker_activation, _morlet_activation,
    _dog_activation, _sech_activation, _sinc_activation,
)

# neat-python 2.0.0 has no DefaultGenome.add_activation classmethod; custom activations
# must be registered into each config's per-instance ActivationFunctionSet. gauss is built-in.
_CUSTOM_ACTS = {
    "sin": _sin_activation, "cos": _cos_activation,
    "ricker": _ricker_activation, "morlet": _morlet_activation,
    "dog": _dog_activation, "sech": _sech_activation, "sinc": _sinc_activation,
}

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "GBP_JPY_pnf_b5_rev3.parquet")
CONFIG_INI = os.path.join(HERE, "neat_pnf_2in_3out.ini")

PIP = 0.01
AMDDP_K = 0.05
AGE_SCALE = 5.0          # in1 = tanh(signed_age / AGE_SCALE)
AMDDP_SCALE = 20.0       # in2 = clip(tanh(running_amddp5 / AMDDP_SCALE), -0.9, 0.9)
FLAT_SENTINEL = -1.0
CLIP = 0.9
SESSION_MINUTES = 6.0 * 60.0   # 6h session bound (force-close)
MIN_TRADES_PER_CHUNK = 30
HARD_NEG = -1e6


# ──────────────────────────────────────────────────────────────────────────────
# Capped genome: hard cap on TOTAL hidden-node COUNT (reject add_node beyond cap).
# This is a budget, NOT a depth/layer limit — NEAT wires the budget into any
# feed-forward DAG it wants (e.g. 4->2->1->1 across several layers). Raising the
# number is a pure capacity increase, never a topology-shape constraint.
# ──────────────────────────────────────────────────────────────────────────────
MAX_HIDDEN = 8


class CappedGenome(neat.genome.DefaultGenome):
    """DefaultGenome that refuses add_node mutations beyond MAX_HIDDEN hidden nodes."""

    def _n_hidden(self, config):
        out = set(config.output_keys)
        # hidden = nodes that are not output nodes (inputs are negative keys, never in self.nodes)
        return sum(1 for k in self.nodes if k not in out)

    def mutate_add_node(self, config):
        if self._n_hidden(config) >= MAX_HIDDEN:
            # Cap reached: substitute a connection mutation instead of growing.
            self.mutate_add_connection(config)
            return
        try:
            super().mutate_add_node(config)
        except AssertionError:
            # neat-python 2.0.0 get_new_node_key() asserts when a crossover child
            # inherited a node key the shared node_indexer later regenerates.
            # Skip this structural mutation rather than crash the island.
            self.mutate_add_connection(config)


# ──────────────────────────────────────────────────────────────────────────────
# Numba simulation kernel — inlines the NEAT forward pass (no Python per-event call).
# ──────────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False)
def _simulate(
    signed_age, mid, bid, ask, dt_min,            # event arrays (float64), dt_min[i]=minutes since prev event
    start, end,                                    # event range [start, end)
    pip, amddp_k, age_scale, amddp_scale, flat_sentinel, clip_v, session_minutes,
    random_entry, rnd_dir,                         # control A: if 1, entries forced to rnd_dir[i] in {-1,+1}
    # network arrays
    n_inputs, n_eval, total_values,
    node_bias, node_response, node_act,
    conn_from, conn_to, conn_weight, output_indices,
):
    """One pass over box events. Returns per-trade arrays + counts.

    Position semantics: target_pos in {-1,0,+1} from argmax(out0=long,out1=flat,out2=short).
    On change: close open trade (realized), open new at event mid (+spread up front at entry).
    cum_dd accumulates underwater pip-minutes; amddp5 = pnl - k*cum_dd.
    Session bound: if cumulative minutes since entry >= session_minutes, force-close (real).
    """
    n = end - start
    values = np.zeros(total_values)

    pnl_arr = np.empty(n, dtype=np.float64)
    amddp_arr = np.empty(n, dtype=np.float64)
    hold_arr = np.empty(n, dtype=np.float64)       # minutes held
    dd_arr = np.empty(n, dtype=np.float64)
    entry_idx_arr = np.empty(n, dtype=np.int64)
    exit_idx_arr = np.empty(n, dtype=np.int64)
    dir_arr = np.empty(n, dtype=np.int64)
    nt = 0

    pos = 0                 # current position -1/0/+1
    entry_mid = 0.0
    spread_entry = 0.0      # pips, paid up front
    entry_event = 0
    cum_dd = 0.0            # underwater pip-minutes
    min_in_trade = 0.0      # minutes since entry

    for i in range(start, end):
        sa = signed_age[i]
        m = mid[i]

        # ── running unrealized AMDDP path update at THIS event (causal) ──
        # Add the underwater contribution of the interval that just elapsed (dt_min[i]).
        if pos != 0:
            min_in_trade += dt_min[i]
            upnl = pos * (m - entry_mid) / pip - spread_entry
            if upnl < 0.0:
                cum_dd += (-upnl) * dt_min[i]
            running_amddp5 = upnl - amddp_k * cum_dd
        else:
            running_amddp5 = 0.0

        # ── build inputs ──
        in1 = np.tanh(sa / age_scale)
        if pos == 0:
            in2 = flat_sentinel
        else:
            t = np.tanh(running_amddp5 / amddp_scale)
            if t > clip_v:
                t = clip_v
            elif t < -clip_v:
                t = -clip_v
            in2 = t
        values[0] = in1
        values[1] = in2

        # ── forward pass (inlined _activate) ──
        for e in range(n_eval):
            values[n_inputs + e] = 0.0
        # accumulate weighted sums
        # (use a local loop; node_sums via re-read is fine for tiny nets)
        for e in range(n_eval):
            s = 0.0
            # sum over connections targeting node e
            for c in range(len(conn_from)):
                if conn_to[c] == e:
                    s += values[conn_from[c]] * conn_weight[c]
            x = node_bias[e] + node_response[e] * s
            act = node_act[e]
            if act == 0:
                values[n_inputs + e] = np.tanh(x)
            elif act == 1:
                values[n_inputs + e] = 1.0 / (1.0 + np.exp(-x))
            elif act == 2:
                values[n_inputs + e] = max(0.0, x)
            elif act == 4:
                values[n_inputs + e] = np.sin(x)
            elif act == 5:
                values[n_inputs + e] = np.cos(x)
            elif act == 6:
                values[n_inputs + e] = np.exp(-x * x)
            elif act == 7:
                values[n_inputs + e] = (1.0 - x * x) * np.exp(-x * x / 2.0)
            elif act == 8:
                values[n_inputs + e] = np.cos(5.0 * x) * np.exp(-x * x / 2.0)
            elif act == 9:
                values[n_inputs + e] = -x * np.exp(-x * x / 2.0)
            elif act == 10:
                values[n_inputs + e] = 1.0 / np.cosh(x)
            elif act == 11:
                values[n_inputs + e] = 1.0 if abs(x) < 1e-9 else np.sin(x) / x
            else:
                values[n_inputs + e] = x

        # argmax over 3 outputs -> target position
        o0 = values[output_indices[0]]
        o1 = values[output_indices[1]]
        o2 = values[output_indices[2]]
        amax = 0
        bv = o0
        if o1 > bv:
            bv = o1; amax = 1
        if o2 > bv:
            bv = o2; amax = 2
        if amax == 0:
            target = 1
        elif amax == 1:
            target = 0
        else:
            target = -1

        # control A: net chooses only flat-vs-in-trade; entry direction is random.
        if random_entry == 1 and pos == 0 and target != 0:
            target = rnd_dir[i]

        # ── session bound: force close (real, scored) ──
        force_close = False
        if pos != 0 and min_in_trade >= session_minutes:
            force_close = True

        # ── act on target / forced close ──
        if force_close or (target != pos):
            if pos != 0:
                # close at this event mid (spread already paid up front at entry)
                pnl = pos * (m - entry_mid) / pip - spread_entry
                if nt < n:
                    pnl_arr[nt] = pnl
                    amddp_arr[nt] = pnl - amddp_k * cum_dd
                    hold_arr[nt] = min_in_trade
                    dd_arr[nt] = cum_dd
                    entry_idx_arr[nt] = entry_event
                    exit_idx_arr[nt] = i
                    dir_arr[nt] = pos
                    nt += 1
                pos = 0
                cum_dd = 0.0
                min_in_trade = 0.0
            if force_close:
                # after a forced close we go flat this event; re-entry can happen next event
                pass
            else:
                # open the new target (if non-flat)
                if target != 0:
                    pos = target
                    entry_mid = m
                    spread_entry = (ask[i] - bid[i]) / pip
                    entry_event = i
                    cum_dd = 0.0
                    min_in_trade = 0.0

    # close any open trade at the end of the range (real, scored)
    if pos != 0 and end > start:
        last = end - 1
        m = mid[last]
        pnl = pos * (m - entry_mid) / pip - spread_entry
        if nt < n:
            pnl_arr[nt] = pnl
            amddp_arr[nt] = pnl - amddp_k * cum_dd
            hold_arr[nt] = min_in_trade
            dd_arr[nt] = cum_dd
            entry_idx_arr[nt] = entry_event
            exit_idx_arr[nt] = last
            dir_arr[nt] = pos
            nt += 1

    return (pnl_arr[:nt], amddp_arr[:nt], hold_arr[:nt], dd_arr[:nt],
            entry_idx_arr[:nt], exit_idx_arr[:nt], dir_arr[:nt])


# ──────────────────────────────────────────────────────────────────────────────
# Data loading + synthetic sine positive-control series.
# ──────────────────────────────────────────────────────────────────────────────
def load_real_series(path=CACHE):
    df = pd.read_parquet(path)
    ts = df["ts"].values.astype("datetime64[ns]")
    # dt in minutes between consecutive events (first event dt = 0)
    dt_ns = np.diff(ts).astype("timedelta64[s]").astype(np.float64)
    dt_min = np.empty(len(ts), dtype=np.float64)
    dt_min[0] = 0.0
    dt_min[1:] = dt_ns / 60.0
    return {
        "signed_age": df["signed_age"].values.astype(np.float64),
        "mid": df["mid"].values.astype(np.float64),
        "bid": df["bid"].values.astype(np.float64),
        "ask": df["ask"].values.astype(np.float64),
        "dt_min": dt_min,
        "ts": ts,
    }


def make_sine_series(n=20000, period=40, amp_pips=12.0, spread_pips=1.0, seed=0):
    """Synthetic positive-control: signed_age follows a sine; mid tracks the same sine so
    that 'long while sine rising, short while sine falling' is profitable net of a small
    spread. A capable harness MUST learn this. signed_age is the only signal the net sees.

    Construction: phase advances each event; signed_age = round(8*sin(phase)). mid moves with
    the *derivative* sign so trend-age direction predicts the next move. Spread small (1p).
    """
    rng = np.random.default_rng(seed)
    phase = np.cumsum(np.full(n, 2.0 * np.pi / period))
    s = np.sin(phase)
    signed_age = np.round(8.0 * s).astype(np.float64)
    # mid follows the sine in pips so that the direction of signed_age leads price up/down.
    base = 139.50
    mid = base + (amp_pips * PIP) * s
    half = (spread_pips * PIP) / 2.0
    bid = (mid - half).astype(np.float64)
    ask = (mid + half).astype(np.float64)
    dt_min = np.full(n, 5.0, dtype=np.float64)   # uniform 5-min cadence
    dt_min[0] = 0.0
    # random alternating directions for control-A use
    return {
        "signed_age": signed_age,
        "mid": mid.astype(np.float64),
        "bid": bid, "ask": ask,
        "dt_min": dt_min,
        "ts": None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3-way split + WF chunk indices.
# ──────────────────────────────────────────────────────────────────────────────
def make_splits(n, train=0.6, val=0.2):
    tr_end = int(n * train)
    val_end = int(n * (train + val))
    train_rng = (0, tr_end)
    val_rng = (tr_end, val_end)
    test_rng = (val_end, n)
    # 3 WF chunks inside train
    edges = np.linspace(0, tr_end, 4).astype(int)
    wf_chunks = [(int(edges[k]), int(edges[k + 1])) for k in range(3)]
    return train_rng, val_rng, test_rng, wf_chunks


# ──────────────────────────────────────────────────────────────────────────────
# Fitness + evaluation helpers.
# ──────────────────────────────────────────────────────────────────────────────
def _eval_range(net_arrays, data, rng, random_entry=False, rnd_dir=None):
    (n_inputs, n_outputs, n_eval, total_values,
     node_bias, node_response, node_act,
     conn_from, conn_to, conn_weight, output_indices) = net_arrays
    start, end = rng
    if rnd_dir is None:
        rnd_dir = np.ones(len(data["mid"]), dtype=np.int64)
    res = _simulate(
        data["signed_age"], data["mid"], data["bid"], data["ask"], data["dt_min"],
        start, end,
        PIP, AMDDP_K, AGE_SCALE, AMDDP_SCALE, FLAT_SENTINEL, CLIP, SESSION_MINUTES,
        1 if random_entry else 0, rnd_dir,
        n_inputs, n_eval, total_values,
        node_bias, node_response, node_act,
        conn_from, conn_to, conn_weight, output_indices,
    )
    return res


# Hard bidirectional gate: documented fix for the one-directional rot
# (JOURNEY 2026-04-12; the only clean causal winner was "perfectly bidirectional 9L/9S").
# In each WF chunk the minority direction must be >= MIN_DIR_RATIO of trades, else the
# genome is disqualified. This kills the "always short GBP_JPY" collapse that sank
# 10/16 islands in the first campaign (which had no directional term in fitness at all).
MIN_DIR_RATIO = 0.15


def fitness_from_wf(net_arrays, data, wf_chunks, exp, random_entry=False, rnd_dir=None):
    """min over WF chunks of (mean_amddp5 * n_trades^exp); hard min-trades + bidir gates."""
    worst = np.inf
    for rng in wf_chunks:
        pnl, amddp, hold, dd, ei, xi, di = _eval_range(
            net_arrays, data, rng, random_entry, rnd_dir)
        n = len(amddp)
        if n < MIN_TRADES_PER_CHUNK:
            return HARD_NEG
        # bidirectional enforcement (skip for control-A, whose direction is random by design)
        if not random_entry:
            n_long = int(np.sum(di > 0)); n_short = int(np.sum(di < 0))
            if min(n_long, n_short) < MIN_DIR_RATIO * n:
                return HARD_NEG
        score = float(np.mean(amddp)) * (n ** exp)
        if score < worst:
            worst = score
    return worst


def trade_stats(net_arrays, data, rng, random_entry=False, rnd_dir=None):
    pnl, amddp, hold, dd, ei, xi, di = _eval_range(
        net_arrays, data, rng, random_entry, rnd_dir)
    n = len(amddp)
    if n == 0:
        return dict(n=0, amddp_sum=0.0, pnl_sum=0.0, wr=0.0, mean_amddp=0.0,
                    amddp_per_day=0.0, pnl_per_day=0.0, trades_per_day=0.0,
                    mean_hold_min=0.0)
    start, end = rng
    if data["ts"] is not None:
        span_days = max(1e-9, (data["ts"][end - 1] - data["ts"][start]) /
                        np.timedelta64(1, "D"))
    else:
        span_days = max(1e-9, np.sum(data["dt_min"][start:end]) / (60.0 * 24.0))
    wr = 100.0 * float(np.mean(pnl > 0))
    return dict(
        n=n,
        amddp_sum=float(np.sum(amddp)),
        pnl_sum=float(np.sum(pnl)),
        wr=wr,
        mean_amddp=float(np.mean(amddp)),
        amddp_per_day=float(np.sum(amddp)) / span_days,
        pnl_per_day=float(np.sum(pnl)) / span_days,
        trades_per_day=n / span_days,
        mean_hold_min=float(np.mean(hold)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# NEAT run driver.
# ──────────────────────────────────────────────────────────────────────────────
def build_config(pop_size=None, config_ini=CONFIG_INI):
    cfg = neat.Config(
        CappedGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_ini,
    )
    # register custom activations into this config's activation set (per-instance)
    for name, fn in _CUSTOM_ACTS.items():
        if not cfg.genome_config.activation_defs.is_valid(name):
            cfg.genome_config.activation_defs.add(name, fn)
    if pop_size is not None:
        cfg.pop_size = pop_size
    return cfg


def run_neat(data, label, generations=30, pop_size=50, exp=0.5,
             random_entry=False, seed=0, verbose=True):
    """Single-island NEAT run. Returns (best_genome, config, history, rnd_dir)."""
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)

    cfg = build_config(pop_size=pop_size)
    train_rng, val_rng, test_rng, wf_chunks = make_splits(len(data["mid"]))

    # fixed random-entry direction stream (control A) — frozen for reproducibility
    rnd = np.random.default_rng(seed + 1234)
    rnd_dir = np.where(rnd.random(len(data["mid"])) > 0.5, 1, -1).astype(np.int64)

    history = []   # (gen, best_fit, best_val_amddp_per_day)

    def eval_pop(genomes, config):
        for gid, genome in genomes:
            try:
                na = extract_network(genome, config)
                genome.fitness = fitness_from_wf(
                    na, data, wf_chunks, exp, random_entry, rnd_dir)
            except Exception:
                genome.fitness = HARD_NEG

    pop = neat.Population(cfg)

    best_overall = None
    best_overall_fit = -np.inf

    for gen in range(generations):
        pop.run(eval_pop, 1)
        # current best of population
        best = None
        best_fit = -np.inf
        for genome in pop.population.values():
            if genome.fitness is not None and genome.fitness > best_fit:
                best_fit = genome.fitness
                best = genome
        if best is None:
            continue
        if best_fit > best_overall_fit:
            best_overall_fit = best_fit
            best_overall = best
        # validation amddp/day for the running best (ranking signal, NOT fitness)
        na = extract_network(best, cfg)
        vs = trade_stats(na, data, val_rng, random_entry, rnd_dir)
        history.append((gen, best_fit, vs["amddp_per_day"], vs["n"]))
        if verbose:
            print(f"  [{label}] gen {gen:3d}  best_fit={best_fit:10.2f}  "
                  f"val_amddp/d={vs['amddp_per_day']:8.2f}  val_trades={vs['n']:4d}",
                  flush=True)

    return best_overall, cfg, history, rnd_dir, (train_rng, val_rng, test_rng, wf_chunks)


# ──────────────────────────────────────────────────────────────────────────────
# Sanity entrypoint.
# ──────────────────────────────────────────────────────────────────────────────
def sanity():
    print("=" * 78)
    print("PHASE 1 SANITY RUN — sine positive-control + real GBP_JPY rev3")
    print("=" * 78)

    # ── (B) SINE POSITIVE CONTROL ──
    print("\n[B] SINE POSITIVE-CONTROL (harness must LEARN a planted signal)")
    sine = make_sine_series(n=20000, period=40, amp_pips=12.0, spread_pips=1.0, seed=0)
    t0 = time.time()
    sb, scfg, shist, srnd, ssplits = run_neat(
        sine, "sine", generations=30, pop_size=50, exp=0.5, seed=0)
    print(f"  (sine run: {time.time()-t0:.1f}s)")
    f0 = shist[0][1] if shist else float("nan")
    fL = shist[-1][1] if shist else float("nan")
    early = np.mean([h[1] for h in shist[:5]]) if len(shist) >= 5 else f0
    late = np.mean([h[1] for h in shist[-5:]]) if len(shist) >= 5 else fL
    climbed = late > early + 1e-6 and fL > 0
    na = extract_network(sb, scfg)
    s_val = trade_stats(na, sine, ssplits[1])
    s_train = trade_stats(na, sine, ssplits[0])
    print(f"  sine fitness: gen0={f0:.2f}  genLast={fL:.2f}  "
          f"mean(first5)={early:.2f}  mean(last5)={late:.2f}")
    print(f"  sine best train: amddp/d={s_train['amddp_per_day']:.2f} "
          f"pnl/d={s_train['pnl_per_day']:.2f} trades/d={s_train['trades_per_day']:.2f} "
          f"WR={s_train['wr']:.1f}%")
    print(f"  sine best val:   amddp/d={s_val['amddp_per_day']:.2f} "
          f"pnl/d={s_val['pnl_per_day']:.2f} WR={s_val['wr']:.1f}%")
    print(f"  >>> SINE CONTROL {'LEARNED (fitness climbs, positive)' if climbed else 'DID NOT LEARN — HARNESS SUSPECT'}")

    if not climbed:
        print("\n[STOP] Sine positive-control did not learn. The harness cannot learn a "
              "planted signal — do NOT trust real-series results. Investigate before training.")
        return False

    # ── (A) RANDOM-ENTRY CONTROL on real series (exit-only) ──
    print("\n[A] RANDOM-ENTRY CONTROL on real GBP_JPY rev3 (net manages EXITS only)")
    real = load_real_series()
    print(f"  real series: {len(real['mid'])} box events, "
          f"{(real['ts'][-1]-real['ts'][0])/np.timedelta64(1,'D'):.0f} days")
    t0 = time.time()
    rb, rcfg, rhist, rrnd, rsplits = run_neat(
        real, "rand-entry", generations=30, pop_size=50, exp=0.5,
        random_entry=True, seed=0)
    na = extract_network(rb, rcfg)
    ra_val = trade_stats(na, real, rsplits[1], random_entry=True, rnd_dir=rrnd)
    print(f"  (rand-entry run: {time.time()-t0:.1f}s)")
    print(f"  rand-entry best val: amddp/d={ra_val['amddp_per_day']:.2f} "
          f"pnl/d={ra_val['pnl_per_day']:.2f} trades/d={ra_val['trades_per_day']:.2f} "
          f"WR={ra_val['wr']:.1f}%")

    # ── FULL NET on real series ──
    print("\n[FULL] FULL NET on real GBP_JPY rev3 (net learns entry + exit)")
    t0 = time.time()
    fb, fcfg, fhist, frnd, fsplits = run_neat(
        real, "full", generations=30, pop_size=50, exp=0.5, seed=0)
    print(f"  (full run: {time.time()-t0:.1f}s)")
    na = extract_network(fb, fcfg)
    f_train = trade_stats(na, real, fsplits[0])
    f_val = trade_stats(na, real, fsplits[1])
    print(f"  full best train: amddp/d={f_train['amddp_per_day']:.2f} "
          f"pnl/d={f_train['pnl_per_day']:.2f} trades/d={f_train['trades_per_day']:.2f} "
          f"WR={f_train['wr']:.1f}% mean_hold={f_train['mean_hold_min']:.0f}min n={f_train['n']}")
    print(f"  full best val:   amddp/d={f_val['amddp_per_day']:.2f} "
          f"pnl/d={f_val['pnl_per_day']:.2f} trades/d={f_val['trades_per_day']:.2f} "
          f"WR={f_val['wr']:.1f}% n={f_val['n']}")

    # ── report card ──
    print("\n" + "=" * 78)
    print("SANITY REPORT CARD")
    print("=" * 78)
    print(f"  SINE control learned?        {'YES' if climbed else 'NO'}  "
          f"(fit {f0:.1f} -> {fL:.1f})")
    print(f"  Real run completed?          YES (no errors)")
    print(f"  FULL  train amddp/d={f_train['amddp_per_day']:8.2f}  val amddp/d={f_val['amddp_per_day']:8.2f}")
    print(f"  RAND  (exit-only) val amddp/d={ra_val['amddp_per_day']:8.2f}")
    print(f"  FULL  train trades/d={f_train['trades_per_day']:.2f}  WR={f_train['wr']:.1f}%")
    print(f"  >>> Full net beats random-entry control on val? "
          f"{'YES' if f_val['amddp_per_day'] > ra_val['amddp_per_day'] else 'NO (edge may be exit-deferral only)'}")
    print("  (NOTE: sanity is 1 island x 50 pop x 30 gens — a HYPOTHESIS, not an edge. "
          "Full 16x400x400 + OOS/MC/cross-pair required per PLAN.)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true", help="run the small local sanity probe")
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--pop", type=int, default=50)
    args = ap.parse_args()

    if args.sanity or len(sys.argv) == 1:
        sanity()
    else:
        print("Use --sanity for the local probe. Full Hetzner run is out of scope for this file.")


if __name__ == "__main__":
    main()
