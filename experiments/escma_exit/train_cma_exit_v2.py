"""
train_cma_exit_v2.py — SIMPLIFIED ESCMA exit-learner (2026-06-12).

A deliberate minimal redesign of train_cma_exit.py:

  Inputs (6):
    5 × scaled momentum   x_k = tanh( mn_k / (k_const · MAD_pooled) )
                          MAD_pooled = ONE shared pooled MAD over the 5 mn_*
                          series, computed IS-only and FROZEN (R5-clean).
                          Sign-exact (odd squash, no centering), bounded (-1,1),
                          robust (median-deviation scale).
    1 × position          x_5 = tanh( (u - λ·cum_dd) / 20 )  — the LIVE AMDDP5
                          REWARD-to-date itself (penalized unrealized P/L), the
                          exact quantity the net is optimized on. NOT raw u.

    DROPPED vs v1: the 8 RFF context features, raw OHLC, raw mom_*, hold-frac,
    acc-dd. No pre-entry window. The net sees only the current bar + its P/L.

  Network: 6 → 3 → 1   (W1 18 + b1 3 + W2 3 + b2 1 = 25 params).
           Output tanh → exit if > exit_threshold.

  Activation search — the hidden layer's nonlinearity is searched over a BANK:
      0 sin   1 cos   2 tanh   3 gaussian   4 morlet   5 ricker(mexican-hat)
    TWO modes, both run per seed:
      • SWEEP (outer):   one CMA run per activation; all 3 nodes use it.
                         → clean leaderboard "which single activation wins".
      • EVOLVED (inner): genome carries per-node softmax logits over the bank
                         (25 + 3·B params); CMA blends activations per node.
                         → "does a mixed-activation layer beat any single one".

  Reward: AMDDP5 (λ=5%). Same rails (disaster SL 100p / runaway TP 200p / time
  cap) and the same naive baselines (first-neg-tick / fixed-TP / hold) as v1,
  so results are directly comparable to the 225-param run.

  Entry events: meta3 (default the z2_k2 sighted shock-fade) — apples-to-apples.

Run ONE seed per process (load batch once, sweep whole bank + evolved). Drive
seeds 42/7/123 from a STRICTLY SEQUENTIAL shell script (never parallel — the
parallel-CMA freeze rule still holds).
"""
from __future__ import annotations
import argparse, gc, time
from pathlib import Path

import numpy as np
import cma
from numba import njit, prange

import train_cma_exit as v1   # reuse loader, baselines, constants

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Architecture ──────────────────────────────────────────────────────────
N_MN   = 5
N_POS  = 1
N_IN   = N_MN + N_POS          # 6
N_HID  = 3
N_OUT  = 1
N_NET  = N_IN * N_HID + N_HID + N_HID * N_OUT + N_OUT   # 18+3+3+1 = 25

# param layout (net): W1 [0:18] (row j → j*6+k), b1 [18:21], W2 [21:24], b2 [24]
B1_OFF = N_IN * N_HID          # 18
W2_OFF = B1_OFF + N_HID        # 21
B2_OFF = W2_OFF + N_HID        # 24
LOGIT_OFF = N_NET             # evolved logits start here (mode=evolved)

# mn_* live-feature column indices inside v1's 15-vector
#   [0 open,1 high,2 low,3 close, 4-8 mom_*, 9-13 mn_*, 14 spread]
MN_COL0 = 9

BANK_NAMES = ["sin", "cos", "tanh", "gauss", "morlet", "ricker"]
N_BANK = len(BANK_NAMES)

CAUSE_NET, CAUSE_SL, CAUSE_TP, CAUSE_TIME = 0, 1, 2, 3


@njit(cache=True, inline="always")
def _act(z, code):
    if code == 0:
        return np.sin(z)
    elif code == 1:
        return np.cos(z)
    elif code == 2:
        return np.tanh(z)
    elif code == 3:
        return np.exp(-z * z)
    elif code == 4:                       # real Morlet (ω0=5)
        return np.cos(5.0 * z) * np.exp(-0.5 * z * z)
    else:                                 # ricker / mexican hat
        return (1.0 - z * z) * np.exp(-0.5 * z * z)


@njit(cache=True, inline="always")
def _walk(x, h, live_feats, mid, bid, ask, direction, e_px, pip,
          sl_cap, tp_cap, t_max, weights, exit_threshold,
          bar_minutes, scale_div, act_code, lambda_k):
    """One trade exit walk with CALLER-PROVIDED x,h buffers (no per-call
    allocation). act_code>=0 → fixed activation; act_code<0 → evolved per-node
    softmax blend (logits in weights[LOGIT_OFF:])."""
    T = live_feats.shape[0]
    if t_max > T:
        t_max = T
    cum_dd = 0.0
    pnl_pips = 0.0
    exit_cause = CAUSE_TIME
    exit_t = t_max - 1

    for t in range(t_max):
        u = direction * (mid[t] - e_px) / pip
        if u < 0.0:
            cum_dd += -u * bar_minutes

        if u < -sl_cap:
            ex = bid[t] if direction == 1 else ask[t]
            return direction * (ex - e_px) / pip, cum_dd, t + 1, CAUSE_SL, t
        if u > tp_cap:
            ex = bid[t] if direction == 1 else ask[t]
            return direction * (ex - e_px) / pip, cum_dd, t + 1, CAUSE_TP, t

        # 5 scaled mn (sign-exact, bounded) + 1 position feature.
        # Position feature = the LIVE AMDDP5 REWARD-to-date (u - λ·cum_dd) — the
        # exact quantity the net is optimized on, NOT raw unrealized P/L. cum_dd
        # already includes this bar. tanh only bounds it for the sin net
        # (sign-preserving, same convention as the mn features).
        for k in range(N_MN):
            x[k] = np.tanh(live_feats[t, MN_COL0 + k] / scale_div)
        x[N_MN] = np.tanh((u - lambda_k * cum_dd) / 20.0)

        # hidden layer
        for j in range(N_HID):
            z = weights[B1_OFF + j]
            base = j * N_IN
            for k in range(N_IN):
                z += weights[base + k] * x[k]
            if act_code >= 0:
                h[j] = _act(z, act_code)
            else:
                # softmax blend over the bank
                lo = LOGIT_OFF + j * N_BANK
                mx = weights[lo]
                for b in range(1, N_BANK):
                    if weights[lo + b] > mx:
                        mx = weights[lo + b]
                ssum = 0.0
                acc = 0.0
                for b in range(N_BANK):
                    w = np.exp(weights[lo + b] - mx)
                    ssum += w
                    acc += w * _act(z, b)
                h[j] = acc / ssum

        out_z = weights[B2_OFF]
        for j in range(N_HID):
            out_z += weights[W2_OFF + j] * h[j]
        if np.tanh(out_z) > exit_threshold:
            ex = bid[t] if direction == 1 else ask[t]
            return direction * (ex - e_px) / pip, cum_dd, t + 1, CAUSE_NET, t

    if t_max > 0:
        ex = bid[t_max - 1] if direction == 1 else ask[t_max - 1]
        pnl_pips = direction * (ex - e_px) / pip
    return pnl_pips, cum_dd, t_max, exit_cause, exit_t


@njit(cache=True)
def simulate_one_v2(live_feats, mid, bid, ask, direction, e_px, pip,
                    sl_cap, tp_cap, t_max, weights, exit_threshold,
                    bar_minutes, scale_div, act_code, lambda_k):
    """Thin wrapper (allocates buffers) — used by the OOS eval path."""
    x = np.zeros(N_IN, dtype=np.float64)
    h = np.zeros(N_HID, dtype=np.float64)
    return _walk(x, h, live_feats, mid, bid, ask, direction, e_px, pip,
                 sl_cap, tp_cap, t_max, weights, exit_threshold,
                 bar_minutes, scale_div, act_code, lambda_k)


@njit(cache=True, parallel=True)
def eval_population(pop, live_feats, mid, bid, ask, direction, e_px, pip,
                   sl_cap, tp_cap, t_actual, exit_threshold, bar_minutes,
                   scale_div, act_code, lambda_k, asym):
    """Evaluate a whole CMA population in ONE parallel call.
    prange over candidates → each core takes a candidate, walks all N samples.
    asym=0: training reward = AMDDP5 (p − λ·dd).  asym=1: ASYMMETRIC reward
    = max(0, AMDDP5) — losing trades are FREE, removing the bail-immediately
    incentive so the net can explore holding for the winners. (OOS is still
    scored on TRUE pnl/AMDDP5 elsewhere — this only shapes the search.)
    """
    P = pop.shape[0]
    N = live_feats.shape[0]
    out = np.empty(P, dtype=np.float64)
    for c in prange(P):
        x = np.zeros(N_IN, dtype=np.float64)
        h = np.zeros(N_HID, dtype=np.float64)
        w = pop[c]
        s = 0.0
        for i in range(N):
            p, dd, hh, cc, et = _walk(
                x, h, live_feats[i], mid[i], bid[i], ask[i],
                int(direction[i]), e_px[i], pip[i], sl_cap[i], tp_cap[i],
                int(t_actual[i]), w, exit_threshold, bar_minutes,
                scale_div, act_code, lambda_k)
            r = p - lambda_k * dd
            if asym == 1 and r < 0.0:
                r = 0.0
            s += r
        out[c] = s
    return out


def simulate_batch_v2(batch, weights, lambda_pct, exit_threshold,
                      bar_minutes, scale_div, act_code):
    """Single-genome eval with full diagnostics (OOS reporting path)."""
    N = batch.live_feats.shape[0]
    pnl = np.empty(N); cum_dd = np.empty(N); hold = np.empty(N, np.int64)
    cause = np.empty(N, np.int64)
    lambda_k = lambda_pct / 100.0
    w64 = weights.astype(np.float64, copy=False)
    for i in range(N):
        p, dd, hh, c, _ = simulate_one_v2(
            batch.live_feats[i], batch.mid[i], batch.bid[i], batch.ask[i],
            int(batch.direction[i]), float(batch.e_px[i]), float(batch.pip[i]),
            float(batch.sl_cap[i]), float(batch.tp_cap[i]),
            int(batch.t_actual[i]), w64, float(exit_threshold),
            float(bar_minutes), float(scale_div), int(act_code), float(lambda_k))
        pnl[i] = p; cum_dd[i] = dd; hold[i] = hh; cause[i] = c
    amddp = pnl - lambda_k * cum_dd
    return {"sum_amddp": float(amddp.sum()), "sum_pnl": float(pnl.sum()),
            "mean_hold": float(hold.mean()), "wr": float((pnl > 0).mean()),
            "n": N, "cause": cause}


def compute_is_frozen_mad(pair, data_dir, batch, k_const):
    """Shared pooled MAD over the 5 mn_* series, IS-only (bar_idx < OOS start).
    Returns scale_div = k_const · MAD_pooled (frozen scalar)."""
    store = v1._load_features(pair, data_dir)
    # IS/OOS time boundary = first event bar of any OOS sample
    # (meta3 split is temporal 70/30). Use t_actual>0 OOS sample mid start —
    # simplest robust proxy: cut at the 70th percentile of all event indices.
    # We reconstruct event indices from the loaded meta3 via batch ordering is
    # not available here, so derive from split mask + a conservative bound.
    mn_cols = ["mn_S5", "mn_M1", "mn_5m", "mn_15m", "mn_1h"]
    n_src = store["close"].shape[0]
    # IS region = first 70% of the source series (matches the 70/30 temporal split)
    is_end = int(n_src * 0.70)
    pooled = []
    for c in mn_cols:
        a = store[c][:is_end]
        a = a[np.isfinite(a)]
        pooled.append(a)
    pooled = np.concatenate(pooled)
    med = float(np.median(pooled))
    mad = float(np.median(np.abs(pooled - med)))
    del pooled, store
    gc.collect()
    return k_const * mad, med, mad


def _batch_arrays(b, lambda_pct):
    """Pull the raw, numba-ready arrays out of a SampleBatch once (so the hot
    eval_population call passes contiguous arrays, not Python attribute lookups)."""
    return (b.live_feats, b.mid, b.bid, b.ask,
            b.direction.astype(np.int64), b.e_px.astype(np.float64),
            b.pip.astype(np.float64), b.sl_cap.astype(np.float64),
            b.tp_cap.astype(np.float64), b.t_actual.astype(np.int64))


def run_cma(is_batch, n_params, act_code, seed, gens, popsize, sigma0,
            bound, lambda_pct, exit_threshold, bar_minutes, scale_div, label,
            x0_init=None, asym=0):
    arrs = _batch_arrays(is_batch, lambda_pct)
    lambda_k = lambda_pct / 100.0
    x0 = np.zeros(n_params) if x0_init is None else np.asarray(x0_init, dtype=np.float64)
    opts = {"bounds": [-bound, bound], "maxiter": gens, "seed": seed,
            "verbose": -9, "tolfun": 1e-9, "tolx": 1e-11}
    if popsize > 0:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    t0 = time.time()
    best_f = 1e18; best_x = x0
    gen = 0
    while not es.stop():
        sols = es.ask()
        pop = np.asarray(sols, dtype=np.float64)            # (P, n_params)
        sums = eval_population(pop, *arrs, float(exit_threshold),
                               float(bar_minutes), float(scale_div),
                               int(act_code), float(lambda_k), int(asym))
        es.tell(sols, (-sums).tolist())            # CMA minimises → feed -AMDDP5
        jbest = int(np.argmax(sums))               # best candidate = max AMDDP5
        if -sums[jbest] < best_f:
            best_f = -sums[jbest]; best_x = pop[jbest].copy()
        gen += 1
        if gen % 20 == 0:
            print(f"      [{label}] gen {gen:3d}  IS_amddp5={-best_f:+8.1f}p  "
                  f"σ={es.sigma:.3f}  t={time.time()-t0:.0f}s", flush=True)
    return best_x, -best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="USD_JPY")
    ap.add_argument("--data-dir", default=str(SCRIPT_DIR))
    ap.add_argument("--meta3-name", default="meta3_USD_JPY_wav_z2_k2.parquet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--popsize", type=int, default=0)
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--bound", type=float, default=3.0)
    ap.add_argument("--lambda-pct", type=float, default=5.0)
    ap.add_argument("--exit-threshold", type=float, default=0.5)
    ap.add_argument("--bar-minutes", type=float, default=1.0 / 12.0)
    ap.add_argument("--k-const", type=float, default=4.45,
                    help="divisor = k_const · MAD_pooled  (4.45 ≈ 3 robust-σ)")
    ap.add_argument("--t-max-alloc", type=int, default=1440)
    ap.add_argument("--mode", choices=["sweep", "evolved", "both"], default="both")
    ap.add_argument("--save-weights", action="store_true",
                    help="save best weights per config to results/v2w_<pair>_<label>_s<seed>.npy")
    ap.add_argument("--init-dir", default=None,
                    help="warm-start: dir holding v2w_<init-pair>_*.npy from a prior run; "
                         "each config seeds CMA x0 from its matching saved weights")
    ap.add_argument("--init-pair", default="SINE",
                    help="source pair tag for warm-start weight files")
    ap.add_argument("--reward-mode", choices=["amddp5", "asym"], default="amddp5",
                    help="asym = max(0,AMDDP5) training reward (losers free) to kill "
                         "bail-immediately; OOS still scored on true pnl/AMDDP5")
    args = ap.parse_args()
    RESULTS_DIR = SCRIPT_DIR / "results"; RESULTS_DIR.mkdir(exist_ok=True)

    def _safe(label):
        return label.replace(":", "_").replace("[", "_").replace("]", "").replace("+", "-")

    def _save_w(label, w):
        if args.save_weights:
            np.save(RESULTS_DIR / f"v2w_{args.pair}_{_safe(label)}_s{args.seed}.npy", w)

    def _load_init(label, n_params):
        if not args.init_dir:
            return None
        p = Path(args.init_dir) / f"v2w_{args.init_pair}_{_safe(label)}_s{args.seed}.npy"
        if not p.exists():
            print(f"      [warm-start] no init for {label} ({p.name}) — cold start", flush=True)
            return None
        w = np.load(p)
        if w.shape[0] != n_params:    # mismatch (e.g. logits) → pad/truncate to fit
            ww = np.zeros(n_params); k = min(n_params, w.shape[0]); ww[:k] = w[:k]; w = ww
        print(f"      [warm-start] init {label} from {p.name}", flush=True)
        return w

    data_dir = Path(args.data_dir)
    print(f"\n{'='*72}\nSIMPLIFIED ESCMA v2  pair={args.pair}  seed={args.seed}  "
          f"gens={args.gens}  mode={args.mode}\nmeta3={args.meta3_name}\n{'='*72}", flush=True)

    # ── Load batch once (reuse v1 loader; rff ignored) ──
    W_proj, b_proj = v1.make_rff(1337)
    print("[load] building batch ...", flush=True)
    batch = v1.load_real(args.pair, data_dir, W_proj, b_proj,
                         t_max_alloc=args.t_max_alloc, meta3_name=args.meta3_name)

    # IS-frozen pooled MAD scale
    scale_div, med, mad = compute_is_frozen_mad(args.pair, data_dir, batch, args.k_const)
    print(f"[scale] IS-frozen pooled MAD={mad:.4f} (median={med:+.4f})  "
          f"=> divisor k·MAD = {scale_div:.4f}", flush=True)
    # free the 1.4GB feature cache before CMA
    v1._FEATURES_CACHE.clear(); gc.collect()

    is_mask = batch.split == "IS"
    oos_mask = batch.split == "OOS"
    is_b = batch.filter(is_mask)
    oos_b = batch.filter(oos_mask)
    print(f"[split] IS n={int(is_mask.sum())}  OOS n={int(oos_mask.sum())}", flush=True)

    popsize = args.popsize
    asym = 1 if args.reward_mode == "asym" else 0
    common = dict(seed=args.seed, gens=args.gens, popsize=popsize,
                  sigma0=args.sigma0, bound=args.bound, lambda_pct=args.lambda_pct,
                  exit_threshold=args.exit_threshold, bar_minutes=args.bar_minutes,
                  scale_div=scale_div, asym=asym)
    print(f"[reward] training reward = {args.reward_mode}"
          + ("  (max(0,AMDDP5) — losers free)" if asym else ""), flush=True)

    # naive baseline on OOS (the bar to beat)
    base = v1.run_baseline(oos_b, "firstneg", 5.0, args.bar_minutes)
    print(f"\n[baseline] OOS first-negative-tick  amddp5={base['sum_amddp']:+8.1f}p  "
          f"pnl={base['sum_pnl']:+7.1f}p  WR={base['wr']:.1%}", flush=True)

    results = []

    if args.mode in ("sweep", "both"):
        print(f"\n{'─'*72}\nSWEEP — one CMA per activation (25 params, homogeneous layer)\n{'─'*72}", flush=True)
        for code, name in enumerate(BANK_NAMES):
            print(f"\n  >>> activation = {name}", flush=True)
            bx, isf = run_cma(is_b, N_NET, code, label=f"sweep:{name}",
                              x0_init=_load_init(f"sweep:{name}", N_NET), **common)
            _save_w(f"sweep:{name}", bx)
            r = simulate_batch_v2(oos_b, bx, 5.0, args.exit_threshold,
                                  args.bar_minutes, scale_div, code)
            print(f"      OOS amddp5={r['sum_amddp']:+8.1f}p  pnl={r['sum_pnl']:+7.1f}p  "
                  f"WR={r['wr']:.1%}  hold={r['mean_hold']:.0f}b  (IS={isf:+.0f})", flush=True)
            results.append((f"sweep:{name}", isf, r))

    if args.mode in ("evolved", "both"):
        print(f"\n{'─'*72}\nEVOLVED — per-node activation logits ({N_NET}+{3*N_BANK} params)\n{'─'*72}", flush=True)
        n_par = N_NET + N_HID * N_BANK
        bx, isf = run_cma(is_b, n_par, -1, label="evolved",
                          x0_init=_load_init("evolved", n_par), **common)
        _save_w("evolved", bx)
        r = simulate_batch_v2(oos_b, bx, 5.0, args.exit_threshold,
                              args.bar_minutes, scale_div, -1)
        # report dominant activation per node
        doms = []
        for j in range(N_HID):
            lo = LOGIT_OFF + j * N_BANK
            doms.append(BANK_NAMES[int(np.argmax(bx[lo:lo + N_BANK]))])
        print(f"      OOS amddp5={r['sum_amddp']:+8.1f}p  pnl={r['sum_pnl']:+7.1f}p  "
              f"WR={r['wr']:.1%}  hold={r['mean_hold']:.0f}b  (IS={isf:+.0f})  "
              f"nodes={doms}", flush=True)
        results.append((f"evolved[{'+'.join(doms)}]", isf, r))

    # leaderboard
    print(f"\n{'='*72}\nLEADERBOARD (seed={args.seed})  — OOS AMDDP5, naive bar = {base['sum_amddp']:+.1f}\n{'='*72}", flush=True)
    for name, isf, r in sorted(results, key=lambda z: -z[2]["sum_amddp"]):
        beat = "🟢 beats naive" if r["sum_amddp"] > base["sum_amddp"] else "🔴 loses"
        print(f"  {name:22s}  OOS={r['sum_amddp']:+8.1f}  pnl={r['sum_pnl']:+7.1f}  "
              f"WR={r['wr']:.1%}  hold={r['mean_hold']:.0f}b  IS={isf:+8.1f}  {beat}", flush=True)
    print("="*72, flush=True)


if __name__ == "__main__":
    main()
