#!/usr/bin/env python3
"""
Sine-wave trading proof-of-concept: can a pure-sin fixed network learn to trade?

Architecture: n_in → N_HID hidden (all sin) → 3 output (linear, argmax → BUY/SELL/FLAT)
Neuroevolution: CMA-ES on weights + biases ONLY. No topology, no activation mutation.

Inputs (2 market + 3 position state = 5 total):
  - price_norm:  tanh((close - sma) / amplitude)   — where in the wave
  - momentum:    tanh(Δclose / amplitude)           — which direction
  - upnl:        tanh(unrealized_pnl / 20)
  - mae:         tanh(max_adverse / 20)
  - mfe:         tanh(max_favorable / 20)

Synthetic data: pure sine wave with configurable period, amplitude, and optional noise.
Spread = 0 by default (prove the concept first, then add friction).

Usage:
    python3 sine_wave_test.py                          # quick proof
    python3 sine_wave_test.py --noise 0.1              # with noise
    python3 sine_wave_test.py --spread 1.5 --noise 0.1 # with spread + noise
    python3 sine_wave_test.py --n-hid 2                # minimal 2-node network
"""
import argparse
import time

import numpy as np
from numba import njit

import cma

# ── Architecture ──────────────────────────────────────────────────────
N_OUT = 3          # BUY, SELL, FLATTEN
N_POS_STATE = 3    # upnl, mae, mfe


@njit(cache=True, inline="always")
def sin_activate(z):
    return np.sin(z)


def n_params(n_in, n_hid):
    """W1 + b1 + W2 + b2, no activation genes."""
    return n_in * n_hid + n_hid + n_hid * N_OUT + N_OUT


# ── Generate synthetic sine-wave price series ─────────────────────────
def make_sine_data(n_bars, period, amplitude, base_price, noise_frac, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n_bars, dtype=np.float64)
    clean = base_price + amplitude * np.sin(2 * np.pi * t / period)
    noise = rng.normal(0, amplitude * noise_frac, n_bars) if noise_frac > 0 else 0.0
    return (clean + noise).astype(np.float64)


# ── Compute market features from price ────────────────────────────────
@njit(cache=True)
def compute_features(mid, sma_window, amplitude):
    """Returns (2, n_bars) array: [price_norm, momentum]."""
    n = len(mid)
    out = np.zeros((2, n))
    for i in range(sma_window, n):
        # SMA
        s = 0.0
        for j in range(i - sma_window, i):
            s += mid[j]
        sma = s / sma_window
        out[0, i] = np.tanh((mid[i] - sma) / amplitude)
        out[1, i] = np.tanh((mid[i] - mid[i - 1]) / amplitude)
    return out


# ── Bar-by-bar simulator ─────────────────────────────────────────────
@njit(cache=True)
def simulate(market, mid, pip, spread_pips, max_hold, weights, n_in, n_hid):
    """Walk bars, return (n_trades, total_pnl_pips, n_long, n_short, trades_pnl)."""
    n_market = n_in - N_POS_STATE
    w1_end = n_in * n_hid
    b1_end = w1_end + n_hid
    w2_end = b1_end + n_hid * N_OUT
    # b2 starts at w2_end, length N_OUT

    n_bars = len(mid)
    start = max(20, 1)
    pnls = np.zeros(n_bars)
    n_trades = 0
    n_long = 0
    n_short = 0

    position = 0
    entry_price = 0.0
    entry_bar = 0
    upnl_pips = 0.0
    mae_pips = 0.0
    mfe_pips = 0.0

    x = np.zeros(n_in)
    h = np.zeros(n_hid)

    for i in range(start, n_bars):
        # Update position metrics
        if position != 0:
            upnl_pips = (mid[i] - entry_price) / pip * position
            adverse = -upnl_pips
            if adverse > mae_pips:
                mae_pips = adverse
            if upnl_pips > mfe_pips:
                mfe_pips = upnl_pips
        else:
            upnl_pips = 0.0
            mae_pips = 0.0
            mfe_pips = 0.0

        # Build input: market features + position state
        for k in range(n_market):
            x[k] = market[k, i]
        x[n_market]     = np.tanh(upnl_pips / 20.0)
        x[n_market + 1] = np.tanh(mae_pips / 20.0)
        x[n_market + 2] = np.tanh(mfe_pips / 20.0)

        # Forward pass: all sin activations
        for j in range(n_hid):
            z = weights[w1_end + j]  # b1[j]
            for k in range(n_in):
                z += weights[j * n_in + k] * x[k]
            h[j] = sin_activate(z)

        out_buy  = weights[w2_end + 0]  # b2[0]
        out_sell = weights[w2_end + 1]
        out_flat = weights[w2_end + 2]
        for j in range(n_hid):
            out_buy  += weights[b1_end + 0 * n_hid + j] * h[j]
            out_sell += weights[b1_end + 1 * n_hid + j] * h[j]
            out_flat += weights[b1_end + 2 * n_hid + j] * h[j]

        if out_buy >= out_sell and out_buy >= out_flat:
            action = 1   # BUY
        elif out_sell >= out_buy and out_sell >= out_flat:
            action = 2   # SELL
        else:
            action = 0   # FLATTEN

        # Force-close on max_hold
        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid[i] - entry_price) / pip * position
            pnls[n_trades] = pnl
            if position > 0:
                n_long += 1
            else:
                n_short += 1
            n_trades += 1
            position = 0
            entry_price = 0.0
            mae_pips = 0.0
            mfe_pips = 0.0

        # Apply action
        if position == 0:
            if action == 1:
                position = 1
                entry_price = mid[i] + spread_pips * pip
                entry_bar = i
                mae_pips = spread_pips
                mfe_pips = 0.0
            elif action == 2:
                position = -1
                entry_price = mid[i] - spread_pips * pip
                entry_bar = i
                mae_pips = spread_pips
                mfe_pips = 0.0
        else:
            close_now = False
            new_pos = 0
            if action == 0:
                close_now = True
            elif position == 1 and action == 2:
                close_now = True
                new_pos = -1
            elif position == -1 and action == 1:
                close_now = True
                new_pos = 1
            if close_now:
                pnl = (mid[i] - entry_price) / pip * position
                pnls[n_trades] = pnl
                if position > 0:
                    n_long += 1
                else:
                    n_short += 1
                n_trades += 1
                position = new_pos
                if new_pos == 1:
                    entry_price = mid[i] + spread_pips * pip
                    entry_bar = i
                    mae_pips = spread_pips
                    mfe_pips = 0.0
                elif new_pos == -1:
                    entry_price = mid[i] - spread_pips * pip
                    entry_bar = i
                    mae_pips = spread_pips
                    mfe_pips = 0.0
                else:
                    entry_price = 0.0

    # Close any remaining position at end
    if position != 0:
        pnl = (mid[n_bars - 1] - entry_price) / pip * position
        pnls[n_trades] = pnl
        if position > 0:
            n_long += 1
        else:
            n_short += 1
        n_trades += 1

    total_pnl = 0.0
    for k in range(n_trades):
        total_pnl += pnls[k]
    return n_trades, total_pnl, n_long, n_short


# ── Fitness (simple: pips/bar, penalize one-sidedness) ────────────────
def fitness_neg(weights, market, mid, pip, spread_pips, max_hold, n_in, n_hid):
    nt, pnl, nl, ns = simulate(market, mid, pip, spread_pips, max_hold,
                                weights, n_in, n_hid)
    n_bars = len(mid)
    pps = pnl / n_bars if n_bars > 0 else 0.0

    if nt < 5:
        return 500.0 - pps

    dir_ratio = min(nl, ns) / nt
    asym_pen = (1.0 - 2.0 * dir_ratio) * 50.0
    return -(pps * 1000.0 - asym_pen)  # scale pps up for CMA-ES resolution


# ── Main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hid", type=int, default=4, help="Hidden nodes (default 4)")
    ap.add_argument("--period", type=int, default=100, help="Sine period in bars")
    ap.add_argument("--amplitude", type=float, default=0.005, help="Sine amplitude")
    ap.add_argument("--base-price", type=float, default=1.1000)
    ap.add_argument("--n-bars", type=int, default=10000)
    ap.add_argument("--noise", type=float, default=0.0, help="Noise as fraction of amplitude")
    ap.add_argument("--spread", type=float, default=0.0, help="Spread in pips")
    ap.add_argument("--pip", type=float, default=0.0001)
    ap.add_argument("--max-hold", type=int, default=200, help="Max hold in bars")
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--popsize", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n_hid = args.n_hid
    n_in = 2 + N_POS_STATE  # 2 market features + 3 position state
    np_total = n_params(n_in, n_hid)

    print(f"{'='*60}")
    print(f"  Sine-wave trading test")
    print(f"  Network: {n_in} → {n_hid} (sin) → {N_OUT}  ({np_total} params)")
    print(f"  Data: {args.n_bars} bars, period={args.period}, "
          f"amplitude={args.amplitude}, noise={args.noise}")
    print(f"  Spread: {args.spread} pips, pip={args.pip}")
    print(f"  Max hold: {args.max_hold} bars")
    print(f"  CMA-ES: popsize={args.popsize}, gens={args.gens}, seed={args.seed}")
    print(f"{'='*60}")

    # Generate data
    mid = make_sine_data(args.n_bars, args.period, args.amplitude,
                         args.base_price, args.noise, args.seed)
    market = compute_features(mid, sma_window=20, amplitude=args.amplitude)

    # Theoretical max pips: buy at every trough, sell at every peak
    n_cycles = args.n_bars / args.period
    max_pips_per_cycle = 2 * args.amplitude / args.pip  # trough-to-peak in pips
    theoretical_max = max_pips_per_cycle * n_cycles
    print(f"\n  Theoretical max (perfect trading): {theoretical_max:.0f} pips "
          f"({max_pips_per_cycle:.0f} pips/cycle × {n_cycles:.0f} cycles)")
    print(f"  Amplitude = {args.amplitude / args.pip:.0f} pips peak-to-zero\n")

    # JIT warmup
    print("JIT warming up...")
    _w = np.zeros(np_total)
    simulate(market[:, :100], mid[:100], args.pip, args.spread, 50, _w, n_in, n_hid)
    print("  done.\n")

    # CMA-ES
    t0 = time.time()
    opts = {
        "popsize": args.popsize,
        "seed": args.seed,
        "verbose": -9,
        "maxiter": args.gens,
        "tolx": 1e-9,
        "tolfun": 1e-4,
    }
    es = cma.CMAEvolutionStrategy(np.zeros(np_total), 0.5, opts)

    best_fit = 1e18
    best_vec = None
    gen = 0

    while not es.stop():
        candidates = es.ask()
        fits = [fitness_neg(c, market, mid, args.pip, args.spread,
                            args.max_hold, n_in, n_hid) for c in candidates]
        es.tell(candidates, fits)

        gmin = min(fits)
        if gmin < best_fit:
            best_fit = gmin
            best_vec = np.array(candidates[fits.index(gmin)])

        if gen % 10 == 0:
            # Evaluate current best
            nt, pnl, nl, ns = simulate(market, mid, args.pip, args.spread,
                                       args.max_hold, best_vec, n_in, n_hid)
            dr = min(nl, ns) / nt if nt > 0 else 0
            print(f"  Gen {gen:>3}: fit={best_fit:>10.2f} | "
                  f"{pnl:>+8.1f}p ({nt} trades, L/S={nl}/{ns}, "
                  f"dr={dr:.2f}) | σ={es.sigma:.4f} | "
                  f"{time.time()-t0:.1f}s")
        gen += 1

    # Final eval
    elapsed = time.time() - t0
    nt, pnl, nl, ns = simulate(market, mid, args.pip, args.spread,
                                args.max_hold, best_vec, n_in, n_hid)
    dr = min(nl, ns) / nt if nt > 0 else 0
    pips_per_cycle = pnl / n_cycles if n_cycles > 0 else 0
    efficiency = pnl / theoretical_max * 100 if theoretical_max > 0 else 0

    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"  Total P&L:      {pnl:>+.1f} pips")
    print(f"  Trades:          {nt} (L={nl}, S={ns}, dir_ratio={dr:.2f})")
    print(f"  Pips/cycle:      {pips_per_cycle:>+.1f} / {max_pips_per_cycle:.0f} theoretical")
    print(f"  Efficiency:      {efficiency:.1f}% of theoretical max")
    print(f"  Elapsed:         {elapsed:.1f}s ({gen} gens)")
    print(f"{'='*60}")

    # Dump the trade log for first 3 cycles so we can see behavior
    print(f"\n  Trade log (first {3*args.period} bars):")
    _dump_trades(market, mid, args.pip, args.spread, args.max_hold,
                 best_vec, n_in, n_hid, 3 * args.period)


@njit(cache=True)
def _get_trade_log(market, mid, pip, spread_pips, max_hold, weights, n_in, n_hid,
                   max_bar):
    """Like simulate but returns arrays of trade details."""
    n_market = n_in - N_POS_STATE
    w1_end = n_in * n_hid
    b1_end = w1_end + n_hid
    w2_end = b1_end + n_hid * N_OUT

    cap = 500
    t_open = np.zeros(cap, dtype=np.int64)
    t_close = np.zeros(cap, dtype=np.int64)
    t_dir = np.zeros(cap, dtype=np.int64)
    t_pnl = np.zeros(cap)
    n_trades = 0

    position = 0
    entry_price = 0.0
    entry_bar = 0
    upnl_pips = 0.0
    mae_pips = 0.0
    mfe_pips = 0.0
    x = np.zeros(n_in)
    h = np.zeros(n_hid)

    end = min(max_bar, len(mid))
    for i in range(20, end):
        if position != 0:
            upnl_pips = (mid[i] - entry_price) / pip * position
            adverse = -upnl_pips
            if adverse > mae_pips:
                mae_pips = adverse
            if upnl_pips > mfe_pips:
                mfe_pips = upnl_pips
        else:
            upnl_pips = 0.0; mae_pips = 0.0; mfe_pips = 0.0

        for k in range(n_market):
            x[k] = market[k, i]
        x[n_market] = np.tanh(upnl_pips / 20.0)
        x[n_market + 1] = np.tanh(mae_pips / 20.0)
        x[n_market + 2] = np.tanh(mfe_pips / 20.0)

        for j in range(n_hid):
            z = weights[w1_end + j]
            for k in range(n_in):
                z += weights[j * n_in + k] * x[k]
            h[j] = np.sin(z)

        out_buy  = weights[w2_end + 0]
        out_sell = weights[w2_end + 1]
        out_flat = weights[w2_end + 2]
        for j in range(n_hid):
            out_buy  += weights[b1_end + 0 * n_hid + j] * h[j]
            out_sell += weights[b1_end + 1 * n_hid + j] * h[j]
            out_flat += weights[b1_end + 2 * n_hid + j] * h[j]

        if out_buy >= out_sell and out_buy >= out_flat:
            action = 1
        elif out_sell >= out_buy and out_sell >= out_flat:
            action = 2
        else:
            action = 0

        if position != 0 and (i - entry_bar) >= max_hold:
            pnl = (mid[i] - entry_price) / pip * position
            if n_trades < cap:
                t_open[n_trades] = entry_bar
                t_close[n_trades] = i
                t_dir[n_trades] = position
                t_pnl[n_trades] = pnl
                n_trades += 1
            position = 0; entry_price = 0.0; mae_pips = 0.0; mfe_pips = 0.0

        if position == 0:
            if action == 1:
                position = 1; entry_price = mid[i] + spread_pips * pip
                entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0
            elif action == 2:
                position = -1; entry_price = mid[i] - spread_pips * pip
                entry_bar = i; mae_pips = spread_pips; mfe_pips = 0.0
        else:
            close_now = False; new_pos = 0
            if action == 0: close_now = True
            elif position == 1 and action == 2: close_now = True; new_pos = -1
            elif position == -1 and action == 1: close_now = True; new_pos = 1
            if close_now:
                pnl = (mid[i] - entry_price) / pip * position
                if n_trades < cap:
                    t_open[n_trades] = entry_bar
                    t_close[n_trades] = i
                    t_dir[n_trades] = position
                    t_pnl[n_trades] = pnl
                    n_trades += 1
                position = new_pos
                if new_pos == 1:
                    entry_price = mid[i] + spread_pips * pip; entry_bar = i
                    mae_pips = spread_pips; mfe_pips = 0.0
                elif new_pos == -1:
                    entry_price = mid[i] - spread_pips * pip; entry_bar = i
                    mae_pips = spread_pips; mfe_pips = 0.0
                else:
                    entry_price = 0.0

    return t_open[:n_trades], t_close[:n_trades], t_dir[:n_trades], t_pnl[:n_trades], n_trades


def _dump_trades(market, mid, pip, spread, max_hold, weights, n_in, n_hid, max_bar):
    opens, closes, dirs, pnls, nt = _get_trade_log(
        market, mid, pip, spread, max_hold, weights, n_in, n_hid, max_bar)
    for k in range(nt):
        d = "LONG " if dirs[k] > 0 else "SHORT"
        print(f"    bar {opens[k]:>5} → {closes[k]:>5} ({closes[k]-opens[k]:>3} bars)  "
              f"{d}  {pnls[k]:>+7.1f}p")


if __name__ == "__main__":
    main()
