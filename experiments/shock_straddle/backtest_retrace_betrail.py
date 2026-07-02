#!/usr/bin/env python3
"""
Retrace BE/Trail Exit Sweep — compare break-even + trailing stop vs fixed HORIZON
==================================================================================
Fixes the best validated config (thr=2.5 peak=44 sd=3 tp=20) and sweeps exit strategy:
  - Baseline: HORIZON=600 timeout only
  - BE only:  move SL to break-even once profit >= be_thresh pips
  - Trail:    once profit >= trail_start, trail at trail_gap pips behind peak
  - BE+Trail: both active

Reports OOS p/d, WR, MFE/MAE stats, and WF (3 chunks, 4 pairs = 12 sub-tests).
"""
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from numba import njit
import warnings; warnings.filterwarnings("ignore")

PROJECT  = Path(__file__).resolve().parents[3]
S5_DIR   = PROJECT / "data" / "s5_ba"
RESULTS  = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PAIRS    = ["GBP_JPY", "USD_JPY", "EUR_JPY", "AUD_JPY"]
PIP      = {"GBP_JPY": 0.01, "USD_JPY": 0.01, "EUR_JPY": 0.01, "AUD_JPY": 0.01}

# Fixed best config from original backtest
THR        = 2.5
PEAK_BARS  = 44
STOP_PIPS  = 3.0
TP_PIPS    = 20.0
HORIZON    = 600
Z_WINDOW   = 6
MAD_WIN    = 2048
IS_FRAC    = 0.70
WF_CHUNKS  = 3

# Exit sweep parameters
BE_THRESHS   = [0.0, 3.0, 5.0, 7.0]       # 0 = disabled
TRAIL_STARTS = [0.0, 5.0, 8.0, 10.0, 12.0] # 0 = disabled
TRAIL_GAPS   = [3.0, 5.0, 7.0]


# ── Shock detection ─────────────────────────────────────────────────────────────

def compute_shock_z(close: np.ndarray, pip: float, w: int = 6,
                    mad_win: int = 2048) -> tuple:
    n   = len(close)
    vel = np.empty(n, dtype=np.float64)
    vel[:w] = 0.0
    vel[w:] = (close[w:] - close[:n - w]) / pip
    vel_s = pd.Series(vel)
    rm    = vel_s.rolling(mad_win, min_periods=50, center=False).median()
    ad    = (vel_s - rm).abs()
    rmad  = ad.rolling(mad_win, min_periods=50, center=False).median()
    z     = ((vel_s - rm) / (1.4826 * rmad.clip(lower=1e-6))).fillna(0).values
    flag  = (np.abs(z) > THR).astype(np.int8)
    return z.astype(np.float64), vel.astype(np.float64), flag


# ── Numba simulator with BE + trailing stop ──────────────────────────────────────

@njit
def sim_retrace_betrail(bid, ask, close, shock_flag, vel, pip,
                         peak_bars, stop_pips, tp_pips, horizon,
                         be_thresh, trail_start, trail_gap):
    """
    Returns (pnl, mfe, mae, reason) arrays.
    reason: 1=tp  2=be_stop  3=trail_stop  4=timeout
    be_thresh=0  → BE disabled
    trail_start=0 → trailing disabled
    """
    n        = len(close)
    pb_int   = int(peak_bars)
    max_ev   = n // 10
    pnl_out    = np.zeros(max_ev, dtype=np.float64)
    mfe_out    = np.zeros(max_ev, dtype=np.float64)
    mae_out    = np.zeros(max_ev, dtype=np.float64)
    reason_out = np.zeros(max_ev, dtype=np.int8)
    ev_count = 0
    cooldown = 0

    for t in range(Z_WINDOW, n - pb_int - int(horizon) - 2):
        if cooldown > 0:
            cooldown -= 1
            continue
        if shock_flag[t] != 1:
            continue

        d = np.int8(1) if vel[t] > 0 else np.int8(-1)

        # Find peak in observation window
        peak_ask = ask[t]
        peak_bid = bid[t]
        for k in range(1, pb_int + 1):
            j = t + k
            if ask[j] > peak_ask: peak_ask = ask[j]
            if bid[j] < peak_bid: peak_bid = bid[j]

        sp          = (ask[t] - bid[t]) / pip
        watch_start = t + pb_int + 1
        watch_end   = t + pb_int + int(horizon)
        if watch_start >= n or watch_end >= n:
            continue

        # Stop entry
        if d == 1:
            entry = peak_ask - stop_pips * pip
        else:
            entry = peak_bid + stop_pips * pip

        pnl        = 0.0
        reason     = np.int8(0)
        mfe        = 0.0
        mae        = 0.0
        peak_pft   = 0.0
        fld        = 0
        fill_price = entry
        tp_level   = 0.0
        be_active  = False

        for j in range(watch_start, min(watch_end + 1, n - 1)):
            lo = bid[j]; hi = ask[j]; mid = close[j]

            # Fill check
            if fld == 0:
                if d == 1 and lo <= entry:
                    fld = 1; fill_price = entry
                    tp_level = fill_price - tp_pips * pip
                elif d == -1 and hi >= entry:
                    fld = 1; fill_price = entry
                    tp_level = fill_price + tp_pips * pip
                if fld == 0:
                    continue
                # Same-bar TP
                if d == 1 and lo <= tp_level:
                    pnl = tp_pips - sp; reason = 1; mfe = tp_pips; break
                elif d == -1 and hi >= tp_level:
                    pnl = tp_pips - sp; reason = 1; mfe = tp_pips; break
                continue

            # Unrealized pips (mid close based, matches live service)
            if d == 1:
                unreal = (fill_price - mid) / pip
            else:
                unreal = (mid - fill_price) / pip

            if unreal > peak_pft:
                peak_pft = unreal
            if unreal > mfe:
                mfe = unreal
            if unreal < mae:
                mae = unreal

            # TP check
            if d == 1 and lo <= tp_level:
                pnl = tp_pips - sp; reason = 1; mfe = max(mfe, tp_pips); break
            elif d == -1 and hi >= tp_level:
                pnl = tp_pips - sp; reason = 1; mfe = max(mfe, tp_pips); break

            # Activate BE once profit >= be_thresh
            if be_thresh > 0.0 and peak_pft >= be_thresh and not be_active:
                be_active = True

            # BE SL check (SL at entry price)
            if be_active:
                if d == 1 and hi >= fill_price:
                    exit_p = fill_price
                    pnl = -sp; reason = 2; break
                elif d == -1 and lo <= fill_price:
                    exit_p = fill_price
                    pnl = -sp; reason = 2; break

            # Trailing stop
            if trail_start > 0.0 and peak_pft >= trail_start:
                locked = peak_pft - trail_gap
                if locked > 0.0:
                    if d == 1:
                        trail_sl = fill_price - locked * pip
                        if hi >= trail_sl:
                            exit_p = trail_sl
                            pnl = locked - sp; reason = 3; break
                    else:
                        trail_sl = fill_price + locked * pip
                        if lo <= trail_sl:
                            exit_p = trail_sl
                            pnl = locked - sp; reason = 3; break

        # Timeout (HORIZON reached, position still open)
        if fld == 1 and reason == 0:
            end_j  = min(watch_end, n - 1)
            exit_p = bid[end_j] if d == 1 else ask[end_j]
            if d == 1:
                pnl = (fill_price - exit_p) / pip - sp
            else:
                pnl = (exit_p - fill_price) / pip - sp
            reason = 4

        if reason > 0 and ev_count < max_ev:
            pnl_out[ev_count]    = pnl
            mfe_out[ev_count]    = mfe
            mae_out[ev_count]    = mae
            reason_out[ev_count] = reason
            ev_count += 1

        cooldown = (pb_int + int(horizon)) // 2

    return (pnl_out[:ev_count], mfe_out[:ev_count],
            mae_out[:ev_count], reason_out[:ev_count])


# ── Data loading ─────────────────────────────────────────────────────────────────

def load_pair(pair: str):
    path = S5_DIR / f"{pair}_S5_BA.parquet"
    df   = pd.read_parquet(path).sort_index()
    bid  = df["bid_c"].astype(np.float64).values
    ask  = df["ask_c"].astype(np.float64).values
    close = ((bid + ask) / 2).astype(np.float64)
    return bid, ask, close


# ── WF helper ────────────────────────────────────────────────────────────────────

def run_wf(bid, ask, close, shock_flag, vel, pip, oos_start,
           be_thresh, trail_start, trail_gap):
    oos_len = len(close) - oos_start
    chunk   = oos_len // WF_CHUNKS
    passes  = 0
    for c in range(WF_CHUNKS):
        s = oos_start + c * chunk
        e = s + chunk if c < WF_CHUNKS - 1 else len(close)
        pnl, _, _, _ = sim_retrace_betrail(
            bid[s:e], ask[s:e], close[s:e],
            shock_flag[s:e], vel[s:e], pip,
            float(PEAK_BARS), STOP_PIPS, TP_PIPS, float(HORIZON),
            be_thresh, trail_start, trail_gap,
        )
        days = (e - s) / (12 * 60 * 24 / 5)  # approximate trading days
        if len(pnl) > 0 and pnl.sum() / max(days, 1) > 0:
            passes += 1
    return passes


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    # Build config list: baseline + BE variants + trail variants + combined
    configs = []
    configs.append((0.0, 0.0, 0.0))  # baseline
    for be in BE_THRESHS[1:]:
        configs.append((be, 0.0, 0.0))  # BE only
    for ts, tg in product(TRAIL_STARTS[1:], TRAIL_GAPS):
        configs.append((0.0, ts, tg))   # trail only
    for be, ts, tg in product(BE_THRESHS[1:], TRAIL_STARTS[1:], TRAIL_GAPS):
        configs.append((be, ts, tg))    # BE + trail
    # deduplicate
    configs = list(dict.fromkeys(configs))

    results = []

    for pair in PAIRS:
        print(f"\n{'='*60}")
        print(f"  {pair}")
        print(f"{'='*60}")
        bid, ask, close = load_pair(pair)
        pip = PIP[pair]
        _, vel, shock_flag = compute_shock_z(close, pip)
        oos_start = int(len(close) * IS_FRAC)
        oos_days  = (len(close) - oos_start) / (12 * 60 * 24 / 5)

        for be_thresh, trail_start, trail_gap in configs:
            pnl, mfe, mae, reason = sim_retrace_betrail(
                bid[oos_start:], ask[oos_start:], close[oos_start:],
                shock_flag[oos_start:], vel[oos_start:], pip,
                float(PEAK_BARS), STOP_PIPS, TP_PIPS, float(HORIZON),
                be_thresh, trail_start, trail_gap,
            )
            if len(pnl) == 0:
                continue

            n        = len(pnl)
            total    = pnl.sum()
            ppd      = total / oos_days
            wr       = (pnl > 0).mean() * 100
            avg_mfe  = mfe.mean()
            avg_mae  = mae.mean()
            cap      = np.where(mfe > 0, pnl / mfe, 0.0).mean()

            # Exit reason breakdown
            n_tp      = (reason == 1).sum()
            n_be      = (reason == 2).sum()
            n_trail   = (reason == 3).sum()
            n_timeout = (reason == 4).sum()

            wf = run_wf(bid, ask, close, shock_flag, vel, pip, oos_start,
                        be_thresh, trail_start, trail_gap)

            results.append(dict(
                pair=pair, be=be_thresh, ts=trail_start, tg=trail_gap,
                n=n, ppd=ppd, wr=wr, avg_mfe=avg_mfe, avg_mae=avg_mae,
                cap=cap, pct_tp=n_tp/n*100, pct_be=n_be/n*100,
                pct_trail=n_trail/n*100, pct_timeout=n_timeout/n*100,
                wf=wf,
            ))

            tag = f"be={be_thresh:.0f} ts={trail_start:.0f} tg={trail_gap:.0f}"
            print(f"  {tag:22s}  n={n:5d}  ppd={ppd:+7.1f}  WR={wr:5.1f}%  "
                  f"mfe={avg_mfe:5.1f}  mae={avg_mae:5.1f}  cap={cap:.2f}  "
                  f"tp={n_tp/n*100:4.0f}% be={n_be/n*100:3.0f}% "
                  f"tr={n_trail/n*100:3.0f}% to={n_timeout/n*100:3.0f}%  wf={wf}/3")

        gc.collect()

    # ── Portfolio summary ────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    port = (df.groupby(["be", "ts", "tg"])
              .agg(ppd=("ppd", "sum"), wr=("wr", "mean"),
                   avg_mfe=("avg_mfe", "mean"), avg_mae=("avg_mae", "mean"),
                   cap=("cap", "mean"), wf=("wf", "sum"),
                   pct_tp=("pct_tp", "mean"), pct_be=("pct_be", "mean"),
                   pct_trail=("pct_trail", "mean"), pct_timeout=("pct_timeout", "mean"))
              .reset_index()
              .sort_values("ppd", ascending=False))

    print(f"\n{'='*90}")
    print("  PORTFOLIO SUMMARY (all 4 pairs, OOS, sorted by p/d)")
    print(f"{'='*90}")
    print(f"  {'be':>4} {'ts':>4} {'tg':>4}  {'ppd':>8}  {'WR%':>6}  "
          f"{'mfe':>5}  {'mae':>6}  {'cap':>5}  {'tp%':>5}  {'be%':>5}  "
          f"{'tr%':>5}  {'to%':>5}  {'WF':>5}")
    for _, r in port.iterrows():
        tag = "BASELINE" if r.be == 0 and r.ts == 0 else ""
        print(f"  {r.be:>4.0f} {r.ts:>4.0f} {r.tg:>4.0f}  {r.ppd:>+8.1f}  {r.wr:>6.1f}%  "
              f"{r.avg_mfe:>5.1f}  {r.avg_mae:>6.1f}  {r.cap:>5.2f}  "
              f"{r.pct_tp:>5.1f}%  {r.pct_be:>5.1f}%  {r.pct_trail:>5.1f}%  "
              f"{r.pct_timeout:>5.1f}%  {int(r.wf):>2}/12  {tag}")

    # Save
    out = RESULTS / "betrail_sweep.csv"
    port.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
