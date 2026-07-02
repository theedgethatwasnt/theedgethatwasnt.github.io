"""
Breakout Study — Session 053.

Three-section research script on GBP_USD M5 + H1:

  1. SMA10 persistence  — after hi<sma10(hi) / lo>sma10(lo) fires, how many
                          bars does the signal persist? Distribution at M5 and H1.
                          Also: next-N-bar return conditional on signal.

  2. Tape-ratio signal  — rolling W-bar % of bars where hi<sma10(hi)  (bear tape)
                          or lo>sma10(lo) (bull tape). Correlation & quartile study.

  3. Breakout/Fade sim  — for N-bar highest high:
                           touch + close above  → LONG  (penetration)
                           touch + close below  → SHORT (fade/rejection)
                          Symmetric for N-bar lowest low.
                          Sweep 4 exit methods × parameter grid:
                            a) MFE/MAE ratio  threshold ∈ {1.5, 2.0, 3.0, 5.0}
                            b) Fixed trailing stop (pips) ∈ {5, 10, 20}
                            c) SMA-cross  period ∈ {5, 10, 20}
                            d) PSAR  (AF0=0.02 step=0.02 max=0.2)
                          Report: n_trades, win%, avg_pnl, p/d, avg_mfe, avg_mae.

All P/L in pips (mid-based, spread deducted at entry).
"""

import time, math
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')

PAIR     = "GBP_USD"
PIP      = 0.0001
MAX_HOLD = 60       # bars — max trade duration (5h on M5, 60h on H1)
SMA_W    = 10       # SMA window for signal
TAPE_W   = [5, 10]  # rolling window for tape-ratio
N_VALS   = [5, 10, 20, 50]   # lookback for N-bar high/low
TRAIL_DISTS  = [5.0, 10.0, 20.0]
SMA_EXIT_PERIODS = [5, 10, 20]
MFE_MAE_THRS = [1.5, 2.0, 3.0, 5.0]
OOS_FRAC = 0.30   # study on OOS slice only (avoids IS contamination)

OUT_DIR = Path(__file__).parent / "breakout_study"
OUT_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def sma(arr: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(arr).rolling(w, min_periods=w).mean().values


def compute_psar(hi: np.ndarray, lo: np.ndarray,
                 af0: float = 0.02, step: float = 0.02, af_max: float = 0.2
                 ) -> np.ndarray:
    """Standard Parabolic SAR. Returns psar array (stop price)."""
    n = len(hi)
    psar = np.empty(n)
    ep   = np.empty(n)
    af   = np.empty(n)
    tr   = np.empty(n, dtype=np.int8)   # 1=long, -1=short

    tr[0] = 1; psar[0] = lo[0]; ep[0] = hi[0]; af[0] = af0
    for i in range(1, n):
        if tr[i-1] == 1:
            raw = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            raw = min(raw, lo[i-1], lo[i-2] if i >= 2 else lo[i-1])
            if lo[i] < raw:                    # flip to short
                tr[i] = -1; psar[i] = ep[i-1]; ep[i] = lo[i]; af[i] = af0
            else:
                tr[i] = 1; psar[i] = raw
                ep[i] = max(ep[i-1], hi[i])
                af[i] = min(af[i-1] + step, af_max) if hi[i] > ep[i-1] else af[i-1]
        else:
            raw = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            raw = max(raw, hi[i-1], hi[i-2] if i >= 2 else hi[i-1])
            if hi[i] > raw:                    # flip to long
                tr[i] = 1; psar[i] = ep[i-1]; ep[i] = hi[i]; af[i] = af0
            else:
                tr[i] = -1; psar[i] = raw
                ep[i] = min(ep[i-1], lo[i])
                af[i] = min(af[i-1] + step, af_max) if lo[i] < ep[i-1] else af[i-1]
    return psar


def simulate_trade(entry_price: float, direction: int,
                   hi_fwd: np.ndarray, lo_fwd: np.ndarray,
                   cl_fwd: np.ndarray, sp_fwd: np.ndarray,
                   psar_fwd: np.ndarray,
                   sma_arrs: dict,    # period → sma array (aligned to fwd)
                   pip: float, max_hold: int
                   ) -> dict:
    """
    Forward-simulate one trade from entry.
    hi_fwd[0], lo_fwd[0], ... = bar immediately after entry.
    Returns dict of {exit_method: pnl_pips} plus mfe, mae at each bar.
    """
    n = min(len(cl_fwd), max_hold)
    d = direction

    # Running trackers
    mfe = 0.0
    mae = 0.0
    peak_for_trail = entry_price  # high-water (long) or low-water (short)

    exits = {}   # method_key → pnl at exit
    trail_peaks = {t: entry_price for t in TRAIL_DISTS}

    for i in range(n):
        h = hi_fwd[i]; l = lo_fwd[i]; c = cl_fwd[i]; sp = sp_fwd[i]

        # Update MFE / MAE
        if d == 1:
            mfe = max(mfe, (h - entry_price) / pip)
            mae = max(mae, (entry_price - l) / pip)
        else:
            mfe = max(mfe, (entry_price - l) / pip)
            mae = max(mae, (h - entry_price) / pip)

        bar_pnl = (d * (c - entry_price) / pip) - sp  # exit at close, deduct spread

        # ── a) MFE/MAE ratio exits ─────────────────────────────────────────
        for thr in MFE_MAE_THRS:
            key = f'mfemae_{thr}'
            if key not in exits:
                ratio = mfe / max(mae, 0.1)
                if ratio >= thr:
                    exits[key] = bar_pnl

        # ── b) Fixed trailing stop ─────────────────────────────────────────
        for td in TRAIL_DISTS:
            key = f'trail_{int(td)}p'
            if key not in exits:
                if d == 1:
                    trail_peaks[td] = max(trail_peaks[td], h)
                    if l <= trail_peaks[td] - td * pip:
                        exits[key] = d * (trail_peaks[td] - td * pip - entry_price) / pip - sp
                else:
                    trail_peaks[td] = min(trail_peaks[td], l)
                    if h >= trail_peaks[td] + td * pip:
                        exits[key] = d * (entry_price - (trail_peaks[td] + td * pip)) / pip - sp

        # ── c) SMA cross ───────────────────────────────────────────────────
        for period, sma_arr in sma_arrs.items():
            key = f'sma{period}'
            if key not in exits and i < len(sma_arr):
                sv = sma_arr[i]
                if not math.isnan(sv):
                    if d == 1 and c < sv:
                        exits[key] = bar_pnl
                    elif d == -1 and c > sv:
                        exits[key] = bar_pnl

        # ── d) PSAR ────────────────────────────────────────────────────────
        if 'psar' not in exits and i < len(psar_fwd):
            pv = psar_fwd[i]
            if d == 1 and l < pv:
                exits['psar'] = d * (pv - entry_price) / pip - sp
            elif d == -1 and h > pv:
                exits['psar'] = d * (entry_price - pv) / pip - sp

        # Max hold exit
        if i == n - 1:
            for m in _all_exit_keys():
                if m not in exits:
                    exits[m] = bar_pnl  # close at last bar

    # Fill any method that never fired (should not happen after max_hold)
    final_pnl = d * (cl_fwd[n-1] - entry_price) / pip - sp_fwd[n-1]
    for m in _all_exit_keys():
        if m not in exits:
            exits[m] = final_pnl

    return {'exits': exits, 'mfe': mfe, 'mae': mae}


def _all_exit_keys():
    keys = [f'mfemae_{t}' for t in MFE_MAE_THRS]
    keys += [f'trail_{int(t)}p' for t in TRAIL_DISTS]
    keys += [f'sma{p}' for p in SMA_EXIT_PERIODS]
    keys += ['psar']
    return keys


# ── Section 1: SMA10 Persistence ─────────────────────────────────────────────

def section1_persistence(hi_m5, lo_cl_m5, cl_m5, hi_h1, lo_h1, cl_h1,
                          oos_days_m5, oos_days_h1):
    print("\n" + "═"*80)
    print("  SECTION 1 — SMA10 Signal Persistence")
    print("═"*80)

    results = []
    for name, hi, lo, cl, oos_days in [
        ('M5',  hi_m5, lo_cl_m5, cl_m5, oos_days_m5),
        ('H1',  hi_h1, lo_h1,    cl_h1, oos_days_h1),
    ]:
        for sig_name, cond_arr in [
            ('hi<sma10_hi', hi < sma(hi, SMA_W)),
            ('lo>sma10_lo', lo > sma(lo, SMA_W)),
        ]:
            valid = ~np.isnan(cond_arr if isinstance(cond_arr, np.ndarray) else np.array([False]))
            fire_idx = np.where(cond_arr)[0]
            if len(fire_idx) == 0:
                continue

            persist_lens = []
            for fi in fire_idx:
                length = 0
                j = fi
                while j < len(cond_arr) and cond_arr[j]:
                    length += 1; j += 1
                persist_lens.append(length)
            persist_lens = np.array(persist_lens)

            # Next-bar directional accuracy
            direction = 1 if 'lo>' in sig_name else -1   # lo>sma10_lo → bullish, hi<sma10_hi → bearish
            nb_rets = []
            n5_rets = []
            for fi in fire_idx:
                if fi + 1 < len(cl):
                    nb_rets.append(direction * (cl[fi+1] - cl[fi]) / PIP)
                if fi + 5 < len(cl):
                    n5_rets.append(direction * (cl[fi+5] - cl[fi]) / PIP)

            r = dict(
                tf=name, signal=sig_name,
                n_events=len(fire_idx),
                pct_fire=f"{len(fire_idx)/len(cond_arr)*100:.1f}%",
                mean_persist=f"{persist_lens.mean():.1f}b",
                p25=int(np.percentile(persist_lens, 25)),
                p50=int(np.percentile(persist_lens, 50)),
                p75=int(np.percentile(persist_lens, 75)),
                p90=int(np.percentile(persist_lens, 90)),
                nb_ret=f"{np.mean(nb_rets):+.2f}p" if nb_rets else "n/a",
                n5_ret=f"{np.mean(n5_rets):+.2f}p" if n5_rets else "n/a",
            )
            results.append(r)
            print(f"  {name} {sig_name}: n={len(fire_idx):,} ({r['pct_fire']}) "
                  f"persist p25/50/75/90={r['p25']}/{r['p50']}/{r['p75']}/{r['p90']}b "
                  f"mean={r['mean_persist']}  "
                  f"nb_ret={r['nb_ret']}  n5_ret={r['n5_ret']}")

    pd.DataFrame(results).to_csv(OUT_DIR / "persistence.csv", index=False)
    print(f"\n  → {OUT_DIR/'persistence.csv'}")
    return results


# ── Section 2: Tape Ratio Signal ─────────────────────────────────────────────

def section2_tape_ratio(hi, lo, cl, oos_days):
    print("\n" + "═"*80)
    print("  SECTION 2 — Tape-Ratio Signal (rolling % above/below SMA10)")
    print("═"*80)

    sma_hi = sma(hi, SMA_W)
    sma_lo = sma(lo, SMA_W)
    bear = hi < sma_hi   # bar high below sma10(highs)
    bull = lo > sma_lo   # bar low above sma10(lows)

    results = []
    for w in TAPE_W:
        bear_tape = pd.Series(bear.astype(float)).rolling(w, min_periods=w).mean().values
        bull_tape = pd.Series(bull.astype(float)).rolling(w, min_periods=w).mean().values

        for sig_name, tape in [('bear_tape', bear_tape), ('bull_tape', bull_tape)]:
            direction = -1 if sig_name == 'bear_tape' else 1
            valid = ~np.isnan(tape)
            t_arr = tape[valid]
            c_arr = cl[valid]

            next1 = np.concatenate([c_arr[1:] - c_arr[:-1], [np.nan]]) / PIP * direction
            next5 = np.concatenate([c_arr[5:] - c_arr[:-5], [np.nan]*5]) / PIP * direction
            next1 = next1[~np.isnan(next1)]
            next5 = next5[~np.isnan(next5)]

            qt = np.percentile(t_arr, [0, 25, 50, 75, 100])

            print(f"\n  W={w} {sig_name} quartile analysis (next-1-bar / next-5-bar avg return):")
            for qi in range(4):
                lo_q = qt[qi]; hi_q = qt[qi+1]
                mask = (t_arr >= lo_q) & (t_arr <= hi_q)
                n_q = mask.sum()
                if n_q == 0: continue
                avg1 = next1[mask[:len(next1)]].mean() if n_q > 1 else float('nan')
                avg5 = next5[mask[:len(next5)]].mean() if n_q > 1 else float('nan')
                print(f"    Q{qi+1} tape=[{lo_q:.2f},{hi_q:.2f}]  n={n_q:,}  "
                      f"nb={avg1:+.3f}p  n5={avg5:+.3f}p")
                results.append(dict(w=w, sig=sig_name, q=qi+1,
                                    tape_lo=round(lo_q,3), tape_hi=round(hi_q,3),
                                    n=int(n_q), nb=round(avg1,3), n5=round(avg5,3)))

    pd.DataFrame(results).to_csv(OUT_DIR / "tape_ratio.csv", index=False)
    print(f"\n  → {OUT_DIR/'tape_ratio.csv'}")
    return results


# ── Section 3: Breakout / Fade Simulation ────────────────────────────────────

def build_entry_events(hi, lo, cl, op, sp_arr, N):
    """
    Returns list of (bar_idx, direction, entry_price, spread) for N-bar breakout/fade.
    Uses a minimum gap of 1 bar between entries to avoid stacking.
    """
    n = len(cl)
    events = []
    last_entry = -2

    for i in range(N + 1, n - MAX_HOLD - 1):
        if i - last_entry < 2:  # cooldown
            continue
        hh = hi[i-N:i].max()
        ll = lo[i-N:i].min()
        sp = sp_arr[i]

        if hi[i] >= hh:                           # touched N-bar high
            if cl[i] > hh:                        # penetrated → LONG
                events.append((i, 1, cl[i] + sp * PIP, sp))
                last_entry = i
            elif cl[i] <= hh - 0.5 * sp * PIP:   # clear rejection → SHORT fade
                events.append((i, -1, cl[i] - sp * PIP, sp))
                last_entry = i

        elif lo[i] <= ll:                          # touched N-bar low
            if cl[i] < ll:                        # penetrated → SHORT
                events.append((i, -1, cl[i] - sp * PIP, sp))
                last_entry = i
            elif cl[i] >= ll + 0.5 * sp * PIP:   # clear rejection → LONG fade
                events.append((i, 1, cl[i] + sp * PIP, sp))
                last_entry = i

    return events


def section3_breakout(hi, lo, cl, op, sp_arr, oos_days):
    print("\n" + "═"*80)
    print("  SECTION 3 — Breakout/Fade Entry + Exit Comparison")
    print("═"*80)

    psar_all = compute_psar(hi, lo)
    sma_all  = {p: sma(cl, p) for p in SMA_EXIT_PERIODS}

    all_results = []

    for N in N_VALS:
        events = build_entry_events(hi, lo, cl, op, sp_arr, N)
        if not events:
            continue

        # Accumulate per-method P/L
        method_pnls = {k: [] for k in _all_exit_keys()}
        mfe_list = []; mae_list = []
        dirs = []

        for bar_idx, direction, entry_price, sp in events:
            end = min(bar_idx + 1 + MAX_HOLD, len(cl))
            sl = slice(bar_idx + 1, end)
            fwd_len = end - bar_idx - 1
            if fwd_len < 2:
                continue

            sma_fwd = {p: sma_all[p][sl] for p in SMA_EXIT_PERIODS}
            psar_fwd = psar_all[sl]

            res = simulate_trade(
                entry_price, direction,
                hi[sl], lo[sl], cl[sl], sp_arr[sl],
                psar_fwd, sma_fwd, PIP, MAX_HOLD
            )
            for k, v in res['exits'].items():
                method_pnls[k].append(v)
            mfe_list.append(res['mfe'])
            mae_list.append(res['mae'])
            dirs.append(direction)

        if not mfe_list:
            continue

        n_trades = len(mfe_list)
        avg_mfe = np.mean(mfe_list)
        avg_mae = np.mean(mae_list)
        n_long  = sum(1 for d in dirs if d == 1)
        n_short = n_trades - n_long

        print(f"\n  N={N}  events={n_trades}  long={n_long} short={n_short}  "
              f"avg_mfe={avg_mfe:.1f}p  avg_mae={avg_mae:.1f}p  "
              f"mfe/mae={avg_mfe/max(avg_mae,0.01):.2f}x")
        print(f"  {'exit_method':<20} {'n':>5} {'win%':>6} {'avg_p/l':>8} "
              f"{'p/d':>7} {'mfe>0%':>6}")
        print(f"  {'-'*60}")

        for method, pnls in sorted(method_pnls.items()):
            if not pnls: continue
            pnls_arr = np.array(pnls)
            win_pct  = np.mean(pnls_arr > 0) * 100
            avg_pnl  = np.mean(pnls_arr)
            ppd      = pnls_arr.sum() / oos_days
            mfe_pos  = np.mean(np.array(mfe_list) > 0) * 100
            print(f"  {method:<20} {len(pnls_arr):>5} {win_pct:>6.1f}% {avg_pnl:>7.2f}p "
                  f"{ppd:>7.1f} {mfe_pos:>5.0f}%")
            all_results.append(dict(N=N, method=method, n=len(pnls_arr),
                                    win_pct=round(win_pct,1), avg_pnl=round(avg_pnl,2),
                                    ppd=round(ppd,1), avg_mfe=round(avg_mfe,1),
                                    avg_mae=round(avg_mae,1)))

    df_r = pd.DataFrame(all_results)
    df_r.to_csv(OUT_DIR / "breakout_exits.csv", index=False)
    print(f"\n  → {OUT_DIR/'breakout_exits.csv'}")
    return df_r


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading data...", flush=True)
    mid = (pd.read_parquet(DATA_DIR_MID / f'{PAIR}_M5.parquet')
           .sort_values('timestamp').reset_index(drop=True))
    ba  = (pd.read_parquet(DATA_DIR_BA  / f'{PAIR}_M5_BA.parquet')
           .sort_values('timestamp').reset_index(drop=True))
    mid['ts_key'] = mid['timestamp'].astype(str).str[:19]
    ba['ts_key']  = ba['timestamp'].astype(str).str[:19]
    df = (mid.merge(ba[['ts_key', 'bid_c', 'ask_c']], on='ts_key', how='inner')
          .reset_index(drop=True))
    print(f"  {len(df):,} bars  {df.timestamp.min()} → {df.timestamp.max()}")

    nb      = len(df)
    is_end  = int(nb * (1 - OOS_FRAC))
    oos_len = nb - is_end
    oos_days_m5 = oos_len / (24.0 * 12.0)

    # OOS slice for all three sections
    df_oos = df.iloc[is_end:].reset_index(drop=True)
    hi_m5  = df_oos.high.values.astype(np.float64)
    lo_m5  = df_oos.low.values.astype(np.float64)
    cl_m5  = df_oos.close.values.astype(np.float64)
    op_m5  = df_oos.open.values.astype(np.float64)
    sp_m5  = ((df_oos.ask_c - df_oos.bid_c) / PIP).clip(lower=0.1).values.astype(np.float64)
    gate   = float(np.percentile(((df.ask_c - df.bid_c)/PIP).clip(lower=0.1).values[:is_end], 90))

    # Apply spread gate (same as ZR backtest)
    sp_gated = np.where(sp_m5 > gate, gate, sp_m5)

    # H1 via resample
    df_h1 = (df_oos.set_index('timestamp')[['high', 'low', 'close']]
             .resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last'})
             .dropna().reset_index())
    hi_h1 = df_h1.high.values.astype(np.float64)
    lo_h1 = df_h1.low.values.astype(np.float64)
    cl_h1 = df_h1.close.values.astype(np.float64)
    oos_days_h1 = len(df_h1) / 24.0

    print(f"  OOS: {oos_len:,} M5 bars = {oos_days_m5:.1f} days  |  "
          f"{len(df_h1):,} H1 bars = {oos_days_h1:.1f} days  |  gate={gate:.2f}p")

    # Section 1
    section1_persistence(hi_m5, lo_m5, cl_m5, hi_h1, lo_h1, cl_h1,
                         oos_days_m5, oos_days_h1)

    # Section 2
    section2_tape_ratio(hi_m5, lo_m5, cl_m5, oos_days_m5)

    # Section 3
    section3_breakout(hi_m5, lo_m5, cl_m5, op_m5, sp_gated, oos_days_m5)

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
