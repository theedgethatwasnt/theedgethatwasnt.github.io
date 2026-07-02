"""
chop_retrace_entries.py — export the live retrace entries as an ESCMA meta3 (2026-06-12).

Reuses the VALIDATED shock detector (compute_shock_z, thr=2.5, peak=44b) from the live
retrace strategy. Entry = counter-trend fade at watch_start = t_shock + 44 + 1; direction
= −sign(vel) (short after upshock, long after downshock). Writes meta3_retrace_<PAIR>.parquet
(t_pre/t_event/t_timeout/direction/split) so the proven-optimal ESCMA exit learner can learn
a better exit than the fixed TP20/SL30/timeout — on the ONE entry with real edge.

Causal: shock z uses trailing rolling MAD (2048), entry at a bar strictly after the shock.
70/30 temporal split (IS first, OOS last) — same convention as the live OOS validation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
THR, PEAK_BARS, HORIZON, Z_WINDOW, MAD_WIN = 2.5, 44, 600, 6, 2048
# Markov D1 regime filter (validated: mw=10 mt=0.002 sig_thr=0.20, +70.4 p/d WF=9/12)
MK_MW, MK_MT, MK_SIG_THR, MK_MIN_PRIME = 10, 0.002, 0.20, 30
BULL, SIDE, BEAR = 0, 1, 2


def markov_signal_per_bar(ts, close):
    """Causal D1 Markov regime signal mapped to each S5 bar (prev-day signal).
    Mirrors backtest_markov_retrace_filter.build_markov_signals."""
    s = pd.Series(close, index=pd.to_datetime(ts))
    d1 = s.resample("1D").last().dropna()
    lr = np.log(d1 / d1.shift(1)).dropna()
    roll = lr.rolling(MK_MW).sum()
    states = roll.apply(lambda r: (BULL if r > MK_MT else (BEAR if r < -MK_MT else SIDE))
                        if not np.isnan(r) else np.nan).dropna().astype(int)
    T = np.zeros((3, 3)); sig = {}
    for i in range(len(states) - 1):
        st = states.iloc[i]; rs = T[st].sum()
        sig[states.index[i].date()] = (T[st, BULL] - T[st, BEAR]) / rs if rs >= MK_MIN_PRIME else 0.0
        T[st, states.iloc[i + 1]] += 1.0
    days = sorted(sig.keys())
    prev = {d: (sig[days[i-1]] if i > 0 else 0.0) for i, d in enumerate(days)}
    bar_days = pd.DatetimeIndex(pd.to_datetime(ts)).date          # array of datetime.date
    out = np.zeros(len(close), dtype=np.float32)
    last = 0.0; cur_day = None
    for i in range(len(close)):
        d = bar_days[i]
        if d != cur_day:
            cur_day = d; last = prev.get(d, last)
        out[i] = last
    return out


def compute_shock_z(close, pip, w=Z_WINDOW, mad_win=MAD_WIN):
    n = len(close)
    vel = np.empty(n); vel[:w] = 0.0
    vel[w:] = (close[w:] - close[:n-w]) / pip
    vs = pd.Series(vel)
    rm = vs.rolling(mad_win, min_periods=50).median()
    rmad = (vs - rm).abs().rolling(mad_win, min_periods=50).median()
    z = ((vs - rm) / (1.4826 * rmad.clip(lower=1e-6))).fillna(0).values
    return z.astype(np.float64), vel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="USD_JPY")
    ap.add_argument("--markov", action="store_true", help="apply the D1 Markov regime filter (live subset)")
    ap.add_argument("--max-n", type=int, default=0, help="cap entries (subsample evenly) for memory safety")
    args = ap.parse_args()
    pip = 0.01 if "JPY" in args.pair else 0.0001
    tbl = pq.read_table(SCRIPT_DIR / f"features_{args.pair}.parquet", columns=["close", "timestamp"])
    close = tbl.column("close").to_numpy().astype(np.float64)
    ts = tbl.column("timestamp").to_numpy()
    n = close.shape[0]
    z, vel = compute_shock_z(close, pip)
    mk = markov_signal_per_bar(ts, close) if args.markov else None

    cooldown_len = (PEAK_BARS + HORIZON) // 2
    events = []
    sid = 0; cd = 0; n_raw = 0
    warm = MAD_WIN + 100
    for t in range(warm, n - PEAK_BARS - HORIZON - 2):
        if cd > 0:
            cd -= 1; continue
        if abs(z[t]) <= THR:
            continue
        d_shock = 1 if vel[t] > 0 else -1
        n_raw += 1
        cd = cooldown_len
        if mk is not None and not (d_shock * mk[t] > MK_SIG_THR):   # Markov gate
            continue
        trade_dir = -d_shock                       # fade
        watch_start = t + PEAK_BARS + 1
        events.append((sid, watch_start - 60, watch_start, watch_start + HORIZON, trade_dir))
        sid += 1

    m = pd.DataFrame(events, columns=["sample_id", "t_pre", "t_event", "t_timeout", "direction"])
    if args.max_n and len(m) > args.max_n:
        idx = np.linspace(0, len(m)-1, args.max_n).astype(int)
        m = m.iloc[idx].reset_index(drop=True); m["sample_id"] = np.arange(len(m))
    cut = int(len(m) * 0.70)
    m["split"] = ["IS"]*cut + ["OOS"]*(len(m)-cut)
    tag = "retrace_markov" if args.markov else "retrace"
    out = SCRIPT_DIR / f"meta3_{tag}_{args.pair}.parquet"
    m.to_parquet(out)
    keep = f"{len(m)}/{n_raw} ({100*len(events)/max(1,n_raw):.0f}% pass filter)" if args.markov else f"{len(m)}"
    print(f"[retrace-chop{'+markov' if args.markov else ''}] {args.pair}: {keep} entries "
          f"(IS={cut} OOS={len(m)-cut})  long={(m.direction==1).sum()} short={(m.direction==-1).sum()}  -> {out.name}")


if __name__ == "__main__":
    main()
