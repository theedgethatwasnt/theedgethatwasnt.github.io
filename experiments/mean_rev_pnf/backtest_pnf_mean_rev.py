#!/usr/bin/env python3
"""
Session 061 — P&F Entry Direction Sweep: Momentum vs Counter-Trend

Tests BOTH directions on the same P&F framework:
  momentum    (entry_dir=1): LONG after N boxes UP,   SHORT after N boxes DOWN
  counter-trend (entry_dir=0): LONG after N boxes DOWN, SHORT after N boxes UP

Parameters: box ∈ {5,10} × rev ∈ {1,2,3} × entry_n ∈ {1,2,3,4,5,8} × 8 exits × 2 dirs
= 576 configs per pair × 12 pairs = 6,912 total backtests.

SOP: R1 closed bars, R2 within-bar ordering, R3 mid+spread, R5 IS-only gate.
IS/OOS 70/30.
"""

import time
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[3]
BA_DIR  = ROOT / "data" / "m5_ba"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAIRS = [
    "AUD_JPY", "AUD_USD", "CAD_JPY", "CHF_JPY",
    "EUR_GBP", "EUR_JPY", "EUR_USD",
    "GBP_JPY", "GBP_USD",
    "NZD_JPY", "NZD_USD", "USD_JPY",
]
PIP_SIZE = {
    "AUD_JPY": 0.01, "CAD_JPY": 0.01, "CHF_JPY": 0.01, "EUR_JPY": 0.01,
    "GBP_JPY": 0.01, "NZD_JPY": 0.01, "USD_JPY": 0.01,
    "AUD_USD": 0.0001, "EUR_GBP": 0.0001, "EUR_USD": 0.0001,
    "GBP_USD": 0.0001, "NZD_USD": 0.0001,
}

IS_FRAC = 0.70

# ── Parameter space ────────────────────────────────────────────────────────────
# dir × box × rev × entry_n × exit  =  2 × 2 × 3 × 6 × 8 = 576 configs
DIRS      = np.array([0, 1],             dtype=np.int32)  # 0=counter, 1=momentum
BOX_PIPS  = np.array([5, 10],            dtype=np.int32)
REVERSALS = np.array([1, 2, 3],          dtype=np.int32)
ENTRY_N   = np.array([1, 2, 3, 4, 5, 8], dtype=np.int32)

EXIT_DEFS = [
    (0,  3, 2),   # TP=3×box  SL=2×box
    (1,  5, 3),   # TP=5×box  SL=3×box
    (2,  8, 4),   # TP=8×box  SL=4×box
    (3,  1, 0),   # trail 1-box
    (4,  2, 0),   # trail 2-box
    (5,  3, 0),   # trail 3-box
    (6,  5, 0),   # trail 5-box
    (7, 10, 0),   # trail 10-box
]
EXIT_NAMES = ["TP3SL2","TP5SL3","TP8SL4","TR1","TR2","TR3","TR5","TR10"]

def build_configs():
    rows = []
    for d in DIRS:
        for bp in BOX_PIPS:
            for rv in REVERSALS:
                for en in ENTRY_N:
                    for (ec, p1, p2) in EXIT_DEFS:
                        rows.append((d, bp, rv, en, ec, p1, p2))
    return np.array(rows, dtype=np.int32)

CONFIGS = build_configs()
N_CONFIGS = len(CONFIGS)
MAX_TRADES = 15000

# ── Numba kernel ───────────────────────────────────────────────────────────────
@nb.njit(parallel=True)
def run_kernel(opens, highs, lows, closes, spreads,
               configs, spread_gate, pip, is_end,
               trade_pnl, trade_is_flag, trade_cnt):
    N_BARS    = len(opens)
    N_CFGS    = configs.shape[0]

    for ci in prange(N_CFGS):
        entry_dir = configs[ci, 0]  # 0=counter-trend, 1=momentum
        bp_pips   = configs[ci, 1]
        rev       = configs[ci, 2]
        entry_n   = configs[ci, 3]
        exit_c    = configs[ci, 4]
        xp1       = configs[ci, 5]
        xp2       = configs[ci, 6]

        bs = bp_pips * pip          # box size in price units

        # P&F state
        pnf_idx   = 0
        pnf_dir   = 0              # 0=uninit, +1=X(up), -1=O(down)
        col_count = 0
        entry_fired = False        # has entry already fired this column?

        # Position state
        pos      = 0               # 0=flat, +1=long, -1=short
        entry_px = 0.0
        hw_lvl   = 0.0             # high-water P&F level for trail

        t_cnt = 0

        for i in range(N_BARS):
            opn = opens[i];  hi = highs[i];  lo = lows[i];  cl = closes[i]
            sp  = spreads[i]
            in_is = (i < is_end)

            bull = (cl >= opn)
            p1   = hi if bull else lo    # R2: bull → high first
            p2   = lo if bull else hi

            # ── P&F update (two ticks per bar) ────────────────────────────
            entry_signal = 0

            for tick in range(2):
                px = p1 if tick == 0 else p2

                if pnf_dir == 0:
                    pnf_idx   = int(px / bs)
                    pnf_dir   = 1
                    col_count = 1
                    entry_fired = False
                    continue

                delta = int(px / bs) - pnf_idx

                if pnf_dir == 1:       # current column: UP (X column)
                    if delta >= 1:
                        old_cc     = col_count
                        pnf_idx   += delta
                        col_count  += delta
                        if not entry_fired and old_cc < entry_n <= col_count:
                            # momentum: ride the up move (LONG)
                            # counter:  fade the up move (SHORT)
                            entry_signal = 1 if entry_dir == 1 else -1
                            entry_fired  = True
                    elif delta <= -rev:
                        pnf_dir    = -1
                        pnf_idx   += delta
                        col_count  = -delta
                        entry_fired = False

                else:                  # pnf_dir == -1: DOWN (O column)
                    if delta <= -1:
                        old_cc     = col_count
                        pnf_idx   += delta
                        col_count  += (-delta)
                        if not entry_fired and old_cc < entry_n <= col_count:
                            # momentum: ride the down move (SHORT)
                            # counter:  fade the down move (LONG)
                            entry_signal = -1 if entry_dir == 1 else 1
                            entry_fired  = True
                    elif delta >= rev:
                        pnf_dir    = 1
                        pnf_idx   += delta
                        col_count  = delta
                        entry_fired = False

            # ── Update high-water for trailing stop ───────────────────────
            if pos == 1:
                curr_lvl = pnf_idx * bs
                if curr_lvl > hw_lvl:
                    hw_lvl = curr_lvl
            elif pos == -1:
                curr_lvl = pnf_idx * bs
                if curr_lvl < hw_lvl:
                    hw_lvl = curr_lvl

            # ── EXIT logic ─────────────────────────────────────────────────
            exit_triggered = False
            exit_px_val    = cl    # default close

            if pos != 0:
                curr_lvl = pnf_idx * bs

                if exit_c <= 2:
                    # Fixed TP / SL
                    tp_b = float(xp1); sl_b = float(xp2)
                    if pos == 1:
                        tp_price = entry_px + tp_b * bs
                        sl_price = entry_px - sl_b * bs
                        if hi >= tp_price:
                            exit_triggered = True; exit_px_val = tp_price
                        elif lo <= sl_price:
                            exit_triggered = True; exit_px_val = sl_price
                    else:
                        tp_price = entry_px - tp_b * bs
                        sl_price = entry_px + sl_b * bs
                        if lo <= tp_price:
                            exit_triggered = True; exit_px_val = tp_price
                        elif hi >= sl_price:
                            exit_triggered = True; exit_px_val = sl_price
                else:
                    # Trail: exit when level gives back xp1 boxes from high-water
                    trail_dist = float(xp1) * bs
                    if pos == 1:
                        trail_stop = hw_lvl - trail_dist
                        if curr_lvl <= trail_stop and curr_lvl < hw_lvl:
                            exit_triggered = True; exit_px_val = cl
                    else:
                        trail_stop = hw_lvl + trail_dist
                        if curr_lvl >= trail_stop and curr_lvl > hw_lvl:
                            exit_triggered = True; exit_px_val = cl

            if exit_triggered:
                half_sp = sp / 2.0
                if pos == 1:
                    ep = exit_px_val - half_sp
                    pnl = (ep - entry_px) / pip
                else:
                    ep = exit_px_val + half_sp
                    pnl = (entry_px - ep) / pip
                if t_cnt < MAX_TRADES:
                    trade_pnl[ci, t_cnt]     = pnl
                    trade_is_flag[ci, t_cnt] = 1 if in_is else 0
                    t_cnt += 1
                pos = 0

            # ── ENTRY logic ───────────────────────────────────────────────
            if pos == 0 and entry_signal != 0:
                if sp <= spread_gate:     # R5: skip high-spread bars (price units)
                    half_sp  = sp / 2.0
                    if entry_signal == 1:
                        entry_px = cl + half_sp      # buy: pay ask
                    else:
                        entry_px = cl - half_sp      # sell: receive bid
                    pos      = entry_signal
                    hw_lvl   = pnf_idx * bs          # start trail at entry level

        # Close any open position at end
        if pos != 0:
            sp_last = spreads[N_BARS - 1]
            cl_last = closes[N_BARS - 1]
            half_sp = sp_last / 2.0
            if pos == 1:
                ep  = cl_last - half_sp
                pnl = (ep - entry_px) / pip
            else:
                ep  = cl_last + half_sp
                pnl = (entry_px - ep) / pip
            if t_cnt < MAX_TRADES:
                trade_pnl[ci, t_cnt]     = pnl
                trade_is_flag[ci, t_cnt] = 0   # last bar is always OOS for end-of-data
                t_cnt += 1

        trade_cnt[ci] = t_cnt


# ── Stats ──────────────────────────────────────────────────────────────────────
def calc_stats(pnl_arr):
    n = len(pnl_arr)
    if n == 0:
        return None
    wins   = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]
    wr     = len(wins) / n * 100
    pf     = abs(wins.sum() / losses.sum()) if len(losses) and losses.sum() != 0 else np.inf
    total  = pnl_arr.sum()
    return dict(n=n, wr=wr, pf=pf, total=total,
                aw=wins.mean()   if len(wins)   else 0.0,
                al=losses.mean() if len(losses) else 0.0)

def ppd(total, ts_arr, s, e):
    t0 = pd.Timestamp(ts_arr[s],     unit="ns", tz="UTC")
    t1 = pd.Timestamp(ts_arr[e - 1], unit="ns", tz="UTC")
    t_d = (t1 - t0).total_seconds() / 86400 * 5 / 7
    return total / t_d if t_d > 0 else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────
all_results = []

for pair in PAIRS:
    pip = PIP_SIZE[pair]
    path = BA_DIR / f"{pair}_M5_BA.parquet"
    assert path.exists(), f"Missing: {path}"

    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    opens   = df["open"].values.astype(np.float64)
    highs   = df["high"].values.astype(np.float64)
    lows    = df["low"].values.astype(np.float64)
    closes  = df["close"].values.astype(np.float64)
    spreads = (df["ask_c"] - df["bid_c"]).values.astype(np.float64)  # price units
    ts_ns   = df.timestamp.values.astype(np.int64)

    N      = len(df)
    is_end = int(N * IS_FRAC)

    # Spread gate: IS P90 in price units (R5 — computed once, IS-only)
    sp_gate = float(np.percentile(spreads[:is_end], 90))  # price units

    print(f"\n{pair}: {N:,} bars  IS={is_end:,}  sp_gate={sp_gate/pip:.2f}p ({sp_gate:.5f})")

    trade_pnl  = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.float32)
    trade_is   = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.int8)
    trade_cnt  = np.zeros(N_CONFIGS, dtype=np.int32)

    t0 = time.time()
    run_kernel(opens, highs, lows, closes, spreads,
               CONFIGS, sp_gate, pip, is_end,   # sp_gate in price units
               trade_pnl, trade_is, trade_cnt)
    print(f"  kernel: {time.time()-t0:.1f}s")

    for ci in range(N_CONFIGS):
        ed, bp, rv, en, ec, p1, p2 = CONFIGS[ci]
        nc = trade_cnt[ci]
        if nc == 0:
            continue
        pnl_all = trade_pnl[ci, :nc].astype(np.float64)
        is_flag = trade_is[ci, :nc]

        pnl_is  = pnl_all[is_flag == 1]
        pnl_oos = pnl_all[is_flag == 0]

        si = calc_stats(pnl_is)
        so = calc_stats(pnl_oos)
        if not si or not so:
            continue

        is_ppd  = ppd(si["total"],  ts_ns, 0, is_end)
        oos_ppd = ppd(so["total"],  ts_ns, is_end, N)

        all_results.append(dict(
            pair=pair, dir="mom" if ed == 1 else "ctr",
            box=int(bp), rev=int(rv), entry_n=int(en),
            exit=EXIT_NAMES[ec],
            is_n=si["n"], is_wr=si["wr"], is_pf=si["pf"],
            is_ppd=is_ppd,
            oos_n=so["n"], oos_wr=so["wr"], oos_pf=so["pf"],
            oos_ppd=oos_ppd,
        ))

rdf = pd.DataFrame(all_results).sort_values("oos_ppd", ascending=False)

# ── Print ──────────────────────────────────────────────────────────────────────
W = 128
bar = "━" * W

def fr(r):
    return (f"{r['pair']:10s} {r['dir']:3s} b={r['box']:2d} rv={r['rev']} en={r['entry_n']:2d} {r['exit']:8s}  "
            f"IS: n={r['is_n']:5.0f} WR={r['is_wr']:5.1f}% p/d={r['is_ppd']:+8.2f}  "
            f"OOS: n={r['oos_n']:5.0f} WR={r['oos_wr']:5.1f}% p/d={r['oos_ppd']:+9.2f}")

print(f"\n{bar}")
print("TOP 50 CONFIGS BY OOS p/d  (mom=momentum, ctr=counter-trend)")
print(bar)
for _, r in rdf.head(50).iterrows():
    print(fr(r))

print(f"\n{bar}")
print("WORST 10 BY OOS p/d")
print(bar)
for _, r in rdf.tail(10).iterrows():
    print(fr(r))

print(f"\n{bar}")
print("HEAD-TO-HEAD: momentum vs counter-trend — best OOS p/d per pair")
print(bar)
print(f"  {'pair':12s}  {'mom best':>10s}  {'ctr best':>10s}  {'winner':>8s}  best mom config                    best ctr config")
for p in PAIRS:
    sub = rdf[rdf.pair == p]
    mom = sub[sub.dir == "mom"]; ctr = sub[sub.dir == "ctr"]
    if len(mom) == 0 or len(ctr) == 0: continue
    m_best = mom.loc[mom.oos_ppd.idxmax()]; c_best = ctr.loc[ctr.oos_ppd.idxmax()]
    m_val  = m_best.oos_ppd;               c_val  = c_best.oos_ppd
    winner = "mom ←" if m_val > c_val else "ctr ←"
    m_cfg  = f"b={m_best['box']} rv={m_best['rev']} en={m_best['entry_n']} {m_best['exit']}"
    c_cfg  = f"b={c_best['box']} rv={c_best['rev']} en={c_best['entry_n']} {c_best['exit']}"
    print(f"  {p:12s}  {m_val:+10.2f}  {c_val:+10.2f}  {winner:>8s}  {m_cfg:35s}  {c_cfg}")

print(f"\n{bar}")
print("AGGREGATE BY direction: mean OOS p/d across all pairs, boxes, entry_n")
print(bar)
for d in ["mom", "ctr"]:
    sub = rdf[rdf.dir == d]
    print(f"  {d}  mean={sub.oos_ppd.mean():+8.2f}  "
          f"pos%={(sub.oos_ppd > 0).mean()*100:4.0f}%  "
          f"best={sub.oos_ppd.max():+8.2f}")

print(f"\n{bar}")
print("BY entry_n × direction: mean OOS p/d")
print(bar)
for en in ENTRY_N:
    for d in ["mom", "ctr"]:
        sub = rdf[(rdf.entry_n == en) & (rdf.dir == d)]
        if len(sub) == 0: continue
        print(f"  entry_n={en} {d}  mean={sub.oos_ppd.mean():+8.2f}  "
              f"pos%={(sub.oos_ppd > 0).mean()*100:4.0f}%  "
              f"best={sub.oos_ppd.max():+8.2f}")

print(f"\n{bar}")
print("BY exit × direction: mean OOS p/d")
print(bar)
for ex in EXIT_NAMES:
    for d in ["mom", "ctr"]:
        sub = rdf[(rdf.exit == ex) & (rdf.dir == d)]
        if len(sub) == 0: continue
        print(f"  {ex:8s} {d}  mean={sub.oos_ppd.mean():+8.2f}  "
              f"pos%={(sub.oos_ppd > 0).mean()*100:4.0f}%  "
              f"best={sub.oos_ppd.max():+8.2f}")

out = OUT_DIR / "results_all_pairs.csv"
rdf.to_csv(out, index=False)
print(f"\n→ {out}")
