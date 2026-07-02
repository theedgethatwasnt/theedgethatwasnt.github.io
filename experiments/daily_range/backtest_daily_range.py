"""
Daily Range Regime Backtest (Session 065) — M5

Regime gate: prior calendar-day high/low (PDH/PDL)
  Inside PDH-PDL  → mean-revert toward midpoint (PDH+PDL)/2
  Outside PDH-PDL → trend-follow with trailing stop (first-cross entry)

Design:
  Range center: (PDH+PDL)/2 — simpler/more robust than VWAP
  Day boundary: 00:00 UTC (calendar day shift)
  Mean-rev exit: (a) fixed TP at midpoint OR (b) trail stop [swept]
  Mean-rev SL  : PDL-1pip (for LONG), PDH+1pip (for SHORT)
  Breakout exit: trail only (acts as both SL and trailing profit lock)
  Session filter: London 07:00-16:00 UTC vs all-hours [swept]

R-rule compliance:
  R1 Closed bars only (entry at close, exit at bar extremes)
  R3 Mid OHLC, spread = (ask_c - bid_c) / pip deducted at close
  R5 sp_gate = IS P90, hardcoded per pair
  R8 OOS evaluated once after all IS/MC gates

Params: 4 min_dist × 4 mr_exit × 4 br_trail × 2 session = 128 configs/pair
Pairs:  GBP_JPY, USD_JPY, EUR_JPY, GBP_USD
"""

import sys, time, math
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from pathlib import Path

BASE    = Path(__file__).resolve().parents[3]
BA_DIR  = BASE / "data/m5_ba"
IS_FRAC = 0.70
MAX_T   = 15000  # max trades per config in prange buffer

PAIRS = [
    ("GBP_JPY", 0.01,   4.0),
    ("USD_JPY", 0.01,   2.1),
    ("EUR_JPY", 0.01,   2.5),
    ("GBP_USD", 0.0001, 2.4),
]

MIN_DISTS    = [3, 5, 8, 10]   # pips from mid: min distance for mean-rev entry
MR_TRAILS    = [3, 5, 10, -1]  # pips trail for mean-rev; -1 = fixed TP at mid
BR_TRAILS    = [5, 10, 15, 20] # pips trailing stop for breakout
LONDON_FLAGS = [0, 1]           # 0=all hours, 1=07:00–16:00 UTC


def preprocess(ba_path: Path, pip: float):
    df = pd.read_parquet(ba_path)
    ts = pd.to_datetime(df["timestamp"], utc=True)

    df["sp_pips"]   = (df["ask_c"] - df["bid_c"]) / pip
    df["in_london"] = ((ts.dt.hour >= 7) & (ts.dt.hour < 16)).astype(np.int8)
    df["bar_date"]  = ts.dt.normalize()

    # Prior calendar-day high/low (shift(1) on trading days = correct Mon→Fri link)
    daily = df.groupby("bar_date").agg(
        day_high=("high", "max"), day_low=("low", "min")
    )
    daily["pdh"]    = daily["day_high"].shift(1)
    daily["pdl"]    = daily["day_low"].shift(1)
    daily["pd_mid"] = (daily["pdh"] + daily["pdl"]) / 2

    df = df.join(daily[["pdh", "pdl", "pd_mid"]], on="bar_date")
    df[["pdh", "pdl", "pd_mid"]] = df[["pdh", "pdl", "pd_mid"]].fillna(0.0)
    return df


def build_configs():
    """Return (128, 4) array: [min_dist_p, mr_trail_p, br_trail_p, london_only]"""
    rows = []
    for md in MIN_DISTS:
        for mrt in MR_TRAILS:
            for brt in BR_TRAILS:
                for lo in LONDON_FLAGS:
                    rows.append([float(md), float(mrt), float(brt), float(lo)])
    return np.array(rows, dtype=np.float64)


def config_name(row):
    md, mrt, brt, lo = row
    mr_str = "tp_mid" if mrt < 0 else f"mr{int(mrt)}p"
    sess   = "lon" if lo > 0.5 else "all"
    return f"md{int(md)}_{mr_str}_br{int(brt)}_{sess}"


@nb.njit(parallel=True)
def run_batch(
    opens, highs, lows, closes, sp_pips,
    pdh, pdl, pd_mid, in_london, bar_chunks,
    configs, pip, sp_gate, is_end,
    tpnl, tchunk, tcnt,
):
    N  = len(opens)
    NC = configs.shape[0]

    for ci in prange(NC):
        md_p    = configs[ci, 0]        # min_dist in pips
        mrt_p   = configs[ci, 1]        # mr_trail pips (or <0 = fixed TP)
        brt_p   = configs[ci, 2]        # br_trail pips
        lo_flag = configs[ci, 3] > 0.5
        use_tp  = mrt_p < 0.0          # True → fixed TP at mid for mean-rev

        pos = 0; entry_px = 0.0; hw_pnl = 0.0
        sl_px = 0.0; tp_px = 0.0; mode = 0  # mode 1=mean-rev, 2=breakout
        prev_cl = 0.0; t = 0

        for i in range(N):
            hi = highs[i]; lo = lows[i]; cl = closes[i]
            sp = sp_pips[i]; ck = bar_chunks[i]
            pdh_i = pdh[i]; pdl_i = pdl[i]; mid_i = pd_mid[i]
            exited = False

            # ── Exit check ──────────────────────────────────────────────────
            if pos != 0 and t < MAX_T:
                tr_p = brt_p if mode == 2 else mrt_p  # trail pips for this mode

                if pos == 1:  # LONG
                    pnl_hi = (hi - entry_px) / pip
                    if pnl_hi > hw_pnl: hw_pnl = pnl_hi

                    if mode == 1 and use_tp and hi >= tp_px:
                        tpnl[ci, t] = np.float32((tp_px - entry_px) / pip - sp)
                        tchunk[ci, t] = ck; t += 1; exited = True

                    if not exited and tr_p > 0.0:
                        trail_px = entry_px + (hw_pnl - tr_p) * pip
                        if lo <= trail_px:
                            tpnl[ci, t] = np.float32((trail_px - entry_px) / pip - sp)
                            tchunk[ci, t] = ck; t += 1; exited = True

                    if not exited and mode == 1 and lo <= sl_px:
                        tpnl[ci, t] = np.float32((sl_px - entry_px) / pip - sp)
                        tchunk[ci, t] = ck; t += 1; exited = True

                else:  # SHORT
                    pnl_lo = (entry_px - lo) / pip
                    if pnl_lo > hw_pnl: hw_pnl = pnl_lo

                    if mode == 1 and use_tp and lo <= tp_px:
                        tpnl[ci, t] = np.float32((entry_px - tp_px) / pip - sp)
                        tchunk[ci, t] = ck; t += 1; exited = True

                    if not exited and tr_p > 0.0:
                        trail_px = entry_px - (hw_pnl - tr_p) * pip
                        if hi >= trail_px:
                            tpnl[ci, t] = np.float32((entry_px - trail_px) / pip - sp)
                            tchunk[ci, t] = ck; t += 1; exited = True

                    if not exited and mode == 1 and hi >= sl_px:
                        tpnl[ci, t] = np.float32((entry_px - sl_px) / pip - sp)
                        tchunk[ci, t] = ck; t += 1; exited = True

                if exited:
                    pos = 0; hw_pnl = 0.0; mode = 0

            # ── Entry check (at bar close) ───────────────────────────────────
            if pos == 0 and sp <= sp_gate and pdh_i > 0.0:
                if not lo_flag or in_london[i] > 0:
                    if pdl_i <= cl <= pdh_i:
                        # Inside range: mean-rev toward mid
                        dist = (cl - mid_i) / pip
                        if dist >= md_p:
                            pos = -1; entry_px = cl; hw_pnl = 0.0; mode = 1
                            sl_px = pdh_i + pip; tp_px = mid_i
                        elif dist <= -md_p:
                            pos = 1; entry_px = cl; hw_pnl = 0.0; mode = 1
                            sl_px = pdl_i - pip; tp_px = mid_i
                    else:
                        # Outside range: breakout — first-cross entry only
                        if cl > pdh_i and prev_cl <= pdh_i:
                            pos = 1; entry_px = cl; hw_pnl = 0.0; mode = 2
                        elif cl < pdl_i and prev_cl >= pdl_i:
                            pos = -1; entry_px = cl; hw_pnl = 0.0; mode = 2

            prev_cl = cl

        tcnt[ci] = t


def run_pair(pair, pip, sp_gate):
    ba_path = BA_DIR / f"{pair}_M5_BA.parquet"
    df = preprocess(ba_path, pip)
    n  = len(df)
    is_end  = int(n * IS_FRAC)
    oos_days = (n - is_end) / 288.0

    ck0 = is_end // 3; ck1 = 2 * (is_end // 3)
    bar_chunks = np.zeros(n, np.int8)
    bar_chunks[ck0:ck1]  = 1
    bar_chunks[ck1:is_end] = 2
    bar_chunks[is_end:]  = 3

    opens  = df["open"].values.astype(np.float64)
    highs  = df["high"].values.astype(np.float64)
    lows   = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    sp_arr = df["sp_pips"].values.astype(np.float64)
    pdh_a  = df["pdh"].values.astype(np.float64)
    pdl_a  = df["pdl"].values.astype(np.float64)
    mid_a  = df["pd_mid"].values.astype(np.float64)
    lon_a  = df["in_london"].values.astype(np.int8)

    configs = build_configs()
    NC = len(configs)

    tpnl   = np.zeros((NC, MAX_T), np.float32)
    tchunk = np.zeros((NC, MAX_T), np.int8)
    tcnt   = np.zeros(NC, np.int64)

    t0 = time.time()
    run_batch(opens, highs, lows, closes, sp_arr,
              pdh_a, pdl_a, mid_a, lon_a, bar_chunks,
              configs, pip, sp_gate, is_end,
              tpnl, tchunk, tcnt)
    elapsed = time.time() - t0

    total_trades = int(tcnt.sum())
    print(f"  Done in {elapsed:.1f}s  |  {total_trades:,} total trades  ({total_trades//NC}/config avg)")

    # ── Stage 1: IS walk-forward ─────────────────────────────────────────────
    wf_pass = []
    for ci in range(NC):
        t = int(tcnt[ci])
        pnl_ci    = tpnl[ci, :t].astype(np.float64)
        chunk_ci  = tchunk[ci, :t]
        ok = True
        for ck in range(3):
            m = chunk_ci == ck
            if m.sum() < 5 or pnl_ci[m].sum() <= 0:
                ok = False; break
        if ok:
            wf_pass.append(ci)
    print(f"  Stage 1: IS walk-forward screen... {len(wf_pass)}/{NC} passed")

    if not wf_pass:
        return []

    # ── Stage 2: MC t-test (top 100) ────────────────────────────────────────
    candidates = sorted(wf_pass,
        key=lambda ci: tpnl[ci, :int(tcnt[ci])][tchunk[ci, :int(tcnt[ci])] < 3].sum(),
        reverse=True)[:100]

    mc_pass = []
    for ci in candidates:
        t = int(tcnt[ci])
        is_mask = tchunk[ci, :t] < 3
        pnl_is  = tpnl[ci, :t][is_mask].astype(np.float64)
        n_is = len(pnl_is)
        if n_is < 10: continue
        mean_v = float(pnl_is.mean()); std_v = float(pnl_is.std())
        if std_v <= 0: continue
        t_stat = mean_v / (std_v / math.sqrt(n_is))
        # one-tailed p-value: P(T > t_stat) = erfc(t_stat / sqrt(2)) / 2
        p_val = math.erfc(t_stat / math.sqrt(2.0)) / 2.0
        if p_val < 0.05:
            mc_pass.append(ci)
    print(f"  Stage 2: MC t-test... {len(mc_pass)}/{len(candidates)} passed")

    if not mc_pass:
        return []

    # ── Stage 3: OOS (sealed) ────────────────────────────────────────────────
    oos_results = []
    for ci in mc_pass:
        t = int(tcnt[ci])
        pnl_ci   = tpnl[ci, :t].astype(np.float64)
        chunk_ci = tchunk[ci, :t]
        oos_mask = chunk_ci == 3
        oos_pnl  = pnl_ci[oos_mask]
        n_oos    = int(oos_mask.sum())
        if n_oos < 10: continue
        oos_pd   = float(oos_pnl.sum()) / oos_days
        oos_wr   = float((oos_pnl > 0).sum()) / n_oos
        if oos_pd > 0:
            oos_results.append((ci, oos_pd, n_oos, oos_wr))

    oos_results.sort(key=lambda x: x[1], reverse=True)
    print(f"  Stage 3: OOS... {len(oos_results)}/{len(mc_pass)} positive OOS p/d")
    return oos_results


# ── Main ──────────────────────────────────────────────────────────────────────

configs = build_configs()

print("Daily Range Regime Backtest")
print(f"Pairs: {[p for p,_,_ in PAIRS]}")
print(f"Configs/pair: {len(configs)}  (4 min_dist × 4 mr_exit × 4 br_trail × 2 session)")
print()

# Warmup
print("Warming up Numba JIT...", end=" ", flush=True)
dummy   = np.ones(500, np.float64)
dummy_i = np.zeros(500, np.int8)
dummy_c = np.zeros(500, np.int64)
dummy_cfg = configs[:1].copy()
dummy_pnl = np.zeros((1, MAX_T), np.float32)
dummy_chk = np.zeros((1, MAX_T), np.int8)
t0 = time.time()
run_batch(dummy, dummy, dummy, dummy, dummy,
          dummy, dummy, dummy, dummy_i, dummy_i,
          dummy_cfg, 0.01, 4.0, 350,
          dummy_pnl, dummy_chk, dummy_c)
print(f"done in {time.time()-t0:.1f}s\n")

summary = []
for pair, pip, sp_gate in PAIRS:
    ba_path = BA_DIR / f"{pair}_M5_BA.parquet"
    df_info = pd.read_parquet(ba_path, columns=["open"])
    n = len(df_info)
    is_end = int(n * IS_FRAC)
    print(f"{'='*60}")
    print(f"  {pair}  (pip={pip}, sp_gate={sp_gate}p)")
    print(f"{'='*60}")
    print(f"  Bars={n:,}  IS={is_end:,}  OOS={n-is_end:,}  ({(n-is_end)/288:.0f} trading days OOS)")

    results = run_pair(pair, pip, sp_gate)

    if results:
        print(f"\n  🟢 Top OOS configs for {pair}:")
        print(f"  {'config':<30} {'oos_pd':>8} {'n_oos':>7} {'wr':>6}")
        for ci, oos_pd, n_oos, oos_wr in results[:8]:
            name = config_name(configs[ci])
            print(f"  {name:<30} {oos_pd:>8.2f} {n_oos:>7d} {oos_wr*100:>5.0f}%")
        best_pd = results[0][1]
        best_cfg = config_name(configs[results[0][0]])
    else:
        print(f"  🔴 No configs with positive OOS p/d")
        best_pd = 0.0; best_cfg = "—"

    summary.append((pair, len(results), best_pd, best_cfg))
    print()

print(f"{'='*60}")
print("SUMMARY — Daily Range sweep")
print(f"{'='*60}")
print(f"{'pair':<12} {'oos_winners':>11} {'best_oos_pd':>12} {'best_config'}")
for pair, winners, best_pd, best_cfg in summary:
    sym = "🟢" if winners > 0 else "🔴"
    print(f"{sym} {pair:<10} {winners:>11} {best_pd:>12.2f}  {best_cfg}")
