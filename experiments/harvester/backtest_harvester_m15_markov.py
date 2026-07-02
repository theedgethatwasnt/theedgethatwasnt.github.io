#!/usr/bin/env python3
"""
Harvester M15 + Markov D1 Regime Filter
========================================
Test whether the D1 Markov regime signal can fix WF3 trending-regime failure
in the M15 Harvester.

Background:
  - M15 Harvester best IS p/d = +0.19 (EUR_USD sma=14 nc=2 dist=4-5×sp tp=0.6 sl=1.5)
  - WF3 failure: ECB hiking cycle Jan 2023 – Aug 2024 → trending EUR/USD kills mean-rev
  - Markov D1 signal: IC is NEGATIVE (counter-trend at D1 scale)
    Bull regime (signal > 0) → expect bearish next day → supports SHORT fade
    Bear regime (signal < 0) → expect bullish next day → supports LONG fade

Filter logic (matches retrace strategy convention):
  exhaustion_dir = -direction_of_trade
    (fading bull bars → direction=-1 → exhaustion_dir=+1)
  Allow entry when: exhaustion_dir * markov_signal > sig_thr
    sig_thr = -999 → no filter (baseline)
    sig_thr =  0.0 → allow when regime signal aligns with exhaustion direction
    sig_thr =  0.1 → stronger alignment required

Sweep:
  markov_window ∈ {5, 10}
  markov_thr    ∈ {0.002, 0.005}
  sig_thr       ∈ {-999, 0.0, 0.05, 0.10, 0.20}

Candidate configs: top-N per pair from existing M15 sweep (IS p/d > 0).
"""

import gc, time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from numba import njit, prange
import warnings; warnings.filterwarnings("ignore")

PROJECT      = Path("/path/to/projects/fx-core")
M5_DIR       = PROJECT / "data/m5_ba"
PREV_RESULTS = PROJECT / "research/experiments/harvester/results/harvester_m15_results.csv"
RES_DIR      = PROJECT / "research/experiments/harvester/results"
RES_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = {
    "EUR_USD": {"file_m5": "EUR_USD_M5_BA.parquet", "pip": 0.0001},
    "EUR_JPY": {"file_m5": "EUR_JPY_M5_BA.parquet", "pip": 0.01},
    "GBP_USD": {"file_m5": "GBP_USD_M5_BA.parquet", "pip": 0.0001},
}

IS_FRAC       = 0.70
N_WF_CHUNKS   = 3
MIN_IS_TRADES = 30   # relaxed — regime filter reduces trade count
MC_SHUFFLES   = 1000
MAX_HOLD_BARS = 8    # 2h at M15
ATR_PERIOD    = 14
SESSION_START = 7
SESSION_END   = 21

SMA_OPT = [7, 10, 14, 20]

MARKOV_WINDOWS = [5, 10]
MARKOV_THRS    = [0.002, 0.005]
SIG_THRESHOLDS = [-999.0, 0.0, 0.05, 0.10, 0.20]
MIN_PRIME      = 30

BULL, SIDE, BEAR = 0, 1, 2


# ── Markov D1 signal ───────────────────────────────────────────────────────────

def build_markov_signals(pair: str, m_window: int, m_thr: float) -> pd.Series:
    """Causal D1 Markov signal. Returns Series indexed by date."""
    df     = pd.read_parquet(M5_DIR / f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    d1     = df["close"].resample("1D").last().dropna()
    lr     = np.log(d1 / d1.shift(1)).dropna()
    roll   = lr.rolling(m_window).sum()
    states = roll.apply(
        lambda r: BULL if r > m_thr else (BEAR if r < -m_thr else SIDE)
        if not np.isnan(r) else np.nan
    ).dropna().astype(int)

    T       = np.zeros((3, 3), dtype=np.float64)
    signals = {}
    for i in range(len(states) - 1):
        day = states.index[i].date()
        s   = states.iloc[i]
        # Generate signal BEFORE updating T (causal — signal uses only past transitions)
        row_sum = T[s].sum()
        if row_sum >= MIN_PRIME:
            signals[day] = (T[s, BULL] - T[s, BEAR]) / row_sum
        else:
            signals[day] = 0.0
        T[s, states.iloc[i + 1]] += 1.0

    return pd.Series(signals)


def map_m15_to_signal(m15_timestamps, daily_signals: pd.Series) -> np.ndarray:
    """Map D1 Markov signal to each M15 bar (causal: prior trading-day signal)."""
    sig_dict    = daily_signals.to_dict()
    sorted_days = sorted(sig_dict.keys())
    prev        = {}
    for i, d in enumerate(sorted_days):
        prev[d] = sig_dict[sorted_days[i - 1]] if i > 0 else 0.0

    out = np.zeros(len(m15_timestamps), dtype=np.float64)
    for i, ts in enumerate(m15_timestamps):
        d = pd.Timestamp(ts).date()
        out[i] = prev.get(d, 0.0)
    return out


# ── Data loading ───────────────────────────────────────────────────────────────

def load_m15(pair_cfg: dict) -> pd.DataFrame:
    path = M5_DIR / pair_cfg["file_m5"]
    df = duckdb.query(
        f'SELECT timestamp, open, high, low, close, bid_c, ask_c '
        f'FROM "{path}" ORDER BY timestamp'
    ).df()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close", "bid_c", "ask_c"]:
        df[col] = df[col].astype(np.float64)
    m15 = df.resample("15min", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "bid_c": "last", "ask_c": "last",
    }).dropna(subset=["open", "close"])
    m15 = m15.reset_index()
    m15["spread_pips"] = ((m15["ask_c"] - m15["bid_c"]).astype(np.float64)
                          / pair_cfg["pip"]).clip(0.1, 30.0)
    return m15


# ── Numba kernels ──────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def compute_sma(close, period):
    n = len(close); out = np.empty(n, dtype=np.float64); s = np.float64(0.0)
    for i in range(n):
        s += close[i]
        if i >= period: s -= close[i - period]
        out[i] = s / min(i + 1, period)
    return out


@njit(cache=True, fastmath=True)
def compute_atr14(high, low, close):
    n = len(close); atr = np.empty(n, dtype=np.float64); atr[0] = high[0] - low[0]
    for i in range(1, n):
        tr     = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr[i] = (atr[i-1] * (ATR_PERIOD - 1) + tr) / ATR_PERIOD \
                 if i >= ATR_PERIOD else (atr[i-1] * (i-1) + tr) / i
    return atr


@njit(cache=True, fastmath=True)
def run_trade_m15(close, high, low, spread_pips, entry_bar,
                  entry_px, direction, tp_price, sl_price, pip_size, max_hold):
    """TP/SL in price units. Returns (pnl_pips, bars_held, exit_reason)."""
    n      = len(close)
    tp_lev = entry_px + np.float64(direction) * tp_price
    sl_lev = entry_px - np.float64(direction) * sl_price
    end    = min(entry_bar + max_hold + 1, n)
    for j in range(entry_bar + 1, end):
        bull = close[j] >= close[j-1]
        hit_tp = False; hit_sl = False
        if direction == 1:
            if bull: hit_tp = high[j] >= tp_lev; hit_sl = low[j]  <= sl_lev
            else:    hit_sl = low[j]  <= sl_lev; hit_tp = high[j] >= tp_lev
        else:
            if not bull: hit_tp = low[j] <= tp_lev; hit_sl = high[j] >= sl_lev
            else:        hit_sl = high[j] >= sl_lev; hit_tp = low[j] <= tp_lev
        if hit_tp: return tp_price / pip_size,  np.int64(j - entry_bar), np.int64(0)
        if hit_sl: return -sl_price / pip_size, np.int64(j - entry_bar), np.int64(1)
    last = min(entry_bar + max_hold, n - 1)
    pnl  = (close[last] - entry_px) * np.float64(direction) / pip_size \
           - np.float64(0.5) * spread_pips[last]
    return pnl, np.int64(last - entry_bar), np.int64(2)


@njit(cache=True, fastmath=True)
def run_segment_m15_mk(close, high, low, open_, spread_pips, sma, atr,
                       in_session, pip_size, markov_arr, sig_thr_f,
                       sma_period, n_consec, dist_mult_min, dist_mult_max,
                       tp_frac, sl_frac, atr_mult,
                       seg_start, seg_end, sp_gate):
    """run_segment_m15 + Markov D1 filter.

    Filter: when sig_thr_f > -900, allow entry only when
      (-direction) * markov_arr[i] > sig_thr_f
    i.e., exhaustion direction aligns with D1 regime signal.
    """
    warmup     = sma_period + n_consec + 2
    start      = max(seg_start, np.int64(warmup))
    next_entry = start
    use_mk     = sig_thr_f > np.float64(-900.0)

    total_pips = np.float64(0.0)
    n_trades   = np.int64(0)
    n_wins     = np.int64(0)
    n_tp       = np.int64(0)

    dist_lo = dist_mult_min * sp_gate
    dist_hi = dist_mult_max * sp_gate

    i = start
    while i < seg_end - 1:
        if i < next_entry: i += 1; continue
        if not in_session[i]: i += 1; continue

        sp = spread_pips[i]
        if sp > sp_gate: i += 1; continue

        dist = abs(close[i] - sma[i]) / pip_size
        if dist < dist_lo or dist > dist_hi: i += 1; continue

        tp_price = dist * tp_frac * pip_size
        sl_price = dist * sl_frac * pip_size
        if tp_price < np.float64(0.5) * sp * pip_size: i += 1; continue

        above_sma = close[i] > sma[i]
        all_bull = True; all_bear = True
        for k in range(np.int64(1), n_consec + np.int64(1)):
            j = i - k
            if close[j] < open_[j]: all_bull = False
            if close[j] >= open_[j]: all_bear = False
            if not all_bull and not all_bear: break

        direction = np.int64(0)
        if all_bull and above_sma:     direction = np.int64(-1)
        elif all_bear and not above_sma: direction = np.int64(1)
        if direction == np.int64(0): i += 1; continue

        if atr_mult > np.float64(0.0):
            if abs(close[i] - close[i - n_consec]) < atr_mult * atr[i]:
                i += 1; continue

        # Markov regime filter
        if use_mk:
            exh_dir = np.float64(-direction)   # exhaustion direction
            if exh_dir * markov_arr[i] <= sig_thr_f:
                i += 1; continue

        entry_px = close[i] + np.float64(direction) * np.float64(0.5) * sp * pip_size
        pnl_pips, bars, reason = run_trade_m15(
            close, high, low, spread_pips, i, entry_px, direction,
            tp_price, sl_price, pip_size, np.int64(MAX_HOLD_BARS))

        total_pips += pnl_pips
        n_trades   += np.int64(1)
        if pnl_pips > np.float64(0.0): n_wins += np.int64(1)
        if reason  == np.int64(0):     n_tp   += np.int64(1)
        next_entry = i + bars + np.int64(1)
        i += 1

    return total_pips, n_trades, n_wins, n_tp


@njit(cache=True, fastmath=True)
def run_segment_pnl_m15_mk(close, high, low, open_, spread_pips, sma, atr,
                            in_session, pip_size, markov_arr, sig_thr_f,
                            sma_period, n_consec, dist_mult_min, dist_mult_max,
                            tp_frac, sl_frac, atr_mult,
                            seg_start, seg_end, sp_gate):
    """Returns per-trade PnL array for MC validation."""
    MAX_T = 5_000
    pnl_arr = np.empty(MAX_T, dtype=np.float64)
    n_t     = np.int64(0)

    warmup     = sma_period + n_consec + 2
    start      = max(seg_start, np.int64(warmup))
    next_entry = start
    use_mk     = sig_thr_f > np.float64(-900.0)

    dist_lo = dist_mult_min * sp_gate
    dist_hi = dist_mult_max * sp_gate

    i = start
    while i < seg_end - 1 and n_t < MAX_T:
        if i < next_entry: i += 1; continue
        if not in_session[i]: i += 1; continue

        sp = spread_pips[i]
        if sp > sp_gate: i += 1; continue

        dist = abs(close[i] - sma[i]) / pip_size
        if dist < dist_lo or dist > dist_hi: i += 1; continue

        tp_price = dist * tp_frac * pip_size
        sl_price = dist * sl_frac * pip_size
        if tp_price < np.float64(0.5) * sp * pip_size: i += 1; continue

        above_sma = close[i] > sma[i]
        all_bull = True; all_bear = True
        for k in range(np.int64(1), n_consec + np.int64(1)):
            j = i - k
            if close[j] < open_[j]: all_bull = False
            if close[j] >= open_[j]: all_bear = False
            if not all_bull and not all_bear: break

        direction = np.int64(0)
        if all_bull and above_sma:       direction = np.int64(-1)
        elif all_bear and not above_sma: direction = np.int64(1)
        if direction == np.int64(0): i += 1; continue

        if atr_mult > np.float64(0.0):
            if abs(close[i] - close[i - n_consec]) < atr_mult * atr[i]:
                i += 1; continue

        if use_mk:
            exh_dir = np.float64(-direction)
            if exh_dir * markov_arr[i] <= sig_thr_f:
                i += 1; continue

        entry_px = close[i] + np.float64(direction) * np.float64(0.5) * sp * pip_size
        pnl_pips, bars, reason = run_trade_m15(
            close, high, low, spread_pips, i, entry_px, direction,
            tp_price, sl_price, pip_size, np.int64(MAX_HOLD_BARS))

        pnl_arr[n_t]  = pnl_pips
        n_t          += np.int64(1)
        next_entry    = i + bars + np.int64(1)
        i += 1

    return pnl_arr[:n_t]


# ── Monte Carlo ────────────────────────────────────────────────────────────────

def run_mc(pnl_arr, is_days, n_shuffles=MC_SHUFFLES, seed=42):
    if len(pnl_arr) < 30:
        return np.nan
    actual_pd = pnl_arr.sum() / is_days
    rng    = np.random.default_rng(seed)
    signs  = rng.choice(np.array([-1.0, 1.0]), size=(n_shuffles, len(pnl_arr)))
    shuffl = (np.abs(pnl_arr) * signs).sum(axis=1) / is_days
    return float((shuffl >= actual_pd).mean())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load candidate configs from prior M15 sweep
    prev_df = pd.read_csv(PREV_RESULTS)
    print(f"Loaded {len(prev_df)} prior M15 results from {PREV_RESULTS.name}")

    candidates = {}
    for pair in PAIRS:
        cands = prev_df[(prev_df["pair"] == pair) & (prev_df["is_pd"] > 0.0)].copy()
        candidates[pair] = cands
        print(f"  {pair}: {len(cands)} candidate configs (IS p/d > 0)")

    all_rows = []
    compiled = [False]

    for pair_name, pair_cfg in PAIRS.items():
        m5_path = M5_DIR / pair_cfg["file_m5"]
        if not m5_path.exists():
            print(f"\n  {pair_name}: parquet missing — skip"); continue

        cands = candidates[pair_name]
        if len(cands) == 0:
            print(f"\n  {pair_name}: no candidate configs — skip"); continue

        print(f"\n{'─'*60}")
        print(f"  {pair_name}  |  {len(cands)} candidate configs")
        pip = pair_cfg["pip"]

        df    = load_m15(pair_cfg)
        close = np.ascontiguousarray(df["close"].values.astype(np.float64))
        high  = np.ascontiguousarray(df["high"].values.astype(np.float64))
        low   = np.ascontiguousarray(df["low"].values.astype(np.float64))
        open_ = np.ascontiguousarray(df["open"].values.astype(np.float64))
        sp    = np.ascontiguousarray(df["spread_pips"].values.astype(np.float64))
        atr   = np.ascontiguousarray(compute_atr14(high, low, close))

        hours   = df["timestamp"].dt.hour.values
        in_sess = np.ascontiguousarray(((hours >= SESSION_START) & (hours < SESSION_END)).astype(np.uint8))

        sma_cache = {p: np.ascontiguousarray(compute_sma(close, p)) for p in SMA_OPT}

        n      = len(close)
        is_end = int(n * IS_FRAC)

        is_sess_sp = sp[:is_end][in_sess[:is_end].astype(bool)]
        sp_gate    = float(np.percentile(is_sess_sp, 90))

        ts_idx   = df["timestamp"].dt.normalize()
        is_days  = max(1, int(ts_idx.iloc[:is_end].nunique()))
        oos_days = max(1, int(ts_idx.iloc[is_end:].nunique()))

        chunk     = is_end // N_WF_CHUNKS
        wf_starts = [k * chunk     for k in range(N_WF_CHUNKS)]
        wf_ends   = [(k+1) * chunk for k in range(N_WF_CHUNKS)]

        m15_ts = df["timestamp"].values  # numpy datetime64 array

        # Warm up Numba (first pass)
        if not compiled[0]:
            print("  Compiling Numba ...", end="", flush=True)
            t0 = time.time()
            _dummy_mk = np.zeros(200, dtype=np.float64)
            run_segment_m15_mk(
                close[:200], high[:200], low[:200], open_[:200], sp[:200],
                sma_cache[14][:200], atr[:200], in_sess[:200], np.float64(pip),
                _dummy_mk, np.float64(-999.0),
                np.int64(14), np.int64(2),
                np.float64(4.0), np.float64(5.0),
                np.float64(0.6), np.float64(1.5), np.float64(1.0),
                np.int64(0), np.int64(200), np.float64(sp_gate))
            compiled[0] = True
            print(f" {time.time()-t0:.1f}s")

        # Build Markov signals for all (mw, mt) combinations
        mk_signals = {}
        for mw in MARKOV_WINDOWS:
            for mt in MARKOV_THRS:
                daily_sig     = build_markov_signals(pair_name, mw, mt)
                mk_arr        = map_m15_to_signal(m15_ts, daily_sig)
                mk_signals[(mw, mt)] = np.ascontiguousarray(mk_arr)
                nonzero = (mk_arr != 0.0).mean() * 100
                print(f"    Markov mw={mw} mt={mt:.3f}  nonzero={nonzero:.0f}%  "
                      f"mean(nonzero)={mk_arr[mk_arr != 0].mean():+.3f}")

        # Sweep: for each Markov config + sig_thr, run all candidate configs
        print(f"  Sweeping {len(cands)} configs × "
              f"{len(MARKOV_WINDOWS)*len(MARKOV_THRS)} mk_params × "
              f"{len(SIG_THRESHOLDS)} sig_thrs ...", end="", flush=True)
        t0 = time.time()

        for mw in MARKOV_WINDOWS:
            for mt in MARKOV_THRS:
                mk_arr = mk_signals[(mw, mt)]

                for sig_thr in SIG_THRESHOLDS:
                    sig_thr_f = np.float64(sig_thr)

                    for _, row in cands.iterrows():
                        sma_p  = int(row["sma"])
                        nc     = int(row["n_consec"])
                        dmin_m = float(row["dist_min_m"])
                        dmax_m = float(row["dist_max_m"])
                        tf     = float(row["tp_frac"])
                        sf     = float(row["sl_frac"])
                        am     = float(row["atr_mult"])

                        sma_v = sma_cache[sma_p]

                        def seg(s, e):
                            return run_segment_m15_mk(
                                close, high, low, open_, sp, sma_v, atr, in_sess,
                                np.float64(pip), mk_arr, sig_thr_f,
                                np.int64(sma_p), np.int64(nc),
                                np.float64(dmin_m), np.float64(dmax_m),
                                np.float64(tf), np.float64(sf), np.float64(am),
                                np.int64(s), np.int64(e), np.float64(sp_gate))

                        is_pips, is_n, is_w, is_tp = seg(0, is_end)
                        oos_pips, oos_n, oos_w, _   = seg(is_end, n)

                        wf_pds = []
                        for k in range(N_WF_CHUNKS):
                            pk, nk, _, _ = seg(wf_starts[k], wf_ends[k])
                            wf_pds.append(pk / (is_days / N_WF_CHUNKS) if nk > 0 else 0.0)

                        is_pd  = is_pips  / is_days  if is_n  > 0 else 0.0
                        oos_pd = oos_pips / oos_days if oos_n > 0 else 0.0
                        wf_pass = (int(is_n) >= MIN_IS_TRADES) and all(p > 0 for p in wf_pds)

                        mc_p = np.nan
                        if wf_pass:
                            pnl_arr = run_segment_pnl_m15_mk(
                                close, high, low, open_, sp, sma_v, atr, in_sess,
                                np.float64(pip), mk_arr, sig_thr_f,
                                np.int64(sma_p), np.int64(nc),
                                np.float64(dmin_m), np.float64(dmax_m),
                                np.float64(tf), np.float64(sf), np.float64(am),
                                np.int64(0), np.int64(is_end), np.float64(sp_gate))
                            mc_p = run_mc(pnl_arr, is_days)

                        all_rows.append({
                            "pair":     pair_name,
                            "mw":       mw, "mt": mt, "sig_thr": sig_thr,
                            "sma":      sma_p, "n_consec": nc,
                            "dist_min_m": dmin_m, "dist_max_m": dmax_m,
                            "tp_frac":  tf, "sl_frac": sf, "atr_mult": am,
                            "is_pd":    round(is_pd,  2),
                            "is_wr":    round(int(is_w) / int(is_n) if is_n > 0 else 0.0, 3),
                            "is_n":     int(is_n),
                            "wf1":      round(wf_pds[0], 2),
                            "wf2":      round(wf_pds[1], 2),
                            "wf3":      round(wf_pds[2], 2),
                            "wf_pass":  wf_pass,
                            "oos_pd":   round(oos_pd, 2),
                            "oos_wr":   round(int(oos_w) / int(oos_n) if oos_n > 0 else 0.0, 3),
                            "oos_n":    int(oos_n),
                            "mc_p":     round(float(mc_p), 4) if not np.isnan(mc_p) else np.nan,
                        })

        print(f" {time.time()-t0:.1f}s")

        del df, close, high, low, open_, sp, atr, in_sess, sma_cache, mk_signals
        gc.collect()

    # ── Report ──────────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(all_rows)
    out_path = RES_DIR / "harvester_m15_markov_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nResults → {out_path}  ({len(df_out)} rows)")

    print("\n" + "=" * 80)

    wf_mc   = df_out[df_out["wf_pass"] & (df_out["mc_p"] < 0.05)].sort_values("is_pd", ascending=False)
    wf_only = df_out[df_out["wf_pass"]].sort_values("is_pd", ascending=False)

    if len(wf_mc) > 0:
        print(f"\n🟢 WF + MC survivors ({len(wf_mc)} configs):")
        print(wf_mc[["pair","mw","mt","sig_thr","sma","n_consec",
                      "dist_min_m","dist_max_m","tp_frac","sl_frac","atr_mult",
                      "is_pd","is_wr","is_n","wf1","wf2","wf3","oos_pd","mc_p"]
                    ].to_string(index=False))
    else:
        print("\n🔴 No WF + MC survivors.")

    if len(wf_only) > 0 and len(wf_mc) == 0:
        print(f"\nWF-only survivors ({len(wf_only)}):")
        print(wf_only[["pair","mw","mt","sig_thr","sma","n_consec",
                        "dist_min_m","dist_max_m","tp_frac","sl_frac","atr_mult",
                        "is_pd","is_wr","is_n","wf1","wf2","wf3","mc_p"]
                      ].head(20).to_string(index=False))

    # Per-pair summary: compare baseline (sig_thr=-999) vs best Markov
    print("\n" + "─" * 80)
    print("Per-pair summary: baseline vs best Markov filter\n")
    cols = ["pair", "mw", "mt", "sig_thr", "is_pd", "is_n", "wf1", "wf2", "wf3", "wf_pass", "mc_p"]
    for pair_name in PAIRS:
        sub = df_out[df_out["pair"] == pair_name]
        if len(sub) == 0: continue

        # Best canonical config (top IS p/d in original sweep)
        cands = candidates[pair_name]
        if len(cands) == 0: continue
        best_cfg = cands.nlargest(1, "is_pd").iloc[0]
        sma_p  = int(best_cfg["sma"])
        nc     = int(best_cfg["n_consec"])
        dmin_m = float(best_cfg["dist_min_m"])
        dmax_m = float(best_cfg["dist_max_m"])
        tf     = float(best_cfg["tp_frac"])
        sf     = float(best_cfg["sl_frac"])
        am     = float(best_cfg["atr_mult"])

        cfg_rows = sub[
            (sub["sma"] == sma_p) & (sub["n_consec"] == nc) &
            (sub["dist_min_m"] == dmin_m) & (sub["dist_max_m"] == dmax_m) &
            (sub["tp_frac"] == tf) & (sub["sl_frac"] == sf) & (sub["atr_mult"] == am)
        ].sort_values("sig_thr")

        print(f"  {pair_name} — best original config: sma={sma_p} nc={nc} "
              f"dist={dmin_m}-{dmax_m}×sp tp={tf} sl={sf} atr={am}")
        print(cfg_rows[cols].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
