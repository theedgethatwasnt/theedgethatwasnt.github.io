#!/usr/bin/env python3
"""
Markov Regime Filter on Post-Shock Retrace — Phase 3
=====================================================
Question: Does conditioning retrace entries on the D1 Markov regime signal
improve performance vs unfiltered baseline?

Regime signal = P(Bull|state_t) - P(Bear|state_t) from causal transition matrix
  — negative IC confirmed in Phase 1 → COUNTER-TREND at D1 scale
  — aligned with retrace direction: upshock→SHORT favored in Bull regime (signal>0)
                                   downshock→LONG favored in Bear regime (signal<0)

Filter condition: shock_dir * markov_signal > signal_threshold
  (only enter when regime signal aligns with AND exceeds threshold)

Baseline: thr=2.5, peak=44b, sd=3, tp=20p → +56 p/d OOS WF=12/12 mc_p=0.0000
Sweep: markov_window={5,10}, markov_thr={0.002,0.005,0.010}
       signal_threshold={0.0, 0.1, 0.2, 0.3}
"""

import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
from scipy.stats import spearmanr
import requests, os, sys
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from dotenv import load_dotenv; load_dotenv()

PROJECT  = Path(__file__).resolve().parents[3]
S5_DIR   = PROJECT / "data" / "s5_ba"
M5_DIR   = PROJECT / "data" / "m5_ba"
RESULTS  = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PAIRS   = ["GBP_JPY", "USD_JPY", "EUR_JPY", "AUD_JPY"]
PIP     = {p: 0.01 for p in PAIRS}
IS_FRAC = 0.70

# ── Retrace params (validated baseline) ───────────────────────────────────────
Z_WIN    = 6
MAD_WIN  = 2048
PEAK_B   = 44
SD       = 3.0      # stop distance (honest benchmark: sd=3 = zero fill inflation)
TP       = 20.0
HORIZON  = 600
Z_THR    = 2.5

# ── Markov params to sweep ────────────────────────────────────────────────────
MARKOV_WINDOWS = [5, 10]
MARKOV_THRS    = [0.002, 0.005, 0.010]
SIG_THRESHOLDS = [0.0, 0.1, 0.2, 0.3]   # min |signal| alignment required
MIN_PRIME      = 30   # min transitions before trusting matrix

# ── WF / MC validation ────────────────────────────────────────────────────────
N_WF_CHUNKS = 3
N_MC        = 1000
WF_PASS     = 10   # out of 12 (4 pairs × 3 chunks)

BULL, SIDE, BEAR = 0, 1, 2

# ── Telegram ──────────────────────────────────────────────────────────────────
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.environ.get("TELEGRAM_CHAT_ID",  "")

def tg(msg: str):
    if not (TG_TOKEN and TG_CHAT):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg},
            timeout=5,
        )
    except Exception:
        pass


# ── D1 Markov signal builder ──────────────────────────────────────────────────

def build_markov_signals(pair: str, m_window: int, m_thr: float) -> pd.Series:
    """
    Causal D1 Markov regime signal for one pair.
    Signal(D) = P(Bull|state_D) - P(Bear|state_D) from T built on [0, D-1].
    Returns pd.Series indexed by date (day D → signal usable at OPEN of D).
    """
    df   = pd.read_parquet(M5_DIR / f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    d1   = df["close"].resample("1D").last().dropna()
    lr   = np.log(d1 / d1.shift(1)).dropna()

    roll = lr.rolling(m_window).sum()
    states = roll.apply(
        lambda r: BULL if r > m_thr else (BEAR if r < -m_thr else SIDE)
        if not np.isnan(r) else np.nan
    ).dropna().astype(int)

    T = np.zeros((3, 3), dtype=np.float64)
    signals = {}

    for i in range(len(states) - 1):
        day = states.index[i].date()
        s   = states.iloc[i]

        row_sum = T[s].sum()
        if row_sum >= MIN_PRIME:
            signals[day] = (T[s, BULL] - T[s, BEAR]) / row_sum
        else:
            signals[day] = 0.0

        T[s, states.iloc[i + 1]] += 1.0

    return pd.Series(signals)


def map_s5_to_signal(s5_timestamps, daily_signals: pd.Series) -> np.ndarray:
    """
    For each S5 bar, return the Markov signal from the PREVIOUS trading day
    (causal: signal for day D is computed using data up to D-1 close).
    """
    sig_dict = daily_signals.to_dict()
    sorted_days = sorted(sig_dict.keys())
    # prev_day_signal[D] = signal available at OPEN of D = signal computed at close of D-1
    prev = {}
    for i, d in enumerate(sorted_days):
        prev[d] = sig_dict[sorted_days[i - 1]] if i > 0 else 0.0

    out = np.zeros(len(s5_timestamps), dtype=np.float32)
    for i, ts in enumerate(s5_timestamps):
        d = pd.Timestamp(ts).date()
        out[i] = prev.get(d, 0.0)
    return out


# ── S5 shock detection ────────────────────────────────────────────────────────

def compute_shock_z(close, pip, w=Z_WIN, mad_win=MAD_WIN):
    n   = len(close)
    vel = np.empty(n, dtype=np.float32)
    vel[:w] = 0.0
    vel[w:] = (close[w:] - close[:n-w]) / pip
    vs  = pd.Series(vel)
    rm  = vs.rolling(mad_win, min_periods=50).median()
    ad  = (vs - rm).abs()
    rmd = ad.rolling(mad_win, min_periods=50).median()
    return ((vs - rm) / (1.4826 * rmd.clip(lower=1e-6))).fillna(0).values.astype(np.float32)


# ── Core sim (Numba, mirrors backtest_post_shock_retrace.py) ──────────────────

@njit
def sim_retrace_filtered(bid, ask, close, shock_flag, vel,
                          pip, peak_bars, stop_pips, tp_pips, horizon,
                          s5_markov, sig_thresh):
    """
    Retrace sim with Markov filter gate.
    sig_thresh < -900 → no filter (baseline).
    Filter condition: (vel[t] > 0 ? +1 : -1) * s5_markov[t] > sig_thresh
    """
    n      = len(close)
    pb     = int(peak_bars)
    hor    = int(horizon)
    max_ev = n // 10

    pnl_arr    = np.zeros(max_ev, dtype=np.float32)
    tp_hit_arr = np.zeros(max_ev, dtype=np.int8)
    ev_count   = 0
    cooldown   = 0

    for t in range(Z_WIN, n - pb - hor - 2):
        if cooldown > 0:
            cooldown -= 1
            continue
        if shock_flag[t] != 1:
            continue

        # Markov filter
        if sig_thresh > -900.0:
            shock_dir_f = 1.0 if vel[t] > 0.0 else -1.0
            if shock_dir_f * s5_markov[t] <= sig_thresh:
                continue

        d = 1 if vel[t] > 0.0 else -1

        # Track peak during observation window
        peak_ask = ask[t]
        peak_bid = bid[t]
        for k in range(1, pb + 1):
            j = t + k
            if ask[j] > peak_ask: peak_ask = ask[j]
            if bid[j] < peak_bid: peak_bid = bid[j]

        watch_start = t + pb + 1
        watch_end   = t + pb + hor
        if watch_start >= n or watch_end >= n:
            continue

        sp = (ask[watch_start] - bid[watch_start]) / pip

        # sd=0 → market entry; sd>0 → stop entry from peak
        fld = 0
        fill_price = 0.0
        tp_level   = 0.0

        if stop_pips == 0.0:
            fld = 1
            fill_price = bid[watch_start] if d == 1 else ask[watch_start]
            tp_level   = fill_price - tp_pips * pip * d
        else:
            entry = (peak_ask - stop_pips * pip) if d == 1 else (peak_bid + stop_pips * pip)
            tp_level = entry - tp_pips * pip * d

            for k in range(watch_start, watch_end + 1):
                triggered = bid[k] <= entry if d == 1 else ask[k] >= entry
                if triggered:
                    fld = 1
                    fill_price = entry
                    break

        if fld == 0:
            cooldown = (pb + hor) // 2
            continue

        tp_hit = 0
        for k in range(watch_start + (0 if stop_pips == 0.0 else 1), watch_end + 1):
            at_tp = bid[k] <= tp_level if d == 1 else ask[k] >= tp_level
            if at_tp:
                tp_hit = 1
                break

        if tp_hit == 1:
            pnl = tp_pips - sp
        else:
            end = watch_end
            exit_px = bid[end] if d == 1 else ask[end]
            pnl = (fill_price - exit_px) / pip * d - sp

        if ev_count < max_ev:
            pnl_arr[ev_count]    = pnl
            tp_hit_arr[ev_count] = tp_hit
            ev_count += 1

        cooldown = (pb + hor) // 2

    return pnl_arr[:ev_count], tp_hit_arr[:ev_count]


# ── Per-config evaluator ──────────────────────────────────────────────────────

def eval_filtered(bid, ask, close, shock_flag, vel, pip, oos_days,
                  s5_markov, sig_thresh, return_pnl=False):
    pnl, tp = sim_retrace_filtered(
        bid, ask, close, shock_flag, vel, pip,
        float(PEAK_B), SD, TP, float(HORIZON),
        s5_markov, sig_thresh,
    )
    if len(pnl) == 0:
        return (0.0, 0.0, 0, 0.0) if not return_pnl else (0.0, 0.0, 0, 0.0, pnl)
    ppd  = pnl.sum() / oos_days
    wr   = tp.mean() * 100
    n    = len(pnl)
    evpd = n / oos_days
    if return_pnl:
        return ppd, wr, n, evpd, pnl
    return ppd, wr, n, evpd


def mc_pvalue(pnl_lists, days_list, actual_ppd, n_mc=N_MC):
    """Portfolio sign-shuffle MC."""
    rng     = np.random.default_rng(42)
    sim_ppd = np.zeros(n_mc)
    for pnls, days in zip(pnl_lists, days_list):
        if len(pnls) == 0 or days == 0:
            continue
        for m in range(n_mc):
            signs    = rng.choice([-1, 1], size=len(pnls))
            sim_ppd[m] += (pnls * signs).sum() / days
    return float((sim_ppd >= actual_ppd).mean())


# ── Warmup ────────────────────────────────────────────────────────────────────

tg("📊 Markov×Retrace Phase 3 starting\n"
   f"Pairs: {', '.join(PAIRS)}\n"
   f"Baseline: sd=3 thr=2.5 peak=44b tp=20p → +56p/d WF=12/12\n"
   f"Testing regime filter at D1 scale (IC<0 → counter-trend aligned)")

print("Warming up Numba …")
_b = np.ones(3000, dtype=np.float64) * 150.0
_a = _b + 0.03
_c = _b + 0.015
_sf = np.zeros(3000, dtype=np.int8); _sf[100]=1; _sf[600]=1; _sf[1200]=1
_v  = np.zeros(3000, dtype=np.float32); _v[100]=5.0; _v[600]=-5.0; _v[1200]=3.0
_mk = np.zeros(3000, dtype=np.float32)
sim_retrace_filtered(_b, _a, _c, _sf, _v, 0.01, 44.0, 3.0, 20.0, 600, _mk, -999.0)
print("Numba ready.\n")

# ── Load & precompute all data ────────────────────────────────────────────────

pair_data = {}   # pair → {bid, ask, close, shock_flag, vel, timestamps, oos_days}
for pair in PAIRS:
    path = S5_DIR / f"{pair}_S5_BA.parquet"
    df   = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    df   = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    n_is = int(len(df) * IS_FRAC)
    oos  = df.iloc[n_is:].reset_index(drop=True)
    pip  = PIP[pair]
    close = oos["close"].values.astype(np.float64)
    bid   = oos["bid_c"].values.astype(np.float64)
    ask   = oos["ask_c"].values.astype(np.float64)
    z     = compute_shock_z(close, pip)
    sf    = (np.abs(z) > Z_THR).astype(np.int8)
    vel   = np.empty(len(close), dtype=np.float32)
    vel[:Z_WIN] = 0.0
    vel[Z_WIN:] = (close[Z_WIN:] - close[:len(close)-Z_WIN]) / pip
    oos_days  = len(oos) / 17280
    timestamps = oos["timestamp"].values
    pair_data[pair] = dict(bid=bid, ask=ask, close=close, shock_flag=sf,
                           vel=vel, timestamps=timestamps, oos_days=oos_days)
    print(f"{pair}: {n_is:,} IS + {len(oos):,} OOS S5 bars ({oos_days:.0f} trading days)  "
          f"shocks={sf.sum()}")
    del df, oos; gc.collect()

# ── Build Markov signals for all (pair, window, thr) ─────────────────────────

print("\nBuilding causal D1 Markov signals …")
markov_cache = {}   # (pair, mw, mt) → np.array per S5 OOS bar
for pair in PAIRS:
    for mw in MARKOV_WINDOWS:
        for mt in MARKOV_THRS:
            key = (pair, mw, mt)
            daily_sig  = build_markov_signals(pair, mw, mt)
            s5_markov  = map_s5_to_signal(pair_data[pair]["timestamps"], daily_sig)
            markov_cache[key] = s5_markov
            nonzero = (s5_markov != 0).mean() * 100
            print(f"  {pair} mw={mw} mt={mt:.3f}  "
                  f"signal_nonzero={nonzero:.1f}%  "
                  f"mean={s5_markov[s5_markov!=0].mean():+.3f}")

# ── Phase 3 sweep ─────────────────────────────────────────────────────────────

tg("🔄 Phase 3 sweep running: 2 windows × 3 thresholds × 4 signal levels × 4 pairs")
print(f"\n{'='*75}")
print("PHASE 3 SWEEP — Markov Filter on Retrace")
print(f"{'='*75}")
print(f"{'mw':>3} {'mt':>6} {'sig_thr':>7}  "
      f"{'ppd':>8}  {'WR%':>5}  {'n/d':>5}  {'n_tot':>5}")
print(f"{'─'*75}")

all_rows = []

for mw in MARKOV_WINDOWS:
    for mt in MARKOV_THRS:
        for sig_thresh in SIG_THRESHOLDS:
            pair_ppd  = []
            pair_days = []
            pair_pnls = []

            for pair in PAIRS:
                pd_   = pair_data[pair]
                mk    = markov_cache[(pair, mw, mt)]
                ppd, wr, n, evpd = eval_filtered(
                    pd_["bid"], pd_["ask"], pd_["close"],
                    pd_["shock_flag"], pd_["vel"], PIP[pair],
                    pd_["oos_days"], mk, sig_thresh,
                )
                pair_ppd.append(ppd)
                pair_days.append(pd_["oos_days"])

            port_ppd = sum(pair_ppd)
            row = dict(mw=mw, mt=mt, sig_thr=sig_thresh, ppd=port_ppd)
            all_rows.append(row)
            print(f"{mw:>3} {mt:>6.3f} {sig_thresh:>7.2f}  "
                  f"{port_ppd:>+8.1f}p")

# Baseline (no filter, using sig_thresh=-999)
print(f"\n{'─'*75}")
print("BASELINE (no Markov filter, sig_thresh=−∞):")
base_ppd_list = []
for pair in PAIRS:
    pd_ = pair_data[pair]
    mk  = np.zeros(len(pd_["bid"]), dtype=np.float32)
    ppd, wr, n, evpd = eval_filtered(
        pd_["bid"], pd_["ask"], pd_["close"],
        pd_["shock_flag"], pd_["vel"], PIP[pair],
        pd_["oos_days"], mk, -999.0,
    )
    base_ppd_list.append(ppd)
    print(f"  {pair}: ppd={ppd:+.1f}  WR={wr:.1f}%  n/d={evpd:.2f}")
base_port_ppd = sum(base_ppd_list)
print(f"  Portfolio: {base_port_ppd:+.1f} p/d")

df_sweep = pd.DataFrame(all_rows)
df_sweep["vs_base"] = df_sweep["ppd"] - base_port_ppd
df_sweep.to_csv(RESULTS / "markov_retrace_filter_sweep.csv", index=False)

# Best configs
best_rows = df_sweep.nlargest(5, "ppd")
print(f"\n{'='*75}")
print("TOP 5 FILTERED CONFIGS (portfolio ppd)")
print(f"{'='*75}")
print(best_rows[["mw","mt","sig_thr","ppd","vs_base"]].to_string(index=False,
      float_format="{:+.2f}".format))

best_row = best_rows.iloc[0]
best_mw   = int(best_row["mw"])
best_mt   = float(best_row["mt"])
best_st   = float(best_row["sig_thr"])
best_ppd  = float(best_row["ppd"])

tg(f"✅ Sweep done.\n"
   f"Baseline: {base_port_ppd:+.1f} p/d\n"
   f"Best filtered: {best_ppd:+.1f} p/d\n"
   f"  mw={best_mw} mt={best_mt:.3f} sig_thr={best_st:.2f}\n"
   f"Δ vs baseline: {best_ppd-base_port_ppd:+.1f} p/d\n"
   f"Running WF+MC validation …")

# ── WF validation on best config ─────────────────────────────────────────────

print(f"\n{'='*75}")
print(f"WALK-FORWARD VALIDATION — best config: mw={best_mw} mt={best_mt:.3f} sig_thr={best_st:.2f}")
print(f"{'='*75}")

wf_passes = 0
wf_results = []
pair_oos_pnls = []
pair_oos_days = []

for pair in PAIRS:
    pd_   = pair_data[pair]
    mk    = markov_cache[(pair, best_mw, best_mt)]
    n     = len(pd_["bid"])
    chunk = n // N_WF_CHUNKS

    pair_wf = []
    for c in range(N_WF_CHUNKS):
        lo = c * chunk
        hi = lo + chunk if c < N_WF_CHUNKS - 1 else n
        mk_c   = mk[lo:hi]
        days_c = (hi - lo) / 17280
        pnl_c, _ = sim_retrace_filtered(
            pd_["bid"][lo:hi], pd_["ask"][lo:hi], pd_["close"][lo:hi],
            pd_["shock_flag"][lo:hi], pd_["vel"][lo:hi],
            PIP[pair], float(PEAK_B), SD, TP, float(HORIZON),
            mk_c, best_st,
        )
        ppd_c = pnl_c.sum() / days_c if days_c > 0 else 0.0
        pass_c = 1 if ppd_c > 0 else 0
        wf_passes += pass_c
        pair_wf.append(ppd_c)
        print(f"  {pair} chunk {c+1}: ppd={ppd_c:+.1f}  {'✓' if pass_c else '✗'}")

    wf_results.append(pair_wf)

    # Full OOS pnl for MC
    ppd_oos, wr_oos, n_oos, evpd_oos, pnl_oos = eval_filtered(
        pd_["bid"], pd_["ask"], pd_["close"],
        pd_["shock_flag"], pd_["vel"], PIP[pair],
        pd_["oos_days"], mk, best_st, return_pnl=True,
    )
    pair_oos_pnls.append(pnl_oos)
    pair_oos_days.append(pd_["oos_days"])
    print(f"  {pair} OOS: ppd={ppd_oos:+.1f}  WR={wr_oos:.1f}%  n/d={evpd_oos:.2f}\n")

print(f"  WF pass: {wf_passes}/{N_WF_CHUNKS * len(PAIRS)}")

# ── MC validation ─────────────────────────────────────────────────────────────

print(f"\n{'='*75}")
print("MONTE CARLO VALIDATION (1000 sign-shuffles, portfolio)")
print(f"{'='*75}")

actual_ppd = sum(p.sum() / d for p, d in zip(pair_oos_pnls, pair_oos_days))
mc_p       = mc_pvalue(pair_oos_pnls, pair_oos_days, actual_ppd, N_MC)
print(f"  Actual portfolio ppd: {actual_ppd:+.1f}")
print(f"  MC p-value: {mc_p:.4f}  ({'✓ PASS' if mc_p < 0.05 else '✗ FAIL'})")

# ── Baseline WF for comparison ────────────────────────────────────────────────

print(f"\n{'='*75}")
print("BASELINE WF (no filter) for comparison")
print(f"{'='*75}")

base_wf_passes = 0
base_pair_pnls = []
for pair in PAIRS:
    pd_  = pair_data[pair]
    mk0  = np.zeros(len(pd_["bid"]), dtype=np.float32)
    n    = len(pd_["bid"])
    chunk = n // N_WF_CHUNKS
    for c in range(N_WF_CHUNKS):
        lo = c * chunk
        hi = lo + chunk if c < N_WF_CHUNKS - 1 else n
        mk_c   = mk0[lo:hi]
        days_c = (hi - lo) / 17280
        pnl_c, _ = sim_retrace_filtered(
            pd_["bid"][lo:hi], pd_["ask"][lo:hi], pd_["close"][lo:hi],
            pd_["shock_flag"][lo:hi], pd_["vel"][lo:hi],
            PIP[pair], float(PEAK_B), SD, TP, float(HORIZON),
            mk_c, -999.0,
        )
        ppd_c = pnl_c.sum() / days_c if days_c > 0 else 0.0
        base_wf_passes += 1 if ppd_c > 0 else 0
        print(f"  {pair} chunk {c+1}: ppd={ppd_c:+.1f}  {'✓' if ppd_c>0 else '✗'}")

    _, _, _, _, pnl_b = eval_filtered(
        pd_["bid"], pd_["ask"], pd_["close"],
        pd_["shock_flag"], pd_["vel"], PIP[pair],
        pd_["oos_days"], mk0, -999.0, return_pnl=True,
    )
    base_pair_pnls.append(pnl_b)

print(f"  Baseline WF pass: {base_wf_passes}/{N_WF_CHUNKS * len(PAIRS)}")
base_mc_ppd = sum(p.sum() / d for p, d in zip(base_pair_pnls, pair_oos_days))
base_mc_p   = mc_pvalue(base_pair_pnls, pair_oos_days, base_mc_ppd, N_MC)
print(f"  Baseline MC p-value: {base_mc_p:.4f}")

# ── Final summary ─────────────────────────────────────────────────────────────

verdict = "✅ IMPROVEMENT" if best_ppd > base_port_ppd and mc_p < 0.05 else (
          "⚠️ MIXED" if mc_p < 0.05 else "❌ NO IMPROVEMENT")

n_filtered_trades = sum(len(p) for p in pair_oos_pnls)
n_baseline_trades = sum(len(p) for p in base_pair_pnls)
trade_retention   = n_filtered_trades / max(n_baseline_trades, 1) * 100

summary = (
    f"📊 Markov×Retrace Phase 3 Complete\n\n"
    f"BASELINE (no filter):\n"
    f"  ppd={base_port_ppd:+.1f}p  WF={base_wf_passes}/12  mc_p={base_mc_p:.4f}\n"
    f"  trades={n_baseline_trades}\n\n"
    f"BEST FILTER (mw={best_mw} mt={best_mt:.3f} sig_thr={best_st:.2f}):\n"
    f"  ppd={best_ppd:+.1f}p  WF={wf_passes}/12  mc_p={mc_p:.4f}\n"
    f"  trades={n_filtered_trades} ({trade_retention:.0f}% of baseline)\n"
    f"  Δppd={best_ppd-base_port_ppd:+.1f}p\n\n"
    f"{verdict}"
)

print(f"\n{'='*75}")
print(summary)
print(f"{'='*75}")
tg(summary)
