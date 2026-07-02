#!/usr/bin/env python3
"""
Post-Shock Retrace Backtest v2 — Walk-Forward + Monte Carlo
============================================================
Hypothesis:
  After a momentum shock, price retraces counter-trend before settling.
  We detect the shock, wait for the peak to develop, then enter short (after
  upshock) or long (after downshock) via stop order.

MSP cross-scale xcorr finding: D3→D5 peak at ~22 bars (110s); sweep also
tests 10b (50s) and 44b (220s) to find the optimal observation window.

sd=0 fix vs v1:
  v1 filled at peak_ask (historical max) — not achievable live.
  v2: sd=0 → market fill at bid/ask at watch_start bar (honest market order).
  sd>0 → SELL STOP at peak_ask - sd*pip (achievable stop order).

Validation pipeline:
  1. Main sweep — full OOS (30% of data, never touched IS)
  2. Walk-Forward — re-run on 3 equal OOS sub-chunks; WF_score = positive chunks / 12
  3. Monte Carlo — sign-shuffle on portfolio P&L; p-value < 0.05 = significant

S5 data: GBP_JPY, USD_JPY, EUR_JPY, AUD_JPY (BA format)
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

PAIRS = ["GBP_JPY", "USD_JPY", "EUR_JPY", "AUD_JPY"]
PIP   = {"GBP_JPY": 0.01, "USD_JPY": 0.01, "EUR_JPY": 0.01, "AUD_JPY": 0.01}

PEAK_BARS_LIST = [10, 22, 44]
STOP_DISTS     = [0, 1, 2, 3, 5]
TP_DISTS       = [5, 8, 10, 15, 20]
THRESHOLDS     = [2.0, 2.5, 3.0]
HORIZON        = 600
Z_WINDOW       = 6
MAD_WIN        = 2048
IS_FRAC        = 0.70
WF_CHUNKS      = 3
N_MC           = 1000
TOP_N_WF       = 40    # run WF on top N positive configs
TOP_N_MC       = 20    # run MC on WF-passing configs
WF_PASS_THRESH = 10    # of 12 (4 pairs × 3 chunks) must be positive


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
    return z.astype(np.float64), vel.astype(np.float64)


# ── Numba simulator ─────────────────────────────────────────────────────────────

@njit
def sim_retrace(bid, ask, close, shock_flag, vel,
                pip, peak_bars, stop_pips, tp_pips, horizon):
    """
    For each shock at bar t:
      1. Direction d = sign(vel[t])
      2. Find peak in [t, t+peak_bars]:
           upshock   → peak_ask  = max ask
           downshock → peak_bid  = min bid
      3a. sd=0 (market entry): fill at bid[watch_start] (sell) or ask[watch_start] (buy)
      3b. sd>0 (stop entry):   SELL STOP at peak_ask - sd*pip (upshock)
                                BUY STOP  at peak_bid + sd*pip (downshock)
      4. TP = tp_pips from fill; horizon expire → close at market
    Returns: filled, tp_hit, pnl_pips, dir arrays
    """
    n        = len(close)
    pb_int   = int(peak_bars)    # Numba requires int for array indexing
    max_ev   = n // 10
    filled   = np.zeros(max_ev, dtype=np.int8)
    tp_hit   = np.zeros(max_ev, dtype=np.int8)
    pnl_out  = np.zeros(max_ev, dtype=np.float64)
    dir_out  = np.zeros(max_ev, dtype=np.int8)
    ev_count = 0
    cooldown = 0

    for t in range(Z_WINDOW, n - pb_int - int(horizon) - 2):
        if cooldown > 0:
            cooldown -= 1
            continue
        if shock_flag[t] != 1:
            continue

        d = np.int8(1) if vel[t] > 0 else np.int8(-1)

        # Find peak in [t, t+pb_int]
        peak_ask = ask[t]
        peak_bid = bid[t]
        for k in range(1, pb_int + 1):
            j = t + k
            if ask[j] > peak_ask:
                peak_ask = ask[j]
            if bid[j] < peak_bid:
                peak_bid = bid[j]

        sp = (ask[t] - bid[t]) / pip

        watch_start = t + pb_int + 1
        watch_end   = t + pb_int + int(horizon)
        if watch_start >= n or watch_end >= n:
            continue

        fld = 0; tp = 0
        fill_price = 0.0; pnl = 0.0
        tp_level = 0.0

        if stop_pips == 0.0:
            # ── Market entry at watch_start (honest live fill) ───────────────
            fld = 1
            if d == 1:
                fill_price = bid[watch_start]
            else:
                fill_price = ask[watch_start]
            tp_level = fill_price - tp_pips * pip * d
            # same-bar TP check
            if d == 1 and bid[watch_start] <= tp_level:
                tp = 1; pnl = tp_pips - sp
            elif d == -1 and ask[watch_start] >= tp_level:
                tp = 1; pnl = tp_pips - sp
            loop_start = watch_start + 1
        else:
            # ── Stop-entry ────────────────────────────────────────────────────
            if d == 1:
                entry = peak_ask - stop_pips * pip
            else:
                entry = peak_bid + stop_pips * pip
            tp_level = entry - tp_pips * pip * d
            loop_start = watch_start

        # Watch bars for fill (stop mode) then TP
        for j in range(loop_start, min(watch_end + 1, n - 1)):
            lo = bid[j]; hi = ask[j]

            if stop_pips > 0.0 and fld == 0:
                if d == 1 and lo <= entry:
                    fld = 1; fill_price = entry
                    if lo <= tp_level:
                        tp = 1; pnl = tp_pips - sp
                elif d == -1 and hi >= entry:
                    fld = 1; fill_price = entry
                    if hi >= tp_level:
                        tp = 1; pnl = tp_pips - sp

            if fld == 1 and tp == 0:
                if d == 1 and lo <= tp_level:
                    tp = 1; pnl = tp_pips - sp
                elif d == -1 and hi >= tp_level:
                    tp = 1; pnl = tp_pips - sp

            if fld == 1 and tp == 1:
                break

        # Honest market-close for filled + TP-miss
        if fld == 1 and tp == 0:
            end_j = min(watch_end, n - 1)
            if d == 1:
                pnl = (fill_price - bid[end_j]) / pip - sp
            else:
                pnl = (ask[end_j] - fill_price) / pip - sp
        elif fld == 0:
            pnl = 0.0

        if ev_count < max_ev:
            filled[ev_count]  = fld
            tp_hit[ev_count]  = tp
            pnl_out[ev_count] = pnl
            dir_out[ev_count] = d
            ev_count += 1

        cooldown = (pb_int + int(horizon)) // 2

    return (filled[:ev_count], tp_hit[:ev_count],
            pnl_out[:ev_count], dir_out[:ev_count])


# ── Monte Carlo sign-shuffle ────────────────────────────────────────────────────

def mc_pvalue(pair_pnls: list, pair_days: list, actual_ppd: float,
              n_mc: int = 1000) -> float:
    """Portfolio sign-shuffle: fraction of shuffles with ppd >= actual_ppd."""
    beat = 0
    arrs = [np.array(p, dtype=np.float64) for p in pair_pnls]
    for _ in range(n_mc):
        shuffled = sum(
            (a * np.where(np.random.random(len(a)) > 0.5, 1.0, -1.0)).sum() / d
            for a, d in zip(arrs, pair_days)
        )
        if shuffled >= actual_ppd:
            beat += 1
    return beat / n_mc


# ── Single-config result extractor ─────────────────────────────────────────────

def eval_config(bid, ask, close, shock_flag, vel, pip, pb, sd, tp, oos_days,
                return_pnl=False):
    fld, tph, pnl, drs = sim_retrace(
        bid, ask, close, shock_flag, vel, pip,
        float(pb), float(sd), float(tp), HORIZON)
    n = len(fld)
    if n == 0:
        if return_pnl:
            return None, np.zeros(0)
        return None
    n_fill = fld.sum()
    ppd    = pnl.sum() / oos_days
    row = dict(n_events=n,
               fill_rate=n_fill / n * 100,
               tp_rate=tph.sum() / (n_fill + 1e-9) * 100,
               total_pnl=pnl.sum(),
               ppd=ppd,
               ev_per_day=n / oos_days)
    if return_pnl:
        return row, pnl
    return row


# ── Warmup ─────────────────────────────────────────────────────────────────────

print("Warming up Numba …")
_b  = np.ones(3000, dtype=np.float64) * 214.0
_a  = _b + 0.03
_c  = _b + 0.015
_v  = np.zeros(3000, dtype=np.float64); _v[100]=1.2; _v[500]=-0.8
_sf = np.zeros(3000, dtype=np.int8);   _sf[100]=1; _sf[500]=1; _sf[1200]=1
sim_retrace(_b, _a, _c, _sf, _v, 0.01, 22, 3.0, 10.0, 120)
sim_retrace(_b, _a, _c, _sf, _v, 0.01, 22, 0.0, 10.0, 120)
print("Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Main sweep (full OOS)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("PHASE 1 — Full OOS sweep")
print("=" * 72)

all_results = []

for pair in PAIRS:
    path = S5_DIR / f"{pair}_S5_BA.parquet"
    print(f"\nLoading {pair} …")
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    n_is = int(len(df) * IS_FRAC)
    df   = df.iloc[n_is:].reset_index(drop=True)
    oos_days = len(df) / 17280
    print(f"  OOS bars: {len(df):,}  ({oos_days:.0f} trading days)")

    close = df["close"].values.astype(np.float64)
    bid   = df["bid_c"].values.astype(np.float64)
    ask   = df["ask_c"].values.astype(np.float64)
    pip   = PIP[pair]

    z, vel = compute_shock_z(close, pip)

    for thr in THRESHOLDS:
        shock_flag = (np.abs(z) > thr).astype(np.int8)
        n_ev = shock_flag.sum()
        print(f"  thr={thr}: {n_ev:,} shocks ({n_ev/len(df)*100:.2f}%)")
        for pb, sd, tp in product(PEAK_BARS_LIST, STOP_DISTS, TP_DISTS):
            row = eval_config(bid, ask, close, shock_flag, vel, pip,
                              pb, sd, tp, oos_days)
            if row:
                row.update(pair=pair, thr=thr, peak_bars=pb, sd=sd, tp=tp)
                all_results.append(row)

    del df, close, bid, ask, z, vel; gc.collect()

df_res = pd.DataFrame(all_results)
df_res.to_csv(RESULTS / "post_shock_retrace_v2_full.csv", index=False)

# Portfolio aggregate
agg = (df_res.groupby(["thr", "peak_bars", "sd", "tp"])
       .agg(n_events=("n_events", "sum"),
            ev_per_day=("ev_per_day", "sum"),
            ppd=("ppd", "sum"),
            fill_rate=("fill_rate", "mean"),
            tp_rate=("tp_rate", "mean"))
       .reset_index())

print(f"\nPhase 1 complete. {len(agg[agg['ppd']>0])} positive portfolio configs.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Walk-Forward on top positive configs
# ══════════════════════════════════════════════════════════════════════════════

pos_agg = agg[agg["ppd"] > 0].nlargest(TOP_N_WF, "ppd")
print(f"\n{'='*72}")
print(f"PHASE 2 — Walk-Forward ({WF_CHUNKS} OOS sub-chunks) on {len(pos_agg)} configs")
print(f"{'='*72}")

# Key: (thr, peak_bars, sd, tp) → dict[pair] → list of chunk ppds
wf_data = {}

for pair in PAIRS:
    path = S5_DIR / f"{pair}_S5_BA.parquet"
    print(f"\n  {pair} …")
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    n_is = int(len(df) * IS_FRAC)
    df   = df.iloc[n_is:].reset_index(drop=True)
    close = df["close"].values.astype(np.float64)
    bid   = df["bid_c"].values.astype(np.float64)
    ask   = df["ask_c"].values.astype(np.float64)
    pip   = PIP[pair]
    z, vel = compute_shock_z(close, pip)
    n = len(close)
    cs = n // WF_CHUNKS   # chunk size in bars

    for _, cfg in pos_agg.iterrows():
        thr, pb, sd, tp = cfg.thr, int(cfg.peak_bars), cfg.sd, cfg.tp
        key = (thr, pb, sd, tp)
        shock_flag = (np.abs(z) > thr).astype(np.int8)

        chunk_ppds = []
        for ch in range(WF_CHUNKS):
            start = ch * cs
            end   = (ch + 1) * cs if ch < WF_CHUNKS - 1 else n
            ch_days = (end - start) / 17280
            fld, tph, pnl, drs = sim_retrace(
                bid[start:end], ask[start:end], close[start:end],
                shock_flag[start:end], vel[start:end],
                pip, float(pb), float(sd), float(tp), HORIZON)
            chunk_ppds.append(pnl.sum() / ch_days if ch_days > 0 else 0.0)

        if key not in wf_data:
            wf_data[key] = {}
        wf_data[key][pair] = chunk_ppds

    del df, close, bid, ask, z, vel; gc.collect()

# Compute WF score for each config
wf_rows = []
for _, cfg in pos_agg.iterrows():
    key = (cfg.thr, int(cfg.peak_bars), cfg.sd, cfg.tp)
    pos_count = 0; total = 0
    for pair in PAIRS:
        if pair in wf_data.get(key, {}):
            for cp in wf_data[key][pair]:
                total += 1
                if cp > 0:
                    pos_count += 1
    wf_rows.append(dict(thr=cfg.thr, peak_bars=int(cfg.peak_bars),
                        sd=cfg.sd, tp=cfg.tp,
                        ppd=cfg.ppd, ev_per_day=cfg.ev_per_day,
                        fill_rate=cfg.fill_rate, tp_rate=cfg.tp_rate,
                        wf_score=pos_count, wf_total=total,
                        wf_pass=(pos_count >= WF_PASS_THRESH)))

wf_df = pd.DataFrame(wf_rows)
wf_df.to_csv(RESULTS / "post_shock_retrace_v2_wf.csv", index=False)

n_wf_pass = wf_df["wf_pass"].sum()
print(f"\n  WF threshold: {WF_PASS_THRESH}/{WF_CHUNKS*len(PAIRS)} chunks positive")
print(f"  Configs passing WF: {n_wf_pass} / {len(wf_df)}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Monte Carlo sign-shuffle for WF-passing configs
# ══════════════════════════════════════════════════════════════════════════════

mc_cfgs = wf_df[wf_df["wf_pass"]].nlargest(TOP_N_MC, "ppd")

print(f"\n{'='*72}")
print(f"PHASE 3 — Monte Carlo ({N_MC} shuffles) on {len(mc_cfgs)} WF-passing configs")
print(f"{'='*72}")

if len(mc_cfgs) == 0:
    print("  No WF-passing configs. Hypothesis REJECTED.")
    mc_df = pd.DataFrame()
else:
    # Collect per-pair pnl arrays for MC configs
    mc_pnls  = {(r.thr, int(r.peak_bars), r.sd, r.tp): {"arrs": [], "days": []}
                for _, r in mc_cfgs.iterrows()}

    for pair in PAIRS:
        path = S5_DIR / f"{pair}_S5_BA.parquet"
        print(f"\n  {pair} …")
        df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
        df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
        n_is = int(len(df) * IS_FRAC)
        df   = df.iloc[n_is:].reset_index(drop=True)
        close = df["close"].values.astype(np.float64)
        bid   = df["bid_c"].values.astype(np.float64)
        ask   = df["ask_c"].values.astype(np.float64)
        pip   = PIP[pair]
        oos_days = len(df) / 17280
        z, vel = compute_shock_z(close, pip)

        for _, cfg in mc_cfgs.iterrows():
            key = (cfg.thr, int(cfg.peak_bars), cfg.sd, cfg.tp)
            shock_flag = (np.abs(z) > cfg.thr).astype(np.int8)
            row, pnl = eval_config(bid, ask, close, shock_flag, vel, pip,
                                   int(cfg.peak_bars), cfg.sd, cfg.tp, oos_days,
                                   return_pnl=True)
            mc_pnls[key]["arrs"].append(pnl)
            mc_pnls[key]["days"].append(oos_days)

        del df, close, bid, ask, z, vel; gc.collect()

    # Compute p-values
    mc_rows = []
    print("\n  Computing p-values …")
    for _, cfg in mc_cfgs.iterrows():
        key = (cfg.thr, int(cfg.peak_bars), cfg.sd, cfg.tp)
        arrs = mc_pnls[key]["arrs"]
        days = mc_pnls[key]["days"]
        actual_ppd = sum(a.sum() / d for a, d in zip(arrs, days))
        p_val = mc_pvalue(arrs, days, actual_ppd, N_MC)
        mc_rows.append(dict(thr=cfg.thr, peak_bars=int(cfg.peak_bars),
                            sd=cfg.sd, tp=cfg.tp,
                            ppd=actual_ppd,
                            wf_score=cfg.wf_score,
                            mc_p=round(p_val, 4)))
        print(f"    thr={cfg.thr} peak={int(cfg.peak_bars)}b "
              f"sd={cfg.sd:.0f}p tp={cfg.tp:.0f}p → "
              f"ppd={actual_ppd:+.1f} mc_p={p_val:.4f}")

    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(RESULTS / "post_shock_retrace_v2_mc.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*72}")
print("PHASE 1  — Top 20 by portfolio p/d (all positive configs)")
print(f"{'='*72}")
top20 = agg[agg["ppd"] > 0].nlargest(20, "ppd")
print(f"  {'thr':>4}  {'peak':>5}  {'sd':>3}  {'tp':>3}  "
      f"{'ev/d':>5}  {'fill%':>5}  {'tp%':>5}  {'ppd':>8}")
for _, r in top20.iterrows():
    print(f"  {r['thr']:>4.1f}  {int(r['peak_bars']):>4}b  "
          f"{r['sd']:>3.0f}p  {r['tp']:>3.0f}p  "
          f"{r['ev_per_day']:>5.1f}  {r['fill_rate']:>5.1f}%  "
          f"{r['tp_rate']:>5.1f}%  {r['ppd']:>+8.1f}p")

print(f"\n{'='*72}")
print(f"PHASE 2  — Walk-Forward results (top {TOP_N_WF} positive configs)")
print(f"{'='*72}")
print(f"  WF pass = {WF_PASS_THRESH}+ of {WF_CHUNKS*len(PAIRS)} (pair×chunk) windows positive")
print(f"\n  {'thr':>4}  {'peak':>5}  {'sd':>3}  {'tp':>3}  "
      f"{'ppd':>8}  {'WF':>5}  {'PASS':>5}")
for _, r in wf_df.sort_values("ppd", ascending=False).iterrows():
    mk = " ✓" if r["wf_pass"] else ""
    print(f"  {r['thr']:>4.1f}  {int(r['peak_bars']):>4}b  "
          f"{r['sd']:>3.0f}p  {r['tp']:>3.0f}p  "
          f"{r['ppd']:>+8.1f}p  {int(r['wf_score'])}/{int(r['wf_total'])}  "
          f"{'YES':>5}{mk}")

if len(mc_df) > 0:
    print(f"\n{'='*72}")
    print("PHASE 3  — Monte Carlo sign-shuffle (WF-passing configs)")
    print(f"{'='*72}")
    print(f"  mc_p < 0.05 = significant (direction is not random noise)")
    print(f"\n  {'thr':>4}  {'peak':>5}  {'sd':>3}  {'tp':>3}  "
          f"{'ppd':>8}  {'WF':>5}  {'mc_p':>6}  {'SIG':>4}")
    for _, r in mc_df.sort_values("ppd", ascending=False).iterrows():
        sig = "✓" if r["mc_p"] < 0.05 else " "
        print(f"  {r['thr']:>4.1f}  {int(r['peak_bars']):>4}b  "
              f"{r['sd']:>3.0f}p  {r['tp']:>3.0f}p  "
              f"{r['ppd']:>+8.1f}p  {int(r['wf_score'])}/12  "
              f"{r['mc_p']:>6.4f}  {sig:>4}")

    final = mc_df[mc_df["mc_p"] < 0.05].sort_values("ppd", ascending=False)
    print(f"\n{'='*72}")
    print(f"DEPLOYABLE CONFIGS (WF_pass + mc_p<0.05): {len(final)}")
    print(f"{'='*72}")
    if len(final) == 0:
        print("  None — retrace hypothesis NOT confirmed after full validation")
    else:
        for _, r in final.iterrows():
            print(f"  🟢 thr={r['thr']:.1f} peak={int(r['peak_bars'])}b "
                  f"sd={r['sd']:.0f}p tp={r['tp']:.0f}p → "
                  f"ppd={r['ppd']:+.1f}  WF={int(r['wf_score'])}/12  mc_p={r['mc_p']:.4f}")
else:
    print("\n  No WF-passing configs — hypothesis REJECTED at WF stage")

print(f"\nResults saved → {RESULTS}/post_shock_retrace_v2_*.csv")
