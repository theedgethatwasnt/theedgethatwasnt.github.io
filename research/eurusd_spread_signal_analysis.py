"""
EUR/USD — Signal frequency, spread dynamics, exploitability.

Answers three questions from Session 049:
1. How often does the pre-big-bar z_tpm signal fire?
2. Spread profile T-25min → T+10min around big M5 events
3. Net exploitability after real spread costs
4. Does rolling body ratio (C-O)/(H-L) of window add predictive power?

Uses: data/s30_ohlc/EUR_USD_S30_BA.parquet
      data/m5_ba/EUR_USD_M5_BA.parquet   (for big-bar ground truth + spread)
"""

import numpy as np
import pandas as pd
from numba import njit
import warnings, gc
warnings.filterwarnings("ignore")

PIP = 0.0001
SIGMA = 3.0


# ── helpers ────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data …")
    s30 = pd.read_parquet("data/s30_ohlc/EUR_USD_S30_BA.parquet")
    m5  = pd.read_parquet("data/m5_ba/EUR_USD_M5_BA.parquet")
    s30 = s30.sort_values("timestamp").reset_index(drop=True)
    m5  = m5.sort_values("timestamp").reset_index(drop=True)
    # common window
    t0 = max(s30["timestamp"].iloc[0], m5["timestamp"].iloc[0])
    t1 = min(s30["timestamp"].iloc[-1], m5["timestamp"].iloc[-1])
    s30 = s30[(s30["timestamp"] >= t0) & (s30["timestamp"] <= t1)].reset_index(drop=True)
    m5  = m5[(m5["timestamp"]  >= t0) & (m5["timestamp"]  <= t1)].reset_index(drop=True)
    print(f"  S30: {len(s30):,}  M5: {len(m5):,}  window: {t0.date()} → {t1.date()}")
    return s30, m5


def compute_tr_pips(df):
    """True range in pips."""
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    prev_c = np.empty_like(c); prev_c[0] = c[0]; prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return tr / PIP


def big_bar_threshold(tr_pips, sigma=SIGMA):
    mu = np.mean(tr_pips); sd = np.std(tr_pips)
    return mu + sigma * sd


def rolling_micro(close, volume, win, tf_seconds):
    """Vectorised ppm / tpm / ppt rolling sums → z-scores."""
    n = len(close)
    pips = np.abs(np.diff(close.astype(np.float64), prepend=close[0])) / PIP
    tpm_raw = volume / (tf_seconds / 60.0)
    ppm_raw = pips   / (tf_seconds / 60.0)
    ppt_raw = np.where(volume > 0, pips / volume, 0.0)

    def roll_z(x, w, ref_w):
        cs = np.cumsum(np.concatenate([[0.0], x]))
        rolling = (cs[w:] - cs[:-w]) / w
        ref = (cs[ref_w:] - cs[:-ref_w]) / ref_w
        padded_ref = np.concatenate([np.full(ref_w - w, np.nan), ref])
        m = np.nanmean(padded_ref[:len(rolling)])
        s = np.nanstd(padded_ref[:len(rolling)])
        return (rolling - m) / (s + 1e-12)

    ref = min(4 * win, n // 2)
    z_tpm = roll_z(tpm_raw, win, ref)
    z_ppt = roll_z(ppt_raw, win, ref)
    # pad to length n
    pad = n - len(z_tpm)
    z_tpm = np.concatenate([np.full(pad, np.nan), z_tpm])
    z_ppt = np.concatenate([np.full(pad, np.nan), z_ppt])
    return z_tpm, z_ppt


def rolling_body_ratio(df, win):
    """Rolling mean of signed body ratio (C-O)/(H-L) over window."""
    hl = df["high"].values.astype(np.float64) - df["low"].values.astype(np.float64)
    co = df["close"].values.astype(np.float64) - df["open"].values.astype(np.float64)
    ratio = np.where(hl > 0, co / hl, 0.0)
    cs = np.cumsum(np.concatenate([[0.0], ratio]))
    roll = (cs[win:] - cs[:-win]) / win
    return np.concatenate([np.full(win - 1, np.nan), roll])


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Signal frequency
# ══════════════════════════════════════════════════════════════════════════════

def signal_frequency(s30):
    print("\n" + "═"*68)
    print("  PART 1 — SIGNAL FREQUENCY (pre-big-bar z_tpm)")
    print("═"*68)

    WIN = 20; TF = 30
    z_tpm, z_ppt = rolling_micro(s30["close"].values, s30["volume"].values, WIN, TF)
    s30_ts = s30["timestamp"].values

    # Spread in pips
    sp = (s30["ask_c"].values.astype(np.float64) - s30["bid_c"].values.astype(np.float64)) / PIP

    n = len(z_tpm)
    days = (pd.Timestamp(s30_ts[-1]) - pd.Timestamp(s30_ts[0])).days
    trading_days = days * 5 / 7  # approx

    print(f"\n  Data span: {days} calendar days ≈ {trading_days:.0f} trading days")
    print(f"  S30 bars: {n:,}   avg spread: {np.nanmean(sp):.2f}p   P90 spread: {np.nanpercentile(sp, 90):.2f}p")

    # Signal definition: z_tpm > threshold for >= consec consecutive bars
    thresholds = [0.3, 0.5, 0.7, 1.0]
    consec_list = [5, 10, 15, 20]

    print(f"\n  Signal fires = first bar where z_tpm > Z for >= C consecutive bars")
    print(f"  {'Z':>5}  {'C':>5}  {'fires':>7}  {'fires/day':>10}  {'avg_sp_at_entry':>17}")
    print("  " + "-"*55)

    for Z in thresholds:
        above = (z_tpm > Z).astype(np.int8)
        for C in consec_list:
            # Rolling count of consecutive trues — detect rising edge
            fires = []
            run = 0
            for i in range(n):
                if np.isnan(z_tpm[i]):
                    run = 0; continue
                if above[i]:
                    run += 1
                    if run == C:
                        fires.append(i)
                else:
                    run = 0
            nf = len(fires)
            fpd = nf / trading_days if trading_days > 0 else 0
            sp_at = np.mean([sp[f] for f in fires]) if fires else np.nan
            print(f"  {Z:>5.1f}  {C:>5}  {nf:>7}  {fpd:>10.1f}  {sp_at:>17.2f}p")
        print()

    # Best candidate: Z=0.5, C=10 — compute time-of-day breakdown
    Z, C = 0.5, 10
    above = (z_tpm > Z).astype(np.int8)
    fires_idx = []
    run = 0
    for i in range(n):
        if np.isnan(z_tpm[i]): run = 0; continue
        if above[i]:
            run += 1
            if run == C: fires_idx.append(i)
        else:
            run = 0

    if fires_idx:
        fire_hours = pd.DatetimeIndex([pd.Timestamp(s30_ts[i]) for i in fires_idx]).hour
        print(f"\n  Z={Z} C={C}: {len(fires_idx)} signals across {trading_days:.0f} trading days = {len(fires_idx)/trading_days:.1f}/day")
        print("  Hour-of-day distribution (UTC):")
        for h in range(24):
            cnt = np.sum(fire_hours == h)
            if cnt > 0:
                bar = "█" * (cnt // max(1, len(fires_idx)//80))
                print(f"    {h:02d}:00  {cnt:4d}  {bar}")

    return fires_idx, z_tpm, z_ppt, sp


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Spread profile around big M5 events
# ══════════════════════════════════════════════════════════════════════════════

def spread_dynamics(s30, m5):
    print("\n" + "═"*68)
    print("  PART 2 — SPREAD DYNAMICS: T-50 → T+20 S30 bars around big M5")
    print("═"*68)

    # M5 spread + TR
    m5_sp = ((m5["ask_c"].values.astype(np.float64) - m5["bid_c"].values.astype(np.float64)) / PIP)
    m5_tr = compute_tr_pips(m5)
    thresh_m5 = big_bar_threshold(m5_tr)
    big_m5 = m5_tr >= thresh_m5

    print(f"\n  M5 big-bar threshold: {thresh_m5:.2f}p")
    print(f"  M5 big-bar events: {big_m5.sum()} ({100*big_m5.mean():.2f}%)")
    print(f"  Avg M5 spread: {np.mean(m5_sp):.2f}p   P90: {np.percentile(m5_sp, 90):.2f}p")

    # S30 spread
    s30_sp = ((s30["ask_c"].values.astype(np.float64) - s30["bid_c"].values.astype(np.float64)) / PIP)
    s30_ts = s30["timestamp"].values.astype(np.int64)
    m5_ts  = m5["timestamp"].values.astype(np.int64)

    # Map each M5 big bar → nearest S30 bar index
    m5_big_idx = np.where(big_m5)[0]
    big_m5_ts  = m5_ts[m5_big_idx]
    # searchsorted to find S30 index
    s30_pos = np.searchsorted(s30_ts, big_m5_ts, side="right") - 1
    valid = (s30_pos >= 50) & (s30_pos < len(s30_ts) - 20)
    s30_pos = s30_pos[valid]
    print(f"  Events with full S30 context: {len(s30_pos)}")

    BACK, FWD = 50, 20
    window = BACK + FWD + 1

    sp_matrix = np.full((len(s30_pos), window), np.nan)
    for k, pos in enumerate(s30_pos):
        sp_matrix[k] = s30_sp[pos - BACK: pos + FWD + 1]

    mean_sp = np.nanmean(sp_matrix, axis=0)
    baseline_sp = np.nanmean(sp_matrix[:, :20])  # first 20 bars = far background

    print(f"\n  Mean spread profile around big M5 bars (S30 bars, lag 0 = bar containing M5 open)")
    print(f"  Baseline spread (lag -50 to -31): {baseline_sp:.3f}p")
    print()
    print(f"  {'Lag':>5}  {'Spread(p)':>10}  {'vs_baseline':>12}  Bar")
    print("  " + "-"*50)

    key_lags = list(range(-50, -29, 5)) + list(range(-30, -9, 2)) + list(range(-10, 21, 1))
    for lag in key_lags:
        idx = BACK + lag
        sp_val = mean_sp[idx]
        delta = sp_val - baseline_sp
        bar_len = int(abs(delta) / 0.01)
        bar = ("▲" if delta > 0 else "▼") * min(bar_len, 20)
        print(f"  {lag:>5}  {sp_val:>10.3f}  {delta:>+12.3f}  {bar}")

    # peak spread
    peak_idx = np.argmax(mean_sp[BACK-10:BACK+5]) + (BACK - 10)
    peak_lag = peak_idx - BACK
    print(f"\n  Peak spread: lag {peak_lag:+d}, value {mean_sp[peak_idx]:.3f}p (Δ={mean_sp[peak_idx]-baseline_sp:+.3f}p vs baseline)")

    # Spread at key points
    lag_map = {"-25": mean_sp[BACK-25], "-15": mean_sp[BACK-15],
               "-5":  mean_sp[BACK-5],  "0":   mean_sp[BACK],
               "+5":  mean_sp[BACK+5],  "+10": mean_sp[BACK+10]}
    print("\n  Spread at key sequence points:")
    for lbl, val in lag_map.items():
        print(f"    lag {lbl:>4}: {val:.3f}p  (Δ{val-baseline_sp:+.3f}p vs baseline)")

    return baseline_sp, lag_map, mean_sp, BACK


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Exploitability with real spread costs
# ══════════════════════════════════════════════════════════════════════════════

def exploitability(fires_idx, s30, m5, baseline_sp, lag_map, z_tpm):
    print("\n" + "═"*68)
    print("  PART 3 — EXPLOITABILITY WITH REAL SPREAD COSTS")
    print("═"*68)

    if not fires_idx:
        print("  No signal fires to analyze."); return

    s30_ts = s30["timestamp"].values.astype(np.int64)
    m5_ts  = m5["timestamp"].values.astype(np.int64)
    m5_cl  = m5["close"].values.astype(np.float64)
    m5_op  = m5["open"].values.astype(np.float64)
    m5_sp  = ((m5["ask_c"].values.astype(np.float64) - m5["bid_c"].values.astype(np.float64)) / PIP)
    m5_tr  = compute_tr_pips(m5)
    thresh_m5 = big_bar_threshold(m5_tr)
    big_m5 = m5_tr >= thresh_m5

    s30_cl = s30["close"].values.astype(np.float64)
    s30_sp_arr = ((s30["ask_c"].values.astype(np.float64) - s30["bid_c"].values.astype(np.float64)) / PIP)

    # For each signal fire: find M5 bar that contains or follows the S30 signal bar
    # Entry: open of next M5 bar after signal fires
    # Exit:  close of M5 bar at lag +1, +2, +3, +5
    # Direction: sign of recent S30 price movement (last 5 S30 bars)

    lags = [1, 2, 3, 5, 8]
    raw_rets  = {l: [] for l in lags}
    gross_rets = {l: [] for l in lags}
    net_rets  = {l: [] for l in lags}
    n_big_hit = {l: [] for l in lags}  # was there a big M5 in the hold period?
    spread_entry_list = []
    spread_exit_list  = []

    for si in fires_idx:
        fire_ts = s30_ts[si]
        entry_sp = s30_sp_arr[si]

        # Find the M5 bar that starts at or after fire_ts
        m5_pos = np.searchsorted(m5_ts, fire_ts, side="right")  # first M5 bar strictly after signal
        if m5_pos >= len(m5_ts) - max(lags) - 1:
            continue

        # Direction: sign of S30 close change over last 5 bars
        if si < 5:
            continue
        s30_move = s30_cl[si] - s30_cl[si - 5]
        direction = 1 if s30_move >= 0 else -1

        entry_price = m5_op[m5_pos]  # enter at open of first M5 after signal
        exit_sp_entry = m5_sp[m5_pos]
        spread_entry_list.append(exit_sp_entry)

        for lag in lags:
            exit_idx = m5_pos + lag
            if exit_idx >= len(m5_cl):
                continue
            exit_price = m5_cl[exit_idx]
            exit_sp = m5_sp[exit_idx]
            spread_exit_list.append(exit_sp)

            raw_pips = direction * (exit_price - entry_price) / PIP
            total_cost = (entry_sp + exit_sp) / 2  # approximate round-trip as avg entry+exit half-spread each side
            # Actually: cost = half-spread at entry + half-spread at exit = (entry_sp + exit_sp) / 2
            net_pips = raw_pips - total_cost

            raw_rets[lag].append(raw_pips)
            net_rets[lag].append(net_pips)
            n_big = np.any(big_m5[m5_pos:exit_idx+1])
            n_big_hit[lag].append(int(n_big))

    n_signals = len(fires_idx)
    days = (pd.Timestamp(s30_ts[-1]) - pd.Timestamp(s30_ts[0])).days
    trading_days = days * 5 / 7

    print(f"\n  Signal: z_tpm(W=20) > 0.5 for 10+ consecutive S30 bars")
    print(f"  n_signals = {n_signals}  ({n_signals/trading_days:.1f}/trading day)")
    print(f"  Direction: sign of 5-bar S30 close change")
    print(f"  Avg spread at entry: {np.mean(spread_entry_list):.2f}p  P90: {np.percentile(spread_entry_list, 90):.2f}p")

    print(f"\n  Forward return analysis (enter at M5 open, exit at M5 close after lag bars):")
    print(f"  {'lag':>5}  {'n':>6}  {'raw_p':>8}  {'net_p':>8}  {'win%':>7}  {'net_win%':>9}  {'p_bigbar':>9}")
    print("  " + "-"*65)

    for lag in lags:
        rr = np.array(raw_rets[lag])
        nr = np.array(net_rets[lag])
        nb = np.array(n_big_hit[lag])
        if len(rr) == 0:
            continue
        print(f"  {lag:>5}  {len(rr):>6}  {np.mean(rr):>+8.2f}  {np.mean(nr):>+8.2f}  "
              f"{100*np.mean(rr>0):>7.1f}%  {100*np.mean(nr>0):>9.1f}%  {100*np.mean(nb):>9.1f}%")

    # Expected daily P&L at lag=3
    lag = 3
    nr = np.array(net_rets[lag])
    spd = n_signals / trading_days
    edp = np.mean(nr) * spd
    print(f"\n  Expected daily P&L (lag={lag}): {np.mean(nr):+.2f}p/trade × {spd:.1f} signals/day = {edp:+.1f}p/day")

    # Spread sensitivity analysis
    print(f"\n  Spread sensitivity (lag={lag}, raw={np.mean(raw_rets[lag]):+.2f}p/trade):")
    for sp_cost in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        net = np.mean(raw_rets[lag]) - sp_cost
        daily = net * spd
        flag = "🟢" if daily > 0 else "🔴"
        print(f"    spread {sp_cost:.1f}p → net {net:+.2f}p/trade → {daily:+.1f}p/day  {flag}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Rolling body ratio as predictor
# ══════════════════════════════════════════════════════════════════════════════

def body_ratio_predictor(s30, m5):
    print("\n" + "═"*68)
    print("  PART 4 — ROLLING BODY RATIO (C-O)/(H-L) AS PREDICTOR")
    print("═"*68)

    m5_tr = compute_tr_pips(m5)
    thresh_m5 = big_bar_threshold(m5_tr)
    big_m5 = (m5_tr >= thresh_m5).astype(np.float64)

    s30_ts = s30["timestamp"].values.astype(np.int64)
    m5_ts  = m5["timestamp"].values.astype(np.int64)

    # Align S30 → M5
    pos = np.searchsorted(m5_ts, s30_ts, side="right") - 1
    valid = (pos >= 0) & (pos < len(m5_ts))

    results = []
    for win in [5, 10, 20, 40]:
        rbr = rolling_body_ratio(s30, win)

        # Forward big_m5 in next 1 and 3 M5 bars
        for lag in [1, 3]:
            hits1, absr, n = [], [], 0
            for i in range(len(s30_ts)):
                if not valid[i] or np.isnan(rbr[i]):
                    continue
                mp = pos[i]
                if mp + lag >= len(big_m5):
                    continue
                hit = np.any(big_m5[mp+1:mp+lag+1])
                hits1.append(hit)
                absr.append(abs(rbr[i]))

            hits1 = np.array(hits1); absr = np.array(absr)
            # Quintile analysis
            q = np.percentile(absr, [20, 40, 60, 80])
            masks = [absr <= q[0],
                     (absr > q[0]) & (absr <= q[1]),
                     (absr > q[1]) & (absr <= q[2]),
                     (absr > q[2]) & (absr <= q[3]),
                     absr > q[3]]
            qs = [f"Q{k+1}:{100*np.mean(hits1[m]):.1f}%" for k, m in enumerate(masks) if m.sum() > 10]
            results.append((win, lag, qs))

    print("\n  |body_ratio| quintile → P(big M5 in next N bars)")
    print("  (Q1=flattest candles, Q5=strongest marubozu candles in rolling window)")
    print()
    for win, lag, qs in results:
        print(f"  W={win:>3}  lag=+{lag}M5:  {' | '.join(qs)}")

    # Also: signed body ratio direction agreement
    print("\n  Signed body ratio (directional agreement) effect:")
    print("  Stratified by: strong_bull (rbr>0.3), neutral (-0.3..0.3), strong_bear (rbr<-0.3)")
    WIN = 20
    rbr = rolling_body_ratio(s30, WIN)
    fwd_rets = []
    for i in range(len(s30_ts)):
        if not valid[i] or np.isnan(rbr[i]):
            continue
        mp = pos[i]
        if mp + 3 >= len(m5["close"].values):
            continue
        m5_cl = m5["close"].values.astype(np.float64)
        ret = (m5_cl[mp+3] - m5_cl[mp]) / PIP
        fwd_rets.append((rbr[i], ret))

    if fwd_rets:
        rbr_arr = np.array([x[0] for x in fwd_rets])
        ret_arr = np.array([x[1] for x in fwd_rets])
        for lbl, mask in [("strong_bull rbr>0.3",  rbr_arr > 0.3),
                           ("neutral   |rbr|<0.3",  np.abs(rbr_arr) < 0.3),
                           ("strong_bear rbr<-0.3", rbr_arr < -0.3)]:
            n = mask.sum()
            if n < 10: continue
            print(f"  {lbl}: n={n:>7}  fwd+3M5 = {np.mean(ret_arr[mask]):+.2f}p  win%={100*np.mean(ret_arr[mask]>0):.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  EUR/USD — Spread, Signal Frequency & Exploitability")
    print("=" * 68)

    s30, m5 = load_data()

    fires_idx, z_tpm, z_ppt, sp = signal_frequency(s30)
    baseline_sp, lag_map, mean_sp, BACK = spread_dynamics(s30, m5)
    exploitability(fires_idx, s30, m5, baseline_sp, lag_map, z_tpm)
    body_ratio_predictor(s30, m5)

    print("\n" + "═"*68)
    print("  DONE")
    print("═"*68)


if __name__ == "__main__":
    main()
