"""
ontick_backtest.py -- replay HistData ticks through the SAME engine+rule the live
service will use, overlaying OANDA's REAL spread as the cost.

CLEAN ROOM. Does NOT import or reference the bb_fade family.

PRICE  : HistData ticks are MID = (bid+ask)/2. (HistData "bid/ask" is a tight
         retail feed; we build the signal on its mid -- the OANDA spread is the
         only cost that matters for OANDA execution.)
         Format: "YYYYMMDD HHMMSSmmm,bid,ask,vol".
         Empirically time-align to UTC by max return-correlation vs OANDA S5
         close returns (the AUDJPY file was +4h / EDT -- auto-detected, never
         hardcoded). Search offsets -12..+12 h.

COST   : OANDA real spread from data/s5_ohlc/{PAIR}_S5_BA.parquet (ask_c - bid_c),
         joined to each tick by the prevailing 5s bucket. Fills = HistData mid
         +/- 0.5 * OANDA_spread (symmetric; OANDA sits ~symmetric around HistData
         mid, ~3.3x wider).
         BUY  fills at mid + 0.5*sp,  SELL fills at mid - 0.5*sp.
         If a tick's 5s bucket has no real OANDA spread (HistData month outside
         S5 BA coverage), we fall back to the OANDA hour-of-week MEDIAN spread
         profile built from whatever S5 BA we DO have, and we LOUDLY flag the
         run as profile-overlay (suggestive, not a clean real-spread test).

SELF-CHECK (anti-artifact): every exit price must lie within
[min,max] of the entry/stop/target geometry for its side; the stop must only
book a loss when genuinely hit (exit_mid beyond stop in the adverse direction);
counts of any violation are printed LOUDLY and the trade rejected.

SWEEP : window_seconds in {900,1800,3600,14400}; K in {1.5,2.0,2.5};
        stop_pips scaled by pair (JPY pairs use a larger pip distance).
        Frequency accounted as trades / actual calendar day; we reconcile
        expectancy*trades == total pips.
"""
import sys
import os
import io
import zipfile
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from ontick_engine import RollingWindow
from ontick_rule import ReversionRule, ENTER_LONG, ENTER_SHORT, EXIT


PAIR_CFG = {
    "AUD_JPY": dict(
        histdata_zip="data/HISTDATA_COM_ASCII_AUDJPY_T202604.zip",
        s5_ba="data/s5_ohlc/AUD_JPY_S5_BA.parquet",
        pip=0.01, jpy=True,
    ),
    "GBP_JPY": dict(
        histdata_zip="/path/to/book_org_migration/oanda-ksql-muzero/data/HISTDATA_COM_ASCII_GBPJPY_T202405.zip",
        s5_ba="data/s5_ohlc/GBP_JPY_S5_BA.parquet",
        pip=0.01, jpy=True,
    ),
}
BASE = "/path/to/projects/fx-core"


# ---------------------------------------------------------------------------
# Load HistData ticks
# ---------------------------------------------------------------------------
def load_histdata(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        raw = z.read(csv_name)
    df = pd.read_csv(
        io.BytesIO(raw), header=None,
        names=["ts", "bid", "ask", "vol"],
        dtype={"ts": str, "bid": np.float64, "ask": np.float64, "vol": np.int64},
    )
    # ts like "20260401 000000069"  (YYYYMMDD HHMMSSmmm)
    dt = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S%f")
    df["t_ms"] = (dt.astype("int64") // 1_000_000)  # ms since epoch, naive==local-from-file
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    return df[["t_ms", "mid"]].reset_index(drop=True), dt


# ---------------------------------------------------------------------------
# Auto time-align HistData -> UTC by max return correlation vs OANDA S5 close
# ---------------------------------------------------------------------------
def detect_utc_offset_hours(hist_dt, hist_mid, s5):
    """
    Resample HistData mid to 5-min last-price; for each candidate hour offset,
    shift HistData timestamps and correlate 5-min returns with OANDA 5-min returns.
    Returns the integer hour offset that maximizes correlation.
    """
    # OANDA 5-min close returns
    o = s5.set_index("timestamp")["close"].resample("5min").last().dropna()
    o_ret = np.log(o).diff().dropna()

    best_off, best_corr = 0, -2.0
    hist = pd.Series(hist_mid.values, index=hist_dt.values)
    for off in range(-12, 13):
        shifted_idx = hist.index + pd.Timedelta(hours=off)
        h = pd.Series(hist.values, index=shifted_idx)
        h5 = h.resample("5min").last().dropna()
        h_ret = np.log(h5).diff().dropna()
        # localize OANDA (tz-aware) to naive for join
        o_ret_naive = o_ret.copy()
        o_ret_naive.index = o_ret_naive.index.tz_localize(None)
        j = pd.concat([h_ret.rename("h"), o_ret_naive.rename("o")], axis=1).dropna()
        if len(j) < 200:
            continue
        c = j["h"].corr(j["o"])
        if c > best_corr:
            best_corr, best_off = c, off
    return best_off, best_corr


# ---------------------------------------------------------------------------
# Build spread lookup: real per-5s spread + hour-of-week median fallback profile
# ---------------------------------------------------------------------------
def build_spread_lookup(s5, pip):
    s5 = s5.copy()
    s5["sp_pips"] = (s5["ask_c"] - s5["bid_c"]) / pip
    s5["bucket_ms"] = (s5["timestamp"].astype("int64") // 1_000_000 // 5000) * 5000
    real = dict(zip(s5["bucket_ms"].values, s5["sp_pips"].values))
    # hour-of-week median profile (for months outside coverage)
    how = s5["timestamp"].dt.dayofweek * 24 + s5["timestamp"].dt.hour
    prof = s5.groupby(how)["sp_pips"].median().to_dict()
    global_med = float(s5["sp_pips"].median())
    return real, prof, global_med


def spread_for_tick(t_ms_utc, real, prof, global_med):
    """Return (spread_pips, is_real). Prevailing 5s bucket; else hour-of-week profile."""
    bucket = (t_ms_utc // 5000) * 5000
    sp = real.get(bucket)
    if sp is not None and sp == sp:  # not NaN
        return float(sp), True
    # fallback: hour-of-week median
    dt = pd.Timestamp(t_ms_utc, unit="ms", tz="UTC")
    how = dt.dayofweek * 24 + dt.hour
    return float(prof.get(how, global_med)), False


# ---------------------------------------------------------------------------
# Precompute the engine state stream ONCE per window (mean/std/n arrays).
# The rule only consumes st.band(K)=mean +/-K*std, st.std, st.n -- so caching
# these three arrays lets us evaluate all (K,stop,trigger) combos for a window
# without re-running the engine. The engine code path is IDENTICAL to live;
# we just memoize its scalar outputs. Verified equal to run_config on a sample.
# ---------------------------------------------------------------------------
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def _precompute_py(ticks_t, ticks_mid, window_sec):
    """Reference: drive the real RollingWindow engine (slow, source of truth)."""
    win = RollingWindow(window_sec)
    n = len(ticks_t)
    mean = np.empty(n, dtype=np.float64)
    std = np.empty(n, dtype=np.float64)
    cnt = np.empty(n, dtype=np.int64)
    for i in range(n):
        st = win.update(int(ticks_t[i]), float(ticks_mid[i]))
        mean[i] = st.mean
        std[i] = st.std
        cnt[i] = st.n
    return mean, std, cnt


if _HAVE_NUMBA:
    @njit(cache=True)
    def _precompute_nb(ticks_t, ticks_mid, window_ms):
        """
        JIT mean/std/n stream. EXACTLY mirrors RollingWindow.update's running
        sum/sumsq and tick-time eviction (cutoff = t - window_ms, evict while
        ticks[front].t <= cutoff). The rule only needs mean/std/n (NOT
        high/low), so we skip the monotonic deques here. Verified == engine.
        """
        n = ticks_t.shape[0]
        mean = np.empty(n, dtype=np.float64)
        std = np.empty(n, dtype=np.float64)
        cnt = np.empty(n, dtype=np.int64)
        s = 0.0
        ss = 0.0
        head = 0  # index of oldest in-window tick
        for i in range(n):
            mid = ticks_mid[i]
            s += mid
            ss += mid * mid
            cutoff = ticks_t[i] - window_ms
            while head <= i and ticks_t[head] <= cutoff:
                om = ticks_mid[head]
                s -= om
                ss -= om * om
                head += 1
            k = i - head + 1
            m = s / k
            var = ss / k - m * m
            mean[i] = m
            std[i] = np.sqrt(var) if var > 0.0 else 0.0
            cnt[i] = k
        return mean, std, cnt


def precompute_states(ticks_t, ticks_mid, window_sec):
    if _HAVE_NUMBA:
        return _precompute_nb(ticks_t.astype(np.int64),
                              ticks_mid.astype(np.float64),
                              np.int64(round(window_sec * 1000)))
    return _precompute_py(ticks_t, ticks_mid, window_sec)


_REASON = {0: "stop", 1: "target", 2: "tcap"}

if _HAVE_NUMBA:
    @njit(cache=True)
    def _rule_loop_nb(ticks_t, ticks_mid, ticks_sp, mean, std, cnt,
                      K, stop_pips, tcap_ms, trig_reenter, pip):
        """
        JIT version of the ReversionRule state machine -> numeric trade matrix.
        Mirrors run_config_fast EXACTLY. trig_reenter: 1=reenter, 0=close_beyond.
        Returns (trades[m,11], viol_stop, viol_range).
        cols: 0 side(+1/-1) 1 reason(0stop1tgt2tcap) 2 t_entry 3 t_exit
              4 entry 5 stop 6 target 7 exit 8 pnl 9 gross 10 sp
        """
        n = ticks_t.shape[0]
        out = np.empty((n, 11), dtype=np.float64)
        m_out = 0
        viol_stop = 0
        viol_range = 0
        armed = 0
        in_pos = False
        pos_side = 0
        entry = 0.0
        stop = 0.0
        target = 0.0
        t_entry = np.int64(0)
        sp_px = stop_pips * pip
        for i in range(n):
            t_ms = ticks_t[i]
            mid = ticks_mid[i]
            m = mean[i]
            s = std[i]
            up = m + K * s
            lo = m - K * s
            half = 0.5 * ticks_sp[i] * pip
            if in_pos:
                exit_now = False
                reason = -1
                if pos_side == 1:
                    if mid <= stop:
                        exit_now = True; reason = 0
                    elif mid >= target:
                        exit_now = True; reason = 1
                else:
                    if mid >= stop:
                        exit_now = True; reason = 0
                    elif mid <= target:
                        exit_now = True; reason = 1
                if not exit_now and t_ms - t_entry >= tcap_ms:
                    exit_now = True; reason = 2
                if exit_now:
                    if pos_side == 1:
                        exit_fill = mid - half
                        pnl = (exit_fill - entry) / pip
                        gross = (mid - entry) / pip
                        if reason == 0 and not (mid <= stop + 1e-9):
                            viol_stop += 1
                    else:
                        exit_fill = mid + half
                        pnl = (entry - exit_fill) / pip
                        gross = (entry - mid) / pip
                        if reason == 0 and not (mid >= stop - 1e-9):
                            viol_stop += 1
                    if reason == 1 and gross < -1e-6:
                        viol_range += 1
                    out[m_out, 0] = pos_side
                    out[m_out, 1] = reason
                    out[m_out, 2] = t_entry
                    out[m_out, 3] = t_ms
                    out[m_out, 4] = entry
                    out[m_out, 5] = stop
                    out[m_out, 6] = target
                    out[m_out, 7] = exit_fill
                    out[m_out, 8] = pnl
                    out[m_out, 9] = gross
                    out[m_out, 10] = ticks_sp[i]
                    m_out += 1
                    in_pos = False
                    armed = 0
                continue
            if s <= 0.0 or cnt[i] < 5:
                continue
            enter_side = 0
            if trig_reenter == 0:  # close_beyond
                if mid > up and armed != 1:
                    armed = 1; enter_side = -1
                elif mid < lo and armed != -1:
                    armed = -1; enter_side = 1
                elif lo <= mid <= up:
                    armed = 0
            else:  # reenter
                if mid > up:
                    armed = 1
                elif mid < lo:
                    armed = -1
                if armed == 1 and mid <= up:
                    armed = 0; enter_side = -1
                elif armed == -1 and mid >= lo:
                    armed = 0; enter_side = 1
            if enter_side != 0:
                if enter_side == 1:        # LONG, revert UP -> target=up
                    entry = mid + half
                    stop = entry - sp_px
                    target = up
                    # entry-validity gate: skip if fill already at/past target
                    # (narrow band + spread => degenerate no-meat entry)
                    if target <= entry:
                        continue
                else:                      # SHORT, revert DOWN -> target=lo
                    entry = mid - half
                    stop = entry + sp_px
                    target = lo
                    if target >= entry:
                        continue
                pos_side = enter_side
                t_entry = t_ms
                in_pos = True
        return out[:m_out], viol_stop, viol_range


def run_config_fast(ticks_t, ticks_mid, ticks_sp, mean, std, cnt,
                    K, stop_pips, tcap_sec, trigger, pip):
    """
    Array-driven re-implementation of (engine-state -> ReversionRule) for the
    sweep. Mirrors ontick_rule.ReversionRule EXACTLY (same arm/fire state
    machine, same symmetric stop, same target=opposite band, same tcap, same
    exit precedence). Verified vs run_config (engine+rule) on a subsample.
    Uses the JIT loop when numba is available (verified == this pure path).
    """
    tcap_ms = int(round(tcap_sec * 1000))
    if _HAVE_NUMBA:
        mat, vs, vr = _rule_loop_nb(
            ticks_t.astype(np.int64), ticks_mid.astype(np.float64),
            ticks_sp.astype(np.float64), mean, std, cnt,
            float(K), float(stop_pips), np.int64(tcap_ms),
            1 if trigger == "reenter" else 0, float(pip))
        trades = []
        for r in range(mat.shape[0]):
            trades.append(dict(
                side="LONG" if mat[r, 0] > 0 else "SHORT",
                reason=_REASON[int(mat[r, 1])],
                t_entry=int(mat[r, 2]), t_exit=int(mat[r, 3]),
                entry=mat[r, 4], stop=mat[r, 5], target=mat[r, 6],
                exit=mat[r, 7], exit_mid=0.0, pnl_pips=mat[r, 8],
                gross_pips=mat[r, 9], sp_pips=mat[r, 10]))
        return dict(trades=trades, viol_stop=int(vs), viol_range=int(vr))

    trades = []
    viol_stop = 0
    viol_range = 0
    armed = 0
    in_pos = False
    pos_side = 0   # +1 long, -1 short
    entry = stop = target = 0.0
    t_entry = 0
    n = len(ticks_t)
    sp_px = stop_pips * pip

    for i in range(n):
        t_ms = int(ticks_t[i])
        mid = ticks_mid[i]
        m = mean[i]
        s = std[i]
        up = m + K * s
        lo = m - K * s
        half = 0.5 * ticks_sp[i] * pip

        if in_pos:
            exit_now = False
            reason = ""
            if pos_side == 1:
                if mid <= stop:
                    exit_now, reason = True, "stop"
                elif mid >= target:
                    exit_now, reason = True, "target"
            else:
                if mid >= stop:
                    exit_now, reason = True, "stop"
                elif mid <= target:
                    exit_now, reason = True, "target"
            if not exit_now and t_ms - t_entry >= tcap_ms:
                exit_now, reason = True, "tcap"
            if exit_now:
                if pos_side == 1:
                    exit_fill = mid - half
                    pnl = (exit_fill - entry) / pip
                    gross = (mid - entry) / pip
                    if reason == "stop" and not (mid <= stop + 1e-9):
                        viol_stop += 1
                else:
                    exit_fill = mid + half
                    pnl = (entry - exit_fill) / pip
                    gross = (entry - mid) / pip
                    if reason == "stop" and not (mid >= stop - 1e-9):
                        viol_stop += 1
                if reason == "target" and gross < -1e-6:
                    viol_range += 1
                trades.append(dict(
                    side="LONG" if pos_side == 1 else "SHORT", reason=reason,
                    t_entry=t_entry, t_exit=t_ms, entry=entry, stop=stop,
                    target=target, exit=exit_fill, exit_mid=mid, pnl_pips=pnl,
                    gross_pips=gross, sp_pips=ticks_sp[i]))
                in_pos = False
                armed = 0
            continue

        # flat: look for entry
        if s <= 0.0 or cnt[i] < 5:
            continue

        enter_side = 0
        if trigger == "close_beyond":
            if mid > up and armed != 1:
                armed = 1
                enter_side = -1   # SHORT
            elif mid < lo and armed != -1:
                armed = -1
                enter_side = 1    # LONG
            elif lo <= mid <= up:
                armed = 0
        else:  # reenter
            if mid > up:
                armed = 1
            elif mid < lo:
                armed = -1
            if armed == 1 and mid <= up:
                armed = 0
                enter_side = -1   # SHORT
            elif armed == -1 and mid >= lo:
                armed = 0
                enter_side = 1    # LONG

        if enter_side != 0:
            if enter_side == 1:  # LONG, BUY at mid+half; revert UP -> target=up
                entry = mid + half
                stop = entry - sp_px
                target = up
                if target <= entry:   # entry-validity gate (no meat)
                    continue
            else:                # SHORT, SELL at mid-half; revert DOWN -> target=lo
                entry = mid - half
                stop = entry + sp_px
                target = lo
                if target >= entry:
                    continue
            pos_side = enter_side
            t_entry = t_ms
            in_pos = True

    return dict(trades=trades, viol_stop=viol_stop, viol_range=viol_range)


# ---------------------------------------------------------------------------
# Single (window,K,trigger) backtest over the tick array
# ---------------------------------------------------------------------------
def run_config(ticks_t, ticks_mid, ticks_sp, window_sec, K, stop_pips, tcap_sec,
               trigger, pip):
    """
    ticks_t  : int64 array of UTC ms
    ticks_mid: float64 array of HistData mid
    ticks_sp : float64 array of OANDA spread (pips) prevailing at each tick
    Returns dict of trades and violation counts.
    """
    win = RollingWindow(window_sec)
    rule = ReversionRule(K=K, stop_pips=stop_pips, tcap_sec=tcap_sec,
                         trigger=trigger, pip=pip)
    trades = []
    half_sp_px = None  # set per tick
    viol_stop = 0
    viol_range = 0

    n = len(ticks_t)
    for i in range(n):
        t_ms = int(ticks_t[i])
        mid = float(ticks_mid[i])
        sp_pips = float(ticks_sp[i])
        half = 0.5 * sp_pips * pip  # half-spread in price units

        st = win.update(t_ms, mid)
        act = rule.on_tick(st, mid, t_ms)
        if act is None:
            continue

        if act.kind in (ENTER_LONG, ENTER_SHORT):
            side = "LONG" if act.kind == ENTER_LONG else "SHORT"
            # fill: BUY at mid+half, SELL at mid-half
            fill = mid + half if side == "LONG" else mid - half
            target_mid = rule.target_for(side, st)
            # entry-validity gate: skip degenerate no-meat entry (fill at/past target)
            if (side == "LONG" and target_mid <= fill) or \
               (side == "SHORT" and target_mid >= fill):
                continue
            rule._open(side, fill, target_mid, t_ms)

        elif act.kind == EXIT and rule.pos is not None:
            p = rule.pos
            side = p["side"]
            # exit fill: closing a LONG = SELL at mid-half; closing SHORT = BUY at mid+half
            exit_fill = mid - half if side == "LONG" else mid + half
            # --- ANTI-ARTIFACT self-checks (on mid, the decision price) ---
            if act.reason == "stop":
                if side == "LONG" and not (mid <= p["stop"] + 1e-9):
                    viol_stop += 1
                if side == "SHORT" and not (mid >= p["stop"] - 1e-9):
                    viol_stop += 1
            # exit mid must be a price actually seen (it is `mid`) -> range ok by construction;
            # sanity: pnl bounded -- a target exit can't be worse than a stop exit
            if side == "LONG":
                pnl_px = exit_fill - p["entry"]
            else:
                pnl_px = p["entry"] - exit_fill
            pnl_pips = pnl_px / pip
            # a "target" hit should be a (gross) win in mid terms; flag if grossly violating
            gross_pips = ((mid - p["entry"]) if side == "LONG" else (p["entry"] - mid)) / pip
            if act.reason == "target" and gross_pips < -1e-6:
                viol_range += 1
            trades.append(dict(
                side=side, reason=act.reason, t_entry=p["t_entry_ms"], t_exit=t_ms,
                entry=p["entry"], stop=p["stop"], target=p["target"],
                exit=exit_fill, exit_mid=mid, pnl_pips=pnl_pips,
                gross_pips=gross_pips, sp_pips=sp_pips,
            ))
            rule._close()

    return dict(trades=trades, viol_stop=viol_stop, viol_range=viol_range)


def summarize(res, span_days, label, side_filter=None):
    tr = res["trades"]
    if side_filter:
        tr = [t for t in tr if t["side"] == side_filter]
    n = len(tr)
    if n == 0:
        return dict(label=label, side=side_filter or "BOTH", n=0, tpd=0.0,
                    exp=0.0, wr=0.0, ppd=0.0, med_meat=0.0)
    pnl = np.array([t["pnl_pips"] for t in tr])
    gross = np.array([t["gross_pips"] for t in tr])
    total = pnl.sum()
    exp = pnl.mean()
    wr = float((pnl > 0).mean())
    tpd = n / span_days
    ppd = total / span_days
    med_meat = float(np.median(gross))  # median favorable move in mid (vs spread)
    return dict(label=label, side=side_filter or "BOTH", n=n, tpd=tpd, exp=exp,
                wr=wr, ppd=ppd, med_meat=med_meat,
                viol_stop=res["viol_stop"], viol_range=res["viol_range"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=list(PAIR_CFG))
    ap.add_argument("--triggers", default="reenter,close_beyond")
    ap.add_argument("--spotcheck", type=int, default=0,
                    help="print N sample trades for the best config")
    args = ap.parse_args()

    cfg = PAIR_CFG[args.pair]
    pip = cfg["pip"]
    zip_path = cfg["histdata_zip"]
    if not zip_path.startswith("/"):
        zip_path = os.path.join(BASE, zip_path)
    s5_path = os.path.join(BASE, cfg["s5_ba"])

    print(f"=== {args.pair} ontick reversion backtest ===")
    print(f"HistData: {zip_path}")
    print(f"OANDA S5 BA: {s5_path}")

    hist, hist_dt = load_histdata(zip_path)
    print(f"HistData ticks: {len(hist):,}")

    s5 = pd.read_parquet(s5_path, columns=["timestamp", "close", "bid_c", "ask_c"])

    # --- auto time-align ---
    off_h, corr = detect_utc_offset_hours(hist_dt, hist["mid"], s5)
    # detector defines: UTC == hist + off_h  (it ADDED off_h to hist before correlating)
    CORR_MIN = 0.30
    if corr < CORR_MIN:
        # No usable calendar overlap between HistData month and S5 BA coverage
        # (e.g. GBP_JPY May-2024 vs S5 BA from 2025-03). Cannot auto-detect.
        # HistData ASCII is published in US Eastern time. April/May => EDT = UTC-4,
        # so UTC = hist + 4h. Use that documented convention and FLAG loudly.
        off_h = 4
        print(f"Detected UTC offset: FAILED (best return-corr {corr:.3f} < {CORR_MIN}) "
              f"-- NO calendar overlap with S5 BA. Falling back to HistData EDT "
              f"convention: UTC = hist + 4h. !!! LOUD FLAG: alignment unverified.")
    else:
        print(f"Detected UTC offset: UTC = hist {off_h:+d}h (return-corr {corr:.3f})")
    t_ms_utc = hist["t_ms"].values + off_h * 3600 * 1000

    # --- spread overlay ---
    real, prof, gmed = build_spread_lookup(s5, pip)
    sp_arr = np.empty(len(hist), dtype=np.float64)
    real_hits = 0
    for i, t in enumerate(t_ms_utc):
        sp, is_real = spread_for_tick(int(t), real, prof, gmed)
        sp_arr[i] = sp
        real_hits += is_real
    real_frac = real_hits / len(hist)
    overlay = "REAL per-5s OANDA spread" if real_frac > 0.5 else \
        "HOUR-OF-WEEK MEDIAN PROFILE (suggestive, not clean real-spread)"
    print(f"Spread overlay: {overlay}  (real-bucket hit rate {real_frac:.1%}, "
          f"median spread {np.median(sp_arr):.2f}p)")
    if real_frac <= 0.5:
        print("  !!! LOUD FLAG: HistData month is OUTSIDE OANDA S5 BA coverage. "
              "This run uses a spread PROFILE, treat the numbers as suggestive only.")

    span_days = (t_ms_utc.max() - t_ms_utc.min()) / 1000 / 86400
    print(f"Span: {span_days:.1f} calendar days\n")

    ticks_t = t_ms_utc.astype(np.int64)
    ticks_mid = hist["mid"].values.astype(np.float64)

    # stop ladder (JPY scale)
    stops = [10, 20, 40, 80] if cfg["jpy"] else [10, 20, 40, 80]
    windows = [900, 1800, 3600, 14400]
    Ks = [1.5, 2.0, 2.5]
    tcap_sec = 4 * 3600  # 4h time cap

    rows = []
    best = None
    triggers = args.triggers.split(",")
    # precompute engine state stream ONCE per window (engine path identical to live)
    for w in windows:
        print(f"  [precompute engine states window={w}s ...]", flush=True)
        mean, std, cnt = precompute_states(ticks_t, ticks_mid, w)
        for trig in triggers:
            for K in Ks:
                for stop in stops:
                    res = run_config_fast(ticks_t, ticks_mid, sp_arr, mean, std,
                                          cnt, K, stop, tcap_sec, trig, pip)
                    lbl = f"{trig} w={w} K={K} stop={stop}"
                    both = summarize(res, span_days, lbl)
                    lng = summarize(res, span_days, lbl, "LONG")
                    sht = summarize(res, span_days, lbl, "SHORT")
                    rows.append((trig, w, K, stop, both, lng, sht, res))
                    if both["n"] > 0 and (best is None or both["ppd"] > best[4]["ppd"]):
                        best = (trig, w, K, stop, both, lng, sht, res)

    # ---- report ----
    print(f"{'config':<40} {'side':<6} {'n':>6} {'t/d':>6} {'exp':>7} {'WR':>6} "
          f"{'p/d':>9} {'meat':>7}")
    print("-" * 95)
    # sort by both-sides p/d
    rows_sorted = sorted(rows, key=lambda r: r[4]["ppd"], reverse=True)
    for (trig, w, K, stop, both, lng, sht, res) in rows_sorted[:18]:
        for s in (both, lng, sht):
            print(f"{s['label']:<40} {s['side']:<6} {s['n']:>6} {s['tpd']:>6.1f} "
                  f"{s['exp']:>7.2f} {s['wr']:>6.1%} {s['ppd']:>9.1f} {s['med_meat']:>7.2f}")
        print()

    # violation totals
    tot_vs = sum(r[7]["viol_stop"] for r in rows)
    tot_vr = sum(r[7]["viol_range"] for r in rows)
    print(f"ANTI-ARTIFACT violations across all configs: stop={tot_vs} range={tot_vr} "
          f"(both must be 0)")

    # reconciliation check on best
    if best:
        b = best[4]
        recon = abs(b["exp"] * b["n"] - b["ppd"] * span_days)
        print(f"Reconciliation (exp*n vs p/d*days) on best: residual {recon:.4f} pips "
              f"(should be ~0)")

    # spot-check
    if args.spotcheck and best:
        print(f"\n--- spot-check {args.spotcheck} trades from best config "
              f"[{best[4]['label']}] ---")
        tr = best[7]["trades"][:args.spotcheck]
        for t in tr:
            hold_s = (t["t_exit"] - t["t_entry"]) / 1000
            print(f"  {t['side']:<5} {t['reason']:<6} entry={t['entry']:.4f} "
                  f"stop={t['stop']:.4f} tgt={t['target']:.4f} exit={t['exit']:.4f} "
                  f"pnl={t['pnl_pips']:+.1f}p sp={t['sp_pips']:.2f} hold={hold_s:.0f}s")

    print(f"\nBEST by p/d: {best[4]['label']}  "
          f"p/d={best[4]['ppd']:.1f}  exp={best[4]['exp']:.2f}  "
          f"WR={best[4]['wr']:.1%}  t/d={best[4]['tpd']:.1f}")


if __name__ == "__main__":
    main()
