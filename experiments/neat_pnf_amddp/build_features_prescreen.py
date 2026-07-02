#!/usr/bin/env python3
"""
Round-3 input build + information-content PRE-SCREEN (AUD_JPY).

Builds the causal candidate inputs for the P&F+AMDDP5 net and measures each one's
information content BEFORE we burn a cloud run. IC is a CHEAP PRIOR, not a gate
(IC != performance: spread, exit-management, interaction, and regime all break the
IC->edge link). It only (a) sanity-checks that mc_d/mc_dd aren't noise on AUD_JPY
at our box-event sampling, and (b) orders priors. The keep/drop arbiter remains
realized OOS AMDDP5 via leave-one-out through WF/MC/surrogate.

Inputs assembled (all causal):
  1. signed_trend_age      — from the P&F box cache (already causal, R4a)
  2. in_trade_running_amddp5 — position-state, computed in-sim (NOT pre-screenable here)
  3. reversal_proximity    — ternary, P&F-native: sign vs nearest recent completed-column
                             reversal level (col_hist tops/bottoms), within K boxes. R4a.
  4. mc_d  (mc_d_a)        — causal MC slope, read from M5 causal parquet
  5. mc_dd (mc_dd_a)       — causal MC curvature, read from M5 causal parquet

LOOKAHEAD GUARD (R3a/R4/R9): M5 bars are timestamped at bar OPEN (OANDA convention,
the StrengthSpread 55-min-leak source). A box event at S5 time t may only use the
M5 bar that has CLOSED by t, i.e. m5_open_ts + 5min <= t. We asof-merge events
against (m5_ts + 5min), direction='backward'. Never the forming M5 bar.

usage: build_features_prescreen.py [PAIR=AUD_JPY] [BOX=5] [REV=3] [K=2]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[3]
PAIR = sys.argv[1] if len(sys.argv) > 1 else "AUD_JPY"
BOX  = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
REV  = int(sys.argv[3]) if len(sys.argv) > 3 else 3
K    = int(sys.argv[4]) if len(sys.argv) > 4 else 2     # proximity window in boxes
PIP  = 0.01 if "JPY" in PAIR else 0.0001
HERE = Path(__file__).parent
CACHE = HERE / "cache"


@njit(cache=True)
def pnf_with_levels(o, h, l, c, bs, rev):
    """P&F kernel (R2 order, paint-all-boxes) that ALSO emits, per bar:
       box_idx (current column index) and rev_level (box idx of the just-completed
       column's extreme at each dir-flip; carries -1 when no new flip this bar)."""
    n = len(c)
    dir_arr = np.zeros(n, np.int64); col_arr = np.zeros(n, np.int64)
    idx_arr = np.zeros(n, np.int64); flip_lvl = np.full(n, -10**9, np.int64)
    pnf_dir = 0; pnf_idx = 0; col_count = 0
    for i in range(n):
        prices = (h[i], l[i]) if c[i] >= o[i] else (l[i], h[i])
        for price in prices:
            if pnf_dir == 0:
                pnf_idx = int(price / bs); pnf_dir = 1; col_count = 1
                continue
            new_idx = int(price / bs); delta = new_idx - pnf_idx
            if pnf_dir == 1:
                if delta >= 1:
                    pnf_idx = new_idx; col_count += delta
                elif delta <= -rev:
                    old_idx = pnf_idx; pnf_dir = -1; pnf_idx = new_idx
                    col_count = max(1, old_idx - new_idx)
                    flip_lvl[i] = old_idx          # completed UP column topped here = resistance
            else:
                if delta <= -1:
                    pnf_idx = new_idx; col_count += (-delta)
                elif delta >= rev:
                    old_idx = pnf_idx; pnf_dir = 1; pnf_idx = new_idx
                    col_count = max(1, new_idx - old_idx)
                    flip_lvl[i] = old_idx          # completed DOWN column bottomed here = support
        dir_arr[i] = pnf_dir; col_arr[i] = col_count; idx_arr[i] = pnf_idx
    return dir_arr, col_arr, idx_arr, flip_lvl


def reversal_proximity(box_idx_ev, flip_levels_ev, K, M=8):
    """Causal ternary per event: using ONLY reversal levels confirmed at or before
       this event, +1 if current box is above the nearest level within K boxes
       (level=support below), -1 if below (resistance above), 0 if not near."""
    out = np.zeros(len(box_idx_ev), np.int8)
    recent = []                                    # rolling last-M confirmed flip levels
    for e in range(len(box_idx_ev)):
        lvl = flip_levels_ev[e]
        if lvl > -10**8:                           # a flip confirmed AT this event
            recent.append(lvl)
            if len(recent) > M: recent.pop(0)
        if not recent:
            continue
        cur = box_idx_ev[e]
        # nearest recent level within K boxes
        d = [abs(cur - L) for L in recent]
        j = int(np.argmin(d))
        if d[j] <= K:
            out[e] = 1 if cur >= recent[j] else -1
    return out


def main():
    s5 = pd.read_parquet(PROJECT / "data" / "s5_ba" / f"{PAIR}_S5_BA.parquet").sort_values("timestamp").reset_index(drop=True)
    o = s5["open"].values.astype(np.float64); h = s5["high"].values.astype(np.float64)
    l = s5["low"].values.astype(np.float64);  c = s5["close"].values.astype(np.float64)
    ts = s5["timestamp"].values
    bs = BOX * PIP
    print(f"{PAIR} S5: {len(c):,} bars  box={BOX}p rev={REV} K={K}")

    dir_arr, col_arr, idx_arr, flip_lvl = pnf_with_levels(o, h, l, c, bs, REV)
    signed_age = (dir_arr * col_arr).astype(np.int64)
    chg = np.empty(len(c), bool); chg[0] = True; chg[1:] = signed_age[1:] != signed_age[:-1]
    ev_idx = np.where(chg)[0]

    rev_prox = reversal_proximity(idx_arr[ev_idx], flip_lvl[ev_idx], K)
    ev = pd.DataFrame({
        "timestamp": ts[ev_idx],
        "signed_age": signed_age[ev_idx],
        "mid": c[ev_idx], "bid": s5["bid_c"].values[ev_idx], "ask": s5["ask_c"].values[ev_idx],
        "rev_prox": rev_prox,
    })
    print(f"  box events: {len(ev):,}   rev_prox nonzero: {(rev_prox!=0).mean()*100:.1f}% "
          f"(+{(rev_prox>0).mean()*100:.1f}/-{(rev_prox<0).mean()*100:.1f})")

    # ── causal mc_d/mc_dd from M5 parquet, aligned to last CLOSED M5 (lookahead guard) ──
    m5 = pd.read_parquet(PROJECT / "data" / "m5_ohlc" / f"{PAIR}_M5_causal_features.parquet")
    m5 = m5[["timestamp", "mc_d_a", "mc_dd_a"]].dropna().sort_values("timestamp").reset_index(drop=True)
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)    # naive S5 -> UTC (match M5)
    if m5["timestamp"].dt.tz is None:
        m5["timestamp"] = m5["timestamp"].dt.tz_localize("UTC")
    m5["avail_ts"] = m5["timestamp"] + pd.Timedelta(minutes=5)     # bar CLOSE time (lookahead guard)
    ev = ev.sort_values("timestamp")
    ev = pd.merge_asof(ev, m5[["avail_ts", "mc_d_a", "mc_dd_a"]].rename(columns={"avail_ts": "timestamp"}),
                       on="timestamp", direction="backward")
    cov = ev["mc_d_a"].notna().mean() * 100
    print(f"  mc coverage (M5 causal, closed-bar aligned): {cov:.1f}%  "
          f"(M5 ends {m5['timestamp'].max()}, S5 ends {pd.Timestamp(ts[-1])})")
    ev = ev.dropna(subset=["mc_d_a", "mc_dd_a"]).reset_index(drop=True)

    # ── forward-return targets at the box-event clock (net of spread, both signs) ──
    mid = ev["mid"].values
    sp = (ev["ask"].values - ev["bid"].values) / PIP
    for H in (1, 3, 8):
        fwd = np.full(len(mid), np.nan)
        fwd[:-H] = (mid[H:] - mid[:-H]) / PIP
        ev[f"fwd{H}"] = fwd

    # ── PRE-SCREEN: IC (Spearman) of each input vs forward return ──
    print("\n=== INFORMATION-CONTENT PRE-SCREEN (Spearman IC; PRIOR, not a gate) ===")
    print(f"  {'input':<14}{'IC@H1':>9}{'IC@H3':>9}{'IC@H8':>9}   note")
    inputs = {"signed_age": ev["signed_age"].values, "rev_prox": ev["rev_prox"].values,
              "mc_d": ev["mc_d_a"].values, "mc_dd": ev["mc_dd_a"].values}
    notes = {"signed_age": "trend-age (momentum proxy)", "rev_prox": "CONDITIONER — judge in-net, not by IC",
             "mc_d": "directional — IC meaningful", "mc_dd": "directional — IC meaningful"}
    for name, x in inputs.items():
        ics = []
        for H in (1, 3, 8):
            y = ev[f"fwd{H}"].values
            m = np.isfinite(x) & np.isfinite(y)
            ic = spearmanr(x[m], y[m]).correlation if m.sum() > 100 else np.nan
            ics.append(ic)
        print(f"  {name:<14}{ics[0]:>+9.4f}{ics[1]:>+9.4f}{ics[2]:>+9.4f}   {notes[name]}")
    print(f"\n  median spread {np.median(sp):.2f}p  |  reminder: IC over sub-spread moves "
          f"is NOT tradeable edge (oracle: ~half of optimal legs sub-spread).")

    # ── oracle 'good-trade-start' agreement (if labels exist) ──
    olabels = HERE.parent / "oracle_traits" / f"{PAIR}_oracle_trades.parquet"
    if olabels.exists():
        ot = pd.read_parquet(olabels)
        starts = pd.to_datetime(ts[ot["entry_idx"].values.astype(int)], utc=True)
        sdir = ot["dir"].values
        st = pd.DataFrame({"timestamp": starts, "odir": sdir}).sort_values("timestamp")
        ev2 = pd.merge_asof(ev[["timestamp", "signed_age", "rev_prox", "mc_d_a", "mc_dd_a"]],
                            st, on="timestamp", direction="nearest",
                            tolerance=pd.Timedelta("30s"))
        hit = ev2["odir"].notna()
        print(f"\n  oracle good-trade-start overlap: {hit.mean()*100:.1f}% of box events sit on an optimal entry")
        if hit.sum() > 50:
            for name in ("mc_d_a", "mc_dd_a"):
                agree = np.sign(ev2.loc[hit, name]) == np.sign(ev2.loc[hit, "odir"])
                print(f"    {name} sign agrees with optimal-trade dir at those starts: {agree.mean()*100:.1f}%")
    else:
        print(f"\n  (oracle labels for {PAIR} not ready yet — rerun for the good-trade-start check)")

    out = CACHE / f"{PAIR}_feat_b{int(BOX)}_rev{REV}_K{K}.parquet"
    ev.to_parquet(out)
    print(f"\n  enriched features -> {out.name}  ({len(ev):,} rows, "
          f"cols: signed_age, rev_prox, mc_d_a, mc_dd_a + targets)")


if __name__ == "__main__":
    main()
