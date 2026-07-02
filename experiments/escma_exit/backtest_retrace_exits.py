"""
backtest_retrace_exits.py — deterministic exit improvements on retrace entries (2026-06-12).

Same VALIDATED shock entry (thr=2.5, peak=44b, fade fill at watch_start), but swap the EXIT
to attack the two measured leaks in the live 009 book:
  - 122 dead-on-arrival trades (MFE<5p) bled −367p → SCRATCH exit: if MFE<W by bar T_act, cut.
  - near-winners reached +15-18 MFE, missed the fixed 20p TP, gave back → TRAILING exit.
Baseline = current fixed TP20 + horizon close (+ 30p SL). All net of real per-bar spread.
3 live pairs (GBP_JPY, USD_JPY, AUD_JPY), IS/OOS 70/30, vs baseline.
"""
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
S5_DIR = PROJECT / "data" / "s5_ba"
PAIRS = ["GBP_JPY", "USD_JPY", "AUD_JPY"]
THR, PEAK_BARS, HORIZON, Z_WINDOW, MAD_WIN = 2.5, 44, 600, 6, 2048
SL_PIPS, TP_PIPS = 30.0, 20.0
IS_FRAC = 0.70


def compute_shock_z(close, pip, w=Z_WINDOW, mad_win=MAD_WIN):
    n = len(close); vel = np.empty(n); vel[:w] = 0.0
    vel[w:] = (close[w:] - close[:n-w]) / pip
    vs = pd.Series(vel)
    rm = vs.rolling(mad_win, min_periods=50).median()
    rmad = (vs - rm).abs().rolling(mad_win, min_periods=50).median()
    z = ((vs - rm) / (1.4826 * rmad.clip(lower=1e-6))).fillna(0).values
    return z.astype(np.float64), vel


@njit
def sim_exits(bid, ask, close, shock_flag, vel, pip,
              mode, scr_T, scr_W, trail_act, trail_d):
    """mode: 0=baseline(TP20/SL30/horizon) 1=scratch 2=trailing.
    Returns pnl array (pips, net of entry spread)."""
    n = len(close); pb = PEAK_BARS
    max_ev = n // 10
    pnl_out = np.zeros(max_ev); ev = 0; cd = 0
    for t in range(Z_WINDOW, n - pb - HORIZON - 2):
        if cd > 0:
            cd -= 1; continue
        if shock_flag[t] != 1:
            continue
        d = 1 if vel[t] > 0 else -1          # shock dir; trade is the fade (short if d=1)
        sp = (ask[t] - bid[t]) / pip
        ws = t + pb + 1
        if ws >= n:
            continue
        # fade fill at watch_start: d=1 (upshock)→SHORT fill bid; d=-1→LONG fill ask
        fill = bid[ws] if d == 1 else ask[ws]
        tp_lvl = fill - TP_PIPS * pip * d
        sl_lvl = fill + SL_PIPS * pip * d
        peak_mfe = 0.0
        pnl = 0.0; done = 0
        end = min(ws + HORIZON, n - 1)
        for j in range(ws + 1, end + 1):
            lo = bid[j]; hi = ask[j]
            # current unrealized (mid-ish via worst side) for SHORT(d=1): u=(fill-ask)/pip ; LONG(d=-1): u=(bid-fill)/pip
            if d == 1:
                u = (fill - ask[j]) / pip
                mfe_now = (fill - lo) / pip       # best for short = lowest ask~bid
            else:
                u = (bid[j] - fill) / pip
                mfe_now = (hi - fill) / pip
            if mfe_now > peak_mfe:
                peak_mfe = mfe_now
            # SL (all modes)
            if d == 1 and hi >= sl_lvl:
                pnl = -SL_PIPS - sp; done = 1; break
            if d == -1 and lo <= sl_lvl:
                pnl = -SL_PIPS - sp; done = 1; break
            # TP (baseline + scratch; trailing replaces it)
            if mode != 2:
                if d == 1 and lo <= tp_lvl:
                    pnl = TP_PIPS - sp; done = 1; break
                if d == -1 and hi >= tp_lvl:
                    pnl = TP_PIPS - sp; done = 1; break
            # scratch: cut dead trade
            if mode == 1 and (j - ws) >= scr_T and peak_mfe < scr_W:
                pnl = u - sp; done = 1; break
            # trailing: once activated, exit on retrace from peak
            if mode == 2 and peak_mfe >= trail_act and (peak_mfe - u) >= trail_d:
                pnl = u - sp; done = 1; break
        if done == 0:                             # horizon close at market
            if d == 1:
                pnl = (fill - bid[end]) / pip - sp
            else:
                pnl = (ask[end] - fill) / pip - sp
        if ev < max_ev:
            pnl_out[ev] = pnl; ev += 1
        cd = (pb + HORIZON) // 2
    return pnl_out[:ev]


# warmup
_b = np.ones(3000)*150.0; _a = _b+0.02; _c = _b+0.01
_v = np.zeros(3000); _v[100]=1.2; _sf = np.zeros(3000, np.int8); _sf[100]=1
sim_exits(_b,_a,_c,_sf,_v,0.01,0,0,0,0,0); sim_exits(_b,_a,_c,_sf,_v,0.01,1,30,5,0,0); sim_exits(_b,_a,_c,_sf,_v,0.01,2,0,0,8,4)


def split_pd(pnl, eidx, nbar):
    cut = int(nbar * IS_FRAC)
    ism = eidx < cut; oosm = ~ism
    is_d = cut / 17280; oos_d = (nbar - cut) / 17280
    return (pnl[ism].sum()/is_d if is_d else 0, pnl[oosm].sum()/oos_d if oos_d else 0,
            int(ism.sum()), int(oosm.sum()), pnl[oosm].sum())


def main():
    data = {}
    for p in PAIRS:
        df = pd.read_parquet(S5_DIR / f"{p}_S5_BA.parquet").sort_values("timestamp").reset_index(drop=True)
        close = df["close"].values.astype(np.float64)
        bid = df["bid_c"].values.astype(np.float64); ask = df["ask_c"].values.astype(np.float64)
        z, vel = compute_shock_z(close, 0.01)
        sf = (np.abs(z) > THR).astype(np.int8)
        data[p] = (bid, ask, close, sf, vel, len(close))
        del df, z; gc.collect()

    # need entry bar indices for IS/OOS split — recompute deterministically
    def run(mode, scr_T=0, scr_W=0, ta=0, td=0):
        agg_is = agg_oos = 0.0; n_oos = 0; oos_pnls = []
        for p, (bid, ask, close, sf, vel, nbar) in data.items():
            pnl = sim_exits(bid, ask, close, sf, vel, 0.01, mode, scr_T, scr_W, ta, td)
            # reconstruct entry positions (same loop) for split
            eidx = []
            cd = 0
            for t in range(Z_WINDOW, nbar - PEAK_BARS - HORIZON - 2):
                if cd > 0: cd -= 1; continue
                if sf[t] != 1: continue
                eidx.append(t + PEAK_BARS + 1); cd = (PEAK_BARS + HORIZON)//2
            eidx = np.array(eidx[:len(pnl)])
            isd, oosd, nis, nos, opnl = split_pd(pnl, eidx, nbar)
            agg_is += isd; agg_oos += oosd; n_oos += nos; oos_pnls.append(pnl[eidx >= int(nbar*IS_FRAC)])
        return agg_is, agg_oos, n_oos, oos_pnls

    print(f"{'='*78}\nRETRACE EXIT VARIANTS — 3 live pairs, IS/OOS, net spread\n{'='*78}")
    print(f"{'variant':34s} {'IS p/d':>8s} {'OOS p/d':>8s} {'OOS_n':>6s}")
    bis, boos, bn, _ = run(0)
    print(f"{'baseline (TP20/SL30/horizon)':34s} {bis:>8.1f} {boos:>8.1f} {bn:>6d}  <-- current live")
    print("--- (2) SCRATCH: cut if MFE<W by T bars ---")
    for T in (30, 60, 120):
        for W in (3, 5, 8):
            i, o, nn, _ = run(1, T, W, 0, 0)
            flag = "🟢" if o > boos else "🔴"
            print(f"  scratch T={T}b W={W}p{'':18s} {i:>8.1f} {o:>8.1f} {nn:>6d}  {flag} (Δoos {o-boos:+.1f})")
    print("--- (3) TRAILING: activate +A, exit on -D from peak ---")
    for A in (8, 12, 15):
        for D in (3, 5, 8):
            i, o, nn, _ = run(2, 0, 0, A, D)
            flag = "🟢" if o > boos else "🔴"
            print(f"  trail A={A}p D={D}p{'':19s} {i:>8.1f} {o:>8.1f} {nn:>6d}  {flag} (Δoos {o-boos:+.1f})")
    print("="*78)


if __name__ == "__main__":
    main()
