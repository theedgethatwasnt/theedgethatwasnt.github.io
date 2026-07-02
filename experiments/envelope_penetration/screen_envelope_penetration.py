#!/usr/bin/env python3
"""
Envelope Penetration Ratio P(t) — feature build + screen.

FEATURE (built on S5 bars, data/s5_ohlc/{PAIR}_S5_BA.parquet):
  1. Fast Bollinger on close: basis=SMA5, bands = basis +- 1.0*rolling_std5 (trailing 5 S5 bars).
  2. Per bar, frac_out = (portion of [low,high] above upper + portion below lower) / true_range, clamp [0,1].
     Also frac_above (above upper / range) and frac_below (below lower / range) separately.
  3. P(t)       = rolling mean of frac_out over trailing 60 S5 bars (= 5 min).
  4. Psigned(t) = rolling mean of (frac_above - frac_below) over trailing 60 S5 bars.

ALIGNMENT: P is computed at S5 cadence; we sample it at M5 boundaries (every 60 S5 bars).
The value of P at the close of an M5 bar uses ONLY the 60 S5 bars that compose that M5 bar
(trailing window == that M5 bar exactly). The forward target starts AFTER that window -> no overlap.

LEAKAGE CONTROL: clean no-overlap forward return. P(t) at M5 bar k is built from S5 bars
  composing M5 bar k. The forward return is measured from the END of M5 bar k onward
  (M5 bar k+1's close - M5 bar k's close, etc). Predictor and target share NO bars.

SCREEN (12 pairs, net of real spread = ask_c - bid_c):
  1. Direction:  IC(Psigned, clean_fwd_ret), IC(P, clean_fwd_ret)
  2. Magnitude:  IC(P, |clean_fwd_ret|), IC(P, fwd_realized_vol)
  3. Persistence: autocorr(P) at several lags
  4. Conditional: decile buckets of P -> next-5min dir / mag / vol, net spread.

Gates: Newey-West t (|t|>2), walk-forward sign (3-split all same sign), MC permutation (p<0.05).
"""
import os, sys, json, gc
import numpy as np
import pandas as pd
from numba import njit

DATA_DIR = os.environ.get("S5_DIR", "data/s5_ohlc")
PAIRS = ["AUD_JPY","AUD_USD","CAD_JPY","CHF_JPY","EUR_GBP","EUR_JPY",
         "EUR_USD","GBP_JPY","GBP_USD","NZD_JPY","NZD_USD","USD_JPY"]
S5_PER_M5 = 60          # 60 * 5s = 300s = 5 min
BB_N = 5                # SMA5 / std5
P_WIN = 60              # rolling mean window for P (60 S5 = 5 min)
FWD_HORIZONS_M5 = [1, 2, 3, 6, 12]   # M5 bars forward (5, 10, 15, 30, 60 min)
SEED = 42

def pip_size(pair):
    return 0.01 if pair.endswith("JPY") else 0.0001


@njit
def compute_frac_components(close, high, low):
    """Per-S5-bar frac_above, frac_below using trailing-5 fast Bollinger (1.0 sigma).
    Bands at bar i use close[i-4..i] (inclusive, trailing 5). Returns NaN for warmup."""
    n = len(close)
    frac_above = np.full(n, np.nan)
    frac_below = np.full(n, np.nan)
    for i in range(BB_N - 1, n):
        # SMA5 and population std over trailing BB_N closes
        s = 0.0
        for j in range(i - BB_N + 1, i + 1):
            s += close[j]
        mean = s / BB_N
        ss = 0.0
        for j in range(i - BB_N + 1, i + 1):
            d = close[j] - mean
            ss += d * d
        std = np.sqrt(ss / BB_N)
        upper = mean + 1.0 * std
        lower = mean - 1.0 * std
        hi = high[i]; lo = low[i]
        rng = hi - lo
        if rng <= 0.0:
            # zero-range bar: treat as point; outside if beyond a band
            fa = 0.0; fb = 0.0
            if hi > upper:
                fa = 1.0
            if lo < lower:
                fb = 1.0
            frac_above[i] = fa
            frac_below[i] = fb
            continue
        # portion of [lo,hi] above upper
        above = hi - upper
        if above < 0.0:
            above = 0.0
        if above > rng:
            above = rng
        # portion below lower
        below = lower - lo
        if below < 0.0:
            below = 0.0
        if below > rng:
            below = rng
        fa = above / rng
        fb = below / rng
        # clamp combined to <=1 (overlap impossible unless band collapsed; guard anyway)
        if fa + fb > 1.0:
            scale = 1.0 / (fa + fb)
            fa *= scale
            fb *= scale
        frac_above[i] = fa
        frac_below[i] = fb
    return frac_above, frac_below


@njit
def rolling_mean_trailing(x, win):
    """Trailing rolling mean ignoring nan in warmup. Value at i uses x[i-win+1..i]."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(win - 1, n):
        s = 0.0
        ok = True
        for j in range(i - win + 1, i + 1):
            v = x[j]
            if np.isnan(v):
                ok = False
                break
            s += v
        if ok:
            out[i] = s / win
    return out


def build_m5_screen_table(pair):
    """Returns a DataFrame indexed by M5 bar with:
       P, Psigned (predictor at M5-bar-k close), and clean forward targets.
    """
    path = os.path.join(DATA_DIR, f"{pair}_S5_BA.parquet")
    df = pd.read_parquet(path, columns=["timestamp","high","low","close","bid_c","ask_c"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].values
    close = df["close"].astype(np.float64).values
    high  = df["high"].astype(np.float64).values
    low   = df["low"].astype(np.float64).values
    bid   = df["bid_c"].astype(np.float64).values
    ask   = df["ask_c"].astype(np.float64).values

    # S5-level penetration components
    fa, fb = compute_frac_components(close, high, low)
    frac_out = fa + fb
    frac_out = np.clip(frac_out, 0.0, 1.0)
    frac_signed = fa - fb

    P = rolling_mean_trailing(frac_out, P_WIN)
    Psigned = rolling_mean_trailing(frac_signed, P_WIN)

    # Build M5 grid by flooring timestamp to 5-min. We need P sampled at M5 close,
    # and the M5 close price (mid) + spread. We snap S5 -> M5 bins.
    tser = pd.to_datetime(ts, utc=True)
    m5bin = tser.floor("5min")

    s5 = pd.DataFrame({
        "m5bin": m5bin,
        "close": close,
        "high": high,
        "low": low,
        "bid": bid,
        "ask": ask,
        "P": P,
        "Psigned": Psigned,
    })
    # M5 aggregate: take the LAST S5 bar in each M5 bin as the bar close.
    # P/Psigned at that last bar use trailing 60 S5 == this M5 bar (when bin is full).
    grp = s5.groupby("m5bin", sort=True)
    last = grp.tail(1).set_index("m5bin")
    counts = grp.size().rename("nb5")
    m5 = last.join(counts, how="left")
    # M5 OHLC for realized vol / range targets
    m5["m5_high"] = grp["high"].max().values
    m5["m5_low"]  = grp["low"].min().values
    m5["spread_pips"] = (m5["ask"] - m5["bid"]) / pip_size(pair)

    # Require full M5 bars (60 S5) so the trailing-60 P window == this M5 bar exactly.
    m5 = m5[m5["nb5"] == S5_PER_M5].copy()
    m5 = m5[np.isfinite(m5["P"].values) & np.isfinite(m5["Psigned"].values)].copy()
    m5 = m5.reset_index().rename(columns={"m5bin": "ts"})

    pip = pip_size(pair)
    # Clean forward targets: measured from THIS M5 bar's close forward (no overlap with P window).
    c = m5["close"].values
    spread = m5["spread_pips"].values
    n = len(m5)
    for h in FWD_HORIZONS_M5:
        fwd = np.full(n, np.nan)
        fwd_abs = np.full(n, np.nan)
        fwd_vol = np.full(n, np.nan)
        for i in range(n - h):
            ret_pips = (c[i + h] - c[i]) / pip
            fwd[i] = ret_pips
            fwd_abs[i] = abs(ret_pips)
        # realized vol = std of per-M5 log-ish returns over next h bars (in pips)
        step = (np.diff(c) / pip)
        for i in range(n - h):
            seg = step[i:i + h]
            if len(seg) > 0:
                fwd_vol[i] = np.std(seg)
        # net-spread directional return: subtract round-trip spread cost from |move| sign
        # for direction net spread we evaluate sign(Psigned)*ret - spread later in conditional.
        m5[f"fwd{h}"] = fwd
        m5[f"fwd{h}_abs"] = fwd_abs
        m5[f"fwd{h}_vol"] = fwd_vol
    del df, s5, grp, last
    gc.collect()
    return m5


# ---------- stats helpers ----------
def newey_west_tstat(x, lag):
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return 0.0
    xbar = np.mean(x)
    gamma0 = np.mean((x - xbar) ** 2)
    nw_var = gamma0
    for j in range(1, lag + 1):
        if j >= n:
            break
        w = 1.0 - j / (lag + 1.0)
        gamma_j = np.mean((x[j:] - xbar) * (x[:-j] - xbar))
        nw_var += 2.0 * w * gamma_j
    nw_var = max(nw_var / n, 1e-20)
    return xbar / np.sqrt(nw_var)


def spearman_ic(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        return np.nan
    xr = pd.Series(x[m]).rank().values
    yr = pd.Series(y[m]).rank().values
    xr = xr - xr.mean(); yr = yr - yr.mean()
    denom = np.sqrt((xr * xr).sum() * (yr * yr).sum())
    if denom == 0:
        return np.nan
    return float((xr * yr).sum() / denom)


def block_ic_tstat(x, y, block=200):
    """IC computed per block, NW t-stat over block ICs (mirrors feature_statistics)."""
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(x)
    if n < block * 3:
        ic = spearman_ic(x, y)
        return ic, 0.0, ic
    ics = []
    for s in range(0, n - block + 1, block):
        ics.append(spearman_ic(x[s:s+block], y[s:s+block]))
    ics = np.array([v for v in ics if np.isfinite(v)])
    if len(ics) < 3:
        return spearman_ic(x, y), 0.0, spearman_ic(x, y)
    return float(np.mean(ics)), newey_west_tstat(ics, max(1, int(len(ics) ** (1/3)))), spearman_ic(x, y)


def walkforward_ic_sign(x, y, splits=3):
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(x)
    if n < splits * 100:
        return []
    out = []
    bs = n // splits
    for s in range(splits):
        a = s * bs
        b = (s + 1) * bs if s < splits - 1 else n
        out.append(spearman_ic(x[a:b], y[a:b]))
    return out


def mc_permutation_pval(x, y, n_perm=500, seed=SEED):
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 100:
        return np.nan
    obs = abs(spearman_ic(x, y))
    rng = np.random.default_rng(seed)
    cnt = 0
    xr = pd.Series(x).rank().values
    for _ in range(n_perm):
        yp = rng.permutation(y)
        yr = pd.Series(yp).rank().values
        a = xr - xr.mean(); b = yr - yr.mean()
        d = np.sqrt((a*a).sum()*(b*b).sum())
        ic = 0.0 if d == 0 else (a*b).sum()/d
        if abs(ic) >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def autocorr(x, lags):
    x = x[np.isfinite(x)]
    x = x - x.mean()
    var = (x * x).sum()
    out = {}
    for L in lags:
        if L >= len(x):
            out[L] = np.nan
            continue
        out[L] = float((x[L:] * x[:-L]).sum() / var) if var > 0 else np.nan
    return out


def main():
    out = {"feature": "envelope_penetration_ratio", "params": {
        "BB_N": BB_N, "P_WIN": P_WIN, "S5_PER_M5": S5_PER_M5,
        "fwd_horizons_m5": FWD_HORIZONS_M5, "sigma": 1.0}, "pairs": {}}

    # Accumulators for pooled (cross-pair) screen
    pool = {h: {"P": [], "Psig": [], "fwd": [], "fwdabs": [], "fwdvol": [],
                "spread": []} for h in FWD_HORIZONS_M5}
    ac_accum = []

    for pair in PAIRS:
        print(f"\n=== {pair} ===", flush=True)
        try:
            m5 = build_m5_screen_table(pair)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            out["pairs"][pair] = {"error": str(e)}
            continue
        n = len(m5)
        print(f"  M5 bars: {n}", flush=True)
        P = m5["P"].values
        Psig = m5["Psigned"].values
        ac = autocorr(P, [1, 3, 6, 12, 24])
        ac_accum.append(ac)
        pres = {"n_m5": int(n), "P_mean": float(np.nanmean(P)),
                "P_std": float(np.nanstd(P)), "Psig_mean": float(np.nanmean(Psig)),
                "mean_spread_pips": float(np.nanmean(m5["spread_pips"].values)),
                "autocorr_P": ac, "horizons": {}}
        for h in FWD_HORIZONS_M5:
            fwd = m5[f"fwd{h}"].values
            fwdabs = m5[f"fwd{h}_abs"].values
            fwdvol = m5[f"fwd{h}_vol"].values
            spr = m5["spread_pips"].values
            # Direction
            ic_dir_signed, t_dir_signed, _ = block_ic_tstat(Psig, fwd)
            ic_dir_uns, t_dir_uns, _ = block_ic_tstat(P, fwd)
            # Magnitude
            ic_mag, t_mag, _ = block_ic_tstat(P, fwdabs)
            ic_vol, t_vol, _ = block_ic_tstat(P, fwdvol)
            pres["horizons"][h] = {
                "ic_dir_signed": ic_dir_signed, "t_dir_signed": t_dir_signed,
                "ic_dir_unsigned": ic_dir_uns, "t_dir_unsigned": t_dir_uns,
                "ic_mag": ic_mag, "t_mag": t_mag,
                "ic_vol": ic_vol, "t_vol": t_vol,
            }
            # pool
            mm = np.isfinite(fwd) & np.isfinite(P) & np.isfinite(Psig)
            pool[h]["P"].append(P[mm]); pool[h]["Psig"].append(Psig[mm])
            pool[h]["fwd"].append(fwd[mm]); pool[h]["fwdabs"].append(fwdabs[mm])
            pool[h]["fwdvol"].append(fwdvol[mm]); pool[h]["spread"].append(spr[mm])
        out["pairs"][pair] = pres
        # print per-pair quick line
        h = 1
        hp = pres["horizons"][h]
        print(f"  h={h}: dir_signed IC={hp['ic_dir_signed']:+.4f}(t={hp['t_dir_signed']:+.2f}) "
              f"mag IC={hp['ic_mag']:+.4f}(t={hp['t_mag']:+.2f}) "
              f"vol IC={hp['ic_vol']:+.4f}(t={hp['t_vol']:+.2f}) ac1={ac.get(1):.3f}", flush=True)
        del m5; gc.collect()

    # ---- Pooled cross-pair screen with WF + MC + conditional deciles ----
    print("\n=== POOLED CROSS-PAIR SCREEN ===", flush=True)
    out["pooled"] = {}
    for h in FWD_HORIZONS_M5:
        P = np.concatenate(pool[h]["P"]) if pool[h]["P"] else np.array([])
        Psig = np.concatenate(pool[h]["Psig"]) if pool[h]["Psig"] else np.array([])
        fwd = np.concatenate(pool[h]["fwd"]) if pool[h]["fwd"] else np.array([])
        fwdabs = np.concatenate(pool[h]["fwdabs"]) if pool[h]["fwdabs"] else np.array([])
        fwdvol = np.concatenate(pool[h]["fwdvol"]) if pool[h]["fwdvol"] else np.array([])
        spr = np.concatenate(pool[h]["spread"]) if pool[h]["spread"] else np.array([])
        if len(P) < 1000:
            continue
        ic_dir_signed, t_dir_signed, _ = block_ic_tstat(Psig, fwd)
        ic_dir_uns, t_dir_uns, _ = block_ic_tstat(P, fwd)
        ic_mag, t_mag, _ = block_ic_tstat(P, fwdabs)
        ic_vol, t_vol, _ = block_ic_tstat(P, fwdvol)
        wf_dir = walkforward_ic_sign(Psig, fwd)
        wf_mag = walkforward_ic_sign(P, fwdabs)
        wf_vol = walkforward_ic_sign(P, fwdvol)
        mc_dir = mc_permutation_pval(Psig, fwd)
        mc_mag = mc_permutation_pval(P, fwdabs, n_perm=500)
        mc_vol = mc_permutation_pval(P, fwdvol, n_perm=500)

        # Conditional decile buckets on P (magnitude/vol/dir)
        dec = pd.qcut(pd.Series(P), 10, labels=False, duplicates="drop")
        buckets = []
        for d in range(int(np.nanmax(dec)) + 1):
            msk = (dec == d).values
            if msk.sum() < 50:
                continue
            buckets.append({
                "decile": d,
                "n": int(msk.sum()),
                "P_mean": float(np.nanmean(P[msk])),
                "fwd_dir_mean": float(np.nanmean(fwd[msk])),     # signed mean move (pips)
                "fwd_abs_mean": float(np.nanmean(fwdabs[msk])),  # magnitude (pips)
                "fwd_vol_mean": float(np.nanmean(fwdvol[msk])),
                "spread_mean": float(np.nanmean(spr[msk])),
            })
        # Signed-direction conditional: does sign(Psigned) earn net spread?
        # net = sign(Psig)*fwd - spread (round-trip approx = 1*spread)
        sig = np.sign(Psig)
        net_dir = sig * fwd - spr
        # top/bottom Psigned decile net
        dsig = pd.qcut(pd.Series(Psig), 10, labels=False, duplicates="drop")
        sig_buckets = []
        for d in range(int(np.nanmax(dsig)) + 1):
            msk = (dsig == d).values
            if msk.sum() < 50:
                continue
            # follow-breach: long if Psig>0, else short; net of spread
            follow = np.where(Psig[msk] >= 0, fwd[msk], -fwd[msk]) - spr[msk]
            fade = -follow - 2 * spr[msk] + spr[msk]  # fade = opposite, recompute cleanly below
            fade = np.where(Psig[msk] >= 0, -fwd[msk], fwd[msk]) - spr[msk]
            sig_buckets.append({
                "Psig_decile": d,
                "n": int(msk.sum()),
                "Psig_mean": float(np.nanmean(Psig[msk])),
                "fwd_dir_mean": float(np.nanmean(fwd[msk])),
                "follow_net_pips": float(np.nanmean(follow)),
                "fade_net_pips": float(np.nanmean(fade)),
                "spread_mean": float(np.nanmean(spr[msk])),
            })

        out["pooled"][h] = {
            "n": int(len(P)),
            "ic_dir_signed": ic_dir_signed, "t_dir_signed": t_dir_signed,
            "ic_dir_unsigned": ic_dir_uns, "t_dir_unsigned": t_dir_uns,
            "ic_mag": ic_mag, "t_mag": t_mag,
            "ic_vol": ic_vol, "t_vol": t_vol,
            "wf_dir_signed": wf_dir, "wf_mag": wf_mag, "wf_vol": wf_vol,
            "mc_p_dir": mc_dir, "mc_p_mag": mc_mag, "mc_p_vol": mc_vol,
            "deciles_P": buckets,
            "deciles_Psigned": sig_buckets,
        }
        print(f"h={h}: POOLED n={len(P)} | dir_signed IC={ic_dir_signed:+.4f}(t={t_dir_signed:+.2f},mc={mc_dir}) "
              f"| mag IC={ic_mag:+.4f}(t={t_mag:+.2f},mc={mc_mag}) "
              f"| vol IC={ic_vol:+.4f}(t={t_vol:+.2f},mc={mc_vol})", flush=True)
        print(f"      WF dir={['%+.3f'%v for v in wf_dir]} mag={['%+.3f'%v for v in wf_mag]} vol={['%+.3f'%v for v in wf_vol]}", flush=True)

    # mean autocorr across pairs
    if ac_accum:
        lags = [1,3,6,12,24]
        out["mean_autocorr_P"] = {L: float(np.nanmean([a.get(L, np.nan) for a in ac_accum])) for L in lags}

    with open(os.path.join(os.path.dirname(__file__) or ".", "envelope_penetration_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nSaved envelope_penetration_results.json", flush=True)
    return out


if __name__ == "__main__":
    main()
