"""
HFT micro-bite Stage 1 — IC screen of streaming-derived features at S5.

Question: do our NEW streaming indicators (directional efficiency, short-horizon
momentum/acceleration, spread_rel, range, volume) carry ANY out-of-sample
directional information at the 5-30 s horizon — enough to support a "1 pip to
1 spread" bite NET of spread?

Why S5: true 250 ms ticks have no history (capture started today, 3 h rolling).
S5 (5 s bars, bid/ask, ~18 months) is the finest backtestable proxy for the
live stream — the 5 s tick window ≈ 1 S5 bar.

SOP: signals on mid = (bid+ask)/2 close; cost = real per-bar spread deducted
once (long: buy ask / sell bid); IS/OOS 70/30, OOS only read after IS; causal
features (rolling, no lookahead); forward target strictly bar t+H.

Decisive output per pair: OOS net pips/trade after spread for
  (a) momentum sign-rule, (b) contrarian sign-rule, (c) efficiency-GATED
  momentum (the new-indicator test). If all <= 0 across pairs, the 1-pip-bite
  direction idea is confirmed dead and we stop. Any positive, consistent
  IS+OOS survivor → Stage 2 (build entry).
"""
import glob, os
import numpy as np
import pandas as pd

DATA = sorted(glob.glob("data/s5_ohlc/*_S5_BA.parquet"))
IS_FRAC = 0.70
HORIZONS = [1, 2, 3, 6, 12]          # S5 bars → 5,10,15,30,60 s
MOM_LAGS = [3, 6, 12]                # 15,30,60 s momentum windows
EFF_WIN = 12                         # efficiency lookback (60 s)
SPREAD_AVG_WIN = 240                 # ~20 min rolling spread baseline


def pip_of(path):
    return 0.01 if "JPY" in os.path.basename(path) else 0.0001


def causal_features(mid, hi, lo, vol, spread, pip):
    n = len(mid)
    f = {}
    dmid = np.diff(mid, prepend=mid[0])
    # short-horizon momentum (pips)
    for L in MOM_LAGS:
        m = np.full(n, np.nan)
        m[L:] = (mid[L:] - mid[:-L]) / pip
        f[f"mom{L}"] = m
    # acceleration of mom3
    a = np.full(n, np.nan)
    m3 = f["mom3"]
    a[3:] = m3[3:] - m3[:-3]
    f["accel"] = a
    # directional efficiency over EFF_WIN: |net| / path
    net = np.full(n, np.nan); path = np.full(n, np.nan)
    abscum = np.cumsum(np.abs(dmid))
    net[EFF_WIN:] = np.abs(mid[EFF_WIN:] - mid[:-EFF_WIN])
    path[EFF_WIN:] = abscum[EFF_WIN:] - abscum[:-EFF_WIN]
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(path > 0, net / path, 0.0)
    f["eff"] = eff
    # spread_rel = spread / rolling-mean spread
    sp_avg = pd.Series(spread).rolling(SPREAD_AVG_WIN, min_periods=20).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        f["spread_rel"] = np.where(sp_avg > 0, spread / sp_avg, 1.0)
    # range over EFF_WIN (pips)
    rng = np.full(n, np.nan)
    hi_s = pd.Series(hi).rolling(EFF_WIN).max().values
    lo_s = pd.Series(lo).rolling(EFF_WIN).min().values
    rng = (hi_s - lo_s) / pip
    f["range"] = rng
    # volume z over EFF_WIN
    vm = pd.Series(vol).rolling(SPREAD_AVG_WIN, min_periods=20).mean().values
    vs = pd.Series(vol).rolling(SPREAD_AVG_WIN, min_periods=20).std().values
    with np.errstate(divide="ignore", invalid="ignore"):
        f["vol_z"] = np.where(vs > 0, (vol - vm) / vs, 0.0)
    return f


def net_sign_rule(sig, fwd_pips, spread_pips, contrarian=False):
    """Mean net pips/trade for a sign rule: dir=sign(sig); enter every bar with a
    nonzero signal; net = dir*fwd - spread (full round-trip cost once)."""
    d = np.sign(sig)
    if contrarian:
        d = -d
    mask = (d != 0) & np.isfinite(fwd_pips) & np.isfinite(spread_pips) & np.isfinite(sig)
    if mask.sum() < 500:
        return np.nan, 0
    net = d[mask] * fwd_pips[mask] - spread_pips[mask]
    return float(net.mean()), int(mask.sum())


def main():
    print(f"{'pair':<9}{'sp50':>6}  H*  "
          f"{'mom_IC':>7}{'momNet':>8}{'ctrNet':>8}  effGate(thr.6)  "
          f"{'eff→|f| IC':>10}")
    print("-" * 86)
    summary = []
    for path in DATA:
        pair = os.path.basename(path).replace("_S5_BA.parquet", "")
        pip = pip_of(path)
        df = pd.read_parquet(path, columns=["close", "high", "low", "bid_c", "ask_c", "volume"])
        mid = df["close"].values.astype(np.float64)
        hi = df["high"].values.astype(np.float64)
        lo = df["low"].values.astype(np.float64)
        vol = df["volume"].values.astype(np.float64)
        spread = (df["ask_c"].values - df["bid_c"].values) / pip
        n = len(mid)
        is_end = int(n * IS_FRAC)
        feats = causal_features(mid, hi, lo, vol, spread, pip)
        sp50 = float(np.nanmedian(spread))

        # best directional feature/horizon by IS IC, then report OOS
        best = None
        for L in MOM_LAGS:
            sig = feats[f"mom{L}"]
            for H in HORIZONS:
                fwd = np.full(n, np.nan)
                fwd[:-H] = (mid[H:] - mid[:-H]) / pip
                m_is = np.isfinite(sig[:is_end]) & np.isfinite(fwd[:is_end])
                if m_is.sum() < 1000:
                    continue
                ic = np.corrcoef(sig[:is_end][m_is], fwd[:is_end][m_is])[0, 1]
                if best is None or abs(ic) > abs(best[0]):
                    best = (ic, L, H)
        ic_is, L, H = best
        # OOS evaluation at the IS-chosen (L,H)
        sig = feats[f"mom{L}"]
        fwd = np.full(n, np.nan); fwd[:-H] = (mid[H:] - mid[:-H]) / pip
        oos = slice(is_end, n)
        m_oos = np.isfinite(sig[oos]) & np.isfinite(fwd[oos])
        ic_oos = np.corrcoef(sig[oos][m_oos], fwd[oos][m_oos])[0, 1] if m_oos.sum() > 1000 else np.nan
        mom_net, _ = net_sign_rule(sig[oos], fwd[oos], spread[oos])
        ctr_net, _ = net_sign_rule(sig[oos], fwd[oos], spread[oos], contrarian=True)
        # efficiency-GATED momentum (the new-indicator test): trade only when eff>0.6
        eff = feats["eff"]
        gate = eff[oos] > 0.6
        sg = np.where(gate, sig[oos], 0.0)
        eff_net, eff_n = net_sign_rule(sg, fwd[oos], spread[oos])
        # does efficiency predict |forward move| (magnitude)?  OOS IC(eff, |fwd|)
        af = np.abs(fwd[oos])
        me = np.isfinite(eff[oos]) & np.isfinite(af)
        eff_mag_ic = np.corrcoef(eff[oos][me], af[me])[0, 1] if me.sum() > 1000 else np.nan

        print(f"{pair:<9}{sp50:>6.2f}  {H:>2}  "
              f"{ic_oos:>7.3f}{mom_net:>8.3f}{ctr_net:>8.3f}  "
              f"{eff_net:>7.3f} (n={eff_n:>6})  {eff_mag_ic:>10.3f}")
        summary.append((pair, sp50, ic_oos, mom_net, ctr_net, eff_net, eff_mag_ic))

    print("-" * 86)
    arr = summary
    mompos = sum(1 for s in arr if s[3] > 0)
    ctrpos = sum(1 for s in arr if s[4] > 0)
    effpos = sum(1 for s in arr if s[5] > 0)
    print(f"OOS net>0 (after spread):  momentum {mompos}/{len(arr)} · "
          f"contrarian {ctrpos}/{len(arr)} · eff-gated-mom {effpos}/{len(arr)}")
    print(f"mean OOS net pips/trade:  mom {np.nanmean([s[3] for s in arr]):+.3f} · "
          f"ctr {np.nanmean([s[4] for s in arr]):+.3f} · "
          f"effgate {np.nanmean([s[5] for s in arr]):+.3f}")
    print(f"eff→|fwd| magnitude IC (mean): {np.nanmean([s[6] for s in arr]):+.3f}  "
          f"(>0 ⇒ efficiency predicts move SIZE, useful for vol timing not direction)")
    print("\nUnits: pips/trade NET of full spread. >0 on a majority IS+OOS = real "
          "micro-edge → Stage 2. All <=0 = direction unpredictable net of spread (expected).")


if __name__ == "__main__":
    main()
