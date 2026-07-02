#!/usr/bin/env python3
"""
H4 Feature Screening — 8A-D pipeline on H4 bars for 4 live pairs
=================================================================
Adapts the Session 049 microstructure 8A-D framework to H4 timeframe.
No sub-bar alignment needed — everything computed directly at H4 resolution.

Differences from M5 study:
  - TF = H4 (240 min bars), windows = [5, 10, 20, 40]
  - Volume = sum of M5 volumes within each H4 bar
  - Two targets:
      dir_next  = sign(next_bar_close - next_bar_open)   [directional]
      big_h4    = |next_ret| > 75th percentile            [magnitude]
  - 8A-D run independently per pair, then cross-pair consensus reported

Usage:
    python3 research/experiments/technical/h4_feature_screening.py
"""

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# ── Re-use Numba kernels + complexity functions from microstructure script ────
_ms_path = ROOT / "research" / "eurusd_rolling_microstructure.py"
import importlib.util
_spec = importlib.util.spec_from_file_location("_ms", _ms_path)
_ms   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ms)

rolling_complexity  = _ms.rolling_complexity
CPLX_KEYS           = _ms.CPLX_KEYS
CPLX_POLARITY       = _ms.CPLX_POLARITY

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data" / "m5_ba"
PAIRS = {
    "GBP_JPY": 0.01,
    "USD_JPY": 0.01,
    "EUR_JPY": 0.01,
    "GBP_USD": 0.0001,
}
H4_MIN   = 240.0          # minutes per H4 bar
WINDOWS  = [5, 10, 20, 40]
BIG_PCT  = 75             # percentile threshold for big-bar target

_ALL_SERIES = ("close", "logret", "ema8r", "ema21r", "madist", "macdh",
               "mom10", "roc10", "atrr", "bbpos", "rvol", "obvr",
               "ppm", "tpm", "ppt")

# ── Data loading ──────────────────────────────────────────────────────────────

def load_h4(pair: str) -> pd.DataFrame:
    path = DATA_DIR / f"{pair}_M5_BA.parquet"
    m5 = pd.read_parquet(path)
    m5.columns = [c.lower() for c in m5.columns]
    if "timestamp" in m5.columns:
        m5.index = pd.to_datetime(m5["timestamp"], utc=True)
        m5 = m5.drop(columns=["timestamp"])
    else:
        m5.index = pd.to_datetime(m5.index, utc=True)
    for c in ["open","high","low","close","bid_c","ask_c"]:
        m5[c] = m5[c].astype(float)
    if "volume" not in m5.columns:
        m5["volume"] = 1.0
    else:
        m5["volume"] = m5["volume"].astype(float)

    h4 = m5.resample("4h", label="left", closed="left").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"),
        bid_c=("bid_c","last"), ask_c=("ask_c","last"),
        volume=("volume","sum"),
    ).dropna(subset=["open","close"])
    h4 = h4[h4["open"] > 0].copy()
    return h4

# ── Feature builder ───────────────────────────────────────────────────────────

def build_features(h4: pd.DataFrame, pip: float) -> dict:
    """Compute all H4 features. Returns dict of {name: np.array(N)}."""
    n  = len(h4)
    cl = h4["close"].values.astype(float)
    op = h4["open"].values.astype(float)
    hi = h4["high"].values.astype(float)
    lo = h4["low"].values.astype(float)
    vol= h4["volume"].values.astype(float)
    _eps = 1e-12

    # True range (in pips)
    prev_cl = np.empty(n); prev_cl[0] = cl[0]; prev_cl[1:] = cl[:-1]
    tr_pips = np.maximum(hi - lo, np.maximum(
        np.abs(hi - prev_cl), np.abs(lo - prev_cl))) / pip

    # Log returns
    lr = np.concatenate([[np.nan], np.log(cl[1:] / (cl[:-1] + _eps))])

    # EMA/momentum/BB
    _s  = pd.Series(cl)
    e8  = _s.ewm(span=8,  adjust=False).mean().values
    e21 = _s.ewm(span=21, adjust=False).mean().values
    e12 = _s.ewm(span=12, adjust=False).mean().values
    e26 = _s.ewm(span=26, adjust=False).mean().values
    atr14 = pd.Series(tr_pips).ewm(alpha=1/14, adjust=False).mean().values
    atr_s = pd.Series(atr14).rolling(20, min_periods=20).mean().values
    sma20 = _s.rolling(20, min_periods=20).mean().values
    std20 = _s.rolling(20, min_periods=20).std().values
    macdl = e12 - e26
    mom10 = np.concatenate([[np.nan]*10, cl[10:] - cl[:-10]])

    ema8r  = cl / (e8  + _eps) - 1.0
    ema21r = cl / (e21 + _eps) - 1.0
    madist = np.where(atr14 > 0, (e8 - e21) / atr14, np.nan)
    macdh  = macdl - pd.Series(macdl).ewm(span=9, adjust=False).mean().values
    mom10n = np.where(atr14 > 0, mom10 / atr14, np.nan)
    roc10  = np.concatenate([[np.nan]*10, (cl[10:]-cl[:-10])/(cl[:-10]+_eps)])
    atrr   = np.where(atr_s > 0, atr14 / atr_s, np.nan)
    bbpos  = np.where(std20 > 0, (cl - sma20) / (2*std20), np.nan)

    # Volume-based (microstructure on H4)
    vol_safe = np.where(vol > 0, vol, np.nan)
    ppt0 = tr_pips / vol_safe          # pips/tick (book thinness)
    tpm0 = vol_safe / H4_MIN           # ticks/min
    ppm0 = tr_pips / H4_MIN            # pips/min

    vol_ma20 = pd.Series(vol).rolling(20, min_periods=1).mean().values
    rvol     = np.where(vol_ma20 > 0, vol / vol_ma20, np.nan)
    sign_arr = np.sign(cl - op)
    obvr     = sign_arr * rvol

    raw_series = {
        "close": cl, "logret": lr, "ema8r": ema8r, "ema21r": ema21r,
        "madist": madist, "macdh": macdh, "mom10": mom10n, "roc10": roc10,
        "atrr": atrr, "bbpos": bbpos, "rvol": rvol, "obvr": obvr,
        "ppm": ppm0, "tpm": tpm0, "ppt": ppt0,
    }

    # Rolling summary stats per window (mean, z-score)
    feat = {}
    for W in WINDOWS:
        for sname, arr in raw_series.items():
            s = pd.Series(arr)
            feat[f"mean_{sname}_{W}"] = s.rolling(W, min_periods=W).mean().values
            mu = s.rolling(W*4, min_periods=W*4).mean()
            sd = s.rolling(W*4, min_periods=W*4).std().clip(lower=1e-9)
            feat[f"z_{sname}_{W}"] = ((s - mu) / sd).values

    # Complexity metrics (W >= 20 only, step=1 since H4 bars are few)
    _step = 1
    for W in [w for w in WINDOWS if w >= 20]:
        for sname, arr in raw_series.items():
            cx = rolling_complexity(np.asarray(arr, dtype=np.float64), W, step=_step)
            for k, v in cx.items():
                feat[f"{k}_{sname}_{W}"] = v

    return feat, raw_series

# ── Target construction ───────────────────────────────────────────────────────

def build_targets(h4: pd.DataFrame):
    cl  = h4["close"].values.astype(float)
    op  = h4["open"].values.astype(float)
    n   = len(h4)
    # next-bar direction: sign of (next close - next open)
    dir_next = np.empty(n); dir_next[:] = np.nan
    dir_next[:-1] = np.sign(cl[1:] - op[1:])
    # next-bar magnitude: |next close - prev close| / pip
    ret_next = np.empty(n); ret_next[:] = np.nan
    ret_next[:-1] = np.abs(cl[1:] - cl[:-1])
    thresh = np.nanpercentile(ret_next, BIG_PCT)
    big_next = (ret_next >= thresh).astype(float)
    big_next[np.isnan(ret_next)] = np.nan
    return dir_next, big_next, thresh

# ── 8A-D pipeline (adapted: no sub-TF alignment) ─────────────────────────────

def run_8ad(pair: str, feat: dict, target: np.ndarray, target_name: str,
            min_r: float = 0.04, dedup_thresh: float = 0.80):
    n_total = len(feat)
    p_bonf  = 0.05 / max(n_total, 1)
    n       = len(target)

    print(f"\n{'─'*68}")
    print(f"  [{pair}] 8A-D  target={target_name}  n_feat={n_total}  Bonf p<{p_bonf:.1e}")
    print(f"{'─'*68}")

    # ── 8A ────────────────────────────────────────────────────────────────────
    ranked = []
    for key, arr in feat.items():
        both = ~np.isnan(arr) & ~np.isnan(target)
        if both.sum() < 50:
            continue
        try:
            r, p = scipy_stats.pearsonr(arr[both], target[both])
        except Exception:
            continue
        ranked.append((abs(r), r, p, key, arr))
    ranked.sort(reverse=True)

    survivors = [(ar, r, p, k, a) for ar, r, p, k, a in ranked
                 if ar >= min_r and p < p_bonf]
    print(f"  8A survivors: {len(survivors)} / {len(ranked)}")
    if not survivors:
        print("  → No features pass significance filter.")
        return [], []

    for absr, r, p, key, _ in survivors[:25]:
        bar = "█" * min(int(absr * 250), 30)
        print(f"    {key:<32}  r={r:>+7.4f}  p={p:>9.2e}  {bar}")
    if len(survivors) > 25:
        print(f"    … {len(survivors)-25} more")

    # ── 8B ────────────────────────────────────────────────────────────────────
    kept = []
    n_dropped = 0
    for cand in survivors:
        _, _, _, _, arr_c = cand
        both_c = ~np.isnan(arr_c)
        redundant = False
        for _, _, _, _, arr_k in kept:
            both_ck = both_c & ~np.isnan(arr_k)
            if both_ck.sum() < 20:
                continue
            r_pair, _ = scipy_stats.pearsonr(arr_c[both_ck], arr_k[both_ck])
            if abs(r_pair) >= dedup_thresh:
                redundant = True; break
        if redundant:
            n_dropped += 1
        else:
            kept.append(cand)

    print(f"\n  8B independent: {len(kept)}  (dropped {n_dropped} redundant)")
    for _, r, _, key, _ in kept:
        print(f"    {key:<32}  r={r:>+7.4f}")

    # ── 8C ────────────────────────────────────────────────────────────────────
    lgb_order = []
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        feat_names = [key for _, _, _, key, _ in kept]
        X = np.column_stack([a for _, _, _, _, a in kept]).astype(np.float32)
        y = target.copy()
        valid = ~np.isnan(y)
        if target_name == "dir_next":
            y = ((y > 0).astype(np.int32))
        else:
            y = y.astype(np.int32)
        y[~valid] = 0

        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                med = float(np.nanmedian(X[:, j]))
                X[mask, j] = med if not np.isnan(med) else 0.0

        n_pos = int(y[valid].sum())
        n_neg = int(valid.sum()) - n_pos
        spw   = n_neg / max(n_pos, 1)

        params = dict(objective="binary", metric="auc", verbosity=-1,
                      n_estimators=300, learning_rate=0.05, num_leaves=15,
                      scale_pos_weight=spw, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=5, min_child_samples=10)

        cv = StratifiedKFold(n_splits=3, shuffle=False)
        imp = np.zeros(len(feat_names))
        aucs = []
        for tr_i, va_i in cv.split(X[valid], y[valid]):
            tr_abs = np.where(valid)[0][tr_i]
            va_abs = np.where(valid)[0][va_i]
            mdl = lgb.LGBMClassifier(**params)
            mdl.fit(X[tr_abs], y[tr_abs],
                    eval_set=[(X[va_abs], y[va_abs])],
                    callbacks=[lgb.early_stopping(30, verbose=False),
                                lgb.log_evaluation(-1)])
            aucs.append(roc_auc_score(y[va_abs], mdl.predict_proba(X[va_abs])[:,1]))
            imp += mdl.booster_.feature_importance(importance_type="gain")

        print(f"\n  8C LightGBM AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
              f"  (folds: {', '.join(f'{a:.4f}' for a in aucs)})")
        ord_ = np.argsort(imp)[::-1]
        best = imp[ord_[0]]
        for i in ord_:
            if imp[i] < 0.01: continue
            rel = imp[i] / best * 100
            bar = "█" * int(rel/100*28)
            print(f"    {feat_names[i]:<32}  gain={imp[i]:>8.1f}  {rel:4.1f}%  {bar}")
        lgb_order = [feat_names[i] for i in ord_ if imp[i] >= 0.01]
    except ImportError:
        print("  8C [skipped — lightgbm not installed]")
    except Exception as e:
        print(f"  8C [error: {e}]")

    # ── 8D ────────────────────────────────────────────────────────────────────
    fold_n = n // 3
    folds  = [(0, fold_n), (fold_n, 2*fold_n), (2*fold_n, n)]
    print(f"\n  8D 3-fold walk-forward stability:")
    print(f"  {'Key':<32}  {'F1':>7}  {'F2':>7}  {'F3':>7}  stable?")

    stable_keys = []
    for _, r_full, _, key, arr in kept:
        frs = []
        for f0, f1 in folds:
            sub = arr[f0:f1]; tsub = target[f0:f1]
            v   = ~np.isnan(sub) & ~np.isnan(tsub)
            if v.sum() < 20:
                frs.append(np.nan)
            else:
                fr, _ = scipy_stats.pearsonr(sub[v], tsub[v])
                frs.append(fr)
        valid_signs = [np.sign(fr) for fr in frs if not np.isnan(fr)]
        stable = (len(valid_signs) >= 2
                  and len(set(valid_signs)) == 1
                  and valid_signs[0] == np.sign(r_full))
        mark = "🟢" if stable else "🔴"
        if stable:
            stable_keys.append(key)
        rs = "  ".join(f"{fr:>+6.3f}" if not np.isnan(fr) else "   n/a" for fr in frs)
        print(f"  {key:<32}  {rs}  {mark}")

    print(f"\n  Stable: {len(stable_keys)} / {len(kept)}")
    if stable_keys:
        for k in stable_keys:
            r = next(r for _, r, _, kk, _ in kept if kk == k)
            print(f"    🟢 {k}  r={r:>+7.4f}")

    return kept, stable_keys


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("  H4 Feature Screening — 8A-D Pipeline")
    print(f"  Pairs: {list(PAIRS.keys())}")
    print(f"  Windows: {WINDOWS}  (× 4h = {[w*4 for w in WINDOWS]}h lookbacks)")
    print(f"  Targets: dir_next (directional) + big_h4 (magnitude >{BIG_PCT}th pct)")
    print("=" * 68)

    cross_stable = {}   # key → list of pairs where it's stable

    for pair, pip in PAIRS.items():
        print(f"\n{'═'*68}")
        print(f"  PAIR: {pair}  pip={pip}")
        print(f"{'═'*68}")

        h4 = load_h4(pair)
        print(f"  H4 bars: {len(h4)}  "
              f"({h4.index[0].date()} → {h4.index[-1].date()})")

        feat, raw_series = build_features(h4, pip)
        dir_next, big_next, big_thresh = build_targets(h4)

        pct_up   = np.nanmean(dir_next > 0) * 100
        pct_big  = np.nanmean(big_next) * 100
        print(f"  dir_next: {pct_up:.1f}% up bars  |  "
              f"big_h4 threshold: {big_thresh/pip:.1f}p  ({pct_big:.1f}% bars)")
        print(f"  Features computed: {len(feat)}")

        for target, tname in [(dir_next, "dir_next"), (big_next, "big_h4")]:
            kept, stable = run_8ad(pair, feat, target, tname)
            for k in stable:
                if k not in cross_stable:
                    cross_stable[k] = []
                cross_stable[k].append((pair, tname))

    # ── Cross-pair consensus ──────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print("  CROSS-PAIR CONSENSUS — features stable in 2+ pairs")
    print(f"{'═'*68}")
    consensus = {k: v for k, v in cross_stable.items() if len(v) >= 2}
    if consensus:
        for k, pairs_list in sorted(consensus.items(),
                                    key=lambda x: -len(x[1])):
            hits = ', '.join(f"{p}({t})" for p, t in pairs_list)
            print(f"  🟢 {k:<34}  [{len(pairs_list)}/4]  {hits}")
    else:
        print("  No feature is stable across 2+ pairs.")

    # ── dir_next-only consensus ───────────────────────────────────────────────
    dir_consensus = {k: [p for p, t in v if t == "dir_next"]
                     for k, v in cross_stable.items()}
    dir_consensus = {k: v for k, v in dir_consensus.items() if len(v) >= 2}
    print(f"\n  DIRECTIONAL consensus (2+ pairs, dir_next only):")
    if dir_consensus:
        for k, pl in sorted(dir_consensus.items(), key=lambda x: -len(x[1])):
            print(f"  🟢 {k:<34}  [{len(pl)}/4]  {', '.join(pl)}")
    else:
        print("  None — no directional feature is cross-pair stable.")

    print(f"\n{'='*68}")
    print("  Done.")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
