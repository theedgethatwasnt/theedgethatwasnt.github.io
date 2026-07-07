"""
EUR/USD Rolling Microstructure Study
======================================
pips/tick  = book thinness (how much each order moves price)
ticks/min  = order flow rate (market activity)
pips/min   = velocity = pips/tick × ticks/min

Computed at W = 5, 10, 20, 60 bar rolling windows on S30, M1, M5.
Complexity metrics run on 13 series: ppm, tpm, ppt, close, logret,
  ema8r, ema21r, madist, macdh, mom10, roc10, atrr, bbpos.

Analyses:
  1. Distribution and autocorrelation of each rolling metric
  2. Z-score time series (normalised by 60-bar baseline)
  3. Predictive power: metric quintile vs P(next big-M5 bar)
  4. Event study: mean metric profile [-40..+10 S30 bars] around big M5 events
  5. Lead/lag scan: at what lag does the S30 metric best predict big M5 bar?
  6. ASCII sparkline of recent S30 microstructure (last 200 bars)
  7. Complexity / Fractal metrics on all 13 series: correlation ranking
  8. Feature selection: Bonferroni filter → dedup → LightGBM → walk-forward stability
"""
import sys, warnings
from pathlib import Path
from datetime import timezone

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.ndimage import uniform_filter1d
from numba import njit, prange

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PIP = 0.0001
TF_MIN = {"S30": 0.5, "M1": 1.0, "M5": 5.0}

# ── helpers ──────────────────────────────────────────────────────────────────

def true_range_pips(df):
    hi = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    cl = df["close"].values.astype(float)
    pc = np.empty_like(cl); pc[0]=cl[0]; pc[1:]=cl[:-1]
    return np.maximum(hi-lo, np.maximum(np.abs(hi-pc), np.abs(lo-pc))) / PIP

def load_mid(path):
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index":"timestamp","time":"timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "open" not in df.columns:
        for fr,to in [("o","open"),("h","high"),("l","low"),("c","close")]:
            df[to] = (df[f"bid_{fr}"].astype(float) + df[f"ask_{fr}"].astype(float)) / 2
    else:
        for c in ["open","high","low","close"]:
            df[c] = df[c].astype(float)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def align_low_to_high(df_low, df_high, sec_high):
    lo_ts = df_low["timestamp"].values.astype(np.int64)
    hi_ts = df_high["timestamp"].values.astype(np.int64)
    hi_dur = sec_high * int(1e9)
    pos = np.searchsorted(hi_ts, lo_ts, side="right") - 1
    valid = (pos >= 0) & (lo_ts < hi_ts[np.clip(pos, 0, len(hi_ts)-1)] + hi_dur)
    pos[~valid] = -1
    return pos

# ── Complexity / Trendiness Metrics ──────────────────────────────────────────

# ── Numba JIT complexity kernel ──────────────────────────────────────────────

@njit(cache=True)
def _ord3(a, b, c):
    """Ordinal pattern index (0-5) for triplet (a,b,c) — argsort of 3 elements."""
    if a <= b:
        if b <= c:   return 0   # a≤b≤c → [0,1,2]
        elif a <= c: return 1   # a≤c<b → [0,2,1]
        else:        return 4   # c<a≤b → [2,0,1]
    else:
        if a <= c:   return 2   # b<a≤c → [1,0,2]
        elif b <= c: return 3   # b≤c<a → [1,2,0]
        else:        return 5   # c<b<a → [2,1,0]


@njit(cache=True)
def _higuchi_nb(x, k_max):
    """Higuchi FD (JIT). Returns value clipped to [1,2]."""
    N = len(x)
    lk = np.empty(k_max, dtype=np.float64)
    ll = np.empty(k_max, dtype=np.float64)
    n_pts = 0
    for k in range(1, k_max + 1):
        total = 0.0
        cnt   = 0
        for m in range(1, k + 1):
            n = (N - m) // k
            if n < 1:
                continue
            s = 0.0
            for j in range(1, n + 1):
                s += abs(x[m - 1 + j * k] - x[m - 1 + (j - 1) * k])
            total += s * (N - 1) / (n * k * k)
            cnt   += 1
        if cnt > 0:
            L = total / cnt
            if L > 0.0:
                lk[n_pts] = -np.log(float(k))
                ll[n_pts] = np.log(L)
                n_pts += 1
    if n_pts < 2:
        return np.nan
    # OLS slope via normal equations
    sx = 0.0; sy = 0.0; sxy = 0.0; sxx = 0.0
    for i in range(n_pts):
        sx  += lk[i]; sy  += ll[i]
        sxy += lk[i] * ll[i]; sxx += lk[i] * lk[i]
    denom = n_pts * sxx - sx * sx
    if abs(denom) < 1e-12:
        return np.nan
    slope = (n_pts * sxy - sx * sy) / denom
    if slope < 1.0: slope = 1.0
    if slope > 2.0: slope = 2.0
    return slope


@njit(cache=True)
def _cplx_nb(x):
    """16 complexity metrics for window x. Returns float64[16].
    tv=0 er=1 plr=2 vol=3 rv=4 lr_vol=5 zcr=6 ci=7
    katz_fd=8 higuchi_fd=9 hurst_rs=10 fd_hurst=11
    perm_ent=12 spec_ent=13(NaN) wav_e1=14 wav_e2=15
    """
    N   = len(x)
    out = np.empty(16, dtype=np.float64)
    for i in range(16):
        out[i] = np.nan

    # Mean
    mu = 0.0
    for i in range(N): mu += x[i]
    mu /= N

    # TV, ER, PLR
    tv = 0.0
    for i in range(N - 1): tv += abs(x[i + 1] - x[i])
    net = abs(x[N - 1] - x[0])
    out[0] = tv
    out[1] = (net / tv) if tv > 0.0 else 0.0
    out[2] = (tv / net) if net > 0.0 else np.nan

    # Vol (population std)
    var = 0.0
    for i in range(N): var += (x[i] - mu) ** 2
    var /= N
    out[3] = var ** 0.5

    # RV + lr_vol
    eps = 1e-12
    rv = 0.0; lr_s = 0.0; lr_s2 = 0.0
    for i in range(N - 1):
        lr   = np.log(abs(x[i + 1]) + eps) - np.log(abs(x[i]) + eps)
        rv  += lr * lr
        lr_s += lr; lr_s2 += lr * lr
    out[4] = rv
    lr_mu  = lr_s / (N - 1)
    lr_var = lr_s2 / (N - 1) - lr_mu * lr_mu
    out[5] = lr_var ** 0.5 if lr_var > 0.0 else 0.0

    # ZCR
    zcr = 0
    for i in range(N - 1):
        si = 1 if x[i]     > mu else (-1 if x[i]     < mu else 0)
        sj = 1 if x[i + 1] > mu else (-1 if x[i + 1] < mu else 0)
        if si * sj < 0: zcr += 1
    out[6] = zcr / (N - 1)

    # CI
    xmin = x[0]; xmax = x[0]
    for i in range(1, N):
        if x[i] < xmin: xmin = x[i]
        if x[i] > xmax: xmax = x[i]
    xrng = xmax - xmin
    if xrng > 0.0 and N > 1:
        tv_s = tv if tv > 1e-12 else 1e-12
        out[7] = 100.0 * np.log10(tv_s / xrng) / np.log10(float(N))

    # Katz FD
    dk = 0.0
    for i in range(1, N):
        d = abs(x[i] - x[0])
        if d > dk: dk = d
    ns = N - 1
    if tv > 0.0 and dk > 0.0 and ns > 1:
        out[8] = np.log10(float(ns)) / (np.log10(float(ns)) + np.log10(dk / tv))

    # Higuchi FD
    out[9] = _higuchi_nb(x, 4)

    # Hurst R/S
    cs = 0.0; cs_min = 0.0; cs_max = 0.0
    for i in range(N):
        cs += x[i] - mu
        if cs < cs_min: cs_min = cs
        if cs > cs_max: cs_max = cs
    R = cs_max - cs_min
    S = out[3]
    if S > 0.0:
        out[10] = np.log(max(R / S, 1e-12)) / np.log(float(N))
        out[11] = 2.0 - out[10]

    # Permutation Entropy (m=3)
    counts = np.zeros(6, dtype=np.int64)
    n_p = N - 2
    for i in range(n_p):
        counts[_ord3(x[i], x[i + 1], x[i + 2])] += 1
    pe = 0.0
    for j in range(6):
        if counts[j] > 0:
            p = counts[j] / n_p
            pe -= p * np.log2(p)
    out[12] = pe

    # spec_ent (index 13) stays NaN — FFT not JIT-able

    # Wavelet Energy (Haar L1 + L2)
    n2 = (N // 2) * 2
    if n2 >= 4:
        s2  = 1.0 / 2.0 ** 0.5
        nh  = n2 // 2
        we1 = 0.0
        app = np.empty(nh, dtype=np.float64)
        for i in range(nh):
            det  = (x[2 * i] - x[2 * i + 1]) * s2
            we1 += det * det
            app[i] = (x[2 * i] + x[2 * i + 1]) * s2
        out[14] = we1
        n4 = (nh // 2) * 2
        if n4 >= 2:
            we2 = 0.0
            for i in range(n4 // 2):
                det2  = (app[2 * i] - app[2 * i + 1]) * s2
                we2  += det2 * det2
            out[15] = we2

    return out


@njit(parallel=True, cache=True)
def _rolling_cplx_nb(arr, window, step):
    """Parallel rolling complexity over 1D arr. Returns float64[N, 16]."""
    N       = len(arr)
    n_steps = (N - window) // step + 1
    out     = np.full((N, 16), np.nan, dtype=np.float64)
    for si in prange(n_steps):
        i   = window - 1 + si * step
        if i >= N: continue
        seg = arr[i - window + 1: i + 1]
        # NaN / constant check
        has_nan = False
        mn = seg[0]; mx = seg[0]
        for j in range(1, window):
            v = seg[j]
            if v != v:          # NaN
                has_nan = True
                break
            if v < mn: mn = v
            if v > mx: mx = v
        if has_nan or mn == mx:
            continue
        res = _cplx_nb(seg)
        for m in range(16):
            out[i, m] = res[m]
    return out


# Stable key ordering for all complexity metrics
CPLX_KEYS = ('tv', 'er', 'plr', 'vol', 'rv', 'lr_vol', 'zcr', 'ci',
             'katz_fd', 'higuchi_fd', 'hurst_rs', 'fd_hurst',
             'perm_ent', 'spec_ent', 'wav_e1', 'wav_e2')

# Polarity: True=high value→trending, False=low value→trending, None=ambiguous
CPLX_POLARITY = {
    'tv': None, 'er': True, 'plr': False, 'vol': None,
    'rv': None, 'lr_vol': None, 'zcr': False, 'ci': False,
    'katz_fd': False, 'higuchi_fd': False, 'hurst_rs': True, 'fd_hurst': False,
    'perm_ent': False, 'spec_ent': False, 'wav_e1': None, 'wav_e2': None,
}


def rolling_complexity(arr, window, step=5):
    """Rolling complexity metrics via Numba JIT + parallel. Step-and-forward-fill."""
    arr_f = np.asarray(arr, dtype=np.float64)
    raw   = _rolling_cplx_nb(arr_f, window, step)   # float64[N, 16]
    # Forward-fill (step > 1 leaves gaps)
    if step > 1:
        df_tmp = pd.DataFrame(raw).ffill()
        raw = df_tmp.values
    out = {}
    for ki, k in enumerate(CPLX_KEYS):
        out[k] = raw[:, ki]
    return out


def _agg_m5_max(arr, lo_to_m5, N5):
    """Max-aggregate sub-TF array to M5 bars. Vectorized O(N_low)."""
    valid = (lo_to_m5 >= 0) & ~np.isnan(arr)
    if not valid.any():
        return np.full(N5, np.nan)
    out = np.full(N5, -np.inf)
    np.maximum.at(out, lo_to_m5[valid], arr[valid])
    out[out == -np.inf] = np.nan
    return out


def _agg_m5_min(arr, lo_to_m5, N5):
    """Min-aggregate sub-TF array to M5 bars. Vectorized O(N_low)."""
    valid = (lo_to_m5 >= 0) & ~np.isnan(arr)
    if not valid.any():
        return np.full(N5, np.nan)
    out = np.full(N5, np.inf)
    np.minimum.at(out, lo_to_m5[valid], arr[valid])
    out[out == np.inf] = np.nan
    return out


def rolling_microstructure(df, tf_name, windows=(5, 10, 20, 60)):
    """
    Returns dict of rolling arrays for each window W:
      ppm_W, tpm_W, ppt_W, z_ppm_W, z_tpm_W, z_ppt_W
    Plus instantaneous: ppt0, tpm0, ppm0
    Plus complexity metrics for W >= 20: {metric}_{flow}_{W}
    """
    tr  = true_range_pips(df)
    vol = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(df))
    tfm = TF_MIN[tf_name]

    # Guard: avoid division by zero
    vol_safe = np.where(vol > 0, vol, np.nan)

    # Instantaneous
    ppt0 = tr / vol_safe          # pips / tick
    tpm0 = vol_safe / tfm         # ticks / min
    ppm0 = tr / tfm               # pips / min  (= ppt0 * tpm0)

    results = dict(tr=tr, vol=vol, ppt0=ppt0, tpm0=tpm0, ppm0=ppm0)

    for W in windows:
        # Sum-based rolling (consistent: ppm = ppt * tpm exactly)
        tr_sum  = pd.Series(tr).rolling(W, min_periods=W).sum().values
        vol_sum = pd.Series(vol).rolling(W, min_periods=W).sum().values

        ppm_W = tr_sum  / (W * tfm)               # total TR / total time
        tpm_W = vol_sum / (W * tfm)               # total vol / total time
        ppt_W = np.where(vol_sum > 0, tr_sum / vol_sum, np.nan)  # total TR / total vol

        # Z-score: rolling mean / std of each metric using 4× window baseline
        base = max(W * 4, 60)
        def zscore(x):
            s = pd.Series(x)
            mu = s.rolling(base, min_periods=base).mean()
            sd = s.rolling(base, min_periods=base).std()
            return ((s - mu) / sd.clip(lower=1e-9)).values

        results[f"ppm_{W}"] = ppm_W
        results[f"tpm_{W}"] = tpm_W
        results[f"ppt_{W}"] = ppt_W
        results[f"z_ppm_{W}"] = zscore(ppm_W)
        results[f"z_tpm_{W}"] = zscore(tpm_W)
        results[f"z_ppt_{W}"] = zscore(ppt_W)

    # Complexity / trendiness metrics on instantaneous series (W >= 20 only)
    cl     = df["close"].values.astype(float)
    _eps   = 1e-12
    lr_arr = np.concatenate([[np.nan], np.log(cl[1:] / (cl[:-1] + _eps))])

    # ── 8 derived MA / momentum series ───────────────────────────────────────
    _cl_s   = pd.Series(cl)
    _ema8   = _cl_s.ewm(span=8,  adjust=False).mean().values
    _ema21  = _cl_s.ewm(span=21, adjust=False).mean().values
    _ema12  = _cl_s.ewm(span=12, adjust=False).mean().values
    _ema26  = _cl_s.ewm(span=26, adjust=False).mean().values
    _atr14  = pd.Series(tr).ewm(alpha=1.0 / 14, adjust=False).mean().values
    _atr_s  = pd.Series(_atr14).rolling(20, min_periods=20).mean().values
    _sma20  = _cl_s.rolling(20, min_periods=20).mean().values
    _std20  = _cl_s.rolling(20, min_periods=20).std().values
    _macdl  = _ema12 - _ema26
    _mom10  = np.concatenate([[np.nan] * 10, cl[10:] - cl[:-10]])

    ema8r_arr  = cl / (_ema8  + _eps) - 1.0                        # price/EMA8  - 1
    ema21r_arr = cl / (_ema21 + _eps) - 1.0                        # price/EMA21 - 1
    madist_arr = np.where(_atr14 > 0, (_ema8 - _ema21) / _atr14, np.nan)   # (EMA8-EMA21)/ATR
    macdh_arr  = _macdl - pd.Series(_macdl).ewm(span=9, adjust=False).mean().values
    mom10_arr  = np.where(_atr14 > 0, _mom10 / _atr14, np.nan)     # 10-bar momentum / ATR
    roc10_arr  = np.concatenate([[np.nan] * 10,
                                  (cl[10:] - cl[:-10]) / (cl[:-10] + _eps)])  # 10-bar ROC
    atrr_arr   = np.where(_atr_s > 0, _atr14 / _atr_s, np.nan)    # ATR / SMA(ATR)
    bbpos_arr  = np.where(_std20 > 0, (cl - _sma20) / (2.0 * _std20), np.nan)  # BB position

    # ── 3 direct volume series ────────────────────────────────────────────────
    _vol_ma20  = pd.Series(vol).rolling(20, min_periods=1).mean().values
    rvol_arr   = np.where(_vol_ma20 > 0, vol / _vol_ma20, np.nan)  # bar_vol / 20-bar mean vol
    _sign_arr  = np.sign(cl - df["open"].values.astype(float))
    obvr_arr   = _sign_arr * rvol_arr                               # signed rvol (OBV rate)
    # vol0: raw ticks-per-bar normalised by 20-bar mean (= rvol without tfm division)
    results["rvol0"] = rvol_arr
    results["obvr0"] = obvr_arr

    _cx_step = max(1, len(tr) // 15_000)
    for W in [w for w in windows if w >= 20]:
        for mname, raw in [
            ("ppm",    ppm0),
            ("tpm",    tpm0),
            ("ppt",    ppt0),
            ("close",  cl),
            ("logret", lr_arr),
            ("ema8r",  ema8r_arr),
            ("ema21r", ema21r_arr),
            ("madist", madist_arr),
            ("macdh",  macdh_arr),
            ("mom10",  mom10_arr),
            ("roc10",  roc10_arr),
            ("atrr",   atrr_arr),
            ("bbpos",  bbpos_arr),
            ("rvol",   rvol_arr),
            ("obvr",   obvr_arr),
        ]:
            cx = rolling_complexity(raw, W, step=_cx_step)
            for k, v in cx.items():
                results[f"{k}_{mname}_{W}"] = v

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Distribution + autocorrelation
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_distributions(name, ms, windows):
    print(f"\n{'─'*70}")
    print(f"  [{name}] METRIC DISTRIBUTION & AUTOCORRELATION")
    print(f"{'─'*70}")
    print(f"  {'Metric':<12} {'W':>4}  {'mean':>7}  {'P50':>7}  {'P90':>7}  {'P99':>7}  "
          f"{'skew':>6}  {'AC1':>6}  {'AC5':>6}")
    for W in windows:
        for mname in ("ppm", "tpm", "ppt"):
            arr = ms[f"{mname}_{W}"]
            v   = arr[~np.isnan(arr)]
            if len(v) < 10: continue
            ac1 = pd.Series(v).autocorr(lag=1)
            ac5 = pd.Series(v).autocorr(lag=5)
            sk  = scipy_stats.skew(v)
            p50, p90, p99 = np.percentile(v, [50, 90, 99])
            print(f"  {mname:<12} {W:>4}  {v.mean():>7.3f}  {p50:>7.3f}  {p90:>7.3f}  "
                  f"{p99:>7.3f}  {sk:>6.2f}  {ac1:>6.3f}  {ac5:>6.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Predictive power: metric quintile → P(big M5 within next N M5 bars)
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_predictive(name, ms_low, df_low, df_m5, big_m5, sec_low, windows, horizons=(1,3)):
    print(f"\n{'─'*70}")
    print(f"  [{name}→M5] PREDICTIVE POWER: metric quintile → P(big M5 in next N bars)")
    print(f"{'─'*70}")

    # For each low-TF bar, find its M5 bar index
    lo_to_m5 = align_low_to_high(df_low, df_m5, 300)
    N5 = len(df_m5)

    for W in windows:
        for mname in ("ppt", "tpm", "ppm"):
            arr = ms_low[f"z_{mname}_{W}"]
            valid = (lo_to_m5 >= 0) & ~np.isnan(arr)
            if valid.sum() < 100:
                continue

            # Assign quintile to each low-TF bar
            vals = arr[valid]
            q_edges = np.percentile(vals, [0, 20, 40, 60, 80, 100])
            q_idx   = np.digitize(vals, q_edges[1:-1])  # 0..4

            m5_idx  = lo_to_m5[valid]

            print(f"\n  z_{mname} W={W}:")
            print(f"  {'Quintile':<12}", end="")
            for H in horizons:
                print(f"  P(big<=+{H}M5)%", end="")
            print(f"  {'n':>6}")

            for qi in range(5):
                qmask = q_idx == qi
                if qmask.sum() < 5:
                    continue
                mi5 = m5_idx[qmask]
                row_n = qmask.sum()
                parts = [f"  Q{qi+1}({q_edges[qi]:.1f}–{q_edges[qi+1]:.1f})", ""]
                hits = []
                for H in horizons:
                    # Does any M5 bar in [m5_idx, m5_idx+H] have big_m5?
                    hit = 0
                    for i in mi5:
                        end = min(i + H + 1, N5)
                        if big_m5[i:end].any():
                            hit += 1
                    hits.append(100 * hit / row_n)
                qlabel = f"Q{qi+1}({q_edges[qi]:.1f}→{q_edges[qi+1]:.1f})"
                print(f"  {qlabel:<25}", end="")
                for h in hits:
                    bar = "█" * int(h / 2)
                    print(f"  {h:5.1f}% {bar:<20}", end="")
                print(f"  {row_n:>6}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Event study: mean metric profile around big M5 events
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_event_study(name, ms_low, df_low, df_m5, big_m5, sec_low, windows):
    print(f"\n{'─'*70}")
    print(f"  [{name}→M5] EVENT STUDY: mean metric in [-40..+10 {name} bars] around big M5")
    print(f"{'─'*70}")

    # Timestamps of big M5 bar OPEN (we want to detect BEFORE the M5 bar closes)
    big_m5_opens = df_m5["timestamp"].values[big_m5].astype(np.int64)
    lo_ts = df_low["timestamp"].values.astype(np.int64)
    N_lo  = len(df_low)
    N_ev  = len(big_m5_opens)
    if N_ev < 5:
        print("  Too few events."); return

    PRE, POST = 40, 10  # low-TF bars before/after M5 open

    for W in [10, 20]:  # focus on these two windows
        for mname in ("ppt", "tpm", "ppm"):
            key = f"z_{mname}_{W}"
            arr = ms_low[key]
            profiles = []

            for ev_ts in big_m5_opens:
                # Find last low-TF bar before this M5 open
                pos = np.searchsorted(lo_ts, ev_ts, side="right") - 1
                if pos < PRE or pos + POST >= N_lo:
                    continue
                window_arr = arr[pos - PRE: pos + POST + 1]
                if np.isnan(window_arr).mean() > 0.3:
                    continue
                profiles.append(window_arr)

            if len(profiles) < 5:
                continue
            profiles = np.array(profiles)
            mean_profile = np.nanmean(profiles, axis=0)
            lags = list(range(-PRE, POST+1))

            print(f"\n  z_{mname} W={W}  n_events={len(profiles)}")
            print(f"  Lag  Mean-z  Bar")
            for lag, val in zip(lags, mean_profile):
                if np.isnan(val): continue
                bar_len = int(abs(val) * 6)
                bar = ("█" * bar_len) if val >= 0 else ("░" * bar_len)
                mark = " ◄ M5 OPEN" if lag == 0 else ("        " if lag % 5 != 0 else f" [lag{lag:+d}]")
                print(f"  {lag:+4d}  {val:+6.3f}  {bar}{mark}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Lead/lag scan: at what lag does S30 z-ppt best predict big M5?
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_lead_lag(name, ms_low, df_low, df_m5, big_m5, sec_low, W=20, max_lag=60):
    print(f"\n{'─'*70}")
    print(f"  [{name}→M5] LEAD/LAG SCAN: z_ppt W={W} vs big-M5 signal")
    print(f"{'─'*70}")

    lo_to_m5 = align_low_to_high(df_low, df_m5, 300)
    N5 = len(df_m5)
    arr = ms_low[f"z_ppt_{W}"]

    # Build per-M5-bar: max z_ppt in the bar
    m5_max_zppt = np.full(N5, np.nan)
    valid = (lo_to_m5 >= 0) & ~np.isnan(arr)
    for i in range(N5):
        in_bar = valid & (lo_to_m5 == i)
        if in_bar.any():
            m5_max_zppt[i] = arr[in_bar].max()

    # Scan lags: pearson corr between m5_max_zppt[t-lag] and big_m5[t]
    print(f"  {'Lag':>5}  {'Pearson r':>10}  {'p-val':>8}  Bar")
    best_r, best_lag = 0, 0
    for lag in range(0, max_lag+1, 5):
        n = N5 - lag
        x = m5_max_zppt[:n]
        y = big_m5[lag:lag+n].astype(float)
        valid_both = ~np.isnan(x)
        if valid_both.sum() < 50:
            continue
        r, p = scipy_stats.pearsonr(x[valid_both], y[valid_both])
        bar_len = max(0, int(abs(r) * 300))
        bar = ("█" * bar_len) if r > 0 else ("░" * bar_len)
        sig = "🟢" if p < 0.05 else ("🟡" if p < 0.15 else "  ")
        print(f"  {lag:>5}  {r:>+10.4f}  {p:>8.4f}  {bar} {sig}")
        if abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    print(f"\n  Best: lag={best_lag} bars  r={best_r:+.4f}  "
          f"({best_lag * TF_MIN[name]:.0f} min before M5 bar)")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ASCII time-series sparkline (last 200 bars of each TF)
# ═══════════════════════════════════════════════════════════════════════════════

SPARK_CHARS = " ▁▂▃▄▅▆▇█"

def sparkline(arr, width=80):
    v = arr[-width:].copy()
    v = np.where(np.isnan(v), 0, v)
    lo, hi = v.min(), v.max()
    if hi == lo: return "─" * len(v)
    idx = ((v - lo) / (hi - lo) * (len(SPARK_CHARS) - 1)).astype(int)
    idx = np.clip(idx, 0, len(SPARK_CHARS)-1)
    return "".join(SPARK_CHARS[i] for i in idx)

def analysis_sparklines(name, ms, df, big_flag, windows=(10, 20), tail=100):
    print(f"\n{'─'*70}")
    print(f"  [{name}] RECENT TIME-SERIES SPARKLINES (last {tail} bars)")
    print(f"{'─'*70}")
    print(f"  Legend: ▁=low  █=high  | = big-bar event\n")

    # Big-bar markers
    big_mark = np.where(big_flag[-tail:], "|", "·")

    for W in windows:
        for mname in ("ppt", "tpm", "ppm"):
            key = f"z_{mname}_{W}"
            arr = ms[key]
            spark = sparkline(arr, tail)
            # Overlay big-bar markers
            overlay = list(spark)
            for i, m in enumerate(big_mark):
                if m == "|":
                    overlay[i] = "│"
            print(f"  z_{mname} W={W:>2}:  {''.join(overlay)}")
        print()

    # Show the last 20 bars in detail
    print(f"  Last 20 bars detail (z-scores, W=20):")
    print(f"  {'Bar':>5}  {'z_ppt':>7}  {'z_tpm':>7}  {'z_ppm':>7}  {'big':>4}  {'timestamp'}")
    arr_ppt = ms[f"z_ppt_20"][-20:]
    arr_tpm = ms[f"z_tpm_20"][-20:]
    arr_ppm = ms[f"z_ppm_20"][-20:]
    flag20  = big_flag[-20:]
    ts20    = df["timestamp"].values[-20:]
    for i in range(20):
        ts_str = pd.Timestamp(ts20[i]).strftime("%m-%d %H:%M")
        b = "🔴" if flag20[i] else "   "
        def fv(x): return f"{x:+7.2f}" if not np.isnan(x) else "    n/a"
        print(f"  {i-19:>5}  {fv(arr_ppt[i])}  {fv(arr_tpm[i])}  {fv(arr_ppm[i])}  {b}  {ts_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Composite signal: z_ppt AND z_tpm both elevated → P(big M5)
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_composite(name, ms_low, df_low, df_m5, big_m5, W=20, thresh_z=1.5):
    print(f"\n{'─'*70}")
    print(f"  [{name}→M5] COMPOSITE: (z_ppt>{thresh_z} AND z_tpm>{thresh_z}) → P(big M5)")
    print(f"  Window W={W}")
    print(f"{'─'*70}")

    lo_to_m5 = align_low_to_high(df_low, df_m5, 300)
    N5 = len(df_m5)

    z_ppt = ms_low[f"z_ppt_{W}"]
    z_tpm = ms_low[f"z_tpm_{W}"]
    z_ppm = ms_low[f"z_ppm_{W}"]

    # Per-M5 bar: max of each z-score among all sub-bars
    def max_in_m5(arr):
        out = np.full(N5, np.nan)
        valid = (lo_to_m5 >= 0) & ~np.isnan(arr)
        for i in range(N5):
            mask = valid & (lo_to_m5 == i)
            if mask.any():
                out[i] = arr[mask].max()
        return out

    mz_ppt = max_in_m5(z_ppt)
    mz_tpm = max_in_m5(z_tpm)
    mz_ppm = max_in_m5(z_ppm)

    valid = ~np.isnan(mz_ppt) & ~np.isnan(mz_tpm)
    N_valid = valid.sum()

    signals = {
        "baseline":              np.ones(N5, bool),
        f"z_ppt>{thresh_z}":     mz_ppt > thresh_z,
        f"z_tpm>{thresh_z}":     mz_tpm > thresh_z,
        f"z_ppm>{thresh_z}":     mz_ppm > thresh_z,
        f"ppt+tpm>{thresh_z}":   (mz_ppt > thresh_z) & (mz_tpm > thresh_z),
        f"ppt+tpm+ppm>{thresh_z}": (mz_ppt > thresh_z) & (mz_tpm > thresh_z) & (mz_ppm > thresh_z),
    }

    print(f"\n  Forward returns (directional, in M5 direction):")
    print(f"  {'Signal':<28}   n   P(big_same_M5)  lag+1M5_mean  lag+5M5_mean")
    cl5  = df_m5["close"].values.astype(float)
    dir5 = np.sign(df_m5["close"].values - df_m5["open"].values).astype(float)

    for sname, smask in signals.items():
        n = smask.sum()
        if n < 5: continue
        p_big_same = (smask & big_m5).sum() / n * 100

        # Forward directional return
        def fwd_dir(lag):
            ok = smask & (np.arange(N5) < N5 - lag)
            idx = np.where(ok)[0]
            fwd = (cl5[idx + lag] - cl5[idx]) / PIP
            return dir5[idx] * fwd

        fwd1 = fwd_dir(1); fwd5 = fwd_dir(5)
        m1 = fwd1.mean() if len(fwd1)>0 else 0
        m5 = fwd5.mean() if len(fwd5)>0 else 0
        mark1 = "🟢" if m1 > 0.1 else ("🔴" if m1 < -0.1 else "─")
        mark5 = "🟢" if m5 > 0.5 else ("🔴" if m5 < -0.5 else "─")
        print(f"  {sname:<28}  {n:>5}  {p_big_same:>13.1f}%  "
              f"{m1:>+9.2f}p {mark1}  {m5:>+9.2f}p {mark5}")

    # Threshold sweep
    print(f"\n  Threshold sweep for ppt+tpm composite:")
    print(f"  {'Z_thresh':>8}  {'n':>6}  {'P(big)%':>8}  {'fwd+1':>8}  {'fwd+5':>8}")
    for z in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        mask = (mz_ppt > z) & (mz_tpm > z)
        n = mask.sum()
        if n < 3: continue
        p_big = (mask & big_m5).sum() / n * 100
        ok1 = mask & (np.arange(N5) < N5-1)
        idx1 = np.where(ok1)[0]
        f1 = (dir5[idx1] * (cl5[idx1+1]-cl5[idx1])/PIP).mean() if len(idx1)>0 else 0
        ok5 = mask & (np.arange(N5) < N5-5)
        idx5 = np.where(ok5)[0]
        f5 = (dir5[idx5] * (cl5[idx5+5]-cl5[idx5])/PIP).mean() if len(idx5)>0 else 0
        print(f"  {z:>8.1f}  {n:>6}  {p_big:>7.1f}%  {f1:>+7.2f}p  {f5:>+7.2f}p")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Complexity / Fractal metrics on ppm / tpm / ppt
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_complexity_metrics(name, ms_low, df_low, df_m5, big_m5, W=20):
    """
    Part 7: Rolling complexity / fractal metrics on ppm, tpm, ppt.
    Tests which measure(s) best predict big M5 bars.

    Polarity — high=trending: ER, Hurst-RS
               low=trending : PLR, ZCR, CI, Katz-FD, Higuchi-FD,
                              FD-Hurst, Perm-Ent, Spec-Ent
    """
    print(f"\n{'─'*70}")
    print(f"  [{name}→M5] COMPLEXITY / FRACTAL METRICS  W={W}")
    print(f"  Polarity — high=trending: ER, Hurst-RS")
    print(f"             low=trending : PLR, ZCR, CI, Katz-FD, Higuchi-FD,")
    print(f"                           FD-Hurst, Perm-Ent, Spec-Ent")
    print(f"{'─'*70}")

    lo_to_m5 = align_low_to_high(df_low, df_m5, 300)
    N5   = len(df_m5)
    cl5  = df_m5["close"].values.astype(float)
    dir5 = np.sign(df_m5["close"].values - df_m5["open"].values).astype(float)

    ALL_SERIES = ("tpm", "ppm", "ppt", "close", "logret",
                  "ema8r", "ema21r", "madist", "macdh",
                  "mom10", "roc10", "atrr", "bbpos",
                  "rvol", "obvr")

    # Pre-aggregate all metrics from sub-TF bars to M5 bars (vectorized)
    m5_cx: dict = {}
    for mname in ALL_SERIES:
        for k in CPLX_KEYS:
            key = f"{k}_{mname}_{W}"
            if key not in ms_low:
                continue
            pol = CPLX_POLARITY.get(k)
            if pol is False:    # low=trending → take min per M5 bar
                m5_cx[key] = _agg_m5_min(ms_low[key], lo_to_m5, N5)
            else:               # high=trending or neutral → take max
                m5_cx[key] = _agg_m5_max(ms_low[key], lo_to_m5, N5)

    # ── A. Pearson |r| with big_m5 flag, sorted descending ───────────────────
    print(f"\n  A. Pearson correlation with big_m5 flag — sorted by |r|:")
    print(f"  {'Metric':<14} {'Flow':>7}  {'r':>8}  {'p':>8}  bar")

    ranked = []
    for mname in ALL_SERIES:
        for k in CPLX_KEYS:
            key = f"{k}_{mname}_{W}"
            arr = m5_cx.get(key)
            if arr is None:
                continue
            both = ~np.isnan(arr)
            if both.sum() < 30:
                continue
            r, p = scipy_stats.pearsonr(arr[both], big_m5[both].astype(float))
            ranked.append((abs(r), r, p, k, mname, key))
    ranked.sort(reverse=True)

    for absr, r, p, k, mname, key in ranked:
        sig  = "🟢" if p < 0.05 else ("🟡" if p < 0.15 else "  ")
        pol  = CPLX_POLARITY.get(k)
        pols = "(↑tr)" if pol is True else ("(↓tr)" if pol is False else "     ")
        bar  = "█" * min(int(absr * 250), 35)
        print(f"  {k:<14} {mname:>7}  {r:>+8.4f}  {p:>8.4f}  {bar} {pols} {sig}")

    # ── B. Quintile → P(big M5) + fwd returns for top-3 ─────────────────────
    print(f"\n  B. Quintile → P(big M5) + fwd returns — top-3 predictors:")
    for absr, r, p, k, mname, key in ranked[:3]:
        arr  = m5_cx[key]
        both = ~np.isnan(arr)
        vals    = arr[both]
        big_sub = big_m5[both]
        idx_all = np.where(both)[0]
        q_edges = np.percentile(vals, [0, 20, 40, 60, 80, 100])
        q_idx   = np.digitize(vals, q_edges[1:-1])
        pol  = CPLX_POLARITY.get(k)
        note = "(Q1=trending)" if pol is False else ("(Q5=trending)" if pol is True else "")

        print(f"\n  {key}  r={r:+.4f}  {note}")
        print(f"  {'Quintile':<34}  P(big)%  fwd+1p  fwd+5p    n")
        for qi in range(5):
            qmask = q_idx == qi
            n = qmask.sum()
            if n < 5:
                continue
            p_big = big_sub[qmask].mean() * 100
            gi    = idx_all[qmask]
            ok1   = gi[gi < N5 - 1]
            f1    = (dir5[ok1] * (cl5[ok1 + 1] - cl5[ok1]) / PIP).mean() if len(ok1) > 0 else 0.0
            ok5   = gi[gi < N5 - 5]
            f5    = (dir5[ok5] * (cl5[ok5 + 5] - cl5[ok5]) / PIP).mean() if len(ok5) > 0 else 0.0
            m1    = "🟢" if f1 > 0.1 else ("🔴" if f1 < -0.1 else "─")
            m5s   = "🟢" if f5 > 0.5 else ("🔴" if f5 < -0.5 else "─")
            ql    = f"Q{qi+1}({q_edges[qi]:.3f}→{q_edges[qi+1]:.3f})"
            print(f"  {ql:<36}  {p_big:>6.1f}%  {f1:>+5.2f}p{m1}  {f5:>+5.2f}p{m5s}  {n:>5}")

    # ── C. Composite: ER_tpm high AND Higuchi-FD_tpm low ─────────────────────
    er_key  = f"er_tpm_{W}"
    hfd_key = f"higuchi_fd_tpm_{W}"
    if er_key in m5_cx and hfd_key in m5_cx:
        print(f"\n  C. Composite: er_tpm > er_thr  AND  higuchi_fd_tpm < fd_thr:")
        print(f"  {'er_thr':>7}  {'fd_thr':>7}  {'n':>6}  {'P(big)%':>8}  "
              f"{'fwd+1':>8}  {'fwd+5':>8}  {'freq%':>7}")
        er5  = m5_cx[er_key]
        hfd5 = m5_cx[hfd_key]
        base = (~np.isnan(er5) & ~np.isnan(hfd5)).sum()
        for er_thr in (0.20, 0.30, 0.40, 0.50):
            for fd_thr in (1.30, 1.50, 1.70):
                mask = (er5 > er_thr) & (hfd5 < fd_thr) & ~np.isnan(er5) & ~np.isnan(hfd5)
                n = mask.sum()
                if n < 3:
                    continue
                p_big = (mask & big_m5).sum() / n * 100
                freq  = n / base * 100
                ok1   = mask & (np.arange(N5) < N5 - 1)
                i1    = np.where(ok1)[0]
                f1    = (dir5[i1] * (cl5[i1 + 1] - cl5[i1]) / PIP).mean() if len(i1) > 0 else 0.0
                ok5   = mask & (np.arange(N5) < N5 - 5)
                i5    = np.where(ok5)[0]
                f5    = (dir5[i5] * (cl5[i5 + 5] - cl5[i5]) / PIP).mean() if len(i5) > 0 else 0.0
                m1    = "🟢" if f1 > 0.1 else ("🔴" if f1 < -0.1 else "─")
                m5s   = "🟢" if f5 > 0.5 else ("🔴" if f5 < -0.5 else "─")
                print(f"  {er_thr:>7.2f}  {fd_thr:>7.2f}  {n:>6}  {p_big:>7.1f}%  "
                      f"{f1:>+7.2f}p{m1}  {f5:>+7.2f}p{m5s}  {freq:>6.1f}%")
    else:
        print(f"\n  C. [skipped — er_tpm_{W} or higuchi_fd_tpm_{W} not in ms]")

    # ── D. Price-regime composite: ER_close high AND Higuchi-FD_close low ────
    er_cl_key  = f"er_close_{W}"
    hfd_cl_key = f"higuchi_fd_close_{W}"
    if er_cl_key in m5_cx and hfd_cl_key in m5_cx:
        print(f"\n  D. Price-regime composite: er_close > er_thr  AND  higuchi_fd_close < fd_thr")
        print(f"     (Kaufman ER + Higuchi FD on price — trending price regime gate)")
        print(f"  {'er_thr':>7}  {'fd_thr':>7}  {'n':>6}  {'P(big)%':>8}  "
              f"{'fwd+1':>8}  {'fwd+5':>8}  {'freq%':>7}")
        er_cl  = m5_cx[er_cl_key]
        hfd_cl = m5_cx[hfd_cl_key]
        base   = (~np.isnan(er_cl) & ~np.isnan(hfd_cl)).sum()
        for er_thr in (0.20, 0.30, 0.40, 0.50):
            for fd_thr in (1.30, 1.50, 1.70):
                mask = (er_cl > er_thr) & (hfd_cl < fd_thr) & ~np.isnan(er_cl) & ~np.isnan(hfd_cl)
                n = mask.sum()
                if n < 3:
                    continue
                p_big = (mask & big_m5).sum() / n * 100
                freq  = n / base * 100
                ok1   = mask & (np.arange(N5) < N5 - 1)
                i1    = np.where(ok1)[0]
                f1    = (dir5[i1] * (cl5[i1 + 1] - cl5[i1]) / PIP).mean() if len(i1) > 0 else 0.0
                ok5   = mask & (np.arange(N5) < N5 - 5)
                i5    = np.where(ok5)[0]
                f5    = (dir5[i5] * (cl5[i5 + 5] - cl5[i5]) / PIP).mean() if len(i5) > 0 else 0.0
                m1    = "🟢" if f1 > 0.1 else ("🔴" if f1 < -0.1 else "─")
                m5s   = "🟢" if f5 > 0.5 else ("🔴" if f5 < -0.5 else "─")
                print(f"  {er_thr:>7.2f}  {fd_thr:>7.2f}  {n:>6}  {p_big:>7.1f}%  "
                      f"{f1:>+7.2f}p{m1}  {f5:>+7.2f}p{m5s}  {freq:>6.1f}%")
    else:
        print(f"\n  D. [skipped — er_close_{W} or higuchi_fd_close_{W} not in ms]")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Feature selection: Bonferroni → dedup → LightGBM → walk-forward stability
# ═══════════════════════════════════════════════════════════════════════════════

_ALL_SERIES_NAMES = ("tpm", "ppm", "ppt", "close", "logret",
                     "ema8r", "ema21r", "madist", "macdh",
                     "mom10", "roc10", "atrr", "bbpos",
                     "rvol", "obvr")


def analysis_feature_selection(name, ms_low, df_low, df_m5, big_m5, W=20,
                                min_r=0.05, dedup_thresh=0.85):
    """
    Part 8: Multi-stage feature selection across all complexity metrics.
    8A — Bonferroni-corrected significance filter (|r|>min_r AND p<p_bonf)
    8B — Greedy pairwise-corr deduplication (drop if |r_pair|>dedup_thresh)
    8C — LightGBM feature importance on deduplicated set
    8D — 3-fold walk-forward stability check
    """
    print(f"\n{'─'*70}")
    print(f"  [{name}→M5] FEATURE SELECTION  W={W}  "
          f"min_r={min_r}  dedup={dedup_thresh}")
    print(f"{'─'*70}")

    lo_to_m5 = align_low_to_high(df_low, df_m5, 300)
    N5   = len(df_m5)
    cl5  = df_m5["close"].values.astype(float)
    dir5 = np.sign(df_m5["close"].values - df_m5["open"].values).astype(float)

    # Pre-aggregate all series × metrics to M5 bars (vectorized)
    m5_all: dict = {}
    for mname in _ALL_SERIES_NAMES:
        for k in CPLX_KEYS:
            key = f"{k}_{mname}_{W}"
            if key not in ms_low:
                continue
            pol = CPLX_POLARITY.get(k)
            if pol is False:
                m5_all[key] = _agg_m5_min(ms_low[key], lo_to_m5, N5)
            else:
                m5_all[key] = _agg_m5_max(ms_low[key], lo_to_m5, N5)

    n_total = len(m5_all)
    p_bonf  = 0.05 / max(n_total, 1)

    # ── 8A. Significance filter ───────────────────────────────────────────────
    print(f"\n  8A. Significance filter  "
          f"(n_total={n_total}  Bonferroni p<{p_bonf:.1e}  |r|>{min_r}):")

    ranked = []
    for key, arr in m5_all.items():
        both = ~np.isnan(arr)
        if both.sum() < 30 or big_m5[both].sum() < 2:
            continue
        try:
            r, p = scipy_stats.pearsonr(arr[both], big_m5[both].astype(float))
        except ValueError:
            continue
        ranked.append((abs(r), r, p, key, arr))
    ranked.sort(reverse=True)

    survivors = [(ar, r, p, k, a) for ar, r, p, k, a in ranked
                 if ar >= min_r and p < p_bonf]
    print(f"  Survivors: {len(survivors)} / {len(ranked)}")

    print(f"\n  {'Key':<28}  {'r':>8}  {'p':>10}  bar")
    for absr, r, p, key, arr in survivors[:40]:
        bar = "█" * min(int(absr * 200), 28)
        print(f"  {key:<28}  {r:>+8.4f}  {p:>10.2e}  {bar}")
    if len(survivors) > 40:
        print(f"  … {len(survivors) - 40} more survivors")

    if not survivors:
        print("  No survivors — relax thresholds.")
        return [], []

    # ── 8B. Greedy deduplication ──────────────────────────────────────────────
    print(f"\n  8B. Greedy deduplication  (|r_pair|>{dedup_thresh} = redundant):")

    kept: list = []
    n_dropped = 0
    for candidate in survivors:
        absr_c, r_c, p_c, key_c, arr_c = candidate
        both_c = ~np.isnan(arr_c)
        redundant = False
        for _, _, _, _, arr_k in kept:
            both_ck = both_c & ~np.isnan(arr_k)
            if both_ck.sum() < 20:
                continue
            r_pair, _ = scipy_stats.pearsonr(arr_c[both_ck], arr_k[both_ck])
            if abs(r_pair) >= dedup_thresh:
                redundant = True
                break
        if redundant:
            n_dropped += 1
        else:
            kept.append(candidate)

    print(f"  Independent signals: {len(kept)}  "
          f"(dropped {n_dropped} redundant)")
    print(f"\n  {'Key':<28}  {'r_big_m5':>9}")
    for _, r, _, key, _ in kept:
        print(f"  {key:<28}  {r:>+9.4f}")

    # ── 8C. LightGBM feature importance ──────────────────────────────────────
    print(f"\n  8C. LightGBM feature importance → big_m5:")
    try:
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        feat_names = [key for _, _, _, key, _ in kept]
        arrs       = [arr for _, _, _, _, arr in kept]
        X = np.column_stack(arrs).astype(np.float32)
        y = big_m5.astype(np.int32)

        # Impute NaN with per-feature median
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                med = float(np.nanmedian(X[:, j]))
                X[mask, j] = med if not np.isnan(med) else 0.0

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        spw   = n_neg / max(n_pos, 1)

        params = dict(
            objective="binary", metric="auc", verbosity=-1,
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=spw, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=5, min_child_samples=20,
        )

        cv_splits = StratifiedKFold(n_splits=3, shuffle=False)
        importances = np.zeros(len(feat_names))
        aucs: list = []
        for tr_idx, va_idx in cv_splits.split(X, y):
            mdl = lgb.LGBMClassifier(**params)
            mdl.fit(X[tr_idx], y[tr_idx],
                    eval_set=[(X[va_idx], y[va_idx])],
                    callbacks=[lgb.early_stopping(30, verbose=False),
                                lgb.log_evaluation(-1)])
            aucs.append(roc_auc_score(y[va_idx],
                                      mdl.predict_proba(X[va_idx])[:, 1]))
            importances += mdl.booster_.feature_importance(importance_type="gain")

        print(f"  CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
              f"(folds: {', '.join(f'{a:.4f}' for a in aucs)})")

        imp_order = np.argsort(importances)[::-1]
        best_imp  = importances[imp_order[0]]
        print(f"\n  {'Feature':<28}  {'gain':>9}  {'rel%':>6}  bar")
        for i in imp_order:
            if importances[i] < 0.01:
                continue
            rel = importances[i] / best_imp * 100
            bar = "█" * int(rel / 100 * 30)
            print(f"  {feat_names[i]:<28}  {importances[i]:>9.1f}  "
                  f"{rel:>5.1f}%  {bar}")

        lgb_order = [feat_names[i] for i in imp_order if importances[i] >= 0.01]

    except ImportError:
        print("  [skipped — lightgbm not installed]")
        lgb_order = []
    except Exception as e:
        print(f"  [error: {e}]")
        lgb_order = []

    # ── 8D. Walk-forward stability ────────────────────────────────────────────
    print(f"\n  8D. 3-fold walk-forward stability:")
    fold_n = N5 // 3
    folds  = [(0, fold_n), (fold_n, 2 * fold_n), (2 * fold_n, N5)]

    print(f"  {'Key':<28}  {'F1':>7}  {'F2':>7}  {'F3':>7}  stable?")
    stable_keys = []
    for _, r_full, _, key, arr in kept:
        fold_rs = []
        for f0, f1 in folds:
            sub  = arr[f0:f1]
            bsub = big_m5[f0:f1]
            v    = ~np.isnan(sub)
            if v.sum() < 20:
                fold_rs.append(np.nan)
            else:
                fr, _ = scipy_stats.pearsonr(sub[v], bsub[v].astype(float))
                fold_rs.append(fr)

        valid_signs = [np.sign(fr) for fr in fold_rs if not np.isnan(fr)]
        # Stable = all folds agree on sign AND same sign as full-period r
        stable = (len(valid_signs) >= 2
                  and len(set(valid_signs)) == 1
                  and valid_signs[0] == np.sign(r_full))
        mark = "🟢" if stable else "🔴"
        if stable:
            stable_keys.append(key)

        rs = "  ".join(f"{fr:>+6.3f}" if not np.isnan(fr) else "   n/a"
                       for fr in fold_rs)
        print(f"  {key:<28}  {rs}  {mark}")

    print(f"\n  Stable signals: {len(stable_keys)} / {len(kept)}")
    if stable_keys:
        print(f"  → {', '.join(stable_keys)}")

    return kept, stable_keys


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    WINDOWS = (5, 10, 20, 60)
    REF_DAYS = 90
    SIGMA = 3.0

    print(f"\n{'═'*70}")
    print(f"  EUR/USD Rolling Microstructure: pips/tick · ticks/min · pips/min")
    print(f"  Windows: {WINDOWS}   ref_days={REF_DAYS}   sigma={SIGMA}")
    print(f"{'═'*70}\n")

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=REF_DAYS)

    print("Loading data …")
    df_m5  = load_mid(ROOT / "data" / "m5_ba"    / "EUR_USD_M5_BA.parquet")
    df_m1  = load_mid(ROOT / "data" / "m1_ohlc"  / "EUR_USD_M1_BA.parquet")
    df_s30 = load_mid(ROOT / "data" / "s30_ohlc" / "EUR_USD_S30_BA.parquet")

    df_m5  = df_m5 [df_m5 ["timestamp"] >= cutoff].reset_index(drop=True)
    df_m1  = df_m1 [df_m1 ["timestamp"] >= cutoff].reset_index(drop=True)
    df_s30 = df_s30[df_s30["timestamp"] >= cutoff].reset_index(drop=True)
    print(f"  M5: {len(df_m5):,}  M1: {len(df_m1):,}  S30: {len(df_s30):,}")

    # Big-bar flags
    def big_mask(df, sigma=SIGMA):
        tr = true_range_pips(df)
        thr = tr.mean() + sigma*tr.std()
        return tr >= thr, thr

    big_m5,  thr5  = big_mask(df_m5)
    big_m1,  thr1  = big_mask(df_m1)
    big_s30, thr30 = big_mask(df_s30)
    print(f"  Thresholds: M5={thr5:.2f}p  M1={thr1:.2f}p  S30={thr30:.2f}p\n")

    # Compute rolling microstructure + complexity metrics for each TF
    print("Computing rolling microstructure + complexity metrics …")
    ms_m5  = rolling_microstructure(df_m5,  "M5",  WINDOWS); print("  M5  done.")
    ms_m1  = rolling_microstructure(df_m1,  "M1",  WINDOWS); print("  M1  done.")
    ms_s30 = rolling_microstructure(df_s30, "S30", WINDOWS); print("  S30 done.")
    print("  All done.\n")

    # ── Part 1: Distributions ────────────────────────────────────────────────
    print("═"*70)
    print("  PART 1 — DISTRIBUTION & AUTOCORRELATION")
    print("═"*70)
    for (nm, ms) in [("S30", ms_s30), ("M1", ms_m1), ("M5", ms_m5)]:
        analysis_distributions(nm, ms, WINDOWS)

    # ── Part 2: Predictive power ─────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  PART 2 — PREDICTIVE POWER: quintile → P(big M5)")
    print("═"*70)
    # Only W=20 to keep output manageable
    for (nm, ms, df, sec) in [("S30", ms_s30, df_s30, 30), ("M1", ms_m1, df_m1, 60)]:
        analysis_predictive(nm, ms, df, df_m5, big_m5, sec, windows=(20,), horizons=(1, 3))

    # ── Part 3: Event study ──────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  PART 3 — EVENT STUDY: metric profile around big M5")
    print("═"*70)
    for (nm, ms, df, sec) in [("S30", ms_s30, df_s30, 30), ("M1", ms_m1, df_m1, 60)]:
        analysis_event_study(nm, ms, df, df_m5, big_m5, sec, WINDOWS)

    # ── Part 4: Lead/lag scan ────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  PART 4 — LEAD/LAG SCAN: S30 z_ppt → big M5")
    print("═"*70)
    analysis_lead_lag("S30", ms_s30, df_s30, df_m5, big_m5, 30, W=20, max_lag=60)
    analysis_lead_lag("M1",  ms_m1,  df_m1,  df_m5, big_m5, 60, W=20, max_lag=20)

    # ── Part 5: Sparklines ───────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  PART 5 — SPARKLINES (recent time series)")
    print("═"*70)
    analysis_sparklines("S30", ms_s30, df_s30, big_s30, windows=(10, 20), tail=120)
    analysis_sparklines("M1",  ms_m1,  df_m1,  big_m1,  windows=(10, 20), tail=120)

    # ── Part 6: Composite signal ─────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  PART 6 — COMPOSITE SIGNAL: z_ppt+z_tpm elevated")
    print("═"*70)
    analysis_composite("S30", ms_s30, df_s30, df_m5, big_m5, W=20, thresh_z=1.5)
    analysis_composite("M1",  ms_m1,  df_m1,  df_m5, big_m5, W=20, thresh_z=1.5)

    # ── Part 7: Complexity / Fractal metrics ─────────────────────────────────
    print(f"\n{'═'*70}")
    print("  PART 7 — COMPLEXITY / FRACTAL METRICS (all 13 series)")
    print("═"*70)
    analysis_complexity_metrics("S30", ms_s30, df_s30, df_m5, big_m5, W=20)
    analysis_complexity_metrics("M1",  ms_m1,  df_m1,  df_m5, big_m5, W=20)

    # ── Part 8: Feature selection pipeline ───────────────────────────────────
    print(f"\n{'═'*70}")
    print("  PART 8 — FEATURE SELECTION: Bonferroni → dedup → LightGBM → WF")
    print("═"*70)
    kept_s30, stable_s30 = analysis_feature_selection(
        "S30", ms_s30, df_s30, df_m5, big_m5, W=20)
    kept_m1, stable_m1 = analysis_feature_selection(
        "M1",  ms_m1,  df_m1,  df_m5, big_m5, W=20)

    # Summary: features stable in both TFs
    both_stable = set(stable_s30) & set(stable_m1)
    print(f"\n{'─'*70}")
    print(f"  CROSS-TF STABLE SIGNALS (survive S30 AND M1 walk-forward):")
    if both_stable:
        for k in sorted(both_stable):
            print(f"    {k}")
    else:
        print("    (none — signals may be TF-specific)")
    print(f"{'─'*70}")

    # ── Part 9: TSFresh on tpm ────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  PART 9 — TSFRESH FEATURE BANK: tpm  (MinimalFCParameters)")
    print("═"*70)
    ts_kept_s30, ts_stable_s30 = analysis_tsfresh(
        "S30", ms_s30, df_s30, df_m5, big_m5,
        series=("tpm", "rvol"), fc_params="minimal")
    ts_kept_m1, ts_stable_m1 = analysis_tsfresh(
        "M1", ms_m1, df_m1, df_m5, big_m5,
        series=("tpm", "rvol"), fc_params="minimal")

    ts_both = set(ts_stable_s30) & set(ts_stable_m1)
    print(f"\n{'─'*70}")
    print(f"  TSFRESH CROSS-TF STABLE (S30 ∩ M1):")
    if ts_both:
        for k in sorted(ts_both):
            print(f"    {k}")
    else:
        print("    (none)")
    print(f"{'─'*70}")

    print("\nDone.\n")


def analysis_tsfresh(name, ms_low, df_low, df_m5, big_m5, W=20, step=None,
                     series=("tpm",), fc_params="minimal",
                     min_r=0.05, dedup_thresh=0.85):
    """
    Part 9: TSFresh feature extraction on rolling windows → 8A-D pipeline.
    fc_params: 'minimal' | 'efficient' | 'full'
    """
    try:
        from tsfresh import extract_features
        from tsfresh.utilities.dataframe_functions import impute
        from tsfresh.feature_extraction import (
            MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters)
    except ImportError:
        print("  [tsfresh not installed — pip install tsfresh]")
        return [], []

    print(f"\n{'═'*70}")
    print(f"  PART 9 — TSFRESH  [{name}→M5]  W={W}  "
          f"series={series}  params={fc_params}")
    print(f"{'═'*70}")

    N         = len(df_low)
    N5        = len(df_m5)
    lo_to_m5  = align_low_to_high(df_low, df_m5, 300)

    if step is None:
        step = max(1, N // 15_000)

    fc_map = {
        "minimal":   MinimalFCParameters(),
        "efficient": EfficientFCParameters(),
        "full":      ComprehensiveFCParameters(),
    }
    fc = fc_map.get(fc_params, MinimalFCParameters())

    # ── Build rolling windows for each series ────────────────────────────────
    raw_feats: dict = {}

    for sname in series:
        if sname + "0" in ms_low:
            arr_f = np.asarray(ms_low[sname + "0"], dtype=np.float64)
        elif sname + "_5" in ms_low:
            arr_f = np.asarray(ms_low[sname + "_5"], dtype=np.float64)
        else:
            print(f"  [series '{sname}' not found — skipping]")
            continue

        # Collect valid positions (no NaN in window)
        positions = np.arange(W - 1, N, step)
        valid_pos = [i for i in positions
                     if not np.isnan(arr_f[i - W + 1: i + 1]).any()]
        n_win = len(valid_pos)
        print(f"  Building TSFresh input: {n_win:,} windows × W={W} ({sname}) …",
              flush=True)

        # Vectorised long-format construction (fast vs row-append)
        id_col = np.repeat(np.array(valid_pos, dtype=np.int64), W)
        t_col  = np.tile(np.arange(W, dtype=np.int32), n_win)
        v_col  = np.concatenate([arr_f[i - W + 1: i + 1] for i in valid_pos])

        df_ts = pd.DataFrame({"id": id_col, "t": t_col, "v": v_col})
        print(f"  DataFrame: {len(df_ts):,} rows — extracting …", flush=True)

        feats = extract_features(
            df_ts,
            column_id="id", column_sort="t", column_value="v",
            default_fc_parameters=fc,
            disable_progressbar=True,
            n_jobs=4,
        )
        impute(feats)
        print(f"  Extracted {feats.shape[1]} raw features for '{sname}'.")

        # Map computed positions back to full N-length arrays + ffill
        col_rename = {c: f"ts_{sname}_{c.replace('v__', '')}"
                      for c in feats.columns}
        for orig, new_key in col_rename.items():
            full = np.full(N, np.nan)
            for idx_val in feats.index:
                full[int(idx_val)] = float(feats.at[idx_val, orig])
            raw_feats[new_key] = pd.Series(full).ffill().values

    if not raw_feats:
        print("  No features extracted.")
        return [], []

    n_feats = len(raw_feats)
    print(f"\n  Total TSFresh features: {n_feats}")

    # ── Aggregate to M5 (max — polarity resolved by sign of r in 8A) ─────────
    m5_all = {k: _agg_m5_max(v, lo_to_m5, N5) for k, v in raw_feats.items()}

    n_total = len(m5_all)
    p_bonf  = 0.05 / max(n_total, 1)

    # ── 8A. Significance filter ───────────────────────────────────────────────
    print(f"\n  8A. Significance filter  "
          f"(n_total={n_total}  Bonferroni p<{p_bonf:.1e}  |r|>{min_r}):")

    ranked = []
    for key, arr in m5_all.items():
        both = ~np.isnan(arr)
        if both.sum() < 30 or big_m5[both].sum() < 2:
            continue
        try:
            r, p = scipy_stats.pearsonr(arr[both], big_m5[both].astype(float))
        except ValueError:
            continue
        ranked.append((abs(r), r, p, key, arr))
    ranked.sort(reverse=True)

    survivors = [(ar, r, p, k, a) for ar, r, p, k, a in ranked
                 if ar >= min_r and p < p_bonf]
    print(f"  Survivors: {len(survivors)} / {len(ranked)}")
    print(f"\n  {'Key':<50}  {'r':>8}  {'p':>10}  bar")
    for absr, r, p, key, _ in survivors[:40]:
        bar = "█" * min(int(absr * 200), 30)
        print(f"  {key:<50}  {r:>+8.4f}  {p:>10.2e}  {bar}")
    if len(survivors) > 40:
        print(f"  … {len(survivors) - 40} more survivors")

    if not survivors:
        print("  No survivors — try EfficientFCParameters for more features.")
        return [], []

    # ── 8B. Greedy deduplication ──────────────────────────────────────────────
    print(f"\n  8B. Greedy deduplication  (|r_pair|>{dedup_thresh} = redundant):")
    kept: list = []
    n_dropped = 0
    for candidate in survivors:
        _, _, _, key_c, arr_c = candidate
        both_c = ~np.isnan(arr_c)
        redundant = any(
            (both_c & ~np.isnan(arr_k)).sum() >= 20
            and abs(scipy_stats.pearsonr(
                arr_c[both_c & ~np.isnan(arr_k)],
                arr_k[both_c & ~np.isnan(arr_k)])[0]) >= dedup_thresh
            for _, _, _, _, arr_k in kept
        )
        if redundant:
            n_dropped += 1
        else:
            kept.append(candidate)

    print(f"  Independent signals: {len(kept)}  (dropped {n_dropped} redundant)")
    print(f"\n  {'Key':<50}  {'r_big_m5':>9}")
    for _, r, _, key, _ in kept:
        print(f"  {key:<50}  {r:>+9.4f}")

    # ── 8C. LightGBM feature importance ──────────────────────────────────────
    print(f"\n  8C. LightGBM feature importance → big_m5:")
    try:
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        feat_names = [key for _, _, _, key, _ in kept]
        X = np.column_stack([a for _, _, _, _, a in kept]).astype(np.float32)
        y = big_m5.astype(np.int32)
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            if m.any():
                med = float(np.nanmedian(X[:, j]))
                X[m, j] = med if not np.isnan(med) else 0.0

        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        params = dict(objective="binary", metric="auc", verbosity=-1,
                      n_estimators=300, learning_rate=0.05, num_leaves=31,
                      scale_pos_weight=n_neg / max(n_pos, 1),
                      feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=5, min_child_samples=20)
        cv    = StratifiedKFold(n_splits=3, shuffle=False)
        imps  = np.zeros(len(feat_names)); aucs: list = []
        for tr_i, va_i in cv.split(X, y):
            mdl = lgb.LGBMClassifier(**params)
            mdl.fit(X[tr_i], y[tr_i], eval_set=[(X[va_i], y[va_i])],
                    callbacks=[lgb.early_stopping(30, verbose=False),
                                lgb.log_evaluation(-1)])
            aucs.append(roc_auc_score(y[va_i], mdl.predict_proba(X[va_i])[:, 1]))
            imps += mdl.booster_.feature_importance(importance_type="gain")

        print(f"  CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
              f"(folds: {', '.join(f'{a:.4f}' for a in aucs)})")
        order = np.argsort(imps)[::-1]; best = imps[order[0]]
        print(f"\n  {'Feature':<50}  {'gain':>9}  {'rel%':>6}  bar")
        for i in order[:20]:
            if imps[i] < 0.01: continue
            rel = imps[i] / best * 100
            print(f"  {feat_names[i]:<50}  {imps[i]:>9.1f}  "
                  f"{rel:>5.1f}%  {'█' * int(rel / 100 * 30)}")
    except ImportError:
        print("  [lightgbm not installed]")
    except Exception as e:
        print(f"  [8C error: {e}]")

    # ── 8D. Walk-forward stability ────────────────────────────────────────────
    print(f"\n  8D. 3-fold walk-forward stability:")
    fold_n = N5 // 3
    folds  = [(0, fold_n), (fold_n, 2 * fold_n), (2 * fold_n, N5)]
    print(f"  {'Key':<50}  {'F1':>7}  {'F2':>7}  {'F3':>7}  stable?")
    stable_keys = []
    for _, r_full, _, key, arr in kept:
        fold_rs = []
        for f0, f1 in folds:
            sub = arr[f0:f1]; bsub = big_m5[f0:f1]
            v   = ~np.isnan(sub)
            if v.sum() < 20:
                fold_rs.append(np.nan)
            else:
                fr, _ = scipy_stats.pearsonr(sub[v], bsub[v].astype(float))
                fold_rs.append(fr)
        valid_signs = [np.sign(fr) for fr in fold_rs if not np.isnan(fr)]
        stable = (len(valid_signs) >= 2 and len(set(valid_signs)) == 1
                  and valid_signs[0] == np.sign(r_full))
        if stable:
            stable_keys.append(key)
        rs = "  ".join(f"{fr:>+6.3f}" if not np.isnan(fr) else "   n/a"
                       for fr in fold_rs)
        print(f"  {key:<50}  {rs}  {'🟢' if stable else '🔴'}")

    print(f"\n  Stable TSFresh signals: {len(stable_keys)} / {len(kept)}")
    if stable_keys:
        print(f"  → {', '.join(stable_keys)}")
    return kept, stable_keys


if __name__ == "__main__":
    main()
