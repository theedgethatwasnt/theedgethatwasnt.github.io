#!/usr/bin/env python3
"""
FX statistical arbitrage — Avellaneda–Lee residual mean-reversion (2010), adapted to
12 OANDA majors (8 currencies). The ONE eigen-mode idea we never ran: trade the
RESIDUAL after removing the currency factors (PCA eigen-portfolios), not the factors.

Pipeline per the project SOP (R1-R9, IS/OOS/WF/MC, spread-net):
  1. M5 mid -> resample to the chosen horizon (D1 NY-close or H4). Aligned panel.
  2. Rolling window L: corr-matrix PCA -> top-k risk-weighted eigen-portfolios (factors F).
  3. Regress each pair's return on F over the window -> residual epsilon.
  4. Auxiliary process X = cumsum(epsilon); fit OU -> kappa, m, sigma_eq -> s-score.
  5. FAIL-FAST: IC of s-score vs FORWARD HEDGED residual return (reversion => IC<0).
  6. If IC reverts: trading sim. V1 hedged residual P&L (does the phenomenon pay?),
     V2 naked single-pair (retail-deployable, net of that pair's spread).

Everything causal: window uses bars strictly < t; the regression/OU at bar t use only
data up to t; forward return is t -> t+H. No lookahead.
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

PROJECT = Path("/path/to/projects/fx-core")
DATA = PROJECT / "data" / "m5_ohlc"

# pip size + realistic OANDA retail median spread (pips) — from research/experiments/_lib
PAIRS = {
    "USD_JPY": (0.01,   2.10), "EUR_JPY": (0.01,   2.50), "GBP_JPY": (0.01, 4.00),
    "AUD_JPY": (0.01,   2.30), "CAD_JPY": (0.01,   2.60), "CHF_JPY": (0.01, 3.00),
    "NZD_JPY": (0.01,   3.00), "EUR_USD": (0.0001, 1.70), "GBP_USD": (0.0001, 2.40),
    "AUD_USD": (0.0001, 1.60), "EUR_GBP": (0.0001, 2.00), "NZD_USD": (0.0001, 2.00),
}
PLIST = list(PAIRS.keys())
IS_FRAC = 4/6


def load_panel(horizon):
    """Aligned close panel (T x N) of mid prices at the horizon, plus the index."""
    rule = {"D": "1D", "H4": "4h", "H8": "8h"}[horizon]
    con = duckdb.connect()
    series = {}
    for p in PLIST:
        f = DATA / f"{p}_M5.parquet"
        df = con.execute(f"SELECT timestamp, close FROM '{f}' ORDER BY timestamp").df()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        # NY-close daily boundary: FX day rolls at 17:00 NY = 21:00/22:00 UTC. Shift so
        # resample('1D') buckets a full trading day. For H4/H8 plain UTC buckets are fine.
        if horizon == "D":
            s = df["close"].shift(0).resample(rule, offset="21h").last()
        else:
            s = df["close"].resample(rule).last()
        series[p] = s
    con.close()
    panel = pd.DataFrame(series).dropna(how="any")
    return panel


def eig_factors(Rw, k):
    """Risk-weighted eigen-portfolios from a window of returns Rw (L x N).
    Returns factor returns over the window (L x k) and the eigen weights Q (k x N)."""
    sig = Rw.std(axis=0)
    sig[sig == 0] = 1e-12
    Z = (Rw - Rw.mean(axis=0)) / sig
    C = np.corrcoef(Z, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    w, V = np.linalg.eigh(C)              # ascending
    idx = np.argsort(w)[::-1][:k]
    Vk = V[:, idx]                        # N x k
    Q = (Vk.T / sig)                      # k x N  (risk-weighted weights)
    F = Rw @ Q.T                          # L x k  factor returns
    return F, Q


def ou_sscore(resid_cum):
    """Fit OU to the auxiliary process X (cumulative residual). Return s-score of the
    last point and kappa (reversion speed, per-bar). resid_cum length L."""
    X = resid_cum
    x0, x1 = X[:-1], X[1:]
    vx = x0.var()
    if vx < 1e-18 or len(x0) < 10:
        return 0.0, 0.0
    b = np.cov(x0, x1, bias=True)[0, 1] / vx
    if b <= 0 or b >= 1:
        return 0.0, 0.0
    a = x1.mean() - b * x0.mean()
    zeta = x1 - (a + b * x0)
    var_z = zeta.var()
    m = a / (1 - b)
    sigma_eq = np.sqrt(var_z / (1 - b * b)) if var_z > 0 else 0.0
    if sigma_eq <= 0:
        return 0.0, 0.0
    s = (X[-1] - m) / sigma_eq
    kappa = -np.log(b)                    # per-bar mean-reversion speed
    return s, kappa


def run(horizon="D", L=60, k=3, H=1, kappa_min=0.0, verbose=True):
    panel = load_panel(horizon)
    R = np.log(panel.values[1:] / panel.values[:-1])     # T x N log returns
    idx = panel.index[1:]
    T, N = R.shape
    if verbose:
        print(f"[{horizon}] panel {panel.shape} -> returns {R.shape}  {idx[0]} .. {idx[-1]}")

    # outputs: for every (bar t >= L, asset i): s-score, kappa, forward HEDGED residual ret
    s_all, k_all, fwd_resid, fwd_naked, t_all, i_all = [], [], [], [], [], []
    # precompute nothing fancy — rolling regression each bar (T~1900 for D, fast enough)
    for t in range(L, T - H):
        Rw = R[t - L:t]                                  # window strictly before t
        F, Q = eig_factors(Rw, k)
        # regress each asset on factors over window (with intercept)
        A = np.column_stack([np.ones(L), F])             # L x (k+1)
        beta, *_ = np.linalg.lstsq(A, Rw, rcond=None)    # (k+1) x N
        resid_w = Rw - A @ beta                           # L x N residuals in-window
        Xcum = np.cumsum(resid_w, axis=0)                 # auxiliary process per asset
        # forward window factor returns use the SAME loadings (causal: built at t)
        Rf = R[t:t + H]                                   # H x N forward returns
        Ff = Rf @ Q.T                                     # H x k forward factor returns
        # forward residual return = forward asset return minus forward factor explanation
        fwd_res = (Rf - (Ff @ beta[1:])).sum(axis=0)      # N  (hedged residual fwd ret)
        fwd_nak = Rf.sum(axis=0)                           # N  (naked asset fwd ret)
        for i in range(N):
            s, kp = ou_sscore(Xcum[:, i])
            if s == 0.0 and kp == 0.0:
                continue
            s_all.append(s); k_all.append(kp)
            fwd_resid.append(fwd_res[i]); fwd_naked.append(fwd_nak[i])
            t_all.append(t); i_all.append(i)

    s_all = np.array(s_all); k_all = np.array(k_all)
    fwd_resid = np.array(fwd_resid); fwd_naked = np.array(fwd_naked)
    t_all = np.array(t_all); i_all = np.array(i_all)

    # kappa filter (mean-reversion fast enough relative to horizon)
    msk = k_all >= kappa_min
    s_f, fr_f, fn_f, t_f, i_f = s_all[msk], fwd_resid[msk], fwd_naked[msk], t_all[msk], i_all[msk]

    is_cut = L + int((T - L) * IS_FRAC)
    is_m = t_f < is_cut

    def ic(s, f):
        if len(s) < 30: return np.nan, np.nan, len(s)
        # Spearman-ish: use Pearson on ranks
        rs = pd.Series(s).rank().values; rf = pd.Series(f).rank().values
        r = np.corrcoef(rs, rf)[0, 1]
        tstat = r * np.sqrt(len(s) - 2) / np.sqrt(max(1 - r*r, 1e-12))
        return r, tstat, len(s)

    print(f"\n=== IC test (reversion => negative IC of s-score vs forward residual ret) ===")
    print(f"  L={L} k={k} H={H} kappa_min={kappa_min}  obs={len(s_f)} (IS={is_m.sum()} OOS={(~is_m).sum()})")
    for lab, mask in [("IS", is_m), ("OOS", ~is_m), ("ALL", np.ones(len(s_f), bool))]:
        r_res, t_res, n = ic(s_f[mask], fr_f[mask])
        r_nak, t_nak, _ = ic(s_f[mask], fn_f[mask])
        print(f"  {lab:>4}: residual IC={r_res:+.4f} (t={t_res:+.2f})   naked IC={r_nak:+.4f} (t={t_nak:+.2f})   n={n}")

    return dict(s=s_f, fr=fr_f, fn=fn_f, t=t_f, i=i_f, is_cut=is_cut, idx=idx, N=N, T=T, L=L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="D", choices=["D", "H4", "H8"])
    ap.add_argument("--L", type=int, default=60)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--kappa-min", type=float, default=0.0)
    a = ap.parse_args()
    run(a.horizon, a.L, a.k, a.H, a.kappa_min)
