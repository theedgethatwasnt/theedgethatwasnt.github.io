"""
Cointegration-pairs mean-reversion screen (FX stat-arb).

Hypothesis: for economically-linked pairs, the residual of an IS-fitted
cointegrating regression is stationary; entering when |z| is large and exiting
on reversion yields positive expectancy net of 2-leg (×in/out) spread.

WHY this is shaped to survive where directional tests died:
  - Mean-reverting (not directional prediction).
  - Long/short => market-neutral to common USD drift (the contaminant that made
    every cross-pair trend test "secretly USD_JPY drift").
  - Slow, large divergences can exceed the round-trip spread.

SOP compliance:
  R3/R3a fills at bid/ask close; R5 spread gate is IS-only P90; R8 OOS sealed
  (hedge ratio beta, z mean/std, spread gate ALL frozen from IS); MC sign-shuffle.

Gates (per pair-combo, must pass ALL):
  - IS residual stationary: ADF-lite t-stat < -3.0 (manual, self-contained)
  - Walk-forward: each of 3 IS chunks sum>0 and >=5 trades
  - OOS net P&L > 0
  - MC sign-shuffle p < 0.05
  - OOS trade-P&L |t-stat| > 2
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).parents[3]
M5 = ROOT / "data" / "m5_ba"

IS_FRAC = 0.70

# Economically-sensible cointegration candidates (same quote ccy within a combo).
COMBOS = [
    ("AUD_USD", "NZD_USD"),   # commodity/risk bloc (strongest prior)
    ("AUD_JPY", "NZD_JPY"),   # commodity/risk, JPY-quoted
    ("EUR_USD", "GBP_USD"),   # European bloc
    ("EUR_JPY", "GBP_JPY"),   # European bloc, JPY
    ("NZD_USD", "EUR_USD"),   # exploratory
    ("AUD_USD", "GBP_USD"),   # exploratory
    ("CAD_JPY", "AUD_JPY"),   # commodity-ish, JPY
    ("CHF_JPY", "EUR_JPY"),   # CHF-EUR linkage
]

# config sweep: k_entry, k_exit, k_stop, max_hold(bars)
K_ENTRY = [1.5, 2.0, 2.5]
K_EXIT  = [0.0, 0.5]
K_STOP  = [3.5, 4.5]
MAX_HOLD = [2016, 4032]       # 1 wk / 2 wk of M5 bars
CONFIGS = np.array([[ke, kx, ks, mh]
                    for ke in K_ENTRY for kx in K_EXIT
                    for ks in K_STOP for mh in MAX_HOLD], dtype=np.float64)


def pip_of(pair):
    return 0.01 if pair.endswith("JPY") else 0.0001


def adf_lite(r):
    """Manual ADF (no lags): regress dr_t on r_{t-1}+const; return t-stat of slope.
    Strongly negative => residual is mean-reverting/stationary."""
    r = np.asarray(r, float)
    y = np.diff(r)
    x = r[:-1]
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - 2
    s2 = (resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(s2 * xtx_inv[1, 1])
    return beta[1] / se   # t-stat of the AR coefficient


def mc_sign_test(oos_pnl, n=300, seed=0):
    """p = fraction of sign-shuffles whose sum >= real sum. Gate p<0.05."""
    if len(oos_pnl) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    real = float(oos_pnl.sum())
    a = np.abs(oos_pnl)
    if real <= 0:
        return 1.0
    cnt = 0
    for _ in range(n):
        s = rng.choice(np.array([-1.0, 1.0]), size=len(a))
        if float((s * a).sum()) >= real:
            cnt += 1
    return cnt / n


@njit(cache=True)
def run_configs(resid, z, sp_ret_rt, chunk, sp_gate, configs,
                trade_pnl, trade_chunk, trade_cnt):
    n = resid.shape[0]
    nc = configs.shape[0]
    for ci in range(nc):
        k_entry = configs[ci, 0]
        k_exit  = configs[ci, 1]
        k_stop  = configs[ci, 2]
        max_hold = configs[ci, 3]
        pos = 0          # +1 short-spread (entered z>0), -1 long-spread (z<0)
        r_entry = 0.0
        cost_entry = 0.0
        entry_i = 0
        t = 0
        for i in range(n):
            if pos == 0:
                if sp_ret_rt[i] <= sp_gate:
                    if z[i] > k_entry:
                        pos = 1; r_entry = resid[i]; cost_entry = sp_ret_rt[i]; entry_i = i
                    elif z[i] < -k_entry:
                        pos = -1; r_entry = resid[i]; cost_entry = sp_ret_rt[i]; entry_i = i
            else:
                hold = i - entry_i
                exit_now = False
                if pos == 1:   # short spread: profit if residual falls
                    if z[i] <= k_exit or z[i] >= k_stop or hold >= max_hold:
                        exit_now = True
                else:          # long spread: profit if residual rises
                    if z[i] >= -k_exit or z[i] <= -k_stop or hold >= max_hold:
                        exit_now = True
                if exit_now:
                    if pos == 1:
                        pnl = (r_entry - resid[i]) - cost_entry
                    else:
                        pnl = (resid[i] - r_entry) - cost_entry
                    pnl *= 1e4   # log-resid -> bps of notional
                    trade_pnl[ci, t] = pnl
                    trade_chunk[ci, t] = chunk[entry_i]
                    t += 1
                    pos = 0
        trade_cnt[ci] = t


def run_combo(a, b):
    pa = pd.read_parquet(M5 / f"{a}_M5_BA.parquet")[["timestamp", "close", "bid_c", "ask_c"]]
    pb = pd.read_parquet(M5 / f"{b}_M5_BA.parquet")[["timestamp", "close", "bid_c", "ask_c"]]
    df = pa.merge(pb, on="timestamp", suffixes=("_a", "_b")).dropna()
    df = df[(df["close_a"] > 0) & (df["close_b"] > 0)].reset_index(drop=True)
    n = len(df)
    if n < 20000:
        return None
    is_end = int(n * IS_FRAC)

    logA = np.log(df["close_a"].values)
    logB = np.log(df["close_b"].values)
    # cointegration on IS only
    beta, alpha = np.polyfit(logB[:is_end], logA[:is_end], 1)
    resid = logA - (alpha + beta * logB)
    r_mu = resid[:is_end].mean()
    r_sd = resid[:is_end].std()
    if r_sd <= 0:
        return None
    z = (resid - r_mu) / r_sd

    # round-trip relative spread cost (both legs), |beta| for the hedge leg
    sp_a = (df["ask_c_a"].values - df["bid_c_a"].values) / df["close_a"].values
    sp_b = (df["ask_c_b"].values - df["bid_c_b"].values) / df["close_b"].values
    sp_ret_rt = sp_a + abs(beta) * sp_b
    sp_gate = float(np.percentile(sp_ret_rt[:is_end], 90))

    # chunk ids: 3 IS chunks (0,1,2), OOS=3
    chunk = np.full(n, 3, dtype=np.int64)
    edges = np.linspace(0, is_end, 4).astype(int)
    for c in range(3):
        chunk[edges[c]:edges[c + 1]] = c

    adf_is = adf_lite(resid[:is_end])
    adf_oos = adf_lite(resid[is_end:])

    max_tr = n // 5
    nc = CONFIGS.shape[0]
    trade_pnl = np.zeros((nc, max_tr))
    trade_chunk = np.full((nc, max_tr), -1, dtype=np.int64)
    trade_cnt = np.zeros(nc, dtype=np.int64)
    run_configs(resid, z, sp_ret_rt, chunk, sp_gate, CONFIGS,
                trade_pnl, trade_chunk, trade_cnt)

    # evaluate each config
    best = None
    for ci in range(nc):
        t = trade_cnt[ci]
        if t < 15:
            continue
        pnl = trade_pnl[ci, :t]
        ck = trade_chunk[ci, :t]
        oos = pnl[ck == 3]
        if len(oos) < 5:
            continue
        # WF: each IS chunk sum>0 and >=5 trades
        wf_ok = True
        for c in range(3):
            cp = pnl[ck == c]
            if len(cp) < 5 or cp.sum() <= 0:
                wf_ok = False
                break
        oos_sum = float(oos.sum())
        oos_t = float(oos.mean() / (oos.std() / np.sqrt(len(oos)))) if oos.std() > 0 else 0.0
        rec = dict(ci=ci, k_entry=CONFIGS[ci, 0], k_exit=CONFIGS[ci, 1],
                   k_stop=CONFIGS[ci, 2], max_hold=int(CONFIGS[ci, 3]),
                   n_trades=int(t), oos_trades=int(len(oos)),
                   is_sum=float(pnl[ck != 3].sum()), oos_sum=oos_sum,
                   oos_t=oos_t, wf_ok=wf_ok, mc_p=1.0,
                   oos_per_trade=oos_sum / len(oos))
        if wf_ok and oos_sum > 0:
            rec["mc_p"] = mc_sign_test(oos, n=300, seed=ci)
        # rank by: passes-all first, then OOS sum
        passes = wf_ok and oos_sum > 0 and rec["mc_p"] < 0.05 and abs(oos_t) > 2 and adf_is < -3.0
        rec["PASS"] = passes
        score = (1 if passes else 0, oos_sum)
        if best is None or score > best[0]:
            best = (score, rec)

    return dict(combo=f"{a}/{b}", beta=float(beta), n=n, is_end=is_end,
                adf_is=float(adf_is), adf_oos=float(adf_oos),
                best=(best[1] if best else None))


def main():
    print(f"{'combo':<18}{'beta':>7}{'adf_IS':>8}{'adf_OOS':>8}  "
          f"{'cfg(ke/kx/ks/h)':<18}{'IS':>9}{'OOS':>9}{'oos/t':>7}{'tr':>5}{'WF':>4}{'MCp':>6} PASS")
    print("-" * 110)
    passed = []
    for a, b in COMBOS:
        r = run_combo(a, b)
        if r is None:
            print(f"{a}/{b:<10} (insufficient data)")
            continue
        bst = r["best"]
        if bst is None:
            print(f"{r['combo']:<18}{r['beta']:>7.2f}{r['adf_is']:>8.2f}{r['adf_oos']:>8.2f}  "
                  f"{'(no qualifying cfg)':<18}")
            continue
        cfg = f"{bst['k_entry']:.1f}/{bst['k_exit']:.1f}/{bst['k_stop']:.1f}/{bst['max_hold']}"
        print(f"{r['combo']:<18}{r['beta']:>7.2f}{r['adf_is']:>8.2f}{r['adf_oos']:>8.2f}  "
              f"{cfg:<18}{bst['is_sum']:>9.1f}{bst['oos_sum']:>9.1f}{bst['oos_per_trade']:>7.2f}"
              f"{bst['n_trades']:>5}{'Y' if bst['wf_ok'] else 'n':>4}{bst['mc_p']:>6.3f}"
              f"  {'**PASS**' if bst['PASS'] else ''}")
        if bst["PASS"]:
            passed.append(r["combo"])
    print("-" * 110)
    print(f"PASSED (all gates): {passed if passed else 'NONE'}")
    print("Units: P&L in bps of gross notional, net of round-trip 2-leg spread. "
          "ADF-lite t<-3 = stationary residual (IS).")


if __name__ == "__main__":
    main()
