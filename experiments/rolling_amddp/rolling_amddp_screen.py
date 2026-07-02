"""
Rolling-AMDDP5 predictivity screen
===================================

CONCEPT
-------
At each cadence step t, score the *hypothetical* trade that opened at the
window-open price (t-10min) and is held to now (t), using the project's AMDDP5
reward:

    AMDDP5 = pnl_pips - 0.05 * SUM_held_bars max(0, -u_b) * bar_minutes

where u_b is running mid-based unrealized P&L (pips). This is the EXACT
definition from research/experiments/amddp5/scorer.py (canonical) and
lib/fast_eval.py (the training kernel).

We build a SIGNED feature A(t):
    direction = sign(mid_close[t] - mid_close[t-W])      # the realized move dir
    pnl       = direction * (close[t] - close[t-W]) / pip
    underwater = SUM_b max(0, -direction*(close[b]-open_px)/pip) * bar_minutes
    amddp     = pnl - 0.05 * underwater                  # AMDDP5 of the move
    A(t)      = direction * amddp                          # signed by direction

High |A| with +sign = a clean, efficient UP move over the last 10 min;
high |A| with -sign = a clean DOWN move.  Low |A| = choppy / went nowhere.

NOTE on what AMDDP "in the move direction" means: by construction the
hypothetical trade is in the direction the price actually went, so pnl is the
|net move| and the penalty captures how much it was underwater along the way.
A clean monotone move pays ~zero penalty -> |A| ~ |net move|. A whippy move
that ended up positive but dipped a lot pays a large penalty -> |A| shrinks.

THE SCREEN  (W = 120 S5 bars = 10 min; bar_minutes = 1/12)
----------------------------------------------------------
Cadence = non-overlapping consecutive windows (stride = W) to keep IC samples
near-independent and avoid the autocorrelation inflation of overlapping windows.

1. CONTINUATION (direction):
     IC( A(t), fwd_ret(t -> t+W) )            signed Spearman
     fwd_ret = mid net return next 10 min (pips).
     Also net-of-spread expectancy of trading sign(A) into the next window.

2. MAGNITUDE:
     IC( |A(t)|, |fwd_ret| )  and  IC( |A(t)|, fwd_realized_vol )
     fwd_realized_vol = std of S5 mid returns over next window (pips).

3. AUTOCORR / REGIME PERSISTENCE:
     corr( A(t), A(t+W) ),  corr( |A(t)|, |A(t+W)| ), and Markov persistence
     of top-decile |A| patches (P(high_t+1 | high_t)).

4. CONDITIONAL EXPECTANCY:
     Bucket by A(t) decile; report next-window continuation expectancy
     net of spread in the top/bottom |A| deciles. Tradeable continuation in
     high-A patches specifically?

GATES (project feature-statistics):
    |Newey-West t-stat| > 2 ; walk-forward sign-consistency (chunked IC) ;
    MC order-shuffle permutation p < 0.05.

Net of OANDA spread throughout. Adversarial: spread floor compared explicitly.
"""
from __future__ import annotations
import sys, os, json, time
import numpy as np
import pandas as pd
from numba import njit
from scipy import stats

PAIRS = ["AUD_JPY","AUD_USD","CAD_JPY","CHF_JPY","EUR_GBP","EUR_JPY",
         "EUR_USD","GBP_JPY","GBP_USD","NZD_JPY","NZD_USD","USD_JPY"]
DATA = "/path/to/projects/fx-core/data/s5_ohlc"
W = 120            # 10 min on S5
BAR_MIN = 1.0/12.0 # minutes per S5 bar
AMDDP_K = 0.05
N_MC = 2000
N_WF_CHUNKS = 5    # walk-forward chunks for sign-consistency


def pip_size(pair): return 0.01 if pair.endswith("JPY") else 0.0001


@njit(cache=False)
def build_rolling_amddp(close, open_, spread, pip, W, bar_min, k):
    """For each window-end index t (stride=W, non-overlapping), compute the
    signed rolling-AMDDP5 feature A(t), the forward 10-min mid net return,
    forward |abs| net return, forward realized vol, and the avg spread over
    the forward window (for net-of-spread accounting).

    Returns parallel arrays indexed by window number.
    """
    n = len(close)
    n_win = (n - 1) // W  # number of complete back-window/forward-window pairs
    A      = np.empty(n_win, dtype=np.float64)
    absA   = np.empty(n_win, dtype=np.float64)
    fwd    = np.empty(n_win, dtype=np.float64)  # signed fwd net return (pips)
    fwd_abs= np.empty(n_win, dtype=np.float64)
    fwd_vol= np.empty(n_win, dtype=np.float64)
    fwd_sp = np.empty(n_win, dtype=np.float64)  # avg spread fwd window (pips)
    net_move = np.empty(n_win, dtype=np.float64)  # raw |net move| of back window
    m = 0
    # window k covers back [t0, t1], forward [t1, t2]; t1 is the "now" index
    for w in range(1, n_win):
        t1 = w * W           # window-end / "now"
        t0 = t1 - W          # window-open (10 min ago)
        t2 = t1 + W          # forward end
        if t2 >= n:
            break
        open_px = close[t0]  # window-open mid price
        net = (close[t1] - open_px) / pip
        direction = 1.0 if net >= 0.0 else -1.0
        # underwater integral over the back window (mid-based running u)
        under = 0.0
        for b in range(t0 + 1, t1 + 1):
            u = direction * (close[b] - open_px) / pip
            if u < 0.0:
                under += -u * bar_min
        pnl = direction * (close[t1] - open_px) / pip
        amddp = pnl - k * under
        A[m]    = direction * amddp
        absA[m] = amddp if amddp >= 0.0 else -amddp
        net_move[m] = pnl   # = |net move|, always >= 0
        # forward window
        fr = (close[t2] - close[t1]) / pip
        fwd[m] = fr
        fwd_abs[m] = fr if fr >= 0.0 else -fr
        # realized vol of S5 mid returns over forward window
        s = 0.0; ss = 0.0; cnt = 0
        for b in range(t1 + 1, t2 + 1):
            r = (close[b] - close[b-1]) / pip
            s += r; ss += r*r; cnt += 1
        if cnt > 1:
            mean = s / cnt
            var = ss / cnt - mean*mean
            fwd_vol[m] = np.sqrt(var) if var > 0.0 else 0.0
        else:
            fwd_vol[m] = 0.0
        # avg spread over forward window (entry+exit cost proxy)
        sps = 0.0; spc = 0
        for b in range(t1, t2 + 1):
            sps += spread[b] / pip; spc += 1
        fwd_sp[m] = sps / spc if spc > 0 else 0.0
        m += 1
    return A[:m], absA[:m], fwd[:m], fwd_abs[:m], fwd_vol[:m], fwd_sp[:m], net_move[:m]


def newey_west_tstat(x, lag=5):
    """NW t-stat that the mean of x differs from 0."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10: return 0.0
    mean = x.mean()
    e = x - mean
    gamma0 = (e*e).sum() / n
    var = gamma0
    for L in range(1, lag+1):
        w = 1.0 - L/(lag+1)
        cov = (e[L:]*e[:-L]).sum() / n
        var += 2.0 * w * cov
    se = np.sqrt(var / n)
    return mean / se if se > 0 else 0.0


def ic_with_tstat(x, y, lag=5):
    """Spearman IC + NW t-stat on the per-chunk IC series + overall p."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 50:
        return 0.0, 0.0, 1.0
    rho, p = stats.spearmanr(x, y)
    # NW t-stat from chunked ICs (matches feature_statistics convention)
    nchunk = 20
    cs = np.array_split(np.arange(len(x)), nchunk)
    ics = []
    for c in cs:
        if len(c) > 20:
            r,_ = stats.spearmanr(x[c], y[c])
            if np.isfinite(r): ics.append(r)
    if len(ics) >= 5:
        ics = np.array(ics)
        t = ics.mean() / (ics.std(ddof=1)/np.sqrt(len(ics))) if ics.std(ddof=1)>0 else 0.0
    else:
        t = 0.0
    return rho, t, p


def wf_sign_consistency(x, y, nchunks=N_WF_CHUNKS):
    """Sign of IC in each temporal chunk; return list of signs and the IC."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    cs = np.array_split(np.arange(len(x)), nchunks)
    signs = []
    for c in cs:
        if len(c) > 30:
            r,_ = stats.spearmanr(x[c], y[c])
            signs.append(np.sign(r) if np.isfinite(r) else 0)
        else:
            signs.append(0)
    return signs


def mc_perm_pvalue(x, y, n_mc=N_MC, seed=0):
    """Order-shuffle MC: permute y vs x, P(|rho_shuf| >= |rho_obs|)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 50: return 1.0
    rho_obs,_ = stats.spearmanr(x, y)
    rho_obs = abs(rho_obs)
    rng = np.random.default_rng(seed)
    xr = stats.rankdata(x); yr = stats.rankdata(y)
    xr = xr - xr.mean()
    cnt = 0
    n = len(x)
    denom = np.sqrt((xr*xr).sum())
    for _ in range(n_mc):
        ys = yr[rng.permutation(n)]
        ys = ys - ys.mean()
        r = (xr*ys).sum() / (denom*np.sqrt((ys*ys).sum()) + 1e-12)
        if abs(r) >= rho_obs:
            cnt += 1
    return cnt / n_mc


def analyze_pair(pair):
    pip = pip_size(pair)
    df = pd.read_parquet(f"{DATA}/{pair}_S5_BA.parquet",
                         columns=["close","open","bid_c","ask_c"])
    close = df["close"].to_numpy(np.float64)
    open_ = df["open"].to_numpy(np.float64)
    spread = (df["ask_c"].to_numpy(np.float64) - df["bid_c"].to_numpy(np.float64))
    A, absA, fwd, fwd_abs, fwd_vol, fwd_sp, net_move = build_rolling_amddp(
        close, open_, spread, pip, W, BAR_MIN, AMDDP_K)
    return dict(A=A, absA=absA, fwd=fwd, fwd_abs=fwd_abs, fwd_vol=fwd_vol,
                fwd_sp=fwd_sp, net_move=net_move, pip=pip)


def conditional_expectancy(A, fwd, fwd_sp):
    """Bucket by A decile; in top/bottom |A| deciles, continuation expectancy
    net of spread. sign(A) is the proposed continuation direction.
    cont_pnl = sign(A) * fwd  - spread_cost (entry+exit ~ 2*spread? use 1x avg
    as conservative single-leg proxy of round trip via fwd_sp which already
    averages spread; deduct full spread as round-trip cost)."""
    absA = np.abs(A)
    n = len(A)
    order = np.argsort(absA)
    dec = n // 10
    out = {}
    for name, idx in [("top_absA", order[-dec:]), ("bot_absA", order[:dec]),
                      ("all", np.arange(n))]:
        a = A[idx]; f = fwd[idx]; sp = fwd_sp[idx]
        cont = np.sign(a) * f                 # gross continuation pips
        net = cont - sp                       # net of one round-trip spread
        out[name] = dict(
            n=int(len(idx)),
            gross_cont_mean=float(np.mean(cont)),
            net_cont_mean=float(np.mean(net)),
            avg_spread=float(np.mean(sp)),
            wr_gross=float(np.mean(cont > 0)),
            wr_net=float(np.mean(net > 0)),
        )
    return out


def main():
    t0 = time.time()
    results = {}
    pooled = {k: [] for k in ["A","absA","fwd","fwd_abs","fwd_vol","fwd_sp"]}
    pooled_pair = []
    print(f"{'='*78}\nROLLING-AMDDP5 PREDICTIVITY SCREEN  (W={W} S5 bars = 10 min)\n{'='*78}")
    print(f"AMDDP5: pnl - {AMDDP_K}*SUM max(0,-u)*{BAR_MIN:.4f}min  | non-overlapping windows\n")

    for pair in PAIRS:
        d = analyze_pair(pair)
        n = len(d["A"])
        for k in pooled: pooled[k].append(d[k])
        pooled_pair.append(np.full(n, pair))

        # TEST 1: continuation (direction)
        ic1, t1_, p1 = ic_with_tstat(d["A"], d["fwd"])
        wf1 = wf_sign_consistency(d["A"], d["fwd"])
        mc1 = mc_perm_pvalue(d["A"], d["fwd"])
        # TEST 2: magnitude
        ic2a, t2a, _ = ic_with_tstat(d["absA"], d["fwd_abs"])
        ic2b, t2b, _ = ic_with_tstat(d["absA"], d["fwd_vol"])
        # TEST 3: autocorrelation/persistence
        a = d["A"]; aa = d["absA"]
        ac_A   = np.corrcoef(a[:-1], a[1:])[0,1] if len(a) > 2 else 0.0
        ac_abs = np.corrcoef(aa[:-1], aa[1:])[0,1] if len(aa) > 2 else 0.0
        thr = np.percentile(aa, 90)
        hi = aa >= thr
        # P(high_{t+1} | high_t)
        if hi[:-1].sum() > 0:
            persist = float(hi[1:][hi[:-1]].mean())
        else:
            persist = 0.0
        base_hi = float(hi.mean())
        # TEST 4: conditional expectancy
        ce = conditional_expectancy(d["A"], d["fwd"], d["fwd_sp"])

        results[pair] = dict(
            n=n,
            cont_ic=ic1, cont_t=t1_, cont_mc_p=mc1, cont_wf=wf1,
            mag_ic_ret=ic2a, mag_t_ret=t2a, mag_ic_vol=ic2b, mag_t_vol=t2b,
            ac_A=float(ac_A), ac_absA=float(ac_abs),
            persist_hi=persist, base_hi=base_hi,
            avg_spread=float(np.mean(d["fwd_sp"])),
            cond=ce,
        )
        wfs = "".join("+" if s>0 else ("-" if s<0 else "0") for s in wf1)
        print(f"{pair:8s} n={n:6d} | CONT ic={ic1:+.4f} t={t1_:+.1f} mc_p={mc1:.3f} wf[{wfs}] "
              f"| MAG ic(|ret|)={ic2a:+.3f} ic(vol)={ic2b:+.3f} | AC(A)={ac_A:+.3f} AC(|A|)={ac_abs:+.3f} "
              f"persist={persist:.2f}(base{base_hi:.2f})")
        net_top = ce["top_absA"]["net_cont_mean"]; gr_top = ce["top_absA"]["gross_cont_mean"]
        print(f"         topABSA: gross_cont={gr_top:+.3f}p net={net_top:+.3f}p "
              f"(spread {ce['top_absA']['avg_spread']:.2f}p, wr_net {ce['top_absA']['wr_net']:.2f}) | "
              f"all: net={ce['all']['net_cont_mean']:+.3f}p")

    # POOLED
    print(f"\n{'='*78}\nPOOLED (all 12 pairs)\n{'='*78}")
    P = {k: np.concatenate(v) for k,v in pooled.items()}
    ic1, t1_, p1 = ic_with_tstat(P["A"], P["fwd"])
    wf1 = wf_sign_consistency(P["A"], P["fwd"])
    mc1 = mc_perm_pvalue(P["A"], P["fwd"])
    ic2a, t2a, _ = ic_with_tstat(P["absA"], P["fwd_abs"])
    ic2b, t2b, _ = ic_with_tstat(P["absA"], P["fwd_vol"])
    ce = conditional_expectancy(P["A"], P["fwd"], P["fwd_sp"])
    wfs = "".join("+" if s>0 else ("-" if s<0 else "0") for s in wf1)
    print(f"TEST 1 CONTINUATION:  IC={ic1:+.4f}  t={t1_:+.2f}  MC_p={mc1:.4f}  WF[{wfs}]")
    print(f"TEST 2 MAGNITUDE:     IC(|A|,|fwd_ret|)={ic2a:+.4f} t={t2a:+.1f}  |  IC(|A|,fwd_vol)={ic2b:+.4f} t={t2b:+.1f}")
    ac_A = np.corrcoef(P['A'][:-1],P['A'][1:])[0,1]
    ac_abs = np.corrcoef(P['absA'][:-1],P['absA'][1:])[0,1]
    print(f"TEST 3 PERSISTENCE:   AC(A)={ac_A:+.4f}  AC(|A|)={ac_abs:+.4f}")
    print(f"TEST 4 COND EXPECTANCY (top |A| decile): gross_cont={ce['top_absA']['gross_cont_mean']:+.4f}p  "
          f"net={ce['top_absA']['net_cont_mean']:+.4f}p  avg_spread={ce['top_absA']['avg_spread']:.3f}p  "
          f"wr_net={ce['top_absA']['wr_net']:.3f}")
    print(f"                       (bottom |A| decile): net={ce['bot_absA']['net_cont_mean']:+.4f}p ; "
          f"(all): net={ce['all']['net_cont_mean']:+.4f}p")

    # cross-pair sign consistency on continuation IC
    cont_ics = np.array([results[p]["cont_ic"] for p in PAIRS])
    n_pos = int((cont_ics > 0).sum()); n_neg = int((cont_ics < 0).sum())
    mag_ics = np.array([results[p]["mag_ic_vol"] for p in PAIRS])
    print(f"\nCROSS-PAIR: continuation IC sign  +{n_pos}/-{n_neg} of 12 "
          f"(mean {cont_ics.mean():+.4f}); magnitude IC(vol) mean {mag_ics.mean():+.4f} "
          f"({int((mag_ics>0).sum())}/12 positive)")

    # ── VERDICT ──
    print(f"\n{'='*78}\nVERDICT\n{'='*78}")
    cont_pass = (abs(t1_) > 2) and (mc1 < 0.05) and (wfs.count("+")>=4 or wfs.count("-")>=4) and (n_pos>=10 or n_neg>=10)
    cont_tradeable = ce['top_absA']['net_cont_mean'] > 0 and abs(ic1) > 0.02
    mag_pass = (abs(t2b) > 2) and (ic2b > 0.05) and int((mag_ics>0).sum()) >= 10
    print(f"CONTINUATION (direction) signal? {'YES' if cont_pass else 'NO'} "
          f"[t={t1_:+.2f} mc_p={mc1:.4f} wf={wfs} crosspair +{n_pos}/-{n_neg}]")
    print(f"   Net-of-spread tradeable in high-A patches? {'YES' if cont_tradeable else 'NO'} "
          f"[top-decile net {ce['top_absA']['net_cont_mean']:+.4f}p vs spread {ce['top_absA']['avg_spread']:.3f}p]")
    print(f"MAGNITUDE/VOL signal? {'YES' if mag_pass else 'NO'} "
          f"[IC(|A|,fwd_vol)={ic2b:+.4f} t={t2b:+.1f} crosspair {int((mag_ics>0).sum())}/12]")

    if cont_pass and cont_tradeable:
        verdict = "CONTINUATION — directional edge net of spread"
    elif mag_pass:
        verdict = "MAGNITUDE/VOLATILITY ONLY — use as confidence/vol gate on a separate edge, NOT a standalone direction signal"
    else:
        verdict = "NOISE — no usable direction signal; check magnitude line for gate value"
    print(f"\n>>> {verdict}")
    print(f"\n[done in {time.time()-t0:.0f}s]")

    # persist machine-readable
    out = {"verdict": verdict, "W": W, "pooled": {
        "cont_ic": ic1, "cont_t": t1_, "cont_mc_p": mc1, "cont_wf": wfs,
        "mag_ic_ret": ic2a, "mag_ic_vol": ic2b, "mag_t_vol": t2b,
        "ac_A": float(ac_A), "ac_absA": float(ac_abs),
        "top_decile_net_cont": ce['top_absA']['net_cont_mean'],
        "top_decile_spread": ce['top_absA']['avg_spread'],
        "crosspair_cont_pos": n_pos, "crosspair_cont_neg": n_neg,
        "crosspair_magvol_pos": int((mag_ics>0).sum()),
    }, "per_pair": {p: {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                        for k,v in results[p].items() if k != "cond"} for p in PAIRS}}
    with open(os.path.join(os.path.dirname(__file__), "rolling_amddp_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
