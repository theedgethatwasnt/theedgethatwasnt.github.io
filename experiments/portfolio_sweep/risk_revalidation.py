#!/usr/bin/env python3
"""
Risk Re-Validation of the Deployed Momentum/Exhaust Book
=========================================================
Answers the question the existing pipeline never asked: is the backtested edge
realizable under finite margin, and does the SIGNAL actually beat random entry
under the same TP-only/no-SL exit rules?

Re-validates the 6 deployed configs (accounts 001-004, 011, 012).

Part A — Per-trade risk profile: MAE (max adverse excursion) + hold distribution,
         plus mark-to-market of the open-at-OOS-end position the old backtest dropped.
Part B — Closeout model: aggregate concurrent open drawdown across all 12 pairs,
         converted to account $ vs the live balance → would OANDA have closed us out?
Part C — Entry-randomization MC: replace the signal with random entry at the same
         fire-rate + direction split, keep identical TP/no-SL exits, compare p/d.
         Tests whether the SIGNAL has value beyond the exit structure.
Part D — Forward-return IC: does sign(signal) predict forward mid return?

All read-only on data/m5_ba. No orders, no live touch.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit, prange
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA    = PROJECT / "data" / "m5_ba"
RESULTS = Path(__file__).parent / "results"

PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY   = {"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
IS_FRAC = 0.70
UNITS_PER_DOLLAR = 1.25
MARGIN_RATE = 0.03          # ~33:1, representative OANDA major; stated assumption
CLOSEOUT_RATIO = 0.50       # OANDA closes when NAV <= 0.5 * margin_used
MC_N = 300                  # entry-randomization shuffles
IC_HORIZONS = (12, 48, 288) # 1h, 4h, 24h on M5

def pip_sz(pair): return 0.01 if pair in JPY else 0.0001

# Approx pip value per 1 unit, in USD (stated assumption — order-of-magnitude)
def pip_usd(pair):
    if pair == "EUR_GBP": return 0.0001 * 1.27       # quote GBP → USD
    if pair in JPY:       return 0.01 / 150.0        # quote JPY → USD
    return 0.0001                                    # xxx_USD quote

# ── Deployed configs ──────────────────────────────────────────────────────────
CONFIGS = [
    dict(acct="001", name="pmom H1+M30",   kind="mom", tf1="1h",    tf2="30min", sma_n=0,  lags=(8,10,20), tp=15.0, bal=2.04),
    dict(acct="002", name="sma16 M30+M15", kind="mom", tf1="30min", tf2="15min", sma_n=16, lags=(1,10,20), tp=20.0, bal=2.02),
    dict(acct="003", name="exhaust A",     kind="exh", n_consec=4,  dist_mult=2.0, tp=15.0, bal=2.04),
    dict(acct="004", name="exhaust B",     kind="exh", n_consec=2,  dist_mult=1.0, tp=10.0, bal=2.01),
    dict(acct="011", name="pmom M15+M5",   kind="mom", tf1="15min", tf2="5min",  sma_n=0,  lags=(1,3,8),  tp=10.0, bal=17.79),
    dict(acct="012", name="sma16 H1+M30",  kind="mom", tf1="1h",    tf2="30min", sma_n=16, lags=(8,10,15), tp=20.0, bal=17.01),
]
SMA_N_EXHAUST = 14


# ── Signal builders (causal, mirror deployed services) ─────────────────────────
def build_momentum_sig(df, tf1, tf2, sma_n, lags):
    moms = []
    for tf in (tf1, tf2):
        rs = df["close"].resample(tf).last().dropna()
        if sma_n > 0:
            rs = rs.rolling(sma_n, min_periods=sma_n).mean()
        rs_s = rs.shift(1).reindex(df.index, method="ffill")
        for k in lags:
            moms.append(rs_s - rs_s.shift(k))
    score = (pd.concat(moms, axis=1) > 0).sum(axis=1)
    n_ind = len(moms)
    sig = pd.Series(np.int8(0), index=df.index)
    sig[score == n_ind] = np.int8(1)
    sig[score == 0]     = np.int8(-1)
    return sig.values.astype(np.int8)


@njit(cache=True)
def compute_sma(close, period):
    n = len(close); out = np.full(n, np.nan)
    run = 0.0
    for i in range(n):
        run += close[i]
        if i >= period: run -= close[i - period]
        if i >= period - 1: out[i] = run / period
    return out


@njit(cache=True)
def build_exhaust_sig(close_p, open_p, sma, n_consec, dist_mult, sp_gate):
    n = len(close_p); sig = np.zeros(n, dtype=np.int8)
    for i in range(n_consec - 1, n - 1):
        if np.isnan(sma[i]): continue
        all_bull = True; all_bear = True
        for j in range(i - n_consec + 1, i + 1):
            if close_p[j] <= open_p[j]: all_bull = False
            if close_p[j] >= open_p[j]: all_bear = False
        dist = close_p[i] - sma[i]
        if all_bull and dist >= dist_mult * sp_gate: sig[i] = 1
        elif all_bear and (-dist) >= dist_mult * sp_gate: sig[i] = -1
    return sig


# ── Part A/B kernel: per-trade profile + open-position unrealized series ───────
@njit(cache=True)
def sim_profile(mid, low, high, bid, ask, sp, sig, tp_pips, sp_gate):
    """Returns realized pnls, holds, MAEs (per completed trade), the open_upl[t]
    series (unrealized pips of currently-open pos, 0 if flat), and the final
    open-trade mark (unrealized pips, hold, mae) the old backtest dropped."""
    n = len(mid)
    pnl = np.empty(n, dtype=np.float64); hold = np.empty(n, dtype=np.float64)
    mae = np.empty(n, dtype=np.float64); k = 0
    open_upl = np.zeros(n, dtype=np.float64)
    in_trade = False; dir_ = 0; ep = 0.0; ei = 0; worst = 0.0
    for i in range(1, n):
        if in_trade:
            # mark-to-market (mid vs entry fill), track worst adverse excursion
            adv = (low[i] - ep) * dir_ if dir_ == 1 else (ep - high[i]) * dir_ * -1.0
            # adverse = how far against us intrabar (negative = loss), in pips
            mtm_low  = (low[i]  - ep) * dir_
            mtm_high = (high[i] - ep) * dir_
            cur_worst = mtm_low if dir_ == 1 else mtm_high
            if cur_worst < worst: worst = cur_worst
            open_upl[i] = (mid[i] - ep) * dir_
            if (mid[i] - ep) * dir_ >= tp_pips:
                ex = bid[i] if dir_ == 1 else ask[i]
                pnl[k] = (ex - ep) * dir_ - sp[i]
                hold[k] = i - ei
                mae[k] = worst
                k += 1
                in_trade = False; open_upl[i] = 0.0
        else:
            nd = sig[i-1]
            if nd != 0 and sp[i] <= sp_gate:
                ep = ask[i] if nd == 1 else bid[i]
                dir_ = nd; ei = i; in_trade = True; worst = 0.0
                open_upl[i] = (mid[i] - ep) * dir_
    final_open_upl = 0.0; final_open_hold = 0.0; final_open_mae = 0.0
    if in_trade:
        final_open_upl = (mid[n-1] - ep) * dir_
        final_open_hold = (n - 1) - ei
        final_open_mae = worst
    return (pnl[:k], hold[:k], mae[:k], open_upl,
            final_open_upl, final_open_hold, final_open_mae)


# ── Part C kernel: random-entry sim (same TP/no-SL exits) ──────────────────────
@njit(cache=True)
def sim_random_entry(mid, bid, ask, sp, fire_rate, p_long, tp_pips, sp_gate, seed):
    np.random.seed(seed)
    n = len(mid); pnl_sum = 0.0; n_t = 0
    in_trade = False; dir_ = 0; ep = 0.0
    for i in range(1, n):
        if in_trade:
            if (mid[i] - ep) * dir_ >= tp_pips:
                ex = bid[i] if dir_ == 1 else ask[i]
                pnl_sum += (ex - ep) * dir_ - sp[i]; n_t += 1
                in_trade = False
        else:
            if np.random.random() < fire_rate and sp[i] <= sp_gate:
                nd = 1 if np.random.random() < p_long else -1
                ep = ask[i] if nd == 1 else bid[i]
                dir_ = nd; in_trade = True
    return pnl_sum, n_t


@njit(parallel=True, cache=True)
def random_entry_mc(mid, bid, ask, sp, fire_rate, p_long, tp_pips, sp_gate, oos_days, n_mc):
    out = np.empty(n_mc, dtype=np.float64)
    for s in prange(n_mc):
        ps, _ = sim_random_entry(mid, bid, ask, sp, fire_rate, p_long, tp_pips, sp_gate, s + 1)
        out[s] = ps / oos_days if oos_days > 0 else 0.0
    return out


def build_sig_for_config(cfg, df, pip, sp_gate):
    if cfg["kind"] == "exh":
        close_p = (df["close"].values / pip).astype(np.float64)
        open_p  = (df["open"].values  / pip).astype(np.float64)
        sma = compute_sma(close_p, SMA_N_EXHAUST)
        return build_exhaust_sig(close_p, open_p, sma,
                                 int(cfg["n_consec"]), float(cfg["dist_mult"]), sp_gate)
    return build_momentum_sig(df, cfg["tf1"], cfg["tf2"], cfg["sma_n"], cfg["lags"])


# ── Main ───────────────────────────────────────────────────────────────────────
print("Loading 12-pair M5 BA …")
cache = {}
for pair in PAIRS:
    df = pd.read_parquet(DATA / f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
    df = df.astype({c: "float64" for c in df.select_dtypes("float32").columns})
    pip = pip_sz(pair); n_is = int(len(df) * IS_FRAC)
    sp_gate = float(np.percentile((df["ask_c"] - df["bid_c"]).iloc[:n_is] / pip, 90))
    cache[pair] = dict(df=df, pip=pip, n_is=n_is, sp_gate=sp_gate,
                       oos_days=len(df.iloc[n_is:]) / 288.0)
print(f"  loaded {len(PAIRS)} pairs.\n")

summary = []
for cfg in CONFIGS:
    units = max(1, round(cfg["bal"] * UNITS_PER_DOLLAR))
    print("=" * 78)
    print(f"  {cfg['acct']}  {cfg['name']}   TP={cfg['tp']}p   bal=${cfg['bal']:.2f}  units={units}")
    print("=" * 78)

    all_pnl = []; all_hold = []; all_mae = []
    dropped_upl_pips = 0.0; dropped_mae_pips = 0.0; n_open_end = 0
    open_upl_by_pair = {}
    real_oos_pd = 0.0; mc_beat = 0; ic_by_h = {h: [] for h in IC_HORIZONS}

    for pair in PAIRS:
        c = cache[pair]; df = c["df"]; pip = c["pip"]; sg = c["sp_gate"]; n_is = c["n_is"]
        sig = build_sig_for_config(cfg, df, pip, sg)

        mid = (df["close"].values / pip).astype(np.float64)
        low = (df["low"].values   / pip).astype(np.float64)
        high= (df["high"].values  / pip).astype(np.float64)
        bid = (df["bid_c"].values / pip).astype(np.float64)
        ask = (df["ask_c"].values / pip).astype(np.float64)
        sp  = ask - bid

        # OOS slices
        m_o, l_o, h_o = mid[n_is:], low[n_is:], high[n_is:]
        b_o, a_o, sp_o, sig_o = bid[n_is:], ask[n_is:], sp[n_is:], sig[n_is:]

        (pnl, hold, mae, open_upl,
         f_upl, f_hold, f_mae) = sim_profile(m_o, l_o, h_o, b_o, a_o, sp_o, sig_o, cfg["tp"], sg)

        all_pnl.append(pnl); all_hold.append(hold); all_mae.append(mae)
        open_upl_by_pair[pair] = pd.Series(open_upl, index=df.index[n_is:])
        if f_hold > 0:
            n_open_end += 1; dropped_upl_pips += f_upl; dropped_mae_pips += f_mae

        oos_days = c["oos_days"]
        pd_real = pnl.sum() / oos_days if oos_days > 0 else 0.0
        real_oos_pd += pd_real

        # Part C — random-entry MC for this pair
        fire_rate = float((sig_o != 0).mean())
        n_long = int((sig_o == 1).sum()); n_act = int((sig_o != 0).sum())
        p_long = n_long / n_act if n_act > 0 else 0.5
        rnd_pd = random_entry_mc(m_o, b_o, a_o, sp_o, fire_rate, p_long,
                                  cfg["tp"], sg, oos_days, MC_N)
        # accumulate at portfolio level below; store per-pair real vs rnd mean
        cache[pair]["_rnd_pd"] = rnd_pd
        cache[pair]["_real_pd"] = pd_real

        # Part D — forward IC
        s_full = sig.astype(np.float64)
        for H in IC_HORIZONS:
            fwd = np.full(len(mid), np.nan)
            fwd[:-H] = (mid[H:] - mid[:-H])
            m = (sig != 0) & ~np.isnan(fwd)
            m[:n_is] = False  # OOS only
            if m.sum() > 50:
                a = s_full[m]; bv = fwd[m]
                if a.std() > 0 and bv.std() > 0:
                    ic_by_h[H].append(float(np.corrcoef(a, bv)[0, 1]))

    # ── Part A aggregate ──
    pnl_all  = np.concatenate(all_pnl) if all_pnl else np.array([0.0])
    mae_all  = np.concatenate(all_mae) if all_mae else np.array([0.0])
    hold_all = np.concatenate(all_hold) if all_hold else np.array([0.0])
    hold_h   = hold_all * 5 / 60.0
    n_tr = len(pnl_all)
    wr = float((pnl_all > 0).mean() * 100) if n_tr else 0.0
    print(f"\n  [A] Per-trade profile (OOS, {n_tr} completed trades):")
    print(f"      WR={wr:.1f}%  p/d(realized winners only)={real_oos_pd:+.1f}")
    print(f"      MAE pips  p50={np.percentile(mae_all,50):.0f}  p90={np.percentile(mae_all,10):.0f}  "
          f"p99={np.percentile(mae_all,1):.0f}  worst={mae_all.min():.0f}")
    for thr in (15, 30, 50, 100):
        print(f"      trades with MAE >= {thr:>3}p: {(mae_all <= -thr).mean()*100:5.1f}%")
    print(f"      hold(h)   p50={np.percentile(hold_h,50):.1f}  p90={np.percentile(hold_h,90):.1f}  "
          f"max={hold_h.max():.1f}")
    print(f"      dropped open-at-end: {n_open_end} pairs, "
          f"total unrealized={dropped_upl_pips:+.0f}p, summed MAE={dropped_mae_pips:+.0f}p")

    # ── Part B closeout model ──
    upl_df = pd.DataFrame(open_upl_by_pair).fillna(0.0)
    # convert each pair's open unrealized pips → USD: pips * units * pip_usd(pair)
    usd = pd.DataFrame({p: upl_df[p] * units * pip_usd(p) for p in upl_df.columns})
    agg_usd = usd.sum(axis=1)                          # aggregate unrealized $ across all open pos
    n_open  = (upl_df != 0).sum(axis=1)                # concurrent open positions
    nav = cfg["bal"] + agg_usd
    margin_used = n_open * units * MARGIN_RATE * 1.0   # ≈ units*notional(~$1)*rate per open pos
    worst_agg_loss_usd = -agg_usd.min()                # max aggregate unrealized loss ($)
    worst_agg_loss_pips = -upl_df.sum(axis=1).min()
    max_concurrent = int(n_open.max())
    nav_min = nav.min()
    wiped = worst_agg_loss_usd >= cfg["bal"]
    closeout_mask = nav <= CLOSEOUT_RATIO * margin_used
    closeout_ever = bool(closeout_mask.any())
    closeout_when = str(closeout_mask[closeout_mask].index[0])[:10] if closeout_ever else "—"
    print(f"\n  [B] Finite-margin closeout model (bal=${cfg['bal']:.2f}, units={units}, "
          f"margin_rate={MARGIN_RATE}):")
    print(f"      max concurrent open positions : {max_concurrent}/12")
    print(f"      worst aggregate open DD       : {worst_agg_loss_pips:.0f} pips  =  ${worst_agg_loss_usd:.2f}")
    print(f"      min NAV over OOS              : ${nav_min:.2f}")
    print(f"      NAV<0 (account wiped)?        : {'YES ⛔' if wiped else 'no'}")
    print(f"      OANDA closeout fired?         : {'YES ⛔ first ~'+closeout_when if closeout_ever else 'no'}")

    # ── Part C portfolio random-entry MC ──
    rnd_port = np.zeros(MC_N)
    real_port = 0.0
    for pair in PAIRS:
        rnd_port += cache[pair]["_rnd_pd"]
        real_port += cache[pair]["_real_pd"]
    pct = float((rnd_port >= real_port).mean())
    print(f"\n  [C] Entry-randomization MC ({MC_N} shuffles, same TP/no-SL exits):")
    print(f"      real signal p/d  = {real_port:+.1f}")
    print(f"      random-entry p/d = {rnd_port.mean():+.1f}  (p5={np.percentile(rnd_port,5):+.1f}, "
          f"p95={np.percentile(rnd_port,95):+.1f})")
    print(f"      P(random >= real) = {pct:.3f}   "
          f"{'signal NO better than random ❌' if pct >= 0.05 else 'signal beats random ✅'}")

    # ── Part D IC ──
    print(f"\n  [D] Forward-return IC of sign(signal), OOS, mean over pairs:")
    ic_line = "      "
    for H in IC_HORIZONS:
        vals = ic_by_h[H]; m = float(np.mean(vals)) if vals else float("nan")
        ic_line += f"H={H}({H*5//60}h): IC={m:+.4f}   "
    print(ic_line)

    summary.append(dict(
        acct=cfg["acct"], name=cfg["name"], tp=cfg["tp"], units=units, bal=cfg["bal"],
        wr=round(wr,1), n_trades=n_tr, real_pd=round(real_port,1),
        mae_p90=round(float(np.percentile(mae_all,10)),0),
        mae_worst=round(float(mae_all.min()),0),
        hold_p90_h=round(float(np.percentile(hold_h,90)),1),
        max_concurrent=max_concurrent,
        worst_agg_dd_pips=round(float(worst_agg_loss_pips),0),
        worst_agg_dd_usd=round(float(worst_agg_loss_usd),2),
        nav_min=round(float(nav_min),2), wiped=wiped, closeout=closeout_ever,
        rnd_pd_mean=round(float(rnd_port.mean()),1),
        mc_p_random=round(pct,3),
        ic_24h=round(float(np.mean(ic_by_h[288])) if ic_by_h[288] else float("nan"),4),
    ))
    print()

sm = pd.DataFrame(summary)
RESULTS.mkdir(exist_ok=True)
sm.to_csv(RESULTS / "risk_revalidation.csv", index=False)
print("=" * 78)
print("SUMMARY  →  results/risk_revalidation.csv")
print("=" * 78)
print(sm.to_string(index=False))
