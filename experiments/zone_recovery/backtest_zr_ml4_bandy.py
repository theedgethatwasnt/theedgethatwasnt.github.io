"""
ZR ML=4 Force-Close — Howard Bandy CAR-25 + Bristle Chart
==========================================================
Uses validated GBP_USD ta/td+PSAR design (backtest_zr_ph_tight.py baseline,
7,319 p/d OOS, IS=3/3, OOS=3/3) with hard ML cap at 4 legs.

At ML=4: if a 5th leg would open, force-close ALL open legs at bar-close price.
This bounds worst-case loss per cycle but sacrifices deep-recovery cycles.

Howard Bandy metrics (bootstrap over OOS cycle sequence):
  CAR-25  : compound annual return at 25th percentile (75% of paths do better)
  MaxDD-75: max drawdown at 75th percentile (25% of paths see worse)
  P_ruin  : P(equity ever falls below ruin threshold from starting NAV)

Bristle chart: 500 semi-transparent MC equity paths + P5/P25/P50/P75/P95 bands.
"""

import math, time
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_DIR      = Path(__file__).parent

PAIR   = "GBP_USD"
PIP    = 0.0001
ZW     = 30.0
TGT    = 21.0
PF     = 1.25
BODY   = 0.5
TA     = 6.0    # 1-leg trail activation (pips MFE)
TD     = 1.0    # 1-leg trail distance from peak
AF0    = 0.01
AFST   = 0.01
AFMX   = 0.20
ML     = 4      # hard cap: force-close when leg 5 would open

OOS_FRAC = 0.30

# Howard Bandy MC parameters
N_PATHS    = 2000   # MC paths
YEAR_DAYS  = 252    # trading days per simulated year
N_YEARS    = 1      # forecast horizon in years

# Dollar conversion — calibrated from live trail-lock: $0.07/acct/day ÷ 1,388 vp/day
# Same unit sizing assumed for ta/td design (same 1 base unit = same lot size)
NAV_START      = 17.0       # starting account NAV in USD
DOLLAR_PER_VP  = 0.00005    # $ per vol-pip at current OANDA sizing (1 base unit)
RUIN_PCTS      = [0.25, 0.50, 0.75]  # ruin if NAV drops by this fraction


# ── Numba kernel: ta/td + PSAR exit, hard ML cap, force-close flag ────────────

@njit
def sim_zr_tight_ml4(op, hi, lo, cl, sp_arr,
                     pip, pf, zw, tgt, body_thresh,
                     ta, td, af0, af_step, af_max, ml):
    """
    ZR fixed-target (ta/td + PSAR) with ML=4 hard cap.
    No partial hedge (f=0 baseline from backtest_zr_ph_tight.py).

    Returns (pnl[:nc], nlegs[:nc], is_fc[:nc], nc)
      is_fc[i]=1  → force-close (leg ML+1 boundary triggered)
      is_fc[i]=0  → normal trail exit or PSAR exit
    """
    n = len(cl)
    pnl_out   = np.zeros(n, dtype=np.float64)
    nlegs_out = np.zeros(n, dtype=np.int32)
    fc_out    = np.zeros(n, dtype=np.int8)
    nc = 0

    lv = np.zeros(ml + 1, dtype=np.float64)
    ld = np.zeros(ml + 1, dtype=np.float64)
    lp = np.zeros(ml + 1, dtype=np.float64)

    i = 0
    d = 1  # alternating entry direction

    while i < n:
        # Body absorption gate
        op_i = op[i]; hi_i = hi[i]; lo_i = lo[i]; cl_i = cl[i]
        rng_i = hi_i - lo_i
        if body_thresh > 0.0 and rng_i > 1e-10:
            adv = (op_i - cl_i) if (d == 1 and op_i > cl_i) else \
                  (cl_i - op_i) if (d == -1 and cl_i > op_i) else 0.0
            if adv / rng_i > body_thresh:
                i += 1; continue

        # Open cycle
        e = cl_i; fd = float(d)
        if d == 1:
            uz = e;              lz = e - zw * pip
            ut = e + tgt * pip;  lt = lz - tgt * pip
        else:
            lz = e;              uz = e + zw * pip
            lt = e - tgt * pip;  ut = uz + tgt * pip

        lv[0] = 1.0; ld[0] = fd; lp[0] = e
        nl   = 1
        lu   = -1; ll = -1    # zone-cross guard (bar index)

        peak_mfe = 0.0
        ton      = False

        psar_on  = False
        psar_val = 0.0
        ep_val   = 0.0
        af_cur   = af0
        net_dir  = 0.0

        ex = False
        i += 1

        while i < n and not ex:
            h = hi[i]; l = lo[i]; c = cl[i]; sp = sp_arr[i]
            bull = (c >= op[i])

            # 1. PSAR exit (highest priority when active)
            if psar_on:
                if net_dir > 0:
                    if h > ep_val:
                        ep_val = h
                        af_cur = min(af_cur + af_step, af_max)
                    psar_val = ep_val - (ep_val - psar_val) * af_cur
                    if l <= psar_val:
                        net = 0.0; tv = 0.0
                        for k in range(nl):
                            net += lv[k] * ld[k] * (psar_val - lp[k]) / pip
                            tv  += lv[k]
                        pnl_out[nc] = net - tv * sp
                        nlegs_out[nc] = nl; fc_out[nc] = 0
                        nc += 1; ex = True; break
                else:
                    if l < ep_val:
                        ep_val = l
                        af_cur = min(af_cur + af_step, af_max)
                    psar_val = ep_val + (psar_val - ep_val) * af_cur
                    if h >= psar_val:
                        net = 0.0; tv = 0.0
                        for k in range(nl):
                            net += lv[k] * ld[k] * (psar_val - lp[k]) / pip
                            tv  += lv[k]
                        pnl_out[nc] = net - tv * sp
                        nlegs_out[nc] = nl; fc_out[nc] = 0
                        nc += 1; ex = True; break
                i += 1; continue

            # 2. 1-leg ta/td trailing stop
            if nl == 1:
                mfe = (h - e) / pip if d == 1 else (e - l) / pip
                if mfe > peak_mfe: peak_mfe = mfe
                if peak_mfe >= ta: ton = True
                if ton:
                    if d == 1:
                        be = e + sp * pip
                        ts = e + (peak_mfe - td) * pip
                        if ts < be: ts = be
                        if l <= ts:
                            pnl_out[nc] = (ts - e) / pip - sp
                            nlegs_out[nc] = 1; fc_out[nc] = 0
                            nc += 1; ex = True; break
                    else:
                        be = e - sp * pip
                        ts = e - (peak_mfe - td) * pip
                        if ts > be: ts = be
                        if h >= ts:
                            pnl_out[nc] = (e - ts) / pip - sp
                            nlegs_out[nc] = 1; fc_out[nc] = 0
                            nc += 1; ex = True; break
            if ex: break

            # 3. Intra-bar zone crossings
            for pass_idx in range(2):
                if ex: break
                is_hi = (bull == (pass_idx == 0))
                px    = h if is_hi else l

                # Upper crossing → LONG recovery (net short)
                if is_hi and px >= uz and lu != i:
                    lu = i
                    net_at = 0.0; tv_at = 0.0
                    for k in range(nl):
                        net_at += lv[k] * ld[k] * (ut - lp[k]) / pip
                        tv_at  += lv[k]
                    net_at -= tv_at * sp
                    if nl >= ml:
                        # FORCE CLOSE at bar-close price
                        net2 = 0.0; tv2 = 0.0
                        for k in range(nl):
                            net2 += lv[k] * ld[k] * (c - lp[k]) / pip
                            tv2  += lv[k]
                        pnl_out[nc] = net2 - tv2 * sp
                        nlegs_out[nc] = nl; fc_out[nc] = 1
                        nc += 1; ex = True; break
                    if net_at < 0.0:
                        npu = max(tgt - sp, 1e-8)
                        v = max(1.0, math.ceil(-net_at / npu * pf))
                        lv[nl] = v; ld[nl] = 1.0; lp[nl] = uz; nl += 1

                if ex: break

                # Lower crossing → SHORT recovery (net long)
                if (not is_hi) and px <= lz and ll != i:
                    ll = i
                    nat = 0.0; tvt = 0.0
                    for k in range(nl):
                        nat += lv[k] * ld[k] * (lt - lp[k]) / pip
                        tvt += lv[k]
                    nat -= tvt * sp
                    if nl >= ml:
                        # FORCE CLOSE at bar-close price
                        net2 = 0.0; tv2 = 0.0
                        for k in range(nl):
                            net2 += lv[k] * ld[k] * (c - lp[k]) / pip
                            tv2  += lv[k]
                        pnl_out[nc] = net2 - tv2 * sp
                        nlegs_out[nc] = nl; fc_out[nc] = 1
                        nc += 1; ex = True; break
                    if nat < 0.0:
                        npu = max(tgt - sp, 1e-8)
                        v = max(1.0, math.ceil(-nat / npu * pf))
                        lv[nl] = v; ld[nl] = -1.0; lp[nl] = lz; nl += 1

            if ex: break

            # 4d. Target cross → activate PSAR (bar-range check, same as ph_tight)
            if not psar_on:
                if l <= ut <= h:
                    net_v = 0.0
                    for k in range(nl): net_v += lv[k] * ld[k]
                    net_dir  = 1.0 if net_v >= 0.0 else -1.0
                    psar_on  = True; af_cur = af0; ep_val = ut
                    psar_val = ut - tgt * pip if net_dir > 0 else ut + tgt * pip
                elif l <= lt <= h:
                    net_v = 0.0
                    for k in range(nl): net_v += lv[k] * ld[k]
                    net_dir  = 1.0 if net_v >= 0.0 else -1.0
                    psar_on  = True; af_cur = af0; ep_val = lt
                    psar_val = lt - tgt * pip if net_dir > 0 else lt + tgt * pip

            i += 1
        d = -d

    return pnl_out[:nc], nlegs_out[:nc], fc_out[:nc], nc


# ── Howard Bandy MC bootstrap ─────────────────────────────────────────────────

def bandy_mc(cycle_pnl, cycle_fc, cycles_per_day, n_paths, year_days,
             nav_usd, dollar_per_vp, ruin_pcts, rng, n_plot=500):
    """
    Bootstrap MC. Returns equity curves (n_plot paths) + stats dict.
    y-axis in USD, x-axis = cycle number.
    """
    n_cycles_year = int(cycles_per_day * year_days)
    nc = len(cycle_pnl)

    eq_curves   = np.zeros((n_plot, n_cycles_year + 1), dtype=np.float32)
    final_eqs   = np.zeros(n_paths)
    min_eqs     = np.zeros(n_paths)
    max_dds     = np.zeros(n_paths)
    fc_counts   = np.zeros(n_paths, dtype=int)

    for p in range(n_paths):
        idx = rng.integers(0, nc, size=n_cycles_year)
        pnl  = cycle_pnl[idx] * dollar_per_vp
        is_fc = cycle_fc[idx]

        eq = np.empty(n_cycles_year + 1)
        eq[0] = nav_usd
        for t in range(n_cycles_year):
            eq[t + 1] = eq[t] + pnl[t]

        peak  = nav_usd
        max_dd = 0.0
        min_eq = nav_usd
        for t in range(1, n_cycles_year + 1):
            if eq[t] > peak: peak = eq[t]
            dd = (peak - eq[t]) / peak if peak > 0 else 0.0
            if dd > max_dd: max_dd = dd
            if eq[t] < min_eq: min_eq = eq[t]

        final_eqs[p]  = eq[-1]
        min_eqs[p]    = min_eq
        max_dds[p]    = max_dd
        fc_counts[p]  = int(is_fc.sum())
        if p < n_plot:
            eq_curves[p] = eq.astype(np.float32)

    # Howard Bandy metrics
    car25   = (np.percentile(final_eqs, 25) / nav_usd - 1.0) * 100.0
    car50   = (np.percentile(final_eqs, 50) / nav_usd - 1.0) * 100.0
    car75   = (np.percentile(final_eqs, 75) / nav_usd - 1.0) * 100.0
    maxdd75 = np.percentile(max_dds, 75) * 100.0
    maxdd50 = np.percentile(max_dds, 50) * 100.0

    p_ruins = {}
    for r in ruin_pcts:
        ruin_level = nav_usd * (1.0 - r)
        p_ruins[r] = float(np.mean(min_eqs < ruin_level))

    return {
        'eq_curves'    : eq_curves,
        'final_eqs'    : final_eqs,
        'min_eqs'      : min_eqs,
        'max_dds'      : max_dds,
        'fc_counts'    : fc_counts,
        'car25'        : car25,
        'car50'        : car50,
        'car75'        : car75,
        'maxdd75'      : maxdd75,
        'maxdd50'      : maxdd50,
        'p_ruins'      : p_ruins,
        'n_cycles_year': n_cycles_year,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_bristle_chart(res, cycle_pnl, cycle_fc, cycle_nlegs, nav_usd,
                       dollar_per_vp, out_path):

    BG  = '#0d1117'
    PAN = '#161b22'
    plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white',
                         'xtick.color': 'white', 'ytick.color': 'white',
                         'axes.edgecolor': '#444'})

    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    eq_curves    = res['eq_curves'].astype(np.float64)
    n_plot       = eq_curves.shape[0]
    n_cyc        = res['n_cycles_year']
    final_eqs    = res['final_eqs']
    car25        = res['car25']
    car50        = res['car50']
    car75        = res['car75']
    maxdd75      = res['maxdd75']
    maxdd50      = res['maxdd50']
    p_ruins      = res['p_ruins']

    x = np.arange(n_cyc + 1)

    # ── Panel 1: Bristle chart ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(PAN)

    norm = (final_eqs[:n_plot] - nav_usd) / nav_usd
    for i in range(n_plot):
        c = '#2dba4e' if norm[i] > 0 else '#f85149'
        ax1.plot(x, eq_curves[i], color=c, alpha=0.015, linewidth=0.4)

    # Percentile bands
    pcts = np.percentile(eq_curves, [5, 25, 50, 75, 95], axis=0)
    spec = [('#ff4d4d', 'P5',  1.8), ('#ff9999', 'P25', 1.4),
            ('white',   'P50', 2.5), ('#99ff99', 'P75', 1.4),
            ('#4dff4d', 'P95', 1.8)]
    for (col, lbl, lw), p in zip(spec, pcts):
        ax1.plot(x, p, color=col, linewidth=lw, label=lbl, zorder=5)

    ax1.axhline(nav_usd, color='yellow', ls='--', lw=1.2, label=f'Start ${nav_usd:.0f}', zorder=6)

    # Secondary top x-axis in days
    ax1b = ax1.twiny()
    ax1b.set_xlim(ax1.get_xlim())
    tks  = np.linspace(0, n_cyc, 7, dtype=int)
    cpd  = n_cyc / YEAR_DAYS
    ax1b.set_xticks(tks)
    ax1b.set_xticklabels([f'{int(t/cpd)}d' for t in tks], color='#aaa', fontsize=8)
    ax1b.tick_params(axis='x', colors='#aaa')

    ax1.set_xlabel(f'Cycle # (1 year = {n_cyc} cycles @ {cpd:.1f}/day)')
    ax1.set_ylabel('Equity (USD)')
    ax1.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e', labelcolor='white',
               framealpha=0.8, ncol=6)

    title = (f'ZR ML=4 Force-Close — GBP_USD  ({n_plot} MC paths, 1-year horizon)\n'
             f'CAR-25 = {car25:+.0f}%   CAR-50 = {car50:+.0f}%   CAR-75 = {car75:+.0f}%   '
             f'MaxDD-75 = {maxdd75:.1f}%   '
             + '   '.join(f'P(ruin {int(r*100)}%) = {v*100:.1f}%'
                           for r, v in p_ruins.items()))
    ax1.set_title(title, fontsize=9, color='white', pad=8)

    # ── Panel 2: Final equity distribution ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PAN)

    bins = np.linspace(np.percentile(res['final_eqs'], 1),
                       np.percentile(res['final_eqs'], 99), 60)
    counts, edges = np.histogram(res['final_eqs'], bins=bins)
    mid = (edges[:-1] + edges[1:]) / 2
    colors = ['#2dba4e' if m >= nav_usd else '#f85149' for m in mid]
    ax2.barh(mid, counts, height=(edges[1]-edges[0])*0.9, color=colors, alpha=0.8)
    ax2.axhline(nav_usd,  color='yellow', ls='--', lw=1.2, label=f'Start ${nav_usd:.0f}')
    ax2.axhline(np.percentile(res['final_eqs'], 25), color='#ff9999', ls=':', lw=1.2, label='P25')
    ax2.axhline(np.percentile(res['final_eqs'], 50), color='white',   ls=':', lw=1.5, label='P50')
    ax2.axhline(np.percentile(res['final_eqs'], 75), color='#99ff99', ls=':', lw=1.2, label='P75')
    ax2.set_xlabel('Count')
    ax2.set_ylabel('Final equity ($)')
    ax2.set_title('Final equity distribution\n(1 year)', fontsize=9, color='white')
    ax2.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)

    # ── Panel 3: Cycle P&L distribution ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PAN)

    normal_pnl = cycle_pnl[cycle_fc == 0] * dollar_per_vp
    fc_pnl     = cycle_pnl[cycle_fc == 1] * dollar_per_vp

    q01 = np.percentile(cycle_pnl * dollar_per_vp, 0.5)
    q99 = np.percentile(cycle_pnl * dollar_per_vp, 99.5)
    bins_c = np.linspace(q01, q99, 80)

    ax3.hist(normal_pnl, bins=bins_c, color='#2dba4e', alpha=0.7, label=f'Normal ({len(normal_pnl):,})', density=True)
    if len(fc_pnl) > 0:
        ax3.hist(fc_pnl, bins=bins_c, color='#f85149', alpha=0.9, label=f'Force-close ({len(fc_pnl):,})', density=True)
        for v in fc_pnl:
            ax3.axvline(v, color='#ff0000', alpha=0.3, linewidth=0.8)

    ax3.axvline(0, color='yellow', ls='--', lw=1)
    ax3.set_xlabel('Cycle P&L ($)')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Cycle P&L distribution\nML=4 | FC rate = {len(fc_pnl)/len(cycle_pnl)*100:.2f}%',
                  fontsize=9, color='white')
    ax3.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)

    # ── Panel 4: Max drawdown distribution ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PAN)

    dd_pct = res['max_dds'] * 100
    ax4.hist(dd_pct, bins=60, color='#e05252', alpha=0.8, edgecolor='none')
    ax4.axvline(maxdd50, color='white',   ls='--', lw=1.5, label=f'P50={maxdd50:.1f}%')
    ax4.axvline(maxdd75, color='#ff9900', ls='--', lw=1.5, label=f'P75={maxdd75:.1f}%')
    ax4.axvline(np.percentile(dd_pct, 95), color='#ff4444', ls='--', lw=1.5,
                label=f'P95={np.percentile(dd_pct,95):.1f}%')
    ax4.set_xlabel('Max drawdown (%)')
    ax4.set_ylabel('Count')
    ax4.set_title('Max drawdown distribution\n(1-year MC paths)', fontsize=9, color='white')
    ax4.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)

    # ── Panel 5: Leg distribution + force-close stats ───────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(PAN)

    leg_vals, leg_cts = np.unique(cycle_nlegs, return_counts=True)
    fc_per_leg = {l: 0 for l in leg_vals}
    for nl, fc in zip(cycle_nlegs, cycle_fc):
        if fc: fc_per_leg[nl] += 1

    colors_leg = ['#f85149' if fc_per_leg.get(l, 0) > 0 else '#2dba4e' for l in leg_vals]
    bars = ax5.bar(leg_vals.astype(str), leg_cts, color=colors_leg, alpha=0.85, edgecolor='none')
    for bar, val, l in zip(bars, leg_cts, leg_vals):
        pct = val / len(cycle_nlegs) * 100
        fc_n = fc_per_leg.get(l, 0)
        lbl = f'{pct:.1f}%'
        if fc_n > 0: lbl += f'\n(FC={fc_n})'
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 lbl, ha='center', va='bottom', fontsize=8, color='white')

    fc_total = int(cycle_fc.sum())
    ax5.set_xlabel('Legs per cycle')
    ax5.set_ylabel('Count (OOS)')
    ax5.set_title(f'Leg distribution (OOS)\nForce-close events: {fc_total} ({fc_total/len(cycle_nlegs)*100:.2f}%)',
                  fontsize=9, color='white')

    # Add summary text box
    summary = (
        f'PARAMETERS\n'
        f'ZW={ZW:.0f}p  TGT={TGT:.0f}p  PF={PF}  body={BODY}\n'
        f'ML={ML} (force-close at leg {ML+1})\n'
        f'ta={TA}p  td={TD}p  PSAR\n\n'
        f'BANDY METRICS (1-year MC)\n'
        f'CAR-25 : {car25:+.0f}%\n'
        f'CAR-50 : {car50:+.0f}%\n'
        f'CAR-75 : {car75:+.0f}%\n'
        f'MaxDD-75: {maxdd75:.1f}%\n'
        + '\n'.join(f'P(ruin {int(r*100)}%): {v*100:.1f}%'
                    for r, v in p_ruins.items()) +
        f'\n\nDollar scaling\n'
        f'${nav_usd:.0f} NAV  ${dollar_per_vp*1e5:.1f}×10⁻⁵/vp'
    )
    ax5.text(1.05, 0.98, summary, transform=ax5.transAxes,
             fontsize=7.5, color='white', va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', alpha=0.9))

    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("Loading data...", flush=True)
    mid = (pd.read_parquet(DATA_DIR_MID / f'{PAIR}_M5.parquet')
           .sort_values('timestamp').reset_index(drop=True))
    ba  = (pd.read_parquet(DATA_DIR_BA  / f'{PAIR}_M5_BA.parquet')
           .sort_values('timestamp').reset_index(drop=True))
    mid['ts'] = mid['timestamp'].astype(str).str[:19]
    ba['ts']  = ba['timestamp'].astype(str).str[:19]
    df = mid.merge(ba[['ts', 'bid_c', 'ask_c']], on='ts', how='inner').reset_index(drop=True)
    print(f"  {len(df):,} bars  {df.timestamp.min()} → {df.timestamp.max()}")

    op = df.open.values.astype(np.float64)
    hi = df.high.values.astype(np.float64)
    lo = df.low.values.astype(np.float64)
    cl = df.close.values.astype(np.float64)
    sp = ((df.ask_c - df.bid_c) / PIP).clip(lower=0.1).values.astype(np.float64)

    nb      = len(df)
    is_end  = int(nb * (1 - OOS_FRAC))
    oos_days = (nb - is_end) / (24.0 * 12.0)
    gate    = float(np.percentile(sp[:is_end], 90))
    print(f"  IS={is_end:,} bars  OOS={nb-is_end:,} bars  oos_days={oos_days:.1f}  spread_gate={gate:.2f}p")

    # JIT warmup
    print("JIT compile...", end=' ', flush=True)
    sim_zr_tight_ml4(op[:2000], hi[:2000], lo[:2000], cl[:2000], sp[:2000],
                     PIP, PF, ZW, TGT, BODY, TA, TD, AF0, AFST, AFMX, ML)
    print("done")

    # Run on OOS only (sealed, same window as all other ZR backtests)
    print(f"\nRunning ML={ML} force-close sim on OOS...", flush=True)
    pnl, nlegs, is_fc, nc = sim_zr_tight_ml4(
        op[is_end:], hi[is_end:], lo[is_end:], cl[is_end:], sp[is_end:],
        PIP, PF, ZW, TGT, BODY, TA, TD, AF0, AFST, AFMX, ML
    )
    print(f"  OOS cycles: {nc:,}  force-close: {is_fc.sum()} ({is_fc.mean()*100:.2f}%)")

    # OOS summary stats
    ppd = pnl.sum() / oos_days
    fc_mask = is_fc == 1
    normal_mask = ~fc_mask.astype(bool)
    print(f"\n  OOS p/d (vol-pips): {ppd:,.1f}")
    print(f"  Normal cycles: n={normal_mask.sum():,}  mean={pnl[normal_mask].mean():.1f}  "
          f"P5={np.percentile(pnl[normal_mask],5):.1f}  min={pnl[normal_mask].min():.1f}")
    if fc_mask.sum() > 0:
        print(f"  Force-close cycles: n={fc_mask.sum()}  mean={pnl[fc_mask].mean():.1f}  "
              f"min={pnl[fc_mask].min():.1f}  max={pnl[fc_mask].max():.1f}")
    print(f"\n  Leg distribution:")
    for nl_val in sorted(np.unique(nlegs)):
        mask = nlegs == nl_val
        fc_n = int((is_fc[mask]).sum())
        print(f"    {nl_val} legs: {mask.sum():,} ({mask.mean()*100:.1f}%)  "
              f"mean={pnl[mask].mean():.1f}p  FC={fc_n}")

    # Compare with ML=10 (uncapped) for reference
    print(f"\nRunning uncapped ML=10 sim on OOS for comparison...", flush=True)
    pnl10, nlegs10, fc10, nc10 = sim_zr_tight_ml4(
        op[is_end:], hi[is_end:], lo[is_end:], cl[is_end:], sp[is_end:],
        PIP, PF, ZW, TGT, BODY, TA, TD, AF0, AFST, AFMX, 10
    )
    ppd10 = pnl10.sum() / oos_days
    print(f"  ML=10: cycles={nc10:,}  p/d={ppd10:,.1f}  P5={np.percentile(pnl10,5):.1f}")
    print(f"  ML=4 : cycles={nc:,}    p/d={ppd:,.1f}  P5={np.percentile(pnl,5):.1f}")
    print(f"  ML=4 cost: {(ppd-ppd10)/ppd10*100:+.1f}% p/d vs ML=10")

    # Howard Bandy MC — ML=4
    cycles_per_day = nc / oos_days
    rng = np.random.default_rng(42)
    print(f"\nHoward Bandy MC ML=4: {N_PATHS} paths × {YEAR_DAYS} days "
          f"({int(cycles_per_day*YEAR_DAYS):,} cycles/path)...", flush=True)
    res4 = bandy_mc(pnl, is_fc, cycles_per_day, N_PATHS, YEAR_DAYS,
                    NAV_START, DOLLAR_PER_VP, RUIN_PCTS, rng, n_plot=500)

    # Howard Bandy MC — ML=10 (uncapped)
    cpd10 = nc10 / oos_days
    rng2  = np.random.default_rng(43)
    fc10_zeros = np.zeros(nc10, dtype=np.int8)   # no force closes in ML=10
    print(f"Howard Bandy MC ML=10: {N_PATHS} paths × {YEAR_DAYS} days "
          f"({int(cpd10*YEAR_DAYS):,} cycles/path)...", flush=True)
    res10 = bandy_mc(pnl10, fc10_zeros, cpd10, N_PATHS, YEAR_DAYS,
                     NAV_START, DOLLAR_PER_VP, RUIN_PCTS, rng2, n_plot=500)

    # ML=10 leg distribution
    print(f"\nML=10 leg distribution (OOS):")
    for nl_val in sorted(np.unique(nlegs10)):
        mask10 = nlegs10 == nl_val
        print(f"    {nl_val} legs: {mask10.sum():,} ({mask10.mean()*100:.1f}%)  "
              f"mean={pnl10[mask10].mean():.1f}p  min={pnl10[mask10].min():.1f}p")

    print(f"\n{'='*62}")
    print(f"  HOWARD BANDY COMPARISON — GBP_USD  ZW=30 TGT=21 PF=1.25")
    print(f"{'='*62}")
    print(f"  {'Metric':<18} {'ML=4 (cap)':>14} {'ML=10 (live)':>14}")
    print(f"  {'-'*46}")
    print(f"  {'CAR-25':<18} {res4['car25']:>+13.0f}% {res10['car25']:>+13.0f}%")
    print(f"  {'CAR-50':<18} {res4['car50']:>+13.0f}% {res10['car50']:>+13.0f}%")
    print(f"  {'CAR-75':<18} {res4['car75']:>+13.0f}% {res10['car75']:>+13.0f}%")
    print(f"  {'MaxDD-50':<18} {res4['maxdd50']:>13.1f}% {res10['maxdd50']:>13.1f}%")
    print(f"  {'MaxDD-75':<18} {res4['maxdd75']:>13.1f}% {res10['maxdd75']:>13.1f}%")
    for r in RUIN_PCTS:
        lbl = f'P(ruin {int(r*100)}%)'
        print(f"  {lbl:<18} {res4['p_ruins'][r]*100:>13.2f}% {res10['p_ruins'][r]*100:>13.2f}%")
    print(f"  {'p/d vol-pips':<18} {ppd:>+13.1f}  {ppd10:>+13.1f}")
    print(f"  {'FC rate':<18} {is_fc.mean()*100:>13.2f}%  {'0.00':>13}%")
    print(f"{'='*62}")
    print(f"  Dollar scaling: ${NAV_START:.0f} NAV  ${DOLLAR_PER_VP*1e5:.1f}e-5/vp")
    print(f"  ML=10 $/day (CAR-50): ${res10['car50']/100*NAV_START/YEAR_DAYS:.4f}/account/day")

    res = res4   # use ML=4 for the primary bristle chart

    # Save bristle chart
    out_img = OUT_DIR / 'zr_ml4_bandy_bristle.png'
    print(f"\nPlotting ML=4 bristle chart...", flush=True)
    make_bristle_chart(res4, pnl, is_fc, nlegs, NAV_START, DOLLAR_PER_VP, out_img)

    # Save ML=10 bristle chart
    out_img10 = OUT_DIR / 'zr_ml10_bandy_bristle.png'
    print(f"Plotting ML=10 bristle chart...", flush=True)
    make_bristle_chart(res10, pnl10, fc10_zeros, nlegs10, NAV_START, DOLLAR_PER_VP, out_img10)

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
