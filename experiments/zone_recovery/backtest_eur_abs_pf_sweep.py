"""
EUR_USD Absorption PF Sweep
============================
Best validated absorption config: body=0.5, wick=0.00, mom_bars=3
  → 2,120.7 p/d, P5=532.8, IS=3/3, OOS=3/3 (at PF=1.25)

Question: what happens to the edge at lower PF (safer progression)?
  PF=1.25 → baseline (Session 036 result)
  PF=1.10 → moderately tighter
  PF=1.05 → very tight
  PF=1.50 → more aggressive (comparison)

Adds CumL5 and risk_adj_p5 to quantify capital risk at each PF.
Compare to no-filter baseline at each PF.
"""
import math, sys, itertools
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_PATH     = Path(__file__).parent / 'results/eur_abs_pf_results.csv'
OUT_PATH.parent.mkdir(exist_ok=True)

ML       = 10
OOS_FRAC = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 500
N_PERM     = 200

AF0  = 0.01; AFST = 0.01; AFMX = 0.20

PAIR = "EUR_USD"; PIP = 0.0001; ZW = 30.0; TGT = 21.0; TA = 10.0; TD = 1.0

PFS = [1.05, 1.10, 1.25, 1.50]

# Configs to test: (body, wick, mom) label
CONFIGS = [
    (0.0, 0.0, 0, "baseline"),
    (0.5, 0.0, 3, "body+mom3"),
]


def compute_cum_l5(zw, tgt, pf, spread):
    net_per_unit = tgt - spread
    if net_per_unit <= 0: return float('inf')
    uz=0.0; lz=-zw; ut=+tgt; lt=-zw-tgt
    legs = [(1.0, +1.0, uz)]
    for _ in range(4):
        target = lt if len(legs) % 2 == 1 else ut
        net = sum(v*d*(target-p) for v,d,p in legs)
        net -= sum(v for v,d,p in legs) * spread
        if net >= 0: break
        vol_new = max(1.0, math.ceil(-net / net_per_unit * pf))
        if len(legs) % 2 == 1: legs.append((vol_new, -1.0, lz))
        else: legs.append((vol_new, +1.0, uz))
    return sum(v for v,d,p in legs)


def growth_rate(zw, tgt, pf, spread):
    net_per = tgt - spread
    if net_per <= 0: return float('inf')
    return pf * (zw + tgt) / net_per


@njit
def check_absorption(op, hi, lo, cl, i, d, body_thresh, wick_thresh, mom_bars):
    rng = hi[i] - lo[i]; eps = 1e-10
    if body_thresh > 0.0 and rng > eps:
        if d == 1: adverse = max(0.0, op[i] - cl[i]) / rng
        else:      adverse = max(0.0, cl[i] - op[i]) / rng
        if adverse > body_thresh: return False
    if wick_thresh > 0.0 and rng > eps:
        lo_body = cl[i] if cl[i] < op[i] else op[i]
        hi_body = cl[i] if cl[i] > op[i] else op[i]
        if d == 1:
            if (lo_body - lo[i]) / rng < wick_thresh: return False
        else:
            if (hi[i] - hi_body) / rng < wick_thresh: return False
    if mom_bars > 0 and i >= mom_bars:
        all_adv = True
        for j in range(1, mom_bars + 1):
            prev = i - j
            if d == 1:
                if cl[prev] >= op[prev]: all_adv = False; break
            else:
                if cl[prev] <= op[prev]: all_adv = False; break
        if all_adv: return False
    return True


@njit
def sim_zr(op, hi, lo, cl, sp_arr, pip, pf, ml, zw, tgt, ta, td,
           af0, af_step, af_max, body_thresh, wick_thresh, mom_bars):
    n     = len(cl)
    pnl   = np.zeros(n, dtype=np.float64)
    nlegs = np.zeros(n, dtype=np.int32)
    etype = np.zeros(n, dtype=np.int32)
    nc = 0; n_skip = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1

    while i < n:
        if not check_absorption(op, hi, lo, cl, i, d, body_thresh, wick_thresh, mom_bars):
            n_skip += 1; i += 1; continue

        e = cl[i]
        if d == 1: uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:      lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak=0.0; ton=False
        psar_on=False; psar_val=0.0; ep_val=0.0; af_cur=af0; net_dir=0.0
        i += 1

        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; sp=sp_arr[i]; bull=(c>=op[i])

            if psar_on:
                if net_dir > 0:
                    if h > ep_val: ep_val=h; af_cur=min(af_cur+af_step, af_max)
                    psar_val = ep_val - (ep_val - psar_val) * af_cur
                    if l <= psar_val:
                        net=0.0; tv=0.0
                        for k in range(nl): net+=lv[k]*ld[k]*(psar_val-lp[k])/pip; tv+=lv[k]
                        pnl[nc]=net-tv*sp; nlegs[nc]=nl; etype[nc]=5; nc+=1; ex=True
                else:
                    if l < ep_val: ep_val=l; af_cur=min(af_cur+af_step, af_max)
                    psar_val = ep_val + (psar_val - ep_val) * af_cur
                    if h >= psar_val:
                        net=0.0; tv=0.0
                        for k in range(nl): net+=lv[k]*ld[k]*(psar_val-lp[k])/pip; tv+=lv[k]
                        pnl[nc]=net-tv*sp; nlegs[nc]=nl; etype[nc]=5; nc+=1; ex=True
                if ex: break
                i += 1; continue

            if nl == 1:
                mfe = (h-e)/pip if d==1 else (e-l)/pip
                if mfe > peak: peak = mfe
                if peak >= ta: ton = True
                if ton:
                    if d == 1:
                        be=e+sp*pip; ts=e+(peak-td)*pip
                        if ts < be: ts = be
                        if l <= ts: pnl[nc]=(ts-e)/pip-sp; nlegs[nc]=1; etype[nc]=1; nc+=1; ex=True
                    else:
                        be=e-sp*pip; ts=e-(peak-td)*pip
                        if ts > be: ts = be
                        if h >= ts: pnl[nc]=(e-ts)/pip-sp; nlegs[nc]=1; etype[nc]=1; nc+=1; ex=True
            if ex: break

            for pi2 in range(2):
                if ex: break
                is_hi = (bull == (pi2 == 0))
                if is_hi and h >= uz and lu != i:
                    lu = i; net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    net -= tv*sp
                    if net < 0:
                        npu=max(tgt-sp,1e-8); v=max(1.0,math.ceil(-net/npu*pf))
                        if nl >= ml:
                            nc2=0.0; tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp; nlegs[nc]=nl; etype[nc]=3; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if (not is_hi) and l <= lz and ll != i:
                    ll = i; net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    net -= tv*sp
                    if net < 0:
                        npu=max(tgt-sp,1e-8); v=max(1.0,math.ceil(-net/npu*pf))
                        if nl >= ml:
                            nc2=0.0; tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp; nlegs[nc]=nl; etype[nc]=3; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
                if ex: break
                if l <= ut <= h:
                    net_v=0.0
                    for k in range(nl): net_v+=lv[k]*ld[k]
                    net_dir=1.0 if net_v>=0 else -1.0
                    psar_on=True; af_cur=af0; ep_val=ut
                    psar_val=ut-tgt*pip if net_dir>0 else ut+tgt*pip
                    break
                if l <= lt <= h:
                    net_v=0.0
                    for k in range(nl): net_v+=lv[k]*ld[k]
                    net_dir=1.0 if net_v>=0 else -1.0
                    psar_on=True; af_cur=af0; ep_val=lt
                    psar_val=lt-tgt*pip if net_dir>0 else lt+tgt*pip
                    break
            i += 1

        d = -d

    return pnl[:nc], nlegs[:nc], etype[:nc], nc, n_skip


# JIT warm-up
_df0 = pd.read_parquet(DATA_DIR_MID/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df0['open'].values[:2000].astype(np.float64)
_h=_df0['high'].values[:2000].astype(np.float64)
_l=_df0['low'].values[:2000].astype(np.float64)
_c=_df0['close'].values[:2000].astype(np.float64)
_s=np.full(2000,0.00016,dtype=np.float64)
sim_zr(_o,_h,_l,_c,_s,0.0001,1.25,10,30.,21.,10.,1.,0.01,0.01,0.20,0.0,0.0,0)
print("JIT compiled.", flush=True)


def permutation_p(pnl, n_perm):
    if len(pnl) == 0: return 1.0
    obs = pnl.sum(); cnt = 0
    for _ in range(n_perm):
        signs = np.random.choice([-1,1], size=len(pnl))
        if (pnl*signs).sum() >= obs: cnt += 1
    return cnt / n_perm


def run_wf_boot(pnl, nc, n_days, n_chunks, n_boot):
    chunk_size = nc // n_chunks
    if chunk_size == 0: return 0, 0.0, 0.0, 0.0
    wf_pass = sum(1 for ch in range(n_chunks)
                  if pnl[ch*chunk_size : (ch+1)*chunk_size if ch<n_chunks-1 else nc].sum() > 0)
    ppd = pnl.sum() / max(n_days, 1)
    sums = np.array([np.random.choice(pnl, size=len(pnl), replace=True).sum() for _ in range(n_boot)])
    p5   = float(np.percentile(sums / max(n_days, 1), 5))
    p_pos = float((sums > 0).mean())
    return wf_pass, ppd, p5, p_pos


# Load data
mid = pd.read_parquet(DATA_DIR_MID/f'{PAIR}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
ba  = pd.read_parquet(DATA_DIR_BA /f'{PAIR}_M5_BA.parquet').sort_values('timestamp').reset_index(drop=True)
mid['ts_key'] = mid['timestamp'].astype(str).str[:19]
ba['ts_key']  = ba['timestamp'].astype(str).str[:19]
merged = mid.merge(ba[['ts_key','bid_c','ask_c']], on='ts_key', how='left')
merged['spread'] = ((merged['ask_c'] - merged['bid_c']) / PIP).clip(0, 50)
merged['spread'] = merged['spread'].fillna(merged['spread'].median())

op = merged['open'].values.astype(np.float64)
hi = merged['high'].values.astype(np.float64)
lo = merged['low'].values.astype(np.float64)
cl = merged['close'].values.astype(np.float64)
sp = merged['spread'].values.astype(np.float64)

n_total = len(cl)
n_oos   = int(n_total * OOS_FRAC)
n_is    = n_total - n_oos
oos_days= n_oos * 5 / (60 * 24)
sp_med  = float(np.median(sp[n_is:]))

print(f"\nEUR_USD PF sweep with absorption filter (body=0.5, mom=3)")
print(f"OOS={oos_days:.0f}d  median_spread={sp_med:.2f}p")

rows = []
print(f"\n{'config':>12} {'pf':>5} | {'p/d':>8} {'P5':>8} {'nc':>6} {'skip%':>6} "
      f"{'1L%':>5} {'5+%':>4} {'IS':>4} {'OOS':>4} {'perm_p':>7} "
      f"{'CumL5':>7} {'growR':>6} {'rAdjP5':>8} | status")
print('-'*105)

for body, wick, mom, label in CONFIGS:
    for pf in PFS:
        # IS walk-forward
        is_wf = 0; is_chunk = n_is // IS_CHUNKS
        for ch in range(IS_CHUNKS):
            s = ch * is_chunk
            e = s + is_chunk if ch < IS_CHUNKS-1 else n_is
            pnl_c, _, _, nc_c, _ = sim_zr(op[s:e],hi[s:e],lo[s:e],cl[s:e],sp[s:e],
                                           PIP,pf,ML,ZW,TGT,TA,TD,AF0,AFST,AFMX,body,wick,mom)
            if pnl_c.sum() > 0: is_wf += 1

        # Full OOS
        pnl_oos, nl_oos, et_oos, nc_oos, n_skip_oos = sim_zr(
            op[n_is:],hi[n_is:],lo[n_is:],cl[n_is:],sp[n_is:],
            PIP,pf,ML,ZW,TGT,TA,TD,AF0,AFST,AFMX,body,wick,mom)

        # OOS walk-forward
        oos_wf = 0; oos_chunk = nc_oos // OOS_CHUNKS
        if oos_chunk > 0:
            for ch in range(OOS_CHUNKS):
                s = ch * oos_chunk
                e = s + oos_chunk if ch < OOS_CHUNKS-1 else nc_oos
                if pnl_oos[s:e].sum() > 0: oos_wf += 1

        _, ppd, p5, p_pos = run_wf_boot(pnl_oos, nc_oos, oos_days, OOS_CHUNKS, N_BOOT)
        perm_p = permutation_p(pnl_oos, N_PERM)

        n_bars_oos = n_total - n_is
        skip_pct  = n_skip_oos / max(n_bars_oos, 1) * 100
        one_leg_pct= (nl_oos == 1).sum() / max(nc_oos, 1) * 100
        five_plus_pct=(nl_oos >= 5).sum() / max(nc_oos, 1) * 100

        cum_l5  = compute_cum_l5(ZW, TGT, pf, sp_med)
        gr      = growth_rate(ZW, TGT, pf, sp_med)
        r_adj_p5= p5 / cum_l5 if (cum_l5 < 1e6 and p5 > 0) else 0.0

        gates_ok = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
                    and p5 > 0 and p_pos > 0.95 and perm_p < 0.05)
        if gates_ok: status = "🟢 PASS"
        else:
            fails = []
            if is_wf < IS_CHUNKS: fails.append(f"IS:{is_wf}/{IS_CHUNKS}")
            if oos_wf < OOS_CHUNKS: fails.append(f"OOS:{oos_wf}/{OOS_CHUNKS}")
            if p5 <= 0: fails.append("P5≤0")
            if p_pos <= 0.95: fails.append(f"P+={p_pos:.2f}")
            if perm_p >= 0.05: fails.append(f"p={perm_p:.2f}")
            status = " | ".join(fails)

        tag = f"{label}/pf{pf:.2f}"
        print(f"{tag:>14} {pf:>5.2f} | {ppd:>8.1f} {p5:>8.1f} {nc_oos:>6} {skip_pct:>6.1f}% "
              f"{one_leg_pct:>5.1f}% {five_plus_pct:>4.1f}% {is_wf:>2}/{IS_CHUNKS} {oos_wf:>2}/{OOS_CHUNKS} "
              f"{perm_p:>7.3f} {cum_l5:>7.0f} {gr:>6.2f} {r_adj_p5:>8.3f} | {status}")

        rows.append(dict(label=label, pf=pf, body=body, wick=wick, mom=mom,
                         ppd=ppd, p5=p5, nc=nc_oos, skip_pct=skip_pct,
                         one_leg_pct=one_leg_pct, five_plus_pct=five_plus_pct,
                         is_wf=is_wf, oos_wf=oos_wf, perm_p=perm_p,
                         cum_l5=cum_l5, growth_r=gr, risk_adj_p5=r_adj_p5,
                         gates_ok=gates_ok))

pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")

print("\n=== PASSING CONFIGS ===")
for r in rows:
    if r['gates_ok']:
        print(f"  {r['label']:>12} PF={r['pf']:.2f}  p/d={r['ppd']:.1f}  P5={r['p5']:.1f}  "
              f"CumL5={r['cum_l5']:.0f}  growR={r['growth_r']:.2f}  risk_adj_p5={r['risk_adj_p5']:.3f}")
