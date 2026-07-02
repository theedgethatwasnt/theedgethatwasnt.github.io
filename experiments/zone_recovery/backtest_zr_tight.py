"""
Exp — Tight Progression: TGT/ZW ratio × PF sweep
==================================================
Hypothesis: ZR edge lives primarily in the 1-leg trailing stop (ta/td mechanism),
which fires independent of TGT value. TGT governs only the multi-leg hedge math.
Moving TGT→ZW (ratio→1.0) with lower PF cuts the convex growth rate:

  r = PF × (ZW + TGT) / (TGT - spread)

EUR_USD baseline (TGT=21, PF=1.25, sp=1.4): r=3.25 → Cum_L5=197
Tight target   (TGT=30, PF=1.05, sp=1.4): r=1.84 → Cum_L5=~52

Primary optimization metric: risk_adj_p5 = P5 / Cum_L5
  (P5 pips of floor return per unit of cumulative leg-5 exposure)
  Higher = better edge per unit of capital risk.

Secondary metric: 1/Cum_L5 = efficiency (base leg fraction of total leg-5 exposure)

Grid:
  TGT_FRACS = [0.70, 0.80, 0.90, 1.00]   (TGT = frac × ZW)
  PFS       = [1.25, 1.10, 1.05]
  BODY      = [0.0, 0.5]                  (no filter vs absorption filter)

Pairs: EUR_USD (ZW=30, ta=10, td=1)
       GBP_USD (ZW=30, ta=6,  td=1)
       EUR_JPY (ZW=50, ta=6,  td=1)

Total: 3 × 4 × 3 × 2 = 72 combinations
Gates: IS=3/3, OOS=3/3, P5>0, P(+)>0.95, perm_p<0.05
"""
import math, sys, itertools
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_PATH     = Path(__file__).parent / 'results/zr_tight_results.csv'
OUT_PATH.parent.mkdir(exist_ok=True)

ML       = 10
OOS_FRAC = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 500
N_PERM     = 200

# PSAR escape (same as deployed)
AF0  = 0.01
AFST = 0.01
AFMX = 0.20

# ── Pairs: (pair, pip, ZW, ta, td) ───────────────────────────────────────────
# TGT is swept as TGT_FRAC × ZW
PAIRS = [
    ("EUR_USD", 0.0001, 30.0, 10.0, 1.0),
    ("GBP_USD", 0.0001, 30.0,  6.0, 1.0),
    ("EUR_JPY", 0.01,   50.0,  6.0, 1.0),
]

# ── Parameter grid ────────────────────────────────────────────────────────────
TGT_FRACS    = [0.70, 0.80, 0.90, 1.00]  # TGT/ZW ratio
PFS          = [1.25, 1.10, 1.05]
BODY_THRESHS = [0.0, 0.5]                 # 0=no filter, 0.5=absorption filter

# Total: 3 × 4 × 3 × 2 = 72 combinations


# ── Analytical Cum_L5 computation ────────────────────────────────────────────
def compute_cum_l5(zw: float, tgt: float, pf: float, spread: float) -> float:
    """
    Trace first 5 legs analytically (in pip units) to compute cumulative volume.
    Assumes initial LONG entry at uz=0, with lz=-zw, ut=+tgt, lt=-zw-tgt.
    Returns total volume across all legs at the moment leg 5 would open.
    Returns inf if net_per_unit ≤ 0 (TGT ≤ spread → unprofitable per unit).
    """
    uz = 0.0
    lz = -zw
    ut = +tgt
    lt = -zw - tgt
    net_per_unit = tgt - spread
    if net_per_unit <= 0:
        return float('inf')

    legs = [(1.0, +1.0, uz)]  # (vol, dir, price_in_pips)

    for _ in range(4):  # try to add legs 2-5
        # Odd leg count → next price event is hitting lz → target is lt
        # Even leg count → next price event is hitting uz → target is ut
        if len(legs) % 2 == 1:
            target = lt
        else:
            target = ut

        net = sum(v * d * (target - p) for v, d, p in legs)
        net -= sum(v for v, d, p in legs) * spread

        if net >= 0:
            break  # no new leg needed

        vol_new = max(1.0, math.ceil(-net / net_per_unit * pf))

        if len(legs) % 2 == 1:
            legs.append((vol_new, -1.0, lz))  # SHORT at lz
        else:
            legs.append((vol_new, +1.0, uz))  # LONG at uz

    return sum(v for v, d, p in legs)


def growth_rate(zw: float, tgt: float, pf: float, spread: float) -> float:
    """Asymptotic geometric growth factor r = PF*(ZW+TGT)/(TGT-spread)."""
    net_per = tgt - spread
    if net_per <= 0:
        return float('inf')
    return pf * (zw + tgt) / net_per


# ── Numba simulation (identical to backtest_zr_absorption) ───────────────────
@njit
def _check_body(op, hi, lo, cl, i, d, body_thresh):
    """Body filter only (wick/mom disabled — tight experiment focuses on body)."""
    rng = hi[i] - lo[i]
    eps = 1e-10
    if body_thresh > 0.0 and rng > eps:
        if d == 1:
            adverse_body = max(0.0, op[i] - cl[i]) / rng
        else:
            adverse_body = max(0.0, cl[i] - op[i]) / rng
        if adverse_body > body_thresh:
            return False
    return True


@njit
def sim_zr_tight(op, hi, lo, cl, sp_arr, pip, pf, ml, zw, tgt, ta, td,
                 af0, af_step, af_max, body_thresh):
    """
    ZR with PSAR escape + optional body absorption filter.
    Wick/mom filters excluded — tight experiment isolates TGT/PF effect.
    Returns: (pnl array, nlegs array, etype array, nc, n_skipped)
    """
    n     = len(cl)
    pnl   = np.zeros(n, dtype=np.float64)
    nlegs = np.zeros(n, dtype=np.int32)
    etype = np.zeros(n, dtype=np.int32)
    nc     = 0
    n_skip = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1

    while i < n:
        # ── Body absorption gate ───────────────────────────────────────────
        if not _check_body(op, hi, lo, cl, i, d, body_thresh):
            n_skip += 1
            i += 1
            continue

        # ── Entry ─────────────────────────────────────────────────────────
        e = cl[i]
        if d == 1: uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:      lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak=0.0; ton=False
        psar_on=False; psar_val=0.0; ep_val=0.0; af_cur=af0; net_dir=0.0
        i += 1

        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; sp=sp_arr[i]; bull=(c>=op[i])

            # ── PSAR escape ────────────────────────────────────────────────
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

            # ── 1-leg trailing stop ────────────────────────────────────────
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

            # ── Zone crossings then escape targets ─────────────────────────
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


# ── JIT warm-up ──────────────────────────────────────────────────────────────
_df0 = pd.read_parquet(DATA_DIR_MID/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df0['open'].values[:2000].astype(np.float64)
_h=_df0['high'].values[:2000].astype(np.float64)
_l=_df0['low'].values[:2000].astype(np.float64)
_c=_df0['close'].values[:2000].astype(np.float64)
_s=np.full(2000, 0.00016, dtype=np.float64)
sim_zr_tight(_o,_h,_l,_c,_s, 0.0001,1.25,10,30.,21.,10.,1., 0.01,0.01,0.20, 0.0)
print("JIT compiled.", flush=True)


# ── WF + bootstrap helpers ────────────────────────────────────────────────────
def permutation_p(pnl: np.ndarray, n_perm: int) -> float:
    if len(pnl) == 0: return 1.0
    obs = pnl.sum()
    cnt = sum(1 for _ in range(n_perm)
              if (pnl * np.random.choice([-1,1], size=len(pnl))).sum() >= obs)
    return cnt / n_perm


def run_wf_boot(pnl, nc, n_days, n_chunks, n_boot):
    chunk_size = nc // n_chunks
    if chunk_size == 0:
        return 0, 0.0, 0.0, 0.0
    wf_pass = sum(1 for ch in range(n_chunks)
                  if pnl[ch*chunk_size:(ch+1)*chunk_size if ch < n_chunks-1 else nc].sum() > 0)
    ppd = pnl.sum() / max(n_days, 1)
    sums = np.array([np.random.choice(pnl, size=len(pnl), replace=True).sum()
                     for _ in range(n_boot)])
    p5    = float(np.percentile(sums / max(n_days, 1), 5))
    p_pos = float((sums > 0).mean())
    return wf_pass, ppd, p5, p_pos


# ── Main sweep ────────────────────────────────────────────────────────────────
rows = []
total = len(PAIRS) * len(TGT_FRACS) * len(PFS) * len(BODY_THRESHS)
done  = 0

for pair, pip, zw, ta, td in PAIRS:
    mid = pd.read_parquet(DATA_DIR_MID/f'{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
    ba  = pd.read_parquet(DATA_DIR_BA /f'{pair}_M5_BA.parquet').sort_values('timestamp').reset_index(drop=True)
    mid['ts_key'] = mid['timestamp'].astype(str).str[:19]
    ba['ts_key']  = ba['timestamp'].astype(str).str[:19]
    merged = mid.merge(ba[['ts_key','bid_c','ask_c']], on='ts_key', how='left')
    merged['spread'] = ((merged['ask_c'] - merged['bid_c']) / pip).clip(0, 50)
    merged['spread'] = merged['spread'].fillna(merged['spread'].median())
    spread_med = merged['spread'].median()

    op = merged['open'].values.astype(np.float64)
    hi = merged['high'].values.astype(np.float64)
    lo = merged['low'].values.astype(np.float64)
    cl = merged['close'].values.astype(np.float64)
    sp = merged['spread'].values.astype(np.float64)

    n_total  = len(cl)
    n_oos    = int(n_total * OOS_FRAC)
    n_is     = n_total - n_oos
    oos_days = n_oos * 5 / (60 * 24)

    print(f"\n{'='*80}")
    print(f"PAIR: {pair}  ZW={zw}p  ta={ta}  td={td}  sp_med={spread_med:.2f}p  OOS={oos_days:.0f}d")
    print(f"{'tgt_f':>5} {'tgt':>5} {'pf':>5} {'body':>5} | "
          f"{'p/d':>8} {'P5':>8} {'nc':>6} {'1L%':>5} {'5+%':>5} "
          f"{'CL5':>5} {'r':>5} {'radp5':>7} "
          f"{'IS':>4} {'OOS':>4} {'perm_p':>7} | status")
    print('-'*95)

    for tgt_frac, pf, body_thresh in itertools.product(TGT_FRACS, PFS, BODY_THRESHS):
        done += 1
        tgt = round(tgt_frac * zw, 2)
        tag = f"f{tgt_frac:.2f}_pf{pf:.2f}_b{body_thresh:.1f}"

        # Analytical metrics
        cum_l5 = compute_cum_l5(zw, tgt, pf, spread_med)
        r      = growth_rate(zw, tgt, pf, spread_med)

        # IS walk-forward
        is_wf = 0
        is_chunk = n_is // IS_CHUNKS
        for ch in range(IS_CHUNKS):
            s = ch * is_chunk
            e = s + is_chunk if ch < IS_CHUNKS - 1 else n_is
            p, _, _, nc_ch, _ = sim_zr_tight(
                op[s:e], hi[s:e], lo[s:e], cl[s:e], sp[s:e],
                pip, pf, ML, zw, tgt, ta, td, AF0, AFST, AFMX, body_thresh)
            if nc_ch > 0 and p.sum() > 0: is_wf += 1

        # OOS walk-forward
        oos_wf = 0
        oos_chunk = n_oos // OOS_CHUNKS
        for ch in range(OOS_CHUNKS):
            s = n_is + ch * oos_chunk
            e = n_is + (s - n_is + oos_chunk) if ch < OOS_CHUNKS - 1 else n_total
            p, _, _, nc_ch, _ = sim_zr_tight(
                op[s:e], hi[s:e], lo[s:e], cl[s:e], sp[s:e],
                pip, pf, ML, zw, tgt, ta, td, AF0, AFST, AFMX, body_thresh)
            if nc_ch > 0 and p.sum() > 0: oos_wf += 1

        # Full OOS metrics
        pnl_full, nlegs_full, _, nc_full, n_skip = sim_zr_tight(
            op[n_is:], hi[n_is:], lo[n_is:], cl[n_is:], sp[n_is:],
            pip, pf, ML, zw, tgt, ta, td, AF0, AFST, AFMX, body_thresh)

        skip_pct = 100.0 * n_skip / max(n_skip + nc_full, 1)
        l1_pct   = 100.0 * (nlegs_full == 1).sum() / max(nc_full, 1)
        l5p_pct  = 100.0 * (nlegs_full >= 5).sum() / max(nc_full, 1)
        ppd      = pnl_full.sum() / max(oos_days, 1) if nc_full > 0 else 0.0

        _, ppd2, p5, p_pos = run_wf_boot(pnl_full, nc_full, oos_days, OOS_CHUNKS, N_BOOT)
        perm_p = permutation_p(pnl_full, N_PERM)

        # Risk-adjusted P5
        risk_adj_p5 = p5 / cum_l5 if cum_l5 > 0 and cum_l5 < 1e6 else 0.0

        gate_pass = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
                     and p5 > 0 and p_pos > 0.95 and perm_p < 0.05)
        status = "🟢 PASS" if gate_pass else f"{is_wf}/{oos_wf} p={perm_p:.2f}"

        cum_l5_disp = f"{int(cum_l5)}" if cum_l5 < 1e6 else "inf"
        print(f"{tgt_frac:>5.2f} {tgt:>5.1f} {pf:>5.2f} {body_thresh:>5.1f} | "
              f"{ppd:>8.1f} {p5:>8.1f} {nc_full:>6d} "
              f"{l1_pct:>4.1f}% {l5p_pct:>4.1f}% "
              f"{cum_l5_disp:>5} {r:>5.2f} {risk_adj_p5:>7.3f} "
              f"{is_wf:>4}/{IS_CHUNKS} {oos_wf:>4}/{OOS_CHUNKS} {perm_p:>7.3f} | {status}")
        sys.stdout.flush()

        rows.append(dict(
            pair=pair, zw=zw, tgt_frac=tgt_frac, tgt=tgt, pf=pf,
            body_thresh=body_thresh, ta=ta, td=td,
            ppd=round(ppd, 1), p5=round(p5, 1), p_pos=round(p_pos, 3),
            nc=nc_full, skip_pct=round(skip_pct, 1),
            l1_pct=round(l1_pct, 1), l5p_pct=round(l5p_pct, 1),
            cum_l5=int(cum_l5) if cum_l5 < 1e6 else -1,
            growth_r=round(r, 3),
            risk_adj_p5=round(risk_adj_p5, 4),
            is_wf=is_wf, oos_wf=oos_wf, perm_p=round(perm_p, 3),
            gate_pass=gate_pass,
        ))

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
print(f"\n\nResults saved → {OUT_PATH}")

# ── Summary: gate-passing configs sorted by risk_adj_p5 ───────────────────────
print("\n=== ALL PASSING CONFIGS — sorted by risk_adj_p5 desc ===")
pass_df = df[df.gate_pass].sort_values('risk_adj_p5', ascending=False)
if pass_df.empty:
    print("NO passing configs.")
else:
    print(pass_df[['pair','tgt_frac','tgt','pf','body_thresh',
                   'ppd','p5','l5p_pct','cum_l5','growth_r','risk_adj_p5',
                   'nc','perm_p']].to_string(index=False))

# ── Per-pair summary ──────────────────────────────────────────────────────────
print("\n=== TOP 5 PER PAIR (gate pass, by risk_adj_p5) ===")
for pair, *_ in PAIRS:
    sub = df[(df.pair == pair) & df.gate_pass].sort_values('risk_adj_p5', ascending=False)
    if sub.empty:
        print(f"\n{pair}: NO passing configs")
        continue
    print(f"\n{pair} — {len(sub)} passing configs (top 5):")
    print(sub[['tgt_frac','pf','body_thresh','ppd','p5','l5p_pct',
               'cum_l5','growth_r','risk_adj_p5']].head(5).to_string(index=False))

# ── Key insight: does 1-leg% stay stable across TGT configs? ─────────────────
print("\n=== 1-LEG % BY TGT_FRAC (EUR_USD, body=0, pf=1.25) ===")
sub = df[(df.pair == 'EUR_USD') & (df.body_thresh == 0.0) & (df.pf == 1.25)].sort_values('tgt_frac')
if not sub.empty:
    print(sub[['tgt_frac','tgt','ppd','p5','l1_pct','l5p_pct','cum_l5','growth_r','risk_adj_p5']].to_string(index=False))

print("\nBaseline reference:")
print("  EUR_USD ZW=30 TGT=21(0.70) PF=1.25 body=0: p/d=1442 P5=688 nc=484 5+%=3.5 CumL5=197")
print("  GBP_USD ZW=30 TGT=21(0.70) PF=1.25 body=0.5: p/d=7319 P5=1739 nc=~200 5+%=2.0 CumL5=197")
