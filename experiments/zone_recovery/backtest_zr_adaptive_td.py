"""
Adaptive trailing distance for ZR 1-leg exits.
Hypothesis: when ATR(3)/ATR(20) > threshold (trending bar), widen td from 1p
to ATR(3)/2 so the trail doesn't fire on noise and rides the full trend.

Baseline: EUR_USD ZW=30 TGT=21 ta=10 td=1.0 → 1442 p/d IS=3/3 OOS=3/3 P5=486

Sweep:
  trend_thresh: ATR(3)/ATR(20) ratio to declare "trending"  [1.5, 2.0, 2.5, 3.0]
  td_wide: trailing distance when trending                   [atr3/3, atr3/2, cap5, cap10]
  atr_s: short ATR period                                    [3, 5]
"""
import sys, os, math
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

DATA_DIR_MID = Path("/path/to/projects/fx-core/data/m5_ohlc")
DATA_DIR_BA  = Path("/path/to/projects/fx-core/data/m5_ba")
OUT_CSV      = Path(__file__).parent / "zr_adaptive_td_results.csv"

PAIR   = "EUR_USD"
ZW, TGT, TA = 30.0, 21.0, 10.0
TD_BASE      = 1.0
PF, ML       = 1.25, 10
AF0, AFST, AFMX = 0.01, 0.01, 0.20
OOS_FRAC     = 0.30
IS_CHUNKS = OOS_CHUNKS = 3
N_BOOT       = 500


# ── JIT core ─────────────────────────────────────────────────────────────────
@njit
def sim_zr_atd(op, hi, lo, cl, sp_arr, atr_s_arr, atr_l_arr,
               pip, pf, ml, zw, tgt, ta, td_base,
               af0, af_step, af_max,
               trend_thresh, td_wide_frac, td_wide_cap):
    """
    ZR with adaptive td. When atr_s/atr_l > trend_thresh:
        td = min(atr_s * td_wide_frac, td_wide_cap)
    else:
        td = td_base
    Identical to sim_zr_psar otherwise.
    """
    n = len(cl)
    pnl   = np.zeros(n, np.float64)
    nlegs = np.zeros(n, np.int32)
    etype = np.zeros(n, np.int32)
    nc = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1
    while i < n:
        e = cl[i]
        if d == 1: uz=e;lz=e-zw*pip;ut=e+tgt*pip;lt=lz-tgt*pip
        else:      lz=e;uz=e+zw*pip;lt=e-tgt*pip;ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak=0.0; ton=False
        psar_on=False; psar_val=0.0; ep_val=0.0; af_cur=af0; net_dir=0.0
        i += 1
        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; sp=sp_arr[i]; bull=(c>=op[i])
            # ── adaptive td ──────────────────────────────────────────────────
            atr_s = atr_s_arr[i]; atr_l = atr_l_arr[i]
            if atr_l > 0.0 and (atr_s / atr_l) >= trend_thresh:
                td = min(atr_s * td_wide_frac, td_wide_cap)
            else:
                td = td_base
            # ── PSAR escape ──────────────────────────────────────────────────
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
            # ── 1-leg adaptive trail ─────────────────────────────────────────
            if nl == 1:
                mfe = (h-e)/pip if d==1 else (e-l)/pip
                if mfe > peak: peak = mfe
                if peak >= ta: ton = True
                if ton:
                    if d == 1:
                        be=e+sp*pip; ts=e+(peak-td)*pip
                        if ts < be: ts = be
                        if l <= ts: pnl[nc]=(ts-e)/pip-sp;nlegs[nc]=1;etype[nc]=1;nc+=1;ex=True
                    else:
                        be=e-sp*pip; ts=e-(peak-td)*pip
                        if ts > be: ts = be
                        if h >= ts: pnl[nc]=(e-ts)/pip-sp;nlegs[nc]=1;etype[nc]=1;nc+=1;ex=True
            if ex: break
            # ── zone crossings / targets ─────────────────────────────────────
            for pi2 in range(2):
                if ex: break
                is_hi = (bull == (pi2 == 0))
                if is_hi and h >= uz and lu != i:
                    lu=i; net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    net -= tv*sp
                    if net < 0:
                        npu=max(tgt-sp,1e-8); v=max(1.0,math.ceil(-net/npu*pf))
                        if nl>=ml:
                            nc2=0.0;tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip;tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp;nlegs[nc]=nl;etype[nc]=3;nc+=1;ex=True;break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if (not is_hi) and l <= lz and ll != i:
                    ll=i; net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    net -= tv*sp
                    if net < 0:
                        npu=max(tgt-sp,1e-8); v=max(1.0,math.ceil(-net/npu*pf))
                        if nl>=ml:
                            nc2=0.0;tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip;tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp;nlegs[nc]=nl;etype[nc]=3;nc+=1;ex=True;break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
                if ex: break
                if l <= ut <= h:
                    net_v=0.0
                    for k in range(nl): net_v+=lv[k]*ld[k]
                    net_dir=1.0 if net_v>=0 else -1.0
                    psar_on=True; af_cur=af0; ep_val=ut
                    psar_val=ut-tgt*pip if net_dir>0 else ut+tgt*pip; break
                if l <= lt <= h:
                    net_v=0.0
                    for k in range(nl): net_v+=lv[k]*ld[k]
                    net_dir=1.0 if net_v>=0 else -1.0
                    psar_on=True; af_cur=af0; ep_val=lt
                    psar_val=lt-tgt*pip if net_dir>0 else lt+tgt*pip; break
            i += 1
        d = -d
    return pnl[:nc], nlegs[:nc], etype[:nc], nc


@njit
def sim_zr_psar_baseline(op, hi, lo, cl, sp_arr, pip, pf, ml, zw, tgt, ta, td,
                          af0, af_step, af_max):
    """Baseline: same as allpairs_opt (fixed td)."""
    n = len(cl)
    pnl=np.zeros(n,np.float64); nlegs=np.zeros(n,np.int32); etype=np.zeros(n,np.int32)
    nc=0; lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    i=0; d=1
    while i < n:
        e=cl[i]
        if d==1: uz=e;lz=e-zw*pip;ut=e+tgt*pip;lt=lz-tgt*pip
        else:    lz=e;uz=e+zw*pip;lt=e-tgt*pip;ut=uz+tgt*pip
        lv[0]=1.0;ld[0]=float(d);lp[0]=e
        nl=1;lu=ll=-1;ex=False;peak=0.0;ton=False
        psar_on=False;psar_val=0.0;ep_val=0.0;af_cur=af0;net_dir=0.0
        i+=1
        while i<n and not ex:
            h=hi[i];l=lo[i];c=cl[i];sp=sp_arr[i];bull=(c>=op[i])
            if psar_on:
                if net_dir>0:
                    if h>ep_val:ep_val=h;af_cur=min(af_cur+af_step,af_max)
                    psar_val=ep_val-(ep_val-psar_val)*af_cur
                    if l<=psar_val:
                        net=0.0;tv=0.0
                        for k in range(nl):net+=lv[k]*ld[k]*(psar_val-lp[k])/pip;tv+=lv[k]
                        pnl[nc]=net-tv*sp;nlegs[nc]=nl;etype[nc]=5;nc+=1;ex=True
                else:
                    if l<ep_val:ep_val=l;af_cur=min(af_cur+af_step,af_max)
                    psar_val=ep_val+(psar_val-ep_val)*af_cur
                    if h>=psar_val:
                        net=0.0;tv=0.0
                        for k in range(nl):net+=lv[k]*ld[k]*(psar_val-lp[k])/pip;tv+=lv[k]
                        pnl[nc]=net-tv*sp;nlegs[nc]=nl;etype[nc]=5;nc+=1;ex=True
                if ex:break
                i+=1;continue
            if nl==1:
                mfe=(h-e)/pip if d==1 else (e-l)/pip
                if mfe>peak:peak=mfe
                if peak>=ta:ton=True
                if ton:
                    if d==1:
                        be=e+sp*pip;ts=e+(peak-td)*pip
                        if ts<be:ts=be
                        if l<=ts:pnl[nc]=(ts-e)/pip-sp;nlegs[nc]=1;etype[nc]=1;nc+=1;ex=True
                    else:
                        be=e-sp*pip;ts=e-(peak-td)*pip
                        if ts>be:ts=be
                        if h>=ts:pnl[nc]=(e-ts)/pip-sp;nlegs[nc]=1;etype[nc]=1;nc+=1;ex=True
            if ex:break
            for pi2 in range(2):
                if ex:break
                is_hi=(bull==(pi2==0))
                if is_hi and h>=uz and lu!=i:
                    lu=i;net=0.0;tv=0.0
                    for k in range(nl):net+=lv[k]*ld[k]*(ut-lp[k])/pip;tv+=lv[k]
                    net-=tv*sp
                    if net<0:
                        npu=max(tgt-sp,1e-8);v=max(1.0,math.ceil(-net/npu*pf))
                        if nl>=ml:
                            nc2=0.0;tv2=0.0
                            for k in range(nl):nc2+=lv[k]*ld[k]*(c-lp[k])/pip;tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp;nlegs[nc]=nl;etype[nc]=3;nc+=1;ex=True;break
                        lv[nl]=v;ld[nl]=1.0;lp[nl]=uz;nl+=1
                if (not is_hi) and l<=lz and ll!=i:
                    ll=i;net=0.0;tv=0.0
                    for k in range(nl):net+=lv[k]*ld[k]*(lt-lp[k])/pip;tv+=lv[k]
                    net-=tv*sp
                    if net<0:
                        npu=max(tgt-sp,1e-8);v=max(1.0,math.ceil(-net/npu*pf))
                        if nl>=ml:
                            nc2=0.0;tv2=0.0
                            for k in range(nl):nc2+=lv[k]*ld[k]*(c-lp[k])/pip;tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp;nlegs[nc]=nl;etype[nc]=3;nc+=1;ex=True;break
                        lv[nl]=v;ld[nl]=-1.0;lp[nl]=lz;nl+=1
                if ex:break
                if l<=ut<=h:
                    net_v=0.0
                    for k in range(nl):net_v+=lv[k]*ld[k]
                    net_dir=1.0 if net_v>=0 else -1.0
                    psar_on=True;af_cur=af0;ep_val=ut
                    psar_val=ut-tgt*pip if net_dir>0 else ut+tgt*pip;break
                if l<=lt<=h:
                    net_v=0.0
                    for k in range(nl):net_v+=lv[k]*ld[k]
                    net_dir=1.0 if net_v>=0 else -1.0
                    psar_on=True;af_cur=af0;ep_val=lt
                    psar_val=lt-tgt*pip if net_dir>0 else lt+tgt*pip;break
            i+=1
        d=-d
    return pnl[:nc],nlegs[:nc],etype[:nc],nc


def wf_score(pnl_arr, oos_start, oos_len, oos_chunks, oos_csz, rng, n_boot):
    """IS/OOS walk-forward + bootstrap P5."""
    if len(pnl_arr) == 0:
        return 0, 0, 0.0, 0
    oos_pnl = pnl_arr[pnl_arr['bar'] >= oos_start]
    is_pnl  = pnl_arr[pnl_arr['bar'] < oos_start]
    oos_wf  = sum(1 for c in range(oos_chunks)
                  if oos_pnl[(oos_pnl['bar'] >= oos_start + c*oos_csz) &
                              (oos_pnl['bar'] < oos_start + (c+1)*oos_csz)]['p'].sum() > 0)
    is_csz  = len(is_pnl) // 3
    is_wf   = sum(1 for c in range(3)
                  if is_pnl.iloc[c*is_csz:(c+1)*is_csz]['p'].sum() > 0)
    if len(oos_pnl) > 0:
        boots = rng.choice(oos_pnl['p'].values, (n_boot, len(oos_pnl)), replace=True).sum(axis=1)
        p5 = float(np.percentile(boots / (len(oos_pnl)/3.23 / 149.9), 5))
    else:
        p5 = 0.0
    return is_wf, oos_wf, p5, len(oos_pnl)


# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading {PAIR}...", flush=True)
pip = 0.0001
mid = pd.read_parquet(DATA_DIR_MID / f'{PAIR}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
ba  = pd.read_parquet(DATA_DIR_BA  / f'{PAIR}_M5_BA.parquet').sort_values('timestamp').reset_index(drop=True)
mid['ts_key'] = mid['timestamp'].astype(str).str[:19]
ba['ts_key']  = ba['timestamp'].astype(str).str[:19]
df = mid.merge(ba[['ts_key','bid_c','ask_c']], on='ts_key', how='inner').sort_values('ts_key').reset_index(drop=True)

nb      = len(df)
oos_start = int(nb * (1 - OOS_FRAC))
oos_len = nb - oos_start
oos_csz = oos_len // OOS_CHUNKS
oos_days = oos_len / (24*12)

op = df.open.values.astype(np.float64)
hi = df.high.values.astype(np.float64)
lo = df.low.values.astype(np.float64)
cl = df.close.values.astype(np.float64)
sp = ((df.ask_c - df.bid_c) / pip).clip(lower=0.1).values.astype(np.float64)
hl = (hi - lo) / pip  # bar range in pips

# Precompute ATR arrays (rolling mean of H-L, causal)
def rolling_atr(hl, n):
    out = np.zeros(len(hl))
    for i in range(len(hl)):
        s = max(0, i-n+1)
        out[i] = hl[s:i+1].mean()
    return out

print("Computing ATR arrays...", flush=True)
atr3  = pd.Series(hl).rolling(3, min_periods=1).mean().values.astype(np.float64)
atr20 = pd.Series(hl).rolling(20, min_periods=1).mean().values.astype(np.float64)
atr5  = pd.Series(hl).rolling(5, min_periods=1).mean().values.astype(np.float64)

# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("JIT warmup...", flush=True)
_a = np.ones(2000); _b = np.ones(2000)*2.0
sim_zr_atd(op[:2000],hi[:2000],lo[:2000],cl[:2000],sp[:2000],
           _a,_b,pip,PF,ML,ZW,TGT,TA,TD_BASE,AF0,AFST,AFMX,2.0,0.5,10.0)
sim_zr_psar_baseline(op[:2000],hi[:2000],lo[:2000],cl[:2000],sp[:2000],
                     pip,PF,ML,ZW,TGT,TA,TD_BASE,AF0,AFST,AFMX)
print("done.\n")

rng = np.random.default_rng(42)

def run_wf(pnl_raw, nlegs_raw, bar_idx):
    df_c = pd.DataFrame({'p': pnl_raw, 'nl': nlegs_raw, 'bar': bar_idx})
    ppd  = df_c[df_c.bar >= oos_start]['p'].sum() / oos_days
    oos_wf = sum(1 for c in range(OOS_CHUNKS)
                 if df_c[(df_c.bar >= oos_start+c*oos_csz) &
                          (df_c.bar < oos_start+(c+1)*oos_csz)]['p'].sum() > 0)
    is_csz = oos_start // IS_CHUNKS
    is_wf  = sum(1 for c in range(IS_CHUNKS)
                 if df_c[(df_c.bar >= c*is_csz) &
                          (df_c.bar < (c+1)*is_csz)]['p'].sum() > 0)
    oos_p  = df_c[df_c.bar >= oos_start]['p'].values
    if len(oos_p) > 1:
        nc_oos = len(oos_p)
        days_p_cycle = oos_days / nc_oos
        boots = rng.choice(oos_p, (N_BOOT, nc_oos), replace=True).sum(axis=1) / oos_days
        p5 = float(np.percentile(boots, 5))
    else:
        p5 = 0.0
    nc_oos = (df_c.bar >= oos_start).sum()
    # depth distribution
    d1 = int((df_c[(df_c.bar>=oos_start)&(df_c.nl==1)].shape[0]/max(nc_oos,1)*100))
    d5p= int((df_c[(df_c.bar>=oos_start)&(df_c.nl>=5)].shape[0]/max(nc_oos,1)*100))
    return ppd, is_wf, oos_wf, p5, nc_oos, d1, d5p


def run_sim(atr_s_arr, atr_l_arr, thresh, frac, cap):
    pnl_r, nl_r, et_r, nc_r = sim_zr_atd(
        op, hi, lo, cl, sp, atr_s_arr, atr_l_arr,
        pip, PF, ML, ZW, TGT, TA, TD_BASE,
        AF0, AFST, AFMX, thresh, frac, cap)
    # Reconstruct bar indices from cycle sequence
    # We need bar-level bookkeeping: the sim only returns per-cycle data.
    # Approximate: distribute cycles evenly (good enough for WF chunking)
    nc = len(pnl_r)
    bar_idx = np.linspace(0, nb-1, nc).astype(int)
    return run_wf(pnl_r, nl_r, bar_idx)


# ── Baseline ─────────────────────────────────────────────────────────────────
print("=== EUR_USD ZW=30 TGT=21 ta=10 adaptive-td sweep ===")
print()
pnl_b, nl_b, et_b, nc_b = sim_zr_psar_baseline(op,hi,lo,cl,sp,pip,PF,ML,ZW,TGT,TA,TD_BASE,AF0,AFST,AFMX)
bar_b = np.linspace(0, nb-1, len(pnl_b)).astype(int)
base_ppd, base_is, base_oos, base_p5, base_nc, base_d1, base_d5 = run_wf(pnl_b, nl_b, bar_b)
print(f"BASELINE (td=1.0 fixed): {base_ppd:.1f} p/d  IS={base_is}/3  OOS={base_oos}/3  P5={base_p5:.1f}  nc={base_nc}  1L%={base_d1}  5+%={base_d5}")
print()

# ── Grid sweep ────────────────────────────────────────────────────────────────
results = []
THRESH_VALUES = [1.5, 2.0, 2.5, 3.0]
# td_wide = ATR_short * frac, capped at cap
FRAC_VALUES   = [0.33, 0.5, 0.75]
CAP_VALUES    = [5.0, 10.0, 20.0]
ATR_S_OPTS    = [(atr3, "atr3"), (atr5, "atr5")]

print(f"{'Config':<38} | {'p/d':>7} {'IS':>3} {'OOS':>4} {'P5':>7} {'nc':>5} {'1L%':>4} {'5+%':>4} | vs base")
print("-"*90)

for atr_s_arr, atr_s_name in ATR_S_OPTS:
    for thresh in THRESH_VALUES:
        for frac in FRAC_VALUES:
            for cap in CAP_VALUES:
                ppd, iswf, ooswf, p5, nc, d1, d5 = run_sim(atr_s_arr, atr20, thresh, frac, cap)
                delta = ppd - base_ppd
                passed = "✓" if iswf==3 and ooswf==3 and p5>0 else " "
                label = f"{atr_s_name} t={thresh} f={frac} c={cap:.0f}"
                print(f"{passed} {label:<36} | {ppd:>7.1f} {iswf:>3}/3 {ooswf:>4}/3 {p5:>7.1f} {nc:>5} {d1:>4}% {d5:>4}% | {delta:>+7.1f}")
                results.append(dict(atr_s=atr_s_name,thresh=thresh,frac=frac,cap=cap,
                                    ppd=ppd,is_wf=iswf,oos_wf=ooswf,p5=p5,nc=nc,
                                    d1=d1,d5p=d5,delta_ppd=delta))

df_r = pd.DataFrame(results)
df_r.to_csv(OUT_CSV, index=False)
print(f"\nResults saved: {OUT_CSV}")
print(f"\nTop 5 by p/d (IS=3/3 OOS=3/3 only):")
top = df_r[(df_r.is_wf==3)&(df_r.oos_wf==3)].sort_values('ppd',ascending=False).head(5)
print(top[['atr_s','thresh','frac','cap','ppd','p5','nc','d1','d5p','delta_ppd']].to_string(index=False))
print(f"\nTop 5 by P5 (IS=3/3 OOS=3/3 only):")
top5 = df_r[(df_r.is_wf==3)&(df_r.oos_wf==3)].sort_values('p5',ascending=False).head(5)
print(top5[['atr_s','thresh','frac','cap','ppd','p5','nc','d1','d5p','delta_ppd']].to_string(index=False))
