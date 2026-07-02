"""
PSAR 1-leg trail experiment — EUR_JPY ZW=50 TGT=25 ta=6.

Replaces fixed td=1 trail (90.8% of exits) with PSAR-style accelerating trail.
Escape exits remain flat.

  baseline (td=1 fixed): 543 p/d, P5=215, IS=3/3 OOS=3/3
  PSAR escape only (af0=0.01): 646 p/d, P5=317

PSAR 1-leg: at ta=6p activation, place trail init_dist pips behind ep.
AF accelerates on each new extreme. be_floor = entry±spread always enforced.
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')

PAIR      = "EUR_JPY"
PIP       = 0.01
PF        = 1.25
ZW        = 50.0
TGT       = 25.0
TA        = 6.0
ML        = 10
OOS_FRAC  = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 2000

# (init_dist, af0, af_step, af_max)  — af_max fixed at 0.20 unless noted
CONFIGS = [
    (1.0,  1.0,  0.00, 1.00),   # baseline: AF=1 → psar=ep always ≡ td=1 fixed trail
    (1.0,  0.01, 0.01, 0.20),
    (1.0,  0.02, 0.02, 0.20),
    (1.0,  0.04, 0.04, 0.20),
    (1.0,  0.10, 0.10, 0.20),
    (2.0,  0.01, 0.01, 0.20),
    (2.0,  0.02, 0.02, 0.20),
    (2.0,  0.04, 0.04, 0.20),
    (3.0,  0.01, 0.01, 0.20),
    (3.0,  0.02, 0.02, 0.20),
    (3.0,  0.04, 0.04, 0.20),
    (5.0,  0.01, 0.01, 0.20),
    (5.0,  0.02, 0.02, 0.20),
    (5.0,  0.04, 0.04, 0.20),
    (8.0,  0.01, 0.01, 0.20),
    (8.0,  0.02, 0.02, 0.20),
    (12.0, 0.01, 0.01, 0.20),
    (12.0, 0.02, 0.02, 0.20),
    (20.0, 0.01, 0.01, 0.20),
    (20.0, 0.02, 0.02, 0.20),
]


@njit
def sim_zr_psar1(op, hi, lo, cl, sp_arr, pip, pf, ml, zw, tgt, ta,
                 init_dist, af0, af_step, af_max, gate):
    """
    ZR sim with PSAR 1-leg trail + flat escape exits.
    etype: 1=PSAR_1leg  2=escape_flat  3=maxlegs
    """
    n = len(cl)
    pnl   = np.zeros(n, dtype=np.float64)
    nlegs = np.zeros(n, dtype=np.int32)
    etype = np.zeros(n, dtype=np.int32)
    nc = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1
    while i < n:
        sp_e = sp_arr[i]
        if gate > 0 and sp_e > gate:
            i += 1; continue
        e = cl[i]
        if d == 1:
            uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:
            lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak=0.0
        psar_on=False; psar_val=0.0; ep_val=0.0; af_cur=af0
        i += 1
        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; sp=sp_arr[i]; bull=(c>=op[i])

            # ── 1-leg PSAR trail ─────────────────────────────────────────────
            if nl == 1:
                if not psar_on:
                    mfe = (h-e)/pip if d==1 else (e-l)/pip
                    if mfe > peak: peak = mfe
                    if peak >= ta:
                        af_cur = af0
                        if d == 1:
                            ep_val = e + peak * pip
                            psar_val = ep_val - init_dist * pip
                            be = e + sp * pip
                            if psar_val < be: psar_val = be
                        else:
                            ep_val = e - peak * pip
                            psar_val = ep_val + init_dist * pip
                            be = e - sp * pip
                            if psar_val > be: psar_val = be
                        psar_on = True

                if psar_on:
                    be = e + sp*pip if d==1 else e - sp*pip
                    if d == 1:
                        if h > ep_val:
                            ep_val = h
                            af_cur = min(af_cur + af_step, af_max)
                        psar_val = psar_val + af_cur * (ep_val - psar_val)
                        if psar_val < be: psar_val = be
                        if l <= psar_val:
                            pnl[nc] = (psar_val - e) / pip - sp
                            nlegs[nc] = 1; etype[nc] = 1
                            nc += 1; ex = True
                    else:
                        if l < ep_val:
                            ep_val = l
                            af_cur = min(af_cur + af_step, af_max)
                        psar_val = psar_val + af_cur * (ep_val - psar_val)
                        if psar_val > be: psar_val = be
                        if h >= psar_val:
                            pnl[nc] = (e - psar_val) / pip - sp
                            nlegs[nc] = 1; etype[nc] = 1
                            nc += 1; ex = True
            if ex: break

            # ── Zones first, then flat escape targets ─────────────────────────
            for pass_idx in range(2):
                if ex: break
                is_hi = (bull == (pass_idx == 0))
                if is_hi and h >= uz and lu != i:
                    lu = i
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    net -= tv*sp
                    if net < 0:
                        npu = max(tgt-sp, 1e-8)
                        v = max(1.0, math.ceil(-net/npu*pf))
                        if nl >= ml:
                            nc2=0.0; tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp; nlegs[nc]=nl; etype[nc]=3
                            nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if (not is_hi) and l <= lz and ll != i:
                    ll = i
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    net -= tv*sp
                    if net < 0:
                        npu = max(tgt-sp, 1e-8)
                        v = max(1.0, math.ceil(-net/npu*pf))
                        if nl >= ml:
                            nc2=0.0; tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp; nlegs[nc]=nl; etype[nc]=3
                            nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
                if ex: break
                if l <= ut <= h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    pnl[nc]=net-tv*sp; nlegs[nc]=nl; etype[nc]=2
                    nc+=1; ex=True; break
                if l <= lt <= h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    pnl[nc]=net-tv*sp; nlegs[nc]=nl; etype[nc]=2
                    nc+=1; ex=True; break
            i += 1
        d = -d
    return pnl[:nc], nlegs[:nc], etype[:nc], nc


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR_MID/'EUR_JPY_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_s0  = np.full(2000, 2.3)
_o=_df0.open.values[:2000].astype(np.float64)
_h=_df0.high.values[:2000].astype(np.float64)
_l=_df0.low.values[:2000].astype(np.float64)
_c=_df0.close.values[:2000].astype(np.float64)
sim_zr_psar1(_o,_h,_l,_c,_s0,PIP,PF,ML,ZW,TGT,TA,1.0,0.01,0.01,0.20,0.)
sim_zr_psar1(_o,_h,_l,_c,_s0,PIP,PF,ML,ZW,TGT,TA,5.0,0.02,0.02,0.20,0.)
print("done.\n")

# ── Load full data ────────────────────────────────────────────────────────────
mid = pd.read_parquet(DATA_DIR_MID/f'{PAIR}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
ba  = pd.read_parquet(DATA_DIR_BA /f'{PAIR}_M5_BA.parquet').sort_values('timestamp').reset_index(drop=True)
mid['ts_key'] = mid['timestamp'].astype(str).str[:19]
ba['ts_key']  = ba['timestamp'].astype(str).str[:19]
df = mid.merge(ba[['ts_key','bid_c','ask_c']], on='ts_key', how='inner').sort_values('ts_key').reset_index(drop=True)

nb      = len(df)
is_end  = int(nb * (1 - OOS_FRAC))
is_csz  = is_end // IS_CHUNKS
oos_len = nb - is_end
oos_csz = oos_len // OOS_CHUNKS
oos_days = oos_len / (24 * 12)

op = df.open.values.astype(np.float64)
hi = df.high.values.astype(np.float64)
lo = df.low.values.astype(np.float64)
cl = df.close.values.astype(np.float64)
sp = ((df.ask_c - df.bid_c) / PIP).clip(lower=0.1).values.astype(np.float64)
gate = float(np.percentile(sp[:is_end], 90))

print(f"Pair: {PAIR}  ZW={ZW}p  TGT={TGT}p  ta={TA}")
print(f"gate={gate:.2f}p  OOS days={oos_days:.1f}")
print(f"baseline: td=1 fixed → 543 p/d, P5=215, IS=3/3 OOS=3/3\n")

sep = "─" * 115
hdr = (f"  {'dist':>5} {'af0':>5} {'step':>5} | {'p/d':>8} | {'IS':>2} {'OS':>2} | "
       f"{'P5':>8} {'P+':>6} | {'1leg%':>6} {'esc%':>5} {'ml%':>4} | note")
print(sep); print(hdr); print(sep)

rng = np.random.default_rng(42)

def run(s, e2, cfg):
    init_d, af0, af_st, af_mx = cfg
    return sim_zr_psar1(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                        sp[s:e2], PIP, PF, ML, ZW, TGT, TA,
                        init_d, af0, af_st, af_mx, gate)

for cfg in CONFIGS:
    init_d, af0, af_st, af_mx = cfg
    cyc, legs, et, nc = run(is_end, nb, cfg)
    if nc == 0: continue

    ppd = cyc.sum() / oos_days
    tr1_pct  = np.mean(et==1)*100
    flat_pct = np.mean(et==2)*100
    ml_pct   = np.mean(et==3)*100

    is_wf = 0
    for ch in range(IS_CHUNKS):
        s_ = ch * is_csz
        e_ = (ch+1)*is_csz if ch < IS_CHUNKS-1 else is_end
        c2, _, _, nc2 = run(s_, e_, cfg)
        if nc2 > 0 and c2.sum() > 0: is_wf += 1

    oos_wf = 0
    for ch in range(OOS_CHUNKS):
        s_ = is_end + ch * oos_csz
        e_ = is_end + (ch+1)*oos_csz if ch < OOS_CHUNKS-1 else nb
        c2, _, _, nc2 = run(s_, e_, cfg)
        if nc2 > 0 and c2.sum() > 0: oos_wf += 1

    p5 = prob = float('nan')
    if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS:
        boot = np.array([rng.choice(cyc, nc, replace=True).sum() / oos_days
                         for _ in range(N_BOOT)])
        p5   = float(np.percentile(boot, 5))
        prob = float(np.mean(boot > 0))

    passed = (is_wf==IS_CHUNKS and oos_wf==OOS_CHUNKS
              and not math.isnan(p5) and p5>0 and prob>0.95)
    note = "🟢 PASS" if passed else ("🟡 near" if (ppd>0 and is_wf>=2 and oos_wf>=2) else "🔴")
    tag  = "(baseline≡td=1)" if af0 == 1.0 else ""

    print(f"  {init_d:>5.1f} {af0:>5.2f} {af_st:>5.2f} | {ppd:>8.1f} | {is_wf:>2} {oos_wf:>2} | "
          f"{p5:>8.1f} {prob:>6.3f} | {tr1_pct:>6.1f} {flat_pct:>5.1f} {ml_pct:>4.1f} | "
          f"{note} {tag}")
    sys.stdout.flush()

print(sep)
print()
print("dist=init_dist (pips behind ep at ta activation)")
print("af0=initial AF, step=AF increment per new extreme, af_max=0.20")
print("esc%=flat escape exits (no PSAR), 1leg%=PSAR 1-leg exits")
