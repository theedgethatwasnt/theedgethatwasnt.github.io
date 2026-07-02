"""
All-pairs ZR optimization sweep — gate=0, PSAR escape af=0.01.

For each of the 11 pairs tested in Session 028 (real spread data):
  Sweep ZW, TGT, ta over a grid.
  td=1 fixed (shown optimal in all prior experiments).
  gate=0 (spread gate removed — be_floor protects 1-leg exits).
  PSAR escape: af=0.01, step=0.01, max=0.20 (deployed best config).

Grid:
  ZW  ∈ [20, 30, 40, 50, 60, 80, 100] pips
  TGT ∈ [ZW*0.3, ZW*0.5, ZW*0.7]     (relative to ZW)
  ta  ∈ [4, 6, 8, 10]                 (trail activation pips)
  td  = 1.0 (fixed)

Validation gates (same as all prior sessions):
  IS=3/3, OOS=3/3, P5>0, P(+)>95%

Output: CSV with per-pair best validated configs sorted by p/d.
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')

PIP_MAP = {
    "EUR_JPY": 0.01, "GBP_JPY": 0.01, "USD_JPY": 0.01,
    "AUD_JPY": 0.01, "CAD_JPY": 0.01, "CHF_JPY": 0.01, "NZD_JPY": 0.01,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001, "NZD_USD": 0.0001,
}

PAIRS = list(PIP_MAP.keys())

# ── Capital efficiency constants (empirically confirmed 2026-05-07) ────────────
# OANDA marginRate per pair (API confirmed): GBP/JPY crosses = 0.05 (20:1), EUR_USD = 0.02 (50:1)
LEVERAGE = {
    "EUR_JPY": 20, "GBP_JPY": 20, "USD_JPY": 20,
    "AUD_JPY": 20, "CAD_JPY": 20, "CHF_JPY": 20, "NZD_JPY": 20,
    "EUR_USD": 50, "GBP_USD": 20, "AUD_USD": 20, "NZD_USD": 20,
}

# pip value in USD per 1 OANDA unit per 1 pip move
PIP_USD = {
    "EUR_JPY": 0.0000649, "GBP_JPY": 0.0000649, "USD_JPY": 0.0000649,
    "AUD_JPY": 0.0000649, "CAD_JPY": 0.0000649, "CHF_JPY": 0.0000649, "NZD_JPY": 0.0000649,
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001, "NZD_USD": 0.0001,
}

# Margin per 1 OANDA unit in USD = base_currency_price_in_USD / leverage
# Using approximate mid-2026 prices (update periodically)
MARGIN_PER_UNIT = {
    "EUR_JPY": 1.084/20,   # EUR in USD / 20:1
    "GBP_JPY": 1.355/20,   # GBP in USD / 20:1
    "USD_JPY": 1.000/20,   # USD / 20:1
    "AUD_JPY": 0.645/20,
    "CAD_JPY": 0.732/20,
    "CHF_JPY": 1.100/20,
    "NZD_JPY": 0.595/20,
    "EUR_USD": 1.084/50,   # 50:1 leverage!
    "GBP_USD": 1.355/20,
    "AUD_USD": 0.645/20,
    "NZD_USD": 0.595/20,
}

NAV_PER_ACCOUNT = 17.0   # approximate current NAV; update as account grows
OANDA_CLOSEOUT  = 0.50   # OANDA forces closeout at margin/NAV >= 50%


def compute_vol_series(zw, tgt, sp, pf=1.25, ml=10):
    """
    Compute ZR leg volume series analytically for LONG start.
    Zone fixed: uz=0, lz=-zw, ut=tgt, lt=-zw-tgt (pip units).
    Returns list of (vol, dir) per leg.
    """
    lv = [1.0]; ld = [1.0]; lp = [0.0]
    uz = 0.0; lz = -zw; ut = tgt; lt = -zw - tgt
    for _ in range(ml - 1):
        nl = len(lv)
        if nl % 2 == 1:   # odd legs placed → next boundary = lower zone → SHORT
            tgt_price = lt
            net = sum(lv[k]*ld[k]*(tgt_price - lp[k]) for k in range(nl)) - nl*sp
            if net >= 0:
                break
            npu = max(tgt - sp, 1e-8)
            lv.append(max(1.0, math.ceil(-net/npu*pf))); ld.append(-1.0); lp.append(lz)
        else:             # even legs placed → next boundary = upper zone → LONG
            tgt_price = ut
            net = sum(lv[k]*ld[k]*(tgt_price - lp[k]) for k in range(nl)) - nl*sp
            if net >= 0:
                break
            npu = max(tgt - sp, 1e-8)
            lv.append(max(1.0, math.ceil(-net/npu*pf))); ld.append(1.0); lp.append(uz)
    return list(zip(lv, ld))


def capital_metrics(zw, tgt, sp, pair):
    """
    Returns dict with safety and dollar-efficiency metrics.
    max_B_4leg: largest B safe through 4-leg cycle at NAV_PER_ACCOUNT.
    dollar_ppd: ppd × max_B_4leg × pip_usd (per day in USD at safe B).
    min_B_1cent_tgt: min B so TGT exit ≥ $0.01 (smallest meaningful win).
    min_NAV_1cent_tgt: min account balance to safely run min_B_1cent_tgt @ 4-leg.
    """
    mu     = MARGIN_PER_UNIT[pair]
    pip_u  = PIP_USD[pair]
    vs     = compute_vol_series(zw, tgt, sp, PF, ML)
    # Cumulative SHORT on 012 at leg 4 (legs 2, 4 = indices 1, 3)
    cum012 = sum(v for v, d in vs[:4] if d < 0)
    # Cumulative LONG on 011 at leg 3 (legs 1, 3 = indices 0, 2)
    cum011_leg3 = sum(v for v, d in vs[:3] if d > 0)
    # Margin per B at worst leg
    margin_per_B_leg4 = cum012 * mu
    margin_per_B_leg3 = cum011_leg3 * mu
    worst_per_B = max(margin_per_B_leg4, margin_per_B_leg3)
    max_B_4leg  = max(1, int((NAV_PER_ACCOUNT * OANDA_CLOSEOUT) / worst_per_B)) if worst_per_B > 0 else 999
    # Minimum B for ≥ $0.01 on TGT escape exit (1-leg scenario)
    min_B_1cent = math.ceil(0.01 / (tgt * pip_u)) if tgt * pip_u > 0 else 999
    # Minimum NAV for min_B_1cent to be 4-leg safe
    min_NAV_1cent = (min_B_1cent * worst_per_B) / OANDA_CLOSEOUT if worst_per_B > 0 else 0
    return dict(
        vol_series   = [int(v) for v, d in vs],
        cum012_leg4  = int(cum012),
        margin_B_leg4= round(margin_per_B_leg4, 4),
        max_B_4leg   = max_B_4leg,
        min_B_1cent  = min_B_1cent,
        min_NAV_1cent= round(min_NAV_1cent, 2),
    )


PF        = 1.25
TD        = 1.0
ML        = 10
OOS_FRAC  = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 500   # lighter MC for speed over many pairs

# PSAR escape params (deployed best)
AF0   = 0.01
AFST  = 0.01
AFMX  = 0.20

# Parameter grid
ZW_VALUES  = [20, 30, 40, 50, 60, 80, 100]
TGT_FRACS  = [0.3, 0.5, 0.7]   # TGT = ZW * frac
TA_VALUES  = [4, 6, 8, 10]


@njit
def sim_zr_psar(op, hi, lo, cl, sp_arr, pip, pf, ml, zw, tgt, ta, td,
                af0, af_step, af_max):
    """gate=0 ZR with PSAR escape trail. Same as deployed."""
    n = len(cl)
    pnl   = np.zeros(n, dtype=np.float64)
    nlegs = np.zeros(n, dtype=np.int32)
    etype = np.zeros(n, dtype=np.int32)
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
                        if l <= ts: pnl[nc]=(ts-e)/pip-sp;nlegs[nc]=1;etype[nc]=1;nc+=1;ex=True
                    else:
                        be=e-sp*pip; ts=e-(peak-td)*pip
                        if ts > be: ts = be
                        if h >= ts: pnl[nc]=(e-ts)/pip-sp;nlegs[nc]=1;etype[nc]=1;nc+=1;ex=True
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
                            nc2=0.0;tv2=0.0
                            for k in range(nl): nc2+=lv[k]*ld[k]*(c-lp[k])/pip;tv2+=lv[k]
                            pnl[nc]=nc2-tv2*sp;nlegs[nc]=nl;etype[nc]=3;nc+=1;ex=True;break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if (not is_hi) and l <= lz and ll != i:
                    ll = i; net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    net -= tv*sp
                    if net < 0:
                        npu=max(tgt-sp,1e-8); v=max(1.0,math.ceil(-net/npu*pf))
                        if nl >= ml:
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
    return pnl[:nc], nlegs[:nc], etype[:nc], nc


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR_MID/'EUR_JPY_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_s0  = np.full(2000, 2.3)
_o=_df0.open.values[:2000].astype(np.float64);_h=_df0.high.values[:2000].astype(np.float64)
_l=_df0.low.values[:2000].astype(np.float64);_c=_df0.close.values[:2000].astype(np.float64)
sim_zr_psar(_o,_h,_l,_c,_s0, 0.01, PF, ML, 50., 25., 6., 1., AF0, AFST, AFMX)
print("done.\n")

all_rows = []

for pair in PAIRS:
    pip = PIP_MAP[pair]
    mid_f = DATA_DIR_MID / f'{pair}_M5.parquet'
    ba_f  = DATA_DIR_BA  / f'{pair}_M5_BA.parquet'
    if not mid_f.exists() or not ba_f.exists():
        print(f"[{pair}] Missing data — skip")
        continue

    mid = pd.read_parquet(mid_f).sort_values('timestamp').reset_index(drop=True)
    ba  = pd.read_parquet(ba_f).sort_values('timestamp').reset_index(drop=True)
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
    sp = ((df.ask_c - df.bid_c) / pip).clip(lower=0.1).values.astype(np.float64)

    med_sp = float(np.median(sp[:is_end]))
    atr20  = float(pd.Series(hi[:is_end] - lo[:is_end]).rolling(20).mean().iloc[-1] / pip)

    print(f"\n{'='*70}")
    print(f"[{pair}]  bars={nb}  OOS={oos_days:.1f}d  med_sp={med_sp:.2f}p  ATR20={atr20:.1f}p")
    print(f"{'='*70}")
    print(f"  {'ZW':>5} {'TGT':>5} {'ta':>4} | {'p/d':>8} {'IS':>2} {'OS':>2} {'P5':>8} {'P+':>6} {'cycles':>7} | status")
    print(f"  {'-'*70}")

    rng = np.random.default_rng(42)

    def run_pair(s, e2, zw, tgt, ta):
        return sim_zr_psar(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                           sp[s:e2], pip, PF, ML, zw, tgt, ta, TD,
                           AF0, AFST, AFMX)

    best_ppd = -1e9
    for zw in ZW_VALUES:
        for tgt_frac in TGT_FRACS:
            tgt = round(zw * tgt_frac, 1)
            if tgt < 5.0: continue          # minimum meaningful target
            if tgt >= zw: continue          # target must be less than ZW
            for ta in TA_VALUES:
                if ta >= tgt: continue      # trail activation must be < target
                cyc, legs, et, nc = run_pair(is_end, nb, float(zw), float(tgt), float(ta))
                if nc == 0: continue

                ppd = cyc.sum() / oos_days

                is_wf = 0
                for ch in range(IS_CHUNKS):
                    s_ = ch * is_csz
                    e_ = (ch+1)*is_csz if ch < IS_CHUNKS-1 else is_end
                    c2, _, _, nc2 = run_pair(s_, e_, float(zw), float(tgt), float(ta))
                    if nc2 > 0 and c2.sum() > 0: is_wf += 1

                oos_wf = 0
                for ch in range(OOS_CHUNKS):
                    s_ = is_end + ch * oos_csz
                    e_ = is_end + (ch+1)*oos_csz if ch < OOS_CHUNKS-1 else nb
                    c2, _, _, nc2 = run_pair(s_, e_, float(zw), float(tgt), float(ta))
                    if nc2 > 0 and c2.sum() > 0: oos_wf += 1

                p5 = prob = float('nan')
                if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS:
                    boot = np.array([rng.choice(cyc, nc, replace=True).sum() / oos_days
                                     for _ in range(N_BOOT)])
                    p5   = float(np.percentile(boot, 5))
                    prob = float(np.mean(boot > 0))

                passed = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
                          and not math.isnan(p5) and p5 > 0 and prob > 0.95)
                note = "🟢 PASS" if passed else ("🟡" if (ppd > 0 and is_wf >= 2 and oos_wf >= 2) else "🔴")

                if passed and ppd > best_ppd:
                    best_ppd = ppd

                print(f"  {zw:>5.0f} {tgt:>5.1f} {ta:>4.0f} | {ppd:>8.1f} {is_wf:>2} {oos_wf:>2} "
                      f"{p5:>8.1f} {prob:>6.3f} {nc:>7} | {note}")
                sys.stdout.flush()

                cm = capital_metrics(float(zw), float(tgt), med_sp, pair)
                pip_u = PIP_USD[pair]
                dollar_ppd = round(ppd * cm['max_B_4leg'] * pip_u, 4)

                all_rows.append(dict(
                    pair=pair, zw=zw, tgt=tgt, ta=ta, td=TD,
                    ppd=round(ppd,1), is_wf=is_wf, oos_wf=oos_wf,
                    p5=round(p5,1) if not math.isnan(p5) else None,
                    prob=round(prob,3) if not math.isnan(prob) else None,
                    nc=nc, oos_days=round(oos_days,1),
                    med_sp=round(med_sp,2), atr20=round(atr20,1),
                    passed=passed,
                    # Capital efficiency (Session 035)
                    leverage=LEVERAGE[pair],
                    pip_usd=pip_u,
                    max_B_4leg=cm['max_B_4leg'],
                    cum012_leg4=cm['cum012_leg4'],
                    margin_B_leg4=cm['margin_B_leg4'],
                    dollar_ppd=dollar_ppd,
                    min_B_1cent=cm['min_B_1cent'],
                    min_NAV_1cent=cm['min_NAV_1cent'],
                ))

out = Path(__file__).parent / 'zr_allpairs_opt_results.csv'
pd.DataFrame(all_rows).to_csv(out, index=False)
print(f"\n\nSaved {len(all_rows)} rows -> {out}")

# ── Summary: best validated config per pair ──────────────────────────────────
print("\n" + "="*110)
print("BEST VALIDATED CONFIG PER PAIR — sorted by dollar p/d (pip_p/d × max_B × pip_usd)")
print("gate=0, PSAR af=0.01, IS=3/3 OOS=3/3 P5>0 P(+)>95%")
print("="*110)
print(f"  {'Pair':>10} {'ZW':>5} {'TGT':>5} {'ta':>4} {'lev':>5} | "
      f"{'p/d':>8} {'P5':>8} | {'B_4leg':>6} {'$/day':>7} | {'B_1¢':>5} {'minNAV_1¢':>10}")
print(f"  {'-'*100}")

df_res = pd.DataFrame(all_rows)
best = (df_res[df_res['passed']]
        .sort_values('dollar_ppd', ascending=False)
        .groupby('pair').first()
        .reset_index())

for _, r in best.sort_values('dollar_ppd', ascending=False).iterrows():
    print(f"  {r['pair']:>10} {r['zw']:>5.0f} {r['tgt']:>5.1f} {r['ta']:>4.0f} {r['leverage']:>4}:1 | "
          f"{r['ppd']:>8.1f} {r['p5']:>8.1f} | "
          f"{r['max_B_4leg']:>6} {r['dollar_ppd']:>7.3f} | "
          f"{r['min_B_1cent']:>5} {r['min_NAV_1cent']:>10.2f}")

print()
print("Columns: B_4leg=max safe base_units at 4-leg (OANDA won't force-close)")
print("         $/day=dollar p/d at B_4leg  |  B_1¢=min units for TGT exit≥$0.01")
print("         minNAV_1¢=min account balance to run B_1¢ safely at 4-leg")
