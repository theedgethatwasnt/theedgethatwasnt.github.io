"""
Cross-pair ZR sweep: GBP_USD + EUR_USD (+ EUR_JPY as reference).

Two parallel ZW candidate sets per pair:
  A) ATR-based: [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0] × ATR20 (rounded to 5p, min 5p)
  B) Hard-coded pips: 5, 10, 15, 20, ..., 300 (step 5)
  Combined (union, deduped, sorted).

TGT/ZW ratio ∈ [0.3, 0.5, 0.75, 1.0, 1.5, 2.0], ta=6, td=1, ml=10
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_PATH     = Path(__file__).parent / 'backtest_zr_pair_sweep_results.csv'

PF         = 1.25
OOS_FRAC   = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 1000
MAX_LEGS   = 10
TA         = 6.0
TD         = 1.0

ATR_FRACS      = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
HARDCODED_ZW   = list(range(5, 305, 5))   # 5,10,...,300
TGT_FRACS      = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

PAIRS = [
    ("GBP_USD", 0.0001),
    ("EUR_USD", 0.0001),
    ("EUR_JPY", 0.01),
]


@njit
def sim_zr(op, hi, lo, cl, sp_arr, pip, pf, ml, zw, tgt, ta, td, gate):
    """Fixed ZR sim — zones first, both targets active, (tgt-sp) denom, be_floor trail."""
    n = len(cl)
    pnl   = np.zeros(n, dtype=np.float64)
    nlegs = np.zeros(n, dtype=np.int32)
    etype = np.zeros(n, dtype=np.int32)   # 1=trail 2=escape 3=maxlegs
    nc = 0; n_skip = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    i = 0; d = 1
    while i < n:
        sp_e = sp_arr[i]
        if gate > 0 and sp_e > gate:
            n_skip += 1; i += 1; continue
        e = cl[i]
        if d == 1:
            uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:
            lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak=0.0; ton=False
        i += 1
        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; sp=sp_arr[i]; bull=(c>=op[i])
            # ── 1-leg trail ──────────────────────────────────────────────────
            if nl == 1:
                mfe = (h-e)/pip if d==1 else (e-l)/pip
                if mfe > peak: peak = mfe
                if peak >= ta: ton = True
                if ton:
                    if d == 1:
                        be = e + sp*pip
                        ts = e + (peak-td)*pip
                        if ts < be: ts = be
                        if l <= ts:
                            pnl[nc]=(ts-e)/pip-sp; nlegs[nc]=1; etype[nc]=1
                            nc+=1; ex=True
                    else:
                        be = e - sp*pip
                        ts = e - (peak-td)*pip
                        if ts > be: ts = be
                        if h >= ts:
                            pnl[nc]=(e-ts)/pip-sp; nlegs[nc]=1; etype[nc]=1
                            nc+=1; ex=True
            if ex: break
            # ── Zones first, then targets ─────────────────────────────────────
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
    return pnl[:nc], nlegs[:nc], etype[:nc], nc, n_skip


# ── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR_MID/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_s0  = np.full(2000, 1.0)
_o=_df0.open.values[:2000].astype(np.float64)
_h=_df0.high.values[:2000].astype(np.float64)
_l=_df0.low.values[:2000].astype(np.float64)
_c=_df0.close.values[:2000].astype(np.float64)
sim_zr(_o,_h,_l,_c,_s0,0.0001,PF,MAX_LEGS,0.003,0.0015,5.,1.,0.)
print("done.\n")

rows = []

for pair, pip in PAIRS:
    mid = pd.read_parquet(DATA_DIR_MID/f'{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
    ba  = pd.read_parquet(DATA_DIR_BA /f'{pair}_M5_BA.parquet').sort_values('timestamp').reset_index(drop=True)
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

    hl_pip = (df.high - df.low).iloc[:is_end] / pip
    atr20  = hl_pip.rolling(20).mean().median()
    sp_med = float(np.median(sp[:is_end]))
    gate   = float(np.percentile(sp[:is_end], 90))

    print(f"{'='*100}")
    print(f"  {pair}  pip={pip}  ATR20={atr20:.1f}p  spread: med={sp_med:.2f}p  gate={gate:.2f}p")
    print(f"  IS bars: {is_end}  OOS days: {oos_days:.1f}  spread/ATR = {sp_med/atr20:.3f}")
    print(f"{'='*100}")

    # ZW candidates: ATR-based + hardcoded, union
    atr_based = {max(5.0, round(atr20 * f / 5) * 5.0) for f in ATR_FRACS}
    zw_cands  = sorted(atr_based | set(float(z) for z in HARDCODED_ZW))
    atr_set   = atr_based  # for annotation

    print(f"  ATR-based ZW: {sorted(atr_based)}  |  hardcoded: 5..300 step 5")
    print(f"  Total ZW candidates: {len(zw_cands)}")
    print()

    hdr = (f"  {'ZW':>5} {'ZW/ATR':>6} {'src':>4} {'TGT':>5} {'T/ZW':>5} | "
           f"{'p/d':>10} | {'IS':>2} {'OS':>2} | {'P5':>7} {'P+':>6} | "
           f"{'1L%':>5} {'2L%':>5} {'3L%':>5} {'4L%':>5} {'5+%':>5} | "
           f"{'tr%':>5} {'esc%':>5} {'ml%':>4} | status")
    sep = "─" * 115
    print(sep); print(hdr); print(sep)

    rng = np.random.default_rng(42)

    def run(s, e2):
        return sim_zr(op[s:e2], hi[s:e2], lo[s:e2], cl[s:e2],
                      sp[s:e2], pip, PF, MAX_LEGS, zw, tgt, TA, TD, gate)

    for zw in zw_cands:
        src = "ATR" if zw in atr_set else "pip"
        first_row_for_zw = True

        for tf in TGT_FRACS:
            tgt = round(zw * tf / 5) * 5.0
            if tgt < 5.0: tgt = 5.0
            if tgt <= gate: continue

            cyc, legs, et, nc, _ = run(is_end, nb)
            if nc == 0: continue

            ppd  = cyc.sum() / oos_days

            l1 = np.mean(legs==1)*100; l2 = np.mean(legs==2)*100
            l3 = np.mean(legs==3)*100; l4 = np.mean(legs==4)*100
            l5 = np.mean(legs>=5)*100
            tr_pct  = np.mean(et==1)*100
            esc_pct = np.mean(et==2)*100
            ml_pct  = np.mean(et==3)*100

            # WF (single call per chunk — no double-call)
            is_wf = 0
            for ch in range(IS_CHUNKS):
                s_ = ch * is_csz
                e_ = (ch+1)*is_csz if ch < IS_CHUNKS-1 else is_end
                c2, _, _, nc2, _ = run(s_, e_)
                if nc2 > 0 and c2.sum() > 0: is_wf += 1

            oos_wf = 0
            for ch in range(OOS_CHUNKS):
                s_ = is_end + ch * oos_csz
                e_ = is_end + (ch+1)*oos_csz if ch < OOS_CHUNKS-1 else nb
                c2, _, _, nc2, _ = run(s_, e_)
                if nc2 > 0 and c2.sum() > 0: oos_wf += 1

            p5 = prob = float('nan')
            if is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS:
                boot = np.array([rng.choice(cyc, nc, replace=True).sum() / oos_days
                                 for _ in range(N_BOOT)])
                p5   = float(np.percentile(boot, 5))
                prob = float(np.mean(boot > 0))

            passed = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
                      and not math.isnan(p5) and p5 > 0 and prob > 0.95)
            if passed:
                status = "🟢 PASS"
            elif ppd > 0 and is_wf >= 2 and oos_wf >= 2:
                status = "🟡 near"
            elif ppd < 0:
                status = "🔴"
            else:
                status = f"{is_wf}/{oos_wf}"

            # Suppress ZW/ATR columns on repeated rows for same ZW
            zw_str    = f"{zw:>5.0f}" if first_row_for_zw else " " * 5
            ratio_str = f"{zw/atr20:>6.2f}" if first_row_for_zw else " " * 6
            src_str   = f"{src:>4}"   if first_row_for_zw else " " * 4
            first_row_for_zw = False

            print(f"  {zw_str} {ratio_str} {src_str} {tgt:>5.0f} {tf:>5.2f} | "
                  f"{ppd:>10.1f} | {is_wf:>2} {oos_wf:>2} | "
                  f"{p5:>7.1f} {prob:>6.3f} | "
                  f"{l1:>5.1f} {l2:>5.1f} {l3:>5.1f} {l4:>5.1f} {l5:>5.2f} | "
                  f"{tr_pct:>5.1f} {esc_pct:>5.1f} {ml_pct:>4.1f} | {status}")
            sys.stdout.flush()

            rows.append(dict(
                pair=pair, zw=zw, zw_atr=round(zw/atr20,2), tgt=tgt, tgt_zw=tf,
                spread_atr=round(sp_med/atr20,3), src=src,
                ppd=round(ppd,1), is_wf=is_wf, oos_wf=oos_wf,
                p5=round(p5,1) if not math.isnan(p5) else None,
                prob=round(prob,3) if not math.isnan(prob) else None,
                l1=round(l1,1), l2=round(l2,1), l3=round(l3,1),
                l4=round(l4,1), l5=round(l5,2),
                tr_pct=round(tr_pct,1), esc_pct=round(esc_pct,1), ml_pct=round(ml_pct,1),
            ))
        if not first_row_for_zw:
            print()   # blank line between ZW groups

pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")

# ── Summary: best validated per pair ─────────────────────────────────────────
print("\n=== BEST VALIDATED CONFIG PER PAIR (IS=3 OOS=3 P5>0 P(+)>95%) ===")
print(f"  {'pair':>8} {'ZW':>5} {'ZW/ATR':>6} {'TGT':>5} {'T/ZW':>5} | "
      f"{'p/d':>10} | {'P5':>7} | {'1L%':>5} {'2L%':>5} {'5+%':>5} | {'sp/ATR':>7}")
best = {}
for r in rows:
    if (r.get('p5') and r['p5'] > 0 and r.get('prob', 0) > 0.95
            and r['is_wf'] == IS_CHUNKS and r['oos_wf'] == OOS_CHUNKS):
        k = r['pair']
        if k not in best or r['ppd'] > best[k]['ppd']:
            best[k] = r
for r in best.values():
    print(f"  {r['pair']:>8} {r['zw']:>5.0f} {r['zw_atr']:>6.2f} {r['tgt']:>5.0f} {r['tgt_zw']:>5.2f} | "
          f"{r['ppd']:>10.1f} | {r['p5']:>7.1f} | "
          f"{r['l1']:>5.1f} {r['l2']:>5.1f} {r['l5']:>5.2f} | {r['spread_atr']:>7.3f}")

# ── Leg-depth at TGT/ZW ≈ 0.5 ─────────────────────────────────────────────
print("\n=== LEG-DEPTH AT TGT/ZW=0.5 (reference: EUR_JPY ZW=50 deployed) ===")
print(f"  {'pair':>8} {'ZW':>5} {'ZW/ATR':>6} | "
      f"{'1L%':>6} {'2L%':>6} {'3L%':>6} {'4L%':>6} {'5+%':>5} | "
      f"{'trail%':>7} {'esc%':>6} {'ml%':>5} | {'p/d':>10}")
ref_zw = {p: None for p, _ in PAIRS}
for r in rows:
    if abs(r['tgt_zw'] - 0.5) < 0.01:
        rk = (r['pair'], r['zw'])
        # collect all, print selected
        pass
for r in sorted(rows, key=lambda x: (x['pair'], x['zw'])):
    if abs(r['tgt_zw'] - 0.5) < 0.01:
        print(f"  {r['pair']:>8} {r['zw']:>5.0f} {r['zw_atr']:>6.2f} | "
              f"{r['l1']:>6.1f} {r['l2']:>6.1f} {r['l3']:>6.1f} {r['l4']:>6.1f} {r['l5']:>5.2f} | "
              f"{r['tr_pct']:>7.1f} {r['esc_pct']:>6.1f} {r['ml_pct']:>5.1f} | {r['ppd']:>10.1f}")
