"""
Extended P&F sweep: box={5,10,15} × rev={1,1.5,2,2.5,3} × dir={alt,col} × 10 pairs.
Fractional reversals supported (rev_n is float64).
IS=70%, OOS=30%, WF=3 chunks.
Output: zr_pnf_rev_sweep_results.csv
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR = Path('/path/to/projects/fx-core/data/m5_ohlc')
OUT_PATH  = Path('/path/to/projects/fx-core/research/experiments/zone_recovery/zr_pnf_rev_sweep_results.csv')

SPREAD = 1.4; MAX_LEGS = 10; PF = 1.25
OOS_FRAC  = 0.30
WF_CHUNKS = 3

PIP_MAP = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "AUD_USD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001,
    "USD_JPY": 0.01,   "EUR_JPY": 0.01,   "GBP_JPY": 0.01,
    "CHF_JPY": 0.01,   "AUD_JPY": 0.01,   "NZD_JPY": 0.01,
    "CAD_JPY": 0.01,
}

PAIR_CFG = {
    "CHF_JPY": dict(zw=40.0, tgt=20.0, ta=5.0,  td=3.0),
    "USD_JPY": dict(zw=40.0, tgt=20.0, ta=10.0, td=5.0),
    "GBP_USD": dict(zw=30.0, tgt=15.0, ta=10.0, td=7.0),
    "EUR_JPY": dict(zw=50.0, tgt=25.0, ta=5.0,  td=3.0),
    "NZD_JPY": dict(zw=40.0, tgt=20.0, ta=5.0,  td=3.0),
    "AUD_JPY": dict(zw=50.0, tgt=25.0, ta=5.0,  td=3.0),
    "NZD_USD": dict(zw=25.0, tgt=12.5, ta=5.0,  td=3.0),
    "AUD_USD": dict(zw=30.0, tgt=15.0, ta=5.0,  td=3.0),
    "EUR_GBP": dict(zw=40.0, tgt=20.0, ta=5.0,  td=3.0),
    "CAD_JPY": dict(zw=50.0, tgt=12.5, ta=5.0,  td=3.0),
}

RANDOM_OOS_PPD = {
    "CHF_JPY": 357.5, "USD_JPY": 296.1, "GBP_USD": 240.1,
    "EUR_JPY": 218.4, "NZD_JPY": 136.6, "AUD_JPY":  50.9,
    "NZD_USD":  37.3, "AUD_USD":  30.3, "EUR_GBP":  27.6,
    "CAD_JPY":  24.1,
}

BOX_PIPS  = [5, 10, 15]
REVERSALS = [1.0, 1.5, 2.0, 2.5, 3.0]   # float reversal multipliers


# ─── P&F reversal builder (float rev_n) ──────────────────────────────────────

@njit
def build_pnf_reversals(hi, lo, box_size, rev_n):
    """
    rev_n is float — reversal threshold = rev_n * box_size.
    Supports fractional reversals (e.g. 1.5 boxes).
    """
    n = len(hi)
    rev_bars = np.zeros(n, dtype=np.int64)
    rev_dirs = np.zeros(n, dtype=np.int8)
    n_rev = 0
    if n < 2:
        return rev_bars[:0], rev_dirs[:0]

    col_dir    = np.int8(1)
    col_extreme = (hi[0] + lo[0]) * 0.5
    rev_thresh  = rev_n * box_size

    for i in range(1, n):
        h = hi[i]; l = lo[i]
        if col_dir == 1:
            if h >= col_extreme + box_size:
                col_extreme += math.floor((h - col_extreme) / box_size) * box_size
            elif l <= col_extreme - rev_thresh:
                col_dir = np.int8(-1)
                col_extreme -= box_size
                rev_bars[n_rev] = i; rev_dirs[n_rev] = np.int8(-1); n_rev += 1
        else:
            if l <= col_extreme - box_size:
                col_extreme -= math.floor((col_extreme - l) / box_size) * box_size
            elif h >= col_extreme + rev_thresh:
                col_dir = np.int8(1)
                col_extreme += box_size
                rev_bars[n_rev] = i; rev_dirs[n_rev] = np.int8(1); n_rev += 1

    return rev_bars[:n_rev], rev_dirs[:n_rev]


# ─── ZR sim with P&F entry clock ─────────────────────────────────────────────

@njit
def sim_zr_pnf(op, hi, lo, cl, rev_bars, rev_dirs, dir_mode,
               pip, spread, pf, ml, zw, tgt, ta, td):
    n = len(cl); n_rev = len(rev_bars)
    total = 0.0; nc = 0
    lv = np.zeros(ml); ld = np.zeros(ml); lp = np.zeros(ml)
    d = np.int8(1); ri = 0

    while ri < n_rev:
        entry_bar = rev_bars[ri]; col_d = rev_dirs[ri]; ri += 1
        if entry_bar >= n: break

        direction = float(d) if dir_mode == 0 else float(col_d)
        e = cl[entry_bar]
        if direction == 1.0:
            uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:
            lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip

        lv[0]=1.0; ld[0]=direction; lp[0]=e
        nl=1; lu=-1; ll=-1; ex=False
        peak_mfe=0.0; trail_on=False
        i = entry_bar + 1

        while i < n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; bull=c>=op[i]

            if nl == 1:
                cur_mfe = (h-e)/pip if direction==1.0 else (e-l)/pip
                if cur_mfe > peak_mfe: peak_mfe = cur_mfe
                if peak_mfe >= ta: trail_on = True
                if trail_on:
                    if direction == 1.0:
                        ts = e + (peak_mfe - td) * pip
                        if l <= ts: total += (ts-e)/pip - spread; nc += 1; ex = True
                    else:
                        ts = e - (peak_mfe - td) * pip
                        if h >= ts: total += (e-ts)/pip - spread; nc += 1; ex = True
            if ex: break

            for pn in range(2):
                if ex: break
                dh = (bull and pn==0) or (not bull and pn==1)
                if l<=ut<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; nc+=1; ex=True; break
                if l<=lt<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; nc+=1; ex=True; break
                if dh and h>=uz and lu!=i:
                    lu=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c>=ut: total+=nt2; nc+=1; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            total+=net-tv2*spread; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if not dh and l<=lz and ll!=i:
                    ll=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c<=lt: total+=nt2; nc+=1; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            total+=net-tv2*spread; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            i += 1

        if dir_mode == 0: d = np.int8(-d)
        while ri < n_rev and rev_bars[ri] <= i: ri += 1

    return total, nc


# ─── JIT warm-up ─────────────────────────────────────────────────────────────

print("Compiling JIT...", end=' ', flush=True)
_df0 = pd.read_parquet(DATA_DIR/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df0.open.values[:2000].astype(np.float64); _h=_df0.high.values[:2000].astype(np.float64)
_l=_df0.low.values[:2000].astype(np.float64);  _c=_df0.close.values[:2000].astype(np.float64)
_rb,_rd = build_pnf_reversals(_h, _l, 0.0005, 1.5)
sim_zr_pnf(_o,_h,_l,_c,_rb,_rd,0,0.0001,SPREAD,PF,MAX_LEGS,20.,10.,5.,3.)
sim_zr_pnf(_o,_h,_l,_c,_rb,_rd,1,0.0001,SPREAD,PF,MAX_LEGS,20.,10.,5.,3.)
print("done.\n")


# ─── Main sweep ──────────────────────────────────────────────────────────────

rows = []
total_cfgs = len(PAIR_CFG) * len(BOX_PIPS) * len(REVERSALS) * 2
done = 0

for pair, cfg in PAIR_CFG.items():
    pip  = PIP_MAP[pair]
    zw=cfg['zw']; tgt=cfg['tgt']; ta=cfg['ta']; td=cfg['td']
    base = RANDOM_OOS_PPD[pair]

    df = pd.read_parquet(DATA_DIR/f'{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
    op=df.open.values.astype(np.float64); hi=df.high.values.astype(np.float64)
    lo=df.low.values.astype(np.float64);  cl=df.close.values.astype(np.float64)
    nb = len(cl)

    is_end   = int(nb*(1-OOS_FRAC))
    is_op=op[:is_end]; is_hi=hi[:is_end]; is_lo=lo[:is_end]; is_cl=cl[:is_end]
    oos_op=op[is_end:]; oos_hi=hi[is_end:]; oos_lo=lo[is_end:]; oos_cl=cl[is_end:]
    oos_bars = len(oos_cl)
    oos_td   = oos_bars / (24*12)
    is_bars  = is_end
    is_td    = is_bars / (24*12)
    is_chunk = is_bars // WF_CHUNKS

    print(f"\n{'='*72}")
    print(f"{pair}  zw={zw} tgt={tgt} | base={base:.1f} p/d")
    print(f"  {'box':>5} {'rev':>5} {'dir':>12} | {'is_ppd':>8} {'oos_ppd':>8} {'wf':>3} {'n_cyc':>6} {'vs_rnd':>9}")

    for box_pips in BOX_PIPS:
        box_size = box_pips * pip

        for rev_n in REVERSALS:
            is_rb,  is_rd  = build_pnf_reversals(is_hi,  is_lo,  box_size, rev_n)
            oos_rb, oos_rd = build_pnf_reversals(oos_hi, oos_lo, box_size, rev_n)

            ch_rb_list = []
            for ch in range(WF_CHUNKS):
                cs = ch * is_chunk
                ce = (ch+1)*is_chunk if ch < WF_CHUNKS-1 else is_bars
                crb, crd = build_pnf_reversals(is_hi[cs:ce], is_lo[cs:ce], box_size, rev_n)
                ch_rb_list.append((cs, ce, crb, crd))

            for dir_mode in [0, 1]:
                dir_name = 'alt' if dir_mode == 0 else 'col'

                is_tot, is_nc = sim_zr_pnf(
                    is_op,is_hi,is_lo,is_cl,is_rb,is_rd,dir_mode,
                    pip,SPREAD,PF,MAX_LEGS,zw,tgt,ta,td)
                is_ppd = is_tot / is_td

                wf = 0
                for cs, ce, crb, crd in ch_rb_list:
                    ct, _ = sim_zr_pnf(
                        is_op[cs:ce],is_hi[cs:ce],is_lo[cs:ce],is_cl[cs:ce],
                        crb,crd,dir_mode,pip,SPREAD,PF,MAX_LEGS,zw,tgt,ta,td)
                    wf += (ct > 0)

                oos_tot, oos_nc = sim_zr_pnf(
                    oos_op,oos_hi,oos_lo,oos_cl,oos_rb,oos_rd,dir_mode,
                    pip,SPREAD,PF,MAX_LEGS,zw,tgt,ta,td)
                oos_ppd = oos_tot / oos_td
                vs_rnd  = oos_ppd - base

                flag = "🟢" if vs_rnd > 0 else ("🟡" if oos_ppd > 0 else "🔴")
                print(f"  {box_pips:>5} {rev_n:>5.1f} {dir_name:>12} | "
                      f"{is_ppd:>8.1f} {oos_ppd:>8.1f} {wf:>3} "
                      f"{oos_nc:>6} {vs_rnd:>+9.1f} {flag}")

                rows.append(dict(
                    pair=pair, box_pips=box_pips, reversal=rev_n,
                    direction=dir_name,
                    is_ppd=round(is_ppd,1), oos_ppd=round(oos_ppd,1),
                    wf=wf, n_cycles=oos_nc,
                    vs_random=round(vs_rnd,1),
                    random_base_ppd=base,
                ))
                done += 1
                if done % 30 == 0:
                    print(f"  [{done}/{total_cfgs} done]", flush=True)

    sys.stdout.flush()

# ─── Save + summary ───────────────────────────────────────────────────────────

df_res = pd.DataFrame(rows)
df_res.to_csv(OUT_PATH, index=False)
print(f"\n\nSaved {len(df_res)} rows → {OUT_PATH}")

for dir_name in ['alt', 'col']:
    sub = df_res[(df_res.direction==dir_name) & (df_res.wf==3)].sort_values('oos_ppd', ascending=False)
    print(f"\n=== TOP 20 — {dir_name.upper()} (WF=3, by oos_ppd) ===")
    print(sub[['pair','box_pips','reversal','is_ppd','oos_ppd','wf','n_cycles','vs_random']].head(20).to_string(index=False))

print("\n=== BEST PER PAIR (WF=3, highest oos_ppd across all configs) ===")
wf3 = df_res[df_res.wf==3].copy()
best = wf3.sort_values('oos_ppd', ascending=False).drop_duplicates('pair')
print(best[['pair','box_pips','reversal','direction','oos_ppd','n_cycles','vs_random']].sort_values('oos_ppd', ascending=False).to_string(index=False))
