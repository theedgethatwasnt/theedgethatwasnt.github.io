"""
Permutation test + Bootstrap MC on all high-frequency P&F configs not yet tested.
Candidates: wf=3, oos_ppd>50, c/day>4, not already in permtest results.
Appends to zr_pnf_permtest_results.csv.
"""
import math, sys
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR = Path('/path/to/projects/fx-core/data/m5_ohlc')
PERM_CSV = Path(__file__).parent / 'zr_pnf_permtest_results.csv'
SWEEP_CSV= Path(__file__).parent / 'zr_pnf_rev_sweep_results.csv'

SPREAD   = 1.4
MAX_LEGS = 10
PF       = 1.25
OOS_FRAC = 0.30
N_PERM   = 2000
N_BOOT   = 2000

PIP_MAP = {
    "CHF_JPY":0.01,"NZD_JPY":0.01,"AUD_JPY":0.01,"EUR_JPY":0.01,
    "USD_JPY":0.01,"CAD_JPY":0.01,"GBP_USD":0.0001,"NZD_USD":0.0001,
    "AUD_USD":0.0001,"EUR_GBP":0.0001,
}
PAIR_CFG = {
    "CHF_JPY": dict(zw=40.0, tgt=20.0, ta=5.0, td=1.0),
    "NZD_JPY": dict(zw=40.0, tgt=20.0, ta=5.0, td=3.0),
    "AUD_JPY": dict(zw=50.0, tgt=25.0, ta=5.0, td=3.0),
    "EUR_JPY": dict(zw=50.0, tgt=25.0, ta=5.0, td=3.0),
    "USD_JPY": dict(zw=40.0, tgt=20.0, ta=10.0, td=5.0),
    "CAD_JPY": dict(zw=50.0, tgt=12.5, ta=5.0, td=3.0),
    "GBP_USD": dict(zw=30.0, tgt=15.0, ta=10.0, td=7.0),
    "NZD_USD": dict(zw=25.0, tgt=12.5, ta=5.0, td=3.0),
    "AUD_USD": dict(zw=30.0, tgt=15.0, ta=5.0, td=3.0),
    "EUR_GBP": dict(zw=40.0, tgt=20.0, ta=5.0, td=3.0),
}

# ── Build candidate list from sweep CSV, excluding already-permtested ─────────
done = set()
if PERM_CSV.exists():
    for _, r in pd.read_csv(PERM_CSV).iterrows():
        done.add((r['pair'], str(r['box_pips']), str(r['reversal']), r['direction']))

sweep = pd.read_csv(SWEEP_CSV)
sweep['cday'] = sweep['n_cycles'] / 408
candidates = sweep[
    (sweep['wf'] == 3) &
    (sweep['oos_ppd'] > 50) &
    (sweep['cday'] > 4)
].copy()

# Exclude already done
mask = candidates.apply(
    lambda r: (r['pair'], str(int(r['box_pips'])), str(r['reversal']), r['direction']) not in done,
    axis=1)
candidates = candidates[mask].sort_values('oos_ppd', ascending=False).reset_index(drop=True)
print(f"Candidates to test: {len(candidates)}")
print(candidates[['pair','box_pips','reversal','direction','oos_ppd','cday','wf']].to_string(index=False))
print()


@njit
def build_pnf_reversals(hi, lo, box_size, rev_n):
    n=len(hi)
    rev_bars=np.zeros(n,dtype=np.int64); rev_dirs=np.zeros(n,dtype=np.int8); n_rev=0
    if n<2: return rev_bars[:0],rev_dirs[:0]
    col_dir=np.int8(1); col_extreme=(hi[0]+lo[0])*0.5
    rev_thresh=rev_n*box_size
    for i in range(1,n):
        h=hi[i]; l=lo[i]
        if col_dir==1:
            if h>=col_extreme+box_size:
                col_extreme+=math.floor((h-col_extreme)/box_size)*box_size
            elif l<=col_extreme-rev_thresh:
                col_dir=np.int8(-1); col_extreme-=box_size
                rev_bars[n_rev]=i; rev_dirs[n_rev]=np.int8(-1); n_rev+=1
        else:
            if l<=col_extreme-box_size:
                col_extreme-=math.floor((col_extreme-l)/box_size)*box_size
            elif h>=col_extreme+rev_thresh:
                col_dir=np.int8(1); col_extreme+=box_size
                rev_bars[n_rev]=i; rev_dirs[n_rev]=np.int8(1); n_rev+=1
    return rev_bars[:n_rev],rev_dirs[:n_rev]


@njit
def sim_zr_pnf_cycles(op,hi,lo,cl,rev_bars,rev_dirs,dir_mode,
                       pip,spread,pf,ml,zw,tgt,ta,td):
    n=len(cl); n_rev=len(rev_bars)
    cycle_pnl=np.zeros(n_rev,dtype=np.float64); nc=0
    lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    d=np.int8(1); ri=0
    while ri<n_rev:
        entry_bar=rev_bars[ri]; col_d=rev_dirs[ri]; ri+=1
        if entry_bar>=n: break
        direction=float(d) if dir_mode==0 else float(col_d)
        e=cl[entry_bar]
        if direction==1.0:
            uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:
            lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=direction; lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak_mfe=0.0; trail_on=False
        i=entry_bar+1
        while i<n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; bull=c>=op[i]
            if nl==1:
                cur_mfe=(h-e)/pip if direction==1.0 else (e-l)/pip
                if cur_mfe>peak_mfe: peak_mfe=cur_mfe
                if peak_mfe>=ta: trail_on=True
                if trail_on:
                    if direction==1.0:
                        ts=e+(peak_mfe-td)*pip
                        if l<=ts: cycle_pnl[nc]=(ts-e)/pip-spread; nc+=1; ex=True
                    else:
                        ts=e-(peak_mfe-td)*pip
                        if h>=ts: cycle_pnl[nc]=(e-ts)/pip-spread; nc+=1; ex=True
            if ex: break
            for pn in range(2):
                if ex: break
                dh=(bull and pn==0) or (not bull and pn==1)
                if l<=ut<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    cycle_pnl[nc]=net-tv*spread; nc+=1; ex=True; break
                if l<=lt<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    cycle_pnl[nc]=net-tv*spread; nc+=1; ex=True; break
                if dh and h>=uz and lu!=i:
                    lu=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c>=ut: cycle_pnl[nc]=nt2; nc+=1; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            cycle_pnl[nc]=net-tv2*spread; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if not dh and l<=lz and ll!=i:
                    ll=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c<=lt: cycle_pnl[nc]=nt2; nc+=1; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            cycle_pnl[nc]=net-tv2*spread; nc+=1; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            i+=1
        if dir_mode==0: d=np.int8(-d)
        while ri<n_rev and rev_bars[ri]<=i: ri+=1
    return cycle_pnl[:nc], nc


@njit
def sim_zr_entry_bars(op,hi,lo,cl,rev_bars,pip,spread,pf,ml,zw,tgt,ta,td):
    n=len(cl); n_rev=len(rev_bars); total=0.0
    lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    d=np.int8(1); ri=0
    while ri<n_rev:
        entry_bar=rev_bars[ri]; ri+=1
        if entry_bar>=n: break
        direction=float(d); e=cl[entry_bar]
        if direction==1.0:
            uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:
            lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=direction; lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak_mfe=0.0; trail_on=False
        i=entry_bar+1
        while i<n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; bull=c>=op[i]
            if nl==1:
                cur_mfe=(h-e)/pip if direction==1.0 else (e-l)/pip
                if cur_mfe>peak_mfe: peak_mfe=cur_mfe
                if peak_mfe>=ta: trail_on=True
                if trail_on:
                    if direction==1.0:
                        ts=e+(peak_mfe-td)*pip
                        if l<=ts: total+=(ts-e)/pip-spread; ex=True
                    else:
                        ts=e-(peak_mfe-td)*pip
                        if h>=ts: total+=(e-ts)/pip-spread; ex=True
            if ex: break
            for pn in range(2):
                if ex: break
                dh=(bull and pn==0) or (not bull and pn==1)
                if l<=ut<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; ex=True; break
                if l<=lt<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; ex=True; break
                if dh and h>=uz and lu!=i:
                    lu=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c>=ut: total+=nt2; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            total+=net-tv2*spread; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if not dh and l<=lz and ll!=i:
                    ll=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c<=lt: total+=nt2; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            total+=net-tv2*spread; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            i+=1
        d=np.int8(-d)
        while ri<n_rev and rev_bars[ri]<=i: ri+=1
    return total


print("Compiling JIT...", end=' ', flush=True)
_df0=pd.read_parquet(DATA_DIR/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df0.open.values[:2000].astype(np.float64); _h=_df0.high.values[:2000].astype(np.float64)
_l=_df0.low.values[:2000].astype(np.float64);  _c=_df0.close.values[:2000].astype(np.float64)
_rb,_rd=build_pnf_reversals(_h,_l,0.00075,1.5)
sim_zr_pnf_cycles(_o,_h,_l,_c,_rb,_rd,0,0.0001,SPREAD,PF,MAX_LEGS,20.,10.,5.,3.)
sim_zr_entry_bars(_o,_h,_l,_c,_rb,0.0001,SPREAD,PF,MAX_LEGS,20.,10.,5.,3.)
print("done.\n")

rng  = np.random.default_rng(42)
new_rows = []
_loaded = {}

for _, row in candidates.iterrows():
    pair     = row['pair']
    box_pips = int(row['box_pips'])
    rev_n    = float(row['reversal'])
    dir_mode = 0 if row['direction']=='alt' else 1
    dir_lbl  = row['direction']
    pip      = PIP_MAP[pair]
    pcfg     = PAIR_CFG[pair]
    zw=pcfg['zw']; tgt=pcfg['tgt']; ta=pcfg['ta']; td=pcfg['td']
    box_size = box_pips * pip

    if pair not in _loaded:
        df=pd.read_parquet(DATA_DIR/f'{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
        nb=len(df)
        is_end=int(nb*(1-OOS_FRAC))
        _loaded[pair] = dict(
            op=df.open.values.astype(np.float64),
            hi=df.high.values.astype(np.float64),
            lo=df.low.values.astype(np.float64),
            cl=df.close.values.astype(np.float64),
            is_end=is_end,
            oos_days=(nb-is_end)/(24*12),
        )
    dd=_loaded[pair]
    oos_op=dd['op'][dd['is_end']:]; oos_hi=dd['hi'][dd['is_end']:]
    oos_lo=dd['lo'][dd['is_end']:]; oos_cl=dd['cl'][dd['is_end']:]
    oos_days=dd['oos_days']; oos_bars=len(oos_cl)

    oos_rb,oos_rd=build_pnf_reversals(oos_hi,oos_lo,box_size,rev_n)
    n_rev=len(oos_rb)

    cyc,nc=sim_zr_pnf_cycles(oos_op,oos_hi,oos_lo,oos_cl,oos_rb,oos_rd,
                               dir_mode,pip,SPREAD,PF,MAX_LEGS,zw,tgt,ta,td)
    obs_ppd=cyc.sum()/oos_days

    tag=f"{pair} b={box_pips} r={rev_n} {dir_lbl}"
    print(f"\n{'─'*65}")
    print(f"{tag}  obs={obs_ppd:.1f} p/d  n_cyc={nc}  c/day={nc/oos_days:.1f}")

    # Permutation test
    print(f"  Perm ({N_PERM})...", end=' ', flush=True)
    all_bars=np.arange(oos_bars,dtype=np.int64)
    perm_ppd=np.empty(N_PERM)
    for k in range(N_PERM):
        sh=rng.choice(all_bars,size=n_rev,replace=False); sh.sort()
        perm_ppd[k]=sim_zr_entry_bars(oos_op,oos_hi,oos_lo,oos_cl,sh,
                                       pip,SPREAD,PF,MAX_LEGS,zw,tgt,ta,td)/oos_days
    p_val=np.mean(perm_ppd>=obs_ppd)
    perm_med=np.median(perm_ppd); perm_p95=np.percentile(perm_ppd,95)
    print(f"p={p_val:.4f} ({'PASS' if p_val<0.05 else 'FAIL'})  null_med={perm_med:.1f}  null_p95={perm_p95:.1f}")

    # Bootstrap
    print(f"  Boot ({N_BOOT})...", end=' ', flush=True)
    boot_ppd=np.array([rng.choice(cyc,size=nc,replace=True).sum()/oos_days for _ in range(N_BOOT)])
    p5,p25,p50,p75,p95=np.percentile(boot_ppd,[5,25,50,75,95])
    sharpe=boot_ppd.mean()/(boot_ppd.std()+1e-9)
    prob_pos=np.mean(boot_ppd>0)
    print(f"P5={p5:.0f}  P50={p50:.0f}  P95={p95:.0f}  Sharpe={sharpe:.2f}  P(+)={prob_pos:.3f}")

    gp=p_val<0.05; g5=p5>0; gpp=prob_pos>0.95
    gates=sum([gp,g5,gpp])
    print(f"  Gates: perm={'✅' if gp else '❌'}  P5={'✅' if g5 else '❌'}  P(+)={'✅' if gpp else '❌'}  → {gates}/3")
    sys.stdout.flush()

    new_rows.append(dict(
        pair=pair, box_pips=box_pips, reversal=rev_n, direction=dir_lbl,
        obs_ppd=round(obs_ppd,1), n_cycles=nc,
        perm_null_median=round(perm_med,1), perm_null_p95=round(perm_p95,1),
        p_value=round(p_val,4),
        boot_p5=round(p5,0), boot_p25=round(p25,0), boot_median=round(p50,0),
        boot_p75=round(p75,0), boot_p95=round(p95,0),
        sharpe=round(sharpe,2), prob_pos=round(prob_pos,3),
        gate_perm=int(gp), gate_p5=int(g5), gate_prob=int(gpp), gates=gates,
    ))

df_new=pd.DataFrame(new_rows)
if PERM_CSV.exists():
    df_all=pd.concat([pd.read_csv(PERM_CSV),df_new],ignore_index=True)
else:
    df_all=df_new
df_all.to_csv(PERM_CSV,index=False)
print(f"\n\nAppended {len(df_new)} rows → {PERM_CSV}  (total={len(df_all)})")

print("\n=== NEW RESULTS — 3/3 GATES ===")
passed=df_new[df_new.gates==3].sort_values('obs_ppd',ascending=False)
print(passed[['pair','box_pips','reversal','direction','obs_ppd','n_cycles','p_value','boot_p5','prob_pos','gates']].to_string(index=False))

print("\n=== ALL NEW — sorted by ppd ===")
print(f"{'Config':38} {'ppd':>8} {'p-val':>8} {'P5':>7} {'P(+)':>6} {'gates':>5}")
print("─"*80)
for _,r in df_new.sort_values('obs_ppd',ascending=False).iterrows():
    tag=f"{r.pair} b={r.box_pips} r={r.reversal} {r.direction}"
    pf2="✅" if r.p_value<0.05 else "❌"
    print(f"{tag:38} {r.obs_ppd:>8.0f} {r.p_value:>7.4f}{pf2} "
          f"{r.boot_p5:>7.0f} {r.prob_pos:>6.3f} {r.gates:>5}/3")
