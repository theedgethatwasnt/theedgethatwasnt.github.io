"""
Random-entry Zone Recovery — 12-pair sweep.
OOS only (last 30%). Walk-forward 3 chunks per pair.
Tests top configs from EUR_USD analysis.
"""
import numpy as np, pandas as pd, math
from numba import njit
from pathlib import Path

DATA_DIR = Path('/path/to/projects/fx-core/data/m5_ohlc')
PAIRS = ["GBP_JPY","EUR_JPY","USD_JPY","AUD_JPY","CAD_JPY","CHF_JPY",
         "EUR_USD","GBP_USD","AUD_USD","NZD_USD","EUR_GBP","NZD_JPY"]
PIP_MAP     = {p: 0.01 if "JPY" in p else 0.0001 for p in PAIRS}
PIP_USD_MAP = {"GBP_JPY":0.000091,"EUR_JPY":0.000064,"USD_JPY":0.000064,
               "AUD_JPY":0.000067,"CAD_JPY":0.000069,"CHF_JPY":0.000107,
               "NZD_JPY":0.000061,"EUR_USD":0.000100,"GBP_USD":0.000100,
               "AUD_USD":0.000100,"NZD_USD":0.000100,"EUR_GBP":0.000126}
SPREAD=1.4; MAX_LEGS=10; PF=1.25; OOS_FRAC=0.30

@njit
def sim_zr(op,hi,lo,cl,pip,spread,pf,ml,N,zw,tgt):
    n=len(cl); total=0.0; nc=0; nt=0; nm=0; sl=0.0
    lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    i=0; d=1
    while i<n:
        e=cl[i]
        if d==1: uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:    lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e; nl=1; lu=ll=-1; ex=False; i+=1
        while i<n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; b=c>=op[i]
            for pn in range(2):
                if ex: break
                dh=(b and pn==0) or (not b and pn==1)
                if l<=ut<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; nc+=1; nt+=1; sl+=nl; ex=True; break
                if l<=lt<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; nc+=1; nt+=1; sl+=nl; ex=True; break
                if dh and h>=uz and lu!=i:
                    lu=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c>=ut: total+=nt2; nc+=1; nt+=1; sl+=nl; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv+=lv[k]
                            total+=net-tv*spread; nc+=1; nm+=1; sl+=nl; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if not dh and l<=lz and ll!=i:
                    ll=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c<=lt: total+=nt2; nc+=1; nt+=1; sl+=nl; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv+=lv[k]
                            total+=net-tv*spread; nc+=1; nm+=1; sl+=nl; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            i+=1
        d=-d; i+=N-1
    return total, nc, nt, nm, sl/max(nt+nm,1)

# warmup
_df = pd.read_parquet(DATA_DIR/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_oos = _df.iloc[int(len(_df)*0.70):].reset_index(drop=True)
_op=_oos.open.to_numpy(float); _hi=_oos.high.to_numpy(float)
_lo=_oos.low.to_numpy(float);  _cl=_oos.close.to_numpy(float)
_=sim_zr(_op[:2000],_hi[:2000],_lo[:2000],_cl[:2000],0.0001,SPREAD,PF,MAX_LEGS,1,20.,10.)
print("JIT compiled")

# Top validated configs from EUR_USD sweep
CONFIGS = [
    (2, 25., 1.00),   # best ppc, 3/3 WF
    (3, 25., 0.50),   # best c/day+pday balance, 3/3 WF
    (3, 25., 1.00),   # 3/3 WF
    (1, 25., 1.00),   # 3/3 WF
    (1, 40., 1.00),   # highest ppc, 3/3 WF
    (2, 20., 1.00),   # 3/3 WF
    (1, 20., 0.50),   # fastest cycling, 3/3 WF
    (1, 30., 1.00),   # 3/3 WF
    (1, 30., 0.50),   # 3/3 WF
    (1, 40., 0.25),   # fast+wide, 3/3 WF
    (3, 20., 1.00),   # 3/3 WF
]

all_rows = []
for pair in PAIRS:
    pip = PIP_MAP[pair]; pip_usd = PIP_USD_MAP[pair]
    pf_path = DATA_DIR / f"{pair}_M5.parquet"
    if not pf_path.exists(): continue
    df = pd.read_parquet(pf_path).sort_values('timestamp').reset_index(drop=True)
    n_split = int(len(df)*0.70)
    oos = df.iloc[n_split:].reset_index(drop=True)
    oos_days = len(oos)*5/(60*24*5/7)
    ch_len = len(oos)//3
    chunks = [oos.iloc[i*ch_len:(i+1)*ch_len].reset_index(drop=True) for i in range(3)]

    print(f"{pair} ({len(oos):,} OOS bars, {oos_days:.0f} days)...", flush=True)
    op=oos.open.to_numpy(float); hi=oos.high.to_numpy(float)
    lo=oos.low.to_numpy(float);  cl=oos.close.to_numpy(float)

    for (N,zw,tf) in CONFIGS:
        tgt=zw*tf
        tp,nc,_,nm,avgl=sim_zr(op,hi,lo,cl,pip,SPREAD,PF,MAX_LEGS,N,zw,tgt)
        ppd=tp/oos_days; cpd=nc/oos_days; ppc=tp/max(nc,1)
        ml_pct=nm/max(nc,1)*100; usd=tp*pip_usd
        wf=0
        for ch in chunks:
            cop=ch.open.to_numpy(float); chi=ch.high.to_numpy(float)
            clo=ch.low.to_numpy(float);  ccl=ch.close.to_numpy(float)
            ctp,_,_,_,_=sim_zr(cop,chi,clo,ccl,pip,SPREAD,PF,MAX_LEGS,N,zw,tgt)
            if ctp>0: wf+=1
        all_rows.append(dict(pair=pair,N=N,zw=zw,tf=tf,tp=tp,ppd=ppd,cpd=cpd,
                             ppc=ppc,ml=ml_pct,avgl=avgl,usd=usd,wf=wf,oos_days=oos_days))

df_all = pd.DataFrame(all_rows)
df_all.to_csv('/tmp/zr_12pair_results.csv', index=False)
print(f"\nSaved {len(df_all)} rows.\n")

# Aggregate by config
agg = (df_all.groupby(['N','zw','tf'])
       .agg(total_pips=('tp','sum'), total_usd=('usd','sum'),
            n_pairs_pos=('tp', lambda x:(x>0).sum()),
            wf_all3=('wf', lambda x:(x==3).sum()),
            wf_any=('wf', lambda x:(x>0).sum()),
            avg_ppd=('ppd','mean'), avg_cpd=('cpd','mean'),
            avg_ppc=('ppc','mean'), avg_ml=('ml','mean'),
            avg_legs=('avgl','mean'))
       .reset_index())
agg['total_pips_per_day'] = agg['total_pips'] / df_all.groupby(['N','zw','tf'])['oos_days'].first().values
agg = agg.sort_values('total_pips', ascending=False)

print("="*100)
print("12-PAIR AGGREGATE — RANDOM ENTRY ZONE RECOVERY (OOS 30%, ~573 days per pair)")
print("="*100)
print(f"{'N':>3} {'ZW':>4} {'tf':>5} | {'agg_pips':>10} {'$/1ku':>8} | {'pos':>4} {'WF3':>4} | {'ppd_avg':>8} {'cpd_avg':>8} {'ppc_avg':>8} {'ml%':>5}")
print("─"*100)
for _,r in agg.iterrows():
    usd_1ku = r.total_usd  # already at 1 unit; ×1000 for 1ku
    print(f"{r.N:>3} {r.zw:>4.0f} {r.tf:>5.2f} | {r.total_pips:>10.0f} {usd_1ku*1000:>8.0f} | "
          f"{r.n_pairs_pos:>3.0f}/12 {r.wf_all3:>3.0f}/12 | "
          f"{r.avg_ppd:>8.1f} {r.avg_cpd:>8.2f} {r.avg_ppc:>8.1f} {r.avg_ml:>4.1f}%")

print("\n\nPER-PAIR DETAIL (best config per pair by total pips):")
best = df_all.sort_values('tp',ascending=False).drop_duplicates('pair')
print(f"{'pair':>10} {'N':>3} {'ZW':>4} {'tf':>5} | {'pips':>8} {'ppd':>7} {'cpd':>6} {'ppc':>7} {'WF':>4}")
print("─"*65)
for _,r in best.sort_values('tp',ascending=False).iterrows():
    print(f"{r.pair:>10} {r.N:>3} {r.zw:>4.0f} {r.tf:>5.2f} | {r.tp:>8.0f} {r.ppd:>7.1f} {r.cpd:>6.2f} {r.ppc:>7.1f} {r.wf:.0f}/3")
