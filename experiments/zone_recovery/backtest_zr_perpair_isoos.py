"""
Per-pair ZR sweet spot: sweep on IS (first 70%), pick winner, validate OOS (last 30%).
Dense grid: N × ZW × tgt_f. Report IS winner, OOS result, IS/OOS consistency.
"""
import numpy as np, pandas as pd, math
from numba import njit
from pathlib import Path
from itertools import product

DATA_DIR = Path('/path/to/projects/fx-core/data/m5_ohlc')
PAIRS = ["EUR_USD","AUD_USD","GBP_USD","NZD_USD","EUR_GBP",
         "EUR_JPY","CHF_JPY","NZD_JPY","AUD_JPY","CAD_JPY","USD_JPY","GBP_JPY"]
PIP_MAP     = {p:0.01 if "JPY" in p else 0.0001 for p in PAIRS}
PIP_USD_MAP = {"GBP_JPY":0.000091,"EUR_JPY":0.000064,"USD_JPY":0.000064,
               "AUD_JPY":0.000067,"CAD_JPY":0.000069,"CHF_JPY":0.000107,
               "NZD_JPY":0.000061,"EUR_USD":0.000100,"GBP_USD":0.000100,
               "AUD_USD":0.000100,"NZD_USD":0.000100,"EUR_GBP":0.000126}
SPREAD=1.4; MAX_LEGS=10; PF=1.25

@njit
def sim_zr(op,hi,lo,cl,pip,spread,pf,ml,N,zw,tgt):
    n=len(cl); total=0.0; nc=0; nm=0; sl=0.0
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
                    total+=net-tv*spread; nc+=1; sl+=nl; ex=True; break
                if l<=lt<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; nc+=1; sl+=nl; ex=True; break
                if dh and h>=uz and lu!=i:
                    lu=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c>=ut: total+=nt2; nc+=1; sl+=nl; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            total+=net-tv2*spread; nc+=1; nm+=1; sl+=nl; ex=True; break
                        lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if not dh and l<=lz and ll!=i:
                    ll=i; nt2=0.0; tv=0.0
                    for k in range(nl): nt2+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    nt2-=tv*spread
                    if nt2>=0:
                        if c<=lt: total+=nt2; nc+=1; sl+=nl; ex=True; break
                    else:
                        v=max(1.0,math.ceil(-nt2/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv2=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv2+=lv[k]
                            total+=net-tv2*spread; nc+=1; nm+=1; sl+=nl; ex=True; break
                        lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            i+=1
        d=-d; i+=N-1
    return total, nc, nm, sl/max(nc,1)

_df=pd.read_parquet(DATA_DIR/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df.open.to_numpy(float)[:2000]; _h=_df.high.to_numpy(float)[:2000]
_l=_df.low.to_numpy(float)[:2000]; _c=_df.close.to_numpy(float)[:2000]
_=sim_zr(_o,_h,_l,_c,0.0001,SPREAD,PF,MAX_LEGS,1,20.,10.)
print("JIT compiled\n")

# Dense grid
NS    = [1, 2, 3, 6, 12]
ZWS   = [10., 15., 20., 25., 30., 40., 50., 60.]
TGT_F = [0.25, 0.50, 1.00]

results = []
print(f"{'pair':>10} | {'IS_winner':>22} | {'IS_ppd':>8} {'IS_cpd':>7} | {'OOS_ppd':>8} {'OOS_cpd':>7} {'WF':>4} | {'OOS_ppc':>8}")
print("─"*88)

for pair in PAIRS:
    pip = PIP_MAP[pair]; pip_usd = PIP_USD_MAP[pair]
    df = pd.read_parquet(DATA_DIR/f"{pair}_M5.parquet").sort_values('timestamp').reset_index(drop=True)
    n_split = int(len(df)*0.70)
    is_df  = df.iloc[:n_split].reset_index(drop=True)
    oos_df = df.iloc[n_split:].reset_index(drop=True)
    is_days  = len(is_df)*5/(60*24*5/7)
    oos_days = len(oos_df)*5/(60*24*5/7)
    oos_ch_len = len(oos_df)//3

    iso  = is_df.open.to_numpy(float);  ish  = is_df.high.to_numpy(float)
    isl  = is_df.low.to_numpy(float);   isc  = is_df.close.to_numpy(float)
    ooso = oos_df.open.to_numpy(float); oosh = oos_df.high.to_numpy(float)
    oosl = oos_df.low.to_numpy(float);  oosc = oos_df.close.to_numpy(float)

    # Sweep IS
    best_is_ppd = -1e9; best_cfg = None; best_is_nc = 0
    for N,zw,tf in product(NS, ZWS, TGT_F):
        tgt = zw*tf
        tp,nc,nm,_ = sim_zr(iso,ish,isl,isc,pip,SPREAD,PF,MAX_LEGS,N,zw,tgt)
        ml_pct = nm/max(nc,1)
        if ml_pct > 0.001: continue          # reject if >0.1% max-legs
        ppd = tp/is_days
        if ppd > best_is_ppd and nc >= 30:   # min 30 IS cycles
            best_is_ppd = ppd; best_cfg = (N,zw,tf,tgt); best_is_nc = nc

    if best_cfg is None:
        print(f"{pair:>10} | {'NO VALID IS CONFIG':>22} |")
        continue

    N,zw,tf,tgt = best_cfg
    is_cpd = best_is_nc / is_days

    # Validate OOS
    oos_tp,oos_nc,_,avgl = sim_zr(ooso,oosh,oosl,oosc,pip,SPREAD,PF,MAX_LEGS,N,zw,tgt)
    oos_ppd = oos_tp/oos_days; oos_cpd = oos_nc/oos_days
    oos_ppc = oos_tp/max(oos_nc,1)

    # WF check (3 OOS chunks)
    wf=0
    for k in range(3):
        s=k*oos_ch_len; e=(k+1)*oos_ch_len if k<2 else len(oos_df)
        ch=oos_df.iloc[s:e].reset_index(drop=True)
        ctp,_,_,_=sim_zr(ch.open.to_numpy(float),ch.high.to_numpy(float),
                          ch.low.to_numpy(float),ch.close.to_numpy(float),
                          pip,SPREAD,PF,MAX_LEGS,N,zw,tgt)
        if ctp>0: wf+=1

    cfg_str = f"N={N} ZW={zw:.0f} tf={tf:.2f}"
    flag = "✅" if (oos_tp>0 and wf>=2) else ("⚠️" if oos_tp>0 else "❌")
    print(f"{pair:>10} | {cfg_str:>22} | {best_is_ppd:>8.1f} {is_cpd:>7.2f} | "
          f"{oos_ppd:>8.1f} {oos_cpd:>7.2f} {wf}/3 | {oos_ppc:>8.1f} {flag}")

    results.append(dict(pair=pair,N=N,zw=zw,tf=tf,tgt=tgt,
                        is_ppd=best_is_ppd,is_cpd=is_cpd,
                        oos_ppd=oos_ppd,oos_cpd=oos_cpd,oos_ppc=oos_ppc,
                        avgl=avgl,wf=wf,pip_usd=pip_usd,
                        oos_usd_ppd=oos_ppd*pip_usd))

rdf = pd.DataFrame(results)
rdf.to_csv('/tmp/zr_perpair_is_oos.csv', index=False)

print("\n\n=== DEPLOYABILITY RANKING (IS-selected, OOS-validated) ===")
valid = rdf[rdf.oos_ppd>0].sort_values('oos_usd_ppd', ascending=False)
print(f"{'Rk':>3} {'pair':>10} {'N':>3} {'ZW':>4} {'tf':>5} | {'IS_ppd':>8} {'OOS_ppd':>8} {'OOS_cpd':>7} {'OOS_ppc':>8} | {'WF':>4} {'$ppd':>8}")
print("─"*88)
for rk,(_,r) in enumerate(valid.iterrows(),1):
    flag = "✅" if r.wf>=2 else "⚠️"
    print(f"{rk:>3}. {r.pair:>10} {r.N:>3} {r.zw:>4.0f} {r.tf:>5.2f} | "
          f"{r.is_ppd:>8.1f} {r.oos_ppd:>8.1f} {r.oos_cpd:>7.2f} {r.oos_ppc:>8.1f} | "
          f"{r.wf}/3 {r.oos_usd_ppd*1000:>7.2f}$ {flag}")
