"""
Rolling predictive WF test: train on window N, trade window N+1 with that winner.
Compare against fixed reference config and naive "always use global sweetspot".
"""
import numpy as np, pandas as pd, math
from numba import njit
from pathlib import Path

DATA_DIR = Path('/path/to/projects/fx-core/data/m5_ohlc')
PIP_MAP  = {"CHF_JPY":0.01,"GBP_USD":0.0001,"USD_JPY":0.01}
SPREAD=1.4; MAX_LEGS=10; PF=1.25

PAIR_CFG = {
    "CHF_JPY": dict(N_ref=1, zw_ref=40.0, tgt_ref=20.0, ta_ref=5, td_ref=3),
    "GBP_USD": dict(N_ref=6, zw_ref=30.0, tgt_ref=15.0, ta_ref=10, td_ref=7),
    "USD_JPY": dict(N_ref=1, zw_ref=40.0, tgt_ref=20.0, ta_ref=10, td_ref=5),
}

ZWS    = [20, 25, 30, 35, 40, 45, 50, 55, 60]
NS     = [1, 2, 3, 6, 12]
TGT_FS = [0.25, 0.50, 1.00]
TAS    = [5, 7, 10, 14, 20, 30]
TDS    = [3, 5, 7, 10]
TATD   = [(ta,td) for ta in TAS for td in TDS if td < ta]

@njit
def sim_zr_trail(op,hi,lo,cl,pip,spread,pf,ml,N,zw,tgt,ta,td):
    n=len(cl); total=0.0; nc=0
    lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    i=0; d=1
    while i<n:
        e=cl[i]
        if d==1: uz=e; lz=e-zw*pip; ut=e+tgt*pip; lt=lz-tgt*pip
        else:    lz=e; uz=e+zw*pip; lt=e-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(d); lp[0]=e
        nl=1; lu=ll=-1; ex=False; peak_mfe=0.0; trail_on=False
        i+=1
        while i<n and not ex:
            h=hi[i]; l=lo[i]; c=cl[i]; bull=c>=op[i]
            if nl==1:
                cur_mfe=(h-e)/pip if d==1 else (e-l)/pip
                if cur_mfe>peak_mfe: peak_mfe=cur_mfe
                if peak_mfe>=ta: trail_on=True
                if trail_on:
                    if d==1:
                        ts=e+(peak_mfe-td)*pip
                        if l<=ts: total+=(ts-e)/pip-spread; nc+=1; ex=True
                    else:
                        ts=e-(peak_mfe-td)*pip
                        if h>=ts: total+=(e-ts)/pip-spread; nc+=1; ex=True
            if ex: break
            for pn in range(2):
                if ex: break
                dh=(bull and pn==0) or (not bull and pn==1)
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
            i+=1
        d=-d; i+=N-1
    return total, nc

_df0=pd.read_parquet(DATA_DIR/'EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
_o=_df0.open.values[:2000].astype(float); _h=_df0.high.values[:2000].astype(float)
_l=_df0.low.values[:2000].astype(float); _c=_df0.close.values[:2000].astype(float)
sim_zr_trail(_o,_h,_l,_c,0.0001,SPREAD,PF,MAX_LEGS,1,20.,10.,10.,5.)
print("JIT compiled\n")

OOS_FRAC=0.30; N_WINDOWS=6; MIN_CYCLES=10

for pair, cfg in PAIR_CFG.items():
    pip=PIP_MAP[pair]
    df=pd.read_parquet(DATA_DIR/f'{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
    op=df.open.values.astype(float); hi=df.high.values.astype(float)
    lo=df.low.values.astype(float); cl=df.close.values.astype(float)
    nb=len(cl)
    oos_start=int(nb*(1-OOS_FRAC))
    oo=op[oos_start:]; oh=hi[oos_start:]; ol=lo[oos_start:]; oc=cl[oos_start:]
    oos_bars=len(oc); win_sz=oos_bars//N_WINDOWS

    print(f"\n{'='*65}")
    print(f"{pair}  (ref: N={cfg['N_ref']} ZW={cfg['zw_ref']} tgt={cfg['tgt_ref']} ta={cfg['ta_ref']} td={cfg['td_ref']})")
    print(f"{'='*65}")
    print(f"{'Win':>3} {'Trained-on':>10} {'Traded-on':>10} {'Adaptive':>10} {'Fixed-Ref':>10} {'Delta':>8}")
    print(f"{'---':>3} {'----------':>10} {'----------':>10} {'--------':>10} {'--------':>10} {'-----':>8}")

    prev_winner = None
    adaptive_total = 0.0; ref_total = 0.0
    results = []

    for w in range(N_WINDOWS):
        train_s=w*win_sz; train_e=(w+1)*win_sz if w<N_WINDOWS-1 else oos_bars
        # Train window
        wo_tr=oo[train_s:train_e]; wh_tr=oh[train_s:train_e]
        wl_tr=ol[train_s:train_e]; wc_tr=oc[train_s:train_e]
        tr_days=(train_e-train_s)/(24*12)

        # Find winner on train window
        best_ppd=-1e9; best_cfg=None
        for N in NS:
            for zw in ZWS:
                for tgt_f in TGT_FS:
                    tgt=zw*tgt_f
                    if tgt<5: continue
                    for (ta,td) in TATD:
                        tot,nc=sim_zr_trail(wo_tr,wh_tr,wl_tr,wc_tr,pip,SPREAD,PF,MAX_LEGS,
                                            N,float(zw),tgt,float(ta),float(td))
                        if nc<MIN_CYCLES: continue
                        ppd=tot/(train_e-train_s)*(24*12)
                        if ppd>best_ppd: best_ppd=ppd; best_cfg=(N,zw,tgt_f,zw*tgt_f,ta,td)

        # Test window = next window (w+1)
        if w < N_WINDOWS-1:
            test_s=train_e; test_e=(w+2)*win_sz if w+1<N_WINDOWS-1 else oos_bars
            wo_te=oo[test_s:test_e]; wh_te=oh[test_s:test_e]
            wl_te=ol[test_s:test_e]; wc_te=oc[test_s:test_e]
            te_days=(test_e-test_s)/(24*12)

            # Adaptive: use THIS window's winner on NEXT window
            if best_cfg:
                aN,aZW,atf,atgt,ata,atd = best_cfg
                adapt_tot,adapt_nc=sim_zr_trail(wo_te,wh_te,wl_te,wc_te,pip,SPREAD,PF,MAX_LEGS,
                                                aN,float(aZW),atgt,float(ata),float(atd))
                adapt_ppd=adapt_tot/(test_e-test_s)*(24*12)
            else:
                adapt_ppd=0.0

            # Fixed reference
            ref_tot,ref_nc=sim_zr_trail(wo_te,wh_te,wl_te,wc_te,pip,SPREAD,PF,MAX_LEGS,
                                         cfg['N_ref'],cfg['zw_ref'],cfg['tgt_ref'],
                                         float(cfg['ta_ref']),float(cfg['td_ref']))
            ref_ppd=ref_tot/(test_e-test_s)*(24*12)

            delta=adapt_ppd-ref_ppd
            sign="✓" if adapt_ppd>ref_ppd else "✗"
            print(f"  W{w+1}→W{w+2}  train=W{w+1}({tr_days:.0f}d)  test=W{w+2}({te_days:.0f}d)"
                  f"  adapt={adapt_ppd:8.0f}  ref={ref_ppd:8.0f}  Δ={delta:+8.0f} {sign}")
            if best_cfg:
                aN,aZW,atf,atgt,ata,atd=best_cfg
                print(f"         winner: N={aN} ZW={aZW} tgt_f={atf} ta={ata} td={atd} (IS={best_ppd:.0f} p/d)")
            adaptive_total+=adapt_ppd; ref_total+=ref_ppd
            results.append((adapt_ppd, ref_ppd))

    n_wins=sum(1 for a,r in results if a>r)
    avg_adapt=np.mean([a for a,r in results])
    avg_ref=np.mean([r for a,r in results])
    print(f"\n  SUMMARY: adaptive wins {n_wins}/{len(results)} forward windows")
    print(f"  Avg forward p/d: adaptive={avg_adapt:.0f}  fixed-ref={avg_ref:.0f}  Δ={avg_adapt-avg_ref:+.0f}")
    print(f"  Conclusion: {'adaptive BEATS fixed' if avg_adapt>avg_ref else 'fixed ref BEATS adaptive'}")
