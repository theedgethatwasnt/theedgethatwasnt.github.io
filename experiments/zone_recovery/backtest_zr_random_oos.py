"""OOS-only validation of best random-entry ZR configs on EUR_USD."""
import numpy as np, pandas as pd, math
from numba import njit

df = pd.read_parquet('/path/to/projects/fx-core/data/m5_ohlc/EUR_USD_M5.parquet').sort_values('timestamp').reset_index(drop=True)
# OOS = last 30%
oos = df.iloc[int(len(df)*0.70):].reset_index(drop=True)
op = oos.open.to_numpy(float); hi = oos.high.to_numpy(float)
lo = oos.low.to_numpy(float);  cl = oos.close.to_numpy(float)
PIP = 0.0001; SPREAD = 1.4; MAX_LEGS = 10
oos_days = len(oos) * 5 / (60*24*5/7)
print(f"OOS: {len(oos):,} bars  ≈ {oos_days:.0f} trading days")

@njit
def sim_zr(op,hi,lo,cl,pip,spread,pf,ml,entry_n,zw,tgt):
    n=len(cl); total=0.0; nc=0; nt=0; nm=0; sl=0.0
    lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    i=0; direction=1
    while i<n:
        entry=cl[i]
        if direction==1: uz=entry; lz=entry-zw*pip; ut=entry+tgt*pip; lt=lz-tgt*pip
        else:            lz=entry; uz=entry+zw*pip; lt=entry-tgt*pip; ut=uz+tgt*pip
        lv[0]=1.0; ld[0]=float(direction); lp[0]=entry
        nl=1; lu=ll=-1; exited=False; i+=1
        while i<n and not exited:
            h=hi[i]; l=lo[i]; c=cl[i]; bull=c>=op[i]
            for pn in range(2):
                if exited: break
                dh=(bull and pn==0) or (not bull and pn==1)
                if l<=ut<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; nc+=1; nt+=1; sl+=nl; exited=True; break
                if l<=lt<=h:
                    net=0.0; tv=0.0
                    for k in range(nl): net+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    total+=net-tv*spread; nc+=1; nt+=1; sl+=nl; exited=True; break
                if dh and h>=uz and lu!=i:
                    lu=i; net_t=0.0; tv=0.0
                    for k in range(nl): net_t+=lv[k]*ld[k]*(ut-lp[k])/pip; tv+=lv[k]
                    net_t-=tv*spread
                    if net_t>=0:
                        if c>=ut: total+=net_t; nc+=1; nt+=1; sl+=nl; exited=True; break
                    else:
                        vol=max(1.0,math.ceil(-net_t/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv+=lv[k]
                            total+=net-tv*spread; nc+=1; nm+=1; sl+=nl; exited=True; break
                        lv[nl]=vol; ld[nl]=1.0; lp[nl]=uz; nl+=1
                if not dh and l<=lz and ll!=i:
                    ll=i; net_t=0.0; tv=0.0
                    for k in range(nl): net_t+=lv[k]*ld[k]*(lt-lp[k])/pip; tv+=lv[k]
                    net_t-=tv*spread
                    if net_t>=0:
                        if c<=lt: total+=net_t; nc+=1; nt+=1; sl+=nl; exited=True; break
                    else:
                        vol=max(1.0,math.ceil(-net_t/tgt*pf))
                        if nl>=ml:
                            net=0.0; tv=0.0
                            for k in range(nl): net+=lv[k]*ld[k]*(c-lp[k])/pip; tv+=lv[k]
                            total+=net-tv*spread; nc+=1; nm+=1; sl+=nl; exited=True; break
                        lv[nl]=vol; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            i+=1
        direction=-direction; i+=entry_n-1
    return total, nc, nt, nm, sl/max(nt+nm,1)

_=sim_zr(op[:2000],hi[:2000],lo[:2000],cl[:2000],PIP,SPREAD,1.25,MAX_LEGS,1,30.,7.5)
print("JIT compiled\n")

# Top configs from full-data sweep + some extras
configs = [
    (3, 30, 0.25),   # best overall
    (6, 30, 0.50),   # second
    (3, 40, 0.50),   # third
    (1, 56, 0.25),   # fast+wide
    (6, 40, 0.50),
    (12,30, 0.50),
    (24,30, 0.50),
    (24,40, 1.00),
    (1, 30, 0.50),
    (1, 30, 1.00),
    (3, 20, 0.50),
]

print(f"{'N':>4} {'ZW':>4} {'tgt_f':>6} | {'pips':>9} {'p/day':>7} {'c/day':>6} {'ppc':>7} {'ml%':>6} {'avgl':>5}")
print("─"*66)
for (n, zw, tf) in configs:
    tgt = zw*tf
    tp,nc,nt,nm,avgl = sim_zr(op,hi,lo,cl,PIP,SPREAD,1.25,MAX_LEGS,n,zw,tgt)
    ppd=tp/oos_days; cpd=nc/oos_days; ppc=tp/max(nc,1); ml_pct=nm/max(nc,1)*100
    flag="🟢" if tp>0 else "🔴"
    print(f"{n:>4} {zw:>4} {tf:>6.2f} | {tp:>9.0f} {ppd:>7.1f} {cpd:>6.2f} {ppc:>7.1f} {ml_pct:>5.1f}% {avgl:>5.1f} {flag}")
