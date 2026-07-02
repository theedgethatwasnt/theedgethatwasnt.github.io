import os, math
import numpy as np
import pandas as pd
from numba import njit

DATA_DIR = os.path.expanduser('~/projects/fx-core/data/m5_ohlc')
PAIRS = ['AUD_JPY','AUD_USD','CAD_JPY','CHF_JPY','EUR_GBP','EUR_JPY',
         'EUR_USD','GBP_JPY','GBP_USD','NZD_JPY','NZD_USD','USD_JPY']
PIP_USD_MAP = {'AUD_JPY':0.000067,'AUD_USD':0.000100,'CAD_JPY':0.000069,
               'CHF_JPY':0.000107,'EUR_GBP':0.000126,'EUR_JPY':0.000064,
               'EUR_USD':0.000100,'GBP_JPY':0.000091,'GBP_USD':0.000100,
               'NZD_JPY':0.000061,'NZD_USD':0.000100,'USD_JPY':0.000064}
PIP_MAP = {p: 0.01 if 'JPY' in p else 0.0001 for p in PAIRS}
UNITS=1_000; MAX_LEGS=10; PF=1.25; SPREAD=1.4

@njit(cache=True)
def _sim(close_a, open_a, high_a, low_a, act_h, act_l, tgt_frac, pip, spread, pf, ml):
    tp=0.0; nc=0; nt=0; nm=0; sl=0.0
    n=len(close_a)
    lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    def nb(nl,price):
        g=0.0;c=0.0
        for k in range(nl): g+=lv[k]*ld[k]*(price-lp[k])/pip; c+=lv[k]
        return g-c*spread
    def bv(nl,tgt,tp2):
        net=nb(nl,tgt)
        if net>=0.0: return 0.0
        return max(1.0,math.ceil(-net/tp2*pf))
    i=0
    while i<n:
        uh=act_h[i]; ul=act_l[i]
        if uh!=uh or ul!=ul or uh<=ul: i+=1; continue
        zw=(uh-ul)/pip; tp2=zw*tgt_frac; tb=tp2*pip
        entry=close_a[i]
        if entry<=ul: dr=1.0
        elif entry>=uh: dr=-1.0
        else: i+=1; continue
        ut=uh+tb; lt=ul-tb
        lv[0]=1.0; ld[0]=dr; lp[0]=entry; nl=1; lu=ll=-1; cl2=False; ep=entry; it=False; im=False
        i+=1
        while i<n and not cl2:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            for p2 in range(2):
                if cl2: break
                if (bull and p2==0) or (not bull and p2==1):
                    if hi>=ut: ep=ut; cl2=True; it=True; break
                    if hi>=uh and lu!=i:
                        lu=i; v=bv(nl,ut,tp2)
                        if v>0:
                            if nl>=ml: ep=cl; cl2=True; im=True; break
                            lv[nl]=v; ld[nl]=1.0; lp[nl]=uh; nl+=1
                else:
                    if lo<=lt: ep=lt; cl2=True; it=True; break
                    if lo<=ul and ll!=i:
                        ll=i; v=bv(nl,lt,tp2)
                        if v>0:
                            if nl>=ml: ep=cl; cl2=True; im=True; break
                            lv[nl]=v; ld[nl]=-1.0; lp[nl]=ul; nl+=1
            if not cl2: i+=1
        tp+=nb(nl,ep); nc+=1; sl+=nl
        if it: nt+=1
        if im: nm+=1
        if not cl2: break
    return tp,nc,nt,nm,sl

@njit(cache=True)
def _base(close_a, open_a, high_a, low_a, rng_dirs, pip, spread, pf, ml):
    ZW=56.0; TGT=28.0; tp=0.0; nc=0; nt=0; nm=0; sl=0.0
    n=len(close_a); lv=np.zeros(ml); ld=np.zeros(ml); lp=np.zeros(ml)
    def nb(nl,price):
        g=0.0;c=0.0
        for k in range(nl): g+=lv[k]*ld[k]*(price-lp[k])/pip; c+=lv[k]
        return g-c*spread
    def bv(nl,tgt):
        net=nb(nl,tgt)
        if net>=0.0: return 0.0
        return max(1.0,math.ceil(-net/TGT*pf))
    i=0; ri=0
    while i<n:
        entry=close_a[i]; dr=rng_dirs[ri % len(rng_dirs)]; ri+=1
        if dr==1: uz=entry; lz=entry-ZW*pip; ut=entry+TGT*pip; lt=lz-TGT*pip
        else:     lz=entry; uz=entry+ZW*pip; lt=entry-TGT*pip; ut=uz+TGT*pip
        lv[0]=1.0; ld[0]=dr; lp[0]=entry; nl=1; lu=ll=-1; cl2=False; ep=entry; it=False; im=False
        i+=1
        while i<n and not cl2:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            for p2 in range(2):
                if cl2: break
                if (bull and p2==0) or (not bull and p2==1):
                    if hi>=ut: ep=ut; cl2=True; it=True; break
                    if hi>=uz and lu!=i:
                        lu=i; v=bv(nl,ut)
                        if v>0:
                            if nl>=ml: ep=cl; cl2=True; im=True; break
                            lv[nl]=v; ld[nl]=1.0; lp[nl]=uz; nl+=1
                else:
                    if lo<=lt: ep=lt; cl2=True; it=True; break
                    if lo<=lz and ll!=i:
                        ll=i; v=bv(nl,lt)
                        if v>0:
                            if nl>=ml: ep=cl; cl2=True; im=True; break
                            lv[nl]=v; ld[nl]=-1.0; lp[nl]=lz; nl+=1
            if not cl2: i+=1
        tp+=nb(nl,ep); nc+=1; sl+=nl
        if it: nt+=1
        if im: nm+=1
        if not cl2: break
    return tp,nc,nt,nm,sl

def build_sr(m5_per_tf, h, l):
    n_oos=len(h); tf_hi=[]; tf_lo=[]
    for s in range(0,n_oos,m5_per_tf):
        e=min(s+m5_per_tf,n_oos); tf_hi.append(float(np.max(h[s:e]))); tf_lo.append(float(np.min(l[s:e])))
    tf_hi=np.array(tf_hi); tf_lo=np.array(tf_lo); n_tf=len(tf_hi)
    def ta(c,idx,t,v):
        if c and c[-1][1]==t:
            if t=='H' and v>c[-1][2]: c[-1]=[idx,t,v]
            elif t=='L' and v<c[-1][2]: c[-1]=[idx,t,v]
        else: c.append([idx,t,v])
    def s3(raw):
        sig=[]; lh=ll=float('nan'); glh=glh2=False
        for idx,t,v in raw:
            if t=='H':
                if math.isnan(lh) or v>lh or glh: sig.append([idx,t,v]); lh=v; glh=False; glh2=True
            else:
                if math.isnan(ll) or v<ll or glh2: sig.append([idx,t,v]); ll=v; glh2=False; glh=True
        return sig
    conf=[]; ah=np.full(n_tf,np.nan); al=np.full(n_tf,np.nan)
    for i in range(1,n_tf):
        if i>=2:
            if tf_hi[i-1]>tf_hi[i-2] and tf_hi[i-1]>tf_hi[i]: ta(conf,i-1,'H',tf_hi[i-1])
            if tf_lo[i-1]<tf_lo[i-2] and tf_lo[i-1]<tf_lo[i]: ta(conf,i-1,'L',tf_lo[i-1])
            conf=s3(conf)
        ch=cl=float('nan')
        for _,t,v in reversed(conf):
            if t=='H' and math.isnan(ch): ch=v
            if t=='L' and math.isnan(cl): cl=v
            if not math.isnan(ch) and not math.isnan(cl): break
        ah[i]=ch; al[i]=cl
    ends=list(range(m5_per_tf-1,n_oos,m5_per_tf))
    if len(ends)<n_tf: ends.append(n_oos-1)
    ahm=np.full(n_oos,np.nan); alm=np.full(n_oos,np.nan)
    for ti in range(n_tf):
        em=ends[ti]; nxt=ends[ti+1] if ti+1<n_tf else n_oos
        ahm[em:nxt]=ah[ti]; alm[em:nxt]=al[ti]
    for i in range(1,n_oos):
        if math.isnan(ahm[i]): ahm[i]=ahm[i-1]
        if math.isnan(alm[i]): alm[i]=alm[i-1]
    return ahm,alm

# JIT warmup
_sim(np.ones(100),np.ones(100),np.ones(100)*1.01,np.ones(100)*0.99,
     np.ones(100)*1.005,np.ones(100)*0.995,0.25,0.01,1.4,1.25,10)
rng=np.random.RandomState(42)
_base(np.ones(100),np.ones(100),np.ones(100)*1.01,np.ones(100)*0.99,
      rng.choice(np.array([-1.0,1.0]),100),0.01,1.4,1.25,10)
print("JIT compiled", flush=True)

agg_base=0.0; agg_h1=0.0; agg_h4=0.0
print(f"\n{'Pair':<10} | {'Base$@1ku':>10} | {'H1-25$@1ku':>11} {'vs%':>5} | {'H4-25$@1ku':>11} {'vs%':>5}")
print("─"*68)
for pair in PAIRS:
    pip=PIP_MAP[pair]; pu=PIP_USD_MAP[pair]
    df=pd.read_parquet(f'{DATA_DIR}/{pair}_M5.parquet').sort_index()
    df.columns=[c.lower() for c in df.columns]
    n=len(df); oos=df.iloc[int(n*0.70):].reset_index(drop=True)
    oa=oos['open'].values.astype(np.float64); ha=oos['high'].values.astype(np.float64)
    la=oos['low'].values.astype(np.float64); ca=oos['close'].values.astype(np.float64)
    noos=len(ca)
    # Build S/R
    h1h,h1l=build_sr(12,ha,la)
    h4h,h4l=build_sr(48,ha,la)
    # Baseline
    rng2=np.random.RandomState(42)
    dirs=rng2.choice(np.array([-1.0,1.0]),noos)
    btp,_,_,_,_=_base(ca,oa,ha,la,dirs,pip,SPREAD,PF,MAX_LEGS)
    busd=btp*pu*UNITS
    # H1-25
    h1tp,_,_,_,_=_sim(ca,oa,ha,la,h1h,h1l,0.25,pip,SPREAD,PF,MAX_LEGS)
    h1usd=h1tp*pu*UNITS
    # H4-25
    h4tp,_,_,_,_=_sim(ca,oa,ha,la,h4h,h4l,0.25,pip,SPREAD,PF,MAX_LEGS)
    h4usd=h4tp*pu*UNITS
    vs1=(h1usd-busd)/max(abs(busd),1)*100; vs4=(h4usd-busd)/max(abs(busd),1)*100
    f1='🟢' if vs1>50 else ('🟡' if vs1>0 else '🔴')
    f4='🟢' if vs4>50 else ('🟡' if vs4>0 else '🔴')
    print(f"{pair:<10} | {busd:>+10,.0f} | {h1usd:>+11,.0f} {vs1:>+4.0f}% {f1} | {h4usd:>+11,.0f} {vs4:>+4.0f}% {f4}", flush=True)
    agg_base+=busd; agg_h1+=h1usd; agg_h4+=h4usd
print("─"*68)
vs1a=(agg_h1-agg_base)/max(abs(agg_base),1)*100; vs4a=(agg_h4-agg_base)/max(abs(agg_base),1)*100
print(f"{'AGGREGATE':<10} | {agg_base:>+10,.0f} | {agg_h1:>+11,.0f} {vs1a:>+4.0f}%   | {agg_h4:>+11,.0f} {vs4a:>+4.0f}%")
