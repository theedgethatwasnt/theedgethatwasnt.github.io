#!/usr/bin/env python3
"""
Multi-TF + ASI swing S/R sweep — 12 pairs.
Tests H1, H2, H4, ASI-swing, H1+H4 consensus vs baseline fixed ZW=56 random.
All directional (LONG at support, SHORT at resistance), tgt=0.25×ZW and 0.50×ZW.

Usage:
  python3 multitf_12pair_sweep.py
  python3 multitf_12pair_sweep.py --pairs EUR_JPY GBP_JPY USD_JPY
"""
import os, math, argparse
import numpy as np
import pandas as pd

_default_data = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'm5_ohlc')
DATA_DIR = os.environ.get('ZR_DATA_DIR', _default_data)

ALL_PAIRS = [
    'AUD_JPY','AUD_USD','CAD_JPY','CHF_JPY',
    'EUR_GBP','EUR_JPY','EUR_USD','GBP_JPY',
    'GBP_USD','NZD_JPY','NZD_USD','USD_JPY',
]
PIP_USD_MAP = {
    'AUD_JPY':0.000067,'AUD_USD':0.000100,'CAD_JPY':0.000069,
    'CHF_JPY':0.000107,'EUR_GBP':0.000126,'EUR_JPY':0.000064,
    'EUR_USD':0.000100,'GBP_JPY':0.000091,'GBP_USD':0.000100,
    'NZD_JPY':0.000061,'NZD_USD':0.000100,'USD_JPY':0.000064,
}
PIP_MAP = {p: 0.01 if 'JPY' in p else 0.0001 for p in ALL_PAIRS}
SPREAD_MAP = {
    'AUD_JPY':2.1,'AUD_USD':1.3,'CAD_JPY':2.3,'CHF_JPY':3.5,
    'EUR_GBP':1.4,'EUR_JPY':2.3,'EUR_USD':1.6,'GBP_JPY':3.3,
    'GBP_USD':1.9,'NZD_JPY':2.7,'NZD_USD':1.5,'USD_JPY':1.7,
}
UNITS    = 1_000
MAX_LEGS = 10
PF       = 1.25


# ── TopsBots helpers ──────────────────────────────────────────────────────────
def _try_add(confirmed, item):
    idx, t, v = item
    if confirmed and confirmed[-1][1] == t:
        if t == 'H' and v > confirmed[-1][2]: confirmed[-1] = [idx, t, v]
        elif t == 'L' and v < confirmed[-1][2]: confirmed[-1] = [idx, t, v]
    else:
        confirmed.append([idx, t, v])

def _rerun_stage3(raw):
    sig=[]; lh=ll=float('nan'); glh=glh2=False
    for (idx,t,v) in raw:
        if t=='H':
            if math.isnan(lh) or v>lh or glh:
                sig.append([idx,t,v]); lh=v; glh=False; glh2=True
        else:
            if math.isnan(ll) or v<ll or glh2:
                sig.append([idx,t,v]); ll=v; glh2=False; glh=True
    return lh,ll,glh,glh2,sig

def _get_act(confirmed):
    cur_h=cur_l=float('nan')
    for _,t,v in reversed(confirmed):
        if t=='H' and math.isnan(cur_h): cur_h=v
        if t=='L' and math.isnan(cur_l): cur_l=v
        if not math.isnan(cur_h) and not math.isnan(cur_l): break
    return cur_h, cur_l


def build_ohlc_sr(high_a, low_a, n_oos, m5_per_tf):
    tf_hi=[]; tf_lo=[]; tf_end_m5=[]
    for start in range(0, n_oos, m5_per_tf):
        end=min(start+m5_per_tf, n_oos)
        tf_hi.append(float(np.max(high_a[start:end])))
        tf_lo.append(float(np.min(low_a[start:end])))
        tf_end_m5.append(end-1)
    tf_hi=np.array(tf_hi); tf_lo=np.array(tf_lo); n_tf=len(tf_hi)

    confirmed=[]; act_h_tf=np.full(n_tf,float('nan')); act_l_tf=np.full(n_tf,float('nan'))
    for i in range(1, n_tf):
        if i>=2:
            if tf_hi[i-1]>tf_hi[i-2] and tf_hi[i-1]>tf_hi[i]:
                _try_add(confirmed,(i-1,'H',tf_hi[i-1]))
            if tf_lo[i-1]<tf_lo[i-2] and tf_lo[i-1]<tf_lo[i]:
                _try_add(confirmed,(i-1,'L',tf_lo[i-1]))
            _,_,_,_,confirmed=_rerun_stage3(confirmed)
        act_h_tf[i],act_l_tf[i]=_get_act(confirmed)

    act_h_m5=np.full(n_oos,float('nan')); act_l_m5=np.full(n_oos,float('nan'))
    for ti,end_m5 in enumerate(tf_end_m5):
        nxt=tf_end_m5[ti+1] if ti+1<n_tf else n_oos
        act_h_m5[end_m5:nxt]=act_h_tf[ti]; act_l_m5[end_m5:nxt]=act_l_tf[ti]
    for i in range(1,n_oos):
        if math.isnan(act_h_m5[i]): act_h_m5[i]=act_h_m5[i-1]
        if math.isnan(act_l_m5[i]): act_l_m5[i]=act_l_m5[i-1]

    valid=np.sum(~np.isnan(act_h_m5)&~np.isnan(act_l_m5))
    zw=(act_h_m5-act_l_m5)[~np.isnan(act_h_m5)]/PIP_MAP.get('EUR_JPY',0.01)  # placeholder, overridden per-pair
    return act_h_m5, act_l_m5, valid


def build_asi_sr(open_a, high_a, low_a, close_a, n_oos, atr_period=14, atr_mult=3.0):
    EPSILON=1e-10
    tr_arr=np.maximum(high_a[1:]-low_a[1:],
           np.maximum(np.abs(high_a[1:]-close_a[:-1]),np.abs(low_a[1:]-close_a[:-1])))
    atr=np.zeros(n_oos); atr[0]=high_a[0]-low_a[0]
    for i in range(1,n_oos):
        if i<atr_period: atr[i]=atr[i-1]+(tr_arr[i-1]-atr[i-1])/(i+1)
        else: atr[i]=(atr[i-1]*(atr_period-1)+tr_arr[i-1])/atr_period

    C2,O2,H2,L2=close_a[1:],open_a[1:],high_a[1:],low_a[1:]
    C1,O1=close_a[:-1],open_a[:-1]
    N=(C2-C1)+0.5*(C2-O2)+0.25*(C1-O1)
    t1=np.abs(H2-C1)-0.5*np.abs(L2-C1)+0.25*np.abs(C1-O1)
    t2=np.abs(L2-C1)-0.5*np.abs(H2-C1)+0.25*np.abs(C1-O1)
    t3=(H2-L2)+0.25*np.abs(C1-O1)
    R=np.maximum(np.maximum(t1,t2),np.maximum(t3,EPSILON))
    K=np.maximum(np.abs(H2-C1),np.abs(L2-C1))
    SI=50.0*(N/R)*(K/np.maximum(atr_mult*atr[1:],EPSILON))
    asi_vals=np.zeros(n_oos); asi_vals[1:]=np.cumsum(SI)

    confirmed=[]; act_h_m5=np.full(n_oos,float('nan')); act_l_m5=np.full(n_oos,float('nan'))
    prev_len=0
    for i in range(2,n_oos):
        if asi_vals[i-1]>asi_vals[i-2] and asi_vals[i-1]>asi_vals[i]:
            _try_add(confirmed,(i-1,'H',close_a[i-1]))
        if asi_vals[i-1]<asi_vals[i-2] and asi_vals[i-1]<asi_vals[i]:
            _try_add(confirmed,(i-1,'L',close_a[i-1]))
        if len(confirmed)!=prev_len:
            _,_,_,_,confirmed=_rerun_stage3(confirmed); prev_len=len(confirmed)
        act_h_m5[i],act_l_m5[i]=_get_act(confirmed)

    valid=np.sum(~np.isnan(act_h_m5)&~np.isnan(act_l_m5)&(act_h_m5>act_l_m5))
    return act_h_m5, act_l_m5, valid


def net_basket(legs, price, pip, spread):
    gross=sum(l['vol']*l['dir']*(price-l['price'])/pip for l in legs)
    cost=sum(l['vol'] for l in legs)*spread
    return gross-cost


def simulate(open_a,high_a,low_a,close_a,n_oos,pip,spread,act_h,act_l,tgt_frac=0.25):
    def bvol(legs,target,tgt_pips):
        net=net_basket(legs,target,pip,spread)
        if net>=0: return 0.0
        return max(1.0,math.ceil(-net/tgt_pips*PF))

    cycles=[]; i=0
    while i<n_oos:
        uh=act_h[i]; ul=act_l[i]
        if math.isnan(uh) or math.isnan(ul) or uh<=ul: i+=1; continue
        zw_pips=(uh-ul)/pip; tgt_pips=zw_pips*tgt_frac; tgt_b=tgt_pips*pip
        entry=close_a[i]
        if   entry<=ul: direction=1
        elif entry>=uh: direction=-1
        else: i+=1; continue
        upper_target=uh+tgt_b; lower_target=ul-tgt_b
        legs=[{'dir':direction,'price':entry,'vol':1.0}]
        lc=lcb=None; closed=False; er='eod'; ep=entry; i+=1
        while i<n_oos and not closed:
            hi=high_a[i]; lo=low_a[i]; cl=close_a[i]; bull=cl>=open_a[i]
            seq=[(hi,True),(lo,False)] if bull else [(lo,False),(hi,True)]
            for extreme,is_high in seq:
                if closed: break
                if is_high and hi>=upper_target: ep,er=upper_target,'target'; closed=True; break
                if not is_high and lo<=lower_target: ep,er=lower_target,'target'; closed=True; break
                if is_high and hi>=uh:
                    if not(lc=='upper' and lcb==i):
                        lc,lcb='upper',i; vol=bvol(legs,upper_target,tgt_pips)
                        if vol>0:
                            if len(legs)>=MAX_LEGS: ep,er=cl,'max_legs'; closed=True; break
                            legs.append({'dir':1,'price':uh,'vol':vol})
                if not is_high and lo<=ul:
                    if not(lc=='lower' and lcb==i):
                        lc,lcb='lower',i; vol=bvol(legs,lower_target,tgt_pips)
                        if vol>0:
                            if len(legs)>=MAX_LEGS: ep,er=cl,'max_legs'; closed=True; break
                            legs.append({'dir':-1,'price':ul,'vol':vol})
            if not closed: i+=1
        net=net_basket(legs,ep,pip,spread)
        cycles.append({'net_pips':net,'exit_reason':er,'n_legs':len(legs),'zw_pips':zw_pips})
        if not closed: break
    return cycles


def simulate_consensus(open_a,high_a,low_a,close_a,n_oos,pip,spread,
                        act_h1,act_l1,act_h2,act_l2,tgt_frac=0.25,use_wider=False):
    def bvol(legs,target,tgt_pips):
        net=net_basket(legs,target,pip,spread)
        if net>=0: return 0.0
        return max(1.0,math.ceil(-net/tgt_pips*PF))

    cycles=[]; i=0
    while i<n_oos:
        uh1=act_h1[i];ul1=act_l1[i];uh2=act_h2[i];ul2=act_l2[i]
        if any(math.isnan(x) for x in [uh1,ul1,uh2,ul2]): i+=1; continue
        if uh1<=ul1 or uh2<=ul2: i+=1; continue
        entry=close_a[i]
        if   entry<=ul1 and entry<=ul2: direction=1
        elif entry>=uh1 and entry>=uh2: direction=-1
        else: i+=1; continue
        uh=uh2 if use_wider else uh1; ul=ul2 if use_wider else ul1
        if uh<=ul: i+=1; continue
        zw_pips=(uh-ul)/pip; tgt_pips=zw_pips*tgt_frac; tgt_b=tgt_pips*pip
        upper_target=uh+tgt_b; lower_target=ul-tgt_b
        legs=[{'dir':direction,'price':entry,'vol':1.0}]
        lc=lcb=None; closed=False; er='eod'; ep=entry; i+=1
        while i<n_oos and not closed:
            hi=high_a[i];lo=low_a[i];cl=close_a[i];bull=cl>=open_a[i]
            seq=[(hi,True),(lo,False)] if bull else [(lo,False),(hi,True)]
            for extreme,is_high in seq:
                if closed: break
                if is_high and hi>=upper_target: ep,er=upper_target,'target'; closed=True; break
                if not is_high and lo<=lower_target: ep,er=lower_target,'target'; closed=True; break
                if is_high and hi>=uh:
                    if not(lc=='upper' and lcb==i):
                        lc,lcb='upper',i; vol=bvol(legs,upper_target,tgt_pips)
                        if vol>0:
                            if len(legs)>=MAX_LEGS: ep,er=cl,'max_legs'; closed=True; break
                            legs.append({'dir':1,'price':uh,'vol':vol})
                if not is_high and lo<=ul:
                    if not(lc=='lower' and lcb==i):
                        lc,lcb='lower',i; vol=bvol(legs,lower_target,tgt_pips)
                        if vol>0:
                            if len(legs)>=MAX_LEGS: ep,er=cl,'max_legs'; closed=True; break
                            legs.append({'dir':-1,'price':ul,'vol':vol})
            if not closed: i+=1
        net=net_basket(legs,ep,pip,spread)
        cycles.append({'net_pips':net,'exit_reason':er,'n_legs':len(legs),'zw_pips':zw_pips})
        if not closed: break
    return cycles


def simulate_baseline(open_a,high_a,low_a,close_a,n_oos,pip,spread):
    ZW=56; TGT=28; rng=np.random.RandomState(42)
    def bvol_b(legs,target):
        net=net_basket(legs,target,pip,spread)
        if net>=0: return 0.0
        return max(1.0,math.ceil(-net/TGT*PF))
    cycles=[]; i=0
    while i<n_oos:
        entry=close_a[i]; direction=int(rng.choice([-1,1]))
        if direction==1: uz=entry;lz=entry-ZW*pip;ut=entry+TGT*pip;lt=lz-TGT*pip
        else:            lz=entry;uz=entry+ZW*pip;lt=entry-TGT*pip;ut=uz+TGT*pip
        legs=[{'dir':direction,'price':entry,'vol':1.0}]
        lc=lcb=None; closed=False; er='eod'; ep=entry; i+=1
        while i<n_oos and not closed:
            hi=high_a[i];lo=low_a[i];cl=close_a[i];bull=cl>=open_a[i]
            seq=[(hi,True),(lo,False)] if bull else [(lo,False),(hi,True)]
            for ex,ih in seq:
                if closed: break
                if ih and hi>=ut: ep,er=ut,'target'; closed=True; break
                if not ih and lo<=lt: ep,er=lt,'target'; closed=True; break
                if ih and hi>=uz:
                    if not(lc=='upper' and lcb==i):
                        lc,lcb='upper',i; vol=bvol_b(legs,ut)
                        if vol>0:
                            if len(legs)>=MAX_LEGS: ep,er=cl,'max_legs'; closed=True; break
                            legs.append({'dir':1,'price':uz,'vol':vol})
                if not ih and lo<=lz:
                    if not(lc=='lower' and lcb==i):
                        lc,lcb='lower',i; vol=bvol_b(legs,lt)
                        if vol>0:
                            if len(legs)>=MAX_LEGS: ep,er=cl,'max_legs'; closed=True; break
                            legs.append({'dir':-1,'price':lz,'vol':vol})
            if not closed: i+=1
        net=net_basket(legs,ep,pip,spread)
        cycles.append({'net_pips':net,'exit_reason':er,'n_legs':len(legs)})
        if not closed: break
    return cycles


def summarise(cycles, pip_usd, units=UNITS):
    if not cycles: return None
    df_c=pd.DataFrame(cycles); tp=df_c['net_pips'].sum()
    return {'total_pips':tp,'total_usd':tp*pip_usd*units,'n_cyc':len(df_c),
            'pct_tgt':(df_c['exit_reason']=='target').sum()/len(df_c)*100,
            'n_ml':(df_c['exit_reason']=='max_legs').sum(),
            'avg_legs':df_c['n_legs'].mean(),
            'avg_zw':df_c['zw_pips'].mean() if 'zw_pips' in df_c else 56.0}


def run_pair(pair):
    pip=PIP_MAP[pair]; pip_usd=PIP_USD_MAP[pair]; spread=SPREAD_MAP[pair]
    path=f'{DATA_DIR}/{pair}_M5.parquet'
    if not os.path.exists(path): print(f"  {pair}: MISSING DATA"); return None

    df=pd.read_parquet(path).sort_index(); df.columns=[c.lower() for c in df.columns]
    n=len(df); n_oos_start=int(n*0.70)
    df_oos=df.iloc[n_oos_start:].reset_index(drop=True)
    open_a=df_oos['open'].values.astype(np.float64)
    high_a=df_oos['high'].values.astype(np.float64)
    low_a=df_oos['low'].values.astype(np.float64)
    close_a=df_oos['close'].values.astype(np.float64)
    n_oos=len(close_a)

    h1_h,h1_l,_=build_ohlc_sr(high_a,low_a,n_oos,12)
    h2_h,h2_l,_=build_ohlc_sr(high_a,low_a,n_oos,24)
    h4_h,h4_l,_=build_ohlc_sr(high_a,low_a,n_oos,48)
    asi_h,asi_l,_=build_asi_sr(open_a,high_a,low_a,close_a,n_oos)

    args=(open_a,high_a,low_a,close_a,n_oos,pip,spread)
    base_cyc   =simulate_baseline(*args)
    h1_25_cyc  =simulate(*args,h1_h,h1_l,0.25)
    h1_50_cyc  =simulate(*args,h1_h,h1_l,0.50)
    h2_25_cyc  =simulate(*args,h2_h,h2_l,0.25)
    h2_50_cyc  =simulate(*args,h2_h,h2_l,0.50)
    h4_25_cyc  =simulate(*args,h4_h,h4_l,0.25)
    h4_50_cyc  =simulate(*args,h4_h,h4_l,0.50)
    c_h1h4_n   =simulate_consensus(*args,h1_h,h1_l,h4_h,h4_l,0.25,False)
    c_h1h4_w   =simulate_consensus(*args,h1_h,h1_l,h4_h,h4_l,0.25,True)
    c_h1h2_n   =simulate_consensus(*args,h1_h,h1_l,h2_h,h2_l,0.25,False)
    asi_25_cyc =simulate(*args,asi_h,asi_l,0.25)
    asi_50_cyc =simulate(*args,asi_h,asi_l,0.50)

    configs=[
        ("BASELINE",    base_cyc),
        ("H1-0.25",     h1_25_cyc),
        ("H1-0.50",     h1_50_cyc),
        ("H2-0.25",     h2_25_cyc),
        ("H2-0.50",     h2_50_cyc),
        ("H4-0.25",     h4_25_cyc),
        ("H4-0.50",     h4_50_cyc),
        ("H1+H4-narrow",c_h1h4_n),
        ("H1+H4-wide",  c_h1h4_w),
        ("H1+H2-narrow",c_h1h2_n),
        ("ASI-0.25",    asi_25_cyc),
        ("ASI-0.50",    asi_50_cyc),
    ]
    results={}
    for label,cyc in configs:
        s=summarise(cyc,pip_usd)
        results[label]=s
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pairs',nargs='+',default=ALL_PAIRS)
    args=parser.parse_args()
    pairs=args.pairs

    CONFIGS=["BASELINE","H1-0.25","H1-0.50","H2-0.25","H2-0.50",
             "H4-0.25","H4-0.50","H1+H4-narrow","H1+H4-wide","H1+H2-narrow",
             "ASI-0.25","ASI-0.50"]

    all_results={}
    for pair in pairs:
        print(f"Running {pair}...", flush=True)
        r=run_pair(pair)
        if r: all_results[pair]=r

    # ── Per-pair summary table ────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  Multi-TF + ASI Sweep — Per-Pair OOS USD (1,000 units)  PF={PF}")
    print(f"{'='*110}")
    hdr="  Pair       |"+" ".join(f"{c:>12}" for c in CONFIGS)
    print(hdr); print("─"*len(hdr))

    agg={c:0.0 for c in CONFIGS}
    agg_pips={c:0.0 for c in CONFIGS}
    for pair,results in all_results.items():
        row=f"  {pair:<10} |"
        for c in CONFIGS:
            s=results.get(c)
            usd=s['total_usd'] if s else 0.0
            agg[c]+=usd
            agg_pips[c]+=(s['total_pips'] if s else 0.0)
            row+=f"  {usd:>+9,.0f} "
        print(row)

    print("─"*len(hdr))
    agg_row="  AGGREGATE  |"+" ".join(f"  {agg[c]:>+9,.0f} " for c in CONFIGS)
    print(agg_row)

    # ── Best config per pair ──────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("  Best config per pair (by OOS USD, excluding BASELINE):")
    print(f"{'='*110}")
    non_base=[c for c in CONFIGS if c!="BASELINE"]
    base_ref=sum(all_results[p]['BASELINE']['total_usd'] for p in all_results if all_results[p]['BASELINE'])
    for pair,results in all_results.items():
        best_c=max(non_base,key=lambda c:results.get(c,{}).get('total_usd',-1e9) if results.get(c) else -1e9)
        s=results.get(best_c); base_s=results.get('BASELINE')
        if s and base_s:
            vs_base=(s['total_usd']-base_s['total_usd'])/max(abs(base_s['total_usd']),1)*100
            flag='🟢' if s['total_usd']>0 else '🔴'
            print(f"  {flag} {pair:<10}: {best_c:<15} USD={s['total_usd']:>+9,.0f}  "
                  f"pips={s['total_pips']:>+8,.0f}  cyc={s['n_cyc']:>5}  "
                  f"pct_tgt={s['pct_tgt']:.1f}%  vs_base={vs_base:>+.1f}%")

    # ── Aggregate ranking ─────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("  Aggregate ranking across all pairs:")
    print(f"{'='*110}")
    ranked=sorted(non_base,key=lambda c:agg[c],reverse=True)
    for c in ranked:
        base_usd=agg['BASELINE']
        vs_base=(agg[c]-base_usd)/max(abs(base_usd),1)*100
        flag='🟢' if agg[c]>0 else ('🟡' if agg[c]>base_usd else '🔴')
        print(f"  {flag} {c:<16}: agg_USD={agg[c]:>+12,.0f}  "
              f"agg_pips={agg_pips[c]:>+10,.0f}  vs_baseline={vs_base:>+.1f}%")

    print(f"\n  Baseline aggregate: USD={agg['BASELINE']:>+12,.0f}  pips={agg_pips['BASELINE']:>+10,.0f}")
    print(f"{'='*110}")

if __name__ == '__main__':
    main()
