"""
s5_exit_sim.py — does S5 (5-second) EXIT monitoring recover the edge the 60s/bar-close service loses?
Entry stays bar-based (signal on the TF close, fill at next TF-bar open). EXITS are checked at S5
resolution: walk the 5-second bars during the hold and exit at the S5 close that first breaches the
opposite band (causal: last-completed TF band) / extension-peak stop / time cap. Real per-bar spread.
Compare three cadences:  bar-close (multi-bar only)  vs  S5-cadence  vs  intrabar-ideal (the +43.7 bt).
Uses the 1.5y S5 BA data. One pair at a time.
"""
import duckdb, numpy as np, pandas as pd, gc
SMA=9;K=1.0
PAIRS={"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
       "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}
# tf: (pandas rule, S5-per-TF-bar, time_cap_TFbars, meat)
TF=[("5min",60,288,4.0),("15min",180,96,6.0),("1h",720,24,6.0),("4h",2880,12,10.0)]

def sim_pair(df, pip, sp, rule, s5pb, tcap, meat):
    # TF bars + bands + entries
    d=df.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    o=d.open.values;h=d.high.values;l=d.low.values;c=d.close.values; tts=d.index.values
    n=len(c); cl=c; basis=np.full(n,np.nan);sd=np.full(n,np.nan)
    for i in range(SMA-1,n): w=cl[i-SMA+1:i+1];basis[i]=w.mean();sd[i]=w.std()
    up=basis+K*sd; lo=basis-K*sd
    # S5 arrays + index of which TF bar each S5 bar belongs to
    s5=df; s5o=s5.open.values;s5h=s5.high.values;s5l=s5.low.values;s5c=s5.close.values; s5ts=s5.index.values
    tf_of_s5=np.searchsorted(tts, s5ts, side="right")-1   # last completed TF bar at each S5 ts (causal-ish)
    pos=0;ext=0;peak=0; entries=[]
    for i in range(SMA,n-1):
        if np.isnan(basis[i]): continue
        uo=l[i]>up[i];do=h[i]<lo[i]
        if uo: peak=h[i] if ext!=1 else max(peak,h[i]);ext=1
        elif do: peak=l[i] if ext!=-1 else min(peak,l[i]);ext=-1
        if pos==0:
            if l[i-1]>up[i-1] and l[i]<=up[i] and 0.5*(c[i]-basis[i])/pip-sp>=meat:
                entries.append((i+1,-1,peak)); pos=-1
            elif h[i-1]<lo[i-1] and h[i]>=lo[i] and 0.5*(basis[i]-c[i])/pip-sp>=meat:
                entries.append((i+1,1,peak)); pos=1
        # release pos at the TF bar where a bar-close exit would occur (so entries don't overlap)
        if pos!=0:
            if (pos==-1 and (h[i]>peak or l[i]<=lo[i])) or (pos==1 and (l[i]<peak or h[i]>=up[i])): pos=0
    # walk S5 for each entry
    out_s5=[]
    s5_start=np.searchsorted(s5ts, tts)   # first S5 idx of each TF bar
    for (tfi,dr,pk) in entries:
        if tfi>=n: continue
        j0=s5_start[tfi] if tfi<len(s5_start) else None
        if j0 is None or j0>=len(s5c): continue
        ent=s5o[j0]; pnl=None
        jend=min(j0+tcap*s5pb, len(s5c)-1)
        for j in range(j0, jend):
            cur_tf=tf_of_s5[j]
            if cur_tf<0: continue
            blo=lo[cur_tf]; bup=up[cur_tf]
            if dr==-1:
                if s5h[j]>pk: pnl=(ent-pk)/pip-sp; break          # ext-peak stop
                if s5l[j]<=blo: pnl=(ent-s5c[j])/pip-sp; break    # opp band (exit at S5 close = 5s fill)
            else:
                if s5l[j]<pk: pnl=(pk-ent)/pip-sp; break
                if s5h[j]>=bup: pnl=(s5c[j]-ent)/pip-sp; break
        if pnl is None: pnl=(ent-s5c[jend])/pip-sp if dr==-1 else (s5c[jend]-ent)/pip-sp
        out_s5.append(pnl)
    return out_s5, (s5ts[-1]-s5ts[0])/np.timedelta64(1,'D')

def main():
    con=duckdb.connect(); med={}
    for p,pip in PAIRS.items():
        s=con.execute(f"SELECT (ask_c-bid_c) s FROM 'data/s5_ohlc/{p}_S5_BA.parquet'").df(); med[p]=float(np.nanmedian(s.s.values)/pip)
    print("S5-cadence EXIT simulation (entry bar-based, exits monitored every 5s), real spread, 1.5y BA")
    print("="*70)
    print(f"  {'TF':>5} {'trades':>7} {'S5-cadence pips/day':>20} {'p/trade':>9} {'WR':>5}")
    for rule,s5pb,tcap,meat in TF:
        allp=[]; span=0
        for p,pip in PAIRS.items():
            df=con.execute(f"SELECT timestamp,open,high,low,close FROM 'data/s5_ohlc/{p}_S5_BA.parquet' ORDER BY timestamp").df()
            df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
            ps,sp_days=sim_pair(df,pip,med[p],rule,s5pb,tcap,meat); allp+=ps; span=max(span,sp_days); del df; gc.collect()
        a=np.array(allp)
        print(f"  {rule:>5} {len(a):>7} {a.sum()/span:>+19.1f} {a.mean():>+9.2f} {100*(a>0).mean():>4.0f}%",flush=True)

if __name__=="__main__": main()
