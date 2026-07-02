"""
Does HIGH tick-volume in a bar predict DIRECTION?
Two questions, pooled across 12 pairs, net of spread, IS/OOS:
  (a) single-bar: high vol[t] -> does bar t+1 CONTINUE bar t?
  (b) multi-bar : high vol over last m bars -> do next k bars continue the last m?

Metric = CONTINUATION pnl = sign(prior move) * forward move, in pips, minus spread.
  >0  => volume predicts continuation (direction confirmation)
  <0  => volume predicts reversal
  ~0  => volume says nothing about direction (only magnitude)
Bucket by vrel = vol / trailing-mean. Compare HIGH-vol vs LOW-vol buckets.
TFs: M5 (native), H1, H4.
"""
from pathlib import Path
import numpy as np, pandas as pd, duckdb

DATA=Path("/path/to/projects/fx-core")/"data"/"m5_ohlc"
PAIRS={"USD_JPY":(0.01,2.1),"EUR_JPY":(0.01,2.5),"GBP_JPY":(0.01,4.0),"AUD_JPY":(0.01,2.3),
 "CAD_JPY":(0.01,2.6),"CHF_JPY":(0.01,3.0),"NZD_JPY":(0.01,3.0),"EUR_USD":(0.0001,1.7),
 "GBP_USD":(0.0001,2.4),"AUD_USD":(0.0001,1.6),"EUR_GBP":(0.0001,2.0),"NZD_USD":(0.0001,2.0)}
IS_FRAC=0.6; VW=20; RNG=np.random.default_rng(7)
TFS={"M5":None,"H1":"1h","H4":"4h"}
CASES=[("single (m1->k1)",1,1),("few->few (m3->k3)",3,3),("few->next (m3->k1)",3,1)]


def series(con,pair,rule):
    f=DATA/f"{pair}_M5.parquet"
    df=con.execute(f"SELECT timestamp,high,low,close,volume FROM '{f}' ORDER BY timestamp").df()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True); df=df.set_index("timestamp")
    if rule:
        df=df.resample(rule).agg({"high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    return df


def main():
    con=duckdb.connect()
    for tf,rule in TFS.items():
        print(f"\n================  TF = {tf}  ================")
        for label,m,k in CASES:
            # pool continuation pnl + vrel across pairs
            allc=[]; allv=[]; alli=[]
            for pair,(pip,sp) in PAIRS.items():
                df=series(con,pair,rule)
                c=df.close.values; v=df.volume.values; n=len(c)
                if n<200: continue
                vrel=v/pd.Series(v).rolling(VW).mean().shift(1).values
                is_cut=int(n*IS_FRAC)
                for t in range(VW+m, n-k):
                    if np.isnan(vrel[t]): continue
                    mom=c[t]-c[t-m]
                    if mom==0: continue
                    fwd=c[t+k]-c[t]
                    cont=np.sign(mom)*fwd/pip - sp        # continuation pnl, net spread
                    allc.append(cont); allv.append(vrel[t]); alli.append(t<is_cut)
            cont=np.array(allc); vr=np.array(allv); isf=np.array(alli)
            # quintile buckets on IS vrel
            qe=np.quantile(vr[isf],[0,.2,.4,.6,.8,1.0])
            qe=np.unique(np.round(qe,3))
            print(f"\n  {label}:  continuation pips by vol bucket (net spread)   [n={len(cont):,}]")
            print(f"    {'vol bucket':>14} | {'IS cont':>8} {'OOS cont':>9} {'n_oos':>7} {'OOS WR':>7}")
            for a,b in zip(qe[:-1],qe[1:]):
                msel=(vr>=a)&(vr<b) if b!=qe[-1] else (vr>=a)&(vr<=b)
                ci=cont[msel&isf]; co=cont[msel&~isf]
                if len(co)<50: continue
                print(f"    {a:5.2f}-{b:<6.2f} | {ci.mean():>+8.3f} {co.mean():>+9.3f} {len(co):>7,} {100*(co>0).mean():>6.1f}%")
            # HIGH vs LOW bucket OOS gap + bootstrap on HIGH bucket
            hi=(vr>=qe[-2]); lo=(vr<qe[1])
            hio=cont[hi&~isf]; loo=cont[lo&~isf]
            bh=np.array([RNG.choice(hio,len(hio),replace=True).mean() for _ in range(1500)])
            print(f"    => HIGH-vol OOS {hio.mean():+.3f}p (MC P(<=0)={(bh<=0).mean():.3f})  "
                  f"LOW-vol OOS {loo.mean():+.3f}p   HIGH-LOW gap {hio.mean()-loo.mean():+.3f}p")
    con.close()
    print("\nRead: HIGH-vol cont >0 & > LOW => volume confirms direction. ~0 => magnitude only. <0 => reversal.")


if __name__=="__main__":
    main()
