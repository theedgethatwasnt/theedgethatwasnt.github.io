#!/usr/bin/env python3
"""Render the Stage-1 momentum×efficiency grids as visual heatmaps."""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT/"data"/"m5_ba"; RES = Path(__file__).parent/"results"
PAIRS = ["GBP_JPY","USD_JPY","EUR_JPY","GBP_USD","AUD_JPY","EUR_USD",
         "EUR_GBP","AUD_USD","NZD_JPY","CHF_JPY","NZD_USD","CAD_JPY"]
JPY={"GBP_JPY","USD_JPY","EUR_JPY","AUD_JPY","NZD_JPY","CHF_JPY","CAD_JPY"}
def pip_sz(p): return 0.01 if p in JPY else 0.0001
NQ=5; WINDOWS=[15,30,60,120]

def grid_for(W):
    bw=W//5; per=[]
    for pair in PAIRS:
        df=pd.read_parquet(DATA/f"{pair}_M5_BA.parquet").set_index("timestamp").sort_index()
        df=df.astype({c:"float64" for c in df.select_dtypes("float32").columns})
        pip=pip_sz(pair); c=df["close"]
        net=(c-c.shift(bw))/pip; mom=net/W
        path=(c.diff().abs()/pip).rolling(bw).sum()
        eff=net.abs()/path.replace(0,np.nan)
        spread=(df["ask_c"]-df["bid_c"])/pip
        fwd=(c.shift(-bw)-c)/pip
        cont=np.sign(mom)*fwd-spread
        d=pd.DataFrame({"am":mom.abs(),"eff":eff,"cont":cont}).dropna()
        d=d[d["am"]>0]
        if len(d)<2000: continue
        d["mq"]=pd.qcut(d["am"],NQ,labels=False,duplicates="drop")
        d["eq"]=pd.qcut(d["eff"],NQ,labels=False,duplicates="drop")
        g=d.groupby(["mq","eq"])["cont"].mean().unstack().reindex(index=range(NQ),columns=range(NQ))
        per.append(g.values)
    return np.nanmean(np.stack(per),axis=0)

grids={W:grid_for(W) for W in WINDOWS}
vmax=max(abs(np.nanmin(g)) for g in grids.values());
fig,axes=plt.subplots(2,2,figsize=(12,10))
for ax,W in zip(axes.flat,WINDOWS):
    g=grids[W]
    im=ax.imshow(g,cmap="RdYlGn",vmin=-vmax,vmax=vmax,origin="lower",aspect="auto")
    for i in range(NQ):
        for j in range(NQ):
            ax.text(j,i,f"{g[i,j]:+.1f}",ha="center",va="center",fontsize=9,
                    color="black")
    ax.set_title(f"W={W}min  (hold {W}min)\ncell = continuation pnl net spread (pips)",fontsize=10)
    ax.set_xlabel("efficiency quintile  (Q0 wiggly → Q4 clean)")
    ax.set_ylabel("|momentum| quintile  (Q0 slow → Q4 fast)")
    ax.set_xticks(range(NQ)); ax.set_yticks(range(NQ))
fig.suptitle("Stage 1: Momentum × Efficiency → next-move continuation (12-pair avg)\n"
             "RED = momentum REVERSES (fade pays) · GREEN = continues · all cells <0 ⇒ no continuation edge",
             fontsize=12)
fig.colorbar(im,ax=axes,shrink=0.6,label="pips (net of spread)")
out=RES/"stage1_heatmap.png"; fig.savefig(out,dpi=110,bbox_inches="tight")
print(f"saved {out}")
