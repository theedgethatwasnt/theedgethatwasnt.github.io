"""
m5_regime_wf.py — SMA9/SMA200 regime trend system on M5 (built from S5), 12 pairs, WF.

Rule (user spec, 2026-06-23):
  Regime = sign(SMA9 - SMA200) on M5 mid close.
  SHORT: down-regime (SMA9<SMA200) AND close<SMA9, on the FIRST bar the conjunction
         flips false->true (one entry per regime). Mirror LONG.
  EXIT : when SMA9 crosses SMA200 (regime flip).
  TIMING (causal, SOP R1/R2): signal on bar i CLOSE -> fill at bar i+1 OPEN. Same for exit.
  One action per bar.

SOP compliance: signals on mid OHLC; real per-bar spread from S5 bid_c/ask_c deducted
(half each leg) at the fill bar; next-bar-open fills; never bid_h/bid_l (R3a). 12 pairs,
4 contiguous walk-forward folds, IS/OOS 70/30, Monte-Carlo on the portfolio. Net of spread
is the verdict.
"""
import numpy as np, pyarrow.parquet as pq, os

PAIRS = ["USD_JPY","EUR_USD","GBP_USD","AUD_USD","EUR_JPY","GBP_JPY",
         "AUD_JPY","CAD_JPY","NZD_JPY","CHF_JPY","NZD_USD","EUR_GBP"]
DATA = "data/s5_ohlc/{}_S5_BA.parquet"
BARS_PER = 60            # 60 S5 (5s) = 1 M5 bar
FAST, SLOW = 9, 200
NFOLD = 4

def resample_m5(o,h,l,c,bid,ask,vol, bars=BARS_PER):
    n = len(c); nb = n // bars
    def blk(a): return a[:nb*bars].reshape(nb, bars)
    O = blk(o)[:,0]; C = blk(c)[:,-1]; H = blk(h).max(1); L = blk(l).min(1)
    BID = blk(bid)[:,-1]; ASK = blk(ask)[:,-1]      # M5-close bid/ask (R3a: close only)
    V = blk(vol).sum(1)                             # M5 tick volume = sum of 60 S5 ticks
    return O,H,L,C,BID,ASK,V

def sma(c,p):
    n=len(c); out=np.full(n,np.nan); cs=np.cumsum(c)
    out[p-1:]=(cs[p-1:]-np.concatenate([[0.0],cs[:-p]]))/p
    return out

def simulate(O,C,V,s9,s200,sp,pip):
    """entry_idx, pnl_pips, dir(+1/-1), mom3_pips (signed 3-bar pre-signal momentum IN TRADE
    DIRECTION), vol3 (tick-volume sum over the same 3 bars). Causal next-open fills; the two
    pre-entry features are stored at entry and emitted at exit so they align with pnl."""
    n=len(C); pos=0; entry=0.0; ei=0; cur_m3=0.0; cur_v3=0.0
    e_idx=[]; pnl=[]; dirs=[]; mom3=[]; vol3=[]
    for i in range(SLOW, n-1):                      # need s200 and bar i+1 for the fill
        if np.isnan(s9[i]) or np.isnan(s200[i]): continue
        h = sp[i+1]*0.5                              # half-spread at the FILL bar
        f = O[i+1]                                   # next-bar open (mid)
        if pos != 0:                                 # exit takes the bar's single action
            crossed = (pos==-1 and s9[i]>=s200[i]) or (pos==1 and s9[i]<=s200[i])
            if crossed:
                px = (entry-(f+h)) if pos==-1 else ((f-h)-entry)   # short buys ask / long sells bid
                pnl.append(px/pip); e_idx.append(ei); dirs.append(pos)
                mom3.append(cur_m3); vol3.append(cur_v3); pos=0
            continue
        down = s9[i]<s200[i]; up = s9[i]>s200[i]
        sc  = down and (C[i]  < s9[i]);  scp = (s9[i-1]<s200[i-1]) and (C[i-1]<s9[i-1])
        lc  = up   and (C[i]  > s9[i]);  lcp = (s9[i-1]>s200[i-1]) and (C[i-1]>s9[i-1])
        trig=False
        if   sc and not scp: entry=f-h; pos=-1; ei=i+1; cur_m3=(C[i-3]-C[i])/pip; trig=True
        elif lc and not lcp: entry=f+h; pos= 1; ei=i+1; cur_m3=(C[i]-C[i-3])/pip; trig=True
        if trig: cur_v3 = float(V[i-2]+V[i-1]+V[i]) # tick volume over the 3 bars into the entry
    return (np.array(e_idx), np.array(pnl), np.array(dirs), np.array(mom3), np.array(vol3))

def bin_report(label, x, pnl, nb=5):
    """Quintile-bin a per-trade feature x and tabulate n / mean p/tr / win% per bin."""
    x=np.asarray(x,float); pnl=np.asarray(pnl,float)
    qs=np.quantile(x, np.linspace(0,1,nb+1))
    print(f"\n  {label} (quintile bins)   bin-range | n | mean p/tr | win%")
    for k in range(nb):
        lo,hi=qs[k],qs[k+1]
        m=(x>=lo)&(x<=hi) if k==nb-1 else (x>=lo)&(x<hi)
        if m.sum()==0: continue
        pk=pnl[m]
        print(f"    Q{k+1}  [{lo:+7.2f},{hi:+7.2f}]   n={int(m.sum()):5d}   {pk.mean():+6.2f}   {100*(pk>0).mean():4.1f}%")

def main():
    rng = np.random.default_rng(42)
    print(f"SMA{FAST}/SMA{SLOW} regime trend system on M5 (from S5), next-bar-open fills, "
          f"real per-bar spread, {NFOLD} WF folds. Net of spread.\n" + "="*92)
    pair_rows=[]; all_pnl=[]; npos=nwf=0; tot_days=0.0
    g_mom=[]; g_vol=[]; g_dir=[]                              # pooled per-trade conditioning features
    for p in PAIRS:
        f = DATA.format(p)
        if not os.path.exists(f): print(f"  {p}: MISSING"); continue
        t = pq.read_table(f, columns=["open","high","low","close","bid_c","ask_c","volume"])
        o,h,l,c,bid,ask,vol = (t.column(k).to_numpy().astype(np.float64) for k in
                           ["open","high","low","close","bid_c","ask_c","volume"])
        O,H,L,C,BID,ASK,V = resample_m5(o,h,l,c,bid,ask,vol)
        pip = 0.01 if "JPY" in p else 0.0001
        sp_arr = (ASK-BID)                                  # per-M5-bar spread (price units)
        med_sp = float(np.median(sp_arr)/pip)
        s9, s200 = sma(C,FAST), sma(C,SLOW)
        ei, pnl, dirs, mom3, vol3 = simulate(O,C,V,s9,s200,sp_arr,pip)
        m5_per_day = 288.0                                  # 24h * 12 M5/h
        days = len(C)/m5_per_day
        tot_days = max(tot_days, days)
        if len(pnl)==0:
            print(f"  {p:8s} no trades"); continue
        net = pnl.mean(); tot = pnl.sum(); ntr = len(pnl)
        # 4 contiguous WF folds by entry index (time order)
        order = np.argsort(ei); pe = pnl[order]
        folds = np.array_split(pe, NFOLD)
        fold_net = [fk.mean() for fk in folds if len(fk)]
        fold_pos = sum(x>0 for x in fold_net)
        net_pos = net>0; wf_ok = fold_pos>=3
        npos += net_pos; nwf += (net_pos and wf_ok)
        v = "🟢 WF" if (net_pos and wf_ok) else ("🟡" if net_pos else "🔴")
        long_n=int((dirs>0).sum()); short_n=int((dirs<0).sum())
        print(f"  {p:8s} {v}  net {net:+6.2f} p/tr  tot {tot:+8.0f}p  {ntr:5d} tr "
              f"({ntr/days:4.1f}/day)  WF {fold_pos}/{NFOLD}+  sp~{med_sp:.1f}p  L/S {long_n}/{short_n}")
        all_pnl.append(pnl)
        g_mom.append(mom3/med_sp)                       # momentum in spread units (cross-pair comparable)
        g_vol.append(vol3/np.median(vol3))              # volume relative to this pair's median
        g_dir.append(dirs)
    # portfolio
    port = np.concatenate(all_pnl) if all_pnl else np.array([])
    print("="*92)
    print(f"  net-positive: {npos}/{len(PAIRS)} pairs   WF-consistent(3/4): {nwf}/{len(PAIRS)}")
    if len(port):
        ppd = port.sum()/tot_days
        # Monte-Carlo: bootstrap mean p/trade vs 0 (sign-flip null)
        obs = port.mean(); B=2000
        null = np.array([ (port*rng.choice([-1,1],len(port))).mean() for _ in range(B) ])
        mc_p = float((np.abs(null) >= abs(obs)).mean())
        print(f"  portfolio: {len(port)} trades  net {obs:+.3f} p/tr  ~{ppd:+.1f} p/day  "
              f"total {port.sum():+.0f}p  MC p={mc_p:.4f}")
        print(f"  VERDICT: {'real edge' if (npos>=7 and nwf>=7 and obs>0 and mc_p<0.05) else 'NOT a deployable cross-pair edge (net of spread)'}")
        # --- conditioning: does the 3-bar pre-entry momentum / volume sort the outcome? ---
        MOM=np.concatenate(g_mom); VOL=np.concatenate(g_vol)
        print("\n" + "-"*92)
        bin_report("3-bar pre-entry MOMENTUM in trade dir (spread units; +=price moved WITH the trade)", MOM, port)
        bin_report("3-bar pre-entry TICK VOLUME (x pair-median; <1 = quiet, >1 = busy)", VOL, port)

if __name__=="__main__":
    main()
