#!/usr/bin/env python3
"""
bb_realizable_latency.py — DECISIVE re-validation of the BB-fade edge under realistic LIVE ENTRY.

The live diagnosis (bb_fill_faithfulness.py, 7/8 live trades stopped): the backtest enters at the
signal-bar CLOSE / next-open, but live places the order ~10-15s LATER at the streamed S5 price. In a
fast continuation move that S5 fill lands AT/PAST the extension-peak stop ("born stopped") -> instant
stop-out. This script measures whether a REALIZABLE edge survives once entry latency is modeled
honestly, per TF, with the real S5 path for BOTH entry timing and exit.

Method (per pair x TF in {M5,M15,H1,H4}):
  (1) SIGNALS — resample S5 BA -> TF bars; run the EXACT lib.bb_fade.backtest signal logic. For each
      signal record: direction, signal-bar CLOSE timestamp, ext-peak STOP, target band, baseline
      next-open entry (the backtest fill), and the per-bar real spread (ask_c-bid_c).
  (2) REALISTIC LIVE ENTRY — locate the S5 bar at (signal-bar-close + LATENCY) and take its mid close
      as the fill. LATENCY swept in {5s, 15s, 30s}.
  (3) ENTRY-SLIPPAGE GATE — skip if the realistic fill is already PAST the ext-peak stop (born
      stopped), or > X pips worse than the signal-bar close. X swept in {2,5,10,inf}.
  (4) EXIT FROM REALISTIC ENTRY — walk the ACTUAL S5 mid path forward from the fill; first-touch of
      stop vs target vs time-cap (R2 within-S5-bar sequencing: bull S5 bar -> low then high, bear ->
      high then low). PnL net of the real entry+exit spread.
  (5) COMPARE per TF: baseline backtest vs realizable(no gate) vs realizable(+best gate), per latency:
      trades, %born-stopped, %gated, expectancy(p/trade), pips/day, WR.

R-SOP compliance: mid OHLC for signal + bands; spread deducted explicitly (R3); incremental/causal
signal (the validated backtest); one code path for the signal. The exit here is the S5 GROUND TRUTH
(first-touch), which the live S5-monitor approximates.

Run: python3 research/experiments/daily_ma/bb_realizable_latency.py [--pairs P1,P2] [--tfs M5,H1]
Heavy (S5 path walk over ~5M bars/pair) -> numba JIT + one pair in memory at a time.
"""
import argparse, gc, sys, os
import numpy as np
import duckdb
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SMA = 9; K = 1.0

PAIRS = {"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
         "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}
# tf -> (pandas resample rule, time_cap_bars, meat_pips, bar_seconds)  [matches the live service TFS]
TFS = {"M5":("5min",288,4.0,300),"M15":("15min",96,6.0,900),"H1":("1h",24,6.0,3600),"H4":("4h",12,10.0,14400)}
LATENCIES = [5,15,30]        # seconds after the signal bar closes -> realistic fill
GATES = [2.0,5.0,10.0,np.inf]  # max pips the fill may be worse than the signal-bar close before we skip


@njit(cache=True)
def gen_signals(o,h,l,c,sp,bar_ts,basis,sd,pip,meat,tcap):
    """EXACT lib.bb_fade.backtest signal generation. Returns for each fired signal:
       sig_close_sec, dir, baseline_entry(next-open), ext_peak(stop), target_band, entry_spread, sig_close_px.
    NOTE: baseline exits are computed separately (the batch backtest) — here we only need the SIGNAL
    geometry so we can re-fill it from the S5 path. Position state mirrors backtest() so signals fire
    at the same bars (no overlapping entries while a position is notionally open)."""
    n=len(c); up=basis+K*sd; lo=basis-K*sd
    pos=0; ei=0; ent=0.0; stp=0.0; ext=0; peak=0.0
    # outputs
    cap = n
    out_sec=np.empty(cap,np.float64); out_dir=np.empty(cap,np.int64); out_ent=np.empty(cap,np.float64)
    out_stop=np.empty(cap,np.float64); out_tgt=np.empty(cap,np.float64); out_sp=np.empty(cap,np.float64)
    out_close=np.empty(cap,np.float64); out_sigi=np.empty(cap,np.int64)
    k=0
    for i in range(SMA,n-1):
        if np.isnan(basis[i]) or np.isnan(sd[i]) or np.isnan(sp[i]): continue
        uo = l[i]>up[i]; do = h[i]<lo[i]
        if uo: peak = h[i] if ext!=1 else max(peak,h[i]); ext=1
        elif do: peak = l[i] if ext!=-1 else min(peak,l[i]); ext=-1
        # manage notional position so signal cadence matches backtest()
        if pos!=0:
            ex=np.nan
            if pos==-1:
                if h[i]>stp: ex=stp
                elif l[i]<=lo[i]: ex=lo[i]
            else:
                if l[i]<stp: ex=stp
                elif h[i]>=up[i]: ex=up[i]
            if np.isnan(ex) and (i-ei)>=tcap: ex=c[i]
            if not np.isnan(ex): pos=0
        if pos==0:
            e=o[i+1]
            if l[i-1]>up[i-1] and l[i]<=up[i] and 0.5*(c[i]-basis[i])/pip-sp[i]>=meat:
                pos=-1; ent=e; ei=i+1; stp=peak
                out_sec[k]=bar_ts[i]; out_dir[k]=-1; out_ent[k]=e; out_stop[k]=peak; out_tgt[k]=lo[i+1] if not np.isnan(lo[i+1]) else lo[i]
                out_sp[k]=sp[i]; out_close[k]=c[i]; out_sigi[k]=i; k+=1
            elif h[i-1]<lo[i-1] and h[i]>=lo[i] and 0.5*(basis[i]-c[i])/pip-sp[i]>=meat:
                pos=1; ent=e; ei=i+1; stp=peak
                out_sec[k]=bar_ts[i]; out_dir[k]=1; out_ent[k]=e; out_stop[k]=peak; out_tgt[k]=up[i+1] if not np.isnan(up[i+1]) else up[i]
                out_sp[k]=sp[i]; out_close[k]=c[i]; out_sigi[k]=i; k+=1
    return (out_sec[:k],out_dir[:k],out_ent[:k],out_stop[:k],out_tgt[:k],out_sp[:k],out_close[:k],out_sigi[:k])


@njit(cache=True)
def baseline_backtest(o,h,l,c,sp,basis,sd,pip,meat,tcap):
    """The validated batch backtest with REAL per-bar spread (entry+exit half-spread each).
    Returns array of net pnl (pips) per trade + entry-bar-seconds for span calc. Mirrors
    validate_bb_reentry_full.gen but with the meat gate and opp-band target (the live exit)."""
    n=len(c); up=basis+K*sd; lo=basis-K*sd
    pos=0; ent=0.0; ei=0; es=0.0; stp=0.0; ext=0; peak=0.0
    pnl=np.empty(n,np.float64); k=0
    for i in range(SMA,n-1):
        if np.isnan(basis[i]) or np.isnan(sd[i]) or np.isnan(sp[i]): continue
        uo=l[i]>up[i]; do=h[i]<lo[i]
        if uo: peak=h[i] if ext!=1 else max(peak,h[i]); ext=1
        elif do: peak=l[i] if ext!=-1 else min(peak,l[i]); ext=-1
        if pos!=0:
            ex=np.nan; band=up[i] if pos==1 else lo[i]
            if pos==-1:
                if h[i]>stp: ex=stp
                elif l[i]<=lo[i]: ex=lo[i]
            else:
                if l[i]<stp: ex=stp
                elif h[i]>=up[i]: ex=up[i]
            if np.isnan(ex) and (i-ei)>=tcap: ex=c[i]
            if not np.isnan(ex):
                net=pos*(ex-ent)/pip - 0.5*(es+sp[i]); pnl[k]=net; k+=1; pos=0
        if pos==0:
            e=o[i+1]
            if l[i-1]>up[i-1] and l[i]<=up[i] and 0.5*(c[i]-basis[i])/pip-sp[i]>=meat:
                pos=-1; ent=e; ei=i+1; es=sp[i+1] if not np.isnan(sp[i+1]) else sp[i]; stp=peak
            elif h[i-1]<lo[i-1] and h[i]>=lo[i] and 0.5*(basis[i]-c[i])/pip-sp[i]>=meat:
                pos=1; ent=e; ei=i+1; es=sp[i+1] if not np.isnan(sp[i+1]) else sp[i]; stp=peak
    return pnl[:k]


@njit(cache=True)
def realize_trades(sig_sec, sig_dir, sig_stop, sig_tgt, sig_sp, sig_close,
                   s5_ts, s5_o, s5_h, s5_l, s5_c, s5_sp,
                   pip, latency, tcap_sec):
    """For each signal: find the S5 bar at (sig_close + latency) -> realistic fill = its mid close.
    Then walk the S5 path forward (first-touch stop/target/time-cap, R2 within-bar sequencing) ->
    realizable net pnl. Returns arrays: pnl, fill_px, born(0/1), slip_pips(fill worse than sig_close),
    entry_idx_sec, exit_reason(0 stop,1 target,2 timecap,3 no-path). slip>0 means worse fill."""
    ns=len(sig_sec); m=len(s5_ts)
    pnl=np.full(ns,np.nan); fillpx=np.full(ns,np.nan); born=np.zeros(ns,np.int64)
    slip=np.full(ns,np.nan); reason=np.full(ns,-1,np.int64); entsec=np.full(ns,np.nan)
    j=0  # rolling S5 cursor (signals are time-ordered)
    for s in range(ns):
        target_t = sig_sec[s] + latency
        # advance cursor to the first S5 bar whose ts >= target_t
        while j<m and s5_ts[j] < target_t: j+=1
        if j>=m: continue
        fi=j
        fill = s5_c[fi]   # mid close of the fill S5 bar
        d=sig_dir[s]; stop=sig_stop[s]; tgt=sig_tgt[s]; entry_sp=s5_sp[fi]
        if np.isnan(entry_sp): entry_sp=sig_sp[s]
        fillpx[s]=fill; entsec[s]=s5_ts[fi]
        # slippage vs signal-bar close (worse = adverse to the fade direction)
        # short fade: worse fill = HIGHER price; long fade: worse = LOWER price
        if d==-1: sl=(fill - sig_close[s])/pip
        else:     sl=(sig_close[s] - fill)/pip
        slip[s]=sl
        # born stopped: fill already at/beyond the ext-peak stop
        if d==-1: bs = fill >= stop          # short stop is ABOVE
        else:     bs = fill <= stop          # long stop is BELOW
        born[s]=1 if bs else 0
        # walk S5 forward from the fill bar; first-touch
        last_c = fill
        found=False
        for t in range(fi, m):
            if s5_ts[t] - s5_ts[fi] > tcap_sec:   # time cap
                last_c=s5_c[t-1] if t>fi else fill
                reason[s]=2
                pnl[s]= d*(last_c-fill)/pip - entry_sp   # exit spread paid once on the closing side; entry already in fill realism
                found=True; break
            hi=s5_h[t]; loo=s5_l[t]
            bull = s5_c[t] >= s5_o[t]
            hit_stop=False; hit_tgt=False; exitpx=0.0
            if d==-1:
                # short: stop = hi>=stop ; target = lo<=tgt. within-bar order:
                # bear bar -> high(then low); bull bar -> low(then high)
                if bull:
                    if loo<=tgt: hit_tgt=True; exitpx=tgt
                    elif hi>=stop: hit_stop=True; exitpx=stop
                else:
                    if hi>=stop: hit_stop=True; exitpx=stop
                    elif loo<=tgt: hit_tgt=True; exitpx=tgt
            else:
                # long: stop = lo<=stop ; target = hi>=tgt
                if bull:
                    if loo<=stop: hit_stop=True; exitpx=stop
                    elif hi>=tgt: hit_tgt=True; exitpx=tgt
                else:
                    if hi>=tgt: hit_tgt=True; exitpx=tgt
                    elif loo<=stop: hit_stop=True; exitpx=stop
            if hit_stop or hit_tgt:
                reason[s]= 0 if hit_stop else 1
                pnl[s]= d*(exitpx-fill)/pip - entry_sp
                found=True; break
        if not found:
            reason[s]=3   # ran off the end with no exit -> mark to-market at last bar
            pnl[s]= d*(s5_c[m-1]-fill)/pip - entry_sp
    return pnl, fillpx, born, slip, reason, entsec


def summarize(pnls, span_days, rng=None):
    pnls=pnls[~np.isnan(pnls)]
    if len(pnls)==0: return dict(n=0,pt=0.0,wr=0.0,ppd=0.0)
    return dict(n=len(pnls), pt=float(pnls.mean()), wr=100*float((pnls>0).mean()),
                ppd=float(pnls.sum()/span_days))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--tfs", default=",".join(TFS))
    args=ap.parse_args()
    pairs=[p for p in args.pairs.split(",") if p in PAIRS]
    tfs=[t for t in args.tfs.split(",") if t in TFS]
    con=duckdb.connect()

    # accumulators: per (tf) -> baseline pnl list, and per (tf,latency,gate) -> realizable pnl list + counters
    base_pnl={tf:[] for tf in tfs}; base_span={tf:0.0 for tf in tfs}
    real_pnl={(tf,lat,g):[] for tf in tfs for lat in LATENCIES for g in GATES}
    real_born={(tf,lat):0 for tf in tfs for lat in LATENCIES}   # born-stopped count (no gate)
    real_total={(tf,lat):0 for tf in tfs for lat in LATENCIES}  # total signals
    real_gated={(tf,lat,g):0 for tf in tfs for lat in LATENCIES for g in GATES}  # gated-out count
    span_days={tf:0.0 for tf in tfs}
    pair_used={tf:[] for tf in tfs}; pair_excluded=[]
    perpair_real={(tf,lat,g):{} for tf in tfs for lat in LATENCIES for g in GATES}

    for pair in pairs:
        pip=PAIRS[pair]
        print(f"\n[{pair}] loading S5 BA ...", flush=True)
        s5=con.execute(f"""SELECT epoch(timestamp) AS ts, open, high, low, close, bid_c, ask_c
                           FROM 'data/s5_ohlc/{pair}_S5_BA.parquet' ORDER BY timestamp""").df()
        if len(s5)<10000:
            print(f"  {pair}: insufficient S5 ({len(s5)} rows) — EXCLUDE"); pair_excluded.append(pair); del s5; gc.collect(); continue
        s5_ts=s5.ts.values.astype(np.float64)
        s5_o=s5.open.values.astype(np.float64); s5_h=s5.high.values.astype(np.float64)
        s5_l=s5.low.values.astype(np.float64); s5_c=s5.close.values.astype(np.float64)
        s5_sp=((s5.ask_c.values-s5.bid_c.values)/pip).astype(np.float64)
        span_d=(s5_ts[-1]-s5_ts[0])/86400.0
        print(f"  {pair}: {len(s5):,} S5 bars, span {span_d:.0f}d", flush=True)

        # build pandas index once for TF resample
        import pandas as pd
        s5i=s5.set_index(pd.to_datetime(s5.ts, unit="s", utc=True))
        for tf in tfs:
            rule,tcap,meat,bsec=TFS[tf]
            d=s5i.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last",
                                      "bid_c":"last","ask_c":"last","ts":"first"}).dropna()
            o=d.open.values.astype(np.float64); h=d.high.values.astype(np.float64)
            l=d.low.values.astype(np.float64); c=d.close.values.astype(np.float64)
            sp=((d.ask_c.values-d.bid_c.values)/pip).astype(np.float64)
            bar_ts=d.ts.values.astype(np.float64)   # signal-bar OPEN second; close = +bsec
            basis=pd.Series(c).rolling(SMA).mean().values; sdv=pd.Series(c).rolling(SMA).std().values
            if len(c)<SMA+5: continue

            # baseline backtest (validated path, real spread)
            bp=baseline_backtest(o,h,l,c,sp,basis,sdv,pip,float(meat),tcap)
            base_pnl[tf].append(bp); base_span[tf]+=span_d; span_days[tf]+=span_d
            pair_used[tf].append(pair)

            # signals (geometry only) — sig_close_sec is the bar OPEN ts; the bar CLOSES at +bsec
            (sig_sec_open,sig_dir,sig_ent,sig_stop,sig_tgt,sig_sp,sig_close,sig_sigi)=gen_signals(
                o,h,l,c,sp,bar_ts,basis,sdv,pip,float(meat),tcap)
            sig_close_sec=sig_sec_open+bsec   # the moment the signal bar closes (live "now")
            if len(sig_sec_open)==0: continue

            for lat in LATENCIES:
                pnl,fillpx,born,slip,reason,entsec=realize_trades(
                    sig_close_sec,sig_dir,sig_stop,sig_tgt,sig_sp,sig_close,
                    s5_ts,s5_o,s5_h,s5_l,s5_c,s5_sp,
                    pip,float(lat),float(tcap*bsec))
                valid=~np.isnan(pnl)
                real_total[(tf,lat)]+=int(valid.sum())
                real_born[(tf,lat)]+=int((born[valid]==1).sum())
                for g in GATES:
                    if np.isinf(g):
                        keep = valid & (born==0)            # only born-stopped removed
                    else:
                        keep = valid & (born==0) & (slip<=g)  # also slip gate
                    gated_out = int((valid & (~keep)).sum())
                    real_gated[(tf,lat,g)]+=gated_out
                    real_pnl[(tf,lat,g)].append(pnl[keep])
                    pk=pnl[keep]
                    if len(pk)>=10: perpair_real[(tf,lat,g)][pair]=float(pk.mean())
        del s5,s5i; gc.collect()

    # ---- REPORT ----
    print("\n"+"="*100)
    print("BB-FADE REALIZABLE-LATENCY VALIDATION — baseline vs realistic live entry (S5 ground-truth exit)")
    print("="*100)
    print(f"pairs used per TF: " + "; ".join(f"{tf}:{len(pair_used[tf])}" for tf in tfs))
    if pair_excluded: print(f"EXCLUDED (insufficient S5): {pair_excluded}")

    # baseline summary per TF
    print("\n--- BASELINE BACKTEST (next-open entry, real per-bar spread, opp-band/ext-peak/time-cap exit) ---")
    print(f"{'TF':>4} {'trades':>8} {'p/trade':>9} {'WR':>6} {'p/day':>9} {'span(d)':>8}")
    base_stat={}
    for tf in tfs:
        if not base_pnl[tf]: continue
        allp=np.concatenate(base_pnl[tf]); sd=span_days[tf]/max(1,len(pair_used[tf]))  # avg per-pair span; pips/day = total/avg-span
        s=summarize(allp, sd)
        base_stat[tf]=s
        print(f"{tf:>4} {s['n']:>8} {s['pt']:>+9.3f} {s['wr']:>5.1f}% {s['ppd']:>+9.2f} {sd:>8.0f}")

    # realizable: for each latency, table of no-gate + each gate
    for lat in LATENCIES:
        print(f"\n--- REALIZABLE: entry latency = {lat}s after signal-bar close ---")
        print(f"{'TF':>4} {'gate':>5} {'signals':>8} {'born%':>6} {'gated%':>7} {'trades':>7} {'p/trade':>9} {'WR':>6} {'p/day':>9} {'pairs+':>7}")
        for tf in tfs:
            if not base_pnl[tf]: continue
            sd=span_days[tf]/max(1,len(pair_used[tf]))
            tot=real_total[(tf,lat)]; bornc=real_born[(tf,lat)]
            for g in GATES:
                glab = "inf" if np.isinf(g) else f"{g:.0f}"
                pk=real_pnl[(tf,lat,g)]
                allp=np.concatenate(pk) if pk else np.array([])
                s=summarize(allp, sd)
                gated=real_gated[(tf,lat,g)]
                bornpct=100*bornc/tot if tot else 0
                gatedpct=100*gated/tot if tot else 0
                pp=perpair_real[(tf,lat,g)]; npos=sum(1 for v in pp.values() if v>0); ntot=len(pp)
                print(f"{tf:>4} {glab:>5} {tot:>8} {bornpct:>5.1f}% {gatedpct:>6.1f}% {s['n']:>7} {s['pt']:>+9.3f} {s['wr']:>5.1f}% {s['ppd']:>+9.2f} {npos:>3}/{ntot:<3}")

    # headline comparison: baseline vs realizable(no gate, lat=15s) vs realizable(best gate, lat=15s)
    print("\n"+"="*100)
    print("HEADLINE — baseline vs realizable @15s (no gate) vs realizable @15s (gate=5p), per TF")
    print("="*100)
    print(f"{'TF':>4} | {'baseline p/t':>12} {'p/day':>8} | {'real(no gate) p/t':>17} {'p/day':>8} {'born%':>6} | {'real(g=5) p/t':>13} {'p/day':>8} {'gated%':>7}")
    for tf in tfs:
        if tf not in base_stat: continue
        sd=span_days[tf]/max(1,len(pair_used[tf]))
        bs=base_stat[tf]
        ng=summarize(np.concatenate(real_pnl[(tf,15,np.inf)]) if real_pnl[(tf,15,np.inf)] else np.array([]), sd)
        wg=summarize(np.concatenate(real_pnl[(tf,15,5.0)]) if real_pnl[(tf,15,5.0)] else np.array([]), sd)
        bornpct=100*real_born[(tf,15)]/real_total[(tf,15)] if real_total[(tf,15)] else 0
        gatedpct=100*real_gated[(tf,15,5.0)]/real_total[(tf,15)] if real_total[(tf,15)] else 0
        print(f"{tf:>4} | {bs['pt']:>+12.3f} {bs['ppd']:>+8.2f} | {ng['pt']:>+17.3f} {ng['ppd']:>+8.2f} {bornpct:>5.1f}% | {wg['pt']:>+13.3f} {wg['ppd']:>+8.2f} {gatedpct:>6.1f}%")
    print("\n(p/day is total-net-pips / avg-per-pair-span; sums over the multi-pair portfolio at that TF.)")


if __name__=="__main__":
    main()
