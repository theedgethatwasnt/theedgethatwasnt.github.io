#!/usr/bin/env python3
"""
bb_fill_faithfulness.py — FILL-FAITHFULNESS check on the live BB-fade trades.

Question: does the live S5 tick-monitored exit (fires on first adverse tick) faithfully reproduce
reality (the actual S5 path), and does the backtest's bar+within-bar-sequencing exit OVERSTATE wins?

For each live bb_fade trade in trades.duckdb:
  1. Reconstruct the SIGNAL: fetch ~150 signal-TF MID bars up to entry_time, run BBFadeProcessor with
     the EXACT live config (meat per TF, real fetched spread). Confirm it reproduces the live entry
     (same direction, entry price near the recorded fill). Extract ext_peak (stop) + target band.
  2. ACTUAL S5 PATH (ground truth): fetch S5 MID candles from the entry bar through exit_time+margin;
     walk ticks in time order; which level (stop vs target) is touched FIRST, and when.
  3. BACKTEST bar+R2: replay the signal-TF bars AFTER entry through lib.bb_fade.backtest's exact
     within-bar exit logic; record stop vs target.
  4. Compare (a) LIVE recorded, (b) ACTUAL S5, (c) BACKTEST.

Run inside the curator container (has OANDA_API_KEY + v20 + the lib on path):
  docker exec fx-core-fx-data-curator-1 python3 /app/research/experiments/daily_ma/bb_fill_faithfulness.py
(the repo is mounted at /app inside the container)
"""
import os, sys
from datetime import datetime, timezone, timedelta
import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import v20
from lib.bb_fade import BBFadeProcessor, bands, SMA, K

# ── live config (services/strategy_bb_fade_live/main.py) ──
PIP = {"EUR_USD":1e-4,"GBP_USD":1e-4,"AUD_USD":1e-4,"NZD_USD":1e-4,"EUR_GBP":1e-4,
       "USD_JPY":1e-2,"EUR_JPY":1e-2,"GBP_JPY":1e-2,"AUD_JPY":1e-2,"CAD_JPY":1e-2,"NZD_JPY":1e-2,"CHF_JPY":1e-2}
MEAT = {"M5":4.0, "M15":6.0, "H1":6.0, "H4":10.0}
GRAN = {"M5":"M5","M15":"M15","H1":"H1","H4":"H4"}
BAR_SEC = {"M5":300,"M15":900,"H1":3600,"H4":14400}

CTX = v20.Context(hostname='api-fxtrade.oanda.com', port='443', token=os.environ['OANDA_API_KEY'])

def _clamp_future(toTime):
    """OANDA rejects count+toTime when toTime is in the future. Clamp to ~now-1s."""
    if toTime is None: return None
    try:
        t=datetime.fromisoformat(iso(toTime))
        now=datetime.now(timezone.utc)
        if t>now: return (now-timedelta(seconds=2)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    except Exception: pass
    return toTime

def fetch_candles(pair, gran, count=None, fromTime=None, toTime=None, price="M"):
    toTime=_clamp_future(toTime)
    kw=dict(granularity=gran, price=price)
    if count is not None: kw["count"]=count
    if fromTime is not None: kw["fromTime"]=fromTime
    if toTime is not None: kw["toTime"]=toTime
    r=CTX.instrument.candles(pair, **kw)
    if "candles" not in r.body:
        raise RuntimeError(f"OANDA error for {pair} {gran}: {r.body}")
    out=[]
    for c in r.body["candles"]:
        if not c.complete and toTime is None and count is not None:
            pass  # keep; we filter by time later
        m=c.mid
        out.append({"t":c.time, "o":float(m.o),"h":float(m.h),"l":float(m.l),"c":float(m.c),
                    "complete":c.complete})
    return out

def fetch_ba_spread(pair, gran, toTime):
    """Representative spread (pips) at entry: avg (ask_c - bid_c) over the last few completed bars."""
    try:
        toTime=_clamp_future(toTime)
        rb=CTX.instrument.candles(pair, granularity=gran, count=5, toTime=toTime, price="B")
        ra=CTX.instrument.candles(pair, granularity=gran, count=5, toTime=toTime, price="A")
        sb=[float(c.bid.c) for c in rb.body["candles"] if c.complete]
        sa=[float(c.ask.c) for c in ra.body["candles"] if c.complete]
        if sb and sa:
            n=min(len(sb),len(sa))
            return float(np.mean([(sa[i]-sb[i])/PIP[pair] for i in range(n)]))
    except Exception as e:
        print(f"  spread fetch fail: {e}")
    return 2.0

def iso(ts):
    return ts.replace("Z","+00:00")
def tsec(ts):
    return datetime.fromisoformat(iso(ts)).timestamp()

def reconstruct_signal(pair, tf, entry_dt, live_entry_px):
    """Run BBFadeProcessor over ~150 signal-TF bars ending just before entry_dt (the bar that fired
    the signal closes just before the next-open entry). Return (dir, entry_open, ext_peak, target_band,
    sig_bar_idx, bars) where bars is the list of bar dicts used."""
    pip=PIP[pair]; meat=MEAT[tf]; gran=GRAN[tf]; bsec=BAR_SEC[tf]
    # fetch bars up to a little past entry so we capture the signal bar + the entry (next-open) bar
    toT = (entry_dt + timedelta(seconds=bsec*3)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    bars = fetch_candles(pair, gran, count=160, toTime=toT)
    spread = fetch_ba_spread(pair, gran, toT)
    proc = BBFadeProcessor(pip, spread, meat, 10**9)
    entry_target_sec = entry_dt.timestamp()
    sig = None  # best candidate (next_open closest to live entry & close <= entry time)
    cands = []
    for i,b in enumerate(bars):
        if not b["complete"]: continue
        bt = tsec(b["t"])
        prev_pending = proc.pending
        proc.update(b["o"],b["h"],b["l"],b["c"])
        if proc.pending is not None and prev_pending is None:
            d,stp = proc.pending
            tgt = proc.cur_lo if d==-1 else proc.cur_up
            nxt = bars[i+1]["o"] if i+1<len(bars) else None
            cands.append({"dir":d,"ext_peak":stp,"target":tgt,"sig_close_t":b["t"],"sig_idx":i,
                          "next_open":nxt,"spread":spread,"basis":proc.cur_basis,"up":proc.cur_up,"lo":proc.cur_lo,
                          "sig_close_sec":bt})
            # service keeps proc flat (pure detector): clear pending/pos so detection continues
            proc.pending=None; proc.pos=0
    # pick the candidate whose next_open is closest to the recorded live fill AND whose signal close
    # is within ~2 bars of the live entry timestamp (live fills next-open right after the signal close).
    if cands:
        window=[c for c in cands if abs(c["sig_close_sec"]-entry_target_sec) <= BAR_SEC[tf]*2.5 and c["next_open"] is not None]
        if window:
            sig = min(window, key=lambda c: abs(c["next_open"]-live_entry_px))
    return sig, bars, spread, cands

def actual_s5_first_touch(pair, entry_dt, exit_dt, ext_peak, target, direction):
    """Fetch S5 MID candles from entry bar through exit+margin; walk in time order; return which level
    is touched FIRST (stop/target/neither) and the time. For SHORT: stop = hi>=ext_peak, target = lo<=target."""
    pip=PIP[pair]
    fromT=(entry_dt - timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    toT=(exit_dt + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    s5=fetch_candles(pair, "S5", fromTime=fromT, toTime=toT)
    es=entry_dt.timestamp()
    touched=None
    for c in s5:
        if not c["complete"]: continue
        ct=tsec(c["t"])
        if ct < es - 5: continue   # only ticks at/after entry (5s grace)
        hi=c["h"]; lo=c["l"]
        if direction==-1:
            hit_stop = hi>=ext_peak
            hit_tgt  = (target is not None) and lo<=target
        else:
            hit_stop = lo<=ext_peak
            hit_tgt  = (target is not None) and hi>=target
        # within a single S5 candle, if BOTH hit, we can't order them from OHLC alone — flag it.
        if hit_stop and hit_tgt:
            return {"first":"BOTH_in_5s_candle","t":c["t"],"hi":hi,"lo":lo}
        if hit_stop:
            return {"first":"stop","t":c["t"],"hi":hi,"lo":lo}
        if hit_tgt:
            return {"first":"target","t":c["t"],"hi":hi,"lo":lo}
    return {"first":"neither","t":None}

def backtest_outcome(pair, tf, sig, bars, spread):
    """Replay the signal-TF bars AFTER the signal bar through the EXACT backtest exit logic
    (lib.bb_fade.backtest: SHORT -> high-then-low; LONG -> low-then-high). Return stop/target/timecap +
    which bar + whether the FIRST exit bar's range touched BOTH levels."""
    pip=PIP[pair]; direction=sig["dir"]; ent=sig["next_open"]; stp=sig["ext_peak"]
    if ent is None: return {"outcome":"NO_NEXT_OPEN"}
    # recompute bands per bar exactly as backtest() does
    cl=np.array([b["c"] for b in bars]); n=len(bars)
    basis=np.full(n,np.nan); sd=np.full(n,np.nan)
    for i in range(SMA-1,n):
        w=cl[i-SMA+1:i+1]; basis[i]=w.mean(); sd[i]=w.std()
    up=basis+K*sd; lo=basis-K*sd
    ei=sig["sig_idx"]+1   # entry bar = next bar after signal bar
    first_bar_both=False
    for i in range(ei, n):
        b=bars[i]
        if not b["complete"]: continue
        h=b["h"]; l=b["l"]
        if direction==-1:
            hit_stop = h>stp
            hit_tgt  = l<=lo[i]
            if i==ei and hit_stop and hit_tgt: first_bar_both=True
            if hit_stop: return {"outcome":"stop","bar":i,"exit_px":stp,"both_first_bar":first_bar_both,"band":lo[i]}
            if hit_tgt:  return {"outcome":"target","bar":i,"exit_px":lo[i],"both_first_bar":first_bar_both,"band":lo[i]}
        else:
            hit_stop = l<stp
            hit_tgt  = h>=up[i]
            if i==ei and hit_stop and hit_tgt: first_bar_both=True
            if hit_stop: return {"outcome":"stop","bar":i,"exit_px":stp,"both_first_bar":first_bar_both,"band":up[i]}
            if hit_tgt:  return {"outcome":"target","bar":i,"exit_px":up[i],"both_first_bar":first_bar_both,"band":up[i]}
    return {"outcome":"timecap/open","bar":n-1}

# confirmed bb_fade live trades (queried from trades.duckdb 2026-06-25; embedded so we don't fight the
# writer's lock). cols: entry_time, exit_time, pair, direction, pnl_pips, exit_reason, label,
# entry_price, exit_price, hours_held
EMBEDDED_TRADES=[
 ("2026-06-25 01:20:11","2026-06-25 01:20:23","GBP_JPY",-1,-2.9,"stop","bb_fade_M5",213.112,213.141,0.0),
 ("2026-06-25 12:45:02","2026-06-25 12:45:14","EUR_USD",-1,-0.9,"stop","bb_fade_M5",1.13616,1.13625,0.0),
 ("2026-06-25 12:45:02","2026-06-25 12:45:15","EUR_JPY",-1,-2.7,"stop","bb_fade_M5",183.814,183.841,0.0),
 ("2026-06-25 12:45:02","2026-06-25 12:45:16","CHF_JPY",-1,-6.4,"stop","bb_fade_M5",199.414,199.478,0.0),
 ("2026-06-25 12:55:09","2026-06-25 13:16:18","GBP_JPY",-1,-8.5,"stop","bb_fade_M5",213.274,213.359,0.35),
 ("2026-06-25 13:40:07","2026-06-25 13:40:20","GBP_USD",-1,-1.0,"stop","bb_fade_M5",1.31969,1.31979,0.0),
 ("2026-06-25 16:00:04","2026-06-25 16:00:17","AUD_JPY",-1,-2.5,"stop","bb_fade_M15",111.877,111.902,0.0),
 ("2026-06-25 16:00:05","2026-06-25 16:00:19","CAD_JPY",-1,-4.5,"stop","bb_fade_H1",113.923,113.968,0.0),
]

def main():
    rows=None
    try:
        import duckdb
        con=duckdb.connect("/data/db/trades.duckdb", read_only=True)
        rows=con.execute("""SELECT entry_time, exit_time, pair, direction, pnl_pips, exit_reason, label,
                            entry_price, exit_price, hours_held FROM trades
                            WHERE label LIKE 'bb_fade%' ORDER BY entry_time""").fetchall()
    except Exception as e:
        print(f"(DB locked, using embedded trade list: {e})")
        rows=EMBEDDED_TRADES
    print(f"{len(rows)} bb_fade trades\n")
    hdr=["pair/TF","dir","entry","stop(ext)","target","LIVE","S5-first(t)","BT-R2","sigOK","agree"]
    table=[]
    n_both_bt=0; n_bt_overstate=0; bt_overstate_pips=0.0
    n_live_match_s5=0; n_assessable=0; n_born_stopped=0
    for r in rows:
        entry_time, exit_time, pair, direction, pnl, reason, label, entry_px, exit_px, hrs = r
        tf=label.replace("bb_fade_","")
        entry_dt=datetime.fromisoformat(str(entry_time)).replace(tzinfo=timezone.utc) if datetime.fromisoformat(str(entry_time)).tzinfo is None else datetime.fromisoformat(str(entry_time))
        exit_dt =datetime.fromisoformat(str(exit_time)).replace(tzinfo=timezone.utc) if datetime.fromisoformat(str(exit_time)).tzinfo is None else datetime.fromisoformat(str(exit_time))
        print(f"=== {pair} {tf} dir={direction} live_entry={entry_px} live_exit={exit_px} pnl={pnl} reason={reason} ===")
        sig, bars, spread, cands = reconstruct_signal(pair, tf, entry_dt, entry_px)
        if sig is None:
            print(f"  SIGNAL NOT RECONSTRUCTED ({len(cands)} candidates total) — excluding\n")
            table.append([f"{pair}/{tf}",direction,entry_px,"-","-",reason,"-","-","NO","EXCLUDE"])
            continue
        # show the few candidate signals near the entry for diagnosis
        near=[c for c in cands if abs(c["sig_close_sec"]-entry_dt.timestamp())<BAR_SEC[tf]*4]
        for c in near:
            print(f"    cand sig_close={c['sig_close_t'][11:19]} dir={c['dir']} next_open={c['next_open']} ext_peak={c['ext_peak']:.3f} tgt={c['target']:.3f}")
        # confirm reconstruction vs live
        ent=sig["next_open"]; stp=sig["ext_peak"]; tgt=sig["target"]
        dir_ok = (sig["dir"]==direction)
        px_err = abs(ent-entry_px)/PIP[pair] if ent else 9999
        sigOK = dir_ok and px_err<5.0   # within 5 pips of the recorded fill
        # "born stopped": is the entry fill already at/beyond the extension-peak stop?
        if sig["dir"]==-1:
            beyond = (ent - stp)/PIP[pair]   # short: positive => entry ABOVE stop => already breached
        else:
            beyond = (stp - ent)/PIP[pair]
        born_stopped = beyond >= 0
        print(f"  recon: dir={sig['dir']} next_open={ent} (live {entry_px}, err={px_err:.1f}p) ext_peak={stp} target_band={tgt} spread={spread:.2f}p")
        print(f"  entry-vs-stop: {'BORN STOPPED' if born_stopped else 'ok'} (entry is {beyond:+.1f}p {'beyond' if born_stopped else 'inside'} the stop at fill)")
        # ACTUAL S5
        s5res=actual_s5_first_touch(pair, entry_dt, exit_dt, stp, tgt, sig["dir"])
        # BACKTEST
        btres=backtest_outcome(pair, tf, sig, bars, spread)
        print(f"  S5 first-touch: {s5res['first']} @ {s5res.get('t')}")
        print(f"  BT-R2: {btres}")
        live_out=reason  # 'stop'
        s5_out=s5res["first"]
        bt_out=btres.get("outcome","?")
        # bookkeeping
        if sigOK:
            n_assessable+=1
            if born_stopped: n_born_stopped+=1
            if live_out==s5_out or (s5_out=="BOTH_in_5s_candle" and live_out=="stop"):
                n_live_match_s5+=1
            if btres.get("both_first_bar"): n_both_bt+=1
            # overstatement: BT says target (win) where S5 reality was stop-first (or both)
            if bt_out=="target" and s5_out in ("stop","BOTH_in_5s_candle"):
                n_bt_overstate+=1
                # pip overstatement = BT target pnl - actual stop pnl
                bt_pnl = sig["dir"]*(btres["exit_px"]-ent)/PIP[pair]-spread
                bt_overstate_pips += (bt_pnl - pnl)
        agree = "live=S5" if (live_out==s5_out) else f"live!={s5_out}"
        table.append([f"{pair}/{tf}",direction,round(ent,5) if ent else "-",
                      round(stp,5),round(tgt,5) if tgt else "-",live_out,
                      f"{s5_out}@{(s5res.get('t') or '')[11:19]}",bt_out,
                      "Y" if sigOK else "N",agree])
        print()
    # print table
    print("\n================ PER-TRADE TABLE ================")
    w=[max(len(str(row[i])) for row in [hdr]+table) for i in range(len(hdr))]
    def fmt(row): return " | ".join(str(row[i]).ljust(w[i]) for i in range(len(row)))
    print(fmt(hdr)); print("-"*(sum(w)+3*len(w)))
    for row in table: print(fmt(row))
    print("\n================ SUMMARY ================")
    print(f"assessable (signal cleanly reconstructed): {n_assessable}/{len(rows)}")
    print(f"BORN STOPPED (entry fill already at/beyond the ext-peak stop): {n_born_stopped}/{n_assessable}")
    print(f"LIVE outcome matches ACTUAL S5 first-touch: {n_live_match_s5}/{n_assessable}")
    print(f"signal bar touched BOTH stop+target (sequencing matters): {n_both_bt}")
    print(f"BACKTEST recorded TARGET(win) where S5 reality was STOP-first: {n_bt_overstate}")
    print(f"  -> total backtest pip OVERSTATEMENT on these trades: {bt_overstate_pips:+.1f}p")

if __name__=="__main__":
    main()
