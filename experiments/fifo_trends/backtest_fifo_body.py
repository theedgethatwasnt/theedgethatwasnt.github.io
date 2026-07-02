"""
FIFO-Trends Body Filter Experiment
===================================
Extends the 2,700-config sweep with a candle body_frac parameter.

Hypothesis: gating the E2 entry bar on a minimum body fraction in the trade
direction (bull body for LONG, bear body for SHORT) filters out doji/counter-body
entries and improves OOS p/d without killing trade frequency.

Body filter applied at: E2 confirmation bar AND E1 reversal bar.
  - For LONG  (pnf_dir=+1): requires (close-open)/range >= body_frac, close>=open
  - For SHORT (pnf_dir=-1): requires (open-close)/range >= body_frac, close<=open
  - Flat bar (range==0): rejected when filter active

Fracs: 0.0 (baseline), 0.3, 0.5, 0.7  → body_frac_x10 col = 0, 3, 5, 7
Total configs: 2,700 × 4 = 10,800

Pairs: GBP_JPY, USD_JPY, EUR_JPY, GBP_USD  (top 4 from original sweep)

SOP (CLAUDE.md §Backtest-Live Consistency):
  R1  Body computed from closed bar[i] OHLC — causal
  R3a Mid open/high/low/close used — no bid/ask in body calc
  R6  Same body check will go into live service entry logic
  R8  OOS touched exactly once per pair after IS+MC pass

Run:
  cd /path/to/projects/fx-core
  python3 research/experiments/fifo_trends/backtest_fifo_body.py
"""

import os, sys, time, requests
from pathlib import Path
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from dotenv import load_dotenv

load_dotenv()
BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))

BA_DIR  = BASE / "data/m5_ba"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

TARGET_PAIRS = [
    ("GBP_JPY", 0.01),
    ("USD_JPY", 0.01),
    ("EUR_JPY", 0.01),
    ("GBP_USD", 0.0001),
]

IS_FRAC    = 0.70
MAX_TRADES = 20000
MAX_K      = 10

def tg(msg):
    if not TELEGRAM_TOKEN:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception:
        pass

# ─── Parameter space: 10,800 configs ──────────────────────────────────────────
BOX_SIZES      = np.array([5, 10, 15, 20, 30], dtype=np.int32)
REVERSALS      = np.array([1, 2, 3],            dtype=np.int32)
MIN_COLS       = np.array([2, 3, 4, 5, 6, 8],  dtype=np.int32)
BODY_FRACS_X10 = np.array([0, 3, 5, 7],         dtype=np.int32)
EXIT_DEFS = [
    (0,3,2),(1,5,3),(2,8,4),(3,0,0),
    (4,1,0),(5,2,0),(6,3,0),
    (7,3,0),(8,5,0),(9,8,0),
    (10,1,3),(11,1,5),(12,2,3),(13,2,5),(14,3,5),
]
EXIT_NAMES = [
    "X1_3","X1_5","X1_8","X2",
    "X3b_1","X3b_2","X3b_3",
    "X7_3","X7_5","X7_8",
    "X3c_1_3","X3c_1_5","X3c_2_3","X3c_2_5","X3c_3_5",
]
ENTRY_LABEL = ["E1", "E2"]

def build_configs():
    rows = []
    for bs in BOX_SIZES:
        for rv in REVERSALS:
            for nc in MIN_COLS:
                for et in [0, 1]:
                    for (xt, xp1, xp2) in EXIT_DEFS:
                        for bf in BODY_FRACS_X10:
                            rows.append((bs, rv, nc, et, xt, xp1, xp2, bf))
    return np.array(rows, dtype=np.int32)

CONFIGS   = build_configs()
N_CONFIGS = len(CONFIGS)

CONFIG_NAMES = []
for ci in range(N_CONFIGS):
    bs,rv,nc,et,xt,xp1,xp2,bf = CONFIGS[ci]
    bf_str = f"_bf{bf}" if bf > 0 else ""
    CONFIG_NAMES.append(f"b{bs}_r{rv}_n{nc}_{ENTRY_LABEL[et]}_{EXIT_NAMES[xt]}{bf_str}")

# ─── Numba kernel ─────────────────────────────────────────────────────────────
@nb.njit(inline="always")
def col_sma(hist, ptr, n_valid, k):
    count = min(k, n_valid)
    if count == 0:
        return 0.0
    total = 0.0
    for j in range(count):
        total += hist[(ptr - 1 - j) % MAX_K]
    return total / count


@nb.njit(parallel=True)
def run_kernel(
    opens, highs, lows, closes, spreads, bar_chunks,
    configs, spread_gate, pip, is_end,
    trade_pnl, trade_chunk, trade_cnt,
):
    N_BARS    = len(opens)
    N_CONFIGS = configs.shape[0]

    for ci in prange(N_CONFIGS):
        bs_pips = configs[ci, 0];  rev     = configs[ci, 1]
        n_min   = configs[ci, 2];  entry_t = configs[ci, 3]
        exit_t  = configs[ci, 4];  xp1     = configs[ci, 5];  xp2 = configs[ci, 6]
        body_frac = configs[ci, 7] / 10.0   # 0→0.0, 3→0.3, 5→0.5, 7→0.7

        bs = bs_pips * pip

        pnf_idx=0; pnf_level=0.0; pnf_dir=0; col_count=0; prev_col=0
        col_hist=np.zeros(MAX_K, dtype=np.float64)
        col_hist_ptr=0; col_hist_n=0
        pos=0; entry_px=0.0; hw_level=0.0; pending=0
        t_cnt=0

        for i in range(N_BARS):
            opn=opens[i]; hi=highs[i]; lo=lows[i]; cl=closes[i]
            sp=spreads[i]; ck=bar_chunks[i]
            bull=(cl>=opn)
            p1=hi if bull else lo
            p2=lo if bull else hi

            # Body values for current bar (R1: bar i is already closed)
            bar_range = hi - lo
            bar_body  = cl - opn   # positive=bull, negative=bear

            did_reverse_p1=False; did_reverse_p2=False
            prev_col_p1=0; prev_col_p2=0

            for tick in range(2):
                px = p1 if tick==0 else p2
                if pnf_dir==0:
                    pnf_idx=int(px/bs); pnf_level=pnf_idx*bs
                    pnf_dir=1; col_count=1; continue
                delta=int(px/bs)-pnf_idx
                if pnf_dir==1:
                    if delta>=1:
                        pnf_idx+=delta; pnf_level=pnf_idx*bs; col_count+=delta
                    elif delta<=-rev:
                        prev_col=col_count
                        col_hist[col_hist_ptr%MAX_K]=prev_col; col_hist_ptr+=1
                        if col_hist_n<MAX_K: col_hist_n+=1
                        pnf_dir=-1; pnf_idx+=delta; pnf_level=pnf_idx*bs; col_count=-delta
                        if tick==0: did_reverse_p1=True; prev_col_p1=prev_col
                        else:       did_reverse_p2=True; prev_col_p2=prev_col
                elif pnf_dir==-1:
                    if delta<=-1:
                        pnf_idx+=delta; pnf_level=pnf_idx*bs; col_count+=(-delta)
                    elif delta>=rev:
                        prev_col=col_count
                        col_hist[col_hist_ptr%MAX_K]=prev_col; col_hist_ptr+=1
                        if col_hist_n<MAX_K: col_hist_n+=1
                        pnf_dir=1; pnf_idx+=delta; pnf_level=pnf_idx*bs; col_count=delta
                        if tick==0: did_reverse_p1=True; prev_col_p1=prev_col
                        else:       did_reverse_p2=True; prev_col_p2=prev_col

            did_reverse=did_reverse_p1 or did_reverse_p2
            prev_col_at_rev=prev_col_p1 if did_reverse_p1 else prev_col_p2

            if pos==1:
                if pnf_dir==1 and pnf_level>hw_level: hw_level=pnf_level
            elif pos==-1:
                if pnf_dir==-1 and pnf_level<hw_level: hw_level=pnf_level

            exit_triggered=False; exit_px_val=0.0
            if pos!=0:
                if exit_t<=2:
                    tp_b=float(xp1); sl_b=float(xp2)
                    if pos==1:
                        sl_p=entry_px-sl_b*bs; tp_p=entry_px+tp_b*bs
                        if lo<=sl_p: exit_px_val=sl_p; exit_triggered=True
                        elif hi>=tp_p: exit_px_val=tp_p; exit_triggered=True
                    else:
                        sl_p=entry_px+sl_b*bs; tp_p=entry_px-tp_b*bs
                        if hi>=sl_p: exit_px_val=sl_p; exit_triggered=True
                        elif lo<=tp_p: exit_px_val=tp_p; exit_triggered=True
                elif exit_t==3:
                    if did_reverse and pnf_dir!=pos:
                        exit_px_val=cl; exit_triggered=True
                elif 4<=exit_t<=6:
                    d=float(xp1)
                    if pos==1:
                        trail=hw_level-d*bs
                        if lo<=trail: exit_px_val=trail; exit_triggered=True
                    else:
                        trail=hw_level+d*bs
                        if hi>=trail: exit_px_val=trail; exit_triggered=True
                elif 7<=exit_t<=9:
                    k=xp1
                    if pnf_dir!=pos:
                        sma_k=col_sma(col_hist,col_hist_ptr,col_hist_n,k)
                        if sma_k>0.0 and col_count>=sma_k:
                            exit_px_val=cl; exit_triggered=True
                else:
                    d=float(xp1); k=xp2
                    if pos==1:
                        trail=hw_level-d*bs
                        if lo<=trail: exit_px_val=trail; exit_triggered=True
                    else:
                        trail=hw_level+d*bs
                        if hi>=trail: exit_px_val=trail; exit_triggered=True
                    if not exit_triggered and pnf_dir!=pos:
                        sma_k=col_sma(col_hist,col_hist_ptr,col_hist_n,k)
                        if sma_k>0.0 and col_count>=sma_k:
                            exit_px_val=cl; exit_triggered=True

            if exit_triggered and t_cnt<MAX_TRADES:
                pnl_pips=(exit_px_val-entry_px)*pos/pip-sp
                trade_pnl[ci,t_cnt]=np.float32(pnl_pips)
                trade_chunk[ci,t_cnt]=ck
                t_cnt+=1; pos=0; entry_px=0.0; hw_level=0.0

            if pos==0:
                can_enter=(sp<=spread_gate)
                if can_enter:
                    # Body check for this bar: does it confirm the intended direction?
                    # Computed here, applied conditionally below (R3a: mid OHLC only)
                    if body_frac>0.0 and bar_range>0.0:
                        body_abs = bar_body if bar_body>0.0 else -bar_body
                        body_frac_actual = body_abs / bar_range
                        bull_ok  = bar_body>=0.0 and body_frac_actual>=body_frac
                        bear_ok  = bar_body<=0.0 and body_frac_actual>=body_frac
                    elif body_frac>0.0:
                        # Flat bar — reject when filter is active
                        bull_ok  = False
                        bear_ok  = False
                    else:
                        # No filter — always passes
                        bull_ok  = True
                        bear_ok  = True

                    if entry_t==0:
                        # E1: enter on reversal bar; body must confirm new column direction
                        if did_reverse and prev_col_at_rev>=n_min:
                            body_ok = bull_ok if pnf_dir==1 else bear_ok
                            if body_ok:
                                pos=pnf_dir; entry_px=cl; hw_level=pnf_level
                    else:
                        # E2: set pending on qualifying reversal (no body check yet)
                        if did_reverse and prev_col_at_rev>=n_min: pending=pnf_dir
                        # Clear pending on adverse reversal
                        if did_reverse and pending!=0 and pnf_dir!=pending: pending=0
                        # Enter when confirmation fires AND body check passes.
                        # If body fails: stay pending — retry next bar.
                        if pending!=0 and pnf_dir==pending and col_count>rev:
                            body_ok = bull_ok if pending==1 else bear_ok
                            if body_ok:
                                pos=pending; entry_px=cl; hw_level=pnf_level; pending=0
                else:
                    if did_reverse and pending!=0 and pnf_dir!=pending: pending=0

        trade_cnt[ci]=t_cnt


# ─── Validation pipeline ──────────────────────────────────────────────────────
def mc_permutation_test(pnl_arr, n_shuffles=1000, seed=42):
    rng = np.random.default_rng(seed)
    actual = float(pnl_arr.sum())
    sums = np.array([(np.abs(pnl_arr)*rng.choice([-1.,1.],size=len(pnl_arr))).sum()
                     for _ in range(n_shuffles)])
    return float((sums>=actual).mean()), float(np.percentile(sums, 95))

def bootstrap_p5(pnl_arr, days_oos=1, n_boot=2000, seed=99):
    rng = np.random.default_rng(seed)
    n = len(pnl_arr)
    if n==0: return 0.0
    sums = np.array([float(pnl_arr[rng.integers(0,n,size=n)].sum()) for _ in range(n_boot)])
    return float(np.percentile(sums, 5)) / max(days_oos, 1)


def run_pair(pair, pip):
    ba_path = BA_DIR / f"{pair}_M5_BA.parquet"
    if not ba_path.exists():
        return None, f"No BA data: {ba_path}"

    ba = pd.read_parquet(ba_path)
    n  = len(ba)
    if n < 200_000:
        return None, f"{pair}: only {n:,} bars"

    is_end   = int(n * IS_FRAC)
    oos_days = (n - is_end) / 288
    is_days  = is_end / 288

    opens   = ba["open"].values.astype(np.float64)
    highs   = ba["high"].values.astype(np.float64)
    lows    = ba["low"].values.astype(np.float64)
    closes  = ba["close"].values.astype(np.float64)
    spreads = ((ba["ask_c"]-ba["bid_c"])/pip).values.astype(np.float64)

    sp_gate = float(np.percentile(spreads[:is_end], 90))

    c0e=is_end//3; c1e=2*(is_end//3)
    bar_chunks=np.zeros(n, dtype=np.int8)
    bar_chunks[c0e:c1e]=1; bar_chunks[c1e:is_end]=2; bar_chunks[is_end:]=3

    trade_pnl   = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.float32)
    trade_chunk = np.zeros((N_CONFIGS, MAX_TRADES), dtype=np.int8)
    trade_cnt   = np.zeros(N_CONFIGS, dtype=np.int32)

    print(f"  {pair}: {n:,} bars  IS={int(is_days)}d  OOS={int(oos_days)}d  sp_gate={sp_gate:.2f}p")
    t0 = time.time()
    run_kernel(opens,highs,lows,closes,spreads,bar_chunks,
               CONFIGS, sp_gate, pip, is_end,
               trade_pnl, trade_chunk, trade_cnt)
    elapsed = time.time()-t0
    print(f"  Kernel: {elapsed:.1f}s  trades_total={int(trade_cnt.sum()):,}")

    # Stage 1: IS walk-forward (all 3 chunks profitable, ≥30 IS trades)
    s1 = []
    for ci in range(N_CONFIGS):
        tc=trade_cnt[ci]
        if tc==0: continue
        pnl=trade_pnl[ci,:tc].astype(np.float64)
        ck=trade_chunk[ci,:tc].astype(np.int32)
        is_mask=ck<=2
        if is_mask.sum()<30: continue
        c0s=pnl[ck==0].sum(); c1s=pnl[ck==1].sum(); c2s=pnl[ck==2].sum()
        if c0s<=0 or c1s<=0 or c2s<=0: continue
        bf = int(CONFIGS[ci, 7])
        s1.append({"ci":ci, "name":CONFIG_NAMES[ci], "bf":bf,
                   "is_pnl":round(float(pnl[is_mask].sum()),1),
                   "is_ntrd":int(is_mask.sum()),
                   "c0":round(float(c0s),1), "c1":round(float(c1s),1), "c2":round(float(c2s),1)})

    if not s1:
        return {"pair":pair,"stage1":0,"stage2":0,"stage3":0,
                "top_name":"none","top_oos_pd":0,"top_bf":0,
                "is_days":int(is_days),"oos_days":int(oos_days),
                "sp_gate":sp_gate,"elapsed":elapsed}, None

    s1_df = pd.DataFrame(s1).sort_values("is_pnl", ascending=False)

    # Stage 2: MC + bootstrap (top 200)
    s2 = []
    for _,row in s1_df.head(200).iterrows():
        ci=int(row["ci"]); tc=trade_cnt[ci]
        pnl=trade_pnl[ci,:tc].astype(np.float64)
        ck=trade_chunk[ci,:tc].astype(np.int32)
        is_pnl=pnl[ck<=2]
        pv,_ = mc_permutation_test(is_pnl)
        p5   = bootstrap_p5(is_pnl, days_oos=oos_days)
        s2.append({"ci":ci,"name":row["name"],"bf":row["bf"],
                   "is_pnl":row["is_pnl"],"is_ntrd":row["is_ntrd"],
                   "mc_pval":round(pv,3), "bootstrap_p5":round(p5,2),
                   "passed_mc":int(pv<0.05)})
    s2_df = pd.DataFrame(s2)

    # Stage 3: OOS (sealed — R8)
    s3 = []
    for _,row in s2_df[s2_df.passed_mc==1].iterrows():
        ci=int(row["ci"]); tc=trade_cnt[ci]
        pnl=trade_pnl[ci,:tc].astype(np.float64)
        ck=trade_chunk[ci,:tc].astype(np.int32)
        oos_pnl=pnl[ck==3]
        oos_tot=float(oos_pnl.sum())
        oos_ntrd=len(oos_pnl)
        oos_pd=oos_tot/max(oos_days,1)
        oos_pass=int(oos_tot>0 and oos_ntrd>=10)
        s3.append({**row.to_dict(),
                   "oos_pnl":round(oos_tot,1),"oos_ntrd":oos_ntrd,
                   "oos_pd":round(oos_pd,2),"oos_pass":oos_pass})
    s3_df = pd.DataFrame(s3).sort_values("oos_pd",ascending=False) if s3 else pd.DataFrame()

    # Save per-pair CSVs
    slug = pair.lower().replace("_","")
    s1_df.to_csv(OUT_DIR/f"{slug}_body_stage1.csv", index=False)
    s2_df.to_csv(OUT_DIR/f"{slug}_body_stage2.csv", index=False)
    if len(s3_df):
        s3_df.to_csv(OUT_DIR/f"{slug}_body_final.csv", index=False)

    winners = s3_df[s3_df.oos_pass==1] if len(s3_df) else pd.DataFrame()
    top_name = winners.iloc[0]["name"]  if len(winners) else "none"
    top_pd   = winners.iloc[0]["oos_pd"] if len(winners) else 0.0
    top_bf   = int(winners.iloc[0]["bf"])  if len(winners) else 0

    # Body-frac tier breakdown (among OOS winners)
    tier_lines = []
    for bf_val in [0, 3, 5, 7]:
        if len(winners):
            sub = winners[winners.bf==bf_val]
        else:
            sub = pd.DataFrame()
        n_w  = len(sub)
        best = sub.iloc[0]["oos_pd"] if n_w>0 else 0.0
        avg_t= sub["oos_ntrd"].mean() if n_w>0 else 0
        tier_lines.append(f"  bf={bf_val/10:.1f}: {n_w:3d} winners  best={best:.1f}p/d  avg_trades={avg_t:.0f}")

    summary = {
        "pair":pair,"n_bars":n,"is_days":int(is_days),"oos_days":int(oos_days),
        "sp_gate":round(sp_gate,2),"elapsed":round(elapsed,1),
        "stage1":len(s1_df),"stage2":int(s2_df.passed_mc.sum()),
        "stage3":int(winners.oos_pass.sum()) if len(winners) else 0,
        "top_name":top_name,"top_oos_pd":top_pd,"top_bf":top_bf,
    }
    return summary, "\n".join(tier_lines)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("FIFO-Trends Body Filter Experiment")
    print(f"Configs: {N_CONFIGS:,}  (2700 base × 4 body_frac values)")
    print(f"Pairs:   {[p for p,_ in TARGET_PAIRS]}")
    print("="*60)

    # JIT warmup
    print("\nJIT warmup...")
    ba0 = pd.read_parquet(BA_DIR / "EUR_JPY_M5_BA.parquet")
    o=ba0["open"].values[:500].astype(np.float64)
    h=ba0["high"].values[:500].astype(np.float64)
    l=ba0["low"].values[:500].astype(np.float64)
    c=ba0["close"].values[:500].astype(np.float64)
    sp=((ba0["ask_c"]-ba0["bid_c"])/0.01).values[:500].astype(np.float64)
    bch=np.zeros(500,dtype=np.int8)
    dp=np.zeros((1,MAX_TRADES),dtype=np.float32)
    dc=np.zeros((1,MAX_TRADES),dtype=np.int8)
    dk=np.zeros(1,dtype=np.int32)
    t0=time.time()
    run_kernel(o,h,l,c,sp,bch,CONFIGS[:1],2.5,0.01,350,dp,dc,dk)
    print(f"  Compiled in {time.time()-t0:.1f}s")
    tg(f"🔬 FIFO body filter experiment starting — {N_CONFIGS:,} configs × {len(TARGET_PAIRS)} pairs")

    all_summaries = []
    for pair, pip in TARGET_PAIRS:
        print(f"\n{'='*50}")
        print(f"Pair: {pair}  pip={pip}")
        summary, tier_info = run_pair(pair, pip)

        if summary is None:
            print(f"  SKIP: {tier_info}")
            tg(f"⚠️ {pair}: {tier_info}")
            continue

        all_summaries.append(summary)

        icon = "🟢" if summary["top_oos_pd"]>=10 else ("🟡" if summary["top_oos_pd"]>0 else "🔴")
        msg = (
            f"{icon} {pair} body-filter results\n"
            f"Stage1={summary['stage1']}  Stage2={summary['stage2']}  Stage3={summary['stage3']}\n"
            f"Best: {summary['top_name']}\n"
            f"OOS: {summary['top_oos_pd']:.1f} p/d  (bf={summary['top_bf']/10:.1f})\n"
            f"OOS period: {summary['oos_days']}d\n"
            f"\nBy body_frac:\n{tier_info}\n"
            f"Time: {summary['elapsed']:.0f}s"
        )
        print(msg)
        tg(msg)

    # Final summary table
    if all_summaries:
        df = pd.DataFrame(all_summaries).sort_values("top_oos_pd", ascending=False)
        df.to_csv(OUT_DIR/"body_filter_summary.csv", index=False)

        lines = ["\n📊 FIFO Body Filter — 4-pair summary\n"]
        for _,r in df.iterrows():
            icon="🟢" if r.top_oos_pd>=10 else ("🟡" if r.top_oos_pd>0 else "🔴")
            lines.append(f"{icon} {r.pair:<10} {r.top_oos_pd:>7.1f} p/d  "
                         f"bf={r.top_bf/10:.1f}  winners={int(r.stage3):>3}  {r.top_name}")
        final_msg = "\n".join(lines)
        print(final_msg)
        tg(final_msg)

    print(f"\nResults in {OUT_DIR}")


if __name__ == "__main__":
    main()
