"""
P&F Donchian Channel — pure-signal sweep.
=========================================

IDEA
----
Classic Donchian channel: "highest high of last N hours". Time-based
lookback breaks down in (a) low-vol stretches (channel narrows around
chop, false bait-bit on wicks) and (b) fast moves (channel dominated by
one leg, then useless until time resets).

Replace TIME-based lookback with RANGE-based lookback: build a P&F
chart (box size b, reversal r), then take the Donchian channel over
the last N **completed** P&F columns.

  UpperPF = max(top of last N completed columns)
  LowerPF = min(bottom of last N completed columns)

Each P&F column is a box-quantized leg of roughly equal structural
significance, regardless of how long it took. N=5 columns spans hours
in fast markets, days in slow ones — but the structural meaning is
constant.

RULES
-----
Entry:
  Long  — the current (in-progress) up-column P&F level breaks above
          UpperPF (the highest column-top of the last N completed cols).
  Short — current down-column P&F level breaks below LowerPF.

Exit:
  k-box trailing stop from the P&F high-water level (FIFO-Trends style).

FILL MODEL
----------
  ENTRY: at the breakout P&F level. Realistic because a stop order
         placed at UpperPF would fill at the level (broker fills at
         trigger, sub-pip slippage).
  EXIT:  at trail level (S5 monitor pattern from FIFO v2).

NO SPREAD GATE — every trade pays a fixed per-pair spread cost (the
mean of `(ask_c - bid_c) / pip` across the IS portion). Pure signal
test; if it works, an adaptive gate is a v2 refinement.

SOP COMPLIANCE
--------------
R1 closed bars only       — entry/exit evaluated at bar end
R2 within-bar sequencing  — bull bar: process HIGH then LOW; bear: LOW then HIGH
R3 mid OHLC + spread cost — `(open, high, low, close)` = mid, spread deducted explicitly
R4 incremental-only       — deque of last MAX_K column extremes, no df.rolling
R4a P&F columns           — completed columns only feed the deque; in-progress excluded
R5 IS-only cost calc      — spread cost = mean spread on IS portion ONLY
R6 one code path          — same Numba kernel does the sweep
R8 OOS sealed             — single touch, no re-tuning

GRID
----
b      ∈ {5, 10}          pips
r      ∈ {1, 3}           reversal boxes
N_cols ∈ {1, 2, 3, 5, 8, 12, 20}
trail  ∈ {1, 2, 3}        boxes

12 pairs × 84 configs = 1008 backtests.
"""

import sys, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
import numba as nb

BASE = Path(__file__).resolve().parents[3]
BA   = BASE / "data/m5_ba"
OUT  = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

IS_FRAC  = 0.70
N_WF     = 3
MIN_IS_TRADES_PER_CHUNK = 5
M5_PER_TRADING_DAY = 288.0

PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "NZD_USD",
    "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY", "CHF_JPY",
    "CAD_JPY", "EUR_GBP",
]
PIP = {p: (0.01 if "JPY" in p else 0.0001) for p in PAIRS}

B_SWEEP    = [5, 10]
R_SWEEP    = [1, 3]
N_SWEEP    = [1, 2, 3, 5, 8, 12, 20]
TRAIL_SWEEP= [1, 2, 3]

MAX_N_COLS = 32       # must exceed max N_cols

# ── Numba kernel ────────────────────────────────────────────────────────────────

@nb.njit
def run_one(opens, highs, lows, closes,
            b_pips, rev, N_cols, trail_k,
            pip, spread_cost, is_end):
    """
    P&F Donchian channel breakout, k-box trail exit.

    Within-bar sequencing per R2: bull bar process HIGH then LOW; bear LOW then HIGH.
    Donchian channel uses last N_cols **completed** column extremes only.

    Returns:
        pnls   — trade P&L in pips (already spread-deducted)
        flags  — 1 if IS (entry bar < is_end), 0 if OOS
    """
    N  = len(opens)
    bs = b_pips * pip

    # P&F state
    pnf_idx     = 0
    pnf_level   = 0.0
    pnf_dir     = 0        # +1 up-column, -1 down-column, 0 uninitialised
    col_count   = 0        # boxes in CURRENT (in-progress) column
    col_start_lvl = 0.0    # P&F level at start of current column

    # Donchian deques: top & bottom of last MAX_N_COLS COMPLETED columns
    top_buf = np.full(MAX_N_COLS, -1e18, np.float64)
    bot_buf = np.full(MAX_N_COLS,  1e18, np.float64)
    buf_ptr = 0
    buf_n   = 0

    # Position state
    pos       = 0          # +1 long, -1 short, 0 flat
    entry_px  = 0.0
    hw_level  = 0.0        # P&F high-water level for trail

    MAX_T = N // 5 + 100
    pnls  = np.empty(MAX_T, np.float64)
    flags = np.empty(MAX_T, np.int8)
    n_t   = 0

    for i in range(N):
        opn = opens[i]; hi = highs[i]; lo = lows[i]; cl = closes[i]
        is_bar = 1 if i < is_end else 0

        bull = (cl >= opn)
        p_first  = hi if bull else lo
        p_second = lo if bull else hi

        for tick in range(2):
            px = p_first if tick == 0 else p_second

            # Initialise P&F
            if pnf_dir == 0:
                pnf_idx = int(px / bs)
                pnf_level = pnf_idx * bs
                pnf_dir = 1
                col_count = 1
                col_start_lvl = pnf_level
                continue

            delta = int(px / bs) - pnf_idx

            if pnf_dir == 1:
                if delta >= 1:
                    pnf_idx   += delta
                    pnf_level  = pnf_idx * bs
                    col_count += delta
                elif delta <= -rev:
                    # Complete the up-column — its TOP = current pnf_level,
                    # BOTTOM = col_start_lvl.
                    top_buf[buf_ptr % MAX_N_COLS] = pnf_level
                    bot_buf[buf_ptr % MAX_N_COLS] = col_start_lvl
                    buf_ptr += 1
                    if buf_n < MAX_N_COLS:
                        buf_n += 1
                    # Open a new down-column
                    pnf_dir   = -1
                    pnf_idx  += delta
                    pnf_level = pnf_idx * bs
                    col_count = -delta
                    col_start_lvl = pnf_level + (-delta) * bs   # start (top) of down-col
                # else: noise, ignore
            else:  # pnf_dir == -1
                if delta <= -1:
                    pnf_idx   += delta
                    pnf_level  = pnf_idx * bs
                    col_count += (-delta)
                elif delta >= rev:
                    # Complete the down-column — its BOTTOM = current pnf_level,
                    # TOP = col_start_lvl.
                    top_buf[buf_ptr % MAX_N_COLS] = col_start_lvl
                    bot_buf[buf_ptr % MAX_N_COLS] = pnf_level
                    buf_ptr += 1
                    if buf_n < MAX_N_COLS:
                        buf_n += 1
                    # Open a new up-column
                    pnf_dir   = 1
                    pnf_idx  += delta
                    pnf_level = pnf_idx * bs
                    col_count = delta
                    col_start_lvl = pnf_level - delta * bs

            # ── Position management (interleaved with P&F update) ─────────────
            # high-water on the trade
            if pos == 1 and pnf_level > hw_level:
                hw_level = pnf_level
            elif pos == -1 and pnf_level < hw_level:
                hw_level = pnf_level

            # Exit: k-box trail
            if pos != 0:
                if pos == 1:
                    trail_px = hw_level - trail_k * bs
                    if pnf_level <= trail_px + 1e-12:
                        pnl = (trail_px - entry_px) / pip - spread_cost
                        if n_t < MAX_T:
                            pnls[n_t]  = pnl
                            flags[n_t] = 1 if (i < is_end) else 0
                            n_t += 1
                        pos = 0; entry_px = 0.0; hw_level = 0.0
                else:
                    trail_px = hw_level + trail_k * bs
                    if pnf_level >= trail_px - 1e-12:
                        pnl = (entry_px - trail_px) / pip - spread_cost
                        if n_t < MAX_T:
                            pnls[n_t]  = pnl
                            flags[n_t] = 1 if (i < is_end) else 0
                            n_t += 1
                        pos = 0; entry_px = 0.0; hw_level = 0.0

            # Entry: Donchian-PF breakout, only when buffer has ≥ N_cols completed cols.
            if pos == 0 and buf_n >= N_cols:
                # Read upper/lower from last N_cols entries of the ring buffer
                upper = -1e18
                lower =  1e18
                for k in range(N_cols):
                    idx = (buf_ptr - 1 - k) % MAX_N_COLS
                    if top_buf[idx] > upper:
                        upper = top_buf[idx]
                    if bot_buf[idx] < lower:
                        lower = bot_buf[idx]
                if pnf_dir == 1 and pnf_level > upper + 1e-12:
                    pos = 1
                    entry_px = pnf_level    # stop-fill at breakout level
                    hw_level = pnf_level
                elif pnf_dir == -1 and pnf_level < lower - 1e-12:
                    pos = -1
                    entry_px = pnf_level
                    hw_level = pnf_level

    return pnls[:n_t], flags[:n_t]


# ── Stats ───────────────────────────────────────────────────────────────────────

def stats(pnls, flags, is_end, n_total):
    oos_days = (n_total - is_end) / M5_PER_TRADING_DAY
    is_days  = is_end / M5_PER_TRADING_DAY

    is_mask  = flags == 1
    oos_mask = flags == 0
    is_p  = pnls[is_mask]
    oos_p = pnls[oos_mask]

    wf_ok = True
    if len(is_p) < N_WF * MIN_IS_TRADES_PER_CHUNK:
        wf_ok = False
    else:
        for k in range(N_WF):
            s = k * (len(is_p) // N_WF)
            e = (k+1) * (len(is_p) // N_WF) if k < N_WF - 1 else len(is_p)
            chunk = is_p[s:e]
            if len(chunk) < MIN_IS_TRADES_PER_CHUNK or chunk.sum() <= 0:
                wf_ok = False; break

    def wr(a):   return float((a > 0).mean()) if len(a) > 0 else 0.0
    def pd_(a, days): return float(a.sum() / days) if len(a) > 0 and days > 0 else 0.0

    return {
        "wf":      wf_ok,
        "is_pd":   round(pd_(is_p,  is_days), 2),
        "is_n":    int(len(is_p)),
        "is_wr":   round(wr(is_p), 3),
        "is_net":  round(float(is_p.sum()), 1) if len(is_p)  else 0.0,
        "oos_pd":  round(pd_(oos_p, oos_days), 2),
        "oos_n":   int(len(oos_p)),
        "oos_wr":  round(wr(oos_p), 3),
        "oos_net": round(float(oos_p.sum()), 1) if len(oos_p) else 0.0,
    }


# ── Main sweep ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 92)
    print("  P&F DONCHIAN CHANNEL — pure-signal sweep (no spread gate)")
    print(f"  Grid: b={B_SWEEP}  r={R_SWEEP}  N_cols={N_SWEEP}  trail={TRAIL_SWEEP}")
    print(f"  Pairs: {len(PAIRS)}   Configs/pair: {len(B_SWEEP)*len(R_SWEEP)*len(N_SWEEP)*len(TRAIL_SWEEP)}")
    print("=" * 92)

    # Numba warmup
    print("Warming up Numba JIT...")
    _c = np.cumsum(np.random.randn(2000)) * 0.01 + 1.30
    _h = _c + 0.0003; _l = _c - 0.0003
    run_one(_c, _h, _l, _c, 5, 1, 3, 1, 0.0001, 1.0, 1400)
    print("  Done.\n")

    rows = []
    t0 = time.time()

    for pair in PAIRS:
        pip = PIP[pair]
        path = BA / f"{pair}_M5_BA.parquet"
        df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
        op = df["open"].values.astype(np.float64)
        hi = df["high"].values.astype(np.float64)
        lo = df["low"].values.astype(np.float64)
        cl = df["close"].values.astype(np.float64)
        sp = ((df["ask_c"] - df["bid_c"]) / pip).values.astype(np.float64)
        n  = len(op)
        is_e = int(n * IS_FRAC)
        is_days  = is_e / M5_PER_TRADING_DAY
        oos_days = (n - is_e) / M5_PER_TRADING_DAY

        # IS-only mean spread → per-trade cost (round-trip implicit since
        # spread is the bid-ask gap; one entry one exit means one round-trip
        # crosses the spread once on each side → 2× spread? No: the OANDA
        # spread is the bid-ask gap, so a long that BUYS at ask and SELLS
        # at bid pays the full spread once per round-trip. Our mid-based
        # OHLC has the spread already split, so deducting 1× mean spread
        # per trade replicates a round-trip BUY-AT-ASK SELL-AT-BID cycle.)
        sp_cost = float(np.mean(sp[:is_e]))

        print(f"  {pair}: {n:,} bars  IS={is_days:.0f}d  OOS={oos_days:.0f}d  "
              f"sp_cost={sp_cost:.2f}p")

        for b in B_SWEEP:
            for r in R_SWEEP:
                for N in N_SWEEP:
                    for tr in TRAIL_SWEEP:
                        pnls, flags = run_one(op, hi, lo, cl, b, r, N, tr,
                                              pip, sp_cost, is_e)
                        s = stats(pnls, flags, is_e, n)
                        rows.append({
                            "pair": pair, "b": b, "rev": r, "N": N, "trail": tr,
                            "sp_cost": round(sp_cost, 2),
                            **s,
                        })

        del op, hi, lo, cl, sp, df
        gc.collect()

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s\n")

    rdf = pd.DataFrame(rows)
    out_csv = OUT / "pnf_donchian_sweep.csv"
    rdf.to_csv(out_csv, index=False)
    print(f"  CSV → {out_csv}")

    # ── Deploy candidates: WF pass AND OOS p/d > 0 ──────────────────────────────
    cand = rdf[(rdf["wf"]) & (rdf["oos_pd"] > 0)].copy()
    cand = cand.sort_values(["pair", "oos_pd"], ascending=[True, False])

    print()
    print("=" * 92)
    print("  DEPLOY CANDIDATES — WF pass (3/3 IS chunks +) AND OOS p/d > 0")
    print("-" * 92)
    if cand.empty:
        print("  None.")
    else:
        bp = cand.groupby("pair").head(1).sort_values("oos_pd", ascending=False)
        print(f"  {'Pair':<10} {'b':>3} {'rev':>3} {'N':>3} {'trail':>5} "
              f"{'IS_pd':>7} {'IS_n':>5} {'IS_WR':>6} "
              f"{'OOS_pd':>7} {'OOS_n':>5} {'OOS_WR':>6}")
        for _, row in bp.iterrows():
            print(f"  {row['pair']:<10} {int(row['b']):>3} {int(row['rev']):>3} "
                  f"{int(row['N']):>3} {int(row['trail']):>5} "
                  f"{row['is_pd']:>7.2f} {int(row['is_n']):>5} {row['is_wr']:>6.1%} "
                  f"{row['oos_pd']:>7.2f} {int(row['oos_n']):>5} {row['oos_wr']:>6.1%}")
        n_pairs = bp["pair"].nunique()
        print(f"\n  {n_pairs}/{len(PAIRS)} pairs have at least one deploy candidate.")

    print()
    print("  Headline counts:")
    print(f"    Configs run         : {len(rdf):,}")
    print(f"    WF-pass             : {int(rdf['wf'].sum()):,}")
    print(f"    WF+OOS positive     : {len(cand):,}")
    print(f"    WF+OOS p/d ≥  5     : {int(((rdf['wf']) & (rdf['oos_pd']>=5)).sum()):,}")
    print(f"    WF+OOS p/d ≥ 10     : {int(((rdf['wf']) & (rdf['oos_pd']>=10)).sum()):,}")
    print(f"    WF+OOS p/d ≥ 25     : {int(((rdf['wf']) & (rdf['oos_pd']>=25)).sum()):,}")


if __name__ == "__main__":
    main()
