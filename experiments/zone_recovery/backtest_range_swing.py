"""
Exp 2 — Range-Bound Swing (No ZR, FIFO)
=========================================
Hypothesis: in a confirmed trading range (no new swing highs/lows), buying at
confirmed multi-touch support (stacked sell-side liquidity wall) and selling at
confirmed multi-touch resistance (stacked buy-side liquidity wall) captures
mean-reversion edge — WITHOUT zone recovery legs.

Key concepts from the liquidity transcript:
  - Stacked liquidity wall = multiple swing highs/lows at same level → stronger S/R
  - Wick rejection at the wall = absorption > aggression → entry signal
  - Body close through the wall = run/sweep → stop loss exits
  - Range regime = price oscillates between confirmed walls (no new extremes)

Strategy mechanics:
  - LONG: enter when bar LOW touches confirmed support (N-touch swing low cluster)
           AND bar close shows rejection wick (lower wick > wick_thresh)
  - SHORT: enter when bar HIGH touches confirmed resistance (N-touch swing high cluster)
           AND bar close shows upper wick rejection
  - Target: fixed pip distance OR opposite confirmed S/R level (whichever closer)
  - Stop: bar CLOSE below support (for LONG) = wall broken = run, not sweep
  - FIFO: only one position at a time (011=LONG, 012=SHORT in live; here combined)
  - Range filter: require no new swing high/low in last range_bars (confirms range regime)

Parameter grid:
  swing_window  : bars each side for swing detection {3, 5, 7}
  cluster_tol   : pip tolerance to cluster swings as "same level" {5, 10, 15}
  min_touches   : minimum swing touches to qualify as stacked wall {2, 3}
  wick_thresh   : rejection wick fraction required {0.0, 0.15, 0.25, 0.35}
  stop_atr_frac : stop = entry_level ± stop_atr_frac × ATR20 {0.5, 1.0, 1.5}
  range_bars    : lookback to confirm no new swing extremes {0(off), 50, 100}
  tgt_pips      : fixed pip target distance {15, 20, 25, 30} (or 0 = use opp S/R)

Pairs: EUR_USD (primary), EUR_JPY, GBP_USD
Gates: IS=3/3, OOS=3/3, P5>0, P(+)>95%, permutation p<0.05

Output: results/range_swing_results.csv
"""
import math, sys, itertools
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA_DIR_MID = Path('/path/to/projects/fx-core/data/m5_ohlc')
DATA_DIR_BA  = Path('/path/to/projects/fx-core/data/m5_ba')
OUT_PATH     = Path(__file__).parent / 'results/range_swing_results.csv'
OUT_PATH.parent.mkdir(exist_ok=True)

OOS_FRAC   = 0.30
IS_CHUNKS  = 3
OOS_CHUNKS = 3
N_BOOT     = 500
N_PERM     = 200

PAIRS = [
    ("EUR_USD", 0.0001, 1.6),  # pair, pip, median_spread_pips
    ("EUR_JPY", 0.01,   2.3),
    ("GBP_USD", 0.0001, 1.9),
]

# ── Parameter grid ────────────────────────────────────────────────────────────
SWING_WINDOWS = [3, 5, 7]       # bars each side for swing point detection
CLUSTER_TOLS  = [5, 10, 15]     # pips to cluster swings as same level
MIN_TOUCHES   = [2, 3]          # min touch count for stacked wall
WICK_THRESHS  = [0.0, 0.15, 0.25, 0.35]   # wick rejection threshold
STOP_ATR_FRACS= [0.5, 1.0, 1.5]  # stop distance as fraction of ATR20
RANGE_BARS    = [0, 50, 100]    # 0=off; N=require no new swing in last N bars
TGT_PIPS      = [15, 20, 25, 30]   # fixed target in pips (0=use opp S/R level)
# Total: 3×3×2×4×3×3×4 = 2,592 — reduce to key combos via Cartesian product
# Primary sweep: swing_window=5, cluster_tol=10, range_bars=50 fixed, tgt_pips=20
# → 1×4×3×4 = 48 per pair (fast first pass)
# Full grid after seeing first-pass winners.


# ── Swing point detection (non-JIT, precomputed) ─────────────────────────────
def find_swings(hi, lo, window):
    """
    Simple swing detection: bar i is a swing high if hi[i] > all hi in [i-w, i-1]
    and > all hi in [i+1, i+w]. Similar for swing lows.
    Returns (s_hi, s_lo) boolean arrays (True at swing point bars).
    Note: last `window` bars cannot be confirmed.
    """
    n = len(hi)
    s_hi = np.zeros(n, dtype=bool)
    s_lo = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        if hi[i] >= hi[i-window:i].max() and hi[i] >= hi[i+1:i+window+1].max():
            s_hi[i] = True
        if lo[i] <= lo[i-window:i].min() and lo[i] <= lo[i+1:i+window+1].min():
            s_lo[i] = True
    return s_hi, s_lo


def build_sr_history(hi, lo, s_hi, s_lo, pip, cluster_tol, min_touches, lookback):
    """
    For each bar i, find the nearest confirmed support and resistance levels
    that have min_touches within cluster_tol pips in the last `lookback` bars.

    Returns:
      support_lvl[i]    : nearest stacked support below close[i], or NaN
      support_touches[i]: touch count for that level
      resist_lvl[i]     : nearest stacked resistance above close[i], or NaN
      resist_touches[i] : touch count
    """
    n = len(hi)
    tol = cluster_tol * pip
    support_lvl    = np.full(n, np.nan)
    support_touches= np.zeros(n, dtype=int)
    resist_lvl     = np.full(n, np.nan)
    resist_touches = np.zeros(n, dtype=int)
    cl = (hi + lo) / 2.0   # midpoint proxy for current price position

    for i in range(lookback, n):
        start = max(0, i - lookback)
        # Collect swing highs and lows in the lookback window
        sh_levels = hi[start:i][s_hi[start:i]]
        sl_levels = lo[start:i][s_lo[start:i]]
        cur_price = cl[i]

        # Cluster swing lows → supports below current price
        best_supp = np.nan; best_supp_touches = 0
        for level in sl_levels:
            if level >= cur_price: continue   # not below price
            # Count how many swing lows cluster at this level
            touches = int(((sl_levels >= level - tol) & (sl_levels <= level + tol)).sum())
            if touches >= min_touches:
                # Take nearest support (highest below current price)
                if np.isnan(best_supp) or level > best_supp:
                    best_supp = level; best_supp_touches = touches
        support_lvl[i]     = best_supp
        support_touches[i] = best_supp_touches

        # Cluster swing highs → resistance above current price
        best_res = np.nan; best_res_touches = 0
        for level in sh_levels:
            if level <= cur_price: continue
            touches = int(((sh_levels >= level - tol) & (sh_levels <= level + tol)).sum())
            if touches >= min_touches:
                if np.isnan(best_res) or level < best_res:
                    best_res = level; best_res_touches = touches
        resist_lvl[i]     = best_res
        resist_touches[i] = best_res_touches

    return support_lvl, support_touches, resist_lvl, resist_touches


def range_regime_mask(s_hi_arr, s_lo_arr, hi, lo, range_bars):
    """
    Returns boolean mask: True at bar i if no NEW swing high above historical max
    and no NEW swing low below historical min in the last range_bars.
    Disabled if range_bars == 0.
    """
    n = len(hi)
    if range_bars == 0:
        return np.ones(n, dtype=bool)
    in_range = np.zeros(n, dtype=bool)
    for i in range(range_bars, n):
        start = max(0, i - range_bars)
        # In range = current swing highs are NOT new (no breakout of prior range)
        prev_hi_max = hi[start:i-range_bars//2].max() if i > range_bars//2 else hi[start:i].max()
        prev_lo_min = lo[start:i-range_bars//2].min() if i > range_bars//2 else lo[start:i].min()
        cur_sh = s_hi_arr[i]
        cur_sl = s_lo_arr[i]
        if cur_sh and hi[i] > prev_hi_max:
            in_range[i] = False   # new high = potential breakout, not range
        elif cur_sl and lo[i] < prev_lo_min:
            in_range[i] = False
        else:
            in_range[i] = True
    return in_range


@njit
def sim_range_swing(op, hi, lo, cl, sp_arr, pip,
                    support_lvl, support_touches, resist_lvl, resist_touches,
                    in_range_arr,
                    wick_thresh, stop_atr_frac, tgt_pips_fixed,
                    atr_window=20):
    """
    Range-bound swing sim (no ZR legs).
    LONG: enter when low touches support + wick rejection. Exit at target or stop.
    SHORT: enter when high touches resistance + wick rejection. Exit at target or stop.
    One position at a time (FIFO).

    Returns: (pnl array, exit_type array, nc)
    exit_type: 1=target hit, 2=stop hit
    """
    n    = len(cl)
    pnl  = np.zeros(n, dtype=np.float64)
    etype= np.zeros(n, dtype=np.int32)
    nc   = 0

    # Rolling ATR (causal)
    hl_pips = np.zeros(atr_window, dtype=np.float64)
    atr_buf_n = 0

    i = 0
    in_trade  = False
    trade_dir = 0     # +1 LONG, -1 SHORT
    trade_entry = 0.0
    trade_stop  = 0.0
    trade_tgt   = 0.0

    while i < n:
        rng = hi[i] - lo[i]
        rng_pips = rng / pip

        # Update ATR buffer
        hl_pips[atr_buf_n % atr_window] = rng_pips
        atr_buf_n += 1
        atr20 = 0.0
        filled = min(atr_buf_n, atr_window)
        for k in range(filled): atr20 += hl_pips[k]
        atr20 /= max(filled, 1)

        sp = sp_arr[i]

        if in_trade:
            # ── Check exit conditions ─────────────────────────────────────
            if trade_dir == 1:   # LONG
                # Target hit: high touched target
                if hi[i] >= trade_tgt:
                    net = (trade_tgt - trade_entry) / pip - sp
                    pnl[nc] = net; etype[nc] = 1; nc += 1; in_trade = False
                # Stop hit: close below stop
                elif cl[i] < trade_stop:
                    net = (trade_stop - trade_entry) / pip - sp
                    pnl[nc] = net; etype[nc] = 2; nc += 1; in_trade = False
            else:                # SHORT
                if lo[i] <= trade_tgt:
                    net = (trade_entry - trade_tgt) / pip - sp
                    pnl[nc] = net; etype[nc] = 1; nc += 1; in_trade = False
                elif cl[i] > trade_stop:
                    net = (trade_entry - trade_stop) / pip - sp
                    pnl[nc] = net; etype[nc] = 2; nc += 1; in_trade = False
            i += 1
            continue

        # ── Entry conditions (only when flat) ─────────────────────────────
        if not in_range_arr[i]:
            i += 1; continue
        if np.isnan(support_lvl[i]) and np.isnan(resist_lvl[i]):
            i += 1; continue

        lo_body = cl[i] if cl[i] < op[i] else op[i]
        hi_body = cl[i] if cl[i] > op[i] else op[i]

        # LONG entry: low touches support, wick rejection upward
        supp = support_lvl[i]
        if not np.isnan(supp):
            touch_zone = supp + atr20 * 0.3 * pip   # within 30% ATR of level
            if lo[i] <= touch_zone:
                lower_wick = (lo_body - lo[i]) / (rng + 1e-10)
                if lower_wick >= wick_thresh:
                    entry_price = cl[i]
                    stop_price  = supp - stop_atr_frac * atr20 * pip
                    if tgt_pips_fixed > 0:
                        tgt_price = entry_price + tgt_pips_fixed * pip
                    else:
                        # Use nearest resistance as target
                        res = resist_lvl[i]
                        tgt_price = (res - sp*pip) if not np.isnan(res) else entry_price + 20*pip
                    if tgt_price > entry_price + sp*pip:   # viable trade
                        in_trade   = True
                        trade_dir  = 1
                        trade_entry= entry_price
                        trade_stop = stop_price
                        trade_tgt  = tgt_price
                        i += 1; continue

        # SHORT entry: high touches resistance, wick rejection downward
        res = resist_lvl[i]
        if not np.isnan(res):
            touch_zone = res - atr20 * 0.3 * pip
            if hi[i] >= touch_zone:
                upper_wick = (hi[i] - hi_body) / (rng + 1e-10)
                if upper_wick >= wick_thresh:
                    entry_price = cl[i]
                    stop_price  = res + stop_atr_frac * atr20 * pip
                    if tgt_pips_fixed > 0:
                        tgt_price = entry_price - tgt_pips_fixed * pip
                    else:
                        s = support_lvl[i]
                        tgt_price = (s + sp*pip) if not np.isnan(s) else entry_price - 20*pip
                    if tgt_price < entry_price - sp*pip:
                        in_trade   = True
                        trade_dir  = -1
                        trade_entry= entry_price
                        trade_stop = stop_price
                        trade_tgt  = tgt_price
                        i += 1; continue

        i += 1

    return pnl[:nc], etype[:nc], nc


# ── Bootstrap + permutation helpers ──────────────────────────────────────────
def bootstrap_metrics(pnl, n_days, n_boot):
    if len(pnl) == 0:
        return 0.0, 0.0, 0.0
    ppd = pnl.sum() / max(n_days, 1)
    sums = np.array([np.random.choice(pnl, size=len(pnl), replace=True).sum()
                     for _ in range(n_boot)])
    p5   = float(np.percentile(sums / max(n_days, 1), 5))
    ppos = float((sums > 0).mean())
    return ppd, p5, ppos


def permutation_p(pnl, n_perm):
    """One-sided permutation test: P(shuffled >= observed)."""
    if len(pnl) == 0: return 1.0
    obs = pnl.sum()
    cnt = 0
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=len(pnl))
        if (pnl * signs).sum() >= obs: cnt += 1
    return cnt / n_perm


# ── Main sweep ────────────────────────────────────────────────────────────────
# First-pass grid: fix swing_window=5, cluster_tol=10, min_touches=2
# Sweep: wick_thresh × stop_atr_frac × range_bars × tgt_pips
SWING_W_FIXED  = 5
CLUSTER_FIXED  = 10
TOUCHES_FIXED  = 2
LOOKBACK_FIXED = 200   # bars of history to build S/R levels

rows = []
for pair, pip, med_sp in PAIRS:
    mid = pd.read_parquet(DATA_DIR_MID/f'{pair}_M5.parquet').sort_values('timestamp').reset_index(drop=True)
    ba  = pd.read_parquet(DATA_DIR_BA /f'{pair}_M5_BA.parquet').sort_values('timestamp').reset_index(drop=True)
    mid['ts_key'] = mid['timestamp'].astype(str).str[:19]
    ba['ts_key']  = ba['timestamp'].astype(str).str[:19]
    merged = mid.merge(ba[['ts_key','bid_c','ask_c']], on='ts_key', how='left')
    merged['spread'] = ((merged['ask_c'] - merged['bid_c']) / pip).clip(0, 50)
    merged['spread'] = merged['spread'].fillna(merged['spread'].median())

    op = merged['open'].values.astype(np.float64)
    hi = merged['high'].values.astype(np.float64)
    lo = merged['low'].values.astype(np.float64)
    cl = merged['close'].values.astype(np.float64)
    sp = merged['spread'].values.astype(np.float64)
    n_total = len(cl)
    n_oos   = int(n_total * OOS_FRAC)
    n_is    = n_total - n_oos
    oos_days= n_oos * 5 / (60 * 24)

    # Precompute swings (entire dataset, causal — swing at i only visible at i+window)
    s_hi, s_lo = find_swings(hi, lo, SWING_W_FIXED)

    # Build S/R history once — doesn't depend on wick/stop/tgt parameters
    print(f"  Computing S/R history...", flush=True)
    supp_lvl, supp_tc, res_lvl, res_tc = build_sr_history(
        hi, lo, s_hi, s_lo, pip, CLUSTER_FIXED, TOUCHES_FIXED, LOOKBACK_FIXED)

    # Precompute range regime masks for each distinct range_bars value
    in_range_cache = {}
    for rb in RANGE_BARS:
        in_range_cache[rb] = range_regime_mask(s_hi, s_lo, hi, lo, rb)
    print(f"  S/R + regime masks done.", flush=True)

    print(f"\n{'='*72}")
    print(f"PAIR: {pair}  swing_w={SWING_W_FIXED} cluster={CLUSTER_FIXED}p "
          f"touches={TOUCHES_FIXED}  OOS={oos_days:.0f}d")
    print(f"{'wick':>6} {'stop_f':>6} {'range':>6} {'tgt':>5} | "
          f"{'p/d':>8} {'P5':>8} {'nc':>6} {'hit%':>6} {'stop%':>6} "
          f"{'IS':>4} {'OOS':>4} {'perm_p':>7} | status")
    print('-'*72)

    for wick_thresh, stop_atr_frac, range_bars, tgt_pips in itertools.product(
            WICK_THRESHS, STOP_ATR_FRACS, RANGE_BARS, TGT_PIPS):

        in_range = in_range_cache[range_bars]

        # IS walk-forward
        is_wf = 0
        is_chunk = n_is // IS_CHUNKS
        for ch in range(IS_CHUNKS):
            s = ch * is_chunk
            e = s + is_chunk if ch < IS_CHUNKS - 1 else n_is
            p, et, nc_ch = sim_range_swing(
                op[s:e], hi[s:e], lo[s:e], cl[s:e], sp[s:e], pip,
                supp_lvl[s:e], supp_tc[s:e], res_lvl[s:e], res_tc[s:e],
                in_range[s:e], wick_thresh, stop_atr_frac, float(tgt_pips))
            if nc_ch > 0 and p.sum() > 0: is_wf += 1

        # OOS walk-forward
        oos_wf = 0
        oos_chunk = n_oos // OOS_CHUNKS
        for ch in range(OOS_CHUNKS):
            s = n_is + ch * oos_chunk
            e = n_is + (s - n_is + oos_chunk) if ch < OOS_CHUNKS - 1 else n_total
            p, et, nc_ch = sim_range_swing(
                op[s:e], hi[s:e], lo[s:e], cl[s:e], sp[s:e], pip,
                supp_lvl[s:e], supp_tc[s:e], res_lvl[s:e], res_tc[s:e],
                in_range[s:e], wick_thresh, stop_atr_frac, float(tgt_pips))
            if nc_ch > 0 and p.sum() > 0: oos_wf += 1

        # Full OOS
        pnl_full, et_full, nc_full = sim_range_swing(
            op[n_is:], hi[n_is:], lo[n_is:], cl[n_is:], sp[n_is:], pip,
            supp_lvl[n_is:], supp_tc[n_is:], res_lvl[n_is:], res_tc[n_is:],
            in_range[n_is:], wick_thresh, stop_atr_frac, float(tgt_pips))

        if nc_full == 0:
            print(f"{wick_thresh:>6.2f} {stop_atr_frac:>6.1f} {range_bars:>6d} {tgt_pips:>5d} | "
                  f"  NO TRADES"); continue

        ppd, p5, p_pos = bootstrap_metrics(pnl_full, oos_days, N_BOOT)
        perm_p = permutation_p(pnl_full, N_PERM)
        hit_pct  = 100.0 * (et_full == 1).sum() / nc_full
        stop_pct = 100.0 * (et_full == 2).sum() / nc_full

        gate_pass = (is_wf == IS_CHUNKS and oos_wf == OOS_CHUNKS
                     and p5 > 0 and p_pos > 0.95 and perm_p < 0.05)
        status = "🟢 PASS" if gate_pass else f"{is_wf}/{oos_wf} p={perm_p:.2f}"

        print(f"{wick_thresh:>6.2f} {stop_atr_frac:>6.1f} {range_bars:>6d} {tgt_pips:>5d} | "
              f"{ppd:>8.1f} {p5:>8.1f} {nc_full:>6d} {hit_pct:>5.1f}% {stop_pct:>5.1f}% "
              f"{is_wf:>4}/{IS_CHUNKS} {oos_wf:>4}/{OOS_CHUNKS} {perm_p:>7.3f} | {status}")
        sys.stdout.flush()

        rows.append(dict(
            pair=pair, pip=pip,
            swing_w=SWING_W_FIXED, cluster_tol=CLUSTER_FIXED, min_touches=TOUCHES_FIXED,
            wick_thresh=wick_thresh, stop_atr_frac=stop_atr_frac,
            range_bars=range_bars, tgt_pips=tgt_pips,
            ppd=round(ppd, 1), p5=round(p5, 1), p_pos=round(p_pos, 3),
            nc=nc_full, hit_pct=round(hit_pct, 1), stop_pct=round(stop_pct, 1),
            perm_p=round(perm_p, 3), is_wf=is_wf, oos_wf=oos_wf, gate_pass=gate_pass,
        ))

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
print(f"\n\nResults saved → {OUT_PATH}")

print("\n=== PASSING CONFIGS (all gates) ===")
passes = df[df['gate_pass']].sort_values('ppd', ascending=False)
if passes.empty:
    print("None passed — check if range filter too strict or wick too high.")
    print("\nBest non-passing (IS=3/3 only):")
    print(df[df['is_wf']==3].sort_values('ppd', ascending=False).head(10)[
        ['pair','wick_thresh','stop_atr_frac','range_bars','tgt_pips',
         'ppd','p5','nc','hit_pct','stop_pct','is_wf','oos_wf']].to_string(index=False))
else:
    print(passes[['pair','wick_thresh','stop_atr_frac','range_bars','tgt_pips',
                  'ppd','p5','nc','hit_pct','stop_pct','perm_p']].to_string(index=False))

print("\nNote: if this strategy shows positive edge, also run backtest_range_swing_v2.py")
print("  v2 adds: ZR fallback when stop hit (hybrid directional → ZR on sweep)")
