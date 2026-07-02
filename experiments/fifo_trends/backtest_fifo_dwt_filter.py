"""
FIFO-Trends + DWT D3 Shock Filter — Phase 10.

At each P&F E2 entry signal, check if a D3 shock occurred in the same
direction within the past DWT_FWD_WIN M5 bars (~20 min). If yes → enter.
If no → skip.

D3 at M5 resolution: db4 level=6, D3 = 40-min scale.
MAD z-score threshold 2.5.

Test pairs: GBP_JPY, USD_JPY, EUR_JPY (top FIFO performers, OOS refs from v2).
Baseline configs (b5 r1, E2, 1-box trail) from backtest_fifo_pnf_v2.py.
Compare: baseline OOS p/d vs DWT-filtered OOS p/d, WR, trade count.
"""
import warnings; warnings.filterwarnings("ignore")
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pywt

BASE   = Path(__file__).resolve().parents[3]
BA_DIR = BASE / "data" / "m5_ba"
OUT    = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

IS_FRAC  = 0.70
M5_PER_TRADING_DAY = 288.0
WAVELET  = "db4"
DWT_LEVEL_M5 = 6   # D3 = 40-min scale at M5 resolution
MAD_WIN_M5   = 288  # 1 trading day
D3_SHOCK_Z   = 2.5
DWT_FWD_WIN  = 4   # look back 4 M5 bars (20 min) for D3 shock

PIP_MAP = {"GBP_JPY": 0.01, "USD_JPY": 0.01, "EUR_JPY": 0.01}
SP_GATE = {"GBP_JPY": 4.00, "USD_JPY": 2.10, "EUR_JPY": 2.50}

# Best configs from CLAUDE.md / backtest_fifo_pnf_v2.py
# (pair, b_pips, rev, n_min, entry_t, trail_k)
CONFIGS = [
    ("GBP_JPY", 5, 1, 4, 2, 1),
    ("USD_JPY", 5, 1, 3, 2, 1),
    ("EUR_JPY", 5, 1, 3, 2, 1),
]


def load_m5ba(pair: str) -> pd.DataFrame:
    df = pd.read_parquet(BA_DIR / f"{pair}_M5_BA.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def compute_d3_shock(close: np.ndarray) -> tuple:
    """DWT D3 at M5 resolution. Returns (d3_shock_bool, d3_coef_sign) len=n."""
    n = len(close)
    coeffs = pywt.wavedec(close, WAVELET, level=DWT_LEVEL_M5, mode="periodization")
    # coeffs[0]=A6, [1]=D6, [2]=D5, [3]=D4, [4]=D3, [5]=D2, [6]=D1
    cd3 = coeffs[4]   # D3 detail band
    factor = (n + len(cd3) - 1) // len(cd3)
    d3_full = np.repeat(cd3, factor)[:n]
    log_e = np.log(d3_full ** 2 + 1e-12)
    s = pd.Series(log_e)
    rm  = s.rolling(MAD_WIN_M5, center=True, min_periods=max(10, MAD_WIN_M5//4)).median()
    ad  = (s - rm).abs()
    rm2 = ad.rolling(MAD_WIN_M5, center=True, min_periods=max(10, MAD_WIN_M5//4)).median()
    z   = ((s - rm) / (1.4826 * rm2.clip(lower=1e-12))).fillna(0).values
    return z > D3_SHOCK_Z, np.sign(d3_full)


def run_backtest(pair: str, b_pips: float, rev: int, n_min: int,
                 entry_t: int, k: int, use_dwt_filter: bool) -> dict:
    df      = load_m5ba(pair)
    pip     = PIP_MAP[pair]
    sp_gate = SP_GATE[pair]
    close   = df["close"].values.astype(np.float64)
    high    = df["high"].values.astype(np.float64)
    low     = df["low"].values.astype(np.float64)
    sp      = ((df["ask_c"] - df["bid_c"]) / pip).values.astype(np.float64)
    n       = len(close)
    is_n    = int(n * IS_FRAC)
    b       = b_pips * pip

    d3_shock, d3_sign = compute_d3_shock(close)

    # P&F state
    box_hi = close[0]; box_lo = close[0] - b
    col = 0; col_count = 0
    in_trade = False; entry_px = 0.0; direction = 0; trail = 0.0; n_col_entry = 0

    trades = []
    filtered_out = 0
    total_signals = 0

    for i in range(1, n):
        c = close[i]; h = high[i]; l = low[i]; sp_i = sp[i]
        if sp_i > sp_gate:
            continue

        # Update P&F chart
        if col == 0:
            if c >= box_hi + b: col = 1; col_count = 1
            elif c <= box_lo - b: col = -1; col_count = 1
        elif col == 1:  # X column
            while c >= box_hi + b:
                box_hi += b; box_lo += b; col_count += 1
            if c <= box_hi - rev * b:
                col = -1; col_count = 1; box_hi -= rev * b; box_lo = box_hi - b
        else:           # O column
            while c <= box_lo - b:
                box_lo -= b; box_hi -= b; col_count += 1
            if c >= box_lo + rev * b:
                col = 1; col_count = 1; box_lo += rev * b; box_hi = box_lo + b

        # Entry: E2 confirmation (col_count == n_min at new column)
        if not in_trade and col_count == n_min:
            dir_entry = col
            total_signals += 1
            if use_dwt_filter:
                look_start = max(0, i - DWT_FWD_WIN)
                shock_ok = any(
                    d3_shock[j] and (d3_sign[j] * dir_entry > 0)
                    for j in range(look_start, i + 1)
                )
                if not shock_ok:
                    filtered_out += 1
                    continue
            entry_px  = close[i]
            in_trade  = True
            direction = dir_entry
            trail     = entry_px - k * b * direction
            n_col_entry = col_count

        # Trail update + exit
        if in_trade:
            new_trail = close[i] - k * b * direction
            if direction == 1 and new_trail > trail:
                trail = new_trail
            elif direction == -1 and new_trail < trail:
                trail = new_trail
            # Trail exit (S5 monitoring fills at trail level)
            if (direction == 1 and l <= trail) or (direction == -1 and h >= trail):
                exit_px = trail
                net = (exit_px - entry_px) * direction / pip - sp_gate
                trades.append((i, net, direction))
                in_trade = False
            # X7 exit: reversal past entry column depth
            elif col != direction and col_count > n_col_entry + entry_t:
                exit_px = close[i]
                net = (exit_px - entry_px) * direction / pip - sp_gate
                trades.append((i, net, direction))
                in_trade = False

    if not trades:
        return {}
    tdf = pd.DataFrame(trades, columns=["bar", "net_pips", "dir"])
    tdf_oos = tdf[tdf["bar"] > is_n]
    n_oos_days = (n - is_n) / M5_PER_TRADING_DAY
    p_d  = tdf_oos["net_pips"].sum() / n_oos_days if n_oos_days > 0 else 0
    wr   = (tdf_oos["net_pips"] > 0).mean() if len(tdf_oos) > 0 else 0
    avg_win  = tdf_oos.loc[tdf_oos["net_pips"] > 0, "net_pips"].mean() if (tdf_oos["net_pips"] > 0).any() else 0
    avg_loss = tdf_oos.loc[tdf_oos["net_pips"] <= 0, "net_pips"].mean() if (tdf_oos["net_pips"] <= 0).any() else 0
    filter_rate = filtered_out / total_signals if total_signals > 0 else 0
    return {
        "pair": pair, "b_pips": b_pips, "rev": rev, "n_min": n_min, "k": k,
        "dwt_filter": use_dwt_filter,
        "total_signals": total_signals, "filtered_out": filtered_out,
        "filter_rate_pct": round(filter_rate * 100, 1),
        "oos_trades": len(tdf_oos), "oos_pd": round(p_d, 2),
        "wr_pct": round(wr * 100, 1),
        "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "oos_days": round(n_oos_days, 1)
    }


if __name__ == "__main__":
    results = []
    for pair, b_pips, rev, n_min, entry_t, k in CONFIGS:
        print(f"\n{'='*50}\n{pair}  b={b_pips}p r{rev} n{n_min} k{k}")
        print(f"  Computing DWT D3 shock mask...")
        for use_filter in [False, True]:
            tag = "DWT_FILTER" if use_filter else "baseline"
            r = run_backtest(pair, b_pips, rev, n_min, entry_t, k, use_filter)
            if r:
                results.append(r)
                filter_info = (f"  filter_rate={r['filter_rate_pct']}%"
                               if use_filter else "")
                print(f"  [{tag}] OOS: {r['oos_pd']} p/d  WR={r['wr_pct']}%  "
                      f"n={r['oos_trades']} trades  days={r['oos_days']}"
                      f"  avgW={r['avg_win']} avgL={r['avg_loss']}{filter_info}")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "fifo_dwt_filter_results.csv", index=False)
    print(f"\nSaved: results/fifo_dwt_filter_results.csv")

    print("\n=== COMPARISON ===")
    for pair in df["pair"].unique():
        sub = df[df["pair"] == pair][
            ["dwt_filter","oos_pd","wr_pct","oos_trades","filter_rate_pct","avg_win","avg_loss"]
        ]
        print(f"\n{pair}:")
        print(sub.to_string(index=False))
        rows = sub.to_dict("records")
        if len(rows) == 2:
            base, filt = (rows[0], rows[1]) if not rows[0]["dwt_filter"] else (rows[1], rows[0])
            delta_wr = filt["wr_pct"] - base["wr_pct"]
            delta_pd = filt["oos_pd"] - base["oos_pd"]
            print(f"  ΔWR={delta_wr:+.1f}pp  Δp/d={delta_pd:+.1f}  "
                  f"GATE(ΔWR≥+3): {'✅' if delta_wr >= 3 else '❌'}  "
                  f"GATE(p/d≥80%base): {'✅' if base['oos_pd'] > 0 and filt['oos_pd'] >= 0.8 * base['oos_pd'] else '❌'}")
