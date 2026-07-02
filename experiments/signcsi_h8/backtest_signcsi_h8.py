#!/usr/bin/env python3
"""
Experiment A: SignedCSI H8 Contrarian Strategy
===============================================
Book Review V2 Campaign — 2026-04-24

Reproduces and validates the book's §20.19.1 finding:
    SignedCSI H8, H=1 bar hold, N=4 pairs/side
    Reported: Sharpe 0.752, worst WF fold 0.192, p=0.0000

Strategy:
    - Rank 24 G10 FX pairs by SignedCSI at each H8 bar close
    - CONTRARIAN signal: SHORT top-N (highest SignedCSI), LONG bottom-N (lowest)
    - Hold for H bars, then rebalance
    - Spread deducted at entry only

Causality guarantee:
    - SignedCSI at bar t uses only OHLC data from bars ≤ t (rolling windows)
    - Forward return at bar t = close[t+H]/close[t] - 1  (future target, not feature)
    - No cross-TF merge_asof (native H8 cadence throughout)

Data: ~/projects/csi_factor_study/data/{PAIR}_H8.parquet (OHLC only)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
CSI_STUDY_DIR = Path.home() / "projects" / "csi_factor_study"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CSI_STUDY_DIR))

from src.factors.csi import compute_adx_adxr
from src.factors.signed_csi import compute_regression_slope

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Pairs & pip sizes ─────────────────────────────────────────────────
ALL_PAIRS = [
    "AUD_CHF", "AUD_JPY", "AUD_NZD", "AUD_USD",
    "CAD_JPY", "CHF_JPY", "EUR_AUD", "EUR_CAD",
    "EUR_CHF", "EUR_GBP", "EUR_JPY", "EUR_NZD",
    "EUR_USD", "GBP_AUD", "GBP_CAD", "GBP_CHF",
    "GBP_JPY", "GBP_NZD", "GBP_USD", "NZD_JPY",
    "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY",
]

PIP_SIZE = {p: (0.01 if "JPY" in p else 0.0001) for p in ALL_PAIRS}
TYPICAL_SPREAD = {
    "EUR_USD": 1.6, "GBP_USD": 1.9, "USD_JPY": 1.7, "AUD_USD": 1.3,
    "EUR_JPY": 2.3, "GBP_JPY": 3.3, "AUD_JPY": 2.1, "CAD_JPY": 2.3,
    "CHF_JPY": 3.5, "NZD_JPY": 2.7, "NZD_USD": 1.5, "EUR_GBP": 1.4,
    "USD_CAD": 2.0, "USD_CHF": 2.2, "EUR_AUD": 2.5, "EUR_CAD": 2.5,
    "EUR_CHF": 2.2, "EUR_NZD": 3.0, "GBP_AUD": 3.2, "GBP_CAD": 3.2,
    "GBP_CHF": 3.0, "GBP_NZD": 3.5, "AUD_CHF": 2.5, "AUD_NZD": 2.5,
}

H8_BARS_PER_YEAR = 365 * 3  # 3 H8 bars per day × 365 days


# ── SignedCSI computation (causal) ────────────────────────────────────

def compute_signed_csi_causal(df: pd.DataFrame, pair: str,
                               atr_period: int = 14, reg_lookback: int = 20) -> pd.Series:
    """
    Compute causal SignedCSI series from OHLC DataFrame.
    Uses same formula as CSI study: CSI * sign(regression slope).
    All rolling windows operate on past data only — causal by construction.
    """
    indicators = compute_adx_adxr(df, atr_period=atr_period)
    csi_raw = indicators["ADXR"] * indicators["ATR"] / PIP_SIZE[pair]

    slope = compute_regression_slope(df["close"], lookback=reg_lookback)
    sign = np.sign(slope)

    signed_csi = csi_raw * sign
    signed_csi.name = f"SignedCSI_{pair}"
    return signed_csi


def compute_strength_spread_causal(df: pd.DataFrame, pair: str,
                                    lookback: int = 1) -> pd.Series:
    """
    Per-pair strength spread (base strength - quote strength proxy):
    Use pct return over lookback bars as a simple momentum proxy.
    """
    ret = df["close"].pct_change(lookback)
    ret.name = f"StrengthSpread_{pair}"
    return ret


# ── Data loading ──────────────────────────────────────────────────────

def load_h8_data() -> dict:
    """Load H8 OHLC for all available pairs from CSI study data dir."""
    data_dir = CSI_STUDY_DIR / "data"
    loaded = {}
    for pair in ALL_PAIRS:
        fpath = data_dir / f"{pair}_H8.parquet"
        if fpath.exists():
            df = pd.read_parquet(fpath)
            df.index = pd.to_datetime(df.index, utc=True)
            df.sort_index(inplace=True)
            loaded[pair] = df
    print(f"Loaded {len(loaded)} pairs for H8 analysis: {list(loaded.keys())}")
    return loaded


# ── Factor panel construction ─────────────────────────────────────────

def build_factor_panel(data: dict, factor: str = "SignedCSI",
                        reg_lookback: int = 20, atr_period: int = 14) -> pd.DataFrame:
    """
    Build aligned factor panel: DatetimeIndex × pairs.
    Only includes bars where all pairs have data (inner join after alignment).
    """
    series = {}
    for pair, df in data.items():
        if factor == "SignedCSI":
            s = compute_signed_csi_causal(df, pair, atr_period, reg_lookback)
        elif factor == "StrengthSpread":
            s = compute_strength_spread_causal(df, pair)
        else:
            raise ValueError(f"Unknown factor: {factor}")
        series[pair] = s

    panel = pd.DataFrame(series)
    # Drop early warmup rows where any pair has NaN
    panel = panel.dropna(how="any")
    return panel


def build_fwd_return_panel(data: dict, H: int = 1) -> pd.DataFrame:
    """
    Forward return panel: return over the next H bars at each bar close.
    fwd_ret[t] = close[t+H] / close[t] - 1  (future observation, correct as target)
    """
    series = {}
    for pair, df in data.items():
        fwd = df["close"].pct_change(H).shift(-H)  # shift(-H) moves future return to current index
        series[pair] = fwd
    return pd.DataFrame(series)


# ── Backtest engine ───────────────────────────────────────────────────

def run_backtest(factor_panel: pd.DataFrame, fwd_panel: pd.DataFrame,
                 H: int = 1, N: int = 4, slippage_mult: float = 1.5,
                 direction: str = "contrarian") -> pd.Series:
    """
    Non-overlapping rebalancing backtest.

    direction='contrarian': long BOTTOM-N, short TOP-N (mean reversion)
    direction='momentum':   long TOP-N, short BOTTOM-N (trend following)

    Hold for H bars → harvest forward return.
    Returns gross pip returns per rebalance, net of transaction costs.
    """
    # Align factor and forward panels
    common_idx = factor_panel.index.intersection(fwd_panel.dropna(how="all").index)
    factor_aligned = factor_panel.loc[common_idx]
    fwd_aligned = fwd_panel.loc[common_idx]

    # Drop last H bars (no forward return available)
    valid_idx = factor_aligned.index[:-H] if H > 0 else factor_aligned.index
    rebal_times = valid_idx[::H]  # non-overlapping every H bars

    returns_list = []
    prev_longs = set()
    prev_shorts = set()

    for t in rebal_times:
        row = factor_aligned.loc[t].dropna()
        if len(row) < 2 * N:
            continue

        bottom = set(row.nsmallest(N).index.tolist())
        top = set(row.nlargest(N).index.tolist())

        # Direction determines which is long and which is short
        if direction == "contrarian":
            longs, shorts = bottom, top   # long weak, short strong
        else:
            longs, shorts = top, bottom   # long strong, short weak

        fwd_row = fwd_aligned.loc[t]

        # Gross return: mean(long returns) - mean(short returns)
        long_ret = fwd_row[list(longs)].mean()
        short_ret = fwd_row[list(shorts)].mean()
        gross_ret = long_ret - short_ret

        # Transaction cost: only changed positions pay spread
        changed_longs = len(longs - prev_longs) + len(prev_longs - longs)
        changed_shorts = len(shorts - prev_shorts) + len(prev_shorts - shorts)
        n_changed = changed_longs + changed_shorts

        # Average spread cost for changed pairs (in return units)
        changed_pairs = (longs - prev_longs) | (shorts - prev_shorts)
        if len(changed_pairs) > 0:
            avg_spread_pct = np.mean([
                TYPICAL_SPREAD.get(p, 2.0) * PIP_SIZE.get(p, 0.0001)
                for p in changed_pairs
            ]) * slippage_mult
        else:
            avg_spread_pct = 0.0

        # Cost applies to fraction of changed positions per side
        cost = avg_spread_pct * (n_changed / (2 * 2 * N))  # n_changed/total positions
        net_ret = gross_ret - cost

        returns_list.append((t, gross_ret, net_ret, long_ret, short_ret))
        prev_longs = longs
        prev_shorts = shorts

    if not returns_list:
        return pd.Series(dtype=float)

    idx, gross, net, lr, sr = zip(*returns_list)
    result = pd.DataFrame({
        "gross_pct": gross,
        "net_pct": net,
        "long_ret": lr,
        "short_ret": sr,
    }, index=idx)
    return result


# ── Walk-forward splitter ─────────────────────────────────────────────

def walk_forward_splits(index: pd.DatetimeIndex, n_splits: int = 5):
    """
    Produce (is_idx, oos_idx) tuples for walk-forward validation.
    Each split adds one chunk of data to IS, tests on the next chunk.
    """
    n = len(index)
    chunk = n // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        is_end = i * chunk
        oos_end = min((i + 1) * chunk, n)
        splits.append((index[:is_end], index[is_end:oos_end]))
    return splits


# ── Sharpe helper ─────────────────────────────────────────────────────

def annualised_sharpe(returns: pd.Series, bars_per_year: float = H8_BARS_PER_YEAR) -> float:
    if len(returns) < 4 or returns.std() < 1e-12:
        return float("nan")
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(bars_per_year))


# ── Main sweep ────────────────────────────────────────────────────────

def run_sweep(factor: str = "SignedCSI",
              H_values: list = None, N_values: list = None,
              oos_fraction: float = 0.4,
              n_wf_splits: int = 5,
              n_perms: int = 500,
              directions: list = None,
              verbose: bool = True) -> list:
    """Full parameter sweep with 5-gate validation on top results."""
    if H_values is None:
        H_values = [8, 16, 32]  # 64h, 128h, 256h at H8 cadence
    if N_values is None:
        N_values = [3, 4, 5]
    if directions is None:
        directions = ["contrarian", "momentum"]

    print(f"\n{'='*60}")
    print(f"Experiment A: SignedCSI H8 Sweep (Contrarian + Momentum)")
    print(f"Factor: {factor}, H in {H_values}, N in {N_values}, dir: {directions}")
    print(f"OOS fraction: {oos_fraction}, WF splits: {n_wf_splits}")
    print(f"{'='*60}\n")

    data = load_h8_data()
    factor_panel = build_factor_panel(data, factor=factor)
    print(f"Factor panel: {factor_panel.shape} ({factor_panel.index[0].date()} → {factor_panel.index[-1].date()})")

    # IS/OOS split
    n_total = len(factor_panel)
    n_is = int(n_total * (1 - oos_fraction))
    is_end = factor_panel.index[n_is]
    print(f"IS: {factor_panel.index[0].date()} → {is_end.date()} ({n_is} bars)")
    print(f"OOS: {is_end.date()} → {factor_panel.index[-1].date()} ({n_total - n_is} bars)")

    results = []
    for H in H_values:
        fwd_panel = build_fwd_return_panel(data, H=H)
        # Align fwd with factor
        fwd_panel = fwd_panel.reindex(factor_panel.index)

        for N in N_values:
          for direction in directions:
            print(f"\n--- H={H}, N={N}, dir={direction} ---")
            bt = run_backtest(factor_panel, fwd_panel, H=H, N=N, direction=direction)
            if len(bt) == 0:
                print("  No trades generated")
                continue

            oos_bt = bt[bt.index >= is_end]
            is_bt = bt[bt.index < is_end]

            rebal_periods_per_year = H8_BARS_PER_YEAR / H
            oos_sharpe = annualised_sharpe(oos_bt["net_pct"], bars_per_year=rebal_periods_per_year)
            is_sharpe = annualised_sharpe(is_bt["net_pct"], bars_per_year=rebal_periods_per_year)

            print(f"  IS Sharpe: {is_sharpe:.3f}  |  OOS Sharpe: {oos_sharpe:.3f}")
            print(f"  OOS trades: {len(oos_bt)}  |  OOS mean net: {oos_bt['net_pct'].mean()*100:.4f}%")

            # Walk-forward folds (5-fold on OOS period)
            oos_splits = walk_forward_splits(factor_panel.loc[factor_panel.index >= is_end].index, n_splits=5)
            fold_sharpes = []
            for is_fold_idx, oos_fold_idx in oos_splits:
                fold_bt = bt[bt.index.isin(oos_fold_idx)]
                fs = annualised_sharpe(fold_bt["net_pct"], bars_per_year=rebal_periods_per_year)
                fold_sharpes.append(float(fs) if not np.isnan(fs) else 0.0)

            n_pos_folds = sum(1 for s in fold_sharpes if s > 0)
            print(f"  WF folds: {[f'{s:.3f}' for s in fold_sharpes]} → {n_pos_folds}/5 positive")

            # Quick permutation test (shuffle factor rankings at each bar)
            perm_sharpes = []
            rng = np.random.default_rng(42)
            factor_oos = factor_panel[factor_panel.index >= is_end]
            fwd_oos = fwd_panel[fwd_panel.index >= is_end]

            for _ in range(n_perms):
                # Shuffle pair rankings at each time step (cross-sectional null)
                shuffled_vals = factor_oos.values.copy()
                for row_i in range(shuffled_vals.shape[0]):
                    rng.shuffle(shuffled_vals[row_i])
                shuffled_panel = pd.DataFrame(shuffled_vals,
                                              index=factor_oos.index,
                                              columns=factor_oos.columns)
                pbt = run_backtest(shuffled_panel, fwd_oos, H=H, N=N, direction=direction)
                s = annualised_sharpe(pbt["net_pct"], bars_per_year=rebal_periods_per_year)
                perm_sharpes.append(float(s) if not np.isnan(s) else 0.0)

            p_value = float(np.mean(np.array(perm_sharpes) >= oos_sharpe))
            perm_95th = float(np.percentile(perm_sharpes, 95))
            print(f"  Permutation p={p_value:.4f}  (real={oos_sharpe:.3f} vs 95th={perm_95th:.3f})")

            # Bootstrap CI
            oos_net = oos_bt["net_pct"].values
            boot_sharpes = []
            for _ in range(1000):
                idx = rng.integers(0, len(oos_net), size=len(oos_net))
                boot_sharpes.append(annualised_sharpe(pd.Series(oos_net[idx]), bars_per_year=rebal_periods_per_year))
            ci_lo = float(np.nanpercentile(boot_sharpes, 2.5))
            ci_hi = float(np.nanpercentile(boot_sharpes, 97.5))
            print(f"  Bootstrap CI 95%: [{ci_lo:.3f}, {ci_hi:.3f}]")

            # Drop-one-year test
            oos_bt_copy = oos_bt.copy()
            oos_bt_copy["year"] = oos_bt_copy.index.year
            years = sorted(oos_bt_copy["year"].unique())
            drop_one_sharpes = {}
            all_drop_one_positive = True
            for yr in years:
                kept = oos_bt_copy[oos_bt_copy["year"] != yr]["net_pct"]
                s = annualised_sharpe(kept, bars_per_year=rebal_periods_per_year)
                drop_one_sharpes[yr] = round(float(s), 4)
                if s <= 0:
                    all_drop_one_positive = False
            print(f"  Drop-one: {drop_one_sharpes}")

            # Gate results
            g1 = oos_sharpe > 0.5
            g2 = p_value < 0.05
            g3 = ci_lo > 0.0
            g4 = n_pos_folds == 5
            g5 = all_drop_one_positive
            n_gates = sum([g1, g2, g3, g4, g5])

            gate_str = "".join(["✅" if g else "❌" for g in [g1, g2, g3, g4, g5]])
            print(f"  Gates: {gate_str}  ({n_gates}/5)")

            row = {
                "factor": factor,
                "direction": direction,
                "H": H,
                "N": N,
                "oos_sharpe": round(oos_sharpe, 4),
                "is_sharpe": round(is_sharpe, 4),
                "n_oos_trades": len(oos_bt),
                "perm_p": round(p_value, 4),
                "perm_95th": round(perm_95th, 4),
                "boot_ci_lo": round(ci_lo, 4),
                "boot_ci_hi": round(ci_hi, 4),
                "wf_fold_sharpes": fold_sharpes,
                "n_wf_positive": n_pos_folds,
                "drop_one_sharpes": drop_one_sharpes,
                "all_drop_one_positive": all_drop_one_positive,
                "gates_passed": n_gates,
                "gate_detail": [g1, g2, g3, g4, g5],
                "gate_str": gate_str,
            }
            results.append(row)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default="SignedCSI",
                        choices=["SignedCSI", "StrengthSpread"])
    parser.add_argument("--H", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--N", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--direction", nargs="+", default=["contrarian", "momentum"])
    parser.add_argument("--oos-frac", type=float, default=0.4)
    parser.add_argument("--perms", type=int, default=500)
    args = parser.parse_args()

    results = run_sweep(
        factor=args.factor,
        H_values=args.H,
        N_values=args.N,
        oos_fraction=args.oos_frac,
        n_perms=args.perms,
        directions=args.direction,
    )

    # Save results
    out_path = RESULTS_DIR / f"signcsi_h8_{args.factor}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to {out_path}")

    # Summary table
    print("\n\n" + "="*70)
    print("SUMMARY — configurations by gates passed:")
    print("="*70)
    sorted_results = sorted(results, key=lambda r: (-r["gates_passed"], -r["oos_sharpe"]))
    for r in sorted_results:
        print(f"  {r.get('direction','?'):11s}  H={r['H']:2d}  N={r['N']}  "
              f"Sharpe={r['oos_sharpe']:6.3f}  p={r['perm_p']:.4f}  "
              f"{r['gate_str']}  ({r['gates_passed']}/5 gates)")

    # 5-gate survivors
    survivors = [r for r in results if r["gates_passed"] == 5]
    print(f"\n  Deployable (5/5 gates): {len(survivors)}")
    for r in survivors:
        print(f"    → H={r['H']}, N={r['N']}, Sharpe={r['oos_sharpe']:.3f}, p={r['perm_p']:.4f}")

    return results


if __name__ == "__main__":
    main()
