"""Cost engine + analysis harness (master doc §9, §13-15, Part VI steps 4-6).

All win rates reported relative to the 66.7% barrier baseline and to each
broker's cost-adjusted p*. IS = first 70% by time; OOS sealed until the
review gate (R8) and evaluated exactly once via analyze(pair, oos=True).
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ECN_COST = 0.7
COST_MULTS = (1.0, 1.5, 2.0)
TP_LEVELS = {"t15": 1.5, "t18": 1.8, "t32": 3.2, "t40": 4.0}
HORIZONS = {"h2": 2, "h4": 4}
IS_FRAC = 0.70


def p_star(tp, sl, cost):
    return (sl + cost) / ((tp - cost) + (sl + cost))


def assign_arms(df, seed):
    df = df[df["drift"] != 0].copy()
    df["dir_with"] = np.sign(df["drift"]).astype(np.int64)
    df["dir_against"] = -df["dir_with"]
    rng = np.random.default_rng(seed)
    df["dir_coin"] = rng.choice([-1, 1], size=len(df))
    return df


def fifo_realize(df, held_col):
    """One position at a time. held is in S5 bars (5 s each)."""
    open_until = None
    out = np.zeros(len(df), dtype=bool)
    ts = df["ts"].values
    held = df[held_col].values
    for i in range(len(df)):
        if open_until is None or ts[i] >= open_until:
            out[i] = True
            open_until = ts[i] + np.timedelta64(int(held[i]) * 5, "s")
    return pd.Series(out, index=df.index)


def day_block_bootstrap(df, stat_fn, n=2000, seed=0):
    days = df["ts"].dt.floor("D")
    groups = {d: g for d, g in df.groupby(days)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for b in range(n):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        sample = pd.concat([groups[keys[k]] for k in pick])
        stats[b] = stat_fn(sample)
    return float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))


def er_bucket_edges(er_is):
    return (float(er_is.quantile(1/3)), float(er_is.quantile(2/3)))


def bucket_table(df, arm, tp_key, h_key, er_edges):
    """Per ER bucket stats for one arm x geometry. df must carry dir_<arm>."""
    tp = TP_LEVELS[tp_key]; sl = 2 * tp
    d = df.copy()
    dirs = d[f"dir_{arm}"].values
    for col in ("label", "exit_pips", "bars_held"):
        long_c = d[f"{tp_key}_{h_key}_long_{col}"].values
        short_c = d[f"{tp_key}_{h_key}_short_{col}"].values
        d[col] = np.where(dirs == 1, long_c, short_c)
    d = d[d["label"] != -9]
    d = d[np.isfinite(d["er"].values)]
    d["bucket"] = np.digitize(d["er"].values, er_edges)   # 0 lo, 1 mid, 2 hi
    rows = []
    for b, g in d.groupby("bucket"):
        dec = g[g["label"] != 0]
        n_dec = len(dec)
        wr = float((dec["label"] == 1).mean()) if n_dec else np.nan
        rec = {"bucket": ["loER", "midER", "hiER"][b], "n": len(g),
               "n_decided": n_dec, "wr": wr, "wr_minus_667": wr - 2/3,
               "timeout_share": float((g["label"] == 0).mean()),
               "timeout_pnl_gross": float(g.loc[g["label"] == 0, "exit_pips"].mean()),
               "gross_mean": float(np.where(g["label"] == 1, tp,
                              np.where(g["label"] == -1, -sl, g["exit_pips"])).mean())}
        for name, cost_series in (("oanda", g["spread_pips"]),
                                  ("ecn", pd.Series(ECN_COST, index=g.index))):
            for m in COST_MULTS:
                c = cost_series * m
                net = np.where(g["label"] == 1, tp - c,
                      np.where(g["label"] == -1, -(sl + c), g["exit_pips"] - c))
                rec[f"net_{name}_{m:g}x"] = float(net.mean())
            ps = p_star(tp, sl, float(cost_series.mean()))
            rec[f"pstar_{name}"] = ps
            rec[f"wr_minus_pstar_{name}"] = wr - ps
            if n_dec >= 30:
                # ps can exceed 1.0 when mean cost > tp (e.g. OANDA spread ~1.6p
                # vs t15 TP=1.5p) -- a valid signal that clearing cost is
                # impossible at this TP, but not a valid binomtest probability.
                # Clip for the test only; the uncapped ps above still reports
                # the infeasibility magnitude.
                ps_clipped = min(ps, 1.0)
                rec[f"binom_p_{name}"] = binomtest(int((dec["label"] == 1).sum()),
                                                   n_dec, ps_clipped, "greater").pvalue
        rows.append(rec)
    return pd.DataFrame(rows)


def analyze(pair, oos=False, seed=20260706, out_dir="signals"):
    df = pd.read_parquet(f"{out_dir}/{pair}_signals.parquet")
    df = df.sort_values("ts").reset_index(drop=True)
    cut = int(len(df) * IS_FRAC)
    er_edges = er_bucket_edges(df["er"].iloc[:cut].dropna())   # IS-only always
    part = df.iloc[cut:] if oos else df.iloc[:cut]
    part = assign_arms(part, seed)
    results = {"pair": pair, "oos": oos, "er_edges": er_edges, "n": len(part)}
    for arm in ("against", "with", "coin"):
        for tp_key in TP_LEVELS:
            for h_key in HORIZONS:
                results[(arm, tp_key, h_key)] = bucket_table(
                    part, arm, tp_key, h_key, er_edges)

    # FIFO + session + all-signals-vs-realized comparison (confirmatory geometry)
    conf = part.copy()
    dirs = conf["dir_against"].values
    conf["label"] = np.where(dirs == 1, conf["t32_h2_long_label"], conf["t32_h2_short_label"])
    conf["bars_held"] = np.where(dirs == 1, conf["t32_h2_long_bars_held"], conf["t32_h2_short_bars_held"])
    conf = conf[conf["label"] != -9]
    mask = fifo_realize(conf, held_col="bars_held")
    results["fifo_realized_frac"] = float(mask.mean())
    dec_all = conf[conf["label"] != 0]
    dec_fifo = conf[mask & (conf["label"] != 0)]
    results["wr_all_signals"] = float((dec_all["label"] == 1).mean())
    results["wr_fifo"] = float((dec_fifo["label"] == 1).mean())
    results["per_session"] = {
        s: float((g.loc[g["label"] != 0, "label"] == 1).mean())
        for s, g in conf.groupby("session")}

    return results


def render_report(results, path):
    lines = [f"# Regime-entry {'OOS (SEALED-RUN)' if results['oos'] else 'IS'} "
             f"report — {results['pair']}",
             f"n={results['n']}  ER tercile edges (IS): {results['er_edges']}",
             "", "Baseline 66.7%. Confirmatory cell: against/t32/h2/hiER.", ""]
    for key, tbl in results.items():
        if isinstance(key, tuple):
            lines.append(f"## arm={key[0]} TP={key[1]} horizon={key[2]}")
            lines.append(tbl.round(4).to_markdown(index=False))
            lines.append("")
    lines.append("## FIFO-selection-bias control + session breakdown "
                 "(confirmatory geometry: against/t32/h2)")
    lines.append(f"fifo_realized_frac={results['fifo_realized_frac']:.4f}  "
                 f"wr_all_signals={results['wr_all_signals']:.4f}  "
                 f"wr_fifo={results['wr_fifo']:.4f}")
    lines.append("")
    lines.append("per_session decided win rate:")
    for s, wr in results["per_session"].items():
        lines.append(f"- {s}: {wr:.4f}")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    import sys
    pair = sys.argv[1] if len(sys.argv) > 1 else "EUR_USD"
    oos = "--oos" in sys.argv
    if oos:
        confirm = input("OOS is evaluated ONCE (R8). Type 'UNSEAL' to proceed: ")
        if confirm != "UNSEAL":
            sys.exit("aborted")
    res = analyze(pair, oos=oos)
    render_report(res, f"report_{'oos' if oos else 'is'}_{pair}.md")
