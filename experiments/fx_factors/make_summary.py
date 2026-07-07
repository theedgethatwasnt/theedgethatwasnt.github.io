#!/usr/bin/env python3
"""
make_summary.py — fx_factors: results/is_summary.md (gate table, per-factor table, verdict).
Reads results/{is_monthly.csv, null_r10.csv, gate_table.csv, gate_extra.json}.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def per_factor_table(monthly_df, null_df):
    rows = []
    null_mean = float(null_df["mean_net_pips"].mean())
    null_p95 = float(np.percentile(null_df["mean_net_pips"], 95))
    for variant, g in monthly_df.groupby("variant"):
        g = g.sort_values("signal_date")
        net = g["net_pips"]
        cum = net.cumsum()
        dd = cum - cum.cummax()
        max_dd = float(dd.min()) if len(dd) else float("nan")
        span = null_p95 - null_mean
        pctile_proxy = float((net.mean() - null_mean) / span * 100) if span != 0 else float("nan")
        rows.append({
            "variant": variant,
            "n_rebalances": len(g),
            "mean_net_pips": float(net.mean()) if len(g) else float("nan"),
            "cum_net_pips": float(net.sum()) if len(g) else float("nan"),
            "max_dd_pips": max_dd,
            "vs_null_mean_pips": float(net.mean() - null_mean) if len(g) else float("nan"),
            "vs_null_p95_pctile_proxy": pctile_proxy,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = args.out_dir

    monthly_df = pd.read_csv(os.path.join(out, "is_monthly.csv"))
    null_df = pd.read_csv(os.path.join(out, "null_r10.csv"))
    gate_table = pd.read_csv(os.path.join(out, "gate_table.csv"))
    with open(os.path.join(out, "gate_extra.json")) as f:
        extra = json.load(f)

    ft = per_factor_table(monthly_df, null_df)
    ft.to_csv(os.path.join(out, "is_per_factor_summary.csv"), index=False)

    def pass_str(v):
        if v is True:
            return "PASS"
        if v is False:
            return "FAIL"
        return "SEE PYTEST"

    lines = []
    lines.append("# FX Factor Suite — IS Summary\n")
    lines.append(
        "Governed by `PREREGISTRATION.md` (LOCKED 2026-07-07). IS window only: "
        "2020-11-11 -> 2024-08-30 (OOS sealed, never read — `is_data.load_pair_is_d1()` "
        "hard-filters every load, pushed down to pyarrow at read time + a second independent "
        "post-read assertion).\n"
    )

    lines.append("## Gate table (gates 1-4, PREREGISTRATION.md \"Gates before OOS\")\n")
    lines.append("| Gate | Name | Result | Detail |")
    lines.append("|---|---|---|---|")
    for _, r in gate_table.iterrows():
        lines.append(f"| {int(r['gate'])} | {r['name']} | **{pass_str(r['pass'])}** | {r['detail']} |")
    lines.append("")

    checked = gate_table[gate_table["pass"].notna()]
    n_pass = int((checked["pass"] == True).sum())  # noqa: E712
    lines.append(
        f"**{n_pass}/{len(checked)} numerically-checked gates pass** "
        "(gate 2 is proven separately by test_rebalance_engine.py::test_carry_accrual_parity).\n"
    )

    lines.append("## Per-factor table (IS, pooled across all monthly rebalances)\n")
    lines.append("| Variant | n | mean net p/rebal | cum net (p) | max DD (p) | vs null mean | vs null p95 (pctile proxy) |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in ft.iterrows():
        lines.append(
            f"| {r['variant']} | {int(r['n_rebalances'])} | {r['mean_net_pips']:+.3f} | "
            f"{r['cum_net_pips']:+.2f} | {r['max_dd_pips']:+.2f} | {r['vs_null_mean_pips']:+.3f} | "
            f"{r['vs_null_p95_pctile_proxy']:.1f}% |"
        )
    lines.append("")

    lines.append(
        f"R10 null: n_seeds={len(null_df)}, mean={null_df['mean_net_pips'].mean():+.4f}p, "
        f"p95={np.percentile(null_df['mean_net_pips'], 95):+.4f}p, "
        f"p5={np.percentile(null_df['mean_net_pips'], 5):+.4f}p, "
        f"std={null_df['mean_net_pips'].std(ddof=1):.4f}p.\n"
    )

    lines.append("## Verdict\n")
    gate3 = bool(gate_table.loc[gate_table["gate"] == 3, "pass"].iloc[0])
    gate4 = bool(gate_table.loc[gate_table["gate"] == 4, "pass"].iloc[0])
    carry_row = ft[ft["variant"] == "carry_gated"].iloc[0]
    v = []
    if gate3 and gate4:
        v.append("Gates 3-4 PASS: gated carry clears its own R10 null (95th pct) and both chronological IS halves are net-positive.")
        v.append("Per the pre-registration's decision rule, this unlocks the single sealed OOS look (H1) — pending user gate; OOS has NOT been touched by this run.")
    else:
        v.append(f"Gate 3={'PASS' if gate3 else 'FAIL'}, Gate 4={'PASS' if gate4 else 'FAIL'} — the two IS gates do not both clear.")
        v.append("Per the pre-registration's decision rule, the program stops here on this locked configuration: OOS stays sealed for a future amended shot, not opened on this run.")
    v.append(
        f"Gated carry (primary) IS mean net = {carry_row['mean_net_pips']:+.3f} p/rebalance over "
        f"{int(carry_row['n_rebalances'])} monthly rebalances (cum {carry_row['cum_net_pips']:+.1f}p, "
        f"max DD {carry_row['max_dd_pips']:+.1f}p)."
    )
    v.append("Momentum, value and the equal-weight composite are reported for completeness only and are never promoted without their own separate confirmation (pre-reg).")
    v.append("Small-N caveat (pre-reg, disclosed): ~45 IS monthly rebalances is an inherently wide-CI regime for a factor-investing horizon; the academic prior for FX carry/momentum/value carries part of the burden here, stated not hidden.")
    lines.extend(v)
    lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(out, "is_summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
