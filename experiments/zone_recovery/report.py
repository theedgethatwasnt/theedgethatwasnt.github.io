"""
Zone Recovery Experiment Report Generator
Produces markdown report + summary tables.
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"


def generate_report() -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Zone Recovery Experiment Report",
        f"Generated: {ts}",
        f"Pair: EUR_USD | Data: M5 2021-2026 | Post-RCA causal standards\n",
    ]

    # Phase 1: Classic grid
    p1_path = RESULTS_DIR / "phase1_classic_grid.csv"
    if p1_path.exists():
        df1 = pd.read_csv(p1_path).sort_values("sharpe", ascending=False)
        lines += [
            "## Phase 1: Classic cBot Parameter Grid",
            f"Configs tested: {len(df1)}",
            "",
            "### Top 10 by Sharpe",
            df1.head(10)[["half_zone_pips", "target_beyond_pips", "profit_factor",
                           "n_cycles", "net_pnl_pips", "sharpe", "win_rate",
                           "avg_legs", "max_legs_hit_pct"]].to_markdown(index=False),
            "",
            f"**Best**: hz={df1.iloc[0]['half_zone_pips']}p tgt={df1.iloc[0]['target_beyond_pips']}p "
            f"pf={df1.iloc[0]['profit_factor']:.2f} → "
            f"pnl={df1.iloc[0]['net_pnl_pips']:+.1f}p sharpe={df1.iloc[0]['sharpe']:.3f}",
            "",
        ]

    # Phase 2: ATR grid
    p2_path = RESULTS_DIR / "phase2_atr_grid.csv"
    if p2_path.exists():
        df2 = pd.read_csv(p2_path).sort_values("sharpe", ascending=False)
        lines += [
            "## Phase 2: ATR-Calibrated Parameter Grid",
            f"Configs tested: {len(df2)}",
            "",
            "### Top 10 by Sharpe",
            df2.head(10)[["half_zone_mult", "target_mult", "median_ez_ratio",
                           "n_cycles", "net_pnl_pips", "sharpe", "sqn",
                           "avg_legs"]].to_markdown(index=False),
            "",
        ]

    # Phase 3: Validation
    p3_path = RESULTS_DIR / "phase3_validation.json"
    if p3_path.exists():
        with open(p3_path) as f:
            v_results = json.load(f)

        passes = [r for r in v_results if r["verdict"] == "PASS"]
        lines += [
            "## Phase 3: 5-Gate Walk-Forward Validation",
            f"Configs validated: {len(v_results)} | PASS: {len(passes)}/{len(v_results)}",
            "",
        ]

        for r in v_results:
            icon = "🟢" if r["verdict"] == "PASS" else "🔴"
            lines += [
                f"### {icon} {r['config']} [{r['verdict']}]",
                f"Gates passed: {r['gates_passed']}/5",
                "",
            ]
            for gname, gval in r["gates"].items():
                gi = "✅" if gval else "❌"
                lines.append(f"- {gi} {gname}")

            oos = r["oos_metrics"]
            lines += [
                "",
                f"OOS: pnl={oos['net_pnl_pips']:+.1f}p | sharpe={oos['sharpe']:.3f} | "
                f"sqn={oos['sqn']:.2f} | n={oos['n_cycles']} trades",
                f"Win rate: {oos['win_rate']*100:.1f}% | MaxDD: {oos['max_drawdown_pips']:.1f}p",
                "",
            ]

    # Final verdict
    lines += [
        "## Verdict",
        "",
    ]
    p3_path = RESULTS_DIR / "phase3_validation.json"
    if p3_path.exists():
        with open(p3_path) as f:
            v_results = json.load(f)
        passes = [r for r in v_results if r["verdict"] == "PASS"]
        if passes:
            best = max(passes, key=lambda r: r["oos_metrics"]["net_pnl_pips"])
            lines += [
                f"🟢 **DEPLOYABLE** candidate found: `{best['config']}`",
                f"- OOS pnl: {best['oos_metrics']['net_pnl_pips']:+.1f} pips",
                f"- OOS Sharpe: {best['oos_metrics']['sharpe']:.3f}",
                f"- SQN: {best['oos_metrics']['sqn']:.2f}",
                f"- Gates: {best['gates_passed']}/5",
                "",
                "**Next step**: Deploy to OANDA accounts 011 (long) + 012 (short) at 0.001 lot, 1-week shadow validation.",
            ]
        else:
            lines += [
                "🔴 **No deployable candidates** found. All configs fail ≥2 gates.",
                "Recommended next steps:",
                "- Try session-filtered entries (London/NY hours only)",
                "- Try convergent zone shape (ATR trending down = zone narrows)",
                "- Investigate directional bias entry (not random)",
            ]

    report = "\n".join(lines)

    report_path = RESULTS_DIR / "zone_recovery_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")
    return report


if __name__ == "__main__":
    print(generate_report())
