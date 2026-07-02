#!/usr/bin/env python3
"""
Conservative 010 orchestrator — full backtest pipeline with real BA spreads,
2-pip stop slippage, portfolio IS/OOS + WF + MC, margin simulation at $100/$500/$1000,
PDF report, and Telegram delivery.

Usage: cd /path/to/projects/fx-core && python3 research/experiments/conservative_010/run.py

R3b: hard-fails if any pair's BA coverage is incomplete (load_pair_ba asserts).
"""
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project paths ─────────────────────────────────────────────────────────────
PROJECT   = Path("/path/to/projects/fx-core")
SMA_EXITS = PROJECT / "research" / "experiments" / "sma_exits"
CONS010   = PROJECT / "research" / "experiments" / "conservative_010"
sys.path.insert(0, str(SMA_EXITS))
sys.path.insert(0, str(CONS010))

from data        import load_pair_ba, CFG
from engine      import backtest_pair
from validate    import equity_drawdown, split_is_oos, walk_forward, monte_carlo
from margin_sim  import simulate_account
from report_pdf  import build_pdf, send_telegram_document
from _lib        import IS_FRAC

PAIRS   = ["EUR_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
OUT_DIR = PROJECT / "data" / "conservative_010"
PDF_PATH = OUT_DIR / "report.pdf"
MD_PATH  = OUT_DIR / "report.md"


# ── helpers ───────────────────────────────────────────────────────────────────

def _ts_to_sortable(ts_val) -> int:
    """Convert any timestamp representation to a sortable int (unix ns)."""
    import pandas as pd
    return int(pd.Timestamp(ts_val).value)


def _ts_to_year(ts_val) -> int:
    """Convert any timestamp representation (Timestamp, datetime64, int ns) to year."""
    import pandas as pd
    return int(pd.Timestamp(ts_val).year)


def _make_equity_fig(cum_pips: np.ndarray, title: str = "Portfolio Cumulative P&L") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cum_pips, linewidth=0.8, color="#1f77b4")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative pips")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _make_drawdown_fig(cum_pips: np.ndarray, title: str = "Portfolio Drawdown") -> plt.Figure:
    peak = np.maximum.accumulate(cum_pips)
    dd   = cum_pips - peak
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(range(len(dd)), dd, 0, color="#d62728", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Drawdown (pips)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _make_peryear_fig(peryear: dict, title: str = "Portfolio Net Pips by Year") -> plt.Figure:
    if not peryear:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes)
        return fig
    years  = sorted(peryear.keys())
    values = [peryear[y] for y in years]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(y) for y in years], values, color=colors)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Net pips")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Conservative 010 Backtest")
    print("Real BA spreads | 2p stop slippage | fence 200p | flip exits")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: per-pair load + backtest ──────────────────────────────────────
    pair_data   = {}
    pair_trades = {}

    for pair in PAIRS:
        print(f"\n[{pair}] loading BA data ...")
        d = load_pair_ba(pair)          # R3b asserts BA 100% finite — hard-fail if not
        pair_data[pair] = d

        print(f"[{pair}] running conservative backtest ...")
        trades = backtest_pair(d, CFG[pair], slippage_pips=2.0, no_flip=False)
        pair_trades[pair] = trades

        net  = trades['pnl_pips_net']
        n    = len(trades)
        rsn  = trades['exit_reason_code']
        fence_n = int((rsn == 0).sum())
        psar_n  = int((rsn == 1).sum())
        tp_n    = int((rsn == 2).sum())
        flip_n  = int((rsn == 3).sum())

        # per-pair IS/OOS (using that pair's bar-index is_end)
        eb       = trades['entry_bar']
        is_end_p = d['is_end']
        is_mask  = eb < is_end_p
        oos_mask = ~is_mask
        is_net_p  = float(net[is_mask].sum())  if is_mask.any()  else 0.0
        oos_net_p = float(net[oos_mask].sum()) if oos_mask.any() else 0.0
        exp_p = float(net.mean()) if n else 0.0
        wr_p  = float((net > 0).mean() * 100) if n else 0.0

        print(f"  {n} trades | exp={exp_p:+.2f}p | WR={wr_p:.0f}% | "
              f"IS={is_net_p:+.0f}p | OOS={oos_net_p:+.0f}p")
        print(f"  exits: fence={fence_n} psar={psar_n} tp={tp_n} flip={flip_n}")

    # ── Step 2: portfolio net-pip stream ──────────────────────────────────────
    print("\nBuilding portfolio stream ...")

    rows = []   # list of (exit_ts_sortable, pnl_pips_net, pair, raw_ts)
    for pair in PAIRS:
        d      = pair_data[pair]
        trades = pair_trades[pair]
        ts     = d['ts']
        for t in trades:
            exit_ts = ts[t['exit_bar']]
            rows.append((_ts_to_sortable(exit_ts), float(t['pnl_pips_net']), pair, exit_ts))

    # Sort by exit time (time-ordered portfolio)
    rows.sort(key=lambda r: r[0])

    exit_times  = np.array([r[0] for r in rows], dtype=np.int64)   # unix ns sortable ints
    port_net    = np.array([r[1] for r in rows], dtype=np.float64)
    port_pairs  = [r[2] for r in rows]
    exit_ts_raw = [r[3] for r in rows]   # original timestamp objects for year extraction
    n_total     = len(rows)

    # Global time-rank indices (0..n_total-1) for IS/OOS + WF
    port_entry_bars = np.arange(n_total, dtype=np.int64)
    # Portfolio IS split: first IS_FRAC fraction of time-sorted trades
    is_end_port = int(n_total * IS_FRAC)

    cum_port = np.cumsum(port_net)
    total_net = float(cum_port[-1]) if n_total else 0.0
    print(f"  Portfolio: {n_total} trades | total net={total_net:+.0f}p")

    # ── Step 3: statistics ────────────────────────────────────────────────────
    print("\nComputing IS/OOS split, walk-forward, Monte Carlo ...")

    iso = split_is_oos(port_entry_bars, port_net, is_end=is_end_port)
    wf  = walk_forward(port_entry_bars, port_net, n_total=n_total, n_folds=6)
    mc  = monte_carlo(port_net, n=300, seed=0)
    dd  = equity_drawdown(cum_port)

    print(f"  IS  : n={iso['is_n']:>5} | net={iso['is_net']:>+8.0f}p")
    print(f"  OOS : n={iso['oos_n']:>5} | net={iso['oos_net']:>+8.0f}p | WR={iso['oos_wr']:.1f}%")
    print(f"  Max DD   : {dd['max_dd']:+.0f}p | longest underwater: {dd['longest_underwater']} trades")
    print(f"  MC p_net : {mc['p_net']:.3f}   MC p_maxdd : {mc['p_maxdd']:.3f}")
    print("\n  Walk-Forward Folds:")
    print(f"  {'Fold':>5}  {'N':>6}  {'Net (p)':>10}  {'WR (%)':>8}")
    wf_positive = 0
    for f in wf:
        sign = "+" if f['net'] >= 0 else " "
        print(f"  {f['fold']:>5}  {f['n']:>6}  {sign}{f['net']:>9.0f}  {f['wr']:>7.1f}%")
        if f['net'] > 0:
            wf_positive += 1
    print(f"  → {wf_positive}/{len(wf)} positive folds")

    # Per-year breakdown
    peryear: dict = {}
    for ts_val, pnl in zip(exit_ts_raw, port_net):
        yr = _ts_to_year(ts_val)
        peryear[yr] = peryear.get(yr, 0.0) + pnl
    print("\n  Per-year net pips:")
    for yr in sorted(peryear):
        print(f"    {yr}: {peryear[yr]:+.0f}p")

    # ── Step 4: margin simulation ──────────────────────────────────────────────
    print("\nRunning margin simulation ...")

    # Build list of {exit_time, pnl_pips, pair} dicts for simulate_account
    margin_trades = [{"exit_time": exit_times[i], "pnl_pips": port_net[i], "pair": port_pairs[i]}
                     for i in range(n_total)]

    margin_results = []
    for start_bal in [100, 500, 1000]:
        r = simulate_account(margin_trades, start_balance=start_bal)
        margin_results.append(r)
        status = "HALTED" if r['halted'] else "survived"
        print(f"  ${start_bal:>5}: {status:>8} | final=${r['final_balance']:>8.2f} | "
              f"max_dd=${r['max_dd_usd']:>7.2f} | max_util={r['max_util']:.3f}"
              + (f" | reason={r['halted_reason']} @trade={r['halted_at']}" if r['halted'] else ""))

    # ── Step 5: figures ────────────────────────────────────────────────────────
    print("\nRendering figures ...")
    figs = {
        "equity":   _make_equity_fig(cum_port),
        "drawdown": _make_drawdown_fig(cum_port),
        "peryear":  _make_peryear_fig(peryear),
    }

    # ── Step 6: compose summary + honest verdict ───────────────────────────────
    net_is_fmt  = f"{iso['is_net']:+.0f}"
    net_oos_fmt = f"{iso['oos_net']:+.0f}"
    max_dd_fmt  = f"{dd['max_dd']:+.0f}"

    wf_summary = f"{wf_positive}/{len(wf)} positive folds"
    mc_summary = f"p_net={mc['p_net']:.3f} p_maxdd={mc['p_maxdd']:.3f}"

    margin_lines = []
    for r in margin_results:
        bal = r['start_balance']
        if r['halted']:
            margin_lines.append(f"${bal} HALTED ({r['halted_reason']})")
        else:
            margin_lines.append(f"${bal} survived (final ${r['final_balance']:.2f})")
    margin_summary = "; ".join(margin_lines)

    span_start = _ts_to_year(exit_ts_raw[0])  if n_total else "?"
    span_end   = _ts_to_year(exit_ts_raw[-1]) if n_total else "?"

    verdict = (
        f"Conservative 010 backtest ({span_start}–{span_end}, {n_total} portfolio trades, "
        f"4 pairs: EUR_JPY / EUR_USD / GBP_USD / USD_JPY). "
        f"Fills use real per-bar bid/ask spread + 2-pip slippage on fence/PSAR stops; "
        f"TP fills at the limit level; flip exits at bar-close bid/ask. "
        f"IS net ({IS_FRAC*100:.0f}% of trades): {net_is_fmt}p. "
        f"OOS net: {net_oos_fmt}p (OOS WR {iso['oos_wr']:.1f}%). "
        f"Max drawdown: {max_dd_fmt}p (longest underwater: {dd['longest_underwater']} trades). "
        f"Walk-forward: {wf_summary}. "
        f"Monte Carlo (n=300): {mc_summary}. "
        f"Margin simulation: {margin_summary}. "
        f"This backtest supersedes the rosy 4-week live read and the fixed-spread equity "
        f"curve — it includes real spreads, the 200p fence, and finite-margin sizing. "
        f"OOS being {'positive' if iso['oos_net'] > 0 else 'NEGATIVE'} with p_net={mc['p_net']:.3f} "
        f"({'significant' if mc['p_net'] < 0.05 else 'NOT significant at 5%'}) "
        f"is the honest read of 010's conservative edge."
    )

    summary = {
        "verdict": verdict,
        "net_is":  net_is_fmt,
        "net_oos": net_oos_fmt,
        "max_dd":  max_dd_fmt,
        "wf":      wf,
        "mc":      mc,
        "margin":  margin_results,
    }

    # ── Step 7: build PDF + write report.md ───────────────────────────────────
    print(f"\nBuilding PDF → {PDF_PATH} ...")
    pdf_path = build_pdf(summary, figs, str(PDF_PATH))

    # Confirm PDF was written
    pdf_size = Path(pdf_path).stat().st_size
    print(f"  PDF written: {pdf_path} ({pdf_size:,} bytes)")

    # Write report.md
    md_lines = [
        "# Conservative 010 Backtest Report",
        "",
        "## Verdict",
        "",
    ]
    for para in verdict.split(". "):
        md_lines.append(para.strip() + ".")
        md_lines.append("")
    md_lines += [
        "## Headline Numbers",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Portfolio trades | {n_total} |",
        f"| IS net (pips) | {net_is_fmt} |",
        f"| OOS net (pips) | {net_oos_fmt} |",
        f"| OOS win rate | {iso['oos_wr']:.1f}% |",
        f"| Max drawdown (pips) | {max_dd_fmt} |",
        f"| MC p_net | {mc['p_net']:.3f} |",
        f"| MC p_maxdd | {mc['p_maxdd']:.3f} |",
        "",
        "## Walk-Forward Folds",
        "",
        "| Fold | N | Net (p) | WR (%) |",
        "|------|---|---------|--------|",
    ]
    for f in wf:
        md_lines.append(f"| {f['fold']} | {f['n']} | {f['net']:+.0f} | {f['wr']:.1f}% |")
    md_lines += [
        "",
        "## Margin Simulation",
        "",
        "| Start $ | Halted | Reason | Final $ | Max DD $ | Max Util |",
        "|---------|--------|--------|---------|----------|----------|",
    ]
    for r in margin_results:
        md_lines.append(
            f"| ${r['start_balance']} | {'YES' if r['halted'] else 'no'} | "
            f"{r['halted_reason'] or ''} | ${r['final_balance']:.2f} | "
            f"${r['max_dd_usd']:.2f} | {r['max_util']:.3f} |"
        )
    md_lines += [
        "",
        "## Per-Year Net Pips",
        "",
        "| Year | Net (p) |",
        "|------|---------|",
    ]
    for yr in sorted(peryear):
        md_lines.append(f"| {yr} | {peryear[yr]:+.0f} |")

    MD_PATH.write_text("\n".join(md_lines) + "\n")
    print(f"  report.md written: {MD_PATH}")

    # ── Step 8: send to Telegram ───────────────────────────────────────────────
    caption = (
        f"Conservative 010 backtest | "
        f"IS: {net_is_fmt}p | OOS: {net_oos_fmt}p | "
        f"MaxDD: {max_dd_fmt}p | WF: {wf_summary} | "
        f"MC p_net: {mc['p_net']:.3f} | "
        f"Margin: {margin_summary}"
    )
    print(f"\nSending PDF to Telegram ...")
    msg_id = send_telegram_document(pdf_path, caption)

    if msg_id is not None:
        print(f"  Telegram message_id: {msg_id}")
    else:
        print("  Telegram delivery failed (no token/chat_id or API error).")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CONSERVATIVE 010 BACKTEST — FINAL VERDICT")
    print("=" * 65)
    for line in textwrap.wrap(verdict, width=63):
        print(f"  {line}")
    print("=" * 65)
    print(f"  Portfolio trades : {n_total}")
    print(f"  IS net           : {net_is_fmt} pips")
    print(f"  OOS net          : {net_oos_fmt} pips")
    print(f"  OOS WR           : {iso['oos_wr']:.1f}%")
    print(f"  Max drawdown     : {max_dd_fmt} pips")
    print(f"  WF folds+        : {wf_summary}")
    print(f"  MC p_net         : {mc['p_net']:.3f}")
    print(f"  MC p_maxdd       : {mc['p_maxdd']:.3f}")
    print("  Margin survival  :")
    for r in margin_results:
        status = "HALTED" if r['halted'] else "survived"
        print(f"    ${r['start_balance']:>5}: {status}")
    print(f"  Telegram msg_id  : {msg_id}")
    print("=" * 65)

    return {
        "n_total":   n_total,
        "net_is":    iso['is_net'],
        "net_oos":   iso['oos_net'],
        "max_dd":    dd['max_dd'],
        "wf":        wf,
        "wf_pos":    wf_positive,
        "mc":        mc,
        "margin":    margin_results,
        "msg_id":    msg_id,
        "pdf_path":  pdf_path,
    }


if __name__ == "__main__":
    main()
