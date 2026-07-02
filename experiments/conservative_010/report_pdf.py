"""
report_pdf.py — Multi-page PDF report for the conservative-010 backtest.

Pages:
  p1 — Title + verdict text + headline IS/OOS pips + max-DD
  p2 — Equity + drawdown figures (matplotlib Figure objects) or placeholder text
  p3 — Per-year/regime notes + walk-forward fold table (summary['wf'])
  p4 — Monte-Carlo p-values + margin-realizable survival verdict table

All rendering via matplotlib PdfPages (no LaTeX required).
"""

import os
import textwrap
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _new_fig(title: str):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    return fig, ax


def _mono(ax, text: str, x: float = 0.02, y: float = 0.92, fontsize: int = 8):
    """Render monospace text block."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontfamily="monospace", fontsize=fontsize,
            verticalalignment="top", wrap=False)


# ---------------------------------------------------------------------------
# page builders
# ---------------------------------------------------------------------------

def _page1(pdf: PdfPages, summary: dict) -> None:
    """Title + verdict + headline numbers."""
    fig, ax = _new_fig("Conservative-010 Backtest Report")

    verdict = str(summary.get("verdict", "—"))
    net_is  = summary.get("net_is",  "N/A")
    net_oos = summary.get("net_oos", "N/A")
    max_dd  = summary.get("max_dd",  "N/A")

    lines = [
        "=" * 60,
        "VERDICT",
        "=" * 60,
        "",
    ]
    # wrap long verdict text at 60 chars
    for para in verdict.split("\n"):
        lines.extend(textwrap.wrap(para, width=60) or [""])

    lines += [
        "",
        "=" * 60,
        "HEADLINE NUMBERS",
        "=" * 60,
        f"  Net IS   : {net_is} pips",
        f"  Net OOS  : {net_oos} pips",
        f"  Max DD   : {max_dd} pips",
    ]

    _mono(ax, "\n".join(lines), fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def _page2(pdf: PdfPages, figs: dict) -> None:
    """Equity + drawdown figures or placeholder."""
    equity_fig   = figs.get("equity")
    drawdown_fig = figs.get("drawdown")

    if equity_fig is not None:
        pdf.savefig(equity_fig)
        plt.close(equity_fig)
    else:
        fig, ax = _new_fig("Equity Curve")
        _mono(ax, "(equity figure not provided — run backtest to generate)", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

    if drawdown_fig is not None:
        pdf.savefig(drawdown_fig)
        plt.close(drawdown_fig)
    else:
        fig, ax = _new_fig("Drawdown")
        _mono(ax, "(drawdown figure not provided — run backtest to generate)", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)


def _page3(pdf: PdfPages, summary: dict) -> None:
    """Per-year/regime notes + walk-forward fold table."""
    fig, ax = _new_fig("Walk-Forward Fold Results")

    wf = summary.get("wf", [])

    # header
    hdr  = f"{'Fold':>5}  {'N':>6}  {'Net (p)':>10}  {'WR (%)':>8}"
    sep  = "-" * len(hdr)
    rows = [hdr, sep]

    for f in wf:
        fold = f.get("fold", "?")
        n    = f.get("n",    0)
        net  = f.get("net",  0.0)
        wr   = f.get("wr",   0.0)
        rows.append(f"{fold:>5}  {n:>6}  {net:>10.2f}  {wr:>8.1f}")

    if not wf:
        rows.append("  (no walk-forward folds available)")

    notes = [
        "",
        "Notes",
        "-----",
        "• IS = in-sample (training) period",
        "• OOS = out-of-sample (test) period",
        "• Each fold uses an expanding IS window",
        "• WR = win rate of closed trades in that fold",
    ]

    _mono(ax, "\n".join(rows + notes), fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def _page4(pdf: PdfPages, summary: dict) -> None:
    """Monte-Carlo p-values + margin survival verdict table."""
    fig, ax = _new_fig("Monte-Carlo & Margin Simulation")

    mc     = summary.get("mc", {})
    margin = summary.get("margin", [])

    p_net   = mc.get("p_net",   "N/A")
    p_maxdd = mc.get("p_maxdd", "N/A")

    mc_lines = [
        "Monte-Carlo (trade shuffle, n=1000)",
        "------------------------------------",
        f"  p_net   = {p_net}  (fraction of shuffles ≥ observed net)",
        f"  p_maxdd = {p_maxdd}  (fraction of shuffles ≤ observed max-DD)",
        "",
    ]

    # margin survival table
    mhdr = (f"{'Start$':>8}  {'Halted':>7}  {'Reason':<18}  "
            f"{'Final$':>8}  {'MaxDD$':>8}  {'MaxUtil':>8}")
    msep = "-" * len(mhdr)
    mrows = [
        "Margin-Realizable Survival Scenarios",
        "-------------------------------------",
        mhdr, msep,
    ]
    for m in margin:
        halted  = "YES" if m.get("halted") else "no"
        reason  = str(m.get("halted_reason", ""))[:18]
        mrows.append(
            f"{m.get('start_balance', 0):>8.2f}  {halted:>7}  {reason:<18}  "
            f"{m.get('final_balance', 0):>8.2f}  {m.get('max_dd_usd', 0):>8.2f}  "
            f"{m.get('max_util', 0):>8.3f}"
        )
    if not margin:
        mrows.append("  (no margin scenarios available)")

    _mono(ax, "\n".join(mc_lines + mrows), fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build_pdf(summary: dict, figs: dict, out_path: str) -> str:
    """Build a 4-page PDF report and return the path.

    Parameters
    ----------
    summary : dict
        Keys: verdict, net_is, net_oos, max_dd, wf (list), mc (dict), margin (list).
    figs : dict
        Optional pre-rendered matplotlib Figure objects: 'equity', 'drawdown'.
    out_path : str
        Destination file path (will be created / overwritten).

    Returns
    -------
    str
        Absolute path to the written PDF.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with PdfPages(out_path) as pdf:
        _page1(pdf, summary)
        _page2(pdf, figs)
        _page3(pdf, summary)
        _page4(pdf, summary)

        # embed metadata
        d = pdf.infodict()
        d["Title"]   = "Conservative-010 Backtest Report"
        d["Subject"] = "FX-Core conservative backtest with margin simulation"

    return os.path.abspath(out_path)


def send_telegram_document(path: str, caption: str) -> Optional[int]:
    """POST a document to Telegram.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
    Returns the message_id on success, None on failure.
    Does NOT print the token.
    """
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID",   "")

    if not token or not chat_id:
        return None

    url = f"https://api.telegram.org/bot{token}/sendDocument"

    try:
        import requests  # type: ignore
        with open(path, "rb") as fh:
            resp = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption},
                files={"document": fh},
                timeout=30,
            )
        data = resp.json()
        if data.get("ok"):
            return data["result"]["message_id"]
        return None
    except ImportError:
        pass

    # fallback: subprocess curl
    import subprocess
    result = subprocess.run(
        [
            "curl", "-s",
            "-F", f"chat_id={chat_id}",
            "-F", f"caption={caption}",
            "-F", f"document=@{path}",
            url,
        ],
        capture_output=True, text=True, timeout=30,
    )
    try:
        import json
        data = json.loads(result.stdout)
        if data.get("ok"):
            return data["result"]["message_id"]
    except Exception:
        pass
    return None
