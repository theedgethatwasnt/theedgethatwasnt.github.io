#!/usr/bin/env python3
"""run_with_telegram.py — orchestrate the CMA-NN exit-learner end-to-end with
Telegram progress updates:

  1. Poll for chopper completion (samples_USD_JPY.parquet + meta_USD_JPY.parquet).
  2. When detected → ping Telegram "chopper done, starting trainer".
  3. Launch train_cma_exit.py with output to a log file.
  4. Tail the log for generation lines and forward milestones to Telegram.
  5. On training completion → send final summary (best fitness, OOS report).
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Add fx-core lib to path so we can import notify
PROJ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ_ROOT))
from lib.notify import _send as tg_send

HERE = Path(__file__).resolve().parent
SAMPLES_PATH = HERE / "samples_USD_JPY.parquet"
META_PATH    = HERE / "meta_USD_JPY.parquet"
TRAIN_LOG    = HERE / "train_cma_exit.log"
TRAIN_SCRIPT = HERE / "train_cma_exit.py"

GEN_LINE   = re.compile(r"^Gen\s+(\d+)\s")
RESTART_LN = re.compile(r"STAGNATION|IPOP restart|restart #")
OOS_HEADER = re.compile(r"AMDDP[15]?0?\s+sum")

POLL_SECS = 30
MILESTONE_EVERY = 10        # generations
MAX_WAIT_FOR_CHOPPER = 7200 # seconds (2h); chopper should finish much sooner


def wait_for_chopper() -> bool:
    """Poll until both chopper outputs exist, or timeout."""
    tg_send(f"⏳ <b>CMA-NN exit-learner orchestrator</b>\nWaiting for chopper output:\n<code>{SAMPLES_PATH.name}</code>\n<code>{META_PATH.name}</code>")
    t0 = time.time()
    while time.time() - t0 < MAX_WAIT_FOR_CHOPPER:
        if SAMPLES_PATH.exists() and META_PATH.exists():
            return True
        time.sleep(POLL_SECS)
    tg_send(f"❌ <b>Chopper timeout</b> ({MAX_WAIT_FOR_CHOPPER//60} min elapsed). Aborting.")
    return False


def chopper_summary() -> str:
    """Quick stats from the meta parquet so the first Telegram message is informative."""
    try:
        import pandas as pd
        meta = pd.read_parquet(META_PATH)
        n_total = len(meta)
        n_is = (meta['split'] == 'IS').sum() if 'split' in meta.columns else 0
        n_oos = (meta['split'] == 'OOS').sum() if 'split' in meta.columns else 0
        n_long = (meta['direction'] == 1).sum() if 'direction' in meta.columns else 0
        n_short = (meta['direction'] == -1).sum() if 'direction' in meta.columns else 0
        size_mb = SAMPLES_PATH.stat().st_size / (1024 * 1024)
        return (f"<b>Chopper done.</b> {n_total} events.\n"
                f"<code>IS={n_is}  OOS={n_oos}  long={n_long}  short={n_short}</code>\n"
                f"<code>samples on disk: {size_mb:.1f} MB</code>")
    except Exception as e:
        return f"<b>Chopper done.</b> Meta read error: {e}"


OOS_AMDDP5 = re.compile(r"AMDDP5\s+sum\s*=\s*([+-]?[\d.,]+)", re.IGNORECASE)


def stream_trainer(gens: int, mode: str) -> float | None:
    """Launch trainer for one mode, tail stdout for milestones, return parsed
    OOS AMDDP5 sum (or None if not found)."""
    log_path = HERE / f"train_{mode}.log"
    tg_send(f"🟢 <b>CMA-NN trainer — {mode.upper()}</b>\n<code>--gens {gens} --pair USD_JPY --mode {mode}</code>\nMilestones every {MILESTONE_EVERY} gens.")

    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        ["python3", str(TRAIN_SCRIPT), "--gens", str(gens),
         "--pair", "USD_JPY", "--mode", mode],
        cwd=str(PROJ_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )

    last_milestone_gen = -1
    last_gen_line = ""
    oos_buffer = []
    in_oos_section = False
    oos_amddp5 = None

    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            log_f.write(raw); log_f.flush()

            m = GEN_LINE.match(line)
            if m:
                last_gen_line = line
                gen = int(m.group(1))
                if gen >= last_milestone_gen + MILESTONE_EVERY:
                    last_milestone_gen = gen
                    tg_send(f"📊 <b>{mode} · Gen {gen}/{gens}</b>\n<code>{line}</code>")

            if RESTART_LN.search(line):
                tg_send(f"🔄 <b>{mode} · restart</b>\n<code>{line}</code>")

            am = OOS_AMDDP5.search(line)
            if am:
                try:
                    oos_amddp5 = float(am.group(1).replace(",", ""))
                except ValueError:
                    pass

            if "OOS" in line and "report" in line.lower():
                in_oos_section = True
            if in_oos_section:
                oos_buffer.append(line)
                if "Exit causes" in line or "Hold to time cap" in line:
                    tg_send(f"📋 <b>{mode} · Final OOS</b>\n<pre>" + "\n".join(oos_buffer[-30:]) + "</pre>")
                    oos_buffer = []
                    in_oos_section = False
    finally:
        log_f.close()

    proc.wait()
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-25:] if log_path.exists() else []
        tg_send(f"⚠️ <b>{mode} finished rc={proc.returncode}</b>\n<pre>{chr(10).join(tail)[-2000:]}</pre>")
    return oos_amddp5


def run_both(gens: int):
    results = {}
    for mode in ("continuation", "fade"):
        results[mode] = stream_trainer(gens, mode)

    c = results.get("continuation")
    f = results.get("fade")
    lines = ["🏁 <b>A/B complete — OOS AMDDP5 sum</b>"]
    lines.append(f"<code>continuation: {c if c is not None else 'n/a'}</code>")
    lines.append(f"<code>fade:         {f if f is not None else 'n/a'}</code>")
    if c is not None and f is not None:
        winner = "FADE (shocks revert)" if f > c else "CONTINUATION (shocks continue)"
        lines.append(f"<b>Winner: {winner}</b>")
        if max(c, f) < 0:
            lines.append("⚠️ Both negative — no tradeable direction; momentum-shock entry has no edge.")
    tg_send("\n".join(lines))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--mode", choices=["continuation", "fade", "both"], default="both")
    ap.add_argument("--no-wait", action="store_true",
                    help="Skip chopper polling (data already in place)")
    args = ap.parse_args()

    # The trainer now sources features_USD_JPY.parquet + meta3_USD_JPY.parquet.
    FEAT = HERE / "features_USD_JPY.parquet"
    META3 = HERE / "meta3_USD_JPY.parquet"

    if not args.no_wait:
        if not wait_for_chopper():
            sys.exit(1)

    if not FEAT.exists() or not META3.exists():
        tg_send(f"❌ Missing trainer inputs:\n<code>{FEAT.name}</code> exists={FEAT.exists()}\n<code>{META3.name}</code> exists={META3.exists()}")
        sys.exit(1)

    if args.mode == "both":
        run_both(args.gens)
    else:
        stream_trainer(args.gens, args.mode)
