#!/usr/bin/env python3
"""
Phase 9 sweep monitor.
- Watches phase9_output.txt line by line
- Sends Telegram updates as each pair completes
- Auto-deploys GBP_JPY winner to live service when found
"""
import re
import sys
import os
import time
import subprocess
import requests

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "results", "phase9_output.txt")
SERVICE_MAIN = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                             "services", "strategy_zone_recovery", "main.py")
SERVICE_MAIN = os.path.normpath(SERVICE_MAIN)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

def tg(msg: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg},
            timeout=10
        )
    except Exception as e:
        print(f"[tg error] {e}")

def parse_best_row(lines_for_pair: list) -> dict | None:
    """Extract the ◄ BEST row from the buffered lines for a pair."""
    for line in lines_for_pair:
        if "◄ BEST" not in line:
            continue
        # Format: pair zw tgt ez g n avgH $/hr@1u $/day@1ku maxDD@1ku Sharpe SQN prior
        parts = line.strip().replace("$", "").replace("◄ BEST", "").split()
        if len(parts) < 12:
            continue
        try:
            return {
                "pair":      parts[0],
                "zw":        int(parts[1]),
                "tgt":       int(parts[2]),
                "ez":        float(parts[3]),
                "gates":     int(parts[4]),
                "n":         int(parts[5]),
                "avg_hrs":   float(parts[6]),
                "pnl_hr":    float(parts[7]),
                "pnl_day":   float(parts[8]),
                "dd":        float(parts[9]),
                "sharpe":    float(parts[10]),
                "sqn":       float(parts[11]),
            }
        except (ValueError, IndexError):
            continue
    return None

def parse_no_pass(lines_for_pair: list) -> str | None:
    for line in lines_for_pair:
        if "NO PASSING" in line or "ERROR" in line:
            m = re.search(r'(EUR_\w+|GBP_\w+|AUD_\w+|NZD_\w+|USD_\w+|CAD_\w+|CHF_\w+)', line)
            pair = m.group(1) if m else "?"
            return pair
    return None

def deploy_gbpjpy(zw: int, tgt: int, ez: float, gates: int, n: int,
                  avg_hrs: float, pnl_day: float, dd: float,
                  sharpe: float, sqn: float):
    """Patch main.py with new GBP_JPY params and restart the container."""
    print(f"[deploy] Reading {SERVICE_MAIN}")
    with open(SERVICE_MAIN) as f:
        src = f.read()

    # Replace ZONE_WIDTH and TARGET_BEYOND lines
    src = re.sub(
        r"ZONE_WIDTH\s*=\s*\d+ \* PIP.*",
        f"ZONE_WIDTH    = {zw} * PIP         # {zw} pips zone width (boundary geometry)",
        src
    )
    src = re.sub(
        r"TARGET_BEYOND\s*=\s*\d+ \* PIP.*",
        f"TARGET_BEYOND = {tgt} * PIP         # {tgt} pips beyond zone boundary (E/Z = {ez:.1f})",
        src
    )
    src = re.sub(
        r"TARGET_PIPS\s*=\s*TARGET_BEYOND / PIP.*",
        f"TARGET_PIPS   = TARGET_BEYOND / PIP   # {float(tgt):.1f}",
        src
    )
    # Update docstring config line
    src = re.sub(
        r"Config: zw=\d+ pips.*",
        f"Config: zw={zw} pips, tgt={tgt} pips, E/Z={ez:.1f}, ml=10, break-even sizing",
        src
    )
    # Update Phase reference line
    src = re.sub(
        r"Phase \d+ result:.*",
        f"Phase 9 result: GBP_JPY zw={zw} tgt={tgt} — ${pnl_day:.0f}/day@1ku, "
        f"SQN {sqn:.1f}, {gates}/5 gates, {n} OOS cycles, avg {avg_hrs:.1f}h.",
        src
    )

    with open(SERVICE_MAIN, "w") as f:
        f.write(src)
    print(f"[deploy] Patched {SERVICE_MAIN}")

    # Restart container via docker compose
    result = subprocess.run(
        ["docker", "compose", "restart", "fx-zone-recovery"],
        cwd="/path/to/projects/fx-core",
        capture_output=True, text=True
    )
    print(f"[deploy] docker restart: {result.stdout.strip()} {result.stderr.strip()}")
    return result.returncode == 0


def main():
    tg("🔬 Phase 9 zone recovery sweep started — 12 pairs, break-even sizing, wide grid.")
    print("[monitor] Watching", OUTPUT_FILE)

    current_pair = None
    pair_lines   = []
    finished     = False
    gbpjpy_deployed = False

    with open(OUTPUT_FILE, "r") as f:
        while True:
            line = f.readline()
            if not line:
                if finished:
                    break
                time.sleep(1)
                continue

            print(line, end="", flush=True)
            stripped = line.strip()

            # Detect pair start
            m = re.match(r'\s+(EUR_\w+|GBP_\w+|AUD_\w+|NZD_\w+|USD_\w+|CAD_\w+|CHF_\w+)', line)
            if m:
                new_pair = m.group(1)
                if new_pair != current_pair:
                    current_pair = new_pair
                    pair_lines   = [line]
                else:
                    pair_lines.append(line)
            elif current_pair:
                pair_lines.append(line)

            # Detect "NO PASSING CONFIGS" message
            if "NO PASSING" in stripped and current_pair:
                tg(f"🔴 {current_pair}: no passing configs")
                current_pair = None
                pair_lines   = []

            # Detect pair separator (end of pair block)
            if re.search(r'\[\d+s\]', stripped) and current_pair:
                winner = parse_best_row(pair_lines)
                if winner:
                    pair = winner["pair"]
                    msg = (
                        f"✅ {pair} done — zw={winner['zw']}p tgt={winner['tgt']}p "
                        f"E/Z={winner['ez']:.1f} {winner['gates']}/5 gates\n"
                        f"n={winner['n']} avg={winner['avg_hrs']:.1f}h "
                        f"${winner['pnl_day']:.2f}/day SQN={winner['sqn']:.1f} "
                        f"Sharpe={winner['sharpe']:.2f}"
                    )
                    tg(msg)
                    print(f"[tg] {msg}")

                    # Auto-deploy GBP_JPY winner
                    if pair == "GBP_JPY" and not gbpjpy_deployed:
                        ok = deploy_gbpjpy(
                            winner["zw"], winner["tgt"], winner["ez"],
                            winner["gates"], winner["n"], winner["avg_hrs"],
                            winner["pnl_day"], winner["dd"],
                            winner["sharpe"], winner["sqn"]
                        )
                        if ok:
                            tg(
                                f"🚀 GBP_JPY LIVE UPDATED → zw={winner['zw']}p "
                                f"tgt={winner['tgt']}p (E/Z={winner['ez']:.1f})\n"
                                f"Container restarted. Running on accts 011/012."
                            )
                            gbpjpy_deployed = True
                        else:
                            tg("⚠️ GBP_JPY deploy failed — check logs.")
                else:
                    tg(f"🔴 {current_pair}: no passing configs")

                current_pair = None
                pair_lines   = []

            # Detect final summary
            if "WINNER PER PAIR" in stripped:
                finished = True

            # Detect end of file / process done
            if "Saved" in stripped and "rows" in stripped:
                finished = True

    # Send final summary
    tg("📊 Phase 9 sweep complete — see JOURNEY for full results.")
    print("[monitor] Done.")


if __name__ == "__main__":
    main()
