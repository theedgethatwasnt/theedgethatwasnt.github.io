#!/usr/bin/env python3
"""Live monitor: pull live_results.jsonl from each Hetzner server, print leaderboard.

Usage:
    python3 monitor_progress.py                # one-shot dump
    python3 monitor_progress.py --watch 30     # refresh every 30s
    python3 monitor_progress.py --servers IP1 IP2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

LOCAL_RESULTS = Path(__file__).resolve().parent / "results" / "live_results.jsonl"
REMOTE_PATH = "/root/elementary_ma/research/experiments/elementary_ma/results/live_results.jsonl"


def pull_from_servers(servers):
    """rsync live_results.jsonl from each server; concat to local file."""
    LOCAL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for ip in servers:
        out = f"/tmp/live_results_{ip.replace('.', '_')}.jsonl"
        try:
            subprocess.run(
                ["scp", "-q", "-o", "ConnectTimeout=5",
                 "-o", "StrictHostKeyChecking=no",
                 f"root@{ip}:{REMOTE_PATH}", out],
                check=False, timeout=30,
            )
            with open(out) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        entry["_server"] = ip
                        lines.append(entry)
        except Exception as e:
            print(f"  [warn] {ip}: {e}", file=sys.stderr)
    return lines


def print_leaderboard(entries):
    if not entries:
        print("No results yet.")
        return

    by_arm = defaultdict(list)
    for e in entries:
        by_arm[e.get("arm", "?")].append(e)

    print(f"\n{'=' * 90}")
    print(f"  ELEMENTARY-MA LEADERBOARD — {len(entries)} runs at {time.strftime('%H:%M:%S')}")
    print(f"{'=' * 90}")

    # Per-arm summary
    print(f"\n{'arm':12}{'n_runs':>8}{'avg OOS':>10}{'max OOS':>10}{'min OOS':>10}{'positive':>10}")
    for arm in sorted(by_arm):
        runs = by_arm[arm]
        pps = [r["oos_pps"] for r in runs]
        n_pos = sum(1 for p in pps if p > 0)
        avg = sum(pps) / len(pps)
        print(f"{arm:12}{len(runs):>8}{avg:>10.2f}{max(pps):>10.2f}{min(pps):>10.2f}{n_pos:>10}/{len(pps)}")

    # Top-15 runs overall
    entries.sort(key=lambda e: e.get("oos_pps", -999), reverse=True)
    print(f"\nTop 15 runs by OOS pips/day:")
    print(f"{'rank':>4}  {'arm':<11}{'pair':<9}{'seed':>5}{'OOS pps':>9}{'IS pps':>9}{'trades':>8}{'dir':>6}")
    for i, e in enumerate(entries[:15], 1):
        print(f"{i:>4}  {e.get('arm','?'):<11}{e.get('pair','?'):<9}"
              f"{e.get('seed', '?'):>5}{e.get('oos_pps', 0):>9.2f}"
              f"{e.get('is_pps', 0):>9.2f}{e.get('oos_trades', 0):>8}"
              f"{e.get('oos_dir_ratio', 0):>6.2f}")

    # Plausibility flag (learned from 4-17 RCA)
    suspicious = [e for e in entries if e.get("oos_pps", 0) > 10.0]
    if suspicious:
        print(f"\n⚠️  {len(suspicious)} run(s) with OOS > 10 p/d — flag for RCA (possible lookahead)")
        for e in suspicious[:5]:
            print(f"      {e.get('arm')} {e.get('pair')} s{e.get('seed')}: {e.get('oos_pps')} p/d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--servers", nargs="+", default=[],
                        help="Hetzner server IPs (space-separated)")
    parser.add_argument("--watch", type=int, default=0,
                        help="Refresh interval in seconds (0 = one-shot)")
    parser.add_argument("--local-only", action="store_true",
                        help="Don't pull from remote; read local live_results.jsonl")
    args = parser.parse_args()

    while True:
        if args.local_only:
            entries = []
            if LOCAL_RESULTS.exists():
                with open(LOCAL_RESULTS) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
        else:
            entries = pull_from_servers(args.servers)
        print_leaderboard(entries)
        if args.watch <= 0:
            break
        print(f"\n(refreshing in {args.watch}s — Ctrl+C to stop)")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
