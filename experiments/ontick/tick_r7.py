"""
tick_r7.py -- determinism / consistency gate (SOP R7).

Replays an OANDA tick capture from /data/ticks/*.jsonl (format
{"ms": int, "p": pair, "b": bid, "a": ask}) through the SAME engine TWICE
and asserts byte-identical state evolution (determinism of the engine).

NOTE: The full live==replay assertion (R7 proper) runs once the live ontick
service exists -- it will replay the same captured ticks through the live
service's warmup and assert live-state == backtest-state at bar N within
< 0.001 pips. Until that service exists, this gate proves the ENGINE half:
the engine is a pure deterministic function of the tick stream, so any future
live divergence is isolable to the live wiring, not the engine.

Usage:
    python3 tick_r7.py [path_to_jsonl] [PAIR]
Default sample lives at /tmp/oanda_ticks_sample.jsonl (copied from the
fx-tick-mom container's /data/ticks volume).
"""
import sys
import json
import hashlib

from ontick_engine import RollingWindow

DEFAULT_PATH = "/tmp/oanda_ticks_sample.jsonl"
WINDOW_SEC = 1800  # 30m


def load_ticks(path, pair):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("p") != pair:
                continue
            mid = (float(d["b"]) + float(d["a"])) / 2.0
            out.append((int(d["ms"]), mid))
    return out


def replay_digest(ticks, window_sec):
    """Run ticks through a fresh engine; return a hash of the full state trace."""
    win = RollingWindow(window_sec)
    h = hashlib.sha256()
    for t_ms, mid in ticks:
        st = win.update(t_ms, mid)
        # serialize state deterministically (repr of floats is exact round-trip)
        rec = f"{st.t_ms}|{st.n}|{st.mean!r}|{st.std!r}|{st.high!r}|{st.low!r}|{st.open!r}|{st.close!r}\n"
        h.update(rec.encode())
    return h.hexdigest()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    pair = sys.argv[2] if len(sys.argv) > 2 else None

    # auto-pick the busiest pair if none given
    if pair is None:
        counts = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line).get("p")
                counts[p] = counts.get(p, 0) + 1
        pair = max(counts, key=counts.get)

    ticks = load_ticks(path, pair)
    assert len(ticks) > 100, f"need ticks, got {len(ticks)} for {pair}"
    print(f"R7 determinism gate: {pair}, {len(ticks)} ticks, window={WINDOW_SEC}s")

    d1 = replay_digest(ticks, WINDOW_SEC)
    d2 = replay_digest(ticks, WINDOW_SEC)

    print(f"  run1 sha256: {d1}")
    print(f"  run2 sha256: {d2}")
    if d1 == d2:
        print("  PASS -- byte-identical engine state across two replays (deterministic).")
        return 0
    print("  FAIL -- non-deterministic engine state. DO NOT DEPLOY.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
