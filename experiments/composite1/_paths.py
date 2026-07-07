"""_paths.py — Composite 1: resolves the sibling cot_positioning code directory for
VERBATIM reuse of its COT parser / z-score / release-lag / carry / D1-loader / bootstrap
code (cot_signal.py, release_lag.py, d1_data.py, carry_model.py, carry_splice.py,
bootstrap.py) via direct IMPORT, not copy-paste — the strongest possible form of "same
code" (PREREGISTRATION.md gate 2: "same code, parity +-0.1p/wk"), with zero drift risk
between the two experiments.

Local repo layout: research/experiments/composite1/ and research/experiments/cot_positioning/
are siblings -> default resolves one level up, into ../cot_positioning.

Hetzner deployment layout (per task brief): /root/work/code_comp/ (this directory, flat) and
/root/work/code_cot/ (rsynced flat copy of cot_positioning/) are siblings under /root/work/ ->
set COT_CODE_DIR=/root/work/code_cot explicitly (same override pattern cot_positioning's own
d1_data.py already documents for COT_D1_DATA_DIR).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT = os.path.join(HERE, "..", "cot_positioning")
COT_CODE_DIR = os.path.abspath(os.environ.get("COT_CODE_DIR", _DEFAULT))

if not os.path.isdir(COT_CODE_DIR):
    raise RuntimeError(
        f"COT_CODE_DIR not found: {COT_CODE_DIR!r} — set the COT_CODE_DIR env var to the "
        f"rsynced cot_positioning code directory (e.g. /root/work/code_cot on Hetzner)."
    )

if COT_CODE_DIR not in sys.path:
    # APPEND, never insert(0, ...): composite1 ships its OWN same-named modules for a few
    # files (is_data.py, compute_gates.py, run_is_battery.py — each intentionally
    # DIFFERENT from cot_positioning's own module of that name, e.g. composite1's is_data.py
    # is its own IS/OOS boundary, not cot_positioning's). The calling script's own directory
    # is already sys.path[0] (Python's default), so appending here guarantees composite1's
    # own modules always shadow (take precedence over) any same-named cot_positioning
    # module, while modules that exist ONLY on the cot_positioning side (cot_signal,
    # release_lag, carry_model, carry_splice, d1_data, bootstrap) still resolve correctly by
    # falling through. An earlier version of this file used insert(0, ...), which silently
    # shadowed composite1's own is_data.py with cot_positioning's — caught immediately by
    # running run_is_battery.py on Hetzner (AttributeError: no restrict_cot_to_is).
    sys.path.append(COT_CODE_DIR)
