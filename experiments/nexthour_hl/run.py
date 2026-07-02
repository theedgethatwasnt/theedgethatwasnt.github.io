# research/experiments/nexthour_hl/run.py
"""Run the next-hour high/low forecaster end-to-end and print the verdict.

From repo root:  python3 research/experiments/nexthour_hl/run.py
Builds (or loads) the supervised table, runs purged-WF LightGBM for `up` and `dn`,
compares to ATR x hour-of-week climatology, and writes data/nexthour_hl/report.{json,md}.
"""
import json
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from research.experiments.nexthour_hl.build_dataset import build, FEATURES, SRC_MTF, SRC_S5, OUT  # noqa: E402
from research.experiments.nexthour_hl.evaluate import run_wf  # noqa: E402

REPORT_JSON = os.path.join(REPO, "data/nexthour_hl/report.json")
REPORT_MD = os.path.join(REPO, "data/nexthour_hl/report.md")


def main():
    if os.path.exists(OUT):
        df = pd.read_parquet(OUT)
        print(f"loaded supervised table {df.shape} from {OUT}")
    else:
        df = build(SRC_MTF, SRC_S5, 0.0001)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        df.to_parquet(OUT, index=False)
        print(f"built supervised table {df.shape}")

    results = {}
    for target in ["up", "dn"]:
        res = run_wf(df, FEATURES, target, n_folds=5, embargo_min=120, q=0.5)
        results[target] = res
        lo, hi = res["ci_vs_clim"]
        better = lo > 0
        print(f"\n=== {target} ===")
        print(f"  model pinball  : {res['model_pinball']:.4f}")
        print(f"  clim  pinball  : {res['clim_pinball']:.4f}")
        print(f"  flat  pinball  : {res['flat_pinball']:.4f}")
        print(f"  improvement CI vs clim (95%): [{lo:.4f}, {hi:.4f}]  "
              f"{'BEATS clim' if better else 'no edge'}   NW t={res['nw_tstat']:.2f}  n_test={res['n_test']}")
        print(f"  top features   : {list(res['importance'])[:6]}")

    verdict = {t: (results[t]['ci_vs_clim'][0] > 0) for t in results}
    success = any(verdict.values())
    print(f"\nVERDICT: {'SUCCESS' if success else 'no edge over ATR x hour-of-week'}  {verdict}")

    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w") as f:
        json.dump({"results": results, "verdict": verdict, "success": success}, f, indent=2, default=float)
    with open(REPORT_MD, "w") as f:
        f.write("# Next-hour high/low forecaster — results\n\n")
        for t in results:
            r = results[t]; lo, hi = r["ci_vs_clim"]
            f.write(f"## {t}\n- model pinball: {r['model_pinball']:.4f}\n"
                    f"- clim pinball: {r['clim_pinball']:.4f}\n- flat pinball: {r['flat_pinball']:.4f}\n"
                    f"- improvement CI vs clim: [{lo:.4f}, {hi:.4f}]  (NW t={r['nw_tstat']:.2f}, n_test={r['n_test']})\n"
                    f"- top features: {list(r['importance'])[:8]}\n\n")
        f.write(f"**VERDICT:** {'SUCCESS' if success else 'no edge over ATR x hour-of-week'}  {verdict}\n")
    print("wrote", REPORT_JSON, "and", REPORT_MD)


if __name__ == "__main__":
    main()
