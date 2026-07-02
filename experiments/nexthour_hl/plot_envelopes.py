#!/usr/bin/env python3
"""Predicted vs actual next-hour high/low ENVELOPE, from out-of-fold purged-WF predictions.
Mirrors evaluate.run_wf exactly (same folds, thinned train, hour-anchor eval, quantile LGBM),
but collects the predictions to visualize. Two panels:
  (1) time-series envelope over a sample window: predicted [-dn, +up] band vs actual reach,
      with the ATR x hour-of-week climatology envelope for comparison.
  (2) calibration scatter (predicted vs actual) for up and dn, with the 45-degree line.
Units: ATR (average true range) multiples; 0 = start-of-hour price.
"""
import os, sys, gc
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = "/path/to/projects/fx-core"; sys.path.insert(0, REPO)
from research.experiments.nexthour_hl.build_dataset import FEATURES, OUT
from research.experiments.nexthour_hl.evaluate import purged_folds, climatology_baseline, _LGB_PARAMS
import lightgbm as lgb

cols = list(dict.fromkeys(['ts'] + FEATURES + ['how', 'up', 'dn', 'valid',
                                               'is_minute_anchor', 'is_hour_anchor']))
df = pd.read_parquet(OUT, columns=cols)
base = df[df['valid']].reset_index(drop=True); del df; gc.collect()
folds = purged_folds(base['ts'], n_folds=5, embargo_min=120)
print(f"loaded {len(base)} valid rows, {len(folds)} folds", flush=True)

recs = []   # aligned per hour-anchor eval row
for tr_idx, te_idx in folds:
    tr = base.iloc[tr_idx]; te = base.iloc[te_idx]
    tr_fit = tr[tr['is_minute_anchor']]; te_eval = te[te['is_hour_anchor']]
    if len(tr_fit) < 500 or len(te_eval) < 10:
        continue
    out = {'ts': te_eval['ts'].to_numpy()}
    for tgt in ['up', 'dn']:
        m = lgb.LGBMRegressor(alpha=0.5, **_LGB_PARAMS)
        m.fit(tr_fit[FEATURES], tr_fit[tgt])
        out[f'{tgt}_pred'] = m.predict(te_eval[FEATURES])
        out[f'{tgt}_act'] = te_eval[tgt].to_numpy()
        out[f'{tgt}_clim'] = climatology_baseline(tr_fit, te_eval, tgt)
    recs.append(pd.DataFrame(out))
    print(f"  fold: train={len(tr_fit)} eval={len(te_eval)}", flush=True)

R = pd.concat(recs).sort_values('ts').reset_index(drop=True)
print(f"OOF hour-anchor predictions: {len(R)}", flush=True)

fig = plt.figure(figsize=(13, 9))
# --- Panel 1: envelope time-series (a clean sample window) ---
ax1 = fig.add_subplot(2, 1, 1)
w0 = len(R) // 2; W = 180; s = slice(w0, w0 + W); x = np.arange(W)
up_p = R['up_pred'].to_numpy()[s]; dn_p = R['dn_pred'].to_numpy()[s]
up_a = R['up_act'].to_numpy()[s]; dn_a = R['dn_act'].to_numpy()[s]
up_c = R['up_clim'].to_numpy()[s]; dn_c = R['dn_clim'].to_numpy()[s]
ax1.fill_between(x, -dn_p, up_p, color="#1f77b4", alpha=0.15, label="predicted range envelope")
ax1.plot(x, up_p, color="#1f77b4", lw=1.3, label="predicted high / low")
ax1.plot(x, -dn_p, color="#1f77b4", lw=1.3)
ax1.plot(x, up_c, color="#d62728", lw=0.9, ls="--", alpha=0.7, label="climatology (ATR x hour-of-week)")
ax1.plot(x, -dn_c, color="#d62728", lw=0.9, ls="--", alpha=0.7)
ax1.plot(x, up_a, "k.", ms=4, label="actual high / low reached")
ax1.plot(x, -dn_a, "k.", ms=4)
ax1.axhline(0, color="gray", lw=0.6)
ax1.set_title("Next-hour high/low: predicted envelope vs actual reach (EUR/USD, out-of-fold, "
              f"{W} consecutive hours)")
ax1.set_ylabel("excursion (ATR multiples)"); ax1.set_xlabel("hour (sample window)")
ax1.legend(loc="upper right", fontsize=8, ncol=2); ax1.grid(alpha=0.25)

# --- Panel 2: calibration scatter, pred vs actual ---
for j, (tgt, name) in enumerate([('up', 'HIGH excursion'), ('dn', 'LOW excursion')]):
    ax = fig.add_subplot(2, 2, 3 + j)
    p = R[f'{tgt}_pred'].to_numpy(); a = R[f'{tgt}_act'].to_numpy()
    lim = np.nanpercentile(np.concatenate([p, a]), 99)
    ax.plot([0, lim], [0, lim], color="gray", lw=1, ls="--", label="perfect (y=x)")
    ax.scatter(p, a, s=4, alpha=0.10, color="#1f77b4")
    # binned mean of actual vs predicted (calibration curve)
    bins = np.linspace(0, lim, 12); bi = np.digitize(p, bins)
    bx = [p[bi == k].mean() for k in range(1, len(bins))]
    by = [a[bi == k].mean() for k in range(1, len(bins))]
    ax.plot(bx, by, "o-", color="#ff7f0e", ms=4, lw=1.4, label="binned mean actual")
    r = np.corrcoef(p, a)[0, 1]
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f"{name}: predicted vs actual (corr={r:.3f})")
    ax.set_xlabel("predicted (ATR)"); ax.set_ylabel("actual (ATR)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.25)

fig.tight_layout()
out_png = os.path.join(REPO, "data/nexthour_hl/envelope_pred_vs_actual.png")
os.makedirs(os.path.dirname(out_png), exist_ok=True)
fig.savefig(out_png, dpi=130, bbox_inches="tight")
# coverage: fraction of hours where actual high/low fell within predicted envelope
cov_up = float((R['up_act'] <= R['up_pred']).mean()); cov_dn = float((R['dn_act'] <= R['dn_pred']).mean())
print(f"WROTE {out_png}")
print(f"envelope coverage: actual high <= pred {100*cov_up:.0f}%, actual low <= pred {100*cov_dn:.0f}% "
      f"(median q=0.5, so ~50% by design)")
print(f"corr up {np.corrcoef(R['up_pred'],R['up_act'])[0,1]:.3f}  dn {np.corrcoef(R['dn_pred'],R['dn_act'])[0,1]:.3f}")
