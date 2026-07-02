#!/usr/bin/env python3
"""Version A: a CONTAINMENT envelope. Refit the quantile forecaster at q=0.9 for the high and
low reach, so the predicted band contains ~90% of the actual next-hour excursions (reads as a
real 'envelope'). Median (q=0.5) forecast kept only for the calibration panel.
Top: 90% range envelope (blue) vs actual reach (dots) vs static climatology 90th-pct (red dashed).
Bottom: median-forecast calibration (predicted vs actual). Units: ATR (average true range) multiples.
"""
import os, sys, gc
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = "/path/to/projects/fx-core"; sys.path.insert(0, REPO)
from research.experiments.nexthour_hl.build_dataset import FEATURES, OUT
from research.experiments.nexthour_hl.evaluate import purged_folds, _LGB_PARAMS
import lightgbm as lgb

cols = list(dict.fromkeys(['ts'] + FEATURES + ['how', 'up', 'dn', 'valid',
                                               'is_minute_anchor', 'is_hour_anchor']))
df = pd.read_parquet(OUT, columns=cols)
base = df[df['valid']].reset_index(drop=True); del df; gc.collect()
folds = purged_folds(base['ts'], n_folds=5, embargo_min=120)
print(f"{len(base)} valid rows, {len(folds)} folds", flush=True)

recs = []
for tr_idx, te_idx in folds:
    tr = base.iloc[tr_idx]; te = base.iloc[te_idx]
    tr_fit = tr[tr['is_minute_anchor']]; te_eval = te[te['is_hour_anchor']]
    if len(tr_fit) < 500 or len(te_eval) < 10:
        continue
    out = {'ts': te_eval['ts'].to_numpy()}
    for tgt in ['up', 'dn']:
        for q, lab in [(0.5, 'q50'), (0.9, 'q90')]:
            m = lgb.LGBMRegressor(alpha=q, **_LGB_PARAMS)
            m.fit(tr_fit[FEATURES], tr_fit[tgt])
            out[f'{tgt}_{lab}'] = m.predict(te_eval[FEATURES])
        out[f'{tgt}_act'] = te_eval[tgt].to_numpy()
        clim90 = tr_fit.groupby('how')[tgt].quantile(0.9)
        out[f'{tgt}_clim90'] = te_eval['how'].map(clim90).fillna(tr_fit[tgt].quantile(0.9)).to_numpy()
    recs.append(pd.DataFrame(out))
    print(f"  fold eval={len(te_eval)}", flush=True)

R = pd.concat(recs).sort_values('ts').reset_index(drop=True)
cov_up = float((R['up_act'] <= R['up_q90']).mean()); cov_dn = float((R['dn_act'] <= R['dn_q90']).mean())
print(f"OOF hours={len(R)}  q90 containment: high {100*cov_up:.0f}%  low {100*cov_dn:.0f}%", flush=True)

fig = plt.figure(figsize=(13, 9))
ax1 = fig.add_subplot(2, 1, 1)
w0 = len(R) // 2; W = 180; s = slice(w0, w0 + W); x = np.arange(W)
up9 = R['up_q90'].to_numpy()[s]; dn9 = R['dn_q90'].to_numpy()[s]
ua = R['up_act'].to_numpy()[s]; da = R['dn_act'].to_numpy()[s]
uc = R['up_clim90'].to_numpy()[s]; dc = R['dn_clim90'].to_numpy()[s]
ax1.fill_between(x, -dn9, up9, color="#1f77b4", alpha=0.18, label="predicted 90% range envelope")
ax1.plot(x, up9, color="#1f77b4", lw=1.4); ax1.plot(x, -dn9, color="#1f77b4", lw=1.4)
ax1.plot(x, uc, color="#d62728", lw=0.9, ls="--", alpha=0.55, label="climatology 90th-pct (static, hour-of-week)")
ax1.plot(x, -dc, color="#d62728", lw=0.9, ls="--", alpha=0.55)
inside = (ua <= up9) & (da <= dn9)
ax1.plot(x[inside], ua[inside], "k.", ms=4, label="actual reach (inside band)")
ax1.plot(x[inside], -da[inside], "k.", ms=4)
ax1.plot(x[~inside], ua[~inside], ".", color="#ff7f0e", ms=6, label="actual reach (band exceeded)")
ax1.plot(x[~inside], -da[~inside], ".", color="#ff7f0e", ms=6)
ax1.axhline(0, color="gray", lw=0.6)
ax1.set_title(f"Next-hour high/low: predicted 90% range envelope vs actual reach "
              f"(EUR/USD, out-of-fold) — band contains {100*(cov_up+cov_dn)/2:.0f}% of reaches")
ax1.set_ylabel("excursion (ATR multiples)"); ax1.set_xlabel("hour (sample window)")
ax1.legend(loc="upper right", fontsize=8, ncol=2); ax1.grid(alpha=0.25)

for j, (tgt, name) in enumerate([('up', 'HIGH'), ('dn', 'LOW')]):
    ax = fig.add_subplot(2, 2, 3 + j)
    p = R[f'{tgt}_q50'].to_numpy(); a = R[f'{tgt}_act'].to_numpy()
    lim = float(np.nanpercentile(np.concatenate([p, a]), 99))
    ax.plot([0, lim], [0, lim], color="gray", lw=1, ls="--", label="perfect (y=x)")
    ax.scatter(p, a, s=4, alpha=0.10, color="#1f77b4")
    bins = np.linspace(0, lim, 12); bi = np.digitize(p, bins)
    bx = [p[bi == k].mean() for k in range(1, len(bins))]; by = [a[bi == k].mean() for k in range(1, len(bins))]
    ax.plot(bx, by, "o-", color="#ff7f0e", ms=4, lw=1.4, label="binned mean actual")
    r = np.corrcoef(p, a)[0, 1]
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f"{name} reach — median forecast calibration (corr={r:.2f})")
    ax.set_xlabel("predicted (ATR)"); ax.set_ylabel("actual (ATR)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.25)

fig.tight_layout()
out_png = os.path.join(REPO, "data/nexthour_hl/envelope_pred_vs_actual_A.png")
fig.savefig(out_png, dpi=140, bbox_inches="tight")
print(f"WROTE {out_png}")
