"""Purged walk-forward, baselines, pinball loss, and block-bootstrap CI."""
import numpy as np
import pandas as pd


def pinball_loss(y, yhat, q=0.5):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    e = y - yhat
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def purged_folds(ts, n_folds=5, embargo_min=120):
    """Expanding-window walk-forward with an embargo. Train = [0, cut) minus the last
    `embargo` of time; test = [cut, next_cut) trimmed to start >= train_end + embargo."""
    ts_ns = pd.DatetimeIndex(ts).asi8
    n = len(ts_ns)
    emb = np.int64(embargo_min) * 60 * 1_000_000_000
    bounds = np.linspace(0, n, n_folds + 2, dtype=int)
    folds = []
    for i in range(1, n_folds + 1):
        cut = bounds[i]; nxt = bounds[i + 1]
        if cut <= 1 or nxt <= cut:
            continue
        tr_end_time = ts_ns[cut - 1]
        train = np.arange(0, cut)
        train = train[ts_ns[train] <= tr_end_time - emb]
        test = np.arange(cut, nxt)
        test = test[ts_ns[test] >= tr_end_time + emb]
        if len(train) and len(test):
            folds.append((train, test))
    return folds


def climatology_baseline(train_df, test_df, target):
    g = train_df.groupby("how")[target].mean()
    glob = float(train_df[target].mean())
    return test_df["how"].map(g).fillna(glob).to_numpy()


def flat_baseline(train_df, test_df, target):
    return np.full(len(test_df), float(train_df[target].mean()))


def block_bootstrap_ci(diff, block=6, n=2000, alpha=0.05, seed=0):
    diff = np.asarray(diff, float)
    m = len(diff)
    if m == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(m / block))
    means = np.empty(n)
    starts_max = max(1, m - block + 1)
    for b in range(n):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:m]
        idx = np.clip(idx, 0, m - 1)
        means[b] = diff[idx].mean()
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


import os
import sys
import lightgbm as lgb

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.feature_statistics import newey_west_tstat  # noqa: E402

_LGB_PARAMS = dict(objective="quantile", n_estimators=300, learning_rate=0.05,
                   num_leaves=31, min_child_samples=200, subsample=0.8,
                   colsample_bytree=0.8, verbosity=-1)


def _pinball_pointwise(y, yhat, q):
    """Per-point pinball loss (array), for the bootstrap/NW diff."""
    e = y - yhat
    return np.maximum(q * e, (q - 1.0) * e)


def run_wf(df, features, target, n_folds=5, embargo_min=120, q=0.5):
    base = df[df["valid"]].reset_index(drop=True)
    folds = purged_folds(base["ts"], n_folds=n_folds, embargo_min=embargo_min)
    m_losses, c_losses, f_losses, diffs, imp = [], [], [], [], {}
    n_test = 0
    for tr_idx, te_idx in folds:
        tr = base.iloc[tr_idx]
        te = base.iloc[te_idx]
        tr_fit = tr[tr["is_minute_anchor"]]                    # thinned training
        te_eval = te[te["is_hour_anchor"]]                     # non-overlapping eval
        if len(tr_fit) < 500 or len(te_eval) < 10:
            continue
        model = lgb.LGBMRegressor(alpha=q, **_LGB_PARAMS)
        model.fit(tr_fit[features], tr_fit[target])
        pred = model.predict(te_eval[features])
        y = te_eval[target].to_numpy()
        clim = climatology_baseline(tr_fit, te_eval, target)
        flat = flat_baseline(tr_fit, te_eval, target)
        m_losses.append(pinball_loss(y, pred, q))
        c_losses.append(pinball_loss(y, clim, q))
        f_losses.append(pinball_loss(y, flat, q))
        # per-point pinball difference (clim - model): positive = model better
        diffs.append(_pinball_pointwise(y, clim, q) - _pinball_pointwise(y, pred, q))
        for name, gain in zip(features, model.booster_.feature_importance("gain")):
            imp[name] = imp.get(name, 0.0) + float(gain)
        n_test += len(te_eval)
    diff = np.concatenate(diffs) if diffs else np.array([])
    return {
        "target": target,
        "model_pinball": float(np.mean(m_losses)) if m_losses else float("nan"),
        "clim_pinball": float(np.mean(c_losses)) if c_losses else float("nan"),
        "flat_pinball": float(np.mean(f_losses)) if f_losses else float("nan"),
        "ci_vs_clim": block_bootstrap_ci(diff, block=6, n=2000, alpha=0.05, seed=0),
        "nw_tstat": newey_west_tstat(diff, lag=24) if len(diff) else 0.0,
        "n_test": n_test,
        "importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])),
    }
