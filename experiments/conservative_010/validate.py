"""Conservative backtest validation: drawdown, IS/OOS, walk-forward, Monte Carlo."""
import numpy as np

def equity_drawdown(net_pips):
    eq = np.asarray(net_pips, float)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    under = dd < -1e-9
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return {"max_dd": float(dd.min()) if len(dd) else 0.0,
            "longest_underwater": int(longest),
            "frac_over_200": float((dd < -200).mean()) if len(dd) else 0.0}

def split_is_oos(entry_bars, net, is_end):
    eb = np.asarray(entry_bars); net = np.asarray(net, float)
    m = eb < is_end
    o = ~m
    return {"is_net": float(net[m].sum()), "oos_net": float(net[o].sum()),
            "is_n": int(m.sum()), "oos_n": int(o.sum()),
            "oos_wr": float((net[o] > 0).mean()*100) if o.any() else 0.0}

def walk_forward(entry_bars, net, n_total, n_folds=6):
    eb = np.asarray(entry_bars); net = np.asarray(net, float)
    edges = np.linspace(0, n_total, n_folds+1, dtype=int)
    out = []
    for k in range(n_folds):
        m = (eb >= edges[k]) & (eb < edges[k+1])
        out.append({"fold": k+1, "n": int(m.sum()),
                    "net": float(net[m].sum()),
                    "wr": float((net[m] > 0).mean()*100) if m.any() else 0.0})
    return out

def monte_carlo(net, n=300, seed=0):
    """p_net: bootstrap-resample WITH REPLACEMENT (so the sum varies) -> fraction of
    resamples with net <= 0; a one-sided prob the edge is non-positive (lower = more
    significant positive edge). p_maxdd: PERMUTATION of trade order (path-dependent,
    sum preserved) -> fraction of shuffles whose max drawdown is <= observed (how
    ordinary the observed drawdown is among reorderings)."""
    net = np.asarray(net, float)
    rng = np.random.default_rng(seed)
    obs_dd = equity_drawdown(np.cumsum(net))["max_dd"]
    m = len(net)
    le0 = wo_dd = 0
    for _ in range(n):
        b = rng.choice(net, size=m, replace=True)        # bootstrap: sum varies
        if b.sum() <= 0: le0 += 1
        s = rng.permutation(net)                          # permutation: drawdown null
        if equity_drawdown(np.cumsum(s))["max_dd"] <= obs_dd: wo_dd += 1
    return {"p_net": le0/n, "p_maxdd": wo_dd/n}
