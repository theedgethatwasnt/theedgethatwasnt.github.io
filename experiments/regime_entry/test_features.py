import numpy as np
from features import compute_regime_features, FEATURE_NAMES

PIP = 1e-4

def test_feature_names_order():
    assert FEATURE_NAMES == ["drift", "er", "rv", "peak_slope", "trough_slope",
                             "leg_expansion", "vr4", "n_legs", "vr2", "vr8"]

def test_pure_trend_er_near_one():
    closes = 1.1000 + np.arange(60) * PIP * 0.5          # monotone up, 29.5p net
    f = compute_regime_features(closes, PIP)
    drift, er = f[0], f[1]
    assert abs(drift - 29.5) < 1e-9
    assert er > 0.99                                      # one leg, |net|==path

def test_pure_chop_er_near_zero_and_drift_zero():
    closes = 1.1000 + np.tile([0.0, 5*PIP], 30)[:60]      # A-B-A-B zigzag
    f = compute_regime_features(closes, PIP)
    assert abs(f[0] - 5.0) < 1e-9                          # ends on high: 5p drift
    assert f[1] < 0.05                                     # path ~295p, net 5p

def test_envelope_slopes_uptrend_both_positive():
    t = np.arange(60)
    closes = 1.1000 + (t * 0.5 + 3.0 * np.sin(t * 2*np.pi/10)) * PIP  # rising zigzag
    f = compute_regime_features(closes, PIP)
    assert f[3] > 0 and f[4] > 0                           # peak & trough slopes up

def test_envelope_slopes_megaphone_diverge():
    t = np.arange(60)
    grow = 1.0 + t/15.0
    closes = 1.1000 + grow * np.sin(t * 2*np.pi/10) * 3.0 * PIP  # expanding
    f = compute_regime_features(closes, PIP)
    assert f[3] > 0 and f[4] < 0                           # peaks up, troughs down
    assert f[5] > 0                                        # legs growing

def test_vr_random_walk_near_one():
    rng = np.random.default_rng(7)
    vrs = []
    for _ in range(500):
        closes = 1.1 + np.cumsum(rng.normal(0, PIP, 60))
        vrs.append(compute_regime_features(closes, PIP)[6])
    assert abs(np.nanmean(vrs) - 1.0) < 0.15               # VR(4) ~ 1 on RW

def test_vr_trending_gt_meanreverting():
    rng = np.random.default_rng(8)
    def ar1(phi):
        e = rng.normal(0, PIP, 60); r = np.zeros(60)
        for i in range(1, 60): r[i] = phi*r[i-1] + e[i]
        return 1.1 + np.cumsum(r)
    vr_mom = np.nanmean([compute_regime_features(ar1(+0.6), PIP)[6] for _ in range(300)])
    vr_rev = np.nanmean([compute_regime_features(ar1(-0.6), PIP)[6] for _ in range(300)])
    assert vr_mom > 1.2 and vr_rev < 0.8

def test_flat_window_returns_nans_not_crash():
    closes = np.full(60, 1.1000)
    f = compute_regime_features(closes, PIP)
    assert f[0] == 0.0 and np.isnan(f[1]) and np.isnan(f[6])
