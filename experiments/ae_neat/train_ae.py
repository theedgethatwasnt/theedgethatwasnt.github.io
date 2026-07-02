#!/usr/bin/env python3
"""
Experiment A: Train Autoencoder on 12 market features.

Features (from curator_identical + asi_mc_indicators parquets + computed):
  mc_d, mc_dd, er_norm,          ← curator / asi_mc parquets
  range_pos, bb_width,           ← new from filter study
  rsi_extreme, zscore, bb_pos,   ← top IC features (approximated from close-only)
  stoch_k, williams_r,           ← momentum oscillators (close-only approximation)
  breakout_ch, ha_direction      ← channel breakout + HA direction

Architecture: 12 → 8 → latent_dim → 8 → 12
Loss:         MSE reconstruction + λ * L1 sparsity on latent
Optimizer:    Adam lr=1e-3, 100 epochs, batch=256
"""

import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT     = Path(__file__).parent.parent.parent.parent
DATA_CUR = ROOT / 'data' / 'curator_identical'
DATA_ASI = ROOT / 'data' / 'asi_mc_indicators'
OUT_DIR  = Path(__file__).parent / 'results'
OUT_DIR.mkdir(exist_ok=True)

PAIRS = [
    'AUD_JPY', 'AUD_USD', 'CAD_JPY', 'CHF_JPY',
    'EUR_GBP', 'EUR_JPY', 'EUR_USD', 'GBP_JPY',
    'GBP_USD', 'NZD_JPY', 'NZD_USD', 'USD_JPY',
]
IS_FRAC  = 0.70
L1_LAMBDA = 1e-4
EPOCHS   = 100
BATCH    = 256
LR       = 1e-3
LATENT_DIMS = [2, 3, 4, 6]
FEATURE_NAMES = [
    'mc_d', 'mc_dd', 'er_norm',
    'range_pos', 'bb_width',
    'rsi_extreme', 'zscore', 'bb_pos',
    'stoch_k', 'williams_r',
    'breakout_ch', 'ha_direction',
]
N_FEATURES = len(FEATURE_NAMES)


# ─── Feature computation ──────────────────────────────────────────────────────

def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # Wilder smoothing via EMA
    alpha = 1.0 / period
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(close)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return (rsi / 50.0) - 1.0   # rescale [0,100] → [-1, +1]


def compute_features(df: pd.DataFrame) -> np.ndarray:
    """Compute all 12 features from a single-pair DataFrame."""
    close = df['mid_close'].values.astype(np.float64)
    mc_d  = df['mc_d_curator'].values.astype(np.float64)
    mc_dd = df['mc_dd_curator'].values.astype(np.float64)

    n = len(close)

    # er_norm: load from asi_mc parquet if available, else approx
    if 'er_norm' in df.columns:
        er_norm = df['er_norm'].values.astype(np.float64)
    else:
        # Kaufman ER(14): |change| / sum(|step|)
        change = np.abs(np.diff(close, prepend=close[0]))
        er = np.zeros(n)
        for i in range(14, n):
            path = np.sum(np.abs(np.diff(close[i-14:i+1])))
            er[i] = abs(close[i] - close[i-14]) / (path + 1e-12)
        er_norm = (2.0 / math.pi) * np.arctan(er * 5)

    # Rolling stats (period=20)
    W20 = 20
    W30 = 30
    W14 = 14

    def rolling_mean(x, w):
        out = np.full(n, np.nan)
        cs = np.cumsum(np.concatenate([[0], x]))
        for i in range(w - 1, n):
            out[i] = (cs[i+1] - cs[i-w+1]) / w
        return out

    def rolling_std(x, w):
        m = rolling_mean(x, w)
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = np.std(x[i-w+1:i+1], ddof=0)
        return out

    def rolling_max(x, w):
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = np.max(x[i-w+1:i+1])
        return out

    def rolling_min(x, w):
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = np.min(x[i-w+1:i+1])
        return out

    mean20 = rolling_mean(close, W20)
    std20  = rolling_std(close, W20)
    mean30 = rolling_mean(close, W30)
    max30  = rolling_max(close, W30)
    min30  = rolling_min(close, W30)
    max20  = rolling_max(close, W20)
    min14  = rolling_min(close, W14)
    max14  = rolling_max(close, W14)

    # range_pos: price position in 30-bar range → [0, 1]
    denom30 = max30 - min30
    range_pos = np.where(denom30 > 1e-12, (close - min30) / denom30, 0.5)

    # bb_width: bandwidth = 2*std / mean (normalised)
    bb_width = np.where(mean20 > 1e-12, 2.0 * std20 / mean20, 0.0)

    # rsi_extreme: RSI(14) → [-1, +1]
    rsi_extreme = compute_rsi(close, 14)

    # zscore: (close - mean20) / std20
    zscore = np.where(std20 > 1e-12, (close - mean20) / std20, 0.0)

    # bb_pos: (close - lower) / (upper - lower)  → [0,1] then → [-1,+1]
    upper = mean20 + 2 * std20
    lower = mean20 - 2 * std20
    denom_bb = upper - lower
    bb_pos = np.where(denom_bb > 1e-12, (close - lower) / denom_bb, 0.5)
    bb_pos = bb_pos * 2.0 - 1.0  # → [-1,+1]

    # stoch_k: (close - min14) / (max14 - min14) → [-1,+1]
    denom14 = max14 - min14
    stoch_k = np.where(denom14 > 1e-12, (close - min14) / denom14, 0.5)
    stoch_k = stoch_k * 2.0 - 1.0

    # williams_r: complement of stoch_k
    williams_r = -stoch_k

    # breakout_ch: (close - max20) / (std20 * 4.5)  (negative = below high)
    breakout_ch = np.where(std20 > 1e-12,
                           (close - max20) / (std20 * 4.5), 0.0)
    breakout_ch = np.clip(breakout_ch, -1, 1)

    # ha_direction: sign of momentum
    ha_direction = np.sign(close - np.roll(close, 1))
    ha_direction[0] = 0.0

    mat = np.column_stack([
        mc_d, mc_dd, er_norm,
        range_pos, bb_width,
        rsi_extreme, zscore, bb_pos,
        stoch_k, williams_r,
        breakout_ch, ha_direction,
    ])
    return mat


def load_all_pairs() -> tuple[np.ndarray, np.ndarray]:
    """Load & compute features for all pairs. Returns (IS_data, OOS_data)."""
    all_is, all_oos = [], []
    for pair in PAIRS:
        cur_path = DATA_CUR / f'{pair}_curator.parquet'
        asi_path = DATA_ASI / f'{pair}_asi_mc.parquet'
        if not cur_path.exists():
            print(f'  SKIP {pair}: parquet not found')
            continue

        df = pd.read_parquet(cur_path)

        # Merge er_norm from asi_mc if available
        if asi_path.exists():
            df_asi = pd.read_parquet(asi_path)[['timestamp', 'er_norm']]
            df = df.merge(df_asi, on='timestamp', how='left')

        mat = compute_features(df)
        # Drop rows with NaN (warmup period)
        valid = ~np.any(np.isnan(mat), axis=1)
        mat = mat[valid]

        n = len(mat)
        split = int(n * IS_FRAC)
        all_is.append(mat[:split])
        all_oos.append(mat[split:])
        print(f'  {pair}: {n} bars ({split} IS, {n-split} OOS)')

    return np.concatenate(all_is), np.concatenate(all_oos)


# ─── Autoencoder ──────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.Tanh(),
            nn.Linear(8, latent_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.Tanh(),
            nn.Linear(8, n_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-feature R² scores."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def train_ae(X_is: np.ndarray, X_oos: np.ndarray,
             latent_dim: int) -> tuple[Autoencoder, dict]:
    """Train one AE and return (model, metrics)."""
    # Normalise (fit on IS only)
    mu = X_is.mean(axis=0)
    sigma = X_is.std(axis=0) + 1e-8
    X_is_n  = (X_is  - mu) / sigma
    X_oos_n = (X_oos - mu) / sigma

    device = torch.device('cpu')
    t_is  = torch.tensor(X_is_n,  dtype=torch.float32, device=device)
    t_oos = torch.tensor(X_oos_n, dtype=torch.float32, device=device)

    model = Autoencoder(N_FEATURES, latent_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    ds    = TensorDataset(t_is)
    dl    = DataLoader(ds, batch_size=BATCH, shuffle=True)

    losses = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for (xb,) in dl:
            opt.zero_grad()
            recon, z = model(xb)
            mse  = nn.functional.mse_loss(recon, xb)
            l1   = L1_LAMBDA * z.abs().mean()
            loss = mse + l1
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        losses.append(ep_loss / len(t_is))

    # Evaluate
    model.eval()
    with torch.no_grad():
        recon_is,  z_is  = model(t_is)
        recon_oos, z_oos = model(t_oos)

    r2_is  = r2_score(X_is_n,  recon_is.numpy())
    r2_oos = r2_score(X_oos_n, recon_oos.numpy())

    # Latent inter-correlation
    z_np = z_oos.numpy()
    if latent_dim > 1:
        corr = np.corrcoef(z_np.T)
        off_diag = corr[np.triu_indices(latent_dim, k=1)]
        max_corr = np.max(np.abs(off_diag))
    else:
        max_corr = 0.0

    metrics = {
        'latent_dim':   latent_dim,
        'r2_is_mean':   float(r2_is.mean()),
        'r2_oos_mean':  float(r2_oos.mean()),
        'r2_oos_min':   float(r2_oos.min()),
        'r2_oos_per_feature': r2_oos.tolist(),
        'max_latent_corr': float(max_corr),
        'train_loss':   losses,
        'mu': mu.tolist(),
        'sigma': sigma.tolist(),
    }
    elapsed = time.time() - t0
    print(f'  latent={latent_dim}: R²_IS={r2_is.mean():.3f} R²_OOS={r2_oos.mean():.3f} '
          f'min_OOS={r2_oos.min():.3f} max_corr={max_corr:.3f} ({elapsed:.0f}s)')

    return model, metrics


# ─── Results plot ─────────────────────────────────────────────────────────────

def plot_results(all_metrics: list[dict], X_oos: np.ndarray,
                 best_model: Autoencoder, best_mu, best_sigma, best_dim):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_color('#546e7a')

    # 1. R² by latent dim
    ax = axes[0]
    dims  = [m['latent_dim']  for m in all_metrics]
    r2_is = [m['r2_is_mean']  for m in all_metrics]
    r2_oos= [m['r2_oos_mean'] for m in all_metrics]
    ax.plot(dims, r2_is,  'o-', color='#3498db', label='IS R²', lw=2)
    ax.plot(dims, r2_oos, 's-', color='#2ecc71', label='OOS R²', lw=2)
    ax.axhline(0.85, color='#f39c12', ls='--', lw=1.2, label='Target 0.85')
    ax.set_xlabel('Latent dim', color='white'); ax.set_ylabel('Mean R²', color='white')
    ax.set_title('R² vs Latent Dim', color='white', fontweight='bold')
    ax.legend(labelcolor='white', facecolor='#16213e', edgecolor='#546e7a')

    # 2. Per-feature R² for best model
    ax = axes[1]
    best_m = next(m for m in all_metrics if m['latent_dim'] == best_dim)
    r2f = best_m['r2_oos_per_feature']
    colors = ['#2ecc71' if r >= 0.85 else '#e74c3c' if r < 0.5 else '#f39c12'
              for r in r2f]
    bars = ax.barh(FEATURE_NAMES, r2f, color=colors, alpha=0.85)
    ax.axvline(0.85, color='#f39c12', ls='--', lw=1.2)
    ax.set_xlabel('OOS R²', color='white')
    ax.set_title(f'Per-feature R² (latent={best_dim})', color='white', fontweight='bold')
    ax.set_xlim(-0.2, 1.1)
    for bar, val in zip(bars, r2f):
        ax.text(max(val + 0.02, 0.01), bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=8, color='white')

    # 3. Training loss curve for best model
    ax = axes[2]
    ax.plot(best_m['train_loss'], color='#3498db', lw=1.5)
    ax.set_xlabel('Epoch', color='white'); ax.set_ylabel('Loss', color='white')
    ax.set_title(f'Training Loss (latent={best_dim})', color='white', fontweight='bold')

    plt.suptitle(f'Autoencoder Training — 12 features → {best_dim} latent dims\n'
                 f'12 pairs, IS={IS_FRAC*100:.0f}% OOS={100-IS_FRAC*100:.0f}%',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = OUT_DIR / 'ae_reconstruction.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {out}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=== Autoencoder Training — Experiment A ===')
    print(f'Features ({N_FEATURES}): {FEATURE_NAMES}')
    print(f'Latent dims to test: {LATENT_DIMS}')
    print()

    print('Loading data...')
    X_is, X_oos = load_all_pairs()
    print(f'\nIS: {len(X_is):,} bars | OOS: {len(X_oos):,} bars')
    print()

    # Normalise globally for stats (model trains its own normalisation)
    mu  = X_is.mean(axis=0)
    sig = X_is.std(axis=0) + 1e-8

    all_metrics = []
    all_models  = []
    print('Training autoencoders...')
    for ld in LATENT_DIMS:
        model, metrics = train_ae(X_is, X_oos, ld)
        all_metrics.append(metrics)
        all_models.append(model)

    print()
    print('=== Results ===')
    print(f'{"dim":>4}  {"R²_IS":>6}  {"R²_OOS":>7}  {"min_OOS":>8}  {"maxCorr":>8}  Status')
    print('-' * 60)

    best_dim = None
    best_model = None
    best_mu_g = mu
    best_sigma_g = sig

    for m, model in zip(all_metrics, all_models):
        ld   = m['latent_dim']
        ok   = m['r2_oos_mean'] >= 0.85 and m['r2_oos_min'] >= 0.50
        flag = '✓ PASS' if ok else ('? WEAK' if m['r2_oos_min'] >= 0.30 else '✗ FAIL')
        print(f'{ld:>4}  {m["r2_is_mean"]:>6.3f}  {m["r2_oos_mean"]:>7.3f}  '
              f'{m["r2_oos_min"]:>8.3f}  {m["max_latent_corr"]:>8.3f}  {flag}')
        if ok and best_dim is None:
            best_dim   = ld
            best_model = model
            best_mu_g  = np.array(m['mu'])
            best_sigma_g = np.array(m['sigma'])

    print()
    if best_dim is None:
        # Pick best available even if below threshold
        best_idx   = max(range(len(all_metrics)),
                         key=lambda i: all_metrics[i]['r2_oos_mean'])
        best_dim   = all_metrics[best_idx]['latent_dim']
        best_model = all_models[best_idx]
        best_mu_g  = np.array(all_metrics[best_idx]['mu'])
        best_sigma_g = np.array(all_metrics[best_idx]['sigma'])
        print(f'WARNING: No dim passed R²>0.85 threshold.')
        print(f'Best available: latent_dim={best_dim} '
              f'R²={all_metrics[best_idx]["r2_oos_mean"]:.3f}')
        print('Recommendation: Fall back to manual 6-input NEAT (see PLAN.md)')
    else:
        print(f'SELECTED: latent_dim={best_dim} (first passing R²>0.85)')

    # Save encoder
    encoder_path = ROOT / 'models' / f'ae_encoder_12to{best_dim}.pkl'
    save_obj = {
        'model_state': best_model.encoder.state_dict(),
        'latent_dim':  best_dim,
        'n_features':  N_FEATURES,
        'feature_names': FEATURE_NAMES,
        'mu':    best_mu_g.tolist(),
        'sigma': best_sigma_g.tolist(),
        'metrics': [m for m in all_metrics if m['latent_dim'] == best_dim][0],
    }
    with open(encoder_path, 'wb') as f:
        pickle.dump(save_obj, f)
    print(f'Encoder saved: {encoder_path}')

    # Save full results
    results_path = OUT_DIR / 'ae_training_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump({'metrics': all_metrics, 'best_dim': best_dim}, f)

    # Plot
    plot_results(all_metrics, X_oos, best_model, best_mu_g, best_sigma_g, best_dim)

    # Per-feature breakdown for best dim
    best_m = next(m for m in all_metrics if m['latent_dim'] == best_dim)
    print()
    print(f'Per-feature OOS R² (latent={best_dim}):')
    for name, r2 in zip(FEATURE_NAMES, best_m['r2_oos_per_feature']):
        bar = '█' * int(max(0, r2) * 20)
        flag = '✓' if r2 >= 0.85 else ('~' if r2 >= 0.5 else '✗')
        print(f'  {flag} {name:<15} {r2:>6.3f}  {bar}')

    print()
    if best_m['r2_oos_mean'] >= 0.85:
        print('NEXT: Run Experiment B (run_neat_ae.py) with latent inputs + UPnL')
    else:
        print('NEXT: Consider fallback — use top 3 IC features + 3 existing as NEAT inputs')


if __name__ == '__main__':
    main()
