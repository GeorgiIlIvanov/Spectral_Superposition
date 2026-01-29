#!/usr/bin/env python3
"""
Slope-Eigenvalue Analysis with Spectral Localization Coloring
==============================================================

This script extends the original slope-eigenvalue analysis by computing
spectral localization metrics for each feature cluster, allowing us to
color the slope vs 1/λ plot by degree of localization.

Theory Background:
------------------
The conjecture κ_i ≈ 1/λ_k posits that the slope of D_i vs ||W_i||² for features
in eigenspace k equals the inverse eigenvalue. However, this only works when
features are well-localized to a single eigenspace.

For delocalized features (those spread across multiple eigenspaces), the
relationship breaks down. This analysis colors the slope vs 1/λ scatter plot
by the mean localization of features in each cluster, revealing why the fit
works for some clusters but not others.

Localization Metrics:
- Participation Ratio (PR): 1/Σ_k p_ik² - effective number of eigenspaces
- Max Projection: max_k p_ik - dominance of primary eigenspace
- Inverse Participation Ratio (IPR): Σ_k p_ik² - localization strength

Usage:
------
    python slope_localization_analysis.py [--num-gpus 8] [--checkpoint final]

Author: Claude Code Analysis Pipeline
Date: 2026-01-29
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.cm as cm


# Configuration
INPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
SVD_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/svd_results')
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/svd_analysis_2')

CHECKPOINT_INDICES = {
    'early': 10,
    'mid': 30,
    'final': -1,
}


def compute_localization_metrics_gpu(
    weights: torch.Tensor,  # (m, n)
    U: torch.Tensor,        # (m, m)
    eigenvalues: torch.Tensor,  # (m,)
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Compute feature-level localization metrics on GPU.

    Returns:
        Dict with:
        - projection_weights: (m, n) - p_ik = |u_k^T w_i|² / ||w_i||²
        - participation_ratio: (n,) - 1 / Σ_k p_ik² (higher = more delocalized)
        - max_projection: (n,) - max_k p_ik (higher = more localized)
        - dominant_eigenspace: (n,) - argmax_k p_ik
        - feature_norms_sq: (n,) - ||w_i||²
    """
    m, n = weights.shape

    # Project features onto eigenspaces: z_ki = u_k^T w_i
    projections = U.T @ weights  # (m, n)
    projections_sq = projections ** 2  # (m, n)

    # Feature norms squared
    norms_sq = (weights ** 2).sum(dim=0)  # (n,)
    norms_sq_safe = torch.clamp(norms_sq, min=1e-12)

    # Normalized projection weights: p_ik = |u_k^T w_i|² / ||w_i||²
    p_ik = projections_sq / norms_sq_safe.unsqueeze(0)  # (m, n)

    # Participation ratio: PR_i = 1 / Σ_k p_ik²
    p_sq_sum = (p_ik ** 2).sum(dim=0)  # (n,)
    p_sq_sum_safe = torch.clamp(p_sq_sum, min=1e-12)
    participation_ratio = 1.0 / p_sq_sum_safe  # (n,)

    # Max projection (localization indicator)
    max_projection, dominant_k = p_ik.max(dim=0)  # (n,), (n,)

    # Entropy of projection distribution: H_i = -Σ_k p_ik log(p_ik)
    p_ik_safe = torch.clamp(p_ik, min=1e-12)
    entropy = -(p_ik * torch.log(p_ik_safe)).sum(dim=0)  # (n,)

    return {
        'projection_weights': p_ik,
        'participation_ratio': participation_ratio,
        'max_projection': max_projection,
        'dominant_eigenspace': dominant_k,
        'feature_norms_sq': norms_sq,
        'projection_entropy': entropy,
    }


def compute_cluster_slopes_with_localization(
    feature_norms_sq: np.ndarray,  # (n,)
    fractional_dims: np.ndarray,   # (n,)
    cluster_assignments: np.ndarray,  # (n,)
    eigenvalues: np.ndarray,       # (m,)
    participation_ratio: np.ndarray,  # (n,)
    max_projection: np.ndarray,    # (n,)
    min_cluster_size: int = 10
) -> List[Dict]:
    """
    Compute slope of D_i vs ||W_i||² for each cluster, along with
    mean localization metrics for features in that cluster.

    Returns:
        List of dicts, one per cluster with valid fit.
    """
    unique_clusters = np.unique(cluster_assignments)
    cluster_stats = []

    for k in unique_clusters:
        mask = cluster_assignments == k
        n_features = np.sum(mask)

        if n_features < min_cluster_size:
            continue

        x = feature_norms_sq[mask]  # ||W_i||²
        y = fractional_dims[mask]   # D_i

        # Filter invalid values
        valid = np.isfinite(x) & np.isfinite(y) & (x > 1e-8)
        if np.sum(valid) < min_cluster_size:
            continue

        x_valid = x[valid]
        y_valid = y[valid]

        # Linear regression: D_i = slope * ||W_i||² + intercept
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)

        # Eigenvalue and predicted slope
        lambda_k = eigenvalues[k]
        predicted_slope = 1.0 / lambda_k if lambda_k > 1e-10 else np.nan

        # Localization metrics for this cluster
        pr_values = participation_ratio[mask][valid]
        mp_values = max_projection[mask][valid]

        cluster_stats.append({
            'cluster_k': int(k),
            'n_features': int(n_features),
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'eigenvalue': float(lambda_k),
            'inv_eigenvalue': float(1.0 / lambda_k) if lambda_k > 1e-10 else np.nan,
            'predicted_slope': float(predicted_slope),
            'slope_ratio': float(slope / predicted_slope) if np.isfinite(predicted_slope) and predicted_slope != 0 else np.nan,
            # Localization metrics (cluster means)
            'mean_participation_ratio': float(np.mean(pr_values)),
            'std_participation_ratio': float(np.std(pr_values)),
            'mean_max_projection': float(np.mean(mp_values)),
            'std_max_projection': float(np.std(mp_values)),
            # Inverse participation ratio (IPR = 1/PR, higher = more localized)
            'mean_ipr': float(np.mean(1.0 / (pr_values + 1e-10))),
            'mean_norm_sq': float(np.mean(x_valid)),
            'mean_D': float(np.mean(y_valid)),
        })

    return cluster_stats


def process_single_file(
    input_path: Path,
    svd_path: Path,
    checkpoint_idx: int,
    device: torch.device
) -> Tuple[Dict, List[Dict]]:
    """
    Process a single file and compute slopes with localization metrics.
    """
    # Load original data
    with h5py.File(input_path, 'r') as f:
        weights = f['weights'][checkpoint_idx]  # (m, n)
        fractional_dims = f['fractional_dims'][checkpoint_idx]  # (n,)
        feature_norms = f['feature_norms'][checkpoint_idx]  # (n,)
        checkpoint_step = f['checkpoint_steps'][checkpoint_idx]
        m_hidden = int(f.attrs['m_hidden'])
        sparsity = float(f.attrs['sparsity'])
        seed = int(f.attrs['seed'])

    # Load SVD results
    with h5py.File(svd_path, 'r') as f:
        U = f['U'][checkpoint_idx]  # (m, m)
        eigenvalues = f['eigenvalues'][checkpoint_idx]  # (m,)

    # Move to GPU
    W_gpu = torch.tensor(weights, dtype=torch.float64, device=device)
    U_gpu = torch.tensor(U, dtype=torch.float64, device=device)
    lam_gpu = torch.tensor(eigenvalues, dtype=torch.float64, device=device)

    # Compute localization metrics
    loc_metrics = compute_localization_metrics_gpu(W_gpu, U_gpu, lam_gpu, device)

    # Move back to CPU
    cluster_assignments = loc_metrics['dominant_eigenspace'].cpu().numpy()
    participation_ratio = loc_metrics['participation_ratio'].cpu().numpy()
    max_projection = loc_metrics['max_projection'].cpu().numpy()
    feature_norms_sq = loc_metrics['feature_norms_sq'].cpu().numpy()

    # Compute cluster slopes with localization
    cluster_stats = compute_cluster_slopes_with_localization(
        feature_norms_sq, fractional_dims, cluster_assignments,
        eigenvalues, participation_ratio, max_projection
    )

    # Add file metadata to each cluster stat
    for cs in cluster_stats:
        cs['m_hidden'] = m_hidden
        cs['sparsity'] = sparsity
        cs['seed'] = seed
        cs['checkpoint_step'] = int(checkpoint_step)

    # Run-level statistics
    run_stats = {
        'm_hidden': m_hidden,
        'sparsity': sparsity,
        'seed': seed,
        'checkpoint_step': int(checkpoint_step),
        'n_clusters': len(cluster_stats),
        'mean_participation_ratio': float(np.mean(participation_ratio)),
        'mean_max_projection': float(np.mean(max_projection)),
    }

    return run_stats, cluster_stats


def aggregate_and_analyze(all_cluster_stats: List[Dict]) -> pd.DataFrame:
    """
    Aggregate all cluster statistics into a DataFrame for analysis.
    """
    if len(all_cluster_stats) == 0:
        print("WARNING: No cluster statistics to aggregate!")
        return pd.DataFrame()

    df = pd.DataFrame(all_cluster_stats)

    # Filter for quality: R² > 0.1 and n_features >= 20
    df_filtered = df[(df['r_squared'] > 0.1) & (df['n_features'] >= 20)].copy()

    # Additional computed columns
    df_filtered['slope_times_lambda'] = df_filtered['slope'] * df_filtered['eigenvalue']
    df_filtered['log_slope'] = np.log10(df_filtered['slope'].clip(lower=1e-10))
    df_filtered['log_inv_eigenvalue'] = np.log10(df_filtered['inv_eigenvalue'].clip(lower=1e-10))

    # Localization score (normalized, 0=delocalized, 1=localized)
    # Using max_projection as primary indicator
    mp_min, mp_max = df_filtered['mean_max_projection'].min(), df_filtered['mean_max_projection'].max()
    df_filtered['localization_score'] = (df_filtered['mean_max_projection'] - mp_min) / (mp_max - mp_min + 1e-10)

    return df_filtered


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """
    Create visualization plots with localization coloring.
    """
    slopes = df['slope'].values
    inv_eig = df['inv_eigenvalue'].values
    r2 = df['r_squared'].values
    n_feat = df['n_features'].values
    localization = df['mean_max_projection'].values  # Higher = more localized
    participation = df['mean_participation_ratio'].values  # Higher = more delocalized

    # Figure 1: Main plot - κ vs 1/λ colored by localization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel 1: κ vs 1/λ colored by max projection (localization)
    ax = axes[0, 0]

    # Custom colormap: blue (delocalized) -> red (localized)
    cmap = plt.cm.RdYlBu_r  # Red = high localization, Blue = low

    sc = ax.scatter(inv_eig, slopes, c=localization, cmap=cmap,
                    alpha=0.7, s=20, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Mean Max Projection (Localization)', fontsize=11)

    # Add y=x reference line
    max_val = max(np.percentile(inv_eig, 98), np.percentile(slopes, 98))
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.7, label='κ = 1/λ')

    # Regression line
    valid = np.isfinite(inv_eig) & np.isfinite(slopes)
    reg = stats.linregress(inv_eig[valid], slopes[valid])
    x_line = np.linspace(0, max_val, 100)
    ax.plot(x_line, reg.slope * x_line + reg.intercept, 'g-', linewidth=2,
            alpha=0.8, label=f'Fit: κ = {reg.slope:.3f}/λ + {reg.intercept:.4f}')

    ax.set_xlabel('1/λ (inverse eigenvalue)', fontsize=12)
    ax.set_ylabel('κ (cluster slope)', fontsize=12)
    ax.set_title(f'Slope vs Inverse Eigenvalue\nColored by Localization (r={reg.rvalue:.3f})', fontsize=13)
    ax.legend(loc='upper left')
    ax.set_xlim(0, np.percentile(inv_eig, 98))
    ax.set_ylim(0, np.percentile(slopes, 98))
    ax.grid(True, alpha=0.3)

    # Panel 2: κ vs 1/λ colored by participation ratio (inverse localization)
    ax = axes[0, 1]

    # Clip extreme participation ratios for better visualization
    pr_clipped = np.clip(participation, 1, np.percentile(participation, 95))

    sc = ax.scatter(inv_eig, slopes, c=pr_clipped, cmap='viridis_r',
                    alpha=0.7, s=20, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Mean Participation Ratio (Delocalization)', fontsize=11)

    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.7, label='κ = 1/λ')

    ax.set_xlabel('1/λ (inverse eigenvalue)', fontsize=12)
    ax.set_ylabel('κ (cluster slope)', fontsize=12)
    ax.set_title('Slope vs Inverse Eigenvalue\nColored by Participation Ratio', fontsize=13)
    ax.legend(loc='upper left')
    ax.set_xlim(0, np.percentile(inv_eig, 98))
    ax.set_ylim(0, np.percentile(slopes, 98))
    ax.grid(True, alpha=0.3)

    # Panel 3: Residual analysis by localization
    ax = axes[1, 0]

    # Predicted slope from eigenvalue
    predicted = 1.0 / df['eigenvalue'].values
    residuals = slopes - predicted
    relative_error = residuals / (predicted + 1e-10)

    sc = ax.scatter(localization, relative_error, c=r2, cmap='plasma',
                    alpha=0.6, s=20, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Cluster R² (fit quality)', fontsize=11)

    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Mean Max Projection (Localization)', fontsize=12)
    ax.set_ylabel('Relative Error: (κ - 1/λ) / (1/λ)', fontsize=12)
    ax.set_title('Fit Error vs Localization', fontsize=13)
    ax.set_ylim(np.percentile(relative_error, 1), np.percentile(relative_error, 99))
    ax.grid(True, alpha=0.3)

    # Panel 4: κ×λ distribution by localization bins
    ax = axes[1, 1]

    slope_times_lambda = df['slope_times_lambda'].values

    # Bin by localization
    n_bins = 4
    loc_bins = np.percentile(localization, np.linspace(0, 100, n_bins + 1))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, n_bins))

    for i in range(n_bins):
        mask = (localization >= loc_bins[i]) & (localization < loc_bins[i+1])
        if i == n_bins - 1:
            mask = (localization >= loc_bins[i])

        data = slope_times_lambda[mask]
        if len(data) > 10:
            label = f'Loc: {loc_bins[i]:.2f}-{loc_bins[i+1]:.2f}'
            ax.hist(data, bins=30, density=True, alpha=0.5, color=colors[i],
                   label=label, histtype='stepfilled')

    ax.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Predicted (κλ=1)')
    ax.set_xlabel('κ × λ (slope × eigenvalue)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of κ×λ by Localization Bins', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'slope_eigenvalue_localization.png', dpi=200, bbox_inches='tight')
    plt.savefig(output_dir / 'slope_eigenvalue_localization.pdf', dpi=200, bbox_inches='tight')
    print(f"Saved: slope_eigenvalue_localization.png/pdf")
    plt.close()

    # Figure 2: Correlation analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Localization vs fit quality
    ax = axes[0]
    ax.scatter(localization, r2, alpha=0.5, s=15, c='steelblue')

    # Bin and show trend
    loc_centers = []
    r2_means = []
    r2_stds = []
    for i in range(10):
        lo = np.percentile(localization, i*10)
        hi = np.percentile(localization, (i+1)*10)
        mask = (localization >= lo) & (localization < hi)
        if mask.sum() > 10:
            loc_centers.append((lo + hi) / 2)
            r2_means.append(np.mean(r2[mask]))
            r2_stds.append(np.std(r2[mask]))

    ax.errorbar(loc_centers, r2_means, yerr=r2_stds, fmt='ro-',
                capsize=3, capthick=2, linewidth=2, markersize=8, label='Binned mean')

    ax.set_xlabel('Mean Max Projection (Localization)', fontsize=12)
    ax.set_ylabel('Cluster R² (D vs ||W||² fit)', fontsize=12)
    ax.set_title('Fit Quality vs Localization', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Localization vs slope error
    ax = axes[1]
    slope_error = np.abs(slope_times_lambda - 1.0)
    ax.scatter(localization, slope_error, alpha=0.5, s=15, c='steelblue')

    # Bin and show trend
    err_means = []
    err_stds = []
    for i in range(10):
        lo = np.percentile(localization, i*10)
        hi = np.percentile(localization, (i+1)*10)
        mask = (localization >= lo) & (localization < hi)
        if mask.sum() > 10:
            err_means.append(np.mean(slope_error[mask]))
            err_stds.append(np.std(slope_error[mask]))

    ax.errorbar(loc_centers, err_means, yerr=err_stds, fmt='ro-',
                capsize=3, capthick=2, linewidth=2, markersize=8, label='Binned mean')

    ax.set_xlabel('Mean Max Projection (Localization)', fontsize=12)
    ax.set_ylabel('|κλ - 1| (absolute error)', fontsize=12)
    ax.set_title('Conjecture Error vs Localization', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Sparsity vs localization
    ax = axes[2]
    sparsity = df['sparsity'].values
    sc = ax.scatter(sparsity, localization, c=r2, cmap='viridis', alpha=0.5, s=15)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('R²', fontsize=11)

    ax.set_xlabel('Sparsity', fontsize=12)
    ax.set_ylabel('Mean Max Projection (Localization)', fontsize=12)
    ax.set_title('Localization vs Sparsity', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'localization_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'localization_analysis.pdf', dpi=150, bbox_inches='tight')
    print(f"Saved: localization_analysis.png/pdf")
    plt.close()

    # Figure 3: Stratified slope-eigenvalue plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Stratify by localization quartiles
    quartiles = np.percentile(localization, [0, 25, 50, 75, 100])
    quartile_labels = ['Q1: Low localization', 'Q2: Med-low localization',
                       'Q3: Med-high localization', 'Q4: High localization']

    for idx, ax in enumerate(axes.flat):
        lo, hi = quartiles[idx], quartiles[idx + 1]
        if idx == 3:
            mask = (localization >= lo)
        else:
            mask = (localization >= lo) & (localization < hi)

        x = inv_eig[mask]
        y = slopes[mask]
        r2_sub = r2[mask]

        sc = ax.scatter(x, y, c=r2_sub, cmap='plasma', alpha=0.6, s=25)
        plt.colorbar(sc, ax=ax, label='R²')

        max_val = max(np.percentile(x, 98) if len(x) > 0 else 1,
                      np.percentile(y, 98) if len(y) > 0 else 1)
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.7, label='κ = 1/λ')

        if len(x) > 10:
            reg = stats.linregress(x, y)
            x_line = np.linspace(0, max_val, 100)
            ax.plot(x_line, reg.slope * x_line + reg.intercept, 'g-', linewidth=2)

            title = f'{quartile_labels[idx]}\nr={reg.rvalue:.3f}, a={reg.slope:.3f}'
        else:
            title = f'{quartile_labels[idx]}\nn={len(x)} (insufficient data)'

        ax.set_xlabel('1/λ', fontsize=11)
        ax.set_ylabel('κ', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, max_val * 1.1)
        ax.set_ylim(0, max_val * 1.1)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Slope vs Inverse Eigenvalue Stratified by Localization Quartile', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'stratified_by_localization.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'stratified_by_localization.pdf', dpi=150, bbox_inches='tight')
    print(f"Saved: stratified_by_localization.png/pdf")
    plt.close()


def compute_summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for the analysis.
    """
    slopes = df['slope'].values
    inv_eig = df['inv_eigenvalue'].values
    localization = df['mean_max_projection'].values
    slope_times_lambda = df['slope_times_lambda'].values

    # Overall correlation
    valid = np.isfinite(inv_eig) & np.isfinite(slopes)
    pearson_r, pearson_p = stats.pearsonr(inv_eig[valid], slopes[valid])
    spearman_r, spearman_p = stats.spearmanr(inv_eig[valid], slopes[valid])
    reg = stats.linregress(inv_eig[valid], slopes[valid])

    # By localization quartile
    quartiles = np.percentile(localization, [0, 25, 50, 75, 100])
    quartile_stats = []

    for i in range(4):
        lo, hi = quartiles[i], quartiles[i + 1]
        if i == 3:
            mask = (localization >= lo)
        else:
            mask = (localization >= lo) & (localization < hi)

        x = inv_eig[mask]
        y = slopes[mask]

        if len(x) > 10:
            r, p = stats.pearsonr(x, y)
            sub_reg = stats.linregress(x, y)
            quartile_stats.append({
                'quartile': i + 1,
                'loc_range': [float(lo), float(hi)],
                'n_clusters': int(mask.sum()),
                'pearson_r': float(r),
                'pearson_p': float(p),
                'regression_slope': float(sub_reg.slope),
                'regression_r_squared': float(sub_reg.rvalue ** 2),
                'mean_slope_times_lambda': float(np.mean(slope_times_lambda[mask])),
                'std_slope_times_lambda': float(np.std(slope_times_lambda[mask])),
            })

    return {
        'n_clusters_analyzed': int(len(df)),
        'n_files': int(df.groupby(['m_hidden', 'sparsity', 'seed']).ngroups),

        # Overall correlation
        'pearson_correlation': float(pearson_r),
        'pearson_pvalue': float(pearson_p),
        'spearman_correlation': float(spearman_r),
        'spearman_pvalue': float(spearman_p),

        # Regression
        'regression_a': float(reg.slope),
        'regression_b': float(reg.intercept),
        'regression_r_squared': float(reg.rvalue ** 2),

        # κ×λ statistics
        'slope_times_lambda_mean': float(np.mean(slope_times_lambda)),
        'slope_times_lambda_std': float(np.std(slope_times_lambda)),
        'slope_times_lambda_median': float(np.median(slope_times_lambda)),

        # Localization statistics
        'mean_localization': float(np.mean(localization)),
        'std_localization': float(np.std(localization)),

        # By localization quartile
        'quartile_statistics': quartile_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Slope-Eigenvalue Analysis with Spectral Localization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--checkpoint', type=str, default='final',
                        choices=['early', 'mid', 'final'],
                        help='Which checkpoint to analyze')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Number of GPUs to use')
    parser.add_argument('--sample', type=int, default=None,
                        help='Process only N files for testing')

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Slope-Eigenvalue Analysis with Spectral Localization")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Using {args.num_gpus} GPU(s)")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Get list of SVD files
    svd_files = sorted(SVD_DIR.glob('svd_n1024_m*.h5'))
    print(f"Found {len(svd_files)} SVD files")

    if args.sample:
        svd_files = svd_files[:args.sample]
        print(f"Sampling {len(svd_files)} files for testing")

    checkpoint_idx = CHECKPOINT_INDICES[args.checkpoint]

    # Process all files
    all_cluster_stats = []
    all_run_stats = []

    start_time = time.time()

    for idx, svd_path in enumerate(tqdm(svd_files, desc="Processing files")):
        # Round-robin GPU assignment
        gpu_id = idx % args.num_gpus
        device = torch.device(f'cuda:{gpu_id}')

        # Get corresponding input file
        input_name = svd_path.stem.replace('svd_', '') + '.h5'
        input_path = INPUT_DIR / input_name

        if not input_path.exists():
            continue

        try:
            run_stats, cluster_stats = process_single_file(
                input_path, svd_path, checkpoint_idx, device
            )
            all_run_stats.append(run_stats)
            all_cluster_stats.extend(cluster_stats)
        except Exception as e:
            print(f"\nError processing {svd_path.name}: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\nProcessing complete in {elapsed:.1f}s")
    print(f"Total clusters: {len(all_cluster_stats)}")

    # Aggregate and analyze
    print("\nAggregating results...")
    df = aggregate_and_analyze(all_cluster_stats)

    if len(df) == 0:
        print("ERROR: No valid clusters found. Check data paths.")
        return

    print(f"Filtered clusters (R² > 0.1, n >= 20): {len(df)}")

    # Save data
    df.to_csv(OUTPUT_DIR / 'cluster_localization_data.csv', index=False)
    print(f"Saved: cluster_localization_data.csv")

    # Full cluster stats (unfiltered)
    pd.DataFrame(all_cluster_stats).to_csv(OUTPUT_DIR / 'all_cluster_stats.csv', index=False)
    print(f"Saved: all_cluster_stats.csv")

    # Run stats
    pd.DataFrame(all_run_stats).to_csv(OUTPUT_DIR / 'run_stats.csv', index=False)
    print(f"Saved: run_stats.csv")

    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary = compute_summary_statistics(df)

    with open(OUTPUT_DIR / 'summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: summary_statistics.json")

    # Print key results
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS")
    print("=" * 70)
    print(f"Clusters analyzed: {summary['n_clusters_analyzed']}")
    print(f"Files processed: {summary['n_files']}")
    print()
    print("Overall Correlation (κ vs 1/λ):")
    print(f"  Pearson r: {summary['pearson_correlation']:.4f}")
    print(f"  Regression: κ = {summary['regression_a']:.4f}/λ + {summary['regression_b']:.4f}")
    print(f"  R²: {summary['regression_r_squared']:.4f}")
    print()
    print("κ×λ Statistics (conjecture predicts = 1):")
    print(f"  Mean: {summary['slope_times_lambda_mean']:.4f} ± {summary['slope_times_lambda_std']:.4f}")
    print(f"  Median: {summary['slope_times_lambda_median']:.4f}")
    print()
    print("By Localization Quartile:")
    for qs in summary['quartile_statistics']:
        print(f"  Q{qs['quartile']} (loc {qs['loc_range'][0]:.2f}-{qs['loc_range'][1]:.2f}):")
        print(f"    r = {qs['pearson_r']:.3f}, κλ = {qs['mean_slope_times_lambda']:.3f} ± {qs['std_slope_times_lambda']:.3f}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("Creating visualizations...")
    print("=" * 70)
    create_visualizations(df, OUTPUT_DIR)

    print(f"\nAnalysis complete. End time: {datetime.now().isoformat()}")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
