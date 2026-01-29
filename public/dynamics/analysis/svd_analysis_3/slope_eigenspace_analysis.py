#!/usr/bin/env python3
"""
Slope-Eigenvalue Analysis with Eigenspace Localization (CORRECTED)
===================================================================

CRITICAL CORRECTION from svd_analysis_2:
----------------------------------------
The previous analysis computed projections onto individual EIGENVECTORS.
This is WRONG because:
1. Degenerate eigenvalues have multi-dimensional eigenspaces
2. The choice of basis within a degenerate eigenspace is arbitrary
3. Features cluster along eigenSPACES, not individual eigenvectors

This analysis correctly:
- Groups eigenvectors by their eigenvalue (with tolerance for near-degeneracy)
- Computes projection onto each eigenSPACE as sum of squared projections
- Measures localization as concentration across eigenspaces, not eigenvectors

Correct Formulation:
-------------------
For eigenspace S_λ (spanned by all eigenvectors with eigenvalue λ):
    p_{i,λ} = Σ_{k: λ_k = λ} |u_k^T w_i|² / ||w_i||²

This is the fraction of feature i's squared norm that lies in eigenspace S_λ.

Localization Metrics (over eigenspaces):
- Max eigenspace projection: max_λ p_{i,λ}
- Participation ratio: 1 / Σ_λ p_{i,λ}²
- Entropy: -Σ_λ p_{i,λ} log(p_{i,λ})

Usage:
------
    python slope_eigenspace_analysis.py [--num-gpus 8] [--eigenvalue-tol 1e-6]

Author: Claude Code Analysis Pipeline
Date: 2026-01-29
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Configuration
INPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
SVD_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/svd_results')
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/svd_analysis_3')

CHECKPOINT_INDICES = {
    'early': 10,
    'mid': 30,
    'final': -1,
}


def identify_eigenspaces(eigenvalues: np.ndarray, rel_tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify eigenspaces by grouping eigenvalues that are equal (within tolerance).

    For degenerate eigenvalues (λ_k = λ_{k+1} = ... = λ_{k+d-1}),
    all corresponding eigenvectors span a single d-dimensional eigenspace.

    Args:
        eigenvalues: (m,) array of eigenvalues in descending order
        rel_tol: relative tolerance for considering eigenvalues equal

    Returns:
        space_assignments: (m,) array mapping each eigenvector to its eigenspace index
        space_eigenvalues: (n_spaces,) array of unique eigenvalue for each space
    """
    m = len(eigenvalues)
    if m == 0:
        return np.array([], dtype=int), np.array([])

    # Use relative tolerance based on largest eigenvalue
    max_eig = np.abs(eigenvalues[0]) if eigenvalues[0] != 0 else 1.0
    abs_tol = rel_tol * max_eig

    # Group consecutive eigenvalues that are within tolerance
    space_assignments = np.zeros(m, dtype=int)
    space_eigenvalues = []

    current_space = 0
    current_eig = eigenvalues[0]
    space_start = 0

    for k in range(m):
        if np.abs(eigenvalues[k] - current_eig) > abs_tol:
            # New eigenspace starts
            # Record the mean eigenvalue for the previous space
            space_eigenvalues.append(np.mean(eigenvalues[space_start:k]))
            current_space += 1
            current_eig = eigenvalues[k]
            space_start = k
        space_assignments[k] = current_space

    # Don't forget the last space
    space_eigenvalues.append(np.mean(eigenvalues[space_start:]))

    return space_assignments, np.array(space_eigenvalues)


def compute_eigenspace_projections_gpu(
    weights: torch.Tensor,      # (m, n)
    U: torch.Tensor,            # (m, m)
    eigenvalues: torch.Tensor,  # (m,)
    space_assignments: torch.Tensor,  # (m,) int - which eigenspace each eigenvector belongs to
    n_spaces: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Compute projections onto eigenSPACES (not individual eigenvectors).

    For each eigenspace s:
        p_{i,s} = Σ_{k ∈ space s} |u_k^T w_i|² / ||w_i||²

    This is the correct basis-invariant measure of how much of feature i
    lies in eigenspace s.

    Returns:
        Dict with:
        - eigenspace_projections: (n_spaces, n) - p_{i,s} for each feature and space
        - participation_ratio: (n,) - 1 / Σ_s p_{i,s}² (over spaces)
        - max_space_projection: (n,) - max_s p_{i,s}
        - dominant_space: (n,) - argmax_s p_{i,s}
        - feature_norms_sq: (n,) - ||w_i||²
    """
    m, n = weights.shape

    # Project features onto all eigenvectors: z_ki = u_k^T w_i
    # U: (m, m) where columns are eigenvectors
    # weights: (m, n) where columns are features
    projections = U.T @ weights  # (m, n)
    projections_sq = projections ** 2  # (m, n) - |u_k^T w_i|²

    # Feature norms squared
    norms_sq = (weights ** 2).sum(dim=0)  # (n,)
    norms_sq_safe = torch.clamp(norms_sq, min=1e-12)

    # Aggregate projections by eigenspace
    # p_{i,s} = Σ_{k: space[k]=s} |u_k^T w_i|² / ||w_i||²
    eigenspace_proj = torch.zeros(n_spaces, n, dtype=torch.float64, device=device)

    for s in range(n_spaces):
        mask = space_assignments == s  # which eigenvectors belong to space s
        if mask.any():
            # Sum squared projections onto all eigenvectors in this space
            eigenspace_proj[s] = projections_sq[mask].sum(dim=0) / norms_sq_safe

    # Localization metrics over eigenspaces
    # Participation ratio: 1 / Σ_s p_{i,s}²
    p_sq_sum = (eigenspace_proj ** 2).sum(dim=0)  # (n,)
    p_sq_sum_safe = torch.clamp(p_sq_sum, min=1e-12)
    participation_ratio = 1.0 / p_sq_sum_safe  # (n,)

    # Max eigenspace projection
    max_space_proj, dominant_space = eigenspace_proj.max(dim=0)  # (n,), (n,)

    # Entropy over eigenspaces: H_i = -Σ_s p_{i,s} log(p_{i,s})
    p_safe = torch.clamp(eigenspace_proj, min=1e-12)
    entropy = -(eigenspace_proj * torch.log(p_safe)).sum(dim=0)  # (n,)

    return {
        'eigenspace_projections': eigenspace_proj,  # (n_spaces, n)
        'participation_ratio': participation_ratio,  # (n,)
        'max_space_projection': max_space_proj,      # (n,)
        'dominant_space': dominant_space,            # (n,)
        'feature_norms_sq': norms_sq,               # (n,)
        'projection_entropy': entropy,              # (n,)
    }


def compute_cluster_slopes_with_localization(
    feature_norms_sq: np.ndarray,     # (n,)
    fractional_dims: np.ndarray,      # (n,)
    dominant_space: np.ndarray,       # (n,) - which eigenspace each feature belongs to
    space_eigenvalues: np.ndarray,    # (n_spaces,) - eigenvalue of each space
    participation_ratio: np.ndarray,  # (n,) - over eigenspaces
    max_space_projection: np.ndarray, # (n,) - max projection onto any eigenspace
    min_cluster_size: int = 10
) -> List[Dict]:
    """
    Compute slope of D_i vs ||W_i||² for features clustered by dominant eigenSPACE.

    Returns:
        List of dicts, one per eigenspace with valid fit.
    """
    n_spaces = len(space_eigenvalues)
    cluster_stats = []

    for s in range(n_spaces):
        mask = dominant_space == s
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

        # Eigenvalue of this space and predicted slope
        lambda_s = space_eigenvalues[s]
        predicted_slope = 1.0 / lambda_s if lambda_s > 1e-10 else np.nan

        # Localization metrics for features in this eigenspace cluster
        pr_values = participation_ratio[mask][valid]
        mp_values = max_space_projection[mask][valid]

        cluster_stats.append({
            'eigenspace_idx': int(s),
            'eigenvalue': float(lambda_s),
            'inv_eigenvalue': float(1.0 / lambda_s) if lambda_s > 1e-10 else np.nan,
            'n_features': int(n_features),
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'predicted_slope': float(predicted_slope),
            'slope_ratio': float(slope / predicted_slope) if np.isfinite(predicted_slope) and predicted_slope != 0 else np.nan,
            # Localization metrics (over eigenSPACES, not eigenvectors)
            'mean_participation_ratio': float(np.mean(pr_values)),
            'std_participation_ratio': float(np.std(pr_values)),
            'mean_max_space_projection': float(np.mean(mp_values)),
            'std_max_space_projection': float(np.std(mp_values)),
            'mean_norm_sq': float(np.mean(x_valid)),
            'mean_D': float(np.mean(y_valid)),
        })

    return cluster_stats


def process_single_file(
    input_path: Path,
    svd_path: Path,
    checkpoint_idx: int,
    device: torch.device,
    eigenvalue_tol: float = 1e-4
) -> Tuple[Dict, List[Dict]]:
    """
    Process a single file with eigenspace-based analysis.
    """
    # Load original data
    with h5py.File(input_path, 'r') as f:
        weights = f['weights'][checkpoint_idx]  # (m, n)
        fractional_dims = f['fractional_dims'][checkpoint_idx]  # (n,)
        checkpoint_step = f['checkpoint_steps'][checkpoint_idx]
        m_hidden = int(f.attrs['m_hidden'])
        sparsity = float(f.attrs['sparsity'])
        seed = int(f.attrs['seed'])

    # Load SVD results
    with h5py.File(svd_path, 'r') as f:
        U = f['U'][checkpoint_idx]  # (m, m)
        eigenvalues = f['eigenvalues'][checkpoint_idx]  # (m,)

    # Identify eigenspaces (group degenerate eigenvalues)
    space_assignments, space_eigenvalues = identify_eigenspaces(eigenvalues, rel_tol=eigenvalue_tol)
    n_spaces = len(space_eigenvalues)

    # Move to GPU
    W_gpu = torch.tensor(weights, dtype=torch.float64, device=device)
    U_gpu = torch.tensor(U, dtype=torch.float64, device=device)
    lam_gpu = torch.tensor(eigenvalues, dtype=torch.float64, device=device)
    space_assign_gpu = torch.tensor(space_assignments, dtype=torch.int64, device=device)

    # Compute eigenspace projections (correct method)
    proj_metrics = compute_eigenspace_projections_gpu(
        W_gpu, U_gpu, lam_gpu, space_assign_gpu, n_spaces, device
    )

    # Move back to CPU
    dominant_space = proj_metrics['dominant_space'].cpu().numpy()
    participation_ratio = proj_metrics['participation_ratio'].cpu().numpy()
    max_space_proj = proj_metrics['max_space_projection'].cpu().numpy()
    feature_norms_sq = proj_metrics['feature_norms_sq'].cpu().numpy()

    # Compute cluster slopes with eigenspace localization
    cluster_stats = compute_cluster_slopes_with_localization(
        feature_norms_sq, fractional_dims, dominant_space,
        space_eigenvalues, participation_ratio, max_space_proj
    )

    # Add file metadata to each cluster stat
    for cs in cluster_stats:
        cs['m_hidden'] = m_hidden
        cs['sparsity'] = sparsity
        cs['seed'] = seed
        cs['checkpoint_step'] = int(checkpoint_step)
        cs['n_eigenspaces'] = n_spaces
        cs['n_eigenvectors'] = len(eigenvalues)

    # Run-level statistics
    run_stats = {
        'm_hidden': m_hidden,
        'sparsity': sparsity,
        'seed': seed,
        'checkpoint_step': int(checkpoint_step),
        'n_eigenspaces': n_spaces,
        'n_eigenvectors': len(eigenvalues),
        'degeneracy_ratio': n_spaces / len(eigenvalues) if len(eigenvalues) > 0 else 1.0,
        'n_clusters': len(cluster_stats),
        'mean_participation_ratio': float(np.mean(participation_ratio)),
        'mean_max_space_projection': float(np.mean(max_space_proj)),
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

    if len(df_filtered) == 0:
        print("WARNING: No clusters pass quality filter!")
        return df

    # Additional computed columns
    df_filtered['slope_times_lambda'] = df_filtered['slope'] * df_filtered['eigenvalue']
    df_filtered['log_slope'] = np.log10(df_filtered['slope'].clip(lower=1e-10))
    df_filtered['log_inv_eigenvalue'] = np.log10(df_filtered['inv_eigenvalue'].clip(lower=1e-10))

    # Localization score (normalized)
    mp_min = df_filtered['mean_max_space_projection'].min()
    mp_max = df_filtered['mean_max_space_projection'].max()
    df_filtered['localization_score'] = (df_filtered['mean_max_space_projection'] - mp_min) / (mp_max - mp_min + 1e-10)

    return df_filtered


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """
    Create visualization plots with eigenspace localization coloring.
    """
    slopes = df['slope'].values
    inv_eig = df['inv_eigenvalue'].values
    r2 = df['r_squared'].values
    localization = df['mean_max_space_projection'].values
    participation = df['mean_participation_ratio'].values

    # Figure 1: Main plot - κ vs 1/λ colored by eigenspace localization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel 1: κ vs 1/λ colored by max eigenspace projection
    ax = axes[0, 0]
    cmap = plt.cm.RdYlBu_r

    sc = ax.scatter(inv_eig, slopes, c=localization, cmap=cmap,
                    alpha=0.7, s=20, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Mean Max Eigenspace Projection', fontsize=11)

    # Reference line and regression
    max_val = max(np.percentile(inv_eig, 98), np.percentile(slopes, 98))
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.7, label='κ = 1/λ')

    valid = np.isfinite(inv_eig) & np.isfinite(slopes)
    reg = stats.linregress(inv_eig[valid], slopes[valid])
    x_line = np.linspace(0, max_val, 100)
    ax.plot(x_line, reg.slope * x_line + reg.intercept, 'g-', linewidth=2,
            alpha=0.8, label=f'Fit: κ = {reg.slope:.3f}/λ + {reg.intercept:.4f}')

    ax.set_xlabel('1/λ (inverse eigenvalue)', fontsize=12)
    ax.set_ylabel('κ (cluster slope)', fontsize=12)
    ax.set_title(f'Slope vs Inverse Eigenvalue (EIGENSPACE Localization)\nr = {reg.rvalue:.3f}', fontsize=13)
    ax.legend(loc='upper left')
    ax.set_xlim(0, np.percentile(inv_eig, 98))
    ax.set_ylim(0, np.percentile(slopes, 98))
    ax.grid(True, alpha=0.3)

    # Panel 2: κ vs 1/λ colored by participation ratio (over eigenspaces)
    ax = axes[0, 1]
    pr_clipped = np.clip(participation, 1, np.percentile(participation, 95))

    sc = ax.scatter(inv_eig, slopes, c=pr_clipped, cmap='viridis_r',
                    alpha=0.7, s=20, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Participation Ratio (# eigenspaces)', fontsize=11)

    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.7, label='κ = 1/λ')
    ax.set_xlabel('1/λ (inverse eigenvalue)', fontsize=12)
    ax.set_ylabel('κ (cluster slope)', fontsize=12)
    ax.set_title('Colored by Participation Ratio (over eigenspaces)', fontsize=13)
    ax.legend(loc='upper left')
    ax.set_xlim(0, np.percentile(inv_eig, 98))
    ax.set_ylim(0, np.percentile(slopes, 98))
    ax.grid(True, alpha=0.3)

    # Panel 3: Residual analysis
    ax = axes[1, 0]
    predicted = 1.0 / df['eigenvalue'].values
    relative_error = (slopes - predicted) / (predicted + 1e-10)

    sc = ax.scatter(localization, relative_error, c=r2, cmap='plasma',
                    alpha=0.6, s=20, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Cluster R²', fontsize=11)

    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Mean Max Eigenspace Projection', fontsize=12)
    ax.set_ylabel('Relative Error: (κ - 1/λ) / (1/λ)', fontsize=12)
    ax.set_title('Fit Error vs Eigenspace Localization', fontsize=13)
    ax.set_ylim(np.percentile(relative_error, 1), np.percentile(relative_error, 99))
    ax.grid(True, alpha=0.3)

    # Panel 4: κ×λ distribution by localization bins
    ax = axes[1, 1]
    slope_times_lambda = df['slope_times_lambda'].values

    n_bins = 4
    loc_bins = np.percentile(localization, np.linspace(0, 100, n_bins + 1))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, n_bins))

    for i in range(n_bins):
        mask = (localization >= loc_bins[i]) & (localization < loc_bins[i+1])
        if i == n_bins - 1:
            mask = localization >= loc_bins[i]

        data = slope_times_lambda[mask]
        if len(data) > 10:
            label = f'Loc: {loc_bins[i]:.2f}-{loc_bins[i+1]:.2f}'
            ax.hist(data, bins=30, density=True, alpha=0.5, color=colors[i],
                   label=label, histtype='stepfilled')

    ax.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Predicted (κλ=1)')
    ax.set_xlabel('κ × λ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of κ×λ by Eigenspace Localization', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'slope_eigenspace_localization.png', dpi=200, bbox_inches='tight')
    plt.savefig(output_dir / 'slope_eigenspace_localization.pdf', dpi=200, bbox_inches='tight')
    print(f"Saved: slope_eigenspace_localization.png/pdf")
    plt.close()

    # Figure 2: Stratified by localization quartile
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    quartiles = np.percentile(localization, [0, 25, 50, 75, 100])
    quartile_labels = ['Q1: Low eigenspace loc.', 'Q2: Med-low eigenspace loc.',
                       'Q3: Med-high eigenspace loc.', 'Q4: High eigenspace loc.']

    for idx, ax in enumerate(axes.flat):
        lo, hi = quartiles[idx], quartiles[idx + 1]
        if idx == 3:
            mask = localization >= lo
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
            sub_reg = stats.linregress(x, y)
            x_line = np.linspace(0, max_val, 100)
            ax.plot(x_line, sub_reg.slope * x_line + sub_reg.intercept, 'g-', linewidth=2)
            title = f'{quartile_labels[idx]}\nr={sub_reg.rvalue:.3f}, a={sub_reg.slope:.3f}'
        else:
            title = f'{quartile_labels[idx]}\nn={len(x)} (insufficient)'

        ax.set_xlabel('1/λ', fontsize=11)
        ax.set_ylabel('κ', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, max_val * 1.1)
        ax.set_ylim(0, max_val * 1.1)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Slope vs 1/λ Stratified by EIGENSPACE Localization', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'stratified_by_eigenspace_loc.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'stratified_by_eigenspace_loc.pdf', dpi=150, bbox_inches='tight')
    print(f"Saved: stratified_by_eigenspace_loc.png/pdf")
    plt.close()

    # Figure 3: Degeneracy analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Get sparsity for coloring
    sparsity = df['sparsity'].values

    # Panel 1: Localization vs fit quality, colored by sparsity
    ax = axes[0]
    sc = ax.scatter(localization, r2, alpha=0.6, s=15, c=sparsity, cmap='viridis', vmin=0, vmax=1)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Sparsity', fontsize=11)

    loc_centers, r2_means, r2_stds = [], [], []
    for i in range(10):
        lo = np.percentile(localization, i*10)
        hi = np.percentile(localization, (i+1)*10)
        m = (localization >= lo) & (localization < hi)
        if m.sum() > 10:
            loc_centers.append((lo + hi) / 2)
            r2_means.append(np.mean(r2[m]))
            r2_stds.append(np.std(r2[m]))

    ax.errorbar(loc_centers, r2_means, yerr=r2_stds, fmt='ko-',
                capsize=3, capthick=2, linewidth=2, markersize=8, label='Binned mean')
    ax.set_xlabel('Mean Max Eigenspace Projection', fontsize=12)
    ax.set_ylabel('Cluster R²', fontsize=12)
    ax.set_title('Fit Quality vs Eigenspace Localization', fontsize=13)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # Panel 2: Sparsity vs eigenspace localization
    ax = axes[1]
    sc = ax.scatter(sparsity, localization, c=r2, cmap='viridis', alpha=0.5, s=15)
    plt.colorbar(sc, ax=ax, label='R²')
    ax.set_xlabel('Sparsity', fontsize=12)
    ax.set_ylabel('Mean Max Eigenspace Projection', fontsize=12)
    ax.set_title('Eigenspace Localization vs Sparsity', fontsize=13)
    ax.grid(True, alpha=0.3)

    # Panel 3: Degeneracy ratio
    ax = axes[2]
    if 'n_eigenspaces' in df.columns and 'n_eigenvectors' in df.columns:
        degeneracy = df['n_eigenspaces'].values / df['n_eigenvectors'].values
        sc = ax.scatter(degeneracy, localization, c=r2, cmap='viridis', alpha=0.5, s=15)
        plt.colorbar(sc, ax=ax, label='R²')
        ax.set_xlabel('Degeneracy Ratio (n_spaces / n_vectors)', fontsize=12)
        ax.set_ylabel('Mean Max Eigenspace Projection', fontsize=12)
        ax.set_title('Localization vs Eigenvalue Degeneracy', fontsize=13)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'eigenspace_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'eigenspace_analysis.pdf', dpi=150, bbox_inches='tight')
    print(f"Saved: eigenspace_analysis.png/pdf")
    plt.close()

    # Figure 4: STANDALONE - Fit Quality vs Localization colored by Sparsity
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot colored by sparsity (purple → blue → green → yellow)
    sc = ax.scatter(localization, r2, c=sparsity, cmap='viridis',
                    alpha=0.7, s=25, edgecolors='none', vmin=0, vmax=1)

    # Colorbar with larger font
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Sparsity', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    # Binned means with error bars (black for visibility)
    loc_centers, r2_means, r2_stds = [], [], []
    for i in range(10):
        lo = np.percentile(localization, i*10)
        hi = np.percentile(localization, (i+1)*10)
        m = (localization >= lo) & (localization < hi)
        if m.sum() > 10:
            loc_centers.append((lo + hi) / 2)
            r2_means.append(np.mean(r2[m]))
            r2_stds.append(np.std(r2[m]))

    ax.errorbar(loc_centers, r2_means, yerr=r2_stds, fmt='ko-',
                capsize=4, capthick=2, linewidth=2.5, markersize=10,
                label='Binned mean ± std', zorder=10)

    # Labels and title with larger fonts
    ax.set_xlabel('Mean Max Eigenspace Projection (Localization)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cluster R² (Fit Quality)', fontsize=14, fontweight='bold')
    ax.set_title('Fit Quality vs Eigenspace Localization\nColored by Sparsity', fontsize=16, fontweight='bold')

    # Legend with larger font
    ax.legend(loc='lower left', fontsize=12, framealpha=0.9)

    # Tick labels
    ax.tick_params(axis='both', labelsize=12)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set axis limits with some padding
    ax.set_xlim(0, localization.max() * 1.05)
    ax.set_ylim(r2.min() * 0.95, 1.02)

    plt.tight_layout()
    plt.savefig(output_dir / 'fit_quality_vs_localization.png', dpi=200, bbox_inches='tight')
    plt.savefig(output_dir / 'fit_quality_vs_localization.pdf', dpi=200, bbox_inches='tight')
    print(f"Saved: fit_quality_vs_localization.png/pdf (STANDALONE)")
    plt.close()


def compute_summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics.
    """
    slopes = df['slope'].values
    inv_eig = df['inv_eigenvalue'].values
    localization = df['mean_max_space_projection'].values
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
        mask = (localization >= lo) if i == 3 else (localization >= lo) & (localization < hi)

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
                'regression_slope': float(sub_reg.slope),
                'regression_r_squared': float(sub_reg.rvalue ** 2),
                'mean_slope_times_lambda': float(np.mean(slope_times_lambda[mask])),
                'std_slope_times_lambda': float(np.std(slope_times_lambda[mask])),
            })

    return {
        'n_clusters_analyzed': int(len(df)),
        'n_files': int(df.groupby(['m_hidden', 'sparsity', 'seed']).ngroups),
        'pearson_correlation': float(pearson_r),
        'pearson_pvalue': float(pearson_p),
        'spearman_correlation': float(spearman_r),
        'spearman_pvalue': float(spearman_p),
        'regression_a': float(reg.slope),
        'regression_b': float(reg.intercept),
        'regression_r_squared': float(reg.rvalue ** 2),
        'slope_times_lambda_mean': float(np.mean(slope_times_lambda)),
        'slope_times_lambda_std': float(np.std(slope_times_lambda)),
        'slope_times_lambda_median': float(np.median(slope_times_lambda)),
        'mean_localization': float(np.mean(localization)),
        'std_localization': float(np.std(localization)),
        'quartile_statistics': quartile_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Slope-Eigenvalue Analysis with EIGENSPACE Localization (Corrected)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--checkpoint', type=str, default='final',
                        choices=['early', 'mid', 'final'])
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--eigenvalue-tol', type=float, default=1e-4,
                        help='Relative tolerance for grouping degenerate eigenvalues')
    parser.add_argument('--sample', type=int, default=None)

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Slope-Eigenvalue Analysis with EIGENSPACE Localization (CORRECTED)")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Eigenvalue tolerance: {args.eigenvalue_tol}")
    print(f"Using {args.num_gpus} GPU(s)")
    print()
    print("CORRECTION: Now computing projection onto eigenSPACES, not eigenvectors")
    print("            Degenerate eigenvalues are grouped into single subspaces")
    print()

    svd_files = sorted(SVD_DIR.glob('svd_n1024_m*.h5'))
    print(f"Found {len(svd_files)} SVD files")

    if args.sample:
        svd_files = svd_files[:args.sample]
        print(f"Sampling {len(svd_files)} files")

    checkpoint_idx = CHECKPOINT_INDICES[args.checkpoint]

    all_cluster_stats = []
    all_run_stats = []

    start_time = time.time()

    for idx, svd_path in enumerate(tqdm(svd_files, desc="Processing")):
        gpu_id = idx % args.num_gpus
        device = torch.device(f'cuda:{gpu_id}')

        input_name = svd_path.stem.replace('svd_', '') + '.h5'
        input_path = INPUT_DIR / input_name

        if not input_path.exists():
            continue

        try:
            run_stats, cluster_stats = process_single_file(
                input_path, svd_path, checkpoint_idx, device, args.eigenvalue_tol
            )
            all_run_stats.append(run_stats)
            all_cluster_stats.extend(cluster_stats)
        except Exception as e:
            print(f"\nError processing {svd_path.name}: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\nProcessing complete in {elapsed:.1f}s")
    print(f"Total clusters: {len(all_cluster_stats)}")

    # Aggregate
    df = aggregate_and_analyze(all_cluster_stats)

    if len(df) == 0:
        print("ERROR: No valid clusters found")
        return

    print(f"Filtered clusters: {len(df)}")

    # Save data
    df.to_csv(OUTPUT_DIR / 'eigenspace_cluster_data.csv', index=False)
    pd.DataFrame(all_cluster_stats).to_csv(OUTPUT_DIR / 'all_eigenspace_stats.csv', index=False)
    pd.DataFrame(all_run_stats).to_csv(OUTPUT_DIR / 'run_stats.csv', index=False)

    # Summary statistics
    summary = compute_summary_statistics(df)
    with open(OUTPUT_DIR / 'summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print results
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS (EIGENSPACE-BASED)")
    print("=" * 70)
    print(f"Clusters analyzed: {summary['n_clusters_analyzed']}")
    print(f"Files processed: {summary['n_files']}")
    print()
    print("Overall Correlation (κ vs 1/λ):")
    print(f"  Pearson r: {summary['pearson_correlation']:.4f}")
    print(f"  Regression: κ = {summary['regression_a']:.4f}/λ + {summary['regression_b']:.4f}")
    print(f"  R²: {summary['regression_r_squared']:.4f}")
    print()
    print("κ×λ Statistics:")
    print(f"  Mean: {summary['slope_times_lambda_mean']:.4f} ± {summary['slope_times_lambda_std']:.4f}")
    print(f"  Median: {summary['slope_times_lambda_median']:.4f}")
    print()
    print("By Eigenspace Localization Quartile:")
    for qs in summary['quartile_statistics']:
        print(f"  Q{qs['quartile']} (loc {qs['loc_range'][0]:.2f}-{qs['loc_range'][1]:.2f}):")
        print(f"    r = {qs['pearson_r']:.3f}, κλ = {qs['mean_slope_times_lambda']:.3f} ± {qs['std_slope_times_lambda']:.3f}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("Creating visualizations...")
    create_visualizations(df, OUTPUT_DIR)

    print(f"\nAnalysis complete. End time: {datetime.now().isoformat()}")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
