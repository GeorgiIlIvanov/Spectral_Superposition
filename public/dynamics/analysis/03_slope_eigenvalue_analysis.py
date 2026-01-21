#!/usr/bin/env python3
"""
Phase 3: Slope-Eigenvalue Correlation Analysis

Test the conjecture: κ_i ≈ 1/λ_k where:
  - κ_i = slope of D_i vs ||W_i||^2 for features in cluster k
  - λ_k = eigenvalue of eigenspace k (= σ_k^2 from SVD)

This script:
1. Groups features by their dominant eigenspace
2. For each cluster, computes the slope κ_k of D_i vs ||W_i||^2
3. Tests correlation between κ_k and 1/λ_k

Usage:
    python 03_slope_eigenvalue_analysis.py [--checkpoint final|early|all]
"""

import h5py
import numpy as np
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime


# Configuration
INPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
SVD_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/svd_results')
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/slope_eigenvalue_results')

# Checkpoint indices to analyze
CHECKPOINT_INDICES = {
    'early': 10,   # ~step 2000
    'mid': 30,     # ~step 7500
    'final': -1,   # step 25000
}


def compute_cluster_slopes(feature_norms, fractional_dims, cluster_assignments, eigenvalues, min_cluster_size=10):
    """
    Compute the slope of D_i vs ||W_i||^2 for each eigenspace cluster.

    Args:
        feature_norms: (n,) - ||W_i||^2 for each feature
        fractional_dims: (n,) - D_i for each feature
        cluster_assignments: (n,) - dominant eigenspace index for each feature
        eigenvalues: (m,) - eigenvalues of WW^T
        min_cluster_size: minimum features in cluster to compute slope

    Returns:
        cluster_stats: dict mapping cluster_id to stats
    """
    unique_clusters = np.unique(cluster_assignments)
    cluster_stats = {}

    for k in unique_clusters:
        mask = cluster_assignments == k
        n_features = np.sum(mask)

        if n_features < min_cluster_size:
            continue

        x = feature_norms[mask]  # ||W_i||^2
        y = fractional_dims[mask]  # D_i

        # Filter out invalid values
        valid = np.isfinite(x) & np.isfinite(y) & (x > 1e-6)
        if np.sum(valid) < min_cluster_size:
            continue

        x_valid = x[valid]
        y_valid = y[valid]

        # Linear regression: D_i = slope * ||W_i||^2 + intercept
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)

        # The conjecture: slope ≈ 1/λ_k
        lambda_k = eigenvalues[k]
        predicted_slope = 1.0 / lambda_k if lambda_k > 1e-10 else np.nan

        cluster_stats[int(k)] = {
            'n_features': int(n_features),
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'eigenvalue': float(lambda_k),
            'predicted_slope': float(predicted_slope),
            'slope_ratio': float(slope / predicted_slope) if np.isfinite(predicted_slope) and predicted_slope != 0 else np.nan,
            'mean_norm': float(np.mean(x_valid)),
            'mean_D': float(np.mean(y_valid)),
        }

    return cluster_stats


def analyze_single_file(input_path, svd_path, checkpoint_idx=-1):
    """Analyze slope-eigenvalue relationship for a single file."""

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

    m, n = weights.shape

    # Compute feature projections and cluster assignments
    # Project each feature onto eigenspaces
    projections = U.T @ weights  # (m, n) - projections[k, i] = u_k^T @ W_i
    proj_squared = projections ** 2  # (m, n)

    # Normalize by feature norms
    norms_sq = np.sum(weights ** 2, axis=0)  # (n,)
    norms_sq = np.maximum(norms_sq, 1e-10)
    normalized_proj = proj_squared / norms_sq  # (m, n)

    # Dominant eigenspace for each feature
    cluster_assignments = np.argmax(normalized_proj, axis=0)  # (n,)

    # Compute cluster slopes
    cluster_stats = compute_cluster_slopes(
        feature_norms, fractional_dims, cluster_assignments, eigenvalues
    )

    # Also compute overall slope (all features together)
    valid = np.isfinite(feature_norms) & np.isfinite(fractional_dims) & (feature_norms > 1e-6)
    if np.sum(valid) > 10:
        overall_slope, overall_intercept, r_val, _, _ = stats.linregress(
            feature_norms[valid], fractional_dims[valid]
        )
        overall_r2 = r_val ** 2
    else:
        overall_slope, overall_intercept, overall_r2 = np.nan, np.nan, np.nan

    return {
        'm_hidden': m_hidden,
        'sparsity': sparsity,
        'seed': seed,
        'checkpoint_step': int(checkpoint_step),
        'cluster_stats': cluster_stats,
        'overall_slope': float(overall_slope),
        'overall_intercept': float(overall_intercept),
        'overall_r_squared': float(overall_r2),
        'eigenvalues': eigenvalues.tolist(),
        'n_clusters': len(cluster_stats),
    }


def aggregate_results(all_results):
    """Aggregate results and test the main conjecture."""

    # Collect all (slope, eigenvalue) pairs
    all_slopes = []
    all_eigenvalues = []
    all_inv_eigenvalues = []
    all_r_squared = []
    all_n_features = []
    all_metadata = []

    for r in all_results:
        for k, cstats in r['cluster_stats'].items():
            if cstats['r_squared'] > 0.1 and cstats['n_features'] >= 20:  # Quality filter
                all_slopes.append(cstats['slope'])
                all_eigenvalues.append(cstats['eigenvalue'])
                all_inv_eigenvalues.append(1.0 / cstats['eigenvalue'] if cstats['eigenvalue'] > 1e-10 else np.nan)
                all_r_squared.append(cstats['r_squared'])
                all_n_features.append(cstats['n_features'])
                all_metadata.append({
                    'm_hidden': r['m_hidden'],
                    'sparsity': r['sparsity'],
                    'cluster': k
                })

    slopes = np.array(all_slopes)
    eigenvalues = np.array(all_eigenvalues)
    inv_eigenvalues = np.array(all_inv_eigenvalues)
    r_squared = np.array(all_r_squared)
    n_features = np.array(all_n_features)

    # Filter valid entries
    valid = np.isfinite(slopes) & np.isfinite(inv_eigenvalues) & (slopes > 0)
    slopes_valid = slopes[valid]
    inv_eig_valid = inv_eigenvalues[valid]
    eig_valid = eigenvalues[valid]
    r2_valid = r_squared[valid]
    n_feat_valid = n_features[valid]

    # Test 1: Correlation between slope and 1/λ
    corr_inv, p_corr_inv = stats.pearsonr(slopes_valid, inv_eig_valid)
    spearman_inv, p_spearman_inv = stats.spearmanr(slopes_valid, inv_eig_valid)

    # Test 2: Linear regression: slope = a * (1/λ) + b
    # Expectation: a ≈ 1, b ≈ 0
    reg_slope, reg_intercept, reg_r, reg_p, reg_stderr = stats.linregress(inv_eig_valid, slopes_valid)

    # Test 3: Weighted regression (weight by R² and n_features)
    weights = r2_valid * np.sqrt(n_feat_valid)
    weights /= weights.sum()

    # Weighted mean of slope ratio κ * λ (should be ≈ 1)
    slope_times_lambda = slopes_valid * eig_valid
    weighted_mean_ratio = np.average(slope_times_lambda, weights=weights)
    weighted_std_ratio = np.sqrt(np.average((slope_times_lambda - weighted_mean_ratio)**2, weights=weights))

    return {
        'n_clusters_analyzed': int(len(slopes_valid)),
        'n_files': len(all_results),

        # Correlation tests
        'pearson_correlation': float(corr_inv),
        'pearson_pvalue': float(p_corr_inv),
        'spearman_correlation': float(spearman_inv),
        'spearman_pvalue': float(p_spearman_inv),

        # Regression: slope = a * (1/λ) + b
        'regression_a': float(reg_slope),
        'regression_b': float(reg_intercept),
        'regression_r_squared': float(reg_r ** 2),
        'regression_pvalue': float(reg_p),

        # Direct test: κ * λ ≈ 1
        'slope_times_lambda_mean': float(weighted_mean_ratio),
        'slope_times_lambda_std': float(weighted_std_ratio),

        # Raw data for plotting
        'slopes': slopes_valid.tolist(),
        'eigenvalues': eig_valid.tolist(),
        'inv_eigenvalues': inv_eig_valid.tolist(),
        'cluster_r_squared': r2_valid.tolist(),
        'cluster_n_features': n_feat_valid.tolist(),
    }


def create_visualizations(aggregate, output_dir):
    """Create visualization plots."""

    slopes = np.array(aggregate['slopes'])
    inv_eig = np.array(aggregate['inv_eigenvalues'])
    eig = np.array(aggregate['eigenvalues'])
    r2 = np.array(aggregate['cluster_r_squared'])
    n_feat = np.array(aggregate['cluster_n_features'])

    # Figure 1: Slope vs 1/λ scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: κ vs 1/λ
    ax = axes[0, 0]
    sc = ax.scatter(inv_eig, slopes, c=r2, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(sc, ax=ax, label='Cluster R²')

    # Add y=x reference line
    max_val = max(np.percentile(inv_eig, 95), np.percentile(slopes, 95))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='κ = 1/λ')

    # Add regression line
    reg_a = aggregate['regression_a']
    reg_b = aggregate['regression_b']
    x_line = np.linspace(0, max_val, 100)
    ax.plot(x_line, reg_a * x_line + reg_b, 'g-', linewidth=2,
            label=f'Fit: κ = {reg_a:.3f}/λ + {reg_b:.4f}')

    ax.set_xlabel('1/λ (inverse eigenvalue)', fontsize=12)
    ax.set_ylabel('κ (cluster slope)', fontsize=12)
    ax.set_title(f'Slope vs Inverse Eigenvalue\nPearson r = {aggregate["pearson_correlation"]:.3f}', fontsize=14)
    ax.legend()
    ax.set_xlim(0, np.percentile(inv_eig, 98))
    ax.set_ylim(0, np.percentile(slopes, 98))
    ax.grid(True, alpha=0.3)

    # Panel 2: κ * λ histogram (should peak at 1)
    ax = axes[0, 1]
    slope_times_lambda = slopes * eig
    ax.hist(slope_times_lambda, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Predicted (κλ = 1)')
    ax.axvline(aggregate['slope_times_lambda_mean'], color='green', linestyle='-', linewidth=2,
               label=f'Mean = {aggregate["slope_times_lambda_mean"]:.3f}')
    ax.set_xlabel('κ × λ (slope × eigenvalue)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of κ × λ\n(Conjecture: should peak at 1)', fontsize=14)
    ax.legend()
    ax.set_xlim(0, min(3, np.percentile(slope_times_lambda, 99)))
    ax.grid(True, alpha=0.3)

    # Panel 3: Log-log plot
    ax = axes[1, 0]
    valid_log = (slopes > 0) & (inv_eig > 0)
    ax.scatter(np.log10(inv_eig[valid_log]), np.log10(slopes[valid_log]),
               c=np.log10(n_feat[valid_log]), cmap='plasma', alpha=0.6, s=20)

    # Add y=x line in log space
    log_range = np.linspace(np.log10(inv_eig[valid_log].min()), np.log10(inv_eig[valid_log].max()), 100)
    ax.plot(log_range, log_range, 'r--', linewidth=2, label='κ = 1/λ')

    ax.set_xlabel('log₁₀(1/λ)', fontsize=12)
    ax.set_ylabel('log₁₀(κ)', fontsize=12)
    ax.set_title('Log-Log Plot of Slope vs Inverse Eigenvalue', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Residuals
    ax = axes[1, 1]
    predicted = reg_a * inv_eig + reg_b
    residuals = slopes - predicted
    ax.scatter(inv_eig, residuals, c=r2, cmap='viridis', alpha=0.6, s=20)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('1/λ', fontsize=12)
    ax.set_ylabel('Residual (κ - predicted)', fontsize=12)
    ax.set_title('Regression Residuals', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'slope_eigenvalue_correlation.png', dpi=150, bbox_inches='tight')
    print(f"Saved: slope_eigenvalue_correlation.png")
    plt.close()

    # Figure 2: Summary statistics by regime
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Binned analysis by eigenvalue magnitude
    ax = axes[0]
    eig_bins = np.logspace(np.log10(eig.min()), np.log10(eig.max()), 10)
    bin_means = []
    bin_stds = []
    bin_centers = []

    for i in range(len(eig_bins) - 1):
        mask = (eig >= eig_bins[i]) & (eig < eig_bins[i+1])
        if np.sum(mask) > 5:
            ratio = slopes[mask] * eig[mask]
            bin_means.append(np.mean(ratio))
            bin_stds.append(np.std(ratio))
            bin_centers.append(np.sqrt(eig_bins[i] * eig_bins[i+1]))

    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=5, capthick=2)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='κλ = 1')
    ax.set_xscale('log')
    ax.set_xlabel('Eigenvalue λ', fontsize=12)
    ax.set_ylabel('κ × λ', fontsize=12)
    ax.set_title('κ × λ vs Eigenvalue Magnitude', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Quality of linear fits
    ax = axes[1]
    ax.hist(r2, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(np.mean(r2), color='red', linestyle='-', linewidth=2, label=f'Mean R² = {np.mean(r2):.3f}')
    ax.set_xlabel('Cluster R² (quality of D vs ||W||² fit)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Linear Fit Quality', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'slope_eigenvalue_summary.png', dpi=150, bbox_inches='tight')
    print(f"Saved: slope_eigenvalue_summary.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Slope-Eigenvalue Correlation Analysis')
    parser.add_argument('--checkpoint', type=str, default='final',
                        choices=['early', 'mid', 'final', 'all'],
                        help='Which checkpoint to analyze')
    parser.add_argument('--sample', type=int, default=None,
                        help='Process only N files for testing')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3: Slope-Eigenvalue Correlation Analysis")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")

    # Get list of SVD files
    svd_files = sorted(SVD_DIR.glob('svd_n1024_m*.h5'))
    print(f"Found {len(svd_files)} SVD files")

    if args.sample:
        svd_files = svd_files[:args.sample]
        print(f"Sampling {len(svd_files)} files for testing")

    # Determine checkpoint indices
    if args.checkpoint == 'all':
        checkpoint_indices = list(CHECKPOINT_INDICES.values())
        checkpoint_names = list(CHECKPOINT_INDICES.keys())
    else:
        checkpoint_indices = [CHECKPOINT_INDICES[args.checkpoint]]
        checkpoint_names = [args.checkpoint]

    # Process for each checkpoint
    for ckpt_idx, ckpt_name in zip(checkpoint_indices, checkpoint_names):
        print(f"\n--- Analyzing checkpoint: {ckpt_name} (index {ckpt_idx}) ---")

        all_results = []

        for svd_path in tqdm(svd_files, desc=f"Processing ({ckpt_name})"):
            input_name = svd_path.stem.replace('svd_', '') + '.h5'
            input_path = INPUT_DIR / input_name

            if not input_path.exists():
                continue

            try:
                result = analyze_single_file(input_path, svd_path, ckpt_idx)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {svd_path.name}: {e}")
                continue

        print(f"Successfully processed {len(all_results)} files")

        # Aggregate and test conjecture
        print("\n--- Testing Conjecture: κ ≈ 1/λ ---")
        aggregate = aggregate_results(all_results)

        # Print results
        print(f"\nResults for {ckpt_name} checkpoint:")
        print(f"  Clusters analyzed: {aggregate['n_clusters_analyzed']}")
        print(f"  Pearson correlation (κ vs 1/λ): {aggregate['pearson_correlation']:.4f} (p={aggregate['pearson_pvalue']:.2e})")
        print(f"  Spearman correlation: {aggregate['spearman_correlation']:.4f} (p={aggregate['spearman_pvalue']:.2e})")
        print(f"  Regression: κ = {aggregate['regression_a']:.4f} × (1/λ) + {aggregate['regression_b']:.4f}")
        print(f"  Regression R²: {aggregate['regression_r_squared']:.4f}")
        print(f"  Mean κ×λ: {aggregate['slope_times_lambda_mean']:.4f} ± {aggregate['slope_times_lambda_std']:.4f}")
        print(f"  (Conjecture predicts κ×λ ≈ 1)")

        # Save results
        output_prefix = f"{ckpt_name}_" if len(checkpoint_names) > 1 else ""

        with open(OUTPUT_DIR / f'{output_prefix}correlation_results.json', 'w') as f:
            # Don't save the large arrays to JSON
            summary = {k: v for k, v in aggregate.items()
                      if k not in ['slopes', 'eigenvalues', 'inv_eigenvalues', 'cluster_r_squared', 'cluster_n_features']}
            json.dump(summary, f, indent=2)

        # Save arrays
        np.savez(OUTPUT_DIR / f'{output_prefix}correlation_data.npz',
                 slopes=np.array(aggregate['slopes']),
                 eigenvalues=np.array(aggregate['eigenvalues']),
                 inv_eigenvalues=np.array(aggregate['inv_eigenvalues']),
                 cluster_r_squared=np.array(aggregate['cluster_r_squared']),
                 cluster_n_features=np.array(aggregate['cluster_n_features']))

        # Create visualizations
        print("\n--- Creating Visualizations ---")
        create_visualizations(aggregate, OUTPUT_DIR)

        # Save detailed per-file results
        with open(OUTPUT_DIR / f'{output_prefix}per_file_results.json', 'w') as f:
            # Simplify for JSON
            simplified = []
            for r in all_results:
                simplified.append({
                    'm_hidden': r['m_hidden'],
                    'sparsity': r['sparsity'],
                    'seed': r['seed'],
                    'overall_slope': r['overall_slope'],
                    'overall_r_squared': r['overall_r_squared'],
                    'n_clusters': r['n_clusters'],
                })
            json.dump(simplified, f, indent=2)

    print("\n" + "=" * 60)
    print("Phase 3 Complete")
    print("=" * 60)
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
