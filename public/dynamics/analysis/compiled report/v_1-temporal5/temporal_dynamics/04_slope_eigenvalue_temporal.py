#!/usr/bin/env python3
"""
Analysis 4: Slope-Eigenvalue Relationship Over Time

Track how well the κ ≈ 1/λ conjecture holds at each checkpoint.
This extends the SVD analysis to show temporal evolution of the spectral relationship.

Key outputs:
- κλ distribution at early, mid, late training
- Mean(κλ) and std(κλ) vs training step
- Regime-specific analysis (λ > 1 vs λ < 1)
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import os


INPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
SVD_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/svd_results')
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/dark_matter_analysis/temporal_analysis')
PLOTS_DIR = OUTPUT_DIR / 'plots'
RESULTS_DIR = OUTPUT_DIR / 'results'


def compute_slope_r2_vectorized(x, y):
    """Vectorized linear regression for multiple features.

    Args:
        x: (n,) array
        y: (n,) array

    Returns:
        slope, r2
    """
    valid = np.isfinite(x) & np.isfinite(y) & (x > 1e-8)
    if np.sum(valid) < 3:
        return np.nan, np.nan

    x_v = x[valid]
    y_v = y[valid]

    if np.std(x_v) < 1e-10:
        return np.nan, np.nan

    x_mean = np.mean(x_v)
    y_mean = np.mean(y_v)

    x_centered = x_v - x_mean
    y_centered = y_v - y_mean

    ss_xx = np.sum(x_centered ** 2)
    ss_yy = np.sum(y_centered ** 2)
    ss_xy = np.sum(x_centered * y_centered)

    if ss_xx < 1e-10:
        return np.nan, np.nan

    slope = ss_xy / ss_xx
    r = ss_xy / np.sqrt(ss_xx * ss_yy) if ss_yy > 1e-10 else 0
    r2 = r ** 2

    return slope, r2


def compute_cluster_slopes_at_checkpoint(weights, fractional_dims, feature_norms, U, eigenvalues, min_cluster_size=10):
    """
    Compute the slope κ for each eigenspace cluster at a single checkpoint.

    Returns:
        cluster_data: list of dicts with slope, eigenvalue, etc. for each cluster
    """
    m, n = weights.shape

    # Compute eigenspace assignments
    projections = U.T @ weights  # (m, n)
    proj_squared = projections ** 2
    norms_sq = np.maximum(np.sum(weights ** 2, axis=0), 1e-10)
    normalized_proj = proj_squared / norms_sq
    cluster_assignments = np.argmax(normalized_proj, axis=0)

    cluster_data = []
    unique_clusters = np.unique(cluster_assignments)

    for k in unique_clusters:
        mask = cluster_assignments == k
        n_features = np.sum(mask)

        if n_features < min_cluster_size:
            continue

        x = feature_norms[mask]
        y = fractional_dims[mask]

        slope, r2 = compute_slope_r2_vectorized(x, y)

        if np.isnan(slope) or np.isnan(r2):
            continue

        lambda_k = eigenvalues[k]
        if lambda_k > 1e-10 and slope > 0:
            kappa_lambda = slope * lambda_k
            cluster_data.append({
                'cluster': int(k),
                'n_features': int(n_features),
                'slope': float(slope),
                'eigenvalue': float(lambda_k),
                'kappa_lambda': float(kappa_lambda),
                'r_squared': float(r2),
                'regime': 'large' if lambda_k > 1 else 'small',
            })

    return cluster_data


def analyze_file(input_path, svd_path):
    """Analyze slope-eigenvalue relationship across all checkpoints."""

    with h5py.File(input_path, 'r') as f:
        weights_all = f['weights'][:]  # (T, m, n)
        feature_norms_all = f['feature_norms'][:]  # (T, n)
        fractional_dims_all = f['fractional_dims'][:]  # (T, n)
        checkpoint_steps = f['checkpoint_steps'][:]
        sparsity = float(f.attrs['sparsity'])
        m_hidden = int(f.attrs['m_hidden'])
        seed = int(f.attrs['seed'])

    with h5py.File(svd_path, 'r') as f:
        U_all = f['U'][:]  # (T, m, m)
        eigenvalues_all = f['eigenvalues'][:]  # (T, m)

    n_checkpoints = weights_all.shape[0]

    # Analyze at each checkpoint
    temporal_stats = []
    for t in range(n_checkpoints):
        cluster_data = compute_cluster_slopes_at_checkpoint(
            weights_all[t], fractional_dims_all[t], feature_norms_all[t],
            U_all[t], eigenvalues_all[t]
        )

        if len(cluster_data) == 0:
            temporal_stats.append({
                'checkpoint': int(checkpoint_steps[t]),
                'n_clusters': 0,
                'mean_kappa_lambda': np.nan,
                'std_kappa_lambda': np.nan,
                'mean_kappa_lambda_large': np.nan,
                'mean_kappa_lambda_small': np.nan,
            })
            continue

        kappa_lambdas = [c['kappa_lambda'] for c in cluster_data]
        large_kl = [c['kappa_lambda'] for c in cluster_data if c['regime'] == 'large']
        small_kl = [c['kappa_lambda'] for c in cluster_data if c['regime'] == 'small']

        temporal_stats.append({
            'checkpoint': int(checkpoint_steps[t]),
            'n_clusters': len(cluster_data),
            'mean_kappa_lambda': float(np.mean(kappa_lambdas)),
            'std_kappa_lambda': float(np.std(kappa_lambdas)),
            'mean_kappa_lambda_large': float(np.mean(large_kl)) if large_kl else np.nan,
            'std_kappa_lambda_large': float(np.std(large_kl)) if large_kl else np.nan,
            'mean_kappa_lambda_small': float(np.mean(small_kl)) if small_kl else np.nan,
            'std_kappa_lambda_small': float(np.std(small_kl)) if small_kl else np.nan,
            'n_large_clusters': len(large_kl),
            'n_small_clusters': len(small_kl),
        })

    # Detailed analysis at key checkpoints
    early_idx = 10
    mid_idx = len(checkpoint_steps) // 2
    final_idx = -1

    early_clusters = compute_cluster_slopes_at_checkpoint(
        weights_all[early_idx], fractional_dims_all[early_idx], feature_norms_all[early_idx],
        U_all[early_idx], eigenvalues_all[early_idx]
    )
    mid_clusters = compute_cluster_slopes_at_checkpoint(
        weights_all[mid_idx], fractional_dims_all[mid_idx], feature_norms_all[mid_idx],
        U_all[mid_idx], eigenvalues_all[mid_idx]
    )
    final_clusters = compute_cluster_slopes_at_checkpoint(
        weights_all[final_idx], fractional_dims_all[final_idx], feature_norms_all[final_idx],
        U_all[final_idx], eigenvalues_all[final_idx]
    )

    return {
        'sparsity': sparsity,
        'm_hidden': m_hidden,
        'seed': seed,
        'temporal_stats': temporal_stats,
        'early_kappa_lambdas': [c['kappa_lambda'] for c in early_clusters],
        'mid_kappa_lambdas': [c['kappa_lambda'] for c in mid_clusters],
        'final_kappa_lambdas': [c['kappa_lambda'] for c in final_clusters],
    }


def process_file_pair(args):
    """Wrapper for multiprocessing with file pairs."""
    input_path, svd_path = args
    try:
        return analyze_file(input_path, svd_path)
    except Exception as e:
        print(f"Error processing {svd_path.name}: {e}")
        return None


def aggregate_results(all_results):
    """Aggregate temporal statistics."""

    sparsity_buckets = {
        'low': (0.0, 0.3),
        'medium': (0.3, 0.7),
        'high': (0.7, 0.95),
        'extreme': (0.95, 1.0),
    }

    aggregated = {bucket: defaultdict(list) for bucket in sparsity_buckets}

    for r in all_results:
        for bucket, (low, high) in sparsity_buckets.items():
            if low <= r['sparsity'] < high:
                aggregated[bucket]['temporal_stats'].append(r['temporal_stats'])
                aggregated[bucket]['early_kl'].extend(r['early_kappa_lambdas'])
                aggregated[bucket]['mid_kl'].extend(r['mid_kappa_lambdas'])
                aggregated[bucket]['final_kl'].extend(r['final_kappa_lambdas'])
                break

    summary = {}
    for bucket, data in aggregated.items():
        if len(data['temporal_stats']) == 0:
            continue

        # Stack temporal stats
        n_experiments = len(data['temporal_stats'])
        n_checkpoints = len(data['temporal_stats'][0])

        mean_kl_over_time = np.zeros(n_checkpoints)
        std_kl_over_time = np.zeros(n_checkpoints)
        mean_kl_large_over_time = np.zeros(n_checkpoints)
        mean_kl_small_over_time = np.zeros(n_checkpoints)

        for t in range(n_checkpoints):
            kls = [exp[t]['mean_kappa_lambda'] for exp in data['temporal_stats']
                   if not np.isnan(exp[t]['mean_kappa_lambda'])]
            kls_large = [exp[t]['mean_kappa_lambda_large'] for exp in data['temporal_stats']
                        if not np.isnan(exp[t].get('mean_kappa_lambda_large', np.nan))]
            kls_small = [exp[t]['mean_kappa_lambda_small'] for exp in data['temporal_stats']
                        if not np.isnan(exp[t].get('mean_kappa_lambda_small', np.nan))]

            mean_kl_over_time[t] = np.mean(kls) if kls else np.nan
            std_kl_over_time[t] = np.std(kls) if kls else np.nan
            mean_kl_large_over_time[t] = np.mean(kls_large) if kls_large else np.nan
            mean_kl_small_over_time[t] = np.mean(kls_small) if kls_small else np.nan

        checkpoints = [exp[0]['checkpoint'] for exp in data['temporal_stats']]

        summary[bucket] = {
            'n_experiments': n_experiments,
            'checkpoints': [data['temporal_stats'][0][t]['checkpoint'] for t in range(n_checkpoints)],
            'mean_kappa_lambda': mean_kl_over_time.tolist(),
            'std_kappa_lambda': std_kl_over_time.tolist(),
            'mean_kappa_lambda_large': mean_kl_large_over_time.tolist(),
            'mean_kappa_lambda_small': mean_kl_small_over_time.tolist(),
            'early_kl_dist': {
                'mean': float(np.mean(data['early_kl'])) if data['early_kl'] else np.nan,
                'std': float(np.std(data['early_kl'])) if data['early_kl'] else np.nan,
            },
            'mid_kl_dist': {
                'mean': float(np.mean(data['mid_kl'])) if data['mid_kl'] else np.nan,
                'std': float(np.std(data['mid_kl'])) if data['mid_kl'] else np.nan,
            },
            'final_kl_dist': {
                'mean': float(np.mean(data['final_kl'])) if data['final_kl'] else np.nan,
                'std': float(np.std(data['final_kl'])) if data['final_kl'] else np.nan,
            },
        }

    return summary, aggregated


def create_visualizations(summary, aggregated, output_dir):
    """Create visualization plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {'low': 'blue', 'medium': 'green', 'high': 'orange', 'extreme': 'red'}

    # Panel 1: κλ evolution over training
    ax = axes[0, 0]
    for bucket, data in summary.items():
        steps = np.array(data['checkpoints'])
        mean_kl = np.array(data['mean_kappa_lambda'])
        std_kl = np.array(data['std_kappa_lambda'])

        valid = np.isfinite(mean_kl)
        ax.plot(steps[valid], mean_kl[valid], color=colors[bucket],
                label=bucket, linewidth=2)
        ax.fill_between(steps[valid],
                       (mean_kl - std_kl)[valid],
                       (mean_kl + std_kl)[valid],
                       color=colors[bucket], alpha=0.2)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=2, label='κλ = 1 (conjecture)')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Mean κλ', fontsize=12)
    ax.set_title('Slope × Eigenvalue Over Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2)

    # Panel 2: λ>1 vs λ<1 regimes
    ax = axes[0, 1]
    for bucket, data in summary.items():
        steps = np.array(data['checkpoints'])
        mean_large = np.array(data['mean_kappa_lambda_large'])
        mean_small = np.array(data['mean_kappa_lambda_small'])

        valid_large = np.isfinite(mean_large)
        valid_small = np.isfinite(mean_small)

        if np.sum(valid_large) > 0:
            ax.plot(steps[valid_large], mean_large[valid_large],
                   color=colors[bucket], linestyle='-', linewidth=2, label=f'{bucket} (λ>1)')
        if np.sum(valid_small) > 0:
            ax.plot(steps[valid_small], mean_small[valid_small],
                   color=colors[bucket], linestyle='--', linewidth=2, label=f'{bucket} (λ≤1)')

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=2)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Mean κλ', fontsize=12)
    ax.set_title('κλ by Eigenvalue Regime (solid=λ>1, dashed=λ≤1)', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2)

    # Panel 3: κλ distributions at key checkpoints (high sparsity)
    ax = axes[1, 0]
    if 'high' in aggregated and aggregated['high']['early_kl']:
        early = aggregated['high']['early_kl']
        mid = aggregated['high']['mid_kl']
        final = aggregated['high']['final_kl']

        bins = np.linspace(0, 2, 40)
        ax.hist(early, bins=bins, alpha=0.5, label=f'Early (μ={np.mean(early):.2f})', color='blue', density=True)
        ax.hist(mid, bins=bins, alpha=0.5, label=f'Mid (μ={np.mean(mid):.2f})', color='green', density=True)
        ax.hist(final, bins=bins, alpha=0.5, label=f'Final (μ={np.mean(final):.2f})', color='red', density=True)

        ax.axvline(1.0, color='black', linestyle='--', linewidth=2, label='κλ = 1')
        ax.set_xlabel('κλ', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('κλ Distribution Evolution (High Sparsity)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No high sparsity data', transform=ax.transAxes, ha='center')

    # Panel 4: Summary statistics by regime
    ax = axes[1, 1]
    buckets = list(summary.keys())
    x = np.arange(len(buckets))
    width = 0.25

    final_means = [summary[b]['final_kl_dist']['mean'] for b in buckets]
    final_stds = [summary[b]['final_kl_dist']['std'] for b in buckets]

    # Distance from 1.0
    distances = [abs(m - 1.0) if np.isfinite(m) else np.nan for m in final_means]

    ax.bar(x, final_means, width*2, yerr=final_stds, capsize=5, color='steelblue', alpha=0.7)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Target κλ = 1')

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel('Sparsity Bucket', fontsize=12)
    ax.set_ylabel('Final κλ', fontsize=12)
    ax.set_title('Final κλ by Sparsity Regime', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'slope_eigenvalue_temporal.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'slope_eigenvalue_temporal.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Slope-Eigenvalue Temporal Analysis')
    parser.add_argument('--sample', type=int, default=None, help='Process only N files for testing')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Analysis 4: Slope-Eigenvalue Relationship Over Time")
    print("=" * 60)

    svd_files = sorted(SVD_DIR.glob('svd_n1024_m*.h5'))
    print(f"Found {len(svd_files)} SVD files")

    if args.sample:
        svd_files = svd_files[:args.sample]
        print(f"Sampling {len(svd_files)} files for testing")

    # Build list of (input_path, svd_path) pairs
    file_pairs = []
    for svd_path in svd_files:
        input_name = svd_path.stem.replace('svd_', '') + '.h5'
        input_path = INPUT_DIR / input_name
        if input_path.exists():
            file_pairs.append((input_path, svd_path))

    print(f"Found {len(file_pairs)} matching file pairs")

    # Determine number of workers
    n_workers = args.workers if args.workers else min(16, cpu_count())
    print(f"Using {n_workers} parallel workers")

    # Process files in parallel
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_file_pair, file_pairs),
            total=len(file_pairs),
            desc="Processing files"
        ))

    all_results = [r for r in results if r is not None]
    print(f"Successfully processed {len(all_results)} files")

    summary, aggregated = aggregate_results(all_results)

    print("\n--- Summary by Sparsity Bucket ---")
    for bucket, data in summary.items():
        print(f"\n{bucket.upper()} sparsity:")
        print(f"  Experiments: {data['n_experiments']}")
        print(f"  Final mean κλ: {data['final_kl_dist']['mean']:.3f} ± {data['final_kl_dist']['std']:.3f}")
        print(f"  Early mean κλ: {data['early_kl_dist']['mean']:.3f} ± {data['early_kl_dist']['std']:.3f}")

    # Save results (simplified)
    save_summary = {}
    for bucket, data in summary.items():
        save_summary[bucket] = {
            'n_experiments': data['n_experiments'],
            'checkpoints': data['checkpoints'],
            'mean_kappa_lambda': data['mean_kappa_lambda'],
            'final_kl_dist': data['final_kl_dist'],
            'early_kl_dist': data['early_kl_dist'],
        }

    with open(RESULTS_DIR / 'slope_eigenvalue_temporal.json', 'w') as f:
        json.dump(save_summary, f, indent=2)
    print(f"\nSaved results to: {RESULTS_DIR / 'slope_eigenvalue_temporal.json'}")

    print("\n--- Creating Visualizations ---")
    create_visualizations(summary, aggregated, PLOTS_DIR)

    print("\n" + "=" * 60)
    print("Analysis 4 Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
