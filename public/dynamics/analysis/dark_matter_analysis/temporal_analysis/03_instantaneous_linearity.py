#!/usr/bin/env python3
"""
Analysis 3: Instantaneous vs Cumulative Linearity

Compare linear fit quality using different temporal windows:
1. Cumulative R²: using all checkpoints [0, t]
2. Instantaneous slope: local derivative at each checkpoint

Key question: Is poor cumulative R² due to slope changes during training,
or intrinsic nonlinearity at each instant?
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
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/dark_matter_analysis/temporal_analysis')
PLOTS_DIR = OUTPUT_DIR / 'plots'
RESULTS_DIR = OUTPUT_DIR / 'results'


def compute_slope_vectorized(x, y):
    """Vectorized linear regression slope computation.

    Args:
        x: (window, n_features) array
        y: (window, n_features) array

    Returns:
        slopes: (n_features,) array of slopes
    """
    valid = np.isfinite(x) & np.isfinite(y) & (x > 1e-8)
    x_masked = np.where(valid, x, np.nan)
    y_masked = np.where(valid, y, np.nan)

    n_valid = np.sum(valid, axis=0)
    x_mean = np.nanmean(x_masked, axis=0)
    y_mean = np.nanmean(y_masked, axis=0)

    x_centered = x_masked - x_mean
    y_centered = y_masked - y_mean

    ss_xx = np.nansum(x_centered ** 2, axis=0)
    ss_xy = np.nansum(x_centered * y_centered, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = ss_xy / ss_xx

    # Invalidate slopes with insufficient data or zero variance
    slopes = np.where((n_valid >= 3) & (ss_xx > 1e-10), slopes, np.nan)
    return slopes


def compute_r2_vectorized(x, y):
    """Vectorized R² computation for all features at once.

    Args:
        x: (T, n_features) array of feature norms
        y: (T, n_features) array of fractional dims

    Returns:
        r2: (n_features,) array of R² values
        slopes: (n_features,) array of slopes
    """
    valid = np.isfinite(x) & np.isfinite(y) & (x > 1e-8)
    x_masked = np.where(valid, x, np.nan)
    y_masked = np.where(valid, y, np.nan)

    n_valid = np.sum(valid, axis=0)
    x_mean = np.nanmean(x_masked, axis=0)
    y_mean = np.nanmean(y_masked, axis=0)

    x_centered = x_masked - x_mean
    y_centered = y_masked - y_mean

    ss_xx = np.nansum(x_centered ** 2, axis=0)
    ss_yy = np.nansum(y_centered ** 2, axis=0)
    ss_xy = np.nansum(x_centered * y_centered, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        r = ss_xy / np.sqrt(ss_xx * ss_yy)
        r2 = r ** 2
        slopes = ss_xy / ss_xx

    # Invalidate results with insufficient data
    r2 = np.where((n_valid >= 5) & (ss_xx > 1e-10), r2, np.nan)
    slopes = np.where((n_valid >= 5) & (ss_xx > 1e-10), slopes, np.nan)

    return r2, slopes


def compute_instantaneous_slope(feature_norms, fractional_dims, window=5):
    """
    Compute instantaneous slope dD/d||W||² at each checkpoint using a sliding window.
    Vectorized implementation for speed.

    Returns:
        slopes: (T, n) array of instantaneous slopes
    """
    n_checkpoints, n_features = feature_norms.shape
    slopes = np.full((n_checkpoints, n_features), np.nan)

    half_window = window // 2
    for t in range(n_checkpoints):
        start = max(0, t - half_window)
        end = min(n_checkpoints, t + half_window + 1)

        if end - start < 3:
            continue

        x = feature_norms[start:end, :]
        y = fractional_dims[start:end, :]
        slopes[t, :] = compute_slope_vectorized(x, y)

    return slopes


def analyze_file(filepath):
    """Analyze instantaneous vs cumulative linearity for one experiment."""
    with h5py.File(filepath, 'r') as f:
        feature_norms = f['feature_norms'][:]  # (T, n)
        fractional_dims = f['fractional_dims'][:]  # (T, n)
        checkpoint_steps = f['checkpoint_steps'][:]
        sparsity = float(f.attrs['sparsity'])
        m_hidden = int(f.attrs['m_hidden'])
        seed = int(f.attrs['seed'])

    n_checkpoints, n_features = feature_norms.shape

    # Compute instantaneous slopes
    inst_slopes = compute_instantaneous_slope(feature_norms, fractional_dims, window=5)

    # Compute slope statistics for each feature
    slope_means = np.nanmean(inst_slopes, axis=0)  # (n,)
    slope_stds = np.nanstd(inst_slopes, axis=0)  # (n,)
    slope_cv = slope_stds / np.abs(slope_means + 1e-10)  # Coefficient of variation

    # Compute final cumulative R² - vectorized
    final_r2, final_slope = compute_r2_vectorized(feature_norms, fractional_dims)

    # Identify features by slope stability
    # Stable: low CV, slope doesn't change much
    # Unstable: high CV, slope varies significantly

    valid_cv = np.isfinite(slope_cv)
    if np.sum(valid_cv) > 0:
        cv_median = np.nanmedian(slope_cv)
        cv_75 = np.nanpercentile(slope_cv[valid_cv], 75)
    else:
        cv_median, cv_75 = np.nan, np.nan

    stable_threshold = cv_75 if np.isfinite(cv_75) else 1.0
    stable_features = slope_cv < stable_threshold
    unstable_features = slope_cv >= stable_threshold

    # Mean R² for stable vs unstable features
    stable_r2 = np.nanmean(final_r2[stable_features]) if np.sum(stable_features) > 0 else np.nan
    unstable_r2 = np.nanmean(final_r2[unstable_features]) if np.sum(unstable_features) > 0 else np.nan

    # Slope convergence: does the slope stabilize in late training?
    late_slopes = inst_slopes[-10:, :]  # Last 10 checkpoints
    early_slopes = inst_slopes[5:15, :]  # Early training
    late_slope_std = np.nanstd(late_slopes, axis=0)
    early_slope_std = np.nanstd(early_slopes, axis=0)

    convergence_ratio = late_slope_std / (early_slope_std + 1e-10)
    converged_features = convergence_ratio < 0.5  # Slope variance reduced by half

    # Sample feature trajectories for visualization
    # Select diverse features: high/low R², stable/unstable slope
    sample_indices = []
    valid_final = np.isfinite(final_r2)
    if np.sum(valid_final) > 0:
        r2_sorted = np.argsort(final_r2[valid_final])
        indices_sorted = np.where(valid_final)[0][r2_sorted]

        # Low R² features
        if len(indices_sorted) > 5:
            sample_indices.extend(indices_sorted[:5].tolist())
        # High R² features
        if len(indices_sorted) > 5:
            sample_indices.extend(indices_sorted[-5:].tolist())

    return {
        'sparsity': sparsity,
        'm_hidden': m_hidden,
        'seed': seed,
        'checkpoint_steps': checkpoint_steps.tolist(),
        'mean_slope_cv': float(np.nanmean(slope_cv)),
        'median_slope_cv': float(cv_median) if np.isfinite(cv_median) else None,
        'stable_features_r2': float(stable_r2) if np.isfinite(stable_r2) else None,
        'unstable_features_r2': float(unstable_r2) if np.isfinite(unstable_r2) else None,
        'fraction_stable': float(np.mean(stable_features[np.isfinite(slope_cv)])) if np.sum(np.isfinite(slope_cv)) > 0 else None,
        'fraction_converged': float(np.nanmean(converged_features)),
        'mean_convergence_ratio': float(np.nanmean(convergence_ratio)),
        'mean_final_r2': float(np.nanmean(final_r2)),
        'mean_final_slope': float(np.nanmean(final_slope)),
        'n_features': n_features,
        # Store sample trajectories
        'sample_indices': sample_indices,
        'sample_slopes': inst_slopes[:, sample_indices].tolist() if sample_indices else [],
        'sample_norms': feature_norms[:, sample_indices].tolist() if sample_indices else [],
        'sample_dims': fractional_dims[:, sample_indices].tolist() if sample_indices else [],
        'sample_r2': final_r2[sample_indices].tolist() if sample_indices else [],
    }


def process_file_wrapper(filepath):
    """Wrapper for multiprocessing."""
    try:
        return analyze_file(filepath)
    except Exception as e:
        print(f"Error processing {filepath.name}: {e}")
        return None


def aggregate_results(all_results):
    """Aggregate results by sparsity bucket."""
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
                aggregated[bucket]['mean_slope_cv'].append(r['mean_slope_cv'])
                aggregated[bucket]['stable_features_r2'].append(r['stable_features_r2'])
                aggregated[bucket]['unstable_features_r2'].append(r['unstable_features_r2'])
                aggregated[bucket]['fraction_stable'].append(r['fraction_stable'])
                aggregated[bucket]['fraction_converged'].append(r['fraction_converged'])
                aggregated[bucket]['mean_convergence_ratio'].append(r['mean_convergence_ratio'])
                break

    summary = {}
    for bucket, data in aggregated.items():
        if len(data['mean_slope_cv']) == 0:
            continue

        summary[bucket] = {
            'n_experiments': len(data['mean_slope_cv']),
            'mean_slope_cv': float(np.nanmean(data['mean_slope_cv'])),
            'mean_stable_r2': float(np.nanmean([x for x in data['stable_features_r2'] if x is not None])) if any(x is not None for x in data['stable_features_r2']) else None,
            'mean_unstable_r2': float(np.nanmean([x for x in data['unstable_features_r2'] if x is not None])) if any(x is not None for x in data['unstable_features_r2']) else None,
            'mean_fraction_stable': float(np.nanmean([x for x in data['fraction_stable'] if x is not None])) if any(x is not None for x in data['fraction_stable']) else None,
            'mean_fraction_converged': float(np.nanmean(data['fraction_converged'])),
            'mean_convergence_ratio': float(np.nanmean(data['mean_convergence_ratio'])),
        }

    return summary


def create_visualizations(summary, all_results, output_dir):
    """Create visualization plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {'low': 'blue', 'medium': 'green', 'high': 'orange', 'extreme': 'red'}

    # Panel 1: Slope stability comparison
    ax = axes[0, 0]
    buckets = list(summary.keys())
    x = np.arange(len(buckets))
    width = 0.35

    stable_r2 = [summary[b].get('mean_stable_r2', 0) or 0 for b in buckets]
    unstable_r2 = [summary[b].get('mean_unstable_r2', 0) or 0 for b in buckets]

    ax.bar(x - width/2, stable_r2, width, label='Stable Slope Features', color='steelblue')
    ax.bar(x + width/2, unstable_r2, width, label='Unstable Slope Features', color='coral')

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel('Sparsity Bucket', fontsize=12)
    ax.set_ylabel('Mean R²', fontsize=12)
    ax.set_title('R² by Slope Stability Category', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5, label='R²=0.9 threshold')

    # Panel 2: Convergence metrics
    ax = axes[0, 1]
    convergence = [summary[b].get('mean_fraction_converged', 0) * 100 for b in buckets]
    cv = [summary[b].get('mean_slope_cv', 0) for b in buckets]

    ax2 = ax.twinx()
    bars1 = ax.bar(x - 0.2, convergence, 0.4, label='% Converged', color='teal', alpha=0.7)
    bars2 = ax2.bar(x + 0.2, cv, 0.4, label='Mean Slope CV', color='purple', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel('Sparsity Bucket', fontsize=12)
    ax.set_ylabel('% Converged (slope stabilized)', fontsize=12, color='teal')
    ax2.set_ylabel('Coefficient of Variation', fontsize=12, color='purple')
    ax.set_title('Slope Convergence Metrics', fontsize=14)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Panel 3: Example trajectories (slope over time)
    ax = axes[1, 0]

    # Collect sample trajectories from high sparsity experiments
    high_sparsity_results = [r for r in all_results if 0.7 <= r['sparsity'] < 0.95]
    if high_sparsity_results and high_sparsity_results[0]['sample_slopes']:
        sample_result = high_sparsity_results[0]
        steps = np.array(sample_result['checkpoint_steps'])
        sample_slopes = np.array(sample_result['sample_slopes'])
        sample_r2 = sample_result['sample_r2']

        for i in range(min(5, len(sample_slopes[0]))):
            color = 'green' if sample_r2[i] >= 0.9 else 'red'
            alpha = 0.8 if sample_r2[i] >= 0.9 else 0.5
            label = f'R²={sample_r2[i]:.2f}' if i < 3 else None
            ax.plot(steps, sample_slopes[:, i], color=color, alpha=alpha,
                   linewidth=1.5, label=label)

        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Instantaneous Slope κ(t)', fontsize=12)
        ax.set_title('Sample Feature Slope Trajectories (High Sparsity)', fontsize=14)
        ax.legend(title='Green=Well-behaved, Red=Dark Matter')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No sample data available', transform=ax.transAxes,
               ha='center', va='center')
        ax.set_title('Sample Feature Slope Trajectories', fontsize=14)

    # Panel 4: Phase trajectories
    ax = axes[1, 1]

    if high_sparsity_results and high_sparsity_results[0]['sample_norms']:
        sample_result = high_sparsity_results[0]
        sample_norms = np.array(sample_result['sample_norms'])
        sample_dims = np.array(sample_result['sample_dims'])
        sample_r2 = sample_result['sample_r2']

        for i in range(min(5, len(sample_norms[0]))):
            color = 'green' if sample_r2[i] >= 0.9 else 'red'
            alpha = 0.8 if sample_r2[i] >= 0.9 else 0.5
            ax.plot(sample_norms[:, i], sample_dims[:, i], color=color,
                   alpha=alpha, linewidth=1.5)
            # Mark start and end
            ax.scatter(sample_norms[0, i], sample_dims[0, i], color=color,
                      marker='o', s=30, zorder=5)
            ax.scatter(sample_norms[-1, i], sample_dims[-1, i], color=color,
                      marker='s', s=30, zorder=5)

        ax.set_xlabel('||W_i||²', fontsize=12)
        ax.set_ylabel('D_i', fontsize=12)
        ax.set_title('Phase Space Trajectories (○=start, □=end)', fontsize=14)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No sample data available', transform=ax.transAxes,
               ha='center', va='center')
        ax.set_title('Phase Space Trajectories', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'instantaneous_linearity.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'instantaneous_linearity.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Instantaneous Linearity Analysis')
    parser.add_argument('--sample', type=int, default=None, help='Process only N files for testing')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Analysis 3: Instantaneous vs Cumulative Linearity")
    print("=" * 60)

    files = sorted(INPUT_DIR.glob('n1024_m*.h5'))
    print(f"Found {len(files)} experiment files")

    if args.sample:
        files = files[:args.sample]
        print(f"Sampling {len(files)} files for testing")

    # Determine number of workers
    n_workers = args.workers if args.workers else min(16, cpu_count())
    print(f"Using {n_workers} parallel workers")

    # Process files in parallel
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_file_wrapper, files),
            total=len(files),
            desc="Processing files"
        ))

    all_results = [r for r in results if r is not None]
    print(f"Successfully processed {len(all_results)} files")

    summary = aggregate_results(all_results)

    print("\n--- Summary by Sparsity Bucket ---")
    for bucket, data in summary.items():
        print(f"\n{bucket.upper()} sparsity:")
        print(f"  Experiments: {data['n_experiments']}")
        print(f"  Mean slope CV: {data['mean_slope_cv']:.3f}")
        if data['mean_stable_r2']:
            print(f"  Stable features R²: {data['mean_stable_r2']:.3f}")
        if data['mean_unstable_r2']:
            print(f"  Unstable features R²: {data['mean_unstable_r2']:.3f}")
        if data['mean_fraction_stable']:
            print(f"  Fraction stable: {data['mean_fraction_stable']*100:.1f}%")
        print(f"  Fraction converged: {data['mean_fraction_converged']*100:.1f}%")

    # Save results (without large trajectory data)
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items()
                  if k not in ['sample_slopes', 'sample_norms', 'sample_dims']}
        save_results.append(save_r)

    with open(RESULTS_DIR / 'instantaneous_linearity.json', 'w') as f:
        json.dump({
            'summary_by_sparsity': summary,
            'per_file_results': save_results[:100] if len(save_results) > 100 else save_results,
        }, f, indent=2)
    print(f"\nSaved results to: {RESULTS_DIR / 'instantaneous_linearity.json'}")

    print("\n--- Creating Visualizations ---")
    create_visualizations(summary, all_results, PLOTS_DIR)

    print("\n" + "=" * 60)
    print("Analysis 3 Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
