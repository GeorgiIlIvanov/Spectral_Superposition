#!/usr/bin/env python3
"""
Stratified Linearity Analysis: Feature-Level D_i vs ||W_i||² Trajectories

This script implements the unbiased analysis protocol to test whether the linear
scaling law D_i ∝ ||W_i||² holds universally for superposition features.

Protocol:
1. Primary Stratification by Sparsity (S):
   - Low: S ∈ [0.0, 0.2] - expect mostly orthogonal features
   - High: S ∈ [0.9, 1.0] - expect mostly superposition features
   - Extreme: S = 0.99 - the critical regime

2. Secondary Stratification by Feature Dimensionality:
   - Filter for D_i < 0.5 at final checkpoint ("Deep Superposition")

3. Feature-Level Linearity Test:
   - For each feature, fit D_i(t) vs ||W_i(t)||² across ALL training checkpoints
   - Calculate R² for every single feature trajectory

4. Verification:
   - Generate R² histograms to test if linearity is universal
   - Identify "dark matter" features (R² < 0.9) that defy prediction

Author: Analysis Protocol Implementation
Date: 2025-01-21
"""

import h5py
import numpy as np
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Configuration
INPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/stratified_linearity')

# Sparsity bucket definitions
SPARSITY_BUCKETS = {
    'low': (0.0, 0.2),       # Expect mostly orthogonal features
    'high': (0.9, 1.0),      # Expect mostly superposition features
    'extreme': (0.99, 1.0),  # The critical regime (only S=0.99)
}

# Deep superposition threshold
DEEP_SUPERPOSITION_THRESHOLD = 0.5  # D_i < 0.5 at final checkpoint


def get_sparsity_bucket(sparsity):
    """Determine which sparsity bucket a file belongs to."""
    buckets = []
    for name, (low, high) in SPARSITY_BUCKETS.items():
        if name == 'extreme':
            # Only exact 0.99 for extreme
            if np.isclose(sparsity, 0.99, atol=0.001):
                buckets.append(name)
        else:
            if low <= sparsity <= high:
                buckets.append(name)
    return buckets


def analyze_feature_trajectory(feature_norms, fractional_dims):
    """
    Analyze a single feature's trajectory: D_i(t) vs ||W_i(t)||²

    Args:
        feature_norms: (T,) - ||W_i||² at each checkpoint
        fractional_dims: (T,) - D_i at each checkpoint

    Returns:
        dict with slope, intercept, r_squared, and diagnostics
    """
    # Filter invalid values
    valid = (np.isfinite(feature_norms) &
             np.isfinite(fractional_dims) &
             (feature_norms > 1e-8))

    n_valid = np.sum(valid)

    if n_valid < 5:  # Need at least 5 points for meaningful fit
        return {
            'slope': np.nan,
            'intercept': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'std_err': np.nan,
            'n_points': int(n_valid),
            'valid': False,
            'norm_range': (0, 0),
            'dim_range': (0, 0),
        }

    x = feature_norms[valid]
    y = fractional_dims[valid]

    # Linear regression: D_i = slope * ||W_i||² + intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value ** 2),
        'p_value': float(p_value),
        'std_err': float(std_err),
        'n_points': int(n_valid),
        'valid': True,
        'norm_range': (float(x.min()), float(x.max())),
        'dim_range': (float(y.min()), float(y.max())),
    }


def process_single_file(filepath):
    """
    Process a single checkpoint file, extracting per-feature trajectory statistics.

    Returns:
        dict containing file metadata and per-feature analysis results
    """
    with h5py.File(filepath, 'r') as f:
        # Metadata
        m_hidden = int(f.attrs['m_hidden'])
        sparsity = float(f.attrs['sparsity'])
        seed = int(f.attrs['seed'])
        n_features = int(f.attrs['n_features'])

        # Trajectory data: shape (T, n_features) where T = 56 checkpoints
        feature_norms = f['feature_norms'][:]  # (T, 1024)
        fractional_dims = f['fractional_dims'][:]  # (T, 1024)
        checkpoint_steps = f['checkpoint_steps'][:]  # (T,)

    T, n = feature_norms.shape

    # Get final checkpoint dimensionalities for filtering
    final_dims = fractional_dims[-1, :]  # (1024,)
    final_norms = feature_norms[-1, :]  # (1024,)

    # Analyze each feature's trajectory
    feature_results = []

    for i in range(n):
        # Extract this feature's trajectory
        norms_traj = feature_norms[:, i]  # (T,)
        dims_traj = fractional_dims[:, i]  # (T,)

        # Analyze trajectory linearity
        traj_stats = analyze_feature_trajectory(norms_traj, dims_traj)

        feature_results.append({
            'feature_idx': i,
            'final_dim': float(final_dims[i]),
            'final_norm': float(final_norms[i]),
            'is_deep_superposition': final_dims[i] < DEEP_SUPERPOSITION_THRESHOLD,
            **traj_stats
        })

    return {
        'm_hidden': m_hidden,
        'sparsity': sparsity,
        'seed': seed,
        'n_features': n_features,
        'n_checkpoints': T,
        'checkpoint_steps': checkpoint_steps.tolist(),
        'features': feature_results,
        'filename': filepath.name,
    }


def aggregate_by_bucket(all_file_results, bucket_name, bucket_range):
    """
    Aggregate results for a specific sparsity bucket.

    Returns comprehensive statistics for the bucket.
    """
    # Filter files that belong to this bucket
    bucket_files = []
    for result in all_file_results:
        if bucket_name == 'extreme':
            if np.isclose(result['sparsity'], 0.99, atol=0.001):
                bucket_files.append(result)
        else:
            low, high = bucket_range
            if low <= result['sparsity'] <= high:
                bucket_files.append(result)

    if not bucket_files:
        return None

    # Collect all features from this bucket
    all_features = []
    deep_superposition_features = []

    for file_result in bucket_files:
        for feat in file_result['features']:
            feat_with_meta = {
                **feat,
                'm_hidden': file_result['m_hidden'],
                'sparsity': file_result['sparsity'],
                'seed': file_result['seed'],
            }
            all_features.append(feat_with_meta)

            if feat['is_deep_superposition'] and feat['valid']:
                deep_superposition_features.append(feat_with_meta)

    # Compute statistics for ALL features
    all_r2 = np.array([f['r_squared'] for f in all_features if f['valid']])
    all_slopes = np.array([f['slope'] for f in all_features if f['valid']])

    # Compute statistics for DEEP SUPERPOSITION features only
    deep_r2 = np.array([f['r_squared'] for f in deep_superposition_features])
    deep_slopes = np.array([f['slope'] for f in deep_superposition_features])
    deep_final_dims = np.array([f['final_dim'] for f in deep_superposition_features])

    # Key metrics requested in the protocol
    if len(deep_r2) > 0:
        median_r2_deep = float(np.median(deep_r2))
        pct_below_0_9 = float(np.mean(deep_r2 < 0.9) * 100)
        pct_below_0_8 = float(np.mean(deep_r2 < 0.8) * 100)
        pct_below_0_7 = float(np.mean(deep_r2 < 0.7) * 100)
        pct_below_0_5 = float(np.mean(deep_r2 < 0.5) * 100)
        mean_r2_deep = float(np.mean(deep_r2))
        std_r2_deep = float(np.std(deep_r2))
    else:
        median_r2_deep = np.nan
        pct_below_0_9 = np.nan
        pct_below_0_8 = np.nan
        pct_below_0_7 = np.nan
        pct_below_0_5 = np.nan
        mean_r2_deep = np.nan
        std_r2_deep = np.nan

    return {
        'bucket_name': bucket_name,
        'bucket_range': bucket_range,
        'n_files': len(bucket_files),
        'sparsity_values': sorted(list(set([f['sparsity'] for f in bucket_files]))),

        # All features statistics
        'n_total_features': len(all_features),
        'n_valid_features': len(all_r2),
        'all_features_mean_r2': float(np.mean(all_r2)) if len(all_r2) > 0 else np.nan,
        'all_features_median_r2': float(np.median(all_r2)) if len(all_r2) > 0 else np.nan,

        # Deep superposition features statistics (THE KEY METRICS)
        'n_deep_superposition_features': len(deep_superposition_features),
        'deep_superposition_threshold': DEEP_SUPERPOSITION_THRESHOLD,

        # Protocol requested metrics
        'median_r2_deep_superposition': median_r2_deep,
        'mean_r2_deep_superposition': mean_r2_deep,
        'std_r2_deep_superposition': std_r2_deep,
        'pct_features_r2_below_0_9': pct_below_0_9,
        'pct_features_r2_below_0_8': pct_below_0_8,
        'pct_features_r2_below_0_7': pct_below_0_7,
        'pct_features_r2_below_0_5': pct_below_0_5,

        # Distribution data for histograms
        'r2_values_deep': deep_r2.tolist() if len(deep_r2) > 0 else [],
        'slopes_deep': deep_slopes.tolist() if len(deep_slopes) > 0 else [],
        'final_dims_deep': deep_final_dims.tolist() if len(deep_final_dims) > 0 else [],
        'r2_values_all': all_r2.tolist() if len(all_r2) > 0 else [],

        # Percentile breakdown
        'r2_percentiles_deep': {
            'p5': float(np.percentile(deep_r2, 5)) if len(deep_r2) > 0 else np.nan,
            'p10': float(np.percentile(deep_r2, 10)) if len(deep_r2) > 0 else np.nan,
            'p25': float(np.percentile(deep_r2, 25)) if len(deep_r2) > 0 else np.nan,
            'p50': float(np.percentile(deep_r2, 50)) if len(deep_r2) > 0 else np.nan,
            'p75': float(np.percentile(deep_r2, 75)) if len(deep_r2) > 0 else np.nan,
            'p90': float(np.percentile(deep_r2, 90)) if len(deep_r2) > 0 else np.nan,
            'p95': float(np.percentile(deep_r2, 95)) if len(deep_r2) > 0 else np.nan,
        },
    }


def create_histogram_plot(bucket_stats, output_dir):
    """
    Create the R² histogram for deep superposition features.
    This is the key visualization for the verification step.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    bucket_order = ['low', 'high', 'extreme']

    for idx, bucket_name in enumerate(bucket_order):
        stats = bucket_stats.get(bucket_name)
        if stats is None:
            continue

        # Top row: R² histograms for deep superposition features
        ax = axes[0, idx]
        r2_values = np.array(stats['r2_values_deep'])

        if len(r2_values) > 0:
            # Histogram with bins focused on high R² region
            bins = np.linspace(0, 1, 51)
            ax.hist(r2_values, bins=bins, density=True, alpha=0.7,
                   color='steelblue', edgecolor='white', linewidth=0.5)

            # Mark key thresholds
            ax.axvline(0.9, color='red', linestyle='--', linewidth=2,
                      label=f'R²=0.9 ({stats["pct_features_r2_below_0_9"]:.1f}% below)')
            ax.axvline(stats['median_r2_deep_superposition'], color='green',
                      linestyle='-', linewidth=2,
                      label=f'Median={stats["median_r2_deep_superposition"]:.3f}')

            ax.set_xlabel('R² (Feature Trajectory Linearity)', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'{bucket_name.upper()} Sparsity: S ∈ {stats["bucket_range"]}\n'
                        f'Deep Superposition Features (D < {DEEP_SUPERPOSITION_THRESHOLD})\n'
                        f'n = {len(r2_values):,}', fontsize=12)
            ax.legend(loc='upper left', fontsize=9)
            ax.set_xlim(0, 1.05)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No deep superposition\nfeatures in this bucket',
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'{bucket_name.upper()} Sparsity', fontsize=12)

        # Bottom row: CDF of R² values
        ax = axes[1, idx]

        if len(r2_values) > 0:
            sorted_r2 = np.sort(r2_values)
            cdf = np.arange(1, len(sorted_r2) + 1) / len(sorted_r2)

            ax.plot(sorted_r2, cdf, 'b-', linewidth=2)
            ax.axvline(0.9, color='red', linestyle='--', linewidth=2)
            ax.axhline(stats['pct_features_r2_below_0_9'] / 100, color='red',
                      linestyle=':', linewidth=1, alpha=0.7)

            ax.set_xlabel('R²', fontsize=12)
            ax.set_ylabel('Cumulative Fraction', fontsize=12)
            ax.set_title(f'CDF of R² Values\n'
                        f'{stats["pct_features_r2_below_0_9"]:.1f}% have R² < 0.9', fontsize=12)
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'r2_histograms_by_sparsity.png', dpi=150, bbox_inches='tight')
    print(f"Saved: r2_histograms_by_sparsity.png")
    plt.close()


def create_detailed_high_sparsity_plot(bucket_stats, output_dir):
    """
    Create detailed analysis plot for the high sparsity regime.
    """
    stats = bucket_stats.get('high')
    if stats is None or len(stats['r2_values_deep']) == 0:
        print("No high sparsity data available for detailed plot")
        return

    r2_values = np.array(stats['r2_values_deep'])
    slopes = np.array(stats['slopes_deep'])
    final_dims = np.array(stats['final_dims_deep'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: R² histogram with detailed breakdown
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 101)
    counts, edges, _ = ax.hist(r2_values, bins=bins, density=True, alpha=0.7,
                               color='steelblue', edgecolor='white', linewidth=0.3)

    # Add vertical lines at key thresholds
    for thresh, color, label in [(0.5, 'purple', 'R²=0.5'),
                                  (0.7, 'orange', 'R²=0.7'),
                                  (0.9, 'red', 'R²=0.9'),
                                  (0.95, 'darkred', 'R²=0.95')]:
        pct_below = np.mean(r2_values < thresh) * 100
        ax.axvline(thresh, color=color, linestyle='--', linewidth=1.5,
                  label=f'{label} ({pct_below:.1f}% below)')

    ax.axvline(stats['median_r2_deep_superposition'], color='green',
              linestyle='-', linewidth=2,
              label=f'Median={stats["median_r2_deep_superposition"]:.3f}')

    ax.set_xlabel('R² (Feature Trajectory Linearity)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'HIGH SPARSITY (S > 0.9): R² Distribution\n'
                f'Deep Superposition Features (D < {DEEP_SUPERPOSITION_THRESHOLD})\n'
                f'n = {len(r2_values):,}', fontsize=12)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 2: Scatter plot of R² vs final dimensionality
    ax = axes[0, 1]
    scatter = ax.scatter(final_dims, r2_values, alpha=0.3, s=10, c='steelblue')
    ax.axhline(0.9, color='red', linestyle='--', linewidth=2, label='R² = 0.9')
    ax.axvline(0.25, color='orange', linestyle=':', linewidth=1.5, label='D = 0.25')

    ax.set_xlabel('Final Fractional Dimensionality (D)', fontsize=12)
    ax.set_ylabel('Trajectory R²', fontsize=12)
    ax.set_title('R² vs Final Dimensionality\n(Do more superposed features fit better or worse?)', fontsize=12)
    ax.legend()
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 3: Slope distribution
    ax = axes[1, 0]
    valid_slopes = slopes[np.isfinite(slopes)]
    if len(valid_slopes) > 0:
        # Clip extreme values for visualization
        slope_clipped = np.clip(valid_slopes, np.percentile(valid_slopes, 1),
                               np.percentile(valid_slopes, 99))
        ax.hist(slope_clipped, bins=50, density=True, alpha=0.7,
               color='coral', edgecolor='white')
        ax.axvline(np.median(valid_slopes), color='red', linestyle='-', linewidth=2,
                  label=f'Median slope = {np.median(valid_slopes):.4f}')

    ax.set_xlabel('Slope (dD/d||W||²)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Trajectory Slopes', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: "Dark Matter" analysis - features with R² < 0.9
    ax = axes[1, 1]
    dark_matter_mask = r2_values < 0.9
    dark_matter_dims = final_dims[dark_matter_mask]
    good_fit_dims = final_dims[~dark_matter_mask]

    if len(dark_matter_dims) > 0 and len(good_fit_dims) > 0:
        bins = np.linspace(0, 0.5, 26)
        ax.hist(good_fit_dims, bins=bins, alpha=0.6, density=True,
               label=f'R² ≥ 0.9 (n={len(good_fit_dims):,})', color='green')
        ax.hist(dark_matter_dims, bins=bins, alpha=0.6, density=True,
               label=f'R² < 0.9 "Dark Matter" (n={len(dark_matter_dims):,})', color='red')

    ax.set_xlabel('Final Fractional Dimensionality (D)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('"Dark Matter" Analysis\n(Features that defy linear scaling)', fontsize=12)
    ax.legend()
    ax.set_xlim(0, 0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'high_sparsity_detailed_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: high_sparsity_detailed_analysis.png")
    plt.close()


def create_extreme_sparsity_plot(bucket_stats, output_dir):
    """
    Create focused analysis for the extreme sparsity regime (S=0.99).
    """
    stats = bucket_stats.get('extreme')
    if stats is None or len(stats['r2_values_deep']) == 0:
        print("No extreme sparsity data available")
        return

    r2_values = np.array(stats['r2_values_deep'])
    slopes = np.array(stats['slopes_deep'])
    final_dims = np.array(stats['final_dims_deep'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: R² histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    ax.hist(r2_values, bins=bins, density=True, alpha=0.7,
           color='darkred', edgecolor='white', linewidth=0.5)
    ax.axvline(0.9, color='black', linestyle='--', linewidth=2,
              label=f'R²=0.9 ({stats["pct_features_r2_below_0_9"]:.1f}% below)')
    ax.axvline(stats['median_r2_deep_superposition'], color='gold',
              linestyle='-', linewidth=2,
              label=f'Median={stats["median_r2_deep_superposition"]:.3f}')

    ax.set_xlabel('R²', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'EXTREME SPARSITY (S=0.99)\nR² Distribution, n={len(r2_values):,}', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 2: R² vs D scatter
    ax = axes[1]
    ax.scatter(final_dims, r2_values, alpha=0.4, s=15, c='darkred')
    ax.axhline(0.9, color='black', linestyle='--', linewidth=2)

    ax.set_xlabel('Final D', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Trajectory Fit Quality vs\nSuperpositon Depth', fontsize=14)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 3: Slope vs R² (colored by D)
    ax = axes[2]
    valid = np.isfinite(slopes)
    if np.sum(valid) > 0:
        sc = ax.scatter(slopes[valid], r2_values[valid],
                       c=final_dims[valid], cmap='plasma',
                       alpha=0.5, s=15, vmin=0, vmax=0.5)
        plt.colorbar(sc, ax=ax, label='Final D')
        ax.axhline(0.9, color='black', linestyle='--', linewidth=2)

    ax.set_xlabel('Slope', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Slope vs R² colored by D', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'extreme_sparsity_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: extreme_sparsity_analysis.png")
    plt.close()


def create_comparison_summary(bucket_stats, output_dir):
    """
    Create a summary comparison across all sparsity buckets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bucket_names = ['low', 'high', 'extreme']
    colors = ['green', 'steelblue', 'darkred']

    # Panel 1: Median R² comparison
    ax = axes[0]
    medians = []
    labels = []
    for name in bucket_names:
        stats = bucket_stats.get(name)
        if stats is not None and not np.isnan(stats['median_r2_deep_superposition']):
            medians.append(stats['median_r2_deep_superposition'])
            labels.append(f'{name.upper()}\n(S∈{stats["bucket_range"]})')

    if medians:
        bars = ax.bar(range(len(medians)), medians, color=colors[:len(medians)], alpha=0.7)
        ax.set_xticks(range(len(medians)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.axhline(0.9, color='red', linestyle='--', linewidth=2, label='R² = 0.9 threshold')
        ax.set_ylabel('Median R² of Deep Superposition Features', fontsize=12)
        ax.set_title('Comparison of Linear Fit Quality Across Sparsity Regimes', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, medians):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel 2: % features with R² < 0.9 ("dark matter" fraction)
    ax = axes[1]
    dark_matter_pct = []
    for name in bucket_names:
        stats = bucket_stats.get(name)
        if stats is not None and not np.isnan(stats['pct_features_r2_below_0_9']):
            dark_matter_pct.append(stats['pct_features_r2_below_0_9'])

    if dark_matter_pct:
        bars = ax.bar(range(len(dark_matter_pct)), dark_matter_pct,
                     color=colors[:len(dark_matter_pct)], alpha=0.7)
        ax.set_xticks(range(len(dark_matter_pct)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('% Features with R² < 0.9 ("Dark Matter")', fontsize=12)
        ax.set_title('Fraction of Superposition Features That Defy Linear Scaling', fontsize=14)
        ax.set_ylim(0, max(dark_matter_pct) * 1.2 if dark_matter_pct else 100)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, dark_matter_pct):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'sparsity_regime_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: sparsity_regime_comparison.png")
    plt.close()


def generate_markdown_report(bucket_stats, output_dir, processing_time):
    """
    Generate comprehensive markdown report.
    """
    report = []
    report.append("# Stratified Linearity Analysis Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Processing Time:** {processing_time:.1f} seconds")
    report.append(f"\n---\n")

    report.append("## Executive Summary\n")
    report.append("This analysis tests whether the linear scaling law **D_i ∝ ||W_i||²** holds ")
    report.append("universally for superposition features, without assuming the eigenvalue relationship a priori.\n")

    report.append("### Key Findings\n")

    for bucket_name in ['high', 'extreme']:
        stats = bucket_stats.get(bucket_name)
        if stats and len(stats['r2_values_deep']) > 0:
            report.append(f"**{bucket_name.upper()} Sparsity (S ∈ {stats['bucket_range']}):**\n")
            report.append(f"- Median R² of Deep Superposition Features: **{stats['median_r2_deep_superposition']:.4f}**\n")
            report.append(f"- Percentage with R² < 0.9 (\"Dark Matter\"): **{stats['pct_features_r2_below_0_9']:.2f}%**\n")
            report.append(f"- Number of features analyzed: {stats['n_deep_superposition_features']:,}\n\n")

    report.append("---\n")

    report.append("## Analysis Protocol\n")
    report.append("""
### 1. Primary Stratification by Sparsity (S)
- **Low Sparsity:** S ∈ [0.0, 0.2] - expect mostly orthogonal features
- **High Sparsity:** S ∈ [0.9, 1.0] - expect mostly superposition features
- **Extreme Sparsity:** S = 0.99 - the critical regime

### 2. Secondary Stratification by Feature Dimensionality
- Filter for D_i < 0.5 at final checkpoint ("Deep Superposition")

### 3. Feature-Level Linearity Test
- For each feature, fit D_i(t) vs ||W_i(t)||² across ALL training checkpoints
- Calculate R² for every single feature trajectory

### 4. Verification
- If the conjecture is robust, the R² histogram should be strongly peaked at 1.0
- "Dark Matter" = features with R² < 0.9 that defy the geometric prediction
""")

    report.append("\n---\n")

    report.append("## Detailed Results by Sparsity Bucket\n")

    for bucket_name in ['low', 'high', 'extreme']:
        stats = bucket_stats.get(bucket_name)
        if stats is None:
            continue

        report.append(f"### {bucket_name.upper()} Sparsity: S ∈ {stats['bucket_range']}\n")
        report.append(f"- **Files in bucket:** {stats['n_files']}\n")
        report.append(f"- **Sparsity values:** {', '.join([f'{s:.2f}' for s in stats['sparsity_values'][:10]])}")
        if len(stats['sparsity_values']) > 10:
            report.append(f" ... ({len(stats['sparsity_values'])} total)")
        report.append("\n\n")

        report.append("#### All Features\n")
        report.append(f"- Total features: {stats['n_total_features']:,}\n")
        report.append(f"- Valid features (enough data points): {stats['n_valid_features']:,}\n")
        report.append(f"- Mean R²: {stats['all_features_mean_r2']:.4f}\n")
        report.append(f"- Median R²: {stats['all_features_median_r2']:.4f}\n\n")

        report.append(f"#### Deep Superposition Features (D < {DEEP_SUPERPOSITION_THRESHOLD})\n")
        if stats['n_deep_superposition_features'] > 0:
            report.append(f"- **Count:** {stats['n_deep_superposition_features']:,}\n")
            report.append(f"- **Median R²:** {stats['median_r2_deep_superposition']:.4f}\n")
            report.append(f"- **Mean R²:** {stats['mean_r2_deep_superposition']:.4f} ± {stats['std_r2_deep_superposition']:.4f}\n\n")

            report.append("##### \"Dark Matter\" Analysis (features defying linear scaling)\n")
            report.append(f"- R² < 0.9: **{stats['pct_features_r2_below_0_9']:.2f}%**\n")
            report.append(f"- R² < 0.8: {stats['pct_features_r2_below_0_8']:.2f}%\n")
            report.append(f"- R² < 0.7: {stats['pct_features_r2_below_0_7']:.2f}%\n")
            report.append(f"- R² < 0.5: {stats['pct_features_r2_below_0_5']:.2f}%\n\n")

            report.append("##### R² Percentiles\n")
            p = stats['r2_percentiles_deep']
            report.append(f"| Percentile | R² Value |\n")
            report.append(f"|------------|----------|\n")
            for pct in ['p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']:
                report.append(f"| {pct.upper()} | {p[pct]:.4f} |\n")
            report.append("\n")
        else:
            report.append("*No deep superposition features found in this bucket.*\n\n")

    report.append("---\n")

    report.append("## Interpretation\n\n")

    high_stats = bucket_stats.get('high')
    if high_stats and len(high_stats['r2_values_deep']) > 0:
        median_r2 = high_stats['median_r2_deep_superposition']
        dark_matter_pct = high_stats['pct_features_r2_below_0_9']

        if median_r2 > 0.95 and dark_matter_pct < 10:
            report.append("**STRONG SUPPORT for the conjecture:** The vast majority of deep superposition ")
            report.append("features exhibit highly linear D_i vs ||W_i||² trajectories (median R² > 0.95, ")
            report.append(f"only {dark_matter_pct:.1f}% \"dark matter\").\n\n")
        elif median_r2 > 0.85 and dark_matter_pct < 25:
            report.append("**MODERATE SUPPORT for the conjecture:** Most deep superposition features show ")
            report.append("approximately linear scaling, though a non-trivial fraction deviates from the prediction.\n\n")
        else:
            report.append("**WEAK SUPPORT or REJECTION:** A significant fraction of superposition features ")
            report.append("do not follow the predicted linear scaling law. This suggests the geometric ")
            report.append("picture may be incomplete or requires refinement.\n\n")

    report.append("## Generated Files\n\n")
    report.append("### Python Scripts\n")
    report.append("- `stratified_linearity_analysis.py` - Main analysis script\n\n")
    report.append("### Visualizations\n")
    report.append("- `r2_histograms_by_sparsity.png` - R² histograms for all sparsity buckets\n")
    report.append("- `high_sparsity_detailed_analysis.png` - Detailed analysis of high sparsity regime\n")
    report.append("- `extreme_sparsity_analysis.png` - Analysis of S=0.99 critical regime\n")
    report.append("- `sparsity_regime_comparison.png` - Cross-bucket comparison summary\n\n")
    report.append("### Data Files\n")
    report.append("- `bucket_statistics.json` - Complete statistics for all buckets\n")
    report.append("- `analysis_report.md` - This report\n")

    report_text = ''.join(report)

    with open(output_dir / 'analysis_report.md', 'w') as f:
        f.write(report_text)

    print(f"Saved: analysis_report.md")
    return report_text


def main():
    """Main analysis pipeline."""
    import time
    start_time = time.time()

    print("=" * 70)
    print("Stratified Linearity Analysis: Feature-Level D_i vs ||W_i||² Trajectories")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all checkpoint files
    checkpoint_files = sorted(INPUT_DIR.glob('n1024_m*.h5'))
    print(f"Found {len(checkpoint_files)} checkpoint files")

    # Process all files
    print("\n--- Phase 1: Processing individual files ---")
    all_file_results = []

    for filepath in tqdm(checkpoint_files, desc="Processing files"):
        try:
            result = process_single_file(filepath)
            all_file_results.append(result)
        except Exception as e:
            print(f"\nError processing {filepath.name}: {e}")
            continue

    print(f"\nSuccessfully processed {len(all_file_results)} files")

    # Aggregate by sparsity bucket
    print("\n--- Phase 2: Aggregating by sparsity bucket ---")
    bucket_stats = {}

    for bucket_name, bucket_range in SPARSITY_BUCKETS.items():
        print(f"Processing bucket: {bucket_name} (S ∈ {bucket_range})")
        stats = aggregate_by_bucket(all_file_results, bucket_name, bucket_range)
        if stats is not None:
            bucket_stats[bucket_name] = stats
            print(f"  - Files: {stats['n_files']}")
            print(f"  - Deep superposition features: {stats['n_deep_superposition_features']:,}")
            if stats['n_deep_superposition_features'] > 0:
                print(f"  - Median R²: {stats['median_r2_deep_superposition']:.4f}")
                print(f"  - % with R² < 0.9: {stats['pct_features_r2_below_0_9']:.2f}%")

    # Save raw statistics
    print("\n--- Phase 3: Saving results ---")

    # Convert numpy arrays to lists for JSON serialization
    bucket_stats_serializable = {}
    for name, stats in bucket_stats.items():
        bucket_stats_serializable[name] = {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in stats.items()
        }

    with open(OUTPUT_DIR / 'bucket_statistics.json', 'w') as f:
        json.dump(bucket_stats_serializable, f, indent=2)
    print("Saved: bucket_statistics.json")

    # Create visualizations
    print("\n--- Phase 4: Creating visualizations ---")
    create_histogram_plot(bucket_stats, OUTPUT_DIR)
    create_detailed_high_sparsity_plot(bucket_stats, OUTPUT_DIR)
    create_extreme_sparsity_plot(bucket_stats, OUTPUT_DIR)
    create_comparison_summary(bucket_stats, OUTPUT_DIR)

    # Generate report
    print("\n--- Phase 5: Generating report ---")
    processing_time = time.time() - start_time
    generate_markdown_report(bucket_stats, OUTPUT_DIR, processing_time)

    # Print final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print("\n### KEY RESULTS ###\n")

    for bucket_name in ['high', 'extreme']:
        stats = bucket_stats.get(bucket_name)
        if stats and stats['n_deep_superposition_features'] > 0:
            print(f"{bucket_name.upper()} SPARSITY (S ∈ {stats['bucket_range']}):")
            print(f"  Median R² of Deep Superposition Features: {stats['median_r2_deep_superposition']:.4f}")
            print(f"  Percentage with R² < 0.9 ('Dark Matter'): {stats['pct_features_r2_below_0_9']:.2f}%")
            print(f"  Number of features analyzed: {stats['n_deep_superposition_features']:,}")
            print()

    print(f"Processing time: {processing_time:.1f} seconds")
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
