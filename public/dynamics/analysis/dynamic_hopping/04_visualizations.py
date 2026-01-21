#!/usr/bin/env python3
"""
Analysis 4: Visualizations for Dynamic Hopping Analysis

Creates comprehensive visualizations including:

1. Rayleigh Quotient Heatmaps: κ_i(t) over time for all features
2. Jump Detection Plots: Locations and magnitudes of detected jumps
3. Volatility Evolution: How hopping activity changes during training
4. Feature Classification: Distribution of hopping behaviors
5. Synchrony Analysis: Temporal structure of coordinated hopping
6. Sparsity Comparison: How hopping differs across sparsity regimes

Outputs:
--------
- All plots saved to plots/ directory
- Summary figure combining key results
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import load_config, get_sparsity_bucket


def plot_rayleigh_heatmap(x: np.ndarray, checkpoint_steps: np.ndarray,
                          title: str, output_path: Path, cfg: dict):
    """
    Create heatmap of log-transformed Rayleigh quotients x_i(t) over time.

    x-axis: checkpoint (time)
    y-axis: feature index
    color: x_i(t) = log(κ_i(t))
    """
    T, n = x.shape

    # Sort features by mean value for better visualization
    mean_x = np.nanmean(x, axis=0)
    sort_idx = np.argsort(mean_x)[::-1]  # Descending
    x_sorted = x[:, sort_idx]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create heatmap
    vmin, vmax = np.nanpercentile(x, [1, 99])
    im = ax.imshow(x_sorted.T, aspect='auto', cmap='viridis',
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    # Labels
    ax.set_xlabel('Checkpoint', fontsize=12)
    ax.set_ylabel('Feature (sorted by mean κ)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='log(κ + ε)')

    # X-axis ticks (show actual training steps)
    n_ticks = min(10, T)
    tick_indices = np.linspace(0, T-1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f'{checkpoint_steps[i]:,}' for i in tick_indices], rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_jump_events(x: np.ndarray, checkpoint_steps: np.ndarray,
                     z_threshold: float, epsilon_mad: float,
                     title: str, output_path: Path):
    """
    Visualize jump events as a scatter plot.

    Shows when and where jumps occur, with size/color indicating magnitude.
    """
    T, n = x.shape

    # Compute jumps
    delta_x = np.diff(x, axis=0)
    sigma_global = 1.4826 * np.nanmedian(np.abs(delta_x - np.nanmedian(delta_x)))
    threshold = z_threshold * (sigma_global + epsilon_mad)

    is_jump = np.abs(delta_x) > threshold

    # Get jump locations
    jump_times, jump_features = np.where(is_jump)
    jump_magnitudes = np.abs(delta_x[is_jump])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: Jump scatter plot
    ax = axes[0, 0]
    if len(jump_times) > 0:
        scatter = ax.scatter(jump_times, jump_features,
                            c=jump_magnitudes, s=5,
                            cmap='hot', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='|Δx|')
    ax.set_xlabel('Checkpoint Index', fontsize=11)
    ax.set_ylabel('Feature Index', fontsize=11)
    ax.set_title('Jump Events (magnitude-colored)', fontsize=12)
    ax.set_xlim(0, T-1)
    ax.set_ylim(0, n)

    # Top-right: Jumps per checkpoint
    ax = axes[0, 1]
    jumps_per_time = np.sum(is_jump, axis=1)
    ax.bar(range(len(jumps_per_time)), jumps_per_time, color='steelblue', alpha=0.7)
    ax.axhline(np.mean(jumps_per_time), color='red', linestyle='--',
               label=f'Mean: {np.mean(jumps_per_time):.1f}')
    ax.set_xlabel('Checkpoint Index', fontsize=11)
    ax.set_ylabel('Number of Jumps', fontsize=11)
    ax.set_title('Jumps per Checkpoint', fontsize=12)
    ax.legend()

    # Bottom-left: Jumps per feature (histogram)
    ax = axes[1, 0]
    jumps_per_feature = np.sum(is_jump, axis=0)
    ax.hist(jumps_per_feature, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(jumps_per_feature), color='red', linestyle='--',
               label=f'Mean: {np.mean(jumps_per_feature):.2f}')
    ax.set_xlabel('Number of Jumps', fontsize=11)
    ax.set_ylabel('Number of Features', fontsize=11)
    ax.set_title('Distribution of Jumps per Feature', fontsize=12)
    ax.legend()

    # Bottom-right: Jump magnitude distribution
    ax = axes[1, 1]
    if len(jump_magnitudes) > 0:
        ax.hist(jump_magnitudes, bins=50, color='coral', alpha=0.7, edgecolor='black')
        ax.axvline(threshold, color='red', linestyle='--',
                   label=f'Threshold: {threshold:.3f}')
    ax.set_xlabel('Jump Magnitude |Δx|', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Jump Magnitude Distribution', fontsize=12)
    ax.legend()

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_volatility_evolution(x: np.ndarray, checkpoint_steps: np.ndarray,
                              title: str, output_path: Path, window_size: int = 10):
    """
    Plot how volatility evolves over training.
    """
    T, n = x.shape

    # Compute rolling volatility
    n_windows = T - window_size + 1
    volatility = np.zeros((n_windows, n))
    for i in range(n_windows):
        window = x[i:i + window_size, :]
        volatility[i] = np.nanstd(window, axis=0)

    # Mean and percentiles
    mean_vol = np.nanmean(volatility, axis=1)
    p25_vol = np.nanpercentile(volatility, 25, axis=1)
    p75_vol = np.nanpercentile(volatility, 75, axis=1)
    p10_vol = np.nanpercentile(volatility, 10, axis=1)
    p90_vol = np.nanpercentile(volatility, 90, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Volatility over time
    ax = axes[0]
    window_centers = np.arange(window_size // 2, window_size // 2 + n_windows)

    ax.fill_between(window_centers, p10_vol, p90_vol, alpha=0.2, color='blue', label='10-90th percentile')
    ax.fill_between(window_centers, p25_vol, p75_vol, alpha=0.4, color='blue', label='25-75th percentile')
    ax.plot(window_centers, mean_vol, 'b-', linewidth=2, label='Mean')

    ax.set_xlabel('Checkpoint Index (window center)', fontsize=11)
    ax.set_ylabel('Rolling Volatility (σ)', fontsize=11)
    ax.set_title('Volatility Evolution Over Training', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Early vs Late volatility scatter
    ax = axes[1]
    third = n_windows // 3
    early_vol = np.nanmean(volatility[:third, :], axis=0)
    late_vol = np.nanmean(volatility[2*third:, :], axis=0)

    ax.scatter(early_vol, late_vol, alpha=0.3, s=10)
    ax.plot([0, max(early_vol.max(), late_vol.max())],
            [0, max(early_vol.max(), late_vol.max())],
            'r--', label='y=x (no change)')

    ax.set_xlabel('Early Training Volatility', fontsize=11)
    ax.set_ylabel('Late Training Volatility', fontsize=11)
    ax.set_title('Early vs Late Volatility per Feature', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate percentages
    converging = np.sum(late_vol < 0.7 * early_vol)
    diverging = np.sum(late_vol > 1.5 * early_vol)
    stable = n - converging - diverging
    ax.annotate(f'Converging: {converging} ({100*converging/n:.1f}%)\n'
                f'Diverging: {diverging} ({100*diverging/n:.1f}%)\n'
                f'Stable: {stable} ({100*stable/n:.1f}%)',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_classification_summary(json_path: Path, output_path: Path, cfg: dict):
    """
    Plot summary of feature classifications across all experiments.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    summary = data['summary_by_sparsity']
    buckets = list(summary.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Colors for sparsity buckets
    colors = {'low': 'blue', 'medium': 'green', 'high': 'orange', 'extreme': 'red'}

    # Top-left: Volatility class distribution by sparsity
    ax = axes[0, 0]
    volatility_classes = ['stable', 'moderate', 'active', 'extreme']
    x = np.arange(len(buckets))
    width = 0.2

    for i, vc in enumerate(volatility_classes):
        values = [summary[b]['volatility_class_means'].get(vc, 0) for b in buckets]
        ax.bar(x + i * width, values, width, label=vc.capitalize())

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([b.capitalize() for b in buckets])
    ax.set_xlabel('Sparsity Bucket', fontsize=11)
    ax.set_ylabel('Mean Count per Experiment', fontsize=11)
    ax.set_title('Feature Volatility Classes by Sparsity', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Top-right: Temporal pattern distribution
    ax = axes[0, 1]
    patterns = ['converging', 'diverging', 'steady', 'episodic']
    width = 0.18

    for i, p in enumerate(patterns):
        values = [summary[b]['temporal_pattern_means'].get(p, 0) for b in buckets]
        ax.bar(x + i * width, values, width, label=p.capitalize())

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([b.capitalize() for b in buckets])
    ax.set_xlabel('Sparsity Bucket', fontsize=11)
    ax.set_ylabel('Mean Count per Experiment', fontsize=11)
    ax.set_title('Temporal Patterns by Sparsity', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-left: Synchrony ratio by sparsity
    ax = axes[1, 0]
    sync_ratios = [summary[b]['mean_synchrony_ratio'] for b in buckets]
    bars = ax.bar(buckets, sync_ratios, color=[colors[b] for b in buckets], alpha=0.7)
    ax.axhline(1.0, color='gray', linestyle='--', label='Independent (ratio=1)')
    ax.set_xlabel('Sparsity Bucket', fontsize=11)
    ax.set_ylabel('Mean Synchrony Ratio', fontsize=11)
    ax.set_title('Feature Hopping Synchrony by Sparsity', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-right: Trend direction counts
    ax = axes[1, 1]
    trend_dirs = ['increasing', 'stable', 'decreasing']
    width = 0.25

    for i, td in enumerate(trend_dirs):
        values = [summary[b]['trend_direction_counts'].get(td, 0) for b in buckets]
        ax.bar(x + i * width, values, width, label=td.capitalize())

    ax.set_xticks(x + width)
    ax.set_xticklabels([b.capitalize() for b in buckets])
    ax.set_xlabel('Sparsity Bucket', fontsize=11)
    ax.set_ylabel('Number of Experiments', fontsize=11)
    ax.set_title('Volatility Trend Direction by Sparsity', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sparsity_comparison(json_path: Path, output_path: Path, cfg: dict):
    """
    Create comparison plot of hopping metrics across sparsity regimes.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    per_file = data['per_file_results']

    # Extract data by sparsity
    sparsities = [r['sparsity'] for r in per_file]
    total_jumps = [r['total_jumps'] for r in per_file]
    sigma_global = [r['sigma_global'] for r in per_file]
    late_sigma_ratio = [r['late_sigma_ratio'] for r in per_file]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: Total jumps vs sparsity
    ax = axes[0, 0]
    ax.scatter(sparsities, total_jumps, alpha=0.5, s=20)
    ax.set_xlabel('Sparsity', fontsize=11)
    ax.set_ylabel('Total Jumps', fontsize=11)
    ax.set_title('Total Jump Events vs Sparsity', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Top-right: Global sigma vs sparsity
    ax = axes[0, 1]
    ax.scatter(sparsities, sigma_global, alpha=0.5, s=20, color='orange')
    ax.set_xlabel('Sparsity', fontsize=11)
    ax.set_ylabel('Global σ (robust)', fontsize=11)
    ax.set_title('Volatility vs Sparsity', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Late/Early sigma ratio vs sparsity
    ax = axes[1, 0]
    ax.scatter(sparsities, late_sigma_ratio, alpha=0.5, s=20, color='green')
    ax.axhline(1.0, color='red', linestyle='--', label='No change')
    ax.set_xlabel('Sparsity', fontsize=11)
    ax.set_ylabel('Late/Early σ Ratio', fontsize=11)
    ax.set_title('Volatility Change Over Training vs Sparsity', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Sigma vs total jumps (colored by sparsity)
    ax = axes[1, 1]
    scatter = ax.scatter(sigma_global, total_jumps, c=sparsities,
                        cmap='coolwarm', alpha=0.5, s=20)
    plt.colorbar(scatter, ax=ax, label='Sparsity')
    ax.set_xlabel('Global σ (robust)', fontsize=11)
    ax.set_ylabel('Total Jumps', fontsize=11)
    ax.set_title('Jumps vs Volatility (colored by sparsity)', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_figure(results_dir: Path, output_path: Path, cfg: dict):
    """
    Create a comprehensive summary figure combining key results.
    """
    # Load all result files
    jump_results_path = results_dir / 'jump_detection_results.json'
    temporal_results_path = results_dir / 'temporal_patterns_results.json'

    if not jump_results_path.exists() or not temporal_results_path.exists():
        print("Missing result files for summary figure")
        return

    with open(jump_results_path, 'r') as f:
        jump_data = json.load(f)
    with open(temporal_results_path, 'r') as f:
        temporal_data = json.load(f)

    jump_summary = jump_data['summary_by_sparsity']
    temporal_summary = temporal_data['summary_by_sparsity']

    buckets = list(jump_summary.keys())
    bucket_colors = {'low': '#3498db', 'medium': '#2ecc71',
                     'high': '#e67e22', 'extreme': '#e74c3c'}

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Mean jumps per feature by sparsity
    ax = fig.add_subplot(gs[0, 0])
    values = [jump_summary[b]['mean_jumps_per_feature'] for b in buckets]
    bars = ax.bar(buckets, values, color=[bucket_colors[b] for b in buckets])
    ax.set_ylabel('Mean Jumps/Feature')
    ax.set_title('Jump Frequency by Sparsity')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Global sigma by sparsity
    ax = fig.add_subplot(gs[0, 1])
    values = [jump_summary[b]['mean_sigma_global'] for b in buckets]
    ax.bar(buckets, values, color=[bucket_colors[b] for b in buckets])
    ax.set_ylabel('Mean σ_global')
    ax.set_title('Volatility by Sparsity')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Late/Early sigma ratio
    ax = fig.add_subplot(gs[0, 2])
    values = [jump_summary[b]['mean_sigma_ratio'] for b in buckets]
    ax.bar(buckets, values, color=[bucket_colors[b] for b in buckets])
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Late/Early σ Ratio')
    ax.set_title('Volatility Trend')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Synchrony ratio
    ax = fig.add_subplot(gs[1, 0])
    values = [temporal_summary[b]['mean_synchrony_ratio'] for b in buckets]
    ax.bar(buckets, values, color=[bucket_colors[b] for b in buckets])
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Synchrony Ratio')
    ax.set_title('Feature Hopping Synchrony')
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Volatility classes stacked bar
    ax = fig.add_subplot(gs[1, 1])
    vol_classes = ['stable', 'moderate', 'active', 'extreme']
    vol_colors = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c']
    bottoms = np.zeros(len(buckets))

    for vc, color in zip(vol_classes, vol_colors):
        values = [temporal_summary[b]['volatility_class_means'].get(vc, 0) for b in buckets]
        ax.bar(buckets, values, bottom=bottoms, color=color, label=vc.capitalize())
        bottoms += values

    ax.set_ylabel('Features')
    ax.set_title('Volatility Class Distribution')
    ax.legend(loc='upper right', fontsize=8)

    # 6. Temporal patterns stacked bar
    ax = fig.add_subplot(gs[1, 2])
    patterns = ['converging', 'steady', 'episodic', 'diverging']
    pat_colors = ['#3498db', '#95a5a6', '#9b59b6', '#e74c3c']
    bottoms = np.zeros(len(buckets))

    for p, color in zip(patterns, pat_colors):
        values = [temporal_summary[b]['temporal_pattern_means'].get(p, 0) for b in buckets]
        ax.bar(buckets, values, bottom=bottoms, color=color, label=p.capitalize())
        bottoms += values

    ax.set_ylabel('Features')
    ax.set_title('Temporal Pattern Distribution')
    ax.legend(loc='upper right', fontsize=8)

    # 7-9: Scatter plots spanning bottom row
    ax = fig.add_subplot(gs[2, :])

    # Create sparsity vs multiple metrics scatter
    per_file_jump = jump_data['per_file_results']

    sparsities = [r['sparsity'] for r in per_file_jump]
    jumps = [r['mean_jumps_per_feature'] for r in per_file_jump]
    sigmas = [r['sigma_global'] for r in per_file_jump]

    # Normalize for combined plot
    jumps_norm = np.array(jumps) / max(jumps)
    sigmas_norm = np.array(sigmas) / max(sigmas)

    ax.scatter(sparsities, jumps_norm, alpha=0.5, s=30, label='Jumps/Feature (normalized)')
    ax.scatter(sparsities, sigmas_norm, alpha=0.5, s=30, marker='s', label='σ_global (normalized)')

    ax.set_xlabel('Sparsity', fontsize=12)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Hopping Metrics vs Sparsity (all experiments)', fontsize=13)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Main title
    fig.suptitle('Dynamic Feature Hopping Analysis Summary', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create visualizations for dynamic hopping analysis')
    parser.add_argument('--sample-plots', type=int, default=5,
                        help='Number of individual file plots to create')
    args = parser.parse_args()

    cfg = load_config()
    cfg['plots_dir'].mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Analysis 4: Visualization Generation")
    print("=" * 70)

    # Find precomputed data
    per_file_dir = cfg['results_dir'] / 'per_file'
    files = sorted(per_file_dir.glob('*_rayleigh.npz'))

    if len(files) == 0:
        print("\nNo data files found. Run previous analyses first.")
        return

    print(f"\nFound {len(files)} data files")
    print(f"Creating {args.sample_plots} individual file plots")

    # Select sample files across sparsity range
    sample_files = []
    if len(files) <= args.sample_plots:
        sample_files = files
    else:
        # Select evenly spaced
        indices = np.linspace(0, len(files) - 1, args.sample_plots, dtype=int)
        sample_files = [files[i] for i in indices]

    # Create individual file plots
    for filepath in tqdm(sample_files, desc="Creating file plots"):
        try:
            data = np.load(filepath)
            x = data['x']
            kappa = data['kappa']
            checkpoint_steps = data['checkpoint_steps']
            sparsity = float(data['sparsity'])

            base_name = filepath.stem.replace('_rayleigh', '')

            # 1. Rayleigh heatmap
            plot_rayleigh_heatmap(
                x, checkpoint_steps,
                f'Log Rayleigh Quotient: {base_name}\n(sparsity={sparsity:.3f})',
                cfg['plots_dir'] / f'{base_name}_rayleigh_heatmap.png',
                cfg
            )

            # 2. Jump events
            plot_jump_events(
                x, checkpoint_steps,
                cfg['z_threshold'], cfg['epsilon_mad'],
                f'Jump Detection: {base_name}\n(sparsity={sparsity:.3f})',
                cfg['plots_dir'] / f'{base_name}_jump_events.png'
            )

            # 3. Volatility evolution
            plot_volatility_evolution(
                x, checkpoint_steps,
                f'Volatility Evolution: {base_name}\n(sparsity={sparsity:.3f})',
                cfg['plots_dir'] / f'{base_name}_volatility_evolution.png',
                cfg['local_window_size']
            )

        except Exception as e:
            print(f"\nError plotting {filepath.name}: {e}")
            continue

    # Create aggregate plots
    print("\n--- Creating Aggregate Plots ---")

    # Feature classification summary
    temporal_results = cfg['results_dir'] / 'temporal_patterns_results.json'
    if temporal_results.exists():
        print("Creating feature classification summary...")
        plot_feature_classification_summary(
            temporal_results,
            cfg['plots_dir'] / 'feature_classification_summary.png',
            cfg
        )

    # Sparsity comparison
    jump_results = cfg['results_dir'] / 'jump_detection_results.json'
    if jump_results.exists():
        print("Creating sparsity comparison...")
        plot_sparsity_comparison(
            jump_results,
            cfg['plots_dir'] / 'sparsity_comparison.png',
            cfg
        )

    # Summary figure
    if temporal_results.exists() and jump_results.exists():
        print("Creating summary figure...")
        create_summary_figure(
            cfg['results_dir'],
            cfg['plots_dir'] / 'dynamic_hopping_summary.png',
            cfg
        )

    print(f"\nAll plots saved to: {cfg['plots_dir']}")

    print("\n" + "=" * 70)
    print("Analysis 4 Complete: Visualizations Created")
    print("=" * 70)


if __name__ == '__main__':
    main()
