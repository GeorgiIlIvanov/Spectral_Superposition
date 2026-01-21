#!/usr/bin/env python3
"""
Analysis 1: Dark Matter Evolution Over Training

Track how the fraction of "dark matter" features (R² < 0.9 for D_i vs ||W_i||²)
evolves during training to determine if dark matter is transient or persistent.

Key outputs:
- Dark matter fraction at each checkpoint (cumulative and windowed R²)
- Stratified by sparsity regime
- Identification of features that transition from dark matter to well-behaved (or vice versa)
"""

import h5py
import numpy as np
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
from collections import defaultdict
from multiprocessing import Pool, cpu_count


INPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/dark_matter_analysis/temporal_analysis')
PLOTS_DIR = OUTPUT_DIR / 'plots'
RESULTS_DIR = OUTPUT_DIR / 'results'

# Sparsity buckets
SPARSITY_BUCKETS = {
    'low': (0.0, 0.3),
    'medium': (0.3, 0.7),
    'high': (0.7, 0.95),
    'extreme': (0.95, 1.0),
}

R2_THRESHOLD = 0.9  # Dark matter threshold


def compute_r2_vectorized(x, y):
    """
    Vectorized R² computation for multiple features simultaneously.

    Args:
        x: (T, n) array - independent variable for each feature
        y: (T, n) array - dependent variable for each feature

    Returns:
        r2_values: (n,) array of R² for each feature
    """
    # Mask invalid values
    valid = np.isfinite(x) & np.isfinite(y) & (x > 1e-8)

    # Replace invalid with nan for computation
    x_masked = np.where(valid, x, np.nan)
    y_masked = np.where(valid, y, np.nan)

    # Count valid points per feature
    n_valid = np.sum(valid, axis=0)

    # Compute means (ignoring nan)
    x_mean = np.nanmean(x_masked, axis=0)
    y_mean = np.nanmean(y_masked, axis=0)

    # Compute variance and covariance
    x_centered = x_masked - x_mean
    y_centered = y_masked - y_mean

    ss_xx = np.nansum(x_centered ** 2, axis=0)
    ss_yy = np.nansum(y_centered ** 2, axis=0)
    ss_xy = np.nansum(x_centered * y_centered, axis=0)

    # Correlation coefficient
    with np.errstate(divide='ignore', invalid='ignore'):
        r = ss_xy / np.sqrt(ss_xx * ss_yy)
        r2 = r ** 2

    # Mask features with insufficient data or zero variance
    r2 = np.where((n_valid >= 5) & (ss_xx > 1e-10), r2, np.nan)

    return r2


def compute_cumulative_r2(feature_norms, fractional_dims, up_to_checkpoint):
    """
    Compute R² for D_i vs ||W_i||² using checkpoints [0, up_to_checkpoint].
    Vectorized version - processes all features simultaneously.
    """
    x = feature_norms[:up_to_checkpoint+1, :]
    y = fractional_dims[:up_to_checkpoint+1, :]
    return compute_r2_vectorized(x, y)


def compute_windowed_r2(feature_norms, fractional_dims, center_checkpoint, window_size=10):
    """
    Compute R² using a sliding window around center_checkpoint.
    Vectorized version - processes all features simultaneously.
    """
    n_checkpoints = feature_norms.shape[0]
    start = max(0, center_checkpoint - window_size // 2)
    end = min(n_checkpoints, center_checkpoint + window_size // 2 + 1)

    x = feature_norms[start:end, :]
    y = fractional_dims[start:end, :]
    return compute_r2_vectorized(x, y)


def analyze_file(filepath):
    """Analyze a single experiment file."""
    with h5py.File(filepath, 'r') as f:
        feature_norms = f['feature_norms'][:]  # (T, n)
        fractional_dims = f['fractional_dims'][:]  # (T, n)
        checkpoint_steps = f['checkpoint_steps'][:]  # (T,)
        sparsity = float(f.attrs['sparsity'])
        m_hidden = int(f.attrs['m_hidden'])
        seed = int(f.attrs['seed'])

    n_checkpoints, n_features = feature_norms.shape

    # Compute cumulative R² at each checkpoint
    cumulative_r2 = np.zeros((n_checkpoints, n_features))
    for t in range(n_checkpoints):
        if t < 5:  # Need minimum points for regression
            cumulative_r2[t, :] = np.nan
        else:
            cumulative_r2[t, :] = compute_cumulative_r2(feature_norms, fractional_dims, t)

    # Compute windowed R² at each checkpoint
    windowed_r2 = np.zeros((n_checkpoints, n_features))
    for t in range(n_checkpoints):
        windowed_r2[t, :] = compute_windowed_r2(feature_norms, fractional_dims, t, window_size=10)

    # Final checkpoint statistics
    final_r2 = cumulative_r2[-1, :]

    # Dark matter evolution (fraction of features with R² < threshold at each checkpoint)
    cumulative_dm_fraction = np.zeros(n_checkpoints)
    windowed_dm_fraction = np.zeros(n_checkpoints)

    for t in range(n_checkpoints):
        valid_cum = np.isfinite(cumulative_r2[t, :])
        valid_win = np.isfinite(windowed_r2[t, :])

        if np.sum(valid_cum) > 0:
            cumulative_dm_fraction[t] = np.mean(cumulative_r2[t, valid_cum] < R2_THRESHOLD)
        else:
            cumulative_dm_fraction[t] = np.nan

        if np.sum(valid_win) > 0:
            windowed_dm_fraction[t] = np.mean(windowed_r2[t, valid_win] < R2_THRESHOLD)
        else:
            windowed_dm_fraction[t] = np.nan

    # Track feature transitions (dark matter <-> well-behaved)
    # A feature "transitions to well-behaved" if it goes from R² < 0.9 to R² >= 0.9
    transitions_to_wellbehaved = 0
    transitions_to_darkmatter = 0
    persistent_darkmatter = 0
    persistent_wellbehaved = 0

    for i in range(n_features):
        trajectory = cumulative_r2[10:, i]  # Skip early checkpoints
        valid = np.isfinite(trajectory)
        if np.sum(valid) < 2:
            continue

        traj_valid = trajectory[valid]
        first_dm = traj_valid[0] < R2_THRESHOLD
        last_dm = traj_valid[-1] < R2_THRESHOLD

        if first_dm and not last_dm:
            transitions_to_wellbehaved += 1
        elif not first_dm and last_dm:
            transitions_to_darkmatter += 1
        elif first_dm and last_dm:
            persistent_darkmatter += 1
        else:
            persistent_wellbehaved += 1

    return {
        'sparsity': sparsity,
        'm_hidden': m_hidden,
        'seed': seed,
        'checkpoint_steps': checkpoint_steps.tolist(),
        'cumulative_dm_fraction': cumulative_dm_fraction.tolist(),
        'windowed_dm_fraction': windowed_dm_fraction.tolist(),
        'transitions_to_wellbehaved': transitions_to_wellbehaved,
        'transitions_to_darkmatter': transitions_to_darkmatter,
        'persistent_darkmatter': persistent_darkmatter,
        'persistent_wellbehaved': persistent_wellbehaved,
        'n_features': n_features,
        'final_mean_r2': float(np.nanmean(final_r2)),
        'final_dm_fraction': float(np.nanmean(final_r2 < R2_THRESHOLD)),
    }


def aggregate_by_sparsity(all_results):
    """Aggregate results by sparsity bucket."""
    aggregated = {bucket: defaultdict(list) for bucket in SPARSITY_BUCKETS}

    for r in all_results:
        for bucket, (low, high) in SPARSITY_BUCKETS.items():
            if low <= r['sparsity'] < high:
                aggregated[bucket]['cumulative_dm'].append(r['cumulative_dm_fraction'])
                aggregated[bucket]['windowed_dm'].append(r['windowed_dm_fraction'])
                aggregated[bucket]['checkpoint_steps'].append(r['checkpoint_steps'])
                aggregated[bucket]['transitions_to_wellbehaved'].append(r['transitions_to_wellbehaved'])
                aggregated[bucket]['transitions_to_darkmatter'].append(r['transitions_to_darkmatter'])
                aggregated[bucket]['persistent_darkmatter'].append(r['persistent_darkmatter'])
                aggregated[bucket]['persistent_wellbehaved'].append(r['persistent_wellbehaved'])
                break

    # Compute mean trajectories
    summary = {}
    for bucket, data in aggregated.items():
        if len(data['cumulative_dm']) == 0:
            continue

        # Stack and average
        cum_dm = np.array(data['cumulative_dm'])
        win_dm = np.array(data['windowed_dm'])

        summary[bucket] = {
            'n_experiments': len(data['cumulative_dm']),
            'mean_cumulative_dm': np.nanmean(cum_dm, axis=0).tolist(),
            'std_cumulative_dm': np.nanstd(cum_dm, axis=0).tolist(),
            'mean_windowed_dm': np.nanmean(win_dm, axis=0).tolist(),
            'std_windowed_dm': np.nanstd(win_dm, axis=0).tolist(),
            'checkpoint_steps': data['checkpoint_steps'][0],  # Same for all
            'total_transitions_to_wellbehaved': sum(data['transitions_to_wellbehaved']),
            'total_transitions_to_darkmatter': sum(data['transitions_to_darkmatter']),
            'total_persistent_darkmatter': sum(data['persistent_darkmatter']),
            'total_persistent_wellbehaved': sum(data['persistent_wellbehaved']),
        }

    return summary


def create_visualizations(summary, output_dir):
    """Create visualization plots."""

    # Plot 1: Dark matter evolution by sparsity (cumulative)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    colors = {'low': 'blue', 'medium': 'green', 'high': 'orange', 'extreme': 'red'}

    # Top left: Cumulative dark matter fraction
    ax = axes[0, 0]
    for bucket, data in summary.items():
        steps = np.array(data['checkpoint_steps'])
        mean_dm = np.array(data['mean_cumulative_dm'])
        std_dm = np.array(data['std_cumulative_dm'])

        valid = np.isfinite(mean_dm)
        ax.plot(steps[valid], mean_dm[valid], color=colors[bucket],
                label=f'{bucket} ({SPARSITY_BUCKETS[bucket][0]:.1f}-{SPARSITY_BUCKETS[bucket][1]:.1f})',
                linewidth=2)
        ax.fill_between(steps[valid],
                       (mean_dm - std_dm)[valid],
                       (mean_dm + std_dm)[valid],
                       color=colors[bucket], alpha=0.2)

    ax.axhline(R2_THRESHOLD, color='gray', linestyle='--', alpha=0.5, label='R² threshold')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Dark Matter Fraction (Cumulative R² < 0.9)', fontsize=12)
    ax.set_title('Dark Matter Evolution (Cumulative R²)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Top right: Windowed dark matter fraction
    ax = axes[0, 1]
    for bucket, data in summary.items():
        steps = np.array(data['checkpoint_steps'])
        mean_dm = np.array(data['mean_windowed_dm'])
        std_dm = np.array(data['std_windowed_dm'])

        valid = np.isfinite(mean_dm)
        ax.plot(steps[valid], mean_dm[valid], color=colors[bucket],
                label=f'{bucket}', linewidth=2)
        ax.fill_between(steps[valid],
                       (mean_dm - std_dm)[valid],
                       (mean_dm + std_dm)[valid],
                       color=colors[bucket], alpha=0.2)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Dark Matter Fraction (Windowed R² < 0.9)', fontsize=12)
    ax.set_title('Dark Matter Evolution (Windowed R²)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Bottom left: Transition counts
    ax = axes[1, 0]
    buckets = list(summary.keys())
    x = np.arange(len(buckets))
    width = 0.2

    to_wb = [summary[b]['total_transitions_to_wellbehaved'] for b in buckets]
    to_dm = [summary[b]['total_transitions_to_darkmatter'] for b in buckets]
    persist_dm = [summary[b]['total_persistent_darkmatter'] for b in buckets]
    persist_wb = [summary[b]['total_persistent_wellbehaved'] for b in buckets]

    ax.bar(x - 1.5*width, to_wb, width, label='Transitioned to Well-behaved', color='green')
    ax.bar(x - 0.5*width, to_dm, width, label='Transitioned to Dark Matter', color='red')
    ax.bar(x + 0.5*width, persist_dm, width, label='Persistent Dark Matter', color='darkred')
    ax.bar(x + 1.5*width, persist_wb, width, label='Persistent Well-behaved', color='darkgreen')

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel('Sparsity Bucket', fontsize=12)
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title('Feature Transition Statistics', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom right: Transition ratios
    ax = axes[1, 1]
    ratios_to_wb = []
    ratios_to_dm = []
    for b in buckets:
        total = (summary[b]['total_transitions_to_wellbehaved'] +
                 summary[b]['total_transitions_to_darkmatter'] +
                 summary[b]['total_persistent_darkmatter'] +
                 summary[b]['total_persistent_wellbehaved'])
        if total > 0:
            ratios_to_wb.append(100 * summary[b]['total_transitions_to_wellbehaved'] / total)
            ratios_to_dm.append(100 * summary[b]['total_transitions_to_darkmatter'] / total)
        else:
            ratios_to_wb.append(0)
            ratios_to_dm.append(0)

    ax.bar(x - 0.2, ratios_to_wb, 0.4, label='% Transitioned to Well-behaved', color='green')
    ax.bar(x + 0.2, ratios_to_dm, 0.4, label='% Transitioned to Dark Matter', color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel('Sparsity Bucket', fontsize=12)
    ax.set_ylabel('Percentage of Features', fontsize=12)
    ax.set_title('Feature Transition Rates', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'dark_matter_evolution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'dark_matter_evolution.png'}")
    plt.close()


def process_file_wrapper(filepath):
    """Wrapper for multiprocessing - handles exceptions."""
    try:
        return analyze_file(filepath)
    except Exception as e:
        print(f"Error processing {filepath.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Dark Matter Evolution Analysis')
    parser.add_argument('--sample', type=int, default=None, help='Process only N files for testing')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Analysis 1: Dark Matter Evolution Over Training")
    print("=" * 60)

    # Get all h5 files
    files = sorted(INPUT_DIR.glob('n1024_m*.h5'))
    print(f"Found {len(files)} experiment files")

    if args.sample:
        files = files[:args.sample]
        print(f"Sampling {len(files)} files for testing")

    # Determine number of workers
    n_workers = args.workers if args.workers else min(cpu_count(), 32)
    print(f"Using {n_workers} parallel workers")

    # Process all files in parallel
    all_results = []
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(process_file_wrapper, files),
                           total=len(files), desc="Processing files"))
        all_results = [r for r in results if r is not None]

    print(f"Successfully processed {len(all_results)} files")

    # Aggregate by sparsity
    summary = aggregate_by_sparsity(all_results)

    # Print summary
    print("\n--- Summary by Sparsity Bucket ---")
    for bucket, data in summary.items():
        print(f"\n{bucket.upper()} sparsity ({SPARSITY_BUCKETS[bucket]}):")
        print(f"  Experiments: {data['n_experiments']}")
        print(f"  Final dark matter fraction: {data['mean_cumulative_dm'][-1]:.3f}")
        print(f"  Transitions to well-behaved: {data['total_transitions_to_wellbehaved']}")
        print(f"  Transitions to dark matter: {data['total_transitions_to_darkmatter']}")
        print(f"  Persistent dark matter: {data['total_persistent_darkmatter']}")
        print(f"  Persistent well-behaved: {data['total_persistent_wellbehaved']}")

    # Save results
    with open(RESULTS_DIR / 'dark_matter_temporal.json', 'w') as f:
        json.dump({
            'summary_by_sparsity': summary,
            'per_file_results': all_results[:100] if len(all_results) > 100 else all_results,  # Limit size
            'r2_threshold': R2_THRESHOLD,
        }, f, indent=2)
    print(f"\nSaved results to: {RESULTS_DIR / 'dark_matter_temporal.json'}")

    # Create visualizations
    print("\n--- Creating Visualizations ---")
    create_visualizations(summary, PLOTS_DIR)

    print("\n" + "=" * 60)
    print("Analysis 1 Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
