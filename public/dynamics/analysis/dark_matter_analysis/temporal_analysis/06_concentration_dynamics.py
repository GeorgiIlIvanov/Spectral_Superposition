#!/usr/bin/env python3
"""
Analysis 6: Eigenspace Concentration Dynamics

Track how eigenspace concentration and entropy evolve during training.
Key question: Do features become more concentrated over time (cleaner eigenspace assignments)?
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

R2_THRESHOLD = 0.9


def compute_r2_vectorized(x, y):
    """Vectorized R² computation for all features at once.

    Args:
        x: (T, n_features) array of feature norms
        y: (T, n_features) array of fractional dims

    Returns:
        r2: (n_features,) array of R² values
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

    # Invalidate results with insufficient data
    r2 = np.where((n_valid >= 5) & (ss_xx > 1e-10), r2, np.nan)
    return r2


def fast_pearsonr(x, y):
    """Fast Pearson correlation without scipy."""
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan
    x_v = x[valid]
    y_v = y[valid]
    x_centered = x_v - np.mean(x_v)
    y_centered = y_v - np.mean(y_v)
    ss_xx = np.sum(x_centered ** 2)
    ss_yy = np.sum(y_centered ** 2)
    ss_xy = np.sum(x_centered * y_centered)
    if ss_xx < 1e-10 or ss_yy < 1e-10:
        return np.nan
    return ss_xy / np.sqrt(ss_xx * ss_yy)


def compute_concentration_entropy(weights, U):
    """
    Compute eigenspace concentration and entropy for each feature.

    Returns:
        concentrations: (n,) - max projection fraction for each feature
        entropies: (n,) - Shannon entropy of projection distribution
    """
    m, n = weights.shape

    # Project features onto eigenspaces
    projections = U.T @ weights  # (m, n)
    proj_squared = projections ** 2

    # Normalize
    norms_sq = np.sum(weights ** 2, axis=0)  # (n,)
    norms_sq = np.maximum(norms_sq, 1e-10)
    p = proj_squared / norms_sq  # (m, n) - probability distribution over eigenspaces

    # Concentration = max probability
    concentrations = np.max(p, axis=0)

    # Entropy = -sum(p * log(p))
    p_safe = np.maximum(p, 1e-15)
    entropies = -np.sum(p_safe * np.log(p_safe), axis=0)

    return concentrations, entropies


def analyze_file(input_path, svd_path):
    """Analyze concentration dynamics for one experiment."""

    with h5py.File(input_path, 'r') as f:
        weights_all = f['weights'][:]  # (T, m, n)
        feature_norms = f['feature_norms'][:]  # (T, n)
        fractional_dims = f['fractional_dims'][:]  # (T, n)
        checkpoint_steps = f['checkpoint_steps'][:]
        sparsity = float(f.attrs['sparsity'])
        m_hidden = int(f.attrs['m_hidden'])
        seed = int(f.attrs['seed'])

    with h5py.File(svd_path, 'r') as f:
        U_all = f['U'][:]  # (T, m, m)
        eigenvalues_all = f['eigenvalues'][:]  # (T, m)

    n_checkpoints, m, n_features = weights_all.shape

    # Compute concentration and entropy at each checkpoint
    all_concentrations = np.zeros((n_checkpoints, n_features))
    all_entropies = np.zeros((n_checkpoints, n_features))

    for t in range(n_checkpoints):
        conc, ent = compute_concentration_entropy(weights_all[t], U_all[t])
        all_concentrations[t] = conc
        all_entropies[t] = ent

    # Compute mean statistics over time
    mean_concentration = np.mean(all_concentrations, axis=1)
    std_concentration = np.std(all_concentrations, axis=1)
    mean_entropy = np.mean(all_entropies, axis=1)
    std_entropy = np.std(all_entropies, axis=1)

    # Compute final R² for each feature - vectorized
    final_r2 = compute_r2_vectorized(feature_norms, fractional_dims)

    # Correlation between concentration and R²
    valid_r2 = np.isfinite(final_r2)
    final_concentrations = all_concentrations[-1]
    final_entropies = all_entropies[-1]

    if np.sum(valid_r2) > 10:
        conc_r2_corr = fast_pearsonr(final_concentrations[valid_r2], final_r2[valid_r2])
        ent_r2_corr = fast_pearsonr(final_entropies[valid_r2], final_r2[valid_r2])
    else:
        conc_r2_corr, ent_r2_corr = np.nan, np.nan

    # Track concentration convergence
    # Does concentration stabilize?
    early_conc = all_concentrations[5:15, :]
    late_conc = all_concentrations[-10:, :]
    conc_increase = np.mean(late_conc, axis=0) - np.mean(early_conc, axis=0)
    fraction_increased = np.mean(conc_increase > 0.01)

    # Concentration trajectory for dark matter vs well-behaved
    is_dark_matter = final_r2 < R2_THRESHOLD
    is_wellbehaved = final_r2 >= R2_THRESHOLD

    dm_conc_trajectory = np.mean(all_concentrations[:, is_dark_matter], axis=1) if np.sum(is_dark_matter) > 0 else np.full(n_checkpoints, np.nan)
    wb_conc_trajectory = np.mean(all_concentrations[:, is_wellbehaved], axis=1) if np.sum(is_wellbehaved) > 0 else np.full(n_checkpoints, np.nan)

    return {
        'sparsity': sparsity,
        'm_hidden': m_hidden,
        'seed': seed,
        'checkpoint_steps': checkpoint_steps.tolist(),
        'mean_concentration': mean_concentration.tolist(),
        'std_concentration': std_concentration.tolist(),
        'mean_entropy': mean_entropy.tolist(),
        'std_entropy': std_entropy.tolist(),
        'dm_conc_trajectory': dm_conc_trajectory.tolist(),
        'wb_conc_trajectory': wb_conc_trajectory.tolist(),
        'conc_r2_correlation': float(conc_r2_corr) if np.isfinite(conc_r2_corr) else None,
        'entropy_r2_correlation': float(ent_r2_corr) if np.isfinite(ent_r2_corr) else None,
        'fraction_concentration_increased': float(fraction_increased),
        'final_mean_concentration': float(np.mean(final_concentrations)),
        'final_mean_entropy': float(np.mean(final_entropies)),
        'n_features': n_features,
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
    """Aggregate concentration dynamics results."""

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
                aggregated[bucket]['mean_concentration'].append(r['mean_concentration'])
                aggregated[bucket]['mean_entropy'].append(r['mean_entropy'])
                aggregated[bucket]['dm_conc_trajectory'].append(r['dm_conc_trajectory'])
                aggregated[bucket]['wb_conc_trajectory'].append(r['wb_conc_trajectory'])
                aggregated[bucket]['conc_r2_correlation'].append(r['conc_r2_correlation'])
                aggregated[bucket]['fraction_increased'].append(r['fraction_concentration_increased'])
                break

    summary = {}
    for bucket, data in aggregated.items():
        if not data['mean_concentration']:
            continue

        conc = np.array(data['mean_concentration'])
        ent = np.array(data['mean_entropy'])
        dm_conc = np.array(data['dm_conc_trajectory'])
        wb_conc = np.array(data['wb_conc_trajectory'])

        summary[bucket] = {
            'n_experiments': len(data['mean_concentration']),
            'checkpoints': all_results[0]['checkpoint_steps'],
            'mean_concentration': np.nanmean(conc, axis=0).tolist(),
            'std_concentration': np.nanstd(conc, axis=0).tolist(),
            'mean_entropy': np.nanmean(ent, axis=0).tolist(),
            'std_entropy': np.nanstd(ent, axis=0).tolist(),
            'dm_mean_concentration': np.nanmean(dm_conc, axis=0).tolist(),
            'wb_mean_concentration': np.nanmean(wb_conc, axis=0).tolist(),
            'mean_conc_r2_correlation': float(np.nanmean([x for x in data['conc_r2_correlation'] if x is not None])) if any(x is not None for x in data['conc_r2_correlation']) else None,
            'mean_fraction_increased': float(np.mean(data['fraction_increased'])),
        }

    return summary


def create_visualizations(summary, output_dir):
    """Create visualization plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {'low': 'blue', 'medium': 'green', 'high': 'orange', 'extreme': 'red'}

    # Panel 1: Mean concentration over training
    ax = axes[0, 0]
    for bucket, data in summary.items():
        steps = np.array(data['checkpoints'])
        mean_conc = np.array(data['mean_concentration'])
        std_conc = np.array(data['std_concentration'])

        ax.plot(steps, mean_conc, color=colors[bucket], label=bucket, linewidth=2)
        ax.fill_between(steps, mean_conc - std_conc, mean_conc + std_conc,
                       color=colors[bucket], alpha=0.2)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Mean Eigenspace Concentration', fontsize=12)
    ax.set_title('Eigenspace Concentration Over Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Mean entropy over training
    ax = axes[0, 1]
    for bucket, data in summary.items():
        steps = np.array(data['checkpoints'])
        mean_ent = np.array(data['mean_entropy'])
        std_ent = np.array(data['std_entropy'])

        ax.plot(steps, mean_ent, color=colors[bucket], label=bucket, linewidth=2)
        ax.fill_between(steps, mean_ent - std_ent, mean_ent + std_ent,
                       color=colors[bucket], alpha=0.2)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Mean Eigenspace Entropy', fontsize=12)
    ax.set_title('Eigenspace Entropy Over Training (lower = more concentrated)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Dark matter vs well-behaved concentration (high sparsity)
    ax = axes[1, 0]
    if 'high' in summary:
        data = summary['high']
        steps = np.array(data['checkpoints'])
        dm_conc = np.array(data['dm_mean_concentration'])
        wb_conc = np.array(data['wb_mean_concentration'])

        valid_dm = np.isfinite(dm_conc)
        valid_wb = np.isfinite(wb_conc)

        ax.plot(steps[valid_dm], dm_conc[valid_dm], 'r-', linewidth=2, label='Dark Matter (R²<0.9)')
        ax.plot(steps[valid_wb], wb_conc[valid_wb], 'g-', linewidth=2, label='Well-behaved (R²≥0.9)')

        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Mean Concentration', fontsize=12)
        ax.set_title('Concentration: Dark Matter vs Well-behaved (High Sparsity)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No high sparsity data', transform=ax.transAxes, ha='center')

    # Panel 4: Concentration-R² correlation and convergence
    ax = axes[1, 1]
    buckets = list(summary.keys())
    x = np.arange(len(buckets))

    correlations = [summary[b].get('mean_conc_r2_correlation', 0) or 0 for b in buckets]
    frac_increased = [summary[b].get('mean_fraction_increased', 0) * 100 for b in buckets]

    ax2 = ax.twinx()
    bars1 = ax.bar(x - 0.2, correlations, 0.4, label='Conc-R² Correlation', color='steelblue', alpha=0.7)
    bars2 = ax2.bar(x + 0.2, frac_increased, 0.4, label='% Features w/ Increased Conc', color='coral', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel('Sparsity Bucket', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12, color='steelblue')
    ax2.set_ylabel('% Features', fontsize=12, color='coral')
    ax.set_title('Concentration-Linearity Relationship', fontsize=14)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'concentration_dynamics.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'concentration_dynamics.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Concentration Dynamics Analysis')
    parser.add_argument('--sample', type=int, default=None, help='Process only N files for testing')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Analysis 6: Eigenspace Concentration Dynamics")
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

    summary = aggregate_results(all_results)

    print("\n--- Summary by Sparsity Bucket ---")
    for bucket, data in summary.items():
        print(f"\n{bucket.upper()} sparsity:")
        print(f"  Experiments: {data['n_experiments']}")
        print(f"  Final mean concentration: {data['mean_concentration'][-1]:.3f}")
        print(f"  Initial mean concentration: {data['mean_concentration'][0]:.3f}")
        if data['mean_conc_r2_correlation'] is not None:
            print(f"  Concentration-R² correlation: {data['mean_conc_r2_correlation']:.3f}")
        print(f"  % features with increased concentration: {data['mean_fraction_increased']*100:.1f}%")

    with open(RESULTS_DIR / 'concentration_dynamics.json', 'w') as f:
        json.dump({
            'summary_by_sparsity': summary,
        }, f, indent=2)
    print(f"\nSaved results to: {RESULTS_DIR / 'concentration_dynamics.json'}")

    print("\n--- Creating Visualizations ---")
    create_visualizations(summary, PLOTS_DIR)

    print("\n" + "=" * 60)
    print("Analysis 6 Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
