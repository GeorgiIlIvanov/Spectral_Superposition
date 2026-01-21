#!/usr/bin/env python3
"""
Analysis 2: Eigenspace Stability and Hopping

Track how features migrate between eigenspace clusters during training.
Key question: Do features stabilize into eigenspaces, or do they persistently hop?

Key outputs:
- Hopping rate vs training step
- Distribution of stabilization times
- Correlation between hopping frequency and dark matter status
- Eigenspace migration heatmaps
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


INPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
SVD_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/svd_results')
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/temporal_analysis')
PLOTS_DIR = OUTPUT_DIR / 'plots'
RESULTS_DIR = OUTPUT_DIR / 'results'

R2_THRESHOLD = 0.9


def compute_eigenspace_assignments(weights, U):
    """
    Compute dominant eigenspace for each feature based on projections.

    Args:
        weights: (m, n) weight matrix
        U: (m, m) eigenvector matrix from SVD

    Returns:
        assignments: (n,) dominant eigenspace index for each feature
        concentrations: (n,) concentration in dominant eigenspace
    """
    m, n = weights.shape

    # Project features onto eigenspaces
    projections = U.T @ weights  # (m, n)
    proj_squared = projections ** 2  # (m, n)

    # Compute norms
    norms_sq = np.sum(weights ** 2, axis=0)  # (n,)
    norms_sq = np.maximum(norms_sq, 1e-10)

    # Normalized projections
    normalized_proj = proj_squared / norms_sq  # (m, n)

    # Dominant eigenspace
    assignments = np.argmax(normalized_proj, axis=0)  # (n,)

    # Concentration in dominant eigenspace
    concentrations = np.max(normalized_proj, axis=0)  # (n,)

    return assignments, concentrations


def analyze_file(input_path, svd_path):
    """Analyze eigenspace stability for a single experiment."""

    with h5py.File(input_path, 'r') as f:
        weights_all = f['weights'][:]  # (T, m, n)
        feature_norms = f['feature_norms'][:]  # (T, n)
        fractional_dims = f['fractional_dims'][:]  # (T, n)
        checkpoint_steps = f['checkpoint_steps'][:]
        sparsity = float(f.attrs['sparsity'])
        m_hidden = int(f.attrs['m_hidden'])
        seed = int(f.attrs['seed'])

    # Load SVD results (eigenvalues and eigenvectors at each checkpoint)
    with h5py.File(svd_path, 'r') as f:
        U_all = f['U'][:]  # (T, m, m)
        eigenvalues_all = f['eigenvalues'][:]  # (T, m)

    n_checkpoints, m, n_features = weights_all.shape

    # Compute eigenspace assignments at each checkpoint
    assignments = np.zeros((n_checkpoints, n_features), dtype=int)
    concentrations = np.zeros((n_checkpoints, n_features))

    for t in range(n_checkpoints):
        assignments[t], concentrations[t] = compute_eigenspace_assignments(
            weights_all[t], U_all[t]
        )

    # Compute hopping events
    # A "hop" occurs when a feature changes dominant eigenspace between consecutive checkpoints
    hops = np.diff(assignments, axis=0) != 0  # (T-1, n)
    hop_counts = np.sum(hops, axis=0)  # (n,) total hops per feature
    hopping_rate = np.mean(hops, axis=1)  # (T-1,) fraction hopping at each step

    # Compute stabilization time for each feature
    # Stabilization = last checkpoint after which no more hops occur
    stabilization_times = np.zeros(n_features)
    for i in range(n_features):
        feature_hops = hops[:, i]
        if np.sum(feature_hops) == 0:
            # Never hopped, stabilized from start
            stabilization_times[i] = 0
        else:
            # Find last hop
            last_hop_idx = np.max(np.where(feature_hops)[0])
            stabilization_times[i] = checkpoint_steps[last_hop_idx + 1]

    # Compute final R² (dark matter indicator)
    final_r2 = np.zeros(n_features)
    for i in range(n_features):
        x = feature_norms[:, i]
        y = fractional_dims[:, i]
        valid = np.isfinite(x) & np.isfinite(y) & (x > 1e-8)
        if np.sum(valid) < 5:
            final_r2[i] = np.nan
            continue
        slope, intercept, r_val, _, _ = stats.linregress(x[valid], y[valid])
        final_r2[i] = r_val ** 2

    # Correlation between hopping and dark matter
    valid_r2 = np.isfinite(final_r2)
    if np.sum(valid_r2) > 10:
        hop_dm_corr, hop_dm_pval = stats.pearsonr(
            hop_counts[valid_r2], final_r2[valid_r2]
        )
        conc_dm_corr, conc_dm_pval = stats.pearsonr(
            concentrations[-1, valid_r2], final_r2[valid_r2]
        )
    else:
        hop_dm_corr, hop_dm_pval = np.nan, np.nan
        conc_dm_corr, conc_dm_pval = np.nan, np.nan

    # Eigenspace migration matrix (where do features go?)
    # Count transitions from eigenspace i to eigenspace j
    migration_matrix = np.zeros((m_hidden, m_hidden), dtype=int)
    for t in range(n_checkpoints - 1):
        for i in range(n_features):
            from_eigen = assignments[t, i]
            to_eigen = assignments[t + 1, i]
            migration_matrix[from_eigen, to_eigen] += 1

    # Eigenvalue regime analysis
    final_eigenvalues = eigenvalues_all[-1]
    final_assignments = assignments[-1]

    # Features in lambda > 1 vs lambda < 1 regimes
    in_large_eigen = final_eigenvalues[final_assignments] > 1
    in_small_eigen = final_eigenvalues[final_assignments] <= 1

    large_eigen_hopcount = hop_counts[in_large_eigen].mean() if np.sum(in_large_eigen) > 0 else np.nan
    small_eigen_hopcount = hop_counts[in_small_eigen].mean() if np.sum(in_small_eigen) > 0 else np.nan

    return {
        'sparsity': sparsity,
        'm_hidden': m_hidden,
        'seed': seed,
        'checkpoint_steps': checkpoint_steps.tolist(),
        'hopping_rate': hopping_rate.tolist(),
        'mean_hop_count': float(np.mean(hop_counts)),
        'std_hop_count': float(np.std(hop_counts)),
        'max_hop_count': int(np.max(hop_counts)),
        'median_stabilization_time': float(np.median(stabilization_times)),
        'mean_stabilization_time': float(np.mean(stabilization_times)),
        'fraction_never_hopped': float(np.mean(hop_counts == 0)),
        'fraction_frequent_hoppers': float(np.mean(hop_counts > 5)),
        'hop_dm_correlation': float(hop_dm_corr) if np.isfinite(hop_dm_corr) else None,
        'hop_dm_pvalue': float(hop_dm_pval) if np.isfinite(hop_dm_pval) else None,
        'concentration_dm_correlation': float(conc_dm_corr) if np.isfinite(conc_dm_corr) else None,
        'concentration_dm_pvalue': float(conc_dm_pval) if np.isfinite(conc_dm_pval) else None,
        'large_eigen_mean_hops': float(large_eigen_hopcount) if np.isfinite(large_eigen_hopcount) else None,
        'small_eigen_mean_hops': float(small_eigen_hopcount) if np.isfinite(small_eigen_hopcount) else None,
        'mean_final_concentration': float(np.mean(concentrations[-1])),
        'n_features': n_features,
        # Don't store full migration matrix in JSON (too large)
    }


def aggregate_results(all_results):
    """Aggregate results across experiments."""

    # Group by sparsity bucket
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
                aggregated[bucket]['hopping_rate'].append(r['hopping_rate'])
                aggregated[bucket]['mean_hop_count'].append(r['mean_hop_count'])
                aggregated[bucket]['median_stabilization_time'].append(r['median_stabilization_time'])
                aggregated[bucket]['hop_dm_correlation'].append(r['hop_dm_correlation'])
                aggregated[bucket]['large_eigen_mean_hops'].append(r['large_eigen_mean_hops'])
                aggregated[bucket]['small_eigen_mean_hops'].append(r['small_eigen_mean_hops'])
                aggregated[bucket]['mean_final_concentration'].append(r['mean_final_concentration'])
                aggregated[bucket]['fraction_never_hopped'].append(r['fraction_never_hopped'])
                break

    summary = {}
    for bucket, data in aggregated.items():
        if len(data['hopping_rate']) == 0:
            continue

        hop_rates = np.array(data['hopping_rate'])

        summary[bucket] = {
            'n_experiments': len(data['hopping_rate']),
            'mean_hopping_rate': np.nanmean(hop_rates, axis=0).tolist(),
            'std_hopping_rate': np.nanstd(hop_rates, axis=0).tolist(),
            'mean_hop_count': float(np.nanmean(data['mean_hop_count'])),
            'mean_stabilization_time': float(np.nanmean(data['median_stabilization_time'])),
            'mean_hop_dm_correlation': float(np.nanmean([x for x in data['hop_dm_correlation'] if x is not None])) if any(x is not None for x in data['hop_dm_correlation']) else None,
            'mean_large_eigen_hops': float(np.nanmean([x for x in data['large_eigen_mean_hops'] if x is not None])) if any(x is not None for x in data['large_eigen_mean_hops']) else None,
            'mean_small_eigen_hops': float(np.nanmean([x for x in data['small_eigen_mean_hops'] if x is not None])) if any(x is not None for x in data['small_eigen_mean_hops']) else None,
            'mean_final_concentration': float(np.nanmean(data['mean_final_concentration'])),
            'mean_fraction_never_hopped': float(np.nanmean(data['fraction_never_hopped'])),
        }

    return summary


def create_visualizations(summary, all_results, output_dir):
    """Create visualization plots."""

    # Get checkpoint steps from first result
    checkpoint_steps = np.array(all_results[0]['checkpoint_steps'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {'low': 'blue', 'medium': 'green', 'high': 'orange', 'extreme': 'red'}

    # Panel 1: Hopping rate over training
    ax = axes[0, 0]
    for bucket, data in summary.items():
        steps = checkpoint_steps[1:]  # Hopping rate starts at step 1
        mean_rate = np.array(data['mean_hopping_rate'])
        std_rate = np.array(data['std_hopping_rate'])

        ax.plot(steps, mean_rate, color=colors[bucket], label=bucket, linewidth=2)
        ax.fill_between(steps, mean_rate - std_rate, mean_rate + std_rate,
                       color=colors[bucket], alpha=0.2)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Hopping Rate (fraction of features)', fontsize=12)
    ax.set_title('Eigenspace Hopping Rate During Training', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    # Panel 2: Total hops vs concentration
    ax = axes[0, 1]
    for bucket in summary.keys():
        bucket_results = [r for r in all_results
                        if bucket == next((b for b, (l, h) in
                                          [('low', (0.0, 0.3)), ('medium', (0.3, 0.7)),
                                           ('high', (0.7, 0.95)), ('extreme', (0.95, 1.0))]
                                          if l <= r['sparsity'] < h), None)]
        if not bucket_results:
            continue

        hop_counts = [r['mean_hop_count'] for r in bucket_results]
        concentrations = [r['mean_final_concentration'] for r in bucket_results]

        ax.scatter(hop_counts, concentrations, color=colors[bucket],
                  label=bucket, alpha=0.6, s=30)

    ax.set_xlabel('Mean Hop Count per Feature', fontsize=12)
    ax.set_ylabel('Mean Final Eigenspace Concentration', fontsize=12)
    ax.set_title('Hopping vs Eigenspace Concentration', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Lambda regime comparison
    ax = axes[1, 0]
    buckets = list(summary.keys())
    x = np.arange(len(buckets))
    width = 0.35

    large_hops = [summary[b].get('mean_large_eigen_hops', 0) or 0 for b in buckets]
    small_hops = [summary[b].get('mean_small_eigen_hops', 0) or 0 for b in buckets]

    ax.bar(x - width/2, large_hops, width, label='λ > 1 regime', color='steelblue')
    ax.bar(x + width/2, small_hops, width, label='λ ≤ 1 regime', color='coral')

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel('Sparsity Bucket', fontsize=12)
    ax.set_ylabel('Mean Hop Count', fontsize=12)
    ax.set_title('Eigenspace Hopping by Eigenvalue Regime', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Hop-DM correlation and stability metrics
    ax = axes[1, 1]

    buckets = list(summary.keys())
    x = np.arange(len(buckets))

    # Correlation values
    correlations = [abs(summary[b].get('mean_hop_dm_correlation', 0) or 0) for b in buckets]
    never_hopped = [summary[b].get('mean_fraction_never_hopped', 0) * 100 for b in buckets]

    ax2 = ax.twinx()

    bars1 = ax.bar(x - 0.2, correlations, 0.4, label='|Hop-R² Correlation|', color='purple', alpha=0.7)
    bars2 = ax2.bar(x + 0.2, never_hopped, 0.4, label='% Never Hopped', color='teal', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel('Sparsity Bucket', fontsize=12)
    ax.set_ylabel('|Correlation|', fontsize=12, color='purple')
    ax2.set_ylabel('% Features Never Hopped', fontsize=12, color='teal')
    ax.set_title('Hopping Correlates with Dark Matter Status', fontsize=14)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'eigenspace_stability.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'eigenspace_stability.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Eigenspace Stability Analysis')
    parser.add_argument('--sample', type=int, default=None, help='Process only N files for testing')
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Analysis 2: Eigenspace Stability and Hopping")
    print("=" * 60)

    # Get files that have SVD results
    svd_files = sorted(SVD_DIR.glob('svd_n1024_m*.h5'))
    print(f"Found {len(svd_files)} SVD files")

    if args.sample:
        svd_files = svd_files[:args.sample]
        print(f"Sampling {len(svd_files)} files for testing")

    all_results = []
    for svd_path in tqdm(svd_files, desc="Processing files"):
        # Find corresponding input file
        input_name = svd_path.stem.replace('svd_', '') + '.h5'
        input_path = INPUT_DIR / input_name

        if not input_path.exists():
            continue

        try:
            result = analyze_file(input_path, svd_path)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {svd_path.name}: {e}")
            continue

    print(f"Successfully processed {len(all_results)} files")

    # Aggregate results
    summary = aggregate_results(all_results)

    # Print summary
    print("\n--- Summary by Sparsity Bucket ---")
    for bucket, data in summary.items():
        print(f"\n{bucket.upper()} sparsity:")
        print(f"  Experiments: {data['n_experiments']}")
        print(f"  Mean hop count: {data['mean_hop_count']:.2f}")
        print(f"  Mean stabilization time: {data['mean_stabilization_time']:.0f}")
        print(f"  Hop-DM correlation: {data['mean_hop_dm_correlation']:.3f}" if data['mean_hop_dm_correlation'] else "  Hop-DM correlation: N/A")
        print(f"  Mean final concentration: {data['mean_final_concentration']:.3f}")
        print(f"  % never hopped: {data['mean_fraction_never_hopped']*100:.1f}%")
        if data['mean_large_eigen_hops'] is not None:
            print(f"  Mean hops (λ>1): {data['mean_large_eigen_hops']:.2f}")
        if data['mean_small_eigen_hops'] is not None:
            print(f"  Mean hops (λ≤1): {data['mean_small_eigen_hops']:.2f}")

    # Save results
    with open(RESULTS_DIR / 'eigenspace_stability.json', 'w') as f:
        json.dump({
            'summary_by_sparsity': summary,
            'per_file_results': all_results[:100] if len(all_results) > 100 else all_results,
        }, f, indent=2)
    print(f"\nSaved results to: {RESULTS_DIR / 'eigenspace_stability.json'}")

    # Create visualizations
    print("\n--- Creating Visualizations ---")
    create_visualizations(summary, all_results, PLOTS_DIR)

    print("\n" + "=" * 60)
    print("Analysis 2 Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
