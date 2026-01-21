#!/usr/bin/env python3
"""
Analysis 5: Feature Trajectory Classification

Classify features by their temporal trajectory shapes in the (||W_i||², D_i) space.

Trajectory types:
1. Linear monotonic: D_i increases linearly with ||W_i||² throughout training
2. Curved stable: Nonlinear but stable trajectory (converges to a ray)
3. Oscillatory: Features that move back and forth in phase space
4. Transient: Features that change slope mid-training (switch rays)
5. Collapsed: Features that shrink to near-zero norm
"""

import h5py
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
from collections import defaultdict


INPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/temporal_analysis')
PLOTS_DIR = OUTPUT_DIR / 'plots'
RESULTS_DIR = OUTPUT_DIR / 'results'


def classify_trajectory(norms, dims):
    """
    Classify a feature's trajectory in phase space.

    Args:
        norms: (T,) array of ||W_i||² over checkpoints
        dims: (T,) array of D_i over checkpoints

    Returns:
        classification: str, one of 'linear', 'curved', 'oscillatory', 'transient', 'collapsed'
        metrics: dict of trajectory metrics
    """
    # Filter valid values
    valid = np.isfinite(norms) & np.isfinite(dims) & (norms > 1e-10)

    if np.sum(valid) < 10:
        return 'insufficient_data', {}

    norms_v = norms[valid]
    dims_v = dims[valid]

    # Check if collapsed (final norm near zero)
    if norms_v[-1] < 0.01:
        return 'collapsed', {'final_norm': float(norms_v[-1])}

    # Linear fit quality
    if np.std(norms_v) < 1e-10:
        return 'static', {}

    slope, intercept, r_val, _, _ = stats.linregress(norms_v, dims_v)
    r2 = r_val ** 2

    # Compute curvature (second derivative approximation)
    # Using finite differences on the ratio D/||W||²
    ratios = dims_v / norms_v
    if len(ratios) > 4:
        d_ratio = np.diff(ratios)
        d2_ratio = np.diff(d_ratio)
        curvature = np.mean(d2_ratio)
        curvature_std = np.std(d2_ratio)
    else:
        curvature = 0
        curvature_std = 0

    # Check for oscillations (sign changes in velocity)
    d_norms = np.diff(norms_v)
    d_dims = np.diff(dims_v)
    norm_sign_changes = np.sum(np.diff(np.sign(d_norms)) != 0)
    dim_sign_changes = np.sum(np.diff(np.sign(d_dims)) != 0)

    # Check for slope changes (transient behavior)
    # Compute local slope in windows
    window_size = max(5, len(norms_v) // 4)
    local_slopes = []
    for i in range(0, len(norms_v) - window_size, window_size // 2):
        window_norms = norms_v[i:i + window_size]
        window_dims = dims_v[i:i + window_size]
        if np.std(window_norms) > 1e-10:
            local_slope, _, _, _, _ = stats.linregress(window_norms, window_dims)
            local_slopes.append(local_slope)

    slope_variation = np.std(local_slopes) / (np.abs(np.mean(local_slopes)) + 1e-10) if local_slopes else 0

    metrics = {
        'r2': float(r2),
        'slope': float(slope),
        'curvature': float(curvature),
        'curvature_std': float(curvature_std),
        'norm_sign_changes': int(norm_sign_changes),
        'dim_sign_changes': int(dim_sign_changes),
        'slope_variation': float(slope_variation),
        'final_norm': float(norms_v[-1]),
        'final_dim': float(dims_v[-1]),
    }

    # Classification logic
    is_linear = r2 > 0.9 and slope_variation < 0.3
    is_oscillatory = norm_sign_changes > 3 or dim_sign_changes > 5
    is_transient = slope_variation > 0.5 and not is_oscillatory
    is_curved = abs(curvature) > 0.01 and not is_linear and not is_transient

    if is_linear:
        return 'linear', metrics
    elif is_oscillatory:
        return 'oscillatory', metrics
    elif is_transient:
        return 'transient', metrics
    elif is_curved:
        return 'curved', metrics
    else:
        return 'mixed', metrics


def analyze_file(filepath):
    """Analyze trajectory types for all features in an experiment."""
    with h5py.File(filepath, 'r') as f:
        feature_norms = f['feature_norms'][:]  # (T, n)
        fractional_dims = f['fractional_dims'][:]  # (T, n)
        checkpoint_steps = f['checkpoint_steps'][:]
        sparsity = float(f.attrs['sparsity'])
        m_hidden = int(f.attrs['m_hidden'])
        seed = int(f.attrs['seed'])

    n_checkpoints, n_features = feature_norms.shape

    # Classify each feature
    classifications = defaultdict(int)
    metrics_by_class = defaultdict(list)
    sample_trajectories = {cls: [] for cls in ['linear', 'curved', 'oscillatory', 'transient', 'collapsed', 'mixed']}

    for i in range(n_features):
        cls, metrics = classify_trajectory(feature_norms[:, i], fractional_dims[:, i])
        classifications[cls] += 1

        if metrics:
            metrics_by_class[cls].append(metrics)

        # Store some sample trajectories
        if len(sample_trajectories.get(cls, [])) < 5:
            sample_trajectories[cls].append({
                'norms': feature_norms[:, i].tolist(),
                'dims': fractional_dims[:, i].tolist(),
                'index': i,
            })

    # Compute aggregate statistics
    total_valid = sum(classifications.values()) - classifications.get('insufficient_data', 0)
    class_fractions = {cls: count / total_valid for cls, count in classifications.items()
                      if cls != 'insufficient_data' and total_valid > 0}

    # Average metrics by class
    avg_metrics = {}
    for cls, metrics_list in metrics_by_class.items():
        if metrics_list:
            avg_metrics[cls] = {
                'mean_r2': float(np.mean([m['r2'] for m in metrics_list])),
                'mean_slope': float(np.mean([m['slope'] for m in metrics_list])),
                'mean_curvature': float(np.mean([m['curvature'] for m in metrics_list])),
                'n_features': len(metrics_list),
            }

    return {
        'sparsity': sparsity,
        'm_hidden': m_hidden,
        'seed': seed,
        'classifications': dict(classifications),
        'class_fractions': class_fractions,
        'avg_metrics': avg_metrics,
        'checkpoint_steps': checkpoint_steps.tolist(),
        'sample_trajectories': sample_trajectories,
        'n_features': n_features,
    }


def aggregate_results(all_results):
    """Aggregate classification results."""

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
                for cls, frac in r['class_fractions'].items():
                    aggregated[bucket][f'frac_{cls}'].append(frac)
                break

    summary = {}
    trajectory_types = ['linear', 'curved', 'oscillatory', 'transient', 'collapsed', 'mixed']

    for bucket, data in aggregated.items():
        if not data:
            continue

        summary[bucket] = {
            'n_experiments': len(data.get('frac_linear', [])),
        }
        for cls in trajectory_types:
            key = f'frac_{cls}'
            if key in data:
                summary[bucket][f'mean_frac_{cls}'] = float(np.mean(data[key]))
                summary[bucket][f'std_frac_{cls}'] = float(np.std(data[key]))

    return summary


def create_visualizations(summary, all_results, output_dir):
    """Create visualization plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors_cls = {
        'linear': 'green',
        'curved': 'blue',
        'oscillatory': 'red',
        'transient': 'orange',
        'collapsed': 'gray',
        'mixed': 'purple',
    }

    # Panel 1: Trajectory type distribution by sparsity
    ax = axes[0, 0]
    buckets = list(summary.keys())
    x = np.arange(len(buckets))
    width = 0.12
    trajectory_types = ['linear', 'curved', 'oscillatory', 'transient', 'collapsed', 'mixed']

    for i, cls in enumerate(trajectory_types):
        fracs = [summary[b].get(f'mean_frac_{cls}', 0) * 100 for b in buckets]
        offset = (i - len(trajectory_types)/2 + 0.5) * width
        ax.bar(x + offset, fracs, width, label=cls.capitalize(), color=colors_cls[cls])

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel('Sparsity Bucket', fontsize=12)
    ax.set_ylabel('Percentage of Features', fontsize=12)
    ax.set_title('Trajectory Type Distribution by Sparsity', fontsize=14)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Linear vs non-linear fraction
    ax = axes[0, 1]
    linear_fracs = [summary[b].get('mean_frac_linear', 0) * 100 for b in buckets]
    nonlinear_fracs = [100 - lf for lf in linear_fracs]

    ax.bar(x - 0.2, linear_fracs, 0.4, label='Linear (R² > 0.9)', color='green')
    ax.bar(x + 0.2, nonlinear_fracs, 0.4, label='Non-linear', color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel('Sparsity Bucket', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('Linear vs Non-linear Trajectories', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Sample trajectories from high sparsity
    ax = axes[1, 0]
    high_sparsity_results = [r for r in all_results if 0.7 <= r['sparsity'] < 0.95]
    if high_sparsity_results:
        sample_result = high_sparsity_results[0]
        steps = np.array(sample_result['checkpoint_steps'])

        for cls in ['linear', 'oscillatory', 'transient']:
            trajectories = sample_result['sample_trajectories'].get(cls, [])
            if trajectories:
                traj = trajectories[0]
                norms = np.array(traj['norms'])
                dims = np.array(traj['dims'])
                ax.plot(norms, dims, color=colors_cls[cls], linewidth=2, label=cls.capitalize())
                ax.scatter(norms[0], dims[0], color=colors_cls[cls], marker='o', s=50, zorder=5)
                ax.scatter(norms[-1], dims[-1], color=colors_cls[cls], marker='s', s=50, zorder=5)

        ax.set_xlabel('||W_i||²', fontsize=12)
        ax.set_ylabel('D_i', fontsize=12)
        ax.set_title('Sample Trajectories (○=start, □=end)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No high sparsity data', transform=ax.transAxes, ha='center')

    # Panel 4: Trajectory type vs R² (from metrics)
    ax = axes[1, 1]
    all_classes = []
    all_r2 = []

    for r in all_results:
        for cls, metrics in r['avg_metrics'].items():
            if 'mean_r2' in metrics:
                all_classes.append(cls)
                all_r2.append(metrics['mean_r2'])

    # Box plot alternative: bar chart of mean R² by class
    class_r2 = defaultdict(list)
    for cls, r2 in zip(all_classes, all_r2):
        class_r2[cls].append(r2)

    classes = list(class_r2.keys())
    mean_r2s = [np.mean(class_r2[c]) for c in classes]
    std_r2s = [np.std(class_r2[c]) for c in classes]
    colors_bar = [colors_cls.get(c, 'gray') for c in classes]

    x_cls = np.arange(len(classes))
    ax.bar(x_cls, mean_r2s, yerr=std_r2s, capsize=5, color=colors_bar, alpha=0.7)
    ax.axhline(0.9, color='red', linestyle='--', label='R² = 0.9')

    ax.set_xticks(x_cls)
    ax.set_xticklabels([c.capitalize() for c in classes], rotation=45, ha='right')
    ax.set_xlabel('Trajectory Type', fontsize=12)
    ax.set_ylabel('Mean R²', fontsize=12)
    ax.set_title('Linear Fit Quality by Trajectory Type', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_classification.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'trajectory_classification.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Trajectory Classification Analysis')
    parser.add_argument('--sample', type=int, default=None, help='Process only N files for testing')
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Analysis 5: Feature Trajectory Classification")
    print("=" * 60)

    files = sorted(INPUT_DIR.glob('n1024_m*.h5'))
    print(f"Found {len(files)} experiment files")

    if args.sample:
        files = files[:args.sample]
        print(f"Sampling {len(files)} files for testing")

    all_results = []
    for filepath in tqdm(files, desc="Processing files"):
        try:
            result = analyze_file(filepath)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")
            continue

    print(f"Successfully processed {len(all_results)} files")

    summary = aggregate_results(all_results)

    print("\n--- Summary by Sparsity Bucket ---")
    for bucket, data in summary.items():
        print(f"\n{bucket.upper()} sparsity ({data['n_experiments']} experiments):")
        for cls in ['linear', 'curved', 'oscillatory', 'transient', 'collapsed', 'mixed']:
            key = f'mean_frac_{cls}'
            if key in data:
                print(f"  {cls}: {data[key]*100:.1f}%")

    # Save results (without large trajectory data)
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items() if k != 'sample_trajectories'}
        save_results.append(save_r)

    with open(RESULTS_DIR / 'trajectory_classification.json', 'w') as f:
        json.dump({
            'summary_by_sparsity': summary,
            'per_file_results': save_results[:100] if len(save_results) > 100 else save_results,
        }, f, indent=2)
    print(f"\nSaved results to: {RESULTS_DIR / 'trajectory_classification.json'}")

    print("\n--- Creating Visualizations ---")
    create_visualizations(summary, all_results, PLOTS_DIR)

    print("\n" + "=" * 60)
    print("Analysis 5 Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
