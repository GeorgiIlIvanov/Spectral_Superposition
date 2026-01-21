#!/usr/bin/env python3
"""
Analysis 2: Jump Detection using Robust Statistics

Detects "jumps" in feature behavior using the log-transformed Rayleigh quotient
x_i(t) = log(κ_i(t) + ε).

A jump is detected when the change in x exceeds a threshold:
    |Δx_i(t)| = |x_i(t) - x_i(t-1)| > z * σ_robust

where:
    - z = 4.0 (configurable threshold in robust sigma units)
    - σ_robust = 1.4826 * MAD (Median Absolute Deviation from median)
    - MAD = median(|Δx - median(Δx)|)

The factor 1.4826 makes σ_robust equivalent to standard deviation for Gaussian data.

Outputs:
--------
- jump_detection_results.json: Summary statistics
- per-file jump counts and locations
- Aggregated jump statistics by sparsity bucket
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import load_config, get_sparsity_bucket


def compute_robust_sigma(delta_x: np.ndarray, epsilon_mad: float = 1e-9) -> float:
    """
    Compute robust standard deviation using Median Absolute Deviation (MAD).

    σ_robust = 1.4826 * MAD
    MAD = median(|x - median(x)|)

    The factor 1.4826 makes σ_robust consistent with standard deviation for
    normally distributed data.

    Args:
        delta_x: Array of differences Δx_i(t)
        epsilon_mad: Small constant to prevent division by zero

    Returns:
        Robust standard deviation estimate
    """
    # Flatten and remove NaN/Inf
    flat = delta_x.flatten()
    valid = flat[np.isfinite(flat)]

    if len(valid) < 2:
        return np.nan

    median_val = np.median(valid)
    mad = np.median(np.abs(valid - median_val))

    # Convert MAD to robust sigma (equivalent to std for Gaussian)
    sigma_robust = 1.4826 * (mad + epsilon_mad)

    return sigma_robust


def detect_jumps(x: np.ndarray, z_threshold: float, epsilon_mad: float) -> dict:
    """
    Detect jumps in the log-transformed Rayleigh quotient time series.

    Args:
        x: Log-transformed Rayleigh quotients, shape (T, n)
        z_threshold: Threshold in robust sigma units (default 4.0)
        epsilon_mad: Small constant for MAD computation

    Returns:
        Dictionary containing jump statistics and locations
    """
    T, n = x.shape

    # Compute differences Δx_i(t) = x_i(t) - x_i(t-1)
    delta_x = np.diff(x, axis=0)  # Shape: (T-1, n)

    # Compute per-feature robust sigma
    sigma_per_feature = np.zeros(n)
    for i in range(n):
        sigma_per_feature[i] = compute_robust_sigma(delta_x[:, i], epsilon_mad)

    # Global robust sigma (across all features and times)
    sigma_global = compute_robust_sigma(delta_x, epsilon_mad)

    # Detect jumps using global threshold
    # A jump occurs when |Δx_i(t)| > z * σ_global
    threshold = z_threshold * sigma_global
    is_jump = np.abs(delta_x) > threshold  # Shape: (T-1, n)

    # Count jumps per feature
    jumps_per_feature = np.sum(is_jump, axis=0)  # Shape: (n,)

    # Count jumps per checkpoint
    jumps_per_checkpoint = np.sum(is_jump, axis=1)  # Shape: (T-1,)

    # Total jumps
    total_jumps = np.sum(is_jump)

    # Classify jump directions (positive = increase, negative = decrease)
    positive_jumps = np.sum((delta_x > threshold), axis=0)
    negative_jumps = np.sum((delta_x < -threshold), axis=0)

    # Find the largest jumps (most extreme deviations)
    max_jump_magnitudes = np.nanmax(np.abs(delta_x), axis=0)  # Max per feature
    overall_max_jump = np.nanmax(np.abs(delta_x))

    # Jump timing statistics
    # For each feature, find when its first and last jump occurred
    first_jump_time = np.full(n, np.nan)
    last_jump_time = np.full(n, np.nan)
    for i in range(n):
        jump_times = np.where(is_jump[:, i])[0]
        if len(jump_times) > 0:
            first_jump_time[i] = jump_times[0]
            last_jump_time[i] = jump_times[-1]

    # Classify features by hopping activity
    # Using robust sigma per feature
    active_hoppers = np.sum(sigma_per_feature > z_threshold * 0.5)  # Features with high volatility
    stable_features = np.sum(sigma_per_feature < z_threshold * 0.2)

    return {
        'total_jumps': int(total_jumps),
        'jumps_per_feature': jumps_per_feature.tolist(),
        'jumps_per_checkpoint': jumps_per_checkpoint.tolist(),
        'positive_jumps_per_feature': positive_jumps.tolist(),
        'negative_jumps_per_feature': negative_jumps.tolist(),
        'sigma_per_feature': sigma_per_feature.tolist(),
        'sigma_global': float(sigma_global),
        'threshold_used': float(threshold),
        'max_jump_per_feature': max_jump_magnitudes.tolist(),
        'overall_max_jump': float(overall_max_jump),
        'mean_jumps_per_feature': float(np.mean(jumps_per_feature)),
        'std_jumps_per_feature': float(np.std(jumps_per_feature)),
        'features_with_jumps': int(np.sum(jumps_per_feature > 0)),
        'features_without_jumps': int(np.sum(jumps_per_feature == 0)),
        'active_hoppers': int(active_hoppers),
        'stable_features': int(stable_features),
        'first_jump_time': first_jump_time.tolist(),
        'last_jump_time': last_jump_time.tolist(),
    }


def detect_late_window_jumps(x: np.ndarray, z_threshold: float, epsilon_mad: float,
                             late_window_fraction: float) -> dict:
    """
    Analyze jumps specifically in the late training window (last 20% of checkpoints).

    This helps identify if hopping persists late in training or settles down.
    """
    T, n = x.shape
    late_start = int(T * (1 - late_window_fraction))

    x_late = x[late_start:, :]
    delta_x_late = np.diff(x_late, axis=0)

    sigma_late = compute_robust_sigma(delta_x_late, epsilon_mad)
    threshold_late = z_threshold * sigma_late

    is_jump_late = np.abs(delta_x_late) > threshold_late
    late_jumps_per_feature = np.sum(is_jump_late, axis=0)

    # Compare to early window
    early_end = int(T * late_window_fraction)
    x_early = x[:early_end, :]
    delta_x_early = np.diff(x_early, axis=0)

    sigma_early = compute_robust_sigma(delta_x_early, epsilon_mad)
    threshold_early = z_threshold * sigma_early

    is_jump_early = np.abs(delta_x_early) > threshold_early
    early_jumps_per_feature = np.sum(is_jump_early, axis=0)

    # Compute ratio of late to early jumping
    with np.errstate(divide='ignore', invalid='ignore'):
        jump_ratio = np.where(
            early_jumps_per_feature > 0,
            late_jumps_per_feature / early_jumps_per_feature,
            np.where(late_jumps_per_feature > 0, np.inf, 1.0)
        )

    return {
        'late_window_start_idx': late_start,
        'late_total_jumps': int(np.sum(is_jump_late)),
        'early_total_jumps': int(np.sum(is_jump_early)),
        'late_sigma': float(sigma_late),
        'early_sigma': float(sigma_early),
        'sigma_ratio_late_over_early': float(sigma_late / (sigma_early + 1e-10)),
        'mean_jump_ratio': float(np.nanmean(jump_ratio[np.isfinite(jump_ratio)])),
        'features_more_active_late': int(np.sum(jump_ratio > 1.0)),
        'features_less_active_late': int(np.sum(jump_ratio < 1.0)),
    }


def process_file(filepath: Path, cfg: dict) -> dict:
    """Process a single file for jump detection."""
    # Load precomputed Rayleigh quotient data
    data = np.load(filepath)

    x = data['x']  # Log-transformed Rayleigh quotients (T, n)
    kappa = data['kappa']
    sparsity = float(data['sparsity'])
    m_hidden = int(data['m_hidden'])
    seed = int(data['seed'])
    checkpoint_steps = data['checkpoint_steps']

    # Detect jumps (full time series)
    jump_results = detect_jumps(x, cfg['z_threshold'], cfg['epsilon_mad'])

    # Detect late window jumps
    late_results = detect_late_window_jumps(
        x, cfg['z_threshold'], cfg['epsilon_mad'], cfg['late_window_fraction']
    )

    return {
        'filename': filepath.stem,
        'sparsity': sparsity,
        'm_hidden': m_hidden,
        'seed': seed,
        'n_checkpoints': x.shape[0],
        'n_features': x.shape[1],
        'jump_results': jump_results,
        'late_window_results': late_results,
    }


def aggregate_by_sparsity(all_results: list, cfg: dict) -> dict:
    """Aggregate jump detection results by sparsity bucket."""
    buckets = cfg['sparsity_buckets']
    aggregated = {bucket: defaultdict(list) for bucket in buckets}

    for r in all_results:
        bucket = get_sparsity_bucket(r['sparsity'], buckets)
        if bucket == 'unknown':
            continue

        jr = r['jump_results']
        lr = r['late_window_results']

        aggregated[bucket]['total_jumps'].append(jr['total_jumps'])
        aggregated[bucket]['mean_jumps_per_feature'].append(jr['mean_jumps_per_feature'])
        aggregated[bucket]['features_with_jumps'].append(jr['features_with_jumps'])
        aggregated[bucket]['sigma_global'].append(jr['sigma_global'])
        aggregated[bucket]['late_sigma'].append(lr['late_sigma'])
        aggregated[bucket]['early_sigma'].append(lr['early_sigma'])
        aggregated[bucket]['sigma_ratio'].append(lr['sigma_ratio_late_over_early'])

    # Compute summary statistics per bucket
    summary = {}
    for bucket, data in aggregated.items():
        if len(data['total_jumps']) == 0:
            continue

        summary[bucket] = {
            'n_experiments': len(data['total_jumps']),
            'mean_total_jumps': float(np.mean(data['total_jumps'])),
            'std_total_jumps': float(np.std(data['total_jumps'])),
            'mean_jumps_per_feature': float(np.mean(data['mean_jumps_per_feature'])),
            'mean_features_with_jumps': float(np.mean(data['features_with_jumps'])),
            'mean_sigma_global': float(np.mean(data['sigma_global'])),
            'mean_late_sigma': float(np.mean(data['late_sigma'])),
            'mean_early_sigma': float(np.mean(data['early_sigma'])),
            'mean_sigma_ratio': float(np.mean(data['sigma_ratio'])),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Detect jumps in Rayleigh quotient time series')
    parser.add_argument('--sample', type=int, default=None, help='Process only N files')
    args = parser.parse_args()

    cfg = load_config()

    print("=" * 70)
    print("Analysis 2: Jump Detection using Robust Statistics")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Z-threshold: {cfg['z_threshold']} sigma")
    print(f"  Late window fraction: {cfg['late_window_fraction']}")
    print(f"  MAD epsilon: {cfg['epsilon_mad']}")

    # Find precomputed Rayleigh quotient files
    per_file_dir = cfg['results_dir'] / 'per_file'
    files = sorted(per_file_dir.glob('*_rayleigh.npz'))

    if len(files) == 0:
        print("\nNo Rayleigh quotient files found. Run 01_rayleigh_quotients.py first.")
        return

    print(f"\nFound {len(files)} precomputed Rayleigh quotient files")

    if args.sample:
        files = files[:args.sample]
        print(f"Sampling {len(files)} files")

    # Process all files
    all_results = []
    for filepath in tqdm(files, desc="Detecting jumps"):
        try:
            result = process_file(filepath, cfg)
            all_results.append(result)
        except Exception as e:
            print(f"\nError processing {filepath.name}: {e}")
            continue

    print(f"\nSuccessfully processed {len(all_results)} files")

    # Aggregate by sparsity
    sparsity_summary = aggregate_by_sparsity(all_results, cfg)

    # Print summary
    print("\n--- Summary by Sparsity Bucket ---")
    for bucket, data in sparsity_summary.items():
        bucket_range = cfg['sparsity_buckets'][bucket]
        print(f"\n{bucket.upper()} sparsity ({bucket_range[0]:.2f}-{bucket_range[1]:.2f}):")
        print(f"  Experiments: {data['n_experiments']}")
        print(f"  Mean total jumps: {data['mean_total_jumps']:.1f} +/- {data['std_total_jumps']:.1f}")
        print(f"  Mean jumps per feature: {data['mean_jumps_per_feature']:.3f}")
        print(f"  Mean global σ: {data['mean_sigma_global']:.4f}")
        print(f"  Late/Early σ ratio: {data['mean_sigma_ratio']:.3f}")

    # Global statistics
    all_total_jumps = [r['jump_results']['total_jumps'] for r in all_results]
    all_sigma = [r['jump_results']['sigma_global'] for r in all_results]

    print("\n--- Global Statistics ---")
    print(f"  Mean total jumps across all files: {np.mean(all_total_jumps):.1f}")
    print(f"  Mean global σ across all files: {np.mean(all_sigma):.4f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'z_threshold': cfg['z_threshold'],
            'late_window_fraction': cfg['late_window_fraction'],
            'epsilon_mad': cfg['epsilon_mad'],
        },
        'n_files_processed': len(all_results),
        'summary_by_sparsity': sparsity_summary,
        'per_file_results': [
            {
                'filename': r['filename'],
                'sparsity': r['sparsity'],
                'total_jumps': r['jump_results']['total_jumps'],
                'mean_jumps_per_feature': r['jump_results']['mean_jumps_per_feature'],
                'features_with_jumps': r['jump_results']['features_with_jumps'],
                'sigma_global': r['jump_results']['sigma_global'],
                'late_sigma_ratio': r['late_window_results']['sigma_ratio_late_over_early'],
            }
            for r in all_results
        ],
    }

    output_path = cfg['results_dir'] / 'jump_detection_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {output_path}")

    print("\n" + "=" * 70)
    print("Analysis 2 Complete: Jump Detection")
    print("=" * 70)


if __name__ == '__main__':
    main()
