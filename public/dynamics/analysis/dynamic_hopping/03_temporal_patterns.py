#!/usr/bin/env python3
"""
Analysis 3: Temporal Patterns in Feature Hopping

Analyzes the temporal structure of feature hopping to identify patterns:

1. Trend Analysis: Is hopping increasing, decreasing, or stable over training?
2. Phase Detection: Are there distinct phases of hopping activity?
3. Feature Classification: Classify features by their hopping behavior
4. Correlation Analysis: Do features hop together or independently?

Key metrics:
- Temporal volatility: How much does x_i(t) = log(κ_i(t)) vary over time?
- Trend coefficient: Slope of σ(t) over time (increasing = more chaotic)
- Phase boundaries: Significant changes in hopping rate
- Hopping synchrony: Correlation between feature jump times

Outputs:
--------
- temporal_patterns_results.json: Classification and pattern statistics
- Feature classifications by hopping behavior
- Phase transition points
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from datetime import datetime
from collections import defaultdict
from scipy import stats
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import load_config, get_sparsity_bucket


def compute_rolling_volatility(x: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Compute rolling volatility (standard deviation) of x over time.

    Args:
        x: Time series of shape (T, n) or (T,)
        window_size: Window size for rolling computation

    Returns:
        Rolling volatility of shape (T - window_size + 1, ...)
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]

    T, n = x.shape
    n_windows = T - window_size + 1

    volatility = np.zeros((n_windows, n))

    for i in range(n_windows):
        window = x[i:i + window_size, :]
        volatility[i] = np.nanstd(window, axis=0)

    return volatility.squeeze() if n == 1 else volatility


def detect_phases(volatility: np.ndarray, n_features: int) -> dict:
    """
    Detect distinct phases in hopping activity based on volatility changes.

    Uses changepoint detection to find significant shifts in hopping behavior.
    """
    # Average volatility across features
    mean_volatility = np.nanmean(volatility, axis=1) if volatility.ndim > 1 else volatility

    # Smooth the volatility signal
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(mean_volatility, sigma=3)

    # Find peaks and troughs in volatility (phase boundaries)
    peaks, peak_props = find_peaks(smoothed, prominence=0.01, distance=5)
    troughs, trough_props = find_peaks(-smoothed, prominence=0.01, distance=5)

    # Compute trend (is volatility increasing or decreasing overall?)
    T = len(mean_volatility)
    if T > 5:
        time_idx = np.arange(T)
        valid = np.isfinite(mean_volatility)
        if np.sum(valid) > 5:
            slope, intercept, r_value, p_value, _ = stats.linregress(
                time_idx[valid], mean_volatility[valid]
            )
        else:
            slope, intercept, r_value, p_value = 0, 0, 0, 1
    else:
        slope, intercept, r_value, p_value = 0, 0, 0, 1

    # Classify overall trend
    if p_value < 0.05:
        if slope > 0.001:
            trend = 'increasing'
        elif slope < -0.001:
            trend = 'decreasing'
        else:
            trend = 'stable'
    else:
        trend = 'stable'

    return {
        'n_peaks': len(peaks),
        'n_troughs': len(troughs),
        'peak_indices': peaks.tolist(),
        'trough_indices': troughs.tolist(),
        'trend_slope': float(slope),
        'trend_intercept': float(intercept),
        'trend_r_squared': float(r_value ** 2),
        'trend_p_value': float(p_value),
        'trend_direction': trend,
        'mean_volatility_early': float(np.nanmean(mean_volatility[:len(mean_volatility)//3])),
        'mean_volatility_mid': float(np.nanmean(mean_volatility[len(mean_volatility)//3:2*len(mean_volatility)//3])),
        'mean_volatility_late': float(np.nanmean(mean_volatility[2*len(mean_volatility)//3:])),
    }


def classify_feature(x_feature: np.ndarray, cfg: dict) -> dict:
    """
    Classify a single feature by its hopping behavior.

    Categories:
    - stable: Very low volatility, predictable behavior
    - moderate: Some hopping but generally consistent
    - active: Significant hopping activity
    - extreme: Highly volatile, unpredictable

    Also classifies temporal pattern:
    - converging: Volatility decreases over time (settling down)
    - diverging: Volatility increases over time (becoming chaotic)
    - steady: Constant volatility level
    - episodic: Bursts of activity followed by quiet periods
    """
    T = len(x_feature)

    # Compute basic statistics
    mean_x = np.nanmean(x_feature)
    std_x = np.nanstd(x_feature)

    # Compute differences
    delta_x = np.diff(x_feature)
    abs_delta = np.abs(delta_x)

    # Robust volatility
    mad = np.nanmedian(abs_delta - np.nanmedian(abs_delta))
    sigma_robust = 1.4826 * (mad + cfg['epsilon_mad'])

    # Classify by volatility level
    thresholds = cfg['hopping_categories']
    if sigma_robust < thresholds['stable']:
        volatility_class = 'stable'
    elif sigma_robust < thresholds['moderate']:
        volatility_class = 'moderate'
    elif sigma_robust < thresholds['active']:
        volatility_class = 'active'
    else:
        volatility_class = 'extreme'

    # Classify temporal pattern
    # Split into thirds and compare volatility
    third = T // 3
    if third > 5:
        vol_early = np.nanstd(x_feature[:third])
        vol_mid = np.nanstd(x_feature[third:2*third])
        vol_late = np.nanstd(x_feature[2*third:])

        # Compute trend
        if vol_late < 0.7 * vol_early:
            temporal_pattern = 'converging'
        elif vol_late > 1.5 * vol_early:
            temporal_pattern = 'diverging'
        elif vol_mid > 1.3 * max(vol_early, vol_late):
            temporal_pattern = 'episodic'
        else:
            temporal_pattern = 'steady'
    else:
        temporal_pattern = 'unknown'
        vol_early = vol_mid = vol_late = np.nan

    return {
        'volatility_class': volatility_class,
        'temporal_pattern': temporal_pattern,
        'sigma_robust': float(sigma_robust),
        'mean_x': float(mean_x),
        'std_x': float(std_x),
        'volatility_early': float(vol_early) if not np.isnan(vol_early) else None,
        'volatility_mid': float(vol_mid) if not np.isnan(vol_mid) else None,
        'volatility_late': float(vol_late) if not np.isnan(vol_late) else None,
    }


def compute_hopping_synchrony(x: np.ndarray, z_threshold: float, epsilon_mad: float) -> dict:
    """
    Analyze whether features hop together (synchronously) or independently.

    Computes correlation of jump events across features.
    """
    T, n = x.shape

    # Compute jumps as binary events
    delta_x = np.diff(x, axis=0)

    # Per-feature robust sigma for thresholding
    sigma_global = 1.4826 * np.nanmedian(np.abs(delta_x - np.nanmedian(delta_x)))
    threshold = z_threshold * (sigma_global + epsilon_mad)

    is_jump = np.abs(delta_x) > threshold  # (T-1, n)

    # Count simultaneous jumps (jumps at same checkpoint)
    jumps_per_time = np.sum(is_jump, axis=1)  # (T-1,)

    # Expected if independent: P(k jumps) = Binomial(n, p) where p = P(jump)
    p_jump = np.mean(is_jump)
    expected_simultaneous = n * p_jump ** 2 * (T - 1)

    # Observed simultaneous (pairs of features jumping together)
    # Count pairs: for each time t, number of pairs = C(jumps_per_time[t], 2)
    observed_pairs = np.sum(jumps_per_time * (jumps_per_time - 1) / 2)

    # Synchrony ratio: observed / expected (>1 = synchronized, <1 = anti-synchronized)
    if expected_simultaneous > 0:
        synchrony_ratio = observed_pairs / expected_simultaneous
    else:
        synchrony_ratio = 1.0

    # Find peak synchrony times
    peak_synchrony_times = np.where(jumps_per_time > np.mean(jumps_per_time) + 2 * np.std(jumps_per_time))[0]

    return {
        'synchrony_ratio': float(synchrony_ratio),
        'mean_simultaneous_jumps': float(np.mean(jumps_per_time)),
        'max_simultaneous_jumps': int(np.max(jumps_per_time)),
        'peak_synchrony_times': peak_synchrony_times.tolist()[:20],  # Top 20
        'p_jump': float(p_jump),
        'interpretation': 'synchronized' if synchrony_ratio > 1.5 else ('independent' if synchrony_ratio > 0.7 else 'anti-synchronized'),
    }


def process_file(filepath: Path, cfg: dict) -> dict:
    """Process a single file for temporal pattern analysis."""
    data = np.load(filepath)

    x = data['x']  # Log-transformed Rayleigh quotients (T, n)
    kappa = data['kappa']
    sparsity = float(data['sparsity'])
    m_hidden = int(data['m_hidden'])
    seed = int(data['seed'])
    checkpoint_steps = data['checkpoint_steps']

    T, n = x.shape

    # Compute rolling volatility
    window_size = min(cfg['local_window_size'], T // 3)
    if window_size < 3:
        window_size = 3

    volatility = compute_rolling_volatility(x, window_size)

    # Detect phases
    phase_results = detect_phases(volatility, n)

    # Classify each feature
    feature_classes = []
    volatility_class_counts = defaultdict(int)
    temporal_pattern_counts = defaultdict(int)

    for i in range(n):
        fc = classify_feature(x[:, i], cfg)
        feature_classes.append(fc)
        volatility_class_counts[fc['volatility_class']] += 1
        temporal_pattern_counts[fc['temporal_pattern']] += 1

    # Compute hopping synchrony
    synchrony = compute_hopping_synchrony(x, cfg['z_threshold'], cfg['epsilon_mad'])

    # Summary statistics
    all_sigma = [fc['sigma_robust'] for fc in feature_classes]

    return {
        'filename': filepath.stem,
        'sparsity': sparsity,
        'm_hidden': m_hidden,
        'seed': seed,
        'n_checkpoints': T,
        'n_features': n,
        'phase_analysis': phase_results,
        'volatility_class_counts': dict(volatility_class_counts),
        'temporal_pattern_counts': dict(temporal_pattern_counts),
        'synchrony_analysis': synchrony,
        'mean_feature_sigma': float(np.mean(all_sigma)),
        'median_feature_sigma': float(np.median(all_sigma)),
        'feature_classifications': feature_classes,  # Keep for detailed analysis
    }


def aggregate_by_sparsity(all_results: list, cfg: dict) -> dict:
    """Aggregate temporal pattern results by sparsity bucket."""
    buckets = cfg['sparsity_buckets']
    aggregated = {bucket: defaultdict(list) for bucket in buckets}

    for r in all_results:
        bucket = get_sparsity_bucket(r['sparsity'], buckets)
        if bucket == 'unknown':
            continue

        aggregated[bucket]['trend_slopes'].append(r['phase_analysis']['trend_slope'])
        aggregated[bucket]['trend_directions'].append(r['phase_analysis']['trend_direction'])
        aggregated[bucket]['synchrony_ratios'].append(r['synchrony_analysis']['synchrony_ratio'])
        aggregated[bucket]['mean_sigmas'].append(r['mean_feature_sigma'])

        for vc, count in r['volatility_class_counts'].items():
            aggregated[bucket][f'volatility_{vc}'].append(count)
        for tp, count in r['temporal_pattern_counts'].items():
            aggregated[bucket][f'pattern_{tp}'].append(count)

    summary = {}
    for bucket, data in aggregated.items():
        if len(data['trend_slopes']) == 0:
            continue

        # Count trend directions
        trend_counts = defaultdict(int)
        for td in data['trend_directions']:
            trend_counts[td] += 1

        summary[bucket] = {
            'n_experiments': len(data['trend_slopes']),
            'mean_trend_slope': float(np.mean(data['trend_slopes'])),
            'trend_direction_counts': dict(trend_counts),
            'mean_synchrony_ratio': float(np.mean(data['synchrony_ratios'])),
            'mean_sigma': float(np.mean(data['mean_sigmas'])),
            'volatility_class_means': {
                k.replace('volatility_', ''): float(np.mean(v)) if v else 0
                for k, v in data.items() if k.startswith('volatility_')
            },
            'temporal_pattern_means': {
                k.replace('pattern_', ''): float(np.mean(v)) if v else 0
                for k, v in data.items() if k.startswith('pattern_')
            },
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze temporal patterns in feature hopping')
    parser.add_argument('--sample', type=int, default=None, help='Process only N files')
    args = parser.parse_args()

    cfg = load_config()

    print("=" * 70)
    print("Analysis 3: Temporal Patterns in Feature Hopping")
    print("=" * 70)

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
    for filepath in tqdm(files, desc="Analyzing temporal patterns"):
        try:
            result = process_file(filepath, cfg)
            # Remove per-feature classifications for the aggregate to save space
            result_summary = {k: v for k, v in result.items() if k != 'feature_classifications'}
            all_results.append(result_summary)
        except Exception as e:
            print(f"\nError processing {filepath.name}: {e}")
            import traceback
            traceback.print_exc()
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
        print(f"  Mean trend slope: {data['mean_trend_slope']:.6f}")
        print(f"  Trend directions: {data['trend_direction_counts']}")
        print(f"  Mean synchrony ratio: {data['mean_synchrony_ratio']:.3f}")
        print(f"  Volatility classes: {data['volatility_class_means']}")
        print(f"  Temporal patterns: {data['temporal_pattern_means']}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'z_threshold': cfg['z_threshold'],
            'local_window_size': cfg['local_window_size'],
            'hopping_categories': cfg['hopping_categories'],
        },
        'n_files_processed': len(all_results),
        'summary_by_sparsity': sparsity_summary,
        'per_file_results': all_results,
    }

    output_path = cfg['results_dir'] / 'temporal_patterns_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {output_path}")

    print("\n" + "=" * 70)
    print("Analysis 3 Complete: Temporal Patterns")
    print("=" * 70)


if __name__ == '__main__':
    main()
