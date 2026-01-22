#!/usr/bin/env python3
"""
Aggregate Spectral Results Script
==================================

This script aggregates the per-file refined spectral analysis results into
summary statistics and verification metrics.

Outputs:
--------
1. Global statistics across all configurations
2. Per-(m_hidden, sparsity) aggregated metrics
3. Verification of kappa_i = E[lambda] relationship
4. Cluster analysis summary

Usage:
------
    python aggregate_spectral_results.py [--input-dir refined_spectral_results]

Author: Claude Code (Anthropic)
Date: 2026-01-22
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import h5py
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_spectral_file(filepath: str) -> Dict[str, np.ndarray]:
    """Load a spectral results file."""
    return dict(np.load(filepath, allow_pickle=True))


def compute_kappa_verification(
    kappa_actual: np.ndarray,
    kappa_expected: np.ndarray,
    eigenvalues: np.ndarray,
    dominant_eigenspace: np.ndarray
) -> Dict[str, float]:
    """
    Verify the relationship kappa_i = E_{mu_i}[lambda].

    Parameters
    ----------
    kappa_actual : np.ndarray
        Actual Rayleigh quotient (T, n)
    kappa_expected : np.ndarray
        Expected eigenvalue from projections (T, n)
    eigenvalues : np.ndarray
        Eigenvalues (T, m)
    dominant_eigenspace : np.ndarray
        Dominant eigenspace indices (T, n)

    Returns
    -------
    Dict with verification metrics
    """
    T, n = kappa_actual.shape
    m = eigenvalues.shape[1]

    # Flatten for global statistics (exclude early training)
    # Use last 75% of checkpoints for stable statistics
    t_start = T // 4
    kappa_a = kappa_actual[t_start:].flatten()
    kappa_e = kappa_expected[t_start:].flatten()

    # Remove NaN/Inf values
    valid_mask = np.isfinite(kappa_a) & np.isfinite(kappa_e) & (kappa_a > 0)
    kappa_a = kappa_a[valid_mask]
    kappa_e = kappa_e[valid_mask]

    # Correlation between actual and expected kappa
    if len(kappa_a) > 10:
        pearson_r, pearson_p = stats.pearsonr(kappa_a, kappa_e)
        spearman_r, spearman_p = stats.spearmanr(kappa_a, kappa_e)

        # Mean absolute error
        mae = np.mean(np.abs(kappa_a - kappa_e))
        # Mean relative error
        mre = np.mean(np.abs(kappa_a - kappa_e) / kappa_a)
        # R-squared
        ss_res = np.sum((kappa_a - kappa_e) ** 2)
        ss_tot = np.sum((kappa_a - np.mean(kappa_a)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan
        mae = mre = r_squared = np.nan

    # Check kappa * lambda_dominant ≈ constant relationship
    dom_eigenvalues = np.zeros((T, n))
    for t in range(T):
        for i in range(n):
            k = dominant_eigenspace[t, i]
            if 0 <= k < m:
                dom_eigenvalues[t, i] = eigenvalues[t, k]

    kappa_times_lambda = kappa_actual[t_start:].flatten() * dom_eigenvalues[t_start:].flatten()
    valid_ktl = kappa_times_lambda[np.isfinite(kappa_times_lambda) & (kappa_times_lambda > 0)]

    return {
        'pearson_correlation': float(pearson_r),
        'pearson_pvalue': float(pearson_p),
        'spearman_correlation': float(spearman_r),
        'spearman_pvalue': float(spearman_p),
        'mean_absolute_error': float(mae),
        'mean_relative_error': float(mre),
        'r_squared': float(r_squared),
        'kappa_times_lambda_mean': float(np.mean(valid_ktl)) if len(valid_ktl) > 0 else np.nan,
        'kappa_times_lambda_std': float(np.std(valid_ktl)) if len(valid_ktl) > 0 else np.nan,
        'n_samples': int(len(kappa_a)),
    }


def compute_eigengap_statistics(eigengaps: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute eigengap statistics.

    Parameters
    ----------
    eigengaps : np.ndarray
        Eigengaps (T, m-1)

    Returns
    -------
    Dict with eigengap statistics
    """
    T, m_minus_1 = eigengaps.shape

    return {
        'eigengap_mean_over_time': np.mean(eigengaps, axis=1).astype(np.float32),  # (T,)
        'eigengap_max_over_time': np.max(eigengaps, axis=1).astype(np.float32),  # (T,)
        'eigengap_argmax_over_time': np.argmax(eigengaps, axis=1).astype(np.int16),  # (T,)
        'eigengap_mean_per_k': np.mean(eigengaps, axis=0).astype(np.float32),  # (m-1,)
    }


def compute_rotation_statistics(
    rotation_angles: np.ndarray,
    rotation_diag: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute rotation statistics.

    Parameters
    ----------
    rotation_angles : np.ndarray
        Rotation angles in radians (T-1, m)
    rotation_diag : np.ndarray
        Diagonal of rotation matrix (T-1, m)

    Returns
    -------
    Dict with rotation statistics
    """
    T_minus_1, m = rotation_angles.shape

    return {
        'rotation_angle_mean_over_time': np.mean(rotation_angles, axis=1).astype(np.float32),  # (T-1,)
        'rotation_angle_max_over_time': np.max(rotation_angles, axis=1).astype(np.float32),  # (T-1,)
        'rotation_angle_mean_per_k': np.mean(rotation_angles, axis=0).astype(np.float32),  # (m,)
        'rotation_stability': np.mean(rotation_diag, axis=1).astype(np.float32),  # (T-1,) - closer to 1 = more stable
    }


def compute_participation_statistics(
    participation_ratio: np.ndarray,
    n_significant: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute participation statistics.

    Parameters
    ----------
    participation_ratio : np.ndarray
        Participation ratio (T, n)
    n_significant : np.ndarray
        Number of significant eigenspaces per feature (T, n)

    Returns
    -------
    Dict with participation statistics
    """
    T, n = participation_ratio.shape

    return {
        'participation_ratio_mean': np.mean(participation_ratio, axis=1).astype(np.float32),  # (T,)
        'participation_ratio_std': np.std(participation_ratio, axis=1).astype(np.float32),  # (T,)
        'n_significant_mean': np.mean(n_significant, axis=1).astype(np.float32),  # (T,)
        'n_significant_std': np.std(n_significant, axis=1).astype(np.float32),  # (T,)
        'participation_ratio_distribution': np.percentile(
            participation_ratio[-1], [10, 25, 50, 75, 90]
        ).astype(np.float32),  # Final checkpoint percentiles
    }


def aggregate_all_files(input_dir: str, output_dir: str):
    """
    Aggregate all spectral result files.

    Parameters
    ----------
    input_dir : str
        Directory containing spectral result files
    output_dir : str
        Directory for output files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all spectral result files
    result_files = sorted(input_path.glob("spectral_*.npz"))
    logger.info(f"Found {len(result_files)} result files")

    if len(result_files) == 0:
        logger.error("No result files found!")
        return

    # Organize by (m_hidden, sparsity)
    results_by_config = defaultdict(list)

    # Global aggregation arrays
    all_verification_results = []
    all_kappa_correlations = []

    for i, filepath in enumerate(result_files):
        if i % 100 == 0:
            logger.info(f"Processing file {i+1}/{len(result_files)}")

        try:
            data = load_spectral_file(str(filepath))

            m_hidden = int(data['m_hidden'])
            sparsity = float(data['sparsity'])
            seed = int(data['seed'])

            config_key = (m_hidden, sparsity)

            # Extract verification metrics
            verification = compute_kappa_verification(
                data['kappa_actual'],
                data['kappa_expected'],
                data['eigenvalues'],
                data['dominant_eigenspace']
            )
            verification['m_hidden'] = m_hidden
            verification['sparsity'] = sparsity
            verification['seed'] = seed
            all_verification_results.append(verification)

            # Store correlation for this file
            all_kappa_correlations.append(verification['pearson_correlation'])

            # Eigengap statistics
            eigengap_stats = compute_eigengap_statistics(data['eigengaps'])

            # Rotation statistics
            rotation_stats = compute_rotation_statistics(
                data['rotation_angles'],
                data['rotation_matrix_diag']
            )

            # Participation statistics
            participation_stats = compute_participation_statistics(
                data['participation_ratio'],
                data['n_significant_eigenspaces']
            )

            # Store for per-config aggregation
            results_by_config[config_key].append({
                'seed': seed,
                'verification': verification,
                'eigengap': eigengap_stats,
                'rotation': rotation_stats,
                'participation': participation_stats,
                'cluster_kappa_mean': data['cluster_kappa_mean'],  # (T, m)
                'cluster_kappa_variance': data['cluster_kappa_variance'],  # (T, m)
                'cluster_size': data['cluster_size'],  # (T, m)
                'kappa_error': data['kappa_error'],  # (T, n)
                'kappa_relative_error': data['kappa_relative_error'],  # (T, n)
            })

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            continue

    # =========================================================
    # Save global verification results
    # =========================================================
    logger.info("Saving global verification results...")

    verification_summary = {
        'timestamp': datetime.now().isoformat(),
        'n_files': len(all_verification_results),
        'mean_pearson_correlation': float(np.nanmean([v['pearson_correlation'] for v in all_verification_results])),
        'std_pearson_correlation': float(np.nanstd([v['pearson_correlation'] for v in all_verification_results])),
        'mean_r_squared': float(np.nanmean([v['r_squared'] for v in all_verification_results])),
        'std_r_squared': float(np.nanstd([v['r_squared'] for v in all_verification_results])),
        'mean_relative_error': float(np.nanmean([v['mean_relative_error'] for v in all_verification_results])),
        'mean_kappa_times_lambda': float(np.nanmean([v['kappa_times_lambda_mean'] for v in all_verification_results])),
        'std_kappa_times_lambda': float(np.nanstd([v['kappa_times_lambda_mean'] for v in all_verification_results])),
    }

    with open(output_path / 'verification_summary.json', 'w') as f:
        json.dump(verification_summary, f, indent=2)

    # Save per-file verification details
    with open(output_path / 'verification_per_file.json', 'w') as f:
        json.dump(all_verification_results, f, indent=2)

    # =========================================================
    # Aggregate by (m_hidden, sparsity)
    # =========================================================
    logger.info("Aggregating by configuration...")

    # Get unique m_hidden and sparsity values
    m_values = sorted(set(k[0] for k in results_by_config.keys()))
    s_values = sorted(set(k[1] for k in results_by_config.keys()))

    n_m = len(m_values)
    n_s = len(s_values)

    logger.info(f"Configurations: {n_m} m_hidden values x {n_s} sparsity values")

    # Create index mappings
    m_to_idx = {m: i for i, m in enumerate(m_values)}
    s_to_idx = {s: i for i, s in enumerate(s_values)}

    # Initialize aggregation arrays
    # These are averaged over seeds
    agg_pearson_corr = np.full((n_m, n_s), np.nan, dtype=np.float32)
    agg_r_squared = np.full((n_m, n_s), np.nan, dtype=np.float32)
    agg_mre = np.full((n_m, n_s), np.nan, dtype=np.float32)
    agg_kappa_lambda = np.full((n_m, n_s), np.nan, dtype=np.float32)
    agg_participation = np.full((n_m, n_s), np.nan, dtype=np.float32)
    agg_n_significant = np.full((n_m, n_s), np.nan, dtype=np.float32)

    for (m_hidden, sparsity), file_results in results_by_config.items():
        m_idx = m_to_idx[m_hidden]
        s_idx = s_to_idx[sparsity]

        # Average over seeds
        pearson_vals = [r['verification']['pearson_correlation'] for r in file_results]
        r_sq_vals = [r['verification']['r_squared'] for r in file_results]
        mre_vals = [r['verification']['mean_relative_error'] for r in file_results]
        kl_vals = [r['verification']['kappa_times_lambda_mean'] for r in file_results]
        part_vals = [r['participation']['participation_ratio_mean'][-1] for r in file_results]  # Final checkpoint
        nsig_vals = [r['participation']['n_significant_mean'][-1] for r in file_results]

        agg_pearson_corr[m_idx, s_idx] = np.nanmean(pearson_vals)
        agg_r_squared[m_idx, s_idx] = np.nanmean(r_sq_vals)
        agg_mre[m_idx, s_idx] = np.nanmean(mre_vals)
        agg_kappa_lambda[m_idx, s_idx] = np.nanmean(kl_vals)
        agg_participation[m_idx, s_idx] = np.nanmean(part_vals)
        agg_n_significant[m_idx, s_idx] = np.nanmean(nsig_vals)

    # Save aggregated results
    np.savez_compressed(
        output_path / 'aggregated_by_config.npz',
        m_hidden_values=np.array(m_values),
        sparsity_values=np.array(s_values),
        pearson_correlation=agg_pearson_corr,
        r_squared=agg_r_squared,
        mean_relative_error=agg_mre,
        kappa_times_lambda=agg_kappa_lambda,
        participation_ratio_final=agg_participation,
        n_significant_eigenspaces_final=agg_n_significant,
    )

    # =========================================================
    # Save time-series aggregations
    # =========================================================
    logger.info("Computing time-series aggregations...")

    # Get checkpoint info from first file
    first_data = load_spectral_file(str(result_files[0]))
    checkpoint_steps = first_data['checkpoint_steps']
    n_checkpoints = len(checkpoint_steps)

    # Aggregate participation ratio over time (mean across all configs)
    all_participation_over_time = []
    all_n_significant_over_time = []

    for config_results in results_by_config.values():
        for file_result in config_results:
            all_participation_over_time.append(file_result['participation']['participation_ratio_mean'])
            all_n_significant_over_time.append(file_result['participation']['n_significant_mean'])

    np.savez_compressed(
        output_path / 'time_series_aggregation.npz',
        checkpoint_steps=checkpoint_steps,
        participation_ratio_mean=np.mean(all_participation_over_time, axis=0).astype(np.float32),
        participation_ratio_std=np.std(all_participation_over_time, axis=0).astype(np.float32),
        n_significant_mean=np.mean(all_n_significant_over_time, axis=0).astype(np.float32),
        n_significant_std=np.std(all_n_significant_over_time, axis=0).astype(np.float32),
    )

    logger.info(f"\nAggregation complete!")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"\nKey findings:")
    logger.info(f"  Mean kappa-expected correlation: {verification_summary['mean_pearson_correlation']:.4f}")
    logger.info(f"  Mean R-squared: {verification_summary['mean_r_squared']:.4f}")
    logger.info(f"  Mean relative error: {verification_summary['mean_relative_error']:.4f}")
    logger.info(f"  Mean kappa * lambda: {verification_summary['mean_kappa_times_lambda']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate spectral analysis results')
    parser.add_argument(
        '--input-dir', type=str, default='refined_spectral_results',
        help='Directory containing spectral result files'
    )
    parser.add_argument(
        '--output-dir', type=str, default='refined_spectral_results/aggregated',
        help='Directory for aggregated output files'
    )

    args = parser.parse_args()
    aggregate_all_files(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
