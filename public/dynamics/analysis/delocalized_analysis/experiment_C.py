#!/usr/bin/env python3
"""
Experiment C: Within-Cluster Variance of Kappa and Slope
=========================================================

This module implements Experiment Family C from the spectral superposition analysis plan.

Hypothesis: Features clustered by dominant eigenspace show different within-cluster variance
patterns in the lambda <= 1 (bulk) vs lambda > 1 (spiked) regimes.

Key Analyses:
-------------
C0. Define clusters via:
    - Index clusters: C_k = {i : k*(T-1,i) = k}
    - Eigenvalue bands: Group by lambda ranges
C1. Compute within-cluster statistics:
    - var_kappa: variance of kappa within cluster
    - var_slope: variance of D/N (= 1/kappa) within cluster
    - mean_concentration: mean of pmax within cluster
    - DM_frac: fraction of persistent DM features
C2. Analyze:
    - var_kappa vs lambda (expect higher variance in bulk)
    - var_kappa vs cluster size (stratified by regime)
    - var_kappa vs eigengap (expect highest at small gaps)

Expected Results:
- lambda > 1 clusters: low within-cluster variance
- Large lambda <= 1 clusters: high variance, especially at small gaps and low concentration

Author: Claude Code (Anthropic)
Date: 2026-01-22
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict

import numpy as np
from scipy import stats

from config import (
    AnalysisConfig, OUTPUT_DIR, N_FEATURES, N_CHECKPOINTS,
    DEFAULT_SPARSITY_BUCKETS
)
from data_loader import (
    RefinedSpectralLoader, BatchLoader, RunData,
    compute_r2_linear_fit
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ClusterStats:
    """
    Statistics for a single eigenspace cluster.

    Attributes
    ----------
    cluster_id : int
        Cluster identifier (eigenvalue index k)
    size : int
        Number of features in cluster
    lambda_value : float
        Eigenvalue of this cluster
    is_bulk : bool
        Whether lambda <= 1
    eigengap : float
        Gap to next eigenvalue (lambda_k - lambda_{k+1})

    Within-Cluster Statistics
    -------------------------
    kappa_mean : float
        Mean Rayleigh quotient
    kappa_var : float
        Variance of Rayleigh quotient
    slope_mean : float
        Mean of D/N = 1/kappa
    slope_var : float
        Variance of D/N
    pmax_mean : float
        Mean concentration (max projection)
    dm_frac : float
        Fraction of persistent DM features
    """
    cluster_id: int
    size: int
    lambda_value: float
    is_bulk: bool
    eigengap: float

    kappa_mean: float
    kappa_var: float
    slope_mean: float
    slope_var: float
    pmax_mean: float
    dm_frac: float


@dataclass
class BandStats:
    """
    Statistics for an eigenvalue band.

    Attributes
    ----------
    band_id : int
        Band identifier
    band_min : float
        Minimum eigenvalue (inclusive)
    band_max : float
        Maximum eigenvalue (exclusive)
    n_features : int
        Number of features in band
    n_clusters : int
        Number of eigenspace clusters in band
    """
    band_id: int
    band_min: float
    band_max: float
    n_features: int
    n_clusters: int

    kappa_mean: float
    kappa_var: float
    slope_mean: float
    slope_var: float
    pmax_mean: float
    dm_frac: float


# =============================================================================
# C0: Define Clusters
# =============================================================================

def define_index_clusters(
    dominant_eigenspace: np.ndarray,
    eigenvalues: np.ndarray,
    eigengaps: np.ndarray,
    config: AnalysisConfig
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
    """
    Define clusters based on dominant eigenspace at final checkpoint.

    Parameters
    ----------
    dominant_eigenspace : np.ndarray
        k*(t,i), shape (T, n)
    eigenvalues : np.ndarray
        lambda_k(t), shape (T, m)
    eigengaps : np.ndarray
        Eigengaps, shape (T, m-1)
    config : AnalysisConfig
        Configuration

    Returns
    -------
    Tuple[Dict[int, np.ndarray], Dict[int, Dict]]
        (cluster_members: k -> array of feature indices,
         cluster_info: k -> {lambda, is_bulk, eigengap})
    """
    T, n = dominant_eigenspace.shape
    m = eigenvalues.shape[1]

    # Use final checkpoint
    k_final = dominant_eigenspace[-1]  # (n,)
    lambda_final = eigenvalues[-1]     # (m,)
    gaps_final = eigengaps[-1]         # (m-1,)

    cluster_members = {}
    cluster_info = {}

    for k in range(m):
        members = np.where(k_final == k)[0]
        if len(members) > 0:
            cluster_members[k] = members

            # Gap to next eigenvalue (if exists)
            gap = gaps_final[k] if k < len(gaps_final) else 0.0

            cluster_info[k] = {
                'lambda': float(lambda_final[k]),
                'is_bulk': lambda_final[k] <= config.lambda_bulk_threshold,
                'eigengap': float(gap),
            }

    return cluster_members, cluster_info


def define_eigenvalue_bands(
    dominant_eigenspace: np.ndarray,
    eigenvalues: np.ndarray,
    config: AnalysisConfig
) -> Dict[int, np.ndarray]:
    """
    Define bands based on dominant eigenvalue at final checkpoint.

    Parameters
    ----------
    dominant_eigenspace : np.ndarray
        k*(t,i), shape (T, n)
    eigenvalues : np.ndarray
        lambda_k(t), shape (T, m)
    config : AnalysisConfig
        Configuration

    Returns
    -------
    Dict[int, np.ndarray]
        band_id -> array of feature indices
    """
    T, n = dominant_eigenspace.shape

    # Get lambda for each feature at final checkpoint
    k_final = dominant_eigenspace[-1]
    lambda_final = eigenvalues[-1]
    lambda_dom = lambda_final[k_final]  # (n,)

    bands = config.eigenvalue_bands
    band_members = {}

    for band_id, (band_min, band_max) in enumerate(bands):
        members = np.where((lambda_dom >= band_min) & (lambda_dom < band_max))[0]
        if len(members) > 0:
            band_members[band_id] = members

    return band_members


# =============================================================================
# C1: Compute Within-Cluster Statistics
# =============================================================================

def compute_cluster_stats(
    cluster_members: Dict[int, np.ndarray],
    cluster_info: Dict[int, Dict],
    kappa: np.ndarray,
    fractional_dims: np.ndarray,
    feature_norms: np.ndarray,
    pmax: np.ndarray,
    is_dm: np.ndarray
) -> List[ClusterStats]:
    """
    Compute within-cluster statistics for all clusters.

    Parameters
    ----------
    cluster_members : Dict[int, np.ndarray]
        k -> array of feature indices
    cluster_info : Dict[int, Dict]
        k -> {lambda, is_bulk, eigengap}
    kappa : np.ndarray
        Rayleigh quotient at final checkpoint, shape (n,)
    fractional_dims : np.ndarray
        D_i at final checkpoint, shape (n,)
    feature_norms : np.ndarray
        N_i at final checkpoint, shape (n,)
    pmax : np.ndarray
        Max projection at final checkpoint, shape (n,)
    is_dm : np.ndarray
        Dark matter labels, shape (n,)

    Returns
    -------
    List[ClusterStats]
        Statistics for each cluster
    """
    stats_list = []

    for k, members in cluster_members.items():
        if len(members) == 0:
            continue

        info = cluster_info[k]

        # Extract cluster values
        cluster_kappa = kappa[members]
        cluster_D = fractional_dims[members]
        cluster_N = feature_norms[members]
        cluster_pmax = pmax[members]
        cluster_dm = is_dm[members]

        # Compute slope D/N = 1/kappa (where N > 0)
        valid_N = cluster_N > 1e-10
        if np.sum(valid_N) > 0:
            slope = cluster_D[valid_N] / cluster_N[valid_N]
            slope_mean = np.mean(slope)
            slope_var = np.var(slope) if len(slope) > 1 else 0.0
        else:
            slope_mean = np.nan
            slope_var = np.nan

        stats_list.append(ClusterStats(
            cluster_id=k,
            size=len(members),
            lambda_value=info['lambda'],
            is_bulk=info['is_bulk'],
            eigengap=info['eigengap'],
            kappa_mean=float(np.mean(cluster_kappa)),
            kappa_var=float(np.var(cluster_kappa)) if len(cluster_kappa) > 1 else 0.0,
            slope_mean=float(slope_mean),
            slope_var=float(slope_var),
            pmax_mean=float(np.mean(cluster_pmax)),
            dm_frac=float(np.mean(cluster_dm)),
        ))

    return stats_list


def compute_band_stats(
    band_members: Dict[int, np.ndarray],
    cluster_members: Dict[int, np.ndarray],
    kappa: np.ndarray,
    fractional_dims: np.ndarray,
    feature_norms: np.ndarray,
    pmax: np.ndarray,
    is_dm: np.ndarray,
    bands: List[Tuple[float, float]]
) -> List[BandStats]:
    """
    Compute statistics for each eigenvalue band.

    Parameters
    ----------
    band_members : Dict[int, np.ndarray]
        band_id -> array of feature indices
    cluster_members : Dict[int, np.ndarray]
        k -> array of feature indices (for counting clusters per band)
    kappa : np.ndarray
        Rayleigh quotient at final checkpoint
    fractional_dims : np.ndarray
        D_i at final checkpoint
    feature_norms : np.ndarray
        N_i at final checkpoint
    pmax : np.ndarray
        Max projection at final checkpoint
    is_dm : np.ndarray
        Dark matter labels
    bands : List[Tuple[float, float]]
        Band definitions

    Returns
    -------
    List[BandStats]
        Statistics for each band
    """
    stats_list = []

    for band_id, (band_min, band_max) in enumerate(bands):
        if band_id not in band_members:
            continue

        members = band_members[band_id]

        # Count clusters in this band
        n_clusters = 0
        for k, cluster_mems in cluster_members.items():
            # Check if cluster overlaps with this band
            if len(np.intersect1d(cluster_mems, members)) > 0:
                n_clusters += 1

        # Extract values
        band_kappa = kappa[members]
        band_D = fractional_dims[members]
        band_N = feature_norms[members]
        band_pmax = pmax[members]
        band_dm = is_dm[members]

        # Compute slope
        valid_N = band_N > 1e-10
        if np.sum(valid_N) > 0:
            slope = band_D[valid_N] / band_N[valid_N]
            slope_mean = np.mean(slope)
            slope_var = np.var(slope) if len(slope) > 1 else 0.0
        else:
            slope_mean = np.nan
            slope_var = np.nan

        stats_list.append(BandStats(
            band_id=band_id,
            band_min=band_min,
            band_max=band_max,
            n_features=len(members),
            n_clusters=n_clusters,
            kappa_mean=float(np.mean(band_kappa)),
            kappa_var=float(np.var(band_kappa)) if len(band_kappa) > 1 else 0.0,
            slope_mean=float(slope_mean),
            slope_var=float(slope_var),
            pmax_mean=float(np.mean(band_pmax)),
            dm_frac=float(np.mean(band_dm)),
        ))

    return stats_list


# =============================================================================
# C2: Analysis Functions
# =============================================================================

def analyze_variance_vs_lambda(cluster_stats: List[ClusterStats]) -> Dict:
    """
    Analyze relationship between kappa variance and eigenvalue.

    Parameters
    ----------
    cluster_stats : List[ClusterStats]
        Cluster statistics

    Returns
    -------
    Dict
        Analysis results
    """
    if len(cluster_stats) < 5:
        return {'error': 'Insufficient clusters'}

    lambdas = np.array([s.lambda_value for s in cluster_stats])
    kappa_vars = np.array([s.kappa_var for s in cluster_stats])
    sizes = np.array([s.size for s in cluster_stats])

    # Filter valid (non-zero variance clusters)
    valid = (kappa_vars > 0) & (sizes >= 3)

    if np.sum(valid) < 5:
        return {'error': 'Insufficient valid clusters'}

    # Correlation
    rho, p = stats.spearmanr(lambdas[valid], kappa_vars[valid])

    # Compare bulk vs spiked
    bulk_mask = np.array([s.is_bulk for s in cluster_stats])
    bulk_valid = valid & bulk_mask
    spiked_valid = valid & ~bulk_mask

    results = {
        'lambda_vs_var_kappa_rho': float(rho),
        'lambda_vs_var_kappa_p': float(p),
        'n_clusters': int(np.sum(valid)),
    }

    if np.sum(bulk_valid) >= 3:
        results['bulk_var_kappa_mean'] = float(np.mean(kappa_vars[bulk_valid]))
        results['bulk_var_kappa_median'] = float(np.median(kappa_vars[bulk_valid]))
        results['n_bulk_clusters'] = int(np.sum(bulk_valid))

    if np.sum(spiked_valid) >= 3:
        results['spiked_var_kappa_mean'] = float(np.mean(kappa_vars[spiked_valid]))
        results['spiked_var_kappa_median'] = float(np.median(kappa_vars[spiked_valid]))
        results['n_spiked_clusters'] = int(np.sum(spiked_valid))

    # Compare bulk vs spiked
    if np.sum(bulk_valid) >= 3 and np.sum(spiked_valid) >= 3:
        stat, p = stats.mannwhitneyu(
            kappa_vars[bulk_valid],
            kappa_vars[spiked_valid],
            alternative='greater'
        )
        results['bulk_vs_spiked_p'] = float(p)
        results['bulk_higher_variance'] = bool(
            np.mean(kappa_vars[bulk_valid]) > np.mean(kappa_vars[spiked_valid])
        )

    return results


def analyze_variance_vs_size(cluster_stats: List[ClusterStats]) -> Dict:
    """
    Analyze relationship between kappa variance and cluster size.

    Parameters
    ----------
    cluster_stats : List[ClusterStats]
        Cluster statistics

    Returns
    -------
    Dict
        Analysis results
    """
    if len(cluster_stats) < 5:
        return {'error': 'Insufficient clusters'}

    sizes = np.array([s.size for s in cluster_stats])
    kappa_vars = np.array([s.kappa_var for s in cluster_stats])
    is_bulk = np.array([s.is_bulk for s in cluster_stats])

    valid = (kappa_vars > 0) & (sizes >= 3)

    results = {}

    # Overall correlation
    if np.sum(valid) >= 5:
        rho, p = stats.spearmanr(sizes[valid], kappa_vars[valid])
        results['size_vs_var_kappa_rho'] = float(rho)
        results['size_vs_var_kappa_p'] = float(p)

    # Stratified by regime
    bulk_valid = valid & is_bulk
    spiked_valid = valid & ~is_bulk

    if np.sum(bulk_valid) >= 5:
        rho, p = stats.spearmanr(sizes[bulk_valid], kappa_vars[bulk_valid])
        results['bulk_size_vs_var_rho'] = float(rho)
        results['bulk_size_vs_var_p'] = float(p)

    if np.sum(spiked_valid) >= 5:
        rho, p = stats.spearmanr(sizes[spiked_valid], kappa_vars[spiked_valid])
        results['spiked_size_vs_var_rho'] = float(rho)
        results['spiked_size_vs_var_p'] = float(p)

    return results


def analyze_variance_vs_gap(cluster_stats: List[ClusterStats]) -> Dict:
    """
    Analyze relationship between kappa variance and eigengap.

    Parameters
    ----------
    cluster_stats : List[ClusterStats]
        Cluster statistics

    Returns
    -------
    Dict
        Analysis results
    """
    if len(cluster_stats) < 5:
        return {'error': 'Insufficient clusters'}

    gaps = np.array([s.eigengap for s in cluster_stats])
    kappa_vars = np.array([s.kappa_var for s in cluster_stats])
    sizes = np.array([s.size for s in cluster_stats])
    is_bulk = np.array([s.is_bulk for s in cluster_stats])

    valid = (kappa_vars > 0) & (sizes >= 3) & (gaps > 0)

    results = {}

    # Overall correlation
    if np.sum(valid) >= 5:
        rho, p = stats.spearmanr(gaps[valid], kappa_vars[valid])
        results['gap_vs_var_kappa_rho'] = float(rho)
        results['gap_vs_var_kappa_p'] = float(p)
        results['gap_negatively_correlated'] = bool(rho < 0)

    # Just within bulk
    bulk_valid = valid & is_bulk
    if np.sum(bulk_valid) >= 5:
        rho, p = stats.spearmanr(gaps[bulk_valid], kappa_vars[bulk_valid])
        results['bulk_gap_vs_var_rho'] = float(rho)
        results['bulk_gap_vs_var_p'] = float(p)

    # Bin by gap quantiles
    if np.sum(valid) >= 10:
        gap_bins = np.percentile(gaps[valid], [0, 25, 50, 75, 100])
        bin_indices = np.digitize(gaps[valid], gap_bins[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, 3)

        bin_means = []
        for b in range(4):
            mask = bin_indices == b
            if np.sum(mask) > 0:
                bin_means.append(float(np.mean(kappa_vars[valid][mask])))
            else:
                bin_means.append(np.nan)

        results['var_kappa_by_gap_quartile'] = bin_means
        results['gap_quartile_edges'] = gap_bins.tolist()

    return results


def analyze_dm_concentration_by_lambda(cluster_stats: List[ClusterStats]) -> Dict:
    """
    Analyze where DM features concentrate in terms of lambda.

    Parameters
    ----------
    cluster_stats : List[ClusterStats]
        Cluster statistics

    Returns
    -------
    Dict
        Analysis results
    """
    if len(cluster_stats) < 5:
        return {'error': 'Insufficient clusters'}

    lambdas = np.array([s.lambda_value for s in cluster_stats])
    dm_fracs = np.array([s.dm_frac for s in cluster_stats])
    sizes = np.array([s.size for s in cluster_stats])
    is_bulk = np.array([s.is_bulk for s in cluster_stats])

    valid = sizes >= 3

    results = {}

    # Correlation between lambda and DM fraction
    if np.sum(valid) >= 5:
        rho, p = stats.spearmanr(lambdas[valid], dm_fracs[valid])
        results['lambda_vs_dm_frac_rho'] = float(rho)
        results['lambda_vs_dm_frac_p'] = float(p)

    # Bulk vs spiked DM rates
    bulk_valid = valid & is_bulk
    spiked_valid = valid & ~is_bulk

    if np.sum(bulk_valid) > 0:
        # Weighted by cluster size
        bulk_dm_rate = np.average(dm_fracs[bulk_valid], weights=sizes[bulk_valid])
        results['bulk_dm_rate'] = float(bulk_dm_rate)

    if np.sum(spiked_valid) > 0:
        spiked_dm_rate = np.average(dm_fracs[spiked_valid], weights=sizes[spiked_valid])
        results['spiked_dm_rate'] = float(spiked_dm_rate)

    return results


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment_C_single_file(
    data: RunData,
    config: AnalysisConfig
) -> Dict:
    """
    Run Experiment C for a single file.

    Parameters
    ----------
    data : RunData
        Run data
    config : AnalysisConfig
        Configuration

    Returns
    -------
    Dict
        Results dictionary
    """
    # Compute DM labels
    late_idx = config.get_late_window_indices(data.T)
    D_late = data.fractional_dims[late_idx]
    N_late = data.feature_norms[late_idx]
    r2_late = compute_r2_linear_fit(D_late, N_late)

    variance_N = np.var(N_late, axis=0)
    is_degenerate = variance_N < config.variance_floor
    is_dm = (r2_late < config.r2_threshold) & ~is_degenerate

    # Use final checkpoint for clustering
    T = data.T
    kappa_final = data.kappa_actual[-1]
    D_final = data.fractional_dims[-1]
    N_final = data.feature_norms[-1]
    pmax_final = data.max_projection[-1]

    # C0: Define clusters
    cluster_members, cluster_info = define_index_clusters(
        data.dominant_eigenspace,
        data.eigenvalues,
        data.eigengaps,
        config
    )

    band_members = define_eigenvalue_bands(
        data.dominant_eigenspace,
        data.eigenvalues,
        config
    )

    # C1: Compute cluster statistics
    cluster_stats = compute_cluster_stats(
        cluster_members,
        cluster_info,
        kappa_final,
        D_final,
        N_final,
        pmax_final,
        is_dm
    )

    band_stats = compute_band_stats(
        band_members,
        cluster_members,
        kappa_final,
        D_final,
        N_final,
        pmax_final,
        is_dm,
        config.eigenvalue_bands
    )

    # C2: Analysis
    var_vs_lambda = analyze_variance_vs_lambda(cluster_stats)
    var_vs_size = analyze_variance_vs_size(cluster_stats)
    var_vs_gap = analyze_variance_vs_gap(cluster_stats)
    dm_concentration = analyze_dm_concentration_by_lambda(cluster_stats)

    return {
        'run_id': data.run_id,
        'm_hidden': data.m_hidden,
        'sparsity': data.sparsity,
        'seed': data.seed,
        'n_clusters': len(cluster_stats),
        'n_bands': len(band_stats),
        'dm_rate': float(np.mean(is_dm)),
        'cluster_stats': [asdict(s) for s in cluster_stats],
        'band_stats': [asdict(s) for s in band_stats],
        'var_vs_lambda': var_vs_lambda,
        'var_vs_size': var_vs_size,
        'var_vs_gap': var_vs_gap,
        'dm_concentration': dm_concentration,
    }


def run_experiment_C(
    output_dir: Path = None,
    config: AnalysisConfig = None,
    max_files: int = None
) -> Dict:
    """
    Run full Experiment C across all files.

    Parameters
    ----------
    output_dir : Path, optional
        Output directory
    config : AnalysisConfig, optional
        Analysis configuration
    max_files : int, optional
        Maximum number of files to process

    Returns
    -------
    Dict
        Aggregated results
    """
    if config is None:
        config = AnalysisConfig()

    if output_dir is None:
        output_dir = OUTPUT_DIR / 'experiment_C'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Experiment C: Within-Cluster Variance")

    # Load data
    loader = RefinedSpectralLoader()
    files = list(loader.iter_files())

    if max_files:
        files = files[:max_files]

    logger.info(f"Processing {len(files)} files")

    # Process all files
    all_results = []
    for i, filepath in enumerate(files):
        try:
            data = loader.load_file(filepath=filepath)
            result = run_experiment_C_single_file(data, config)
            all_results.append(result)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(files)} files")

        except Exception as e:
            logger.warning(f"Error processing {filepath}: {e}")

    # Aggregate results
    aggregated = aggregate_experiment_C_results(all_results)

    # Save results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)

    logger.info(f"Experiment C complete. Results saved to {output_dir}")

    return aggregated


def aggregate_experiment_C_results(results: List[Dict]) -> Dict:
    """
    Aggregate results across all runs.

    Parameters
    ----------
    results : List[Dict]
        List of per-run results

    Returns
    -------
    Dict
        Aggregated statistics
    """
    if not results:
        return {'error': 'No results'}

    # Aggregate var vs lambda
    bulk_higher_counts = []
    lambda_vs_var_rhos = []
    gap_vs_var_rhos = []

    bulk_dm_rates = []
    spiked_dm_rates = []

    for r in results:
        vl = r.get('var_vs_lambda', {})
        if 'lambda_vs_var_kappa_rho' in vl:
            lambda_vs_var_rhos.append(vl['lambda_vs_var_kappa_rho'])
        if 'bulk_higher_variance' in vl:
            bulk_higher_counts.append(vl['bulk_higher_variance'])

        vg = r.get('var_vs_gap', {})
        if 'gap_vs_var_kappa_rho' in vg:
            gap_vs_var_rhos.append(vg['gap_vs_var_kappa_rho'])

        dc = r.get('dm_concentration', {})
        if 'bulk_dm_rate' in dc:
            bulk_dm_rates.append(dc['bulk_dm_rate'])
        if 'spiked_dm_rate' in dc:
            spiked_dm_rates.append(dc['spiked_dm_rate'])

    return {
        'n_runs': len(results),
        'var_vs_lambda': {
            'lambda_vs_var_rho_mean': float(np.nanmean(lambda_vs_var_rhos)) if lambda_vs_var_rhos else np.nan,
            'bulk_higher_variance_rate': float(np.mean(bulk_higher_counts)) if bulk_higher_counts else np.nan,
        },
        'var_vs_gap': {
            'gap_vs_var_rho_mean': float(np.nanmean(gap_vs_var_rhos)) if gap_vs_var_rhos else np.nan,
            'gap_negatively_correlated_expected': 'Smaller gaps -> higher variance',
        },
        'dm_concentration': {
            'bulk_dm_rate_mean': float(np.nanmean(bulk_dm_rates)) if bulk_dm_rates else np.nan,
            'spiked_dm_rate_mean': float(np.nanmean(spiked_dm_rates)) if spiked_dm_rates else np.nan,
            'dm_concentrates_in_bulk': bool(
                np.nanmean(bulk_dm_rates) > np.nanmean(spiked_dm_rates)
            ) if bulk_dm_rates and spiked_dm_rates else None,
        },
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Experiment C: Within-Cluster Variance')
    parser.add_argument('--max-files', type=int, default=None, help='Max files to process')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    results = run_experiment_C(output_dir=output_dir, max_files=args.max_files)

    print("\n" + "=" * 60)
    print("EXPERIMENT C SUMMARY")
    print("=" * 60)
    print(f"Runs processed: {results.get('n_runs', 0)}")

    vl = results.get('var_vs_lambda', {})
    print(f"Lambda vs Var correlation: {vl.get('lambda_vs_var_rho_mean', np.nan):.4f}")
    print(f"Bulk higher variance rate: {vl.get('bulk_higher_variance_rate', np.nan):.1%}")

    dc = results.get('dm_concentration', {})
    print(f"Bulk DM rate: {dc.get('bulk_dm_rate_mean', np.nan):.4f}")
    print(f"Spiked DM rate: {dc.get('spiked_dm_rate_mean', np.nan):.4f}")
