#!/usr/bin/env python3
"""
Experiments D-H: Global Projective Linearity and Global Regularity
===================================================================

This module implements Experiment Families D through H from the spectral superposition
analysis plan. These experiments investigate the "global projective linearity" hypothesis:
that there exists a time-varying global slope alpha(t) that relates D and N cross-sectionally.

Experiments:
------------
D. Cross-sectional linearity audit per checkpoint
   - Fit alpha(t) from D(t,.) ~ alpha * N(t,.) (through origin)
   - Record R2_cross, dispersion of s_i = D/N

E. Does DM slope match lambda > 1 slope?
   - Compare slopes across feature subsets

F. What sets alpha(t)? (Global spectral functional)
   - Test predictors: alpha_pred = m / sum_k lambda_k, or 1 / median(kappa)

G. Universal diffusion analysis
   - For DM features, examine distribution of kappa and spectral similarity

H. Factor out alpha(t) to reconcile linearity
   - Normalize by alpha(t) and check if DM becomes stable

Author: Claude Code (Anthropic)
Date: 2026-01-22
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.special import rel_entr

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
# D: Cross-Sectional Linearity Audit
# =============================================================================

@dataclass
class CrossSectionalFit:
    """Results of cross-sectional D ~ alpha * N fit at a checkpoint."""
    checkpoint: int
    alpha: float          # Slope
    r2: float             # R^2 of fit
    dispersion_cv: float  # Coefficient of variation of s_i = D/N
    dispersion_iqr: float # IQR/median of s_i
    n_valid: int          # Number of valid features


def fit_cross_sectional_slope(
    D: np.ndarray,
    N: np.ndarray,
    return_residuals: bool = False
) -> Tuple[float, float, Optional[np.ndarray]]:
    """
    Fit D ~ alpha * N (through origin).

    Parameters
    ----------
    D : np.ndarray
        Fractional dimensions, shape (n,)
    N : np.ndarray
        Feature norms, shape (n,)
    return_residuals : bool
        Whether to return residuals

    Returns
    -------
    Tuple[float, float, Optional[np.ndarray]]
        (alpha, r2, residuals if requested)
    """
    # Filter valid
    valid = (N > 1e-10) & np.isfinite(D) & np.isfinite(N)
    if np.sum(valid) < 10:
        if return_residuals:
            return np.nan, np.nan, None
        return np.nan, np.nan

    D_valid = D[valid]
    N_valid = N[valid]

    # Through-origin fit: alpha = sum(D*N) / sum(N^2)
    alpha = np.sum(D_valid * N_valid) / np.sum(N_valid ** 2)

    # R^2
    D_pred = alpha * N_valid
    ss_res = np.sum((D_valid - D_pred) ** 2)
    ss_tot = np.sum((D_valid - np.mean(D_valid)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if return_residuals:
        residuals = np.full_like(D, np.nan)
        residuals[valid] = D_valid - D_pred
        return alpha, r2, residuals

    return alpha, r2


def compute_slope_dispersion(D: np.ndarray, N: np.ndarray) -> Tuple[float, float]:
    """
    Compute dispersion of s_i = D_i / N_i.

    Parameters
    ----------
    D : np.ndarray
        Fractional dimensions
    N : np.ndarray
        Feature norms

    Returns
    -------
    Tuple[float, float]
        (coefficient of variation, IQR/median)
    """
    valid = (N > 1e-10) & np.isfinite(D) & np.isfinite(N)
    if np.sum(valid) < 10:
        return np.nan, np.nan

    s = D[valid] / N[valid]

    cv = np.std(s) / (np.mean(s) + 1e-10)
    iqr = np.percentile(s, 75) - np.percentile(s, 25)
    iqr_ratio = iqr / (np.median(s) + 1e-10)

    return float(cv), float(iqr_ratio)


def run_cross_sectional_audit(
    data: RunData,
    config: AnalysisConfig
) -> Dict:
    """
    Run cross-sectional linearity audit for all checkpoints.

    Parameters
    ----------
    data : RunData
        Run data
    config : AnalysisConfig
        Configuration

    Returns
    -------
    Dict
        Audit results
    """
    T = data.T
    n = data.n

    # Compute DM labels
    late_idx = config.get_late_window_indices(T)
    D_late = data.fractional_dims[late_idx]
    N_late = data.feature_norms[late_idx]
    r2_late_per_feature = compute_r2_linear_fit(D_late, N_late)

    variance_N = np.var(N_late, axis=0)
    is_degenerate = variance_N < config.variance_floor
    is_dm = (r2_late_per_feature < config.r2_threshold) & ~is_degenerate

    # Get lambda_dom for each feature - compute manually
    T_full, n = data.dominant_eigenspace.shape
    lambda_dom = np.zeros((T_full, n))
    for t in range(T_full):
        for i in range(n):
            k = data.dominant_eigenspace[t, i]
            lambda_dom[t, i] = data.eigenvalues[t, k]  # (T, n)

    # Fit for each checkpoint
    fits_all = []
    fits_dm = []
    fits_wb = []
    fits_spiked = []
    fits_bulk = []

    for t in range(T):
        D_t = data.fractional_dims[t]
        N_t = data.feature_norms[t]
        pmax_t = data.max_projection[t]
        lambda_dom_t = lambda_dom[t]

        # All features
        alpha, r2 = fit_cross_sectional_slope(D_t, N_t)
        cv, iqr = compute_slope_dispersion(D_t, N_t)
        fits_all.append(CrossSectionalFit(
            checkpoint=t, alpha=alpha, r2=r2,
            dispersion_cv=cv, dispersion_iqr=iqr,
            n_valid=int(np.sum(np.isfinite(D_t) & (N_t > 1e-10)))
        ))

        # DM features
        alpha_dm, r2_dm = fit_cross_sectional_slope(D_t[is_dm], N_t[is_dm])
        fits_dm.append({'checkpoint': t, 'alpha': alpha_dm, 'r2': r2_dm})

        # Well-behaved features
        alpha_wb, r2_wb = fit_cross_sectional_slope(D_t[~is_dm], N_t[~is_dm])
        fits_wb.append({'checkpoint': t, 'alpha': alpha_wb, 'r2': r2_wb})

        # Spiked features (lambda > 1)
        spiked = lambda_dom_t > config.lambda_bulk_threshold
        alpha_sp, r2_sp = fit_cross_sectional_slope(D_t[spiked], N_t[spiked])
        fits_spiked.append({'checkpoint': t, 'alpha': alpha_sp, 'r2': r2_sp})

        # Bulk features (lambda <= 1)
        bulk = lambda_dom_t <= config.lambda_bulk_threshold
        alpha_bk, r2_bk = fit_cross_sectional_slope(D_t[bulk], N_t[bulk])
        fits_bulk.append({'checkpoint': t, 'alpha': alpha_bk, 'r2': r2_bk})

    # Extract time series
    alpha_all = np.array([f.alpha for f in fits_all])
    r2_all = np.array([f.r2 for f in fits_all])
    alpha_dm = np.array([f['alpha'] for f in fits_dm])
    alpha_spiked = np.array([f['alpha'] for f in fits_spiked])

    # Check if DM shares slope with all/spiked
    late_alpha_all = np.nanmean(alpha_all[late_idx])
    late_alpha_dm = np.nanmean(alpha_dm[late_idx])
    late_alpha_spiked = np.nanmean(alpha_spiked[late_idx])

    return {
        'fits_all': [asdict(f) for f in fits_all],
        'fits_dm': fits_dm,
        'fits_wb': fits_wb,
        'fits_spiked': fits_spiked,
        'fits_bulk': fits_bulk,
        'summary': {
            'late_alpha_all': float(late_alpha_all),
            'late_alpha_dm': float(late_alpha_dm),
            'late_alpha_spiked': float(late_alpha_spiked),
            'late_r2_all_mean': float(np.nanmean(r2_all[late_idx])),
            'dm_slope_ratio': float(late_alpha_dm / late_alpha_all) if late_alpha_all != 0 else np.nan,
            'spiked_slope_ratio': float(late_alpha_spiked / late_alpha_all) if late_alpha_all != 0 else np.nan,
        },
    }


# =============================================================================
# E: Does DM Slope Match Lambda > 1 Slope?
# =============================================================================

def test_slope_matching(audit_results: Dict, config: AnalysisConfig) -> Dict:
    """
    Test whether DM slope matches the lambda > 1 reference slope.

    Parameters
    ----------
    audit_results : Dict
        Results from cross-sectional audit
    config : AnalysisConfig
        Configuration

    Returns
    -------
    Dict
        Slope matching test results
    """
    fits_dm = audit_results['fits_dm']
    fits_spiked = audit_results['fits_spiked']

    alpha_dm = np.array([f['alpha'] for f in fits_dm])
    alpha_spiked = np.array([f['alpha'] for f in fits_spiked])

    # Valid comparisons
    valid = np.isfinite(alpha_dm) & np.isfinite(alpha_spiked) & (alpha_spiked != 0)

    if np.sum(valid) < 10:
        return {'error': 'Insufficient valid comparisons'}

    ratio = alpha_dm[valid] / alpha_spiked[valid]

    # Test if ratio is concentrated near 1
    results = {
        'ratio_mean': float(np.mean(ratio)),
        'ratio_std': float(np.std(ratio)),
        'ratio_median': float(np.median(ratio)),
        'near_unity': bool(np.abs(np.mean(ratio) - 1) < 0.1),
        'n_valid': int(np.sum(valid)),
    }

    # One-sample t-test for ratio = 1
    t_stat, p_value = stats.ttest_1samp(ratio, 1.0)
    results['t_stat'] = float(t_stat)
    results['p_value'] = float(p_value)

    return results


# =============================================================================
# F: What Sets Alpha(t)? Global Spectral Functional
# =============================================================================

def identify_alpha_predictor(data: RunData, audit_results: Dict) -> Dict:
    """
    Identify what predicts alpha(t).

    Candidates:
    - alpha_pred_trace = m / sum_k lambda_k
    - alpha_pred_kappa = 1 / median(kappa)

    Parameters
    ----------
    data : RunData
        Run data
    audit_results : Dict
        Cross-sectional audit results

    Returns
    -------
    Dict
        Predictor comparison results
    """
    T = data.T
    m = data.m

    # Actual alpha(t)
    alpha_actual = np.array([f['alpha'] for f in audit_results['fits_all']])

    # Predictor 1: m / trace(WW^T) = m / sum_k lambda_k
    trace = np.sum(data.eigenvalues, axis=1)  # (T,)
    alpha_pred_trace = m / (trace + 1e-10)

    # Predictor 2: 1 / median(kappa)
    median_kappa = np.median(data.kappa_actual, axis=1)  # (T,)
    alpha_pred_kappa = 1.0 / (median_kappa + 1e-10)

    # Predictor 3: 1 / mean(kappa)
    mean_kappa = np.mean(data.kappa_actual, axis=1)
    alpha_pred_mean_kappa = 1.0 / (mean_kappa + 1e-10)

    # Valid comparisons
    valid = np.isfinite(alpha_actual)

    if np.sum(valid) < 10:
        return {'error': 'Insufficient valid data'}

    results = {}

    # Correlation with trace predictor
    r_trace, p_trace = stats.pearsonr(alpha_actual[valid], alpha_pred_trace[valid])
    results['trace_predictor'] = {
        'correlation': float(r_trace),
        'p_value': float(p_trace),
        'rmse': float(np.sqrt(np.mean((alpha_actual[valid] - alpha_pred_trace[valid]) ** 2))),
    }

    # Correlation with median kappa predictor
    r_kappa, p_kappa = stats.pearsonr(alpha_actual[valid], alpha_pred_kappa[valid])
    results['median_kappa_predictor'] = {
        'correlation': float(r_kappa),
        'p_value': float(p_kappa),
        'rmse': float(np.sqrt(np.mean((alpha_actual[valid] - alpha_pred_kappa[valid]) ** 2))),
    }

    # Mean kappa predictor
    r_mean, p_mean = stats.pearsonr(alpha_actual[valid], alpha_pred_mean_kappa[valid])
    results['mean_kappa_predictor'] = {
        'correlation': float(r_mean),
        'p_value': float(p_mean),
        'rmse': float(np.sqrt(np.mean((alpha_actual[valid] - alpha_pred_mean_kappa[valid]) ** 2))),
    }

    # Best predictor
    correlations = {
        'trace': abs(r_trace),
        'median_kappa': abs(r_kappa),
        'mean_kappa': abs(r_mean),
    }
    results['best_predictor'] = max(correlations, key=correlations.get)

    return results


# =============================================================================
# G: Universal Diffusion Analysis
# =============================================================================

def analyze_universal_diffusion(
    data: RunData,
    config: AnalysisConfig
) -> Dict:
    """
    For DM features, analyze if they exhibit universal diffusion pattern.

    Universal diffusion = high entropy/PR but similar kappa and small pairwise JS.

    Parameters
    ----------
    data : RunData
        Run data
    config : AnalysisConfig
        Configuration

    Returns
    -------
    Dict
        Universal diffusion analysis results
    """
    T = data.T
    n = data.n

    # Compute DM labels
    late_idx = config.get_late_window_indices(T)
    D_late = data.fractional_dims[late_idx]
    N_late = data.feature_norms[late_idx]
    r2_late = compute_r2_linear_fit(D_late, N_late)

    variance_N = np.var(N_late, axis=0)
    is_degenerate = variance_N < config.variance_floor
    is_dm = (r2_late < config.r2_threshold) & ~is_degenerate

    n_dm = np.sum(is_dm)
    if n_dm < 10:
        return {'error': 'Insufficient DM features', 'n_dm': int(n_dm)}

    # Get DM features' properties at final checkpoint
    dm_indices = np.where(is_dm)[0]

    # Kappa distribution among DM
    kappa_dm = data.kappa_actual[-1, is_dm]
    kappa_dm_cv = np.std(kappa_dm) / (np.mean(kappa_dm) + 1e-10)

    # Entropy distribution among DM
    entropy_dm = data.projection_entropy[-1, is_dm]

    # PR distribution among DM
    pr_dm = data.participation_ratio[-1, is_dm]

    # Compute pairwise JS divergence (sample)
    np.random.seed(42)
    n_pairs = min(1000, n_dm * (n_dm - 1) // 2)
    js_values = []

    # Sample pairs
    for _ in range(n_pairs):
        i, j = np.random.choice(n_dm, 2, replace=False)
        p_i = data.projection_weights[-1, dm_indices[i]]
        p_j = data.projection_weights[-1, dm_indices[j]]

        # Normalize
        p_i = p_i / (np.sum(p_i) + 1e-10)
        p_j = p_j / (np.sum(p_j) + 1e-10)

        # JS divergence
        m_avg = 0.5 * (p_i + p_j)
        js = 0.5 * (np.sum(rel_entr(p_i + 1e-10, m_avg + 1e-10)) +
                    np.sum(rel_entr(p_j + 1e-10, m_avg + 1e-10)))
        js_values.append(np.sqrt(max(0, js)))

    js_array = np.array(js_values)

    return {
        'n_dm': int(n_dm),
        'kappa_dm': {
            'mean': float(np.mean(kappa_dm)),
            'std': float(np.std(kappa_dm)),
            'cv': float(kappa_dm_cv),
        },
        'entropy_dm': {
            'mean': float(np.mean(entropy_dm)),
            'std': float(np.std(entropy_dm)),
        },
        'pr_dm': {
            'mean': float(np.mean(pr_dm)),
            'std': float(np.std(pr_dm)),
        },
        'pairwise_js': {
            'mean': float(np.mean(js_array)),
            'std': float(np.std(js_array)),
            'median': float(np.median(js_array)),
            'max': float(np.max(js_array)),
        },
        'universal_diffusion': {
            # High entropy, low kappa dispersion, low JS
            'high_entropy': bool(np.mean(entropy_dm) > 1.0),
            'low_kappa_cv': bool(kappa_dm_cv < 0.5),
            'low_js': bool(np.mean(js_array) < 0.2),
        },
    }


# =============================================================================
# H: Factor Out Alpha(t) Normalization
# =============================================================================

def analyze_alpha_normalization(
    data: RunData,
    audit_results: Dict,
    config: AnalysisConfig
) -> Dict:
    """
    Normalize D by alpha(t) * N and check if DM becomes stable.

    r_i(t) = D_i(t) / (alpha(t) * N_i(t))

    If normalization collapses DM, nonlinearity is largely global scaling.

    Parameters
    ----------
    data : RunData
        Run data
    audit_results : Dict
        Cross-sectional audit results
    config : AnalysisConfig
        Configuration

    Returns
    -------
    Dict
        Normalization analysis results
    """
    T = data.T
    n = data.n

    # Get alpha(t)
    alpha_t = np.array([f['alpha'] for f in audit_results['fits_all']])  # (T,)

    # Compute normalized residual
    # r_i(t) = D_i(t) / (alpha(t) * N_i(t))
    alpha_expanded = alpha_t[:, np.newaxis]  # (T, 1)
    denom = alpha_expanded * data.feature_norms  # (T, n)

    # Avoid division by zero
    valid_denom = np.abs(denom) > 1e-10
    r = np.full((T, n), np.nan)
    r[valid_denom] = data.fractional_dims[valid_denom] / denom[valid_denom]

    # Compute DM labels before normalization
    late_idx = config.get_late_window_indices(T)
    D_late = data.fractional_dims[late_idx]
    N_late = data.feature_norms[late_idx]
    r2_before = compute_r2_linear_fit(D_late, N_late)

    variance_N = np.var(N_late, axis=0)
    is_degenerate = variance_N < config.variance_floor
    is_dm_before = (r2_before < config.r2_threshold) & ~is_degenerate

    # Compute stability of normalized r_i over late window
    r_late = r[late_idx]  # (late_window, n)

    # For each feature, compute coefficient of variation of r over time
    r_cv = np.nanstd(r_late, axis=0) / (np.nanmean(r_late, axis=0) + 1e-10)

    # A feature is "stable after normalization" if r_cv is small
    stability_threshold = 0.2
    is_stable = r_cv < stability_threshold

    # Check overlap between DM and stability
    dm_became_stable = is_dm_before & is_stable
    dm_still_unstable = is_dm_before & ~is_stable

    return {
        'n_dm_before': int(np.sum(is_dm_before)),
        'n_dm_became_stable': int(np.sum(dm_became_stable)),
        'n_dm_still_unstable': int(np.sum(dm_still_unstable)),
        'dm_stabilization_rate': float(np.sum(dm_became_stable) / (np.sum(is_dm_before) + 1e-10)),
        'interpretation': {
            'high_stabilization': 'Nonlinearity largely due to global alpha(t)',
            'low_stabilization': 'Feature-specific diffusion remains dominant',
        },
        'r_cv_stats': {
            'dm_mean': float(np.nanmean(r_cv[is_dm_before])) if np.sum(is_dm_before) > 0 else np.nan,
            'wb_mean': float(np.nanmean(r_cv[~is_dm_before])) if np.sum(~is_dm_before) > 0 else np.nan,
        },
    }


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiments_D_H_single_file(
    data: RunData,
    config: AnalysisConfig
) -> Dict:
    """
    Run Experiments D-H for a single file.

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
    # D: Cross-sectional audit
    audit_results = run_cross_sectional_audit(data, config)

    # E: Slope matching
    slope_matching = test_slope_matching(audit_results, config)

    # F: Alpha predictor
    alpha_predictor = identify_alpha_predictor(data, audit_results)

    # G: Universal diffusion
    universal_diffusion = analyze_universal_diffusion(data, config)

    # H: Normalization effect
    normalization = analyze_alpha_normalization(data, audit_results, config)

    return {
        'run_id': data.run_id,
        'm_hidden': data.m_hidden,
        'sparsity': data.sparsity,
        'seed': data.seed,
        'D_audit_summary': audit_results['summary'],
        'E_slope_matching': slope_matching,
        'F_alpha_predictor': alpha_predictor,
        'G_universal_diffusion': universal_diffusion,
        'H_normalization': normalization,
    }


def run_experiments_D_H(
    output_dir: Path = None,
    config: AnalysisConfig = None,
    max_files: int = None
) -> Dict:
    """
    Run full Experiments D-H across all files.

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
        output_dir = OUTPUT_DIR / 'experiments_D_H'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Experiments D-H: Global Projective Linearity")

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
            result = run_experiments_D_H_single_file(data, config)
            all_results.append(result)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(files)} files")

        except Exception as e:
            logger.warning(f"Error processing {filepath}: {e}")

    # Aggregate results
    aggregated = aggregate_experiments_D_H_results(all_results)

    # Save results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)

    logger.info(f"Experiments D-H complete. Results saved to {output_dir}")

    return aggregated


def aggregate_experiments_D_H_results(results: List[Dict]) -> Dict:
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

    # D: Audit summary
    late_r2_all = [r['D_audit_summary']['late_r2_all_mean'] for r in results]
    dm_slope_ratios = [r['D_audit_summary']['dm_slope_ratio'] for r in results
                       if np.isfinite(r['D_audit_summary'].get('dm_slope_ratio', np.nan))]

    # E: Slope matching
    ratio_means = [r['E_slope_matching'].get('ratio_mean', np.nan) for r in results]
    near_unity = [r['E_slope_matching'].get('near_unity', False) for r in results]

    # F: Best predictors
    best_predictors = [r['F_alpha_predictor'].get('best_predictor', 'unknown') for r in results]
    predictor_counts = {}
    for p in best_predictors:
        predictor_counts[p] = predictor_counts.get(p, 0) + 1

    # G: Universal diffusion
    kappa_cvs = [r['G_universal_diffusion'].get('kappa_dm', {}).get('cv', np.nan)
                 for r in results if 'kappa_dm' in r.get('G_universal_diffusion', {})]
    js_means = [r['G_universal_diffusion'].get('pairwise_js', {}).get('mean', np.nan)
                for r in results if 'pairwise_js' in r.get('G_universal_diffusion', {})]

    # H: Normalization
    stab_rates = [r['H_normalization'].get('dm_stabilization_rate', np.nan) for r in results]

    return {
        'n_runs': len(results),
        'D_cross_sectional': {
            'late_r2_mean': float(np.nanmean(late_r2_all)),
            'late_r2_std': float(np.nanstd(late_r2_all)),
            'dm_slope_ratio_mean': float(np.nanmean(dm_slope_ratios)) if dm_slope_ratios else np.nan,
        },
        'E_slope_matching': {
            'ratio_mean': float(np.nanmean(ratio_means)),
            'ratio_std': float(np.nanstd(ratio_means)),
            'near_unity_rate': float(np.mean(near_unity)),
        },
        'F_alpha_predictor': {
            'best_predictor_distribution': predictor_counts,
        },
        'G_universal_diffusion': {
            'kappa_cv_mean': float(np.nanmean(kappa_cvs)) if kappa_cvs else np.nan,
            'js_mean': float(np.nanmean(js_means)) if js_means else np.nan,
        },
        'H_normalization': {
            'stabilization_rate_mean': float(np.nanmean(stab_rates)),
            'stabilization_rate_std': float(np.nanstd(stab_rates)),
        },
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Experiments D-H: Global Linearity')
    parser.add_argument('--max-files', type=int, default=None, help='Max files to process')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    results = run_experiments_D_H(output_dir=output_dir, max_files=args.max_files)

    print("\n" + "=" * 60)
    print("EXPERIMENTS D-H SUMMARY")
    print("=" * 60)
    print(f"Runs processed: {results.get('n_runs', 0)}")

    print("\nD. Cross-sectional linearity:")
    print(f"  Late R^2 mean: {results['D_cross_sectional'].get('late_r2_mean', np.nan):.4f}")

    print("\nE. Slope matching (DM vs spiked):")
    print(f"  Ratio mean: {results['E_slope_matching'].get('ratio_mean', np.nan):.4f}")
    print(f"  Near unity rate: {results['E_slope_matching'].get('near_unity_rate', np.nan):.1%}")

    print("\nF. Alpha predictor:")
    print(f"  Distribution: {results['F_alpha_predictor'].get('best_predictor_distribution', {})}")

    print("\nH. Normalization effect:")
    print(f"  DM stabilization rate: {results['H_normalization'].get('stabilization_rate_mean', np.nan):.1%}")
