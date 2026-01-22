#!/usr/bin/env python3
"""
Experiment B: Spectral Spread Metrics and Dark Matter Characterization
=======================================================================

This module implements Experiment Family B from the spectral superposition analysis plan.

Hypothesis: Dark matter features exhibit large spectral spread - their energy is distributed
across multiple eigenspaces rather than being concentrated in a single dominant one.

Key Quantities:
--------------
- Entropy: H_i(t) = -sum_k p_{ik}(t) * log(p_{ik}(t))
- Participation Ratio: PR_i(t) = 1 / sum_k p_{ik}^2 (effective number of eigenspaces)
- Spectral Variance: Var_lambda_i(t) = sum_k p_{ik} * lambda_k^2 - kappa_i^2
- Bulk Mass: m_{<=1}_i(t) = sum_{k: lambda_k <= 1} p_{ik}(t)
- Max Projection: pmax_i(t) = max_k p_{ik}(t)
- Spectral Drift: JS divergence between p_i(t) and p_i(t+1)

Key Tests:
----------
B1. Validate p_{ik} computation by checking kappa consistency
B2. Compare distributions of spread metrics for DM vs well-behaved features
B3. Predictive model: persistent_DM ~ H + PR + Var + drift + pmax + lambda_dom
B4. "Not merely lambda < 1" falsifier: check if spread separates DM within lambda <= 1 subset

Expected Results (if hypothesis B is driver):
- Persistent DM has higher entropy/PR/Var_lambda and/or higher drift
- These effects persist even within the lambda <= 1 subset

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
    compute_late_window_statistics, compute_r2_linear_fit
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# B0: Compute p_{ik} from SVD (Already in refined spectral results)
# =============================================================================

def validate_kappa_consistency(
    projection_weights: np.ndarray,
    eigenvalues: np.ndarray,
    kappa_actual: np.ndarray,
    sample_size: int = 100
) -> Dict:
    """
    Validate that sum_k p_{ik} * lambda_k matches kappa_actual.

    Parameters
    ----------
    projection_weights : np.ndarray
        p_{ik}(t), shape (T, n, m)
    eigenvalues : np.ndarray
        lambda_k(t), shape (T, m)
    kappa_actual : np.ndarray
        Actual Rayleigh quotient, shape (T, n)
    sample_size : int
        Number of samples to check

    Returns
    -------
    Dict
        Validation statistics
    """
    T, n, m = projection_weights.shape

    # Compute expected kappa
    kappa_expected = np.sum(
        projection_weights * eigenvalues[:, np.newaxis, :],
        axis=2
    )  # (T, n)

    # Compute relative error
    rel_error = np.abs(kappa_expected - kappa_actual) / (np.abs(kappa_actual) + 1e-10)

    # Sample statistics
    np.random.seed(42)
    sample_t = np.random.randint(0, T, sample_size)
    sample_i = np.random.randint(0, n, sample_size)
    sample_errors = rel_error[sample_t, sample_i]

    return {
        'mean_rel_error': float(np.mean(rel_error)),
        'max_rel_error': float(np.max(rel_error)),
        'median_rel_error': float(np.median(rel_error)),
        'sample_mean_error': float(np.mean(sample_errors)),
        'sample_max_error': float(np.max(sample_errors)),
        'passed': float(np.mean(rel_error)) < 0.01,  # 1% threshold
    }


# =============================================================================
# B2: Spectral Spread Metrics (Most already computed in refined spectral)
# =============================================================================

def compute_spectral_variance(
    projection_weights: np.ndarray,
    eigenvalues: np.ndarray,
    kappa: np.ndarray
) -> np.ndarray:
    """
    Compute spectral variance: Var_lambda_i = sum_k p_{ik} * lambda_k^2 - kappa_i^2.

    Parameters
    ----------
    projection_weights : np.ndarray
        p_{ik}(t), shape (T, n, m)
    eigenvalues : np.ndarray
        lambda_k(t), shape (T, m)
    kappa : np.ndarray
        Mean eigenvalue kappa_i(t), shape (T, n)

    Returns
    -------
    np.ndarray
        Spectral variance, shape (T, n)
    """
    # E[lambda^2] = sum_k p_{ik} * lambda_k^2
    lambda_sq = eigenvalues ** 2  # (T, m)
    e_lambda_sq = np.sum(
        projection_weights * lambda_sq[:, np.newaxis, :],
        axis=2
    )  # (T, n)

    # Var = E[lambda^2] - E[lambda]^2
    var_lambda = e_lambda_sq - kappa ** 2

    return np.maximum(var_lambda, 0)  # Clamp to non-negative


def compute_bulk_mass(
    projection_weights: np.ndarray,
    eigenvalues: np.ndarray,
    threshold: float = 1.0
) -> np.ndarray:
    """
    Compute mass in bulk eigenspaces (lambda <= threshold).

    Parameters
    ----------
    projection_weights : np.ndarray
        p_{ik}(t), shape (T, n, m)
    eigenvalues : np.ndarray
        lambda_k(t), shape (T, m)
    threshold : float
        Bulk threshold (default: 1.0)

    Returns
    -------
    np.ndarray
        Bulk mass per feature, shape (T, n)
    """
    T, n, m = projection_weights.shape

    bulk_mass = np.zeros((T, n))
    for t in range(T):
        bulk_mask = eigenvalues[t] <= threshold  # (m,) boolean
        # Sum over eigenspaces where lambda <= threshold
        for i in range(n):
            bulk_mass[t, i] = np.sum(projection_weights[t, i, bulk_mask])

    return bulk_mass


def compute_spectral_drift(
    projection_weights: np.ndarray,
    metric: str = 'js'
) -> np.ndarray:
    """
    Compute spectral drift (divergence between consecutive distributions).

    Parameters
    ----------
    projection_weights : np.ndarray
        p_{ik}(t), shape (T, n, m)
    metric : str
        Divergence metric: 'js' (Jensen-Shannon) or 'kl' (KL divergence)

    Returns
    -------
    np.ndarray
        Drift per feature per transition, shape (T-1, n)
    """
    T, n, m = projection_weights.shape
    drift = np.zeros((T - 1, n))

    for t in range(T - 1):
        p_t = projection_weights[t]      # (n, m)
        p_t1 = projection_weights[t + 1]  # (n, m)

        # Add small epsilon for numerical stability
        eps = 1e-10
        p_t = np.clip(p_t, eps, 1 - eps)
        p_t1 = np.clip(p_t1, eps, 1 - eps)

        # Normalize
        p_t = p_t / p_t.sum(axis=1, keepdims=True)
        p_t1 = p_t1 / p_t1.sum(axis=1, keepdims=True)

        if metric == 'js':
            # Jensen-Shannon divergence
            m_avg = 0.5 * (p_t + p_t1)
            js = 0.5 * (
                np.sum(rel_entr(p_t, m_avg), axis=1) +
                np.sum(rel_entr(p_t1, m_avg), axis=1)
            )
            drift[t] = np.sqrt(js)  # JS distance
        elif metric == 'kl':
            # KL divergence
            drift[t] = np.sum(rel_entr(p_t, p_t1), axis=1)

    return drift


# =============================================================================
# Late Window Summaries
# =============================================================================

@dataclass
class SpreadMetrics:
    """Container for late-window spread metrics per feature."""
    entropy_late: np.ndarray      # H_late(i)
    pr_late: np.ndarray           # PR_late(i)
    var_lambda_late: np.ndarray   # Var_lambda_late(i)
    bulk_mass_late: np.ndarray    # m_{<=1}_late(i)
    pmax_late: np.ndarray         # pmax_late(i)
    drift_late: np.ndarray        # Sum of JS over late window
    lambda_dom_late: np.ndarray   # Median dominant eigenvalue


def compute_late_spread_metrics(
    data: RunData,
    config: AnalysisConfig
) -> SpreadMetrics:
    """
    Compute all spread metrics summarized over late window.

    Parameters
    ----------
    data : RunData
        Run data
    config : AnalysisConfig
        Configuration

    Returns
    -------
    SpreadMetrics
        Container with late-window metrics
    """
    late_idx = config.get_late_window_indices(data.T)

    # Entropy and PR are already in data
    entropy_late = np.median(data.projection_entropy[late_idx], axis=0)
    pr_late = np.median(data.participation_ratio[late_idx], axis=0)
    pmax_late = np.median(data.max_projection[late_idx], axis=0)

    # Compute spectral variance
    var_lambda = compute_spectral_variance(
        data.projection_weights,
        data.eigenvalues,
        data.kappa_expected
    )
    var_lambda_late = np.median(var_lambda[late_idx], axis=0)

    # Compute bulk mass
    bulk_mass = compute_bulk_mass(
        data.projection_weights,
        data.eigenvalues,
        config.lambda_bulk_threshold
    )
    bulk_mass_late = np.median(bulk_mass[late_idx], axis=0)

    # Compute spectral drift
    drift = compute_spectral_drift(data.projection_weights)
    # Sum drift over late transitions
    late_drift_idx = late_idx[late_idx < data.T - 1]
    if len(late_drift_idx) > 0:
        drift_late = np.sum(drift[late_drift_idx], axis=0)
    else:
        drift_late = np.sum(drift[-10:], axis=0)

    # Dominant eigenvalue - compute manually to avoid dimension issues
    T, n = data.dominant_eigenspace.shape
    lambda_dom = np.zeros((T, n))
    for t in range(T):
        for i in range(n):
            k = data.dominant_eigenspace[t, i]
            lambda_dom[t, i] = data.eigenvalues[t, k]
    lambda_dom_late = np.median(lambda_dom[late_idx], axis=0)

    return SpreadMetrics(
        entropy_late=entropy_late,
        pr_late=pr_late,
        var_lambda_late=var_lambda_late,
        bulk_mass_late=bulk_mass_late,
        pmax_late=pmax_late,
        drift_late=drift_late,
        lambda_dom_late=lambda_dom_late,
    )


# =============================================================================
# B3: Statistical Tests
# =============================================================================

def compare_distributions(
    metrics: SpreadMetrics,
    is_dm: np.ndarray
) -> Dict:
    """
    Compare distributions of spread metrics for DM vs well-behaved features.

    Parameters
    ----------
    metrics : SpreadMetrics
        Late-window spread metrics
    is_dm : np.ndarray
        Boolean mask for dark matter features

    Returns
    -------
    Dict
        Comparison results with test statistics
    """
    results = {}

    # List of metrics to compare
    metric_arrays = {
        'entropy': metrics.entropy_late,
        'participation_ratio': metrics.pr_late,
        'spectral_variance': metrics.var_lambda_late,
        'bulk_mass': metrics.bulk_mass_late,
        'pmax': metrics.pmax_late,
        'drift': metrics.drift_late,
        'lambda_dom': metrics.lambda_dom_late,
    }

    for name, arr in metric_arrays.items():
        dm_vals = arr[is_dm]
        wb_vals = arr[~is_dm]

        # Filter NaN/Inf
        dm_vals = dm_vals[np.isfinite(dm_vals)]
        wb_vals = wb_vals[np.isfinite(wb_vals)]

        if len(dm_vals) < 5 or len(wb_vals) < 5:
            results[name] = {'error': 'Insufficient samples'}
            continue

        # Mann-Whitney U test
        stat, p_val = stats.mannwhitneyu(dm_vals, wb_vals, alternative='two-sided')

        # Effect size (rank-biserial correlation)
        n1, n2 = len(dm_vals), len(wb_vals)
        r = 1 - (2 * stat) / (n1 * n2)

        results[name] = {
            'dm_mean': float(np.mean(dm_vals)),
            'dm_std': float(np.std(dm_vals)),
            'dm_median': float(np.median(dm_vals)),
            'wb_mean': float(np.mean(wb_vals)),
            'wb_std': float(np.std(wb_vals)),
            'wb_median': float(np.median(wb_vals)),
            'mann_whitney_stat': float(stat),
            'p_value': float(p_val),
            'effect_size_r': float(r),
            'dm_larger': bool(np.mean(dm_vals) > np.mean(wb_vals)),
        }

    return results


def fit_predictive_model(
    metrics: SpreadMetrics,
    is_dm: np.ndarray
) -> Dict:
    """
    Fit logistic model: DM ~ entropy + PR + var_lambda + drift + pmax + lambda_dom.

    Parameters
    ----------
    metrics : SpreadMetrics
        Late-window spread metrics
    is_dm : np.ndarray
        Boolean mask for dark matter features

    Returns
    -------
    Dict
        Model results
    """
    from scipy.special import expit

    # Build feature matrix
    X_raw = np.column_stack([
        metrics.entropy_late,
        metrics.pr_late,
        metrics.var_lambda_late,
        metrics.drift_late,
        metrics.pmax_late,
        metrics.lambda_dom_late,
    ])

    y = is_dm.astype(float)

    # Filter valid
    valid = np.all(np.isfinite(X_raw), axis=1)
    X_valid = X_raw[valid]
    y_valid = y[valid]

    if np.sum(valid) < 50:
        return {'error': 'Insufficient valid samples'}

    # Standardize
    X_mean = np.mean(X_valid, axis=0)
    X_std = np.std(X_valid, axis=0) + 1e-10
    X_scaled = (X_valid - X_mean) / X_std

    # Add intercept
    X = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])

    # Gradient descent
    n_iter = 1000
    lr = 0.1
    beta = np.zeros(X.shape[1])

    for _ in range(n_iter):
        logits = X @ beta
        probs = expit(logits)
        grad = X.T @ (probs - y_valid) / len(y_valid)
        beta -= lr * grad

    # Pseudo-R^2
    ll_model = np.sum(y_valid * np.log(expit(X @ beta) + 1e-10) +
                      (1 - y_valid) * np.log(1 - expit(X @ beta) + 1e-10))
    ll_null = np.sum(y_valid * np.log(np.mean(y_valid) + 1e-10) +
                     (1 - y_valid) * np.log(1 - np.mean(y_valid) + 1e-10))
    pseudo_r2 = 1 - ll_model / ll_null if ll_null != 0 else 0

    # Predictions for AUC
    probs = expit(X @ beta)

    # Simple AUC calculation
    n_pos = np.sum(y_valid)
    n_neg = len(y_valid) - n_pos
    if n_pos > 0 and n_neg > 0:
        sorted_idx = np.argsort(probs)[::-1]
        sorted_y = y_valid[sorted_idx]
        tpr = np.cumsum(sorted_y) / n_pos
        fpr = np.cumsum(1 - sorted_y) / n_neg
        auc = np.trapz(tpr, fpr)
    else:
        auc = 0.5

    return {
        'coefficients': {
            'intercept': float(beta[0]),
            'entropy': float(beta[1]),
            'participation_ratio': float(beta[2]),
            'spectral_variance': float(beta[3]),
            'drift': float(beta[4]),
            'pmax': float(beta[5]),
            'lambda_dom': float(beta[6]),
        },
        'pseudo_r2': float(pseudo_r2),
        'auc': float(auc),
        'n_samples': int(np.sum(valid)),
        'n_positive': int(np.sum(y_valid)),
    }


# =============================================================================
# B4: Lambda <= 1 Falsifier
# =============================================================================

def run_lambda_le1_falsifier(
    data: RunData,
    metrics: SpreadMetrics,
    is_dm: np.ndarray,
    config: AnalysisConfig
) -> Dict:
    """
    Test whether spread metrics separate DM within lambda <= 1 subset.

    This is the "not merely lambda < 1" falsifier - if spread still matters
    within the bulk, then the hypothesis is stronger.

    Parameters
    ----------
    data : RunData
        Run data
    metrics : SpreadMetrics
        Late-window spread metrics
    is_dm : np.ndarray
        Dark matter labels
    config : AnalysisConfig
        Configuration

    Returns
    -------
    Dict
        Falsifier test results
    """
    # Identify features with lambda_dom <= 1
    in_bulk = metrics.lambda_dom_late <= config.lambda_bulk_threshold

    n_bulk = np.sum(in_bulk)
    n_dm_in_bulk = np.sum(is_dm & in_bulk)
    n_wb_in_bulk = np.sum(~is_dm & in_bulk)

    if n_dm_in_bulk < 5 or n_wb_in_bulk < 5:
        return {
            'n_bulk_features': int(n_bulk),
            'n_dm_in_bulk': int(n_dm_in_bulk),
            'n_wb_in_bulk': int(n_wb_in_bulk),
            'error': 'Insufficient samples in bulk subset',
        }

    # Filter to bulk subset
    bulk_is_dm = is_dm[in_bulk]

    # Create subset metrics
    subset_metrics = SpreadMetrics(
        entropy_late=metrics.entropy_late[in_bulk],
        pr_late=metrics.pr_late[in_bulk],
        var_lambda_late=metrics.var_lambda_late[in_bulk],
        bulk_mass_late=metrics.bulk_mass_late[in_bulk],
        pmax_late=metrics.pmax_late[in_bulk],
        drift_late=metrics.drift_late[in_bulk],
        lambda_dom_late=metrics.lambda_dom_late[in_bulk],
    )

    # Compare distributions within bulk
    comparisons = compare_distributions(subset_metrics, bulk_is_dm)

    # Fit predictive model within bulk
    model = fit_predictive_model(subset_metrics, bulk_is_dm)

    return {
        'n_bulk_features': int(n_bulk),
        'n_dm_in_bulk': int(n_dm_in_bulk),
        'n_wb_in_bulk': int(n_wb_in_bulk),
        'dm_rate_in_bulk': float(n_dm_in_bulk / n_bulk) if n_bulk > 0 else 0,
        'comparisons': comparisons,
        'predictive_model': model,
        'spread_still_separates': model.get('auc', 0.5) > 0.6,
    }


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment_B_single_file(
    data: RunData,
    config: AnalysisConfig
) -> Dict:
    """
    Run Experiment B for a single file.

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
    # B1: Validate p_{ik} consistency
    validation = validate_kappa_consistency(
        data.projection_weights,
        data.eigenvalues,
        data.kappa_actual
    )

    # Compute DM labels
    late_idx = config.get_late_window_indices(data.T)
    D_late = data.fractional_dims[late_idx]
    N_late = data.feature_norms[late_idx]
    r2_late = compute_r2_linear_fit(D_late, N_late)

    variance_N = np.var(N_late, axis=0)
    is_degenerate = variance_N < config.variance_floor
    is_dm = (r2_late < config.r2_threshold) & ~is_degenerate

    # B2: Compute spread metrics
    metrics = compute_late_spread_metrics(data, config)

    # B3: Compare distributions
    comparisons = compare_distributions(metrics, is_dm)

    # Fit predictive model
    model = fit_predictive_model(metrics, is_dm)

    # B4: Lambda <= 1 falsifier
    falsifier = run_lambda_le1_falsifier(data, metrics, is_dm, config)

    return {
        'run_id': data.run_id,
        'm_hidden': data.m_hidden,
        'sparsity': data.sparsity,
        'seed': data.seed,
        'validation': validation,
        'dm_rate': float(np.mean(is_dm)),
        'n_dm': int(np.sum(is_dm)),
        'comparisons': comparisons,
        'predictive_model': model,
        'falsifier': falsifier,
    }


def run_experiment_B(
    output_dir: Path = None,
    config: AnalysisConfig = None,
    max_files: int = None
) -> Dict:
    """
    Run full Experiment B across all files.

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
        output_dir = OUTPUT_DIR / 'experiment_B'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Experiment B: Spectral Spread Metrics")

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
            result = run_experiment_B_single_file(data, config)
            all_results.append(result)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(files)} files")

        except Exception as e:
            logger.warning(f"Error processing {filepath}: {e}")

    # Aggregate results
    aggregated = aggregate_experiment_B_results(all_results)

    # Save results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)

    logger.info(f"Experiment B complete. Results saved to {output_dir}")

    return aggregated


def aggregate_experiment_B_results(results: List[Dict]) -> Dict:
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

    # Validation
    validations_passed = [r['validation']['passed'] for r in results if 'passed' in r.get('validation', {})]

    # DM rates
    dm_rates = [r['dm_rate'] for r in results]

    # Aggregate comparison effect sizes
    metrics = ['entropy', 'participation_ratio', 'spectral_variance', 'drift', 'pmax', 'bulk_mass']
    effect_sizes = {m: [] for m in metrics}
    dm_larger = {m: [] for m in metrics}

    for r in results:
        for m in metrics:
            if m in r.get('comparisons', {}):
                comp = r['comparisons'][m]
                if 'effect_size_r' in comp:
                    effect_sizes[m].append(comp['effect_size_r'])
                if 'dm_larger' in comp:
                    dm_larger[m].append(comp['dm_larger'])

    # Aggregate model performance
    aucs = [r['predictive_model'].get('auc', np.nan) for r in results
            if 'auc' in r.get('predictive_model', {})]
    pseudo_r2s = [r['predictive_model'].get('pseudo_r2', np.nan) for r in results
                  if 'pseudo_r2' in r.get('predictive_model', {})]

    # Aggregate falsifier
    falsifier_aucs = [r['falsifier'].get('predictive_model', {}).get('auc', np.nan)
                      for r in results if 'predictive_model' in r.get('falsifier', {})]
    spread_separates = [r['falsifier'].get('spread_still_separates', False) for r in results]

    return {
        'n_runs': len(results),
        'validation_pass_rate': float(np.mean(validations_passed)) if validations_passed else np.nan,
        'dm_rate_mean': float(np.mean(dm_rates)),
        'dm_rate_std': float(np.std(dm_rates)),
        'effect_sizes': {
            m: {
                'mean': float(np.nanmean(effect_sizes[m])) if effect_sizes[m] else np.nan,
                'std': float(np.nanstd(effect_sizes[m])) if effect_sizes[m] else np.nan,
                'dm_larger_rate': float(np.mean(dm_larger[m])) if dm_larger[m] else np.nan,
            }
            for m in metrics
        },
        'predictive_model': {
            'auc_mean': float(np.nanmean(aucs)) if aucs else np.nan,
            'auc_std': float(np.nanstd(aucs)) if aucs else np.nan,
            'pseudo_r2_mean': float(np.nanmean(pseudo_r2s)) if pseudo_r2s else np.nan,
        },
        'falsifier': {
            'auc_mean': float(np.nanmean(falsifier_aucs)) if falsifier_aucs else np.nan,
            'spread_separates_rate': float(np.mean(spread_separates)) if spread_separates else np.nan,
        },
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Experiment B: Spectral Spread')
    parser.add_argument('--max-files', type=int, default=None, help='Max files to process')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    results = run_experiment_B(output_dir=output_dir, max_files=args.max_files)

    print("\n" + "=" * 60)
    print("EXPERIMENT B SUMMARY")
    print("=" * 60)
    print(f"Runs processed: {results.get('n_runs', 0)}")
    print(f"Mean DM rate: {results.get('dm_rate_mean', np.nan):.4f}")
    print(f"Model AUC: {results['predictive_model'].get('auc_mean', np.nan):.4f}")
    print(f"Falsifier AUC (lambda<=1): {results['falsifier'].get('auc_mean', np.nan):.4f}")

    print("\nEffect sizes (DM vs well-behaved):")
    for m, es in results.get('effect_sizes', {}).items():
        print(f"  {m}: r={es.get('mean', np.nan):.4f}, DM larger: {es.get('dm_larger_rate', np.nan):.1%}")
