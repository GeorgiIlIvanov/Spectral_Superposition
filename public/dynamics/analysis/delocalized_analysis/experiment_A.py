#!/usr/bin/env python3
"""
Experiment A: Eigengaps (lambda <= 1 bulk) vs Projector Rotation vs Persistent Dark Matter
===========================================================================================

This module implements Experiment Family A from the spectral superposition analysis plan.

Hypothesis: Persistent dark matter features are associated with small eigengaps in the bulk
eigenvalue regime (lambda <= 1), which leads to basis instability and larger subspace rotation.

Key Steps:
----------
A0. Define bulk indices: K_bulk = {k : lambda[T-1, k] <= 1}
A1. Compute eigengaps per checkpoint: gap[t,k] = lambda[t,k] - lambda[t,k+1]
A2. Define degenerate blocks from gaps at final checkpoint
A3. Compute projector/subspace rotation per block using chordal distance
A4. Assign features to blocks (index-based or mass-based)
A5. Test correlations between eigengaps, rotation, and persistent DM

Expected Results (if hypothesis A is driver):
- Persistent DM concentrates in smallest-gap bins
- Smallest gaps correspond to higher subspace rotation
- Effects persist after controlling for concentration proxies

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
from scipy.special import expit

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
# Data Structures
# =============================================================================

@dataclass
class EigengapBlock:
    """
    Represents a degenerate eigenspace block defined by small eigengaps.

    Attributes
    ----------
    block_id : int
        Unique identifier for this block
    indices : List[int]
        Eigenvalue indices k that belong to this block
    dimension : int
        Size of the block (number of eigenvalues)
    gap_min : float
        Minimum gap within the block
    gap_mean : float
        Mean gap within the block
    lambda_mean : float
        Mean eigenvalue of the block
    is_bulk : bool
        Whether all eigenvalues in block are <= 1
    """
    block_id: int
    indices: List[int]
    dimension: int
    gap_min: float
    gap_mean: float
    lambda_mean: float
    is_bulk: bool


@dataclass
class FeaturePredictors:
    """
    Feature-level predictors for persistent dark matter analysis.

    Attributes
    ----------
    feature_idx : int
        Feature index i
    gap_i : float
        Eigengap at assigned block (min gap)
    rot_i : float
        Late rotation at assigned block
    dim_i : int
        Dimension of assigned block
    pmax_late : float
        Median max projection over late window
    lambda_dom_late : float
        Median dominant eigenvalue over late window
    r2_late : float
        R^2 of D vs N fit in late window
    is_persistent_dm : bool
        Whether this feature is persistent dark matter
    block_id : int
        ID of assigned block
    """
    feature_idx: int
    gap_i: float
    rot_i: float
    dim_i: int
    pmax_late: float
    lambda_dom_late: float
    r2_late: float
    is_persistent_dm: bool
    block_id: int


# =============================================================================
# A0: Define Bulk Indices
# =============================================================================

def define_bulk_indices(eigenvalues_final: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """
    Define bulk eigenvalue indices at the final checkpoint.

    Parameters
    ----------
    eigenvalues_final : np.ndarray
        Eigenvalues at final checkpoint, shape (m,)
    threshold : float
        Threshold for bulk (default: lambda <= 1.0)

    Returns
    -------
    np.ndarray
        Boolean mask of shape (m,) indicating bulk indices
    """
    return eigenvalues_final <= threshold


# =============================================================================
# A1: Eigengaps Per Checkpoint
# =============================================================================

def compute_eigengaps(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute eigengaps for all checkpoints.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues, shape (T, m)

    Returns
    -------
    np.ndarray
        Eigengaps, shape (T, m-1) where gap[t,k] = lambda[t,k] - lambda[t,k+1]
    """
    return eigenvalues[:, :-1] - eigenvalues[:, 1:]


# =============================================================================
# A2: Define Degenerate Blocks
# =============================================================================

def construct_eigengap_blocks(
    gaps_final: np.ndarray,
    eigenvalues_final: np.ndarray,
    bulk_mask: np.ndarray,
    config: AnalysisConfig
) -> List[EigengapBlock]:
    """
    Construct degenerate blocks from eigengaps at final checkpoint.

    Blocks are formed by grouping consecutive eigenvalue indices where
    the gap is below a threshold.

    Parameters
    ----------
    gaps_final : np.ndarray
        Eigengaps at final checkpoint, shape (m-1,)
    eigenvalues_final : np.ndarray
        Eigenvalues at final checkpoint, shape (m,)
    bulk_mask : np.ndarray
        Boolean mask for bulk indices, shape (m,)
    config : AnalysisConfig
        Configuration with gap threshold settings

    Returns
    -------
    List[EigengapBlock]
        List of constructed blocks
    """
    m = len(eigenvalues_final)

    # Compute gap threshold
    bulk_gaps = gaps_final[bulk_mask[:-1]]  # Gaps within bulk
    if len(bulk_gaps) == 0:
        logger.warning("No bulk gaps found, using all gaps")
        bulk_gaps = gaps_final

    if config.use_percentile_gap:
        gap_threshold = np.percentile(bulk_gaps, config.eigengap_percentile)
    else:
        gap_threshold = config.eigengap_scale * np.median(bulk_gaps)

    logger.info(f"Gap threshold: {gap_threshold:.6f}")

    # Scan and group consecutive small-gap indices
    blocks = []
    current_block_indices = [0]

    for k in range(len(gaps_final)):
        if gaps_final[k] < gap_threshold:
            # Continue current block
            current_block_indices.append(k + 1)
        else:
            # Close current block if non-trivial
            if len(current_block_indices) > 0:
                block = _create_block(
                    block_id=len(blocks),
                    indices=current_block_indices,
                    gaps_final=gaps_final,
                    eigenvalues_final=eigenvalues_final,
                    bulk_mask=bulk_mask
                )
                blocks.append(block)
            # Start new block
            current_block_indices = [k + 1]

    # Handle last block
    if len(current_block_indices) > 0:
        block = _create_block(
            block_id=len(blocks),
            indices=current_block_indices,
            gaps_final=gaps_final,
            eigenvalues_final=eigenvalues_final,
            bulk_mask=bulk_mask
        )
        blocks.append(block)

    logger.info(f"Constructed {len(blocks)} eigengap blocks")
    return blocks


def _create_block(
    block_id: int,
    indices: List[int],
    gaps_final: np.ndarray,
    eigenvalues_final: np.ndarray,
    bulk_mask: np.ndarray
) -> EigengapBlock:
    """Create an EigengapBlock from indices."""
    indices = list(indices)

    # Compute gap statistics (gaps between consecutive indices in block)
    block_gaps = []
    for i in range(len(indices) - 1):
        k = indices[i]
        if k < len(gaps_final):
            block_gaps.append(gaps_final[k])

    if len(block_gaps) == 0:
        gap_min = gaps_final[indices[0]] if indices[0] < len(gaps_final) else 0.0
        gap_mean = gap_min
    else:
        gap_min = np.min(block_gaps)
        gap_mean = np.mean(block_gaps)

    # Eigenvalue statistics
    block_eigenvalues = eigenvalues_final[indices]
    lambda_mean = np.mean(block_eigenvalues)

    # Check if block is in bulk
    is_bulk = np.all(bulk_mask[indices])

    return EigengapBlock(
        block_id=block_id,
        indices=indices,
        dimension=len(indices),
        gap_min=gap_min,
        gap_mean=gap_mean,
        lambda_mean=lambda_mean,
        is_bulk=is_bulk
    )


# =============================================================================
# A3: Projector/Subspace Rotation Per Block
# =============================================================================

def compute_block_rotation(
    rotation_matrix_diag: np.ndarray,
    blocks: List[EigengapBlock],
    config: AnalysisConfig
) -> Dict[int, np.ndarray]:
    """
    Compute subspace rotation for each block.

    Uses the chordal distance metric:
    chordal_dist_sq = d - ||Q_t^T Q_{t+1}||_F^2
    rot = sqrt(chordal_dist_sq / d)

    Since we have the diagonal of U(t+1)^T @ U(t), we can approximate
    the rotation within each block.

    Parameters
    ----------
    rotation_matrix_diag : np.ndarray
        Diagonal of R(t) = U(t+1)^T @ U(t), shape (T-1, m)
    blocks : List[EigengapBlock]
        List of eigengap blocks
    config : AnalysisConfig
        Configuration

    Returns
    -------
    Dict[int, np.ndarray]
        block_id -> rotation time series of shape (T-1,)
    """
    T_minus_1, m = rotation_matrix_diag.shape
    block_rotations = {}

    for block in blocks:
        indices = block.indices
        d = block.dimension

        if d == 0:
            block_rotations[block.block_id] = np.zeros(T_minus_1)
            continue

        # Extract diagonal elements for this block
        block_diag = rotation_matrix_diag[:, indices]  # (T-1, d)

        # Frobenius norm squared of diagonal approximation
        # For a unitary block, ||M||_F^2 = sum_k cos^2(theta_k) for diagonal
        # This is an approximation; full rotation would need off-diagonal terms
        frob_sq = np.sum(block_diag ** 2, axis=1)  # (T-1,)

        # Chordal distance
        chordal_dist_sq = np.maximum(0, d - frob_sq)
        rotation = np.sqrt(chordal_dist_sq / d)

        block_rotations[block.block_id] = rotation

    return block_rotations


def compute_late_rotation(
    block_rotations: Dict[int, np.ndarray],
    config: AnalysisConfig,
    T: int = N_CHECKPOINTS
) -> Dict[int, float]:
    """
    Compute median rotation over late window for each block.

    Parameters
    ----------
    block_rotations : Dict[int, np.ndarray]
        block_id -> rotation time series
    config : AnalysisConfig
        Configuration
    T : int
        Total number of checkpoints

    Returns
    -------
    Dict[int, float]
        block_id -> median late rotation
    """
    late_idx = config.get_late_window_indices(T)
    # Adjust for T-1 rotation array
    late_rot_idx = late_idx[late_idx < T - 1]

    late_rotations = {}
    for block_id, rotations in block_rotations.items():
        if len(late_rot_idx) > 0:
            late_rotations[block_id] = np.median(rotations[late_rot_idx])
        else:
            late_rotations[block_id] = np.median(rotations[-10:])

    return late_rotations


# =============================================================================
# A4: Assign Features to Blocks
# =============================================================================

def assign_features_to_blocks_index_based(
    dominant_eigenspace: np.ndarray,
    blocks: List[EigengapBlock],
    config: AnalysisConfig
) -> np.ndarray:
    """
    Assign features to blocks based on modal dominant eigenspace over late window.

    Parameters
    ----------
    dominant_eigenspace : np.ndarray
        Dominant eigenspace k*(t,i), shape (T, n)
    blocks : List[EigengapBlock]
        List of eigengap blocks
    config : AnalysisConfig
        Configuration

    Returns
    -------
    np.ndarray
        Block assignments, shape (n,), values are block_ids
    """
    T, n = dominant_eigenspace.shape
    late_idx = config.get_late_window_indices(T)

    # Create mapping from eigenvalue index to block id
    idx_to_block = {}
    for block in blocks:
        for k in block.indices:
            idx_to_block[k] = block.block_id

    # Compute mode of dominant eigenspace over late window
    late_dominant = dominant_eigenspace[late_idx]  # (late_window, n)
    k_mode = np.apply_along_axis(
        lambda x: np.bincount(x.astype(np.int32), minlength=1).argmax(),
        axis=0,
        arr=late_dominant
    )

    # Map to block id
    block_assignments = np.array([
        idx_to_block.get(k, -1) for k in k_mode
    ])

    return block_assignments


def assign_features_to_blocks_mass_based(
    projection_weights: np.ndarray,
    blocks: List[EigengapBlock],
    config: AnalysisConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign features to blocks based on mass of projections in each block.

    Parameters
    ----------
    projection_weights : np.ndarray
        Full p_{ik}(t), shape (T, n, m)
    blocks : List[EigengapBlock]
        List of eigengap blocks
    config : AnalysisConfig
        Configuration

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (block_assignments of shape (n,), mass_matrix of shape (n, n_blocks))
    """
    T, n, m = projection_weights.shape
    late_idx = config.get_late_window_indices(T)
    n_blocks = len(blocks)

    # Compute mass per block for each feature
    mass_matrix = np.zeros((n, n_blocks))

    for block in blocks:
        indices = np.array(block.indices)
        # Mean over late window of sum of projections in block
        block_mass = np.mean(
            np.sum(projection_weights[late_idx][:, :, indices], axis=2),
            axis=0
        )  # (n,)
        mass_matrix[:, block.block_id] = block_mass

    # Assign to block with maximum mass
    block_assignments = np.argmax(mass_matrix, axis=1)

    return block_assignments, mass_matrix


# =============================================================================
# A5: Feature-Level Predictors and Tests
# =============================================================================

def compute_persistent_dm_labels(
    fractional_dims: np.ndarray,
    feature_norms: np.ndarray,
    config: AnalysisConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute persistent dark matter labels based on late-window R^2.

    Parameters
    ----------
    fractional_dims : np.ndarray
        D_i(t), shape (T, n)
    feature_norms : np.ndarray
        N_i(t), shape (T, n)
    config : AnalysisConfig
        Configuration

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (is_persistent_dm mask (n,), r2_late values (n,))
    """
    T, n = fractional_dims.shape
    late_idx = config.get_late_window_indices(T)

    D_late = fractional_dims[late_idx]
    N_late = feature_norms[late_idx]

    r2_late = compute_r2_linear_fit(D_late, N_late)

    # Check for degeneracy (variance floor)
    variance_N = np.var(N_late, axis=0)
    is_degenerate = variance_N < config.variance_floor

    # Persistent DM: low R^2 and not degenerate
    is_persistent_dm = (r2_late < config.r2_threshold) & ~is_degenerate

    return is_persistent_dm, r2_late


def build_feature_predictors(
    data: RunData,
    blocks: List[EigengapBlock],
    block_assignments: np.ndarray,
    late_rotations: Dict[int, float],
    config: AnalysisConfig
) -> List[FeaturePredictors]:
    """
    Build feature-level predictors for all features.

    Parameters
    ----------
    data : RunData
        Run data
    blocks : List[EigengapBlock]
        List of eigengap blocks
    block_assignments : np.ndarray
        Block assignments, shape (n,)
    late_rotations : Dict[int, float]
        block_id -> late rotation
    config : AnalysisConfig
        Configuration

    Returns
    -------
    List[FeaturePredictors]
        Predictor object for each feature
    """
    # Compute DM labels
    is_dm, r2_late = compute_persistent_dm_labels(
        data.fractional_dims, data.feature_norms, config
    )

    # Compute late window statistics
    late_idx = config.get_late_window_indices(data.T)
    pmax_late = np.median(data.max_projection[late_idx], axis=0)

    # Compute lambda_dom_late for each feature
    # dominant_eigenspace is (T, n), eigenvalues is (T, m)
    lambda_dom_late = np.zeros(data.n)
    for t_idx in late_idx:
        for i in range(data.n):
            k = data.dominant_eigenspace[t_idx, i]
            lambda_dom_late[i] += data.eigenvalues[t_idx, k]
    lambda_dom_late /= len(late_idx)

    # Create block lookup
    block_dict = {b.block_id: b for b in blocks}

    predictors = []
    for i in range(data.n):
        block_id = block_assignments[i]
        if block_id < 0 or block_id not in block_dict:
            # Unassigned feature
            predictors.append(FeaturePredictors(
                feature_idx=i,
                gap_i=np.nan,
                rot_i=np.nan,
                dim_i=0,
                pmax_late=pmax_late[i],
                lambda_dom_late=lambda_dom_late[i],
                r2_late=r2_late[i],
                is_persistent_dm=is_dm[i],
                block_id=-1
            ))
        else:
            block = block_dict[block_id]
            predictors.append(FeaturePredictors(
                feature_idx=i,
                gap_i=block.gap_min,
                rot_i=late_rotations.get(block_id, np.nan),
                dim_i=block.dimension,
                pmax_late=pmax_late[i],
                lambda_dom_late=lambda_dom_late[i],
                r2_late=r2_late[i],
                is_persistent_dm=is_dm[i],
                block_id=block_id
            ))

    return predictors


def compute_enrichment_by_gap_bins(
    predictors: List[FeaturePredictors],
    n_bins: int = 5
) -> Dict:
    """
    Compute persistent DM enrichment by gap quantile bins.

    Parameters
    ----------
    predictors : List[FeaturePredictors]
        Feature predictors
    n_bins : int
        Number of quantile bins

    Returns
    -------
    Dict
        Enrichment results with bin edges and DM rates
    """
    gaps = np.array([p.gap_i for p in predictors if not np.isnan(p.gap_i)])
    dm_flags = np.array([p.is_persistent_dm for p in predictors if not np.isnan(p.gap_i)])

    if len(gaps) == 0:
        return {'error': 'No valid gaps'}

    # Compute quantile bins
    bin_edges = np.percentile(gaps, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(gaps, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute DM rate per bin
    dm_rates = []
    bin_sizes = []
    bin_means = []

    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            dm_rates.append(np.mean(dm_flags[mask]))
            bin_sizes.append(int(np.sum(mask)))
            bin_means.append(np.mean(gaps[mask]))
        else:
            dm_rates.append(np.nan)
            bin_sizes.append(0)
            bin_means.append(np.nan)

    return {
        'bin_edges': bin_edges.tolist(),
        'dm_rates': dm_rates,
        'bin_sizes': bin_sizes,
        'bin_means': bin_means,
        'overall_dm_rate': float(np.mean(dm_flags)),
    }


def compute_correlations(predictors: List[FeaturePredictors]) -> Dict:
    """
    Compute Spearman correlations between predictors.

    Parameters
    ----------
    predictors : List[FeaturePredictors]
        Feature predictors

    Returns
    -------
    Dict
        Correlation results
    """
    # Extract arrays
    gap = np.array([p.gap_i for p in predictors])
    rot = np.array([p.rot_i for p in predictors])
    pmax = np.array([p.pmax_late for p in predictors])
    lambda_dom = np.array([p.lambda_dom_late for p in predictors])
    r2 = np.array([p.r2_late for p in predictors])
    dm = np.array([p.is_persistent_dm for p in predictors]).astype(float)

    # Filter valid
    valid = ~(np.isnan(gap) | np.isnan(rot) | np.isnan(pmax) | np.isnan(lambda_dom))

    results = {}

    # Gap vs Rotation
    if np.sum(valid) > 10:
        rho, p = stats.spearmanr(gap[valid], rot[valid])
        results['gap_vs_rotation'] = {'rho': float(rho), 'p': float(p)}

    # Gap vs DM
    valid_dm = valid & ~np.isnan(r2)
    if np.sum(valid_dm) > 10:
        rho, p = stats.spearmanr(gap[valid_dm], dm[valid_dm])
        results['gap_vs_dm'] = {'rho': float(rho), 'p': float(p)}

    # Rotation vs DM
    if np.sum(valid_dm) > 10:
        rho, p = stats.spearmanr(rot[valid_dm], dm[valid_dm])
        results['rotation_vs_dm'] = {'rho': float(rho), 'p': float(p)}

    # pmax vs DM
    if np.sum(valid_dm) > 10:
        rho, p = stats.spearmanr(pmax[valid_dm], dm[valid_dm])
        results['pmax_vs_dm'] = {'rho': float(rho), 'p': float(p)}

    return results


def fit_logistic_model(predictors: List[FeaturePredictors]) -> Dict:
    """
    Fit logistic regression: persistent_DM ~ log(gap) + rot + dim + pmax + lambda_dom.

    Parameters
    ----------
    predictors : List[FeaturePredictors]
        Feature predictors

    Returns
    -------
    Dict
        Model coefficients and statistics
    """
    # Extract features
    gap = np.array([p.gap_i for p in predictors])
    rot = np.array([p.rot_i for p in predictors])
    dim = np.array([p.dim_i for p in predictors], dtype=float)
    pmax = np.array([p.pmax_late for p in predictors])
    lambda_dom = np.array([p.lambda_dom_late for p in predictors])
    y = np.array([p.is_persistent_dm for p in predictors]).astype(float)

    # Valid mask
    valid = ~(np.isnan(gap) | np.isnan(rot) | np.isnan(pmax) |
              np.isnan(lambda_dom) | (gap <= 0))

    if np.sum(valid) < 50:
        return {'error': 'Not enough valid samples'}

    # Build design matrix (standardized)
    log_gap = np.log(gap[valid] + 1e-10)
    X = np.column_stack([
        (log_gap - np.mean(log_gap)) / (np.std(log_gap) + 1e-10),
        (rot[valid] - np.mean(rot[valid])) / (np.std(rot[valid]) + 1e-10),
        (dim[valid] - np.mean(dim[valid])) / (np.std(dim[valid]) + 1e-10),
        (pmax[valid] - np.mean(pmax[valid])) / (np.std(pmax[valid]) + 1e-10),
        (lambda_dom[valid] - np.mean(lambda_dom[valid])) / (np.std(lambda_dom[valid]) + 1e-10),
    ])
    X = np.column_stack([np.ones(X.shape[0]), X])  # Add intercept
    y_valid = y[valid]

    # Simple gradient descent for logistic regression
    # (Could use sklearn but keeping dependencies minimal)
    n_iter = 1000
    lr = 0.1
    beta = np.zeros(X.shape[1])

    for _ in range(n_iter):
        logits = X @ beta
        probs = expit(logits)
        grad = X.T @ (probs - y_valid) / len(y_valid)
        beta -= lr * grad

    # Compute pseudo-R^2 (McFadden)
    ll_model = np.sum(y_valid * np.log(expit(X @ beta) + 1e-10) +
                      (1 - y_valid) * np.log(1 - expit(X @ beta) + 1e-10))
    ll_null = np.sum(y_valid * np.log(np.mean(y_valid) + 1e-10) +
                     (1 - y_valid) * np.log(1 - np.mean(y_valid) + 1e-10))
    pseudo_r2 = 1 - ll_model / ll_null if ll_null != 0 else 0

    return {
        'coefficients': {
            'intercept': float(beta[0]),
            'log_gap': float(beta[1]),
            'rotation': float(beta[2]),
            'block_dim': float(beta[3]),
            'pmax_late': float(beta[4]),
            'lambda_dom_late': float(beta[5]),
        },
        'pseudo_r2': float(pseudo_r2),
        'n_samples': int(np.sum(valid)),
        'n_positive': int(np.sum(y_valid)),
    }


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment_A_single_file(
    data: RunData,
    config: AnalysisConfig
) -> Dict:
    """
    Run Experiment A for a single file.

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
    # A0: Define bulk indices
    eigenvalues_final = data.eigenvalues[-1]
    bulk_mask = define_bulk_indices(eigenvalues_final, config.lambda_bulk_threshold)

    # A1: Eigengaps (already in data.eigengaps)
    gaps_final = data.eigengaps[-1]

    # A2: Construct blocks
    blocks = construct_eigengap_blocks(gaps_final, eigenvalues_final, bulk_mask, config)

    # A3: Projector rotation
    block_rotations = compute_block_rotation(data.rotation_matrix_diag, blocks, config)
    late_rotations = compute_late_rotation(block_rotations, config, data.T)

    # A4: Assign features to blocks (mass-based)
    block_assignments, mass_matrix = assign_features_to_blocks_mass_based(
        data.projection_weights, blocks, config
    )

    # A5: Build predictors
    predictors = build_feature_predictors(
        data, blocks, block_assignments, late_rotations, config
    )

    # Compute tests
    enrichment = compute_enrichment_by_gap_bins(predictors)
    correlations = compute_correlations(predictors)
    logistic_model = fit_logistic_model(predictors)

    return {
        'run_id': data.run_id,
        'm_hidden': data.m_hidden,
        'sparsity': data.sparsity,
        'seed': data.seed,
        'n_blocks': len(blocks),
        'n_bulk_blocks': sum(1 for b in blocks if b.is_bulk),
        'blocks': [asdict(b) for b in blocks],
        'late_rotations': late_rotations,
        'enrichment': enrichment,
        'correlations': correlations,
        'logistic_model': logistic_model,
        'dm_rate': float(np.mean([p.is_persistent_dm for p in predictors])),
        'n_features': len(predictors),
    }


def run_experiment_A(
    output_dir: Path = None,
    config: AnalysisConfig = None,
    max_files: int = None
) -> Dict:
    """
    Run full Experiment A across all files.

    Parameters
    ----------
    output_dir : Path, optional
        Output directory
    config : AnalysisConfig, optional
        Analysis configuration
    max_files : int, optional
        Maximum number of files to process (for testing)

    Returns
    -------
    Dict
        Aggregated results
    """
    if config is None:
        config = AnalysisConfig()

    if output_dir is None:
        output_dir = OUTPUT_DIR / 'experiment_A'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Experiment A: Eigengaps vs Projector Rotation")

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
            result = run_experiment_A_single_file(data, config)
            all_results.append(result)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(files)} files")

        except Exception as e:
            logger.warning(f"Error processing {filepath}: {e}")

    # Aggregate results
    aggregated = aggregate_experiment_A_results(all_results)

    # Save results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)

    logger.info(f"Experiment A complete. Results saved to {output_dir}")

    return aggregated


def aggregate_experiment_A_results(results: List[Dict]) -> Dict:
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

    # Collect statistics
    dm_rates = [r['dm_rate'] for r in results]
    n_blocks = [r['n_blocks'] for r in results]

    # Aggregate enrichment
    all_dm_rates_by_bin = defaultdict(list)
    for r in results:
        if 'dm_rates' in r['enrichment']:
            for i, rate in enumerate(r['enrichment']['dm_rates']):
                if not np.isnan(rate):
                    all_dm_rates_by_bin[i].append(rate)

    enrichment_agg = {
        f'bin_{i}_mean': np.mean(rates) if rates else np.nan
        for i, rates in all_dm_rates_by_bin.items()
    }
    enrichment_agg.update({
        f'bin_{i}_std': np.std(rates) if rates else np.nan
        for i, rates in all_dm_rates_by_bin.items()
    })

    # Aggregate correlations
    all_gap_rot = [r['correlations'].get('gap_vs_rotation', {}).get('rho', np.nan) for r in results]
    all_gap_dm = [r['correlations'].get('gap_vs_dm', {}).get('rho', np.nan) for r in results]
    all_rot_dm = [r['correlations'].get('rotation_vs_dm', {}).get('rho', np.nan) for r in results]

    # Aggregate logistic coefficients
    log_gap_coefs = [r['logistic_model'].get('coefficients', {}).get('log_gap', np.nan)
                     for r in results if 'coefficients' in r.get('logistic_model', {})]
    rot_coefs = [r['logistic_model'].get('coefficients', {}).get('rotation', np.nan)
                 for r in results if 'coefficients' in r.get('logistic_model', {})]

    return {
        'n_runs': len(results),
        'dm_rate_mean': float(np.mean(dm_rates)),
        'dm_rate_std': float(np.std(dm_rates)),
        'n_blocks_mean': float(np.mean(n_blocks)),
        'enrichment': enrichment_agg,
        'correlations': {
            'gap_vs_rotation_mean': float(np.nanmean(all_gap_rot)),
            'gap_vs_dm_mean': float(np.nanmean(all_gap_dm)),
            'rotation_vs_dm_mean': float(np.nanmean(all_rot_dm)),
        },
        'logistic_model': {
            'log_gap_coef_mean': float(np.nanmean(log_gap_coefs)) if log_gap_coefs else np.nan,
            'log_gap_coef_std': float(np.nanstd(log_gap_coefs)) if log_gap_coefs else np.nan,
            'rotation_coef_mean': float(np.nanmean(rot_coefs)) if rot_coefs else np.nan,
            'rotation_coef_std': float(np.nanstd(rot_coefs)) if rot_coefs else np.nan,
        },
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Experiment A: Eigengaps vs Rotation')
    parser.add_argument('--max-files', type=int, default=None, help='Max files to process')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    results = run_experiment_A(output_dir=output_dir, max_files=args.max_files)

    print("\n" + "=" * 60)
    print("EXPERIMENT A SUMMARY")
    print("=" * 60)
    print(f"Runs processed: {results.get('n_runs', 0)}")
    print(f"Mean DM rate: {results.get('dm_rate_mean', np.nan):.4f}")
    print(f"Gap vs Rotation correlation: {results['correlations'].get('gap_vs_rotation_mean', np.nan):.4f}")
    print(f"Gap vs DM correlation: {results['correlations'].get('gap_vs_dm_mean', np.nan):.4f}")
