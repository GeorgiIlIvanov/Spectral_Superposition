#!/usr/bin/env python3
"""
Configuration Module for Delocalized Spectral Analysis
=======================================================

This module defines all paths, constants, and configuration parameters for the
spectral superposition analysis experiments (A, B, C, D-H).

Data Layout (per run, 3200 total runs):
---------------------------------------
- start/*.h5: Source files with fractional_dims (T=56, n=1024), feature_norms (T=56, n=1024)
- svd_results/*.h5: U (T, m, m), S (T, m), Vt (T, m, n), eigenvalues (T, m)
- clustering_results/clustering_results.h5: dominant_eigenspace, max_projection_frac, eigenvalue_of_dominant
- dynamic_hopping/results/per_file/*.npz: kappa (T, n) Rayleigh quotients

Key Identity:
-------------
D_i(t) = ||w_i(t)||^2 / kappa_i(t)
kappa_i(t) = w_i(t)^T (WW^T) w_i(t) / ||w_i(t)||^2

Author: Claude Code (Anthropic)
Date: 2026-01-22
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# =============================================================================
# Directory Paths
# =============================================================================

# Base directories (relative to analysis folder)
BASE_DIR = Path(__file__).parent.parent.absolute()
START_DIR = BASE_DIR.parent / "start"
SVD_DIR = BASE_DIR / "svd_results"
CLUSTERING_FILE = BASE_DIR / "clustering_results" / "clustering_results.h5"
RAYLEIGH_DIR = BASE_DIR / "dynamic_hopping" / "results" / "per_file"
REFINED_SPECTRAL_DIR = BASE_DIR / "refined_spectral_results"

# Output directory for this analysis
OUTPUT_DIR = BASE_DIR / "delocalized_analysis" / "results"


# =============================================================================
# Data Dimensions
# =============================================================================

N_FEATURES = 1024          # Number of features (n)
N_CHECKPOINTS = 56         # Number of training checkpoints (T)
N_RUNS = 3200              # Total number of runs


# =============================================================================
# Analysis Parameters
# =============================================================================

@dataclass
class AnalysisConfig:
    """
    Configuration for spectral analysis experiments.

    Attributes
    ----------
    late_window_size : int
        Number of checkpoints in the late window (default: last 10 checkpoints)
    late_window_frac : float
        Alternative: fraction of training for late window (default: 0.2 = last 20%)
    r2_threshold : float
        R^2 threshold for persistent dark matter labeling (default: 0.9)
    eigengap_percentile : float
        Percentile for eigengap threshold in block construction (default: 5)
    eigengap_scale : float
        Alternative: scale factor for median gap threshold (default: 0.5)
    significant_projection_threshold : float
        Threshold for counting significant eigenspace projections (default: 0.01)
    pmax_threshold : float
        Threshold for high-concentration feature classification (default: 0.5)
    lambda_bulk_threshold : float
        Eigenvalue threshold for bulk vs spiked (default: 1.0)
    variance_floor : float
        Minimum variance for degeneracy detection (default: 1e-8)
    entropy_epsilon : float
        Small constant to avoid log(0) in entropy computation (default: 1e-10)

    GPU Configuration
    -----------------
    gpus : List[int]
        List of GPU device IDs to use
    batch_size : int
        Number of files per processing batch
    n_workers : int
        Number of parallel workers (typically = number of GPUs)
    """

    # Late window configuration
    late_window_size: int = 10
    late_window_frac: float = 0.2
    use_fixed_late_window: bool = True  # If True, use late_window_size; else use fraction

    # Dark matter labeling
    r2_threshold: float = 0.9

    # Eigengap block construction
    eigengap_percentile: float = 5.0
    eigengap_scale: float = 0.5
    use_percentile_gap: bool = True  # If True, use percentile; else use scaled median

    # Projection thresholds
    significant_projection_threshold: float = 0.01
    pmax_threshold: float = 0.5

    # Eigenvalue regime thresholds
    lambda_bulk_threshold: float = 1.0
    lambda_sensitivity_lower: float = 0.8
    lambda_sensitivity_upper: float = 1.2

    # Numerical stability
    variance_floor: float = 1e-8
    entropy_epsilon: float = 1e-10
    norm_floor: float = 1e-10

    # GPU configuration
    gpus: List[int] = field(default_factory=lambda: list(range(8)))
    batch_size: int = 50
    n_workers: int = 8

    # Eigenvalue bands for C experiments
    eigenvalue_bands: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0),
        (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, float('inf'))
    ])

    def get_late_window_indices(self, n_checkpoints: int = N_CHECKPOINTS) -> np.ndarray:
        """
        Get indices for the late window.

        Parameters
        ----------
        n_checkpoints : int
            Total number of checkpoints

        Returns
        -------
        np.ndarray
            Array of checkpoint indices in the late window
        """
        if self.use_fixed_late_window:
            start_idx = max(0, n_checkpoints - self.late_window_size)
        else:
            start_idx = int(n_checkpoints * (1 - self.late_window_frac))
        return np.arange(start_idx, n_checkpoints)

    def get_bulk_indices(self, eigenvalues: np.ndarray) -> np.ndarray:
        """
        Get indices of eigenvalues in the bulk (lambda <= threshold).

        Parameters
        ----------
        eigenvalues : np.ndarray
            Array of eigenvalues (shape depends on context)

        Returns
        -------
        np.ndarray
            Boolean mask or indices of bulk eigenvalues
        """
        return eigenvalues <= self.lambda_bulk_threshold


@dataclass
class SparsityBucket:
    """
    Configuration for sparsity-stratified analysis.

    Attributes
    ----------
    name : str
        Human-readable name for this bucket
    min_sparsity : float
        Minimum sparsity value (inclusive)
    max_sparsity : float
        Maximum sparsity value (exclusive)
    """
    name: str
    min_sparsity: float
    max_sparsity: float

    def contains(self, sparsity: float) -> bool:
        """Check if a sparsity value falls in this bucket."""
        return self.min_sparsity <= sparsity < self.max_sparsity


# Default sparsity buckets for stratified analysis
DEFAULT_SPARSITY_BUCKETS = [
    SparsityBucket("very_sparse", 0.0, 0.1),
    SparsityBucket("sparse", 0.1, 0.3),
    SparsityBucket("moderate", 0.3, 0.5),
    SparsityBucket("dense", 0.5, 0.7),
    SparsityBucket("very_dense", 0.7, 1.0),
]


# =============================================================================
# File Naming Conventions
# =============================================================================

def parse_filename(filename: str) -> Tuple[int, float, int]:
    """
    Parse m_hidden, sparsity, seed from filename.

    Parameters
    ----------
    filename : str
        Filename in format: n1024_m{M}_s{SPARSITY}_seed{SEED}.h5

    Returns
    -------
    Tuple[int, float, int]
        (m_hidden, sparsity, seed)
    """
    basename = os.path.basename(filename).replace('.h5', '').replace('.npz', '')
    # Handle prefix variations
    basename = basename.replace('svd_', '').replace('spectral_', '')
    basename = basename.replace('_rayleigh', '')

    parts = basename.split('_')

    m_hidden = None
    sparsity = None
    seed = None

    for part in parts:
        if part.startswith('m') and not part.startswith('me'):
            m_hidden = int(part[1:])
        elif part.startswith('s') and not part.startswith('seed'):
            sparsity = float(part[1:])
        elif part.startswith('seed'):
            seed = int(part[4:])

    return m_hidden, sparsity, seed


def get_clustering_key(m_hidden: int, sparsity: float, seed: int) -> str:
    """
    Get the HDF5 group key for clustering results.

    Parameters
    ----------
    m_hidden : int
        Hidden dimension
    sparsity : float
        Sparsity value
    seed : int
        Random seed

    Returns
    -------
    str
        Group key like 'm112_s0.000000_seed0'
    """
    return f"m{m_hidden}_s{sparsity:.6f}_seed{seed}"


def get_file_paths(m_hidden: int, sparsity: float, seed: int) -> dict:
    """
    Get all file paths for a specific run.

    Parameters
    ----------
    m_hidden : int
        Hidden dimension
    sparsity : float
        Sparsity value
    seed : int
        Random seed

    Returns
    -------
    dict
        Dictionary with keys: 'start', 'svd', 'rayleigh', 'refined_spectral', 'clustering_key'
    """
    basename = f"n1024_m{m_hidden}_s{sparsity:.6f}_seed{seed}"

    return {
        'start': START_DIR / f"{basename}.h5",
        'svd': SVD_DIR / f"svd_{basename}.h5",
        'rayleigh': RAYLEIGH_DIR / f"{basename}_rayleigh.npz",
        'refined_spectral': REFINED_SPECTRAL_DIR / f"spectral_{basename}.npz",
        'clustering_key': get_clustering_key(m_hidden, sparsity, seed),
    }


# =============================================================================
# Output Paths
# =============================================================================

def get_output_paths(output_dir: Path = OUTPUT_DIR) -> dict:
    """
    Get output paths for all experiment results.

    Parameters
    ----------
    output_dir : Path
        Base output directory

    Returns
    -------
    dict
        Dictionary of output paths for each experiment
    """
    output_dir = Path(output_dir)

    return {
        # Experiment A outputs
        'A_eigengap_blocks': output_dir / 'experiment_A' / 'eigengap_blocks.npz',
        'A_projector_rotation': output_dir / 'experiment_A' / 'projector_rotation.npz',
        'A_feature_predictors': output_dir / 'experiment_A' / 'feature_predictors.parquet',
        'A_enrichment': output_dir / 'experiment_A' / 'enrichment_results.json',
        'A_correlations': output_dir / 'experiment_A' / 'correlation_results.json',
        'A_logistic_model': output_dir / 'experiment_A' / 'logistic_model.json',

        # Experiment B outputs
        'B_spectral_measures': output_dir / 'experiment_B' / 'spectral_measures.parquet',
        'B_spread_comparison': output_dir / 'experiment_B' / 'spread_comparison.json',
        'B_predictive_model': output_dir / 'experiment_B' / 'predictive_model.json',
        'B_falsifier': output_dir / 'experiment_B' / 'lambda_le1_falsifier.json',

        # Experiment C outputs
        'C_cluster_stats': output_dir / 'experiment_C' / 'cluster_statistics.parquet',
        'C_variance_analysis': output_dir / 'experiment_C' / 'variance_analysis.json',

        # Experiments D-H outputs
        'D_cross_sectional': output_dir / 'experiments_D_H' / 'cross_sectional_linearity.parquet',
        'E_slope_matching': output_dir / 'experiments_D_H' / 'slope_matching.json',
        'F_global_functional': output_dir / 'experiments_D_H' / 'global_functional.json',
        'G_universal_diffusion': output_dir / 'experiments_D_H' / 'universal_diffusion.json',
        'H_normalization': output_dir / 'experiments_D_H' / 'normalization_effect.parquet',

        # Aggregate outputs
        'summary': output_dir / 'aggregate_summary.json',
        'figures_dir': output_dir / 'figures',
    }


# =============================================================================
# Validation
# =============================================================================

def validate_data_directories() -> dict:
    """
    Validate that all required data directories exist and contain expected files.

    Returns
    -------
    dict
        Validation results with counts and any missing items
    """
    results = {
        'start_files': 0,
        'svd_files': 0,
        'rayleigh_files': 0,
        'refined_spectral_files': 0,
        'clustering_file_exists': False,
        'missing': [],
        'valid': True,
    }

    # Count files
    if START_DIR.exists():
        results['start_files'] = len(list(START_DIR.glob("*.h5")))
    else:
        results['missing'].append(str(START_DIR))

    if SVD_DIR.exists():
        results['svd_files'] = len(list(SVD_DIR.glob("*.h5")))
    else:
        results['missing'].append(str(SVD_DIR))

    if RAYLEIGH_DIR.exists():
        results['rayleigh_files'] = len(list(RAYLEIGH_DIR.glob("*_rayleigh.npz")))
    else:
        results['missing'].append(str(RAYLEIGH_DIR))

    if REFINED_SPECTRAL_DIR.exists():
        results['refined_spectral_files'] = len(list(REFINED_SPECTRAL_DIR.glob("spectral_*.npz")))

    results['clustering_file_exists'] = CLUSTERING_FILE.exists()
    if not results['clustering_file_exists']:
        results['missing'].append(str(CLUSTERING_FILE))

    results['valid'] = len(results['missing']) == 0

    return results


if __name__ == '__main__':
    # Print configuration summary when run directly
    print("=" * 60)
    print("Delocalized Spectral Analysis Configuration")
    print("=" * 60)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Start files: {START_DIR}")
    print(f"SVD results: {SVD_DIR}")
    print(f"Clustering: {CLUSTERING_FILE}")
    print(f"Rayleigh: {RAYLEIGH_DIR}")
    print(f"Refined spectral: {REFINED_SPECTRAL_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    print("\n" + "-" * 60)
    print("Validating data directories...")
    validation = validate_data_directories()
    print(f"  Start files: {validation['start_files']}")
    print(f"  SVD files: {validation['svd_files']}")
    print(f"  Rayleigh files: {validation['rayleigh_files']}")
    print(f"  Refined spectral files: {validation['refined_spectral_files']}")
    print(f"  Clustering file exists: {validation['clustering_file_exists']}")

    if validation['missing']:
        print("\nMissing:")
        for m in validation['missing']:
            print(f"  - {m}")
    else:
        print("\nAll data directories validated successfully!")
