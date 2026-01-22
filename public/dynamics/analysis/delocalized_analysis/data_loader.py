#!/usr/bin/env python3
"""
Data Loader Module for Delocalized Spectral Analysis
=====================================================

This module provides efficient data loading utilities for the spectral analysis experiments.
It primarily loads from the pre-computed `refined_spectral_results/` directory which contains
all necessary quantities for experiments A, B, C, D-H.

Pre-computed Data Available (per file):
---------------------------------------
- projection_weights: (T=56, n=1024, m) - Full p_{ik}(t) = |u_k^T w_i|^2 / ||w_i||^2
- kappa_actual: (T, n) - Rayleigh quotient from original computation
- kappa_expected: (T, n) - Expected eigenvalue sum_k p_{ik} * lambda_k
- eigengaps: (T, m-1) - lambda_k - lambda_{k+1}
- eigenvalues: (T, m) - Sorted eigenvalues
- rotation_angles: (T-1, m) - Projector rotation angles theta_k(t)
- rotation_matrix_diag: (T-1, m) - Diagonal of U(t+1)^T @ U(t)
- participation_ratio: (T, n) - 1 / sum_k p_{ik}^2
- projection_entropy: (T, n) - -sum_k p_{ik} * log(p_{ik})
- dominant_eigenspace: (T, n) - argmax_k p_{ik}
- max_projection: (T, n) - max_k p_{ik}
- fractional_dims: (T, n) - D_i(t) = ||w_i||^2 / kappa_i
- feature_norms: (T, n) - N_i(t) = ||w_i||^2
- cluster_* statistics: Per-eigenspace cluster stats

Usage:
------
    from data_loader import RefinedSpectralLoader, BatchLoader

    # Load single file
    loader = RefinedSpectralLoader()
    data = loader.load_file(m_hidden=112, sparsity=0.0, seed=0)

    # Load all files for analysis
    batch_loader = BatchLoader(n_workers=8)
    all_data = batch_loader.load_all()

Author: Claude Code (Anthropic)
Date: 2026-01-22
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import numpy as np
import h5py

from config import (
    REFINED_SPECTRAL_DIR, CLUSTERING_FILE, START_DIR, SVD_DIR,
    N_FEATURES, N_CHECKPOINTS, parse_filename, get_clustering_key,
    AnalysisConfig, DEFAULT_SPARSITY_BUCKETS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Container Classes
# =============================================================================

@dataclass
class RunData:
    """
    Container for all data from a single run.

    This class holds all pre-computed spectral quantities for one (m_hidden, sparsity, seed)
    configuration, making it easy to pass around and access during analysis.

    Attributes
    ----------
    m_hidden : int
        Hidden dimension (m)
    sparsity : float
        Sparsity level
    seed : int
        Random seed
    checkpoint_steps : np.ndarray
        Training step at each checkpoint, shape (T,)

    Spectral Quantities
    -------------------
    projection_weights : np.ndarray
        Full p_{ik}(t), shape (T, n, m)
    eigenvalues : np.ndarray
        Eigenvalues lambda_k(t), shape (T, m)
    eigengaps : np.ndarray
        Gaps lambda_k - lambda_{k+1}, shape (T, m-1)
    kappa_actual : np.ndarray
        Rayleigh quotient, shape (T, n)
    kappa_expected : np.ndarray
        Expected eigenvalue, shape (T, n)

    Derived Quantities
    ------------------
    fractional_dims : np.ndarray
        D_i(t), shape (T, n)
    feature_norms : np.ndarray
        N_i(t) = ||w_i||^2, shape (T, n)
    participation_ratio : np.ndarray
        PR_i(t), shape (T, n)
    projection_entropy : np.ndarray
        H_i(t), shape (T, n)
    dominant_eigenspace : np.ndarray
        k*(t, i), shape (T, n)
    max_projection : np.ndarray
        pmax(t, i), shape (T, n)

    Rotation Quantities
    -------------------
    rotation_angles : np.ndarray
        theta_k(t), shape (T-1, m)
    rotation_matrix_diag : np.ndarray
        Diagonal of R(t), shape (T-1, m)

    Cluster Statistics
    ------------------
    cluster_kappa_mean : np.ndarray
        Mean kappa per cluster, shape (T, m)
    cluster_kappa_variance : np.ndarray
        Variance of kappa per cluster, shape (T, m)
    cluster_size : np.ndarray
        Size of each cluster, shape (T, m)
    """

    # Metadata
    m_hidden: int
    sparsity: float
    seed: int
    checkpoint_steps: np.ndarray

    # Core spectral quantities
    projection_weights: np.ndarray  # (T, n, m)
    eigenvalues: np.ndarray         # (T, m)
    eigengaps: np.ndarray           # (T, m-1)
    kappa_actual: np.ndarray        # (T, n)
    kappa_expected: np.ndarray      # (T, n)

    # Derived quantities
    fractional_dims: np.ndarray     # (T, n) - D_i(t)
    feature_norms: np.ndarray       # (T, n) - N_i(t)
    participation_ratio: np.ndarray # (T, n)
    projection_entropy: np.ndarray  # (T, n)
    dominant_eigenspace: np.ndarray # (T, n)
    max_projection: np.ndarray      # (T, n)
    n_significant: np.ndarray       # (T, n)

    # Rotation quantities
    rotation_angles: np.ndarray     # (T-1, m)
    rotation_matrix_diag: np.ndarray # (T-1, m)

    # Cluster statistics
    cluster_kappa_mean: np.ndarray      # (T, m)
    cluster_kappa_variance: np.ndarray  # (T, m)
    cluster_size: np.ndarray            # (T, m)

    # Errors
    kappa_error: np.ndarray         # (T, n)
    kappa_relative_error: np.ndarray # (T, n)

    @property
    def T(self) -> int:
        """Number of checkpoints."""
        return self.eigenvalues.shape[0]

    @property
    def n(self) -> int:
        """Number of features."""
        return self.fractional_dims.shape[1]

    @property
    def m(self) -> int:
        """Hidden dimension."""
        return self.eigenvalues.shape[1]

    @property
    def run_id(self) -> str:
        """Unique identifier for this run."""
        return f"m{self.m_hidden}_s{self.sparsity:.6f}_seed{self.seed}"

    def get_late_window_data(self, config: AnalysisConfig) -> Dict[str, np.ndarray]:
        """
        Extract data for the late training window.

        Parameters
        ----------
        config : AnalysisConfig
            Analysis configuration with late window settings

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of late-window data slices
        """
        late_idx = config.get_late_window_indices(self.T)

        return {
            'projection_weights': self.projection_weights[late_idx],
            'eigenvalues': self.eigenvalues[late_idx],
            'kappa_actual': self.kappa_actual[late_idx],
            'kappa_expected': self.kappa_expected[late_idx],
            'fractional_dims': self.fractional_dims[late_idx],
            'feature_norms': self.feature_norms[late_idx],
            'participation_ratio': self.participation_ratio[late_idx],
            'projection_entropy': self.projection_entropy[late_idx],
            'dominant_eigenspace': self.dominant_eigenspace[late_idx],
            'max_projection': self.max_projection[late_idx],
            'rotation_angles': self.rotation_angles[late_idx[:-1]] if late_idx[-1] == self.T - 1 else self.rotation_angles[late_idx],
        }


# =============================================================================
# Main Data Loader
# =============================================================================

class RefinedSpectralLoader:
    """
    Loader for pre-computed refined spectral analysis results.

    This is the primary data loader for all experiments. It reads from the
    refined_spectral_results/ directory which contains pre-computed npz files.

    Parameters
    ----------
    base_dir : Path, optional
        Base directory for refined spectral results

    Examples
    --------
    >>> loader = RefinedSpectralLoader()
    >>> data = loader.load_file(m_hidden=112, sparsity=0.0, seed=0)
    >>> print(data.projection_weights.shape)
    (56, 1024, 112)
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else REFINED_SPECTRAL_DIR

        if not self.base_dir.exists():
            raise FileNotFoundError(f"Refined spectral directory not found: {self.base_dir}")

        # Cache list of available files
        self._files = sorted(self.base_dir.glob("spectral_*.npz"))
        logger.info(f"Found {len(self._files)} refined spectral files")

    def get_filename(self, m_hidden: int, sparsity: float, seed: int) -> Path:
        """Construct the filename for a specific run."""
        return self.base_dir / f"spectral_n1024_m{m_hidden}_s{sparsity:.6f}_seed{seed}.npz"

    def load_file(self, m_hidden: int = None, sparsity: float = None,
                  seed: int = None, filepath: Path = None) -> RunData:
        """
        Load data for a single run.

        Parameters
        ----------
        m_hidden : int, optional
            Hidden dimension
        sparsity : float, optional
            Sparsity value
        seed : int, optional
            Random seed
        filepath : Path, optional
            Direct path to file (alternative to m/s/seed)

        Returns
        -------
        RunData
            Container with all run data
        """
        if filepath is None:
            if m_hidden is None or sparsity is None or seed is None:
                raise ValueError("Must provide either filepath or (m_hidden, sparsity, seed)")
            filepath = self.get_filename(m_hidden, sparsity, seed)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Load all data from npz
        with np.load(filepath) as data:
            run_data = RunData(
                # Metadata
                m_hidden=int(data['m_hidden']),
                sparsity=float(data['sparsity']),
                seed=int(data['seed']),
                checkpoint_steps=data['checkpoint_steps'],

                # Core spectral
                projection_weights=data['projection_weights'],
                eigenvalues=data['eigenvalues'],
                eigengaps=data['eigengaps'],
                kappa_actual=data['kappa_actual'],
                kappa_expected=data['kappa_expected'],

                # Derived
                fractional_dims=data['fractional_dims'],
                feature_norms=data['feature_norms'],
                participation_ratio=data['participation_ratio'],
                projection_entropy=data['projection_entropy'],
                dominant_eigenspace=data['dominant_eigenspace'],
                max_projection=data['max_projection'],
                n_significant=data['n_significant_eigenspaces'],

                # Rotation
                rotation_angles=data['rotation_angles'],
                rotation_matrix_diag=data['rotation_matrix_diag'],

                # Cluster stats
                cluster_kappa_mean=data['cluster_kappa_mean'],
                cluster_kappa_variance=data['cluster_kappa_variance'],
                cluster_size=data['cluster_size'],

                # Errors
                kappa_error=data['kappa_error'],
                kappa_relative_error=data['kappa_relative_error'],
            )

        return run_data

    def iter_files(self) -> Iterator[Path]:
        """Iterate over all available files."""
        return iter(self._files)

    def iter_runs(self) -> Iterator[RunData]:
        """
        Iterate over all runs, loading each one.

        Yields
        ------
        RunData
            Data for each run
        """
        for filepath in self._files:
            try:
                yield self.load_file(filepath=filepath)
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")
                continue

    def get_file_metadata(self) -> List[Dict]:
        """
        Get metadata for all files without loading full data.

        Returns
        -------
        List[Dict]
            List of dicts with m_hidden, sparsity, seed, filepath
        """
        metadata = []
        for filepath in self._files:
            m, s, seed = parse_filename(filepath.name)
            metadata.append({
                'm_hidden': m,
                'sparsity': s,
                'seed': seed,
                'filepath': filepath,
            })
        return metadata


# =============================================================================
# Batch Loader for Parallel Processing
# =============================================================================

def _load_file_worker(filepath: str) -> Optional[Dict]:
    """Worker function for parallel loading."""
    try:
        with np.load(filepath) as data:
            return {key: data[key] for key in data.keys()}
    except Exception as e:
        logger.warning(f"Error loading {filepath}: {e}")
        return None


class BatchLoader:
    """
    Batch loader for parallel processing of all runs.

    Parameters
    ----------
    n_workers : int
        Number of parallel workers
    base_dir : Path, optional
        Base directory for refined spectral results

    Examples
    --------
    >>> batch_loader = BatchLoader(n_workers=8)
    >>> for batch in batch_loader.iter_batches(batch_size=100):
    ...     process_batch(batch)
    """

    def __init__(self, n_workers: int = 8, base_dir: Optional[Path] = None):
        self.n_workers = n_workers
        self.loader = RefinedSpectralLoader(base_dir)
        self._files = list(self.loader.iter_files())

    def iter_batches(self, batch_size: int = 100) -> Iterator[List[RunData]]:
        """
        Iterate over batches of runs.

        Parameters
        ----------
        batch_size : int
            Number of runs per batch

        Yields
        ------
        List[RunData]
            Batch of run data
        """
        for i in range(0, len(self._files), batch_size):
            batch_files = self._files[i:i + batch_size]
            batch_data = []

            for filepath in batch_files:
                try:
                    data = self.loader.load_file(filepath=filepath)
                    batch_data.append(data)
                except Exception as e:
                    logger.warning(f"Error loading {filepath}: {e}")

            yield batch_data

    def load_all_aggregated(self, keys: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Load specific keys from all files and aggregate.

        Parameters
        ----------
        keys : List[str], optional
            Keys to load. If None, loads all keys.

        Returns
        -------
        Dict[str, np.ndarray]
            Aggregated arrays with run dimension prepended
        """
        if keys is None:
            # Default keys for aggregation
            keys = [
                'eigenvalues', 'eigengaps', 'kappa_actual', 'kappa_expected',
                'fractional_dims', 'feature_norms', 'participation_ratio',
                'projection_entropy', 'dominant_eigenspace', 'max_projection',
                'rotation_angles', 'cluster_kappa_variance', 'cluster_size',
            ]

        aggregated = {key: [] for key in keys}
        metadata = {'m_hidden': [], 'sparsity': [], 'seed': []}

        for filepath in self._files:
            try:
                with np.load(filepath) as data:
                    for key in keys:
                        if key in data:
                            aggregated[key].append(data[key])

                    metadata['m_hidden'].append(int(data['m_hidden']))
                    metadata['sparsity'].append(float(data['sparsity']))
                    metadata['seed'].append(int(data['seed']))

            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")

        # Stack arrays
        result = {}
        for key, arrays in aggregated.items():
            if arrays:
                result[key] = np.stack(arrays, axis=0)

        result['metadata'] = metadata
        return result

    def group_by_sparsity(self) -> Dict[str, List[Path]]:
        """
        Group files by sparsity bucket.

        Returns
        -------
        Dict[str, List[Path]]
            Mapping from bucket name to list of filepaths
        """
        groups = {bucket.name: [] for bucket in DEFAULT_SPARSITY_BUCKETS}
        groups['other'] = []

        for filepath in self._files:
            _, sparsity, _ = parse_filename(filepath.name)
            placed = False
            for bucket in DEFAULT_SPARSITY_BUCKETS:
                if bucket.contains(sparsity):
                    groups[bucket.name].append(filepath)
                    placed = True
                    break
            if not placed:
                groups['other'].append(filepath)

        return groups

    def group_by_m_hidden(self) -> Dict[int, List[Path]]:
        """
        Group files by hidden dimension.

        Returns
        -------
        Dict[int, List[Path]]
            Mapping from m_hidden to list of filepaths
        """
        groups = {}
        for filepath in self._files:
            m, _, _ = parse_filename(filepath.name)
            if m not in groups:
                groups[m] = []
            groups[m].append(filepath)
        return groups


# =============================================================================
# Clustering Data Loader
# =============================================================================

class ClusteringLoader:
    """
    Loader for clustering results from the consolidated HDF5 file.

    Parameters
    ----------
    filepath : Path, optional
        Path to clustering_results.h5
    """

    def __init__(self, filepath: Optional[Path] = None):
        self.filepath = filepath or CLUSTERING_FILE

        if not self.filepath.exists():
            raise FileNotFoundError(f"Clustering file not found: {self.filepath}")

    def load_run(self, m_hidden: int, sparsity: float, seed: int) -> Dict[str, np.ndarray]:
        """
        Load clustering data for a specific run.

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
        Dict[str, np.ndarray]
            Clustering data with keys: dominant_eigenspace, max_projection_frac, etc.
        """
        key = get_clustering_key(m_hidden, sparsity, seed)

        with h5py.File(self.filepath, 'r') as f:
            if key not in f:
                raise KeyError(f"Run {key} not found in clustering file")

            grp = f[key]
            return {name: grp[name][:] for name in grp.keys()}

    def list_runs(self) -> List[str]:
        """List all available run keys."""
        with h5py.File(self.filepath, 'r') as f:
            return list(f.keys())


# =============================================================================
# Utility Functions
# =============================================================================

def _compute_lambda_dom_late(data: 'RunData', late_idx: np.ndarray) -> np.ndarray:
    """Compute median dominant eigenvalue over late window."""
    n = data.n
    lambda_dom_values = []
    for t_idx in late_idx:
        t_lambda_dom = np.zeros(n)
        for i in range(n):
            k = data.dominant_eigenspace[t_idx, i]
            t_lambda_dom[i] = data.eigenvalues[t_idx, k]
        lambda_dom_values.append(t_lambda_dom)
    return np.median(np.array(lambda_dom_values), axis=0)


def compute_late_window_statistics(data: RunData, config: AnalysisConfig) -> Dict[str, np.ndarray]:
    """
    Compute summary statistics over the late window.

    Parameters
    ----------
    data : RunData
        Run data container
    config : AnalysisConfig
        Analysis configuration

    Returns
    -------
    Dict[str, np.ndarray]
        Late-window summary statistics per feature
    """
    late_idx = config.get_late_window_indices(data.T)

    return {
        # Median over late window
        'pmax_late': np.median(data.max_projection[late_idx], axis=0),
        'kappa_late': np.median(data.kappa_actual[late_idx], axis=0),
        'entropy_late': np.median(data.projection_entropy[late_idx], axis=0),
        'PR_late': np.median(data.participation_ratio[late_idx], axis=0),

        # Mode of dominant eigenspace
        'k_mode': np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=data.dominant_eigenspace[late_idx].astype(np.int32)
        ),

        # Eigenvalue at dominant eigenspace - compute manually
        'lambda_dom_late': _compute_lambda_dom_late(data, late_idx),
    }


def compute_r2_linear_fit(D: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Compute R^2 for D ~ N linear fit (through origin) per feature.

    Parameters
    ----------
    D : np.ndarray
        Fractional dimensions, shape (T_window, n)
    N : np.ndarray
        Feature norms, shape (T_window, n)

    Returns
    -------
    np.ndarray
        R^2 values per feature, shape (n,)
    """
    n_features = D.shape[1]
    r2 = np.zeros(n_features)

    for i in range(n_features):
        d = D[:, i]
        n = N[:, i]

        # Through-origin fit: slope = sum(d*n) / sum(n^2)
        denom = np.sum(n ** 2)
        if denom < 1e-10:
            r2[i] = np.nan
            continue

        slope = np.sum(d * n) / denom
        d_pred = slope * n
        ss_res = np.sum((d - d_pred) ** 2)
        ss_tot = np.sum((d - np.mean(d)) ** 2)

        if ss_tot < 1e-10:
            r2[i] = np.nan
        else:
            r2[i] = 1 - ss_res / ss_tot

    return r2


if __name__ == '__main__':
    # Test the loader
    print("=" * 60)
    print("Testing Data Loader")
    print("=" * 60)

    loader = RefinedSpectralLoader()
    print(f"Found {len(list(loader.iter_files()))} files")

    # Load a sample file
    sample = loader.load_file(m_hidden=112, sparsity=0.0, seed=0)
    print(f"\nSample run: {sample.run_id}")
    print(f"  T = {sample.T}, n = {sample.n}, m = {sample.m}")
    print(f"  projection_weights: {sample.projection_weights.shape}")
    print(f"  eigenvalues: {sample.eigenvalues.shape}")

    # Test late window
    config = AnalysisConfig()
    late_stats = compute_late_window_statistics(sample, config)
    print(f"\nLate window statistics:")
    for key, val in late_stats.items():
        print(f"  {key}: shape={val.shape}, mean={np.nanmean(val):.4f}")

    print("\nData loader tests passed!")
