#!/usr/bin/env python3
"""
Refined Spectral Analysis Script
================================

This script computes the full spectral decomposition measures for the Spectral Superposition project.

Computed Quantities:
--------------------
1. Full projection weights p_{ik}(t) = |u_k^T w_i|^2 / ||w_i||^2 for ALL k
2. Expected eigenvalue kappa_expected_i(t) = sum_k p_{ik}(t) * lambda_k
3. Eigengaps: delta_k(t) = lambda_k(t) - lambda_{k+1}(t)
4. Projector-rotation matrices: R(t) = U(t+1) @ U(t).T
5. Rotation angles (from R matrix): theta_k(t) = arccos(R_kk(t))
6. Within-cluster variance of kappa_i
7. Participation ratio: PR_i(t) = (sum_k p_{ik})^2 / sum_k p_{ik}^2 = 1 / sum_k p_{ik}^2
8. Effective rank per feature: how many eigenspaces contribute significantly

Usage:
------
    python refined_spectral_analysis.py [--gpus 0,1,2,3,4,5,6,7] [--batch-size 50]

Author: Claude Code (Anthropic)
Date: 2026-01-22
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import h5py
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('refined_spectral_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SpectralConfig:
    """Configuration for spectral analysis computation."""
    source_dir: str = "../start"
    svd_dir: str = "svd_results"
    output_dir: str = "refined_spectral_results"
    clustering_file: str = "clustering_results/clustering_results.h5"
    rayleigh_dir: str = "dynamic_hopping/results/per_file"

    # Computation parameters
    gpus: List[int] = None
    batch_size: int = 50  # Files per GPU batch
    n_features: int = 1024
    n_checkpoints: int = 56

    # Thresholds
    significant_projection_threshold: float = 0.01  # 1% threshold for counting significant projections

    def __post_init__(self):
        if self.gpus is None:
            self.gpus = list(range(8))


@dataclass
class FileResult:
    """Results for a single file."""
    filename: str
    m_hidden: int
    sparsity: float
    seed: int

    # Processing info
    success: bool
    error_message: str = ""
    processing_time: float = 0.0
    gpu_id: int = -1


def parse_filename(filename: str) -> Tuple[int, float, int]:
    """Parse m_hidden, sparsity, seed from filename."""
    # Format: n1024_m{M}_s{SPARSITY}_seed{SEED}.h5
    basename = os.path.basename(filename).replace('.h5', '')
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


def compute_spectral_measures_gpu(
    weights: np.ndarray,  # (n_checkpoints, m, n_features)
    U: np.ndarray,        # (n_checkpoints, m, m)
    eigenvalues: np.ndarray,  # (n_checkpoints, m)
    feature_norms: np.ndarray,  # (n_checkpoints, n_features)
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Compute all spectral measures on GPU.

    Parameters
    ----------
    weights : np.ndarray
        Weight matrix W of shape (T, m, n_features)
    U : np.ndarray
        Left singular vectors of shape (T, m, m)
    eigenvalues : np.ndarray
        Eigenvalues lambda_k of shape (T, m)
    feature_norms : np.ndarray
        Squared norms ||w_i||^2 of shape (T, n_features)
    device : torch.device
        GPU device to use

    Returns
    -------
    Dict with computed measures
    """
    T, m, n_features = weights.shape

    # Move to GPU
    W = torch.from_numpy(weights).to(device, dtype=torch.float32)  # (T, m, n)
    U_t = torch.from_numpy(U).to(device, dtype=torch.float32)      # (T, m, m)
    lam = torch.from_numpy(eigenvalues).to(device, dtype=torch.float32)  # (T, m)
    norms = torch.from_numpy(feature_norms).to(device, dtype=torch.float32)  # (T, n)

    # Avoid division by zero
    norms_safe = torch.clamp(norms, min=1e-10)

    # ============================================================
    # 1. Full projection weights p_{ik}(t) = |u_k^T w_i|^2 / ||w_i||^2
    # ============================================================
    # U_t: (T, m, m) - columns are eigenvectors
    # W: (T, m, n) - columns are feature vectors w_i
    #
    # For each t: projection = U.T @ W -> (m, n), then square elementwise
    # p_{ik} = (U[:, k].T @ w_i)^2 / ||w_i||^2

    # U.T @ W: (T, m, m) @ (T, m, n) -> need batch matmul
    # U.transpose(-2, -1): (T, m, m)
    # Result: (T, m, n) where result[t, k, i] = u_k(t)^T w_i(t)
    projections = torch.bmm(U_t.transpose(-2, -1), W)  # (T, m, n)
    projections_sq = projections ** 2  # (T, m, n) - |u_k^T w_i|^2

    # Normalize by feature norms: p_{ik} = projections_sq / ||w_i||^2
    # norms_safe: (T, n) -> need to broadcast to (T, m, n)
    p_ik = projections_sq / norms_safe.unsqueeze(1)  # (T, m, n)

    # Transpose to (T, n, m) for easier feature-centric access
    p_ik = p_ik.transpose(1, 2)  # (T, n_features, m)

    # ============================================================
    # 2. Expected eigenvalue: kappa_expected_i = sum_k p_{ik} * lambda_k
    # ============================================================
    # lam: (T, m)
    # p_ik: (T, n, m)
    # kappa_expected: (T, n) = sum over k of p_ik * lambda_k
    kappa_expected = torch.sum(p_ik * lam.unsqueeze(1), dim=-1)  # (T, n)

    # ============================================================
    # 3. Eigengaps: delta_k = lambda_k - lambda_{k+1}
    # ============================================================
    # Pad with zeros for the last eigenvalue
    eigengaps = lam[:, :-1] - lam[:, 1:]  # (T, m-1)

    # ============================================================
    # 4. Projector-rotation matrices: R(t) = U(t+1) @ U(t).T
    # ============================================================
    # R[t] measures rotation from time t to t+1
    # R: (T-1, m, m)
    U_next = U_t[1:]  # (T-1, m, m)
    U_curr = U_t[:-1]  # (T-1, m, m)
    R = torch.bmm(U_next.transpose(-2, -1), U_curr)  # (T-1, m, m)

    # ============================================================
    # 5. Rotation angles: theta_k = arccos(|R_kk|)
    # ============================================================
    # Diagonal elements of R give the cosine of rotation angle for each eigenspace
    R_diag = torch.diagonal(R, dim1=-2, dim2=-1)  # (T-1, m)
    # Clamp to [-1, 1] for numerical stability
    R_diag_clamped = torch.clamp(R_diag.abs(), min=-1.0, max=1.0)
    rotation_angles = torch.acos(R_diag_clamped)  # (T-1, m) in radians

    # ============================================================
    # 6. Participation ratio: PR_i = 1 / sum_k p_{ik}^2
    # ============================================================
    # Measures effective number of eigenspaces feature i participates in
    p_ik_sq_sum = torch.sum(p_ik ** 2, dim=-1)  # (T, n)
    p_ik_sq_sum_safe = torch.clamp(p_ik_sq_sum, min=1e-10)
    participation_ratio = 1.0 / p_ik_sq_sum_safe  # (T, n)

    # ============================================================
    # 7. Dominant eigenspace index (for verification)
    # ============================================================
    dominant_k = torch.argmax(p_ik, dim=-1)  # (T, n)
    max_projection = torch.max(p_ik, dim=-1).values  # (T, n)

    # ============================================================
    # 8. Number of significant eigenspaces per feature
    # ============================================================
    # Count how many eigenspaces have projection > threshold
    significant_mask = p_ik > 0.01  # 1% threshold
    n_significant = significant_mask.sum(dim=-1)  # (T, n)

    # ============================================================
    # 9. Entropy of projection distribution
    # ============================================================
    # H_i = -sum_k p_{ik} * log(p_{ik})
    p_ik_safe = torch.clamp(p_ik, min=1e-10)
    entropy = -torch.sum(p_ik * torch.log(p_ik_safe), dim=-1)  # (T, n)

    # Move results back to CPU
    results = {
        'projection_weights': p_ik.cpu().numpy().astype(np.float32),  # (T, n, m)
        'kappa_expected': kappa_expected.cpu().numpy().astype(np.float32),  # (T, n)
        'eigengaps': eigengaps.cpu().numpy().astype(np.float32),  # (T, m-1)
        'rotation_matrix_diag': R_diag.cpu().numpy().astype(np.float32),  # (T-1, m)
        'rotation_angles': rotation_angles.cpu().numpy().astype(np.float32),  # (T-1, m)
        'participation_ratio': participation_ratio.cpu().numpy().astype(np.float32),  # (T, n)
        'dominant_eigenspace': dominant_k.cpu().numpy().astype(np.int16),  # (T, n)
        'max_projection': max_projection.cpu().numpy().astype(np.float32),  # (T, n)
        'n_significant_eigenspaces': n_significant.cpu().numpy().astype(np.int16),  # (T, n)
        'projection_entropy': entropy.cpu().numpy().astype(np.float32),  # (T, n)
    }

    # Clear GPU memory
    del W, U_t, lam, norms, projections, projections_sq, p_ik
    del kappa_expected, eigengaps, R, R_diag, rotation_angles
    del participation_ratio, dominant_k, max_projection, entropy
    torch.cuda.empty_cache()

    return results


def compute_cluster_statistics(
    projection_weights: np.ndarray,  # (T, n, m)
    kappa_actual: np.ndarray,        # (T, n) from Rayleigh files
    kappa_expected: np.ndarray,      # (T, n)
    eigenvalues: np.ndarray,         # (T, m)
    dominant_eigenspace: np.ndarray  # (T, n)
) -> Dict[str, np.ndarray]:
    """
    Compute within-cluster statistics.

    Parameters
    ----------
    projection_weights : np.ndarray
        Full projection weights p_{ik} of shape (T, n, m)
    kappa_actual : np.ndarray
        Actual Rayleigh quotient of shape (T, n)
    kappa_expected : np.ndarray
        Expected eigenvalue from projections of shape (T, n)
    eigenvalues : np.ndarray
        Eigenvalues of shape (T, m)
    dominant_eigenspace : np.ndarray
        Dominant eigenspace index of shape (T, n)

    Returns
    -------
    Dict with cluster statistics
    """
    T, n, m = projection_weights.shape

    # Initialize arrays
    cluster_kappa_mean = np.zeros((T, m), dtype=np.float32)
    cluster_kappa_var = np.zeros((T, m), dtype=np.float32)
    cluster_kappa_expected_mean = np.zeros((T, m), dtype=np.float32)
    cluster_kappa_expected_var = np.zeros((T, m), dtype=np.float32)
    cluster_size = np.zeros((T, m), dtype=np.int32)

    # Kappa error: how well does E[lambda] predict kappa?
    kappa_error = kappa_actual - kappa_expected  # (T, n)
    kappa_rel_error = kappa_error / np.maximum(kappa_actual, 1e-10)  # (T, n)

    # Per-cluster statistics
    for t in range(T):
        for k in range(m):
            mask = dominant_eigenspace[t] == k
            cluster_size[t, k] = np.sum(mask)

            if cluster_size[t, k] > 0:
                cluster_kappa = kappa_actual[t, mask]
                cluster_kappa_exp = kappa_expected[t, mask]

                cluster_kappa_mean[t, k] = np.mean(cluster_kappa)
                cluster_kappa_var[t, k] = np.var(cluster_kappa)
                cluster_kappa_expected_mean[t, k] = np.mean(cluster_kappa_exp)
                cluster_kappa_expected_var[t, k] = np.var(cluster_kappa_exp)

    results = {
        'kappa_error': kappa_error.astype(np.float32),  # (T, n)
        'kappa_relative_error': kappa_rel_error.astype(np.float32),  # (T, n)
        'cluster_kappa_mean': cluster_kappa_mean,  # (T, m)
        'cluster_kappa_variance': cluster_kappa_var,  # (T, m)
        'cluster_kappa_expected_mean': cluster_kappa_expected_mean,  # (T, m)
        'cluster_kappa_expected_variance': cluster_kappa_expected_var,  # (T, m)
        'cluster_size': cluster_size,  # (T, m)
    }

    return results


def process_single_file(
    source_path: str,
    svd_path: str,
    rayleigh_path: str,
    output_path: str,
    gpu_id: int,
    config: SpectralConfig
) -> FileResult:
    """
    Process a single file and compute all spectral measures.

    Parameters
    ----------
    source_path : str
        Path to source .h5 file with weights
    svd_path : str
        Path to SVD .h5 file
    rayleigh_path : str
        Path to Rayleigh .npz file
    output_path : str
        Path for output .npz file
    gpu_id : int
        GPU device ID to use
    config : SpectralConfig
        Configuration object

    Returns
    -------
    FileResult with processing status
    """
    start_time = time.time()
    m_hidden, sparsity, seed = parse_filename(source_path)

    result = FileResult(
        filename=os.path.basename(source_path),
        m_hidden=m_hidden,
        sparsity=sparsity,
        seed=seed,
        success=False,
        gpu_id=gpu_id
    )

    try:
        # Set GPU device
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)

        # Load source data (weights)
        with h5py.File(source_path, 'r') as f:
            weights = f['weights'][:]  # (T, m, n)
            feature_norms = f['feature_norms'][:]  # (T, n)
            checkpoint_steps = f['checkpoint_steps'][:]  # (T,)
            fractional_dims = f['fractional_dims'][:]  # (T, n)

        # Load SVD data
        with h5py.File(svd_path, 'r') as f:
            U = f['U'][:]  # (T, m, m)
            eigenvalues = f['eigenvalues'][:]  # (T, m)
            S = f['S'][:]  # (T, m) - singular values

        # Load Rayleigh quotient (kappa_actual)
        rayleigh_data = np.load(rayleigh_path)
        kappa_actual = rayleigh_data['kappa']  # (T, n)

        # Compute spectral measures on GPU
        spectral_results = compute_spectral_measures_gpu(
            weights, U, eigenvalues, feature_norms, device
        )

        # Compute cluster statistics on CPU
        cluster_results = compute_cluster_statistics(
            spectral_results['projection_weights'],
            kappa_actual,
            spectral_results['kappa_expected'],
            eigenvalues,
            spectral_results['dominant_eigenspace']
        )

        # Combine all results
        all_results = {
            # Metadata
            'm_hidden': m_hidden,
            'sparsity': sparsity,
            'seed': seed,
            'checkpoint_steps': checkpoint_steps,
            'n_features': config.n_features,

            # From source
            'feature_norms': feature_norms,
            'fractional_dims': fractional_dims,

            # From SVD
            'eigenvalues': eigenvalues,
            'singular_values': S,

            # From Rayleigh
            'kappa_actual': kappa_actual,

            # Computed spectral measures
            **spectral_results,

            # Cluster statistics
            **cluster_results,
        }

        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, **all_results)

        result.success = True
        result.processing_time = time.time() - start_time

    except Exception as e:
        result.error_message = str(e)
        result.processing_time = time.time() - start_time
        logger.error(f"Error processing {source_path}: {e}")

    return result


def worker_process(
    file_batch: List[Tuple[str, str, str, str]],
    gpu_id: int,
    config_dict: dict
) -> List[FileResult]:
    """
    Worker process that handles a batch of files on a single GPU.

    Parameters
    ----------
    file_batch : List[Tuple]
        List of (source_path, svd_path, rayleigh_path, output_path) tuples
    gpu_id : int
        GPU device ID
    config_dict : dict
        Configuration as dictionary

    Returns
    -------
    List of FileResult objects
    """
    config = SpectralConfig(**config_dict)
    results = []

    for source_path, svd_path, rayleigh_path, output_path in file_batch:
        result = process_single_file(
            source_path, svd_path, rayleigh_path, output_path, gpu_id, config
        )
        results.append(result)

        # Log progress
        status = "OK" if result.success else f"FAILED: {result.error_message}"
        logger.info(f"[GPU {gpu_id}] {result.filename}: {status} ({result.processing_time:.2f}s)")

    return results


def get_file_list(config: SpectralConfig) -> List[Tuple[str, str, str, str]]:
    """
    Get list of all files to process.

    Returns
    -------
    List of (source_path, svd_path, rayleigh_path, output_path) tuples
    """
    source_files = sorted(Path(config.source_dir).glob("*.h5"))
    file_list = []

    for source_path in source_files:
        basename = source_path.stem  # e.g., n1024_m112_s0.000000_seed0

        # Construct paths
        svd_path = Path(config.svd_dir) / f"svd_{basename}.h5"
        rayleigh_path = Path(config.rayleigh_dir) / f"{basename}_rayleigh.npz"
        output_path = Path(config.output_dir) / f"spectral_{basename}.npz"

        # Check if all input files exist
        if svd_path.exists() and rayleigh_path.exists():
            # Skip if output already exists
            if not output_path.exists():
                file_list.append((
                    str(source_path),
                    str(svd_path),
                    str(rayleigh_path),
                    str(output_path)
                ))
        else:
            if not svd_path.exists():
                logger.warning(f"Missing SVD file: {svd_path}")
            if not rayleigh_path.exists():
                logger.warning(f"Missing Rayleigh file: {rayleigh_path}")

    return file_list


def run_parallel_processing(config: SpectralConfig):
    """
    Run parallel processing across multiple GPUs.

    Parameters
    ----------
    config : SpectralConfig
        Configuration object
    """
    # Get file list
    file_list = get_file_list(config)
    total_files = len(file_list)

    if total_files == 0:
        logger.info("No files to process. All outputs may already exist.")
        return

    logger.info(f"Found {total_files} files to process using {len(config.gpus)} GPUs")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Distribute files across GPUs
    n_gpus = len(config.gpus)
    batches_per_gpu = [[] for _ in range(n_gpus)]

    for i, file_tuple in enumerate(file_list):
        gpu_idx = i % n_gpus
        batches_per_gpu[gpu_idx].append(file_tuple)

    # Log distribution
    for gpu_idx, batch in enumerate(batches_per_gpu):
        logger.info(f"GPU {config.gpus[gpu_idx]}: {len(batch)} files")

    # Convert config to dict for multiprocessing
    config_dict = {
        'source_dir': config.source_dir,
        'svd_dir': config.svd_dir,
        'output_dir': config.output_dir,
        'clustering_file': config.clustering_file,
        'rayleigh_dir': config.rayleigh_dir,
        'gpus': config.gpus,
        'batch_size': config.batch_size,
        'n_features': config.n_features,
        'n_checkpoints': config.n_checkpoints,
        'significant_projection_threshold': config.significant_projection_threshold,
    }

    # Track progress
    start_time = time.time()
    all_results = []
    completed = 0

    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        futures = {}

        for gpu_idx, batch in enumerate(batches_per_gpu):
            if batch:
                gpu_id = config.gpus[gpu_idx]
                future = executor.submit(worker_process, batch, gpu_id, config_dict)
                futures[future] = gpu_id

        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                completed += len(results)

                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_files - completed) / rate if rate > 0 else 0

                logger.info(
                    f"Progress: {completed}/{total_files} files "
                    f"({100*completed/total_files:.1f}%) | "
                    f"Rate: {rate:.2f} files/s | ETA: {eta/60:.1f} min"
                )

            except Exception as e:
                logger.error(f"Worker for GPU {gpu_id} failed: {e}")

    # Summary
    elapsed = time.time() - start_time
    successful = sum(1 for r in all_results if r.success)
    failed = len(all_results) - successful

    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {elapsed/60:.2f} minutes")
    logger.info(f"Average rate: {total_files/elapsed:.2f} files/second")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_files': total_files,
        'successful': successful,
        'failed': failed,
        'elapsed_seconds': elapsed,
        'gpus_used': config.gpus,
        'failed_files': [
            {'filename': r.filename, 'error': r.error_message}
            for r in all_results if not r.success
        ]
    }

    summary_path = Path(config.output_dir) / 'processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compute refined spectral analysis measures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--gpus', type=str, default='0,1,2,3,4,5,6,7',
        help='Comma-separated list of GPU IDs to use (default: 0,1,2,3,4,5,6,7)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=50,
        help='Number of files to process per GPU batch (default: 50)'
    )
    parser.add_argument(
        '--source-dir', type=str, default='../start',
        help='Directory containing source .h5 files'
    )
    parser.add_argument(
        '--svd-dir', type=str, default='svd_results',
        help='Directory containing SVD .h5 files'
    )
    parser.add_argument(
        '--output-dir', type=str, default='refined_spectral_results',
        help='Directory for output files'
    )
    parser.add_argument(
        '--rayleigh-dir', type=str, default='dynamic_hopping/results/per_file',
        help='Directory containing Rayleigh .npz files'
    )

    args = parser.parse_args()

    # Parse GPU list
    gpus = [int(g.strip()) for g in args.gpus.split(',')]

    # Create config
    config = SpectralConfig(
        source_dir=args.source_dir,
        svd_dir=args.svd_dir,
        output_dir=args.output_dir,
        rayleigh_dir=args.rayleigh_dir,
        gpus=gpus,
        batch_size=args.batch_size,
    )

    logger.info(f"Configuration:")
    logger.info(f"  Source directory: {config.source_dir}")
    logger.info(f"  SVD directory: {config.svd_dir}")
    logger.info(f"  Rayleigh directory: {config.rayleigh_dir}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  GPUs: {config.gpus}")
    logger.info(f"  Batch size: {config.batch_size}")

    # Run processing
    run_parallel_processing(config)


if __name__ == '__main__':
    # Required for multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
