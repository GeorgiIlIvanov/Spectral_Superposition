#!/usr/bin/env python3
"""
Phase 2: Feature-to-Eigenspace Clustering

For each checkpoint, determine which eigenspace each feature predominantly lives in.
This is done by projecting each feature column W_i onto the left singular vectors U
and finding which singular direction(s) capture most of its variance.

Output:
    - cluster_assignments.h5: For each (file, checkpoint, feature), the dominant eigenspace index
    - projection_strengths.h5: The projection coefficients |U^T @ W_i|^2 / ||W_i||^2

Usage:
    python 02_cluster_features.py [--threshold 0.5]
"""

import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from datetime import datetime
from collections import defaultdict


# Configuration
INPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
SVD_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/svd_results')
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/clustering_results')


def compute_feature_projections(W, U):
    """
    Compute how much each feature projects onto each eigenspace.

    Args:
        W: (m, n) weight matrix
        U: (m, m) left singular vectors (columns are eigenvectors of WW^T)

    Returns:
        projections: (n, m) where projections[i, k] = |u_k^T @ W_i|^2
        normalized: (n, m) where normalized[i, k] = |u_k^T @ W_i|^2 / ||W_i||^2
    """
    m, n = W.shape

    # W_i is the i-th column of W (feature i's representation)
    # Project onto each left singular vector: U^T @ W gives (m, n)
    # Each column k of result is projections of all features onto u_k
    projections_matrix = U.T @ W  # (m, n)

    # Square to get projection strengths
    proj_squared = projections_matrix ** 2  # (m, n)

    # Transpose to get (n, m) - projections[i, k] = projection of feature i onto eigenspace k
    projections = proj_squared.T  # (n, m)

    # Compute feature norms for normalization
    feature_norms_sq = np.sum(W ** 2, axis=0)  # (n,)

    # Avoid division by zero
    feature_norms_sq = np.maximum(feature_norms_sq, 1e-10)

    # Normalized projections: fraction of feature variance in each eigenspace
    normalized = projections / feature_norms_sq[:, np.newaxis]  # (n, m)

    return projections, normalized


def assign_features_to_eigenspaces(normalized_projections, threshold=0.5):
    """
    Assign each feature to its dominant eigenspace(s).

    Args:
        normalized_projections: (n, m) - fraction of variance in each eigenspace
        threshold: minimum fraction to be considered "in" an eigenspace

    Returns:
        dominant_idx: (n,) - index of dominant eigenspace for each feature
        is_mixed: (n,) - True if feature has significant projection on multiple eigenspaces
    """
    n, m = normalized_projections.shape

    # Find dominant eigenspace (argmax)
    dominant_idx = np.argmax(normalized_projections, axis=1)  # (n,)

    # Check if feature is "mixed" (significant projection on multiple eigenspaces)
    # A feature is mixed if its second-largest projection is > threshold * largest
    sorted_projs = np.sort(normalized_projections, axis=1)[:, ::-1]  # descending
    is_mixed = sorted_projs[:, 1] > threshold * sorted_projs[:, 0]

    return dominant_idx, is_mixed


def process_single_file(input_path, svd_path, threshold=0.5):
    """Process a single file and compute clustering for all checkpoints."""

    # Load original weights
    with h5py.File(input_path, 'r') as f:
        weights = f['weights'][:]  # (56, m, n)
        checkpoint_steps = f['checkpoint_steps'][:]
        fractional_dims = f['fractional_dims'][:]  # (56, n)
        feature_norms = f['feature_norms'][:]  # (56, n)
        m_hidden = int(f.attrs['m_hidden'])
        sparsity = float(f.attrs['sparsity'])
        seed = int(f.attrs['seed'])

    # Load SVD results
    with h5py.File(svd_path, 'r') as f:
        U = f['U'][:]  # (56, m, m)
        S = f['S'][:]  # (56, m) - singular values
        eigenvalues = f['eigenvalues'][:]  # (56, m) = S^2

    n_checkpoints, m, n = weights.shape

    results = {
        'dominant_eigenspace': np.zeros((n_checkpoints, n), dtype=np.int16),
        'is_mixed': np.zeros((n_checkpoints, n), dtype=bool),
        'max_projection_frac': np.zeros((n_checkpoints, n), dtype=np.float32),
        'eigenvalue_of_dominant': np.zeros((n_checkpoints, n), dtype=np.float32),
    }

    for t in range(n_checkpoints):
        W_t = weights[t]  # (m, n)
        U_t = U[t]  # (m, m)
        eigs_t = eigenvalues[t]  # (m,)

        # Compute projections
        _, normalized = compute_feature_projections(W_t, U_t)  # (n, m)

        # Assign to eigenspaces
        dominant_idx, is_mixed = assign_features_to_eigenspaces(normalized, threshold)

        # Get the maximum projection fraction
        max_proj_frac = np.max(normalized, axis=1)

        # Get eigenvalue of dominant eigenspace for each feature
        eigenvalue_of_dominant = eigs_t[dominant_idx]

        results['dominant_eigenspace'][t] = dominant_idx
        results['is_mixed'][t] = is_mixed
        results['max_projection_frac'][t] = max_proj_frac
        results['eigenvalue_of_dominant'][t] = eigenvalue_of_dominant

    return {
        'm_hidden': m_hidden,
        'sparsity': sparsity,
        'seed': seed,
        'checkpoint_steps': checkpoint_steps,
        'fractional_dims': fractional_dims,
        'feature_norms': feature_norms,
        'eigenvalues': eigenvalues,
        **results
    }


def main():
    parser = argparse.ArgumentParser(description='Cluster features by eigenspace')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Threshold for mixed assignment (default: 0.3)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Process only N files for testing')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2: Feature-to-Eigenspace Clustering")
    print("=" * 60)
    print(f"Threshold: {args.threshold}")

    # Get list of SVD files
    svd_files = sorted(SVD_DIR.glob('svd_n1024_m*.h5'))
    print(f"Found {len(svd_files)} SVD files")

    if args.sample:
        svd_files = svd_files[:args.sample]
        print(f"Sampling {len(svd_files)} files for testing")

    # Process all files
    all_results = []

    for svd_path in tqdm(svd_files, desc="Processing files"):
        # Construct input path from SVD path
        input_name = svd_path.stem.replace('svd_', '') + '.h5'
        input_path = INPUT_DIR / input_name

        if not input_path.exists():
            print(f"Warning: Input file not found: {input_path}")
            continue

        try:
            result = process_single_file(input_path, svd_path, args.threshold)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {svd_path.name}: {e}")
            continue

    print(f"\nSuccessfully processed {len(all_results)} files")

    # Aggregate statistics
    print("\n--- Computing Aggregate Statistics ---")

    # Group by (m_hidden, sparsity) - average over seeds
    grouped = defaultdict(list)
    for r in all_results:
        key = (r['m_hidden'], r['sparsity'])
        grouped[key].append(r)

    # Compute statistics for final checkpoint
    stats = []
    for (m_hidden, sparsity), results_list in grouped.items():
        # Average over seeds
        mixed_fracs = [np.mean(r['is_mixed'][-1]) for r in results_list]
        max_proj_fracs = [np.mean(r['max_projection_frac'][-1]) for r in results_list]

        # Count features per eigenspace
        eigenspace_counts = defaultdict(list)
        for r in results_list:
            unique, counts = np.unique(r['dominant_eigenspace'][-1], return_counts=True)
            for u, c in zip(unique, counts):
                eigenspace_counts[u].append(c)

        # Number of "active" eigenspaces (with >1% of features)
        n_features = results_list[0]['fractional_dims'].shape[1]
        active_eigenspaces = sum(1 for k, v in eigenspace_counts.items()
                                  if np.mean(v) > 0.01 * n_features)

        stats.append({
            'm_hidden': m_hidden,
            'sparsity': sparsity,
            'mixed_fraction_mean': float(np.mean(mixed_fracs)),
            'mixed_fraction_std': float(np.std(mixed_fracs)),
            'max_projection_mean': float(np.mean(max_proj_fracs)),
            'max_projection_std': float(np.std(max_proj_fracs)),
            'active_eigenspaces': active_eigenspaces,
        })

    # Save aggregate stats
    stats.sort(key=lambda x: (x['m_hidden'], x['sparsity']))
    with open(OUTPUT_DIR / 'clustering_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Save detailed results to HDF5
    print("\n--- Saving Detailed Results ---")

    with h5py.File(OUTPUT_DIR / 'clustering_results.h5', 'w') as f:
        for i, r in enumerate(tqdm(all_results, desc="Saving")):
            grp_name = f"m{r['m_hidden']}_s{r['sparsity']:.6f}_seed{r['seed']}"
            grp = f.create_group(grp_name)

            grp.attrs['m_hidden'] = r['m_hidden']
            grp.attrs['sparsity'] = r['sparsity']
            grp.attrs['seed'] = r['seed']

            grp.create_dataset('checkpoint_steps', data=r['checkpoint_steps'])
            grp.create_dataset('dominant_eigenspace', data=r['dominant_eigenspace'], compression='lzf')
            grp.create_dataset('is_mixed', data=r['is_mixed'], compression='lzf')
            grp.create_dataset('max_projection_frac', data=r['max_projection_frac'], compression='lzf')
            grp.create_dataset('eigenvalue_of_dominant', data=r['eigenvalue_of_dominant'], compression='lzf')
            grp.create_dataset('fractional_dims', data=r['fractional_dims'], compression='lzf')
            grp.create_dataset('feature_norms', data=r['feature_norms'], compression='lzf')
            grp.create_dataset('eigenvalues', data=r['eigenvalues'], compression='lzf')

    # Print summary
    print("\n" + "=" * 60)
    print("Phase 2 Complete")
    print("=" * 60)
    print(f"Files processed: {len(all_results)}")
    print(f"Output saved to: {OUTPUT_DIR}")
    print(f"  - clustering_stats.json (aggregate statistics)")
    print(f"  - clustering_results.h5 (detailed per-file results)")

    # Sample statistics
    if stats:
        mixed_overall = np.mean([s['mixed_fraction_mean'] for s in stats])
        proj_overall = np.mean([s['max_projection_mean'] for s in stats])
        print(f"\nOverall statistics (final checkpoint):")
        print(f"  Mean mixed fraction: {mixed_overall:.3f}")
        print(f"  Mean max projection: {proj_overall:.3f}")


if __name__ == '__main__':
    main()
