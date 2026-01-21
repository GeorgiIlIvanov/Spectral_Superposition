#!/usr/bin/env python3
"""
Analysis 1: Core Rayleigh Quotient Computation

Computes the Rayleigh quotient κ_i(t) for each feature i at each checkpoint t:
    κ_i(t) = (w_i(t)^T S(t) w_i(t)) / ||w_i(t)||²

where S(t) = W(t) W(t)^T is the feature covariance matrix.

Implementation Notes:
---------------------
Since S = W W^T, we have:
    w_i^T S w_i = w_i^T (W W^T) w_i = (W^T w_i)^T (W^T w_i) = ||W^T w_i||²

So the Rayleigh quotient simplifies to:
    κ_i = ||W^T w_i||² / ||w_i||²

This is computed efficiently on GPU using batched matrix operations.

Outputs:
--------
- rayleigh_quotients.h5: Contains κ_i(t), a_i(t), x_i(t) for all files
- Individual per-file results in results/per_file/
"""

import sys
import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from datetime import datetime

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import load_config

def compute_rayleigh_quotients_gpu(W: torch.Tensor, epsilon: float = 1e-12) -> tuple:
    """
    Compute Rayleigh quotients for all features at all checkpoints using GPU.

    Args:
        W: Weight tensor of shape (T, d, n) where T=checkpoints, d=hidden, n=features
        epsilon: Small constant for numerical stability

    Returns:
        kappa: Rayleigh quotients of shape (T, n)
        a: Squared norms of shape (T, n), a_i = ||w_i||²
        x: Log-transformed quotients of shape (T, n), x_i = log(κ_i + ε)

    Mathematical derivation:
        κ_i = w_i^T S w_i / ||w_i||²
            = w_i^T (W W^T) w_i / ||w_i||²
            = ||W^T w_i||² / ||w_i||²
    """
    T, d, n = W.shape
    device = W.device

    # Compute squared norms: a_i = ||w_i||² = sum_j W[j,i]²
    # Shape: (T, n)
    a = torch.sum(W ** 2, dim=1)  # Sum over hidden dimension

    # Compute Rayleigh quotients
    # For each checkpoint t:
    #   For each feature i: κ_i = ||W^T w_i||² / ||w_i||²
    #   where W^T w_i is a vector of shape (n,) containing dot products

    kappa = torch.zeros((T, n), device=device, dtype=W.dtype)

    for t in range(T):
        W_t = W[t]  # Shape: (d, n)

        # Compute W^T W which is (n, n) - the Gram matrix
        # Then κ_i = (W^T W)[i,i] would give just the diagonal...
        # But we need ||W^T w_i||² = sum_j (W[:,j]^T w_i)² for each i

        # Method: W^T W gives us (n, n) where entry (i,j) = w_i^T w_j
        # We need: sum_j (w_j^T w_i)² for each i = sum over j of (W^T W)[j,i]²
        # This equals: ||col_i of (W^T W)||² = sum_j (W^T W)[j,i]²

        WtW = torch.mm(W_t.T, W_t)  # (n, n)

        # For each feature i, κ_i = sum_j (WtW[j,i])² / a[t,i]
        # = ||column i of WtW||² / a[t,i]
        kappa_t_numerator = torch.sum(WtW ** 2, dim=0)  # Sum over rows for each column
        kappa[t] = kappa_t_numerator / (a[t] + epsilon)

    # Log-transform for stability
    x = torch.log(kappa + epsilon)

    return kappa, a, x


def compute_rayleigh_quotients_efficient(W: torch.Tensor, epsilon: float = 1e-12) -> tuple:
    """
    More memory-efficient version that processes checkpoints one at a time.
    Uses less peak memory for large weight matrices.
    """
    T, d, n = W.shape
    device = W.device

    # Compute squared norms
    a = torch.sum(W ** 2, dim=1)

    kappa = torch.zeros((T, n), device=device, dtype=W.dtype)

    for t in range(T):
        W_t = W[t]  # (d, n)

        # Compute M = W^T W  in blocks if n is large
        # For n=1024, d=112, this is manageable
        WtW = torch.mm(W_t.T, W_t)  # (n, n)

        # κ_i = sum_j (WtW[j,i])² / a_i = ||WtW[:,i]||² / a_i
        kappa[t] = torch.sum(WtW ** 2, dim=0) / (a[t] + epsilon)

        # Free intermediate memory
        del WtW

    x = torch.log(kappa + epsilon)

    return kappa, a, x


def process_single_file(filepath: Path, device: torch.device, cfg: dict) -> dict:
    """
    Process a single HDF5 file and compute Rayleigh quotients.

    Args:
        filepath: Path to HDF5 file
        device: PyTorch device (cuda:X or cpu)
        cfg: Configuration dictionary

    Returns:
        Dictionary containing all computed quantities
    """
    with h5py.File(filepath, 'r') as f:
        # Load data
        weights = f['weights'][:]  # (T, d, n)
        checkpoint_steps = f['checkpoint_steps'][:]
        feature_norms_precomputed = f['feature_norms'][:]  # Already have ||W_i||²
        fractional_dims = f['fractional_dims'][:]

        # Attributes
        sparsity = float(f.attrs['sparsity'])
        m_hidden = int(f.attrs['m_hidden'])
        n_features = int(f.attrs['n_features'])
        seed = int(f.attrs['seed'])

    T, d, n = weights.shape

    # Move to GPU
    W = torch.tensor(weights, dtype=torch.float32, device=device)

    # Compute Rayleigh quotients
    kappa, a, x = compute_rayleigh_quotients_efficient(W, cfg['epsilon'])

    # Move back to CPU/numpy
    kappa_np = kappa.cpu().numpy()
    a_np = a.cpu().numpy()
    x_np = x.cpu().numpy()

    # Verify a matches precomputed norms (sanity check)
    norm_diff = np.abs(a_np - feature_norms_precomputed).max()

    return {
        'filepath': str(filepath),
        'filename': filepath.name,
        'sparsity': sparsity,
        'm_hidden': m_hidden,
        'n_features': n_features,
        'seed': seed,
        'n_checkpoints': T,
        'checkpoint_steps': checkpoint_steps.tolist(),
        'kappa': kappa_np,  # (T, n) Rayleigh quotients
        'a': a_np,          # (T, n) squared norms
        'x': x_np,          # (T, n) log-transformed quotients
        'fractional_dims': fractional_dims,  # (T, n) pre-computed D_i
        'norm_verification_diff': float(norm_diff),
    }


def main():
    parser = argparse.ArgumentParser(description='Compute Rayleigh quotients for feature hopping analysis')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
    parser.add_argument('--sample', type=int, default=None, help='Process only N files (for testing)')
    parser.add_argument('--output-prefix', type=str, default='', help='Prefix for output files')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config()

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    cfg['plots_dir'].mkdir(parents=True, exist_ok=True)
    cfg['results_dir'].mkdir(parents=True, exist_ok=True)
    per_file_dir = cfg['results_dir'] / 'per_file'
    per_file_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Analysis 1: Rayleigh Quotient Computation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Epsilon: {cfg['epsilon']}")
    print(f"  Z-threshold: {cfg['z_threshold']}")
    print(f"  Input dir: {cfg['input_dir']}")

    # Get all HDF5 files
    files = sorted(cfg['input_dir'].glob('n1024_m*.h5'))
    print(f"\nFound {len(files)} experiment files")

    if args.sample:
        files = files[:args.sample]
        print(f"Sampling {len(files)} files for testing")

    # Process all files
    all_results = []
    aggregated_kappa = []
    aggregated_x = []

    for filepath in tqdm(files, desc="Computing Rayleigh quotients"):
        try:
            result = process_single_file(filepath, device, cfg)
            all_results.append({
                'filename': result['filename'],
                'sparsity': result['sparsity'],
                'm_hidden': result['m_hidden'],
                'seed': result['seed'],
                'n_checkpoints': result['n_checkpoints'],
                'checkpoint_steps': result['checkpoint_steps'],
                'norm_verification_diff': result['norm_verification_diff'],
            })

            # Save per-file results
            per_file_path = per_file_dir / f"{filepath.stem}_rayleigh.npz"
            np.savez_compressed(
                per_file_path,
                kappa=result['kappa'],
                a=result['a'],
                x=result['x'],
                fractional_dims=result['fractional_dims'],
                checkpoint_steps=np.array(result['checkpoint_steps']),
                sparsity=result['sparsity'],
                m_hidden=result['m_hidden'],
                seed=result['seed'],
            )

            aggregated_kappa.append(result['kappa'])
            aggregated_x.append(result['x'])

        except Exception as e:
            print(f"\nError processing {filepath.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nSuccessfully processed {len(all_results)} files")

    # Compute aggregate statistics
    print("\n--- Computing Aggregate Statistics ---")

    if len(aggregated_kappa) > 0:
        # Stack all kappa values: (n_files, T, n)
        all_kappa = np.stack(aggregated_kappa, axis=0)
        all_x = np.stack(aggregated_x, axis=0)

        # Global statistics
        mean_kappa = np.nanmean(all_kappa, axis=(0, 2))  # Mean over files and features, shape (T,)
        std_kappa = np.nanstd(all_kappa, axis=(0, 2))

        mean_x = np.nanmean(all_x, axis=(0, 2))
        std_x = np.nanstd(all_x, axis=(0, 2))

        # Per-feature statistics across time
        kappa_temporal_std = np.nanstd(all_kappa, axis=1)  # Std over time for each file/feature
        mean_temporal_volatility = np.nanmean(kappa_temporal_std)

        print(f"  Mean κ over all: {np.nanmean(all_kappa):.4f}")
        print(f"  Std κ over all: {np.nanstd(all_kappa):.4f}")
        print(f"  Mean temporal volatility: {mean_temporal_volatility:.4f}")

        # Save aggregate results
        aggregate_path = cfg['results_dir'] / f'{args.output_prefix}rayleigh_aggregate.npz'
        np.savez_compressed(
            aggregate_path,
            mean_kappa_over_time=mean_kappa,
            std_kappa_over_time=std_kappa,
            mean_x_over_time=mean_x,
            std_x_over_time=std_x,
            n_files=len(all_results),
        )
        print(f"\nSaved aggregate results to: {aggregate_path}")

    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'n_files_processed': len(all_results),
        'config': {
            'epsilon': cfg['epsilon'],
            'z_threshold': cfg['z_threshold'],
        },
        'per_file_summary': all_results,
    }

    summary_path = cfg['results_dir'] / f'{args.output_prefix}rayleigh_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")

    print("\n" + "=" * 70)
    print("Analysis 1 Complete: Rayleigh Quotients Computed")
    print("=" * 70)


if __name__ == '__main__':
    main()
