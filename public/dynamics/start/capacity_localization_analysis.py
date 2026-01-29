#!/usr/bin/env python3
"""
GPU-accelerated Capacity Localization Analysis for Toy Models of Superposition.

This script implements the theoretical framework for analyzing rank defect vs
delocalization defect in ReLU(W^T W x + b) models trained on sparse inputs.

Uses PyTorch for GPU-accelerated eigendecomposition across multiple GPUs.

Steps implemented:
- Step 0: Compute rank defect vs delocalization defect per run
- Step 1: Per-feature localization diagnostics (leverage, Rayleigh quotient, residual, slack)
- Step 2: Leverage-weighted aggregates for regime structure analysis
- Step 3: Diracness proxy validation

Author: Claude Code Analysis Pipeline
Date: 2026-01-28
"""

import argparse
import math
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import h5py
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')


def parse_filename(path: Path) -> Dict:
    """Parse experiment filename to extract parameters."""
    parts = path.stem.split('_')
    return dict(
        n=int(parts[0][1:]),
        m=int(parts[1][1:]),
        s=float(parts[2][1:]),
        seed=int(parts[3][4:]),
    )


def group_eigenvalues(lam: torch.Tensor, rtol: float = 1e-6, atol: float = 1e-10) -> List[torch.Tensor]:
    """
    Group eigenvalues by near-degeneracy.

    This is important for Step 3 where we need to distinguish true delocalization
    from eigenspace mixing within degenerate subspaces.

    Args:
        lam: Sorted eigenvalues (descending), positive only
        rtol: Relative tolerance for grouping
        atol: Absolute tolerance for grouping

    Returns:
        List of index arrays, one per eigenvalue group
    """
    if len(lam) == 0:
        return []

    groups = []
    start = 0
    lam_cpu = lam.cpu().numpy()

    for i in range(1, len(lam_cpu)):
        if not np.isclose(lam_cpu[i], lam_cpu[start], rtol=rtol, atol=atol):
            groups.append(torch.arange(start, i, device=lam.device))
            start = i
    groups.append(torch.arange(start, len(lam_cpu), device=lam.device))

    return groups


def process_single_file(fpath: Path, device: torch.device, compute_feature_metrics: bool = True) -> Tuple[Dict, List[Dict]]:
    """
    Process a single experiment file on the specified GPU.

    Implements Steps 0, 1, 2, 3 metrics computation.

    Args:
        fpath: Path to HDF5 file
        device: PyTorch device (GPU)
        compute_feature_metrics: Whether to compute per-feature metrics (Step 1, 3)

    Returns:
        Tuple of (run_metrics_dict, list_of_feature_metrics_dicts)
    """
    meta = parse_filename(fpath)

    try:
        with h5py.File(fpath, "r") as hf:
            # Load final checkpoint
            W = torch.tensor(hf["weights"][-1], dtype=torch.float64, device=device)  # (m, n)
            D = torch.tensor(hf["fractional_dims"][-1], dtype=torch.float64, device=device)  # (n,)

            m = int(hf.attrs["m_hidden"])
            s = float(hf.attrs["sparsity"])
            n = int(hf.attrs["n_features"])
    except Exception as e:
        return None, []

    # ========== STEP 0: Rank and Saturation Metrics ==========

    # Frame operator F = W @ W^T
    F = W @ W.T  # (m, m)

    # Eigendecomposition (GPU-accelerated)
    # torch.linalg.eigh returns eigenvalues in ascending order
    lam, U = torch.linalg.eigh(F)

    # Reverse to descending order
    lam = torch.flip(lam, [0])
    U = torch.flip(U, [1])

    # Compute effective rank
    lam_max = lam[0].item() if lam.numel() > 0 else 0.0
    tol = 1e-8 * lam_max
    pos_mask = lam > tol
    lam_pos = lam[pos_mask]
    U_pos = U[:, pos_mask]
    r = int(lam_pos.numel())

    # Saturation metrics
    sumD = D.sum().item()
    rho_m = sumD / m if m > 0 else float('nan')
    rho_r = sumD / r if r > 0 else float('nan')
    rank_ratio = r / m if m > 0 else float('nan')

    # ========== STEP 1: Per-Feature Localization Diagnostics ==========

    if r == 0:
        # Degenerate case: no positive eigenvalues
        run_metrics = dict(
            file=fpath.name, m=m, s=s, seed=meta["seed"], n=n,
            rank=r, rank_ratio=rank_ratio,
            sumD=sumD, rho_m=rho_m, rho_r=rho_r,
            slack_run=float('nan'),
            mean_resid=float('nan'),
            mean_sigma=float('nan'),
            mean_kappa=float('nan'),
            mean_leverage=float('nan'),
            tail_mass_001=float('nan'),
            tail_mass_005=float('nan'),
            tail_mass_010=float('nan'),
        )
        return run_metrics, []

    # Project features into positive eigenspace: Z = U_pos^T @ W, shape (r, n)
    Z = U_pos.T @ W
    Z2 = Z * Z  # squared coefficients

    # Feature norms in eigenspace (should equal ||w_i||^2)
    norm2 = Z2.sum(dim=0)  # (n,)

    # Filter alive features (non-zero norm)
    alive_mask = norm2 > 1e-14
    alive_indices = torch.where(alive_mask)[0]

    # Initialize metrics tensors
    ell = torch.zeros(n, dtype=torch.float64, device=device)  # leverage
    kappa = torch.zeros(n, dtype=torch.float64, device=device)  # Rayleigh quotient
    resid = torch.zeros(n, dtype=torch.float64, device=device)  # eigenvector residual
    sigma = torch.full((n,), float('nan'), dtype=torch.float64, device=device)  # relative slack

    # Compute leverage: ell_i = sum_k (u_k^T w_i)^2 / lambda_k = sum_k Z2[k,i] / lam_pos[k]
    # This is w_i^T F^+ w_i without explicit pseudoinverse
    inv_lam = 1.0 / lam_pos  # (r,)
    ell[alive_mask] = (Z2[:, alive_mask] * inv_lam[:, None]).sum(dim=0)

    # Compute Rayleigh quotient: kappa_i = (w_i^T F w_i) / ||w_i||^2 = E_{mu_i}[lambda]
    # = sum_k Z2[k,i] * lam_pos[k] / norm2[i]
    kappa[alive_mask] = (Z2[:, alive_mask] * lam_pos[:, None]).sum(dim=0) / norm2[alive_mask]

    # Compute eigenvector residual: res_i = ||F w_i - kappa_i w_i|| / ||w_i||
    # This equals sqrt(Var_{mu_i}(lambda))
    # Var = E[lambda^2] - (E[lambda])^2
    E2 = torch.zeros(n, dtype=torch.float64, device=device)
    E2[alive_mask] = (Z2[:, alive_mask] * (lam_pos[:, None] ** 2)).sum(dim=0) / norm2[alive_mask]
    var = torch.clamp(E2 - kappa ** 2, min=0.0)
    resid = torch.sqrt(var)

    # Compute relative slack: sigma_i = 1 - D_i / ell_i
    # This is the normalized Cauchy-Schwarz slack
    ratio = torch.full((n,), float('nan'), dtype=torch.float64, device=device)
    valid_ell_mask = alive_mask & (ell > 0)
    ratio[valid_ell_mask] = D[valid_ell_mask] / ell[valid_ell_mask]
    sigma = 1.0 - ratio

    # ========== STEP 2: Leverage-Weighted Aggregates ==========

    ell_alive = ell[alive_mask]
    ell_sum = ell_alive.sum().item()

    # Slack run: should equal 1 - rho_r
    slack_run = 1.0 - rho_r if np.isfinite(rho_r) else float('nan')

    # Weighted mean functions
    def wmean(x):
        if ell_sum == 0:
            return float('nan')
        return (ell_alive * x[alive_mask]).sum().item() / ell_sum

    # Mean residual (leverage-weighted)
    mean_resid = wmean(resid)

    # Mean sigma (leverage-weighted, treat NaN as 0)
    sigma_clean = torch.nan_to_num(sigma, nan=0.0)
    mean_sigma = wmean(sigma_clean)

    # Mean kappa and leverage for diagnostics
    mean_kappa = wmean(kappa)
    mean_leverage = ell_alive.mean().item() if ell_alive.numel() > 0 else float('nan')

    # Tail mass curves: fraction of leverage in high-slack features
    def tail_mass(tau):
        high_slack_mask = alive_mask & (sigma >= tau)
        if ell_sum == 0:
            return float('nan')
        return ell[high_slack_mask].sum().item() / r if r > 0 else float('nan')

    tail_001 = tail_mass(0.01)
    tail_005 = tail_mass(0.05)
    tail_010 = tail_mass(0.10)

    run_metrics = dict(
        file=fpath.name, m=m, s=s, seed=meta["seed"], n=n,
        rank=r, rank_ratio=rank_ratio,
        sumD=sumD, rho_m=rho_m, rho_r=rho_r,
        slack_run=slack_run,
        mean_resid=mean_resid,
        mean_sigma=mean_sigma,
        mean_kappa=mean_kappa,
        mean_leverage=mean_leverage,
        tail_mass_001=tail_001,
        tail_mass_005=tail_005,
        tail_mass_010=tail_010,
    )

    feat_metrics_list = []

    if compute_feature_metrics:
        # ========== STEP 3: Diracness Proxy via Eigenvalue Grouping ==========

        groups = group_eigenvalues(lam_pos, rtol=1e-6, atol=tol)

        # Compute q_i = max_group p_{i, group} (Diracness proxy)
        q = torch.zeros(n, dtype=torch.float64, device=device)

        if r > 0 and len(groups) > 0:
            pmax = torch.zeros(n, dtype=torch.float64, device=device)
            for g in groups:
                # Group probability: sum of squared coefficients in this eigenspace
                pg = torch.zeros(n, dtype=torch.float64, device=device)
                pg[alive_mask] = Z2[g][:, alive_mask].sum(dim=0) / norm2[alive_mask]
                pmax = torch.maximum(pmax, pg)
            q = pmax

        # Collect feature metrics (move to CPU for storage)
        D_cpu = D.cpu().numpy()
        ell_cpu = ell.cpu().numpy()
        ratio_cpu = ratio.cpu().numpy()
        sigma_cpu = sigma.cpu().numpy()
        kappa_cpu = kappa.cpu().numpy()
        resid_cpu = resid.cpu().numpy()
        q_cpu = q.cpu().numpy()
        alive_cpu = alive_mask.cpu().numpy()

        for i in range(n):
            if not alive_cpu[i]:
                continue
            feat_metrics_list.append(dict(
                file=fpath.name, m=m, s=s, seed=meta["seed"], i=i,
                D=float(D_cpu[i]), ell=float(ell_cpu[i]), ratio=float(ratio_cpu[i]),
                sigma=float(sigma_cpu[i]), kappa=float(kappa_cpu[i]), resid=float(resid_cpu[i]),
                q=float(q_cpu[i]),
            ))

    return run_metrics, feat_metrics_list


def worker_process(gpu_id: int, file_queue: mp.Queue, result_queue: mp.Queue, compute_feature_metrics: bool):
    """Worker process that processes files on a specific GPU."""
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    while True:
        try:
            fpath = file_queue.get(timeout=1)
            if fpath is None:  # Poison pill
                break

            run_metrics, feat_metrics = process_single_file(fpath, device, compute_feature_metrics)
            result_queue.put((run_metrics, feat_metrics))
        except Exception as e:
            # Queue timeout or processing error
            if file_queue.empty():
                break
            continue


def run_analysis_multiprocess(
    files: List[Path],
    num_gpus: int,
    compute_feature_metrics: bool = True,
    show_progress: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run analysis across multiple GPUs using multiprocessing.

    Args:
        files: List of HDF5 file paths
        num_gpus: Number of GPUs to use
        compute_feature_metrics: Whether to compute per-feature metrics
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (run_metrics_list, feature_metrics_list)
    """
    # For simplicity and stability, use sequential processing with GPU round-robin
    # This avoids multiprocessing complexities with CUDA

    run_metrics_list = []
    feat_metrics_list = []

    pbar = tqdm(files, desc="Processing experiments", disable=not show_progress)

    for idx, fpath in enumerate(pbar):
        gpu_id = idx % num_gpus
        device = torch.device(f'cuda:{gpu_id}')

        run_metrics, feat_metrics = process_single_file(fpath, device, compute_feature_metrics)

        if run_metrics is not None:
            run_metrics_list.append(run_metrics)
            feat_metrics_list.extend(feat_metrics)

        # Update progress bar description with current file info
        if run_metrics:
            pbar.set_postfix(m=run_metrics['m'], s=f"{run_metrics['s']:.2f}", gpu=gpu_id)

    return run_metrics_list, feat_metrics_list


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated Capacity Localization Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python capacity_localization_analysis.py --data-dir ./start --out ./results
    python capacity_localization_analysis.py --data-dir ./start --out ./results --save-feature-metrics --num-gpus 8
        """
    )

    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing experiment HDF5 files")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for results")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of files to process (for testing)")
    parser.add_argument("--save-feature-metrics", action="store_true",
                        help="Save per-feature metrics (larger output)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all available)")
    parser.add_argument("--pattern", type=str, default="n1024_m*_s*_seed*.h5",
                        help="Glob pattern for finding experiment files")

    args = parser.parse_args()

    # Detect available GPUs
    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")

    print(f"=" * 60)
    print("Capacity Localization Analysis")
    print(f"=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.out}")
    print(f"Using {num_gpus} GPU(s)")
    print(f"Save feature metrics: {args.save_feature_metrics}")

    # Find experiment files
    files = sorted(args.data_dir.glob(args.pattern))
    if args.max_files:
        files = files[:args.max_files]

    print(f"Found {len(files)} experiment files")

    if len(files) == 0:
        print("No files found. Check --data-dir and --pattern.")
        return

    # Run analysis
    print(f"\nProcessing files...")
    run_metrics_list, feat_metrics_list = run_analysis_multiprocess(
        files, num_gpus, args.save_feature_metrics, show_progress=True
    )

    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)

    # Save run metrics
    runs_df = pd.DataFrame(run_metrics_list)
    runs_df.to_csv(args.out / "run_metrics.csv", index=False)
    print(f"\nSaved run metrics to {args.out / 'run_metrics.csv'}")
    print(f"  Total runs: {len(runs_df)}")

    # Save feature metrics if requested
    if args.save_feature_metrics and feat_metrics_list:
        feats_df = pd.DataFrame(feat_metrics_list)
        feats_df.to_csv(args.out / "feature_metrics.csv", index=False)
        print(f"Saved feature metrics to {args.out / 'feature_metrics.csv'}")
        print(f"  Total features: {len(feats_df)}")

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print("Summary Statistics")
    print(f"{'=' * 60}")

    print("\nRun-level metrics:")
    for col in ['rank_ratio', 'rho_m', 'rho_r', 'slack_run', 'mean_resid', 'mean_sigma']:
        if col in runs_df.columns:
            vals = runs_df[col].dropna()
            print(f"  {col:15s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                  f"min={vals.min():.4f}, max={vals.max():.4f}")

    # Group by sparsity bins for quick overview
    print("\nBy sparsity range:")
    runs_df['s_bin'] = pd.cut(runs_df['s'], bins=[0, 0.3, 0.6, 0.9, 1.0],
                              labels=['low (0-0.3)', 'mid (0.3-0.6)', 'high (0.6-0.9)', 'very high (0.9-1.0)'])
    summary = runs_df.groupby('s_bin', observed=True)[['rho_m', 'rho_r', 'rank_ratio', 'mean_sigma']].mean()
    print(summary.to_string())

    # Save metadata
    metadata = {
        'analysis_date': datetime.now().isoformat(),
        'data_directory': str(args.data_dir.absolute()),
        'output_directory': str(args.out.absolute()),
        'num_files_processed': len(runs_df),
        'num_gpus_used': num_gpus,
        'save_feature_metrics': args.save_feature_metrics,
        'file_pattern': args.pattern,
    }

    with open(args.out / "analysis_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAnalysis complete. End time: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
