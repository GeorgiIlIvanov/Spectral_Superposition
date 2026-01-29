#!/usr/bin/env python3
"""
GPU-accelerated Capacity Localization Analysis for Toy Models of Superposition (v2).

Fixed version implementing:
- Binned spectral measure for basis-invariant Diracness proxy (q_bin, H_bin, n_eff)
- Normalized spectral spread (coefficient of variation, cv)
- Defect identity verification (gap1 vs gap2)
- D vs D_hat sanity checks for Plot C
- Assignment test metrics (b_star, lambda_hat, qual_rate)

Uses PyTorch for GPU-accelerated eigendecomposition across multiple GPUs.

Author: Claude Code Analysis Pipeline
Date: 2026-01-29
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


# Configuration
DEFAULT_NUM_BINS = 60
DEFAULT_EIGENVALUE_TOL_FACTOR = 1e-8
LOCALIZATION_THRESHOLD = 0.9  # q_bin >= this for "confidently localized"


def parse_filename(path: Path) -> Dict:
    """Parse experiment filename to extract parameters."""
    parts = path.stem.split('_')
    return dict(
        n=int(parts[0][1:]),
        m=int(parts[1][1:]),
        s=float(parts[2][1:]),
        seed=int(parts[3][4:]),
    )


def compute_binned_spectral_measure(
    lam_pos: torch.Tensor,
    Z2: torch.Tensor,
    norm2: torch.Tensor,
    alive_mask: torch.Tensor,
    num_bins: int = DEFAULT_NUM_BINS,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Compute binned spectral measure for basis-invariant Diracness proxy.

    Given eigenpairs (lambda_k, u_k), and projections z_k,i = u_k^T w_i:
    - p_k,i = z_k,i^2 / ||w_i||^2 (probability measure on eigenvalues)
    - For log-spaced bins B_b over lambda, compute mu_i(B_b) = sum_{k: lambda_k in B_b} p_k,i

    Returns:
        Dict with tensors for q_bin, H_bin, n_eff, b_star, lambda_hat
    """
    r = lam_pos.numel()
    n = norm2.numel()

    if device is None:
        device = lam_pos.device

    # Initialize outputs
    q_bin = torch.zeros(n, dtype=torch.float64, device=device)
    H_bin = torch.zeros(n, dtype=torch.float64, device=device)
    n_eff = torch.zeros(n, dtype=torch.float64, device=device)
    b_star = torch.full((n,), -1, dtype=torch.int64, device=device)
    lambda_hat = torch.zeros(n, dtype=torch.float64, device=device)

    if r == 0 or not alive_mask.any():
        return dict(
            q_bin=q_bin, H_bin=H_bin, n_eff=n_eff,
            b_star=b_star, lambda_hat=lambda_hat,
            bin_edges=torch.zeros(1, device=device)
        )

    # Get positive eigenvalues range
    lam_min = lam_pos[-1].item()  # smallest positive (lam_pos is descending)
    lam_max = lam_pos[0].item()

    if lam_min <= 0 or lam_max <= 0 or lam_min >= lam_max:
        # Degenerate case: all same eigenvalue
        # Everything in one bin
        q_bin[alive_mask] = 1.0
        H_bin[alive_mask] = 0.0
        n_eff[alive_mask] = 1.0
        b_star[alive_mask] = 0
        lambda_hat[alive_mask] = lam_max
        return dict(
            q_bin=q_bin, H_bin=H_bin, n_eff=n_eff,
            b_star=b_star, lambda_hat=lambda_hat,
            bin_edges=torch.tensor([lam_min, lam_max], device=device)
        )

    # Create log-spaced bin edges
    log_min = math.log10(lam_min)
    log_max = math.log10(lam_max)
    edges_np = np.logspace(log_min, log_max, num_bins + 1)
    edges = torch.tensor(edges_np, dtype=torch.float64, device=device)

    # Compute bin centers (geometric mean of edges)
    bin_centers = torch.sqrt(edges[:-1] * edges[1:])  # (num_bins,)

    # Assign each eigenvalue to a bin
    # Use searchsorted: b = searchsorted(edges, lam, side='right') - 1, clamp to [0, num_bins-1]
    lam_pos_cpu = lam_pos.cpu().numpy()
    bin_indices_np = np.searchsorted(edges_np, lam_pos_cpu, side='right') - 1
    bin_indices_np = np.clip(bin_indices_np, 0, num_bins - 1)
    bin_indices = torch.tensor(bin_indices_np, dtype=torch.int64, device=device)

    # Compute per-feature probability measure p_k,i = Z2[k,i] / norm2[i]
    # Only for alive features
    p = torch.zeros_like(Z2)  # (r, n)
    p[:, alive_mask] = Z2[:, alive_mask] / norm2[alive_mask].unsqueeze(0)

    # Aggregate into bins for each feature: mu_i(B_b) = sum_{k: bin[k]=b} p[k,i]
    # Shape: (num_bins, n)
    mu = torch.zeros(num_bins, n, dtype=torch.float64, device=device)
    for b in range(num_bins):
        mask_b = bin_indices == b
        if mask_b.any():
            mu[b] = p[mask_b].sum(dim=0)

    # q_bin[i] = max_b mu_i(B_b)
    q_bin, b_star_temp = mu.max(dim=0)
    b_star[alive_mask] = b_star_temp[alive_mask]

    # lambda_hat[i] = bin_centers[b_star[i]]
    for i in torch.where(alive_mask)[0]:
        if b_star[i] >= 0:
            lambda_hat[i] = bin_centers[b_star[i]]

    # H_bin[i] = -sum_b mu_i(B_b) * log(mu_i(B_b) + eps)
    eps = 1e-12
    mu_safe = mu + eps
    H_bin_all = -(mu * torch.log(mu_safe)).sum(dim=0)
    H_bin[alive_mask] = H_bin_all[alive_mask]

    # n_eff[i] = exp(H_bin[i]) (effective number of bins)
    n_eff[alive_mask] = torch.exp(H_bin[alive_mask])

    # Alternative: n_eff_ipr = 1 / sum_b mu^2 (inverse participation ratio)
    # n_eff_ipr = 1.0 / ((mu ** 2).sum(dim=0) + eps)

    return dict(
        q_bin=q_bin,
        H_bin=H_bin,
        n_eff=n_eff,
        b_star=b_star,
        lambda_hat=lambda_hat,
        bin_edges=edges
    )


def process_single_file(
    fpath: Path,
    device: torch.device,
    compute_feature_metrics: bool = True,
    num_bins: int = DEFAULT_NUM_BINS
) -> Tuple[Dict, List[Dict]]:
    """
    Process a single experiment file on the specified GPU.

    Implements all analysis steps with fixed metrics:
    - Step 0: Rank and saturation metrics
    - Step 1: Per-feature localization diagnostics (leverage, kappa, resid, sigma, cv)
    - Step 2: Leverage-weighted aggregates
    - Step 3: Binned spectral measure (q_bin, H_bin, n_eff)
    - Sanity checks: defect_error, negD_frac, D_Dhat_corr
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
    tol = DEFAULT_EIGENVALUE_TOL_FACTOR * lam_max
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
            mean_cv=float('nan'),
            wmean_q_bin=float('nan'),
            wmean_H_bin=float('nan'),
            wmean_n_eff=float('nan'),
            defect_error=float('nan'),
            negD_frac=float('nan'),
            D_Dhat_corr=float('nan'),
            qual_rate=float('nan'),
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
    n_alive = alive_mask.sum().item()

    # Initialize metrics tensors
    ell = torch.zeros(n, dtype=torch.float64, device=device)  # leverage
    kappa = torch.zeros(n, dtype=torch.float64, device=device)  # Rayleigh quotient
    resid = torch.zeros(n, dtype=torch.float64, device=device)  # eigenvector residual
    sigma = torch.full((n,), float('nan'), dtype=torch.float64, device=device)  # relative slack
    cv = torch.zeros(n, dtype=torch.float64, device=device)  # coefficient of variation

    # Compute leverage: ell_i = sum_k (u_k^T w_i)^2 / lambda_k = sum_k Z2[k,i] / lam_pos[k]
    inv_lam = 1.0 / lam_pos  # (r,)
    ell[alive_mask] = (Z2[:, alive_mask] * inv_lam[:, None]).sum(dim=0)

    # Compute Rayleigh quotient: kappa_i = (w_i^T F w_i) / ||w_i||^2 = E_{mu_i}[lambda]
    kappa[alive_mask] = (Z2[:, alive_mask] * lam_pos[:, None]).sum(dim=0) / norm2[alive_mask]

    # Compute eigenvector residual: sqrt(Var_{mu_i}(lambda))
    E2 = torch.zeros(n, dtype=torch.float64, device=device)
    E2[alive_mask] = (Z2[:, alive_mask] * (lam_pos[:, None] ** 2)).sum(dim=0) / norm2[alive_mask]
    var = torch.clamp(E2 - kappa ** 2, min=0.0)
    resid = torch.sqrt(var)

    # Compute coefficient of variation: cv[i] = resid[i] / kappa[i]
    cv[alive_mask] = resid[alive_mask] / (kappa[alive_mask] + 1e-12)

    # Compute relative slack: sigma_i = 1 - D_i / ell_i
    ratio = torch.full((n,), float('nan'), dtype=torch.float64, device=device)
    valid_ell_mask = alive_mask & (ell > 0)
    ratio[valid_ell_mask] = D[valid_ell_mask] / ell[valid_ell_mask]
    sigma = 1.0 - ratio

    # ========== Compute D_hat from W and kappa ==========
    # D_hat[i] = ||w_i||^2 / kappa[i]
    D_hat = torch.zeros(n, dtype=torch.float64, device=device)
    D_hat[alive_mask] = norm2[alive_mask] / (kappa[alive_mask] + 1e-12)

    # ========== STEP 3: Binned Spectral Measure ==========

    binned_metrics = compute_binned_spectral_measure(
        lam_pos, Z2, norm2, alive_mask, num_bins=num_bins, device=device
    )

    q_bin = binned_metrics['q_bin']
    H_bin = binned_metrics['H_bin']
    n_eff = binned_metrics['n_eff']
    b_star = binned_metrics['b_star']
    lambda_hat = binned_metrics['lambda_hat']
    bin_edges = binned_metrics['bin_edges']

    # ========== STEP 2: Leverage-Weighted Aggregates ==========

    ell_alive = ell[alive_mask]
    ell_sum = ell_alive.sum().item()

    # Slack run: should equal 1 - rho_r
    slack_run = 1.0 - rho_r if np.isfinite(rho_r) else float('nan')

    # Weighted mean helper
    def wmean(x):
        if ell_sum == 0:
            return float('nan')
        return (ell_alive * x[alive_mask]).sum().item() / ell_sum

    # Mean residual (leverage-weighted)
    mean_resid = wmean(resid)

    # Mean sigma (leverage-weighted, treat NaN as 0)
    sigma_clean = torch.nan_to_num(sigma, nan=0.0)
    mean_sigma = wmean(sigma_clean)

    # Mean kappa, leverage, cv
    mean_kappa = wmean(kappa)
    mean_leverage = ell_alive.mean().item() if ell_alive.numel() > 0 else float('nan')
    mean_cv = wmean(cv)

    # Weighted means for binned metrics
    wmean_q_bin = wmean(q_bin)
    wmean_H_bin = wmean(H_bin)
    wmean_n_eff = wmean(n_eff)

    # Tail mass curves
    def tail_mass(tau):
        high_slack_mask = alive_mask & (sigma >= tau)
        if ell_sum == 0:
            return float('nan')
        return ell[high_slack_mask].sum().item() / r if r > 0 else float('nan')

    tail_001 = tail_mass(0.01)
    tail_005 = tail_mass(0.05)
    tail_010 = tail_mass(0.10)

    # ========== Defect Identity Check ==========
    # gap1 = m - sum(D)
    # gap2 = sum(ell - D) for alive features
    D_alive = D[alive_mask]
    gap1 = m - sumD
    gap2 = (ell_alive - D_alive).sum().item()
    defect_error = abs(gap1 - gap2)

    # ========== D vs D_hat Sanity Checks ==========
    # Fraction of negative D values
    D_alive_cpu = D[alive_mask].cpu().numpy()
    negD_frac = (D_alive_cpu < 0).sum() / max(1, n_alive)

    # Correlation between D and D_hat
    D_hat_alive = D_hat[alive_mask].cpu().numpy()
    if n_alive > 1:
        # Filter finite values
        valid_corr = np.isfinite(D_alive_cpu) & np.isfinite(D_hat_alive)
        if valid_corr.sum() > 1:
            D_Dhat_corr = np.corrcoef(D_alive_cpu[valid_corr], D_hat_alive[valid_corr])[0, 1]
        else:
            D_Dhat_corr = float('nan')
    else:
        D_Dhat_corr = float('nan')

    # ========== Qualification Rate for Assignment Test ==========
    # qual_rate = (sum of leverage for features with q_bin >= threshold) / total leverage
    qualified_mask = alive_mask & (q_bin >= LOCALIZATION_THRESHOLD)
    if ell_sum > 0:
        qual_rate = ell[qualified_mask].sum().item() / ell_sum
    else:
        qual_rate = float('nan')

    # ========== Run Metrics ==========
    run_metrics = dict(
        file=fpath.name, m=m, s=s, seed=meta["seed"], n=n,
        rank=r, rank_ratio=rank_ratio,
        sumD=sumD, rho_m=rho_m, rho_r=rho_r,
        slack_run=slack_run,
        mean_resid=mean_resid,
        mean_sigma=mean_sigma,
        mean_kappa=mean_kappa,
        mean_leverage=mean_leverage,
        mean_cv=mean_cv,
        wmean_q_bin=wmean_q_bin,
        wmean_H_bin=wmean_H_bin,
        wmean_n_eff=wmean_n_eff,
        defect_error=defect_error,
        negD_frac=negD_frac,
        D_Dhat_corr=D_Dhat_corr,
        qual_rate=qual_rate,
        tail_mass_001=tail_001,
        tail_mass_005=tail_005,
        tail_mass_010=tail_010,
    )

    feat_metrics_list = []

    if compute_feature_metrics:
        # Collect feature metrics (move to CPU for storage)
        D_cpu = D.cpu().numpy()
        D_hat_cpu = D_hat.cpu().numpy()
        ell_cpu = ell.cpu().numpy()
        ratio_cpu = ratio.cpu().numpy()
        sigma_cpu = sigma.cpu().numpy()
        kappa_cpu = kappa.cpu().numpy()
        resid_cpu = resid.cpu().numpy()
        cv_cpu = cv.cpu().numpy()
        norm2_cpu = norm2.cpu().numpy()
        q_bin_cpu = q_bin.cpu().numpy()
        H_bin_cpu = H_bin.cpu().numpy()
        n_eff_cpu = n_eff.cpu().numpy()
        b_star_cpu = b_star.cpu().numpy()
        lambda_hat_cpu = lambda_hat.cpu().numpy()
        alive_cpu = alive_mask.cpu().numpy()

        for i in range(n):
            if not alive_cpu[i]:
                continue

            # D_hat_pred from lambda_hat (for assignment test)
            D_hat_pred = norm2_cpu[i] / (lambda_hat_cpu[i] + 1e-12) if lambda_hat_cpu[i] > 0 else float('nan')

            feat_metrics_list.append(dict(
                file=fpath.name, m=m, s=s, seed=meta["seed"], i=i,
                D=float(D_cpu[i]),
                D_hat=float(D_hat_cpu[i]),
                ell=float(ell_cpu[i]),
                ratio=float(ratio_cpu[i]),
                sigma=float(sigma_cpu[i]),
                kappa=float(kappa_cpu[i]),
                resid=float(resid_cpu[i]),
                cv=float(cv_cpu[i]),
                norm2=float(norm2_cpu[i]),
                q_bin=float(q_bin_cpu[i]),
                H_bin=float(H_bin_cpu[i]),
                n_eff=float(n_eff_cpu[i]),
                b_star=int(b_star_cpu[i]),
                lambda_hat=float(lambda_hat_cpu[i]),
                D_hat_pred=float(D_hat_pred),
            ))

    return run_metrics, feat_metrics_list


def run_analysis_multiprocess(
    files: List[Path],
    num_gpus: int,
    compute_feature_metrics: bool = True,
    num_bins: int = DEFAULT_NUM_BINS,
    show_progress: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run analysis across multiple GPUs using round-robin scheduling.
    """
    run_metrics_list = []
    feat_metrics_list = []

    pbar = tqdm(files, desc="Processing experiments", disable=not show_progress)

    for idx, fpath in enumerate(pbar):
        gpu_id = idx % num_gpus
        device = torch.device(f'cuda:{gpu_id}')

        run_metrics, feat_metrics = process_single_file(
            fpath, device, compute_feature_metrics, num_bins=num_bins
        )

        if run_metrics is not None:
            run_metrics_list.append(run_metrics)
            feat_metrics_list.extend(feat_metrics)

        # Update progress bar description
        if run_metrics:
            pbar.set_postfix(m=run_metrics['m'], s=f"{run_metrics['s']:.2f}", gpu=gpu_id)

    return run_metrics_list, feat_metrics_list


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated Capacity Localization Analysis (v2 - fixed metrics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python capacity_localization_analysis_v2.py --data-dir ./start --out ./results
    python capacity_localization_analysis_v2.py --data-dir ./start --out ./results --save-feature-metrics --num-gpus 8
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
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS,
                        help=f"Number of log-spaced bins for spectral measure (default: {DEFAULT_NUM_BINS})")
    parser.add_argument("--pattern", type=str, default="n1024_m*_s*_seed*.h5",
                        help="Glob pattern for finding experiment files")

    args = parser.parse_args()

    # Detect available GPUs
    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")

    print(f"=" * 70)
    print("Capacity Localization Analysis v2 (Fixed Metrics)")
    print(f"=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.out}")
    print(f"Using {num_gpus} GPU(s)")
    print(f"Number of spectral bins: {args.num_bins}")
    print(f"Save feature metrics: {args.save_feature_metrics}")
    print()
    print("Fixes in v2:")
    print("  - Binned spectral measure for basis-invariant Diracness proxy (q_bin)")
    print("  - Coefficient of variation (cv) for normalized spectral spread")
    print("  - Defect identity verification (defect_error)")
    print("  - D_hat reconstruction and sanity checks (negD_frac, D_Dhat_corr)")
    print("  - Assignment test metrics (b_star, lambda_hat, qual_rate)")
    print()

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
        files, num_gpus, args.save_feature_metrics,
        num_bins=args.num_bins, show_progress=True
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
    print(f"\n{'=' * 70}")
    print("Summary Statistics")
    print(f"{'=' * 70}")

    print("\nRun-level metrics (v2 additions in bold):")
    metrics_to_print = [
        'rank_ratio', 'rho_m', 'rho_r', 'slack_run',
        'mean_resid', 'mean_sigma', 'mean_cv',
        'wmean_q_bin', 'wmean_H_bin', 'wmean_n_eff',
        'defect_error', 'negD_frac', 'D_Dhat_corr', 'qual_rate'
    ]
    for col in metrics_to_print:
        if col in runs_df.columns:
            vals = runs_df[col].dropna()
            if len(vals) > 0:
                print(f"  {col:18s}: mean={vals.mean():8.4f}, std={vals.std():8.4f}, "
                      f"min={vals.min():8.4f}, max={vals.max():8.4f}")

    # Group by sparsity bins for quick overview
    print("\nBy sparsity range:")
    runs_df['s_bin'] = pd.cut(
        runs_df['s'],
        bins=[0, 0.3, 0.6, 0.9, 1.0],
        labels=['low (0-0.3)', 'mid (0.3-0.6)', 'high (0.6-0.9)', 'very high (0.9-1.0)']
    )
    summary_cols = ['rho_m', 'rho_r', 'rank_ratio', 'mean_sigma', 'mean_cv', 'wmean_q_bin', 'qual_rate']
    summary_cols = [c for c in summary_cols if c in runs_df.columns]
    summary = runs_df.groupby('s_bin', observed=True)[summary_cols].mean()
    print(summary.to_string())

    # Validation checks
    print(f"\n{'=' * 70}")
    print("Validation Checks")
    print(f"{'=' * 70}")

    # Check defect identity
    if 'defect_error' in runs_df.columns:
        max_defect_error = runs_df['defect_error'].max()
        print(f"\nDefect identity (gap1 = m - sumD vs gap2 = sum(ell-D)):")
        print(f"  Max defect error: {max_defect_error:.6e}")
        if max_defect_error < 1e-3:
            print(f"  PASS: Defect identity holds within tolerance")
        else:
            print(f"  WARNING: Defect identity error larger than expected")

    # Check negative D fraction
    if 'negD_frac' in runs_df.columns:
        max_negD_frac = runs_df['negD_frac'].max()
        mean_negD_frac = runs_df['negD_frac'].mean()
        print(f"\nNegative D values (impossible for fractional dimension):")
        print(f"  Mean negD_frac: {mean_negD_frac:.4f}")
        print(f"  Max negD_frac:  {max_negD_frac:.4f}")
        if max_negD_frac > 0:
            print(f"  WARNING: Some runs have negative D values. Use D_hat in plots.")
        else:
            print(f"  PASS: No negative D values")

    # Check q_bin separation by sparsity
    if 'wmean_q_bin' in runs_df.columns:
        print(f"\nq_bin (binned Diracness proxy) by sparsity:")
        q_by_s = runs_df.groupby('s_bin', observed=True)['wmean_q_bin'].mean()
        print(q_by_s.to_string())
        # Check if q_bin decreases with sparsity
        q_vals = q_by_s.values
        if len(q_vals) > 1 and q_vals[0] > q_vals[-1]:
            print(f"  PASS: q_bin decreases with sparsity (delocalization signature)")
        else:
            print(f"  Note: q_bin trend with sparsity may need investigation")

    # Check qual_rate
    if 'qual_rate' in runs_df.columns:
        print(f"\nQualification rate (fraction of leverage in q_bin >= {LOCALIZATION_THRESHOLD}):")
        qr_by_s = runs_df.groupby('s_bin', observed=True)['qual_rate'].mean()
        print(qr_by_s.to_string())

    # Save metadata
    metadata = {
        'analysis_date': datetime.now().isoformat(),
        'data_directory': str(args.data_dir.absolute()),
        'output_directory': str(args.out.absolute()),
        'num_files_processed': len(runs_df),
        'num_gpus_used': num_gpus,
        'num_bins': args.num_bins,
        'localization_threshold': LOCALIZATION_THRESHOLD,
        'save_feature_metrics': args.save_feature_metrics,
        'file_pattern': args.pattern,
        'version': 'v2',
    }

    with open(args.out / "analysis_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAnalysis complete. End time: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
