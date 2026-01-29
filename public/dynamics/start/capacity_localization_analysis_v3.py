#!/usr/bin/env python3
"""
GPU-accelerated Capacity Localization Analysis (v3 - EIGENSPACE-BASED).

CRITICAL CORRECTION from v2:
v2 projected onto individual eigenvectors, which is incorrect for degenerate
eigenvalues. v3 correctly projects onto eigenSPACES (subspaces spanned by
all eigenvectors with the same eigenvalue).

Key changes:
- identify_eigenspaces(): Groups eigenvectors by eigenvalue
- Projections computed per eigenspace, not per eigenvector
- All localization metrics (q_bin, H_bin, n_eff) computed over eigenspaces

Author: Claude Code Analysis Pipeline
Date: 2026-01-29
"""

import argparse
import math
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import h5py
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuration
DEFAULT_NUM_BINS = 60
DEFAULT_EIGENVALUE_TOL = 1e-4  # Relative tolerance for grouping degenerate eigenvalues
LOCALIZATION_THRESHOLD = 0.9


def parse_filename(path: Path) -> Dict:
    """Parse experiment filename to extract parameters."""
    parts = path.stem.split('_')
    return dict(
        n=int(parts[0][1:]),
        m=int(parts[1][1:]),
        s=float(parts[2][1:]),
        seed=int(parts[3][4:]),
    )


def identify_eigenspaces(
    eigenvalues: torch.Tensor,
    rel_tol: float = DEFAULT_EIGENVALUE_TOL
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Identify eigenspaces by grouping eigenvalues that are equal (within tolerance).

    For degenerate eigenvalues (λ_k = λ_{k+1} = ... = λ_{k+d-1}),
    all corresponding eigenvectors span a single d-dimensional eigenspace.

    Args:
        eigenvalues: (m,) tensor of eigenvalues in descending order
        rel_tol: relative tolerance for considering eigenvalues equal

    Returns:
        space_assignments: (m,) tensor mapping each eigenvector to its eigenspace index
        space_eigenvalues: (n_spaces,) tensor of eigenvalue for each space
        n_spaces: number of distinct eigenspaces
    """
    m = eigenvalues.numel()
    device = eigenvalues.device

    if m == 0:
        return torch.tensor([], dtype=torch.long, device=device), \
               torch.tensor([], device=device), 0

    # Use relative tolerance based on largest eigenvalue
    max_eig = eigenvalues[0].abs().item() if eigenvalues[0] != 0 else 1.0
    abs_tol = rel_tol * max_eig

    # Group consecutive eigenvalues that are within tolerance
    space_assignments = torch.zeros(m, dtype=torch.long, device=device)
    space_eigenvalues_list = []

    current_space = 0
    current_eig = eigenvalues[0].item()
    space_start = 0

    for k in range(m):
        if abs(eigenvalues[k].item() - current_eig) > abs_tol:
            # New eigenspace starts - record mean eigenvalue for previous space
            space_eigenvalues_list.append(eigenvalues[space_start:k].mean().item())
            current_space += 1
            current_eig = eigenvalues[k].item()
            space_start = k
        space_assignments[k] = current_space

    # Don't forget the last space
    space_eigenvalues_list.append(eigenvalues[space_start:].mean().item())

    space_eigenvalues = torch.tensor(space_eigenvalues_list, dtype=torch.float64, device=device)
    n_spaces = len(space_eigenvalues_list)

    return space_assignments, space_eigenvalues, n_spaces


def compute_eigenspace_projections(
    Z2: torch.Tensor,  # (r, n) - squared projections onto eigenvectors
    norm2: torch.Tensor,  # (n,) - feature norms squared
    space_assignments: torch.Tensor,  # (r,) - eigenspace for each eigenvector
    n_spaces: int,
    alive_mask: torch.Tensor,  # (n,) - which features are alive
    device: torch.device
) -> torch.Tensor:
    """
    Compute projection weights onto eigenSPACES (not eigenvectors).

    For each eigenspace s:
        p_{i,s} = Σ_{k ∈ space s} |u_k^T w_i|² / ||w_i||²

    This is the correct basis-invariant measure.

    Returns:
        eigenspace_proj: (n_spaces, n) - projection weight per feature per eigenspace
    """
    r, n = Z2.shape

    eigenspace_proj = torch.zeros(n_spaces, n, dtype=torch.float64, device=device)

    for s in range(n_spaces):
        mask_s = space_assignments == s
        if mask_s.any():
            # Sum squared projections onto all eigenvectors in this eigenspace
            eigenspace_proj[s, alive_mask] = Z2[mask_s][:, alive_mask].sum(dim=0) / norm2[alive_mask]

    return eigenspace_proj


def compute_binned_spectral_measure_eigenspace(
    space_eigenvalues: torch.Tensor,  # (n_spaces,) - eigenvalue per space
    eigenspace_proj: torch.Tensor,  # (n_spaces, n) - projection per feature per space
    alive_mask: torch.Tensor,
    num_bins: int = DEFAULT_NUM_BINS,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Compute binned spectral measure over eigenSPACES (not eigenvectors).

    Returns:
        Dict with tensors for q_bin, H_bin, n_eff, b_star, lambda_hat
    """
    n_spaces = space_eigenvalues.numel()
    n = eigenspace_proj.shape[1]

    if device is None:
        device = space_eigenvalues.device

    # Initialize outputs
    q_bin = torch.zeros(n, dtype=torch.float64, device=device)
    H_bin = torch.zeros(n, dtype=torch.float64, device=device)
    n_eff = torch.zeros(n, dtype=torch.float64, device=device)
    b_star = torch.full((n,), -1, dtype=torch.int64, device=device)
    lambda_hat = torch.zeros(n, dtype=torch.float64, device=device)

    if n_spaces == 0 or not alive_mask.any():
        return dict(q_bin=q_bin, H_bin=H_bin, n_eff=n_eff,
                    b_star=b_star, lambda_hat=lambda_hat)

    # Filter positive eigenvalues
    pos_mask = space_eigenvalues > 1e-10
    if not pos_mask.any():
        q_bin[alive_mask] = 1.0
        return dict(q_bin=q_bin, H_bin=H_bin, n_eff=n_eff,
                    b_star=b_star, lambda_hat=lambda_hat)

    lam_pos = space_eigenvalues[pos_mask]
    proj_pos = eigenspace_proj[pos_mask]  # (n_pos_spaces, n)

    lam_min = lam_pos.min().item()
    lam_max = lam_pos.max().item()

    if lam_min <= 0 or lam_max <= 0 or lam_min >= lam_max:
        q_bin[alive_mask] = 1.0
        return dict(q_bin=q_bin, H_bin=H_bin, n_eff=n_eff,
                    b_star=b_star, lambda_hat=lambda_hat)

    # Create log-spaced bin edges
    log_min = math.log10(lam_min)
    log_max = math.log10(lam_max)
    edges_np = np.logspace(log_min, log_max, num_bins + 1)
    edges = torch.tensor(edges_np, dtype=torch.float64, device=device)
    bin_centers = torch.sqrt(edges[:-1] * edges[1:])

    # Assign each eigenspace to a bin
    lam_pos_np = lam_pos.cpu().numpy()
    bin_indices_np = np.searchsorted(edges_np, lam_pos_np, side='right') - 1
    bin_indices_np = np.clip(bin_indices_np, 0, num_bins - 1)
    bin_indices = torch.tensor(bin_indices_np, dtype=torch.int64, device=device)

    # Aggregate eigenspace projections into bins
    mu = torch.zeros(num_bins, n, dtype=torch.float64, device=device)
    for b in range(num_bins):
        mask_b = bin_indices == b
        if mask_b.any():
            mu[b] = proj_pos[mask_b].sum(dim=0)

    # q_bin = max bin mass
    q_bin_vals, b_star_temp = mu.max(dim=0)
    q_bin[alive_mask] = q_bin_vals[alive_mask]
    b_star[alive_mask] = b_star_temp[alive_mask]

    # lambda_hat from b_star
    for i in torch.where(alive_mask)[0]:
        if b_star[i] >= 0 and b_star[i] < num_bins:
            lambda_hat[i] = bin_centers[b_star[i]]

    # H_bin = entropy
    eps = 1e-12
    mu_safe = mu + eps
    H_bin_all = -(mu * torch.log(mu_safe)).sum(dim=0)
    H_bin[alive_mask] = H_bin_all[alive_mask]

    # n_eff = exp(H)
    n_eff[alive_mask] = torch.exp(H_bin[alive_mask])

    return dict(q_bin=q_bin, H_bin=H_bin, n_eff=n_eff,
                b_star=b_star, lambda_hat=lambda_hat)


def process_single_file(
    fpath: Path,
    device: torch.device,
    compute_feature_metrics: bool = True,
    num_bins: int = DEFAULT_NUM_BINS,
    eigenvalue_tol: float = DEFAULT_EIGENVALUE_TOL
) -> Tuple[Dict, List[Dict]]:
    """
    Process a single experiment file with EIGENSPACE-BASED projections.
    """
    meta = parse_filename(fpath)

    try:
        with h5py.File(fpath, "r") as hf:
            W = torch.tensor(hf["weights"][-1], dtype=torch.float64, device=device)
            D = torch.tensor(hf["fractional_dims"][-1], dtype=torch.float64, device=device)
            m = int(hf.attrs["m_hidden"])
            s = float(hf.attrs["sparsity"])
            n = int(hf.attrs["n_features"])
    except Exception as e:
        return None, []

    # ========== Eigendecomposition ==========
    F = W @ W.T
    lam, U = torch.linalg.eigh(F)
    lam = torch.flip(lam, [0])
    U = torch.flip(U, [1])

    # Effective rank
    lam_max = lam[0].item() if lam.numel() > 0 else 0.0
    tol = 1e-8 * lam_max
    pos_mask = lam > tol
    lam_pos = lam[pos_mask]
    U_pos = U[:, pos_mask]
    r = int(lam_pos.numel())

    sumD = D.sum().item()
    rho_m = sumD / m if m > 0 else float('nan')
    rho_r = sumD / r if r > 0 else float('nan')
    rank_ratio = r / m if m > 0 else float('nan')

    if r == 0:
        run_metrics = dict(
            file=fpath.name, m=m, s=s, seed=meta["seed"], n=n,
            rank=r, rank_ratio=rank_ratio, sumD=sumD, rho_m=rho_m, rho_r=rho_r,
            n_eigenspaces=0, degeneracy_ratio=float('nan'),
            slack_run=float('nan'), mean_resid=float('nan'), mean_sigma=float('nan'),
            mean_kappa=float('nan'), mean_leverage=float('nan'), mean_cv=float('nan'),
            wmean_q_bin=float('nan'), wmean_H_bin=float('nan'), wmean_n_eff=float('nan'),
            defect_error=float('nan'), negD_frac=float('nan'), D_Dhat_corr=float('nan'),
            qual_rate=float('nan'),
        )
        return run_metrics, []

    # ========== Identify Eigenspaces ==========
    space_assignments, space_eigenvalues, n_spaces = identify_eigenspaces(
        lam_pos, rel_tol=eigenvalue_tol
    )
    degeneracy_ratio = n_spaces / r if r > 0 else 1.0

    # ========== Project onto eigenvectors ==========
    Z = U_pos.T @ W  # (r, n)
    Z2 = Z * Z
    norm2 = Z2.sum(dim=0)  # (n,)

    alive_mask = norm2 > 1e-14
    n_alive = alive_mask.sum().item()

    # ========== Compute eigenspace projections ==========
    eigenspace_proj = compute_eigenspace_projections(
        Z2, norm2, space_assignments, n_spaces, alive_mask, device
    )

    # ========== Per-feature metrics (still use eigenvector-level for kappa, resid) ==========
    ell = torch.zeros(n, dtype=torch.float64, device=device)
    kappa = torch.zeros(n, dtype=torch.float64, device=device)
    resid = torch.zeros(n, dtype=torch.float64, device=device)
    sigma = torch.full((n,), float('nan'), dtype=torch.float64, device=device)
    cv = torch.zeros(n, dtype=torch.float64, device=device)

    inv_lam = 1.0 / lam_pos
    ell[alive_mask] = (Z2[:, alive_mask] * inv_lam[:, None]).sum(dim=0)
    kappa[alive_mask] = (Z2[:, alive_mask] * lam_pos[:, None]).sum(dim=0) / norm2[alive_mask]

    E2 = torch.zeros(n, dtype=torch.float64, device=device)
    E2[alive_mask] = (Z2[:, alive_mask] * (lam_pos[:, None] ** 2)).sum(dim=0) / norm2[alive_mask]
    var = torch.clamp(E2 - kappa ** 2, min=0.0)
    resid = torch.sqrt(var)
    cv[alive_mask] = resid[alive_mask] / (kappa[alive_mask] + 1e-12)

    ratio = torch.full((n,), float('nan'), dtype=torch.float64, device=device)
    valid_ell_mask = alive_mask & (ell > 0)
    ratio[valid_ell_mask] = D[valid_ell_mask] / ell[valid_ell_mask]
    sigma = 1.0 - ratio

    D_hat = torch.zeros(n, dtype=torch.float64, device=device)
    D_hat[alive_mask] = norm2[alive_mask] / (kappa[alive_mask] + 1e-12)

    # ========== Binned Spectral Measure (EIGENSPACE-BASED) ==========
    binned = compute_binned_spectral_measure_eigenspace(
        space_eigenvalues, eigenspace_proj, alive_mask, num_bins=num_bins, device=device
    )
    q_bin = binned['q_bin']
    H_bin = binned['H_bin']
    n_eff = binned['n_eff']
    b_star = binned['b_star']
    lambda_hat = binned['lambda_hat']

    # ========== Aggregates ==========
    ell_alive = ell[alive_mask]
    ell_sum = ell_alive.sum().item()
    slack_run = 1.0 - rho_r if np.isfinite(rho_r) else float('nan')

    def wmean(x):
        if ell_sum == 0:
            return float('nan')
        return (ell_alive * x[alive_mask]).sum().item() / ell_sum

    mean_resid = wmean(resid)
    sigma_clean = torch.nan_to_num(sigma, nan=0.0)
    mean_sigma = wmean(sigma_clean)
    mean_kappa = wmean(kappa)
    mean_leverage = ell_alive.mean().item() if ell_alive.numel() > 0 else float('nan')
    mean_cv = wmean(cv)
    wmean_q_bin = wmean(q_bin)
    wmean_H_bin = wmean(H_bin)
    wmean_n_eff = wmean(n_eff)

    # Defect identity check
    D_alive = D[alive_mask]
    gap1 = m - sumD
    gap2 = (ell_alive - D_alive).sum().item()
    defect_error = abs(gap1 - gap2)

    # D sanity checks
    D_alive_cpu = D[alive_mask].cpu().numpy()
    negD_frac = (D_alive_cpu < 0).sum() / max(1, n_alive)
    D_hat_alive = D_hat[alive_mask].cpu().numpy()
    if n_alive > 1:
        valid_corr = np.isfinite(D_alive_cpu) & np.isfinite(D_hat_alive)
        if valid_corr.sum() > 1:
            D_Dhat_corr = np.corrcoef(D_alive_cpu[valid_corr], D_hat_alive[valid_corr])[0, 1]
        else:
            D_Dhat_corr = float('nan')
    else:
        D_Dhat_corr = float('nan')

    # Qualification rate
    qualified_mask = alive_mask & (q_bin >= LOCALIZATION_THRESHOLD)
    qual_rate = ell[qualified_mask].sum().item() / ell_sum if ell_sum > 0 else float('nan')

    # ========== Eigenspace-specific metrics ==========
    # Max eigenspace projection per feature
    max_space_proj, dominant_space = eigenspace_proj.max(dim=0)
    participation_ratio = torch.zeros(n, dtype=torch.float64, device=device)
    proj_sq_sum = (eigenspace_proj ** 2).sum(dim=0)
    participation_ratio[alive_mask] = 1.0 / (proj_sq_sum[alive_mask] + 1e-12)

    wmean_max_space_proj = wmean(max_space_proj)
    wmean_participation_ratio = wmean(participation_ratio)

    run_metrics = dict(
        file=fpath.name, m=m, s=s, seed=meta["seed"], n=n,
        rank=r, rank_ratio=rank_ratio, sumD=sumD, rho_m=rho_m, rho_r=rho_r,
        n_eigenspaces=n_spaces, degeneracy_ratio=degeneracy_ratio,
        slack_run=slack_run, mean_resid=mean_resid, mean_sigma=mean_sigma,
        mean_kappa=mean_kappa, mean_leverage=mean_leverage, mean_cv=mean_cv,
        wmean_q_bin=wmean_q_bin, wmean_H_bin=wmean_H_bin, wmean_n_eff=wmean_n_eff,
        wmean_max_space_proj=wmean_max_space_proj,
        wmean_participation_ratio=wmean_participation_ratio,
        defect_error=defect_error, negD_frac=negD_frac, D_Dhat_corr=D_Dhat_corr,
        qual_rate=qual_rate,
    )

    feat_metrics_list = []
    if compute_feature_metrics:
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
        max_space_proj_cpu = max_space_proj.cpu().numpy()
        participation_ratio_cpu = participation_ratio.cpu().numpy()
        dominant_space_cpu = dominant_space.cpu().numpy()
        alive_cpu = alive_mask.cpu().numpy()

        for i in range(n):
            if not alive_cpu[i]:
                continue
            D_hat_pred = norm2_cpu[i] / (lambda_hat_cpu[i] + 1e-12) if lambda_hat_cpu[i] > 0 else float('nan')
            feat_metrics_list.append(dict(
                file=fpath.name, m=m, s=s, seed=meta["seed"], i=i,
                D=float(D_cpu[i]), D_hat=float(D_hat_cpu[i]),
                ell=float(ell_cpu[i]), ratio=float(ratio_cpu[i]),
                sigma=float(sigma_cpu[i]), kappa=float(kappa_cpu[i]),
                resid=float(resid_cpu[i]), cv=float(cv_cpu[i]),
                norm2=float(norm2_cpu[i]),
                q_bin=float(q_bin_cpu[i]), H_bin=float(H_bin_cpu[i]),
                n_eff=float(n_eff_cpu[i]), b_star=int(b_star_cpu[i]),
                lambda_hat=float(lambda_hat_cpu[i]), D_hat_pred=float(D_hat_pred),
                max_space_proj=float(max_space_proj_cpu[i]),
                participation_ratio=float(participation_ratio_cpu[i]),
                dominant_space=int(dominant_space_cpu[i]),
            ))

    return run_metrics, feat_metrics_list


def main():
    parser = argparse.ArgumentParser(
        description="Capacity Localization Analysis v3 (EIGENSPACE-BASED)"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--save-feature-metrics", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument("--eigenvalue-tol", type=float, default=DEFAULT_EIGENVALUE_TOL)
    parser.add_argument("--pattern", type=str, default="n1024_m*_s*_seed*.h5")

    args = parser.parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")

    print("=" * 70)
    print("Capacity Localization Analysis v3 (EIGENSPACE-BASED)")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Using {num_gpus} GPU(s)")
    print(f"Eigenvalue tolerance: {args.eigenvalue_tol}")
    print()
    print("CORRECTION: Projections now computed onto eigenSPACES, not eigenvectors")
    print()

    files = sorted(args.data_dir.glob(args.pattern))
    if args.max_files:
        files = files[:args.max_files]
    print(f"Found {len(files)} experiment files")

    run_metrics_list = []
    feat_metrics_list = []

    for idx, fpath in enumerate(tqdm(files, desc="Processing")):
        gpu_id = idx % num_gpus
        device = torch.device(f'cuda:{gpu_id}')
        run_metrics, feat_metrics = process_single_file(
            fpath, device, args.save_feature_metrics,
            args.num_bins, args.eigenvalue_tol
        )
        if run_metrics:
            run_metrics_list.append(run_metrics)
            feat_metrics_list.extend(feat_metrics)

    args.out.mkdir(parents=True, exist_ok=True)

    runs_df = pd.DataFrame(run_metrics_list)
    runs_df.to_csv(args.out / "run_metrics.csv", index=False)
    print(f"\nSaved run metrics: {len(runs_df)} runs")

    if args.save_feature_metrics and feat_metrics_list:
        feats_df = pd.DataFrame(feat_metrics_list)
        feats_df.to_csv(args.out / "feature_metrics.csv", index=False)
        print(f"Saved feature metrics: {len(feats_df)} features")

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary by sparsity:")
    runs_df['s_bin'] = pd.cut(runs_df['s'], bins=[0, 0.3, 0.6, 0.9, 1.0],
                               labels=['low', 'mid', 'high', 'very high'])
    summary_cols = ['rho_r', 'mean_sigma', 'mean_cv', 'wmean_q_bin',
                    'wmean_max_space_proj', 'wmean_participation_ratio']
    summary_cols = [c for c in summary_cols if c in runs_df.columns]
    print(runs_df.groupby('s_bin', observed=True)[summary_cols].mean().to_string())

    metadata = {
        'version': 'v3_eigenspace',
        'analysis_date': datetime.now().isoformat(),
        'num_files': len(runs_df),
        'eigenvalue_tol': args.eigenvalue_tol,
        'num_bins': args.num_bins,
    }
    with open(args.out / "analysis_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nComplete. End time: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
