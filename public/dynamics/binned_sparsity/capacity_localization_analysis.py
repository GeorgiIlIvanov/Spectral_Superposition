#!/usr/bin/env python3
"""
GPU-accelerated Capacity Localization Analysis for Binned Sparsity Experiments.

Adapted for experiments with per-feature sparsity (sparsity_per_feature array).

Author: Claude Code Analysis Pipeline
Date: 2026-01-28
"""

import argparse
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


def parse_filename(path: Path) -> Dict:
    """Parse experiment filename to extract parameters."""
    # Format: n1024_m256_binned_seed385.h5
    parts = path.stem.split('_')
    return dict(
        n=int(parts[0][1:]),
        m=int(parts[1][1:]),
        exp_type=parts[2],
        seed=int(parts[3][4:]),
    )


def group_eigenvalues(lam: torch.Tensor, rtol: float = 1e-6, atol: float = 1e-10) -> List[torch.Tensor]:
    """Group eigenvalues by near-degeneracy."""
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
    """Process a single experiment file on the specified GPU."""
    meta = parse_filename(fpath)

    try:
        with h5py.File(fpath, "r") as hf:
            W = torch.tensor(hf["weights"][-1], dtype=torch.float64, device=device)
            D = torch.tensor(hf["fractional_dims"][-1], dtype=torch.float64, device=device)
            sparsity_per_feature = torch.tensor(hf["sparsity_per_feature"][:], dtype=torch.float64, device=device)

            m = int(hf.attrs["m_hidden"])
            n = int(hf.attrs["n_features"])
            exp_type = str(hf.attrs.get("sparsity_type", "unknown"))
    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        return None, []

    # Frame operator F = W @ W^T
    F = W @ W.T

    # Eigendecomposition
    lam, U = torch.linalg.eigh(F)
    lam = torch.flip(lam, [0])
    U = torch.flip(U, [1])

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

    # Sparsity statistics
    mean_sparsity = sparsity_per_feature.mean().item()
    min_sparsity = sparsity_per_feature.min().item()
    max_sparsity = sparsity_per_feature.max().item()

    if r == 0:
        run_metrics = dict(
            file=fpath.name, m=m, seed=meta["seed"], n=n, exp_type=exp_type,
            mean_sparsity=mean_sparsity, min_sparsity=min_sparsity, max_sparsity=max_sparsity,
            rank=r, rank_ratio=rank_ratio,
            sumD=sumD, rho_m=rho_m, rho_r=rho_r,
            slack_run=float('nan'), mean_resid=float('nan'), mean_sigma=float('nan'),
            mean_kappa=float('nan'), mean_leverage=float('nan'),
            tail_mass_001=float('nan'), tail_mass_005=float('nan'), tail_mass_010=float('nan'),
        )
        return run_metrics, []

    # Project features into positive eigenspace
    Z = U_pos.T @ W
    Z2 = Z * Z
    norm2 = Z2.sum(dim=0)

    alive_mask = norm2 > 1e-14

    # Initialize metrics
    ell = torch.zeros(n, dtype=torch.float64, device=device)
    kappa = torch.zeros(n, dtype=torch.float64, device=device)
    resid = torch.zeros(n, dtype=torch.float64, device=device)
    sigma = torch.full((n,), float('nan'), dtype=torch.float64, device=device)

    inv_lam = 1.0 / lam_pos
    ell[alive_mask] = (Z2[:, alive_mask] * inv_lam[:, None]).sum(dim=0)
    kappa[alive_mask] = (Z2[:, alive_mask] * lam_pos[:, None]).sum(dim=0) / norm2[alive_mask]

    E2 = torch.zeros(n, dtype=torch.float64, device=device)
    E2[alive_mask] = (Z2[:, alive_mask] * (lam_pos[:, None] ** 2)).sum(dim=0) / norm2[alive_mask]
    var = torch.clamp(E2 - kappa ** 2, min=0.0)
    resid = torch.sqrt(var)

    ratio = torch.full((n,), float('nan'), dtype=torch.float64, device=device)
    valid_ell_mask = alive_mask & (ell > 0)
    ratio[valid_ell_mask] = D[valid_ell_mask] / ell[valid_ell_mask]
    sigma = 1.0 - ratio

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

    def tail_mass(tau):
        high_slack_mask = alive_mask & (sigma >= tau)
        if ell_sum == 0:
            return float('nan')
        return ell[high_slack_mask].sum().item() / r if r > 0 else float('nan')

    run_metrics = dict(
        file=fpath.name, m=m, seed=meta["seed"], n=n, exp_type=exp_type,
        mean_sparsity=mean_sparsity, min_sparsity=min_sparsity, max_sparsity=max_sparsity,
        rank=r, rank_ratio=rank_ratio,
        sumD=sumD, rho_m=rho_m, rho_r=rho_r,
        slack_run=slack_run, mean_resid=mean_resid, mean_sigma=mean_sigma,
        mean_kappa=mean_kappa, mean_leverage=mean_leverage,
        tail_mass_001=tail_mass(0.01), tail_mass_005=tail_mass(0.05), tail_mass_010=tail_mass(0.10),
    )

    feat_metrics_list = []

    if compute_feature_metrics:
        groups = group_eigenvalues(lam_pos, rtol=1e-6, atol=tol)
        q = torch.zeros(n, dtype=torch.float64, device=device)

        if r > 0 and len(groups) > 0:
            pmax = torch.zeros(n, dtype=torch.float64, device=device)
            for g in groups:
                pg = torch.zeros(n, dtype=torch.float64, device=device)
                pg[alive_mask] = Z2[g][:, alive_mask].sum(dim=0) / norm2[alive_mask]
                pmax = torch.maximum(pmax, pg)
            q = pmax

        # Move to CPU
        D_cpu = D.cpu().numpy()
        ell_cpu = ell.cpu().numpy()
        ratio_cpu = ratio.cpu().numpy()
        sigma_cpu = sigma.cpu().numpy()
        kappa_cpu = kappa.cpu().numpy()
        resid_cpu = resid.cpu().numpy()
        q_cpu = q.cpu().numpy()
        alive_cpu = alive_mask.cpu().numpy()
        sparsity_cpu = sparsity_per_feature.cpu().numpy()

        for i in range(n):
            if not alive_cpu[i]:
                continue
            feat_metrics_list.append(dict(
                file=fpath.name, m=m, seed=meta["seed"], i=i,
                feature_sparsity=float(sparsity_cpu[i]),
                D=float(D_cpu[i]), ell=float(ell_cpu[i]), ratio=float(ratio_cpu[i]),
                sigma=float(sigma_cpu[i]), kappa=float(kappa_cpu[i]), resid=float(resid_cpu[i]),
                q=float(q_cpu[i]),
            ))

    return run_metrics, feat_metrics_list


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated Capacity Localization Analysis (Binned Sparsity)")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--save-feature-metrics", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--pattern", type=str, default="n1024_m*_*_seed*.h5")

    args = parser.parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")

    print(f"=" * 60)
    print("Capacity Localization Analysis (Per-Feature Sparsity)")
    print(f"=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.out}")
    print(f"Using {num_gpus} GPU(s)")

    files = sorted(args.data_dir.glob(args.pattern))
    if args.max_files:
        files = files[:args.max_files]

    print(f"Found {len(files)} experiment files")

    if len(files) == 0:
        print("No files found.")
        return

    run_metrics_list = []
    feat_metrics_list = []

    pbar = tqdm(files, desc="Processing experiments")
    for idx, fpath in enumerate(pbar):
        gpu_id = idx % num_gpus
        device = torch.device(f'cuda:{gpu_id}')

        run_metrics, feat_metrics = process_single_file(fpath, device, args.save_feature_metrics)

        if run_metrics is not None:
            run_metrics_list.append(run_metrics)
            feat_metrics_list.extend(feat_metrics)
            pbar.set_postfix(seed=run_metrics['seed'], gpu=gpu_id)

    args.out.mkdir(parents=True, exist_ok=True)

    runs_df = pd.DataFrame(run_metrics_list)
    runs_df.to_csv(args.out / "run_metrics.csv", index=False)
    print(f"\nSaved run metrics: {len(runs_df)} rows")

    if args.save_feature_metrics and feat_metrics_list:
        feats_df = pd.DataFrame(feat_metrics_list)
        feats_df.to_csv(args.out / "feature_metrics.csv", index=False)
        print(f"Saved feature metrics: {len(feats_df)} rows")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary Statistics")
    print(f"{'=' * 60}")
    for col in ['rank_ratio', 'rho_m', 'rho_r', 'mean_resid', 'mean_sigma']:
        if col in runs_df.columns:
            vals = runs_df[col].dropna()
            print(f"  {col:15s}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    metadata = {
        'analysis_date': datetime.now().isoformat(),
        'data_directory': str(args.data_dir.absolute()),
        'num_files_processed': len(runs_df),
        'num_gpus_used': num_gpus,
        'experiment_type': 'per_feature_sparsity',
    }
    with open(args.out / "analysis_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAnalysis complete. End time: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
