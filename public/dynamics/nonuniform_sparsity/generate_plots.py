#!/usr/bin/env python3
"""
Visualization script for Per-Feature Sparsity Experiments.

Author: Claude Code Analysis Pipeline
Date: 2026-01-28
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def plot_a_rank_delocalization(runs_df: pd.DataFrame, out_dir: Path):
    """Plot A: Run-level metrics overview."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.hist(runs_df['rank_ratio'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Rank Ratio (r/m)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Rank Ratio')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)

    ax = axes[0, 1]
    ax.hist(runs_df['rho_r'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel(r'$\rho_r$ (saturation wrt rank)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Capacity Saturation')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)

    ax = axes[1, 0]
    ax.hist(runs_df['mean_sigma'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Mean Relative Slack (sigma)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Mean Delocalization')

    ax = axes[1, 1]
    ax.hist(runs_df['mean_resid'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Mean Eigenvector Residual')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Mean Spectral Spread')

    plt.suptitle('Plot A: Run-Level Metrics Overview', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'plot_A_run_metrics.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Plot A to {out_dir / 'plot_A_run_metrics.png'}")


def plot_b_featurewise_by_sparsity(feats_df: pd.DataFrame, out_dir: Path):
    """Plot B: Feature metrics by per-feature sparsity."""
    # Bin features by their individual sparsity
    feats_df['sparsity_bin'] = pd.cut(feats_df['feature_sparsity'],
                                       bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                       labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Sigma by sparsity bin
    ax = axes[0, 0]
    for label in ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']:
        subset = feats_df[feats_df['sparsity_bin'] == label]['sigma'].dropna()
        if len(subset) > 0:
            ax.hist(subset.clip(-0.1, 0.5), bins=50, alpha=0.5, label=f's={label}', density=True)
    ax.set_xlabel(r'Relative Slack $\sigma_i$')
    ax.set_ylabel('Density')
    ax.set_title('Delocalization Slack by Feature Sparsity')
    ax.legend()

    # Resid by sparsity bin
    ax = axes[0, 1]
    for label in ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']:
        subset = feats_df[feats_df['sparsity_bin'] == label]['resid'].dropna()
        if len(subset) > 0:
            ax.hist(subset, bins=50, alpha=0.5, label=f's={label}', density=True)
    ax.set_xlabel('Eigenvector Residual')
    ax.set_ylabel('Density')
    ax.set_title('Spectral Spread by Feature Sparsity')
    ax.legend()

    # D by sparsity bin
    ax = axes[1, 0]
    for label in ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']:
        subset = feats_df[feats_df['sparsity_bin'] == label]['D'].dropna()
        if len(subset) > 0:
            ax.hist(subset, bins=50, alpha=0.5, label=f's={label}', density=True)
    ax.set_xlabel('Fractional Dimension D')
    ax.set_ylabel('Density')
    ax.set_title('Fractional Dimension by Feature Sparsity')
    ax.legend()

    # Scatter: D vs feature_sparsity colored by sigma
    ax = axes[1, 1]
    sample = feats_df.sample(n=min(10000, len(feats_df)), random_state=42)
    scatter = ax.scatter(sample['feature_sparsity'], sample['D'],
                        c=sample['sigma'].clip(0, 0.3), cmap='RdYlBu_r',
                        alpha=0.5, s=5)
    ax.set_xlabel('Feature Sparsity')
    ax.set_ylabel('Fractional Dimension D')
    ax.set_title('D vs Feature Sparsity (colored by sigma)')
    plt.colorbar(scatter, ax=ax, label=r'$\sigma_i$')

    plt.suptitle('Plot B: Feature-Level Metrics by Per-Feature Sparsity', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'plot_B_featurewise_by_sparsity.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Plot B to {out_dir / 'plot_B_featurewise_by_sparsity.png'}")


def plot_c_eigenvalue_reciprocal(feats_df: pd.DataFrame, out_dir: Path):
    """Plot C: D vs 1/kappa relationship by sparsity."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    sparsity_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

    for idx, (s_low, s_high) in enumerate(sparsity_ranges):
        ax = axes[idx]
        mask = (feats_df['feature_sparsity'] >= s_low) & (feats_df['feature_sparsity'] < s_high)
        subset = feats_df[mask]

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f's in [{s_low}, {s_high})')
            continue

        # Sample for plotting
        sample = subset.sample(n=min(5000, len(subset)), random_state=42)

        kappa_vals = sample['kappa'].values
        D_vals = sample['D'].values
        sigma_vals = sample['sigma'].values

        valid = np.isfinite(kappa_vals) & (kappa_vals > 1e-10) & np.isfinite(D_vals) & np.isfinite(sigma_vals)
        if valid.sum() == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f's in [{s_low}, {s_high})')
            continue

        kappa_vals = kappa_vals[valid]
        D_vals = D_vals[valid]
        sigma_vals = sigma_vals[valid]
        inv_kappa = 1.0 / kappa_vals

        norm = Normalize(vmin=0, vmax=max(0.3, np.percentile(sigma_vals, 95)))
        scatter = ax.scatter(inv_kappa, D_vals, c=sigma_vals, cmap='RdYlBu_r',
                            norm=norm, alpha=0.6, s=8)
        ax.set_xlabel(r'$1/\kappa_i$')
        ax.set_ylabel(r'$D_i$')
        ax.set_title(f's in [{s_low}, {s_high})')
        plt.colorbar(scatter, ax=ax, label=r'$\sigma_i$')

    # Use last subplot for legend/summary
    ax = axes[5]
    ax.axis('off')
    summary_text = "Interpretation:\n\n"
    summary_text += "- Blue points: well-localized (low sigma)\n"
    summary_text += "- Red points: delocalized (high sigma)\n\n"
    summary_text += "Linear D ~ 1/kappa relationship\n"
    summary_text += "should hold for blue points.\n\n"
    summary_text += "Red points breaking the linear\n"
    summary_text += "fit indicate spectral delocalization."
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Plot C: Eigenvalue-Reciprocal Relationship by Feature Sparsity', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'plot_C_eigenvalue_reciprocal.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Plot C to {out_dir / 'plot_C_eigenvalue_reciprocal.png'}")


def plot_diracness_by_sparsity(feats_df: pd.DataFrame, out_dir: Path):
    """Diracness proxy by feature sparsity."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    sparsity_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

    for idx, (s_low, s_high) in enumerate(sparsity_ranges):
        ax = axes[idx]
        mask = (feats_df['feature_sparsity'] >= s_low) & (feats_df['feature_sparsity'] < s_high)
        subset = feats_df[mask]

        q_vals = subset['q'].dropna().values
        if len(q_vals) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f's in [{s_low}, {s_high})')
            continue

        ax.hist(q_vals, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel(r'Diracness Proxy $q_i$')
        ax.set_ylabel('Density')
        ax.set_title(f's in [{s_low}, {s_high}), mean q={q_vals.mean():.3f}')

    ax = axes[5]
    ax.axis('off')
    ax.text(0.1, 0.5, "q_i = max_group p_{i,group}\n\nq ~ 1: quasi-Dirac (localized)\nq < 1: delocalized",
            transform=ax.transAxes, fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Diracness Proxy Distribution by Feature Sparsity', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'plot_diracness_by_sparsity.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved diracness plot to {out_dir / 'plot_diracness_by_sparsity.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.results_dir}")

    runs_df = pd.read_csv(args.results_dir / "run_metrics.csv")
    print(f"Loaded {len(runs_df)} run records")

    feats_path = args.results_dir / "feature_metrics.csv"
    feats_df = None
    if feats_path.exists():
        feats_df = pd.read_csv(feats_path)
        print(f"Loaded {len(feats_df)} feature records")

    print(f"\nGenerating plots...")

    plot_a_rank_delocalization(runs_df, out_dir)

    if feats_df is not None:
        plot_b_featurewise_by_sparsity(feats_df, out_dir)
        plot_c_eigenvalue_reciprocal(feats_df, out_dir)
        plot_diracness_by_sparsity(feats_df, out_dir)
    else:
        print("Feature metrics not found. Skipping feature-level plots.")

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
