#!/usr/bin/env python3
"""
Generate plots for v3 eigenspace-based capacity localization analysis.

Key changes from v2:
- Plots use eigenspace-based metrics (max_space_proj, participation_ratio)
- plot_A_supplementary_diagnostics split into two separate plots:
  - plot_mean_sigma_vs_sparsity.png (delocalization slack)
  - plot_mean_resid_vs_sparsity.png (spectral spread)

Author: Claude Code Analysis Pipeline
Date: 2026-01-29
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def plot_rank_saturation(runs_df: pd.DataFrame, out_dir: Path):
    """Plot rank ratio and capacity saturation vs sparsity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Rank ratio
    ax = axes[0]
    scatter = ax.scatter(runs_df['s'], runs_df['rank_ratio'],
                        c=runs_df['m'], cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('Sparsity (s)', fontsize=12)
    ax.set_ylabel('Rank Ratio (r/m)', fontsize=12)
    ax.set_title('Effective Rank vs Sparsity', fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='m')

    # Panel 2: Capacity saturation
    ax = axes[1]
    scatter = ax.scatter(runs_df['s'], runs_df['rho_r'],
                        c=runs_df['m'], cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('Sparsity (s)', fontsize=12)
    ax.set_ylabel('ρ_r = Σᵢ Dᵢ / r', fontsize=12)
    ax.set_title('Capacity Saturation vs Sparsity', fontsize=13)
    ax.axhline(1, color='red', linestyle='--', alpha=0.7, label='Full saturation')
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='m')

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_rank_saturation.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_rank_saturation.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: plot_rank_saturation.png")


def plot_mean_sigma_vs_sparsity(runs_df: pd.DataFrame, out_dir: Path):
    """
    STANDALONE plot: Mean Delocalization Slack (sigma) vs Sparsity.
    Colored by hidden dimension m.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(runs_df['s'], runs_df['mean_sigma'],
                        c=runs_df['m'], cmap='viridis', alpha=0.7, s=25,
                        edgecolors='none')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Hidden Dimension (m)', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel('Sparsity (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Relative Slack (σ)', fontsize=14, fontweight='bold')
    ax.set_title('Mean Delocalization Slack vs Sparsity', fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_mean_sigma_vs_sparsity.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_mean_sigma_vs_sparsity.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: plot_mean_sigma_vs_sparsity.png")


def plot_mean_resid_vs_sparsity(runs_df: pd.DataFrame, out_dir: Path):
    """
    STANDALONE plot: Mean Spectral Spread (resid) vs Sparsity.
    Colored by hidden dimension m.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(runs_df['s'], runs_df['mean_resid'],
                        c=runs_df['m'], cmap='viridis', alpha=0.7, s=25,
                        edgecolors='none')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Hidden Dimension (m)', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel('Sparsity (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Spectral Spread (resid)', fontsize=14, fontweight='bold')
    ax.set_title('Mean Spectral Spread vs Sparsity', fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_mean_resid_vs_sparsity.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_mean_resid_vs_sparsity.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: plot_mean_resid_vs_sparsity.png")


def plot_eigenspace_localization(runs_df: pd.DataFrame, out_dir: Path):
    """
    Plot eigenspace-specific metrics: max_space_proj and participation_ratio.
    """
    if 'wmean_max_space_proj' not in runs_df.columns:
        print("Skipping eigenspace localization plot (v3 metrics not found)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Max eigenspace projection
    ax = axes[0]
    scatter = ax.scatter(runs_df['s'], runs_df['wmean_max_space_proj'],
                        c=runs_df['m'], cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('Sparsity (s)', fontsize=12)
    ax.set_ylabel('Weighted Mean Max Eigenspace Projection', fontsize=12)
    ax.set_title('Eigenspace Localization vs Sparsity', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='m')

    # Panel 2: Participation ratio
    ax = axes[1]
    scatter = ax.scatter(runs_df['s'], runs_df['wmean_participation_ratio'],
                        c=runs_df['m'], cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('Sparsity (s)', fontsize=12)
    ax.set_ylabel('Weighted Mean Participation Ratio', fontsize=12)
    ax.set_title('Eigenspace Delocalization vs Sparsity', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='m')

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_eigenspace_localization.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_eigenspace_localization.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: plot_eigenspace_localization.png")


def plot_q_bin_vs_sparsity(runs_df: pd.DataFrame, out_dir: Path):
    """Plot binned Diracness proxy (q_bin) vs sparsity."""
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(runs_df['s'], runs_df['wmean_q_bin'],
                        c=runs_df['m'], cmap='viridis', alpha=0.7, s=25)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Hidden Dimension (m)', fontsize=13, fontweight='bold')

    ax.set_xlabel('Sparsity (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Weighted Mean q_bin', fontsize=14, fontweight='bold')
    ax.set_title('Binned Diracness Proxy vs Sparsity\n(Eigenspace-based)', fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_q_bin_vs_sparsity.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_q_bin_vs_sparsity.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: plot_q_bin_vs_sparsity.png")


def plot_cv_vs_sparsity(runs_df: pd.DataFrame, out_dir: Path):
    """Plot coefficient of variation vs sparsity."""
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(runs_df['s'], runs_df['mean_cv'],
                        c=runs_df['m'], cmap='viridis', alpha=0.7, s=25)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Hidden Dimension (m)', fontsize=13, fontweight='bold')

    ax.set_xlabel('Sparsity (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean CV (resid/kappa)', fontsize=14, fontweight='bold')
    ax.set_title('Coefficient of Variation vs Sparsity', fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_cv_vs_sparsity.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_cv_vs_sparsity.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: plot_cv_vs_sparsity.png")


def plot_degeneracy_analysis(runs_df: pd.DataFrame, out_dir: Path):
    """Plot eigenvalue degeneracy analysis."""
    if 'n_eigenspaces' not in runs_df.columns or 'degeneracy_ratio' not in runs_df.columns:
        print("Skipping degeneracy analysis (v3 metrics not found)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Degeneracy ratio vs m
    ax = axes[0]
    scatter = ax.scatter(runs_df['m'], runs_df['degeneracy_ratio'],
                        c=runs_df['s'], cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('Hidden Dimension (m)', fontsize=12)
    ax.set_ylabel('Degeneracy Ratio (n_spaces / rank)', fontsize=12)
    ax.set_title('Eigenvalue Degeneracy vs Model Size', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Sparsity')

    # Panel 2: Number of eigenspaces vs m
    ax = axes[1]
    scatter = ax.scatter(runs_df['m'], runs_df['n_eigenspaces'],
                        c=runs_df['s'], cmap='viridis', alpha=0.6, s=20)
    ax.plot([0, runs_df['m'].max()], [0, runs_df['m'].max()], 'r--', alpha=0.5, label='n_spaces = m')
    ax.set_xlabel('Hidden Dimension (m)', fontsize=12)
    ax.set_ylabel('Number of Eigenspaces', fontsize=12)
    ax.set_title('Distinct Eigenspaces vs Model Size', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Sparsity')

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_degeneracy_analysis.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_degeneracy_analysis.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: plot_degeneracy_analysis.png")


def main():
    parser = argparse.ArgumentParser(description="Generate v3 analysis plots")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing run_metrics.csv")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory for plots (default: results-dir)")

    args = parser.parse_args()

    out_dir = args.out or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    runs_df = pd.read_csv(args.results_dir / "run_metrics.csv")
    print(f"Loaded {len(runs_df)} runs from {args.results_dir}")

    # Generate all plots
    print("\nGenerating plots...")

    plot_rank_saturation(runs_df, out_dir)
    plot_mean_sigma_vs_sparsity(runs_df, out_dir)  # SPLIT: standalone
    plot_mean_resid_vs_sparsity(runs_df, out_dir)  # SPLIT: standalone
    plot_eigenspace_localization(runs_df, out_dir)
    plot_q_bin_vs_sparsity(runs_df, out_dir)
    plot_cv_vs_sparsity(runs_df, out_dir)
    plot_degeneracy_analysis(runs_df, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
