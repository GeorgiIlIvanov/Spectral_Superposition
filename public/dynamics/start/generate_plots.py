#!/usr/bin/env python3
"""
Visualization script for Capacity Localization Analysis.

Generates:
- Plot A: Rank-vs-delocalization decomposition scatter plots
- Plot B: Featurewise localization distributions across sparsity regimes
- Plot C: Eigenvalue-reciprocal fit analysis with delocalization coloring

Author: Claude Code Analysis Pipeline
Date: 2026-01-28
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Set up plotting style
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def plot_a_rank_delocalization(runs_df: pd.DataFrame, out_dir: Path):
    """
    Plot A: Rank-vs-delocalization decomposition.

    Scatter over all runs:
    - x = sparsity s
    - y1 = r/m (rank ratio)
    - y2 = rho_r (saturation wrt rank)
    - y3 = rho_m (saturation wrt m)

    This reveals whether the high-s peak is rank loss, delocalization, or both.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: r/m vs sparsity
    ax = axes[0, 0]
    scatter = ax.scatter(runs_df['s'], runs_df['rank_ratio'],
                        c=runs_df['m'], cmap='viridis', alpha=0.6, s=15)
    ax.set_xlabel('Sparsity (s)')
    ax.set_ylabel('Rank Ratio (r/m)')
    ax.set_title('Panel 1: Rank Ratio vs Sparsity')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Full rank')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='m (hidden dim)')

    # Panel 2: rho_r vs sparsity
    ax = axes[0, 1]
    scatter = ax.scatter(runs_df['s'], runs_df['rho_r'],
                        c=runs_df['m'], cmap='viridis', alpha=0.6, s=15)
    ax.set_xlabel('Sparsity (s)')
    ax.set_ylabel(r'$\rho_r$ (saturation wrt rank)')
    ax.set_title('Panel 2: Saturation wrt Rank vs Sparsity')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Full saturation')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='m (hidden dim)')

    # Panel 3: rho_m vs sparsity
    ax = axes[1, 0]
    scatter = ax.scatter(runs_df['s'], runs_df['rho_m'],
                        c=runs_df['m'], cmap='viridis', alpha=0.6, s=15)
    ax.set_xlabel('Sparsity (s)')
    ax.set_ylabel(r'$\rho_m$ (saturation wrt m)')
    ax.set_title('Panel 3: Saturation wrt Hidden Dim vs Sparsity')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Full saturation')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='m (hidden dim)')

    # Panel 4: Comparison rho_r vs rho_m
    ax = axes[1, 1]
    # Group by sparsity bins for clearer visualization
    low_s = runs_df[runs_df['s'] <= 0.3]
    mid_s = runs_df[(runs_df['s'] > 0.3) & (runs_df['s'] <= 0.6)]
    high_s = runs_df[(runs_df['s'] > 0.6) & (runs_df['s'] <= 0.9)]
    vhigh_s = runs_df[runs_df['s'] > 0.9]

    ax.scatter(low_s['rho_m'], low_s['rho_r'], alpha=0.5, s=15, label='Low s (0-0.3)', c='blue')
    ax.scatter(mid_s['rho_m'], mid_s['rho_r'], alpha=0.5, s=15, label='Mid s (0.3-0.6)', c='green')
    ax.scatter(high_s['rho_m'], high_s['rho_r'], alpha=0.5, s=15, label='High s (0.6-0.9)', c='orange')
    ax.scatter(vhigh_s['rho_m'], vhigh_s['rho_r'], alpha=0.5, s=15, label='V.High s (0.9-1.0)', c='red')
    ax.plot([0, 1.5], [0, 1.5], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel(r'$\rho_m$ (saturation wrt m)')
    ax.set_ylabel(r'$\rho_r$ (saturation wrt rank)')
    ax.set_title(r'Panel 4: $\rho_r$ vs $\rho_m$ by Sparsity Regime')
    ax.legend()
    ax.set_xlim(0, max(1.1, runs_df['rho_m'].max() * 1.05))
    ax.set_ylim(0, max(1.1, runs_df['rho_r'].max() * 1.05))

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_A_rank_delocalization.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_A_rank_delocalization.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved Plot A to {out_dir / 'plot_A_rank_delocalization.png'}")

    # Additional diagnostic plot: mean_sigma and mean_resid vs sparsity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    scatter = ax.scatter(runs_df['s'], runs_df['mean_sigma'],
                        c=runs_df['m'], cmap='viridis', alpha=0.6, s=15)
    ax.set_xlabel('Sparsity (s)')
    ax.set_ylabel('Mean Relative Slack (sigma)')
    ax.set_title('Mean Delocalization Slack vs Sparsity')
    plt.colorbar(scatter, ax=ax, label='m')

    ax = axes[1]
    scatter = ax.scatter(runs_df['s'], runs_df['mean_resid'],
                        c=runs_df['m'], cmap='viridis', alpha=0.6, s=15)
    ax.set_xlabel('Sparsity (s)')
    ax.set_ylabel('Mean Eigenvector Residual')
    ax.set_title('Mean Spectral Spread vs Sparsity')
    plt.colorbar(scatter, ax=ax, label='m')

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_A_supplementary_diagnostics.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_b_featurewise_localization(feats_df: pd.DataFrame, runs_df: pd.DataFrame, out_dir: Path):
    """
    Plot B: Featurewise localization changes across the phase transition.

    For representative sparsities (s in {0.0, 0.2, 0.6, 0.9}) and fixed m values,
    plot the distribution of sigma_i (or res_i), weighted by leverage.

    Prediction:
    - low s: mass concentrated near 0 (quasi-Dirac mu_i)
    - high s: heavier tail (delocalized residue)
    """
    # Select representative m values
    m_values = sorted(feats_df['m'].unique())
    m_representative = [m_values[len(m_values)//4], m_values[len(m_values)//2], m_values[3*len(m_values)//4]]

    # Sparsity bins
    s_bins = [0.0, 0.2, 0.6, 0.9]
    s_tolerance = 0.05

    fig, axes = plt.subplots(len(m_representative), len(s_bins), figsize=(16, 4*len(m_representative)))

    for i, m in enumerate(m_representative):
        for j, s_target in enumerate(s_bins):
            ax = axes[i, j] if len(m_representative) > 1 else axes[j]

            # Filter data
            mask = (feats_df['m'] == m) & (np.abs(feats_df['s'] - s_target) < s_tolerance)
            subset = feats_df[mask]

            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'm={m}, s~{s_target}')
                continue

            # Create leverage-weighted histogram of sigma
            sigma_vals = subset['sigma'].values
            ell_vals = subset['ell'].values

            # Filter out NaN and infinite values
            valid = np.isfinite(sigma_vals) & np.isfinite(ell_vals)
            sigma_vals = sigma_vals[valid]
            ell_vals = ell_vals[valid]

            if len(sigma_vals) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'm={m}, s~{s_target}')
                continue

            # Clip sigma for better visualization
            sigma_clipped = np.clip(sigma_vals, -0.5, 1.5)

            # Weighted histogram
            bins = np.linspace(-0.5, 1.5, 50)
            hist, bin_edges = np.histogram(sigma_clipped, bins=bins, weights=ell_vals, density=True)

            ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.7, edgecolor='black', linewidth=0.3)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Perfect localization')
            ax.set_xlabel(r'Relative Slack $\sigma_i = 1 - D_i/\ell_i$')
            ax.set_ylabel('Leverage-weighted density')
            ax.set_title(f'm={m}, s~{s_target:.1f}')

            # Add statistics
            mean_sigma = np.average(sigma_vals, weights=ell_vals)
            ax.axvline(x=mean_sigma, color='green', linestyle='-', alpha=0.7, label=f'Mean={mean_sigma:.3f}')
            ax.legend(fontsize=8)

    plt.suptitle('Plot B: Featurewise Localization Distribution by Sparsity Regime\n(Leverage-weighted)', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'plot_B_featurewise_localization.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_B_featurewise_localization.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved Plot B to {out_dir / 'plot_B_featurewise_localization.png'}")

    # Additional: Residual distribution
    fig, axes = plt.subplots(len(m_representative), len(s_bins), figsize=(16, 4*len(m_representative)))

    for i, m in enumerate(m_representative):
        for j, s_target in enumerate(s_bins):
            ax = axes[i, j] if len(m_representative) > 1 else axes[j]

            mask = (feats_df['m'] == m) & (np.abs(feats_df['s'] - s_target) < s_tolerance)
            subset = feats_df[mask]

            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'm={m}, s~{s_target}')
                continue

            resid_vals = subset['resid'].values
            ell_vals = subset['ell'].values

            valid = np.isfinite(resid_vals) & np.isfinite(ell_vals)
            resid_vals = resid_vals[valid]
            ell_vals = ell_vals[valid]

            if len(resid_vals) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'm={m}, s~{s_target}')
                continue

            bins = np.linspace(0, resid_vals.max() * 1.1, 50)
            hist, bin_edges = np.histogram(resid_vals, bins=bins, weights=ell_vals, density=True)

            ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.7, edgecolor='black', linewidth=0.3)
            ax.set_xlabel(r'Eigenvector Residual $\sqrt{Var_{\mu_i}(\lambda)}$')
            ax.set_ylabel('Leverage-weighted density')
            ax.set_title(f'm={m}, s~{s_target:.1f}')

            mean_resid = np.average(resid_vals, weights=ell_vals)
            ax.axvline(x=mean_resid, color='green', linestyle='-', alpha=0.7, label=f'Mean={mean_resid:.3f}')
            ax.legend(fontsize=8)

    plt.suptitle('Plot B (Supplement): Eigenvector Residual Distribution by Sparsity Regime\n(Leverage-weighted)', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'plot_B_residual_distribution.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_c_eigenvalue_reciprocal(feats_df: pd.DataFrame, runs_df: pd.DataFrame, out_dir: Path):
    """
    Plot C: Why eigenvalue-reciprocal fits fail.

    Color features by sigma_i or resid_i to show that points breaking the
    linear reciprocal-eigenvalue fit are exactly the high-sigma/high-resid subset.

    This validates that discrepancy is a feature-level delocalization phenomenon.
    """
    # Select a few representative runs with varying sparsity
    # Pick runs with m close to median and different s values
    m_median = runs_df['m'].median()
    closest_m = runs_df.iloc[(runs_df['m'] - m_median).abs().argsort()[:1]]['m'].values[0]

    # Get runs with this m value
    runs_subset = runs_df[runs_df['m'] == closest_m]

    s_targets = [0.0, 0.3, 0.6, 0.9]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, s_target in enumerate(s_targets):
        ax = axes[idx]

        # Find closest sparsity
        s_diffs = np.abs(runs_subset['s'] - s_target)
        if len(s_diffs) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        closest_run = runs_subset.iloc[s_diffs.argmin()]
        actual_s = closest_run['s']
        actual_m = closest_run['m']
        actual_seed = closest_run['seed']

        # Get features for this run
        feat_mask = (feats_df['m'] == actual_m) & \
                   (np.abs(feats_df['s'] - actual_s) < 0.01) & \
                   (feats_df['seed'] == actual_seed)
        feat_subset = feats_df[feat_mask]

        if len(feat_subset) == 0:
            ax.text(0.5, 0.5, 'No feature data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f's~{s_target}')
            continue

        # Plot: x = 1/kappa (reciprocal effective eigenvalue), y = D_i, color = sigma_i
        kappa_vals = feat_subset['kappa'].values
        D_vals = feat_subset['D'].values
        sigma_vals = feat_subset['sigma'].values

        # Filter valid
        valid = np.isfinite(kappa_vals) & (kappa_vals > 1e-10) & np.isfinite(D_vals) & np.isfinite(sigma_vals)
        kappa_vals = kappa_vals[valid]
        D_vals = D_vals[valid]
        sigma_vals = sigma_vals[valid]

        if len(kappa_vals) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f's~{s_target}')
            continue

        inv_kappa = 1.0 / kappa_vals

        # Normalize sigma for coloring
        norm = Normalize(vmin=0, vmax=max(0.5, np.percentile(sigma_vals, 95)))
        colors = cm.RdYlBu_r(norm(np.clip(sigma_vals, 0, 1)))

        scatter = ax.scatter(inv_kappa, D_vals, c=sigma_vals, cmap='RdYlBu_r',
                            norm=norm, alpha=0.7, s=10)

        # Fit line to low-sigma points (well-localized)
        low_sigma_mask = sigma_vals < 0.1
        if np.sum(low_sigma_mask) > 10:
            x_fit = inv_kappa[low_sigma_mask]
            y_fit = D_vals[low_sigma_mask]
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            x_line = np.array([inv_kappa.min(), inv_kappa.max()])
            ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5,
                   label=f'Linear fit (low-$\\sigma$): slope={slope:.2f}')
            ax.legend(fontsize=8)

        ax.set_xlabel(r'$1/\kappa_i$ (reciprocal effective eigenvalue)')
        ax.set_ylabel(r'$D_i$ (fractional dimension)')
        ax.set_title(f'm={actual_m}, s={actual_s:.2f}, seed={actual_seed}')
        plt.colorbar(scatter, ax=ax, label=r'$\sigma_i$ (relative slack)')

    plt.suptitle('Plot C: Eigenvalue-Reciprocal Relationship\nColored by Delocalization Slack', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'plot_C_eigenvalue_reciprocal.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_C_eigenvalue_reciprocal.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved Plot C to {out_dir / 'plot_C_eigenvalue_reciprocal.png'}")


def plot_diracness_validation(feats_df: pd.DataFrame, runs_df: pd.DataFrame, out_dir: Path):
    """
    Step 3 Validation: Diracness proxy analysis.

    Validate that:
    - Low-s runs have q_i ≈ 1 for almost all leverage mass
    - High-s runs have nontrivial tail of q_i < 1
    """
    # Sparsity bins
    s_bins = [(0, 0.3, 'Low s (0-0.3)'),
              (0.3, 0.6, 'Mid s (0.3-0.6)'),
              (0.6, 0.9, 'High s (0.6-0.9)'),
              (0.9, 1.0, 'V.High s (0.9-1.0)')]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (s_low, s_high, label) in enumerate(s_bins):
        ax = axes[idx]

        mask = (feats_df['s'] >= s_low) & (feats_df['s'] < s_high)
        subset = feats_df[mask]

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        q_vals = subset['q'].values
        ell_vals = subset['ell'].values

        valid = np.isfinite(q_vals) & np.isfinite(ell_vals)
        q_vals = q_vals[valid]
        ell_vals = ell_vals[valid]

        if len(q_vals) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        # Weighted histogram
        bins = np.linspace(0, 1, 50)
        hist, bin_edges = np.histogram(q_vals, bins=bins, weights=ell_vals, density=True)

        ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Dirac')
        ax.set_xlabel(r'Diracness Proxy $q_i = \max_{group} p_{i,group}$')
        ax.set_ylabel('Leverage-weighted density')
        ax.set_title(label)

        # Statistics
        mean_q = np.average(q_vals, weights=ell_vals)
        frac_localized = np.sum(ell_vals[q_vals > 0.9]) / np.sum(ell_vals)
        ax.axvline(x=mean_q, color='green', linestyle='-', alpha=0.7)
        ax.text(0.05, 0.95, f'Mean q: {mean_q:.3f}\nFrac q>0.9: {frac_localized:.2%}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(fontsize=8)

    plt.suptitle('Step 3 Validation: Diracness Proxy Distribution by Sparsity\n(Leverage-weighted)', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'plot_step3_diracness_validation.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_step3_diracness_validation.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved Diracness validation plot to {out_dir / 'plot_step3_diracness_validation.png'}")


def plot_tail_mass_curves(runs_df: pd.DataFrame, out_dir: Path):
    """
    Plot tail mass curves vs sparsity for different thresholds.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for col, label, color in [('tail_mass_001', r'$\tau=0.01$', 'blue'),
                               ('tail_mass_005', r'$\tau=0.05$', 'green'),
                               ('tail_mass_010', r'$\tau=0.10$', 'red')]:
        if col not in runs_df.columns:
            continue

        # Bin by sparsity and compute mean
        runs_df['s_bin'] = pd.cut(runs_df['s'], bins=20)
        grouped = runs_df.groupby('s_bin', observed=True)[col].agg(['mean', 'std'])
        grouped = grouped.dropna()

        if len(grouped) == 0:
            continue

        # Get bin centers
        s_centers = [interval.mid for interval in grouped.index]

        ax.errorbar(s_centers, grouped['mean'], yerr=grouped['std']/2, fmt='o-',
                   label=label, color=color, alpha=0.7, capsize=3)

    ax.set_xlabel('Sparsity (s)')
    ax.set_ylabel('Tail Mass (fraction of leverage in high-slack features)')
    ax.set_title('Tail Mass Curves: Delocalization vs Sparsity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_tail_mass_curves.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved tail mass curves to {out_dir / 'plot_tail_mass_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualization plots for localization analysis")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing run_metrics.csv and feature_metrics.csv")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory for plots (default: same as results-dir)")

    args = parser.parse_args()

    out_dir = args.out or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.results_dir}")

    # Load run metrics
    runs_path = args.results_dir / "run_metrics.csv"
    if not runs_path.exists():
        raise FileNotFoundError(f"Run metrics not found: {runs_path}")
    runs_df = pd.read_csv(runs_path)
    print(f"Loaded {len(runs_df)} run records")

    # Load feature metrics if available
    feats_path = args.results_dir / "feature_metrics.csv"
    feats_df = None
    if feats_path.exists():
        feats_df = pd.read_csv(feats_path)
        print(f"Loaded {len(feats_df)} feature records")

    print(f"\nGenerating plots...")

    # Plot A: Rank-vs-delocalization decomposition
    plot_a_rank_delocalization(runs_df, out_dir)

    # Plot tail mass curves
    plot_tail_mass_curves(runs_df, out_dir)

    # Plots requiring feature metrics
    if feats_df is not None:
        # Plot B: Featurewise localization distributions
        plot_b_featurewise_localization(feats_df, runs_df, out_dir)

        # Plot C: Eigenvalue-reciprocal fits
        plot_c_eigenvalue_reciprocal(feats_df, runs_df, out_dir)

        # Step 3: Diracness validation
        plot_diracness_validation(feats_df, runs_df, out_dir)
    else:
        print("\nNote: Feature metrics not found. Skipping Plots B, C, and Diracness validation.")
        print("Re-run analysis with --save-feature-metrics to generate these plots.")

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
