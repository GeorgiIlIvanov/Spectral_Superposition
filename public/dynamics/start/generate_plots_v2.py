#!/usr/bin/env python3
"""
Visualization script for Capacity Localization Analysis (v2).

Fixed version implementing:
- Binned Diracness proxy (q_bin) distribution plots instead of broken eigenvalue grouping
- Coefficient of variation (cv) vs sparsity plots
- Qualification rate (qual_rate) vs sparsity plots
- Fixed Plot C using D_hat when negD_frac > 0
- Assignment test: D_hat vs D_hat_pred for confidently localized features

Author: Claude Code Analysis Pipeline
Date: 2026-01-29
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy import stats

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

# Sparsity bins for stratification
S_BINS = [
    (0, 0.3, 'Low s (0-0.3)', 'blue'),
    (0.3, 0.6, 'Mid s (0.3-0.6)', 'green'),
    (0.6, 0.9, 'High s (0.6-0.9)', 'orange'),
    (0.9, 1.0, 'V.High s (0.9-1.0)', 'red')
]

LOCALIZATION_THRESHOLD = 0.9  # Must match analysis script


def load_metadata(results_dir: Path) -> dict:
    """Load analysis metadata if available."""
    meta_path = results_dir / "analysis_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def plot_a_rank_delocalization(runs_df: pd.DataFrame, out_dir: Path):
    """
    Plot A: Rank-vs-delocalization decomposition.

    Scatter over all runs:
    - x = sparsity s
    - y1 = r/m (rank ratio)
    - y2 = rho_r (saturation wrt rank)
    - y3 = rho_m (saturation wrt m)
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
    for s_low, s_high, label, color in S_BINS:
        subset = runs_df[(runs_df['s'] >= s_low) & (runs_df['s'] < s_high)]
        if len(subset) > 0:
            ax.scatter(subset['rho_m'], subset['rho_r'], alpha=0.5, s=15,
                      label=label, c=color)
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

    # Additional diagnostic: mean_sigma and mean_resid vs sparsity
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


def plot_cv_vs_sparsity(runs_df: pd.DataFrame, out_dir: Path):
    """
    Plot mean coefficient of variation (cv) vs sparsity.

    cv = resid / kappa is the scale-free spectral spread measure.
    Should increase with sparsity and correlate with sigma.
    """
    if 'mean_cv' not in runs_df.columns:
        print("Warning: mean_cv not in run metrics, skipping cv plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: mean_cv vs sparsity
    ax = axes[0]
    scatter = ax.scatter(runs_df['s'], runs_df['mean_cv'],
                        c=runs_df['m'], cmap='viridis', alpha=0.6, s=15)
    ax.set_xlabel('Sparsity (s)')
    ax.set_ylabel('Mean CV (resid/kappa)')
    ax.set_title('Coefficient of Variation vs Sparsity\n(Scale-free spectral spread)')
    plt.colorbar(scatter, ax=ax, label='m')

    # Panel 2: mean_cv vs mean_sigma (should correlate)
    ax = axes[1]
    scatter = ax.scatter(runs_df['mean_sigma'], runs_df['mean_cv'],
                        c=runs_df['s'], cmap='RdYlBu_r', alpha=0.6, s=15)
    ax.set_xlabel('Mean Relative Slack (sigma)')
    ax.set_ylabel('Mean CV (resid/kappa)')
    ax.set_title('CV vs Sigma (Correlation check)')

    # Add regression line
    valid = np.isfinite(runs_df['mean_sigma']) & np.isfinite(runs_df['mean_cv'])
    if valid.sum() > 2:
        x = runs_df.loc[valid, 'mean_sigma'].values
        y = runs_df.loc[valid, 'mean_cv'].values
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        ax.plot([x.min(), x.max()],
               [slope * x.min() + intercept, slope * x.max() + intercept],
               'k--', alpha=0.5, label=f'R²={r_value**2:.3f}')
        ax.legend()

    plt.colorbar(scatter, ax=ax, label='Sparsity (s)')

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_cv_vs_sparsity.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_cv_vs_sparsity.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved CV plot to {out_dir / 'plot_cv_vs_sparsity.png'}")


def plot_q_bin_distributions(feats_df: pd.DataFrame, runs_df: pd.DataFrame, out_dir: Path):
    """
    Plot binned Diracness proxy (q_bin) distributions stratified by sparsity.

    This replaces the broken eigenvalue-grouping based q metric.
    Expectation:
    - Low sparsity: mass concentrated near q_bin=1 (localized)
    - High sparsity: mass shifts toward lower q_bin (delocalized)
    """
    if 'q_bin' not in feats_df.columns:
        print("Warning: q_bin not in feature metrics, skipping q_bin distribution plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (s_low, s_high, label, color) in enumerate(S_BINS):
        ax = axes[idx]

        mask = (feats_df['s'] >= s_low) & (feats_df['s'] < s_high)
        subset = feats_df[mask]

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        q_vals = subset['q_bin'].values
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

        ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.7,
              edgecolor='black', linewidth=0.3, color=color)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Dirac')
        ax.axvline(x=LOCALIZATION_THRESHOLD, color='purple', linestyle=':',
                  alpha=0.7, label=f'Threshold ({LOCALIZATION_THRESHOLD})')
        ax.set_xlabel(r'Binned Diracness $q_{bin} = \max_b \mu_i(B_b)$')
        ax.set_ylabel('Leverage-weighted density')
        ax.set_title(label)

        # Statistics
        mean_q = np.average(q_vals, weights=ell_vals)
        frac_localized = np.sum(ell_vals[q_vals >= LOCALIZATION_THRESHOLD]) / np.sum(ell_vals)
        ax.axvline(x=mean_q, color='green', linestyle='-', alpha=0.7)
        ax.text(0.05, 0.95, f'Mean q_bin: {mean_q:.3f}\n'
                           f'Frac q>={LOCALIZATION_THRESHOLD}: {frac_localized:.1%}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(fontsize=8)

    plt.suptitle('Binned Diracness Proxy (q_bin) Distribution by Sparsity\n'
                '(Basis-invariant, Leverage-weighted)', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'plot_q_bin_distribution.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_q_bin_distribution.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved q_bin distribution plot to {out_dir / 'plot_q_bin_distribution.png'}")


def plot_qual_rate_vs_sparsity(runs_df: pd.DataFrame, out_dir: Path):
    """
    Plot qualification rate vs sparsity.

    qual_rate = leverage-weighted fraction of features with q_bin >= threshold.
    Should decrease with sparsity (fewer confidently localized features).
    """
    if 'qual_rate' not in runs_df.columns:
        print("Warning: qual_rate not in run metrics, skipping qual_rate plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: qual_rate vs sparsity (scatter)
    ax = axes[0]
    scatter = ax.scatter(runs_df['s'], runs_df['qual_rate'],
                        c=runs_df['m'], cmap='viridis', alpha=0.6, s=15)
    ax.set_xlabel('Sparsity (s)')
    ax.set_ylabel(f'Qualification Rate (q_bin >= {LOCALIZATION_THRESHOLD})')
    ax.set_title('Fraction of Confidently Localized Features vs Sparsity')
    ax.set_ylim(-0.05, 1.05)
    plt.colorbar(scatter, ax=ax, label='m')

    # Panel 2: qual_rate by sparsity bin (boxplot-style)
    ax = axes[1]
    data_by_bin = []
    labels = []
    for s_low, s_high, label, color in S_BINS:
        subset = runs_df[(runs_df['s'] >= s_low) & (runs_df['s'] < s_high)]
        if len(subset) > 0:
            data_by_bin.append(subset['qual_rate'].dropna().values)
            labels.append(label.split()[0] + '\n' + label.split()[1])

    if data_by_bin:
        bp = ax.boxplot(data_by_bin, labels=labels, patch_artist=True)
        colors = [S_BINS[i][3] for i in range(len(data_by_bin))]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_ylabel(f'Qualification Rate')
        ax.set_title(f'Qualification Rate Distribution by Sparsity Regime\n'
                    f'(q_bin >= {LOCALIZATION_THRESHOLD})')
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_qual_rate_vs_sparsity.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_qual_rate_vs_sparsity.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved qualification rate plot to {out_dir / 'plot_qual_rate_vs_sparsity.png'}")


def plot_b_featurewise_localization(feats_df: pd.DataFrame, runs_df: pd.DataFrame, out_dir: Path):
    """
    Plot B: Featurewise localization changes across sparsity regimes.

    Shows leverage-weighted distribution of sigma_i.
    """
    # Select representative m values
    m_values = sorted(feats_df['m'].unique())
    n_m = len(m_values)
    m_representative = [
        m_values[n_m // 4],
        m_values[n_m // 2],
        m_values[3 * n_m // 4]
    ]

    s_targets = [0.0, 0.3, 0.6, 0.9]
    s_tolerance = 0.05

    fig, axes = plt.subplots(len(m_representative), len(s_targets),
                            figsize=(16, 4 * len(m_representative)))

    for i, m in enumerate(m_representative):
        for j, s_target in enumerate(s_targets):
            ax = axes[i, j] if len(m_representative) > 1 else axes[j]

            mask = (feats_df['m'] == m) & (np.abs(feats_df['s'] - s_target) < s_tolerance)
            subset = feats_df[mask]

            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'm={m}, s~{s_target}')
                continue

            sigma_vals = subset['sigma'].values
            ell_vals = subset['ell'].values

            valid = np.isfinite(sigma_vals) & np.isfinite(ell_vals)
            sigma_vals = sigma_vals[valid]
            ell_vals = ell_vals[valid]

            if len(sigma_vals) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'm={m}, s~{s_target}')
                continue

            # Clip sigma for visualization
            sigma_clipped = np.clip(sigma_vals, -0.5, 1.5)

            # Weighted histogram
            bins = np.linspace(-0.5, 1.5, 50)
            hist, bin_edges = np.histogram(sigma_clipped, bins=bins, weights=ell_vals, density=True)

            ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.7,
                  edgecolor='black', linewidth=0.3)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Perfect localization')
            ax.set_xlabel(r'Relative Slack $\sigma_i = 1 - D_i/\ell_i$')
            ax.set_ylabel('Leverage-weighted density')
            ax.set_title(f'm={m}, s~{s_target:.1f}')

            mean_sigma = np.average(sigma_vals, weights=ell_vals)
            ax.axvline(x=mean_sigma, color='green', linestyle='-', alpha=0.7,
                      label=f'Mean={mean_sigma:.3f}')
            ax.legend(fontsize=8)

    plt.suptitle('Plot B: Featurewise Localization Distribution by Sparsity Regime\n'
                '(Leverage-weighted)', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'plot_B_featurewise_localization.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_B_featurewise_localization.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved Plot B to {out_dir / 'plot_B_featurewise_localization.png'}")


def plot_c_assignment_test(feats_df: pd.DataFrame, runs_df: pd.DataFrame, out_dir: Path):
    """
    Plot C (Fixed): Assignment test for confidently localized features.

    For features with q_bin >= threshold (confidently localized to a bin):
    - D_hat_pred[i] = norm2[i] / lambda_hat[i]
    - Compare D_hat vs D_hat_pred

    This replaces the broken D vs 1/kappa plot which was trivially identity.

    Expectation:
    - Low sparsity: many features qualify, prediction is tight
    - High sparsity: fewer qualify OR prediction loosens
    """
    required_cols = ['q_bin', 'D_hat', 'D_hat_pred', 'norm2', 'lambda_hat']
    if not all(col in feats_df.columns for col in required_cols):
        print(f"Warning: Missing columns for assignment test: {required_cols}")
        print(f"Available: {list(feats_df.columns)}")
        return

    # Check if we should use D_hat instead of D
    use_D_hat = True  # Always use D_hat in the fixed version
    if 'negD_frac' in runs_df.columns:
        max_negD_frac = runs_df['negD_frac'].max()
        if max_negD_frac > 0:
            print(f"Note: negD_frac={max_negD_frac:.4f} > 0, using D_hat instead of D")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    fit_results = []

    for idx, (s_low, s_high, label, color) in enumerate(S_BINS):
        ax = axes[idx]

        # Filter by sparsity regime
        mask = (feats_df['s'] >= s_low) & (feats_df['s'] < s_high)
        subset = feats_df[mask]

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        # Filter to confidently localized features
        qualified = subset[subset['q_bin'] >= LOCALIZATION_THRESHOLD]
        n_total = len(subset)
        n_qualified = len(qualified)

        if n_qualified < 10:
            ax.text(0.5, 0.5, f'Too few qualified\n({n_qualified}/{n_total})',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label}\n(n_qualified={n_qualified})')
            continue

        D_hat = qualified['D_hat'].values
        D_hat_pred = qualified['D_hat_pred'].values

        # Filter valid values
        valid = np.isfinite(D_hat) & np.isfinite(D_hat_pred) & (D_hat > 0) & (D_hat_pred > 0)
        D_hat = D_hat[valid]
        D_hat_pred = D_hat_pred[valid]

        if len(D_hat) < 10:
            ax.text(0.5, 0.5, f'Too few valid\n({len(D_hat)})',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        # Scatter plot
        scatter = ax.scatter(D_hat_pred, D_hat, alpha=0.3, s=5, c=color)

        # Fit line
        slope, intercept, r_value, _, _ = stats.linregress(D_hat_pred, D_hat)
        x_line = np.array([D_hat_pred.min(), D_hat_pred.max()])
        ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.7,
               label=f'Fit: R²={r_value**2:.3f}, slope={slope:.3f}')

        # Identity line
        max_val = max(D_hat.max(), D_hat_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r-', alpha=0.3, label='y=x')

        ax.set_xlabel(r'$\hat{D}_{pred,i} = ||w_i||^2 / \hat{\lambda}_i$')
        ax.set_ylabel(r'$\hat{D}_i = ||w_i||^2 / \kappa_i$')
        ax.set_title(f'{label}\nn_qual={n_qualified}/{n_total} ({100*n_qualified/n_total:.1f}%)')
        ax.legend(fontsize=8)

        fit_results.append({
            's_regime': label,
            'n_total': n_total,
            'n_qualified': n_qualified,
            'qual_frac': n_qualified / n_total,
            'R2': r_value ** 2,
            'slope': slope,
            'intercept': intercept
        })

    plt.suptitle('Plot C (Fixed): Assignment Test for Localized Features\n'
                f'(q_bin >= {LOCALIZATION_THRESHOLD})', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'plot_C_assignment_test.png', dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / 'plot_C_assignment_test.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved Plot C to {out_dir / 'plot_C_assignment_test.png'}")

    # Save fit results
    if fit_results:
        fit_df = pd.DataFrame(fit_results)
        fit_df.to_csv(out_dir / 'plot_C_assignment_test_fits.csv', index=False)
        print(f"Saved assignment test fits to {out_dir / 'plot_C_assignment_test_fits.csv'}")
        print("\nAssignment Test Summary:")
        print(fit_df.to_string(index=False))


def plot_d_hat_comparison(feats_df: pd.DataFrame, runs_df: pd.DataFrame, out_dir: Path):
    """
    Diagnostic plot: D vs D_hat comparison.

    Shows whether stored D values are consistent with reconstructed D_hat.
    If negD_frac > 0, this reveals the issue.
    """
    if 'D_hat' not in feats_df.columns:
        print("Warning: D_hat not in feature metrics, skipping D comparison plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (s_low, s_high, label, color) in enumerate(S_BINS):
        ax = axes[idx]

        mask = (feats_df['s'] >= s_low) & (feats_df['s'] < s_high)
        subset = feats_df[mask]

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        D_vals = subset['D'].values
        D_hat_vals = subset['D_hat'].values

        valid = np.isfinite(D_vals) & np.isfinite(D_hat_vals)
        D_vals = D_vals[valid]
        D_hat_vals = D_hat_vals[valid]

        if len(D_vals) < 10:
            ax.text(0.5, 0.5, 'Too few valid', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        # Count negative D values
        n_neg = (D_vals < 0).sum()
        neg_frac = n_neg / len(D_vals)

        # Scatter plot
        colors = np.where(D_vals < 0, 'red', color)
        ax.scatter(D_hat_vals, D_vals, alpha=0.3, s=5, c=colors)

        # Correlation
        if len(D_vals) > 2:
            corr = np.corrcoef(D_vals, D_hat_vals)[0, 1]
        else:
            corr = float('nan')

        # Identity line
        max_val = max(D_vals.max(), D_hat_vals.max())
        min_val = min(D_vals.min(), D_hat_vals.min(), 0)
        ax.plot([min_val, max_val], [min_val, max_val], 'r-', alpha=0.5, label='y=x')
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel(r'$\hat{D}_i = ||w_i||^2 / \kappa_i$')
        ax.set_ylabel(r'$D_i$ (stored fractional dim)')
        ax.set_title(f'{label}\nCorr={corr:.3f}, negD_frac={neg_frac:.3f}')
        ax.legend(fontsize=8)

        if neg_frac > 0:
            ax.text(0.95, 0.05, f'{n_neg} negative D values\n(shown in red)',
                   transform=ax.transAxes, ha='right', va='bottom',
                   fontsize=9, color='red',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Diagnostic: D (stored) vs D_hat (reconstructed)\n'
                'Red points indicate impossible negative D values', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'plot_D_vs_D_hat_diagnostic.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved D vs D_hat diagnostic to {out_dir / 'plot_D_vs_D_hat_diagnostic.png'}")


def plot_H_bin_distribution(feats_df: pd.DataFrame, runs_df: pd.DataFrame, out_dir: Path):
    """
    Plot H_bin (binned entropy) and n_eff (effective support) distributions.
    """
    if 'H_bin' not in feats_df.columns or 'n_eff' not in feats_df.columns:
        print("Warning: H_bin/n_eff not in feature metrics, skipping entropy plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top row: H_bin distributions by sparsity
    ax = axes[0, 0]
    for s_low, s_high, label, color in S_BINS:
        mask = (feats_df['s'] >= s_low) & (feats_df['s'] < s_high)
        subset = feats_df[mask]
        if len(subset) > 0:
            H_vals = subset['H_bin'].values
            ell_vals = subset['ell'].values
            valid = np.isfinite(H_vals) & np.isfinite(ell_vals)
            if valid.sum() > 0:
                mean_H = np.average(H_vals[valid], weights=ell_vals[valid])
                ax.hist(H_vals[valid], bins=50, alpha=0.5, label=f'{label[:8]} (mean={mean_H:.2f})',
                       color=color, density=True, weights=ell_vals[valid])
    ax.set_xlabel('Binned Entropy $H_{bin}$')
    ax.set_ylabel('Leverage-weighted density')
    ax.set_title('Binned Spectral Entropy Distribution')
    ax.legend(fontsize=8)

    # n_eff distributions
    ax = axes[0, 1]
    for s_low, s_high, label, color in S_BINS:
        mask = (feats_df['s'] >= s_low) & (feats_df['s'] < s_high)
        subset = feats_df[mask]
        if len(subset) > 0:
            neff_vals = subset['n_eff'].values
            ell_vals = subset['ell'].values
            valid = np.isfinite(neff_vals) & np.isfinite(ell_vals) & (neff_vals < 100)
            if valid.sum() > 0:
                mean_neff = np.average(neff_vals[valid], weights=ell_vals[valid])
                ax.hist(neff_vals[valid], bins=50, alpha=0.5,
                       label=f'{label[:8]} (mean={mean_neff:.2f})',
                       color=color, density=True, weights=ell_vals[valid])
    ax.set_xlabel('Effective Support $n_{eff} = e^{H_{bin}}$')
    ax.set_ylabel('Leverage-weighted density')
    ax.set_title('Effective Number of Eigenvalue Bins')
    ax.legend(fontsize=8)

    # Run-level means
    if 'wmean_H_bin' in runs_df.columns:
        ax = axes[1, 0]
        scatter = ax.scatter(runs_df['s'], runs_df['wmean_H_bin'],
                            c=runs_df['m'], cmap='viridis', alpha=0.6, s=15)
        ax.set_xlabel('Sparsity (s)')
        ax.set_ylabel('Mean Binned Entropy')
        ax.set_title('Leverage-weighted Mean H_bin vs Sparsity')
        plt.colorbar(scatter, ax=ax, label='m')

    if 'wmean_n_eff' in runs_df.columns:
        ax = axes[1, 1]
        scatter = ax.scatter(runs_df['s'], runs_df['wmean_n_eff'],
                            c=runs_df['m'], cmap='viridis', alpha=0.6, s=15)
        ax.set_xlabel('Sparsity (s)')
        ax.set_ylabel('Mean Effective Support')
        ax.set_title('Leverage-weighted Mean n_eff vs Sparsity')
        plt.colorbar(scatter, ax=ax, label='m')

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_entropy_distribution.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved entropy distribution plot to {out_dir / 'plot_entropy_distribution.png'}")


def plot_tail_mass_curves(runs_df: pd.DataFrame, out_dir: Path):
    """
    Plot tail mass curves vs sparsity.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for col, label, color in [('tail_mass_001', r'$\tau=0.01$', 'blue'),
                               ('tail_mass_005', r'$\tau=0.05$', 'green'),
                               ('tail_mass_010', r'$\tau=0.10$', 'red')]:
        if col not in runs_df.columns:
            continue

        # Bin by sparsity and compute mean
        runs_df['s_bin_temp'] = pd.cut(runs_df['s'], bins=20)
        grouped = runs_df.groupby('s_bin_temp', observed=True)[col].agg(['mean', 'std'])
        grouped = grouped.dropna()

        if len(grouped) == 0:
            continue

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


def plot_defect_identity_check(runs_df: pd.DataFrame, out_dir: Path):
    """
    Diagnostic: Verify defect identity gap1 ≈ gap2.
    """
    if 'defect_error' not in runs_df.columns:
        print("Warning: defect_error not in run metrics, skipping defect check plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: defect_error histogram
    ax = axes[0]
    defect_errors = runs_df['defect_error'].dropna()
    ax.hist(defect_errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=1e-6, color='green', linestyle='--', alpha=0.7, label='1e-6 threshold')
    ax.axvline(x=1e-3, color='orange', linestyle='--', alpha=0.7, label='1e-3 threshold')
    ax.set_xlabel('Defect Error |gap1 - gap2|')
    ax.set_ylabel('Count')
    ax.set_title('Defect Identity Verification\n(gap1 = m - sumD, gap2 = sum(ell - D))')
    ax.set_xscale('log')
    ax.legend()

    # Panel 2: defect_error vs sparsity
    ax = axes[1]
    scatter = ax.scatter(runs_df['s'], runs_df['defect_error'],
                        c=runs_df['m'], cmap='viridis', alpha=0.6, s=15)
    ax.set_xlabel('Sparsity (s)')
    ax.set_ylabel('Defect Error')
    ax.set_title('Defect Error vs Sparsity')
    ax.set_yscale('log')
    plt.colorbar(scatter, ax=ax, label='m')

    plt.tight_layout()
    fig.savefig(out_dir / 'plot_defect_identity_check.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved defect identity check to {out_dir / 'plot_defect_identity_check.png'}")


def create_summary_report(runs_df: pd.DataFrame, feats_df: pd.DataFrame, out_dir: Path):
    """
    Create a summary report of key findings.
    """
    report_lines = [
        "=" * 70,
        "Capacity Localization Analysis v2 - Summary Report",
        "=" * 70,
        f"Generated: {datetime.now().isoformat()}",
        f"Number of runs: {len(runs_df)}",
        f"Number of features: {len(feats_df) if feats_df is not None else 'N/A'}",
        "",
        "-" * 70,
        "Key Metrics Summary",
        "-" * 70,
    ]

    # Check q_bin separation
    if 'wmean_q_bin' in runs_df.columns:
        report_lines.append("\n1. Binned Diracness Proxy (q_bin) by Sparsity:")
        for s_low, s_high, label, _ in S_BINS:
            subset = runs_df[(runs_df['s'] >= s_low) & (runs_df['s'] < s_high)]
            if len(subset) > 0:
                mean_q = subset['wmean_q_bin'].mean()
                report_lines.append(f"   {label}: mean q_bin = {mean_q:.4f}")

    # Check cv trend
    if 'mean_cv' in runs_df.columns:
        report_lines.append("\n2. Coefficient of Variation (cv) by Sparsity:")
        for s_low, s_high, label, _ in S_BINS:
            subset = runs_df[(runs_df['s'] >= s_low) & (runs_df['s'] < s_high)]
            if len(subset) > 0:
                mean_cv = subset['mean_cv'].mean()
                report_lines.append(f"   {label}: mean cv = {mean_cv:.4f}")

    # Check defect identity
    if 'defect_error' in runs_df.columns:
        max_err = runs_df['defect_error'].max()
        mean_err = runs_df['defect_error'].mean()
        report_lines.append(f"\n3. Defect Identity Check:")
        report_lines.append(f"   Mean error: {mean_err:.2e}")
        report_lines.append(f"   Max error:  {max_err:.2e}")
        report_lines.append(f"   Status: {'PASS' if max_err < 1e-3 else 'WARNING'}")

    # Check negative D
    if 'negD_frac' in runs_df.columns:
        max_neg = runs_df['negD_frac'].max()
        mean_neg = runs_df['negD_frac'].mean()
        report_lines.append(f"\n4. Negative D Check:")
        report_lines.append(f"   Mean negD_frac: {mean_neg:.4f}")
        report_lines.append(f"   Max negD_frac:  {max_neg:.4f}")
        if max_neg > 0:
            report_lines.append(f"   Status: WARNING - Use D_hat in plots")
        else:
            report_lines.append(f"   Status: PASS - No negative D values")

    # Check qualification rate
    if 'qual_rate' in runs_df.columns:
        report_lines.append(f"\n5. Qualification Rate (q_bin >= {LOCALIZATION_THRESHOLD}) by Sparsity:")
        for s_low, s_high, label, _ in S_BINS:
            subset = runs_df[(runs_df['s'] >= s_low) & (runs_df['s'] < s_high)]
            if len(subset) > 0:
                mean_qr = subset['qual_rate'].mean()
                report_lines.append(f"   {label}: mean qual_rate = {mean_qr:.4f}")

    report_lines.extend([
        "",
        "-" * 70,
        "Plots Generated",
        "-" * 70,
        "  - plot_A_rank_delocalization.png: Rank/saturation vs sparsity",
        "  - plot_cv_vs_sparsity.png: Coefficient of variation analysis",
        "  - plot_q_bin_distribution.png: Binned Diracness proxy distributions",
        "  - plot_qual_rate_vs_sparsity.png: Qualification rate trends",
        "  - plot_B_featurewise_localization.png: Sigma distributions",
        "  - plot_C_assignment_test.png: D_hat vs D_hat_pred for localized features",
        "  - plot_D_vs_D_hat_diagnostic.png: D sanity check",
        "  - plot_entropy_distribution.png: H_bin and n_eff distributions",
        "  - plot_tail_mass_curves.png: Delocalization tail mass",
        "  - plot_defect_identity_check.png: Gap verification",
        "",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(out_dir / "analysis_summary.txt", "w") as f:
        f.write(report_text)

    print(f"\nSaved summary report to {out_dir / 'analysis_summary.txt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization plots for Capacity Localization Analysis v2"
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing run_metrics.csv and feature_metrics.csv")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory for plots (default: same as results-dir)")

    args = parser.parse_args()

    out_dir = args.out or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 70)
    print("Capacity Localization Analysis v2 - Plot Generation")
    print(f"=" * 70)
    print(f"Loading results from {args.results_dir}")

    # Load metadata
    metadata = load_metadata(args.results_dir)
    if metadata:
        print(f"Analysis version: {metadata.get('version', 'unknown')}")
        print(f"Localization threshold: {metadata.get('localization_threshold', LOCALIZATION_THRESHOLD)}")

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

    print(f"\nGenerating plots to {out_dir}...")

    # Plot A: Rank-vs-delocalization decomposition
    plot_a_rank_delocalization(runs_df, out_dir)

    # CV vs sparsity (new in v2)
    plot_cv_vs_sparsity(runs_df, out_dir)

    # Qualification rate (new in v2)
    plot_qual_rate_vs_sparsity(runs_df, out_dir)

    # Tail mass curves
    plot_tail_mass_curves(runs_df, out_dir)

    # Defect identity check (new in v2)
    plot_defect_identity_check(runs_df, out_dir)

    # Plots requiring feature metrics
    if feats_df is not None:
        # Plot B: Featurewise localization distributions
        plot_b_featurewise_localization(feats_df, runs_df, out_dir)

        # q_bin distributions (replaces broken q plot)
        plot_q_bin_distributions(feats_df, runs_df, out_dir)

        # Plot C (fixed): Assignment test
        plot_c_assignment_test(feats_df, runs_df, out_dir)

        # D vs D_hat diagnostic
        plot_d_hat_comparison(feats_df, runs_df, out_dir)

        # Entropy distributions
        plot_H_bin_distribution(feats_df, runs_df, out_dir)

        # Create summary report
        create_summary_report(runs_df, feats_df, out_dir)
    else:
        print("\nNote: Feature metrics not found. Skipping feature-level plots.")
        print("Re-run analysis with --save-feature-metrics to generate these plots.")
        create_summary_report(runs_df, None, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
