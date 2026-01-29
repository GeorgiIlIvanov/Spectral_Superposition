#!/usr/bin/env python3
"""
Create a standalone plot showing the phase transition at s ≈ 0.28-0.30.

This plot illustrates the critical sparsity threshold where the system
transitions from localized to delocalized behavior.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'analysis_results_v3'

# Load v3 run metrics
runs_df = pd.read_csv(OUTPUT_DIR / 'run_metrics.csv')

# Bin by sparsity (finer bins around transition)
sparsity_bins = np.concatenate([
    np.arange(0, 0.35, 0.02),  # Fine bins around transition
    np.arange(0.35, 1.01, 0.05)  # Coarser bins elsewhere
])

runs_df['s_bin'] = pd.cut(runs_df['s'], bins=sparsity_bins)
binned = runs_df.groupby('s_bin', observed=True).agg({
    'mean_sigma': ['mean', 'std'],
    'mean_cv': ['mean', 'std'],
    'wmean_max_space_proj': ['mean', 'std'],
    's': 'mean'
}).reset_index()

# Flatten column names
binned.columns = ['s_bin', 'sigma_mean', 'sigma_std', 'cv_mean', 'cv_std',
                  'max_proj_mean', 'max_proj_std', 's_center']

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Mean Sigma (Delocalization Slack)
ax = axes[0]
ax.errorbar(binned['s_center'], binned['sigma_mean'], yerr=binned['sigma_std'],
            fmt='o-', capsize=3, capthick=1.5, linewidth=2, markersize=6,
            color='steelblue', ecolor='steelblue', alpha=0.8)
ax.axvline(0.28, color='red', linestyle='--', linewidth=2, alpha=0.7, label='s = 0.28')
ax.axvline(0.30, color='red', linestyle=':', linewidth=2, alpha=0.7, label='s = 0.30')
ax.axvspan(0.28, 0.30, alpha=0.15, color='red', label='Critical region')
ax.set_xlabel('Sparsity (s)', fontsize=13, fontweight='bold')
ax.set_ylabel('Mean Delocalization Slack (σ)', fontsize=13, fontweight='bold')
ax.set_title('Delocalization Slack vs Sparsity', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-0.02, 1.02)

# Panel 2: Mean CV (Spectral Spread)
ax = axes[1]
ax.errorbar(binned['s_center'], binned['cv_mean'], yerr=binned['cv_std'],
            fmt='s-', capsize=3, capthick=1.5, linewidth=2, markersize=6,
            color='darkorange', ecolor='darkorange', alpha=0.8)
ax.axvline(0.28, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(0.30, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax.axvspan(0.28, 0.30, alpha=0.15, color='red')
ax.set_xlabel('Sparsity (s)', fontsize=13, fontweight='bold')
ax.set_ylabel('Mean CV (resid/kappa)', fontsize=13, fontweight='bold')
ax.set_title('Spectral Spread vs Sparsity', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-0.02, 1.02)

# Panel 3: Max Eigenspace Projection (Localization)
ax = axes[2]
ax.errorbar(binned['s_center'], binned['max_proj_mean'], yerr=binned['max_proj_std'],
            fmt='d-', capsize=3, capthick=1.5, linewidth=2, markersize=6,
            color='forestgreen', ecolor='forestgreen', alpha=0.8)
ax.axvline(0.28, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(0.30, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax.axvspan(0.28, 0.30, alpha=0.15, color='red')
ax.set_xlabel('Sparsity (s)', fontsize=13, fontweight='bold')
ax.set_ylabel('Mean Max Eigenspace Projection', fontsize=13, fontweight='bold')
ax.set_title('Eigenspace Localization vs Sparsity', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-0.02, 1.02)

plt.suptitle('Phase Transition at Critical Sparsity s ≈ 0.28-0.30',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plot_phase_transition.png', dpi=200, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'plot_phase_transition.pdf', bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'plot_phase_transition.png'}")
plt.close()

# Also create a single focused plot
fig, ax = plt.subplots(figsize=(10, 7))

# Normalize metrics to [0, 1] for comparison
sigma_norm = (binned['sigma_mean'] - binned['sigma_mean'].min()) / (binned['sigma_mean'].max() - binned['sigma_mean'].min())
cv_norm = (binned['cv_mean'] - binned['cv_mean'].min()) / (binned['cv_mean'].max() - binned['cv_mean'].min())
proj_norm = (binned['max_proj_mean'] - binned['max_proj_mean'].min()) / (binned['max_proj_mean'].max() - binned['max_proj_mean'].min())

ax.plot(binned['s_center'], sigma_norm, 'o-', linewidth=2.5, markersize=7,
        label='Delocalization Slack (σ)', color='steelblue')
ax.plot(binned['s_center'], cv_norm, 's-', linewidth=2.5, markersize=7,
        label='Spectral Spread (CV)', color='darkorange')
ax.plot(binned['s_center'], 1 - proj_norm, 'd-', linewidth=2.5, markersize=7,
        label='Eigenspace Delocalization', color='forestgreen')

ax.axvline(0.28, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
ax.axvline(0.30, color='red', linestyle=':', linewidth=2.5, alpha=0.8)
ax.axvspan(0.28, 0.30, alpha=0.2, color='red', label='Critical region (0.28-0.30)')

ax.set_xlabel('Sparsity (s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Normalized Metric (0-1)', fontsize=14, fontweight='bold')
ax.set_title('Phase Transition at Critical Sparsity\nAll Metrics Show Sharp Transition at s ≈ 0.28-0.30',
             fontsize=15, fontweight='bold')
ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax.tick_params(axis='both', labelsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plot_phase_transition_combined.png', dpi=200, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'plot_phase_transition_combined.pdf', bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'plot_phase_transition_combined.png'}")
plt.close()
