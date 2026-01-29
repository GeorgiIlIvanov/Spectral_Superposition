#!/usr/bin/env python3
"""
Quick script to regenerate just the fit quality vs localization plot.
Uses the already-computed eigenspace_cluster_data.csv.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# Load pre-computed data
df = pd.read_csv(OUTPUT_DIR / 'eigenspace_cluster_data.csv')

localization = df['mean_max_space_projection'].values
r2 = df['r_squared'].values
sparsity = df['sparsity'].values
slope_times_lambda = df['slope_times_lambda'].values
kappa_lambda_error = np.abs(slope_times_lambda - 1.0)  # |κλ - 1|

# Create standalone figure
fig, ax = plt.subplots(figsize=(12, 9))

# Scatter plot colored by sparsity (purple → blue → green → yellow)
sc = ax.scatter(localization, r2, c=sparsity, cmap='viridis',
                alpha=0.5, s=20, edgecolors='none', vmin=0, vmax=1)

# Colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Sparsity', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# Equal-width localization bins, cycling through sparsity
bin_width = 0.04
n_sparsity_bins = 5
sparsity_edges = np.linspace(0, 1, n_sparsity_bins + 1)

# Colormap for sparsity bins
cmap = cm.viridis
norm = Normalize(vmin=0, vmax=1)

# Create localization bins from 0 to max with equal width
loc_max = localization.max()
loc_bins = np.arange(0, loc_max + bin_width, bin_width)

# Offset for each sparsity bin within a localization bin
jitter_total = bin_width * 0.8
jitter_offsets = np.linspace(-jitter_total/2, jitter_total/2, n_sparsity_bins)

# Collect and plot all valid (loc_bin, sparsity_bin) combinations
for i, l_lo in enumerate(loc_bins[:-1]):
    l_hi = l_lo + bin_width
    l_center = (l_lo + l_hi) / 2

    for j in range(n_sparsity_bins):
        s_lo, s_hi = sparsity_edges[j], sparsity_edges[j+1]
        s_center = (s_lo + s_hi) / 2

        mask = (localization >= l_lo) & (localization < l_hi) & \
               (sparsity >= s_lo) & (sparsity < s_hi)

        if mask.sum() >= 5:
            r2_mean = np.mean(r2[mask])
            kl_err = np.mean(kappa_lambda_error[mask])

            # x position: localization center + jitter based on sparsity bin
            x_pos = l_center + jitter_offsets[j]
            color = cmap(norm(s_center))

            ax.errorbar(x_pos, r2_mean, yerr=kl_err,
                       fmt='o', color=color, ecolor=color,
                       capsize=2, capthick=1.2, linewidth=1.2,
                       markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                       zorder=10, alpha=0.9)

# Add a proxy artist for legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=14, markeredgecolor='black', markeredgewidth=1.5, label='Bin center'),
    Line2D([0], [0], color='gray', linewidth=3, label=r'Error = $|\kappa\lambda - 1|$')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=16, framealpha=0.95,
          markerscale=1.2, handlelength=2, borderpad=1)

# Labels
ax.set_xlabel('Mean Max Eigenspace Projection (Localization)', fontsize=14, fontweight='bold')
ax.set_ylabel('Cluster R² (Fit Quality)', fontsize=14, fontweight='bold')
ax.set_title(r'Fit Quality vs Eigenspace Localization' + '\n' +
             r'Error bars show mean $|\kappa\lambda - 1|$, colored by sparsity bin',
             fontsize=16, fontweight='bold')
ax.tick_params(axis='both', labelsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, localization.max() * 1.05)
ax.set_ylim(r2.min() * 0.95, 1.02)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fit_quality_vs_localization.png', dpi=200, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fit_quality_vs_localization.pdf', dpi=200, bbox_inches='tight')
print("Saved: fit_quality_vs_localization.png/pdf")
plt.close()
