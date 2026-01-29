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
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot colored by sparsity (purple → blue → green → yellow)
sc = ax.scatter(localization, r2, c=sparsity, cmap='viridis',
                alpha=0.7, s=25, edgecolors='none', vmin=0, vmax=1)

# Colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Sparsity', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# Binned |κλ - 1| error (shown as error bars on R² means)
loc_centers, r2_means, kl_error_means, kl_error_stds = [], [], [], []
for i in range(10):
    lo = np.percentile(localization, i*10)
    hi = np.percentile(localization, (i+1)*10)
    m = (localization >= lo) & (localization < hi)
    if m.sum() > 10:
        loc_centers.append((lo + hi) / 2)
        r2_means.append(np.mean(r2[m]))
        kl_error_means.append(np.mean(kappa_lambda_error[m]))
        kl_error_stds.append(np.std(kappa_lambda_error[m]))

# Plot binned means with |κλ - 1| as error bars
# Scale error bars to be visible on R² scale (multiply by a factor)
kl_error_scaled = np.array(kl_error_means)  # Already in ~0.01-0.1 range

ax.errorbar(loc_centers, r2_means, yerr=kl_error_scaled, fmt='ko-',
            capsize=4, capthick=2, linewidth=2.5, markersize=10,
            label=r'Binned mean R², error = $|\kappa\lambda - 1|$', zorder=10)

# Labels
ax.set_xlabel('Mean Max Eigenspace Projection (Localization)', fontsize=14, fontweight='bold')
ax.set_ylabel('Cluster R² (Fit Quality)', fontsize=14, fontweight='bold')
ax.set_title(r'Fit Quality vs Eigenspace Localization' + '\n' + r'Error bars show mean $|\kappa\lambda - 1|$',
             fontsize=16, fontweight='bold')
ax.legend(loc='lower left', fontsize=12, framealpha=0.9)
ax.tick_params(axis='both', labelsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, localization.max() * 1.05)
ax.set_ylim(r2.min() * 0.95, 1.02)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fit_quality_vs_localization.png', dpi=200, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'fit_quality_vs_localization.pdf', dpi=200, bbox_inches='tight')
print("Saved: fit_quality_vs_localization.png/pdf")
plt.close()
