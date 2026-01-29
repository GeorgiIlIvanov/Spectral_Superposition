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

# Create standalone figure
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot colored by sparsity (purple → blue → green → yellow)
sc = ax.scatter(localization, r2, c=sparsity, cmap='viridis',
                alpha=0.7, s=25, edgecolors='none', vmin=0, vmax=1)

# Colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Sparsity', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# Binned means
loc_centers, r2_means, r2_stds = [], [], []
for i in range(10):
    lo = np.percentile(localization, i*10)
    hi = np.percentile(localization, (i+1)*10)
    m = (localization >= lo) & (localization < hi)
    if m.sum() > 10:
        loc_centers.append((lo + hi) / 2)
        r2_means.append(np.mean(r2[m]))
        r2_stds.append(np.std(r2[m]))

ax.errorbar(loc_centers, r2_means, yerr=r2_stds, fmt='ko-',
            capsize=4, capthick=2, linewidth=2.5, markersize=10,
            label='Binned mean ± std', zorder=10)

# Labels
ax.set_xlabel('Mean Max Eigenspace Projection (Localization)', fontsize=14, fontweight='bold')
ax.set_ylabel('Cluster R² (Fit Quality)', fontsize=14, fontweight='bold')
ax.set_title('Fit Quality vs Eigenspace Localization\nColored by Sparsity', fontsize=16, fontweight='bold')
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
