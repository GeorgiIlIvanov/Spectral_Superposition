"""
Visualization of D_i vs ||W_i||^2 linear fit results.

Creates:
1. Heatmaps of slope and R² by (sparsity, compression ratio)
2. Histograms of slope distributions and accuracy by sparsity/hidden dim
3. Scatter plots from 10 percentile bins of fit accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import h5py
import json

# Set style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

def load_data():
    """Load the linear fit arrays."""
    data = np.load('analysis/linear_fit_arrays.npz')
    return {
        'slopes': data['slopes'],
        'r_squared': data['r_squared'],
        'residual_variance': data['residual_variance'],
        'm_hidden_vals': data['m_hidden_vals'],
        'sparsity_vals': data['sparsity_vals'],
        'target_steps': data['target_steps']
    }

def load_aggregated():
    """Load aggregated JSON results."""
    with open('analysis/linear_fit_aggregated.json', 'r') as f:
        return json.load(f)

# ============================================================================
# 1. HEATMAP PLOTS
# ============================================================================

def plot_heatmaps(data, step_idx, step_name):
    """Plot heatmaps of slope and R² for a given step."""
    n_features = 1024
    m_vals = data['m_hidden_vals']
    s_vals = data['sparsity_vals']

    # Compression ratio = m/n
    compression_ratios = m_vals / n_features

    slopes = data['slopes'][:, :, step_idx]
    r_squared = data['r_squared'][:, :, step_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Slope heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(slopes, aspect='auto', origin='lower',
                     extent=[s_vals.min(), s_vals.max(),
                            compression_ratios.min(), compression_ratios.max()],
                     cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    ax1.set_xlabel('Sparsity', fontsize=12)
    ax1.set_ylabel('Compression Ratio (m/n)', fontsize=12)
    ax1.set_title(f'Slope of D_i vs ||W_i||² (Step {step_name})', fontsize=13)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Slope', fontsize=11)

    # R² heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(r_squared, aspect='auto', origin='lower',
                     extent=[s_vals.min(), s_vals.max(),
                            compression_ratios.min(), compression_ratios.max()],
                     cmap='viridis', vmin=0, vmax=1)
    ax2.set_xlabel('Sparsity', fontsize=12)
    ax2.set_ylabel('Compression Ratio (m/n)', fontsize=12)
    ax2.set_title(f'R² of Linear Fit (Step {step_name})', fontsize=13)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('R²', fontsize=11)

    plt.tight_layout()
    return fig

# ============================================================================
# 2. HISTOGRAM PLOTS
# ============================================================================

def plot_slope_histograms(data):
    """Plot histograms of slope distributions for each step."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    steps = data['target_steps']

    for idx, (ax, step) in enumerate(zip(axes, steps)):
        slopes = data['slopes'][:, :, idx].flatten()
        slopes = slopes[np.isfinite(slopes)]

        ax.hist(slopes, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Slope = 1')
        ax.axvline(x=np.mean(slopes), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean = {np.mean(slopes):.3f}')
        ax.set_xlabel('Slope', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Slope Distribution (Step {step})', fontsize=13)
        ax.legend(fontsize=9)
        ax.set_xlim(-1.5, 2)

    plt.tight_layout()
    return fig

def plot_r2_histograms(data):
    """Plot histograms of R² distributions for each step."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    steps = data['target_steps']

    for idx, (ax, step) in enumerate(zip(axes, steps)):
        r2 = data['r_squared'][:, :, idx].flatten()
        r2 = r2[np.isfinite(r2)]

        ax.hist(r2, bins=50, edgecolor='black', alpha=0.7, color='forestgreen')
        ax.axvline(x=np.mean(r2), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean = {np.mean(r2):.4f}')
        ax.axvline(x=np.median(r2), color='red', linestyle='--', linewidth=2,
                   label=f'Median = {np.median(r2):.4f}')
        ax.set_xlabel('R²', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'R² Distribution (Step {step})', fontsize=13)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1.05)

    plt.tight_layout()
    return fig

def plot_metrics_by_sparsity(data):
    """Plot mean slope and R² vs sparsity (averaged over m)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    steps = data['target_steps']
    s_vals = data['sparsity_vals']

    for idx, step in enumerate(steps):
        slopes = data['slopes'][:, :, idx]  # (m, s)
        r2 = data['r_squared'][:, :, idx]

        # Mean and std over m dimension
        slope_mean = np.nanmean(slopes, axis=0)
        slope_std = np.nanstd(slopes, axis=0)
        r2_mean = np.nanmean(r2, axis=0)
        r2_std = np.nanstd(r2, axis=0)

        # Slope vs sparsity
        ax1 = axes[0, idx]
        ax1.fill_between(s_vals, slope_mean - slope_std, slope_mean + slope_std,
                         alpha=0.3, color='steelblue')
        ax1.plot(s_vals, slope_mean, 'o-', color='steelblue', markersize=3)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Sparsity', fontsize=11)
        ax1.set_ylabel('Slope', fontsize=11)
        ax1.set_title(f'Slope vs Sparsity (Step {step})', fontsize=12)
        ax1.set_ylim(-0.5, 1.5)

        # R² vs sparsity
        ax2 = axes[1, idx]
        ax2.fill_between(s_vals, r2_mean - r2_std, r2_mean + r2_std,
                         alpha=0.3, color='forestgreen')
        ax2.plot(s_vals, r2_mean, 'o-', color='forestgreen', markersize=3)
        ax2.set_xlabel('Sparsity', fontsize=11)
        ax2.set_ylabel('R²', fontsize=11)
        ax2.set_title(f'R² vs Sparsity (Step {step})', fontsize=12)
        ax2.set_ylim(0.5, 1.05)

    plt.tight_layout()
    return fig

def plot_metrics_by_hidden_dim(data):
    """Plot mean slope and R² vs hidden dimension (averaged over sparsity)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    steps = data['target_steps']
    m_vals = data['m_hidden_vals']
    n_features = 1024
    compression = m_vals / n_features

    for idx, step in enumerate(steps):
        slopes = data['slopes'][:, :, idx]  # (m, s)
        r2 = data['r_squared'][:, :, idx]

        # Mean and std over sparsity dimension
        slope_mean = np.nanmean(slopes, axis=1)
        slope_std = np.nanstd(slopes, axis=1)
        r2_mean = np.nanmean(r2, axis=1)
        r2_std = np.nanstd(r2, axis=1)

        # Slope vs compression ratio
        ax1 = axes[0, idx]
        ax1.fill_between(compression, slope_mean - slope_std, slope_mean + slope_std,
                         alpha=0.3, color='steelblue')
        ax1.plot(compression, slope_mean, 'o-', color='steelblue', markersize=4)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Compression Ratio (m/n)', fontsize=11)
        ax1.set_ylabel('Slope', fontsize=11)
        ax1.set_title(f'Slope vs Compression (Step {step})', fontsize=12)
        ax1.set_ylim(-0.5, 1.5)

        # R² vs compression ratio
        ax2 = axes[1, idx]
        ax2.fill_between(compression, r2_mean - r2_std, r2_mean + r2_std,
                         alpha=0.3, color='forestgreen')
        ax2.plot(compression, r2_mean, 'o-', color='forestgreen', markersize=4)
        ax2.set_xlabel('Compression Ratio (m/n)', fontsize=11)
        ax2.set_ylabel('R²', fontsize=11)
        ax2.set_title(f'R² vs Compression (Step {step})', fontsize=12)
        ax2.set_ylim(0.5, 1.05)

    plt.tight_layout()
    return fig

# ============================================================================
# 3. SCATTER PLOTS FROM PERCENTILE BINS
# ============================================================================

def get_checkpoint_data(m_hidden, sparsity, seed=0):
    """Load D_i and ||W_i||² from checkpoint for specific (m, s)."""
    # Find the file
    start_dir = Path('start')
    pattern = f'n1024_m{m_hidden}_s{sparsity:.16f}_seed{seed}.h5'

    # Find matching file (sparsity formatting may vary)
    matching_files = list(start_dir.glob(f'n1024_m{m_hidden}_s*_seed{seed}.h5'))

    # Find closest sparsity match
    best_file = None
    best_diff = float('inf')
    for f in matching_files:
        # Extract sparsity from filename
        parts = f.stem.split('_')
        file_sparsity = float(parts[2][1:])  # Remove 's' prefix
        diff = abs(file_sparsity - sparsity)
        if diff < best_diff:
            best_diff = diff
            best_file = f

    if best_file is None:
        return None

    with h5py.File(best_file, 'r') as f:
        checkpoint_steps = f['checkpoint_steps'][:]
        fractional_dims = f['fractional_dims'][:]
        feature_norms = f['feature_norms'][:]
        actual_sparsity = f.attrs['sparsity']

    return {
        'steps': checkpoint_steps,
        'D_i': fractional_dims,
        'W_norm_sq': feature_norms,
        'sparsity': actual_sparsity
    }

def plot_percentile_scatter_plots(data, aggregated):
    """Create 10 scatter plots, one from each decile of R² accuracy."""
    # Use step 20000 (final) for ranking
    step_idx = 2  # 20000
    r2_flat = data['r_squared'][:, :, step_idx]

    # Create list of (m_idx, s_idx, r2) for all combinations
    entries = []
    for mi, m in enumerate(data['m_hidden_vals']):
        for si, s in enumerate(data['sparsity_vals']):
            r2_val = r2_flat[mi, si]
            if np.isfinite(r2_val):
                entries.append({
                    'm_hidden': int(m),
                    'sparsity': float(s),
                    'm_idx': mi,
                    's_idx': si,
                    'r2': r2_val
                })

    # Sort by R²
    entries.sort(key=lambda x: x['r2'])
    n_entries = len(entries)

    # Target steps for plotting
    target_steps = [400, 7000, 20000]
    colors = ['#e74c3c', '#f39c12', '#27ae60']  # red, orange, green

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    np.random.seed(42)  # reproducibility

    for i in range(10):
        ax = axes[i]

        # Get percentile range
        lower_pct = i * 10
        upper_pct = (i + 1) * 10
        lower_idx = int(n_entries * lower_pct / 100)
        upper_idx = int(n_entries * upper_pct / 100)

        # Sample from this percentile bin
        bin_entries = entries[lower_idx:upper_idx]
        if len(bin_entries) == 0:
            continue

        sampled = np.random.choice(len(bin_entries))
        entry = bin_entries[sampled]

        # Load checkpoint data
        ckpt_data = get_checkpoint_data(entry['m_hidden'], entry['sparsity'])
        if ckpt_data is None:
            ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', transform=ax.transAxes)
            continue

        # Plot for each target step
        for step, color in zip(target_steps, colors):
            step_idx_local = np.argmin(np.abs(ckpt_data['steps'] - step))
            actual_step = ckpt_data['steps'][step_idx_local]

            D_i = ckpt_data['D_i'][step_idx_local]
            W_norm_sq = ckpt_data['W_norm_sq'][step_idx_local]

            # Filter valid points
            mask = np.isfinite(D_i) & np.isfinite(W_norm_sq)
            x = W_norm_sq[mask]
            y = D_i[mask]

            ax.scatter(x, y, alpha=0.3, s=8, color=color, label=f'Step {actual_step}')

            # Add fit line
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                x_line = np.array([x.min(), x.max()])
                ax.plot(x_line, slope * x_line, '--', color=color, linewidth=1.5)

        ax.set_xlabel('||W_i||²', fontsize=10)
        ax.set_ylabel('D_i', fontsize=10)
        ax.set_title(f'{lower_pct}-{upper_pct}th pctl\nm={entry["m_hidden"]}, s={entry["sparsity"]:.3f}\nR²={entry["r2"]:.4f}',
                     fontsize=9)
        if i == 0:
            ax.legend(fontsize=7, loc='upper left')

    plt.suptitle('D_i vs ||W_i||² Scatter Plots by R² Percentile', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading data...")
    data = load_data()
    aggregated = load_aggregated()

    output_dir = Path('analysis/plots')
    output_dir.mkdir(exist_ok=True)

    steps = data['target_steps']

    # 1. Heatmaps for each step
    print("Creating heatmaps...")
    for idx, step in enumerate(steps):
        fig = plot_heatmaps(data, idx, step)
        fig.savefig(output_dir / f'heatmap_step_{step}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Combined heatmap figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    n_features = 1024
    m_vals = data['m_hidden_vals']
    s_vals = data['sparsity_vals']
    compression_ratios = m_vals / n_features

    for idx, step in enumerate(steps):
        slopes = data['slopes'][:, :, idx]
        r2 = data['r_squared'][:, :, idx]

        # Slope
        im1 = axes[0, idx].imshow(slopes, aspect='auto', origin='lower',
                                   extent=[s_vals.min(), s_vals.max(),
                                          compression_ratios.min(), compression_ratios.max()],
                                   cmap='RdBu_r', vmin=-1.5, vmax=1.5)
        axes[0, idx].set_xlabel('Sparsity')
        axes[0, idx].set_ylabel('m/n')
        axes[0, idx].set_title(f'Slope (Step {step})')
        plt.colorbar(im1, ax=axes[0, idx])

        # R²
        im2 = axes[1, idx].imshow(r2, aspect='auto', origin='lower',
                                   extent=[s_vals.min(), s_vals.max(),
                                          compression_ratios.min(), compression_ratios.max()],
                                   cmap='viridis', vmin=0, vmax=1)
        axes[1, idx].set_xlabel('Sparsity')
        axes[1, idx].set_ylabel('m/n')
        axes[1, idx].set_title(f'R² (Step {step})')
        plt.colorbar(im2, ax=axes[1, idx])

    plt.suptitle('Linear Fit: D_i vs ||W_i||² Across Training', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / 'heatmaps_combined.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. Histograms
    print("Creating histograms...")
    fig = plot_slope_histograms(data)
    fig.savefig(output_dir / 'histogram_slopes.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = plot_r2_histograms(data)
    fig.savefig(output_dir / 'histogram_r2.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. Metrics by sparsity and hidden dim
    print("Creating metrics plots...")
    fig = plot_metrics_by_sparsity(data)
    fig.savefig(output_dir / 'metrics_by_sparsity.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = plot_metrics_by_hidden_dim(data)
    fig.savefig(output_dir / 'metrics_by_hidden_dim.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 4. Percentile scatter plots
    print("Creating percentile scatter plots...")
    fig = plot_percentile_scatter_plots(data, aggregated)
    fig.savefig(output_dir / 'scatter_percentiles.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll plots saved to {output_dir}/")
    print("Files created:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")

if __name__ == '__main__':
    main()
