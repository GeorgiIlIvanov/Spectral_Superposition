#!/usr/bin/env python3
"""
Analysis 7: Phase Diagram Animation & Snapshots

Create visualizations of the phase diagram (||W_i||², D_i) evolution over training.
Includes animated GIFs and static comparison plots.
"""

import h5py
import numpy as np
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import json
import argparse


INPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/temporal_analysis')
PLOTS_DIR = OUTPUT_DIR / 'plots'
RESULTS_DIR = OUTPUT_DIR / 'results'

R2_THRESHOLD = 0.9


def compute_feature_r2(feature_norms, fractional_dims):
    """Compute R² for each feature trajectory."""
    n_features = feature_norms.shape[1]
    r2_values = np.zeros(n_features)

    for i in range(n_features):
        x = feature_norms[:, i]
        y = fractional_dims[:, i]
        valid = np.isfinite(x) & np.isfinite(y) & (x > 1e-8)
        if np.sum(valid) < 5:
            r2_values[i] = np.nan
            continue
        _, _, r_val, _, _ = stats.linregress(x[valid], y[valid])
        r2_values[i] = r_val ** 2

    return r2_values


def create_phase_snapshot(norms, dims, r2_values, checkpoint_step, ax, title_suffix=''):
    """Create a single phase diagram snapshot."""
    ax.clear()

    # Color by R² (dark matter vs well-behaved)
    is_dark_matter = r2_values < R2_THRESHOLD
    is_wellbehaved = r2_values >= R2_THRESHOLD

    # Plot dark matter in red, well-behaved in green
    ax.scatter(norms[is_dark_matter], dims[is_dark_matter],
              c='red', alpha=0.3, s=3, label='Dark Matter (R²<0.9)', rasterized=True)
    ax.scatter(norms[is_wellbehaved], dims[is_wellbehaved],
              c='green', alpha=0.3, s=3, label='Well-behaved (R²≥0.9)', rasterized=True)

    # Reference lines
    ax.plot([0, 1.5], [0, 1.5], 'k--', alpha=0.3, label='D = ||W||²')
    ax.plot([0, 1.5], [0, 0.75], 'b--', alpha=0.3, label='D = ||W||²/2')

    ax.set_xlabel('||W_i||²', fontsize=12)
    ax.set_ylabel('D_i', fontsize=12)
    ax.set_title(f'Phase Diagram - Step {checkpoint_step}{title_suffix}', fontsize=14)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)


def create_animation(filepath, output_path, n_frames=30):
    """Create animated GIF of phase diagram evolution."""

    with h5py.File(filepath, 'r') as f:
        feature_norms = f['feature_norms'][:]
        fractional_dims = f['fractional_dims'][:]
        checkpoint_steps = f['checkpoint_steps'][:]
        sparsity = float(f.attrs['sparsity'])
        m_hidden = int(f.attrs['m_hidden'])

    # Compute R² (using full trajectory)
    r2_values = compute_feature_r2(feature_norms, fractional_dims)

    n_checkpoints = feature_norms.shape[0]

    # Select frames (subsample if too many)
    if n_checkpoints > n_frames:
        frame_indices = np.linspace(0, n_checkpoints - 1, n_frames, dtype=int)
    else:
        frame_indices = np.arange(n_checkpoints)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

    def update(frame_idx):
        t = frame_indices[frame_idx]
        create_phase_snapshot(
            feature_norms[t], fractional_dims[t], r2_values,
            checkpoint_steps[t], ax,
            title_suffix=f' (S={sparsity:.2f}, m={m_hidden})'
        )
        return [ax]

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=200, blit=False)

    writer = PillowWriter(fps=5)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"Saved animation: {output_path}")


def create_static_comparison(results_dir, output_dir):
    """Create static comparison of phase diagrams at early, mid, late training."""

    # Select a few representative experiments
    files = sorted(INPUT_DIR.glob('n1024_m*_s0.9*_seed0.h5'))[:3]  # High sparsity examples

    if not files:
        files = sorted(INPUT_DIR.glob('n1024_m*.h5'))[:3]

    fig, axes = plt.subplots(len(files), 3, figsize=(15, 4 * len(files)))

    for row, filepath in enumerate(files):
        with h5py.File(filepath, 'r') as f:
            feature_norms = f['feature_norms'][:]
            fractional_dims = f['fractional_dims'][:]
            checkpoint_steps = f['checkpoint_steps'][:]
            sparsity = float(f.attrs['sparsity'])
            m_hidden = int(f.attrs['m_hidden'])

        r2_values = compute_feature_r2(feature_norms, fractional_dims)

        # Early, mid, final
        checkpoints = [10, len(checkpoint_steps) // 2, -1]
        titles = ['Early Training', 'Mid Training', 'Final']

        for col, (ckpt_idx, title) in enumerate(zip(checkpoints, titles)):
            if len(files) > 1:
                ax = axes[row, col]
            else:
                ax = axes[col]

            norms = feature_norms[ckpt_idx]
            dims = fractional_dims[ckpt_idx]

            is_dm = r2_values < R2_THRESHOLD
            is_wb = r2_values >= R2_THRESHOLD

            ax.scatter(norms[is_dm], dims[is_dm], c='red', alpha=0.3, s=3, rasterized=True)
            ax.scatter(norms[is_wb], dims[is_wb], c='green', alpha=0.3, s=3, rasterized=True)

            ax.plot([0, 1.5], [0, 1.5], 'k--', alpha=0.3)
            ax.set_xlim(0, 1.5)
            ax.set_ylim(0, 1.1)
            ax.set_xlabel('||W_i||²', fontsize=10)
            ax.set_ylabel('D_i', fontsize=10)
            ax.set_title(f'{title}\nS={sparsity:.2f}, m={m_hidden}, step={checkpoint_steps[ckpt_idx]}', fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Phase Diagram Evolution (Red=Dark Matter, Green=Well-behaved)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'phase_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'phase_comparison.png'}")
    plt.close()


def create_tracer_plot(filepath, output_dir, n_tracers=20):
    """Create plot showing individual feature trajectories through phase space."""

    with h5py.File(filepath, 'r') as f:
        feature_norms = f['feature_norms'][:]
        fractional_dims = f['fractional_dims'][:]
        checkpoint_steps = f['checkpoint_steps'][:]
        sparsity = float(f.attrs['sparsity'])
        m_hidden = int(f.attrs['m_hidden'])

    n_features = feature_norms.shape[1]
    r2_values = compute_feature_r2(feature_norms, fractional_dims)

    # Select tracer features (mix of dark matter and well-behaved)
    valid_r2 = np.where(np.isfinite(r2_values))[0]
    dm_indices = valid_r2[r2_values[valid_r2] < R2_THRESHOLD]
    wb_indices = valid_r2[r2_values[valid_r2] >= R2_THRESHOLD]

    n_dm = min(n_tracers // 2, len(dm_indices))
    n_wb = min(n_tracers // 2, len(wb_indices))

    tracer_dm = np.random.choice(dm_indices, n_dm, replace=False) if n_dm > 0 else []
    tracer_wb = np.random.choice(wb_indices, n_wb, replace=False) if n_wb > 0 else []

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot background scatter at final checkpoint
    ax.scatter(feature_norms[-1], fractional_dims[-1], c='gray', alpha=0.1, s=1, rasterized=True)

    # Plot tracer trajectories
    for idx in tracer_dm:
        norms = feature_norms[:, idx]
        dims = fractional_dims[:, idx]
        ax.plot(norms, dims, 'r-', alpha=0.5, linewidth=1)
        ax.scatter(norms[0], dims[0], c='red', marker='o', s=20, zorder=5)
        ax.scatter(norms[-1], dims[-1], c='red', marker='s', s=20, zorder=5)

    for idx in tracer_wb:
        norms = feature_norms[:, idx]
        dims = fractional_dims[:, idx]
        ax.plot(norms, dims, 'g-', alpha=0.5, linewidth=1)
        ax.scatter(norms[0], dims[0], c='green', marker='o', s=20, zorder=5)
        ax.scatter(norms[-1], dims[-1], c='green', marker='s', s=20, zorder=5)

    ax.plot([0, 1.5], [0, 1.5], 'k--', alpha=0.3, label='D = ||W||²')
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('||W_i||²', fontsize=12)
    ax.set_ylabel('D_i', fontsize=12)
    ax.set_title(f'Feature Trajectories (S={sparsity:.2f}, m={m_hidden})\n○=start, □=end, Red=DM, Green=WB', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'phase_tracers.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'phase_tracers.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Phase Animation Analysis')
    parser.add_argument('--skip-animation', action='store_true', help='Skip GIF generation (faster)')
    parser.add_argument('--sample', type=int, default=None, help='Process only N files')
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Analysis 7: Phase Diagram Animation & Snapshots")
    print("=" * 60)

    # Create static comparison
    print("\n--- Creating Static Phase Comparison ---")
    create_static_comparison(INPUT_DIR, PLOTS_DIR)

    # Create tracer plot (select one high-sparsity file)
    print("\n--- Creating Tracer Plot ---")
    high_sparsity_files = sorted(INPUT_DIR.glob('n1024_m*_s0.9*_seed0.h5'))
    if not high_sparsity_files:
        high_sparsity_files = sorted(INPUT_DIR.glob('n1024_m*_s0.8*_seed0.h5'))
    if not high_sparsity_files:
        high_sparsity_files = sorted(INPUT_DIR.glob('n1024_m*.h5'))

    if high_sparsity_files:
        create_tracer_plot(high_sparsity_files[0], PLOTS_DIR)

    # Create animation (optional, slow)
    if not args.skip_animation and high_sparsity_files:
        print("\n--- Creating Animation (this may take a while) ---")
        create_animation(high_sparsity_files[0], PLOTS_DIR / 'phase_evolution.gif', n_frames=30)

    print("\n" + "=" * 60)
    print("Analysis 7 Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
