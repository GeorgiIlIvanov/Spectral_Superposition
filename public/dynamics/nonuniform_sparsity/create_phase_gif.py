#!/usr/bin/env python3
"""
Create animated GIF of spectral phase diagram for non-uniform sparsity experiment.

Features are colored by their sparsity S_i = i/1024.
Tracers highlight specific features across the sparsity gradient.

Color gradient (warm): #FF4E50 (S=0) → #FC913A → #F9D423 (S=1)
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pathlib import Path
import imageio.v2 as imageio
import os
from tqdm import tqdm
import argparse

DATA_DIR = Path('results')
OUTPUT_DIR = Path('.')

# Warm colormap for sparsity: #FF4E50 → #FC913A → #F9D423
def create_warm_cmap():
    colors = ['#FF4E50', '#FC913A', '#F9D423']
    return LinearSegmentedColormap.from_list('warm_sparsity', colors, N=256)

WARM_CMAP = create_warm_cmap()


def load_all_data(data_dir, sample_seeds=None):
    """Load data from all seed files."""
    files = sorted(data_dir.glob('*.h5'))
    if sample_seeds is not None:
        files = files[:sample_seeds]

    print(f"Loading {len(files)} experiments...")

    all_data = []
    for f in tqdm(files, desc='Loading'):
        with h5py.File(f, 'r') as hf:
            all_data.append({
                'seed': hf.attrs['seed'],
                'steps': hf['checkpoint_steps'][:],
                'norms': hf['feature_norms'][:],
                'dims': hf['fractional_dims'][:],
                'sparsity': hf['sparsity_per_feature'][:],
            })

    return all_data


def select_tracer_features(n_features=1024, n_tracers_per_bin=3, n_bins=10):
    """
    Select tracer feature indices across sparsity bins.
    Returns list of (feature_idx, sparsity, color) tuples.
    """
    tracers = []
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        s_low, s_high = bin_edges[i], bin_edges[i + 1]
        # Find features in this sparsity bin
        # S_i = i/n, so feature index for sparsity s is approximately s * n
        idx_low = int(s_low * n_features)
        idx_high = int(s_high * n_features)

        if idx_high > idx_low:
            # Sample features from this bin
            indices = np.linspace(idx_low, idx_high - 1, n_tracers_per_bin, dtype=int)
            for idx in indices:
                sparsity = idx / n_features
                color = WARM_CMAP(sparsity)
                tracers.append({
                    'feature_idx': idx,
                    'sparsity': sparsity,
                    'color': color,
                    'bin_idx': i,
                })

    return tracers


def create_phase_gif(data_dir, output_dir, sample_seeds=64, n_tracers_per_bin=3,
                     trail_length=10):
    """Create phase diagram GIF with sparsity coloring and tracers."""

    all_data = load_all_data(data_dir, sample_seeds)

    n_checkpoints = len(all_data[0]['steps'])
    steps = all_data[0]['steps']
    n_features = len(all_data[0]['sparsity'])
    sparsity_per_feature = all_data[0]['sparsity']  # Same for all seeds

    print(f"\nCheckpoints: {n_checkpoints}")
    print(f"Features: {n_features}")
    print(f"Seeds loaded: {len(all_data)}")

    # Select tracer features
    tracers = select_tracer_features(n_features, n_tracers_per_bin)
    print(f"Tracers: {len(tracers)} features across sparsity bins")

    # Pre-compute tracer trajectories (average across seeds)
    tracer_trajectories = []
    for t in tracers:
        idx = t['feature_idx']
        # Average norm and dim across all seeds for this feature
        norms_across_seeds = np.array([d['norms'][:, idx] for d in all_data])
        dims_across_seeds = np.array([d['dims'][:, idx] for d in all_data])

        tracer_trajectories.append({
            'feature_idx': idx,
            'sparsity': t['sparsity'],
            'color': t['color'],
            'norms_mean': norms_across_seeds.mean(axis=0),
            'norms_std': norms_across_seeds.std(axis=0),
            'dims_mean': dims_across_seeds.mean(axis=0),
            'dims_std': dims_across_seeds.std(axis=0),
        })

    # Create frames
    temp_dir = output_dir / 'gif_frames_nonuniform'
    temp_dir.mkdir(exist_ok=True)

    frames = []

    # Find global axis limits
    all_final_norms = np.concatenate([d['norms'][-1] for d in all_data])
    x_max = np.percentile(all_final_norms, 99) * 1.1

    print(f"\nGenerating {n_checkpoints} frames...")

    for cp in tqdm(range(n_checkpoints), desc='Creating frames'):
        fig, ax = plt.subplots(figsize=(14, 10))

        # Aggregate all features from all seeds at this checkpoint
        X_all, Y_all, C_all = [], [], []
        for d in all_data:
            X_all.append(d['norms'][cp])
            Y_all.append(d['dims'][cp])
            C_all.append(sparsity_per_feature)  # Color by sparsity

        X = np.concatenate(X_all)
        Y = np.concatenate(Y_all)
        C = np.concatenate(C_all)

        # Subsample for plotting if too many points
        if len(X) > 100000:
            idx = np.random.choice(len(X), 100000, replace=False)
            X, Y, C = X[idx], Y[idx], C[idx]

        # Main scatter plot colored by sparsity
        sc = ax.scatter(X, Y, c=C, cmap=WARM_CMAP, s=3, alpha=0.3,
                       rasterized=True, vmin=0, vmax=1)

        # Reference lines (D = ||W||^2 / μ)
        x_ref = np.linspace(0, x_max, 100)
        for mu in [1, 2, 3, 4, 5, 6, 8]:
            ax.plot(x_ref, x_ref / mu, '--', alpha=0.4, linewidth=1.5,
                   color='gray', label=f'μ={mu}' if mu <= 3 else '')

        # Draw tracer trajectories with trails
        for traj in tracer_trajectories:
            if cp > 0:
                trail_start = max(0, cp - trail_length)
                trail_x = traj['norms_mean'][trail_start:cp+1]
                trail_y = traj['dims_mean'][trail_start:cp+1]

                # Draw trail with fading alpha
                for i in range(len(trail_x) - 1):
                    alpha = 0.2 + 0.6 * (i / max(1, len(trail_x) - 1))
                    ax.plot(trail_x[i:i+2], trail_y[i:i+2],
                           color=traj['color'], alpha=alpha, linewidth=2.5)

            # Current position - larger circle with white stroke
            current_x = traj['norms_mean'][cp]
            current_y = traj['dims_mean'][cp]
            ax.scatter(current_x, current_y, c=[traj['color']], s=200,
                      edgecolors='white', linewidths=2.5, zorder=10, marker='o')

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(r'Feature Norm $\|W_i\|^2$', fontsize=14)
        ax.set_ylabel(r'Fractional Dimensionality $D_i$', fontsize=14)
        ax.set_title(f'Non-Uniform Sparsity Phase Diagram\n'
                    f'$S_i = i/n$ | m=256, n=1024 | Step {steps[cp]:,}',
                    fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)

        # Colorbar for sparsity
        cbar = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label('Feature Sparsity $S_i = i/n$', fontsize=12)

        # Legend for tracer sparsity bins
        legend_sparsities = [0.05, 0.25, 0.5, 0.75, 0.95]
        legend_elements = []
        for s in legend_sparsities:
            color = WARM_CMAP(s)
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                       markeredgecolor='white', markeredgewidth=1.5, markersize=12,
                       label=f'S={s:.2f}')
            )
        ax.legend(handles=legend_elements, loc='upper right', title='Tracer Sparsity',
                 fontsize=9, title_fontsize=10, framealpha=0.9)

        plt.tight_layout()

        frame_path = temp_dir / f'frame_{cp:03d}.png'
        plt.savefig(frame_path, dpi=120, bbox_inches='tight')
        frames.append(frame_path)
        plt.close()

    # Create GIF
    print("\nCreating GIF...")
    images = [imageio.imread(str(f)) for f in frames]
    images.extend([images[-1]] * 12)  # Pause at end

    gif_path = output_dir / 'phase_diagram_nonuniform.gif'
    imageio.mimsave(str(gif_path), images, duration=0.15, loop=0)
    print(f"Saved: {gif_path}")

    # Cleanup
    for f in frames:
        os.remove(f)
    temp_dir.rmdir()

    return gif_path


def main():
    parser = argparse.ArgumentParser(description='Create phase diagram GIF for non-uniform sparsity')
    parser.add_argument('--sample-seeds', type=int, default=64,
                        help='Number of seeds to sample (default: 64)')
    parser.add_argument('--n-tracers', type=int, default=3,
                        help='Tracers per sparsity bin (default: 3)')
    parser.add_argument('--trail-length', type=int, default=10,
                        help='Trail length in frames (default: 10)')
    args = parser.parse_args()

    create_phase_gif(DATA_DIR, OUTPUT_DIR, args.sample_seeds, args.n_tracers,
                     args.trail_length)


if __name__ == '__main__':
    main()
