#!/usr/bin/env python3
"""
Create animated GIF of spectral phase diagram for binned sparsity experiment.

Features are colored by their discrete sparsity S ∈ {0.1, 0.2, ..., 0.9}.
Tracers highlight features from each sparsity bin.

Color gradient (warm): #FF4E50 (S=0.1) → #FC913A → #F9D423 (S=0.9)
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

# Sparsity bins
SPARSITY_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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


def select_tracer_features(sparsity_array, n_tracers_per_bin=3):
    """
    Select tracer feature indices from each discrete sparsity bin.
    Returns list of tracer dicts.
    """
    tracers = []

    for s_val in SPARSITY_VALUES:
        # Find features with this exact sparsity
        mask = np.isclose(sparsity_array, s_val)
        indices = np.where(mask)[0]

        if len(indices) > 0:
            # Sample evenly from this bin
            selected = np.linspace(0, len(indices) - 1, n_tracers_per_bin, dtype=int)
            for sel_idx in selected:
                idx = indices[sel_idx]
                # Map sparsity 0.1-0.9 to colormap 0-1
                color_val = (s_val - 0.1) / 0.8
                color = WARM_CMAP(color_val)
                tracers.append({
                    'feature_idx': idx,
                    'sparsity': s_val,
                    'color': color,
                })

    return tracers


def create_phase_gif(data_dir, output_dir, sample_seeds=64, n_tracers_per_bin=3,
                     trail_length=10):
    """Create phase diagram GIF with binned sparsity coloring and tracers."""

    all_data = load_all_data(data_dir, sample_seeds)

    n_checkpoints = len(all_data[0]['steps'])
    steps = all_data[0]['steps']
    n_features = len(all_data[0]['sparsity'])
    sparsity_per_feature = all_data[0]['sparsity']  # Same for all seeds

    print(f"\nCheckpoints: {n_checkpoints}")
    print(f"Features: {n_features}")
    print(f"Seeds loaded: {len(all_data)}")
    print(f"Sparsity bins: {SPARSITY_VALUES}")

    # Select tracer features
    tracers = select_tracer_features(sparsity_per_feature, n_tracers_per_bin)
    print(f"Tracers: {len(tracers)} features ({n_tracers_per_bin} per bin × {len(SPARSITY_VALUES)} bins)")

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
            'dims_mean': dims_across_seeds.mean(axis=0),
        })

    # Create frames
    temp_dir = output_dir / 'gif_frames_binned'
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
            # Map sparsity 0.1-0.9 to color range 0-1
            colors = (sparsity_per_feature - 0.1) / 0.8
            C_all.append(colors)

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
        ax.set_title(f'Binned Sparsity Phase Diagram\n'
                    f'$S \\in \\{{0.1, 0.2, ..., 0.9\\}}$ | m=256, n=1024 | Step {steps[cp]:,}',
                    fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)

        # Colorbar for sparsity
        cbar = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label('Feature Sparsity $S$', fontsize=12)
        # Set colorbar ticks to show actual sparsity values
        cbar.set_ticks([(s - 0.1) / 0.8 for s in SPARSITY_VALUES])
        cbar.set_ticklabels([f'{s:.1f}' for s in SPARSITY_VALUES])

        # Legend for tracer sparsity bins
        legend_elements = []
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
            color_val = (s - 0.1) / 0.8
            color = WARM_CMAP(color_val)
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                       markeredgecolor='white', markeredgewidth=1.5, markersize=12,
                       label=f'S={s:.1f}')
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

    gif_path = output_dir / 'phase_diagram_binned.gif'
    imageio.mimsave(str(gif_path), images, duration=0.15, loop=0)
    print(f"Saved: {gif_path}")

    # Cleanup
    for f in frames:
        os.remove(f)
    temp_dir.rmdir()

    return gif_path


def main():
    parser = argparse.ArgumentParser(description='Create phase diagram GIF for binned sparsity')
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
