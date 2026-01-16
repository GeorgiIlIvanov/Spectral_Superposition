#!/usr/bin/env python3
"""
Create animated GIF of spectral phase diagram for binned sparsity experiment.

Features are colored by their discrete sparsity S ∈ {0.1, 0.2, ..., 0.9}.

Color gradient (warm): #FF4E50 (S=0.1) → #FC913A → #F9D423 (S=0.9)
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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


def create_phase_gif(data_dir, output_dir, sample_seeds=64):
    """Create phase diagram GIF with sparsity coloring."""

    all_data = load_all_data(data_dir, sample_seeds)

    n_checkpoints = len(all_data[0]['steps'])
    steps = all_data[0]['steps']
    n_features = len(all_data[0]['sparsity'])
    sparsity_per_feature = all_data[0]['sparsity']  # Same for all seeds

    print(f"\nCheckpoints: {n_checkpoints}")
    print(f"Features: {n_features}")
    print(f"Seeds loaded: {len(all_data)}")
    print(f"Sparsity bins: {SPARSITY_VALUES}")

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

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(r'Feature Norm $\|W_i\|^2$', fontsize=14)
        ax.set_ylabel(r'Fractional Dimensionality $D_i$', fontsize=14)
        ax.set_title(f'Binned Sparsity Phase Diagram\n'
                    f'$S \in \{{0.1, 0.2, ..., 0.9\}}$ | m=256, n=1024 | Step {steps[cp]:,}',
                    fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)

        # Colorbar for sparsity
        cbar = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label('Feature Sparsity $S$', fontsize=12)
        # Set colorbar ticks to show actual sparsity values
        cbar.set_ticks([(s - 0.1) / 0.8 for s in SPARSITY_VALUES])
        cbar.set_ticklabels([f'{s:.1f}' for s in SPARSITY_VALUES])

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
    args = parser.parse_args()

    create_phase_gif(DATA_DIR, OUTPUT_DIR, args.sample_seeds)


if __name__ == '__main__':
    main()
