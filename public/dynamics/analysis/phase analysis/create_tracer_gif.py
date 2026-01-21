#!/usr/bin/env python3
"""
Create animated GIF of spectral phase diagram with fluorescent tracer features.

Highlights a subset of features across different sparsity bins to visualize
individual feature trajectories through the phase space during training.

Tracer colors by sparsity:
- Red: S < 0.2
- Orange: 0.2 ≤ S < 0.4
- Gold: 0.4 ≤ S < 0.6
- Green: 0.6 ≤ S < 0.8
- Cyan: S ≥ 0.8

Usage:
    python create_tracer_gif.py [--n-tracers N] [--trail-length L]

    --n-tracers N: Number of tracers per sparsity bin (default: 10)
    --trail-length L: Length of trajectory trail in frames (default: 15)
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import imageio.v2 as imageio
import os
from tqdm import tqdm
import argparse

DATA_DIR = Path('../start')
OUTPUT_DIR = Path('.')

# Sparsity bin configuration
SPARSITY_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SPARSITY_COLORS = ['#FF0000', '#FF8C00', '#FFD700', '#00FF00', '#00BFFF']
SPARSITY_LABELS = ['S<0.2', '0.2≤S<0.4', '0.4≤S<0.6', '0.6≤S<0.8', 'S≥0.8']


def select_tracers(all_data, n_tracers_per_bin=10, seed=42):
    """Select tracer features stratified by sparsity bins."""
    np.random.seed(seed)
    tracers = []

    for bin_idx in range(len(SPARSITY_BINS) - 1):
        s_low, s_high = SPARSITY_BINS[bin_idx], SPARSITY_BINS[bin_idx + 1]
        color = SPARSITY_COLORS[bin_idx]

        matching_exps = [(i, d) for i, d in enumerate(all_data) if s_low <= d['s'] < s_high]

        if matching_exps:
            selected_exps = np.random.choice(len(matching_exps),
                                            min(n_tracers_per_bin, len(matching_exps)),
                                            replace=False)

            for sel_idx in selected_exps:
                exp_idx, exp_data = matching_exps[sel_idx]
                final_norms = exp_data['norms'][-1]
                active_features = np.where(final_norms > 0.05)[0]
                if len(active_features) > 0:
                    feat_idx = np.random.choice(active_features)
                    tracers.append((exp_idx, feat_idx, color, bin_idx))

    return tracers


def create_tracer_gif(data_dir, output_dir, n_tracers_per_bin=10, trail_length=15,
                      sample_every=10):
    """Create GIF with fluorescent tracer features."""

    files = sorted(data_dir.glob('*.h5'))[::sample_every]
    print(f"Using {len(files)} experiments")

    # Load data
    all_data = []
    for f in tqdm(files, desc='Loading'):
        parts = f.stem.split('_')
        m = int(parts[1][1:])
        s = float(parts[2][1:])
        with h5py.File(f, 'r') as hf:
            all_data.append({
                'm': m,
                's': s,
                'rho': 1024 / m,
                'steps': hf['checkpoint_steps'][:],
                'norms': hf['feature_norms'][:],
                'dims': hf['fractional_dims'][:],
            })

    n_checkpoints = len(all_data[0]['steps'])
    steps = all_data[0]['steps']

    # Select tracers
    tracers = select_tracers(all_data, n_tracers_per_bin)
    print(f"\nSelected {len(tracers)} tracer features")

    # Pre-compute trajectories
    tracer_trajectories = []
    for exp_idx, feat_idx, color, bin_idx in tracers:
        exp = all_data[exp_idx]
        traj = {
            'norms': exp['norms'][:, feat_idx],
            'dims': exp['dims'][:, feat_idx],
            'color': color,
            'bin_idx': bin_idx,
            'rho': exp['rho'],
            's': exp['s'],
        }
        tracer_trajectories.append(traj)

    # Create frames
    temp_dir = output_dir / 'gif_frames_tracer'
    temp_dir.mkdir(exist_ok=True)

    frames = []
    all_final_norms = np.concatenate([d['norms'][-1] for d in all_data])
    x_max = np.percentile(all_final_norms, 99) * 1.1

    print(f"\nGenerating {n_checkpoints} frames...")

    for cp in tqdm(range(n_checkpoints), desc='Creating frames'):
        fig, ax = plt.subplots(figsize=(14, 10))

        # Background scatter
        X_all, Y_all, C_all = [], [], []
        for d in all_data:
            X_all.append(d['norms'][cp])
            Y_all.append(d['dims'][cp])
            C_all.append(np.full(len(d['norms'][cp]), d['rho']))

        X = np.concatenate(X_all)
        Y = np.concatenate(Y_all)
        C = np.concatenate(C_all)

        if len(X) > 60000:
            idx = np.random.choice(len(X), 60000, replace=False)
            X, Y, C = X[idx], Y[idx], C[idx]

        sc = ax.scatter(X, Y, c=C, cmap='turbo', s=3, alpha=0.25, rasterized=True)

        # Reference lines
        x_ref = np.linspace(0, x_max, 100)
        for mu in [1, 2, 3, 4, 5, 6, 8]:
            ax.plot(x_ref, x_ref / mu, '--', alpha=0.3, linewidth=1, color='gray')

        # Tracer trajectories with trails
        for traj in tracer_trajectories:
            if cp > 0:
                trail_start = max(0, cp - trail_length)
                trail_x = traj['norms'][trail_start:cp+1]
                trail_y = traj['dims'][trail_start:cp+1]

                for i in range(len(trail_x) - 1):
                    alpha = 0.1 + 0.4 * (i / (len(trail_x) - 1))
                    ax.plot(trail_x[i:i+2], trail_y[i:i+2],
                           color=traj['color'], alpha=alpha, linewidth=1.5)

            current_x = traj['norms'][cp]
            current_y = traj['dims'][cp]
            ax.scatter(current_x, current_y, c=traj['color'], s=120,
                      edgecolors='white', linewidths=1.5, zorder=10, marker='o')

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(r'Feature Norm $\|W_i\|^2$', fontsize=14)
        ax.set_ylabel(r'Fractional Dimensionality $D_i$', fontsize=14)
        ax.set_title(f'Spectral Phase Diagram with Feature Tracers\nStep {steps[cp]:,}',
                     fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label('Compression n/m', fontsize=11)

        legend_elements = [Patch(facecolor=SPARSITY_COLORS[i], edgecolor='white',
                                label=SPARSITY_LABELS[i]) for i in range(len(SPARSITY_COLORS))]
        ax.legend(handles=legend_elements, loc='upper right', title='Tracer Sparsity',
                 fontsize=9, title_fontsize=10)

        plt.tight_layout()

        frame_path = temp_dir / f'frame_{cp:03d}.png'
        plt.savefig(frame_path, dpi=120, bbox_inches='tight')
        frames.append(frame_path)
        plt.close()

    # Create GIF
    print("\nCreating GIF...")
    images = [imageio.imread(str(f)) for f in frames]
    images.extend([images[-1]] * 15)

    gif_path = output_dir / 'phase_diagram_tracers.gif'
    imageio.mimsave(str(gif_path), images, duration=0.18, loop=0)
    print(f"Saved: {gif_path}")

    # Cleanup
    for f in frames:
        os.remove(f)
    temp_dir.rmdir()

    return gif_path


def main():
    parser = argparse.ArgumentParser(description='Create phase diagram GIF with tracers')
    parser.add_argument('--n-tracers', type=int, default=10,
                        help='Tracers per sparsity bin')
    parser.add_argument('--trail-length', type=int, default=15,
                        help='Trail length in frames')
    args = parser.parse_args()

    create_tracer_gif(DATA_DIR, OUTPUT_DIR, args.n_tracers, args.trail_length)


if __name__ == '__main__':
    main()
