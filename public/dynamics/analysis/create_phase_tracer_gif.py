#!/usr/bin/env python3
"""
Create animated GIF of spectral phase diagram with tracer features.

Combines the aggregated phase plot (colored by capacity ratio m/n) with
tracer features from different sparsity bins (colored by sparsity).

Color gradients:
- Cool (capacity m/n): #00C9FF → #005BEA → #8E2DE2
- Warm (sparsity S): #FF4E50 → #FC913A → #F9D423

Usage:
    python create_phase_tracer_gif.py [--n-tracers N] [--trail-length L]
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import imageio.v2 as imageio
import os
from tqdm import tqdm
import argparse

DATA_DIR = Path('../start')
OUTPUT_DIR = Path('.')

# Sparsity bins: 10 bins from 0 to 1
SPARSITY_BINS = [i / 10 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]

# Custom colormaps
def create_cool_cmap():
    """Cool gradient: #00C9FF → #005BEA → #8E2DE2"""
    colors = ['deepskyblue', 'dodgerblue', 'mediumorchid']
    colors = ['#00C9FF', '#005BEA', '#8E2DE2']
    return LinearSegmentedColormap.from_list('cool_capacity', colors, N=256)

def create_warm_cmap():
    """Warm gradient: #FF4E50 → #FC913A → #F9D423"""
    colors = ['#FF4E50', '#FC913A', '#F9D423']
    return LinearSegmentedColormap.from_list('warm_sparsity', colors, N=256)

COOL_CMAP = create_cool_cmap()
WARM_CMAP = create_warm_cmap()


def get_sparsity_color(sparsity):
    """Get color for a sparsity value using warm colormap."""
    return WARM_CMAP(sparsity)


def get_sparsity_bin_color(bin_idx, n_bins=10):
    """Get color for a sparsity bin index."""
    # Map bin index to sparsity midpoint
    midpoint = (bin_idx + 0.5) / n_bins
    return WARM_CMAP(midpoint)


def select_tracers(all_data, n_tracers_per_bin=5, seed=42):
    """Select tracer features stratified by 10 sparsity bins."""
    np.random.seed(seed)
    tracers = []
    n_bins = len(SPARSITY_BINS) - 1

    for bin_idx in range(n_bins):
        s_low, s_high = SPARSITY_BINS[bin_idx], SPARSITY_BINS[bin_idx + 1]
        color = get_sparsity_bin_color(bin_idx, n_bins)

        # Find experiments in this sparsity bin
        matching_exps = [(i, d) for i, d in enumerate(all_data) if s_low <= d['s'] < s_high]

        if matching_exps:
            n_select = min(n_tracers_per_bin, len(matching_exps))
            selected_exps = np.random.choice(len(matching_exps), n_select, replace=False)

            for sel_idx in selected_exps:
                exp_idx, exp_data = matching_exps[sel_idx]
                final_norms = exp_data['norms'][-1]
                active_features = np.where(final_norms > 0.05)[0]
                if len(active_features) > 0:
                    feat_idx = np.random.choice(active_features)
                    tracers.append({
                        'exp_idx': exp_idx,
                        'feat_idx': feat_idx,
                        'color': color,
                        'bin_idx': bin_idx,
                        's': exp_data['s'],
                        'm': exp_data['m'],
                    })

    return tracers


def create_phase_tracer_gif(data_dir, output_dir, n_tracers_per_bin=5, trail_length=15,
                            sample_every=5):
    """Create GIF with phase diagram and tracer features."""

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
                'capacity': m / 1024,  # m/n ratio
                'steps': hf['checkpoint_steps'][:],
                'norms': hf['feature_norms'][:],
                'dims': hf['fractional_dims'][:],
            })

    n_checkpoints = len(all_data[0]['steps'])
    steps = all_data[0]['steps']

    # Select tracers
    tracers = select_tracers(all_data, n_tracers_per_bin)
    print(f"\nSelected {len(tracers)} tracer features across {len(SPARSITY_BINS)-1} sparsity bins")

    # Pre-compute trajectories
    tracer_trajectories = []
    for t in tracers:
        exp = all_data[t['exp_idx']]
        traj = {
            'norms': exp['norms'][:, t['feat_idx']],
            'dims': exp['dims'][:, t['feat_idx']],
            'color': t['color'],
            'bin_idx': t['bin_idx'],
            's': t['s'],
            'm': t['m'],
            'capacity': exp['capacity'],
        }
        tracer_trajectories.append(traj)

    # Create frames
    temp_dir = output_dir / 'gif_frames_phase_tracer'
    temp_dir.mkdir(exist_ok=True)

    frames = []
    all_final_norms = np.concatenate([d['norms'][-1] for d in all_data])
    x_max = np.percentile(all_final_norms, 99) * 1.1

    print(f"\nGenerating {n_checkpoints} frames...")

    for cp in tqdm(range(n_checkpoints), desc='Creating frames'):
        fig, ax = plt.subplots(figsize=(14, 10))

        # Background scatter - aggregate all experiments
        X_all, Y_all, C_all = [], [], []
        for d in all_data:
            X_all.append(d['norms'][cp])
            Y_all.append(d['dims'][cp])
            C_all.append(np.full(len(d['norms'][cp]), d['capacity']))

        X = np.concatenate(X_all)
        Y = np.concatenate(Y_all)
        C = np.concatenate(C_all)

        # Subsample for plotting
        if len(X) > 80000:
            idx = np.random.choice(len(X), 80000, replace=False)
            X, Y, C = X[idx], Y[idx], C[idx]

        # Main scatter plot with cool colormap (capacity m/n)
        sc = ax.scatter(X, Y, c=C, cmap=COOL_CMAP, s=3, alpha=0.3,
                       rasterized=True, vmin=0, vmax=1)

        # Reference lines
        x_ref = np.linspace(0, x_max, 100)
        for mu in [1, 2, 3, 4, 5, 6, 8]:
            ax.plot(x_ref, x_ref / mu, '--', alpha=0.4, linewidth=1.5,
                   color='gray', label=f'μ={mu}' if mu <= 3 else '')

        # Tracer trajectories with trails
        for traj in tracer_trajectories:
            if cp > 0:
                trail_start = max(0, cp - trail_length)
                trail_x = traj['norms'][trail_start:cp+1]
                trail_y = traj['dims'][trail_start:cp+1]

                # Draw trail with fading alpha
                for i in range(len(trail_x) - 1):
                    alpha = 0.15 + 0.5 * (i / max(1, len(trail_x) - 1))
                    ax.plot(trail_x[i:i+2], trail_y[i:i+2],
                           color=traj['color'], alpha=alpha, linewidth=2)

            # Current position - circle with thick stroke
            current_x = traj['norms'][cp]
            current_y = traj['dims'][cp]
            ax.scatter(current_x, current_y, c=[traj['color']], s=180,
                      edgecolors='white', linewidths=2.5, zorder=10, marker='o')

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(r'Feature Norm $\|W_i\|^2$', fontsize=14)
        ax.set_ylabel(r'Fractional Dimensionality $D_i$', fontsize=14)
        ax.set_title(f'Spectral Phase Diagram with Feature Tracers\nStep {steps[cp]:,}',
                     fontsize=15, weight='bold')
        ax.grid(True, alpha=0.3)

        # Colorbar for capacity ratio (cool)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label('Capacity Ratio m/n', fontsize=12)

        # Legend for sparsity bins (warm colors)
        n_bins = len(SPARSITY_BINS) - 1
        # Show subset of bins in legend to avoid clutter
        legend_bins = [0, 2, 4, 6, 9]  # S: 0-0.1, 0.2-0.3, 0.4-0.5, 0.6-0.7, 0.9-1.0
        legend_elements = []
        for i in legend_bins:
            s_low, s_high = SPARSITY_BINS[i], SPARSITY_BINS[i + 1]
            color = get_sparsity_bin_color(i, n_bins)
            label = f'S∈[{s_low:.1f},{s_high:.1f})'
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                       markeredgecolor='white', markeredgewidth=1.5, markersize=12,
                       label=label)
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

    gif_path = output_dir / 'phase_diagram_with_tracers.gif'
    imageio.mimsave(str(gif_path), images, duration=0.2, loop=0)
    print(f"Saved: {gif_path}")

    # Cleanup
    for f in frames:
        os.remove(f)
    temp_dir.rmdir()

    return gif_path


def main():
    parser = argparse.ArgumentParser(description='Create phase diagram GIF with tracers')
    parser.add_argument('--n-tracers', type=int, default=5,
                        help='Tracers per sparsity bin (default: 5)')
    parser.add_argument('--trail-length', type=int, default=15,
                        help='Trail length in frames (default: 15)')
    parser.add_argument('--sample-every', type=int, default=5,
                        help='Sample every Nth experiment (default: 5)')
    args = parser.parse_args()

    create_phase_tracer_gif(DATA_DIR, OUTPUT_DIR, args.n_tracers, args.trail_length,
                            args.sample_every)


if __name__ == '__main__':
    main()
