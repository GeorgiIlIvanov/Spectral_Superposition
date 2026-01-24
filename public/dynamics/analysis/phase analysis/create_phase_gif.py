#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v2 as imageio
import os
from tqdm import tqdm
import argparse

DATA_DIR = Path('../start')
OUTPUT_DIR = Path('.')


def find_experiment(data_dir, target_m, target_s, seed=0):
    best_file, best_dist = None, float('inf')
    for f in data_dir.glob('*.h5'):
        parts = f.stem.split('_')
        m = int(parts[1][1:])
        s = float(parts[2][1:])
        sd = int(parts[3][4:])
        if sd == seed:
            dist = abs(m - target_m)/100 + abs(s - target_s)
            if dist < best_dist:
                best_dist = dist
                best_file = f
    return best_file


def create_single_experiment_gif(data_dir, output_dir, target_m=256, target_s=0.5):

    exp_file = find_experiment(data_dir, target_m, target_s)
    print(f"Using: {exp_file.name}")

    with h5py.File(exp_file, 'r') as f:
        steps = f['checkpoint_steps'][:]
        frac_dims = f['fractional_dims'][:]
        norms = f['feature_norms'][:]
        losses = f['losses'][:]
        m_hidden = f.attrs['m_hidden']
        sparsity = f.attrs['sparsity']

    print(f"m={m_hidden}, S={sparsity:.3f}, checkpoints={len(steps)}")

    # Create temp directory
    temp_dir = output_dir / 'gif_frames'
    temp_dir.mkdir(exist_ok=True)

    frames = []
    x_max = np.percentile(norms[-1], 99.5) * 1.1

    for i in tqdm(range(len(steps)), desc='Creating frames'):
        fig, ax = plt.subplots(figsize=(10, 8))

        X, Y = norms[i], frac_dims[i]

        sc = ax.scatter(X, Y, c=Y, cmap='viridis', s=8, alpha=0.6, vmin=0, vmax=0.7)

        # Reference lines
        x_ref = np.linspace(0, x_max, 100)
        for mu in [1, 2, 3, 4, 5, 6, 8]:
            ax.plot(x_ref, x_ref / mu, '--', alpha=0.4, linewidth=1.5, color='red')

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(r'Feature Norm $\|W_i\|^2$', fontsize=12)
        ax.set_ylabel(r'Fractional Dimensionality $D_i$', fontsize=12)
        ax.set_title(f'Spectral Phase Diagram Evolution\nm={m_hidden}, S={sparsity:.2f} | Step {steps[i]:,} | Loss={losses[i]:.4f}',
                     fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3)

        plt.colorbar(sc, ax=ax, label='$D_i$')

        active = np.sum(X > 0.01)
        mean_di = np.mean(Y[X > 0.01]) if active > 0 else 0
        ax.text(0.02, 0.98, f'Active features: {active}\nMean D_i (active): {mean_di:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        frame_path = temp_dir / f'frame_{i:03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        frames.append(frame_path)
        plt.close()

    # Create GIF
    print("\nCreating GIF...")
    images = [imageio.imread(str(f)) for f in frames]
    images.extend([images[-1]] * 10)  # Pause at end

    gif_path = output_dir / 'phase_diagram_evolution.gif'
    imageio.mimsave(str(gif_path), images, duration=0.2, loop=0)
    print(f"Saved: {gif_path}")

    # Cleanup
    for f in frames:
        os.remove(f)
    temp_dir.rmdir()

    return gif_path


def create_multi_experiment_gif(data_dir, output_dir, sample_every=5):

    files = sorted(data_dir.glob('*.h5'))[::sample_every]
    print(f"Using {len(files)} experiments")

    # Load all data
    all_data = []
    for f in tqdm(files, desc='Loading'):
        parts = f.stem.split('_')
        m = int(parts[1][1:])
        with h5py.File(f, 'r') as hf:
            all_data.append({
                'm': m,
                'rho': m / 1024,
                'steps': hf['checkpoint_steps'][:],
                'norms': hf['feature_norms'][:],
                'dims': hf['fractional_dims'][:],
            })

    n_checkpoints = len(all_data[0]['steps'])
    steps = all_data[0]['steps']

    temp_dir = output_dir / 'gif_frames_multi'
    temp_dir.mkdir(exist_ok=True)

    frames = []

    # Find global axis limits
    all_final_norms = np.concatenate([d['norms'][-1] for d in all_data])
    x_max = np.percentile(all_final_norms, 99) * 1.1

    for cp in tqdm(range(n_checkpoints), desc='Creating frames'):
        fig, ax = plt.subplots(figsize=(12, 9))

        # Aggregate all experiments at this checkpoint
        X_all, Y_all, C_all = [], [], []
        for d in all_data:
            X_all.append(d['norms'][cp])
            Y_all.append(d['dims'][cp])
            C_all.append(np.full(len(d['norms'][cp]), d['rho']))

        X = np.concatenate(X_all)
        Y = np.concatenate(Y_all)
        C = np.concatenate(C_all)

        # Subsample for plotting
        if len(X) > 100000:
            idx = np.random.choice(len(X), 100000, replace=False)
            X, Y, C = X[idx], Y[idx], C[idx]

        sc = ax.scatter(X, Y, c=C, cmap='turbo', s=3, alpha=0.4, rasterized=True)

        # Reference lines
        x_ref = np.linspace(0, x_max, 100)
        for mu in [1, 2, 3, 4, 5, 6, 8]:
            ax.plot(x_ref, x_ref / mu, '--', alpha=0.5, linewidth=1.5, color='black',
                    label=f'μ={mu}' if mu <= 3 else '')

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(r'Feature Norm $\|W_i\|^2$', fontsize=14)
        ax.set_ylabel(r'Fractional Dimensionality $D_i$', fontsize=14)
        ax.set_title(f'Spectral Phase Diagram Evolution (Aggregated)\nStep {steps[cp]:,}',
                     fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Capacity Ratio m/n', fontsize=11)

        plt.tight_layout()

        frame_path = temp_dir / f'frame_{cp:03d}.png'
        plt.savefig(frame_path, dpi=120, bbox_inches='tight')
        frames.append(frame_path)
        plt.close()

    # Create GIF
    print("\nCreating GIF...")
    images = [imageio.imread(str(f)) for f in frames]
    images.extend([images[-1]] * 10)

    gif_path = output_dir / 'phase_diagram_evolution_multi.gif'
    imageio.mimsave(str(gif_path), images, duration=0.2, loop=0)
    print(f"Saved: {gif_path}")

    # Cleanup
    for f in frames:
        os.remove(f)
    temp_dir.rmdir()

    return gif_path


def main():
    parser = argparse.ArgumentParser(description='Create phase diagram evolution GIF')
    parser.add_argument('--multi', action='store_true', help='Aggregate multiple experiments')
    parser.add_argument('--m', type=int, default=256, help='Target hidden dimension')
    parser.add_argument('--s', type=float, default=0.5, help='Target sparsity')
    args = parser.parse_args()

    if args.multi:
        create_multi_experiment_gif(DATA_DIR, OUTPUT_DIR)
    else:
        create_single_experiment_gif(DATA_DIR, OUTPUT_DIR, args.m, args.s)


if __name__ == '__main__':
    main()
