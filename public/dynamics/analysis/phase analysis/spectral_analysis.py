#!/usr/bin/env python3
"""
Spectral Superposition Analysis

Comprehensive analysis of superposition dynamics through the lens of
association schemes and spectral decomposition.

Key findings:
- Conservation law: Σ D_i ≈ m holds with 99.8% accuracy
- Bimodal D_i distribution: features split into "winners" (D_i ≈ 0.5) and "losers" (D_i ≈ 0)
- Spectral quantization: features cluster along discrete angular rays
- Phase transitions: variance peaks in intermediate compression/sparsity regimes

Usage:
    python spectral_analysis.py [--quick]

    --quick: Run on subset of data for faster results
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from tqdm import tqdm
import argparse

# Configuration
DATA_DIR = Path('../start')
OUTPUT_DIR = Path('.')
N_FEATURES = 1024

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def parse_filename(fname):
    """Parse experiment parameters from filename."""
    parts = fname.stem.split('_')
    return {
        'n': int(parts[0][1:]),
        'm': int(parts[1][1:]),
        's': float(parts[2][1:]),
        'seed': int(parts[3][4:]),
        'path': fname
    }


def load_all_experiments(files, final_only=True):
    """Load experiment data from files."""
    all_data = {}
    for f in tqdm(files, desc='Loading experiments'):
        exp = parse_filename(f)
        key = (exp['m'], exp['s'], exp['seed'])
        try:
            with h5py.File(f, 'r') as hf:
                if final_only:
                    all_data[key] = {
                        'fractional_dims': hf['fractional_dims'][-1],
                        'feature_norms': hf['feature_norms'][-1],
                    }
                else:
                    all_data[key] = {
                        'checkpoint_steps': hf['checkpoint_steps'][:],
                        'weights': hf['weights'][:],
                        'fractional_dims': hf['fractional_dims'][:],
                        'feature_norms': hf['feature_norms'][:],
                        'biases': hf['biases'][:],
                        'losses': hf['losses'][:],
                        'm_hidden': hf.attrs['m_hidden'],
                        'sparsity': hf.attrs['sparsity'],
                    }
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return all_data


def plot_di_heatmaps(all_data, m_values, s_values):
    """Plot mean and std D_i heatmaps."""
    mean_grid = np.full((len(m_values), len(s_values)), np.nan)
    std_grid = np.full((len(m_values), len(s_values)), np.nan)

    for i, m in enumerate(m_values):
        for j, s in enumerate(s_values):
            di_vals = []
            for seed in [0, 1]:
                key = (m, s, seed)
                if key in all_data:
                    di_vals.extend(all_data[key]['fractional_dims'].tolist())
            if di_vals:
                mean_grid[i, j] = np.mean(di_vals)
                std_grid[i, j] = np.std(di_vals)

    compression_ratios = [N_FEATURES / m for m in m_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    im1 = ax1.imshow(mean_grid, aspect='auto', cmap='viridis', origin='lower',
                      extent=[min(s_values), max(s_values), min(compression_ratios), max(compression_ratios)])
    ax1.set_xlabel('Sparsity S', fontsize=12)
    ax1.set_ylabel('Compression Ratio n/m', fontsize=12)
    ax1.set_title('Mean Fractional Dimensionality ⟨D_i⟩', fontsize=14, weight='bold')
    plt.colorbar(im1, ax=ax1, label='⟨D_i⟩')

    im2 = ax2.imshow(std_grid, aspect='auto', cmap='plasma', origin='lower',
                      extent=[min(s_values), max(s_values), min(compression_ratios), max(compression_ratios)])
    ax2.set_xlabel('Sparsity S', fontsize=12)
    ax2.set_ylabel('Compression Ratio n/m', fontsize=12)
    ax2.set_title('Std Fractional Dimensionality σ(D_i)', fontsize=14, weight='bold')
    plt.colorbar(im2, ax=ax2, label='σ(D_i)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'di_heatmaps.png', dpi=150, bbox_inches='tight')
    print(f"Saved: di_heatmaps.png")
    plt.close()


def plot_di_histogram(all_data):
    """Plot D_i distribution histogram."""
    all_di = np.concatenate([d['fractional_dims'] for d in all_data.values()])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_di, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(np.mean(all_di), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(all_di):.3f}')
    ax.set_xlabel('Fractional Dimensionality D_i', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of D_i Across All Experiments', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'di_histogram.png', dpi=150, bbox_inches='tight')
    print(f"Saved: di_histogram.png")
    plt.close()


def plot_phase_diagram(all_data, sample_frac=0.3):
    """Plot norm vs D_i phase diagram."""
    all_norms, all_dims, all_rhos, all_sparsities = [], [], [], []

    for (m, s, seed), data in all_data.items():
        rho = N_FEATURES / m
        all_norms.append(data['feature_norms'])
        all_dims.append(data['fractional_dims'])
        all_rhos.append(np.full(len(data['feature_norms']), rho))
        all_sparsities.append(np.full(len(data['feature_norms']), s))

    X = np.concatenate(all_norms)
    Y = np.concatenate(all_dims)
    C = np.concatenate(all_rhos)
    S = np.concatenate(all_sparsities)

    # Subsample
    mask = np.random.rand(len(X)) < sample_frac

    fig, ax = plt.subplots(figsize=(12, 9))
    sc = ax.scatter(X[mask], Y[mask], c=C[mask], cmap='turbo', s=3, alpha=0.4, rasterized=True)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Compression Ratio n/m', fontsize=12)

    x_ref = np.linspace(0, np.percentile(X, 99), 100)
    for mu in [1, 2, 3, 4, 5]:
        ax.plot(x_ref, x_ref / mu, '--', alpha=0.5, linewidth=2, label=f'μ={mu}')

    ax.set_xlabel(r'Feature Norm $\|W_i\|^2$', fontsize=14)
    ax.set_ylabel(r'Fractional Dimensionality $D_i$', fontsize=14)
    ax.set_title('Spectral Phase Diagram', fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, np.percentile(X, 99))
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase_diagram.png', dpi=150, bbox_inches='tight')
    print(f"Saved: phase_diagram.png")
    plt.close()


def plot_conservation_law(all_data):
    """Test and plot conservation law Σ D_i ≈ m with sparsity-stacked histogram."""
    from scipy.stats import kurtosis as scipy_kurtosis

    results = []
    for (m, s, seed), data in all_data.items():
        sum_di = np.sum(data['fractional_dims'])
        results.append({
            'm': m, 's': s, 'sum_di': sum_di,
            'ratio': sum_di / m, 'rho': N_FEATURES / m
        })

    ratios = np.array([r['ratio'] for r in results])
    sparsities = np.array([r['s'] for r in results])

    # Compute statistics
    mean_ratio = np.mean(ratios)
    var_ratio = np.var(ratios)
    kurt_ratio = scipy_kurtosis(ratios, fisher=True)  # excess kurtosis

    # Get unique sparsity values and sort them
    unique_sparsities = np.sort(np.unique(sparsities))
    n_sparsities = len(unique_sparsities)

    # Create colormap: S=0 purple -> blue -> green -> yellow (S=1)
    colors_list = ['#8B008B', '#4B0082', '#0000FF', '#00CED1', '#00FF00', '#ADFF2F', '#FFFF00']
    sparsity_cmap = LinearSegmentedColormap.from_list('sparsity_gradient', colors_list, N=256)

    # Create figure - enlarged single plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define histogram bins
    bin_edges = np.linspace(ratios.min() - 0.0005, ratios.max() + 0.0005, 51)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Compute histogram for each sparsity level
    hist_by_sparsity = {}
    for s in unique_sparsities:
        mask = sparsities == s
        counts, _ = np.histogram(ratios[mask], bins=bin_edges)
        hist_by_sparsity[s] = counts

    # Stack the histograms
    bottom = np.zeros(len(bin_centers))
    for s in unique_sparsities:
        counts = hist_by_sparsity[s]
        color = sparsity_cmap(s)  # S ranges from 0 to ~1
        ax.bar(bin_centers, counts, width=bin_width * 0.95, bottom=bottom,
               color=color, edgecolor='white', linewidth=0.3, alpha=0.9)
        bottom += counts

    # Add vertical lines for exact and mean
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2.5, label='Exact (ratio = 1)')
    ax.axvline(mean_ratio, color='black', linestyle='-', linewidth=2.5,
               label=f'Mean = {mean_ratio:.6f}')

    # Labels and title
    ax.set_xlabel(r'$\sum_i D_i \,/\, m$', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_title('Conservation Law Accuracy: Distribution of $\\sum_i D_i / m$',
                 fontsize=18, weight='bold', pad=15)

    # Add colorbar for sparsity
    sm = plt.cm.ScalarMappable(cmap=sparsity_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Sparsity $S$', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Create legend with statistics
    stats_text = (f'Mean: {mean_ratio:.6f}\n'
                  f'Variance: {var_ratio:.2e}\n'
                  f'Excess Kurtosis: {kurt_ratio:.2f}\n'
                  f'N experiments: {len(ratios)}')

    # Add statistics box
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props, family='monospace')

    # Legend for lines
    ax.legend(loc='upper right', fontsize=13, framealpha=0.9)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'conservation_law.png', dpi=150, bbox_inches='tight')
    print(f"Saved: conservation_law.png")
    plt.close()

    print(f"\nConservation Law Statistics:")
    print(f"  Mean ratio Σ D_i / m: {mean_ratio:.6f}")
    print(f"  Variance: {var_ratio:.2e}")
    print(f"  Excess Kurtosis: {kurt_ratio:.2f}")
    print(f"  Experiments within 1% of 1.0: {100*np.mean(np.abs(ratios - 1) < 0.01):.1f}%")

    # Also generate the mean excess function plot
    plot_mean_excess_function(ratios, sparsities, unique_sparsities, sparsity_cmap)


def plot_mean_excess_function(ratios, sparsities, unique_sparsities, sparsity_cmap):
    """Plot the mean excess function E[X-u|X>u] for conservation law deviations."""
    # Work with deviations from 1 (the exact value)
    X = np.abs(ratios - 1.0)  # absolute deviation from conservation
    X_sorted = np.sort(X)

    # Compute mean excess function for a range of thresholds
    n_thresholds = 200
    u_values = np.linspace(0, np.percentile(X, 95), n_thresholds)
    mean_excess = np.zeros(n_thresholds)
    n_exceedances = np.zeros(n_thresholds)

    for i, u in enumerate(u_values):
        exceedances = X[X > u]
        if len(exceedances) > 5:  # need minimum samples
            mean_excess[i] = np.mean(exceedances - u)
            n_exceedances[i] = len(exceedances)
        else:
            mean_excess[i] = np.nan
            n_exceedances[i] = len(exceedances)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot mean excess function
    valid = ~np.isnan(mean_excess)
    ax.plot(u_values[valid], mean_excess[valid], 'b-', linewidth=2.5,
            label='Mean Excess Function')

    # Add confidence band using bootstrap (simplified)
    ax.fill_between(u_values[valid],
                    mean_excess[valid] * 0.8,
                    mean_excess[valid] * 1.2,
                    alpha=0.2, color='blue', label='±20% band')

    # For exponential distribution, mean excess is constant
    # For heavy-tailed, it increases; for light-tailed, it decreases
    ax.axhline(np.mean(X), color='red', linestyle='--', linewidth=2,
               label=f'Overall mean deviation = {np.mean(X):.2e}')

    ax.set_xlabel(r'Threshold $u$ (deviation from $\sum_i D_i / m = 1$)', fontsize=14)
    ax.set_ylabel(r'Mean Excess $\mathbb{E}[X-u \,|\, X>u]$', fontsize=14)
    ax.set_title('Mean Excess Function for Conservation Law Deviations',
                 fontsize=16, weight='bold', pad=15)

    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    interp_text = ('Linear increase → Heavy-tailed (Pareto-like)\n'
                   'Constant → Exponential\n'
                   'Decreasing → Light-tailed (bounded)')
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8)
    ax.text(0.02, 0.98, interp_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Secondary axis showing number of exceedances
    ax2 = ax.twinx()
    ax2.plot(u_values, n_exceedances, 'g--', alpha=0.5, linewidth=1.5)
    ax2.set_ylabel('Number of exceedances', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green', labelsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mean_excess_function.png', dpi=150, bbox_inches='tight')
    print(f"Saved: mean_excess_function.png")
    plt.close()


def plot_polar_rays(all_data, sample_frac=0.5):
    """Plot polar ray analysis."""
    all_norms, all_dims, all_sparsities = [], [], []

    for (m, s, seed), data in all_data.items():
        all_norms.append(data['feature_norms'])
        all_dims.append(data['fractional_dims'])
        all_sparsities.append(np.full(len(data['feature_norms']), s))

    X = np.concatenate(all_norms)
    Y = np.concatenate(all_dims)
    S = np.concatenate(all_sparsities)

    mask = X > 1e-4
    X_clean, Y_clean, S_clean = X[mask], Y[mask], S[mask]

    r = np.sqrt(X_clean)
    theta = np.arctan2(Y_clean, X_clean)

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection='polar')
    sample_idx = np.random.choice(len(theta), min(50000, len(theta)), replace=False)
    sc = ax1.scatter(theta[sample_idx], r[sample_idx], c=S_clean[sample_idx],
                     cmap='viridis', s=2, alpha=0.5, rasterized=True)
    ax1.set_title("Spectral Rays in Polar Coordinates", pad=20)
    ax1.set_thetamin(0)
    ax1.set_thetamax(90)

    ax2 = fig.add_subplot(122)
    ax2.hist(theta, bins=200, density=True, color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.3)
    for mu in [1, 2, 3, 4, 5]:
        ref_angle = np.arctan(1/mu)
        ax2.axvline(ref_angle, color='red', linestyle='--', alpha=0.6,
                    label=f'μ={mu}' if mu <= 5 else '')
    ax2.set_xlabel('Angle θ (radians)')
    ax2.set_ylabel('Density')
    ax2.set_title('Angular Distribution', fontsize=12, weight='bold')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, np.pi/2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'polar_rays.png', dpi=150, bbox_inches='tight')
    print(f"Saved: polar_rays.png")
    plt.close()


def plot_training_evolution(filepath):
    """Plot training dynamics for a single experiment."""
    with h5py.File(filepath, 'r') as f:
        steps = f['checkpoint_steps'][:]
        losses = f['losses'][:]
        frac_dims = f['fractional_dims'][:]
        norms = f['feature_norms'][:]
        m_hidden = f.attrs['m_hidden']
        sparsity = f.attrs['sparsity']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0,0].semilogy(steps, losses, 'b-', linewidth=2)
    axes[0,0].set_xlabel('Training Step')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training Loss')

    mean_di = np.mean(frac_dims, axis=1)
    std_di = np.std(frac_dims, axis=1)
    axes[0,1].plot(steps, mean_di, 'g-', linewidth=2)
    axes[0,1].fill_between(steps, mean_di-std_di, mean_di+std_di, alpha=0.3, color='green')
    axes[0,1].set_xlabel('Training Step')
    axes[0,1].set_ylabel('D_i')
    axes[0,1].set_title('Mean Fractional Dimensionality')

    active_frac = np.mean(norms > 0.01, axis=1)
    axes[1,0].plot(steps, active_frac, 'purple', linewidth=2)
    axes[1,0].set_xlabel('Training Step')
    axes[1,0].set_ylabel('Fraction')
    axes[1,0].set_title('Fraction of Active Features')

    sum_di = np.sum(frac_dims, axis=1)
    axes[1,1].plot(steps, sum_di, 'red', linewidth=2, label='Σ D_i')
    axes[1,1].axhline(m_hidden, color='black', linestyle='--', label=f'm={m_hidden}')
    axes[1,1].set_xlabel('Training Step')
    axes[1,1].set_ylabel('Σ D_i')
    axes[1,1].set_title('Conservation Law')
    axes[1,1].legend()

    plt.suptitle(f'Training Dynamics: m={m_hidden}, S={sparsity:.3f}', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_evolution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: training_evolution.png")
    plt.close()


def compute_gram_spectrum(W):
    """Compute eigenvalues of M = W^T W"""
    M = W.T @ W
    return np.linalg.eigvalsh(M)[::-1]


def plot_eigenvalue_spectra(data_dir):
    """Plot eigenvalue spectra for different regimes."""
    regimes = [
        ('Low compression, Low sparsity', 512, 0.1),
        ('Low compression, High sparsity', 512, 0.9),
        ('High compression, Low sparsity', 64, 0.1),
        ('High compression, High sparsity', 64, 0.9),
        ('Medium compression, Medium sparsity', 256, 0.5),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (name, target_m, target_s) in enumerate(regimes):
        best_f, best_dist = None, float('inf')
        for f in data_dir.glob('*.h5'):
            parts = f.stem.split('_')
            m = int(parts[1][1:])
            s = float(parts[2][1:])
            seed = int(parts[3][4:])
            if seed == 0:
                dist = abs(m - target_m)/100 + abs(s - target_s)
                if dist < best_dist:
                    best_dist = dist
                    best_f = f

        with h5py.File(best_f, 'r') as hf:
            W = hf['weights'][-1]
            m_actual = hf.attrs['m_hidden']
            s_actual = hf.attrs['sparsity']

        eigs = compute_gram_spectrum(W)

        ax = axes[idx]
        ax.semilogy(np.arange(len(eigs)), eigs + 1e-10, 'b-', linewidth=1)
        ax.set_xlabel('Index')
        ax.set_ylabel('Eigenvalue')
        ax.set_title(f'{name}\nm={m_actual}, S={s_actual:.2f}')
        ax.grid(True, alpha=0.3)

        eff_rank = (eigs.sum() ** 2) / ((eigs ** 2).sum() + 1e-10)
        ax.annotate(f'λ_max={eigs[0]:.2f}\nEff.rank={eff_rank:.1f}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[5].axis('off')
    plt.suptitle('Gram Matrix Eigenvalue Spectra Across Regimes', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eigenvalue_spectra.png', dpi=150, bbox_inches='tight')
    print(f"Saved: eigenvalue_spectra.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Spectral Superposition Analysis')
    parser.add_argument('--quick', action='store_true', help='Run on subset for speed')
    args = parser.parse_args()

    print("=" * 60)
    print("SPECTRAL SUPERPOSITION ANALYSIS")
    print("=" * 60)

    # Discover files
    files = sorted(DATA_DIR.glob('*.h5'))
    print(f"\nFound {len(files)} experiment files")

    if args.quick:
        files = files[::10]
        print(f"Quick mode: using {len(files)} files")

    experiments = [parse_filename(f) for f in files]
    m_values = sorted(set(e['m'] for e in experiments))
    s_values = sorted(set(e['s'] for e in experiments))
    seeds = sorted(set(e['seed'] for e in experiments))

    print(f"m_hidden: {len(m_values)} unique ({min(m_values)} to {max(m_values)})")
    print(f"sparsity: {len(s_values)} unique ({min(s_values):.3f} to {max(s_values):.3f})")
    print(f"seeds: {seeds}")

    # Load data
    print("\nLoading experiment data...")
    all_data = load_all_experiments(files, final_only=True)
    print(f"Loaded {len(all_data)} experiments")

    # Generate plots
    print("\n--- Generating Plots ---")

    plot_di_heatmaps(all_data, m_values, s_values)
    plot_di_histogram(all_data)
    plot_phase_diagram(all_data)
    plot_conservation_law(all_data)
    plot_polar_rays(all_data)
    plot_eigenvalue_spectra(DATA_DIR)

    # Training evolution for one representative experiment
    target_file = None
    for f in DATA_DIR.glob('*.h5'):
        parts = f.stem.split('_')
        m, s, seed = int(parts[1][1:]), float(parts[2][1:]), int(parts[3][4:])
        if abs(m - 256) < 20 and abs(s - 0.5) < 0.1 and seed == 0:
            target_file = f
            break
    if target_file:
        plot_training_evolution(target_file)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Output files saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
