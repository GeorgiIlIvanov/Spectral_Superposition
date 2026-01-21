#!/usr/bin/env python3
"""
Dark Matter Investigation: Characterizing Features That Defy Linear Scaling

This script investigates features that do NOT follow the linear scaling law
D_i ∝ ||W_i||². We examine:
1. What distinguishes "well-behaved" features (R² > 0.9) from "dark matter" (R² < 0.9)
2. Whether eigenspace clustering explains the variance
3. The shape of non-linear trajectories
4. Correlation with model architecture (m_hidden)

Author: Analysis Protocol Implementation
Date: 2025-01-21
"""

import h5py
import numpy as np
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')


INPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
SVD_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/svd_results')
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/stratified_linearity')

# Focus on high sparsity regime
HIGH_SPARSITY_RANGE = (0.9, 1.0)
DEEP_SUPERPOSITION_THRESHOLD = 0.5


def analyze_trajectory_shape(norms, dims):
    """
    Analyze the shape of a feature trajectory beyond simple R².

    Returns:
        dict with shape characteristics
    """
    valid = np.isfinite(norms) & np.isfinite(dims) & (norms > 1e-8)
    if np.sum(valid) < 5:
        return None

    x = norms[valid]
    y = dims[valid]

    # Linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r2_linear = r_value ** 2

    # Compute residuals
    y_pred_linear = slope * x + intercept
    residuals = y - y_pred_linear

    # Check for systematic non-linearity: fit quadratic
    if len(x) >= 6:
        try:
            coeffs = np.polyfit(x, y, 2)
            y_pred_quad = np.polyval(coeffs, x)
            ss_res_quad = np.sum((y - y_pred_quad) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0

            # Curvature: sign of quadratic coefficient
            curvature = coeffs[0]
        except:
            r2_quad = np.nan
            curvature = np.nan
    else:
        r2_quad = np.nan
        curvature = np.nan

    # Check for monotonicity
    if len(x) > 1:
        # Sort by x (norm)
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        dy = np.diff(y_sorted)
        monotonic_increasing = np.all(dy >= -1e-6)
        monotonic_decreasing = np.all(dy <= 1e-6)
        is_monotonic = monotonic_increasing or monotonic_decreasing

        # Count direction changes
        direction_changes = np.sum(np.diff(np.sign(dy)) != 0)
    else:
        is_monotonic = True
        direction_changes = 0

    # Early vs late phase comparison
    mid_idx = len(x) // 2
    if mid_idx > 2:
        # Early phase slope
        early_slope, _, early_r, _, _ = stats.linregress(x[:mid_idx], y[:mid_idx])
        early_r2 = early_r ** 2

        # Late phase slope
        late_slope, _, late_r, _, _ = stats.linregress(x[mid_idx:], y[mid_idx:])
        late_r2 = late_r ** 2

        slope_ratio = late_slope / early_slope if abs(early_slope) > 1e-10 else np.nan
    else:
        early_r2 = np.nan
        late_r2 = np.nan
        slope_ratio = np.nan

    return {
        'r2_linear': float(r2_linear),
        'r2_quadratic': float(r2_quad),
        'slope': float(slope),
        'intercept': float(intercept),
        'curvature': float(curvature) if np.isfinite(curvature) else 0.0,
        'is_monotonic': bool(is_monotonic),
        'direction_changes': int(direction_changes),
        'early_r2': float(early_r2) if np.isfinite(early_r2) else np.nan,
        'late_r2': float(late_r2) if np.isfinite(late_r2) else np.nan,
        'slope_ratio': float(slope_ratio) if np.isfinite(slope_ratio) else np.nan,
        'norm_range': float(x.max() - x.min()),
        'dim_range': float(y.max() - y.min()),
        'mean_norm': float(np.mean(x)),
        'final_dim': float(y[-1]),
    }


def get_eigenspace_info(filepath, svd_filepath):
    """
    Get eigenspace assignment information for features.
    """
    with h5py.File(filepath, 'r') as f:
        weights = f['weights'][-1]  # Final checkpoint
        feature_norms = f['feature_norms'][-1]
        fractional_dims = f['fractional_dims'][-1]

    with h5py.File(svd_filepath, 'r') as f:
        U = f['U'][-1]  # (m, m)
        eigenvalues = f['eigenvalues'][-1]  # (m,)

    m, n = weights.shape

    # Project features onto eigenspaces
    projections = U.T @ weights  # (m, n)
    proj_squared = projections ** 2

    # Normalize by feature norms
    norms_sq = np.maximum(np.sum(weights ** 2, axis=0), 1e-10)
    normalized_proj = proj_squared / norms_sq  # (m, n)

    # Dominant eigenspace and concentration
    dominant_eigenspace = np.argmax(normalized_proj, axis=0)  # (n,)
    max_concentration = np.max(normalized_proj, axis=0)  # (n,)

    # Get eigenvalue of dominant eigenspace
    dominant_eigenvalue = eigenvalues[dominant_eigenspace]

    # Entropy of eigenspace distribution (measure of diffuseness)
    proj_normalized = normalized_proj / (np.sum(normalized_proj, axis=0) + 1e-10)
    entropy = -np.sum(proj_normalized * np.log(proj_normalized + 1e-10), axis=0)

    return {
        'dominant_eigenspace': dominant_eigenspace,
        'max_concentration': max_concentration,
        'dominant_eigenvalue': dominant_eigenvalue,
        'eigenspace_entropy': entropy,
        'feature_norms': feature_norms,
        'fractional_dims': fractional_dims,
    }


def analyze_file_detailed(filepath, svd_filepath):
    """
    Detailed analysis of a single file for dark matter investigation.
    """
    with h5py.File(filepath, 'r') as f:
        m_hidden = int(f.attrs['m_hidden'])
        sparsity = float(f.attrs['sparsity'])
        seed = int(f.attrs['seed'])

        feature_norms = f['feature_norms'][:]  # (T, n)
        fractional_dims = f['fractional_dims'][:]  # (T, n)

    # Get eigenspace info
    eigen_info = get_eigenspace_info(filepath, svd_filepath)

    T, n = feature_norms.shape

    results = []
    for i in range(n):
        # Trajectory analysis
        shape_stats = analyze_trajectory_shape(feature_norms[:, i], fractional_dims[:, i])
        if shape_stats is None:
            continue

        final_dim = fractional_dims[-1, i]
        is_deep_superposition = final_dim < DEEP_SUPERPOSITION_THRESHOLD

        results.append({
            'feature_idx': i,
            'm_hidden': m_hidden,
            'sparsity': sparsity,
            'seed': seed,
            'is_deep_superposition': is_deep_superposition,
            'dominant_eigenspace': int(eigen_info['dominant_eigenspace'][i]),
            'max_concentration': float(eigen_info['max_concentration'][i]),
            'dominant_eigenvalue': float(eigen_info['dominant_eigenvalue'][i]),
            'eigenspace_entropy': float(eigen_info['eigenspace_entropy'][i]),
            **shape_stats,
        })

    return results


def create_dark_matter_comparison_plots(well_behaved, dark_matter, output_dir):
    """
    Create comparison plots between well-behaved and dark matter features.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Convert to arrays for easier plotting
    wb = {k: np.array([f[k] for f in well_behaved]) for k in well_behaved[0].keys()
          if isinstance(well_behaved[0][k], (int, float))}
    dm = {k: np.array([f[k] for f in dark_matter]) for k in dark_matter[0].keys()
          if isinstance(dark_matter[0][k], (int, float))}

    # 1. Eigenspace concentration comparison
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 51)
    ax.hist(wb['max_concentration'], bins=bins, alpha=0.6, density=True,
           label=f'Well-behaved (n={len(well_behaved):,})', color='green')
    ax.hist(dm['max_concentration'], bins=bins, alpha=0.6, density=True,
           label=f'Dark matter (n={len(dark_matter):,})', color='red')
    ax.set_xlabel('Eigenspace Concentration (max projection²/||W||²)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Eigenspace Concentration:\nDark Matter vs Well-Behaved', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Eigenspace entropy comparison
    ax = axes[0, 1]
    max_entropy = max(np.percentile(wb['eigenspace_entropy'], 99),
                      np.percentile(dm['eigenspace_entropy'], 99))
    bins = np.linspace(0, max_entropy, 51)
    ax.hist(wb['eigenspace_entropy'], bins=bins, alpha=0.6, density=True,
           label='Well-behaved', color='green')
    ax.hist(dm['eigenspace_entropy'], bins=bins, alpha=0.6, density=True,
           label='Dark matter', color='red')
    ax.set_xlabel('Eigenspace Distribution Entropy', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Eigenspace Entropy:\n(Higher = More Diffuse)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Quadratic vs Linear R² improvement
    ax = axes[0, 2]
    r2_improvement_wb = wb['r2_quadratic'] - wb['r2_linear']
    r2_improvement_dm = dm['r2_quadratic'] - dm['r2_linear']
    valid_wb = np.isfinite(r2_improvement_wb)
    valid_dm = np.isfinite(r2_improvement_dm)

    if np.sum(valid_wb) > 0 and np.sum(valid_dm) > 0:
        bins = np.linspace(-0.1, 0.3, 51)
        ax.hist(r2_improvement_wb[valid_wb], bins=bins, alpha=0.6, density=True,
               label='Well-behaved', color='green')
        ax.hist(r2_improvement_dm[valid_dm], bins=bins, alpha=0.6, density=True,
               label='Dark matter', color='red')
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('R²(quadratic) - R²(linear)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Non-linearity Test:\n(Positive = Quadratic Fits Better)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Curvature comparison
    ax = axes[1, 0]
    valid_wb = np.isfinite(wb['curvature'])
    valid_dm = np.isfinite(dm['curvature'])
    if np.sum(valid_wb) > 0 and np.sum(valid_dm) > 0:
        curv_wb = wb['curvature'][valid_wb]
        curv_dm = dm['curvature'][valid_dm]
        max_curv = np.percentile(np.abs(np.concatenate([curv_wb, curv_dm])), 99)
        bins = np.linspace(-max_curv, max_curv, 51)
        ax.hist(curv_wb, bins=bins, alpha=0.6, density=True,
               label='Well-behaved', color='green')
        ax.hist(curv_dm, bins=bins, alpha=0.6, density=True,
               label='Dark matter', color='red')
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Quadratic Coefficient (Curvature)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Trajectory Curvature:\n(Neg = Concave Down)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Slope ratio (late/early)
    ax = axes[1, 1]
    valid_wb = np.isfinite(wb['slope_ratio']) & (wb['slope_ratio'] > 0) & (wb['slope_ratio'] < 10)
    valid_dm = np.isfinite(dm['slope_ratio']) & (dm['slope_ratio'] > 0) & (dm['slope_ratio'] < 10)
    if np.sum(valid_wb) > 0 and np.sum(valid_dm) > 0:
        bins = np.linspace(0, 5, 51)
        ax.hist(wb['slope_ratio'][valid_wb], bins=bins, alpha=0.6, density=True,
               label='Well-behaved', color='green')
        ax.hist(dm['slope_ratio'][valid_dm], bins=bins, alpha=0.6, density=True,
               label='Dark matter', color='red')
        ax.axvline(1, color='black', linestyle='--', linewidth=1, label='Constant slope')
    ax.set_xlabel('Slope Ratio (Late Phase / Early Phase)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Slope Evolution During Training:\n(1 = Constant, <1 = Decelerating)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Dominant eigenvalue comparison
    ax = axes[1, 2]
    max_eig = np.percentile(np.concatenate([wb['dominant_eigenvalue'],
                                             dm['dominant_eigenvalue']]), 99)
    bins = np.linspace(0, max_eig, 51)
    ax.hist(wb['dominant_eigenvalue'], bins=bins, alpha=0.6, density=True,
           label='Well-behaved', color='green')
    ax.hist(dm['dominant_eigenvalue'], bins=bins, alpha=0.6, density=True,
           label='Dark matter', color='red')
    ax.set_xlabel('Dominant Eigenvalue (λ)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Dominant Eigenvalue Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'dark_matter_characterization.png', dpi=150, bbox_inches='tight')
    print(f"Saved: dark_matter_characterization.png")
    plt.close()


def create_m_hidden_analysis(all_features, output_dir):
    """
    Analyze how dark matter fraction varies with m_hidden.
    """
    # Group by m_hidden
    m_values = sorted(set(f['m_hidden'] for f in all_features))

    dark_matter_fraction = []
    median_r2 = []
    sample_sizes = []

    for m in m_values:
        features_m = [f for f in all_features if f['m_hidden'] == m and f['is_deep_superposition']]
        if len(features_m) < 100:
            continue

        r2_values = [f['r2_linear'] for f in features_m]
        dark_matter_fraction.append(np.mean(np.array(r2_values) < 0.9) * 100)
        median_r2.append(np.median(r2_values))
        sample_sizes.append(len(features_m))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Dark matter fraction vs m
    ax = axes[0]
    ax.plot(m_values[:len(dark_matter_fraction)], dark_matter_fraction, 'o-', markersize=8)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax.set_xlabel('Hidden Dimension (m)', fontsize=12)
    ax.set_ylabel('% Features with R² < 0.9', fontsize=12)
    ax.set_title('Dark Matter Fraction vs Model Size', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Median R² vs m
    ax = axes[1]
    ax.plot(m_values[:len(median_r2)], median_r2, 'o-', markersize=8, color='green')
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='R² = 0.9')
    ax.set_xlabel('Hidden Dimension (m)', fontsize=12)
    ax.set_ylabel('Median R² of Deep Superposition Features', fontsize=12)
    ax.set_title('Linear Fit Quality vs Model Size', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'm_hidden_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: m_hidden_analysis.png")
    plt.close()

    return {
        'm_values': m_values[:len(dark_matter_fraction)],
        'dark_matter_fraction': dark_matter_fraction,
        'median_r2': median_r2,
        'sample_sizes': sample_sizes,
    }


def main():
    """Main analysis pipeline for dark matter investigation."""
    import time
    start_time = time.time()

    print("=" * 70)
    print("Dark Matter Investigation: Characterizing Non-Linear Trajectories")
    print("=" * 70)

    # Get high-sparsity files
    checkpoint_files = sorted(INPUT_DIR.glob('n1024_m*.h5'))
    high_sparsity_files = []

    for f in checkpoint_files:
        with h5py.File(f, 'r') as h5f:
            sparsity = float(h5f.attrs['sparsity'])
            if HIGH_SPARSITY_RANGE[0] <= sparsity <= HIGH_SPARSITY_RANGE[1]:
                svd_file = SVD_DIR / f"svd_{f.name}"
                if svd_file.exists():
                    high_sparsity_files.append((f, svd_file))

    print(f"Found {len(high_sparsity_files)} high-sparsity files with SVD results")

    # Sample files for detailed analysis (use all for complete picture)
    # For speed, we'll sample 100 files
    if len(high_sparsity_files) > 100:
        np.random.seed(42)
        indices = np.random.choice(len(high_sparsity_files), 100, replace=False)
        sampled_files = [high_sparsity_files[i] for i in indices]
    else:
        sampled_files = high_sparsity_files

    print(f"Analyzing {len(sampled_files)} files in detail")

    # Detailed analysis
    print("\n--- Detailed trajectory analysis ---")
    all_features = []

    for filepath, svd_filepath in tqdm(sampled_files, desc="Analyzing"):
        try:
            features = analyze_file_detailed(filepath, svd_filepath)
            all_features.extend(features)
        except Exception as e:
            print(f"\nError processing {filepath.name}: {e}")
            continue

    print(f"Analyzed {len(all_features)} features total")

    # Split into deep superposition features
    deep_features = [f for f in all_features if f['is_deep_superposition']]
    print(f"Deep superposition features: {len(deep_features)}")

    # Split into well-behaved and dark matter
    well_behaved = [f for f in deep_features if f['r2_linear'] >= 0.9]
    dark_matter = [f for f in deep_features if f['r2_linear'] < 0.9]

    print(f"\nWell-behaved (R² >= 0.9): {len(well_behaved)} ({100*len(well_behaved)/len(deep_features):.1f}%)")
    print(f"Dark matter (R² < 0.9): {len(dark_matter)} ({100*len(dark_matter)/len(deep_features):.1f}%)")

    # Create comparison plots
    print("\n--- Creating visualizations ---")
    create_dark_matter_comparison_plots(well_behaved, dark_matter, OUTPUT_DIR)

    # M-hidden analysis
    m_analysis = create_m_hidden_analysis(deep_features, OUTPUT_DIR)

    # Compute summary statistics
    print("\n--- Computing summary statistics ---")

    def compute_stats(features, name):
        r2_values = [f['r2_linear'] for f in features]
        conc_values = [f['max_concentration'] for f in features]
        entropy_values = [f['eigenspace_entropy'] for f in features]
        curvature_values = [f['curvature'] for f in features if np.isfinite(f['curvature'])]

        stats = {
            'name': name,
            'n': len(features),
            'r2_mean': np.mean(r2_values),
            'r2_median': np.median(r2_values),
            'r2_std': np.std(r2_values),
            'concentration_mean': np.mean(conc_values),
            'concentration_median': np.median(conc_values),
            'entropy_mean': np.mean(entropy_values),
            'entropy_median': np.median(entropy_values),
            'curvature_mean': np.mean(curvature_values) if curvature_values else np.nan,
            'curvature_std': np.std(curvature_values) if curvature_values else np.nan,
            'pct_positive_curvature': np.mean(np.array(curvature_values) > 0) * 100 if curvature_values else np.nan,
        }
        return stats

    wb_stats = compute_stats(well_behaved, 'well_behaved')
    dm_stats = compute_stats(dark_matter, 'dark_matter')

    # Statistical tests
    print("\n--- Statistical tests ---")

    # Mann-Whitney U test for concentration
    conc_wb = [f['max_concentration'] for f in well_behaved]
    conc_dm = [f['max_concentration'] for f in dark_matter]
    u_stat, p_conc = stats.mannwhitneyu(conc_wb, conc_dm, alternative='two-sided')
    print(f"Concentration difference p-value: {p_conc:.2e}")

    # Mann-Whitney U test for entropy
    ent_wb = [f['eigenspace_entropy'] for f in well_behaved]
    ent_dm = [f['eigenspace_entropy'] for f in dark_matter]
    u_stat, p_ent = stats.mannwhitneyu(ent_wb, ent_dm, alternative='two-sided')
    print(f"Entropy difference p-value: {p_ent:.2e}")

    # Save results
    results = {
        'well_behaved_stats': wb_stats,
        'dark_matter_stats': dm_stats,
        'statistical_tests': {
            'concentration_mannwhitney_pvalue': float(p_conc),
            'entropy_mannwhitney_pvalue': float(p_ent),
        },
        'm_hidden_analysis': m_analysis,
    }

    with open(OUTPUT_DIR / 'dark_matter_investigation.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x))

    print(f"\nSaved: dark_matter_investigation.json")

    # Print summary
    print("\n" + "=" * 70)
    print("DARK MATTER INVESTIGATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Well-Behaved':<15} {'Dark Matter':<15}")
    print("-" * 60)
    print(f"{'Count':<30} {wb_stats['n']:<15,} {dm_stats['n']:<15,}")
    print(f"{'R² (mean ± std)':<30} {wb_stats['r2_mean']:.3f} ± {wb_stats['r2_std']:.3f}    {dm_stats['r2_mean']:.3f} ± {dm_stats['r2_std']:.3f}")
    print(f"{'Eigenspace Concentration':<30} {wb_stats['concentration_mean']:.3f}           {dm_stats['concentration_mean']:.3f}")
    print(f"{'Eigenspace Entropy':<30} {wb_stats['entropy_mean']:.3f}           {dm_stats['entropy_mean']:.3f}")
    print(f"{'Curvature (mean)':<30} {wb_stats['curvature_mean']:.4f}          {dm_stats['curvature_mean']:.4f}")
    print(f"{'% Positive Curvature':<30} {wb_stats['pct_positive_curvature']:.1f}%           {dm_stats['pct_positive_curvature']:.1f}%")

    print(f"\nProcessing time: {time.time() - start_time:.1f} seconds")


if __name__ == '__main__':
    main()
