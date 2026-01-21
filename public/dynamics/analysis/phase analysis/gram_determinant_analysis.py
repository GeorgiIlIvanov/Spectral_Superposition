"""
Analysis of determinant det(W W^T - α I) and det(W W^T - I/α)
to check if the slope α relates to eigenvalues of the Gram matrix.

Note: We use W W^T (m x m) instead of W^T W (n x n) because:
- W has shape (m, n) with m << n typically
- W^T W is rank-deficient (rank m) with n-m zero eigenvalues
- W W^T has the same non-zero eigenvalues but is computationally tractable
- If α is an eigenvalue of W W^T, then det(W W^T - α I) ≈ 0
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
ANALYSIS_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis')
N_FEATURES = 1024
TARGET_STEPS = [400, 7000, 20000]
STEP_INDICES = {400: 2, 7000: 29, 20000: 50}

def compute_gram_determinants(W, alpha):
    """
    Compute det(W W^T - α I) and det(W W^T - (1/α) I)
    using log determinant for numerical stability.

    Returns: (log_det_alpha, sign_alpha, log_det_inv_alpha, sign_inv_alpha)
    """
    # W is (m, n), so W W^T is (m, m)
    gram = W @ W.T
    m = gram.shape[0]

    # Compute det(gram - α I)
    shifted_alpha = gram - alpha * np.eye(m)
    sign_alpha, logdet_alpha = np.linalg.slogdet(shifted_alpha)

    # Compute det(gram - (1/α) I)
    if np.abs(alpha) > 1e-10:
        shifted_inv_alpha = gram - (1.0 / alpha) * np.eye(m)
        sign_inv_alpha, logdet_inv_alpha = np.linalg.slogdet(shifted_inv_alpha)
    else:
        sign_inv_alpha, logdet_inv_alpha = 0, np.nan

    return logdet_alpha, sign_alpha, logdet_inv_alpha, sign_inv_alpha


def compute_eigenvalue_distances(W, alpha):
    """
    Compute minimum distance from α and 1/α to eigenvalues of W W^T.
    This is an alternative metric to check if α relates to eigenvalues.
    """
    gram = W @ W.T
    eigenvalues = np.linalg.eigvalsh(gram)

    min_dist_alpha = np.min(np.abs(eigenvalues - alpha))

    if np.abs(alpha) > 1e-10:
        min_dist_inv_alpha = np.min(np.abs(eigenvalues - 1.0/alpha))
    else:
        min_dist_inv_alpha = np.nan

    return min_dist_alpha, min_dist_inv_alpha, eigenvalues


def main():
    # Load slope data
    fit_data = np.load(ANALYSIS_DIR / 'linear_fit_arrays.npz')
    slopes = fit_data['slopes']  # shape: (32, 50, 3) = (m_vals, sparsity_vals, steps)
    m_hidden_vals = fit_data['m_hidden_vals']  # 32 values
    sparsity_vals = fit_data['sparsity_vals']  # 50 values

    n_m = len(m_hidden_vals)
    n_s = len(sparsity_vals)
    n_steps = len(TARGET_STEPS)

    # Compression ratios (m/n)
    compression_ratios = m_hidden_vals / N_FEATURES

    print(f"Analyzing {n_m} m_hidden values x {n_s} sparsity values x {n_steps} steps")
    print(f"m_hidden range: {m_hidden_vals[0]} to {m_hidden_vals[-1]}")
    print(f"sparsity range: {sparsity_vals[0]:.2f} to {sparsity_vals[-1]:.2f}")
    print(f"compression ratio range: {compression_ratios[0]:.3f} to {compression_ratios[-1]:.3f}")

    # Arrays to store results
    # Average over seeds
    logdet_alpha = np.zeros((n_m, n_s, n_steps))
    logdet_inv_alpha = np.zeros((n_m, n_s, n_steps))
    min_dist_alpha = np.zeros((n_m, n_s, n_steps))
    min_dist_inv_alpha = np.zeros((n_m, n_s, n_steps))

    total_experiments = n_m * n_s
    processed = 0

    for i_m, m in enumerate(m_hidden_vals):
        for i_s, s in enumerate(sparsity_vals):
            # Average over seeds
            logdet_alpha_seeds = []
            logdet_inv_alpha_seeds = []
            min_dist_alpha_seeds = []
            min_dist_inv_alpha_seeds = []

            for seed in [0, 1]:
                # Load weight file
                filename = f'n{N_FEATURES}_m{m}_s{s:.6f}_seed{seed}.h5'
                filepath = DATA_DIR / filename

                if not filepath.exists():
                    print(f"Warning: {filepath} not found")
                    continue

                with h5py.File(filepath, 'r') as f:
                    weights = f['weights'][:]  # shape: (56, m, n)

                    step_results_logdet_alpha = []
                    step_results_logdet_inv_alpha = []
                    step_results_min_dist_alpha = []
                    step_results_min_dist_inv_alpha = []

                    for i_step, step in enumerate(TARGET_STEPS):
                        idx = STEP_INDICES[step]
                        W = weights[idx]  # shape: (m, n)
                        alpha = slopes[i_m, i_s, i_step]

                        # Compute determinants
                        lda, _, ldia, _ = compute_gram_determinants(W, alpha)
                        step_results_logdet_alpha.append(lda)
                        step_results_logdet_inv_alpha.append(ldia)

                        # Compute eigenvalue distances
                        mda, mdia, _ = compute_eigenvalue_distances(W, alpha)
                        step_results_min_dist_alpha.append(mda)
                        step_results_min_dist_inv_alpha.append(mdia)

                    logdet_alpha_seeds.append(step_results_logdet_alpha)
                    logdet_inv_alpha_seeds.append(step_results_logdet_inv_alpha)
                    min_dist_alpha_seeds.append(step_results_min_dist_alpha)
                    min_dist_inv_alpha_seeds.append(step_results_min_dist_inv_alpha)

            # Average over seeds
            if logdet_alpha_seeds:
                logdet_alpha[i_m, i_s, :] = np.mean(logdet_alpha_seeds, axis=0)
                logdet_inv_alpha[i_m, i_s, :] = np.mean(logdet_inv_alpha_seeds, axis=0)
                min_dist_alpha[i_m, i_s, :] = np.mean(min_dist_alpha_seeds, axis=0)
                min_dist_inv_alpha[i_m, i_s, :] = np.mean(min_dist_inv_alpha_seeds, axis=0)

            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{total_experiments} experiments")

    print(f"Completed processing {processed} experiments")

    # Save results
    np.savez(ANALYSIS_DIR / 'gram_determinant_results.npz',
             logdet_alpha=logdet_alpha,
             logdet_inv_alpha=logdet_inv_alpha,
             min_dist_alpha=min_dist_alpha,
             min_dist_inv_alpha=min_dist_inv_alpha,
             m_hidden_vals=m_hidden_vals,
             sparsity_vals=sparsity_vals,
             compression_ratios=compression_ratios,
             target_steps=np.array(TARGET_STEPS))

    print(f"Results saved to {ANALYSIS_DIR / 'gram_determinant_results.npz'}")

    # Create plots
    create_heatmap_plots(logdet_alpha, logdet_inv_alpha, min_dist_alpha, min_dist_inv_alpha,
                         sparsity_vals, compression_ratios, TARGET_STEPS)


def create_heatmap_plots(logdet_alpha, logdet_inv_alpha, min_dist_alpha, min_dist_inv_alpha,
                         sparsity_vals, compression_ratios, target_steps):
    """
    Create heatmap plots with sparsity on x-axis and compression ratio on y-axis.
    """

    # Plot 1: log det(W W^T - α I) heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(r'$\log|\det(W W^T - \alpha I)|$ where $\alpha$ is the slope of $D_i$ vs $||W_i||^2$', fontsize=14)

    for i, (ax, step) in enumerate(zip(axes, target_steps)):
        data = logdet_alpha[:, :, i]  # shape: (n_m, n_s)

        # Handle NaN/inf values
        data = np.nan_to_num(data, nan=0, posinf=np.nanmax(data[np.isfinite(data)]),
                            neginf=np.nanmin(data[np.isfinite(data)]))

        im = ax.imshow(data, aspect='auto', origin='lower',
                      extent=[sparsity_vals[0], sparsity_vals[-1],
                              compression_ratios[0], compression_ratios[-1]],
                      cmap='viridis')
        ax.set_xlabel('Sparsity', fontsize=12)
        ax.set_ylabel('Compression Ratio (m/n)', fontsize=12)
        ax.set_title(f'Step {step}', fontsize=12)
        plt.colorbar(im, ax=ax, label=r'$\log|\det|$')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'gram_logdet_alpha.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {ANALYSIS_DIR / 'gram_logdet_alpha.png'}")

    # Plot 2: log det(W W^T - (1/α) I) heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(r'$\log|\det(W W^T - I/\alpha)|$ where $\alpha$ is the slope of $D_i$ vs $||W_i||^2$', fontsize=14)

    for i, (ax, step) in enumerate(zip(axes, target_steps)):
        data = logdet_inv_alpha[:, :, i]
        data = np.nan_to_num(data, nan=0, posinf=np.nanmax(data[np.isfinite(data)]),
                            neginf=np.nanmin(data[np.isfinite(data)]))

        im = ax.imshow(data, aspect='auto', origin='lower',
                      extent=[sparsity_vals[0], sparsity_vals[-1],
                              compression_ratios[0], compression_ratios[-1]],
                      cmap='viridis')
        ax.set_xlabel('Sparsity', fontsize=12)
        ax.set_ylabel('Compression Ratio (m/n)', fontsize=12)
        ax.set_title(f'Step {step}', fontsize=12)
        plt.colorbar(im, ax=ax, label=r'$\log|\det|$')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'gram_logdet_inv_alpha.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {ANALYSIS_DIR / 'gram_logdet_inv_alpha.png'}")

    # Plot 3: Minimum eigenvalue distance to α
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(r'$\min_i |\lambda_i - \alpha|$ (distance from $\alpha$ to nearest eigenvalue of $W W^T$)', fontsize=14)

    for i, (ax, step) in enumerate(zip(axes, target_steps)):
        data = min_dist_alpha[:, :, i]
        data = np.nan_to_num(data, nan=0)

        im = ax.imshow(data, aspect='auto', origin='lower',
                      extent=[sparsity_vals[0], sparsity_vals[-1],
                              compression_ratios[0], compression_ratios[-1]],
                      cmap='viridis')
        ax.set_xlabel('Sparsity', fontsize=12)
        ax.set_ylabel('Compression Ratio (m/n)', fontsize=12)
        ax.set_title(f'Step {step}', fontsize=12)
        plt.colorbar(im, ax=ax, label=r'$\min|\lambda_i - \alpha|$')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'gram_min_dist_alpha.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {ANALYSIS_DIR / 'gram_min_dist_alpha.png'}")

    # Plot 4: Minimum eigenvalue distance to 1/α
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(r'$\min_i |\lambda_i - 1/\alpha|$ (distance from $1/\alpha$ to nearest eigenvalue of $W W^T$)', fontsize=14)

    for i, (ax, step) in enumerate(zip(axes, target_steps)):
        data = min_dist_inv_alpha[:, :, i]
        data = np.nan_to_num(data, nan=0)

        im = ax.imshow(data, aspect='auto', origin='lower',
                      extent=[sparsity_vals[0], sparsity_vals[-1],
                              compression_ratios[0], compression_ratios[-1]],
                      cmap='viridis')
        ax.set_xlabel('Sparsity', fontsize=12)
        ax.set_ylabel('Compression Ratio (m/n)', fontsize=12)
        ax.set_title(f'Step {step}', fontsize=12)
        plt.colorbar(im, ax=ax, label=r'$\min|\lambda_i - 1/\alpha|$')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'gram_min_dist_inv_alpha.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {ANALYSIS_DIR / 'gram_min_dist_inv_alpha.png'}")

    # Create combined figure with both log det plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(r'Gram Matrix Determinant Analysis: Testing if slope $\alpha$ relates to eigenvalues', fontsize=14)

    # Top row: log det(W W^T - α I)
    for i, (ax, step) in enumerate(zip(axes[0], target_steps)):
        data = logdet_alpha[:, :, i]
        data = np.nan_to_num(data, nan=0, posinf=np.nanmax(data[np.isfinite(data)]),
                            neginf=np.nanmin(data[np.isfinite(data)]))

        im = ax.imshow(data, aspect='auto', origin='lower',
                      extent=[sparsity_vals[0], sparsity_vals[-1],
                              compression_ratios[0], compression_ratios[-1]],
                      cmap='viridis')
        ax.set_xlabel('Sparsity', fontsize=11)
        ax.set_ylabel('Compression (m/n)', fontsize=11)
        ax.set_title(f'Step {step}: ' + r'$\log|\det(WW^T - \alpha I)|$', fontsize=11)
        plt.colorbar(im, ax=ax)

    # Bottom row: log det(W W^T - (1/α) I)
    for i, (ax, step) in enumerate(zip(axes[1], target_steps)):
        data = logdet_inv_alpha[:, :, i]
        data = np.nan_to_num(data, nan=0, posinf=np.nanmax(data[np.isfinite(data)]),
                            neginf=np.nanmin(data[np.isfinite(data)]))

        im = ax.imshow(data, aspect='auto', origin='lower',
                      extent=[sparsity_vals[0], sparsity_vals[-1],
                              compression_ratios[0], compression_ratios[-1]],
                      cmap='viridis')
        ax.set_xlabel('Sparsity', fontsize=11)
        ax.set_ylabel('Compression (m/n)', fontsize=11)
        ax.set_title(f'Step {step}: ' + r'$\log|\det(WW^T - I/\alpha)|$', fontsize=11)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'gram_logdet_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {ANALYSIS_DIR / 'gram_logdet_combined.png'}")


if __name__ == '__main__':
    main()
