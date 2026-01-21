"""
Linear fit analysis: D_i vs ||W_i||^2 across sparsities and hidden dimensions.

For each (sparsity, m_hidden) combination, fits D_i = slope * ||W_i||^2 + intercept
at steps 400, 7000, and 20000, saving slope and R² variance.
"""

import h5py
import numpy as np
from pathlib import Path
from scipy import stats
import json
from tqdm import tqdm

def get_step_index(checkpoint_steps, target_step):
    """Find the index of the target step in checkpoint_steps array."""
    idx = np.where(checkpoint_steps == target_step)[0]
    if len(idx) == 0:
        # Find closest step
        idx = np.argmin(np.abs(checkpoint_steps - target_step))
        return idx, checkpoint_steps[idx]
    return idx[0], target_step

def linear_fit(x, y):
    """Perform linear regression and return slope, intercept, R², and residual variance."""
    # Filter out any NaN or Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return np.nan, np.nan, np.nan, np.nan

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    r_squared = r_value ** 2

    # Compute residual variance
    y_pred = slope * x_clean + intercept
    residuals = y_clean - y_pred
    residual_var = np.var(residuals)

    return slope, intercept, r_squared, residual_var

def analyze_checkpoint(filepath, target_steps):
    """Analyze a single checkpoint file for the given target steps."""
    results = {}

    with h5py.File(filepath, 'r') as f:
        checkpoint_steps = f['checkpoint_steps'][:]
        fractional_dims = f['fractional_dims'][:]  # (56, 1024)
        feature_norms = f['feature_norms'][:]      # (56, 1024)

        m_hidden = f.attrs['m_hidden']
        sparsity = f.attrs['sparsity']
        seed = f.attrs['seed']

    for target_step in target_steps:
        step_idx, actual_step = get_step_index(checkpoint_steps, target_step)

        D_i = fractional_dims[step_idx]  # (1024,)
        W_norm_sq = feature_norms[step_idx]  # (1024,)

        slope, intercept, r_squared, residual_var = linear_fit(W_norm_sq, D_i)

        results[target_step] = {
            'actual_step': int(actual_step),
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'residual_variance': float(residual_var)
        }

    return {
        'm_hidden': int(m_hidden),
        'sparsity': float(sparsity),
        'seed': int(seed),
        'fits': results
    }

def main():
    start_dir = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
    output_dir = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis')

    target_steps = [400, 7000, 20000]

    # Get all checkpoint files
    checkpoint_files = sorted(start_dir.glob('n1024_m*.h5'))
    print(f"Found {len(checkpoint_files)} checkpoint files")

    all_results = []

    for filepath in tqdm(checkpoint_files, desc="Analyzing checkpoints"):
        try:
            result = analyze_checkpoint(filepath, target_steps)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")
            continue

    # Save raw results
    with open(output_dir / 'linear_fit_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Aggregate results by (m_hidden, sparsity) - averaging over seeds
    aggregated = {}
    for r in all_results:
        key = (r['m_hidden'], r['sparsity'])
        if key not in aggregated:
            aggregated[key] = {step: [] for step in target_steps}
        for step in target_steps:
            aggregated[key][step].append(r['fits'][step])

    # Compute mean values across seeds
    final_results = []
    for (m_hidden, sparsity), step_data in aggregated.items():
        entry = {
            'm_hidden': m_hidden,
            'sparsity': sparsity,
            'fits': {}
        }
        for step in target_steps:
            fits = step_data[step]
            entry['fits'][step] = {
                'slope_mean': np.nanmean([f['slope'] for f in fits]),
                'slope_std': np.nanstd([f['slope'] for f in fits]),
                'r_squared_mean': np.nanmean([f['r_squared'] for f in fits]),
                'r_squared_std': np.nanstd([f['r_squared'] for f in fits]),
                'residual_variance_mean': np.nanmean([f['residual_variance'] for f in fits]),
                'residual_variance_std': np.nanstd([f['residual_variance'] for f in fits]),
            }
        final_results.append(entry)

    # Sort by m_hidden, then sparsity
    final_results.sort(key=lambda x: (x['m_hidden'], x['sparsity']))

    # Save aggregated results
    with open(output_dir / 'linear_fit_aggregated.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # Also save as NumPy arrays for easier analysis
    m_hidden_vals = sorted(set(r['m_hidden'] for r in final_results))
    sparsity_vals = sorted(set(r['sparsity'] for r in final_results))

    # Create arrays: shape (n_m_hidden, n_sparsity, n_steps)
    n_m = len(m_hidden_vals)
    n_s = len(sparsity_vals)
    n_steps = len(target_steps)

    slopes = np.zeros((n_m, n_s, n_steps))
    r_squared = np.zeros((n_m, n_s, n_steps))
    residual_var = np.zeros((n_m, n_s, n_steps))

    m_idx_map = {m: i for i, m in enumerate(m_hidden_vals)}
    s_idx_map = {s: i for i, s in enumerate(sparsity_vals)}

    for r in final_results:
        mi = m_idx_map[r['m_hidden']]
        si = s_idx_map[r['sparsity']]
        for ti, step in enumerate(target_steps):
            slopes[mi, si, ti] = r['fits'][step]['slope_mean']
            r_squared[mi, si, ti] = r['fits'][step]['r_squared_mean']
            residual_var[mi, si, ti] = r['fits'][step]['residual_variance_mean']

    np.savez(
        output_dir / 'linear_fit_arrays.npz',
        slopes=slopes,
        r_squared=r_squared,
        residual_variance=residual_var,
        m_hidden_vals=np.array(m_hidden_vals),
        sparsity_vals=np.array(sparsity_vals),
        target_steps=np.array(target_steps)
    )

    print(f"\nResults saved to:")
    print(f"  - {output_dir / 'linear_fit_results.json'} (raw)")
    print(f"  - {output_dir / 'linear_fit_aggregated.json'} (aggregated)")
    print(f"  - {output_dir / 'linear_fit_arrays.npz'} (numpy arrays)")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"  m_hidden values: {len(m_hidden_vals)} ({min(m_hidden_vals)} to {max(m_hidden_vals)})")
    print(f"  sparsity values: {len(sparsity_vals)} ({min(sparsity_vals):.3f} to {max(sparsity_vals):.3f})")
    print(f"  target steps: {target_steps}")

    for ti, step in enumerate(target_steps):
        print(f"\n  Step {step}:")
        print(f"    Slope range: {np.nanmin(slopes[:,:,ti]):.4f} to {np.nanmax(slopes[:,:,ti]):.4f}")
        print(f"    R² range: {np.nanmin(r_squared[:,:,ti]):.4f} to {np.nanmax(r_squared[:,:,ti]):.4f}")
        print(f"    Mean R²: {np.nanmean(r_squared[:,:,ti]):.4f}")

if __name__ == '__main__':
    main()
