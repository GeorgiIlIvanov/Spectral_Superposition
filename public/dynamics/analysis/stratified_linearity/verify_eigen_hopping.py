import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from tqdm import tqdm

# --- Configuration (Matches your existing paths) ---
INPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/stratified_linearity')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis Parameters
SPARSITY_RANGE = (0.9, 1.0)      # Where "Dark Matter" is prevalent
R2_THRESHOLD_DARK = 0.8          # Treat features with R^2 < 0.8 as "Dark Matter"
R2_THRESHOLD_STABLE = 0.98       # Treat features with R^2 > 0.98 as "Stable"/Reference
NORM_DELTA_THRESHOLD = 0.05      # Minimum change in norm to calculate a slope (avoids noise)

def get_instantaneous_slopes(norms_sq, dims):
    """
    Calculates the slope (d_dim / d_norm_sq) between consecutive checkpoints.
    Only considers steps where the feature actually grew/shrank significantly.
    """
    # Calculate deltas
    delta_norms = np.diff(norms_sq)
    delta_dims = np.diff(dims)
    
    # Filter for meaningful changes in norm (avoid division by near-zero)
    significant_mask = np.abs(delta_norms) > NORM_DELTA_THRESHOLD
    
    if np.sum(significant_mask) == 0:
        return np.array([])
        
    # Calculate instantaneous slopes: ΔD / Δ||W||²
    instantaneous_slopes = delta_dims[significant_mask] / delta_norms[significant_mask]
    
    # Filter outliers (optional, keeps histogram clean)
    valid_mask = (instantaneous_slopes > -0.5) & (instantaneous_slopes < 2.0)
    return instantaneous_slopes[valid_mask]

def collect_slope_statistics():
    print(f"Scanning files in {INPUT_DIR} with Sparsity {SPARSITY_RANGE}...")
    
    stable_global_slopes = []       # Slopes of well-behaved features (The "Truth")
    dark_matter_instant_slopes = [] # Instantaneous slopes of messy features (The "Hypothesis")
    
    files = sorted([f for f in INPUT_DIR.glob('*.h5') if 's0.9' in f.name])
    
    for filepath in tqdm(files[:100]): # Limit to 100 files for speed, increase if needed
        # Parse sparsity from filename to filter
        try:
            sparsity = float(filepath.name.split('_s')[1].split('_')[0])
            if not (SPARSITY_RANGE[0] <= sparsity < SPARSITY_RANGE[1]):
                continue
        except:
            continue
            
        with h5py.File(filepath, 'r') as f:
            # Load Data
            # Shape: (n_checkpoints, n_features)
            frac_dims = f['fractional_dims'][:]
            norms = f['feature_norms'][:]
            
            # Use squared norms for linearity check
            norms_sq = norms ** 2
            
            n_features = frac_dims.shape[1]
            
            for i in range(n_features):
                feat_dims = frac_dims[:, i]
                feat_norms = norms_sq[:, i]
                
                # Check for dead features
                if np.max(feat_norms) < 0.1:
                    continue
                
                # 1. Calculate Global Fit
                slope, intercept, r_value, p_value, std_err = stats.linregress(feat_norms, feat_dims)
                r_squared = r_value ** 2
                
                # 2. Classify & Collect
                if r_squared > R2_THRESHOLD_STABLE:
                    # This is a "Reference" feature - its global slope is the eigenvalue
                    stable_global_slopes.append(slope)
                    
                elif r_squared < R2_THRESHOLD_DARK:
                    # This is "Dark Matter" - we want its instantaneous movements
                    inst_slopes = get_instantaneous_slopes(feat_norms, feat_dims)
                    dark_matter_instant_slopes.extend(inst_slopes)

    return np.array(stable_global_slopes), np.array(dark_matter_instant_slopes)

def plot_results(stable, dark):
    plt.figure(figsize=(12, 6))
    
    # Plot 1: The Stable Features (Ground Truth Eigenvalues)
    plt.hist(stable, bins=100, range=(0, 1.2), density=True, alpha=0.5, 
             color='blue', label='Stable Features (Global Slope)')
             
    # Plot 2: The Dark Matter (Instantaneous Slopes)
    plt.hist(dark, bins=100, range=(0, 1.2), density=True, alpha=0.5, 
             color='red', label='Dark Matter (Instantaneous Slopes)')
    
    plt.title('Evidence of Eigen-Hopping: Aligning Instantaneous vs Global Slopes')
    plt.xlabel('Slope (ΔD / Δ||W||²)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = OUTPUT_DIR / 'eigen_hopping_confirmation.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    stable, dark = collect_slope_statistics()
    print(f"Stats: Analyzed {len(stable)} stable features and {len(dark)} dark matter segments.")
    plot_results(stable, dark)
