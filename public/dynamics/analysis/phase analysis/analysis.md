1. Locate and parse files from '../start/*.h5' to make sure all of them are present
2. Completion Heatmap - verify that all seeds for all m_hidden and sparsity are present
3. Validate and Load data - check for corruptions, such as not all elements in the list being present or NaN/Inf's 
4. Plot  histograms for different bins of sparstiy and model capacity  of $D_i = \frac{(M_{ii})^2}{(M^2)_{ii}}$, where M = W^\intercal W is the Gram matrix and $D_i$ is the fractional dimensionality. Note, this should already be present as 'fractonal_dims' in the data. 
5. Plot a 3D evolution of the histogram throughout the training using the  checkpoints - more specifically, the xz planes should be histograms at individual checkpoints y = training steps
6. Create a rectangular grid where the x-axis is sparsity and the y-axis is compression ratio (m_hidden/1024) and each bin for the corresponding m_hidden and sparsity should be colored with the following linear gradient w.r.t to fractional dimensionality:
function hexToGradientValue(hex) {
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  
  // Use blue channel (larger dynamic range)
  return (240 - b) / 240;
}
6. Plot a heatmap (m vs Sparsity) of the mean D_i and std D_i
7. Plot a superposition phase diagram. The one attached below is for a different set of experiments, so repurpose it for this specific set:

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.colors as colors

def collect_spectral_data(results_dir='results', subsample_prob=0.1):
    """
    Iterates over all HDF5 files and collects feature statistics.
    
    Args:
        results_dir: Directory containing the .h5 files.
        subsample_prob: Probability of keeping a specific feature (to reduce plot size).
                        Set to 1.0 to plot all points (warning: can be slow with >1M points).
    """
    
    # Store lists of individual feature stats
    all_norms = []
    all_dims = []
    all_rhos = []  # Compression ratio (n/m) for coloring
    
    # Find all h5 files
    files = list(Path(results_dir).glob('*.h5'))
    print(f"Found {len(files)} experiment files. Processing...")
    
    for fpath in tqdm(files):
        try:
            with h5py.File(fpath, 'r') as f:
                # Load configuration
                n = f.attrs['n_features']
                m = f.attrs['m_hidden']
                rho = n / m  # Compression ratio
                
                # Load final checkpoint stats
                # fractional_dims shape: (checkpoints, n_features)
                # We take the last checkpoint [-1]
                final_dims = f['fractional_dims'][-1]
                final_norms = f['feature_norms'][-1]
                
                # Random subsampling to keep plot manageable
                if subsample_prob < 1.0:
                    mask = np.random.rand(len(final_dims)) < subsample_prob
                    final_dims = final_dims[mask]
                    final_norms = final_norms[mask]
                
                # Append to lists
                all_norms.append(final_norms)
                all_dims.append(final_dims)
                # Repeat rho for every feature in this experiment
                all_rhos.append(np.full(len(final_norms), rho))
                
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue
            
    # Concatenate all features into single arrays
    if not all_norms:
        print("No data collected.")
        return None, None, None

    print("Concatenating data...")
    X = np.concatenate(all_norms)
    Y = np.concatenate(all_dims)
    C = np.concatenate(all_rhos)
    
    return X, Y, C

def plot_spectral_phase_diagram(X, Y, C):
    """
    Plots Feature Norm vs Fractional Dimensionality, colored by Compression Ratio.
    """
    print(f"Plotting {len(X)} features...")
    
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    
    # Use a colormap that highlights the transition
    cmap = plt.get_cmap('turbo') 
    
    # Scatter plot
    # s=1: small points to see structure
    # alpha: transparency to show density
    sc = ax.scatter(X, Y, c=C, cmap=cmap, s=2, alpha=0.3, rasterized=True)
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Compression Ratio (n features / m hidden)', fontsize=12, weight='bold')
    
    # Labels and Titles
    ax.set_xlabel(r'Feature Norm $||W_i||^2$', fontsize=12)
    ax.set_ylabel(r'Fractional Dimensionality $D_i$', fontsize=12)
    ax.set_title('Spectral Superposition: Phase Diagram', fontsize=16, weight='bold')
    
    # Interpretation lines (Guides for the eye)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='D = Norm (Orthogonal Limit)')
    
    ax.grid(True, which='both', alpha=0.2)
    
    # Zoom in on the interesting part (optional, adjust based on your data)
    # Most interesting "rays" usually appear for norms < 1.5
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1.1)
    
    plt.tight_layout()
    plt.savefig('spectral_phase_diagram.png')
    print("Saved plot to 'spectral_phase_diagram.png'")
    plt.show()

if __name__ == "__main__":
    # Adjust directory if your .h5 files are elsewhere
    RESULTS_DIR = "results" 
    
    # 1.0 = use all data (might be heavy), 0.1 = use 10%
    SUBSAMPLE = 0.2 
    
    X, Y, C = collect_spectral_data(RESULTS_DIR, SUBSAMPLE)
    
    if X is not None:
        plot_spectral_phase_diagram(X, Y, C)
8. Plot the spectral phase diagram in angular coordinates. As above, this is from a previous set of experiments. 
def plot_polar_spectral_rays(X, Y, S):
    """
    Plots the spectral data in Polar Coordinates to visualize the 'ray' quantization.
    
    Coordinate Transformation:
    - Radius r = ||W_i|| (calculated as sqrt(X) since X is ||W_i||^2)
    - Angle θ = arctan(D_i / ||W_i||^2) = arctan(Y / X)
    
    If the 'Spectral Ansatz' holds (D_i ≈ ||W_i||^2 / μ), features should cluster 
    around specific angles corresponding to discrete eigenvalues μ.
    """
    
    # 1. Transform Coordinates
    # Filter out zero-norm features to avoid division errors
    mask = X > 1e-6
    X_clean = X[mask]
    Y_clean = Y[mask]
    S_clean = S[mask]
    
    # Radius = ||W_i||
    r = np.sqrt(X_clean)
    
    # Angle = arctan(D_i / ||W_i||^2)
    # This captures the slope of the rays. Constant slope = Constant Angle.
    theta = np.arctan2(Y_clean, X_clean)
    
    # 2. Setup Plot
    fig = plt.figure(figsize=(14, 8))
    
    # --- Subplot 1: Polar Projection ---
    # This visualizes the physical 'rays' radiating from the origin
    ax1 = fig.add_subplot(121, projection='polar')
    
    # Scatter plot
    # theta maps to azimuth, r maps to radius
    sc1 = ax1.scatter(theta, r, c=S_clean, cmap='viridis', s=2, alpha=0.5)
    
    ax1.set_title("Spectral Quantization (Polar Projection)\n$r=\|W_i\|, \\theta=\\arctan(D_i / \|W_i\|^2)$", pad=20)
    ax1.set_thetamin(0)
    ax1.set_thetamax(90) # Data is in first quadrant
    ax1.set_xlabel("Spectral Angle $\\theta$")
    
    # --- Subplot 2: Unrolled Quantization (Histogram) ---
    # If quantization is strict, we should see sharp peaks in the distribution of Theta
    ax2 = fig.add_subplot(122)
    
    # We plot a histogram of the angles to see the density of the rays
    counts, bins, _ = ax2.hist(theta, bins=200, color='k', alpha=0.7, density=True)
    
    ax2.set_title("Density of Spectral Angles")
    ax2.set_xlabel("Angle $\\theta$ (rad)")
    ax2.set_ylabel("Density of Features")
    ax2.grid(True, alpha=0.3)
    
    # Shared Colorbar
    cbar = fig.colorbar(sc1, ax=[ax1, ax2], fraction=0.02, pad=0.05)
    cbar.set_label('Sparsity $S$', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Run the plot using variables from your previous cells
# Assumes X (squared norms), Y (frac dims), and S (sparsity) are loaded
if 'X' in locals() and 'Y' in locals():
    print("Generating Polar Spectral Analysis...")
    plot_polar_spectral_rays(X, Y, S)
else:
    print("Error: Data variables X, Y, S not found in environment.")

9. Plot graphs for the evolution of structure metrics during training - angular entropy, concentration near reference angles, training loss and fraction of features with ||W_i||^2>0.01
