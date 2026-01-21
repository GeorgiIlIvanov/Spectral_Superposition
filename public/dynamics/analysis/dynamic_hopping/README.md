# Dynamic Feature Hopping Analysis

**Goal**: Quantify and visualize "dynamic hopping" of features during training via time-variation of the Rayleigh quotient.

## Mathematical Foundation

### The Rayleigh Quotient

For a weight matrix W(t) ∈ ℝ^{d×n} at checkpoint t, where:
- d = hidden dimension (e.g., 112)
- n = number of features (e.g., 1024)
- w_i(t) = column i of W(t), the i-th feature vector

The **Rayleigh quotient** for feature i at time t is defined as:

```
κ_i(t) = (w_i(t)^T S(t) w_i(t)) / ||w_i(t)||²
```

where **S(t) = W(t) W(t)^T** is the feature covariance matrix.

### Simplified Computation

Since S = W W^T, we can simplify:

```
κ_i = w_i^T (W W^T) w_i / ||w_i||²
    = ||W^T w_i||² / ||w_i||²
    = Σ_j (w_j^T w_i)² / ||w_i||²
```

This measures how much feature i "overlaps" with all other features relative to its own magnitude.

### Derived Quantities

| Symbol | Definition | Interpretation |
|--------|------------|----------------|
| a_i(t) | \|\|w_i(t)\|\|² | Squared L2 norm of feature i |
| κ_i(t) | Rayleigh quotient | Feature's participation in covariance structure |
| x_i(t) | log(κ_i(t) + ε) | Log-transformed quotient (for numerical stability) |
| D_i(t) | a_i(t) / κ_i(t) | Related to effective dimension |

### Jump Detection

A "jump" indicates sudden change in feature behavior. We detect jumps using **robust statistics**:

1. **Compute differences**: Δx_i(t) = x_i(t) - x_i(t-1)

2. **Robust standard deviation** using Median Absolute Deviation (MAD):
   ```
   σ_robust = 1.4826 × median(|Δx - median(Δx)|)
   ```
   The factor 1.4826 makes σ_robust equivalent to standard deviation for Gaussian data.

3. **Jump threshold**: A jump occurs when |Δx_i(t)| > z × σ_robust
   - Default z = 4.0 (very significant deviations only)

## Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| ε | 1e-12 | Guard for log/division operations |
| ε_MAD | 1e-9 | Guard for MAD computation |
| z | 4.0 | Jump threshold in σ_robust units |
| late_window | Last 20% | Fraction of checkpoints for late-training analysis |

## File Structure

```
dynamic_hopping/
├── README.md                      # This documentation
├── 00_config.py                   # Configuration constants
├── config_loader.py               # Configuration loading utility
├── 01_rayleigh_quotients.py       # Compute κ_i(t) for all features
├── 02_jump_detection.py           # Detect jumps using robust statistics
├── 03_temporal_patterns.py        # Analyze temporal structure
├── 04_visualizations.py           # Generate all plots
├── run_all.sh                     # Pipeline runner script
├── results/                       # Output data
│   ├── per_file/                  # Per-experiment Rayleigh data
│   ├── rayleigh_summary.json      # Aggregate Rayleigh statistics
│   ├── jump_detection_results.json
│   └── temporal_patterns_results.json
└── plots/                         # Generated visualizations
```

## Scripts Explained

### 00_config.py
Defines all analysis constants in one place:
- Numerical guards (ε, ε_MAD)
- Jump detection threshold (z=4.0)
- Sparsity buckets for stratified analysis
- File paths
- GPU configuration

### 01_rayleigh_quotients.py
**Purpose**: Core computation of Rayleigh quotients κ_i(t).

**Algorithm**:
1. Load weight matrix W(t) of shape (T, d, n)
2. For each checkpoint t:
   - Compute Gram matrix: M = W^T W (shape n×n)
   - For each feature i: κ_i = ||M[:,i]||² / ||w_i||²
3. Compute log-transformed x_i = log(κ_i + ε)

**Outputs**:
- `per_file/<experiment>_rayleigh.npz`: Arrays κ, a, x per experiment
- `rayleigh_summary.json`: Aggregate statistics

**GPU Acceleration**: Uses PyTorch for batched matrix operations.

### 02_jump_detection.py
**Purpose**: Detect sudden changes in feature behavior.

**Algorithm**:
1. Compute differences Δx_i(t) = x_i(t) - x_i(t-1)
2. Calculate robust σ using MAD
3. Flag jumps where |Δx| > z × σ
4. Analyze jump distribution across:
   - Features (which features hop most?)
   - Time (when do jumps occur?)
   - Direction (positive vs negative jumps)

**Late Window Analysis**: Compares hopping in early vs late training to detect if hopping settles down.

**Outputs**:
- `jump_detection_results.json`: Complete jump statistics

### 03_temporal_patterns.py
**Purpose**: Analyze temporal structure of hopping behavior.

**Feature Classification** by volatility:
| Class | Criterion | Interpretation |
|-------|-----------|----------------|
| stable | σ < 0.5 | Predictable, low hopping |
| moderate | 0.5 ≤ σ < 1.5 | Some variation |
| active | 1.5 ≤ σ < 3.0 | Significant hopping |
| extreme | σ ≥ 3.0 | Highly volatile |

**Temporal Pattern Classification**:
| Pattern | Criterion | Interpretation |
|---------|-----------|----------------|
| converging | late_vol < 0.7 × early_vol | Settling down |
| diverging | late_vol > 1.5 × early_vol | Becoming chaotic |
| steady | Consistent volatility | Stable behavior |
| episodic | mid_vol > max(early, late) | Bursts of activity |

**Synchrony Analysis**: Measures if features hop together (synchronized) or independently.

**Outputs**:
- `temporal_patterns_results.json`: Classifications and synchrony metrics

### 04_visualizations.py
**Purpose**: Generate comprehensive plots.

**Individual File Plots**:
1. **Rayleigh Heatmap**: x_i(t) over time for all features (sorted by mean κ)
2. **Jump Events**: Scatter plot of when/where jumps occur
3. **Volatility Evolution**: How volatility changes during training

**Aggregate Plots**:
1. **Feature Classification Summary**: Distribution of volatility classes by sparsity
2. **Sparsity Comparison**: How metrics vary with sparsity
3. **Summary Figure**: Combined view of key results

## Usage

### Quick Start
```bash
# Run full pipeline
./run_all.sh

# Test with small sample
./run_all.sh --sample 10

# Use specific GPU
./run_all.sh --gpu 2
```

### Individual Steps
```bash
PYTHON=/home/georgi/Spectral_Superposition/public/dynamics/venv/bin/python

# Step 1: Compute Rayleigh quotients
$PYTHON 01_rayleigh_quotients.py --gpu 0

# Step 2: Detect jumps
$PYTHON 02_jump_detection.py

# Step 3: Temporal patterns
$PYTHON 03_temporal_patterns.py

# Step 4: Visualizations
$PYTHON 04_visualizations.py --sample-plots 10
```

## Outputs

### Results Files

**rayleigh_summary.json**
```json
{
  "n_files_processed": 200,
  "per_file_summary": [
    {
      "filename": "n1024_m112_s0.5_seed0",
      "sparsity": 0.5,
      "norm_verification_diff": 1e-6
    }
  ]
}
```

**jump_detection_results.json**
```json
{
  "summary_by_sparsity": {
    "low": {
      "mean_total_jumps": 150.2,
      "mean_sigma_global": 0.032
    }
  },
  "per_file_results": [...]
}
```

**temporal_patterns_results.json**
```json
{
  "summary_by_sparsity": {
    "low": {
      "volatility_class_means": {"stable": 800, "active": 50},
      "temporal_pattern_means": {"converging": 600, "steady": 300}
    }
  }
}
```

### Plot Descriptions

| Plot | Description |
|------|-------------|
| `*_rayleigh_heatmap.png` | 2D heatmap: x-axis=time, y-axis=features, color=log(κ) |
| `*_jump_events.png` | 4-panel: jump scatter, jumps/time, jumps/feature histogram, magnitude distribution |
| `*_volatility_evolution.png` | Rolling volatility over training with early/late comparison |
| `feature_classification_summary.png` | Bar charts of feature classifications by sparsity |
| `sparsity_comparison.png` | Scatter plots of metrics vs sparsity |
| `dynamic_hopping_summary.png` | 9-panel summary of all key metrics |

## Interpretation Guide

### What is "Hopping"?
A feature "hops" when its relationship to other features changes suddenly. High κ_i means feature i overlaps strongly with others; a sudden drop means it became more independent.

### Key Questions Answered

1. **How much hopping occurs?**
   - Check `mean_jumps_per_feature` and `total_jumps`
   - Higher values = more dynamic behavior

2. **Does hopping increase or decrease during training?**
   - Check `late_sigma_ratio`:
     - < 1: Hopping decreases (converging to stable)
     - > 1: Hopping increases (becoming chaotic)
     - ≈ 1: Consistent hopping

3. **Are features hopping together?**
   - Check `synchrony_ratio`:
     - > 1.5: Synchronized hopping (features move together)
     - < 0.7: Anti-synchronized
     - ≈ 1: Independent hopping

4. **How does sparsity affect hopping?**
   - Compare metrics across sparsity buckets
   - Higher sparsity often shows different dynamics

## Sparsity Buckets

| Bucket | Sparsity Range | Description |
|--------|----------------|-------------|
| low | [0.0, 0.3) | Dense features, high overlap |
| medium | [0.3, 0.7) | Moderate sparsity |
| high | [0.7, 0.95) | Sparse features |
| extreme | [0.95, 1.0) | Very sparse, mostly zeros |

## Dependencies

- Python 3.8+
- PyTorch (with CUDA for GPU acceleration)
- NumPy
- SciPy
- Matplotlib
- tqdm
- h5py

## Performance Notes

- **GPU Memory**: Each file uses ~50MB GPU memory
- **Processing Time**: ~0.5s per file on L4 GPU
- **Disk Space**: Results ~1MB per file (compressed npz)

## Related Analyses

This analysis complements:
- `dark_matter_analysis/`: Studies R² relationship of D_i vs ||W_i||²
- `svd_analysis/`: Singular value decomposition studies
- `phase_analysis/`: Phase space trajectory analysis
