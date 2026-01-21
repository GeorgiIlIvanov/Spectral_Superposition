# Script Documentation

Detailed documentation for each Python script in the Dynamic Hopping Analysis pipeline.

---

## 00_config.py - Configuration Constants

### Purpose
Centralized configuration file containing all constants, paths, and parameters used throughout the analysis pipeline.

### Contents

#### Numerical Constants
```python
EPSILON = 1e-12        # Guard for log/division (prevents log(0))
EPSILON_MAD = 1e-9     # Guard for MAD computation
Z_THRESHOLD = 4.0      # Jump detection threshold (σ units)
LATE_WINDOW_FRACTION = 0.2  # Last 20% of training
MIN_VALID_POINTS = 5   # Minimum points for regression
```

#### Sparsity Buckets
Analysis is stratified by sparsity level:
- **low**: [0.0, 0.3) - Dense features
- **medium**: [0.3, 0.7) - Moderate sparsity
- **high**: [0.7, 0.95) - Sparse features
- **extreme**: [0.95, 1.0) - Very sparse

#### Paths
All I/O paths defined relative to project structure.

#### Hopping Categories
Feature classification thresholds:
- **stable**: σ < 0.5
- **moderate**: 0.5 ≤ σ < 1.5
- **active**: 1.5 ≤ σ < 3.0
- **extreme**: σ ≥ 3.0

---

## config_loader.py - Configuration Loader

### Purpose
Helper module to load configuration as a dictionary, making it easy to pass around.

### Usage
```python
from config_loader import load_config, get_sparsity_bucket

cfg = load_config()
print(cfg['z_threshold'])  # 4.0

bucket = get_sparsity_bucket(0.5, cfg['sparsity_buckets'])
print(bucket)  # 'medium'
```

---

## 01_rayleigh_quotients.py - Core Computation

### Purpose
Compute Rayleigh quotients κ_i(t) for all features at all checkpoints.

### Mathematical Details

The Rayleigh quotient measures how much a feature participates in the overall covariance structure:

```
κ_i = (w_i^T S w_i) / ||w_i||²
```

where S = W W^T is the covariance matrix.

**Efficient computation** (avoids forming n×n matrix when n is large):
```
κ_i = ||W^T w_i||² / ||w_i||²
    = Σ_j (w_j · w_i)² / ||w_i||²
```

This is computed via:
1. Form Gram matrix M = W^T W (n×n)
2. For each i: κ_i = ||M[:,i]||² / ||w_i||²

### GPU Acceleration
Uses PyTorch for GPU-accelerated matrix multiplication:
```python
W_t = W[t]  # (d, n)
WtW = torch.mm(W_t.T, W_t)  # (n, n) Gram matrix
kappa[t] = torch.sum(WtW ** 2, dim=0) / (a[t] + epsilon)
```

### Command Line Arguments
```
--gpu N       : Use GPU device N (default: 0)
--sample N    : Process only N files (for testing)
--output-prefix: Prefix for output files
```

### Output Files
| File | Contents |
|------|----------|
| `per_file/<name>_rayleigh.npz` | κ, a, x arrays for each experiment |
| `rayleigh_aggregate.npz` | Aggregated statistics |
| `rayleigh_summary.json` | Human-readable summary |

### Memory Usage
- Peak GPU memory: ~100MB per file (for n=1024, d=112)
- Processes files sequentially to minimize memory

---

## 02_jump_detection.py - Jump Detection

### Purpose
Detect sudden changes ("jumps") in feature behavior using robust statistics.

### Algorithm

1. **Compute differences**:
   ```
   Δx_i(t) = x_i(t) - x_i(t-1)
   ```

2. **Robust standard deviation** using MAD:
   ```
   MAD = median(|Δx - median(Δx)|)
   σ_robust = 1.4826 × MAD
   ```

   The factor 1.4826 makes σ_robust equivalent to standard deviation for Gaussian distributions.

3. **Jump detection**:
   ```
   is_jump[t, i] = |Δx_i(t)| > z × σ_robust
   ```

   Default z = 4.0 means only very significant deviations are flagged.

### Late Window Analysis
Compares hopping behavior in early vs late training:
- **early_window**: First 20% of checkpoints
- **late_window**: Last 20% of checkpoints

Metrics:
- `sigma_ratio = late_sigma / early_sigma`
  - < 1: Hopping decreases (converging)
  - > 1: Hopping increases (diverging)

### Output Statistics
| Metric | Description |
|--------|-------------|
| `total_jumps` | Total jump events across all features/times |
| `jumps_per_feature` | Array of jump counts per feature |
| `jumps_per_checkpoint` | Array of jump counts per time |
| `sigma_global` | Global robust standard deviation |
| `features_with_jumps` | Count of features that jumped at least once |
| `late_sigma_ratio` | Ratio of late to early volatility |

---

## 03_temporal_patterns.py - Pattern Analysis

### Purpose
Analyze temporal structure of hopping and classify features by behavior.

### Rolling Volatility
Computes local volatility in sliding windows:
```python
for i in range(n_windows):
    window = x[i:i + window_size, :]
    volatility[i] = np.nanstd(window, axis=0)
```

### Phase Detection
Identifies phases of hopping activity:
1. Smooth the mean volatility signal
2. Find peaks (high activity) and troughs (low activity)
3. Compute overall trend via linear regression

### Feature Classification

**By volatility level**:
| Class | Threshold | Meaning |
|-------|-----------|---------|
| stable | σ < 0.5 | Predictable |
| moderate | σ < 1.5 | Some variation |
| active | σ < 3.0 | Significant hopping |
| extreme | σ ≥ 3.0 | Highly volatile |

**By temporal pattern**:
| Pattern | Criterion | Meaning |
|---------|-----------|---------|
| converging | late_vol < 0.7 × early_vol | Settling down |
| diverging | late_vol > 1.5 × early_vol | Becoming chaotic |
| steady | Neither | Stable behavior |
| episodic | mid_vol > max(early, late) | Bursts of activity |

### Synchrony Analysis
Measures if features hop together:

```
synchrony_ratio = observed_simultaneous / expected_independent
```

- `> 1.5`: Synchronized (features move together)
- `< 0.7`: Anti-synchronized
- `≈ 1.0`: Independent

---

## 04_visualizations.py - Plot Generation

### Purpose
Generate comprehensive visualizations of all analysis results.

### Individual File Plots

#### Rayleigh Heatmap (`*_rayleigh_heatmap.png`)
- **X-axis**: Checkpoint (training step)
- **Y-axis**: Feature index (sorted by mean κ)
- **Color**: log(κ + ε)
- **Purpose**: Show full temporal evolution of all features

#### Jump Events (`*_jump_events.png`)
4-panel figure:
1. **Scatter plot**: Jump locations (time vs feature), colored by magnitude
2. **Bar chart**: Jumps per checkpoint
3. **Histogram**: Distribution of jumps per feature
4. **Histogram**: Jump magnitude distribution

#### Volatility Evolution (`*_volatility_evolution.png`)
2-panel figure:
1. **Time series**: Rolling volatility with confidence bands
2. **Scatter**: Early vs late volatility per feature

### Aggregate Plots

#### Feature Classification Summary
4-panel figure showing distributions by sparsity:
1. Volatility class counts
2. Temporal pattern counts
3. Synchrony ratios
4. Trend directions

#### Sparsity Comparison
4-panel figure:
1. Total jumps vs sparsity
2. Global σ vs sparsity
3. Late/Early σ ratio vs sparsity
4. Jumps vs σ colored by sparsity

#### Summary Figure
9-panel comprehensive overview combining all key metrics.

### Command Line Arguments
```
--sample-plots N  : Create N individual file plots (default: 5)
```

---

## run_all.sh - Pipeline Runner

### Purpose
Execute the complete analysis pipeline in correct order.

### Usage
```bash
# Full analysis
./run_all.sh

# Test mode
./run_all.sh --sample 10

# Specific GPU
./run_all.sh --gpu 2
```

### Execution Order
1. Create output directories
2. Run 01_rayleigh_quotients.py (GPU-accelerated)
3. Run 02_jump_detection.py
4. Run 03_temporal_patterns.py
5. Run 04_visualizations.py

### Error Handling
Uses `set -e` to stop on any error.
