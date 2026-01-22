# Refined Spectral Analysis Documentation

**Date:** 2026-01-22
**Author:** Claude Code (Anthropic)
**Project:** Spectral Superposition Analysis

---

## Table of Contents

1. [Overview](#overview)
2. [Training Sweep Details](#training-sweep-details)
3. [Data Pipeline](#data-pipeline)
4. [Computed Quantities](#computed-quantities)
5. [File Formats](#file-formats)
6. [Mathematical Background](#mathematical-background)
7. [Usage Instructions](#usage-instructions)
8. [Output Structure](#output-structure)

---

## Overview

This refined spectral analysis extends the existing SVD-based feature clustering analysis to compute the **full projection weight distribution** `p_{ik}(t)` for all features and eigenspaces, enabling verification of the key theoretical relationship:

$$\kappa_i(t) = \mathbb{E}_{\mu_i(t)}[\lambda] = \sum_k p_{ik}(t) \lambda_k$$

where:
- `κ_i(t)` is the Rayleigh quotient for feature `i` at time `t`
- `p_{ik}(t)` is the projection weight of feature `i` onto eigenspace `k`
- `λ_k` is the `k`-th eigenvalue of `WW^T`

---

## Training Sweep Details

### Model Architecture
- **Input dimension:** 1024 features
- **Hidden dimension (m):** 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512 (32 values)
- **Output dimension:** 1024 (reconstruction)

### Training Parameters
- **Learning rate:** 1e-3
- **Batch size:** 1024
- **Total training steps:** 25,000
- **Checkpoint frequency:** Every 200 steps + initial (step 0)
- **Total checkpoints per run:** 56

### Sparsity Sweep
- **Range:** 0.0 to ~0.99
- **Number of values:** 50
- **Distribution:** Linear spacing

### Random Seeds
- **Seeds:** 0, 1 (2 seeds per configuration)

### Total Configurations
```
32 (m values) × 50 (sparsity values) × 2 (seeds) = 3,200 experiments
```

### Training Data
- Sparse binary vectors with configurable sparsity
- Feature importance follows power-law distribution

---

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                          SOURCE DATA                                 │
│         ../start/n1024_m{M}_s{S}_seed{SEED}.h5                      │
│  Contains: weights (56, m, 1024), feature_norms, fractional_dims    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SVD DECOMPOSITION                             │
│              svd_results/svd_n1024_m{M}_s{S}_seed{SEED}.h5          │
│       Contains: U (56, m, m), S (56, m), Vt (56, m, 1024),          │
│                 eigenvalues (56, m)                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   CLUSTERING    │  │    RAYLEIGH     │  │  PHASE ANALYSIS │
│   RESULTS       │  │    QUOTIENTS    │  │                 │
│ (dominant k)    │  │  (kappa values) │  │ (linear fits)   │
└────────┬────────┘  └────────┬────────┘  └─────────────────┘
         │                    │
         └────────┬───────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  REFINED SPECTRAL ANALYSIS (NEW)                     │
│          refined_spectral_results/spectral_n1024_m{M}_s{S}_seed{SEED}.npz          │
│                                                                      │
│  Contains:                                                           │
│  - projection_weights: (56, 1024, m) full p_{ik}(t) matrix          │
│  - kappa_expected: (56, 1024) = sum_k p_{ik} * lambda_k             │
│  - eigengaps: (56, m-1) = lambda_k - lambda_{k+1}                   │
│  - rotation_angles: (55, m) eigenspace rotation between timesteps   │
│  - participation_ratio: (56, 1024) effective # eigenspaces          │
│  - cluster_kappa_variance: (56, m) within-cluster κ variance        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Computed Quantities

### 1. Full Projection Weights `p_{ik}(t)`

**Definition:**
```
p_{ik}(t) = |u_k(t)^T w_i(t)|² / ||w_i(t)||²
```

**Shape:** `(T, n_features, m)` = `(56, 1024, m)`

**Interpretation:** Fraction of feature `i`'s variance explained by eigenspace `k`.

**Properties:**
- `sum_k p_{ik}(t) = 1` (projections sum to 1)
- `p_{ik} ≥ 0` (non-negative)

### 2. Expected Eigenvalue `κ_expected`

**Definition:**
```
κ_expected_i(t) = sum_k p_{ik}(t) * λ_k(t)
```

**Shape:** `(T, n_features)` = `(56, 1024)`

**Key Relationship:**
```
κ_actual_i(t) ≈ κ_expected_i(t)

where κ_actual_i = ||W^T w_i||² / ||w_i||²
```

### 3. Eigengaps

**Definition:**
```
δ_k(t) = λ_k(t) - λ_{k+1}(t)
```

**Shape:** `(T, m-1)` = `(56, m-1)`

**Interpretation:** Spectral gap between consecutive eigenvalues. Large gaps indicate well-separated eigenspaces.

### 4. Projector-Rotation Matrix

**Definition:**
```
R(t) = U(t+1)^T @ U(t)
```

**Shape:** `(T-1, m, m)` = `(55, m, m)`

**Stored:** Diagonal elements `R_kk(t)` and derived rotation angles.

**Interpretation:** `R_kk ≈ 1` means eigenspace `k` is stable; `R_kk ≈ 0` means it rotated significantly.

### 5. Rotation Angles

**Definition:**
```
θ_k(t) = arccos(|R_kk(t)|)
```

**Shape:** `(T-1, m)` = `(55, m)`

**Range:** `[0, π/2]` radians

**Interpretation:** Angle of rotation for eigenspace `k` between consecutive checkpoints.

### 6. Participation Ratio

**Definition:**
```
PR_i(t) = 1 / sum_k p_{ik}(t)²
```

**Shape:** `(T, n_features)` = `(56, 1024)`

**Range:** `[1, m]`

**Interpretation:** Effective number of eigenspaces feature `i` participates in.
- `PR = 1`: Feature dominated by single eigenspace
- `PR = m`: Feature uniformly spread across all eigenspaces

### 7. Number of Significant Eigenspaces

**Definition:**
```
n_sig_i(t) = count(p_{ik}(t) > 0.01)
```

**Shape:** `(T, n_features)` = `(56, 1024)`

**Interpretation:** Number of eigenspaces with >1% projection for feature `i`.

### 8. Projection Entropy

**Definition:**
```
H_i(t) = -sum_k p_{ik}(t) * log(p_{ik}(t))
```

**Shape:** `(T, n_features)` = `(56, 1024)`

**Range:** `[0, log(m)]`

**Interpretation:** Shannon entropy of projection distribution. Higher = more uniform.

### 9. Within-Cluster Statistics

For each eigenspace `k`, computed over features where `dominant_k = k`:

- `cluster_kappa_mean[t, k]`: Mean of `κ_actual` for cluster `k`
- `cluster_kappa_variance[t, k]`: Variance of `κ_actual` within cluster `k`
- `cluster_size[t, k]`: Number of features in cluster `k`

### 10. Kappa Error Metrics

**Definition:**
```
kappa_error_i(t) = κ_actual_i(t) - κ_expected_i(t)
kappa_relative_error_i(t) = (κ_actual_i - κ_expected_i) / κ_actual_i
```

**Shape:** `(T, n_features)` = `(56, 1024)`

---

## File Formats

### Per-File Output: `spectral_n1024_m{M}_s{S}_seed{SEED}.npz`

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `m_hidden` | `()` | int64 | Hidden dimension |
| `sparsity` | `()` | float64 | Sparsity value |
| `seed` | `()` | int64 | Random seed |
| `checkpoint_steps` | `(56,)` | int32 | Training steps |
| `n_features` | `()` | int64 | Number of features (1024) |
| `feature_norms` | `(56, 1024)` | float32 | `\|\|w_i\|\|²` |
| `fractional_dims` | `(56, 1024)` | float32 | `D_i` values |
| `eigenvalues` | `(56, m)` | float32 | `λ_k` |
| `singular_values` | `(56, m)` | float32 | `σ_k = √λ_k` |
| `kappa_actual` | `(56, 1024)` | float32 | Rayleigh quotient |
| `projection_weights` | `(56, 1024, m)` | float32 | `p_{ik}(t)` |
| `kappa_expected` | `(56, 1024)` | float32 | `E[λ]` |
| `eigengaps` | `(56, m-1)` | float32 | `λ_k - λ_{k+1}` |
| `rotation_matrix_diag` | `(55, m)` | float32 | `R_kk(t)` |
| `rotation_angles` | `(55, m)` | float32 | `θ_k(t)` |
| `participation_ratio` | `(56, 1024)` | float32 | `PR_i` |
| `dominant_eigenspace` | `(56, 1024)` | int16 | `argmax_k p_{ik}` |
| `max_projection` | `(56, 1024)` | float32 | `max_k p_{ik}` |
| `n_significant_eigenspaces` | `(56, 1024)` | int16 | Count of significant `k` |
| `projection_entropy` | `(56, 1024)` | float32 | `H_i` |
| `kappa_error` | `(56, 1024)` | float32 | `κ_actual - κ_expected` |
| `kappa_relative_error` | `(56, 1024)` | float32 | Relative error |
| `cluster_kappa_mean` | `(56, m)` | float32 | Per-cluster mean |
| `cluster_kappa_variance` | `(56, m)` | float32 | Per-cluster variance |
| `cluster_kappa_expected_mean` | `(56, m)` | float32 | Expected mean per cluster |
| `cluster_kappa_expected_variance` | `(56, m)` | float32 | Expected variance per cluster |
| `cluster_size` | `(56, m)` | int32 | Cluster sizes |

### Storage Estimate

Per file:
- `projection_weights`: `56 × 1024 × m × 4 bytes`
  - m=512: ~117 MB
  - m=16: ~3.5 MB
- Other arrays: ~2-3 MB

**Total for 3,200 files:** ~60-100 GB (depending on m distribution)

---

## Mathematical Background

### Rayleigh Quotient Identity

The Rayleigh quotient for feature `i` with respect to the correlation matrix `S = WW^T`:

```
κ_i = w_i^T S w_i / ||w_i||²
    = w_i^T (WW^T) w_i / ||w_i||²
    = ||W^T w_i||² / ||w_i||²
```

### Spectral Decomposition

SVD: `W = U Σ V^T`

Eigendecomposition: `WW^T = U Λ U^T` where `Λ = Σ²`

### Projection-Eigenvalue Relationship

For any feature vector `w_i`:

```
w_i^T S w_i = w_i^T (U Λ U^T) w_i
            = (U^T w_i)^T Λ (U^T w_i)
            = sum_k λ_k (u_k^T w_i)²
```

Normalizing:
```
κ_i = sum_k λ_k |u_k^T w_i|² / ||w_i||²
    = sum_k λ_k p_{ik}
    = E_μ_i[λ]
```

This is the key theoretical relationship being verified.

### Within-Cluster Variance

If features cluster by dominant eigenspace, within cluster `k`:

```
Var(κ_i | i ∈ cluster_k) = Var(E_μ_i[λ] | dominant_i = k)
```

Low variance indicates that features in the same eigenspace have similar spectral properties.

---

## Usage Instructions

### Running the Analysis

```bash
# Activate environment
source ../venv/bin/activate

# Run with all 8 GPUs
python refined_spectral_analysis.py --gpus 0,1,2,3,4,5,6,7

# Run with specific GPUs
python refined_spectral_analysis.py --gpus 0,1,2,3

# Custom directories
python refined_spectral_analysis.py \
    --source-dir ../start \
    --svd-dir svd_results \
    --output-dir refined_spectral_results \
    --rayleigh-dir dynamic_hopping/results/per_file
```

### Running in tmux (Recommended)

```bash
# Create tmux session
tmux new-session -d -s spectral

# Run the script
tmux send-keys -t spectral 'cd /home/georgi/Spectral_Superposition/public/dynamics/analysis && source ../venv/bin/activate && python refined_spectral_analysis.py 2>&1 | tee spectral_run.log' Enter

# Attach to monitor
tmux attach -t spectral

# Detach: Ctrl+B, then D
```

### Aggregating Results

After processing completes:

```bash
python aggregate_spectral_results.py \
    --input-dir refined_spectral_results \
    --output-dir refined_spectral_results/aggregated
```

### Monitoring Progress

```bash
# Watch log file
tail -f refined_spectral_analysis.log

# Check GPU utilization
watch -n 1 nvidia-smi

# Count completed files
ls refined_spectral_results/spectral_*.npz | wc -l
```

---

## Output Structure

```
refined_spectral_results/
├── spectral_n1024_m16_s0.000000_seed0.npz
├── spectral_n1024_m16_s0.000000_seed1.npz
├── ...
├── spectral_n1024_m512_s0.989796_seed1.npz
├── processing_summary.json          # Processing statistics
└── aggregated/
    ├── verification_summary.json    # Global verification metrics
    ├── verification_per_file.json   # Per-file verification details
    ├── aggregated_by_config.npz     # Metrics by (m, sparsity)
    └── time_series_aggregation.npz  # Time-series averages
```

---

## Log Files

- `refined_spectral_analysis.log` - Main processing log
- `spectral_run.log` - Full console output (if using tee)
- `processing_summary.json` - Machine-readable processing summary

---

## Expected Runtime

Based on file sizes and GPU memory:

| Configuration | Estimated Time |
|---------------|----------------|
| 8 L4 GPUs, 3,200 files | ~30-60 minutes |
| 4 L4 GPUs, 3,200 files | ~60-120 minutes |
| 1 L4 GPU, 3,200 files | ~4-8 hours |

Key factors:
- I/O speed (reading H5 files)
- GPU compute (matrix operations)
- Memory bandwidth (m=512 files are larger)

---

## Verification Checklist

After processing, verify:

1. **File count:** `ls refined_spectral_results/*.npz | wc -l` should equal 3,200
2. **Processing summary:** Check `processing_summary.json` for failures
3. **Correlation check:** `verification_summary.json` should show high correlation (>0.9)
4. **Sanity check:** `projection_weights` should sum to ~1 across k dimension

```python
import numpy as np
data = np.load('refined_spectral_results/spectral_n1024_m64_s0.500000_seed0.npz')
print("Projection sums:", data['projection_weights'].sum(axis=-1).mean())  # Should be ~1.0
print("Kappa correlation:", np.corrcoef(data['kappa_actual'].flat, data['kappa_expected'].flat)[0,1])
```

---

## Contact

For issues or questions about this analysis:
- GitHub: https://github.com/anthropics/claude-code/issues
