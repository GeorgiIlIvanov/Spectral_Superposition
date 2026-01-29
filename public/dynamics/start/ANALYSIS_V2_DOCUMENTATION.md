# Capacity Localization Analysis v2 - Documentation

## Overview

This document describes the fixes and new diagnostics implemented in version 2 of the capacity localization analysis pipeline. The changes address issues identified in the original implementation:

1. Broken "Diracness proxy" q metric (mean ~0.04 across all sparsities)
2. Negative D values in Plot C (impossible for fractional dimension)
3. Missing scale-free spectral spread measure
4. Lack of numerical verification of defect identity

## Files Created

| File | Description |
|------|-------------|
| `capacity_localization_analysis_v2.py` | Fixed analysis script with new metrics |
| `generate_plots_v2.py` | Fixed plotting script with new visualizations |
| `ANALYSIS_V2_DOCUMENTATION.md` | This documentation file |

## Changes Summary

### 1. Binned Spectral Measure (Fixes Diracness Proxy)

**Problem**: The original eigenvalue "grouping" method used `np.isclose` with strict `rtol=1e-6`, causing nearly all eigenvalues to be placed in separate groups. This made the "Diracness proxy" q trivially small (~0.04) for all features regardless of sparsity.

**Solution**: Replace eigenvalue grouping with log-spaced binning of the spectral measure.

**Implementation**:
```python
# For each feature i with weight vector w_i:
# 1. Compute spectral measure: p_k,i = (u_k^T w_i)^2 / ||w_i||^2
# 2. Create log-spaced bins over positive eigenvalues: num_bins = 60
# 3. Aggregate measure into bins: mu_i(B_b) = sum_{k: lambda_k in B_b} p_k,i
```

**New Metrics**:
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `q_bin[i]` | max_b μ_i(B_b) | Diracness proxy: 1 = localized to single bin |
| `H_bin[i]` | -Σ_b μ_i(B_b) log(μ_i(B_b)) | Binned entropy: 0 = perfectly localized |
| `n_eff[i]` | exp(H_bin[i]) | Effective number of bins: 1 = Dirac-like |
| `b_star[i]` | argmax_b μ_i(B_b) | Assigned bin index |
| `lambda_hat[i]` | geometric_mean(edges[b_star]) | Assigned eigenvalue estimate |

**Run-level Aggregates** (leverage-weighted):
- `wmean_q_bin`: Should decrease with sparsity (delocalization signature)
- `wmean_H_bin`: Should increase with sparsity
- `wmean_n_eff`: Should increase with sparsity

### 2. Normalized Spectral Spread (Coefficient of Variation)

**Problem**: The residual `resid = sqrt(Var(λ))` is not scale-free and depends on the eigenvalue magnitude.

**Solution**: Add coefficient of variation:
```python
cv[i] = resid[i] / (kappa[i] + 1e-12)
```

**Interpretation**:
- `cv` is dimensionless and comparable across runs with different eigenvalue scales
- Should correlate with `sigma` (relative slack)
- Should increase with sparsity

**Metrics Added**:
- `cv` per feature
- `mean_cv` run-level (leverage-weighted)

### 3. Defect Identity Verification

**Problem**: No numerical verification that the theoretical identity holds.

**Theory**: Since rank ratio r/m = 1, the saturation defect should equal leverage slack:
- `gap1 = m - Σ_i D_i`
- `gap2 = Σ_i (ℓ_i - D_i)` (over alive features)
- These should be equal: `gap1 = gap2`

**Implementation**:
```python
defect_error = |gap1 - gap2|
```

**Acceptance Criteria**:
- `defect_error < 1e-6`: Excellent
- `defect_error < 1e-3`: Acceptable (numerical precision)
- `defect_error > 1e-3`: Indicates bug or inconsistent data

### 4. D vs D_hat Sanity Checks (Fixes Plot C)

**Problem**: Plot C shows negative D values, which is impossible for fractional dimension (D_i >= 0 by definition).

**Solution**: Compute reconstructed D_hat and compare with stored D.

**Implementation**:
```python
# Reconstructed fractional dimension
D_hat[i] = norm2[i] / (kappa[i] + 1e-12)
# where norm2[i] = ||w_i||^2, kappa[i] = E_{mu_i}[lambda]
```

**Sanity Check Metrics**:
| Metric | Formula | Expected |
|--------|---------|----------|
| `negD_frac` | fraction of D_i < 0 | Should be 0 |
| `D_Dhat_corr` | correlation(D, D_hat) | Should be ~1 if D stored correctly |

**Plot C Fix**: If `negD_frac > 0` for any run, use `D_hat` instead of stored `D` in all plots.

### 5. Assignment Test (Replaces Broken Plot C)

**Problem**: Plotting D vs 1/κ is not meaningful if D = ||w||²/κ (it's an identity by construction).

**Solution**: Test discrete eigenspace assignment for "confidently localized" features.

**Implementation**:
```python
# For features with q_bin >= 0.9 (confidently localized):
D_hat_pred[i] = norm2[i] / lambda_hat[i]
# where lambda_hat[i] = bin center of assigned bin

# Compare D_hat vs D_hat_pred
```

**Metrics**:
- `qual_rate`: Leverage-weighted fraction of features with q_bin >= 0.9
- R², slope, intercept of D_hat vs D_hat_pred regression

**Expectations**:
- **Low sparsity**: Many features qualify (high qual_rate), prediction is tight (R² ~ 1)
- **High sparsity**: Fewer qualify OR prediction loosens (bins broad, measures not Dirac-like)

## Output Files

### Run Metrics (run_metrics.csv)

New columns added in v2:
| Column | Description |
|--------|-------------|
| `mean_cv` | Leverage-weighted mean coefficient of variation |
| `wmean_q_bin` | Leverage-weighted mean binned Diracness proxy |
| `wmean_H_bin` | Leverage-weighted mean binned entropy |
| `wmean_n_eff` | Leverage-weighted mean effective support |
| `defect_error` | |gap1 - gap2| verification |
| `negD_frac` | Fraction of negative D values |
| `D_Dhat_corr` | Correlation between D and D_hat |
| `qual_rate` | Fraction of leverage in confidently localized features |

### Feature Metrics (feature_metrics.csv)

New columns added in v2:
| Column | Description |
|--------|-------------|
| `D_hat` | Reconstructed fractional dimension = norm2/kappa |
| `cv` | Coefficient of variation = resid/kappa |
| `norm2` | ||w_i||² |
| `q_bin` | Binned Diracness proxy |
| `H_bin` | Binned entropy |
| `n_eff` | Effective support |
| `b_star` | Assigned bin index |
| `lambda_hat` | Assigned eigenvalue estimate |
| `D_hat_pred` | Predicted D from bin assignment |

## Plots Generated

### Run-Level Plots

| Plot | Description | Key Diagnostic |
|------|-------------|----------------|
| `plot_A_rank_delocalization.png` | Rank/saturation vs sparsity | Unchanged from v1 |
| `plot_A_supplementary_diagnostics.png` | mean_sigma, mean_resid vs sparsity | Unchanged from v1 |
| `plot_cv_vs_sparsity.png` | Coefficient of variation analysis | **NEW**: cv should increase with s |
| `plot_qual_rate_vs_sparsity.png` | Qualification rate vs sparsity | **NEW**: qual_rate should decrease with s |
| `plot_tail_mass_curves.png` | Delocalization tail mass | Unchanged from v1 |
| `plot_defect_identity_check.png` | Gap verification | **NEW**: defect_error should be tiny |

### Feature-Level Plots

| Plot | Description | Key Diagnostic |
|------|-------------|----------------|
| `plot_B_featurewise_localization.png` | Sigma distributions | Unchanged from v1 |
| `plot_q_bin_distribution.png` | Binned Diracness proxy | **NEW**: Replaces broken q plot |
| `plot_C_assignment_test.png` | D_hat vs D_hat_pred | **NEW**: Fixed Plot C |
| `plot_D_vs_D_hat_diagnostic.png` | D sanity check | **NEW**: Shows negative D issues |
| `plot_entropy_distribution.png` | H_bin and n_eff | **NEW**: Entropy analysis |

## Usage

### Run Analysis
```bash
# Basic usage with 8 GPUs
python capacity_localization_analysis_v2.py \
    --data-dir /path/to/h5/files \
    --out ./analysis_results_v2 \
    --save-feature-metrics \
    --num-gpus 8

# With custom bin count
python capacity_localization_analysis_v2.py \
    --data-dir /path/to/h5/files \
    --out ./analysis_results_v2 \
    --save-feature-metrics \
    --num-gpus 8 \
    --num-bins 100
```

### Generate Plots
```bash
python generate_plots_v2.py \
    --results-dir ./analysis_results_v2 \
    --out ./plots_v2
```

## Acceptance Criteria

The analysis is successful if:

1. **q_bin separates sparsity regimes**: Low-s runs should have higher `wmean_q_bin` than high-s runs
2. **CV tracks sigma**: `mean_cv` should increase with sparsity and correlate with `mean_sigma`
3. **Defect identity holds**: `defect_error` should be small (< 1e-3)
4. **No negative D or use D_hat**: Either `negD_frac = 0` OR Plot C uses `D_hat`
5. **Assignment test validates**: For low sparsity, D_hat vs D_hat_pred should have high R² and many qualified features; for high sparsity, either fewer qualify or R² decreases

## Technical Details

### Eigenvalue Binning

The binning uses log-spacing to handle the wide dynamic range of eigenvalues:
```python
edges = np.logspace(log10(lam_min), log10(lam_max), num_bins + 1)
```

Bin assignment:
```python
b = np.searchsorted(edges, lam, side='right') - 1
b = np.clip(b, 0, num_bins - 1)
```

### GPU Acceleration

Both scripts use PyTorch for GPU-accelerated eigendecomposition. Files are processed with round-robin GPU assignment:
```python
gpu_id = idx % num_gpus
device = torch.device(f'cuda:{gpu_id}')
```

### Numerical Stability

Small epsilon values are used to prevent division by zero:
- Eigenvalue tolerance: `tol = 1e-8 * lam_max`
- Entropy computation: `mu_safe = mu + 1e-12`
- D_hat computation: `kappa + 1e-12`

## Changelog

### v2 (2026-01-29)
- Fixed binned spectral measure for Diracness proxy
- Added coefficient of variation (cv)
- Added defect identity verification
- Added D vs D_hat sanity checks
- Replaced Plot C with assignment test
- Added entropy and effective support metrics
- Added qualification rate analysis
