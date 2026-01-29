# SVD Analysis v2: Results Summary

## Analysis Date: 2026-01-29

## Overview

This analysis extends the slope-eigenvalue conjecture testing by incorporating spectral localization metrics, allowing us to understand **why** the fit works for some clusters but not others.

## Key Findings

### Overall Correlation
- **Clusters analyzed**: 12,403
- **Files processed**: 2,153/3,200 (files with valid clusters)
- **Pearson correlation (κ vs 1/λ)**: r = 0.969
- **Regression**: κ = 0.943/λ + 0.022
- **R²**: 0.938

### κ×λ Statistics (Conjecture: κ×λ = 1)
- **Mean**: 0.974 ± 0.070
- **Median**: 0.995

The conjecture κ ≈ 1/λ holds remarkably well overall, with the median κ×λ almost exactly 1.

## Stratified Analysis by Localization

| Quartile | Localization Range | n_clusters | Pearson r | κ×λ Mean ± Std |
|----------|-------------------|------------|-----------|----------------|
| Q1 (Low) | 0.03 - 0.13 | 3,101 | 0.988 | 0.981 ± 0.046 |
| **Q2** | 0.13 - 0.18 | 3,100 | **0.994** | **0.993 ± 0.036** |
| Q3 | 0.18 - 0.28 | 3,101 | 0.970 | 0.981 ± 0.067 |
| Q4 (High) | 0.28 - 1.00 | 3,101 | 0.892 | 0.941 ± 0.102 |

### Key Insight: The "Goldilocks Zone"

Surprisingly, the **best fit is NOT for highly localized features** (Q4), but for the medium-low localization range (Q2).

**Why?**

1. **Q1-Q2 (Delocalized)**: Many features cluster into the same dominant eigenspace
   - Better statistical averaging over many features
   - Reduces noise in slope estimation
   - Excellent correlation: r = 0.988-0.994

2. **Q3 (Transitional)**: Intermediate regime
   - Still reasonable averaging, but more variance
   - Correlation: r = 0.970

3. **Q4 (Highly Localized)**: Features are well-separated
   - Fewer features per cluster → more noise in slope estimation
   - Individual feature idiosyncrasies dominate
   - Correlation drops: r = 0.892
   - Also shows κ×λ ≈ 0.94 (deviation from 1)

### Interpretation

The conjecture κ ≈ 1/λ is fundamentally about **eigenspace structure**, not individual features. The cluster-level slope analysis performs best when:

1. Multiple features share the same dominant eigenspace
2. These features have diverse ||W||² values (providing good range for linear fit)
3. The averaging over features reduces measurement noise

For highly localized features (each uniquely aligned to one eigenvector), the "cluster" often contains very few features, making the slope estimation unreliable.

## Generated Outputs

### Data Files
| File | Size | Description |
|------|------|-------------|
| `cluster_localization_data.csv` | 5.4 MB | Filtered clusters (R² > 0.1, n ≥ 20) with localization metrics |
| `all_cluster_stats.csv` | 14.8 MB | All 12,403 cluster statistics |
| `run_stats.csv` | 228 KB | Per-file run-level statistics |
| `summary_statistics.json` | 2.2 KB | Aggregated statistics by quartile |

### Visualization Files
| File | Description |
|------|-------------|
| `slope_eigenvalue_localization.png/pdf` | Main κ vs 1/λ plot colored by localization |
| `localization_analysis.png/pdf` | Correlation analysis: fit quality vs localization |
| `stratified_by_localization.png/pdf` | κ vs 1/λ by localization quartile |

## Conclusions

1. **The conjecture κ ≈ 1/λ holds strongly overall** (r = 0.97, median κ×λ = 0.995)

2. **Localization has a non-monotonic effect on fit quality**:
   - Best fit at intermediate localization (Q2)
   - Poorest fit at highest localization (Q4)

3. **Statistical averaging is key**: The cluster-level analysis benefits from having multiple features per cluster, which happens more often for delocalized features

4. **High localization features show slight deviation**: κ×λ ≈ 0.94 instead of 1.0, suggesting the conjecture may need refinement for highly localized cases

## Future Directions

1. **Feature-level analysis**: Instead of clustering, directly predict D_i from ||W_i||² and kappa_expected = Σ_k p_ik λ_k

2. **Weighted regression**: Weight the slope-eigenvalue correlation by cluster size

3. **Sparsity interaction**: Investigate how sparsity affects localization and fit quality jointly

## Computational Details

- **Runtime**: ~90 seconds on 8x L4 GPUs
- **Processing rate**: ~35 files/second
- **GPU utilization**: Round-robin assignment across 8 GPUs
