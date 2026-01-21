# Stratified Linearity Analysis Report
**Generated:** 2026-01-21 06:41:48**Processing Time:** 1557.6 seconds
---
## Executive Summary
This analysis tests whether the linear scaling law **D_i ∝ ||W_i||²** holds universally for superposition features, without assuming the eigenvalue relationship a priori.
### Key Findings
**HIGH Sparsity (S ∈ (0.9, 1.0)):**
- Median R² of Deep Superposition Features: **0.8169**
- Percentage with R² < 0.9 ("Dark Matter"): **73.70%**
- Number of features analyzed: 324,542

**EXTREME Sparsity (S ∈ (0.99, 1.0)):**
- Median R² of Deep Superposition Features: **0.7115**
- Percentage with R² < 0.9 ("Dark Matter"): **82.55%**
- Number of features analyzed: 64,844

---
## Analysis Protocol

### 1. Primary Stratification by Sparsity (S)
- **Low Sparsity:** S ∈ [0.0, 0.2] - expect mostly orthogonal features
- **High Sparsity:** S ∈ [0.9, 1.0] - expect mostly superposition features
- **Extreme Sparsity:** S = 0.99 - the critical regime

### 2. Secondary Stratification by Feature Dimensionality
- Filter for D_i < 0.5 at final checkpoint ("Deep Superposition")

### 3. Feature-Level Linearity Test
- For each feature, fit D_i(t) vs ||W_i(t)||² across ALL training checkpoints
- Calculate R² for every single feature trajectory

### 4. Verification
- If the conjecture is robust, the R² histogram should be strongly peaked at 1.0
- "Dark Matter" = features with R² < 0.9 that defy the geometric prediction

---
## Detailed Results by Sparsity Bucket
### LOW Sparsity: S ∈ (0.0, 0.2)
- **Files in bucket:** 640
- **Sparsity values:** 0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18

#### All Features
- Total features: 655,360
- Valid features (enough data points): 655,360
- Mean R²: 0.4819
- Median R²: 0.4716

#### Deep Superposition Features (D < 0.5)
- **Count:** 501,815
- **Median R²:** 0.4629
- **Mean R²:** 0.4776 ± 0.3379

##### "Dark Matter" Analysis (features defying linear scaling)
- R² < 0.9: **84.77%**
- R² < 0.8: 73.54%
- R² < 0.7: 65.98%
- R² < 0.5: 52.60%

##### R² Percentiles
| Percentile | R² Value |
|------------|----------|
| P5 | 0.0068 |
| P10 | 0.0267 |
| P25 | 0.1484 |
| P50 | 0.4629 |
| P75 | 0.8163 |
| P90 | 0.9481 |
| P95 | 0.9625 |

### HIGH Sparsity: S ∈ (0.9, 1.0)
- **Files in bucket:** 320
- **Sparsity values:** 0.91, 0.93, 0.95, 0.97, 0.99

#### All Features
- Total features: 327,680
- Valid features (enough data points): 327,680
- Mean R²: 0.7285
- Median R²: 0.8146

#### Deep Superposition Features (D < 0.5)
- **Count:** 324,542
- **Median R²:** 0.8169
- **Mean R²:** 0.7314 ± 0.2386

##### "Dark Matter" Analysis (features defying linear scaling)
- R² < 0.9: **73.70%**
- R² < 0.8: 46.65%
- R² < 0.7: 32.10%
- R² < 0.5: 16.00%

##### R² Percentiles
| Percentile | R² Value |
|------------|----------|
| P5 | 0.1801 |
| P10 | 0.3652 |
| P25 | 0.6291 |
| P50 | 0.8169 |
| P75 | 0.9035 |
| P90 | 0.9476 |
| P95 | 0.9761 |

### EXTREME Sparsity: S ∈ (0.99, 1.0)
- **Files in bucket:** 64
- **Sparsity values:** 0.99

#### All Features
- Total features: 65,536
- Valid features (enough data points): 65,536
- Mean R²: 0.6389
- Median R²: 0.7072

#### Deep Superposition Features (D < 0.5)
- **Count:** 64,844
- **Median R²:** 0.7115
- **Mean R²:** 0.6426 ± 0.2680

##### "Dark Matter" Analysis (features defying linear scaling)
- R² < 0.9: **82.55%**
- R² < 0.8: 61.63%
- R² < 0.7: 48.58%
- R² < 0.5: 27.48%

##### R² Percentiles
| Percentile | R² Value |
|------------|----------|
| P5 | 0.0761 |
| P10 | 0.2201 |
| P25 | 0.4709 |
| P50 | 0.7115 |
| P75 | 0.8724 |
| P90 | 0.9243 |
| P95 | 0.9433 |

---
## Interpretation

**WEAK SUPPORT or REJECTION:** A significant fraction of superposition features do not follow the predicted linear scaling law. This suggests the geometric picture may be incomplete or requires refinement.

## Generated Files

### Python Scripts
- `stratified_linearity_analysis.py` - Main analysis script

### Visualizations
- `r2_histograms_by_sparsity.png` - R² histograms for all sparsity buckets
- `high_sparsity_detailed_analysis.png` - Detailed analysis of high sparsity regime
- `extreme_sparsity_analysis.png` - Analysis of S=0.99 critical regime
- `sparsity_regime_comparison.png` - Cross-bucket comparison summary

### Data Files
- `bucket_statistics.json` - Complete statistics for all buckets
- `analysis_report.md` - This report
