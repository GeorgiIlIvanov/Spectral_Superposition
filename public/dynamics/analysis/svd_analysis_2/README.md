# SVD Analysis v2: Slope-Eigenvalue Correlation with Spectral Localization

## Overview

This analysis extends the original slope-eigenvalue conjecture testing by incorporating **spectral localization metrics**. The key insight is that the conjecture κ ≈ 1/λ (where κ is the slope of D vs ||W||² and λ is the eigenvalue) only holds for features that are well-localized to a single eigenspace.

## Theoretical Background

### The Original Conjecture

For features clustered by their dominant eigenspace k:
- **D_i = κ × ||W_i||² + c** (linear relationship within cluster)
- **Conjecture**: κ ≈ 1/λ_k

The original analysis (in `svd_analysis/`) found:
- Pearson correlation: r = 0.97
- Regression: κ = 0.94/λ + 0.02
- Mean κ×λ = 0.94 (close to predicted 1.0)

However, this is a **cluster-level** analysis that averages over all features. The fit quality varies significantly.

### Why Localization Matters

The spectral measure of a feature describes how its weight vector projects onto the eigenspaces:

```
p_ik = |u_k^T w_i|² / ||w_i||²
```

For a **localized** feature: one p_ik ≈ 1, all others ≈ 0
For a **delocalized** feature: p_ik spread across multiple eigenspaces

The κ ≈ 1/λ relationship holds best when features are localized because:
1. The feature effectively "lives" in a single eigenspace
2. Its fractional dimension D_i is determined by that eigenspace's eigenvalue
3. The relationship D = ||W||² / λ becomes exact

For delocalized features:
- The feature spans multiple eigenspaces with different eigenvalues
- The effective eigenvalue is a weighted average: κ_expected = Σ_k p_ik λ_k
- The simple 1/λ relationship breaks down

## Localization Metrics

### Max Projection (Primary Localization Indicator)
```
max_projection_i = max_k p_ik
```
- Range: [1/m, 1]
- Higher values = more localized
- Value of 1 means feature is exactly aligned with one eigenvector

### Participation Ratio (Inverse Localization)
```
PR_i = 1 / Σ_k p_ik²
```
- Range: [1, m]
- Higher values = more delocalized
- Value of 1 means completely localized to one eigenspace
- Value of m means uniformly spread across all eigenspaces

### Projection Entropy
```
H_i = -Σ_k p_ik log(p_ik)
```
- Higher values = more delocalized
- Measures uncertainty in eigenspace assignment

## Analysis Pipeline

### Step 1: Per-File Processing
For each experiment file (with a specific m_hidden, sparsity, seed):

1. Load weights W (m × n) and SVD decomposition U, λ
2. Compute projection weights p_ik for all features
3. Compute localization metrics (max_projection, PR, entropy)
4. Cluster features by dominant eigenspace
5. For each cluster:
   - Fit D_i vs ||W_i||² to get slope κ
   - Record R² and number of features
   - Compute mean localization metrics of cluster members

### Step 2: Aggregation
- Filter clusters with R² > 0.1 and n_features >= 20
- Compute correlation between κ and 1/λ
- Stratify by localization quartiles

### Step 3: Visualization
- **Main plot**: κ vs 1/λ colored by localization
- **Stratified plots**: Same correlation within localization quartiles
- **Error analysis**: How fit error varies with localization

## Key Outputs

| File | Description |
|------|-------------|
| `cluster_localization_data.csv` | Filtered cluster-level data with localization metrics |
| `all_cluster_stats.csv` | All cluster statistics (unfiltered) |
| `run_stats.csv` | Per-file run-level statistics |
| `summary_statistics.json` | Aggregated statistics and correlation by quartile |
| `slope_eigenvalue_localization.png/pdf` | Main visualization with localization coloring |
| `localization_analysis.png/pdf` | Correlation and error analysis plots |
| `stratified_by_localization.png/pdf` | κ vs 1/λ by localization quartile |

## Expected Results

### Hypothesis
The conjecture κ ≈ 1/λ should work **better** for:
- Higher localization (max_projection closer to 1)
- Lower participation ratio (closer to 1)
- Lower sparsity (denser representations are more localized)

### Predicted Patterns
1. **High localization clusters**: κ×λ ≈ 1, high R²
2. **Low localization clusters**: κ×λ deviates from 1, lower R²
3. **Stratified analysis**: Correlation r should be highest in Q4 (most localized)

## Usage

```bash
# Activate environment
source /home/georgi/Spectral_Superposition/public/dynamics/venv/bin/activate

# Run analysis (uses 8 GPUs)
python slope_localization_analysis.py --num-gpus 8 --checkpoint final

# Test with subset
python slope_localization_analysis.py --sample 100 --num-gpus 8
```

## Relationship to Previous Work

### Builds on
- `svd_analysis/03_slope_eigenvalue_analysis.py` - Original slope-eigenvalue test
- `refined_spectral_analysis.py` - Full projection weight computation
- `capacity_localization_analysis_v2.py` - Binned spectral measure (q_bin)

### Key Differences
- Original analysis only tracked cluster slopes and eigenvalues
- This analysis adds localization metrics per cluster
- Allows coloring the scatter plot by localization degree
- Enables stratified analysis to identify when conjecture holds

## Date
Created: 2026-01-29
Author: Claude Code Analysis Pipeline
