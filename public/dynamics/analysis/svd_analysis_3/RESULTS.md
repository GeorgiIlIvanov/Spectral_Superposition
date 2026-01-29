# SVD Analysis v3: EIGENSPACE-Based Results (Corrected)

## Analysis Date: 2026-01-29

## Critical Correction Applied

**This analysis fixes a fundamental flaw in v2**: projections are now computed onto **eigenspaces** (entire subspaces) rather than individual eigenvectors. This is the only basis-invariant approach for degenerate eigenvalues.

## Key Results

### Overall Correlation
- **Clusters analyzed**: 16,407 (vs 12,403 in v2)
- **Files processed**: 2,640/3,200
- **Pearson correlation (κ vs 1/λ)**: r = **0.972**
- **Regression**: κ = 0.957/λ + 0.015
- **R²**: 0.945

### κ×λ Statistics (Conjecture: κ×λ = 1)
- **Mean**: 0.978 ± 0.067
- **Median**: 0.996 (almost exactly 1!)

## Stratified Analysis by Eigenspace Localization

| Quartile | Localization Range | n_clusters | Pearson r | Regression Slope | κ×λ Mean ± Std |
|----------|-------------------|------------|-----------|------------------|----------------|
| Q1 (Low) | 0.02 - 0.08 | ~4,100 | **0.991** | 1.004 | 0.981 ± 0.056 |
| Q2 | 0.08 - 0.13 | ~4,100 | **0.992** | 0.984 | 0.992 ± 0.034 |
| Q3 | 0.13 - 0.23 | ~4,100 | **0.991** | 0.978 | 0.992 ± 0.044 |
| Q4 (High) | 0.23 - 1.00 | ~4,100 | 0.889 | 0.890 | 0.946 ± 0.102 |

## Key Insight: The Conjecture Works Best for Delocalized Features

With the correct eigenspace-based analysis, we see a **much cleaner pattern**:

1. **Q1-Q3 (Delocalized features)**: All have r > 0.99 and regression slopes very close to 1
   - The conjecture κ ≈ 1/λ is essentially **exact** for these features
   - Median κ×λ approaches 1.0

2. **Q4 (Highly localized features)**: Shows deviation (r = 0.889, slope = 0.890)
   - These features are concentrated in a single eigenspace
   - The fit is still good, but not as tight

## Comparison: v3 (Correct) vs v2 (Flawed)

| Metric | v2 (Eigenvector) | v3 (Eigenspace) |
|--------|-----------------|-----------------|
| Total clusters | 12,403 | 16,407 |
| Overall r | 0.969 | **0.972** |
| Q1 correlation | 0.988 | **0.991** |
| Q2 correlation | 0.994 | **0.992** |
| Q3 correlation | 0.970 | **0.991** |
| Q4 correlation | 0.892 | 0.889 |

The corrected analysis shows:
- **More uniform high correlation** across Q1-Q3 (all > 0.99)
- **Cleaner separation**: Only Q4 shows significant deviation
- Q3 improved dramatically (0.970 → 0.991) because eigenvector ambiguity was removed

## Physical Interpretation

The **max eigenspace projection** measures what fraction of a feature's variance lies in its dominant eigenspace:

- **Low localization** (0.02-0.13): Features spread across many eigenspaces
  - These behave most "linearly" in the sense of the conjecture
  - The collective behavior averages nicely

- **High localization** (0.23-1.00): Features concentrated in one eigenspace
  - Stronger individual character
  - Deviations from the collective κ ≈ 1/λ relationship

## Output Files

| File | Size | Description |
|------|------|-------------|
| `eigenspace_cluster_data.csv` | 7.5 MB | Filtered cluster data |
| `all_eigenspace_stats.csv` | 28 MB | All 52,909 cluster statistics |
| `run_stats.csv` | 320 KB | Per-file stats including degeneracy |
| `summary_statistics.json` | 2.2 KB | Aggregated statistics |

## Conclusions

1. **The conjecture κ ≈ 1/λ is essentially exact** for features delocalized across eigenspaces (Q1-Q3: r > 0.99)

2. **Eigenspace-based analysis is essential** - eigenvector-based analysis gave misleading results for Q3

3. **High localization causes deviation** - features strongly confined to one eigenspace show κ×λ ≈ 0.95 instead of 1.0

4. **The conjecture describes collective behavior** - it works best when features participate in the eigenspace structure collectively, not individually
