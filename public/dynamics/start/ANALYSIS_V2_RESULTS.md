# Capacity Localization Analysis v2 - Results

## Analysis Summary

**Run Date:** 2026-01-29
**Files Processed:** 3200 experiments
**Features Analyzed:** 3,276,800 total
**GPUs Used:** 8x L4

## Acceptance Criteria Results

### 1. q_bin Separates Sparsity Regimes

| Sparsity Range | Mean q_bin | Trend |
|----------------|------------|-------|
| Low (0-0.3) | 0.2571 | Baseline |
| Mid (0.3-0.6) | 0.2599 | Similar |
| High (0.6-0.9) | 0.1673 | Decreasing |
| V.High (0.9-1.0) | 0.0790 | Lowest |

**Result: PASS** - q_bin decreases at high sparsity, showing delocalization signature.

### 2. CV Increases with Sparsity

| Sparsity Range | Mean CV |
|----------------|---------|
| Low (0-0.3) | 0.0124 |
| Mid (0.3-0.6) | 0.0296 |
| High (0.6-0.9) | 0.0332 |
| V.High (0.9-1.0) | 0.0503 |

**Result: PASS** - CV (coefficient of variation) increases monotonically with sparsity, tracking the expected spectral spread trend.

### 3. Defect Identity Holds

- Mean error: 1.77e-13
- Max error: 7.44e-13
- Threshold: 1e-3

**Result: PASS** - The identity `gap1 = m - sumD = sum(ell - D) = gap2` holds to machine precision.

### 4. No Negative D Values

- Mean negD_frac: 0.0000
- Max negD_frac: 0.0000

**Result: PASS** - All stored fractional dimension values are non-negative. No need to use D_hat as fallback.

### 5. Assignment Test Validates Eigenspace Localization

| Sparsity Range | n_qualified | qual_frac | R² | Slope |
|----------------|-------------|-----------|-----|-------|
| Low (0-0.3) | 49,490 | 5.03% | 0.99999 | 1.0006 |
| Mid (0.3-0.6) | 37,022 | 3.77% | 0.99994 | 0.9989 |
| High (0.6-0.9) | 5,395 | 0.55% | 0.99997 | 0.9995 |
| V.High (0.9-1.0) | Too few | N/A | N/A | N/A |

**Result: PASS** - For confidently localized features (q_bin >= 0.9):
- D_hat vs D_hat_pred shows near-perfect linear relationship (R² > 0.9999)
- Qualification rate drops significantly at high sparsity (5% -> 0.5%)
- Very high sparsity has essentially no confidently localized features

## Key Findings

### Delocalization at High Sparsity is Confirmed

1. **q_bin decreases** from ~0.26 at low sparsity to ~0.08 at very high sparsity
2. **CV increases** from ~0.01 to ~0.05, indicating broader spectral spread
3. **Qualification rate drops** from 5% to <1% of leverage-weighted features being confidently localized

### Saturation Defect is Genuine Slack

1. **Rank ratio = 1** for all runs (no rank defect)
2. **Defect identity holds** exactly: the gap `m - sumD` equals `sum(ell - D)`
3. This confirms the saturation defect is from Cauchy-Schwarz slack (delocalization), not rank loss

### Data Quality is Good

1. **No negative D values** anywhere in the dataset
2. **D and D_hat correlate perfectly** (corr = 1.0000)
3. **All numerical identities hold** to machine precision

## Output Files

### Analysis Results
- `analysis_results_v2/run_metrics.csv` - 3200 run-level metrics
- `analysis_results_v2/feature_metrics.csv` - 3,276,800 feature-level metrics
- `analysis_results_v2/analysis_metadata.json` - Analysis configuration

### Plots Generated
| File | Description |
|------|-------------|
| `plot_A_rank_delocalization.png/pdf` | Rank/saturation vs sparsity |
| `plot_A_supplementary_diagnostics.png` | mean_sigma, mean_resid vs sparsity |
| `plot_B_featurewise_localization.png/pdf` | Sigma distributions by sparsity |
| `plot_C_assignment_test.png/pdf` | D_hat vs D_hat_pred for localized features |
| `plot_C_assignment_test_fits.csv` | Regression statistics |
| `plot_cv_vs_sparsity.png/pdf` | Coefficient of variation analysis |
| `plot_q_bin_distribution.png/pdf` | Binned Diracness proxy distributions |
| `plot_qual_rate_vs_sparsity.png/pdf` | Qualification rate trends |
| `plot_D_vs_D_hat_diagnostic.png` | D sanity check |
| `plot_entropy_distribution.png` | H_bin and n_eff distributions |
| `plot_tail_mass_curves.png` | Delocalization tail mass |
| `plot_defect_identity_check.png` | Gap verification |
| `analysis_summary.txt` | Text summary of results |

## Reproducing the Analysis

```bash
# Run analysis (using all 8 GPUs)
python capacity_localization_analysis_v2.py \
    --data-dir . \
    --out ./analysis_results_v2 \
    --save-feature-metrics \
    --num-gpus 8

# Generate plots
python generate_plots_v2.py \
    --results-dir ./analysis_results_v2 \
    --out ./plots_v2
```

## Comparison with Original (v1) Analysis

| Metric | v1 (Broken) | v2 (Fixed) | Status |
|--------|-------------|------------|--------|
| Diracness proxy (q) | Mean ~0.04 everywhere | q_bin varies 0.08-0.26 | FIXED |
| Negative D values | Present | None | FIXED |
| Scale-free spread | Not available | CV metric added | NEW |
| Defect verification | Not checked | Verified to 1e-13 | NEW |
| Assignment test | Identity (trivial) | Meaningful test | FIXED |

## Technical Notes

### Binning Parameters
- Number of bins: 60 (log-spaced)
- Eigenvalue tolerance: 1e-8 * lambda_max
- Localization threshold: q_bin >= 0.9

### Numerical Precision
- All computations in float64
- Epsilon values: 1e-12 for log stability
- Defect errors: O(1e-13)

### Performance
- Processing time: ~9 minutes for 3200 files on 8x L4 GPUs
- Feature metrics file: ~1.1 GB
- Peak GPU memory: ~2 GB per GPU
