# Comprehensive Spectral Superposition Analysis Report

**Generated:** 2026-01-22
**Total Runs Analyzed:** 3,200 configurations
**Experiments:** A (Eigengaps), B (Spectral Spread), C (Within-Cluster Variance), D-H (Global Projective Linearity)

---

# EXECUTIVE SUMMARY

## Core Finding

**Dark matter (DM) features are primarily a manifestation of global spectral dynamics, not feature-specific anomalies.**

The D = α(t)×N relationship (fractional dimension proportional to feature norm) holds universally across all features with R² = 0.914. Normalizing by α(t) stabilizes 72% of DM features, and slope ratios between DM and well-behaved features are essentially unity (0.998).

## Key Numbers

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Overall DM rate | 35.4% | One-third of features are DM |
| Critical sparsity | 0.28-0.30 | Phase transition threshold |
| Cross-sectional R² | 0.914 | Strong D = αN linearity |
| DM stabilization rate | 71.7% | α-normalization works |
| DM in spiked eigenspaces | 99.1% | Not a bulk phenomenon |
| Best α predictor | trace (38%) | Global spectral quantity |
| Predictive AUC (spread) | 0.743 | Moderate classification power |

---

# PARAMETER SPACE

## Configuration Grid

| Parameter | Values | Count |
|-----------|--------|-------|
| m_hidden | 16, 32, 48, ..., 512 | 32 |
| sparsity | 0.00, 0.02, ..., 0.99 | 50 |
| seed | 0, 1 | 2 |
| **Total** | | **3,200** |

## Phase Diagram

The (m_hidden, sparsity) plane exhibits three regimes:

1. **Localized Regime** (s < 0.28): DM ≈ 0%, features are well-behaved
2. **Transition Region** (0.28 < s < 0.55): DM emerges, highest variability
3. **Delocalized Regime** (s > 0.55): DM > 50%, universal behavior

---

# EXPERIMENT A: EIGENGAPS VS PROJECTOR ROTATION

## Hypothesis
Small eigengaps in the bulk lead to projector instability and DM emergence.

## Key Results

| Metric | Value |
|--------|-------|
| Mean blocks per run | 192.2 |
| Bulk blocks suppress DM | Yes (0 bulk → DM=0.68; 2+ bulk → DM≈0) |
| Gap-rotation correlation | +0.434 (moderate positive) |
| Gap-DM correlation | +0.239 (weak positive) |
| Rotation-DM correlation | +0.243 (weak positive) |

## DM Rate by Sparsity

| Sparsity | DM Rate | Zero DM Rate |
|----------|---------|--------------|
| 0.00 | 0.000 | 98.4% |
| 0.30 | 0.006 | 39.1% |
| 0.40 | 0.276 | 6.2% |
| 0.50 | 0.536 | 1.6% |
| 0.60 | 0.558 | 0.0% |
| 0.99 | 0.827 | 0.0% |

## Critical Finding
**Bulk blocks suppress delocalization.** Zero bulk blocks → high DM (0.68); 2+ bulk blocks → near-zero DM. The phase transition at s ≈ 0.28-0.30 coincides with bulk block disappearance.

---

# EXPERIMENT B: SPECTRAL SPREAD METRICS

## Hypothesis
DM features exhibit large spectral spread (high entropy, participation ratio, drift).

## Effect Sizes (Cohen's d)

| Metric | Mean d | DM Larger Rate | Verdict |
|--------|--------|----------------|---------|
| Entropy | -0.154 | 60.1% | Weak support |
| Participation Ratio | -0.175 | 60.5% | Weak support |
| **Spectral Variance** | +0.536 | 8.4% | **REJECTED** (WB higher) |
| **Drift** | -0.400 | **89.8%** | **Strong support** |
| Bulk Mass | +0.363 | 2.3% | WB has more bulk mass |

## Predictive Model

| Metric | Value |
|--------|-------|
| Mean AUC | 0.743 |
| AUC > 0.8 | 44.8% |
| AUC > 0.9 | 34.3% |
| Best AUC (at s≈0.42) | 0.968 |

## Critical Finding
**Drift is the primary discriminator.** DM features show much higher temporal instability (90% of cases), while spectral variance is actually LOWER for DM (contrary to hypothesis).

---

# EXPERIMENT C: WITHIN-CLUSTER VARIANCE

## Hypothesis
DM features concentrate in bulk (λ ≤ 1) eigenspaces where projector instability is highest.

## DM Concentration

| Region | DM Rate | Percentage |
|--------|---------|------------|
| Bulk (λ ≤ 1) | 0.077 | 7.7% |
| Spiked (λ > 1) | 0.397 | 39.7% |

- DM concentrates in spiked: **99.1%** of runs
- DM concentrates in bulk: **0.9%** of runs

## Variance Analysis

| Correlation | Value | Expected |
|-------------|-------|----------|
| λ vs variance | -0.036 | Negative |
| Gap vs variance | +0.026 | Negative |
| Bulk higher variance | 47.9% | >50% |

## Critical Finding
**DM is a spiked-eigenspace phenomenon, NOT a bulk phenomenon.** The hypothesis that DM concentrates in bulk (stable) eigenspaces is strongly rejected. DM overwhelmingly appears in unstable (λ > 1) regions.

---

# EXPERIMENTS D-H: GLOBAL PROJECTIVE LINEARITY

## Hypothesis
D_i(t) = α(t) × N_i(t) holds universally, and normalizing by α(t) should stabilize DM features.

## D: Cross-Sectional Linearity

| Metric | Value |
|--------|-------|
| Mean R² | **0.914** |
| R² > 0.99 | 82.0% |
| R² > 0.999 | 70.2% |
| DM slope ratio | 0.998 |
| Spiked slope ratio | 1.000 |

## E: Slope Matching

- Slope ratio across subgroups: **0.998 ± 0.004**
- Near-unity rate: **100%** (when valid)
- **Slopes are universal across all feature types**

## F: Best α Predictor

| Predictor | Wins | Percentage |
|-----------|------|------------|
| trace | 1,223 | **38.2%** |
| mean_κ | 1,039 | 32.5% |
| median_κ | 938 | 29.3% |

## G: Universal Diffusion

| Metric | Value |
|--------|-------|
| κ CV | 0.037 (very low) |
| JS divergence | 0.570 |

## H: Normalization Test

| Metric | Value |
|--------|-------|
| DM stabilization rate | **71.7%** |
| Rate at s > 0.5 | **>98%** |

## Critical Finding
**DM is largely a global α(t) effect.** Normalizing D by the global diffusion coefficient α(t) stabilizes 72% of DM features. At high sparsity, this rate exceeds 98%. The remaining DM has genuine feature-specific dynamics.

---

# SYNTHESIS: UNIFIED THEORY OF DARK MATTER

## Causal Chain

```
High Sparsity (s > 0.28)
        ↓
Bulk blocks disappear
        ↓
Features pushed to spiked eigenspaces (λ > 1)
        ↓
Global α(t) dynamics dominate
        ↓
D_i(t) = α(t) × N_i(t) with feature-independent α
        ↓
Drift in α(t) creates apparent DM behavior
        ↓
Normalization by α(t) recovers linearity
```

## What DM IS

1. **A manifestation of global spectral dynamics**: α(t) varies, not individual D_i/N_i ratios
2. **Associated with spiked eigenspaces**: λ > 1, not bulk (λ ≤ 1)
3. **Characterized by high drift**: Temporal instability in projections
4. **Recoverable through normalization**: 72% of DM stabilizes when dividing by α(t)

## What DM is NOT

1. **Not feature-specific**: Slope ratios are unity
2. **Not a bulk phenomenon**: 99% in spiked eigenspaces
3. **Not due to high spectral variance**: WB features have higher variance
4. **Not due to small eigengaps**: Gap-variance correlation is wrong sign

---

# PARAMETER DEPENDENCIES SUMMARY

## By m_hidden (Hidden Dimension)

| m | DM Rate | R² | α | Bulk DM | Spiked DM |
|---|---------|-----|-----|---------|-----------|
| 16 | 0.033 | 0.995 | 0.73 | 0.000 | 0.033 |
| 144 | 0.236 | 0.990 | 0.63 | 0.066 | 0.284 |
| 272 | 0.416 | 0.943 | 0.63 | 0.052 | 0.532 |
| 400 | 0.484 | 0.872 | 0.65 | 0.077 | 0.506 |
| 512 | 0.514 | 0.753 | 0.66 | - | 0.530 |

**Trend**: DM rate increases with m; R² decreases at large m.

## By Sparsity

| s | DM Rate | Zero DM | R² | α | Stab Rate |
|---|---------|---------|-----|-----|-----------|
| 0.00 | 0.000 | 98% | 1.00 | 1.00 | 1.6% |
| 0.30 | 0.006 | 39% | 1.00 | 0.92 | 61% |
| 0.40 | 0.276 | 6% | 0.83 | 0.79 | 94% |
| 0.50 | 0.536 | 0% | 0.96 | 0.55 | 98% |
| 0.60 | 0.558 | 0% | 0.98 | 0.50 | 100% |
| 0.80 | 0.610 | 0% | 0.87 | 0.37 | 99% |
| 0.99 | 0.775 | 0% | 0.63 | 0.27 | 100% |

**Key transitions**:
- s ≈ 0.28: DM phase transition begins
- s ≈ 0.40: Best predictive model performance (AUC > 0.95)
- s > 0.55: All runs have DM; normalization nearly perfect

---

# DATA QUALITY

## Validation

| Check | Pass Rate |
|-------|-----------|
| κ consistency (Σ p_{ik}λ_k = κ) | 100% |
| Mean relative error | 4.6×10⁻⁵ |
| Max relative error | 6.1×10⁻⁵ |

## Error Rates by Experiment

| Experiment | Error Rate | Cause |
|------------|------------|-------|
| B (comparisons) | 39% | Insufficient DM or WB features |
| E (slope matching) | 36% | Insufficient valid comparisons |
| G (universal diffusion) | 36% | No DM features |

Errors occur primarily in low-sparsity runs where DM is absent.

---

# INDIVIDUAL EXPERIMENT REPORTS

For detailed analysis of each experiment, see:

- [`experiment_A/analysis_report.md`](experiment_A/analysis_report.md) - Eigengaps vs Projector Rotation
- [`experiment_B/analysis_report.md`](experiment_B/analysis_report.md) - Spectral Spread Metrics
- [`experiment_C/analysis_report.md`](experiment_C/analysis_report.md) - Within-Cluster Variance
- [`experiments_D_H/analysis_report.md`](experiments_D_H/analysis_report.md) - Global Projective Linearity

---

# APPENDIX: RAW AGGREGATED DATA

## Experiment A
```json
{
  "n_runs": 3200,
  "dm_rate_mean": 0.354,
  "dm_rate_std": 0.365,
  "n_blocks_mean": 192.23,
  "correlations": {
    "gap_vs_rotation_mean": 0.434,
    "gap_vs_dm_mean": 0.239,
    "rotation_vs_dm_mean": 0.243
  }
}
```

## Experiment B
```json
{
  "n_runs": 3200,
  "effect_sizes": {
    "entropy": {"mean": -0.154, "dm_larger_rate": 0.601},
    "drift": {"mean": -0.400, "dm_larger_rate": 0.898}
  },
  "predictive_model": {"auc_mean": 0.743},
  "falsifier": {"auc_mean": 0.680}
}
```

## Experiment C
```json
{
  "n_runs": 3200,
  "var_vs_lambda": {"bulk_higher_variance_rate": 0.479},
  "dm_concentration": {
    "bulk_dm_rate_mean": 0.077,
    "spiked_dm_rate_mean": 0.397,
    "dm_concentrates_in_bulk": false
  }
}
```

## Experiments D-H
```json
{
  "n_runs": 3200,
  "D_cross_sectional": {"late_r2_mean": 0.914, "dm_slope_ratio_mean": 0.998},
  "E_slope_matching": {"ratio_mean": 0.998, "near_unity_rate": 0.638},
  "F_alpha_predictor": {"trace": 1223, "mean_kappa": 1039, "median_kappa": 938},
  "G_universal_diffusion": {"kappa_cv_mean": 0.037},
  "H_normalization": {"stabilization_rate_mean": 0.717}
}
```

---

# CONCLUSIONS

1. **DM is global, not local**: The D = αN relationship is universal; DM arises from α(t) dynamics
2. **DM concentrates in spiked eigenspaces**: λ > 1, not bulk
3. **Drift is the signature**: DM features show high temporal instability
4. **Normalization recovers linearity**: 72% of DM stabilizes with α-correction
5. **Phase transition at s ≈ 0.28-0.30**: Critical threshold for DM emergence
6. **Trace predicts α**: Global spectral quantity controls diffusion

---

*Report compiled from 3,200 experimental configurations spanning m_hidden ∈ [16, 512] and sparsity ∈ [0, 0.99].*
