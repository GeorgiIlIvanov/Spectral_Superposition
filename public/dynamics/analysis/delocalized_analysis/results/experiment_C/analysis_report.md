# Experiment C: Within-Cluster Variance Analysis Report

**Generated:** 2026-01-22
**Data Source:** `all_results.json` (3,200 runs)
**Status:** Complete analysis

---

## 1. EXPERIMENTAL OVERVIEW

### 1.1 Hypothesis

Dark matter features should exhibit high within-cluster variance in the eigenvalue space, particularly concentrating in the bulk (λ ≤ 1) region where small eigengaps lead to projector instability.

### 1.2 Key Questions

1. Does variance correlate with eigenvalue (λ)? Higher variance in bulk?
2. Does variance correlate with eigengaps? Smaller gaps → higher variance?
3. Do DM features concentrate in bulk or spiked eigenspaces?

### 1.3 Cluster/Band Structure

| Metric | Mean | Std | Range |
|--------|------|-----|-------|
| Clusters per run | 219.1 | 117.0 | Variable |
| Bands per run | 1.9 | 0.6 | 1-3 |

The eigenvalue spectrum is partitioned into ~2 bands on average (bulk and spiked).

---

## 2. KEY FINDINGS

### 2.1 Variance vs Lambda Analysis

**Correlation between λ and within-cluster variance:**

- Mean correlation: **-0.036** (near zero)
- Bulk higher variance rate: **47.9%**

**Interpretation:** There is essentially no correlation between eigenvalue magnitude and within-cluster variance. The hypothesis that bulk eigenspaces (λ ≤ 1) have higher variance is **NOT supported** - variance is approximately equally distributed.

### 2.2 Variance vs Gap Analysis

**Correlation between eigengap and variance:**

- Mean correlation: **+0.026** (near zero)
- Expected: Negative (smaller gaps → higher variance)

**Interpretation:** The expected negative correlation (smaller gaps causing higher variance due to projector instability) is **NOT observed**. Gaps and variance appear uncorrelated.

### 2.3 DM Concentration: Bulk vs Spiked

**Critical Finding:**

| Region | DM Rate | Interpretation |
|--------|---------|----------------|
| Bulk (λ ≤ 1) | **0.077** | Low DM in stable region |
| Spiked (λ > 1) | **0.397** | High DM in unstable region |

- DM concentrates in **spiked**: **99.1%** of runs
- DM concentrates in bulk: **0.9%** of runs

**This is a major result:** Dark matter does NOT concentrate in the bulk (stable) eigenspaces. Instead, DM overwhelmingly concentrates in spiked (unstable, λ > 1) eigenspaces.

---

## 3. PARAMETER DEPENDENCIES

### 3.1 By Hidden Dimension (m_hidden)

| m | Overall DM | Bulk DM | Spiked DM | Spiked/Bulk Ratio |
|---|------------|---------|-----------|-------------------|
| 16 | 0.033 | 0.000 | 0.033 | ∞ |
| 80 | 0.140 | 0.052 | 0.130 | 2.5× |
| 144 | 0.236 | 0.066 | 0.284 | 4.3× |
| 208 | 0.326 | 0.052 | 0.431 | 8.3× |
| 272 | 0.416 | 0.052 | 0.532 | 10.2× |
| 336 | 0.465 | 0.058 | 0.519 | 9.0× |
| 400 | 0.484 | 0.077 | 0.506 | 6.6× |
| 464 | 0.523 | 0.049 | 0.530 | 10.8× |

**Observations:**
1. Bulk DM stays low and relatively constant (0.05-0.08) across all m
2. Spiked DM grows with m (0.03 → 0.53)
3. The spiked/bulk ratio increases dramatically with m

### 3.2 By Sparsity

| Sparsity | Overall DM | Bulk DM | Spiked DM |
|----------|------------|---------|-----------|
| 0.00 | 0.000 | 0.000 | 0.000 |
| 0.10 | 0.000 | 0.000 | 0.000 |
| 0.20 | 0.001 | 0.000 | 0.000 |
| 0.30 | 0.006 | 0.000 | 0.005 |
| 0.40 | 0.276 | 0.000 | 0.330 |
| 0.50 | 0.536 | 0.050 | 0.701 |
| 0.60 | 0.558 | **0.255** | 0.742 |
| 0.70 | 0.511 | 0.221 | 0.595 |
| 0.80 | 0.610 | 0.000 | 0.614 |
| 0.90 | 0.664 | 0.000 | 0.665 |

**Critical Observations:**
1. **s < 0.40:** No DM in either bulk or spiked
2. **s ≈ 0.40-0.50:** DM appears primarily in spiked eigenspaces
3. **s ≈ 0.55-0.70:** Some bulk DM emerges (max ~25% at s≈0.60)
4. **s > 0.80:** Bulk DM disappears again, all DM in spiked

The bulk DM peak at s ≈ 0.55-0.70 suggests a narrow window where projector instability in the bulk region can produce DM.

---

## 4. DETAILED ANALYSIS

### 4.1 Variance Distribution Across Bands

The within-cluster variance analysis reveals:

1. **No systematic λ-variance relationship**: Variance is not concentrated in low-λ regions
2. **No gap-variance relationship**: Small gaps don't systematically produce higher variance
3. **Variance is feature-specific**, not eigenvalue-specific

### 4.2 Why DM Concentrates in Spiked Eigenspaces

Possible explanations:
1. **Eigenvalue instability**: λ > 1 implies exponential growth, inherently creating DM behavior
2. **Projector rotation**: Spiked eigenspaces may have faster projector dynamics
3. **Selection effect**: Features that become DM may be pushed into spiked eigenspaces

### 4.3 The Bulk DM Window (s ≈ 0.55-0.70)

This narrow sparsity window where bulk DM appears suggests:
1. Sufficient network fragmentation to allow projector instability
2. But not so much that all bulk structure disappears
3. This matches the "transition region" identified in Experiment A

---

## 5. CRITICAL INSIGHTS SUMMARY

### 5.1 Hypothesis Testing

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| Bulk has higher variance | **REJECTED** | ~50-50 split |
| Small gaps → higher variance | **REJECTED** | ρ ≈ 0.026 (positive!) |
| DM concentrates in bulk | **REJECTED** | 99% in spiked |
| Variance drives DM | **REJECTED** | No correlation |

### 5.2 Key Conclusions

1. **DM is a spiked-eigenspace phenomenon**: 99% of runs show DM concentrating in λ > 1 regions
2. **Bulk is stable**: Very low DM rates in bulk eigenspaces (7.7% vs 39.7% overall)
3. **Variance is not the mechanism**: Within-cluster variance does not explain DM
4. **The bulk DM window**: A narrow sparsity range (0.55-0.70) allows some bulk DM

### 5.3 Implications for Theory

The concentration of DM in spiked eigenspaces suggests:
- **Eigenvalue magnitude (λ > 1)** may be the primary driver, not eigengaps
- The "spectral spread" story from Experiment B may be secondary
- Need to investigate what pushes features into spiked eigenspaces

---

## 6. RAW AGGREGATED STATISTICS

```json
{
  "n_runs": 3200,
  "var_vs_lambda": {
    "lambda_vs_var_rho_mean": -0.036,
    "bulk_higher_variance_rate": 0.479
  },
  "var_vs_gap": {
    "gap_vs_var_rho_mean": 0.026,
    "gap_negatively_correlated_expected": "Smaller gaps -> higher variance"
  },
  "dm_concentration": {
    "bulk_dm_rate_mean": 0.077,
    "spiked_dm_rate_mean": 0.397,
    "dm_concentrates_in_bulk": false
  }
}
```

---

## 7. OPEN QUESTIONS

1. Why does bulk DM only appear in the s ≈ 0.55-0.70 window?
2. What determines whether a feature ends up in bulk vs spiked eigenspaces?
3. Is λ > 1 a cause or consequence of DM behavior?
4. How does this relate to the projector rotation findings from Experiment A?
