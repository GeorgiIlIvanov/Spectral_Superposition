# Experiment B: Spectral Spread Metrics Analysis Report

**Generated:** 2026-01-22
**Data Source:** `all_results.json` (11.8MB, 3,200 runs)
**Status:** Complete analysis

---

## 1. EXPERIMENTAL OVERVIEW

### 1.1 Hypothesis

Dark matter (DM) features exhibit large spectral spread - their energy is distributed across multiple eigenspaces rather than being concentrated in a single dominant one.

### 1.2 Key Quantities Measured

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Entropy** | H_i(t) = -Σ_k p_{ik}(t) log(p_{ik}(t)) | Higher = more spread |
| **Participation Ratio** | PR_i(t) = 1 / Σ_k p_{ik}² | Effective number of eigenspaces |
| **Spectral Variance** | Var_λ_i(t) = Σ_k p_{ik} λ_k² - κ_i² | Spread in eigenvalue scale |
| **Bulk Mass** | m_{≤1}_i(t) = Σ_{k: λ_k ≤ 1} p_{ik}(t) | Mass in stable eigenspaces |
| **Max Projection** | p_max_i(t) = max_k p_{ik}(t) | Concentration measure |
| **Drift** | JS divergence between p_i(t) and p_i(t+1) | Temporal instability |

### 1.3 Parameter Space

| Parameter | Values | Total |
|-----------|--------|-------|
| m_hidden | 16, 32, ..., 512 (32 values) | |
| sparsity | 0.0 to 0.99 (50 values) | |
| seed | 0, 1 | |
| **Total** | | **3,200 runs** |

---

## 2. KEY FINDINGS

### 2.1 Validation Results

**Kappa Consistency Check (p_{ik} computation):**
- Mean relative error: 4.57×10⁻⁵
- Max relative error: 6.08×10⁻⁵
- **100% pass rate** - all runs validated

This confirms that Σ_k p_{ik}(t) × λ_k(t) ≈ κ_actual(t) with high precision.

### 2.2 Dark Matter Rate

**Overall Statistics:**
- Mean DM rate: **0.354** (std: 0.365)
- Median: 0.228
- Zero DM rate: **28.2%** of runs
- High DM (>0.5): **37.1%** of runs

### 2.3 Effect Sizes (Cohen's d)

| Metric | Mean d | Std d | DM Larger Rate | Interpretation |
|--------|--------|-------|----------------|----------------|
| **Entropy** | -0.154 | 0.584 | 60.1% | DM has slightly higher entropy |
| **Participation Ratio** | -0.175 | 0.574 | 60.5% | DM uses more eigenspaces |
| **Spectral Variance** | 0.536 | 0.401 | 8.4% | Well-behaved has MUCH higher variance |
| **Drift** | -0.400 | 0.386 | **89.8%** | DM has much higher drift |
| **Pmax** | 0.203 | 0.547 | 35.2% | Well-behaved is more concentrated |
| **Bulk Mass** | 0.363 | 0.418 | 2.3% | Well-behaved has more bulk mass |

**Critical Finding:** DM features show:
1. **Higher drift** (90% of cases) - main distinguishing characteristic
2. Slightly higher entropy/PR (~60% of cases)
3. **Lower** spectral variance (contrary to hypothesis!)
4. Less bulk mass (consistent with spiked eigenspace dominance)

### 2.4 Predictive Model Performance

**Logistic Regression Model (DM ~ spread metrics):**

| Metric | Mean | Std | Interpretation |
|--------|------|-----|----------------|
| **AUC** | 0.743 | 0.201 | Good predictive power |
| AUC > 0.7 | 55.8% | | |
| AUC > 0.8 | 44.8% | | |
| AUC > 0.9 | 34.3% | | Excellent in many runs |

**Model interpretation:** Spread metrics can predict DM status with AUC=0.74 on average, indicating moderate but meaningful predictive value.

### 2.5 Falsifier Test (λ ≤ 1 Subset)

Tests whether spread metrics separate DM from well-behaved features *within* the bulk (λ ≤ 1) subset:

- Mean AUC: **0.680**
- Spread separates in bulk: **15.1%** of valid runs

**Interpretation:** Spread metrics provide some separation even within the λ ≤ 1 subset, but the effect is weaker. This partially supports the hypothesis that spread is not merely a proxy for λ.

---

## 3. PARAMETER DEPENDENCIES

### 3.1 By Hidden Dimension (m_hidden)

| m | DM Rate | Zero% | High% | Model AUC |
|---|---------|-------|-------|-----------|
| 16 | 0.033 | 53% | 0% | 0.699 |
| 80 | 0.140 | 28% | 8% | 0.780 |
| 144 | 0.236 | 21% | 12% | 0.788 |
| 208 | 0.326 | 17% | 23% | 0.789 |
| 272 | 0.416 | 22% | 56% | 0.764 |
| 336 | 0.465 | 31% | 60% | 0.722 |
| 400 | 0.484 | 38% | 56% | 0.697 |
| 512 | 0.514 | 27% | 60% | 0.622 |

**Observations:**
1. DM rate increases with m_hidden (0.033 → 0.514)
2. Best model AUC at intermediate m (144-208): ~0.79
3. AUC decreases at large m - spread becomes less discriminative

### 3.2 By Sparsity

| Sparsity | DM Rate | Zero% | Model AUC |
|----------|---------|-------|-----------|
| 0.00 | 0.000 | 98% | 0.507 |
| 0.20 | 0.001 | 91% | 0.533 |
| 0.30 | 0.006 | 39% | 0.739 |
| 0.40 | 0.276 | 6% | **0.968** |
| 0.50 | 0.536 | 0% | 0.960 |
| 0.60 | 0.558 | 0% | 0.811 |
| 0.70 | 0.525 | 0% | 0.739 |
| 0.80 | 0.588 | 0% | 0.884 |
| 0.90 | 0.664 | 0% | 0.670 |
| 0.99 | 0.775 | 0% | 0.633 |

**Critical Phase Transition:**
- s < 0.30: Almost no DM, model useless (AUC ≈ 0.5)
- s ≈ 0.40-0.50: Phase transition, **best model performance** (AUC > 0.95)
- s > 0.55: Universal DM, model less discriminative

---

## 4. DETAILED COMPARISON: DM vs WELL-BEHAVED

### 4.1 Mean Values (valid comparisons)

| Metric | DM Mean | WB Mean | Difference |
|--------|---------|---------|------------|
| Entropy | 4.343 | 4.107 | +0.236 (DM higher) |
| PR | 73.19 | 51.87 | +21.32 (DM higher) |

### 4.2 Interpretation

1. **Entropy**: DM features have ~5.7% higher entropy on average, indicating more spread across eigenspaces
2. **Participation Ratio**: DM features use ~41% more effective eigenspaces (73 vs 52)
3. **Drift**: Most distinguishing feature - DM features are temporally unstable

---

## 5. CRITICAL INSIGHTS SUMMARY

### 5.1 Hypothesis Support

| Claim | Support | Evidence |
|-------|---------|----------|
| DM has high entropy | **Partial** | 60% higher, small effect size |
| DM has high PR | **Partial** | 60% higher, small effect size |
| DM has high variance | **REJECTED** | WB has higher variance (92%) |
| DM has high drift | **STRONG** | 90% of cases |
| Spread separates in bulk | **Weak** | Only 15% of runs |

### 5.2 Key Conclusions

1. **Drift is the primary discriminator** - DM features show much higher temporal instability in their spectral projections
2. Entropy and PR provide weak but consistent signal
3. **Spectral variance is NOT higher for DM** - contradicts initial hypothesis
4. Model performance peaks in the phase transition region (s ≈ 0.4-0.5)
5. At high sparsity, spread metrics become less discriminative as most features are DM

### 5.3 Open Questions

- Why do well-behaved features have higher spectral variance?
- Is drift capturing the same information as projector rotation from Experiment A?
- Why does model performance degrade at high sparsity?

---

## 6. RAW AGGREGATED STATISTICS

```json
{
  "n_runs": 3200,
  "validation_pass_rate": 1.0,
  "dm_rate_mean": 0.354,
  "dm_rate_std": 0.365,
  "effect_sizes": {
    "entropy": {"mean": -0.154, "std": 0.584, "dm_larger_rate": 0.601},
    "participation_ratio": {"mean": -0.175, "std": 0.574, "dm_larger_rate": 0.605},
    "spectral_variance": {"mean": 0.536, "std": 0.401, "dm_larger_rate": 0.084},
    "drift": {"mean": -0.400, "std": 0.386, "dm_larger_rate": 0.898},
    "pmax": {"mean": 0.203, "std": 0.547, "dm_larger_rate": 0.352},
    "bulk_mass": {"mean": 0.363, "std": 0.418, "dm_larger_rate": 0.023}
  },
  "predictive_model": {
    "auc_mean": 0.743,
    "auc_std": 0.201
  },
  "falsifier": {
    "auc_mean": 0.680,
    "spread_separates_rate": 0.151
  }
}
```
