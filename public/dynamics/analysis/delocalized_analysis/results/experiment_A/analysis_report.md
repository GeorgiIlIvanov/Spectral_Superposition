# Experiment A: Comprehensive Analysis Report

**Generated:** 2026-01-22
**Data Source:** `all_results.json` (168MB, 3,200 runs)
**Status:** Initial analysis - to be refined

---

## 1. EXPERIMENTAL SETUP

### 1.1 Overview

This experiment investigates the relationship between network sparsity, hidden layer dimensions, and delocalized mode (DM) dynamics in spectral superposition systems. The experiment systematically varies two primary parameters (m_hidden and sparsity) across multiple random seeds to characterize phase transitions and identify the conditions under which delocalization emerges.

### 1.2 Parameter Space

| Parameter | Description | Values |
|-----------|-------------|--------|
| **m_hidden** | Hidden layer dimension | 32 values: 16, 32, 48, ..., 512 (step 16) |
| **sparsity** | Connection sparsity (fraction of zero weights) | 50 values: 0.0 to 0.99 (uniform spacing) |
| **seed** | Random seed for reproducibility | 2 values: 0, 1 |
| **Total runs** | Full factorial design | 32 × 50 × 2 = **3,200 runs** |

### 1.3 Measured Quantities

For each run, the following are recorded:

- **Block decomposition**: Number of blocks, block dimensions, bulk vs non-bulk classification
- **Spectral properties**: Eigenvalues (lambda), spectral gaps (gap_min, gap_mean)
- **Late rotations**: Rotation magnitudes per block
- **Delocalized mode (DM) rate**: Overall and binned by gap quintile
- **Correlations**: Gap-rotation, gap-DM, rotation-DM correlations

---

## 2. KEY FINDINGS

### 2.1 Delocalized Mode (DM) Rate

**Overall Statistics:**
- Mean DM rate: **0.354** (std: 0.365)
- Median: 0.228
- 28.2% of runs have **zero DM rate**
- 37.1% have DM rate > 0.5
- 2.0% achieve **perfect delocalization** (DM = 1.0)

**Critical Phase Transition:**
- **Critical sparsity ≈ 0.28-0.30**: DM rate transitions from near-zero to significant values
- Below s=0.20: ~90%+ runs have zero DM
- Above s=0.40: <10% runs have zero DM
- Above s=0.55: **0% runs have zero DM**

**DM Rate by Sparsity (key thresholds):**

| Sparsity | Mean DM | Zero Rate |
|----------|---------|-----------|
| 0.00 | 0.0000 | 98.4% |
| 0.30 | 0.0059 | 39.1% |
| 0.40 | 0.2764 | 6.2% |
| 0.50 | 0.5361 | 1.6% |
| 0.60 | 0.5583 | 0.0% |
| 0.80 | 0.6103 | 0.0% |
| 0.99 | **0.8272** | 0.0% |

**DM Rate by m_hidden:**
- Small m (16): mean DM = 0.033, 53% zero rate
- Medium m (144): mean DM = 0.237, 21% zero rate
- Large m (464): mean DM = 0.523, 34% zero rate

---

### 2.2 Spectral Gap Analysis

**Gap Statistics:**
- Minimum gap per run: mean = 5×10⁻⁶ (extremely small)
- Mean gap per run: 0.0054
- **Gap decreases with m_hidden**: m=16 (0.035) → m=512 (0.001)

**Gap vs Sparsity (non-monotonic):**
- s=0.0: gap = 0.0042
- s=0.2: gap = 0.0008 (minimum)
- s=0.5: gap = 0.0038
- s=0.99: gap = 0.0056

---

### 2.3 Block Structure

**Block Statistics:**
- Mean blocks per run: **192** (std: 159)
- Range: 2 to 487 blocks
- Bulk blocks: mean = 1.61 (max 13)

**Block Dimension Distribution:**

| Dimension | Count | Percentage |
|-----------|-------|------------|
| dim = 1 | 581,190 | 94.5% |
| dim 2-10 | 33,146 | 5.4% |
| dim 11-50 | 112 | 0.02% |
| dim 51-100 | 126 | 0.02% |
| dim > 100 | 571 | 0.09% |

**Large blocks (>100 dim) appear only at low sparsity:**
- s≈0.0: 156 blocks, mean dim = 300
- s≈0.1: 254 blocks, mean dim = 304
- s≈0.2: 152 blocks, mean dim = 363
- s≈0.3: 9 blocks, mean dim = 478
- s>0.3: **No large blocks**

---

### 2.4 Bulk Block Analysis

**Critical Finding - Inverse Relationship:**

| Bulk Blocks | Mean DM Rate | Count |
|-------------|--------------|-------|
| 0 | **0.676** | 1,021 |
| 1 | 0.283 | 1,566 |
| 2+ | **~0.000** | 613 |

**Bulk blocks disappear at high sparsity:**
- s=0.0: mean 10 bulk blocks
- s=0.5: mean 0.75 bulk blocks (25% have zero)
- s≥0.8: **100% have zero bulk blocks**

---

### 2.5 Late Rotation Analysis

**Overall:**
- Mean: 0.9946 (highly conserved)
- **97.7% of rotations > 0.99**
- Only 0.39% < 0.5

**By Block Index:**
- Block 0-1: ~0.95 (slightly lower)
- Block 2+: 0.97-0.99 (increasing trend)

**By Sparsity:**
- s=0.0: 3.9% low rotations
- s≥0.8: **0% low rotations**

---

### 2.6 Correlation Analysis

**Gap vs Rotation:**
- Mean correlation: **0.434** (moderately positive)
- 82.1% positive correlations
- 48.2% strongly positive (>0.5)
- **Non-monotonic with sparsity:**
  - s=0.0: **-0.49** (negative!)
  - s=0.5-0.6: **0.65-0.69** (peak)
  - s=0.8-1.0: ~0.1 (weak)

**Gap vs DM:** mean ρ = 0.239 (weak positive)
**Rotation vs DM:** mean ρ = 0.243 (weak positive)

---

### 2.7 Enrichment Analysis (DM by Gap Quintile)

**Overall trend (not strictly monotonic):**

| Gap Bin | Mean DM | n |
|---------|---------|---|
| 0 (smallest gaps) | 0.362 | 1,926 |
| 1 | 0.454 | 1,743 |
| 2 | 0.494 | 1,953 |
| 3 | **0.531** | 2,291 |
| 4 (largest gaps) | 0.434 | 3,200 |

**Sparsity-dependent enrichment:**
- s=0.4: **3.79x** enrichment (bin_4/bin_0)
- s=0.5-0.6: ~1.8-1.9x enrichment
- s≥0.8: **No enrichment** (~1.0x)

---

### 2.8 Eigenvalue (Lambda) Analysis

**Critical Finding - Instability Signatures:**
- Mean λ = 2.18 (std: 1.29)
- Range: [0.82, 27.32]
- **99.16% of blocks have λ > 1.0**
- **90.67% have λ > 1.1**

**Bulk vs Non-bulk:**
- Bulk blocks: λ = 0.960 ± 0.033 (stable, <1)
- Non-bulk blocks: λ = 2.19 ± 1.30 (unstable, >1)

---

### 2.9 Seed Consistency

- Mean |seed0 - seed1| difference: 0.012 (very consistent)
- Only 1.7% of runs differ by >0.1
- **High variance region:** m=16-96, s=0.6-0.8 (up to 0.60 difference)

---

### 2.10 m_hidden × Sparsity Interaction

**Optimal sparsity for maximum DM rate (by m_hidden):**
- m=16: optimal s=0.77, DM=0.40
- m=144: optimal s=0.99, DM=0.85
- m=272: optimal s=0.95, DM=0.99
- m=400+: optimal s≈0.95, DM=1.00

**Scaling at fixed sparsity (s≈0.5):**
- m=64: DM=0.08
- m=128: DM=0.23
- m=256: DM=0.52
- m=384: DM=0.80
- m=512: DM=1.00

---

## 3. CRITICAL INSIGHTS SUMMARY

1. **Phase Transition at s≈0.28-0.30**: Sharp transition from localized (DM≈0) to delocalized regime. This is the critical sparsity threshold.

2. **Bulk Blocks Suppress Delocalization**: Zero bulk blocks → high DM (0.68); 2+ bulk blocks → near-zero DM. Bulk blocks act as "anchors" preventing delocalization.

3. **Large Blocks Only at Low Sparsity**: Blocks >100 dimensions only exist at s<0.3. At higher sparsity, the system fragments into many small (dim=1) blocks.

4. **Eigenvalue Instability**: 99%+ of non-bulk blocks have λ>1, indicating potential exponential growth dynamics. Only bulk blocks are stable (λ<1).

5. **Gap-Rotation Correlation Flips Sign**: Negative at s=0 (anti-correlated), peaks positive at s≈0.5-0.6, then weakens. The correlation structure fundamentally changes across the phase transition.

6. **Optimal Delocalization**: Achieved at high sparsity (s≈0.9-0.99) with larger m (≥200). DM rate reaches 1.0 for m≥320 at s≈0.95.

7. **Enrichment is Transient**: Gap-DM enrichment (higher gaps → more DM) is strongest in the transition region (s≈0.4) and disappears at high sparsity where DM is uniformly high.

---

## 4. OPEN QUESTIONS FOR FURTHER INVESTIGATION

- Why does the gap-rotation correlation flip sign at low sparsity?
- What mechanism causes bulk blocks to suppress delocalization?
- Is the λ>1 instability a feature or artifact of the block decomposition?
- Why is enrichment strongest at intermediate sparsity (s≈0.4)?
- Can the critical sparsity be predicted from m_hidden?

---

## 5. RAW AGGREGATED STATISTICS

From `aggregated_results.json`:

```json
{
  "n_runs": 3200,
  "dm_rate_mean": 0.354,
  "dm_rate_std": 0.365,
  "n_blocks_mean": 192.23,
  "enrichment": {
    "bin_0_mean": 0.362,
    "bin_1_mean": 0.454,
    "bin_2_mean": 0.494,
    "bin_3_mean": 0.531,
    "bin_4_mean": 0.434
  },
  "correlations": {
    "gap_vs_rotation_mean": 0.434,
    "gap_vs_dm_mean": 0.239,
    "rotation_vs_dm_mean": 0.243
  },
  "logistic_model": {
    "log_gap_coef_mean": 0.018,
    "rotation_coef_mean": 0.052
  }
}
```
