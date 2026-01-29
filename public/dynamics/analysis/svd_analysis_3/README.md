# SVD Analysis v3: Eigenspace-Based Localization (CORRECTED)

## Critical Correction from v2

**The previous analysis (svd_analysis_2) was fundamentally flawed.**

### The Problem

v2 computed projections onto individual **eigenvectors**:
```
p_ik = |u_k^T w_i|² / ||w_i||²  (WRONG)
```

This is incorrect because:

1. **Degenerate eigenvalues**: When multiple eigenvectors share the same eigenvalue λ, they span a multi-dimensional eigenspace
2. **Arbitrary basis**: The choice of orthonormal basis within a degenerate eigenspace is arbitrary (any rotation works)
3. **Physical meaning**: Features cluster along **eigenspaces** (subspaces), not individual eigenvectors

### The Correct Approach

v3 computes projections onto **eigenspaces** (subspaces spanned by all eigenvectors with the same eigenvalue):

```
p_{i,λ} = Σ_{k: λ_k = λ} |u_k^T w_i|² / ||w_i||²  (CORRECT)
```

This is:
- **Basis-invariant**: Doesn't depend on arbitrary choice of eigenvectors within degenerate subspace
- **Physically meaningful**: Measures how much of feature i lives in the eigenspace associated with λ
- **Properly normalized**: Σ_λ p_{i,λ} = 1

## Algorithm

### Step 1: Identify Eigenspaces

Group eigenvectors by their eigenvalue (with relative tolerance for near-degeneracy):

```python
def identify_eigenspaces(eigenvalues, rel_tol=1e-4):
    # eigenvalues = [λ₁, λ₁, λ₂, λ₂, λ₂, λ₃, ...]
    # returns: space_assignments = [0, 0, 1, 1, 1, 2, ...]
    #          space_eigenvalues = [λ₁, λ₂, λ₃, ...]
```

For example, if eigenvalues are [10, 10, 5, 5, 5, 1]:
- Eigenspace 0: λ = 10, dimension 2 (eigenvectors 0, 1)
- Eigenspace 1: λ = 5, dimension 3 (eigenvectors 2, 3, 4)
- Eigenspace 2: λ = 1, dimension 1 (eigenvector 5)

### Step 2: Compute Eigenspace Projections

For each feature i and eigenspace s:
```
p_{i,s} = Σ_{k ∈ space s} |u_k^T w_i|² / ||w_i||²
```

This is the total squared projection onto the entire subspace, which is invariant to basis choice.

### Step 3: Compute Localization Metrics

Over **eigenspaces** (not eigenvectors):

- **Max eigenspace projection**: max_s p_{i,s}
- **Participation ratio**: 1 / Σ_s p_{i,s}² (effective number of eigenspaces)
- **Entropy**: -Σ_s p_{i,s} log(p_{i,s})

### Step 4: Cluster and Analyze

Cluster features by their dominant **eigenspace** (not eigenvector), then compute slopes as before.

## Key Difference in Interpretation

| Metric | v2 (Wrong) | v3 (Correct) |
|--------|-----------|--------------|
| Projection | Onto single eigenvector | Onto entire eigenspace |
| Participation ratio | # of eigenvectors | # of eigenspaces |
| Max projection | Projection onto best eigenvector | Projection onto best eigenspace |
| Clustering | By dominant eigenvector | By dominant eigenspace |

## Why This Matters

Consider a 2D degenerate eigenspace (λ₁ = λ₂ = λ):

- **v2 approach**: A feature aligned with u₁ + u₂ would have p_{i,1} = p_{i,2} = 0.5, appearing "delocalized"
- **v3 approach**: The same feature has p_{i,λ} = 1.0 (fully in the eigenspace), correctly appearing "localized"

The conjecture κ ≈ 1/λ relates slopes to **eigenvalues**, not individual eigenvectors. Using eigenspace projections is the only physically correct approach.

## Usage

```bash
source /home/georgi/Spectral_Superposition/public/dynamics/venv/bin/activate
python slope_eigenspace_analysis.py --num-gpus 8 --eigenvalue-tol 1e-4
```

The `--eigenvalue-tol` parameter controls how close eigenvalues must be to be considered degenerate (default: 1e-4 relative tolerance).

## Expected Outputs

- `eigenspace_cluster_data.csv` - Filtered cluster data with eigenspace localization
- `all_eigenspace_stats.csv` - All cluster statistics
- `run_stats.csv` - Per-file statistics including degeneracy ratio
- `summary_statistics.json` - Aggregated results by localization quartile
- `slope_eigenspace_localization.png/pdf` - Main visualization
- `stratified_by_eigenspace_loc.png/pdf` - Stratified analysis
- `eigenspace_analysis.png/pdf` - Degeneracy and localization analysis
