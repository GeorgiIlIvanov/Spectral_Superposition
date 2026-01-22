# Temporal Analysis of Spectral Superposition

**Goal**: Investigate whether "dark matter" features (those with poor D_i ~ ||W_i||² linear scaling) are transient states during training that eventually stabilize into clean eigenspaces, or represent a persistent structural phenomenon.

## Background

Previous analyses established:

1. **Phase Analysis**: Features cluster on rays in the (||W_i||², D_i) phase diagram, with slopes corresponding to 1/λ for eigenspaces with eigenvalue λ.

2. **SVD Analysis**: For features cleanly assigned to eigenspaces (λ > 1), the relationship κ ≈ 1/λ holds with R² = 0.94 (Pearson r = 0.97).

3. **Stratified Linearity**: 73-84% of features are "dark matter" with R² < 0.9 for the D_i vs ||W_i||² fit. These features have:
   - Lower eigenspace concentration (spread across multiple eigenspaces)
   - Higher eigenspace entropy
   - Negative curvature in their D_i(t) vs ||W_i(t)|² trajectories

**Key Question**: Is the dark matter a transient phenomenon (features "in transit" between eigenspaces during training) or a persistent structural feature of superposition?

---

## Analysis Plan

### Analysis 1: Dark Matter Evolution
**File**: `01_dark_matter_evolution.py`

Track the **fraction of dark matter features** as a function of training step.

**Metrics computed at each checkpoint t**:
- For each feature i, compute R² of D_i(τ) vs ||W_i(τ)||² for τ ∈ [0, t] (cumulative)
- Count features with R² < 0.9 (dark matter) vs R² ≥ 0.9 (well-behaved)
- Plot: dark_matter_fraction(t) for different sparsity regimes

**Expected outcomes**:
- If dark matter is transient: fraction should decrease with training
- If persistent: fraction should plateau or oscillate

---

### Analysis 2: Eigenspace Stability & Hopping
**File**: `02_eigenspace_stability.py`

Track how features migrate between eigenspace clusters during training.

**Metrics**:
- For each feature, compute dominant eigenspace assignment at each checkpoint
- Define "eigenspace hopping" as a change in dominant eigenspace between consecutive checkpoints
- Compute hopping rate: H(t) = fraction of features that changed eigenspace at step t
- Compute stabilization time: first checkpoint after which a feature never hops again

**Outputs**:
- Hopping rate vs training step (should decrease if eigenspaces stabilize)
- Distribution of stabilization times
- Correlation between hopping frequency and final R² (dark matter status)
- Sankey diagrams showing eigenspace migration flows

---

### Analysis 3: Instantaneous vs Cumulative Linearity
**File**: `03_instantaneous_linearity.py`

Compare linear fit quality using different temporal windows.

**Two approaches**:
1. **Cumulative R²**: Fit D_i vs ||W_i||² using all checkpoints [0, t]
2. **Instantaneous slope**: Compute local slope dD_i/d||W_i||² at each checkpoint

**Key questions**:
- Does instantaneous slope converge to a stable value?
- Is the poor R² due to slope changes during training, or intrinsic nonlinearity?

**Outputs**:
- Instantaneous slope κ_i(t) trajectories for sampled features
- Slope variance within features over time
- Comparison: features that stabilize early vs late vs never

---

### Analysis 4: Slope-Eigenvalue Relationship Over Time
**File**: `04_slope_eigenvalue_temporal.py`

Track how well the κ ≈ 1/λ conjecture holds at each checkpoint.

**Metrics at each checkpoint**:
- Compute cluster slopes κ_k(t) for each eigenspace cluster
- Compute corresponding eigenvalues λ_k(t)
- Test correlation κ_k(t) × λ_k(t) ≈ 1 at each t

**Outputs**:
- κλ distribution at early, mid, late training
- Mean(κλ) and std(κλ) vs training step
- Identify eigenvalue regimes (λ > 1 vs λ < 1) where the conjecture holds/fails over time

---

### Analysis 5: Feature Trajectory Classification
**File**: `05_trajectory_classification.py`

Classify features by their temporal trajectory shapes in the (||W_i||², D_i) space.

**Trajectory types**:
1. **Linear monotonic**: D_i increases linearly with ||W_i||² throughout training
2. **Curved stable**: Nonlinear but stable trajectory (converges to a ray)
3. **Oscillatory**: Features that move back and forth in phase space
4. **Transient**: Features that change slope mid-training (switch rays)
5. **Collapsed**: Features that shrink to near-zero norm

**Outputs**:
- Distribution of trajectory types by sparsity and m_hidden
- Correlation between trajectory type and final R²
- Example trajectory visualizations for each type

---

### Analysis 6: Eigenspace Concentration Dynamics
**File**: `06_concentration_dynamics.py`

Track eigenspace concentration and entropy over training.

**Metrics for each feature at each checkpoint**:
- Concentration: c_i(t) = max_k |u_k^T W_i|² / ||W_i||²
- Entropy: H_i(t) = -Σ_k p_k log(p_k) where p_k = |u_k^T W_i|² / ||W_i||²

**Questions**:
- Do features become more concentrated over time?
- Is there a critical time where concentration stabilizes?
- Correlation between concentration trajectory and dark matter status

---

### Analysis 7: Phase Diagram Animation & Snapshots
**File**: `07_phase_animation.py`

Create visualizations of the phase diagram evolution.

**Outputs**:
- Animated GIF of (||W_i||², D_i) scatter evolving over training
- Highlight dark matter vs well-behaved features with different colors
- Show eigenspace assignments changing
- Create "feature tracers" following individual features through time

---

### Analysis 8: Aggregate Summary Statistics
**File**: `08_aggregate_summary.py`

Compile comprehensive statistics across all experiments.

**Summary metrics**:
- Dark matter fraction at final checkpoint by (sparsity, m_hidden)
- Mean stabilization time by regime
- Fraction of features that ever hopped eigenspaces
- Correlation matrix of all temporal metrics

**Output**: Summary JSON and comprehensive plots

---

## File Structure

```
temporal_analysis/
├── README.md                          # This file
├── 01_dark_matter_evolution.py        # Dark matter fraction over time
├── 02_eigenspace_stability.py         # Eigenspace hopping analysis
├── 03_instantaneous_linearity.py      # Local vs global linearity
├── 04_slope_eigenvalue_temporal.py    # κ ≈ 1/λ over checkpoints
├── 05_trajectory_classification.py    # Feature trajectory types
├── 06_concentration_dynamics.py       # Eigenspace concentration over time
├── 07_phase_animation.py              # Animated phase diagrams
├── 08_aggregate_summary.py            # Compile all results
├── run_all_analyses.sh                # Master script
├── plots/                             # Output visualizations
│   ├── dark_matter_evolution.png
│   ├── eigenspace_stability.png
│   ├── hopping_rate.png
│   ├── slope_convergence.png
│   ├── trajectory_examples.png
│   ├── concentration_dynamics.png
│   ├── phase_evolution.gif
│   └── summary_heatmaps.png
└── results/                           # Output data
    ├── dark_matter_temporal.json
    ├── eigenspace_hopping.npz
    ├── trajectory_classes.json
    └── aggregate_summary.json
```

---

## Usage

```bash
# Run individual analysis
source ../venv/bin/activate
python 01_dark_matter_evolution.py

# Run all analyses
chmod +x run_all_analyses.sh
./run_all_analyses.sh

# Quick test with subset of data
python 01_dark_matter_evolution.py --sample 50
```

---

## Expected Findings

Based on preliminary observations and the "dark matter" characterization, we hypothesize:

1. **Dark matter fraction will NOT decrease monotonically** - suggesting it's not purely transient
2. **Eigenspace hopping will be concentrated in early training** - features largely stabilize
3. **Some features will show persistent oscillation** - never cleanly settling into one eigenspace
4. **The λ < 1 regime will show worse temporal stability** - consistent with the breakdown of the spectral interpretation for small eigenvalues

These analyses will provide evidence for or against the hypothesis that dark matter is a transient training phenomenon.
