# Plot Documentation

Detailed documentation for all generated visualizations in the Dynamic Hopping Analysis.

---

## Individual Experiment Plots

These plots are generated for sample experiments (configurable number).

### 1. Rayleigh Quotient Heatmap
**Filename**: `<experiment>_rayleigh_heatmap.png`

![Rayleigh Heatmap Example](plots/example_rayleigh_heatmap.png)

#### Description
A 2D heatmap showing the evolution of log-transformed Rayleigh quotients x_i(t) = log(κ_i(t) + ε) over training.

#### Axes
- **X-axis**: Training checkpoint (time). Labels show actual training step numbers.
- **Y-axis**: Feature index (sorted by mean κ in descending order). Features with highest average Rayleigh quotient at top.
- **Color**: Value of x_i(t). Warmer colors = higher κ (more overlap with other features).

#### Interpretation
- **Horizontal bands**: Features maintaining stable κ throughout training
- **Vertical lines**: Global events affecting many features simultaneously
- **Bright spots/streaks**: Individual features with unusually high κ
- **Dark spots/streaks**: Features becoming more independent

#### What to Look For
1. **Overall pattern**: Is there a gradient from left to right? (Training effect)
2. **Feature groupings**: Do certain features cluster with similar dynamics?
3. **Anomalies**: Are there sudden changes (jumps) visible as color discontinuities?

---

### 2. Jump Events Visualization
**Filename**: `<experiment>_jump_events.png`

#### Description
A 4-panel figure visualizing detected jump events.

#### Panel 1 (Top-Left): Jump Scatter Plot
- **X-axis**: Checkpoint index (time)
- **Y-axis**: Feature index
- **Points**: Each point is a detected jump
- **Color**: Jump magnitude |Δx|
- **Interpretation**: Shows spatial-temporal distribution of jumps. Clusters indicate coordinated hopping.

#### Panel 2 (Top-Right): Jumps per Checkpoint
- **X-axis**: Checkpoint index
- **Y-axis**: Number of jumps
- **Red line**: Mean number of jumps per checkpoint
- **Interpretation**: Peaks indicate times when many features jumped simultaneously.

#### Panel 3 (Bottom-Left): Jumps per Feature Histogram
- **X-axis**: Number of jumps a feature experienced
- **Y-axis**: Number of features with that jump count
- **Red line**: Mean jumps per feature
- **Interpretation**: Shows distribution of hopping activity across features. Long tail = few very active hoppers.

#### Panel 4 (Bottom-Right): Jump Magnitude Distribution
- **X-axis**: Jump magnitude |Δx|
- **Y-axis**: Count
- **Red line**: Detection threshold (z × σ)
- **Interpretation**: Shows how extreme the detected jumps are. Only jumps above threshold are detected.

---

### 3. Volatility Evolution
**Filename**: `<experiment>_volatility_evolution.png`

#### Description
A 2-panel figure showing how hopping volatility changes during training.

#### Panel 1 (Left): Volatility Over Time
- **X-axis**: Checkpoint index (window center)
- **Y-axis**: Rolling volatility (standard deviation)
- **Blue line**: Mean volatility across all features
- **Shaded areas**:
  - Light blue: 10th-90th percentile range
  - Dark blue: 25th-75th percentile (IQR)
- **Interpretation**:
  - Decreasing trend → Features settling down (converging)
  - Increasing trend → Features becoming chaotic (diverging)
  - Flat trend → Stable behavior

#### Panel 2 (Right): Early vs Late Volatility Scatter
- **X-axis**: Mean volatility in early training (first third)
- **Y-axis**: Mean volatility in late training (last third)
- **Red dashed line**: y=x (no change)
- **Points below line**: Features that settled down
- **Points above line**: Features that became more volatile
- **Annotation**: Percentage breakdown of converging/diverging/stable features

---

## Aggregate Plots

These plots summarize results across all experiments.

### 4. Feature Classification Summary
**Filename**: `feature_classification_summary.png`

#### Description
A 4-panel figure showing feature classifications stratified by sparsity.

#### Panel 1 (Top-Left): Volatility Class Distribution
- **X-axis**: Sparsity bucket (low, medium, high, extreme)
- **Y-axis**: Mean count per experiment
- **Bars**: Four volatility classes (stable, moderate, active, extreme)
- **Interpretation**: Shows how many features fall into each volatility category for different sparsity levels.

#### Panel 2 (Top-Right): Temporal Pattern Distribution
- **X-axis**: Sparsity bucket
- **Y-axis**: Mean count per experiment
- **Bars**: Four temporal patterns (converging, diverging, steady, episodic)
- **Interpretation**: Shows how feature dynamics evolve differently across sparsity regimes.

#### Panel 3 (Bottom-Left): Synchrony Ratio
- **X-axis**: Sparsity bucket
- **Y-axis**: Mean synchrony ratio
- **Gray line**: Ratio = 1 (independent hopping)
- **Interpretation**:
  - Ratio > 1.5: Features hop together (synchronized)
  - Ratio < 0.7: Anti-synchronized (features avoid hopping together)
  - Ratio ≈ 1: Independent hopping

#### Panel 4 (Bottom-Right): Trend Direction Counts
- **X-axis**: Sparsity bucket
- **Y-axis**: Number of experiments
- **Bars**: Trend directions (increasing, stable, decreasing)
- **Interpretation**: Shows whether volatility tends to increase or decrease during training.

---

### 5. Sparsity Comparison
**Filename**: `sparsity_comparison.png`

#### Description
A 4-panel figure comparing hopping metrics across sparsity levels.

#### Panel 1 (Top-Left): Total Jumps vs Sparsity
- **X-axis**: Sparsity (0 to 1)
- **Y-axis**: Total jump events
- **Points**: Individual experiments
- **Interpretation**: Shows relationship between sparsity and jump frequency.

#### Panel 2 (Top-Right): Global σ vs Sparsity
- **X-axis**: Sparsity
- **Y-axis**: Global robust standard deviation
- **Points**: Individual experiments (orange)
- **Interpretation**: Shows how overall volatility varies with sparsity.

#### Panel 3 (Bottom-Left): Late/Early σ Ratio vs Sparsity
- **X-axis**: Sparsity
- **Y-axis**: Ratio of late to early volatility
- **Red line**: Ratio = 1 (no change)
- **Interpretation**:
  - Points above line: Volatility increased during training
  - Points below line: Volatility decreased during training

#### Panel 4 (Bottom-Right): Jumps vs Volatility
- **X-axis**: Global σ
- **Y-axis**: Total jumps
- **Color**: Sparsity value (blue=low, red=high)
- **Interpretation**: Shows correlation between volatility and jump frequency, colored by sparsity.

---

### 6. Summary Figure
**Filename**: `dynamic_hopping_summary.png`

#### Description
A 9-panel comprehensive summary combining all key metrics.

#### Row 1: Core Metrics by Sparsity
1. **Mean Jumps/Feature**: How frequently features hop in each sparsity regime
2. **Mean σ_global**: Overall volatility level by sparsity
3. **Late/Early σ Ratio**: Whether hopping settles down or increases

#### Row 2: Feature Behavior Patterns
4. **Synchrony Ratio**: Do features hop together?
5. **Volatility Classes** (stacked): Distribution of stable/moderate/active/extreme
6. **Temporal Patterns** (stacked): Distribution of converging/steady/episodic/diverging

#### Row 3: Scatter Analysis
7-9. **Combined scatter plot**: All experiments showing normalized jumps and σ vs sparsity

---

## Color Conventions

### Sparsity Buckets
| Bucket | Color | RGB |
|--------|-------|-----|
| low | Blue | #3498db |
| medium | Green | #2ecc71 |
| high | Orange | #e67e22 |
| extreme | Red | #e74c3c |

### Volatility Classes
| Class | Color | RGB |
|-------|-------|-----|
| stable | Green | #27ae60 |
| moderate | Yellow | #f1c40f |
| active | Orange | #e67e22 |
| extreme | Red | #e74c3c |

### Temporal Patterns
| Pattern | Color | RGB |
|---------|-------|-----|
| converging | Blue | #3498db |
| steady | Gray | #95a5a6 |
| episodic | Purple | #9b59b6 |
| diverging | Red | #e74c3c |

---

## Resolution and Format

All plots are saved as:
- **Format**: PNG
- **DPI**: 150 (suitable for reports and presentations)
- **Background**: White

Individual experiment plots can be created at higher resolution by modifying the `dpi` parameter in `04_visualizations.py`.
