# Capacity Localization Analysis - Run Log

## Overview

**Analysis Date:** 2026-01-28
**Data Directory:** `/home/georgi/Spectral_Superposition/public/dynamics/start/`
**Output Directory:** `/home/georgi/Spectral_Superposition/public/dynamics/start/results/`

## System Configuration

- **Platform:** Google Cloud VM with 8x NVIDIA L4 GPUs
- **GPU Memory:** 23034 MiB per GPU
- **PyTorch Version:** 2.9.1+cu128
- **Python Environment:** `/home/georgi/Spectral_Superposition/public/dynamics/venv/`

## Experiment Dataset

- **Total Files Processed:** 3,200 experiments
- **File Pattern:** `n1024_m*_s*_seed*.h5`
- **Feature Dimensions:** n = 1024 features per experiment
- **Hidden Dimensions:** m ranging from 16 to 512
- **Sparsity Range:** s from 0.0 to 0.99
- **Seeds per Configuration:** 2 (seed 0 and seed 1)

## Analysis Pipeline

### Step 0: Rank Defect vs Delocalization Defect
For each experiment:
1. Loaded final checkpoint weights W (shape m x n) and fractional dimensions D (length n)
2. Computed frame operator F = W @ W^T
3. Performed eigendecomposition on GPU using `torch.linalg.eigh()`
4. Estimated effective rank r = count of eigenvalues > tol (tol = 1e-8 * max eigenvalue)
5. Computed saturation metrics:
   - `rho_m = sum(D) / m` (saturation w.r.t. hidden dimension)
   - `rho_r = sum(D) / r` (saturation w.r.t. effective rank)
   - `rank_ratio = r / m`

### Step 1: Per-Feature Localization Diagnostics
For each feature vector w_i = W[:, i]:
1. **Leverage:** `ell_i = sum_k (u_k^T w_i)^2 / lambda_k`
2. **Rayleigh Quotient:** `kappa_i = (w_i^T F w_i) / ||w_i||^2`
3. **Eigenvector Residual:** `resid_i = sqrt(Var_{mu_i}(lambda))`
4. **Relative Slack:** `sigma_i = 1 - D_i / ell_i`

### Step 2: Leverage-Weighted Aggregates
Computed per run:
- `slack_run = 1 - rho_r`
- `mean_resid = leverage-weighted mean of resid_i`
- `mean_sigma = leverage-weighted mean of sigma_i`
- Tail mass curves for tau in {0.01, 0.05, 0.10}

### Step 3: Diracness Proxy Validation
- Grouped eigenvalues by near-degeneracy (rtol=1e-6)
- Computed `q_i = max_group p_{i,group}` (Diracness proxy)

## Scripts Created

1. **`capacity_localization_analysis.py`** - Main GPU-accelerated analysis script
   - Uses PyTorch for eigendecomposition
   - Processes files sequentially with GPU round-robin across 8 GPUs
   - Outputs: `run_metrics.csv`, `feature_metrics.csv`, `analysis_metadata.json`

2. **`generate_plots.py`** - Visualization script
   - Plot A: Rank-vs-delocalization decomposition
   - Plot B: Featurewise localization distributions
   - Plot C: Eigenvalue-reciprocal fit analysis
   - Additional: Tail mass curves, Diracness validation

## Output Files Generated

### Data Files
| File | Size | Description |
|------|------|-------------|
| `run_metrics.csv` | 804 KB | 3,200 rows with per-run aggregate metrics |
| `feature_metrics.csv` | 618 MB | 3,276,800 rows with per-feature metrics |
| `analysis_metadata.json` | 350 B | Analysis configuration and parameters |
| `analysis_summary.json` | ~1 KB | Key numerical findings |

### Plot Files
| File | Description |
|------|-------------|
| `plot_A_rank_delocalization.png/pdf` | 4-panel rank/saturation vs sparsity |
| `plot_A_supplementary_diagnostics.png` | Mean sigma and residual vs sparsity |
| `plot_B_featurewise_localization.png/pdf` | Leverage-weighted sigma distributions |
| `plot_B_residual_distribution.png` | Leverage-weighted residual distributions |
| `plot_C_eigenvalue_reciprocal.png/pdf` | D vs 1/kappa colored by sigma |
| `plot_step3_diracness_validation.png/pdf` | Diracness proxy distributions |
| `plot_tail_mass_curves.png` | Tail mass vs sparsity |

## Run Statistics

- **Start Time:** 2026-01-28 19:20:11
- **End Time:** 2026-01-28 19:27:31
- **Total Runtime:** ~7 minutes for 3,200 experiments
- **Average per Experiment:** ~130ms (including I/O and GPU eigendecomposition)

## Reproducibility

To reproduce this analysis:

```bash
cd /home/georgi/Spectral_Superposition/public/dynamics/start
source ../venv/bin/activate

# Run analysis
python capacity_localization_analysis.py \
    --data-dir . \
    --out ./results \
    --save-feature-metrics \
    --num-gpus 8

# Generate plots
python generate_plots.py \
    --results-dir ./results \
    --out ./results
```

## Version Information

- NumPy: 2.4.1
- Pandas: 3.0.0
- PyTorch: 2.9.1+cu128
- h5py: Latest
- matplotlib: Latest
