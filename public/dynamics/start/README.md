---
license: mit
task_categories:
  - feature-extraction
tags:
  - neural-networks
  - superposition
  - mechanistic-interpretability
  - toy-models
  - training-dynamics
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: "*.h5"
---

# Spectral Superposition Training Dynamics Dataset

This dataset contains complete training trajectories for 3,200 two-layer ReLU autoencoder models trained to study **neural network superposition** — the phenomenon where networks learn to represent more features than their hidden dimension allows.

## Dataset Description

### Summary

- **Total Files:** 3,200 HDF5 files
- **Total Size:** ~183 GB
- **Format:** HDF5 with LZF compression
- **Checkpoints per File:** 56 training snapshots
- **Features Tracked:** Weight matrices, fractional dimensionality, feature norms, biases, losses

### Research Context

This dataset explores how neural networks compress information when bottlenecked. When a network has fewer hidden units than input features, it must learn **superposed representations** — encoding multiple features along shared directions. This dataset provides comprehensive training dynamics across varying:

- **Model capacity** (hidden dimension): 16 to 512 units
- **Input sparsity**: 0% to 99% sparse inputs
- **Random seeds**: 2 seeds per configuration

## Dataset Structure

### File Naming Convention

```
n{N_FEATURES}_m{M_HIDDEN}_s{SPARSITY:.6f}_seed{SEED}.h5
```

**Examples:**
- `n1024_m16_s0.000000_seed0.h5` — smallest model, dense inputs
- `n1024_m512_s0.990000_seed1.h5` — largest model, 99% sparse inputs
- `n1024_m256_s0.545510_seed1.h5` — mid-range configuration

### HDF5 Schema

Each file contains:

#### Attributes (Metadata)

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_features` | int | Input/output dimension (always 1024) |
| `m_hidden` | int | Hidden layer dimension (16-512) |
| `sparsity` | float | Input sparsity level (0.0-0.99) |
| `seed` | int | Random seed for initialization (0 or 1) |
| `learning_rate` | float | Adam learning rate (0.001) |
| `batch_size` | int | Training batch size (1024) |
| `total_steps` | int | Total training steps (25000) |

#### Datasets

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `checkpoint_steps` | (56,) | int32 | Training step indices for each checkpoint |
| `weights` | (56, m_hidden, 1024) | float32 | Weight matrix W at each checkpoint (LZF compressed) |
| `fractional_dims` | (56, 1024) | float32 | Fractional dimensionality per feature |
| `feature_norms` | (56, 1024) | float32 | L2 norm of each feature column |
| `biases` | (56, 1024) | float32 | Bias vector b at each checkpoint |
| `losses` | (56,) | float32 | MSE training loss at each checkpoint |

### Checkpoint Schedule

Checkpoints are saved at varying intervals to capture both early dynamics and late-stage convergence:

| Training Phase | Steps | Interval | Count |
|----------------|-------|----------|-------|
| Early (rapid change) | 0–4,800 | 200 | 25 |
| Mid (transition) | 5,000–14,500 | 500 | 20 |
| Late (convergence) | 15,000–25,000 | 1,000 | 11 |

**Total:** 56 checkpoints per experiment

## Parameter Grid

### Hidden Dimensions (32 values)

```
[16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224,
 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432,
 448, 464, 480, 496, 512]
```

Compression ratios range from **64:1** (m=16) to **2:1** (m=512).

### Sparsity Levels (50 values)

```
np.linspace(0.0, 0.99, 50)
# [0.0, 0.0204, 0.0408, ..., 0.9696, 0.99]
```

### Seeds

Two random seeds (0 and 1) for reproducibility analysis.

**Total experiments:** 32 × 50 × 2 = **3,200 files**

## Model Architecture

```
Input x ∈ ℝ¹⁰²⁴
    ↓
h = W·x           # Encoding (hidden: m_hidden dimensions)
    ↓
x' = ReLU(Wᵀ·h + b)  # Decoding (output: 1024 dimensions)
    ↓
Loss = MSE(x, x')
```

- **Encoder:** Linear projection W ∈ ℝᵐˣ¹⁰²⁴
- **Decoder:** Transposed weights Wᵀ with ReLU activation and bias
- **Initialization:** Xavier normal (weights), -0.1 constant (bias)
- **Optimizer:** Adam (lr=0.001)

## Key Metrics

### Fractional Dimensionality

Measures how concentrated each feature's representation is along a single direction:

```
D_i = M_ii² / (M²)_ii   where M = WᵀW
```

- **D_i ≈ 1:** Feature i has a dedicated direction (no superposition)
- **D_i < 1:** Feature i is superposed with others

### Feature Norms

The L2 norm of each feature's column in W:

```
||W_i||² = Σⱼ W[j,i]²
```

Larger norms indicate stronger/more important features in the learned representation.

## Data Generation

Training inputs are generated with controlled sparsity:

```python
def generate_batch(batch_size, n_features, sparsity):
    x = torch.rand(batch_size, n_features)
    mask = torch.rand(batch_size, n_features) > sparsity
    return x * mask
```

Each element is independently sampled from Uniform(0,1) with probability (1 - sparsity), otherwise set to 0.

## Usage

### Loading a Single File

```python
import h5py
import numpy as np

with h5py.File('n1024_m256_s0.500000_seed0.h5', 'r') as f:
    # Metadata
    m_hidden = f.attrs['m_hidden']
    sparsity = f.attrs['sparsity']

    # Training data
    steps = f['checkpoint_steps'][:]      # (56,)
    weights = f['weights'][:]             # (56, 256, 1024)
    frac_dims = f['fractional_dims'][:]   # (56, 1024)
    norms = f['feature_norms'][:]         # (56, 1024)
    losses = f['losses'][:]               # (56,)
```

### Loading Multiple Files

```python
from pathlib import Path
import h5py

data_dir = Path('start/')
results = []

for h5_file in data_dir.glob('*.h5'):
    with h5py.File(h5_file, 'r') as f:
        results.append({
            'm_hidden': f.attrs['m_hidden'],
            'sparsity': f.attrs['sparsity'],
            'seed': f.attrs['seed'],
            'final_loss': f['losses'][-1],
            'final_frac_dims': f['fractional_dims'][-1, :].mean()
        })
```

### Analyzing Training Dynamics

```python
import matplotlib.pyplot as plt

with h5py.File('n1024_m128_s0.500000_seed0.h5', 'r') as f:
    steps = f['checkpoint_steps'][:]
    losses = f['losses'][:]
    frac_dims = f['fractional_dims'][:]

# Plot loss curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(steps, losses)
plt.xlabel('Training Step')
plt.ylabel('MSE Loss')
plt.title('Training Loss')

# Plot fractional dimensionality evolution
plt.subplot(1, 2, 2)
plt.imshow(frac_dims.T, aspect='auto', cmap='viridis')
plt.xlabel('Checkpoint')
plt.ylabel('Feature Index')
plt.colorbar(label='Fractional Dim')
plt.title('Feature Dimensionality Over Training')
plt.tight_layout()
plt.show()
```

## File Sizes

| m_hidden | Approximate Size |
|----------|------------------|
| 16 | 12-14 MB |
| 128 | 30-35 MB |
| 256 | 55-65 MB |
| 512 | 110-120 MB |

Variation depends on compression effectiveness (LZF).

## Dataset Statistics

- **Completion:** 100% (all 3,200 experiments)
- **Generation Time:** ~106 minutes on 8× NVIDIA L4 GPUs
- **Validation:** All files verified for data integrity

## Intended Uses

1. **Phase transition analysis** — Study emergence of superposition across capacity/sparsity
2. **Spectral analysis** — Examine eigenstructure of learned representations
3. **Training dynamics** — Track how features organize during learning
4. **Mechanistic interpretability** — Understand feature encoding in bottlenecked networks
5. **Reproducibility research** — Compare across random seeds

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{spectral_superposition_dynamics,
  title={Spectral Superposition Training Dynamics Dataset},
  author={},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/}}
}
```

## License

MIT License
