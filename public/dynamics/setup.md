# Spectral Superposition Experiment v2 - Claude Code Instructions

## Task Overview

Run a parameter sweep for the Toy Models of Superposition experiment, tracking full weight matrix dynamics across training. The experiment studies how neural networks learn to represent more features than they have dimensions (superposition), with the key observable being the emergence of quantized "ray structure" in the fractional dimensionality vs feature norm phase space.

## Compute Environment

- **GPUs**: 8× NVIDIA L4 (24GB each)
- **Parallelization**: 8 independent processes, one per GPU, round-robin partitioning of experiments
- **Cloud**: GCP (specific instance configuration to be provided separately)

## Experimental Grid

```python
import numpy as np

N_FEATURES = 1024

# Hidden dimension: 32 linearly-spaced values
M_VALUES = np.linspace(16, 512, 32).astype(int)
# [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 
#  272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]

# Sparsity: 50 linearly-spaced values
S_VALUES = np.linspace(0, 0.99, 50)
# [0.0, 0.0202, 0.0404, ..., 0.9698, 0.99]

# Seeds
SEEDS = [0, 1]

# Total: 32 × 50 × 2 = 3,200 experiments
# Per GPU: 400 experiments
```

## Training Configuration

```python
TOTAL_STEPS = 25_000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
OPTIMIZER = 'Adam'

# Adaptive checkpoint schedule (55 total)
CHECKPOINT_STEPS = (
    list(range(0, 5000, 200)) +       # 25 checkpoints: steps 0-4800
    list(range(5000, 15000, 500)) +   # 20 checkpoints: steps 5000-14500
    list(range(15000, 25001, 1000))   # 11 checkpoints: steps 15000-25000
)
# Total: 56 checkpoints
```

## Model Architecture

```
h = W @ x           # Projection: R^n → R^m  
x' = ReLU(W.T @ h + b)  # Reconstruction: R^m → R^n
Loss = MSE(x, x')
```

Input distribution: `x_i = 0` with probability `S`, else `x_i ~ Uniform(0,1)`

## Output Specification

### Directory Structure
```
results_v2/
├── n1024_m16_s0.000000_seed0.h5
├── n1024_m16_s0.000000_seed1.h5
├── n1024_m16_s0.020200_seed0.h5
├── ...
└── n1024_m512_s0.990000_seed1.h5
```

### HDF5 File Schema

Each file contains one complete experiment run:

```
Attributes:
    n_features: int (1024)
    m_hidden: int (16-512)
    sparsity: float (0.0-0.99)
    seed: int (0 or 1)
    learning_rate: float (0.001)
    batch_size: int (1024)
    total_steps: int (25000)

Datasets:
    checkpoint_steps: shape (56,), dtype int32
        Training steps at which checkpoints were saved
    
    weights: shape (56, m_hidden, 1024), dtype float32
        Full W matrix at each checkpoint
        IMPORTANT: This is the primary data - enables post-hoc spectral analysis
        Use gzip compression (level 4) to reduce storage
    
    fractional_dims: shape (56, 1024), dtype float32
        D_i = M_ii² / (M²)_ii where M = W.T @ W
    
    feature_norms: shape (56, 1024), dtype float32  
        ||W_i||² = sum over rows of W[:, i]²
    
    biases: shape (56, 1024), dtype float32
        Bias vector b at each checkpoint
    
    losses: shape (56,), dtype float32
        Training loss at each checkpoint
```

## Implementation

### File 1: `training_loop.py`

```python
"""
Core training loop with weight matrix checkpointing.
"""
import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path


class SuperpositionModel(nn.Module):
    """ReLU output model: h = Wx, x' = ReLU(W^T h + b)"""
    
    def __init__(self, n_features: int, m_hidden: int):
        super().__init__()
        self.n_features = n_features
        self.m_hidden = m_hidden
        
        self.W = nn.Parameter(torch.empty(m_hidden, n_features))
        nn.init.xavier_normal_(self.W)
        self.b = nn.Parameter(torch.full((n_features,), -0.1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.W.T
        return torch.relu(h @ self.W + self.b)
    
    def compute_fractional_dims(self) -> np.ndarray:
        with torch.no_grad():
            M = self.W.T @ self.W
            M_diag = torch.diag(M)
            M2_diag = torch.diag(M @ M)
            D = (M_diag ** 2) / (M2_diag + 1e-10)
            return D.cpu().numpy()
    
    def compute_feature_norms(self) -> np.ndarray:
        with torch.no_grad():
            return (self.W ** 2).sum(dim=0).cpu().numpy()


def generate_sparse_batch(batch_size: int, n_features: int, sparsity: float, 
                          device: torch.device) -> torch.Tensor:
    mask = torch.rand(batch_size, n_features, device=device) > sparsity
    values = torch.rand(batch_size, n_features, device=device)
    return mask.float() * values


def train_model(
    n_features: int,
    m_hidden: int,
    sparsity: float,
    seed: int,
    total_steps: int,
    checkpoint_steps: list,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    device: torch.device = None,
    output_path: Path = None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = SuperpositionModel(n_features, m_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    n_checkpoints = len(checkpoint_steps)
    checkpoint_data = {
        'steps': np.array(checkpoint_steps, dtype=np.int32),
        'weights': np.zeros((n_checkpoints, m_hidden, n_features), dtype=np.float32),
        'fractional_dims': np.zeros((n_checkpoints, n_features), dtype=np.float32),
        'feature_norms': np.zeros((n_checkpoints, n_features), dtype=np.float32),
        'biases': np.zeros((n_checkpoints, n_features), dtype=np.float32),
        'losses': np.zeros(n_checkpoints, dtype=np.float32),
    }
    
    checkpoint_idx = 0
    checkpoint_set = set(checkpoint_steps)
    
    for step in range(total_steps + 1):
        if step in checkpoint_set:
            with torch.no_grad():
                x_eval = generate_sparse_batch(batch_size * 4, n_features, sparsity, device)
                loss_val = criterion(model(x_eval), x_eval).item()
                
                checkpoint_data['weights'][checkpoint_idx] = model.W.cpu().numpy()
                checkpoint_data['fractional_dims'][checkpoint_idx] = model.compute_fractional_dims()
                checkpoint_data['feature_norms'][checkpoint_idx] = model.compute_feature_norms()
                checkpoint_data['biases'][checkpoint_idx] = model.b.cpu().numpy()
                checkpoint_data['losses'][checkpoint_idx] = loss_val
            checkpoint_idx += 1
        
        if step >= total_steps:
            break
        
        x = generate_sparse_batch(batch_size, n_features, sparsity, device)
        loss = criterion(model(x), x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            f.attrs['n_features'] = n_features
            f.attrs['m_hidden'] = m_hidden
            f.attrs['sparsity'] = sparsity
            f.attrs['seed'] = seed
            f.attrs['learning_rate'] = learning_rate
            f.attrs['batch_size'] = batch_size
            f.attrs['total_steps'] = total_steps
            
            f.create_dataset('checkpoint_steps', data=checkpoint_data['steps'])
            f.create_dataset('weights', data=checkpoint_data['weights'],
                           compression='gzip', compression_opts=4)
            f.create_dataset('fractional_dims', data=checkpoint_data['fractional_dims'])
            f.create_dataset('feature_norms', data=checkpoint_data['feature_norms'])
            f.create_dataset('biases', data=checkpoint_data['biases'])
            f.create_dataset('losses', data=checkpoint_data['losses'])
        
        return output_path
    
    return checkpoint_data
```

### File 2: `sweep.py`

```python
"""
Grid sweep with GPU partitioning.
"""
import numpy as np
import argparse
from pathlib import Path
from itertools import product
import torch
from tqdm import tqdm

from training_loop import train_model

# === EXPERIMENT GRID ===
N_FEATURES = 1024
M_VALUES = np.linspace(16, 512, 32).astype(int)
S_VALUES = np.linspace(0, 0.99, 50)
SEEDS = [0, 1]

# === TRAINING CONFIG ===
TOTAL_STEPS = 25_000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3

# === CHECKPOINT SCHEDULE ===
CHECKPOINT_STEPS = (
    list(range(0, 5000, 200)) +
    list(range(5000, 15000, 500)) +
    list(range(15000, 25001, 1000))
)


def generate_experiment_grid():
    experiments = []
    for m, s, seed in product(M_VALUES, S_VALUES, SEEDS):
        experiments.append({
            'n_features': N_FEATURES,
            'm_hidden': int(m),
            'sparsity': float(s),
            'seed': seed,
        })
    return experiments


def run_experiments(gpu_id: int, total_gpus: int, results_dir: Path):
    device = torch.device(f'cuda:{gpu_id}')
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_experiments = generate_experiment_grid()
    my_experiments = [exp for i, exp in enumerate(all_experiments) if i % total_gpus == gpu_id]
    
    print(f"GPU {gpu_id}: Running {len(my_experiments)} / {len(all_experiments)} experiments")
    
    for exp in tqdm(my_experiments, desc=f"GPU {gpu_id}"):
        filename = f"n{exp['n_features']}_m{exp['m_hidden']}_s{exp['sparsity']:.6f}_seed{exp['seed']}.h5"
        output_path = results_dir / filename
        
        if output_path.exists():
            continue
        
        try:
            train_model(
                n_features=exp['n_features'],
                m_hidden=exp['m_hidden'],
                sparsity=exp['sparsity'],
                seed=exp['seed'],
                total_steps=TOTAL_STEPS,
                checkpoint_steps=CHECKPOINT_STEPS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                device=device,
                output_path=output_path
            )
        except Exception as e:
            print(f"GPU {gpu_id}: Error on {filename}: {e}")
            continue
    
    print(f"GPU {gpu_id}: Completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--total-gpus', type=int, default=8)
    parser.add_argument('--results-dir', type=str, default='results_v2')
    args = parser.parse_args()
    
    run_experiments(args.gpu, args.total_gpus, Path(args.results_dir))
```

### File 3: `run_sweep.sh`

```bash
#!/bin/bash
set -e

RESULTS_DIR="${1:-results_v2}"
NUM_GPUS="${2:-8}"

mkdir -p "$RESULTS_DIR" logs

echo "Starting sweep: $NUM_GPUS GPUs, output to $RESULTS_DIR"
echo "Total experiments: 3200"
echo "Experiments per GPU: 400"

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching GPU $gpu_id..."
    CUDA_VISIBLE_DEVICES=$gpu_id python sweep.py \
        --gpu 0 \
        --total-gpus $NUM_GPUS \
        --results-dir $RESULTS_DIR \
        > "logs/gpu_${gpu_id}.log" 2>&1 &
done

echo "All processes launched."
echo "Monitor: tail -f logs/gpu_*.log"
echo "Count:   watch -n 30 'ls $RESULTS_DIR/*.h5 2>/dev/null | wc -l'"

wait
echo "Sweep complete!"
```

### File 4: `verify_results.py`

```python
"""
Verify sweep completion and data integrity.
"""
import h5py
import numpy as np
from pathlib import Path


def verify_sweep(results_dir: str = 'results_v2'):
    results_dir = Path(results_dir)
    h5_files = list(results_dir.glob('*.h5'))
    
    expected = 32 * 50 * 2  # 3200
    print(f"Found: {len(h5_files)} / {expected} ({100*len(h5_files)/expected:.1f}%)")
    
    if not h5_files:
        print("No files found!")
        return
    
    # Sample verification
    corrupted = []
    for f in h5_files[:50]:
        try:
            with h5py.File(f, 'r') as hf:
                assert hf['weights'].shape[0] == 56, f"Wrong checkpoint count: {hf['weights'].shape}"
                assert hf['weights'].shape[2] == 1024, f"Wrong n_features: {hf['weights'].shape}"
        except Exception as e:
            corrupted.append((f.name, str(e)))
    
    if corrupted:
        print(f"\nCorrupted: {len(corrupted)}")
        for name, err in corrupted[:5]:
            print(f"  {name}: {err}")
    else:
        print("Sample verification passed (50 files)")
    
    # Storage
    total_bytes = sum(f.stat().st_size for f in h5_files)
    print(f"\nStorage: {total_bytes/1e9:.2f} GB")
    print(f"Avg/file: {total_bytes/len(h5_files)/1e6:.2f} MB")


if __name__ == '__main__':
    verify_sweep()
```

## Storage Estimate

| m_hidden | W matrix size | Per file (56 ckpts) | 
|----------|---------------|---------------------|
| 16 | 64 KB | ~4 MB |
| 256 | 1 MB | ~60 MB |
| 512 | 2 MB | ~120 MB |

**Total estimate**: ~75-80 GB for 3,200 experiments (with gzip compression)

## Expected Resource Usage per GPU (L4-24GB)

- **Memory**: < 1 GB (model + batch + overhead)
- **Utilization**: ~25-30% (Python/transfer overhead dominates)
- **Time per experiment**: ~25-35 seconds (25k steps, based on v1 empirical data)
- **Total time**: 400 experiments × 30s ≈ **3-4 hours wall clock**

Note: v1 ran 225 experiments/GPU × 50k steps in ~3 hours on A100. 
v2 runs 400 experiments/GPU × 25k steps, so similar total compute.

## Execution Checklist

1. [ ] Provision 8× L4 GPUs on GCP
2. [ ] Install dependencies: `pip install torch numpy h5py tqdm`
3. [ ] Create files: `training_loop.py`, `sweep.py`, `run_sweep.sh`, `verify_results.py`
4. [ ] Make executable: `chmod +x run_sweep.sh`
5. [ ] Launch: `./run_sweep.sh results_v2 8`
6. [ ] Monitor: `tail -f logs/gpu_*.log`
7. [ ] Verify: `python verify_results.py`

## Post-Completion

After sweep completes, the weight matrices enable:
- Eigenvalue decomposition of M = W^T W
- Resolvent analysis G(z) = (M - zI)^{-1}
- Temporal dynamics of spectral structure
- Association scheme detection

These should be done in a separate analysis phase, not during training.

---

**GCP Configuration**: [To be filled by Gemini]
