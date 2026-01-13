# Spectral Superposition Experiment v2.1 - Optimized for 8×L4 (GCP)

## Task Overview
Run a high-throughput parameter sweep for Toy Models of Superposition. This version is optimized to saturate 8× NVIDIA L4 GPUs by running multiple concurrent experiments per device to overcome Python/CPU bottlenecks.

## Compute Environment
- **Instance**: `g2-standard-96` (96 vCPUs, 8× L4 GPUs).
- **Parallelization**: Multi-process worker pool (4 workers per GPU, 32 total concurrent experiments).
- **Optimization**: `torch.compile` for fused ReLU/Linear kernels and LZF compression for faster I/O.

## Experimental Grid
- **N_FEATURES**: 1024.
- **M_VALUES**: 32 values (16 to 512).
- **S_VALUES**: 50 values (0.0 to 0.99).
- **SEEDS**: [0, 1].
- **Total Experiments**: 3,200.

## Implementation: training_loop.py

```python
import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path

class SuperpositionModel(nn.Module):
    def __init__(self, n_features: int, m_hidden: int):
        super().__init__()
        self.n_features = n_features
        self.m_hidden = m_hidden
        self.W = nn.Parameter(torch.empty(m_hidden, n_features))
        nn.init.xavier_normal_(self.W)
        self.b = nn.Parameter(torch.full((n_features,), -0.1))
        # Compile the forward pass for kernel fusion
        self.forward_optimized = torch.compile(self.forward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.W.T
        return torch.relu(h @ self.W + self.b)

def generate_sparse_batch(batch_size, n_features, sparsity, device):
    # Optimized in-place generation
    mask = (torch.rand(batch_size, n_features, device=device) > sparsity).to(torch.float32)
    values = torch.rand(batch_size, n_features, device=device)
    return values.mul_(mask)

def train_model(n_features, m_hidden, sparsity, seed, total_steps, checkpoint_steps, 
                batch_size=1024, learning_rate=1e-3, device=None, output_path=None):
    torch.manual_seed(seed)
    model = SuperpositionModel(n_features, m_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Pre-allocate data structures
    n_ckpts = len(checkpoint_steps)
    results = {k: np.zeros((n_ckpts, n_features) if 'loss' not in k else n_ckpts) for k in ['weights', 'fd', 'fn', 'biases', 'losses']}
    results['weights'] = np.zeros((n_ckpts, m_hidden, n_features))

    checkpoint_set = set(checkpoint_steps)
    ckpt_idx = 0

    for step in range(total_steps + 1):
        if step in checkpoint_set:
            with torch.no_grad():
                x_eval = generate_sparse_batch(batch_size*2, n_features, sparsity, device)
                results['losses'][ckpt_idx] = criterion(model(x_eval), x_eval).item()
                results['weights'][ckpt_idx] = model.W.cpu().numpy()
                results['biases'][ckpt_idx] = model.b.cpu().numpy()
                # Post-processing dims/norms
                M = model.W.T @ model.W
                results['fd'][ckpt_idx] = (torch.diag(M)**2 / (torch.diag(M@M)+1e-10)).cpu().numpy()
                results['fn'][ckpt_idx] = (model.W**2).sum(0).cpu().numpy()
            ckpt_idx += 1
        
        if step < total_steps:
            x = generate_sparse_batch(batch_size, n_features, sparsity, device)
            loss = criterion(model.forward_optimized(x), x)
            optimizer.zero_grad(set_to_none=True) # Faster than zero_grad()
            loss.backward()
            optimizer.step()

    if output_path:
        with h5py.File(output_path, 'w') as f:
            f.attrs.update({'m': m_hidden, 's': sparsity, 'seed': seed})
            f.create_dataset('weights', data=results['weights'], compression='lzf') # LZF is faster for L4 throughput
            f.create_dataset('fractional_dims', data=results['fd'])
            f.create_dataset('feature_norms', data=results['fn'])
            f.create_dataset('losses', data=results['losses'])

```

## Implementation: sweep.py (Multi-Worker)

import concurrent.futures
from multiprocessing import Manager
import torch
from pathlib import Path
from training_loop import train_model, CHECKPOINT_STEPS # assumed constants

def worker(gpu_id, queue, results_dir):
    device = torch.device(f'cuda:{gpu_id}')
    while not queue.empty():
        try: exp = queue.get_nowait()
        except: break
        
        path = Path(results_dir) / f"n1024_m{exp['m']}_s{exp['s']:.6f}_seed{exp['seed']}.h5"
        if not path.exists():
            train_model(1024, exp['m'], exp['s'], exp['seed'], 25000, 
                        CHECKPOINT_STEPS, device=device, output_path=path)

if __name__ == '__main__':
    # ... grid generation ...
    manager = Manager()
    queue = manager.Queue()
    for e in experiments: queue.put(e)
    
    # 4 workers per GPU to maximize L4 utilization
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        for i in range(32):
            executor.submit(worker, i % 8, queue, 'results_v2')
```

## Execution Checklist

1. Provision `g2-standard-96` on GCP
2. Use `compression='lzf' in HDF5 to prevent CPU bottlenecks during I/O
3. Ensure torch.compile is active to leverage Ada Lovelace acrhitecture
