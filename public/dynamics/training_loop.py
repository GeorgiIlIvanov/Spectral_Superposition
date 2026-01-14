"""
Core training loop with weight matrix checkpointing.
Optimized for 8x L4 GPUs with torch.compile and efficient batching.
"""
import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict, List

# === EXPERIMENT CONSTANTS ===
N_FEATURES = 1024
M_VALUES = np.linspace(16, 512, 32).astype(int)
S_VALUES = np.linspace(0, 0.99, 50)
SEEDS = [0, 1]

# === TRAINING CONFIG ===
TOTAL_STEPS = 25_000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3

# === CHECKPOINT SCHEDULE (56 total) ===
CHECKPOINT_STEPS = (
    list(range(0, 5000, 200)) +       # 25 checkpoints: steps 0-4800
    list(range(5000, 15000, 500)) +   # 20 checkpoints: steps 5000-14500
    list(range(15000, 25001, 1000))   # 11 checkpoints: steps 15000-25000
)


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

    def compute_metrics(self) -> tuple:
        """Compute fractional dims and feature norms efficiently."""
        with torch.no_grad():
            M = self.W.T @ self.W
            M_diag = torch.diag(M)
            M2_diag = torch.diag(M @ M)
            fractional_dims = (M_diag ** 2) / (M2_diag + 1e-10)
            feature_norms = (self.W ** 2).sum(dim=0)
            return fractional_dims.cpu().numpy(), feature_norms.cpu().numpy()


def generate_sparse_batch(batch_size: int, n_features: int, sparsity: float,
                          device: torch.device) -> torch.Tensor:
    """Generate sparse input batch with in-place multiplication."""
    mask = (torch.rand(batch_size, n_features, device=device) > sparsity).to(torch.float32)
    values = torch.rand(batch_size, n_features, device=device)
    return values.mul_(mask)


def train_model(
    n_features: int,
    m_hidden: int,
    sparsity: float,
    seed: int,
    total_steps: int,
    checkpoint_steps: List[int],
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    output_path: Optional[Path] = None,
    use_compile: bool = True
) -> Optional[Dict]:
    """
    Train a superposition model and save checkpoints.

    Returns checkpoint data dict if output_path is None, else saves to HDF5.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = SuperpositionModel(n_features, m_hidden).to(device)

    # torch.compile for kernel fusion (Ada Lovelace optimization)
    if use_compile and hasattr(torch, 'compile'):
        try:
            forward_fn = torch.compile(model.forward, mode='reduce-overhead')
        except Exception:
            forward_fn = model.forward
    else:
        forward_fn = model.forward

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Pre-allocate checkpoint storage
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
        # Save checkpoint
        if step in checkpoint_set:
            with torch.no_grad():
                # Evaluation loss on larger batch
                x_eval = generate_sparse_batch(batch_size * 2, n_features, sparsity, device)
                loss_val = criterion(model(x_eval), x_eval).item()

                # Store checkpoint data
                checkpoint_data['weights'][checkpoint_idx] = model.W.cpu().numpy()
                fd, fn = model.compute_metrics()
                checkpoint_data['fractional_dims'][checkpoint_idx] = fd
                checkpoint_data['feature_norms'][checkpoint_idx] = fn
                checkpoint_data['biases'][checkpoint_idx] = model.b.cpu().numpy()
                checkpoint_data['losses'][checkpoint_idx] = loss_val
            checkpoint_idx += 1

        if step >= total_steps:
            break

        # Training step
        x = generate_sparse_batch(batch_size, n_features, sparsity, device)
        loss = criterion(forward_fn(x), x)
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        loss.backward()
        optimizer.step()

    # Save to HDF5
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            # Metadata
            f.attrs['n_features'] = n_features
            f.attrs['m_hidden'] = m_hidden
            f.attrs['sparsity'] = sparsity
            f.attrs['seed'] = seed
            f.attrs['learning_rate'] = learning_rate
            f.attrs['batch_size'] = batch_size
            f.attrs['total_steps'] = total_steps

            # Datasets - LZF compression is faster than gzip for this workload
            f.create_dataset('checkpoint_steps', data=checkpoint_data['steps'])
            f.create_dataset('weights', data=checkpoint_data['weights'],
                           compression='lzf')
            f.create_dataset('fractional_dims', data=checkpoint_data['fractional_dims'])
            f.create_dataset('feature_norms', data=checkpoint_data['feature_norms'])
            f.create_dataset('biases', data=checkpoint_data['biases'])
            f.create_dataset('losses', data=checkpoint_data['losses'])

        return None

    return checkpoint_data


if __name__ == '__main__':
    # Quick test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run quick test')
    args = parser.parse_args()

    if args.test:
        print("Running test training...")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        test_steps = [0, 100, 200]
        result = train_model(
            n_features=1024,
            m_hidden=64,
            sparsity=0.5,
            seed=0,
            total_steps=200,
            checkpoint_steps=test_steps,
            batch_size=512,
            device=device
        )

        print(f"Checkpoints: {len(result['steps'])}")
        print(f"Weights shape: {result['weights'].shape}")
        print(f"Final loss: {result['losses'][-1]:.6f}")
        print("Test passed!")
