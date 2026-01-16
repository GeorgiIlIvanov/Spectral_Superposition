"""
Training loop for binned sparsity experiment.
Features are assigned to 9 discrete sparsity bins: S ∈ {0.1, 0.2, ..., 0.9}
Each bin has approximately 114 features (1024 / 9 ≈ 114).
Optimized for 8x L4 GPUs with torch.compile.
"""
import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict, List

# === EXPERIMENT CONSTANTS ===
N_FEATURES = 1024
M_HIDDEN = 256  # Fixed m/n = 0.25

# Sparsity bins: 9 discrete values
SPARSITY_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_BINS = len(SPARSITY_VALUES)

# === TRAINING CONFIG ===
TOTAL_STEPS = 25_000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
SEEDS = list(range(512))  # 512 seeds

# === CHECKPOINT SCHEDULE (59 total) ===
# 0-200: every 10 steps (21 checkpoints)
# 200-1200: every 100 steps (10 checkpoints)
# 1200-6200: every 500 steps (10 checkpoints)
# 6200-25000: every 1000 steps (18 checkpoints)
CHECKPOINT_STEPS = (
    list(range(0, 201, 10)) +           # 0, 10, 20, ..., 200 (21 checkpoints)
    list(range(300, 1201, 100)) +       # 300, 400, ..., 1200 (10 checkpoints)
    list(range(1700, 6201, 500)) +      # 1700, 2200, ..., 6200 (10 checkpoints)
    list(range(7200, 25001, 1000))      # 7200, 8200, ..., 25000 (18 checkpoints)
)


def create_binned_sparsity(n_features: int = 1024) -> np.ndarray:
    """
    Create sparsity array with discrete bins.

    Returns array where features are assigned to bins:
    - Features 0-113: S=0.1
    - Features 114-227: S=0.2
    - ...
    - Features 912-1023: S=0.9
    """
    sparsity = np.zeros(n_features, dtype=np.float32)
    features_per_bin = n_features // N_BINS  # 113 for 1024/9
    remainder = n_features % N_BINS  # 7 extra features to distribute

    start_idx = 0
    for i, s_val in enumerate(SPARSITY_VALUES):
        # Distribute remainder across first few bins
        bin_size = features_per_bin + (1 if i < remainder else 0)
        end_idx = start_idx + bin_size
        sparsity[start_idx:end_idx] = s_val
        start_idx = end_idx

    return sparsity


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


def generate_binned_sparse_batch(
    batch_size: int,
    n_features: int,
    sparsity_per_feature: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Generate batch with binned sparsity per feature.
    Each feature has a discrete sparsity value from {0.1, 0.2, ..., 0.9}.
    """
    # Random values for the batch
    values = torch.rand(batch_size, n_features, device=device)

    # Per-feature mask: each feature has its own sparsity threshold
    random_mask = torch.rand(batch_size, n_features, device=device)
    mask = (random_mask >= sparsity_per_feature).to(torch.float32)

    return values * mask


def train_model(
    n_features: int,
    m_hidden: int,
    seed: int,
    total_steps: int,
    checkpoint_steps: List[int],
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    output_path: Optional[Path] = None,
    use_compile: bool = True,
    require_cuda: bool = False
) -> Optional[Dict]:
    """
    Train a superposition model with binned sparsity and save checkpoints.

    Sparsity is binned: features assigned to S ∈ {0.1, 0.2, ..., 0.9}.

    Returns checkpoint data dict if output_path is None, else saves to HDF5.
    """
    if device is None:
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPUs are available")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = SuperpositionModel(n_features, m_hidden).to(device)

    # Pre-compute binned sparsity per feature
    sparsity_np = create_binned_sparsity(n_features)
    sparsity_per_feature = torch.from_numpy(sparsity_np).to(device)

    # torch.compile for kernel fusion
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
                x_eval = generate_binned_sparse_batch(
                    batch_size * 2, n_features, sparsity_per_feature, device
                )
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
        x = generate_binned_sparse_batch(
            batch_size, n_features, sparsity_per_feature, device
        )
        loss = criterion(forward_fn(x), x)
        optimizer.zero_grad(set_to_none=True)
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
            f.attrs['seed'] = seed
            f.attrs['learning_rate'] = learning_rate
            f.attrs['batch_size'] = batch_size
            f.attrs['total_steps'] = total_steps
            f.attrs['sparsity_type'] = 'binned'
            f.attrs['sparsity_bins'] = str(SPARSITY_VALUES)

            # Store the sparsity schedule for reference
            f.create_dataset('sparsity_per_feature', data=sparsity_np)

            # Datasets - LZF compression for speed
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run quick test')
    args = parser.parse_args()

    if args.test:
        print("Running test training with binned sparsity...")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # Show sparsity distribution
        sparsity = create_binned_sparsity(N_FEATURES)
        print(f"\nSparsity distribution:")
        for s_val in SPARSITY_VALUES:
            count = np.sum(sparsity == s_val)
            print(f"  S={s_val}: {count} features")

        test_steps = [0, 100, 200]
        result = train_model(
            n_features=N_FEATURES,
            m_hidden=M_HIDDEN,
            seed=0,
            total_steps=200,
            checkpoint_steps=test_steps,
            batch_size=512,
            device=device
        )

        print(f"\nCheckpoints: {len(result['steps'])}")
        print(f"Weights shape: {result['weights'].shape}")
        print(f"Final loss: {result['losses'][-1]:.6f}")
        print("Test passed!")
