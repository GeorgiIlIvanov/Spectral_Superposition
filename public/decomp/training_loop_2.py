#!/usr/bin/env python3.9
"""
Toy Models of Superposition - Training loop with fractional dimensionality tracking.
Model: h = Wx, x' = ReLU(W^T h + b), Loss = MSE

Version 2: Added live progress printing during training.
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
import sys
import time


class ReLUAutoencoder(nn.Module):
    """ReLU output model from Toy Models of Superposition paper."""

    def __init__(self, n_features: int, m_hidden: int):
        super().__init__()
        # W: (m_hidden, n_features)
        self.W = nn.Parameter(torch.randn(m_hidden, n_features) * (1.0 / np.sqrt(m_hidden)))
        self.b = nn.Parameter(torch.zeros(n_features))
        self.n_features = n_features
        self.m_hidden = m_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_features)
        h = Wx: (batch, m_hidden)
        x' = ReLU(W^T h + b): (batch, n_features)
        """
        h = torch.matmul(x, self.W.T)  # (batch, m_hidden)
        x_hat = torch.relu(torch.matmul(h, self.W) + self.b)  # (batch, n_features)
        return x_hat

    def compute_fractional_dims(self) -> np.ndarray:
        """
        Compute per-feature fractional dimensionality.
        D_i = M_ii^2 / (M^2)_ii where M = W^T W
        """
        with torch.no_grad():
            M = torch.matmul(self.W.T, self.W)  # (n, n)
            M_diag = torch.diag(M)  # M_ii
            M_sq = torch.matmul(M, M)  # M @ M
            M_sq_diag = torch.diag(M_sq)  # (M^2)_ii
            # Avoid division by zero
            D = M_diag ** 2 / (M_sq_diag + 1e-12)
            return D.cpu().numpy()

    def compute_feature_norms(self) -> np.ndarray:
        """Compute ||W_i||^2 for each feature i."""
        with torch.no_grad():
            # W is (m_hidden, n_features), want sum over hidden dim
            norms = (self.W ** 2).sum(dim=0)  # (n_features,)
            return norms.cpu().numpy()


def generate_sparse_batch(
    batch_size: int,
    n_features: int,
    sparsity: float,
    device: torch.device
) -> torch.Tensor:
    """
    Generate sparse input vectors.
    x_i = 0 with probability S, else Uniform(0, 1)
    """
    x = torch.rand(batch_size, n_features, device=device)
    mask = (torch.rand(batch_size, n_features, device=device) >= sparsity).float()
    return x * mask


def train_model(
    n_features: int,
    m_hidden: int,
    sparsity: float,
    seed: int,
    output_path: str,
    total_steps: int = 50000,
    checkpoint_every: int = 500,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str = 'cuda',
    print_progress: bool = True,
    progress_every: int = 5000,
    job_id: int = None,
    total_jobs: int = None
) -> float:
    """
    Train the ReLU autoencoder and save checkpoints to HDF5.

    Args:
        n_features: Number of input features
        m_hidden: Hidden dimension size
        sparsity: Sparsity level (probability of zero)
        seed: Random seed
        output_path: Path to save HDF5 results
        total_steps: Total training steps
        checkpoint_every: Save checkpoint every N steps
        batch_size: Training batch size
        lr: Learning rate
        device: Device to train on
        print_progress: Whether to print live progress
        progress_every: Print progress every N steps
        job_id: Current job number (for progress display)
        total_jobs: Total number of jobs (for progress display)

    Returns:
        Final loss value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if 'cuda' in device:
        torch.cuda.manual_seed(seed)

    device = torch.device(device)
    model = ReLUAutoencoder(n_features, m_hidden).to(device)
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Storage for checkpoints
    checkpoint_steps = []
    fractional_dims_list = []
    feature_norms_list = []
    losses_list = []

    start_time = time.time()

    for step in range(total_steps + 1):
        # Generate batch
        x = generate_sparse_batch(batch_size, n_features, sparsity, device)

        # Forward pass
        x_hat = model(x)
        loss = ((x - x_hat) ** 2).mean()

        # Checkpoint before training step at step 0
        if step % checkpoint_every == 0:
            checkpoint_steps.append(step)
            fractional_dims_list.append(model.compute_fractional_dims())
            feature_norms_list.append(model.compute_feature_norms())
            losses_list.append(loss.item())

        # Print live progress
        if print_progress and step > 0 and step % progress_every == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            remaining_steps = total_steps - step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

            job_str = f"[{job_id}/{total_jobs}] " if job_id is not None else ""
            print(f"    {job_str}Step {step:>6}/{total_steps} ({100*step/total_steps:5.1f}%) | "
                  f"Loss: {loss.item():.6f} | "
                  f"Speed: {steps_per_sec:.1f} steps/s | "
                  f"ETA: {eta_seconds/60:.1f}min", flush=True)

        # Backward pass (skip at final step since we just checkpoint)
        if step < total_steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Save to HDF5
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Config as attributes
        f.attrs['n_features'] = n_features
        f.attrs['m_hidden'] = m_hidden
        f.attrs['sparsity'] = sparsity
        f.attrs['seed'] = seed
        f.attrs['total_steps'] = total_steps
        f.attrs['checkpoint_every'] = checkpoint_every
        f.attrs['batch_size'] = batch_size
        f.attrs['lr'] = lr

        # Checkpoint data
        f.create_dataset('checkpoint_steps', data=np.array(checkpoint_steps, dtype=np.int32))
        f.create_dataset('fractional_dims', data=np.stack(fractional_dims_list, axis=0).astype(np.float32))
        f.create_dataset('feature_norms', data=np.stack(feature_norms_list, axis=0).astype(np.float32))
        f.create_dataset('losses', data=np.array(losses_list, dtype=np.float32))

    return losses_list[-1]


if __name__ == '__main__':
    # Quick test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=1024)
    parser.add_argument('--m_hidden', type=int, default=256)
    parser.add_argument('--sparsity', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str, default='test_output.h5')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--total_steps', type=int, default=50000)
    args = parser.parse_args()

    final_loss = train_model(
        n_features=args.n_features,
        m_hidden=args.m_hidden,
        sparsity=args.sparsity,
        seed=args.seed,
        output_path=args.output,
        total_steps=args.total_steps,
        device=args.device,
        print_progress=True
    )
    print(f"Training complete. Final loss: {final_loss:.6f}")
