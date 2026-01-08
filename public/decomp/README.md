# Description

The purpose of this folder is to track the full spectrum of fractional dimensionalities in the Toy Models of Superposition paper instead of just the mean.Model.

ReLU output model from the paper: h = Wx, x' = ReLU(W^T h + b).
The LOSS is MSE.
Input: sparse vectors x_i = 0 with prob S, else Uniform(0,1).

Key quantity: Per-feature fractional dimensionality D_i = M_ii^2 / (M^2)\_ii
where M = W^T W.

# Experimental Setup

- n_features = 1024. m = 20; log-spaced values from 16 to 1024.
- Sparsity S: 30 log-spaced values in 1/(1-S) from 1 to 100. 3 seeds per config.
- Each run save (HDF5) the config (n, m, S, seed) at each checkpoint (every
  500 steps, 50k total):
- full D_i vector (all 512 values), feature norms ||W_i||^2 loss.
- Paralelization: 8 A100 GPUs.
  Partition of experiments is round-robin by GPU. Each GPU runs its share
  sequentally. Create 1. training_loop.py - model, training loop,
  spectrum tracking. 2 sweep.py - generates grid, partitions by
  GPU, runs experiments. 3. run_all.sh - launches 8 background
  proceses, one per GPU. se python3.9. Dependencies already
  installed: torch, numpy, h5py, tqdm, matplotlib.

Files:

- training_loop.py - Model, training loop, spectrum tracking
- sweep.py - Grid generation, GPU partitioning, experiment runner
- run_all.sh - Launches 8 background processes

Grid configuration:

- 1800 total experiments (20 × 30 × 3)
- ~225 experiments per GPU
- m_hidden: 20 values from 16 to 1024 (log-spaced)
- Sparsity: 30 values with 1/(1-S) from 1 to 100

HDF5 output per experiment:

- Attributes: n_features, m_hidden, sparsity, seed, etc.
- Datasets:
  - checkpoint_steps: (101,) int32 - steps 0, 500, ..., 50000
  - fractional_dims: (101, 1024) float32 - full D_i spectrum at each checkpoint
  - feature_norms: (101, 1024) float32 - ||W_i||^2
  - losses: (101,) float32

Usage:

# Run all experiments (results in ./results, logs in ./logs)

./run_all.sh

# Custom output directory and steps

./run_all.sh my_results 50000

# Monitor progress

tail -f logs/gpu\_\*.log
