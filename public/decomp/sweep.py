#!/usr/bin/env python3.9
"""
Sweep script for Toy Models of Superposition experiments.
Generates experiment grid, partitions by GPU, runs experiments sequentially.
"""

import argparse
import numpy as np
from pathlib import Path
from itertools import product
from tqdm import tqdm
import os

from training_loop import train_model


def generate_experiment_grid():
    """
    Generate the full experiment grid.

    Returns:
        List of (n_features, m_hidden, sparsity, seed) tuples
    """
    n_features = 1024

    # m: 20 log-spaced values from 16 to 1024
    m_values = np.logspace(np.log10(16), np.log10(1024), 20).astype(int)
    # Ensure unique values (log spacing can create duplicates when cast to int)
    m_values = sorted(set(m_values))

    # Sparsity S: 30 log-spaced values where 1/(1-S) goes from 1 to 100
    # When 1/(1-S) = k, S = 1 - 1/k
    # k=1 -> S=0 (no sparsity), k=100 -> S=0.99 (very sparse)
    k_values = np.logspace(0, 2, 30)  # 10^0=1 to 10^2=100
    sparsity_values = 1.0 - 1.0 / k_values

    # 3 seeds per config
    seeds = [0, 1, 2]

    # Generate all combinations
    experiments = []
    for m, sparsity, seed in product(m_values, sparsity_values, seeds):
        experiments.append({
            'n_features': n_features,
            'm_hidden': int(m),
            'sparsity': float(sparsity),
            'seed': int(seed)
        })

    return experiments


def partition_experiments(experiments, n_gpus, gpu_id):
    """
    Partition experiments round-robin by GPU ID.

    Args:
        experiments: List of experiment configs
        n_gpus: Total number of GPUs
        gpu_id: This GPU's ID (0 to n_gpus-1)

    Returns:
        List of experiments for this GPU
    """
    return [exp for i, exp in enumerate(experiments) if i % n_gpus == gpu_id]


def get_output_path(exp, output_dir):
    """Generate unique output path for an experiment."""
    filename = f"n{exp['n_features']}_m{exp['m_hidden']}_s{exp['sparsity']:.6f}_seed{exp['seed']}.h5"
    return Path(output_dir) / filename


def run_sweep(gpu_id, n_gpus=8, output_dir='results', total_steps=50000):
    """
    Run all experiments assigned to this GPU.

    Args:
        gpu_id: GPU index (0-7)
        n_gpus: Total number of GPUs
        output_dir: Directory to save results
        total_steps: Training steps per experiment
    """
    # Set CUDA device
    device = f'cuda:{gpu_id}'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Generate and partition experiments
    all_experiments = generate_experiment_grid()
    my_experiments = partition_experiments(all_experiments, n_gpus, gpu_id)

    print(f"[GPU {gpu_id}] Running {len(my_experiments)} / {len(all_experiments)} experiments")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run experiments sequentially
    for i, exp in enumerate(tqdm(my_experiments, desc=f"GPU {gpu_id}", position=gpu_id)):
        out_file = get_output_path(exp, output_dir)

        # Skip if already completed
        if out_file.exists():
            continue

        try:
            train_model(
                n_features=exp['n_features'],
                m_hidden=exp['m_hidden'],
                sparsity=exp['sparsity'],
                seed=exp['seed'],
                output_path=str(out_file),
                total_steps=total_steps,
                device='cuda:0'  # Always use cuda:0 since CUDA_VISIBLE_DEVICES is set
            )
        except Exception as e:
            print(f"[GPU {gpu_id}] Error in experiment {exp}: {e}")
            continue

    print(f"[GPU {gpu_id}] Completed all experiments")


def print_grid_info():
    """Print information about the experiment grid."""
    experiments = generate_experiment_grid()

    # Extract unique values
    m_values = sorted(set(e['m_hidden'] for e in experiments))
    s_values = sorted(set(e['sparsity'] for e in experiments))

    print("Experiment Grid Information")
    print("=" * 50)
    print(f"n_features: 1024")
    print(f"m_hidden values ({len(m_values)}): {m_values}")
    print(f"Sparsity values ({len(s_values)}):")
    print(f"  min: {min(s_values):.6f} (1/(1-S) = 1)")
    print(f"  max: {max(s_values):.6f} (1/(1-S) = 100)")
    print(f"Seeds: [0, 1, 2]")
    print(f"Total experiments: {len(experiments)}")
    print(f"Experiments per GPU (8 GPUs): ~{len(experiments) // 8}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run superposition sweep experiments')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID (0-7)')
    parser.add_argument('--n_gpus', type=int, default=8, help='Total number of GPUs')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--total_steps', type=int, default=50000, help='Training steps')
    parser.add_argument('--info', action='store_true', help='Print grid info and exit')

    args = parser.parse_args()

    if args.info:
        print_grid_info()
    else:
        run_sweep(
            gpu_id=args.gpu_id,
            n_gpus=args.n_gpus,
            output_dir=args.output_dir,
            total_steps=args.total_steps
        )
