#!/usr/bin/env python3.9
"""
Sweep script for Toy Models of Superposition experiments - Version 2.

IMPROVEMENTS OVER sweep.py:
1. Scans existing results and only runs missing/incomplete experiments
2. Live progress printing with job ID, parameters, ETA
3. Designed for long-running sessions (tmux/screen compatible)

USAGE:
=======

Option 1: Using tmux (RECOMMENDED)
----------------------------------
# Start a new tmux session
tmux new -s sweep

# Run the sweep (single GPU example)
python sweep_2.py --gpu_id 0 --n_gpus 1

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t sweep

Option 2: Using screen
----------------------
screen -S sweep
python sweep_2.py --gpu_id 0 --n_gpus 1
# Detach: Ctrl+A, then D
# Reattach: screen -r sweep

Option 3: Using nohup
---------------------
nohup python -u sweep_2.py --gpu_id 0 --n_gpus 1 > sweep.log 2>&1 &
tail -f sweep.log  # Monitor progress

Multi-GPU (8 GPUs):
-------------------
# In separate tmux windows/panes:
python sweep_2.py --gpu_id 0 --n_gpus 8
python sweep_2.py --gpu_id 1 --n_gpus 8
# ... etc for gpu_id 2-7
"""

import argparse
import numpy as np
from pathlib import Path
from itertools import product
import os
import sys
import time
from datetime import datetime, timedelta

from training_loop_2 import train_model


def generate_experiment_grid():
    """
    Generate the full experiment grid.

    Returns:
        List of experiment config dicts
    """
    n_features = 1024

    # m: 20 log-spaced values from 16 to 1024
    m_values = np.logspace(np.log10(16), np.log10(1024), 20).astype(int)
    m_values = sorted(set(m_values))

    # Sparsity S: 30 log-spaced values where 1/(1-S) goes from 1 to 100
    k_values = np.logspace(0, 2, 30)
    sparsity_values = 1.0 - 1.0 / k_values

    seeds = [0, 1, 2]

    experiments = []
    for m, sparsity, seed in product(m_values, sparsity_values, seeds):
        experiments.append({
            'n_features': n_features,
            'm_hidden': int(m),
            'sparsity': float(sparsity),
            'seed': int(seed)
        })

    return experiments


def get_output_path(exp, output_dir):
    """Generate unique output path for an experiment."""
    filename = f"n{exp['n_features']}_m{exp['m_hidden']}_s{exp['sparsity']:.6f}_seed{exp['seed']}.h5"
    return Path(output_dir) / filename


def scan_existing_results(output_dir):
    """
    Scan the output directory for existing valid results.

    Returns:
        Set of (m_hidden, sparsity_rounded, seed) tuples that are complete
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return set()

    completed = set()
    h5_files = list(output_path.glob('*.h5'))

    for f in h5_files:
        try:
            # Parse filename: n1024_m16_s0.000000_seed0.h5
            name = f.stem
            parts = name.split('_')
            m = int(parts[1][1:])
            s = float(parts[2][1:])
            seed = int(parts[3][4:])

            # Verify file is valid (can be opened and has required datasets)
            import h5py
            with h5py.File(f, 'r') as hf:
                if 'fractional_dims' in hf and 'losses' in hf:
                    # Check it's complete (has all checkpoints)
                    if hf['losses'].shape[0] > 0:
                        completed.add((m, round(s, 6), seed))
        except Exception:
            # Skip corrupted files
            continue

    return completed


def partition_experiments(experiments, n_gpus, gpu_id):
    """Partition experiments round-robin by GPU ID."""
    return [exp for i, exp in enumerate(experiments) if i % n_gpus == gpu_id]


def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def run_sweep(gpu_id, n_gpus=1, output_dir='results', total_steps=50000):
    """
    Run remaining experiments assigned to this GPU.

    Args:
        gpu_id: GPU index
        n_gpus: Total number of GPUs
        output_dir: Directory to save results
        total_steps: Training steps per experiment
    """
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print("=" * 70)
    print(f"SWEEP v2 - GPU {gpu_id}/{n_gpus}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    sys.stdout.flush()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Scan existing results
    print("\nScanning existing results...")
    sys.stdout.flush()
    completed = scan_existing_results(output_dir)
    print(f"Found {len(completed)} completed experiments")
    sys.stdout.flush()

    # Generate full grid and partition for this GPU
    all_experiments = generate_experiment_grid()
    my_experiments = partition_experiments(all_experiments, n_gpus, gpu_id)

    # Filter out already completed experiments
    pending_experiments = []
    for exp in my_experiments:
        key = (exp['m_hidden'], round(exp['sparsity'], 6), exp['seed'])
        if key not in completed:
            pending_experiments.append(exp)

    print(f"\nExperiment status for GPU {gpu_id}:")
    print(f"  Total assigned: {len(my_experiments)}")
    print(f"  Already done: {len(my_experiments) - len(pending_experiments)}")
    print(f"  Remaining: {len(pending_experiments)}")
    print("=" * 70)
    sys.stdout.flush()

    if len(pending_experiments) == 0:
        print("\nAll experiments complete! Nothing to do.")
        return

    # Estimate total time
    # Rough estimate: ~2-3 minutes per 50k steps on typical GPU
    est_per_job_minutes = 2.5
    est_total_minutes = len(pending_experiments) * est_per_job_minutes
    print(f"\nEstimated time remaining: ~{format_duration(est_total_minutes * 60)}")
    print(f"Expected completion: ~{(datetime.now() + timedelta(minutes=est_total_minutes)).strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print()
    sys.stdout.flush()

    # Run experiments
    sweep_start = time.time()
    completed_count = 0
    failed_count = 0

    for i, exp in enumerate(pending_experiments):
        job_num = i + 1
        total_jobs = len(pending_experiments)

        out_file = get_output_path(exp, output_dir)

        # Double-check it doesn't exist (in case of race condition with other GPUs)
        if out_file.exists():
            print(f"[{job_num}/{total_jobs}] SKIP (already exists): m={exp['m_hidden']}, S={exp['sparsity']:.4f}, seed={exp['seed']}")
            sys.stdout.flush()
            continue

        # Print job header
        elapsed = time.time() - sweep_start
        avg_time = elapsed / max(completed_count, 1)
        remaining_jobs = total_jobs - job_num + 1
        eta_seconds = avg_time * remaining_jobs if completed_count > 0 else remaining_jobs * 150

        print(f"\n[{job_num}/{total_jobs}] START: m={exp['m_hidden']:>4}, S={exp['sparsity']:.6f}, seed={exp['seed']}")
        print(f"    Output: {out_file.name}")
        if completed_count > 0:
            print(f"    Avg time/job: {format_duration(avg_time)} | ETA: {format_duration(eta_seconds)}")
        sys.stdout.flush()

        job_start = time.time()

        try:
            final_loss = train_model(
                n_features=exp['n_features'],
                m_hidden=exp['m_hidden'],
                sparsity=exp['sparsity'],
                seed=exp['seed'],
                output_path=str(out_file),
                total_steps=total_steps,
                device='cuda:0',  # Always cuda:0 since CUDA_VISIBLE_DEVICES is set
                print_progress=True,
                progress_every=10000,
                job_id=job_num,
                total_jobs=total_jobs
            )

            job_time = time.time() - job_start
            completed_count += 1

            print(f"[{job_num}/{total_jobs}] DONE: Loss={final_loss:.6f} | Time: {format_duration(job_time)}")
            sys.stdout.flush()

        except Exception as e:
            failed_count += 1
            print(f"[{job_num}/{total_jobs}] ERROR: {e}")
            sys.stdout.flush()
            continue

    # Final summary
    total_time = time.time() - sweep_start
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {format_duration(total_time)}")
    print(f"Completed: {completed_count}")
    print(f"Failed: {failed_count}")
    if completed_count > 0:
        print(f"Average time per job: {format_duration(total_time / completed_count)}")
    print("=" * 70)
    sys.stdout.flush()


def print_status(output_dir='results'):
    """Print current completion status."""
    print("Scanning results directory...")

    all_experiments = generate_experiment_grid()
    completed = scan_existing_results(output_dir)

    # Count by (m, sparsity) combination
    from collections import defaultdict
    combo_seeds = defaultdict(set)
    for m, s, seed in completed:
        combo_seeds[(m, s)].add(seed)

    full_complete = sum(1 for seeds in combo_seeds.values() if len(seeds) == 3)
    partial = sum(1 for seeds in combo_seeds.values() if 0 < len(seeds) < 3)
    total_combos = len(set((e['m_hidden'], round(e['sparsity'], 6)) for e in all_experiments))

    print(f"\nStatus Report:")
    print(f"  Total experiments: {len(all_experiments)}")
    print(f"  Completed files: {len(completed)}")
    print(f"  Remaining: {len(all_experiments) - len(completed)}")
    print(f"\n  (m, sparsity) combinations:")
    print(f"    Complete (3/3 seeds): {full_complete}")
    print(f"    Partial (1-2 seeds): {partial}")
    print(f"    Missing (0 seeds): {total_combos - full_complete - partial}")
    print(f"    Total: {total_combos}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run superposition sweep experiments (v2 - resume support)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Check current status
  python sweep_2.py --status

  # Run on single GPU
  python sweep_2.py --gpu_id 0 --n_gpus 1

  # Run in tmux (recommended)
  tmux new -s sweep
  python sweep_2.py --gpu_id 0 --n_gpus 1

  # Run with nohup
  nohup python -u sweep_2.py --gpu_id 0 --n_gpus 1 > sweep.log 2>&1 &
        """
    )
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--n_gpus', type=int, default=1, help='Total number of GPUs')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--total_steps', type=int, default=50000, help='Training steps')
    parser.add_argument('--status', action='store_true', help='Print status and exit')

    args = parser.parse_args()

    if args.status:
        print_status(args.output_dir)
    else:
        run_sweep(
            gpu_id=args.gpu_id,
            n_gpus=args.n_gpus,
            output_dir=args.output_dir,
            total_steps=args.total_steps
        )
