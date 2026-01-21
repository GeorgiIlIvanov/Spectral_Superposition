#!/usr/bin/env python3
"""
Phase 1: Compute and store full SVD for all weight matrix checkpoints.

This script parallelizes SVD computation across 8 GPUs, processing files
in parallel with each GPU handling a subset of the data.

Output: One HDF5 file per input containing:
    - U: (56, m, m) left singular vectors
    - S: (56, m) singular values
    - Vt: (56, m, 1024) right singular vectors
    - eigenvalues: (56, m) = S² (eigenvalues of WW^T)
    - checkpoint_steps: (56,)
    - metadata attributes

Usage:
    python 01_compute_svds.py [--num-workers 8] [--dry-run]
"""

import os
import sys
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import argparse
import time
import json
from datetime import datetime


# Configuration
INPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/start')
OUTPUT_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics/analysis/svd_results')
LOG_FILE = OUTPUT_DIR / 'svd_computation.log'


def log_message(msg, log_file=None):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(formatted + '\n')


def get_output_filename(input_path):
    """Generate output filename from input path."""
    return f"svd_{input_path.stem}.h5"


def process_file(input_path, output_path, device):
    """Process a single HDF5 file: load weights, compute SVD, save results."""
    try:
        # Load input data
        with h5py.File(input_path, 'r') as f:
            weights = f['weights'][:]  # (56, m, 1024)
            checkpoint_steps = f['checkpoint_steps'][:]
            m_hidden = int(f.attrs['m_hidden'])
            sparsity = float(f.attrs['sparsity'])
            seed = int(f.attrs['seed'])

        n_checkpoints, m, n = weights.shape

        # Move to GPU and compute batched SVD
        W_gpu = torch.from_numpy(weights).to(device, dtype=torch.float32)

        # Compute SVD: W = U @ diag(S) @ Vt
        # full_matrices=False gives reduced SVD
        U, S, Vt = torch.linalg.svd(W_gpu, full_matrices=False)

        # Move results back to CPU
        U_np = U.cpu().numpy()      # (56, m, m)
        S_np = S.cpu().numpy()      # (56, m)
        Vt_np = Vt.cpu().numpy()    # (56, m, 1024)

        # Compute eigenvalues of WW^T (= S²)
        eigenvalues = S_np ** 2

        # Free GPU memory
        del W_gpu, U, S, Vt
        torch.cuda.empty_cache()

        # Save results
        with h5py.File(output_path, 'w') as f:
            # Store SVD components with compression
            f.create_dataset('U', data=U_np, compression='lzf')
            f.create_dataset('S', data=S_np, compression='lzf')
            f.create_dataset('Vt', data=Vt_np, compression='lzf')
            f.create_dataset('eigenvalues', data=eigenvalues, compression='lzf')
            f.create_dataset('checkpoint_steps', data=checkpoint_steps)

            # Store metadata
            f.attrs['m_hidden'] = m_hidden
            f.attrs['sparsity'] = sparsity
            f.attrs['seed'] = seed
            f.attrs['n_features'] = n
            f.attrs['n_checkpoints'] = n_checkpoints
            f.attrs['source_file'] = input_path.name
            f.attrs['computation_device'] = str(device)
            f.attrs['computation_time'] = datetime.now().isoformat()

        return True, None

    except Exception as e:
        return False, str(e)


def worker_process(gpu_id, file_queue, result_queue, progress_queue):
    """Worker process that processes files on a specific GPU."""
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    while True:
        item = file_queue.get()
        if item is None:  # Poison pill
            break

        input_path, output_path = item
        success, error = process_file(input_path, output_path, device)
        result_queue.put((input_path.name, success, error, gpu_id))
        progress_queue.put(1)


def main():
    parser = argparse.ArgumentParser(description='Compute SVD for all checkpoints')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of GPU workers (default: 8)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without executing')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already processed files')
    args = parser.parse_args()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_message(f"=" * 60, LOG_FILE)
    log_message(f"SVD Computation Started", LOG_FILE)
    log_message(f"=" * 60, LOG_FILE)

    # Get list of input files
    input_files = sorted(INPUT_DIR.glob('n1024_m*.h5'))
    log_message(f"Found {len(input_files)} input files", LOG_FILE)

    # Determine which files need processing
    files_to_process = []
    for input_path in input_files:
        output_path = OUTPUT_DIR / get_output_filename(input_path)
        if args.resume and output_path.exists():
            continue
        files_to_process.append((input_path, output_path))

    log_message(f"Files to process: {len(files_to_process)}", LOG_FILE)

    if args.dry_run:
        log_message("DRY RUN - No files will be processed", LOG_FILE)
        for input_path, output_path in files_to_process[:10]:
            log_message(f"  Would process: {input_path.name} -> {output_path.name}", LOG_FILE)
        if len(files_to_process) > 10:
            log_message(f"  ... and {len(files_to_process) - 10} more", LOG_FILE)
        return

    if not files_to_process:
        log_message("No files to process. Use --resume=False to recompute.", LOG_FILE)
        return

    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    num_workers = min(args.num_workers, num_gpus)
    log_message(f"Using {num_workers} GPUs out of {num_gpus} available", LOG_FILE)

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        log_message(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB", LOG_FILE)

    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)

    file_queue = mp.Queue()
    result_queue = mp.Queue()
    progress_queue = mp.Queue()

    # Start worker processes
    workers = []
    for gpu_id in range(num_workers):
        p = mp.Process(target=worker_process,
                       args=(gpu_id, file_queue, result_queue, progress_queue))
        p.start()
        workers.append(p)

    # Add files to queue
    for item in files_to_process:
        file_queue.put(item)

    # Add poison pills to stop workers
    for _ in range(num_workers):
        file_queue.put(None)

    # Track progress
    start_time = time.time()
    processed = 0
    successes = 0
    failures = []

    # Progress bar
    pbar = tqdm(total=len(files_to_process), desc="Computing SVDs",
                unit="file", ncols=100)

    while processed < len(files_to_process):
        # Update progress
        while not progress_queue.empty():
            progress_queue.get()
            pbar.update(1)

        # Collect results
        while not result_queue.empty():
            filename, success, error, gpu_id = result_queue.get()
            processed += 1
            if success:
                successes += 1
            else:
                failures.append((filename, error))
                log_message(f"FAILED: {filename} - {error}", LOG_FILE)

        time.sleep(0.1)

    pbar.close()

    # Wait for workers to finish
    for p in workers:
        p.join()

    # Final summary
    elapsed = time.time() - start_time
    log_message(f"\n" + "=" * 60, LOG_FILE)
    log_message(f"SVD Computation Complete", LOG_FILE)
    log_message(f"=" * 60, LOG_FILE)
    log_message(f"Total files processed: {processed}", LOG_FILE)
    log_message(f"Successful: {successes}", LOG_FILE)
    log_message(f"Failed: {len(failures)}", LOG_FILE)
    log_message(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)", LOG_FILE)
    log_message(f"Average time per file: {elapsed/max(processed,1)*1000:.1f} ms", LOG_FILE)

    if failures:
        log_message(f"\nFailed files:", LOG_FILE)
        for filename, error in failures:
            log_message(f"  {filename}: {error}", LOG_FILE)

    # Compute output size
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob('svd_*.h5'))
    log_message(f"\nTotal output size: {total_size / 1e9:.2f} GB", LOG_FILE)

    # Save summary
    summary = {
        'total_files': len(input_files),
        'processed': processed,
        'successes': successes,
        'failures': len(failures),
        'failed_files': [f[0] for f in failures],
        'elapsed_seconds': elapsed,
        'output_size_bytes': total_size,
        'num_workers': num_workers,
        'completion_time': datetime.now().isoformat()
    }

    with open(OUTPUT_DIR / 'computation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    log_message(f"Summary saved to {OUTPUT_DIR / 'computation_summary.json'}", LOG_FILE)


if __name__ == '__main__':
    main()
