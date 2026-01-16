"""
Sweep for non-uniform sparsity experiment.
Fixed m=256, n=1024, with S_i = i/n sparsity gradient.
512 seeds, parallelized across 8x L4 GPUs.
"""
import numpy as np
import argparse
import os
import time
import signal
import sys
import json
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue, Value
from queue import Empty
import torch

from training_loop import (
    train_model, N_FEATURES, M_HIDDEN, SEEDS,
    TOTAL_STEPS, CHECKPOINT_STEPS, BATCH_SIZE, LEARNING_RATE
)

# === PARALLELIZATION CONFIG ===
WORKERS_PER_GPU = 4  # L4 can handle multiple small models concurrently
DEFAULT_NUM_GPUS = 8


def write_progress_state(results_dir: Path, state: dict):
    """Write progress state to JSON file for monitoring."""
    state_file = results_dir / '.sweep_state.json'
    state['last_updated'] = datetime.now().isoformat()
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def read_progress_state(results_dir: Path) -> dict:
    """Read progress state from JSON file."""
    state_file = results_dir / '.sweep_state.json'
    try:
        if state_file.exists():
            with open(state_file, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def generate_experiment_grid():
    """Generate all experiment configurations (just seeds for this experiment)."""
    experiments = []
    for seed in SEEDS:
        experiments.append({
            'n_features': N_FEATURES,
            'm_hidden': M_HIDDEN,
            'seed': int(seed),
        })
    return experiments


def get_output_filename(exp: dict) -> str:
    """Generate consistent filename for experiment."""
    return f"n{exp['n_features']}_m{exp['m_hidden']}_nonuniform_seed{exp['seed']}.h5"


def worker_process(
    worker_id: int,
    gpu_id: int,
    task_queue: Queue,
    results_dir: Path,
    completed_counter: Value,
    total_experiments: int,
    stop_flag: Value,
    allow_cpu: bool = False
):
    """Worker process that pulls experiments from queue and trains models."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        worker_name = f"GPU{gpu_id}-W{worker_id % WORKERS_PER_GPU}"
    elif allow_cpu:
        device = torch.device('cpu')
        worker_name = f"CPU-W{worker_id}"
    else:
        print(f"[GPU{gpu_id}-W{worker_id % WORKERS_PER_GPU}] No CUDA GPUs available, exiting")
        return

    warmup_done = False

    while not stop_flag.value:
        try:
            exp = task_queue.get(timeout=1.0)
        except Empty:
            if task_queue.empty():
                break
            continue

        filename = get_output_filename(exp)
        output_path = results_dir / filename

        # Skip if already completed
        if output_path.exists():
            with completed_counter.get_lock():
                completed_counter.value += 1
            continue

        try:
            start_time = time.time()

            # Warmup torch.compile on first experiment
            if not warmup_done:
                _ = train_model(
                    n_features=N_FEATURES,
                    m_hidden=M_HIDDEN,
                    seed=0,
                    total_steps=100,
                    checkpoint_steps=[0, 100],
                    batch_size=BATCH_SIZE,
                    device=device,
                    use_compile=True
                )
                warmup_done = True

            # Run actual experiment
            train_model(
                n_features=exp['n_features'],
                m_hidden=exp['m_hidden'],
                seed=exp['seed'],
                total_steps=TOTAL_STEPS,
                checkpoint_steps=CHECKPOINT_STEPS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                device=device,
                output_path=output_path,
                use_compile=True
            )

            elapsed = time.time() - start_time
            with completed_counter.get_lock():
                completed_counter.value += 1
                done = completed_counter.value

            # Progress logging
            if done % 32 == 0 or done == total_experiments:
                print(f"[{worker_name}] {done}/{total_experiments} "
                      f"({100*done/total_experiments:.1f}%) - "
                      f"seed={exp['seed']} in {elapsed:.1f}s")

        except Exception as e:
            print(f"[{worker_name}] ERROR on {filename}: {e}", file=sys.stderr)
            continue

    print(f"[{worker_name}] Worker finished")


def progress_monitor(
    results_dir: Path,
    completed_counter: Value,
    total_experiments: int,
    start_time: float,
    stop_flag: Value,
    update_interval: int = 30
):
    """Background thread to periodically update progress state file."""
    import threading

    def update():
        while not stop_flag.value:
            try:
                completed = completed_counter.value
                elapsed = time.time() - start_time
                rate = completed / max(1, elapsed) * 3600

                state = {
                    'completed': completed,
                    'total': total_experiments,
                    'percent': round(100 * completed / total_experiments, 2),
                    'elapsed_minutes': round(elapsed / 60, 1),
                    'rate_per_hour': round(rate, 1),
                    'status': 'running',
                    'started': datetime.fromtimestamp(start_time).isoformat(),
                }

                if completed > 0 and completed < total_experiments:
                    remaining = total_experiments - completed
                    eta_seconds = remaining / (completed / elapsed)
                    state['eta_minutes'] = round(eta_seconds / 60, 1)

                write_progress_state(results_dir, state)
            except Exception:
                pass

            for _ in range(update_interval):
                if stop_flag.value:
                    break
                time.sleep(1)

    thread = threading.Thread(target=update, daemon=True)
    thread.start()
    return thread


def run_sweep(
    num_gpus: int = DEFAULT_NUM_GPUS,
    workers_per_gpu: int = WORKERS_PER_GPU,
    results_dir: str = 'results',
    resume: bool = True,
    allow_cpu: bool = False
):
    """Run the non-uniform sparsity experiment sweep."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check disk space
    disk_stat = os.statvfs(results_dir)
    free_gb = (disk_stat.f_frsize * disk_stat.f_bavail) / (1024**3)
    print(f"Disk space available: {free_gb:.1f} GB")

    # Generate experiment grid
    all_experiments = generate_experiment_grid()
    total_experiments = len(all_experiments)

    # Filter out completed experiments if resuming
    if resume:
        existing = set(f.name for f in results_dir.glob('*.h5'))
        experiments_to_run = [
            exp for exp in all_experiments
            if get_output_filename(exp) not in existing
        ]
        completed_initial = total_experiments - len(experiments_to_run)
    else:
        experiments_to_run = all_experiments
        completed_initial = 0

    print(f"=" * 60)
    print(f"Non-Uniform Sparsity Sweep (S_i = i/n)")
    print(f"=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"PID: {os.getpid()}")
    print(f"Configuration:")
    print(f"  n_features: {N_FEATURES}")
    print(f"  m_hidden: {M_HIDDEN}")
    print(f"  m/n ratio: {M_HIDDEN/N_FEATURES:.4f}")
    print(f"  Sparsity: S_i = i/{N_FEATURES} (linear gradient)")
    print(f"  Total steps: {TOTAL_STEPS}")
    print(f"  Checkpoints: {len(CHECKPOINT_STEPS)}")
    print(f"Total experiments (seeds): {total_experiments}")
    print(f"Already completed: {completed_initial}")
    print(f"To run: {len(experiments_to_run)}")
    print(f"GPUs: {num_gpus}")
    print(f"Workers per GPU: {workers_per_gpu}")
    print(f"Total workers: {num_gpus * workers_per_gpu}")
    print(f"Output: {results_dir.absolute()}")
    print(f"=" * 60)
    sys.stdout.flush()

    if not experiments_to_run:
        print("All experiments already completed!")
        write_progress_state(results_dir, {
            'completed': total_experiments,
            'total': total_experiments,
            'percent': 100.0,
            'status': 'completed'
        })
        return

    # Shared state
    task_queue = Queue()
    completed_counter = Value('i', completed_initial)
    stop_flag = Value('b', False)

    # Shuffle for load balancing
    np.random.seed(42)
    np.random.shuffle(experiments_to_run)

    # Populate queue
    for exp in experiments_to_run:
        task_queue.put(exp)

    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n[{datetime.now().isoformat()}] Received signal {signum}, stopping workers...")
        sys.stdout.flush()
        stop_flag.value = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Launch workers
    processes = []
    total_workers = num_gpus * workers_per_gpu

    print(f"\n[{datetime.now().isoformat()}] Launching {total_workers} workers...")
    sys.stdout.flush()
    start_time = time.time()

    # Start progress monitor
    monitor_thread = progress_monitor(
        results_dir, completed_counter, total_experiments, start_time, stop_flag
    )

    # Write initial state
    write_progress_state(results_dir, {
        'completed': completed_initial,
        'total': total_experiments,
        'percent': round(100 * completed_initial / total_experiments, 2),
        'status': 'starting',
        'started': datetime.fromtimestamp(start_time).isoformat(),
        'pid': os.getpid()
    })

    for worker_id in range(total_workers):
        gpu_id = worker_id % num_gpus
        p = Process(
            target=worker_process,
            args=(worker_id, gpu_id, task_queue, results_dir,
                  completed_counter, total_experiments, stop_flag, allow_cpu)
        )
        p.start()
        processes.append(p)

    # Wait for completion
    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    final_completed = completed_counter.value

    # Write final state
    write_progress_state(results_dir, {
        'completed': final_completed,
        'total': total_experiments,
        'percent': round(100 * final_completed / total_experiments, 2),
        'elapsed_minutes': round(elapsed / 60, 1),
        'status': 'completed' if final_completed >= total_experiments else 'interrupted',
        'started': datetime.fromtimestamp(start_time).isoformat(),
        'finished': datetime.now().isoformat()
    })

    print(f"\n" + "=" * 60)
    print(f"Sweep {'completed' if final_completed >= total_experiments else 'interrupted'}!")
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Experiments completed: {final_completed}/{total_experiments}")
    if final_completed > completed_initial:
        print(f"Average time per experiment: {elapsed/max(1, final_completed - completed_initial):.1f}s")
    print(f"=" * 60)
    sys.stdout.flush()


def count_pending(results_dir: str = 'results'):
    """Count completed vs pending experiments."""
    results_dir = Path(results_dir)
    all_experiments = generate_experiment_grid()
    existing = set(f.name for f in results_dir.glob('*.h5'))

    completed = sum(1 for exp in all_experiments
                   if get_output_filename(exp) in existing)

    print(f"Completed: {completed}/{len(all_experiments)} "
          f"({100*completed/len(all_experiments):.1f}%)")
    return completed, len(all_experiments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run non-uniform sparsity experiment sweep')
    parser.add_argument('--gpus', type=int, default=DEFAULT_NUM_GPUS,
                       help='Number of GPUs to use')
    parser.add_argument('--workers-per-gpu', type=int, default=WORKERS_PER_GPU,
                       help='Number of workers per GPU')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not skip completed experiments')
    parser.add_argument('--count', action='store_true',
                       help='Just count completed experiments')
    parser.add_argument('--allow-cpu', action='store_true',
                       help='Allow CPU fallback when GPUs unavailable (slow)')

    args = parser.parse_args()

    if args.count:
        count_pending(args.results_dir)
    else:
        run_sweep(
            num_gpus=args.gpus,
            workers_per_gpu=args.workers_per_gpu,
            results_dir=args.results_dir,
            resume=not args.no_resume,
            allow_cpu=args.allow_cpu
        )
