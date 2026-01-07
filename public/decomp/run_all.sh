#!/bin/bash
#
# Launch 8 parallel sweep processes, one per A100 GPU.
# Each process runs its partition of experiments sequentially.
#

set -e

OUTPUT_DIR="${1:-results}"
TOTAL_STEPS="${2:-50000}"
N_GPUS=8

echo "=========================================="
echo "Toy Models of Superposition - Sweep"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Total steps: $TOTAL_STEPS"
echo "GPUs: $N_GPUS"
echo ""

# Print grid info
python3.9 sweep.py --gpu_id 0 --info
echo ""

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "Launching $N_GPUS processes..."
echo ""

# Launch one process per GPU in background
for gpu_id in $(seq 0 $((N_GPUS - 1))); do
    echo "Starting GPU $gpu_id..."
    CUDA_VISIBLE_DEVICES=$gpu_id python3.9 sweep.py \
        --gpu_id $gpu_id \
        --n_gpus $N_GPUS \
        --output_dir "$OUTPUT_DIR" \
        --total_steps $TOTAL_STEPS \
        > "logs/gpu_${gpu_id}.log" 2>&1 &

    echo "  PID: $!"
done

echo ""
echo "All processes launched. Logs in logs/gpu_*.log"
echo "Monitor progress with: tail -f logs/gpu_*.log"
echo "Check status with: ps aux | grep sweep.py"
echo ""

# Wait for all background processes
echo "Waiting for all processes to complete..."
wait

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
