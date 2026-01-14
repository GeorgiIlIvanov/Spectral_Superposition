#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="$1"
NUM_GPUS="$2"
WORKERS_PER_GPU="$3"
LOG_FILE="$4"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

echo "========================================"
echo "Experiment started at $(date)"
echo "PID: $$"
echo "Python: $(which python)"
echo "Results: $RESULTS_DIR"
echo "GPUs: $NUM_GPUS"
echo "Workers/GPU: $WORKERS_PER_GPU"
echo "========================================"

# Run with automatic restart on failure
MAX_RETRIES=5
RETRY_COUNT=0

while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
    python sweep.py \
        --gpus "$NUM_GPUS" \
        --workers-per-gpu "$WORKERS_PER_GPU" \
        --results-dir "$RESULTS_DIR"

    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "========================================"
        echo "Experiment completed successfully at $(date)"
        echo "========================================"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "========================================"
        echo "Experiment failed with code $EXIT_CODE at $(date)"
        echo "Retry $RETRY_COUNT / $MAX_RETRIES in 30 seconds..."
        echo "========================================"
        sleep 30
    fi
done

if [[ $RETRY_COUNT -ge $MAX_RETRIES ]]; then
    echo "Max retries exceeded. Check logs."
fi
