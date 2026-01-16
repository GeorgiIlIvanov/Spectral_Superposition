#!/bin/bash
# Launch non-uniform sparsity experiment sweep
# S_i = i/n sparsity gradient, m=256, n=1024, 512 seeds

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source ../venv/bin/activate

# Create results and logs directories
mkdir -p results
mkdir -p logs

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/sweep_${TIMESTAMP}.log"

echo "========================================"
echo "Non-Uniform Sparsity Experiment"
echo "Started at $(date)"
echo "Log file: $LOG_FILE"
echo "========================================"

# Run the sweep with output to both console and log file
python sweep.py \
    --gpus 8 \
    --workers-per-gpu 4 \
    --results-dir results \
    2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "Experiment completed at $(date)"
echo "========================================"
