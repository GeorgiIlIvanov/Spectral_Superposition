#!/bin/bash
# Refined Spectral Analysis Launcher
# This script runs the full spectral analysis pipeline
# Date: 2026-01-22

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/../venv"
LOG_FILE="${SCRIPT_DIR}/spectral_run.log"
GPUS="0,1,2,3,4,5,6,7"

echo "=============================================="
echo "Refined Spectral Analysis Pipeline"
echo "=============================================="
echo "Script directory: ${SCRIPT_DIR}"
echo "Virtual environment: ${VENV_DIR}"
echo "Log file: ${LOG_FILE}"
echo "GPUs: ${GPUS}"
echo "=============================================="

# Activate virtual environment
source "${VENV_DIR}/bin/activate"
echo "Python: $(which python)"
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=============================================="

# Change to script directory
cd "${SCRIPT_DIR}"

# Phase 1: Run main spectral analysis
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 1: Main spectral analysis..."
echo ""

python refined_spectral_analysis.py \
    --gpus ${GPUS} \
    --source-dir ../start \
    --svd-dir svd_results \
    --output-dir refined_spectral_results \
    --rayleigh-dir dynamic_hopping/results/per_file \
    2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 1 complete!"
echo ""

# Phase 2: Aggregate results
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 2: Aggregating results..."
echo ""

python aggregate_spectral_results.py \
    --input-dir refined_spectral_results \
    --output-dir refined_spectral_results/aggregated \
    2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2 complete!"
echo ""

# Summary
echo "=============================================="
echo "ANALYSIS COMPLETE"
echo "=============================================="
echo "Results directory: ${SCRIPT_DIR}/refined_spectral_results/"
echo "Aggregated results: ${SCRIPT_DIR}/refined_spectral_results/aggregated/"
echo "Log file: ${LOG_FILE}"
echo ""
echo "To verify results:"
echo "  python -c \"import numpy as np; d=np.load('refined_spectral_results/spectral_n1024_m64_s0.500000_seed0.npz'); print('Correlation:', np.corrcoef(d['kappa_actual'].flat, d['kappa_expected'].flat)[0,1])\""
echo "=============================================="
