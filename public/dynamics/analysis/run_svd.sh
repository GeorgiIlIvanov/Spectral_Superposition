#!/bin/bash
# Wrapper script to run SVD computation with proper environment

set -e

cd /home/georgi/Spectral_Superposition/public/dynamics

# Activate virtual environment
source venv/bin/activate

# Run the SVD computation
echo "Starting SVD computation at $(date)"
echo "========================================"

python analysis/01_compute_svds.py --num-workers 8 --resume

echo "========================================"
echo "SVD computation completed at $(date)"
echo ""
echo "Press any key to exit, or this session will remain open for inspection."
read -n 1
