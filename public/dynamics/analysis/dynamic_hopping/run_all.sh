#!/bin/bash
# ==============================================================================
# Dynamic Hopping Analysis Pipeline
# ==============================================================================
#
# This script runs the complete feature hopping analysis pipeline:
# 1. Compute Rayleigh quotients κ_i(t) for all checkpoints
# 2. Detect jumps using robust statistics (MAD-based)
# 3. Analyze temporal patterns and classify features
# 4. Generate comprehensive visualizations
#
# Usage:
#   ./run_all.sh              # Run full analysis
#   ./run_all.sh --sample 10  # Run on 10 files only (testing)
#   ./run_all.sh --gpu 1      # Use GPU 1
#
# ==============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/georgi/Spectral_Superposition/public/dynamics/venv/bin/python"

# Parse arguments
SAMPLE_ARG=""
GPU_ARG="--gpu 0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --sample)
            SAMPLE_ARG="--sample $2"
            shift 2
            ;;
        --gpu)
            GPU_ARG="--gpu $2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "=============================================================="
echo "Dynamic Feature Hopping Analysis Pipeline"
echo "=============================================================="
echo ""
echo "Working directory: $SCRIPT_DIR"
echo "Python: $PYTHON"
echo "Arguments: $SAMPLE_ARG $GPU_ARG"
echo ""

# Create output directories
mkdir -p "$SCRIPT_DIR/results/per_file"
mkdir -p "$SCRIPT_DIR/plots"

# Step 1: Compute Rayleigh quotients
echo ""
echo "=============================================================="
echo "Step 1: Computing Rayleigh Quotients"
echo "=============================================================="
$PYTHON "$SCRIPT_DIR/01_rayleigh_quotients.py" $GPU_ARG $SAMPLE_ARG

# Step 2: Detect jumps
echo ""
echo "=============================================================="
echo "Step 2: Jump Detection"
echo "=============================================================="
$PYTHON "$SCRIPT_DIR/02_jump_detection.py" $SAMPLE_ARG

# Step 3: Analyze temporal patterns
echo ""
echo "=============================================================="
echo "Step 3: Temporal Pattern Analysis"
echo "=============================================================="
$PYTHON "$SCRIPT_DIR/03_temporal_patterns.py" $SAMPLE_ARG

# Step 4: Generate visualizations
echo ""
echo "=============================================================="
echo "Step 4: Generating Visualizations"
echo "=============================================================="
$PYTHON "$SCRIPT_DIR/04_visualizations.py"

echo ""
echo "=============================================================="
echo "Pipeline Complete!"
echo "=============================================================="
echo ""
echo "Results saved to: $SCRIPT_DIR/results/"
echo "Plots saved to: $SCRIPT_DIR/plots/"
echo ""
