#!/bin/bash
#
# Run all temporal analyses for Spectral Superposition project
#
# Usage:
#   ./run_all_analyses.sh [--sample N] [--skip-animation]
#
# Options:
#   --sample N       Process only N files for quick testing
#   --skip-animation Skip GIF generation in phase animation
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source /home/georgi/Spectral_Superposition/public/dynamics/venv/bin/activate

# Parse arguments
SAMPLE_ARG=""
SKIP_ANIM=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample)
            SAMPLE_ARG="--sample $2"
            shift 2
            ;;
        --skip-animation)
            SKIP_ANIM="--skip-animation"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directories
mkdir -p plots results

echo "============================================================"
echo "Temporal Analysis Suite for Spectral Superposition"
echo "============================================================"
echo ""
echo "Output directories:"
echo "  Plots:   $SCRIPT_DIR/plots/"
echo "  Results: $SCRIPT_DIR/results/"
echo ""

# Analysis 1: Dark Matter Evolution
echo "============================================================"
echo "Running Analysis 1: Dark Matter Evolution"
echo "============================================================"
python 01_dark_matter_evolution.py $SAMPLE_ARG
echo ""

# Analysis 2: Eigenspace Stability (requires SVD results)
if [ -d "../../svd_results" ] || [ -d "../svd_results" ]; then
    echo "============================================================"
    echo "Running Analysis 2: Eigenspace Stability"
    echo "============================================================"
    python 02_eigenspace_stability.py $SAMPLE_ARG
    echo ""
else
    echo "Skipping Analysis 2: SVD results not found"
    echo "Run the SVD analysis first: cd ../svd_analysis && ./run_svd.sh"
    echo ""
fi

# Analysis 3: Instantaneous Linearity
echo "============================================================"
echo "Running Analysis 3: Instantaneous Linearity"
echo "============================================================"
python 03_instantaneous_linearity.py $SAMPLE_ARG
echo ""

# Analysis 4: Slope-Eigenvalue Temporal (requires SVD results)
if [ -d "../../svd_results" ] || [ -d "../svd_results" ]; then
    echo "============================================================"
    echo "Running Analysis 4: Slope-Eigenvalue Temporal"
    echo "============================================================"
    python 04_slope_eigenvalue_temporal.py $SAMPLE_ARG
    echo ""
else
    echo "Skipping Analysis 4: SVD results not found"
    echo ""
fi

# Analysis 5: Trajectory Classification
echo "============================================================"
echo "Running Analysis 5: Trajectory Classification"
echo "============================================================"
python 05_trajectory_classification.py $SAMPLE_ARG
echo ""

# Analysis 6: Concentration Dynamics (requires SVD results)
if [ -d "../../svd_results" ] || [ -d "../svd_results" ]; then
    echo "============================================================"
    echo "Running Analysis 6: Concentration Dynamics"
    echo "============================================================"
    python 06_concentration_dynamics.py $SAMPLE_ARG
    echo ""
else
    echo "Skipping Analysis 6: SVD results not found"
    echo ""
fi

# Analysis 7: Phase Animation
echo "============================================================"
echo "Running Analysis 7: Phase Animation"
echo "============================================================"
python 07_phase_animation.py $SKIP_ANIM
echo ""

# Analysis 8: Aggregate Summary
echo "============================================================"
echo "Running Analysis 8: Aggregate Summary"
echo "============================================================"
python 08_aggregate_summary.py
echo ""

echo "============================================================"
echo "All analyses complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - plots/dark_matter_evolution.png"
echo "  - plots/eigenspace_stability.png"
echo "  - plots/instantaneous_linearity.png"
echo "  - plots/slope_eigenvalue_temporal.png"
echo "  - plots/trajectory_classification.png"
echo "  - plots/concentration_dynamics.png"
echo "  - plots/phase_comparison.png"
echo "  - plots/phase_tracers.png"
echo "  - plots/aggregate_summary.png"
echo ""
echo "Report: ANALYSIS_REPORT.md"
