#!/bin/bash
# Master script to run complete slope-eigenvalue analysis
# Run this after SVD computation is complete

set -e

cd /home/georgi/Spectral_Superposition/public/dynamics
source venv/bin/activate

echo "========================================"
echo "Full Slope-Eigenvalue Analysis Pipeline"
echo "========================================"
echo "Started at $(date)"
echo ""

# Check SVD completion
SVD_COUNT=$(ls -1 analysis/svd_results/svd_*.h5 2>/dev/null | wc -l)
echo "SVD files found: $SVD_COUNT / 3200"

if [ "$SVD_COUNT" -lt 3200 ]; then
    echo "WARNING: SVD computation may not be complete!"
    echo "Proceeding anyway..."
fi

echo ""
echo "========================================"
echo "Phase 2: Feature-to-Eigenspace Clustering"
echo "========================================"
python analysis/02_cluster_features.py

echo ""
echo "========================================"
echo "Phase 3: Slope-Eigenvalue Correlation Analysis"
echo "========================================"
python analysis/03_slope_eigenvalue_analysis.py --checkpoint final

echo ""
echo "========================================"
echo "Analysis Complete!"
echo "========================================"
echo "Finished at $(date)"
echo ""
echo "Results saved in:"
echo "  - analysis/clustering_results/"
echo "  - analysis/slope_eigenvalue_results/"
echo ""
echo "Key output files:"
echo "  - slope_eigenvalue_results/correlation_results.json"
echo "  - slope_eigenvalue_results/slope_eigenvalue_correlation.png"
