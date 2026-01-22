#!/usr/bin/env python3
"""
Configuration constants for Dynamic Hopping Analysis.

This module defines all constants used across the analysis pipeline for computing
and analyzing feature "hopping" via time-variation of the Rayleigh quotient.

Mathematical Background:
------------------------
The Rayleigh quotient for feature i at time t is:
    κ_i(t) = (w_i(t)^T S(t) w_i(t)) / ||w_i(t)||²

where:
    - w_i(t): column i of W(t), the i-th feature vector at checkpoint t
    - S(t) = W(t) W(t)^T: the feature covariance matrix
    - ||w_i(t)||²: squared L2 norm of feature i

Derived quantities:
    - a_i(t) = ||w_i(t)||²: squared norm
    - x_i(t) = log(κ_i(t) + ε): log-transformed Rayleigh quotient (for stability)
    - D_i(t) = a_i(t) / κ_i(t): related to fractional dimension
"""

from pathlib import Path

# ============================================================================
# NUMERICAL CONSTANTS
# ============================================================================

# Guard for logarithms and divisions to prevent numerical instability
EPSILON = 1e-12

# Guard for Median Absolute Deviation (MAD) computation
EPSILON_MAD = 1e-9

# Jump threshold in robust-sigma (MAD-based) units
# A jump is detected when |Δx_i(t)| > z * σ_robust
Z_THRESHOLD = 4.0

# Late window: fraction of checkpoints to consider as "late training"
LATE_WINDOW_FRACTION = 0.2  # Last 20% of checkpoints

# Minimum valid values for regression
MIN_VALID_POINTS = 5

# ============================================================================
# SPARSITY BUCKETS (for stratified analysis)
# ============================================================================

SPARSITY_BUCKETS = {
    'low': (0.0, 0.3),
    'medium': (0.3, 0.7),
    'high': (0.7, 0.95),
    'extreme': (0.95, 1.0),
}

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path('/home/georgi/Spectral_Superposition/public/dynamics')
INPUT_DIR = BASE_DIR / 'start'
OUTPUT_DIR = BASE_DIR / 'analysis' / 'dynamic_hopping'
PLOTS_DIR = OUTPUT_DIR / 'plots'
RESULTS_DIR = OUTPUT_DIR / 'results'

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

# Block size for batched GEMM operations (for computing Rayleigh quotients)
# Smaller values use less memory but may be slower
FEATURE_BLOCK_SIZE = 256

# Number of files to process in parallel (for multi-GPU)
FILES_PER_GPU = 8

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Window size for computing local statistics (in number of checkpoints)
LOCAL_WINDOW_SIZE = 10

# Smoothing parameter for trend analysis
SMOOTHING_WINDOW = 5

# Classification thresholds for hopping patterns
HOPPING_CATEGORIES = {
    'stable': 0.5,      # σ_robust < 0.5: stable feature
    'moderate': 1.5,    # 0.5 <= σ_robust < 1.5: moderate hopping
    'active': 3.0,      # 1.5 <= σ_robust < 3.0: active hopping
    # σ_robust >= 3.0: extreme hopping
}
