"""
Delocalized Spectral Analysis Package
======================================

This package implements the spectral superposition experiment execution plan,
analyzing the relationship between eigengap structure, spectral spread, and
persistent dark matter features in neural network training dynamics.

Experiment Families:
-------------------
A. Eigengaps vs Projector Rotation
B. Spectral Spread Metrics
C. Within-Cluster Variance
D-H. Global Projective Linearity

Key Identity:
-------------
D_i(t) = ||w_i(t)||^2 / kappa_i(t)
kappa_i(t) = w_i(t)^T (WW^T) w_i(t) / ||w_i(t)||^2

Usage:
------
    from delocalized_analysis import run_all_experiments
    from delocalized_analysis.config import AnalysisConfig
    from delocalized_analysis.data_loader import RefinedSpectralLoader

Author: Claude Code (Anthropic)
Date: 2026-01-22
"""

from .config import AnalysisConfig, OUTPUT_DIR
from .data_loader import RefinedSpectralLoader, RunData
from .run_all_experiments import run_all_experiments, EXPERIMENTS

__version__ = "1.0.0"
__author__ = "Claude Code (Anthropic)"

__all__ = [
    'AnalysisConfig',
    'OUTPUT_DIR',
    'RefinedSpectralLoader',
    'RunData',
    'run_all_experiments',
    'EXPERIMENTS',
]
