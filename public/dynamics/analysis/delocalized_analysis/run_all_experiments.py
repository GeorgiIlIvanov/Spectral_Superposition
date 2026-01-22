#!/usr/bin/env python3
"""
Main Orchestrator for Delocalized Spectral Analysis Experiments
================================================================

This script runs all experiments (A, B, C, D-H) from the spectral superposition
analysis plan and produces aggregated summary reports.

Usage:
------
    # Run all experiments
    python run_all_experiments.py

    # Run specific experiments
    python run_all_experiments.py --experiments A B

    # Limit files for testing
    python run_all_experiments.py --max-files 100

    # Parallel processing (using multiprocessing for experiments)
    python run_all_experiments.py --parallel

Output:
-------
Results are saved to delocalized_analysis/results/ with the following structure:
    results/
        experiment_A/
        experiment_B/
        experiment_C/
        experiments_D_H/
        aggregate_summary.json
        figures/

Author: Claude Code (Anthropic)
Date: 2026-01-22
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import AnalysisConfig, OUTPUT_DIR, validate_data_directories
from data_loader import RefinedSpectralLoader
from experiment_A import run_experiment_A
from experiment_B import run_experiment_B
from experiment_C import run_experiment_C
from experiments_D_H import run_experiments_D_H

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / 'experiment_run.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Experiment Registry
# =============================================================================

EXPERIMENTS = {
    'A': {
        'name': 'Eigengaps vs Projector Rotation',
        'function': run_experiment_A,
        'description': 'Tests whether small eigengaps in bulk lead to high rotation and DM',
    },
    'B': {
        'name': 'Spectral Spread Metrics',
        'function': run_experiment_B,
        'description': 'Tests whether DM features have high spectral entropy/PR',
    },
    'C': {
        'name': 'Within-Cluster Variance',
        'function': run_experiment_C,
        'description': 'Analyzes variance patterns in bulk vs spiked eigenvalue regimes',
    },
    'D_H': {
        'name': 'Global Projective Linearity',
        'function': run_experiments_D_H,
        'description': 'Tests global slope alpha(t) hypothesis and universal diffusion',
    },
}


# =============================================================================
# Main Runner
# =============================================================================

def run_single_experiment(
    experiment_key: str,
    config: AnalysisConfig,
    max_files: Optional[int] = None
) -> Dict:
    """
    Run a single experiment.

    Parameters
    ----------
    experiment_key : str
        Experiment key (A, B, C, or D_H)
    config : AnalysisConfig
        Analysis configuration
    max_files : int, optional
        Maximum files to process

    Returns
    -------
    Dict
        Experiment results
    """
    if experiment_key not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_key}")

    exp_info = EXPERIMENTS[experiment_key]
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Experiment {experiment_key}: {exp_info['name']}")
    logger.info(f"Description: {exp_info['description']}")
    logger.info(f"{'='*60}\n")

    start_time = time.time()

    try:
        result = exp_info['function'](
            config=config,
            max_files=max_files
        )
        elapsed = time.time() - start_time

        result['_metadata'] = {
            'experiment': experiment_key,
            'name': exp_info['name'],
            'elapsed_seconds': elapsed,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
        }

        logger.info(f"Experiment {experiment_key} completed in {elapsed:.1f}s")
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Experiment {experiment_key} failed: {e}")
        logger.error(traceback.format_exc())

        return {
            '_metadata': {
                'experiment': experiment_key,
                'name': exp_info['name'],
                'elapsed_seconds': elapsed,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }
        }


def run_all_experiments(
    experiments: List[str] = None,
    config: AnalysisConfig = None,
    max_files: Optional[int] = None,
    parallel: bool = False
) -> Dict:
    """
    Run all specified experiments.

    Parameters
    ----------
    experiments : List[str], optional
        List of experiment keys to run. If None, runs all.
    config : AnalysisConfig, optional
        Analysis configuration
    max_files : int, optional
        Maximum files per experiment
    parallel : bool
        Whether to run experiments in parallel

    Returns
    -------
    Dict
        Aggregated results from all experiments
    """
    if config is None:
        config = AnalysisConfig()

    if experiments is None:
        experiments = list(EXPERIMENTS.keys())

    # Validate experiments
    invalid = [e for e in experiments if e not in EXPERIMENTS]
    if invalid:
        raise ValueError(f"Unknown experiments: {invalid}")

    logger.info(f"\n{'#'*60}")
    logger.info("DELOCALIZED SPECTRAL ANALYSIS")
    logger.info(f"{'#'*60}")
    logger.info(f"Experiments to run: {experiments}")
    logger.info(f"Max files per experiment: {max_files or 'all'}")
    logger.info(f"Parallel execution: {parallel}")

    # Validate data directories
    validation = validate_data_directories()
    if not validation['valid']:
        logger.error(f"Data validation failed: {validation['missing']}")
        raise RuntimeError("Required data directories missing")

    logger.info(f"Data validation passed:")
    logger.info(f"  Refined spectral files: {validation['refined_spectral_files']}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = {}
    total_start = time.time()

    if parallel and len(experiments) > 1:
        # Run in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(len(experiments), 4)) as executor:
            futures = {
                executor.submit(run_single_experiment, exp, config, max_files): exp
                for exp in experiments
            }

            for future in as_completed(futures):
                exp = futures[future]
                try:
                    result = future.result()
                    all_results[exp] = result
                except Exception as e:
                    logger.error(f"Experiment {exp} failed in parallel execution: {e}")
                    all_results[exp] = {
                        '_metadata': {
                            'experiment': exp,
                            'status': 'failed',
                            'error': str(e),
                        }
                    }
    else:
        # Run sequentially
        for exp in experiments:
            result = run_single_experiment(exp, config, max_files)
            all_results[exp] = result

    total_elapsed = time.time() - total_start

    # Generate summary
    summary = generate_summary(all_results, total_elapsed)

    # Save summary
    summary_path = OUTPUT_DIR / 'aggregate_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n{'#'*60}")
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"{'#'*60}")
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info(f"Summary saved to: {summary_path}")

    # Print summary table
    print_summary_table(summary)

    return summary


def generate_summary(results: Dict, total_elapsed: float) -> Dict:
    """
    Generate aggregate summary from all experiment results.

    Parameters
    ----------
    results : Dict
        Results from all experiments
    total_elapsed : float
        Total elapsed time in seconds

    Returns
    -------
    Dict
        Aggregate summary
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_elapsed_seconds': total_elapsed,
        'experiments': {},
        'key_findings': {},
    }

    for exp_key, result in results.items():
        metadata = result.get('_metadata', {})
        summary['experiments'][exp_key] = {
            'name': metadata.get('name', exp_key),
            'status': metadata.get('status', 'unknown'),
            'elapsed_seconds': metadata.get('elapsed_seconds', 0),
        }

        # Extract key findings per experiment
        if metadata.get('status') == 'success':
            findings = extract_key_findings(exp_key, result)
            summary['key_findings'][exp_key] = findings

    return summary


def extract_key_findings(exp_key: str, result: Dict) -> Dict:
    """Extract key findings from experiment results."""
    findings = {}

    if exp_key == 'A':
        findings['dm_rate'] = result.get('dm_rate_mean', np.nan)
        findings['gap_rotation_correlation'] = result.get('correlations', {}).get('gap_vs_rotation_mean', np.nan)
        findings['gap_dm_correlation'] = result.get('correlations', {}).get('gap_vs_dm_mean', np.nan)

    elif exp_key == 'B':
        findings['dm_rate'] = result.get('dm_rate_mean', np.nan)
        findings['model_auc'] = result.get('predictive_model', {}).get('auc_mean', np.nan)
        findings['falsifier_auc'] = result.get('falsifier', {}).get('auc_mean', np.nan)
        findings['spread_separates_in_bulk'] = result.get('falsifier', {}).get('spread_separates_rate', np.nan)

    elif exp_key == 'C':
        findings['bulk_higher_variance'] = result.get('var_vs_lambda', {}).get('bulk_higher_variance_rate', np.nan)
        findings['bulk_dm_rate'] = result.get('dm_concentration', {}).get('bulk_dm_rate_mean', np.nan)
        findings['spiked_dm_rate'] = result.get('dm_concentration', {}).get('spiked_dm_rate_mean', np.nan)

    elif exp_key == 'D_H':
        findings['cross_sectional_r2'] = result.get('D_cross_sectional', {}).get('late_r2_mean', np.nan)
        findings['slope_near_unity_rate'] = result.get('E_slope_matching', {}).get('near_unity_rate', np.nan)
        findings['dm_stabilization_rate'] = result.get('H_normalization', {}).get('stabilization_rate_mean', np.nan)

    return findings


def print_summary_table(summary: Dict):
    """Print a summary table to console."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    # Status table
    print(f"\n{'Experiment':<15} {'Status':<10} {'Time (s)':<12} {'Key Metric'}")
    print("-" * 70)

    for exp_key, exp_info in summary['experiments'].items():
        status = exp_info['status']
        elapsed = exp_info['elapsed_seconds']

        # Get a key metric
        findings = summary['key_findings'].get(exp_key, {})
        if exp_key == 'A':
            key_metric = f"gap-DM corr: {findings.get('gap_dm_correlation', np.nan):.3f}"
        elif exp_key == 'B':
            key_metric = f"AUC: {findings.get('model_auc', np.nan):.3f}"
        elif exp_key == 'C':
            key_metric = f"bulk>spiked var: {findings.get('bulk_higher_variance', np.nan):.1%}"
        elif exp_key == 'D_H':
            key_metric = f"R2: {findings.get('cross_sectional_r2', np.nan):.3f}"
        else:
            key_metric = ""

        print(f"{exp_key:<15} {status:<10} {elapsed:<12.1f} {key_metric}")

    print("-" * 70)
    print(f"Total time: {summary['total_elapsed_seconds']/60:.1f} minutes")
    print("=" * 70)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run Delocalized Spectral Analysis Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--experiments', '-e',
        nargs='+',
        choices=list(EXPERIMENTS.keys()),
        default=None,
        help='Experiments to run (default: all)'
    )

    parser.add_argument(
        '--max-files', '-n',
        type=int,
        default=None,
        help='Maximum files to process per experiment'
    )

    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Run experiments in parallel'
    )

    parser.add_argument(
        '--r2-threshold',
        type=float,
        default=0.9,
        help='R^2 threshold for persistent DM labeling (default: 0.9)'
    )

    parser.add_argument(
        '--late-window',
        type=int,
        default=10,
        help='Late window size in checkpoints (default: 10)'
    )

    parser.add_argument(
        '--list-experiments',
        action='store_true',
        help='List available experiments and exit'
    )

    args = parser.parse_args()

    if args.list_experiments:
        print("\nAvailable Experiments:")
        print("-" * 60)
        for key, info in EXPERIMENTS.items():
            print(f"  {key}: {info['name']}")
            print(f"      {info['description']}")
        return

    # Create configuration
    config = AnalysisConfig(
        r2_threshold=args.r2_threshold,
        late_window_size=args.late_window,
    )

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run experiments
    run_all_experiments(
        experiments=args.experiments,
        config=config,
        max_files=args.max_files,
        parallel=args.parallel
    )


if __name__ == '__main__':
    main()
