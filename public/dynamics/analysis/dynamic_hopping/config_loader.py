#!/usr/bin/env python3
"""
Configuration loader for Dynamic Hopping Analysis.

Loads constants from 00_config.py and provides them as a dictionary.
"""

from pathlib import Path


def load_config() -> dict:
    """Load configuration from 00_config.py"""
    from importlib import import_module
    import sys

    # Ensure the module can be found
    module_dir = Path(__file__).parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    # Import the config module
    try:
        # Try importing as a proper module name
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", module_dir / "00_config.py")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
    except Exception as e:
        raise ImportError(f"Could not load 00_config.py: {e}")

    return {
        # Numerical constants
        'epsilon': config_module.EPSILON,
        'epsilon_mad': config_module.EPSILON_MAD,
        'z_threshold': config_module.Z_THRESHOLD,
        'late_window_fraction': config_module.LATE_WINDOW_FRACTION,
        'min_valid_points': config_module.MIN_VALID_POINTS,

        # Sparsity buckets
        'sparsity_buckets': config_module.SPARSITY_BUCKETS,

        # Paths
        'base_dir': config_module.BASE_DIR,
        'input_dir': config_module.INPUT_DIR,
        'output_dir': config_module.OUTPUT_DIR,
        'plots_dir': config_module.PLOTS_DIR,
        'results_dir': config_module.RESULTS_DIR,

        # GPU configuration
        'feature_block_size': config_module.FEATURE_BLOCK_SIZE,
        'files_per_gpu': config_module.FILES_PER_GPU,

        # Analysis parameters
        'local_window_size': config_module.LOCAL_WINDOW_SIZE,
        'smoothing_window': config_module.SMOOTHING_WINDOW,
        'hopping_categories': config_module.HOPPING_CATEGORIES,
    }


def get_sparsity_bucket(sparsity: float, buckets: dict) -> str:
    """Get the sparsity bucket for a given sparsity value."""
    for bucket_name, (low, high) in buckets.items():
        if low <= sparsity < high:
            return bucket_name
    return 'unknown'
