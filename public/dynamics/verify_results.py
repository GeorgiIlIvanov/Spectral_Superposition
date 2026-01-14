"""
Verify sweep completion and data integrity.
"""
import h5py
import numpy as np
from pathlib import Path
from itertools import product
import argparse
from collections import defaultdict

# Expected configuration
N_FEATURES = 1024
M_VALUES = np.linspace(16, 512, 32).astype(int)
S_VALUES = np.linspace(0, 0.99, 50)
SEEDS = [0, 1]
N_CHECKPOINTS = 56


def get_expected_files():
    """Generate set of all expected filenames."""
    expected = set()
    for m, s, seed in product(M_VALUES, S_VALUES, SEEDS):
        filename = f"n{N_FEATURES}_m{int(m)}_s{float(s):.6f}_seed{int(seed)}.h5"
        expected.add(filename)
    return expected


def verify_file(filepath: Path) -> tuple:
    """
    Verify a single HDF5 file for data integrity.

    Returns: (is_valid, error_message)
    """
    try:
        with h5py.File(filepath, 'r') as f:
            # Check required datasets exist
            required = ['checkpoint_steps', 'weights', 'fractional_dims',
                       'feature_norms', 'biases', 'losses']
            for key in required:
                if key not in f:
                    return False, f"Missing dataset: {key}"

            # Check shapes
            m_hidden = f.attrs.get('m_hidden', f['weights'].shape[1])
            n_features = f.attrs.get('n_features', N_FEATURES)

            if f['weights'].shape != (N_CHECKPOINTS, m_hidden, n_features):
                return False, f"Wrong weights shape: {f['weights'].shape}"

            if f['fractional_dims'].shape != (N_CHECKPOINTS, n_features):
                return False, f"Wrong fractional_dims shape: {f['fractional_dims'].shape}"

            if f['feature_norms'].shape != (N_CHECKPOINTS, n_features):
                return False, f"Wrong feature_norms shape: {f['feature_norms'].shape}"

            if f['losses'].shape != (N_CHECKPOINTS,):
                return False, f"Wrong losses shape: {f['losses'].shape}"

            # Check for NaN/Inf in critical arrays
            if np.any(np.isnan(f['losses'][:]) | np.isinf(f['losses'][:])):
                return False, "NaN/Inf in losses"

            # Sanity check: final loss should be reasonable
            final_loss = f['losses'][-1]
            if final_loss > 10.0:
                return False, f"Unusually high final loss: {final_loss}"

        return True, None

    except Exception as e:
        return False, str(e)


def verify_sweep(results_dir: str = 'results_v2', full_check: bool = False):
    """
    Verify sweep completion and data integrity.

    Args:
        results_dir: Directory containing HDF5 results
        full_check: If True, verify all files; else sample check
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Get expected and actual files
    expected_files = get_expected_files()
    actual_files = set(f.name for f in results_dir.glob('*.h5'))
    h5_files = list(results_dir.glob('*.h5'))

    # Completion stats
    expected_count = len(expected_files)
    actual_count = len(actual_files)
    completion_pct = 100 * actual_count / expected_count

    print("=" * 60)
    print("Sweep Verification Report")
    print("=" * 60)
    print(f"\nCompletion: {actual_count} / {expected_count} ({completion_pct:.1f}%)")

    # Missing files
    missing = expected_files - actual_files
    if missing:
        print(f"\nMissing files: {len(missing)}")
        if len(missing) <= 10:
            for f in sorted(missing):
                print(f"  - {f}")
        else:
            print("  (showing first 10)")
            for f in sorted(missing)[:10]:
                print(f"  - {f}")

    # Extra files (unexpected)
    extra = actual_files - expected_files
    if extra:
        print(f"\nUnexpected files: {len(extra)}")
        for f in sorted(extra)[:5]:
            print(f"  - {f}")

    # Storage stats
    if h5_files:
        total_bytes = sum(f.stat().st_size for f in h5_files)
        avg_bytes = total_bytes / len(h5_files)
        print(f"\nStorage:")
        print(f"  Total: {total_bytes / 1e9:.2f} GB")
        print(f"  Per file: {avg_bytes / 1e6:.2f} MB")

    # Data integrity check
    print(f"\nData Integrity:")
    if full_check:
        print(f"  Checking all {len(h5_files)} files...")
        files_to_check = h5_files
    else:
        # Sample check: random subset + edge cases
        sample_size = min(100, len(h5_files))
        np.random.seed(42)
        sample_indices = np.random.choice(len(h5_files), sample_size, replace=False)
        files_to_check = [h5_files[i] for i in sample_indices]
        print(f"  Sampling {sample_size} files...")

    errors = []
    for filepath in files_to_check:
        is_valid, error = verify_file(filepath)
        if not is_valid:
            errors.append((filepath.name, error))

    if errors:
        print(f"\n  Found {len(errors)} corrupted files:")
        for name, err in errors[:10]:
            print(f"    - {name}: {err}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
    else:
        check_type = "All" if full_check else "Sample"
        print(f"  {check_type} files passed integrity check")

    # Parameter coverage analysis
    print(f"\nParameter Coverage:")
    m_coverage = defaultdict(int)
    s_coverage = defaultdict(int)
    seed_coverage = defaultdict(int)

    for filename in actual_files:
        # Parse filename: n1024_m{m}_s{s}_seed{seed}.h5
        parts = filename.replace('.h5', '').split('_')
        try:
            m = int(parts[1][1:])  # m{value}
            s = float(parts[2][1:])  # s{value}
            seed = int(parts[3][4:])  # seed{value}
            m_coverage[m] += 1
            s_coverage[round(s, 6)] += 1
            seed_coverage[seed] += 1
        except (IndexError, ValueError):
            continue

    # Expected counts per parameter
    expected_per_m = len(S_VALUES) * len(SEEDS)  # 100
    expected_per_s = len(M_VALUES) * len(SEEDS)  # 64
    expected_per_seed = len(M_VALUES) * len(S_VALUES)  # 1600

    incomplete_m = [m for m, c in m_coverage.items() if c < expected_per_m]
    incomplete_s = [s for s, c in s_coverage.items() if c < expected_per_s]

    if incomplete_m:
        print(f"  Incomplete m_hidden values: {len(incomplete_m)}")
        if len(incomplete_m) <= 5:
            for m in sorted(incomplete_m):
                print(f"    m={m}: {m_coverage[m]}/{expected_per_m}")
    else:
        print(f"  All {len(M_VALUES)} m_hidden values complete")

    if incomplete_s:
        print(f"  Incomplete sparsity values: {len(incomplete_s)}")
    else:
        print(f"  All {len(S_VALUES)} sparsity values complete")

    for seed in SEEDS:
        print(f"  Seed {seed}: {seed_coverage[seed]}/{expected_per_seed}")

    print("\n" + "=" * 60)

    return {
        'completed': actual_count,
        'expected': expected_count,
        'missing': len(missing),
        'corrupted': len(errors),
        'storage_gb': total_bytes / 1e9 if h5_files else 0
    }


def list_missing(results_dir: str = 'results_v2', output_file: str = None):
    """Generate list of missing experiments for re-running."""
    results_dir = Path(results_dir)
    expected = get_expected_files()
    actual = set(f.name for f in results_dir.glob('*.h5'))
    missing = expected - actual

    if output_file:
        with open(output_file, 'w') as f:
            for filename in sorted(missing):
                f.write(filename + '\n')
        print(f"Wrote {len(missing)} missing filenames to {output_file}")
    else:
        for filename in sorted(missing):
            print(filename)

    return missing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify sweep results')
    parser.add_argument('--results-dir', type=str, default='results_v2',
                       help='Results directory')
    parser.add_argument('--full-check', action='store_true',
                       help='Check all files (slow)')
    parser.add_argument('--list-missing', action='store_true',
                       help='List missing experiments')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for missing list')

    args = parser.parse_args()

    if args.list_missing:
        list_missing(args.results_dir, args.output)
    else:
        verify_sweep(args.results_dir, args.full_check)
