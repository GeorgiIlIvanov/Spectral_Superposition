#!/usr/bin/env python3
"""
Parallel GIF generation for pseudospectrum episodes (one GIF per instance),
PLUS (optionally) a 2D Stieltjes transform heatmap and eigenvalue trajectories,
rendered side-by-side in the same evolving GIF.

Usage examples:
  # 3-panel GIF: pseudospectrum | Stieltjes heatmap | eigenvalue trajectories
  python src/scripts/multiple_spectra_evolutions.py \
    --snaps public/checkpoints/W_snaps_granular_hidden_dim_2.pkl --out_dir gifs --which feature \
    --grid_size 250 --stieltjes_grid 128 --stieltjes_mode logabs \
    --layout 3panel --fps 3 --max_workers 8

  # 2-panel GIF: pseudospectrum | Stieltjes heatmap
  python src/scripts/multiple_spectra_evolutions.py \
    --snaps W_snaps_granular_hidden_dim_2.pkl --out_dir gifs --which hidden \
    --grid_size 250 --layout 2panel --fps 3

Key points for FAST mode:
- No pyplot.show().
- Matplotlib "Agg" backend.
- Force PseudoPy to draw into a given Axes via plt.sca(ax).
"""
import os

# Avoid BLAS oversubscription when using multiple processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # IMPORTANT for headless + multiprocessing
import matplotlib.pyplot as plt

import imageio.v2 as imageio


def grams_from_Wi(Wi: np.ndarray):
    """Wi: [n_features, n_hidden] -> (G_feature, G_hidden)"""
    Gf = Wi @ Wi.T
    Gh = Wi.T @ Wi
    return Gf, Gh


def _fig_to_rgb_array(fig):
    """Convert a Matplotlib figure to an RGB numpy array (H, W, 3)."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape((h, w, 4))[:, :, :3]  # drop alpha
    return buf


def generate_pseudospectrum(
    weights,
    step=None,
    pad=0.5,
    xlim=None,
    ylim=None,
    ax=None,
):
    """
    Draw epsilon-pseudospectrum into ax (fast PseudoPy method).
    NEVER calls show(); returns (fig, ax, info) for GIF capture.
    """
    A = np.asarray(weights)
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "weights must be a square matrix"

    eigs = np.linalg.eigvals(A)

    # Infer plot window if not provided
    if xlim is None or ylim is None:
        re = eigs.real
        im = eigs.imag
        xmin, xmax = re.min() - pad, re.max() + pad
        ymin, ymax = im.min() - pad, im.max() + pad
        if np.isclose(xmin, xmax):
            xmin, xmax = xmin - 1.0, xmax + 1.0
        if np.isclose(ymin, ymax):
            ymin, ymax = ymin - 1.0, ymax + 1.0
        if xlim is None:
            xlim = (xmin, xmax)
        if ylim is None:
            ylim = (ymin, ymax)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure

    ax.cla()

    try:
        from pseudopy import NonnormalAuto
        from scipy.linalg import eigvals as scipy_eigvals
    except Exception as e:
        raise RuntimeError(
            "PseudoPy fast mode requires pseudopy and scipy to be installed."
        ) from e

    # Set limits BEFORE plotting; re-apply after plotting as PseudoPy may change them.
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    plt.sca(ax)

    pseudo = NonnormalAuto(A, 1e-5, 1)
    levels = [10**k for k in range(-4, 0)]
    spectrum = scipy_eigvals(A)
    pseudo.plot(levels, spectrum=spectrum)

    ax.set_title(f"ε-pseudospectrum (PseudoPy fast) step={step}")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    info = dict(levels=levels, xlim=xlim, ylim=ylim, eigenvalues=eigs)
    return fig, ax, info


def stieltjes_heatmap_from_eigs(
    eigs: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    grid: int,
    mode: str = "logabs",
    eps_im: float = 1e-6,
):
    """
    Compute Stieltjes transform m(z)=mean(1/(λ-z)) over a complex grid using eigenvalues λ.
    Returns Zplot (real-valued) for imshow.
    """
    xs = np.linspace(xlim[0], xlim[1], grid)
    ys = np.linspace(ylim[0], ylim[1], grid)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y

    # Avoid exact hits on the real axis / eigenvalues
    Z = Z + 1j * eps_im

    lam = eigs.reshape(1, 1, -1)
    denom = lam - Z[..., None]
    m = np.mean(1.0 / denom, axis=-1)

    if mode == "logabs":
        Zplot = np.log10(np.abs(m) + 1e-30)
    elif mode == "abs":
        Zplot = np.abs(m)
    elif mode == "im":
        Zplot = np.imag(m)
    else:
        raise ValueError(f"Unknown stieltjes mode: {mode}")

    return Zplot


def plot_stieltjes_heatmap(ax, eigs, xlim, ylim, grid, mode="logabs", vmin=None, vmax=None):
    ax.cla()
    Zplot = stieltjes_heatmap_from_eigs(eigs, xlim, ylim, grid=grid, mode=mode)
    im = ax.imshow(
        Zplot,
        origin="lower",
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"Stieltjes heatmap: {mode}")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    return im


def plot_eig_trajectories(ax, eigs_sorted_over_time: np.ndarray, frame_idx: int):
    """
    For Gram matrices eigenvalues are (numerically) real.
    Plot trajectories by connecting sorted-eigenvalue index over time.
    """
    ax.cla()
    T, n = eigs_sorted_over_time.shape
    upto = frame_idx + 1

    for k in range(n):
        ax.plot(eigs_sorted_over_time[:upto, k], np.zeros(upto), linewidth=0.8, alpha=0.35)

    curr = eigs_sorted_over_time[frame_idx]
    ax.scatter(curr, np.zeros_like(curr), s=12)

    ax.set_title("Eigenvalue trajectories (real axis)")
    ax.set_xlabel("λ")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)

    all_vals = eigs_sorted_over_time.reshape(-1)
    xmin, xmax = float(all_vals.min()), float(all_vals.max())
    if np.isclose(xmin, xmax):
        xmin, xmax = xmin - 1.0, xmax + 1.0
    ax.set_xlim(xmin, xmax)


def _parse_int_list(s):
    if s is None:
        return None
    parts = []
    for chunk in s.replace(",", " ").split():
        if chunk.strip():
            parts.append(int(chunk))
    return parts if parts else None


def _resolve_steps(W_snaps, steps_arg):
    available_steps = sorted(W_snaps["steps"].keys())
    if steps_arg is None:
        return available_steps
    return [s for s in steps_arg if s in W_snaps["steps"]]


def _resolve_instances(W_snaps, instances_arg):
    steps = sorted(W_snaps["steps"].keys())
    W0 = W_snaps["steps"][steps[0]]
    n_instances = W0.shape[0]
    if instances_arg is None:
        return list(range(n_instances))
    return [i for i in instances_arg if 0 <= i < n_instances]


def _compute_limits_from_all_steps(W_snaps, instance_id, steps, which, pad):
    """Compute global limits across all steps for consistent framing."""
    all_re = []
    all_im = []
    for step in steps:
        Wi = W_snaps["steps"][step][instance_id]
        Gf, Gh = grams_from_Wi(Wi)
        A = Gf if which == "feature" else Gh
        eigs = np.linalg.eigvals(np.asarray(A))
        all_re.append(eigs.real)
        all_im.append(eigs.imag)

    all_re = np.concatenate(all_re)
    all_im = np.concatenate(all_im)

    xlim = (all_re.min() - pad, all_re.max() + pad)
    ylim = (all_im.min() - pad, all_im.max() + pad)
    if np.isclose(xlim[0], xlim[1]):
        xlim = (xlim[0] - 1.0, xlim[1] + 1.0)
    if np.isclose(ylim[0], ylim[1]):
        ylim = (ylim[0] - 1.0, ylim[1] + 1.0)
    return xlim, ylim


def generate_gif_for_instance(
    snaps_path: str,
    out_dir: str,
    instance_id: int,
    which: str,
    grid_size: int,
    fps: int,
    pad: float,
    steps_list,
    layout: str,
    stieltjes_mode: str,
    stieltjes_grid: Optional[int],
    stieltjes_fix_scale: bool,
):
    start_time = time.time()

    # Load inside worker to avoid pickling huge objects
    load_start = time.time()
    with open(snaps_path, "rb") as f:
        W_snaps = pickle.load(f)
    load_time = time.time() - load_start
    print(f"[Instance {instance_id}] Loaded pickle in {load_time:.2f}s", flush=True)

    steps = _resolve_steps(W_snaps, steps_list)
    print(f"[Instance {instance_id}] Processing {len(steps)} steps", flush=True)

    # Global complex-plane limits for pseudospectrum + stieltjes panels
    xlim, ylim = _compute_limits_from_all_steps(W_snaps, instance_id, steps, which, pad)

    # Precompute Gram eigenvalues over time (fast + stable for symmetric PSD)
    eigs_per_step: List[np.ndarray] = []
    for step in steps:
        Wi = W_snaps["steps"][step][instance_id]
        Gf, Gh = grams_from_Wi(Wi)
        A = Gf if which == "feature" else Gh
        vals = np.linalg.eigvalsh(A)  # real, sorted ascending
        eigs_per_step.append(vals)

    eigs_sorted_over_time = np.stack([np.sort(v) for v in eigs_per_step], axis=0)  # (T, n)

    st_grid = int(stieltjes_grid) if stieltjes_grid is not None else int(grid_size)

    # Optionally fix the heatmap color scale across frames (less "flicker")
    vmin = vmax = None
    if stieltjes_fix_scale:
        # Compute a few frames worth to estimate scale cheaply:
        # here we compute on ALL frames for correctness; if too slow, sub-sample steps.
        mins, maxs = [], []
        for t in range(len(steps)):
            Zplot = stieltjes_heatmap_from_eigs(
                eigs_sorted_over_time[t], xlim, ylim, grid=st_grid, mode=stieltjes_mode
            )
            mins.append(float(np.nanmin(Zplot)))
            maxs.append(float(np.nanmax(Zplot)))
        vmin = float(np.min(mins))
        vmax = float(np.max(maxs))

    frames = []
    for idx, step in enumerate(steps):
        step_start = time.time()
        Wi = W_snaps["steps"][step][instance_id]
        Gf, Gh = grams_from_Wi(Wi)
        A = Gf if which == "feature" else Gh
        print(f"[Instance {instance_id}] Step {step}: matrix shape {A.shape}", flush=True)

        # Create side-by-side panels
        if layout == "2panel":
            fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
            ax_ps, ax_st = axs
            ax_tr = None
        else:
            fig, axs = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
            ax_ps, ax_st, ax_tr = axs

        # Panel 1: pseudospectrum
        generate_pseudospectrum(
            A,
            step=step,
            pad=pad,
            xlim=xlim,
            ylim=ylim,
            ax=ax_ps,
        )

        # Panel 2: Stieltjes heatmap
        im = plot_stieltjes_heatmap(
            ax_st,
            eigs_sorted_over_time[idx],
            xlim,
            ylim,
            grid=st_grid,
            mode=stieltjes_mode,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(im, ax=ax_st, fraction=0.046, pad=0.04)

        # Panel 3: eigenvalue trajectories (real axis)
        if ax_tr is not None:
            plot_eig_trajectories(ax_tr, eigs_sorted_over_time, frame_idx=idx)

        fig.suptitle(f"Instance {instance_id} | {which} Gram | step={step}", fontsize=12)

        frames.append(_fig_to_rgb_array(fig))
        plt.close(fig)

        step_time = time.time() - step_start
        print(
            f"[Instance {instance_id}] Step {step} completed in {step_time:.2f}s ({idx+1}/{len(steps)})",
            flush=True,
        )

    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(
        out_dir,
        f"pseudospectrum_stieltjes_{layout}_instance_{instance_id}_{which}.gif",
    )
    save_start = time.time()
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    save_time = time.time() - save_start
    total_time = time.time() - start_time
    print(f"[Instance {instance_id}] Saved GIF in {save_time:.2f}s (total: {total_time:.2f}s)", flush=True)
    return gif_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", type=str, required=True, help="Path to W_snaps pickle (e.g., W_snaps.pkl)")
    ap.add_argument("--out_dir", type=str, default="pseudospectrum_gifs", help="Output directory for GIFs")
    ap.add_argument("--which", type=str, default="feature", choices=["feature", "hidden"])
    ap.add_argument("--grid_size", type=int, default=250)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--pad", type=float, default=0.5)
    ap.add_argument("--layout", type=str, default="3panel", choices=["2panel", "3panel"],
                    help="2panel: (pseudospectrum|stieltjes), 3panel: (+ trajectories)")
    ap.add_argument("--stieltjes_mode", type=str, default="logabs", choices=["logabs", "abs", "im"],
                    help="Heatmap value to plot for m(z)")
    ap.add_argument("--stieltjes_grid", type=int, default=None,
                    help="Grid size for Stieltjes heatmap (default: --grid_size)")
    ap.add_argument("--stieltjes_fix_scale", action="store_true",
                    help="Fix heatmap color scale across frames (slower, less flicker)")

    ap.add_argument("--instances", type=str, default=None, help='e.g. "0,1,2"')
    ap.add_argument("--steps", type=str, default=None, help='e.g. "0,200,1000"')
    ap.add_argument("--max_workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")

    args = ap.parse_args()

    instances_arg = _parse_int_list(args.instances)
    steps_arg = _parse_int_list(args.steps)

    # Read once to resolve instances/steps
    with open(args.snaps, "rb") as f:
        W_snaps = pickle.load(f)

    instances = _resolve_instances(W_snaps, instances_arg)
    steps = _resolve_steps(W_snaps, steps_arg)

    import multiprocessing
    max_workers = args.max_workers if args.max_workers is not None else multiprocessing.cpu_count()

    print(f"Found {len(instances)} instances; {len(steps)} steps; max_workers={max_workers}")
    print(f"CPU count: {multiprocessing.cpu_count()}")
    print(f"Writing GIFs to: {args.out_dir}")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for i in instances:
            futures.append(ex.submit(
                generate_gif_for_instance,
                args.snaps,
                args.out_dir,
                i,
                args.which,
                args.grid_size,
                args.fps,
                args.pad,
                steps,
                args.layout,
                args.stieltjes_mode,
                args.stieltjes_grid,
                args.stieltjes_fix_scale,
            ))

        with tqdm(total=len(instances), desc="Generating GIFs", unit="instance") as pbar:
            for fut in as_completed(futures):
                try:
                    gif_path = fut.result()
                    pbar.set_postfix_str(f"✓ {os.path.basename(gif_path)}")
                    pbar.update(1)
                except Exception as e:
                    pbar.set_postfix_str(f"✗ Error: {str(e)[:50]}")
                    pbar.update(1)

    print("Done.")


if __name__ == "__main__":
    main()
