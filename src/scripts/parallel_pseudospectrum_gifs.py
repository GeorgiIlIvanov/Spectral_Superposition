#!/usr/bin/env python3
"""
Parallel GIF generation for pseudospectrum episodes (one GIF per instance),
supporting your PseudoPy "fast" mode.

Usage:
  python parallel_pseudospectrum_gifs_fast.py --snaps W_snaps.pkl --out_dir pseudospectrum_gifs --which feature --epsilon 1e-2 --grid_size 250 --fps 3 --fast --max_workers 8

Key points for FAST mode:
- We DO NOT call pyplot.show().
- We render using Matplotlib "Agg" backend and capture the figure buffer.
- We force PseudoPy to draw into a specific Axes by setting it as current via plt.sca(ax).
  (PseudoPy's plot() uses pyplot state, as shown in the project's examples.) citeturn0search3
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
    # Use buffer_rgba() for newer matplotlib versions (replaces deprecated tostring_rgb())
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    # Convert RGBA to RGB by dropping alpha channel
    buf = buf.reshape((h, w, 4))[:, :, :3]
    return buf


def generate_pseudospectrum(
    weights,
    epsilon,
    step=None,
    grid_size=250,
    pad=0.5,
    xlim=None,
    ylim=None,
    show_field=True,
    log_scale=True,
    ax=None,
    fast=True,
):
    """
    Your function, adapted so that BOTH branches:
      - draw into a supplied ax (or create one),
      - NEVER call pyplot.show(),
      - return (fig, ax, info) for GIF capture.
    """
    A = np.asarray(weights)
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "weights must be a square matrix"
    assert epsilon > 0, "epsilon must be > 0"

    eigs = np.linalg.eigvals(A)

    # Infer plot window if not provided
    if xlim is None or ylim is None:
        re = eigs.real
        im = eigs.imag
        xmin, xmax = re.min() - pad, re.max() + pad
        ymin, ymax = im.min() - pad, im.max() + pad
        if np.isclose(xmin, xmax): xmin, xmax = xmin - 1.0, xmax + 1.0
        if np.isclose(ymin, ymax): ymin, ymax = ymin - 1.0, ymax + 1.0
        if xlim is None: xlim = (xmin, xmax)
        if ylim is None: ylim = (ymin, ymax)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure

    # Clear any prior artists on this ax (important when reusing axes)
    ax.cla()

    if not fast:
        # ---- slow grid evaluation method (your original) ----
        n = A.shape[0]
        xs = np.linspace(xlim[0], xlim[1], grid_size)
        ys = np.linspace(ylim[0], ylim[1], grid_size)
        X, Y = np.meshgrid(xs, ys)
        Z = X + 1j * Y

        I = np.eye(n, dtype=complex)
        resolvent_norm = np.empty(Z.shape, dtype=float)

        for i in range(grid_size):
            for j in range(grid_size):
                M = (Z[i, j] * I) - A
                s = np.linalg.svd(M, compute_uv=False)
                smin = s[-1]
                resolvent_norm[i, j] = np.inf if smin == 0 else (1.0 / smin)

        level = 1.0 / epsilon
        if show_field:
            field = np.log10(resolvent_norm) if log_scale else resolvent_norm
            imh = ax.imshow(
                field,
                origin="lower",
                extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                aspect="auto",
            )
            cbar = fig.colorbar(imh, ax=ax, shrink=0.9)
            cbar.set_label("log10 ||(zI-A)^(-1)||" if log_scale else "||(zI-A)^(-1)||")

        cs = ax.contour(X, Y, resolvent_norm, levels=[level], linewidths=2)
        ax.clabel(cs, inline=True, fontsize=9, fmt={level: f"1/ε = {level:.2g}"})

        ax.scatter(eigs.real, eigs.imag, s=30, marker="x", label="eigs")
        ax.set_title(f"ε-pseudospectrum level set (ε={epsilon:g}) step={step}")
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.legend(loc="best")

        info = dict(level=level, xlim=xlim, ylim=ylim, eigenvalues=eigs)
        return fig, ax, info

    # ---- fast PseudoPy method ----
    # PseudoPy example usage is pseudo.plot(...); pyplot.show(). We omit show() and capture fig. citeturn0search3
    try:
        from pseudopy import NonnormalAuto
        from scipy.linalg import eigvals as scipy_eigvals
    except Exception as e:
        raise RuntimeError(
            "fast=True requires pseudopy and scipy to be installed in this environment."
        ) from e

    # Set axis limits BEFORE plotting so PseudoPy respects them
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    # Ensure PseudoPy draws onto *this* axes (it uses pyplot state).
    plt.sca(ax)

    # Create a fresh NonnormalAuto instance for each step to avoid caching issues
    pseudo = NonnormalAuto(A, 1e-5, 1)
    levels = [10**k for k in range(-4, 0)]
    spectrum = scipy_eigvals(A)
    pseudo.plot(levels, spectrum=spectrum)

    # Post-formatting to keep your titles/limits consistent
    ax.set_title(f"ε-pseudospectrum (PseudoPy fast) step={step}")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    # Re-apply limits after plotting (PseudoPy might change them)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    info = dict(levels=levels, xlim=xlim, ylim=ylim, eigenvalues=eigs)
    return fig, ax, info


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
        # Default: process all instances instead of just 3
        return list(range(n_instances))
    return [i for i in instances_arg if 0 <= i < n_instances]


def _compute_limits_from_all_steps(W_snaps, instance_id, steps, which, pad):
    """Compute global limits across all steps for consistent GIF framing."""
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
    if np.isclose(xlim[0], xlim[1]): xlim = (xlim[0] - 1.0, xlim[1] + 1.0)
    if np.isclose(ylim[0], ylim[1]): ylim = (ylim[0] - 1.0, ylim[1] + 1.0)
    return xlim, ylim


def generate_gif_for_instance(
    snaps_path: str,
    out_dir: str,
    instance_id: int,
    which: str,
    epsilon: float,
    grid_size: int,
    fps: int,
    pad: float,
    steps_list,
    fast: bool,
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

    # Compute global limits across all steps for consistent framing
    xlim, ylim = _compute_limits_from_all_steps(W_snaps, instance_id, steps, which, pad)

    frames = []
    for idx, step in enumerate(steps):
        step_start = time.time()
        Wi = W_snaps["steps"][step][instance_id]
        Gf, Gh = grams_from_Wi(Wi)
        A = Gf if which == "feature" else Gh
        print(f"[Instance {instance_id}] Step {step}: matrix shape {A.shape}", flush=True)

        fig, ax, _ = generate_pseudospectrum(
            A,
            epsilon=epsilon,
            step=step,
            grid_size=grid_size,
            pad=pad,
            xlim=xlim,
            ylim=ylim,
            fast=fast,
        )
        frames.append(_fig_to_rgb_array(fig))
        plt.close(fig)
        step_time = time.time() - step_start
        print(f"[Instance {instance_id}] Step {step} completed in {step_time:.2f}s ({idx+1}/{len(steps)})", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, f"pseudospectrum_instance_{instance_id}_{which}.gif")
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
    ap.add_argument("--epsilon", type=float, default=1e-2)
    ap.add_argument("--grid_size", type=int, default=250)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--pad", type=float, default=0.5)
    ap.add_argument("--fast", action="store_true", help="Use PseudoPy fast mode (requires pseudopy + scipy)")
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

    # Use CPU count if max_workers not specified
    import multiprocessing
    max_workers = args.max_workers if args.max_workers is not None else multiprocessing.cpu_count()
    
    print(f"Found {len(instances)} instances; {len(steps)} steps; fast={args.fast}; max_workers={max_workers}")
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
                args.epsilon,
                args.grid_size,
                args.fps,
                args.pad,
                steps,
                args.fast,
            ))

        # Use tqdm to track instance completion
        with tqdm(total=len(instances), desc="Generating GIFs", unit="instance") as pbar:
            for fut in as_completed(futures):
                try:
                    gif_path = fut.result()
                    pbar.set_postfix_str(f"✓ {gif_path.split('/')[-1]}")
                    pbar.update(1)
                except Exception as e:
                    pbar.set_postfix_str(f"✗ Error: {str(e)[:50]}")
                    pbar.update(1)

    print("Done.")


if __name__ == "__main__":
    main()
