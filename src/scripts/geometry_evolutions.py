#!/usr/bin/env python3
"""
Parallel GIF generation for pseudospectrum episodes (one GIF per instance),
with evolving side-by-side panels:

Panel order (by layout number):
  [1] Feature geometry graph
  [2] + ε-pseudospectrum (PseudoPy fast)   (currently on Gram as before)
  [3] + Stieltjes transform heatmap (2D)   (on Gram eigenvalues as before)
  [4] + Eigenvalue trajectories            (on Gram eigenvalues as before)
  [5] + Weight geometry (2D vectors from origin; uses W directly)

Usage examples:
    # 5 panels (geometry + 4 spectral panels)
    python src/scripts/geometry_evolutions.py \
    --train --train_steps 10000 --n_features 5 --n_hidden 2 --n_instances 10 \
    --out_dir hf_gifs --which feature --layout 5panel --fps 3 --feature_graph_k 5


  # 4 panels (feature graph + 3 spectral panels)
  python src/scripts/geometry_evolutions.py \
    --snaps public/checkpoints/W_snaps_granular_hidden_dim_2.pkl \
    --out_dir gifs --which feature \
    --grid_size 250 --stieltjes_grid 128 --stieltjes_mode logabs \
    --layout 4panel --fps 3 --max_workers 8

  # Train + save snapshots + generate GIFs
  python src/scripts/geometry_evolutions.py \
    --train --train_steps 10000 --n_features 50 --n_hidden 10 --n_instances 10 \
    --out_dir gifs --which feature --layout 4panel --fps 3

Notes:
- The geometry panel uses W directly and assumes hidden_dim >= 2.
- The pseudospectrum/Stieltjes/trajectory panels are still computed from the Gram matrix
  as in your current script (because PseudoPy expects square matrices).
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
from matplotlib import collections as mc

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


# -----------------------------
# Panel 1: weight geometry (W directly)
# -----------------------------
def plot_weight_geometry(
    ax,
    Wi: np.ndarray,
    colors: np.ndarray,
    z: float = 1.5,
    facecolor: str = "#FCFBF8",
):
    """
    Reproduces your "intro diagram" style for a single instance:
    - scatter W[:,0], W[:,1] with consistent colors
    - line segments from origin to each point
    - centered axes spines, minimal ticks

    Wi: [n_features, n_hidden], requires n_hidden >= 2
    colors: [n_features, 4] RGBA
    """
    ax.cla()

    W = np.asarray(Wi)
    if W.ndim != 2 or W.shape[1] < 2:
        raise ValueError(f"Weight geometry requires Wi with shape [n_features, n_hidden>=2], got {W.shape}")

    if W.shape[1] > 2:
        # Project to 2D via PCA (SVD) for higher-dimensional features.
        Wc = W - W.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(Wc, full_matrices=False)
        basis = vh[:2].T  # [n_hidden, 2]
        pts = Wc @ basis
    else:
        pts = W[:, :2]
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=18)

    # Line segments from origin to each point
    segs = np.stack((np.zeros_like(pts), pts), axis=1)  # (n, 2, 2)
    ax.add_collection(mc.LineCollection(segs, colors=colors, linewidths=1.0, alpha=0.9))

    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(facecolor)
    ax.set_xlim((-z, z))
    ax.set_ylim((-z, z))

    # Match your styling
    ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_position("center")


# -----------------------------
# Panel 5: feature geometry graph (k-NN)
# -----------------------------
def _feature_graph_knn_coords(Wi: np.ndarray, k: int):
    """
    Build a symmetric k-NN graph using abs dot-product similarity on L2-normalized features,
    then compute a 2D spectral embedding from the graph Laplacian.
    Returns coords (n, 2) and adjacency matrix A (n, n).
    """
    W = np.asarray(Wi)
    if W.ndim != 2:
        raise ValueError(f"Expected Wi as 2D array [n_features, n_hidden], got {W.shape}")

    n = W.shape[0]
    if n <= 2:
        # Degenerate case: place points on a line
        coords = np.zeros((n, 2), dtype=float)
        coords[:, 0] = np.arange(n, dtype=float)
        A = np.zeros((n, n), dtype=float)
        return coords, A

    k = max(1, min(int(k), n - 1))
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    Wn = W / norms

    sim = np.abs(Wn @ Wn.T)
    np.fill_diagonal(sim, 0.0)

    A = np.zeros_like(sim)
    for i in range(n):
        nn = np.argpartition(sim[i], -k)[-k:]
        A[i, nn] = sim[i, nn]

    # Symmetrize: keep edge if either node selects the other
    A = np.maximum(A, A.T)

    deg = A.sum(axis=1)
    L = np.diag(deg) - A

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        order = np.argsort(eigvals)
        eigvecs = eigvecs[:, order]
        if eigvecs.shape[1] >= 3:
            coords = eigvecs[:, 1:3]
        else:
            coords = eigvecs[:, 1:2]
            coords = np.hstack([coords, np.zeros((n, 1))])
    except np.linalg.LinAlgError:
        # Fallback: use first two dimensions of Wn
        coords = Wn[:, :2] if Wn.shape[1] >= 2 else np.hstack([Wn, np.zeros((n, 1))])

    return coords, A


def plot_feature_graph(ax, Wi: np.ndarray, k: int, node_colors: np.ndarray):
    """
    Plot a symmetric k-NN feature graph using abs dot-product similarity on L2-normalized features.
    Nodes are positioned via spectral embedding of the graph Laplacian.
    """
    ax.cla()
    coords, A = _feature_graph_knn_coords(Wi, k=k)
    n = coords.shape[0]

    # Draw edges with alpha scaled by weight
    if n > 1 and np.any(A > 0):
        max_w = float(np.max(A))
        if max_w <= 0:
            max_w = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                w = A[i, j]
                if w <= 0:
                    continue
                alpha = 0.1 + 0.8 * (w / max_w)
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color="#666666",
                    linewidth=0.8,
                    alpha=alpha,
                )

    ax.scatter(coords[:, 0], coords[:, 1], c=node_colors, s=18, zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# -----------------------------
# Panel 2: pseudospectrum (square matrix)
# -----------------------------
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

    ax.set_title(f"ε-pseudospectrum (PseudoPy fast)\nstep={step}")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    info = dict(levels=levels, xlim=xlim, ylim=ylim, eigenvalues=eigs)
    return fig, ax, info


# -----------------------------
# Panel 3: Stieltjes heatmap (from eigenvalues)
# -----------------------------
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


# -----------------------------
# Panel 4: eigenvalue trajectories
# -----------------------------
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


def generate_granular_snapshots(max_steps: int = 10000):
    """
    Generate a granular snapshot schedule with more frequent snapshots early in training.
    """
    steps = set()

    # Every step from 0-50
    steps.update(range(0, min(51, max_steps + 1)))

    # Every 5 steps from 50-200
    if max_steps > 50:
        steps.update(range(50, min(201, max_steps + 1), 5))

    # Every 10 steps from 200-500
    if max_steps > 200:
        steps.update(range(200, min(501, max_steps + 1), 10))

    # Every 25 steps from 500-1000
    if max_steps > 500:
        steps.update(range(500, min(1001, max_steps + 1), 25))

    # Every 50 steps from 1000-2000
    if max_steps > 1000:
        steps.update(range(1000, min(2001, max_steps + 1), 50))

    # Every 100 steps from 2000-5000
    if max_steps > 2000:
        steps.update(range(2000, min(5001, max_steps + 1), 100))

    # Every 500 steps from 5000 onwards
    if max_steps > 5000:
        steps.update(range(5000, max_steps + 1, 500))

    return sorted(steps)


def train_and_save_snaps(args) -> str:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        import einops
        from dataclasses import dataclass
        from tqdm import trange
    except Exception as exc:
        raise RuntimeError(
            "Training requires torch + einops to be installed in this environment."
        ) from exc

    @dataclass
    class Config:
        n_features: int
        n_hidden: int
        n_instances: int

    class Model(nn.Module):
        def __init__(
            self,
            config,
            feature_probability: Optional[torch.Tensor] = None,
            importance: Optional[torch.Tensor] = None,
            device: str = "cuda",
        ):
            super().__init__()
            self.config = config
            self.W = nn.Parameter(
                torch.empty((config.n_instances, config.n_features, config.n_hidden), device=device)
            )
            nn.init.xavier_normal_(self.W)
            self.b_final = nn.Parameter(torch.zeros((config.n_instances, config.n_features), device=device))

            if feature_probability is None:
                feature_probability = torch.ones(())
            self.feature_probability = feature_probability.to(device)
            if importance is None:
                importance = torch.ones(1, config.n_features)
            self.importance = importance.to(device)

        def forward(self, features):
            # features: [..., instance, n_features]
            # W: [instance, n_features, n_hidden]
            hidden = torch.einsum("...if,ifh->...ih", features, self.W)
            out = torch.einsum("...ih,ifh->...if", hidden, self.W)
            out = out + self.b_final
            out = F.relu(out)
            return out

        def generate_batch(self, n_batch: int):
            feat = torch.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device)
            batch = torch.where(
                torch.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device)
                <= self.feature_probability,
                feat,
                torch.zeros((), device=self.W.device),
            )
            return batch

    def constant_lr(*_):
        return 1.0

    def linear_lr(step, steps):
        return (1 - (step / steps))

    def cosine_decay_lr(step, steps):
        return np.cos(0.5 * np.pi * step / (steps - 1))

    def optimize(
        model,
        n_batch: int = 1024,
        steps: int = 10_000,
        print_freq: int = 100,
        lr: float = 1e-3,
        lr_scale=constant_lr,
        hooks=None,
        snapshot_steps=None,
    ):
        cfg = model.config

        opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

        if snapshot_steps is None:
            snapshot_steps = []
        if hooks is None:
            hooks = []

        W_snaps = {
            "feature_probability": model.feature_probability.detach().cpu().numpy().copy(),
            "importance": model.importance.detach().cpu().numpy().copy(),
            "steps": {},
        }

        with trange(steps) as t:
            for step in t:
                step_lr = lr * lr_scale(step, steps)
                for group in opt.param_groups:
                    group["lr"] = step_lr
                opt.zero_grad(set_to_none=True)
                batch = model.generate_batch(n_batch)
                out = model(batch)
                error = (model.importance * (batch.abs() - out) ** 2)
                loss = einops.reduce(error, "b i f -> i", "mean").sum()
                loss.backward()
                opt.step()
                if step in snapshot_steps:
                    W_snaps["steps"][step] = model.W.detach().cpu().numpy().copy()

                if hooks:
                    hook_data = dict(
                        model=model,
                        step=step,
                        opt=opt,
                        error=error,
                        loss=loss,
                        lr=step_lr,
                    )
                    for h in hooks:
                        h(hook_data)
                if step % print_freq == 0 or (step + 1 == steps):
                    t.set_postfix(
                        loss=loss.item() / cfg.n_instances,
                        lr=step_lr,
                    )
        return W_snaps

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = "cpu"

    cfg = Config(
        n_features=args.n_features,
        n_hidden=args.n_hidden,
        n_instances=args.n_instances,
    )

    feature_probability = (args.feature_prob_base ** -torch.linspace(0, 1, cfg.n_instances))[:, None]
    importance = torch.ones(1, cfg.n_features)

    model = Model(
        config=cfg,
        device=device,
        importance=importance,
        feature_probability=feature_probability,
    )

    lr_scale = {
        "constant": constant_lr,
        "linear": linear_lr,
        "cosine": cosine_decay_lr,
    }[args.lr_schedule]

    snapshot_steps = _parse_int_list(args.snapshot_steps)
    if snapshot_steps is None:
        snapshot_steps = generate_granular_snapshots(max_steps=args.train_steps)
    snapshot_steps = set(snapshot_steps)

    W_snaps = optimize(
        model,
        n_batch=args.batch_size,
        steps=args.train_steps,
        print_freq=args.print_freq,
        lr=args.lr,
        lr_scale=lr_scale,
        snapshot_steps=snapshot_steps,
    )

    W_snaps["config"] = dict(
        n_features=cfg.n_features,
        n_hidden=cfg.n_hidden,
        n_instances=cfg.n_instances,
    )

    snap_dir = args.snaps_dir
    if snap_dir is None:
        snap_dir = os.path.join("public", "checkpoints")
    os.makedirs(snap_dir, exist_ok=True)

    if args.snaps_out is not None:
        snap_path = args.snaps_out
    elif args.snaps is not None:
        snap_path = args.snaps
    else:
        snap_name = f"W_snaps_nf{cfg.n_features}_nh{cfg.n_hidden}_ni{cfg.n_instances}.pkl"
        snap_path = os.path.join(snap_dir, snap_name)

    with open(snap_path, "wb") as f:
        pickle.dump(W_snaps, f)

    print(f"Saved snapshots to: {snap_path}")
    return snap_path


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
    """Compute global limits across all steps for consistent spectral framing."""
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


def _compute_geometry_limits_from_all_steps(W_snaps, instance_id, steps, z_pad: float = 0.1):
    """
    Compute a global symmetric limit z for weight-geometry panel from W[:, :2] over all steps,
    so the geometry panel doesn't "jump" in scale.
    """
    max_abs = 0.0
    for step in steps:
        Wi = np.asarray(W_snaps["steps"][step][instance_id])
        if Wi.ndim != 2 or Wi.shape[1] < 2:
            raise ValueError(f"Weight geometry requires Wi with shape [n_features, n_hidden>=2], got {Wi.shape}")
        pts = Wi[:, :2]
        max_abs = max(max_abs, float(np.max(np.abs(pts))))
    z = max_abs * (1.0 + z_pad)
    if not np.isfinite(z) or z <= 0:
        z = 1.5
    return z


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
    geometry_facecolor: str,
    geometry_z: Optional[float],
    feature_graph_k: int,
    panels: List[str],
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

    # Global complex-plane limits for pseudospectrum + stieltjes panels (Gram-based)
    xlim, ylim = _compute_limits_from_all_steps(W_snaps, instance_id, steps, which, pad)

    # Determine consistent geometry zoom z (weight-based)
    if geometry_z is None:
        geometry_z = _compute_geometry_limits_from_all_steps(W_snaps, instance_id, steps, z_pad=0.15)

    # Precompute Gram eigenvalues over time (fast + stable for symmetric PSD)
    eigs_per_step: List[np.ndarray] = []
    for step in steps:
        Wi = W_snaps["steps"][step][instance_id]
        Gf, Gh = grams_from_Wi(Wi)
        A = Gf if which == "feature" else Gh
        vals = np.linalg.eigvalsh(A)
        eigs_per_step.append(vals)

    eigs_sorted_over_time = np.stack([np.sort(v) for v in eigs_per_step], axis=0)  # (T, n)

    st_grid = int(stieltjes_grid) if stieltjes_grid is not None else int(grid_size)

    # Fix Stieltjes scale across frames (optional)
    vmin = vmax = None
    if stieltjes_fix_scale:
        mins, maxs = [], []
        for t in range(len(steps)):
            Zplot = stieltjes_heatmap_from_eigs(
                eigs_sorted_over_time[t], xlim, ylim, grid=st_grid, mode=stieltjes_mode
            )
            mins.append(float(np.nanmin(Zplot)))
            maxs.append(float(np.nanmax(Zplot)))
        vmin = float(np.min(mins))
        vmax = float(np.max(maxs))

    # Consistent colors for geometry: one per feature (n_features)
    # (Matches your intent of consistent per-feature coloring without requiring model.importance.)
    W0 = np.asarray(W_snaps["steps"][steps[0]][instance_id])
    n_features = W0.shape[0]
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.0, 1.0, n_features))  # RGBA

    frames = []
    for idx, step in enumerate(steps):
        step_start = time.time()
        Wi = W_snaps["steps"][step][instance_id]
        Gf, Gh = grams_from_Wi(Wi)
        A = Gf if which == "feature" else Gh
        print(f"[Instance {instance_id}] Step {step}: Wi {np.asarray(Wi).shape} | Gram {A.shape}", flush=True)

        # Create panels
        n_panels = len(panels)
        fig, axs = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.2), constrained_layout=True)
        if n_panels == 1:
            axs = [axs]
        panel_axes = dict(zip(panels, axs))

        ax_graph = panel_axes.get("feature_graph")
        ax_ps = panel_axes.get("pseudospectrum")
        ax_st = panel_axes.get("stieltjes")
        ax_tr = panel_axes.get("trajectories")
        ax_geom = panel_axes.get("geometry")

        # Panel: weight geometry
        if ax_geom is not None:
            plot_weight_geometry(
                ax_geom,
                Wi=np.asarray(Wi),
                colors=colors,
                z=float(geometry_z),
                facecolor=geometry_facecolor,
            )
            ax_geom.set_title("Weight geometry (W[:, :2])")

        # Panel: pseudospectrum (Gram)
        if ax_ps is not None:
            generate_pseudospectrum(
                A,
                step=step,
                pad=pad,
                xlim=xlim,
                ylim=ylim,
                ax=ax_ps,
            )

        # Panel: Stieltjes heatmap (Gram eigenvalues)
        if ax_st is not None:
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

        # Panel: eigenvalue trajectories (Gram eigenvalues)
        if ax_tr is not None:
            plot_eig_trajectories(ax_tr, eigs_sorted_over_time, frame_idx=idx)

        # Panel: feature geometry graph (k-NN)
        if ax_graph is not None:
            plot_feature_graph(ax_graph, np.asarray(Wi), k=feature_graph_k, node_colors=colors)
            ax_graph.set_title(f"Feature graph (k={feature_graph_k})")

        fig.suptitle(f"Instance {instance_id} | {which} | step={step}", fontsize=12)

        frames.append(_fig_to_rgb_array(fig))
        plt.close(fig)

        step_time = time.time() - step_start
        print(
            f"[Instance {instance_id}] Step {step} completed in {step_time:.2f}s ({idx+1}/{len(steps)})",
            flush=True,
        )

    os.makedirs(out_dir, exist_ok=True)
    cfg = W_snaps.get("config", None)
    if cfg is not None:
        cfg_tag = f"nf{cfg.get('n_features','?')}_nh{cfg.get('n_hidden','?')}_ni{cfg.get('n_instances','?')}"
        name_prefix = f"{layout}_{cfg_tag}"
    else:
        name_prefix = f"{layout}"
    gif_path = os.path.join(
        out_dir,
        f"{name_prefix}_instance_{instance_id}_{which}.gif",
    )
    save_start = time.time()
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    save_time = time.time() - save_start
    total_time = time.time() - start_time
    print(f"[Instance {instance_id}] Saved GIF in {save_time:.2f}s (total: {total_time:.2f}s)", flush=True)
    return gif_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", type=str, default=None, help="Path to W_snaps pickle (e.g., W_snaps.pkl)")
    ap.add_argument("--out_dir", type=str, default="hf_gifs", help="Output directory for GIFs")
    ap.add_argument("--which", type=str, default="feature", choices=["feature", "hidden"])
    ap.add_argument("--grid_size", type=int, default=250)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--pad", type=float, default=0.5)

    ap.add_argument("--layout", type=str, default="4panel",
                    choices=["1panel", "2panel", "3panel", "4panel", "5panel"],
                    help="Panel count in the default order: feature_graph, pseudospectrum, "
                         "stieltjes, trajectories, geometry")
    ap.add_argument("--stieltjes_mode", type=str, default="logabs", choices=["logabs", "abs", "im"],
                    help="Heatmap value to plot for m(z)")
    ap.add_argument("--stieltjes_grid", type=int, default=None,
                    help="Grid size for Stieltjes heatmap (default: --grid_size)")
    ap.add_argument("--stieltjes_fix_scale", action="store_true",
                    help="Fix heatmap color scale across frames (slower, less flicker)")

    # Geometry panel controls
    ap.add_argument("--geometry_facecolor", type=str, default="#FCFBF8",
                    help="Background color for geometry panel")
    ap.add_argument("--geometry_z", type=float, default=None,
                    help="Half-range for geometry axes (default: auto from data over all steps)")
    ap.add_argument("--feature_graph_k", type=int, default=5,
                    help="k for symmetric k-NN feature graph (abs dot-product on normalized features)")
    ap.add_argument("--panels", type=str, default=None,
                    help=("Comma-separated panels to render, in order. Options: "
                          "feature_graph,pseudospectrum,stieltjes,trajectories,geometry"))

    ap.add_argument("--instances", type=str, default=None, help='e.g. "0,1,2"')
    ap.add_argument("--steps", type=str, default=None, help='e.g. "0,200,1000"')
    ap.add_argument("--max_workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")

    # Training options (to generate snapshots on the fly)
    ap.add_argument("--train", action="store_true",
                    help="Train a toy model and save W_snaps before generating GIFs")
    ap.add_argument("--snaps_out", type=str, default=None,
                    help="Output path for generated snapshots (overrides --snaps when --train)")
    ap.add_argument("--snaps_dir", type=str, default=None,
                    help="Directory for generated snapshots when --snaps_out is not set")
    ap.add_argument("--train_steps", type=int, default=10_000)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--print_freq", type=int, default=100)
    ap.add_argument("--lr_schedule", type=str, default="constant",
                    choices=["constant", "linear", "cosine"])
    ap.add_argument("--snapshot_steps", type=str, default=None,
                    help='Override snapshot steps, e.g. "0,200,1000" (default: granular schedule)')
    ap.add_argument("--n_features", type=int, default=50)
    ap.add_argument("--n_hidden", type=int, default=10)
    ap.add_argument("--n_instances", type=int, default=10)
    ap.add_argument("--feature_prob_base", type=float, default=20.0,
                    help="Base for feature probability schedule (1 to 1/base)")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = ap.parse_args()

    if args.train:
        args.snaps = train_and_save_snaps(args)

    if args.snaps is None:
        raise SystemExit("Missing --snaps (or use --train to generate snapshots).")

    instances_arg = _parse_int_list(args.instances)
    steps_arg = _parse_int_list(args.steps)

    # Read once to resolve instances/steps
    with open(args.snaps, "rb") as f:
        W_snaps = pickle.load(f)

    instances = _resolve_instances(W_snaps, instances_arg)
    steps = _resolve_steps(W_snaps, steps_arg)

    import multiprocessing
    max_workers = args.max_workers if args.max_workers is not None else multiprocessing.cpu_count()

    print(f"Found {len(instances)} instances; {len(steps)} steps; layout={args.layout}; max_workers={max_workers}")
    print(f"CPU count: {multiprocessing.cpu_count()}")
    print(f"Writing GIFs to: {args.out_dir}")

    default_panel_order = ["feature_graph", "pseudospectrum", "stieltjes", "trajectories", "geometry"]
    if args.panels is not None:
        panels = [p.strip() for p in args.panels.split(",") if p.strip()]
        invalid = [p for p in panels if p not in default_panel_order]
        if invalid:
            raise SystemExit(f"Unknown panels: {', '.join(invalid)}")
    else:
        n_panels = int(args.layout.replace("panel", ""))
        panels = default_panel_order[:n_panels]

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
                args.geometry_facecolor,
                args.geometry_z,
                args.feature_graph_k,
                panels,
            ))

        with tqdm(total=len(instances), desc="Generating GIFs", unit="instance") as pbar:
            for fut in as_completed(futures):
                try:
                    gif_path = fut.result()
                    pbar.set_postfix_str(f"✓ {os.path.basename(gif_path)}")
                    pbar.update(1)
                except Exception as e:
                    pbar.set_postfix_str(f"✗ Error: {str(e)[:80]}")
                    pbar.update(1)
    print("Done.")

if __name__ == "__main__":
    main()
