import argparse, math, os, sys, time, json, hashlib, functools, itertools
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    torch = None 

def set_torch_defaults(float64=True, deterministic=False):
    if not TORCH_AVAILABLE:
        return
    if float64:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

def select_device(local_rank: Optional[int] = None):
    if not TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        if local_rank is not None:
            torch.cuda.set_device(local_rank)
            return f"cuda:{local_rank}"
        return "cuda"
    return "cpu"

def seed_everything(seed: int):
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        np.random.seed(seed)

def as_tensor(x, device):
    if TORCH_AVAILABLE:
        return torch.as_tensor(x, device=device)
    raise RuntimeError("PyTorch not available. Please install torch+cuda to run.")


@dataclass
class TrainConfig:
    n: int = 16
    steps: int = 500
    eta0: float = 1e-2
    eta1: float = 1e-2
    alpha0: Optional[float] = None  
    alpha1: Optional[float] = None  
    dtype64: bool = True
    loss_ceiling: float = 1e12     
    grad_ceiling: float = 1e12
    nan_infinity_is_divergence: bool = True
    record_every: int = 1

@dataclass
class Dataset:
    X: "torch.Tensor"  
    y: "torch.Tensor"  

@dataclass
class RunResult:
    diverged: bool
    blue_sum: float   
    red_sum: float    
    final_loss: float
    steps_done: int
    mean_frac_dim: Optional[float] 

def make_dataset(n: int, device, seed: int) -> Dataset:
    N = n * n + n
    X = torch.randn(N, n, device=device)
    y = torch.randn(N, device=device)
    return Dataset(X=X, y=y)

def init_weights(n: int, device, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    W0 = torch.randn(n, n, generator=g, device=device)
    W1 = torch.randn(1, n, generator=g, device=device)
    return W0, W1

def compute_fractional_dimensionality(W0: "torch.Tensor") -> float:
    Wi = W0  
    G = Wi.T @ Wi  
    diag = torch.diag(G)  
    num = diag ** 2       
    denom = (G ** 2).sum(dim=1)  
    D = num / (denom + 1e-30)
    return float(D.mean().item())

def train_single_run(config: TrainConfig, data: Dataset, W0_init, W1_init, device) -> RunResult:
    n = config.n
    alpha0 = math.sqrt(2.0 / n) if config.alpha0 is None else config.alpha0
    alpha1 = 1.0 / n if config.alpha1 is None else config.alpha1

    W0 = W0_init.clone().detach().to(device).requires_grad_(True)
    W1 = W1_init.clone().detach().to(device).requires_grad_(True)

    X, y = data.X, data.y
    blue_sum = 0.0
    red_sum = 0.0
    diverged = False
    steps_done = 0

    for t in range(config.steps):
        Hpre = alpha0 * (X @ W0.T)          
        H = torch.relu(Hpre)                
        yhat = alpha1 * (H @ W1.T).squeeze(1)
        loss = torch.mean((y - yhat) ** 2)

        if not torch.isfinite(loss):
            diverged = True
            break

        blue_sum += float(loss.item())
        red_sum  += float(1.0 / (loss.item() + 1e-30))

        if loss.item() > config.loss_ceiling:
            diverged = True
            break

        W0.grad = None
        W1.grad = None
        loss.backward()

        if config.grad_ceiling is not None:
            g0 = torch.nan_to_num(W0.grad, nan=1e30, posinf=1e30, neginf=-1e30)
            g1 = torch.nan_to_num(W1.grad, nan=1e30, posinf=1e30, neginf=-1e30)
            if (g0.abs().max() > config.grad_ceiling) or (g1.abs().max() > config.grad_ceiling):
                diverged = True
                break

        with torch.no_grad():
            W0 -= config.eta0 * W0.grad
            W1 -= config.eta1 * W1.grad

        steps_done += 1

    mean_frac_dim = None
    if not diverged:
        try:
            with torch.no_grad():
                mean_frac_dim = compute_fractional_dimensionality(W0)
        except Exception:
            mean_frac_dim = None

    return RunResult(
        diverged=diverged,
        blue_sum=blue_sum,
        red_sum=red_sum,
        final_loss=float(loss.item()) if 'loss' in locals() and torch.is_tensor(loss) else float('inf'),
        steps_done=steps_done,
        mean_frac_dim=mean_frac_dim
    )


def _lin_or_log_space(vmin, vmax, steps, logspace: bool):
    if logspace:
        return np.geomspace(vmin, vmax, steps)
    else:
        return np.linspace(vmin, vmax, steps)

def evaluate_grid(n: int,
                  steps: int,
                  eta0_range: Tuple[float, float],
                  eta1_range: Tuple[float, float],
                  grid: Tuple[int, int],
                  log_eta0: bool = True,
                  log_eta1: bool = True,
                  seed: int = 123,
                  dtype64: bool = True,
                  device = "cuda",
                  record_fractional_dim: bool = True) -> Dict[str, np.ndarray]:
    """
    Evaluate a 2D grid of learning-rate pairs.
    Returns a dict of numpy arrays: diverged_mask, blue, red, final_loss, frac_dim (optional).
    """
    set_torch_defaults(float64=dtype64)
    seed_everything(seed)

    data = make_dataset(n=n, device=device, seed=seed + 17)
    W0_init, W1_init = init_weights(n=n, device=device, seed=seed + 1234)

    H, W = grid
    eta0_vals = _lin_or_log_space(eta0_range[0], eta0_range[1], W, log_eta0)
    eta1_vals = _lin_or_log_space(eta1_range[0], eta1_range[1], H, log_eta1)

    diverged = np.zeros((H, W), dtype=bool)
    blue     = np.zeros((H, W), dtype=np.float64)
    red      = np.zeros((H, W), dtype=np.float64)
    finalL   = np.zeros((H, W), dtype=np.float64)
    fdim     = np.full((H, W), np.nan, dtype=np.float64)

    for i, eta1 in enumerate(eta1_vals):
        for j, eta0 in enumerate(eta0_vals):
            cfg = TrainConfig(n=n, steps=steps, eta0=float(eta0), eta1=float(eta1), dtype64=dtype64)
            rr = train_single_run(cfg, data, W0_init, W1_init, device)

            diverged[i, j] = rr.diverged
            blue[i, j]     = rr.blue_sum
            red[i, j]      = rr.red_sum
            finalL[i, j]   = rr.final_loss
            if record_fractional_dim and (rr.mean_frac_dim is not None):
                fdim[i, j] = rr.mean_frac_dim

    return dict(
        eta0_vals=eta0_vals,
        eta1_vals=eta1_vals,
        diverged=diverged,
        blue=blue,
        red=red,
        final_loss=finalL,
        frac_dim=fdim
    )


def _normalize_channel(x: np.ndarray, mask: np.ndarray, eps: float = 1e-12):

    x = x.copy()
    out = np.zeros_like(x, dtype=np.float64)
    if mask.sum() > 0:
        vals = x[mask]
        p99 = np.percentile(vals, 99.0)
        if p99 < eps:
            p99 = float(vals.max() + eps)
        out[mask] = np.clip(vals / (p99 + eps), 0.0, 1.0)
    return out

def compose_rgb(diverged: np.ndarray, blue: np.ndarray, red: np.ndarray) -> np.ndarray:
    converged_mask = ~diverged
    red_norm  = _normalize_channel(red, diverged)
    blue_norm = _normalize_channel(blue, converged_mask)

    
    R = red_norm
    G = np.zeros_like(R)
    B = blue_norm
    rgb = np.stack([R, G, B], axis=-1)
    return rgb

def save_rgb_image(path: str, rgb: np.ndarray, dpi: int = 100):
    import matplotlib.pyplot as plt
    H, W, _ = rgb.shape
    fig = plt.figure(figsize=(W/ dpi, H/ dpi), dpi=dpi)
    ax = plt.axes([0,0,1,1])
    ax.imshow(rgb, origin='lower', interpolation='nearest')
    ax.set_axis_off()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _grid_indices(num_points: int, world_size: int, rank: int):

    return list(range(rank, num_points, world_size))

def evaluate_grid_distributed(n: int,
                              steps: int,
                              eta0_range: Tuple[float, float],
                              eta1_range: Tuple[float, float],
                              grid: Tuple[int, int],
                              log_eta0: bool = True,
                              log_eta1: bool = True,
                              seed: int = 123,
                              dtype64: bool = True,
                              out_path: Optional[str] = None):
    """
    Multi-GPU evaluation via torch.distributed. Launch via:
    torchrun --standalone --nproc_per_node=8 fractal_trainability.py --ddp ... [args]
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for distributed execution.")


    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = select_device(rank)

    set_torch_defaults(float64=dtype64)
    seed_everything(seed + rank)


    data = make_dataset(n=n, device=device, seed=seed + 17)
    W0_init, W1_init = init_weights(n=n, device=device, seed=seed + 1234)

    H, W = grid
    eta0_vals = _lin_or_log_space(eta0_range[0], eta0_range[1], W, log_eta0)
    eta1_vals = _lin_or_log_space(eta1_range[0], eta1_range[1], H, log_eta1)


    coords = [(i, j) for i in range(H) for j in range(W)]
    my_indices = _grid_indices(len(coords), world, rank)

    diverged = np.zeros((H, W), dtype=bool)
    blue     = np.zeros((H, W), dtype=np.float64)
    red      = np.zeros((H, W), dtype=np.float64)
    finalL   = np.zeros((H, W), dtype=np.float64)
    fdim     = np.full((H, W), np.nan, dtype=np.float64)


    for idx in my_indices:
        i, j = coords[idx]
        eta1 = eta1_vals[i]
        eta0 = eta0_vals[j]
        cfg = TrainConfig(n=n, steps=steps, eta0=float(eta0), eta1=float(eta1), dtype64=dtype64)
        rr = train_single_run(cfg, data, W0_init, W1_init, device)

        diverged[i, j] = rr.diverged
        blue[i, j]     = rr.blue_sum
        red[i, j]      = rr.red_sum
        finalL[i, j]   = rr.final_loss
        if rr.mean_frac_dim is not None:
            fdim[i, j] = rr.mean_frac_dim


    tensors = []
    for arr, kind in [(diverged.astype(np.int64), 'i'),
                      (blue, 'f'),
                      (red, 'f'),
                      (finalL, 'f'),
                      (np.nan_to_num(fdim, nan=0.0), 'f'),
                      (np.isnan(fdim).astype(np.int64), 'i')]:
        t = torch.from_numpy(arr).to(device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        tensors.append(t.cpu().numpy())

    diverged_sum_i = tensors[0]
    blue_sum       = tensors[1]
    red_sum        = tensors[2]
    finalL_sum     = tensors[3]
    fdim_sum       = tensors[4]
    fdim_nan_count = tensors[5]


    diverged_out = (diverged_sum_i > 0).astype(bool)
    blue_out     = blue_sum
    red_out      = red_sum
    finalL_out   = finalL_sum

    denom = np.maximum(world - fdim_nan_count, 1)
    fdim_out = fdim_sum / denom

    if rank == 0 and out_path is not None:
        payload = dict(
            n=n, steps=steps, eta0_vals=eta0_vals, eta1_vals=eta1_vals,
            diverged=diverged_out, blue=blue_out, red=red_out, final_loss=finalL_out, frac_dim=fdim_out
        )
        with open(out_path, "wb") as f:
            np.savez_compressed(f, **payload)
        print(f"[rank 0] wrote {out_path}")

    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description="Fractal trainability map generator")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--eta0-min", type=float, default=1e-3)
    p.add_argument("--eta0-max", type=float, default=1e+1)
    p.add_argument("--eta1-min", type=float, default=1e-3)
    p.add_argument("--eta1-max", type=float, default=1e+3)
    p.add_argument("--width", type=int, default=1024, help="grid columns (eta0 axis)")
    p.add_argument("--height", type=int, default=1024, help="grid rows (eta1 axis)")
    p.add_argument("--log-eta0", action="store_true", default=True)
    p.add_argument("--log-eta1", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--float32", action="store_true", help="use float32 (default float64)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="fractal_grid.npz")
    p.add_argument("--render", type=str, default="fractal_grid.png")
    p.add_argument("--ddp", action="store_true", help="use torch.distributed across GPUs")
    args = p.parse_args()

    dtype64 = not args.float32

    if args.ddp:
        evaluate_grid_distributed(n=args.n,
                                  steps=args.steps,
                                  eta0_range=(args.eta0_min, args.eta0_max),
                                  eta1_range=(args.eta1_min, args.eta1_max),
                                  grid=(args.height, args.width),
                                  log_eta0=args.log_eta0,
                                  log_eta1=args.log_eta1,
                                  seed=args.seed,
                                  dtype64=dtype64,
                                  out_path=args.out)
    else:
        device = select_device(None if args.device == "cuda" else None) if args.device == "cuda" else args.device
        res = evaluate_grid(n=args.n,
                            steps=args.steps,
                            eta0_range=(args.eta0_min, args.eta0_max),
                            eta1_range=(args.eta1_min, args.eta1_max),
                            grid=(args.height, args.width),
                            log_eta0=args.log_eta0,
                            log_eta1=args.log_eta1,
                            seed=args.seed,
                            dtype64=dtype64,
                            device=device,
                            record_fractional_dim=True)
        # Save grid
        with open(args.out, "wb") as f:
            np.savez_compressed(f, **res)
        # Render RGB
        rgb = compose_rgb(res["diverged"], res["blue"], res["red"])
        save_rgb_image(args.render, rgb, dpi=100)
        print(f"Wrote {args.out} and {args.render}")

if __name__ == "__main__":
    main()
