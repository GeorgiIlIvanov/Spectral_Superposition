import argparse, math, os, sys, time
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except Exception:
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

@dataclass
class DataSynth:
    X: Optional["torch.Tensor"] 
    I: Optional["torch.Tensor"]  
    S: float                     

def sample_dataset(n: int, N: int, S: float, device, seed: int) -> "torch.Tensor":
    """
    Draw N samples of sparse features: each coord 0 w.p. S else U[0,1].
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    mask = torch.bernoulli((1 - S) * torch.ones(N, n, device=device, generator=g))
    vals = torch.rand(N, n, device=device, generator=g)  # U[0,1]
    X = mask * vals
    return X


def relu(x):
    if TORCH_AVAILABLE:
        return torch.clamp(x, min=0.0)
    else:
        raise RuntimeError("Torch required")

def loss_sampled(W: "torch.Tensor", b: "torch.Tensor",
                 X: "torch.Tensor", I: Optional["torch.Tensor"]) -> "torch.Tensor":

    H = X @ W.t()            
    Xhat = relu(H @ W + b)   
    diff = X - Xhat
    if I is not None:
        diff = diff * I
    return torch.mean(diff.pow(2))

def loss_L1_analytic(W: "torch.Tensor", b: "torch.Tensor",
                     I: Optional["torch.Tensor"]) -> "torch.Tensor":

    m, n = W.shape
    G = W.t() @ W
    diag = torch.diag(G)  

    t1 = (1.0 - relu(diag + b)).pow(2)            
    if I is not None:
        t1 = I * t1
    t1_sum = t1.sum()
    Gj = G.t()  
    Bj = b.view(-1, 1)  
    M = relu(Gj + Bj).pow(2)
    M = M - torch.diag(torch.diag(M))  
    if I is not None:
        M = (I.view(-1, 1)) * M  
    t2_sum = M.sum()

    return t1_sum + t2_sum


@dataclass
class TrainConfig:
    n: int = 256   
    m: int = 32    
    S: float = 0.95 
    steps: int = 500
    eta_W: float = 1e-2
    eta_b: float = 1e-2
    mode: str = "analytic-L1"  # "analytic-L1"/ "sampled"
    N: int = 8192      
    seed: int = 123
    dtype64: bool = True
    loss_ceiling: float = 1e12
    grad_ceiling: float = 1e12
    param_ceiling: float = 1e6
    record_every: int = 1
    importance_uniform: bool = True

@dataclass
class RunResult:
    diverged: bool
    blue_sum: float
    red_sum: float
    final_loss: float
    steps_done: int
    D_mean: Optional[float]
    D_star: Optional[float]

def init_params(m: int, n: int, device, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    W = torch.randn(m, n, device=device, generator=g) * (1.0 / math.sqrt(m))
    b = torch.zeros(n, device=device)  #
    return W, b

def fractional_dimensionality(W: "torch.Tensor"):
    G = W.t() @ W    
    diag = torch.diag(G)
    num = diag.pow(2)
    denom = (G.pow(2)).sum(dim=1) + 1e-30
    D = num / denom
    D_mean = float(D.mean().item())
    D_star = float(W.shape[0] / (W.pow(2).sum().item() + 1e-30))  
    return D_mean, D_star

def train_single_run(cfg: TrainConfig, device, shared_init=None, shared_data=None) -> RunResult:
    if cfg.dtype64:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    if shared_init is not None:
        W0, b0 = shared_init
        W = W0.clone().detach().to(device).requires_grad_(True)
        b = b0.clone().detach().to(device).requires_grad_(True)
    else:
        W, b = init_params(cfg.m, cfg.n, device=device, seed=cfg.seed)
        W.requires_grad_(True); b.requires_grad_(True)

    if cfg.importance_uniform:
        I = torch.ones(cfg.n, device=device)
    else:
        g = torch.Generator(device=device); g.manual_seed(cfg.seed+999)
        I = torch.rand(cfg.n, generator=g, device=device)

    if cfg.mode == "sampled":
        X = shared_data if shared_data is not None else sample_dataset(cfg.n, cfg.N, cfg.S, device, seed=cfg.seed+17)
        data = DataSynth(X=X, I=I, S=cfg.S)
    else:
        data = DataSynth(X=None, I=I, S=cfg.S)

    blue_sum = 0.0
    red_sum  = 0.0
    diverged = False
    steps_done = 0

    for t in range(cfg.steps):
        if cfg.mode == "sampled":
            loss = loss_sampled(W, b, data.X, data.I)
        else:
            loss = loss_L1_analytic(W, b, data.I if not cfg.importance_uniform else None)

        if not torch.isfinite(loss):
            diverged = True
            break

        blue_sum += float(loss.item())
        red_sum  += float(1.0 / (loss.item() + 1e-30))

        if loss.item() > cfg.loss_ceiling:
            diverged = True
            break
        W.grad = None; b.grad = None
        loss.backward()

        with torch.no_grad():
            if cfg.grad_ceiling is not None:
                if (torch.nan_to_num(W.grad, nan=1e30, posinf=1e30, neginf=-1e30).abs().max() > cfg.grad_ceiling) or \
                   (torch.nan_to_num(b.grad, nan=1e30, posinf=1e30, neginf=-1e30).abs().max() > cfg.grad_ceiling):
                    diverged = True
                    break
            W -= cfg.eta_W * W.grad
            b -= cfg.eta_b * b.grad

            if W.abs().max() > cfg.param_ceiling or b.abs().max() > cfg.param_ceiling:
                diverged = True
                break

        steps_done += 1

    D_mean = None; D_star = None
    if not diverged:
        try:
            with torch.no_grad():
                D_mean, D_star = fractional_dimensionality(W)
        except Exception:
            D_mean, D_star = None, None

    return RunResult(diverged=diverged, blue_sum=blue_sum, red_sum=red_sum,
                     final_loss=float(loss.item()) if 'loss' in locals() and torch.is_tensor(loss) else float('inf'),
                     steps_done=steps_done, D_mean=D_mean, D_star=D_star)


def _lin_or_log_space(vmin, vmax, steps, logspace: bool):
    if logspace:
        return np.geomspace(vmin, vmax, steps)
    else:
        return np.linspace(vmin, vmax, steps)

def evaluate_grid(n: int, m: int, S: float, steps: int,
                  etaW_range: Tuple[float, float],
                  etaB_range: Tuple[float, float],
                  grid: Tuple[int, int],
                  mode: str = "analytic-L1",
                  log_etaW: bool = True,
                  log_etaB: bool = True,
                  seed: int = 123,
                  dtype64: bool = True,
                  device: str = "cuda",
                  record_fractional_dim: bool = True,
                  N: int = 8192,
                  importance_uniform: bool = True):
    """
    Evaluate a 2D grid of (eta_W, eta_b) for the Toy Models ReLU-output setup.
    """
    set_torch_defaults(float64=dtype64)
    seed_everything(seed)
    W0, b0 = init_params(m, n, device=device, seed=seed + 111)
    Xshared = sample_dataset(n, N, S, device, seed=seed + 222) if mode == "sampled" else None

    H, Ww = grid
    etaW_vals = _lin_or_log_space(etaW_range[0], etaW_range[1], Ww, log_etaW)
    etaB_vals = _lin_or_log_space(etaB_range[0], etaB_range[1], H,  log_etaB)

    diverged = np.zeros((H, Ww), dtype=bool)
    blue     = np.zeros((H, Ww), dtype=np.float64)
    red      = np.zeros((H, Ww), dtype=np.float64)
    finalL   = np.zeros((H, Ww), dtype=np.float64)
    Dmean    = np.full((H, Ww), np.nan, dtype=np.float64)
    Dstar    = np.full((H, Ww), np.nan, dtype=np.float64)

    for i, eta_b in enumerate(etaB_vals):
        for j, eta_W in enumerate(etaW_vals):
            cfg = TrainConfig(n=n, m=m, S=S, steps=steps, eta_W=float(eta_W), eta_b=float(eta_b),
                              mode=mode, N=N, seed=seed, dtype64=dtype64, importance_uniform=importance_uniform)
            rr = train_single_run(cfg, device=device, shared_init=(W0, b0), shared_data=Xshared)

            diverged[i, j] = rr.diverged
            blue[i, j]     = rr.blue_sum
            red[i, j]      = rr.red_sum
            finalL[i, j]   = rr.final_loss
            if record_fractional_dim and (rr.D_mean is not None):
                Dmean[i, j]  = rr.D_mean
                Dstar[i, j]  = rr.D_star

    return dict(
        etaW_vals=etaW_vals,
        etaB_vals=etaB_vals,
        diverged=diverged,
        blue=blue,
        red=red,
        final_loss=finalL,
        Dmean=Dmean,
        Dstar=Dstar
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
    """
    Converged = blue channel (intensity ~ sum loss); Diverged = red channel (intensity ~ sum 1/loss).
    """
    converged_mask = ~diverged
    red_norm  = _normalize_channel(red, diverged)
    blue_norm = _normalize_channel(blue, converged_mask)

    R = red_norm
    G = np.zeros_like(R)
    B = blue_norm
    return np.stack([R, G, B], axis=-1)

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

def evaluate_grid_distributed(n: int, m: int, S: float, steps: int,
                              etaW_range: Tuple[float, float],
                              etaB_range: Tuple[float, float],
                              grid: Tuple[int, int],
                              mode: str = "analytic-L1",
                              log_etaW: bool = True,
                              log_etaB: bool = True,
                              seed: int = 123,
                              dtype64: bool = True,
                              N: int = 8192,
                              importance_uniform: bool = True,
                              out_path: Optional[str] = None):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for DDP")

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = select_device(rank)

    set_torch_defaults(float64=dtype64)
    seed_everything(seed + rank)

    W0, b0 = init_params(m, n, device=device, seed=seed + 111)
    Xshared = sample_dataset(n, N, S, device, seed=seed + 222) if mode == "sampled" else None

    H, Ww = grid
    etaW_vals = _lin_or_log_space(etaW_range[0], etaW_range[1], Ww, log_etaW)
    etaB_vals = _lin_or_log_space(etaB_range[0], etaB_range[1], H,  log_etaB)

    coords = [(i, j) for i in range(H) for j in range(Ww)]
    my_ids = _grid_indices(len(coords), world, rank)

    diverged = np.zeros((H, Ww), dtype=bool)
    blue     = np.zeros((H, Ww), dtype=np.float64)
    red      = np.zeros((H, Ww), dtype=np.float64)
    finalL   = np.zeros((H, Ww), dtype=np.float64)
    Dmean    = np.full((H, Ww), np.nan, dtype=np.float64)
    Dstar    = np.full((H, Ww), np.nan, dtype=np.float64)

    for idx in my_ids:
        i, j = coords[idx]
        eta_b = etaB_vals[i]
        eta_W = etaW_vals[j]
        cfg = TrainConfig(n=n, m=m, S=S, steps=steps, eta_W=float(eta_W), eta_b=float(eta_b),
                          mode=mode, N=N, seed=seed, dtype64=dtype64, importance_uniform=importance_uniform)
        rr = train_single_run(cfg, device=device, shared_init=(W0, b0), shared_data=Xshared)
        diverged[i, j] = rr.diverged
        blue[i, j]     = rr.blue_sum
        red[i, j]      = rr.red_sum
        finalL[i, j]   = rr.final_loss
        if rr.D_mean is not None:
            Dmean[i, j] = rr.D_mean
            Dstar[i, j] = rr.D_star

    tensors = []
    for arr, typ in [(diverged.astype(np.int64), 'i'),
                     (blue, 'f'), (red, 'f'), (finalL, 'f'),
                     (np.nan_to_num(Dmean, nan=0.0), 'f'),
                     (np.isnan(Dmean).astype(np.int64), 'i'),
                     (np.nan_to_num(Dstar, nan=0.0), 'f'),
                     (np.isnan(Dstar).astype(np.int64), 'i')]:
        t = torch.from_numpy(arr).to(device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        tensors.append(t.cpu().numpy())

    diverged_sum = tensors[0]
    blue_sum     = tensors[1]
    red_sum      = tensors[2]
    finalL_sum   = tensors[3]
    Dmean_sum    = tensors[4]
    Dmean_nan    = tensors[5]
    Dstar_sum    = tensors[6]
    Dstar_nan    = tensors[7]

    diverged_out = (diverged_sum > 0).astype(bool)
    blue_out     = blue_sum
    red_out      = red_sum
    finalL_out   = finalL_sum

    denom_Dmean  = np.maximum(world - Dmean_nan, 1)
    Dmean_out    = Dmean_sum / denom_Dmean

    denom_Dstar  = np.maximum(world - Dstar_nan, 1)
    Dstar_out    = Dstar_sum / denom_Dstar

    if rank == 0 and out_path is not None:
        with open(out_path, "wb") as f:
            np.savez_compressed(f,
                                n=n, m=m, S=S, steps=steps,
                                etaW_vals=etaW_vals, etaB_vals=etaB_vals,
                                diverged=diverged_out, blue=blue_out, red=red_out, final_loss=finalL_out,
                                Dmean=Dmean_out, Dstar=Dstar_out)
        print(f"[rank 0] wrote {out_path}")

    dist.destroy_process_group()

def main():
    p = argparse.ArgumentParser(description="Toy Models (ReLU output) fractal boundary maps")
    p.add_argument("--n", type=int, default=256, help="# features")
    p.add_argument("--m", type=int, default=32, help="embedding dims (m << n)")
    p.add_argument("--S", type=float, default=0.95, help="sparsity P(x_i=0)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--mode", type=str, default="analytic-L1", choices=["analytic-L1","sampled"])
    p.add_argument("--N", type=int, default=8192, help="dataset size when mode='sampled'")
    p.add_argument("--etaW-min", type=float, default=1e-4)
    p.add_argument("--etaW-max", type=float, default=1e+1)
    p.add_argument("--etaB-min", type=float, default=1e-6)
    p.add_argument("--etaB-max", type=float, default=1e+1)
    p.add_argument("--width", type=int, default=1024, help="grid columns (eta_W axis)")
    p.add_argument("--height", type=int, default=1024, help="grid rows (eta_b axis)")
    p.add_argument("--log-etaW", action="store_true", default=True)
    p.add_argument("--log-etaB", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--float32", action="store_true", help="use float32 (default float64)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="tms_grid.npz")
    p.add_argument("--render", type=str, default="tms_grid.png")
    p.add_argument("--ddp", action="store_true", help="use torch.distributed")
    p.add_argument("--importance-nonuniform", action="store_true", help="draw I_i ~ U[0,1]")
    args = p.parse_args()

    dtype64 = not args.float32
    importance_uniform = not args.importance_nonuniform

    if args.ddp:
        evaluate_grid_distributed(n=args.n, m=args.m, S=args.S, steps=args.steps,
                                  etaW_range=(args.etaW_min, args.etaW_max),
                                  etaB_range=(args.etaB_min, args.etaB_max),
                                  grid=(args.height, args.width),
                                  mode=args.mode, log_etaW=args.log_etaW, log_etaB=args.log_etaB,
                                  seed=args.seed, dtype64=dtype64, N=args.N,
                                  importance_uniform=importance_uniform, out_path=args.out)
    else:
        device = select_device() if args.device == "cuda" else args.device
        res = evaluate_grid(n=args.n, m=args.m, S=args.S, steps=args.steps,
                            etaW_range=(args.etaW_min, args.etaW_max),
                            etaB_range=(args.etaB_min, args.etaB_max),
                            grid=(args.height, args.width),
                            mode=args.mode, log_etaW=args.log_etaW, log_etaB=args.log_etaB,
                            seed=args.seed, dtype64=dtype64, device=device,
                            N=args.N, importance_uniform=importance_uniform)
        with open(args.out, "wb") as f:
            np.savez_compressed(f, **res)
        rgb = compose_rgb(res["diverged"], res["blue"], res["red"])
        save_rgb_image(args.render, rgb, dpi=100)
        print(f"Wrote {args.out} and {args.render}")

if __name__ == "__main__":
    main()
