#!/usr/bin/env python3
"""
Train a sweep of MLPs on modular arithmetic (e.g. modular addition) and log
train/test loss+accuracy curves to check for grokking-like dynamics.

This script is intentionally standalone (no wandb, no notebook glue).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def cross_entropy_high_precision(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Match the notebook's "high precision" CE to avoid float32 underflow when the
    model gets extremely confident.
    """

    logprobs = F.log_softmax(logits.to(torch.float32), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    return -torch.mean(prediction_logprobs)


def gen_train_test_pairs(p: int, frac_train: float, seed: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    pairs = [(i, j) for i in range(p) for j in range(p)]
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    pairs = [pairs[k] for k in perm]
    div = int(frac_train * len(pairs))
    return pairs[:div], pairs[div:]


def make_all_data(p: int, fn: Callable[[int, int], int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    # all_pairs: [p*p, 2], labels: [p*p]
    all_pairs = torch.tensor([(i, j) for i in range(p) for j in range(p)], dtype=torch.long, device=device)
    labels = torch.tensor([fn(i, j) for i, j in all_pairs.tolist()], dtype=torch.long, device=device)
    return all_pairs, labels


def make_split_masks(p: int, train_pairs: List[Tuple[int, int]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    train_set = set(train_pairs)
    is_train = torch.tensor([(i, j) in train_set for i in range(p) for j in range(p)], dtype=torch.bool, device=device)
    is_test = ~is_train
    return is_train, is_test


class PairMLP(nn.Module):
    """
    Simple MLP that maps (a,b) -> logits over {0..p-1}.

    We embed a and b separately, concatenate, then run a (depth configurable)
    feedforward network.
    """

    def __init__(self, p: int, d_embed: int, hidden_dim: int, num_hidden_layers: int, act: str = "relu") -> None:
        super().__init__()
        self.p = p
        self.emb_a = nn.Embedding(p, d_embed)
        self.emb_b = nn.Embedding(p, d_embed)

        act = act.lower()
        if act == "relu":
            act_layer: nn.Module = nn.ReLU()
        elif act == "gelu":
            act_layer = nn.GELU()
        else:
            raise ValueError(f"Unknown act: {act}")

        layers: List[nn.Module] = []
        in_dim = 2 * d_embed
        if num_hidden_layers <= 0:
            layers.append(nn.Linear(in_dim, p))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_layer)
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_layer)
            layers.append(nn.Linear(hidden_dim, p))
        self.net = nn.Sequential(*layers)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        # pairs: [batch, 2]
        a = pairs[:, 0]
        b = pairs[:, 1]
        x = torch.cat([self.emb_a(a), self.emb_b(b)], dim=-1)
        return self.net(x)


class Transformer(nn.Module):
    """
    Apple-to-apple baseline: matches the custom Transformer used in
    `Non_Modular_Addition_Grokking_Tasks (1).ipynb`.

    Key properties:
    - explicit per-head Q/K/V/O weight tensors (no nn.MultiheadAttention)
    - causal masking via a stored lower-triangular mask
    - optional `attn_only` (skip MLP)
    - optional separate `d_vocab_out` for the unembed head
    - no LayerNorm in the forward pass (as in the notebook; LN is defined but gated elsewhere)
    """

    def __init__(
        self,
        *,
        num_layers: int,
        d_vocab: int,
        d_vocab_out: int | None,
        d_model: int,
        d_mlp: int,
        d_head: int,
        num_heads: int,
        n_ctx: int,
        act_type: str = "ReLU",
        attn_only: bool = False,
        use_pos: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.d_vocab = d_vocab
        self.d_vocab_out = d_vocab if d_vocab_out is None else d_vocab_out
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.d_head = d_head
        self.num_heads = num_heads
        self.n_ctx = n_ctx
        self.attn_only = attn_only
        self.use_pos = use_pos

        # Embed / unembed parameterizations copied from the notebook.
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / math.sqrt(d_model))
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d_model) / math.sqrt(d_model)) if use_pos else None
        self.W_U = nn.Parameter(torch.randn(d_model, self.d_vocab_out) / math.sqrt(self.d_vocab_out))

        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    d_model=d_model,
                    d_mlp=d_mlp,
                    d_head=d_head,
                    num_heads=num_heads,
                    n_ctx=n_ctx,
                    act_type=act_type,
                    attn_only=attn_only,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [batch, n_ctx]
        # Embed: einsum-style indexing from notebook
        x = torch.einsum("dbp->bpd", self.W_E[:, tokens])  # [b, p, d]
        if self.use_pos and self.W_pos is not None:
            x = x + self.W_pos[: x.shape[-2]]
        for block in self.blocks:
            x = block(x)
        logits = x @ self.W_U  # [b, p, d_vocab_out]
        return logits


class _Attention(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int, d_head: int, n_ctx: int) -> None:
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) / math.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) / math.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) / math.sqrt(d_model))
        # Note: notebook uses W_O shape [num_heads, d_model, d_head] and sums heads.
        self.W_O = nn.Parameter(torch.randn(num_heads, d_model, d_head) / math.sqrt(d_model))
        self.register_buffer("mask", torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, pos, d_model]
        k = torch.einsum("ihd,bpd->biph", self.W_K, x)
        q = torch.einsum("ihd,bpd->biph", self.W_Q, x)
        v = torch.einsum("ihd,bpd->biph", self.W_V, x)

        attn_scores_pre = torch.einsum("biph,biqh->biqp", k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[: x.shape[-2], : x.shape[-2]])
        attn = F.softmax(attn_scores_masked / math.sqrt(self.d_head), dim=-1)

        z = torch.einsum("biph,biqp->biqh", v, attn)
        result = torch.einsum("idh,biqh->biqd", self.W_O, z)
        out = result.sum(dim=1)  # sum over heads -> [batch, pos, d_model]
        return out

class _Autoencoder(nn.Module):
    """
    Bottleneck autoencoder.

    Note: `_MLP` maps d_model -> d_mlp -> d_model, so using it for both encoder
    and decoder does *not* create a bottleneck latent. This does.
    """

    def __init__(self, *, d_model: int, d_latent: int, act_type: str) -> None:
        super().__init__()
        if act_type not in {"ReLU", "GeLU"}:
            raise ValueError("act_type must be ReLU or GeLU (matching notebook)")
        self.act_type = act_type
        self.encoder = nn.Linear(d_model, d_latent, bias=True)
        self.decoder = nn.Linear(d_latent, d_model, bias=True)

    def forward(self, x: torch.Tensor, *, return_latent: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if self.act_type == "ReLU":
            z = F.relu(self.encoder(x))
        else:
            z = F.gelu(self.encoder(x))
        x_hat = self.decoder(z)
        if return_latent:
            return x_hat, z
        return x_hat


class _MLP(nn.Module):
    def __init__(self, *, d_model: int, d_mlp: int, act_type: str) -> None:
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / math.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / math.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        if act_type not in {"ReLU", "GeLU"}:
            raise ValueError("act_type must be ReLU or GeLU (matching notebook)")
        self.act_type = act_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in
        if self.act_type == "ReLU":
            x = F.relu(x)
        else:
            x = F.gelu(x)
        x = torch.einsum("dm,bpm->bpd", self.W_out, x) + self.b_out
        return x


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_mlp: int,
        d_head: int,
        num_heads: int,
        n_ctx: int,
        act_type: str,
        attn_only: bool,
    ) -> None:
        super().__init__()
        self.attn = _Attention(d_model=d_model, num_heads=num_heads, d_head=d_head, n_ctx=n_ctx)
        self.attn_only = attn_only
        if not attn_only:
            self.mlp = _MLP(d_model=d_model, d_mlp=d_mlp, act_type=act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        if not self.attn_only:
            x = x + self.mlp(x)
        return x


@dataclass(frozen=True)
class SweepConfig:
    # Task
    p: int = 113
    fn_name: str = "add"  # add|subtract|x2xyy2|rand
    frac_train: float = 0.3
    seed: int = 0

    # Model
    d_embed: int = 128
    hidden_dims: Tuple[int, ...] = (64, 128, 256, 512, 1024)
    num_hidden_layers: int = 1
    act: str = "relu"

    # Transformer baseline (matches grokking notebook defaults)
    include_transformer_baseline: bool = False
    transformer_d_model: int = 128
    transformer_num_heads: int = 4
    transformer_d_mlp: int = 512
    transformer_d_head: int = 32
    transformer_num_layers: int = 1
    transformer_attn_only: bool = False
    transformer_use_pos: bool = True

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1.0
    num_epochs: int = 50_000
    warmup_steps: int = 10

    # Logging / saving
    eval_every: int = 10
    save_every: int = 1000
    stopping_thresh: float = -1.0  # stop if test loss < thresh, disabled if <0
    out_dir: str = "saved_runs/mlp_sweep"

    # Runtime
    device: str = "cuda"  # cuda|cpu

    # Optional logging
    use_wandb: bool = False
    wandb_project: str = "mlp-grokking"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Tuple[str, ...] = ()

    # Autoencoder (trained post-MLP, on learned embedding-pair vectors)
    include_autoencoder: bool = False
    ae_d_latent: int = 64
    ae_act_type: str = "ReLU"  # ReLU|GeLU (match notebook naming)
    ae_lr: float = 1e-3
    ae_weight_decay: float = 0.0
    ae_num_epochs: int = 5_000
    ae_batch_size: int = 4096
    ae_eval_every: int = 50
    ae_save_every: int = 1000
    ae_noise_std: float = 0.0  # optional denoising AE


def get_fn(p: int, fn_name: str, seed: int) -> Callable[[int, int], int]:
    if fn_name == "add":
        return lambda x, y: (x + y) % p
    if fn_name == "subtract":
        return lambda x, y: (x - y) % p
    if fn_name == "x2xyy2":
        return lambda x, y: (x * x + x * y + y * y) % p
    if fn_name == "rand":
        g = torch.Generator().manual_seed(seed)
        table = torch.randint(low=0, high=p, size=(p, p), generator=g).tolist()
        return lambda x, y: table[x][y]
    raise ValueError(f"Unknown fn_name: {fn_name}")


@torch.inference_mode()
def eval_split(model: nn.Module, pairs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    logits = model(pairs)
    loss = cross_entropy_high_precision(logits, labels).item()
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


def _mlp_pair_features(model: PairMLP, pairs: torch.Tensor) -> torch.Tensor:
    """
    Extract the concatenated embedding features used as PairMLP input:
    x = concat(emb_a(a), emb_b(b)).
    Returns a [N, 2*d_embed] float tensor on the same device as `pairs`.
    """
    a = pairs[:, 0]
    b = pairs[:, 1]
    return torch.cat([model.emb_a(a), model.emb_b(b)], dim=-1)


def train_autoencoder_on_mlp_embeddings(
    cfg: SweepConfig,
    *,
    mlp_model: PairMLP,
    train_pairs: torch.Tensor,
    test_pairs: torch.Tensor,
    run_dir: Path,
    wandb_run: object | None,
) -> Dict:
    """
    Train a small bottleneck autoencoder on the learned PairMLP input features.
    """
    device = next(mlp_model.parameters()).device
    run_dir.mkdir(parents=True, exist_ok=False)

    mlp_model.eval()
    with torch.inference_mode():
        x_train = _mlp_pair_features(mlp_model, train_pairs).to(device)
        x_test = _mlp_pair_features(mlp_model, test_pairs).to(device)

    d_model = int(x_train.shape[-1])
    ae = _Autoencoder(d_model=d_model, d_latent=cfg.ae_d_latent, act_type=cfg.ae_act_type).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=cfg.ae_lr, weight_decay=cfg.ae_weight_decay, betas=(0.9, 0.98))

    def lr_scale(step: int) -> float:
        if cfg.warmup_steps <= 0:
            return 1.0
        return min((step + 1) / cfg.warmup_steps, 1.0)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_scale)

    dl = DataLoader(
        TensorDataset(x_train),
        batch_size=min(cfg.ae_batch_size, int(x_train.shape[0])),
        shuffle=True,
        drop_last=False,
    )

    @torch.inference_mode()
    def eval_mse(x: torch.Tensor) -> float:
        ae.eval()
        x_hat = ae(x.unsqueeze(1)).squeeze(1)
        return F.mse_loss(x_hat, x).item()

    steps: List[int] = []
    train_mse: List[float] = []
    test_mse: List[float] = []

    tr0 = eval_mse(x_train)
    te0 = eval_mse(x_test)
    steps.append(0)
    train_mse.append(tr0)
    test_mse.append(te0)

    if wandb_run is not None:
        import wandb  # type: ignore

        wandb.log({"ae/train_mse": tr0, "ae/test_mse": te0, "ae/lr": opt.param_groups[0]["lr"]}, step=0)

    opt.zero_grad(set_to_none=True)
    for epoch in range(cfg.ae_num_epochs):
        ae.train()
        for (xb,) in dl:
            x_in = xb
            if cfg.ae_noise_std > 0:
                x_in = x_in + cfg.ae_noise_std * torch.randn_like(x_in)
            x_hat = ae(x_in.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(x_hat, xb)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        if (epoch + 1) % cfg.ae_eval_every == 0:
            tr = eval_mse(x_train)
            te = eval_mse(x_test)
            steps.append(epoch + 1)
            train_mse.append(tr)
            test_mse.append(te)

            if wandb_run is not None:
                import wandb  # type: ignore

                wandb.log(
                    {"ae/train_mse": tr, "ae/test_mse": te, "ae/lr": opt.param_groups[0]["lr"]},
                    step=epoch + 1,
                )

            if (epoch + 1) % max(cfg.ae_eval_every * 10, 100) == 0:
                print(f"[autoencoder] epoch={epoch+1:>6} train_mse={tr:.6g} test_mse={te:.6g}")

        if cfg.ae_save_every > 0 and (epoch + 1) % cfg.ae_save_every == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": ae.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "config": {**asdict(cfg), "ae_d_model": d_model, "device_used": str(device), "ae_source": "mlp_input_embeddings"},
                "metrics": {"steps": steps, "train_mse": train_mse, "test_mse": test_mse},
            }
            torch.save(ckpt, run_dir / f"ckpt_epoch_{epoch+1}.pth")

    final = {
        "epoch": steps[-1],
        "model_state_dict": ae.state_dict(),
        "config": {**asdict(cfg), "ae_d_model": d_model, "device_used": str(device), "ae_source": "mlp_input_embeddings"},
        "metrics": {"steps": steps, "train_mse": train_mse, "test_mse": test_mse},
        "split": {"train_size": int(x_train.shape[0]), "test_size": int(x_test.shape[0])},
    }
    torch.save(final, run_dir / "final.pth")
    return final


def train_one_width(cfg: SweepConfig, hidden_dim: int, run_dir: Path) -> Dict:
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Repro
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    fn = get_fn(cfg.p, cfg.fn_name, cfg.seed)
    all_pairs, all_labels = make_all_data(cfg.p, fn, device=device)
    train_pairs, test_pairs = gen_train_test_pairs(cfg.p, cfg.frac_train, cfg.seed)
    is_train, is_test = make_split_masks(cfg.p, train_pairs, device=device)

    train_x, train_y = all_pairs[is_train], all_labels[is_train]
    test_x, test_y = all_pairs[is_test], all_labels[is_test]

    model = PairMLP(
        p=cfg.p,
        d_embed=cfg.d_embed,
        hidden_dim=hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        act=cfg.act,
    ).to(device)

    wandb_run = None
    if cfg.use_wandb:
        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "wandb logging requested but wandb isn't importable. "
                "Install it with `pip install wandb` (and set WANDB_API_KEY)."
            ) from e

        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            tags=list(cfg.wandb_tags),
            config={**asdict(cfg), "hidden_dim": hidden_dim, "device_used": str(device)},
            name=f"hidden_{hidden_dim}",
        )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))

    def lr_scale(step: int) -> float:
        if cfg.warmup_steps <= 0:
            return 1.0
        return min((step + 1) / cfg.warmup_steps, 1.0)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_scale)

    # Logs
    steps: List[int] = []
    train_losses: List[float] = []
    test_losses: List[float] = []
    train_accs: List[float] = []
    test_accs: List[float] = []

    # Initial eval
    tl0, ta0 = eval_split(model, train_x, train_y)
    vl0, va0 = eval_split(model, test_x, test_y)
    steps.append(0)
    train_losses.append(tl0)
    test_losses.append(vl0)
    train_accs.append(ta0)
    test_accs.append(va0)

    if wandb_run is not None:
        import wandb  # type: ignore

        wandb.log(
            {"train/loss": tl0, "test/loss": vl0, "train/acc": ta0, "test/acc": va0, "lr": opt.param_groups[0]["lr"]},
            step=0,
        )

    opt.zero_grad(set_to_none=True)
    for epoch in range(cfg.num_epochs):
        model.train()
        logits = model(train_x)
        loss = cross_entropy_high_precision(logits, train_y)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if (epoch + 1) % cfg.eval_every == 0:
            model.eval()
            tl, ta = eval_split(model, train_x, train_y)
            vl, va = eval_split(model, test_x, test_y)
            steps.append(epoch + 1)
            train_losses.append(tl)
            test_losses.append(vl)
            train_accs.append(ta)
            test_accs.append(va)

            if wandb_run is not None:
                import wandb  # type: ignore

                wandb.log(
                    {
                        "train/loss": tl,
                        "test/loss": vl,
                        "train/acc": ta,
                        "test/acc": va,
                        "lr": opt.param_groups[0]["lr"],
                    },
                    step=epoch + 1,
                )

            if (epoch + 1) % max(cfg.eval_every * 10, 100) == 0:
                print(
                    f"[hidden={hidden_dim}] epoch={epoch+1:>6} "
                    f"train_loss={math.log(tl): .4f} test_loss={math.log(vl): .4f} "
                    f"train_acc={ta: .3f} test_acc={va: .3f}"
                )

            if cfg.stopping_thresh >= 0 and vl < cfg.stopping_thresh:
                print(f"[hidden={hidden_dim}] early stop at epoch={epoch+1}, test_loss={vl:g}")
                break

        if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "config": {**asdict(cfg), "hidden_dim": hidden_dim, "device_used": str(device)},
                "metrics": {
                    "steps": steps,
                    "train_loss": train_losses,
                    "test_loss": test_losses,
                    "train_acc": train_accs,
                    "test_acc": test_accs,
                },
            }
            torch.save(ckpt, run_dir / f"ckpt_epoch_{epoch+1}.pth")

    ae_result: Dict | None = None
    if cfg.include_autoencoder:
        ae_dir = run_dir / "autoencoder"
        ae_result = train_autoencoder_on_mlp_embeddings(
            cfg,
            mlp_model=model,
            train_pairs=train_x,
            test_pairs=test_x,
            run_dir=ae_dir,
            wandb_run=wandb_run,
        )

    final = {
        "epoch": steps[-1],
        "model_state_dict": model.state_dict(),
        "config": {**asdict(cfg), "hidden_dim": hidden_dim, "device_used": str(device)},
        "metrics": {
            "steps": steps,
            "train_loss": train_losses,
            "test_loss": test_losses,
            "train_acc": train_accs,
            "test_acc": test_accs,
        },
        "split": {"train_size": int(train_x.shape[0]), "test_size": int(test_x.shape[0])},
        "autoencoder": ae_result,
    }
    torch.save(final, run_dir / "final.pth")

    if wandb_run is not None:
        wandb_run.finish()
    return final


def train_transformer_baseline(cfg: SweepConfig, run_dir: Path) -> Dict:
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Repro
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    fn = get_fn(cfg.p, cfg.fn_name, cfg.seed)
    all_pairs, all_labels = make_all_data(cfg.p, fn, device=device)
    train_pairs, _ = gen_train_test_pairs(cfg.p, cfg.frac_train, cfg.seed)
    is_train, is_test = make_split_masks(cfg.p, train_pairs, device=device)

    equals_tok = cfg.p  # extra token in vocab p+1
    all_seq = torch.stack([all_pairs[:, 0], all_pairs[:, 1], torch.full_like(all_pairs[:, 0], equals_tok)], dim=-1)
    train_x, train_y = all_seq[is_train], all_labels[is_train]
    test_x, test_y = all_seq[is_test], all_labels[is_test]

    model = Transformer(
        num_layers=cfg.transformer_num_layers,
        d_vocab=cfg.p + 1,
        d_vocab_out=cfg.p,
        d_model=cfg.transformer_d_model,
        d_mlp=cfg.transformer_d_mlp,
        d_head=cfg.transformer_d_head,
        num_heads=cfg.transformer_num_heads,
        n_ctx=3,
        act_type="ReLU" if cfg.act.lower() == "relu" else "GeLU",
        attn_only=cfg.transformer_attn_only,
        use_pos=cfg.transformer_use_pos,
    ).to(device)

    wandb_run = None
    if cfg.use_wandb:
        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "wandb logging requested but wandb isn't importable. "
                "Install it with `pip install wandb` (and set WANDB_API_KEY)."
            ) from e

        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            tags=list(cfg.wandb_tags),
            config={**asdict(cfg), "model_type": "transformer_baseline", "device_used": str(device)},
            name="baseline_transformer",
        )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))

    def lr_scale(step: int) -> float:
        if cfg.warmup_steps <= 0:
            return 1.0
        return min((step + 1) / cfg.warmup_steps, 1.0)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_scale)

    steps: List[int] = []
    train_losses: List[float] = []
    test_losses: List[float] = []
    train_accs: List[float] = []
    test_accs: List[float] = []

    @torch.inference_mode()
    def eval_seq(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        logits = model(x)[:, -1, :]  # final position, vocab_out == p
        loss = cross_entropy_high_precision(logits, y).item()
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        return loss, acc

    tl0, ta0 = eval_seq(train_x, train_y)
    vl0, va0 = eval_seq(test_x, test_y)
    steps.append(0)
    train_losses.append(tl0)
    test_losses.append(vl0)
    train_accs.append(ta0)
    test_accs.append(va0)

    if wandb_run is not None:
        import wandb  # type: ignore

        wandb.log(
            {"train/loss": tl0, "test/loss": vl0, "train/acc": ta0, "test/acc": va0, "lr": opt.param_groups[0]["lr"]},
            step=0,
        )

    opt.zero_grad(set_to_none=True)
    for epoch in range(cfg.num_epochs):
        model.train()
        logits = model(train_x)[:, -1, :]
        loss = cross_entropy_high_precision(logits, train_y)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if (epoch + 1) % cfg.eval_every == 0:
            model.eval()
            tl, ta = eval_seq(train_x, train_y)
            vl, va = eval_seq(test_x, test_y)
            steps.append(epoch + 1)
            train_losses.append(tl)
            test_losses.append(vl)
            train_accs.append(ta)
            test_accs.append(va)

            if wandb_run is not None:
                import wandb  # type: ignore

                wandb.log(
                    {
                        "train/loss": tl,
                        "test/loss": vl,
                        "train/acc": ta,
                        "test/acc": va,
                        "lr": opt.param_groups[0]["lr"],
                    },
                    step=epoch + 1,
                )

            if (epoch + 1) % max(cfg.eval_every * 10, 100) == 0:
                print(
                    f"[transformer] epoch={epoch+1:>6} "
                    f"train_loss={math.log(tl): .4f} test_loss={math.log(vl): .4f} "
                    f"train_acc={ta: .3f} test_acc={va: .3f}"
                )

            if cfg.stopping_thresh >= 0 and vl < cfg.stopping_thresh:
                print(f"[transformer] early stop at epoch={epoch+1}, test_loss={vl:g}")
                break

        if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "config": {**asdict(cfg), "model_type": "transformer_baseline", "device_used": str(device)},
                "metrics": {
                    "steps": steps,
                    "train_loss": train_losses,
                    "test_loss": test_losses,
                    "train_acc": train_accs,
                    "test_acc": test_accs,
                },
            }
            torch.save(ckpt, run_dir / f"ckpt_epoch_{epoch+1}.pth")

    final = {
        "epoch": steps[-1],
        "model_state_dict": model.state_dict(),
        "config": {**asdict(cfg), "model_type": "transformer_baseline", "device_used": str(device)},
        "metrics": {
            "steps": steps,
            "train_loss": train_losses,
            "test_loss": test_losses,
            "train_acc": train_accs,
            "test_acc": test_accs,
        },
        "split": {"train_size": int(train_x.shape[0]), "test_size": int(test_x.shape[0])},
    }
    torch.save(final, run_dir / "final.pth")

    if wandb_run is not None:
        wandb_run.finish()
    return final


def maybe_save_plot(run_dir: Path, metrics: Dict[str, List[float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    steps = metrics["steps"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(steps, metrics["train_loss"], label="train")
    axes[0].plot(steps, metrics["test_loss"], label="test")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss (log scale)")
    axes[0].legend()

    axes[1].plot(steps, metrics["train_acc"], label="train")
    axes[1].plot(steps, metrics["test_acc"], label="test")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=160)
    plt.close(fig)


def parse_args() -> SweepConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p", type=int, default=SweepConfig.p)
    p.add_argument("--fn-name", type=str, default=SweepConfig.fn_name, choices=["add", "subtract", "x2xyy2", "rand"])
    p.add_argument("--frac-train", type=float, default=SweepConfig.frac_train)
    p.add_argument("--seed", type=int, default=SweepConfig.seed)

    p.add_argument("--d-embed", type=int, default=SweepConfig.d_embed)
    #here is what you would vary
    p.add_argument("--hidden-dims", type=int, nargs="+", default=list(SweepConfig.hidden_dims))
    p.add_argument("--num-hidden-layers", type=int, default=SweepConfig.num_hidden_layers)
    p.add_argument("--act", type=str, default=SweepConfig.act, choices=["relu", "gelu"])

    p.add_argument(
        "--include-transformer-baseline",
        action="store_true",
        help="Also train a tiny 1-layer causal transformer baseline (like the Grokking_Analysis notebook).",
    )
    p.add_argument("--transformer-d-model", type=int, default=SweepConfig.transformer_d_model)
    p.add_argument("--transformer-num-heads", type=int, default=SweepConfig.transformer_num_heads)
    p.add_argument("--transformer-d-mlp", type=int, default=SweepConfig.transformer_d_mlp)
    p.add_argument("--transformer-d-head", type=int, default=SweepConfig.transformer_d_head)
    p.add_argument("--transformer-num-layers", type=int, default=SweepConfig.transformer_num_layers)
    p.add_argument("--transformer-attn-only", action="store_true", default=SweepConfig.transformer_attn_only)
    p.add_argument("--no-transformer-pos", action="store_true", help="Disable positional embeddings in transformer baseline.")

    p.add_argument("--lr", type=float, default=SweepConfig.lr)
    p.add_argument("--weight-decay", type=float, default=SweepConfig.weight_decay)
    p.add_argument("--num-epochs", type=int, default=SweepConfig.num_epochs)
    p.add_argument("--warmup-steps", type=int, default=SweepConfig.warmup_steps)

    p.add_argument("--eval-every", type=int, default=SweepConfig.eval_every)
    p.add_argument("--save-every", type=int, default=SweepConfig.save_every)
    p.add_argument("--stopping-thresh", type=float, default=SweepConfig.stopping_thresh)
    p.add_argument("--out-dir", type=str, default=SweepConfig.out_dir)

    p.add_argument("--device", type=str, default=SweepConfig.device, choices=["cuda", "cpu"])

    # Optional wandb logging
    p.add_argument("--use-wandb", action="store_true", help="Log metrics to Weights & Biases (wandb).")
    p.add_argument("--wandb-project", type=str, default=SweepConfig.wandb_project)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, nargs="*", default=[])

    # Autoencoder training (post-MLP, per width)
    p.add_argument("--include-autoencoder", action="store_true", help="Train an autoencoder on learned MLP embedding-pair features.")
    p.add_argument("--ae-d-latent", type=int, default=SweepConfig.ae_d_latent)
    p.add_argument("--ae-act-type", type=str, default=SweepConfig.ae_act_type, choices=["ReLU", "GeLU"])
    p.add_argument("--ae-lr", type=float, default=SweepConfig.ae_lr)
    p.add_argument("--ae-weight-decay", type=float, default=SweepConfig.ae_weight_decay)
    p.add_argument("--ae-num-epochs", type=int, default=SweepConfig.ae_num_epochs)
    p.add_argument("--ae-batch-size", type=int, default=SweepConfig.ae_batch_size)
    p.add_argument("--ae-eval-every", type=int, default=SweepConfig.ae_eval_every)
    p.add_argument("--ae-save-every", type=int, default=SweepConfig.ae_save_every)
    p.add_argument("--ae-noise-std", type=float, default=SweepConfig.ae_noise_std)

    args = p.parse_args()
    return SweepConfig(
        p=args.p,
        fn_name=args.fn_name,
        frac_train=args.frac_train,
        seed=args.seed,
        d_embed=args.d_embed,
        hidden_dims=tuple(args.hidden_dims),
        num_hidden_layers=args.num_hidden_layers,
        act=args.act,
        include_transformer_baseline=bool(args.include_transformer_baseline),
        transformer_d_model=args.transformer_d_model,
        transformer_num_heads=args.transformer_num_heads,
        transformer_d_mlp=args.transformer_d_mlp,
        transformer_d_head=args.transformer_d_head,
        transformer_num_layers=args.transformer_num_layers,
        transformer_attn_only=bool(args.transformer_attn_only),
        transformer_use_pos=not bool(args.no_transformer_pos),
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        stopping_thresh=args.stopping_thresh,
        out_dir=args.out_dir,
        device=args.device,
        use_wandb=bool(args.use_wandb),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_tags=tuple(args.wandb_tags),
        include_autoencoder=bool(args.include_autoencoder),
        ae_d_latent=args.ae_d_latent,
        ae_act_type=args.ae_act_type,
        ae_lr=args.ae_lr,
        ae_weight_decay=args.ae_weight_decay,
        ae_num_epochs=args.ae_num_epochs,
        ae_batch_size=args.ae_batch_size,
        ae_eval_every=args.ae_eval_every,
        ae_save_every=args.ae_save_every,
        ae_noise_std=args.ae_noise_std,
    )


def main() -> None:
    cfg = parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(cfg.out_dir) / f"{cfg.fn_name}_p{cfg.p}_ft{cfg.frac_train}_seed{cfg.seed}_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=False)

    (sweep_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    print(f"Saving sweep to: {sweep_dir.resolve()}")

    summary = []

    if cfg.include_transformer_baseline:
        run_dir = sweep_dir / "baseline_transformer"
        run_dir.mkdir(parents=True, exist_ok=False)
        result = train_transformer_baseline(cfg, run_dir=run_dir)
        maybe_save_plot(run_dir, result["metrics"])

        last = {
            "model_type": "transformer_baseline",
            "hidden_dim": None,
            "epoch": result["epoch"],
            "train_loss": result["metrics"]["train_loss"][-1],
            "test_loss": result["metrics"]["test_loss"][-1],
            "train_acc": result["metrics"]["train_acc"][-1],
            "test_acc": result["metrics"]["test_acc"][-1],
        }
        summary.append(last)
        (run_dir / "summary.json").write_text(json.dumps(last, indent=2) + "\n")

    for hidden in cfg.hidden_dims:
        run_dir = sweep_dir / f"hidden_{hidden}"
        run_dir.mkdir(parents=True, exist_ok=False)
        result = train_one_width(cfg, hidden_dim=hidden, run_dir=run_dir)
        maybe_save_plot(run_dir, result["metrics"])

        last = {
            "model_type": "mlp",
            "hidden_dim": hidden,
            "epoch": result["epoch"],
            "train_loss": result["metrics"]["train_loss"][-1],
            "test_loss": result["metrics"]["test_loss"][-1],
            "train_acc": result["metrics"]["train_acc"][-1],
            "test_acc": result["metrics"]["test_acc"][-1],
        }
        summary.append(last)
        (run_dir / "summary.json").write_text(json.dumps(last, indent=2) + "\n")

    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Done.")


if __name__ == "__main__":
    main()

