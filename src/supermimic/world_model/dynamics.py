"""Dynamics MLP f_omega (paper Table A.4: MLP (2048, 2048, 2048)) and shared network building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

# Optional autocast dtype for every MLP (config ``amp``: None = TF32 matmuls as in the paper,
# torch.bfloat16 = bf16 matmuls). State integration, FK and rewards stay fp32.
AMP_DTYPE: torch.dtype | None = None

ACTIVATIONS = {"silu": nn.SiLU, "elu": nn.ELU}


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.SiLU, zero_last: bool = False, last_gain: float | None = None, layer_norm: bool = False):
        super().__init__()
        if isinstance(act, str):
            act = ACTIVATIONS[act]
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h)] + ([nn.LayerNorm(h)] if layer_norm else []) + [act()]
            d = h
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if AMP_DTYPE is not None and x.is_cuda:
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                return self.net(x).float()
        return self.net(x)


class RunningNorm(nn.Module):
    """Running mean / variance normaliser (as in supermimic_vibed), updated from data without
    gradients and frozen once ``max_count`` samples have been seen. Updates are in place and
    host-sync free, so compiled graphs that read the buffers stay valid."""

    def __init__(self, dim: int, eps: float = 1e-2, max_count: float = 2e8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.eps, self.max_count = eps, max_count

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + x.shape[0]
        w = x.shape[0] / tot
        delta = m - self.mean
        active = self.count < self.max_count
        self.mean.copy_(torch.where(active, self.mean + delta * w, self.mean))
        self.var.copy_(torch.where(active, self.var * (1 - w) + v * w + delta * delta * w * (1 - w), self.var))
        self.count.copy_(torch.where(active, tot, self.count))

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std()


class DynamicsMLP(nn.Module):
    """delta = f(canonical(s), a, xi_norm) -> (…, 70). Zero-initialised last layer (identity dynamics at init)."""

    def __init__(self, feat_dim: int = 71, act_dim: int = 29, xi_dim: int = 33, hidden=(2048, 2048, 2048), out_dim: int = 70, layer_norm: bool = False):
        super().__init__()
        self.mlp = MLP(feat_dim + act_dim + xi_dim, tuple(hidden), out_dim, zero_last=True, layer_norm=layer_norm)

    def forward(self, feat: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor, in_norm: RunningNorm | None = None) -> torch.Tensor:
        x = torch.cat([feat, action, xi_norm], dim=-1)
        return self.mlp(in_norm(x) if in_norm is not None else x)
