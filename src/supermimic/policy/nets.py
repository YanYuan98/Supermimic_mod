"""Policy (goal encoder + shared decoder) and critic (paper Table A.4).

Terrain (docs/VISION_TERRAIN_PLAN.md §2.5): with ``PolicyCfg.terrain_dim > 0`` a ``TerrainEncoder`` maps the
(noisy) height scan to a 32-d latent and the decoder input becomes [z_motion, z_terrain, obs_hist];
``CriticCfg.terrain_dim > 0`` appends the noise-free terrain features to the critic input. Both default to
0, which builds exactly the flat networks (old checkpoints load strictly).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from supermimic.world_model.dynamics import MLP, RunningNorm
from supermimic.world_model.heads import CRITIC_ERR_DIM, OBS_TERMS

GOAL_DIM = 680
LATENT_DIM = 32
ACT_DIM = 29


@dataclass
class PolicyCfg:
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = LATENT_DIM
    history: int = 10
    # Fixed per-term observation scales (divide): joint_pos, joint_vel, base_ang_vel, projected_gravity, actions.
    obs_term_scale: tuple[float, ...] = (1.0, 5.0, 2.0, 1.0, 1.0)
    # Fixed goal feature scales: joint_pos 29, joint_vel 29, grav 3, ang_vel_b 3, lin_vel_h 3, height 1.
    goal_term_scale: tuple[float, ...] = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    decoder_last_gain: float = 0.01
    goal_dim: int = GOAL_DIM  # future_frames * goal_frame_dim + anchor dims (set by the trainer)
    future_frames: int = 10
    goal_frame_dim: int = 68  # 68 ("v1") or 77 ("traj")
    # Options from supermimic_vibed: "fixed" per-term input scales, or "running" mean/std of the
    # observation history and goal (updated from simulator observations during collection).
    input_norm: str = "running"
    activation: str = "elu"  # "silu" | "elu"
    # Terrain scan input (set by the trainer from TerrainCfg): 187 (17x11 scan) or 0 (flat policy).
    terrain_dim: int = 0
    terrain_hidden: tuple[int, ...] = (256, 128)
    terrain_latent_dim: int = 32


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (256, 128)
    # Privileged tracking-error features (IMPROVEMENT_PLAN F5); False = root errors only (4).
    error_features: bool = True
    # Noise-free terrain features appended to the input (set by the trainer): 215 (foot 28 + scan 187) or 0.
    terrain_dim: int = 0


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor:
    parts = []
    for (name, d), s in zip(OBS_TERMS, cfg.obs_term_scale):
        parts.append(torch.full((cfg.history * d,), 1.0 / s))
    return torch.cat(parts)  # term-major layout matches the flattened history


def _goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor:
    sizes = (29, 29, 3, 3, 3, 1)
    frame_dim = getattr(cfg, "goal_frame_dim", 68)
    per_frame = torch.cat([torch.full((n,), 1.0 / s) for n, s in zip(sizes, cfg.goal_term_scale)] + [torch.ones(frame_dim - 68)])
    k = getattr(cfg, "future_frames", 10)
    extra = goal_dim - k * frame_dim
    if extra < 0:  # configs from before future_frames/goal_frame_dim were stored
        k, extra = divmod(goal_dim, frame_dim)
    return torch.cat([per_frame.repeat(k), torch.ones(extra)])


def _running(cfg: PolicyCfg) -> bool:
    mode = getattr(cfg, "input_norm", "fixed")
    if mode not in ("fixed", "running"):
        raise ValueError(f"unknown policy input_norm {mode!r}")
    return mode == "running"


class MotionEncoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, goal_dim: int | None = None):
        super().__init__()
        goal_dim = goal_dim or cfg.goal_dim
        self.mlp = MLP(goal_dim, cfg.encoder_hidden, cfg.latent_dim, act=getattr(cfg, "activation", "silu"))
        self.register_buffer("scale", _goal_scale_vector(cfg, goal_dim))
        self.norm = RunningNorm(goal_dim) if _running(cfg) else None

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(self.norm(goal) if self.norm is not None else goal * self.scale))


def _terrain_latent(cfg: PolicyCfg) -> int:
    return cfg.terrain_latent_dim if getattr(cfg, "terrain_dim", 0) else 0


class TerrainEncoder(nn.Module):
    """z_terrain = tanh(MLP(RunningNorm(scan))) (plan §2.5); a depth student only replaces this module."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.norm = RunningNorm(cfg.terrain_dim)
        self.mlp = MLP(cfg.terrain_dim, tuple(cfg.terrain_hidden), cfg.terrain_latent_dim, act=getattr(cfg, "activation", "silu"))

    def forward(self, scan: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(self.norm(scan)))


class Decoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, obs_dim: int = 930):
        super().__init__()
        self.mlp = MLP(cfg.latent_dim + _terrain_latent(cfg) + obs_dim, cfg.decoder_hidden, ACT_DIM, act=getattr(cfg, "activation", "silu"), last_gain=cfg.decoder_last_gain)
        self.register_buffer("scale", _obs_scale_vector(cfg))
        self.norm = RunningNorm(obs_dim) if _running(cfg) else None

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor, z_terrain: torch.Tensor | None = None) -> torch.Tensor:
        o = self.norm(obs_hist) if self.norm is not None else obs_hist * self.scale
        return self.mlp(torch.cat([z, o] if z_terrain is None else [z, z_terrain, o], dim=-1))


class Policy(nn.Module):
    """a = dec(enc_motion(g), [enc_terrain(scan)], o_hist). Extra goal interfaces can be registered later
    (plan Appendix B)."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({"motion": MotionEncoder(cfg)})
        self.terrain = TerrainEncoder(cfg) if getattr(cfg, "terrain_dim", 0) else None
        self.decoder = Decoder(cfg)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion", *, scan: torch.Tensor | None = None) -> torch.Tensor:
        """``scan`` (N, 187): policy height scan; required iff the policy has a terrain encoder (ignored otherwise)."""
        z = self.encoders[interface](goal)
        if self.terrain is None:
            return self.decoder(z, obs_hist)
        if scan is None:
            raise ValueError("this policy has a terrain encoder: pass scan=")
        return self.decoder(z, obs_hist, self.terrain(scan))

    @torch.no_grad()
    def update_input_norms(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion", *, scan: torch.Tensor | None = None) -> None:
        """Feed simulator observations to the running input normalisers (no-op for "fixed"; the terrain
        encoder's scan normaliser is always a running one)."""
        enc = self.encoders[interface]
        if enc.norm is not None:
            enc.norm.update(goal)
        if self.decoder.norm is not None:
            self.decoder.norm.update(obs_hist)
        if self.terrain is not None and scan is not None:
            self.terrain.norm.update(scan)

    @property
    def has_running_norms(self) -> bool:
        return self.decoder.norm is not None or self.terrain is not None


class Critic(nn.Module):
    """V(canonical state 71, goal, xi 33, privileged tracking errors 84 | root error 4 [, terrain features 215])
    -> scalar (one per interface)."""

    def __init__(self, cfg: CriticCfg, feat_dim: int = 71, goal_dim: int | None = None, xi_dim: int = 33, pcfg: PolicyCfg | None = None):
        super().__init__()
        pcfg = pcfg or PolicyCfg()
        goal_dim = goal_dim or pcfg.goal_dim
        self.err_dim = CRITIC_ERR_DIM if cfg.error_features else 4
        self.terrain_dim = getattr(cfg, "terrain_dim", 0)
        self.mlp = MLP(feat_dim + goal_dim + xi_dim + self.err_dim + self.terrain_dim, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale_vector(pcfg, goal_dim))

    def forward(self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor, err: torch.Tensor, *, terr: torch.Tensor | None = None) -> torch.Tensor:
        x = [feat, goal * self.goal_scale, xi_norm, err]
        if self.terrain_dim:
            if terr is None:
                raise ValueError("this critic takes terrain features: pass terr=")
            x.append(terr)
        return self.mlp(torch.cat(x, dim=-1)).squeeze(-1)
