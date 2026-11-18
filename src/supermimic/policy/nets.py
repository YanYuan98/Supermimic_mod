"""Policy (goal encoder + shared decoder) and critic (paper Table A.4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from supermimic.world_model.dynamics import MLP, RunningNorm
from supermimic.world_model.heads import CRITIC_ERR_DIM, CRITIC_ERR_DIM_VIBED, OBS_TERMS

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


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (256, 128)
    # Privileged tracking-error features (IMPROVEMENT_PLAN F5); False = root errors only (4).
    error_features: bool = True
    # v5 ablation D1: "v4" (torso-anchored, 84) or "vibed" (pelvis-anchored as supermimic_vibed:
    # joint error 29 + body position error 42 + root position error 3 + relative orientation 6 = 80).
    features: str = "v4"
    # v5 ablation C1: "fixed" (per-block fixed scales) or "running" (running mean/std of the whole
    # input, updated from the imagined critic inputs, as supermimic_vibed); activation "silu" | "elu".
    input_norm: str = "fixed"
    activation: str = "silu"


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor:
    parts = []
    for (name, d), s in zip(OBS_TERMS, cfg.obs_term_scale):
        parts.append(torch.full((cfg.history * d,), 1.0 / s))
    return torch.cat(parts)  # term-major layout matches the flattened history


def _goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor:
    sizes = (29, 29, 3, 3, 3, 1)
    frame_dim = getattr(cfg, "goal_frame_dim", 68)
    if frame_dim == 73:  # v5 "vibed" goal: no v1 block (joint velocities are already scaled by 0.1)
        per_frame = torch.ones(73)
    else:
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


class Decoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, obs_dim: int = 930):
        super().__init__()
        self.mlp = MLP(cfg.latent_dim + obs_dim, cfg.decoder_hidden, ACT_DIM, act=getattr(cfg, "activation", "silu"), last_gain=cfg.decoder_last_gain)
        self.register_buffer("scale", _obs_scale_vector(cfg))
        self.norm = RunningNorm(obs_dim) if _running(cfg) else None

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z, self.norm(obs_hist) if self.norm is not None else obs_hist * self.scale], dim=-1))


class Policy(nn.Module):
    """a = dec(enc_motion(g), o_hist). Extra goal interfaces can be registered later (plan Appendix B)."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({"motion": MotionEncoder(cfg)})
        self.decoder = Decoder(cfg)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion") -> torch.Tensor:
        return self.decoder(self.encoders[interface](goal), obs_hist)

    @torch.no_grad()
    def update_input_norms(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion") -> None:
        """Feed simulator observations to the running input normalisers (no-op for "fixed")."""
        enc = self.encoders[interface]
        if enc.norm is not None:
            enc.norm.update(goal)
        if self.decoder.norm is not None:
            self.decoder.norm.update(obs_hist)


class Critic(nn.Module):
    """V(canonical state 71, goal, xi 33, privileged tracking errors 84 | root error 4) -> scalar (one per interface)."""

    def __init__(self, cfg: CriticCfg, feat_dim: int = 71, goal_dim: int | None = None, xi_dim: int = 33, pcfg: PolicyCfg | None = None):
        super().__init__()
        pcfg = pcfg or PolicyCfg()
        goal_dim = goal_dim or pcfg.goal_dim
        features = getattr(cfg, "features", "v4")
        if features not in ("v4", "vibed"):
            raise ValueError(f"unknown critic features {features!r}")
        self.feature_mode = ("vibed" if features == "vibed" else "v4") if cfg.error_features else "root"
        self.err_dim = {"v4": CRITIC_ERR_DIM, "vibed": CRITIC_ERR_DIM_VIBED, "root": 4}[self.feature_mode]
        in_dim = feat_dim + goal_dim + xi_dim + self.err_dim
        self.mlp = MLP(in_dim, cfg.hidden, 1, act=getattr(cfg, "activation", "silu"))
        self.register_buffer("goal_scale", _goal_scale_vector(pcfg, goal_dim))
        mode = getattr(cfg, "input_norm", "fixed")
        if mode not in ("fixed", "running"):
            raise ValueError(f"unknown critic input_norm {mode!r}")
        self.norm = RunningNorm(in_dim) if mode == "running" else None

    def _inputs(self, feat, goal, xi_norm, err) -> torch.Tensor:
        if self.norm is not None:
            return torch.cat([feat, goal, xi_norm, err], dim=-1)
        return torch.cat([feat, goal * self.goal_scale, xi_norm, err], dim=-1)

    @torch.no_grad()
    def update_norm(self, feat, goal, xi_norm, err) -> None:
        if self.norm is not None:
            self.norm.update(self._inputs(feat, goal, xi_norm, err))

    def forward(self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        x = self._inputs(feat, goal, xi_norm, err)
        return self.mlp(self.norm(x) if self.norm is not None else x).squeeze(-1)
