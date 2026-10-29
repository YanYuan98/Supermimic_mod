"""Short-horizon actor–critic through the frozen world model (paper §3.3, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.rollout import ImagRollout, imagine
from supermimic.policy.nets import Critic, Policy
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.world_model import WorldModel

if TYPE_CHECKING:
    from supermimic.terrain.depth import DepthSensor
    from supermimic.terrain.inputs import TerrainInputs


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    history: int = 10
    future_frames: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = "default"  # None | "default" | "reduce-overhead" | "max-autotune-no-cudagraphs"
    anchor_obs: str = "ori"  # set from EnvCfg by the trainer
    goal_features: str = "v1"  # set from EnvCfg by the trainer
    # "sum": loss = -J (Eq. 2, batch mean) so Table A.5's clipping acts on the paper's objective;
    # "per_step": loss = -J / H (the v1 behaviour).
    objective_scale: str = "sum"
    # supermimic_vibed option: only start imagined rollouts at states whose target frame is at least
    # ``start_margin`` frames before the clip end (0 = any state; past the end the last frame is held).
    start_margin: int = 32
    # Depth policy: weight of the auxiliary scan regression (visible cells only) added to the actor loss.
    aux_weight: float = 0.1


class PolicyTrainer:
    def __init__(
        self,
        wm: WorldModel,
        policy: Policy,
        critic: Critic,
        lib: MotionLibrary,
        cfg: PolicyTrainerCfg,
        *,
        terrain: TerrainInputs | None = None,
        depth: DepthSensor | None = None,
    ):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        self.terrain = terrain
        self.depth = depth
        fused = torch.cuda.is_available()
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas, fused=fused)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, fused=fused)
        from supermimic.imagination.compiled import ImagHeads, ImagStep

        self.step_fn = ImagStep(wm, policy, lib.default_joint_pos, anchor_obs=cfg.anchor_obs, obs_noise=cfg.obs_noise, compile_mode=cfg.compile, terrain=terrain, depth=depth)
        self.heads_fn = ImagHeads(wm, critic_features=critic.err_dim > 4, compile_mode=cfg.compile, terrain=terrain)

    def set_lr(self, lr: float, critic_lr: float | None = None) -> None:
        """Policy LR ``lr``; critic LR ``critic_lr`` (None: the same as the policy)."""
        for g in self.opt.param_groups:
            g["lr"] = lr
        for g in self.critic_opt.param_groups:
            g["lr"] = lr if critic_lr is None else critic_lr

    def _critic_inputs(self, ro: ImagRollout):
        """(canonical state, goals, tracking errors, terrain features | None), each (H+1, n, ·). With terrain
        the canonical root height is taken above the ground."""
        H1, n, _ = ro.states.shape
        gz = None if ro.ground_z is None else ro.ground_z.reshape(H1 * n)
        feat = self.wm.spec.canonical(ro.states.reshape(H1 * n, -1), ground_z=gz).reshape(H1, n, -1)
        return feat, ro.goals, ro.crit_feat, ro.crit_terr

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        cfg = self.cfg
        extra = None
        if cfg.start_margin > 0:
            extra = buffer.frame + cfg.start_margin <= self.lib.last_frame_of(buffer.frame)
        n_hist = cfg.history if self.depth is None else max(cfg.history, self.depth.cfg.start_hist)
        start = buffer.sample_starts(cfg.batch_size, n_hist, extra_mask=extra)
        xi_norm = self.wm.spec.normalize_xi(start.xi)

        # ---- actor: BPTT through the frozen world model ------------------------------------
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        ro = imagine(self.lib, start, cfg.horizon, self.step_fn, self.heads_fn, cfg.future_frames, cfg.goal_features, obs_history=cfg.history)
        feat, goals, err, terr = self._critic_inputs(ro)
        if cfg.use_critic:
            v_H = self.critic(feat[-1], goals[-1], xi_norm, err[-1], terr=None if terr is None else terr[-1])
        else:
            v_H = None
        J = policy_objective(ro, v_H, cfg.gamma)
        loss = -J if cfg.objective_scale == "sum" else -J / cfg.horizon
        aux = None
        if ro.aux_h is not None:
            # Auxiliary scan regression of the depth modules: noise-free scan of s_t at the cells the slot cameras saw.
            pred = self.policy.depth.aux(ro.aux_h)
            target = ro.crit_terr[:-1, :, -pred.shape[-1] :].detach()
            m = ro.aux_vis.to(pred.dtype)
            aux = (m * (pred - target).square()).sum() / m.sum().clamp_min(1.0)
            loss = loss + cfg.aux_weight * aux
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        allreduce_grads(self.policy)
        depth_gn = None
        if self.policy.depth is not None:
            grads = [p.grad for p in self.policy.depth.encoder.parameters() if p.grad is not None]
            depth_gn = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])) if grads else torch.zeros((), device=loss.device)
        gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
        self.opt.step()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)

        # Metrics stay on-device until one host sync at the end of the update.
        metrics = {
            "imag_objective": J.detach(),
            "imag_return_undisc": (ro.survival[:-1] * ro.rewards).sum(0).mean().detach(),
            "imag_reward_mean": ro.rewards.mean().detach(),
            "imag_survival_H": ro.survival[-1].mean().detach(),
            "imag_dhat_mean": ro.dhat[1:].mean().detach(),
            "policy_grad_norm": gn.detach(),
            "action_abs_mean": ro.actions.abs().mean().detach(),
        }
        for k, v in ro.reward_terms.items():
            metrics[f"imag_r/{k}"] = v
        if aux is not None:
            with torch.no_grad():
                h = ro.aux_h.reshape(-1, ro.aux_h.shape[-1])
                z = torch.tanh(self.policy.depth.agg_out(h.float()))
                zstd = z.std(0)
                metrics["depth/aux_loss"] = aux.detach()
                metrics["depth/aux_visible_frac"] = ro.aux_vis.float().mean()
                metrics["depth/z_std_mean"] = zstd.mean()
                metrics["depth/z_dead_frac"] = (zstd < 1e-3).float().mean()  # constant latent dims
                metrics["depth/z_sat_frac"] = (z.abs() > 0.99).float().mean()
                metrics["depth/h_inactive_frac"] = (h.std(0) < 1e-4).float().mean()
                metrics["depth/encoder_grad_norm"] = depth_gn

        # ---- critic: fit to GAE targets of this rollout (targets fixed) ---------------------
        if cfg.use_critic:
            with torch.no_grad():
                feat_d, goals_d, err_d = feat.detach(), goals.detach(), err.detach()
                H1, n, _ = feat_d.shape
                xi_rep = xi_norm[None].expand(H1, n, -1)
                terr_d = None if terr is None else terr.detach().reshape(H1 * n, -1)
                values = self.critic(feat_d.reshape(H1 * n, -1), goals_d.reshape(H1 * n, -1), xi_rep.reshape(H1 * n, -1), err_d.reshape(H1 * n, -1), terr=terr_d).reshape(H1, n)
                targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda)  # (H, n)
                X_feat = feat_d[:-1].reshape(-1, feat_d.shape[-1])
                X_goal = goals_d[:-1].reshape(-1, goals_d.shape[-1])
                X_xi = xi_rep[:-1].reshape(-1, xi_rep.shape[-1])
                X_err = err_d[:-1].reshape(-1, err_d.shape[-1])
                X_terr = None if terr_d is None else terr_d[: (H1 - 1) * n]
                Y = targets.reshape(-1)
            N = Y.shape[0]
            mb = N // cfg.critic_minibatches
            last = torch.zeros((), device=Y.device)
            for _ in range(cfg.critic_epochs):
                perm = torch.randperm(N, device=Y.device)
                for i in range(cfg.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    pred = self.critic(X_feat[idx], X_goal[idx], X_xi[idx], X_err[idx], terr=None if X_terr is None else X_terr[idx])
                    closs = torch.nn.functional.mse_loss(pred, Y[idx])
                    self.critic_opt.zero_grad(set_to_none=True)
                    closs.backward()
                    allreduce_grads(self.critic)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip)
                    self.critic_opt.step()
                    last = closs.detach()
            metrics["critic_loss"] = last
            metrics["critic_value_mean"] = values.mean()
            metrics["critic_target_mean"] = targets.mean()
        keys = list(metrics)
        vals = torch.stack([metrics[k].float().reshape(()) for k in keys]).tolist()
        return dict(zip(keys, vals))
