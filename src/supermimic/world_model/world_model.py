"""World model: learned dynamics + learned termination head + analytic observation/reward heads.

Terrain (docs/VISION_TERRAIN_PLAN.md §2.4): with ``WorldModelCfg.terrain`` = foot | scan | both the dynamics
take the noise-free terrain condition c_t of the *current* (predicted) state, looked up from the map at
every step (``TerrainInputs.wm_condition``), normalised by its own ``terr_norm`` and appended after
``in_norm``; the canonical root height is then taken above the blurred ground. "none" is the flat model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from supermimic.terrain.heightmap import WM_TERRAIN_DIMS
from supermimic.world_model.dynamics import MLP, DynamicsMLP, RunningNorm
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

if TYPE_CHECKING:
    from supermimic.terrain.heightmap import HeightGrid
    from supermimic.terrain.inputs import TerrainInputs


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    term_hidden: tuple[int, ...] = (64, 64)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    term_loss_weight: float = 1.0
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8
    # Options from supermimic_vibed (IMPROVEMENT_PLAN §5): "fixed" nominal input/output scales, or
    # "running" mean/std of the inputs and of the logged one-step increments; LayerNorm after every
    # hidden layer; "learned" (BCE) or "analytic" (sigmoid-softened thresholds) termination head.
    normalization: str = "running"
    layer_norm: bool = True
    termination: str = "analytic"
    term_softness: tuple[float, float] = (0.02, 0.04)  # analytic head: sigmoid widths for height [m] and tilt
    # Terrain condition c_t (set by the trainer from TerrainCfg.wm_terrain): none | foot (28) | scan (187) | both (215).
    terrain: str = "none"


class TerminationHead(nn.Module):
    """P(terminal | state, reference). Input: analytic termination features and their magnitudes."""

    def __init__(self, hidden=(64, 64), height_thr: float = 0.25, tilt_thr: float = 0.8):
        super().__init__()
        self.mlp = MLP(4, tuple(hidden), 1)
        self.register_buffer("scale", torch.tensor([1.0 / height_thr, 1.0 / tilt_thr]))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (…, 2) = [height error, tilt error] -> logits (…,)"""
        x = feats * self.scale
        return self.mlp(torch.cat([x, x.abs()], dim=-1)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeads, device, terrain: TerrainInputs | None = None):
        super().__init__()
        self.cfg = cfg
        self.spec = StateSpec(cfg.state_spec, device)
        mode = getattr(cfg, "terrain", "none")
        if mode not in WM_TERRAIN_DIMS:
            raise ValueError(f"unknown world-model terrain input {mode!r}")
        self.terr_dim = WM_TERRAIN_DIMS[mode]
        self.dynamics = DynamicsMLP(hidden=cfg.hidden, layer_norm=cfg.layer_norm, terr_dim=self.terr_dim)
        self.term_head = TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold)
        self.heads = heads  # analytic observation / reward / FK (no parameters)
        self.running_norm = cfg.normalization == "running"
        if self.running_norm:
            self.in_norm = RunningNorm(71 + 29 + 33, eps=1e-2)
            self.out_norm = RunningNorm(70, eps=1e-4)
        elif cfg.normalization != "fixed":
            raise ValueError(f"unknown world-model normalization {cfg.normalization!r}")
        if self.terr_dim:
            # Separate normaliser: in_norm's shared count freezes at 2e8 samples (plan §2.4, review #1).
            self.terr_norm = RunningNorm(self.terr_dim, eps=1e-2)
            if terrain is None or terrain.wm_mode != mode:
                raise ValueError(f"world model with terrain input {mode!r} needs TerrainInputs(wm_terrain={mode!r})")
        if cfg.termination not in ("learned", "analytic"):
            raise ValueError(f"unknown termination head {cfg.termination!r}")
        self.terrain = terrain  # plain object (map lookups), never a module / buffer
        self.to(device)

    def terrain_inputs(self, s: torch.Tensor, origin: torch.Tensor | None, grid: HeightGrid | None = None):
        """(c_t, ground z) of state s (relative to ``origin``); (None, None) for the flat model."""
        if not self.terr_dim:
            return None, None
        assert origin is not None, "terrain world model: pass the episode origin"
        return self.terrain.wm_condition(s, origin, grid)

    @torch.no_grad()
    def update_normalizers(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor, s_next: torch.Tensor, *, origin: torch.Tensor | None = None, grid: HeightGrid | None = None) -> None:
        """Feed logged transitions (s, a_exec, xi, s') to the running normalisers (input / output: no-op for
        "fixed"; the terrain normaliser always runs)."""
        terr, gz = self.terrain_inputs(s, origin, grid)
        if terr is not None:
            self.terr_norm.update(terr)
        if not self.running_norm:
            return
        x = torch.cat([self.spec.canonical(s, ground_z=gz), action, self.spec.normalize_xi(xi)], dim=-1)
        self.in_norm.update(x)
        self.out_norm.update(self.spec.target_delta(s, s_next))

    # ------------------------------------------------------------------ one step

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor, *, terr: torch.Tensor | None = None, ground_z: torch.Tensor | None = None) -> torch.Tensor:
        """s_{t+1} = integrate(s_t, f(s_t, a_t, xi [, c_t])). ``terr`` = c_t of s_t (raw, normalised here) and
        ``ground_z`` = ground under the pelvis of s_t (``terrain_inputs``); both required iff terrain input."""
        feat = self.spec.canonical(s, ground_z=ground_z)
        if self.terr_dim:
            assert terr is not None and ground_z is not None, "terrain world model: pass terr= and ground_z="
            terr = self.terr_norm(terr)
        if self.running_norm:
            y = self.dynamics(feat, action, self.spec.normalize_xi(xi), self.in_norm, terr=terr)
            return self.spec.integrate_physical(s, y * self.out_norm.std() + self.out_norm.mean)
        delta = self.dynamics(feat, action, self.spec.normalize_xi(xi), terr=terr)
        return self.spec.integrate(s, delta)

    def step_at(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor, origin: torch.Tensor | None = None, grid: HeightGrid | None = None) -> torch.Tensor:
        """``step`` with the terrain condition looked up at s (re-queried from the map at every step)."""
        terr, gz = self.terrain_inputs(s, origin, grid)
        return self.step(s, action, xi, terr=terr, ground_z=gz)

    def _term_logits_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        if self.cfg.termination == "learned":
            return self.term_head(feats)
        # Analytic: P(terminal) = 1 - (1 - sig((|e_z| - 0.25) / wz)) (1 - sig((|e_tilt| - 0.8) / wt)).
        wz, wt = self.cfg.term_softness
        alive = torch.sigmoid(-(feats[..., 0].abs() - self.cfg.height_threshold) / wz) * torch.sigmoid(-(feats[..., 1].abs() - self.cfg.tilt_threshold) / wt)
        return torch.logit(1.0 - alive, eps=1e-6)

    def termination_logits(self, s_next: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        # The pelvis is the free-joint body, so its pose is the state's root pose: no FK needed.
        feats = self.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ref)
        return self._term_logits_from_features(feats)

    # ------------------------------------------------------------------ Eq. 1

    def _window(self, states, a_exec, xi, done, ref_root_pos, ref_root_quat, origin=None, grid=None):
        """k-step autoregressive pass of Eq. 1 (one graph, compiled as a whole like supermimic_vibed's
        window rollout). Returns (loss, state_loss, term_bce, nse_1step, nse_kstep, term_acc). With terrain
        input the condition is recomputed from the *predicted* state at every step."""
        k = a_exec.shape[0]
        s = states[0]
        total_state = total_term = acc = 0.0
        nse = []
        for j in range(k):
            s = self.step_at(s, a_exec[j], xi, origin, grid)
            e = self.spec.error_vector(s, states[j + 1])
            sq = e * e
            feats = self.heads.termination_features(s[..., 0:3], s[..., 3:7], ReferenceFrame(None, None, None, None, ref_root_pos[j], ref_root_quat[j]))
            logits = self._term_logits_from_features(feats)
            total_state = total_state + sq.sum(-1).mean()
            total_term = total_term + F.binary_cross_entropy_with_logits(logits, done[j])
            nse.append(sq.detach().mean())
            acc = acc + ((logits.detach() > 0).float() == done[j]).float().mean()
        loss = total_state / k + self.cfg.term_loss_weight * total_term / k
        return loss, (total_state / k).detach(), (total_term / k).detach(), nse[0], nse[-1], acc / k

    def enable_compile(self, mode: str | None = "default") -> None:
        self._window_fn = torch.compile(self._window, mode=None if mode == "default" else mode, dynamic=False) if mode else self._window

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, done: torch.Tensor, ref_root_pos: torch.Tensor, ref_root_quat: torch.Tensor, *, origin: torch.Tensor | None = None, grid: HeightGrid | None = None):
        """k-step autoregressive loss.

        states: (k+1, n, 71) logged states; a_exec: (k, n, 29); xi: (n, 33); done: (k, n) termination
        label of states 1..k; ref_root_pos / ref_root_quat: (k, n, 3/4) reference root of transition j
        (for the termination features of state j+1; terrain: lifted); origin: (n, 3) episode origin of the
        windows (terrain input only); grid: the map tensors.
        """
        fn = getattr(self, "_window_fn", None) or self._window
        if self.terr_dim:
            out = fn(states, a_exec, xi, done, ref_root_pos, ref_root_quat, origin, grid)
        else:
            out = fn(states, a_exec, xi, done, ref_root_pos, ref_root_quat)
        loss, state_loss, term_bce, nse1, nsek, acc = out
        # Metrics stay on-device (detached tensors): a float() here would stall the CPU on the
        # forward pass every minibatch, which in data-parallel runs turns into lockstep waiting.
        return loss, dict(state_loss=state_loss, term_bce=term_bce, nse_1step=nse1, nse_kstep=nsek, term_acc=acc)

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def window_errors(self, states, a_exec, xi, ref_root_pos, ref_root_quat, *, origin=None, grid=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-window version of ``_window``'s metrics (same arguments as ``window_loss``): open-loop normalised
        squared error (k, n) of each predicted state 1..k and its termination logits (k, n)."""
        k = a_exec.shape[0]
        s = states[0]
        err, logits = [], []
        for j in range(k):
            s = self.step_at(s, a_exec[j], xi, origin, grid)
            err.append(self.spec.normalized_sq_error(s, states[j + 1]))
            feats = self.heads.termination_features(s[..., 0:3], s[..., 3:7], ReferenceFrame(None, None, None, None, ref_root_pos[j], ref_root_quat[j]))
            logits.append(self._term_logits_from_features(feats))
        return torch.stack(err), torch.stack(logits)

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, *, origin: torch.Tensor | None = None, grid: HeightGrid | None = None) -> torch.Tensor:
        """Open-loop k-step normalised squared error per step (k,), averaged over the batch."""
        k = a_exec.shape[0]
        s = states[0]
        out = []
        for j in range(k):
            s = self.step_at(s, a_exec[j], xi, origin, grid)
            out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        return torch.stack(out)
