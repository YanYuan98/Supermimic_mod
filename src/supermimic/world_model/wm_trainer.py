"""World-model fitting on k-step replay windows (paper Eq. 1, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.world_model.heads import ReferenceFrame
from supermimic.training.distributed import allreduce_buffers, allreduce_grads
from supermimic.world_model.world_model import WorldModel

if TYPE_CHECKING:
    from supermimic.terrain.inputs import TerrainInputs


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    compile: str | None = "default"


def reference_at(lib: MotionLibrary, frame: torch.Tensor) -> ReferenceFrame:
    r = lib.ref(frame)
    b = lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w, r.joint_pos)


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg, *, terrain: TerrainInputs | None = None):
        self.wm = wm
        self.lib = lib
        self.cfg = cfg
        self.terrain = terrain  # terrain env: lifted references + the world model's terrain condition
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=torch.cuda.is_available())
        wm.enable_compile(cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def _terrain_kw(self, w: WindowBatch) -> dict:
        return {} if self.terrain is None else {"origin": w.origin, "grid": self.terrain.grid}

    def reference_roots(self, w: WindowBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """(k, n, 3 / 4) reference root pose of each transition (terrain: lifted by the ground under it). Only the
        root pose of the references is needed for the termination features."""
        k, n = w.frame.shape
        r = self.lib.ref(w.frame.reshape(-1))
        root_pos = r.root_pos_w.reshape(k, n, 3)
        if self.terrain is not None and not self.terrain.is_plane and getattr(self.wm.cfg, "reference_lift", "reference") != "robot":
            lift = self.terrain.root_lift(self.lib, w.frame, w.origin[None], self.terrain.grid)  # (k, n)
            root_pos = root_pos + torch.nn.functional.pad(lift[..., None], (2, 0))
        return root_pos, r.root_quat_w.reshape(k, n, 4)

    def loss_on(self, w: WindowBatch):
        root_pos, root_quat = self.reference_roots(w)
        return self.wm.window_loss(w.states, w.a_exec, w.xi, w.done, root_pos, root_quat, **self._terrain_kw(w))

    def _minibatch(self, w: WindowBatch, agg: dict[str, torch.Tensor]) -> None:
        """One gradient step on window batch ``w``; adds its (device) metrics to ``agg``."""
        self.wm.update_normalizers(w.states[0], w.a_exec[0], w.xi, w.states[1], **self._terrain_kw(w))
        loss, m = self.loss_on(w)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        allreduce_grads(self.wm)
        if self.wm.running_norm:
            allreduce_buffers(self.wm)
        gn = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.grad_clip)
        self.opt.step()
        m["loss"] = loss.detach()
        m["grad_norm"] = gn.detach()
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        mask = buffer.window_mask(self.cfg.window)  # shared by the batches of this update
        for _ in range(self.cfg.batches_per_update):
            self._minibatch(buffer.sample_windows(self.cfg.batch_size, self.cfg.window, mask=mask), agg)
        # Single host sync per update (keeps the CPU queuing GPU work across minibatches).
        return {k: float(v) / self.cfg.batches_per_update for k, v in agg.items()}

    def update_on(self, batches: list[WindowBatch], sync: bool = True) -> dict[str, float] | dict[str, torch.Tensor]:
        """``update`` on given window batches (e.g. the same batches for several world models, M2 ablation).
        ``sync=False`` returns the mean metrics as device tensors (no host sync: models on other GPUs keep running)."""
        agg: dict[str, torch.Tensor] = {}
        for w in batches:
            self._minibatch(w, agg)
        n = max(len(batches), 1)
        return {k: (float(v) if sync else v) / n for k, v in agg.items()}

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on (recent) replay windows; (k,) tensor."""
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi, **self._terrain_kw(w))
