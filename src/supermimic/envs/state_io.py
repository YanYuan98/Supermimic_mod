"""Read / write the full simulator state (qpos | qvel) as one 71-d tensor."""

from __future__ import annotations

import torch

STATE_DIM = 36 + 35
NQ = 36
NV = 35


class StateIO:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        assert self.sim.data.qpos.shape[-1] == NQ and self.sim.data.qvel.shape[-1] == NV, (
            self.sim.data.qpos.shape,
            self.sim.data.qvel.shape,
        )

    @property
    def origins(self) -> torch.Tensor:
        """(N, 3) live episode origins. States are stored relative to each env's origin so that they are
        directly comparable with the (origin-free) motion library and translation-invariant across envs;
        with terrain the motion command rewrites ``env_origins`` (o_ep) at every reset."""
        return self.env.scene.env_origins

    def read(self) -> torch.Tensor:
        """(N, 71) copy of [qpos | qvel]."""
        # Index with [:] to get plain tensors out of mjlab's TorchArray proxies (torch.cat on the
        # proxies recurses inside their __torch_function__).
        s = torch.cat([self.sim.data.qpos[:], self.sim.data.qvel[:]], dim=-1).clone()
        s[:, 0:3] -= self.origins
        return s

    @torch.no_grad()
    def write(self, env_ids: torch.Tensor, state: torch.Tensor, forward: bool = True) -> None:
        """Overwrite qpos/qvel of ``env_ids`` and refresh derived quantities."""
        state = state.to(self.sim.data.qpos.dtype).clone()
        state[:, 0:3] += self.origins[env_ids]
        self.sim.data.qpos[env_ids] = state[:, :NQ]
        self.sim.data.qvel[env_ids] = state[:, NQ:]
        if hasattr(self.sim.data, "qacc_warmstart"):
            self.sim.data.qacc_warmstart[env_ids] = 0.0
        if forward:
            self.sim.forward()
