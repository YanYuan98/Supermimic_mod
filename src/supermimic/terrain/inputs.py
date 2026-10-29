"""Terrain inputs of the learned models (plan §0, §2.3-2.6), all pure functions of (state, episode origin, map):

* world model: the noise-free condition c_t (``wm_terrain``: foot 28 | scan 187 | both 215) and the blurred
  ground under the pelvis (``StateSpec.canonical(s, ground_z=...)`` then uses the height above it);
* policy: the scan queried at the episode's shifted points (``ScanNoiseParams.offset_xy``), pushed into a short
  history and corrupted by ``noisy_scan`` (latency, z offset, point noise, dropout) - the sim side uses the
  same functions (``MultiClipMotionCommand``);
* critic: noise-free [foot features, scan] (215) and the ground height;
* references: ``lifted_reference`` / ``reference_lift``.

A plain object (never a module buffer, see ``TerrainMap``); compiled callers pass the map tensors (``grid``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from supermimic.terrain.heightmap import FOOT_FEAT_DIM, SCAN_DIM, WM_TERRAIN_DIMS, HeightGrid, ScanNoiseCfg, ScanNoiseParams, TerrainMap
from supermimic.terrain.reference import lifted_reference, reference_lift

if TYPE_CHECKING:
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import ReferenceFrame

CRITIC_TERRAIN_DIM = FOOT_FEAT_DIM + SCAN_DIM  # 215


class TerrainInputs:
    def __init__(self, tmap: TerrainMap, fk: G1Kinematics, wm_terrain: str = "foot", scan_noise: ScanNoiseCfg | None = None, grad: bool = False):
        if wm_terrain not in WM_TERRAIN_DIMS:
            raise ValueError(f"unknown wm_terrain {wm_terrain!r}")
        self.tmap, self.fk, self.wm_mode = tmap, fk, wm_terrain
        self.scan_noise = scan_noise if scan_noise is not None else ScanNoiseCfg()
        self.detach = not grad

    @property
    def grid(self) -> HeightGrid | None:
        return self.tmap.grid

    @property
    def is_plane(self) -> bool:
        return self.tmap.is_plane

    @property
    def wm_dim(self) -> int:
        return WM_TERRAIN_DIMS[self.wm_mode]

    def ground_z(self, s: torch.Tensor, origin: torch.Tensor, grid: HeightGrid | None = None) -> torch.Tensor:
        """(...) blurred ground g under the pelvis of s (..., >=2; relative to origin (..., 3))."""
        return self.tmap.ground(s[..., 0:2] + origin[..., 0:2], grid, self.detach)

    def wm_condition(self, s: torch.Tensor, origin: torch.Tensor, grid: HeightGrid | None = None, scan_clean: torch.Tensor | None = None):
        """(c_t, ground z) of state s for the world model; (None, None) for ``wm_terrain="none"`` (flat WM)."""
        if self.wm_mode == "none":
            return None, None
        c = self.tmap.wm_condition(self.wm_mode, s, origin, self.fk, grid, scan_clean, self.detach)
        return c, self.ground_z(s, origin, grid)

    def critic_features(self, s: torch.Tensor, origin: torch.Tensor, grid: HeightGrid | None = None):
        """((..., 215) noise-free [foot features 28, scan 187], ground z (...))."""
        foot = self.tmap.foot_features(s, origin, self.fk, grid, self.detach)
        scan = self.tmap.scan_state(s, origin, grid, None, self.detach)
        return torch.cat([foot, scan], dim=-1), self.ground_z(s, origin, grid)

    def policy_scan_query(self, s: torch.Tensor, origin: torch.Tensor, noise: ScanNoiseParams, grid: HeightGrid | None = None) -> torch.Tensor:
        """(..., 187) scan at the episode's shifted query points: the entry pushed into the policy's scan history."""
        return self.tmap.scan_state(s, origin, grid, noise.offset_xy, self.detach)

    def reference(self, lib: MotionLibrary, frames: torch.Tensor, origin: torch.Tensor, grid: HeightGrid | None = None) -> ReferenceFrame:
        return lifted_reference(lib, frames, origin, self.tmap, grid)

    def root_lift(self, lib: MotionLibrary, frames: torch.Tensor, origin: torch.Tensor, grid: HeightGrid | None = None) -> torch.Tensor:
        return reference_lift(lib, frames, origin, self.tmap, grid)
