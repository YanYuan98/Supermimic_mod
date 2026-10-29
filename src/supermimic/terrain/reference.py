"""Spawn sampling, spawn lift and terrain-lifted references (plan §2.2).

Coordinates: the episode origin ``o_ep = (p_spawn.xy - lib_root_xy(start_frame), 0)`` is the only
owner of the reference-to-world translation; Phase B writes it into ``env.scene.env_origins``. Library
references and states are origin-relative ("local"); world = o_ep + local. Since o_ep.z = 0, local z
is world z, and the reference is lifted by the blurred ground under its root:
``lift(t) = g(o_ep.xy + lib_root_xy(t))`` added to every body / root z of frame t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.terrain.heightmap import HeightGrid, TerrainMap
    from supermimic.terrain.presets import TerrainLayout
    from supermimic.world_model.heads import ReferenceFrame

FPS = 50.0  # motion library frame rate (= control rate)


def default_window_frames(episode_length_s: float = 10.0, horizon: int = 32, fps: float = FPS) -> int:
    """Frames a spawn must cover: one episode plus the imagination horizon (500 + 32 = 532)."""
    return round(episode_length_s * fps) + horizon


def lib_root_xy(lib: MotionLibrary, frames: torch.Tensor) -> torch.Tensor:
    """(..., 2) library (origin-free) root xy of the given frames."""
    return lib.ref(frames).root_pos_w[..., 0:2]


# ---------------------------------------------------------------------------------------------- path bbox


def _window_reduce(x: torch.Tensor, last: torch.Tensor, length: int, op) -> torch.Tensor:
    """out[i] = op over x[i .. min(i + length - 1, last[i])] (clip-clamped forward window) in O(F log W):
    doubling of power-of-two windows, combined along the binary expansion of ``length``."""
    idx = torch.arange(x.shape[0], device=x.device)
    level = x  # level[i] = op over [i, min(i + 2^k - 1, last[i])]
    out = None
    offset = 0
    k = 0
    while (1 << k) <= length:
        if length & (1 << k):
            v = level[torch.minimum(idx + offset, last)]
            out = v if out is None else op(out, v)
            offset += 1 << k
        if (1 << (k + 1)) <= length:
            level = op(level, level[torch.minimum(idx + (1 << k), last)])
        k += 1
    assert out is not None
    return out


@torch.no_grad()
def path_bbox(lib: MotionLibrary, window_frames: int) -> torch.Tensor:
    """(num_frames, 4) = [x_lo, y_lo, x_hi, y_hi] of the root-xy path over frames [f, min(f + W, clip_last)],
    relative to the root xy at f (so x_lo, y_lo <= 0 <= x_hi, y_hi). O(F log W) memory-light sliding min/max."""
    xy = lib.root_pos_w[:, 0:2].to(torch.float32)
    last = lib.clip_last[lib.frame_clip]
    n = window_frames + 1  # inclusive window [f, f + W]
    lo = torch.stack([_window_reduce(xy[:, i].contiguous(), last, n, torch.minimum) for i in range(2)], -1)
    hi = torch.stack([_window_reduce(xy[:, i].contiguous(), last, n, torch.maximum) for i in range(2)], -1)
    return torch.cat([lo - xy, hi - xy], dim=-1)


# ---------------------------------------------------------------------------------------------- spawn


@dataclass
class SpawnSample:
    origin: torch.Tensor  # (N, 3) episode origin o_ep = (p_spawn.xy - lib_root_xy(start), 0)
    spawn_xy: torch.Tensor  # (N, 2) world xy of the reference root at the start frame
    row: torch.Tensor  # (N,) long tile row (difficulty)
    col: torch.Tensor  # (N,) long tile column (terrain type)
    fits: torch.Tensor  # (N,) bool: the path bbox + margin fits in the tile (else centred, best effort)


def sample_spawn(
    lib: MotionLibrary,
    start_frame: torch.Tensor,
    bbox: torch.Tensor,
    layout: TerrainLayout,
    level: int | torch.Tensor,
    margin: float = 0.5,
    generator: torch.Generator | None = None,
    row: torch.Tensor | None = None,
    col: torch.Tensor | None = None,
    centred: torch.Tensor | bool = False,
) -> SpawnSample:
    """Column ~ type proportions, row ~ U{0..L} (L = curriculum level, clamped to the grid), then a spawn
    xy uniform over the positions that keep the reference path bbox (+ margin) inside the tile. Pass
    ``row`` / ``col`` to pin the tile (evaluation per type / difficulty; negative entries are sampled) and
    ``centred`` (bool or (N,) mask) to centre the path bbox in the tile instead (deterministic placement).
    Host-sync free."""
    device = start_frame.device
    n = start_frame.shape[0]
    p = torch.tensor(layout.proportions, device=device, dtype=torch.float32)
    sampled = torch.multinomial(p, n, replacement=True, generator=generator)
    col = sampled if col is None else torch.where(col >= 0, col, sampled)
    lv = torch.as_tensor(level, device=device).clamp(0, layout.num_rows - 1)
    u = torch.rand(n, device=device, generator=generator)
    sampled = torch.minimum((u * (lv + 1)).long(), lv.long())
    row = sampled if row is None else torch.where(row >= 0, row, sampled)
    centre = layout.tile_centres(device)[row, col]  # (N, 2)
    half = torch.tensor(layout.tile_size, device=device, dtype=torch.float32) * 0.5
    b = bbox[start_frame]  # (N, 4) relative to the start root xy
    lo_rel, hi_rel = b[:, 0:2], b[:, 2:4]
    p_min = centre - half + margin - lo_rel
    p_max = centre + half - margin - hi_rel
    fits = (p_max >= p_min).all(-1)
    u = torch.rand(n, 2, device=device, generator=generator)
    mid = centre - 0.5 * (lo_rel + hi_rel)  # path bbox centred in the tile
    spawn = torch.where((fits & ~torch.as_tensor(centred, device=device))[:, None], p_min + u * (p_max - p_min), mid)
    origin = torch.cat([spawn - lib_root_xy(lib, start_frame), torch.zeros(n, 1, device=device)], dim=-1)
    return SpawnSample(origin, spawn, row, col, fits)


@torch.no_grad()
def spawn_lift(qpos: torch.Tensor, origin: torch.Tensor, fk: G1Kinematics, tmap: TerrainMap, margin: float = 0.01) -> torch.Tensor:
    """(N,) extra root z so no sole point of the spawn pose penetrates the terrain:
    max(0, max_i(h(p_i) - z_i)) + margin. ``qpos`` (N, >=36) is the (lifted, noised) spawn pose relative to
    ``origin`` (N, 3). The penetration uses ``TerrainMap.height_upper`` (never below bilinear h) so sole
    points just on the high side of a step edge are not under-estimated. Plane mode returns 0 (keeps
    main's spawn for plane regression)."""
    if tmap.is_plane:
        return torch.zeros(qpos.shape[:-1], device=qpos.device, dtype=qpos.dtype)
    sole = tmap.foot_points(qpos, origin, fk).sole.flatten(-3, -2)  # (N, 10, 3)
    pen = tmap.height_upper(sole[..., 0:2]) - sole[..., 2]
    return pen.amax(-1).clamp_min(0.0) + margin


# ---------------------------------------------------------------------------------------------- lifted reference


def reference_lift(
    lib: MotionLibrary,
    frames: torch.Tensor,
    origin: torch.Tensor,
    tmap: TerrainMap,
    grid: HeightGrid | None = None,
    with_velocity: bool = False,
    dt: float = 1.0 / FPS,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """lift(t) = g(o.xy + lib_root_xy(t)), shape = frames.shape (origin (..., 3) broadcastable).
    With ``with_velocity`` also the vertical velocity (lift(t+1) - lift(t)) / dt (clip-end clamped)."""
    o = origin[..., 0:2]
    lift = tmap.ground(o + lib_root_xy(lib, frames), grid)
    if not with_velocity:
        return lift
    nxt = torch.minimum(frames + 1, lib.last_frame_of(frames))
    v = (tmap.ground(o + lib_root_xy(lib, nxt), grid) - lift) / dt
    return lift, v


def lifted_reference(
    lib: MotionLibrary,
    frames: torch.Tensor,
    origin: torch.Tensor,
    tmap: TerrainMap,
    grid: HeightGrid | None = None,
    vertical_velocity: bool = False,
    dt: float = 1.0 / FPS,
) -> ReferenceFrame:
    """``wm_trainer.reference_at`` with every body / root z lifted by ``lift(t)`` (uniform per frame).
    frames (...), origin (..., 3) broadcastable. ``vertical_velocity`` adds d lift / dt to every body's
    linear velocity z (off by default; if enabled it must also enter the spawn root_lin_vel)."""
    from supermimic.world_model.heads import ReferenceFrame

    r = lib.ref(frames)
    b = lib.bodies(frames)
    out = reference_lift(lib, frames, origin, tmap, grid, vertical_velocity, dt)
    lift, v = out if vertical_velocity else (out, None)
    zero = torch.zeros_like(lift)
    dz = torch.stack([zero, zero, lift], dim=-1)  # (..., 3)
    lin_vel = b.lin_vel_w
    if v is not None:
        lin_vel = lin_vel + torch.stack([zero, zero, v], dim=-1)[..., None, :]
    return ReferenceFrame(b.pos_w + dz[..., None, :], b.quat_w, lin_vel, b.ang_vel_w, r.root_pos_w + dz, r.root_quat_w, r.joint_pos)
