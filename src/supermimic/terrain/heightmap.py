"""Global terrain height map (plan §2.1, §2.3, §2.4).

The terrain is static during an episode, so everything the robot "sees" of it is a pure function of
(terrain, state). The map is rasterised once at start-up from the compiled scene with vertical rays
(MuJoCo Warp, a standalone nworld=1 ``RenderContext`` with geom group 0 only), cached on disk, and
afterwards only looked up with a 4-corner gather + bilinear interpolation.

``TerrainMap`` is a plain Python object, never an ``nn.Module`` / buffer: ``training/distributed.py``
broadcasts every module buffer from rank 0 and all-reduces float buffers after each WM minibatch.
The lookup functions are host-sync free and ``torch.compile(dynamic=False)`` friendly; the map
tensors (``HeightGrid``) can be passed explicitly as function inputs.

Conventions:
  * world xy; states s = [qpos(36) | qvel(35)] have their root position relative to the episode
    origin ``o`` (``StateIO``), so world = o + local;
  * the grid stores cell-centre samples: cell (i, j) covers [x0 + i res, x0 + (i+1) res) x [...];
    step edges built from 0.3 m treads on the 12 m tile grid fall exactly on cell boundaries;
  * outside the rasterised extent (tile grid + outer border) every lookup returns 0;
  * channels: ``h`` raw height (scan, foot clearances), ``g`` Gaussian-blurred ground (sigma 0.15 m;
    reference lift, root height above ground, terminations), both fp16;
  * plane mode: h = g = 0 everywhere, nothing is rasterised, all APIs still work.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import mujoco
import numpy as np
import torch

from supermimic.kinematics.rotations import quat_apply

if TYPE_CHECKING:
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.terrain.presets import TerrainLayout

RASTER_VERSION = 1  # bump when the rasterisation changes (invalidates the cache)

# Policy scan (plan §2.3): mjlab GridPatternCfg(size=(1.6, 1.0), resolution=0.1), pelvis-centred, yaw-aligned.
SCAN_SIZE = (1.6, 1.0)
SCAN_RES = 0.1
SCAN_SHAPE = (11, 17)  # (ny, nx); flattened y-major like mjlab's meshgrid(x, y, indexing="xy")
SCAN_DIM = 187
SCAN_OFFSET = 0.75  # nominal pelvis height subtracted from (pelvis_z - h)
SCAN_CLIP = 1.0

# World-model terrain features (plan §2.4): 2 feet x 5 sole clearances + 2 feet x 3x3 patch.
SOLE_POINTS = 5  # 4 corners + centre of the sole
PATCH_SPACING = 0.06
FOOT_FEAT_DIM = 2 * SOLE_POINTS + 2 * 9  # 28
# Finer variant (plan §7 mitigation, Phase C M2): a 5 x 3 grid of sole clearances spanning the whole sole (contains
# the 4 corners) + a 5 x 5 patch at 5 cm around the sole centre (covers toe and heel), per foot.
FINE_SOLE_GRID = (5, 3)  # (along the foot x, across y)
FINE_PATCH_N, FINE_PATCH_SPACING = 5, 0.05
FOOT_FINE_DIM = 2 * FINE_SOLE_GRID[0] * FINE_SOLE_GRID[1] + 2 * FINE_PATCH_N**2  # 80
WM_TERRAIN_DIMS = {"none": 0, "foot": FOOT_FEAT_DIM, "scan": SCAN_DIM, "both": FOOT_FEAT_DIM + SCAN_DIM, "foot_fine": FOOT_FINE_DIM}

_FOOT_GEOM_RE = re.compile(r"(?:^|/)(left|right)_foot[1-7]_collision$")
_FOOT_SITE_RE = re.compile(r"(?:^|/)(left|right)_foot$")
_SIDES = ("left", "right")


# ---------------------------------------------------------------------------------------------- grid


@dataclass(frozen=True)
class GridSpec:
    """Raster geometry; plain Python scalars, i.e. compile-time constants."""

    x0: float  # world x of the lower grid edge
    y0: float
    res: float
    nx: int
    ny: int

    @classmethod
    def from_extent(cls, extent: tuple[float, float, float, float], res: float) -> GridSpec:
        x_lo, x_hi, y_lo, y_hi = extent
        nx, ny = round((x_hi - x_lo) / res), round((y_hi - y_lo) / res)
        assert abs(nx * res - (x_hi - x_lo)) < 1e-6 and abs(ny * res - (y_hi - y_lo)) < 1e-6, (extent, res)
        return cls(float(x_lo), float(y_lo), float(res), int(nx), int(ny))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.x0, self.x0 + self.nx * self.res, self.y0, self.y0 + self.ny * self.res)

    def cell_centres(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        xs = self.x0 + (torch.arange(self.nx, dtype=torch.float64) + 0.5) * self.res
        ys = self.y0 + (torch.arange(self.ny, dtype=torch.float64) + 0.5) * self.res
        return xs.to(device=device, dtype=dtype), ys.to(device=device, dtype=dtype)


class HeightGrid(NamedTuple):
    """The map tensors, (nx, ny) each, indexed [i_x, j_y]."""

    h: torch.Tensor  # raw height
    g: torch.Tensor  # blurred ground


def _cell_coords(xy: torch.Tensor, spec: GridSpec) -> tuple[torch.Tensor, torch.Tensor]:
    return (xy[..., 0] - spec.x0) / spec.res, (xy[..., 1] - spec.y0) / spec.res


def in_bounds(xy: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    u, v = _cell_coords(xy, spec)
    return (u >= 0) & (u <= spec.nx) & (v >= 0) & (v <= spec.ny)


def bilinear(field: torch.Tensor, xy: torch.Tensor, spec: GridSpec, detach_value: bool = True) -> torch.Tensor:
    """Bilinear interpolation of a cell-centre field at world xy (..., 2) -> (...) in xy's dtype.

    ``detach_value`` cuts the gradient through the interpolation weights (the terrain slope); the map
    values themselves never carry gradients. Outside the grid the result is 0 (no gradient)."""
    if detach_value:
        xy = xy.detach()
    u, v = _cell_coords(xy, spec)
    inside = (u >= 0) & (u <= spec.nx) & (v >= 0) & (v <= spec.ny)
    u = u - 0.5  # continuous index between cell centres
    v = v - 0.5
    i0 = torch.floor(u).clamp(0, spec.nx - 2)
    j0 = torch.floor(v).clamp(0, spec.ny - 2)
    fu = (u - i0).clamp(0.0, 1.0)  # clamping replicates the outermost half cell
    fv = (v - j0).clamp(0.0, 1.0)
    idx = i0.long() * spec.ny + j0.long()
    flat = field.reshape(-1)
    dt = xy.dtype
    a = flat[idx].to(dt)
    b = flat[idx + spec.ny].to(dt)
    c = flat[idx + 1].to(dt)
    d = flat[idx + spec.ny + 1].to(dt)
    val = (1 - fu) * ((1 - fv) * a + fv * c) + fu * ((1 - fv) * b + fv * d)
    return torch.where(inside, val, torch.zeros_like(val))


def cell_value(field: torch.Tensor, xy: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    """Value of the cell containing xy (nearest cell centre; no gradient); 0 outside the grid."""
    u, v = _cell_coords(xy.detach(), spec)
    inside = (u >= 0) & (u <= spec.nx) & (v >= 0) & (v <= spec.ny)
    i = torch.floor(u).clamp(0, spec.nx - 1).long()
    j = torch.floor(v).clamp(0, spec.ny - 1).long()
    val = field.reshape(-1)[i * spec.ny + j].to(xy.dtype)
    return torch.where(inside, val, torch.zeros_like(val))


def edge_distance(h: torch.Tensor, res: float, threshold: float = 0.03, max_dist: float = 1.0) -> torch.Tensor:
    """(nx, ny) fp32 xy distance (m) from every cell centre to the nearest *edge cell*, i.e. a cell whose height
    differs from one of its 4 neighbours by more than ``threshold`` (both sides of a step are edge cells; their
    distance is 0). Capped at ``max_dist``. Exact Euclidean distance transform (scipy, host, once per map)."""
    from scipy import ndimage

    x = h.float()
    edge = torch.zeros_like(x, dtype=torch.bool)
    dx = (x[1:] - x[:-1]).abs() > threshold
    dy = (x[:, 1:] - x[:, :-1]).abs() > threshold
    edge[1:] |= dx
    edge[:-1] |= dx
    edge[:, 1:] |= dy
    edge[:, :-1] |= dy
    if not edge.any():
        return torch.full_like(x, max_dist)
    d = ndimage.distance_transform_edt(~edge.cpu().numpy()) * res
    return torch.as_tensor(d, dtype=torch.float32, device=h.device).clamp_max(max_dist)


def gaussian_blur(h: torch.Tensor, sigma_cells: float) -> torch.Tensor:
    """Separable Gaussian blur (zero padding = flat ground at z=0 outside), deterministic fp32 shifted adds."""
    r = math.ceil(3.0 * sigma_cells)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float64) / sigma_cells) ** 2)
    k = (k / k.sum()).tolist()
    x = h.float()
    nx, ny = x.shape
    xp = torch.nn.functional.pad(x, (0, 0, r, r))
    acc = torch.zeros_like(x)
    for t, w in enumerate(k):
        acc.add_(xp[t : t + nx], alpha=w)
    yp = torch.nn.functional.pad(acc, (r, r, 0, 0))
    out = torch.zeros_like(x)
    for t, w in enumerate(k):
        out.add_(yp[:, t : t + ny], alpha=w)
    return out


# ---------------------------------------------------------------------------------------------- rasterisation


@dataclass
class TerrainMapCfg:
    res: float = 0.025
    ground_sigma: float = 0.15  # m, Gaussian sigma of the ground channel g
    cache_dir: str | None = "logs/terrain_cache"  # None disables the disk cache
    chunk: int = 1 << 22  # rays per mujoco_warp.rays call
    ray_margin: float = 1.0  # rays start this far above the highest terrain geom


def _wp_device(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(f"rasterisation needs a CUDA device, got {device}")
    return f"cuda:{device.index if device.index is not None else torch.cuda.current_device()}"


def _terrain_geom_ids(mj_model: mujoco.MjModel) -> np.ndarray:
    ids = np.nonzero(np.asarray(mj_model.geom_group) == 0)[0]
    if len(ids) == 0:
        raise ValueError("no geom group 0 (terrain) geoms in the model")
    return ids


def _top_z(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, ids: np.ndarray) -> float:
    """Highest point of the AABBs of the given geoms (world)."""
    aabb = np.asarray(mj_model.geom_aabb)[ids]  # local (centre(3), half(3))
    xmat = np.asarray(mj_data.geom_xmat)[ids].reshape(-1, 3, 3)
    xpos = np.asarray(mj_data.geom_xpos)[ids]
    rz = xmat[:, 2, :]  # world-z row of the rotation
    top = xpos[:, 2] + (rz * aabb[:, :3]).sum(-1) + (np.abs(rz) * aabb[:, 3:]).sum(-1)
    top = top[np.isfinite(top)]
    if top.size == 0:
        raise ValueError("terrain geoms have no finite extent (plane?)")
    return float(top.max())


def geometry_digest(mj_model: mujoco.MjModel) -> str:
    """Hash of the world pose, size and data of every group-0 geom (content key of the raster cache)."""
    d = mujoco.MjData(mj_model)
    mujoco.mj_kinematics(mj_model, d)
    ids = _terrain_geom_ids(mj_model)
    hsh = hashlib.sha1()
    for arr in (
        np.asarray(mj_model.geom_type)[ids],
        np.asarray(mj_model.geom_size)[ids],
        np.asarray(mj_model.geom_dataid)[ids],
        np.asarray(d.geom_xpos)[ids],
        np.asarray(d.geom_xmat)[ids],
        np.asarray(mj_model.hfield_size),
        np.asarray(mj_model.hfield_nrow),
        np.asarray(mj_model.hfield_ncol),
        np.asarray(mj_model.hfield_data),
    ):
        hsh.update(np.ascontiguousarray(arr).tobytes())
    return hsh.hexdigest()


def rasterize(mj_model: mujoco.MjModel, spec: GridSpec, device: str | torch.device = "cuda", chunk: int = 1 << 22, ray_margin: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Top-surface height of the group-0 geoms at every cell centre: vertical rays through MuJoCo Warp
    with a standalone nworld=1 RenderContext (its own BVH; no refit since the terrain is static).
    Robot geoms are in groups 2/3 and therefore ignored. Returns ((nx, ny) fp32, stats)."""
    import mujoco_warp as mjwarp
    import warp as wp

    t0 = time.perf_counter()
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    mjd = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mjd)
    ids = _terrain_geom_ids(mj_model)
    z_top = _top_z(mj_model, mjd, ids) + ray_margin

    n_total = spec.nx * spec.ny
    chunk = min(chunk, n_total)
    xs, ys = spec.cell_centres(device)
    pnt = torch.empty(1, chunk, 3, device=device)
    pnt[..., 2] = z_top
    vec = torch.zeros(1, chunk, 3, device=device)
    vec[..., 2] = -1.0
    dist = torch.empty(1, chunk, device=device)
    geomid = torch.empty(1, chunk, device=device, dtype=torch.int32)
    normal = torch.empty(1, chunk, 3, device=device)
    bodyexclude = torch.full((chunk,), -1, device=device, dtype=torch.int32)
    out = torch.empty(n_total, device=device)
    vec6 = wp.types.vector(length=6, dtype=float)
    groups = vec6(-1, 0, 0, 0, 0, 0)  # mujoco_warp: -1 include, 0 exclude (group 0 only)

    with wp.ScopedDevice(_wp_device(device)), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device))):
        m = mjwarp.put_model(mj_model)
        d = mjwarp.put_data(mj_model, mjd, nworld=1)
        rc = mjwarp.create_render_context(
            mj_model, nworld=1, cam_active=[False] * mj_model.ncam, enabled_geom_groups=[0], use_textures=False, use_shadows=False
        )
        t_ctx = time.perf_counter() - t0
        for start in range(0, n_total, chunk):
            n = min(chunk, n_total - start)
            k = torch.arange(start, start + n, device=device)
            pnt[0, :n, 0] = xs[k // spec.ny]
            pnt[0, :n, 1] = ys[k % spec.ny]
            mjwarp.rays(
                m,
                d,
                wp.from_torch(pnt[:, :n], dtype=wp.vec3),
                wp.from_torch(vec[:, :n], dtype=wp.vec3),
                groups,
                True,
                wp.from_torch(bodyexclude[:n]),
                wp.from_torch(dist[:, :n]),
                wp.from_torch(geomid[:, :n]),
                wp.from_torch(normal[:, :n], dtype=wp.vec3),
                rc=rc,
            )
            dn = dist[0, :n]
            out[start : start + n] = torch.where(dn >= 0, z_top - dn, torch.full_like(dn, float("nan")))
    h = out.view(spec.nx, spec.ny)
    misses = int(torch.isnan(h).sum().item())
    if misses:
        h = _fill_misses(h)
    torch.cuda.synchronize(device)
    stats = {"misses": misses, "z_top": z_top, "context_s": t_ctx, "raster_s": time.perf_counter() - t0, "num_rays": n_total}
    return h, stats


def _fill_misses(h: torch.Tensor, max_iter: int = 64) -> torch.Tensor:
    """Fill NaN cells with the mean of their valid 3x3 neighbours (repeatedly); leftovers -> 0."""
    h = h.clone()
    kernel = torch.ones(1, 1, 3, 3, device=h.device)
    for _ in range(max_iter):
        bad = torch.isnan(h)
        if not bad.any():
            break
        valid = (~bad).float()[None, None]
        s = torch.nn.functional.conv2d(torch.nan_to_num(h)[None, None] * valid, kernel, padding=1)[0, 0]
        c = torch.nn.functional.conv2d(valid, kernel, padding=1)[0, 0]
        fill = bad & (c > 0)
        h[fill] = s[fill] / c[fill]
    return torch.nan_to_num(h, nan=0.0)


def _mjlab_ident() -> dict:
    from importlib import metadata

    ident = {"mjlab": metadata.version("mjlab"), "mujoco": mujoco.__version__}
    try:
        ident["mujoco_warp"] = metadata.version("mujoco_warp")
    except metadata.PackageNotFoundError:  # pragma: no cover
        ident["mujoco_warp"] = "?"
    try:
        url = json.loads(metadata.distribution("mjlab").read_text("direct_url.json") or "{}")
        ident["mjlab_commit"] = url.get("vcs_info", {}).get("commit_id", "")
    except Exception:  # noqa: BLE001  (editable / wheel installs have no direct_url)
        ident["mjlab_commit"] = ""
    return ident


def cache_key(gen_cfg: TerrainGeneratorCfg, spec: GridSpec, cfg: TerrainMapCfg, digest: str) -> tuple[str, dict]:
    """(hash, key dict): terrain seed, generator config repr, mjlab version/commit, resolution, plus the
    compiled geometry digest (guards against configs that repr the same but build differently)."""
    key = {
        "raster_version": RASTER_VERSION,
        "seed": gen_cfg.seed,
        "generator": repr(gen_cfg),
        **_mjlab_ident(),
        "grid": asdict(spec),
        "ground_sigma": cfg.ground_sigma,
        "geometry": digest,
    }
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20], key


# ---------------------------------------------------------------------------------------------- feet


@dataclass
class FootGeometry:
    """Sole points and foot sites in the ``*_ankle_roll_link`` frames, read from the MuJoCo model."""

    body_names: tuple[str, str]  # (left, right) ankle roll link (unprefixed)
    sole: np.ndarray  # (2, 5, 3): 4 corners + centre of the sole bottom
    site: np.ndarray  # (2, 3): left_foot / right_foot site position

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> FootGeometry:
        """From the ``*_foot{1..7}_collision`` capsules (axis endpoints x/y range, lowest axis z minus the
        radius). Works on the standalone g1.xml and on the mjlab scene (``robot/`` prefixes)."""
        sole, site, bodies = [], [], []
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
        site_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or "" for i in range(model.nsite)]
        for side in _SIDES:
            ids = [i for i, n in enumerate(names) if (mt := _FOOT_GEOM_RE.search(n)) and mt.group(1) == side]
            if not ids:
                raise ValueError(f"no {side}_foot*_collision geoms in the model")
            body_ids = {int(model.geom_bodyid[i]) for i in ids}
            assert len(body_ids) == 1, f"{side} foot capsules on several bodies"
            body = body_ids.pop()
            ends = []
            for i in ids:
                assert model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE, names[i]
                r, half = float(model.geom_size[i, 0]), float(model.geom_size[i, 1])
                axis = np.zeros(3)
                mujoco.mju_rotVecQuat(axis, np.array([0.0, 0.0, half]), np.asarray(model.geom_quat[i], dtype=np.float64))
                p = np.asarray(model.geom_pos[i], dtype=np.float64)
                ends += [(p + axis, r), (p - axis, r)]
            pts = np.stack([e for e, _ in ends])
            z = min(e[2] - r for e, r in ends)
            x_lo, x_hi = pts[:, 0].min(), pts[:, 0].max()
            y_lo, y_hi = pts[:, 1].min(), pts[:, 1].max()
            sole.append([[x_lo, y_lo, z], [x_lo, y_hi, z], [x_hi, y_lo, z], [x_hi, y_hi, z], [0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), z]])
            sid = [i for i, n in enumerate(site_names) if (mt := _FOOT_SITE_RE.search(n)) and mt.group(1) == side]
            assert len(sid) == 1, f"expected one {side}_foot site, found {len(sid)}"
            assert int(model.site_bodyid[sid[0]]) == body, f"{side}_foot site is not on the sole body"
            site.append(np.asarray(model.site_pos[sid[0]], dtype=np.float64))
            bodies.append((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "").split("/")[-1])
        return cls(tuple(bodies), np.asarray(sole), np.asarray(site))  # type: ignore[arg-type]


class FootPoints(NamedTuple):
    sole: torch.Tensor  # (..., 2, 5, 3) world sole points
    site: torch.Tensor  # (..., 2, 3) world foot-site positions
    heading: torch.Tensor  # (..., 2, 2) (cos, sin) of each foot's yaw


# ---------------------------------------------------------------------------------------------- helpers


def scan_grid_offsets(device: str | torch.device = "cpu") -> torch.Tensor:
    """(187, 2) local xy offsets of the policy scan, bit-identical to mjlab's GridPatternCfg.generate_rays."""
    sx, sy = SCAN_SIZE
    x = torch.arange(-sx / 2, sx / 2 + SCAN_RES * 0.5, SCAN_RES, device=device, dtype=torch.float32)
    y = torch.arange(-sy / 2, sy / 2 + SCAN_RES * 0.5, SCAN_RES, device=device, dtype=torch.float32)
    gx, gy = torch.meshgrid(x, y, indexing="xy")
    out = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
    assert out.shape == (SCAN_DIM, 2)
    return out


def fine_sole_points(sole: np.ndarray) -> np.ndarray:
    """(2, 15, 3) sole grid (x-major, ``FINE_SOLE_GRID``) spanned by the 4 sole corners of ``FootGeometry.sole``."""
    nx, ny = FINE_SOLE_GRID
    x = np.linspace(sole[:, 0, 0], sole[:, 2, 0], nx, axis=-1)  # (2, nx): corner 0 = (x_lo, y_lo), 2 = (x_hi, y_lo)
    y = np.linspace(sole[:, 0, 1], sole[:, 1, 1], ny, axis=-1)  # (2, ny): corner 1 = (x_lo, y_hi)
    gx, gy = np.broadcast_arrays(x[:, :, None], y[:, None, :])
    z = np.broadcast_to(sole[:, :1, 2:3], (2, nx * ny, 1))
    return np.concatenate([np.stack([gx, gy], -1).reshape(2, nx * ny, 2), z], -1)


def patch_offsets_fine(device: str | torch.device = "cpu") -> torch.Tensor:
    """(25, 2) offsets of the 5 x 5 fine foot patch (y-major like the scan)."""
    a = (torch.arange(FINE_PATCH_N, dtype=torch.float32, device=device) - (FINE_PATCH_N - 1) / 2) * FINE_PATCH_SPACING
    gx, gy = torch.meshgrid(a, a, indexing="xy")
    return torch.stack([gx.flatten(), gy.flatten()], dim=-1)


def patch_offsets(device: str | torch.device = "cpu") -> torch.Tensor:
    """(9, 2) offsets of the 3x3 foot patch (same y-major ordering as the scan)."""
    a = torch.tensor([-PATCH_SPACING, 0.0, PATCH_SPACING], device=device)
    gx, gy = torch.meshgrid(a, a, indexing="xy")
    return torch.stack([gx.flatten(), gy.flatten()], dim=-1)


def yaw_cos_sin(quat: torch.Tensor) -> torch.Tensor:
    """(..., 2) unit (cos, sin) of the heading of a wxyz quaternion: the body x-axis projected on the
    ground plane, falling back to the y-axis near +-90 deg pitch (mjlab ``_extract_yaw_rotation``)."""
    w, x, y, z = quat.unbind(-1)
    xc = 1 - 2 * (y * y + z * z)  # R[0, 0]
    xs = 2 * (x * y + w * z)  # R[1, 0]
    yx = 2 * (x * y - w * z)  # R[0, 1]
    yy = 1 - 2 * (x * x + z * z)  # R[1, 1]
    singular = torch.sqrt(xc * xc + xs * xs) < 0.1
    c = torch.where(singular, yy, xc)
    s = torch.where(singular, -yx, xs)
    n = torch.sqrt(c * c + s * s).clamp_min(1e-6)
    return torch.stack([c / n, s / n], dim=-1)


def _rotate_offsets(cs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """cs (..., 2), offsets (P, 2) -> (..., P, 2) offsets rotated by the yaw."""
    c, s = cs[..., 0:1], cs[..., 1:2]
    ox, oy = offsets[:, 0].to(cs.dtype), offsets[:, 1].to(cs.dtype)
    return torch.stack([c * ox - s * oy, s * ox + c * oy], dim=-1)


# ---------------------------------------------------------------------------------------------- map


class TerrainMap:
    """Static global height map + the analytic terrain heads (scan, foot features).

    Build with ``TerrainMap.build(mj_model, gen_cfg, device)`` (``gen_cfg=None`` -> plane mode) or
    ``TerrainMap.plane(device)``. Every lookup takes an optional ``grid`` (``HeightGrid``) so compiled
    callers can pass the map tensors as graph inputs; it defaults to ``self.grid``.
    """

    def __init__(
        self,
        grid: HeightGrid | None,
        spec: GridSpec | None,
        feet: FootGeometry,
        device: str | torch.device = "cuda",
        layout: TerrainLayout | None = None,
        info: dict | None = None,
    ):
        assert (grid is None) == (spec is None)
        self.device = torch.device(device)
        self.grid = grid
        self.spec = spec
        self.layout = layout
        self.feet = feet
        self.info = info or {}
        self.sole_local = torch.as_tensor(feet.sole, dtype=torch.float32, device=self.device)  # (2, 5, 3)
        self.sole_fine_local = torch.as_tensor(fine_sole_points(feet.sole), dtype=torch.float32, device=self.device)  # (2, 15, 3)
        self.patch_offsets_fine = patch_offsets_fine(self.device)  # (25, 2)
        self.site_local = torch.as_tensor(feet.site, dtype=torch.float32, device=self.device)  # (2, 3)
        self.scan_offsets = scan_grid_offsets(self.device)  # (187, 2)
        self.patch_offsets = patch_offsets(self.device)  # (9, 2)
        self.oob_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._edge_dist: dict[tuple[float, float], torch.Tensor] = {}  # edge_distance_field cache

    # ------------------------------------------------------------------ construction

    @classmethod
    def plane(cls, device: str | torch.device = "cuda", feet: FootGeometry | None = None) -> TerrainMap:
        if feet is None:
            from supermimic.data.motion_convert import load_g1_model

            feet = FootGeometry.from_model(load_g1_model())
        return cls(None, None, feet, device, info={"kind": "plane"})

    @classmethod
    def build(
        cls,
        mj_model: mujoco.MjModel,
        gen_cfg: TerrainGeneratorCfg | None,
        device: str | torch.device = "cuda",
        cfg: TerrainMapCfg | None = None,
        feet: FootGeometry | None = None,
        verbose: bool = False,
    ) -> TerrainMap:
        """Rasterise (or load from the cache) the terrain of a compiled scene built from ``gen_cfg``."""
        from supermimic.terrain.presets import TerrainLayout

        cfg = cfg or TerrainMapCfg()
        if feet is None:
            try:
                feet = FootGeometry.from_model(mj_model)
            except ValueError:  # terrain-only model
                from supermimic.data.motion_convert import load_g1_model

                feet = FootGeometry.from_model(load_g1_model())
        if gen_cfg is None:
            return cls.plane(device, feet)
        layout = TerrainLayout.from_generator_cfg(gen_cfg)
        spec = GridSpec.from_extent(layout.extent, cfg.res)
        digest = geometry_digest(mj_model)
        key, key_dict = cache_key(gen_cfg, spec, cfg, digest)
        path = Path(cfg.cache_dir) / f"{key}.pt" if cfg.cache_dir else None
        device = torch.device(device)
        info: dict = {"kind": "generator", "cache_key": key, "cache_path": str(path) if path else None}
        if path is not None and path.exists():
            t0 = time.perf_counter()
            blob = torch.load(path, map_location="cpu", weights_only=False)
            assert blob["key"] == key_dict, f"terrain cache {path} key mismatch"
            grid = HeightGrid(blob["h"].to(device), blob["g"].to(device))
            info.update(blob["stats"], cache_hit=True, load_s=time.perf_counter() - t0)
        else:
            h, stats = rasterize(mj_model, spec, device, cfg.chunk, cfg.ray_margin)
            t0 = time.perf_counter()
            g = gaussian_blur(h, cfg.ground_sigma / cfg.res)
            torch.cuda.synchronize(device)
            stats["blur_s"] = time.perf_counter() - t0
            grid = HeightGrid(h.half().contiguous(), g.half().contiguous())
            info.update(stats, cache_hit=False)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".tmp{os.getpid()}")
                torch.save({"key": key_dict, "h": grid.h.cpu(), "g": grid.g.cpu(), "stats": stats}, tmp)
                os.replace(tmp, path)  # atomic: concurrent ranks never see a partial file
        if verbose:
            print(f"[terrain] {spec.nx}x{spec.ny} @ {spec.res} m, {info}")
        return cls(grid, spec, feet, device, layout, info)

    # ------------------------------------------------------------------ properties

    @property
    def is_plane(self) -> bool:
        return self.grid is None

    def grid_as(self, dtype: torch.dtype) -> HeightGrid | None:
        """Copy of the map in another dtype (fp64 for gradcheck)."""
        if self.grid is None:
            return None
        return HeightGrid(self.grid.h.to(dtype), self.grid.g.to(dtype))

    # ------------------------------------------------------------------ point lookups

    def _field(self, grid: HeightGrid | None, ch: str) -> torch.Tensor:
        g = self.grid if grid is None else grid
        return g.h if ch == "h" else g.g

    def height(self, xy_w: torch.Tensor, grid: HeightGrid | None = None, detach_value: bool = True) -> torch.Tensor:
        """Raw terrain height h at world xy (..., 2) -> (...)."""
        if self.spec is None:
            return torch.zeros_like(xy_w[..., 0])
        return bilinear(self._field(grid, "h"), xy_w, self.spec, detach_value)

    def ground(self, xy_w: torch.Tensor, grid: HeightGrid | None = None, detach_value: bool = True) -> torch.Tensor:
        """Blurred ground g at world xy (..., 2) -> (...)."""
        if self.spec is None:
            return torch.zeros_like(xy_w[..., 0])
        return bilinear(self._field(grid, "g"), xy_w, self.spec, detach_value)

    def height_upper(self, xy_w: torch.Tensor, grid: HeightGrid | None = None) -> torch.Tensor:
        """max(bilinear h, h of the containing cell): never under-estimates the surface just on the high
        side of a step edge (where bilinear blends in the low side). No gradient."""
        if self.spec is None:
            return torch.zeros_like(xy_w[..., 0])
        f = self._field(grid, "h")
        return torch.maximum(bilinear(f, xy_w, self.spec, True), cell_value(f, xy_w, self.spec))

    def edge_distance_field(self, threshold: float = 0.03, max_dist: float = 1.0) -> torch.Tensor | None:
        """(nx, ny) distance to the nearest step edge (``edge_distance`` of h), computed once per threshold and
        cached on the object (diagnostics: M2's near-edge statistics). ``None`` in plane mode."""
        if self.grid is None:
            return None
        key = (float(threshold), float(max_dist))
        if key not in self._edge_dist:
            self._edge_dist[key] = edge_distance(self.grid.h, self.spec.res, threshold, max_dist)
        return self._edge_dist[key]

    def edge_dist(self, xy_w: torch.Tensor, threshold: float = 0.03, max_dist: float = 1.0) -> torch.Tensor:
        """(...) distance (m) from world xy (..., 2) to the nearest step edge (cell lookup, no gradient); ``max_dist``
        off the map and in plane mode."""
        f = self.edge_distance_field(threshold, max_dist)
        if f is None:
            return torch.full_like(xy_w[..., 0], max_dist)
        d = cell_value(f, xy_w, self.spec)
        return torch.where(in_bounds(xy_w, self.spec), d, torch.full_like(d, max_dist))

    def out_of_bounds(self, xy_w: torch.Tensor) -> torch.Tensor:
        """(...) bool: query outside the rasterised extent (never in plane mode)."""
        if self.spec is None:
            return torch.zeros_like(xy_w[..., 0], dtype=torch.bool)
        return ~in_bounds(xy_w, self.spec)

    @torch.no_grad()
    def count_out_of_bounds(self, xy_w: torch.Tensor) -> torch.Tensor:
        """Accumulate the number of out-of-bounds queries into ``self.oob_count`` (no host sync; eager)."""
        n = self.out_of_bounds(xy_w).sum()
        self.oob_count += n
        return n

    # ------------------------------------------------------------------ policy scan

    def scan_points(self, root_pos_w: torch.Tensor, root_quat: torch.Tensor, xy_shift: torch.Tensor | None = None) -> torch.Tensor:
        """(..., 187, 2) world xy of the yaw-aligned 17x11 grid around the pelvis (+ optional map shift (..., 2))."""
        cs = yaw_cos_sin(root_quat)
        p = root_pos_w[..., None, 0:2] + _rotate_offsets(cs, self.scan_offsets)
        if xy_shift is not None:
            p = p + xy_shift[..., None, :]
        return p

    def scan_heights(self, root_pos_w, root_quat, grid=None, xy_shift=None, detach_value: bool = True) -> torch.Tensor:
        """(..., 187) raw terrain height h at the scan points."""
        return self.height(self.scan_points(root_pos_w, root_quat, xy_shift), grid, detach_value)

    def scan(self, root_pos_w: torch.Tensor, root_quat: torch.Tensor, grid: HeightGrid | None = None, xy_shift: torch.Tensor | None = None, detach_value: bool = True) -> torch.Tensor:
        """(..., 187) noise-free scan clip(pelvis_z - h(p) - 0.75, -1, 1); root_pos_w is the world pelvis position."""
        h = self.scan_heights(root_pos_w, root_quat, grid, xy_shift, detach_value)
        return (root_pos_w[..., 2:3] - h - SCAN_OFFSET).clamp(-SCAN_CLIP, SCAN_CLIP)

    def scan_state(self, s: torch.Tensor, origin: torch.Tensor, grid: HeightGrid | None = None, xy_shift: torch.Tensor | None = None, detach_value: bool = True) -> torch.Tensor:
        """``scan`` from a state s (..., >=7; root relative to ``origin`` (..., 3), broadcastable)."""
        return self.scan(s[..., 0:3] + origin, s[..., 3:7], grid, xy_shift, detach_value)

    # ------------------------------------------------------------------ world-model foot features

    def foot_points(self, qpos: torch.Tensor, origin: torch.Tensor | None, fk: G1Kinematics) -> FootPoints:
        """World sole points, foot sites and foot headings from qpos (..., >=36; the 71-d state works too),
        root relative to ``origin`` (..., 3) (``None``: qpos is already in world coordinates)."""
        pos, quat = self._foot_frames(qpos, origin, fk)
        site = pos + quat_apply(quat, self.site_local.to(pos.dtype).expand(*pos.shape[:-2], 2, 3))
        return FootPoints(self._in_feet(pos, quat, self.sole_local), site, yaw_cos_sin(quat))

    def _foot_frames(self, qpos: torch.Tensor, origin: torch.Tensor | None, fk: G1Kinematics) -> tuple[torch.Tensor, torch.Tensor]:
        """World pose (pos (..., 2, 3), quat (..., 2, 4)) of both ankle roll links: one stacked chain (half the
        kernels of two ``chain_pose`` calls)."""
        pos, quat = fk.chain_poses(qpos[..., :36], tuple(fk.body_index(name) for name in self.feet.body_names))
        return (pos if origin is None else pos + origin[..., None, :]), quat

    @staticmethod
    def _in_feet(pos: torch.Tensor, quat: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        """World positions (..., 2, P, 3) of foot-frame points ``local`` (2, P, 3)."""
        batch, p = pos.shape[:-2], local.shape[1]
        return pos[..., None, :] + quat_apply(quat[..., None, :].expand(*batch, 2, p, 4), local.to(pos.dtype).expand(*batch, 2, p, 3))

    def foot_features_fine(self, qpos: torch.Tensor, origin: torch.Tensor | None, fk: G1Kinematics, grid: HeightGrid | None = None, detach_value: bool = True) -> torch.Tensor:
        """(..., 80) = [5 x 3 sole clearances z - h (left 15, right 15), foot-yaw-frame 5 x 5 patches h - sole-centre z
        at 5 cm (left 25, right 25)] (``wm_terrain="foot_fine"``)."""
        pos, quat = self._foot_frames(qpos, origin, fk)
        sole = self._in_feet(pos, quat, self.sole_fine_local)  # (..., 2, 15, 3)
        clear = sole[..., 2] - self.height(sole[..., 0:2], grid, detach_value)
        centre = self._in_feet(pos, quat, self.sole_local[:, SOLE_POINTS - 1 :])[..., 0, :]  # (..., 2, 3) sole centre
        pts = centre[..., None, 0:2] + _rotate_offsets(yaw_cos_sin(quat), self.patch_offsets_fine)  # (..., 2, 25, 2)
        patch = self.height(pts, grid, detach_value) - centre[..., None, 2]
        return torch.cat([clear.flatten(-2), patch.flatten(-2)], dim=-1)

    def foot_features(self, qpos: torch.Tensor, origin: torch.Tensor | None, fk: G1Kinematics, grid: HeightGrid | None = None, detach_value: bool = True) -> torch.Tensor:
        """(..., 28) = [sole clearances z - h (left 5, right 5), foot-yaw-frame 3x3 patches h - site_z
        (left 9, right 9)]. Noise-free; ``detach_value`` keeps gradients through the query z only."""
        fp = self.foot_points(qpos, origin, fk)
        clear = fp.sole[..., 2] - self.height(fp.sole[..., 0:2], grid, detach_value)  # (..., 2, 5)
        pts = fp.site[..., None, 0:2] + _rotate_offsets(fp.heading, self.patch_offsets)  # (..., 2, 9, 2)
        patch = self.height(pts, grid, detach_value) - fp.site[..., None, 2]  # (..., 2, 9)
        return torch.cat([clear.flatten(-2), patch.flatten(-2)], dim=-1)

    def wm_condition(
        self,
        mode: str,
        s: torch.Tensor,
        origin: torch.Tensor,
        fk: G1Kinematics,
        grid: HeightGrid | None = None,
        scan_clean: torch.Tensor | None = None,
        detach_value: bool = True,
    ) -> torch.Tensor:
        """Noise-free WM terrain condition c_t for ``wm_terrain`` mode: foot (28) | scan (187) | both (215,
        [foot, scan]) | foot_fine (80) | none (0). ``scan_clean`` may pass an already computed *unshifted* scan of s."""
        if mode not in WM_TERRAIN_DIMS:
            raise ValueError(f"unknown wm_terrain {mode!r}")
        if mode == "none":
            return s.new_zeros(s.shape[:-1] + (0,))
        if mode == "foot_fine":
            return self.foot_features_fine(s, origin, fk, grid, detach_value)
        parts = []
        if mode in ("foot", "both"):
            parts.append(self.foot_features(s, origin, fk, grid, detach_value))
        if mode in ("scan", "both"):
            parts.append(scan_clean if scan_clean is not None else self.scan_state(s, origin, grid, None, detach_value))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]


# ---------------------------------------------------------------------------------------------- scan noise


@dataclass
class ScanNoiseCfg:
    """Policy-side scan corruption (plan §2.3); the world model and the critic never see it."""

    enabled: bool = True
    point: float = 0.02  # per-point uniform half-width, m
    offset_z: float = 0.03  # per-episode map z offset half-width, m
    offset_xy: float = 0.03  # per-episode map xy offset half-width (query-point shift), m
    dropout: float = 0.05  # per-point probability of a missing cell (-> clip max)
    max_lag: int = 2  # per-episode latency in control steps, uniform in [0, max_lag]

    @property
    def history_len(self) -> int:
        return self.max_lag + 1


class ScanNoiseParams(NamedTuple):
    """Per-episode noise parameters, batch shape (*B,) (sampled at reset)."""

    offset_z: torch.Tensor  # (*B,)
    offset_xy: torch.Tensor  # (*B, 2): pass as ``xy_shift`` when computing the scan that is pushed to the history
    lag: torch.Tensor  # (*B,) long


def _uniform(shape, half: float, device, generator=None, dtype=torch.float32) -> torch.Tensor:
    return (torch.rand(shape, device=device, generator=generator, dtype=dtype) * 2 - 1) * half


def sample_scan_noise(batch_shape: tuple[int, ...] | torch.Size | int, cfg: ScanNoiseCfg, device: str | torch.device, generator: torch.Generator | None = None) -> ScanNoiseParams:
    shape = (batch_shape,) if isinstance(batch_shape, int) else tuple(batch_shape)
    if not cfg.enabled:
        return ScanNoiseParams(torch.zeros(shape, device=device), torch.zeros(shape + (2,), device=device), torch.zeros(shape, device=device, dtype=torch.long))
    lag = torch.randint(0, cfg.max_lag + 1, shape, device=device, generator=generator)
    return ScanNoiseParams(_uniform(shape, cfg.offset_z, device, generator), _uniform(shape + (2,), cfg.offset_xy, device, generator), lag)


def resample_scan_noise_(params: ScanNoiseParams, env_ids: torch.Tensor, cfg: ScanNoiseCfg, generator: torch.Generator | None = None) -> None:
    """In-place re-draw of the episode parameters of ``env_ids`` (sim side, at reset)."""
    new = sample_scan_noise((env_ids.numel(),), cfg, params.lag.device, generator)
    params.offset_z[env_ids] = new.offset_z
    params.offset_xy[env_ids] = new.offset_xy
    params.lag[env_ids] = new.lag


def init_scan_history(scan: torch.Tensor, length: int) -> torch.Tensor:
    """(L, *B, 187) history filled with the current scan (reset: every lag reads the reset scan)."""
    return scan[None].expand(length, *scan.shape).clone()


def push_scan(hist: torch.Tensor, scan: torch.Tensor) -> torch.Tensor:
    """Append the current scan (oldest -> newest along dim 0); lag 0 then reads ``scan`` itself."""
    return torch.cat([hist[1:], scan[None]], dim=0)


def noisy_scan(hist: torch.Tensor, params: ScanNoiseParams, cfg: ScanNoiseCfg, generator: torch.Generator | None = None) -> torch.Tensor:
    """Policy scan (*B, 187) from the history of noise-free scans (L, *B, 187), oldest -> newest.

    The history holds the scans queried at the episode's shifted points
    (``scan(..., xy_shift=params.offset_xy)``); this applies the latency, the map z offset, the per-point
    noise and the dropout. Same code on the sim and the imagination side."""
    L = hist.shape[0]
    idx = (L - 1 - params.lag.clamp(0, L - 1)).reshape(1, *hist.shape[1:-1], 1).expand(1, *hist.shape[1:])
    out = torch.gather(hist, 0, idx)[0]
    if not cfg.enabled:
        return out
    out = out - params.offset_z[..., None].to(out.dtype)  # terrain reported dz too high -> pelvis-relative value drops
    if generator is None:
        noise = torch.rand_like(out)
        drop = torch.rand_like(out) < cfg.dropout
    else:
        noise = torch.rand(out.shape, device=out.device, dtype=out.dtype, generator=generator)
        drop = torch.rand(out.shape, device=out.device, dtype=out.dtype, generator=generator) < cfg.dropout
    out = out + (noise * 2 - 1) * cfg.point
    out = torch.where(drop, torch.full_like(out, SCAN_CLIP), out)
    return out.clamp(-SCAN_CLIP, SCAN_CLIP)
