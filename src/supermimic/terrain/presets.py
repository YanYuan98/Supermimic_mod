"""Terrain generator presets (plan §2.7) and the tile layout used by the spawn sampler.

``mixed`` is mjlab's curriculum grid: one column per terrain type, ``num_rows`` rows of increasing
difficulty ``row / (num_rows - 1)``. The grid is centred on the world origin with rows along x and
columns along y (``TerrainGenerator._get_sub_terrain_position``), surrounded by a flat border whose
top is at z = 0. ``plane`` is mjlab's infinite ground plane (no generator).
"""

from __future__ import annotations

from dataclasses import dataclass

import mjlab.terrains as T
import mujoco
import numpy as np
import torch
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput

TERRAIN_KINDS = ("plane", "mixed")

# Stairs: 0.30 m treads, 6 steps per side inside a 0.5 m frame; the platform width only sets the step
# count (``int((size - 2 border - platform) / (2 step_width))``), the platform box itself is whatever is left.
STAIR_STEPS = 6
STAIR_TREAD = 0.30
STAIR_FRAME = 0.5

# Solid below every terrain surface. MuJoCo (Warp) resolves a foot capsule against a box or a height-field prism
# (``ccd_hfield_kernel``: prisms span [-base, surface]) along the minimum-translation direction, so once the
# capsule centre sinks below the solid's mid-height (penetration > radius + thickness / 2) the contact normal
# points *down* and pushes the foot through. mjlab sizes height-field bases as ratio x elevation range (1-5 mm
# at difficulty 0: perlin 1 mm, wave ratio 0.25) and stair boxes as small multiples of the step height (the pit
# platform and its innermost ring are one step, 2 cm, thick at difficulty 0), while impacts of a landing or
# running G1 (foot capsules r = 1 cm) reach 2-5 cm penetration: feet fell through flat-looking tiles. Thickening
# only moves bottoms (boxes extend downwards, hfield bases grow): surfaces, raster and contacts are unchanged.
MIN_TERRAIN_THICKNESS = 0.5  # m


def _thicken(geom: TerrainGeometry, thickness: float) -> None:
    """Give one sub-terrain geometry at least ``thickness`` of solid below its top surface (in place)."""
    if geom.hfield is not None:
        size = [float(x) for x in geom.hfield.size]
        size[3] = max(size[3], thickness)
        geom.hfield.size = size
    elif geom.geom is not None and geom.geom.type == mujoco.mjtGeom.mjGEOM_BOX and 2 * geom.geom.size[2] < thickness:
        assert np.allclose(geom.geom.quat, (1.0, 0.0, 0.0, 0.0)), "only axis-aligned terrain boxes are thickened"
        top = float(geom.geom.pos[2] + geom.geom.size[2])
        geom.geom.size = [float(geom.geom.size[0]), float(geom.geom.size[1]), 0.5 * thickness]
        geom.geom.pos = [float(geom.geom.pos[0]), float(geom.geom.pos[1]), top - 0.5 * thickness]


@dataclass(kw_only=True)
class _Solid:
    """Mixin for mjlab sub-terrains: every box / height field is at least ``min_thickness`` thick below its
    surface. The surfaces (and every random draw) are those of the wrapped mjlab terrain."""

    min_thickness: float = MIN_TERRAIN_THICKNESS

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        out = super().function(difficulty, spec, rng)  # type: ignore[misc]
        for geom in out.geometries:
            _thicken(geom, self.min_thickness)
        return out


@dataclass(kw_only=True)
class BoxPyramidStairsCfg(_Solid, T.BoxPyramidStairsTerrainCfg):
    pass


@dataclass(kw_only=True)
class BoxInvertedPyramidStairsCfg(_Solid, T.BoxInvertedPyramidStairsTerrainCfg):
    pass


@dataclass(kw_only=True)
class HfPyramidSlopedCfg(_Solid, T.HfPyramidSlopedTerrainCfg):
    pass


@dataclass(kw_only=True)
class HfRandomUniformCfg(_Solid, T.HfRandomUniformTerrainCfg):
    pass


@dataclass(kw_only=True)
class HfWaveCfg(_Solid, T.HfWaveTerrainCfg):
    pass


@dataclass(kw_only=True)
class HfPerlinNoiseCfg(_Solid, T.HfPerlinNoiseTerrainCfg):
    pass


@dataclass(kw_only=True)
class HfDiscreteObstaclesCfg(_Solid, T.HfDiscreteObstaclesTerrainCfg):
    pass


_MJLAB_CFG = {
    BoxPyramidStairsCfg: T.BoxPyramidStairsTerrainCfg,
    BoxInvertedPyramidStairsCfg: T.BoxInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedCfg: T.HfPyramidSlopedTerrainCfg,
    HfRandomUniformCfg: T.HfRandomUniformTerrainCfg,
    HfWaveCfg: T.HfWaveTerrainCfg,
    HfPerlinNoiseCfg: T.HfPerlinNoiseTerrainCfg,
    HfDiscreteObstaclesCfg: T.HfDiscreteObstaclesTerrainCfg,
}


def mixed_sub_terrains(
    size: float = 12.0, solid: bool = True, stairs_platform: float | None = None, slope_platform: float | None = None
) -> dict[str, SubTerrainCfg]:
    """The 9 column types of plan §2.7 (order = column index). ``solid=False`` returns mjlab's own configs
    (thin stair boxes / height-field bases at low difficulty, see ``MIN_TERRAIN_THICKNESS``; diagnostics only).
    ``stairs_platform`` (evaluation / video only, never used in training): platform width of both stairs columns
    instead of the 6-step default; more steps per side, see ``stairs_layout``. ``slope_platform`` (same, evaluation /
    video only): central platform of both slope columns instead of 3 m."""
    if stairs_platform is None:
        stair_platform, stair_frame = round(size - 2 * STAIR_FRAME - 2 * STAIR_STEPS * STAIR_TREAD, 6), STAIR_FRAME  # 7.4 m on a 12 m tile
    else:  # mjlab floors the step count: pass a hair less so the platform box (the remainder) is exactly the request
        stair_platform, stair_frame = float(stairs_platform) - 1e-6, stairs_frame(size, stairs_platform)[1]
    stairs = {"step_height_range": (0.02, 0.15), "step_width": STAIR_TREAD, "platform_width": stair_platform, "border_width": stair_frame}
    slope = {"slope_range": (0.0, 0.36), "platform_width": 3.0 if slope_platform is None else float(slope_platform), "border_width": 0.25}
    c = (lambda cls: cls) if solid else (lambda cls: _MJLAB_CFG[cls])  # noqa: E731
    subs: dict[str, SubTerrainCfg] = {
        "flat": T.BoxFlatTerrainCfg(proportion=0.10),  # one 1 m thick box (mjlab make_plane)
        "stairs_up": c(BoxInvertedPyramidStairsCfg)(proportion=0.15, **stairs),  # pit: centre low
        "stairs_down": c(BoxPyramidStairsCfg)(proportion=0.15, **stairs),  # pyramid: centre high
        "slope_up": c(HfPyramidSlopedCfg)(proportion=0.10, inverted=True, **slope),
        "slope_down": c(HfPyramidSlopedCfg)(proportion=0.10, inverted=False, **slope),
        "rough": c(HfRandomUniformCfg)(
            proportion=0.10, noise_range=(0.0, 0.06), noise_step=0.005, downsampled_scale=0.2, border_width=0.25, scale_with_difficulty=True
        ),
        "wave": c(HfWaveCfg)(proportion=0.10, amplitude_range=(0.0, 0.15), num_waves=3, border_width=0.25),
        "perlin": c(HfPerlinNoiseCfg)(proportion=0.10, height_range=(0.0, 0.20), border_width=0.25),
        "blocks": c(HfDiscreteObstaclesCfg)(
            proportion=0.10, obstacle_width_range=(0.4, 1.2), obstacle_height_range=(0.02, 0.12), num_obstacles=60, platform_width=1.0, border_width=0.25
        ),
    }
    for sub in subs.values():
        sub.size = (size, size)
    return subs


def stairs_frame(size: float, platform: float) -> tuple[int, float]:
    """(steps per side, frame width) of the stairs variant whose central platform is exactly ``platform``: 30 cm treads
    as in training, the step count that keeps the frame nearest to 0.5 m, the frame takes the rest (1.4 m: 16 steps,
    0.5 m frame; 1.5 m: 16 steps, 0.45 m frame)."""
    n = round((size - 2 * STAIR_FRAME - platform) / (2 * STAIR_TREAD))
    return n, round((size - platform - 2 * n * STAIR_TREAD) / 2, 6)


def stairs_layout(size: float = 12.0, platform_width: float | None = None) -> tuple[int, float]:
    """(steps per side, platform side) of the stairs columns (``platform_width``: the ``stairs_platform`` variant, see
    ``stairs_frame``). The platform top is (steps + 1) step heights above (pyramid) / below (pit) the frame."""
    if platform_width is None:
        return STAIR_STEPS, round(size - 2 * STAIR_FRAME - 2 * STAIR_STEPS * STAIR_TREAD, 6)
    return stairs_frame(size, platform_width)[0], float(platform_width)


def make_terrain_generator_cfg(
    kind: str,
    seed: int,
    size: float = 12.0,
    num_rows: int = 10,
    border: float = 4.0,
    stairs_platform: float | None = None,
    slope_platform: float | None = None,
) -> TerrainGeneratorCfg | None:
    """mjlab generator config for ``kind`` (``None`` for ``plane``). The seed must be explicit: with
    ``seed=None`` mjlab falls back to the per-rank numpy RNG and every rank would build a different map.
    ``stairs_platform`` / ``slope_platform``: evaluation / video-only variants (see ``mixed_sub_terrains``); ``None`` = training map."""
    if kind not in TERRAIN_KINDS:
        raise ValueError(f"unknown terrain kind {kind!r}; expected one of {TERRAIN_KINDS}")
    if kind == "plane":
        return None
    if seed is None:
        raise ValueError("terrain seed must be explicit (identical on every rank)")
    subs = mixed_sub_terrains(size, stairs_platform=stairs_platform, slope_platform=slope_platform)
    return TerrainGeneratorCfg(
        seed=int(seed),
        curriculum=True,
        size=(size, size),
        border_width=border,
        num_rows=num_rows,
        num_cols=len(subs),
        sub_terrains=subs,
        difficulty_range=(0.0, 1.0),
    )


def compile_terrain(gen_cfg: TerrainGeneratorCfg) -> tuple[mujoco.MjModel, np.ndarray]:
    """Terrain-only MjModel (no robot) and the generator's terrain_origins (rows, cols, 3); tests and
    benchmarks. In an env use ``env.sim.mj_model`` and ``env.scene.terrain.terrain_origins`` instead."""
    from mjlab.terrains.terrain_generator import TerrainGenerator

    spec = mujoco.MjSpec()
    gen = TerrainGenerator(gen_cfg)
    gen.compile(spec)
    return spec.compile(), gen.terrain_origins.copy()


@dataclass(frozen=True)
class TerrainLayout:
    """Tile metadata of a curriculum terrain grid (no geometry)."""

    col_names: tuple[str, ...]  # terrain type per column
    num_rows: int
    tile_size: tuple[float, float]  # (x, y) extent of one tile, m
    border: float  # outer flat border width, m
    proportions: tuple[float, ...]  # normalised spawn weight per column
    difficulty_range: tuple[float, float] = (0.0, 1.0)

    @classmethod
    def from_generator_cfg(cls, cfg: TerrainGeneratorCfg) -> TerrainLayout:
        if not cfg.curriculum:
            raise ValueError("random-mode grids do not record per-tile type/difficulty; use curriculum=True")
        names = tuple(cfg.sub_terrains.keys())
        p = np.array([s.proportion for s in cfg.sub_terrains.values()], dtype=np.float64)
        return cls(
            col_names=names,
            num_rows=int(cfg.num_rows),
            tile_size=(float(cfg.size[0]), float(cfg.size[1])),
            border=float(cfg.border_width),
            proportions=tuple(float(x) for x in p / p.sum()),
            difficulty_range=(float(cfg.difficulty_range[0]), float(cfg.difficulty_range[1])),
        )

    @property
    def num_cols(self) -> int:
        return len(self.col_names)

    @property
    def row_difficulty(self) -> np.ndarray:
        """(num_rows,) generator difficulty of each row."""
        lo, hi = self.difficulty_range
        return lo + (hi - lo) * np.arange(self.num_rows) / max(self.num_rows - 1, 1)

    @property
    def grid_half_extent(self) -> tuple[float, float]:
        """Half extent (x, y) of the tile grid without the border; the grid is centred on the world origin."""
        return 0.5 * self.num_rows * self.tile_size[0], 0.5 * self.num_cols * self.tile_size[1]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(x_lo, x_hi, y_lo, y_hi) of the whole terrain including the outer border."""
        hx, hy = self.grid_half_extent
        return (-hx - self.border, hx + self.border, -hy - self.border, hy + self.border)

    def tile_centres(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """(num_rows, num_cols, 2) world xy of every tile centre (= mjlab's terrain_origins xy)."""
        hx, hy = self.grid_half_extent
        sx, sy = self.tile_size
        cx = -hx + (torch.arange(self.num_rows, dtype=torch.float64) + 0.5) * sx
        cy = -hy + (torch.arange(self.num_cols, dtype=torch.float64) + 0.5) * sy
        c = torch.stack(torch.meshgrid(cx, cy, indexing="ij"), dim=-1)
        return c.to(device=device, dtype=dtype)

    def col_index(self, name: str) -> int:
        return self.col_names.index(name)

    def check_origins(self, terrain_origins: np.ndarray | torch.Tensor, atol: float = 1e-4) -> None:
        """Assert that a compiled generator's ``terrain_origins`` (rows, cols, 3) match this layout."""
        o = torch.as_tensor(np.asarray(terrain_origins.cpu() if isinstance(terrain_origins, torch.Tensor) else terrain_origins))
        assert o.shape[:2] == (self.num_rows, self.num_cols), (o.shape, self.num_rows, self.num_cols)
        err = (o[..., :2].double() - self.tile_centres(dtype=torch.float64)).abs().max().item()
        assert err < atol, f"terrain_origins differ from the layout's tile centres by {err}"
