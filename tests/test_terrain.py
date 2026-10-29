"""Terrain package (plan §2.1-2.4, M1): raster and scan vs MuJoCo ground truth (CPU mj_ray and mjlab's
RayCastSensor), fp64 gradients, detach semantics, determinism + cache, spawn sampling / lift, path bbox,
plane mode and the policy scan-noise model. Uses a reduced 2-row ``mixed`` grid to stay fast."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
DEV = "cuda:0"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (MuJoCo Warp rasterisation)")
needs_npz = pytest.mark.skipif(not NPZ.exists(), reason="LAFAN1 npz not built")

N_ENVS = 24
SEED = 3
G0 = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # mj_ray: terrain (geom group 0) only


def _ray_height(m, d, xy: np.ndarray) -> np.ndarray:
    """(P,) CPU ground truth: top-surface height of the group-0 geoms under each xy (P, 2)."""
    gid = np.zeros(1, np.int32)
    z0, down = 10.0, np.array([0.0, 0.0, -1.0])
    return np.array([z0 - mujoco.mj_ray(m, d, np.array([x, y, z0]), down, G0, 1, -1, gid) for x, y in xy])


def _near_edge(m, d, xy: np.ndarray, res: float, tol: float = 0.005) -> np.ndarray:
    """(P,) bool: a step / kink within one cell (second differences over +-res along x, y and diagonals)."""
    offs = [(-res, 0), (res, 0), (0, -res), (0, res), (-res, -res), (res, res), (-res, res), (res, -res)]
    c = _ray_height(m, d, xy)
    nb = [_ray_height(m, d, xy + np.array(o)) for o in offs]
    sec = [np.abs(nb[k] + nb[k + 1] - 2 * c) for k in range(0, 8, 2)]
    return np.max(sec, axis=0) > tol


@pytest.fixture(scope="module")
def scene():
    """mjlab scene: 2-row mixed grid + G1 + mjlab's 17x11 yaw-aligned group-0 terrain scan on the pelvis."""
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
    from mjlab.sim import SimulationCfg
    from mjlab.sim.sim import Simulation
    from mjlab.terrains import TerrainEntityCfg

    from supermimic.terrain import (
        TerrainLayout,
        TerrainMap,
        TerrainMapCfg,
        make_terrain_generator_cfg,
    )

    gen_cfg = make_terrain_generator_cfg("mixed", seed=SEED, num_rows=2)
    sensor = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="pelvis", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )
    sc = Scene(SceneCfg(terrain=TerrainEntityCfg(terrain_type="generator", terrain_generator=gen_cfg), num_envs=N_ENVS, entities={"robot": get_g1_robot_cfg()}, sensors=(sensor,)), DEV)
    sim = Simulation(num_envs=N_ENVS, cfg=SimulationCfg(nconmax=64, njmax=300), model=sc.compile(), device=DEV)
    sc.initialize(sim.mj_model, sim.model, sim.data)
    sim.set_sensor_context(sc.sensor_context)
    layout = TerrainLayout.from_generator_cfg(gen_cfg)
    layout.check_origins(sc.terrain.terrain_origins)
    tmap = TerrainMap.build(sim.mj_model, gen_cfg, DEV, TerrainMapCfg(cache_dir=None))
    mjd = mujoco.MjData(sim.mj_model)
    mujoco.mj_forward(sim.mj_model, mjd)
    return SimpleNamespace(scene=sc, sim=sim, gen_cfg=gen_cfg, layout=layout, tmap=tmap, m=sim.mj_model, d=mjd)


@pytest.fixture(scope="module")
def fk64():
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.kinematics.g1_fk import G1Kinematics

    return G1Kinematics(load_g1_model(), device=DEV, dtype=torch.float64)


@pytest.fixture(scope="module")
def lib():
    from supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES

    files = discover_npz(NPZ, names=["walk1_subject1", "dance1_subject1", "jumps1_subject1"])
    return MotionLibrary(files, TRACKED_BODIES, torch.zeros(29), device=DEV)


def _tile_xy(layout, row: int, name: str, rng: np.random.Generator, n: int, r_lo: float, r_hi: float) -> np.ndarray:
    """n random xy at a distance in [r_lo, r_hi] (Chebyshev) from the centre of tile (row, name)."""
    c = layout.tile_centres()[row, layout.col_index(name)].numpy().astype(np.float64)
    out = []
    while len(out) < n:
        p = rng.uniform(-r_hi, r_hi, 2)
        if np.abs(p).max() >= r_lo:
            out.append(c + p)
    return np.array(out)


def _state_at(xy: np.ndarray, z: float, yaw: np.ndarray, dtype=torch.float64) -> torch.Tensor:
    n = xy.shape[0]
    s = torch.zeros(n, 71, dtype=dtype, device=DEV)
    s[:, 0:2] = torch.as_tensor(xy, dtype=dtype)
    s[:, 2] = z
    s[:, 3] = torch.as_tensor(np.cos(yaw / 2), dtype=dtype)
    s[:, 6] = torch.as_tensor(np.sin(yaw / 2), dtype=dtype)
    s[:, 7:36] = torch.as_tensor(np.random.default_rng(0).uniform(-0.3, 0.3, (n, 29)), dtype=dtype)
    return s


# ---------------------------------------------------------------------------------------------- (a) ground truth


def test_layout_and_preset_counts(scene):
    from supermimic.terrain import make_terrain_generator_cfg
    from supermimic.terrain.presets import compile_terrain

    lay = scene.layout
    assert lay.col_names == ("flat", "stairs_up", "stairs_down", "slope_up", "slope_down", "rough", "wave", "perlin", "blocks")
    assert lay.extent == (-16.0, 16.0, -58.0, 58.0)
    assert abs(sum(lay.proportions) - 1) < 1e-9 and lay.proportions[1] == pytest.approx(0.15)
    np.testing.assert_allclose(lay.row_difficulty, [0.0, 1.0])
    assert make_terrain_generator_cfg("plane", seed=0) is None
    m, _ = compile_terrain(make_terrain_generator_cfg("mixed", seed=0))  # full grid: plan §2.7 geom counts
    assert (m.ngeom, int((m.geom_type == mujoco.mjtGeom.mjGEOM_BOX).sum()), m.nhfield) == (654, 594, 60)


def test_raster_matches_mj_ray(scene):
    tm, m, d = scene.tmap, scene.m, scene.d
    spec = tm.spec
    assert tm.info["misses"] == 0 and tm.grid.h.dtype == torch.float16 and tm.grid.h.shape == (spec.nx, spec.ny)
    rng = np.random.default_rng(0)
    # Cell centres: the raster itself (fp16 rounding only).
    i, j = rng.integers(0, spec.nx, 2000), rng.integers(0, spec.ny, 2000)
    xs, ys = spec.cell_centres("cpu", torch.float64)
    gt = _ray_height(m, d, np.stack([xs[i].numpy(), ys[j].numpy()], -1))
    np.testing.assert_allclose(tm.grid.h[torch.as_tensor(i), torch.as_tensor(j)].float().cpu().numpy(), gt, atol=1e-3)
    # Random points: bilinear lookup < 1 cm except within one cell of a step edge / kink.
    x_lo, x_hi, y_lo, y_hi = spec.extent
    xy = rng.uniform([x_lo, y_lo], [x_hi, y_hi], size=(3000, 2))
    gt = _ray_height(m, d, xy)
    edge = _near_edge(m, d, xy, spec.res)
    h = tm.height(torch.as_tensor(xy, dtype=torch.float32, device=DEV)).cpu().numpy()
    err = np.abs(h - gt)
    assert edge.mean() < 0.1, edge.mean()
    assert err[~edge].max() < 0.01, err[~edge].max()
    # Out of bounds -> 0 and counted.
    q = torch.tensor([[x_hi + 0.5, 0.0], [0.0, y_lo - 3.0], [0.0, 0.0]], device=DEV)
    assert tm.height(q)[:2].abs().max().item() == 0 and tm.out_of_bounds(q).tolist() == [True, True, False]
    tm.count_out_of_bounds(q)
    assert tm.oob_count.item() == 2


def test_scan_matches_mjlab_raycast_sensor(scene):
    from mjlab.sensor import GridPatternCfg

    from supermimic.terrain.heightmap import SCAN_OFFSET, scan_grid_offsets

    # Same ray pattern and ordering as mjlab's GridPatternCfg.
    local, _ = GridPatternCfg(size=(1.6, 1.0), resolution=0.1).generate_rays(None, DEV)
    assert torch.equal(scan_grid_offsets(DEV), local[:, :2])

    tm, sim = scene.tmap, scene.sim
    rng = np.random.default_rng(1)
    hx, hy = scene.layout.grid_half_extent
    xy = rng.uniform([-hx + 1, -hy + 1], [hx - 1, hy - 1], size=(N_ENVS, 2))
    yaw, roll, pitch = rng.uniform(-np.pi, np.pi, N_ENVS), rng.uniform(-0.3, 0.3, N_ENVS), rng.uniform(-0.3, 0.3, N_ENVS)
    quat = np.zeros((N_ENVS, 4))
    for k in range(N_ENVS):
        mujoco.mju_euler2Quat(quat[k], np.array([roll[k], pitch[k], yaw[k]]), "xyz")
    root = torch.zeros(N_ENVS, 3, device=DEV)
    root[:, :2] = torch.as_tensor(xy, dtype=torch.float32)
    q = torch.as_tensor(quat, dtype=torch.float32, device=DEV)
    root[:, 2] = tm.scan_heights(root, q).amax(-1) + torch.as_tensor(rng.uniform(0.3, 0.9, N_ENVS), dtype=torch.float32, device=DEV)
    qpos = sim.data.qpos[:].clone()
    qpos[:, 0:3], qpos[:, 3:7] = root, q
    sim.data.qpos[:] = qpos
    sim.forward()
    sim.sense()
    data = scene.scene["terrain_scan"].data
    assert (data.distances >= 0).all()
    hit = data.hit_pos_w  # (N, 187, 3)
    pts = tm.scan_points(root, q)
    assert (pts - hit[..., :2]).abs().max().item() < 1e-4
    edge = _near_edge(scene.m, scene.d, hit[..., :2].reshape(-1, 2).double().cpu().numpy(), tm.spec.res).reshape(N_ENVS, -1)
    edge = torch.as_tensor(edge, device=DEV)
    assert edge.float().mean().item() < 0.15
    err_h = (tm.scan_heights(root, q) - hit[..., 2]).abs()
    assert err_h[~edge].max().item() < 0.01, err_h[~edge].max().item()
    ref_scan = (data.frame_pos_w[:, :, 2] - hit[..., 2] - SCAN_OFFSET).clamp(-1, 1)  # mjlab height_scan(offset=0.75), clipped
    assert (tm.scan(root, q) - ref_scan).abs()[~edge].max().item() < 0.01
    # (T, N) batches.
    s2 = tm.scan(root.expand(3, -1, -1), q.expand(3, -1, -1))
    assert s2.shape == (3, N_ENVS, 187) and torch.equal(s2[1], tm.scan(root, q))


# ---------------------------------------------------------------------------------------------- (b, c) gradients


def test_gradcheck_height_and_foot_features(scene, fk64):
    tm = scene.tmap
    g64 = tm.grid_as(torch.float64)
    rng = np.random.default_rng(2)
    xy = np.concatenate([_tile_xy(scene.layout, 1, name, rng, 3, 2.0, 5.5) for name in ("slope_up", "rough", "wave")])
    xy_t = torch.as_tensor(xy, dtype=torch.float64, device=DEV).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda p: tm.height(p, g64, detach_value=False), (xy_t,))
    assert torch.autograd.gradcheck(lambda p: tm.ground(p, g64, detach_value=False), (xy_t,))
    # Foot features w.r.t. the full state (root pose + joints) through FK.
    xy = _tile_xy(scene.layout, 1, "slope_down", rng, 4, 2.0, 5.0)
    s = _state_at(xy, 1.2, rng.uniform(-np.pi, np.pi, 4)).requires_grad_(True)
    origin = torch.tensor([[0.3, -0.2, 0.0]], dtype=torch.float64, device=DEV)
    f = lambda s_: tm.foot_features(s_, origin, fk64, g64, detach_value=False)
    assert f(s).shape == (4, 28)
    assert torch.autograd.gradcheck(f, (s,))
    assert torch.autograd.gradcheck(lambda s_: tm.scan_state(s_, origin, g64, detach_value=False), (s,))


def test_detach_value_semantics(scene, fk64):
    tm = scene.tmap
    g64 = tm.grid_as(torch.float64)
    rng = np.random.default_rng(3)
    xyz = torch.as_tensor(np.c_[_tile_xy(scene.layout, 1, "slope_up", rng, 6, 2.0, 5.0), np.ones(6)], dtype=torch.float64, device=DEV).requires_grad_(True)
    # detach_value=True: no gradient through the terrain heights, the query's own z still differentiates.
    h = tm.height(xyz[:, :2], g64)
    assert not h.requires_grad
    (gz,) = torch.autograd.grad((xyz[:, 2] - h).sum(), xyz)
    assert torch.equal(gz[:, :2], torch.zeros_like(gz[:, :2])) and torch.equal(gz[:, 2], torch.ones_like(gz[:, 2]))
    # detach_value=False: the slope comes through (36 % on this tile).
    (gs,) = torch.autograd.grad((xyz[:, 2] - tm.height(xyz[:, :2], g64, detach_value=False)).sum(), xyz)
    slope = gs[:, :2].norm(dim=-1)
    assert (slope > 0.1).all() and (slope < 0.6).all(), slope
    # Foot features: d clearance / d root z = 1 either way; d / d root x = -slope only without detach.
    s = _state_at(_tile_xy(scene.layout, 1, "slope_up", rng, 3, 2.0, 5.0), 0.5, np.zeros(3)).requires_grad_(True)
    o = torch.zeros(1, 3, dtype=torch.float64, device=DEV)
    for detach in (True, False):
        (g,) = torch.autograd.grad(tm.foot_features(s, o, fk64, g64, detach)[:, :10].sum(), s)
        torch.testing.assert_close(g[:, 2], torch.full_like(g[:, 2], 10.0))
        assert (g[:, 0].abs().max().item() == 0) == detach


# ---------------------------------------------------------------------------------------------- (d) determinism + cache


def test_same_seed_bit_identical_and_cache(tmp_path):
    from supermimic.terrain import TerrainMap, TerrainMapCfg, make_terrain_generator_cfg
    from supermimic.terrain.presets import compile_terrain

    def build(seed, cache_dir=None):
        gc = make_terrain_generator_cfg("mixed", seed=seed, num_rows=2)
        m, _ = compile_terrain(gc)
        return TerrainMap.build(m, gc, DEV, TerrainMapCfg(cache_dir=cache_dir))

    a, b, c = build(11), build(11), build(12)
    assert torch.equal(a.grid.h, b.grid.h) and torch.equal(a.grid.g, b.grid.g)
    assert not torch.equal(a.grid.h, c.grid.h)
    w = build(11, str(tmp_path))
    r = build(11, str(tmp_path))
    assert not w.info["cache_hit"] and r.info["cache_hit"] and len(list(tmp_path.glob("*.pt"))) == 1
    assert torch.equal(r.grid.h, a.grid.h) and torch.equal(r.grid.g, a.grid.g)
    assert build(12, str(tmp_path)).info["cache_key"] != r.info["cache_key"]


# ---------------------------------------------------------------------------------------------- (e, f) spawn, bbox


@needs_npz
def test_path_bbox_brute_force(lib):
    from supermimic.terrain import path_bbox

    xy = lib.root_pos_w[:, :2].cpu().numpy()
    last = lib.clip_last[lib.frame_clip].cpu().numpy()
    for w in (532, 37, 1, 0):
        bb = path_bbox(lib, w).cpu().numpy()
        frames = [0, 1, 250, int(lib.clip_last[0]) - 600, int(lib.clip_last[0]) - 3, int(lib.clip_last[0]), int(lib.clip_start[1]), int(lib.clip_last[-1]) - 100]
        for f in frames:
            seg = xy[f : min(f + w, last[f]) + 1] - xy[f]
            np.testing.assert_allclose(bb[f], np.r_[seg.min(0), seg.max(0)], atol=1e-6)
        assert (bb[:, :2] <= 0).all() and (bb[:, 2:] >= 0).all()


@needs_npz
def test_sample_spawn_and_lift(scene, lib):
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.terrain import (
        default_window_frames,
        lifted_reference,
        path_bbox,
        reference_lift,
        sample_spawn,
        spawn_lift,
    )
    from supermimic.terrain.reference import lib_root_xy
    from supermimic.world_model.wm_trainer import reference_at

    tm, layout = scene.tmap, scene.layout
    W, margin = default_window_frames(), 0.5
    bbox = path_bbox(lib, W)
    g = torch.Generator(device=DEV).manual_seed(0)
    n = 4096
    start = torch.randint(0, lib.num_frames, (n,), device=DEV, generator=g)
    sp = sample_spawn(lib, start, bbox, layout, level=1, margin=margin, generator=g)
    assert sp.origin.shape == (n, 3) and (sp.origin[:, 2] == 0).all()
    torch.testing.assert_close(sp.origin[:, :2] + lib_root_xy(lib, start), sp.spawn_xy)
    assert set(sp.row.unique().tolist()) == {0, 1}
    freq = torch.bincount(sp.col, minlength=layout.num_cols).float() / n
    torch.testing.assert_close(freq, torch.tensor(layout.proportions, device=DEV, dtype=torch.float32), atol=0.02, rtol=0)
    assert sp.fits.float().mean().item() > 0.8
    # Whole reference path of the window (+ margin) inside the chosen tile for the envs where it fits.
    centre = layout.tile_centres(DEV)[sp.row, sp.col]
    offs = torch.arange(W + 1, device=DEV)
    fr = torch.minimum(start[:, None] + offs, lib.last_frame_of(start)[:, None])
    path = sp.origin[:, None, :2] + lib_root_xy(lib, fr)
    dev_ = (path - centre[:, None]).abs().amax(dim=(1, 2))
    assert (dev_[sp.fits] <= 6.0 - margin + 1e-4).all()
    # level clamps and pinned tiles
    pinned = sample_spawn(lib, start[:8], bbox, layout, level=0, row=torch.ones(8, dtype=torch.long, device=DEV), col=torch.full((8,), 2, device=DEV))
    assert (pinned.row == 1).all() and (pinned.col == 2).all()

    # Lifted spawn pose: no sole point below the terrain after spawn_lift.
    fk = G1Kinematics(load_g1_model(), device=DEV)
    ref = lifted_reference(lib, start, sp.origin, tm)
    base = reference_at(lib, start)
    torch.testing.assert_close(ref.root_pos_w[:, :2], base.root_pos_w[:, :2])
    lift = ref.root_pos_w[:, 2] - base.root_pos_w[:, 2]
    torch.testing.assert_close(ref.body_pos_w[..., 2] - base.body_pos_w[..., 2], lift[:, None].expand_as(base.body_pos_w[..., 2]))
    torch.testing.assert_close(lift, tm.ground(sp.spawn_xy))
    # (T, N) frames with a per-env origin; optional vertical velocity of the lift.
    fT = torch.stack([start[:8], torch.minimum(start[:8] + 1, lib.last_frame_of(start[:8]))])
    refT = lifted_reference(lib, fT, sp.origin[:8], tm, vertical_velocity=True)
    assert refT.body_pos_w.shape == (2, 8, len(lib.body_names), 3)
    torch.testing.assert_close(refT.root_pos_w[0], ref.root_pos_w[:8])
    lT, vT = reference_lift(lib, fT, sp.origin[:8], tm, with_velocity=True)
    torch.testing.assert_close(vT[0], (lT[1] - lT[0]) * 50.0)
    torch.testing.assert_close(refT.body_lin_vel_w[..., 2] - lib.bodies(fT).lin_vel_w[..., 2], vT[..., None].expand(2, 8, len(lib.body_names)))
    r = lib.ref(start)
    qpos = torch.cat([ref.root_pos_w, r.root_quat_w, r.joint_pos], -1)
    extra = spawn_lift(qpos, sp.origin, fk, tm)
    assert (extra >= 0.01 - 1e-6).all()
    qpos[:, 2] += extra
    sole = tm.foot_points(qpos, sp.origin, fk).sole.reshape(n, 10, 3)
    clear = sole[..., 2] - tm.height(sole[..., :2])
    assert clear.min().item() >= 0.01 - 1e-4, clear.min().item()
    # ... and against the true geometry (CPU rays) on a subset.
    sub = sole[:256].reshape(-1, 3).double().cpu().numpy()
    true_clear = sub[:, 2] - _ray_height(scene.m, scene.d, sub[:, :2])
    assert true_clear.min() >= -1e-3, true_clear.min()


# ---------------------------------------------------------------------------------------------- (g) plane


@needs_npz
def test_plane_mode(lib):
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.terrain import WM_TERRAIN_DIMS, TerrainMap, lifted_reference, spawn_lift
    from supermimic.terrain.heightmap import SCAN_OFFSET
    from supermimic.world_model.wm_trainer import reference_at

    tm = TerrainMap.plane(DEV)
    assert tm.is_plane and tm.grid_as(torch.float64) is None
    fk = G1Kinematics(load_g1_model(), device=DEV)
    xy = torch.randn(5, 7, 2, device=DEV) * 100
    assert torch.equal(tm.height(xy), torch.zeros(5, 7, device=DEV)) and torch.equal(tm.ground(xy), torch.zeros(5, 7, device=DEV))
    assert not tm.out_of_bounds(xy).any()
    s = torch.zeros(3, 4, 71, device=DEV)
    s[..., 2], s[..., 3] = 0.8, 1.0
    origin = torch.randn(4, 3, device=DEV) * 10
    origin[:, 2] = 0
    sc = tm.scan_state(s, origin)
    assert sc.shape == (3, 4, 187)
    torch.testing.assert_close(sc, torch.full_like(sc, 0.8 - SCAN_OFFSET))
    ff = tm.foot_features(s, origin, fk)
    fp = tm.foot_points(s, origin, fk)
    assert ff.shape == (3, 4, 28)
    torch.testing.assert_close(ff[..., :10], fp.sole[..., 2].flatten(-2))
    torch.testing.assert_close(ff[..., 10:], (-fp.site[..., 2:3]).expand(3, 4, 2, 9).flatten(-2))
    for mode, dim in WM_TERRAIN_DIMS.items():
        assert tm.wm_condition(mode, s, origin, fk).shape == (3, 4, dim)
    f = torch.tensor([0, 10, 500], device=DEV)
    ref, base = lifted_reference(lib, f, origin[:3], tm), reference_at(lib, f)
    assert torch.equal(ref.body_pos_w, base.body_pos_w) and torch.equal(ref.root_pos_w, base.root_pos_w)
    assert torch.equal(spawn_lift(s[0, :, :36], origin, fk, tm), torch.zeros(4, device=DEV))


def test_wm_condition_layout(scene):
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.kinematics.g1_fk import G1Kinematics

    tm = scene.tmap
    fk = G1Kinematics(load_g1_model(), device=DEV)
    rng = np.random.default_rng(4)
    s = _state_at(_tile_xy(scene.layout, 1, "stairs_down", rng, 6, 1.0, 5.0), 1.6, rng.uniform(-3, 3, 6), torch.float32)
    o = torch.zeros(6, 3, device=DEV)
    foot, scan = tm.wm_condition("foot", s, o, fk), tm.wm_condition("scan", s, o, fk)
    both = tm.wm_condition("both", s, o, fk, scan_clean=scan)
    assert foot.shape == (6, 28) and scan.shape == (6, 187) and torch.equal(both, torch.cat([foot, scan], -1))
    # Patches are laid out in each foot's yaw frame: patch point (+6 cm, 0) lies along the ankle x-axis.
    from supermimic.kinematics.rotations import quat_apply
    from supermimic.terrain.heightmap import patch_offsets

    fp = tm.foot_points(s, o, fk)
    _, quat = fk.chain_pose(s[:, :36], fk.body_index("left_ankle_roll_link"))
    ax = quat_apply(quat, torch.tensor([1.0, 0.0, 0.0], device=DEV).expand(6, 3))[:, :2]
    torch.testing.assert_close(fp.heading[:, 0], torch.nn.functional.normalize(ax, dim=-1))
    assert patch_offsets(DEV)[5].tolist() == pytest.approx([0.06, 0.0])
    pt = fp.site[:, 0, :2] + 0.06 * fp.heading[:, 0]
    torch.testing.assert_close(foot[:, 10 + 5], tm.height(pt) - fp.site[:, 0, 2])
    # Flat ground: all patch values of a foot equal minus the site height.
    flat = _state_at(_tile_xy(scene.layout, 1, "flat", rng, 1, 0.0, 3.0), 0.8, np.zeros(1), torch.float32)
    p0 = tm.foot_features(flat, o[:1], fk)[:, 10:19]
    torch.testing.assert_close(p0, p0[:, :1].expand(-1, 9))


# ---------------------------------------------------------------------------------------------- (h) noise


def test_scan_noise_model():
    from supermimic.terrain import (
        ScanNoiseCfg,
        init_scan_history,
        noisy_scan,
        push_scan,
        resample_scan_noise_,
        sample_scan_noise,
    )

    n = 4096
    scans = [torch.rand(n, 187, device=DEV) * 1.6 - 0.8 for _ in range(4)]
    # Noise off: lag 0 is the current clean scan, exactly.
    off = ScanNoiseCfg(enabled=False)
    p = sample_scan_noise(n, off, DEV)
    hist = init_scan_history(scans[0], off.history_len)
    assert hist.shape == (3, n, 187) and torch.equal(noisy_scan(hist, p, off), scans[0])
    for sc in scans[1:]:
        hist = push_scan(hist, sc)
        assert torch.equal(noisy_scan(hist, p, off), sc)
    # Latency only: lag k reads the scan pushed k steps ago.
    lag_only = ScanNoiseCfg(point=0.0, offset_z=0.0, offset_xy=0.0, dropout=0.0)
    g = torch.Generator(device=DEV).manual_seed(0)
    p = sample_scan_noise(n, lag_only, DEV, g)
    assert set(p.lag.unique().tolist()) == {0, 1, 2}
    out = noisy_scan(hist, p, lag_only)
    for k in range(3):
        sel = p.lag == k
        assert torch.equal(out[sel], scans[3 - k][sel])
    # Full model: bounds, dropout rate, per-episode offsets, reproducibility with a generator.
    cfg = ScanNoiseCfg()
    p = sample_scan_noise(n, cfg, DEV, g)
    assert p.offset_z.abs().max() <= 0.03 and p.offset_xy.abs().max() <= 0.03 and p.offset_xy.shape == (n, 2)
    a = noisy_scan(hist, p, cfg, torch.Generator(device=DEV).manual_seed(5))
    b = noisy_scan(hist, p, cfg, torch.Generator(device=DEV).manual_seed(5))
    assert torch.equal(a, b) and a.abs().max() <= 1.0
    drop = a == 1.0
    assert 0.04 < drop.float().mean().item() < 0.06
    lagged = noisy_scan(hist, p, ScanNoiseCfg(enabled=False))
    resid = (a - (lagged - p.offset_z[:, None]))[~drop]
    assert resid.abs().max().item() <= 0.02 + 1e-6
    ids = torch.arange(0, n, 2, device=DEV)
    old = p.offset_z.clone()
    resample_scan_noise_(p, ids, cfg, g)
    assert torch.equal(p.offset_z[1::2], old[1::2]) and not torch.equal(p.offset_z[::2], old[::2])
    # (T, N) batch shape for the imagination side.
    hT = torch.stack([hist, hist], 1)  # (L, 2, n, 187)
    pT = sample_scan_noise((2, n), lag_only, DEV, g)
    assert noisy_scan(hT, pT, lag_only).shape == (2, n, 187)


def test_compiled_lookup_matches_eager(scene):
    tm = scene.tmap
    g = torch.Generator(device=DEV).manual_seed(0)
    root = torch.zeros(512, 3, device=DEV)
    root[:, :2] = (torch.rand(512, 2, device=DEV, generator=g) * 2 - 1) * torch.tensor([15.0, 50.0], device=DEV)
    root[:, 2] = 1.5
    q = torch.nn.functional.normalize(torch.randn(512, 4, device=DEV, generator=g), dim=-1)
    fn = torch.compile(lambda r, q, grid: tm.scan(r, q, grid), dynamic=False)
    # fp32 re-association only; next to a 15 cm step the bilinear slope is 6 m/m, so ~4e-6 m of xy rounding
    # at |y| ~ 50 m shows up as a few 1e-5 in the height.
    torch.testing.assert_close(fn(root, q, tm.grid), tm.scan(root, q), atol=1e-4, rtol=0)
