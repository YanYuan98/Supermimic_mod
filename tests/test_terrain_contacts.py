"""Terrain contact quality (VISION_TERRAIN_PLAN Phase C): solid terrain geometry, MuJoCo Warp's flipped normals on thin
solids and on deeply penetrating capsules (and the height-field normal filter), the edge-distance map, the stacked
foot FK against the per-foot path, and the benchmark's tile-drift bookkeeping."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

DEV = "cuda:0"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (MuJoCo Warp)")


def _mixed_model(solid: bool, seed: int = 3, rows: int = 2):
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    from supermimic.terrain.presets import compile_terrain, mixed_sub_terrains

    gen = TerrainGeneratorCfg(seed=seed, curriculum=True, size=(12.0, 12.0), border_width=4.0, num_rows=rows, num_cols=9, sub_terrains=mixed_sub_terrains(12.0, solid=solid), difficulty_range=(0.0, 1.0))
    return compile_terrain(gen)[0]


def test_solid_terrain_thickens_without_moving_surfaces():
    from supermimic.terrain.presets import MIN_TERRAIN_THICKNESS, make_terrain_generator_cfg

    raw, solid = _mixed_model(False), _mixed_model(True)
    box = solid.geom_type == mujoco.mjtGeom.mjGEOM_BOX
    hf = solid.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD
    # mjlab's difficulty-0 tiles are thin (the Phase B failure): hfield bases of 1-5 mm, 2 cm stair boxes.
    assert raw.hfield_size[:, 3].min() <= 0.005 and (2 * raw.geom_size[box, 2]).min() <= 0.021
    assert solid.hfield_size[:, 3].min() >= MIN_TERRAIN_THICKNESS and (2 * solid.geom_size[box, 2]).min() >= MIN_TERRAIN_THICKNESS - 1e-9
    # Surfaces unchanged: box tops, xy, hfield elevation data / sizes / poses.
    top = lambda m: m.geom_pos[box, 2] + m.geom_size[box, 2]  # noqa: E731
    np.testing.assert_allclose(top(solid), top(raw), atol=1e-9)
    np.testing.assert_array_equal(solid.geom_pos[:, :2], raw.geom_pos[:, :2])
    np.testing.assert_array_equal(solid.geom_pos[hf], raw.geom_pos[hf])
    np.testing.assert_array_equal(solid.hfield_size[:, :3], raw.hfield_size[:, :3])
    np.testing.assert_array_equal(solid.hfield_data, raw.hfield_data)
    gen = make_terrain_generator_cfg("mixed", seed=0)  # the training preset is the solid one
    assert all(getattr(c, "min_thickness", MIN_TERRAIN_THICKNESS) == MIN_TERRAIN_THICKNESS for c in gen.sub_terrains.values())


def _capsule_scene(ground: str, half: float = 0.04) -> mujoco.MjModel:
    xml = f"""<mujoco><option ccd_iterations="100"><flag multiccd="enable"/></option>
      <asset><hfield name="hf" nrow="61" ncol="61" size="3 3 0.005 {{base}}"/></asset>
      <worldbody>{ground}<body pos="0 0 0.5"><freejoint/><geom type="capsule" size="0.01 {half}" mass="1"/></body></worldbody></mujoco>"""
    return xml


def _contacts(xml: str, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(world id, contact normal z from the ground to the capsule, dist) of every contact (MuJoCo Warp collision)."""
    import mujoco_warp as mjw
    import warp as wp

    m = mujoco.MjModel.from_xml_string(xml)
    with wp.ScopedDevice(DEV):
        mw = mjw.put_model(m)
        dw = mjw.put_data(m, mujoco.MjData(m), nworld=qpos.shape[0], nconmax=8, njmax=32)
        wp.copy(dw.qpos, wp.array(qpos.astype(np.float32), dtype=float))
        mjw.kinematics(mw, dw)
        mjw.collision(mw, dw)
        n = int(dw.nacon.numpy()[0])
        geom = dw.contact.geom.numpy()[:n]
        nz = dw.contact.frame.numpy()[:n, 0, 2] * np.where(geom[:, 0] == 0, 1.0, -1.0)
        return dw.contact.worldid.numpy()[:n], nz, dw.contact.dist.numpy()[:n]


def _flat_capsule_qpos(pens: np.ndarray) -> np.ndarray:
    q = np.zeros((len(pens), 7))
    q[:, 0:2] = [0.013, 0.021]
    q[:, 2] = 0.01 - pens  # capsule bottom at -pen (surface at z = 0)
    q[:, 3:7] = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]  # axis horizontal
    return q


@needs_cuda
@pytest.mark.parametrize("kind", ["hfield", "box"])
def test_thin_solids_flip_contact_normals(kind):
    """MuJoCo Warp resolves along the minimum-translation direction: once the capsule centre is below the solid's
    mid-thickness (penetration > radius + thickness / 2) the normal points down; the solid presets never get there."""
    from supermimic.terrain.mjwarp_patch import install_hfield_normal_filter, uninstall_hfield_normal_filter
    from supermimic.terrain.presets import MIN_TERRAIN_THICKNESS

    pens = np.array([0.004, 0.009, 0.014, 0.016] if kind == "hfield" else [0.004, 0.009, 0.022, 0.026])  # still overlapping
    q = _flat_capsule_qpos(pens)
    uninstall_hfield_normal_filter()  # mixed envs install it process-wide; this documents the raw behaviour
    try:
        for thickness, expect_flip in ((0.005 if kind == "hfield" else 0.02, True), (MIN_TERRAIN_THICKNESS, False)):
            ground = '<geom type="hfield" hfield="hf"/>' if kind == "hfield" else f'<geom type="box" size="3 3 {thickness / 2}" pos="0 0 {-thickness / 2}"/>'
            wid, nz, _ = _contacts(_capsule_scene(ground).replace("{base}", str(thickness)), q)
            min_nz = np.array([nz[wid == i].min() for i in range(len(pens))])
            assert (min_nz[:2] > 0.99).all(), min_nz  # shallow: up
            assert ((min_nz[2:] < -0.99).all() if expect_flip else (min_nz > -0.5).all()), (kind, thickness, min_nz)
    finally:
        install_hfield_normal_filter()


def _random_tilted_capsules(n: int, rng, half: float = 0.091) -> tuple[np.ndarray, np.ndarray]:
    yaw = rng.uniform(-np.pi, np.pi, n)
    tilt = rng.uniform(-0.3, 0.3, n)
    pen = rng.uniform(0.0, 0.015, n)
    a = (np.pi / 2 + tilt) / 2
    qy = np.stack([np.cos(a), 0 * a, np.sin(a), 0 * a], -1)
    cz, sz = np.cos(yaw / 2), np.sin(yaw / 2)
    quat = np.stack([cz * qy[:, 0], -sz * qy[:, 2], cz * qy[:, 2], sz * qy[:, 0]], -1)  # qz(yaw) * qy
    q = np.zeros((n, 7))
    q[:, 0:2] = rng.uniform(-2.4, 2.4, (n, 2))
    q[:, 2] = 0.01 + half * np.abs(np.sin(tilt)) - pen
    q[:, 3:7] = quat
    return q, pen


@needs_cuda
def test_hfield_normal_filter_drops_reversed_epa_contacts():
    """Deep (> radius) capsule penetration of a thick flat height field: EPA returns reversed normals for some prisms;
    the patch drops exactly those, never leaves a penetrating capsule without contact and keeps shallow contacts."""
    from supermimic.terrain.mjwarp_patch import install_hfield_normal_filter, uninstall_hfield_normal_filter

    xml = _capsule_scene('<geom type="hfield" hfield="hf"/>', half=0.091).replace("{base}", "0.5")
    q, pen = _random_tilted_capsules(20000, np.random.default_rng(0))
    uninstall_hfield_normal_filter()
    w0, nz0, d0 = _contacts(xml, q)
    install_hfield_normal_filter()
    assert not install_hfield_normal_filter()  # idempotent
    w1, nz1, d1 = _contacts(xml, q)
    assert (nz0 < -0.5).mean() > 0.005  # the MuJoCo Warp failure mode this patch exists for
    assert (nz1 < -0.1).sum() == 0
    touching = np.nonzero(pen > 5e-4)[0]
    assert (np.isin(touching, w1) | ~np.isin(touching, w0)).all()  # no penetrating capsule loses all its contacts
    # Worlds whose capsule cores stay outside the field (penetration < radius) see the same contacts.
    for i in np.nonzero(pen < 0.009)[0][:200]:
        a, b = np.sort(d0[w0 == i]), np.sort(d1[w1 == i])
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, atol=1e-7)


def test_edge_distance_map():
    from supermimic.terrain.heightmap import edge_distance

    res = 0.025
    h = torch.zeros(40, 30)
    h[20:] = 0.05  # a 5 cm step between x-cells 19 and 20
    h[:, 25:] += 0.02  # a 2 cm step along y (below the 3 cm threshold)
    d = edge_distance(h, res, threshold=0.03, max_dist=0.5)
    assert d[19].max() == 0 and d[20].max() == 0  # both sides of the step are edge cells
    np.testing.assert_allclose(d[10, 5].item(), 9 * res, atol=1e-6)
    np.testing.assert_allclose(d[35, 5].item(), 15 * res, atol=1e-6)
    assert d[0, 0].item() == pytest.approx(0.475) and d[:, 25:].min() == 0  # cap; the 2 cm step is not an edge here
    assert d[5, 24].item() == pytest.approx(14 * res)
    d2 = edge_distance(h, res, threshold=0.01, max_dist=0.5)
    assert d2[5, 24].item() == 0  # ... but it is with a 1 cm threshold


def _foot_points_per_foot(tmap, qpos, origin, fk):
    """Phase B ``TerrainMap.foot_points``: one ``chain_pose`` per foot."""
    from supermimic.kinematics.rotations import quat_apply
    from supermimic.terrain.heightmap import SOLE_POINTS, yaw_cos_sin

    q = qpos[..., :36]
    batch = q.shape[:-1]
    sole, site, heading = [], [], []
    for k, name in enumerate(tmap.feet.body_names):
        pos, quat = fk.chain_pose(q, fk.body_index(name))
        pos = pos + origin
        sl = tmap.sole_local[k].to(pos.dtype).expand(*batch, SOLE_POINTS, 3)
        sole.append(pos[..., None, :] + quat_apply(quat[..., None, :].expand(*batch, SOLE_POINTS, 4), sl))
        site.append(pos + quat_apply(quat, tmap.site_local[k].to(pos.dtype).expand(*batch, 3)))
        heading.append(yaw_cos_sin(quat))
    return torch.stack(sole, -3), torch.stack(site, -2), torch.stack(heading, -2)


@pytest.mark.parametrize("dtype,atol", [(torch.float64, 1e-12), (torch.float32, 2e-5)])
def test_foot_points_match_per_foot_path(dtype, atol):
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.terrain.heightmap import TerrainMap

    model = load_g1_model()
    fk = G1Kinematics(model, "cpu", dtype)
    tmap = TerrainMap.plane("cpu")
    g = torch.Generator().manual_seed(0)
    q = torch.randn(3, 128, 36, generator=g, dtype=dtype)
    q[..., 3:7] = q[..., 3:7] / q[..., 3:7].norm(dim=-1, keepdim=True)
    origin = torch.randn(1, 128, 3, generator=g, dtype=dtype)
    new = tmap.foot_points(q, origin, fk)
    old = _foot_points_per_foot(tmap, q, origin, fk)
    for a, b in zip(new, old):
        assert a.shape == b.shape
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=atol)


def test_benchmark_tile_drift_bookkeeping():
    from supermimic.eval.benchmark import EVAL_TERRAIN_BORDER, ClipResult, format_terrain_table, summarize_terrain, terrain_env_kwargs_for_eval

    mk = lambda failed, left: ClipResult("c", 100, 50, 0.5, 0.8, failed, 50 if failed else -1, terrain="rough", row=0, left_tile_step=left)  # noqa: E731
    t = summarize_terrain([mk(True, -1), mk(True, 30), mk(False, 10), mk(False, -1)])["rough"]["0"]
    assert (t["clips"], t["failed"], t["failed_on_tile"], t["left_tile"]) == (4, 2, 1, 2)
    assert "  1" in format_terrain_table({"rough": {"0": t}}, "failed_on_tile")
    kw = terrain_env_kwargs_for_eval({}, "mixed")
    assert kw["terrain_border"] == EVAL_TERRAIN_BORDER and kw["terrain_seed"] == 1000
    assert terrain_env_kwargs_for_eval({}, "mixed", border=None)["terrain_border"] == 4.0


def test_m2_ablation_statistics():
    """scripts/wm_terrain_ablation.py: per-group window statistics and their window-weighted merge."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("wm_terrain_ablation", Path(__file__).resolve().parents[1] / "scripts/wm_terrain_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    err = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])  # (k=2, n=4)
    logits = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [1.0, -1.0, -1.0, 1.0]])
    done = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    sel = torch.tensor([True, True, False, True])
    g = mod.group_stats(err, logits, done, sel)
    assert g["n"] == 3 and g["nse_1"] == pytest.approx(7 / 3) and g["nse_k"] == pytest.approx(14 / 3)
    assert g["term_pos"] == 2 and g["term_recall"] == pytest.approx(0.5) and g["term_acc"] == pytest.approx(4 / 6)
    assert mod.group_stats(err, logits, done, torch.zeros(4, dtype=torch.bool)) == {"n": 0}
    a = {"_counts": {"windows": 3}, "foot": {"all": g}}
    b = {"_counts": {"windows": 1}, "foot": {"all": mod.group_stats(err, logits, done, torch.tensor([False, False, True, False]))}}
    m = mod.merge_evals([a, b], ["foot"])
    assert m["_counts"]["windows"] == 4 and m["foot"]["all"]["n"] == 4
    assert m["foot"]["all"]["nse_k"] == pytest.approx(err[1].mean().item()) and m["foot"]["all"]["term_recall"] == pytest.approx(0.5)


def test_foot_fine_features_layout():
    """wm_terrain="foot_fine": the 5 x 3 sole grid contains the coarse corners (same clearances), the 5 x 5 patch is
    centred on the sole centre in the foot's yaw frame, values relative to the sole-centre height."""
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.kinematics.rotations import quat_apply
    from supermimic.terrain.heightmap import FOOT_FINE_DIM, WM_TERRAIN_DIMS, FootGeometry, GridSpec, HeightGrid, TerrainMap

    model = load_g1_model()
    fk = G1Kinematics(model, "cpu")
    spec = GridSpec(-2.0, -2.0, 0.025, 160, 160)
    xs, _ = spec.cell_centres()
    h = torch.where(xs[:, None] > 0.07, 0.08, 0.0).expand(160, 160).contiguous()  # an 8 cm step at x = 0.07
    tmap = TerrainMap(HeightGrid(h, h), spec, FootGeometry.from_model(model), "cpu")
    g = torch.Generator().manual_seed(0)
    q = torch.zeros(16, 36)
    q[:, 0:2] = torch.rand(16, 2, generator=g) * 0.4 - 0.2
    q[:, 2] = 0.79
    yaw = torch.rand(16, generator=g) * 6.28
    q[:, 3], q[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    q[:, 7:] = (torch.rand(16, 29, generator=g) - 0.5) * 0.4
    o = torch.zeros(16, 3)
    fine, coarse = tmap.wm_condition("foot_fine", q, o, fk), tmap.foot_features(q, o, fk)
    assert fine.shape == (16, FOOT_FINE_DIM) == (16, WM_TERRAIN_DIMS["foot_fine"])
    corners = [0, 2, 12, 14]  # (x_lo, y_lo), (x_lo, y_hi), (x_hi, y_lo), (x_hi, y_hi) of the x-major 5 x 3 grid
    for foot in range(2):
        torch.testing.assert_close(fine[:, 15 * foot + torch.tensor(corners)], coarse[:, 5 * foot : 5 * foot + 4])
    pos, quat = fk.chain_poses(q, (fk.body_index("left_ankle_roll_link"), fk.body_index("right_ankle_roll_link")))
    centre = pos + quat_apply(quat, tmap.sole_local[:, 4].expand(16, 2, 3))
    heading = tmap.foot_points(q, o, fk).heading
    torch.testing.assert_close(fine[:, 30 + 12], tmap.height(centre[:, 0, :2]) - centre[:, 0, 2])  # left patch centre
    toe = centre[:, 1, :2] + 0.10 * heading[:, 1]  # right patch (+2 x 5 cm, 0): element 25 + 14
    torch.testing.assert_close(fine[:, 55 + 14], tmap.height(toe) - centre[:, 1, 2])
