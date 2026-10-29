"""Rendered depth camera of the depth policy (terrain/depth.py, depth-direct plan M0):

* the height-map ray march matches MuJoCo Warp ray casting on the real mixed terrain geometry;
* the capsule intersection matches ``mujoco.mj_ray`` on random capsules / spheres;
* the frame history the simulator builds equals the one the imagination rebuilds from the replay buffer;
* the policy-side inputs only depend on what the robot can measure (g_cam: no root position / heading);
* the compiled sensor equals the eager one; a tiny depth training run (warm start, aux loss, finite weights).
"""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.terrain.depth import DepthCfg, DepthParams, DepthSensor, sample_depth_params

DEV = "cuda:0"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def fk():
    return G1Kinematics(load_g1_model(), device=DEV)


@pytest.fixture(scope="module")
def mixed():
    from supermimic.terrain.heightmap import TerrainMap, TerrainMapCfg
    from supermimic.terrain.presets import compile_terrain, make_terrain_generator_cfg

    gc = make_terrain_generator_cfg("mixed", seed=0, num_rows=4)
    m, _ = compile_terrain(gc)
    return m, TerrainMap.build(m, gc, DEV, TerrainMapCfg(cache_dir="logs/terrain_cache"))


def _standing(fk, tmap, n, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.zeros(n, 36, device=DEV)
    hx, hy = tmap.layout.grid_half_extent
    q[:, 0] = (torch.rand(n, device=DEV, generator=g) * 2 - 1) * (hx - 1)
    q[:, 1] = (torch.rand(n, device=DEV, generator=g) * 2 - 1) * (hy - 1)
    yaw = torch.rand(n, device=DEV, generator=g) * 2 * math.pi
    q[:, 3], q[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    q[:, 2] = tmap.height(q[:, 0:2]) + 0.78
    return q


def test_terrain_rays_match_mujoco_warp(fk, mixed):
    """z-depth of the 2.5D march vs MuJoCo Warp rays on the compiled terrain geoms (group 0): p99 error < 2 cm away
    from depth discontinuities (where the 2.5 cm raster and the bisection resolve the silhouette differently)."""
    import mujoco_warp as mjwarp
    import warp as wp

    m, tmap = mixed
    cfg = DepthCfg(render_body=False, noise=False)
    ds = DepthSensor(cfg, fk, tmap)
    n = 256
    q = _standing(fk, tmap, n)
    prm = sample_depth_params(n, cfg, DEV)
    z, _, cam_o, cam_R, _ = ds.render_raw(q, prm)
    P = ds.dirs.shape[0]
    d = torch.einsum("bij,pj->bpi", cam_R, ds.dirs)
    pnt = cam_o[:, None].expand(n, P, 3).reshape(1, -1, 3).contiguous()
    vec = d.reshape(1, -1, 3).contiguous()
    N = pnt.shape[1]
    dist = torch.empty(1, N, device=DEV)
    geomid = torch.empty(1, N, device=DEV, dtype=torch.int32)
    normal = torch.empty(1, N, 3, device=DEV)
    bex = torch.full((N,), -1, device=DEV, dtype=torch.int32)
    mjd = mujoco.MjData(m)
    mujoco.mj_forward(m, mjd)
    with wp.ScopedDevice(f"cuda:{torch.cuda.current_device()}"), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        wm, wd = mjwarp.put_model(m), mjwarp.put_data(m, mjd, nworld=1)
        rc = mjwarp.create_render_context(m, nworld=1, cam_active=[False] * m.ncam, enabled_geom_groups=[0], use_textures=False, use_shadows=False)
        vec6 = wp.types.vector(length=6, dtype=float)
        mjwarp.rays(wm, wd, wp.from_torch(pnt, dtype=wp.vec3), wp.from_torch(vec, dtype=wp.vec3), vec6(-1, 0, 0, 0, 0, 0), True, wp.from_torch(bex),
                    wp.from_torch(dist), wp.from_torch(geomid), wp.from_torch(normal, dtype=wp.vec3), rc=rc)
    torch.cuda.synchronize()
    ref = dist[0].view(n, P)
    ref = torch.where(ref < 0, torch.full_like(ref, 1e9), ref) * ds.dirs[None, :, 0]
    ref = ref.clamp(max=cfg.far)
    zz, rr = z.view(n, cfg.out_h, cfg.width), ref.view(n, cfg.out_h, cfg.width)
    from supermimic.terrain.depth import depth_edges

    edge = depth_edges(rr, 0.03) | depth_edges(zz, 0.03)
    edge = edge | torch.nn.functional.max_pool2d(edge[:, None].float(), 3, 1, 1)[:, 0].bool()  # +1 px around
    err = (zz - rr).abs()[~edge]
    assert (~edge).float().mean() > 0.5, "too many discontinuity pixels to judge"
    assert torch.quantile(err, 0.99) < 0.02, f"p99 {float(torch.quantile(err, 0.99)):.4f} m"
    assert float((err > 0.05).float().mean()) < 0.002


def test_capsule_hits_match_mj_ray(fk):
    """Body capsules: our kernel (plane, no terrain hits within range) vs mujoco.mj_ray on the same shapes."""
    from supermimic.terrain.depth import BodyCapsules
    from supermimic.terrain.heightmap import TerrainMap

    rng = np.random.default_rng(0)
    C = 12
    xml = ["<mujoco><worldbody>"]
    a, b, r = [], [], []
    for k in range(C):
        p0 = rng.uniform([0.4, -0.4, 0.6], [1.2, 0.4, 1.4])
        if k % 4 == 3:  # sphere
            p1 = p0.copy()
        else:
            p1 = p0 + rng.normal(size=3) * 0.15
        rad = float(rng.uniform(0.02, 0.08))
        a.append(p0), b.append(p1), r.append(rad)
        if np.allclose(p0, p1):
            xml.append(f'<geom type="sphere" size="{rad}" pos="{p0[0]} {p0[1]} {p0[2]}"/>')
        else:
            xml.append(f'<geom type="capsule" size="{rad}" fromto="{p0[0]} {p0[1]} {p0[2]} {p1[0]} {p1[1]} {p1[2]}"/>')
    xml.append("</worldbody></mujoco>")
    m = mujoco.MjModel.from_xml_string("".join(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cfg = DepthCfg(noise=False, mask_inflate=0.0, far=10.0)
    ds = DepthSensor(cfg, fk, TerrainMap.plane(DEV))
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=DEV)  # noqa: E731
    cam_o = t([[0.0, 0.0, 1.0]])
    cam_R = torch.eye(3, device=DEV)[None]
    caps = BodyCapsules(torch.zeros(C, dtype=torch.long, device=DEV), t(a), t(b), t(r))
    z, mask = torch.ops.supermimic.depth_rays(ds._dummy_h, ds._spec(), cam_o, cam_R, cam_o, cam_R, ds.dirs, caps.a[None], caps.b[None], caps.r, [0.02, 0.05, cfg.far, 5.0, 0.0])
    gid = np.zeros(1, dtype=np.int32)
    ref = []
    for dvec in ds.dirs.cpu().numpy().astype(np.float64):
        dist = mujoco.mj_ray(m, d, np.array([0.0, 0.0, 1.0]), dvec, None, 1, -1, gid)
        ref.append(dist * dvec[0] if dist >= 0 else np.inf)
    ref = torch.tensor(ref, dtype=torch.float32, device=DEV)
    hit_ref = torch.isfinite(ref)  # the scene has only the capsules; our kernel also sees the ground at z = 0 (>= 1.8 m)
    assert hit_ref.sum() > 20
    assert torch.equal(mask[0] > 0.5, hit_ref), "mask (no inflation, same pose) must equal the capsule hits"
    assert torch.allclose(z[0][hit_ref], ref[hit_ref], atol=1e-4)


def test_g_cam_is_root_position_and_heading_free(fk, mixed):
    """g_cam (a policy input) must not depend on what the robot cannot measure: root position and heading."""
    _, tmap = mixed
    cfg = DepthCfg(noise=False)
    ds = DepthSensor(cfg, fk, tmap)
    q = _standing(fk, tmap, 64, seed=1)
    q[:, 7:] = (torch.rand(64, 29, device=DEV) - 0.5) * 0.4
    prm = sample_depth_params(64, cfg, DEV)
    o = torch.zeros(64, 3, device=DEV)
    g0 = ds.observe(q, o, prm).g_cam
    q2 = q.clone()
    q2[:, 0:3] += torch.tensor([3.0, -2.0, 0.5], device=DEV)
    yaw = torch.full((64,), 1.1, device=DEV)
    qy = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], -1)
    from supermimic.kinematics.rotations import quat_mul

    q2[:, 3:7] = quat_mul(qy, q[:, 3:7])
    g1 = ds.observe(q2, o, prm).g_cam
    assert torch.allclose(g0, g1, atol=1e-5)
    # local qpos + origin == world qpos (the imagination's and the simulator's calls)
    org = torch.randn(64, 3, device=DEV)
    a = ds.observe(q, torch.zeros_like(org), prm)
    ql = q.clone()
    ql[:, 0:3] -= org
    b = ds.observe(ql, org, prm)
    # fp32 rounding of the camera position can move a bisection step (5 cm / 32): <= ~1.5 mm, i.e. 1e-3 normalised
    assert (a.img - b.img).abs().max() < 3e-3 and (a.img - b.img).abs().mean() < 1e-4
    assert torch.allclose(a.g_cam, b.g_cam, atol=1e-6)


def test_compiled_observe_matches_eager(fk, mixed):
    _, tmap = mixed
    cfg = DepthCfg(noise=False)
    ds = DepthSensor(cfg, fk, tmap)
    q = _standing(fk, tmap, 128, seed=2)
    prm = sample_depth_params(128, cfg, DEV)
    o = torch.zeros(128, 3, device=DEV)
    e = ds.observe(q, o, prm)
    c = torch.compile(ds.observe, dynamic=False)(q, o, prm)
    d = (e.img - c.img).abs()
    assert d.max() < 3e-3 and d.mean() < 1e-4  # last-bit pose differences can move a bisection step
    assert torch.allclose(e.g_cam, c.g_cam, atol=1e-6)


def test_sim_frame_history_matches_imagination(tmp_path):
    """Frames the simulator shows the policy == frames the imagination rebuilds from the logged states (noise off:
    fixed lag, no jitter / drops; phase taken from the simulator), incl. histories that reach before the episode."""
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.imagination.rollout import _init_depth_history

    dcfg = DepthCfg()
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=["data/npz/lafan1/walk1_subject1.npz"], num_envs=8, obs_noise=False, reset_noise=False, domain_randomization=False,
            terrain="mixed", terrain_rows=3, terrain_level=2, depth=dcfg, seed=0,
        ),
        device=DEV,
    )
    try:
        cmd = env.command_manager.get_term("motion")
        ds = cmd.depth_sensor
        buf = env.recorder_manager.get_term("replay").buffer
        env.reset()
        F = dcfg.hist_frames
        Hh = dcfg.start_hist
        g = torch.Generator(device=DEV).manual_seed(0)
        checked = diff_px = total_px = 0
        for k in range(150):
            snap_img, snap_g = cmd.depth_img.clone(), cmd.depth_g.clone()
            step_k = cmd.depth_step - 1  # step index at which the current state was processed
            phase_sim = cmd.depth_params.phase.clone()
            a = (torch.rand(8, 29, device=DEV, generator=g) - 0.5) * 0.2
            _, _, term, tout, _ = env.step(a)
            if k < 40 or k % 37:
                continue
            for e in range(8):
                if bool(term[e] | tout[e]):
                    continue
                mask = torch.zeros_like(buf.valid)
                mask[e, (buf.ptr[e] - 1) % buf.T] = True
                start = buf.sample_starts(1, Hh, extra_mask=mask)
                ph = ((step_k[e] + phase_sim[e]) % dcfg.sample_every).view(1)
                prm = DepthParams(*(x[e : e + 1].clone() for x in cmd.depth_params[:3]), ph)
                fn = SimpleNamespace(depth=ds, depth_noise=False, policy=SimpleNamespace(depth=SimpleNamespace(encoder=lambda img: img.flatten(1))))
                _, _, e_hist, _ = _init_depth_history(fn, start, cmd.tmap.grid, dparams=prm)
                im_img = e_hist[:, 0, :-3].view(F, 2, dcfg.out_h, dcfg.width)
                im_g = e_hist[:, 0, -3:]
                sim_img, sim_g = snap_img[:, e].float(), snap_g[:, e]
                if int(ph) == 0:  # the state itself was a sample step: the simulator already holds its frame
                    sim_img, sim_g, im_img, im_g = sim_img[:-1], sim_g[:-1], im_img[1:], im_g[1:]
                assert torch.allclose(sim_g, im_g, atol=1e-5), (k, e, float((sim_g - im_g).abs().max()))
                # Grazing rays at silhouettes may flip between hit / miss on last-bit pose differences (compiled vs
                # eager, local + origin): bound the fraction of differing pixels, not the maximum.
                diff_px += int(((sim_img - im_img).abs() > 2e-3).sum())
                total_px += sim_img.numel()
                checked += 1
        assert checked >= 8
        assert diff_px / total_px < 1e-3, diff_px / total_px
    finally:
        env.close()


def test_training_smoke_depth(tmp_path):
    """Warm start from a flat checkpoint into a depth policy on ``mixed``: fresh depth modules, zero z_terrain columns,
    aux loss logged, compiled imagination, finite weights."""
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    from tests.test_terrain_stack import _tiny

    flat = _tiny(TrainConfig(), "flat", tmp_path)
    flat.schedule.total_iterations = 6
    flat.schedule.checkpoint_interval = 6
    tr = Trainer(flat)
    tr.run()
    tr.env.close()

    cfg = _tiny(TrainConfig(), "depth", tmp_path)
    cfg.init_from = tmp_path / "flat" / "ckpt_latest.pt"
    cfg.terrain.kind = "mixed"
    cfg.terrain.num_rows = 3
    cfg.terrain.wm_terrain = "scan"
    cfg.terrain.policy_obs = "depth"
    cfg.schedule.total_iterations = 8
    cfg.schedule.checkpoint_interval = 8
    tr = Trainer(cfg)
    assert tr.policy.depth is not None and tr.policy.terrain is None
    w = tr.policy.decoder.mlp.net[0].weight
    assert torch.equal(w[:, 32:64], torch.zeros_like(w[:, 32:64]))
    tr.run()
    assert all(torch.isfinite(p).all() for m in (tr.policy, tr.critic, tr.wm) for p in m.parameters())
    m = tr.policy_trainer.update(tr.buffer)
    assert m["depth/aux_loss"] > 0 and 0 < m["depth/aux_visible_frac"] < 1
    tr.env.close()
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    pol, _ = load_policy_from_checkpoint(tmp_path / "depth" / "ckpt_latest.pt", DEV)
    assert pol.depth is not None
