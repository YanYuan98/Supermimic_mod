"""World-model unit tests: state spec consistency, identity at init, gradients, and a quick overfit."""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"


def _random_states(n, device, dtype=torch.float32):
    s = torch.randn(n, 71, device=device, dtype=dtype)
    s[:, 3:7] = s[:, 3:7] / s[:, 3:7].norm(dim=-1, keepdim=True)
    return s


def test_state_spec_identity_and_errors():
    from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

    dev = "cpu"
    spec = StateSpec(StateSpecCfg(explicit_integration=False), dev, torch.float64)
    s = _random_states(64, dev, torch.float64)
    s2 = spec.integrate(s, torch.zeros(64, 70, dtype=torch.float64))
    np.testing.assert_allclose(s2.numpy(), s.numpy(), atol=1e-12)
    assert spec.normalized_sq_error(s, s).max() < 1e-20
    # A pure yaw rotation + xy translation of the state leaves the canonical features unchanged.
    from supermimic.kinematics.rotations import quat_apply, quat_from_yaw, quat_mul

    yaw = torch.rand(64, dtype=torch.float64) * 6.28 - 3.14
    qy = quat_from_yaw(yaw)
    s_rot = s.clone()
    s_rot[:, 0:2] += torch.randn(64, 2, dtype=torch.float64)
    s_rot[:, 3:7] = quat_mul(qy, s[:, 3:7])
    s_rot[:, 36:39] = quat_apply(qy, s[:, 36:39])
    np.testing.assert_allclose(spec.canonical(s_rot).numpy(), spec.canonical(s).numpy(), atol=1e-9)
    # Explicit integration with zero delta = constant-velocity extrapolation.
    spec2 = StateSpec(StateSpecCfg(explicit_integration=True), dev, torch.float64)
    s3 = spec2.integrate(s, torch.zeros(64, 70, dtype=torch.float64))
    np.testing.assert_allclose(s3[:, 0:3].numpy(), (s[:, 0:3] + 0.02 * s[:, 36:39]).numpy(), atol=1e-12)
    np.testing.assert_allclose(s3[:, 7:36].numpy(), (s[:, 7:36] + 0.02 * s[:, 42:71]).numpy(), atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
@pytest.mark.parametrize("norm", ["fixed", "running"])
def test_world_model_overfits_replay_windows(norm):
    """Collect a little data with a random policy, fit the WM for a few hundred steps: error must drop a lot."""
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    files = discover_npz(NPZ, names=["walk1_subject1", "dance1_subject1"])
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=256, replay_capacity=400, seed=5), "cuda:0")
    try:
        env.reset()
        for _ in range(300):
            env.step(0.3 * torch.randn(256, 29, device=env.device))
        rec = env.recorder_manager.get_term("replay")
        robot = env.scene["robot"]
        fk = G1Kinematics(load_g1_model(), device=env.device)
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        wm = WorldModel(WorldModelCfg(hidden=(512, 512, 512), normalization=norm, layer_norm=norm == "running"), heads, env.device)
        lib = env.command_manager.get_term("motion").library
        tr = WorldModelTrainer(wm, lib, WMTrainerCfg(batch_size=512, batches_per_update=1, lr=1e-3))
        e0 = tr.online_error(rec.buffer, n=1024, k=8)
        for _ in range(300):
            m = tr.update(rec.buffer)
        e1 = tr.online_error(rec.buffer, n=1024, k=8)
        print("\nbefore:", [f"{v:.4f}" for v in e0.tolist()], "\nafter: ", [f"{v:.4f}" for v in e1.tolist()], "\nlast metrics:", m)
        assert torch.isfinite(e1).all()
        assert e1[0] < 0.5 * e0[0], (e0[0].item(), e1[0].item())
        assert e1[-1] < 0.5 * e0[-1], (e0[-1].item(), e1[-1].item())
        # BPTT gradient through a few imagined steps reaches the action input.
        w = rec.buffer.sample_windows(32, 4)
        a = w.a_exec.clone().requires_grad_(True)
        s = w.states[0]
        for j in range(4):
            s = wm.step(s, a[j], w.xi)
        s.square().sum().backward()
        assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
    finally:
        env.close()


@pytest.mark.skipif(not NPZ.exists(), reason="needs LAFAN1 npz")
@pytest.mark.parametrize("mode", ["foot", "scan", "both"])
def test_world_model_terrain_condition(mode):
    """VISION_TERRAIN_PLAN §2.4: the k-step window recomputes c_t from the *predicted* state at every step
    (same as a manual step_at loop), terr_norm is separate from in_norm, canonical uses the height above the
    ground, and the condition really reaches the dynamics (gradient on the new input columns)."""
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.terrain import WM_TERRAIN_DIMS, TerrainInputs, TerrainMap
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    dev = "cpu"
    files = discover_npz(NPZ, names=["walk1_subject1"])
    lib = MotionLibrary(files, TRACKED_BODIES, torch.zeros(29), device=dev)
    model = load_g1_model()
    fk = G1Kinematics(model, device=dev)
    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, torch.as_tensor(model.jnt_range[1:], dtype=torch.float32), torch.zeros(29))
    tin = TerrainInputs(TerrainMap.plane(dev), fk, mode)
    with pytest.raises(ValueError):
        WorldModel(WorldModelCfg(hidden=(64, 64), terrain=mode), heads, dev)  # terrain WM needs TerrainInputs
    torch.manual_seed(0)
    wm = WorldModel(WorldModelCfg(hidden=(64, 64), terrain=mode), heads, dev, terrain=tin)
    for p in wm.dynamics.parameters():
        p.data.normal_(0, 0.05)
    assert wm.terr_dim == WM_TERRAIN_DIMS[mode] and wm.dynamics.mlp.net[0].in_features == 133 + wm.terr_dim
    k, n = 4, 32
    f = torch.randint(0, lib.num_frames - k - 1, (n,))
    states = torch.stack([torch.cat(lib.ref_qpos_qvel(f + j), -1) for j in range(k + 1)])
    a, xi = torch.randn(k, n, 29) * 0.1, torch.zeros(n, 33)
    origin = torch.randn(n, 3)
    origin[:, 2] = 0
    wm.update_normalizers(states[0], a[0], xi, states[1], origin=origin)
    assert wm.terr_norm.count == n and wm.in_norm.count == n
    done = torch.zeros(k, n)
    r = lib.ref(f)
    rp, rq = r.root_pos_w[None].expand(k, n, 3), r.root_quat_w[None].expand(k, n, 4)
    loss, m = wm.window_loss(states, a, xi, done, rp, rq, origin=origin)
    s, sq = states[0], 0.0
    for j in range(k):
        terr, gz = tin.wm_condition(s, origin)
        s = wm.step(s, a[j], xi, terr=terr, ground_z=gz)
        sq = sq + wm.spec.error_vector(s, states[j + 1]).square().sum(-1).mean()
    torch.testing.assert_close(m["state_loss"], sq / k)
    torch.testing.assert_close(wm.rollout_error(states, a, xi, origin=origin)[-1], wm.spec.normalized_sq_error(s, states[k]).mean())
    with pytest.raises(AssertionError):
        wm.step(states[0], a[0], xi)  # terrain WM without its condition
    loss.backward()
    assert wm.dynamics.mlp.net[0].weight.grad[:, 133:].abs().sum() > 0


@pytest.mark.skipif(not NPZ.exists(), reason="needs LAFAN1 npz")
@pytest.mark.parametrize("mode", ["none", "foot"])
def test_update_on_matches_update(mode):
    """M2 ablation path: ``update_on`` on the batches ``update`` would sample (same RNG) gives the same parameters
    and metrics; ``window_errors`` agrees with ``rollout_error`` and the window's termination logits."""
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.replay_buffer import ReplayBuffer
    from supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.terrain import TerrainInputs, TerrainMap
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    dev = "cpu"
    lib = MotionLibrary(discover_npz(NPZ, names=["walk1_subject1"]), TRACKED_BODIES, torch.zeros(29), device=dev)
    model = load_g1_model()
    fk = G1Kinematics(model, device=dev)
    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, torch.as_tensor(model.jnt_range[1:], dtype=torch.float32), torch.zeros(29))
    tin = TerrainInputs(TerrainMap.plane(dev), fk, mode)
    E, T = 8, 24
    buf = ReplayBuffer(E, T, 71, 29, 33, dev)
    f0 = torch.randint(0, lib.num_frames - T - 1, (E,), generator=torch.Generator().manual_seed(1))
    ids = torch.arange(E)
    for t in range(T):
        s = torch.cat(lib.ref_qpos_qvel(f0 + t), -1)
        z = torch.zeros(E)
        buf.add(ids, s, s[:, 7:36], s[:, 7:36], torch.zeros(E, 33), z, z, f0 + t, lib.frame_clip[f0 + t], z.bool(), torch.full((E,), t == 0), origin=torch.zeros(E, 3))
    cfg = WMTrainerCfg(window=4, batches_per_update=3, batch_size=16, compile=None)

    def make():
        torch.manual_seed(0)
        wm = WorldModel(WorldModelCfg(hidden=(32, 32), terrain=mode), heads, dev, terrain=tin)
        return WorldModelTrainer(wm, lib, cfg, terrain=tin)

    a, b = make(), make()
    torch.manual_seed(7)
    ma = a.update(buf)
    torch.manual_seed(7)
    mask = buf.window_mask(cfg.window)
    batches = [buf.sample_windows(cfg.batch_size, cfg.window, mask=mask) for _ in range(cfg.batches_per_update)]
    mb = b.update_on(batches)
    assert ma.keys() == mb.keys() and all(ma[k] == mb[k] for k in ma)
    for pa, pb in zip(a.wm.state_dict().values(), b.wm.state_dict().values()):
        assert torch.equal(pa, pb)
    mt = b.update_on(batches, sync=False)
    assert all(isinstance(v, torch.Tensor) for v in mt.values())
    w = batches[0]
    rp, rq = b.reference_roots(w)
    err, logits = b.wm.window_errors(w.states, w.a_exec, w.xi, rp, rq, origin=w.origin, grid=tin.grid)
    assert err.shape == logits.shape == (cfg.window, cfg.batch_size)
    torch.testing.assert_close(err.mean(1), b.wm.rollout_error(w.states, w.a_exec, w.xi, origin=w.origin, grid=tin.grid))
    _, m = b.wm.window_loss(w.states, w.a_exec, w.xi, w.done, rp, rq, origin=w.origin, grid=tin.grid)
    torch.testing.assert_close(m["nse_kstep"], err[-1].mean())
    torch.testing.assert_close(m["term_acc"], ((logits > 0).float() == w.done).float().mean())
