"""v5 ablation options (worktree branch v5-ablation): world-model loss weighting, critic inputs, vibed goal."""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
needs_npz = pytest.mark.skipif(not NPZ.exists(), reason="LAFAN1 npz not built")


@pytest.fixture(scope="module")
def cpu_lib():
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs.mdp.motion_library import MotionLibrary, discover_npz
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads

    files = discover_npz(NPZ, names=["walk1_subject1", "dance1_subject1"])
    q0 = torch.zeros(29)
    lib = MotionLibrary(files, TRACKED_BODIES, q0, device="cpu")
    model = load_g1_model()
    lim = torch.as_tensor(model.jnt_range[1:], dtype=torch.float32)
    heads = AnalyticHeads(G1Kinematics(model, device="cpu"), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, lim, q0)
    return lib, heads


def _ref_states(lib, n=64, seed=0):
    f = torch.randint(0, lib.num_frames - 20, (n,), generator=torch.Generator().manual_seed(seed))
    f = torch.minimum(f, lib.last_frame_of(f) - 10)
    qpos, qvel = lib.ref_qpos_qvel(f)
    return f, torch.cat([qpos, qvel], -1)


@needs_npz
def test_local_error_is_zero_on_itself_and_matches_increment(cpu_lib):
    from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

    lib, _ = cpu_lib
    spec = StateSpec(StateSpecCfg(), "cpu")
    f, s0 = _ref_states(lib)
    _, s1 = lib.ref_qpos_qvel(f + 1)
    s1 = torch.cat([_, s1], -1)
    assert spec.local_error(s0, s0).abs().max() < 1e-6
    inc = spec.local_error(s1, s0)
    # joint parts are plain differences; position / velocity parts preserve the norm (pure rotations)
    torch.testing.assert_close(inc[:, 12:41], s1[:, 7:36] - s0[:, 7:36])
    torch.testing.assert_close(inc[:, 0:3].norm(dim=-1), (s1[:, 0:3] - s0[:, 0:3]).norm(dim=-1), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(inc[:, 6:9].norm(dim=-1), (s1[:, 36:39] - s0[:, 36:39]).norm(dim=-1), atol=1e-6, rtol=1e-5)


@needs_npz
@pytest.mark.parametrize("weighting", ["nominal", "delta_std"])
def test_window_loss_weightings(cpu_lib, weighting):
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    lib, heads = cpu_lib
    torch.manual_seed(0)
    wm = WorldModel(WorldModelCfg(hidden=(64, 64), loss_weighting=weighting), heads, "cpu")
    k, n = 4, 32
    f, _ = _ref_states(lib, n)
    frames = f[None, :] + torch.arange(k + 1)[:, None]
    states = torch.stack([torch.cat(lib.ref_qpos_qvel(frames[j]), -1) for j in range(k + 1)])
    a = torch.zeros(k, n, 29)
    xi = torch.cat([torch.full((n, 1), 0.75), torch.zeros(n, 32)], -1)
    wm.update_normalizers(states[0], a[0], xi, states[1])
    r = lib.ref(frames[:k].reshape(-1))
    loss, m = wm.window_loss(states, a, xi, torch.zeros(k, n), r.root_pos_w.reshape(k, n, 3), r.root_quat_w.reshape(k, n, 4))
    assert torch.isfinite(loss)
    loss.backward()
    g = sum(p.grad.abs().sum() for p in wm.dynamics.parameters() if p.grad is not None)
    assert torch.isfinite(g) and g > 0
    if weighting == "delta_std":
        std = torch.maximum(wm.inc_norm.std(), wm.inc_floor)
        assert torch.all(std >= wm.inc_floor) and wm.inc_norm.count > 0


@needs_npz
def test_vibed_critic_features_vanish_on_reference(cpu_lib):
    from supermimic.world_model.wm_trainer import reference_at

    lib, heads = cpu_lib
    f, s = _ref_states(lib)
    ref = reference_at(lib, f)
    e = heads.critic_error_features_vibed(s, ref)
    assert e.shape == (64, 80)
    ident = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expand(64, 6)
    np.testing.assert_allclose(e[:, 74:80].numpy(), ident.numpy(), atol=1e-4)
    assert e[:, :74].abs().max() < 1e-3
    s2 = s.clone().requires_grad_(True)
    e2 = heads.critic_error_features_vibed(s2 + torch.cat([torch.tensor([0.1, 0.0, 0.0]), torch.zeros(68)]), ref)
    torch.testing.assert_close(e2[:, 71:74].norm(dim=-1).detach(), torch.full((64,), 0.1), atol=1e-4, rtol=0)
    e2.sum().backward()
    assert torch.isfinite(s2.grad).all() and s2.grad.abs().sum() > 0


def test_critic_options():
    from supermimic.policy.nets import Critic, CriticCfg, PolicyCfg

    pc = PolicyCfg(goal_frame_dim=77, goal_dim=776)
    for feats, dim in (("v4", 84), ("vibed", 80)):
        for norm in ("fixed", "running"):
            c = Critic(CriticCfg(features=feats, input_norm=norm, activation="elu"), pcfg=pc)
            assert c.err_dim == dim and (c.norm is not None) == (norm == "running")
            x = (torch.randn(8, 71), torch.randn(8, 776), torch.randn(8, 33), torch.randn(8, dim))
            c.update_norm(*x)
            assert c(*x).shape == (8,)
    with pytest.raises(ValueError):
        Critic(CriticCfg(features="nope"), pcfg=pc)


@needs_npz
def test_vibed_goal(cpu_lib):
    from supermimic.goal_spec import goal_dim
    from supermimic.policy.nets import PolicyCfg, _goal_scale_vector

    lib, _ = cpu_lib
    assert goal_dim(10, "vibed", "ori") == 736
    assert torch.all(_goal_scale_vector(PolicyCfg(goal_frame_dim=73), 736) == 1)
    f = torch.tensor([lib.clip_start[0].item(), 100, 2000, lib.clip_last[0].item() - 3])
    g = lib.goal_window(f, 10, "vibed").reshape(4, 10, 73)
    r = lib.ref(f)
    torch.testing.assert_close(g[:, 0, 15:44], r.joint_pos)
    torch.testing.assert_close(g[:, 0, 44:73], 0.1 * r.joint_vel)
    # At the clip start the current-state frame is the target frame itself: zero displacement.
    assert g[0, 0, 0:3].abs().max() < 1e-6
    # Elsewhere the first window frame is one step ahead of the current state (small displacement).
    assert (g[1:, 0, 0:3].norm(dim=-1) < 0.2).all()
    torch.testing.assert_close(g[3, 3:], g[3, 3:4].expand(7, 73))


def test_sampler_cfg_defaults():
    from supermimic.config import EnvCfg
    from supermimic.training.loop import Trainer

    class C:
        env = EnvCfg()

    assert Trainer._sampler_kwargs(C) == {"sampler_init": "gmax", "sampler_credit": "per_step"}
    C.env = EnvCfg(sampler_init="zero", sampler_credit="start_bin", sampler_ema_alpha=0.2, sampler_memory_alpha=0.001)
    assert Trainer._sampler_kwargs(C) == {"sampler_init": "zero", "sampler_credit": "start_bin", "ema_alpha": 0.2, "memory_alpha": 0.001}
