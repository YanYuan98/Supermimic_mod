"""Terrain through the training stack (docs/VISION_TERRAIN_PLAN.md M0):

(a) a terrain-capable model warm-started from a flat checkpoint gives the flat model's single-step outputs on
    ``plane`` (policy action, world-model step, critic value; fp32, <= 1e-5);
(c) replay rows written across resets carry their episode's o_ep (the terminal row keeps the old one);
(d) ``terrain.kind="none"`` builds main's environment config / observation layout byte for byte;
plus a short ``mixed`` training run warm-started with ``init_from`` (curriculum, checkpoint, resume).
The reward / termination parity on ``mixed`` (b) lives in test_analytic_heads.py.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
FLAT_CKPT = ROOT / "logs/g1-selfcol-2gpu/ckpt_latest.pt"
DEV = "cuda:0"
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not NPZ.exists(), reason="LAFAN1 npz not built"),
]


# ---------------------------------------------------------------------------------------------- (d) none == main

# make_env_cfg digests computed on main@6932f1c (scripts in the M0 report) for the argument sets below.
MAIN_LAYOUT_DIGESTS = {
    "{}": "d4da4900b3ff92225ff17083e1a446e3f38ca4a5",
    '{"anchor_obs": "pos_ori", "goal_features": "traj", "self_collision": false}': "794819f7761c302faf39bd03aa22acb5effcc1a3",
    '{"obs_noise": false, "reset_noise": false, "domain_randomization": false, "record_transitions": false, "sampling_mode": "start", "episode_length_s": 100000.0}': "ae99ab66016ee19d62ec5315dbf8ca88ce9590b4",
}
# Command-config fields added for terrain (inert defaults with terrain="none"); excluded from the digest.
NEW_COMMAND_FIELDS = {"terrain", "terrain_level", "terrain_map", "scan_noise", "spawn_margin", "spawn_horizon", "max_step_reward", "terrain_stats_min_steps"}


def _norm(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        out = {"__type__": f"{type(x).__module__}.{type(x).__qualname__}"}
        for f in dataclasses.fields(x):
            if type(x).__name__ == "MultiClipMotionCommandCfg" and f.name in NEW_COMMAND_FIELDS:
                continue
            v = getattr(x, f.name)
            if f.name == "npz_files":
                v = [Path(p).name for p in v]
            out[f.name] = _norm(v)
        return out
    if isinstance(x, dict):
        return {str(k): _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, type) or callable(x):
        return f"{getattr(x, '__module__', '?')}.{getattr(x, '__qualname__', repr(x))}"
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return repr(x)


def layout_digest(env_cfg) -> str:
    return hashlib.sha1(json.dumps(_norm(env_cfg), sort_keys=True).encode()).hexdigest()


def test_kind_none_layout_matches_main():
    from supermimic.config import TrainConfig, terrain_env_kwargs
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg

    cfg = TrainConfig()
    assert cfg.terrain.kind == "none" and terrain_env_kwargs(cfg.terrain) == {}
    npz = ["data/npz/lafan1/walk1_subject1.npz"]
    for kw, digest in MAIN_LAYOUT_DIGESTS.items():
        env_cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=npz, **json.loads(kw)))
        assert layout_digest(env_cfg) == digest, kw
        assert set(env_cfg.observations) == {"policy", "goal"}
        assert env_cfg.commands["motion"].terrain == "none"
    # ... and the networks of a kind="none" run have exactly main's parameters: the main checkpoint loads strictly.
    if FLAT_CKPT.exists():
        from supermimic.eval.benchmark import load_policy_from_checkpoint

        policy, _ = load_policy_from_checkpoint(FLAT_CKPT, "cpu")
        assert policy.terrain is None and not any(k.startswith("terrain") for k in policy.state_dict())


def test_eval_scan_noise_scale():
    """Benchmark robustness option (plan M5-A): the noise-free benchmark env gets the training scan noise x scale."""
    from supermimic.config import TrainConfig
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.eval.benchmark import terrain_env_kwargs_for_eval
    from supermimic.terrain import ScanNoiseCfg

    cfg = dataclasses.asdict(TrainConfig())
    base = dict(npz_files=["data/npz/lafan1/walk1_subject1.npz"], obs_noise=False)
    off = make_env_cfg(SuperMimicEnvCfgArgs(**base, **terrain_env_kwargs_for_eval(cfg, "plane"))).commands["motion"].scan_noise
    assert not off.enabled
    n = make_env_cfg(SuperMimicEnvCfgArgs(**base, **terrain_env_kwargs_for_eval(cfg, "plane", scan_noise_scale=2.0))).commands["motion"].scan_noise
    d = ScanNoiseCfg()
    assert n.enabled and (n.point, n.offset_z, n.offset_xy, n.dropout, n.max_lag) == (2 * d.point, 2 * d.offset_z, 2 * d.offset_xy, 2 * d.dropout, 2 * d.max_lag)


# ---------------------------------------------------------------------------------------------- (a) warm start parity


def _flat_models(device):
    """Flat (policy, critic, world model, config) from the main checkpoint, else small random ones."""
    from supermimic.config import TrainConfig
    from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
    from supermimic.world_model.world_model import WorldModelCfg

    if FLAT_CKPT.exists():
        ck = torch.load(FLAT_CKPT, map_location=device, weights_only=False)
        c = ck["config"]
        tup = lambda d: {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}  # noqa: E731
        pcfg = PolicyCfg(**tup(c["policy"]))
        ccfg = CriticCfg(**tup(c["critic"]))
        wcfg_d = dict(c["world_model"])
        from supermimic.world_model.state_spec import StateSpecCfg

        wcfg_d["state_spec"] = StateSpecCfg(**tup(wcfg_d["state_spec"]))
        wcfg = WorldModelCfg(**tup(wcfg_d))
        return ck, pcfg, ccfg, wcfg, c["env"]
    cfg = TrainConfig()
    pcfg = dataclasses.replace(cfg.policy, encoder_hidden=(128, 64), decoder_hidden=(128, 64), goal_dim=776, goal_frame_dim=77)
    ccfg = cfg.critic
    wcfg = dataclasses.replace(cfg.world_model, hidden=(128, 128))
    torch.manual_seed(0)
    pol, cri = Policy(pcfg).to(device), Critic(ccfg, pcfg=pcfg).to(device)
    return {"policy": pol.state_dict(), "critic": cri.state_dict(), "wm": None}, pcfg, ccfg, wcfg, {"anchor_obs": "ori", "goal_features": "traj", "self_collision": True}


@pytest.mark.parametrize("wm_terrain", ["foot", "both"])
def test_warm_start_plane_parity(wm_terrain):
    """M0 (a): flat checkpoint vs the same weights warm-started into terrain-capable networks, on ``plane``."""
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.policy.nets import Critic, Policy
    from supermimic.terrain import CRITIC_TERRAIN_DIM, SCAN_DIM, TerrainInputs
    from supermimic.training.warm_start import warm_start
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.world_model import WorldModel

    ck, pcfg, ccfg, wcfg, envc = _flat_models(DEV)
    files = discover_npz(NPZ, names=["walk1_subject1", "dance1_subject1", "jumps1_subject1"])
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=64, replay_capacity=64, seed=7, anchor_obs=envc["anchor_obs"], goal_features=envc["goal_features"], self_collision=envc["self_collision"], terrain="plane"), DEV)
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 comparison (make_env turns TF32 on)
    try:
        robot = env.scene["robot"]
        fk = G1Kinematics(load_g1_model(), device=DEV)
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        cmd = env.command_manager.get_term("motion")
        tin = TerrainInputs(cmd.tmap, fk, wm_terrain)
        flat_pol, flat_cri = Policy(pcfg).to(DEV), Critic(ccfg, pcfg=pcfg).to(DEV)
        flat_wm = WorldModel(wcfg, heads, DEV)
        if ck["wm"] is None:  # synthetic: random last layers so the outputs are not trivial
            torch.manual_seed(1)
            for p in flat_wm.parameters():
                p.data.normal_(0, 0.05)
            ck["wm"] = flat_wm.state_dict()
        flat_pol.load_state_dict(ck["policy"])
        flat_cri.load_state_dict(ck["critic"])
        flat_wm.load_state_dict(ck["wm"])
        t_pol = Policy(dataclasses.replace(pcfg, terrain_dim=SCAN_DIM)).to(DEV)
        t_cri = Critic(dataclasses.replace(ccfg, terrain_dim=CRITIC_TERRAIN_DIM), pcfg=pcfg).to(DEV)
        t_wm = WorldModel(dataclasses.replace(wcfg, terrain=wm_terrain), heads, DEV, terrain=tin)
        fresh = warm_start(t_pol, t_cri, t_wm, ck)
        assert all(k.startswith("terrain.") for k in fresh["policy"]) and fresh["policy"] and not fresh["critic"]
        assert set(fresh["wm"]) == {"terr_norm.mean", "terr_norm.var", "terr_norm.count"}

        # Real plane-env inputs (flat policy acting), with non-trivial new normalisers.
        obs, _ = env.reset()
        with torch.no_grad():
            for _ in range(30):
                t_pol.terrain.norm.update(obs["terrain"])
                obs, *_ = env.step(flat_pol(obs["policy"], obs["goal"]))
            a_flat = flat_pol(obs["policy"], obs["goal"])
            a_terr = t_pol(obs["policy"], obs["goal"], scan=obs["terrain"])
        assert t_pol.terrain.norm.count > 0 and obs["terrain"].shape == (64, SCAN_DIM)
        d_pol = (a_flat - a_terr).abs().max().item()

        buf = env.recorder_manager.get_term("replay").buffer
        # Training-size batch: cuBLAS then runs the same kernel for K = 133 and 133 + c, so the zero columns add
        # exactly nothing (small batches can pick K-dependent kernels: ~1e-5 fp32 reassociation noise).
        torch.manual_seed(0)
        w = buf.sample_windows(2048, 1)
        t_wm.update_normalizers(w.states[0], w.a_exec[0], w.xi, w.states[1], origin=w.origin, grid=tin.grid)
        assert t_wm.terr_norm.count > 0
        with torch.no_grad():
            s_flat = flat_wm.step(w.states[0], w.a_exec[0], w.xi)
            s_terr = t_wm.step_at(w.states[0], w.a_exec[0], w.xi, w.origin, tin.grid)
        d_wm = (s_flat - s_terr).abs().max().item()
        assert (s_flat - w.states[0]).abs().max() > 1e-3  # the step is not the identity

        st = buf.sample_starts(256, 10)
        feat = flat_wm.spec.canonical(st.state)
        terr, gz = tin.critic_features(st.state, st.origin)
        torch.testing.assert_close(t_wm.spec.canonical(st.state, ground_z=gz), feat, atol=0, rtol=0)  # plane: ground 0
        goal = torch.randn(256, pcfg.goal_dim, device=DEV)
        xi_n = flat_wm.spec.normalize_xi(st.xi)
        err = torch.randn(256, flat_cri.err_dim, device=DEV)
        with torch.no_grad():
            d_cri = (flat_cri(feat, goal, xi_n, err) - t_cri(feat, goal, xi_n, err, terr=terr)).abs().max().item()
        print(f"\nwarm-start parity ({'main ckpt' if FLAT_CKPT.exists() else 'synthetic'}, wm_terrain={wm_terrain}): policy {d_pol:.2e}, wm step {d_wm:.2e}, critic {d_cri:.2e}")
        assert d_pol <= 1e-5 and d_wm <= 1e-5 and d_cri <= 1e-5, (d_pol, d_wm, d_cri)
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------- (c) replay origins


@pytest.fixture(scope="module")
def mixed_env():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    files = discover_npz(NPZ, names=["walk1_subject1", "dance1_subject1", "jumps1_subject1"])
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(f) for f in files], num_envs=32, replay_capacity=256, seed=11,
        terrain="mixed", terrain_seed=3, terrain_rows=4, terrain_level=3, njmax=600, nconmax=70, ccd_iterations=100,
    )  # fmt: skip
    env = make_env(args, DEV)
    yield env
    env.close()


def test_replay_origin_round_trip(mixed_env):
    """Every replay row stores the o_ep of its episode (as written into env_origins at that episode's reset);
    the terminal row written at reset still has the finished episode's origin."""
    env = mixed_env
    cmd = env.command_manager.get_term("motion")
    buf = env.recorder_manager.get_term("replay").buffer
    env.reset()
    seen: dict[tuple[int, int], torch.Tensor] = {}

    def note():
        ep, o = buf.ep_counter.cpu(), env.scene.env_origins.cpu()
        for e in range(env.num_envs):
            seen.setdefault((e, int(ep[e])), o[e].clone())

    note()
    for _ in range(150):
        env.step(1.5 * torch.randn(env.num_envs, 29, device=DEV))  # falls -> frequent resets
        note()
    assert int(buf.ep_counter.min()) >= 2, "too few resets"
    valid = buf.valid.cpu()
    e_idx, t_idx = valid.nonzero(as_tuple=True)
    ep = buf.ep_id.cpu()[e_idx, t_idx]
    want = torch.stack([seen[(int(e), int(k))] for e, k in zip(e_idx, ep)])
    got = buf.origin.cpu()[e_idx, t_idx]
    assert torch.equal(got, want)
    # Terminal rows: same origin as the rest of their episode, different from the next episode's.
    term = buf.terminal.cpu()[e_idx, t_idx]
    assert term.sum() > 20
    nxt = torch.stack([seen.get((int(e), int(k) + 1), torch.full((3,), float("nan"))) for e, k in zip(e_idx[term], ep[term])])
    assert (got[term] != nxt).any(-1).float().mean() > 0.95
    assert (got[:, 2] == 0).all() and cmd.layout is not None
    # The origin makes the states world-consistent: pelvis height above the ground under it is plausible.
    s = buf.s.cpu()[e_idx, t_idx]
    o = got.to(DEV)
    hag = s[:, 2].to(DEV) - cmd.tmap.ground(s[:, 0:2].to(DEV) + o[:, 0:2])
    ok = ((hag > -0.1) & (hag < 1.3)).float().mean().item()
    assert ok > 0.99, ok
    # Windows / starts carry their episode's origin.
    w = buf.sample_windows(256, 8)
    st = buf.sample_starts(256, 10)
    assert w.origin.shape == (256, 3) and st.origin.shape == (256, 3)
    assert torch.isin(w.origin[:, 0], buf.origin[..., 0][buf.valid]).all()


def test_forced_tile_centres_clip(mixed_env):
    """Evaluation placement: a pinned (row, col) puts the reference path centred in that tile, no noise."""
    env = mixed_env
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    n = env.num_envs
    ids = torch.arange(n, device=DEV)
    cmd.forced_tile[:, 0] = ids % 4
    cmd.forced_tile[:, 1] = ids % cmd.layout.num_cols
    frames = lib.clip_start[ids % lib.num_clips]
    cmd.reset_to_frame(ids, frames)
    assert torch.equal(cmd.ep_row, ids % 4) and torch.equal(cmd.ep_col, ids % cmd.layout.num_cols)
    bb = cmd.path_bbox[frames]
    xy = lib.ref(frames).root_pos_w[:, 0:2] + env.scene.env_origins[:, 0:2]
    centre = cmd.layout.tile_centres(DEV)[cmd.ep_row, cmd.ep_col]
    torch.testing.assert_close(xy + 0.5 * (bb[:, 0:2] + bb[:, 2:4]), centre)
    cmd.forced_tile[:] = -1


# ---------------------------------------------------------------------------------------------- training smoke


def _tiny(cfg, name, tmp_path):
    cfg.run_name, cfg.log_root, cfg.seed = name, tmp_path, 3
    cfg.data.clips = ["walk1_subject1"]
    cfg.env.num_envs = 64
    cfg.env.replay_capacity = 200
    cfg.world_model.hidden = (256, 256)
    cfg.policy.encoder_hidden = (256, 128)
    cfg.policy.decoder_hidden = (256, 128)
    cfg.wm_trainer.batch_size = 256
    cfg.wm_trainer.batches_per_update = 2
    cfg.wm_trainer.compile = None
    cfg.policy_trainer.batch_size = 128
    cfg.policy_trainer.horizon = 8
    cfg.policy_trainer.compile = None
    cfg.schedule.warmup_env_steps = 20
    cfg.schedule.wm_warmup_iterations = 4
    cfg.schedule.log_interval = 4
    cfg.amp = "none"
    return cfg


def test_training_smoke_terrain(tmp_path):
    """Flat run -> ``init_from`` warm start on ``mixed`` (curriculum checks, terrain logging, checkpoint with the
    level) -> resume; plus a world-model-only run on ``plane``."""
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    flat = _tiny(TrainConfig(), "flat", tmp_path)
    flat.schedule.total_iterations = 8
    flat.schedule.checkpoint_interval = 8
    tr = Trainer(flat)
    tr.run()
    tr.env.close()

    cfg = _tiny(TrainConfig(), "mixed", tmp_path)
    cfg.init_from = tmp_path / "flat" / "ckpt_latest.pt"
    cfg.terrain.kind = "mixed"
    cfg.terrain.num_rows = 3
    cfg.terrain.wm_terrain = "both"
    cfg.terrain.curriculum.start_level = 0
    cfg.terrain.curriculum.check_every = 4
    cfg.terrain.curriculum.min_episodes = 0
    cfg.terrain.curriculum.promote_return = -1.0  # always promote: exercises the level update
    cfg.terrain.curriculum.promote_max_fail = 2.0
    cfg.schedule.total_iterations = 12
    cfg.schedule.checkpoint_interval = 12
    tr = Trainer(cfg)
    assert tr.policy.terrain is not None and tr.wm.terr_dim == 215 and tr.critic.terrain_dim == 215
    w = tr.policy.decoder.mlp.net[0].weight
    assert torch.equal(w[:, 32:64], torch.zeros_like(w[:, 32:64]))  # zero-initialised terrain-latent columns
    tr.run()
    assert tr.iteration == 12 and tr.command.terrain_level == 2  # promoted at 4 and 8, capped at num_rows - 1
    assert all(torch.isfinite(p).all() for m in (tr.policy, tr.critic, tr.wm) for p in m.parameters())
    ck = torch.load(tmp_path / "mixed" / "ckpt_latest.pt", map_location="cpu", weights_only=False)
    assert ck["terrain_level"] == 2 and ck["config"]["terrain"]["kind"] == "mixed"
    tr.env.close()
    cfg2 = TrainConfig(**{**cfg.__dict__, "resume": tmp_path / "mixed" / "ckpt_latest.pt", "run_name": "mixed2"})
    tr2 = Trainer(cfg2)
    assert tr2.iteration == 12 and tr2.command.terrain_level == 2
    tr2.env.close()

    wm_only = _tiny(TrainConfig(), "plane_wm", tmp_path)
    wm_only.terrain.kind = "plane"
    wm_only.schedule.wm_only = True
    wm_only.schedule.total_iterations = 6
    wm_only.schedule.checkpoint_interval = 6
    tr3 = Trainer(wm_only)
    p0 = [p.detach().clone() for p in tr3.policy.parameters()]
    tr3.run()
    assert tr3.num_updates == 0 and tr3.policy.terrain.norm.count == 0 and tr3.wm.terr_norm.count > 0
    assert all(torch.equal(a, b) for a, b in zip(p0, tr3.policy.parameters()))
    tr3.env.close()
