"""Record deterministic whole-clip rollouts of a checkpoint for videos (Appendix F settings).

Every env plays one clip from its first frame with the deterministic policy (no observation /
action noise, no reset noise, no domain randomisation, foot friction 1.0). Unlike the benchmark,
the root height / tilt terminations are disabled so the rollout keeps going after a failure;
the first step at which the training termination would have fired is recorded instead.

Output (npz, one ``s{i}_`` prefix per clip, sample k = state after k control steps, compared with
reference frame k): root_pos (origin-free), root_quat (wxyz), joint_pos, ref_root_pos,
ref_root_quat, ref_joint_pos (depth policies also ``depth``: the newest frame the policy sees, (T, H, W) uint8,
0 = hole, 1..255 = near..far clip), plus ``metadata_json`` {dt, segments: [{clip, start_frame,
end_frame, first_failure_frame}]}.

Terrain checkpoints run on the plane by default; ``--terrain mixed --terrain-col stairs_up --terrain-row 6``
centres every clip in that tile of the fixed-seed benchmark map; ``--stairs-platform 1.5`` swaps in the narrow-platform
stairs variant (evaluation / video only: 16 steps per side on a 12 m tile instead of 6, see ``presets.stairs_frame``), ``--slope-platform 1.5``
the slope variant (3 m in training). The reference (``ref_root_pos``) is then
terrain-lifted (the failure test uses it, as in training) and each clip also stores its episode origin
``s{i}_origin`` (world = origin + root_pos); ``metadata_json["terrain"]`` lets render_compare.py rebuild the map.

  uv run python scripts/record_rollouts.py --ckpt logs/<run>/ckpt_latest.pt \\
      --clips walk1_subject1 dance2_subject1 --out logs/<run>/rollouts.npz
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.state_io import StateIO
from supermimic.eval.benchmark import EVAL_TERRAIN_SEED, act, load_policy_from_checkpoint, terrain_env_kwargs_for_eval
from supermimic.goal_spec import anchor_mode
from supermimic.kinematics.rotations import projected_gravity


def main(
    ckpt: Path, clips: list[str], out: Path, npz_dir: Path = Path("data/npz/lafan1"), device: str = "cuda:0", height_threshold: float = 0.25, tilt_threshold: float = 0.8,
    terrain: Literal["none", "plane", "mixed"] | None = None, terrain_seed: int = EVAL_TERRAIN_SEED, terrain_col: str = "flat", terrain_row: int = 0,
    stairs_platform: float | None = None, slope_platform: float | None = None,
) -> None:  # fmt: skip
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    from supermimic.terrain import reference_lift

    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    if terrain is None:
        terrain = "plane" if (policy.terrain is not None or policy.depth is not None) else "none"
    tkw = terrain_env_kwargs_for_eval(cfg, terrain, terrain_seed)
    files = discover_npz(npz_dir, names=clips)
    if sorted(f.stem for f in files) != sorted(clips):
        raise FileNotFoundError(f"clips not found under {npz_dir}: {sorted(set(clips) - {f.stem for f in files})}")
    env_args = SuperMimicEnvCfgArgs(
        npz_files=[str(f) for f in files], num_envs=len(files), future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
        sampling_mode="start", record_transitions=False, seed=0, anchor_obs=anchor_mode(cfg["env"].get("anchor_obs", False)),
        goal_features=cfg["env"].get("goal_features", "v1"), self_collision=cfg["env"].get("self_collision", True), **tkw,
    )  # fmt: skip
    env_cfg = make_env_cfg(env_args)
    if stairs_platform is not None or slope_platform is not None:  # evaluation-only variants (the policy never trained on them)
        from supermimic.terrain import make_terrain_generator_cfg

        assert terrain == "mixed", "--stairs-platform / --slope-platform need --terrain mixed"
        env_cfg.scene.terrain.terrain_generator = make_terrain_generator_cfg(
            "mixed", seed=tkw["terrain_seed"], size=tkw["terrain_size"], num_rows=tkw["terrain_rows"], border=tkw["terrain_border"],
            stairs_platform=stairs_platform, slope_platform=slope_platform,
        )  # fmt: skip
    for name in ("root_height", "root_tilt"):  # keep running after a failure; it is detected below
        env_cfg.terminations.pop(name)
    configure_torch_backends()
    if terrain == "mixed" and os.environ.get("SUPERMIMIC_HFIELD_PATCH", "1") != "0":
        from supermimic.terrain.mjwarp_patch import install_hfield_normal_filter

        install_hfield_normal_filter()  # height-field contact-normal fix, as make_env (training / benchmark) installs it
    env = ManagerBasedRlEnv(env_cfg, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        io = StateIO(env)
        n = lib.num_clips
        env_ids = torch.arange(n, device=device)
        cmd.forced_frame[:] = lib.clip_start[env_ids]  # env i plays library clip i from frame 0
        if terrain == "mixed":
            cmd.forced_tile[:, 0] = terrain_row
            cmd.forced_tile[:, 1] = cmd.layout.col_index(terrain_col)
        lengths = lib.clip_len.cpu().numpy()
        T = int(lengths.max())
        rec = {k: [] for k in ("root_pos", "root_quat", "joint_pos", "ref_root_pos", "ref_root_quat", "ref_joint_pos")}
        if getattr(cmd, "depth_sensor", None) is not None:
            rec["depth"] = []
        failed_at = np.full(n, -1)
        with torch.inference_mode():
            obs, _ = env.reset()
            origin = io.origins.clone()  # episode origins (the env resets again after a short clip ends)
            for k in range(T):
                if k > 0:
                    obs, *_ = env.step(act(policy, obs, env.command_manager.get_term("motion")))
                s = io.read() if terrain != "mixed" else _read_local(env, origin)
                fr = lib.clip_start + torch.clamp(torch.tensor(k, device=device), max=lib.clip_len - 1)
                ref = lib.ref(fr)
                if terrain == "mixed":  # terrain-lifted reference, as the training termination sees it
                    ref.root_pos_w = ref.root_pos_w.clone()
                    if tkw.get("reference_lift") == "robot":  # lifted by the ground under the robot pelvis
                        ref.root_pos_w[:, 2] += cmd.tmap.ground(s[:, 0:2] + origin[:, 0:2])
                    else:
                        ref.root_pos_w[:, 2] += reference_lift(lib, fr, origin, cmd.tmap)
                rec["root_pos"].append(s[:, 0:3])
                rec["root_quat"].append(s[:, 3:7])
                rec["joint_pos"].append(s[:, 7:36])
                rec["ref_root_pos"].append(ref.root_pos_w)
                rec["ref_root_quat"].append(ref.root_quat_w)
                rec["ref_joint_pos"].append(ref.joint_pos)
                if "depth" in rec:  # newest frame of the policy's history: normalised depth (-1 near .. 1 far) + validity
                    img = cmd.depth_img[-1].float()
                    rec["depth"].append(torch.where(img[:, 1] > 0.5, (img[:, 0].clamp(-1, 1) + 1) * 127 + 1, torch.zeros_like(img[:, 0])).to(torch.uint8))
                # Training termination (paper App. A): pelvis height / tilt vs the reference root.
                dz = ref.root_pos_w[:, 2] - s[:, 2]
                if terrain == "mixed" and tkw.get("height_termination") == "ground":  # same test as the env's termination
                    dz = dz + cmd.tmap.ground(s[:, 0:2] + origin[:, 0:2]) - cmd.tmap.ground(ref.root_pos_w[:, 0:2] + origin[:, 0:2])
                dz = dz.abs()
                tilt = (projected_gravity(ref.root_quat_w)[:, 2] - projected_gravity(s[:, 3:7])[:, 2]).abs()
                bad = ((dz > height_threshold) | (tilt > tilt_threshold)).cpu().numpy() & (failed_at < 0) & (k < lengths)
                failed_at[bad] = k
        data = {key: torch.stack(v, 1).cpu().numpy() for key, v in rec.items()}  # (env, T, ·)
        origins = origin.cpu().numpy()
    finally:
        env.close()
    order = [next(i for i, c in enumerate(lib.clips) if c.name == name) for name in clips]
    arrays, segments = {}, []
    for j, i in enumerate(order):
        L = int(lengths[i])
        for key, v in data.items():
            arrays[f"s{j}_{key}"] = v[i, :L]
        if terrain == "mixed":
            arrays[f"s{j}_origin"] = origins[i]
        segments.append({"clip": clips[j], "start_frame": 0, "end_frame": L, "first_failure_frame": None if failed_at[i] < 0 else int(failed_at[i])})
        print(f"{clips[j]:28s} {L} frames  first failure: {segments[-1]['first_failure_frame']}")
    meta = {"dt": 0.02, "checkpoint": str(ckpt), "protocol": "deterministic, no noise / DR, friction 1.0, from frame 0, terminations off (first failure recorded)", "segments": segments}
    if terrain == "mixed":
        meta["terrain"] = {k: tkw[k] for k in ("terrain", "terrain_seed", "terrain_size", "terrain_rows", "terrain_border")} | {"col": terrain_col, "row": terrain_row}
        if stairs_platform is not None:
            meta["terrain"]["stairs_platform"] = stairs_platform
        if slope_platform is not None:
            meta["terrain"]["slope_platform"] = slope_platform
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, metadata_json=json.dumps(meta), **arrays)
    print("saved", out)


def _read_local(env, origin: torch.Tensor) -> torch.Tensor:
    """(N, 71) state relative to the given (first-episode) origins; StateIO would use the live ones, which
    change when an env resets after its (shorter) clip ended."""
    d = env.sim.data
    s = torch.cat([d.qpos[:], d.qvel[:]], dim=-1).clone()
    s[:, 0:3] -= origin
    return s


if __name__ == "__main__":
    tyro.cli(main)
