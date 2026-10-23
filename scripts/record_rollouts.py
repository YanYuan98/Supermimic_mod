"""Record deterministic whole-clip rollouts of a checkpoint for videos (Appendix F settings).

Every env plays one clip from its first frame with the deterministic policy (no observation /
action noise, no reset noise, no domain randomisation, foot friction 1.0). Unlike the benchmark,
the root height / tilt terminations are disabled so the rollout keeps going after a failure;
the first step at which the training termination would have fired is recorded instead.

Output (npz, one ``s{i}_`` prefix per clip, sample k = state after k control steps, compared with
reference frame k): root_pos (origin-free), root_quat (wxyz), joint_pos, ref_root_pos,
ref_root_quat, ref_joint_pos, plus ``metadata_json`` {dt, segments: [{clip, start_frame,
end_frame, first_failure_frame}]}.

  uv run python scripts/record_rollouts.py --ckpt logs/<run>/ckpt_latest.pt \\
      --clips walk1_subject1 dance2_subject1 --out logs/<run>/rollouts.npz
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.state_io import StateIO
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.goal_spec import anchor_mode
from supermimic.kinematics.rotations import projected_gravity


def main(ckpt: Path, clips: list[str], out: Path, npz_dir: Path = Path("data/npz/lafan1"), device: str = "cuda:0", height_threshold: float = 0.25, tilt_threshold: float = 0.8) -> None:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    files = discover_npz(npz_dir, names=clips)
    if sorted(f.stem for f in files) != sorted(clips):
        raise FileNotFoundError(f"clips not found under {npz_dir}: {sorted(set(clips) - {f.stem for f in files})}")
    env_args = SuperMimicEnvCfgArgs(
        npz_files=[str(f) for f in files], num_envs=len(files), future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
        sampling_mode="start", record_transitions=False, seed=0, anchor_obs=anchor_mode(cfg["env"].get("anchor_obs", False)),
        goal_features=cfg["env"].get("goal_features", "v1"), self_collision=cfg["env"].get("self_collision", True),
    )  # fmt: skip
    env_cfg = make_env_cfg(env_args)
    for name in ("root_height", "root_tilt"):  # keep running after a failure; it is detected below
        env_cfg.terminations.pop(name)
    configure_torch_backends()
    env = ManagerBasedRlEnv(env_cfg, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        io = StateIO(env)
        n = lib.num_clips
        env_ids = torch.arange(n, device=device)
        cmd.forced_frame[:] = lib.clip_start[env_ids]  # env i plays library clip i from frame 0
        lengths = lib.clip_len.cpu().numpy()
        T = int(lengths.max())
        rec = {k: [] for k in ("root_pos", "root_quat", "joint_pos", "ref_root_pos", "ref_root_quat", "ref_joint_pos")}
        failed_at = np.full(n, -1)
        with torch.inference_mode():
            obs, _ = env.reset()
            for k in range(T):
                if k > 0:
                    obs, *_ = env.step(policy(obs["policy"], obs["goal"]))
                s = io.read()
                ref = lib.ref(lib.clip_start + torch.clamp(torch.tensor(k, device=device), max=lib.clip_len - 1))
                rec["root_pos"].append(s[:, 0:3])
                rec["root_quat"].append(s[:, 3:7])
                rec["joint_pos"].append(s[:, 7:36])
                rec["ref_root_pos"].append(ref.root_pos_w)
                rec["ref_root_quat"].append(ref.root_quat_w)
                rec["ref_joint_pos"].append(ref.joint_pos)
                # Training termination (paper App. A): pelvis height / tilt vs the reference root.
                dz = (ref.root_pos_w[:, 2] - s[:, 2]).abs()
                tilt = (projected_gravity(ref.root_quat_w)[:, 2] - projected_gravity(s[:, 3:7])[:, 2]).abs()
                bad = ((dz > height_threshold) | (tilt > tilt_threshold)).cpu().numpy() & (failed_at < 0) & (k < lengths)
                failed_at[bad] = k
        data = {key: torch.stack(v, 1).cpu().numpy() for key, v in rec.items()}  # (env, T, ·)
    finally:
        env.close()
    order = [next(i for i, c in enumerate(lib.clips) if c.name == name) for name in clips]
    arrays, segments = {}, []
    for j, i in enumerate(order):
        L = int(lengths[i])
        for key, v in data.items():
            arrays[f"s{j}_{key}"] = v[i, :L]
        segments.append({"clip": clips[j], "start_frame": 0, "end_frame": L, "first_failure_frame": None if failed_at[i] < 0 else int(failed_at[i])})
        print(f"{clips[j]:28s} {L} frames  first failure: {segments[-1]['first_failure_frame']}")
    meta = {"dt": 0.02, "checkpoint": str(ckpt), "protocol": "deterministic, no noise / DR, friction 1.0, from frame 0, terminations off (first failure recorded)", "segments": segments}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, metadata_json=json.dumps(meta), **arrays)
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
