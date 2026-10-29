"""Watch a trained tracking policy replay reference clips in an interactive viewer.

  uv run python scripts/play.py --ckpt logs/e5_bones_seed/ckpt_latest.pt --clips dance1_subject1
  uv run python scripts/play.py --ckpt logs/e3_lafan1_40_anchor/ckpt_latest.pt --npz-dir data/npz/lafan1 --num-envs 4

The grey/blue translucent robot is the reference pose; the solid robot is the policy.
Deterministic evaluation settings (no noise, no DR); every reset starts the clip from its first frame.
Native MuJoCo viewer when a display is available, otherwise viser (browser, http://localhost:8080).
Keys (native): space pause, right-arrow single step, +/- speed, backspace reset.
Terrain checkpoints play on the plane by default; ``--terrain mixed --terrain-col stairs_up --terrain-row 6``
centres every clip in that tile of the (fixed-seed) benchmark terrain.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from supermimic.goal_spec import anchor_mode
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import EVAL_TERRAIN_SEED, act, load_policy_from_checkpoint, terrain_env_kwargs_for_eval


class _DictPolicy:
    """Adapts policy(obs_policy, obs_goal[, scan]) to the viewer's policy(obs) protocol."""

    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def __call__(self, obs):
        return act(self.policy, obs)


def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: list[str] | None = None,
    num_envs: int = 1,
    device: str = "cuda:0",
    viewer: Literal["auto", "native", "viser"] = "auto",
    ghost: bool = True,
    sampling: Literal["start", "uniform"] = "start",
    terrain: Literal["none", "plane", "mixed"] | None = None,
    terrain_seed: int = EVAL_TERRAIN_SEED,
    terrain_col: str | None = None,
    terrain_row: int | None = None,
) -> None:
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    if terrain is None:
        terrain = "plane" if policy.terrain is not None else "none"
    args = SuperMimicEnvCfgArgs(
        npz_files=files,
        num_envs=num_envs,
        future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode=sampling,
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_mode(cfg["env"].get("anchor_obs", False)), goal_features=cfg["env"].get("goal_features", "v1"), self_collision=cfg["env"].get("self_collision", True),
        debug_vis=ghost,
        **terrain_env_kwargs_for_eval(cfg, terrain, terrain_seed),
    )
    raw = make_env(args, device=device)
    cmd = raw.command_manager.get_term("motion")
    if terrain == "mixed" and (terrain_col is not None or terrain_row is not None):
        cmd.forced_tile[:, 0] = terrain_row if terrain_row is not None else -1
        cmd.forced_tile[:, 1] = cmd.layout.col_index(terrain_col) if terrain_col is not None else -1
    env = RslRlVecEnvWrapper(raw)
    pol = _DictPolicy(policy)
    if viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    print(f"[INFO] {len(files)} clip(s), {num_envs} env(s), viewer={viewer}")
    try:
        if viewer == "native":
            NativeMujocoViewer(env, pol).run()
        else:
            ViserPlayViewer(env, pol).run()
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
