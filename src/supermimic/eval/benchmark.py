"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires before the clip ends.

Normalised return (paper §4): the undiscounted reward sum over the episode divided by the
largest value it can take, i.e. max step reward * the number of steps of the *complete* clip,
so a policy that terminates early scores low. ``norm_return_survived`` keeps the v1 definition
(divided by the steps actually taken) for comparison with the old experiment log only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.goal_spec import anchor_mode
from supermimic.policy.nets import Policy, PolicyCfg


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    norm_return_survived: float
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = dict(ck["config"]["policy"])
    pc.setdefault("input_norm", "fixed")  # configs saved before these options existed (v1 / early v2)
    pc.setdefault("activation", "silu")
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    cfg = dict(ck["config"])
    cfg["_meta"] = {k: ck.get(k) for k in ("iteration", "iteration_unit", "env_steps", "num_updates", "train_seconds")}
    return policy, cfg


@torch.no_grad()
def run_benchmark(policy: Policy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int = 10, verbose: bool = True, anchor_obs: str | bool = "none", goal_features: str = "v1", self_collision: bool = True) -> list[ClipResult]:
    num_envs = min(num_envs or 1024, len(npz_files))
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_mode(anchor_obs),
        goal_features=goal_features,
        self_collision=self_collision,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
        max_r = 0.1  # paper reward upper bound per step (dt * 5)

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        results: dict[int, ClipResult] = {}
        remaining = n_clips
        while remaining > 0:
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            steps += 1
            done = (term | tout) & active
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        # A complete clip of L frames takes L env steps here (the reset already
                        # advances the target to frame 1; the clip-end time-out fires one step after
                        # the last frame is reached, which is scored against the last frame again).
                        full = lib.clips[ci].length
                        if not bool(term[e]) and st != full:
                            print(f"[WARN] {lib.clips[ci].name}: completed in {st} steps, expected {full}")
                        results[ci] = ClipResult(
                            name=lib.clips[ci].name,
                            frames=full,
                            steps=st,
                            norm_return=float(ret_sum[e]) / (max_r * max(full, st, 1)),
                            norm_return_survived=float(ret_sum[e]) / (max_r * max(st, 1)),
                            failed=bool(term[e]),
                            fail_step=st if bool(term[e]) else -1,
                            terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                steps[ids] = 0
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Length-weighted return: total reward over total attainable reward of the whole benchmark.
    tot_frames = sum(r.frames for r in results)
    w_ret = sum(r.norm_return * r.frames for r in results) / max(tot_frames, 1)
    old_mean = sum(r.norm_return_survived for r in results) / max(n, 1)
    tot_steps = sum(r.steps for r in results)
    out = {"clips": n, "failed": failed, "mean_norm_return": mean_ret, "step_weighted_norm_return": w_ret, "mean_norm_return_survived_v1": old_mean}
    if results and results[0].terms:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    return out


def save_results(results: list[ClipResult], path: str | Path, meta: dict | None = None) -> None:
    Path(path).write_text(json.dumps({"checkpoint": meta or {}, "summary": summarize(results), "clips": [asdict(r) for r in results]}, indent=1))
